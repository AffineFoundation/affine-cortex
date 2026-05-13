"""
``af servers executor`` — manager that spawns one subprocess per env.

The manager process does very little hot-path work: it spawns N
``WorkerProcess`` instances (one per sampling-enabled env), restarts them if
they die, prints a periodic ``[STATUS]`` line for operators, and owns
the cross-process IPC primitives (``BoundedSemaphore`` for global
dispatch budget; one ``Value(c_int)`` per env for live in-flight
read-back).

Each subprocess has its own asyncio loop + SDKEnvironment (Docker +
SSH tunnel + paramiko Transport), so paramiko thread budget and
connection pools don't have to be shared.
"""

from __future__ import annotations

import asyncio
import math
import multiprocessing
import signal
from typing import Any, Dict, List

import click

from affine.core.setup import logger, setup_logging
from affine.database import close_client, init_client
from affine.database.dao.sample_results import SampleResultsDAO
from affine.database.dao.system_config import SystemConfigDAO
from affine.src.executor.config import GLOBAL_DISPATCH_BUDGET, get_max_concurrent
from affine.src.executor.worker_process import WorkerProcess
from affine.src.scorer.dao_adapters import SampleResultsAdapter


HEALTH_CHECK_INTERVAL_SEC = 10
STATUS_PRINT_INTERVAL_SEC = 60


class ExecutorManager:
    def __init__(
        self,
        envs: List[str],
        *,
        verbosity: int = 1,
    ):
        self.envs = envs
        self.verbosity = verbosity
        self.mp_ctx = multiprocessing.get_context("spawn")
        # Single cross-process bounded semaphore = the real concurrency
        # gate. Sized to the b300 saturation point so the inference
        # backend is the bottleneck (where we want it). Workers first pass
        # their dynamic per-env cap, then contend on this shared sem before
        # every evaluate. The manager adjusts per-env caps from live
        # progress instead of hardcoding env-specific limits.
        self.global_sem = self.mp_ctx.BoundedSemaphore(GLOBAL_DISPATCH_BUDGET)
        # Per-env in-flight counter; worker writes (single writer), manager
        # reads. ``lock=False`` is safe: only one process writes each
        # Value, and aligned c_int reads are atomic on CPython.
        self.in_flight_values: Dict[str, Any] = {
            env: self.mp_ctx.Value("i", 0, lock=False) for env in envs
        }
        initial_cap = _initial_env_cap(envs, GLOBAL_DISPATCH_BUDGET)
        self.env_cap_values: Dict[str, Any] = {
            env: self.mp_ctx.Value("i", initial_cap, lock=False) for env in envs
        }
        self.workers: List[WorkerProcess] = []
        # Per-env running totals from the prior status print, used to
        # compute "finished in last interval" + rate/h.
        self._last_status_at: float = 0.0
        self._last_done: Dict[str, int] = {env: 0 for env in envs}
        self.running = False
        logger.info(
            f"ExecutorManager init: envs={envs}, "
            f"global_dispatch_budget={GLOBAL_DISPATCH_BUDGET}"
        )

    async def start(self) -> None:
        if self.running:
            return
        for idx, env in enumerate(self.envs):
            wp = WorkerProcess(
                worker_id=idx, env=env,
                global_sem=self.global_sem,
                in_flight_value=self.in_flight_values[env],
                env_cap_value=self.env_cap_values[env],
                max_concurrent=get_max_concurrent(env),
                verbosity=self.verbosity,
            )
            wp.start()
            self.workers.append(wp)
        self.running = True
        asyncio.create_task(self._health_checker())
        asyncio.create_task(self._status_printer())
        logger.info(f"Spawned {len(self.workers)} worker subprocesses")

    async def stop(self) -> None:
        self.running = False
        for wp in self.workers:
            try:
                wp.terminate()
            except Exception as e:
                logger.warning(f"terminate({wp.env}) raised: {e}")
        self.workers.clear()

    async def _health_checker(self) -> None:
        while self.running:
            await asyncio.sleep(HEALTH_CHECK_INTERVAL_SEC)
            for wp in self.workers:
                if not wp.is_alive():
                    logger.warning(f"[{wp.env}] subprocess died; restarting")
                    try:
                        self._recover_dead_worker_slots(wp.env)
                        wp.start()
                    except Exception as e:
                        logger.error(f"[{wp.env}] restart raised: {e}")

    def _recover_dead_worker_slots(self, env: str) -> None:
        """Clear dispatch accounting left behind by a dead worker process."""
        value = self.in_flight_values.get(env)
        if value is None:
            return
        stale = max(0, int(value.value))
        if stale:
            logger.warning(
                f"[{env}] recovering {stale} stale dispatch slot(s) "
                "after subprocess death"
            )
        for _ in range(stale):
            try:
                self.global_sem.release()
            except ValueError:
                break
        value.value = 0

    async def _status_printer(self) -> None:
        """Periodic ``[STATUS]`` line in the legacy executor's format.

        Reads done counts from ``sample_results`` for the champion and,
        during battles, the challenger. Per-env in-flight comes from the
        shared ``in_flight_values`` Values. Format:

            [STATUS] running=N/GLOBAL_DISPATCH_BUDGET | env@done/target +delta in Xs rate:Y/h
                run:R | env2@... | total +D in Xs rate:Y/h

        First print after start is the baseline (delta+rate omitted).
        """
        sc = SystemConfigDAO()
        samples = SampleResultsAdapter(
            dao=SampleResultsDAO(), validator_hotkey="executor-manager",
        )
        # Settle period — let warmup complete + first samples land before
        # the first ``finished:+N`` reading, otherwise the delta vs. an
        # empty baseline is misleading.
        await asyncio.sleep(STATUS_PRINT_INTERVAL_SEC)
        while self.running:
            try:
                await self._emit_status_line(sc, samples)
            except Exception as e:
                logger.warning(f"status print failed: {type(e).__name__}: {e}")
            await asyncio.sleep(STATUS_PRINT_INTERVAL_SEC)

    async def _emit_status_line(
        self, sc: SystemConfigDAO, samples: SampleResultsAdapter,
    ) -> None:
        import time as _t
        tids = await sc.get_param_value("current_task_ids") or {}
        champion = await sc.get_param_value("champion") or {}
        battle = await sc.get_param_value("current_battle") or {}
        envs_raw = await sc.get_param_value("environments", default={}) or {}
        task_ids_by_env: Dict[str, List[int]] = tids.get("task_ids", {}) or {}
        refresh_block = int(tids.get("refreshed_at_block", 0) or 0)
        subjects: List[tuple[str, str, str]] = []
        if champion.get("hotkey") and champion.get("revision"):
            subjects.append(("champion", str(champion["hotkey"]), str(champion["revision"])))
        challenger = battle.get("challenger") if isinstance(battle, dict) else None
        if (
            isinstance(challenger, dict)
            and challenger.get("hotkey")
            and challenger.get("revision")
        ):
            pair = ("challenger", str(challenger["hotkey"]), str(challenger["revision"]))
            if pair not in subjects:
                subjects.append(pair)

        now = _t.time()
        elapsed_s = max(1, int(now - self._last_status_at)) if self._last_status_at else 0
        in_flight_total = 0
        total_done = 0
        total_target = 0
        total_delta = 0
        env_stats: Dict[str, Dict[str, int]] = {}
        rows: Dict[str, Dict[str, Any]] = {}

        for env in sorted(self.envs):
            ids = task_ids_by_env.get(env, []) or []
            sampling_count = _sampling_count_for_env(envs_raw, env, len(ids))
            subject_targets = {
                "champion": len(ids),
                "challenger": min(len(ids), sampling_count),
            }
            target = sum(subject_targets[kind] for kind, _, _ in subjects)
            done = 0
            if ids and subjects and refresh_block:
                for kind, hk, rev in subjects:
                    try:
                        count = await samples.count_samples_for_tasks(
                            hk, rev, env, ids, refresh_block=refresh_block,
                        )
                        done += min(count, subject_targets[kind])
                    except Exception as e:
                        logger.debug(f"status count failed for {env}: {e}")
            running = int(self.in_flight_values[env].value)
            prev = self._last_done.get(env, 0)
            delta = max(0, done - prev) if self._last_status_at else 0
            env_stats[env] = {
                "target": target,
                "done": done,
                "running": running,
                "delta": delta,
            }
            rows[env] = {
                "target": target,
                "done": done,
                "running": running,
                "delta": delta,
            }
            self._last_done[env] = done
            in_flight_total += running
            total_done += done
            total_target += target
            total_delta += delta

        new_caps = _compute_adaptive_env_caps(
            self.envs,
            env_stats,
            {env: int(v.value) for env, v in self.env_cap_values.items()},
            global_budget=GLOBAL_DISPATCH_BUDGET,
        )
        for env, cap in new_caps.items():
            self.env_cap_values[env].value = cap

        per_env_parts: List[str] = []
        for env in sorted(self.envs):
            row = rows[env]
            done = int(row["done"])
            target = int(row["target"])
            running = int(row["running"])
            delta = int(row["delta"])
            rate_per_h = (delta / elapsed_s * 3600) if (elapsed_s and delta) else 0
            short = env.split(":")[-1].split("-")[0].lower()
            cap = int(self.env_cap_values[env].value)
            if self._last_status_at:
                per_env_parts.append(
                    f"{short}@{done}/{target} +{delta} "
                    f"rate:{rate_per_h:.0f}/h run:{running}/{cap}"
                )
            else:
                per_env_parts.append(f"{short}@{done}/{target} run:{running}/{cap}")

        total_rate = (total_delta / elapsed_s * 3600) if (elapsed_s and total_delta) else 0
        if self._last_status_at:
            head = (
                f"[STATUS] running={in_flight_total}/{GLOBAL_DISPATCH_BUDGET} | "
                f"done={total_done}/{total_target} +{total_delta} in {elapsed_s}s "
                f"rate:{total_rate:.0f}/h"
            )
        else:
            head = (
                f"[STATUS] running={in_flight_total}/{GLOBAL_DISPATCH_BUDGET} | "
                f"done={total_done}/{total_target} (baseline)"
            )
        logger.info(f"{head} | " + " | ".join(per_env_parts))
        self._last_status_at = now


async def _enabled_envs() -> List[str]:
    """Read system_config.environments and return sampling-enabled env keys.

    Only ``enabled_for_sampling`` starts executor workers.
    """
    config_dao = SystemConfigDAO()
    envs_raw = await config_dao.get_param_value("environments", default={}) or {}
    out = []
    for name, cfg in envs_raw.items():
        if not isinstance(cfg, dict):
            continue
        sampling_flag = cfg.get("enabled_for_sampling", False)
        if sampling_flag:
            out.append(name)
    return out


def _initial_env_cap(envs: List[str], global_budget: int) -> int:
    if not envs:
        return max(1, global_budget)
    return max(1, math.ceil(global_budget / len(envs)))


def _compute_adaptive_env_caps(
    envs: List[str],
    stats: Dict[str, Dict[str, int]],
    previous_caps: Dict[str, int],
    *,
    global_budget: int,
) -> Dict[str, int]:
    """Adapt per-env in-flight caps from live progress.

    The objective is shortest wall-clock completion, not static fairness:
    every backlogged env keeps a probe floor so it cannot starve, envs
    that cannot consume their current cap release capacity, and the
    remaining budget goes to saturated envs weighted by estimated time
    to finish (remaining work divided by recent completions). If no env
    is actually able to consume more slots, the function intentionally
    returns less than ``global_budget`` instead of refilling the pool
    with caps that would only create waiting coroutines.
    """
    active = [
        env for env in envs
        if (stats.get(env, {}).get("target", 0) - stats.get(env, {}).get("done", 0)) > 0
        or stats.get(env, {}).get("running", 0) > 0
    ]
    if not active:
        initial = _initial_env_cap(envs, global_budget)
        return {env: initial for env in envs}
    fair = _initial_env_cap(active, global_budget)
    floor = max(1, fair // 8)
    # Envs with no current work keep their previous cap so a new battle
    # can ramp immediately between status ticks. They do not enter the
    # active budget math below because they are not consuming slots.
    caps: Dict[str, int] = {
        env: max(floor, int(previous_caps.get(env, fair) or fair))
        for env in envs
    }
    weights: Dict[str, float] = {}
    for env in active:
        row = stats.get(env, {})
        target = int(row.get("target", 0) or 0)
        done = int(row.get("done", 0) or 0)
        running = max(0, int(row.get("running", 0) or 0))
        delta = max(0, int(row.get("delta", 0) or 0))
        remaining = max(0, target - done)
        prev = max(1, int(previous_caps.get(env, fair) or fair))
        if remaining <= 0:
            # Let existing calls drain; do not launch a new wave for an
            # env whose current subject set is complete. Keep fair share
            # ready so a newly started battle is not stuck at one-at-a-time
            # until the next status interval.
            caps[env] = max(fair, min(prev, running or fair))
            continue

        saturated = running >= max(floor, int(prev * 0.75))
        if saturated:
            if delta <= 0:
                # Saturated but no recent completions: preserve a bounded
                # fair-share launch cap so long-running calls can finish,
                # but do not keep a previously inflated cap forever.
                caps[env] = max(floor, min(prev, fair))
                continue
            # Saturated envs compete for the shared budget by pressure.
            # Keep only the probe floor before allocation so a nearly done
            # env can release future launch capacity to a lagging peer even
            # while its existing calls are still draining.
            caps[env] = floor
            # Remaining / delta approximates intervals-to-finish. With no
            # completions yet, the branch above keeps the env at its current
            # share instead of amplifying a possible stall.
            weights[env] = float(remaining) / float(delta)
        else:
            # Minimum non-starving probe. Even an env with zero in-flight gets
            # enough slots to prove it can make progress after cold starts,
            # container stalls, or transient DB gaps. Underused envs stay at
            # this local-demand cap and do not receive extra budget.
            caps[env] = max(floor, min(prev, running + floor))

    total = sum(caps[env] for env in active)
    if total > global_budget:
        caps.update(
            _trim_caps(
                {env: caps[env] for env in active},
                global_budget=global_budget,
                floor=1,
            )
        )
        return caps

    extra = global_budget - total
    if extra <= 0 or not weights:
        return caps

    total_weight = sum(weights.values())
    allocations: Dict[str, int] = {}
    fractions: List[tuple[float, str]] = []
    used = 0
    for env, weight in weights.items():
        raw = extra * (weight / total_weight)
        whole = int(raw)
        allocations[env] = whole
        fractions.append((raw - whole, env))
        used += whole
    for _, env in sorted(fractions, reverse=True):
        if used >= extra:
            break
        allocations[env] += 1
        used += 1
    for env, add in allocations.items():
        caps[env] += add
    caps.update(
        _trim_caps(
            {env: caps[env] for env in active},
            global_budget=global_budget,
            floor=1,
        )
    )
    return caps


def _sampling_count_for_env(envs_raw: Any, env: str, default: int) -> int:
    if not isinstance(envs_raw, dict):
        return default
    cfg = envs_raw.get(env) or {}
    if not isinstance(cfg, dict):
        return default
    sampling = cfg.get("sampling") or cfg.get("window_config") or {}
    if not isinstance(sampling, dict):
        return default
    raw = sampling.get("sampling_count", default)
    if raw is None:
        return default
    try:
        count = int(raw)
    except (TypeError, ValueError):
        return default
    return max(0, count)


def _trim_caps(caps: Dict[str, int], *, global_budget: int, floor: int) -> Dict[str, int]:
    out = {env: max(floor, int(cap)) for env, cap in caps.items()}
    total = sum(out.values())
    while total > global_budget:
        candidates = [env for env, cap in out.items() if cap > floor]
        if not candidates:
            break
        env = max(candidates, key=lambda e: out[e])
        out[env] -= 1
        total -= 1
    return out


async def _run() -> None:
    await init_client()
    envs = await _enabled_envs()
    if not envs:
        logger.error(
            "executor: no sampling-enabled envs in system_config.environments"
        )
        return

    manager = ExecutorManager(envs=envs, verbosity=1)
    await manager.start()

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except (NotImplementedError, RuntimeError):
            pass

    try:
        await stop_event.wait()
    finally:
        await manager.stop()
        await close_client()


@click.command()
@click.option("-v", "--verbose", count=True, default=1)
def main(verbose: int) -> None:
    """Run the per-env executor manager."""
    setup_logging(verbosity=verbose, component="executor")
    asyncio.run(_run())


if __name__ == "__main__":
    main()
