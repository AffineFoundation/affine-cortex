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
from typing import Any, Dict, List, Mapping, Optional

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


def _as_uid(raw: Any) -> Optional[int]:
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


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
        # (kind, hotkey, revision) of the subject the previous status was
        # reporting. When it changes between two ticks (challenger lost →
        # role flips to champion, or new challenger picked), the per-env
        # delta would otherwise be `current_subject_count - prior_subject_count`
        # — a meaningless cross-miner subtraction — so we zero deltas for
        # one frame.
        self._last_subject_key: Optional[tuple] = None
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
        # subjects: (kind, uid, hotkey, revision). uid is carried so the
        # STATUS label can disambiguate "[champion U213]" vs the new
        # challenger UID — operators were misreading the role-only label
        # as identity.
        subjects: List[tuple[str, Optional[int], str, str]] = []
        if champion.get("hotkey") and champion.get("revision"):
            subjects.append((
                "champion",
                _as_uid(champion.get("uid")),
                str(champion["hotkey"]),
                str(champion["revision"]),
            ))
        challenger = battle.get("challenger") if isinstance(battle, dict) else None
        if (
            isinstance(challenger, dict)
            and challenger.get("hotkey")
            and challenger.get("revision")
        ):
            pair = (
                "challenger",
                _as_uid(challenger.get("uid")),
                str(challenger["hotkey"]),
                str(challenger["revision"]),
            )
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

        # Display the currently-sampling miner's progress only:
        #   no battle → champion (still ramping samples)
        #   battle in flight → challenger (champion already done by definition)
        # Cumulative champion+challenger numbers are misleading once
        # champion is finished — operators want to see how close the
        # challenger is to ``sampling_count`` so they can predict decide.
        active_kind: Optional[str] = None
        active_uid: Optional[int] = None
        active_subject_key: Optional[tuple] = None
        for kind, uid, hk, rev in subjects:
            if kind == "challenger":
                active_kind = "challenger"
                active_uid = uid
                active_subject_key = (kind, hk, rev)
                break
        if active_kind is None and subjects:
            kind, uid, hk, rev = subjects[0]  # champion phase
            active_kind = kind
            active_uid = uid
            active_subject_key = (kind, hk, rev)

        # The subject changed when the role+identity tuple differs from the
        # previous status print. On the first frame after a change, zero
        # deltas so we don't subtract one miner's count from another's.
        subject_changed = (
            self._last_subject_key is not None
            and active_subject_key is not None
            and active_subject_key != self._last_subject_key
        )

        for env in sorted(self.envs):
            ids = task_ids_by_env.get(env, []) or []
            sampling_count = _sampling_count_for_env(envs_raw, env, len(ids))
            subject_targets = {
                "champion": len(ids),
                "challenger": min(len(ids), sampling_count),
            }
            # Per-subject done (used for adaptive cap math + the displayed
            # active-subject row).
            per_done: Dict[str, int] = {}
            if ids and subjects and refresh_block:
                for kind, _uid, hk, rev in subjects:
                    try:
                        count = await samples.count_samples_for_tasks(
                            hk, rev, env, ids, refresh_block=refresh_block,
                        )
                        per_done[kind] = min(count, subject_targets[kind])
                    except Exception as e:
                        logger.debug(f"status count failed for {env}: {e}")
                        per_done[kind] = 0
            target = subject_targets.get(active_kind, 0) if active_kind else 0
            done = per_done.get(active_kind, 0) if active_kind else 0
            running = int(self.in_flight_values[env].value)
            if subject_changed:
                delta = 0
            else:
                prev = self._last_done.get(env, 0)
                delta = max(0, done - prev) if self._last_status_at else 0
            # Adaptive cap math still uses the combined (champ + chal) target
            # and per-cycle delta so an env that finished one subject doesn't
            # look "complete" from the cap planner's POV while the other
            # subject still has work.
            combined_target = sum(subject_targets[k] for k, _uid, _hk, _rev in subjects)
            combined_done = sum(per_done.get(k, 0) for k, _uid, _hk, _rev in subjects)
            env_stats[env] = {
                "target": combined_target,
                "done": combined_done,
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
            priorities={
                env: _sampling_priority_for_env(envs_raw, env)
                for env in self.envs
            },
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
        if active_kind and active_uid is not None:
            kind_label = f"[{active_kind} U{active_uid}]"
        elif active_kind:
            kind_label = f"[{active_kind}]"
        else:
            kind_label = ""
        if self._last_status_at:
            head = (
                f"[STATUS] {kind_label} "
                f"running={in_flight_total}/{GLOBAL_DISPATCH_BUDGET} | "
                f"done={total_done}/{total_target} +{total_delta} in {elapsed_s}s "
                f"rate:{total_rate:.0f}/h"
            )
        else:
            head = (
                f"[STATUS] {kind_label} "
                f"running={in_flight_total}/{GLOBAL_DISPATCH_BUDGET} | "
                f"done={total_done}/{total_target} (baseline)"
            )
        logger.info(f"{head} | " + " | ".join(per_env_parts))
        self._last_status_at = now
        self._last_subject_key = active_subject_key


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
    priorities: Optional[Mapping[str, int]] = None,
) -> Dict[str, int]:
    """Adapt per-env in-flight caps from live progress.

    The objective is shortest wall-clock completion, not static fairness:
    every backlogged env gets enough launch capacity to make progress,
    envs that cannot consume their current cap release capacity, and
    saturated envs grow gradually instead of one fast-returning env
    taking the whole global budget. Slow envs with no recent completions
    still ramp while they are saturated so long-running environments
    cannot starve behind short-window ``delta=0`` readings.

    ``priorities`` tiers allocation: envs sharing the highest priority
    claim the bulk of ``global_budget`` first, the residue cascades down
    to the next tier, etc. Every backlogged env still keeps a probe
    floor regardless of tier so a lower-tier env cannot starve
    completely. ``priorities`` is ``None`` or all envs sharing one
    priority value is the single-tier case: one loop iteration runs
    with ``tier_budget == global_budget``, equivalent to a direct
    un-tiered pool call.
    """
    priorities_map = priorities or {}
    by_tier: Dict[int, List[str]] = {}
    for env in envs:
        by_tier.setdefault(int(priorities_map.get(env, 0)), []).append(env)
    tiers_desc = sorted(by_tier.keys(), reverse=True)

    def is_active(env: str) -> bool:
        row = stats.get(env, {})
        return (
            int(row.get("target", 0) or 0) - int(row.get("done", 0) or 0) > 0
            or int(row.get("running", 0) or 0) > 0
        )

    # ``reserve_floor`` is the probe size held back per downstream-tier
    # active env so a higher-priority tier cannot starve a backlogged
    # lower-priority env. Sized like the un-tiered allocator's internal
    # floor (``fair // 8``) so a lower-priority env keeps the same probe
    # size it would have received in a flat allocation. With one tier
    # the reservation is naturally zero (no downstream) and
    # ``tier_budget`` collapses to ``global_budget`` — a single loop
    # iteration that delegates to the pool, bit-identical to a direct
    # un-tiered call. With zero active envs anywhere the reservation
    # is also zero — nothing to keep alive — and each tier pool returns
    # placeholder caps from its own budget split.
    n_active = sum(1 for env in envs if is_active(env))
    reserve_floor = (
        max(1, math.ceil(global_budget / n_active) // 8) if n_active else 0
    )

    caps: Dict[str, int] = {}
    remaining_budget = global_budget
    for tier_idx, tier in enumerate(tiers_desc):
        tier_envs = by_tier[tier]
        lower_active = sum(
            1 for t in tiers_desc[tier_idx + 1:]
            for e in by_tier[t] if is_active(e)
        )
        tier_budget = max(1, remaining_budget - lower_active * reserve_floor)
        tier_caps = _allocate_within_pool(
            tier_envs, stats, previous_caps, pool_budget=tier_budget,
        )
        caps.update(tier_caps)
        # Only active envs consume from the cascading budget. Inactive
        # envs hold their previous-cap legacy value (so a new battle can
        # ramp without re-deriving a fair share from scratch) but they
        # are not running anything, so they do not subtract from what
        # lower tiers receive.
        consumed = sum(
            tier_caps.get(env, 0) for env in tier_envs if is_active(env)
        )
        remaining_budget = max(0, remaining_budget - consumed)

    return caps


def _allocate_within_pool(
    envs: List[str],
    stats: Dict[str, Dict[str, int]],
    previous_caps: Dict[str, int],
    *,
    pool_budget: int,
) -> Dict[str, int]:
    """Pressure-weighted cap allocation for a single pool of envs.

    Extracted from ``_compute_adaptive_env_caps`` so the tiered
    orchestrator can call it once per priority tier. Behavior for a
    single pool (``pool_budget == global_budget``, ``envs == all envs``)
    is identical to the historical un-tiered allocator.
    """
    active = [
        env for env in envs
        if (stats.get(env, {}).get("target", 0) - stats.get(env, {}).get("done", 0)) > 0
        or stats.get(env, {}).get("running", 0) > 0
    ]
    if not active:
        initial = _initial_env_cap(envs, pool_budget)
        return {env: initial for env in envs}
    fair = _initial_env_cap(active, pool_budget)
    floor = max(1, fair // 8)
    ramp = max(floor, fair // 4)
    # Envs with no current work keep their previous cap so a new battle
    # can ramp immediately between status ticks. They do not enter the
    # active budget math below because they are not consuming slots.
    caps: Dict[str, int] = {
        env: max(floor, int(previous_caps.get(env, fair) or fair))
        for env in envs
    }
    weights: Dict[str, float] = {}
    opportunistic_weights: Dict[str, float] = {}
    ceilings: Dict[str, int] = {}
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
        remaining_cap = max(floor, min(remaining, fair))
        growth_cap = max(remaining_cap, min(remaining, prev + ramp))
        if running <= 0:
            # Cold start / restart: give every backlogged env its fair
            # launch share immediately. A probe-only start is exactly what
            # starves long-latency envs after executor restarts.
            caps[env] = remaining_cap
            continue

        if saturated:
            if delta <= 0:
                # Saturated but no recent completions means either a slow
                # env or a stall. Grow up to fair share when it is still
                # below fair. Once it has reached fair, it can still receive
                # idle capacity slowly, but never a one-tick blow-up.
                if prev < remaining_cap:
                    caps[env] = max(floor, min(remaining_cap, prev + ramp))
                else:
                    caps[env] = remaining_cap
                    if remaining > caps[env]:
                        ceilings[env] = min(remaining, prev + ramp)
                        opportunistic_weights[env] = max(1.0, float(remaining))
                continue
            # Saturated envs compete for the shared budget by pressure.
            # Keep only the probe floor before allocation so a nearly done
            # env can release future launch capacity to a lagging peer even
            # while its existing calls are still draining.
            caps[env] = floor
            ceilings[env] = growth_cap
            # Remaining / delta approximates intervals-to-finish. With no
            # completions yet, the branch above keeps the env at its current
            # share instead of amplifying a possible stall.
            weights[env] = float(remaining) / float(delta)
        else:
            # Minimum non-starving probe. Even an env with zero in-flight gets
            # enough slots to prove it can make progress after cold starts,
            # container stalls, or transient DB gaps. Underused envs stay at
            # this local-demand cap and do not receive extra budget.
            caps[env] = max(floor, min(remaining, running + ramp))

    total = sum(caps[env] for env in active)
    if total > pool_budget:
        caps.update(
            _trim_caps(
                {env: caps[env] for env in active},
                global_budget=pool_budget,
                floor=1,
            )
        )
        return caps

    extra = pool_budget - total
    if extra <= 0 or (not weights and not opportunistic_weights):
        return caps

    _allocate_weighted_extra(caps, weights, ceilings, extra)
    used_extra = sum(caps[env] for env in active) - total
    if used_extra < extra:
        _allocate_weighted_extra(
            caps,
            opportunistic_weights,
            ceilings,
            extra - used_extra,
        )
    caps.update(
        _trim_caps(
            {env: caps[env] for env in active},
            global_budget=pool_budget,
            floor=1,
        )
    )
    return caps


def _allocate_weighted_extra(
    caps: Dict[str, int],
    weights: Dict[str, float],
    ceilings: Dict[str, int],
    extra: int,
) -> None:
    """Distribute spare cap without allowing a one-tick blow-up.

    The loop is intentionally simple: ``global_budget`` is small (hundreds),
    and one-slot-at-a-time allocation respects per-env ceilings exactly.
    """
    if extra <= 0 or not weights:
        return
    allocated = {env: 0 for env in weights}
    for _ in range(extra):
        candidates = [
            env for env in weights
            if caps.get(env, 0) < ceilings.get(env, caps.get(env, 0))
        ]
        if not candidates:
            return
        env = max(candidates, key=lambda e: weights[e] / (allocated[e] + 1))
        caps[env] += 1
        allocated[env] += 1


def _sampling_field(envs_raw: Any, env: str, key: str, default: Any) -> Any:
    """Walk ``environments[env].(sampling|window_config)[key]`` defensively.

    The DDB-backed config can return non-dict values during partial
    migrations or hand edits, so every nesting step short-circuits to
    ``default``. Returns the raw value (or ``default``); callers cast +
    clamp as the field demands.
    """
    if not isinstance(envs_raw, dict):
        return default
    cfg = envs_raw.get(env) or {}
    if not isinstance(cfg, dict):
        return default
    sampling = cfg.get("sampling") or cfg.get("window_config") or {}
    if not isinstance(sampling, dict):
        return default
    raw = sampling.get(key, default)
    return default if raw is None else raw


def _sampling_count_for_env(envs_raw: Any, env: str, default: int) -> int:
    raw = _sampling_field(envs_raw, env, "sampling_count", default)
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return default


def _sampling_priority_for_env(envs_raw: Any, env: str, default: int = 0) -> int:
    """Read ``environments[env].sampling.priority`` from the live config.

    Higher integer = scheduled before lower-priority envs by
    ``_compute_adaptive_env_caps``. Missing / non-dict / non-numeric
    values fall through to ``default`` so untouched configs preserve
    flat (single-tier) allocation."""
    raw = _sampling_field(envs_raw, env, "priority", default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


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
