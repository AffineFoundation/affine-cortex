"""
``af servers executor`` — manager that spawns one subprocess per env.

The manager process does very little hot-path work: it spawns N
``WorkerProcess`` instances (one per enabled env), restarts them if
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
STATUS_PRINT_INTERVAL_SEC = 30


class ExecutorManager:
    def __init__(self, envs: List[str], *, verbosity: int = 1):
        self.envs = envs
        self.verbosity = verbosity
        self.mp_ctx = multiprocessing.get_context("spawn")
        # Single cross-process bounded semaphore = the real concurrency
        # gate. Sized to the b300 saturation point so the inference
        # backend is the bottleneck (where we want it). Workers
        # contend on this sem before every evaluate(); envs with more
        # remaining work submit more concurrent acquire() attempts and
        # naturally win a proportional share of slots — that's the
        # cross-env priority, no explicit queue needed.
        self.global_sem = self.mp_ctx.BoundedSemaphore(GLOBAL_DISPATCH_BUDGET)
        # Per-env in-flight counter; worker writes (single writer), manager
        # reads. ``lock=False`` is safe: only one process writes each
        # Value, and aligned c_int reads are atomic on CPython.
        self.in_flight_values: Dict[str, Any] = {
            env: self.mp_ctx.Value("i", 0, lock=False) for env in envs
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
                        wp.start()
                    except Exception as e:
                        logger.error(f"[{wp.env}] restart raised: {e}")

    async def _status_printer(self) -> None:
        """Periodic ``[STATUS]`` line in the legacy executor's format.

        Reads done counts from ``sample_results`` (champion-only — the
        challenger row count rides on the same window so isn't reported
        separately) and per-env in-flight from the shared
        ``in_flight_values`` Values. Format:

            [STATUS] running=N/480 | env@done/target +delta in Xs rate:Y/h
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
        task_ids_by_env: Dict[str, List[int]] = tids.get("task_ids", {}) or {}
        refresh_block = int(tids.get("refreshed_at_block", 0) or 0)
        hk = champion.get("hotkey")
        rev = champion.get("revision")

        now = _t.time()
        elapsed_s = max(1, int(now - self._last_status_at)) if self._last_status_at else 0
        in_flight_total = 0
        per_env_parts: List[str] = []
        total_done = 0
        total_target = 0
        total_delta = 0

        for env in sorted(self.envs):
            ids = task_ids_by_env.get(env, []) or []
            target = len(ids)
            done = 0
            if ids and hk and rev and refresh_block:
                try:
                    done = await samples.count_samples_for_tasks(
                        hk, rev, env, ids, refresh_block=refresh_block,
                    )
                except Exception as e:
                    logger.debug(f"status count failed for {env}: {e}")
            running = int(self.in_flight_values[env].value)
            prev = self._last_done.get(env, 0)
            delta = max(0, done - prev) if self._last_status_at else 0
            rate_per_h = (delta / elapsed_s * 3600) if (elapsed_s and delta) else 0
            short = env.split(":")[-1].split("-")[0].lower()
            if self._last_status_at:
                per_env_parts.append(
                    f"{short}@{done}/{target} +{delta} rate:{rate_per_h:.0f}/h run:{running}"
                )
            else:
                per_env_parts.append(f"{short}@{done}/{target} run:{running}")
            self._last_done[env] = done
            in_flight_total += running
            total_done += done
            total_target += target
            total_delta += delta

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
    """Read system_config.environments and return enabled env keys."""
    config_dao = SystemConfigDAO()
    envs_raw = await config_dao.get_param_value("environments", default={}) or {}
    out = []
    for name, cfg in envs_raw.items():
        if isinstance(cfg, dict) and cfg.get("enabled", True):
            out.append(name)
    return out


async def _run() -> None:
    await init_client()
    envs = await _enabled_envs()
    if not envs:
        logger.error("executor: no enabled envs in system_config.environments")
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
