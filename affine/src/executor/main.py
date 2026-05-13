"""
``af servers executor`` — manager that spawns one subprocess per env.

The manager process owns nothing performance-critical: it spawns N
``WorkerProcess`` instances (one per enabled env), restarts them if
they die, and aggregates stats reported via a multiprocessing.Queue.
All actual sampling lives in the subprocesses.

Concurrency: each subprocess has its own asyncio loop + SDKEnvironment
(Docker + SSH tunnel + paramiko Transport). No paramiko thread budget
is shared across envs, so 480 across 7 envs spreads as ~68/env per its
own Transport instead of 480 sharing one.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import queue
import signal
from typing import Any, Dict, List

import click

from affine.core.setup import logger, setup_logging
from affine.database import close_client, init_client
from affine.database.dao.sample_results import SampleResultsDAO
from affine.database.dao.system_config import SystemConfigDAO
from affine.src.executor.config import get_max_concurrent
from affine.src.executor.worker_process import WorkerProcess
from affine.src.scorer.cap_scheduler import compute_caps
from affine.src.scorer.dao_adapters import SampleResultsAdapter


HEALTH_CHECK_INTERVAL_SEC = 10
STATS_DRAIN_INTERVAL_SEC = 0.2
CAP_REFRESH_INTERVAL_SEC = 30
# multiprocessing.Value(c_int) atomic read on CPython, no lock needed for
# our read-mostly cross-process cap broadcast.
_CAP_C_TYPE = "i"


class ExecutorManager:
    def __init__(self, envs: List[str], *, verbosity: int = 1):
        self.envs = envs
        self.verbosity = verbosity
        self.mp_ctx = multiprocessing.get_context("spawn")
        self.stats_queue: multiprocessing.Queue = self.mp_ctx.Queue()
        # Shared atomic-int per env; manager writes, worker reads at every
        # tick to resize its dispatch semaphore. Initial value = static cap
        # from config so workers can start dispatching before the first
        # ``_cap_publisher`` cycle lands.
        self.cap_values: Dict[str, Any] = {
            env: self.mp_ctx.Value(_CAP_C_TYPE, get_max_concurrent(env))
            for env in envs
        }
        self.workers: List[WorkerProcess] = []
        self.aggregated: Dict[str, Dict[str, Any]] = {}
        self.running = False
        logger.info(
            f"ExecutorManager init: envs={envs}, "
            f"max_concurrent_per_env={ {e: get_max_concurrent(e) for e in envs} }"
        )

    async def start(self) -> None:
        if self.running:
            return
        for idx, env in enumerate(self.envs):
            wp = WorkerProcess(
                worker_id=idx, env=env,
                stats_queue=self.stats_queue,
                cap_value=self.cap_values[env],
                max_concurrent=get_max_concurrent(env),
                verbosity=self.verbosity,
            )
            wp.start()
            self.workers.append(wp)
        self.running = True
        asyncio.create_task(self._stats_collector())
        asyncio.create_task(self._health_checker())
        asyncio.create_task(self._cap_publisher())
        logger.info(f"Spawned {len(self.workers)} worker subprocesses")

    async def stop(self) -> None:
        self.running = False
        for wp in self.workers:
            try:
                wp.terminate()
            except Exception as e:
                logger.warning(f"terminate({wp.env}) raised: {e}")
        self.workers.clear()

    async def _stats_collector(self) -> None:
        while self.running:
            try:
                payload = self.stats_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(STATS_DRAIN_INTERVAL_SEC)
                continue
            env = payload.get("env")
            if env:
                self.aggregated[env] = payload

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

    async def _cap_publisher(self) -> None:
        """Periodic cross-env priority rebalance.

        Reads remaining samples per env and broadcasts the resulting
        per-env dispatch cap to each worker via shared atomic Values.
        The env with the most remaining work gets the largest slice of
        the global budget — keeping all envs converging to "done"
        together. No DB writes; the caps are derived from current DB
        state, so a manager restart simply recomputes them.

        Workers read ``cap_value.value`` at the top of every tick;
        atomic int read on CPython is lock-free.
        """
        sc = SystemConfigDAO()
        samples = SampleResultsAdapter(
            dao=SampleResultsDAO(), validator_hotkey="executor-manager",
        )
        while self.running:
            try:
                remaining = await self._compute_remaining(sc, samples)
                # Pre-window state (no task_ids / no champion) reports
                # ``remaining=0`` for every env. Don't downgrade caps to
                # min_cap in that case — keep the static config cap so
                # the first window's first tick has full dispatch.
                if any(r > 0 for r in remaining.values()):
                    caps = compute_caps(remaining)
                    for env, cap in caps.items():
                        val = self.cap_values.get(env)
                        if val is not None:
                            val.value = int(cap)
                    logger.info(
                        "executor caps rebalanced: "
                        + ", ".join(f"{e}={c}" for e, c in sorted(caps.items()))
                    )
            except Exception as e:
                logger.warning(f"cap publish failed: {type(e).__name__}: {e}")
            await asyncio.sleep(CAP_REFRESH_INTERVAL_SEC)

    async def _compute_remaining(
        self, sc: SystemConfigDAO, samples: SampleResultsAdapter,
    ) -> Dict[str, int]:
        """Return ``{env: remaining_for_champion}``. Envs with no champion
        / no task pool yet report 0 so they receive the min_cap floor."""
        tids = await sc.get_param_value("current_task_ids") or {}
        champion = await sc.get_param_value("champion") or {}
        task_ids_by_env: Dict[str, List[int]] = tids.get("task_ids", {}) or {}
        refresh_block = int(tids.get("refreshed_at_block", 0) or 0)
        hk = champion.get("hotkey")
        rev = champion.get("revision")
        remaining: Dict[str, int] = {}
        for env in self.envs:
            ids = task_ids_by_env.get(env, []) or []
            if not ids or not hk or not rev or not refresh_block:
                remaining[env] = 0
                continue
            try:
                done = await samples.count_samples_for_tasks(
                    hk, rev, env, ids, refresh_block=refresh_block,
                )
            except Exception as e:
                logger.debug(f"remaining count failed for {env}: {e}")
                done = 0
            remaining[env] = max(0, len(ids) - int(done))
        return remaining


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
