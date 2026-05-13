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
from affine.database.dao.system_config import SystemConfigDAO
from affine.src.executor.config import get_max_concurrent
from affine.src.executor.worker_process import WorkerProcess


HEALTH_CHECK_INTERVAL_SEC = 10
STATS_DRAIN_INTERVAL_SEC = 0.2


class ExecutorManager:
    def __init__(self, envs: List[str], *, verbosity: int = 1):
        self.envs = envs
        self.verbosity = verbosity
        self.stats_queue: multiprocessing.Queue = multiprocessing.Queue()
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
                max_concurrent=get_max_concurrent(env),
                verbosity=self.verbosity,
            )
            wp.start()
            self.workers.append(wp)
        self.running = True
        asyncio.create_task(self._stats_collector())
        asyncio.create_task(self._health_checker())
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
