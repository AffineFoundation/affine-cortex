"""
Worker subprocess wrapper.

One subprocess per env. The subprocess owns its own asyncio loop +
SDKEnvironment (Docker container + SSH tunnel + paramiko Transport) so
no Python GIL or paramiko thread budget is shared across envs.
"""

from __future__ import annotations

import asyncio
import multiprocessing
from typing import Any, Optional

from affine.core.setup import logger, setup_logging


def run_worker_subprocess(
    worker_id: int,
    env: str,
    max_concurrent: int,
    stats_queue: multiprocessing.Queue,
    cap_value: Any,
    verbosity: int = 1,
) -> None:
    """Subprocess entry point — runs one ExecutorWorker until killed.

    ``cap_value`` is a ``multiprocessing.Value(c_int)`` the manager writes
    to broadcast cross-env priority caps; the worker reads it at every
    tick. Passed positionally so it survives the spawn-context pickle.
    """
    from affine.src.executor.worker import ExecutorWorker

    setup_logging(verbosity, component="executor")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    worker: Optional[ExecutorWorker] = None
    try:
        worker = ExecutorWorker(
            worker_id=worker_id, env=env, max_concurrent=max_concurrent,
            cap_value=cap_value,
        )
        loop.run_until_complete(worker.initialize())
        worker.start()
        loop.run_until_complete(worker.run())
    except KeyboardInterrupt:
        logger.info(f"[{env}] subprocess received SIGINT")
    except Exception as e:
        logger.error(f"[{env}] subprocess fatal: {e}", exc_info=True)
    finally:
        if worker is not None:
            worker.stop()
        try:
            pending = asyncio.all_tasks(loop)
            for t in pending:
                t.cancel()
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
        except Exception as e:
            logger.warning(f"[{env}] cleanup raised: {e}")
        try:
            if not loop.is_closed():
                loop.close()
        except Exception as e:
            logger.warning(f"[{env}] loop close raised: {e}")


class WorkerProcess:
    """Manager-side handle to one subprocess."""

    def __init__(
        self,
        worker_id: int,
        env: str,
        stats_queue: multiprocessing.Queue,
        cap_value: Any,
        *,
        max_concurrent: int = 60,
        verbosity: int = 1,
    ):
        self.worker_id = worker_id
        self.env = env
        self.max_concurrent = max_concurrent
        self.stats_queue = stats_queue
        self.cap_value = cap_value
        self.verbosity = verbosity
        self._proc: Optional[multiprocessing.Process] = None

    def start(self) -> None:
        ctx = multiprocessing.get_context("spawn")
        self._proc = ctx.Process(
            target=run_worker_subprocess,
            args=(self.worker_id, self.env, self.max_concurrent,
                  self.stats_queue, self.cap_value, self.verbosity),
            name=f"executor-{self.env}",
            daemon=False,
        )
        self._proc.start()
        logger.info(f"[{self.env}] subprocess started pid={self._proc.pid}")

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def terminate(self, timeout: float = 10.0) -> None:
        if self._proc is None:
            return
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=timeout)
            if self._proc.is_alive():
                logger.warning(f"[{self.env}] subprocess did not exit; killing")
                self._proc.kill()
                self._proc.join(timeout=5)
        self._proc = None
