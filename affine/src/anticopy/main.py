"""
Entry points for the two CEAC services:

  ``af servers anticopy-refresh``  — daily rollout pool refresher
  ``af servers anticopy-worker``   — long-running forward + verdict worker
"""

from __future__ import annotations

import asyncio
import os
import shlex
import signal
import subprocess
import time
from urllib.parse import urlparse

import aiohttp
import click

from affine.core.setup import logger, setup_logging
from affine.database.client import close_client, init_client


# Verbosity flag mirrors the other servers entries: -v for INFO, -vv DEBUG.
def _verbosity_opt(f):
    return click.option(
        "-v", "--verbose", count=True, default=1,
        help="Increase logging verbosity (-v INFO, -vv DEBUG)",
    )(f)


async def _bootstrap_run(coro_factory):
    """Boot one or several services in a single asyncio process.

    ``coro_factory`` may return either a single service instance
    (back-compat with worker_main) or a list/tuple of them
    (refresh_main runs RolloutRefreshService + VerdictBackfillService
    side by side). Stop signal cancels all runners.
    """
    await init_client()
    produced = coro_factory()
    services = list(produced) if isinstance(produced, (list, tuple)) else [produced]

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:                # Windows / non-main thread
            pass

    runners = [asyncio.create_task(s.run()) for s in services]
    waiter = asyncio.create_task(stop_event.wait())
    try:
        await asyncio.wait(
            set(runners) | {waiter}, return_when=asyncio.FIRST_COMPLETED,
        )
    finally:
        for s in services:
            try:
                s.stop()
            except Exception:
                pass
        for r in runners:
            r.cancel()
        for r in runners:
            try:
                await r
            except (asyncio.CancelledError, Exception):
                pass
        await close_client()


# ----- refresh ----------------------------------------------------------

@click.command("anticopy-refresh")
@_verbosity_opt
def refresh_main(verbose: int):
    """CEAC refresh container: daily rollout pool refresh + per-minute
    verdict backfill side by side."""
    setup_logging(verbose, component="anticopy_refresh")
    from affine.src.anticopy.refresh import RolloutRefreshService
    from affine.src.anticopy.verdict import VerdictBackfillService

    def _factory():
        return [RolloutRefreshService(), VerdictBackfillService()]

    try:
        asyncio.run(_bootstrap_run(_factory))
    except KeyboardInterrupt:
        logger.info("[anticopy.refresh] interrupted")


# ----- worker -----------------------------------------------------------


def _maybe_start_ssh_tunnel() -> "subprocess.Popen | None":
    """When ``ANTICOPY_REMOTE_SSH_HOST`` is set and
    ``ANTICOPY_SGLANG_URL`` points at localhost, establish an SSH
    local-port forward so the worker's HTTP calls reach the GPU
    host's sglang. Returns the ssh subprocess (caller terminates on
    shutdown), or ``None`` if no tunnel is needed (e.g. worker is
    running on the same host as sglang).

    Production deployment runs the worker container on the backend
    CPU host with prod credentials; sglang lives on a separate GPU
    host that **has no AWS keys** — the SSH tunnel is what bridges
    them without putting credentials on the GPU side.
    """
    host = os.getenv("ANTICOPY_REMOTE_SSH_HOST", "")
    key_path = os.getenv("ANTICOPY_REMOTE_SSH_KEY", "")
    sglang_url = os.getenv("ANTICOPY_SGLANG_URL", "")
    if not host or not sglang_url or not key_path:
        return None
    parsed = urlparse(sglang_url)
    if parsed.hostname not in ("localhost", "127.0.0.1"):
        return None
    local_port = parsed.port or 33000
    remote_port = int(os.getenv("ANTICOPY_REMOTE_SGLANG_PORT", "30000"))

    ssh_port = os.getenv("ANTICOPY_REMOTE_SSH_PORT", "")
    cmd = [
        "ssh", "-N", "-T",
        "-i", key_path,
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=30",
        "-o", "ServerAliveInterval=30",
        "-o", "ExitOnForwardFailure=yes",
        "-L", f"{local_port}:127.0.0.1:{remote_port}",
    ]
    if ssh_port:
        cmd += ["-p", ssh_port]
    cmd.append(host)
    logger.info(
        f"[anticopy.worker] starting ssh tunnel "
        f"localhost:{local_port} → {host}:{remote_port}"
    )
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Block until the port answers (or the tunnel dies).
    deadline = time.time() + 30
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"ssh tunnel exited with code {proc.returncode}"
            )
        try:
            import socket
            with socket.create_connection(("127.0.0.1", local_port), timeout=2):
                logger.info(
                    f"[anticopy.worker] ssh tunnel ready on localhost:{local_port}"
                )
                return proc
        except OSError:
            pass
        time.sleep(1)
    proc.terminate()
    raise RuntimeError(f"ssh tunnel never came up after 30s")


@click.command("anticopy-worker")
@_verbosity_opt
def worker_main(verbose: int):
    """CEAC forward worker (CPU-side; drives a remote sglang via SSH)."""
    setup_logging(verbose, component="anticopy_worker")
    from affine.src.anticopy.worker import ForwardWorker

    tunnel = None
    try:
        tunnel = _maybe_start_ssh_tunnel()
    except Exception as e:
        logger.error(f"[anticopy.worker] ssh tunnel failed: {e}")
        # don't crash — worker may be co-located with sglang
        # (no tunnel needed) and will still operate; otherwise the
        # /model_info poll will surface the connection failure.

    def _factory():
        return ForwardWorker()

    try:
        asyncio.run(_bootstrap_run(_factory))
    except KeyboardInterrupt:
        logger.info("[anticopy.worker] interrupted")
    finally:
        if tunnel is not None:
            tunnel.terminate()
            try:
                tunnel.wait(timeout=5)
            except subprocess.TimeoutExpired:
                tunnel.kill()
