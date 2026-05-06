"""Targon deployer service entry point.

Runs as a long-lived container (see compose/docker-compose.backend.yml).
"""

import asyncio
import signal

import click

from affine.core.setup import logger, setup_logging
from affine.database import close_client, init_client

from .service import TargonDeployerService


async def run_service():
    logger.info("Starting Targon Deployer Service")
    await init_client()

    service = TargonDeployerService()
    stop_event = asyncio.Event()

    def handle_shutdown(sig):
        logger.info(f"Received signal {sig}, initiating shutdown...")
        service.stop()
        stop_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: handle_shutdown(s))

    task = asyncio.create_task(service.run())
    try:
        await stop_event.wait()
    finally:
        service.stop()
        try:
            await asyncio.wait_for(task, timeout=10)
        except asyncio.TimeoutError:
            task.cancel()
        await close_client()

    logger.info("Targon Deployer Service shut down")


@click.command()
@click.option(
    "-v", "--verbosity",
    default=None,
    type=click.Choice(["0", "1", "2", "3"]),
    help="Logging verbosity: 0=CRITICAL, 1=INFO, 2=DEBUG, 3=TRACE",
)
def main(verbosity):
    """Affine Targon Deployer - reconcile Targon deployments with the champion."""
    if verbosity is not None:
        # Pass component explicitly — auto-detection scans sys.argv for
        # known service names and our entry point isn't on that list, so
        # without this the file handler lands at /var/log/affine/affine/
        # instead of the per-service /var/log/affine/targon_deployer/
        # directory the compose volume mounts.
        setup_logging(int(verbosity), component="targon_deployer")
    asyncio.run(run_service())


if __name__ == "__main__":
    main()
