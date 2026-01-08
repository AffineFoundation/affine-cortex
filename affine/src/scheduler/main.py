"""
Task Scheduler Service - Main Entry Point

Runs the TaskScheduler as an independent background service.
This service generates sampling tasks for all miners periodically.
"""

import os
import asyncio
import signal
import click
from affine.core.setup import setup_logging, logger
from affine.database import init_client, close_client
from affine.database.dao.task_pool import TaskPoolDAO
from .sampling_scheduler import SamplingScheduler, PerMinerSamplingScheduler


async def run_service(cleanup_interval: int):
    """Run the task scheduler service."""
    logger.info("Starting Task Scheduler Service")
    
    # Initialize database
    try:
        await init_client()
        logger.info("Database client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise
    
    # Setup signal handlers
    shutdown_event = asyncio.Event()
    
    def handle_shutdown(sig):
        logger.info(f"Received signal {sig}, initiating shutdown...")
        shutdown_event.set()
    
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: handle_shutdown(s))
    
    # Initialize schedulers
    sampling_scheduler = None
    per_miner_scheduler = None
    try:
        # Create and start SamplingScheduler (rotation only)
        sampling_scheduler = SamplingScheduler()
        await sampling_scheduler.start()
        logger.info("SamplingScheduler started for sampling list rotation")
        
        # Create and start PerMinerSamplingScheduler (task generation + all cleanup)
        per_miner_scheduler = PerMinerSamplingScheduler(
            default_concurrency=6,
            scheduling_interval=10
        )
        await per_miner_scheduler.start()
        logger.info("PerMinerSamplingScheduler started (task generation + cleanup)")
        
        # Wait for shutdown signal
        await shutdown_event.wait()
        
    except Exception as e:
        logger.error(f"Error running TaskScheduler: {e}", exc_info=True)
        raise
    finally:
        # Cleanup
        if per_miner_scheduler:
            try:
                await per_miner_scheduler.stop()
                logger.info("PerMinerSamplingScheduler stopped")
            except Exception as e:
                logger.error(f"Error stopping PerMinerSamplingScheduler: {e}")
        
        if sampling_scheduler:
            try:
                await sampling_scheduler.stop()
                logger.info("SamplingScheduler stopped")
            except Exception as e:
                logger.error(f"Error stopping SamplingScheduler: {e}")
        
        try:
            await close_client()
            logger.info("Database client closed")
        except Exception as e:
            logger.error(f"Error closing database: {e}")
    
    logger.info("Task Scheduler Service shut down successfully")


@click.command()
@click.option(
    "-v", "--verbosity",
    default=None,
    type=click.Choice(["0", "1", "2", "3"]),
    help="Logging verbosity: 0=CRITICAL, 1=INFO, 2=DEBUG, 3=TRACE"
)
def main(verbosity):
    """
    Affine Task Scheduler - Generate sampling tasks for miners.
    
    This service uses the new per-miner sampling scheduler architecture
    with global concurrency control.
    """
    # Setup logging if verbosity specified
    if verbosity is not None:
        setup_logging(int(verbosity))
    
    # Cleanup interval for expired paused tasks
    cleanup_interval = int(os.getenv("SCHEDULER_CLEANUP_INTERVAL", "300"))

    # Run service
    asyncio.run(run_service(cleanup_interval=cleanup_interval))


if __name__ == "__main__":
    main()