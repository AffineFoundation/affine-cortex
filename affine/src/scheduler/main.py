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
from affine.database.dao.sample_results import SampleResultsDAO
from affine.database.dao.task_pool import TaskPoolDAO
from .task_generator import TaskGeneratorService
from .scheduler import SchedulerService
from .sampling_scheduler import SamplingScheduler, PerMinerSamplingScheduler


async def run_service(task_interval: int, cleanup_interval: int, max_tasks: int):
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
    scheduler = None
    sampling_scheduler = None
    per_miner_scheduler = None
    try:
        # Create DAOs
        sample_results_dao = SampleResultsDAO()
        task_pool_dao = TaskPoolDAO()
        
        # Create TaskGeneratorService (legacy, for cleanup only)
        task_generator = TaskGeneratorService(
            sample_results_dao=sample_results_dao,
            task_pool_dao=task_pool_dao
        )
        
        # Create and start SchedulerService (legacy, for cleanup only)
        scheduler = SchedulerService(
            task_generator=task_generator,
            task_generation_interval=task_interval,
            cleanup_interval=cleanup_interval,
            max_tasks_per_miner_env=max_tasks
        )
        
        await scheduler.start()
        logger.info(
            f"Legacy SchedulerService started (cleanup_interval={cleanup_interval}s)"
        )
        
        # Create and start SamplingScheduler (rotation only)
        sampling_scheduler = SamplingScheduler()
        await sampling_scheduler.start()
        logger.info("SamplingScheduler started for sampling list rotation")
        
        # Create and start PerMinerSamplingScheduler (new architecture)
        per_miner_scheduler = PerMinerSamplingScheduler(
            default_concurrency=5,
            scheduling_interval=10
        )
        await per_miner_scheduler.start()
        logger.info("PerMinerSamplingScheduler started for per-miner task generation")
        
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
        
        if scheduler:
            try:
                await scheduler.stop()
                logger.info("Legacy SchedulerService stopped")
            except Exception as e:
                logger.error(f"Error stopping SchedulerService: {e}")
        
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
    
    This service periodically generates sampling tasks for all active miners
    and performs cleanup of old tasks.
    """
    # Setup logging if verbosity specified
    if verbosity is not None:
        setup_logging(int(verbosity))
    
    # Override with environment variables if present
    task_interval = int(os.getenv("SCHEDULER_TASK_GENERATION_INTERVAL", "600"))
    cleanup_interval = int(os.getenv("SCHEDULER_CLEANUP_INTERVAL", "300"))
    max_tasks = int(os.getenv("SCHEDULER_MAX_TASKS_PER_MINER_ENV", "300"))

    # Run service
    asyncio.run(run_service(
        task_interval=task_interval,
        cleanup_interval=cleanup_interval,
        max_tasks=max_tasks
    ))


if __name__ == "__main__":
    main()