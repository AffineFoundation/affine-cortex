"""
Miners Monitor Service - Main Entry Point

Runs the MinersMonitor as an independent background service.
This service monitors miners from the metagraph and updates the database.
"""

import asyncio
import signal
import click
from affine.core.setup import logger, setup_logging
from affine.database import init_client, close_client
from .live_scores_monitor import LiveScoresMonitor
from .miners_monitor import MinersMonitor

async def run_service():
    """Run the miners monitor service."""
    logger.info("Starting Miners Monitor Service")

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

    # Initialize and start monitors
    monitor = None
    live_scores_monitor = None
    try:
        monitor = await MinersMonitor.initialize()
        logger.info("MinersMonitor started")
        live_scores_monitor = await LiveScoresMonitor.initialize()
        logger.info("LiveScoresMonitor started")

        # Wait for shutdown signal
        await shutdown_event.wait()

    except Exception as e:
        logger.error(f"Error running monitors: {e}", exc_info=True)
        raise
    finally:
        if live_scores_monitor:
            try:
                await live_scores_monitor.stop_background_tasks()
                logger.info("LiveScoresMonitor stopped")
            except Exception as e:
                logger.error(f"Error stopping LiveScoresMonitor: {e}")
        if monitor:
            try:
                await monitor.stop_background_tasks()
                logger.info("MinersMonitor stopped")
            except Exception as e:
                logger.error(f"Error stopping MinersMonitor: {e}")

        try:
            await close_client()
            logger.info("Database client closed")
        except Exception as e:
            logger.error(f"Error closing database: {e}")

    logger.info("Miners Monitor Service shut down successfully")


@click.command()
@click.option(
    "-v", "--verbosity",
    default=None,
    type=click.Choice(["0", "1", "2", "3"]),
    help="Logging verbosity: 0=CRITICAL, 1=INFO, 2=DEBUG, 3=TRACE"
)
def main(verbosity):
    """
    Affine Miners Monitor - Monitor miners from metagraph and update database.
    
    This service continuously monitors the Bittensor metagraph for miner information
    and keeps the database synchronized.
    """
    # Setup logging if verbosity specified
    if verbosity is not None:
        setup_logging(int(verbosity))
    
    # Run service
    asyncio.run(run_service())


if __name__ == "__main__":
    main()