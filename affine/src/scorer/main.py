"""
Scorer Service - Main Entry Point

Runs the Scorer as an independent service or one-time execution.
Calculates miner weights using the four-stage scoring algorithm.
"""

import os
import asyncio
import click
import time

from affine.core.setup import logger
from affine.database import init_client, close_client
from affine.database.dao.score_snapshots import ScoreSnapshotsDAO
from affine.database.dao.scores import ScoresDAO
from affine.database.dao.miner_stats import MinerStatsDAO
from affine.src.scorer.scorer import Scorer
from affine.src.scorer.config import ScorerConfig
from affine.utils.subtensor import get_subtensor
from affine.utils.api_client import cli_api_client


async def fetch_scoring_data(api_client, range_type: str = "scoring") -> dict:
    """Fetch scoring data from API with default timeout.
    
    Args:
        api_client: APIClient instance
        range_type: Type of range to use ('scoring' or 'sampling', default: 'scoring')
    """
    logger.info(f"Fetching scoring data from API (range_type={range_type})...")
    data = await api_client.get(f"/samples/scoring?range_type={range_type}")
    
    # Check for API error response
    if isinstance(data, dict) and "success" in data and data.get("success") is False:
        error_msg = data.get("error", "Unknown API error")
        status_code = data.get("status_code", "unknown")
        logger.error(f"API returned error response: {error_msg} (status: {status_code})")
        raise RuntimeError(f"Failed to fetch scoring data: {error_msg}")
    
    return data


async def fetch_system_config(api_client, range_type: str = "scoring") -> dict:
    """Fetch system configuration from API.
    
    Args:
        api_client: APIClient instance
        range_type: Type of range to use ('scoring' or 'sampling', default: 'scoring')
    
    Returns:
        System config dict with:
        - 'environments': list of enabled environment names
        - 'env_configs': dict mapping env_name -> env_config (including min_completeness)
    """
    try:
        config = await api_client.get("/config/environments")
        
        if isinstance(config, dict):
            value = config.get("param_value")
            if isinstance(value, dict):
                # Filter environments based on range_type
                enabled_envs = []
                env_configs = {}
                
                if range_type == "sampling":
                    # Use enabled_for_sampling flag
                    for env_name, env_config in value.items():
                        if isinstance(env_config, dict) and env_config.get("enabled_for_sampling", False):
                            enabled_envs.append(env_name)
                            env_configs[env_name] = env_config
                    logger.info(f"Fetched sampling environments from API: {enabled_envs}")
                else:
                    # Use enabled_for_scoring flag (default)
                    for env_name, env_config in value.items():
                        if isinstance(env_config, dict) and env_config.get("enabled_for_scoring", False):
                            enabled_envs.append(env_name)
                            env_configs[env_name] = env_config
                    logger.info(f"Fetched scoring environments from API: {enabled_envs}")
                
                if enabled_envs:
                    return {
                        "environments": enabled_envs,
                        "env_configs": env_configs
                    }

        logger.exception("Failed to parse environments config")
                
    except Exception as e:
        logger.error(f"Error fetching system config: {e}")
        raise



async def run_scoring_once(save_to_db: bool, range_type: str = "scoring"):
    """Run scoring calculation once.
    
    Uses CLI context manager for automatic cleanup in both one-time
    and service modes (performance is not critical for scorer).
    
    Args:
        save_to_db: Whether to save results to database
        range_type: Type of range to use ('scoring' or 'sampling', default: 'scoring')
    """
    start_time = time.time()
    
    # Use default config (constants)
    config = ScorerConfig()
    scorer = Scorer(config)
    
    # Always use CLI context manager for automatic cleanup
    async with cli_api_client() as api_client:
        # Fetch data
        logger.info("Fetching data from API...")
        scoring_data = await fetch_scoring_data(api_client, range_type=range_type)
        system_config = await fetch_system_config(api_client, range_type=range_type)
        
        # Extract environments and env_configs
        environments = system_config.get("environments")
        env_configs = system_config.get("env_configs", {})
        logger.info(f"environments: {environments}")
        
        # Get current block number from Bittensor
        logger.info("Fetching current block number from Bittensor...")
        subtensor = await get_subtensor()
        block_number = await subtensor.get_current_block()
        logger.info(f"Current block number: {block_number}")
        
        # Read ELO ratings from MINER_STATS (authoritative source)
        # Always read ratings (even in dry-run) so ELO ranking is meaningful
        logger.info("Loading ELO ratings from MINER_STATS...")
        miner_stats_dao = MinerStatsDAO()
        prev_ratings = {}
        try:
            for composite_key, miner_info in scoring_data.items():
                hotkey = miner_info.get('hotkey', '')
                revision = miner_info.get('model_revision', '')
                if not hotkey or not revision:
                    continue
                stats = await miner_stats_dao.get_miner_stats(hotkey, revision)
                if stats:
                    prev_ratings[hotkey] = {
                        'elo_rating': stats.get('elo_rating') or None,
                        'elo_rounds_played': stats.get('elo_rounds_played') or 0,
                        'elo_model_submit_block': stats.get('elo_model_submit_block') or None,
                    }
        except Exception as e:
            logger.warning(f"Failed to load ELO ratings from MINER_STATS: {e}")
            prev_ratings = {}
        logger.info(f"Loaded ELO ratings for {len(prev_ratings)} miners")

        # Calculate scores
        logger.info("Starting scoring calculation...")
        result = scorer.calculate_scores(
            scoring_data=scoring_data,
            environments=environments,
            env_configs=env_configs,
            block_number=block_number,
            prev_ratings=prev_ratings if prev_ratings else None,
            print_summary=True
        )
        
        # Save to database if requested
        if save_to_db:
            logger.info("Saving results to database...")
            score_snapshots_dao = ScoreSnapshotsDAO()
            scores_dao = ScoresDAO()
            
            await scorer.save_results(
                result=result,
                score_snapshots_dao=score_snapshots_dao,
                scores_dao=scores_dao,
                miner_stats_dao=miner_stats_dao
            )
            logger.info("Results saved successfully")
        
        elapsed = time.time() - start_time
        logger.info(f"Scoring completed in {elapsed:.2f}s")
        
        # Print summary
        summary = result.get_summary()
        logger.info(f"Summary: {summary}")
        
        return result


async def run_service_with_mode(save_to_db: bool, service_mode: bool, interval_minutes: int, range_type: str = "scoring"):
    """Run the scorer service.
    
    Args:
        save_to_db: Whether to save results to database
        service_mode: If True, run continuously; if False, run once and exit
        interval_minutes: Minutes between scoring runs in service mode
        range_type: Type of range to use ('scoring' or 'sampling', default: 'scoring')
    """
    logger.info("Starting Scorer Service")
    logger.info(f"Range type: {range_type}")
    
    # Initialize database if saving results
    if save_to_db:
        try:
            await init_client()
            logger.info("Database client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    try:
        if not service_mode:
            # Run once and exit (DEFAULT)
            logger.info("Running in one-time mode (default)")
            await run_scoring_once(save_to_db, range_type=range_type)
        else:
            # Run continuously with configured interval
            logger.info(f"Running in service mode (continuous, every {interval_minutes} minutes)")
            while True:
                try:
                    await run_scoring_once(save_to_db, range_type=range_type)
                    logger.info(f"Waiting {interval_minutes} minutes until next run...")
                    await asyncio.sleep(interval_minutes * 60)
                except Exception as e:
                    logger.error(f"Error in scoring cycle: {e}", exc_info=True)
                    logger.info(f"Waiting {interval_minutes} minutes before retry...")
                    await asyncio.sleep(interval_minutes * 60)
        
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Error running Scorer: {e}", exc_info=True)
        raise
    finally:
        # Cleanup
        if save_to_db:
            try:
                await close_client()
                logger.info("Database client closed")
            except Exception as e:
                logger.error(f"Error closing database: {e}")
    
    logger.info("Scorer Service completed successfully")


@click.command()
@click.option(
    "--sampling",
    is_flag=True,
    default=False,
    help="Use sampling environments instead of scoring environments"
)
def main(sampling: bool):
    """
    Affine Scorer - Calculate miner weights using four-stage algorithm.
    
    This service fetches scoring data from the API and calculates normalized
    weights for miners using a four-stage algorithm with Pareto filtering.
    
    By default, uses environments with enabled_for_scoring=true.
    With --sampling flag, uses environments with enabled_for_sampling=true.
    
    Run Mode:
    - Default: One-time execution (calculates scores once and exits)
    - SERVICE_MODE=true: Continuous service mode (runs at configured interval)
    
    Configuration:
    - SCORER_SAVE_TO_DB: Enable database saving (default: false)
    - SERVICE_MODE: Run as continuous service (default: false)
    - SCORER_INTERVAL_MINUTES: Minutes between runs in service mode (default: 30)
    - All scoring parameters are constants in config.py
    
    Examples:
        af -v servers scorer                # Use scoring environments
        af -v servers scorer --sampling     # Use sampling environments
    """
    # Determine range type from flag
    range_type = "sampling" if sampling else "scoring"
    
    # Check if should save to database
    save_to_db = os.getenv("SCORER_SAVE_TO_DB", "false").lower() in ("true", "1", "yes")
    
    # Check service mode (default: false = one-time execution)
    service_mode = os.getenv("SERVICE_MODE", "false").lower() in ("true", "1", "yes")
    
    # Get interval in minutes (default: 10 minutes)
    try:
        interval_minutes = int(os.getenv("SCORER_INTERVAL_MINUTES", "30"))
        if interval_minutes <= 0:
            logger.warning(f"Invalid SCORER_INTERVAL_MINUTES={interval_minutes}, using default 30")
            interval_minutes = 30
    except ValueError:
        logger.warning(f"Invalid SCORER_INTERVAL_MINUTES value, using default 30")
        interval_minutes = 30
    
    if save_to_db:
        logger.info("Database saving enabled (SCORER_SAVE_TO_DB=true)")
    logger.info(f"Service mode: {service_mode}")
    if service_mode:
        logger.info(f"Interval: {interval_minutes} minutes")
    
    # Run service
    asyncio.run(run_service_with_mode(
        save_to_db=save_to_db,
        service_mode=service_mode,
        interval_minutes=interval_minutes,
        range_type=range_type
    ))


if __name__ == "__main__":
    main()