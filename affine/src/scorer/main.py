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
from affine.database.dao.openskill_ratings import OpenSkillRatingsDAO
from affine.database.dao.openskill_matches import OpenSkillMatchesDAO
from affine.src.scorer.scorer import Scorer
from affine.src.scorer.config import ScorerConfig
from affine.src.scorer.openskill_config import OpenSkillConfig
from affine.src.scorer.openskill_scorer import OpenSkillScorer
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
                        'elo_last_scored_at': stats.get('elo_last_scored_at') or None,
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
                miner_stats_dao=miner_stats_dao,
                prev_ratings=prev_ratings,
            )
            logger.info("Results saved successfully")
        
        elapsed = time.time() - start_time
        logger.info(f"ELO scoring completed in {elapsed:.2f}s")

        # Print summary
        summary = result.get_summary()
        logger.info(f"Summary: {summary}")

        # Run OpenSkill shadow scoring alongside ELO
        try:
            logger.info("Running OpenSkill shadow scoring...")
            await run_openskill_scoring(
                save_to_db=save_to_db,
                range_type=range_type,
                cold_start=False,
                scoring_data=scoring_data,
                system_config=system_config,
            )
        except Exception as e:
            logger.error(f"OpenSkill shadow scoring failed (non-fatal): {e}", exc_info=True)

        return result


async def run_openskill_scoring(
    save_to_db: bool,
    range_type: str = "scoring",
    cold_start: bool = False,
    scoring_data: dict = None,
    system_config: dict = None,
):
    """Run OpenSkill per-task rating and weight computation.

    Args:
        save_to_db: Whether to save results to database
        range_type: 'scoring' or 'sampling'
        cold_start: If True, process all historical data from scratch
    """
    start_time = time.time()
    os_config = OpenSkillConfig
    ratings_dao = OpenSkillRatingsDAO()
    matches_dao = OpenSkillMatchesDAO()
    scorer = OpenSkillScorer(config=os_config, ratings_dao=ratings_dao, matches_dao=matches_dao)

    # Fetch data from API if not provided (standalone mode)
    if scoring_data is None or system_config is None:
        async with cli_api_client() as api_client:
            if scoring_data is None:
                scoring_data = await fetch_scoring_data(api_client, range_type=range_type)
            if system_config is None:
                system_config = await fetch_system_config(api_client, range_type=range_type)

    environments = system_config.get("environments", [])
    env_configs = system_config.get("env_configs", {})
    logger.info(f"OpenSkill environments: {environments}")

    # Build per-env task_scores and miner_task_counts
    from collections import defaultdict
    env_task_scores = defaultdict(lambda: defaultdict(dict))
    miner_task_counts = defaultdict(lambda: defaultdict(int))

    # Timestamp cutoff for cold start
    cutoff_ms = None
    if cold_start:
        cutoff_ms = int((time.time() - os_config.COLD_START_DAYS * 86400) * 1000)
        logger.info(f"OpenSkill cold start: using last {os_config.COLD_START_DAYS} days")

    for composite_key, miner_entry in scoring_data.items():
        hotkey = miner_entry.get('hotkey', '')
        revision = miner_entry.get('model_revision', '')
        if not hotkey or not revision:
            continue
        miner_key = f"{hotkey}#{revision}"
        env_data = miner_entry.get('env', {})

        for env_name in environments:
            env_info = env_data.get(env_name, {})
            if not env_info:
                continue
            all_samples = env_info.get('all_samples', [])
            for s in all_samples:
                tid = s.get('task_id')
                score = s.get('score')
                if tid is None or score is None:
                    continue
                if cutoff_ms is not None:
                    ts = s.get('timestamp') or 0
                    if ts < cutoff_ms:
                        continue
                env_task_scores[env_name][int(tid)][miner_key] = float(score)
                miner_task_counts[miner_key][env_name] += 1

    # Determine which tasks to process
    env_window_sizes = {}
    for env_name in environments:
        sampling_cfg = env_configs.get(env_name, {}).get('sampling_config', {})
        sampling_list = sampling_cfg.get('sampling_list', [])
        env_window_sizes[env_name] = len(sampling_list)

        if not cold_start:
            # Normal mode: only process tasks that have left the window
            current_window = set(sampling_list)
            all_task_ids = set(env_task_scores[env_name].keys())
            rotated_out = all_task_ids - current_window
            env_task_scores[env_name] = {
                tid: scores for tid, scores in env_task_scores[env_name].items()
                if tid in rotated_out
            }
            logger.info(
                f"OpenSkill {env_name}: {len(rotated_out)} rotated-out tasks, "
                f"{len(current_window)} in current window"
            )

    # Process tasks per env
    total_processed = 0
    for env_name in environments:
        tasks = dict(env_task_scores[env_name])
        if tasks:
            count = await scorer.process_rotated_tasks(env_name, tasks)
            total_processed += count

    # Compute weights
    weights = await scorer.compute_weights(
        environments=environments,
        env_window_sizes=env_window_sizes,
        miner_task_counts=dict(miner_task_counts),
    )

    # Print results
    if weights:
        ranked = sorted(weights.items(), key=lambda x: -x[1])
        logger.info(f"OpenSkill final weights ({len(ranked)} miners):")
        for i, (mk, wt) in enumerate(ranked[:15], 1):
            logger.info(f"  #{i} {mk[:20]}... weight={wt:.4f}")
        w = [wt for _, wt in ranked]
        logger.info(
            f"  top1={w[0]*100:.1f}% top5={sum(w[:5])*100:.1f}% "
            f"top10={sum(w[:10])*100:.1f}%"
        )

    # Save weight snapshot for shadow analysis
    if save_to_db and weights:
        await scorer.save_weight_snapshot(weights, environments)

    elapsed = time.time() - start_time
    logger.info(f"OpenSkill scoring completed in {elapsed:.2f}s, "
                 f"{total_processed} tasks processed")

    return weights


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
@click.option(
    "--openskill",
    is_flag=True,
    default=False,
    help="Run OpenSkill per-task rating instead of ELO"
)
@click.option(
    "--cold-start",
    is_flag=True,
    default=False,
    help="(OpenSkill) Process all historical tasks from scratch"
)
def main(sampling: bool, openskill: bool, cold_start: bool):
    """
    Affine Scorer - Calculate miner weights.

    By default runs the four-stage ELO algorithm.
    With --openskill, runs OpenSkill per-task rating system.

    Examples:
        af -v servers scorer                        # ELO scoring
        af -v servers scorer --openskill             # OpenSkill (incremental)
        af -v servers scorer --openskill --cold-start # OpenSkill (bootstrap)
    """
    # Determine range type from flag
    range_type = "sampling" if sampling else "scoring"

    # Check if should save to database
    save_to_db = os.getenv("SCORER_SAVE_TO_DB", "false").lower() in ("true", "1", "yes")

    # Check service mode (default: false = one-time execution)
    service_mode = os.getenv("SERVICE_MODE", "false").lower() in ("true", "1", "yes")

    # Get interval in minutes (default: 30 minutes)
    try:
        interval_minutes = int(os.getenv("SCORER_INTERVAL_MINUTES", "60"))
        if interval_minutes <= 0:
            interval_minutes = 60
    except ValueError:
        interval_minutes = 60

    if save_to_db:
        logger.info("Database saving enabled (SCORER_SAVE_TO_DB=true)")

    if openskill:
        # OpenSkill mode
        logger.info(f"Running OpenSkill scorer (cold_start={cold_start})")

        async def _run():
            if save_to_db:
                await init_client()
            try:
                await run_openskill_scoring(
                    save_to_db=save_to_db,
                    range_type=range_type,
                    cold_start=cold_start,
                )
            finally:
                if save_to_db:
                    await close_client()

        asyncio.run(_run())
    else:
        # ELO mode (default)
        logger.info(f"Service mode: {service_mode}")
        if service_mode:
            logger.info(f"Interval: {interval_minutes} minutes")

        asyncio.run(run_service_with_mode(
            save_to_db=save_to_db,
            service_mode=service_mode,
            interval_minutes=interval_minutes,
            range_type=range_type
        ))


if __name__ == "__main__":
    main()