"""
Scorer Service - Main Entry Point

Runs the Scorer as an independent service or one-time execution.
Calculates miner weights using the champion challenge scoring algorithm.
"""

import os
import asyncio
import click
import time

from affine.core.setup import logger
from affine.database import init_client, close_client
from affine.database.dao.score_snapshots import ScoreSnapshotsDAO
from affine.database.dao.scores import ScoresDAO
from affine.database.dao.anti_copy import AntiCopyDAO
from affine.database.dao.miner_stats import MinerStatsDAO
from affine.database.dao.system_config import SystemConfigDAO
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
        - 'env_configs': dict mapping env_name -> env_config
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
                    for env_name, env_config in value.items():
                        if isinstance(env_config, dict) and env_config.get("enabled_for_sampling", False):
                            enabled_envs.append(env_name)
                            env_configs[env_name] = env_config
                    logger.info(f"Fetched sampling environments from API: {enabled_envs}")
                else:
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

    Args:
        save_to_db: Whether to save results to database
        range_type: Type of range to use ('scoring' or 'sampling', default: 'scoring')
    """
    start_time = time.time()

    config = ScorerConfig()
    scorer = Scorer(config)
    anticopy_dao = AntiCopyDAO()

    async with cli_api_client() as api_client:
        # Fetch data
        logger.info("Fetching data from API...")
        scoring_data = await fetch_scoring_data(api_client, range_type=range_type)
        system_config = await fetch_system_config(api_client, range_type=range_type)

        environments = system_config.get("environments")
        env_configs = system_config.get("env_configs", {})
        logger.info(f"environments: {environments}")

        # Get current block number from Bittensor
        logger.info("Fetching current block number from Bittensor...")
        subtensor = await get_subtensor()
        block_number = await subtensor.get_current_block()
        logger.info(f"Current block number: {block_number}")

        # Load champion state and challenge states from DB
        config_dao = SystemConfigDAO()
        miner_stats_dao = MinerStatsDAO()

        logger.info("Loading champion state...")
        champion_state = await config_dao.get_param_value('champion', default=None)

        if champion_state:
            logger.info(
                f"Current champion: {champion_state.get('hotkey', '?')[:8]}... "
                f"(UID {champion_state.get('uid')})"
            )
        else:
            # No champion in DB → cold start: geometric mean selection.
            # Use `af db set-champion` before the first run to pre-seed.
            logger.warning(
                "No champion found in system_config. "
                "Cold start will select champion by geometric mean. "
                "Use `af db set-champion` to pre-seed if needed."
            )

        logger.info("Loading challenge states from miner_stats...")
        prev_challenge_states = {}
        for composite_key, miner_info in scoring_data.items():
            hotkey = miner_info.get('hotkey', '')
            revision = miner_info.get('model_revision', '')
            if not hotkey or not revision:
                continue
            try:
                state = await miner_stats_dao.get_challenge_state(hotkey, revision)
                state['revision'] = revision
                prev_challenge_states[hotkey] = state
            except Exception as e:
                logger.warning(f"Failed to load challenge state for {hotkey[:8]}...: {e}")
                # Default: fresh state for this miner only, others unaffected
        logger.info(f"Loaded challenge states for {len(prev_challenge_states)} miners")

        anticopy_records = {}
        for _, miner_info in scoring_data.items():
            model = miner_info.get('model_repo', '')
            revision = miner_info.get('model_revision', '')
            if not model or not revision:
                continue
            try:
                ac = await anticopy_dao.get_latest(model, revision)
                if ac:
                    copy_of = ac.get("copy_of", [])
                    orig = copy_of[0] if copy_of else {}
                    anticopy_records[f"{model}#{revision}"] = {
                        "status": ac.get("status")
                        or ("cheat" if ac.get("is_copy") else "clean"),
                        "copy_of_uid": orig.get("uid"),
                    }
            except Exception as e:
                logger.warning(f"Failed to load anti-copy status for {model}@{revision[:8]}: {e}")
        logger.info(f"Loaded anti-copy records for {len(anticopy_records)} miners")

        # Extract sampling_count per environment for checkpoint calculation
        env_sampling_counts = {}
        for env_name, ec in env_configs.items():
            sc = ec.get('sampling_config', {})
            if sc.get('sampling_count'):
                env_sampling_counts[env_name] = sc['sampling_count']
        logger.info(f"env_sampling_counts: {env_sampling_counts}")

        # Calculate scores
        logger.info("Starting scoring calculation...")
        result = scorer.calculate_scores(
            scoring_data=scoring_data,
            environments=environments,
            block_number=block_number,
            env_sampling_counts=env_sampling_counts,
            champion_state=champion_state,
            prev_challenge_states=prev_challenge_states,
            anticopy_records=anticopy_records,
            print_summary=True,
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
                system_config_dao=config_dao,
                block_number=block_number,
            )
            logger.info("Results saved successfully")

        elapsed = time.time() - start_time
        logger.info(f"Scoring completed in {elapsed:.2f}s")

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

    if save_to_db:
        try:
            await init_client()
            logger.info("Database client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    try:
        if not service_mode:
            logger.info("Running in one-time mode (default)")
            await run_scoring_once(save_to_db, range_type=range_type)
        else:
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
    Affine Scorer - Calculate miner weights using champion challenge system.

    Examples:
        af -v servers scorer                        # Champion challenge scoring
        af -v servers scorer --sampling             # Use sampling environments
    """
    range_type = "sampling" if sampling else "scoring"

    save_to_db = os.getenv("SCORER_SAVE_TO_DB", "false").lower() in ("true", "1", "yes")
    service_mode = os.getenv("SERVICE_MODE", "false").lower() in ("true", "1", "yes")

    try:
        interval_minutes = int(os.getenv("SCORER_INTERVAL_MINUTES", "30"))
        if interval_minutes <= 0:
            interval_minutes = 30
    except ValueError:
        interval_minutes = 30

    if save_to_db:
        logger.info("Database saving enabled (SCORER_SAVE_TO_DB=true)")

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
