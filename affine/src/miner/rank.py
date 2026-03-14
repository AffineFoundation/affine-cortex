"""
Rank Display Module

Fetches and displays miner ranking information from the API,
using the same format as scorer's print_summary.
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from affine.utils.api_client import cli_api_client
from affine.core.setup import logger
from affine.core.miners import miners


async def fetch_latest_scores(client) -> Dict[str, Any]:
    """Fetch latest scores from API.
    
    Args:
        client: APIClient instance
    
    Returns:
        Dict with block_number, calculated_at, and scores list
    """
    logger.debug("Fetching latest scores from API...")
    data = await client.get("/scores/latest?top=256")
    
    if isinstance(data, dict) and "success" in data and data.get("success") is False:
        error_msg = data.get("error", "Unknown API error")
        status_code = data.get("status_code", "unknown")
        logger.error(f"API returned error response: {error_msg} (status: {status_code})")
        raise RuntimeError(f"Failed to fetch scores: {error_msg}")
    
    return data


async def fetch_miner_scores_at_block(client, block_number: int) -> Dict[int, Dict[str, Any]]:
    """Fetch detailed miner scores for a specific block via API.
    
    Args:
        client: APIClient instance
        block_number: Block number to query
        
    Returns:
        Dict mapping UID to detailed score data
    """
    logger.debug(f"Fetching detailed scores for block {block_number}...")
    
    # Use the miner_scores endpoint to get detailed scoring data
    # This endpoint should return all the data needed for print_summary
    data = await client.get(f"/scores/latest?top=256")
    
    return data


async def fetch_environments(client) -> Tuple[List[str], Dict[str, Any]]:
    """Fetch enabled environments from system config.
    
    Args:
        client: APIClient instance
    
    Returns:
        Tuple of (list of environment names enabled for scoring, dict mapping env_name -> env_config)
    """
    try:
        config = await client.get("/config/environments")
        
        if isinstance(config, dict):
            value = config.get("param_value")
            if isinstance(value, dict):
                # Filter environments where enabled_for_scoring=true
                enabled_envs = []
                env_configs = {}
                
                for env_name, env_config in value.items():
                    if isinstance(env_config, dict) and env_config.get("enabled_for_scoring", False):
                        enabled_envs.append(env_name)
                        env_configs[env_name] = env_config
                
                if enabled_envs:
                    logger.debug(f"Fetched environments from API: {enabled_envs}")
                    return sorted(enabled_envs), env_configs
        
        logger.warning("Failed to parse environments config")
        return [], {}
                
    except Exception as e:
        logger.error(f"Error fetching environments: {e}")
        return [], {}


async def fetch_scorer_config(client) -> dict:
    """Fetch scorer configuration from latest snapshot.
    
    Args:
        client: APIClient instance
    
    Returns:
        Dict with scoring configuration parameters
    """
    try:
        weights_data = await client.get("/scores/weights/latest")
        
        if isinstance(weights_data, dict):
            config = weights_data.get("config", {})
            if config:
                logger.debug(f"Fetched scorer config from API: {config}")
                return config
        
        logger.warning("Failed to fetch scorer config, using defaults")
        return {}
                
    except Exception as e:
        logger.error(f"Error fetching scorer config: {e}")
        return {}


def _get_filter_reason_from_api(score: dict, environments: list, env_configs: dict, scorer_config: dict) -> str:
    """Get concise filter reason from API score data (max 12 chars)."""
    filter_info = score.get("filter_info") or {}
    filter_reasons = filter_info.get("filter_reasons", {})
    filtered_subsets = filter_info.get("filtered_subsets", [])

    # Pareto dominated
    if filtered_subsets:
        for reason in filter_reasons.values():
            if isinstance(reason, str) and reason.startswith("dom>"):
                return reason  # e.g. "dom>123"
        return "pareto"

    # Incomplete env
    default_min_completeness = scorer_config.get("min_completeness", 0.9)
    scores_by_env = score.get("scores_by_env", {})
    for env in environments:
        if env in scores_by_env:
            env_data = scores_by_env[env]
            completeness = env_data.get("completeness", 1.0)
            env_config = env_configs.get(env, {})
            env_min = env_config.get("min_completeness", default_min_completeness)
            if completeness < env_min:
                env_short = env.split(':')[-1][:10]
                return f"!{env_short}"
        else:
            env_short = env.split(':')[-1][:10]
            return f"!{env_short}"

    # No valid data
    total_samples = score.get("total_samples", 0)
    if total_samples == 0:
        return "no_data"
    return ""


async def print_rank_table():
    """Fetch and print miner ranking table in scorer format.

    This function replicates the output format of scorer's print_detailed_table,
    but fetches data from the API instead of calculating from raw samples.
    """
    async with cli_api_client() as client:
        scores_data = await fetch_latest_scores(client)
        environments, env_configs = await fetch_environments(client)
        scorer_config = await fetch_scorer_config(client)

        if not scores_data or not scores_data.get('block_number'):
            print("No scores found")
            return

        block_number = scores_data.get("block_number")
        calculated_at = scores_data.get("calculated_at")
        scores_list = scores_data.get("scores", [])

        if not scores_list:
            print(f"No miners scored at block {block_number}")
            return

        # Sort: weight>0 first (by rating desc), then filtered (by rating desc)
        scores_list = sorted(
            scores_list,
            key=lambda s: (
                (s.get("overall_score") or 0) > 0,
                s.get("elo_rating") or 0,
            ),
            reverse=True
        )

        # Build header
        header_parts = ["Hotkey  ", " UID", "Model                    "]

        for env in environments:
            env_cfg = env_configs.get(env, {})
            if isinstance(env_cfg, dict) and env_cfg.get('display_name'):
                env_display = env_cfg['display_name']
            elif ':' in env:
                env_display = env.split(':', 1)[1]
            else:
                env_display = env
            header_parts.append(f"{env_display:>14}")

        header_parts.extend(["Rating", "   Δ", " Rnd", "  Weight  ", "Status      "])

        header_line = " | ".join(header_parts)
        table_width = len(header_line)

        print("=" * table_width, flush=True)
        print(f"MINER RANKING TABLE - Block {block_number}", flush=True)
        print("=" * table_width, flush=True)
        print(header_line, flush=True)
        print("-" * table_width, flush=True)

        for score in scores_list:
            uid = score.get("uid")
            hotkey = score.get("miner_hotkey")
            model = score.get("model")
            overall_score = score.get("overall_score") or 0
            scores_by_env = score.get("scores_by_env", {})
            elo_rating = int(score.get("elo_rating") or 0)
            elo_rounds_played = int(score.get("elo_rounds_played") or 0)
            elo_rating_change = int(score.get("elo_rating_change") or 0)
            is_active = overall_score > 0

            model_display = model[:25]
            filter_reason = _get_filter_reason_from_api(score, environments, env_configs, scorer_config)

            row_parts = [
                f"{hotkey[:8]:8s}",
                f"{uid:4d}",
                f"{model_display:25s}",
            ]

            # Environment scores
            default_min_completeness = scorer_config.get("min_completeness", 0.9)
            for env in environments:
                if env in scores_by_env:
                    env_data = scores_by_env[env]
                    env_score = env_data.get("score", 0.0)
                    sample_count = env_data.get("sample_count", 0)
                    completeness = env_data.get("completeness", 1.0)
                    score_percent = env_score * 100

                    env_config = env_configs.get(env, {})
                    env_min_completeness = env_config.get("min_completeness", default_min_completeness)
                    is_insufficient = completeness < env_min_completeness

                    if is_insufficient:
                        score_str = f"{score_percent:.2f}/{sample_count}!"
                    else:
                        score_str = f"{score_percent:.2f}/{sample_count}"
                    row_parts.append(f"{score_str:>14}")
                else:
                    row_parts.append(f"{'  -  ':>14}")

            if is_active:
                delta_str = f"+{elo_rating_change}" if elo_rating_change >= 0 else str(elo_rating_change)
                row_parts.append(f"{elo_rating:6d}")
                row_parts.append(f"{delta_str:>6s}")
                row_parts.append(f"{elo_rounds_played:4d}")
                row_parts.append(f"{overall_score:>10.6f}")
                row_parts.append(f"{'✓':12s}")
            else:
                row_parts.append(f"{'—':>6}")
                row_parts.append(f"{'—':>6}")
                row_parts.append(f"{'—':>4}")
                row_parts.append(f"{'0':>10}")
                row_parts.append(f"{filter_reason or '✗':12s}")

            print(" | ".join(row_parts), flush=True)

        print("=" * table_width, flush=True)
        print(f"Total miners: {len(scores_list)}", flush=True)
        non_zero = len([s for s in scores_list if (s.get("overall_score") or 0) > 0])
        print(f"Active miners (weight > 0): {non_zero}", flush=True)
        print("=" * table_width, flush=True)


async def get_rank_command():
    """Command handler for get-rank.
    
    Fetches score snapshot from API and displays ranking table.
    """
    try:
        await print_rank_table()
    except Exception as e:
        logger.error(f"Failed to fetch and display ranks: {e}", exc_info=True)
        print(f"Error: {e}")