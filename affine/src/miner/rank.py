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


async def print_rank_table():
    """Fetch and print miner ranking table in scorer format.
    
    This function replicates the output format of scorer's print_detailed_table,
    but fetches data from the API instead of calculating from raw samples.
    """
    # Use CLI context manager to create a single session for all API calls
    async with cli_api_client() as client:
        # Fetch scores, environments, and config
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
    
        # Print header
        print("=" * 180, flush=True)
        print(f"MINER RANKING TABLE - Block {block_number}", flush=True)
        print("=" * 180, flush=True)
        
        # Build header - Hotkey first, then UID, then Model, then First Block, then environments
        header_parts = ["Hotkey  ", " UID", "Model                    ", " FirstBlk "]
        
        # Format environment names - keep everything after ':'
        for env in environments:
            if ':' in env:
                env_display = env.split(':', 1)[1]
            else:
                env_display = env
            # Adjust width to accommodate "score[threshold]/count(!)" format
            header_parts.append(f"{env_display:>20}")
        
        # Find all layers that have non-zero weights
        all_layers = set()
        for score in scores_list:
            scores_by_layer = score.get("scores_by_layer", {})
            for layer_key, weight in scores_by_layer.items():
                if weight > 0:
                    # Extract layer number from "L3" format
                    layer_num = int(layer_key[1:])
                    all_layers.add(layer_num)
        
        active_layers = sorted(all_layers)
        
        for layer in active_layers:
            header_parts.append(f"{'L'+str(layer):>8}")
        
        header_parts.extend(["   Total ", "  Weight ", "V"])
        
        print(" | ".join(header_parts), flush=True)
        print("-" * 180, flush=True)
        
        # Print each miner row
        for score in scores_list:
            uid = score.get("uid")
            hotkey = score.get("miner_hotkey")
            model_revision = score.get("model_revision")
            model = score.get("model")
            first_block = score.get("first_block")
            overall_score = score.get("overall_score")
            scores_by_env = score.get("scores_by_env", {})
            scores_by_layer = score.get("scores_by_layer", {})
            total_samples = score.get("total_samples")
            
            model_display = model[:25]
            
            row_parts = [
                f"{hotkey[:8]:8s}",
                f"{uid:4d}",
                f"{model_display:25s}",
                f"{first_block:10d}"
            ]
            
            # Environment scores - show "score[threshold]/count(!)" format (score × 100, 2 decimals)
            # Get default min_completeness from config (default: 0.9 if not available)
            default_min_completeness = scorer_config.get("min_completeness", 0.9)
            
            for env in environments:
                if env in scores_by_env:
                    env_data = scores_by_env[env]
                    env_score = env_data.get("score", 0.0)
                    sample_count = env_data.get("sample_count", 0)
                    completeness = env_data.get("completeness", 1.0)
                    threshold = env_data.get("threshold", 0.0)
                    
                    score_percent = env_score * 100
                    threshold_percent = threshold * 100
                    
                    # Get environment-specific min_completeness or use default
                    env_config = env_configs.get(env, {})
                    env_min_completeness = env_config.get("min_completeness", default_min_completeness)
                    
                    # Check if sample count is insufficient using environment-specific parameter
                    is_insufficient = completeness < env_min_completeness
                    
                    if is_insufficient:
                        score_str = f"{score_percent:.2f}[{threshold_percent:.2f}]/{sample_count}!"
                    else:
                        score_str = f"{score_percent:.2f}[{threshold_percent:.2f}]/{sample_count}"
                    row_parts.append(f"{score_str:>20}")
                else:
                    row_parts.append(f"{'  -  ':>20}")
            
            # Layer weights - only for active layers
            for layer in active_layers:
                layer_key = f"L{layer}"
                weight = scores_by_layer.get(layer_key, 0.0)
                row_parts.append(f"{weight:>8.4f}")
            
            # Total (cumulative) and Weight (normalized)
            # Use cumulative_weight if available, otherwise fall back to average_score
            cumulative_weight = score.get("cumulative_weight")
            row_parts.append(f"{cumulative_weight:>9.4f}")  # Total: cumulative weight before normalization
            row_parts.append(f"{overall_score:>9.6f}")  # Weight: normalized weight
            row_parts.append("✓" if overall_score > 0 else "✗")
            
            print(" | ".join(row_parts), flush=True)
        
        print("=" * 180, flush=True)
        print(f"Total miners: {len(scores_list)}", flush=True)
        non_zero = len([s for s in scores_list if s.get("overall_score", 0.0) > 0])
        print(f"Active miners (weight > 0): {non_zero}", flush=True)
        print("=" * 180, flush=True)


async def get_rank_command():
    """Command handler for get-rank.
    
    Fetches score snapshot from API and displays ranking table.
    """
    try:
        await print_rank_table()
    except Exception as e:
        logger.error(f"Failed to fetch and display ranks: {e}", exc_info=True)
        print(f"Error: {e}")