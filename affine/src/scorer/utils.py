"""
Scorer Utility Functions

Helper functions for the scoring algorithm.
"""

from typing import List, Dict, Set, Tuple
from itertools import combinations
import math


def generate_all_subsets(envs: List[str], max_layers: int = None) -> Dict[str, Dict[str, any]]:
    """Generate all possible subsets (environment combinations) with layer information.
    
    Args:
        envs: List of environment names
        max_layers: Maximum number of layers to evaluate. If total envs > max_layers,
                   only the top max_layers will be evaluated (e.g., if 8 envs and max_layers=6,
                   evaluate L3-L8, skipping L1-L2)
        
    Returns:
        Dict mapping subset_key to subset metadata:
        {
            "L1_sat": {
                "layer": 1,
                "envs": ["sat"],
                "key": "L1_sat"
            },
            "L2_sat_abd": {
                "layer": 2,
                "envs": ["sat", "abd"],
                "key": "L2_sat_abd"
            },
            ...
        }
    """
    subsets = {}
    n = len(envs)
    
    # Calculate starting layer (skip lower layers if total > max_layers)
    if max_layers is not None and n > max_layers:
        start_layer = n - max_layers + 1  # e.g., 8 envs, max 6 layers: start from L3
    else:
        start_layer = 1
    
    # Generate all combinations for each layer
    for layer in range(start_layer, n + 1):
        for env_combo in combinations(envs, layer):
            # Sort environments alphabetically for consistent keys
            sorted_envs = sorted(env_combo)
            
            # Create subset key: L{layer}_{env1}_{env2}_{...}
            subset_key = f"L{layer}_{'_'.join(sorted_envs)}"
            
            subsets[subset_key] = {
                "layer": layer,
                "envs": sorted_envs,
                "key": subset_key
            }
    
    return subsets


def calculate_layer_weights(n_envs: int, base: int = 2, start_layer: int = 1) -> Dict[int, float]:
    """Calculate weight for each layer based on exponential growth.
    
    Layer weight = N × base^(layer_index)
    where layer_index = layer - start_layer (starts from 0)
    
    Args:
        n_envs: Number of environments
        base: Exponent base (default: 2)
        start_layer: Starting layer number (default: 1)
        
    Returns:
        Dict mapping layer number to total layer weight:
        If start_layer=1: {1: N, 2: N*2, 3: N*4, 4: N*8, ...}
        If start_layer=3: {3: N, 4: N*2, 5: N*4, 6: N*8, ...}
    """
    layer_weights = {}
    for layer in range(start_layer, n_envs + 1):
        layer_index = layer - start_layer  # Relative index from start
        layer_weights[layer] = n_envs * (base ** layer_index)
    return layer_weights


def calculate_subset_weights(
    subsets: Dict[str, Dict[str, any]],
    layer_weights: Dict[int, float]
) -> Dict[str, float]:
    """Calculate individual subset weights by distributing layer weights equally.
    
    Each subset in a layer gets: layer_weight / num_subsets_in_layer
    
    Args:
        subsets: Dict of subset metadata
        layer_weights: Dict mapping layer to total weight
        
    Returns:
        Dict mapping subset_key to individual weight
    """
    # Count subsets per layer
    layer_counts = {}
    for subset_info in subsets.values():
        layer = subset_info["layer"]
        layer_counts[layer] = layer_counts.get(layer, 0) + 1
    
    # Distribute layer weights equally among subsets
    subset_weights = {}
    for subset_key, subset_info in subsets.items():
        layer = subset_info["layer"]
        layer_weight = layer_weights[layer]
        num_subsets = layer_counts[layer]
        
        subset_weights[subset_key] = layer_weight / num_subsets
    
    return subset_weights


def geometric_mean(values: List[float], epsilon: float = 0.0) -> float:
    """Calculate geometric mean of a list of values with optional smoothing.

    When epsilon > 0, applies smoothing to prevent zero scores from collapsing
    the entire geometric mean to 0:
        GM_smoothed = ((v1+ε) × (v2+ε) × ... × (vn+ε))^(1/n) - ε

    When epsilon = 0, uses standard geometric mean (returns 0 if any value is 0).

    Args:
        values: List of numeric values
        epsilon: Smoothing offset. Shifts scores from [0,1] to [ε, 1+ε]
                 before calculation, then shifts back.

    Returns:
        Geometric mean (smoothed if epsilon > 0)
    """
    if not values:
        return 0.0

    n = len(values)

    if epsilon > 0:
        # Smoothed geometric mean: shift up, compute, shift back
        product = 1.0
        for v in values:
            product *= (v + epsilon)
        return max(product ** (1.0 / n) - epsilon, 0.0)

    # Standard geometric mean: any zero collapses result to 0
    if any(v <= 0 for v in values):
        return 0.0

    product = 1.0
    for v in values:
        product *= v

    return product ** (1.0 / n)


def calculate_required_score(
    prior_score: float,
    prior_sample_count: int,
    z_score: float = 1.5,
    min_improvement: float = 0.02,
    max_improvement: float = 0.10
) -> float:
    """Calculate required score threshold using statistical confidence intervals.

    Uses standard error (SE) to adjust threshold based on sample size:
    - More samples → smaller SE → smaller gap → easier to beat
    - Fewer samples → larger SE → larger gap → harder to beat

    Formula:
        SE = sqrt(p * (1-p) / n)
        gap = z * SE
        gap = max(gap, min_improvement)  # floor
        gap = min(gap, max_improvement)  # ceiling
        threshold = prior_score + gap

    Args:
        prior_score: Score of the earlier miner (0.0 to 1.0)
        prior_sample_count: Number of samples for prior score
        z_score: Z-score for confidence level (default: 1.5 ≈ 87% confidence)
        min_improvement: Minimum gap to prevent noise (default: 0.02 = 2%)
        max_improvement: Maximum gap cap (default: 0.10 = 10%)

    Returns:
        Required score threshold to beat the prior miner

    Examples:
        prior=0.5, n=100, z=1.5:
        -> se = sqrt(0.5*0.5/100) = 0.05
        -> gap = 1.5 * 0.05 = 0.075 (7.5%)
        -> threshold = 0.5 + 0.075 = 0.575

        prior=0.5, n=500, z=1.5:
        -> se = sqrt(0.5*0.5/500) ≈ 0.0224
        -> gap = 1.5 * 0.0224 ≈ 0.0336 (3.36%)
        -> threshold = 0.5 + 0.0336 = 0.5336

        prior=0.5, n=50, z=1.5:
        -> se = sqrt(0.5*0.5/50) ≈ 0.0707
        -> gap = 1.5 * 0.0707 ≈ 0.106 → capped to 0.10
        -> threshold = 0.5 + 0.10 = 0.60
    """
    # Handle edge case: no samples
    if prior_sample_count <= 0:
        # Use maximum gap for zero samples (least confident)
        gap = max_improvement
    else:
        # Calculate standard error
        p = prior_score
        se = math.sqrt(p * (1.0 - p) / prior_sample_count)

        # Calculate gap from standard error
        gap = z_score * se

        # Apply bounds: gap must be in [min_improvement, max_improvement]
        gap = max(gap, min_improvement)
        gap = min(gap, max_improvement)

    # Calculate final threshold, capped at 1.0
    return min(prior_score + gap, 1.0)


def normalize_weights(weights: Dict[int, float]) -> Dict[int, float]:
    """Normalize weights to sum to 1.0.
    
    Args:
        weights: Dict mapping UID to raw weight
        
    Returns:
        Dict mapping UID to normalized weight (0.0 to 1.0)
    """
    total = sum(weights.values())
    
    if total == 0:
        return {uid: 0.0 for uid in weights}
    
    return {uid: w / total for uid, w in weights.items()}


def apply_min_threshold(
    weights: Dict[int, float],
    threshold: float = 0.01,
    redistribute_to_uid_zero: bool = False
) -> Dict[int, float]:
    """Set weights below threshold to 0, optionally redistribute to uid 0.

    Args:
        weights: Dict mapping UID to weight
        threshold: Minimum weight threshold (default: 0.01 for 1%)
        redistribute_to_uid_zero: If True, add sub-threshold weights to uid 0

    Returns:
        Dict with sub-threshold weights set to 0 (and redistributed to uid 0 if enabled)
    """
    if not redistribute_to_uid_zero:
        return {
            uid: (w if w >= threshold else 0.0)
            for uid, w in weights.items()
        }

    # Calculate total weight below threshold (excluding uid 0)
    below_threshold_weight = sum(
        w for uid, w in weights.items()
        if uid != 0 and w > 0 and w < threshold
    )

    # Apply threshold and set below-threshold weights to 0
    result = {
        uid: (w if w >= threshold else 0.0)
        for uid, w in weights.items()
    }

    # Add redistributed weight to uid 0
    if below_threshold_weight > 0:
        result[0] = result.get(0, 0.0) + below_threshold_weight

    return result


def aggregate_by_layer(
    subset_weights: Dict[str, float]
) -> Dict[int, float]:
    """Aggregate subset weights by layer.
    
    Args:
        subset_weights: Dict mapping subset_key to weight contribution
        
    Returns:
        Dict mapping layer number to total weight
    """
    layer_totals = {}
    
    for subset_key, weight in subset_weights.items():
        # Extract layer from key (format: L{layer}_...)
        layer_str = subset_key.split('_')[0]  # "L3"
        layer = int(layer_str[1:])  # 3
        
        layer_totals[layer] = layer_totals.get(layer, 0.0) + weight
    
    return layer_totals


def format_score_table_row(
    uid: int,
    hotkey: str,
    env_scores: Dict[str, float],
    env_thresholds: Dict[str, float],
    env_samples: Dict[str, int],
    layer_weights: Dict[int, float],
    total_weight: float,
    is_valid: bool
) -> str:
    """Format a row for the score summary table.
    
    Args:
        uid: Miner UID
        hotkey: Miner hotkey (will be truncated)
        env_scores: Environment scores
        env_thresholds: Threshold upper bounds per environment
        env_samples: Sample counts per environment
        layer_weights: Weights by layer
        total_weight: Total cumulative weight
        is_valid: Whether miner is valid for scoring
        
    Returns:
        Formatted table row string
    """
    # Truncate hotkey
    hotkey_short = f"{hotkey[:8]}..."
    
    # Format environment columns
    env_cols = []
    for env in sorted(env_scores.keys()):
        score = env_scores.get(env, 0.0)
        threshold = env_thresholds.get(env, 0.0)
        samples = env_samples.get(env, 0)
        env_cols.append(f"{score:.3f}/{threshold:.3f}/{samples}")
    
    # Format layer weights
    layer_cols = [f"{layer_weights.get(i, 0.0):.4f}" for i in sorted(layer_weights.keys())]
    
    # Valid indicator
    valid_str = "✓" if is_valid else "✗"
    
    # Build row
    parts = [
        f"{uid:3d}",
        f"{hotkey_short:12s}",
        *env_cols,
        *layer_cols,
        f"{total_weight:.6f}",
        valid_str
    ]
    
    return " | ".join(parts)