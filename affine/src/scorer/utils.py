"""
Scorer Utility Functions
"""

from typing import List
import math


def geometric_mean(values: List[float], epsilon: float = 0.0) -> float:
    """Calculate geometric mean of a list of values with optional smoothing.

    When epsilon > 0, applies smoothing to prevent zero scores from collapsing
    the entire geometric mean to 0:
        GM_smoothed = ((v1+e) * (v2+e) * ... * (vn+e))^(1/n) - e

    Args:
        values: List of numeric values
        epsilon: Smoothing offset (0.0 to disable)

    Returns:
        Geometric mean (smoothed if epsilon > 0)
    """
    if not values:
        return 0.0

    n = len(values)

    if epsilon > 0:
        product = 1.0
        for v in values:
            product *= (v + epsilon)
        return max(product ** (1.0 / n) - epsilon, 0.0)

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
    - More samples -> smaller SE -> smaller gap -> easier to beat
    - Fewer samples -> larger SE -> larger gap -> harder to beat

    Formula:
        SE = sqrt(p * (1-p) / n)
        gap = z * SE
        gap = max(gap, min_improvement)
        gap = min(gap, max_improvement)
        threshold = prior_score + gap

    Args:
        prior_score: Score of the incumbent (0.0 to 1.0)
        prior_sample_count: Number of samples for prior score
        z_score: Z-score for confidence level (default: 1.5)
        min_improvement: Minimum gap (default: 0.02 = 2%)
        max_improvement: Maximum gap cap (default: 0.10 = 10%)

    Returns:
        Required score threshold to beat the incumbent
    """
    if prior_sample_count <= 0:
        gap = max_improvement
    else:
        p = prior_score
        se = math.sqrt(p * (1.0 - p) / prior_sample_count)
        gap = z_score * se
        gap = max(gap, min_improvement)
        gap = min(gap, max_improvement)

    return min(prior_score + gap, 1.0)
