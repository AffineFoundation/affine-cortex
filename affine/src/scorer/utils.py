"""
Scorer Utility Functions
"""

import math
from typing import List


def _safe_log_geometric_mean(values: List[float]) -> float:
    """Compute the geometric mean in log-space.

    Returns 0.0 for any non-positive input or if the computation fails.
    """
    if not values or any(v <= 0 for v in values):
        return 0.0

    try:
        log_mean = sum(math.log(v) for v in values) / len(values)
        return math.exp(log_mean)
    except (ValueError, OverflowError):
        return 0.0


def geometric_mean(values: List[float], epsilon: float = 0.0) -> float:
    """Geometric mean with optional epsilon smoothing.

    When epsilon > 0, applies smoothing to prevent zero scores from
    collapsing the result to 0:
        GM_smoothed = exp((log(v1+e) + log(v2+e) + ... + log(vn+e)) / n) - e

    Args:
        values: list of numeric values (typically in [0, 1])
        epsilon: smoothing offset (0.0 to disable)

    Returns:
        Geometric mean (smoothed if epsilon > 0). Returns 0.0 for empty input.
    """
    if not values:
        return 0.0

    if epsilon > 0:
        adjusted_values = [v + epsilon for v in values]
        return max(_safe_log_geometric_mean(adjusted_values) - epsilon, 0.0)

    return _safe_log_geometric_mean(values)