"""
Scorer Utility Functions
"""

import math
from typing import List


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

    n = len(values)

    if epsilon > 0:
        adjusted_values = [v + epsilon for v in values]
        if any(v <= 0 for v in adjusted_values):
            return 0.0

        try:
            log_mean = sum(math.log(v) for v in adjusted_values) / n
            return max(math.exp(log_mean) - epsilon, 0.0)
        except (ValueError, OverflowError):
            return 0.0

    if any(v <= 0 for v in values):
        return 0.0

    try:
        log_mean = sum(math.log(v) for v in values) / n
        return math.exp(log_mean)
    except (ValueError, OverflowError):
        return 0.0