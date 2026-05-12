from typing import List

def geometric_mean(values: List[float], epsilon: float = 0.0) -> float:
    """Geometric mean with optional epsilon smoothing.

    When epsilon > 0, applies smoothing to prevent zero scores from
    collapsing the result to 0:
        GM_smoothed = ((v1+e) * (v2+e) * ... * (vn+e))^(1/n) - e

    Args:
        values: list of numeric values (typically in [0, 1])
        epsilon: smoothing offset (0.0 to disable)

    Returns:
        Geometric mean (smoothed if epsilon > 0). Returns 0.0 for empty input.
    """
    if not values:
        return 0.0

    product = 1.0
    n = len(values)

    for v in values:
        if epsilon > 0:
            v += epsilon
        if v <= 0:
            return 0.0
        product *= v

    return product ** (1.0 / n) - (epsilon if epsilon > 0 else 0.0)