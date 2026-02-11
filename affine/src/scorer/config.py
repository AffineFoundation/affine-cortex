"""
Scorer Configuration

Central configuration for the scoring algorithm.
All parameters are defined as constants for clarity and maintainability.
"""

from typing import Dict, Any


class ScorerConfig:
    """Configuration for the four-stage scoring algorithm."""
    
    # Stage 2: Pareto Frontier Anti-Plagiarism
    Z_SCORE: float = 1.5
    """
    Z-score for statistical confidence interval in threshold calculation.

    Uses standard error (SE) based approach to adjust threshold by sample size:
    - SE = sqrt(p * (1-p) / n)
    - gap = z * SE

    Z-score values:
    - 1.0: ~68% confidence (more aggressive, smaller gaps)
    - 1.5: ~87% confidence (balanced, recommended)
    - 1.96: 95% confidence (more conservative, larger gaps)

    Higher sample counts → smaller SE → smaller gap → easier to beat.
    Lower sample counts → larger SE → larger gap → harder to beat.

    Recommended value: 1.5
    """

    MIN_IMPROVEMENT: float = 0.02
    """
    Minimum improvement required for later miner to beat earlier miner.

    Ensures that even with very large sample sizes (small SE), there's still
    a minimum gap to prevent noise and random fluctuations from allowing
    copies to beat originals.

    Example: If SE-based gap = 0.01 but MIN_IMPROVEMENT = 0.02,
    the actual gap used will be 0.02.

    Recommended value: 0.02 (2%)
    """

    MAX_IMPROVEMENT: float = 0.10
    """
    Maximum improvement threshold cap.

    Caps the required score gap to prevent unreasonably high thresholds
    when sample size is very small (large SE).

    Example: If SE-based gap = 0.25 but MAX_IMPROVEMENT = 0.10,
    the actual gap used will be capped at 0.10.

    Recommended value: 0.10 (10%)
    """
    
    SCORE_PRECISION: int = 3
    """Number of decimal places for score comparison (avoid floating point issues)."""
    
    # Stage 3: Subset Scoring
    MAX_LAYERS: int = 6
    """Maximum number of layers to evaluate. Due to exponential growth, 6 layers provide sufficient differentiation (2^5 = 32x difference between L1 and L6)."""
    
    SUBSET_WEIGHT_BASE: int = 1
    """Base weight multiplier for subset layers (N for L1, N*2 for L2, N*4 for L3, etc.)."""
    
    SUBSET_WEIGHT_EXPONENT: int = 2
    """Exponent base for layer weights (layer_weight = N * base^(layer-1))."""
    
    GEOMETRIC_MEAN_EPSILON: float = 0.01
    """
    Smoothing epsilon for geometric mean calculation.

    Shifts all scores by +ε before computing geometric mean, then shifts
    back by -ε. This prevents zero scores from collapsing the entire
    geometric mean to 0, which is critical when a new environment is added
    and all miners initially score 0.

    Score range shifts from [0, 1] to [ε, 1+ε].

    Formula: GM_smoothed = ((v1+ε) × (v2+ε) × ... × (vn+ε))^(1/n) - ε

    Set to 0.0 to disable smoothing (original behavior).
    Recommended value: 0.01
    """

    DECAY_FACTOR: float = 0.5
    """
    Rank-based decay factor for score_proportional weighting.

    Applied as: adjusted_score = score × decay_factor^(rank - 1)
    - Rank 1: score × 1.0
    - Rank 2: score × decay_factor^1
    - Rank 3: score × decay_factor^2

    Set to 1.0 to disable decay (all ranks weighted equally).
    Set to 0.5 for exponential decay (each rank gets 50% of previous).
    """
    
    # Stage 4: Weight Normalization
    MIN_WEIGHT_THRESHOLD: float = 0.01
    """Minimum weight threshold (1%). Miners below this are set to 0."""
    
    # Stage 1: Data Collection
    MIN_COMPLETENESS: float = 0.9
    """Minimum sample completeness required."""
    
    # Environment Score Normalization
    # Format: env_name -> (min_score, max_score)
    # Scores will be normalized to [0, 1] range: (score - min) / (max - min)
    ENV_SCORE_RANGES: Dict[str, tuple] = {
        'agentgym:sciworld': (-100, 100.0)  # sciworld 分数范围 0-100
    }

    # Environment-specific threshold difficulty configs
    # Format: env_name -> {z_score, min_improvement, max_improvement}
    # Lower values = easier to beat (lower difficulty)
    # Higher values = harder to beat (higher difficulty)
    ENV_THRESHOLD_CONFIGS: Dict[str, Dict[str, float]] = {
        'GAME': {'z_score': 1},    # easier to beat (default 1.5)
        'PRINT': {'z_score': 2.0},   # harder to beat (default 1.5)
        'SWE-SYNTH': {'z_score': 2.0},
    }
    
    # Database & Storage
    SCORE_RECORD_TTL_DAYS: int = 30
    """TTL for score_snapshots table (in days)."""
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Export configuration as dictionary for storage in snapshots."""
        return {
            'z_score': cls.Z_SCORE,
            'min_improvement': cls.MIN_IMPROVEMENT,
            'max_improvement': cls.MAX_IMPROVEMENT,
            'score_precision': cls.SCORE_PRECISION,
            'max_layers': cls.MAX_LAYERS,
            'subset_weight_base': cls.SUBSET_WEIGHT_BASE,
            'subset_weight_exponent': cls.SUBSET_WEIGHT_EXPONENT,
            'decay_factor': cls.DECAY_FACTOR,
            'min_weight_threshold': cls.MIN_WEIGHT_THRESHOLD,
            'min_completeness': cls.MIN_COMPLETENESS,
            'geometric_mean_epsilon': cls.GEOMETRIC_MEAN_EPSILON,
        }
    
    @classmethod
    def validate(cls):
        """Validate configuration parameters."""
        assert cls.Z_SCORE > 0.0, "Z_SCORE must be positive"
        assert cls.MIN_IMPROVEMENT >= 0.0, "MIN_IMPROVEMENT must be non-negative"
        assert cls.MAX_IMPROVEMENT >= cls.MIN_IMPROVEMENT, "MAX_IMPROVEMENT must be >= MIN_IMPROVEMENT"
        assert cls.SCORE_PRECISION >= 0, "SCORE_PRECISION must be non-negative"
        assert cls.SUBSET_WEIGHT_BASE > 0, "SUBSET_WEIGHT_BASE must be positive"
        assert cls.SUBSET_WEIGHT_EXPONENT >= 2, "SUBSET_WEIGHT_EXPONENT must be >= 2"
        assert 0.0 <= cls.DECAY_FACTOR <= 1.0, "DECAY_FACTOR must be in [0, 1]"
        assert 0.0 <= cls.MIN_WEIGHT_THRESHOLD <= 1.0, "MIN_WEIGHT_THRESHOLD must be in [0, 1]"
        assert 0.0 <= cls.MIN_COMPLETENESS <= 1.0, "MIN_COMPLETENESS must be in [0, 1]"
        assert cls.GEOMETRIC_MEAN_EPSILON >= 0.0, "GEOMETRIC_MEAN_EPSILON must be non-negative"


# Validate configuration on import
ScorerConfig.validate()