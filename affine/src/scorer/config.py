"""
Scorer Configuration

Central configuration for the scoring algorithm.
All parameters are defined as constants for clarity and maintainability.
"""

from typing import Dict, Any


class ScorerConfig:
    """Configuration for the four-stage scoring algorithm."""
    
    # Stage 2: Pareto Frontier Anti-Plagiarism
    Z_SCORE: float = 2.0
    """
    Z-score for statistical confidence interval in threshold calculation.

    Uses standard error (SE) based approach to adjust threshold by sample size:
    - SE = sqrt(p * (1-p) / n)
    - gap = z * SE

    Z-score values:
    - 1.0: ~68% confidence (more aggressive, smaller gaps)
    - 1.5: ~87% confidence (balanced)
    - 1.96: 95% confidence (more conservative, larger gaps)

    Higher sample counts → smaller SE → smaller gap → easier to beat.
    Lower sample counts → larger SE → larger gap → harder to beat.

    Recommended value: 2.0
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
    
    GEOMETRIC_MEAN_EPSILON: float = 0.1
    """
    Smoothing epsilon for geometric mean calculation.

    Shifts all scores by +ε before computing geometric mean, then shifts
    back by -ε. This prevents zero scores from collapsing the entire
    geometric mean to 0, which is critical when a new environment is added
    and all miners initially score 0.

    Score range shifts from [0, 1] to [ε, 1+ε].

    Formula: GM_smoothed = ((v1+ε) × (v2+ε) × ... × (vn+ε))^(1/n) - ε

    Set to 0.0 to disable smoothing (original behavior).
    Recommended value: 0.1
    """

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
        'GAME': {'z_score': 2.0},
        'PRINT': {'z_score': 2.0},
        'SWE-SYNTH': {'z_score': 2.0},
        'SWE-INFINITE': {'z_score': 2.0},
    }
    
    # Champion Challenge Parameters
    CHAMPION_WARMUP_CHECKPOINTS: int = 2
    """Number of initial checkpoints that don't count toward wins/losses.

    Early checkpoints have few common tasks, so thresholds are high and
    results are noisy. During warmup, comparisons are logged but don't
    affect win/loss counters or trigger termination.
    """

    CHAMPION_CONSECUTIVE_WINS_REQUIRED: int = 10
    """Number of consecutive checkpoint wins needed to dethrone champion (N).

    Counted only after warmup checkpoints. The challenger must dominate
    at N consecutive post-warmup checkpoints.
    """

    CHAMPION_TERMINATION_TOTAL_LOSSES: int = 3
    """Accumulated checkpoint losses to stop sampling a challenger (M)."""

    CHAMPION_TERMINATION_CONSECUTIVE_LOSSES: int = 2
    """Consecutive checkpoint losses to stop sampling a challenger (M-1)."""

    PAIRWISE_MIN_WINDOWS: int = 3
    """Minimum windows of common tasks before pairwise Pareto comparison fires.

    Pairwise filter (anti-plagiarism): if miner A (older) and miner B (newer)
    share at least PAIRWISE_MIN_WINDOWS × window_size common tasks, run Pareto
    comparison. The dominated miner is terminated. A copy of an incumbent
    cannot significantly beat the incumbent's threshold, so this filters
    plagiarized models before they can challenge the champion.
    """


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
            'min_completeness': cls.MIN_COMPLETENESS,
            'geometric_mean_epsilon': cls.GEOMETRIC_MEAN_EPSILON,
            'champion_warmup_checkpoints': cls.CHAMPION_WARMUP_CHECKPOINTS,
            'champion_consecutive_wins_required': cls.CHAMPION_CONSECUTIVE_WINS_REQUIRED,
            'champion_termination_total_losses': cls.CHAMPION_TERMINATION_TOTAL_LOSSES,
            'champion_termination_consecutive_losses': cls.CHAMPION_TERMINATION_CONSECUTIVE_LOSSES,
            'pairwise_min_windows': cls.PAIRWISE_MIN_WINDOWS,
        }
    
    @classmethod
    def validate(cls):
        """Validate configuration parameters."""
        assert cls.Z_SCORE > 0.0, "Z_SCORE must be positive"
        assert cls.MIN_IMPROVEMENT >= 0.0, "MIN_IMPROVEMENT must be non-negative"
        assert cls.MAX_IMPROVEMENT >= cls.MIN_IMPROVEMENT, "MAX_IMPROVEMENT must be >= MIN_IMPROVEMENT"
        assert cls.SCORE_PRECISION >= 0, "SCORE_PRECISION must be non-negative"
        assert 0.0 <= cls.MIN_COMPLETENESS <= 1.0, "MIN_COMPLETENESS must be in [0, 1]"
        assert cls.GEOMETRIC_MEAN_EPSILON >= 0.0, "GEOMETRIC_MEAN_EPSILON must be non-negative"
        assert cls.CHAMPION_WARMUP_CHECKPOINTS >= 0, "CHAMPION_WARMUP_CHECKPOINTS must be >= 0"
        assert cls.CHAMPION_CONSECUTIVE_WINS_REQUIRED >= 1, "CHAMPION_CONSECUTIVE_WINS_REQUIRED must be >= 1"
        assert cls.CHAMPION_TERMINATION_TOTAL_LOSSES >= 1, "CHAMPION_TERMINATION_TOTAL_LOSSES must be >= 1"
        assert cls.CHAMPION_TERMINATION_CONSECUTIVE_LOSSES >= 1, "CHAMPION_TERMINATION_CONSECUTIVE_LOSSES must be >= 1"
        assert cls.PAIRWISE_MIN_WINDOWS >= 1, "PAIRWISE_MIN_WINDOWS must be >= 1"


# Validate configuration on import
ScorerConfig.validate()