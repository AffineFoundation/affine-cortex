"""
Scorer Configuration

Central configuration for the champion challenge scoring system.
Every parameter listed here has a clear, single use in the algorithm.
"""

from typing import Dict, Any


class ScorerConfig:
    """Configuration for the champion challenge scoring algorithm."""

    # ── Pareto Comparison ────────────────────────────────────────────────────

    PARETO_MARGIN: float = 0.02
    """Fixed margin a challenger must beat in each environment to win.

    A challenger's score in env E must exceed the champion's score in env E
    by more than PARETO_MARGIN to count as winning that environment. To
    dominate the champion, the challenger must win in ALL environments.

    Replaces the previous dynamic SE-based threshold (which was always
    floored to this value at the sample sizes guaranteed by the checkpoint
    mechanism, so the dynamic part was meaningless).
    """

    # ── Cold Start ───────────────────────────────────────────────────────────

    GEOMETRIC_MEAN_EPSILON: float = 0.1
    """Smoothing offset for cold-start champion selection by geometric mean.

    Prevents zero scores in any single env from collapsing the entire
    geo-mean to 0. Computed as ((v1+ε)·(v2+ε)·...·(vn+ε))^(1/n) - ε.
    """

    # ── Display Only ─────────────────────────────────────────────────────────

    MIN_COMPLETENESS: float = 0.9
    """Threshold below which an environment's data is marked as sparse in
    the rank display (with a "!" indicator). Does NOT affect the algorithm —
    miners with low completeness still participate in scoring.
    """

    ENV_SCORE_RANGES: Dict[str, tuple] = {
        'agentgym:sciworld': (-100, 100.0),
    }
    """Per-environment score normalization. Maps env_name → (min, max) so
    raw scores are normalized to [0, 1] before any comparison."""

    # ── Champion Challenge ───────────────────────────────────────────────────

    CHAMPION_WARMUP_CHECKPOINTS: int = 2
    """First K checkpoints don't count toward wins/losses. Early checkpoints
    have less data and noisier comparisons; this protects good models from
    being terminated by random fluctuations during ramp-up."""

    CHAMPION_CONSECUTIVE_WINS_REQUIRED: int = 10
    """Number of consecutive post-warmup checkpoint wins to take the crown."""

    CHAMPION_TERMINATION_TOTAL_LOSSES: int = 3
    """Accumulated post-warmup losses → terminate sampling."""

    CHAMPION_TERMINATION_CONSECUTIVE_LOSSES: int = 2
    """Consecutive post-warmup losses → terminate sampling."""

    PAIRWISE_MIN_WINDOWS: int = 3
    """Minimum windows of common tasks before pairwise Pareto fires.

    Pairwise filter terminates plagiarized models: when two non-champion
    miners share at least PAIRWISE_MIN_WINDOWS × window_size common tasks,
    run a Pareto comparison. The earlier-registered miner is the incumbent
    and the dominated miner is terminated.
    """

    # ── Export ───────────────────────────────────────────────────────────────

    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        return {
            'pareto_margin': cls.PARETO_MARGIN,
            'geometric_mean_epsilon': cls.GEOMETRIC_MEAN_EPSILON,
            'min_completeness': cls.MIN_COMPLETENESS,
            'champion_warmup_checkpoints': cls.CHAMPION_WARMUP_CHECKPOINTS,
            'champion_consecutive_wins_required': cls.CHAMPION_CONSECUTIVE_WINS_REQUIRED,
            'champion_termination_total_losses': cls.CHAMPION_TERMINATION_TOTAL_LOSSES,
            'champion_termination_consecutive_losses': cls.CHAMPION_TERMINATION_CONSECUTIVE_LOSSES,
            'pairwise_min_windows': cls.PAIRWISE_MIN_WINDOWS,
        }

    @classmethod
    def validate(cls):
        assert 0.0 < cls.PARETO_MARGIN < 1.0, "PARETO_MARGIN must be in (0, 1)"
        assert cls.GEOMETRIC_MEAN_EPSILON >= 0.0, "GEOMETRIC_MEAN_EPSILON must be non-negative"
        assert 0.0 <= cls.MIN_COMPLETENESS <= 1.0, "MIN_COMPLETENESS must be in [0, 1]"
        assert cls.CHAMPION_WARMUP_CHECKPOINTS >= 0, "CHAMPION_WARMUP_CHECKPOINTS must be >= 0"
        assert cls.CHAMPION_CONSECUTIVE_WINS_REQUIRED >= 1, "CHAMPION_CONSECUTIVE_WINS_REQUIRED must be >= 1"
        assert cls.CHAMPION_TERMINATION_TOTAL_LOSSES >= 1, "CHAMPION_TERMINATION_TOTAL_LOSSES must be >= 1"
        assert cls.CHAMPION_TERMINATION_CONSECUTIVE_LOSSES >= 1, "CHAMPION_TERMINATION_CONSECUTIVE_LOSSES must be >= 1"
        assert cls.PAIRWISE_MIN_WINDOWS >= 1, "PAIRWISE_MIN_WINDOWS must be >= 1"


# Validate configuration on import
ScorerConfig.validate()
