"""
Scorer Configuration

Central configuration for the champion challenge scoring system.
Every parameter listed here has a clear, single use in the algorithm.
"""

from typing import Dict, Any


class ScorerConfig:
    """Configuration for the champion challenge scoring algorithm."""

    # ── Pareto Comparison ────────────────────────────────────────────────────

    WIN_MARGIN_START: float = 0.02
    """Margin at the first post-warmup checkpoint (CP = warmup+1).

    Low bar early: protects models from being terminated by bad luck
    when data is still sparse.
    """

    WIN_MARGIN_END: float = 0.03
    """Margin at the dethrone checkpoint (CP = CHAMPION_DETHRONE_MIN_CHECKPOINT).

    Higher bar for the decisive comparison: only genuinely better models
    can take the crown. Margin scales linearly between start and end
    as checkpoints progress.
    """

    WIN_MIN_DOMINANT_ENVS: int = 3
    """Champion challenge: minimum environments where the challenger must
    exceed by PARETO_MARGIN. Remaining environments must not be worse
    (score_b >= score_a).

    0 = strict: must exceed in ALL environments.
    N > 0 = partial: exceed in at least N, not lose in any other.
    """

    PARETO_MIN_DOMINANT_ENVS: int = 0
    """Pareto dominance filter: minimum environments for dominance.

    Defaults to 0 (strict Pareto) — a miner must be dominated in ALL
    environments to be eliminated. This avoids false positives: a
    legitimately improved miner that wins some envs and loses others
    should not be eliminated.
    """

    PARETO_MARGIN: float = 0.04
    """Fixed margin for pairwise and champion dominance comparisons.

    Higher than champion challenge margin to avoid terminating miners
    that simply fine-tuned one env slightly. Pairwise does not use
    dynamic margin — it's always this fixed value.
    """

    # ── Geometric Mean ────────────────────────────────────────────────────

    GEOMETRIC_MEAN_EPSILON: float = 0.1
    """Smoothing offset for geometric mean (dethrone tiebreaker).

    Prevents zero scores in any single env from collapsing the entire
    geo-mean to 0. Computed as ((v1+ε)·(v2+ε)·...·(vn+ε))^(1/n) - ε.
    """

    # ── Score Normalization ──────────────────────────────────────────────────

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

    CHAMPION_DETHRONE_MIN_CHECKPOINT: int = 10
    """Minimum checkpoint to be eligible for dethrone. A challenger must
    reach this CP and dominate the champion in a single comparison to
    take the crown. CP=10 means 10×window_size common tasks
    (e.g., window=100 → 1000 tasks)."""

    CHAMPION_TERMINATION_TOTAL_LOSSES: int = 3
    """Accumulated post-warmup losses → terminate sampling."""

    CHAMPION_TERMINATION_CONSECUTIVE_LOSSES: int = 2
    """Consecutive post-warmup losses → terminate sampling."""

    PARETO_MIN_WINDOWS: int = 3
    """Minimum windows of common tasks before pairwise Pareto fires.

    Pairwise filter terminates plagiarized models: when two non-champion
    miners share at least PARETO_MIN_WINDOWS × window_size common tasks,
    run a Pareto comparison. The earlier-registered miner is the incumbent
    and the dominated miner is terminated.
    """

    # ── Export ───────────────────────────────────────────────────────────────

    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        return {
            'win_margin_start': cls.WIN_MARGIN_START,
            'win_margin_end': cls.WIN_MARGIN_END,
            'win_min_dominant_envs': cls.WIN_MIN_DOMINANT_ENVS,
            'pareto_min_dominant_envs': cls.PARETO_MIN_DOMINANT_ENVS,
            'geometric_mean_epsilon': cls.GEOMETRIC_MEAN_EPSILON,
            'champion_warmup_checkpoints': cls.CHAMPION_WARMUP_CHECKPOINTS,
            'champion_dethrone_min_checkpoint': cls.CHAMPION_DETHRONE_MIN_CHECKPOINT,
            'champion_termination_total_losses': cls.CHAMPION_TERMINATION_TOTAL_LOSSES,
            'champion_termination_consecutive_losses': cls.CHAMPION_TERMINATION_CONSECUTIVE_LOSSES,
            'pareto_min_windows': cls.PARETO_MIN_WINDOWS,
        }

    @classmethod
    def validate(cls):
        assert 0.0 < cls.WIN_MARGIN_START < 1.0, "WIN_MARGIN_START must be in (0, 1)"
        assert 0.0 < cls.WIN_MARGIN_END < 1.0, "WIN_MARGIN_END must be in (0, 1)"
        assert cls.WIN_MARGIN_END >= cls.WIN_MARGIN_START, "WIN_MARGIN_END must be >= START"
        assert cls.GEOMETRIC_MEAN_EPSILON >= 0.0, "GEOMETRIC_MEAN_EPSILON must be non-negative"
        assert cls.CHAMPION_WARMUP_CHECKPOINTS >= 0, "CHAMPION_WARMUP_CHECKPOINTS must be >= 0"
        assert cls.CHAMPION_DETHRONE_MIN_CHECKPOINT >= 1, "CHAMPION_DETHRONE_MIN_CHECKPOINT must be >= 1"
        assert cls.CHAMPION_TERMINATION_TOTAL_LOSSES >= 1, "CHAMPION_TERMINATION_TOTAL_LOSSES must be >= 1"
        assert cls.CHAMPION_TERMINATION_CONSECUTIVE_LOSSES >= 1, "CHAMPION_TERMINATION_CONSECUTIVE_LOSSES must be >= 1"
        assert cls.PARETO_MIN_WINDOWS >= 1, "PARETO_MIN_WINDOWS must be >= 1"


# Validate configuration on import
ScorerConfig.validate()
