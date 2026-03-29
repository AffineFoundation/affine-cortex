"""
OpenSkill Rating System Configuration
"""

from typing import Dict, Any


class OpenSkillConfig:
    """Configuration for OpenSkill (PlackettLuce) per-task rating system."""

    # PlackettLuce model parameters
    MU_INIT: float = 25.0
    SIGMA_INIT: float = 25.0 / 3  # ~8.333
    TAU: float = 0.19
    """Additive dynamics factor. Controls how fast old ratings decay.
    effective_memory ≈ (sigma_init/2 / tau)² ≈ 480 games."""

    # Qualification thresholds
    MIN_QUALIFIED_ENVS: int = 3
    """Miner must be qualified in at least this many envs to receive weight."""

    WINDOW_QUALIFICATION_RATIO: float = 0.8
    """Miner must have completed at least this ratio of the sampling window tasks."""

    # Weight computation
    ORDINAL_Z: float = 3.0
    """ordinal = mu - ORDINAL_Z * sigma"""

    # Match constraints
    MIN_PARTICIPANTS: int = 2
    """Skip tasks with fewer participants."""

    # Cold start
    COLD_START_DAYS: int = 14

    # TTL
    MATCH_TTL_DAYS: int = 30

    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        return {
            'mu_init': cls.MU_INIT,
            'sigma_init': cls.SIGMA_INIT,
            'tau': cls.TAU,
            'ordinal_z': cls.ORDINAL_Z,
            'min_qualified_envs': cls.MIN_QUALIFIED_ENVS,
            'window_qualification_ratio': cls.WINDOW_QUALIFICATION_RATIO,
            'min_participants': cls.MIN_PARTICIPANTS,
            'cold_start_days': cls.COLD_START_DAYS,
        }

    @classmethod
    def validate(cls):
        assert cls.MU_INIT > 0
        assert cls.SIGMA_INIT > 0
        assert cls.TAU >= 0
        assert cls.ORDINAL_Z > 0
        assert cls.MIN_QUALIFIED_ENVS >= 1
        assert 0 < cls.WINDOW_QUALIFICATION_RATIO <= 1.0
        assert cls.MIN_PARTICIPANTS >= 2


OpenSkillConfig.validate()
