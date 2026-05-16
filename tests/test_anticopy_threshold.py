"""``AntiCopyConfig`` reads from SystemConfig, falls back to defaults."""

import asyncio

import pytest

from affine.src.anticopy.threshold import (
    DEFAULT_NLL_THRESHOLD,
    DEFAULT_MIN_OVERLAP,
    DEFAULT_POOL_DAYS,
    DEFAULT_ROLLOUTS_PER_ENV,
    AntiCopyConfig,
    load_anticopy_config,
)


class _FakeDAO:
    """Minimal SystemConfigDAO surface load_anticopy_config touches."""
    def __init__(self, blob):
        self._blob = blob

    async def get_param_value(self, name, default=None):
        if name == "anticopy":
            return self._blob if self._blob is not None else default
        return default


@pytest.mark.asyncio
async def test_missing_block_returns_defaults():
    cfg = await load_anticopy_config(_FakeDAO(None))
    assert cfg.enabled is False
    assert cfg.nll_threshold == DEFAULT_NLL_THRESHOLD
    assert cfg.min_overlap == DEFAULT_MIN_OVERLAP
    assert cfg.pool_days == DEFAULT_POOL_DAYS
    assert cfg.rollouts_per_env == DEFAULT_ROLLOUTS_PER_ENV
    assert isinstance(cfg.enabled_envs, list) and cfg.enabled_envs


@pytest.mark.asyncio
async def test_partial_overrides_merge_with_defaults():
    cfg = await load_anticopy_config(_FakeDAO({
        "enabled": True,
        "nll_threshold": 0.01,
        "enabled_envs": ["CDE", "MTH"],
    }))
    assert cfg.enabled is True
    assert cfg.nll_threshold == 0.01
    # untouched keys still hit defaults
    assert cfg.min_overlap == DEFAULT_MIN_OVERLAP
    assert cfg.enabled_envs == ["CDE", "MTH"]


@pytest.mark.asyncio
async def test_invalid_values_fall_back_to_defaults():
    cfg = await load_anticopy_config(_FakeDAO({
        "nll_threshold": "not-a-number",
        "min_overlap": "still-not",
    }))
    assert cfg.nll_threshold == DEFAULT_NLL_THRESHOLD
    assert cfg.min_overlap == DEFAULT_MIN_OVERLAP
