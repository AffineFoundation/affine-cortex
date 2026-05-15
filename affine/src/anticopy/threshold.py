"""
CEAC thresholds and per-env configuration.

Constants live here as defaults; the real values are pulled from
``system_config["anticopy"]`` at runtime so operators can retune
without a redeploy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from affine.database.dao.system_config import SystemConfigDAO


# ----- defaults -------------------------------------------------------

# Combined decision-position |Δlogp| median below which two miners
# are flagged as the same model (potentially with light noise
# injection). Applied to :attr:`PairResult.decision_median_combined`
# — the median over the union of every env's "uncertain" positions
# (where the reference model's top-1 logprob was below
# ``_DECISION_LOGP_CUTOFF``). Not the all-positions median, which
# pins to ~0 for every Qwen3 fine-tune because trivial-prediction
# tokens dominate.
DEFAULT_NLL_THRESHOLD = 0.04

# Minimum token-position overlap a pair must share before we issue a
# verdict either way. Below this we say "pending: insufficient data"
# rather than risk a false positive off a handful of tokens.
DEFAULT_MIN_OVERLAP = 50

# At least this fraction of overlapping rollout envs must agree on
# "copy" for the final verdict to flip. Matches the design's
# ``ceil(|envs|/2)``.
DEFAULT_AGREEMENT_RATIO = 0.5

# How many of the champion's most recent samples per env get promoted
# to the rollout pool on each refresh tick.
DEFAULT_ROLLOUTS_PER_ENV = 20

# Pool retention. Older rollouts get TTL'd out of the DDB index; the
# R2 blob may linger but won't be selected.
DEFAULT_POOL_DAYS = 7

# UTC hour the refresh service runs once per day.
DEFAULT_REFRESH_UTC_HOUR = 2

# Number of HuggingFace ckpts the worker keeps in the local cache. The
# current job's ckpt is always kept; the prefetched-next ckpt counts as
# one of the slots, leaving headroom = (gc_keep_recent - 2) for prior
# evaluated jobs. 3 means current + next + 1 prior, which covers a
# crash-recovery re-run without wasting disk.
DEFAULT_GC_KEEP_RECENT = 3

# Default envs eligible for rollout pool population, used if
# SystemConfig doesn't enumerate them. Operator overrides via
# ``anticopy.enabled_envs`` are authoritative.
# DISTILL is intentionally excluded — its sample_results.extra has no
# ``conversation`` field, so rollouts can't be derived from it.
DEFAULT_ENABLED_ENVS = ("MEMORY", "TERMINAL", "SWE-INFINITE", "NAVWORLD", "LIVEWEB")


@dataclass
class AntiCopyConfig:
    """Resolved CEAC tuning knobs."""

    enabled: bool = False
    nll_threshold: float = DEFAULT_NLL_THRESHOLD
    min_overlap: int = DEFAULT_MIN_OVERLAP
    agreement_ratio: float = DEFAULT_AGREEMENT_RATIO
    rollouts_per_env: int = DEFAULT_ROLLOUTS_PER_ENV
    pool_days: int = DEFAULT_POOL_DAYS
    refresh_utc_hour: int = DEFAULT_REFRESH_UTC_HOUR
    enabled_envs: List[str] = field(default_factory=lambda: list(DEFAULT_ENABLED_ENVS))
    gc_keep_recent: int = DEFAULT_GC_KEEP_RECENT


async def load_anticopy_config(
    dao: Optional[SystemConfigDAO] = None,
) -> AntiCopyConfig:
    """Read the ``anticopy`` block from SystemConfig and merge with
    defaults. Missing keys fall back to module-level constants — this
    keeps the bootstrap-from-empty path safe."""
    dao = dao or SystemConfigDAO()
    raw: Dict[str, Any] = {}
    try:
        raw = await dao.get_param_value("anticopy", default={}) or {}
    except Exception:
        raw = {}

    def _f(key: str, default: float) -> float:
        try:
            return float(raw.get(key, default))
        except (TypeError, ValueError):
            return default

    def _i(key: str, default: int) -> int:
        try:
            return int(raw.get(key, default))
        except (TypeError, ValueError):
            return default

    enabled_envs_raw = raw.get("enabled_envs")
    if isinstance(enabled_envs_raw, list) and enabled_envs_raw:
        enabled_envs = [str(x) for x in enabled_envs_raw]
    else:
        enabled_envs = list(DEFAULT_ENABLED_ENVS)

    return AntiCopyConfig(
        enabled=bool(raw.get("enabled", False)),
        nll_threshold=_f("nll_threshold", DEFAULT_NLL_THRESHOLD),
        min_overlap=_i("min_overlap", DEFAULT_MIN_OVERLAP),
        agreement_ratio=_f("agreement_ratio", DEFAULT_AGREEMENT_RATIO),
        rollouts_per_env=_i("rollouts_per_env", DEFAULT_ROLLOUTS_PER_ENV),
        pool_days=_i("pool_days", DEFAULT_POOL_DAYS),
        refresh_utc_hour=_i("refresh_utc_hour", DEFAULT_REFRESH_UTC_HOUR),
        enabled_envs=enabled_envs,
        gc_keep_recent=max(2, _i("gc_keep_recent", DEFAULT_GC_KEEP_RECENT)),
    )
