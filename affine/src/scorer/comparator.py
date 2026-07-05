"""
Window comparator — Pareto-with-tolerance rule.

For each env the challenger lands in one of three buckets vs the champion:

  - **dominant**  : ``chal_avg > champ_avg + margin`` — strictly better by
                    more than the per-env margin.
  - **not_worse** : ``chal_avg ≥ champ_avg * (1 - not_worse_tolerance)``
                    — within an acceptable regression tolerance of the
                    champion's score (multiplicative, matching the old
                    Pareto stage-2 behavior).
  - **worse**     : below ``champ_avg * (1 - not_worse_tolerance)``.

The challenger dethrones the champion iff:

  - With ``min_dominant_envs == 0`` (strict Pareto):
      every evaluated env is **dominant** for the challenger.
  - With ``min_dominant_envs > 0`` (partial Pareto):
      ``dominant_count ≥ min_dominant_envs`` **AND** every other
      evaluated env is at least ``not_worse``.

Sample-count gate: an env with ``chal_n < min_tasks_per_env`` is treated
as the challenger having **failed** that env (counts as ``worse``) —
the challenger can't dethrone by simply having no data in some env.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Hardcoded judgment constants.
#
# These shape the dethrone rule itself and only change with code releases —
# not via system_config. Per-env config may override ``margin`` and
# ``not_worse_tolerance`` for dataset-specific tuning, but the defaults
# below are the source of truth.
# ---------------------------------------------------------------------------

#: Default per-env additive margin: challenger's mean must exceed champion's
#: by this absolute amount to count as "dominant" in that env.
DEFAULT_MARGIN: float = 0.03

#: Default multiplicative regression tolerance: an env is "not_worse" iff
#: ``chal_avg >= champ_avg * (1 - tol)``. 0.0 means any regression at all
#: pushes the env to "worse".
DEFAULT_NOT_WORSE_TOLERANCE: float = 0.02

#: How many envs the challenger must be "dominant" in to dethrone the
#: champion. The latest old-system setting is partial Pareto: dominate at
#: least one env and stay not_worse in the rest.
WIN_MIN_DOMINANT_ENVS: int = 1

#: Envs whose score is an unbounded, sign-crossing quantity (e.g. distill's
#: advantage-weighted CE: roughly [-0.7, +0.6], mean near 0, can go negative)
#: rather than a natural [0, 1] success rate. For these the multiplicative
#: not_worse band (``champ * (1 - tol)``) is meaningless — when ``champ`` is
#: near 0 the band collapses, and when ``champ`` is negative the inequality
#: flips. They use an ADDITIVE not_worse band (``champ - margin``) instead.
ADDITIVE_MARGIN_ENVS: frozenset = frozenset({"DISTILL-V2"})

#: Additive margin for ``ADDITIVE_MARGIN_ENVS`` (both the dominant threshold
#: and the not_worse band ``champ ± margin``). Recalibrated 2026-07 after
#: distill-v2 switched to the bounded ``softmax_advantage`` scorer: the new
#: per-cell std grew ~1.7x (0.26 -> 0.44), so the champion-vs-challenger
#: common-cell mean difference at typical M~33 has SE ~0.1. 0.1 keeps the
#: band safely above that noise floor so distill (a noisy, sign-crossing
#: quantity) doesn't spuriously flag ties as regressions or let noise win an
#: env. The old 0.02 was calibrated on the unbounded reward_weighted_ce scale.
DEFAULT_ADDITIVE_MARGIN: float = 0.1


def not_worse_lower_bound(
    champ_basis: float, env: str, *, tolerance: float, additive_margin: float,
) -> float:
    """Lower score below which the challenger regresses (becomes ENV_WORSE).

    Additive for ``ADDITIVE_MARGIN_ENVS`` (sign-crossing scores), multiplicative
    otherwise (natural [0, 1] success rates). Single source of truth shared by
    the comparator, the scheduler's early-regression pre-check, and the rank UI
    so all three stay bit-consistent.
    """
    if env in ADDITIVE_MARGIN_ENVS:
        return champ_basis - additive_margin
    return champ_basis * (1.0 - tolerance)


@dataclass(frozen=True)
class EnvComparisonConfig:
    env: str
    margin: float
    min_tasks_per_env: int
    # Multiplicative tolerance: challenger is "not_worse" iff
    # ``chal_avg >= champ_avg * (1 - not_worse_tolerance)``. 0.0 (default)
    # means any regression at all flips the env to "worse". Used only for
    # natural [0,1] envs; ``ADDITIVE_MARGIN_ENVS`` ignore it.
    not_worse_tolerance: float = 0.0
    # Additive not_worse band (``champ - additive_margin``) for sign-crossing
    # envs in ``ADDITIVE_MARGIN_ENVS``; ignored by multiplicative envs.
    additive_margin: float = DEFAULT_ADDITIVE_MARGIN


# Per-env classification.
ENV_DOMINANT = "dominant"
ENV_NOT_WORSE = "not_worse"
ENV_WORSE = "worse"


@dataclass
class EnvComparison:
    env: str
    champion_avg: Optional[float]
    challenger_avg: Optional[float]
    champion_n: int
    challenger_n: int
    delta: Optional[float]
    margin: float
    not_worse_tolerance: float
    verdict: str  # ENV_DOMINANT | ENV_NOT_WORSE | ENV_WORSE
    reason: str   # detail for outcome.per_env, useful for blocking diagnostics


@dataclass
class ComparisonResult:
    winner: str  # "challenger" | "champion"
    reason: str
    dominant_count: int = 0
    not_worse_count: int = 0
    worse_count: int = 0
    per_env: List[EnvComparison] = field(default_factory=list)

    def first_blocking_env(self) -> Optional[EnvComparison]:
        return next((e for e in self.per_env if e.verdict == ENV_WORSE), None)


class WindowComparator:
    @staticmethod
    def _mean_or_none(values: List[float]) -> Optional[float]:
        return mean(values) if values else None

    def compare(
        self,
        champion_scores: Dict[str, Dict[int, float]],
        challenger_scores: Dict[str, Dict[int, float]],
        env_configs: Dict[str, EnvComparisonConfig],
        min_dominant_envs: int = 0,
    ) -> ComparisonResult:
        """Return the comparison outcome.

        ``min_dominant_envs == 0`` is the strict Pareto path (every env
        must be dominant); ``> 0`` is partial Pareto (N dominant + rest
        not_worse). The scheduler passes the
        :data:`WIN_MIN_DOMINANT_ENVS` module constant in production.
        """
        per_env: List[EnvComparison] = []

        for env, cfg in env_configs.items():
            champ_map = champion_scores.get(env, {})
            chal_map = challenger_scores.get(env, {})
            champ_vals = list(champ_map.values())
            chal_vals = list(chal_map.values())
            champ_avg = self._mean_or_none(champ_vals)
            chal_avg = self._mean_or_none(chal_vals)

            # Challenger sample-count gate.
            if len(chal_vals) < cfg.min_tasks_per_env:
                per_env.append(
                    EnvComparison(
                        env=env,
                        champion_avg=champ_avg,
                        challenger_avg=chal_avg,
                        champion_n=len(champ_vals),
                        challenger_n=len(chal_vals),
                        delta=None,
                        margin=cfg.margin,
                        not_worse_tolerance=cfg.not_worse_tolerance,
                        verdict=ENV_WORSE,
                        reason="insufficient_challenger_samples",
                    )
                )
                continue

            # Champion missing data → treat its mean as 0.0 so a sample-
            # sufficient challenger doesn't get blocked by the absence.
            # This matches the original Pareto stage-2 fallback behavior.
            champ_basis = champ_avg if champ_avg is not None else 0.0
            delta = chal_avg - champ_basis  # type: ignore[operator]

            if chal_avg > champ_basis + cfg.margin + 1e-9:
                verdict = ENV_DOMINANT
                reason = "challenger_better"
            else:
                # not_worse threshold: additive band for sign-crossing envs
                # (distill), multiplicative for natural [0,1] success-rate envs.
                # See ``not_worse_lower_bound``.
                not_worse_threshold = not_worse_lower_bound(
                    champ_basis, env,
                    tolerance=cfg.not_worse_tolerance,
                    additive_margin=cfg.additive_margin,
                )
                if chal_avg >= not_worse_threshold - 1e-9:
                    verdict = ENV_NOT_WORSE
                    reason = "tie_within_tolerance"
                else:
                    verdict = ENV_WORSE
                    reason = "regressed_beyond_tolerance"

            per_env.append(
                EnvComparison(
                    env=env,
                    champion_avg=champ_avg,
                    challenger_avg=chal_avg,
                    champion_n=len(champ_vals),
                    challenger_n=len(chal_vals),
                    delta=delta,
                    margin=cfg.margin,
                    not_worse_tolerance=cfg.not_worse_tolerance,
                    verdict=verdict,
                    reason=reason,
                )
            )

        if not per_env:
            return ComparisonResult(
                winner="champion", reason="no_envs_configured", per_env=[],
            )

        dominant = sum(1 for e in per_env if e.verdict == ENV_DOMINANT)
        not_worse = sum(1 for e in per_env if e.verdict != ENV_WORSE)
        worse = sum(1 for e in per_env if e.verdict == ENV_WORSE)
        n_total = len(per_env)

        # Decision.
        if min_dominant_envs <= 0:
            # Strict Pareto: dominant in every env.
            winner = "challenger" if dominant == n_total else "champion"
            reason = "all_envs_dominant" if winner == "challenger" else "not_all_envs_dominant"
        else:
            # Partial Pareto: ≥N dominant + the rest not_worse (no regress).
            if dominant >= min_dominant_envs and not_worse == n_total:
                winner = "challenger"
                reason = f"dominant_in_{dominant}_of_{n_total}_envs"
            else:
                winner = "champion"
                if dominant < min_dominant_envs:
                    reason = f"insufficient_dominant_envs:{dominant}<{min_dominant_envs}"
                else:
                    blocking = next(e for e in per_env if e.verdict == ENV_WORSE)
                    reason = f"regressed_in_env:{blocking.env}:{blocking.reason}"

        return ComparisonResult(
            winner=winner,
            reason=reason,
            dominant_count=dominant,
            not_worse_count=not_worse,
            worse_count=worse,
            per_env=per_env,
        )
