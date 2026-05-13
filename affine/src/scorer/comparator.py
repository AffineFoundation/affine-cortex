"""
Window comparator — Pareto-with-tolerance rule.

For each env the challenger lands in one of three buckets vs the champion:

  - **dominant**  : ``chal_avg ≥ champ_avg + margin`` — strictly better by
                    at least the per-env margin.
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
DEFAULT_MARGIN: float = 0.01

#: Default multiplicative regression tolerance: an env is "not_worse" iff
#: ``chal_avg >= champ_avg * (1 - tol)``. 0.0 means any regression at all
#: pushes the env to "worse".
DEFAULT_NOT_WORSE_TOLERANCE: float = 0.0

#: How many envs the challenger must be "dominant" in to dethrone the
#: champion. 0 ⇒ strict Pareto (must dominate every env). >0 ⇒ partial
#: Pareto (dominate N envs + not_worse in the rest).
WIN_MIN_DOMINANT_ENVS: int = 0


@dataclass(frozen=True)
class EnvComparisonConfig:
    env: str
    margin: float
    min_tasks_per_env: int
    # Multiplicative tolerance: challenger is "not_worse" iff
    # ``chal_avg >= champ_avg * (1 - not_worse_tolerance)``. 0.0 (default)
    # means any regression at all flips the env to "worse".
    not_worse_tolerance: float = 0.0


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

            if chal_avg >= champ_basis + cfg.margin:
                verdict = ENV_DOMINANT
                reason = "challenger_better"
            else:
                # not_worse threshold is multiplicative against champion's
                # score: chal_avg >= champ * (1 - tol). When champ_basis is
                # negative (some envs allow negative scores), keep the same
                # formula — the caller picks ``not_worse_tolerance`` per env
                # and is responsible for the score scale.
                not_worse_threshold = champ_basis * (1.0 - cfg.not_worse_tolerance)
                if chal_avg >= not_worse_threshold:
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
