"""
Pareto Dominance Comparison

Single comparison primitive used by ChampionChallenge for champion
challenges, champion dominance detection, and pairwise Pareto dominance.

Margin scales linearly from WIN_MARGIN_START to WIN_MARGIN_END
as checkpoint progresses. Supports strict Pareto (all envs) and
partial Pareto (N of M envs + not worse in rest).
"""

from typing import Dict, List, Optional
from affine.src.scorer.models import MinerData, ParetoComparison
from affine.src.scorer.config import ScorerConfig


class Stage2ParetoFilter:

    def __init__(self, config: ScorerConfig = ScorerConfig):
        self.config = config

    def _compare_miners(
        self,
        miner_a: MinerData,
        miner_b: MinerData,
        envs: List[str],
        label: str,
        min_dominant_envs: Optional[int] = None,
        checkpoint: Optional[int] = None,
    ) -> ParetoComparison:
        """Pareto comparison: A vs B across `envs`. A is the incumbent.

        min_dominant_envs: override config (0=strict, N=partial).
        checkpoint: current CP, used to interpolate margin between
            WIN_MARGIN_START (early, lenient) and WIN_MARGIN_END
            (dethrone, strict). None = use MARGIN_START.
        """
        min_dom = min_dominant_envs if min_dominant_envs is not None \
            else self.config.WIN_MIN_DOMINANT_ENVS

        # Determine margin:
        # - Champion challenge: interpolate between MARGIN_START and MARGIN_END by CP
        # - Pairwise / champion dominance (no checkpoint): use fixed PARETO_MARGIN
        if checkpoint is not None:
            m_start = self.config.WIN_MARGIN_START
            m_end = self.config.WIN_MARGIN_END
            cp_first = self.config.CHAMPION_WARMUP_CHECKPOINTS + 1
            cp_last = self.config.CHAMPION_DETHRONE_MIN_CHECKPOINT
            if cp_last > cp_first:
                t = min(1.0, max(0.0, (checkpoint - cp_first) / (cp_last - cp_first)))
            else:
                t = 1.0
            margin = m_start + t * (m_end - m_start)
            not_worse_tol = self.config.WIN_NOT_WORSE_TOLERANCE
        else:
            margin = self.config.PARETO_MARGIN
            not_worse_tol = 1e-9  # No tolerance for pairwise

        suspicious_multiplier = self.config.PARETO_SUSPICIOUS_MARGIN_MULTIPLIER

        def _effective_margin(actor: MinerData, target: MinerData) -> float:
            if (
                getattr(actor, "anticopy_status", "clean") == "suspicious"
                and getattr(actor, "anticopy_target_uid", None) == target.uid
            ):
                return margin * suspicious_multiplier
            return margin

        margin_b_over_a = _effective_margin(miner_b, miner_a)
        margin_a_over_b = _effective_margin(miner_a, miner_b)
        b_dominant = 0   # envs where B beats A by margin
        b_not_worse = 0  # envs where B >= A (no margin required)
        a_dominant = 0
        a_not_worse = 0
        env_details: Dict[str, Dict] = {}
        n_compared = 0

        for env in envs:
            es_a = miner_a.env_scores.get(env)
            es_b = miner_b.env_scores.get(env)
            if not es_a or not es_b:
                env_details[env] = {"winner": None, "reason": "missing_env"}
                continue

            common = set(es_a.all_task_scores) & set(es_b.all_task_scores)
            if not common:
                env_details[env] = {"winner": None, "reason": "no_common_tasks"}
                continue

            score_a = sum(es_a.all_task_scores[t] for t in common) / len(common)
            score_b = sum(es_b.all_task_scores[t] for t in common) / len(common)
            n_compared += 1

            # B exceeds A by margin → B dominant in this env
            if score_b > score_a + margin_b_over_a + 1e-9:
                b_dominant += 1
                b_not_worse += 1
                winner = "B"
            elif score_b >= score_a * (1 - not_worse_tol) - 1e-9:
                # B is not worse (within tolerance of champion's score)
                b_not_worse += 1
                a_not_worse += 1
                winner = "A"  # incumbent advantage: tie goes to A
            else:
                # B is worse
                a_not_worse += 1
                winner = "A"

            # A exceeds B by margin → A dominant in this env
            if score_a > score_b + margin_a_over_b + 1e-9:
                a_dominant += 1

            env_details[env] = {
                "a_score": score_a,
                "b_score": score_b,
                "margin_b_over_a": round(margin_b_over_a, 4),
                "margin_a_over_b": round(margin_a_over_b, 4),
                "threshold": score_a + margin_b_over_a,
                "winner": winner,
                "common_tasks": len(common),
            }

        # "A wins env" = B did not exceed A by margin (incumbent advantage).
        # Under strict Pareto (min_dom=0): A dominates B iff A wins all envs
        # (i.e., B never exceeded margin in any env).
        a_wins = n_compared - b_dominant  # envs where B did NOT beat A by margin

        if min_dom > 0 and n_compared > 0:
            # Partial Pareto: B needs at least min_dom envs exceeding margin,
            # AND not be worse (< score_a) in any env.
            b_dominates_a = (b_dominant >= min_dom and b_not_worse == n_compared)
            a_dominates_b = (a_dominant >= min_dom and a_not_worse == n_compared)
        else:
            # Strict Pareto (original behavior): B must exceed margin in ALL,
            # A dominates if B never exceeded margin in any env.
            b_dominates_a = (b_dominant == n_compared and n_compared > 0)
            a_dominates_b = (a_wins == n_compared and n_compared > 0)

        return ParetoComparison(
            miner_a_uid=miner_a.uid,
            miner_b_uid=miner_b.uid,
            label=label,
            a_dominates_b=a_dominates_b,
            b_dominates_a=b_dominates_a,
            env_comparisons=env_details,
        )
