"""
Pareto Dominance Comparison

Single comparison primitive used by ChampionChallenge for both pairwise
filtering and the champion-vs-challenger checks.

Miner B wins environment E against miner A if B's score on common tasks
exceeds A's score by more than PARETO_MARGIN. B dominates A iff B wins
ALL environments. The margin is fixed — sample-size adequacy is enforced
upstream by the checkpoint mechanism.
"""

from typing import Dict, List
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
    ) -> ParetoComparison:
        """Pareto comparison: A vs B across `envs`. A is the incumbent."""
        margin = self.config.PARETO_MARGIN
        a_wins = b_wins = 0
        env_details: Dict[str, Dict] = {}

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
            threshold = score_a + margin

            if score_b > threshold + 1e-9:
                b_wins += 1
                winner = "B"
            else:
                a_wins += 1
                winner = "A"
            env_details[env] = {
                "a_score": score_a,
                "b_score": score_b,
                "threshold": threshold,
                "winner": winner,
                "common_tasks": len(common),
            }

        n = len(envs)
        return ParetoComparison(
            miner_a_uid=miner_a.uid,
            miner_b_uid=miner_b.uid,
            label=label,
            a_dominates_b=(a_wins == n),
            b_dominates_a=(b_wins == n),
            env_comparisons=env_details,
        )
