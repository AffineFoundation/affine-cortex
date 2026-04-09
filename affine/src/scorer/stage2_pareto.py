"""
Pareto Dominance Comparison

Provides the _compare_miners method used by ChampionChallenge to determine
if a challenger Pareto-dominates the champion across all environments.
"""

from typing import Dict, List
from affine.src.scorer.models import MinerData, ParetoComparison
from affine.src.scorer.config import ScorerConfig
from affine.src.scorer.utils import calculate_required_score


class Stage2ParetoFilter:

    def __init__(self, config: ScorerConfig = ScorerConfig):
        self.config = config

    def _compare_miners(
        self,
        miner_a: MinerData,
        miner_b: MinerData,
        envs: List[str],
        subset_key: str
    ) -> ParetoComparison:
        """Compare two miners using Pareto dominance test.

        A dominates B if A wins ALL environments.
        B dominates A if B wins ALL environments.
        B wins an environment if B's score exceeds threshold(A).
        """
        a_wins_count = 0
        b_wins_count = 0
        env_comparisons = {}

        for env in envs:
            env_score_a = miner_a.env_scores[env]
            env_score_b = miner_b.env_scores[env]

            # Align scores to common tasks
            common_tasks = set(env_score_a.all_task_scores) & set(env_score_b.all_task_scores)
            if common_tasks:
                score_a = sum(env_score_a.all_task_scores[t] for t in common_tasks) / len(common_tasks)
                score_b = sum(env_score_b.all_task_scores[t] for t in common_tasks) / len(common_tasks)
                env_threshold_config = self.config.ENV_THRESHOLD_CONFIGS.get(env, {})
                threshold = calculate_required_score(
                    score_a,
                    len(common_tasks),
                    env_threshold_config.get('z_score', self.config.Z_SCORE),
                    env_threshold_config.get('min_improvement', self.config.MIN_IMPROVEMENT),
                    env_threshold_config.get('max_improvement', self.config.MAX_IMPROVEMENT),
                )
            else:
                score_a = env_score_a.avg_score
                score_b = env_score_b.avg_score
                threshold = env_score_a.threshold

            b_wins_env = score_b > (threshold + 1e-9)

            if b_wins_env:
                b_wins_count += 1
            else:
                a_wins_count += 1

            env_comparisons[env] = {
                "a_score": score_a,
                "b_score": score_b,
                "threshold": threshold,
                "b_beats_threshold": b_wins_env,
                "winner": "B" if b_wins_env else "A",
                "aligned_tasks": len(common_tasks) if common_tasks else None,
            }

        return ParetoComparison(
            miner_a_uid=miner_a.uid,
            miner_b_uid=miner_b.uid,
            subset_key=subset_key,
            a_dominates_b=(a_wins_count == len(envs)),
            b_dominates_a=(b_wins_count == len(envs)),
            env_comparisons=env_comparisons
        )
