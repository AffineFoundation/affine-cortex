"""
Stage 1: Data Collection

Parses scoring data from the API into MinerData objects. Computes only the
fields needed for downstream stages — no validity gates, no thresholds.
The avg_score is computed from the current sampling window for display
and cold-start ranking; all_task_scores stores the full historical data
used by Pareto comparisons.
"""

from typing import Dict, List, Any
from affine.src.scorer.models import MinerData, EnvScore, Stage1Output
from affine.src.scorer.config import ScorerConfig

from affine.core.setup import logger


class Stage1Collector:

    def __init__(self, config: ScorerConfig = ScorerConfig):
        self.config = config

    def collect(
        self,
        scoring_data: Dict[str, Any],
        environments: List[str],
    ) -> Stage1Output:
        """Parse API scoring_data into MinerData objects."""
        if not isinstance(scoring_data, dict):
            raise RuntimeError(f"Invalid scoring_data type: {type(scoring_data)}")

        if "success" in scoring_data and scoring_data.get("success") is False:
            error_msg = scoring_data.get("error", "Unknown error")
            raise RuntimeError(f"API error response: {error_msg}")

        if not scoring_data:
            logger.warning("Received empty scoring_data")
            return Stage1Output(miners={}, environments=environments)

        logger.info(f"Stage 1: collecting data for {len(scoring_data)} miners")
        miners: Dict[int, MinerData] = {}

        for key, entry in scoring_data.items():
            uid = entry.get('uid')
            try:
                uid = int(uid) if uid is not None else None
            except (ValueError, TypeError):
                logger.warning(f"Invalid UID for key {key}: {uid}")
                continue
            if uid is None:
                continue

            hotkey = entry.get('hotkey')
            model_revision = entry.get('model_revision')
            model_repo = entry.get('model_repo')
            first_block = entry.get('first_block')
            if not hotkey or not model_revision or not model_repo:
                logger.warning(f"UID {uid}: missing required field")
                continue

            miner = MinerData(
                uid=uid,
                hotkey=hotkey,
                model_revision=model_revision,
                model_repo=model_repo,
                first_block=first_block,
            )

            env_data = entry.get('env', {})
            for env_name in environments:
                miner.env_scores[env_name] = self._build_env_score(
                    env_data.get(env_name, {}), env_name)

            miners[uid] = miner

        logger.info(f"Stage 1: collected {len(miners)} miners")
        return Stage1Output(miners=miners, environments=environments)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _build_env_score(self, env_info: Dict[str, Any], env_name: str) -> EnvScore:
        """Build an EnvScore from one env's API data."""
        if not env_info:
            return EnvScore(avg_score=0.0, sample_count=0, completeness=0.0)

        all_samples = env_info.get('all_samples', [])
        sampling_ids = set(env_info.get('sampling_task_ids', []))
        completed_count = env_info.get('completed_count', 0)
        completeness = env_info.get('completeness', 0.0)

        # Full historical task→score (used by Pareto comparisons)
        all_task_scores: Dict[int, float] = {}
        for s in all_samples:
            tid = s.get('task_id')
            if tid is not None:
                all_task_scores[int(tid)] = s.get('score', 0.0)

        # Current window average (display + cold start)
        window_scores = [s for tid, s in all_task_scores.items() if tid in sampling_ids]
        avg = sum(window_scores) / len(window_scores) if window_scores else 0.0

        # Apply env-specific normalization if configured
        ranges = self.config.ENV_SCORE_RANGES.get(env_name)
        if ranges:
            lo, hi = ranges
            avg = (avg - lo) / (hi - lo)
            all_task_scores = {tid: (s - lo) / (hi - lo) for tid, s in all_task_scores.items()}

        return EnvScore(
            avg_score=avg,
            sample_count=completed_count,
            completeness=completeness,
            all_task_scores=all_task_scores,
        )
