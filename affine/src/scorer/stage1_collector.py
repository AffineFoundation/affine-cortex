"""
Stage 1: Data Collection

Parses scoring data from the API into MinerData objects. Computes only the
fields needed for downstream stages — no validity gates, no thresholds.

`avg_score` is computed from the most recent N×window samples (where
N = CHAMPION_DETHRONE_MIN_CHECKPOINT). This caps how
far back display scores can drift for long-running champions, while
challengers — whose history never reaches that cap — naturally see the
full set. `all_task_scores` keeps the full historical task→score map and
is what Pareto comparisons use.
"""

from typing import Dict, List, Any, Optional
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
        env_sampling_counts: Optional[Dict[str, int]] = None,
    ) -> Stage1Output:
        """Parse API scoring_data into MinerData objects.

        Args:
            scoring_data: API /samples/scoring response
            environments: enabled environments
            env_sampling_counts: {env_name: window_size}; used to cap
                avg_score to the most recent N × window_size samples.
                If missing for an env, avg_score uses full history.
        """
        env_sampling_counts = env_sampling_counts or {}
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
                window_size = env_sampling_counts.get(env_name)
                miner.env_scores[env_name] = self._build_env_score(
                    env_data.get(env_name, {}), env_name, window_size)

            miners[uid] = miner

        logger.info(f"Stage 1: collected {len(miners)} miners")
        return Stage1Output(miners=miners, environments=environments)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _build_env_score(
        self,
        env_info: Dict[str, Any],
        env_name: str,
        window_size: Optional[int],
    ) -> EnvScore:
        """Build an EnvScore from one env's API data."""
        if not env_info:
            return EnvScore(avg_score=0.0, sample_count=0, completeness=0.0)

        all_samples = env_info.get('all_samples', [])
        completed_count = env_info.get('completed_count', 0)
        completeness = env_info.get('completeness', 0.0)

        # Full historical task→score (used by Pareto comparisons).
        # If a task_id appears multiple times, the last one written wins —
        # this is fine for Pareto since it only needs *some* score per task.
        all_task_scores: Dict[int, float] = {}
        for s in all_samples:
            tid = s.get('task_id')
            if tid is not None:
                all_task_scores[int(tid)] = s.get('score', 0.0)

        historical_count = len(all_task_scores)

        # Display avg: mean over the most recent N × window_size samples
        # (deduped by task_id, keeping the newest score per task). For
        # challengers historical_count < cap → recent == full history.
        # For long-running champions this prevents stale early scores
        # from dragging the displayed average.
        avg = self._recent_avg(all_samples, window_size)

        # Apply env-specific normalization if configured
        ranges = self.config.ENV_SCORE_RANGES.get(env_name)
        if ranges:
            lo, hi = ranges
            avg = (avg - lo) / (hi - lo)
            all_task_scores = {tid: (s - lo) / (hi - lo) for tid, s in all_task_scores.items()}

        return EnvScore(
            avg_score=avg,
            sample_count=completed_count,
            historical_count=historical_count,
            completeness=completeness,
            all_task_scores=all_task_scores,
        )

    def _recent_avg(
        self,
        all_samples: List[Dict[str, Any]],
        window_size: Optional[int],
    ) -> float:
        """Mean score over the most recent N×window_size unique tasks.

        - Sort samples by timestamp descending.
        - Walk the list, keeping the first (newest) score per task_id.
        - Stop once we've collected `cap` unique tasks (cap = N×window_size).
        - If `window_size` is unknown, use the full history.
        """
        if not all_samples:
            return 0.0

        cap: Optional[int] = None
        if window_size and window_size > 0:
            cap = self.config.CHAMPION_DETHRONE_MIN_CHECKPOINT * window_size

        sorted_samples = sorted(
            all_samples,
            key=lambda s: s.get('timestamp') or 0,
            reverse=True,
        )

        seen: set = set()
        recent_scores: List[float] = []
        for s in sorted_samples:
            tid = s.get('task_id')
            if tid is None:
                continue
            tid = int(tid)
            if tid in seen:
                continue
            seen.add(tid)
            recent_scores.append(s.get('score', 0.0))
            if cap is not None and len(recent_scores) >= cap:
                break

        if not recent_scores:
            return 0.0
        return sum(recent_scores) / len(recent_scores)
