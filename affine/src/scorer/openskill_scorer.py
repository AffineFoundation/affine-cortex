"""
OpenSkill (PlackettLuce) Per-Task Rating System

Rates miners on each individual task, then aggregates across environments
via z-score → sigmoid → geometric mean to produce final weights.
"""

import math
import time
from typing import Dict, List, Set, Any, Optional, Tuple
from collections import defaultdict
from openskill.models import PlackettLuce

from affine.src.scorer.openskill_config import OpenSkillConfig
from affine.src.scorer.config import ScorerConfig
from affine.src.scorer.utils import normalize_weights, apply_min_threshold
from affine.database.dao.openskill_ratings import OpenSkillRatingsDAO
from affine.database.dao.openskill_matches import OpenSkillMatchesDAO
from affine.core.setup import logger


class OpenSkillScorer:
    """OpenSkill per-task rating system.

    Flow:
    1. process_rotated_tasks(): Update ratings for tasks leaving the sampling window
    2. compute_weights(): Calculate final weights from current ratings
    """

    def __init__(
        self,
        config: OpenSkillConfig = OpenSkillConfig,
        ratings_dao: Optional[OpenSkillRatingsDAO] = None,
        matches_dao: Optional[OpenSkillMatchesDAO] = None,
    ):
        self.config = config
        self.ratings_dao = ratings_dao or OpenSkillRatingsDAO()
        self.matches_dao = matches_dao or OpenSkillMatchesDAO()
        self.model = PlackettLuce(
            mu=config.MU_INIT,
            sigma=config.SIGMA_INIT,
            tau=config.TAU,
        )
        # In-memory cache: {(hotkey#revision, env): Rating}
        self._cache: Dict[Tuple[str, str], Any] = {}

    # ------------------------------------------------------------------
    # Core: process tasks and update ratings
    # ------------------------------------------------------------------

    async def process_rotated_tasks(
        self,
        env: str,
        task_scores: Dict[int, Dict[str, float]],
    ) -> int:
        """Process tasks that rotated out of the sampling window.

        Args:
            env: Environment name
            task_scores: {task_id: {hotkey#revision: score}}
                Only include tasks that have left the window.

        Returns:
            Number of tasks processed (excluding skipped/already processed)
        """
        if not task_scores:
            return 0

        # Find already-processed tasks
        processed_ids = await self.matches_dao.get_processed_task_ids(env)
        new_tasks = {
            tid: scores for tid, scores in task_scores.items()
            if tid not in processed_ids
        }

        if not new_tasks:
            logger.info(f"OpenSkill {env}: no new tasks to process")
            return 0

        logger.info(f"OpenSkill {env}: processing {len(new_tasks)} new tasks "
                     f"({len(processed_ids)} already processed)")

        # Load existing ratings into cache
        await self._load_ratings_for_env(env, new_tasks)

        # Process tasks in order (by task_id as proxy for chronological)
        match_records = []
        updated_keys: Set[Tuple[str, str]] = set()
        processed_count = 0

        for task_id in sorted(new_tasks.keys()):
            scores = new_tasks[task_id]
            record = self._process_single_task(env, task_id, scores)
            match_records.append(record)
            if not record['skipped']:
                processed_count += 1
                for p in record['participants']:
                    updated_keys.add((f"{p['hotkey']}#{p['revision']}", env))

        # Batch save match records
        await self.matches_dao.batch_save_matches(match_records)

        # Batch save updated ratings
        await self._flush_ratings(env, updated_keys)

        logger.info(f"OpenSkill {env}: {processed_count} tasks rated, "
                     f"{len(new_tasks) - processed_count} skipped")
        return processed_count

    def _process_single_task(
        self,
        env: str,
        task_id: int,
        scores: Dict[str, float],
    ) -> Dict[str, Any]:
        """Process a single task and update in-memory ratings.

        Returns match record dict ready for DB write.
        """
        base_record = {
            'pk': f"ENV#{env}",
            'sk': f"TASK#{task_id}",
            'env': env,
            'task_id': task_id,
            'processed_at': int(time.time()),
            'ttl': self.matches_dao.get_ttl(self.config.MATCH_TTL_DAYS),
        }

        # Check minimum participants
        if len(scores) < self.config.MIN_PARTICIPANTS:
            return {
                **base_record,
                'participants': [],
                'n_participants': len(scores),
                'skipped': True,
                'skip_reason': f'participants={len(scores)} < {self.config.MIN_PARTICIPANTS}',
            }

        # Check all-tie
        unique_scores = set(round(s, 3) for s in scores.values())
        if len(unique_scores) == 1:
            return {
                **base_record,
                'participants': [],
                'n_participants': len(scores),
                'skipped': True,
                'skip_reason': 'all_tie',
            }

        # Rank participants (higher score = lower rank number)
        sorted_parts = sorted(scores.items(), key=lambda x: -x[1])
        ranks = []
        prev_score, prev_rank = None, 0
        for i, (_, score) in enumerate(sorted_parts):
            if round(score, 3) != prev_score:
                prev_rank = i + 1
                prev_score = round(score, 3)
            ranks.append(prev_rank)

        # Build teams and record before-state
        teams = []
        keys = []
        before_states = []
        for miner_key, _ in sorted_parts:
            cache_key = (miner_key, env)
            if cache_key not in self._cache:
                self._cache[cache_key] = self.model.rating(name=miner_key)
            rating = self._cache[cache_key]
            teams.append([rating])
            keys.append(miner_key)
            before_states.append((rating.mu, rating.sigma))

        # Rate
        try:
            new_teams = self.model.rate(teams, ranks=ranks)
        except Exception as e:
            logger.warning(f"OpenSkill rate() failed for {env} task {task_id}: {e}")
            return {
                **base_record,
                'participants': [],
                'n_participants': len(scores),
                'skipped': True,
                'skip_reason': f'rate_error: {e}',
            }

        # Update cache and build participant records
        participants = []
        for i, (miner_key, new_team) in enumerate(zip(keys, new_teams)):
            new_rating = new_team[0]
            self._cache[(miner_key, env)] = new_rating
            mu_before, sigma_before = before_states[i]

            parts = miner_key.split('#', 1)
            hotkey, revision = parts[0], parts[1] if len(parts) > 1 else ''

            participants.append({
                'hotkey': hotkey,
                'revision': revision,
                'score': sorted_parts[i][1],
                'rank': ranks[i],
                'mu_before': round(mu_before, 4),
                'mu_after': round(new_rating.mu, 4),
                'sigma_before': round(sigma_before, 4),
                'sigma_after': round(new_rating.sigma, 4),
            })

        return {
            **base_record,
            'participants': participants,
            'n_participants': len(participants),
            'skipped': False,
        }

    # ------------------------------------------------------------------
    # Weight computation
    # ------------------------------------------------------------------

    async def compute_weights(
        self,
        environments: List[str],
        env_window_sizes: Dict[str, int],
        miner_task_counts: Dict[str, Dict[str, int]],
    ) -> Dict[str, float]:
        """Compute final weights from current ratings.

        Args:
            environments: List of scoring environment names
            env_window_sizes: {env: sampling_list size}
            miner_task_counts: {hotkey#revision: {env: total_completed_tasks}}

        Returns:
            {hotkey#revision: normalized_weight}
        """
        # Load all ratings
        all_ratings = await self.ratings_dao.get_all_ratings()

        if not all_ratings:
            logger.warning("OpenSkill: no ratings found")
            return {}

        # Compute ordinals per env
        env_ordinals: Dict[str, Dict[str, float]] = {}
        for env in environments:
            env_ordinals[env] = {}
            for miner_key, env_ratings in all_ratings.items():
                if env in env_ratings:
                    r = env_ratings[env]
                    env_ordinals[env][miner_key] = (
                        r['mu'] - self.config.ORDINAL_Z * r['sigma']
                    )

        # Determine qualified miners per env
        env_qualified: Dict[str, Set[str]] = {}
        for env in environments:
            threshold = int(
                env_window_sizes.get(env, 0) * self.config.WINDOW_QUALIFICATION_RATIO
            )
            env_qualified[env] = set()
            for miner_key in env_ordinals.get(env, {}):
                task_count = miner_task_counts.get(miner_key, {}).get(env, 0)
                if task_count >= threshold:
                    env_qualified[env].add(miner_key)

        # z-score → sigmoid per env (over qualified miners only)
        env_sigmoid: Dict[str, Dict[str, float]] = {}
        for env in environments:
            qualified = env_qualified.get(env, set())
            ords = {
                k: v for k, v in env_ordinals.get(env, {}).items()
                if k in qualified
            }
            if len(ords) < 2:
                continue

            vals = list(ords.values())
            mean_o = sum(vals) / len(vals)
            std_o = (sum((v - mean_o) ** 2 for v in vals) / len(vals)) ** 0.5
            if std_o < 1e-6:
                continue

            env_sigmoid[env] = {}
            for miner_key, ordinal in ords.items():
                z = (ordinal - mean_o) / std_o
                env_sigmoid[env][miner_key] = 1.0 / (1.0 + math.exp(-z))

            n_q = len(ords)
            top3 = sorted(env_sigmoid[env].values(), reverse=True)[:3]
            logger.info(
                f"OpenSkill {env}: {n_q} qualified, "
                f"top-3 sigmoid sum={sum(top3):.3f}"
            )

        # Geometric mean across envs
        final: Dict[str, float] = {}
        for miner_key in all_ratings:
            vals = [
                env_sigmoid[env][miner_key]
                for env in environments
                if env in env_sigmoid and miner_key in env_sigmoid[env]
            ]
            if len(vals) < self.config.MIN_QUALIFIED_ENVS:
                continue
            # Geometric mean
            product = 1.0
            for v in vals:
                product *= max(v, 1e-9)
            final[miner_key] = product ** (1.0 / len(vals))

        # Rank by geometric mean, apply decay^(rank-1)
        # Reuse DECAY_FACTOR and threshold logic from ScorerConfig
        ranked = sorted(final.items(), key=lambda x: -x[1])
        decay = ScorerConfig.DECAY_FACTOR
        weights: Dict[str, float] = {}
        prev_gm, prev_rank = None, 0
        for i, (miner_key, gm) in enumerate(ranked):
            if round(gm, 8) != prev_gm:
                prev_rank = i + 1
                prev_gm = round(gm, 8)
            weights[miner_key] = decay ** (prev_rank - 1)

        # Normalize and apply min threshold (reuse from utils)
        weights = normalize_weights(weights)
        weights = apply_min_threshold(weights, threshold=ScorerConfig.MIN_WEIGHT_THRESHOLD)
        weights = normalize_weights(weights)

        if weights:
            top_w = sorted(weights.values(), reverse=True)
            logger.info(
                f"OpenSkill: {len(weights)} miners, "
                f"top1={top_w[0]*100:.1f}% top3={sum(top_w[:3])*100:.1f}%"
            )
        return weights

    # ------------------------------------------------------------------
    # Save weight snapshot
    # ------------------------------------------------------------------

    async def save_weight_snapshot(
        self,
        weights: Dict[str, float],
        environments: List[str],
    ):
        """Save weight snapshot to openskill_weights table for shadow analysis."""
        from affine.database.base_dao import BaseDAO
        from affine.database.schema import get_table_name

        now = int(time.time())
        participants = []
        for miner_key, weight in sorted(weights.items(), key=lambda x: -x[1]):
            parts = miner_key.split('#', 1)
            hotkey, revision = parts[0], parts[1] if len(parts) > 1 else ''
            participants.append({
                'hotkey': hotkey,
                'revision': revision,
                'weight': weight,
            })

        dao = BaseDAO.__new__(BaseDAO)
        dao.table_name = get_table_name("openskill_weights")

        item = dao._serialize({
            'pk': 'SNAPSHOT',
            'sk': f"TIME#{now}",
            'timestamp': now,
            'environments': environments,
            'participants': participants,
            'n_miners': len(participants),
            'config': self.config.to_dict(),
            'ttl': BaseDAO.get_ttl(self.config.MATCH_TTL_DAYS),
        })

        from affine.database.client import get_client
        client = get_client()
        await client.put_item(TableName=dao.table_name, Item=item)
        logger.info(f"OpenSkill: saved weight snapshot ({len(participants)} miners)")

    # ------------------------------------------------------------------
    # Cold start
    # ------------------------------------------------------------------

    async def cold_start(
        self,
        env_task_scores: Dict[str, Dict[int, Dict[str, float]]],
    ) -> Dict[str, int]:
        """Bootstrap ratings from historical data.

        Args:
            env_task_scores: {env: {task_id: {hotkey#revision: score}}}
                Pre-collected historical task scores, sorted chronologically.

        Returns:
            {env: tasks_processed}
        """
        results = {}
        for env, task_scores in env_task_scores.items():
            count = await self.process_rotated_tasks(env, task_scores)
            results[env] = count
            logger.info(f"OpenSkill cold start {env}: {count} tasks processed")
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _load_ratings_for_env(
        self,
        env: str,
        task_scores: Dict[int, Dict[str, float]],
    ):
        """Load existing ratings from DB into cache for all miners in task_scores."""
        # Collect all miner keys
        all_miners: Set[str] = set()
        for scores in task_scores.values():
            all_miners.update(scores.keys())

        # Load from DB
        all_ratings = await self.ratings_dao.get_all_ratings()

        for miner_key in all_miners:
            cache_key = (miner_key, env)
            if cache_key in self._cache:
                continue
            env_ratings = all_ratings.get(miner_key, {})
            if env in env_ratings:
                r = env_ratings[env]
                self._cache[cache_key] = self.model.create_rating(
                    [r['mu'], r['sigma']], name=miner_key
                )

    async def _flush_ratings(
        self,
        env: str,
        updated_keys: Set[Tuple[str, str]],
    ):
        """Write updated ratings from cache to DB."""
        ratings_to_save = []
        for miner_key, env_name in updated_keys:
            if env_name != env:
                continue
            cache_key = (miner_key, env)
            if cache_key not in self._cache:
                continue
            rating = self._cache[cache_key]
            parts = miner_key.split('#', 1)
            hotkey, revision = parts[0], parts[1] if len(parts) > 1 else ''
            ratings_to_save.append({
                'hotkey': hotkey,
                'revision': revision,
                'env': env,
                'mu': round(rating.mu, 6),
                'sigma': round(rating.sigma, 6),
            })

        if ratings_to_save:
            await self.ratings_dao.batch_save_ratings(ratings_to_save)
            logger.info(f"OpenSkill {env}: saved {len(ratings_to_save)} ratings")
