"""
Stage 3: ELO Rating Update and Weight Distribution

Uses per-round composite ranking (geometric mean of env avg_scores) to update
ELO ratings, then distributes weights by rating rank with decay.
"""

from typing import Dict, List, Any, Optional, Tuple
from affine.src.scorer.models import (
    MinerData,
    SubsetInfo,
    Stage3Output,
)
from affine.src.scorer.config import ScorerConfig
from affine.src.scorer.utils import geometric_mean
from affine.src.scorer.elo import update_ratings

from affine.core.setup import logger


class Stage3SubsetScorer:
    """Stage 3: ELO Rating Update and Weight Distribution.

    Responsibilities:
    1. Compute round ranks from geometric_mean(env avg_scores)
    2. Load previous ratings from MINER_STATS
    3. Update ELO ratings for participating miners
    4. Distribute weights by ELO rating rank with decay
    """

    def __init__(self, config: ScorerConfig = ScorerConfig):
        self.config = config
        self.decay_factor = config.DECAY_FACTOR
        self.geometric_mean_epsilon = config.GEOMETRIC_MEAN_EPSILON

    def score(
        self,
        miners: Dict[int, MinerData],
        environments: List[str],
        prev_ratings: Optional[Dict[str, Dict[str, Any]]] = None,
        current_block: int = 0,
    ) -> Stage3Output:
        """Execute ELO-based scoring.

        Args:
            miners: Stage 2 output
            environments: Environment list
            prev_ratings: {hotkey: {elo_rating, elo_rounds_played, elo_model_submit_block}}
                         from MINER_STATS, None = first round
            current_block: Current block height (for new revision model_submit_block)
        """
        logger.info(f"Stage 3: ELO update for {len(miners)} miners, {len(environments)} environments")

        # Step 1: Compute round ranks (geometric mean of env avg_scores)
        round_ranks = self._compute_round_ranks(miners, environments)

        # Step 2: Load previous ratings from MINER_STATS
        current_ratings, current_rounds, model_ages = self._load_prev_ratings(
            miners, prev_ratings, current_block
        )

        # Step 3: ELO update (only for miners in round_ranks)
        if round_ranks:
            elo_results = update_ratings(
                round_ranks=round_ranks,
                current_ratings={uid: current_ratings[uid] for uid in round_ranks},
                current_rounds={uid: current_rounds[uid] for uid in round_ranks},
                model_ages={uid: model_ages[uid] for uid in round_ranks},
                D=self.config.ELO_D,
                K_base=self.config.ELO_K_BASE,
                K_provisional=self.config.ELO_K_PROVISIONAL,
                provisional_rounds=self.config.ELO_PROVISIONAL_ROUNDS,
                alpha=self.config.ELO_SENIORITY_ALPHA,
            )

            # Write results to MinerData (enforce rating floor at BASE_RATING)
            for uid, (new_rating, change, new_rounds) in elo_results.items():
                miners[uid].elo_rating = max(new_rating, self.config.ELO_BASE_RATING)
                miners[uid].elo_rating_change = change
                miners[uid].elo_rounds_played = new_rounds

        # Non-ranked miners (absent/Pareto-filtered) keep previous rating unchanged
        for uid, miner in miners.items():
            if uid not in round_ranks:
                miner.elo_rating = current_ratings.get(uid, self.config.ELO_BASE_RATING)
                miner.elo_rating_change = 0.0
                miner.elo_rounds_played = current_rounds.get(uid, 0)

        # Step 4: Distribute weights by ELO rating (exclude Pareto-filtered)
        self._distribute_weights_by_rating(miners)

        ranked_count = len(round_ranks)
        logger.info(f"Stage 3: ELO updated for {ranked_count} ranked miners")

        return Stage3Output(miners=miners, subsets={})

    def _compute_round_ranks(
        self,
        miners: Dict[int, MinerData],
        environments: List[str],
    ) -> Dict[int, int]:
        """Compute round ranks from geometric_mean(env avg_scores).

        Excludes:
        - Miners that are not valid for scoring
        - Miners filtered by Pareto (filtered_subsets non-empty)
        """
        scores = {}
        for uid, miner in miners.items():
            if not miner.is_valid_for_scoring():
                continue
            # Pareto-filtered miners don't participate in ELO ranking
            if miner.filtered_subsets:
                continue
            # Miner must have valid scores in ALL subset environments
            all_valid = all(
                env in miner.env_scores and miner.env_scores[env].is_valid
                for env in environments
            )
            if not all_valid:
                continue
            env_scores = [miner.env_scores[env].avg_score for env in environments]
            scores[uid] = geometric_mean(env_scores, epsilon=self.geometric_mean_epsilon)

        # Rank with tied-rank handling
        sorted_uids = sorted(scores.keys(), key=lambda u: scores[u], reverse=True)
        ranks = {}
        prev_score = None
        prev_rank = 0
        for i, uid in enumerate(sorted_uids):
            if scores[uid] != prev_score:
                prev_rank = i + 1
                prev_score = scores[uid]
            ranks[uid] = prev_rank
        return ranks

    def _load_prev_ratings(
        self,
        miners: Dict[int, MinerData],
        prev_ratings: Optional[Dict[str, Dict[str, Any]]],
        current_block: int,
    ) -> Tuple[Dict[int, float], Dict[int, int], Dict[int, int]]:
        """Load ratings from MINER_STATS, matched by hotkey+revision."""
        current_ratings = {}
        current_rounds = {}
        model_ages = {}

        for uid, miner in miners.items():
            if prev_ratings and miner.hotkey in prev_ratings:
                prev = prev_ratings[miner.hotkey]
                # DynamoDB .get() returns None when key exists but value is None,
                # so use `or` instead of default parameter
                current_ratings[uid] = prev.get('elo_rating') or self.config.ELO_BASE_RATING
                current_rounds[uid] = prev.get('elo_rounds_played') or 0
                submit_block = prev.get('elo_model_submit_block') or current_block
                model_ages[uid] = max(0, current_block - submit_block)
            else:
                # New miner or new revision (no MINER_STATS record)
                current_ratings[uid] = self.config.ELO_BASE_RATING
                current_rounds[uid] = 0
                model_ages[uid] = 0

        return current_ratings, current_rounds, model_ages

    def _distribute_weights_by_rating(self, miners: Dict[int, MinerData]):
        """Distribute weights by ELO rating rank with decay.

        Excludes Pareto-filtered miners — they get weight=0 this round
        but keep their rating for next round.
        """
        rated_miners = [
            (uid, m) for uid, m in miners.items()
            if m.is_valid_for_scoring()
            and m.elo_rounds_played > 0
            and not m.filtered_subsets  # Pareto-filtered get no weight
        ]
        rated_miners.sort(key=lambda x: x[1].elo_rating, reverse=True)

        # Apply decay-based weight distribution
        total = 0.0
        for rank, (uid, miner) in enumerate(rated_miners, 1):
            weight = self.decay_factor ** (rank - 1)
            miner.subset_weights['elo'] = weight
            miner.subset_ranks['elo'] = rank
            total += weight

        # Normalize
        if total > 0:
            for uid, miner in rated_miners:
                miner.subset_weights['elo'] /= total
                miner.cumulative_weight = miner.subset_weights['elo']
