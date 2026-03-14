"""
Stage 4: Weight Normalization and Finalization

Aggregates subset contributions, applies minimum threshold,
and normalizes final weights.
"""

from typing import Dict, Any
from affine.src.scorer.models import (
    MinerData,
    Stage4Output,
)
from affine.src.scorer.config import ScorerConfig
from affine.src.scorer.utils import (
    normalize_weights,
    apply_min_threshold,
)

from affine.core.setup import logger


class Stage4WeightNormalizer:
    """Stage 4: Weight Normalization and Finalization.
    
    Responsibilities:
    1. Accumulate subset weight contributions for each miner
    2. Apply minimum weight threshold (remove miners < 1%)
    3. Normalize weights to sum to 1.0
    4. Generate final weight distribution for chain
    """
    
    def __init__(self, config: ScorerConfig = ScorerConfig):
        """Initialize Stage 4 weight normalizer.
        
        Args:
            config: Scorer configuration (defaults to global config)
        """
        self.config = config
        self.min_threshold = config.MIN_WEIGHT_THRESHOLD
    
    def normalize(
        self,
        miners: Dict[int, MinerData]
    ) -> Stage4Output:
        """Normalize weights and finalize distribution.
        
        Args:
            miners: Dict of MinerData objects from Stage 3
            
        Returns:
            Stage4Output with final normalized weights
        """
        logger.info(f"Stage 4: Normalizing weights for {len(miners)} miners")
        
        # Step 1: Accumulate cumulative weights
        raw_weights: Dict[int, float] = {}
        for uid, miner in miners.items():
            cumulative = sum(miner.subset_weights.values())
            miner.cumulative_weight = cumulative
            raw_weights[uid] = cumulative
        
        logger.debug(f"Accumulated cumulative weights from subset contributions")
        
        # Step 2: Apply minimum threshold
        weights_after_threshold = apply_min_threshold(
            raw_weights,
            self.min_threshold
        )
        
        below_threshold_count = sum(
            1 for uid, weight in raw_weights.items()
            if weight > 0 and weight < self.min_threshold
        )
        
        if below_threshold_count > 0:
            logger.debug(
                f"Removed {below_threshold_count} miners below threshold "
                f"({self.min_threshold:.1%})"
            )
        
        # Step 3: Final normalization (ensure sum = 1.0)
        final_weights = normalize_weights(weights_after_threshold)

        # Step 4: Apply min threshold after normalization and redistribute to uid 0
        final_weights = apply_min_threshold(
            final_weights,
            threshold=self.min_threshold,
            redistribute_to_uid_zero=True
        )

        # Update miner objects with normalized weights
        for uid, weight in final_weights.items():
            if uid in miners:
                miners[uid].normalized_weight = weight
        
        non_zero_count = len([w for w in final_weights.values() if w > 0])
        logger.info(f"Stage 4: Non-zero weights={non_zero_count}")
        
        return Stage4Output(
            final_weights=final_weights,
            below_threshold_count=below_threshold_count
        )
    
    
    @staticmethod
    def _get_filter_reason(miner: MinerData, environments: list) -> str:
        """Get concise filter reason for a miner (max 12 chars).

        Returns empty string if miner is not filtered.
        """
        # Pareto dominated
        if miner.filtered_subsets:
            for reason in miner.filter_reasons.values():
                if isinstance(reason, str) and reason.startswith("dom>"):
                    return reason  # e.g. "dom>123"
            return "pareto"
        # Incomplete env (not all envs valid)
        invalid_envs = [
            env for env in environments
            if env in miner.env_scores and not miner.env_scores[env].is_valid
        ]
        if invalid_envs:
            env_short = invalid_envs[0].split(':')[-1][:10]
            return f"!{env_short}"
        # No valid envs at all
        if not miner.is_valid_for_scoring():
            return "no_data"
        return ""

    def print_detailed_table(self, miners: Dict[int, MinerData], environments: list, env_configs: Dict[str, Any] = None):
        """Print detailed scoring table with all metrics.

        Args:
            miners: Dict of all miners
            environments: List of environment names
            env_configs: Optional env configs dict (used for display_name)
        """
        if env_configs is None:
            env_configs = {}

        # Build header
        header_parts = ["Hotkey  ", " UID", "Model               ", " FirstBlk "]

        for env in sorted(environments):
            env_cfg = env_configs.get(env, {})
            if isinstance(env_cfg, dict) and env_cfg.get('display_name'):
                env_display = env_cfg['display_name']
            elif ':' in env:
                env_display = env.split(':', 1)[1]
            else:
                env_display = env
            header_parts.append(f"{env_display:>16}")

        header_parts.extend(["  Rating", "     Δ", " Rnd", "  Weight ", "Status      "])

        header_line = " | ".join(header_parts)
        table_width = len(header_line)

        print("=" * table_width, flush=True)
        print("DETAILED SCORING TABLE", flush=True)
        print("=" * table_width, flush=True)
        print(header_line, flush=True)
        print("-" * table_width, flush=True)

        # Sort: miners with weight first (by rating desc), then filtered (by rating desc)
        sorted_miners = sorted(
            miners.values(),
            key=lambda m: (m.normalized_weight > 0, m.elo_rating),
            reverse=True
        )

        for miner in sorted_miners:
            model_display = miner.model_repo[:20]
            filter_reason = self._get_filter_reason(miner, environments)
            is_active = miner.normalized_weight > 0

            row_parts = [
                f"{miner.hotkey[:8]:8s}",
                f"{miner.uid:4d}",
                f"{model_display:20s}",
                f"{miner.first_block:10d}"
            ]

            # Environment scores
            for env in sorted(environments):
                if env in miner.env_scores:
                    score = miner.env_scores[env]
                    score_percent = score.avg_score * 100

                    if score.is_valid:
                        score_str = f"{score_percent:.2f}/{score.sample_count}"
                    else:
                        score_str = f"{score_percent:.2f}/{score.sample_count}!"
                    row_parts.append(f"{score_str:>16}")
                else:
                    row_parts.append(f"{'  -  ':>16}")

            if is_active:
                row_parts.append(f"{int(miner.elo_rating):>8d}")
                elo_change = int(miner.elo_rating_change)
                row_parts.append(f"{'+' + str(elo_change) if elo_change >= 0 else str(elo_change):>5}")
                row_parts.append(f"{int(miner.elo_rounds_played):>4d}")
                row_parts.append(f"{miner.normalized_weight:>9.6f}")
                row_parts.append(f"{'✓':12s}")
            else:
                row_parts.append(f"{'—':>8}")
                row_parts.append(f"{'—':>5}")
                row_parts.append(f"{'—':>4}")
                row_parts.append(f"{'0':>9}")
                row_parts.append(f"{filter_reason or '✗':12s}")

            print(" | ".join(row_parts), flush=True)

        print("=" * table_width, flush=True)