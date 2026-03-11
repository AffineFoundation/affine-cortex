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
    
    
    def print_detailed_table(self, miners: Dict[int, MinerData], environments: list, env_configs: Dict[str, Any] = None):
        """Print detailed scoring table with all metrics.

        Args:
            miners: Dict of all miners
            environments: List of environment names
            env_configs: Optional env configs dict (used for display_name)
        """
        if env_configs is None:
            env_configs = {}
        print("=" * 180, flush=True)
        print("DETAILED SCORING TABLE", flush=True)
        print("=" * 180, flush=True)
        
        # Build header - Hotkey first, then UID, then Model, then First Block, then environments
        header_parts = ["Hotkey  ", "UID", "Model               ", " FirstBlk "]
        
        # Format environment names - use display_name if available
        for env in sorted(environments):
            env_cfg = env_configs.get(env, {})
            if isinstance(env_cfg, dict) and env_cfg.get('display_name'):
                env_display = env_cfg['display_name']
            elif ':' in env:
                env_display = env.split(':', 1)[1]  # Keep everything after ':'
            else:
                env_display = env
            # Adjust width to accommodate "score[threshold]/count(!)" format
            header_parts.append(f"{env_display:>20}")
        
        # Add layer columns with fixed width - only non-zero layers
        # Find all layers that have non-zero weights for any miner
        all_layers = set()
        for miner in miners.values():
            for layer, weight in miner.layer_weights.items():
                if weight > 0:
                    all_layers.add(layer)
        
        # Sort layers
        active_layers = sorted(all_layers)
        
        for layer in active_layers:
            header_parts.append(f"{'L'+str(layer):>8}")
        
        header_parts.extend(["   Total ", "  Weight ", "V"])
        
        print(" | ".join(header_parts), flush=True)
        print("-" * 180, flush=True)
        
        # Sort miners by final weight
        sorted_miners = sorted(
            miners.values(),
            key=lambda m: m.normalized_weight,
            reverse=True
        )
        
        # Print each miner row
        for miner in sorted_miners:
            # Use model_repo if available, otherwise use model_revision
            model_display = miner.model_repo[:20]

            row_parts = [
                f"{miner.hotkey[:8]:8s}",  # Hotkey first
                f"{miner.uid:3d}",          # UID second
                f"{model_display:20s}",     # Model repo name (20 chars)
                f"{miner.first_block:10d}"  # First block
            ]
            
            # Environment scores - show "score[threshold]/count(!)" format (score × 100, 2 decimals)
            for env in sorted(environments):
                if env in miner.env_scores:
                    score = miner.env_scores[env]
                    score_percent = score.avg_score * 100  # Convert to percentage
                    
                    # Use stored threshold instead of recalculating
                    threshold_percent = score.threshold * 100
                    
                    if score.is_valid:
                        # Valid: show as "92.30[94.84]/500"
                        score_str = f"{score_percent:.2f}[{threshold_percent:.2f}]/{score.sample_count}"
                        row_parts.append(f"{score_str:>20}")
                    else:
                        # Invalid (below threshold): show as "92.30[94.84]/50!" with ! suffix
                        score_str = f"{score_percent:.2f}[{threshold_percent:.2f}]/{score.sample_count}!"
                        row_parts.append(f"{score_str:>20}")
                else:
                    row_parts.append(f"{'  -  ':>20}")
            
            # Layer weights - only for active layers
            for layer in active_layers:
                weight = miner.layer_weights.get(layer, 0.0)
                row_parts.append(f"{weight:>8.4f}")
            
            # Total (cumulative weight sum) and Weight (normalized)
            row_parts.append(f"{miner.cumulative_weight:>9.4f}")  # Total: raw sum
            row_parts.append(f"{miner.normalized_weight:>9.6f}")  # Weight: normalized
            row_parts.append("✓" if miner.is_valid_for_scoring() else "✗")
            
            print(" | ".join(row_parts), flush=True)
        
        print("=" * 180, flush=True)