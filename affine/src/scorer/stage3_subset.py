"""
Stage 3: Subset Scoring and Weight Distribution

Calculates geometric mean scores for miners within each subset and
distributes weights proportionally based on performance.
"""

from typing import Dict, List, Any
from affine.src.scorer.models import (
    MinerData,
    SubsetInfo,
    Stage3Output,
)
from affine.src.scorer.config import ScorerConfig
from affine.src.scorer.utils import (
    generate_all_subsets,
    calculate_layer_weights,
    calculate_subset_weights,
    geometric_mean,
)

from affine.core.setup import logger


class Stage3SubsetScorer:
    """Stage 3: Subset Scoring and Weight Distribution.
    
    Responsibilities:
    1. Generate all subset combinations (L1, L2, L3, ...)
    2. Calculate layer weights with exponential growth
    3. For each subset:
       - Calculate geometric mean scores for participating miners
       - Rank miners by score
       - Distribute subset weight proportionally
    4. Apply optional rank-based decay
    """
    
    def __init__(self, config: ScorerConfig = ScorerConfig):
        """Initialize Stage 3 subset scorer.
        
        Args:
            config: Scorer configuration (defaults to global config)
        """
        self.config = config
        self.decay_factor = config.DECAY_FACTOR
        self.score_precision = config.SCORE_PRECISION
        self.geometric_mean_epsilon = config.GEOMETRIC_MEAN_EPSILON
    
    def score(
        self,
        miners: Dict[int, MinerData],
        environments: List[str]
    ) -> Stage3Output:
        """Calculate subset scores and distribute weights.
        
        Args:
            miners: Dict of MinerData objects from Stage 2
            environments: List of environment names
            
        Returns:
            Stage3Output with subset scores and weights
        """
        n_envs = len(environments)
        logger.info(f"Stage 3: Processing {n_envs} environments")
        
        # Generate subsets: evaluate only top MAX_LAYERS
        # e.g., 8 envs with MAX_LAYERS=6 -> evaluate L3-L8, skip L1-L2
        subsets_meta = generate_all_subsets(environments, max_layers=self.config.MAX_LAYERS)
        
        # Calculate starting layer
        start_layer = max(1, n_envs - self.config.MAX_LAYERS + 1) if n_envs > self.config.MAX_LAYERS else 1
        actual_layers = n_envs - start_layer + 1
        
        logger.debug(
            f"Generated {len(subsets_meta)} subsets across layers L{start_layer}-L{n_envs} "
            f"({actual_layers} layers, max_layers={self.config.MAX_LAYERS})"
        )
        
        # Calculate layer and subset weights (starting from the actual start_layer)
        layer_weights = calculate_layer_weights(n_envs, self.config.SUBSET_WEIGHT_EXPONENT, start_layer)
        subset_weights = calculate_subset_weights(subsets_meta, layer_weights)
        
        # Create SubsetInfo objects
        subsets: Dict[str, SubsetInfo] = {}
        for subset_key, subset_meta in subsets_meta.items():
            layer = subset_meta["layer"]
            envs = subset_meta["envs"]
            
            subsets[subset_key] = SubsetInfo(
                key=subset_key,
                layer=layer,
                envs=envs,
                layer_weight=layer_weights[layer],
                subset_weight=subset_weights[subset_key]
            )
        
        # Score each subset
        for subset_key, subset_info in subsets.items():
            self._score_subset(subset_key, subset_info, miners)
        
        # Calculate layer contributions for each miner
        for miner in miners.values():
            layer_totals = {}
            for subset_key, weight in miner.subset_weights.items():
                layer = subsets[subset_key].layer
                layer_totals[layer] = layer_totals.get(layer, 0.0) + weight
            miner.layer_weights = layer_totals
        
        logger.info(f"Stage 3: Scored {len(subsets)} subsets")
        
        return Stage3Output(
            miners=miners,
            subsets=subsets
        )
    
    def _score_subset(
        self,
        subset_key: str,
        subset_info: SubsetInfo,
        miners: Dict[int, MinerData]
    ):
        """Score miners within a single subset and distribute weights.
        
        Args:
            subset_key: Subset identifier
            subset_info: Subset metadata
            miners: Dict of all miners
        """
        envs = subset_info.envs
        
        # Find miners eligible for this subset
        eligible_miners = []
        for miner in miners.values():
            # Skip if filtered from this subset
            if subset_key in miner.filtered_subsets:
                subset_info.filtered_miners.append(miner.uid)
                continue
            
            # Check if miner has valid scores in all subset environments
            has_all_envs = all(
                env in miner.env_scores and miner.env_scores[env].is_valid
                for env in envs
            )
            
            if has_all_envs:
                eligible_miners.append(miner)
                subset_info.valid_miners.append(miner.uid)
        
        # Skip if no eligible miners
        if not eligible_miners:
            return
        
        # Calculate geometric mean scores (always use geometric mean)
        miner_scores = []
        for miner in eligible_miners:
            env_scores = [
                miner.env_scores[env].avg_score
                for env in envs
            ]
            
            # Always use geometric mean to penalize poor performance in any environment
            # Smoothing epsilon prevents zero scores from collapsing the entire result
            score = geometric_mean(env_scores, epsilon=self.geometric_mean_epsilon)
            miner_scores.append((miner.uid, score))
            miner.subset_scores[subset_key] = score
        
        # Sort by score (descending)
        miner_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Assign ranks and apply decay
        adjusted_scores = []
        for rank, (uid, score) in enumerate(miner_scores, start=1):
            miners[uid].subset_ranks[subset_key] = rank
            adjusted = score * (self.decay_factor ** (rank - 1))
            adjusted_scores.append((uid, adjusted))
        
        # Calculate proportional weights
        total_score = sum(score for _, score in adjusted_scores)
        
        if total_score > 0:
            for uid, score in adjusted_scores:
                proportion = score / total_score
                weight_contribution = subset_info.subset_weight * proportion
                miners[uid].subset_weights[subset_key] = weight_contribution
        else:
            # Edge case: all scores are 0
            equal_weight = subset_info.subset_weight / len(adjusted_scores)
            for uid, _ in adjusted_scores:
                miners[uid].subset_weights[subset_key] = equal_weight
    