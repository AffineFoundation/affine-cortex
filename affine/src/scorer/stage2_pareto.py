"""
Stage 2: Pareto Frontier Anti-Plagiarism Filtering

Implements Pareto dominance-based filtering to detect and exclude
plagiarized models using multi-environment performance analysis.
"""

from typing import Dict, List, Any
from affine.src.scorer.models import (
    MinerData,
    ParetoComparison,
    Stage2Output,
)
from affine.src.scorer.config import ScorerConfig

from affine.core.setup import logger


class Stage2ParetoFilter:
    """Stage 2: Pareto Frontier Anti-Plagiarism Filtering.
    
    Core Algorithm:
    1. Sort miners by first_block (earlier = higher priority)
    2. For each subset, compare all miner pairs
    3. Apply Pareto dominance test with error rate reduction threshold
    4. Filter dominated miners from subset participation
    
    Dominance Rule:
    - Miner A dominates Miner B if A came first and B cannot beat A's
      required threshold (0.2 + 0.8 * A's score) in ALL environments
    """
    
    def __init__(self, config: ScorerConfig = ScorerConfig):
        """Initialize Stage 2 Pareto filter.

        Args:
            config: Scorer configuration (defaults to global config)
        """
        self.config = config
        self.score_precision = config.SCORE_PRECISION
    
    def filter(
        self,
        miners: Dict[int, MinerData],
        subsets: Dict[str, Dict[str, Any]]
    ) -> Stage2Output:
        """Apply Pareto filtering to all subsets.
        
        Args:
            miners: Dict of MinerData objects from Stage 1
            subsets: Dict of subset metadata from utils.generate_all_subsets()
                Format: {
                    "L2_sat_abd": {
                        "layer": 2,
                        "envs": ["sat", "abd"],
                        "key": "L2_sat_abd"
                    }
                }
                
        Returns:
            Stage2Output with updated miners and comparison results
        """
        logger.info(f"Stage 2: Starting Pareto filtering for {len(subsets)} subsets")
        
        # Sort miners by first_block (earlier blocks first)
        sorted_miners = sorted(
            miners.values(),
            key=lambda m: (m.first_block, m.uid)
        )
        
        comparisons: List[ParetoComparison] = []
        filtered_count = 0
        
        # Process each subset
        for subset_key, subset_info in subsets.items():
            subset_envs = subset_info["envs"]
            
            # Find miners with valid scores in all subset environments
            valid_miners_for_subset = []
            for miner in sorted_miners:
                # Check if miner has valid scores in all subset environments
                has_all_envs = all(
                    env in miner.env_scores and miner.env_scores[env].is_valid
                    for env in subset_envs
                )
                if has_all_envs:
                    valid_miners_for_subset.append(miner)
            
            # Skip subset if fewer than 2 miners
            if len(valid_miners_for_subset) < 2:
                continue
            
            # Perform pairwise Pareto comparisons
            dominated_in_subset = set()
            
            for i, miner_a in enumerate(valid_miners_for_subset):
                for miner_b in valid_miners_for_subset[i + 1:]:
                    # A came before B (sorted by first_block)
                    comparison = self._compare_miners(
                        miner_a,
                        miner_b,
                        subset_envs,
                        subset_key
                    )
                    
                    comparisons.append(comparison)
                    
                    # Track dominated miners
                    if comparison.a_dominates_b:
                        dominated_in_subset.add(miner_b.uid)
                    elif comparison.b_dominates_a:
                        dominated_in_subset.add(miner_a.uid)
                        logger.debug(
                            f"Subset {subset_key}: UID {miner_b.uid} dominates UID {miner_a.uid}"
                        )
            
            # Update miners with filtering results
            for miner_uid in dominated_in_subset:
                if miner_uid in miners:
                    miners[miner_uid].filtered_subsets.append(subset_key)
                    miners[miner_uid].filter_reasons[subset_key] = "dominated"
                    filtered_count += 1
                    
                    logger.debug(
                        f"UID {miner_uid} filtered from {subset_key} (Pareto dominated)"
                    )
        
        logger.info(
            f"Stage 2: Completed Pareto filtering - "
            f"{filtered_count} subset participations filtered"
        )
        
        return Stage2Output(
            miners=miners,
            comparisons=comparisons,
            filtered_count=filtered_count
        )
    
    def _compare_miners(
        self,
        miner_a: MinerData,
        miner_b: MinerData,
        envs: List[str],
        subset_key: str
    ) -> ParetoComparison:
        """Compare two miners using Pareto dominance test.
        
        Dominance Rule:
        - First determine winner in each environment using threshold
        - A dominates B only if A wins in ALL environments
        - B dominates A only if B wins in ALL environments
        - Otherwise they are non-dominated
        
        Winner Determination:
        - If A came first: B wins if B > threshold(A), else A wins
        - If B came first: A wins if A > threshold(B), else B wins
        
        Args:
            miner_a: Earlier miner (lower first_block)
            miner_b: Later miner (higher first_block)
            envs: List of environment names in subset
            subset_key: Subset identifier for logging
            
        Returns:
            ParetoComparison result
        """
        env_comparisons = {}
        
        # Track who wins in each environment
        a_wins_count = 0
        b_wins_count = 0
        
        for env in envs:
            # Get scores and threshold (already validated in filter())
            env_score_a = miner_a.env_scores[env]
            score_a = env_score_a.avg_score
            score_b = miner_b.env_scores[env].avg_score
            
            # Use stored threshold instead of recalculating
            threshold = env_score_a.threshold
            
            # B wins if it beats the threshold (with epsilon for floating point comparison)
            eps = 1e-9
            b_wins_env = score_b > (threshold + eps)
            
            if b_wins_env:
                b_wins_count += 1
            else:
                a_wins_count += 1
            
            # Store comparison details
            env_comparisons[env] = {
                "a_score": score_a,
                "b_score": score_b,
                "threshold": threshold,
                "b_beats_threshold": b_wins_env,
                "winner": "B" if b_wins_env else "A"
            }
        
        # Determine dominance based on winning all environments
        # A dominates B only if A wins in ALL environments
        a_dominates_b = (a_wins_count == len(envs))
        
        # B dominates A only if B wins in ALL environments
        b_dominates_a = (b_wins_count == len(envs))
        
        return ParetoComparison(
            miner_a_uid=miner_a.uid,
            miner_b_uid=miner_b.uid,
            subset_key=subset_key,
            a_dominates_b=a_dominates_b,
            b_dominates_a=b_dominates_a,
            env_comparisons=env_comparisons
        )
    