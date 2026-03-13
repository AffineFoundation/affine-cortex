"""
Main Scorer Orchestrator

Coordinates the four-stage scoring algorithm and manages result persistence.
"""

import time
from typing import Dict, Any, Optional
from .config import ScorerConfig
from .models import ScoringResult
from .stage1_collector import Stage1Collector
from .stage2_pareto import Stage2ParetoFilter
from .stage3_subset import Stage3SubsetScorer
from .stage4_weights import Stage4WeightNormalizer
from .utils import generate_all_subsets

from affine.core.setup import logger


class Scorer:
    """Main scorer orchestrator.
    
    Coordinates the four-stage scoring algorithm:
    1. Data Collection: Collect and validate sample data
    2. Pareto Filtering: Apply anti-plagiarism filtering
    3. Subset Scoring: Calculate geometric mean scores and distribute weights
    4. Weight Normalization: Apply threshold and normalization
    
    Optionally saves results to database.
    """
    
    def __init__(self, config: ScorerConfig = ScorerConfig):
        """Initialize scorer with configuration.
        
        Args:
            config: Scorer configuration (defaults to global config)
        """
        self.config = config
        
        # Initialize stage processors
        self.stage1 = Stage1Collector(config)
        self.stage2 = Stage2ParetoFilter(config)
        self.stage3 = Stage3SubsetScorer(config)
        self.stage4 = Stage4WeightNormalizer(config)
    
    def calculate_scores(
        self,
        scoring_data: Dict[str, Any],
        environments: list,
        env_configs: Dict[str, Any],
        block_number: int,
        prev_ratings: Optional[Dict[str, Any]] = None,
        print_summary: bool = True
    ) -> ScoringResult:
        """Execute the four-stage scoring algorithm.
        
        Args:
            scoring_data: Response from /api/v1/samples/scoring
            environments: List of environment names participating in scoring
            env_configs: Dict mapping env_name -> env_config (including min_completeness)
            block_number: Current block number
            print_summary: Whether to print detailed summaries (default: True)
            
        Returns:
            ScoringResult with complete scoring data
        """
        start_time = time.time()
        logger.info(f"Total Miners: {len(scoring_data)}")
        
        # Stage 1: Data Collection
        stage1_output = self.stage1.collect(scoring_data, environments, env_configs)
        
        # Stage 2: Pareto Filtering
        # Apply MAX_LAYERS limit: only evaluate top layers (e.g., L3-L8 if 8 envs and MAX_LAYERS=6)
        subsets_meta = generate_all_subsets(environments, max_layers=self.config.MAX_LAYERS)
        stage2_output = self.stage2.filter(stage1_output.miners, subsets_meta)
        
        # Stage 3: ELO Rating Update + Weight Distribution
        stage3_output = self.stage3.score(
            stage2_output.miners, environments,
            prev_ratings=prev_ratings,
            current_block=block_number,
        )
        
        # Stage 4: Weight Normalization
        stage4_output = self.stage4.normalize(stage3_output.miners)
        
        # Build final result
        result = ScoringResult(
            block_number=block_number,
            calculated_at=int(time.time()),
            environments=environments,
            config=self.config.to_dict(),
            miners=stage3_output.miners,
            pareto_comparisons=stage2_output.comparisons,
            subsets=stage3_output.subsets,
            final_weights=stage4_output.final_weights,
            total_miners=len(scoring_data),
            valid_miners=stage1_output.valid_count,
            invalid_miners=stage1_output.invalid_count,
        )
        
        elapsed_time = time.time() - start_time
        non_zero = len([w for w in result.final_weights.values() if w > 0])
        logger.info("=" * 80)
        logger.info(f"SCORING COMPLETED - Time: {elapsed_time:.2f}s, Active: {non_zero}/{len(scoring_data)}")
        logger.info("=" * 80)
        
        # Print detailed summary table
        if print_summary:
            self.stage4.print_detailed_table(stage3_output.miners, environments, env_configs)
        
        return result
    
    async def save_results(
        self,
        result: ScoringResult,
        score_snapshots_dao=None,
        scores_dao=None,
        miner_stats_dao=None
    ):
        """Save scoring results to database.
        
        Args:
            result: ScoringResult to save
            score_snapshots_dao: ScoreSnapshotsDAO instance (optional)
            scores_dao: ScoresDAO instance (optional)
        """
        if not score_snapshots_dao or not scores_dao:
            logger.warning("DAO instances not provided, skipping database save")
            return
        
        logger.info(f"Saving scoring results to database (block {result.block_number})")
        
        # Save snapshot metadata
        statistics = {
            "total_miners": result.total_miners,
            "valid_miners": result.valid_miners,
            "invalid_miners": result.invalid_miners,
            "miner_final_scores": {
                str(uid): weight
                for uid, weight in result.final_weights.items()
            }
        }
        
        await score_snapshots_dao.save_snapshot(
            block_number=result.block_number,
            scorer_hotkey="scorer_service",
            config=result.config,
            statistics=statistics
        )
        
        # Save to scores table (now contains all data - merged with miner_scores)
        logger.info(f"Saving complete scoring data to scores table...")
        for uid, miner in result.miners.items():
            # Calculate total samples
            total_samples = sum(
                env_score.sample_count
                for env_score in miner.env_scores.values()
            )
            
            # Prepare detailed scores by environment (with completeness and threshold)
            scores_by_env = {
                env: {
                    "score": score.avg_score,
                    "sample_count": score.sample_count,
                    "completeness": score.completeness,
                    "threshold": score.threshold
                }
                for env, score in miner.env_scores.items()
            }
            
            # Use normalized weight as overall score
            overall_score = miner.normalized_weight
            
            # Calculate average score from environments
            if scores_by_env:
                average_score = sum(env_data["score"] for env_data in scores_by_env.values()) / len(scores_by_env)
            else:
                average_score = 0.0
            
            # Convert layer_weights keys from int to str for DynamoDB
            scores_by_layer = {
                f"L{layer}": weight
                for layer, weight in miner.layer_weights.items()
            }
            
            # Prepare subset contributions (detailed)
            subset_contributions = {
                subset_key: {
                    "score": miner.subset_scores.get(subset_key, 0.0),
                    "rank": miner.subset_ranks.get(subset_key, 0),
                    "weight": weight
                }
                for subset_key, weight in miner.subset_weights.items()
            }
            
            # Prepare filter info
            filter_info = {
                "filtered_subsets": miner.filtered_subsets,
                "filter_reasons": miner.filter_reasons
            }
            
            # Save complete data to scores table (merged with miner_scores data)
            await scores_dao.save_score(
                block_number=result.block_number,
                miner_hotkey=miner.hotkey,
                uid=uid,
                model_revision=miner.model_revision,
                model=miner.model_repo,
                first_block=miner.first_block,
                overall_score=overall_score,
                average_score=average_score,
                scores_by_layer=scores_by_layer,
                scores_by_env=scores_by_env,
                total_samples=total_samples,
                # Additional detailed fields (formerly in miner_scores)
                subset_contributions=subset_contributions,
                cumulative_weight=miner.cumulative_weight,
                filter_info=filter_info,
                # ELO fields (redundant copy for API display)
                elo_rating=miner.elo_rating,
                elo_rounds_played=miner.elo_rounds_played,
                elo_rating_change=miner.elo_rating_change,
            )

            # Update MINER_STATS (authoritative ELO data source)
            if miner_stats_dao:
                await miner_stats_dao.update_elo_rating(
                    hotkey=miner.hotkey,
                    revision=miner.model_revision,
                    elo_rating=miner.elo_rating,
                    elo_rounds_played=miner.elo_rounds_played,
                    elo_model_submit_block=result.block_number if miner.elo_rounds_played == 1 else None,
                )

        logger.info(f"Successfully saved complete scoring results for {len(result.miners)} miners to scores table")


def create_scorer(config: Optional[ScorerConfig] = None) -> Scorer:
    """Factory function to create a Scorer instance.
    
    Args:
        config: Optional custom configuration
        
    Returns:
        Configured Scorer instance
    """
    if config is None:
        config = ScorerConfig()
    
    return Scorer(config)