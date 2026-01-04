"""
Stage 1: Data Collection and Average Score Calculation

Collects sample data for all valid miners and calculates average scores
per environment with completeness validation.
"""

from typing import Dict, List, Any
from affine.src.scorer.models import (
    MinerData,
    EnvScore,
    Stage1Output,
)
from affine.src.scorer.config import ScorerConfig
from affine.src.scorer.utils import calculate_required_score

from affine.core.setup import logger


class Stage1Collector:
    """Stage 1: Data Collection and Average Score Calculation.
    
    Responsibilities:
    1. Parse scoring data from API response
    2. Calculate average scores per environment for each miner
    3. Validate sample completeness (must be >= 95% of required range)
    4. Build MinerData objects with environment scores
    """
    
    def __init__(self, config: ScorerConfig = ScorerConfig):
        """Initialize Stage 1 collector.
        
        Args:
            config: Scorer configuration (defaults to global config)
        """
        self.config = config
        self.min_completeness = config.MIN_COMPLETENESS
    
    def collect(
        self,
        scoring_data: Dict[str, Any],
        environments: List[str],
        env_configs: Dict[str, Any] = None
    ) -> Stage1Output:
        """Collect and process scoring data for all miners.
        
        Args:
            scoring_data: Response from /api/v1/samples/scoring endpoint
                Format: {
                    "hotkey#revision": {
                        "uid": 5,
                        "hotkey": "5...",
                        "model_revision": "...",
                        "env": {
                            "affine:sat": {
                                "samples": [...],
                                "total_count": 500,
                                "completed_count": 498,
                                "completeness": 0.996
                            }
                        }
                    }
                }
            environments: List of environment names participating in scoring
            env_configs: Dict mapping env_name -> env_config (including min_completeness)
            
        Returns:
            Stage1Output containing processed miner data
            
        Raises:
            RuntimeError: If scoring_data is invalid or contains API error response
        """
        # Initialize env_configs if not provided
        if env_configs is None:
            env_configs = {}
        # Validate scoring_data is not an error response
        if not isinstance(scoring_data, dict):
            raise RuntimeError(f"Invalid scoring_data type: {type(scoring_data)}")
        
        # Check for API error response format
        if "success" in scoring_data and scoring_data.get("success") is False:
            error_msg = scoring_data.get("error", "Unknown error")
            status_code = scoring_data.get("status_code", "unknown")
            logger.error(f"Received API error response: {error_msg} (status: {status_code})")
            raise RuntimeError(f"Cannot process scoring data: API returned error: {error_msg}")
        
        if not scoring_data:
            logger.warning("Received empty scoring_data")
            return Stage1Output(
                miners={},
                environments=environments,
                valid_count=0,
                invalid_count=0
            )
        
        logger.info(f"Stage 1: Starting data collection for {len(scoring_data)} miners")
        
        miners: Dict[int, MinerData] = {}
        valid_count = 0
        invalid_count = 0
        
        for key, miner_entry in scoring_data.items():
            # Extract UID from miner_entry
            uid = miner_entry.get('uid')
            if uid is None:
                logger.warning(f"Missing uid field in miner_entry for key {key}")
                continue
            
            try:
                uid = int(uid)
            except (ValueError, TypeError):
                logger.warning(f"Invalid UID value: {uid} for key {key}")
                continue

            # Extract miner metadata
            hotkey = miner_entry.get('hotkey')
            model_revision = miner_entry.get('model_revision')
            model_repo = miner_entry.get('model_repo')
            first_block = miner_entry.get('first_block')
            env_data = miner_entry.get('env', {})
            
            if not hotkey or not model_revision or not model_repo:
                logger.warning(f"UID {uid}: Missing required field (hotkey={bool(hotkey)}, model_revision={bool(model_revision)}, model_repo={bool(model_repo)})")
                invalid_count += 1
                continue
            
            # Create MinerData object
            miner = MinerData(
                uid=uid,
                hotkey=hotkey,
                model_revision=model_revision,
                model_repo=model_repo,
                first_block=first_block
            )
            
            # Process each environment
            for env_name in environments:
                env_info = env_data.get(env_name, {})
                
                if not env_info:
                    # Environment data missing
                    miner.env_scores[env_name] = EnvScore(
                        avg_score=0.0,
                        sample_count=0,
                        completeness=0.0,
                        is_valid=False,
                        threshold=0.0
                    )
                    continue
                
                # Extract environment data
                samples = env_info.get('samples', [])
                total_count = env_info.get('total_count', 0)
                completed_count = env_info.get('completed_count', 0)
                completeness = env_info.get('completeness', 0.0)
                
                # Calculate average score
                if samples:
                    scores = [s.get('score', 0.0) for s in samples]
                    raw_avg_score = sum(scores) / len(scores)
                else:
                    raw_avg_score = 0.0
                
                # Apply environment-specific normalization if configured
                if env_name in self.config.ENV_SCORE_RANGES:
                    min_score, max_score = self.config.ENV_SCORE_RANGES[env_name]
                    avg_score = (raw_avg_score - min_score) / (max_score - min_score)
                else:
                    avg_score = raw_avg_score
                
                # Get environment-specific min_completeness or use default
                env_config = env_configs.get(env_name, {})
                env_min_completeness = env_config.get('min_completeness', self.min_completeness)
                
                # Validate completeness
                is_valid = completeness >= env_min_completeness

                # Calculate required score threshold (sample-size aware)
                threshold = calculate_required_score(
                    avg_score,
                    completed_count,
                    self.config.Z_SCORE,
                    self.config.MIN_IMPROVEMENT,
                    self.config.MAX_IMPROVEMENT
                )
                
                # Store environment score
                miner.env_scores[env_name] = EnvScore(
                    avg_score=avg_score,
                    sample_count=completed_count,
                    completeness=completeness,
                    is_valid=is_valid,
                    threshold=threshold
                )
                
                # Only log invalid environments in DEBUG mode
                if not is_valid:
                    logger.debug(
                        f"UID {uid} {env_name}: completeness {completeness:.2%} < {env_min_completeness:.0%}"
                    )
            
            # Check if miner has at least one valid environment
            if miner.is_valid_for_scoring():
                valid_count += 1
            else:
                invalid_count += 1
                logger.debug(
                    f"UID {uid} ({hotkey[:8]}...): No valid environments (< {self.min_completeness:.0%})"
                )
            
            miners[uid] = miner
        
        logger.info(
            f"Stage 1: Completed data collection - "
            f"Valid: {valid_count}, Invalid: {invalid_count}"
        )
        
        return Stage1Output(
            miners=miners,
            environments=environments,
            valid_count=valid_count,
            invalid_count=invalid_count
        )
    