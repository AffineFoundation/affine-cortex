"""
Scorer Data Models

Data structures for the four-stage scoring algorithm.
"""

from typing import Dict, List, Any
from dataclasses import dataclass, field


@dataclass
class EnvScore:
    """Score data for a single environment."""

    avg_score: float
    sample_count: int
    completeness: float
    is_valid: bool
    threshold: float
    task_scores: Dict[int, float] = field(default_factory=dict)  # task_id -> normalized score (sampling list only)
    all_task_scores: Dict[int, float] = field(default_factory=dict)  # task_id -> normalized score (all samples, for Pareto)

    def __repr__(self) -> str:
        return f"EnvScore(avg={self.avg_score:.3f}, samples={self.sample_count}, complete={self.completeness:.2%})"


@dataclass
class MinerData:
    """Complete data for a single miner across all environments."""
    
    uid: int
    hotkey: str
    model_revision: str
    model_repo: str
    first_block: int
    
    # Stage 1: Environment scores
    env_scores: Dict[str, EnvScore] = field(default_factory=dict)
    
    # Stage 2: Pareto filtering results
    filtered_subsets: List[str] = field(default_factory=list)
    filter_reasons: Dict[str, str] = field(default_factory=dict)
    
    # Stage 3: ELO
    elo_rating: float = 1200.0
    elo_rating_change: float = 0.0
    elo_rounds_played: int = 0

    # Stage 3: Subset scores
    subset_scores: Dict[str, float] = field(default_factory=dict)
    subset_ranks: Dict[str, int] = field(default_factory=dict)
    subset_weights: Dict[str, float] = field(default_factory=dict)
    
    # Stage 4: Final weights
    layer_weights: Dict[str, float] = field(default_factory=dict)
    cumulative_weight: float = 0.0
    normalized_weight: float = 0.0
    
    def is_valid_for_scoring(self) -> bool:
        """Check if miner has sufficient valid environment scores."""
        return any(env.is_valid for env in self.env_scores.values())
    
    def get_valid_envs(self) -> List[str]:
        """Get list of environments where miner has valid scores."""
        return [env for env, score in self.env_scores.items() if score.is_valid]
    
    def __repr__(self) -> str:
        valid_envs = len(self.get_valid_envs())
        return f"MinerData(uid={self.uid}, hotkey={self.hotkey[:8]}..., valid_envs={valid_envs})"


@dataclass
class SubsetInfo:
    """Information about a subset (environment combination)."""
    
    key: str  # e.g., "L3_sat_abd_ded"
    layer: int
    envs: List[str]
    layer_weight: float
    subset_weight: float
    
    # Miners participating in this subset
    valid_miners: List[int] = field(default_factory=list)
    filtered_miners: List[int] = field(default_factory=list)
    
    def __repr__(self) -> str:
        return f"SubsetInfo(key={self.key}, layer=L{self.layer}, envs={len(self.envs)}, weight={self.subset_weight:.3f})"


@dataclass
class ParetoComparison:
    """Result of Pareto dominance comparison between two miners."""
    
    miner_a_uid: int
    miner_b_uid: int
    subset_key: str
    
    # Comparison results
    a_dominates_b: bool
    b_dominates_a: bool
    
    # Details for logging
    env_comparisons: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Format: {env: {"a_score": 0.9, "b_score": 0.85, "threshold": 0.88}}
    
    def __repr__(self) -> str:
        if self.a_dominates_b:
            return f"Pareto({self.miner_a_uid} dominates {self.miner_b_uid})"
        elif self.b_dominates_a:
            return f"Pareto({self.miner_b_uid} dominates {self.miner_a_uid})"
        else:
            return f"Pareto({self.miner_a_uid} ≈ {self.miner_b_uid} - no dominance)"


@dataclass
class ScoringResult:
    """Complete result from the four-stage scoring algorithm."""
    
    # Metadata
    block_number: int
    calculated_at: int
    environments: List[str]
    
    # Configuration snapshot
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Stage 1: All miner data
    miners: Dict[int, MinerData] = field(default_factory=dict)
    
    # Stage 2: Pareto filtering
    pareto_comparisons: List[ParetoComparison] = field(default_factory=list)
    
    # Stage 3: Subset information
    subsets: Dict[str, SubsetInfo] = field(default_factory=dict)
    
    # Stage 4: Final weights
    final_weights: Dict[int, float] = field(default_factory=dict)
    
    # Statistics
    total_miners: int = 0
    valid_miners: int = 0
    invalid_miners: int = 0
    
    def get_weights_for_chain(self) -> Dict[int, float]:
        """Get normalized weights suitable for setting on-chain.
        
        Returns:
            Dict mapping UID to normalized weight (0.0 to 1.0)
        """
        return self.final_weights.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for logging/display."""
        return {
            'block_number': self.block_number,
            'total_miners': self.total_miners,
            'valid_miners': self.valid_miners,
            'invalid_miners': self.invalid_miners,
            'environments': len(self.environments),
            'subsets': len(self.subsets),
            'non_zero_weights': sum(1 for w in self.final_weights.values() if w > 0),
        }
    
    def __repr__(self) -> str:
        return (
            f"ScoringResult(block={self.block_number}, "
            f"miners={self.total_miners}, "
            f"valid={self.valid_miners}, "
            f"envs={len(self.environments)})"
        )


@dataclass
class Stage1Output:
    """Output from Stage 1: Data Collection."""
    
    miners: Dict[int, MinerData]
    environments: List[str]
    valid_count: int
    invalid_count: int


@dataclass
class Stage2Output:
    """Output from Stage 2: Pareto Filtering."""
    
    miners: Dict[int, MinerData]
    comparisons: List[ParetoComparison]
    filtered_count: int


@dataclass
class Stage3Output:
    """Output from Stage 3: Subset Scoring."""
    
    miners: Dict[int, MinerData]
    subsets: Dict[str, SubsetInfo]


@dataclass
class Stage4Output:
    """Output from Stage 4: Weight Normalization."""
    
    final_weights: Dict[int, float]
    below_threshold_count: int