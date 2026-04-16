"""
Scorer Data Models

Data structures for the champion challenge scoring algorithm.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class EnvScore:
    """Per-environment score data for one miner."""

    avg_score: float                                    # Mean over the most recent N×window samples (display only)
    sample_count: int                                   # Completed tasks in the current sampling window
    completeness: float                                 # sample_count / window_size
    historical_count: int = 0                           # Distinct tasks ever sampled (full lifetime)
    all_task_scores: Dict[int, float] = field(default_factory=dict)  # Full historical task→score (for Pareto)

    def __repr__(self) -> str:
        return (f"EnvScore(avg={self.avg_score:.3f}, "
                f"samples={self.sample_count}, complete={self.completeness:.2%})")


@dataclass
class MinerData:
    """Complete data for a single miner across all environments."""

    uid: int
    hotkey: str
    model_revision: str
    model_repo: str
    first_block: int

    env_scores: Dict[str, EnvScore] = field(default_factory=dict)

    # Champion challenge state
    challenge_consecutive_wins: int = 0
    challenge_total_losses: int = 0
    challenge_consecutive_losses: int = 0
    challenge_checkpoints_passed: int = 0
    challenge_status: str = 'sampling'  # 'sampling' | 'terminated'
    termination_reason: str = ''        # '' | 'challenge_loss' | 'pairwise'
    is_champion: bool = False

    # Final weight (set by champion challenge: 1.0 for champion, 0.0 for others)
    normalized_weight: float = 0.0

    def __repr__(self) -> str:
        return (f"MinerData(uid={self.uid}, hotkey={self.hotkey[:8]}..., "
                f"envs={len(self.env_scores)})")


@dataclass
class ParetoComparison:
    """Result of Pareto dominance comparison between two miners."""

    miner_a_uid: int
    miner_b_uid: int
    label: str  # informational: "champion_challenge" or "pairwise"

    a_dominates_b: bool
    b_dominates_a: bool

    env_comparisons: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __repr__(self) -> str:
        if self.a_dominates_b:
            return f"Pareto({self.miner_a_uid} dominates {self.miner_b_uid})"
        if self.b_dominates_a:
            return f"Pareto({self.miner_b_uid} dominates {self.miner_a_uid})"
        return f"Pareto({self.miner_a_uid} ≈ {self.miner_b_uid})"


@dataclass
class Stage1Output:
    """Output from Stage 1 data collection."""

    miners: Dict[int, MinerData]
    environments: List[str]


@dataclass
class ChampionChallengeOutput:
    """Output from the champion challenge stage."""

    miners: Dict[int, MinerData]
    comparisons: List[ParetoComparison]
    champion_uid: Optional[int]
    champion_hotkey: Optional[str]
    champion_changed: bool
    final_weights: Dict[int, float]


@dataclass
class ScoringResult:
    """Complete result from one scoring round."""

    block_number: int
    calculated_at: int
    environments: List[str]

    config: Dict[str, Any] = field(default_factory=dict)
    miners: Dict[int, MinerData] = field(default_factory=dict)
    final_weights: Dict[int, float] = field(default_factory=dict)

    champion_uid: Optional[int] = None
    champion_hotkey: Optional[str] = None
    total_miners: int = 0

    def get_summary(self) -> Dict[str, Any]:
        return {
            'block_number': self.block_number,
            'total_miners': self.total_miners,
            'environments': len(self.environments),
            'champion_uid': self.champion_uid,
            'champion_hotkey': self.champion_hotkey,
        }

    def __repr__(self) -> str:
        return (f"ScoringResult(block={self.block_number}, "
                f"miners={self.total_miners}, champion={self.champion_uid})")
