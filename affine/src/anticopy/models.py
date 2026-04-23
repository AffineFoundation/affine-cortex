"""
Anti-Copy Detection Data Models
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class MinerLogprobs:
    """Logprob data for one miner across all logprobs-env tasks.

    task_logprobs: task_id -> (n_tokens,) array of logprob values
    task_topk:    task_id -> (n_tokens, top_k, 2) array where [..., 0]=prob, [..., 1]=token_id
                  We store top_k as dict list to preserve token identity.
    task_tokens:  task_id -> list of top-1 token strings (for agreement rate)
    """
    uid: int
    hotkey: str

    # task_id -> shape (n_tokens * TOP_K,) float array of logprobs
    task_logprobs: Dict[int, np.ndarray] = field(default_factory=dict)

    # task_id -> list of dicts [{token: str, prob: float}, ...] per position
    task_topk: Dict[int, List[List[dict]]] = field(default_factory=dict)

    # task_id -> list of top-1 token strings, one per position
    task_tokens: Dict[int, List[str]] = field(default_factory=dict)

    # task_id -> shape (hidden_dim,) float array of hidden states
    task_hidden_states: Dict[int, np.ndarray] = field(default_factory=dict)


@dataclass
class CopyPair:
    """Similarity result between two miners."""
    uid_a: int
    uid_b: int
    hotkey_a: str
    hotkey_b: str

    cosine_similarity: float    # median logprob cosine across tasks; 1.0 = identical
    hs_cosine: float            # median hidden_states cosine; NaN if no data
    js_divergence: float        # median JS divergence; 0.0 = identical
    token_agreement: float      # median token agreement across tasks

    n_tasks: int                # shared tasks used
    verdict: str                # "clean" | "suspicious" | "cheat"
    is_copy: bool
    confidence: float           # 0.0 - 1.0
    votes: int = 0              # number of signals that voted "copy"
    total_votes: int = 0        # number of signals that participated

    # max(|mean(norm_a/norm_b)-1|, std(norm_a/norm_b)); small for copies,
    # large for same-base fine-tunes. NaN if no hs data.
    hs_norm_deviation: float = float("nan")

    # Per-task cosine for debugging
    task_cosines: Dict[int, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        flag = self.verdict.upper()
        hs_str = f"{self.hs_cosine:.5f}" if not np.isnan(self.hs_cosine) else "N/A"
        nd_str = f"{self.hs_norm_deviation:.4f}" if not np.isnan(self.hs_norm_deviation) else "N/A"
        return (
            f"[{flag}] uid={self.uid_a} vs uid={self.uid_b} | "
            f"cos={self.cosine_similarity:.5f} "
            f"hs={hs_str} "
            f"nd={nd_str} "
            f"js={self.js_divergence:.5f} "
            f"agree={self.token_agreement:.3f} "
            f"votes={self.votes}/{self.total_votes} "
            f"tasks={self.n_tasks} conf={self.confidence:.3f}"
        )
