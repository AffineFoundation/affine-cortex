"""
API Request/Response Models

Pydantic models for response serialization.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class MinerScore(BaseModel):
    """Score details for a miner.

    Reflects what the new ``scorer`` writes into the ``scores`` table —
    one row per miner per window with ``overall_score`` ∈ {0.0, 1.0}.
    The CLI surfaces ``is_valid`` alongside so a miner sidelined by the
    monitor (model_mismatch, plagiarism, …) doesn't read as competing.
    """

    miner_hotkey: str
    uid: int
    model_revision: str
    model: str
    first_block: int
    overall_score: float
    average_score: float
    scores_by_env: Dict[str, Dict[str, Any]]
    total_samples: int
    is_valid: Optional[bool] = None
    invalid_reason: Optional[str] = None
    # Surfaced so ``af get-rank`` can display TERMINATED for legacy- and
    # battle-loser miners (``challenge_status`` starting with ``terminated_``).
    challenge_status: Optional[str] = None
    termination_reason: Optional[str] = None


class ScoresResponse(BaseModel):
    """Scores snapshot response."""

    block_number: int
    calculated_at: int
    scores: List[MinerScore]


class MinerInfo(BaseModel):
    """Public miner metadata.

    Basic identity, model, validity, queue status, and commit block fields
    for a miner.
    """

    uid: int
    hotkey: str
    model: Optional[str] = None
    revision: Optional[str] = None
    is_valid: Optional[bool] = None
    challenge_status: Optional[str] = None
    first_block: Optional[int] = None
    block_number: Optional[int] = None
    invalid_reason: Optional[str] = None
    model_hash: Optional[str] = None


class ExecutionLog(BaseModel):
    """Execution log entry."""

    log_id: str
    timestamp: int
    task_id: str
    env: str
    status: str
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    latency_ms: int


class ExecutionLogsResponse(BaseModel):
    """List of execution logs."""

    logs: List[ExecutionLog]
