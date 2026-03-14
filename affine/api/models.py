"""
API Request/Response Models

Pydantic models for request validation and response serialization.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel

class SampleSubmitResponse(BaseModel):
    """Response after submitting a sample."""

    task_id: str
    created_at: int
    message: str


class ExtraData(BaseModel):
    """Extra data with dynamic fields.
    
    Common fields:
    - conversation: List of message dicts with {"role": "user/assistant", "content": "..."}
    - request: Request parameters dict
    - image: Docker image used for evaluation
    
    Note: This model accepts any additional fields dynamically.
    """

    class Config:
        extra = "allow"  # Allow additional fields beyond defined ones

    conversation: Optional[List[Dict[str, Any]]] = None
    request: Optional[Dict[str, Any]] = None
    image: Optional[str] = None


class SampleFullResponse(BaseModel):
    """Full sample details from sample_results table."""

    miner_hotkey: str
    model_revision: str
    model: str
    env: str
    task_id: int
    score: float
    timestamp: int
    block_number: int
    latency_ms: Optional[int] = None
    extra: Optional[ExtraData] = None


class TaskPoolResponse(BaseModel):
    """Task pool entry details."""

    task_uuid: str
    task_id: int
    miner_hotkey: str
    model_revision: str
    model: str
    env: str
    chute_id: str
    status: str
    created_at: int
    assigned_to: Optional[str] = None
    assigned_at: Optional[int] = None
    retry_count: int
    max_retries: int
    last_error: Optional[str] = None
    last_error_code: Optional[str] = None
    last_failed_at: Optional[int] = None
    ttl: int


class TaskFetchResponse(BaseModel):
    """Response from task fetch endpoint."""

    tasks: List[Dict[str, Any]] = []


class MinerScore(BaseModel):
    """Score details for a miner."""

    miner_hotkey: str
    uid: int
    model_revision: str
    model: str
    first_block: int
    overall_score: float
    average_score: float
    scores_by_layer: Dict[str, float]
    scores_by_env: Dict[str, Dict[str, Any]]  # Changed to support {env: {score, sample_count}}
    total_samples: int
    cumulative_weight: Optional[float] = None
    elo_rating: Optional[float] = None
    elo_rounds_played: Optional[int] = None
    elo_rating_change: Optional[float] = None
    filter_info: Optional[Dict[str, Any]] = None


class ScoresResponse(BaseModel):
    """Scores snapshot response."""

    block_number: int
    calculated_at: int
    scores: List[MinerScore]


# Execution Logs
class ExecutionLog(BaseModel):
    """Execution log entry."""

    log_id: str
    timestamp: int
    task_id: str
    env: str
    status: str  # 'success' or 'failed'
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    latency_ms: int


class ExecutionLogsResponse(BaseModel):
    """List of execution logs."""

    logs: List[ExecutionLog]


class MinerResponse(BaseModel):
    """Miner information response."""

    model: Optional[str] = None
    chute_id: Optional[str] = None
    hotkey: str
    revision: Optional[str] = None
    block_number: Optional[int] = None
    first_block: Optional[int] = None
    uid: int
    invalid_reason: Optional[str] = None
    model_hash: Optional[str] = None
    chute_slug: Optional[str] = None
    chute_status: Optional[str] = None
    is_valid: Optional[str] = None
