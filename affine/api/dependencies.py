"""
FastAPI Dependencies

Reusable dependencies for authentication, database access, etc.
"""

import time
from typing import Optional
from fastapi import Depends, HTTPException, Request, Header, status
from affine.api.config import config
from affine.database.dao.sample_results import SampleResultsDAO
from affine.database.dao.task_pool import TaskPoolDAO
from affine.database.dao.execution_logs import ExecutionLogsDAO
from affine.database.dao.scores import ScoresDAO
from affine.database.dao.system_config import SystemConfigDAO
from affine.database.dao.score_snapshots import ScoreSnapshotsDAO
from affine.database.dao.miners import MinersDAO
from affine.api.services.auth import AuthService
from affine.api.config import config
from affine.api.services.task_pool import TaskPoolManager


# Database DAOs (singleton instances)
_sample_results_dao: Optional[SampleResultsDAO] = None
_task_pool_dao: Optional[TaskPoolDAO] = None
_execution_logs_dao: Optional[ExecutionLogsDAO] = None
_scores_dao: Optional[ScoresDAO] = None
_system_config_dao: Optional[SystemConfigDAO] = None
_score_snapshots_dao: Optional[ScoreSnapshotsDAO] = None
_miners_dao: Optional[MinersDAO] = None
_auth_service: Optional[AuthService] = None
_task_pool_manager: Optional[TaskPoolManager] = None


def get_sample_results_dao() -> SampleResultsDAO:
    """Get SampleResultsDAO instance."""
    global _sample_results_dao
    if _sample_results_dao is None:
        _sample_results_dao = SampleResultsDAO()
    return _sample_results_dao


def get_task_pool_dao() -> TaskPoolDAO:
    """Get TaskPoolDAO instance."""
    global _task_pool_dao
    if _task_pool_dao is None:
        _task_pool_dao = TaskPoolDAO()
    return _task_pool_dao


def get_execution_logs_dao() -> ExecutionLogsDAO:
    """Get ExecutionLogsDAO instance."""
    global _execution_logs_dao
    if _execution_logs_dao is None:
        _execution_logs_dao = ExecutionLogsDAO()
    return _execution_logs_dao


def get_scores_dao() -> ScoresDAO:
    """Get ScoresDAO instance."""
    global _scores_dao
    if _scores_dao is None:
        _scores_dao = ScoresDAO()
    return _scores_dao


def get_system_config_dao() -> SystemConfigDAO:
    """Get SystemConfigDAO instance."""
    global _system_config_dao
    if _system_config_dao is None:
        _system_config_dao = SystemConfigDAO()
    return _system_config_dao



def get_score_snapshots_dao() -> ScoreSnapshotsDAO:
    """Get ScoreSnapshotsDAO instance."""
    global _score_snapshots_dao
    if _score_snapshots_dao is None:
        _score_snapshots_dao = ScoreSnapshotsDAO()
    return _score_snapshots_dao


def get_miners_dao() -> MinersDAO:
    """Get MinersDAO instance."""
    global _miners_dao
    if _miners_dao is None:
        _miners_dao = MinersDAO()
    return _miners_dao


def get_auth_service() -> AuthService:
    """Get AuthService instance for executor authentication."""
    global _auth_service
    if _auth_service is None:
        # Create with non-strict mode for development
        # In production, use create_auth_service_from_chain()
        _auth_service = AuthService(
            authorized_validators=set(),
            signature_expiry_seconds=60,  # 1 minute timeout
            strict_mode=False  # Non-strict for development
        )
    return _auth_service


def get_task_pool_manager() -> TaskPoolManager:
    """Get TaskPoolManager instance (lazy initialization)."""
    global _task_pool_manager
    if _task_pool_manager is None:
        _task_pool_manager = TaskPoolManager()
    return _task_pool_manager


async def verify_executor_auth(
    executor_hotkey: str = Header(..., alias="X-Hotkey"),
    executor_signature: str = Header(..., alias="X-Signature"),
    executor_message: str = Header(..., alias="X-Message"),
    auth_service: AuthService = Depends(get_auth_service),
) -> str:
    """
    Verify executor authentication with timestamp-based message.
    
    This dependency validates:
    1. Message format is a valid timestamp (integer string)
    2. Timestamp is within 60 seconds (prevents replay attacks)
    3. Signature is valid for the message
    
    Args:
        executor_hotkey: Executor's hotkey from header
        executor_signature: Signature from header
        executor_message: Timestamp string from header
        auth_service: Auth service instance
    
    Returns:
        Validated executor hotkey
        
    Raises:
        HTTPException: If validation fails
    """
    # Validate message format (should be timestamp)
    try:
        timestamp = int(executor_message)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid message format: expected timestamp"
        )
    
    # Check timestamp is within 60 seconds
    current_time = int(time.time())
    time_diff = abs(current_time - timestamp)
    
    if time_diff > 60:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Message expired (timestamp diff: {time_diff}s, max: 60s)"
        )
    
    # Verify signature
    is_valid = auth_service.verify_signature(
        hotkey=executor_hotkey,
        message=executor_message,
        signature=executor_signature
    )
    
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid executor signature"
        )
    
    # Check if authorized validator (optional in non-strict mode)
    if not auth_service.is_authorized_validator(executor_hotkey):
        if auth_service.strict_mode:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Executor not authorized"
            )
    
    return executor_hotkey

_rate_limit_store: dict = {}


def check_rate_limit(
    identifier: str,
    limit: int,
    window_seconds: int = 60,
) -> bool:
    """
    Check if rate limit is exceeded.
    
    Args:
        identifier: Unique identifier (IP, hotkey, etc.)
        limit: Max requests per window
        window_seconds: Time window in seconds
        
    Returns:
        True if within limit, False if exceeded
    """
    current_time = int(time.time())
    window_start = current_time - window_seconds
    
    # Get or create request history for this identifier
    if identifier not in _rate_limit_store:
        _rate_limit_store[identifier] = []
    
    # Remove old requests outside the window
    _rate_limit_store[identifier] = [
        ts for ts in _rate_limit_store[identifier] if ts > window_start
    ]
    
    # Check if limit is exceeded
    if len(_rate_limit_store[identifier]) >= limit:
        return False
    
    # Add current request
    _rate_limit_store[identifier].append(current_time)
    return True


async def rate_limit_read(request: Request):
    """Dependency for read endpoint rate limiting."""
    if not config.RATE_LIMIT_ENABLED:
        return

    identifier = request.client.host
    if not check_rate_limit(identifier, config.RATE_LIMIT_READ):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )


async def rate_limit_write(request: Request):
    """Dependency for write endpoint rate limiting."""
    if not config.RATE_LIMIT_ENABLED:
        return
    
    if not check_rate_limit(request.client.host, config.RATE_LIMIT_WRITE):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )


async def rate_limit_scoring(request: Request):
    """Dependency for /scoring endpoint - strict rate limiting (1/min)."""
    if not config.RATE_LIMIT_ENABLED:
        return

    identifier = request.client.host
    # Hardcoded: 1 request per 60 seconds for scoring endpoint
    if not check_rate_limit(identifier, limit=1, window_seconds=60):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded for scoring endpoint (1 request per minute)"
        )


async def get_system_config():
    """Get system configuration."""
    dao = get_system_config_dao()
    return await dao.get_config()