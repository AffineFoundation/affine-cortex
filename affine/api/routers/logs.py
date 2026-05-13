"""
Execution Logs Router

Endpoints for querying execution logs and error tracking.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from affine.api.models import (
    ExecutionLogsResponse,
    ExecutionLog,
)
from affine.api.dependencies import (
    get_execution_logs_dao,
    rate_limit_read,
)
from affine.database.dao.execution_logs import ExecutionLogsDAO

router = APIRouter(prefix="/logs", tags=["Logs"])


@router.get("/miner/{hotkey}", response_model=ExecutionLogsResponse, dependencies=[Depends(rate_limit_read)])
async def get_miner_logs(
    hotkey: str,
    limit: int = Query(100, description="Maximum number of logs", ge=1, le=1000),
    success: Optional[bool] = Query(None, description="Filter by success/failure"),
    dao: ExecutionLogsDAO = Depends(get_execution_logs_dao),
):
    """
    Get recent execution logs for a miner.
    
    Query parameters:
    - limit: Maximum logs to return (default: 100, max: 1000)
    - success: Filter by success (true) or failure (false) (optional)
    
    Note: model_revision filtering not supported by DAO
    """
    try:
        status_filter = None
        if success is not None:
            status_filter = "completed" if success else "failed"
        
        # DAO only accepts miner_hotkey, limit, and status
        logs = await dao.get_recent_logs(
            miner_hotkey=hotkey,
            limit=limit,
            status=status_filter,
        )
        
        log_entries = [
            ExecutionLog(
                log_id=log["log_id"],
                timestamp=log["timestamp"],
                task_id=str(log.get("dataset_task_id", log.get("task_id", ""))),
                env=log["env"],
                status=log.get("status", "unknown"),
                error_type=log.get("error_type"),
                error_message=log.get("error_message"),
                latency_ms=int(log.get("latency_ms") or log.get("execution_time_ms") or 0),
            )
            for log in logs
        ]
        
        return ExecutionLogsResponse(logs=log_entries)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve logs: {str(e)}"
        )
