"""
Task Queue Router

Endpoints for managing sampling tasks with weighted random selection.
"""

import time
import asyncio
from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Header, Query, status
from affine.api.models import (
    TaskFetchResponse,
    SampleSubmitResponse,
)
from affine.api.dependencies import (
    get_task_pool_manager,
    verify_executor_auth,
    rate_limit_read,
)
from affine.api.config import config
from affine.core.models import SampleSubmission
from affine.api.services.task_pool import TaskPoolManager
from affine.api.services.sampling_stats import get_stats_collector

from affine.core.setup import logger

router = APIRouter(prefix="/tasks", tags=["Tasks"])


# Conditionally register task management endpoints based on SERVICES_ENABLED
if config.SERVICES_ENABLED:
    @router.post("/fetch", response_model=TaskFetchResponse)
    async def fetch_task(
        env: Optional[str] = Query(None, description="Environment filter (optional)"),
        batch_size: int = Query(1, ge=1, le=50, description="Number of tasks to fetch (1-50)"),
        executor_hotkey: str = Depends(verify_executor_auth),
        task_pool: TaskPoolManager = Depends(get_task_pool_manager),
    ):
        """
        Fetch task(s) using weighted random selection.
        
        Algorithm:
        Select batch_size miners randomly (weighted), return 1 task per miner.
        
        Headers (validated by verify_executor_auth dependency):
        - X-Hotkey: Executor's SS58 hotkey
        - X-Signature: Hex-encoded signature of timestamp
        - X-Message: Unix timestamp (must be within 60 seconds)
        
        Query Parameters:
        - env: Optional environment filter
        - batch_size: Number of tasks to fetch (default: 1, max: 50)
        
        Returns:
        - TaskFetchResponse with tasks list (0 to batch_size elements)
        
        Note:
        - Tasks are already enriched with miner_uid and chute_slug by TaskPoolManager
        - TaskPoolManager uses cached miners data (refreshed every 30s) for fast lookup
        """
        try:
            # Fetch task(s) using TaskPoolManager (returns enriched tasks with miner_uid and chute_slug)
            tasks = await task_pool.fetch_task(
                executor_hotkey=executor_hotkey,
                env=env,
                batch_size=batch_size
            )
            
            if not tasks:
                logger.debug(f"No available tasks for executor {executor_hotkey[:16]}...")
                return TaskFetchResponse(tasks=[])
            
            logger.debug(
                f"Assigned {len(tasks)} tasks to executor {executor_hotkey[:16]}... "
                f"(requested {batch_size})"
            )
            
            return TaskFetchResponse(tasks=tasks)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error fetching task(s): {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to fetch task(s): {str(e)}"
            )


    @router.post("/submit", response_model=SampleSubmitResponse)
    async def submit_sample_from_executor(
        submission: Dict[str, Any],
        executor_hotkey: str = Depends(verify_executor_auth),
        task_pool: TaskPoolManager = Depends(get_task_pool_manager),
    ):
        """
        Submit a sample result from executor.
        
        This endpoint:
        1. Verifies executor authentication via dependency (timestamp-based)
        2. Validates submission signature against task_uuid data
        3. Completes task via TaskPoolManager (saves sample, logs execution, deletes task)
        
        Headers (validated by verify_executor_auth dependency):
        - X-Hotkey: Executor's SS58 hotkey
        - X-Signature: Hex-encoded signature of timestamp
        - X-Message: Unix timestamp (must be within 60 seconds)
        
        Request body (SampleSubmission):
        - task_uuid: Task UUID from queue
        - score: Evaluation score (0.0 to 1.0)
        - latency_ms: Execution time in milliseconds
        - extra: Evaluation details and metadata
        - signature: Executor's signature of the above fields
        """
        # Parse submission
        try:
            sample_sub = SampleSubmission(**submission)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid submission format: {str(e)}"
            )
        
        # Verify submission signature
        if not sample_sub.verify(executor_hotkey):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid submission signature"
            )
        
        # Determine task outcome based on error presence
        error_message = sample_sub.extra.get("error")
        is_success = error_message is None
        
        # Error patterns that should be treated as valid zero-score samples
        # These indicate model limitations rather than temporary failures
        ZERO_SCORE_ERROR_PATTERNS = [
            "is longer than the model",
        ]
        
        # Check if error matches any zero-score patterns
        should_record_zero_score = False
        if error_message:
            error_lower = error_message.lower()
            for pattern in ZERO_SCORE_ERROR_PATTERNS:
                if pattern in error_lower:
                    should_record_zero_score = True
                    break

        # Process submission in background (do not await)
        # This allows the HTTP response to return immediately
        async def _background_submit():
            """Background task to process submission asynchronously."""
            try:
                # Get task metadata BEFORE complete_task to avoid race condition
                # (complete_task may delete the task from pool)
                task_location = await task_pool._get_task_location(sample_sub.task_uuid)
                task_metadata = None
                
                if task_location:
                    pk, sk = task_location
                    task_metadata = await task_pool.dao.get(pk, sk)
                
                # Complete the task (success or failure)
                # For zero-score errors, treat as successful sample with score=0
                if should_record_zero_score:
                    await task_pool.complete_task(
                        task_uuid=sample_sub.task_uuid,
                        executor_hotkey=executor_hotkey,
                        success=True,
                        result={
                            'score': 0.0,
                            'latency_ms': sample_sub.latency_ms,
                            'extra': sample_sub.extra,
                            'execution_time_ms': sample_sub.extra.get('execution_time_ms', 0)
                        },
                        error_message=None,
                        error_code=None,
                        submission_signature=sample_sub.signature
                    )
                else:
                    await task_pool.complete_task(
                        task_uuid=sample_sub.task_uuid,
                        executor_hotkey=executor_hotkey,
                        success=is_success,
                        result={
                            'score': sample_sub.score,
                            'latency_ms': sample_sub.latency_ms,
                            'extra': sample_sub.extra,
                            'execution_time_ms': sample_sub.extra.get('execution_time_ms', 0)
                        } if is_success else None,
                        error_message=error_message,
                        error_code="EXECUTION_ERROR",
                        submission_signature=sample_sub.signature
                    )
                
                # Record sampling statistics if task metadata was retrieved successfully
                # Note: task_metadata is cached before complete_task to avoid deletion race
                if task_metadata:
                    stats_collector = get_stats_collector()
                    stats_collector.record_sample(
                        hotkey=task_metadata['miner_hotkey'],
                        revision=task_metadata['model_revision'],
                        env=task_metadata['env'],
                        success=is_success or should_record_zero_score,
                        error_message=error_message if not should_record_zero_score else None
                    )
                else:
                    logger.warning(
                        f"Task metadata not found for {sample_sub.task_uuid[:8]}..., "
                        f"skipping sampling stats recording"
                    )
                
                logger.debug(
                    f"Background submit completed: task_uuid={sample_sub.task_uuid[:8]}... "
                    f"score={sample_sub.score:.4f}"
                )
            except Exception as e:
                logger.error(
                    f"Background submit failed: task_uuid={sample_sub.task_uuid[:8]}... "
                    f"error={e}", exc_info=True
                )
        
        # Schedule background task (fire-and-forget)
        asyncio.create_task(_background_submit())
        
        # Return immediately without waiting for database operations
        return SampleSubmitResponse(
            task_id=sample_sub.task_uuid,
            created_at=int(time.time()),
            message=f"Sample accepted for processing (score={sample_sub.score:.4f})"
        )

    @router.get("/pool/stats", dependencies=[Depends(rate_limit_read)])
    async def get_pool_stats(
        env: Optional[str] = Query(None, description="Environment filter (optional)"),
        task_pool: TaskPoolManager = Depends(get_task_pool_manager),
    ):
        """
        Get task queue statistics for monitoring.
        
        Query Parameters:
        - env: Optional environment filter (e.g., "agentgym:alfworld")
        
        Returns:
        - pending_count: Number of pending tasks in the queue
        - assigned_count: Number of assigned tasks
        - env: Environment name (if filtered)
        """
        try:
            if env:
                # Get stats using cached data (background refresh every 10s)
                stats = await task_pool.get_pool_stats(env)
                
                return {
                    "env": env,
                    "pending_count": stats.get('pending', 0),
                    "assigned_count": stats.get('assigned', 0),
                    "failed_count": stats.get('failed', 0),
                }
            else:
                # Get total stats across all environments
                # This would require querying all environments
                # For now, return error asking for env parameter
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="env parameter is required"
                )
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting queue stats: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get queue stats: {str(e)}"
            )
