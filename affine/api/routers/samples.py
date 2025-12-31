"""
Sample Results Router

Endpoints for querying sample results.
"""

from typing import Optional, Dict, Any, List, Union
from fastapi import APIRouter, Depends, HTTPException, Request, Query, status
from affine.api.models import (
    SampleFullResponse,
    TaskPoolResponse,
)
from affine.api.dependencies import (
    get_sample_results_dao,
    get_miners_dao,
    get_task_pool_manager,
    get_system_config_dao,
    rate_limit_read,
    rate_limit_scoring,
)
from affine.database.dao.sample_results import SampleResultsDAO
from affine.database.dao.miners import MinersDAO
from affine.database.dao.system_config import SystemConfigDAO
from affine.api.services.task_pool import TaskPoolManager
from affine.api.config import config

router = APIRouter(prefix="/samples", tags=["Samples"])


@router.get("/{hotkey}/{env}/{task_id}", response_model=Union[SampleFullResponse, TaskPoolResponse], dependencies=[Depends(rate_limit_read)])
async def get_sample(
    hotkey: str,
    env: str,
    task_id: str,
    model_revision: str = Query(..., description="Model revision"),
    dao: SampleResultsDAO = Depends(get_sample_results_dao),
    task_pool: TaskPoolManager = Depends(get_task_pool_manager),
):
    """
    Get a specific sample by its natural key components.
    
    Path parameters:
    - hotkey: Miner hotkey
    - env: Environment (e.g., affine:sat)
    - task_id: Task identifier
    
    Query parameters:
    - model_revision: Model revision hash
    
    Returns full sample details including conversation data.
    If not found in sample_results, tries to query from task_pool.
    """
    try:
        # First, try to get from sample_results
        item = await dao.get_sample_by_task_id(
            miner_hotkey=hotkey,
            model_revision=model_revision,
            env=env,
            task_id=task_id,
            include_extra=True
        )
        
        if item:
            return SampleFullResponse(**item)
        
        # If not found in sample_results, try task_pool
        task_id_int = int(task_id) if not isinstance(task_id, int) else task_id
        pool_task = await task_pool.dao.get_task_by_composite_key(
            miner_hotkey=hotkey,
            model_revision=model_revision,
            env=env,
            task_id=task_id_int
        )
        
        if pool_task:
            return TaskPoolResponse(**pool_task)
        
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sample not found in sample_results or task_pool for hotkey={hotkey}, env={env}, task_id={task_id}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve sample: {str(e)}"
        )


@router.get("/uid/{uid}/{env}/{task_id}", response_model=Union[SampleFullResponse, TaskPoolResponse], dependencies=[Depends(rate_limit_read)])
async def get_sample_by_uid(
    uid: int,
    env: str,
    task_id: str,
    sample_dao: SampleResultsDAO = Depends(get_sample_results_dao),
    miners_dao: MinersDAO = Depends(get_miners_dao),
    config_dao: SystemConfigDAO = Depends(get_system_config_dao),
    task_pool: TaskPoolManager = Depends(get_task_pool_manager),
):
    """
    Get a specific sample by UID, env, and task_id.
    
    Path parameters:
    - uid: Miner UID (0-255)
    - env: Environment (e.g., affine:sat or shorthand like sat, alfworld)
    - task_id: Task identifier
    
    Returns full sample details including conversation data.
    Automatically looks up the miner's current hotkey and revision.
    If not found in sample_results, tries to query from task_pool.
    """
    try:
        # Resolve env_name shorthand (e.g., 'alfworld' -> 'agentgym:alfworld')
        environments = await config_dao.get_param_value('environments', default={})
        
        # Check if env is already a full environment name
        if env not in environments:
            # Try to resolve shorthand (e.g., 'alfworld' -> 'agentgym:alfworld')
            if ':' not in env:
                matching_envs = [e for e in environments.keys() if e.endswith(f':{env}')]
                if len(matching_envs) == 0:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Environment not found: {env}. Available: {', '.join(environments.keys())}"
                    )
                elif len(matching_envs) > 1:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Ambiguous environment name: {env}. Matches: {', '.join(matching_envs)}"
                    )
                env = matching_envs[0]
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Environment not found: {env}. Available: {', '.join(environments.keys())}"
                )
        
        # Get miner info by UID
        miner = await miners_dao.get_miner_by_uid(uid)
        
        if not miner:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Miner not found for UID={uid}"
            )
        
        # Extract hotkey and revision
        hotkey = miner['hotkey']
        model_revision = miner['revision']
        
        # First, try to get from sample_results
        item = await sample_dao.get_sample_by_task_id(
            miner_hotkey=hotkey,
            model_revision=model_revision,
            env=env,
            task_id=task_id,
            include_extra=True
        )
        
        if item:
            return SampleFullResponse(**item)
        
        # If not found in sample_results, try task_pool
        task_id_int = int(task_id) if not isinstance(task_id, int) else task_id
        pool_task = await task_pool.dao.get_task_by_composite_key(
            miner_hotkey=hotkey,
            model_revision=model_revision,
            env=env,
            task_id=task_id_int
        )
        
        if pool_task:
            return TaskPoolResponse(**pool_task)
        
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sample not found in sample results or task pool for UID={uid}, env={env}, task_id={task_id}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve sample: {str(e)}"
        )


# Only enable get-pool endpoint for internal services (SERVICES_ENABLED=true)
if config.SERVICES_ENABLED:
    @router.get("/pool/uid/{uid}/{env}", dependencies=[Depends(rate_limit_read)])
    async def get_task_pool(
        uid: int,
        env: str,
        miners_dao: MinersDAO = Depends(get_miners_dao),
        task_pool: TaskPoolManager = Depends(get_task_pool_manager),
        sample_dao: SampleResultsDAO = Depends(get_sample_results_dao),
        config_dao: SystemConfigDAO = Depends(get_system_config_dao),
    ):
        """
        Get task pool status for a specific miner in an environment.

        Path parameters:
        - uid: Miner UID (0-255)
        - env: Environment (e.g., agentgym:webshop or shorthand like webshop, sat)

        Returns:
        - sampled_task_ids: Tasks already completed and in sample_results
        - pool_task_ids: Tasks currently in the task pool (pending/assigned)
        - missing_task_ids: Tasks not yet sampled and not in pool (based on sampling_list)
        """
        try:
            environments = await config_dao.get_param_value('environments', default={})

            # Check if env is already a full environment name
            if env not in environments:
                # Try to resolve shorthand (e.g., 'alfworld' -> 'agentgym:alfworld')
                if ':' not in env:
                    matching_envs = [e for e in environments.keys() if e.endswith(f':{env}')]
                    if len(matching_envs) == 0:
                        raise HTTPException(
                            status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Environment not found: {env}. Available: {', '.join(environments.keys())}"
                        )
                    elif len(matching_envs) > 1:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Ambiguous environment name: {env}. Matches: {', '.join(matching_envs)}"
                        )
                    env = matching_envs[0]
                else:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Environment not found: {env}. Available: {', '.join(environments.keys())}"
                    )

            # Get miner info by UID
            miner = await miners_dao.get_miner_by_uid(uid)

            if not miner:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Miner not found for UID={uid}"
                )

            # Extract hotkey and revision
            hotkey = miner['hotkey']
            model_revision = miner['revision']

            # Get task IDs from sampling_list or fallback to ranges
            from affine.core.sampling_list import get_task_id_set_from_config

            environments = await config_dao.get_param_value('environments', default={})
            if env not in environments:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Environment '{env}' not configured in system config"
                )

            env_config = environments[env]

            # Get all task IDs (prioritize sampling_list, fallback to ranges)
            all_task_ids = get_task_id_set_from_config(env_config)

            # Get sampling config
            sampling_config = env_config.get('sampling_config', {})

            # Get already sampled task_ids from sample_results
            sampled_task_ids = await sample_dao.get_completed_task_ids(
                miner_hotkey=hotkey,
                model_revision=model_revision,
                env=env
            )

            # Get tasks in the task pool
            tasks = await task_pool.dao.get_tasks_by_miner(
                miner_hotkey=hotkey,
                model_revision=model_revision,
                env=env
            )

            # Extract task_ids from pool tasks (pending/assigned)
            pool_task_ids = {
                task['task_id']
                for task in tasks
                if task.get('status') in ['pending', 'assigned']
            }

            # Calculate missing task_ids
            missing_task_ids = all_task_ids - sampled_task_ids - pool_task_ids

            return {
                "uid": uid,
                "hotkey": hotkey,
                "model_revision": model_revision,
                "env": env,
                "sampling_config": sampling_config,
                "total_tasks": len(all_task_ids),
                "sampled_count": len(sampled_task_ids),
                "pool_count": len(pool_task_ids),
                "missing_count": len(missing_task_ids),
                "sampled_task_ids": sorted(list(sampled_task_ids)),
                "pool_task_ids": sorted(list(pool_task_ids)),
                "missing_task_ids": sorted(list(missing_task_ids))
            }

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve task pool: {str(e)}"
            )


@router.get("/scoring", dependencies=[Depends(rate_limit_scoring)])
async def get_scoring_data(
    range_type: str = Query(
        "scoring",
        regex="^(scoring|sampling)$",
        description="Range type: 'scoring' for scoring_range or 'sampling' for sampling_range"
    )
):
    """
    Get scoring data for all valid miners.
    
    Query Parameters:
    - range_type: Type of range to use ('scoring' or 'sampling', default: 'scoring')
    
    Uses proactive cache with background refresh.
    - Startup: Cache prewarmed
    - Runtime: Background refresh every 20 minutes
    - Access: Always returns hot cache (< 100ms)
    """
    from affine.api.services.scoring_cache import get_cached_data
    
    try:
        return await get_cached_data(range_type=range_type)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get scoring data: {str(e)}"
        )