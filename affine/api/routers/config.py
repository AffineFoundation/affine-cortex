"""
Configuration Management Router

Provides REST API endpoints for dynamic configuration management.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
from affine.database.dao.system_config import SystemConfigDAO
from affine.api.dependencies import rate_limit_read

router = APIRouter(prefix="/config", tags=["config"])
config_dao = SystemConfigDAO()


@router.get("", dependencies=[Depends(rate_limit_read)])
async def get_all_configs(prefix: Optional[str] = None):
    """Get all configurations, optionally filtered by prefix.
    
    Args:
        prefix: Config key prefix filter (e.g., "scheduler.")
        
    Returns:
        Dictionary of all matching configs

    Example:
        GET /api/v1/config?prefix=scheduler
        Returns all scheduler.* configs
    """
    all_configs = await config_dao.get_all_params()
    
    if prefix:
        filtered = {k: v for k, v in all_configs.items() if k.startswith(prefix)}
        return {"configs": filtered}
    
    return {"configs": all_configs}


@router.get("/{key}", dependencies=[Depends(rate_limit_read)])
async def get_config(key: str):
    """Get a single configuration parameter.

    Args:
        key: Configuration key

    Returns:
        Full config item with metadata

    Raises:
        404: Config not found

    Example:
        GET /api/v1/config/environments
        GET /api/v1/config/miner_blacklist
    """
    config = await config_dao.get_param(key)

    if not config:
        raise HTTPException(status_code=404, detail=f"Config '{key}' not found")

    # Filter out sampling_list from environments config
    if key == "environments" and config.get("param_value"):
        filtered_envs = {}
        for env_name, env_config in config["param_value"].items():
            filtered_config = {k: v for k, v in env_config.items() if k != "sampling_list"}
            # Also filter sampling_list from nested sampling_config
            if "sampling_config" in filtered_config and isinstance(filtered_config["sampling_config"], dict):
                filtered_config["sampling_config"] = {
                    k: v for k, v in filtered_config["sampling_config"].items()
                    if k != "sampling_list"
                }
            filtered_envs[env_name] = filtered_config
        config["param_value"] = filtered_envs

    return config