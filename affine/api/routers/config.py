"""
Configuration router.

Surfaces a read-only view of ``system_config``. Every value the flow
scheduler writes (``champion``, ``current_battle``, ``current_task_ids``,
``environments``, ``validator_burn_percentage``, ``miner_blacklist``,
``system_miners``) is exposed as-is. No filtering — the legacy
sampling-list-leak shield is unnecessary because the new ``environments``
shape contains no per-window task ids at all.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from affine.api.dependencies import rate_limit_read
from affine.database.dao.system_config import SystemConfigDAO


router = APIRouter(prefix="/config", tags=["config"])
config_dao = SystemConfigDAO()


@router.get("", dependencies=[Depends(rate_limit_read)])
async def get_all_configs(prefix: Optional[str] = None):
    """Get every config key, or only those starting with ``prefix``."""
    all_configs = await config_dao.get_all_params()
    if prefix:
        return {"configs": {k: v for k, v in all_configs.items() if k.startswith(prefix)}}
    return {"configs": all_configs}


@router.get("/{key}", dependencies=[Depends(rate_limit_read)])
async def get_config(key: str):
    """Get a single config row by key."""
    config = await config_dao.get_param(key)
    if not config:
        raise HTTPException(status_code=404, detail=f"Config '{key}' not found")
    return config
