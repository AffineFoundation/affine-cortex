"""Read-only public configuration endpoints."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from affine.api.dependencies import rate_limit_read
from affine.database.dao.system_config import SystemConfigDAO


router = APIRouter(prefix="/config", tags=["config"])
config_dao = SystemConfigDAO()

PUBLIC_CONFIG_KEYS = {
    "validator_burn_percentage",
}


@router.get("", dependencies=[Depends(rate_limit_read)])
async def get_all_configs(prefix: Optional[str] = None):
    """Get public config keys, optionally filtered by prefix."""
    all_configs = await config_dao.get_all_params()
    public_configs = {
        k: v for k, v in all_configs.items()
        if k in PUBLIC_CONFIG_KEYS
    }
    if prefix:
        return {
            "configs": {
                k: v for k, v in public_configs.items()
                if k.startswith(prefix)
            }
        }
    return {"configs": public_configs}


@router.get("/{key}", dependencies=[Depends(rate_limit_read)])
async def get_config(key: str):
    """Get a single public config row by key."""
    if key not in PUBLIC_CONFIG_KEYS:
        raise HTTPException(status_code=404, detail=f"Config '{key}' not found")
    config = await config_dao.get_param(key)
    if not config:
        raise HTTPException(status_code=404, detail=f"Config '{key}' not found")
    return config
