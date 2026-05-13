"""Read-only miner metadata endpoints."""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, status

from affine.api.dependencies import get_miners_dao, rate_limit_read
from affine.api.models import MinerInfo
from affine.database.dao.miners import MinersDAO


router = APIRouter(
    prefix="/miners",
    tags=["Miners"],
    dependencies=[Depends(rate_limit_read)],
)


def _bool_or_none(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return None


def _miner_info(row: Dict[str, Any]) -> MinerInfo:
    return MinerInfo(
        uid=int(row["uid"]),
        hotkey=row["hotkey"],
        model=row.get("model"),
        revision=row.get("revision"),
        is_valid=_bool_or_none(row.get("is_valid")),
        challenge_status=row.get("challenge_status"),
        termination_reason=row.get("termination_reason"),
        first_block=row.get("first_block"),
        block_number=row.get("block_number"),
        invalid_reason=row.get("invalid_reason"),
        model_hash=row.get("model_hash"),
    )


@router.get("/uid/{uid}", response_model=MinerInfo)
async def get_miner_by_uid(
    uid: int,
    dao: MinersDAO = Depends(get_miners_dao),
) -> MinerInfo:
    row = await dao.get_miner_by_uid(uid)
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Miner not found for UID={uid}",
        )
    return _miner_info(row)


@router.get("/hotkey/{hotkey}", response_model=MinerInfo)
async def get_miner_by_hotkey(
    hotkey: str,
    dao: MinersDAO = Depends(get_miners_dao),
) -> MinerInfo:
    row = await dao.get_miner_by_hotkey(hotkey)
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Miner not found for hotkey={hotkey}",
        )
    return _miner_info(row)
