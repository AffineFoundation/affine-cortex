"""
Read-only window state endpoints.

The directory + URL prefix is ``/windows`` for backward compat with
existing miner CLIs and external dashboards, but the underlying model
is flow-based — there's no "window archive" anymore; every endpoint
reads the current state directly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter

from affine.database.dao.miners import MinersDAO
from affine.database.dao.system_config import SystemConfigDAO
from affine.src.scorer.window_state import (
    StateStore,
    SystemConfigKVAdapter,
)


router = APIRouter(prefix="/windows", tags=["windows"])


def _state_store() -> StateStore:
    return StateStore(SystemConfigKVAdapter(SystemConfigDAO(), updated_by="api"))


def _miner_summary(snapshot) -> Optional[Dict[str, Any]]:
    if snapshot is None:
        return None
    return {
        "uid": snapshot.uid,
        "hotkey": snapshot.hotkey,
        "revision": snapshot.revision,
        "model": snapshot.model,
    }


@router.get("/current")
async def get_current_state() -> Dict[str, Any]:
    """Live snapshot: champion, in-flight battle (if any), task-id refresh
    block. No sample-level data."""
    store = _state_store()
    champion = await store.get_champion()
    battle = await store.get_battle()
    task_state = await store.get_task_state()
    return {
        "champion": _miner_summary(champion) if champion else None,
        "champion_base_url": champion.base_url if champion else None,
        "battle": {
            "challenger": _miner_summary(battle.challenger),
            "started_at_block": battle.started_at_block,
        } if battle else None,
        "task_refresh_block": task_state.refreshed_at_block if task_state else None,
    }


@router.get("/queue")
async def get_queue(limit: int = 20) -> List[Dict[str, Any]]:
    """Challenger queue head: pending miners ordered by (first_block, uid)."""
    if limit <= 0 or limit > 100:
        limit = 20
    miners = await MinersDAO().get_valid_miners()
    pending = [m for m in miners if m.get("challenge_status") == "pending"]
    pending.sort(key=lambda m: (m.get("first_block", float("inf")), m.get("uid", 0)))
    out: List[Dict[str, Any]] = []
    for i, m in enumerate(pending[:limit]):
        out.append(
            {
                "position": i + 1,
                "uid": int(m.get("uid", -1)),
                "hotkey": m.get("hotkey", ""),
                "revision": m.get("revision", ""),
                "model": m.get("model", ""),
                "first_block": m.get("first_block"),
                "enqueued_at": m.get("enqueued_at"),
            }
        )
    return out


@router.get("/champion")
async def get_champion() -> Dict[str, Any]:
    """Current champion identity (mirrors ``system_config['champion']``)."""
    champ = await _state_store().get_champion()
    if champ is None:
        return {"uid": None}
    return {
        "uid": champ.uid,
        "hotkey": champ.hotkey,
        "revision": champ.revision,
        "model": champ.model,
        "since_block": champ.since_block,
    }
