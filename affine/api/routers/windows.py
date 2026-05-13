"""
Read-only window state endpoints.

The directory + URL prefix is ``/windows`` for backward compat with
existing miner CLIs and external dashboards, but the underlying model
is flow-based — there's no "window archive" anymore; every endpoint
reads the current state directly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends

from affine.api.dependencies import rate_limit_read

from affine.database.dao.miners import MinersDAO
from affine.database.dao.scores import ScoresDAO
from affine.database.dao.system_config import SystemConfigDAO
from affine.src.scorer.dao_adapters import SampleResultsAdapter
from affine.src.scorer.window_state import (
    ChampionRecord,
    BattleRecord,
    StateStore,
    SystemConfigKVAdapter,
    TaskIdState,
)


router = APIRouter(
    prefix="/windows",
    tags=["windows"],
    dependencies=[Depends(rate_limit_read)],
)


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


async def _infer_champion_from_scores() -> Optional[ChampionRecord]:
    latest = await ScoresDAO().get_latest_scores(limit=None)
    rows = latest.get("scores") or []
    champions = [
        row for row in rows
        if float(row.get("overall_score") or 0.0) > 0.0
    ]
    if len(champions) != 1:
        return None
    row = champions[0]
    uid = row.get("uid")
    hotkey = row.get("miner_hotkey")
    revision = row.get("model_revision")
    model = row.get("model")
    if uid is None or not hotkey or not revision or not model:
        return None
    return ChampionRecord(
        uid=int(uid),
        hotkey=str(hotkey),
        revision=str(revision),
        model=str(model),
        since_block=int(latest.get("block_number") or 0),
    )


async def _sample_counts(
    champion: Optional[ChampionRecord],
    battle: Optional[BattleRecord],
    task_state: Optional[TaskIdState],
) -> Dict[str, Dict[str, int]]:
    if task_state is None:
        return {}
    adapter = SampleResultsAdapter()
    out: Dict[str, Dict[str, int]] = {}

    subjects = []
    if champion is not None:
        subjects.append((str(champion.uid), champion.hotkey, champion.revision))
    if battle is not None:
        challenger = battle.challenger
        subjects.append((str(challenger.uid), challenger.hotkey, challenger.revision))

    for uid, hotkey, revision in subjects:
        env_counts: Dict[str, int] = {}
        for env, task_ids in task_state.task_ids.items():
            env_counts[env] = await adapter.count_samples_for_tasks(
                hotkey,
                revision,
                env,
                task_ids,
                refresh_block=task_state.refreshed_at_block,
            )
        out[uid] = env_counts
    return out


@router.get("/current")
async def get_current_state() -> Dict[str, Any]:
    """Live public snapshot: champion, in-flight battle (if any), task-id
    refresh block, and aggregate sample counts. Deployment URLs and task IDs
    stay internal."""
    store = _state_store()
    champion = await store.get_champion()
    if champion is None:
        champion = await _infer_champion_from_scores()
    battle = await store.get_battle()
    task_state = await store.get_task_state()
    return {
        "champion": _miner_summary(champion) if champion else None,
        "battle": {
            "challenger": _miner_summary(battle.challenger),
            "started_at_block": battle.started_at_block,
        } if battle else None,
        "task_refresh_block": task_state.refreshed_at_block if task_state else None,
        "sample_counts": await _sample_counts(champion, battle, task_state),
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
