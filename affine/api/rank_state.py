"""Internal helpers for the public rank payload."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from affine.database.dao.miner_stats import MinerStatsDAO
from affine.database.dao.miners import MinersDAO
from affine.database.dao.scores import ScoresDAO
from affine.database.dao.system_config import SystemConfigDAO
from affine.src.monitor.live_scores_monitor import LIVE_SCORES_KEY
from affine.src.scorer.window_state import (
    BattleRecord,
    ChampionRecord,
    EnvConfig,
    StateStore,
    SystemConfigKVAdapter,
    TaskIdState,
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


async def _sample_counts_and_averages(
    task_state: Optional[TaskIdState],
) -> tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, float]]]:
    """Per-(uid, env) live count + running average for every valid miner.

    Reads the precomputed cache at ``system_config['live_scores']`` —
    populated by :class:`affine.src.monitor.live_scores_monitor.LiveScoresMonitor`
    on a fixed cadence (~30 min). Returning empty dicts is fine: the
    rank UI then renders ``-`` instead of a stale snapshot value, which
    is exactly what we want when the monitor hasn't produced a payload
    yet (or has produced one for a different refresh_block).

    The cache is keyed by ``refresh_block``. When the scheduler refreshes
    the task pool between monitor cycles the cached entry is treated as
    expired and dropped — readers must not see scores belonging to a
    previous pool.
    """
    if task_state is None:
        return {}, {}
    payload = await SystemConfigDAO().get_param_value(LIVE_SCORES_KEY, default=None)
    if not isinstance(payload, dict):
        return {}, {}
    if int(payload.get("refresh_block") or 0) != int(task_state.refreshed_at_block):
        return {}, {}
    raw = payload.get("scores") or {}
    if not isinstance(raw, dict):
        return {}, {}

    counts: Dict[str, Dict[str, int]] = {}
    averages: Dict[str, Dict[str, float]] = {}
    for uid_key, env_map in raw.items():
        if not isinstance(env_map, dict):
            continue
        uid = str(uid_key)
        env_counts: Dict[str, int] = {}
        env_avgs: Dict[str, float] = {}
        for env, entry in env_map.items():
            if not isinstance(entry, dict):
                continue
            try:
                env_counts[str(env)] = int(entry.get("count") or 0)
                env_avgs[str(env)] = float(entry.get("avg") or 0.0)
            except (TypeError, ValueError):
                continue
        if env_counts:
            counts[uid] = env_counts
            averages[uid] = env_avgs
    return counts, averages


def _live_sampling_uids(
    champion: Optional[ChampionRecord],
    battle: Optional[BattleRecord],
    task_state: Optional[TaskIdState],
    envs: Dict[str, EnvConfig],
    sample_counts: Dict[str, Dict[str, int]],
) -> List[int]:
    if task_state is None:
        return []
    out: List[int] = []

    def _is_active(uid: int) -> bool:
        counts = sample_counts.get(str(uid)) or {}
        for env, cfg in envs.items():
            task_ids = task_state.task_ids.get(env) or []
            if not task_ids:
                continue
            target = min(len(task_ids), int(cfg.sampling_count))
            if target > 0 and int(counts.get(env) or 0) < target:
                return True
        return False

    if champion is not None and _is_active(champion.uid):
        out.append(champion.uid)
    if battle is not None and _is_active(battle.challenger.uid):
        out.append(battle.challenger.uid)
    return out


async def get_current_state() -> Dict[str, Any]:
    """Build the live state section used by ``/rank/current``."""
    store = _state_store()
    champion = await store.get_champion()
    if champion is None:
        champion = await _infer_champion_from_scores()
    battle = await store.get_battle()
    task_state = await store.get_task_state()
    envs = await store.get_environments()
    sample_counts, sample_averages = await _sample_counts_and_averages(task_state)
    return {
        "champion": _miner_summary(champion) if champion else None,
        "battle": {
            "challenger": _miner_summary(battle.challenger),
            "started_at_block": battle.started_at_block,
        } if battle else None,
        "task_refresh_block": task_state.refreshed_at_block if task_state else None,
        "sample_counts": sample_counts,
        # Per-(uid, env) running average over the current refresh_block.
        # Battle subjects show their live score in af get-rank instead
        # of the (stale) last-decided snapshot's 0.00 placeholder.
        "sample_averages": sample_averages,
        "live_sampling_uids": _live_sampling_uids(
            champion, battle, task_state, envs, sample_counts,
        ),
    }


async def get_queue(limit: int = 20) -> List[Dict[str, Any]]:
    """Build the challenger queue head used by ``/rank/current``."""
    if limit <= 0 or limit > 100:
        limit = 20
    miners = await MinersDAO().get_valid_miners()
    state_map = await MinerStatsDAO().build_challenge_state_map(miners)
    pending = []
    for miner in miners:
        state = state_map.get((miner.get("hotkey"), miner.get("revision"))) or {}
        status = str(state.get("challenge_status") or "sampling")
        if status == "sampling":
            row = dict(miner)
            row["challenge_status"] = status
            row["termination_reason"] = state.get("termination_reason") or None
            pending.append(row)
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
                "challenge_status": m.get("challenge_status"),
                "termination_reason": m.get("termination_reason"),
            }
        )
    return out
