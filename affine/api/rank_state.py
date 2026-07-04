"""Internal helpers for the public rank payload."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from affine.database.dao.miner_stats import MinerStatsDAO
from affine.database.dao.miners import MinersDAO
from affine.database.dao.score_snapshots import ScoreSnapshotsDAO
from affine.database.dao.scores import ScoresDAO
from affine.database.dao.system_config import SystemConfigDAO
from affine.src.scorer.window_state import (
    BattleRecord,
    ChampionRecord,
    StateStore,
    SystemConfigKVAdapter,
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
        "model_type": getattr(snapshot, "model_type", None),
        # ``since_block`` is on ``ChampionRecord`` (the canonical champion
        # in system_config) but not on the snapshot-inferred fallback
        # (``_infer_champion_from_scores`` builds a synthetic record
        # with ``since_block=int(latest.block_number)``). Either way the
        # field is exposed for the CLI to show how long the champion has
        # held the crown.
        "since_block": getattr(snapshot, "since_block", None),
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
        model_type=str(row.get("model_type") or ""),
        since_block=int(latest.get("block_number") or 0),
    )


def _split_display_scores(
    display_map: Dict[str, Dict[str, Any]],
) -> tuple[
    Dict[str, Dict[str, int]],
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, Dict[str, float]]],
]:
    """Project ``build_display_scores_map`` output into the four payload
    dicts the rank API exposes.

    The storage is one ``scores_by_env`` per miner; the ``frozen`` flag
    on each entry tells us whether it was written by the comparator at
    termination (frozen=True → exposed as ``terminal_scores``) or by
    :class:`LiveScoresMonitor` at the current refresh_block
    (frozen=False → split into the live ``sample_counts`` /
    ``sample_averages`` / ``champion_overlap_avgs`` dicts the CLI
    already consumes for active rows).
    """
    counts: Dict[str, Dict[str, int]] = {}
    averages: Dict[str, Dict[str, float]] = {}
    overlap_avgs: Dict[str, Dict[str, float]] = {}
    terminal: Dict[str, Dict[str, Dict[str, float]]] = {}
    for uid, entry in display_map.items():
        scores = entry.get("scores") or {}
        if not isinstance(scores, dict) or not scores:
            continue
        if entry.get("frozen"):
            terminal[uid] = scores
            continue
        env_counts: Dict[str, int] = {}
        env_avgs: Dict[str, float] = {}
        env_overlap: Dict[str, float] = {}
        for env, item in scores.items():
            if not isinstance(item, dict):
                continue
            try:
                env_counts[str(env)] = int(item.get("count") or 0)
                env_avgs[str(env)] = float(item.get("avg") or 0.0)
            except (TypeError, ValueError):
                continue
            raw_overlap = item.get("champion_overlap_avg")
            if isinstance(raw_overlap, (int, float)):
                env_overlap[str(env)] = float(raw_overlap)
        if env_counts:
            counts[uid] = env_counts
            averages[uid] = env_avgs
        if env_overlap:
            overlap_avgs[uid] = env_overlap
    return counts, averages, overlap_avgs, terminal


def _live_sampling_uids(
    champion: Optional[ChampionRecord],
    battle: Optional[BattleRecord],
    predeployed: List[BattleRecord],
) -> List[int]:
    """UIDs with a live inference deployment in the active roster.

    ⚡ marks deployment liveness, NOT sampling progress. A role object
    carrying a non-empty ``deployment_id`` (or a ``deployments`` entry) is
    the authoritative, locally known signal that the scheduler has an
    endpoint up for that miner in the current roster (champion / battle
    challenger / pre-deployed). It's read straight off state already loaded
    for this render — no ``sample_results`` GSI look-up, nothing to flap, no
    recency window to tune. ``BattleRecord.deployment_id`` is always set; the
    champion's is Optional (None before its first deploy / during cold-start).

    This tracks the host occupant correctly across the lifecycle: on
    single-instance providers (SSH/sglang) ``deployment_id`` is cleared when
    the host is re-tasked (see flow.py), so a champion frozen at e.g.
    1425/1461 while a battle borrows b300 correctly loses ⚡; on multi-instance
    providers (Targon) the champion owns its endpoint for its full reign, so
    ⚡ persists until a challenger wins.

    What ⚡ can't catch is a deployment that stays up but stops producing (a
    multi-instance champion mid-reign, or a single-instance champion holding
    the host while its samples keep erroring, e.g. SWE-INFINITE "Failed to
    pull image"). ⚡ marks the endpoint as up, not the writes as flowing —
    diagnosing *why* a live deployment stopped producing is the job of the
    deploy ``__cause__`` chain now in the scheduler log. Detecting
    producing-vs-stalled would need per-write recency, which has no usable
    index on sample_results (the declared timestamp-index GSI doesn't exist
    on the live table) and would otherwise cost a per-render query, blank a
    freshly-deployed challenger until its first row, and fail closed on a DDB
    blip.

    Roles are visited in stable declaration order; a uid appearing under more
    than one role (pathological, crash-recovery ticks only) is deduplicated
    keeping first occurrence.
    """
    # A role is "deployed" when it carries a primary ``deployment_id`` OR any
    # entry in its ``deployments`` list. deployment_id is normally backfilled
    # from deployments[0] on load (see _champion_from_dict/_battle_from_dict),
    # but we check both so the icon can't miss a multi-deployment role whose
    # primary field happened not to be populated.
    def _deployed(rec: Any) -> bool:
        return bool(getattr(rec, "deployment_id", None)
                    or getattr(rec, "deployments", None))

    subjects: List[Tuple[int, bool]] = []
    if champion is not None:
        subjects.append((champion.uid, _deployed(champion)))
    if battle is not None:
        subjects.append((battle.challenger.uid, _deployed(battle)))
    for record in predeployed:
        subjects.append((record.challenger.uid, _deployed(record)))

    seen: set = set()
    out: List[int] = []
    for uid, deployed in subjects:
        if deployed and uid not in seen:
            seen.add(uid)
            out.append(uid)
    return out


async def _past_champions_split() -> List[Dict[str, Any]]:
    """Past-N champions currently sharing weight, for ``af get-rank``.

    Returns an empty list when:

      * the split feature is off (``system_config['weights_split_after_block']``
        missing or ``0``), or
      * no eligible past champion currently maps to a registered, valid
        miner, or
      * any DDB-level error occurs while resolving the split — rank
        rendering is a read surface and should not fail just because
        the split lookup transiently can't be served.

    Shape per row: ``{"uid", "hotkey", "model", "revision", "share"}``.
    ``share`` is the same value the validator sets on chain (``1/N``),
    not the snapshot's stale ``overall_score``.
    """
    # Local import to avoid a circular ``routers.scores`` ↔ ``rank_state``
    # cycle at module load (the scores router itself imports nothing from
    # this module today, but keeping the import local insulates against
    # future churn).
    from affine.api.routers.scores import compute_split_payees

    try:
        payees = await compute_split_payees(
            ScoreSnapshotsDAO(), MinersDAO(), SystemConfigDAO(),
        )
    except Exception:
        return []
    return payees or []


async def get_current_state() -> Dict[str, Any]:
    """Build the live state section used by ``/rank/current``."""
    store = _state_store()
    champion = await store.get_champion()
    if champion is None:
        champion = await _infer_champion_from_scores()
    battle = await store.get_battle()
    predeployed = await store.get_predeployed_challengers()
    task_state = await store.get_task_state()
    envs = await store.get_environments()
    # One read per miner pulls both live (LiveScoresMonitor) and frozen
    # (decide-time) scores from miner_stats; the DAO drops live entries
    # not matching ``current_refresh_block``.
    #
    # Uses ``get_all_miners`` (not valid only) so a miner that was
    # terminated and later turned invalid keeps its frozen scores in
    # the rank table.
    all_miners = await MinersDAO().get_all_miners()
    if champion is not None and not any(
        m.get("hotkey") == champion.hotkey
        and str(m.get("revision") or "") == champion.revision
        for m in all_miners
    ):
        all_miners = list(all_miners) + [{
            "uid": champion.uid,
            "hotkey": champion.hotkey,
            "revision": champion.revision,
            "model": champion.model,
            "model_type": getattr(champion, "model_type", ""),
            "first_block": champion.since_block,
            "is_valid": "false",
            "invalid_reason": "deregistered",
        }]
    current_refresh = (
        int(task_state.refreshed_at_block) if task_state else None
    )
    display_map = await MinerStatsDAO().build_display_scores_map(
        all_miners, current_refresh_block=current_refresh,
    )
    sample_counts, sample_averages, champion_overlap_avgs, terminal_scores = (
        _split_display_scores(display_map)
    )
    past_champions = await _past_champions_split()
    return {
        "champion": _miner_summary(champion) if champion else None,
        # Past N champions currently sharing the on-chain weight. Empty
        # list when the split feature is off; otherwise the most recent
        # N distinct hotkeys, resolved to their current uid and each
        # carrying ``share = 1/N``. ``af get-rank`` surfaces this in a
        # dedicated table column / header line so operators can see who
        # is being paid and at what proportion.
        "past_champions": past_champions,
        "battle": {
            "challenger": _miner_summary(battle.challenger),
            "started_at_block": battle.started_at_block,
        } if battle else None,
        "task_refresh_block": task_state.refreshed_at_block if task_state else None,
        "enabled_envs": list(envs.keys()),
        "sample_counts": sample_counts,
        # Per-(uid, env) running average over the current refresh_block.
        # Battle subjects show their live score in af get-rank instead
        # of the (stale) last-decided snapshot's 0.00 placeholder.
        "sample_averages": sample_averages,
        # Per-(uid, env) champion-on-overlap avg from the live cache.
        # Used as the per-row threshold basis for currently-battling
        # challengers; absence → CLI falls back to the champion's
        # full-set avg.
        "champion_overlap_avgs": champion_overlap_avgs,
        # Per-(uid, env) frozen decide-time snapshot for terminated
        # miners: ``{uid_str: {env: {count, avg, champion_overlap_avg}}}``.
        # The CLI uses these as the fallback when the live cache has no
        # entry for the row.
        "terminal_scores": terminal_scores,
        "live_sampling_uids": _live_sampling_uids(champion, battle, predeployed),
    }


async def get_queue(limit: int = 256) -> List[Dict[str, Any]]:
    """Build the challenger queue head used by ``/rank/current``.

    Default and upper bound are both the active subnet ceiling (256
    UIDs on netuid 120) so ``af get-rank`` can render every eligible
    miner as part of the queue, not just the head. Callers wanting a
    shorter slice still pass an explicit smaller ``limit``."""
    if limit <= 0 or limit > 256:
        limit = 256
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
                "model_type": m.get("model_type", ""),
                "first_block": m.get("first_block"),
                "enqueued_at": m.get("enqueued_at"),
                "challenge_status": m.get("challenge_status"),
                "termination_reason": m.get("termination_reason"),
            }
        )
    return out
