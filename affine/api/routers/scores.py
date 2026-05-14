"""
Scores Router

Endpoints for querying score calculations.
"""

import asyncio
import time as _time
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Query, status
from affine.api.models import (
    ScoresResponse,
    MinerScore,
)
from affine.api.dependencies import (
    get_scores_dao,
    get_score_snapshots_dao,
    get_miners_dao,
    get_system_config_dao,
    rate_limit_read,
)
from affine.database.dao.scores import ScoresDAO
from affine.database.dao.score_snapshots import ScoreSnapshotsDAO
from affine.database.dao.miner_stats import MinerStatsDAO
from affine.database.dao.miners import MinersDAO
from affine.database.dao.sample_results import SampleResultsDAO
from affine.database.dao.system_config import SystemConfigDAO
from affine.src.scorer.window_state import StateStore

router = APIRouter(prefix="/scores", tags=["Scores"])


# Process-local cache for per-(hotkey, revision) score aggregates.
# Each entry stores ``(by_env, total, cached_at)``; the caller
# decides freshness at lookup time via ``cache_ttl_s`` so a status
# transition (e.g. sampling → in_progress) automatically invalidates
# a stale entry instead of being trapped by the TTL stored at write
# time. Process-local; survives until the API process restarts. Size
# capped at ~256 keys × ~200B = under 0.5 MB total.
_AVG_CACHE: Dict[
    Tuple[str, str], Tuple[Dict[str, Dict[str, Any]], int, float],
] = {}

# Active and queue-waiting miners refresh at most this often. 10 min
# keeps rank scores within one battle's natural tempo while making
# API request cost effectively independent of request rate (one
# query per miner per 10 min, regardless of caller count).
_ACTIVE_CACHE_TTL_S = 600


def _reset_terminated_cache_for_test() -> None:
    """Test-only hook: clear the cross-request avg cache so each test
    starts from a known state."""
    _AVG_CACHE.clear()


def _ttl_for_status(challenge_status: Optional[str]) -> Optional[float]:
    """Cache TTL for a miner's per-(hotkey, revision) aggregate.

    ``terminated`` is the only status for which the underlying samples
    are truly frozen — the miner can no longer be sampled (one-shot
    challenge already used; re-commit creates a new revision = a
    different cache key). Returns ``None`` (cache forever).

    Every other status (``sampling``, ``in_progress``, ``champion``)
    can transition into an actively-sampling state, so we use a
    finite TTL — caching ``sampling`` permanently would trap an
    empty result and shadow new samples once the miner gets picked.
    """
    if challenge_status == "terminated":
        return None
    return _ACTIVE_CACHE_TTL_S


class _StaticConfigStore:
    def __init__(self, data: dict):
        self._data = data

    async def get(self, key: str, default=None):
        return self._data.get(key, default)

    async def set(self, key: str, value) -> None:
        self._data[key] = value

    async def delete(self, key: str) -> bool:
        return self._data.pop(key, None) is not None


async def _build_validity_map() -> dict:
    """Map hotkey -> (is_valid, invalid_reason, challenge_status,
    termination_reason) from current miners plus historical miner_stats.

    Surfaces validator-side state in the rank UI:
      - ``is_valid`` flags monitor-side rejections (model_mismatch,
        anticopy, multiple_commits, …)
      - ``challenge_status`` comes from ``miner_stats`` so pre-refactor
        terminated miners remain terminated even if they leave/rejoin the
        active 256-UID snapshot.

    This map is also the current-online filter for rank output: score rows
    whose hotkey is absent from the current ``miners`` snapshot are historical
    rows and are not shown by ``af get-rank``. One full scan (256-row cap →
    cheap).
    """
    miners = await MinersDAO().get_all_miners()
    states = await MinerStatsDAO().build_challenge_state_map(miners)
    out: dict = {}
    for m in miners:
        hk = m.get("hotkey")
        if not hk:
            continue
        # is_valid stored as 'true'/'false' string in the GSI partition key.
        is_valid = str(m.get("is_valid") or "").lower() == "true"
        state = states.get((m.get("hotkey"), m.get("revision"))) or {}
        out[hk] = (
            is_valid,
            m.get("invalid_reason") or None,
            state.get("challenge_status") or None,
            state.get("termination_reason") or None,
        )
    return out


async def _avg_scores_for_miner(
    samples_dao: SampleResultsDAO,
    hotkey: str,
    revision: str,
    envs: List[str],
    *,
    cache_ttl_s: Optional[float] = None,
) -> Tuple[Dict[str, Dict[str, Any]], int]:
    """Per-env avg + total sample count from sample_results for one miner.

    Used to fill ``scores_by_env`` for any miner whose current entry in
    the latest decided snapshot is empty (which is everyone except the
    last-decided champion + previous champion). Sample_results spans
    every refresh_block within the table's 30-day TTL, so even
    terminated and post-snapshot miners get real averages instead of
    ``0.00`` placeholders.

    ``cache_ttl_s``:
      - ``None`` → permanent cache hit if any entry exists (only
        valid for terminated miners whose rows are frozen).
      - finite → cache entry is reused if it's younger than this
        many seconds; otherwise we re-query and overwrite. The
        check uses the *caller's* TTL, not anything stored at
        write time, so a status transition (sampling →
        in_progress) naturally invalidates the previous cached
        empty result on the next lookup past TTL.
    """
    if not hotkey or not revision or not envs:
        return {}, 0
    key = (hotkey, revision)
    now = _time.time()
    entry = _AVG_CACHE.get(key)
    if entry is not None:
        by_env, total, cached_at = entry
        if cache_ttl_s is None or (now - cached_at) < cache_ttl_s:
            return by_env, total
    by_env = await samples_dao.get_avg_scores_for_envs(hotkey, revision, envs)
    total = sum(int(p.get("sample_count", 0)) for p in by_env.values())
    _AVG_CACHE[key] = (by_env, total, now)
    return by_env, total


async def _build_score_rows_from_validity(
    validity: dict,
    snapshot_scores: List[Dict[str, Any]],
    scoring_envs: Optional[set],
    samples_dao: SampleResultsDAO,
) -> List[Dict[str, Any]]:
    """Score rows for *every* currently-valid miner.

    Uses ``validity`` (from MinersDAO) as the source-of-truth list, not
    the latest decided snapshot — so newly-online miners and miners
    that joined after the last decide still appear in the rank table
    with their real per-env averages. Snapshot data is layered in on
    top to preserve ``overall_score`` / ``average_score`` ordering for
    miners that did appear in the last decided snapshot.

    Per-env averages come from sample_results aggregation
    (:meth:`SampleResultsDAO.get_avg_scores_for_envs`), which is the
    only place where every miner's actual scores live (the snapshot
    only populates the champion + previous champion).
    """
    snapshot_by_hk: Dict[str, Dict[str, Any]] = {}
    for s in snapshot_scores:
        hk = s.get("miner_hotkey")
        if hk:
            snapshot_by_hk[hk] = s

    miner_meta = {
        m.get("hotkey"): m
        for m in await MinersDAO().get_all_miners()
        if m.get("hotkey")
    }

    env_list = sorted(scoring_envs) if scoring_envs else []

    async def _row(hk: str) -> Dict[str, Any]:
        snap = snapshot_by_hk.get(hk) or {}
        meta = miner_meta.get(hk) or {}
        revision = snap.get("model_revision") or meta.get("revision") or ""
        # validity tuple: (is_valid, invalid_reason, challenge_status, ...)
        challenge_status = (validity.get(hk) or (None, None, None, None))[2]
        agg_scores, agg_total = await _avg_scores_for_miner(
            samples_dao, hk, str(revision), env_list,
            cache_ttl_s=_ttl_for_status(challenge_status),
        )
        # Snapshot's scores_by_env wins when present (champion / previous
        # champion got real comparator-derived metadata at decide time).
        # Aggregated sample_results fills in everyone else.
        snap_env = snap.get("scores_by_env") or {}
        merged_env: Dict[str, Any] = {}
        for env in env_list:
            if env in snap_env and isinstance(snap_env[env], dict):
                merged_env[env] = snap_env[env]
            elif env in agg_scores:
                merged_env[env] = agg_scores[env]
        # Use snapshot's overall_score if present (preserves the
        # 1.0-for-champion sort key the comparator wrote); otherwise
        # default to 0.0 so newly-online miners sort last instead of
        # crashing the sort path.
        overall = snap.get("overall_score")
        if not isinstance(overall, (int, float)):
            overall = 0.0
        avg_score = snap.get("average_score")
        if not isinstance(avg_score, (int, float)):
            # Mean of the per-env averages we just aggregated; falls
            # back to 0.0 when the miner has no samples at all.
            envs_with_score = [
                p["score"] for p in agg_scores.values()
                if isinstance(p, dict) and isinstance(p.get("score"), (int, float))
            ]
            avg_score = (
                sum(envs_with_score) / len(envs_with_score)
                if envs_with_score else 0.0
            )
        # Total samples: prefer snapshot value if present, else fall back
        # to the aggregated count from sample_results.
        total_samples = snap.get("total_samples")
        if not isinstance(total_samples, int):
            total_samples = agg_total
        uid_value = snap.get("uid")
        if not isinstance(uid_value, int):
            uid_value = meta.get("uid") if isinstance(meta.get("uid"), int) else -1
        first_block = snap.get("first_block") or meta.get("first_block") or 0
        return {
            "miner_hotkey": hk,
            "uid": int(uid_value),
            "model": str(snap.get("model") or meta.get("model") or ""),
            "model_revision": str(revision or ""),
            "first_block": int(first_block),
            "overall_score": float(overall),
            "average_score": float(avg_score),
            "scores_by_env": merged_env,
            "total_samples": int(total_samples),
        }

    rows = await asyncio.gather(*(_row(hk) for hk in validity.keys()))
    return list(rows)


async def _scoring_env_names() -> set[str] | None:
    raw = await SystemConfigDAO().get_param_value("environments", default=None)
    if raw is None:
        return None
    if not isinstance(raw, dict):
        return set()
    envs = await StateStore(_StaticConfigStore({"environments": raw})).get_scoring_environments()
    return set(envs.keys())


def _filter_scores_by_env(scores_by_env, scoring_envs: set[str] | None):
    if not isinstance(scores_by_env, dict):
        return {}
    if scoring_envs is None:
        return scores_by_env
    return {
        env: payload
        for env, payload in scores_by_env.items()
        if env in scoring_envs
    }


@router.get("/latest", response_model=ScoresResponse, dependencies=[Depends(rate_limit_read)])
async def get_latest_scores(
    top: int = Query(32, description="Return top N miners by score", ge=1, le=256),
    dao: ScoresDAO = Depends(get_scores_dao),
):
    """Latest scores for every currently-valid miner.

    Source-of-truth for the row list is :class:`MinersDAO`'s current
    valid set, not the latest decided ``scores`` snapshot — so newly
    online miners (and miners who registered after the last decide)
    appear immediately. Per-env averages come from sample_results
    aggregation; the snapshot only contributes ``overall_score``
    (champion = 1.0, sort key) and the comparator-derived
    ``scores_by_env`` for the last champion + previous champion.

    Query parameters:
    - top: Number of top miners to return (default: 256, max: 256)
    """
    try:
        scores_data = await dao.get_latest_scores(limit=None) or {}
        block_number = scores_data.get("block_number")
        calculated_at = scores_data.get("calculated_at")
        snapshot_scores = scores_data.get("scores", []) or []

        validity = await _build_validity_map()
        scoring_envs = await _scoring_env_names()
        samples_dao = SampleResultsDAO()

        rows = await _build_score_rows_from_validity(
            validity, snapshot_scores, scoring_envs, samples_dao,
        )
        rows.sort(key=lambda x: x.get("overall_score", 0.0), reverse=True)
        rows = rows[:top]

        miner_scores = []
        for s in rows:
            hk = s.get("miner_hotkey")
            is_valid, invalid_reason, challenge_status, termination_reason = \
                validity.get(hk, (None, None, None, None))
            miner_scores.append(MinerScore(
                miner_hotkey=hk,
                uid=s.get("uid"),
                model_revision=s.get("model_revision"),
                model=s.get("model"),
                first_block=s.get("first_block"),
                overall_score=s.get("overall_score"),
                average_score=s.get("average_score"),
                scores_by_env=_filter_scores_by_env(
                    s.get("scores_by_env"), scoring_envs,
                ),
                total_samples=s.get("total_samples"),
                is_valid=is_valid,
                invalid_reason=invalid_reason,
                challenge_status=challenge_status,
                termination_reason=termination_reason,
            ))

        return ScoresResponse(
            block_number=block_number,
            calculated_at=calculated_at,
            scores=miner_scores,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve latest scores: {str(e)}"
        )


@router.get("/uid/{uid}", response_model=MinerScore, dependencies=[Depends(rate_limit_read)])
async def get_score_by_uid(
    uid: int,
    dao: ScoresDAO = Depends(get_scores_dao),
    miners_dao: MinersDAO = Depends(get_miners_dao),
):
    """Get score for one miner by UID.

    Returns scores even when the miner doesn't appear in the latest
    decided snapshot (e.g. they registered post-snapshot or have only
    been a non-winning challenger so far) — per-env averages come
    from ``sample_results`` aggregation in that case.
    """
    try:
        miner = await miners_dao.get_miner_by_uid(uid)
        if not miner:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Miner not found for UID={uid}"
            )

        hotkey = miner['hotkey']
        revision = str(miner.get('revision') or '')

        scores_data = await dao.get_latest_scores(limit=None) or {}
        snapshot_scores = scores_data.get("scores", []) or []
        snap = next(
            (s for s in snapshot_scores if s.get("miner_hotkey") == hotkey),
            {},
        ) or {}

        scoring_envs = await _scoring_env_names()
        samples_dao = SampleResultsDAO()

        # Validity (and the terminated flag for cache short-circuit)
        # before the aggregate call so terminated miners hit the
        # process-local cache instead of the DDB query path.
        is_valid, invalid_reason, challenge_status, termination_reason = (
            await _build_validity_map()
        ).get(hotkey, (None, None, None, None))

        env_list = sorted(scoring_envs) if scoring_envs else []
        agg_scores, agg_total = await _avg_scores_for_miner(
            samples_dao, hotkey, revision, env_list,
            cache_ttl_s=_ttl_for_status(challenge_status),
        )
        snap_env = snap.get("scores_by_env") or {}
        merged_env: Dict[str, Any] = {}
        for env in env_list:
            if env in snap_env and isinstance(snap_env[env], dict):
                merged_env[env] = snap_env[env]
            elif env in agg_scores:
                merged_env[env] = agg_scores[env]

        overall = snap.get("overall_score")
        if not isinstance(overall, (int, float)):
            overall = 0.0
        avg_score = snap.get("average_score")
        if not isinstance(avg_score, (int, float)):
            envs_with_score = [
                p["score"] for p in agg_scores.values()
                if isinstance(p, dict) and isinstance(p.get("score"), (int, float))
            ]
            avg_score = (
                sum(envs_with_score) / len(envs_with_score)
                if envs_with_score else 0.0
            )
        total_samples = snap.get("total_samples")
        if not isinstance(total_samples, int):
            total_samples = agg_total
        first_block = snap.get("first_block") or miner.get("first_block") or 0

        return MinerScore(
            miner_hotkey=hotkey,
            uid=snap.get("uid") if isinstance(snap.get("uid"), int) else uid,
            model_revision=str(snap.get("model_revision") or revision or ""),
            model=str(snap.get("model") or miner.get("model") or ""),
            first_block=int(first_block),
            overall_score=float(overall),
            average_score=float(avg_score),
            scores_by_env=_filter_scores_by_env(merged_env, scoring_envs),
            total_samples=int(total_samples),
            is_valid=is_valid,
            invalid_reason=invalid_reason,
            challenge_status=challenge_status,
            termination_reason=termination_reason,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve score: {str(e)}"
        )


# --- past-N-champions reward split ------------------------------------------
#
# Activation threshold lives in ``system_config`` under the key below so an
# operator can flip it on / change it / turn it off with one DAO write
# (``SystemConfigDAO.set_param``) — no env vars, no redeploy. Stores an
# integer block_number; missing or ``0`` keeps the legacy winner-takes-all
# behaviour. Anchored to ``block_number`` (snapshot PK) rather than to a
# uid or a hotkey because uids recycle and hotkeys' snapshots eventually
# TTL out — a block number is immutable and validator-native.
WEIGHTS_SPLIT_AFTER_BLOCK_PARAM = "weights_split_after_block"
WEIGHTS_SPLIT_CHAMPION_COUNT_PARAM = "weights_split_champion_count"
_DEFAULT_SPLIT_CHAMPION_COUNT = 5

# Cap snapshot fan-out. One row per championship transition; TTL is now
# 365 days, so 500 comfortably covers a year of swaps.
_SPLIT_SNAPSHOT_SCAN_LIMIT = 500


def _winner_uid_from_snapshot(snapshot: Dict[str, Any]) -> Optional[int]:
    """Extract the integer winner uid from a snapshot, tolerating three
    historical shapes: the current ``statistics.winner_uid`` field, the
    pre-Stage-U ``statistics.champion_uid`` field, and the implicit shape
    where the winner is the ``"1.0"`` entry in ``final_weights`` /
    ``miner_final_scores``."""
    stats = snapshot.get("statistics", {}) or {}
    for key in ("winner_uid", "champion_uid"):
        direct = stats.get(key)
        if direct is not None:
            try:
                return int(direct)
            except (TypeError, ValueError):
                continue
    raw_weights = (
        stats.get("final_weights")
        or stats.get("miner_final_scores")
        or {}
    )
    for uid_str, weight in raw_weights.items():
        try:
            if float(weight) >= 1.0:
                return int(uid_str)
        except (TypeError, ValueError):
            continue
    return None


def _winner_hotkey_from_snapshot(snapshot: Dict[str, Any]) -> Optional[str]:
    """Return the SS58 hotkey of the snapshot's winner, or ``None`` when
    the snapshot does not carry one. Accepts both the current
    ``winner_hotkey`` field and the legacy ``champion_hotkey`` field so
    pre-Stage-U snapshots aren't silently skipped (which would shrink the
    past-N-champions split to ``N=1`` until the new flow has written N
    transitions of its own)."""
    stats = snapshot.get("statistics", {}) or {}
    for key in ("winner_hotkey", "champion_hotkey"):
        hk = stats.get(key)
        if isinstance(hk, str) and hk:
            return hk
    return None


async def _resolve_split_payees(
    snapshots: List[Dict[str, Any]],
    miners_dao: MinersDAO,
    *,
    needed: int,
) -> List[Dict[str, Any]]:
    """Walk ``snapshots`` (newest-first), dedupe past champions by hotkey,
    look up each hotkey's *current* miner row in the miners table, and
    keep the first ``needed`` whose miner is still registered AND
    currently valid.

    Two filters are applied at resolution time:

    * **Deregistered hotkeys** are skipped. Paying them on their stale
      snapshot uid would in fact pay whoever now sits on that uid.
    * **Currently-invalid hotkeys** (``is_valid != "true"``: model
      mismatch, anticopy, multi-commit, …) are skipped. They were valid
      when they won the championship but have since been flagged; the
      legacy single-champion path can't even surface this case (the
      active champion is valid by construction), so the split path
      shouldn't either.

    Either filter outcome causes us to walk further back in history to
    backfill the count, so churn at the top of the leaderboard doesn't
    shrink the split unnecessarily.

    Returns rows ``{"uid", "hotkey", "model", "revision"}`` so callers
    that need display metadata (e.g. ``af get-rank``) don't have to
    re-query the miners table.
    """
    out: List[Dict[str, Any]] = []
    seen_hotkeys: set = set()
    for snap in snapshots:
        if len(out) >= needed:
            break
        hotkey = _winner_hotkey_from_snapshot(snap)
        if hotkey is None or hotkey in seen_hotkeys:
            continue
        seen_hotkeys.add(hotkey)
        miner = await miners_dao.get_miner_by_hotkey(hotkey)
        if not miner:
            continue  # hotkey no longer registered → don't pay its successor
        # ``is_valid`` is stored as the string ``"true"`` / ``"false"``
        # to be a GSI partition key, so compare against the string form.
        if str(miner.get("is_valid") or "").lower() != "true":
            continue  # registered but currently flagged invalid → skip
        try:
            uid = int(miner.get("uid"))
        except (TypeError, ValueError):
            continue
        out.append({
            "uid": uid,
            "hotkey": hotkey,
            "model": miner.get("model") or "",
            "revision": miner.get("revision") or "",
        })
    return out


async def compute_split_payees(
    snapshots_dao: ScoreSnapshotsDAO,
    miners_dao: MinersDAO,
    config_dao: SystemConfigDAO,
) -> Optional[List[Dict[str, Any]]]:
    """Compute the active reward-split list, or ``None`` if the feature
    is currently off.

    Shared by ``/scores/weights/latest`` (chain weights) and
    ``/rank/current`` (display) so both surfaces stay in lockstep.
    Off → ``None``. On with no eligible payees → ``[]`` (caller decides
    how to handle: weights endpoint 404s, rank surface just hides the
    section). On with payees → list of
    ``{"uid", "hotkey", "model", "revision", "share"}`` rows; ``share``
    is ``1/len(payees)`` so the caller can render percentages directly.
    """
    activation_block = await _get_int_param(
        config_dao, WEIGHTS_SPLIT_AFTER_BLOCK_PARAM, default=0,
    )
    if activation_block <= 0:
        return None

    snapshots = await snapshots_dao.get_recent_snapshots(
        limit=_SPLIT_SNAPSHOT_SCAN_LIMIT,
    )
    if not snapshots:
        return None
    latest_block = snapshots[0].get("block_number")
    if not (isinstance(latest_block, int) and latest_block >= activation_block):
        return None

    champion_count = max(
        1,
        await _get_int_param(
            config_dao,
            WEIGHTS_SPLIT_CHAMPION_COUNT_PARAM,
            default=_DEFAULT_SPLIT_CHAMPION_COUNT,
        ),
    )
    payees = await _resolve_split_payees(
        snapshots, miners_dao, needed=champion_count,
    )
    if not payees:
        return []
    share = 1.0 / len(payees)
    for p in payees:
        p["share"] = share
    return payees


async def _get_int_param(
    dao: SystemConfigDAO, name: str, default: int,
) -> int:
    """Read an integer system_config param, tolerating missing rows and
    string-typed values (DDB items round-trip numbers as strings via the
    document loader in some paths)."""
    raw = await dao.get_param_value(name, default=default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


@router.get("/weights/latest", dependencies=[Depends(rate_limit_read)])
async def get_latest_weights(
    snapshots_dao: ScoreSnapshotsDAO = Depends(get_score_snapshots_dao),
    miners_dao: MinersDAO = Depends(get_miners_dao),
    config_dao: SystemConfigDAO = Depends(get_system_config_dao),
):
    """Return per-UID weights derived from recent score snapshots.

    Pre-activation (``system_config`` row
    ``weights_split_after_block`` missing or ``0``, or the latest
    snapshot's ``block_number`` is below that threshold) this passes
    through the scheduler's winner-takes-all snapshot:
    ``statistics.final_weights`` as ``{uid_str: weight_str}`` — exactly
    one uid at ``"1.0"`` and everyone else at ``"0.0"`` (legacy snapshots
    used the equivalent ``statistics.miner_final_scores`` field; both are
    accepted).

    Post-activation we split weight evenly across the most recent
    ``weights_split_champion_count`` (default 5) distinct champion
    hotkeys, resolved to their *current* on-chain uid via the miners
    table. Each carries ``1/N`` weight. A hotkey that has since
    deregistered is skipped (we don't accidentally pay the successor on
    its recycled uid) and we keep walking history to backfill the count.
    """
    payees = await compute_split_payees(snapshots_dao, miners_dao, config_dao)

    if payees is not None:
        # Feature on. Need at least one valid payee to set anything on
        # chain; refuse rather than send a malformed weight vector.
        if not payees:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No registered past-champion hotkeys in recent snapshots",
            )
        # Latest block lives on the freshest snapshot in the same scan
        # the resolver walked — fetch one more time for the response
        # envelope. (Cheap: GSI Limit=1.)
        latest = await snapshots_dao.get_latest_snapshot()
        return {
            "block_number": (latest or {}).get("block_number"),
            "weights": {
                str(p["uid"]): {"weight": p["share"]} for p in payees
            },
        }

    # Feature off → legacy winner-takes-all pass-through.
    latest = await snapshots_dao.get_latest_snapshot()
    if not latest:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No score snapshots found",
        )
    statistics = latest.get("statistics", {}) or {}
    raw_weights = (
        statistics.get("final_weights")
        or statistics.get("miner_final_scores")
        or {}
    )
    if not raw_weights:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No weights found in latest snapshot",
        )

    weights_response: dict = {}
    for uid_str, weight in raw_weights.items():
        try:
            w = float(weight)
        except (TypeError, ValueError):
            continue
        weights_response[str(uid_str)] = {"weight": w}

    return {
        "block_number": latest.get("block_number"),
        "weights": weights_response,
    }
