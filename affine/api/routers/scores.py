"""
Scores Router

Endpoints for querying score calculations.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from affine.api.models import (
    ScoresResponse,
    MinerScore,
)
from affine.api.dependencies import (
    get_scores_dao,
    get_score_snapshots_dao,
    get_miners_dao,
    rate_limit_read,
)
from affine.database.dao.scores import ScoresDAO
from affine.database.dao.score_snapshots import ScoreSnapshotsDAO
from affine.database.dao.miner_stats import MinerStatsDAO
from affine.database.dao.miners import MinersDAO
from affine.database.dao.system_config import SystemConfigDAO
from affine.src.scorer.window_state import StateStore

router = APIRouter(prefix="/scores", tags=["Scores"])


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
    """
    Get the most recent score snapshot.
    
    Returns top N miners by score at the latest calculated block.
    
    Query parameters:
    - top: Number of top miners to return (default: 256, max: 256)
    """
    try:
        scores_data = await dao.get_latest_scores(limit=None)
        
        if not scores_data or not scores_data.get('block_number'):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No scores found"
            )
        
        # Parse scores data
        block_number = scores_data.get("block_number")
        calculated_at = scores_data.get("calculated_at")
        scores_list = scores_data.get("scores", [])
        
        validity = await _build_validity_map()
        scoring_envs = await _scoring_env_names()
        scores_list = [
            s for s in scores_list
            if s.get("miner_hotkey") in validity
        ]

        # Sort current-online rows by overall_score descending and take top N.
        scores_list.sort(key=lambda x: x.get("overall_score", 0.0), reverse=True)
        scores_list = scores_list[:top]

        # Convert to response models with safe field access
        miner_scores = []
        for s in scores_list:
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
    """
    Get score for a specific miner by UID.
    
    Path parameters:
    - uid: Miner UID (0-255)
    
    Returns the score details for the specified miner from the latest snapshot.
    """
    try:
        miner = await miners_dao.get_miner_by_uid(uid)
        
        if not miner:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Miner not found for UID={uid}"
            )
        
        hotkey = miner['hotkey']
        
        scores_data = await dao.get_latest_scores(limit=None)
        
        if not scores_data or not scores_data.get('block_number'):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No scores found"
            )
        
        scores_list = scores_data.get("scores", [])
        
        miner_score = next(
            (s for s in scores_list if s.get("miner_hotkey") == hotkey),
            None
        )
        
        if not miner_score:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Score not found for UID={uid}"
            )
        
        hk = miner_score.get("miner_hotkey")
        is_valid, invalid_reason, challenge_status, termination_reason = (
            await _build_validity_map()
        ).get(hk, (None, None, None, None))
        scoring_envs = await _scoring_env_names()
        return MinerScore(
            miner_hotkey=hk,
            uid=miner_score.get("uid"),
            model_revision=miner_score.get("model_revision"),
            model=miner_score.get("model"),
            first_block=miner_score.get("first_block"),
            overall_score=miner_score.get("overall_score"),
            average_score=miner_score.get("average_score"),
            scores_by_env=_filter_scores_by_env(
                miner_score.get("scores_by_env"), scoring_envs,
            ),
            total_samples=miner_score.get("total_samples"),
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


@router.get("/weights/latest", dependencies=[Depends(rate_limit_read)])
async def get_latest_weights(
    snapshots_dao: ScoreSnapshotsDAO = Depends(get_score_snapshots_dao),
):
    """Return the latest snapshot's per-UID weights.

    The queue-window scorer writes ``statistics.final_weights`` as
    ``{uid_str: weight_str}`` — exactly one uid carries ``"1.0"`` (the
    champion) and everyone else carries ``"0.0"``. We reshape into
    ``{uid_str: {"weight": float}}`` so :class:`WeightSetter` can consume
    it directly.
    """
    snapshot = await snapshots_dao.get_latest_snapshot()
    if not snapshot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No score snapshots found",
        )

    statistics = snapshot.get("statistics", {}) or {}
    raw_weights = statistics.get("final_weights") or {}
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
        "block_number": snapshot.get("block_number"),
        "weights": weights_response,
    }
