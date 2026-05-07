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
from affine.database.dao.miners import MinersDAO
from affine.database.dao.targon_deployments import TargonDeploymentsDAO

router = APIRouter(prefix="/scores", tags=["Scores"])


async def _build_targon_status_map() -> dict:
    """Map (hotkey, revision) -> 'active' | 'deploying' for live Targon deployments.

    Two GSI queries (status='active', status='deploying'); cheap because
    each return at most a handful of rows. Returns an empty map on any
    DAO error so a Targon-table outage doesn't 500 the rank endpoint —
    the CLI just shows no acceleration markers.
    """
    try:
        dao = TargonDeploymentsDAO()
        active = await dao.list_by_status("active")
        deploying = await dao.list_by_status("deploying")
    except Exception:
        return {}
    out: dict = {}
    # Active wins over deploying when both exist for the same key (a
    # restart can briefly leave two rows).
    for r in deploying:
        hk, rev = r.get("hotkey"), r.get("revision")
        if hk and rev:
            out[(hk, rev)] = "deploying"
    for r in active:
        hk, rev = r.get("hotkey"), r.get("revision")
        if hk and rev:
            out[(hk, rev)] = "active"
    return out


async def _build_validity_map() -> dict:
    """Map hotkey -> (is_valid_bool, invalid_reason_str|None) from miners table.

    Surfaces validator-side invalidation in the rank UI: scoring
    snapshots only carry challenge_status, but a miner can be
    challenge_status='sampling' while is_valid=False (e.g. anticopy
    cheat, model_mismatch). Without this lookup the CLI shows them
    as still competing when in reality sampling has stopped.

    One full scan (256-row cap → cheap). Empty map on any DAO error
    so a miners-table blip just hides reasons rather than 500ing.
    """
    try:
        dao = MinersDAO()
        miners = await dao.get_all_miners()
    except Exception:
        return {}
    out: dict = {}
    for m in miners:
        hk = m.get("hotkey")
        if not hk:
            continue
        # is_valid stored as 'true'/'false' string in the GSI partition key.
        is_valid = str(m.get("is_valid") or "").lower() == "true"
        out[hk] = (is_valid, m.get("invalid_reason") or None)
    return out


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
        
        # Sort by overall_score descending and take top N
        scores_list.sort(key=lambda x: x.get("overall_score", 0.0), reverse=True)
        scores_list = scores_list[:top]

        targon_status = await _build_targon_status_map()
        validity = await _build_validity_map()

        # Convert to response models with safe field access
        miner_scores = []
        for s in scores_list:
            hk = s.get("miner_hotkey")
            is_valid, invalid_reason = validity.get(hk, (None, None))
            miner_scores.append(MinerScore(
                miner_hotkey=hk,
                uid=s.get("uid"),
                model_revision=s.get("model_revision"),
                model=s.get("model"),
                first_block=s.get("first_block"),
                overall_score=s.get("overall_score"),
                average_score=s.get("average_score"),
                scores_by_env=s.get("scores_by_env"),
                total_samples=s.get("total_samples"),
                challenge_info=s.get("challenge_info"),
                targon_status=targon_status.get(
                    (hk, s.get("model_revision"))
                ),
                is_valid=is_valid,
                invalid_reason=invalid_reason,
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
        
        targon_status = await _build_targon_status_map()
        return MinerScore(
            miner_hotkey=miner_score.get("miner_hotkey"),
            uid=miner_score.get("uid"),
            model_revision=miner_score.get("model_revision"),
            model=miner_score.get("model"),
            first_block=miner_score.get("first_block"),
            overall_score=miner_score.get("overall_score"),
            average_score=miner_score.get("average_score"),
            scores_by_env=miner_score.get("scores_by_env"),
            total_samples=miner_score.get("total_samples"),
            challenge_info=miner_score.get("challenge_info"),
            targon_status=targon_status.get(
                (miner_score.get("miner_hotkey"), miner_score.get("model_revision"))
            ),
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
    """
    Get the latest normalized weights from scoring calculation.
    
    Returns the most recent score snapshot with normalized weights
    for all miners, suitable for setting on-chain weights.
    
    Response format:
    {
        "block_number": 12345,
        "config": {
            "error_rate_reduction": 0.2,
            "min_improvement": 0.02,
            "min_completeness": 0.99,
            ...
        },
        "weights": {
            "0": {"hotkey": "5...", "weight": 0.15},
            "1": {"hotkey": "5...", "weight": 0.12},
            ...
        }
    }
    """
    try:
        # Get latest snapshot
        snapshot = await snapshots_dao.get_latest_snapshot()
        
        if not snapshot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No score snapshots found"
            )
        
        # Extract weights from statistics
        statistics = snapshot.get('statistics', {})
        miner_weights = statistics.get('miner_final_scores', {})
        
        if not miner_weights:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No weights found in latest snapshot"
            )
        
        # Format response according to design document
        weights_response = {}
        for uid_str, weight in miner_weights.items():
            # Get hotkey from snapshot metadata if available
            # For now, use uid as key
            weights_response[uid_str] = {
                "weight": weight
            }
        
        return {
            "block_number": snapshot.get('block_number'),
            "config": snapshot.get('config', {}),
            "weights": weights_response
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve latest weights: {str(e)}"
        )