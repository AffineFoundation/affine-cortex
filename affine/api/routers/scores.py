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

router = APIRouter(prefix="/scores", tags=["Scores"])


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
        
        # Convert to response models with safe field access
        miner_scores = [
            MinerScore(
                miner_hotkey=s.get("miner_hotkey"),
                uid=s.get("uid"),
                model_revision=s.get("model_revision"),
                model=s.get("model"),
                first_block=s.get("first_block"),
                overall_score=s.get("overall_score"),
                average_score=s.get("average_score"),
                scores_by_layer=s.get("scores_by_layer"),
                scores_by_env=s.get("scores_by_env"),
                total_samples=s.get("total_samples"),
                cumulative_weight=s.get("cumulative_weight"),
                elo_rating=s.get("elo_rating"),
                elo_rounds_played=s.get("elo_rounds_played"),
                elo_rating_change=s.get("elo_rating_change"),
                filter_info=s.get("filter_info"),
            )
            for s in scores_list
        ]
        
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
        
        return MinerScore(
            miner_hotkey=miner_score.get("miner_hotkey"),
            uid=miner_score.get("uid"),
            model_revision=miner_score.get("model_revision"),
            model=miner_score.get("model"),
            first_block=miner_score.get("first_block"),
            overall_score=miner_score.get("overall_score"),
            average_score=miner_score.get("average_score"),
            scores_by_layer=miner_score.get("scores_by_layer"),
            scores_by_env=miner_score.get("scores_by_env"),
            total_samples=miner_score.get("total_samples"),
            cumulative_weight=miner_score.get("cumulative_weight"),
            elo_rating=miner_score.get("elo_rating"),
            elo_rounds_played=miner_score.get("elo_rounds_played"),
            elo_rating_change=miner_score.get("elo_rating_change"),
            filter_info=miner_score.get("filter_info"),
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