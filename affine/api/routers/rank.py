"""Public aggregated rank endpoint."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, Query

from affine.api.dependencies import rate_limit_read
from affine.api.rank_state import get_current_state, get_queue
from affine.api.routers.scores import get_latest_scores
from affine.database.dao.scores import ScoresDAO


router = APIRouter(
    prefix="/rank",
    tags=["rank"],
    dependencies=[Depends(rate_limit_read)],
)


@router.get("/current")
async def get_current_rank(
    top: int = Query(256, ge=1, le=256),
    queue_limit: int = Query(10, ge=1, le=100),
) -> Dict[str, Any]:
    """One public rank payload for the CLI/table view.

    The split state readers remain implementation details; public callers get
    one response containing only the status, queue head, and score table data
    required by ``af get-rank``.
    """
    return {
        "window": await get_current_state(),
        "queue": await get_queue(limit=queue_limit),
        "scores": await get_latest_scores(top=top, dao=ScoresDAO()),
    }
