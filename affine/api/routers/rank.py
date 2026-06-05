"""Public aggregated rank endpoint."""

from __future__ import annotations

import asyncio
import copy
import os
import time
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Query, status

from affine.api.dependencies import rate_limit_read
from affine.api.rank_state import get_current_state, get_queue
from affine.api.routers.scores import get_latest_scores
from affine.core.setup import logger
from affine.database.dao.scores import ScoresDAO


router = APIRouter(
    prefix="/rank",
    tags=["rank"],
    dependencies=[Depends(rate_limit_read)],
)

_RANK_CANONICAL_TOP = 256
_RANK_CANONICAL_QUEUE_LIMIT = 256
_RANK_CACHE_KEY = (_RANK_CANONICAL_TOP, _RANK_CANONICAL_QUEUE_LIMIT)
_RANK_CACHE_REFRESH_S = float(os.getenv("API_RANK_CACHE_REFRESH_SECONDS", "300"))
_RANK_CACHE_TTL_S = float(os.getenv("API_RANK_CACHE_TTL_SECONDS", "1800"))
_RANK_CACHE: Dict[tuple[int, int], tuple[float, Dict[str, Any]]] = {}
_RANK_CACHE_LOCK = asyncio.Lock()


def _reset_rank_cache_for_test() -> None:
    _RANK_CACHE.clear()


async def _build_current_rank_payload(top: int, queue_limit: int) -> Dict[str, Any]:
    started = time.perf_counter()
    window = await get_current_state()
    queue = await get_queue(limit=queue_limit)
    scores = await get_latest_scores(top=top, dao=ScoresDAO())
    elapsed_ms = (time.perf_counter() - started) * 1000
    return {
        "window": window,
        "queue": queue,
        "scores": scores,
        "meta": {
            "database_query_time_ms": round(elapsed_ms, 2),
        },
    }


def _rank_response(
    payload: Dict[str, Any],
    *,
    cache_status: str,
    cache_age_s: float | None = None,
) -> Dict[str, Any]:
    response = copy.deepcopy(payload)
    meta = dict(response.get("meta") or {})
    meta["cache_status"] = cache_status
    if cache_age_s is not None:
        meta["cache_age_seconds"] = round(cache_age_s, 3)
    response["meta"] = meta
    return response


def _slice_rank_payload(
    payload: Dict[str, Any],
    *,
    top: int,
    queue_limit: int,
) -> Dict[str, Any]:
    response = copy.deepcopy(payload)
    queue = response.get("queue")
    if isinstance(queue, list):
        response["queue"] = queue[:queue_limit]

    scores = response.get("scores")
    if isinstance(scores, dict):
        rows = scores.get("scores")
        if isinstance(rows, list):
            scores["scores"] = rows[:top]
        if "top_seen" in scores:
            scores["top_seen"] = top
    elif hasattr(scores, "scores") and hasattr(scores, "model_copy"):
        response["scores"] = scores.model_copy(update={
            "scores": list(scores.scores[:top]),
        })
    return response


async def refresh_rank_cache_once() -> Dict[str, Any]:
    payload = await _build_current_rank_payload(
        _RANK_CANONICAL_TOP,
        _RANK_CANONICAL_QUEUE_LIMIT,
    )
    async with _RANK_CACHE_LOCK:
        _RANK_CACHE[_RANK_CACHE_KEY] = (time.monotonic(), copy.deepcopy(payload))
    return payload


async def run_rank_cache_refresher() -> None:
    if _RANK_CACHE_REFRESH_S <= 0:
        return
    while True:
        await asyncio.sleep(_RANK_CACHE_REFRESH_S)
        try:
            await refresh_rank_cache_once()
        except Exception as exc:
            logger.error(f"Failed to refresh rank cache: {exc}", exc_info=True)


@router.get("/current")
async def get_current_rank(
    top: int = Query(256, ge=1, le=256),
    queue_limit: int = Query(256, ge=1, le=256),
) -> Dict[str, Any]:
    """One public rank payload for the CLI/table view.

    The split state readers remain implementation details; public callers get
    one response containing only the status, queue head, and score table data
    required by ``af get-rank``.
    """
    now = time.monotonic()
    cached = _RANK_CACHE.get(_RANK_CACHE_KEY)
    if cached is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Rank cache is warming up",
        )

    cached_at, payload = cached
    age = now - cached_at
    if _RANK_CACHE_TTL_S > 0 and age >= _RANK_CACHE_TTL_S:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Rank cache is expired",
        )

    cache_status = "hit"
    if _RANK_CACHE_REFRESH_S > 0 and age >= _RANK_CACHE_REFRESH_S:
        cache_status = "stale"
    return _rank_response(
        _slice_rank_payload(payload, top=top, queue_limit=queue_limit),
        cache_status=cache_status,
        cache_age_s=age,
    )
