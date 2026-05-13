"""
FastAPI dependencies.

After the queue-window refactor the API is read-only — the scheduler service
does all sampling/scoring in-process via DAOs, so there's no executor
auth, no task pool manager, no signature verification on this surface.

The dependencies that remain:
  - DAO singletons used by /miners, /scores, /logs, /config, /windows.
  - A simple per-IP rate limiter for read endpoints.
"""

from __future__ import annotations

import time
from typing import Optional

from fastapi import HTTPException, Request, status

from affine.api.config import config
from affine.database.dao.execution_logs import ExecutionLogsDAO
from affine.database.dao.miners import MinersDAO
from affine.database.dao.score_snapshots import ScoreSnapshotsDAO
from affine.database.dao.scores import ScoresDAO
from affine.database.dao.system_config import SystemConfigDAO


# DAO singletons (cheap to construct, but no reason to re-instantiate)
_execution_logs_dao: Optional[ExecutionLogsDAO] = None
_scores_dao: Optional[ScoresDAO] = None
_system_config_dao: Optional[SystemConfigDAO] = None
_score_snapshots_dao: Optional[ScoreSnapshotsDAO] = None
_miners_dao: Optional[MinersDAO] = None


def get_execution_logs_dao() -> ExecutionLogsDAO:
    global _execution_logs_dao
    if _execution_logs_dao is None:
        _execution_logs_dao = ExecutionLogsDAO()
    return _execution_logs_dao


def get_scores_dao() -> ScoresDAO:
    global _scores_dao
    if _scores_dao is None:
        _scores_dao = ScoresDAO()
    return _scores_dao


def get_system_config_dao() -> SystemConfigDAO:
    global _system_config_dao
    if _system_config_dao is None:
        _system_config_dao = SystemConfigDAO()
    return _system_config_dao


def get_score_snapshots_dao() -> ScoreSnapshotsDAO:
    global _score_snapshots_dao
    if _score_snapshots_dao is None:
        _score_snapshots_dao = ScoreSnapshotsDAO()
    return _score_snapshots_dao


def get_miners_dao() -> MinersDAO:
    global _miners_dao
    if _miners_dao is None:
        _miners_dao = MinersDAO()
    return _miners_dao


# Per-IP rate limit (in-memory; one process per node so this is fine).
_rate_limit_store: dict = {}


def _check_rate_limit(identifier: str, limit: int, window_seconds: int = 60) -> bool:
    now = int(time.time())
    window_start = now - window_seconds
    hits = _rate_limit_store.setdefault(identifier, [])
    hits[:] = [ts for ts in hits if ts > window_start]
    if len(hits) >= limit:
        return False
    hits.append(now)
    return True


async def rate_limit_read(request: Request) -> None:
    if not config.RATE_LIMIT_ENABLED:
        return
    if not _check_rate_limit(request.client.host, config.RATE_LIMIT_READ):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
        )
