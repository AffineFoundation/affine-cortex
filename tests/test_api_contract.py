"""API server public/internal contract tests."""

from __future__ import annotations

import pytest

from affine.api.routers.logs import get_miner_logs


class _FakeExecutionLogsDAO:
    def __init__(self, logs):
        self.logs = logs
        self.calls = []

    async def get_recent_logs(self, miner_hotkey, limit=1000, status=None):
        self.calls.append((miner_hotkey, limit, status))
        return self.logs


def test_public_server_does_not_mount_internal_logs_by_default():
    from affine.api.server import app

    paths = {route.path for route in app.routes}
    assert "/" in paths
    assert "/api/v1/health" in paths
    assert "/api/v1/windows/current" in paths
    assert "/api/v1/scores/latest" in paths
    assert "/api/v1/config" in paths
    assert "/api/v1/logs/miner/{hotkey}" not in paths


@pytest.mark.asyncio
async def test_logs_router_maps_dao_rows_to_response_model():
    dao = _FakeExecutionLogsDAO([
        {
            "log_id": "log-1",
            "timestamp": 123,
            "dataset_task_id": 42,
            "status": "completed",
            "env": "SWE",
            "latency_ms": 87,
        }
    ])

    response = await get_miner_logs("hk", limit=5, success=True, dao=dao)

    assert dao.calls == [("hk", 5, "completed")]
    assert len(response.logs) == 1
    entry = response.logs[0]
    assert entry.log_id == "log-1"
    assert entry.task_id == "42"
    assert entry.status == "completed"
    assert entry.latency_ms == 87


@pytest.mark.asyncio
async def test_logs_router_filters_failed_rows_for_success_false():
    dao = _FakeExecutionLogsDAO([])

    await get_miner_logs("hk", limit=10, success=False, dao=dao)

    assert dao.calls == [("hk", 10, "failed")]
