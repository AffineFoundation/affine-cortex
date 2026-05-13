"""API server public/internal contract tests."""

from __future__ import annotations

import pytest

import affine.api.routers.config as config_router
from affine.api.routers.miners import get_miner_by_hotkey, get_miner_by_uid
from affine.api.routers.logs import get_miner_logs


class _FakeMinersDAO:
    def __init__(self, rows):
        self.rows = rows

    async def get_miner_by_uid(self, uid):
        return self.rows.get(("uid", uid))

    async def get_miner_by_hotkey(self, hotkey):
        return self.rows.get(("hotkey", hotkey))


class _FakeExecutionLogsDAO:
    def __init__(self, logs):
        self.logs = logs
        self.calls = []

    async def get_recent_logs(self, miner_hotkey, limit=1000, status=None):
        self.calls.append((miner_hotkey, limit, status))
        return self.logs


class _FakeSystemConfigDAO:
    def __init__(self, rows):
        self.rows = rows

    async def get_all_params(self):
        return self.rows

    async def get_param(self, key):
        return self.rows.get(key)


def test_public_server_does_not_mount_internal_logs_by_default():
    from affine.api.server import app

    paths = {route.path for route in app.routes}
    assert "/" in paths
    assert "/api/v1/health" in paths
    assert "/api/v1/windows/current" in paths
    assert "/api/v1/miners/uid/{uid}" in paths
    assert "/api/v1/miners/hotkey/{hotkey}" in paths
    assert "/api/v1/scores/latest" in paths
    assert "/api/v1/config" in paths
    assert "/api/v1/logs/miner/{hotkey}" not in paths


@pytest.mark.asyncio
async def test_miners_router_returns_basic_public_metadata_by_uid():
    dao = _FakeMinersDAO({
        ("uid", 7): {
            "uid": 7,
            "hotkey": "hk",
            "model": "org/model",
            "revision": "abc",
            "is_valid": "true",
            "challenge_status": "pending",
            "first_block": 100,
            "block_number": 120,
            "invalid_reason": None,
            "model_hash": "hash",
        }
    })

    response = await get_miner_by_uid(7, dao=dao)

    assert response.uid == 7
    assert response.hotkey == "hk"
    assert response.model == "org/model"
    assert response.revision == "abc"
    assert response.is_valid is True
    assert response.challenge_status == "pending"
    assert response.model_hash == "hash"


@pytest.mark.asyncio
async def test_miners_router_returns_basic_public_metadata_by_hotkey():
    dao = _FakeMinersDAO({
        ("hotkey", "hk"): {
            "uid": 8,
            "hotkey": "hk",
            "model": "org/other",
            "revision": "def",
            "is_valid": "false",
            "invalid_reason": "model_mismatch",
        }
    })

    response = await get_miner_by_hotkey("hk", dao=dao)

    assert response.uid == 8
    assert response.is_valid is False
    assert response.invalid_reason == "model_mismatch"


@pytest.mark.asyncio
async def test_config_router_only_returns_public_config_keys(monkeypatch):
    monkeypatch.setattr(
        config_router,
        "config_dao",
        _FakeSystemConfigDAO({
            "validator_burn_percentage": 0.25,
            "current_task_ids": {"task_ids": {"SWE": [1, 2, 3]}},
            "champion": {"uid": 1},
        }),
    )

    response = await config_router.get_all_configs()

    assert response == {"configs": {"validator_burn_percentage": 0.25}}


@pytest.mark.asyncio
async def test_config_router_blocks_internal_config_keys(monkeypatch):
    monkeypatch.setattr(
        config_router,
        "config_dao",
        _FakeSystemConfigDAO({
            "current_task_ids": {"task_ids": {"SWE": [1, 2, 3]}},
        }),
    )

    with pytest.raises(config_router.HTTPException) as exc:
        await config_router.get_config("current_task_ids")

    assert exc.value.status_code == 404


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
