"""API server public/internal contract tests."""

from __future__ import annotations

import pytest

import affine.api.routers.config as config_router
import affine.api.routers.rank as rank_router
import affine.api.routers.scores as scores_router
from affine.api.routers.miners import get_miner_by_hotkey, get_miner_by_uid
from affine.api.routers.logs import get_miner_logs
from affine.api.routers.scores import get_latest_scores, get_latest_weights


class _FakeMinersDAO:
    def __init__(self, rows):
        self.rows = rows

    async def get_miner_by_uid(self, uid):
        return self.rows.get(("uid", uid))

    async def get_miner_by_hotkey(self, hotkey):
        return self.rows.get(("hotkey", hotkey))

    async def get_all_miners(self):
        return list(self.rows.values())


class _FakeMinerStatsDAO:
    def __init__(self, states):
        self.states = states

    async def get_challenge_state(self, hotkey, revision):
        return self.states.get((hotkey, revision), {
            "challenge_status": "sampling",
            "termination_reason": "",
        })

    async def build_challenge_state_map(self, miners):
        return {
            (m.get("hotkey"), m.get("revision")): await self.get_challenge_state(
                m.get("hotkey"), m.get("revision"),
            )
            for m in miners
            if m.get("hotkey") and m.get("revision")
        }


class _FakeExecutionLogsDAO:
    def __init__(self, logs):
        self.logs = logs
        self.calls = []

    async def get_recent_logs(self, miner_hotkey, limit=1000, status=None):
        self.calls.append((miner_hotkey, limit, status))
        return self.logs


class _FakeScoreSnapshotsDAO:
    def __init__(self, snapshot):
        self.snapshot = snapshot

    async def get_latest_snapshot(self):
        return self.snapshot


class _FakeScoresDAO:
    def __init__(self, payload):
        self.payload = payload

    async def get_latest_scores(self, limit=None):
        return self.payload


class _FakeSystemConfigDAO:
    def __init__(self, rows):
        self.rows = rows

    async def get_all_params(self):
        return self.rows

    async def get_param(self, key):
        return self.rows.get(key)

    async def get_param_value(self, key, default=None):
        return self.rows.get(key, default)


def test_public_server_does_not_mount_internal_logs_by_default():
    from affine.api.server import app

    paths = {route.path for route in app.routes}
    assert "/" in paths
    assert "/api/v1/health" in paths
    assert "/api/v1/rank/current" in paths
    assert "/api/v1/windows/current" not in paths
    assert "/api/v1/windows/queue" not in paths
    assert "/api/v1/miners/uid/{uid}" in paths
    assert "/api/v1/miners/hotkey/{hotkey}" in paths
    assert "/api/v1/scores/latest" in paths
    assert "/api/v1/config" in paths
    assert "/api/v1/logs/miner/{hotkey}" not in paths


@pytest.mark.asyncio
async def test_miners_router_returns_basic_public_metadata_by_uid(monkeypatch):
    monkeypatch.setattr(
        "affine.api.routers.miners.MinerStatsDAO",
        lambda: _FakeMinerStatsDAO({("hk", "abc"): {
            "challenge_status": "sampling",
            "termination_reason": "",
        }}),
    )
    dao = _FakeMinersDAO({
        ("uid", 7): {
            "uid": 7,
            "hotkey": "hk",
            "model": "org/model",
            "revision": "abc",
            "is_valid": "true",
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
    assert response.challenge_status == "sampling"
    assert response.model_hash == "hash"


@pytest.mark.asyncio
async def test_miners_router_returns_termination_reason_when_set(monkeypatch):
    """When a miner has been terminated,
    ``af get-miner`` should be able to read both the status and the
    human-readable reason via the same endpoint."""
    monkeypatch.setattr(
        "affine.api.routers.miners.MinerStatsDAO",
        lambda: _FakeMinerStatsDAO({("hk-lost", "r"): {
            "challenge_status": "terminated",
            "termination_reason": "lost_to_champion:5GepM|DISTILL:0.75vs0.76",
        }}),
    )
    dao = _FakeMinersDAO({
        ("uid", 9): {
            "uid": 9,
            "hotkey": "hk-lost",
            "model": "org/m",
            "revision": "r",
            "is_valid": "true",
        }
    })

    response = await get_miner_by_uid(9, dao=dao)

    assert response.challenge_status == "terminated"
    assert response.termination_reason.startswith("lost_to_champion:")


@pytest.mark.asyncio
async def test_miners_router_returns_basic_public_metadata_by_hotkey(monkeypatch):
    monkeypatch.setattr(
        "affine.api.routers.miners.MinerStatsDAO",
        lambda: _FakeMinerStatsDAO({("hk", "def"): {
            "challenge_status": "sampling",
            "termination_reason": "",
        }}),
    )
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
async def test_rank_router_aggregates_rank_payload(monkeypatch):
    async def fake_current_state():
        return {"champion": {"uid": 7}, "sample_counts": {"7": {"SWE": 300}}}

    async def fake_queue(limit):
        return [{"position": 1, "uid": 8, "limit_seen": limit}]

    async def fake_scores(top, dao):
        return {"scores": [{"uid": 7}], "top_seen": top}

    monkeypatch.setattr(rank_router, "get_current_state", fake_current_state)
    monkeypatch.setattr(rank_router, "get_queue", fake_queue)
    monkeypatch.setattr(rank_router, "get_latest_scores", fake_scores)

    response = await rank_router.get_current_rank(top=32, queue_limit=5)

    assert response == {
        "window": {"champion": {"uid": 7}, "sample_counts": {"7": {"SWE": 300}}},
        "queue": [{"position": 1, "uid": 8, "limit_seen": 5}],
        "scores": {"scores": [{"uid": 7}], "top_seen": 32},
    }


@pytest.mark.asyncio
async def test_scores_latest_filters_to_current_miners(monkeypatch):
    monkeypatch.setattr(
        scores_router,
        "MinersDAO",
        lambda: _FakeMinersDAO({
            ("uid", 7): {
                "uid": 7,
                "hotkey": "current-hk",
                "revision": "current-rev",
                "is_valid": "true",
            },
        }),
    )
    monkeypatch.setattr(
        scores_router,
        "MinerStatsDAO",
        lambda: _FakeMinerStatsDAO({
            ("current-hk", "current-rev"): {
                "challenge_status": "terminated",
                "termination_reason": "lost_to_champion:abc",
            },
        }),
    )
    monkeypatch.setattr(
        scores_router,
        "SystemConfigDAO",
        lambda: _FakeSystemConfigDAO({}),
    )

    response = await get_latest_scores(
        top=256,
        dao=_FakeScoresDAO({
            "block_number": 123,
            "calculated_at": 456,
            "scores": [
                {
                    "uid": 99,
                    "miner_hotkey": "offline-hk",
                    "model_revision": "old",
                    "model": "org/offline",
                    "first_block": 1,
                    "overall_score": 1.0,
                    "average_score": 1.0,
                    "scores_by_env": {},
                    "total_samples": 10,
                },
                {
                    "uid": 7,
                    "miner_hotkey": "current-hk",
                    "model_revision": "current-rev",
                    "model": "org/current",
                    "first_block": 2,
                    "overall_score": 0.0,
                    "average_score": 0.0,
                    "scores_by_env": {},
                    "total_samples": 0,
                },
            ],
        }),
    )

    assert [row.uid for row in response.scores] == [7]
    assert response.scores[0].challenge_status == "terminated"
    assert response.scores[0].termination_reason == "lost_to_champion:abc"


@pytest.mark.asyncio
async def test_scores_latest_filters_disabled_scoring_envs(monkeypatch):
    monkeypatch.setattr(
        scores_router,
        "MinersDAO",
        lambda: _FakeMinersDAO({
            ("uid", 7): {
                "uid": 7,
                "hotkey": "current-hk",
                "revision": "current-rev",
                "is_valid": "true",
            },
        }),
    )
    monkeypatch.setattr(
        scores_router,
        "MinerStatsDAO",
        lambda: _FakeMinerStatsDAO({}),
    )
    monkeypatch.setattr(
        scores_router,
        "SystemConfigDAO",
        lambda: _FakeSystemConfigDAO({
            "environments": {
                "SWE": {
                    "enabled_for_sampling": True,
                    "enabled_for_scoring": True,
                    "sampling": {"sampling_count": 10, "dataset_range": [[0, 100]]},
                },
                "DISTILL": {
                    "enabled_for_sampling": True,
                    "enabled_for_scoring": False,
                    "sampling": {"sampling_count": 10, "dataset_range": [[0, 100]]},
                },
            },
        }),
    )

    response = await get_latest_scores(
        top=256,
        dao=_FakeScoresDAO({
            "block_number": 123,
            "calculated_at": 456,
            "scores": [
                {
                    "uid": 7,
                    "miner_hotkey": "current-hk",
                    "model_revision": "current-rev",
                    "model": "org/current",
                    "first_block": 2,
                    "overall_score": 1.0,
                    "average_score": 1.0,
                    "scores_by_env": {
                        "SWE": {"score": 0.8},
                        "DISTILL": {"score": 0.9},
                    },
                    "total_samples": 20,
                },
            ],
        }),
    )

    assert response.scores[0].scores_by_env == {"SWE": {"score": 0.8}}


@pytest.mark.asyncio
async def test_weights_endpoint_does_not_expose_snapshot_config():
    response = await get_latest_weights(
        snapshots_dao=_FakeScoreSnapshotsDAO({
            "block_number": 123,
            "config": {
                "window_id": 17,
                "environments": ["SWE", "DISTILL"],
                "current_task_ids": {"SWE": [1, 2, 3]},
                "inference_endpoints": [{"name": "ssh-a"}],
            },
            "statistics": {
                "final_weights": {
                    "7": "1.0",
                    "8": "0.0",
                }
            },
        })
    )

    assert response == {
        "block_number": 123,
        "weights": {
            "7": {"weight": 1.0},
            "8": {"weight": 0.0},
        },
    }


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
