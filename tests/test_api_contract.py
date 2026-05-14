"""API server public/internal contract tests."""

from __future__ import annotations

import pytest

import affine.api.routers.config as config_router
import affine.api.routers.rank as rank_router
import affine.api.routers.scores as scores_router
from affine.api.routers.miners import get_miner_by_hotkey, get_miner_by_uid
from affine.api.routers.logs import get_miner_logs
from affine.api.routers.scores import (
    get_latest_scores, get_latest_weights, get_score_by_uid,
)


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


class _FakeSampleResultsDAO:
    """In-test stand-in for sample_results aggregation. Stores per
    (hotkey, revision, env) the list of scores the test wants present
    and counts each call so tests can assert on the cache short-circuit."""

    def __init__(self, by_subject=None):
        # {(hotkey, revision): {env: [scores]}}
        self._by_subject = by_subject or {}
        self.calls: list = []

    async def get_avg_scores_for_envs(self, hotkey, revision, envs):
        self.calls.append((hotkey, revision, tuple(envs)))
        bucket = self._by_subject.get((hotkey, revision)) or {}
        out = {}
        for env in envs:
            scores = bucket.get(env) or []
            if not scores:
                continue
            out[env] = {
                "score": sum(scores) / len(scores),
                "sample_count": len(scores),
            }
        return out


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
    monkeypatch.setattr(
        scores_router, "SampleResultsDAO", lambda: _FakeSampleResultsDAO(),
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
    monkeypatch.setattr(
        scores_router, "SampleResultsDAO", lambda: _FakeSampleResultsDAO(),
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
async def test_scores_latest_includes_miners_missing_from_snapshot(monkeypatch):
    """Newly-online miners (registered after the last decided snapshot)
    must still appear in /scores/latest, with per-env averages computed
    from sample_results — not be silently dropped because the snapshot
    only contains miners that were valid at the last decide block.
    """
    monkeypatch.setattr(
        scores_router,
        "MinersDAO",
        lambda: _FakeMinersDAO({
            ("uid", 7): {
                "uid": 7,
                "hotkey": "old-hk",
                "revision": "rev-old",
                "is_valid": "true",
                "model": "org/old",
                "first_block": 100,
            },
            ("uid", 99): {
                "uid": 99,
                "hotkey": "new-hk",
                "revision": "rev-new",
                "is_valid": "true",
                "model": "org/new",
                "first_block": 500,
            },
        }),
    )
    monkeypatch.setattr(
        scores_router,
        "MinerStatsDAO",
        lambda: _FakeMinerStatsDAO({
            ("old-hk", "rev-old"): {
                "challenge_status": "champion",
                "termination_reason": "",
            },
            ("new-hk", "rev-new"): {
                "challenge_status": "sampling",
                "termination_reason": "",
            },
        }),
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
            },
        }),
    )
    # new-hk has 4 historical samples in sample_results; old-hk has 2.
    monkeypatch.setattr(
        scores_router,
        "SampleResultsDAO",
        lambda: _FakeSampleResultsDAO({
            ("old-hk", "rev-old"): {"SWE": [0.5, 0.6]},
            ("new-hk", "rev-new"): {"SWE": [0.7, 0.7, 0.8, 0.9]},
        }),
    )

    response = await get_latest_scores(
        top=256,
        dao=_FakeScoresDAO({
            "block_number": 100,
            "calculated_at": 200,
            "scores": [
                # only the OLD champion is in the snapshot — new-hk
                # registered after the last decide so it's absent here.
                {
                    "uid": 7,
                    "miner_hotkey": "old-hk",
                    "model_revision": "rev-old",
                    "model": "org/old",
                    "first_block": 100,
                    "overall_score": 1.0,
                    "average_score": 0.55,
                    "scores_by_env": {"SWE": {"score": 0.55, "sample_count": 2}},
                    "total_samples": 2,
                },
            ],
        }),
    )

    by_uid = {row.uid: row for row in response.scores}
    # Both miners must show up — old-hk from snapshot, new-hk from
    # validity + sample_results aggregation.
    assert set(by_uid.keys()) == {7, 99}
    # Snapshot's per-env scores win for the champion.
    assert by_uid[7].scores_by_env["SWE"]["score"] == pytest.approx(0.55)
    # New miner's per-env score comes from sample_results aggregation.
    assert by_uid[99].scores_by_env["SWE"]["score"] == pytest.approx(0.775)
    assert by_uid[99].scores_by_env["SWE"]["sample_count"] == 4
    # Sort order: champion (overall=1.0) above new miner (overall=0.0).
    assert response.scores[0].uid == 7


@pytest.mark.asyncio
async def test_terminated_miners_cache_avoids_repeat_dao_queries(monkeypatch):
    """Terminated state is permanent for a (hotkey, revision) — re-commit
    creates a new revision, which is a different cache key. Two
    consecutive /scores/latest calls should read sample_results once
    per terminated miner and reuse the cached aggregate on the second
    call. Active (sampling/champion) miners keep hitting the DAO every
    time because their data is still moving.
    """
    scores_router._reset_terminated_cache_for_test()
    miners = {
        ("uid", 1): {
            "uid": 1, "hotkey": "champ-hk", "revision": "champ-rev",
            "model": "org/champ", "first_block": 10, "is_valid": "true",
        },
        ("uid", 2): {
            "uid": 2, "hotkey": "term-hk", "revision": "term-rev",
            "model": "org/term", "first_block": 20, "is_valid": "true",
        },
    }
    monkeypatch.setattr(scores_router, "MinersDAO", lambda: _FakeMinersDAO(miners))
    monkeypatch.setattr(
        scores_router, "MinerStatsDAO", lambda: _FakeMinerStatsDAO({
            ("champ-hk", "champ-rev"): {
                "challenge_status": "champion", "termination_reason": "",
            },
            ("term-hk", "term-rev"): {
                "challenge_status": "terminated",
                "termination_reason": "lost_to_champion",
            },
        }),
    )
    monkeypatch.setattr(
        scores_router, "SystemConfigDAO", lambda: _FakeSystemConfigDAO({
            "environments": {
                "SWE": {
                    "enabled_for_sampling": True, "enabled_for_scoring": True,
                    "sampling": {"sampling_count": 10, "dataset_range": [[0, 100]]},
                },
            },
        }),
    )
    samples_dao = _FakeSampleResultsDAO({
        ("champ-hk", "champ-rev"): {"SWE": [0.8, 0.9]},
        ("term-hk", "term-rev"): {"SWE": [0.4, 0.5]},
    })
    monkeypatch.setattr(scores_router, "SampleResultsDAO", lambda: samples_dao)

    fake_scores_dao = _FakeScoresDAO({
        "block_number": 100, "calculated_at": 200, "scores": [],
    })

    # First call: warm-cache for both miners.
    await get_latest_scores(top=256, dao=fake_scores_dao)
    first_call_subjects = {(c[0], c[1]) for c in samples_dao.calls}
    assert ("champ-hk", "champ-rev") in first_call_subjects
    assert ("term-hk", "term-rev") in first_call_subjects
    first_call_count = len(samples_dao.calls)

    # Second call: terminated miner should not re-query, AND champion
    # also hits the cache because we're well within the active TTL
    # (default ~10 min). The point is: API request cost stays bounded
    # even if the caller hammers /scores/latest in a tight loop.
    await get_latest_scores(top=256, dao=fake_scores_dao)

    second_pass_calls = samples_dao.calls[first_call_count:]
    second_pass_subjects = {(c[0], c[1]) for c in second_pass_calls}
    assert ("term-hk", "term-rev") not in second_pass_subjects, (
        "terminated miner should hit permanent cache, not re-query DAO"
    )
    assert ("champ-hk", "champ-rev") not in second_pass_subjects, (
        "champion within TTL window should hit cache; "
        "API requests must not scale linearly with caller rate"
    )


@pytest.mark.asyncio
async def test_active_miners_aggregate_refreshes_after_ttl(monkeypatch):
    """Active miners' (champion + in_progress) cache entry must fall
    out of the cache after the TTL so rank scores trail the live
    battle by at most one window. This pins the TTL behavior down so
    a future change can't accidentally turn the active path into
    permanent caching."""
    scores_router._reset_terminated_cache_for_test()
    monkeypatch.setattr(scores_router, "_ACTIVE_CACHE_TTL_S", 0.1)

    miners = {
        ("uid", 1): {
            "uid": 1, "hotkey": "champ-hk", "revision": "champ-rev",
            "model": "org/champ", "first_block": 10, "is_valid": "true",
        },
    }
    monkeypatch.setattr(scores_router, "MinersDAO", lambda: _FakeMinersDAO(miners))
    monkeypatch.setattr(
        scores_router, "MinerStatsDAO", lambda: _FakeMinerStatsDAO({
            ("champ-hk", "champ-rev"): {
                "challenge_status": "champion", "termination_reason": "",
            },
        }),
    )
    monkeypatch.setattr(
        scores_router, "SystemConfigDAO", lambda: _FakeSystemConfigDAO({
            "environments": {
                "SWE": {
                    "enabled_for_sampling": True, "enabled_for_scoring": True,
                    "sampling": {"sampling_count": 10, "dataset_range": [[0, 100]]},
                },
            },
        }),
    )
    samples_dao = _FakeSampleResultsDAO({
        ("champ-hk", "champ-rev"): {"SWE": [0.5]},
    })
    monkeypatch.setattr(scores_router, "SampleResultsDAO", lambda: samples_dao)

    fake_scores_dao = _FakeScoresDAO({
        "block_number": 100, "calculated_at": 200, "scores": [],
    })

    await get_latest_scores(top=256, dao=fake_scores_dao)
    assert sum(1 for c in samples_dao.calls if c[0] == "champ-hk") == 1

    # Within TTL: cache hit, no new query.
    await get_latest_scores(top=256, dao=fake_scores_dao)
    assert sum(1 for c in samples_dao.calls if c[0] == "champ-hk") == 1

    import asyncio as _asyncio
    await _asyncio.sleep(0.15)  # past the 0.1s TTL set above

    # After TTL: refresh, one more query.
    await get_latest_scores(top=256, dao=fake_scores_dao)
    assert sum(1 for c in samples_dao.calls if c[0] == "champ-hk") == 2


@pytest.mark.asyncio
async def test_score_by_uid_for_miner_missing_from_snapshot(monkeypatch):
    """``af get-miner`` (single-miner score lookup) must not 404 just
    because the miner isn't in the latest decided snapshot — falls
    back to sample_results aggregation for per-env scores."""
    miner_row = {
        "uid": 99,
        "hotkey": "new-hk",
        "revision": "rev-new",
        "model": "org/new",
        "first_block": 500,
        "is_valid": "true",
    }
    miners_dao = _FakeMinersDAO({("uid", 99): miner_row})
    monkeypatch.setattr(scores_router, "MinersDAO", lambda: miners_dao)
    monkeypatch.setattr(
        scores_router,
        "MinerStatsDAO",
        lambda: _FakeMinerStatsDAO({
            ("new-hk", "rev-new"): {
                "challenge_status": "sampling",
                "termination_reason": "",
            },
        }),
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
            },
        }),
    )
    monkeypatch.setattr(
        scores_router,
        "SampleResultsDAO",
        lambda: _FakeSampleResultsDAO({
            ("new-hk", "rev-new"): {"SWE": [0.4, 0.6]},
        }),
    )

    response = await get_score_by_uid(
        uid=99,
        dao=_FakeScoresDAO({
            "block_number": 100,
            "calculated_at": 200,
            "scores": [],
        }),
        miners_dao=miners_dao,
    )

    assert response.uid == 99
    assert response.miner_hotkey == "new-hk"
    assert response.scores_by_env["SWE"]["score"] == pytest.approx(0.5)
    assert response.scores_by_env["SWE"]["sample_count"] == 2


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
async def test_weights_endpoint_accepts_legacy_miner_final_scores():
    response = await get_latest_weights(
        snapshots_dao=_FakeScoreSnapshotsDAO({
            "block_number": 123,
            "statistics": {
                "miner_final_scores": {
                    "7": 1,
                    "8": 0,
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
