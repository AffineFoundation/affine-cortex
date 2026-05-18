"""DAO adapter regression tests."""

from __future__ import annotations

import pytest

import affine.database.client as db_client
from affine.database.dao.miner_stats import MinerStatsDAO
from affine.src.monitor.miners_monitor import MinerInfo, MinersMonitor
from affine.src.scorer.dao_adapters import SampleResultsAdapter


class _FakeDynamoClient:
    def __init__(self, query_pages=None):
        self.put_calls = []
        self.query_pages = list(query_pages or [])

    async def put_item(self, **kwargs):
        self.put_calls.append(kwargs)

    async def query(self, **kwargs):
        if self.query_pages:
            return self.query_pages.pop(0)
        return {"Items": []}


class _FakeSampleDAO:
    table_name = "sample_results"

    def _make_pk(self, hotkey, revision, env):
        return f"MINER#{hotkey}#REV#{revision}#ENV#{env}"

    def _make_sk(self, task_id):
        return f"TASK#{task_id}"

    def compress_data(self, payload):
        return payload.encode("utf-8")

    def _serialize(self, item):
        return item

    def _deserialize(self, item):
        return item


@pytest.mark.asyncio
async def test_sample_results_adapter_persist_uses_real_runtime_imports(monkeypatch):
    client = _FakeDynamoClient()
    monkeypatch.setattr(
        "affine.src.scorer.dao_adapters.get_client",
        lambda: client,
    )

    adapter = SampleResultsAdapter(dao=_FakeSampleDAO())
    await adapter.persist(
        miner_hotkey="hk",
        model_revision="rev",
        model="org/model",
        env="DISTILL",
        task_id=123,
        score=0.75,
        latency_ms=42,
        extra={"ok": True},
        block_number=1000,
        refresh_block=7200,
    )

    assert len(client.put_calls) == 1
    item = client.put_calls[0]["Item"]
    assert item["pk"] == "MINER#hk#REV#rev#ENV#DISTILL"
    assert item["sk"] == "TASK#123"
    assert item["ttl"] > 0
    assert item["refresh_block"] == 7200


@pytest.mark.asyncio
async def test_sample_results_adapter_read_scores_uses_real_runtime_import(monkeypatch):
    client = _FakeDynamoClient([
        {
            "Items": [
                {"task_id": 1, "score": 0.1, "refresh_block": 10},
                {"task_id": 2, "score": 0.2, "refresh_block": 9},
                {"task_id": 3, "score": 0.3, "refresh_block": 10},
            ]
        }
    ])
    monkeypatch.setattr(
        "affine.src.scorer.dao_adapters.get_client",
        lambda: client,
    )

    adapter = SampleResultsAdapter(dao=_FakeSampleDAO())
    scores = await adapter.read_scores_for_tasks(
        "hk", "rev", "DISTILL", [1, 2], refresh_block=10,
    )

    assert scores == {1: 0.1}


class _MemoryMinerStatsDAO(MinerStatsDAO):
    def __init__(self, direct, rows):
        super().__init__()
        self.direct = direct
        self.rows = rows

    async def get_miner_stats(self, hotkey, revision):
        return self.direct.get((hotkey, revision))

    async def query(self, pk, *args, **kwargs):
        return list(self.rows)


class _UpdateRecordingClient:
    def __init__(self):
        self.update_calls = []

    async def update_item(self, **kwargs):
        self.update_calls.append(kwargs)


@pytest.mark.asyncio
async def test_update_challenge_status_persists_scores_and_freeze_marker(monkeypatch):
    """The comparator's frozen view of a terminated miner shares the
    SAME ``scores_by_env`` attribute live updates use — the
    ``terminated_at_block`` flag distinguishes frozen rows. Atomic
    write so a concurrent live-cycle refresh can't slip in between
    the status flip and the score persist."""
    client = _UpdateRecordingClient()
    monkeypatch.setattr(
        "affine.database.client.get_client",
        lambda: client,
    )
    dao = MinerStatsDAO()
    await dao.update_challenge_status(
        hotkey="hk",
        revision="rev",
        status=MinerStatsDAO.STATUS_TERMINATED,
        termination_reason="lost_to_champion:foo",
        scores_by_env={
            "ENV_A": {"count": 178, "avg": 0.48,
                      "champion_overlap_avg": 0.50},
        },
        scores_refresh_block=8000,
        terminated_at_block=9001,
    )
    assert len(client.update_calls) == 1
    call = client.update_calls[0]
    expr = call["UpdateExpression"]
    values = call["ExpressionAttributeValues"]
    assert "scores_by_env = :sc" in expr
    assert "scores_refresh_block = :srb" in expr
    assert "terminated_at_block = :tb" in expr
    assert values[":tb"] == {"N": "9001"}
    assert values[":srb"] == {"N": "8000"}
    env_a = values[":sc"]["M"]["ENV_A"]["M"]
    assert env_a["count"]["N"] == "178"
    assert env_a["avg"]["N"].startswith("0.48")
    assert env_a["champion_overlap_avg"]["N"].startswith("0.5")


@pytest.mark.asyncio
async def test_update_challenge_status_omits_score_fields_when_not_provided(monkeypatch):
    """Status-only callers (recovery, mid-battle invalidation,
    deploy-failed) don't carry comparator data — those paths must keep
    producing a clean status-only ``UpdateExpression`` without empty
    score placeholders."""
    client = _UpdateRecordingClient()
    monkeypatch.setattr(
        "affine.database.client.get_client",
        lambda: client,
    )
    dao = MinerStatsDAO()
    await dao.update_challenge_status(
        hotkey="hk",
        revision="rev",
        status=MinerStatsDAO.STATUS_TERMINATED,
        termination_reason="dethroned_by:bar:recovery",
    )
    expr = client.update_calls[0]["UpdateExpression"]
    values = client.update_calls[0]["ExpressionAttributeValues"]
    assert "scores_by_env" not in expr
    assert "scores_refresh_block" not in expr
    assert "terminated_at_block" not in expr
    assert ":sc" not in values
    assert ":srb" not in values
    assert ":tb" not in values


class _StubMinerStatsForMap(MinerStatsDAO):
    def __init__(self, rows):
        super().__init__()
        self._rows = rows

    async def get_miner_stats(self, hotkey, revision):
        return self._rows.get((hotkey, revision))


@pytest.mark.asyncio
async def test_build_display_scores_map_drops_stale_live_and_marks_frozen():
    """One ``scores_by_env`` field on every row; the ``frozen`` flag
    distinguishes the comparator's decide-time snapshot from a fresh
    live snapshot. Stale-and-not-frozen rows are dropped because the
    CLI must not show numbers from a previous task pool."""
    dao = _StubMinerStatsForMap({
        # Terminated row → frozen=True regardless of refresh_block.
        ("hk1", "r1"): {
            "scores_by_env": {
                "ENV_A": {"count": 50, "avg": 0.4, "champion_overlap_avg": 0.5},
            },
            "scores_refresh_block": 99,  # decide-time block, may differ
            "terminated_at_block": 9001,
        },
        # Fresh live, never terminated → frozen=False.
        ("hk2", "r2"): {
            "scores_by_env": {"ENV_A": {"count": 3, "avg": 0.6}},
            "scores_refresh_block": 100,
        },
        # Stale live (different refresh, not terminated) → dropped.
        ("hk3", "r3"): {
            "scores_by_env": {"ENV_A": {"count": 1, "avg": 0.9}},
            "scores_refresh_block": 99,
        },
        # Sampling row with nothing → absent from result.
        ("hk4", "r4"): {"challenge_status": "sampling"},
    })
    miners = [
        {"hotkey": "hk1", "revision": "r1", "uid": 11},
        {"hotkey": "hk2", "revision": "r2", "uid": 22},
        {"hotkey": "hk3", "revision": "r3", "uid": 33},
        {"hotkey": "hk4", "revision": "r4", "uid": 44},
        {"hotkey": "hk_unknown", "revision": "r9", "uid": 99},  # no row
    ]
    out = await dao.build_display_scores_map(miners, current_refresh_block=100)

    assert set(out.keys()) == {"11", "22"}
    assert out["11"]["frozen"] is True
    assert out["11"]["scores"]["ENV_A"]["count"] == 50
    assert out["22"]["frozen"] is False
    assert out["22"]["scores"]["ENV_A"]["avg"] == 0.6


@pytest.mark.asyncio
async def test_build_display_scores_map_keeps_frozen_when_no_refresh_block():
    """``current_refresh_block=None`` (no task pool yet) → every live
    snapshot is treated as stale, but frozen rows still render. So
    terminated miners survive a cold-start API response."""
    dao = _StubMinerStatsForMap({
        ("hk1", "r1"): {
            "scores_by_env": {"ENV_A": {"count": 50, "avg": 0.4,
                                        "champion_overlap_avg": 0.5}},
            "scores_refresh_block": 100,
            "terminated_at_block": 9000,
        },
        ("hk2", "r2"): {
            "scores_by_env": {"ENV_A": {"count": 7, "avg": 0.8}},
            "scores_refresh_block": 100,
        },
    })
    miners = [
        {"hotkey": "hk1", "revision": "r1", "uid": 11},
        {"hotkey": "hk2", "revision": "r2", "uid": 22},
    ]
    out = await dao.build_display_scores_map(miners, current_refresh_block=None)
    assert list(out.keys()) == ["11"]
    assert out["11"]["frozen"] is True


@pytest.mark.asyncio
async def test_update_live_scores_writes_per_miner_payload(monkeypatch):
    """LiveScoresMonitor's per-miner upsert writes to the SAME
    ``scores_by_env`` attribute the comparator writes to at
    termination. The conditional ``attribute_not_exists(terminated_at_block)``
    is what prevents live cycles from clobbering a frozen row."""
    client = _UpdateRecordingClient()
    monkeypatch.setattr(
        "affine.database.client.get_client",
        lambda: client,
    )
    dao = MinerStatsDAO()
    await dao.update_live_scores(
        hotkey="hk",
        revision="rev",
        scores_by_env={
            "ENV_A": {"count": 10, "avg": 0.45, "champion_overlap_avg": 0.5},
        },
        scores_refresh_block=12345,
    )
    assert len(client.update_calls) == 1
    call = client.update_calls[0]
    expr = call["UpdateExpression"]
    values = call["ExpressionAttributeValues"]
    assert "scores_by_env = :sc" in expr
    assert "scores_refresh_block = :srb" in expr
    assert call.get("ConditionExpression") == "attribute_not_exists(terminated_at_block)"
    assert values[":srb"] == {"N": "12345"}
    env_a = values[":sc"]["M"]["ENV_A"]["M"]
    assert env_a["count"]["N"] == "10"
    assert env_a["avg"]["N"].startswith("0.45")
    assert env_a["champion_overlap_avg"]["N"].startswith("0.5")


@pytest.mark.asyncio
async def test_update_live_scores_swallows_frozen_row_conflict(monkeypatch):
    """A ``ConditionalCheckFailedException`` from a frozen row is the
    expected outcome (live cycle tried to overwrite a comparator-set
    snapshot); the DAO swallows it. Any OTHER ClientError must
    propagate so the LiveScoresMonitor logs it."""
    from botocore.exceptions import ClientError

    class _RejectingClient:
        async def update_item(self, **kwargs):
            raise ClientError(
                error_response={"Error": {"Code": "ConditionalCheckFailedException"}},
                operation_name="UpdateItem",
            )

    monkeypatch.setattr(
        "affine.database.client.get_client",
        lambda: _RejectingClient(),
    )
    dao = MinerStatsDAO()
    # Must not raise.
    await dao.update_live_scores(
        hotkey="hk", revision="rev",
        scores_by_env={"ENV_A": {"count": 1, "avg": 0.5}},
        scores_refresh_block=10,
    )

    class _ThrowingClient:
        async def update_item(self, **kwargs):
            raise ClientError(
                error_response={"Error": {"Code": "ProvisionedThroughputExceededException"}},
                operation_name="UpdateItem",
            )

    monkeypatch.setattr(
        "affine.database.client.get_client",
        lambda: _ThrowingClient(),
    )
    with pytest.raises(ClientError):
        await dao.update_live_scores(
            hotkey="hk", revision="rev",
            scores_by_env={"ENV_A": {"count": 1, "avg": 0.5}},
            scores_refresh_block=10,
        )


@pytest.mark.asyncio
async def test_miner_stats_hotkey_fallback_blocks_later_revision_claim():
    dao = _MemoryMinerStatsDAO(
        direct={
            ("hk", "new"): {
                "hotkey": "hk",
                "revision": "new",
                "challenge_status": "sampling",
            }
        },
        rows=[
            {
                "hotkey": "hk",
                "revision": "old",
                "challenge_status": "terminated",
                "termination_reason": "lost_to_champion",
                "last_updated_at": 10,
            }
        ],
    )

    state = await dao.get_challenge_state("hk", "new")
    claimed = await dao.claim_for_challenge(
        hotkey="hk", revision="new", model="org/model", window_id=1,
    )

    assert state["challenge_status"] == "terminated"
    assert state["termination_reason"] == "lost_to_champion"
    assert claimed is False


class _FakeMinersDAO:
    def __init__(self):
        self.saved = []
        self.table_name = "miners"

    async def save_miner(self, **kwargs):
        self.saved.append(kwargs)


class _FakeMinerStatsDAO:
    def __init__(self):
        self.updated = []

    async def update_miner_info(self, **kwargs):
        self.updated.append(kwargs)


@pytest.mark.asyncio
async def test_monitor_persists_current_metadata_to_miner_stats(monkeypatch):
    monkeypatch.setattr(db_client, "get_client", lambda: _FakeDynamoClient())
    monitor = MinersMonitor()
    monitor.dao = _FakeMinersDAO()
    monitor.stats_dao = _FakeMinerStatsDAO()

    await monitor._persist_miners(
        [
            MinerInfo(
                uid=7,
                hotkey="hk",
                model="org/model",
                revision="rev",
                block=100,
                is_valid=False,
                invalid_reason="model_check:too_large",
                model_hash="hash",
            ),
            MinerInfo(uid=8, hotkey="no-rev", model="", revision="", block=101),
        ],
        current_block=200,
    )

    assert len(monitor.dao.saved) == 2
    assert monitor.stats_dao.updated == [
        {
            "hotkey": "hk",
            "revision": "rev",
            "model": "org/model",
            "uid": 7,
            "first_block": 100,
            "block_number": 200,
            "is_valid": False,
            "invalid_reason": "model_check:too_large",
            "model_hash": "hash",
            "is_online": True,
        }
    ]
