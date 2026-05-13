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
