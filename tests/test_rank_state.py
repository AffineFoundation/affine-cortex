"""Rank state payload helpers.

The public ``/rank/current`` endpoint is a small aggregation layer over these
helpers. These tests keep the response shape stable when system_config is
empty or partially populated.
"""

from __future__ import annotations

import pytest

import affine.api.rank_state as rank_state
from affine.src.scorer.window_state import (
    BattleRecord,
    ChampionRecord,
    InMemoryConfigStore,
    MinerSnapshot,
    StateStore,
    TaskIdState,
)


async def _empty_counts(champion, battle, task_state):
    return {}


async def _no_inferred_champion():
    return None


async def _render_current(monkeypatch, store: StateStore):
    monkeypatch.setattr(rank_state, "_state_store", lambda: store)
    monkeypatch.setattr(rank_state, "_sample_counts", _empty_counts)
    monkeypatch.setattr(rank_state, "_infer_champion_from_scores", _no_inferred_champion)
    return await rank_state.get_current_state()


# ---- empty-state ------------------------------------------------------------


@pytest.mark.asyncio
async def test_current_state_on_empty_state(monkeypatch):
    """Fresh subnet: no champion, no battle, no task_state. Must not raise."""
    store = StateStore(InMemoryConfigStore())
    resp = await _render_current(monkeypatch, store)
    assert resp == {
        "champion": None,
        "battle": None,
        "task_refresh_block": None,
        "sample_counts": {},
        "live_sampling_uids": [],
    }


@pytest.mark.asyncio
async def test_current_state_with_champion_only(monkeypatch):
    """Champion set but no battle yet (idle scheduler holding crown)."""
    kv = InMemoryConfigStore()
    store = StateStore(kv)
    await store.set_champion(ChampionRecord(
        uid=12, hotkey="X", revision="r", model="org/m",
        deployment_id="wrk-1", base_url="https://t/w1", since_block=99,
    ))
    resp = await _render_current(monkeypatch, store)
    assert resp["champion"]["uid"] == 12
    assert "champion_base_url" not in resp
    assert resp["battle"] is None
    assert resp["task_refresh_block"] is None


@pytest.mark.asyncio
async def test_current_state_with_battle_in_flight(monkeypatch):
    store = StateStore(InMemoryConfigStore())
    await store.set_champion(ChampionRecord(
        uid=1, hotkey="A", revision="r1", model="org/a",
        deployment_id="wrk-A", base_url="https://t/A", since_block=0,
    ))
    await store.set_battle(BattleRecord(
        challenger=MinerSnapshot(uid=2, hotkey="B", revision="r2", model="org/b"),
        deployment_id="wrk-B", base_url="https://t/B", started_at_block=42,
    ))
    await store.set_task_state(TaskIdState(
        task_ids={"ENV_A": [1, 2, 3]}, refreshed_at_block=42,
    ))
    resp = await _render_current(monkeypatch, store)
    assert resp["champion"]["uid"] == 1
    assert resp["battle"]["challenger"]["uid"] == 2
    assert resp["battle"]["started_at_block"] == 42
    assert resp["task_refresh_block"] == 42


@pytest.mark.asyncio
async def test_current_state_marks_only_incomplete_subjects_as_live_sampling(monkeypatch):
    async def _counts(champion, battle, task_state):
        return {
            "1": {"ENV_A": 5},
            "2": {"ENV_A": 2},
        }

    kv = InMemoryConfigStore()
    kv.data["environments"] = {
        "ENV_A": {
            "display_name": "A",
            "enabled_for_sampling": True,
            "sampling": {"sampling_count": 5, "dataset_range": [[0, 100]]},
        },
    }
    store = StateStore(kv)
    await store.set_champion(ChampionRecord(
        uid=1, hotkey="A", revision="r1", model="org/a",
        deployment_id="wrk-A", base_url="https://t/A", since_block=0,
    ))
    await store.set_battle(BattleRecord(
        challenger=MinerSnapshot(uid=2, hotkey="B", revision="r2", model="org/b"),
        deployment_id="wrk-B", base_url="https://t/B", started_at_block=42,
    ))
    await store.set_task_state(TaskIdState(
        task_ids={"ENV_A": [1, 2, 3, 4, 5, 6]}, refreshed_at_block=42,
    ))
    monkeypatch.setattr(rank_state, "_state_store", lambda: store)
    monkeypatch.setattr(rank_state, "_sample_counts", _counts)
    monkeypatch.setattr(rank_state, "_infer_champion_from_scores", _no_inferred_champion)

    resp = await rank_state.get_current_state()

    assert resp["sample_counts"] == {"1": {"ENV_A": 5}, "2": {"ENV_A": 2}}
    assert resp["live_sampling_uids"] == [2]


class _FakeScoresDAO:
    def __init__(self, payload):
        self.payload = payload

    async def get_latest_scores(self, limit=None):
        return self.payload


class _FakeMinersDAO:
    async def get_valid_miners(self):
        return [
            {
                "uid": 1,
                "hotkey": "lost",
                "revision": "r1",
                "model": "org/lost",
                "first_block": 10,
                "is_valid": "true",
            },
            {
                "uid": 2,
                "hotkey": "ready",
                "revision": "r2",
                "model": "org/ready",
                "first_block": 20,
                "is_valid": "true",
            },
        ]


class _FakeMinerStatsDAO:
    async def build_challenge_state_map(self, miners):
        return {
            ("lost", "r1"): {
                "challenge_status": "terminated",
                "termination_reason": "lost_to_champion",
            },
            ("ready", "r2"): {
                "challenge_status": "sampling",
                "termination_reason": "",
            },
        }


@pytest.mark.asyncio
async def test_queue_excludes_historical_terminated_miners(monkeypatch):
    monkeypatch.setattr("affine.api.rank_state.MinersDAO", _FakeMinersDAO)
    monkeypatch.setattr("affine.api.rank_state.MinerStatsDAO", _FakeMinerStatsDAO)

    queue = await rank_state.get_queue(limit=10)

    assert [row["uid"] for row in queue] == [2]
    assert queue[0]["challenge_status"] == "sampling"


@pytest.mark.asyncio
async def test_infer_champion_from_latest_weight_snapshot(monkeypatch):
    monkeypatch.setattr(
        "affine.api.rank_state.ScoresDAO",
        lambda: _FakeScoresDAO({
            "block_number": 123,
            "scores": [
                {
                    "uid": 7,
                    "miner_hotkey": "hk",
                    "model_revision": "rev",
                    "model": "org/model",
                    "overall_score": 1.0,
                },
                {
                    "uid": 8,
                    "miner_hotkey": "hk2",
                    "model_revision": "rev2",
                    "model": "org/model2",
                    "overall_score": 0.0,
                },
            ],
        }),
    )

    champ = await rank_state._infer_champion_from_scores()

    assert champ.uid == 7
    assert champ.hotkey == "hk"
    assert champ.revision == "rev"
    assert champ.since_block == 123
