"""API ``/windows/*`` endpoints — empty / partial state handling.

The router pulls from ``StateStore``. Each endpoint must return a sane
response when system_config is empty (fresh subnet, scheduler not yet
ticked) and when only a subset of keys is present (e.g. champion set
but no battle, task_state still missing).

In-memory KV store + the router's response-construction logic — no
FastAPI dependency injection, no live HTTP, no DB.
"""

from __future__ import annotations

import pytest

from affine.src.scorer.window_state import (
    BattleRecord,
    ChampionRecord,
    InMemoryConfigStore,
    MinerSnapshot,
    StateStore,
    TaskIdState,
)
from affine.api.routers.windows import _infer_champion_from_scores


# Pull the same response-shape construction the router uses. Re-implementing
# it here (verbatim from windows.py) keeps the test off FastAPI's machinery.
def _miner_summary(snapshot):
    if snapshot is None:
        return None
    return {
        "uid": snapshot.uid,
        "hotkey": snapshot.hotkey,
        "revision": snapshot.revision,
        "model": snapshot.model,
    }


async def _build_current(store: StateStore):
    champion = await store.get_champion()
    battle = await store.get_battle()
    task_state = await store.get_task_state()
    return {
        "champion": _miner_summary(champion) if champion else None,
        "champion_base_url": champion.base_url if champion else None,
        "battle": {
            "challenger": _miner_summary(battle.challenger),
            "started_at_block": battle.started_at_block,
        } if battle else None,
        "task_refresh_block": task_state.refreshed_at_block if task_state else None,
        "sample_counts": {},
    }


# ---- empty-state ------------------------------------------------------------


@pytest.mark.asyncio
async def test_current_endpoint_on_empty_state():
    """Fresh subnet: no champion, no battle, no task_state. Must not raise."""
    store = StateStore(InMemoryConfigStore())
    resp = await _build_current(store)
    assert resp == {
        "champion": None,
        "champion_base_url": None,
        "battle": None,
        "task_refresh_block": None,
        "sample_counts": {},
    }


@pytest.mark.asyncio
async def test_current_endpoint_with_champion_only():
    """Champion set but no battle yet (idle scheduler holding crown)."""
    kv = InMemoryConfigStore()
    store = StateStore(kv)
    await store.set_champion(ChampionRecord(
        uid=12, hotkey="X", revision="r", model="org/m",
        deployment_id="wrk-1", base_url="https://t/w1", since_block=99,
    ))
    resp = await _build_current(store)
    assert resp["champion"]["uid"] == 12
    assert resp["champion_base_url"] == "https://t/w1"
    assert resp["battle"] is None
    assert resp["task_refresh_block"] is None


@pytest.mark.asyncio
async def test_current_endpoint_with_battle_in_flight():
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
    resp = await _build_current(store)
    assert resp["champion"]["uid"] == 1
    assert resp["battle"]["challenger"]["uid"] == 2
    assert resp["battle"]["started_at_block"] == 42
    assert resp["task_refresh_block"] == 42


@pytest.mark.asyncio
async def test_champion_endpoint_returns_uid_none_when_empty():
    """``/windows/champion`` must return ``{uid: None}`` when no champion."""
    store = StateStore(InMemoryConfigStore())
    champ = await store.get_champion()
    if champ is None:
        result = {"uid": None}
    else:
        result = {"uid": champ.uid, "hotkey": champ.hotkey,
                  "revision": champ.revision, "model": champ.model,
                  "since_block": champ.since_block}
    assert result == {"uid": None}


@pytest.mark.asyncio
async def test_champion_endpoint_with_real_record():
    store = StateStore(InMemoryConfigStore())
    await store.set_champion(ChampionRecord(
        uid=7, hotkey="hk", revision="rv", model="o/m",
        deployment_id="d", base_url="u", since_block=100,
    ))
    champ = await store.get_champion()
    result = {
        "uid": champ.uid, "hotkey": champ.hotkey,
        "revision": champ.revision, "model": champ.model,
        "since_block": champ.since_block,
    }
    assert result["uid"] == 7
    assert result["since_block"] == 100


class _FakeScoresDAO:
    def __init__(self, payload):
        self.payload = payload

    async def get_latest_scores(self, limit=None):
        return self.payload


@pytest.mark.asyncio
async def test_infer_champion_from_latest_weight_snapshot(monkeypatch):
    monkeypatch.setattr(
        "affine.api.routers.windows.ScoresDAO",
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

    champ = await _infer_champion_from_scores()

    assert champ.uid == 7
    assert champ.hotkey == "hk"
    assert champ.revision == "rev"
    assert champ.since_block == 123
