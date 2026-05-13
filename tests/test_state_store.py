"""Round-trip tests for the simplified StateStore."""

from __future__ import annotations

import pytest

from affine.src.scorer.window_state import (
    BattleRecord,
    ChampionRecord,
    EnvConfig,
    InMemoryConfigStore,
    MinerSnapshot,
    StateStore,
    TaskIdState,
)


@pytest.mark.asyncio
async def test_champion_roundtrip():
    store = StateStore(InMemoryConfigStore())
    champ = ChampionRecord(
        uid=5, hotkey="abc", revision="rev1", model="org/m5",
        deployment_id="wrk-001", base_url="https://t/x", since_block=100,
    )
    await store.set_champion(champ)
    got = await store.get_champion()
    assert got == champ


@pytest.mark.asyncio
async def test_champion_clear():
    store = StateStore(InMemoryConfigStore())
    await store.set_champion(ChampionRecord(uid=1, hotkey="x", revision="r", model="o/x"))
    assert await store.get_champion() is not None
    await store.clear_champion()
    assert await store.get_champion() is None


@pytest.mark.asyncio
async def test_battle_roundtrip():
    store = StateStore(InMemoryConfigStore())
    battle = BattleRecord(
        challenger=MinerSnapshot(uid=2, hotkey="bb", revision="rb", model="org/b"),
        deployment_id="wrk-chal", base_url="https://t/c",
        started_at_block=2000,
    )
    await store.set_battle(battle)
    got = await store.get_battle()
    assert got == battle


@pytest.mark.asyncio
async def test_battle_clear():
    store = StateStore(InMemoryConfigStore())
    await store.set_battle(BattleRecord(
        challenger=MinerSnapshot(uid=2, hotkey="bb", revision="rb", model="org/b"),
        deployment_id="x", base_url="y", started_at_block=0,
    ))
    await store.clear_battle()
    assert await store.get_battle() is None


@pytest.mark.asyncio
async def test_task_state_roundtrip():
    store = StateStore(InMemoryConfigStore())
    ts = TaskIdState(
        task_ids={"ENV_A": [1, 2, 3], "ENV_B": [10, 20]},
        refreshed_at_block=7200,
    )
    await store.set_task_state(ts)
    got = await store.get_task_state()
    assert got is not None
    assert got.task_ids == {"ENV_A": [1, 2, 3], "ENV_B": [10, 20]}
    assert got.refreshed_at_block == 7200


@pytest.mark.asyncio
async def test_environments_filter_disabled():
    kv = InMemoryConfigStore()
    kv.data["environments"] = {
        "A": {"display_name": "A", "enabled": True,
              "sampling": {"sampling_count": 10, "dataset_range": [[0, 100]], "sampling_mode": "random"}},
        "B": {"display_name": "B", "enabled": False,
              "sampling": {"sampling_count": 10, "dataset_range": [[0, 100]], "sampling_mode": "random"}},
    }
    store = StateStore(kv)
    envs = await store.get_environments()
    assert set(envs.keys()) == {"A"}
    assert isinstance(envs["A"], EnvConfig)
    assert envs["A"].sampling_count == 10


@pytest.mark.asyncio
async def test_environments_accepts_legacy_window_config_shape():
    """Old seed used ``environments[env].window_config`` instead of
    ``.sampling``. Coercion should accept both."""
    kv = InMemoryConfigStore()
    kv.data["environments"] = {
        "OLD": {"display_name": "Old", "enabled": True,
                "window_config": {"sampling_count": 50, "dataset_range": [[0, 1000]], "sampling_mode": "latest"}},
    }
    store = StateStore(kv)
    envs = await store.get_environments()
    assert envs["OLD"].sampling_count == 50
    assert envs["OLD"].sampling_mode == "latest"


@pytest.mark.asyncio
async def test_no_state_returns_none():
    store = StateStore(InMemoryConfigStore())
    assert await store.get_champion() is None
    assert await store.get_battle() is None
    assert await store.get_task_state() is None
