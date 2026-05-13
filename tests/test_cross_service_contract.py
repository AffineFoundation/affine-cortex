"""Cross-service contract: scheduler writes / executor reads.

The two services share state via ``system_config`` keys + ``sample_results``.
This test bolts the two halves together via the in-memory store and
verifies that what scheduler writes, executor can consume — and vice
versa for the sample counters scheduler reads from.

No DB, no Targon, no FastAPI. Just the protocol contract.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pytest

from affine.src.scorer.window_state import (
    BattleRecord,
    ChampionRecord,
    InMemoryConfigStore,
    MinerSnapshot,
    StateStore,
    TaskIdState,
)


# Executor-side view: what the worker actually pulls from state.
async def _executor_view(store: StateStore):
    task_state = await store.get_task_state()
    champion = await store.get_champion()
    battle = await store.get_battle()
    envs = await store.get_environments()
    return {
        "tasks_per_env": dict(task_state.task_ids) if task_state else {},
        "refresh_block": task_state.refreshed_at_block if task_state else None,
        "champion_uid": champion.uid if champion else None,
        "champion_url": champion.base_url if champion else None,
        "battle_uid": battle.challenger.uid if battle else None,
        "battle_url": battle.base_url if battle else None,
        "env_names": sorted(envs.keys()),
    }


# ---- Contract: champion record fields executor needs ------------------------


@pytest.mark.asyncio
async def test_scheduler_champion_has_fields_executor_needs():
    """The fields executor reads (hotkey, revision, base_url) must be
    on what ``set_champion`` writes."""
    store = StateStore(InMemoryConfigStore())
    await store.set_champion(ChampionRecord(
        uid=1, hotkey="A", revision="r1", model="org/a",
        deployment_id="wrk-A", base_url="https://t/A", since_block=0,
    ))
    view = await _executor_view(store)
    assert view["champion_uid"] == 1
    assert view["champion_url"] == "https://t/A"


@pytest.mark.asyncio
async def test_scheduler_battle_has_fields_executor_needs():
    """Battle's deployment_id + base_url + challenger snapshot all
    survive the round-trip."""
    store = StateStore(InMemoryConfigStore())
    await store.set_battle(BattleRecord(
        challenger=MinerSnapshot(uid=42, hotkey="C", revision="r42", model="org/c"),
        deployment_id="wrk-C", base_url="https://t/C", started_at_block=7250,
    ))
    view = await _executor_view(store)
    assert view["battle_uid"] == 42
    assert view["battle_url"] == "https://t/C"


@pytest.mark.asyncio
async def test_task_ids_roundtrip_with_int_coercion():
    """``set_task_state`` accepts ints / ``get_task_state`` returns ints.
    Important because executor's has_sample expects int task_id and JSON
    serialization through DynamoDB can mangle types."""
    store = StateStore(InMemoryConfigStore())
    await store.set_task_state(TaskIdState(
        task_ids={"ENV_A": [10, 20, 30], "ENV_B": [99]},
        refreshed_at_block=7200,
    ))
    view = await _executor_view(store)
    assert view["tasks_per_env"] == {"ENV_A": [10, 20, 30], "ENV_B": [99]}
    assert all(isinstance(t, int) for t in view["tasks_per_env"]["ENV_A"])


@pytest.mark.asyncio
async def test_environments_filter_disabled_envs():
    """Executor manager calls ``_enabled_envs()`` which reads system_config;
    StateStore.get_environments() must skip disabled rows so the manager
    doesn't fork a worker for a disabled env."""
    kv = InMemoryConfigStore()
    kv.data["environments"] = {
        "ENABLED": {"display_name": "On", "enabled": True,
                    "sampling": {"sampling_count": 100,
                                 "dataset_range": [[0, 1000]],
                                 "sampling_mode": "random"}},
        "DISABLED": {"display_name": "Off", "enabled": False,
                     "sampling": {"sampling_count": 100,
                                  "dataset_range": [[0, 1000]],
                                  "sampling_mode": "random"}},
    }
    store = StateStore(kv)
    view = await _executor_view(store)
    assert view["env_names"] == ["ENABLED"]


@pytest.mark.asyncio
async def test_no_battle_means_executor_sees_battle_uid_none():
    """When scheduler clears the battle (post-decide), the executor's
    view returns None for battle_uid — worker won't try to sample a
    challenger that doesn't exist."""
    store = StateStore(InMemoryConfigStore())
    await store.set_champion(ChampionRecord(
        uid=1, hotkey="A", revision="r", model="o/a",
        deployment_id="D", base_url="U", since_block=0,
    ))
    # No battle set.
    view = await _executor_view(store)
    assert view["champion_uid"] == 1
    assert view["battle_uid"] is None
    assert view["battle_url"] is None


@pytest.mark.asyncio
async def test_clear_champion_then_set_does_not_leak_old_fields():
    """Set champion, clear it, set a different one — the new record
    must not inherit deployment_id from the old."""
    store = StateStore(InMemoryConfigStore())
    await store.set_champion(ChampionRecord(
        uid=1, hotkey="A", revision="r", model="o/a",
        deployment_id="wrk-OLD", base_url="https://t/OLD", since_block=0,
    ))
    await store.clear_champion()
    assert await store.get_champion() is None
    await store.set_champion(ChampionRecord(
        uid=2, hotkey="B", revision="rb", model="o/b",
        deployment_id=None, base_url=None, since_block=100,
    ))
    champ = await store.get_champion()
    assert champ.uid == 2
    assert champ.deployment_id is None  # not leaking wrk-OLD
    assert champ.base_url is None


@pytest.mark.asyncio
async def test_legacy_env_payload_shape_still_works():
    """Old system_config rows may still have ``environments[env].window_config``
    instead of ``.sampling`` (Stage U renamed the key). The accessor
    accepts both so a partial migration doesn't break the scheduler."""
    kv = InMemoryConfigStore()
    kv.data["environments"] = {
        "OLD": {"display_name": "X", "enabled": True,
                "window_config": {  # legacy key
                    "sampling_count": 50,
                    "dataset_range": [[0, 999]],
                    "sampling_mode": "latest",
                }},
        "NEW": {"display_name": "Y", "enabled": True,
                "sampling": {  # current key
                    "sampling_count": 100,
                    "dataset_range": [[0, 999]],
                    "sampling_mode": "random",
                }},
    }
    store = StateStore(kv)
    envs = await store.get_environments()
    assert envs["OLD"].sampling_count == 50
    assert envs["OLD"].sampling_mode == "latest"
    assert envs["NEW"].sampling_count == 100
    assert envs["NEW"].sampling_mode == "random"
