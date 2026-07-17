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
        deployment_generation="7dcab393ba2e4469b38744fc4a87ed73",
        behavior_admission_deadline_at=1_900_000_000,
    )
    await store.set_battle(battle)
    got = await store.get_battle()
    assert got == battle


@pytest.mark.asyncio
async def test_battle_decoder_distinguishes_legacy_missing_generation_from_empty():
    kv = InMemoryConfigStore()
    common = {
        "challenger": {
            "uid": 2,
            "hotkey": "bb",
            "revision": "rb",
            "model": "org/b",
        },
        "deployment_id": "wrk-chal",
        "base_url": "https://t/c",
        "started_at_block": 2000,
    }
    kv.data[StateStore.KEY_BATTLE] = dict(common)
    store = StateStore(kv)

    legacy = await store.get_battle()
    assert legacy is not None
    assert legacy.deployment_generation == "legacy-block:2000"

    # An explicitly persisted empty value is a valid old/default value.  Do
    # not reinterpret it as a missing-key legacy record during a round-trip.
    kv.data[StateStore.KEY_BATTLE] = {
        **common,
        "deployment_generation": "",
    }
    explicit_empty = await store.get_battle()
    assert explicit_empty is not None
    assert explicit_empty.deployment_generation == ""


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
        "A": {"display_name": "A", "enabled_for_sampling": True,
              "sampling": {"sampling_count": 10, "dataset_range": [[0, 100]], "sampling_mode": "random"}},
        "B": {"display_name": "B", "enabled_for_sampling": False,
              "sampling": {"sampling_count": 10, "dataset_range": [[0, 100]], "sampling_mode": "random"}},
    }
    store = StateStore(kv)
    envs = await store.get_environments()
    assert set(envs.keys()) == {"A"}
    assert isinstance(envs["A"], EnvConfig)
    assert envs["A"].sampling_count == 10


@pytest.mark.asyncio
async def test_get_scoring_environments_filters_independently_from_sampling():
    """``enabled_for_sampling`` gates the sampling pipeline;
    ``enabled_for_scoring`` is independently checked at DECIDE so a
    new env can collect data without affecting the comparator."""
    kv = InMemoryConfigStore()
    kv.data["environments"] = {
        "SAMPLE_AND_SCORE": {
            "display_name": "Both", "enabled_for_sampling": True,
            "enabled_for_scoring": True,
            "sampling": {"sampling_count": 10, "dataset_range": [[0, 100]], "sampling_mode": "random"},
        },
        "SAMPLE_ONLY": {
            "display_name": "Onboarding", "enabled_for_sampling": True,
            "enabled_for_scoring": False,
            "sampling": {"sampling_count": 10, "dataset_range": [[0, 100]], "sampling_mode": "random"},
        },
        "DISABLED": {
            "display_name": "Off", "enabled_for_sampling": False,
            "enabled_for_scoring": False,
            "sampling": {"sampling_count": 10, "dataset_range": [[0, 100]], "sampling_mode": "random"},
        },
    }
    store = StateStore(kv)

    sampling = await store.get_environments()
    scoring = await store.get_scoring_environments()

    assert set(sampling.keys()) == {"SAMPLE_AND_SCORE", "SAMPLE_ONLY"}
    assert set(scoring.keys()) == {"SAMPLE_AND_SCORE"}


@pytest.mark.asyncio
async def test_rank_display_includes_sampling_only_derived_env():
    kv = InMemoryConfigStore()
    kv.data["environments"] = {
        "SWE-INFINITE": {
            "display_name": "SWE",
            "enabled_for_sampling": True,
            "enabled_for_scoring": True,
            "sampling": {
                "sampling_count": 10,
                "dataset_range": [[0, 100]],
                "sampling_mode": "random",
            },
        },
        "TOKEN-EFFICIENCY": {
            "display_name": "TOKEN-EFFICIENCY",
            "kind": "derived",
            "derived_metric": "token_efficiency",
            "enabled_for_sampling": True,
            "enabled_for_scoring": False,
            "scoring": {"min_pairs": 10},
        },
    }
    store = StateStore(kv)

    sampling = await store.get_environments()
    scoring = await store.get_scoring_environments()
    derived = await store.get_derived_environments()
    display = await store.get_rank_display_environments()

    assert set(sampling.keys()) == {"SWE-INFINITE"}
    assert set(scoring.keys()) == {"SWE-INFINITE"}
    assert set(derived.keys()) == {"TOKEN-EFFICIENCY"}
    assert set(display.keys()) == {"SWE-INFINITE", "TOKEN-EFFICIENCY"}


@pytest.mark.asyncio
async def test_missing_sampling_gate_is_disabled():
    """Environment rows must explicitly opt into sampling."""
    kv = InMemoryConfigStore()
    kv.data["environments"] = {
        "MISSING_GATE": {
            "display_name": "Missing",
            "sampling": {
                "sampling_count": 5,
                "dataset_range": [[0, 50]],
                "sampling_mode": "random",
            },
        },
    }
    store = StateStore(kv)
    envs = await store.get_environments()
    assert "MISSING_GATE" not in envs
    scoring = await store.get_scoring_environments()
    assert "MISSING_GATE" not in scoring


@pytest.mark.asyncio
async def test_environments_accepts_legacy_window_config_shape():
    """Old seed used ``environments[env].window_config`` instead of
    ``.sampling``. Coercion should accept both."""
    kv = InMemoryConfigStore()
    kv.data["environments"] = {
        "OLD": {"display_name": "Old", "enabled_for_sampling": True,
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


@pytest.mark.asyncio
async def test_kv_adapter_signature_matches_dao():
    """Regression guard: ``SystemConfigKVAdapter`` must call the production
    ``SystemConfigDAO`` with the exact kwargs the DAO accepts. Tests using
    ``InMemoryConfigStore`` cannot catch a mismatch here. Build a stub DAO
    that records the call shape and assert it matches the real DAO's
    ``set_param`` / ``get_param_value`` / ``delete_param`` signatures."""
    import inspect

    from affine.database.dao.system_config import SystemConfigDAO
    from affine.src.scorer.window_state import SystemConfigKVAdapter

    real_set_sig = inspect.signature(SystemConfigDAO.set_param)
    real_get_sig = inspect.signature(SystemConfigDAO.get_param_value)
    real_del_sig = inspect.signature(SystemConfigDAO.delete_param)

    captured = {}

    class StubDAO:
        async def set_param(self, **kwargs):
            # bind against the REAL signature — raises TypeError on any mismatch
            real_set_sig.bind(self, **kwargs)
            captured["set"] = kwargs

        async def get_param_value(self, *args, **kwargs):
            real_get_sig.bind(self, *args, **kwargs)
            captured["get"] = (args, kwargs)
            return None

        async def delete_param(self, *args, **kwargs):
            real_del_sig.bind(self, *args, **kwargs)
            captured["del"] = (args, kwargs)
            return True

    adapter = SystemConfigKVAdapter(StubDAO(), updated_by="t")
    await adapter.set("champion", {"uid": 1})
    await adapter.get("champion")
    await adapter.delete("champion")
    assert captured["set"]["param_name"] == "champion"
    assert captured["set"]["param_value"] == {"uid": 1}
    assert captured["set"]["param_type"] == "dict"
    assert captured["set"]["updated_by"] == "t"


@pytest.mark.asyncio
async def test_kv_adapter_can_preserve_config_description():
    from affine.src.scorer.window_state import SystemConfigKVAdapter

    captured = {}

    class StubDAO:
        async def get_param(self, key):
            assert key == "gpu_autoscaler"
            return {"description": "GPU endpoint capacity policy"}

        async def set_param(self, **kwargs):
            captured.update(kwargs)

    adapter = SystemConfigKVAdapter(StubDAO(), updated_by="gpu-autoscaler")
    await adapter.set_preserving_metadata(
        "gpu_autoscaler",
        {"enabled": True, "endpoints": []},
    )

    assert captured["param_type"] == "dict"
    assert captured["description"] == "GPU endpoint capacity policy"
    assert captured["updated_by"] == "gpu-autoscaler"
