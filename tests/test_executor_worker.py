from affine.src.executor.worker import (
    _base_urls,
    _is_zero_score_error,
    _pick_url,
    _resolve_deployment_id,
)
from affine.src.executor.main import ExecutorManager
from affine.src.scorer.window_state import DeploymentRecord


def _in_flight_probe(v):
    v.value = 42


def test_executor_manager_ipc_handles_survive_spawn():
    """Regression: manager's mp.Value / BoundedSemaphore must round-trip
    through a spawn-context child. Spawn re-imports modules and pickles
    args; mp primitives have to carry their underlying handle correctly
    or the child gets a useless copy."""
    manager = ExecutorManager(["affine:ded-v2"])
    env = "affine:ded-v2"
    val = manager.in_flight_values[env]

    proc = manager.mp_ctx.Process(target=_in_flight_probe, args=(val,))
    proc.start()
    proc.join(timeout=10)

    assert proc.exitcode == 0
    assert val.value == 42

    # Global sem must also be usable from the parent — its acquire
    # accounting is what gates production dispatch.
    assert manager.global_sem.acquire(block=False)
    manager.global_sem.release()


def test_executor_manager_recovers_stale_slots_after_worker_death():
    from affine.src.executor.config import GLOBAL_DISPATCH_BUDGET

    manager = ExecutorManager(["ENV_A"])
    # Simulate a worker dying while holding two dispatch slots.
    assert manager.global_sem.acquire(block=False)
    assert manager.global_sem.acquire(block=False)
    manager.in_flight_values["ENV_A"].value = 2

    manager._recover_dead_worker_slots("ENV_A")

    assert manager.in_flight_values["ENV_A"].value == 0
    acquired = 0
    while manager.global_sem.acquire(block=False):
        acquired += 1
    assert acquired == GLOBAL_DISPATCH_BUDGET
    for _ in range(acquired):
        manager.global_sem.release()


def test_base_urls_prefers_deployments_and_dedupes():
    deployments = [
        DeploymentRecord(endpoint_name="a", deployment_id="da", base_url="http://a/v1"),
        DeploymentRecord(endpoint_name="b", deployment_id="db", base_url="http://b/v1"),
        DeploymentRecord(endpoint_name="a2", deployment_id="da2", base_url="http://a/v1"),
    ]

    assert _base_urls(deployments, fallback="http://fallback/v1") == [
        "http://a/v1",
        "http://b/v1",
    ]


def test_base_urls_uses_legacy_fallback_when_no_deployments():
    assert _base_urls([], fallback="http://single/v1") == ["http://single/v1"]


def test_pick_url_stably_shards_tasks_across_urls():
    urls = ["http://a/v1", "http://b/v1", "http://c/v1"]

    picks = [_pick_url(urls, task_id=t, index=i) for i, t in enumerate([10, 11, 12, 13])]

    assert picks == [
        urls[(10 + 0) % 3],
        urls[(11 + 1) % 3],
        urls[(12 + 2) % 3],
        urls[(13 + 3) % 3],
    ]


def test_is_zero_score_error_matches_context_overflow_patterns():
    cases = [
        "BadRequest: This model's maximum context length is 65536 tokens, but the input is longer than the model can handle",
        "This model supports context lengths of up to 65536 tokens",
        "HTTP 400: prompt exceeds the maximum allowed length",
        "Error: exceeds the maximum context length of 32000 tokens",
    ]
    for msg in cases:
        assert _is_zero_score_error(Exception(msg)), msg


def test_is_zero_score_error_rejects_transport_and_env_failures():
    cases = [
        "ReadError: connection reset by peer",
        "Remote execution failed: {'status': 'failed', 'error': 'worker crashed'}",
        "BackendError: Method call failed: Failed to call method 'evaluate'",
        "EnvironmentError: Method 'evaluate' failed on environment 'liveweb-pool-2'",
        "HTTP 503: service unavailable",
    ]
    for msg in cases:
        assert not _is_zero_score_error(Exception(msg)), msg


def test_is_zero_score_error_walks_exception_chain():
    # Affinetes wraps the original error several layers deep — the pattern
    # can live on any cause/context node.
    leaf = ValueError("prompt is longer than the model context window")
    middle = RuntimeError("Method call failed")
    middle.__cause__ = leaf
    top = RuntimeError("Method 'evaluate' failed on environment")
    top.__cause__ = middle

    assert _is_zero_score_error(top)


def test_is_zero_score_error_is_case_insensitive():
    assert _is_zero_score_error(Exception("EXCEEDS THE MAXIMUM CONTEXT LENGTH"))


def test_executor_does_not_persist_structured_error_results():
    import asyncio as _asyncio
    from affine.core.models import Result
    from affine.src.executor.worker import ExecutorWorker
    from affine.src.scorer.window_state import MinerSnapshot

    worker = ExecutorWorker(worker_id=0, env="NAVWORLD")

    class _EnvStub:
        async def evaluate(self, *, miner, task_id):
            return Result(
                env="NAVWORLD",
                score=0.0,
                latency_seconds=1.0,
                success=False,
                error="Model returned empty reply",
                task_id=task_id,
                extra={"error": "Model returned empty reply"},
            )

    class _SamplesStub:
        def __init__(self):
            self.persist_calls = []

        async def persist(self, **kwargs):
            self.persist_calls.append(kwargs)

    samples = _SamplesStub()
    worker._env_executor = _EnvStub()
    worker._samples = samples

    async def _drive():
        await worker._evaluate_and_persist_gated(
            miner=MinerSnapshot(
                uid=213,
                hotkey="hk",
                revision="rev",
                model="org/model",
            ),
            task_id=123,
            base_url="https://example/v1",
            refresh_block=10,
            miner_obj=object(),
        )

    _asyncio.run(_drive())

    assert samples.persist_calls == []


def test_executor_persists_structured_context_overflow_as_zero():
    import asyncio as _asyncio
    from affine.core.models import Result
    from affine.src.executor.worker import ExecutorWorker
    from affine.src.scorer.window_state import MinerSnapshot

    worker = ExecutorWorker(worker_id=0, env="SWE-INFINITE")

    class _EnvStub:
        async def evaluate(self, *, miner, task_id):
            return Result(
                env="SWE-INFINITE",
                score=0.0,
                latency_seconds=1.0,
                success=False,
                error="prompt exceeds the maximum context length",
                task_id=task_id,
                extra={"error": "prompt exceeds the maximum context length"},
            )

    class _SamplesStub:
        def __init__(self):
            self.persist_calls = []

        async def persist(self, **kwargs):
            self.persist_calls.append(kwargs)

    samples = _SamplesStub()
    worker._env_executor = _EnvStub()
    worker._samples = samples

    async def _drive():
        await worker._evaluate_and_persist_gated(
            miner=MinerSnapshot(
                uid=213,
                hotkey="hk",
                revision="rev",
                model="org/model",
            ),
            task_id=123,
            base_url="https://example/v1",
            refresh_block=10,
            miner_obj=object(),
        )

    _asyncio.run(_drive())

    assert len(samples.persist_calls) == 1
    persisted = samples.persist_calls[0]
    assert persisted["score"] == 0.0
    assert persisted["extra"]["zero_score_reason"] == "context_overflow"


def test_global_slot_acquire_release_round_trip():
    import asyncio as _asyncio
    import multiprocessing as _mp
    from affine.src.executor.worker import ExecutorWorker

    sem = _mp.get_context("spawn").BoundedSemaphore(2)
    worker = ExecutorWorker(worker_id=0, env="MEMORY", global_sem=sem)

    async def _drive():
        await worker._acquire_global_slot()
        await worker._acquire_global_slot()
        # Both slots taken — non-blocking acquire from a peer should fail.
        assert not sem.acquire(block=False)
        worker._release_global_slot()
        # One slot back — peer should be able to grab it now.
        assert sem.acquire(block=False)
        sem.release()
        worker._release_global_slot()

    _asyncio.run(_drive())


def test_global_slot_acquire_polls_until_released():
    """Coroutine should resume once a peer releases."""
    import asyncio as _asyncio
    import multiprocessing as _mp
    from affine.src.executor.worker import ExecutorWorker

    sem = _mp.get_context("spawn").BoundedSemaphore(1)
    sem.acquire(block=False)  # exhaust it
    worker = ExecutorWorker(worker_id=0, env="MEMORY", global_sem=sem)

    async def _drive():
        async def _waiter():
            await worker._acquire_global_slot()

        async def _releaser():
            await _asyncio.sleep(0.15)  # let waiter spin a few polls
            sem.release()

        await _asyncio.wait_for(
            _asyncio.gather(_waiter(), _releaser()),
            timeout=2.0,
        )
        worker._release_global_slot()  # clean up

    _asyncio.run(_drive())


def test_global_slot_helpers_noop_when_unwired():
    import asyncio as _asyncio
    from affine.src.executor.worker import ExecutorWorker

    worker = ExecutorWorker(worker_id=0, env="MEMORY", global_sem=None)

    async def _drive():
        await worker._acquire_global_slot()  # no-op, returns immediately
        worker._release_global_slot()  # no-op

    _asyncio.run(_drive())


class _SharedInt:
    def __init__(self, value: int):
        self.value = value


def test_dynamic_env_cap_defaults_to_max_concurrent_without_shared_value():
    from affine.src.executor.worker import ExecutorWorker

    worker = ExecutorWorker(worker_id=0, env="MEMORY", max_concurrent=17)
    assert worker._current_env_cap() == 17


def test_dynamic_env_cap_reads_shared_value_live():
    from affine.src.executor.worker import ExecutorWorker

    cap = _SharedInt(3)
    worker = ExecutorWorker(
        worker_id=0, env="LIVEWEB", max_concurrent=500, env_cap_value=cap,
    )

    assert worker._current_env_cap() == 3
    cap.value = 8
    assert worker._current_env_cap() == 8


def test_dynamic_dispatch_slot_waits_for_env_cap_room():
    import asyncio as _asyncio
    from affine.src.executor.worker import ExecutorWorker

    cap = _SharedInt(2)
    in_flight = _SharedInt(2)
    worker = ExecutorWorker(
        worker_id=0, env="LIVEWEB", env_cap_value=cap, in_flight_value=in_flight,
    )
    acquired = {"value": 0}

    async def _fake_global():
        acquired["value"] += 1

    worker._acquire_global_slot = _fake_global

    async def _drive():
        waiter = _asyncio.create_task(worker._acquire_dispatch_slot())
        await _asyncio.sleep(0.12)
        assert acquired["value"] == 0
        assert not waiter.done()

        in_flight.value = 1
        await _asyncio.wait_for(waiter, timeout=1.0)
        assert acquired["value"] == 1

    _asyncio.run(_drive())


def test_dynamic_dispatch_slot_releases_global_if_cap_drops_after_acquire():
    import asyncio as _asyncio
    from affine.src.executor.worker import ExecutorWorker

    cap = _SharedInt(2)
    in_flight = _SharedInt(1)
    worker = ExecutorWorker(
        worker_id=0, env="LIVEWEB", env_cap_value=cap, in_flight_value=in_flight,
    )
    acquired = {"value": 0}
    released = {"value": 0}

    async def _fake_global():
        acquired["value"] += 1
        cap.value = 1

    def _fake_release():
        released["value"] += 1
        in_flight.value = 0

    worker._acquire_global_slot = _fake_global
    worker._release_global_slot = _fake_release

    async def _drive():
        await _asyncio.wait_for(worker._acquire_dispatch_slot(), timeout=1.0)
        assert acquired["value"] == 2
        assert released["value"] == 1

    _asyncio.run(_drive())


def test_env_from_payload_ignores_static_max_concurrent():
    from affine.src.scorer.window_state import _env_from_payload

    cfg = _env_from_payload({
        "display_name": "LIVEWEB",
        "enabled_for_sampling": True,
        "sampling": {"sampling_count": 400, "max_concurrent": 100},
    })
    assert not hasattr(cfg, "max_concurrent")


def test_adaptive_caps_keep_probe_floor_for_backlogged_envs():
    from affine.src.executor.main import _compute_adaptive_env_caps

    caps = _compute_adaptive_env_caps(
        ["LIVEWEB", "MEMORY", "NAVWORLD", "SWE-INFINITE"],
        {
            env: {"target": 100, "done": 0, "running": 0, "delta": 0}
            for env in ["LIVEWEB", "MEMORY", "NAVWORLD", "SWE-INFINITE"]
        },
        {},
        global_budget=600,
    )

    assert all(caps[env] >= 1 for env in caps)
    assert sum(caps.values()) < 600


def test_adaptive_caps_start_backlogged_envs_at_fair_share():
    from affine.src.executor.main import _compute_adaptive_env_caps

    envs = ["LIVEWEB", "MEMORY", "NAVWORLD", "SWE-INFINITE"]
    caps = _compute_adaptive_env_caps(
        envs,
        {
            env: {"target": 1000, "done": 0, "running": 0, "delta": 0}
            for env in envs
        },
        {},
        global_budget=600,
    )

    assert caps == {env: 150 for env in envs}


def test_adaptive_caps_shift_capacity_from_underused_to_saturated_backlog():
    from affine.src.executor.main import _compute_adaptive_env_caps

    caps = _compute_adaptive_env_caps(
        ["LIVEWEB", "MEMORY", "NAVWORLD"],
        {
            "LIVEWEB": {"target": 400, "done": 20, "running": 5, "delta": 0},
            "MEMORY": {"target": 200, "done": 60, "running": 120, "delta": 3},
            "NAVWORLD": {"target": 400, "done": 100, "running": 120, "delta": 6},
        },
        {"LIVEWEB": 120, "MEMORY": 120, "NAVWORLD": 120},
        global_budget=300,
    )

    assert caps["LIVEWEB"] < caps["MEMORY"]
    assert caps["LIVEWEB"] < caps["NAVWORLD"]
    assert sum(caps.values()) <= 300


def test_adaptive_caps_prioritize_lagging_saturated_env():
    from affine.src.executor.main import _compute_adaptive_env_caps

    caps = _compute_adaptive_env_caps(
        ["MEMORY", "TERMINAL"],
        {
            "MEMORY": {"target": 500, "done": 100, "running": 150, "delta": 1},
            "TERMINAL": {"target": 500, "done": 450, "running": 150, "delta": 10},
        },
        {"MEMORY": 150, "TERMINAL": 150},
        global_budget=300,
    )

    assert caps["MEMORY"] > caps["TERMINAL"]
    assert sum(caps.values()) <= 300


def test_adaptive_caps_ramp_zero_progress_saturated_env_to_fair_share():
    from affine.src.executor.main import _compute_adaptive_env_caps

    caps = _compute_adaptive_env_caps(
        ["LIVEWEB", "TERMINAL"],
        {
            "LIVEWEB": {"target": 500, "done": 50, "running": 180, "delta": 0},
            "TERMINAL": {"target": 500, "done": 250, "running": 120, "delta": 5},
        },
        {"LIVEWEB": 120, "TERMINAL": 120},
        global_budget=300,
    )

    assert caps["LIVEWEB"] == 150
    assert caps["TERMINAL"] == 150


def test_adaptive_caps_decay_inflated_zero_progress_saturated_env_to_fair_share():
    from affine.src.executor.main import _compute_adaptive_env_caps

    caps = _compute_adaptive_env_caps(
        ["LIVEWEB", "TERMINAL"],
        {
            "LIVEWEB": {"target": 500, "done": 50, "running": 240, "delta": 0},
            "TERMINAL": {"target": 500, "done": 250, "running": 120, "delta": 5},
        },
        {"LIVEWEB": 240, "TERMINAL": 120},
        global_budget=300,
    )

    assert caps["LIVEWEB"] == 150
    assert caps["TERMINAL"] >= caps["LIVEWEB"]


def test_adaptive_caps_release_completed_env_capacity():
    from affine.src.executor.main import _compute_adaptive_env_caps

    caps = _compute_adaptive_env_caps(
        ["MEMORY", "TERMINAL"],
        {
            "MEMORY": {"target": 200, "done": 200, "running": 0, "delta": 0},
            "TERMINAL": {"target": 500, "done": 50, "running": 150, "delta": 5},
        },
        {"MEMORY": 150, "TERMINAL": 150},
        global_budget=300,
    )

    assert 1 < caps["MEMORY"] < caps["TERMINAL"]
    assert caps["TERMINAL"] > 150


def test_adaptive_caps_do_not_hand_entire_budget_to_first_fast_env():
    from affine.src.executor.main import _compute_adaptive_env_caps

    caps = _compute_adaptive_env_caps(
        ["LIVEWEB", "MEMORY", "NAVWORLD", "SWE-INFINITE", "TERMINAL"],
        {
            "LIVEWEB": {"target": 841, "done": 516, "running": 25, "delta": 0},
            "MEMORY": {"target": 421, "done": 421, "running": 0, "delta": 0},
            "NAVWORLD": {"target": 841, "done": 841, "running": 0, "delta": 0},
            "SWE-INFINITE": {"target": 630, "done": 410, "running": 25, "delta": 1},
            "TERMINAL": {"target": 630, "done": 625, "running": 5, "delta": 0},
        },
        {
            "LIVEWEB": 25,
            "MEMORY": 120,
            "NAVWORLD": 120,
            "SWE-INFINITE": 25,
            "TERMINAL": 25,
        },
        global_budget=600,
    )

    assert caps["LIVEWEB"] > 25
    assert caps["SWE-INFINITE"] <= 200
    assert caps["LIVEWEB"] < caps["SWE-INFINITE"]


def test_adaptive_caps_use_idle_budget_for_saturated_slow_env_after_fair_share():
    from affine.src.executor.main import _compute_adaptive_env_caps

    caps = _compute_adaptive_env_caps(
        ["LIVEWEB", "SWE-INFINITE"],
        {
            "LIVEWEB": {"target": 800, "done": 100, "running": 150, "delta": 0},
            "SWE-INFINITE": {"target": 300, "done": 100, "running": 150, "delta": 2},
        },
        {"LIVEWEB": 150, "SWE-INFINITE": 150},
        global_budget=400,
    )

    assert caps["LIVEWEB"] > 150
    assert caps["LIVEWEB"] <= 250
    assert sum(caps.values()) <= 400


def test_sampling_count_for_env_reads_current_config():
    from affine.src.executor.main import _sampling_count_for_env

    envs = {
        "ENV_A": {
            "sampling": {
                "sampling_count": 400,
            },
        },
    }

    assert _sampling_count_for_env(envs, "ENV_A", 441) == 400
    assert _sampling_count_for_env(envs, "MISSING", 441) == 441
    envs["ENV_A"]["sampling"]["sampling_count"] = 0
    assert _sampling_count_for_env(envs, "ENV_A", 441) == 0


def test_sampling_priority_for_env_reads_current_config():
    from affine.src.executor.main import _sampling_priority_for_env

    envs = {
        "ENV_A": {"sampling": {"priority": 2}},
        "ENV_B": {"sampling": {"priority": "1"}},
        "ENV_C": {"sampling": {}},
        "ENV_D": {"sampling": {"priority": None}},
        "ENV_E": {"sampling": "not-a-dict"},
        "ENV_F": "not-a-dict",
    }

    assert _sampling_priority_for_env(envs, "ENV_A") == 2
    assert _sampling_priority_for_env(envs, "ENV_B") == 1
    assert _sampling_priority_for_env(envs, "ENV_C") == 0
    assert _sampling_priority_for_env(envs, "ENV_D") == 0
    assert _sampling_priority_for_env(envs, "ENV_E") == 0
    assert _sampling_priority_for_env(envs, "ENV_F") == 0
    assert _sampling_priority_for_env(envs, "MISSING") == 0
    assert _sampling_priority_for_env("garbage", "ENV_A") == 0
    assert _sampling_priority_for_env(envs, "MISSING", default=5) == 5


def test_adaptive_caps_priority_unset_is_bit_identical_to_flat():
    """priorities=None / single-tier path must not perturb the existing
    allocator — regression guard for the refactor.
    """
    from affine.src.executor.main import _compute_adaptive_env_caps

    envs = ["LIVEWEB", "MEMORY", "NAVWORLD", "SWE-INFINITE", "TERMINAL"]
    stats = {
        "LIVEWEB": {"target": 841, "done": 516, "running": 25, "delta": 0},
        "MEMORY": {"target": 421, "done": 421, "running": 0, "delta": 0},
        "NAVWORLD": {"target": 841, "done": 841, "running": 0, "delta": 0},
        "SWE-INFINITE": {"target": 630, "done": 410, "running": 25, "delta": 1},
        "TERMINAL": {"target": 630, "done": 625, "running": 5, "delta": 0},
    }
    prev = {
        "LIVEWEB": 25, "MEMORY": 120, "NAVWORLD": 120,
        "SWE-INFINITE": 25, "TERMINAL": 25,
    }
    flat = _compute_adaptive_env_caps(envs, stats, prev, global_budget=600)
    untiered = _compute_adaptive_env_caps(
        envs, stats, prev, global_budget=600,
        priorities={env: 0 for env in envs},
    )
    assert flat == untiered


def test_adaptive_caps_priority_tier_gets_dominant_share():
    """High-priority active envs claim most of the budget; low-priority
    envs are throttled to just enough to keep probing.
    """
    from affine.src.executor.main import _compute_adaptive_env_caps

    envs = ["LIVEWEB", "MEMORY", "NAVWORLD", "SWE-INFINITE", "TERMINAL"]
    stats = {env: {"target": 1000, "done": 0, "running": 0, "delta": 0} for env in envs}
    caps = _compute_adaptive_env_caps(
        envs, stats, {}, global_budget=600,
        priorities={
            "MEMORY": 1, "SWE-INFINITE": 1,
            "LIVEWEB": 0, "NAVWORLD": 0, "TERMINAL": 0,
        },
    )

    top = caps["MEMORY"] + caps["SWE-INFINITE"]
    bottom = caps["LIVEWEB"] + caps["NAVWORLD"] + caps["TERMINAL"]
    assert top > bottom
    assert caps["MEMORY"] > caps["LIVEWEB"]
    assert caps["SWE-INFINITE"] > caps["TERMINAL"]
    # Every active env still gets a non-zero probe (no starvation).
    assert all(caps[env] >= 1 for env in envs)


def test_adaptive_caps_priority_cascade_when_top_demand_low():
    """When the top tier's remaining work cannot consume its budget,
    the unused slots flow down to the next tier instead of staying
    parked at high priority.
    """
    from affine.src.executor.main import _compute_adaptive_env_caps

    envs = ["MEMORY", "SWE-INFINITE", "LIVEWEB", "TERMINAL"]
    stats = {
        # Tier 1: tiny remaining each.
        "MEMORY": {"target": 100, "done": 95, "running": 5, "delta": 1},
        "SWE-INFINITE": {"target": 100, "done": 95, "running": 5, "delta": 1},
        # Tier 0: big backlog.
        "LIVEWEB": {"target": 1000, "done": 0, "running": 0, "delta": 0},
        "TERMINAL": {"target": 1000, "done": 0, "running": 0, "delta": 0},
    }
    caps = _compute_adaptive_env_caps(
        envs, stats, {}, global_budget=600,
        priorities={
            "MEMORY": 1, "SWE-INFINITE": 1, "LIVEWEB": 0, "TERMINAL": 0,
        },
    )

    # Top tier can't use more than its remaining work.
    assert caps["MEMORY"] <= 100
    assert caps["SWE-INFINITE"] <= 100
    # Cascaded budget reached the bottom tier.
    assert caps["LIVEWEB"] >= 100
    assert caps["TERMINAL"] >= 100


def test_adaptive_caps_priority_lower_tier_keeps_probe_floor_under_saturation():
    """Top tier with infinite-looking demand cannot starve a backlogged
    bottom-tier env; bottom tier still receives at least a probe floor
    (positive cap) so it can detect a stall or make slow progress.
    """
    from affine.src.executor.main import _compute_adaptive_env_caps

    envs = ["MEMORY", "SWE-INFINITE", "LIVEWEB"]
    stats = {
        "MEMORY": {"target": 10_000, "done": 0, "running": 0, "delta": 0},
        "SWE-INFINITE": {"target": 10_000, "done": 0, "running": 0, "delta": 0},
        "LIVEWEB": {"target": 10_000, "done": 0, "running": 0, "delta": 0},
    }
    caps = _compute_adaptive_env_caps(
        envs, stats, {}, global_budget=600,
        priorities={"MEMORY": 1, "SWE-INFINITE": 1, "LIVEWEB": 0},
    )

    assert caps["LIVEWEB"] >= 1
    assert caps["MEMORY"] > caps["LIVEWEB"]
    assert caps["SWE-INFINITE"] > caps["LIVEWEB"]


def test_adaptive_caps_priority_top_tier_shares_internally_by_pressure():
    """Two envs at the same priority tier still resolve by the same
    pressure-weighted logic the un-tiered allocator uses — a saturated
    laggard within the tier wins capacity from a near-done peer.
    """
    from affine.src.executor.main import _compute_adaptive_env_caps

    envs = ["MEMORY", "SWE-INFINITE", "LIVEWEB"]
    stats = {
        # Same tier; MEMORY is far behind, SWE near done.
        "MEMORY": {"target": 500, "done": 100, "running": 150, "delta": 1},
        "SWE-INFINITE": {"target": 500, "done": 450, "running": 150, "delta": 10},
        # Low tier idle.
        "LIVEWEB": {"target": 0, "done": 0, "running": 0, "delta": 0},
    }
    caps = _compute_adaptive_env_caps(
        envs, stats, {"MEMORY": 150, "SWE-INFINITE": 150, "LIVEWEB": 5},
        global_budget=600,
        priorities={"MEMORY": 1, "SWE-INFINITE": 1, "LIVEWEB": 0},
    )

    assert caps["MEMORY"] > caps["SWE-INFINITE"]


def test_adaptive_caps_priority_no_active_envs_returns_positive_caps():
    """Edge: every env idle (no work anywhere, e.g. just after a battle
    ended and before the next picks up). Each tier still receives a
    placeholder cap from its pool — operationally inert (nothing is
    running) but a fresh battle can ramp immediately without a
    one-tick zero floor.
    """
    from affine.src.executor.main import _compute_adaptive_env_caps

    envs = ["MEMORY", "SWE-INFINITE", "LIVEWEB"]
    stats = {env: {"target": 0, "done": 0, "running": 0, "delta": 0} for env in envs}
    caps = _compute_adaptive_env_caps(
        envs, stats, {}, global_budget=600,
        priorities={"MEMORY": 1, "SWE-INFINITE": 1, "LIVEWEB": 0},
    )

    assert set(caps.keys()) == set(envs)
    assert all(caps[env] >= 1 for env in envs)


def test_adaptive_caps_priority_inactive_top_tier_does_not_block_cascade():
    """If every top-tier env is inactive (nothing to do), the entire
    global budget cascades to lower tiers.
    """
    from affine.src.executor.main import _compute_adaptive_env_caps

    envs = ["MEMORY", "SWE-INFINITE", "LIVEWEB", "TERMINAL"]
    stats = {
        "MEMORY": {"target": 0, "done": 0, "running": 0, "delta": 0},
        "SWE-INFINITE": {"target": 0, "done": 0, "running": 0, "delta": 0},
        "LIVEWEB": {"target": 1000, "done": 0, "running": 0, "delta": 0},
        "TERMINAL": {"target": 1000, "done": 0, "running": 0, "delta": 0},
    }
    caps = _compute_adaptive_env_caps(
        envs, stats, {}, global_budget=600,
        priorities={
            "MEMORY": 1, "SWE-INFINITE": 1, "LIVEWEB": 0, "TERMINAL": 0,
        },
    )

    assert caps["LIVEWEB"] >= 150
    assert caps["TERMINAL"] >= 150
    assert sum(caps.values()) <= 600 + caps["MEMORY"] + caps["SWE-INFINITE"]


def test_status_target_uses_challenger_sampling_count_not_buffered_pool():
    import asyncio as _asyncio

    manager = ExecutorManager(["ENV_A"])
    ids = list(range(441))

    class _SC:
        async def get_param_value(self, name, default=None):
            values = {
                "current_task_ids": {
                    "task_ids": {"ENV_A": ids},
                    "refreshed_at_block": 123,
                },
                "champion": {"hotkey": "champ_hk", "revision": "champ_rev"},
                "current_battle": {
                    "challenger": {
                        "hotkey": "chal_hk",
                        "revision": "chal_rev",
                    },
                },
                "environments": {
                    "ENV_A": {"sampling": {"sampling_count": 400}},
                },
            }
            return values.get(name, default)

    class _Samples:
        async def count_samples_for_tasks(
            self, hotkey, revision, env, task_ids, *, refresh_block
        ):
            if hotkey == "champ_hk":
                return 441
            if hotkey == "chal_hk":
                return 400
            return 0

    _asyncio.run(manager._emit_status_line(_SC(), _Samples()))

    assert manager._last_done["ENV_A"] == 400
    assert manager.env_cap_values["ENV_A"].value == 600


def test_status_label_includes_uid_for_each_role(caplog):
    """STATUS label must carry the UID so operators don't read role as
    identity. ``[champion U213]`` / ``[challenger U228]`` instead of
    ambiguous ``[champion]`` / ``[challenger]``."""
    import asyncio as _asyncio
    import logging

    manager = ExecutorManager(["ENV_A"])
    ids = list(range(10))

    class _SC:
        async def get_param_value(self, name, default=None):
            return {
                "current_task_ids": {"task_ids": {"ENV_A": ids}, "refreshed_at_block": 1},
                "champion": {"uid": 213, "hotkey": "champ_hk", "revision": "champ_rev"},
                "current_battle": {"challenger": {
                    "uid": 228, "hotkey": "chal_hk", "revision": "chal_rev",
                }},
                "environments": {"ENV_A": {"sampling": {"sampling_count": 10}}},
            }.get(name, default)

    class _Samples:
        async def count_samples_for_tasks(self, hotkey, revision, env, task_ids, *, refresh_block):
            return 5

    with caplog.at_level(logging.INFO, logger="affine"):
        _asyncio.run(manager._emit_status_line(_SC(), _Samples()))

    assert any("[challenger U228]" in r.message for r in caplog.records), (
        "active subject is the challenger in a live battle — STATUS label must say [challenger U228]"
    )

    # Now clear the battle: label flips to [champion U213].
    caplog.clear()

    class _SC2:
        async def get_param_value(self, name, default=None):
            return {
                "current_task_ids": {"task_ids": {"ENV_A": ids}, "refreshed_at_block": 1},
                "champion": {"uid": 213, "hotkey": "champ_hk", "revision": "champ_rev"},
                "current_battle": {},
                "environments": {"ENV_A": {"sampling": {"sampling_count": 10}}},
            }.get(name, default)

    with caplog.at_level(logging.INFO, logger="affine"):
        _asyncio.run(manager._emit_status_line(_SC2(), _Samples()))

    assert any("[champion U213]" in r.message for r in caplog.records), (
        "no battle in flight — STATUS label must say [champion U213]"
    )


def test_status_delta_zeroes_on_subject_change(caplog):
    """Between two STATUS prints the active subject can change (battle
    decided → role flips). The per-env delta is otherwise computed as
    ``current_subject_count - previous_subject_count`` — a meaningless
    cross-miner subtraction. The first frame after a subject change
    must report ``+0`` instead of that phantom delta."""
    import asyncio as _asyncio
    import logging

    manager = ExecutorManager(["ENV_A"])
    ids = list(range(10))

    sample_counts = {("chal_hk", "chal_rev"): 7, ("champ_hk", "champ_rev"): 9}

    def _make_sc(battle):
        class _SC:
            async def get_param_value(_self, name, default=None):
                return {
                    "current_task_ids": {"task_ids": {"ENV_A": ids},
                                         "refreshed_at_block": 1},
                    "champion": {"uid": 213, "hotkey": "champ_hk", "revision": "champ_rev"},
                    "current_battle": battle,
                    "environments": {"ENV_A": {"sampling": {"sampling_count": 10}}},
                }.get(name, default)
        return _SC()

    class _Samples:
        async def count_samples_for_tasks(self, hotkey, revision, env, task_ids, *, refresh_block):
            return sample_counts.get((hotkey, revision), 0)

    # First print: challenger active, done=7. Baseline (no delta shown).
    _asyncio.run(manager._emit_status_line(
        _make_sc({"challenger": {"uid": 228, "hotkey": "chal_hk", "revision": "chal_rev"}}),
        _Samples(),
    ))
    assert manager._last_subject_key == ("challenger", "chal_hk", "chal_rev")
    assert manager._last_done["ENV_A"] == 7

    # Second print: battle cleared, champion is now the active subject.
    # Done count flips to 9 (champion's count). Without the guard, delta
    # would be 9-7=+2 (false). With the guard, delta=0.
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="affine"):
        _asyncio.run(manager._emit_status_line(_make_sc({}), _Samples()))

    log_text = " ".join(r.message for r in caplog.records)
    assert "[champion U213]" in log_text, "subject flipped to champion"
    assert "+0 " in log_text or "+0\n" in log_text or log_text.endswith("+0"), (
        f"delta on subject change must be 0, got: {log_text!r}"
    )
    assert "+2" not in log_text, "phantom cross-subject delta must be suppressed"


def test_dispatch_proceeds_when_only_battle_has_url():
    """Regression: under SSH single-instance providers, ``_start_battle``
    clears the champion's deployment so the host can serve the
    challenger. Without this guard, ``_dispatch_new`` exits early
    (no champion URL → return 0) and the challenger never gets sampled,
    so ``_battle_overlap_ready`` never satisfies and the battle stalls.
    """
    import asyncio as _asyncio
    from affine.src.executor.worker import ExecutorWorker
    from affine.src.scorer.window_state import (
        BattleRecord, ChampionRecord, DeploymentRecord, EnvConfig,
        MinerSnapshot, TaskIdState,
    )

    worker = ExecutorWorker(worker_id=0, env="ENV_A")
    worker.warmup_sec = 0  # skip — not testing run-loop

    class _StateStub:
        async def get_task_state(self):
            return TaskIdState(
                task_ids={"ENV_A": [1, 2, 3]}, refreshed_at_block=0,
            )

        async def get_environments(self):
            return {
                "ENV_A": EnvConfig(
                    display_name="ENV_A", enabled_for_sampling=True,
                    sampling_count=3, dataset_range=[[0, 100]],
                ),
            }

        async def get_champion(self):
            # Champion exists but has no live deployment — exactly the
            # state after single-instance ``_start_battle`` clears it.
            return ChampionRecord(
                uid=1, hotkey="champ_hk", revision="champ_rev",
                model="org/m1", deployment_id=None, base_url=None,
                deployments=[], since_block=0,
            )

        async def get_battle(self):
            return BattleRecord(
                challenger=MinerSnapshot(
                    uid=2, hotkey="chal_hk", revision="chal_rev",
                    model="org/m2",
                ),
                deployment_id="wrk-bat",
                base_url="https://t/wrk-bat",
                started_at_block=0,
                deployments=[
                    DeploymentRecord(
                        endpoint_name="b300",
                        deployment_id="wrk-bat",
                        base_url="https://t/wrk-bat",
                    ),
                ],
            )

        async def get_predeployed_challengers(self):
            return []

    class _SamplesStub:
        # Champion has already sampled all 3 task_ids — without that the
        # new challenger-overlap rule (challenger only attempts task_ids
        # in champion_done) would dispatch 0 even though the BUG-1 fix
        # this test guards is about the no-champion-URL path.
        def __init__(self, champ_done):
            self._champ_done = champ_done

        async def read_scores_for_tasks(
            self, hotkey, revision, env, task_ids, *, refresh_block,
        ):
            if hotkey == "champ_hk":
                return {int(t): 0.5 for t in task_ids if int(t) in self._champ_done}
            return {}

    dispatched: list = []

    async def _capture_dispatch_one(*, miner, task_id, base_url, **_kwargs):
        dispatched.append((miner.uid, task_id, base_url))

    worker._state = _StateStub()
    worker._samples = _SamplesStub(champ_done={1, 2, 3})
    worker._dispatch_one = _capture_dispatch_one

    async def _drive():
        in_flight_keys: set = set()
        in_flight_tasks: set = set()
        n = await worker._dispatch_new(in_flight_keys, in_flight_tasks)
        # All pending in-flight task wrappers are fire-and-forget — give
        # them a tick to record into ``dispatched``.
        await _asyncio.sleep(0)
        return n

    n = _asyncio.run(_drive())

    # No champion candidates (no URL), but challenger candidates fire.
    assert n == 3, f"expected 3 challenger dispatches, got {n}"
    assert all(uid == 2 for uid, _, _ in dispatched), dispatched
    assert all(url == "https://t/wrk-bat" for _, _, url in dispatched), dispatched


def test_dispatch_skips_when_neither_has_url():
    """If neither champion nor battle has a serving URL there's
    genuinely nothing to dispatch — bail out fast."""
    import asyncio as _asyncio
    from affine.src.executor.worker import ExecutorWorker
    from affine.src.scorer.window_state import (
        ChampionRecord, EnvConfig, TaskIdState,
    )

    worker = ExecutorWorker(worker_id=0, env="ENV_A")
    worker.warmup_sec = 0

    class _StateStub:
        async def get_task_state(self):
            return TaskIdState(
                task_ids={"ENV_A": [1, 2, 3]}, refreshed_at_block=0,
            )

        async def get_environments(self):
            return {
                "ENV_A": EnvConfig(
                    display_name="ENV_A", enabled_for_sampling=True,
                    sampling_count=3, dataset_range=[[0, 100]],
                ),
            }

        async def get_champion(self):
            return ChampionRecord(
                uid=1, hotkey="champ_hk", revision="champ_rev",
                model="org/m1", deployment_id=None, base_url=None,
                deployments=[], since_block=0,
            )

        async def get_battle(self):
            return None

        async def get_predeployed_challengers(self):
            return []

    class _SamplesStub:
        async def has_sample(self, *a, **kw):
            return False

        async def count_samples_for_tasks(self, *a, **kw):
            return 0

        async def read_scores_for_tasks(self, *a, **kw):
            return {}

    worker._state = _StateStub()
    worker._samples = _SamplesStub()

    async def _drive():
        return await worker._dispatch_new(set(), set())

    assert _asyncio.run(_drive()) == 0


# ---- deployment_id drift validation -----------------------------------------
#
# Bug context: during sglang model swap, the executor's in-flight
# ``evaluate`` calls for the old miner kept hitting the same val:8100
# tunnel — but the host behind that tunnel was now serving the new
# miner's model. The results got persisted under the old miner's
# (hotkey, revision), contaminating ~916 sample_results rows for uid=32.
#
# Fix: capture ``deployment_id`` at dispatch and re-validate before
# persist. If the captured token is no longer current for this miner
# (in either champion or challenger role), drop the row.


def test_resolve_deployment_id_picks_matching_url():
    deps = [
        DeploymentRecord(endpoint_name="a", deployment_id="da", base_url="http://a/v1"),
        DeploymentRecord(endpoint_name="b", deployment_id="db", base_url="http://b/v1"),
    ]
    assert _resolve_deployment_id("http://b/v1", deps, legacy_id="legacy") == "db"


def test_resolve_deployment_id_falls_back_to_legacy_when_deployments_empty():
    assert _resolve_deployment_id(
        "http://single/v1", deployments=[], legacy_id="wrk-legacy",
    ) == "wrk-legacy"


def test_resolve_deployment_id_returns_none_when_unmatched_and_no_legacy():
    deps = [DeploymentRecord(endpoint_name="a", deployment_id="da", base_url="http://a/v1")]
    assert _resolve_deployment_id("http://other/v1", deps, legacy_id=None) is None


class _DeploymentDriftFixture:
    """Stubs that let a single ``_evaluate_and_persist_gated`` call run
    end-to-end against a controllable champion / battle / samples.

    The state objects are stored as attributes so a test can rewrite
    them between dispatch (capture) and persist (re-read) to simulate a
    mid-evaluate model swap."""

    def __init__(self, *, env="ENV_A"):
        from affine.core.models import Result
        from affine.src.executor.worker import ExecutorWorker

        self.env = env
        self.worker = ExecutorWorker(worker_id=0, env=env)
        self.worker.warmup_sec = 0
        self.champion = None
        self.battle = None
        self.persist_calls = []
        self.score = 0.7
        self._Result = Result

        fixture = self

        class _StateStub:
            async def get_champion(self):
                return fixture.champion

            async def get_battle(self):
                return fixture.battle

            async def get_predeployed_challengers(self):
                return list(getattr(fixture, "predeployed", []))

        class _SamplesStub:
            async def persist(self, **kwargs):
                fixture.persist_calls.append(kwargs)

        class _EnvStub:
            async def evaluate(self, *, miner, task_id):
                return fixture._Result(
                    env=fixture.env,
                    score=fixture.score,
                    latency_seconds=0.1,
                    success=True,
                    task_id=task_id,
                )

        self.worker._state = _StateStub()
        self.worker._samples = _SamplesStub()
        self.worker._env_executor = _EnvStub()

    async def run(self, *, miner, expected_deployment_id, task_id=10):
        await self.worker._evaluate_and_persist_gated(
            miner=miner, task_id=task_id,
            base_url="https://t/wrk-X",
            refresh_block=0,
            miner_obj=object(),
            expected_deployment_id=expected_deployment_id,
        )


def test_persists_when_deployment_id_unchanged():
    """Happy path: champion's deployment hasn't moved → persist."""
    import asyncio as _asyncio

    from affine.src.scorer.window_state import ChampionRecord, MinerSnapshot

    fx = _DeploymentDriftFixture()
    fx.champion = ChampionRecord(
        uid=1, hotkey="hk", revision="rev", model="org/m",
        deployment_id="wrk-current", base_url="https://t/wrk-X",
        deployments=[], since_block=0,
    )
    miner = MinerSnapshot(uid=1, hotkey="hk", revision="rev", model="org/m")

    _asyncio.run(fx.run(miner=miner, expected_deployment_id="wrk-current"))

    assert len(fx.persist_calls) == 1
    assert fx.worker.metrics.tasks_dropped_drift == 0
    assert fx.worker.metrics.tasks_succeeded == 1


def test_drops_when_champion_deployment_swapped_mid_evaluate():
    """sglang model swap: evaluate returns a result but the deployment
    that produced it is no longer the champion's serving deployment.
    Persisting would attribute new-model output to the old miner."""
    import asyncio as _asyncio

    from affine.src.scorer.window_state import ChampionRecord, MinerSnapshot

    fx = _DeploymentDriftFixture()
    # State at persist time: champion still same identity, but on a new
    # deployment_id (the old one was torn down during swap).
    fx.champion = ChampionRecord(
        uid=1, hotkey="hk", revision="rev", model="org/m",
        deployment_id="wrk-NEW", base_url="https://t/wrk-NEW",
        deployments=[], since_block=0,
    )
    miner = MinerSnapshot(uid=1, hotkey="hk", revision="rev", model="org/m")

    _asyncio.run(fx.run(miner=miner, expected_deployment_id="wrk-OLD"))

    assert fx.persist_calls == []
    assert fx.worker.metrics.tasks_dropped_drift == 1
    # Drift drops are not counted as success or failure — the row will
    # be re-attempted next dispatch tick against the new deployment.
    assert fx.worker.metrics.tasks_succeeded == 0
    assert fx.worker.metrics.tasks_failed == 0


def test_persists_when_challenger_wins_and_deployment_id_transferred():
    """Challenger wins: the challenger's deployment_id is transferred
    into the new champion record (same hotkey/revision). The captured
    token still matches the new champion's deployment_id, so an
    in-flight challenger task that completes after the transfer must
    still persist — those samples are valid for the new champion."""
    import asyncio as _asyncio

    from affine.src.scorer.window_state import ChampionRecord, MinerSnapshot

    fx = _DeploymentDriftFixture()
    # New champion = the former challenger. battle is now cleared.
    fx.champion = ChampionRecord(
        uid=2, hotkey="chal_hk", revision="chal_rev", model="org/m2",
        deployment_id="wrk-CHAL", base_url="https://t/wrk-CHAL",
        deployments=[], since_block=42,
    )
    fx.battle = None
    miner = MinerSnapshot(uid=2, hotkey="chal_hk", revision="chal_rev", model="org/m2")

    _asyncio.run(fx.run(miner=miner, expected_deployment_id="wrk-CHAL"))

    assert len(fx.persist_calls) == 1
    assert fx.worker.metrics.tasks_dropped_drift == 0


def test_drops_when_miner_no_longer_subject():
    """Miner has dropped out of both champion and challenger roles
    (e.g. battle ended with old champion losing, or scheduler purged
    them). No subject role → drop."""
    import asyncio as _asyncio

    from affine.src.scorer.window_state import ChampionRecord, MinerSnapshot

    fx = _DeploymentDriftFixture()
    fx.champion = ChampionRecord(
        uid=99, hotkey="other_hk", revision="other_rev", model="org/o",
        deployment_id="wrk-OTHER", base_url="https://t/wrk-OTHER",
        deployments=[], since_block=0,
    )
    fx.battle = None
    miner = MinerSnapshot(uid=1, hotkey="lost_hk", revision="lost_rev", model="org/m")

    _asyncio.run(fx.run(miner=miner, expected_deployment_id="wrk-LOST"))

    assert fx.persist_calls == []
    assert fx.worker.metrics.tasks_dropped_drift == 1


def test_skips_validation_when_no_token_was_captured():
    """Backward-compat: if dispatch had nothing to capture (legacy record
    with neither ``deployments`` nor ``deployment_id``), we can't detect
    drift, so behave as before — persist."""
    import asyncio as _asyncio

    from affine.src.scorer.window_state import ChampionRecord, MinerSnapshot

    fx = _DeploymentDriftFixture()
    fx.champion = ChampionRecord(
        uid=1, hotkey="hk", revision="rev", model="org/m",
        deployment_id=None, base_url=None, deployments=[], since_block=0,
    )
    miner = MinerSnapshot(uid=1, hotkey="hk", revision="rev", model="org/m")

    _asyncio.run(fx.run(miner=miner, expected_deployment_id=None))

    assert len(fx.persist_calls) == 1
    assert fx.worker.metrics.tasks_dropped_drift == 0


# ---- challenger overlap + champion completion threshold --------------------
#
# See affine/src/scorer/sampling_thresholds.py and the plan in
# /home/claudeuser/.claude-aly2/plans/zippy-foraging-clarke.md.
#
# Champion drains the pool until ``len(champ_done) ≥
# champion_completion_threshold(sampling_count)`` (95% of pool); the
# remaining 5% is the deliberately-abandoned long tail. Challenger
# dispatches every task_id champion has already sampled (NOT the raw
# pool, NOT capped at sampling_count) and early-stops once the
# (champion ∩ challenger) overlap reaches sampling_count.


def _make_overlap_worker(env="ENV_A"):
    """Worker + state stubs for the overlap/threshold tests below.

    The state's task_state, env_cfg, champion, battle, and the samples
    stub's per-miner score dicts are exposed as attributes so each test
    can mutate them between dispatches."""
    import asyncio as _asyncio

    from affine.src.executor.worker import ExecutorWorker
    from affine.src.scorer.window_state import (
        BattleRecord, ChampionRecord, DeploymentRecord, EnvConfig,
        MinerSnapshot, TaskIdState,
    )

    worker = ExecutorWorker(worker_id=0, env=env)
    worker.warmup_sec = 0

    fx = type("Fx", (), {})()
    fx.worker = worker
    # 220 = pool size for sampling_count=200 (1.1× buffer with FP rounds
    # to 221 in production code; we use 220 in tests for arithmetic
    # cleanliness — the threshold helper accepts whatever pool we feed).
    fx.task_ids = list(range(220))
    fx.sampling_count = 200
    fx.champ_scores = {}      # tid → score
    fx.chal_scores = {}       # tid → score

    class _StateStub:
        async def get_task_state(self):
            return TaskIdState(
                task_ids={env: list(fx.task_ids)},
                refreshed_at_block=0,
            )

        async def get_environments(self):
            return {
                env: EnvConfig(
                    display_name=env, enabled_for_sampling=True,
                    sampling_count=fx.sampling_count,
                    dataset_range=[[0, 1000]],
                ),
            }

        async def get_champion(self):
            return ChampionRecord(
                uid=1, hotkey="champ_hk", revision="champ_rev",
                model="org/m1", deployment_id="wrk-champ",
                base_url="https://t/wrk-champ", deployments=[
                    DeploymentRecord(endpoint_name="b300",
                                     deployment_id="wrk-champ",
                                     base_url="https://t/wrk-champ"),
                ],
                since_block=0,
            )

        async def get_battle(self):
            return BattleRecord(
                challenger=MinerSnapshot(
                    uid=2, hotkey="chal_hk", revision="chal_rev",
                    model="org/m2",
                ),
                deployment_id="wrk-chal",
                base_url="https://t/wrk-chal",
                started_at_block=0,
                deployments=[
                    DeploymentRecord(endpoint_name="b300",
                                     deployment_id="wrk-chal",
                                     base_url="https://t/wrk-chal"),
                ],
            )

        async def get_predeployed_challengers(self):
            return list(getattr(fx, "predeployed", []))

    class _SamplesStub:
        async def read_scores_for_tasks(
            self, hotkey, revision, _env, task_ids, *, refresh_block,
        ):
            wanted = {int(t) for t in task_ids}
            if hotkey == "champ_hk":
                return {t: s for t, s in fx.champ_scores.items() if t in wanted}
            if hotkey == "chal_hk":
                return {t: s for t, s in fx.chal_scores.items() if t in wanted}
            for record in getattr(fx, "predeployed", []):
                if hotkey == record.challenger.hotkey:
                    pre_scores = getattr(fx, "pre_scores", {}).get(hotkey, {})
                    return {t: s for t, s in pre_scores.items() if t in wanted}
            return {}

    fx.state = _StateStub()
    fx.samples = _SamplesStub()
    worker._state = fx.state
    worker._samples = fx.samples

    fx.dispatched = []

    async def _capture(*, miner, task_id, base_url, **_kwargs):
        fx.dispatched.append((miner.uid, int(task_id)))

    worker._dispatch_one = _capture

    async def _drive(battle_only=False):
        n = await worker._dispatch_new(set(), set())
        await _asyncio.sleep(0)
        return n

    fx.drive = lambda: _asyncio.run(_drive())
    return fx


def test_champion_dispatch_stops_at_completion_threshold():
    """Champion has reached the 95%-of-pool threshold (210/221 for
    sampling_count=200) → executor must stop adding new champion
    candidates for this env. The remaining ~10 task_ids are the
    deliberately-abandoned long tail."""
    from affine.src.scorer.sampling_thresholds import champion_completion_threshold

    fx = _make_overlap_worker()
    threshold = champion_completion_threshold(fx.sampling_count)
    fx.champ_scores = {t: 0.5 for t in range(threshold)}     # exactly at threshold
    # Battle exists but challenger is also already at sampling_count overlap
    # so we isolate the champion-side check.
    fx.chal_scores = {t: 0.5 for t in range(fx.sampling_count)}

    fx.drive()

    champion_dispatches = [d for d in fx.dispatched if d[0] == 1]
    assert champion_dispatches == [], (
        f"champion at threshold should stop dispatching, got {len(champion_dispatches)}"
    )


def test_champion_dispatch_continues_below_threshold():
    """Champion below the 95% threshold → dispatch every un-sampled
    task_id (no per-tick cap on champion side)."""
    fx = _make_overlap_worker()
    fx.champ_scores = {t: 0.5 for t in range(100)}            # 100 of 220 done
    fx.chal_scores = {}                                       # battle still ramping

    fx.drive()

    champion_dispatches = [tid for uid, tid in fx.dispatched if uid == 1]
    assert len(champion_dispatches) == 220 - 100, (
        f"champion should dispatch 120 remaining, got {len(champion_dispatches)}"
    )
    assert set(champion_dispatches) == set(range(100, 220))


def test_challenger_only_dispatches_task_ids_in_champion_done():
    """Challenger ignores task_ids the champion hasn't sampled yet —
    dispatching them would never contribute to overlap."""
    fx = _make_overlap_worker()
    # Champion done on first 210 tasks; tail 210..219 still pending.
    fx.champ_scores = {t: 0.5 for t in range(210)}
    fx.chal_scores = {}

    fx.drive()

    challenger_dispatches = {tid for uid, tid in fx.dispatched if uid == 2}
    assert challenger_dispatches == set(range(210)), (
        f"challenger should target only champion's 210 done; touched tail "
        f"{challenger_dispatches - set(range(210))}"
    )


def test_challenger_dispatches_all_champion_done_not_capped_at_sampling_count():
    """Anti-stall: challenger MUST dispatch the full champion-done set
    (eg 210), not stop at the base sampling_count (200). Otherwise a
    handful of permanent failures in the first 200 leaves overlap
    forever short of threshold."""
    fx = _make_overlap_worker()
    fx.champ_scores = {t: 0.5 for t in range(210)}            # champion has 210 done
    fx.chal_scores = {}                                       # challenger fresh

    fx.drive()

    challenger_dispatches = [tid for uid, tid in fx.dispatched if uid == 2]
    assert len(challenger_dispatches) == 210, (
        f"challenger must dispatch all 210 of champion's set, got "
        f"{len(challenger_dispatches)}"
    )


def test_challenger_stops_when_overlap_reaches_sampling_count():
    """Challenger early-stop: once (champion ∩ challenger) ≥
    sampling_count, dispatch nothing more for this env. Saves GPU
    while waiting for other envs to catch up to their thresholds."""
    fx = _make_overlap_worker()
    fx.champ_scores = {t: 0.5 for t in range(210)}            # champion: 210 done
    fx.chal_scores = {t: 0.5 for t in range(200)}             # challenger: 200 done, overlap=200

    fx.drive()

    challenger_dispatches = [d for d in fx.dispatched if d[0] == 2]
    assert challenger_dispatches == [], (
        f"challenger at overlap=sampling_count should stop, got {len(challenger_dispatches)}"
    )


def test_challenger_ignores_existing_samples_outside_champion_set():
    """Stale challenger rows from before-this-rule (or from a contaminated
    earlier window) on task_ids champion never sampled don't count
    toward overlap and don't block fresh dispatches inside champion's
    set."""
    fx = _make_overlap_worker()
    # Champion done on {0..209}; challenger has 5 stale rows at {300..304}
    # (outside the pool entirely — even more clearly not in champion set).
    fx.champ_scores = {t: 0.5 for t in range(210)}
    fx.chal_scores = {300: 0.5, 301: 0.5, 302: 0.5, 303: 0.5, 304: 0.5}

    fx.drive()

    challenger_dispatches = [tid for uid, tid in fx.dispatched if uid == 2]
    # 5 stale rows don't count toward overlap (overlap=0); challenger
    # dispatches all 210 of champion's set fresh.
    assert len(challenger_dispatches) == 210
    assert all(tid in range(210) for tid in challenger_dispatches)


# ---- pre-deployed challengers dispatch -----------------------------------


def test_predeployed_challenger_dispatches_to_its_own_url():
    """Pre-deployed miners run on a non-primary endpoint, so their
    dispatches must hit that endpoint's base_url — not the
    primary/battle URL. Ensures the multi-host mode doesn't cross-fire
    samples between miners.
    """
    from affine.src.scorer.window_state import (
        DeploymentRecord, MinerSnapshot, BattleRecord,
    )

    fx = _make_overlap_worker()
    # Champion has sampled 5 task_ids on the primary; battle is None
    # (no active battle) so the only candidates beyond champion are
    # pre-deployed miners. Use a small task pool for arithmetic ease.
    fx.task_ids = list(range(5))
    fx.sampling_count = 5
    fx.champ_scores = {t: 0.5 for t in range(5)}
    fx.chal_scores = {}

    pre_miner = BattleRecord(
        challenger=MinerSnapshot(
            uid=42, hotkey="pre_hk", revision="pre_rev", model="org/p",
        ),
        deployment_id="wrk-pre",
        base_url="https://host-b/wrk-pre",
        started_at_block=0,
        deployments=[DeploymentRecord(
            endpoint_name="host-b",
            deployment_id="wrk-pre",
            base_url="https://host-b/wrk-pre",
        )],
    )
    fx.predeployed = [pre_miner]
    fx.pre_scores = {"pre_hk": {}}

    # Force battle to None to isolate pre-deployed dispatch.
    async def _no_battle(self):
        return None
    fx.state.get_battle = _no_battle.__get__(fx.state)

    fx.drive()

    pre_dispatches = [
        (tid, url) for uid, tid in fx.dispatched
        for url in [None]  # placeholder; we'll re-fetch with base_url
    ]
    # The fixture's _capture only stored (uid, tid). To check URL we
    # need a richer capture — replace and re-run.
    fx.dispatched = []
    captured = []

    async def _capture_full(*, miner, task_id, base_url, **_kwargs):
        captured.append((miner.uid, int(task_id), base_url))

    fx.worker._dispatch_one = _capture_full
    fx.drive()

    pre_rows = [row for row in captured if row[0] == 42]
    assert len(pre_rows) == 5, (
        f"pre-deployed should dispatch 5 task_ids (all champion-done), "
        f"got {len(pre_rows)}"
    )
    assert all(
        url == "https://host-b/wrk-pre" for _, _, url in pre_rows
    ), "pre-deployed dispatches must hit its own endpoint URL"


def test_predeployed_challenger_gated_on_champion_done():
    """Pre-deployed miner only dispatches for task_ids the champion has
    already sampled — same gating as ``battle.challenger``. While
    champion is mid-baseline, other machines stay idle automatically
    (the user's '冠军采样时其他机器空闲' invariant)."""
    from affine.src.scorer.window_state import (
        DeploymentRecord, MinerSnapshot, BattleRecord,
    )

    fx = _make_overlap_worker()
    fx.task_ids = list(range(10))
    fx.sampling_count = 10
    # Champion has only done 3 of 10 so far.
    fx.champ_scores = {0: 0.5, 1: 0.5, 2: 0.5}
    fx.chal_scores = {}

    pre_miner = BattleRecord(
        challenger=MinerSnapshot(
            uid=42, hotkey="pre_hk", revision="pre_rev", model="org/p",
        ),
        deployment_id="wrk-pre",
        base_url="https://host-b/wrk-pre",
        started_at_block=0,
        deployments=[DeploymentRecord(
            endpoint_name="host-b",
            deployment_id="wrk-pre",
            base_url="https://host-b/wrk-pre",
        )],
    )
    fx.predeployed = [pre_miner]
    fx.pre_scores = {"pre_hk": {}}

    async def _no_battle(self):
        return None
    fx.state.get_battle = _no_battle.__get__(fx.state)

    fx.drive()

    pre_tids = {tid for uid, tid in fx.dispatched if uid == 42}
    assert pre_tids == {0, 1, 2}, (
        f"pre-deployed must gate on champion-done set {{0,1,2}}, got {pre_tids}"
    )
