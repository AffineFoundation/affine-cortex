from affine.src.executor.worker import _base_urls, _is_zero_score_error, _pick_url
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

    class _SamplesStub:
        async def has_sample(self, *a, **kw):
            return False

        async def count_samples_for_tasks(self, *a, **kw):
            return 0

    dispatched: list = []

    async def _capture_dispatch_one(*, miner, task_id, base_url, **_kwargs):
        dispatched.append((miner.uid, task_id, base_url))

    worker._state = _StateStub()
    worker._samples = _SamplesStub()
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

    class _SamplesStub:
        async def has_sample(self, *a, **kw):
            return False

        async def count_samples_for_tasks(self, *a, **kw):
            return 0

    worker._state = _StateStub()
    worker._samples = _SamplesStub()

    async def _drive():
        return await worker._dispatch_new(set(), set())

    assert _asyncio.run(_drive()) == 0
