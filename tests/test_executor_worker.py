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


def test_env_concurrency_cap_none_means_no_env_semaphore():
    from affine.src.executor.worker import ExecutorWorker

    worker = ExecutorWorker(worker_id=0, env="MEMORY")
    assert worker._env_sem is None


def test_env_concurrency_cap_creates_asyncio_semaphore():
    import asyncio as _asyncio
    from affine.src.executor.worker import ExecutorWorker

    worker = ExecutorWorker(
        worker_id=0, env="LIVEWEB", env_concurrency_cap=100,
    )
    assert isinstance(worker._env_sem, _asyncio.Semaphore)
    # Acquire 100 times; the 101st should not be immediately available.
    async def _drive():
        for _ in range(100):
            await worker._env_sem.acquire()
        assert worker._env_sem.locked()

    _asyncio.run(_drive())


def test_env_sem_caps_in_flight_evaluate_and_persist():
    """Per-env cap must serialise concurrent evaluate calls inside one
    worker. Without it, a flaky env's infra-retry storm can occupy the
    entire global dispatch budget — exactly the LIVEWEB Stooq-lock
    incident this knob exists to prevent."""
    import asyncio as _asyncio
    from affine.src.executor.worker import ExecutorWorker
    from affine.src.scorer.window_state import MinerSnapshot

    worker = ExecutorWorker(
        worker_id=0, env="LIVEWEB", env_concurrency_cap=2,
    )

    observed_peak = {"value": 0}
    in_flight = {"value": 0}

    async def _fake_gated(**_kwargs):
        in_flight["value"] += 1
        observed_peak["value"] = max(observed_peak["value"], in_flight["value"])
        try:
            await _asyncio.sleep(0.05)
        finally:
            in_flight["value"] -= 1

    worker._evaluate_and_persist_gated = _fake_gated

    miner = MinerSnapshot(uid=1, hotkey="hk", model="m", revision="r")

    async def _drive():
        # Five concurrent calls; cap=2 should keep peak at 2.
        await _asyncio.gather(*[
            worker._evaluate_and_persist(
                miner=miner, task_id=i, base_url="http://x",
                refresh_block=0,
            )
            for i in range(5)
        ])
        assert observed_peak["value"] == 2

    _asyncio.run(_drive())


def test_env_sem_unset_does_not_serialise_calls():
    """Without a cap, the outer wrapper must be a no-op — concurrent
    evaluate calls all proceed in parallel up to the global semaphore."""
    import asyncio as _asyncio
    from affine.src.executor.worker import ExecutorWorker
    from affine.src.scorer.window_state import MinerSnapshot

    worker = ExecutorWorker(worker_id=0, env="MEMORY")  # no cap
    observed_peak = {"value": 0}
    in_flight = {"value": 0}

    async def _fake_gated(**_kwargs):
        in_flight["value"] += 1
        observed_peak["value"] = max(observed_peak["value"], in_flight["value"])
        try:
            await _asyncio.sleep(0.05)
        finally:
            in_flight["value"] -= 1

    worker._evaluate_and_persist_gated = _fake_gated

    miner = MinerSnapshot(uid=1, hotkey="hk", model="m", revision="r")

    async def _drive():
        await _asyncio.gather(*[
            worker._evaluate_and_persist(
                miner=miner, task_id=i, base_url="http://x",
                refresh_block=0,
            )
            for i in range(5)
        ])
        assert observed_peak["value"] == 5

    _asyncio.run(_drive())


def test_env_from_payload_parses_max_concurrent():
    from affine.src.scorer.window_state import _env_from_payload

    cfg = _env_from_payload({
        "display_name": "LIVEWEB",
        "enabled_for_sampling": True,
        "sampling": {
            "sampling_count": 400,
            "dataset_range": [[0, 100]],
            "sampling_mode": "random",
            "max_concurrent": 100,
        },
    })
    assert cfg.max_concurrent == 100


def test_env_from_payload_max_concurrent_defaults_to_none():
    from affine.src.scorer.window_state import _env_from_payload

    cfg = _env_from_payload({
        "display_name": "MEMORY",
        "enabled_for_sampling": True,
        "sampling": {"sampling_count": 200, "dataset_range": [[0, 10]]},
    })
    assert cfg.max_concurrent is None


def test_env_from_payload_rejects_nonpositive_max_concurrent():
    from affine.src.scorer.window_state import _env_from_payload

    # Negative / zero / non-numeric all fall back to None to avoid
    # silently uncapping or deadlocking an env with cap=0.
    for bad in (-1, 0, "abc", None, "0"):
        cfg = _env_from_payload({
            "display_name": "x",
            "enabled_for_sampling": True,
            "sampling": {"max_concurrent": bad},
        })
        assert cfg.max_concurrent is None, f"bad input {bad!r} leaked through"


def test_compute_env_caps_defaults_to_fair_share():
    from affine.src.executor.main import _compute_env_caps

    caps = _compute_env_caps(
        ["LIVEWEB", "MEMORY", "NAVWORLD", "SWE-INFINITE", "TERMINAL"],
        {},
        global_budget=600,
    )

    assert caps == {
        "LIVEWEB": 120,
        "MEMORY": 120,
        "NAVWORLD": 120,
        "SWE-INFINITE": 120,
        "TERMINAL": 120,
    }


def test_compute_env_caps_allows_explicit_override():
    from affine.src.executor.main import _compute_env_caps

    caps = _compute_env_caps(
        ["LIVEWEB", "MEMORY", "NAVWORLD"],
        {
            "LIVEWEB": {
                "enabled_for_sampling": True,
                "sampling": {"max_concurrent": 80},
            },
        },
        global_budget=300,
    )

    assert caps == {
        "LIVEWEB": 80,
        "MEMORY": 110,
        "NAVWORLD": 110,
    }
