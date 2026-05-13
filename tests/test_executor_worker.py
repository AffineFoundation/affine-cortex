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
