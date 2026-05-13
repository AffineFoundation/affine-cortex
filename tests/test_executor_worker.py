from affine.src.executor.worker import _base_urls, _is_zero_score_error, _pick_url
from affine.src.executor.main import ExecutorManager
from affine.src.scorer.window_state import DeploymentRecord


def _queue_probe(q):
    q.put("ok")


def test_executor_manager_stats_queue_can_be_passed_to_spawn_process():
    manager = ExecutorManager(["affine:ded-v2"])
    proc = manager.mp_ctx.Process(target=_queue_probe, args=(manager.stats_queue,))

    proc.start()
    proc.join(timeout=10)

    assert proc.exitcode == 0
    assert manager.stats_queue.get(timeout=1) == "ok"


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


def test_refresh_dispatch_cap_picks_up_managers_value():
    import multiprocessing as _mp
    from affine.src.executor.worker import ExecutorWorker

    cap = _mp.get_context("spawn").Value("i", 60)
    worker = ExecutorWorker(
        worker_id=0, env="MEMORY", max_concurrent=60, cap_value=cap,
    )

    # No change when value matches.
    worker._refresh_dispatch_cap()
    assert worker.max_concurrent == 60

    # Manager rebalances downward.
    cap.value = 25
    worker._refresh_dispatch_cap()
    assert worker.max_concurrent == 25

    # Manager rebalances upward.
    cap.value = 200
    worker._refresh_dispatch_cap()
    assert worker.max_concurrent == 200


def test_refresh_dispatch_cap_noop_when_unwired():
    from affine.src.executor.worker import ExecutorWorker

    worker = ExecutorWorker(
        worker_id=0, env="MEMORY", max_concurrent=42, cap_value=None,
    )
    worker._refresh_dispatch_cap()
    assert worker.max_concurrent == 42


def test_refresh_dispatch_cap_ignores_zero_or_negative():
    import multiprocessing as _mp
    from affine.src.executor.worker import ExecutorWorker

    cap = _mp.get_context("spawn").Value("i", 0)
    worker = ExecutorWorker(
        worker_id=0, env="MEMORY", max_concurrent=60, cap_value=cap,
    )
    worker._refresh_dispatch_cap()
    assert worker.max_concurrent == 60  # zero ignored

    cap.value = -1
    worker._refresh_dispatch_cap()
    assert worker.max_concurrent == 60  # negative ignored
