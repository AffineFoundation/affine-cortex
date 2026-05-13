from affine.src.executor.worker import _base_urls, _pick_url
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
