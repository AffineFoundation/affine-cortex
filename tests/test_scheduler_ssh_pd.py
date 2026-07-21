"""4P4D SSH deployment command and lifecycle coverage."""

from __future__ import annotations

import json

import pytest

from affine.src.scheduler.health import DeploymentHealthState
from affine.src.scheduler.ssh import SSHConfig
from affine.src.scheduler.ssh_pd import (
    DECODE_CONTAINER_PREFIX,
    GATEWAY_CONTAINER_NAME,
    GATEWAY_REQUEST_TIMEOUT_SEC,
    NUM_CONTINUOUS_DECODE_STEPS,
    PREFILL_CONTAINER_PREFIX,
    _build_gateway_docker_cmd,
    _build_worker_docker_cmd,
    _expected_container_labels,
    _gateway_args,
    _wait_gateway_ready,
    _wait_workers_ready,
    _worker_args,
    deploy,
    deployment_health,
    deployment_healthy,
    managed_container_names,
    required_gpu_count,
    reservation_timeout_sec,
    validate_config,
    workers,
)
from affine.src.scheduler.targon import DeployTarget


def _digest(name: str) -> str:
    return f"{name}@sha256:{'a' * 64}"


def _config(**overrides) -> SSHConfig:
    values = {
        "host": "b200.test",
        "endpoint_name": "pd-b200",
        "serving_mode": "pd",
        "sglang_image": _digest("lmsysorg/sglang:v0.5.14"),
        "sglang_pd_gateway_image": _digest("lmsysorg/sgl-model-gateway:v0.5.14"),
    }
    values.update(overrides)
    return SSHConfig(**values)


def _target() -> DeployTarget:
    return DeployTarget(
        uid=42,
        hotkey="hk42",
        model="Qwen/Qwen3-30B-A22B",
        revision="0123456789abcdef",
    )


def test_pd_topology_maps_four_prefill_then_four_decode_gpus():
    config = _config()

    topology = workers(config)

    assert len(topology) == 8
    assert [worker.role for worker in topology] == ["prefill"] * 4 + ["decode"] * 4
    assert [worker.gpu for worker in topology] == list(range(8))
    assert [worker.http_port for worker in topology[:4]] == list(range(11000, 11004))
    assert [worker.bootstrap_port for worker in topology[:4]] == list(
        range(12000, 12004)
    )
    assert [worker.http_port for worker in topology[4:]] == list(range(13000, 13004))
    assert required_gpu_count(config) == 8
    assert reservation_timeout_sec(config) == 3660


def test_pd_worker_args_are_dp1_and_include_tuned_scheduler_flags():
    config = _config()
    prefill, decode = workers(config)[0], workers(config)[4]

    prefill_args = _worker_args(_target(), config, prefill)
    decode_args = _worker_args(_target(), config, decode)

    assert "--dp" not in prefill_args
    assert "--dp" not in decode_args
    assert prefill_args[prefill_args.index("--disaggregation-mode") + 1] == ("prefill")
    assert decode_args[decode_args.index("--disaggregation-mode") + 1] == ("decode")
    assert "--enable-mixed-chunk" in prefill_args
    assert "--enable-mixed-chunk" in decode_args
    assert "--enable-metrics" in prefill_args
    assert "--enable-metrics" in decode_args
    assert (
        decode_args[decode_args.index("--num-continuous-decode-steps") + 1]
        == str(NUM_CONTINUOUS_DECODE_STEPS)
        == "8"
    )
    assert (
        prefill_args[prefill_args.index("--disaggregation-bootstrap-port") + 1]
        == "12000"
    )
    assert "--disaggregation-bootstrap-port" not in decode_args


def test_worker_container_is_loopback_single_gpu_and_mooncake_nvlink():
    config = _config()
    worker = workers(config)[2]

    command = _build_worker_docker_cmd(_target(), config, worker)

    assert f"--name {PREFILL_CONTAINER_PREFIX}-2" in command
    assert "--gpus device=2" in command
    assert "--entrypoint python" in command
    assert "--host 127.0.0.1" in command
    assert "SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK" in command
    assert "MC_INTRANODE_NVLINK=true" in command


def test_gateway_has_four_prefill_and_decode_entries_and_public_port():
    config = _config()
    args = _gateway_args(config)

    assert args.count("--prefill") == 4
    assert args.count("--decode") == 4
    assert args[args.index("--port") + 1] == "10001"
    assert args[args.index("--prefill-policy") + 1] == "cache_aware"
    assert args[args.index("--decode-policy") + 1] == "power_of_two"
    assert args[args.index("--request-timeout-secs") + 1] == str(
        GATEWAY_REQUEST_TIMEOUT_SEC
    )
    for port in range(11000, 11004):
        assert f"http://127.0.0.1:{port}" in args
    for port in range(13000, 13004):
        assert f"http://127.0.0.1:{port}" in args

    command = _build_gateway_docker_cmd(_target(), config)
    assert f"--name {GATEWAY_CONTAINER_NAME}" in command
    assert "--gpus" not in command
    assert "--entrypoint python" in command
    assert "-m sglang_router.launch_router" in command


def test_pd_config_requires_exact_4p4d_unique_ports_and_digests():
    validate_config(_config())

    with pytest.raises(ValueError, match="exactly 4 prefill"):
        validate_config(_config(sglang_pd_prefill_replicas=3))
    with pytest.raises(ValueError, match="ports must be unique"):
        validate_config(_config(sglang_pd_prefill_port_start=10001))
    with pytest.raises(ValueError, match="ports must be unique"):
        validate_config(_config(sglang_pd_decode_port_start=14000))
    with pytest.raises(ValueError, match="sglang_image must be pinned"):
        validate_config(_config(sglang_image="lmsysorg/sglang:v0.5.14"))
    with pytest.raises(ValueError, match="gateway_image must be pinned"):
        validate_config(
            _config(sglang_pd_gateway_image="lmsysorg/sgl-model-gateway:latest")
        )
    with pytest.raises(ValueError, match="may not override PD topology"):
        validate_config(_config(sglang_docker_args=("--gpus", "all")))


@pytest.mark.asyncio
async def test_deploy_persists_only_gateway_as_callable_endpoint(monkeypatch):
    import affine.src.scheduler.ssh_pd as ssh_pd

    commands = []

    async def fake_exec(_config, command):
        commands.append(command)
        return 0, "container-id", ""

    async def noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(ssh_pd, "_ssh_exec", fake_exec)
    monkeypatch.setattr(ssh_pd, "_cleanup_stale_caches", noop)
    monkeypatch.setattr(ssh_pd, "_should_disable_hf_xet", noop)
    monkeypatch.setattr(ssh_pd, "_wait_workers_ready", noop)
    monkeypatch.setattr(ssh_pd, "_wait_gateway_ready", noop)
    monkeypatch.setattr(ssh_pd, "_smoke_test", noop)

    result = await deploy(_config(), _target())

    assert result.deployment_id == "ssh:pd-b200:affine-sglang-pd-gateway"
    assert result.base_url == "http://b200.test:10001/v1"
    assert len(result.deployments) == 1
    assert result.deployments[0].base_url == result.base_url
    assert any(PREFILL_CONTAINER_PREFIX in command for command in commands)
    assert any(DECODE_CONTAINER_PREFIX in command for command in commands)
    assert any(GATEWAY_CONTAINER_NAME in command for command in commands)
    assert any("affine-sglang-current" in command for command in commands)


@pytest.mark.asyncio
async def test_health_requires_all_nine_matching_containers(monkeypatch):
    import affine.src.scheduler.ssh_pd as ssh_pd

    config = _config()
    target = _target()
    labels = _expected_container_labels(target, config)
    documents = [
        {
            "Name": f"/{name}",
            "State": {"Status": "running"},
            "Config": {"Labels": expected},
        }
        for name, expected in labels.items()
    ]

    async def inspect(_config):
        return documents

    async def ready(_url):
        return True

    monkeypatch.setattr(ssh_pd, "_inspect_containers", inspect)
    monkeypatch.setattr(ssh_pd, "_probe_ready", ready)
    health = await deployment_health(config, target)
    assert health.state is DeploymentHealthState.HEALTHY
    assert health.identity == config.health_identity()
    assert await deployment_healthy(config, target)

    documents.pop()
    health = await deployment_health(config, target)
    assert health.state is DeploymentHealthState.UNHEALTHY
    assert health.reason == "container_set_mismatch"


@pytest.mark.asyncio
async def test_health_classifies_public_path_failure_with_local_gateway_ready(
    monkeypatch,
):
    import affine.src.scheduler.ssh_pd as ssh_pd

    config = _config()
    target = _target()
    labels = _expected_container_labels(target, config)
    documents = [
        {
            "Name": f"/{name}",
            "State": {"Status": "running"},
            "Config": {"Labels": expected},
        }
        for name, expected in labels.items()
    ]

    async def inspect(_config):
        return documents

    async def not_ready(_url):
        return False

    async def local_ready(_config):
        return True

    monkeypatch.setattr(ssh_pd, "_inspect_containers", inspect)
    monkeypatch.setattr(ssh_pd, "_probe_ready", not_ready)
    monkeypatch.setattr(ssh_pd, "_probe_remote_gateway_ready", local_ready)

    health = await deployment_health(config, target)

    assert health.state is DeploymentHealthState.TRANSPORT_UNHEALTHY
    assert health.reason == "public_probe_failed_local_ready"


def test_managed_names_are_stable_and_do_not_use_shell_wildcards():
    names = managed_container_names(_config())

    assert len(names) == 9
    assert names[0] == GATEWAY_CONTAINER_NAME
    assert all("*" not in name for name in names)
    assert json.dumps(names)


@pytest.mark.asyncio
async def test_worker_readiness_fails_immediately_after_container_exit(
    monkeypatch,
):
    import affine.src.scheduler.ssh_pd as ssh_pd

    async def not_ready(_config):
        return False

    async def exited(_config, _name):
        return 17

    monkeypatch.setattr(ssh_pd, "_remote_workers_ready", not_ready)
    monkeypatch.setattr(ssh_pd, "_container_exit_code", exited)

    with pytest.raises(RuntimeError, match="exited with code 17"):
        await _wait_workers_ready(_config())


@pytest.mark.asyncio
async def test_gateway_readiness_fails_immediately_after_container_exit(
    monkeypatch,
):
    import affine.src.scheduler.ssh_pd as ssh_pd

    async def not_ready(_url):
        return False

    async def exited(_config, _name):
        return 19

    monkeypatch.setattr(ssh_pd, "_probe_ready", not_ready)
    monkeypatch.setattr(ssh_pd, "_container_exit_code", exited)

    with pytest.raises(RuntimeError, match="exited with code 19"):
        await _wait_gateway_ready(_config())
