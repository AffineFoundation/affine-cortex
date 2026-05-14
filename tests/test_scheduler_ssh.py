"""Tests for the SSH-based sglang provider.

Cover the pure parts (config parsing, command builder, single-instance
state-cleanup invariants in flow.py). The actual SSH + docker
invocations need a real host; those land in the staging integration
suite when b300 is reachable.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from affine.database.dao.inference_endpoints import Endpoint
from affine.src.scheduler.main import _select_ssh_endpoints
from affine.src.scheduler.ssh import (
    CONTAINER_NAME,
    SSHConfig,
    _build_docker_run_cmd,
    _build_sglang_args,
    deploy,
)
from affine.src.scheduler.targon import DeployTarget


@dataclass
class _EndpointStub:
    name: str
    assigned_uid: int | None = None
    assigned_hotkey: str | None = None
    assigned_model: str | None = None
    assigned_revision: str | None = None


# ---- SSHConfig.from_endpoint ----------------------------------------------


def test_ssh_config_from_endpoint_full_url():
    cfg = SSHConfig.from_endpoint(Endpoint(
        name="b300",
        kind="ssh",
        ssh_url="ssh://operator@b300.internal:2222",
        ssh_key_path="/secrets/id_b300",
        public_inference_url="https://b300-edge.example/v1",
    ))
    assert cfg.host == "b300.internal"
    assert cfg.user == "operator"
    assert cfg.port == 2222
    assert cfg.endpoint_name == "b300"
    assert cfg.key_path == "/secrets/id_b300"
    assert cfg.public_inference_url == "https://b300-edge.example/v1"


def test_ssh_config_from_endpoint_defaults_to_root_and_port_22():
    cfg = SSHConfig.from_endpoint(Endpoint(
        name="b300",
        kind="ssh",
        ssh_url="ssh://b300",
    ))
    assert cfg.user == "root"
    assert cfg.port == 22
    assert cfg.host == "b300"


def test_ssh_config_endpoint_missing_url_raises():
    with pytest.raises(ValueError, match="no ssh_url"):
        SSHConfig.from_endpoint(Endpoint(name="b300", kind="ssh"))


def test_ssh_config_bad_scheme_raises():
    with pytest.raises(ValueError, match="ssh://"):
        SSHConfig.from_endpoint(Endpoint(
            name="b300",
            kind="ssh",
            ssh_url="http://b300:22",
        ))


def test_inference_url_defaults_to_host_port():
    cfg = SSHConfig(host="b300", user="root", port=22)
    assert cfg.inference_url() == "http://b300:30000/v1"


def test_inference_url_uses_public_override():
    cfg = SSHConfig(host="b300", user="root", port=22,
                    public_inference_url="https://edge.example/v1")
    assert cfg.inference_url() == "https://edge.example/v1"


def test_deployment_id_includes_endpoint_name():
    cfg = SSHConfig(host="b300", endpoint_name="ssh_b300")
    assert cfg.deployment_id() == f"ssh:ssh_b300:{CONTAINER_NAME}"


# ---- sglang args + docker cmd ---------------------------------------------


def _target():
    return DeployTarget(
        uid=42, hotkey="abc123def", model="Qwen/Qwen3-30B-A22B",
        revision="abcdef01234567",
    )


def _config(**overrides):
    return SSHConfig(host="b300", user="root", port=22, **overrides)


# ---- endpoint selection ----------------------------------------------------


def test_select_ssh_endpoints_reuses_matching_assignment():
    target = _target()
    endpoints = [
        _EndpointStub("a"),
        _EndpointStub(
            "b",
            assigned_uid=target.uid,
            assigned_hotkey=target.hotkey,
            assigned_model=target.model,
            assigned_revision=target.revision,
        ),
    ]
    assert _select_ssh_endpoints(endpoints, target, role="champion") == [endpoints[1]]


def test_select_ssh_endpoints_splits_champion_and_challenger_capacity():
    target = _target()
    endpoints = [_EndpointStub("a"), _EndpointStub("b"), _EndpointStub("c")]

    champion = _select_ssh_endpoints(endpoints, target, role="champion")
    assert [ep.name for ep in champion] == ["a", "b"]

    endpoints[0].assigned_uid = target.uid
    endpoints[1].assigned_uid = target.uid
    challenger = _select_ssh_endpoints(endpoints, target, role="challenger")
    assert [ep.name for ep in challenger] == ["c"]


def test_sglang_args_carry_model_and_revision():
    args = _build_sglang_args(_target(), _config())
    assert "--model-path" in args
    assert args[args.index("--model-path") + 1] == "Qwen/Qwen3-30B-A22B"
    assert args[args.index("--revision") + 1] == "abcdef01234567"


def test_sglang_args_include_required_perf_flags():
    args = _build_sglang_args(_target(), _config())
    # Required flags from the original targon_client.py reference impl.
    must_have_pairs = [
        ("--download-dir", "/data"),
        ("--host", "0.0.0.0"),
        ("--port", "30000"),
        ("--mem-fraction-static", "0.85"),
        ("--chunked-prefill-size", "4096"),
        ("--tool-call-parser", "qwen"),
        ("--dp", "8"),
    ]
    for flag, value in must_have_pairs:
        assert flag in args, f"missing {flag}"
        assert args[args.index(flag) + 1] == value, (
            f"{flag}: expected {value}, got {args[args.index(flag) + 1]}"
        )
    # --trust-remote-code is a flag without a value
    assert "--trust-remote-code" in args


def test_docker_cmd_rms_existing_container_first():
    cmd = _build_docker_run_cmd(_target(), _config())
    # The rm-then-run order matters — it kicks off a fresh state on every
    # deploy, supports the single-instance contract.
    rm_idx = cmd.find(f"docker rm -f {CONTAINER_NAME}")
    run_idx = cmd.find("docker run")
    assert rm_idx >= 0 and run_idx >= 0
    assert rm_idx < run_idx


def test_docker_cmd_passes_gpus_all_and_mounts_cache():
    cmd = _build_docker_run_cmd(_target(), _config())
    assert "--gpus all" in cmd
    assert "-v /data:/data" in cmd
    assert "lmsysorg/sglang:latest" in cmd
    assert "python -m sglang.launch_server" in cmd


def test_docker_cmd_labels_current_machine_assignment():
    cmd = _build_docker_run_cmd(
        _target(), _config(endpoint_name="ssh_b300")
    )
    assert "--label io.affine.endpoint=ssh_b300" in cmd
    assert "--label io.affine.uid=42" in cmd
    assert "--label io.affine.hotkey=abc123def" in cmd
    assert "--label io.affine.model=Qwen/Qwen3-30B-A22B" in cmd
    assert "--label io.affine.revision=abcdef01234567" in cmd


def test_docker_cmd_uses_host_network_and_ipc():
    """Docker 29.x on cgroup-v2 kernels chokes on ``-p`` due to procfs
    sysctl access; production b300 uses ``--network host``. ``--ipc=host``
    + a generous ``--shm-size`` is required for sglang multi-process
    (--dp/--tp) workers to share CUDA IPC handles and tokenizer caches."""
    cmd = _build_docker_run_cmd(_target(), _config())
    assert "--network host" in cmd
    assert "-p " not in cmd, "port mapping triggers OCI procfs bug on b300"
    assert "--ipc=host" in cmd
    assert "--shm-size=" in cmd
    assert "--security-opt label=disable" in cmd


def test_docker_cmd_passes_hf_token_when_env_set(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test_TOKEN_123")
    cmd = _build_docker_run_cmd(_target(), _config())
    assert "-e HF_TOKEN=hf_test_TOKEN_123" in cmd
    assert "-e HUGGING_FACE_HUB_TOKEN=hf_test_TOKEN_123" in cmd


def test_docker_cmd_omits_hf_token_when_env_unset(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    cmd = _build_docker_run_cmd(_target(), _config())
    assert "-e HF_TOKEN=" not in cmd
    assert "-e HUGGING_FACE_HUB_TOKEN=" not in cmd


def test_tool_call_parser_none_omits_flag():
    """Endpoint config can set parser='none' so non-Qwen models don't get
    a spurious parser."""
    args = _build_sglang_args(
        _target(), _config(sglang_tool_call_parser="none")
    )
    assert "--tool-call-parser" not in args


def test_dp_disabled_when_one():
    """Single-GPU configs (DP=1) must not pass ``--dp`` at all."""
    args = _build_sglang_args(_target(), _config(sglang_dp=1))
    assert "--dp" not in args


def test_config_from_endpoint_overrides_module_defaults():
    """Per-endpoint sglang settings flow through into the docker command."""
    ep = Endpoint(
        name="ep1", kind="ssh",
        ssh_url="ssh://root@1.2.3.4:10300",
        public_inference_url="http://val:31000/v1",
        sglang_port=31000, sglang_dp=4,
        sglang_image="ghcr.io/custom/sglang:custom",
        sglang_cache_dir="/srv/cache",
        sglang_context_len=32768,
        sglang_mem_fraction=0.7,
        sglang_chunked_prefill=2048,
        sglang_tool_call_parser="none",
        ready_timeout_sec=900,
        poll_interval_sec=3.0,
    )
    cfg = SSHConfig.from_endpoint(ep)
    assert cfg.ready_timeout_sec == 900
    assert cfg.poll_interval_sec == 3.0
    cmd = _build_docker_run_cmd(_target(), cfg)
    assert "--network host" in cmd
    assert "-p 31000:31000" not in cmd
    # --network host means no -p flag; the sglang --port arg is what binds.
    assert "-v /srv/cache:/data" in cmd
    assert "ghcr.io/custom/sglang:custom" in cmd
    args = _build_sglang_args(_target(), cfg)
    assert args[args.index("--port") + 1] == "31000"
    assert args[args.index("--download-dir") + 1] == "/srv/cache"
    assert "--context-length" not in args
    assert args[args.index("--mem-fraction-static") + 1] == "0.7"
    assert args[args.index("--chunked-prefill-size") + 1] == "2048"
    assert args[args.index("--dp") + 1] == "4"
    assert "--tool-call-parser" not in args
    # inference_url uses the public override, not host:port
    assert cfg.inference_url() == "http://val:31000/v1"


@pytest.mark.asyncio
async def test_deploy_adopts_matching_container_without_restart(monkeypatch):
    calls = []

    async def fake_ssh_exec(config, command):
        calls.append(command)
        if command.startswith("docker inspect"):
            return 0, "\n".join([
                "io.affine.endpoint = ssh_b300",
                "io.affine.uid = 42",
                "io.affine.hotkey = abc123def",
                "io.affine.model = Qwen/Qwen3-30B-A22B",
                "io.affine.revision = abcdef01234567",
            ]), ""
        raise AssertionError(f"unexpected command: {command}")

    async def fake_wait_ready(*args, **kwargs):
        return None

    monkeypatch.setattr("affine.src.scheduler.ssh._ssh_exec", fake_ssh_exec)
    monkeypatch.setattr("affine.src.scheduler.ssh._wait_ready", fake_wait_ready)

    cfg = _config(endpoint_name="ssh_b300")
    result = await deploy(cfg, _target())

    assert result.deployment_id == f"ssh:ssh_b300:{CONTAINER_NAME}"
    assert result.base_url == "http://b300:30000/v1"
    assert len(calls) == 1
    assert calls[0].startswith("docker inspect")


@pytest.mark.asyncio
async def test_deploy_restarts_when_existing_container_is_wrong(monkeypatch):
    calls = []

    async def fake_ssh_exec(config, command):
        calls.append(command)
        if command.startswith("docker inspect"):
            return 0, "\n".join([
                "io.affine.endpoint = ssh_b300",
                "io.affine.uid = 99",
                "io.affine.hotkey = wrong",
                "io.affine.model = other/model",
                "io.affine.revision = otherrev",
            ]), ""
        if command.startswith(f"docker rm -f {CONTAINER_NAME}"):
            return 0, "container1234567890", ""
        raise AssertionError(f"unexpected command: {command}")

    async def fake_wait_ready(*args, **kwargs):
        return None

    monkeypatch.setattr("affine.src.scheduler.ssh._ssh_exec", fake_ssh_exec)
    monkeypatch.setattr("affine.src.scheduler.ssh._wait_ready", fake_wait_ready)

    cfg = _config(endpoint_name="ssh_b300")
    result = await deploy(cfg, _target())

    assert result.deployment_id == f"ssh:ssh_b300:{CONTAINER_NAME}"
    assert len(calls) == 2
    assert calls[0].startswith("docker inspect")
    assert calls[1].startswith(f"docker rm -f {CONTAINER_NAME}")
