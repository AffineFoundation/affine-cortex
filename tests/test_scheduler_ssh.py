"""Tests for the SSH-based sglang provider.

Cover the pure parts (config parsing, command builder, single-instance
state-cleanup invariants in flow.py). The actual SSH + docker
invocations need a real host; those land in the staging integration
suite when b300 is reachable.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from affine.src.scheduler.ssh import (
    CONTAINER_NAME,
    SSHConfig,
    _build_docker_run_cmd,
    _build_sglang_args,
)
from affine.src.scheduler.targon import DeployTarget


# ---- SSHConfig.from_env ---------------------------------------------------


def test_ssh_config_full_url():
    with patch.dict(os.environ, {
        "AFFINE_SSH_PROVIDER_URL": "ssh://operator@b300.internal:2222",
        "AFFINE_SSH_PROVIDER_KEY_PATH": "/secrets/id_b300",
        "AFFINE_SSH_PROVIDER_PUBLIC_URL": "https://b300-edge.example/v1",
    }, clear=False):
        cfg = SSHConfig.from_env()
    assert cfg.host == "b300.internal"
    assert cfg.user == "operator"
    assert cfg.port == 2222
    assert cfg.key_path == "/secrets/id_b300"
    assert cfg.public_inference_url == "https://b300-edge.example/v1"


def test_ssh_config_defaults_to_root_and_port_22():
    with patch.dict(os.environ, {
        "AFFINE_SSH_PROVIDER_URL": "ssh://b300",
    }, clear=False):
        cfg = SSHConfig.from_env()
    assert cfg.user == "root"
    assert cfg.port == 22
    assert cfg.host == "b300"


def test_ssh_config_missing_url_raises():
    env = {k: v for k, v in os.environ.items() if k != "AFFINE_SSH_PROVIDER_URL"}
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(RuntimeError, match="not set"):
            SSHConfig.from_env()


def test_ssh_config_bad_scheme_raises():
    with patch.dict(os.environ, {
        "AFFINE_SSH_PROVIDER_URL": "http://b300:22",
    }, clear=False):
        with pytest.raises(ValueError, match="ssh://"):
            SSHConfig.from_env()


def test_inference_url_defaults_to_host_port():
    cfg = SSHConfig(host="b300", user="root", port=22)
    assert cfg.inference_url() == "http://b300:30000/v1"


def test_inference_url_uses_public_override():
    cfg = SSHConfig(host="b300", user="root", port=22,
                    public_inference_url="https://edge.example/v1")
    assert cfg.inference_url() == "https://edge.example/v1"


# ---- sglang args + docker cmd ---------------------------------------------


def _target():
    return DeployTarget(
        uid=42, hotkey="abc123def", model="Qwen/Qwen3-30B-A22B",
        revision="abcdef01234567",
    )


def _config(**overrides):
    return SSHConfig(host="b300", user="root", port=22, **overrides)


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
        ("--context-length", "65536"),
        ("--mem-fraction-static", "0.8"),
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


def test_tool_call_parser_none_omits_flag(monkeypatch):
    """``AFFINE_SSH_PROVIDER_TOOL_CALL_PARSER=none`` should drop the flag
    entirely so non-Qwen models don't get a spurious parser."""
    monkeypatch.setattr(
        "affine.src.scheduler.ssh.DEFAULT_TOOL_CALL_PARSER", "none"
    )
    args = _build_sglang_args(_target(), _config())
    assert "--tool-call-parser" not in args


def test_dp_disabled_when_one():
    """Single-GPU configs (DP=1) must not pass ``--dp`` at all."""
    args = _build_sglang_args(_target(), _config(sglang_dp=1))
    assert "--dp" not in args


def test_config_from_endpoint_overrides_module_defaults():
    """Per-endpoint sglang_port / sglang_dp / sglang_image / sglang_cache_dir
    flow through into the docker command."""
    from affine.database.dao.inference_endpoints import Endpoint
    ep = Endpoint(
        name="ep1", kind="ssh",
        ssh_url="ssh://root@1.2.3.4:10300",
        public_inference_url="http://val:31000/v1",
        sglang_port=31000, sglang_dp=4,
        sglang_image="ghcr.io/custom/sglang:custom",
        sglang_cache_dir="/srv/cache",
    )
    cfg = SSHConfig.from_endpoint(ep)
    cmd = _build_docker_run_cmd(_target(), cfg)
    assert "-p 31000:31000" in cmd
    assert "-v /srv/cache:/data" in cmd
    assert "ghcr.io/custom/sglang:custom" in cmd
    args = _build_sglang_args(_target(), cfg)
    assert args[args.index("--port") + 1] == "31000"
    assert args[args.index("--download-dir") + 1] == "/srv/cache"
    assert args[args.index("--dp") + 1] == "4"
    # inference_url uses the public override, not host:port
    assert cfg.inference_url() == "http://val:31000/v1"
