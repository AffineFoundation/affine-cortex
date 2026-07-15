"""Tests for the SSH-based sglang provider.

Cover the pure parts (config parsing, command builder, single-instance
state-cleanup invariants in flow.py). The actual SSH + docker
invocations need a real host; those land in the staging integration
suite when b300 is reachable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from affine.database.dao.inference_endpoints import Endpoint
from affine.src.scheduler.main import _select_ssh_endpoints
from affine.src.scheduler.ssh import (
    ACTIVE_HF_INCOMPLETE_CACHE_MAX_AGE_SECONDS,
    CONTAINER_NAME,
    DEFAULT_HF_METADATA_TIMEOUT_SEC,
    HF_XET_MAX_NON_XET_FILE_BYTES,
    HF_ORG_SEPARATOR,
    HF_SNAPSHOT_PREFIX,
    RESTART_POLICY,
    SSHConfig,
    _build_cache_cleanup_cmd,
    _build_docker_run_cmd,
    _build_sglang_args,
    _cleanup_stale_caches,
    _hf_cache_dir_name,
    _is_valid_hf_model_id,
    _max_weight_file_size_bytes,
    _parse_container_exit_status,
    _parse_container_inspect_json,
    _should_disable_hf_xet,
    deployment_healthy,
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
    deployment_id: str | None = None
    assignment_role: str | None = None
    assignment_token: str | None = None
    assignment_status: str | None = None
    assignment_expires_at: int = 0


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


def test_ssh_config_from_endpoint_keeps_extra_docker_args():
    cfg = SSHConfig.from_endpoint(Endpoint(
        name="targon",
        kind="ssh",
        ssh_url="ssh://worker@targon.example",
        sglang_docker_args=["--cgroupns=host"],
    ))
    assert cfg.sglang_docker_args == ("--cgroupns=host",)


def test_ssh_config_bad_scheme_raises():
    with pytest.raises(ValueError, match="ssh://"):
        SSHConfig.from_endpoint(Endpoint(
            name="b300",
            kind="ssh",
            ssh_url="http://b300:22",
        ))


def test_inference_url_defaults_to_host_port():
    cfg = SSHConfig(host="b300", user="root", port=22)
    assert cfg.inference_url() == "http://b300:10001/v1"


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


def test_select_ssh_endpoints_all_roles_take_first_free_endpoint():
    target = _target()
    endpoints = [_EndpointStub("a"), _EndpointStub("b"), _EndpointStub("c")]

    champion = _select_ssh_endpoints(endpoints, target, role="champion")
    assert [ep.name for ep in champion] == ["a"]

    challenger = _select_ssh_endpoints(endpoints, target, role="challenger")
    assert [ep.name for ep in challenger] == ["a"]

    pre = _select_ssh_endpoints(endpoints, target, role="pre_challenger")
    assert [ep.name for ep in pre] == ["a"]


def test_select_ssh_endpoints_skips_busy_endpoints_without_fixed_slots():
    target = _target()
    endpoints = [
        _EndpointStub(
            "a",
            assigned_uid=999,
            assignment_role="champion",
            assignment_token="owner-a",
            assignment_status="ready",
            deployment_id="dep-a",
        ),
        _EndpointStub(
            "b",
            assignment_token="reserving-b",
            assignment_status="deploying",
            assignment_expires_at=4_000_000_000,
        ),
        _EndpointStub("c"),
    ]

    pre = _select_ssh_endpoints(endpoints, target, role="pre_challenger")
    assert [ep.name for ep in pre] == ["c"]


def test_select_ssh_endpoints_challenger_can_replace_champion_when_full():
    target = _target()
    endpoints = [
        _EndpointStub(
            "a",
            assigned_uid=10,
            assignment_role="pre_challenger",
            assignment_token="owner-a",
            assignment_status="ready",
            deployment_id="dep-a",
        ),
        _EndpointStub(
            "b",
            assigned_uid=1,
            assignment_role="champion",
            assignment_token="owner-b",
            assignment_status="ready",
            deployment_id="dep-b",
        ),
    ]

    selected = _select_ssh_endpoints(endpoints, target, role="challenger")

    assert [ep.name for ep in selected] == ["b"]


def test_select_ssh_endpoints_challenger_replaces_legacy_active_role():
    endpoint = _EndpointStub(
        "only-endpoint",
        assigned_uid=1,
        assignment_role="active",
        assignment_token="legacy-owner",
        assignment_status="ready",
        deployment_id="legacy-deployment",
    )

    selected = _select_ssh_endpoints(
        [endpoint], _target(), role="challenger",
    )

    assert selected == [endpoint]


def test_select_ssh_endpoints_can_reclaim_expired_reservation():
    expired = _EndpointStub(
        "expired",
        assigned_uid=99,
        assignment_role="pre_challenger",
        assignment_token="expired-owner",
        assignment_status="deploying",
        assignment_expires_at=1,
    )

    selected = _select_ssh_endpoints(
        [expired], _target(), role="pre_challenger",
    )

    assert selected == [expired]


def test_select_ssh_endpoints_pre_challenger_raises_when_none_free():
    """Single-endpoint deployments have no capacity for pre-sampling, and
    pre-deploy attempts must surface the structured
    :class:`NoEndpointCapacity` so the scheduler's fill loop can tell
    'no capacity' (terminate cleanly) apart from real deploy failures
    (mark FAILED and advance).

    Subclassing ``RuntimeError`` keeps backward compatibility for any
    caller catching the broader type — verified by both
    ``isinstance`` checks below."""
    from affine.src.scheduler.flow import NoEndpointCapacity

    target = _target()
    endpoints = [_EndpointStub(
        "a",
        assigned_uid=1,
        assignment_role="champion",
        assignment_token="owner-a",
        assignment_status="ready",
        deployment_id="dep-a",
    )]

    try:
        _select_ssh_endpoints(endpoints, target, role="pre_challenger")
    except NoEndpointCapacity as e:
        assert isinstance(e, RuntimeError)
    else:
        raise AssertionError(
            "expected NoEndpointCapacity when endpoint is busy"
        )

    endpoints2 = [
        _EndpointStub("a", assigned_uid=98),
        _EndpointStub("b", assigned_uid=99),
    ]
    try:
        _select_ssh_endpoints(endpoints2, target, role="pre_challenger")
    except NoEndpointCapacity as e:
        assert isinstance(e, RuntimeError)
    else:
        raise AssertionError(
            "expected NoEndpointCapacity when every endpoint is assigned"
        )


def test_select_ssh_endpoints_does_not_evict_non_champion_assignment():
    from affine.src.scheduler.main import EndpointReservationConflict

    endpoints = [_EndpointStub(
        "a",
        assigned_uid=99,
        assignment_role="pre_challenger",
        assignment_token="owner-a",
        assignment_status="ready",
        deployment_id="dep-a",
    )]

    with pytest.raises(EndpointReservationConflict, match="no free ssh endpoint"):
        _select_ssh_endpoints(endpoints, _target(), role="challenger")


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
        ("--port", "10001"),
        ("--mem-fraction-static", "0.85"),
        ("--chunked-prefill-size", "4096"),
        ("--tool-call-parser", "qwen"),
        ("--dp", "8"),
        ("--load-balance-method", "total_tokens"),
    ]
    for flag, value in must_have_pairs:
        assert flag in args, f"missing {flag}"
        assert args[args.index(flag) + 1] == value, (
            f"{flag}: expected {value}, got {args[args.index(flag) + 1]}"
        )
    # --trust-remote-code is a flag without a value
    assert "--trust-remote-code" in args


def test_hf_cache_dir_name_flattens_org_separator():
    """HuggingFace snapshots live in ``<prefix><owner><sep><name>``."""
    qwen = _hf_cache_dir_name("Qwen/Qwen3-30B-A22B")
    assert qwen.startswith(HF_SNAPSHOT_PREFIX)
    assert qwen == f"{HF_SNAPSHOT_PREFIX}Qwen{HF_ORG_SEPARATOR}Qwen3-30B-A22B"
    assert _hf_cache_dir_name("prexpert/affine-138-5CqkEFMX") == \
        f"{HF_SNAPSHOT_PREFIX}prexpert{HF_ORG_SEPARATOR}affine-138-5CqkEFMX"


def test_hf_snapshot_prefix_is_non_empty():
    """The cleanup glob ``<cache>/<prefix>*/`` would match every
    directory under ``<cache>`` if the prefix were empty, blowing
    away all caches. The module-level assert prevents that class of
    mistake from compiling."""
    assert HF_SNAPSHOT_PREFIX, "prefix MUST be non-empty"
    assert HF_SNAPSHOT_PREFIX.endswith(HF_ORG_SEPARATOR), (
        "HF cache convention puts the org separator right after the prefix"
    )


def test_is_valid_hf_model_id_accepts_owner_slash_name():
    assert _is_valid_hf_model_id("Qwen/Qwen3-30B-A22B")
    assert _is_valid_hf_model_id("prexpert/affine-138")


def test_is_valid_hf_model_id_rejects_malformed_ids():
    """Malformed ids would produce a keep-dir name that matches NO
    real cache entry — the cleanup would then delete every other
    model dir, including the currently-serving one. Reject early."""
    assert not _is_valid_hf_model_id("")           # empty
    assert not _is_valid_hf_model_id("foo")         # no slash
    assert not _is_valid_hf_model_id("/foo")        # empty owner
    assert not _is_valid_hf_model_id("foo/")        # empty name
    assert not _is_valid_hf_model_id("foo/bar/baz") # extra slash
    assert not _is_valid_hf_model_id(None)          # not a string


@pytest.mark.asyncio
async def test_cleanup_skipped_for_invalid_model_id(monkeypatch):
    """When ``target.model`` isn't a valid HF id (no slash, empty,
    etc.) the cleanup must NOT run — the keep-dir name would match
    nothing, causing the loop to wipe every cache including the live
    one. The function should early-return without any SSH call."""
    calls = []

    async def fake_ssh_exec(config, command):
        calls.append(command)
        return 0, "", ""

    monkeypatch.setattr("affine.src.scheduler.ssh._ssh_exec", fake_ssh_exec)

    bad_target = DeployTarget(uid=1, hotkey="hk", model="", revision="r")
    await _cleanup_stale_caches(_config(), bad_target)
    assert calls == [], "cleanup must not SSH when model id is invalid"

    bad_target2 = DeployTarget(uid=1, hotkey="hk", model="just-a-name", revision="r")
    await _cleanup_stale_caches(_config(), bad_target2)
    assert calls == [], "cleanup must not SSH when model id lacks '/'"


def test_cache_cleanup_cmd_keeps_target_and_active_incomplete_downloads():
    """The cleanup snippet keeps the new target's HF dir and recently
    active incomplete downloads. The latter lets HF resume a large
    challenger download after a transient readiness timeout instead of
    starting from byte zero on every retry."""
    cmd = _build_cache_cleanup_cmd(_target(), _config())
    # Target dir name is computed and quoted into the script.
    expected_dir = _hf_cache_dir_name(_target().model)
    assert expected_dir in cmd, f"target dir {expected_dir!r} missing"
    # Cache dir from config is quoted into the script.
    assert _config().sglang_cache_dir in cmd
    # The cleanup uses ``rm -rf`` only inside the keep-filter loop.
    assert "rm -rf" in cmd
    # The keep-check must precede the rm so we don't delete the target.
    rm_idx = cmd.find("rm -rf")
    target_check_idx = cmd.find("$TARGET_DIR")
    assert target_check_idx < rm_idx, "target keep-check must come before rm"
    assert "*.incomplete" in cmd
    assert "kept-active:" in cmd
    assert str(ACTIVE_HF_INCOMPLETE_CACHE_MAX_AGE_SECONDS // 60) in cmd
    # No previous-model lookup — single keep dir.
    assert "docker inspect" not in cmd
    assert "PREV" not in cmd


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
    assert f"--restart {RESTART_POLICY}" in cmd
    assert "-v /data:/data" in cmd
    assert "lmsysorg/sglang:latest" in cmd
    assert "python -m sglang.launch_server" in cmd


def test_docker_cmd_allows_hf_xet_downloads():
    cmd = _build_docker_run_cmd(_target(), _config())
    assert "-e HF_HOME=/data" in cmd
    assert "-e HF_HUB_CACHE=/data" in cmd
    assert "-e TRANSFORMERS_CACHE=/data" in cmd
    assert "HF_HUB_DISABLE_XET" not in cmd


def test_docker_cmd_disables_hf_xet_when_requested():
    cmd = _build_docker_run_cmd(
        _target(), _config(), disable_hf_xet=True,
    )
    assert "-e HF_HUB_DISABLE_XET=1" in cmd


def test_docker_cmd_dynamic_xet_policy_overrides_endpoint_extra_arg():
    cfg = _config(sglang_docker_args=(
        "--cgroupns=host",
        "-e", "HF_HUB_DISABLE_XET=1",
        "--env", "HF_HUB_DISABLE_XET=1",
        "--env=HF_HUB_DISABLE_XET=1",
        "-eHF_HUB_DISABLE_XET=1",
    ))
    enabled_cmd = _build_docker_run_cmd(
        _target(), cfg, disable_hf_xet=False,
    )
    assert "--cgroupns=host" in enabled_cmd
    assert "HF_HUB_DISABLE_XET" not in enabled_cmd

    disabled_cmd = _build_docker_run_cmd(
        _target(), cfg, disable_hf_xet=True,
    )
    assert "--cgroupns=host" in disabled_cmd
    assert disabled_cmd.count("HF_HUB_DISABLE_XET=1") == 1


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


def test_docker_cmd_includes_configured_extra_docker_args():
    cmd = _build_docker_run_cmd(
        _target(),
        _config(sglang_docker_args=("--cgroupns=host",)),
    )
    assert "--cgroupns=host" in cmd


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


@dataclass
class _Sibling:
    rfilename: str
    size: int | None


def test_max_weight_file_size_uses_largest_single_weight_shard():
    siblings = [
        _Sibling("config.json", 123),
        _Sibling("model-00001-of-00003.safetensors", 10),
        _Sibling("model-00002-of-00003.safetensors", 40),
        _Sibling("model-00003-of-00003.safetensors", 30),
    ]
    assert _max_weight_file_size_bytes(siblings) == 40


def test_max_weight_file_size_ignores_missing_sizes_and_non_weights():
    siblings = [
        _Sibling("README.md", 99),
        _Sibling("model.safetensors.index.json", 100),
        _Sibling("pytorch_model.bin", None),
    ]
    assert _max_weight_file_size_bytes(siblings) is None


@pytest.mark.asyncio
async def test_should_disable_hf_xet_for_sub_50g_max_shard(monkeypatch):
    class _Info:
        siblings = [_Sibling(
            "model.safetensors",
            HF_XET_MAX_NON_XET_FILE_BYTES - 1,
        )]

    class _Api:
        def __init__(self, token=None):
            self.token = token

        def model_info(self, **kwargs):
            assert kwargs["repo_id"] == _target().model
            assert kwargs["revision"] == _target().revision
            assert kwargs["files_metadata"] is True
            assert kwargs["timeout"] == DEFAULT_HF_METADATA_TIMEOUT_SEC
            return _Info()

    monkeypatch.setattr("affine.src.scheduler.ssh.HfApi", _Api)
    assert await _should_disable_hf_xet(_target())


@pytest.mark.asyncio
async def test_should_disable_hf_xet_uses_configured_metadata_timeout(
    monkeypatch,
):
    class _Info:
        siblings = [_Sibling("model.safetensors", 1)]

    class _Api:
        def __init__(self, token=None):
            self.token = token

        def model_info(self, **kwargs):
            assert kwargs["timeout"] == 7.5
            return _Info()

    monkeypatch.setenv("AFFINE_SSH_HF_METADATA_TIMEOUT_SEC", "7.5")
    monkeypatch.setattr("affine.src.scheduler.ssh.HfApi", _Api)

    assert await _should_disable_hf_xet(_target())


@pytest.mark.asyncio
async def test_should_keep_hf_xet_for_50g_or_larger_max_shard(monkeypatch):
    class _Info:
        siblings = [_Sibling(
            "model.safetensors",
            HF_XET_MAX_NON_XET_FILE_BYTES,
        )]

    class _Api:
        def __init__(self, token=None):
            self.token = token

        def model_info(self, **kwargs):
            return _Info()

    monkeypatch.setattr("affine.src.scheduler.ssh.HfApi", _Api)
    assert not await _should_disable_hf_xet(_target())


@pytest.mark.asyncio
async def test_should_keep_hf_xet_when_metadata_lookup_fails(monkeypatch):
    class _Api:
        def __init__(self, token=None):
            self.token = token

        def model_info(self, **kwargs):
            raise RuntimeError("hf unavailable")

    monkeypatch.setattr("affine.src.scheduler.ssh.HfApi", _Api)
    assert not await _should_disable_hf_xet(_target())


def test_parse_container_exit_status_tolerates_targon_banner():
    assert _parse_container_exit_status("exited 1\n") == 1
    assert _parse_container_exit_status(
        "Connecting to container wrk-abc...\nexited 1\n"
    ) == 1
    assert _parse_container_exit_status(
        "Connecting to container wrk-abc...\nrunning 0\n"
    ) is None
    assert _parse_container_exit_status("Connecting to container wrk-abc...\n") is None


def test_parse_container_inspect_json_tolerates_targon_banner():
    payload = {"State": {"Status": "running"}, "Config": {"Labels": {}}}
    assert _parse_container_inspect_json(
        "Connecting to container wrk-abc...\n" + json.dumps(payload) + "\n"
    ) == payload
    assert _parse_container_inspect_json("Connecting only\n") is None


@pytest.mark.asyncio
async def test_deployment_healthy_requires_matching_running_container(monkeypatch):
    target = _target()
    cfg = _config(endpoint_name="ssh_b300")
    labels = {
        "io.affine.endpoint": "ssh_b300",
        "io.affine.uid": str(target.uid),
        "io.affine.hotkey": target.hotkey,
        "io.affine.model": target.model,
        "io.affine.revision": target.revision,
    }
    inspect_payload = {
        "State": {"Status": "running", "ExitCode": 0},
        "Config": {"Labels": labels},
    }
    probed = []

    async def fake_ssh_exec(config, command):
        assert command.startswith("docker inspect")
        return 0, "Connecting to container wrk-abc...\n" + json.dumps(inspect_payload), ""

    async def fake_probe_ready(base_url, *, timeout_sec=5.0):
        probed.append(base_url)
        return True

    monkeypatch.setattr("affine.src.scheduler.ssh._ssh_exec", fake_ssh_exec)
    monkeypatch.setattr("affine.src.scheduler.ssh._probe_ready", fake_probe_ready)

    assert await deployment_healthy(
        cfg, target, base_url="http://edge.example/v1",
    )
    assert probed == ["http://edge.example/v1"]


@pytest.mark.asyncio
async def test_deployment_healthy_rejects_wrong_model_label(monkeypatch):
    target = _target()
    cfg = _config(endpoint_name="ssh_b300")
    inspect_payload = {
        "State": {"Status": "running", "ExitCode": 0},
        "Config": {
            "Labels": {
                "io.affine.endpoint": "ssh_b300",
                "io.affine.uid": str(target.uid),
                "io.affine.hotkey": target.hotkey,
                "io.affine.model": "other/model",
                "io.affine.revision": target.revision,
            }
        },
    }

    async def fake_ssh_exec(config, command):
        return 0, json.dumps(inspect_payload), ""

    async def fake_probe_ready(base_url, *, timeout_sec=5.0):
        raise AssertionError("readiness probe should not run after label mismatch")

    monkeypatch.setattr("affine.src.scheduler.ssh._ssh_exec", fake_ssh_exec)
    monkeypatch.setattr("affine.src.scheduler.ssh._probe_ready", fake_probe_ready)

    assert not await deployment_healthy(cfg, target)


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
    assert "--load-balance-method" not in args


def test_invalid_dp_load_balance_method_raises():
    with pytest.raises(ValueError, match="unsupported SGLang load-balance"):
        _build_sglang_args(
            _target(),
            _config(sglang_load_balance_method="least_connections"),
        )


def _qwen36_target():
    return DeployTarget(
        uid=7, hotkey="qwen36hot", model="Qwen/Qwen3.6-35B-A3B",
        revision="deadbeef0011", model_type="qwen3_5_moe",
    )


def test_qwen36_adds_reasoning_and_coder_parser():
    """Qwen3.6 (qwen3_5_moe) needs reasoning-parser and the qwen3_coder tool
    parser, or it serves broken output. Context-length stays auto-derived."""
    args = _build_sglang_args(_qwen36_target(), _config())
    assert args[args.index("--reasoning-parser") + 1] == "qwen3"
    assert args[args.index("--tool-call-parser") + 1] == "qwen3_coder"
    assert "--context-length" not in args


def test_dense_qwen3_keeps_legacy_flags():
    """Dense qwen3 (and unknown model_type) must NOT get the qwen3.6 flags."""
    args = _build_sglang_args(_target(), _config())  # _target() has no model_type
    assert "--reasoning-parser" not in args
    assert "--context-length" not in args
    assert args[args.index("--tool-call-parser") + 1] == "qwen"


def test_config_from_endpoint_overrides_module_defaults():
    """Per-endpoint sglang settings flow through into the docker command."""
    ep = Endpoint(
        name="ep1", kind="ssh",
        ssh_url="ssh://root@1.2.3.4:10300",
        public_inference_url="http://val:31000/v1",
        sglang_port=31000, sglang_dp=4,
        sglang_load_balance_method="total_requests",
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
    assert (
        args[args.index("--load-balance-method") + 1]
        == "total_requests"
    )
    assert "--tool-call-parser" not in args
    # inference_url uses the public override, not host:port
    assert cfg.inference_url() == "http://val:31000/v1"


@pytest.mark.asyncio
async def test_deploy_always_cleans_existing_container_before_start(monkeypatch):
    calls = []

    async def fake_ssh_exec(config, command):
        calls.append(command)
        # Cache cleanup script: keep-only-target shell loop.
        if "for d in" in command and "rm -rf" in command:
            return 0, "", ""
        if command.startswith(f"docker rm -f {CONTAINER_NAME}"):
            return 0, "container1234567890", ""
        raise AssertionError(f"unexpected command: {command}")

    async def fake_wait_ready(*args, **kwargs):
        return None

    monkeypatch.setattr("affine.src.scheduler.ssh._ssh_exec", fake_ssh_exec)
    monkeypatch.setattr("affine.src.scheduler.ssh._wait_ready", fake_wait_ready)
    async def fake_should_disable_hf_xet(target):
        return False

    monkeypatch.setattr(
        "affine.src.scheduler.ssh._should_disable_hf_xet",
        fake_should_disable_hf_xet,
    )

    cfg = _config(endpoint_name="ssh_b300")
    result = await deploy(cfg, _target())

    assert result.deployment_id == f"ssh:ssh_b300:{CONTAINER_NAME}"
    assert result.base_url == "http://b300:10001/v1"
    # No label-based adoption: every deploy runs startup cleanup first.
    assert len(calls) == 2
    assert "docker inspect" not in "\n".join(calls)
    assert "for d in" in calls[0] and "rm -rf" in calls[0]
    assert calls[1].startswith(f"docker rm -f {CONTAINER_NAME}")


@pytest.mark.asyncio
async def test_wait_ready_raises_early_on_container_exit(monkeypatch):
    """Sglang can crash on startup (transient HF blip, bad config, OOM)
    seconds after ``docker run``. ``_wait_ready`` polls container status
    next to the HTTP probe and raises immediately on a non-running
    container — without this it would silently spin until the full
    ``ready_timeout_sec`` (30 min default), blocking the assigned endpoint."""
    from affine.src.scheduler.ssh import _wait_ready

    async def fake_probe_ready(base_url, *, timeout_sec=5.0):
        return False

    async def fake_ssh_exec(config, command):
        if command.startswith("docker inspect"):
            return 0, "Connecting to container wrk-abc...\nexited 1\n", ""
        raise AssertionError(f"unexpected: {command}")

    monkeypatch.setattr("affine.src.scheduler.ssh._probe_ready", fake_probe_ready)
    monkeypatch.setattr("affine.src.scheduler.ssh._ssh_exec", fake_ssh_exec)

    cfg = _config()
    with pytest.raises(RuntimeError, match="exited with code 1"):
        await _wait_ready(
            "http://b300:30000/v1",
            deadline_sec=60,
            poll_interval_sec=0.01,
            config=cfg,
        )


@pytest.mark.asyncio
async def test_wait_ready_does_not_probe_container_when_config_omitted(monkeypatch):
    """Backward-compat: call sites that don't pass ``config`` skip the
    container poll (legacy behavior — pure HTTP wait until timeout)."""
    from affine.src.scheduler.ssh import _wait_ready

    async def fake_probe_ready(base_url, *, timeout_sec=5.0):
        return False

    monkeypatch.setattr("affine.src.scheduler.ssh._probe_ready", fake_probe_ready)

    with pytest.raises(TimeoutError):
        await _wait_ready(
            "http://b300:30000/v1",
            deadline_sec=0.05,
            poll_interval_sec=0.01,
        )
