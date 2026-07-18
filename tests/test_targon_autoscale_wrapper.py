from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "targon_autoscale_wrapper.py"


def _load_wrapper(monkeypatch, **env):
    defaults = {
        "TARGON_API_KEY": "test-key",
        "TARGON_RENTAL_IMAGE": "example/dind:latest",
        "TARGON_SSH_KEY_UID": "shk-test",
        "TARGON_REQUIRE_SSH_PROBE": "false",
        "TARGON_CREATE_WAIT_TIMEOUT_SEC": "1",
        "TARGON_CREATE_POLL_SECONDS": "0",
    }
    for key, value in {**defaults, **env}.items():
        monkeypatch.setenv(key, value)
    module_name = f"_targon_autoscale_wrapper_{len(sys.modules)}"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_create_registers_deploys_and_returns_provider_handle(monkeypatch):
    wrapper = _load_wrapper(monkeypatch)
    calls = []

    def request(self, method, path, **kwargs):
        calls.append((method, path, kwargs.get("json")))
        if (method, path) == ("GET", "/workloads"):
            return {"workloads": []}
        if (method, path) == ("POST", "/workloads"):
            body = kwargs["json"]
            assert body["name"] == "affine-autoscale-a"
            assert body["image"] == "example/dind:latest"
            assert body["resource_name"] == "b200-xlarge"
            assert body["type"] == "RENTAL"
            assert body["ssh_keys"] == ["shk-test"]
            assert body["ports"] == [
                {"port": 10001, "protocol": "TCP", "routing": "PROXIED"},
            ]
            return {"uid": "wrk-test"}
        if (method, path) == ("POST", "/workloads/wrk-test/deploy"):
            return {"uid": "wrk-test"}
        if (method, path) == ("GET", "/workloads/wrk-test"):
            return {"uid": "wrk-test", "name": "affine-autoscale-a"}
        if (method, path) == ("GET", "/workloads/wrk-test/state"):
            return {
                "status": "RUNNING",
                "ready_replicas": 1,
                "urls": [
                    {
                        "port": 10001,
                        "url": "https://wrk-test-10001.caas.targon.com",
                    }
                ],
            }
        raise AssertionError((method, path))

    monkeypatch.setattr(wrapper.TargonAutoscaleClient, "request", request)

    result = wrapper.TargonAutoscaleClient().create("affine-autoscale-a")

    assert result["instance_id"] == "wrk-test"
    assert result["ssh_url"] == "ssh://wrk-test@ssh.deployments.targon.com:22"
    assert result["public_inference_url"] == "https://wrk-test-10001.caas.targon.com/v1"
    assert result["sglang_port"] == 10001
    assert calls[:3] == [
        ("GET", "/workloads", None),
        ("POST", "/workloads", calls[1][2]),
        ("POST", "/workloads/wrk-test/deploy", None),
    ]


def test_create_adopts_existing_workload_by_name_without_registering(
    monkeypatch,
):
    wrapper = _load_wrapper(monkeypatch)
    calls = []

    def request(self, method, path, **kwargs):
        calls.append((method, path))
        if (method, path) == ("GET", "/workloads"):
            return {
                "workloads": [
                    {
                        "uid": "wrk-existing",
                        "name": "affine-autoscale-eval-a",
                        "status": "RUNNING",
                    }
                ]
            }
        if (method, path) == ("GET", "/workloads/wrk-existing"):
            return {
                "uid": "wrk-existing",
                "name": "affine-autoscale-eval-a",
                "status": "RUNNING",
            }
        if (method, path) == ("GET", "/workloads/wrk-existing/state"):
            return {
                "status": "RUNNING",
                "ready_replicas": 1,
                "urls": [
                    {
                        "port": 10001,
                        "url": "https://wrk-existing-10001.caas.targon.com",
                    }
                ],
            }
        raise AssertionError((method, path))

    monkeypatch.setattr(wrapper.TargonAutoscaleClient, "request", request)

    result = wrapper.TargonAutoscaleClient().create("affine-autoscale-eval-a")

    assert result["instance_id"] == "wrk-existing"
    assert ("POST", "/workloads") not in calls
    assert ("POST", "/workloads/wrk-existing/deploy") not in calls


def test_status_returns_normalized_workload_state(monkeypatch):
    wrapper = _load_wrapper(monkeypatch)
    calls = []

    def request(self, method, path, **kwargs):
        calls.append((method, path))
        if (method, path) == ("GET", "/workloads/wrk-test"):
            return {"uid": "wrk-test", "name": "affine-autoscale-a"}
        if (method, path) == ("GET", "/workloads/wrk-test/state"):
            return {
                "status": "RUNNING",
                "ready_replicas": 1,
                "urls": [
                    {
                        "port": 10001,
                        "url": "https://wrk-test-10001.caas.targon.com",
                    }
                ],
            }
        raise AssertionError((method, path))

    monkeypatch.setattr(wrapper.TargonAutoscaleClient, "request", request)

    result = wrapper.TargonAutoscaleClient().status("wrk-test")

    assert result["instance_id"] == "wrk-test"
    assert result["status"] == "running"
    assert result["ssh_url"] == "ssh://wrk-test@ssh.deployments.targon.com:22"
    assert result["public_inference_url"] == "https://wrk-test-10001.caas.targon.com/v1"
    assert result["sglang_port"] == 10001
    assert calls == [
        ("GET", "/workloads/wrk-test"),
        ("GET", "/workloads/wrk-test/state"),
    ]


def test_create_deploys_adopted_workload_when_left_registered(
    monkeypatch,
):
    wrapper = _load_wrapper(monkeypatch)
    calls = []
    state_calls = 0

    def request(self, method, path, **kwargs):
        nonlocal state_calls
        calls.append((method, path))
        if (method, path) == ("GET", "/workloads"):
            return {
                "workloads": [
                    {
                        "uid": "wrk-existing",
                        "name": "affine-autoscale-eval-a",
                        "status": "CREATED",
                    }
                ]
            }
        if (method, path) == ("GET", "/workloads/wrk-existing"):
            return {
                "uid": "wrk-existing",
                "name": "affine-autoscale-eval-a",
                "status": "CREATED",
            }
        if (method, path) == ("GET", "/workloads/wrk-existing/state"):
            state_calls += 1
            if state_calls == 1:
                return {"status": "CREATED", "ready_replicas": 0}
            return {
                "status": "RUNNING",
                "ready_replicas": 1,
                "urls": [
                    {
                        "port": 10001,
                        "url": "https://wrk-existing-10001.caas.targon.com",
                    }
                ],
            }
        if (method, path) == ("POST", "/workloads/wrk-existing/deploy"):
            return {"uid": "wrk-existing"}
        raise AssertionError((method, path))

    monkeypatch.setattr(wrapper.TargonAutoscaleClient, "request", request)

    result = wrapper.TargonAutoscaleClient().create("affine-autoscale-eval-a")

    assert result["instance_id"] == "wrk-existing"
    assert ("POST", "/workloads") not in calls
    assert ("POST", "/workloads/wrk-existing/deploy") in calls


def test_create_adopts_existing_workload_when_register_response_fails(
    monkeypatch,
):
    wrapper = _load_wrapper(monkeypatch)
    calls = []
    listed = False

    def request(self, method, path, **kwargs):
        nonlocal listed
        calls.append((method, path))
        if (method, path) == ("GET", "/workloads") and not listed:
            listed = True
            return {"workloads": []}
        if (method, path) == ("POST", "/workloads"):
            raise wrapper.TargonWrapperError("500 after creating workload")
        if (method, path) == ("GET", "/workloads"):
            return {
                "workloads": [
                    {
                        "uid": "wrk-orphan",
                        "name": "affine-autoscale-eval-a",
                        "status": "RUNNING",
                    }
                ]
            }
        if (method, path) == ("GET", "/workloads/wrk-orphan"):
            return {"uid": "wrk-orphan", "status": "RUNNING"}
        if (method, path) == ("GET", "/workloads/wrk-orphan/state"):
            return {"status": "RUNNING", "ready_replicas": 1}
        raise AssertionError((method, path))

    monkeypatch.setattr(wrapper.TargonAutoscaleClient, "request", request)

    result = wrapper.TargonAutoscaleClient().create("affine-autoscale-eval-a")

    assert result["instance_id"] == "wrk-orphan"
    assert calls.count(("POST", "/workloads")) == 1


def test_create_cleans_up_workload_when_wait_ready_fails(monkeypatch):
    wrapper = _load_wrapper(monkeypatch)
    calls = []

    def request(self, method, path, **kwargs):
        calls.append((method, path))
        if (method, path) == ("GET", "/workloads"):
            return {"workloads": []}
        if (method, path) == ("POST", "/workloads"):
            return {"uid": "wrk-leaked"}
        if (method, path) == ("POST", "/workloads/wrk-leaked/deploy"):
            return {"uid": "wrk-leaked"}
        if (method, path) == ("DELETE", "/workloads/wrk-leaked"):
            return {}
        raise AssertionError((method, path))

    def wait_ready(self, uid):
        raise wrapper.TargonWrapperError(f"{uid} never became ready")

    monkeypatch.setattr(wrapper.TargonAutoscaleClient, "request", request)
    monkeypatch.setattr(wrapper.TargonAutoscaleClient, "wait_ready", wait_ready)

    with pytest.raises(wrapper.TargonWrapperError):
        wrapper.TargonAutoscaleClient().create("affine-autoscale-a")

    assert ("DELETE", "/workloads/wrk-leaked") in calls


def test_wait_ready_fails_after_running_ssh_preflight_failures(monkeypatch):
    wrapper = _load_wrapper(
        monkeypatch,
        TARGON_REQUIRE_SSH_PROBE="true",
        TARGON_SSH_READY_FAILURE_LIMIT="2",
    )
    probes = []

    def request(self, method, path, **kwargs):
        if (method, path) == ("GET", "/workloads/wrk-bad"):
            return {"uid": "wrk-bad", "status": "RUNNING"}
        if (method, path) == ("GET", "/workloads/wrk-bad/state"):
            return {"status": "RUNNING", "ready_replicas": 1}
        raise AssertionError((method, path))

    def ssh_ready(self, uid):
        probes.append(uid)
        return False

    monkeypatch.setattr(wrapper.TargonAutoscaleClient, "request", request)
    monkeypatch.setattr(wrapper.TargonAutoscaleClient, "_ssh_ready", ssh_ready)

    with pytest.raises(wrapper.TargonWrapperError, match="SSH preflight failed"):
        wrapper.TargonAutoscaleClient().wait_ready("wrk-bad")

    assert probes == ["wrk-bad", "wrk-bad"]


def test_create_falls_back_across_resource_names(monkeypatch):
    wrapper = _load_wrapper(
        monkeypatch,
        TARGON_RESOURCE_NAMES="b200-xlarge,b300-xlarge",
    )
    resources = []

    def request(self, method, path, **kwargs):
        if (method, path) == ("GET", "/workloads"):
            return {"workloads": []}
        if (method, path) == ("POST", "/workloads"):
            resource = kwargs["json"]["resource_name"]
            resources.append(resource)
            if resource == "b200-xlarge":
                raise wrapper.TargonWrapperError("no b200 capacity")
            return {"uid": "wrk-b300"}
        if (method, path) == ("POST", "/workloads/wrk-b300/deploy"):
            return {"uid": "wrk-b300"}
        if (method, path) == ("GET", "/workloads/wrk-b300"):
            return {"uid": "wrk-b300", "name": "affine-autoscale-a"}
        if (method, path) == ("GET", "/workloads/wrk-b300/state"):
            return {
                "status": "RUNNING",
                "ready_replicas": 1,
                "urls": [
                    {
                        "port": 10001,
                        "url": "https://wrk-b300-10001.caas.targon.com",
                    }
                ],
            }
        raise AssertionError((method, path))

    monkeypatch.setattr(wrapper.TargonAutoscaleClient, "request", request)

    result = wrapper.TargonAutoscaleClient().create("affine-autoscale-a")

    assert resources == ["b200-xlarge", "b300-xlarge"]
    assert result["instance_id"] == "wrk-b300"
    assert result["resource_name"] == "b300-xlarge"


def test_create_accepts_json_resource_names(monkeypatch):
    wrapper = _load_wrapper(
        monkeypatch,
        TARGON_RESOURCE_NAMES='["b200-xlarge", "b300-xlarge"]',
    )

    body = wrapper.TargonAutoscaleClient()._create_body(
        "affine-autoscale-a",
        wrapper.RESOURCE_NAMES[0],
    )

    assert wrapper.RESOURCE_NAMES == ["b200-xlarge", "b300-xlarge"]
    assert body["resource_name"] == "b200-xlarge"


def test_delete_refuses_non_autoscaler_workload(monkeypatch):
    wrapper = _load_wrapper(monkeypatch)
    calls = []

    def request(self, method, path, **kwargs):
        calls.append((method, path))
        if (method, path) == ("GET", "/workloads/wrk-manual"):
            return {"uid": "wrk-manual", "name": "manual-prod-gpu"}
        raise AssertionError((method, path))

    monkeypatch.setattr(wrapper.TargonAutoscaleClient, "request", request)

    with pytest.raises(wrapper.TargonWrapperError):
        wrapper.TargonAutoscaleClient().delete("wrk-manual")

    assert ("DELETE", "/workloads/wrk-manual") not in calls


def test_delete_refuses_autoscaler_workload_with_wrong_purpose(monkeypatch):
    wrapper = _load_wrapper(monkeypatch)
    calls = []

    def request(self, method, path, **kwargs):
        calls.append((method, path))
        if (method, path) == ("GET", "/workloads/wrk-bench"):
            return {"uid": "wrk-bench", "name": "affine-autoscale-bench-a"}
        raise AssertionError((method, path))

    monkeypatch.setattr(wrapper.TargonAutoscaleClient, "request", request)

    with pytest.raises(wrapper.TargonWrapperError, match="purpose=eval"):
        wrapper.TargonAutoscaleClient().delete(
            "wrk-bench",
            expected_purpose="eval",
        )

    assert ("DELETE", "/workloads/wrk-bench") not in calls


def test_delete_allows_matching_purpose_and_legacy_without_expected_purpose(
    monkeypatch,
):
    wrapper = _load_wrapper(monkeypatch)
    calls = []

    def request(self, method, path, **kwargs):
        calls.append((method, path))
        if (method, path) == ("GET", "/workloads/wrk-eval"):
            return {"uid": "wrk-eval", "name": "affine-autoscale-eval-a"}
        if (method, path) == ("GET", "/workloads/wrk-legacy"):
            return {"uid": "wrk-legacy", "name": "affine-autoscale-a"}
        if method == "DELETE":
            return {}
        raise AssertionError((method, path))

    monkeypatch.setattr(wrapper.TargonAutoscaleClient, "request", request)

    assert (
        wrapper.TargonAutoscaleClient().delete(
            "wrk-eval",
            expected_purpose="eval",
        )
        is True
    )
    assert wrapper.TargonAutoscaleClient().delete("wrk-legacy") is True
    assert ("DELETE", "/workloads/wrk-eval") in calls
    assert ("DELETE", "/workloads/wrk-legacy") in calls


def test_resource_name_includes_purpose_and_keeps_targon_length_limit(
    monkeypatch,
):
    wrapper = _load_wrapper(monkeypatch)

    name = wrapper._resource_name(
        purpose="Bench",
        suffix="targon-b200-autoscale-primary-extra-long",
        max_len=32,
    )

    assert name.startswith("affine-autoscale-bench-")
    assert len(name) <= 32


def test_delete_missing_workload_is_idempotent(monkeypatch):
    wrapper = _load_wrapper(monkeypatch)

    def request(self, method, path, **kwargs):
        raise wrapper.TargonNotFound(f"{method} {path} -> 404")

    monkeypatch.setattr(wrapper.TargonAutoscaleClient, "request", request)

    assert wrapper.TargonAutoscaleClient().delete("wrk-missing") is True


def test_error_payloads_distinguish_wrapper_and_provider_not_found(monkeypatch):
    wrapper = _load_wrapper(monkeypatch)

    route_payload = wrapper._route_not_found("POST", "/bad")
    provider_payload = wrapper._provider_not_found(
        wrapper.TargonNotFound("GET /workloads/wrk-missing -> 404")
    )

    assert route_payload["error_source"] == "wrapper"
    assert route_payload["code"] == "route_not_found"
    assert provider_payload["error_source"] == "provider"
    assert provider_payload["provider_status_code"] == 404


def test_error_response_structures_provider_and_wrapper_failures(monkeypatch):
    wrapper = _load_wrapper(monkeypatch)

    status, payload = wrapper._error_response(
        wrapper.TargonHTTPError("GET", "/workloads/wrk-1", 403, "denied")
    )
    assert status == 403
    assert payload["error_source"] == "provider"
    assert payload["code"] == "provider_http_error"
    assert payload["provider_status_code"] == 403
    assert payload["provider_path"] == "/workloads/wrk-1"

    status, payload = wrapper._error_response(
        wrapper.TargonRequestError(
            "GET",
            "/workloads",
            TimeoutError("slow"),
            code="provider_timeout",
            http_status=504,
        )
    )
    assert status == 504
    assert payload["error_source"] == "provider"
    assert payload["code"] == "provider_timeout"

    status, payload = wrapper._error_response(
        wrapper.TargonWrapperError(
            "TARGON_API_KEY is required",
            code="wrapper_config_error",
        )
    )
    assert status == 500
    assert payload["error_source"] == "wrapper"
    assert payload["code"] == "wrapper_config_error"

    status, payload = wrapper._error_response(RuntimeError("boom"))
    assert status == 500
    assert payload["error_source"] == "wrapper"
    assert payload["code"] == "wrapper_internal_error"


def test_renew_extends_configured_logical_lease(monkeypatch):
    wrapper = _load_wrapper(monkeypatch, TARGON_LEASE_HOURS="8")
    monkeypatch.setattr(wrapper.time, "time", lambda: 1_000)
    calls = []

    def request(self, method, path, **kwargs):
        calls.append((method, path, kwargs.get("json")))
        if (method, path) == ("GET", "/workloads/wrk-test"):
            return {"uid": "wrk-test", "name": "affine-autoscale-a"}
        if (method, path) == ("GET", "/workloads/wrk-test/state"):
            return {"status": "RUNNING", "ready_replicas": 1}
        raise AssertionError((method, path))

    monkeypatch.setattr(wrapper.TargonAutoscaleClient, "request", request)

    result = wrapper.TargonAutoscaleClient().renew("wrk-test")

    assert result["instance_id"] == "wrk-test"
    assert result["lease_expires_at"] == 1_000 + 8 * 60 * 60
    assert result["status"] == "running"
    assert calls == [
        ("GET", "/workloads/wrk-test", None),
        ("GET", "/workloads/wrk-test/state", None),
    ]


def test_renew_calls_configured_provider_endpoint(monkeypatch):
    wrapper = _load_wrapper(
        monkeypatch,
        TARGON_LEASE_HOURS="2",
        TARGON_RENEW_METHOD="PATCH",
        TARGON_RENEW_PATH_TEMPLATE="/workloads/{uid}",
        TARGON_RENEW_PAYLOAD_JSON=(
            '{"envs":[{"name":"AFFINE_RENEW_UNTIL","value":"{lease_expires_iso}"}]}'
        ),
    )
    monkeypatch.setattr(wrapper.time, "time", lambda: 1_000)
    calls = []

    def request(self, method, path, **kwargs):
        calls.append((method, path, kwargs.get("json")))
        if (method, path) == ("GET", "/workloads/wrk-test"):
            return {"uid": "wrk-test", "name": "affine-autoscale-a"}
        if (method, path) == ("GET", "/workloads/wrk-test/state"):
            return {"status": "RUNNING", "ready_replicas": 1}
        if (method, path) == ("PATCH", "/workloads/wrk-test"):
            assert kwargs["json"] == {
                "envs": [
                    {
                        "name": "AFFINE_RENEW_UNTIL",
                        "value": "1970-01-01T02:16:40Z",
                    }
                ]
            }
            return {"expires_at": "1970-01-01T03:00:00Z"}
        raise AssertionError((method, path))

    monkeypatch.setattr(wrapper.TargonAutoscaleClient, "request", request)

    result = wrapper.TargonAutoscaleClient().renew("wrk-test")

    assert result["lease_expires_at"] == 10_800
    assert calls == [
        ("GET", "/workloads/wrk-test", None),
        ("GET", "/workloads/wrk-test/state", None),
        (
            "PATCH",
            "/workloads/wrk-test",
            {
                "envs": [
                    {
                        "name": "AFFINE_RENEW_UNTIL",
                        "value": "1970-01-01T02:16:40Z",
                    }
                ]
            },
        ),
        ("GET", "/workloads/wrk-test/state", None),
    ]


def test_renew_refuses_non_autoscaler_workload(monkeypatch):
    wrapper = _load_wrapper(monkeypatch, TARGON_LEASE_HOURS="8")

    def request(self, method, path, **kwargs):
        if (method, path) == ("GET", "/workloads/wrk-manual"):
            return {"uid": "wrk-manual", "name": "manual-prod-gpu"}
        raise AssertionError((method, path))

    monkeypatch.setattr(wrapper.TargonAutoscaleClient, "request", request)

    with pytest.raises(wrapper.TargonWrapperError, match="Refusing to renew"):
        wrapper.TargonAutoscaleClient().renew("wrk-manual")


def test_resolves_ssh_key_by_name(monkeypatch):
    wrapper = _load_wrapper(
        monkeypatch,
        TARGON_SSH_KEY_UID="",
        TARGON_SSH_KEY_NAME="online",
    )

    def request(self, method, path, **kwargs):
        assert (method, path) == ("GET", "/ssh-keys")
        return {"items": [{"uid": "shk-online", "name": "online"}]}

    monkeypatch.setattr(wrapper.TargonAutoscaleClient, "request", request)

    body = wrapper.TargonAutoscaleClient()._create_body("affine-autoscale-a")

    assert body["ssh_keys"] == ["shk-online"]


def test_ssh_ready_runs_gpu_preflight_and_no_cgroups_fix(monkeypatch):
    wrapper = _load_wrapper(
        monkeypatch,
        TARGON_REQUIRE_SSH_PROBE="true",
        TARGON_REQUIRE_DOCKER="true",
        TARGON_REQUIRE_DOCKER_GPU="true",
        TARGON_FIX_NVIDIA_NO_CGROUPS="true",
        TARGON_SSH_KEY_PATH="/tmp/test-key",
        TARGON_SSH_PROBE_TIMEOUT_SEC="123",
    )
    calls = []

    class Result:
        returncode = 0
        stdout = "ssh-ready\nready\n"
        stderr = ""

    def run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return Result()

    monkeypatch.setattr(wrapper.subprocess, "run", run)

    assert wrapper.TargonAutoscaleClient()._ssh_ready("wrk-test") is True

    cmd, kwargs = calls[0]
    probe = cmd[-1]
    assert cmd[:3] == ["ssh", "-i", "/tmp/test-key"]
    assert "wrk-test@ssh.deployments.targon.com" in cmd
    assert kwargs["timeout"] == 123
    assert "no-cgroups = true" in probe
    assert "docker run --rm --gpus all" in probe
    assert "nvidia/cuda:12.4.1-base-ubuntu22.04" in probe
    assert "nvidia-smi --query-gpu=index,name,memory.total" in probe
    assert "Fabric" in probe
    assert "libcuda.so.1" in probe
    assert "cuDeviceGetCount" in probe


def test_ssh_probe_can_disable_gpu_preflight(monkeypatch):
    wrapper = _load_wrapper(
        monkeypatch,
        TARGON_REQUIRE_DOCKER="true",
        TARGON_REQUIRE_DOCKER_GPU="false",
    )

    probe = wrapper._ssh_probe_command()

    assert "docker version" in probe
    assert "--gpus all" not in probe
    assert "no-cgroups" not in probe
    assert "libcuda.so.1" not in probe
    assert "Fabric" not in probe


def test_ssh_probe_can_disable_cuda_and_fabric_checks(monkeypatch):
    wrapper = _load_wrapper(
        monkeypatch,
        TARGON_REQUIRE_DOCKER="true",
        TARGON_REQUIRE_DOCKER_GPU="true",
        TARGON_REQUIRE_CUDA_DRIVER="false",
        TARGON_REQUIRE_FABRIC_READY="false",
    )

    probe = wrapper._ssh_probe_command()

    assert "--gpus all" in probe
    assert "libcuda.so.1" not in probe
    assert "Fabric" not in probe
