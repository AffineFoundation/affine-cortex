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
    assert (
        result["public_inference_url"]
        == "https://wrk-test-10001.caas.targon.com/v1"
    )
    assert calls[:2] == [
        ("POST", "/workloads", calls[0][2]),
        ("POST", "/workloads/wrk-test/deploy", None),
    ]


def test_create_cleans_up_workload_when_wait_ready_fails(monkeypatch):
    wrapper = _load_wrapper(monkeypatch)
    calls = []

    def request(self, method, path, **kwargs):
        calls.append((method, path))
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


def test_delete_missing_workload_is_idempotent(monkeypatch):
    wrapper = _load_wrapper(monkeypatch)

    def request(self, method, path, **kwargs):
        raise wrapper.TargonNotFound(f"{method} {path} -> 404")

    monkeypatch.setattr(wrapper.TargonAutoscaleClient, "request", request)

    assert wrapper.TargonAutoscaleClient().delete("wrk-missing") is True


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
