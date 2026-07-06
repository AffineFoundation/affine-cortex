from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_wrapper():
    path = (
        Path(__file__).resolve().parents[1] / "scripts" / ("lium_autoscale_wrapper.py")
    )
    spec = importlib.util.spec_from_file_location("lium_autoscale_wrapper", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_delete_refuses_pods_without_owned_name_prefix():
    lium = _load_wrapper()
    client = lium.LiumClient.__new__(lium.LiumClient)
    calls = []

    def request(method, path, **kwargs):
        if method == "GET":
            return {"uid": "pod-1", "pod_name": "manual-debug-pod"}
        calls.append((method, path))
        return {}

    client.request = request

    with pytest.raises(lium.LiumWrapperError, match="Refusing to delete"):
        client.delete("pod-1")

    assert calls == []


def test_delete_refuses_owned_pod_with_wrong_purpose():
    lium = _load_wrapper()
    client = lium.LiumClient.__new__(lium.LiumClient)
    calls = []

    def request(method, path, **kwargs):
        if method == "GET":
            return {"uid": "pod-1", "pod_name": "affine-autoscale-bench-a"}
        calls.append((method, path))
        return {}

    client.request = request

    with pytest.raises(lium.LiumWrapperError, match="purpose=eval"):
        client.delete("pod-1", expected_purpose="eval")

    assert calls == []


def test_delete_allows_matching_purpose_and_legacy_without_expected_purpose():
    lium = _load_wrapper()
    client = lium.LiumClient.__new__(lium.LiumClient)
    pods = [
        {"uid": "pod-1", "pod_name": "affine-autoscale-eval-a"},
        {"uid": "pod-2", "pod_name": "affine-autoscale-a"},
    ]
    deleted = []

    def request(method, path, **kwargs):
        if method == "GET":
            return pods.pop(0)
        deleted.append(path)
        return {}

    client.request = request

    assert client.delete("pod-1", expected_purpose="eval") is True
    assert client.delete("pod-2") is True
    assert deleted == ["/pods/pod-1", "/pods/pod-2"]


def test_resource_name_includes_purpose_and_keeps_length_limit():
    lium = _load_wrapper()

    name = lium._resource_name(
        purpose="Bench",
        suffix="lium-b200-autoscale-primary-extra-long",
        max_len=32,
    )

    assert name.startswith("affine-autoscale-bench-")
    assert len(name) <= 32


def test_create_adopts_existing_pod_by_name_without_renting():
    lium = _load_wrapper()
    client = lium.LiumClient.__new__(lium.LiumClient)
    calls = []

    def request(method, path, **kwargs):
        calls.append((method, path))
        if (method, path) == ("GET", "/pods"):
            return {
                "pods": [
                    {
                        "uid": "pod-existing",
                        "pod_name": "affine-autoscale-eval-a",
                        "status": "running",
                    }
                ]
            }
        raise AssertionError((method, path))

    client.request = request
    client.select_executor = lambda: (_ for _ in ()).throw(
        AssertionError("should not select executor")
    )
    client.wait_ready = lambda uid: {"instance_id": uid}

    result = client.create("affine-autoscale-eval-a")

    assert result == {"instance_id": "pod-existing"}
    assert calls == [("GET", "/pods")]


def test_create_adopts_existing_pod_when_rent_response_fails():
    lium = _load_wrapper()
    client = lium.LiumClient.__new__(lium.LiumClient)
    find_results = ["", "pod-orphan"]

    client.select_executor = lambda: {"id": "exec-1"}
    client.resolve_template_id = lambda: "template-1"
    client.resolve_ssh_public_key = lambda: "ssh-rsa test"
    client.find_pod_by_name = lambda name, **kwargs: find_results.pop(0)
    client.request = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("provider returned 500 after creating")
    )
    client.wait_ready = lambda uid: {"instance_id": uid}

    result = client.create("affine-autoscale-eval-a")

    assert result == {"instance_id": "pod-orphan"}


def test_delete_treats_missing_owned_pod_as_success():
    lium = _load_wrapper()
    client = lium.LiumClient.__new__(lium.LiumClient)

    def request(method, path, **kwargs):
        raise lium.LiumHTTPError(method, path, 404, "missing")

    client.request = request

    assert client.delete("pod-1") is True


def test_error_payloads_distinguish_wrapper_and_provider_not_found():
    lium = _load_wrapper()

    route_payload = lium._route_not_found("POST", "/bad")
    provider_payload = lium._provider_not_found(
        lium.LiumHTTPError("GET", "/pods/pod-1", 404, "missing")
    )

    assert route_payload["error_source"] == "wrapper"
    assert route_payload["code"] == "route_not_found"
    assert provider_payload["error_source"] == "provider"
    assert provider_payload["provider_status_code"] == 404


def test_error_response_structures_provider_and_wrapper_failures():
    lium = _load_wrapper()

    status, payload = lium._error_response(
        lium.LiumHTTPError("GET", "/pods/pod-1", 401, "bad token")
    )
    assert status == 401
    assert payload["error_source"] == "provider"
    assert payload["code"] == "provider_http_error"
    assert payload["provider_status_code"] == 401
    assert payload["provider_path"] == "/pods/pod-1"

    status, payload = lium._error_response(
        lium.LiumRequestError(
            "GET",
            "/pods",
            TimeoutError("slow"),
            code="provider_timeout",
            http_status=504,
        )
    )
    assert status == 504
    assert payload["error_source"] == "provider"
    assert payload["code"] == "provider_timeout"

    status, payload = lium._error_response(
        lium.LiumWrapperError(
            "LIUM_API_BASE is required",
            code="wrapper_config_error",
        )
    )
    assert status == 500
    assert payload["error_source"] == "wrapper"
    assert payload["code"] == "wrapper_config_error"

    status, payload = lium._error_response(RuntimeError("boom"))
    assert status == 500
    assert payload["error_source"] == "wrapper"
    assert payload["code"] == "wrapper_internal_error"


def test_create_deletes_owned_pod_when_wait_ready_fails():
    lium = _load_wrapper()
    client = lium.LiumClient.__new__(lium.LiumClient)
    deleted = []

    client.select_executor = lambda: {"id": "exec-1"}
    client.resolve_template_id = lambda: "template-1"
    client.resolve_ssh_public_key = lambda: "ssh-rsa test"
    client.request = lambda *args, **kwargs: {"uid": "pod-1"}
    client.wait_ready = lambda uid: (_ for _ in ()).throw(RuntimeError("not ready"))
    client._delete_owned_pod = lambda uid: deleted.append(uid) or True

    with pytest.raises(RuntimeError, match="not ready"):
        client.create("affine-autoscale-lium-b200-1")

    assert deleted == ["pod-1"]
