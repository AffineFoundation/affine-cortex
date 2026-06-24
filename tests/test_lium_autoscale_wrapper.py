from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_wrapper():
    path = Path(__file__).resolve().parents[1] / "scripts" / (
        "lium_autoscale_wrapper.py"
    )
    spec = importlib.util.spec_from_file_location(
        "lium_autoscale_wrapper", path
    )
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


def test_delete_treats_missing_owned_pod_as_success():
    lium = _load_wrapper()
    client = lium.LiumClient.__new__(lium.LiumClient)

    def request(method, path, **kwargs):
        raise lium.LiumHTTPError(method, path, 404, "missing")

    client.request = request

    assert client.delete("pod-1") is True


def test_create_deletes_owned_pod_when_wait_ready_fails():
    lium = _load_wrapper()
    client = lium.LiumClient.__new__(lium.LiumClient)
    deleted = []

    client.select_executor = lambda: {"id": "exec-1"}
    client.resolve_template_id = lambda: "template-1"
    client.resolve_ssh_public_key = lambda: "ssh-rsa test"
    client.request = lambda *args, **kwargs: {"uid": "pod-1"}
    client.wait_ready = lambda uid: (_ for _ in ()).throw(
        RuntimeError("not ready")
    )
    client._delete_owned_pod = lambda uid: deleted.append(uid) or True

    with pytest.raises(RuntimeError, match="not ready"):
        client.create("affine-autoscale-lium-b200-1")

    assert deleted == ["pod-1"]
