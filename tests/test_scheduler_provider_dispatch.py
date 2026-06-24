"""Tests for the DB-driven provider dispatch resolver.

``_resolve_provider_kind`` is the single decision point that decides
whether the scheduler runs the SSH or Targon lifecycle. The previous
``AFFINE_PROVIDER_KIND`` env var was removed; this helper is now the
only thing between the active ``inference_endpoints`` rows and the
provider selection.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from affine.database.dao.inference_endpoints import Endpoint
from affine.src.scheduler.flow import DeploymentStateInvalidatedError
from affine.src.scheduler.main import (
    _deploy_ssh_target,
    _resolve_provider_kind,
)
from affine.src.scheduler.targon import DeployTarget


class _UpdateRecordingClient:
    def __init__(self):
        self.update_calls = []

    async def update_item(self, **kwargs):
        self.update_calls.append(kwargs)


@dataclass
class _Ep:
    name: str
    kind: str


def test_empty_active_raises():
    with pytest.raises(RuntimeError, match="no active inference endpoints"):
        _resolve_provider_kind([])


def test_empty_active_can_resolve_to_ssh_for_autoscaler():
    kind, ssh, targon = _resolve_provider_kind([], empty_provider_kind="ssh")
    assert kind == "ssh"
    assert ssh == []
    assert targon == []


def test_all_ssh_resolves_to_ssh():
    eps = [_Ep("b300", "ssh"), _Ep("b300-2", "ssh")]
    kind, ssh, targon = _resolve_provider_kind(eps)
    assert kind == "ssh"
    assert [ep.name for ep in ssh] == ["b300", "b300-2"]
    assert targon == []


def test_all_targon_resolves_to_targon():
    eps = [_Ep("prod", "targon")]
    kind, ssh, targon = _resolve_provider_kind(eps)
    assert kind == "targon"
    assert ssh == []
    assert [ep.name for ep in targon] == ["prod"]


def test_mixed_kinds_raises():
    eps = [_Ep("b300", "ssh"), _Ep("prod", "targon")]
    with pytest.raises(RuntimeError, match="mixed-kind"):
        _resolve_provider_kind(eps)


def test_unknown_kinds_raises():
    eps = [_Ep("weird", "vllm")]
    with pytest.raises(RuntimeError, match="none are kind=ssh"):
        _resolve_provider_kind(eps)


def test_endpoint_activation_bumps_on_enable_or_runtime_config_change():
    from affine.database.dao.inference_endpoints import InferenceEndpointsDAO

    disabled = Endpoint(
        name="b300",
        kind="ssh",
        active=False,
        ssh_url="ssh://old-host",
    )
    enabled = Endpoint(
        name="b300",
        kind="ssh",
        active=True,
        ssh_url="ssh://old-host",
    )
    moved = Endpoint(
        name="b300",
        kind="ssh",
        active=True,
        ssh_url="ssh://new-host",
    )
    legacy_active = Endpoint(
        name="b300",
        kind="ssh",
        active=True,
        ssh_url="ssh://old-host",
        generation=0,
        activated_at=0,
    )
    current_active = Endpoint(
        name="b300",
        kind="ssh",
        active=True,
        ssh_url="ssh://old-host",
        generation=1,
        activated_at=100,
    )
    assigned = Endpoint(
        name="b300",
        kind="ssh",
        active=True,
        ssh_url="ssh://old-host",
        assigned_uid=123,
    )

    assert InferenceEndpointsDAO._activation_bump_required(None, enabled)
    assert InferenceEndpointsDAO._activation_bump_required(disabled, enabled)
    assert InferenceEndpointsDAO._activation_bump_required(legacy_active, enabled)
    assert InferenceEndpointsDAO._activation_bump_required(enabled, moved)
    assert not InferenceEndpointsDAO._activation_bump_required(current_active, assigned)
    assert not InferenceEndpointsDAO._activation_bump_required(current_active, Endpoint(
        name="b300",
        kind="ssh",
        active=False,
        ssh_url="ssh://old-host",
    ))


@pytest.mark.asyncio
async def test_activate_autoscaled_endpoint_does_not_write_assignment_fields(
    monkeypatch,
):
    from affine.database.dao.inference_endpoints import InferenceEndpointsDAO

    client = _UpdateRecordingClient()
    monkeypatch.setattr("affine.database.client.get_client", lambda: client)
    dao = InferenceEndpointsDAO()

    async def fake_get(name):
        return Endpoint(
            name=name,
            kind="ssh",
            active=False,
            assigned_uid=42,
            assigned_hotkey="hk42",
            generation=3,
            activated_at=100,
        )

    dao.get = fake_get
    await dao.activate_autoscaled_endpoint(
        Endpoint(
            name="b300",
            kind="ssh",
            active=True,
            ssh_url="ssh://root@b300",
            autoscale_instance_id="inst-b300",
            assigned_uid=99,
            assigned_hotkey="new",
        )
    )

    call = client.update_calls[0]
    updated_fields = set(call["ExpressionAttributeNames"].values())
    assert "assigned_uid" not in updated_fields
    assert "assigned_hotkey" not in updated_fields
    assert "deployment_id" not in updated_fields
    assert "autoscale_instance_id" in updated_fields
    assert call["Key"] == {"pk": {"S": "ENDPOINT#b300"}}


@pytest.mark.asyncio
async def test_deactivate_autoscaled_endpoint_is_conditioned_on_instance(
    monkeypatch,
):
    from affine.database.dao.inference_endpoints import InferenceEndpointsDAO

    client = _UpdateRecordingClient()
    monkeypatch.setattr("affine.database.client.get_client", lambda: client)
    dao = InferenceEndpointsDAO()

    await dao.deactivate_autoscaled_endpoint(
        "b300",
        instance_id="inst-b300",
    )

    call = client.update_calls[0]
    removed_fields = set(call["ExpressionAttributeNames"].values())
    assert call["ConditionExpression"] == "#cond_instance = :cond_instance"
    assert call["ExpressionAttributeValues"][":cond_instance"] == {
        "S": "inst-b300"
    }
    assert "autoscale_instance_id" in removed_fields
    assert "assigned_uid" in removed_fields
    assert "assigned_hotkey" in removed_fields
    assert "deployment_id" in removed_fields


class _EndpointsDAOFake:
    def __init__(self, endpoints):
        self.endpoints = endpoints
        self.cleared = []
        self.assignments = []

    async def list_active(self, kind=None):
        return [
            ep for ep in self.endpoints
            if ep.active and (kind is None or ep.kind == kind)
        ]

    async def clear_assignment(self, name):
        self.cleared.append(name)

    async def set_assignment(self, name, **kwargs):
        self.assignments.append((name, kwargs))


def _target():
    return DeployTarget(
        uid=42,
        hotkey="hk42",
        model="Qwen/Qwen3-30B-A22B",
        revision="rev42",
    )


@pytest.mark.asyncio
async def test_ssh_deploy_failure_clears_assignment_and_reports_deployment(
    monkeypatch,
):
    endpoint = Endpoint(
        name="b300",
        kind="ssh",
        ssh_url="ssh://root@b300",
        assigned_uid=99,
        assigned_hotkey="old",
        assigned_model="old/model",
        assigned_revision="oldrev",
    )
    dao = _EndpointsDAOFake([endpoint])

    async def fail_deploy(config, target):
        raise RuntimeError("docker run failed")

    monkeypatch.setattr(
        "affine.src.scheduler.main.ssh_lifecycle.deploy",
        fail_deploy,
    )

    with pytest.raises(DeploymentStateInvalidatedError) as excinfo:
        await _deploy_ssh_target(dao, {}, _target(), role="challenger")

    assert dao.cleared == ["b300"]
    assert dao.assignments == []
    assert excinfo.value.invalidated_deployment_ids == (
        "ssh:b300:affine-sglang-current",
    )
