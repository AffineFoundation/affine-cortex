"""Tests for the DB-driven provider dispatch resolver.

``_resolve_provider_kind`` is the single decision point that decides
whether the scheduler runs the SSH or Targon lifecycle. The previous
``AFFINE_PROVIDER_KIND`` env var was removed; this helper is now the
only thing between the active ``inference_endpoints`` rows and the
provider selection.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest
from botocore.exceptions import ClientError

from affine.database.dao.inference_endpoints import Endpoint
from affine.src.scheduler.flow import (
    DeploymentStateInvalidatedError,
    NoSpareEndpoint,
)
from affine.src.scheduler.main import (
    EndpointReservationConflict,
    _deploy_ssh_target,
    _gpu_autoscaler_empty_provider_kind,
    _resolve_provider_kind,
)
from affine.src.scheduler.ssh import SSHConfig
from affine.src.scheduler.targon import DeployResult, DeployTarget


class _UpdateRecordingClient:
    def __init__(self):
        self.update_calls = []

    async def update_item(self, **kwargs):
        self.update_calls.append(kwargs)


class _ConditionalFailureClient:
    async def update_item(self, **kwargs):
        raise ClientError(
            error_response={
                "Error": {"Code": "ConditionalCheckFailedException"},
            },
            operation_name="UpdateItem",
        )


class _ScanRecordingClient:
    def __init__(self, items):
        self.items = items
        self.scan_calls = []

    async def scan(self, **kwargs):
        self.scan_calls.append(kwargs)
        return {"Items": self.items}


class _ConfigDAOFake:
    def __init__(self, values):
        self.values = values

    async def get_param_value(self, name, default=None):
        return self.values.get(name, default)


@dataclass
class _Ep:
    name: str
    kind: str
    role: str = ""


def test_empty_active_raises():
    with pytest.raises(RuntimeError, match="no active inference endpoints"):
        _resolve_provider_kind([])


def test_empty_active_can_resolve_to_ssh_for_autoscaler():
    kind, ssh, targon = _resolve_provider_kind([], empty_provider_kind="ssh")
    assert kind == "ssh"
    assert ssh == []
    assert targon == []


@pytest.mark.asyncio
async def test_autoscaler_empty_provider_uses_system_config(monkeypatch):
    monkeypatch.delenv("AFFINE_GPU_AUTOSCALER_ENABLED", raising=False)
    config_dao = _ConfigDAOFake({
        "gpu_autoscaler": {
            "enabled": True,
            "endpoints": [{"name": "slot-1", "provider": "lium"}],
        }
    })

    assert await _gpu_autoscaler_empty_provider_kind(config_dao) == "ssh"


@pytest.mark.asyncio
async def test_endpoint_list_uses_consistent_scan(monkeypatch):
    from affine.database.dao.inference_endpoints import InferenceEndpointsDAO

    dao = InferenceEndpointsDAO()
    item = dao._serialize({
        "pk": "ENDPOINT#b300",
        "kind": "ssh",
        "active": True,
        "assigned_uid": 170,
    })
    client = _ScanRecordingClient([item])
    monkeypatch.setattr("affine.database.client.get_client", lambda: client)

    rows = await dao.list_all()

    assert client.scan_calls == [{
        "TableName": dao.table_name,
        "ConsistentRead": True,
    }]
    assert rows[0].name == "b300"
    assert rows[0].assigned_uid == 170


@pytest.mark.asyncio
async def test_endpoint_reservation_rejects_stale_free_snapshot(monkeypatch):
    from affine.database.dao.inference_endpoints import InferenceEndpointsDAO

    endpoint = Endpoint(name="b300", kind="ssh", active=True)
    client = _ConditionalFailureClient()
    monkeypatch.setattr("affine.database.client.get_client", lambda: client)

    reserved = await InferenceEndpointsDAO().try_reserve_assignment(
        endpoint,
        token="attempt-2",
        uid=194,
        hotkey="hk194",
        model="org/model-194",
        revision="rev194",
        role="pre_challenger",
        expires_at=2_000_000_000,
    )

    assert reserved is False


@pytest.mark.asyncio
async def test_endpoint_reservation_sets_owner_before_deploy(monkeypatch):
    from affine.database.dao.inference_endpoints import InferenceEndpointsDAO

    endpoint = Endpoint(name="b300", kind="ssh", active=True)
    client = _UpdateRecordingClient()
    monkeypatch.setattr("affine.database.client.get_client", lambda: client)

    reserved = await InferenceEndpointsDAO().try_reserve_assignment(
        endpoint,
        token="attempt-170",
        uid=170,
        hotkey="hk170",
        model="org/model-170",
        revision="rev170",
        role="pre_challenger",
        expires_at=2_000_000_000,
    )

    assert reserved is True
    call = client.update_calls[0]
    assert "attribute_type(#cond_token, :null_type)" in call[
        "ConditionExpression"
    ]
    assert "attribute_type(#cond_uid, :null_type)" in call[
        "ConditionExpression"
    ]
    assert call["ExpressionAttributeValues"][":null_type"] == {"S": "NULL"}
    serialized_values = list(call["ExpressionAttributeValues"].values())
    assert {"S": "attempt-170"} in serialized_values
    assert {"S": "deploying"} in serialized_values


@pytest.mark.asyncio
async def test_endpoint_reservation_replaces_fully_assigned_snapshot(monkeypatch):
    from affine.database.dao.inference_endpoints import InferenceEndpointsDAO

    endpoint = Endpoint(
        name="b300",
        kind="ssh",
        active=True,
        assigned_uid=210,
        assigned_hotkey="hk210",
        assigned_model="org/model-210",
        assigned_revision="rev210",
        assignment_token="attempt-210",
        assignment_status="ready",
    )
    client = _UpdateRecordingClient()
    monkeypatch.setattr("affine.database.client.get_client", lambda: client)

    reserved = await InferenceEndpointsDAO().try_reserve_assignment(
        endpoint,
        token="attempt-197",
        uid=197,
        hotkey="hk197",
        model="org/model-197",
        revision="rev197",
        role="challenger",
        expires_at=2_000_000_000,
    )

    assert reserved is True
    call = client.update_calls[0]
    assert "#cond_token = :expected_token" in call["ConditionExpression"]
    assert "attribute_type(" not in call["ConditionExpression"]
    assert ":null_type" not in call["ExpressionAttributeValues"]


@pytest.mark.asyncio
async def test_live_endpoint_reservation_cannot_be_taken_over(monkeypatch):
    from affine.database.dao.inference_endpoints import InferenceEndpointsDAO

    endpoint = Endpoint(
        name="b300",
        kind="ssh",
        active=True,
        assigned_uid=170,
        assignment_token="attempt-170",
        assignment_status="deploying",
        assignment_expires_at=4_000_000_000,
    )
    client = _UpdateRecordingClient()
    monkeypatch.setattr("affine.database.client.get_client", lambda: client)

    reserved = await InferenceEndpointsDAO().try_reserve_assignment(
        endpoint,
        token="attempt-194",
        uid=194,
        hotkey="hk194",
        model="org/model-194",
        revision="rev194",
        role="pre_challenger",
        expires_at=4_000_000_100,
    )

    assert reserved is False
    assert client.update_calls == []


def test_all_ssh_resolves_to_ssh():
    eps = [_Ep("b300", "ssh"), _Ep("b300-2", "ssh")]
    kind, ssh, targon = _resolve_provider_kind(eps)
    assert kind == "ssh"
    assert [ep.name for ep in ssh] == ["b300", "b300-2"]
    assert targon == []


def test_non_scoring_endpoints_are_ignored_for_provider_resolution():
    eps = [_Ep("ceac", "ssh", role="anticopy")]

    with pytest.raises(RuntimeError, match="no active inference endpoints"):
        _resolve_provider_kind(eps)

    kind, ssh, targon = _resolve_provider_kind(
        eps,
        empty_provider_kind="ssh",
    )
    assert kind == "ssh"
    assert ssh == []
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
    assert "assignment_token" in removed_fields
    assert "assignment_status" in removed_fields
    assert "assignment_expires_at" in removed_fields


class _EndpointsDAOFake:
    def __init__(self, endpoints):
        self.endpoints = endpoints
        self.cleared = []
        self.assignments = []
        self.reservations = []
        self.releases = []

    async def list_active(self, kind=None):
        return [
            ep for ep in self.endpoints
            if ep.active and (kind is None or ep.kind == kind)
        ]

    async def clear_assignment(self, name):
        self.cleared.append(name)

    async def try_reserve_assignment(self, endpoint, **kwargs):
        self.reservations.append((endpoint.name, kwargs))
        endpoint.assigned_uid = kwargs["uid"]
        endpoint.assigned_hotkey = kwargs["hotkey"]
        endpoint.assigned_model = kwargs["model"]
        endpoint.assigned_revision = kwargs["revision"]
        endpoint.assignment_role = kwargs["role"]
        endpoint.assignment_token = kwargs["token"]
        endpoint.assignment_status = "deploying"
        return True

    async def finalize_assignment(self, name, **kwargs):
        endpoint = next(ep for ep in self.endpoints if ep.name == name)
        if endpoint.assignment_token != kwargs["token"]:
            return False
        endpoint.deployment_id = kwargs["deployment_id"]
        endpoint.base_url = kwargs["base_url"]
        endpoint.assignment_status = "ready"
        self.assignments.append((name, {
            "uid": endpoint.assigned_uid,
            "hotkey": endpoint.assigned_hotkey,
            "model": endpoint.assigned_model,
            "revision": endpoint.assigned_revision,
            "deployment_id": kwargs["deployment_id"],
            "base_url": kwargs["base_url"],
            "role": endpoint.assignment_role,
        }))
        return True

    async def release_assignment(self, name, *, token):
        endpoint = next(ep for ep in self.endpoints if ep.name == name)
        self.releases.append((name, token))
        if endpoint.assignment_token != token:
            return False
        endpoint.assigned_uid = None
        endpoint.assigned_hotkey = None
        endpoint.assigned_model = None
        endpoint.assigned_revision = None
        endpoint.assignment_token = None
        endpoint.assignment_status = None
        return True


class _StaleReadEndpointsDAOFake(_EndpointsDAOFake):
    """Always returns the original free row while preserving atomic owner."""

    def __init__(self, endpoints):
        super().__init__(endpoints)
        self._reservation_owner = None

    async def list_active(self, kind=None):
        return [
            replace(
                ep,
                assigned_uid=None,
                assigned_hotkey=None,
                assigned_model=None,
                assigned_revision=None,
                assignment_token=None,
                assignment_status=None,
            )
            for ep in self.endpoints
            if ep.active and (kind is None or ep.kind == kind)
        ]

    async def try_reserve_assignment(self, endpoint, **kwargs):
        self.reservations.append((endpoint.name, kwargs))
        if self._reservation_owner is not None:
            return False
        self._reservation_owner = kwargs["token"]
        actual = next(ep for ep in self.endpoints if ep.name == endpoint.name)
        actual.assigned_uid = kwargs["uid"]
        actual.assigned_hotkey = kwargs["hotkey"]
        actual.assigned_model = kwargs["model"]
        actual.assigned_revision = kwargs["revision"]
        actual.assignment_role = kwargs["role"]
        actual.assignment_token = kwargs["token"]
        actual.assignment_status = "deploying"
        return True


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

    assert [name for name, _token in dao.releases] == ["b300"]
    assert dao.assignments == []
    assert excinfo.value.invalidated_deployment_ids == (
        "ssh:b300:affine-sglang-current",
    )


@pytest.mark.asyncio
async def test_ssh_deploy_refreshes_cached_endpoint_config(monkeypatch):
    endpoint = Endpoint(
        name="b300",
        kind="ssh",
        ssh_url="ssh://root@b300",
        ssh_key_path="/root/.ssh/new-key",
    )
    dao = _EndpointsDAOFake([endpoint])
    ssh_configs = {
        "b300": SSHConfig(
            host="b300",
            endpoint_name="b300",
            key_path="/root/.ssh/old-key",
        )
    }

    async def deploy(config, target):
        assert config.key_path == "/root/.ssh/new-key"
        return DeployResult(
            deployment_id=config.deployment_id(),
            base_url=config.inference_url(),
        )

    monkeypatch.setattr(
        "affine.src.scheduler.main.ssh_lifecycle.deploy",
        deploy,
    )

    await _deploy_ssh_target(dao, ssh_configs, _target(), role="challenger")

    assert ssh_configs["b300"].key_path == "/root/.ssh/new-key"


@pytest.mark.asyncio
async def test_ssh_deploy_skips_non_scoring_endpoints(monkeypatch):
    anticopy = Endpoint(
        name="ceac",
        kind="ssh",
        role="anticopy",
        ssh_url="ssh://root@ceac",
    )
    scoring = Endpoint(
        name="b300",
        kind="ssh",
        ssh_url="ssh://root@b300",
    )
    dao = _EndpointsDAOFake([anticopy, scoring])

    async def deploy(config, target):
        return DeployResult(
            deployment_id=config.deployment_id(),
            base_url=config.inference_url(),
        )

    monkeypatch.setattr(
        "affine.src.scheduler.main.ssh_lifecycle.deploy",
        deploy,
    )

    result = await _deploy_ssh_target(dao, {}, _target(), role="champion")

    assert result.deployment_id == "ssh:b300:affine-sglang-current"
    assert dao.assignments == [
        (
            "b300",
            {
                "uid": 42,
                "hotkey": "hk42",
                "model": "Qwen/Qwen3-30B-A22B",
                "revision": "rev42",
                "deployment_id": "ssh:b300:affine-sglang-current",
                "base_url": "http://b300:10001/v1",
                "role": "champion",
            },
        )
    ]


@pytest.mark.asyncio
async def test_stale_endpoint_read_cannot_overwrite_predeploy_reservation(
    monkeypatch,
):
    endpoints = [
        Endpoint(name="a-primary", kind="ssh", ssh_url="ssh://root@primary"),
        Endpoint(name="b-spare", kind="ssh", ssh_url="ssh://root@spare"),
    ]
    dao = _StaleReadEndpointsDAOFake(endpoints)
    deployed_uids = []

    async def deploy(config, target):
        deployed_uids.append(target.uid)
        return DeployResult(
            deployment_id=config.deployment_id(),
            base_url=config.inference_url(),
        )

    monkeypatch.setattr(
        "affine.src.scheduler.main.ssh_lifecycle.deploy",
        deploy,
    )

    first = _target()
    await _deploy_ssh_target(dao, {}, first, role="pre_challenger")
    second = DeployTarget(
        uid=194,
        hotkey="hk194",
        model="org/model-194",
        revision="rev194",
    )
    with pytest.raises(NoSpareEndpoint):
        await _deploy_ssh_target(dao, {}, second, role="pre_challenger")

    assert deployed_uids == [42]
    assert endpoints[1].assigned_uid == 42


@pytest.mark.asyncio
async def test_lost_reservation_does_not_clear_new_endpoint_owner(monkeypatch):
    endpoint = Endpoint(
        name="b300",
        kind="ssh",
        ssh_url="ssh://root@b300",
    )
    dao = _EndpointsDAOFake([endpoint])

    async def deploy(config, target):
        endpoint.assignment_token = "new-owner"
        endpoint.assigned_uid = 194
        return DeployResult(
            deployment_id=config.deployment_id(),
            base_url=config.inference_url(),
        )

    monkeypatch.setattr(
        "affine.src.scheduler.main.ssh_lifecycle.deploy",
        deploy,
    )

    with pytest.raises(EndpointReservationConflict):
        await _deploy_ssh_target(dao, {}, _target(), role="challenger")

    assert endpoint.assignment_token == "new-owner"
    assert endpoint.assigned_uid == 194
    assert dao.assignments == []
