"""Static endpoint registration, ownership, and preflight coverage."""

from __future__ import annotations

from typing import Optional

import pytest
from click.testing import CliRunner

import affine.database.cli as db_cli
from affine.cli.main import cli
from affine.database.dao.inference_endpoints import Endpoint, InferenceEndpointsDAO


class _FakeEndpointsDAO:
    def __init__(self, *, stage_error: Optional[Exception] = None):
        self.stage_error = stage_error
        self.staged = []
        self.activated = []

    async def stage_static_endpoint(self, endpoint, *, updated_by):
        if self.stage_error is not None:
            raise self.stage_error
        self.staged.append((endpoint, updated_by))

    async def activate_static_endpoint(
        self, name, *, expected_updated_by, updated_by,
    ):
        self.activated.append((name, expected_updated_by, updated_by))

def _patch_cli_dependencies(monkeypatch, dao):
    async def noop():
        return None

    monkeypatch.setattr(db_cli, "init_client", noop)
    monkeypatch.setattr(db_cli, "close_client", noop)
    monkeypatch.setattr(
        "affine.database.dao.inference_endpoints.InferenceEndpointsDAO",
        lambda: dao,
    )


def test_register_static_endpoint_stages_probes_then_activates(monkeypatch):
    dao = _FakeEndpointsDAO()
    _patch_cli_dependencies(monkeypatch, dao)
    probed = []

    async def probe(endpoint):
        probed.append(endpoint)
        return "SSH, Docker, and 8 NVIDIA GPU(s) ready"

    monkeypatch.setattr(db_cli, "_probe_static_endpoint", probe)
    result = CliRunner().invoke(
        cli,
        [
            "db",
            "register-static-endpoint",
            "--name",
            "manual-b200",
            "--kind",
            "ssh",
            "--ssh-url",
            "ssh://root@gpu.test:2200",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(dao.staged) == 1
    endpoint, updated_by = dao.staged[0]
    assert endpoint.name == "manual-b200"
    assert endpoint.active is False
    assert endpoint.autoscale_managed is False
    assert endpoint.sglang_load_balance_method == "total_tokens"
    assert updated_by.startswith("cli:register-static-endpoint:")
    assert probed == [endpoint]
    assert dao.activated == [
        ("manual-b200", updated_by, updated_by)
    ]
    assert "registered static endpoint 'manual-b200' as inactive" in result.output
    assert "activated static endpoint 'manual-b200'" in result.output


def test_register_static_endpoint_probe_failure_leaves_inactive(monkeypatch):
    dao = _FakeEndpointsDAO()
    _patch_cli_dependencies(monkeypatch, dao)

    async def probe(_endpoint):
        raise RuntimeError("docker unavailable")

    monkeypatch.setattr(db_cli, "_probe_static_endpoint", probe)
    result = CliRunner().invoke(
        cli,
        [
            "db",
            "register-static-endpoint",
            "--name",
            "manual-b200",
            "--kind",
            "ssh",
            "--ssh-url",
            "ssh://root@gpu.test",
        ],
    )

    assert result.exit_code == 1
    assert len(dao.staged) == 1
    assert dao.staged[0][0].active is False
    assert dao.activated == []
    assert "remains inactive: docker unavailable" in result.output


def test_register_static_endpoint_inactive_skips_probe(monkeypatch):
    dao = _FakeEndpointsDAO()
    _patch_cli_dependencies(monkeypatch, dao)

    async def unexpected_probe(_endpoint):
        raise AssertionError("probe should not run")

    monkeypatch.setattr(db_cli, "_probe_static_endpoint", unexpected_probe)
    result = CliRunner().invoke(
        cli,
        [
            "db",
            "register-static-endpoint",
            "--name",
            "manual-b200",
            "--kind",
            "ssh",
            "--ssh-url",
            "ssh://root@gpu.test",
            "--inactive",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(dao.staged) == 1
    assert dao.activated == []


def test_register_static_endpoint_rejects_autoscaler_owned_name(monkeypatch):
    dao = _FakeEndpointsDAO(
        stage_error=ValueError(
            "endpoint 'slot-1' is autoscaler-managed; use "
            "`af gpu replace-endpoint`"
        )
    )
    _patch_cli_dependencies(monkeypatch, dao)
    result = CliRunner().invoke(
        cli,
        [
            "db",
            "register-static-endpoint",
            "--name",
            "slot-1",
            "--kind",
            "ssh",
            "--ssh-url",
            "ssh://root@gpu.test",
        ],
    )

    assert result.exit_code == 1
    assert "autoscaler-managed" in result.output
    assert dao.activated == []


def test_register_static_targon_endpoint_requires_api_url():
    result = CliRunner().invoke(
        cli,
        [
            "db",
            "register-static-endpoint",
            "--name",
            "manual-targon",
            "--kind",
            "targon",
        ],
    )

    assert result.exit_code == 2
    assert "--targon-api-url is required" in result.output


def test_set_endpoint_command_is_removed():
    result = CliRunner().invoke(cli, ["db", "set-endpoint", "--help"])

    assert result.exit_code != 0
    assert "No such command 'set-endpoint'" in result.output


@pytest.mark.asyncio
async def test_stage_static_endpoint_preserves_runtime_assignment(monkeypatch):
    dao = InferenceEndpointsDAO()
    previous = Endpoint(
        name="manual-b200",
        kind="ssh",
        active=True,
        ssh_url="ssh://root@old-host",
        assigned_uid=42,
        assigned_hotkey="hk42",
        assignment_token="claim-42",
        assignment_status="ready",
        deployment_id="ssh:manual-b200:container",
        base_url="http://old-host:10001/v1",
        autoscale_provider="legacy-provider-metadata",
        autoscale_instance_id="legacy-instance-metadata",
    )

    async def get(_name):
        return previous

    calls = []

    async def update(name, **kwargs):
        calls.append((name, kwargs))
        return True

    monkeypatch.setattr(dao, "get", get)
    monkeypatch.setattr(dao, "_update_endpoint_fields", update)

    await dao.stage_static_endpoint(
        Endpoint(
            name="manual-b200",
            kind="ssh",
            active=True,
            ssh_url="ssh://root@new-host",
        )
    )

    assert len(calls) == 1
    name, call = calls[0]
    assert name == "manual-b200"
    assert call["set_values"]["active"] is False
    assert call["set_values"]["ssh_url"] == "ssh://root@new-host"
    for field in (
        "assigned_uid",
        "assigned_hotkey",
        "assignment_token",
        "assignment_status",
        "deployment_id",
        "base_url",
        "autoscale_provider",
        "autoscale_instance_id",
    ):
        assert field not in call["set_values"]
        assert field not in call.get("remove_fields", ())
    assert call["return_false_on_condition_failure"] is True


@pytest.mark.asyncio
async def test_stage_static_endpoint_rejects_managed_existing_row(monkeypatch):
    dao = InferenceEndpointsDAO()

    async def get(_name):
        return Endpoint(
            name="slot-1",
            kind="ssh",
            autoscale_managed=True,
            autoscale_instance_id="instance-1",
        )

    monkeypatch.setattr(dao, "get", get)

    with pytest.raises(ValueError, match="autoscaler-managed"):
        await dao.stage_static_endpoint(
            Endpoint(name="slot-1", kind="ssh")
        )


@pytest.mark.asyncio
async def test_stage_static_endpoint_rejects_concurrent_autoscaler_claim(
    monkeypatch,
):
    dao = InferenceEndpointsDAO()

    async def get(_name):
        return Endpoint(
            name="slot-1",
            kind="ssh",
            autoscale_managed=False,
        )

    async def update(_name, **_kwargs):
        return False

    monkeypatch.setattr(dao, "get", get)
    monkeypatch.setattr(dao, "_update_endpoint_fields", update)

    with pytest.raises(ValueError, match="became autoscaler-managed"):
        await dao.stage_static_endpoint(
            Endpoint(name="slot-1", kind="ssh")
        )


@pytest.mark.asyncio
async def test_activate_static_endpoint_preserves_runtime_assignment(monkeypatch):
    dao = InferenceEndpointsDAO()
    previous = Endpoint(
        name="manual-b200",
        kind="ssh",
        active=False,
        ssh_url="ssh://root@gpu.test",
        assigned_uid=42,
        assignment_token="claim-42",
        generation=3,
        activated_at=100,
    )

    async def get(_name):
        return previous

    calls = []

    async def update(name, **kwargs):
        calls.append((name, kwargs))
        return True

    monkeypatch.setattr(dao, "get", get)
    monkeypatch.setattr(dao, "_update_endpoint_fields", update)

    await dao.activate_static_endpoint(
        "manual-b200",
        expected_updated_by="registration-1",
    )

    _, call = calls[0]
    assert call["set_values"]["active"] is True
    assert call["set_values"]["generation"] == 4
    assert "assigned_uid" not in call["set_values"]
    assert "assignment_token" not in call["set_values"]
    assert call["condition_values"][":cond_updated_by"] == "registration-1"


@pytest.mark.asyncio
async def test_activate_static_endpoint_rejects_config_replaced_after_probe(
    monkeypatch,
):
    dao = InferenceEndpointsDAO()

    async def get(_name):
        return Endpoint(
            name="manual-b200",
            kind="ssh",
            active=False,
            updated_by="registration-2",
        )

    async def update(_name, **_kwargs):
        return False

    monkeypatch.setattr(dao, "get", get)
    monkeypatch.setattr(dao, "_update_endpoint_fields", update)

    with pytest.raises(ValueError, match="changed after its health check"):
        await dao.activate_static_endpoint(
            "manual-b200",
            expected_updated_by="registration-1",
        )


@pytest.mark.asyncio
async def test_probe_host_requires_docker_and_at_least_one_gpu(monkeypatch):
    from affine.src.scheduler import ssh

    commands = []

    async def exec_ok(_config, command):
        commands.append(command)
        return 0, "GPU 0: NVIDIA B200\nGPU 1: NVIDIA B200", ""

    monkeypatch.setattr(ssh, "_ssh_exec", exec_ok)
    count = await ssh.probe_host(ssh.SSHConfig(host="gpu.test"))

    assert count == 2
    assert commands == ["docker info >/dev/null 2>&1 && nvidia-smi -L"]


@pytest.mark.asyncio
async def test_probe_host_rejects_host_without_visible_gpu(monkeypatch):
    from affine.src.scheduler import ssh

    async def exec_without_gpu(_config, _command):
        return 0, "", ""

    monkeypatch.setattr(ssh, "_ssh_exec", exec_without_gpu)

    with pytest.raises(RuntimeError, match="found no NVIDIA GPUs"):
        await ssh.probe_host(ssh.SSHConfig(host="gpu.test"))


@pytest.mark.asyncio
async def test_static_endpoint_probe_requires_enough_gpus_for_dp(monkeypatch):
    from affine.src.scheduler import ssh

    async def probe_host(_config):
        return 4

    monkeypatch.setattr(ssh, "probe_host", probe_host)
    endpoint = Endpoint(
        name="manual-b200",
        kind="ssh",
        ssh_url="ssh://root@gpu.test",
        sglang_dp=8,
    )

    with pytest.raises(RuntimeError, match="sglang_dp=8 requires at least 8"):
        await db_cli._probe_static_endpoint(endpoint)
