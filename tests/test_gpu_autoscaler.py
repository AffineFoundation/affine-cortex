from __future__ import annotations

import asyncio

import pytest

from affine.core.providers.instance_api_client import (
    InstanceAPIConfig,
    InstanceHandle,
    InstanceAPINotFoundError,
)
from affine.database.dao.inference_endpoints import Endpoint
from affine.src.scheduler.gpu_autoscaler import (
    CONFIG_KEY,
    EndpointAddResult,
    EndpointReplaceResult,
    GPUAutoscaler,
    GPUAutoscalerConfig,
    MANUAL_ENDPOINT_OPERATION_ADD,
    MANUAL_ENDPOINT_OPERATION_REMOVE,
    MANUAL_ENDPOINT_OPERATION_REPLACE,
    MANUAL_REPLACEMENT_LOCK_KEY,
    MANUAL_REPLACEMENT_REQUEST_VERSION,
    MANUAL_REPLACEMENT_STATE_KEY,
    ManagedEndpointSlot,
    STATE_KEY,
    _config_without_endpoint_slot,
    _enqueue_manual_add_request,
    _enqueue_manual_remove_request,
    _enqueue_manual_replacement_request,
    _manual_endpoint_result_key,
    _wait_for_manual_endpoint_result,
    enqueue_health_endpoint_replacement_request,
    load_config,
)
from affine.src.scorer.window_state import (
    BattleRecord,
    ChampionRecord,
    DeploymentRecord,
    InMemoryConfigStore,
    MinerSnapshot,
    StateStore,
)


class _Queue:
    def __init__(self, *, pending=0, in_progress=0):
        self.pending = pending
        self.in_progress = in_progress
        self.released = []

    async def peek_next(self, n, *, champion_uid, exclude_uids=None):
        return [object() for _ in range(min(self.pending, n))]

    async def list_in_progress(self):
        return [object() for _ in range(self.in_progress)]

    async def release_claim(self, uid, *, hotkey=None, revision=None):
        self.released.append((uid, hotkey, revision))
        return True


class _ConfigDAO:
    def __init__(self, values):
        self.values = values

    async def get_param_value(self, name, default=None):
        return self.values.get(name, default)


class _Endpoints:
    def __init__(self, endpoints=None):
        self.endpoints = {ep.name: ep for ep in (endpoints or [])}
        self.cleared = []
        self.events = []
        self.fail_activate = False
        self.now = 1000

    async def list_all(self):
        return list(self.endpoints.values())

    async def list_active(self, kind=None):
        return [
            ep for ep in self.endpoints.values()
            if ep.active and (kind is None or ep.kind == kind)
        ]

    async def get(self, name):
        return self.endpoints.get(name)

    async def upsert(self, endpoint, *, updated_by="test"):
        self.endpoints[endpoint.name] = endpoint

    async def activate_autoscaled_endpoint(self, endpoint, *, updated_by="test"):
        if self.fail_activate:
            raise RuntimeError("simulated activation failure")
        self.events.append(("activate", endpoint.name))
        _Client.events.append(("activate", endpoint.name))
        existing = self.endpoints.get(endpoint.name)
        if existing is not None:
            endpoint.assigned_uid = existing.assigned_uid
            endpoint.assigned_hotkey = existing.assigned_hotkey
            endpoint.assigned_model = existing.assigned_model
            endpoint.assigned_revision = existing.assigned_revision
            endpoint.deployment_id = existing.deployment_id
            endpoint.base_url = existing.base_url
            endpoint.assignment_role = existing.assignment_role
            endpoint.assigned_at = existing.assigned_at
        self.endpoints[endpoint.name] = endpoint

    async def update_autoscale_lease(
        self, name, *, instance_id, lease_expires_at, updated_by="test",
    ):
        endpoint = self.endpoints[name]
        if endpoint.autoscale_instance_id != instance_id:
            raise RuntimeError("stale instance id")
        endpoint.autoscale_updated_at = self.now
        endpoint.autoscale_lease_expires_at = lease_expires_at

    async def deactivate_autoscaled_endpoint(
        self, name, *, instance_id, updated_by="test",
    ):
        self.events.append(("deactivate", name, instance_id))
        _Client.events.append(("deactivate", name, instance_id))
        endpoint = self.endpoints[name]
        if endpoint.autoscale_instance_id != instance_id:
            raise RuntimeError("stale instance id")
        endpoint.active = False
        endpoint.ssh_url = None
        endpoint.public_inference_url = None
        endpoint.assigned_uid = None
        endpoint.assigned_hotkey = None
        endpoint.assigned_model = None
        endpoint.assigned_revision = None
        endpoint.deployment_id = None
        endpoint.base_url = None
        endpoint.assignment_role = None
        endpoint.assigned_at = 0
        endpoint.autoscale_instance_id = None
        endpoint.autoscale_purpose = None
        endpoint.autoscale_lease_expires_at = 0

    async def drain_autoscaled_endpoint(
        self, name, *, instance_id, updated_by="test",
    ):
        self.events.append(("drain", name, instance_id))
        _Client.events.append(("drain", name, instance_id))
        endpoint = self.endpoints[name]
        if endpoint.autoscale_instance_id != instance_id:
            raise RuntimeError("stale instance id")
        endpoint.active = False
        endpoint.ssh_url = None
        endpoint.public_inference_url = None
        endpoint.assigned_uid = None
        endpoint.assigned_hotkey = None
        endpoint.assigned_model = None
        endpoint.assigned_revision = None
        endpoint.deployment_id = None
        endpoint.base_url = None
        endpoint.assignment_role = None
        endpoint.assigned_at = 0

    async def clear_assignment(self, name, *, updated_by="test"):
        self.cleared.append(name)


class _Samples:
    def __init__(self, count=0):
        self.count = count

    async def count_samples_for_tasks(
        self, hotkey, revision, env, task_ids, refresh_block
    ):
        return self.count


class _Client:
    deleted = []
    delete_calls = []
    create_calls = []
    events = []
    renewed = []
    status_calls = []
    create_lease_expires_at = 1500
    create_public_inference_url = None
    create_sglang_port = 10001
    renew_expires_at = 0
    create_returns_none_for_names = set()
    delete_returns_false_for_ids = set()
    renew_not_found_for_ids = set()
    status_not_found_for_ids = set()

    def __init__(self, config):
        self.config = config

    async def create(self, *, variables=None, payload_overrides=None):
        self.create_calls.append(
            (dict(variables or {}), dict(payload_overrides or {}))
        )
        name = variables["endpoint_name"]
        self.events.append(("create", name))
        if name in self.create_returns_none_for_names:
            return None
        return InstanceHandle(
            provider=self.config.provider,
            instance_id=f"inst-{name}",
            ssh_url=f"ssh://root@{name}.example.com:22",
            public_inference_url=(
                self.create_public_inference_url
                if self.create_public_inference_url is not None
                else f"http://{name}.example.com:{self.create_sglang_port}/v1"
            ),
            sglang_port=self.create_sglang_port,
            lease_expires_at=self.create_lease_expires_at,
        )

    async def delete(self, instance_id, *, variables=None):
        self.deleted.append(instance_id)
        self.delete_calls.append((instance_id, dict(variables or {})))
        self.events.append(("delete", instance_id))
        if instance_id in self.delete_returns_false_for_ids:
            return False
        return True

    async def renew(self, instance_id):
        self.renewed.append(instance_id)
        if instance_id in self.renew_not_found_for_ids:
            raise InstanceAPINotFoundError(
                "POST",
                f"/instances/{instance_id}/renew",
                500,
                "Pod not found",
            )
        return InstanceHandle(
            provider=self.config.provider,
            instance_id=instance_id,
            lease_expires_at=self.renew_expires_at,
        )

    async def status(self, instance_id):
        self.status_calls.append(instance_id)
        if instance_id in self.status_not_found_for_ids:
            raise InstanceAPINotFoundError(
                "GET",
                f"/instances/{instance_id}",
                404,
                (
                    '{"error_source":"provider",'
                    '"code":"provider_instance_not_found",'
                    '"provider_status_code":404}'
                ),
            )
        return {"instance_id": instance_id, "status": "running"}


@pytest.fixture(autouse=True)
def _reset_client_state():
    _Client.deleted = []
    _Client.delete_calls = []
    _Client.create_calls = []
    _Client.events = []
    _Client.renewed = []
    _Client.status_calls = []
    _Client.create_lease_expires_at = 1500
    _Client.create_public_inference_url = None
    _Client.create_sglang_port = 10001
    _Client.renew_expires_at = 0
    _Client.create_returns_none_for_names = set()
    _Client.delete_returns_false_for_ids = set()
    _Client.renew_not_found_for_ids = set()
    _Client.status_not_found_for_ids = set()


def _config(
    *,
    enabled=True,
    threshold=1,
    idle_seconds=0,
    max_gpu_down_wait_seconds=0,
    lease_duration_seconds=0,
    lease_renew_margin_seconds=3600,
    lease_renew_cooldown_seconds=300,
    endpoint_health_check_interval_seconds=300,
    renew_path="/instances/{instance_id}/renew",
    status_path="",
):
    return GPUAutoscalerConfig(
        enabled=enabled,
        pending_threshold_per_instance=threshold,
        max_gpu_down_wait_seconds=max_gpu_down_wait_seconds,
        idle_seconds=idle_seconds,
        lease_duration_seconds=lease_duration_seconds,
        lease_renew_margin_seconds=lease_renew_margin_seconds,
        lease_renew_cooldown_seconds=lease_renew_cooldown_seconds,
        endpoint_health_check_interval_seconds=(
            endpoint_health_check_interval_seconds
        ),
        min_instances=0,
        max_instances=2,
        providers={
            "lium": InstanceAPIConfig(
                provider="lium",
                api_url="https://lium.example.com",
                create_path="/instances",
                delete_path="/instances/{instance_id}",
                renew_path=renew_path,
                status_path=status_path,
            )
        },
        slots=[
            ManagedEndpointSlot(
                name="lium-b200-1",
                provider="lium",
                endpoint={"ssh_key_path": "/root/.ssh/affine_validator_server"},
            ),
            ManagedEndpointSlot(name="lium-b200-2", provider="lium"),
        ],
    )


def _config_payload():
    return {
        "enabled": True,
        "pending_threshold_per_instance": 1,
        "max_gpu_down_wait_seconds": 0,
        "min_instances": 0,
        "max_instances": 2,
        "providers": {
            "lium": {
                "api_url": "https://lium.example.com",
                "create_path": "/instances",
                "delete_path": "/instances/{instance_id}",
            }
        },
        "endpoints": [
            {"name": "lium-b200-1", "provider": "lium"},
            {"name": "lium-b200-2", "provider": "lium"},
        ],
    }


def _autoscaler(queue, endpoints, kv, *, now=1000, samples=None):
    if hasattr(endpoints, "now"):
        endpoints.now = now
    return GPUAutoscaler(
        queue=queue,
        endpoints_dao=endpoints,
        state_store=StateStore(kv),
        kv_store=kv,
        samples_adapter=samples or _Samples(count=0),
        client_factory=_Client,
        now_fn=lambda: now,
    )


def test_default_gpu_down_wait_is_twelve_hours():
    cfg = GPUAutoscalerConfig.from_mapping({})
    assert cfg.max_gpu_down_wait_seconds == 12 * 60 * 60


@pytest.mark.asyncio
async def test_load_config_prefers_system_config_over_autoscaler_env(monkeypatch):
    monkeypatch.setenv("AFFINE_GPU_AUTOSCALER_ENABLED", "false")
    monkeypatch.setenv("AFFINE_GPU_AUTOSCALER_IDLE_SECONDS", "1")
    monkeypatch.setenv(
        "AFFINE_GPU_AUTOSCALER_CONFIG_JSON",
        '{"enabled": false, "idle_seconds": 2}',
    )

    cfg = await load_config(_ConfigDAO({
        "gpu_autoscaler": {
            "enabled": True,
            "idle_seconds": 123,
            "endpoints": [{"name": "lium-b200-1", "provider": "lium"}],
        }
    }))

    assert cfg.enabled is True
    assert cfg.idle_seconds == 123
    assert [slot.name for slot in cfg.slots] == ["lium-b200-1"]


@pytest.mark.asyncio
async def test_load_config_uses_autoscaler_env_when_system_config_absent(
    monkeypatch,
):
    monkeypatch.setenv("AFFINE_GPU_AUTOSCALER_ENABLED", "true")
    monkeypatch.setenv("AFFINE_GPU_AUTOSCALER_IDLE_SECONDS", "7")
    monkeypatch.setenv(
        "AFFINE_GPU_AUTOSCALER_CONFIG_JSON",
        '{"endpoints": [{"name": "lium-b200-1", "provider": "lium"}]}',
    )

    cfg = await load_config(_ConfigDAO({}))

    assert cfg.enabled is True
    assert cfg.idle_seconds == 7
    assert [slot.name for slot in cfg.slots] == ["lium-b200-1"]


def test_managed_endpoint_slot_accepts_sglang_runtime_overrides():
    slot = ManagedEndpointSlot.from_mapping({
        "name": "targon-b200-autoscale-1",
        "provider": "targon",
        "sglang_load_balance_method": "total_requests",
        "sglang_docker_args": ["--cgroupns=host"],
    })

    assert slot.endpoint["sglang_load_balance_method"] == "total_requests"
    assert slot.endpoint["sglang_docker_args"] == ["--cgroupns=host"]


def test_managed_endpoint_slot_accepts_autoscale_purpose():
    slot = ManagedEndpointSlot.from_mapping({
        "name": "bench-b200-1",
        "provider": "lium",
        "purpose": "Bench Pool",
    })

    assert slot.purpose == "bench-pool"


def test_managed_endpoint_slot_ignores_legacy_tunnel_config():
    slot = ManagedEndpointSlot.from_mapping({
        "name": "lium-b200-1",
        "provider": "lium",
        "public_inference_url": "http://validator.example.com:8102/v1",
        "tunnel": {
            "enabled": True,
            "bind_host": "0.0.0.0",
            "target_port": 10001,
        },
    })

    assert "public_inference_url" not in slot.endpoint
    assert not hasattr(slot, "tunnel")


def test_config_without_endpoint_slot_removes_only_requested_slot():
    payload = {
        "enabled": True,
        "endpoints": [
            {"name": "lium-b200-1", "provider": "lium"},
            {"name": "lium-b200-temp-2", "provider": "lium"},
            {"name": "targon-b200-3", "provider": "targon"},
        ],
        "providers": {"lium": {"api_url": "http://lium"}},
    }

    updated, removed = _config_without_endpoint_slot(
        payload, "lium-b200-temp-2",
    )

    assert removed is True
    assert [slot["name"] for slot in updated["endpoints"]] == [
        "lium-b200-1",
        "targon-b200-3",
    ]
    assert payload["endpoints"][1]["name"] == "lium-b200-temp-2"


def test_instance_api_url_falls_back_to_provider_api_url(monkeypatch):
    monkeypatch.setenv("LIUM_API_URL", "https://lium.example.com/")
    cfg = InstanceAPIConfig.from_mapping("lium", {"create_path": "/instances"})
    assert cfg.resolved_api_url == "https://lium.example.com"


@pytest.mark.asyncio
async def test_scales_up_when_pending_exceeds_threshold():
    kv = InMemoryConfigStore()
    endpoints = _Endpoints()
    autoscaler = _autoscaler(
        _Queue(pending=3),
        endpoints,
        kv,
    )

    result = await autoscaler.tick(_config(threshold=2))

    assert result.action == "scale-up:2"
    assert sorted(endpoints.endpoints) == ["lium-b200-1", "lium-b200-2"]
    ep = endpoints.endpoints["lium-b200-1"]
    assert ep.active is True
    assert ep.autoscale_managed is True
    assert ep.autoscale_provider == "lium"
    assert ep.autoscale_instance_id == "inst-lium-b200-1"
    assert ep.autoscale_purpose == "eval"
    assert ep.autoscale_lease_expires_at == 1500
    assert ep.ssh_key_path == "/root/.ssh/affine_validator_server"
    assert ep.ssh_url == "ssh://root@lium-b200-1.example.com:22"
    assert _Client.create_calls[0] == (
        {
            "endpoint_name": "lium-b200-1",
            "provider": "lium",
            "purpose": "eval",
        },
        {
            "endpoint_name": "lium-b200-1",
            "provider": "lium",
            "purpose": "eval",
        },
    )


@pytest.mark.asyncio
async def test_scale_up_uses_provider_direct_url_and_port():
    _Client.create_public_inference_url = "http://203.0.113.10:45123/v1"
    _Client.create_sglang_port = 40000
    kv = InMemoryConfigStore()
    endpoints = _Endpoints()
    autoscaler = _autoscaler(_Queue(pending=1), endpoints, kv)
    cfg = GPUAutoscalerConfig(
        enabled=True,
        pending_threshold_per_instance=1,
        min_instances=0,
        max_instances=1,
        providers={
            "lium": InstanceAPIConfig(
                provider="lium",
                api_url="https://lium.example.com",
                create_path="/instances",
                delete_path="/instances/{instance_id}",
            )
        },
        slots=[
            ManagedEndpointSlot(
                name="lium-b200-1",
                provider="lium",
                endpoint={
                    "ssh_key_path": "/root/.ssh/affine_validator_server",
                    "public_inference_url": "http://validator.example.com:8102/v1",
                    "sglang_port": 10001,
                },
            )
        ],
    )

    result = await autoscaler.tick(cfg)

    assert result.action == "scale-up:1"
    endpoint = endpoints.endpoints["lium-b200-1"]
    assert endpoint.ssh_url == "ssh://root@lium-b200-1.example.com:22"
    assert endpoint.public_inference_url == "http://203.0.113.10:45123/v1"
    assert endpoint.sglang_port == 40000
    assert _Client.events == [
        ("create", "lium-b200-1"),
        ("activate", "lium-b200-1"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("public_url", "sglang_port"),
    [
        ("", 40000),
        ("http://203.0.113.10/v1", 40000),
        ("http://203.0.113.10:45123/models", 40000),
        ("http://203.0.113.10:45123/v1", 0),
    ],
)
async def test_scale_up_deletes_instance_without_usable_direct_endpoint(
    public_url,
    sglang_port,
):
    _Client.create_public_inference_url = public_url
    _Client.create_sglang_port = sglang_port
    kv = InMemoryConfigStore()
    endpoints = _Endpoints()
    autoscaler = _autoscaler(_Queue(pending=1), endpoints, kv)

    result = await autoscaler.tick(_config(threshold=1))

    assert result.action == "none"
    assert endpoints.endpoints == {}
    assert _Client.deleted == ["inst-lium-b200-1", "inst-lium-b200-2"]
    assert _Client.events == [
        ("create", "lium-b200-1"),
        ("delete", "inst-lium-b200-1"),
        ("create", "lium-b200-2"),
        ("delete", "inst-lium-b200-2"),
    ]


@pytest.mark.asyncio
async def test_scale_up_does_not_reuse_stale_lease_from_inactive_endpoint():
    _Client.create_lease_expires_at = 0
    kv = InMemoryConfigStore()
    endpoints = _Endpoints([
        Endpoint(
            name="lium-b200-1",
            kind="ssh",
            active=False,
            autoscale_managed=True,
            autoscale_provider="lium",
            autoscale_lease_expires_at=9999,
        )
    ])
    autoscaler = _autoscaler(
        _Queue(pending=1),
        endpoints,
        kv,
    )

    result = await autoscaler.tick(_config(threshold=1))

    assert result.action == "scale-up:1"
    ep = endpoints.endpoints["lium-b200-1"]
    assert ep.autoscale_instance_id == "inst-lium-b200-1"
    assert ep.autoscale_lease_expires_at == 0


@pytest.mark.asyncio
async def test_enqueue_manual_replacement_only_writes_request():
    kv = InMemoryConfigStore()
    request = await _enqueue_manual_replacement_request(
        kv,
        old_endpoint_name="lium-b200-1",
        new_slot_name="lium-b200-1",
        updated_by="test",
        now=1000,
    )

    stored = await kv.get(MANUAL_REPLACEMENT_LOCK_KEY)
    assert stored == request
    assert stored["request_version"] == MANUAL_REPLACEMENT_REQUEST_VERSION
    assert stored["status"] == "requested"
    assert stored["operation"] == MANUAL_ENDPOINT_OPERATION_REPLACE
    assert _Client.events == []
    assert _Client.create_calls == []
    assert _Client.deleted == []


@pytest.mark.asyncio
async def test_enqueue_health_replacement_is_fenced_and_idempotent():
    kv = InMemoryConfigStore()

    queued = await enqueue_health_endpoint_replacement_request(
        kv,
        endpoint_name="lium-b200-1",
        instance_id="inst-old",
        endpoint_generation=7,
        reason="ssh_and_public_unavailable",
        now=1000,
    )
    duplicate = await enqueue_health_endpoint_replacement_request(
        kv,
        endpoint_name="lium-b200-1",
        instance_id="inst-old",
        endpoint_generation=7,
        reason="ssh_and_public_unavailable",
        now=1001,
    )
    conflicting = await enqueue_health_endpoint_replacement_request(
        kv,
        endpoint_name="lium-b200-2",
        instance_id="inst-other",
        endpoint_generation=1,
        reason="ssh_and_public_unavailable",
        now=1001,
    )

    assert queued is True
    assert duplicate is True
    assert conflicting is False
    stored = await kv.get(MANUAL_REPLACEMENT_LOCK_KEY)
    assert stored["operation"] == MANUAL_ENDPOINT_OPERATION_REPLACE
    assert stored["old_endpoint_name"] == "lium-b200-1"
    assert stored["new_slot_name"] == "lium-b200-1"
    assert stored["expected_instance_id"] == "inst-old"
    assert stored["expected_generation"] == 7
    assert stored["reason"] == "ssh_and_public_unavailable"


@pytest.mark.asyncio
async def test_daemon_skips_health_replacement_after_endpoint_generation_changes():
    kv = InMemoryConfigStore()
    endpoint = Endpoint(
        name="lium-b200-1",
        kind="ssh",
        active=True,
        autoscale_managed=True,
        autoscale_provider="lium",
        autoscale_instance_id="inst-new",
        autoscale_purpose="eval",
        generation=8,
    )
    endpoints = _Endpoints([endpoint])
    autoscaler = _autoscaler(_Queue(), endpoints, kv, now=1000)
    queued = await enqueue_health_endpoint_replacement_request(
        kv,
        endpoint_name="lium-b200-1",
        instance_id="inst-old",
        endpoint_generation=7,
        reason="ssh_and_public_unavailable",
        now=1000,
    )

    result = await autoscaler.tick(_config())

    assert queued is True
    assert result.action == "manual-replacement:completed"
    assert _Client.events == []
    assert _Client.create_calls == []
    assert _Client.deleted == []
    current = endpoints.endpoints["lium-b200-1"]
    assert current.active is True
    assert current.autoscale_instance_id == "inst-new"
    assert current.generation == 8
    assert await kv.get(MANUAL_REPLACEMENT_LOCK_KEY) is None


@pytest.mark.asyncio
async def test_enqueue_manual_add_only_writes_request():
    kv = InMemoryConfigStore()
    request = await _enqueue_manual_add_request(
        kv,
        slot_name="lium-b200-2",
        updated_by="test",
        now=1000,
    )

    stored = await kv.get(MANUAL_REPLACEMENT_LOCK_KEY)
    assert stored == request
    assert stored["operation"] == MANUAL_ENDPOINT_OPERATION_ADD
    assert stored["old_endpoint_name"] == ""
    assert stored["new_slot_name"] == "lium-b200-2"
    assert _Client.events == []


@pytest.mark.asyncio
async def test_enqueue_manual_remove_only_writes_request():
    kv = InMemoryConfigStore()
    request = await _enqueue_manual_remove_request(
        kv,
        endpoint_name="lium-b200-1",
        keep_slot=True,
        updated_by="test",
        now=1000,
    )

    stored = await kv.get(MANUAL_REPLACEMENT_LOCK_KEY)
    assert stored == request
    assert stored["operation"] == MANUAL_ENDPOINT_OPERATION_REMOVE
    assert stored["old_endpoint_name"] == "lium-b200-1"
    assert stored["new_slot_name"] == ""
    assert stored["keep_slot"] is True
    assert _Client.events == []


@pytest.mark.asyncio
async def test_daemon_executes_add_without_replacing_an_endpoint():
    kv = InMemoryConfigStore()
    endpoints = _Endpoints()
    autoscaler = _autoscaler(
        _Queue(pending=10),
        endpoints,
        kv,
        now=1000,
    )
    request = await _enqueue_manual_add_request(
        kv,
        slot_name="lium-b200-2",
        updated_by="test",
        now=1000,
    )

    result = await autoscaler.tick(_config())

    assert result.action == "manual-add:completed"
    assert _Client.events == [
        ("create", "lium-b200-2"),
        ("activate", "lium-b200-2"),
    ]
    assert endpoints.endpoints["lium-b200-2"].active is True
    assert await kv.get(MANUAL_REPLACEMENT_LOCK_KEY) is None
    payload = await kv.get(_manual_endpoint_result_key(request["token"]))
    assert payload["status"] == "completed"
    assert payload["result"] == {
        "endpoint_name": "lium-b200-2",
        "slot_name": "lium-b200-2",
        "instance_id": "inst-lium-b200-2",
        "created": True,
        "dry_run": False,
    }
    state = await kv.get(STATE_KEY)
    assert state["last_action"] == "manual-add:completed"


@pytest.mark.asyncio
async def test_daemon_executes_remove_and_removes_slot():
    kv = InMemoryConfigStore()
    await kv.set(CONFIG_KEY, _config_payload())
    old = Endpoint(
        name="lium-b200-1",
        kind="ssh",
        active=True,
        autoscale_managed=True,
        autoscale_provider="lium",
        autoscale_instance_id="old-inst",
        autoscale_purpose="eval",
    )
    endpoints = _Endpoints([old])
    autoscaler = _autoscaler(_Queue(), endpoints, kv, now=1000)
    request = await _enqueue_manual_remove_request(
        kv,
        endpoint_name="lium-b200-1",
        keep_slot=False,
        updated_by="test",
        now=1000,
    )

    result = await autoscaler.tick(_config())

    assert result.action == "manual-remove:completed"
    assert _Client.events == [
        ("drain", "lium-b200-1", "old-inst"),
        ("delete", "old-inst"),
        ("deactivate", "lium-b200-1", "old-inst"),
    ]
    config_payload = await kv.get(CONFIG_KEY)
    assert [slot["name"] for slot in config_payload["endpoints"]] == [
        "lium-b200-2"
    ]
    assert await kv.get(MANUAL_REPLACEMENT_LOCK_KEY) is None
    payload = await kv.get(_manual_endpoint_result_key(request["token"]))
    assert payload["status"] == "completed"
    assert payload["result"] == {
        "endpoint_name": "lium-b200-1",
        "instance_id": "old-inst",
        "deleted": True,
        "slot_removed": True,
        "dry_run": False,
    }


@pytest.mark.asyncio
async def test_failed_remove_restores_slot_before_releasing_request():
    _Client.delete_returns_false_for_ids = {"old-inst"}
    kv = InMemoryConfigStore()
    await kv.set(CONFIG_KEY, _config_payload())
    old = Endpoint(
        name="lium-b200-1",
        kind="ssh",
        active=True,
        autoscale_managed=True,
        autoscale_provider="lium",
        autoscale_instance_id="old-inst",
        autoscale_purpose="eval",
    )
    endpoints = _Endpoints([old])
    autoscaler = _autoscaler(_Queue(), endpoints, kv, now=1000)
    request = await _enqueue_manual_remove_request(
        kv,
        endpoint_name="lium-b200-1",
        keep_slot=False,
        updated_by="test",
        now=1000,
    )

    result = await autoscaler.tick(_config())

    assert result.action == "manual-remove:failed"
    config_payload = await kv.get(CONFIG_KEY)
    assert [slot["name"] for slot in config_payload["endpoints"]] == [
        "lium-b200-1",
        "lium-b200-2",
    ]
    assert endpoints.endpoints["lium-b200-1"].active is True
    assert endpoints.endpoints["lium-b200-1"].autoscale_instance_id == "old-inst"
    assert await kv.get(MANUAL_REPLACEMENT_LOCK_KEY) is None
    payload = await kv.get(_manual_endpoint_result_key(request["token"]))
    assert payload["status"] == "failed"
    assert "failed to delete endpoint" in payload["error"]


@pytest.mark.asyncio
async def test_daemon_treats_legacy_keep_old_request_as_add():
    kv = InMemoryConfigStore()
    old = Endpoint(
        name="lium-b200-1",
        kind="ssh",
        active=True,
        autoscale_managed=True,
        autoscale_provider="lium",
        autoscale_instance_id="old-inst",
    )
    endpoints = _Endpoints([old])
    autoscaler = _autoscaler(_Queue(), endpoints, kv, now=1000)
    await kv.set(
        MANUAL_REPLACEMENT_LOCK_KEY,
        {
            "request_version": MANUAL_REPLACEMENT_REQUEST_VERSION,
            "status": "requested",
            "token": "legacy-keep-old",
            "old_endpoint_name": "lium-b200-1",
            "new_slot_name": "lium-b200-2",
            "delete_old": False,
            "requested_at": 1000,
            "expires_at": 1100,
            "updated_by": "test",
        },
    )

    result = await autoscaler.tick(_config())

    assert result.action == "manual-add:completed"
    assert _Client.events == [
        ("create", "lium-b200-2"),
        ("activate", "lium-b200-2"),
    ]
    assert endpoints.endpoints["lium-b200-1"].active is True
    assert endpoints.endpoints["lium-b200-2"].active is True


@pytest.mark.asyncio
async def test_daemon_executes_replacement_without_scaling_fallback_slot():
    kv = InMemoryConfigStore()
    old = Endpoint(
        name="lium-b200-1",
        kind="ssh",
        active=True,
        role="scoring",
        ssh_url="ssh://root@old.example.com:22",
        public_inference_url="http://validator.example.com:8102/v1",
        autoscale_managed=True,
        autoscale_provider="lium",
        autoscale_instance_id="old-inst",
        autoscale_purpose="eval",
    )
    endpoints = _Endpoints([old])
    autoscaler = _autoscaler(
        _Queue(pending=10),
        endpoints,
        kv,
        now=1000,
    )
    cfg = GPUAutoscalerConfig(
        enabled=True,
        pending_threshold_per_instance=1,
        max_gpu_down_wait_seconds=0,
        min_instances=0,
        max_instances=2,
        providers={
            "lium": InstanceAPIConfig(
                provider="lium",
                api_url="https://lium.example.com",
                create_path="/instances",
                delete_path="/instances/{instance_id}",
            )
        },
        slots=[
            ManagedEndpointSlot(
                name="lium-b200-1",
                provider="lium",
            ),
            ManagedEndpointSlot(name="lium-b300-fallback", provider="lium"),
        ],
    )
    queued = await enqueue_health_endpoint_replacement_request(
        kv,
        endpoint_name="lium-b200-1",
        instance_id="old-inst",
        endpoint_generation=0,
        reason="ssh_and_public_unavailable",
        now=1000,
    )
    request = await kv.get(MANUAL_REPLACEMENT_LOCK_KEY)

    result = await autoscaler.tick(cfg)

    assert queued is True
    assert result.action == "manual-replacement:completed"
    assert _Client.events == [
        ("drain", "lium-b200-1", "old-inst"),
        ("delete", "old-inst"),
        ("deactivate", "lium-b200-1", "old-inst"),
        ("create", "lium-b200-1"),
        ("activate", "lium-b200-1"),
    ]
    assert all(event != ("create", "lium-b300-fallback") for event in _Client.events)
    assert await kv.get(MANUAL_REPLACEMENT_LOCK_KEY) is None
    replacement = await kv.get(_manual_endpoint_result_key(request["token"]))
    assert replacement["status"] == "completed"
    state = await kv.get(STATE_KEY)
    assert state["last_tick_at"] == 1000
    assert state["last_action"] == "manual-replacement:completed"
    assert state["last_manual_replacement"]["status"] == "completed"


@pytest.mark.asyncio
async def test_terminal_result_prevents_replaying_request_left_after_cleanup():
    kv = InMemoryConfigStore()
    original_delete_if_token = kv.delete_if_token
    delete_attempts = 0

    async def fail_first_request_delete(key, token, *, token_field="token"):
        nonlocal delete_attempts
        if key == MANUAL_REPLACEMENT_LOCK_KEY:
            delete_attempts += 1
            if delete_attempts == 1:
                return False
        return await original_delete_if_token(
            key,
            token,
            token_field=token_field,
        )

    kv.delete_if_token = fail_first_request_delete
    old = Endpoint(
        name="lium-b200-1",
        kind="ssh",
        active=True,
        autoscale_managed=True,
        autoscale_provider="lium",
        autoscale_instance_id="old-inst",
        autoscale_purpose="eval",
    )
    autoscaler = _autoscaler(
        _Queue(pending=1),
        _Endpoints([old]),
        kv,
        now=1000,
    )
    await _enqueue_manual_replacement_request(
        kv,
        old_endpoint_name="lium-b200-1",
        new_slot_name="lium-b200-1",
        updated_by="test",
        now=1000,
    )

    config = _config(lease_renew_margin_seconds=0)
    first = await autoscaler.tick(config)
    events_after_replacement = list(_Client.events)
    assert await kv.get(MANUAL_REPLACEMENT_LOCK_KEY) is not None

    second = await autoscaler.tick(config)

    assert first.action == "manual-replacement:completed"
    assert second.action == "none"
    assert _Client.events == events_after_replacement
    assert await kv.get(MANUAL_REPLACEMENT_LOCK_KEY) is None
    assert delete_attempts == 2


@pytest.mark.asyncio
async def test_request_arriving_during_tick_waits_for_daemon_boundary():
    kv = InMemoryConfigStore()
    old = Endpoint(
        name="lium-b200-1",
        kind="ssh",
        active=True,
        role="scoring",
        autoscale_managed=True,
        autoscale_provider="lium",
        autoscale_instance_id="old-inst",
        autoscale_purpose="eval",
    )
    endpoints = _Endpoints([old])
    autoscaler = _autoscaler(
        _Queue(pending=1),
        endpoints,
        kv,
        now=1000,
    )
    entered = asyncio.Event()
    release = asyncio.Event()
    original_snapshot = autoscaler._snapshot

    async def blocked_snapshot():
        entered.set()
        await release.wait()
        return await original_snapshot()

    autoscaler._snapshot = blocked_snapshot
    first_tick = asyncio.create_task(autoscaler.tick(_config(threshold=1)))
    await entered.wait()
    await _enqueue_manual_replacement_request(
        kv,
        old_endpoint_name="lium-b200-1",
        new_slot_name="lium-b200-1",
        updated_by="test",
        now=1000,
    )

    assert _Client.events == []
    release.set()
    first_result = await first_tick
    events_after_first_tick = list(_Client.events)
    second_result = await autoscaler.tick(_config(threshold=1))

    assert first_result.action == "none"
    assert events_after_first_tick == []
    assert second_result.action == "manual-replacement:completed"
    assert _Client.events[0] == ("drain", "lium-b200-1", "old-inst")


@pytest.mark.asyncio
async def test_manual_replacement_updates_heartbeat_while_running(monkeypatch):
    monkeypatch.setattr(
        "affine.src.scheduler.gpu_autoscaler.MANUAL_REPLACEMENT_HEARTBEAT_SECONDS",
        0.01,
    )
    kv = InMemoryConfigStore()
    old = Endpoint(
        name="lium-b200-1",
        kind="ssh",
        active=True,
        autoscale_managed=True,
        autoscale_provider="lium",
        autoscale_instance_id="old-inst",
        autoscale_purpose="eval",
    )
    endpoints = _Endpoints([old])
    autoscaler = _autoscaler(
        _Queue(pending=1),
        endpoints,
        kv,
        now=1000,
    )
    now = [1000]
    autoscaler._now = lambda: now[0]
    entered = asyncio.Event()
    release = asyncio.Event()

    async def slow_replace(*args, **kwargs):
        entered.set()
        await release.wait()
        return EndpointReplaceResult(
            old_endpoint_name="lium-b200-1",
            new_endpoint_name="lium-b200-1",
            new_slot_name="lium-b200-1",
            old_instance_id="old-inst",
            new_instance_id="new-inst",
            old_deleted=True,
            new_created=True,
            same_endpoint=True,
        )

    autoscaler.replace_endpoint = slow_replace
    await _enqueue_manual_replacement_request(
        kv,
        old_endpoint_name="lium-b200-1",
        new_slot_name="lium-b200-1",
        updated_by="test",
        now=1000,
    )

    tick_task = asyncio.create_task(autoscaler.tick(_config()))
    await entered.wait()
    now[0] = 1010
    await asyncio.sleep(0.03)

    running_state = await kv.get(STATE_KEY)
    assert running_state["last_tick_at"] == 1010
    assert running_state["last_action"] == "manual-replacement:running"
    release.set()
    result = await tick_task
    assert result.action == "manual-replacement:completed"


@pytest.mark.asyncio
async def test_cancelled_replacement_records_failure_and_releases_request():
    kv = InMemoryConfigStore()
    old = Endpoint(
        name="lium-b200-1",
        kind="ssh",
        active=True,
        autoscale_managed=True,
        autoscale_provider="lium",
        autoscale_instance_id="old-inst",
        autoscale_purpose="eval",
    )
    autoscaler = _autoscaler(
        _Queue(pending=1),
        _Endpoints([old]),
        kv,
        now=1000,
    )
    entered = asyncio.Event()

    async def cancelled_replace(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    autoscaler.replace_endpoint = cancelled_replace
    request = await _enqueue_manual_replacement_request(
        kv,
        old_endpoint_name="lium-b200-1",
        new_slot_name="lium-b200-1",
        updated_by="test",
        now=1000,
    )

    tick_task = asyncio.create_task(autoscaler.tick(_config()))
    await entered.wait()
    tick_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await tick_task

    assert await kv.get(MANUAL_REPLACEMENT_LOCK_KEY) is None
    failure = await kv.get(_manual_endpoint_result_key(request["token"]))
    assert failure["status"] == "failed"
    assert failure["error_type"] == "CancelledError"
    state = await kv.get(STATE_KEY)
    assert state["last_action"] == "manual-replacement:failed"


@pytest.mark.asyncio
async def test_expired_manual_replacement_lock_allows_scale_up():
    kv = InMemoryConfigStore()
    await kv.set(
        MANUAL_REPLACEMENT_LOCK_KEY,
        {
            "request_version": MANUAL_REPLACEMENT_REQUEST_VERSION,
            "status": "requested",
            "old_endpoint_name": "lium-b200-1",
            "new_slot_name": "lium-b200-1",
            "requested_at": 800,
            "expires_at": 999,
            "token": "replace-1",
        },
    )
    endpoints = _Endpoints()
    autoscaler = _autoscaler(
        _Queue(pending=1),
        endpoints,
        kv,
        now=1000,
    )

    cfg = GPUAutoscalerConfig(
        enabled=True,
        pending_threshold_per_instance=1,
        max_gpu_down_wait_seconds=0,
        min_instances=0,
        max_instances=1,
        providers={
            "lium": InstanceAPIConfig(
                provider="lium",
                api_url="https://lium.example.com",
                create_path="/instances",
                delete_path="/instances/{instance_id}",
            )
        },
        slots=[ManagedEndpointSlot(name="lium-b200-1", provider="lium")],
    )

    result = await autoscaler.tick(cfg)

    assert result.action == "scale-up:1"
    assert sorted(endpoints.endpoints) == ["lium-b200-1"]
    assert await kv.get(MANUAL_REPLACEMENT_LOCK_KEY) is None


@pytest.mark.asyncio
async def test_failed_replacement_releases_request_and_next_tick_recovers():
    _Client.create_returns_none_for_names = {"lium-b200-1"}
    kv = InMemoryConfigStore()
    old = Endpoint(
        name="lium-b200-1",
        kind="ssh",
        active=True,
        autoscale_managed=True,
        autoscale_provider="lium",
        autoscale_instance_id="old-inst",
        autoscale_purpose="eval",
    )
    endpoints = _Endpoints([old])
    autoscaler = _autoscaler(
        _Queue(pending=1),
        endpoints,
        kv,
        now=1000,
    )
    request = await _enqueue_manual_replacement_request(
        kv,
        old_endpoint_name="lium-b200-1",
        new_slot_name="lium-b200-1",
        updated_by="test",
        now=1000,
    )
    cfg = _config(threshold=5, max_gpu_down_wait_seconds=3600)

    first = await autoscaler.tick(cfg)

    assert first.action == "manual-replacement:failed"
    assert await kv.get(MANUAL_REPLACEMENT_LOCK_KEY) is None
    failure = await kv.get(_manual_endpoint_result_key(request["token"]))
    assert failure["status"] == "failed"
    assert failure["error_type"] == "RuntimeError"
    assert "was already deleted" in failure["error"]
    state = await kv.get(STATE_KEY)
    assert state["last_action"] == "manual-replacement:failed"
    assert state["last_manual_replacement"]["status"] == "failed"

    _Client.create_returns_none_for_names = set()
    second = await autoscaler.tick(cfg)

    assert second.action == "scale-up:1"
    assert second.desired_instances == 1
    assert sorted(endpoints.endpoints) == ["lium-b200-1"]


@pytest.mark.asyncio
async def test_legacy_manual_replacement_lock_is_discarded_without_pause():
    kv = InMemoryConfigStore()
    await kv.set(
        STATE_KEY,
        {
            MANUAL_REPLACEMENT_STATE_KEY: {
                "old_endpoint_name": "lium-b200-1",
                "new_slot_name": "lium-b200-1",
                "started_at": 900,
                "expires_at": 1100,
                "token": "legacy-replace",
            }
        },
    )
    endpoints = _Endpoints()
    autoscaler = _autoscaler(
        _Queue(pending=1),
        endpoints,
        kv,
        now=1000,
    )

    cfg = GPUAutoscalerConfig(
        enabled=True,
        pending_threshold_per_instance=1,
        max_gpu_down_wait_seconds=0,
        min_instances=0,
        max_instances=1,
        providers={
            "lium": InstanceAPIConfig(
                provider="lium",
                api_url="https://lium.example.com",
                create_path="/instances",
                delete_path="/instances/{instance_id}",
            )
        },
        slots=[ManagedEndpointSlot(name="lium-b200-1", provider="lium")],
    )

    result = await autoscaler.tick(cfg)

    assert result.action == "scale-up:1"
    assert sorted(endpoints.endpoints) == ["lium-b200-1"]
    state = await kv.get(STATE_KEY)
    assert MANUAL_REPLACEMENT_STATE_KEY not in state
    assert state["last_tick_at"] == 1000
    assert state["last_action"] == "scale-up:1"
    assert state["last_manual_replacement"]["status"] == ("discarded-legacy-lock")


@pytest.mark.asyncio
async def test_duplicate_manual_replacement_request_is_rejected():
    kv = InMemoryConfigStore()
    await _enqueue_manual_replacement_request(
        kv,
        old_endpoint_name="lium-b200-1",
        new_slot_name="lium-b200-1",
        updated_by="test",
        now=1000,
    )

    with pytest.raises(RuntimeError, match="already queued"):
        await _enqueue_manual_replacement_request(
            kv,
            old_endpoint_name="lium-b200-2",
            new_slot_name="lium-b200-2",
            updated_by="test",
            now=1001,
        )


@pytest.mark.asyncio
async def test_result_wait_tolerates_eventually_consistent_reads(monkeypatch):
    monkeypatch.setattr(
        "affine.src.scheduler.gpu_autoscaler."
        "MANUAL_REPLACEMENT_RESULT_VISIBILITY_GRACE_SECONDS",
        1.0,
    )
    kv = InMemoryConfigStore()
    request = {
        "token": "replace-eventual",
    }
    result_key = _manual_endpoint_result_key(request["token"])
    await kv.set(
        result_key,
        {
            "token": request["token"],
            "status": "completed",
            "result": {},
        },
    )
    original_get = kv.get
    result_reads = 0

    async def eventually_consistent_get(key, default=None):
        nonlocal result_reads
        if key == result_key and result_reads < 2:
            result_reads += 1
            return None
        return await original_get(key, default)

    kv.get = eventually_consistent_get

    result = await _wait_for_manual_endpoint_result(
        kv,
        request,
        timeout_seconds=1,
        poll_seconds=0.05,
    )

    assert result["status"] == "completed"
    assert result_reads == 2


@pytest.mark.asyncio
async def test_scale_up_deletes_instance_when_endpoint_activation_fails():
    kv = InMemoryConfigStore()
    endpoints = _Endpoints()
    endpoints.fail_activate = True
    autoscaler = _autoscaler(
        _Queue(pending=1),
        endpoints,
        kv,
    )

    result = await autoscaler.tick(_config(threshold=1))

    assert result.action == "none"
    assert _Client.deleted == ["inst-lium-b200-1", "inst-lium-b200-2"]
    assert _Client.delete_calls == [
        (
            "inst-lium-b200-1",
            {
                "endpoint_name": "lium-b200-1",
                "provider": "lium",
                "purpose": "eval",
            },
        ),
        (
            "inst-lium-b200-2",
            {
                "endpoint_name": "lium-b200-2",
                "provider": "lium",
                "purpose": "eval",
            },
        ),
    ]
    assert endpoints.endpoints == {}


@pytest.mark.asyncio
async def test_scale_up_preserves_existing_assignment_fields():
    kv = InMemoryConfigStore()
    endpoints = _Endpoints([
        Endpoint(
            name="lium-b200-1",
            kind="ssh",
            active=False,
            assigned_uid=42,
            assigned_hotkey="hk42",
            assigned_model="org/model42",
            assigned_revision="rev42",
            deployment_id="ssh:old",
            base_url="http://old.example/v1",
            assignment_role="pre_challenger",
            assigned_at=900,
        )
    ])
    autoscaler = _autoscaler(
        _Queue(pending=1),
        endpoints,
        kv,
    )

    result = await autoscaler.tick(_config(threshold=1))

    assert result.action == "scale-up:1"
    ep = endpoints.endpoints["lium-b200-1"]
    assert ep.active is True
    assert ep.autoscale_instance_id == "inst-lium-b200-1"
    assert ep.assigned_uid == 42
    assert ep.assigned_hotkey == "hk42"
    assert ep.assigned_model == "org/model42"
    assert ep.assigned_revision == "rev42"
    assert ep.deployment_id == "ssh:old"
    assert ep.base_url == "http://old.example/v1"
    assert ep.assignment_role == "pre_challenger"
    assert ep.assigned_at == 900


@pytest.mark.asyncio
async def test_waits_when_pending_is_below_threshold_before_max_wait():
    kv = InMemoryConfigStore()
    endpoints = _Endpoints()
    queue = _Queue(pending=0)
    autoscaler = _autoscaler(
        queue,
        endpoints,
        kv,
        now=1000,
    )
    await autoscaler.tick(_config(threshold=5, max_gpu_down_wait_seconds=3600))
    queue.pending = 4

    result = await autoscaler.tick(
        _config(threshold=5, max_gpu_down_wait_seconds=3600)
    )

    assert result.action == "none"
    assert result.desired_instances == 0
    assert result.gpu_down_for_sec == 0
    assert endpoints.endpoints == {}
    state = await kv.get(STATE_KEY)
    assert state["gpu_down_at"] == 1000


@pytest.mark.asyncio
async def test_scales_up_below_threshold_after_gpu_has_been_down_long_enough():
    kv = InMemoryConfigStore()
    await kv.set(STATE_KEY, {"gpu_down_at": 900})
    endpoints = _Endpoints()
    autoscaler = _autoscaler(
        _Queue(pending=1),
        endpoints,
        kv,
        now=1000,
    )

    result = await autoscaler.tick(
        _config(threshold=5, max_gpu_down_wait_seconds=60)
    )

    assert result.action == "scale-up:1"
    assert result.desired_instances == 1
    assert result.gpu_down_for_sec == 100
    assert sorted(endpoints.endpoints) == ["lium-b200-1"]
    state = await kv.get(STATE_KEY)
    assert "gpu_down_at" not in state


@pytest.mark.asyncio
async def test_restart_without_gpu_down_state_scales_up_immediately_when_pending():
    kv = InMemoryConfigStore()
    endpoints = _Endpoints()
    autoscaler = _autoscaler(
        _Queue(pending=1),
        endpoints,
        kv,
        now=1000,
    )

    result = await autoscaler.tick(
        _config(threshold=5, max_gpu_down_wait_seconds=60)
    )

    assert result.action == "scale-up:1"
    assert result.desired_instances == 1
    assert result.gpu_down_for_sec == 0
    assert sorted(endpoints.endpoints) == ["lium-b200-1"]


@pytest.mark.asyncio
async def test_restart_force_start_is_one_shot_after_failed_create():
    kv = InMemoryConfigStore()
    endpoints = _Endpoints()
    autoscaler = _autoscaler(
        _Queue(pending=1),
        endpoints,
        kv,
        now=1000,
    )
    cfg = GPUAutoscalerConfig(
        enabled=True,
        pending_threshold_per_instance=5,
        max_gpu_down_wait_seconds=12 * 60 * 60,
        min_instances=0,
        max_instances=1,
        providers={},
        slots=[ManagedEndpointSlot(name="lium-b200-1", provider="lium")],
    )

    first = await autoscaler.tick(cfg)
    second = await autoscaler.tick(cfg)

    assert first.desired_instances == 1
    assert first.action == "none"
    assert second.desired_instances == 0
    assert second.action == "none"
    assert endpoints.endpoints == {}


@pytest.mark.asyncio
async def test_restart_without_pending_does_not_scale_up():
    kv = InMemoryConfigStore()
    endpoints = _Endpoints()
    autoscaler = _autoscaler(
        _Queue(pending=0),
        endpoints,
        kv,
        now=1000,
    )

    result = await autoscaler.tick(
        _config(threshold=5, max_gpu_down_wait_seconds=60)
    )

    assert result.action == "none"
    assert result.desired_instances == 0
    assert endpoints.endpoints == {}


@pytest.mark.asyncio
async def test_missing_capacity_recovers_incomplete_champion_without_pending():
    kv = InMemoryConfigStore()
    state = StateStore(kv)
    await state.set_champion(ChampionRecord(
        uid=7,
        hotkey="hk7",
        revision="rev7",
        model="repo/model",
    ))
    await kv.set(
        "current_task_ids",
        {"task_ids": {"MEMORY": [1, 2, 3]}, "refreshed_at_block": 100},
    )
    await kv.set(
        "environments",
        {
            "MEMORY": {
                "enabled_for_sampling": True,
                "enabled_for_scoring": True,
                "sampling": {
                    "sampling_count": 3,
                    "dataset_range": [[1, 3]],
                },
            }
        },
    )
    endpoints = _Endpoints()
    autoscaler = _autoscaler(
        _Queue(pending=0),
        endpoints,
        kv,
        now=1000,
        samples=_Samples(count=0),
    )

    result = await autoscaler.tick(
        _config(threshold=5, max_gpu_down_wait_seconds=12 * 60 * 60)
    )

    assert result.action == "scale-up:1"
    assert result.desired_instances == 1
    assert sorted(endpoints.endpoints) == ["lium-b200-1"]


@pytest.mark.asyncio
async def test_scales_down_idle_managed_endpoint_and_clears_champion_deployment():
    _Client.deleted = []
    kv = InMemoryConfigStore()
    state = StateStore(kv)
    await state.set_champion(
        ChampionRecord(
            uid=7,
            hotkey="hk7",
            revision="rev7",
            model="repo/model",
            deployment_id="ssh:lium-b200-1:affine-sglang-current",
            base_url="http://lium-b200-1.example.com:10001/v1",
            deployments=[
                DeploymentRecord(
                    endpoint_name="lium-b200-1",
                    deployment_id="ssh:lium-b200-1:affine-sglang-current",
                    base_url="http://lium-b200-1.example.com:10001/v1",
                )
            ],
        )
    )
    await kv.set(STATE_KEY, {"last_busy_at": 900})
    endpoint = Endpoint(
        name="lium-b200-1",
        kind="ssh",
        active=True,
        assigned_uid=7,
        deployment_id="ssh:lium-b200-1:affine-sglang-current",
        base_url="http://lium-b200-1.example.com:10001/v1",
        autoscale_managed=True,
        autoscale_provider="lium",
        autoscale_instance_id="inst-lium-b200-1",
        autoscale_purpose="eval",
        autoscale_lease_expires_at=1500,
    )
    endpoints = _Endpoints([endpoint])
    autoscaler = _autoscaler(
        _Queue(pending=0),
        endpoints,
        kv,
        now=1000,
    )

    result = await autoscaler.tick(_config(idle_seconds=60))

    assert result.action == "scale-down:1"
    assert _Client.deleted == ["inst-lium-b200-1"]
    assert _Client.delete_calls == [
        (
            "inst-lium-b200-1",
            {
                "endpoint_name": "lium-b200-1",
                "purpose": "eval",
            },
        )
    ]
    assert endpoints.cleared == []
    ep = endpoints.endpoints["lium-b200-1"]
    assert ep.active is False
    assert ep.ssh_url is None
    assert ep.autoscale_instance_id is None
    assert ep.autoscale_purpose is None
    assert ep.autoscale_lease_expires_at == 0
    champion = await state.get_champion()
    assert champion.deployment_id is None
    assert champion.base_url is None
    assert champion.deployments == []


@pytest.mark.asyncio
async def test_scale_down_clears_champion_deployment_by_base_url_fallback():
    _Client.deleted = []
    kv = InMemoryConfigStore()
    state = StateStore(kv)
    await state.set_champion(
        ChampionRecord(
            uid=7,
            hotkey="hk7",
            revision="rev7",
            model="repo/model",
            deployment_id=None,
            base_url="http://lium-b200-1.example.com:10001/v1",
            deployments=[
                DeploymentRecord(
                    endpoint_name="",
                    deployment_id="ssh:lium-b200-1:affine-sglang-current",
                    base_url="http://lium-b200-1.example.com:10001/v1",
                )
            ],
        )
    )
    await kv.set(STATE_KEY, {"last_busy_at": 900})
    endpoint = Endpoint(
        name="lium-b200-1",
        kind="ssh",
        active=True,
        assigned_uid=7,
        deployment_id=None,
        base_url="http://lium-b200-1.example.com:10001/v1",
        autoscale_managed=True,
        autoscale_provider="lium",
        autoscale_instance_id="inst-lium-b200-1",
    )
    endpoints = _Endpoints([endpoint])
    autoscaler = _autoscaler(
        _Queue(pending=0),
        endpoints,
        kv,
        now=1000,
    )

    result = await autoscaler.tick(_config(idle_seconds=60))

    assert result.action == "scale-down:1"
    champion = await state.get_champion()
    assert champion.deployment_id is None
    assert champion.base_url is None
    assert champion.deployments == []


@pytest.mark.asyncio
async def test_does_not_scale_down_while_champion_samples_are_incomplete():
    _Client.deleted = []
    kv = InMemoryConfigStore()
    state = StateStore(kv)
    await state.set_champion(
        ChampionRecord(
            uid=7,
            hotkey="hk7",
            revision="rev7",
            model="repo/model",
            deployment_id="ssh:lium-b200-1:affine-sglang-current",
            base_url="http://lium-b200-1.example.com:10001/v1",
        )
    )
    await kv.set(
        "current_task_ids",
        {"task_ids": {"MEMORY": [1, 2, 3]}, "refreshed_at_block": 100},
    )
    await kv.set(
        "environments",
        {
            "MEMORY": {
                "enabled_for_sampling": True,
                "sampling": {"sampling_count": 3, "dataset_range": [[1, 3]]},
            }
        },
    )
    await kv.set(STATE_KEY, {"last_busy_at": 900})
    endpoints = _Endpoints([
        Endpoint(
            name="lium-b200-1",
            kind="ssh",
            active=True,
            assigned_uid=7,
            autoscale_managed=True,
            autoscale_provider="lium",
            autoscale_instance_id="inst-lium-b200-1",
        )
    ])
    autoscaler = _autoscaler(
        _Queue(pending=0),
        endpoints,
        kv,
        now=1000,
        samples=_Samples(count=1),
    )

    result = await autoscaler.tick(_config(idle_seconds=60))

    assert result.action == "none"
    assert result.idle is False
    assert _Client.deleted == []
    assert endpoints.endpoints["lium-b200-1"].active is True


@pytest.mark.asyncio
async def test_champion_completion_uses_scheduler_threshold_before_scale_down():
    kv = InMemoryConfigStore()
    state = StateStore(kv)
    await state.set_champion(
        ChampionRecord(
            uid=7,
            hotkey="hk7",
            revision="rev7",
            model="repo/model",
            deployment_id="ssh:lium-b200-1:affine-sglang-current",
            base_url="http://lium-b200-1.example.com:10001/v1",
        )
    )
    await kv.set(
        "current_task_ids",
        {
            "task_ids": {"MEMORY": list(range(220))},
            "refreshed_at_block": 100,
        },
    )
    await kv.set(
        "environments",
        {
            "MEMORY": {
                "enabled_for_sampling": True,
                "sampling": {
                    "sampling_count": 200,
                    "dataset_range": [[0, 219]],
                },
            }
        },
    )
    await kv.set(STATE_KEY, {"last_busy_at": 900})
    endpoints = _Endpoints([
        Endpoint(
            name="lium-b200-1",
            kind="ssh",
            active=True,
            assigned_uid=7,
            autoscale_managed=True,
            autoscale_provider="lium",
            autoscale_instance_id="inst-lium-b200-1",
        )
    ])
    autoscaler = _autoscaler(
        _Queue(pending=0),
        endpoints,
        kv,
        now=1000,
        samples=_Samples(count=200),
    )

    result = await autoscaler.tick(_config(idle_seconds=60))

    assert result.action == "none"
    assert result.idle is False
    assert _Client.deleted == []
    assert endpoints.endpoints["lium-b200-1"].active is True


@pytest.mark.asyncio
async def test_renews_busy_managed_endpoint_near_lease_expiry():
    _Client.renewed = []
    _Client.renew_expires_at = 5000
    kv = InMemoryConfigStore()
    endpoint = Endpoint(
        name="lium-b200-1",
        kind="ssh",
        active=True,
        autoscale_managed=True,
        autoscale_provider="lium",
        autoscale_instance_id="inst-lium-b200-1",
        autoscale_created_at=100,
        autoscale_updated_at=100,
        autoscale_lease_expires_at=1050,
    )
    endpoints = _Endpoints([endpoint])
    autoscaler = _autoscaler(
        _Queue(pending=1),
        endpoints,
        kv,
        now=1000,
    )

    result = await autoscaler.tick(
        _config(
            threshold=5,
            idle_seconds=3600,
            lease_renew_margin_seconds=60,
        )
    )

    assert result.action == "renew-lease:1"
    assert result.lease_renewed_count == 1
    assert _Client.renewed == ["inst-lium-b200-1"]
    ep = endpoints.endpoints["lium-b200-1"]
    assert ep.autoscale_updated_at == 1000
    assert ep.autoscale_lease_expires_at == 5000
    state = await kv.get(STATE_KEY)
    assert "lease_renew_attempted_at" not in state


@pytest.mark.asyncio
async def test_replaces_active_endpoint_when_renew_reports_provider_reclaimed():
    _Client.renew_not_found_for_ids = {"old-inst"}
    kv = InMemoryConfigStore()
    endpoint = Endpoint(
        name="lium-b200-1",
        kind="ssh",
        active=True,
        autoscale_managed=True,
        autoscale_provider="lium",
        autoscale_instance_id="old-inst",
        autoscale_created_at=100,
        autoscale_updated_at=100,
        autoscale_lease_expires_at=1050,
        deployment_id="ssh:lium-b200-1:affine-sglang-current",
        base_url="http://old-lium.example.com:10001/v1",
    )
    endpoints = _Endpoints([endpoint])
    autoscaler = _autoscaler(
        _Queue(pending=1),
        endpoints,
        kv,
        now=1000,
    )

    result = await autoscaler.tick(
        _config(
            threshold=1,
            idle_seconds=3600,
            lease_renew_margin_seconds=60,
        )
    )

    assert result.action == "scale-up:1"
    assert result.active_capacity_count == 0
    assert result.lease_reclaimed_count == 1
    assert _Client.renewed == ["old-inst"]
    assert _Client.create_calls[0][0]["endpoint_name"] == "lium-b200-1"
    assert ("deactivate", "lium-b200-1", "old-inst") in _Client.events
    ep = endpoints.endpoints["lium-b200-1"]
    assert ep.active is True
    assert ep.autoscale_instance_id == "inst-lium-b200-1"
    assert ep.deployment_id is None
    assert ep.base_url is None
    state = await kv.get(STATE_KEY)
    assert state["last_lease_reclaimed_count"] == 1
    assert "lease_renew_attempted_at" not in state


@pytest.mark.asyncio
async def test_replaces_active_endpoint_when_status_reports_provider_reclaimed():
    _Client.status_not_found_for_ids = {"old-inst"}
    kv = InMemoryConfigStore()
    endpoint = Endpoint(
        name="lium-b200-1",
        kind="ssh",
        active=True,
        autoscale_managed=True,
        autoscale_provider="lium",
        autoscale_instance_id="old-inst",
        autoscale_created_at=100,
        autoscale_updated_at=100,
        autoscale_lease_expires_at=5000,
        deployment_id="ssh:lium-b200-1:affine-sglang-current",
        base_url="http://old-lium.example.com:10001/v1",
    )
    endpoints = _Endpoints([endpoint])
    autoscaler = _autoscaler(
        _Queue(pending=1),
        endpoints,
        kv,
        now=1000,
    )

    result = await autoscaler.tick(
        _config(
            threshold=1,
            idle_seconds=3600,
            lease_renew_margin_seconds=60,
            status_path="/instances/{instance_id}",
        )
    )

    assert result.action == "scale-up:1"
    assert result.active_capacity_count == 0
    assert result.endpoint_health_checked_count == 1
    assert result.endpoint_health_reclaimed_count == 1
    assert result.lease_reclaimed_count == 0
    assert _Client.status_calls == ["old-inst"]
    assert _Client.renewed == []
    assert _Client.create_calls[0][0]["endpoint_name"] == "lium-b200-1"
    assert ("deactivate", "lium-b200-1", "old-inst") in _Client.events
    ep = endpoints.endpoints["lium-b200-1"]
    assert ep.active is True
    assert ep.autoscale_instance_id == "inst-lium-b200-1"
    assert ep.deployment_id is None
    assert ep.base_url is None
    state = await kv.get(STATE_KEY)
    assert state["last_endpoint_health_checked_count"] == 1
    assert state["last_endpoint_health_reclaimed_count"] == 1
    assert "endpoint_health_checked_at" not in state


@pytest.mark.asyncio
async def test_endpoint_health_check_respects_five_minute_interval():
    kv = InMemoryConfigStore()
    await kv.set(STATE_KEY, {"endpoint_health_checked_at": {"lium-b200-1": 900}})
    endpoint = Endpoint(
        name="lium-b200-1",
        kind="ssh",
        active=True,
        autoscale_managed=True,
        autoscale_provider="lium",
        autoscale_instance_id="inst-lium-b200-1",
        autoscale_lease_expires_at=5000,
    )
    endpoints = _Endpoints([endpoint])
    autoscaler = _autoscaler(
        _Queue(pending=1),
        endpoints,
        kv,
        now=1000,
    )

    result = await autoscaler.tick(
        _config(
            threshold=1,
            idle_seconds=3600,
            lease_renew_margin_seconds=60,
            endpoint_health_check_interval_seconds=300,
            status_path="/instances/{instance_id}",
        )
    )

    assert result.endpoint_health_checked_count == 0
    assert _Client.status_calls == []
    state = await kv.get(STATE_KEY)
    assert state["endpoint_health_checked_at"] == {"lium-b200-1": 900}


@pytest.mark.asyncio
async def test_does_not_renew_idle_endpoint_even_if_lease_is_near_expiry():
    _Client.renewed = []
    kv = InMemoryConfigStore()
    await kv.set(STATE_KEY, {"last_busy_at": 1000})
    endpoint = Endpoint(
        name="lium-b200-1",
        kind="ssh",
        active=True,
        autoscale_managed=True,
        autoscale_provider="lium",
        autoscale_instance_id="inst-lium-b200-1",
        autoscale_lease_expires_at=1050,
    )
    endpoints = _Endpoints([endpoint])
    autoscaler = _autoscaler(
        _Queue(pending=0),
        endpoints,
        kv,
        now=1000,
    )

    result = await autoscaler.tick(
        _config(
            idle_seconds=3600,
            lease_renew_margin_seconds=60,
        )
    )

    assert result.action == "none"
    assert result.idle is True
    assert result.lease_renewed_count == 0
    assert _Client.renewed == []
    assert endpoints.endpoints["lium-b200-1"].autoscale_lease_expires_at == 1050


@pytest.mark.asyncio
async def test_renews_legacy_endpoint_using_configured_lease_duration():
    _Client.renewed = []
    _Client.renew_expires_at = 0
    kv = InMemoryConfigStore()
    endpoint = Endpoint(
        name="lium-b200-1",
        kind="ssh",
        active=True,
        autoscale_managed=True,
        autoscale_provider="lium",
        autoscale_instance_id="inst-lium-b200-1",
        autoscale_created_at=100,
        autoscale_updated_at=100,
    )
    endpoints = _Endpoints([endpoint])
    autoscaler = _autoscaler(
        _Queue(in_progress=1),
        endpoints,
        kv,
        now=1050,
    )

    result = await autoscaler.tick(
        _config(
            idle_seconds=3600,
            lease_duration_seconds=1000,
            lease_renew_margin_seconds=60,
        )
    )

    assert result.action == "renew-lease:1"
    assert _Client.renewed == ["inst-lium-b200-1"]
    ep = endpoints.endpoints["lium-b200-1"]
    assert ep.autoscale_updated_at == 1050
    assert ep.autoscale_lease_expires_at == 2050


@pytest.mark.asyncio
async def test_add_endpoint_creates_configured_slot_without_old_endpoint():
    kv = InMemoryConfigStore()
    endpoints = _Endpoints()
    autoscaler = _autoscaler(_Queue(), endpoints, kv, now=2000)

    result = await autoscaler.add_endpoint(
        _config(),
        slot_name="lium-b200-2",
    )

    assert result == EndpointAddResult(
        endpoint_name="lium-b200-2",
        slot_name="lium-b200-2",
        instance_id="inst-lium-b200-2",
        created=True,
    )
    assert _Client.events == [
        ("create", "lium-b200-2"),
        ("activate", "lium-b200-2"),
    ]
    assert endpoints.endpoints["lium-b200-2"].active is True


@pytest.mark.asyncio
async def test_replace_same_endpoint_drains_old_before_delete_then_creates_new():
    kv = InMemoryConfigStore()
    state = StateStore(kv)
    await state.set_battle(
        BattleRecord(
            challenger=MinerSnapshot(
                uid=42,
                hotkey="hk42",
                revision="rev42",
                model="org/model42",
            ),
            deployment_id="ssh:lium-b200-1:affine-sglang-current",
            base_url="http://old.example.com:10001/v1",
            started_at_block=123,
            deployments=[
                DeploymentRecord(
                    endpoint_name="lium-b200-1",
                    deployment_id="ssh:lium-b200-1:affine-sglang-current",
                    base_url="http://old.example.com:10001/v1",
                )
            ],
        )
    )
    old = Endpoint(
        name="lium-b200-1",
        kind="ssh",
        active=True,
        assigned_uid=42,
        assigned_hotkey="hk42",
        assigned_model="org/model42",
        assigned_revision="rev42",
        deployment_id="ssh:lium-b200-1:affine-sglang-current",
        base_url="http://old.example.com:10001/v1",
        autoscale_managed=True,
        autoscale_provider="lium",
        autoscale_instance_id="old-inst",
        autoscale_purpose="eval",
    )
    endpoints = _Endpoints([old])
    queue = _Queue()
    autoscaler = _autoscaler(queue, endpoints, kv, now=2000)

    result = await autoscaler.replace_endpoint(
        _config(),
        old_endpoint_name="lium-b200-1",
    )

    assert result.same_endpoint is True
    assert result.old_deleted is True
    assert result.new_created is True
    assert result.old_instance_id == "old-inst"
    assert result.new_instance_id == "inst-lium-b200-1"
    assert _Client.events == [
        ("drain", "lium-b200-1", "old-inst"),
        ("delete", "old-inst"),
        ("deactivate", "lium-b200-1", "old-inst"),
        ("create", "lium-b200-1"),
        ("activate", "lium-b200-1"),
    ]
    ep = endpoints.endpoints["lium-b200-1"]
    assert ep.active is True
    assert ep.autoscale_instance_id == "inst-lium-b200-1"
    assert ep.assigned_uid is None
    assert await state.get_battle() is None
    assert queue.released == [(42, "hk42", "rev42")]
    assert MANUAL_REPLACEMENT_STATE_KEY not in await kv.get(STATE_KEY, {})
    assert await kv.get(MANUAL_REPLACEMENT_LOCK_KEY) is None


@pytest.mark.asyncio
async def test_replace_different_slot_creates_new_before_deleting_old():
    kv = InMemoryConfigStore()
    old = Endpoint(
        name="lium-b200-1",
        kind="ssh",
        active=True,
        autoscale_managed=True,
        autoscale_provider="lium",
        autoscale_instance_id="old-inst",
        autoscale_purpose="eval",
    )
    endpoints = _Endpoints([old])
    autoscaler = _autoscaler(_Queue(), endpoints, kv, now=2000)

    result = await autoscaler.replace_endpoint(
        _config(),
        old_endpoint_name="lium-b200-1",
        new_slot_name="lium-b200-2",
    )

    assert result.same_endpoint is False
    assert result.old_deleted is True
    assert result.new_created is True
    assert result.new_endpoint_name == "lium-b200-2"
    assert _Client.events == [
        ("create", "lium-b200-2"),
        ("activate", "lium-b200-2"),
        ("drain", "lium-b200-1", "old-inst"),
        ("delete", "old-inst"),
        ("deactivate", "lium-b200-1", "old-inst"),
    ]
    assert endpoints.endpoints["lium-b200-1"].active is False
    assert endpoints.endpoints["lium-b200-2"].active is True
    assert endpoints.endpoints["lium-b200-2"].autoscale_instance_id == (
        "inst-lium-b200-2"
    )


@pytest.mark.asyncio
async def test_replace_different_slot_rejects_active_target():
    kv = InMemoryConfigStore()
    old = Endpoint(
        name="lium-b200-1",
        kind="ssh",
        active=True,
        autoscale_managed=True,
        autoscale_provider="lium",
        autoscale_instance_id="old-inst",
    )
    target = Endpoint(
        name="lium-b200-2",
        kind="ssh",
        active=True,
        autoscale_managed=True,
        autoscale_provider="lium",
        autoscale_instance_id="target-inst",
    )
    autoscaler = _autoscaler(
        _Queue(),
        _Endpoints([old, target]),
        kv,
        now=2000,
    )

    with pytest.raises(ValueError, match="already active"):
        await autoscaler.replace_endpoint(
            _config(),
            old_endpoint_name="lium-b200-1",
            new_slot_name="lium-b200-2",
        )

    assert _Client.events == []


@pytest.mark.asyncio
async def test_replace_different_slot_leaves_old_when_create_fails():
    _Client.create_returns_none_for_names = {"lium-b200-2"}
    kv = InMemoryConfigStore()
    old = Endpoint(
        name="lium-b200-1",
        kind="ssh",
        active=True,
        autoscale_managed=True,
        autoscale_provider="lium",
        autoscale_instance_id="old-inst",
        autoscale_purpose="eval",
    )
    endpoints = _Endpoints([old])
    autoscaler = _autoscaler(_Queue(), endpoints, kv, now=2000)

    with pytest.raises(RuntimeError, match="old endpoint left untouched"):
        await autoscaler.replace_endpoint(
            _config(),
            old_endpoint_name="lium-b200-1",
            new_slot_name="lium-b200-2",
        )

    assert _Client.events == [("create", "lium-b200-2")]
    assert _Client.deleted == []
    assert endpoints.endpoints["lium-b200-1"].active is True
    assert endpoints.endpoints["lium-b200-1"].autoscale_instance_id == "old-inst"


@pytest.mark.asyncio
async def test_replace_same_endpoint_keeps_instance_id_when_delete_fails():
    _Client.delete_returns_false_for_ids = {"old-inst"}
    kv = InMemoryConfigStore()
    old = Endpoint(
        name="lium-b200-1",
        kind="ssh",
        active=True,
        autoscale_managed=True,
        autoscale_provider="lium",
        autoscale_instance_id="old-inst",
        autoscale_purpose="eval",
    )
    endpoints = _Endpoints([old])
    autoscaler = _autoscaler(_Queue(), endpoints, kv, now=2000)

    with pytest.raises(RuntimeError, match="stopped before creating"):
        await autoscaler.replace_endpoint(
            _config(),
            old_endpoint_name="lium-b200-1",
        )

    assert _Client.events == [
        ("drain", "lium-b200-1", "old-inst"),
        ("delete", "old-inst"),
        ("activate", "lium-b200-1"),
    ]
    ep = endpoints.endpoints["lium-b200-1"]
    assert ep.active is True
    assert ep.autoscale_instance_id == "old-inst"
    assert await kv.get(MANUAL_REPLACEMENT_LOCK_KEY) is None


@pytest.mark.asyncio
async def test_replace_same_endpoint_raises_when_new_create_fails():
    _Client.create_returns_none_for_names = {"lium-b200-1"}
    kv = InMemoryConfigStore()
    old = Endpoint(
        name="lium-b200-1",
        kind="ssh",
        active=True,
        autoscale_managed=True,
        autoscale_provider="lium",
        autoscale_instance_id="old-inst",
        autoscale_purpose="eval",
    )
    endpoints = _Endpoints([old])
    autoscaler = _autoscaler(_Queue(), endpoints, kv, now=2000)

    with pytest.raises(RuntimeError, match="was already deleted"):
        await autoscaler.replace_endpoint(
            _config(),
            old_endpoint_name="lium-b200-1",
        )

    assert _Client.events == [
        ("drain", "lium-b200-1", "old-inst"),
        ("delete", "old-inst"),
        ("deactivate", "lium-b200-1", "old-inst"),
        ("create", "lium-b200-1"),
    ]
    ep = endpoints.endpoints["lium-b200-1"]
    assert ep.active is False
    assert ep.autoscale_instance_id is None


@pytest.mark.asyncio
async def test_add_endpoint_rejects_active_slot():
    kv = InMemoryConfigStore()
    endpoint = Endpoint(
        name="lium-b200-1",
        kind="ssh",
        active=True,
        autoscale_managed=True,
        autoscale_provider="lium",
        autoscale_instance_id="old-inst",
    )
    autoscaler = _autoscaler(_Queue(), _Endpoints([endpoint]), kv, now=2000)

    with pytest.raises(ValueError, match="already active"):
        await autoscaler.add_endpoint(
            _config(),
            slot_name="lium-b200-1",
        )
