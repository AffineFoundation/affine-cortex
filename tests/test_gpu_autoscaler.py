from __future__ import annotations

import pytest

from affine.core.providers.instance_api_client import (
    InstanceAPIConfig,
    InstanceHandle,
)
from affine.database.dao.inference_endpoints import Endpoint
from affine.src.scheduler.gpu_autoscaler import (
    GPUAutoscaler,
    GPUAutoscalerConfig,
    ManagedEndpointSlot,
    STATE_KEY,
)
from affine.src.scorer.window_state import (
    ChampionRecord,
    DeploymentRecord,
    InMemoryConfigStore,
    StateStore,
)


class _Queue:
    def __init__(self, *, pending=0, in_progress=0):
        self.pending = pending
        self.in_progress = in_progress

    async def peek_next(self, n, *, champion_uid, exclude_uids=None):
        return [object() for _ in range(min(self.pending, n))]

    async def list_in_progress(self):
        return [object() for _ in range(self.in_progress)]


class _Endpoints:
    def __init__(self, endpoints=None):
        self.endpoints = {ep.name: ep for ep in (endpoints or [])}
        self.cleared = []

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
    renewed = []
    create_lease_expires_at = 1500
    renew_expires_at = 0

    def __init__(self, config):
        self.config = config

    async def create(self, *, variables=None, payload_overrides=None):
        name = variables["endpoint_name"]
        return InstanceHandle(
            provider=self.config.provider,
            instance_id=f"inst-{name}",
            ssh_url=f"ssh://root@{name}.example.com:22",
            public_inference_url=f"http://{name}.example.com:10001/v1",
            lease_expires_at=self.create_lease_expires_at,
        )

    async def delete(self, instance_id):
        self.deleted.append(instance_id)
        return True

    async def renew(self, instance_id):
        self.renewed.append(instance_id)
        return InstanceHandle(
            provider=self.config.provider,
            instance_id=instance_id,
            lease_expires_at=self.renew_expires_at,
        )


@pytest.fixture(autouse=True)
def _reset_client_state():
    _Client.deleted = []
    _Client.renewed = []
    _Client.create_lease_expires_at = 1500
    _Client.renew_expires_at = 0


def _config(
    *,
    enabled=True,
    threshold=1,
    idle_seconds=0,
    max_gpu_down_wait_seconds=0,
    lease_duration_seconds=0,
    lease_renew_margin_seconds=3600,
    lease_renew_cooldown_seconds=300,
    renew_path="/instances/{instance_id}/renew",
):
    return GPUAutoscalerConfig(
        enabled=enabled,
        pending_threshold_per_instance=threshold,
        max_gpu_down_wait_seconds=max_gpu_down_wait_seconds,
        idle_seconds=idle_seconds,
        lease_duration_seconds=lease_duration_seconds,
        lease_renew_margin_seconds=lease_renew_margin_seconds,
        lease_renew_cooldown_seconds=lease_renew_cooldown_seconds,
        min_instances=0,
        max_instances=2,
        providers={
            "lium": InstanceAPIConfig(
                provider="lium",
                api_url="https://lium.example.com",
                create_path="/instances",
                delete_path="/instances/{instance_id}",
                renew_path=renew_path,
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


def _autoscaler(queue, endpoints, kv, *, now=1000, samples=None):
    return GPUAutoscaler(
        queue=queue,
        endpoints_dao=endpoints,
        state_store=StateStore(kv),
        kv_store=kv,
        samples_adapter=samples or _Samples(count=0),
        client_factory=_Client,
        now_fn=lambda: now,
    )


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
    assert ep.autoscale_lease_expires_at == 1500
    assert ep.ssh_key_path == "/root/.ssh/affine_validator_server"
    assert ep.ssh_url == "ssh://root@lium-b200-1.example.com:22"


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
    assert endpoints.cleared == ["lium-b200-1"]
    ep = endpoints.endpoints["lium-b200-1"]
    assert ep.active is False
    assert ep.ssh_url is None
    assert ep.autoscale_instance_id is None
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
