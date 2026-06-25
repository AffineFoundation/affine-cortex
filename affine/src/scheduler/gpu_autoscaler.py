"""GPU endpoint autoscaler.

This service watches the same challenger queue the scheduler uses. When the
queue grows past a configured threshold it creates provider instances and
activates SSH endpoints; when the system has been idle for long enough it
destroys autoscaled instances and deactivates their endpoint rows.

The provider API shape is configuration-driven because Lium and Targon expose
different machine lifecycle APIs. Affine normalizes only the fields the
scheduler needs: SSH URL, public inference URL, and instance id.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import signal
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Mapping, Optional

import click

from affine.core.providers.instance_api_client import (
    InstanceAPIClient,
    InstanceAPIConfig,
    InstanceHandle,
)
from affine.core.setup import logger, setup_logging
from affine.database import close_client, init_client
from affine.database.dao.inference_endpoints import Endpoint, InferenceEndpointsDAO
from affine.database.dao.miners import MinersDAO
from affine.database.dao.sample_results import SampleResultsDAO
from affine.database.dao.system_config import SystemConfigDAO
from affine.src.scorer.challenger_queue import ChallengerQueue
from affine.src.scorer.dao_adapters import MinersQueueAdapter, SampleResultsAdapter
from affine.src.scorer.sampling_thresholds import champion_completion_threshold
from affine.src.scorer.window_state import StateStore, SystemConfigKVAdapter


CONFIG_KEY = "gpu_autoscaler"
STATE_KEY = "gpu_autoscaler_state"
DEFAULT_POLL_INTERVAL_SEC = 60
DEFAULT_IDLE_SECONDS = 30 * 60
DEFAULT_PENDING_THRESHOLD = 5
DEFAULT_MAX_GPU_DOWN_WAIT_SECONDS = 12 * 60 * 60
DEFAULT_LEASE_RENEW_MARGIN_SECONDS = 60 * 60
DEFAULT_LEASE_RENEW_COOLDOWN_SECONDS = 5 * 60
MAX_PENDING_PEEK = 10_000


_ENDPOINT_OVERRIDE_FIELDS = {
    "ssh_key_path",
    "public_inference_url",
    "ssh_url",
    "sglang_port",
    "sglang_dp",
    "sglang_image",
    "sglang_cache_dir",
    "sglang_context_len",
    "sglang_mem_fraction",
    "sglang_chunked_prefill",
    "sglang_tool_call_parser",
    "sglang_docker_args",
    "ready_timeout_sec",
    "poll_interval_sec",
    "notes",
}


@dataclass(frozen=True)
class ManagedEndpointSlot:
    name: str
    provider: str
    role: str = "scoring"
    endpoint: Dict[str, Any] = field(default_factory=dict)
    create_payload: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ManagedEndpointSlot":
        endpoint = dict(payload.get("endpoint") or {})
        for field in _ENDPOINT_OVERRIDE_FIELDS:
            if field in payload and field not in endpoint:
                endpoint[field] = payload[field]
        return cls(
            name=str(payload.get("name") or ""),
            provider=str(payload.get("provider") or "").lower(),
            role=str(payload.get("role") or "scoring"),
            endpoint=endpoint,
            create_payload=(
                dict(payload.get("create_payload"))
                if isinstance(payload.get("create_payload"), Mapping)
                else {}
            ),
        )


@dataclass(frozen=True)
class GPUAutoscalerConfig:
    enabled: bool = False
    poll_interval_sec: int = DEFAULT_POLL_INTERVAL_SEC
    pending_threshold_per_instance: int = DEFAULT_PENDING_THRESHOLD
    max_gpu_down_wait_seconds: int = DEFAULT_MAX_GPU_DOWN_WAIT_SECONDS
    idle_seconds: int = DEFAULT_IDLE_SECONDS
    lease_duration_seconds: int = 0
    lease_renew_margin_seconds: int = DEFAULT_LEASE_RENEW_MARGIN_SECONDS
    lease_renew_cooldown_seconds: int = DEFAULT_LEASE_RENEW_COOLDOWN_SECONDS
    min_instances: int = 0
    max_instances: int = 1
    dry_run: bool = False
    providers: Dict[str, InstanceAPIConfig] = field(default_factory=dict)
    slots: List[ManagedEndpointSlot] = field(default_factory=list)

    @classmethod
    def from_mapping(
        cls,
        payload: Optional[Mapping[str, Any]],
        *,
        apply_env_overrides: bool = False,
    ) -> "GPUAutoscalerConfig":
        data = dict(payload or {})
        raw_providers = data.get("providers") or {}
        providers = {
            str(name).lower(): InstanceAPIConfig.from_mapping(str(name).lower(), cfg)
            for name, cfg in raw_providers.items()
            if isinstance(cfg, Mapping)
        }
        slots = [
            ManagedEndpointSlot.from_mapping(item)
            for item in (data.get("endpoints") or data.get("slots") or [])
            if isinstance(item, Mapping)
        ]
        slots = [slot for slot in slots if slot.name and slot.provider]
        max_instances = int(data.get("max_instances") or len(slots) or 1)
        cfg = cls(
            enabled=_bool_value(data.get("enabled"), default=False),
            poll_interval_sec=int(
                data.get("poll_interval_sec") or DEFAULT_POLL_INTERVAL_SEC
            ),
            pending_threshold_per_instance=max(
                1,
                int(
                    data.get("pending_threshold_per_instance")
                    or data.get("pending_threshold")
                    or DEFAULT_PENDING_THRESHOLD
                ),
            ),
            max_gpu_down_wait_seconds=max(
                0,
                int(
                    data.get("max_gpu_down_wait_seconds")
                    or data.get("max_pending_wait_seconds")
                    or data.get("max_idle_wait_seconds")
                    or data.get("max_wait_before_sampling_seconds")
                    or DEFAULT_MAX_GPU_DOWN_WAIT_SECONDS
                ),
            ),
            idle_seconds=max(
                0,
                int(
                    data.get("idle_seconds")
                    or data.get("scale_down_idle_seconds")
                    or DEFAULT_IDLE_SECONDS
                ),
            ),
            lease_duration_seconds=max(
                0,
                _seconds_value(
                    data,
                    seconds_keys=("lease_duration_seconds",),
                    hours_keys=("lease_duration_hours",),
                    default=0,
                ),
            ),
            lease_renew_margin_seconds=max(
                0,
                _seconds_value(
                    data,
                    seconds_keys=("lease_renew_margin_seconds",),
                    minutes_keys=("lease_renew_margin_minutes",),
                    default=DEFAULT_LEASE_RENEW_MARGIN_SECONDS,
                ),
            ),
            lease_renew_cooldown_seconds=max(
                0,
                _seconds_value(
                    data,
                    seconds_keys=("lease_renew_cooldown_seconds",),
                    minutes_keys=("lease_renew_cooldown_minutes",),
                    default=DEFAULT_LEASE_RENEW_COOLDOWN_SECONDS,
                ),
            ),
            min_instances=max(0, int(data.get("min_instances") or 0)),
            max_instances=max(0, max_instances),
            dry_run=_bool_value(data.get("dry_run"), default=False),
            providers=providers,
            slots=slots,
        )
        cfg = replace(
            cfg,
            max_instances=max(0, min(cfg.max_instances, len(cfg.slots or []))),
        )
        if apply_env_overrides:
            return cfg.with_env_overrides()
        return cfg

    def with_env_overrides(self) -> "GPUAutoscalerConfig":
        enabled = _env_bool("AFFINE_GPU_AUTOSCALER_ENABLED", self.enabled)
        dry_run = _env_bool("AFFINE_GPU_AUTOSCALER_DRY_RUN", self.dry_run)
        poll = _env_int(
            "AFFINE_GPU_AUTOSCALER_POLL_INTERVAL_SEC",
            self.poll_interval_sec,
        )
        idle = _env_int(
            "AFFINE_GPU_AUTOSCALER_IDLE_SECONDS",
            self.idle_seconds,
        )
        threshold = _env_int(
            "AFFINE_GPU_AUTOSCALER_PENDING_THRESHOLD",
            self.pending_threshold_per_instance,
        )
        max_gpu_down_wait = _env_int(
            "AFFINE_GPU_AUTOSCALER_MAX_PENDING_WAIT_SECONDS",
            self.max_gpu_down_wait_seconds,
        )
        max_gpu_down_wait = _env_int(
            "AFFINE_GPU_AUTOSCALER_MAX_GPU_DOWN_WAIT_SECONDS",
            max_gpu_down_wait,
        )
        lease_duration = _env_int(
            "AFFINE_GPU_AUTOSCALER_LEASE_DURATION_SECONDS",
            self.lease_duration_seconds,
        )
        lease_duration_hours = _env_int(
            "AFFINE_GPU_AUTOSCALER_LEASE_DURATION_HOURS",
            0,
        )
        if lease_duration_hours > 0:
            lease_duration = lease_duration_hours * 60 * 60
        lease_margin = _env_int(
            "AFFINE_GPU_AUTOSCALER_LEASE_RENEW_MARGIN_SECONDS",
            self.lease_renew_margin_seconds,
        )
        lease_cooldown = _env_int(
            "AFFINE_GPU_AUTOSCALER_LEASE_RENEW_COOLDOWN_SECONDS",
            self.lease_renew_cooldown_seconds,
        )
        return replace(
            self,
            enabled=enabled,
            dry_run=dry_run,
            poll_interval_sec=max(1, poll),
            idle_seconds=max(0, idle),
            pending_threshold_per_instance=max(1, threshold),
            max_gpu_down_wait_seconds=max(0, max_gpu_down_wait),
            lease_duration_seconds=max(0, lease_duration),
            lease_renew_margin_seconds=max(0, lease_margin),
            lease_renew_cooldown_seconds=max(0, lease_cooldown),
            max_instances=max(0, min(self.max_instances, len(self.slots or []))),
        )

    def desired_instances(
        self,
        pending_count: int,
        *,
        gpu_down_for_sec: int = 0,
        force_start: bool = False,
    ) -> int:
        if pending_count <= 0:
            return self.min_instances
        if pending_count < self.pending_threshold_per_instance:
            if (
                force_start
                or (
                    self.max_gpu_down_wait_seconds > 0
                    and gpu_down_for_sec >= self.max_gpu_down_wait_seconds
                )
            ):
                return min(self.max_instances, max(self.min_instances, 1))
            return self.min_instances
        desired = math.ceil(pending_count / self.pending_threshold_per_instance)
        desired = max(self.min_instances, desired)
        return min(self.max_instances, desired)


@dataclass(frozen=True)
class AutoscalerSnapshot:
    pending_count: int
    in_progress_count: int
    battle_active: bool
    predeployed_count: int
    champion_uid: Optional[int]
    champion_samples_complete: bool
    active_capacity_count: int
    active_managed: List[Endpoint]
    active_endpoint_names: set
    idle: bool


@dataclass(frozen=True)
class AutoscalerTickResult:
    action: str
    pending_count: int = 0
    active_capacity_count: int = 0
    desired_instances: int = 0
    idle: bool = False
    idle_for_sec: int = 0
    gpu_down_for_sec: int = 0
    lease_renewed_count: int = 0


class GPUAutoscaler:
    def __init__(
        self,
        *,
        queue: ChallengerQueue,
        endpoints_dao: InferenceEndpointsDAO,
        state_store: StateStore,
        kv_store,
        samples_adapter: SampleResultsAdapter,
        client_factory: Optional[Callable[[InstanceAPIConfig], InstanceAPIClient]] = None,
        now_fn: Optional[Callable[[], float]] = None,
    ):
        self._queue = queue
        self._endpoints = endpoints_dao
        self._state = state_store
        self._kv = kv_store
        self._samples = samples_adapter
        self._client_factory = client_factory or InstanceAPIClient
        self._now = now_fn or time.time
        self._force_start_after_restart = True

    async def tick(self, config: GPUAutoscalerConfig) -> AutoscalerTickResult:
        if not config.enabled:
            return AutoscalerTickResult(action="disabled")
        if not config.slots:
            logger.warning("gpu-autoscaler: enabled but no managed endpoints configured")
            return AutoscalerTickResult(action="no-slots")

        snapshot = await self._snapshot()
        state = await self._load_state()
        now = int(self._now())
        if snapshot.idle:
            state.setdefault("last_busy_at", now)
        else:
            state["last_busy_at"] = now

        gpu_down_for_sec = self._update_gpu_down_state(
            state,
            snapshot,
            now,
        )
        force_start = self._should_force_start_after_restart(snapshot)
        desired = config.desired_instances(
            snapshot.pending_count,
            gpu_down_for_sec=gpu_down_for_sec,
            force_start=force_start,
        )
        idle_for = now - int(state.get("last_busy_at", now) or now)
        action = "none"
        lease_renewed = await self._renew_busy_expiring_leases(
            config,
            snapshot,
            state,
            now,
        )
        if lease_renewed:
            action = f"renew-lease:{lease_renewed}"

        if desired > snapshot.active_capacity_count:
            created = await self._scale_up(
                config,
                count=desired - snapshot.active_capacity_count,
            )
            if created:
                action = f"scale-up:{created}"
                state["last_scale_at"] = now
                state.pop("gpu_down_at", None)
                self._force_start_after_restart = False
        elif (
            desired < snapshot.active_capacity_count
            and snapshot.idle
            and idle_for >= config.idle_seconds
        ):
            destroyed = await self._scale_down(
                config,
                snapshot,
                count=snapshot.active_capacity_count - desired,
            )
            if destroyed:
                action = f"scale-down:{destroyed}"
                state["last_scale_at"] = now
                if snapshot.active_capacity_count - destroyed <= 0:
                    state["gpu_down_at"] = now

        state.update(
            {
                "last_tick_at": now,
                "last_pending_count": snapshot.pending_count,
                "last_active_capacity_count": snapshot.active_capacity_count,
                "last_desired_instances": desired,
                "last_idle": snapshot.idle,
                "last_idle_for_sec": idle_for,
                "last_gpu_down_for_sec": gpu_down_for_sec,
                "last_force_start_after_restart": force_start,
                "last_lease_renewed_count": lease_renewed,
            }
        )
        await self._kv.set(STATE_KEY, state)
        logger.info(
            "gpu-autoscaler: action=%s pending=%s active_capacity=%s desired=%s "
            "idle=%s idle_for=%ss gpu_down_for=%ss force_start=%s",
            action,
            snapshot.pending_count,
            snapshot.active_capacity_count,
            desired,
            snapshot.idle,
            idle_for,
            gpu_down_for_sec,
            force_start,
        )
        return AutoscalerTickResult(
            action=action,
            pending_count=snapshot.pending_count,
            active_capacity_count=snapshot.active_capacity_count,
            desired_instances=desired,
            idle=snapshot.idle,
            idle_for_sec=idle_for,
            gpu_down_for_sec=gpu_down_for_sec,
            lease_renewed_count=lease_renewed,
        )

    def _should_force_start_after_restart(
        self,
        snapshot: AutoscalerSnapshot,
    ) -> bool:
        if not self._force_start_after_restart:
            return False
        if snapshot.active_capacity_count > 0:
            self._force_start_after_restart = False
            return False
        if snapshot.pending_count <= 0:
            self._force_start_after_restart = False
            return False
        should_force = (
            snapshot.in_progress_count == 0
            and not snapshot.battle_active
            and snapshot.predeployed_count == 0
        )
        if should_force:
            self._force_start_after_restart = False
        return should_force

    def _update_gpu_down_state(
        self,
        state: Dict[str, Any],
        snapshot: AutoscalerSnapshot,
        now: int,
    ) -> int:
        if snapshot.active_capacity_count > 0:
            state.pop("gpu_down_at", None)
            state.pop("pending_wait_started_at", None)
            return 0
        started_at = int(
            state.get("gpu_down_at")
            or state.get("pending_wait_started_at")
            or now
        )
        state["gpu_down_at"] = started_at
        state.pop("pending_wait_started_at", None)
        return max(0, now - started_at)

    async def _snapshot(self) -> AutoscalerSnapshot:
        champion = await self._state.get_champion()
        champion_uid = champion.uid if champion else None
        battle = await self._state.get_battle()
        predeployed = await self._state.get_predeployed_challengers()
        exclude_uids = {p.challenger.uid for p in predeployed}
        if battle:
            exclude_uids.add(battle.challenger.uid)

        pending = await self._queue.peek_next(
            MAX_PENDING_PEEK,
            champion_uid=champion_uid,
            exclude_uids=exclude_uids,
        )
        in_progress = await self._queue.list_in_progress()
        all_endpoints = await self._endpoints.list_all()
        active_scoring = [
            ep for ep in all_endpoints
            if ep.active
            and ep.kind == "ssh"
            and (ep.role or "scoring") == "scoring"
        ]
        active_managed = [
            ep for ep in active_scoring
            if bool(getattr(ep, "autoscale_managed", False))
        ]
        busy_assignments = [
            ep for ep in active_scoring
            if ep.assigned_uid is not None and ep.assigned_uid != champion_uid
        ]
        champion_complete = await self._champion_samples_complete(champion)
        idle = (
            len(pending) == 0
            and len(in_progress) == 0
            and battle is None
            and not predeployed
            and not busy_assignments
            and champion_complete
        )
        return AutoscalerSnapshot(
            pending_count=len(pending),
            in_progress_count=len(in_progress),
            battle_active=battle is not None,
            predeployed_count=len(predeployed),
            champion_uid=champion_uid,
            champion_samples_complete=champion_complete,
            active_capacity_count=len(active_scoring),
            active_managed=active_managed,
            active_endpoint_names={ep.name for ep in active_scoring},
            idle=idle,
        )

    async def _champion_samples_complete(self, champion) -> bool:
        if not champion or not champion.hotkey or not champion.revision:
            return True
        task_state = await self._state.get_task_state()
        if not task_state or not task_state.task_ids:
            return True
        environments = await self._state.get_environments()
        refresh_block = int(task_state.refreshed_at_block or 0)
        for env, cfg in environments.items():
            if not cfg.enabled_for_scoring:
                continue
            task_ids = list(task_state.task_ids.get(env) or [])
            if not task_ids:
                continue
            target = champion_completion_threshold(cfg.sampling_count)
            if target <= 0:
                continue
            count = await self._samples.count_samples_for_tasks(
                champion.hotkey,
                champion.revision,
                env,
                task_ids,
                refresh_block,
            )
            if count < target:
                return False
        return True

    async def _scale_up(self, config: GPUAutoscalerConfig, *, count: int) -> int:
        created = 0
        active_names = {
            ep.name for ep in await self._endpoints.list_active(kind="ssh")
        }
        for slot in config.slots or []:
            if created >= count:
                break
            if slot.name in active_names:
                continue
            ok = await self._scale_up_slot(config, slot)
            if ok:
                created += 1
                active_names.add(slot.name)
        return created

    async def _scale_up_slot(
        self, config: GPUAutoscalerConfig, slot: ManagedEndpointSlot
    ) -> bool:
        provider_config = (config.providers or {}).get(slot.provider)
        if provider_config is None:
            logger.warning(
                "gpu-autoscaler: endpoint=%s provider=%s has no API config",
                slot.name,
                slot.provider,
            )
            return False
        if config.dry_run:
            logger.info(
                "gpu-autoscaler: dry-run would create endpoint=%s provider=%s",
                slot.name,
                slot.provider,
            )
            return True

        client = self._client_factory(provider_config)
        variables = {
            "endpoint_name": slot.name,
            "provider": slot.provider,
        }
        handle = await client.create(
            variables=variables,
            payload_overrides=slot.create_payload,
        )
        if handle is None:
            return False
        endpoint = await self._endpoint_for_handle(slot, handle)
        if not endpoint.ssh_url:
            logger.warning(
                "gpu-autoscaler: provider=%s instance=%s did not return ssh_url",
                slot.provider,
                handle.instance_id,
            )
            await client.delete(handle.instance_id)
            return False
        try:
            await self._endpoints.activate_autoscaled_endpoint(
                endpoint,
                updated_by="gpu-autoscaler",
            )
        except Exception as e:
            logger.error(
                "gpu-autoscaler: endpoint activation failed after create "
                "endpoint=%s provider=%s instance=%s; deleting instance: "
                "%s: %s",
                endpoint.name,
                slot.provider,
                handle.instance_id,
                type(e).__name__,
                e,
            )
            deleted = await client.delete(handle.instance_id)
            if not deleted:
                logger.error(
                    "gpu-autoscaler: failed to delete instance=%s after "
                    "activation failure; manual cleanup may be required",
                    handle.instance_id,
                )
            return False
        logger.info(
            "gpu-autoscaler: activated endpoint=%s provider=%s instance=%s",
            endpoint.name,
            slot.provider,
            handle.instance_id,
        )
        return True

    async def _endpoint_for_handle(
        self, slot: ManagedEndpointSlot, handle: InstanceHandle
    ) -> Endpoint:
        existing = await self._endpoints.get(slot.name)
        endpoint = existing or Endpoint(name=slot.name, kind="ssh")
        endpoint = replace(
            endpoint,
            kind="ssh",
            active=True,
            role=slot.role or "scoring",
            assigned_uid=None,
            assigned_hotkey=None,
            assigned_model=None,
            assigned_revision=None,
            deployment_id=None,
            base_url=None,
            assignment_role=None,
            assigned_at=0,
            autoscale_managed=True,
            autoscale_provider=slot.provider,
            autoscale_instance_id=handle.instance_id,
            autoscale_created_at=int(self._now()),
            autoscale_updated_at=int(self._now()),
            autoscale_lease_expires_at=int(handle.lease_expires_at or 0),
        )
        for field, value in (slot.endpoint or {}).items():
            if field in _ENDPOINT_OVERRIDE_FIELDS and value is not None:
                setattr(endpoint, field, value)
        endpoint.ssh_url = handle.ssh_url or endpoint.ssh_url
        endpoint.public_inference_url = (
            handle.public_inference_url or endpoint.public_inference_url
        )
        if not endpoint.notes:
            endpoint.notes = f"autoscaled via {slot.provider}"
        return endpoint

    async def _renew_busy_expiring_leases(
        self,
        config: GPUAutoscalerConfig,
        snapshot: AutoscalerSnapshot,
        state: Dict[str, Any],
        now: int,
    ) -> int:
        if snapshot.idle or not snapshot.active_managed:
            return 0
        if config.lease_renew_margin_seconds <= 0:
            return 0

        attempts = dict(state.get("lease_renew_attempted_at") or {})
        changed_attempts = False
        renewed = 0
        for endpoint in snapshot.active_managed:
            expires_at = self._lease_expires_at(endpoint, config)
            if expires_at <= 0:
                continue
            remaining = expires_at - now
            if remaining > config.lease_renew_margin_seconds:
                continue
            last_attempt = int(attempts.get(endpoint.name) or 0)
            if (
                last_attempt
                and config.lease_renew_cooldown_seconds > 0
                and now - last_attempt < config.lease_renew_cooldown_seconds
            ):
                continue
            attempts[endpoint.name] = now
            changed_attempts = True
            ok = await self._renew_endpoint_lease(config, endpoint, now)
            if ok:
                renewed += 1
                attempts.pop(endpoint.name, None)
                changed_attempts = True

        if changed_attempts:
            if attempts:
                state["lease_renew_attempted_at"] = attempts
            else:
                state.pop("lease_renew_attempted_at", None)
        return renewed

    def _lease_expires_at(
        self,
        endpoint: Endpoint,
        config: GPUAutoscalerConfig,
    ) -> int:
        explicit = int(getattr(endpoint, "autoscale_lease_expires_at", 0) or 0)
        if explicit > 0:
            return explicit
        if config.lease_duration_seconds <= 0:
            return 0
        base = int(
            getattr(endpoint, "autoscale_updated_at", 0)
            or getattr(endpoint, "autoscale_created_at", 0)
            or 0
        )
        if base <= 0:
            return 0
        return base + config.lease_duration_seconds

    async def _renew_endpoint_lease(
        self,
        config: GPUAutoscalerConfig,
        endpoint: Endpoint,
        now: int,
    ) -> bool:
        provider = (endpoint.autoscale_provider or "").lower()
        provider_config = (config.providers or {}).get(provider)
        if provider_config is None:
            logger.warning(
                "gpu-autoscaler: endpoint=%s has no provider API config",
                endpoint.name,
            )
            return False
        instance_id = endpoint.autoscale_instance_id or ""
        if not instance_id:
            logger.warning(
                "gpu-autoscaler: endpoint=%s has no autoscale_instance_id",
                endpoint.name,
            )
            return False
        if not provider_config.renew_path:
            logger.warning(
                "gpu-autoscaler: endpoint=%s provider=%s has no renew_path",
                endpoint.name,
                provider,
            )
            return False
        if config.dry_run:
            logger.info(
                "gpu-autoscaler: dry-run would renew endpoint=%s instance=%s",
                endpoint.name,
                instance_id,
            )
            return True

        client = self._client_factory(provider_config)
        handle = await client.renew(instance_id)
        if handle is None:
            return False
        lease_expires_at = handle.lease_expires_at
        if not lease_expires_at and config.lease_duration_seconds > 0:
            lease_expires_at = now + config.lease_duration_seconds
        try:
            await self._endpoints.update_autoscale_lease(
                endpoint.name,
                instance_id=instance_id,
                lease_expires_at=lease_expires_at,
                updated_by="gpu-autoscaler",
            )
        except Exception as e:
            logger.warning(
                "gpu-autoscaler: failed to record renewed lease for "
                "endpoint=%s instance=%s: %s: %s",
                endpoint.name,
                instance_id,
                type(e).__name__,
                e,
            )
            return False
        logger.info(
            "gpu-autoscaler: renewed endpoint=%s provider=%s instance=%s "
            "lease_expires_at=%s",
            endpoint.name,
            provider,
            instance_id,
            lease_expires_at or "-",
        )
        return True

    async def _scale_down(
        self,
        config: GPUAutoscalerConfig,
        snapshot: AutoscalerSnapshot,
        *,
        count: int,
    ) -> int:
        destroyed = 0
        slot_order = {
            slot.name: idx for idx, slot in enumerate(config.slots or [])
        }
        candidates = sorted(
            snapshot.active_managed,
            key=lambda ep: slot_order.get(ep.name, 10_000),
            reverse=True,
        )
        for endpoint in candidates:
            if destroyed >= count:
                break
            ok = await self._scale_down_endpoint(config, endpoint)
            if ok:
                destroyed += 1
        return destroyed

    async def _scale_down_endpoint(
        self, config: GPUAutoscalerConfig, endpoint: Endpoint
    ) -> bool:
        provider = (endpoint.autoscale_provider or "").lower()
        provider_config = (config.providers or {}).get(provider)
        if provider_config is None:
            logger.warning(
                "gpu-autoscaler: endpoint=%s has no provider API config",
                endpoint.name,
            )
            return False
        instance_id = endpoint.autoscale_instance_id or ""
        if not instance_id:
            logger.warning(
                "gpu-autoscaler: endpoint=%s has no autoscale_instance_id",
                endpoint.name,
            )
            return False
        if config.dry_run:
            logger.info(
                "gpu-autoscaler: dry-run would delete endpoint=%s instance=%s",
                endpoint.name,
                instance_id,
            )
            return True

        client = self._client_factory(provider_config)
        deleted = await client.delete(instance_id)
        if not deleted:
            return False

        endpoint_for_refs = replace(endpoint)
        try:
            await self._endpoints.deactivate_autoscaled_endpoint(
                endpoint.name,
                instance_id=instance_id,
                updated_by="gpu-autoscaler",
            )
        except Exception as e:
            logger.warning(
                "gpu-autoscaler: deactivate endpoint=%s instance=%s failed: "
                "%s: %s",
                endpoint.name,
                instance_id,
                type(e).__name__,
                e,
            )
            return False
        await self._clear_deployment_refs(endpoint_for_refs)
        logger.info(
            "gpu-autoscaler: deactivated endpoint=%s provider=%s instance=%s",
            endpoint.name,
            provider,
            instance_id,
        )
        return True

    async def _clear_deployment_refs(self, endpoint: Endpoint) -> None:
        champion = await self._state.get_champion()
        if not champion:
            return
        deployment_id = endpoint.deployment_id or ""
        base_url = endpoint.base_url or endpoint.public_inference_url or ""

        def _matches_endpoint(dep) -> bool:
            return (
                dep.endpoint_name == endpoint.name
                or (deployment_id and dep.deployment_id == deployment_id)
                or (base_url and dep.base_url == base_url)
            )

        deployments = [
            dep for dep in (champion.deployments or [])
            if not _matches_endpoint(dep)
        ]
        changed = len(deployments) != len(champion.deployments or [])
        primary_removed = (
            (deployment_id and champion.deployment_id == deployment_id)
            or (base_url and champion.base_url == base_url)
        )
        if not changed and not primary_removed:
            return
        champion.deployments = deployments
        if deployments:
            champion.deployment_id = deployments[0].deployment_id
            champion.base_url = deployments[0].base_url
        else:
            champion.deployment_id = None
            champion.base_url = None
        await self._state.set_champion(champion)

    async def _load_state(self) -> Dict[str, Any]:
        raw = await self._kv.get(STATE_KEY, default={})
        return dict(raw) if isinstance(raw, Mapping) else {}


async def load_config(config_dao: SystemConfigDAO) -> GPUAutoscalerConfig:
    payload = await config_dao.get_param_value(CONFIG_KEY, default=None)
    if isinstance(payload, Mapping) and payload:
        return GPUAutoscalerConfig.from_mapping(payload)
    payload = {}
    env_payload = _load_env_json("AFFINE_GPU_AUTOSCALER_CONFIG_JSON")
    if env_payload:
        payload = dict(env_payload)
    return GPUAutoscalerConfig.from_mapping(payload, apply_env_overrides=True)


async def _run() -> None:
    await init_client()
    config_dao = SystemConfigDAO()
    kv = SystemConfigKVAdapter(config_dao, updated_by="gpu-autoscaler")
    autoscaler = GPUAutoscaler(
        queue=ChallengerQueue(MinersQueueAdapter(MinersDAO())),
        endpoints_dao=InferenceEndpointsDAO(),
        state_store=StateStore(kv),
        kv_store=kv,
        samples_adapter=SampleResultsAdapter(
            dao=SampleResultsDAO(),
            validator_hotkey="gpu-autoscaler",
        ),
    )

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except (NotImplementedError, RuntimeError):
            pass

    try:
        while not stop_event.is_set():
            config = await load_config(config_dao)
            try:
                await autoscaler.tick(config)
            except Exception as e:
                logger.error(
                    "gpu-autoscaler tick failed: %s: %s",
                    type(e).__name__,
                    e,
                    exc_info=True,
                )
            try:
                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=max(1, config.poll_interval_sec),
                )
            except asyncio.TimeoutError:
                pass
    finally:
        await close_client()


@click.command()
@click.option("-v", "--verbose", count=True, default=1)
def main(verbose: int) -> None:
    """Run the GPU endpoint autoscaler service."""
    setup_logging(verbosity=verbose, component="gpu-autoscaler")
    asyncio.run(_run())


def _bool_value(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _env_bool(name: str, default: bool) -> bool:
    if name not in os.environ:
        return default
    return _bool_value(os.getenv(name), default=default)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _seconds_value(
    data: Mapping[str, Any],
    *,
    seconds_keys: tuple[str, ...] = (),
    minutes_keys: tuple[str, ...] = (),
    hours_keys: tuple[str, ...] = (),
    default: int = 0,
) -> int:
    for key in seconds_keys:
        value = data.get(key)
        if value not in (None, ""):
            return int(value)
    for key in minutes_keys:
        value = data.get(key)
        if value not in (None, ""):
            return int(value) * 60
    for key in hours_keys:
        value = data.get(key)
        if value not in (None, ""):
            return int(value) * 60 * 60
    return default


def _load_env_json(name: str) -> Dict[str, Any]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning("gpu-autoscaler: invalid %s JSON: %s", name, e)
        return {}
    return payload if isinstance(payload, dict) else {}


if __name__ == "__main__":
    main()
