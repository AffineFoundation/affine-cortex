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
import contextlib
import json
import math
import os
import signal
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from asyncio.subprocess import Process
from typing import Any, Callable, Dict, List, Mapping, Optional
from urllib.parse import urlparse

import click

from affine.core.providers.instance_api_client import (
    InstanceAPIClient,
    InstanceAPIConfig,
    InstanceHandle,
    InstanceAPINotFoundError,
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
from affine.src.scheduler.health import tunnel_repair_request_key


CONFIG_KEY = "gpu_autoscaler"
STATE_KEY = "gpu_autoscaler_state"
# Keep the historical key names so queued requests survive upgrades. They now
# serialize every manual endpoint operation, not only replacements.
MANUAL_REPLACEMENT_LOCK_KEY = "gpu_autoscaler_manual_replacement"
MANUAL_REPLACEMENT_STATE_KEY = "manual_replacement"
MANUAL_REPLACEMENT_RESULT_KEY_PREFIX = "gpu_autoscaler_manual_replacement_result:"
MANUAL_REPLACEMENT_REQUEST_VERSION = 2
MANUAL_ENDPOINT_OPERATION_ADD = "add"
MANUAL_ENDPOINT_OPERATION_REMOVE = "remove"
MANUAL_ENDPOINT_OPERATION_REPLACE = "replace"
DEFAULT_POLL_INTERVAL_SEC = 60
DEFAULT_IDLE_SECONDS = 30 * 60
DEFAULT_PENDING_THRESHOLD = 5
DEFAULT_MAX_INSTANCES = 1
DEFAULT_MAX_GPU_DOWN_WAIT_SECONDS = 12 * 60 * 60
DEFAULT_LEASE_RENEW_MARGIN_SECONDS = 60 * 60
DEFAULT_LEASE_RENEW_COOLDOWN_SECONDS = 5 * 60
DEFAULT_ENDPOINT_HEALTH_CHECK_INTERVAL_SECONDS = 5 * 60
DEFAULT_MANUAL_REPLACEMENT_TTL_SECONDS = 60 * 60
DEFAULT_MANUAL_REPLACEMENT_RESULT_POLL_SECONDS = 1.0
MANUAL_REPLACEMENT_RESULT_VISIBILITY_GRACE_SECONDS = 10.0
MANUAL_REPLACEMENT_HEARTBEAT_SECONDS = 30.0
MAX_PENDING_PEEK = 10_000


_ENDPOINT_OVERRIDE_FIELDS = {
    "ssh_key_path",
    "public_inference_url",
    "ssh_url",
    "sglang_port",
    "sglang_dp",
    "sglang_load_balance_method",
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
    purpose: str = "eval"
    endpoint: Dict[str, Any] = field(default_factory=dict)
    create_payload: Dict[str, Any] = field(default_factory=dict)
    tunnel: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ManagedEndpointSlot":
        endpoint = dict(payload.get("endpoint") or {})
        for field_name in _ENDPOINT_OVERRIDE_FIELDS:
            if field_name in payload and field_name not in endpoint:
                endpoint[field_name] = payload[field_name]
        tunnel = (
            dict(payload.get("tunnel"))
            if isinstance(payload.get("tunnel"), Mapping)
            else {}
        )
        return cls(
            name=str(payload.get("name") or ""),
            provider=str(payload.get("provider") or "").lower(),
            role=str(payload.get("role") or "scoring"),
            purpose=_safe_token(payload.get("purpose") or "eval", default="eval"),
            endpoint=endpoint,
            create_payload=(
                dict(payload.get("create_payload"))
                if isinstance(payload.get("create_payload"), Mapping)
                else {}
            ),
            tunnel=tunnel,
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
    endpoint_health_check_interval_seconds: int = (
        DEFAULT_ENDPOINT_HEALTH_CHECK_INTERVAL_SECONDS
    )
    min_instances: int = 0
    max_instances: int = DEFAULT_MAX_INSTANCES
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
        max_instances = int(
            data.get("max_instances") or DEFAULT_MAX_INSTANCES
        )
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
            endpoint_health_check_interval_seconds=max(
                0,
                _seconds_value(
                    data,
                    seconds_keys=("endpoint_health_check_interval_seconds",),
                    minutes_keys=("endpoint_health_check_interval_minutes",),
                    default=DEFAULT_ENDPOINT_HEALTH_CHECK_INTERVAL_SECONDS,
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
        health_check_interval = _env_int(
            "AFFINE_GPU_AUTOSCALER_ENDPOINT_HEALTH_CHECK_INTERVAL_SECONDS",
            self.endpoint_health_check_interval_seconds,
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
            endpoint_health_check_interval_seconds=max(0, health_check_interval),
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
class TunnelSpec:
    endpoint_name: str
    instance_id: str
    ssh_url: str
    ssh_key_path: str
    public_url: str
    bind_host: str = "0.0.0.0"
    target_host: str = "127.0.0.1"
    target_port: int = 10001
    connect_timeout_sec: float = 10.0

    @property
    def local_port(self) -> int:
        parsed = urlparse(self.public_url or "")
        if not parsed.port:
            raise ValueError(
                f"public_inference_url for endpoint {self.endpoint_name!r} "
                "must include an explicit port when tunnel is enabled"
            )
        return int(parsed.port)


@dataclass
class _TunnelProcess:
    spec: TunnelSpec
    process: Process


class SSHTunnelManager:
    def __init__(self):
        self._tunnels: Dict[str, _TunnelProcess] = {}

    async def ensure(self, spec: TunnelSpec) -> None:
        current = self._tunnels.get(spec.endpoint_name)
        if (
            current is not None
            and current.spec == spec
            and current.process.returncode is None
        ):
            return

        await self.stop(spec.endpoint_name)
        if await _local_port_is_open("127.0.0.1", spec.local_port):
            raise RuntimeError(
                f"local tunnel port 127.0.0.1:{spec.local_port} is already "
                "owned by an unmanaged process"
            )
        user, host, port = _parse_ssh_url(spec.ssh_url)
        forward = (
            f"{spec.bind_host}:{spec.local_port}:"
            f"{spec.target_host}:{spec.target_port}"
        )
        cmd = [
            "ssh",
            "-N",
            "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3",
            "-o", "StrictHostKeyChecking=no",
        ]
        if spec.ssh_key_path:
            cmd.extend(["-i", spec.ssh_key_path])
        cmd.extend(["-p", str(port), "-L", forward, f"{user}@{host}"])
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            await _wait_for_tunnel_process(
                process,
                "127.0.0.1",
                spec.local_port,
                timeout_sec=spec.connect_timeout_sec,
            )
        except Exception:
            await _stop_process(process, timeout_sec=2.0)
            raise
        self._tunnels[spec.endpoint_name] = _TunnelProcess(
            spec=spec,
            process=process,
        )
        logger.info(
            "gpu-autoscaler: tunnel ready endpoint=%s instance=%s "
            "local_port=%s target=%s:%s",
            spec.endpoint_name,
            spec.instance_id,
            spec.local_port,
            spec.target_host,
            spec.target_port,
        )

    async def stop(self, endpoint_name: str) -> None:
        current = self._tunnels.pop(endpoint_name, None)
        if current is None:
            return
        await _stop_process(current.process, timeout_sec=5.0)
        logger.info(
            "gpu-autoscaler: tunnel stopped endpoint=%s instance=%s "
            "local_port=%s",
            endpoint_name,
            current.spec.instance_id,
            current.spec.local_port,
        )

    async def restart(self, spec: TunnelSpec) -> None:
        await self.stop(spec.endpoint_name)
        await self.ensure(spec)

    async def stop_all(self) -> None:
        for endpoint_name in list(self._tunnels):
            await self.stop(endpoint_name)


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
    lease_reclaimed_count: int = 0
    endpoint_health_checked_count: int = 0
    endpoint_health_reclaimed_count: int = 0


@dataclass(frozen=True)
class LeaseRenewResult:
    renewed_count: int = 0
    reclaimed_count: int = 0


@dataclass(frozen=True)
class EndpointHealthCheckResult:
    checked_count: int = 0
    reclaimed_count: int = 0
    reclaimed_endpoint_names: set = field(default_factory=set)


@dataclass(frozen=True)
class EndpointAddResult:
    endpoint_name: str
    slot_name: str
    instance_id: str = ""
    created: bool = False
    dry_run: bool = False


@dataclass(frozen=True)
class EndpointRemoveResult:
    endpoint_name: str
    instance_id: str = ""
    deleted: bool = False
    slot_removed: bool = False
    dry_run: bool = False


@dataclass(frozen=True)
class EndpointReplaceResult:
    old_endpoint_name: str
    new_endpoint_name: str
    new_slot_name: str
    old_instance_id: str = ""
    new_instance_id: str = ""
    old_deleted: bool = False
    new_created: bool = False
    dry_run: bool = False
    same_endpoint: bool = False


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
        tunnel_manager: Optional[SSHTunnelManager] = None,
        now_fn: Optional[Callable[[], float]] = None,
    ):
        self._queue = queue
        self._endpoints = endpoints_dao
        self._state = state_store
        self._kv = kv_store
        self._samples = samples_adapter
        self._client_factory = client_factory or InstanceAPIClient
        self._tunnels = tunnel_manager or SSHTunnelManager()
        self._now = now_fn or time.time
        self._force_start_after_restart = True

    async def close(self) -> None:
        await self._tunnels.stop_all()

    async def tick(self, config: GPUAutoscalerConfig) -> AutoscalerTickResult:
        state = await self._load_state()
        now = int(self._now())
        manual_request = await self._manual_endpoint_request(now, state)
        if manual_request is not None:
            return await self._process_manual_endpoint_request(
                config,
                manual_request,
                state,
            )

        if not config.enabled:
            return AutoscalerTickResult(action="disabled")
        if not config.slots:
            logger.warning("gpu-autoscaler: enabled but no managed endpoints configured")
            return AutoscalerTickResult(action="no-slots")

        snapshot = await self._snapshot()
        tunnel_reconcile_failures = await self._reconcile_active_endpoint_tunnels(
            config,
            snapshot,
        )
        if snapshot.idle:
            state.setdefault("last_busy_at", now)
        else:
            state["last_busy_at"] = now

        gpu_down_for_sec = self._update_gpu_down_state(
            state,
            snapshot,
            now,
        )
        champion_recovery_needed = (
            snapshot.active_capacity_count == 0
            and not snapshot.champion_samples_complete
        )
        force_start = (
            self._should_force_start_after_restart(snapshot)
            or champion_recovery_needed
        )
        demand_count = max(
            snapshot.pending_count,
            1 if champion_recovery_needed else 0,
        )
        desired = config.desired_instances(
            demand_count,
            gpu_down_for_sec=gpu_down_for_sec,
            force_start=force_start,
        )
        idle_for = now - int(state.get("last_busy_at", now) or now)
        action = (
            f"tunnel-reconcile-failed:{tunnel_reconcile_failures}"
            if tunnel_reconcile_failures
            else "none"
        )
        health_result = await self._check_active_endpoint_health(
            config,
            snapshot,
            state,
            now,
            blocked_slot_names=set(),
        )
        if health_result.reclaimed_count:
            action = f"endpoint-health-reclaimed:{health_result.reclaimed_count}"

        lease_result = await self._renew_busy_expiring_leases(
            config,
            snapshot,
            state,
            now,
            blocked_slot_names=set(health_result.reclaimed_endpoint_names),
        )
        if lease_result.reclaimed_count:
            action = f"endpoint-reclaimed:{lease_result.reclaimed_count}"
        if lease_result.renewed_count:
            action = f"renew-lease:{lease_result.renewed_count}"

        effective_active_capacity_count = max(
            0,
            snapshot.active_capacity_count
            - lease_result.reclaimed_count
            - health_result.reclaimed_count,
        )

        if desired > effective_active_capacity_count:
            created = await self._scale_up(
                config,
                count=desired - effective_active_capacity_count,
                blocked_slot_names=set(),
            )
            if created:
                action = f"scale-up:{created}"
                state["last_scale_at"] = now
                state.pop("gpu_down_at", None)
                self._force_start_after_restart = False
        elif (
            desired < effective_active_capacity_count
            and snapshot.idle
            and idle_for >= config.idle_seconds
        ):
            destroyed = await self._scale_down(
                config,
                snapshot,
                count=effective_active_capacity_count - desired,
                blocked_slot_names=set(),
            )
            if destroyed:
                action = f"scale-down:{destroyed}"
                state["last_scale_at"] = now
                if effective_active_capacity_count - destroyed <= 0:
                    state["gpu_down_at"] = now

        state.update(
            {
                "last_tick_at": now,
                "last_action": action,
                "last_pending_count": snapshot.pending_count,
                "last_active_capacity_count": effective_active_capacity_count,
                "last_desired_instances": desired,
                "last_idle": snapshot.idle,
                "last_idle_for_sec": idle_for,
                "last_gpu_down_for_sec": gpu_down_for_sec,
                "last_force_start_after_restart": force_start,
                "last_lease_renewed_count": lease_result.renewed_count,
                "last_lease_reclaimed_count": lease_result.reclaimed_count,
                "last_endpoint_health_checked_count": health_result.checked_count,
                "last_endpoint_health_reclaimed_count": health_result.reclaimed_count,
                "last_tunnel_reconcile_failures": tunnel_reconcile_failures,
            }
        )
        await self._kv.set(STATE_KEY, state)
        logger.info(
            "gpu-autoscaler: action=%s pending=%s active_capacity=%s desired=%s "
            "idle=%s idle_for=%ss gpu_down_for=%ss force_start=%s "
            "lease_reclaimed=%s health_checked=%s "
            "health_reclaimed=%s tunnel_reconcile_failures=%s",
            action,
            snapshot.pending_count,
            effective_active_capacity_count,
            desired,
            snapshot.idle,
            idle_for,
            gpu_down_for_sec,
            force_start,
            lease_result.reclaimed_count,
            health_result.checked_count,
            health_result.reclaimed_count,
            tunnel_reconcile_failures,
        )
        return AutoscalerTickResult(
            action=action,
            pending_count=snapshot.pending_count,
            active_capacity_count=effective_active_capacity_count,
            desired_instances=desired,
            idle=snapshot.idle,
            idle_for_sec=idle_for,
            gpu_down_for_sec=gpu_down_for_sec,
            lease_renewed_count=lease_result.renewed_count,
            lease_reclaimed_count=lease_result.reclaimed_count,
            endpoint_health_checked_count=health_result.checked_count,
            endpoint_health_reclaimed_count=health_result.reclaimed_count,
        )

    async def _manual_endpoint_request(
        self,
        now: int,
        state: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        raw = await self._kv.get(MANUAL_REPLACEMENT_LOCK_KEY)
        request = _active_manual_replacement_value(raw, now)
        if request is not None:
            token = str(request["token"])
            terminal_result = await self._kv.get(_manual_endpoint_result_key(token))
            if (
                isinstance(terminal_result, Mapping)
                and terminal_result.get("token") == token
                and terminal_result.get("status") in {"completed", "failed"}
            ):
                await self._kv.delete_if_token(
                    MANUAL_REPLACEMENT_LOCK_KEY,
                    token,
                )
                logger.warning(
                    "gpu-autoscaler: cleared completed manual endpoint "
                    "request left behind after result persistence token=%s",
                    token,
                )
                return None
            return dict(request)

        if isinstance(raw, Mapping):
            token = str(raw.get("token") or "")
            if token:
                await self._kv.delete_if_token(
                    MANUAL_REPLACEMENT_LOCK_KEY,
                    token,
                )
            logger.warning(
                "gpu-autoscaler: discarded invalid or expired manual "
                "replacement request token=%s",
                token or "-",
            )

        # Locks created by the previous direct-execution implementation
        # are not safe to replay. They no longer control daemon execution.
        legacy = state.pop(MANUAL_REPLACEMENT_STATE_KEY, None)
        if legacy is not None:
            state["last_manual_replacement"] = {
                "status": "discarded-legacy-lock",
                "completed_at": now,
                "request": dict(legacy) if isinstance(legacy, Mapping) else {},
            }
            await self._kv.set(STATE_KEY, state)
        return None

    async def _process_manual_endpoint_request(
        self,
        config: GPUAutoscalerConfig,
        request: Mapping[str, Any],
        state: Dict[str, Any],
    ) -> AutoscalerTickResult:
        request_token = str(request["token"])
        operation = _manual_endpoint_operation(request)
        action_prefix = _manual_endpoint_action_prefix(operation)
        started_at = int(self._now())
        await self._record_manual_endpoint_tick(
            state,
            action=f"{action_prefix}:running",
            replacement={
                "status": "running",
                "started_at": started_at,
                "request": dict(request),
            },
        )
        heartbeat = asyncio.create_task(
            self._manual_endpoint_heartbeat(request_token, request)
        )

        operation_result: Optional[
            EndpointAddResult | EndpointRemoveResult | EndpointReplaceResult
        ] = None
        operation_error: Optional[BaseException] = None
        cancellation: Optional[asyncio.CancelledError] = None
        try:
            if operation == MANUAL_ENDPOINT_OPERATION_ADD:
                operation_result = await self.add_endpoint(
                    config,
                    slot_name=str(request["new_slot_name"]),
                    updated_by=str(request.get("updated_by") or "gpu-autoscaler"),
                )
            elif operation == MANUAL_ENDPOINT_OPERATION_REPLACE:
                old_endpoint_name = str(request["old_endpoint_name"])
                new_slot_name = str(request["new_slot_name"])
                expected_instance_id = str(
                    request.get("expected_instance_id") or ""
                )
                expected_generation = request.get("expected_generation")
                current_endpoint = (
                    await self._endpoints.get(old_endpoint_name)
                    if expected_instance_id
                    else None
                )
                stale_health_request = expected_instance_id and (
                    current_endpoint is None
                    or str(current_endpoint.autoscale_instance_id or "")
                    != expected_instance_id
                    or int(current_endpoint.generation or 0)
                    != int(expected_generation or 0)
                )
                if stale_health_request:
                    logger.warning(
                        "gpu-autoscaler: skipped stale health replacement "
                        "endpoint=%s expected_instance=%s expected_generation=%s",
                        old_endpoint_name,
                        expected_instance_id,
                        expected_generation,
                    )
                    operation_result = EndpointReplaceResult(
                        old_endpoint_name=old_endpoint_name,
                        new_endpoint_name=new_slot_name,
                        new_slot_name=new_slot_name,
                        old_instance_id=(
                            str(current_endpoint.autoscale_instance_id or "")
                            if current_endpoint is not None
                            else ""
                        ),
                        same_endpoint=old_endpoint_name == new_slot_name,
                    )
                else:
                    operation_result = await self.replace_endpoint(
                        config,
                        old_endpoint_name=old_endpoint_name,
                        new_slot_name=new_slot_name,
                        updated_by=str(
                            request.get("updated_by") or "gpu-autoscaler"
                        ),
                    )
            else:
                operation_result = await self.remove_endpoint(
                    config,
                    endpoint_name=str(request["old_endpoint_name"]),
                    keep_slot=bool(request.get("keep_slot")),
                    updated_by=str(request.get("updated_by") or "gpu-autoscaler"),
                )
        except asyncio.CancelledError as e:
            operation_error = e
            cancellation = e
        except Exception as e:
            operation_error = e
        finally:
            heartbeat.cancel()
            try:
                await heartbeat
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.warning(
                    "gpu-autoscaler: manual endpoint heartbeat stopped "
                    "with an error token=%s: %s: %s",
                    request_token,
                    type(e).__name__,
                    e,
                )

        completed_at = int(self._now())
        status = "failed" if operation_error is not None else "completed"
        result_payload: Dict[str, Any] = {
            "request_version": MANUAL_REPLACEMENT_REQUEST_VERSION,
            "token": request_token,
            "status": status,
            "started_at": started_at,
            "completed_at": completed_at,
            "request": dict(request),
        }
        if operation_result is not None:
            result_payload["result"] = asdict(operation_result)
        if operation_error is not None:
            result_payload.update(
                {
                    "error_type": type(operation_error).__name__,
                    "error": str(operation_error) or type(operation_error).__name__,
                }
            )

        result_key = _manual_endpoint_result_key(request_token)
        action = f"{action_prefix}:{status}"
        latest_state = dict(state)
        try:
            try:
                latest_state = await self._load_state()
            except Exception as e:
                logger.error(
                    "gpu-autoscaler: failed to reload state after manual "
                    "endpoint operation token=%s: %s: %s",
                    request_token,
                    type(e).__name__,
                    e,
                    exc_info=True,
                )
            try:
                await self._kv.set(result_key, result_payload)
            except Exception as e:
                logger.error(
                    "gpu-autoscaler: failed to persist manual endpoint "
                    "result token=%s: %s: %s",
                    request_token,
                    type(e).__name__,
                    e,
                    exc_info=True,
                )
            try:
                await self._record_manual_endpoint_tick(
                    latest_state,
                    action=action,
                    replacement=result_payload,
                )
            except Exception as e:
                logger.error(
                    "gpu-autoscaler: failed to persist manual endpoint "
                    "state token=%s: %s: %s",
                    request_token,
                    type(e).__name__,
                    e,
                    exc_info=True,
                )
        finally:
            cleared = await self._kv.delete_if_token(
                MANUAL_REPLACEMENT_LOCK_KEY,
                request_token,
            )
            if not cleared:
                logger.warning(
                    "gpu-autoscaler: manual endpoint request token=%s was "
                    "already replaced or removed",
                    request_token,
                )

        if cancellation is not None:
            logger.warning(
                "gpu-autoscaler: manual endpoint operation cancelled after cleanup "
                "old_endpoint=%s new_slot=%s token=%s",
                request.get("old_endpoint_name"),
                request.get("new_slot_name"),
                request_token,
            )
            raise cancellation

        snapshot = await self._snapshot()
        gpu_down_at = int(latest_state.get("gpu_down_at") or 0)
        gpu_down_for_sec = (
            max(0, completed_at - gpu_down_at)
            if snapshot.active_capacity_count == 0 and gpu_down_at
            else 0
        )
        desired = config.desired_instances(
            snapshot.pending_count,
            gpu_down_for_sec=gpu_down_for_sec,
            force_start=False,
        )
        last_busy_at = int(latest_state.get("last_busy_at") or completed_at)
        idle_for = max(0, completed_at - last_busy_at) if snapshot.idle else 0
        replacement_log = (
            logger.warning if operation_error is not None else logger.info
        )
        replacement_log(
            "gpu-autoscaler: action=%s operation=%s old_endpoint=%s "
            "new_slot=%s token=%s error=%s",
            action,
            operation,
            request.get("old_endpoint_name"),
            request.get("new_slot_name"),
            request_token,
            str(operation_error) if operation_error is not None else "-",
        )
        return AutoscalerTickResult(
            action=action,
            pending_count=snapshot.pending_count,
            active_capacity_count=snapshot.active_capacity_count,
            desired_instances=desired,
            idle=snapshot.idle,
            idle_for_sec=idle_for,
            gpu_down_for_sec=gpu_down_for_sec,
        )

    async def _manual_endpoint_heartbeat(
        self,
        request_token: str,
        request: Mapping[str, Any],
    ) -> None:
        while True:
            await asyncio.sleep(MANUAL_REPLACEMENT_HEARTBEAT_SECONDS)
            current = await self._kv.get(MANUAL_REPLACEMENT_LOCK_KEY)
            if not isinstance(current, Mapping) or (
                str(current.get("token") or "") != request_token
            ):
                return
            try:
                state = await self._load_state()
                operation = _manual_endpoint_operation(request)
                action_prefix = _manual_endpoint_action_prefix(operation)
                await self._record_manual_endpoint_tick(
                    state,
                    action=f"{action_prefix}:running",
                    replacement={
                        "status": "running",
                        "started_at": int(request.get("requested_at") or self._now()),
                        "request": dict(request),
                    },
                )
            except Exception as e:
                logger.warning(
                    "gpu-autoscaler: failed to update manual endpoint "
                    "heartbeat token=%s: %s: %s",
                    request_token,
                    type(e).__name__,
                    e,
                )

    async def _record_manual_endpoint_tick(
        self,
        state: Dict[str, Any],
        *,
        action: str,
        replacement: Mapping[str, Any],
    ) -> None:
        state.update(
            {
                "last_tick_at": int(self._now()),
                "last_action": action,
                "last_manual_replacement": dict(replacement),
            }
        )
        await self._kv.set(STATE_KEY, state)

    async def _reconcile_active_endpoint_tunnels(
        self,
        config: GPUAutoscalerConfig,
        snapshot: AutoscalerSnapshot,
    ) -> int:
        failures = 0
        for endpoint in snapshot.active_managed:
            slot = _find_slot(config, endpoint.name)
            if slot is None or not _tunnel_enabled(slot):
                continue
            reconciled = _endpoint_with_slot_tunnel_overrides(endpoint, slot)
            repair_request = await self._pending_tunnel_repair_request(endpoint)
            try:
                if repair_request is None:
                    await self._ensure_endpoint_tunnel(slot, reconciled)
                else:
                    spec = _tunnel_spec_for_endpoint(slot, reconciled)
                    if spec is None:
                        raise RuntimeError(
                            f"endpoint {endpoint.name!r} has no tunnel spec"
                        )
                    await self._tunnels.restart(spec)
            except Exception as e:
                failures += 1
                logger.error(
                    "gpu-autoscaler: failed to reconcile tunnel for "
                    "endpoint=%s instance=%s: %s: %s",
                    endpoint.name,
                    endpoint.autoscale_instance_id,
                    type(e).__name__,
                    e,
                )
                continue
            if repair_request is not None:
                try:
                    cleared = await self._kv.delete_if_token(
                        tunnel_repair_request_key(endpoint.name),
                        str(repair_request["token"]),
                    )
                except Exception as e:
                    cleared = False
                    logger.warning(
                        "gpu-autoscaler: tunnel repaired but request cleanup "
                        "failed endpoint=%s: %s: %s",
                        endpoint.name,
                        type(e).__name__,
                        e,
                    )
                logger.warning(
                    "gpu-autoscaler: tunnel repair completed endpoint=%s "
                    "instance=%s request_cleared=%s",
                    endpoint.name,
                    endpoint.autoscale_instance_id,
                    cleared,
                )
            if reconciled != endpoint:
                try:
                    await self._endpoints.activate_autoscaled_endpoint(
                        reconciled,
                        updated_by="gpu-autoscaler",
                    )
                except Exception as e:
                    failures += 1
                    logger.error(
                        "gpu-autoscaler: failed to reconcile endpoint=%s "
                        "before tunnel setup: %s: %s",
                        endpoint.name,
                        type(e).__name__,
                        e,
                    )
                    continue
        return failures

    async def _pending_tunnel_repair_request(
        self,
        endpoint: Endpoint,
    ) -> Optional[Dict[str, Any]]:
        key = tunnel_repair_request_key(endpoint.name)
        try:
            raw = await self._kv.get(key)
        except Exception as e:
            logger.warning(
                "gpu-autoscaler: failed to read tunnel repair request for "
                "endpoint=%s: %s: %s",
                endpoint.name,
                type(e).__name__,
                e,
            )
            return None
        if not isinstance(raw, Mapping):
            return None

        token = str(raw.get("token") or "")
        now = int(self._now())
        try:
            request_version = int(raw.get("request_version") or 0)
            request_generation = int(raw.get("endpoint_generation") or 0)
            expires_at = int(raw.get("expires_at") or 0)
        except (TypeError, ValueError):
            request_version = -1
            request_generation = -1
            expires_at = 0
        valid = (
            request_version == 1
            and bool(token)
            and str(raw.get("endpoint_name") or "") == endpoint.name
            and str(raw.get("instance_id") or "")
            == str(endpoint.autoscale_instance_id or "")
            and request_generation == int(endpoint.generation or 0)
            and expires_at > now
        )
        if valid:
            return dict(raw)

        try:
            if token:
                await self._kv.delete_if_token(key, token)
            else:
                await self._kv.delete(key)
        except Exception as e:
            logger.warning(
                "gpu-autoscaler: failed to discard stale tunnel repair "
                "request endpoint=%s: %s: %s",
                endpoint.name,
                type(e).__name__,
                e,
            )
        logger.warning(
            "gpu-autoscaler: discarded stale tunnel repair request "
            "endpoint=%s instance=%s",
            endpoint.name,
            endpoint.autoscale_instance_id,
        )
        return None

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

    async def _scale_up(
        self,
        config: GPUAutoscalerConfig,
        *,
        count: int,
        blocked_slot_names: set,
    ) -> int:
        created = 0
        active_names = {
            ep.name for ep in await self._endpoints.list_active(kind="ssh")
        }
        for slot in config.slots or []:
            if created >= count:
                break
            if slot.name in active_names:
                continue
            if slot.name in blocked_slot_names:
                continue
            ok = await self._scale_up_slot(config, slot)
            if ok:
                created += 1
                active_names.add(slot.name)
        return created

    async def _scale_up_slot(
        self,
        config: GPUAutoscalerConfig,
        slot: ManagedEndpointSlot,
        *,
        updated_by: str = "gpu-autoscaler",
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
            "purpose": slot.purpose,
        }
        payload_overrides = {
            **slot.create_payload,
            "endpoint_name": slot.name,
            "provider": slot.provider,
            "purpose": slot.purpose,
        }
        handle = await client.create(
            variables=variables,
            payload_overrides=payload_overrides,
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
            await client.delete(handle.instance_id, variables=variables)
            return False
        try:
            await self._ensure_endpoint_tunnel(slot, endpoint)
        except Exception as e:
            logger.error(
                "gpu-autoscaler: tunnel setup failed after create "
                "endpoint=%s provider=%s instance=%s; deleting instance: "
                "%s: %s",
                endpoint.name,
                slot.provider,
                handle.instance_id,
                type(e).__name__,
                e,
            )
            await self._tunnels.stop(endpoint.name)
            deleted = await client.delete(handle.instance_id, variables=variables)
            if not deleted:
                logger.error(
                    "gpu-autoscaler: failed to delete instance=%s after "
                    "tunnel setup failure; manual cleanup may be required",
                    handle.instance_id,
                )
            return False
        try:
            await self._endpoints.activate_autoscaled_endpoint(
                endpoint,
                updated_by=updated_by,
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
            await self._tunnels.stop(endpoint.name)
            deleted = await client.delete(handle.instance_id, variables=variables)
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
            autoscale_purpose=slot.purpose,
            autoscale_created_at=int(self._now()),
            autoscale_updated_at=int(self._now()),
            autoscale_lease_expires_at=int(handle.lease_expires_at or 0),
        )
        for field_name, value in (slot.endpoint or {}).items():
            if field_name in _ENDPOINT_OVERRIDE_FIELDS and value is not None:
                setattr(endpoint, field_name, value)
        endpoint.ssh_url = handle.ssh_url or endpoint.ssh_url
        slot_public_url = (slot.endpoint or {}).get("public_inference_url")
        endpoint.public_inference_url = (
            slot_public_url
            or handle.public_inference_url
            or endpoint.public_inference_url
        )
        if not endpoint.notes:
            endpoint.notes = f"autoscaled via {slot.provider}"
        return endpoint

    async def _check_active_endpoint_health(
        self,
        config: GPUAutoscalerConfig,
        snapshot: AutoscalerSnapshot,
        state: Dict[str, Any],
        now: int,
        *,
        blocked_slot_names: set,
    ) -> EndpointHealthCheckResult:
        if not snapshot.active_managed:
            return EndpointHealthCheckResult()
        if config.endpoint_health_check_interval_seconds <= 0:
            return EndpointHealthCheckResult()

        attempts = dict(state.get("endpoint_health_checked_at") or {})
        changed_attempts = False
        checked = 0
        reclaimed = 0
        reclaimed_names = set()
        for endpoint in snapshot.active_managed:
            if endpoint.name in blocked_slot_names:
                continue
            last_attempt = int(attempts.get(endpoint.name) or 0)
            if (
                last_attempt
                and now - last_attempt < config.endpoint_health_check_interval_seconds
            ):
                continue
            attempts[endpoint.name] = now
            changed_attempts = True
            outcome = await self._check_endpoint_health(config, endpoint)
            if outcome == "skipped":
                attempts.pop(endpoint.name, None)
                changed_attempts = True
                continue
            checked += 1
            if outcome == "reclaimed":
                reclaimed += 1
                reclaimed_names.add(endpoint.name)
                attempts.pop(endpoint.name, None)
                changed_attempts = True

        if changed_attempts:
            if attempts:
                state["endpoint_health_checked_at"] = attempts
            else:
                state.pop("endpoint_health_checked_at", None)
        return EndpointHealthCheckResult(
            checked_count=checked,
            reclaimed_count=reclaimed,
            reclaimed_endpoint_names=reclaimed_names,
        )

    async def _check_endpoint_health(
        self,
        config: GPUAutoscalerConfig,
        endpoint: Endpoint,
    ) -> str:
        provider = (endpoint.autoscale_provider or "").lower()
        provider_config = (config.providers or {}).get(provider)
        if provider_config is None or not provider_config.status_path:
            return "skipped"
        instance_id = endpoint.autoscale_instance_id or ""
        if not instance_id:
            logger.warning(
                "gpu-autoscaler: endpoint=%s has no autoscale_instance_id",
                endpoint.name,
            )
            return "failed"
        if config.dry_run:
            logger.info(
                "gpu-autoscaler: dry-run would check endpoint=%s instance=%s health",
                endpoint.name,
                instance_id,
            )
            return "healthy"

        client = self._client_factory(provider_config)
        try:
            status = await client.status(instance_id)
        except InstanceAPINotFoundError as e:
            logger.warning(
                "gpu-autoscaler: endpoint=%s provider=%s instance=%s was "
                "missing during health check; deactivating endpoint so "
                "autoscaler can request replacement: %s",
                endpoint.name,
                provider,
                instance_id,
                e,
            )
            deactivated = await self._deactivate_endpoint(
                replace(endpoint),
                instance_id=instance_id,
                provider=provider,
                updated_by="gpu-autoscaler:provider-reclaimed",
            )
            return "reclaimed" if deactivated else "failed"
        if status is None:
            return "failed"
        return "healthy"

    async def _ensure_endpoint_tunnel(
        self,
        slot: ManagedEndpointSlot,
        endpoint: Endpoint,
    ) -> None:
        spec = _tunnel_spec_for_endpoint(slot, endpoint)
        if spec is None:
            return
        await self._tunnels.ensure(spec)

    async def _renew_busy_expiring_leases(
        self,
        config: GPUAutoscalerConfig,
        snapshot: AutoscalerSnapshot,
        state: Dict[str, Any],
        now: int,
        *,
        blocked_slot_names: set,
    ) -> LeaseRenewResult:
        if snapshot.idle or not snapshot.active_managed:
            return LeaseRenewResult()
        if config.lease_renew_margin_seconds <= 0:
            return LeaseRenewResult()

        attempts = dict(state.get("lease_renew_attempted_at") or {})
        changed_attempts = False
        renewed = 0
        reclaimed = 0
        for endpoint in snapshot.active_managed:
            if endpoint.name in blocked_slot_names:
                continue
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
            outcome = await self._renew_endpoint_lease(config, endpoint, now)
            if outcome == "renewed":
                renewed += 1
                attempts.pop(endpoint.name, None)
                changed_attempts = True
            elif outcome == "reclaimed":
                reclaimed += 1
                attempts.pop(endpoint.name, None)
                changed_attempts = True

        if changed_attempts:
            if attempts:
                state["lease_renew_attempted_at"] = attempts
            else:
                state.pop("lease_renew_attempted_at", None)
        return LeaseRenewResult(
            renewed_count=renewed,
            reclaimed_count=reclaimed,
        )

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
    ) -> str:
        provider = (endpoint.autoscale_provider or "").lower()
        provider_config = (config.providers or {}).get(provider)
        if provider_config is None:
            logger.warning(
                "gpu-autoscaler: endpoint=%s has no provider API config",
                endpoint.name,
            )
            return "failed"
        instance_id = endpoint.autoscale_instance_id or ""
        if not instance_id:
            logger.warning(
                "gpu-autoscaler: endpoint=%s has no autoscale_instance_id",
                endpoint.name,
            )
            return "failed"
        if not provider_config.renew_path:
            logger.warning(
                "gpu-autoscaler: endpoint=%s provider=%s has no renew_path",
                endpoint.name,
                provider,
            )
            return "failed"
        if config.dry_run:
            logger.info(
                "gpu-autoscaler: dry-run would renew endpoint=%s instance=%s",
                endpoint.name,
                instance_id,
            )
            return "renewed"

        client = self._client_factory(provider_config)
        try:
            handle = await client.renew(instance_id)
        except InstanceAPINotFoundError as e:
            logger.warning(
                "gpu-autoscaler: endpoint=%s provider=%s instance=%s was "
                "reclaimed by provider; deactivating endpoint so autoscaler "
                "can request replacement: %s",
                endpoint.name,
                provider,
                instance_id,
                e,
            )
            deactivated = await self._deactivate_endpoint(
                replace(endpoint),
                instance_id=instance_id,
                provider=provider,
                updated_by="gpu-autoscaler:provider-reclaimed",
            )
            return "reclaimed" if deactivated else "failed"
        if handle is None:
            return "failed"
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
            return "failed"
        logger.info(
            "gpu-autoscaler: renewed endpoint=%s provider=%s instance=%s "
            "lease_expires_at=%s",
            endpoint.name,
            provider,
            instance_id,
            lease_expires_at or "-",
        )
        return "renewed"

    async def _scale_down(
        self,
        config: GPUAutoscalerConfig,
        snapshot: AutoscalerSnapshot,
        *,
        count: int,
        blocked_slot_names: set,
    ) -> int:
        destroyed = 0
        slot_order = {
            slot.name: idx for idx, slot in enumerate(config.slots or [])
        }
        candidates = sorted(
            [
                ep for ep in snapshot.active_managed
                if ep.name not in blocked_slot_names
            ],
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
        self,
        config: GPUAutoscalerConfig,
        endpoint: Endpoint,
        *,
        updated_by: str = "gpu-autoscaler",
        deactivate_before_delete: bool = False,
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
        delete_vars = {"endpoint_name": endpoint.name}
        purpose = getattr(endpoint, "autoscale_purpose", None)
        if purpose:
            delete_vars["purpose"] = purpose
        endpoint_for_refs = replace(endpoint)
        if deactivate_before_delete:
            drained = await self._drain_endpoint(
                endpoint_for_refs,
                instance_id=instance_id,
                provider=provider,
                updated_by=updated_by,
            )
            if not drained:
                return False

        try:
            deleted = await client.delete(instance_id, variables=delete_vars)
        except Exception as e:
            logger.error(
                "gpu-autoscaler: delete endpoint=%s provider=%s instance=%s "
                "failed: %s: %s",
                endpoint.name,
                provider,
                instance_id,
                type(e).__name__,
                e,
            )
            if deactivate_before_delete:
                await self._restore_drained_endpoint(
                    endpoint_for_refs,
                    instance_id=instance_id,
                    provider=provider,
                    updated_by=updated_by,
                )
            return False
        if not deleted:
            logger.error(
                "gpu-autoscaler: delete endpoint=%s provider=%s instance=%s "
                "returned false",
                endpoint.name,
                provider,
                instance_id,
            )
            if deactivate_before_delete:
                await self._restore_drained_endpoint(
                    endpoint_for_refs,
                    instance_id=instance_id,
                    provider=provider,
                    updated_by=updated_by,
                )
            return False

        if deactivate_before_delete:
            await self._deactivate_endpoint(
                endpoint_for_refs,
                instance_id=instance_id,
                provider=provider,
                updated_by=updated_by,
                clear_refs=False,
            )
            logger.info(
                "gpu-autoscaler: deleted drained endpoint=%s provider=%s "
                "instance=%s",
                endpoint.name,
                provider,
                instance_id,
            )
            return True
        return await self._deactivate_endpoint(
            endpoint_for_refs,
            instance_id=instance_id,
            provider=provider,
            updated_by=updated_by,
        )

    async def _drain_endpoint(
        self,
        endpoint: Endpoint,
        *,
        instance_id: str,
        provider: str,
        updated_by: str,
    ) -> bool:
        try:
            await self._endpoints.drain_autoscaled_endpoint(
                endpoint.name,
                instance_id=instance_id,
                updated_by=updated_by,
            )
        except Exception as e:
            logger.warning(
                "gpu-autoscaler: drain endpoint=%s instance=%s failed: "
                "%s: %s",
                endpoint.name,
                instance_id,
                type(e).__name__,
                e,
            )
            return False
        await self._clear_deployment_refs(endpoint)
        logger.info(
            "gpu-autoscaler: drained endpoint=%s provider=%s instance=%s",
            endpoint.name,
            provider,
            instance_id,
        )
        return True

    async def _restore_drained_endpoint(
        self,
        endpoint: Endpoint,
        *,
        instance_id: str,
        provider: str,
        updated_by: str,
    ) -> None:
        try:
            await self._endpoints.activate_autoscaled_endpoint(
                endpoint,
                updated_by=updated_by,
            )
        except Exception as e:
            logger.error(
                "gpu-autoscaler: failed to restore drained endpoint=%s "
                "provider=%s instance=%s after delete failure; manual "
                "cleanup may be required: %s: %s",
                endpoint.name,
                provider,
                instance_id,
                type(e).__name__,
                e,
            )
            return
        logger.warning(
            "gpu-autoscaler: restored drained endpoint=%s provider=%s "
            "instance=%s after delete failure",
            endpoint.name,
            provider,
            instance_id,
        )

    async def _deactivate_endpoint(
        self,
        endpoint: Endpoint,
        *,
        instance_id: str,
        provider: str,
        updated_by: str,
        clear_refs: bool = True,
    ) -> bool:
        try:
            await self._endpoints.deactivate_autoscaled_endpoint(
                endpoint.name,
                instance_id=instance_id,
                updated_by=updated_by,
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
        if clear_refs:
            await self._clear_deployment_refs(endpoint)
        await self._tunnels.stop(endpoint.name)
        logger.info(
            "gpu-autoscaler: deactivated endpoint=%s provider=%s instance=%s",
            endpoint.name,
            provider,
            instance_id,
        )
        return True

    async def _clear_deployment_refs(self, endpoint: Endpoint) -> None:
        deployment_id = endpoint.deployment_id or ""
        base_url = endpoint.base_url or endpoint.public_inference_url or ""
        endpoint_name = endpoint.name

        def _matches_endpoint(dep) -> bool:
            return (
                dep.endpoint_name == endpoint_name
                or (deployment_id and dep.deployment_id == deployment_id)
                or (base_url and dep.base_url == base_url)
            )

        def _record_matches(record) -> bool:
            return (
                (deployment_id and record.deployment_id == deployment_id)
                or (base_url and record.base_url == base_url)
                or any(_matches_endpoint(dep) for dep in record.deployments)
            )

        champion = await self._state.get_champion()
        if champion:
            deployments = [
                dep for dep in (champion.deployments or [])
                if not _matches_endpoint(dep)
            ]
            changed = len(deployments) != len(champion.deployments or [])
            primary_removed = (
                (deployment_id and champion.deployment_id == deployment_id)
                or (base_url and champion.base_url == base_url)
            )
            if changed or primary_removed:
                champion.deployments = deployments
                if deployments:
                    champion.deployment_id = deployments[0].deployment_id
                    champion.base_url = deployments[0].base_url
                else:
                    champion.deployment_id = None
                    champion.base_url = None
                await self._state.set_champion(champion)

        battle = await self._state.get_battle()
        if battle is not None and _record_matches(battle):
            released = await self._queue.release_claim(
                battle.challenger.uid,
                hotkey=battle.challenger.hotkey,
                revision=battle.challenger.revision,
            )
            await self._state.clear_battle()
            logger.warning(
                "gpu-autoscaler: cleared battle uid=%s after endpoint=%s "
                "deactivation (claim_released=%s)",
                battle.challenger.uid,
                endpoint_name,
                released,
            )

        records = await self._state.get_predeployed_challengers()
        if records:
            kept = [record for record in records if not _record_matches(record)]
            if len(kept) != len(records):
                await self._state.set_predeployed_challengers(kept)
                logger.warning(
                    "gpu-autoscaler: dropped %s predeployed record(s) after "
                    "endpoint=%s deactivation",
                    len(records) - len(kept),
                    endpoint_name,
                )

    async def add_endpoint(
        self,
        config: GPUAutoscalerConfig,
        *,
        slot_name: str,
        updated_by: str = "cli:gpu-add-endpoint",
    ) -> EndpointAddResult:
        """Create one configured slot without replacing another endpoint."""
        slot = _find_slot(config, slot_name)
        if slot is None:
            raise ValueError(
                f"autoscaler config has no endpoint slot {slot_name!r}"
            )
        await self._validate_slot_available(slot)

        created = await self._scale_up_slot(
            config,
            slot,
            updated_by=updated_by,
        )
        if not created:
            raise RuntimeError(
                f"failed to create endpoint from slot {slot.name!r}"
            )

        endpoint = await self._endpoints.get(slot.name)
        instance_id = (
            endpoint.autoscale_instance_id if endpoint is not None else ""
        ) or ""
        return EndpointAddResult(
            endpoint_name=slot.name,
            slot_name=slot.name,
            instance_id=instance_id,
            created=True,
            dry_run=config.dry_run,
        )

    async def _validate_slot_available(self, slot: ManagedEndpointSlot) -> None:
        existing = await self._endpoints.get(slot.name)
        if existing is not None and existing.active:
            raise ValueError(f"endpoint {slot.name!r} is already active")
        if existing is not None and existing.autoscale_instance_id:
            raise ValueError(
                f"endpoint {slot.name!r} still references provider instance "
                f"{existing.autoscale_instance_id!r}; remove it before adding"
            )
        if existing is not None and not existing.autoscale_managed:
            raise ValueError(
                f"endpoint {slot.name!r} is operator-managed, not an "
                "autoscaler slot"
            )

    async def replace_endpoint(
        self,
        config: GPUAutoscalerConfig,
        *,
        old_endpoint_name: str,
        new_slot_name: Optional[str] = None,
        updated_by: str = "cli:gpu-replace-endpoint",
    ) -> EndpointReplaceResult:
        """Execute one replacement inside the daemon's serialized tick.

        External callers must enqueue through ``af gpu replace-endpoint`` so
        provider and tunnel mutations stay owned by the daemon process.
        """
        new_slot_name = new_slot_name or old_endpoint_name
        slot = _find_slot(config, new_slot_name)
        if slot is None:
            raise ValueError(
                f"autoscaler config has no endpoint slot {new_slot_name!r}"
            )

        old_endpoint = await self._endpoints.get(old_endpoint_name)
        if old_endpoint is None:
            raise ValueError(f"old endpoint {old_endpoint_name!r} does not exist")
        _validate_autoscaled_endpoint(old_endpoint)

        same_endpoint = old_endpoint_name == slot.name
        if not same_endpoint:
            await self._validate_slot_available(slot)
        old_instance_id = old_endpoint.autoscale_instance_id or ""
        if config.dry_run:
            return EndpointReplaceResult(
                old_endpoint_name=old_endpoint_name,
                new_endpoint_name=slot.name,
                new_slot_name=slot.name,
                old_instance_id=old_instance_id,
                old_deleted=True,
                new_created=True,
                dry_run=True,
                same_endpoint=same_endpoint,
            )

        old_deleted = False
        new_created = False

        if same_endpoint:
            old_deleted = await self._scale_down_endpoint(
                config,
                old_endpoint,
                updated_by=updated_by,
                deactivate_before_delete=True,
            )
            if not old_deleted:
                raise RuntimeError(
                    f"failed to delete old endpoint {old_endpoint_name!r}; "
                    "same-endpoint replacement stopped before creating "
                    "the new instance"
                )
            new_created = await self._scale_up_slot(
                config,
                slot,
                updated_by=updated_by,
            )
            if not new_created:
                raise RuntimeError(
                    f"failed to create replacement endpoint from slot "
                    f"{slot.name!r}; old endpoint {old_endpoint_name!r} "
                    "was already deleted"
                )
        else:
            new_created = await self._scale_up_slot(
                config,
                slot,
                updated_by=updated_by,
            )
            if not new_created:
                raise RuntimeError(
                    f"failed to create replacement endpoint from slot "
                    f"{slot.name!r}; old endpoint left untouched"
                )
            old_deleted = await self._scale_down_endpoint(
                config,
                old_endpoint,
                updated_by=updated_by,
                deactivate_before_delete=True,
            )
            if not old_deleted:
                logger.error(
                    "gpu-autoscaler: replacement endpoint=%s is active, "
                    "but old endpoint=%s instance=%s was not deleted; "
                    "manual cleanup may be required",
                    slot.name,
                    old_endpoint_name,
                    old_instance_id,
                )

        new_endpoint = await self._endpoints.get(slot.name)
        new_instance_id = (
            new_endpoint.autoscale_instance_id if new_endpoint is not None else ""
        ) or ""
        return EndpointReplaceResult(
            old_endpoint_name=old_endpoint_name,
            new_endpoint_name=slot.name,
            new_slot_name=slot.name,
            old_instance_id=old_instance_id,
            new_instance_id=new_instance_id,
            old_deleted=old_deleted,
            new_created=new_created,
            dry_run=False,
            same_endpoint=same_endpoint,
        )

    async def remove_endpoint(
        self,
        config: GPUAutoscalerConfig,
        *,
        endpoint_name: str,
        keep_slot: bool = False,
        updated_by: str = "cli:gpu-remove-endpoint",
    ) -> EndpointRemoveResult:
        """Remove one endpoint inside the daemon's serialized tick."""
        endpoint = await self._endpoints.get(endpoint_name)
        if endpoint is None:
            raise ValueError(f"endpoint {endpoint_name!r} does not exist")
        _validate_autoscaled_endpoint(endpoint)

        payload = await self._kv.get(CONFIG_KEY, default=None)
        if not isinstance(payload, Mapping) or not payload:
            raise ValueError("gpu_autoscaler config is missing")
        original_payload = dict(payload)
        next_payload, slot_present = _config_without_endpoint_slot(
            original_payload,
            endpoint_name,
        )
        instance_id = endpoint.autoscale_instance_id or ""
        if config.dry_run:
            return EndpointRemoveResult(
                endpoint_name=endpoint_name,
                instance_id=instance_id,
                deleted=True,
                slot_removed=not keep_slot and slot_present,
                dry_run=True,
            )

        slot_removed = not keep_slot and slot_present
        try:
            if slot_removed:
                await self._write_config_payload(next_payload)
            deleted = await self._scale_down_endpoint(
                GPUAutoscalerConfig.from_mapping(original_payload),
                endpoint,
                updated_by=updated_by,
                deactivate_before_delete=True,
            )
            if not deleted:
                raise RuntimeError(f"failed to delete endpoint {endpoint_name!r}")
        except BaseException:
            if slot_removed:
                try:
                    await self._write_config_payload(original_payload)
                except Exception as restore_error:
                    logger.error(
                        "gpu-autoscaler: failed to restore slot after removal "
                        "failure endpoint=%s: %s: %s",
                        endpoint_name,
                        type(restore_error).__name__,
                        restore_error,
                        exc_info=True,
                    )
            raise

        return EndpointRemoveResult(
            endpoint_name=endpoint_name,
            instance_id=instance_id,
            deleted=True,
            slot_removed=slot_removed,
        )

    async def _write_config_payload(self, payload: Mapping[str, Any]) -> None:
        writer = getattr(self._kv, "set_preserving_metadata", None)
        if callable(writer):
            await writer(CONFIG_KEY, dict(payload))
            return
        await self._kv.set(CONFIG_KEY, dict(payload))

    async def _load_state(self) -> Dict[str, Any]:
        raw = await self._kv.get(STATE_KEY, default={})
        return dict(raw) if isinstance(raw, Mapping) else {}


def _active_manual_replacement_value(
    raw: Any,
    now: int,
) -> Optional[Mapping[str, Any]]:
    if not isinstance(raw, Mapping):
        return None
    if int(raw.get("request_version") or 0) != (MANUAL_REPLACEMENT_REQUEST_VERSION):
        return None
    if raw.get("status") != "requested" or not raw.get("token"):
        return None
    expires_at = int(raw.get("expires_at") or 0)
    if expires_at <= now:
        return None
    try:
        _manual_endpoint_operation(raw)
    except ValueError:
        return None
    return raw


def _manual_endpoint_operation(request: Mapping[str, Any]) -> str:
    operation = str(request.get("operation") or "")
    if operation in {
        MANUAL_ENDPOINT_OPERATION_ADD,
        MANUAL_ENDPOINT_OPERATION_REMOVE,
        MANUAL_ENDPOINT_OPERATION_REPLACE,
    }:
        return operation
    if operation:
        raise ValueError(f"unsupported manual endpoint operation {operation!r}")

    # Requests written before add-endpoint was split from --keep-old did not
    # carry an operation. Preserve their behavior while the queue drains.
    if request.get("delete_old") is False:
        return MANUAL_ENDPOINT_OPERATION_ADD
    return MANUAL_ENDPOINT_OPERATION_REPLACE


def _manual_endpoint_action_prefix(operation: str) -> str:
    return {
        MANUAL_ENDPOINT_OPERATION_ADD: "manual-add",
        MANUAL_ENDPOINT_OPERATION_REMOVE: "manual-remove",
        MANUAL_ENDPOINT_OPERATION_REPLACE: "manual-replacement",
    }[operation]


def _tunnel_enabled(slot: ManagedEndpointSlot) -> bool:
    return _bool_value((slot.tunnel or {}).get("enabled"), default=False)


def _endpoint_with_slot_tunnel_overrides(
    endpoint: Endpoint,
    slot: ManagedEndpointSlot,
) -> Endpoint:
    slot_public_url = (slot.endpoint or {}).get("public_inference_url")
    if slot_public_url and endpoint.public_inference_url != slot_public_url:
        return replace(endpoint, public_inference_url=str(slot_public_url))
    return endpoint


def _tunnel_spec_for_endpoint(
    slot: ManagedEndpointSlot,
    endpoint: Endpoint,
) -> Optional[TunnelSpec]:
    if not _tunnel_enabled(slot):
        return None
    instance_id = endpoint.autoscale_instance_id or ""
    ssh_url = endpoint.ssh_url or ""
    public_url = endpoint.public_inference_url or ""
    if not instance_id or not ssh_url or not public_url:
        raise ValueError(
            f"endpoint {endpoint.name!r} tunnel requires instance_id, "
            "ssh_url, and public_inference_url"
        )
    tunnel = slot.tunnel or {}
    target_port = int(
        tunnel.get("target_port")
        or tunnel.get("remote_port")
        or endpoint.sglang_port
        or 10001
    )
    return TunnelSpec(
        endpoint_name=endpoint.name,
        instance_id=instance_id,
        ssh_url=ssh_url,
        ssh_key_path=str(tunnel.get("ssh_key_path") or endpoint.ssh_key_path or ""),
        public_url=public_url,
        bind_host=str(tunnel.get("bind_host") or "0.0.0.0"),
        target_host=str(tunnel.get("target_host") or "127.0.0.1"),
        target_port=target_port,
        connect_timeout_sec=float(tunnel.get("connect_timeout_sec") or 10.0),
    )


def _parse_ssh_url(url: str) -> tuple[str, str, int]:
    parsed = urlparse(url or "")
    if parsed.scheme != "ssh" or not parsed.hostname:
        raise ValueError(f"invalid ssh_url: {url!r}")
    return parsed.username or "root", parsed.hostname, int(parsed.port or 22)


async def _wait_for_tunnel_process(
    process: Process,
    host: str,
    port: int,
    *,
    timeout_sec: float,
) -> None:
    deadline = time.monotonic() + max(0.1, timeout_sec)
    last_error: Optional[Exception] = None
    while time.monotonic() < deadline:
        if process.returncode is not None:
            raise RuntimeError(
                f"ssh tunnel process exited with code {process.returncode}"
            )
        try:
            _, writer = await asyncio.open_connection(host, port)
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
            # An old listener can make the port probe pass while this ssh
            # process is still failing to bind. ExitOnForwardFailure makes
            # that process exit promptly, so require it to remain alive for
            # a short stabilization interval before accepting the tunnel.
            await asyncio.sleep(0.5)
            if process.returncode is None:
                return
            raise RuntimeError(
                f"ssh tunnel process exited with code {process.returncode}"
            )
        except Exception as e:
            last_error = e
            await asyncio.sleep(0.2)
    raise TimeoutError(
        f"local tunnel port {host}:{port} did not become ready"
    ) from last_error


async def _local_port_is_open(host: str, port: int) -> bool:
    try:
        _, writer = await asyncio.open_connection(host, port)
    except Exception:
        return False
    writer.close()
    with contextlib.suppress(Exception):
        await writer.wait_closed()
    return True


async def _stop_process(process: Process, *, timeout_sec: float) -> None:
    if process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=timeout_sec)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()


def _manual_endpoint_result_key(token: str) -> str:
    return f"{MANUAL_REPLACEMENT_RESULT_KEY_PREFIX}{token}"


async def _enqueue_manual_endpoint_request(
    kv,
    *,
    operation: str,
    updated_by: str,
    now: int,
    new_slot_name: str = "",
    old_endpoint_name: str = "",
    keep_slot: bool = False,
    ttl_seconds: int = DEFAULT_MANUAL_REPLACEMENT_TTL_SECONDS,
    expected_instance_id: str = "",
    expected_generation: Optional[int] = None,
    reason: str = "",
) -> Dict[str, Any]:
    if operation not in {
        MANUAL_ENDPOINT_OPERATION_ADD,
        MANUAL_ENDPOINT_OPERATION_REMOVE,
        MANUAL_ENDPOINT_OPERATION_REPLACE,
    }:
        raise ValueError(f"unsupported manual endpoint operation {operation!r}")
    if operation in {
        MANUAL_ENDPOINT_OPERATION_REMOVE,
        MANUAL_ENDPOINT_OPERATION_REPLACE,
    } and not old_endpoint_name:
        raise ValueError(f"{operation} operation requires old_endpoint_name")
    if operation in {
        MANUAL_ENDPOINT_OPERATION_ADD,
        MANUAL_ENDPOINT_OPERATION_REPLACE,
    } and not new_slot_name:
        raise ValueError(f"{operation} operation requires new_slot_name")

    ttl = max(1, int(ttl_seconds or 0))
    request = {
        "request_version": MANUAL_REPLACEMENT_REQUEST_VERSION,
        "status": "requested",
        "token": uuid.uuid4().hex,
        "operation": operation,
        "old_endpoint_name": old_endpoint_name,
        "new_slot_name": new_slot_name,
        "requested_at": int(now),
        "expires_at": int(now) + ttl,
        "updated_by": updated_by,
    }
    if operation == MANUAL_ENDPOINT_OPERATION_REMOVE:
        request["keep_slot"] = bool(keep_slot)
    if expected_instance_id:
        request["expected_instance_id"] = expected_instance_id
    if expected_generation is not None:
        request["expected_generation"] = int(expected_generation)
    if reason:
        request["reason"] = reason
    acquired = await kv.set_if_absent_or_expired(
        MANUAL_REPLACEMENT_LOCK_KEY,
        request,
        expires_at_field="expires_at",
        now=int(now),
    )
    if acquired:
        return request

    existing = await kv.get(MANUAL_REPLACEMENT_LOCK_KEY)
    existing = existing if isinstance(existing, Mapping) else {}
    raise RuntimeError(
        "manual GPU endpoint operation already queued for "
        f"operation={existing.get('operation')!r} "
        f"old_endpoint={existing.get('old_endpoint_name')!r} "
        f"new_slot={existing.get('new_slot_name')!r}; retry after "
        f"expires_at={existing.get('expires_at')}"
    )


async def enqueue_health_endpoint_replacement_request(
    kv,
    *,
    endpoint_name: str,
    instance_id: str,
    endpoint_generation: int,
    reason: str,
    now: int,
) -> bool:
    """Queue a same-slot replacement fenced to the observed machine."""
    try:
        await _enqueue_manual_endpoint_request(
            kv,
            operation=MANUAL_ENDPOINT_OPERATION_REPLACE,
            old_endpoint_name=endpoint_name,
            new_slot_name=endpoint_name,
            updated_by="scheduler:deployment-health",
            now=now,
            expected_instance_id=instance_id,
            expected_generation=endpoint_generation,
            reason=reason,
        )
        return True
    except RuntimeError:
        existing = await kv.get(MANUAL_REPLACEMENT_LOCK_KEY)
        if not isinstance(existing, Mapping):
            return False
        return (
            _active_manual_replacement_value(existing, int(now)) is not None
            and existing.get("operation") == MANUAL_ENDPOINT_OPERATION_REPLACE
            and existing.get("old_endpoint_name") == endpoint_name
            and existing.get("new_slot_name") == endpoint_name
            and str(existing.get("expected_instance_id") or "") == instance_id
            and int(existing.get("expected_generation") or 0)
            == int(endpoint_generation)
        )


async def _enqueue_manual_replacement_request(
    kv,
    *,
    old_endpoint_name: str,
    new_slot_name: str,
    updated_by: str,
    now: int,
    ttl_seconds: int = DEFAULT_MANUAL_REPLACEMENT_TTL_SECONDS,
) -> Dict[str, Any]:
    return await _enqueue_manual_endpoint_request(
        kv,
        operation=MANUAL_ENDPOINT_OPERATION_REPLACE,
        old_endpoint_name=old_endpoint_name,
        new_slot_name=new_slot_name,
        updated_by=updated_by,
        now=now,
        ttl_seconds=ttl_seconds,
    )


async def _enqueue_manual_add_request(
    kv,
    *,
    slot_name: str,
    updated_by: str,
    now: int,
    ttl_seconds: int = DEFAULT_MANUAL_REPLACEMENT_TTL_SECONDS,
) -> Dict[str, Any]:
    return await _enqueue_manual_endpoint_request(
        kv,
        operation=MANUAL_ENDPOINT_OPERATION_ADD,
        new_slot_name=slot_name,
        updated_by=updated_by,
        now=now,
        ttl_seconds=ttl_seconds,
    )


async def _enqueue_manual_remove_request(
    kv,
    *,
    endpoint_name: str,
    keep_slot: bool,
    updated_by: str,
    now: int,
    ttl_seconds: int = DEFAULT_MANUAL_REPLACEMENT_TTL_SECONDS,
) -> Dict[str, Any]:
    return await _enqueue_manual_endpoint_request(
        kv,
        operation=MANUAL_ENDPOINT_OPERATION_REMOVE,
        old_endpoint_name=endpoint_name,
        keep_slot=keep_slot,
        updated_by=updated_by,
        now=now,
        ttl_seconds=ttl_seconds,
    )


async def _wait_for_manual_endpoint_result(
    kv,
    request: Mapping[str, Any],
    *,
    timeout_seconds: float,
    poll_seconds: float = DEFAULT_MANUAL_REPLACEMENT_RESULT_POLL_SECONDS,
) -> Dict[str, Any]:
    token = str(request["token"])
    result_key = _manual_endpoint_result_key(token)
    deadline = time.monotonic() + max(0.1, float(timeout_seconds))
    missing_since: Optional[float] = None
    while time.monotonic() < deadline:
        result = await kv.get(result_key)
        if (
            isinstance(result, Mapping)
            and result.get("token") == token
            and result.get("status") in {"completed", "failed"}
        ):
            return dict(result)

        current = await kv.get(MANUAL_REPLACEMENT_LOCK_KEY)
        if not isinstance(current, Mapping) or current.get("token") != token:
            now = time.monotonic()
            missing_since = missing_since or now
            if (
                now - missing_since
                >= MANUAL_REPLACEMENT_RESULT_VISIBILITY_GRACE_SECONDS
            ):
                raise RuntimeError(
                    "manual GPU endpoint request disappeared before a "
                    f"result was recorded (token={token})"
                )
        else:
            missing_since = None
        await asyncio.sleep(max(0.05, float(poll_seconds)))
    raise TimeoutError(
        "timed out waiting for gpu-autoscaler daemon to complete manual "
        f"endpoint operation (token={token})"
    )


def _endpoint_replace_result_from_payload(
    payload: Mapping[str, Any],
) -> EndpointReplaceResult:
    raw = payload.get("result")
    if not isinstance(raw, Mapping):
        raise RuntimeError("manual GPU replacement completed without result data")
    field_names = EndpointReplaceResult.__dataclass_fields__
    return EndpointReplaceResult(
        **{key: value for key, value in raw.items() if key in field_names}
    )


def _endpoint_add_result_from_payload(
    payload: Mapping[str, Any],
) -> EndpointAddResult:
    raw = payload.get("result")
    if not isinstance(raw, Mapping):
        raise RuntimeError("manual GPU add completed without result data")
    field_names = EndpointAddResult.__dataclass_fields__
    return EndpointAddResult(
        **{key: value for key, value in raw.items() if key in field_names}
    )


def _endpoint_remove_result_from_payload(
    payload: Mapping[str, Any],
) -> EndpointRemoveResult:
    raw = payload.get("result")
    if not isinstance(raw, Mapping):
        raise RuntimeError("manual GPU removal completed without result data")
    field_names = EndpointRemoveResult.__dataclass_fields__
    return EndpointRemoveResult(
        **{key: value for key, value in raw.items() if key in field_names}
    )


def _cli_autoscaler(
    endpoints_dao: InferenceEndpointsDAO,
    kv,
    *,
    validator_hotkey: str,
) -> GPUAutoscaler:
    return GPUAutoscaler(
        queue=ChallengerQueue(MinersQueueAdapter(MinersDAO())),
        endpoints_dao=endpoints_dao,
        state_store=StateStore(kv),
        kv_store=kv,
        samples_adapter=SampleResultsAdapter(
            dao=SampleResultsDAO(),
            validator_hotkey=validator_hotkey,
        ),
    )


async def _cleanup_manual_endpoint_result(
    kv,
    request: Mapping[str, Any],
    *,
    operation: str,
) -> None:
    try:
        await kv.delete(_manual_endpoint_result_key(str(request["token"])))
    except Exception as e:
        logger.warning(
            "gpu-autoscaler: %s succeeded but result cleanup failed "
            "token=%s: %s: %s",
            operation,
            request["token"],
            type(e).__name__,
            e,
        )


async def load_config(config_dao: SystemConfigDAO) -> GPUAutoscalerConfig:
    payload = await config_dao.get_param_value(CONFIG_KEY, default=None)
    if isinstance(payload, Mapping) and payload:
        return GPUAutoscalerConfig.from_mapping(payload)
    payload = {}
    env_payload = _load_env_json("AFFINE_GPU_AUTOSCALER_CONFIG_JSON")
    if env_payload:
        payload = dict(env_payload)
    return GPUAutoscalerConfig.from_mapping(payload, apply_env_overrides=True)


def _find_slot(
    config: GPUAutoscalerConfig,
    name: str,
) -> Optional[ManagedEndpointSlot]:
    for slot in config.slots or []:
        if slot.name == name:
            return slot
    return None


def _validate_autoscaled_endpoint(endpoint: Endpoint) -> None:
    if not endpoint.autoscale_managed:
        raise ValueError(
            f"endpoint {endpoint.name!r} is not autoscaler-managed; refusing "
            "to delete it automatically"
        )
    if not endpoint.autoscale_provider:
        raise ValueError(
            f"endpoint {endpoint.name!r} has no autoscale_provider"
        )
    if not endpoint.autoscale_instance_id:
        raise ValueError(
            f"endpoint {endpoint.name!r} has no autoscale_instance_id"
        )


def _config_without_endpoint_slot(
    payload: Mapping[str, Any], endpoint_name: str,
) -> tuple[Dict[str, Any], bool]:
    """Return a config payload without ``endpoint_name``'s managed slot."""
    updated = dict(payload)
    slots = payload.get("endpoints")
    if not isinstance(slots, list):
        return updated, False
    kept = [
        slot for slot in slots
        if not isinstance(slot, Mapping) or slot.get("name") != endpoint_name
    ]
    updated["endpoints"] = kept
    return updated, len(kept) != len(slots)


async def add_endpoint_command(
    *,
    slot_name: str,
    dry_run: bool = False,
    yes: bool = False,
) -> EndpointAddResult:
    await init_client()
    try:
        config_dao = SystemConfigDAO()
        config = await load_config(config_dao)
        if dry_run:
            config = replace(config, dry_run=True)

        slot = _find_slot(config, slot_name)
        if slot is None:
            raise click.ClickException(
                f"autoscaler config has no endpoint slot {slot_name!r}"
            )

        endpoints_dao = InferenceEndpointsDAO()
        existing = await endpoints_dao.get(slot.name)
        click.echo("GPU endpoint add plan:")
        click.echo(
            "  new slot     : "
            f"{slot.name} ({slot.provider}, purpose={slot.purpose})"
        )
        if existing is None:
            click.echo("  current      : <missing>")
        else:
            click.echo(
                "  current      : "
                f"active={existing.active} "
                f"instance={existing.autoscale_instance_id or '-'}"
            )
        click.echo("  order        : create and activate endpoint")
        if dry_run:
            click.echo("  mode         : dry-run")

        if not dry_run and not yes:
            click.confirm("Proceed with GPU endpoint add?", abort=True)

        kv = SystemConfigKVAdapter(
            config_dao,
            updated_by="cli:gpu-add-endpoint",
        )
        if dry_run:
            autoscaler = _cli_autoscaler(
                endpoints_dao,
                kv,
                validator_hotkey="gpu-add-endpoint",
            )
            try:
                result = await autoscaler.add_endpoint(
                    config,
                    slot_name=slot.name,
                )
            except Exception as e:
                raise click.ClickException(str(e)) from e
            finally:
                await autoscaler.close()
        else:
            try:
                request = await _enqueue_manual_add_request(
                    kv,
                    slot_name=slot.name,
                    updated_by="cli:gpu-add-endpoint",
                    now=int(time.time()),
                )
                click.echo(
                    "  queued       : waiting for gpu-autoscaler daemon "
                    f"(token={request['token']})"
                )
                payload = await _wait_for_manual_endpoint_result(
                    kv,
                    request,
                    timeout_seconds=DEFAULT_MANUAL_REPLACEMENT_TTL_SECONDS,
                )
                if payload.get("status") == "failed":
                    raise RuntimeError(str(payload.get("error") or "add failed"))
                result = _endpoint_add_result_from_payload(payload)
                await _cleanup_manual_endpoint_result(
                    kv,
                    request,
                    operation="add",
                )
            except Exception as e:
                raise click.ClickException(str(e)) from e

        if result.dry_run:
            click.echo("✓ dry-run complete; no provider or database changes made")
        else:
            click.echo(
                "✓ add complete: "
                f"endpoint={result.endpoint_name} "
                f"instance={result.instance_id or '-'}"
            )
        return result
    finally:
        await close_client()


async def replace_endpoint_command(
    *,
    old_endpoint_name: str,
    new_slot_name: Optional[str] = None,
    dry_run: bool = False,
    yes: bool = False,
) -> EndpointReplaceResult:
    await init_client()
    try:
        config_dao = SystemConfigDAO()
        config = await load_config(config_dao)
        if dry_run:
            config = replace(config, dry_run=True)

        endpoints_dao = InferenceEndpointsDAO()
        old_endpoint = await endpoints_dao.get(old_endpoint_name)
        if old_endpoint is None:
            raise click.ClickException(
                f"old endpoint {old_endpoint_name!r} does not exist"
            )
        slot = _find_slot(config, new_slot_name or old_endpoint_name)
        if slot is None:
            raise click.ClickException(
                f"autoscaler config has no endpoint slot "
                f"{new_slot_name or old_endpoint_name!r}"
            )

        same_endpoint = old_endpoint_name == slot.name
        click.echo("GPU endpoint replacement plan:")
        click.echo(f"  old endpoint : {old_endpoint_name}")
        click.echo(
            "  old instance : "
            f"{old_endpoint.autoscale_provider or '-'} / "
            f"{old_endpoint.autoscale_instance_id or '-'}"
        )
        click.echo(
            "  new slot     : "
            f"{slot.name} ({slot.provider}, purpose={slot.purpose})"
        )
        if same_endpoint:
            click.echo("  order        : drain old endpoint, delete old, create new")
            click.echo(
                "  note         : same-slot creates are provider-idempotent by "
                "name, so create-before-delete would adopt the old instance"
            )
        else:
            click.echo(
                "  order        : create new first, then drain and delete old"
            )
        if dry_run:
            click.echo("  mode         : dry-run")

        if not dry_run and not yes:
            click.confirm("Proceed with GPU endpoint replacement?", abort=True)

        config_dao = SystemConfigDAO()
        kv = SystemConfigKVAdapter(
            config_dao,
            updated_by="cli:gpu-replace-endpoint",
        )
        if dry_run:
            autoscaler = _cli_autoscaler(
                endpoints_dao,
                kv,
                validator_hotkey="gpu-replace-endpoint",
            )
            try:
                result = await autoscaler.replace_endpoint(
                    config,
                    old_endpoint_name=old_endpoint_name,
                    new_slot_name=slot.name,
                )
            except Exception as e:
                raise click.ClickException(str(e)) from e
            finally:
                await autoscaler.close()
        else:
            try:
                request = await _enqueue_manual_replacement_request(
                    kv,
                    old_endpoint_name=old_endpoint_name,
                    new_slot_name=slot.name,
                    updated_by="cli:gpu-replace-endpoint",
                    now=int(time.time()),
                )
                click.echo(
                    "  queued       : waiting for gpu-autoscaler daemon "
                    f"(token={request['token']})"
                )
                payload = await _wait_for_manual_endpoint_result(
                    kv,
                    request,
                    timeout_seconds=DEFAULT_MANUAL_REPLACEMENT_TTL_SECONDS,
                )
                if payload.get("status") == "failed":
                    raise RuntimeError(
                        str(payload.get("error") or "replacement failed")
                    )
                result = _endpoint_replace_result_from_payload(payload)
                await _cleanup_manual_endpoint_result(
                    kv,
                    request,
                    operation="replacement",
                )
            except Exception as e:
                raise click.ClickException(str(e)) from e

        if result.dry_run:
            click.echo("✓ dry-run complete; no provider or database changes made")
        else:
            click.echo(
                "✓ replacement complete: "
                f"new_endpoint={result.new_endpoint_name} "
                f"new_instance={result.new_instance_id or '-'} "
                f"old_deleted={result.old_deleted}"
            )
        return result
    finally:
        await close_client()


async def remove_endpoint_command(
    *,
    endpoint_name: str,
    keep_slot: bool = False,
    dry_run: bool = False,
    yes: bool = False,
) -> EndpointRemoveResult:
    """Queue removal so provider and tunnel mutations stay daemon-owned."""
    await init_client()
    try:
        config_dao = SystemConfigDAO()
        config_item = await config_dao.get_param(CONFIG_KEY)
        payload = (
            config_item.get("param_value")
            if isinstance(config_item, Mapping)
            else None
        )
        if not isinstance(payload, Mapping) or not payload:
            raise click.ClickException("gpu_autoscaler config is missing")
        config = GPUAutoscalerConfig.from_mapping(payload)
        if dry_run:
            config = replace(config, dry_run=True)

        endpoints_dao = InferenceEndpointsDAO()
        endpoint = await endpoints_dao.get(endpoint_name)
        if endpoint is None:
            raise click.ClickException(
                f"endpoint {endpoint_name!r} does not exist"
            )
        try:
            _validate_autoscaled_endpoint(endpoint)
        except ValueError as e:
            raise click.ClickException(str(e)) from e

        _, slot_present = _config_without_endpoint_slot(
            payload, endpoint_name,
        )
        click.echo("GPU endpoint removal plan:")
        click.echo(f"  endpoint      : {endpoint_name}")
        click.echo(
            "  instance      : "
            f"{endpoint.autoscale_provider} / {endpoint.autoscale_instance_id}"
        )
        click.echo("  action        : drain, delete rental, deactivate endpoint")
        if keep_slot:
            click.echo("  config slot   : keep (autoscaler may rent it again)")
        elif slot_present:
            click.echo("  config slot   : remove")
        else:
            click.echo("  config slot   : already absent")
        if dry_run:
            click.echo("  mode          : dry-run")

        if not dry_run and not yes:
            click.confirm("Proceed with GPU endpoint removal?", abort=True)

        kv = SystemConfigKVAdapter(
            config_dao,
            updated_by="cli:gpu-remove-endpoint",
        )
        if dry_run:
            autoscaler = _cli_autoscaler(
                endpoints_dao,
                kv,
                validator_hotkey="gpu-remove-endpoint",
            )
            try:
                result = await autoscaler.remove_endpoint(
                    config,
                    endpoint_name=endpoint_name,
                    keep_slot=keep_slot,
                )
            except Exception as e:
                raise click.ClickException(str(e)) from e
            finally:
                await autoscaler.close()
        else:
            try:
                request = await _enqueue_manual_remove_request(
                    kv,
                    endpoint_name=endpoint_name,
                    keep_slot=keep_slot,
                    updated_by="cli:gpu-remove-endpoint",
                    now=int(time.time()),
                )
                click.echo(
                    "  queued        : waiting for gpu-autoscaler daemon "
                    f"(token={request['token']})"
                )
                result_payload = await _wait_for_manual_endpoint_result(
                    kv,
                    request,
                    timeout_seconds=DEFAULT_MANUAL_REPLACEMENT_TTL_SECONDS,
                )
                if result_payload.get("status") == "failed":
                    raise RuntimeError(
                        str(result_payload.get("error") or "removal failed")
                    )
                result = _endpoint_remove_result_from_payload(result_payload)
                await _cleanup_manual_endpoint_result(
                    kv,
                    request,
                    operation="removal",
                )
            except Exception as e:
                raise click.ClickException(str(e)) from e

        if result.dry_run:
            click.echo("dry-run complete; no provider or database changes made")
        else:
            click.echo(
                "removal complete: "
                f"endpoint={result.endpoint_name} "
                f"instance={result.instance_id or '-'} "
                f"slot_removed={result.slot_removed}"
            )
        return result
    finally:
        await close_client()


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
        await autoscaler.close()
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


def _safe_token(value: Any, *, default: str) -> str:
    text = str(value or "").strip().lower()
    safe = "".join(c if c.isalnum() or c == "-" else "-" for c in text)
    safe = "-".join(part for part in safe.split("-") if part)
    return safe or default


if __name__ == "__main__":
    main()
