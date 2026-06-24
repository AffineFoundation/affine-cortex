"""
Inference endpoints DAO (Stage AI).

Provider-agnostic registry for inference endpoints. Operators populate
rows here (one per host/provider) and the scheduler reads them at
startup to build the right provider config — no more env-var-only IP
configuration.

Schema (PK only, no SK):
    pk: ENDPOINT#{name}     unique label per endpoint

Non-key attributes are sparse and provider-specific; see ``Endpoint``
dataclass below for the typed view.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from affine.database.base_dao import BaseDAO
from affine.database.schema import get_table_name


@dataclass
class Endpoint:
    """Typed view of one ``inference_endpoints`` row."""
    name: str
    kind: str                              # "ssh" | "targon"
    active: bool = True
    public_inference_url: Optional[str] = None
    notes: Optional[str] = None

    # Static endpoint purpose. ``"scoring"`` (default) means the
    # scheduler may pick this endpoint to host the champion / a
    # challenger; ``"anticopy"`` means the endpoint is dedicated to the
    # CEAC forward worker and the scheduler must skip it. Older rows
    # without this attribute deserialize to None — treat None as
    # ``"scoring"`` for backwards compat.
    role: Optional[str] = None

    # Runtime assignment. Provider-agnostic: one endpoint/machine may be
    # serving one miner model at a time. Scheduler writes these fields after
    # deploy/adopt and clears them on teardown.
    assigned_uid: Optional[int] = None
    assigned_hotkey: Optional[str] = None
    assigned_model: Optional[str] = None
    assigned_revision: Optional[str] = None
    deployment_id: Optional[str] = None
    base_url: Optional[str] = None
    assignment_role: Optional[str] = None
    assigned_at: int = 0

    # ssh-kind extras
    ssh_url: Optional[str] = None
    ssh_key_path: Optional[str] = None
    sglang_port: int = 10001
    sglang_dp: int = 8
    sglang_image: str = "lmsysorg/sglang:latest"
    sglang_cache_dir: str = "/data"
    sglang_context_len: int = 65536
    sglang_mem_fraction: float = 0.85
    sglang_chunked_prefill: int = 4096
    sglang_tool_call_parser: str = "qwen"
    sglang_docker_args: List[str] = field(default_factory=list)
    ready_timeout_sec: int = 1800
    poll_interval_sec: float = 15.0

    # targon-kind extras
    targon_api_url: Optional[str] = None

    # Autoscaler metadata. These fields describe the outer GPU instance
    # lifecycle (Lium/Targon/etc.) for operator-managed SSH endpoints; they
    # are intentionally separate from scheduler assignment fields.
    autoscale_managed: bool = False
    autoscale_provider: Optional[str] = None
    autoscale_instance_id: Optional[str] = None
    autoscale_created_at: int = 0
    autoscale_updated_at: int = 0
    autoscale_lease_expires_at: int = 0

    # Endpoint lifecycle identity. ``updated_at`` also changes for runtime
    # assignment churn, so scheduler recovery must not use it as "host became
    # active". ``generation`` and ``activated_at`` advance only when the
    # endpoint is enabled or its deployment-relevant config changes.
    generation: int = 0
    activated_at: int = 0

    updated_at: int = 0
    updated_by: str = ""

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "Endpoint":
        # ``pk`` round-trips as ``ENDPOINT#<name>`` — strip the prefix.
        pk = str(row.get("pk", ""))
        name = pk[len("ENDPOINT#"):] if pk.startswith("ENDPOINT#") else pk
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        kw = {k: v for k, v in row.items() if k in fields and k != "name"}
        return cls(name=name, **kw)


_ENDPOINT_RUNTIME_IDENTITY_FIELDS = (
    "kind",
    "public_inference_url",
    "role",
    "ssh_url",
    "ssh_key_path",
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
    "targon_api_url",
)


_ASSIGNMENT_FIELDS = (
    "assigned_uid",
    "assigned_hotkey",
    "assigned_model",
    "assigned_revision",
    "deployment_id",
    "base_url",
    "assignment_role",
    "assigned_at",
)


class InferenceEndpointsDAO(BaseDAO):
    """CRUD over the ``inference_endpoints`` table."""

    def __init__(self):
        self.table_name = get_table_name("inference_endpoints")
        super().__init__()

    @staticmethod
    def _make_pk(name: str) -> str:
        return f"ENDPOINT#{name}"

    async def upsert(self, endpoint: Endpoint, *, updated_by: str = "operator") -> Dict[str, Any]:
        """Insert or overwrite the named endpoint row."""
        now = int(time.time())
        previous = await self.get(endpoint.name)
        generation = int((previous.generation if previous else 0) or 0)
        activated_at = int((previous.activated_at if previous else 0) or 0)
        if self._activation_bump_required(previous, endpoint):
            generation += 1
            activated_at = now

        payload = asdict(endpoint)
        payload.pop("name", None)
        payload["pk"] = self._make_pk(endpoint.name)
        payload["generation"] = generation
        payload["activated_at"] = activated_at
        payload["updated_at"] = now
        payload["updated_by"] = updated_by
        return await self.put(payload)

    @staticmethod
    def _activation_bump_required(
        previous: Optional[Endpoint],
        endpoint: Endpoint,
    ) -> bool:
        if not endpoint.active:
            return False
        if previous is None or not previous.active:
            return True
        if not previous.generation or not previous.activated_at:
            return True
        return any(
            getattr(previous, field) != getattr(endpoint, field)
            for field in _ENDPOINT_RUNTIME_IDENTITY_FIELDS
        )

    async def get(self, name: str) -> Optional[Endpoint]:
        row = await super().get(self._make_pk(name))
        if row is None:
            return None
        return Endpoint.from_row(row)

    async def delete(self, name: str) -> None:
        from affine.database.client import get_client
        client = get_client()
        await client.delete_item(
            TableName=self.table_name,
            Key={"pk": {"S": self._make_pk(name)}},
        )

    async def list_all(self) -> List[Endpoint]:
        from affine.database.client import get_client
        client = get_client()
        resp = await client.scan(TableName=self.table_name)
        items = [self._deserialize(item) for item in resp.get("Items", [])]
        return [Endpoint.from_row(it) for it in items]

    async def list_active(self, kind: Optional[str] = None) -> List[Endpoint]:
        """Convenience: ``active=True`` rows, optionally filtered by kind."""
        out = []
        for ep in await self.list_all():
            if not ep.active:
                continue
            if kind is not None and ep.kind != kind:
                continue
            out.append(ep)
        return out

    async def set_assignment(
        self,
        name: str,
        *,
        uid: int,
        hotkey: str,
        model: str,
        revision: str,
        deployment_id: str,
        base_url: str,
        role: str,
        updated_by: str = "scheduler",
    ) -> None:
        from affine.database.client import get_client
        client = get_client()
        await client.update_item(
            TableName=self.table_name,
            Key={"pk": {"S": self._make_pk(name)}},
            UpdateExpression=(
                "SET assigned_uid = :uid, assigned_hotkey = :hotkey, "
                "assigned_model = :model, assigned_revision = :revision, "
                "deployment_id = :deployment_id, base_url = :base_url, "
                "assignment_role = :role, assigned_at = :now, "
                "updated_at = :now, updated_by = :updated_by"
            ),
            ExpressionAttributeValues={
                ":uid": {"N": str(uid)},
                ":hotkey": {"S": hotkey},
                ":model": {"S": model},
                ":revision": {"S": revision},
                ":deployment_id": {"S": deployment_id},
                ":base_url": {"S": base_url},
                ":role": {"S": role},
                ":now": {"N": str(int(time.time()))},
                ":updated_by": {"S": updated_by},
            },
        )

    async def clear_assignment(
        self,
        name: str,
        *,
        updated_by: str = "scheduler",
    ) -> None:
        from affine.database.client import get_client
        client = get_client()
        now = int(time.time())
        await client.update_item(
            TableName=self.table_name,
            Key={"pk": {"S": self._make_pk(name)}},
            UpdateExpression=(
                "SET updated_at = :now, updated_by = :updated_by "
                "REMOVE assigned_uid, assigned_hotkey, assigned_model, "
                "assigned_revision, deployment_id, base_url, "
                "assignment_role, assigned_at"
            ),
            ExpressionAttributeValues={
                ":now": {"N": str(now)},
                ":updated_by": {"S": updated_by},
            },
        )

    async def activate_autoscaled_endpoint(
        self,
        endpoint: Endpoint,
        *,
        updated_by: str = "gpu-autoscaler",
    ) -> None:
        """Activate/update an autoscaled endpoint without touching runtime
        assignment fields the scheduler owns."""
        now = int(time.time())
        previous = await self.get(endpoint.name)
        generation = int((previous.generation if previous else 0) or 0)
        activated_at = int((previous.activated_at if previous else 0) or 0)
        if self._activation_bump_required(previous, endpoint):
            generation += 1
            activated_at = now

        payload = asdict(endpoint)
        payload.pop("name", None)
        for field in _ASSIGNMENT_FIELDS:
            payload.pop(field, None)
        payload.update({
            "active": True,
            "generation": generation,
            "activated_at": activated_at,
            "updated_at": now,
            "updated_by": updated_by,
        })
        await self._update_endpoint_fields(endpoint.name, set_values=payload)

    async def update_autoscale_lease(
        self,
        name: str,
        *,
        instance_id: str,
        lease_expires_at: int,
        updated_by: str = "gpu-autoscaler",
    ) -> None:
        now = int(time.time())
        await self._update_endpoint_fields(
            name,
            set_values={
                "autoscale_updated_at": now,
                "autoscale_lease_expires_at": lease_expires_at,
                "updated_at": now,
                "updated_by": updated_by,
            },
            condition_expression="#cond_instance = :cond_instance",
            condition_names={"#cond_instance": "autoscale_instance_id"},
            condition_values={":cond_instance": instance_id},
        )

    async def deactivate_autoscaled_endpoint(
        self,
        name: str,
        *,
        instance_id: str,
        updated_by: str = "gpu-autoscaler",
    ) -> None:
        now = int(time.time())
        await self._update_endpoint_fields(
            name,
            set_values={
                "active": False,
                "autoscale_updated_at": now,
                "autoscale_lease_expires_at": 0,
                "updated_at": now,
                "updated_by": updated_by,
            },
            remove_fields=(
                "ssh_url",
                "public_inference_url",
                "autoscale_instance_id",
                *_ASSIGNMENT_FIELDS,
            ),
            condition_expression="#cond_instance = :cond_instance",
            condition_names={"#cond_instance": "autoscale_instance_id"},
            condition_values={":cond_instance": instance_id},
        )

    async def _update_endpoint_fields(
        self,
        name: str,
        *,
        set_values: Dict[str, Any],
        remove_fields=(),
        condition_expression: Optional[str] = None,
        condition_names: Optional[Dict[str, str]] = None,
        condition_values: Optional[Dict[str, Any]] = None,
    ) -> None:
        from affine.database.client import get_client

        names: Dict[str, str] = {}
        values: Dict[str, Dict[str, Any]] = {}
        set_parts = []
        for idx, (field, value) in enumerate(set_values.items()):
            name_key = f"#s{idx}"
            value_key = f":v{idx}"
            names[name_key] = field
            values[value_key] = self._serialize({"value": value})["value"]
            set_parts.append(f"{name_key} = {value_key}")

        remove_parts = []
        for idx, field in enumerate(remove_fields):
            name_key = f"#r{idx}"
            names[name_key] = field
            remove_parts.append(name_key)

        names.update(condition_names or {})
        for key, value in (condition_values or {}).items():
            values[key] = self._serialize({"value": value})["value"]

        update_parts = []
        if set_parts:
            update_parts.append("SET " + ", ".join(set_parts))
        if remove_parts:
            update_parts.append("REMOVE " + ", ".join(remove_parts))

        params = {
            "TableName": self.table_name,
            "Key": {"pk": {"S": self._make_pk(name)}},
            "UpdateExpression": " ".join(update_parts),
            "ExpressionAttributeNames": names,
            "ExpressionAttributeValues": values,
        }
        if condition_expression:
            params["ConditionExpression"] = condition_expression

        client = get_client()
        await client.update_item(**params)
