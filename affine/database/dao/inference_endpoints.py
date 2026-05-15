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
from dataclasses import asdict, dataclass
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
    sglang_port: int = 30000
    sglang_dp: int = 8
    sglang_image: str = "lmsysorg/sglang:latest"
    sglang_cache_dir: str = "/data"
    sglang_context_len: int = 65536
    sglang_mem_fraction: float = 0.85
    sglang_chunked_prefill: int = 4096
    sglang_tool_call_parser: str = "qwen"
    ready_timeout_sec: int = 1800
    poll_interval_sec: float = 15.0

    # targon-kind extras
    targon_api_url: Optional[str] = None

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
        payload = asdict(endpoint)
        payload.pop("name", None)
        payload["pk"] = self._make_pk(endpoint.name)
        payload["updated_at"] = int(time.time())
        payload["updated_by"] = updated_by
        return await self.put(payload)

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
