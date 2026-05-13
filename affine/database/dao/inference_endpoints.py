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

    # ssh-kind extras
    ssh_url: Optional[str] = None
    ssh_key_path: Optional[str] = None
    sglang_port: int = 30000
    sglang_dp: int = 8
    sglang_image: str = "lmsysorg/sglang:latest"
    sglang_cache_dir: str = "/data"

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
        from affine.database.base_dao import BaseDAO as _Base  # noqa
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
