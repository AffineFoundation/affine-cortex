"""Targon Deployments DAO.

Single source of truth for Targon deployment metadata the provider router and
the targon_deployer service share. Keep this row shape stable — consumers read
directly from DynamoDB items.

SECURITY:
    Rows contain `base_url` that points at private Targon workload endpoints
    (https://wrk-*.serverless.targon.com). These URLs are NOT to be exposed
    via the public Affine API. If you add a new endpoint or response model
    that returns data from this table, route it through
    `BaseProvider.public_display_url` or explicitly strip `base_url` before
    serialisation. (The provider router already redacts at sample-write
    time — see affine/core/environments.py::SDKEnvironment._build_result.)

Schema:
    PK: DEPLOYMENT#{deployment_id}
    SK: META
    GSI1 (hotkey-revision-index): gsi1_pk=HOTKEY#{hotkey}, gsi1_sk=REV#{revision}
    GSI2 (status-index):           gsi2_pk=STATUS#{status}
"""

import time
from typing import Any, Dict, List, Optional

from affine.core.setup import logger
from affine.database.base_dao import BaseDAO
from affine.database.client import get_client
from affine.database.schema import get_table_name


class TargonDeploymentsDAO(BaseDAO):
    def __init__(self):
        self.table_name = get_table_name("targon_deployments")
        super().__init__()

    @staticmethod
    def _make_pk(deployment_id: str) -> str:
        return f"DEPLOYMENT#{deployment_id}"

    @staticmethod
    def _make_sk() -> str:
        return "META"

    @staticmethod
    def _make_gsi1(hotkey: str, revision: str) -> Dict[str, str]:
        return {"gsi1_pk": f"HOTKEY#{hotkey}", "gsi1_sk": f"REV#{revision}"}

    @staticmethod
    def _make_gsi2(status: str) -> str:
        return f"STATUS#{status}"

    async def upsert_deployment(
        self,
        *,
        deployment_id: str,
        hotkey: str,
        revision: str,
        model_hf_repo: str,
        image: Optional[str] = None,
        base_url: Optional[str] = None,
        instance_count: int = 0,
        status: str = "deploying",
        mount_path: str = "/data",
        env_vars: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        now = int(time.time())
        existing = await self.get(self._make_pk(deployment_id), self._make_sk())
        created_at = existing.get("created_at", now) if existing else now
        consecutive_failures = existing.get("consecutive_failures", 0) if existing else 0
        next_retry_at = existing.get("next_retry_at", 0) if existing else 0

        item = {
            "pk": self._make_pk(deployment_id),
            "sk": self._make_sk(),
            "deployment_id": deployment_id,
            "hotkey": hotkey,
            "revision": revision,
            "model_hf_repo": model_hf_repo,
            "image": image,
            "base_url": base_url,
            "instance_count": int(instance_count),
            "status": status,
            "mount_path": mount_path,
            "env_vars": env_vars or {},
            "created_at": created_at,
            "updated_at": now,
            "last_health_check_at": now,
            "consecutive_failures": consecutive_failures,
            "next_retry_at": next_retry_at,
            **self._make_gsi1(hotkey, revision),
            "gsi2_pk": self._make_gsi2(status),
        }
        return await self.put(item)

    async def get_deployment(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        return await self.get(self._make_pk(deployment_id), self._make_sk())

    async def get_by_hotkey_revision(
        self, hotkey: str, revision: str
    ) -> Optional[Dict[str, Any]]:
        client = get_client()
        params = {
            "TableName": self.table_name,
            "IndexName": "hotkey-revision-index",
            "KeyConditionExpression": "gsi1_pk = :pk AND gsi1_sk = :sk",
            "ExpressionAttributeValues": {
                ":pk": {"S": f"HOTKEY#{hotkey}"},
                ":sk": {"S": f"REV#{revision}"},
            },
            "Limit": 1,
        }
        response = await client.query(**params)
        items = [self._deserialize(it) for it in response.get("Items", [])]
        return items[0] if items else None

    async def list_by_status(self, status: str) -> List[Dict[str, Any]]:
        client = get_client()
        params = {
            "TableName": self.table_name,
            "IndexName": "status-index",
            "KeyConditionExpression": "gsi2_pk = :pk",
            "ExpressionAttributeValues": {":pk": {"S": self._make_gsi2(status)}},
        }
        response = await client.query(**params)
        return [self._deserialize(it) for it in response.get("Items", [])]

    async def update_health(
        self,
        deployment_id: str,
        *,
        instance_count: int,
        healthy: bool,
        base_url: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        existing = await self.get_deployment(deployment_id)
        if not existing:
            return None
        new_status = "active" if healthy and instance_count > 0 else existing.get("status", "")
        existing["instance_count"] = int(instance_count)
        existing["last_health_check_at"] = int(time.time())
        existing["updated_at"] = int(time.time())
        if base_url:
            existing["base_url"] = base_url
        if healthy and instance_count > 0:
            existing["consecutive_failures"] = 0
            existing["next_retry_at"] = 0
        existing["status"] = new_status
        existing["gsi2_pk"] = self._make_gsi2(new_status)
        await self.put(existing)
        return existing

    async def set_status(
        self,
        deployment_id: str,
        status: str,
        *,
        next_retry_at: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        existing = await self.get_deployment(deployment_id)
        if not existing:
            return None
        existing["status"] = status
        existing["updated_at"] = int(time.time())
        if next_retry_at is not None:
            existing["next_retry_at"] = int(next_retry_at)
        existing["gsi2_pk"] = self._make_gsi2(status)
        await self.put(existing)
        return existing

    async def increment_failure(
        self,
        deployment_id: str,
        *,
        base_backoff: int = 30,
        max_backoff: int = 600,
    ) -> Optional[Dict[str, Any]]:
        existing = await self.get_deployment(deployment_id)
        if not existing:
            return None
        failures = int(existing.get("consecutive_failures", 0)) + 1
        backoff = min(base_backoff * (2 ** (failures - 1)), max_backoff)
        existing["consecutive_failures"] = failures
        existing["next_retry_at"] = int(time.time()) + backoff
        existing["updated_at"] = int(time.time())
        await self.put(existing)
        return existing

    async def mark_deleted(self, deployment_id: str) -> bool:
        """Remove the row entirely (hard delete). Nothing reads the 'deleted'
        state, so keeping tombstones just wastes DDB rows."""
        return await self.delete(self._make_pk(deployment_id), self._make_sk())
