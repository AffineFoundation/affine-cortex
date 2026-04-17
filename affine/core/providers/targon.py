"""Targon inference provider.

Reads live deployment metadata from the `targon_deployments` table (written by
the `targon_deployer` service). The router consults this synchronously on every
fetch, so we never call the Targon SDK from inside the API request path.
"""

from typing import Any, Dict, Optional

from affine.core.providers.base import BaseProvider, ProviderInstanceInfo
from affine.core.providers.targon_client import TargonClient, get_targon_client
from affine.core.setup import logger


class TargonProvider(BaseProvider):
    name = "targon"
    public_display_url = "https://api.targon.com/v1"

    def __init__(self, targon_dao=None, client: Optional[TargonClient] = None):
        # Lazy import: DAO module doesn't exist until PR-2 runs migrations
        from affine.database.dao.targon_deployments import TargonDeploymentsDAO

        self.dao = targon_dao or TargonDeploymentsDAO()
        self.client = client or get_targon_client()

    async def get_instance_info(self, miner_record: Dict[str, Any]) -> ProviderInstanceInfo:
        hotkey = miner_record.get("hotkey")
        revision = miner_record.get("revision")

        if not hotkey or not revision:
            return ProviderInstanceInfo(
                running_instances=0, healthy=False, base_url=None,
                model_identifier=miner_record.get("model", ""), raw={},
            )

        # Any DAO error (table missing on a fresh env, transient DynamoDB glitch)
        # should degrade to "no Targon capacity" so the router transparently
        # falls back to Chutes instead of 500ing the whole fetch_task path.
        try:
            deployment = await self.dao.get_by_hotkey_revision(hotkey, revision)
        except Exception as e:
            logger.warning(f"TargonProvider DAO error for {hotkey[:12]}...: {e}")
            deployment = None
        if not deployment:
            return ProviderInstanceInfo(
                running_instances=0, healthy=False, base_url=None,
                model_identifier=miner_record.get("model", ""), raw={},
            )

        status = deployment.get("status", "")
        if status != "active":
            return ProviderInstanceInfo(
                running_instances=0, healthy=False,
                base_url=deployment.get("base_url"),
                model_identifier=deployment.get("model_hf_repo") or miner_record.get("model", ""),
                raw=deployment,
            )

        return ProviderInstanceInfo(
            running_instances=int(deployment.get("instance_count", 0) or 0),
            healthy=True,
            base_url=deployment.get("base_url"),
            model_identifier=deployment.get("model_hf_repo") or miner_record.get("model", ""),
            raw=deployment,
        )

