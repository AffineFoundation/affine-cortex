"""Chutes inference provider.

Wraps the existing Chutes HTTP client (`affine.utils.api_client`) behind
the BaseProvider interface. Deployment lifecycle is not modelled here —
miners own their Chutes deployments, the validator only reads state.
"""

from typing import Any, Dict, Optional

from affine.core.providers.base import BaseProvider, ProviderInstanceInfo
from affine.core.setup import logger
from affine.utils.api_client import get_chute_info


class ChutesProvider(BaseProvider):
    name = "chutes"

    async def get_instance_info(self, miner_record: Dict[str, Any]) -> ProviderInstanceInfo:
        chute_id = miner_record.get("chute_id")
        chute_slug = miner_record.get("chute_slug")
        model = miner_record.get("model", "")

        base_url = f"https://{chute_slug}.chutes.ai/v1" if chute_slug else None

        if not chute_id:
            return ProviderInstanceInfo(
                running_instances=0,
                healthy=False,
                base_url=base_url,
                model_identifier=model,
                raw={},
            )

        info = await get_chute_info(chute_id)
        if not info:
            return ProviderInstanceInfo(
                running_instances=0,
                healthy=False,
                base_url=base_url,
                model_identifier=model,
                raw={},
            )

        hot = bool(info.get("hot", False))
        instance_count = int(info.get("instance_count", 0) or 0)
        running = instance_count if (hot and instance_count > 0) else (1 if hot else 0)

        return ProviderInstanceInfo(
            running_instances=running,
            healthy=hot,
            base_url=base_url,
            model_identifier=model,
            raw=info,
        )
