"""Chutes inference provider.

Pre-Targon, fetch_task assigned a chute task as long as `chute_slug` was
present — it never inspected hot/cold or any live Chutes state. Match
that exact behavior so the Targon refactor stays a strict superset:
chute_slug existence is the routing signal, no live API call (which
would need CHUTES_API_KEY in the API container) and no hot/cold filter
(which would change pre-Targon semantics by dropping cold-Chutes tasks).
"""

from typing import Any, Dict

from affine.core.providers.base import BaseProvider, ProviderInstanceInfo


class ChutesProvider(BaseProvider):
    name = "chutes"

    async def get_instance_info(self, miner_record: Dict[str, Any]) -> ProviderInstanceInfo:
        chute_slug = miner_record.get("chute_slug")
        base_url = f"https://{chute_slug}.chutes.ai/v1" if chute_slug else None
        return ProviderInstanceInfo(
            running_instances=1 if chute_slug else 0,
            healthy=bool(chute_slug),
            base_url=base_url,
            model_identifier=miner_record.get("model", ""),
            raw={},
        )
