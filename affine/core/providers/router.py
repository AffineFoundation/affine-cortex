"""Provider router.

Per-task decision: which inference provider serves a given miner. An env
opts into the accelerated provider by setting ``accelerated: true`` in
its system_config row under ``environments``. Envs without the flag —
or with it set to false — stay on the standard provider.

Within an accelerated env: take the accelerated provider when it's
healthy (instances > 0 AND no recent probe failures), fall back to
standard on degradation, and last-resort to the accelerated provider
again if standard has no capacity either — a slow response beats a
released task.
"""

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set, Tuple

from affine.core.providers.base import BaseProvider, ProviderInstanceInfo
from affine.core.setup import logger


ENVIRONMENTS_PARAM = "environments"
ACCELERATED_ENVS_TTL = int(os.getenv("ACCELERATED_ENVS_TTL_SEC", "60"))


def _extract_accelerated(envs_config: Any) -> Set[str]:
    """Pull the set of envs flagged ``accelerated: true``.

    The system_config ``environments`` row is a dict of env_name → config;
    each per-env config may carry ``accelerated: true`` to opt in. Names
    are lowercased to match the canonical task env in fetch_task.
    """
    if not isinstance(envs_config, dict):
        return set()
    return {
        name.strip().lower()
        for name, cfg in envs_config.items()
        if isinstance(cfg, dict) and bool(cfg.get("accelerated", False)) and name
    }


@dataclass(frozen=True)
class RouteDecision:
    provider: str
    base_url: str
    model_identifier: str
    public_base_url: Optional[str] = None  # None → persist base_url unchanged


class ProviderRouter:
    def __init__(
        self,
        chutes: BaseProvider,
        targon: BaseProvider,
        instance_cache_ttl: int = 60,
        config_ttl: int = ACCELERATED_ENVS_TTL,
    ):
        self.chutes = chutes
        self.targon = targon

        self._instance_cache_ttl = instance_cache_ttl
        self._instance_cache: Dict[Tuple[str, str, str], Tuple[float, ProviderInstanceInfo]] = {}

        self._config_ttl = config_ttl
        self._config_at: float = 0.0
        self._accelerated: Set[str] = set()

    async def _refresh_config(self) -> None:
        """Reload the accelerated env set from system_config if cache expired.

        Any error keeps the previously cached set so a transient DAO blip
        doesn't accidentally switch acceleration on or off.
        """
        if (time.monotonic() - self._config_at) < self._config_ttl:
            return
        try:
            from affine.database.dao.system_config import SystemConfigDAO
            envs_config = await SystemConfigDAO().get_param_value(
                ENVIRONMENTS_PARAM, default={},
            )
            self._accelerated = _extract_accelerated(envs_config)
        except Exception as e:
            logger.warning(
                f"router: refresh of {ENVIRONMENTS_PARAM} failed, "
                f"keeping previous value: {e}"
            )
        self._config_at = time.monotonic()

    def _is_eligible(self, env: Optional[str]) -> bool:
        return bool(env) and env.lower() in self._accelerated

    async def _instance_info(
        self, provider: BaseProvider, miner_record: Dict[str, Any]
    ) -> ProviderInstanceInfo:
        """Per-miner provider state with a (hotkey, revision) TTL cache."""
        key = (provider.name, miner_record.get("hotkey", ""), miner_record.get("revision", ""))
        now = time.monotonic()
        hit = self._instance_cache.get(key)
        if hit and (now - hit[0]) < self._instance_cache_ttl:
            return hit[1]
        info = await provider.get_instance_info(miner_record)
        self._instance_cache[key] = (time.monotonic(), info)
        return info

    def _decide(self, provider: BaseProvider, info: Dict[str, Any]) -> RouteDecision:
        return RouteDecision(
            provider=provider.name,
            base_url=info.get("base_url"),
            model_identifier=info.get("model_identifier", ""),
            public_base_url=provider.public_display_url,
        )

    async def select(
        self,
        miner_record: Dict[str, Any],
        env: Optional[str] = None,
    ) -> Optional[RouteDecision]:
        await self._refresh_config()

        chute_info = await self._instance_info(self.chutes, miner_record)
        c = int(chute_info.get("running_instances", 0) or 0)

        if not self._is_eligible(env):
            return self._decide(self.chutes, chute_info) if c > 0 else None

        targon_info = await self._instance_info(self.targon, miner_record)
        t = int(targon_info.get("running_instances", 0) or 0)
        # consecutive_failures comes from the deployment row the provider
        # already returned in `raw` — no extra DAO query.
        targon_failures = int(
            (targon_info.get("raw") or {}).get("consecutive_failures", 0) or 0
        )
        targon_healthy = t > 0 and targon_failures == 0

        if targon_healthy:
            return self._decide(self.targon, targon_info)
        if c > 0:
            return self._decide(self.chutes, chute_info)
        if t > 0:
            return self._decide(self.targon, targon_info)
        return None


def build_default_router() -> ProviderRouter:
    from affine.core.providers.chutes import ChutesProvider
    from affine.core.providers.targon import TargonProvider

    return ProviderRouter(chutes=ChutesProvider(), targon=TargonProvider())
