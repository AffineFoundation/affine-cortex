"""Provider router.

Decides which provider (Chutes / Targon / ...) should serve a given miner for
a given task. Called from `TaskPoolManager.fetch_task` so the decision reflects
the freshest capacity snapshot we have. Intentionally provider-count agnostic:
adding a third provider later is a new `BaseProvider` plus one elif.

Routing policy:
    For every (miner, env) pair, look up live capacity on both providers and
    split traffic in proportion to their running instance counts. If one side
    is empty, all traffic goes to the other. If both are empty, return None
    so the caller can release the task back to pending.

    Env gating: ``TARGON_ACCELERATED_ENVS`` (default ``*``) limits which envs
    are eligible for Targon. When set to a comma-separated list, tasks whose
    env isn't in that list bypass Targon entirely and route to Chutes only —
    operators use this to pin a fixed Targon GPU pool to high-cost envs (e.g.
    SWE-bench) without wasting capacity on cheap ones.

    There is no champion-only restriction — an operator deploying a Targon
    workload for any miner opts that miner into the split. This matters for
    miners whose Chutes is cold but whose Targon is live: without the
    deployment they'd be skipped entirely; with it they keep sampling.
"""

import asyncio
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set, Tuple

from affine.core.providers.base import BaseProvider, ProviderInstanceInfo


def _parse_accelerated_envs() -> Optional[Set[str]]:
    """Parse ``TARGON_ACCELERATED_ENVS`` into a whitelist (or None = all).

    Sentinels for "all envs eligible": unset, empty, ``*``, or ``all``.
    Otherwise: comma-separated list of canonical env names. Reading at
    import time is fine because this is a deployment-time toggle, not a
    request-time knob.
    """
    raw = os.getenv("TARGON_ACCELERATED_ENVS", "*").strip()
    if not raw or raw in ("*", "all", "ALL"):
        return None
    parts = {p.strip() for p in raw.split(",") if p.strip()}
    return parts or None


TARGON_ACCELERATED_ENVS: Optional[Set[str]] = _parse_accelerated_envs()


@dataclass(frozen=True)
class RouteDecision:
    """Outcome of a single per-task routing decision."""
    provider: str
    base_url: str
    model_identifier: str
    public_base_url: Optional[str] = None  # None -> persist base_url unchanged


class ProviderRouter:
    def __init__(
        self,
        chutes: BaseProvider,
        targon: BaseProvider,
        instance_cache_ttl: int = 60,
    ):
        self.chutes = chutes
        self.targon = targon

        # (provider_name, hotkey, revision) -> (fetched_at_monotonic, info)
        self._instance_cache_ttl = instance_cache_ttl
        self._instance_cache: Dict[Tuple[str, str, str], Tuple[float, ProviderInstanceInfo]] = {}
        self._instance_cache_lock = asyncio.Lock()

    async def _instance_info(
        self, provider: BaseProvider, miner_record: Dict[str, Any]
    ) -> ProviderInstanceInfo:
        """Provider.get_instance_info() with a (hotkey, revision) TTL cache.

        Upstream APIs (Chutes / Targon) are called at most once per TTL per
        miner-revision; concurrent fetch_task calls coalesce on the lock.
        """
        key = (provider.name, miner_record.get("hotkey", ""), miner_record.get("revision", ""))
        now = time.monotonic()
        async with self._instance_cache_lock:
            hit = self._instance_cache.get(key)
            if hit and (now - hit[0]) < self._instance_cache_ttl:
                return hit[1]
        info = await provider.get_instance_info(miner_record)
        async with self._instance_cache_lock:
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
        # Env gating: if a whitelist is configured and this env isn't on it,
        # skip Targon entirely. We still consult Chutes — non-accelerated
        # envs should keep sampling, just not eat Targon capacity.
        targon_eligible = (
            TARGON_ACCELERATED_ENVS is None
            or (env is not None and env in TARGON_ACCELERATED_ENVS)
        )

        chute_info = await self._instance_info(self.chutes, miner_record)
        c = int(chute_info.get("running_instances", 0) or 0)

        if not targon_eligible:
            if c <= 0:
                return None
            return self._decide(self.chutes, chute_info)

        targon_info = await self._instance_info(self.targon, miner_record)
        t = int(targon_info.get("running_instances", 0) or 0)

        if c <= 0 and t <= 0:
            return None
        if c <= 0:
            return self._decide(self.targon, targon_info)
        if t <= 0:
            return self._decide(self.chutes, chute_info)
        if random.random() < (t / (c + t)):
            return self._decide(self.targon, targon_info)
        return self._decide(self.chutes, chute_info)


def build_default_router() -> ProviderRouter:
    """Wire up the default Chutes+Targon router.

    Targon provider is stubbed out when TARGON_API_KEY is absent so local
    dev environments without Targon credentials don't pay for empty queries.
    """
    from affine.core.providers.chutes import ChutesProvider
    from affine.core.providers.targon import TargonProvider
    from affine.core.providers.base import BaseProvider

    chutes = ChutesProvider()

    if os.getenv("TARGON_API_KEY"):
        targon: BaseProvider = TargonProvider()
    else:
        class _NoopTargon(BaseProvider):
            name = "targon"
            async def get_instance_info(self, miner_record):
                return {"running_instances": 0, "healthy": False,
                        "base_url": None, "model_identifier": "", "raw": {}}
        targon = _NoopTargon()

    return ProviderRouter(chutes=chutes, targon=targon)
