"""Provider abstraction base.

A `BaseProvider` answers two questions the router needs per task:
    1. Is this miner currently servable by me, and how much capacity?
    2. What base_url / model_identifier should inference use?

Deployment lifecycle (create/delete/restart) is NOT part of this interface —
the targon_deployer service and CLI talk to TargonClient directly, because
Chutes has no equivalent lifecycle (miners manage their own chutes) so a
shared interface would be all-NotImplementedError on one side.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TypedDict


class ProviderInstanceInfo(TypedDict, total=False):
    running_instances: int
    healthy: bool
    base_url: Optional[str]
    model_identifier: str
    raw: Dict[str, Any]


class BaseProvider(ABC):
    name: str = "base"

    # Public-safe URL substituted for the real `base_url` when a sample is
    # persisted. None = persist base_url unchanged (Chutes is miner-public
    # by design); overridden by private providers like Targon.
    public_display_url: Optional[str] = None

    @abstractmethod
    async def get_instance_info(self, miner_record: Dict[str, Any]) -> ProviderInstanceInfo:
        """Return live capacity / endpoint for a miner.

        running_instances == 0 means this provider cannot serve the miner
        right now (cold, not deployed, unhealthy). Router treats it as skip.
        """
