"""Shared runtime health signals for scheduler deployment recovery."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


TUNNEL_REPAIR_REQUEST_KEY_PREFIX = "gpu_autoscaler_tunnel_repair:"
TUNNEL_REPAIR_REQUEST_TTL_SECONDS = 5 * 60


class DeploymentHealthState(str, Enum):
    HEALTHY = "healthy"
    SUSPECTED = "suspected"
    TRANSPORT_UNHEALTHY = "transport_unhealthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class DeploymentHealthResult:
    state: DeploymentHealthState
    reason: str = ""
    identity: str = ""


def tunnel_repair_request_key(endpoint_name: str) -> str:
    return f"{TUNNEL_REPAIR_REQUEST_KEY_PREFIX}{endpoint_name}"
