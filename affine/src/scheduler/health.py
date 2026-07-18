"""Shared runtime health signals for scheduler deployment recovery."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


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
    canonical_base_url: str = ""
