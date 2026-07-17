"""Shared behavior-gate lookup helpers.

Executor and scheduler must derive exactly the same deployment fingerprint;
otherwise one service can pass a deployment that the other still sees as
pending.  Keep that small piece of policy in one module.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from affine.src.behavior_guard.models import (
    BehaviorGateConfig,
    VerdictStatus,
    deployment_fingerprint,
)


@dataclass(frozen=True)
class GateSnapshot:
    """Gate state for one exact challenger deployment."""

    status: VerdictStatus
    reason: str
    deployment_fingerprint: str
    row: Optional[Mapping[str, Any]] = None
    admission_deadline_at: Optional[float] = None
    policy_identity: str = ""
    policy_version: str = ""
    admission_hold_seconds: int = 300

    @property
    def passed(self) -> bool:
        return self.status is VerdictStatus.PASSED

    @property
    def failed(self) -> bool:
        return self.status is VerdictStatus.FAILED

    @property
    def admission_expired(self) -> bool:
        """Whether the deployment-level admission hold reached its hard stop."""

        if self.status is VerdictStatus.EXPIRED:
            return True
        if self.status in {VerdictStatus.PASSED, VerdictStatus.FAILED}:
            return False
        deadline = self.admission_deadline_at
        if deadline is None:
            deadline = (self.row or {}).get("admission_deadline_at")
        if isinstance(deadline, bool) or not isinstance(deadline, (int, float)):
            return False
        return float(deadline) <= time.time()


def record_deployment_fingerprint(
    record: Any,
    config: BehaviorGateConfig,
) -> str:
    """Bind a gate verdict to the subject and all serving endpoints."""

    challenger = record.challenger
    deployments = tuple(getattr(record, "deployments", ()) or ())
    fallback = {}
    if not deployments:
        fallback = {
            "deployment_id": getattr(record, "deployment_id", None),
            "base_url": getattr(record, "base_url", None),
        }
    return deployment_fingerprint(
        hotkey=challenger.hotkey,
        revision=challenger.revision,
        policy_version=config.policy_identity,
        deployments=deployments,
        deployment_generation=(
            getattr(record, "deployment_generation", None)
            or getattr(record, "started_at_block", None)
        ),
        **fallback,
    )


async def read_gate_snapshot(
    dao: Any,
    record: Any,
    config: BehaviorGateConfig,
) -> GateSnapshot:
    """Read one strongly-consistent verdict through ``BehaviorGateDAO``."""

    fingerprint = record_deployment_fingerprint(record, config)
    row = await dao.get_verdict(
        record.challenger.hotkey,
        record.challenger.revision,
        config.policy_version,
        fingerprint,
    )
    raw_status = row.get("status") if row else VerdictStatus.PENDING.value
    try:
        status = VerdictStatus(str(raw_status))
    except ValueError:
        status = VerdictStatus.PENDING
    return GateSnapshot(
        status=status,
        reason=str((row or {}).get("reason_code") or "awaiting_preflight"),
        deployment_fingerprint=fingerprint,
        row=row,
        admission_deadline_at=(
            (row or {}).get("admission_deadline_at")
            or getattr(record, "behavior_admission_deadline_at", None)
        ),
        policy_identity=config.policy_identity,
        policy_version=config.policy_version,
        admission_hold_seconds=config.admission_hold_seconds,
    )


def verdict_counts(verdict: Any) -> dict[str, int]:
    """Convert the domain aggregate to the DAO's numeric count map."""

    return {
        "total": int(verdict.total_count),
        "clean": int(verdict.clean_count),
        "quality_failure": int(verdict.quality_failure_count),
        "strikes": int(verdict.strike_count),
        "infra_failure": int(verdict.infra_failure_count),
        "unknown": int(verdict.unknown_count),
        "admissible": int(verdict.admissible_completion_count),
    }
