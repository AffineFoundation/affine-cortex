"""
Executor worker metrics.

Minimal counters reported to the manager process via the stats queue.
The old fetch_count / total_fetch_time / pending_tasks columns are gone —
no HTTP fetch loop in the DB-poll design.
"""

from dataclasses import dataclass, asdict
import time
from typing import Any, Dict, Optional


@dataclass
class WorkerMetrics:
    worker_id: int
    env: str
    running: bool = True
    tasks_succeeded: int = 0
    tasks_failed: int = 0
    # Integrity-invalid rows are completed attempts, but are neither model
    # failures nor benchmark scores. Keep them out of ``tasks_failed`` so
    # operational telemetry cannot silently turn a harness/template incident
    # into model attribution.
    tasks_invalid: int = 0
    tasks_in_flight: int = 0
    # Results dropped because the deployment we dispatched against is no
    # longer the current one for this miner — the model was swapped (or
    # the miner exited the subject role) mid-evaluate, so persisting
    # would attribute new-model output to the old miner.
    tasks_dropped_drift: int = 0
    total_execution_ms: int = 0
    last_task_at: Optional[float] = None

    def record_completion(self, *, success: bool, latency_ms: int) -> None:
        self.last_task_at = time.time()
        self.total_execution_ms += int(latency_ms)
        if success:
            self.tasks_succeeded += 1
        else:
            self.tasks_failed += 1

    def record_invalid(self, *, latency_ms: int) -> None:
        self.last_task_at = time.time()
        self.total_execution_ms += int(latency_ms)
        self.tasks_invalid += 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
