"""Independent behavior preflight coordinator for challenger deployments.

This runs in the executor *manager* process, outside every benchmark
environment.  It reuses the already-deployed challenger inference endpoint,
but owns its own bounded HTTP client and in-memory nonce/action sandbox.  A
failed or missing verdict therefore prevents the expensive SWE/Terminal
workers from fanning out without requiring another GPU.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Any, Callable, Iterable, Optional

from affine.core.setup import logger
from affine.src.behavior_guard.gate import (
    record_deployment_fingerprint,
    verdict_counts,
)
from affine.src.behavior_guard.models import (
    BehaviorGateConfig,
    BehaviorVerdict,
    ProbeClassification,
    ProbeResult,
    VerdictStatus,
    aggregate_probe_results,
    parse_behavior_gate_config,
)
from affine.src.behavior_guard.probe import BehaviorProbeClient, ProbeConfig


ProbeClientFactory = Callable[..., Any]
EndpointLoadReader = Callable[[str], Optional[int]]
_CLIENT_CLEANUP_TIMEOUT_SECONDS = 5.0


class _LeaseLost(RuntimeError):
    """Another coordinator owns this deployment's verdict lease."""


class BehaviorPreflightCoordinator:
    """Watch battle state and produce durable deployment-bound verdicts."""

    def __init__(
        self,
        *,
        state: Any,
        dao: Any,
        poll_interval_seconds: float = 5.0,
        api_key: Optional[str] = None,
        probe_client_factory: ProbeClientFactory = BehaviorProbeClient,
        endpoint_load_reader: Optional[EndpointLoadReader] = None,
        benchmark_load_limit: int = 0,
        endpoint_capacity: int = 0,
        probe_slot_limit: int = 1,
    ) -> None:
        self._state = state
        self._dao = dao
        self._poll_interval_seconds = max(0.1, float(poll_interval_seconds))
        self._api_key = api_key if api_key is not None else os.getenv("API_KEY")
        self._probe_client_factory = probe_client_factory
        self._endpoint_load_reader = endpoint_load_reader
        self._benchmark_load_limit = max(0, int(benchmark_load_limit))
        self._endpoint_capacity = max(0, int(endpoint_capacity))
        self._probe_slot_limit = max(1, int(probe_slot_limit))
        self._endpoint_probe_slots: dict[str, asyncio.BoundedSemaphore] = {}

    async def run(self) -> None:
        """Run until cancelled; individual probe/DB failures are isolated."""

        logger.info("behavior preflight coordinator started (CPU-side)")
        while True:
            try:
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Do not include endpoint URLs or model output in this log.
                logger.warning(
                    "behavior preflight tick failed; will retry: "
                    f"{type(exc).__name__}"
                )
            await asyncio.sleep(self._poll_interval_seconds)

    async def run_once(self) -> None:
        config = parse_behavior_gate_config(
            await self._state.get_behavior_gate_config()
        )
        if not config.enabled:
            return

        battle = await self._state.get_battle()
        predeployed = await self._state.get_predeployed_challengers()
        records = ([battle] if battle is not None else []) + list(predeployed)
        seen: set[tuple[str, str, str]] = set()
        pending: list[tuple[Any, str]] = []
        for record in records:
            fingerprint = record_deployment_fingerprint(record, config)
            identity = (
                record.challenger.hotkey,
                record.challenger.revision,
                fingerprint,
            )
            if identity in seen:
                continue
            seen.add(identity)
            pending.append((record, fingerprint))

        # Deployments occupy independent serving endpoints.  Probe them in
        # parallel so one slow GPU cannot consume another deployment's entire
        # absolute admission window.  The durable lease still guarantees at
        # most one coordinator round per deployment fingerprint.
        outcomes = await asyncio.gather(
            *(
                self._ensure_verdict(record, config, fingerprint)
                for record, fingerprint in pending
            ),
            return_exceptions=True,
        )
        for (record, _fingerprint), outcome in zip(pending, outcomes):
            if isinstance(outcome, asyncio.CancelledError):
                raise outcome
            if isinstance(outcome, BaseException):
                logger.warning(
                    "behavior preflight deployment tick failed; will retry "
                    f"uid={record.challenger.uid}: {type(outcome).__name__}"
                )

    async def _ensure_verdict(
        self,
        record: Any,
        config: BehaviorGateConfig,
        fingerprint: str,
    ) -> None:
        challenger = record.challenger
        args = (
            challenger.hotkey,
            challenger.revision,
            config.policy_version,
            fingerprint,
        )
        existing = await self._dao.get_verdict(*args)
        if existing and str(existing.get("status") or "") in (
            VerdictStatus.PASSED.value,
            VerdictStatus.FAILED.value,
            VerdictStatus.EXPIRED.value,
        ):
            return

        # Anchor the admission hold in the durable deployment row.  Computing
        # a fresh relative timeout in every coordinator round lets retries,
        # process restarts, and lease hand-offs extend a nominal five-minute
        # preflight indefinitely.  ``ensure_pending`` stores this value with
        # ``if_not_exists`` semantics, so the first coordinator wins and all
        # later workers consume the same wall-clock deadline.
        now = time.time()
        created_at = _positive_timestamp((existing or {}).get("created_at"))
        record_deadline_at = _positive_timestamp(
            getattr(record, "behavior_admission_deadline_at", None)
        )
        requested_deadline_at = math.ceil(
            record_deadline_at
            or ((created_at or now) + config.admission_hold_seconds)
        )
        existing = await self._dao.ensure_pending(
            *args,
            admission_deadline_at=requested_deadline_at,
        )
        admission_deadline_at = _positive_timestamp(
            existing.get("admission_deadline_at")
        ) or float(requested_deadline_at)
        if time.time() >= admission_deadline_at:
            return
        if not self._ready_for_probe(existing, config):
            return

        owner = f"executor-preflight-{uuid.uuid4().hex}"
        acquired = await self._dao.acquire_lease(
            *args,
            owner_token=owner,
            lease_seconds=config.lease_seconds,
        )
        if not acquired:
            return

        results: list[ProbeResult] = []
        status = VerdictStatus.DEFERRED
        reason = "preflight_internal_error"
        previous_round_counts = _previous_round_counts(existing)
        try:
            endpoints = _record_endpoints(record)
            remaining_seconds = admission_deadline_at - time.time()
            if remaining_seconds <= 0:
                status = VerdictStatus.DEFERRED
                reason = "deployment_admission_deadline_exceeded"
            else:
                # Every serving endpoint receives the same remaining
                # deployment-wide budget concurrently.  Sequential probing let
                # endpoint zero monopolize the hold and made endpoint order a
                # hidden admission policy.
                endpoint_tasks: list[asyncio.Task] = []
                async with asyncio.TaskGroup() as task_group:
                    for endpoint_number, (base_url, endpoint_name) in enumerate(
                        endpoints
                    ):
                        endpoint_tasks.append(task_group.create_task(
                            self._probe_endpoint(
                                base_url=base_url,
                                endpoint_name=endpoint_name,
                                model=challenger.model,
                                config=config,
                                owner=owner,
                                dao_args=args,
                                endpoint_number=endpoint_number,
                                admission_timeout_seconds=min(
                                    config.admission_timeout_seconds,
                                    remaining_seconds,
                                ),
                            )
                        ))
                endpoint_outcomes = [
                    endpoint_task.result()
                    for endpoint_task in endpoint_tasks
                ]
                endpoint_verdicts: list[BehaviorVerdict] = []
                for endpoint_results, endpoint_verdict in endpoint_outcomes:
                    results.extend(endpoint_results)
                    endpoint_verdicts.append(endpoint_verdict)
                status, reason = _combine_endpoint_verdicts(
                    endpoint_verdicts,
                    expected_endpoint_count=len(endpoints),
                )
                # Definitive proof that completed within the bounded calls wins
                # over the wall-clock boundary.  Only an inconclusive aggregate
                # is converted to deployment expiry.
                if (
                    status not in {VerdictStatus.PASSED, VerdictStatus.FAILED}
                    and (
                        time.time() >= admission_deadline_at
                        or any(
                            verdict.reason
                            == "endpoint_admission_deadline_exceeded"
                            for verdict in endpoint_verdicts
                        )
                    )
                ):
                    status = VerdictStatus.DEFERRED
                    reason = "deployment_admission_deadline_exceeded"
            aggregate = aggregate_probe_results(results, config)
            counts = verdict_counts(aggregate)
            counts.update(previous_round_counts)
            if status is VerdictStatus.SUSPECTED:
                suspected_rounds = (
                    previous_round_counts.get("suspected_rounds", 0) + 1
                )
                counts["suspected_rounds"] = suspected_rounds
                if suspected_rounds >= config.suspected_rounds_to_fail:
                    status = VerdictStatus.FAILED
                    reason = f"repeated_suspected_rounds:{reason}"
            elif status is VerdictStatus.DEFERRED:
                counts["deferred_rounds"] = (
                    previous_round_counts.get("deferred_rounds", 0) + 1
                )
            evidence = {
                "reason_code": reason,
                "attempt_number": len(results),
            }
            written = await self._dao.set_verdict(
                *args,
                status=status.value,
                reason_code=reason,
                counts=counts,
                evidence=evidence,
                owner_token=owner,
            )
            if written:
                level = logger.warning if status is VerdictStatus.FAILED else logger.info
                level(
                    "behavior preflight verdict "
                    f"uid={challenger.uid} status={status.value} "
                    f"reason={reason} attempts={len(results)} "
                    f"mode={config.mode.value}"
                )
        except asyncio.CancelledError:
            raise
        except _LeaseLost:
            logger.info(
                f"behavior preflight lease changed for uid={challenger.uid}; "
                "discarding local round"
            )
        except Exception as exc:
            # An internal client/DB issue is infrastructure-deferred, never a
            # model strike.  Persist only the exception type as a safe code.
            deferred_counts = {
                "total": len(results),
                **previous_round_counts,
                "deferred_rounds": (
                    previous_round_counts.get("deferred_rounds", 0) + 1
                ),
            }
            await self._dao.set_verdict(
                *args,
                status=VerdictStatus.DEFERRED.value,
                reason_code="preflight_internal_error",
                counts=deferred_counts,
                evidence={"error_type": type(exc).__name__, "retryable": True},
                owner_token=owner,
            )
            logger.warning(
                "behavior preflight deferred "
                f"uid={challenger.uid}: {type(exc).__name__}"
            )
        finally:
            # A successful non-running verdict already removes the lease.
            try:
                await asyncio.wait_for(
                    self._dao.release_lease(*args, owner_token=owner),
                    timeout=5.0,
                )
            except asyncio.CancelledError:
                # Never consume manager shutdown merely because it arrived
                # during best-effort lease cleanup.
                raise
            except Exception:
                # Lease expiry is the recovery mechanism.  Never let cleanup
                # hide the original probe error or block executor shutdown.
                pass

    async def _probe_endpoint(
        self,
        *,
        base_url: str,
        endpoint_name: str,
        model: str,
        config: BehaviorGateConfig,
        owner: str,
        dao_args: tuple[str, str, str, str],
        endpoint_number: int,
        admission_timeout_seconds: float,
    ) -> tuple[list[ProbeResult], BehaviorVerdict]:
        results: list[ProbeResult] = []
        endpoint_hash = hashlib.sha256(base_url.encode("utf-8")).hexdigest()[:12]
        probe_config = ProbeConfig(
            first_response_timeout_s=config.first_response_deadline_seconds,
            first_action_timeout_s=config.first_action_deadline_seconds,
            total_timeout_s=config.probe_timeout_seconds,
            max_completion_tokens=min(
                config.max_completion_tokens,
                config.max_tokens_without_progress,
            ),
        )

        try:
            # Keep the existing per-endpoint cap, but also clamp it to the
            # deployment-wide remaining hold supplied by the caller.  The
            # latter is durable and never restarts between endpoints/rounds.
            async with asyncio.timeout(max(0.001, admission_timeout_seconds)):
                async with _probe_client(
                    self._probe_client_factory,
                    base_url=base_url,
                    model=model,
                    api_key=self._api_key,
                    config=probe_config,
                ) as client:
                    control_resolved = False
                    control_admissible = False
                    clean_action_count = 0
                    for attempt in range(config.attempt_budget):
                        probe_type = (
                            "control" if not control_resolved else "action"
                        )
                        probe_id = (
                            f"{uuid.uuid4().hex}-ep{endpoint_number}-"
                            f"{probe_type}"
                        )
                        endpoint_slot = self._endpoint_probe_slot(
                            endpoint_name, endpoint_hash,
                        )
                        async with endpoint_slot:
                            load_before = self._read_endpoint_load(endpoint_name)
                            if probe_type == "control":
                                result = await client.run_control_probe(
                                    probe_id=probe_id
                                )
                            else:
                                result = await client.run_action_probe(
                                    probe_id=probe_id
                                )
                            load_after = self._read_endpoint_load(endpoint_name)
                        endpoint_healthy = self._load_window_is_healthy(
                            load_before, load_after,
                        )
                        result = _guard_model_attribution_by_load(
                            result,
                            endpoint_healthy=endpoint_healthy,
                        )
                        results.append(result)
                        if (
                            probe_type == "control"
                            and not result.is_infra_failure
                        ):
                            control_resolved = True
                            control_admissible = result.is_admissible_completion
                        if (
                            probe_type == "action"
                            and result.classification
                            is ProbeClassification.CLEAN
                            and result.first_action_ms is not None
                        ):
                            clean_action_count += 1

                        renewed = await self._dao.renew_lease(
                            *dao_args,
                            owner_token=owner,
                            lease_seconds=config.lease_seconds,
                        )
                        if not renewed:
                            raise _LeaseLost(
                                "behavior preflight lease expired or changed"
                            )
                        await self._dao.record_attempt(
                            *dao_args,
                            probe_id=result.probe_id,
                            classification=result.classification.value,
                            evidence=_attempt_evidence(
                                result,
                                probe_type=probe_type,
                                endpoint_hash=endpoint_hash,
                                endpoint_healthy=endpoint_healthy,
                                endpoint_in_flight_before=load_before,
                                endpoint_in_flight_after=load_after,
                                endpoint_capacity=self._endpoint_capacity,
                            ),
                            owner_token=owner,
                        )

                        verdict = aggregate_probe_results(results, config)
                        verdict = _require_admission_proofs(
                            verdict,
                            control_admissible=control_admissible,
                            clean_action_count=clean_action_count,
                            config=config,
                        )
                        if verdict.status in (
                            VerdictStatus.PASSED,
                            VerdictStatus.FAILED,
                        ):
                            return results, verdict

                        # The base probe budget is extended only to replace
                        # infra failures.  A single strike stays suspected; it
                        # is not converted into a loss by waiting longer.
                        replacements = min(
                            verdict.infra_failure_count,
                            config.max_infra_retries,
                        )
                        if len(results) >= config.probe_count + replacements:
                            return results, verdict
        except TimeoutError:
            timed_out = aggregate_probe_results(results, config)
            return results, replace(
                timed_out,
                status=VerdictStatus.DEFERRED,
                reason="endpoint_admission_deadline_exceeded",
            )

        return results, _require_admission_proofs(
            aggregate_probe_results(results, config),
            control_admissible=control_admissible,
            clean_action_count=clean_action_count,
            config=config,
        )

    def _endpoint_probe_slot(
        self,
        endpoint_name: str,
        endpoint_hash: str,
    ) -> asyncio.BoundedSemaphore:
        """Share the reserved request budget across concurrent deployments."""

        key = endpoint_name or f"unknown:{endpoint_hash}"
        slot = self._endpoint_probe_slots.get(key)
        if slot is None:
            slot = asyncio.BoundedSemaphore(self._probe_slot_limit)
            self._endpoint_probe_slots[key] = slot
        return slot

    def _read_endpoint_load(self, endpoint_name: str) -> Optional[int]:
        """Read a trusted benchmark-load counter without breaking a probe."""

        if self._endpoint_load_reader is None:
            return None
        try:
            # An empty endpoint name is meaningful for Targon-only managers:
            # with no per-host map they can still return the conservative
            # global in-flight count.  Mixed/named-host managers return None
            # for the same input, keeping attribution fail-safe.
            value = self._endpoint_load_reader(endpoint_name)
            if isinstance(value, bool) or value is None:
                return None
            parsed = int(value)
        except Exception:
            return None
        return parsed if parsed >= 0 else None

    def _load_window_is_healthy(
        self,
        load_before: Optional[int],
        load_after: Optional[int],
    ) -> bool:
        """Require known, in-budget load at both ends of a probe request."""

        if self._benchmark_load_limit <= 0:
            return False
        return (
            load_before is not None
            and load_after is not None
            and load_before <= self._benchmark_load_limit
            and load_after <= self._benchmark_load_limit
            and (
                self._endpoint_capacity <= 0
                or self._benchmark_load_limit < self._endpoint_capacity
            )
        )

    @staticmethod
    def _ready_for_probe(
        existing: Optional[dict[str, Any]],
        config: BehaviorGateConfig,
    ) -> bool:
        if not existing:
            return True
        status = str(existing.get("status") or "")
        if status in (
            VerdictStatus.PASSED.value,
            VerdictStatus.FAILED.value,
            VerdictStatus.EXPIRED.value,
        ):
            return False
        if status in (VerdictStatus.SUSPECTED.value, VerdictStatus.DEFERRED.value):
            updated_at = int(existing.get("updated_at") or 0)
            return int(time.time()) - updated_at >= config.retry_backoff_seconds
        # pending/running rows rely on the DAO lease condition to arbitrate
        # stale owners and process restarts.
        return True


def _record_endpoints(record: Any) -> list[tuple[str, str]]:
    endpoints = [
        (
            str(deployment.base_url),
            str(getattr(deployment, "endpoint_name", "") or ""),
        )
        for deployment in (getattr(record, "deployments", ()) or ())
        if getattr(deployment, "base_url", None)
    ]
    fallback = getattr(record, "base_url", None)
    if not endpoints and fallback:
        endpoints.append((
            str(fallback),
            _endpoint_name_from_deployment_id(
                getattr(record, "deployment_id", None)
            ),
        ))
    return list(dict.fromkeys(endpoints))


def _endpoint_name_from_deployment_id(value: Any) -> str:
    deployment_id = str(value or "")
    if not deployment_id.startswith("ssh:"):
        return ""
    parts = deployment_id.split(":")
    return parts[1] if len(parts) >= 3 else ""


def _previous_round_counts(existing: Optional[dict[str, Any]]) -> dict[str, int]:
    raw = existing.get("counts") if isinstance(existing, dict) else None
    if not isinstance(raw, dict):
        return {}

    counts: dict[str, int] = {}
    for name in ("suspected_rounds", "deferred_rounds"):
        try:
            value = max(0, int(raw.get(name) or 0))
        except (TypeError, ValueError):
            value = 0
        if value:
            counts[name] = value
    return counts


def _positive_timestamp(value: Any) -> Optional[float]:
    """Return a finite positive epoch timestamp from a persisted scalar."""

    try:
        timestamp = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(timestamp) or timestamp <= 0:
        return None
    return timestamp


def _combine_endpoint_verdicts(
    verdicts: Iterable[BehaviorVerdict],
    *,
    expected_endpoint_count: Optional[int] = None,
) -> tuple[VerdictStatus, str]:
    rows = list(verdicts)
    if not rows:
        return VerdictStatus.DEFERRED, "no_serving_endpoint"
    expected = expected_endpoint_count if expected_endpoint_count is not None else len(rows)
    if len(rows) < expected:
        return VerdictStatus.DEFERRED, "endpoint_probe_incomplete"
    failed = [row for row in rows if row.status is VerdictStatus.FAILED]
    if failed:
        if len(rows) == 1 or len(failed) == len(rows):
            return VerdictStatus.FAILED, failed[0].reason
        # One validator endpoint can have a broken proxy/tool parser while
        # its peers serve the model correctly.  Endpoint disagreement is an
        # infrastructure isolation signal, never enough to lose the model.
        return VerdictStatus.DEFERRED, "endpoint_verdict_disagreement"
    suspected = [
        row for row in rows if row.status is VerdictStatus.SUSPECTED
    ]
    if suspected:
        if len(rows) == 1 or len(suspected) == len(rows):
            return VerdictStatus.SUSPECTED, suspected[0].reason
        # Cross-round suspicion is deployment-scoped.  Letting one repeatedly
        # bad proxy accumulate that counter while a peer keeps passing would
        # turn an endpoint disagreement into a model loss.
        return VerdictStatus.DEFERRED, "endpoint_verdict_disagreement"
    for verdict in rows:
        if verdict.status in (VerdictStatus.DEFERRED, VerdictStatus.RUNNING):
            return VerdictStatus.DEFERRED, verdict.reason
    if all(verdict.status is VerdictStatus.PASSED for verdict in rows):
        return VerdictStatus.PASSED, "all_endpoints_bounded_and_action_capable"
    return VerdictStatus.DEFERRED, "inconclusive_endpoint_verdict"


def _require_admission_proofs(
    verdict: BehaviorVerdict,
    *,
    control_admissible: bool,
    clean_action_count: int,
    config: BehaviorGateConfig,
) -> BehaviorVerdict:
    """Never authorize a deployment that only talks but cannot act.

    A bounded refusal is not a resource-abuse strike, but it also is not the
    state-changing proof required before SWE/Terminal fan-out.  One round is
    ``suspected``; repeated suspected rounds are rejected by the coordinator's
    explicit cross-round threshold.
    """
    if verdict.status is not VerdictStatus.PASSED:
        return verdict
    if not control_admissible:
        return replace(
            verdict,
            status=VerdictStatus.SUSPECTED,
            reason="control_proof_missing",
        )
    if clean_action_count >= config.action_proofs_to_pass:
        return verdict
    return replace(
        verdict,
        status=VerdictStatus.SUSPECTED,
        reason="action_proof_missing",
    )


def _attempt_evidence(
    result: ProbeResult,
    *,
    probe_type: str,
    endpoint_hash: str,
    endpoint_healthy: bool,
    endpoint_in_flight_before: Optional[int],
    endpoint_in_flight_after: Optional[int],
    endpoint_capacity: int,
) -> dict[str, Any]:
    # ``endpoint_hash`` intentionally uses an allowlisted code field.  The
    # raw URL is never handed to the persistence layer.
    return {
        "classification": result.classification.value,
        "probe_type": probe_type,
        "elapsed_ms": result.duration_ms,
        "first_response_ms": result.first_response_ms,
        "first_action_ms": result.first_action_ms,
        "completion_tokens": result.completion_tokens,
        "output_bytes": result.output_bytes,
        "action_observed": result.first_action_ms is not None,
        "reason_code": result.reason,
        "response_sha256": result.evidence_hash,
        "request_sha256": endpoint_hash,
        "endpoint_healthy": endpoint_healthy,
        "endpoint_in_flight_before": endpoint_in_flight_before,
        "endpoint_in_flight_after": endpoint_in_flight_after,
        "endpoint_capacity": endpoint_capacity or None,
    }


def _guard_model_attribution_by_load(
    result: ProbeResult,
    *,
    endpoint_healthy: bool,
) -> ProbeResult:
    """A model-like failure is infra evidence until load is attributable."""

    if endpoint_healthy or not result.is_strike:
        return result
    return replace(
        result,
        classification=ProbeClassification.INFRA_FAILURE,
        reason="endpoint_load_unattributable",
    )


@asynccontextmanager
async def _probe_client(factory: ProbeClientFactory, **kwargs: Any):
    client = factory(**kwargs)
    enter = getattr(client, "__aenter__", None)
    exit_ = getattr(client, "__aexit__", None)
    if enter is not None and exit_ is not None:
        managed = await enter()
        try:
            yield managed
        except BaseException as exc:
            suppressed = await _bounded_client_cleanup(
                exit_(type(exc), exc, exc.__traceback__)
            )
            if isinstance(exc, asyncio.CancelledError) or not suppressed:
                raise
        else:
            await _bounded_client_cleanup(exit_(None, None, None))
        return
    try:
        yield client
    finally:
        close = getattr(client, "aclose", None)
        if close is not None:
            await _bounded_client_cleanup(close())


async def _bounded_client_cleanup(cleanup: Any) -> Any:
    task = asyncio.ensure_future(cleanup)
    try:
        done, _pending = await asyncio.wait(
            {task}, timeout=_CLIENT_CLEANUP_TIMEOUT_SECONDS
        )
    except asyncio.CancelledError:
        task.cancel()
        task.add_done_callback(_consume_cleanup_result)
        raise
    if task in done:
        return task.result()

    task.cancel()
    task.add_done_callback(_consume_cleanup_result)
    logger.warning("behavior preflight client cleanup timed out")
    return False


def _consume_cleanup_result(task: asyncio.Future[Any]) -> None:
    try:
        task.result()
    except BaseException:
        pass


__all__ = ["BehaviorPreflightCoordinator"]
