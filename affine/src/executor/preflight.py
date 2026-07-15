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
    ) -> None:
        self._state = state
        self._dao = dao
        self._poll_interval_seconds = max(0.1, float(poll_interval_seconds))
        self._api_key = api_key if api_key is not None else os.getenv("API_KEY")
        self._probe_client_factory = probe_client_factory

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
            await self._ensure_verdict(record, config, fingerprint)

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
        if not self._ready_for_probe(existing, config):
            return
        await self._dao.ensure_pending(*args)

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
            endpoint_verdicts: list[BehaviorVerdict] = []
            base_urls = _record_base_urls(record)
            for endpoint_number, base_url in enumerate(base_urls):
                endpoint_results, endpoint_verdict = (
                    await self._probe_endpoint(
                        base_url=base_url,
                        model=challenger.model,
                        config=config,
                        owner=owner,
                        dao_args=args,
                        endpoint_number=endpoint_number,
                    )
                )
                results.extend(endpoint_results)
                endpoint_verdicts.append(endpoint_verdict)

            status, reason = _combine_endpoint_verdicts(
                endpoint_verdicts,
                expected_endpoint_count=len(base_urls),
            )
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
        model: str,
        config: BehaviorGateConfig,
        owner: str,
        dao_args: tuple[str, str, str, str],
        endpoint_number: int,
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
            # The admission budget is per serving endpoint.  A shared budget
            # lets the first slow endpoint starve every later endpoint, making
            # identical bad replicas defer forever instead of converging.
            async with asyncio.timeout(config.admission_timeout_seconds):
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
                        if probe_type == "control":
                            result = await client.run_control_probe(
                                probe_id=probe_id
                            )
                        else:
                            result = await client.run_action_probe(
                                probe_id=probe_id
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

    @staticmethod
    def _ready_for_probe(
        existing: Optional[dict[str, Any]],
        config: BehaviorGateConfig,
    ) -> bool:
        if not existing:
            return True
        status = str(existing.get("status") or "")
        if status in (VerdictStatus.PASSED.value, VerdictStatus.FAILED.value):
            return False
        if status in (VerdictStatus.SUSPECTED.value, VerdictStatus.DEFERRED.value):
            updated_at = int(existing.get("updated_at") or 0)
            return int(time.time()) - updated_at >= config.retry_backoff_seconds
        # pending/running rows rely on the DAO lease condition to arbitrate
        # stale owners and process restarts.
        return True


def _record_base_urls(record: Any) -> list[str]:
    urls = [
        str(deployment.base_url)
        for deployment in (getattr(record, "deployments", ()) or ())
        if getattr(deployment, "base_url", None)
    ]
    fallback = getattr(record, "base_url", None)
    if not urls and fallback:
        urls.append(str(fallback))
    return list(dict.fromkeys(urls))


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
    for verdict in rows:
        if verdict.status is VerdictStatus.SUSPECTED:
            return VerdictStatus.SUSPECTED, verdict.reason
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
    }


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
