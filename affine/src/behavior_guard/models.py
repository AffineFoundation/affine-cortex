"""Pure policy and value objects for challenger behaviour preflight.

The behaviour gate is deliberately narrower than model-quality scoring.  It
answers whether a deployment completes bounded requests and speaks the model
protocol.  A timely wrong answer or refusal is not a resource-abuse strike,
but admission still requires independent clean action proofs before expensive
benchmark fan-out.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping
from urllib.parse import urlsplit, urlunsplit


class BehaviorGateMode(str, Enum):
    """Whether a configured gate observes or blocks sampling."""

    SHADOW = "shadow"
    ADMISSION_SHADOW = "admission_shadow"
    ENFORCE = "enforce"


POLICY_SCHEMA_EPOCH = "v3-load-attribution"
"""Bump when persisted verdict semantics are not backward compatible."""

TEMPLATE_POLICY_SCHEMA_EPOCH = "template-integrity-v1"
"""Stable epoch for invalid-sample/template quarantine semantics."""


class ProbeClassification(str, Enum):
    """Outcome of one bounded behaviour probe."""

    CLEAN = "clean"
    QUALITY_FAILURE = "quality_failure"
    MODEL_NO_PROGRESS = "model_no_progress"
    MODEL_PROTOCOL_FAILURE = "model_protocol_failure"
    HARNESS_INVALID = "harness_invalid"
    INFRA_FAILURE = "infra_failure"
    UNKNOWN = "unknown"

    # Short aliases keep policy call sites readable while preserving explicit
    # wire values that distinguish model failures from infrastructure failures.
    NO_PROGRESS = MODEL_NO_PROGRESS
    PROTOCOL_FAILURE = MODEL_PROTOCOL_FAILURE


STRIKE_CLASSIFICATIONS = frozenset(
    {
        ProbeClassification.MODEL_NO_PROGRESS,
        ProbeClassification.MODEL_PROTOCOL_FAILURE,
    }
)
ADMISSIBLE_COMPLETION_CLASSIFICATIONS = frozenset(
    {ProbeClassification.CLEAN, ProbeClassification.QUALITY_FAILURE}
)


class VerdictStatus(str, Enum):
    """Persistable aggregate state for one deployment fingerprint."""

    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    SUSPECTED = "suspected"
    FAILED = "failed"
    DEFERRED = "deferred"
    # Scheduler-owned control final.  This is deliberately distinct from a
    # model failure: it only says the deployment exhausted its bounded
    # admission window.  Once claimed atomically, a late probe must not turn
    # the already-released deployment into PASSED or FAILED.
    EXPIRED = "expired"


class FailureOwner(str, Enum):
    """Trusted component attribution supplied by an environment harness."""

    MODEL = "model"
    TEMPLATE = "template"
    HARNESS = "harness"
    INFRASTRUCTURE = "infrastructure"
    UNKNOWN = "unknown"


class InvalidSampleReason(str, Enum):
    """Sample-integrity failures which must never become benchmark scores."""

    POSITIVE_SCORE_ZERO_ACTIVITY = "positive_score_zero_activity"
    HARNESS_DEADLINE_EXCEEDED = "harness_deadline_exceeded"
    HARNESS_RESULT_ERROR = "harness_result_error"
    HARNESS_EXCEPTION = "harness_exception"
    TEMPLATE_FAMILY_QUARANTINED = "template_family_quarantined"


@dataclass(frozen=True)
class BehaviorGateConfig:
    """Validated runtime policy with fail-open bootstrap defaults."""

    enabled: bool = False
    mode: BehaviorGateMode = BehaviorGateMode.SHADOW
    policy_version: str = "v1"
    gated_environments: tuple[str, ...] = ("*",)
    probe_count: int = 3
    clean_to_pass: int = 2
    violations_to_fail: int = 2
    runtime_violations_to_fail: int = 2
    action_proofs_to_pass: int = 2
    suspected_rounds_to_fail: int = 2
    max_infra_retries: int = 2
    probe_concurrency: int = 1
    first_response_deadline_seconds: float = 60.0
    first_action_deadline_seconds: float = 90.0
    probe_timeout_seconds: float = 120.0
    admission_timeout_seconds: float = 390.0
    admission_hold_seconds: int = 300
    max_tokens_without_progress: int = 4096
    max_completion_tokens: int = 8192
    lease_seconds: int = 420
    retry_backoff_seconds: int = 60

    @property
    def enforces(self) -> bool:
        return self.enabled and self.mode is BehaviorGateMode.ENFORCE

    @property
    def blocks_admission(self) -> bool:
        """Whether workers must wait for a verdict or an admission deadline."""

        return self.enabled and self.mode in {
            BehaviorGateMode.ADMISSION_SHADOW,
            BehaviorGateMode.ENFORCE,
        }

    @property
    def attempt_budget(self) -> int:
        """Maximum initial probes plus replacements for infrastructure errors."""

        return self.probe_count + self.max_infra_retries

    def gates_environment(self, environment: str) -> bool:
        if not self.enabled:
            return False
        wanted = str(environment).casefold()
        return any(
            configured == "*" or configured.casefold() == wanted
            for configured in self.gated_environments
        )

    @property
    def policy_identity(self) -> str:
        """Version plus semantic digest used by deployment fingerprints.

        Operators should still bump ``policy_version`` for audit clarity, but
        changing a threshold cannot accidentally reuse an old final verdict.
        Rollout-only fields (enabled/mode/environment allowlist) are omitted.
        """
        payload = {
            "policy_schema_epoch": POLICY_SCHEMA_EPOCH,
            "policy_version": self.policy_version,
            "probe_count": self.probe_count,
            "clean_to_pass": self.clean_to_pass,
            "violations_to_fail": self.violations_to_fail,
            "runtime_violations_to_fail": self.runtime_violations_to_fail,
            "action_proofs_to_pass": self.action_proofs_to_pass,
            "suspected_rounds_to_fail": self.suspected_rounds_to_fail,
            "max_infra_retries": self.max_infra_retries,
            "first_response_deadline_seconds": self.first_response_deadline_seconds,
            "first_action_deadline_seconds": self.first_action_deadline_seconds,
            "probe_timeout_seconds": self.probe_timeout_seconds,
            "admission_timeout_seconds": self.admission_timeout_seconds,
            "admission_hold_seconds": self.admission_hold_seconds,
            "max_tokens_without_progress": self.max_tokens_without_progress,
            "max_completion_tokens": self.max_completion_tokens,
        }
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        return (
            f"{POLICY_SCHEMA_EPOCH}:{self.policy_version}:"
            f"{hashlib.sha256(canonical).hexdigest()[:16]}"
        )

    @property
    def template_policy_identity(self) -> str:
        """Identity for template health, independent of admission tuning.

        Admission thresholds, rollout mode and preflight timeouts may change
        frequently.  None of them changes whether a positive, zero-activity
        row is an invalid sample, so they must not silently discard an active
        template quarantine.
        """

        return TEMPLATE_POLICY_SCHEMA_EPOCH

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "BehaviorGateConfig":
        """Parse untrusted SystemConfig data without making startup unsafe."""

        if not isinstance(raw, Mapping):
            raw = {}
        nested = raw.get("behavior_gate")
        if isinstance(nested, Mapping) and "enabled" not in raw:
            raw = nested

        mode_raw = str(raw.get("mode", BehaviorGateMode.SHADOW.value)).lower()
        try:
            mode = BehaviorGateMode(mode_raw)
        except ValueError:
            mode = BehaviorGateMode.SHADOW

        action_proofs_to_pass = _bounded_int(
            raw.get("action_proofs_to_pass"), 2, minimum=1, maximum=4,
        )
        probe_count = _bounded_int(
            raw.get("probe_count"),
            3,
            minimum=1 + action_proofs_to_pass,
            maximum=5,
        )
        clean_to_pass = _bounded_int(
            raw.get("clean_to_pass"), 2, minimum=1, maximum=probe_count
        )
        violations_to_fail = _bounded_int(
            raw.get("violations_to_fail"), 2, minimum=1, maximum=probe_count
        )
        timeout = _bounded_float(
            raw.get("probe_timeout_seconds"), 120.0, minimum=1.0, maximum=120.0
        )
        first_response = _bounded_float(
            raw.get("first_response_deadline_seconds"),
            60.0,
            minimum=0.1,
            maximum=timeout,
        )
        first_action = _bounded_float(
            raw.get("first_action_deadline_seconds"),
            90.0,
            minimum=first_response,
            maximum=timeout,
        )
        admission_timeout = _bounded_float(
            raw.get("admission_timeout_seconds"),
            300.0,
            minimum=30.0,
            maximum=900.0,
        )
        # A clean but maximally slow endpoint must still be able to complete
        # the configured baseline sequence.  Infrastructure replacements may
        # defer at this outer budget, but ordinary control/action proofs must
        # not fall into a permanent retry loop merely because 3 * 120s is
        # larger than the historical 300s default.
        admission_timeout = max(
            admission_timeout,
            math.ceil(probe_count * timeout) + 30,
        )

        environments_raw = raw.get("gated_environments", ("*",))
        if isinstance(environments_raw, (list, tuple, set, frozenset)):
            environments = tuple(
                dict.fromkeys(
                    str(item).strip()
                    for item in environments_raw
                    if str(item).strip()
                )
            )
        else:
            environments = ("*",)

        policy_version = str(raw.get("policy_version") or "v1").strip() or "v1"
        policy_version = policy_version[:128]
        max_infra_retries = _bounded_int(
            raw.get("max_infra_retries"), 2, minimum=0, maximum=5
        )
        lease_seconds = _bounded_int(
            raw.get("lease_seconds"), 360, minimum=1, maximum=86_400
        )
        # The coordinator owns the lease for the full per-endpoint admission
        # round, not merely one probe request.  A shorter lease can expire
        # before the first renewal/verdict write and leave enforce mode
        # permanently waiting on a result that can never commit.
        lease_seconds = max(
            lease_seconds,
            math.ceil(max(timeout, admission_timeout)) + 30,
        )
        admission_hold_seconds = _bounded_int(
            raw.get("admission_hold_seconds"),
            300,
            minimum=30,
            maximum=900,
        )
        if mode is BehaviorGateMode.ENFORCE:
            # Enforce must not expire and recycle a healthy deployment before
            # its configured worst-case baseline can finish.  Admission-shadow
            # may intentionally use a shorter observation hold because expiry
            # there releases sampling rather than failing the model.
            admission_hold_seconds = max(
                admission_hold_seconds,
                math.ceil(admission_timeout),
            )

        return cls(
            enabled=_coerce_bool(raw.get("enabled"), False),
            mode=mode,
            policy_version=policy_version,
            gated_environments=environments,
            probe_count=probe_count,
            clean_to_pass=clean_to_pass,
            violations_to_fail=violations_to_fail,
            runtime_violations_to_fail=_bounded_int(
                raw.get("runtime_violations_to_fail"),
                2,
                minimum=2,
                maximum=5,
            ),
            action_proofs_to_pass=min(
                action_proofs_to_pass, max(1, probe_count - 1),
            ),
            suspected_rounds_to_fail=_bounded_int(
                raw.get("suspected_rounds_to_fail"),
                2,
                minimum=1,
                maximum=10,
            ),
            max_infra_retries=max_infra_retries,
            probe_concurrency=_bounded_int(
                raw.get("probe_concurrency"),
                1,
                minimum=1,
                maximum=1,
            ),
            first_response_deadline_seconds=first_response,
            first_action_deadline_seconds=first_action,
            probe_timeout_seconds=timeout,
            admission_timeout_seconds=admission_timeout,
            admission_hold_seconds=admission_hold_seconds,
            max_tokens_without_progress=_bounded_int(
                raw.get("max_tokens_without_progress"),
                4096,
                minimum=1,
                maximum=1_000_000,
            ),
            max_completion_tokens=_bounded_int(
                raw.get("max_completion_tokens"),
                8192,
                minimum=1,
                maximum=1_000_000,
            ),
            lease_seconds=lease_seconds,
            retry_backoff_seconds=_bounded_int(
                raw.get("retry_backoff_seconds"),
                60,
                minimum=1,
                maximum=86_400,
            ),
        )


def parse_behavior_gate_config(
    raw: Mapping[str, Any] | None,
) -> BehaviorGateConfig:
    """Public parser used by executor and scheduler integrations."""

    return BehaviorGateConfig.from_mapping(raw)


@dataclass(frozen=True)
class ProbeResult:
    """Sanitized probe evidence; raw model text must not be stored here."""

    probe_id: str
    classification: ProbeClassification
    reason: str = ""
    duration_ms: int = 0
    first_response_ms: int | None = None
    first_action_ms: int | None = None
    completion_tokens: int = 0
    output_bytes: int = 0
    evidence_hash: str = ""

    @property
    def is_strike(self) -> bool:
        return self.classification in STRIKE_CLASSIFICATIONS

    @property
    def is_admissible_completion(self) -> bool:
        return self.classification in ADMISSIBLE_COMPLETION_CLASSIFICATIONS

    @property
    def is_infra_failure(self) -> bool:
        return self.classification is ProbeClassification.INFRA_FAILURE


@dataclass(frozen=True)
class BehaviorVerdict:
    """Deterministic aggregate, suitable for persistence and gate decisions."""

    status: VerdictStatus
    reason: str
    total_count: int
    clean_count: int
    quality_failure_count: int
    strike_count: int
    infra_failure_count: int
    unknown_count: int

    @property
    def admissible_completion_count(self) -> int:
        return self.clean_count + self.quality_failure_count


def aggregate_probe_results(
    results: Iterable[ProbeResult],
    config: BehaviorGateConfig,
) -> BehaviorVerdict:
    """Aggregate probes by counts, independent of arrival or storage order.

    ``QUALITY_FAILURE`` is an admissible *behavioural* completion, not a
    quality pass.  Infrastructure failures never increment ``strike_count``.
    """

    counts = {classification: 0 for classification in ProbeClassification}
    total = 0
    for result in results:
        classification = _parse_classification(result.classification)
        counts[classification] += 1
        total += 1

    clean = counts[ProbeClassification.CLEAN]
    quality = counts[ProbeClassification.QUALITY_FAILURE]
    strikes = sum(counts[item] for item in STRIKE_CLASSIFICATIONS)
    infra = counts[ProbeClassification.INFRA_FAILURE]
    unknown = counts[ProbeClassification.UNKNOWN]
    admissible = clean + quality

    if total == 0:
        status = VerdictStatus.PENDING
        reason = "no_probe_results"
    elif strikes >= config.violations_to_fail:
        status = VerdictStatus.FAILED
        reason = f"strike_threshold:{_dominant_strike(counts).value}"
    elif strikes > 0:
        status = VerdictStatus.SUSPECTED
        reason = "strike_below_threshold"
    elif admissible >= config.clean_to_pass:
        status = VerdictStatus.PASSED
        reason = "bounded_completion_threshold"
    elif total >= config.attempt_budget:
        status = VerdictStatus.DEFERRED
        reason = "inconclusive_after_infra_retries"
    else:
        status = VerdictStatus.RUNNING
        reason = "more_probes_required"

    return BehaviorVerdict(
        status=status,
        reason=reason,
        total_count=total,
        clean_count=clean,
        quality_failure_count=quality,
        strike_count=strikes,
        infra_failure_count=infra,
        unknown_count=unknown,
    )


@dataclass(frozen=True)
class DeploymentIdentity:
    """Non-secret deployment fields used to bind a verdict to a rollout."""

    deployment_id: str
    base_url: str
    endpoint_name: str = ""


def deployment_fingerprint(
    *,
    hotkey: str,
    revision: str,
    policy_version: str,
    deployments: Iterable[DeploymentIdentity | Mapping[str, Any]] = (),
    deployment_id: str | None = None,
    base_url: str | None = None,
    endpoint_name: str | None = None,
    deployment_generation: str | int | None = None,
) -> str:
    """Hash subject, policy and sanitized deployment identities.

    URL credentials, query parameters and fragments are intentionally omitted,
    so bearer material cannot leak into the digest input or rotate verdicts.
    ``deployment_generation`` must change whenever a stable SSH endpoint and
    container name are reused; scheduler records use ``started_at_block``.
    """

    identities: list[tuple[str, str, str]] = []
    for deployment in deployments:
        identities.append(
            (
                str(_deployment_value(deployment, "endpoint_name") or ""),
                str(_deployment_value(deployment, "deployment_id") or ""),
                _sanitize_base_url(
                    str(_deployment_value(deployment, "base_url") or "")
                ),
            )
        )
    if deployment_id is not None or base_url is not None or endpoint_name is not None:
        identities.append(
            (
                str(endpoint_name or ""),
                str(deployment_id or ""),
                _sanitize_base_url(str(base_url or "")),
            )
        )

    payload = {
        "hotkey": str(hotkey),
        "revision": str(revision),
        "policy_version": str(policy_version),
        "deployment_generation": (
            str(deployment_generation)
            if deployment_generation is not None
            else ""
        ),
        "deployments": sorted(set(identities)),
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return f"bg1:{hashlib.sha256(canonical).hexdigest()}"


@dataclass(frozen=True)
class SampleOutcomeEvidence:
    """Minimal benchmark evidence needed for cross-field invariants."""

    score: float | int | None
    commands_executed: int | None
    llm_call_count: int | None
    total_tokens: int | None
    output_bytes: int | None
    terminated_reason: str | None = None


@dataclass(frozen=True)
class ModelBehaviorEvidence:
    """Structured, model-attributable runtime evidence.

    These fields are a producer contract, not hints to be reconstructed from
    exception strings.  In particular, a timeout or protocol error can count
    toward a model verdict only when a trusted harness confirms that the
    request reached the model and both the endpoint and template were healthy.
    A stable template family (or exact template id when no family exists) is
    mandatory so the DAO cannot treat two instances of one broken template as
    independent.
    """

    classification: ProbeClassification
    failure_owner: FailureOwner = FailureOwner.UNKNOWN
    request_dispatched: bool | None = None
    request_reached_model: bool | None = None
    endpoint_healthy: bool | None = None
    template_healthy: bool | None = None
    template_family_id: str | None = None
    template_id: str | None = None
    template_revision: str | None = None
    catalog_revision: str | None = None

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any] | None,
    ) -> "ModelBehaviorEvidence | None":
        if not isinstance(raw, Mapping):
            return None
        try:
            classification = ProbeClassification(str(raw.get("classification")))
        except ValueError:
            return None
        if classification not in {
            ProbeClassification.MODEL_NO_PROGRESS,
            ProbeClassification.MODEL_PROTOCOL_FAILURE,
        }:
            return None
        try:
            owner = FailureOwner(str(raw.get("failure_owner")))
        except ValueError:
            owner = FailureOwner.UNKNOWN
        return cls(
            classification=classification,
            failure_owner=owner,
            request_dispatched=_structured_request_dispatched(raw),
            request_reached_model=_strict_optional_bool(
                raw.get("request_reached_model")
            ),
            endpoint_healthy=_strict_optional_bool(raw.get("endpoint_healthy")),
            template_healthy=_strict_optional_bool(raw.get("template_healthy")),
            template_family_id=_optional_identifier(
                raw.get("template_family_id")
            ),
            template_id=_optional_identifier(raw.get("template_id")),
            template_revision=_optional_identifier(
                raw.get("template_revision")
            ),
            catalog_revision=_optional_identifier(raw.get("catalog_revision")),
        )

    @property
    def eligible_for_model_strike(self) -> bool:
        return (
            self.classification in {
                ProbeClassification.MODEL_NO_PROGRESS,
                ProbeClassification.MODEL_PROTOCOL_FAILURE,
            }
            and self.failure_owner is FailureOwner.MODEL
            and self.request_dispatched is True
            and self.request_reached_model is True
            and self.endpoint_healthy is True
            and self.template_healthy is True
            and (self.template_family_id is not None or self.template_id is not None)
        )

    def template_key_hash(self, environment: str) -> str | None:
        template_key = self.template_family_id or self.template_id
        if template_key is None:
            return None
        return _stable_digest({
            "kind": "behavior-template-family-v1",
            "environment": str(environment),
            "template_key": template_key,
        })

    def catalog_revision_hash(self, environment: str) -> str | None:
        if self.catalog_revision is None:
            return None
        return _stable_digest({
            "kind": "behavior-template-catalog-v1",
            "environment": str(environment),
            "catalog_revision": self.catalog_revision,
        })

    def attribution(self) -> dict[str, Any]:
        """Compact allowlisted payload for ``BehaviorGateDAO``."""

        return {
            "failure_owner": self.failure_owner.value,
            "request_dispatched": self.request_dispatched,
            "request_reached_model": self.request_reached_model,
            "endpoint_healthy": self.endpoint_healthy,
            "template_healthy": self.template_healthy,
        }

    def audit_evidence(self) -> dict[str, Any]:
        return {
            "classification": self.classification.value,
            "failure_owner": self.failure_owner.value,
            "request_dispatched": self.request_dispatched,
            "request_reached_model": self.request_reached_model,
            "endpoint_healthy": self.endpoint_healthy,
            "template_healthy": self.template_healthy,
            "template_family_id": self.template_family_id,
            "template_id": self.template_id,
            "template_revision": self.template_revision,
            "catalog_revision": self.catalog_revision,
        }


def classify_sample_invariant(
    evidence: SampleOutcomeEvidence,
) -> InvalidSampleReason | None:
    """Reject a UID208-style score that contains no model/action evidence.

    A positive benchmark score cannot prove model quality when the harness made
    zero LLM calls, executed zero commands, observed zero tokens and captured
    zero output.  Such a row is invalid evidence even when the task happened to
    pass its initial checks.
    """

    score = evidence.score
    positive_score = (
        isinstance(score, (int, float))
        and not isinstance(score, bool)
        and math.isfinite(float(score))
        and float(score) > 0.0
    )
    exact_zero_activity = all(
        _is_exact_zero(value)
        for value in (
            evidence.commands_executed,
            evidence.llm_call_count,
            evidence.total_tokens,
            evidence.output_bytes,
        )
    )
    if positive_score and exact_zero_activity:
        return InvalidSampleReason.POSITIVE_SCORE_ZERO_ACTIVITY
    return None


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return default


def _bounded_int(value: Any, default: int, *, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        parsed = default
    return min(max(parsed, minimum), maximum)


def _bounded_float(
    value: Any, default: float, *, minimum: float, maximum: float
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        parsed = default
    if not math.isfinite(parsed):
        parsed = default
    return min(max(parsed, minimum), maximum)


def _parse_classification(value: ProbeClassification | str) -> ProbeClassification:
    if isinstance(value, ProbeClassification):
        return value
    try:
        return ProbeClassification(str(value))
    except ValueError:
        return ProbeClassification.UNKNOWN


def _dominant_strike(
    counts: Mapping[ProbeClassification, int],
) -> ProbeClassification:
    priority = (
        ProbeClassification.MODEL_PROTOCOL_FAILURE,
        ProbeClassification.MODEL_NO_PROGRESS,
    )
    return max(priority, key=lambda item: (counts[item], -priority.index(item)))


def _deployment_value(
    deployment: DeploymentIdentity | Mapping[str, Any], field: str
) -> Any:
    if isinstance(deployment, Mapping):
        return deployment.get(field)
    return getattr(deployment, field, None)


def _sanitize_base_url(raw: str) -> str:
    value = raw.strip()
    if not value:
        return ""
    has_scheme = "://" in value
    parsed = urlsplit(value if has_scheme else f"//{value}")
    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").lower()
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    try:
        port = parsed.port
    except ValueError:
        port = None
    netloc = f"{hostname}:{port}" if port is not None else hostname
    path = re.sub(r"/{2,}", "/", parsed.path or "")
    if path != "/":
        path = path.rstrip("/")
    sanitized = urlunsplit((scheme, netloc, path, "", ""))
    if not has_scheme and sanitized.startswith("//"):
        sanitized = sanitized[2:]
    return sanitized


def _is_exact_zero(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) == 0.0
    )


def _strict_optional_bool(value: Any) -> bool | None:
    """Accept only native booleans from the structured producer contract."""

    return value if isinstance(value, bool) else None


def _structured_request_dispatched(raw: Mapping[str, Any]) -> bool | None:
    """Read the canonical wire field, with the deployed legacy alias.

    A conflicting canonical value wins.  This keeps an explicit
    ``request_dispatched=False`` from being upgraded by a stale
    ``request_attempted=True`` field in the same payload.
    """

    canonical = _strict_optional_bool(raw.get("request_dispatched"))
    if canonical is not None:
        return canonical
    return _strict_optional_bool(raw.get("request_attempted"))


def _optional_identifier(value: Any) -> str | None:
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        return None
    text = str(value).strip()
    return text[:256] if text else None


def _stable_digest(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
