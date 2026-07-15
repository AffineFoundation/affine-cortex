"""Persistent verdict, probe-attempt, and lease storage for behavior gating.

The table deliberately stores only aggregate counters and allowlisted,
redacted evidence.  Raw prompts, completions, tool arguments, and exception
messages do not belong in this audit/control plane.
"""

from __future__ import annotations

import re
import time
from decimal import Decimal
from typing import Any, Dict, List, Mapping, Optional
from urllib.parse import quote

from botocore.exceptions import ClientError

from affine.database.base_dao import BaseDAO
from affine.database.client import get_client
from affine.database.schema import get_table_name


class BehaviorGateDAO(BaseDAO):
    """DAO for one behavior-gate partition per exact deployment identity.

    ``VERDICT`` is the durable aggregate and lease row.  Probe attempts use
    ``ATTEMPT#{probe_id}`` rows and conditional puts, so a retried worker
    cannot count the same probe twice.
    """

    STATUS_PENDING = "pending"
    STATUS_RUNNING = "running"
    STATUS_PASSED = "passed"
    STATUS_SUSPECTED = "suspected"
    STATUS_FAILED = "failed"
    STATUS_DEFERRED = "deferred"

    STATUSES = frozenset({
        STATUS_PENDING,
        STATUS_RUNNING,
        STATUS_PASSED,
        STATUS_SUSPECTED,
        STATUS_FAILED,
        STATUS_DEFERRED,
    })
    FINAL_STATUSES = frozenset({STATUS_PASSED, STATUS_FAILED})

    ATTEMPT_CLASSIFICATIONS = frozenset({
        "clean",
        "quality_failure",
        "model_no_progress",
        "model_protocol_failure",
        "harness_invalid",
        "infra_failure",
        "unknown",
    })

    # This is intentionally an allowlist rather than a blocklist.  In
    # particular, fields named output/content/prompt/arguments/message are
    # never persisted by this DAO.
    _EVIDENCE_FIELDS = frozenset({
        "action_observed",
        "commands_executed",
        "attempt_number",
        "chunk_count",
        "classification",
        "completion_tokens",
        "content_chars",
        "done_received",
        "elapsed_ms",
        "eligible_samples",
        "environment",
        "error_type",
        "estimated_tokens",
        "failure_mode",
        "finish_reason",
        "first_action_ms",
        "first_response_ms",
        "harness_failure",
        "http_status",
        "infra_failure",
        "invariant_name",
        "llm_call_count",
        "model",
        "model_timeout_count",
        "prompt_tokens",
        "protocol_error",
        "probe_type",
        "reason_code",
        "request_sha256",
        "response_sha256",
        "retryable",
        "sample_invariant",
        "score",
        "stream_completed",
        "task_id",
        "terminated_reason",
        "timed_out",
        "timeout_rate",
        "timeout_source",
        "tool_call_valid",
        "total_tokens",
        "transport_phase",
        "reasoning_chars",
        "output_bytes",
    })
    _SAFE_CODE_RE = re.compile(r"[^A-Za-z0-9_.:/-]+")
    _COUNT_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")

    def __init__(self):
        self.table_name = get_table_name("behavior_gate")
        super().__init__()

    @classmethod
    def _component(cls, value: object, field: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError(f"{field} must not be empty")
        # Escaping separators prevents two different identity tuples from
        # collapsing onto one DynamoDB key.
        return quote(text, safe="")

    @classmethod
    def make_partition_key(
        cls,
        hotkey: str,
        revision: str,
        policy_version: str,
        deployment_fingerprint: str,
    ) -> str:
        return (
            f"SUBJECT#{cls._component(hotkey, 'hotkey')}"
            f"#REV#{cls._component(revision, 'revision')}"
            f"#POLICY#{cls._component(policy_version, 'policy_version')}"
            f"#DEPLOY#{cls._component(deployment_fingerprint, 'deployment_fingerprint')}"
        )

    @classmethod
    def _identity(
        cls,
        hotkey: str,
        revision: str,
        policy_version: str,
        deployment_fingerprint: str,
    ) -> Dict[str, str]:
        # Validate through make_partition_key before returning the original,
        # human-readable values stored as non-key attributes.
        cls.make_partition_key(
            hotkey, revision, policy_version, deployment_fingerprint,
        )
        return {
            "hotkey": str(hotkey),
            "revision": str(revision),
            "policy_version": str(policy_version),
            "deployment_fingerprint": str(deployment_fingerprint),
        }

    @classmethod
    def _safe_code(cls, value: object, *, limit: int = 160) -> str:
        return cls._SAFE_CODE_RE.sub("_", str(value).strip())[:limit]

    @classmethod
    def _sanitize_evidence(
        cls, evidence: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        sanitized: Dict[str, Any] = {}
        for key, value in (evidence or {}).items():
            if key not in cls._EVIDENCE_FIELDS or value is None:
                continue
            if isinstance(value, bool):
                sanitized[key] = value
            elif isinstance(value, (int, float, Decimal)):
                sanitized[key] = value
            elif isinstance(value, str):
                sanitized[key] = cls._safe_code(value, limit=256)
            # Lists, maps, bytes, and arbitrary objects are deliberately
            # dropped: these commonly contain raw model material.
        return sanitized

    @classmethod
    def _sanitize_counts(
        cls, counts: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        sanitized: Dict[str, Any] = {}
        for key, value in (counts or {}).items():
            if not cls._COUNT_KEY_RE.fullmatch(str(key)):
                raise ValueError(f"invalid count name: {key!r}")
            if isinstance(value, bool) or not isinstance(
                value, (int, float, Decimal),
            ):
                raise ValueError(f"count {key!r} must be numeric")
            if value < 0:
                raise ValueError(f"count {key!r} must be non-negative")
            sanitized[str(key)] = value
        return sanitized

    @staticmethod
    def _conditional_failure(error: ClientError) -> bool:
        return error.response.get("Error", {}).get("Code") == (
            "ConditionalCheckFailedException"
        )

    def _key(
        self,
        hotkey: str,
        revision: str,
        policy_version: str,
        deployment_fingerprint: str,
        sk: str,
    ) -> Dict[str, Dict[str, str]]:
        return {
            "pk": {"S": self.make_partition_key(
                hotkey,
                revision,
                policy_version,
                deployment_fingerprint,
            )},
            "sk": {"S": sk},
        }

    async def get_verdict(
        self,
        hotkey: str,
        revision: str,
        policy_version: str,
        deployment_fingerprint: str,
    ) -> Optional[Dict[str, Any]]:
        """Read the gate row strongly consistently."""
        response = await get_client().get_item(
            TableName=self.table_name,
            Key=self._key(
                hotkey,
                revision,
                policy_version,
                deployment_fingerprint,
                "VERDICT",
            ),
            ConsistentRead=True,
        )
        item = response.get("Item")
        return self._deserialize(item) if item else None

    async def ensure_pending(
        self,
        hotkey: str,
        revision: str,
        policy_version: str,
        deployment_fingerprint: str,
        *,
        now: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Create a pending gate row, preserving any existing verdict."""
        timestamp = int(time.time()) if now is None else int(now)
        identity = self._identity(
            hotkey, revision, policy_version, deployment_fingerprint,
        )
        item: Dict[str, Any] = {
            "pk": self.make_partition_key(
                hotkey, revision, policy_version, deployment_fingerprint,
            ),
            "sk": "VERDICT",
            **identity,
            "status": self.STATUS_PENDING,
            "created_at": timestamp,
            "updated_at": timestamp,
            "counts": {},
            "evidence": {},
        }
        try:
            await get_client().put_item(
                TableName=self.table_name,
                Item=self._serialize(item),
                ConditionExpression=(
                    "attribute_not_exists(pk) AND attribute_not_exists(sk)"
                ),
            )
            return item
        except ClientError as error:
            if not self._conditional_failure(error):
                raise
        existing = await self.get_verdict(
            hotkey, revision, policy_version, deployment_fingerprint,
        )
        if existing is None:  # Defensive: a condition failure implies a row.
            raise RuntimeError("behavior-gate row disappeared after conditional put")
        return existing

    async def acquire_lease(
        self,
        hotkey: str,
        revision: str,
        policy_version: str,
        deployment_fingerprint: str,
        *,
        owner_token: str,
        lease_seconds: int,
        now: Optional[int] = None,
    ) -> bool:
        """Acquire an expired/free gate lease unless a final verdict exists."""
        owner = str(owner_token).strip()
        if not owner:
            raise ValueError("owner_token must not be empty")
        if int(lease_seconds) <= 0:
            raise ValueError("lease_seconds must be positive")
        timestamp = int(time.time()) if now is None else int(now)
        expires_at = timestamp + int(lease_seconds)
        identity = self._identity(
            hotkey, revision, policy_version, deployment_fingerprint,
        )
        values: Dict[str, Dict[str, Any]] = {
            ":running": {"S": self.STATUS_RUNNING},
            ":passed": {"S": self.STATUS_PASSED},
            ":failed": {"S": self.STATUS_FAILED},
            ":owner": {"S": owner},
            ":now": {"N": str(timestamp)},
            ":expires": {"N": str(expires_at)},
        }
        for name, value in identity.items():
            values[f":{name}"] = {"S": value}
        try:
            await get_client().update_item(
                TableName=self.table_name,
                Key=self._key(
                    hotkey,
                    revision,
                    policy_version,
                    deployment_fingerprint,
                    "VERDICT",
                ),
                UpdateExpression=(
                    "SET #status = :running, lease_owner = :owner, "
                    "lease_expires_at = :expires, updated_at = :now, "
                    "created_at = if_not_exists(created_at, :now), "
                    "hotkey = if_not_exists(hotkey, :hotkey), "
                    "revision = if_not_exists(revision, :revision), "
                    "policy_version = if_not_exists(policy_version, :policy_version), "
                    "deployment_fingerprint = "
                    "if_not_exists(deployment_fingerprint, :deployment_fingerprint)"
                ),
                ConditionExpression=(
                    "(attribute_not_exists(#status) OR "
                    "(#status <> :passed AND #status <> :failed)) AND "
                    "(attribute_not_exists(lease_owner) OR "
                    "attribute_not_exists(lease_expires_at) OR "
                    "lease_expires_at <= :now OR lease_owner = :owner)"
                ),
                ExpressionAttributeNames={"#status": "status"},
                ExpressionAttributeValues=values,
            )
            return True
        except ClientError as error:
            if self._conditional_failure(error):
                return False
            raise

    async def renew_lease(
        self,
        hotkey: str,
        revision: str,
        policy_version: str,
        deployment_fingerprint: str,
        *,
        owner_token: str,
        lease_seconds: int,
        now: Optional[int] = None,
    ) -> bool:
        """Extend a currently owned, running lease."""
        owner = str(owner_token).strip()
        if not owner:
            raise ValueError("owner_token must not be empty")
        if int(lease_seconds) <= 0:
            raise ValueError("lease_seconds must be positive")
        timestamp = int(time.time()) if now is None else int(now)
        try:
            await get_client().update_item(
                TableName=self.table_name,
                Key=self._key(
                    hotkey,
                    revision,
                    policy_version,
                    deployment_fingerprint,
                    "VERDICT",
                ),
                UpdateExpression=(
                    "SET lease_expires_at = :expires, updated_at = :now"
                ),
                ConditionExpression=(
                    "lease_owner = :owner AND #status = :running "
                    "AND lease_expires_at > :now"
                ),
                ExpressionAttributeNames={"#status": "status"},
                ExpressionAttributeValues={
                    ":owner": {"S": owner},
                    ":running": {"S": self.STATUS_RUNNING},
                    ":expires": {"N": str(timestamp + int(lease_seconds))},
                    ":now": {"N": str(timestamp)},
                },
            )
            return True
        except ClientError as error:
            if self._conditional_failure(error):
                return False
            raise

    async def release_lease(
        self,
        hotkey: str,
        revision: str,
        policy_version: str,
        deployment_fingerprint: str,
        *,
        owner_token: str,
        now: Optional[int] = None,
    ) -> bool:
        """Release only the caller's lease; leave its current status intact."""
        timestamp = int(time.time()) if now is None else int(now)
        try:
            await get_client().update_item(
                TableName=self.table_name,
                Key=self._key(
                    hotkey,
                    revision,
                    policy_version,
                    deployment_fingerprint,
                    "VERDICT",
                ),
                UpdateExpression=(
                    "SET updated_at = :now REMOVE lease_owner, lease_expires_at"
                ),
                ConditionExpression=(
                    "lease_owner = :owner AND "
                    "(#status <> :passed AND #status <> :failed)"
                ),
                ExpressionAttributeNames={"#status": "status"},
                ExpressionAttributeValues={
                    ":owner": {"S": str(owner_token)},
                    ":now": {"N": str(timestamp)},
                    ":passed": {"S": self.STATUS_PASSED},
                    ":failed": {"S": self.STATUS_FAILED},
                },
            )
            return True
        except ClientError as error:
            if self._conditional_failure(error):
                return False
            raise

    async def record_attempt(
        self,
        hotkey: str,
        revision: str,
        policy_version: str,
        deployment_fingerprint: str,
        *,
        probe_id: str,
        classification: str,
        evidence: Optional[Mapping[str, Any]] = None,
        owner_token: Optional[str] = None,
        created_at: Optional[int] = None,
        ttl_days: int = 7,
    ) -> bool:
        """Conditionally store one redacted probe attempt.

        Returns ``False`` when the same ``probe_id`` was already recorded.
        The owner token is hashed/opaque operational metadata and is not
        needed for deduplication; it is included only to correlate workers.
        """
        if classification not in self.ATTEMPT_CLASSIFICATIONS:
            raise ValueError(f"invalid attempt classification: {classification}")
        if int(ttl_days) <= 0:
            raise ValueError("ttl_days must be positive")
        timestamp = int(time.time()) if created_at is None else int(created_at)
        probe_component = self._component(probe_id, "probe_id")
        identity = self._identity(
            hotkey, revision, policy_version, deployment_fingerprint,
        )
        item: Dict[str, Any] = {
            "pk": self.make_partition_key(
                hotkey, revision, policy_version, deployment_fingerprint,
            ),
            "sk": f"ATTEMPT#{probe_component}",
            **identity,
            "probe_id": str(probe_id),
            "classification": classification,
            "evidence": self._sanitize_evidence(evidence),
            "created_at": timestamp,
            "ttl": max(int(time.time()), timestamp) + int(ttl_days) * 86400,
        }
        if owner_token:
            item["owner_token"] = self._safe_code(owner_token, limit=160)
        try:
            await get_client().put_item(
                TableName=self.table_name,
                Item=self._serialize(item),
                ConditionExpression=(
                    "attribute_not_exists(pk) AND attribute_not_exists(sk)"
                ),
            )
            return True
        except ClientError as error:
            if self._conditional_failure(error):
                return False
            raise

    async def list_attempts(
        self,
        hotkey: str,
        revision: str,
        policy_version: str,
        deployment_fingerprint: str,
        *,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        return await self.query(
            pk=self.make_partition_key(
                hotkey, revision, policy_version, deployment_fingerprint,
            ),
            sk_prefix="ATTEMPT#",
            limit=limit,
        )

    async def record_runtime_invariant_observation(
        self,
        hotkey: str,
        revision: str,
        policy_version: str,
        deployment_fingerprint: str,
        *,
        signature_hash: str,
        task_hash: str,
        classification: str,
        evidence: Optional[Mapping[str, Any]] = None,
        threshold: int = 2,
    ) -> int:
        """Record one distinct-task runtime anomaly and return its count.

        The signature groups identical sanitized telemetry while the task hash
        makes retries idempotent.  A strongly-consistent count after each
        conditional put guarantees that, for two concurrent distinct tasks,
        the later completed writer observes both rows and can close the gate.
        Raw environment output and task identifiers never enter the key.
        """
        signature = str(signature_hash).lower()
        task = str(task_hash).lower()
        digest_re = re.compile(r"^[0-9a-f]{32,64}$")
        if not digest_re.fullmatch(signature):
            raise ValueError("signature_hash must be a 32-64 character hex digest")
        if not digest_re.fullmatch(task):
            raise ValueError("task_hash must be a 32-64 character hex digest")
        required = int(threshold)
        if required < 2 or required > 100:
            raise ValueError("threshold must be between 2 and 100")
        if classification != "harness_invalid":
            raise ValueError("runtime invariant classification must be harness_invalid")

        probe_id = f"runtime-{signature}-{task}"
        await self.record_attempt(
            hotkey,
            revision,
            policy_version,
            deployment_fingerprint,
            probe_id=probe_id,
            classification=classification,
            evidence=evidence,
        )

        response = await get_client().query(
            TableName=self.table_name,
            KeyConditionExpression="pk = :pk AND begins_with(sk, :sk)",
            ExpressionAttributeValues={
                ":pk": {"S": self.make_partition_key(
                    hotkey,
                    revision,
                    policy_version,
                    deployment_fingerprint,
                )},
                ":sk": {"S": f"ATTEMPT#runtime-{signature}-"},
            },
            Select="COUNT",
            Limit=required,
            ConsistentRead=True,
        )
        return min(required, max(0, int(response.get("Count") or 0)))

    async def record_runtime_timeout_outcome(
        self,
        hotkey: str,
        revision: str,
        policy_version: str,
        deployment_fingerprint: str,
        *,
        environment_hash: str,
        task_hash: str,
        classification: str,
        evidence: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, int]:
        """Record one distinct-task timeout outcome and return durable totals.

        Runtime-rate rows are scoped by deployment and environment.  The task
        digest makes retries idempotent, while a strongly consistent query
        ensures a concurrent writer sees every completed observation before
        deciding whether the circuit breaker has reached its threshold.
        Infrastructure and ambiguous outcomes must be filtered by the caller
        and cannot enter this denominator.
        """
        environment = str(environment_hash).lower()
        task = str(task_hash).lower()
        digest_re = re.compile(r"^[0-9a-f]{32,64}$")
        if not digest_re.fullmatch(environment):
            raise ValueError(
                "environment_hash must be a 32-64 character hex digest"
            )
        if not digest_re.fullmatch(task):
            raise ValueError("task_hash must be a 32-64 character hex digest")
        if classification not in {"clean", "model_no_progress"}:
            raise ValueError(
                "runtime timeout classification must be clean or "
                "model_no_progress"
            )

        await self.record_attempt(
            hotkey,
            revision,
            policy_version,
            deployment_fingerprint,
            probe_id=f"runtime-rate-{environment}-{task}",
            classification=classification,
            evidence=evidence,
        )

        params: Dict[str, Any] = {
            "TableName": self.table_name,
            "KeyConditionExpression": "pk = :pk AND begins_with(sk, :sk)",
            "ExpressionAttributeValues": {
                ":pk": {"S": self.make_partition_key(
                    hotkey,
                    revision,
                    policy_version,
                    deployment_fingerprint,
                )},
                ":sk": {"S": f"ATTEMPT#runtime-rate-{environment}-"},
            },
            "ProjectionExpression": "#classification",
            "ExpressionAttributeNames": {
                "#classification": "classification",
            },
            "ConsistentRead": True,
        }
        eligible_samples = 0
        model_timeout_count = 0
        while True:
            response = await get_client().query(**params)
            for raw_item in response.get("Items", []):
                outcome = self._deserialize(raw_item).get("classification")
                if outcome not in {"clean", "model_no_progress"}:
                    continue
                eligible_samples += 1
                if outcome == "model_no_progress":
                    model_timeout_count += 1
            last_key = response.get("LastEvaluatedKey")
            if not last_key:
                break
            params["ExclusiveStartKey"] = last_key

        return {
            "eligible_samples": eligible_samples,
            "model_timeout_count": model_timeout_count,
        }

    async def seal_for_promotion(
        self,
        hotkey: str,
        revision: str,
        policy_version: str,
        deployment_fingerprint: str,
        *,
        now: Optional[int] = None,
    ) -> bool:
        """Atomically establish the cutoff for challenger-only evidence.

        A runtime failure and this seal race on the same verdict row.  Exactly
        one can win: promotion requires ``passed`` and runtime failure requires
        the seal to be absent.  Repeating a successful seal is idempotent so a
        scheduler restart can safely finish promotion from the battle record.
        """
        timestamp = int(time.time()) if now is None else int(now)
        try:
            await get_client().update_item(
                TableName=self.table_name,
                Key=self._key(
                    hotkey,
                    revision,
                    policy_version,
                    deployment_fingerprint,
                    "VERDICT",
                ),
                UpdateExpression=(
                    "SET promotion_sealed_at = "
                    "if_not_exists(promotion_sealed_at, :now), updated_at = :now"
                ),
                ConditionExpression="#status = :passed",
                ExpressionAttributeNames={"#status": "status"},
                ExpressionAttributeValues={
                    ":passed": {"S": self.STATUS_PASSED},
                    ":now": {"N": str(timestamp)},
                },
            )
            return True
        except ClientError as error:
            if not self._conditional_failure(error):
                raise
        existing = await self.get_verdict(
            hotkey, revision, policy_version, deployment_fingerprint,
        )
        return bool(
            existing
            and existing.get("status") == self.STATUS_PASSED
            and existing.get("promotion_sealed_at") is not None
        )

    async def set_verdict(
        self,
        hotkey: str,
        revision: str,
        policy_version: str,
        deployment_fingerprint: str,
        *,
        status: str,
        reason_code: str,
        counts: Optional[Mapping[str, Any]] = None,
        evidence: Optional[Mapping[str, Any]] = None,
        owner_token: Optional[str] = None,
        now: Optional[int] = None,
    ) -> bool:
        """Write a preflight verdict without replacing an existing final.

        When ``owner_token`` is supplied, the update additionally requires
        the active lease to belong to that worker.  Repeating the same final
        status is reported as success without mutating the frozen row, which
        makes an uncertain/retried network response safe.  The sole stricter
        signal allowed to demote a pass is :meth:`fail_runtime_invariant`.
        """
        if status not in self.STATUSES:
            raise ValueError(f"invalid behavior-gate status: {status}")
        timestamp = int(time.time()) if now is None else int(now)
        identity = self._identity(
            hotkey, revision, policy_version, deployment_fingerprint,
        )
        values: Dict[str, Dict[str, Any]] = {
            ":status": {"S": status},
            ":passed": {"S": self.STATUS_PASSED},
            ":failed": {"S": self.STATUS_FAILED},
            ":reason": {"S": self._safe_code(reason_code)},
            ":counts": self._serialize({
                "value": self._sanitize_counts(counts),
            })["value"],
            ":evidence": self._serialize({
                "value": self._sanitize_evidence(evidence),
            })["value"],
            ":now": {"N": str(timestamp)},
        }
        for name, value in identity.items():
            values[f":{name}"] = {"S": value}

        condition = (
            "attribute_not_exists(#status) OR "
            "(#status <> :passed AND #status <> :failed)"
        )
        if owner_token is not None:
            condition = (
                f"({condition}) AND lease_owner = :owner "
                "AND lease_expires_at > :now"
            )
            values[":owner"] = {"S": str(owner_token)}

        update_expression = (
            "SET #status = :status, reason_code = :reason, counts = :counts, "
            "evidence = :evidence, updated_at = :now, "
            "created_at = if_not_exists(created_at, :now), "
            "hotkey = if_not_exists(hotkey, :hotkey), "
            "revision = if_not_exists(revision, :revision), "
            "policy_version = if_not_exists(policy_version, :policy_version), "
            "deployment_fingerprint = "
            "if_not_exists(deployment_fingerprint, :deployment_fingerprint)"
        )
        if status in self.FINAL_STATUSES:
            update_expression += ", decided_at = if_not_exists(decided_at, :now)"
        if status != self.STATUS_RUNNING:
            update_expression += " REMOVE lease_owner, lease_expires_at"

        try:
            await get_client().update_item(
                TableName=self.table_name,
                Key=self._key(
                    hotkey,
                    revision,
                    policy_version,
                    deployment_fingerprint,
                    "VERDICT",
                ),
                UpdateExpression=update_expression,
                ConditionExpression=condition,
                ExpressionAttributeNames={"#status": "status"},
                ExpressionAttributeValues=values,
            )
            return True
        except ClientError as error:
            if not self._conditional_failure(error):
                raise
        if status not in self.FINAL_STATUSES:
            return False
        existing = await self.get_verdict(
            hotkey, revision, policy_version, deployment_fingerprint,
        )
        return bool(existing and existing.get("status") == status)

    async def fail_runtime_invariant(
        self,
        hotkey: str,
        revision: str,
        policy_version: str,
        deployment_fingerprint: str,
        *,
        reason_code: str,
        evidence: Optional[Mapping[str, Any]] = None,
        counts: Optional[Mapping[str, Any]] = None,
        now: Optional[int] = None,
    ) -> bool:
        """Atomically make a deployment fail when runtime evidence is invalid.

        This is the deliberately narrow exception to the preflight-final
        immutability rule: an endpoint that passed a synthetic probe can still
        prove invalid in a real SWE/Terminal sample (for example, a positive
        score with zero LLM calls, commands, tokens, and output).  Such direct
        evidence must dominate every preflight/non-final state.

        A missing row is also eligible because shadow mode may let a real
        sample finish before preflight creates its verdict.  Once this method
        commits ``failed``, all ordinary verdict writes and lease acquisition
        remain blocked by their existing final-state conditions.  Repeating
        the runtime failure is idempotent and never mutates the frozen row.
        A promotion seal is the linearization cutoff: evidence that loses the
        atomic race to that seal cannot retroactively demote a champion.
        """
        timestamp = int(time.time()) if now is None else int(now)
        identity = self._identity(
            hotkey, revision, policy_version, deployment_fingerprint,
        )
        safe_reason = self._safe_code(reason_code) or "runtime_invariant"
        values: Dict[str, Dict[str, Any]] = {
            ":failed": {"S": self.STATUS_FAILED},
            ":reason": {"S": safe_reason},
            ":source": {"S": "runtime_invariant"},
            ":counts": self._serialize({
                "value": self._sanitize_counts(counts),
            })["value"],
            ":evidence": self._serialize({
                "value": self._sanitize_evidence(evidence),
            })["value"],
            ":now": {"N": str(timestamp)},
        }
        for name, value in identity.items():
            values[f":{name}"] = {"S": value}

        try:
            await get_client().update_item(
                TableName=self.table_name,
                Key=self._key(
                    hotkey,
                    revision,
                    policy_version,
                    deployment_fingerprint,
                    "VERDICT",
                ),
                UpdateExpression=(
                    "SET #status = :failed, reason_code = :reason, "
                    "failure_source = :source, counts = :counts, "
                    "evidence = :evidence, updated_at = :now, "
                    "decided_at = :now, runtime_failed_at = :now, "
                    "created_at = if_not_exists(created_at, :now), "
                    "hotkey = if_not_exists(hotkey, :hotkey), "
                    "revision = if_not_exists(revision, :revision), "
                    "policy_version = if_not_exists(policy_version, :policy_version), "
                    "deployment_fingerprint = "
                    "if_not_exists(deployment_fingerprint, :deployment_fingerprint) "
                    "REMOVE lease_owner, lease_expires_at"
                ),
                ConditionExpression=(
                    "(attribute_not_exists(#status) OR #status <> :failed) "
                    "AND attribute_not_exists(promotion_sealed_at)"
                ),
                ExpressionAttributeNames={"#status": "status"},
                ExpressionAttributeValues=values,
            )
            return True
        except ClientError as error:
            if not self._conditional_failure(error):
                raise

        # A competing runtime guard (or preflight failure) won the race.  A
        # strongly-consistent read turns the same terminal result into an
        # idempotent success without rewriting its original evidence/time.
        existing = await self.get_verdict(
            hotkey, revision, policy_version, deployment_fingerprint,
        )
        return bool(existing and existing.get("status") == self.STATUS_FAILED)
