"""Persistent verdict, probe-attempt, and lease storage for behavior gating.

The table deliberately stores only aggregate counters and allowlisted,
redacted evidence.  Raw prompts, completions, tool arguments, and exception
messages do not belong in this audit/control plane.
"""

from __future__ import annotations

import re
import time
from decimal import Decimal
from typing import Any, Dict, Iterable, List, Mapping, Optional
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
    STATUS_EXPIRED = "expired"

    STATUSES = frozenset({
        STATUS_PENDING,
        STATUS_RUNNING,
        STATUS_PASSED,
        STATUS_SUSPECTED,
        STATUS_FAILED,
        STATUS_DEFERRED,
        STATUS_EXPIRED,
    })
    FINAL_STATUSES = frozenset({STATUS_PASSED, STATUS_FAILED, STATUS_EXPIRED})

    ATTEMPT_CLASSIFICATIONS = frozenset({
        "clean",
        "quality_failure",
        "model_no_progress",
        "model_protocol_failure",
        "harness_invalid",
        "infra_failure",
        "unknown",
    })

    MODEL_CLASSIFICATIONS = frozenset({
        "model_no_progress",
        "model_protocol_failure",
    })

    TEMPLATE_STATUS_SUSPECTED = "suspected"
    TEMPLATE_STATUS_QUARANTINED = "quarantined"

    # This is intentionally an allowlist rather than a blocklist.  In
    # particular, fields named output/content/prompt/arguments/message are
    # never persisted by this DAO.
    _EVIDENCE_FIELDS = frozenset({
        "action_observed",
        "catalog_revision",
        "commands_executed",
        "attempt_number",
        "chunk_count",
        "classification",
        "completion_tokens",
        "content_chars",
        "done_received",
        "deadline_exceeded",
        "deployment_hash",
        "elapsed_ms",
        "endpoint_health",
        "endpoint_healthy",
        "endpoint_capacity",
        "endpoint_in_flight_after",
        "endpoint_in_flight_before",
        "environment",
        "error_type",
        "estimated_tokens",
        "failure_owner",
        "finish_reason",
        "first_action_ms",
        "first_action_deadline_ms",
        "first_response_ms",
        "first_response_deadline_ms",
        "http_status",
        "invariant_name",
        "invalid_reason",
        "last_progress_ms",
        "llm_call_count",
        "model",
        "observed_score",
        "policy_identity",
        "progress_deadline_ms",
        "prompt_tokens",
        "protocol_error",
        "probe_type",
        "reason_code",
        "refresh_block",
        "request_dispatched",
        "request_attempted",
        "request_reached_model",
        "request_sha256",
        "response_sha256",
        "retryable",
        "sample_invariant",
        "score",
        "stream_completed",
        "subject_hash",
        "task_hash",
        "task_id",
        "task_instance_hash",
        "template_health",
        "template_family_id",
        "template_healthy",
        "template_id",
        "template_key_hash",
        "template_revision",
        "terminated_reason",
        "tool_call_valid",
        "total_tokens",
        "transport_phase",
        "reasoning_chars",
        "output_bytes",
    })
    _SAFE_CODE_RE = re.compile(r"[^A-Za-z0-9_.:/-]+")
    _COUNT_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    _DIGEST_RE = re.compile(r"^[0-9a-f]{32,64}$")

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
    def make_invalid_sample_partition_key(
        cls,
        hotkey: str,
        revision: str,
        environment: str,
        refresh_block: int,
    ) -> str:
        """Key durable sample-integrity markers independently of deployments.

        An impossible sample belongs to the subject/task/window, not to the
        behavior policy or serving-process generation.  Keeping those fields
        out of this key prevents endpoint rotation from retrying the same
        invalid task forever and also permits champion samples to use the same
        integrity path as challenger samples.
        """

        block = cls._non_negative_int(refresh_block, "refresh_block")
        return (
            f"INVALID_SAMPLE#SUBJECT#{cls._component(hotkey, 'hotkey')}"
            f"#REV#{cls._component(revision, 'revision')}"
            f"#ENV#{cls._component(environment, 'environment')}"
            f"#REFRESH#{block}"
        )

    @classmethod
    def make_template_catalog_partition_key(
        cls,
        policy_identity: str,
        environment: str,
        catalog_revision_hash: str,
    ) -> str:
        """Key exact-task quarantine rows shared by every subject/deployment."""

        catalog = cls._digest(catalog_revision_hash, "catalog_revision_hash")
        return (
            f"TEMPLATE_TASKS#POLICY#{cls._component(policy_identity, 'policy_identity')}"
            f"#ENV#{cls._component(environment, 'environment')}"
            f"#CATALOG#{catalog}"
        )

    @classmethod
    def make_template_partition_key(
        cls,
        policy_identity: str,
        environment: str,
        catalog_revision_hash: str,
        template_key_hash: str,
    ) -> str:
        """Key one template family across subjects and deployment generations."""

        catalog = cls._digest(catalog_revision_hash, "catalog_revision_hash")
        template = cls._digest(template_key_hash, "template_key_hash")
        return (
            f"TEMPLATE#POLICY#{cls._component(policy_identity, 'policy_identity')}"
            f"#ENV#{cls._component(environment, 'environment')}"
            f"#CATALOG#{catalog}#FAMILY#{template}"
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
    def _digest(cls, value: object, field: str) -> str:
        digest = str(value).strip().lower()
        if not cls._DIGEST_RE.fullmatch(digest):
            raise ValueError(f"{field} must be a 32-64 character hex digest")
        return digest

    @staticmethod
    def _non_negative_int(value: object, field: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{field} must be a non-negative integer")
        try:
            parsed = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{field} must be a non-negative integer"
            ) from exc
        if parsed < 0:
            raise ValueError(f"{field} must be a non-negative integer")
        return parsed

    @staticmethod
    def _bounded_threshold(value: object, field: str) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{field} must be between 2 and 100") from exc
        if parsed < 2 or parsed > 100:
            raise ValueError(f"{field} must be between 2 and 100")
        return parsed

    @classmethod
    def _validate_model_attribution(
        cls,
        attribution: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Require explicit model ownership; never infer it from error text."""

        if not isinstance(attribution, Mapping):
            raise ValueError("attribution must be a mapping")
        owner = str(attribution.get("failure_owner") or "").strip().casefold()
        endpoint_healthy = (
            attribution.get("endpoint_healthy") is True
            or str(attribution.get("endpoint_health") or "").casefold()
            == "healthy"
        )
        template_healthy = (
            attribution.get("template_healthy") is True
            or str(attribution.get("template_health") or "").casefold()
            == "healthy"
        )
        required = {
            "failure_owner": owner == "model",
            "request_dispatched": attribution.get("request_dispatched") is True,
            "request_reached_model": (
                attribution.get("request_reached_model") is True
            ),
            "endpoint_healthy": endpoint_healthy,
            "template_healthy": template_healthy,
        }
        missing = [name for name, valid in required.items() if not valid]
        if missing:
            raise ValueError(
                "model attribution requires structured proof: "
                + ",".join(missing)
            )
        normalized = dict(attribution)
        normalized.update({
            "failure_owner": "model",
            "request_dispatched": True,
            "request_reached_model": True,
            "endpoint_healthy": True,
            "template_healthy": True,
        })
        return cls._sanitize_evidence(normalized)

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

    @staticmethod
    def _expiring_put_guard(now: int) -> Dict[str, Any]:
        """Allow idempotent creation, including replacement of a stale TTL row.

        DynamoDB removes expired TTL items asynchronously.  Treating physical
        presence as liveness would otherwise suppress a fresh observation for
        hours or days after its logical expiration.
        """

        return {
            "ConditionExpression": (
                "(attribute_not_exists(pk) AND attribute_not_exists(sk)) "
                "OR #ttl <= :write_now"
            ),
            "ExpressionAttributeNames": {"#ttl": "ttl"},
            "ExpressionAttributeValues": {
                ":write_now": {"N": str(int(now))},
            },
        }

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

    @staticmethod
    def _raw_key(pk: str, sk: str) -> Dict[str, Dict[str, str]]:
        return {"pk": {"S": pk}, "sk": {"S": sk}}

    async def _query_rows(
        self,
        *,
        pk: str,
        sk_prefix: str,
        consistent_read: bool = True,
    ) -> List[Dict[str, Any]]:
        """Read a whole small control-plane partition with TTL filtering."""

        rows: List[Dict[str, Any]] = []
        exclusive_start: Optional[Dict[str, Any]] = None
        while True:
            params: Dict[str, Any] = {
                "TableName": self.table_name,
                "KeyConditionExpression": (
                    "pk = :pk AND begins_with(sk, :sk)"
                ),
                "ExpressionAttributeValues": {
                    ":pk": {"S": pk},
                    ":sk": {"S": sk_prefix},
                },
                "ConsistentRead": bool(consistent_read),
            }
            if exclusive_start:
                params["ExclusiveStartKey"] = exclusive_start
            response = await get_client().query(**params)
            rows.extend(
                self._deserialize(item) for item in response.get("Items", [])
            )
            exclusive_start = response.get("LastEvaluatedKey")
            if not exclusive_start:
                break
        return rows

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
        admission_deadline_at: Optional[int] = None,
        now: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Create a pending row and immutably anchor its admission deadline.

        The deadline uses ``if_not_exists`` semantics on legacy/existing rows;
        retries, lease hand-offs, and process restarts can backfill a missing
        value but can never extend a previously established deadline.
        """
        timestamp = int(time.time()) if now is None else int(now)
        deadline: Optional[int] = None
        if admission_deadline_at is not None:
            deadline = self._non_negative_int(
                admission_deadline_at, "admission_deadline_at",
            )
            if deadline <= 0:
                raise ValueError("admission_deadline_at must be positive")
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
        if deadline is not None:
            item["admission_deadline_at"] = deadline
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
        if deadline is not None:
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
                        "SET admission_deadline_at = "
                        "if_not_exists(admission_deadline_at, :deadline)"
                    ),
                    ConditionExpression=(
                        "attribute_not_exists(admission_deadline_at)"
                    ),
                    ExpressionAttributeValues={
                        ":deadline": {"N": str(deadline)},
                    },
                )
            except ClientError as error:
                if not self._conditional_failure(error):
                    raise
        existing = await self.get_verdict(
            hotkey, revision, policy_version, deployment_fingerprint,
        )
        if existing is None:  # Defensive: a condition failure implies a row.
            raise RuntimeError("behavior-gate row disappeared after conditional put")
        return existing

    async def claim_expired_admission(
        self,
        hotkey: str,
        revision: str,
        policy_version: str,
        deployment_fingerprint: str,
        *,
        now: Optional[int] = None,
    ) -> bool:
        """Atomically claim a deployment whose admission deadline elapsed.

        ``expired`` is a control-plane final state.  Returning ``True`` means
        this call won the CAS; ``False`` means another final writer, another
        expiry claimant, or a not-yet-expired deadline won instead.  Callers
        that need to distinguish those cases should read the verdict.
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
                    "SET #status = :expired, reason_code = :reason, "
                    "admission_expired_at = :now, updated_at = :now, "
                    "decided_at = if_not_exists(decided_at, :now) "
                    "REMOVE lease_owner, lease_expires_at"
                ),
                ConditionExpression=(
                    "admission_deadline_at <= :now AND "
                    "(attribute_not_exists(#status) OR "
                    "(#status <> :passed AND #status <> :failed "
                    "AND #status <> :expired))"
                ),
                ExpressionAttributeNames={"#status": "status"},
                ExpressionAttributeValues={
                    ":expired": {"S": self.STATUS_EXPIRED},
                    ":passed": {"S": self.STATUS_PASSED},
                    ":failed": {"S": self.STATUS_FAILED},
                    ":reason": {
                        "S": "deployment_admission_deadline_exceeded",
                    },
                    ":now": {"N": str(timestamp)},
                },
            )
            return True
        except ClientError as error:
            if self._conditional_failure(error):
                return False
            raise

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
            ":expired": {"S": self.STATUS_EXPIRED},
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
                    "(#status <> :passed AND #status <> :failed "
                    "AND #status <> :expired)) AND "
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
                    "(#status <> :passed AND #status <> :failed "
                    "AND #status <> :expired)"
                ),
                ExpressionAttributeNames={"#status": "status"},
                ExpressionAttributeValues={
                    ":owner": {"S": str(owner_token)},
                    ":now": {"N": str(timestamp)},
                    ":passed": {"S": self.STATUS_PASSED},
                    ":failed": {"S": self.STATUS_FAILED},
                    ":expired": {"S": self.STATUS_EXPIRED},
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
        write_now = int(time.time())
        timestamp = write_now if created_at is None else int(created_at)
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
            "ttl": max(write_now, timestamp) + int(ttl_days) * 86400,
        }
        if owner_token:
            item["owner_token"] = self._safe_code(owner_token, limit=160)
        try:
            await get_client().put_item(
                TableName=self.table_name,
                Item=self._serialize(item),
                **self._expiring_put_guard(write_now),
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

    async def record_invalid_sample(
        self,
        hotkey: str,
        revision: str,
        environment: str,
        refresh_block: int,
        task_id: int,
        *,
        reason_code: str,
        template_key_hash: Optional[str] = None,
        evidence: Optional[Mapping[str, Any]] = None,
        created_at: Optional[int] = None,
        ttl_days: int = 30,
    ) -> bool:
        """Durably complete an invalid subject/task without creating a score.

        These rows deliberately do not share the deployment-bound verdict
        partition.  A process rotation, policy rollout, or champion role must
        not cause an impossible-positive sample to be dispatched repeatedly.
        """

        if int(ttl_days) <= 0:
            raise ValueError("ttl_days must be positive")
        block = self._non_negative_int(refresh_block, "refresh_block")
        task = self._non_negative_int(task_id, "task_id")
        template = None
        if template_key_hash is not None:
            template = self._digest(template_key_hash, "template_key_hash")
        write_now = int(time.time())
        timestamp = write_now if created_at is None else int(created_at)
        pk = self.make_invalid_sample_partition_key(
            hotkey, revision, environment, block,
        )
        item: Dict[str, Any] = {
            "pk": pk,
            "sk": f"INVALID_SAMPLE#TASK#{task}",
            "hotkey": str(hotkey),
            "revision": str(revision),
            "environment": self._safe_code(environment),
            "refresh_block": block,
            "task_id": task,
            "sample_status": "invalid",
            "classification": "harness_invalid",
            "reason_code": self._safe_code(reason_code) or "invalid_sample",
            "evidence": self._sanitize_evidence(evidence),
            "created_at": timestamp,
            "ttl": max(write_now, timestamp) + int(ttl_days) * 86400,
        }
        if template is not None:
            item["template_key_hash"] = template
        try:
            await get_client().put_item(
                TableName=self.table_name,
                Item=self._serialize(item),
                **self._expiring_put_guard(write_now),
            )
            return True
        except ClientError as error:
            if self._conditional_failure(error):
                return False
            raise

    async def list_invalid_sample_task_ids(
        self,
        hotkey: str,
        revision: str,
        environment: str,
        refresh_block: int,
        *,
        task_ids: Optional[Iterable[int]] = None,
    ) -> set[int]:
        """Return invalid completions for one subject/env sampling window."""

        block = self._non_negative_int(refresh_block, "refresh_block")
        wanted = (
            {self._non_negative_int(item, "task_id") for item in task_ids}
            if task_ids is not None
            else None
        )
        rows = await self._query_rows(
            pk=self.make_invalid_sample_partition_key(
                hotkey, revision, environment, block,
            ),
            sk_prefix="INVALID_SAMPLE#TASK#",
        )
        now = int(time.time())
        result: set[int] = set()
        for row in rows:
            if int(row.get("ttl") or now + 1) <= now:
                continue
            try:
                task = int(row["task_id"])
            except (KeyError, TypeError, ValueError):
                continue
            if wanted is None or task in wanted:
                result.add(task)
        return result

    async def _record_invalid_task(
        self,
        policy_identity: str,
        environment: str,
        catalog_revision_hash: str,
        *,
        task_id: int,
        template_key_hash: str,
        subject_hash: str,
        deployment_hash: str,
        task_instance_hash: str,
        reason_code: str,
        evidence: Optional[Mapping[str, Any]],
        created_at: int,
        ttl_days: int,
    ) -> bool:
        """Quarantine one exact task after cross-subject reproduction."""

        task = self._non_negative_int(task_id, "task_id")
        template = self._digest(template_key_hash, "template_key_hash")
        subject = self._digest(subject_hash, "subject_hash")
        deployment = self._digest(deployment_hash, "deployment_hash")
        instance = self._digest(task_instance_hash, "task_instance_hash")
        pk = self.make_template_catalog_partition_key(
            policy_identity, environment, catalog_revision_hash,
        )
        write_now = int(time.time())
        item = {
            "pk": pk,
            "sk": f"INVALID_TASK#TASK#{task}",
            "task_id": task,
            "status": "invalid",
            "classification": "harness_invalid",
            "reason_code": self._safe_code(reason_code) or "template_incident",
            "template_key_hash": template,
            "subject_hash": subject,
            "deployment_hash": deployment,
            "task_instance_hash": instance,
            "evidence": self._sanitize_evidence(evidence),
            "created_at": created_at,
            "ttl": max(write_now, created_at) + int(ttl_days) * 86400,
        }
        try:
            await get_client().put_item(
                TableName=self.table_name,
                Item=self._serialize(item),
                **self._expiring_put_guard(write_now),
            )
            return True
        except ClientError as error:
            if self._conditional_failure(error):
                return False
            raise

    async def list_invalid_task_ids(
        self,
        policy_identity: str,
        environment: str,
        catalog_revision_hash: str,
        *,
        task_ids: Optional[Iterable[int]] = None,
        now: Optional[int] = None,
    ) -> set[int]:
        """Return exact tasks quarantined across every subject/deployment."""

        wanted = (
            {self._non_negative_int(item, "task_id") for item in task_ids}
            if task_ids is not None
            else None
        )
        rows = await self._query_rows(
            pk=self.make_template_catalog_partition_key(
                policy_identity, environment, catalog_revision_hash,
            ),
            sk_prefix="INVALID_TASK#TASK#",
        )
        timestamp = int(time.time()) if now is None else int(now)
        result: set[int] = set()
        for row in rows:
            if int(row.get("ttl") or timestamp + 1) <= timestamp:
                continue
            try:
                task = int(row["task_id"])
            except (KeyError, TypeError, ValueError):
                continue
            if wanted is None or task in wanted:
                result.add(task)
        return result

    async def record_template_incident(
        self,
        policy_identity: str,
        environment: str,
        catalog_revision_hash: str,
        template_key_hash: str,
        *,
        subject_hash: str,
        deployment_hash: str,
        task_instance_hash: str,
        task_id: int,
        reason_code: str,
        evidence: Optional[Mapping[str, Any]] = None,
        distinct_subjects_to_quarantine: int = 2,
        distinct_tasks_to_quarantine: int = 2,
        created_at: Optional[int] = None,
        ttl_days: int = 30,
    ) -> Dict[str, Any]:
        """Record one harness-owned incident and return family health.

        A single subject only creates subject-scoped evidence.  An exact task
        becomes globally invalid after that same task reproduces across the
        configured number of independent subjects.  The wider family is
        quarantined only when both independent-subject and independent-task
        thresholds are met.  Deployment identity is evidence, never a
        partition boundary, so the aggregate survives rotations.
        """

        if int(ttl_days) <= 0:
            raise ValueError("ttl_days must be positive")
        subject_threshold = self._bounded_threshold(
            distinct_subjects_to_quarantine,
            "distinct_subjects_to_quarantine",
        )
        task_threshold = self._bounded_threshold(
            distinct_tasks_to_quarantine,
            "distinct_tasks_to_quarantine",
        )
        template = self._digest(template_key_hash, "template_key_hash")
        subject = self._digest(subject_hash, "subject_hash")
        deployment = self._digest(deployment_hash, "deployment_hash")
        instance = self._digest(task_instance_hash, "task_instance_hash")
        task = self._non_negative_int(task_id, "task_id")
        write_now = int(time.time())
        timestamp = write_now if created_at is None else int(created_at)

        pk = self.make_template_partition_key(
            policy_identity,
            environment,
            catalog_revision_hash,
            template,
        )
        incident_item = {
            "pk": pk,
            "sk": f"INCIDENT#SUBJECT#{subject}#TASK#{instance}",
            "subject_hash": subject,
            "deployment_hash": deployment,
            "task_instance_hash": instance,
            "task_id": task,
            "template_key_hash": template,
            "classification": "harness_invalid",
            "reason_code": self._safe_code(reason_code) or "template_incident",
            "evidence": self._sanitize_evidence(evidence),
            "created_at": timestamp,
            "ttl": max(write_now, timestamp) + int(ttl_days) * 86400,
        }
        incident_recorded = True
        try:
            await get_client().put_item(
                TableName=self.table_name,
                Item=self._serialize(incident_item),
                **self._expiring_put_guard(write_now),
            )
        except ClientError as error:
            if not self._conditional_failure(error):
                raise
            incident_recorded = False

        rows = await self._query_rows(pk=pk, sk_prefix="INCIDENT#")
        active_rows = [
            row for row in rows
            if int(row.get("ttl") or write_now + 1) > write_now
        ]
        subjects = {
            str(row.get("subject_hash"))
            for row in active_rows
            if row.get("subject_hash")
        }
        tasks = {
            str(row.get("task_instance_hash"))
            for row in active_rows
            if row.get("task_instance_hash")
        }
        exact_task_subjects = {
            str(row.get("subject_hash"))
            for row in active_rows
            if row.get("subject_hash") and row.get("task_id") == task
        }
        status = (
            self.TEMPLATE_STATUS_QUARANTINED
            if len(subjects) >= subject_threshold and len(tasks) >= task_threshold
            else self.TEMPLATE_STATUS_SUSPECTED
        )

        # A task is safe to suppress globally after either the exact task was
        # reproduced by independent subjects or the whole family crossed its
        # independent subject/task thresholds.  In the latter case persist all
        # already-observed task ids, not only the incident that closed the
        # breaker, so the family quarantine actually affects dispatch.
        rows_by_task: Dict[int, Dict[str, Any]] = {}
        for row in active_rows:
            row_task = row.get("task_id")
            if isinstance(row_task, int) and row_task >= 0:
                rows_by_task.setdefault(row_task, row)
        tasks_to_invalidate = (
            set(rows_by_task)
            if status == self.TEMPLATE_STATUS_QUARANTINED
            else ({task} if len(exact_task_subjects) >= subject_threshold else set())
        )
        for invalid_task in sorted(tasks_to_invalidate):
            source = rows_by_task[invalid_task]
            await self._record_invalid_task(
                policy_identity,
                environment,
                catalog_revision_hash,
                task_id=invalid_task,
                template_key_hash=template,
                subject_hash=str(source["subject_hash"]),
                deployment_hash=str(source["deployment_hash"]),
                task_instance_hash=str(source["task_instance_hash"]),
                reason_code=str(source.get("reason_code") or reason_code),
                evidence=(
                    source.get("evidence")
                    if isinstance(source.get("evidence"), Mapping)
                    else evidence
                ),
                created_at=int(source.get("created_at") or timestamp),
                ttl_days=ttl_days,
            )
        health: Dict[str, Any] = {
            "status": status,
            "incident_count": len(active_rows),
            "distinct_subjects": len(subjects),
            "distinct_task_instances": len(tasks),
            "subject_threshold": subject_threshold,
            "task_threshold": task_threshold,
            "template_key_hash": template,
            "updated_at": timestamp,
            "ttl": max(write_now, timestamp) + int(ttl_days) * 86400,
        }
        await self._set_template_health(pk=pk, health=health)
        health["incident_recorded"] = incident_recorded
        return health

    async def _set_template_health(
        self,
        *,
        pk: str,
        health: Mapping[str, Any],
    ) -> None:
        """Monotonically persist suspected -> quarantined template state."""

        status = str(health["status"])
        values = {
            ":status": {"S": status},
            ":incidents": {"N": str(int(health["incident_count"]))},
            ":subjects": {"N": str(int(health["distinct_subjects"]))},
            ":tasks": {"N": str(int(health["distinct_task_instances"]))},
            ":subject_threshold": {"N": str(int(health["subject_threshold"]))},
            ":task_threshold": {"N": str(int(health["task_threshold"]))},
            ":template": {"S": str(health["template_key_hash"])},
            ":now": {"N": str(int(health["updated_at"]))},
            ":ttl": {"N": str(int(health["ttl"]))},
        }
        update = (
            "SET #status = :status, incident_count = :incidents, "
            "distinct_subjects = :subjects, "
            "distinct_task_instances = :tasks, "
            "subject_threshold = :subject_threshold, "
            "task_threshold = :task_threshold, "
            "template_key_hash = :template, updated_at = :now, #ttl = :ttl, "
            "created_at = if_not_exists(created_at, :now)"
        )
        condition = None
        if status == self.TEMPLATE_STATUS_QUARANTINED:
            update += ", quarantined_at = if_not_exists(quarantined_at, :now)"
        else:
            condition = (
                "attribute_not_exists(#status) OR #status <> :quarantined "
                "OR #ttl <= :wall_now"
            )
            values[":quarantined"] = {
                "S": self.TEMPLATE_STATUS_QUARANTINED,
            }
            values[":wall_now"] = {"N": str(int(time.time()))}
        params: Dict[str, Any] = {
            "TableName": self.table_name,
            "Key": self._raw_key(pk, "STATE"),
            "UpdateExpression": update,
            "ExpressionAttributeNames": {
                "#status": "status",
                "#ttl": "ttl",
            },
            "ExpressionAttributeValues": values,
        }
        if condition:
            params["ConditionExpression"] = condition
        try:
            await get_client().update_item(**params)
        except ClientError as error:
            if not self._conditional_failure(error):
                raise
            # A concurrent quarantining writer won; never downgrade it.

    async def get_template_health(
        self,
        policy_identity: str,
        environment: str,
        catalog_revision_hash: str,
        template_key_hash: str,
        *,
        now: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Read current family health, treating TTL-delayed rows as absent."""

        pk = self.make_template_partition_key(
            policy_identity,
            environment,
            catalog_revision_hash,
            template_key_hash,
        )
        response = await get_client().get_item(
            TableName=self.table_name,
            Key=self._raw_key(pk, "STATE"),
            ConsistentRead=True,
        )
        item = response.get("Item")
        if not item:
            return None
        row = self._deserialize(item)
        timestamp = int(time.time()) if now is None else int(now)
        if int(row.get("ttl") or timestamp + 1) <= timestamp:
            return None
        return row

    async def record_model_observation(
        self,
        hotkey: str,
        revision: str,
        policy_version: str,
        deployment_fingerprint: str,
        *,
        classification: str,
        template_key_hash: str,
        task_hash: str,
        attribution: Mapping[str, Any],
        evidence: Optional[Mapping[str, Any]] = None,
        threshold: int = 2,
        created_at: Optional[int] = None,
        ttl_days: int = 30,
    ) -> int:
        """Record one model-owned strike per independent template family."""

        if classification not in self.MODEL_CLASSIFICATIONS:
            raise ValueError("classification must be model-owned")
        required = self._bounded_threshold(threshold, "threshold")
        if int(ttl_days) <= 0:
            raise ValueError("ttl_days must be positive")
        template = self._digest(template_key_hash, "template_key_hash")
        task = self._digest(task_hash, "task_hash")
        structured = self._validate_model_attribution(attribution)
        write_now = int(time.time())
        timestamp = write_now if created_at is None else int(created_at)
        identity = self._identity(
            hotkey, revision, policy_version, deployment_fingerprint,
        )
        merged_evidence = dict(evidence or {})
        merged_evidence.update(structured)
        merged_evidence.update({
            "classification": classification,
            "template_key_hash": template,
            "task_hash": task,
        })
        item = {
            "pk": self.make_partition_key(
                hotkey, revision, policy_version, deployment_fingerprint,
            ),
            "sk": f"MODEL_OBSERVATION#TEMPLATE#{template}",
            **identity,
            "classification": classification,
            "template_key_hash": template,
            "task_hash": task,
            "attribution": structured,
            "evidence": self._sanitize_evidence(merged_evidence),
            "created_at": timestamp,
            "ttl": max(write_now, timestamp) + int(ttl_days) * 86400,
        }
        try:
            await get_client().put_item(
                TableName=self.table_name,
                Item=self._serialize(item),
                **self._expiring_put_guard(write_now),
            )
        except ClientError as error:
            if not self._conditional_failure(error):
                raise

        rows = await self._query_rows(
            pk=self.make_partition_key(
                hotkey, revision, policy_version, deployment_fingerprint,
            ),
            sk_prefix="MODEL_OBSERVATION#TEMPLATE#",
        )
        now = int(time.time())
        active_templates = {
            str(row.get("template_key_hash"))
            for row in rows
            if row.get("template_key_hash")
            and int(row.get("ttl") or now + 1) > now
        }
        return min(required, len(active_templates))

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

        A model-behavior failure and this seal race on the same verdict row.
        Exactly one can win: promotion requires ``passed`` and model failure
        requires the seal to be absent.  Repeating a successful seal is
        idempotent so a scheduler restart can safely finish promotion.
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
        signal allowed to demote a pass is :meth:`fail_model_behavior`, which
        requires structured model ownership across independent templates.
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
            ":expired": {"S": self.STATUS_EXPIRED},
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
            "(#status <> :passed AND #status <> :failed "
            "AND #status <> :expired)"
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

    async def fail_model_behavior(
        self,
        hotkey: str,
        revision: str,
        policy_version: str,
        deployment_fingerprint: str,
        *,
        classification: str,
        reason_code: str,
        attribution: Mapping[str, Any],
        evidence: Optional[Mapping[str, Any]] = None,
        counts: Optional[Mapping[str, Any]] = None,
        minimum_distinct_templates: int = 2,
        now: Optional[int] = None,
    ) -> bool:
        """Atomically fail only on structured, independently reproduced proof.

        Unlike the legacy runtime-invariant path, a harness-invalid sample can
        never enter this method: the classification is model-owned, the request
        is proven to have reached the model on healthy infrastructure/template,
        and durable observations must span distinct template families.
        """

        if classification not in self.MODEL_CLASSIFICATIONS:
            raise ValueError("classification must be model-owned")
        minimum = self._bounded_threshold(
            minimum_distinct_templates, "minimum_distinct_templates",
        )
        structured = self._validate_model_attribution(attribution)
        pk = self.make_partition_key(
            hotkey, revision, policy_version, deployment_fingerprint,
        )
        rows = await self._query_rows(
            pk=pk,
            sk_prefix="MODEL_OBSERVATION#TEMPLATE#",
        )
        timestamp = int(time.time()) if now is None else int(now)
        active_templates = {
            str(row.get("template_key_hash"))
            for row in rows
            if row.get("template_key_hash")
            and int(row.get("ttl") or timestamp + 1) > timestamp
        }
        if len(active_templates) < minimum:
            return False

        identity = self._identity(
            hotkey, revision, policy_version, deployment_fingerprint,
        )
        safe_counts = dict(counts or {})
        safe_counts.update({
            "distinct_template_families": len(active_templates),
            "strikes": len(active_templates),
        })
        merged_evidence = dict(evidence or {})
        merged_evidence.update(structured)
        merged_evidence["classification"] = classification
        values: Dict[str, Dict[str, Any]] = {
            ":failed": {"S": self.STATUS_FAILED},
            ":expired": {"S": self.STATUS_EXPIRED},
            ":reason": {
                "S": self._safe_code(reason_code) or "model_behavior"
            },
            ":source": {"S": "model_behavior"},
            ":counts": self._serialize({
                "value": self._sanitize_counts(safe_counts),
            })["value"],
            ":evidence": self._serialize({
                "value": self._sanitize_evidence(merged_evidence),
            })["value"],
            ":now": {"N": str(timestamp)},
        }
        for name, value in identity.items():
            values[f":{name}"] = {"S": value}

        try:
            await get_client().update_item(
                TableName=self.table_name,
                Key=self._raw_key(pk, "VERDICT"),
                UpdateExpression=(
                    "SET #status = :failed, reason_code = :reason, "
                    "failure_source = :source, counts = :counts, "
                    "evidence = :evidence, updated_at = :now, "
                    "decided_at = :now, model_failed_at = :now, "
                    "created_at = if_not_exists(created_at, :now), "
                    "hotkey = if_not_exists(hotkey, :hotkey), "
                    "revision = if_not_exists(revision, :revision), "
                    "policy_version = if_not_exists(policy_version, :policy_version), "
                    "deployment_fingerprint = "
                    "if_not_exists(deployment_fingerprint, :deployment_fingerprint) "
                    "REMOVE lease_owner, lease_expires_at"
                ),
                ConditionExpression=(
                    "(attribute_not_exists(#status) OR "
                    "(#status <> :failed AND #status <> :expired)) "
                    "AND attribute_not_exists(promotion_sealed_at)"
                ),
                ExpressionAttributeNames={"#status": "status"},
                ExpressionAttributeValues=values,
            )
            return True
        except ClientError as error:
            if not self._conditional_failure(error):
                raise
        existing = await self.get_verdict(
            hotkey, revision, policy_version, deployment_fingerprint,
        )
        return bool(existing and existing.get("status") == self.STATUS_FAILED)
