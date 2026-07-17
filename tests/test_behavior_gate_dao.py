"""Focused storage-contract tests for the behavior gate."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from botocore.exceptions import ClientError

import affine.database.dao.behavior_gate as behavior_gate_mod
import affine.database.tables as tables_mod
from affine.database.dao.behavior_gate import BehaviorGateDAO
from affine.database.schema import BEHAVIOR_GATE_SCHEMA, BEHAVIOR_GATE_TTL


def _conditional_error() -> ClientError:
    return ClientError(
        {"Error": {"Code": "ConditionalCheckFailedException"}},
        "UpdateItem",
    )


class _FakeDynamoClient:
    def __init__(self):
        self.items: dict[tuple[str, str], dict[str, Any]] = {}
        self.put_calls: list[dict[str, Any]] = []
        self.update_calls: list[dict[str, Any]] = []
        self.query_calls: list[dict[str, Any]] = []
        self.update_errors: list[ClientError] = []

    @staticmethod
    def _item_key(item):
        return item["pk"]["S"], item["sk"]["S"]

    async def put_item(self, **kwargs):
        self.put_calls.append(kwargs)
        key = self._item_key(kwargs["Item"])
        if kwargs.get("ConditionExpression") and key in self.items:
            existing = self.items[key]
            values = kwargs.get("ExpressionAttributeValues", {})
            write_now = int(values.get(":write_now", {}).get("N", "-1"))
            existing_ttl = (
                int(existing["ttl"]["N"]) if "ttl" in existing else None
            )
            permits_expired = "#ttl <= :write_now" in kwargs[
                "ConditionExpression"
            ]
            if (
                not permits_expired
                or existing_ttl is None
                or existing_ttl > write_now
            ):
                raise _conditional_error()
        self.items[key] = kwargs["Item"]
        return {}

    async def get_item(self, **kwargs):
        key = kwargs["Key"]["pk"]["S"], kwargs["Key"]["sk"]["S"]
        item = self.items.get(key)
        return {"Item": item} if item else {}

    async def update_item(self, **kwargs):
        self.update_calls.append(kwargs)
        if self.update_errors:
            raise self.update_errors.pop(0)
        update_expression = kwargs.get("UpdateExpression", "")
        if "admission_expired_at" in update_expression:
            key = (
                kwargs["Key"]["pk"]["S"],
                kwargs["Key"]["sk"]["S"],
            )
            existing = self.items.get(key, {})
            values = kwargs["ExpressionAttributeValues"]
            status = existing.get("status", {}).get("S")
            deadline = int(
                existing.get("admission_deadline_at", {}).get("N", "0")
            )
            now = int(values[":now"]["N"])
            if status in {"passed", "failed", "expired"} or deadline > now:
                raise _conditional_error()
            row = dict(existing)
            row.update({
                "status": values[":expired"],
                "reason_code": values[":reason"],
                "admission_expired_at": values[":now"],
                "updated_at": values[":now"],
            })
            row.setdefault("decided_at", values[":now"])
            row.pop("lease_owner", None)
            row.pop("lease_expires_at", None)
            self.items[key] = row
        elif "admission_deadline_at" in update_expression:
            key = (
                kwargs["Key"]["pk"]["S"],
                kwargs["Key"]["sk"]["S"],
            )
            existing = self.items.get(key, {})
            if "admission_deadline_at" in existing:
                raise _conditional_error()
            row = dict(existing)
            row["admission_deadline_at"] = kwargs[
                "ExpressionAttributeValues"
            ][":deadline"]
            self.items[key] = row
        elif "promotion_sealed_at" in update_expression:
            key = (
                kwargs["Key"]["pk"]["S"],
                kwargs["Key"]["sk"]["S"],
            )
            existing = self.items.get(key, {})
            if existing.get("status", {}).get("S") != "passed":
                raise _conditional_error()
            values = kwargs["ExpressionAttributeValues"]
            row = dict(existing)
            row.setdefault("promotion_sealed_at", values[":now"])
            row["updated_at"] = values[":now"]
            self.items[key] = row
        elif kwargs["Key"]["sk"]["S"] == "STATE":
            key = (
                kwargs["Key"]["pk"]["S"],
                kwargs["Key"]["sk"]["S"],
            )
            existing = self.items.get(key, {})
            values = kwargs["ExpressionAttributeValues"]
            incoming = values[":status"]["S"]
            if (
                kwargs.get("ConditionExpression")
                and existing.get("status", {}).get("S") == "quarantined"
                and incoming != "quarantined"
                and int(existing.get("ttl", {}).get("N", "0"))
                > int(values.get(":wall_now", {}).get("N", "-1"))
            ):
                raise _conditional_error()
            row = dict(existing)
            row.update({
                "pk": kwargs["Key"]["pk"],
                "sk": kwargs["Key"]["sk"],
                "status": values[":status"],
                "incident_count": values[":incidents"],
                "distinct_subjects": values[":subjects"],
                "distinct_task_instances": values[":tasks"],
                "subject_threshold": values[":subject_threshold"],
                "task_threshold": values[":task_threshold"],
                "template_key_hash": values[":template"],
                "updated_at": values[":now"],
                "ttl": values[":ttl"],
            })
            row.setdefault("created_at", values[":now"])
            if incoming == "quarantined":
                row.setdefault("quarantined_at", values[":now"])
            self.items[key] = row
        elif "failure_source = :source" in update_expression:
            key = (
                kwargs["Key"]["pk"]["S"],
                kwargs["Key"]["sk"]["S"],
            )
            existing = self.items.get(key, {})
            if (
                existing.get("status", {}).get("S") == "failed"
                or "promotion_sealed_at" in existing
            ):
                raise _conditional_error()
            values = kwargs["ExpressionAttributeValues"]
            row = dict(existing)
            row.update({
                "pk": kwargs["Key"]["pk"],
                "sk": kwargs["Key"]["sk"],
                "status": values[":failed"],
                "reason_code": values[":reason"],
                "failure_source": values[":source"],
                "counts": values[":counts"],
                "evidence": values[":evidence"],
                "updated_at": values[":now"],
                "decided_at": values[":now"],
            })
            if "model_failed_at" in update_expression:
                row["model_failed_at"] = values[":now"]
            for field in (
                "hotkey",
                "revision",
                "policy_version",
                "deployment_fingerprint",
            ):
                row.setdefault(field, values[f":{field}"])
            row.setdefault("created_at", values[":now"])
            row.pop("lease_owner", None)
            row.pop("lease_expires_at", None)
            self.items[key] = row
        return {}

    async def query(self, **kwargs):
        self.query_calls.append(kwargs)
        pk = kwargs["ExpressionAttributeValues"][":pk"]["S"]
        prefix = kwargs["ExpressionAttributeValues"].get(":sk", {}).get("S")
        rows = [
            item
            for (item_pk, item_sk), item in self.items.items()
            if item_pk == pk and (prefix is None or item_sk.startswith(prefix))
        ]
        limit = kwargs.get("Limit")
        if limit is not None:
            rows = rows[:limit]
        if kwargs.get("Select") == "COUNT":
            return {"Count": len(rows), "ScannedCount": len(rows)}
        return {"Items": rows}


@pytest.fixture
def fake_client(monkeypatch):
    client = _FakeDynamoClient()
    monkeypatch.setattr(behavior_gate_mod, "get_client", lambda: client)
    return client


def test_behavior_gate_schema_has_composite_key_and_attempt_ttl():
    assert BEHAVIOR_GATE_SCHEMA["TableName"].endswith("_behavior_gate")
    assert BEHAVIOR_GATE_SCHEMA["KeySchema"] == [
        {"AttributeName": "pk", "KeyType": "HASH"},
        {"AttributeName": "sk", "KeyType": "RANGE"},
    ]
    assert BEHAVIOR_GATE_TTL == {"AttributeName": "ttl"}


@pytest.mark.asyncio
async def test_init_tables_registers_behavior_gate_with_ttl(monkeypatch):
    calls = []

    async def fake_create_table(schema, ttl_attribute=None):
        calls.append((schema["TableName"], ttl_attribute))

    monkeypatch.setattr(tables_mod, "create_table", fake_create_table)

    await tables_mod.init_tables()

    assert (
        BEHAVIOR_GATE_SCHEMA["TableName"],
        BEHAVIOR_GATE_TTL["AttributeName"],
    ) in calls


def test_partition_key_is_scoped_to_policy_and_deployment():
    first = BehaviorGateDAO.make_partition_key(
        "hot#key", "rev", "v1", "deployment-a",
    )
    second = BehaviorGateDAO.make_partition_key(
        "hot#key", "rev", "v1", "deployment-b",
    )

    assert first != second
    assert "hot%23key" in first
    assert "#POLICY#v1#DEPLOY#deployment-a" in first


def test_policy_v2_exposes_no_harness_invalid_to_failed_backdoor():
    dao = BehaviorGateDAO()

    assert not hasattr(dao, "record_runtime_invariant_observation")
    assert not hasattr(dao, "fail_runtime_invariant")


@pytest.mark.asyncio
async def test_ensure_pending_is_conditional_and_preserves_existing(fake_client):
    dao = BehaviorGateDAO()
    created = await dao.ensure_pending("hk", "rev", "v1", "dep", now=100)
    existing = await dao.ensure_pending("hk", "rev", "v1", "dep", now=200)

    assert created["status"] == "pending"
    assert existing["created_at"] == 100
    assert existing["updated_at"] == 100
    assert fake_client.put_calls[0]["ConditionExpression"] == (
        "attribute_not_exists(pk) AND attribute_not_exists(sk)"
    )


@pytest.mark.asyncio
async def test_ensure_pending_anchors_and_backfills_deadline_without_extension(
    fake_client,
):
    dao = BehaviorGateDAO()
    created = await dao.ensure_pending(
        "hk", "rev", "v1", "new", now=100, admission_deadline_at=400,
    )
    repeated = await dao.ensure_pending(
        "hk", "rev", "v1", "new", now=200, admission_deadline_at=900,
    )
    assert created["admission_deadline_at"] == 400
    assert repeated["admission_deadline_at"] == 400

    # Simulate a row written by v1, then let v2 atomically backfill it.  A
    # subsequent coordinator cannot move the deadline farther into the future.
    legacy = await dao.ensure_pending("hk", "rev", "v1", "legacy", now=100)
    assert "admission_deadline_at" not in legacy
    backfilled = await dao.ensure_pending(
        "hk", "rev", "v1", "legacy", now=200, admission_deadline_at=500,
    )
    not_extended = await dao.ensure_pending(
        "hk", "rev", "v1", "legacy", now=300, admission_deadline_at=800,
    )
    assert backfilled["admission_deadline_at"] == 500
    assert not_extended["admission_deadline_at"] == 500
    deadline_updates = [
        call for call in fake_client.update_calls
        if "admission_deadline_at" in call.get("UpdateExpression", "")
    ]
    assert all(
        "if_not_exists(admission_deadline_at, :deadline)"
        in call["UpdateExpression"]
        for call in deadline_updates
    )
    assert all(
        call["ConditionExpression"]
        == "attribute_not_exists(admission_deadline_at)"
        for call in deadline_updates
    )


@pytest.mark.asyncio
async def test_admission_expiry_is_single_claim_control_final(fake_client):
    dao = BehaviorGateDAO()
    await dao.ensure_pending(
        "hk", "rev", "v2", "dep",
        now=100,
        admission_deadline_at=400,
    )

    assert not await dao.claim_expired_admission(
        "hk", "rev", "v2", "dep", now=399,
    )
    claims = await asyncio.gather(*(
        dao.claim_expired_admission(
            "hk", "rev", "v2", "dep", now=400,
        )
        for _ in range(2)
    ))
    assert claims.count(True) == 1
    assert claims.count(False) == 1

    verdict = await dao.get_verdict("hk", "rev", "v2", "dep")
    assert verdict["status"] == "expired"
    assert verdict["reason_code"] == "deployment_admission_deadline_exceeded"
    assert verdict["admission_expired_at"] == 400
    assert "lease_owner" not in verdict
    assert not await dao.claim_expired_admission(
        "hk", "rev", "v2", "dep", now=500,
    )

    assert ":expired" in fake_client.update_calls[-1][
        "ExpressionAttributeValues"
    ]


@pytest.mark.asyncio
async def test_attempt_write_is_idempotent_and_drops_raw_evidence(fake_client):
    dao = BehaviorGateDAO()
    evidence = {
        "first_action_ms": 91234,
        "response_sha256": "abc123",
        "finish_reason": "tool calls",
        "content": "secret raw completion",
        "tool_arguments": {"nonce": "secret"},
        "exception_message": "secret transport body",
    }

    first = await dao.record_attempt(
        "hk",
        "rev",
        "v1",
        "dep",
        probe_id="probe#1",
        classification="model_no_progress",
        evidence=evidence,
        created_at=1_000,
    )
    duplicate = await dao.record_attempt(
        "hk",
        "rev",
        "v1",
        "dep",
        probe_id="probe#1",
        classification="model_no_progress",
        evidence=evidence,
        created_at=2_000,
    )

    assert first is True
    assert duplicate is False
    item = next(iter(fake_client.items.values()))
    stored = dao._deserialize(item)
    assert stored["sk"] == "ATTEMPT#probe%231"
    assert stored["evidence"] == {
        "first_action_ms": 91234,
        "response_sha256": "abc123",
        "finish_reason": "tool_calls",
    }
    assert stored["ttl"] > stored["created_at"]


@pytest.mark.asyncio
async def test_expired_ttl_row_can_be_atomically_replaced(
    fake_client,
    monkeypatch,
):
    monkeypatch.setattr(behavior_gate_mod.time, "time", lambda: 100)
    dao = BehaviorGateDAO()

    assert await dao.record_attempt(
        "hk", "rev", "v2", "dep",
        probe_id="probe-1",
        classification="clean",
        created_at=100,
        ttl_days=1,
    )
    key = next(
        key for key in fake_client.items
        if key[1] == "ATTEMPT#probe-1"
    )
    fake_client.items[key]["ttl"] = {"N": "99"}

    assert await dao.record_attempt(
        "hk", "rev", "v2", "dep",
        probe_id="probe-1",
        classification="model_no_progress",
        created_at=101,
        ttl_days=1,
    )
    replaced = dao._deserialize(fake_client.items[key])
    assert replaced["classification"] == "model_no_progress"
    assert replaced["created_at"] == 101

    put = fake_client.put_calls[-1]
    assert "#ttl <= :write_now" in put["ConditionExpression"]
    assert put["ExpressionAttributeNames"] == {"#ttl": "ttl"}
    assert put["ExpressionAttributeValues"][":write_now"] == {"N": "100"}


@pytest.mark.asyncio
async def test_lease_write_excludes_final_rows_and_supports_renewal(fake_client):
    dao = BehaviorGateDAO()
    assert await dao.acquire_lease(
        "hk",
        "rev",
        "v1",
        "dep",
        owner_token="worker-1",
        lease_seconds=30,
        now=100,
    )
    acquire = fake_client.update_calls[-1]
    assert "#status <> :passed" in acquire["ConditionExpression"]
    assert "#status <> :failed" in acquire["ConditionExpression"]
    assert "lease_expires_at <= :now" in acquire["ConditionExpression"]
    assert acquire["ExpressionAttributeValues"][":expires"] == {"N": "130"}

    assert await dao.renew_lease(
        "hk",
        "rev",
        "v1",
        "dep",
        owner_token="worker-1",
        lease_seconds=30,
        now=110,
    )
    renew = fake_client.update_calls[-1]
    assert renew["ConditionExpression"] == (
        "lease_owner = :owner AND #status = :running "
        "AND lease_expires_at > :now"
    )
    assert renew["ExpressionAttributeValues"][":expires"] == {"N": "140"}
    assert renew["ExpressionAttributeValues"][":now"] == {"N": "110"}

    fake_client.update_errors.append(_conditional_error())
    assert not await dao.acquire_lease(
        "hk",
        "rev",
        "v1",
        "dep",
        owner_token="worker-2",
        lease_seconds=30,
        now=111,
    )


@pytest.mark.asyncio
async def test_final_verdict_is_frozen_and_same_result_is_idempotent(fake_client):
    dao = BehaviorGateDAO()
    assert await dao.set_verdict(
        "hk",
        "rev",
        "v1",
        "dep",
        status="passed",
        reason_code="two clean probes",
        counts={"clean": 2, "strong_failures": 0},
        evidence={"content": "must not persist", "stream_completed": True},
        owner_token="worker-1",
        now=500,
    )
    update = fake_client.update_calls[-1]
    assert "#status <> :passed" in update["ConditionExpression"]
    assert "#status <> :failed" in update["ConditionExpression"]
    assert "lease_owner = :owner" in update["ConditionExpression"]
    assert "lease_expires_at > :now" in update["ConditionExpression"]
    assert "decided_at = if_not_exists(decided_at, :now)" in (
        update["UpdateExpression"]
    )
    evidence = dao._deserialize({
        "evidence": update["ExpressionAttributeValues"][":evidence"],
    })["evidence"]
    assert evidence == {"stream_completed": True}

    # Simulate an already committed pass whose response was lost.  A retry
    # reports success but the conditional update cannot mutate the row.
    key = dao._key("hk", "rev", "v1", "dep", "VERDICT")
    fake_client.items[(key["pk"]["S"], key["sk"]["S"])] = dao._serialize({
        "pk": key["pk"]["S"],
        "sk": "VERDICT",
        "status": "passed",
        "decided_at": 500,
    })
    fake_client.update_errors.append(_conditional_error())
    assert await dao.set_verdict(
        "hk",
        "rev",
        "v1",
        "dep",
        status="passed",
        reason_code="retry",
        now=999,
    )
    verdict = await dao.get_verdict("hk", "rev", "v1", "dep")
    assert verdict["decided_at"] == 500

    fake_client.update_errors.append(_conditional_error())
    assert not await dao.set_verdict(
        "hk",
        "rev",
        "v1",
        "dep",
        status="failed",
        reason_code="cannot overwrite pass",
        now=1_000,
    )


@pytest.mark.asyncio
async def test_invalid_sample_completion_survives_deployment_and_policy_rotation(
    fake_client,
):
    dao = BehaviorGateDAO()
    template = "a" * 64

    assert await dao.record_invalid_sample(
        "hk",
        "rev",
        "TERMINAL",
        900,
        101,
        reason_code="positive_score_zero_activity",
        template_key_hash=template,
        evidence={
            "score": 1.0,
            "llm_call_count": 0,
            "content": "must not be persisted",
        },
        created_at=100,
    )
    assert not await dao.record_invalid_sample(
        "hk",
        "rev",
        "TERMINAL",
        900,
        101,
        reason_code="duplicate",
        template_key_hash=template,
        created_at=200,
    )
    assert await dao.record_invalid_sample(
        "hk",
        "rev",
        "TERMINAL",
        900,
        102,
        reason_code="positive_score_zero_activity",
        created_at=100,
    )

    assert await dao.list_invalid_sample_task_ids(
        "hk", "rev", "TERMINAL", 900,
    ) == {101, 102}
    assert await dao.list_invalid_sample_task_ids(
        "hk", "rev", "TERMINAL", 900, task_ids=[102, 999],
    ) == {102}
    assert await dao.list_invalid_sample_task_ids(
        "hk", "rev", "TERMINAL", 901,
    ) == set()

    rows = [
        dao._deserialize(item)
        for item in fake_client.items.values()
        if item["sk"]["S"].startswith("INVALID_SAMPLE#")
    ]
    assert len(rows) == 2
    assert all("POLICY#" not in row["pk"] for row in rows)
    assert all("DEPLOY#" not in row["pk"] for row in rows)
    assert rows[0]["sample_status"] == "invalid"
    assert "score" not in rows[0]
    assert "content" not in rows[0]["evidence"]


@pytest.mark.asyncio
async def test_template_breaker_requires_cross_subject_evidence_for_global_task(
    fake_client,
):
    dao = BehaviorGateDAO()
    catalog = "1" * 64
    family = "2" * 64

    first = await dao.record_template_incident(
        "policy-v2",
        "TERMINAL",
        catalog,
        family,
        subject_hash="3" * 64,
        deployment_hash="4" * 64,
        task_instance_hash="5" * 64,
        task_id=101,
        reason_code="positive_score_zero_activity",
        created_at=100,
    )
    assert first["status"] == "suspected"
    assert first["distinct_subjects"] == 1
    assert first["distinct_task_instances"] == 1
    # One subject proves only that subject's sample was invalid; a transient
    # telemetry failure must not suppress the task for every UID.
    assert await dao.list_invalid_task_ids(
        "policy-v2", "TERMINAL", catalog,
    ) == set()

    same_subject_new_task = await dao.record_template_incident(
        "policy-v2",
        "TERMINAL",
        catalog,
        family,
        subject_hash="3" * 64,
        deployment_hash="6" * 64,
        task_instance_hash="7" * 64,
        task_id=102,
        reason_code="positive_score_zero_activity",
        created_at=101,
    )
    assert same_subject_new_task["status"] == "suspected"
    assert same_subject_new_task["distinct_subjects"] == 1
    assert same_subject_new_task["distinct_task_instances"] == 2
    assert await dao.list_invalid_task_ids(
        "policy-v2", "TERMINAL", catalog,
    ) == set()

    second_subject = await dao.record_template_incident(
        "policy-v2",
        "TERMINAL",
        catalog,
        family,
        subject_hash="8" * 64,
        deployment_hash="9" * 64,
        task_instance_hash="7" * 64,
        task_id=102,
        reason_code="positive_score_zero_activity",
        created_at=102,
    )
    assert second_subject["status"] == "quarantined"
    assert second_subject["distinct_subjects"] == 2
    assert second_subject["distinct_task_instances"] == 2
    assert await dao.list_invalid_task_ids(
        "policy-v2", "TERMINAL", catalog,
    ) == {101, 102}
    assert await dao.list_invalid_task_ids(
        "policy-v2", "TERMINAL", catalog, task_ids=[102, 999],
    ) == {102}

    health = await dao.get_template_health(
        "policy-v2", "TERMINAL", catalog, family,
    )
    assert health is not None
    assert health["status"] == "quarantined"
    assert health["distinct_subjects"] == 2
    assert health["distinct_task_instances"] == 2
    assert health["quarantined_at"] == 102

    state_updates = [
        call for call in fake_client.update_calls
        if call["Key"]["sk"]["S"] == "STATE"
    ]
    assert state_updates
    assert all("#ttl = :ttl" in call["UpdateExpression"] for call in state_updates)
    assert all(
        call["ExpressionAttributeNames"]["#ttl"] == "ttl"
        for call in state_updates
    )

    # Catalog revision is part of both task and family scope, so a fixed image
    # does not inherit stale quarantine state from the previous window.
    other_catalog = "a" * 64
    assert await dao.list_invalid_task_ids(
        "policy-v2", "TERMINAL", other_catalog,
    ) == set()
    assert await dao.get_template_health(
        "policy-v2", "TERMINAL", other_catalog, family,
    ) is None


def _model_attribution(**overrides):
    value = {
        "failure_owner": "model",
        "request_dispatched": True,
        "request_reached_model": True,
        "endpoint_healthy": True,
        "template_healthy": True,
    }
    value.update(overrides)
    return value


@pytest.mark.asyncio
async def test_model_observations_require_structured_ownership_and_distinct_families(
    fake_client,
):
    dao = BehaviorGateDAO()

    with pytest.raises(ValueError, match="classification must be model-owned"):
        await dao.record_model_observation(
            "hk", "rev", "v2", "dep",
            classification="harness_invalid",
            template_key_hash="1" * 64,
            task_hash="2" * 64,
            attribution=_model_attribution(),
        )
    with pytest.raises(ValueError, match="request_reached_model"):
        await dao.record_model_observation(
            "hk", "rev", "v2", "dep",
            classification="model_no_progress",
            template_key_hash="1" * 64,
            task_hash="2" * 64,
            attribution=_model_attribution(request_reached_model=False),
        )

    first = await dao.record_model_observation(
        "hk", "rev", "v2", "dep",
        classification="model_no_progress",
        template_key_hash="1" * 64,
        task_hash="2" * 64,
        attribution=_model_attribution(),
        evidence={
            "deadline_exceeded": True,
            "progress_deadline_ms": 90_000,
            "message": "raw model text",
        },
    )
    duplicate_family = await dao.record_model_observation(
        "hk", "rev", "v2", "dep",
        classification="model_protocol_failure",
        template_key_hash="1" * 64,
        task_hash="3" * 64,
        attribution=_model_attribution(),
    )
    assert (first, duplicate_family) == (1, 1)
    assert not await dao.fail_model_behavior(
        "hk", "rev", "v2", "dep",
        classification="model_no_progress",
        reason_code="model_progress_deadline",
        attribution=_model_attribution(),
    )

    second = await dao.record_model_observation(
        "hk", "rev", "v2", "dep",
        classification="model_protocol_failure",
        template_key_hash="4" * 64,
        task_hash="5" * 64,
        attribution=_model_attribution(
            endpoint_healthy=None,
            endpoint_health="healthy",
            template_healthy=None,
            template_health="healthy",
        ),
    )
    assert second == 2
    assert await dao.fail_model_behavior(
        "hk", "rev", "v2", "dep",
        classification="model_no_progress",
        reason_code="model_progress_deadline",
        attribution=_model_attribution(),
        evidence={"message": "must not persist"},
        now=500,
    )

    verdict = await dao.get_verdict("hk", "rev", "v2", "dep")
    assert verdict["status"] == "failed"
    assert verdict["failure_source"] == "model_behavior"
    assert verdict["model_failed_at"] == 500
    assert verdict["counts"]["distinct_template_families"] == 2
    assert verdict["counts"]["strikes"] == 2
    assert verdict["evidence"]["failure_owner"] == "model"
    assert verdict["evidence"]["request_reached_model"] is True
    assert "message" not in verdict["evidence"]


@pytest.mark.asyncio
async def test_model_failure_cannot_cross_promotion_seal(fake_client):
    dao = BehaviorGateDAO()
    for index, template in enumerate(("a" * 64, "b" * 64)):
        await dao.record_model_observation(
            "hk", "rev", "v2", "dep",
            classification="model_no_progress",
            template_key_hash=template,
            task_hash=str(index + 1) * 64,
            attribution=_model_attribution(),
        )

    key = dao._key("hk", "rev", "v2", "dep", "VERDICT")
    item_key = key["pk"]["S"], key["sk"]["S"]
    fake_client.items[item_key] = dao._serialize({
        "pk": item_key[0],
        "sk": item_key[1],
        "status": "passed",
        "promotion_sealed_at": 400,
    })
    fake_client.update_errors.append(_conditional_error())

    assert not await dao.fail_model_behavior(
        "hk", "rev", "v2", "dep",
        classification="model_no_progress",
        reason_code="too_late",
        attribution=_model_attribution(),
        now=500,
    )
    verdict = await dao.get_verdict("hk", "rev", "v2", "dep")
    assert verdict["status"] == "passed"
    assert verdict["promotion_sealed_at"] == 400
