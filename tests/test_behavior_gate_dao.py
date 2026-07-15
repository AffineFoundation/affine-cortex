"""Focused storage-contract tests for the behavior gate."""

from __future__ import annotations

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
        if "promotion_sealed_at" in update_expression:
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
                "runtime_failed_at": values[":now"],
            })
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
async def test_runtime_observations_count_distinct_tasks_per_signature(fake_client):
    dao = BehaviorGateDAO()
    signature = "a" * 64
    first_task = "b" * 64
    second_task = "c" * 64

    first = await dao.record_runtime_invariant_observation(
        "hk", "rev", "v1", "dep",
        signature_hash=signature,
        task_hash=first_task,
        classification="harness_invalid",
        evidence={"task_id": 101, "terminated_reason": "error_exit_3"},
        threshold=2,
    )
    duplicate = await dao.record_runtime_invariant_observation(
        "hk", "rev", "v1", "dep",
        signature_hash=signature,
        task_hash=first_task,
        classification="harness_invalid",
        evidence={"task_id": 101, "terminated_reason": "error_exit_3"},
        threshold=2,
    )
    second = await dao.record_runtime_invariant_observation(
        "hk", "rev", "v1", "dep",
        signature_hash=signature,
        task_hash=second_task,
        classification="harness_invalid",
        evidence={"task_id": 102, "terminated_reason": "error_exit_3"},
        threshold=2,
    )

    assert (first, duplicate, second) == (1, 1, 2)
    assert len(fake_client.put_calls) == 3
    assert len(fake_client.items) == 2
    count_query = fake_client.query_calls[-1]
    assert count_query["ConsistentRead"] is True
    assert count_query["Select"] == "COUNT"
    assert count_query["Limit"] == 2


@pytest.mark.asyncio
async def test_runtime_timeout_rate_counts_distinct_eligible_tasks(fake_client):
    dao = BehaviorGateDAO()
    environment = "a" * 64

    latest = None
    for index in range(10):
        latest = await dao.record_runtime_timeout_outcome(
            "hk",
            "rev",
            "v1",
            "dep",
            environment_hash=environment,
            task_hash=f"{index + 1:064x}",
            classification=(
                "model_no_progress" if index in {0, 1} else "clean"
            ),
            evidence={
                "timed_out": index in {0, 1},
                "timeout_source": "model" if index in {0, 1} else None,
                "content": "raw output must not be stored",
            },
        )

    duplicate = await dao.record_runtime_timeout_outcome(
        "hk",
        "rev",
        "v1",
        "dep",
        environment_hash=environment,
        task_hash=f"{1:064x}",
        classification="model_no_progress",
    )

    assert latest == {
        "eligible_samples": 10,
        "model_timeout_count": 2,
    }
    assert duplicate == latest
    assert len(fake_client.items) == 10
    rate_query = fake_client.query_calls[-1]
    assert rate_query["ConsistentRead"] is True
    assert rate_query["ProjectionExpression"] == "#classification"
    assert rate_query["ExpressionAttributeNames"] == {
        "#classification": "classification",
    }
    first = dao._deserialize(next(iter(fake_client.items.values())))
    assert first["evidence"] == {
        "timed_out": True,
        "timeout_source": "model",
    }


@pytest.mark.asyncio
async def test_promotion_seal_wins_atomic_race_against_runtime_failure(fake_client):
    dao = BehaviorGateDAO()
    key = dao._key("hk", "rev", "v1", "dep", "VERDICT")
    item_key = key["pk"]["S"], key["sk"]["S"]
    fake_client.items[item_key] = dao._serialize({
        "pk": item_key[0],
        "sk": item_key[1],
        "status": "passed",
        "reason_code": "preflight_passed",
    })

    assert await dao.seal_for_promotion(
        "hk", "rev", "v1", "dep", now=450,
    )
    assert await dao.seal_for_promotion(
        "hk", "rev", "v1", "dep", now=451,
    )
    assert not await dao.fail_runtime_invariant(
        "hk",
        "rev",
        "v1",
        "dep",
        reason_code="too_late",
        evidence={"task_id": 1},
        now=500,
    )

    verdict = await dao.get_verdict("hk", "rev", "v1", "dep")
    assert verdict["status"] == "passed"
    assert verdict["promotion_sealed_at"] == 450
    runtime_update = fake_client.update_calls[-1]
    assert "attribute_not_exists(promotion_sealed_at)" in (
        runtime_update["ConditionExpression"]
    )


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
async def test_runtime_invariant_atomically_overrides_pass_and_clears_lease(
    fake_client,
):
    dao = BehaviorGateDAO()
    key = dao._key("hk", "rev", "v1", "dep", "VERDICT")
    item_key = key["pk"]["S"], key["sk"]["S"]
    fake_client.items[item_key] = dao._serialize({
        "pk": item_key[0],
        "sk": item_key[1],
        "status": "passed",
        "reason_code": "preflight_passed",
        "evidence": {"stream_completed": True},
        "lease_owner": "stale-worker",
        "lease_expires_at": 999,
        "decided_at": 100,
    })

    assert await dao.fail_runtime_invariant(
        "hk",
        "rev",
        "v1",
        "dep",
        reason_code="positive_score_zero_activity",
        evidence={
            "invariant_name": "uid208_zero_activity",
            "score": 1.0,
            "commands_executed": 0,
            "llm_call_count": 0,
            "total_tokens": 0,
            "output_bytes": 0,
            "terminated_reason": "error_exit_3",
            "content": "raw model output must be dropped",
            "tool_arguments": {"secret": True},
        },
        counts={"runtime_invariant_failures": 1},
        now=500,
    )

    verdict = await dao.get_verdict("hk", "rev", "v1", "dep")
    assert verdict["status"] == "failed"
    assert verdict["failure_source"] == "runtime_invariant"
    assert verdict["reason_code"] == "positive_score_zero_activity"
    assert verdict["decided_at"] == 500
    assert verdict["runtime_failed_at"] == 500
    assert "lease_owner" not in verdict
    assert "lease_expires_at" not in verdict
    assert verdict["counts"] == {"runtime_invariant_failures": 1}
    assert verdict["evidence"] == {
        "invariant_name": "uid208_zero_activity",
        "score": 1.0,
        "commands_executed": 0,
        "llm_call_count": 0,
        "total_tokens": 0,
        "output_bytes": 0,
        "terminated_reason": "error_exit_3",
    }
    update = fake_client.update_calls[-1]
    assert update["ConditionExpression"] == (
        "(attribute_not_exists(#status) OR #status <> :failed) "
        "AND attribute_not_exists(promotion_sealed_at)"
    )
    assert "REMOVE lease_owner, lease_expires_at" in update["UpdateExpression"]


@pytest.mark.asyncio
async def test_runtime_invariant_missing_row_creates_failed_verdict(fake_client):
    dao = BehaviorGateDAO()

    assert await dao.fail_runtime_invariant(
        "hk",
        "rev",
        "v1",
        "dep",
        reason_code="runtime invariant",
        evidence={"sample_invariant": "zero activity"},
        now=600,
    )

    verdict = await dao.get_verdict("hk", "rev", "v1", "dep")
    assert verdict["status"] == "failed"
    assert verdict["reason_code"] == "runtime_invariant"
    assert verdict["evidence"] == {"sample_invariant": "zero_activity"}
    assert verdict["created_at"] == 600


@pytest.mark.asyncio
async def test_runtime_invariant_concurrent_and_already_failed_are_idempotent(
    fake_client,
):
    import asyncio

    dao = BehaviorGateDAO()
    results = await asyncio.gather(*(
        dao.fail_runtime_invariant(
            "hk",
            "rev",
            "v1",
            "dep",
            reason_code=f"race_{index}",
            evidence={"task_id": index},
            now=700 + index,
        )
        for index in range(2)
    ))

    assert results == [True, True]
    verdict = await dao.get_verdict("hk", "rev", "v1", "dep")
    assert verdict["status"] == "failed"
    first_reason = verdict["reason_code"]
    first_decided_at = verdict["decided_at"]

    # A later duplicate observes FAILED and reports success without mutating
    # the winning call's reason, evidence, or timestamps.
    assert await dao.fail_runtime_invariant(
        "hk",
        "rev",
        "v1",
        "dep",
        reason_code="late_duplicate",
        evidence={"task_id": 999},
        now=999,
    )
    frozen = await dao.get_verdict("hk", "rev", "v1", "dep")
    assert frozen["reason_code"] == first_reason
    assert frozen["decided_at"] == first_decided_at

    # Ordinary preflight writers can never resurrect a runtime-failed row.
    fake_client.update_errors.append(_conditional_error())
    assert not await dao.set_verdict(
        "hk",
        "rev",
        "v1",
        "dep",
        status="passed",
        reason_code="must_not_resurrect",
        now=1_000,
    )
    still_failed = await dao.get_verdict("hk", "rev", "v1", "dep")
    assert still_failed["status"] == "failed"
    assert still_failed["reason_code"] == first_reason
