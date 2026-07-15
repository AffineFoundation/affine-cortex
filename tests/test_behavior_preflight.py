from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from affine.src.behavior_guard.models import (
    ProbeClassification,
    ProbeResult,
    parse_behavior_gate_config,
)
from affine.src.executor.preflight import BehaviorPreflightCoordinator
from affine.src.scorer.window_state import (
    BattleRecord,
    DeploymentRecord,
    MinerSnapshot,
)


class _State:
    def __init__(self, record, config):
        self.record = record
        self.config = config

    async def get_behavior_gate_config(self):
        return self.config

    async def get_battle(self):
        return self.record

    async def get_predeployed_challengers(self):
        return []


class _DAO:
    def __init__(self):
        self.row = None
        self.attempts = []
        self.owner = None

    async def get_verdict(self, *_args):
        return self.row

    async def ensure_pending(self, *_args):
        if self.row is None:
            self.row = {"status": "pending", "updated_at": 0}
        return self.row

    async def acquire_lease(self, *_args, owner_token, lease_seconds):
        del lease_seconds
        if self.owner is not None:
            return False
        self.owner = owner_token
        self.row = {"status": "running", "updated_at": 0}
        return True

    async def renew_lease(self, *_args, owner_token, lease_seconds):
        del lease_seconds
        return self.owner == owner_token

    async def release_lease(self, *_args, owner_token):
        if self.owner != owner_token:
            return False
        self.owner = None
        return True

    async def record_attempt(
        self, *_args, probe_id, classification, evidence, owner_token,
    ):
        self.attempts.append({
            "probe_id": probe_id,
            "classification": classification,
            "evidence": evidence,
            "owner_token": owner_token,
        })
        return True

    async def set_verdict(
        self,
        *_args,
        status,
        reason_code,
        counts,
        evidence,
        owner_token,
    ):
        assert self.owner == owner_token
        self.row = {
            "status": status,
            "reason_code": reason_code,
            "counts": counts,
            "evidence": evidence,
            "updated_at": 1,
        }
        self.owner = None
        return True


class _ProbeClient:
    def __init__(self, results, concurrency, *, delay=0):
        self.results = list(results)
        self.concurrency = concurrency
        self.delay = delay

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def _next(self, probe_id, *, is_action):
        self.concurrency["active"] += 1
        self.concurrency["max"] = max(
            self.concurrency["max"], self.concurrency["active"]
        )
        try:
            await asyncio.sleep(self.delay)
            classification = self.results.pop(0)
            return ProbeResult(
                probe_id=probe_id,
                classification=classification,
                reason=f"test_{classification.value}",
                duration_ms=1,
                first_response_ms=1,
                first_action_ms=(
                    1
                    if is_action
                    and classification is ProbeClassification.CLEAN
                    else None
                ),
                completion_tokens=1,
                output_bytes=1,
                evidence_hash="a" * 64,
            )
        finally:
            self.concurrency["active"] -= 1

    async def run_control_probe(self, *, probe_id):
        return await self._next(probe_id, is_action=False)

    async def run_action_probe(self, *, probe_id):
        return await self._next(probe_id, is_action=True)


def _record(*, two_endpoints=False):
    deployments = []
    if two_endpoints:
        deployments = [
            DeploymentRecord("gpu-a", "dep-a", "https://gpu-a.invalid/v1"),
            DeploymentRecord("gpu-b", "dep-b", "https://gpu-b.invalid/v1"),
        ]
    return BattleRecord(
        challenger=MinerSnapshot(
            uid=208,
            hotkey="uid208-hotkey",
            revision="uid208-revision",
            model="org/uid208-model",
        ),
        deployment_id="dep-a",
        base_url="https://gpu-a.invalid/v1",
        started_at_block=1,
        deployments=deployments,
    )


def _config(**overrides):
    value = {
        "enabled": True,
        "mode": "enforce",
        "policy_version": "test-v1",
        "probe_count": 3,
        "clean_to_pass": 2,
        "violations_to_fail": 2,
        "max_infra_retries": 2,
        "probe_timeout_seconds": 1,
        "first_response_deadline_seconds": 0.5,
        "first_action_deadline_seconds": 0.75,
    }
    value.update(overrides)
    return value


def _coordinator(record, outcomes):
    dao = _DAO()
    concurrency = {"active": 0, "max": 0}
    by_url = {
        "https://gpu-a.invalid/v1": list(outcomes),
        "https://gpu-b.invalid/v1": list(outcomes),
    }

    def factory(**kwargs):
        return _ProbeClient(by_url[kwargs["base_url"]], concurrency)

    coordinator = BehaviorPreflightCoordinator(
        state=_State(record, _config()),
        dao=dao,
        probe_client_factory=factory,
    )
    return coordinator, dao, concurrency


@pytest.mark.asyncio
async def test_preflight_passes_only_after_every_serving_endpoint_passes():
    coordinator, dao, concurrency = _coordinator(
        _record(two_endpoints=True),
        [
            ProbeClassification.CLEAN,
            ProbeClassification.CLEAN,
            ProbeClassification.CLEAN,
        ],
    )

    await coordinator.run_once()

    assert dao.row["status"] == "passed"
    assert dao.row["counts"]["admissible"] == 6
    assert len(dao.attempts) == 6
    assert concurrency["max"] == 1


@pytest.mark.asyncio
async def test_uid208_empty_protocol_pattern_fails_after_two_short_probes():
    coordinator, dao, _concurrency = _coordinator(
        _record(),
        [
            ProbeClassification.MODEL_PROTOCOL_FAILURE,
            ProbeClassification.MODEL_PROTOCOL_FAILURE,
        ],
    )

    await coordinator.run_once()

    assert dao.row["status"] == "failed"
    assert dao.row["counts"]["strikes"] == 2
    assert len(dao.attempts) == 2


@pytest.mark.asyncio
async def test_infrastructure_failures_are_deferred_and_never_become_strikes():
    coordinator, dao, _concurrency = _coordinator(
        _record(),
        [ProbeClassification.INFRA_FAILURE] * 5,
    )

    await coordinator.run_once()

    assert dao.row["status"] == "deferred"
    assert dao.row["counts"]["infra_failure"] == 5
    assert dao.row["counts"]["strikes"] == 0
    assert len(dao.attempts) == 5


@pytest.mark.asyncio
async def test_bounded_refusals_do_not_strike_but_cannot_satisfy_action_proof():
    coordinator, dao, _concurrency = _coordinator(
        _record(),
        [ProbeClassification.QUALITY_FAILURE] * 3,
    )

    await coordinator.run_once()

    assert dao.row["status"] == "suspected"
    assert dao.row["reason_code"] == "action_proof_missing"
    assert dao.row["counts"]["strikes"] == 0
    assert len(dao.attempts) == 3


@pytest.mark.asyncio
async def test_repeated_missing_action_rounds_converge_to_failed():
    coordinator, dao, _concurrency = _coordinator(
        _record(),
        [ProbeClassification.QUALITY_FAILURE] * 3,
    )

    await coordinator.run_once()
    await coordinator.run_once()

    assert dao.row["status"] == "failed"
    assert dao.row["reason_code"].startswith("repeated_suspected_rounds:")
    assert dao.row["counts"]["suspected_rounds"] == 2


@pytest.mark.asyncio
async def test_one_strike_in_each_round_converges_to_failed():
    coordinator, dao, _concurrency = _coordinator(
        _record(),
        [
            ProbeClassification.CLEAN,
            ProbeClassification.MODEL_PROTOCOL_FAILURE,
            ProbeClassification.CLEAN,
        ],
    )

    await coordinator.run_once()
    await coordinator.run_once()

    assert dao.row["status"] == "failed"
    assert dao.row["counts"]["suspected_rounds"] == 2


@pytest.mark.asyncio
async def test_suspected_rounds_survive_an_intervening_deferred_round():
    record = _record()
    dao = _DAO()
    concurrency = {"active": 0, "max": 0}
    rounds = [
        [ProbeClassification.QUALITY_FAILURE] * 3,
        [ProbeClassification.INFRA_FAILURE] * 5,
        [ProbeClassification.QUALITY_FAILURE] * 3,
    ]

    def factory(**_kwargs):
        return _ProbeClient(rounds.pop(0), concurrency)

    coordinator = BehaviorPreflightCoordinator(
        state=_State(record, _config()),
        dao=dao,
        probe_client_factory=factory,
    )

    await coordinator.run_once()
    assert dao.row["status"] == "suspected"
    assert dao.row["counts"]["suspected_rounds"] == 1

    await coordinator.run_once()
    assert dao.row["status"] == "deferred"
    assert dao.row["counts"]["suspected_rounds"] == 1
    assert dao.row["counts"]["deferred_rounds"] == 1

    await coordinator.run_once()
    assert dao.row["status"] == "failed"
    assert dao.row["reason_code"].startswith("repeated_suspected_rounds:")
    assert dao.row["counts"]["suspected_rounds"] == 2
    assert dao.row["counts"]["deferred_rounds"] == 1


@pytest.mark.asyncio
async def test_each_endpoint_gets_its_own_bounded_action_proof_budget():
    record = _record(two_endpoints=True)
    dao = _DAO()
    concurrency = {"active": 0, "max": 0}
    outcomes = [ProbeClassification.CLEAN] * 3

    def factory(**_kwargs):
        return _ProbeClient(outcomes, concurrency, delay=0.02)

    coordinator = BehaviorPreflightCoordinator(
        state=_State(record, _config()),
        dao=dao,
        probe_client_factory=factory,
    )
    config = replace(
        parse_behavior_gate_config(_config()),
        admission_timeout_seconds=0.09,
    )

    await asyncio.wait_for(
        coordinator._ensure_verdict(record, config, "slow-clean-endpoints"),
        timeout=0.5,
    )

    assert dao.row["status"] == "passed"
    assert len(dao.attempts) == 6
    assert concurrency["max"] == 1


@pytest.mark.asyncio
async def test_all_slow_malicious_endpoints_converge_with_per_endpoint_budgets():
    record = _record(two_endpoints=True)
    dao = _DAO()
    concurrency = {"active": 0, "max": 0}
    outcomes = [ProbeClassification.MODEL_PROTOCOL_FAILURE] * 2

    def factory(**_kwargs):
        return _ProbeClient(outcomes, concurrency, delay=0.02)

    coordinator = BehaviorPreflightCoordinator(
        state=_State(record, _config()),
        dao=dao,
        probe_client_factory=factory,
    )
    config = replace(
        parse_behavior_gate_config(_config()),
        admission_timeout_seconds=0.06,
    )

    await asyncio.wait_for(
        coordinator._ensure_verdict(record, config, "slow-bad-endpoints"),
        timeout=0.5,
    )

    assert dao.row["status"] == "failed"
    assert dao.row["counts"]["strikes"] == 4
    assert len(dao.attempts) == 4
    assert concurrency["max"] == 1


@pytest.mark.asyncio
async def test_multi_endpoint_disagreement_is_infra_deferred_not_model_loss():
    record = _record(two_endpoints=True)
    dao = _DAO()
    concurrency = {"active": 0, "max": 0}
    by_url = {
        "https://gpu-a.invalid/v1": [
            ProbeClassification.MODEL_PROTOCOL_FAILURE,
            ProbeClassification.MODEL_PROTOCOL_FAILURE,
        ],
        "https://gpu-b.invalid/v1": [
            ProbeClassification.CLEAN,
            ProbeClassification.CLEAN,
            ProbeClassification.CLEAN,
        ],
    }

    def factory(**kwargs):
        return _ProbeClient(by_url[kwargs["base_url"]], concurrency)

    coordinator = BehaviorPreflightCoordinator(
        state=_State(record, _config()),
        dao=dao,
        probe_client_factory=factory,
    )

    await coordinator.run_once()

    assert dao.row["status"] == "deferred"
    assert dao.row["reason_code"] == "endpoint_verdict_disagreement"
    assert dao.row["counts"]["strikes"] == 2
    assert len(dao.attempts) == 5
