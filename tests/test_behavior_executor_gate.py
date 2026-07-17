from __future__ import annotations

import asyncio
import time

import pytest

from affine.core.models import Result
from affine.core.environments import EvaluationDeadlineExceeded
from affine.src.executor.worker import ExecutorWorker
from affine.src.scorer.window_state import (
    BattleRecord,
    ChampionRecord,
    EnvConfig,
    MinerSnapshot,
    TaskIdState,
)


class _GateDAO:
    def __init__(
        self,
        status,
        *,
        runtime_results=None,
        invalid_results=None,
        template_results=None,
        model_observation_results=None,
        promotion_sealed_at=None,
        admission_deadline_at=None,
        template_health_status=None,
    ):
        self.status = status
        self.promotion_sealed_at = promotion_sealed_at
        self.admission_deadline_at = admission_deadline_at
        self.template_health_status = template_health_status
        self.calls = 0
        self.runtime_results = list(runtime_results or [])
        self.invalid_results = list(invalid_results or [])
        self.template_results = list(template_results or [])
        self.model_observation_results = list(model_observation_results or [])
        self.ensure_calls = []
        self.invalid_samples = []
        self.template_incidents = []
        self.model_observations = []
        self.model_failures = []
        self._model_templates = {}
        self.subject_invalid_task_ids = set()
        self.global_invalid_task_ids = set()

    async def get_verdict(self, *_args):
        self.calls += 1
        if self.status is None:
            return None
        row = {
            "status": self.status,
            "reason_code": "test",
            "promotion_sealed_at": self.promotion_sealed_at,
        }
        if self.admission_deadline_at is not None:
            row["admission_deadline_at"] = self.admission_deadline_at
        return row

    async def ensure_pending(self, *args, **kwargs):
        self.ensure_calls.append((args, kwargs))
        if self.status is None:
            self.status = "pending"
            self.admission_deadline_at = kwargs.get("admission_deadline_at")
        return await self.get_verdict(*args)

    async def list_invalid_sample_task_ids(self, *_args, **_kwargs):
        return set(self.subject_invalid_task_ids)

    async def list_invalid_task_ids(self, *_args, **_kwargs):
        return set(self.global_invalid_task_ids)

    async def get_template_health(self, *_args, **_kwargs):
        if self.template_health_status is None:
            return None
        return {"status": self.template_health_status}

    async def record_invalid_sample(self, *args, **kwargs):
        self.invalid_samples.append((args, kwargs))
        if self.invalid_results:
            outcome = self.invalid_results.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
        self.subject_invalid_task_ids.add(int(args[4]))
        return True

    async def record_template_incident(self, *args, **kwargs):
        self.template_incidents.append((args, kwargs))
        if self.template_results:
            outcome = self.template_results.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
        self.global_invalid_task_ids.add(int(kwargs["task_id"]))
        return {"status": "suspected"}

    async def record_model_observation(self, *args, **kwargs):
        self.model_observations.append((args, kwargs))
        if self.model_observation_results:
            outcome = self.model_observation_results.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
        templates = self._model_templates.setdefault(tuple(args), set())
        templates.add(kwargs["template_key_hash"])
        return min(len(templates), kwargs["threshold"])

    async def fail_model_behavior(self, *args, **kwargs):
        self.model_failures.append((args, kwargs))
        if self.runtime_results:
            outcome = self.runtime_results.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            if not outcome:
                return False
        self.status = "failed"
        return True


class _State:
    def __init__(self, gate_config):
        self.gate_config = gate_config
        self.champion = ChampionRecord(
            uid=1,
            hotkey="champion",
            revision="champion-revision",
            model="org/champion",
            deployment_id="dep-champion",
            base_url="https://champion.invalid/v1",
        )
        self.battle = BattleRecord(
            challenger=MinerSnapshot(
                uid=208,
                hotkey="challenger",
                revision="challenger-revision",
                model="org/challenger",
            ),
            deployment_id="dep-challenger",
            base_url="https://challenger.invalid/v1",
            started_at_block=1,
        )
        self.predeployed = []
        self.rotation_request = None

    async def get_task_state(self):
        return TaskIdState(task_ids={"TERMINAL": [101]}, refreshed_at_block=10)

    async def get_environments(self):
        return {
            "TERMINAL": EnvConfig(
                display_name="Terminal",
                enabled_for_sampling=True,
                enabled_for_scoring=True,
                sampling_count=1,
                dataset_range=[[0, 10]],
            )
        }

    async def get_champion(self):
        return self.champion

    async def get_battle(self):
        return self.battle

    async def get_predeployed_challengers(self):
        return self.predeployed

    async def get_behavior_gate_config(self):
        return self.gate_config

    async def get_window_rotation_request(self):
        return self.rotation_request

    async def set_window_rotation_request(self, request):
        self.rotation_request = request


class _Samples:
    def __init__(self):
        self.persist_calls = []

    async def read_scores_for_tasks(
        self, hotkey, revision, env, task_ids, *, refresh_block,
    ):
        del revision, env, task_ids, refresh_block
        return {101: 1.0} if hotkey == "champion" else {}

    async def persist(self, **kwargs):
        self.persist_calls.append(kwargs)


def _gate_config(*, mode="enforce", gated_environments=("*",), hold=300):
    return {
        "enabled": True,
        "mode": mode,
        "policy_version": "test-v1",
        "admission_hold_seconds": hold,
        "gated_environments": list(gated_environments),
    }


async def _dispatch_count(
    *, status, mode="enforce", gated_environments=("*",), dao=None,
):
    dao = dao or _GateDAO(status)
    worker = ExecutorWorker(
        worker_id=0, env="TERMINAL", behavior_gate_dao=dao,
    )
    worker._state = _State(_gate_config(
        mode=mode, gated_environments=gated_environments,
    ))
    worker._samples = _Samples()
    launched = []

    async def dispatch_one(**kwargs):
        launched.append(kwargs)

    worker._dispatch_one = dispatch_one
    tasks = set()
    count = await worker._dispatch_new(set(), tasks)
    if tasks:
        await asyncio.gather(*tasks)
    return count, launched, dao, worker


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status", [None, "pending", "running", "suspected", "deferred", "failed"],
)
async def test_enforce_gate_keeps_challenger_dispatch_at_zero_until_passed(status):
    count, launched, dao, _worker = await _dispatch_count(status=status)

    assert count == 0
    assert launched == []
    if status is None:
        assert len(dao.ensure_calls) == 1
        assert dao.admission_deadline_at is not None


@pytest.mark.asyncio
async def test_passed_gate_releases_challenger_sampling():
    count, launched, _dao, _worker = await _dispatch_count(status="passed")

    assert count == 1
    assert len(launched) == 1
    assert launched[0]["miner"].uid == 208


@pytest.mark.asyncio
async def test_shadow_mode_never_holds_sampling():
    count, launched, dao, _worker = await _dispatch_count(
        status="failed", mode="shadow",
    )

    assert count == 1
    assert len(launched) == 1
    assert dao.calls == 0


@pytest.mark.asyncio
async def test_environment_allowlist_leaves_ungated_env_unchanged():
    count, launched, dao, _worker = await _dispatch_count(
        status="failed", gated_environments=("SWE-INFINITE",),
    )

    assert count == 1
    assert len(launched) == 1
    assert dao.calls == 0


@pytest.mark.asyncio
async def test_admission_shadow_waits_for_nonfinal_but_releases_any_final():
    pending = await _dispatch_count(status="running", mode="admission_shadow")
    failed = await _dispatch_count(status="failed", mode="admission_shadow")
    passed = await _dispatch_count(status="passed", mode="admission_shadow")

    assert pending[0] == 0
    assert failed[0] == 1
    assert passed[0] == 1


@pytest.mark.asyncio
async def test_admission_shadow_releases_after_shared_absolute_deadline():
    dao = _GateDAO("running", admission_deadline_at=int(time.time()) - 1)
    count, launched, _dao, _worker = await _dispatch_count(
        status="running", mode="admission_shadow", dao=dao,
    )

    assert count == 1
    assert len(launched) == 1


@pytest.mark.asyncio
async def test_admission_shadow_write_retry_cannot_extend_absolute_hold():
    worker, dao, _samples = await _run_runtime_result(
        mode="admission_shadow",
        extra=_model_behavior_extra("family-a"),
        runtime_results=[False, False],
    )
    await _run_battle_attempt(
        worker, task_id=102, extra=_model_behavior_extra("family-b"),
    )
    assert worker._model_behavior_quarantine

    dao.status = "running"
    dao.admission_deadline_at = int(time.time()) - 1
    config = await worker._behavior_gate_config()

    assert await worker._behavior_gate_allows(worker._state.battle, config)


@pytest.mark.asyncio
@pytest.mark.parametrize("marker", ["subject", "global"])
async def test_invalid_marker_prevents_retry_without_counting_as_score(marker):
    dao = _GateDAO("passed")
    getattr(dao, f"{marker}_invalid_task_ids").add(101)
    count, launched, _dao, worker = await _dispatch_count(
        status="passed", dao=dao,
    )

    assert count == 0
    assert launched == []
    assert worker._state.rotation_request is not None


def _zero_activity_extra(**overrides):
    extra = {
        "commands_executed": 0,
        "llm_call_count": 0,
        "usage": {"total_tokens": 0},
        "output_bytes": 0,
        "terminated_reason": "error_exit_3",
        "failure_owner": "template",
        "request_attempted": False,
        "request_reached_model": False,
        "template_family_id": "terminal:CythonBuild",
        "template_id": "17",
        "template_revision": "r3",
    }
    extra.update(overrides)
    return extra


def _model_behavior_extra(template_family_id="family-a", **overrides):
    evidence = {
        "classification": "model_no_progress",
        "failure_owner": "model",
        "request_attempted": True,
        "request_reached_model": True,
        "endpoint_healthy": True,
        "template_healthy": True,
        "template_family_id": template_family_id,
        "template_id": template_family_id,
        "template_revision": "r1",
        "catalog_revision": "catalog-1",
    }
    evidence.update(overrides)
    return {
        "commands_executed": 1,
        "llm_call_count": 1,
        "usage": {"total_tokens": 10},
        "output_bytes": 5,
        "model_behavior": evidence,
    }


class _ResultEnv:
    docker_image = "terminal:test"

    def __init__(self, *, score=1.0, extra=None, error=None, task_id=101):
        self.result = Result(
            env="TERMINAL",
            score=score,
            latency_seconds=0.01,
            success=True,
            error=error,
            task_id=task_id,
            extra=dict(extra or {}),
        )

    async def evaluate(self, *, miner, task_id):
        del miner, task_id
        return self.result


class _TimeoutEnv:
    docker_image = "terminal:test"

    async def evaluate(self, *, miner, task_id):
        del miner, task_id
        raise EvaluationDeadlineExceeded("TERMINAL", 3600)


class _ExceptionEnv:
    docker_image = "terminal:test"

    async def evaluate(self, *, miner, task_id):
        del miner, task_id
        raise RuntimeError("unstructured harness failure")


async def _run_runtime_result(
    *,
    mode="enforce",
    role="battle",
    extra=None,
    expected_deployment_id=None,
    result_error=None,
    runtime_results=None,
    invalid_results=None,
    template_results=None,
    model_observation_results=None,
    promotion_sealed_at=None,
    task_id=101,
):
    dao = _GateDAO(
        "passed",
        runtime_results=runtime_results,
        invalid_results=invalid_results,
        template_results=template_results,
        model_observation_results=model_observation_results,
        promotion_sealed_at=promotion_sealed_at,
    )
    state = _State(_gate_config(mode=mode))
    if role == "battle":
        miner = state.battle.challenger
        deployment_id = state.battle.deployment_id
    elif role == "predeploy":
        record = BattleRecord(
            challenger=MinerSnapshot(
                uid=209,
                hotkey="predeploy",
                revision="predeploy-revision",
                model="org/predeploy",
            ),
            deployment_id="dep-predeploy",
            base_url="https://predeploy.invalid/v1",
            started_at_block=2,
        )
        state.predeployed = [record]
        miner = record.challenger
        deployment_id = record.deployment_id
    elif role == "champion":
        miner = MinerSnapshot(
            uid=state.champion.uid,
            hotkey=state.champion.hotkey,
            revision=state.champion.revision,
            model=state.champion.model,
        )
        deployment_id = state.champion.deployment_id
    else:  # pragma: no cover
        raise AssertionError(f"unknown role: {role}")

    worker = ExecutorWorker(
        worker_id=0, env="TERMINAL", behavior_gate_dao=dao,
    )
    worker._state = state
    worker._samples = _Samples()
    worker._env_executor = _ResultEnv(
        score=1.0,
        extra=_zero_activity_extra() if extra is None else extra,
        error=result_error,
        task_id=task_id,
    )
    await worker._evaluate_and_persist_gated(
        miner=miner,
        task_id=task_id,
        base_url="https://result.invalid/v1",
        refresh_block=10,
        miner_obj=object(),
        expected_deployment_id=(
            deployment_id
            if expected_deployment_id is None
            else expected_deployment_id
        ),
    )
    return worker, dao, worker._samples


async def _run_battle_attempt(worker, *, task_id, extra):
    battle = worker._state.battle
    worker._env_executor = _ResultEnv(score=1.0, extra=extra, task_id=task_id)
    await worker._evaluate_and_persist_gated(
        miner=battle.challenger,
        task_id=task_id,
        base_url="https://result.invalid/v1",
        refresh_block=10,
        miner_obj=object(),
        expected_deployment_id=battle.deployment_id,
    )


@pytest.mark.asyncio
async def test_typed_harness_timeout_is_tombstoned_without_model_strike():
    dao = _GateDAO("passed")
    state = _State(_gate_config())
    samples = _Samples()
    worker = ExecutorWorker(
        worker_id=0, env="TERMINAL", behavior_gate_dao=dao,
    )
    worker._state = state
    worker._samples = samples
    worker._env_executor = _TimeoutEnv()

    await worker._evaluate_and_persist_gated(
        miner=state.battle.challenger,
        task_id=101,
        base_url=state.battle.base_url,
        refresh_block=10,
        miner_obj=object(),
        expected_deployment_id=state.battle.deployment_id,
    )

    assert samples.persist_calls == []
    assert len(dao.invalid_samples) == 1
    _args, kwargs = dao.invalid_samples[0]
    assert kwargs["reason_code"] == "harness_deadline_exceeded"
    assert kwargs["evidence"]["failure_owner"] == "harness"
    assert kwargs["evidence"]["request_dispatched"] is True
    assert dao.model_observations == []
    assert dao.model_failures == []
    assert worker.metrics.tasks_invalid == 1


@pytest.mark.asyncio
async def test_unstructured_exception_is_immediately_durable_invalid():
    dao = _GateDAO("passed")
    state = _State(_gate_config())
    samples = _Samples()
    worker = ExecutorWorker(
        worker_id=0, env="TERMINAL", behavior_gate_dao=dao,
    )
    worker._state = state
    worker._samples = samples
    worker._env_executor = _ExceptionEnv()
    call = {
        "miner": state.battle.challenger,
        "task_id": 101,
        "base_url": state.battle.base_url,
        "refresh_block": 10,
        "miner_obj": object(),
        "expected_deployment_id": state.battle.deployment_id,
    }

    await worker._evaluate_and_persist_gated(**call)

    assert samples.persist_calls == []
    assert len(dao.invalid_samples) == 1
    _args, kwargs = dao.invalid_samples[0]
    assert kwargs["reason_code"] == "harness_exception"
    assert kwargs["evidence"]["failure_owner"] == "harness"
    assert dao.template_incidents == []
    assert dao.model_observations == []
    assert dao.model_failures == []
    assert worker.metrics.tasks_failed == 0
    assert worker.metrics.tasks_invalid == 1


@pytest.mark.asyncio
async def test_unattributed_result_error_is_invalid_without_any_strike():
    extra = {
        "commands_executed": 1,
        "llm_call_count": 1,
        "usage": {"total_tokens": 10},
        "output_bytes": 5,
        "template_family_id": "family-a",
        "template_id": "template-a",
        "template_revision": "r1",
    }
    worker, dao, samples = await _run_runtime_result(
        extra=extra,
        result_error="unattributed result failure",
    )

    assert samples.persist_calls == []
    assert len(dao.invalid_samples) == 1
    _args, kwargs = dao.invalid_samples[0]
    assert kwargs["reason_code"] == "harness_result_error"
    assert kwargs["evidence"]["failure_owner"] == "harness"
    assert dao.template_incidents == []
    assert dao.model_observations == []
    assert dao.model_failures == []
    assert worker.metrics.tasks_invalid == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["shadow", "admission_shadow", "enforce"])
@pytest.mark.parametrize("role", ["battle", "predeploy", "champion"])
async def test_positive_zero_activity_is_invalid_for_every_mode_and_role(mode, role):
    worker, dao, samples = await _run_runtime_result(mode=mode, role=role)

    assert samples.persist_calls == []
    assert len(dao.invalid_samples) == 1
    args, kwargs = dao.invalid_samples[0]
    assert args[2:5] == ("TERMINAL", 10, 101)
    assert kwargs["reason_code"] == "positive_score_zero_activity"
    assert "score" not in kwargs["evidence"]
    assert kwargs["evidence"]["failure_owner"] == "template"
    assert len(dao.template_incidents) == 1
    assert dao.model_observations == []
    assert dao.model_failures == []
    assert worker.metrics.tasks_invalid == 1
    assert worker.metrics.tasks_failed == 0


@pytest.mark.asyncio
async def test_repeated_invalid_samples_never_accumulate_model_strikes():
    worker, dao, samples = await _run_runtime_result(task_id=101)
    await _run_battle_attempt(
        worker, task_id=102, extra=_zero_activity_extra(),
    )

    assert len(dao.invalid_samples) == 2
    assert len(dao.template_incidents) == 2
    assert dao.model_observations == []
    assert dao.model_failures == []
    assert samples.persist_calls == []
    assert worker.metrics.tasks_invalid == 2


@pytest.mark.asyncio
async def test_quarantined_family_consumes_new_task_without_scoring_it():
    extra = {
        "commands_executed": 1,
        "llm_call_count": 1,
        "usage": {"total_tokens": 10},
        "output_bytes": 5,
        "template_family_id": "family-quarantined",
        "template_id": "task-template-103",
        "template_revision": "r1",
    }
    dao = _GateDAO("passed", template_health_status="quarantined")
    state = _State(_gate_config())
    samples = _Samples()
    worker = ExecutorWorker(
        worker_id=0, env="TERMINAL", behavior_gate_dao=dao,
    )
    worker._state = state
    worker._samples = samples
    worker._env_executor = _ResultEnv(score=1.0, extra=extra, task_id=103)

    await worker._evaluate_and_persist_gated(
        miner=state.battle.challenger,
        task_id=103,
        base_url=state.battle.base_url,
        refresh_block=10,
        miner_obj=object(),
        expected_deployment_id=state.battle.deployment_id,
    )

    assert samples.persist_calls == []
    assert dao.invalid_samples[0][1]["reason_code"] == (
        "template_family_quarantined"
    )
    assert len(dao.template_incidents) == 1
    assert dao.global_invalid_task_ids == {103}
    assert dao.model_observations == []


@pytest.mark.asyncio
async def test_invalid_template_family_hash_is_stable_across_revisions():
    worker, dao, _samples = await _run_runtime_result(
        task_id=101,
        extra=_zero_activity_extra(template_revision="r1"),
    )
    await _run_battle_attempt(
        worker,
        task_id=102,
        extra=_zero_activity_extra(template_revision="r2"),
    )

    hashes = [call[1]["template_key_hash"] for call in dao.invalid_samples]
    assert len(hashes) == 2
    assert hashes[0] == hashes[1]


@pytest.mark.asyncio
async def test_invalid_template_incident_accepts_exact_template_id_only():
    extra = _zero_activity_extra()
    del extra["template_family_id"]
    _worker, dao, _samples = await _run_runtime_result(extra=extra)

    assert len(dao.template_incidents) == 1
    assert dao.invalid_samples[0][1]["template_key_hash"] is not None


@pytest.mark.asyncio
async def test_invalid_sample_precedes_structured_error_without_scoring_it():
    worker, dao, samples = await _run_runtime_result(
        result_error="structured harness failure",
    )

    assert len(dao.invalid_samples) == 1
    assert samples.persist_calls == []
    assert worker.metrics.tasks_invalid == 1
    assert worker.metrics.tasks_failed == 0


@pytest.mark.asyncio
async def test_missing_activity_telemetry_is_unknown_and_remains_a_normal_sample():
    extra = _zero_activity_extra()
    del extra["commands_executed"]
    worker, dao, samples = await _run_runtime_result(extra=extra)

    assert dao.invalid_samples == []
    assert len(samples.persist_calls) == 1
    assert samples.persist_calls[0]["score"] == 1.0
    assert worker.metrics.tasks_invalid == 0


@pytest.mark.asyncio
async def test_invalid_marker_is_written_only_after_deployment_validation():
    _worker, dao, samples = await _run_runtime_result(
        expected_deployment_id="stale-deployment",
    )

    assert dao.invalid_samples == []
    assert dao.template_incidents == []
    assert samples.persist_calls == []


@pytest.mark.asyncio
async def test_failed_invalid_marker_is_a_local_tombstone_until_retry_succeeds():
    worker, dao, samples = await _run_runtime_result(
        invalid_results=[RuntimeError("ddb unavailable")],
    )

    key = ("challenger", "challenger-revision", "TERMINAL", 10, 101)
    assert key in worker._pending_invalid_samples
    assert dao.subject_invalid_task_ids == set()
    assert samples.persist_calls == []

    # The next dispatch poll retries the marker first.  Candidate selection
    # still sees the local tombstone, so this task is never launched again.
    launched = []

    async def dispatch_one(**kwargs):
        launched.append(kwargs)

    worker._dispatch_one = dispatch_one
    tasks = set()
    assert await worker._dispatch_new(set(), tasks) == 0
    assert launched == []
    assert key not in worker._pending_invalid_samples
    assert dao.subject_invalid_task_ids == {101}


@pytest.mark.asyncio
async def test_failed_template_incident_is_retried_after_invalid_marker_is_durable():
    worker, dao, _samples = await _run_runtime_result(
        template_results=[RuntimeError("ddb unavailable")],
    )

    key = ("challenger", "challenger-revision", "TERMINAL", 10, 101)
    assert key not in worker._pending_invalid_samples
    assert key in worker._pending_template_incidents

    await worker._retry_pending_non_score_writes()

    assert key not in worker._pending_template_incidents
    assert dao.global_invalid_task_ids == {101}


@pytest.mark.asyncio
async def test_model_evidence_requires_two_distinct_healthy_template_families():
    worker, dao, samples = await _run_runtime_result(
        extra=_model_behavior_extra("family-a"),
    )
    await _run_battle_attempt(
        worker, task_id=102, extra=_model_behavior_extra("family-b"),
    )

    assert len(dao.model_observations) == 2
    assert (
        dao.model_observations[0][1]["template_key_hash"]
        != dao.model_observations[1][1]["template_key_hash"]
    )
    assert len(dao.model_failures) == 1
    assert dao.model_failures[0][1]["classification"] == "model_no_progress"
    assert dao.status == "failed"
    # The first family is evidence, not yet a breaker: keep its benchmark row
    # so enforce mode cannot redispatch it forever.  The second family closes
    # the breaker and is blocked.
    assert len(samples.persist_calls) == 1
    assert samples.persist_calls[0]["task_id"] == 101


@pytest.mark.asyncio
async def test_model_observation_write_failure_does_not_block_the_sample():
    _worker, dao, samples = await _run_runtime_result(
        extra=_model_behavior_extra("family-a"),
        model_observation_results=[RuntimeError("ddb unavailable")],
    )

    assert len(dao.model_observations) == 1
    assert dao.model_failures == []
    assert len(samples.persist_calls) == 1
    assert samples.persist_calls[0]["score"] == 1.0


@pytest.mark.asyncio
async def test_attributed_model_error_below_threshold_persists_structured_zero():
    _worker, dao, samples = await _run_runtime_result(
        extra=_model_behavior_extra("family-a"),
        result_error="untrusted raw model error",
    )

    assert len(dao.model_observations) == 1
    assert dao.model_failures == []
    assert len(samples.persist_calls) == 1
    row = samples.persist_calls[0]
    assert row["score"] == 0.0
    assert row["extra"]["zero_score_reason"] == "structured_model_behavior"
    assert "untrusted raw model error" not in str(row["extra"])


@pytest.mark.asyncio
async def test_same_template_family_is_idempotent_model_evidence():
    worker, dao, samples = await _run_runtime_result(
        extra=_model_behavior_extra("family-a"),
    )
    await _run_battle_attempt(
        worker, task_id=102, extra=_model_behavior_extra("family-a"),
    )

    assert len(dao.model_observations) == 2
    assert dao.model_failures == []
    assert dao.status == "passed"
    assert [row["task_id"] for row in samples.persist_calls] == [101, 102]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("failure_owner", "harness"),
        ("request_attempted", False),
        ("request_reached_model", False),
        ("endpoint_healthy", False),
        ("template_healthy", False),
    ],
)
async def test_incomplete_model_attribution_never_records_a_strike(field, value):
    extra = _model_behavior_extra()
    extra["model_behavior"][field] = value
    _worker, dao, samples = await _run_runtime_result(extra=extra)

    assert dao.model_observations == []
    assert dao.model_failures == []
    assert len(samples.persist_calls) == 1


@pytest.mark.asyncio
async def test_shadow_records_model_verdict_but_never_blocks_benchmark_sample():
    worker, dao, samples = await _run_runtime_result(
        mode="shadow", extra=_model_behavior_extra("family-a"),
    )
    await _run_battle_attempt(
        worker, task_id=102, extra=_model_behavior_extra("family-b"),
    )

    assert len(dao.model_failures) == 1
    assert dao.status == "failed"
    assert len(samples.persist_calls) == 2
    assert worker._model_behavior_quarantine == {}


@pytest.mark.asyncio
async def test_plain_shadow_retries_uncertain_verdict_without_holding_sampling():
    worker, dao, samples = await _run_runtime_result(
        mode="shadow",
        extra=_model_behavior_extra("family-a"),
        runtime_results=[False, True],
    )
    await _run_battle_attempt(
        worker, task_id=102, extra=_model_behavior_extra("family-b"),
    )

    assert len(samples.persist_calls) == 2
    assert worker._model_behavior_quarantine

    config = await worker._behavior_gate_config()
    assert await worker._behavior_gate_allows(worker._state.battle, config)
    assert worker._model_behavior_quarantine == {}
    assert dao.status == "failed"


@pytest.mark.asyncio
async def test_model_evidence_never_mutates_current_champion_gate():
    _worker, dao, samples = await _run_runtime_result(
        role="champion", extra=_model_behavior_extra(),
    )

    assert dao.model_observations == []
    assert dao.model_failures == []
    assert len(samples.persist_calls) == 1
