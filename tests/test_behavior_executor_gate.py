from __future__ import annotations

import asyncio

import pytest

from affine.core.models import Result
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
        promotion_sealed_at=None,
    ):
        self.status = status
        self.promotion_sealed_at = promotion_sealed_at
        self.calls = 0
        self.runtime_observations = []
        self.runtime_failures = []
        self.runtime_results = list(runtime_results or [])
        self._runtime_tasks = {}

    async def get_verdict(self, *_args):
        self.calls += 1
        if self.status is None:
            return None
        return {
            "status": self.status,
            "reason_code": "test",
            "promotion_sealed_at": self.promotion_sealed_at,
        }

    async def record_runtime_invariant_observation(self, *args, **kwargs):
        self.runtime_observations.append((args, kwargs))
        key = (*args, kwargs["signature_hash"])
        tasks = self._runtime_tasks.setdefault(key, set())
        tasks.add(kwargs["task_hash"])
        return min(len(tasks), kwargs["threshold"])

    async def fail_runtime_invariant(self, *args, **kwargs):
        self.runtime_failures.append((args, kwargs))
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

    async def get_task_state(self):
        return TaskIdState(
            task_ids={"TERMINAL": [101]},
            refreshed_at_block=10,
        )

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


def _gate_config(*, mode="enforce", gated_environments=("*",)):
    return {
        "enabled": True,
        "mode": mode,
        "policy_version": "test-v1",
        "gated_environments": list(gated_environments),
    }


async def _dispatch_count(*, status, mode="enforce", gated_environments=("*",)):
    worker = ExecutorWorker(
        worker_id=0,
        env="TERMINAL",
        behavior_gate_dao=_GateDAO(status),
    )
    worker._state = _State(_gate_config(
        mode=mode,
        gated_environments=gated_environments,
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
    return count, launched, worker._behavior_gate_dao


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [None, "pending", "running", "suspected", "deferred", "failed"])
async def test_enforce_gate_keeps_challenger_dispatch_at_zero_until_passed(status):
    count, launched, dao = await _dispatch_count(status=status)

    assert count == 0
    assert launched == []
    assert dao.calls == 1


@pytest.mark.asyncio
async def test_passed_gate_releases_challenger_sampling():
    count, launched, dao = await _dispatch_count(status="passed")

    assert count == 1
    assert len(launched) == 1
    assert launched[0]["miner"].uid == 208
    assert dao.calls == 1


@pytest.mark.asyncio
async def test_shadow_mode_observes_but_never_blocks_sampling():
    count, launched, dao = await _dispatch_count(status="failed", mode="shadow")

    assert count == 1
    assert len(launched) == 1
    assert dao.calls == 0


@pytest.mark.asyncio
async def test_environment_allowlist_leaves_ungated_env_unchanged():
    count, launched, dao = await _dispatch_count(
        status="failed",
        gated_environments=("SWE-INFINITE",),
    )

    assert count == 1
    assert len(launched) == 1
    assert dao.calls == 0


def _zero_activity_extra(**overrides):
    extra = {
        "commands_executed": 0,
        "llm_call_count": 0,
        "usage": {"total_tokens": 0},
        "output_bytes": 0,
        "terminated_reason": "error_exit_3",
    }
    extra.update(overrides)
    return extra


class _ResultEnv:
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


async def _run_runtime_result(
    *,
    mode="enforce",
    role="battle",
    extra=None,
    expected_deployment_id=None,
    result_error=None,
    runtime_results=None,
    promotion_sealed_at=None,
    task_id=101,
):
    dao = _GateDAO(
        "passed",
        runtime_results=runtime_results,
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
    else:  # pragma: no cover - test helper contract
        raise AssertionError(f"unknown role: {role}")

    worker = ExecutorWorker(
        worker_id=0,
        env="TERMINAL",
        behavior_gate_dao=dao,
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


async def _run_battle_attempt(
    worker,
    *,
    task_id,
    extra=None,
    result_error=None,
):
    battle = worker._state.battle
    worker._env_executor = _ResultEnv(
        score=1.0,
        extra=_zero_activity_extra() if extra is None else extra,
        error=result_error,
        task_id=task_id,
    )
    await worker._evaluate_and_persist_gated(
        miner=battle.challenger,
        task_id=task_id,
        base_url="https://result.invalid/v1",
        refresh_block=10,
        miner_obj=object(),
        expected_deployment_id=battle.deployment_id,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["battle", "predeploy"])
async def test_first_runtime_invariant_is_observed_and_dropped_without_final_failure(
    role,
):
    worker, dao, samples = await _run_runtime_result(role=role)

    assert len(dao.runtime_observations) == 1
    args, kwargs = dao.runtime_observations[0]
    assert args[0] in {"challenger", "predeploy"}
    assert kwargs["classification"] == "harness_invalid"
    assert kwargs["threshold"] == 2
    assert len(kwargs["signature_hash"]) == 64
    assert len(kwargs["task_hash"]) == 64
    assert "task_id" not in kwargs["evidence"]
    assert "terminated_reason" not in kwargs["evidence"]
    assert dao.runtime_failures == []
    assert worker._runtime_gate_quarantine == {}
    assert samples.persist_calls == []
    assert worker.metrics.tasks_failed == 1


@pytest.mark.asyncio
async def test_second_distinct_task_with_same_signature_finally_fails_gate():
    worker, dao, samples = await _run_runtime_result(task_id=101)

    await _run_battle_attempt(worker, task_id=102)

    assert len(dao.runtime_observations) == 2
    first = dao.runtime_observations[0][1]
    second = dao.runtime_observations[1][1]
    assert first["signature_hash"] == second["signature_hash"]
    assert first["task_hash"] != second["task_hash"]
    assert len(dao.runtime_failures) == 1
    _args, kwargs = dao.runtime_failures[0]
    assert kwargs["reason_code"] == "runtime_positive_score_zero_activity"
    assert kwargs["counts"]["harness_invalid"] == 2
    assert kwargs["counts"]["total"] == 2
    assert dao.status == "failed"
    assert samples.persist_calls == []


@pytest.mark.asyncio
async def test_same_task_retry_does_not_increment_runtime_signature_count():
    worker, dao, _samples = await _run_runtime_result(task_id=101)

    await _run_battle_attempt(worker, task_id=101)

    assert len(dao.runtime_observations) == 2
    first = dao.runtime_observations[0][1]
    second = dao.runtime_observations[1][1]
    assert first["signature_hash"] == second["signature_hash"]
    assert first["task_hash"] == second["task_hash"]
    assert dao.runtime_failures == []
    assert worker._runtime_gate_quarantine == {}


@pytest.mark.asyncio
async def test_different_runtime_signatures_do_not_accumulate():
    worker, dao, _samples = await _run_runtime_result(
        task_id=101,
        extra=_zero_activity_extra(terminated_reason="exit-a"),
    )

    await _run_battle_attempt(
        worker,
        task_id=102,
        extra=_zero_activity_extra(terminated_reason="exit-b"),
    )

    first = dao.runtime_observations[0][1]
    second = dao.runtime_observations[1][1]
    assert first["signature_hash"] != second["signature_hash"]
    assert first["task_hash"] != second["task_hash"]
    assert dao.runtime_failures == []
    assert worker._runtime_gate_quarantine == {}


@pytest.mark.asyncio
async def test_shadow_runtime_invariant_records_but_still_persists_sample():
    _worker, dao, samples = await _run_runtime_result(mode="shadow")

    assert len(dao.runtime_observations) == 1
    assert dao.runtime_failures == []
    assert len(samples.persist_calls) == 1
    assert samples.persist_calls[0]["score"] == 1.0


@pytest.mark.asyncio
@pytest.mark.parametrize("error_location", ["result", "extra"])
async def test_runtime_invariant_precedes_structured_error_drop(error_location):
    extra = _zero_activity_extra()
    result_error = None
    if error_location == "result":
        result_error = "structured failure"
    else:
        extra["error"] = "structured failure"

    worker, dao, samples = await _run_runtime_result(
        extra=extra,
        result_error=result_error,
    )

    assert len(dao.runtime_observations) == 1
    assert dao.runtime_failures == []
    assert samples.persist_calls == []
    assert worker.metrics.tasks_failed == 1


@pytest.mark.asyncio
async def test_structured_error_invariant_runs_only_after_deployment_validation():
    _worker, dao, samples = await _run_runtime_result(
        result_error="structured failure",
        expected_deployment_id="stale-deployment",
    )

    assert dao.runtime_observations == []
    assert dao.runtime_failures == []
    assert samples.persist_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "write_failure",
    [False, RuntimeError("database unavailable")],
)
async def test_runtime_write_uncertainty_quarantines_enforce_deployment(
    write_failure,
):
    worker, dao, samples = await _run_runtime_result(
        runtime_results=[write_failure, False],
    )
    await _run_battle_attempt(worker, task_id=102)

    assert samples.persist_calls == []
    assert len(worker._runtime_gate_quarantine) == 1

    config = await worker._behavior_gate_config()
    assert not await worker._behavior_gate_allows(worker._state.battle, config)
    assert len(dao.runtime_failures) == 2
    assert dao.status == "passed"
    assert len(worker._runtime_gate_quarantine) == 1


@pytest.mark.asyncio
async def test_later_gate_check_retries_and_clears_runtime_quarantine():
    worker, dao, _samples = await _run_runtime_result(
        runtime_results=[False, True],
    )
    await _run_battle_attempt(worker, task_id=102)

    config = await worker._behavior_gate_config()
    assert not await worker._behavior_gate_allows(worker._state.battle, config)
    assert len(dao.runtime_failures) == 2
    assert worker._runtime_gate_quarantine == {}
    assert dao.status == "failed"

    # Once durable, the ordinary failed verdict remains fail-closed.
    assert not await worker._behavior_gate_allows(worker._state.battle, config)
    assert dao.calls >= 1


@pytest.mark.asyncio
async def test_promotion_seal_wins_cutoff_without_local_quarantine():
    worker, dao, samples = await _run_runtime_result(
        runtime_results=[False],
        promotion_sealed_at=123,
    )

    await _run_battle_attempt(worker, task_id=102)

    assert len(dao.runtime_failures) == 1
    assert worker._runtime_gate_quarantine == {}
    assert dao.status == "passed"
    assert samples.persist_calls == []

    config = await worker._behavior_gate_config()
    assert not await worker._behavior_gate_allows(worker._state.battle, config)


@pytest.mark.asyncio
async def test_shadow_write_uncertainty_still_persists_sample():
    worker, dao, samples = await _run_runtime_result(
        mode="shadow",
        runtime_results=[False],
    )
    await _run_battle_attempt(worker, task_id=102)

    assert len(dao.runtime_failures) == 1
    assert len(samples.persist_calls) == 2
    assert all(call["score"] == 1.0 for call in samples.persist_calls)
    assert len(worker._runtime_gate_quarantine) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "missing_field",
    [
        "commands_executed",
        "llm_call_count",
        "usage",
        "usage.total_tokens",
        "output_bytes",
    ],
)
async def test_runtime_invariant_requires_every_telemetry_field_to_be_present(
    missing_field,
):
    extra = _zero_activity_extra()
    if missing_field == "usage.total_tokens":
        del extra["usage"]["total_tokens"]
    else:
        del extra[missing_field]

    _worker, dao, samples = await _run_runtime_result(extra=extra)

    assert dao.runtime_observations == []
    assert dao.runtime_failures == []
    assert len(samples.persist_calls) == 1


@pytest.mark.asyncio
async def test_runtime_invariant_never_applies_to_current_champion():
    _worker, dao, samples = await _run_runtime_result(role="champion")

    assert dao.runtime_observations == []
    assert dao.runtime_failures == []
    assert len(samples.persist_calls) == 1


@pytest.mark.asyncio
async def test_runtime_invariant_runs_only_after_deployment_validation():
    _worker, dao, samples = await _run_runtime_result(
        expected_deployment_id="stale-deployment",
    )

    assert dao.runtime_observations == []
    assert dao.runtime_failures == []
    assert samples.persist_calls == []
