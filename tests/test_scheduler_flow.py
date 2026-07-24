"""Tests for the simplified flow scheduler.

In-memory collaborators for state / queue / sampler / sample-count;
fake deploy/teardown callables. Each fake exercises a real protocol the
production wiring uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import pytest

from affine.src.scheduler.flow import (
    DeploymentRoleTransitionResult,
    DeploymentStateInvalidatedError,
    FlowConfig,
    FlowScheduler,
)
from affine.src.scheduler.health import (
    DeploymentHealthResult,
    DeploymentHealthState,
)
from affine.src.scheduler.targon import (
    DeployResult,
    DeployTarget,
)
from affine.src.scorer.challenger_queue import (
    ChallengerQueue,
    STATUS_CHAMPION,
    STATUS_IN_PROGRESS,
    STATUS_SAMPLING,
    STATUS_TERMINATED,
)
from affine.src.scorer.comparator import WindowComparator
from affine.src.scorer.sampler import WindowSampler
from affine.src.scorer.window_state import (
    BattleRecord,
    DeploymentRecord,
    InMemoryConfigStore,
    MinerSnapshot,
    StateStore,
    TaskIdState,
)


WINDOW_BLOCKS = 100


# ---- fakes -----------------------------------------------------------------


class _InMemoryMinerStore:
    def __init__(self, rows: List[dict]):
        self.rows: Dict[int, dict] = {r["uid"]: dict(r) for r in rows}

    async def list_valid_pending(self) -> List[dict]:
        return [r for r in self.rows.values() if str(r.get("is_valid", "")).lower() == "true"]

    async def claim_pending(self, uid: int, window_id: int, *, expected_status=STATUS_SAMPLING) -> bool:
        row = self.rows.get(uid)
        if row is None:
            return False
        cur = row.get("challenge_status")
        if cur is not None and cur != expected_status:
            return False
        row["challenge_status"] = STATUS_IN_PROGRESS
        row["last_window_id"] = window_id
        return True

    async def set_terminal(
        self,
        uid: int,
        new_status: str,
        *,
        reason: str = "",
        hotkey: str | None = None,
        revision: str | None = None,
        model: str = "",
        scores_by_env: dict | None = None,
        opponent_scores_by_env: dict | None = None,
        battle_task_ids: dict | None = None,
        scores_refresh_block: int | None = None,
        terminated_at_block: int | None = None,
    ) -> None:
        self.rows.setdefault(uid, {"uid": uid})["challenge_status"] = new_status
        self.rows.setdefault(uid, {"uid": uid})["termination_reason"] = reason
        if hotkey is not None:
            self.rows[uid]["hotkey"] = hotkey
        if revision is not None:
            self.rows[uid]["revision"] = revision
        if model:
            self.rows[uid]["model"] = model
        if scores_by_env is not None:
            self.rows[uid]["scores_by_env"] = scores_by_env
        if opponent_scores_by_env is not None:
            self.rows[uid]["opponent_scores_by_env"] = opponent_scores_by_env
        if battle_task_ids is not None:
            self.rows[uid]["battle_task_ids"] = battle_task_ids
        if scores_refresh_block is not None:
            self.rows[uid]["scores_refresh_block"] = scores_refresh_block
        if terminated_at_block is not None:
            self.rows[uid]["terminated_at_block"] = terminated_at_block

    async def release_claim(
        self, uid: int, *,
        hotkey: str | None = None,
        revision: str | None = None,
    ) -> bool:
        row = self.rows.get(uid)
        if row is None or row.get("challenge_status") != STATUS_IN_PROGRESS:
            return False
        row["challenge_status"] = STATUS_SAMPLING
        return True

    async def list_in_progress(self) -> List[dict]:
        return [
            dict(r) for r in self.rows.values()
            if r.get("challenge_status") == STATUS_IN_PROGRESS
        ]


@dataclass
class _DeployTracker:
    deploys: List[DeployTarget] = field(default_factory=list)
    teardowns: List[Optional[str]] = field(default_factory=list)
    fail_on_deploy: bool = False
    # Default: simulate a single-endpoint deployment — no free capacity for
    # pre-sampling. ``_predeploy_fill_available`` stops cleanly on this
    # ``NoEndpointCapacity`` exactly like ``_select_ssh_endpoints`` would
    # report under one endpoint. Tests that want pre-sample slots set
    # ``pre_challenger_slots`` to allow that many extra deploys.
    pre_challenger_slots: int = 0
    next_deployment_id: int = 0
    predeploys: List[DeployTarget] = field(default_factory=list)
    promotions: List[Tuple[str, str]] = field(default_factory=list)
    transition_results: List[
        DeploymentRoleTransitionResult
    ] = field(default_factory=list)
    # uids that should raise a non-capacity error on pre-deploy
    # (simulates a real ssh / docker failure). The fill loop must
    # ``mark_terminated(FAILED)`` and advance to the next candidate.
    predeploy_failures: set = field(default_factory=set)
    # uids whose teardown should raise on the FIRST attempt only.
    teardown_failures_once: Set[str] = field(default_factory=set)
    _teardown_attempts: Dict[str, int] = field(default_factory=dict)

    async def deploy(self, target: DeployTarget, role: str = "active") -> DeployResult:
        if role == "pre_challenger":
            if target.uid in self.predeploy_failures:
                # Real deploy failure — distinct from 'no capacity'. ssh.py's
                # ``deploy`` raises generic ``RuntimeError`` on docker
                # errors; we use that exact shape here.
                raise RuntimeError(
                    f"simulated ssh deploy failure for uid={target.uid}"
                )
            if len(self.predeploys) >= self.pre_challenger_slots:
                from affine.src.scheduler.flow import NoEndpointCapacity
                raise NoEndpointCapacity(
                    "no free ssh endpoint for pre_challenger"
                )
            self.predeploys.append(target)
        else:
            self.deploys.append(target)
        if self.fail_on_deploy:
            raise RuntimeError("simulated targon deploy failure")
        self.next_deployment_id += 1
        did = f"wrk-{self.next_deployment_id:03d}"
        base_url = f"https://t/{did}"
        return DeployResult(deployment_id=did, base_url=base_url)

    async def teardown(self, deployment_id: Optional[str]) -> None:
        self.teardowns.append(deployment_id)
        # Optional: simulate a flaky teardown by raising once for a
        # configured uid. Test resets ``teardown_failures_once`` after
        # the expected retry boundary.
        attempts = self._teardown_attempts.get(deployment_id or "", 0) + 1
        self._teardown_attempts[deployment_id or ""] = attempts
        if attempts == 1 and deployment_id in {
            f"wrk-{i:03d}" for i in range(self.next_deployment_id + 1)
        }:
            # Match by deployment_id directly via ``teardown_failures_once``
            # if test inserted the deployment_id there.
            if deployment_id in self.teardown_failures_once:
                raise RuntimeError(
                    f"simulated teardown failure for {deployment_id}"
                )

    async def transition_role(
        self, record: BattleRecord, role: str,
    ) -> DeploymentRoleTransitionResult:
        self.promotions.append((record.deployment_id, role))
        if self.transition_results:
            return self.transition_results.pop(0)
        return DeploymentRoleTransitionResult.UPDATED


@dataclass
class _SingleEndpointDeployTracker:
    """Model the stable deployment id and role ownership of one SSH host."""

    deploys: List[DeployTarget] = field(default_factory=list)
    teardowns: List[Optional[str]] = field(default_factory=list)
    transitions: List[Tuple[str, str]] = field(default_factory=list)
    endpoint_role: Optional[str] = None
    deployment_id: str = "ssh:only-endpoint:affine-sglang-current"
    champion_transition_results: List[
        DeploymentRoleTransitionResult
    ] = field(default_factory=list)

    async def deploy(
        self, target: DeployTarget, role: str = "active",
    ) -> DeployResult:
        if role == "pre_challenger":
            from affine.src.scheduler.flow import NoEndpointCapacity

            raise NoEndpointCapacity("single endpoint is already occupied")
        if (
            role == "challenger"
            and self.endpoint_role not in (None, "champion", "active")
        ):
            raise RuntimeError(
                f"single endpoint cannot replace role={self.endpoint_role!r}"
            )
        self.deploys.append(target)
        self.endpoint_role = role
        return DeployResult(
            deployment_id=self.deployment_id,
            base_url="https://only-endpoint/v1",
        )

    async def teardown(self, deployment_id: Optional[str]) -> None:
        self.teardowns.append(deployment_id)
        if deployment_id == self.deployment_id:
            self.endpoint_role = None

    async def transition_role(
        self, record: BattleRecord, role: str,
    ) -> DeploymentRoleTransitionResult:
        expected_roles = {
            "challenger": ("pre_challenger",),
            "champion": ("challenger", "active"),
        }[role]
        if role == "champion" and self.champion_transition_results:
            result = self.champion_transition_results.pop(0)
            if result is DeploymentRoleTransitionResult.STALE:
                self.endpoint_role = None
            return result
        if record.deployment_id != self.deployment_id:
            return DeploymentRoleTransitionResult.STALE
        if self.endpoint_role not in (*expected_roles, role):
            return DeploymentRoleTransitionResult.STALE
        self.endpoint_role = role
        self.transitions.append((record.deployment_id, role))
        return DeploymentRoleTransitionResult.UPDATED


class _SamplesFake:
    """In-test substitute for SampleResultsAdapter. Stores rows tagged
    with refresh_block so the current-refresh filter is exercised."""

    def __init__(self):
        # (hotkey, revision, env) -> {task_id: (score, refresh_block)}
        self._rows: Dict[Tuple[str, str, str], Dict[int, Tuple[float, int]]] = {}

    def set_samples(
        self, hotkey: str, revision: str, env: str, task_ids: List[int],
        score: float = 0.5, refresh_block: int = 0,
    ) -> None:
        bucket = self._rows.setdefault((hotkey, revision, env), {})
        for t in task_ids:
            bucket[t] = (score, refresh_block)

    async def count_samples_for_tasks(
        self, hotkey: str, revision: str, env: str, task_ids: List[int],
        refresh_block: int,
    ) -> int:
        bucket = self._rows.get((hotkey, revision, env), {})
        return sum(
            1 for t in task_ids
            if t in bucket and bucket[t][1] == refresh_block
        )

    async def read_scores_for_tasks(
        self, hotkey: str, revision: str, env: str, task_ids: List[int],
        refresh_block: int,
    ) -> Dict[int, float]:
        bucket = self._rows.get((hotkey, revision, env), {})
        return {
            t: bucket[t][0]
            for t in task_ids
            if t in bucket and bucket[t][1] == refresh_block
        }


class _WeightWriterFake:
    """Records weight_writer.write calls."""
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    async def write(self, **kwargs):
        self.calls.append(kwargs)


def _make_miner(uid, hotkey, first_block, *, status=STATUS_SAMPLING, valid=True,
                revision=None, model=None):
    return {
        "uid": uid,
        "hotkey": hotkey,
        "model": model or f"org/m{uid}",
        "revision": revision or f"rev{uid}",
        "first_block": first_block,
        "is_valid": "true" if valid else "false",
        "challenge_status": status,
    }


def _seed_state(*, sampling_count=4, with_champion=True) -> InMemoryConfigStore:
    kv = InMemoryConfigStore()
    kv.data["environments"] = {
        "ENV_A": {
            "display_name": "A", "enabled_for_sampling": True,
            "sampling": {"sampling_count": sampling_count,
                         "dataset_range": [[0, 1000]],
                         "sampling_mode": "random"},
        },
        "ENV_B": {
            "display_name": "B", "enabled_for_sampling": True,
            "sampling": {"sampling_count": sampling_count,
                         "dataset_range": [[0, 1000]],
                         "sampling_mode": "random"},
        },
    }
    if with_champion:
        kv.data["champion"] = {
            "uid": 1, "hotkey": "champ_hk", "revision": "champ_rev",
            "model": "org/champ",
            "deployment_id": None, "base_url": None,
            "since_block": 0,
        }
    return kv


def _build_scheduler(
    *, kv, miner_store, deployer, samples, weight_writer=None,
    window_blocks=WINDOW_BLOCKS, deployment_health_fn=None,
    deployment_transport_repair_fn=None,
    deployment_endpoint_repair_fn=None,
    list_active_endpoint_names_fn=None,
    list_active_endpoint_activations_fn=None,
    task_pool_refresh_blocks=None, config=None, behavior_gate_dao=None,
    transition_deployment_role_fn=None,
):
    state = StateStore(kv)
    queue = ChallengerQueue(miner_store)
    sampler = WindowSampler()
    comparator = WindowComparator()
    weight_writer = weight_writer or _WeightWriterFake()

    async def list_valid_miners_fn():
        # Mirror MinersDAO.get_valid_miners — only is_valid=true rows.
        return [
            r for r in miner_store.rows.values()
            if str(r.get("is_valid", "")).lower() == "true"
        ]

    scheduler = FlowScheduler(
        config=config or FlowConfig(
            window_blocks=window_blocks,
            task_pool_refresh_blocks=task_pool_refresh_blocks,
            deployment_health_check_interval_sec=0,
        ),
        state=state,
        queue=queue,
        sampler=sampler,
        comparator=comparator,
        weight_writer=weight_writer,
        deploy_fn=deployer.deploy,
        teardown_fn=deployer.teardown,
        sample_count_fn=samples.count_samples_for_tasks,
        scores_reader=samples.read_scores_for_tasks,
        list_valid_miners_fn=list_valid_miners_fn,
        list_active_endpoint_names_fn=list_active_endpoint_names_fn,
        list_active_endpoint_activations_fn=(
            list_active_endpoint_activations_fn
        ),
        deployment_health_fn=deployment_health_fn,
        deployment_transport_repair_fn=deployment_transport_repair_fn,
        deployment_endpoint_repair_fn=deployment_endpoint_repair_fn,
        behavior_gate_dao=behavior_gate_dao,
        transition_deployment_role_fn=(
            transition_deployment_role_fn or deployer.transition_role
        ),
    )
    return scheduler, state, weight_writer


# ---- tests -----------------------------------------------------------------


def test_flow_config_task_pool_refresh_defaults_to_window_blocks(monkeypatch):
    monkeypatch.delenv("SCHEDULER_TASK_POOL_REFRESH_BLOCKS", raising=False)

    cfg = FlowConfig(window_blocks=123)

    assert cfg.task_pool_refresh_blocks == 123
    assert cfg.deployment_health_check_interval_sec == 240
    assert cfg.deployment_health_failure_threshold == 5


def test_flow_config_task_pool_refresh_reads_env(monkeypatch):
    monkeypatch.setenv("SCHEDULER_TASK_POOL_REFRESH_BLOCKS", "456")

    cfg = FlowConfig(window_blocks=123)

    assert cfg.task_pool_refresh_blocks == 456


def test_flow_config_explicit_task_pool_refresh_beats_env(monkeypatch):
    monkeypatch.setenv("SCHEDULER_TASK_POOL_REFRESH_BLOCKS", "456")

    cfg = FlowConfig(window_blocks=123, task_pool_refresh_blocks=789)

    assert cfg.task_pool_refresh_blocks == 789


def test_flow_config_task_pool_refresh_rejects_invalid_env(monkeypatch):
    monkeypatch.setenv("SCHEDULER_TASK_POOL_REFRESH_BLOCKS", "not-an-int")

    with pytest.raises(ValueError, match="SCHEDULER_TASK_POOL_REFRESH_BLOCKS"):
        FlowConfig(window_blocks=123)


def test_flow_config_task_pool_refresh_rejects_zero_env(monkeypatch):
    monkeypatch.setenv("SCHEDULER_TASK_POOL_REFRESH_BLOCKS", "0")

    with pytest.raises(ValueError, match="positive integer"):
        FlowConfig(window_blocks=123)


class _BehaviorGateFake:
    def __init__(
        self,
        status: str | List[str],
        reason: str = "test_reason",
        *,
        seal_results: Optional[List[bool | BaseException]] = None,
    ):
        self.statuses = [status] if isinstance(status, str) else list(status)
        if not self.statuses:
            raise ValueError("at least one behavior-gate status is required")
        self.reason = reason
        self.calls = []
        self.seal_calls = []
        self.seal_results = list(seal_results or [])

    async def get_verdict(self, *identity):
        self.calls.append(identity)
        index = min(len(self.calls) - 1, len(self.statuses) - 1)
        return {"status": self.statuses[index], "reason_code": self.reason}

    async def seal_for_promotion(self, *identity):
        self.seal_calls.append(identity)
        if self.seal_results:
            result = self.seal_results.pop(0)
            if isinstance(result, BaseException):
                raise result
            return result
        # Default fake semantics mirror the DAO: a currently-passed row can
        # be sealed, and repeating the seal is idempotent.
        index = max(0, min(len(self.calls) - 1, len(self.statuses) - 1))
        return self.statuses[index] == "passed"


async def _build_active_behavior_gate_case(
    *,
    status: str | List[str],
    mode: str = "enforce",
    seal_results: Optional[List[bool | BaseException]] = None,
):
    kv = _seed_state(sampling_count=1)
    kv.data["behavior_gate"] = {
        "enabled": True,
        "mode": mode,
        "policy_version": "test-v1",
        "gated_environments": ["*"],
    }
    kv.data["current_task_ids"] = {
        "task_ids": {"ENV_A": [11], "ENV_B": [22]},
        "refreshed_at_block": 10,
    }
    kv.data["champion"].update({
        "deployment_id": "dep-champion",
        "base_url": "https://champion.invalid/v1",
    })
    miner_store = _InMemoryMinerStore([
        _make_miner(
            1, "champ_hk", 1, status=STATUS_CHAMPION,
            revision="champ_rev",
        ),
        _make_miner(
            2, "chal_hk", 2, status=STATUS_IN_PROGRESS,
            revision="chal_rev",
        ),
    ])
    deployer = _DeployTracker()
    samples = _SamplesFake()
    gate = _BehaviorGateFake(
        status,
        reason="strike_threshold:model_protocol_failure",
        seal_results=seal_results,
    )
    scheduler, state, _ = _build_scheduler(
        kv=kv,
        miner_store=miner_store,
        deployer=deployer,
        samples=samples,
        task_pool_refresh_blocks=100,
        behavior_gate_dao=gate,
    )
    await state.set_battle(BattleRecord(
        challenger=MinerSnapshot(
            uid=2,
            hotkey="chal_hk",
            revision="chal_rev",
            model="org/challenger",
        ),
        deployment_id="dep-challenger",
        base_url="https://challenger.invalid/v1",
        started_at_block=12,
    ))
    return scheduler, state, miner_store, deployer, gate, samples


def _fill_behavior_gate_winning_samples(samples: _SamplesFake) -> None:
    for env, task_id in (("ENV_A", 11), ("ENV_B", 22)):
        samples.set_samples(
            "champ_hk", "champ_rev", env, [task_id],
            score=0.1, refresh_block=10,
        )
        samples.set_samples(
            "chal_hk", "chal_rev", env, [task_id],
            score=0.9, refresh_block=10,
        )


@pytest.mark.asyncio
async def test_failed_behavior_gate_marks_active_challenger_lost_and_tears_down():
    scheduler, state, miners, deployer, gate, _samples = (
        await _build_active_behavior_gate_case(status="failed")
    )

    await scheduler.tick(current_block=20)

    assert await state.get_battle() is None
    assert miners.rows[2]["challenge_status"] == STATUS_TERMINATED
    assert miners.rows[2]["termination_reason"] == (
        "model_behavior:strike_threshold:model_protocol_failure"
    )
    assert deployer.teardowns == ["dep-challenger"]
    assert len(gate.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["pending", "running", "suspected", "deferred"])
async def test_nonfinal_behavior_gate_blocks_decision_without_model_loss(status):
    scheduler, state, miners, deployer, _gate, _samples = (
        await _build_active_behavior_gate_case(status=status)
    )

    await scheduler.tick(current_block=20)

    assert await state.get_battle() is not None
    assert miners.rows[2]["challenge_status"] == STATUS_IN_PROGRESS
    assert deployer.teardowns == []


@pytest.mark.asyncio
async def test_shadow_behavior_gate_never_blocks_or_auto_loses():
    scheduler, state, miners, deployer, gate, _samples = (
        await _build_active_behavior_gate_case(status="failed", mode="shadow")
    )

    await scheduler.tick(current_block=20)

    assert await state.get_battle() is not None
    assert miners.rows[2]["challenge_status"] == STATUS_IN_PROGRESS
    assert deployer.teardowns == []
    assert gate.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("second_status", "expected_terminal"),
    [("failed", True), ("pending", False)],
)
async def test_challenger_promotion_is_fenced_by_fresh_gate_snapshot(
    second_status,
    expected_terminal,
):
    scheduler, state, miners, deployer, gate, samples = (
        await _build_active_behavior_gate_case(
            status=["passed", second_status],
        )
    )
    _fill_behavior_gate_winning_samples(samples)

    battle = await state.get_battle()
    champion = await state.get_champion()
    task_state = await state.get_task_state()
    scoring_envs = await state.get_scoring_environments()
    assert battle is not None and champion is not None and task_state is not None
    first_snapshot = await scheduler._active_behavior_gate_snapshot(
        battle, scoring_envs,
    )
    assert first_snapshot is not None and first_snapshot.passed

    await scheduler._decide(
        champion, battle, scoring_envs, task_state, current_block=20,
    )

    champion = await state.get_champion()
    assert champion is not None and champion.uid == 1
    assert len(gate.calls) == 2
    if expected_terminal:
        assert await state.get_battle() is None
        assert miners.rows[2]["challenge_status"] == STATUS_TERMINATED
        assert miners.rows[2]["termination_reason"].startswith("model_behavior:")
        assert deployer.teardowns == ["dep-challenger"]
    else:
        assert await state.get_battle() is not None
        assert miners.rows[2]["challenge_status"] == STATUS_IN_PROGRESS
        assert deployer.teardowns == []


@pytest.mark.asyncio
async def test_promotion_seals_after_old_champion_teardown(monkeypatch):
    scheduler, state, miners, deployer, gate, samples = (
        await _build_active_behavior_gate_case(status="passed")
    )
    _fill_behavior_gate_winning_samples(samples)
    events = []
    original_teardown = scheduler._teardown_record
    original_seal = gate.seal_for_promotion

    async def tracked_teardown(record):
        events.append(f"teardown:{record.deployment_id}")
        await original_teardown(record)

    async def tracked_seal(*identity):
        events.append("seal")
        return await original_seal(*identity)

    monkeypatch.setattr(scheduler, "_teardown_record", tracked_teardown)
    monkeypatch.setattr(gate, "seal_for_promotion", tracked_seal)

    champion = await state.get_champion()
    battle = await state.get_battle()
    task_state = await state.get_task_state()
    envs = await state.get_scoring_environments()
    assert champion is not None and battle is not None and task_state is not None

    await scheduler._decide(
        champion, battle, envs, task_state, current_block=20,
    )

    promoted = await state.get_champion()
    assert promoted is not None and promoted.uid == 2
    assert events[:2] == ["teardown:dep-champion", "seal"]
    assert len(gate.seal_calls) == 1
    assert miners.rows[2]["challenge_status"] == STATUS_CHAMPION
    assert deployer.teardowns == ["dep-champion"]


@pytest.mark.asyncio
async def test_gate_failure_during_teardown_window_rolls_back_promotion():
    scheduler, state, miners, deployer, gate, samples = (
        await _build_active_behavior_gate_case(
            status=["passed", "failed"],
            seal_results=[False],
        )
    )
    _fill_behavior_gate_winning_samples(samples)
    champion = await state.get_champion()
    battle = await state.get_battle()
    task_state = await state.get_task_state()
    envs = await state.get_scoring_environments()
    assert champion is not None and battle is not None and task_state is not None

    await scheduler._decide(
        champion, battle, envs, task_state, current_block=20,
    )

    retained = await state.get_champion()
    assert retained is not None and retained.uid == 1
    assert retained.deployment_id is None
    assert retained.base_url is None
    assert await state.get_battle() is None
    assert miners.rows[2]["challenge_status"] == STATUS_TERMINATED
    assert miners.rows[2]["termination_reason"].startswith("model_behavior:")
    assert deployer.teardowns == ["dep-champion", "dep-challenger"]
    assert len(gate.seal_calls) == 1
    assert len(gate.calls) == 2


@pytest.mark.asyncio
async def test_pending_promotion_seal_pauses_with_old_deployment_cleared():
    scheduler, state, miners, deployer, gate, samples = (
        await _build_active_behavior_gate_case(
            status=["passed", "pending"],
            seal_results=[False],
        )
    )
    _fill_behavior_gate_winning_samples(samples)
    champion = await state.get_champion()
    battle = await state.get_battle()
    task_state = await state.get_task_state()
    envs = await state.get_scoring_environments()
    assert champion is not None and battle is not None and task_state is not None

    await scheduler._decide(
        champion, battle, envs, task_state, current_block=20,
    )

    retained = await state.get_champion()
    assert retained is not None and retained.uid == 1
    assert retained.deployment_id is None
    assert retained.base_url is None
    assert await state.get_battle() is not None
    assert miners.rows[2]["challenge_status"] == STATUS_IN_PROGRESS
    assert deployer.teardowns == ["dep-champion"]
    assert len(gate.seal_calls) == 1
    assert len(gate.calls) == 2


@pytest.mark.asyncio
async def test_behavior_gate_loss_retries_teardown_before_marking_terminal(
    monkeypatch,
):
    scheduler, state, miners, _deployer, _gate, _samples = (
        await _build_active_behavior_gate_case(status="failed")
    )
    attempts = 0

    async def flaky_teardown(_record):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient teardown failure")

    monkeypatch.setattr(scheduler, "_teardown_record", flaky_teardown)

    with pytest.raises(RuntimeError, match="transient teardown failure"):
        await scheduler.tick(current_block=20)

    assert await state.get_battle() is not None
    assert miners.rows[2]["challenge_status"] == STATUS_IN_PROGRESS

    await scheduler.tick(current_block=21)

    assert attempts == 2
    assert await state.get_battle() is None
    assert miners.rows[2]["challenge_status"] == STATUS_TERMINATED
    assert miners.rows[2]["termination_reason"].startswith("model_behavior:")


@pytest.mark.asyncio
async def test_failed_predeployed_gate_tears_down_and_marks_lost():
    scheduler, state, miners, deployer, gate, _samples = (
        await _build_active_behavior_gate_case(status="failed")
    )
    record = await state.get_battle()
    assert record is not None
    await state.clear_battle()
    await state.set_predeployed_challengers([record])

    await scheduler._predeploy_behavior_gate_sweep()

    assert await state.get_predeployed_challengers() == []
    assert miners.rows[2]["challenge_status"] == STATUS_TERMINATED
    assert miners.rows[2]["termination_reason"].endswith(":predeploy")
    assert deployer.teardowns == ["dep-challenger"]
    assert len(gate.calls) == 1


@pytest.mark.asyncio
async def test_first_tick_refreshes_task_ids_then_returns():
    kv = _seed_state()
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
    ])
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store,
        deployer=_DeployTracker(), samples=_SamplesFake(),
    )

    await scheduler.tick(current_block=50)

    task_state = await state.get_task_state()
    assert task_state is not None
    assert set(task_state.task_ids.keys()) == {"ENV_A", "ENV_B"}
    assert task_state.refreshed_at_block == 50


@pytest.mark.asyncio
async def test_single_instance_refresh_clears_champion_base_url_without_deployment_id():
    kv = _seed_state()
    kv.data["champion"]["deployment_id"] = None
    kv.data["champion"]["base_url"] = "http://old-endpoint/v1"
    kv.data["champion"]["deployments"] = [
        {
            "endpoint_name": "lium-b200-1",
            "deployment_id": "ssh:lium-b200-1:affine-sglang-current",
            "base_url": "http://old-endpoint/v1",
        }
    ]
    miner_store = _InMemoryMinerStore([
        _make_miner(
            1, "champ_hk", 100,
            status=STATUS_CHAMPION, revision="champ_rev",
        ),
    ])
    scheduler, state, _ = _build_scheduler(
        kv=kv,
        miner_store=miner_store,
        deployer=_DeployTracker(),
        samples=_SamplesFake(),
        config=FlowConfig(
            window_blocks=WINDOW_BLOCKS,
            single_instance_provider=True,
        ),
    )

    await scheduler.tick(current_block=50)

    champ = await state.get_champion()
    assert champ.deployment_id is None
    assert champ.base_url is None
    assert champ.deployments == []


@pytest.mark.asyncio
async def test_task_pool_refresh_interval_is_independent_from_window_id_blocks():
    kv = _seed_state()
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
    ])
    scheduler, state, _ = _build_scheduler(
        kv=kv,
        miner_store=miner_store,
        deployer=_DeployTracker(),
        samples=_SamplesFake(),
        window_blocks=WINDOW_BLOCKS,
        task_pool_refresh_blocks=WINDOW_BLOCKS * 10,
    )

    await scheduler.tick(current_block=50)
    first_task = await state.get_task_state()
    assert first_task.refreshed_at_block == 50

    # window_blocks elapsed, but task_pool_refresh_blocks has not.
    await scheduler.tick(current_block=50 + WINDOW_BLOCKS + 1)
    same_task = await state.get_task_state()
    assert same_task.refreshed_at_block == 50

    await scheduler.tick(current_block=50 + WINDOW_BLOCKS * 10 + 1)
    refreshed_task = await state.get_task_state()
    assert refreshed_task.refreshed_at_block == 50 + WINDOW_BLOCKS * 10 + 1


@pytest.mark.asyncio
async def test_window_rotation_request_tears_down_battle_then_refreshes():
    kv = _seed_state()
    kv.data["current_battle"] = {
        "challenger": {
            "uid": 2,
            "hotkey": "chal_hk",
            "revision": "chal_rev",
            "model": "org/chal",
        },
        "deployment_id": "wrk-chal",
        "base_url": "https://t/wrk-chal",
        "started_at_block": 90,
    }
    kv.data["current_task_ids"] = {
        "task_ids": {"ENV_A": [1, 2, 3], "ENV_B": [4, 5, 6]},
        "refreshed_at_block": 95,
    }
    kv.data["window_rotation_request"] = {
        "requested_at_block": 1000,
        "stale_refreshed_at_block": 1000 - WINDOW_BLOCKS - 1,
    }
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
        _make_miner(2, "chal_hk", 200, status=STATUS_IN_PROGRESS, revision="chal_rev"),
    ])
    deployer = _DeployTracker()
    scheduler, state, _ = _build_scheduler(
        kv=kv,
        miner_store=miner_store,
        deployer=deployer,
        samples=_SamplesFake(),
        window_blocks=WINDOW_BLOCKS,
        task_pool_refresh_blocks=WINDOW_BLOCKS * 10,
    )

    await scheduler.tick(current_block=1000)

    assert deployer.teardowns == ["wrk-chal"]
    assert miner_store.rows[2]["challenge_status"] == STATUS_SAMPLING
    assert await state.get_battle() is None
    assert await state.get_window_rotation_request() is None
    task_state = await state.get_task_state()
    assert task_state.refreshed_at_block == 1000


@pytest.mark.asyncio
async def test_window_rotation_request_retries_when_teardown_fails():
    kv = _seed_state()
    kv.data["current_battle"] = {
        "challenger": {
            "uid": 2,
            "hotkey": "chal_hk",
            "revision": "chal_rev",
            "model": "org/chal",
        },
        "deployment_id": "wrk-001",
        "base_url": "https://t/wrk-001",
        "started_at_block": 90,
    }
    kv.data["current_task_ids"] = {
        "task_ids": {"ENV_A": [1, 2, 3], "ENV_B": [4, 5, 6]},
        "refreshed_at_block": 95,
    }
    kv.data["window_rotation_request"] = {
        "requested_at_block": 1000,
        "stale_refreshed_at_block": -1,
    }
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
        _make_miner(2, "chal_hk", 200, status=STATUS_IN_PROGRESS, revision="chal_rev"),
    ])
    deployer = _DeployTracker(next_deployment_id=1)
    deployer.teardown_failures_once.add("wrk-001")
    scheduler, state, _ = _build_scheduler(
        kv=kv,
        miner_store=miner_store,
        deployer=deployer,
        samples=_SamplesFake(),
    )

    await scheduler.tick(current_block=1000)

    assert deployer.teardowns == ["wrk-001"]
    assert miner_store.rows[2]["challenge_status"] == STATUS_IN_PROGRESS
    assert await state.get_battle() is not None
    assert await state.get_window_rotation_request() is not None
    task_state = await state.get_task_state()
    assert task_state.refreshed_at_block == 95


@pytest.mark.asyncio
async def test_task_refresh_blocked_during_battle():
    kv = _seed_state()
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
        _make_miner(2, "chal_hk", 200, revision="chal_rev"),
    ])
    deployer = _DeployTracker()
    samples = _SamplesFake()
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store, deployer=deployer, samples=samples,
    )

    # Tick 1: refresh
    await scheduler.tick(current_block=50)
    # Tick 2: deploy champion
    await scheduler.tick(current_block=51)
    # Fill champion samples
    task_state = await state.get_task_state()
    for env in ("ENV_A", "ENV_B"):
        samples.set_samples("champ_hk", "champ_rev", env, task_state.task_ids[env], refresh_block=task_state.refreshed_at_block)
    # Tick 3: start battle
    await scheduler.tick(current_block=52)
    battle = await state.get_battle()
    assert battle is not None and battle.challenger.uid == 2
    first_refresh = task_state.refreshed_at_block

    # Jump forward many refresh intervals while battle is in flight — no refresh.
    await scheduler.tick(current_block=52 + WINDOW_BLOCKS * 100)
    new_task = await state.get_task_state()
    assert new_task.refreshed_at_block == first_refresh


@pytest.mark.asyncio
async def test_cold_start_promotes_first_pending_miner_no_deploy():
    kv = _seed_state(with_champion=False)
    miner_store = _InMemoryMinerStore([
        _make_miner(5, "early", 50, revision="r5"),
        _make_miner(7, "late", 200, revision="r7"),
    ])
    deployer = _DeployTracker()
    samples = _SamplesFake()
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store, deployer=deployer, samples=samples,
    )

    # Tick 1: refresh
    await scheduler.tick(current_block=50)
    # Tick 2: cold start
    await scheduler.tick(current_block=51)
    champ = await state.get_champion()
    assert champ is not None and champ.uid == 5
    # No Targon deploys yet — that happens on the next tick.
    assert deployer.deploys == []
    # Promoted miner now has STATUS_CHAMPION.
    assert miner_store.rows[5]["challenge_status"] == STATUS_CHAMPION


@pytest.mark.asyncio
async def test_champion_targon_deployed_on_second_tick():
    kv = _seed_state()
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
    ])
    deployer = _DeployTracker()
    samples = _SamplesFake()
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store, deployer=deployer, samples=samples,
    )

    await scheduler.tick(current_block=50)   # refresh task_ids
    await scheduler.tick(current_block=51)   # deploy champion
    champ = await state.get_champion()
    assert champ.deployment_id == "wrk-001"
    assert champ.base_url == "https://t/wrk-001"
    assert len(deployer.deploys) == 1


@pytest.mark.asyncio
async def test_unhealthy_champion_deployment_is_cleared_and_redeployed():
    """A persisted deployment_id/base_url is only a scheduler record; the
    underlying sglang process can exit later. A failed runtime health check
    must clear that stale state and run the normal champion deploy path so
    executor stops sampling a dead URL."""
    kv = _seed_state()
    kv.data["current_task_ids"] = {
        "task_ids": {"ENV_A": [1, 2, 3, 4], "ENV_B": [5, 6, 7, 8]},
        "refreshed_at_block": 50,
    }
    kv.data["champion"]["deployment_id"] = "wrk-stale"
    kv.data["champion"]["base_url"] = "https://t/stale"
    kv.data["champion"]["deployments"] = [
        {
            "endpoint_name": "endpoint-1",
            "deployment_id": "wrk-stale",
            "base_url": "https://t/stale",
        }
    ]
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
    ])
    deployer = _DeployTracker()
    health_checks = []

    async def deployment_health_fn(champion):
        health_checks.append((champion.uid, champion.deployment_id))
        return False

    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store, deployer=deployer,
        samples=_SamplesFake(), deployment_health_fn=deployment_health_fn,
    )

    await scheduler.tick(current_block=51)

    assert health_checks == [(1, "wrk-stale")]
    assert [d.uid for d in deployer.deploys] == [1]
    champ = await state.get_champion()
    assert champ.deployment_id == "wrk-001"
    assert champ.base_url == "https://t/wrk-001"


@pytest.mark.asyncio
async def test_healthy_champion_deployment_is_not_redeployed():
    kv = _seed_state()
    kv.data["current_task_ids"] = {
        "task_ids": {"ENV_A": [1, 2, 3, 4], "ENV_B": [5, 6, 7, 8]},
        "refreshed_at_block": 50,
    }
    kv.data["champion"]["deployment_id"] = "wrk-live"
    kv.data["champion"]["base_url"] = "https://t/live"
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
    ])
    deployer = _DeployTracker()

    async def deployment_health_fn(champion):
        return True

    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store, deployer=deployer,
        samples=_SamplesFake(), deployment_health_fn=deployment_health_fn,
    )

    await scheduler.tick(current_block=51)

    champ = await state.get_champion()
    assert champ.deployment_id == "wrk-live"
    assert champ.base_url == "https://t/live"
    assert deployer.deploys == []


@pytest.mark.asyncio
async def test_suspected_champion_requires_five_consecutive_failures():
    kv = _seed_state()
    kv.data["current_task_ids"] = {
        "task_ids": {"ENV_A": [1, 2, 3, 4], "ENV_B": [5, 6, 7, 8]},
        "refreshed_at_block": 50,
    }
    kv.data["champion"]["deployment_id"] = "wrk-suspected"
    kv.data["champion"]["base_url"] = "https://t/suspected"
    miner_store = _InMemoryMinerStore([
        _make_miner(
            1,
            "champ_hk",
            100,
            status=STATUS_CHAMPION,
            revision="champ_rev",
        ),
    ])
    deployer = _DeployTracker()

    async def deployment_health_fn(champion):
        return DeploymentHealthResult(
            DeploymentHealthState.SUSPECTED,
            reason="both_probes_failed",
        )

    scheduler, state, _ = _build_scheduler(
        kv=kv,
        miner_store=miner_store,
        deployer=deployer,
        samples=_SamplesFake(),
        deployment_health_fn=deployment_health_fn,
    )

    await scheduler.tick(current_block=51)
    await scheduler.tick(current_block=52)
    await scheduler.tick(current_block=53)
    await scheduler.tick(current_block=54)
    assert deployer.deploys == []
    assert (await state.get_champion()).deployment_id == "wrk-suspected"

    await scheduler.tick(current_block=55)
    assert [target.uid for target in deployer.deploys] == [1]
    assert (await state.get_champion()).deployment_id == "wrk-001"


@pytest.mark.asyncio
async def test_health_failure_count_resets_for_new_endpoint_generation():
    kv = _seed_state()
    kv.data["current_task_ids"] = {
        "task_ids": {"ENV_A": [1, 2, 3, 4], "ENV_B": [5, 6, 7, 8]},
        "refreshed_at_block": 50,
    }
    kv.data["champion"]["deployment_id"] = "wrk-stable-name"
    kv.data["champion"]["base_url"] = "https://t/live"
    miner_store = _InMemoryMinerStore([
        _make_miner(
            1,
            "champ_hk",
            100,
            status=STATUS_CHAMPION,
            revision="champ_rev",
        ),
    ])
    deployer = _DeployTracker()
    identities = iter(["generation-1", "generation-1", "generation-2"])

    async def deployment_health_fn(champion):
        return DeploymentHealthResult(
            DeploymentHealthState.SUSPECTED,
            reason="both_probes_failed",
            identity=next(identities),
        )

    scheduler, state, _ = _build_scheduler(
        kv=kv,
        miner_store=miner_store,
        deployer=deployer,
        samples=_SamplesFake(),
        deployment_health_fn=deployment_health_fn,
    )

    await scheduler.tick(current_block=51)
    await scheduler.tick(current_block=52)
    await scheduler.tick(current_block=53)

    assert deployer.deploys == []
    assert (await state.get_champion()).deployment_id == "wrk-stable-name"


@pytest.mark.asyncio
async def test_runtime_health_checks_respect_configured_interval(monkeypatch):
    kv = _seed_state()
    kv.data["current_task_ids"] = {
        "task_ids": {"ENV_A": [1, 2, 3, 4], "ENV_B": [5, 6, 7, 8]},
        "refreshed_at_block": 50,
    }
    kv.data["champion"]["deployment_id"] = "wrk-live"
    kv.data["champion"]["base_url"] = "https://t/live"
    miner_store = _InMemoryMinerStore([
        _make_miner(
            1,
            "champ_hk",
            100,
            status=STATUS_CHAMPION,
            revision="champ_rev",
        ),
    ])
    deployer = _DeployTracker()
    health_checks = []
    now = [100.0]

    async def deployment_health_fn(record):
        health_checks.append(record.deployment_id)
        return DeploymentHealthResult(
            DeploymentHealthState.SUSPECTED,
            reason="both_probes_failed",
        )

    monkeypatch.setattr(
        "affine.src.scheduler.flow.time.monotonic",
        lambda: now[0],
    )
    scheduler, state, _ = _build_scheduler(
        kv=kv,
        miner_store=miner_store,
        deployer=deployer,
        samples=_SamplesFake(),
        config=FlowConfig(
            window_blocks=WINDOW_BLOCKS,
            deployment_health_check_interval_sec=240,
        ),
        deployment_health_fn=deployment_health_fn,
    )

    await scheduler.tick(current_block=51)
    now[0] = 339.0
    await scheduler.tick(current_block=52)
    assert health_checks == ["wrk-live"]

    now[0] = 340.0
    await scheduler.tick(current_block=53)
    assert health_checks == ["wrk-live", "wrk-live"]
    assert deployer.deploys == []
    assert (await state.get_champion()).deployment_id == "wrk-live"


@pytest.mark.asyncio
async def test_transport_failure_repairs_tunnel_without_clearing_deployment():
    kv = _seed_state()
    kv.data["current_task_ids"] = {
        "task_ids": {"ENV_A": [1, 2, 3, 4], "ENV_B": [5, 6, 7, 8]},
        "refreshed_at_block": 50,
    }
    kv.data["champion"]["deployment_id"] = "wrk-live"
    kv.data["champion"]["base_url"] = "https://t/live"
    miner_store = _InMemoryMinerStore([
        _make_miner(
            1,
            "champ_hk",
            100,
            status=STATUS_CHAMPION,
            revision="champ_rev",
        ),
    ])
    deployer = _DeployTracker()
    repairs = []

    async def deployment_health_fn(champion):
        return DeploymentHealthResult(
            DeploymentHealthState.TRANSPORT_UNHEALTHY,
            reason="public_failed_local_ready",
        )

    async def repair_fn(champion, health):
        repairs.append((champion.deployment_id, health.reason))
        return True

    scheduler, state, _ = _build_scheduler(
        kv=kv,
        miner_store=miner_store,
        deployer=deployer,
        samples=_SamplesFake(),
        deployment_health_fn=deployment_health_fn,
        deployment_transport_repair_fn=repair_fn,
    )

    await scheduler.tick(current_block=51)
    await scheduler.tick(current_block=52)
    await scheduler.tick(current_block=53)
    await scheduler.tick(current_block=54)
    await scheduler.tick(current_block=55)

    assert repairs == [("wrk-live", "public_failed_local_ready")]
    assert deployer.deploys == []
    champion = await state.get_champion()
    assert champion.deployment_id == "wrk-live"
    assert champion.base_url == "https://t/live"


@pytest.mark.asyncio
async def test_active_challenger_transport_failure_repairs_without_losing_battle():
    kv = _seed_state()
    kv.data["current_task_ids"] = {
        "task_ids": {"ENV_A": [1, 2, 3, 4], "ENV_B": [5, 6, 7, 8]},
        "refreshed_at_block": 50,
    }
    miner_store = _InMemoryMinerStore([
        _make_miner(
            1,
            "champ_hk",
            100,
            status=STATUS_CHAMPION,
            revision="champ_rev",
        ),
        _make_miner(
            2,
            "chal_hk",
            200,
            status=STATUS_IN_PROGRESS,
            revision="chal_rev",
        ),
    ])
    deployer = _DeployTracker()
    health_checks = []
    repairs = []

    async def deployment_health_fn(record):
        health_checks.append(record.challenger.uid)
        return DeploymentHealthResult(
            DeploymentHealthState.TRANSPORT_UNHEALTHY,
            reason="public_probe_failed_local_ready",
            identity="endpoint-1:instance-1:generation=1",
        )

    async def repair_fn(record, health):
        repairs.append((record.challenger.uid, health.identity))
        return True

    scheduler, state, _ = _build_scheduler(
        kv=kv,
        miner_store=miner_store,
        deployer=deployer,
        samples=_SamplesFake(),
        deployment_health_fn=deployment_health_fn,
        deployment_transport_repair_fn=repair_fn,
    )
    await state.set_battle(BattleRecord(
        challenger=MinerSnapshot(
            uid=2,
            hotkey="chal_hk",
            revision="chal_rev",
            model="org/challenger",
        ),
        deployment_id="wrk-challenger",
        base_url="https://t/challenger",
        started_at_block=50,
    ))

    await scheduler.tick(current_block=51)
    await scheduler.tick(current_block=52)
    await scheduler.tick(current_block=53)
    await scheduler.tick(current_block=54)
    await scheduler.tick(current_block=55)

    assert health_checks == [2, 2, 2, 2, 2]
    assert repairs == [(2, "endpoint-1:instance-1:generation=1")]
    battle = await state.get_battle()
    assert battle is not None
    assert battle.challenger.uid == 2
    assert battle.deployment_id == "wrk-challenger"
    assert miner_store.rows[2]["challenge_status"] == STATUS_IN_PROGRESS
    assert deployer.teardowns == []


@pytest.mark.asyncio
async def test_unknown_runtime_health_queues_fenced_endpoint_replacement():
    kv = _seed_state()
    kv.data["current_task_ids"] = {
        "task_ids": {"ENV_A": [1, 2, 3, 4], "ENV_B": [5, 6, 7, 8]},
        "refreshed_at_block": 50,
    }
    kv.data["champion"].update({
        "deployment_id": "wrk-live",
        "base_url": "https://t/live",
    })
    miner_store = _InMemoryMinerStore([
        _make_miner(
            1,
            "champ_hk",
            100,
            status=STATUS_CHAMPION,
            revision="champ_rev",
        ),
    ])
    deployer = _DeployTracker()
    replacements = []

    async def deployment_health_fn(record):
        return DeploymentHealthResult(
            DeploymentHealthState.UNKNOWN,
            reason="container_inspect_and_public_probe_unavailable",
            identity="endpoint-1:instance-1:generation=4",
        )

    async def endpoint_repair_fn(record, health):
        replacements.append((record.deployment_id, health.identity))
        return True

    scheduler, state, _ = _build_scheduler(
        kv=kv,
        miner_store=miner_store,
        deployer=deployer,
        samples=_SamplesFake(),
        deployment_health_fn=deployment_health_fn,
        deployment_endpoint_repair_fn=endpoint_repair_fn,
    )

    for block in range(51, 56):
        await scheduler.tick(current_block=block)

    assert replacements == [
        ("wrk-live", "endpoint-1:instance-1:generation=4")
    ]
    champion = await state.get_champion()
    assert champion.deployment_id == "wrk-live"
    assert champion.base_url == "https://t/live"
    assert deployer.deploys == []
    assert deployer.teardowns == []


@pytest.mark.asyncio
async def test_healthy_runtime_synchronizes_canonical_endpoint_url():
    kv = _seed_state()
    kv.data["current_task_ids"] = {
        "task_ids": {"ENV_A": [1, 2, 3, 4], "ENV_B": [5, 6, 7, 8]},
        "refreshed_at_block": 50,
    }
    kv.data["champion"].update({
        "deployment_id": "wrk-live",
        "base_url": "http://old-tunnel:8101/v1",
        "deployments": [
            {
                "endpoint_name": "endpoint-1",
                "deployment_id": "wrk-live",
                "base_url": "http://old-tunnel:8101/v1",
            }
        ],
    })
    miner_store = _InMemoryMinerStore([
        _make_miner(
            1,
            "champ_hk",
            100,
            status=STATUS_CHAMPION,
            revision="champ_rev",
        ),
    ])

    async def deployment_health_fn(record):
        return DeploymentHealthResult(
            DeploymentHealthState.HEALTHY,
            identity="endpoint-1:instance-2:generation=2",
            canonical_base_url="http://new-tunnel:8102/v1",
        )

    scheduler, state, _ = _build_scheduler(
        kv=kv,
        miner_store=miner_store,
        deployer=_DeployTracker(),
        samples=_SamplesFake(),
        deployment_health_fn=deployment_health_fn,
    )

    await scheduler.tick(current_block=51)

    champion = await state.get_champion()
    assert champion.base_url == "http://new-tunnel:8102/v1"
    assert champion.deployments == [
        DeploymentRecord(
            endpoint_name="endpoint-1",
            deployment_id="wrk-live",
            base_url="http://new-tunnel:8102/v1",
        )
    ]


@pytest.mark.asyncio
async def test_healthy_challenger_synchronizes_executor_endpoint_url():
    kv = _seed_state()
    kv.data["current_task_ids"] = {
        "task_ids": {"ENV_A": [1, 2, 3, 4], "ENV_B": [5, 6, 7, 8]},
        "refreshed_at_block": 50,
    }
    miner_store = _InMemoryMinerStore([
        _make_miner(
            1,
            "champ_hk",
            100,
            status=STATUS_CHAMPION,
            revision="champ_rev",
        ),
        _make_miner(
            2,
            "chal_hk",
            200,
            status=STATUS_IN_PROGRESS,
            revision="chal_rev",
        ),
    ])

    async def deployment_health_fn(record):
        return DeploymentHealthResult(
            DeploymentHealthState.HEALTHY,
            identity="endpoint-1:instance-2:generation=2",
            canonical_base_url="http://new-tunnel:8102/v1",
        )

    scheduler, state, _ = _build_scheduler(
        kv=kv,
        miner_store=miner_store,
        deployer=_DeployTracker(),
        samples=_SamplesFake(),
        deployment_health_fn=deployment_health_fn,
    )
    await state.set_battle(BattleRecord(
        challenger=MinerSnapshot(
            uid=2,
            hotkey="chal_hk",
            revision="chal_rev",
            model="org/challenger",
        ),
        deployment_id="wrk-challenger",
        base_url="http://old-tunnel:8101/v1",
        deployments=[DeploymentRecord(
            endpoint_name="endpoint-1",
            deployment_id="wrk-challenger",
            base_url="http://old-tunnel:8101/v1",
        )],
        started_at_block=50,
    ))

    await scheduler.tick(current_block=51)

    battle = await state.get_battle()
    assert battle is not None
    assert battle.base_url == "http://new-tunnel:8102/v1"
    assert battle.deployments[0].base_url == "http://new-tunnel:8102/v1"


@pytest.mark.asyncio
async def test_late_health_result_does_not_restore_drained_runtime_state():
    kv = _seed_state()
    kv.data["current_task_ids"] = {
        "task_ids": {"ENV_A": [1, 2, 3, 4], "ENV_B": [5, 6, 7, 8]},
        "refreshed_at_block": 50,
    }
    kv.data["champion"].update({
        "deployment_id": "wrk-drained",
        "base_url": "http://old-tunnel:8101/v1",
    })
    miner_store = _InMemoryMinerStore([
        _make_miner(
            1,
            "champ_hk",
            100,
            status=STATUS_CHAMPION,
            revision="champ_rev",
        ),
    ])
    state = StateStore(kv)

    async def deployment_health_fn(record):
        await state.clear_champion()
        return DeploymentHealthResult(
            DeploymentHealthState.HEALTHY,
            identity="endpoint-1:instance-2:generation=2",
            canonical_base_url="http://new-tunnel:8102/v1",
        )

    scheduler, _, _ = _build_scheduler(
        kv=kv,
        miner_store=miner_store,
        deployer=_DeployTracker(),
        samples=_SamplesFake(),
        deployment_health_fn=deployment_health_fn,
    )

    await scheduler.tick(current_block=51)

    assert await state.get_champion() is None


@pytest.mark.asyncio
async def test_no_active_endpoint_waits_without_deploying_or_predeploying():
    kv = _seed_state()
    miner_store = _InMemoryMinerStore([
        _make_miner(
            1, "champ_hk", 100, status=STATUS_CHAMPION,
            revision="champ_rev",
        ),
        _make_miner(2, "chal_hk", 200, revision="chal_rev"),
    ])
    deployer = _DeployTracker(pre_challenger_slots=1)
    samples = _SamplesFake()

    async def no_active_endpoints():
        return set()

    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store, deployer=deployer, samples=samples,
        config=FlowConfig(
            window_blocks=WINDOW_BLOCKS,
            single_instance_provider=True,
        ),
        list_active_endpoint_names_fn=no_active_endpoints,
    )

    await scheduler.tick(current_block=50)   # refresh, then predeploy skips
    await scheduler.tick(current_block=51)   # champion deploy skips

    assert deployer.deploys == []
    assert deployer.predeploys == []
    assert await state.get_battle() is None
    assert (await state.get_predeployed_challengers()) == []
    assert miner_store.rows[2]["challenge_status"] == STATUS_SAMPLING


@pytest.mark.asyncio
async def test_no_active_endpoint_does_not_claim_challenger():
    kv = _seed_state()
    kv.data["current_task_ids"] = {
        "task_ids": {"ENV_A": [1, 2, 3, 4], "ENV_B": [5, 6, 7, 8]},
        "refreshed_at_block": 50,
    }
    kv.data["champion"]["deployment_id"] = "wrk-live"
    kv.data["champion"]["base_url"] = "https://t/live"
    miner_store = _InMemoryMinerStore([
        _make_miner(
            1, "champ_hk", 100, status=STATUS_CHAMPION,
            revision="champ_rev",
        ),
        _make_miner(2, "chal_hk", 200, revision="chal_rev"),
    ])
    deployer = _DeployTracker()
    samples = _SamplesFake()
    for env, task_ids in kv.data["current_task_ids"]["task_ids"].items():
        samples.set_samples(
            "champ_hk", "champ_rev", env, task_ids,
            refresh_block=50,
        )

    async def no_active_endpoints():
        return set()

    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store, deployer=deployer, samples=samples,
        config=FlowConfig(
            window_blocks=WINDOW_BLOCKS,
            single_instance_provider=True,
        ),
        list_active_endpoint_names_fn=no_active_endpoints,
    )

    await scheduler.tick(current_block=51)

    assert deployer.deploys == []
    assert await state.get_battle() is None
    assert miner_store.rows[2]["challenge_status"] == STATUS_SAMPLING


@pytest.mark.asyncio
async def test_battle_starts_only_after_champion_samples_complete():
    kv = _seed_state()
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
        _make_miner(2, "chal_hk", 200, revision="chal_rev"),
    ])
    deployer = _DeployTracker()
    samples = _SamplesFake()
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store, deployer=deployer, samples=samples,
    )

    await scheduler.tick(current_block=50)   # refresh
    await scheduler.tick(current_block=51)   # deploy champion
    await scheduler.tick(current_block=52)   # champion samples not ready → return
    assert await state.get_battle() is None
    # Champion samples ENV_A done, ENV_B partial → still no battle.
    task_state = await state.get_task_state()
    samples.set_samples("champ_hk", "champ_rev", "ENV_A", task_state.task_ids["ENV_A"], refresh_block=task_state.refreshed_at_block)
    samples.set_samples("champ_hk", "champ_rev", "ENV_B", task_state.task_ids["ENV_B"][:2], refresh_block=task_state.refreshed_at_block)
    await scheduler.tick(current_block=53)
    assert await state.get_battle() is None
    # All champion samples in → battle starts.
    samples.set_samples("champ_hk", "champ_rev", "ENV_B", task_state.task_ids["ENV_B"], refresh_block=task_state.refreshed_at_block)
    await scheduler.tick(current_block=54)
    battle = await state.get_battle()
    assert battle is not None and battle.challenger.uid == 2


@pytest.mark.asyncio
async def test_challenger_wins_promotes_and_writes_weights():
    kv = _seed_state()
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
        _make_miner(2, "chal_hk", 200, revision="chal_rev"),
    ])
    deployer = _DeployTracker()
    samples = _SamplesFake()
    weight_writer = _WeightWriterFake()
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store,
        deployer=deployer, samples=samples, weight_writer=weight_writer,
    )

    # Drive through to a complete battle.
    await scheduler.tick(current_block=50)   # refresh
    await scheduler.tick(current_block=51)   # deploy champ
    task_state = await state.get_task_state()
    for env in ("ENV_A", "ENV_B"):
        # Champion fully sampled at 0.5
        samples.set_samples("champ_hk", "champ_rev", env, task_state.task_ids[env], score=0.5, refresh_block=task_state.refreshed_at_block)
    await scheduler.tick(current_block=52)   # start battle
    battle = await state.get_battle()
    assert battle is not None
    # Challenger crushes champion: 0.9 across the board.
    for env in ("ENV_A", "ENV_B"):
        samples.set_samples("chal_hk", "chal_rev", env, task_state.task_ids[env], score=0.9, refresh_block=task_state.refreshed_at_block)
    await scheduler.tick(current_block=53)   # decide

    new_champ = await state.get_champion()
    assert new_champ.uid == 2
    assert new_champ.hotkey == "chal_hk"
    assert miner_store.rows[1]["challenge_status"] == STATUS_TERMINATED
    assert miner_store.rows[2]["challenge_status"] == STATUS_CHAMPION
    assert await state.get_battle() is None
    # Weight write happened (champion changed).
    assert len(weight_writer.calls) == 1
    # Old champion's Targon torn down; challenger's becomes new champion's.
    assert "wrk-001" in deployer.teardowns
    assert new_champ.deployment_id == "wrk-002"


@pytest.mark.asyncio
async def test_champion_holds_when_challenger_score_lower():
    kv = _seed_state()
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
        _make_miner(2, "chal_hk", 200, revision="chal_rev"),
    ])
    deployer = _DeployTracker()
    samples = _SamplesFake()
    weight_writer = _WeightWriterFake()
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store,
        deployer=deployer, samples=samples, weight_writer=weight_writer,
    )

    await scheduler.tick(current_block=50)
    await scheduler.tick(current_block=51)
    task_state = await state.get_task_state()
    for env in ("ENV_A", "ENV_B"):
        samples.set_samples("champ_hk", "champ_rev", env, task_state.task_ids[env], score=0.9, refresh_block=task_state.refreshed_at_block)
    await scheduler.tick(current_block=52)   # start battle
    for env in ("ENV_A", "ENV_B"):
        samples.set_samples("chal_hk", "chal_rev", env, task_state.task_ids[env], score=0.3, refresh_block=task_state.refreshed_at_block)
    await scheduler.tick(current_block=53)   # decide

    champ = await state.get_champion()
    assert champ.uid == 1  # unchanged
    assert miner_store.rows[2]["challenge_status"] == STATUS_TERMINATED
    assert await state.get_battle() is None
    # Challenger's Targon torn down, champion's preserved.
    assert "wrk-002" in deployer.teardowns
    assert "wrk-001" not in deployer.teardowns
    # No weight write — champion stable.
    assert weight_writer.calls == []


@pytest.mark.asyncio
async def test_displaced_champion_carries_final_scores_to_miner_stats():
    """When the challenger wins, the old champion is terminated. The
    comparator's decide-time view (its own count/avg + the new
    champion's overlap avg as threshold basis) is frozen onto the
    miner_stats row so the rank UI can keep showing real numbers after
    the live cache forgets them."""
    kv = _seed_state()
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
        _make_miner(2, "chal_hk", 200, revision="chal_rev"),
    ])
    samples = _SamplesFake()
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store,
        deployer=_DeployTracker(), samples=samples,
        weight_writer=_WeightWriterFake(),
    )

    await scheduler.tick(current_block=50)   # refresh task_ids
    await scheduler.tick(current_block=51)   # deploy champion
    task_state = await state.get_task_state()
    for env in ("ENV_A", "ENV_B"):
        samples.set_samples("champ_hk", "champ_rev", env, task_state.task_ids[env], score=0.5, refresh_block=task_state.refreshed_at_block)
    await scheduler.tick(current_block=52)   # start battle
    for env in ("ENV_A", "ENV_B"):
        samples.set_samples("chal_hk", "chal_rev", env, task_state.task_ids[env], score=0.9, refresh_block=task_state.refreshed_at_block)
    await scheduler.tick(current_block=99)   # decide

    old_champ_row = miner_store.rows[1]
    assert old_champ_row["challenge_status"] == STATUS_TERMINATED
    # Block recorded at decision time (current_block from the decide tick).
    assert old_champ_row["terminated_at_block"] == 99
    # ``scores_refresh_block`` is the task pool block the comparator
    # actually decided on; readers use it to ignore stale-pool rows
    # for non-terminated rows (terminated rows always render).
    assert old_champ_row["scores_refresh_block"] is not None
    final = old_champ_row["scores_by_env"]
    assert set(final.keys()) == {"ENV_A", "ENV_B"}
    for env in ("ENV_A", "ENV_B"):
        # Old champion's own avg on the overlap; the comparator's basis
        # for this row is the *winning challenger's* avg.
        assert final[env]["count"] > 0
        assert final[env]["avg"] == pytest.approx(0.5)
        assert final[env]["champion_overlap_avg"] == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_losing_challenger_carries_final_scores_to_miner_stats():
    """Challenger loses → terminated row picks up the same frozen
    snapshot, but with own/opponent flipped: challenger's avg as ``avg``,
    champion's avg as ``champion_overlap_avg``."""
    kv = _seed_state()
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
        _make_miner(2, "chal_hk", 200, revision="chal_rev"),
    ])
    samples = _SamplesFake()
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store,
        deployer=_DeployTracker(), samples=samples,
        weight_writer=_WeightWriterFake(),
    )

    await scheduler.tick(current_block=50)
    await scheduler.tick(current_block=51)
    task_state = await state.get_task_state()
    for env in ("ENV_A", "ENV_B"):
        samples.set_samples("champ_hk", "champ_rev", env, task_state.task_ids[env], score=0.9, refresh_block=task_state.refreshed_at_block)
    await scheduler.tick(current_block=52)
    for env in ("ENV_A", "ENV_B"):
        samples.set_samples("chal_hk", "chal_rev", env, task_state.task_ids[env], score=0.3, refresh_block=task_state.refreshed_at_block)
    await scheduler.tick(current_block=77)

    loser_row = miner_store.rows[2]
    assert loser_row["challenge_status"] == STATUS_TERMINATED
    assert loser_row["terminated_at_block"] == 77
    final = loser_row["scores_by_env"]
    for env in ("ENV_A", "ENV_B"):
        assert final[env]["avg"] == pytest.approx(0.3)
        assert final[env]["champion_overlap_avg"] == pytest.approx(0.9)
    # The (winning) champion's row keeps its CHAMPION status. The
    # _decide LOST path doesn't touch the winner — their scores stay
    # under live-monitor management until they later get displaced.
    assert miner_store.rows[1]["challenge_status"] == STATUS_CHAMPION
    assert "terminated_at_block" not in miner_store.rows[1]


@pytest.mark.asyncio
async def test_losing_challenger_freezes_sampling_only_env():
    """Sampling-only envs (``enabled_for_scoring=false`` — distill-v2)
    never enter the comparator, so the frozen ``scores_by_env`` would
    otherwise drop them and ``af get-rank`` would show the column blank
    for every terminated miner. The decide path must read the loser's
    own sampling-only samples and freeze them alongside the scoring
    envs, with the champion's overlap avg as the threshold basis."""
    kv = _seed_state()
    # Add a sampling-only env (sampled but not scored).
    kv.data["environments"]["ENV_S"] = {
        "display_name": "S", "enabled_for_sampling": True,
        "enabled_for_scoring": False,
        "sampling": {"sampling_count": 4, "dataset_range": [[0, 1000]],
                     "sampling_mode": "random"},
    }
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
        _make_miner(2, "chal_hk", 200, revision="chal_rev"),
    ])
    samples = _SamplesFake()
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store,
        deployer=_DeployTracker(), samples=samples,
        weight_writer=_WeightWriterFake(),
    )

    await scheduler.tick(current_block=50)
    await scheduler.tick(current_block=51)
    task_state = await state.get_task_state()
    rb = task_state.refreshed_at_block
    # Champion fills every sampling env (scoring + sampling-only) so the
    # battle reaches the decide step (overlap readiness checks all envs).
    for env in ("ENV_A", "ENV_B"):
        samples.set_samples("champ_hk", "champ_rev", env, task_state.task_ids[env], score=0.9, refresh_block=rb)
    samples.set_samples("champ_hk", "champ_rev", "ENV_S", task_state.task_ids["ENV_S"], score=0.7, refresh_block=rb)
    await scheduler.tick(current_block=52)
    for env in ("ENV_A", "ENV_B"):
        samples.set_samples("chal_hk", "chal_rev", env, task_state.task_ids[env], score=0.3, refresh_block=rb)
    samples.set_samples("chal_hk", "chal_rev", "ENV_S", task_state.task_ids["ENV_S"], score=0.4, refresh_block=rb)
    await scheduler.tick(current_block=77)

    loser_row = miner_store.rows[2]
    assert loser_row["challenge_status"] == STATUS_TERMINATED
    final = loser_row["scores_by_env"]
    # Scoring envs frozen as before, plus the sampling-only env.
    assert set(final.keys()) == {"ENV_A", "ENV_B", "ENV_S"}
    assert final["ENV_S"]["avg"] == pytest.approx(0.4)        # loser's own avg
    assert final["ENV_S"]["count"] > 0
    assert final["ENV_S"]["champion_overlap_avg"] == pytest.approx(0.7)  # basis
    # Opponent (champion) view also carries the sampling-only env.
    assert loser_row["opponent_scores_by_env"]["ENV_S"]["avg"] == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_deploy_failure_keeps_challenger_in_queue():
    """A deploy failure (host crash / sglang OSError / HF transient)
    must not mark the miner FAILED. ``pick_next`` already flipped the
    row to ``in_progress``; the failure path releases the claim back
    to ``sampling`` so the same uid stays re-pickable next tick."""
    kv = _seed_state()
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
        _make_miner(2, "chal_hk", 200, revision="chal_rev"),
    ])
    samples = _SamplesFake()
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store, deployer=_DeployTracker(), samples=samples,
    )

    # Get champion samples ready.
    await scheduler.tick(current_block=50)
    await scheduler.tick(current_block=51)
    task_state = await state.get_task_state()
    for env in ("ENV_A", "ENV_B"):
        samples.set_samples("champ_hk", "champ_rev", env, task_state.task_ids[env], refresh_block=task_state.refreshed_at_block)

    # Now break deploys for the challenger.
    failing_deployer = _DeployTracker(fail_on_deploy=True)
    scheduler._deploy = failing_deployer.deploy
    scheduler._teardown = failing_deployer.teardown

    await scheduler.tick(current_block=52)
    # No mark FAILED — miner stays usable. Status was flipped to
    # ``in_progress`` by pick_next and released back to ``sampling``
    # by the failure handler.
    assert miner_store.rows[2]["challenge_status"] != STATUS_TERMINATED
    assert miner_store.rows[2]["challenge_status"] == STATUS_SAMPLING
    assert miner_store.rows[2].get("termination_reason") in (None, "")
    assert await state.get_battle() is None


@pytest.mark.asyncio
async def test_deploy_failure_forgets_invalidated_champion_deployment():
    """Deploy providers report which deployment ids were invalidated
    before failure. Flow forgets matching state references generically,
    without relying on role-specific champion/challenger assumptions."""
    kv = _seed_state()
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
        _make_miner(2, "chal_hk", 200, revision="chal_rev"),
    ])
    samples = _SamplesFake()
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store, deployer=_DeployTracker(), samples=samples,
    )

    await scheduler.tick(current_block=50)
    await scheduler.tick(current_block=51)
    task_state = await state.get_task_state()
    for env in ("ENV_A", "ENV_B"):
        samples.set_samples(
            "champ_hk", "champ_rev", env, task_state.task_ids[env],
            refresh_block=task_state.refreshed_at_block,
        )
    champ = await state.get_champion()
    assert champ.deployment_id is not None
    assert champ.base_url is not None
    invalidated_deployment_id = champ.deployment_id

    async def failing_deploy(target, role: str = "active"):
        raise DeploymentStateInvalidatedError(
            "endpoint was cleared before deploy failed",
            deployment_ids=[invalidated_deployment_id],
        )

    scheduler._deploy = failing_deploy

    await scheduler.tick(current_block=52)

    assert miner_store.rows[2]["challenge_status"] == STATUS_SAMPLING
    assert await state.get_battle() is None
    champ = await state.get_champion()
    assert champ.deployment_id is None
    assert champ.base_url is None
    assert champ.deployments == []


@pytest.mark.asyncio
async def test_deregistered_champion_retained_as_burn_sentinel():
    kv = _seed_state()
    # Champion uid=1 in system_config, but NOT in valid miners list.
    miner_store = _InMemoryMinerStore([
        _make_miner(2, "successor", 100, revision="r2"),
    ])
    # Inject pre-deployed champion via seed.
    kv.data["champion"]["deployment_id"] = "wrk-old"
    kv.data["champion"]["base_url"] = "https://t/old"
    deployer = _DeployTracker()
    samples = _SamplesFake()
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store, deployer=deployer, samples=samples,
    )

    await scheduler.tick(current_block=50)   # refresh
    await scheduler.tick(current_block=51)   # champion remains offline
    # Offline champions are retained by hotkey/revision/model identity and
    # use uid=-1 as the burn sentinel until the hotkey re-registers.
    assert deployer.teardowns == []
    champ = await state.get_champion()
    assert champ is not None
    assert champ.uid == -1
    assert champ.hotkey == "champ_hk"
    assert miner_store.rows[2]["challenge_status"] == STATUS_SAMPLING


@pytest.mark.asyncio
async def test_idempotent_state_within_tick_storm():
    """Multiple consecutive ticks during champion-samples-warmup must
    not re-deploy Targon or change state."""
    kv = _seed_state()
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
    ])
    deployer = _DeployTracker()
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store, deployer=deployer, samples=_SamplesFake(),
    )

    await scheduler.tick(current_block=50)   # refresh
    await scheduler.tick(current_block=51)   # deploy
    deploys_after_first = len(deployer.deploys)
    # Many idle ticks.
    for b in range(52, 80):
        await scheduler.tick(current_block=b)
    assert len(deployer.deploys) == deploys_after_first  # unchanged


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "transition_result",
    [
        DeploymentRoleTransitionResult.UPDATED,
        DeploymentRoleTransitionResult.STALE,
    ],
)
async def test_post_promotion_crash_recovery_does_not_self_demote(
    transition_result,
):
    """If the scheduler crashes between ``set_champion(new)`` and
    ``clear_battle()``, the saved state has ``champion.uid ==
    battle.challenger.uid`` (the new champion is also the in-flight
    challenger record). On restart, naive recovery would re-run the
    comparator with champion vs itself, tie, and demote the new
    champion.

    This test pins down that recovery clears the battle and keeps the
    promoted champion in place.
    """
    kv = _seed_state()
    # The miner that just won — same uid stored as both champion and as
    # the in-flight battle's challenger (the crash state).
    miner_store = _InMemoryMinerStore([
        _make_miner(2, "winner_hk", 200, status=STATUS_IN_PROGRESS, revision="winner_rev"),
    ])
    # Seed champion = uid 2 (the new one) with a real deployment.
    kv.data["champion"] = {
        "uid": 2, "hotkey": "winner_hk", "revision": "winner_rev",
        "model": "org/m2",
        "deployment_id": "wrk-002", "base_url": "https://t/wrk-002",
        "since_block": 50,
    }
    # Seed battle = same uid 2 (challenger record from before promotion).
    kv.data["current_battle"] = {
        "challenger": {
            "uid": 2, "hotkey": "winner_hk", "revision": "winner_rev",
            "model": "org/m2",
        },
        "deployment_id": "wrk-002",
        "base_url": "https://t/wrk-002",
        "started_at_block": 40,
    }
    # Pre-populate task_state + samples so the flow reaches _decide.
    kv.data["current_task_ids"] = {
        "task_ids": {"ENV_A": [1, 2, 3, 4, 5], "ENV_B": [6, 7, 8, 9, 10]},
        "refreshed_at_block": 0,
    }
    deployer = _DeployTracker(transition_results=[transition_result])
    samples = _SamplesFake()
    for env, tids in [("ENV_A", [1, 2, 3, 4, 5]), ("ENV_B", [6, 7, 8, 9, 10])]:
        samples.set_samples("winner_hk", "winner_rev", env, tids, score=0.5)
    weight_writer = _WeightWriterFake()
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store,
        deployer=deployer, samples=samples, weight_writer=weight_writer,
    )

    await scheduler.tick(current_block=51)

    # Champion preserved as uid 2.
    champ = await state.get_champion()
    assert champ is not None and champ.uid == 2, (
        f"recovery wrongly demoted the just-promoted champion: got {champ}"
    )
    if transition_result is DeploymentRoleTransitionResult.STALE:
        assert champ.deployment_id is None
        assert champ.base_url is None
        assert champ.deployments == []
    else:
        assert champ.deployment_id == "wrk-002"
    # Battle cleared.
    assert await state.get_battle() is None
    assert deployer.promotions == [("wrk-002", "champion")]
    # No teardown of the just-promoted champion's workload.
    assert "wrk-002" not in deployer.teardowns


@pytest.mark.asyncio
async def test_recovery_terminates_old_champion_and_writes_weights():
    """When the crash window is hit (set_champion(new) ran, clear_battle
    didn't), the recovery branch must:
      - mark the new champion's miner_stats row as ``champion``
      - mark the OLD champion's miner_stats row as ``terminated``
        (otherwise two miners stay at status='champion' indefinitely)
      - re-emit on-chain weights using the new champion (otherwise the
        chain still credits the deposed miner)

    Identifying the old champion requires ``BattleRecord.previous_champion``
    — by the time recovery fires, system_config.champion already points at
    the new winner, so the loser's UID has to live on the battle row.
    """
    kv = _seed_state()
    miner_store = _InMemoryMinerStore([
        _make_miner(
            1, "loser_hk", 100,
            status=STATUS_CHAMPION, revision="loser_rev",
        ),
        _make_miner(
            2, "winner_hk", 200,
            status=STATUS_IN_PROGRESS, revision="winner_rev",
        ),
    ])
    kv.data["champion"] = {
        "uid": 2, "hotkey": "winner_hk", "revision": "winner_rev",
        "model": "org/m2",
        "deployment_id": "wrk-002", "base_url": "https://t/wrk-002",
        "since_block": 50,
    }
    kv.data["current_battle"] = {
        "challenger": {
            "uid": 2, "hotkey": "winner_hk", "revision": "winner_rev",
            "model": "org/m2",
        },
        "deployment_id": "wrk-002",
        "base_url": "https://t/wrk-002",
        "started_at_block": 40,
        "previous_champion": {
            "uid": 1, "hotkey": "loser_hk", "revision": "loser_rev",
            "model": "org/m1",
        },
    }
    kv.data["current_task_ids"] = {
        "task_ids": {"ENV_A": [1, 2, 3, 4, 5], "ENV_B": [6, 7, 8, 9, 10]},
        "refreshed_at_block": 0,
    }
    deployer = _DeployTracker()
    samples = _SamplesFake()
    for env, tids in [("ENV_A", [1, 2, 3, 4, 5]), ("ENV_B", [6, 7, 8, 9, 10])]:
        samples.set_samples("winner_hk", "winner_rev", env, tids, score=0.5)
    weight_writer = _WeightWriterFake()
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store,
        deployer=deployer, samples=samples, weight_writer=weight_writer,
    )

    await scheduler.tick(current_block=51)

    # New champion preserved.
    champ = await state.get_champion()
    assert champ is not None and champ.uid == 2

    # New champion marked champion.
    assert miner_store.rows[2]["challenge_status"] == STATUS_CHAMPION
    # Old champion (uid 1) flipped to terminated with a recovery reason.
    assert miner_store.rows[1]["challenge_status"] == STATUS_TERMINATED, (
        "recovery must terminate the old champion's miner_stats row"
    )
    assert "recovery" in miner_store.rows[1]["termination_reason"]
    # Weights re-emitted for the new champion (otherwise the on-chain
    # weight tx still credits the deposed miner).
    assert weight_writer.calls, "recovery must re-emit weights"
    # Battle cleared.
    assert await state.get_battle() is None


@pytest.mark.asyncio
async def test_post_promotion_recovery_failed_gate_restores_previous_champion():
    kv = _seed_state(sampling_count=1)
    kv.data["behavior_gate"] = {
        "enabled": True,
        "mode": "enforce",
        "policy_version": "test-v1",
        "gated_environments": ["*"],
    }
    kv.data["champion"] = {
        "uid": 2,
        "hotkey": "winner_hk",
        "revision": "winner_rev",
        "model": "org/m2",
        "deployment_id": "wrk-002",
        "base_url": "https://t/wrk-002",
        "since_block": 50,
    }
    kv.data["current_battle"] = {
        "challenger": {
            "uid": 2,
            "hotkey": "winner_hk",
            "revision": "winner_rev",
            "model": "org/m2",
        },
        "deployment_id": "wrk-002",
        "base_url": "https://t/wrk-002",
        "started_at_block": 40,
        "previous_champion": {
            "uid": 1,
            "hotkey": "loser_hk",
            "revision": "loser_rev",
            "model": "org/m1",
        },
    }
    kv.data["current_task_ids"] = {
        "task_ids": {"ENV_A": [1], "ENV_B": [2]},
        "refreshed_at_block": 10,
    }
    miners = _InMemoryMinerStore([
        _make_miner(
            1,
            "loser_hk",
            100,
            status=STATUS_TERMINATED,
            revision="loser_rev",
        ),
        _make_miner(
            2,
            "winner_hk",
            200,
            status=STATUS_CHAMPION,
            revision="winner_rev",
        ),
    ])
    deployer = _DeployTracker()
    weights = _WeightWriterFake()
    gate = _BehaviorGateFake("failed", seal_results=[False])
    scheduler, state, _ = _build_scheduler(
        kv=kv,
        miner_store=miners,
        deployer=deployer,
        samples=_SamplesFake(),
        weight_writer=weights,
        behavior_gate_dao=gate,
    )

    await scheduler.tick(current_block=51)

    restored = await state.get_champion()
    assert restored is not None and restored.uid == 1
    assert restored.deployment_id is None
    assert restored.base_url is None
    assert await state.get_battle() is None
    assert miners.rows[1]["challenge_status"] == STATUS_CHAMPION
    assert miners.rows[2]["challenge_status"] == STATUS_TERMINATED
    assert miners.rows[2]["termination_reason"].startswith("model_behavior:")
    assert deployer.teardowns == ["wrk-002"]
    assert gate.seal_calls
    assert gate.calls
    assert weights.calls


@pytest.mark.asyncio
async def test_post_promotion_recovery_failed_gate_without_previous_clears_champion():
    kv = _seed_state(sampling_count=1)
    kv.data["behavior_gate"] = {
        "enabled": True,
        "mode": "enforce",
        "policy_version": "test-v1",
        "gated_environments": ["*"],
    }
    kv.data["champion"] = {
        "uid": 2,
        "hotkey": "winner_hk",
        "revision": "winner_rev",
        "model": "org/m2",
        "deployment_id": "wrk-002",
        "base_url": "https://t/wrk-002",
        "since_block": 50,
    }
    kv.data["current_battle"] = {
        "challenger": {
            "uid": 2,
            "hotkey": "winner_hk",
            "revision": "winner_rev",
            "model": "org/m2",
        },
        "deployment_id": "wrk-002",
        "base_url": "https://t/wrk-002",
        "started_at_block": 40,
    }
    kv.data["current_task_ids"] = {
        "task_ids": {"ENV_A": [1], "ENV_B": [2]},
        "refreshed_at_block": 10,
    }
    miners = _InMemoryMinerStore([
        _make_miner(
            2,
            "winner_hk",
            200,
            status=STATUS_CHAMPION,
            revision="winner_rev",
        ),
    ])
    deployer = _DeployTracker()
    gate = _BehaviorGateFake("failed", seal_results=[False])
    scheduler, state, _ = _build_scheduler(
        kv=kv,
        miner_store=miners,
        deployer=deployer,
        samples=_SamplesFake(),
        behavior_gate_dao=gate,
    )

    await scheduler.tick(current_block=51)

    assert await state.get_champion() is None
    assert await state.get_battle() is None
    assert miners.rows[2]["challenge_status"] == STATUS_TERMINATED
    assert deployer.teardowns == ["wrk-002"]


@pytest.mark.asyncio
async def test_recovery_without_previous_champion_still_safe():
    """Backward-compat: battle records written before the
    ``previous_champion`` field existed shouldn't crash recovery. We
    can't terminate an unknown loser or re-emit weights without
    knowing the previous champion, so the path logs and finalizes only
    the bookkeeping it does have."""
    kv = _seed_state()
    miner_store = _InMemoryMinerStore([
        _make_miner(
            2, "winner_hk", 200,
            status=STATUS_IN_PROGRESS, revision="winner_rev",
        ),
    ])
    kv.data["champion"] = {
        "uid": 2, "hotkey": "winner_hk", "revision": "winner_rev",
        "model": "org/m2",
        "deployment_id": "wrk-002", "base_url": "https://t/wrk-002",
        "since_block": 50,
    }
    # No previous_champion key — simulates an in-flight battle from a
    # scheduler version that hadn't shipped the field yet.
    kv.data["current_battle"] = {
        "challenger": {
            "uid": 2, "hotkey": "winner_hk", "revision": "winner_rev",
            "model": "org/m2",
        },
        "deployment_id": "wrk-002",
        "base_url": "https://t/wrk-002",
        "started_at_block": 40,
    }
    kv.data["current_task_ids"] = {
        "task_ids": {"ENV_A": [1, 2, 3, 4, 5], "ENV_B": [6, 7, 8, 9, 10]},
        "refreshed_at_block": 0,
    }
    samples = _SamplesFake()
    for env, tids in [("ENV_A", [1, 2, 3, 4, 5]), ("ENV_B", [6, 7, 8, 9, 10])]:
        samples.set_samples("winner_hk", "winner_rev", env, tids, score=0.5)
    weight_writer = _WeightWriterFake()
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store,
        deployer=_DeployTracker(), samples=samples,
        weight_writer=weight_writer,
    )

    await scheduler.tick(current_block=51)

    # Doesn't crash; new champion preserved; battle cleared.
    champ = await state.get_champion()
    assert champ is not None and champ.uid == 2
    assert await state.get_battle() is None
    assert miner_store.rows[2]["challenge_status"] == STATUS_CHAMPION


@pytest.mark.asyncio
async def test_start_battle_captures_previous_champion():
    """``_start_battle`` must seed ``BattleRecord.previous_champion`` so
    a later crash + recovery can still terminate the loser. This pins
    the seed write down so the field isn't accidentally dropped."""
    kv = _seed_state()
    miner_store = _InMemoryMinerStore([
        _make_miner(
            1, "champ_hk", 100,
            status=STATUS_CHAMPION, revision="champ_rev",
        ),
        _make_miner(2, "chal_hk", 200, revision="chal_rev"),
    ])
    deployer = _DeployTracker()
    samples = _SamplesFake()
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store, deployer=deployer, samples=samples,
    )

    # Tick 0: refresh task_ids.
    await scheduler.tick(current_block=50)
    # Tick 1: deploy champion.
    await scheduler.tick(current_block=51)
    task_state = await state.get_task_state()
    rb = task_state.refreshed_at_block
    # Fill champion samples so step 7 (start_battle) actually fires.
    for env in ("ENV_A", "ENV_B"):
        samples.set_samples(
            "champ_hk", "champ_rev", env,
            task_state.task_ids[env], score=0.9, refresh_block=rb,
        )
    # Tick 2: start battle. Should write a battle record with
    # previous_champion populated from the current champion.
    await scheduler.tick(current_block=52)

    battle = await state.get_battle()
    assert battle is not None, "battle should have started"
    assert battle.previous_champion is not None, (
        "_start_battle must capture previous_champion before set_champion(new)"
    )
    assert battle.previous_champion.uid == 1
    assert battle.previous_champion.hotkey == "champ_hk"


@pytest.mark.asyncio
async def test_cold_start_set_champion_writes_before_mark_terminated():
    """The order of cold-start writes matters for crash recovery. If
    ``mark_terminated(WON)`` ran first and crashed before ``set_champion``,
    recovery's cold_start would skip the half-promoted miner (their
    status is now ``champion``, not ``pending``) and promote a different
    one — leaving the original miner stranded with no system_config
    record. Pinning the safer order: set_champion first."""
    kv = _seed_state(with_champion=False)
    miner_store = _InMemoryMinerStore([
        _make_miner(5, "early_hk", 50, revision="r5"),
        _make_miner(7, "later_hk", 200, revision="r7"),
    ])

    # Capture write-ordering by recording which DB key is touched first.
    write_log: List[str] = []
    orig_set_param = kv.set
    orig_set_terminal = miner_store.set_terminal

    async def logged_set(key, value):
        if key == "champion":
            write_log.append("set_champion")
        await orig_set_param(key, value)

    async def logged_set_terminal(
        uid,
        status,
        *,
        reason="",
        hotkey=None,
        revision=None,
        model="",
        scores_by_env=None,
        opponent_scores_by_env=None,
        battle_task_ids=None,
        scores_refresh_block=None,
        terminated_at_block=None,
    ):
        if status == STATUS_CHAMPION:
            write_log.append(f"mark_won_uid={uid}")
        await orig_set_terminal(
            uid, status, reason=reason,
            hotkey=hotkey, revision=revision, model=model,
            scores_by_env=scores_by_env,
            opponent_scores_by_env=opponent_scores_by_env,
            battle_task_ids=battle_task_ids,
            scores_refresh_block=scores_refresh_block,
            terminated_at_block=terminated_at_block,
        )

    kv.set = logged_set
    miner_store.set_terminal = logged_set_terminal

    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store,
        deployer=_DeployTracker(), samples=_SamplesFake(),
    )

    await scheduler.tick(current_block=50)   # refresh task_ids
    await scheduler.tick(current_block=51)   # cold_start

    # The two cold-start writes happened in the safer order.
    cold_writes = [w for w in write_log if w == "set_champion" or w.startswith("mark_won_uid=5")]
    assert cold_writes == ["set_champion", "mark_won_uid=5"], (
        f"cold_start wrote in wrong order: {cold_writes}"
    )
    champ = await state.get_champion()
    assert champ is not None and champ.uid == 5


@pytest.mark.asyncio
async def test_challenger_invalidated_mid_battle_does_not_get_promoted():
    """If monitor invalidates the in-flight challenger (multi-commit /
    blacklist / repo-name mismatch) between ``pick_next`` and ``_decide``,
    the scheduler must NOT promote them to champion. Otherwise the
    blacklisted hotkey gets weight 1.0 on chain until they lose a future
    battle."""
    kv = _seed_state()
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
        _make_miner(2, "chal_hk", 200, revision="chal_rev"),
    ])
    deployer = _DeployTracker()
    samples = _SamplesFake()
    weight_writer = _WeightWriterFake()
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store,
        deployer=deployer, samples=samples, weight_writer=weight_writer,
    )

    await scheduler.tick(current_block=50)   # refresh
    await scheduler.tick(current_block=51)   # deploy champ
    task_state = await state.get_task_state()
    for env in ("ENV_A", "ENV_B"):
        samples.set_samples("champ_hk", "champ_rev", env, task_state.task_ids[env], score=0.3, refresh_block=task_state.refreshed_at_block)
    await scheduler.tick(current_block=52)   # start battle
    assert (await state.get_battle()).challenger.uid == 2

    # Challenger samples come in with WINNING scores (would normally promote).
    for env in ("ENV_A", "ENV_B"):
        samples.set_samples("chal_hk", "chal_rev", env, task_state.task_ids[env], score=0.95, refresh_block=task_state.refreshed_at_block)

    # Simulate monitor invalidating uid 2 mid-battle.
    miner_store.rows[2]["is_valid"] = "false"

    await scheduler.tick(current_block=53)   # decide

    # Champion did NOT change — challenger was invalid at decide time.
    champ = await state.get_champion()
    assert champ.uid == 1, f"invalid challenger was wrongly promoted: {champ}"
    # Lifecycle converges to terminated via the in-decide guard.
    assert miner_store.rows[2]["challenge_status"] == STATUS_TERMINATED
    assert miner_store.rows[2]["termination_reason"].startswith(
        "invalidated_mid_battle:"
    )
    # No weight write happened (champion didn't change).
    assert weight_writer.calls == []
    # Challenger's Targon was torn down, champion's preserved.
    assert "wrk-002" in deployer.teardowns
    assert "wrk-001" not in deployer.teardowns


@pytest.mark.asyncio
async def test_task_pool_is_oversampled_by_10_percent():
    """Sampler is called with ``ceil(sampling_count * 1.1)`` — the
    contest can decide as soon as the (champion ∩ challenger) overlap
    reaches the BASE sampling_count, so the buffer absorbs slow-tail
    and errored task_ids."""
    kv = _seed_state(sampling_count=10)  # base 10 → pool 11
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
    ])
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store,
        deployer=_DeployTracker(), samples=_SamplesFake(),
    )
    await scheduler.tick(current_block=50)
    task_state = await state.get_task_state()
    # ceil(10 * 1.1) == 11.
    assert len(task_state.task_ids["ENV_A"]) == 11
    assert len(task_state.task_ids["ENV_B"]) == 11


@pytest.mark.asyncio
async def test_decide_runs_on_overlap_not_full_pool():
    """When champion has 11/11 sampled but challenger has only 10/11
    (missing one task at the tail), the overlap is 10 = sampling_count
    — contest can decide without waiting for the last task."""
    kv = _seed_state(sampling_count=10)
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
        _make_miner(2, "chal_hk", 200, revision="chal_rev"),
    ])
    deployer = _DeployTracker()
    samples = _SamplesFake()
    weight_writer = _WeightWriterFake()
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store,
        deployer=deployer, samples=samples, weight_writer=weight_writer,
    )

    await scheduler.tick(current_block=50)   # refresh — pool=11 per env
    await scheduler.tick(current_block=51)   # deploy champion
    task_state = await state.get_task_state()
    rb = task_state.refreshed_at_block
    # Champion samples ALL 11 in each env.
    for env in ("ENV_A", "ENV_B"):
        samples.set_samples("champ_hk", "champ_rev", env,
                            task_state.task_ids[env], score=0.5, refresh_block=rb)
    await scheduler.tick(current_block=52)   # start battle
    assert (await state.get_battle()).challenger.uid == 2

    # Challenger samples only the first 10 of 11 per env (last is slow tail).
    for env in ("ENV_A", "ENV_B"):
        samples.set_samples("chal_hk", "chal_rev", env,
                            task_state.task_ids[env][:10],
                            score=0.9, refresh_block=rb)

    await scheduler.tick(current_block=53)   # overlap == 10 → decide fires

    # Challenger won the comparator → became champion.
    champ = await state.get_champion()
    assert champ.uid == 2, f"overlap-based decide should have fired: {champ}"
    assert await state.get_battle() is None
    assert len(weight_writer.calls) == 1


@pytest.mark.asyncio
async def test_stale_refresh_samples_are_ignored():
    """Samples written under a previous refresh_block must not count
    toward the current contest. Otherwise the comparator would mix old
    + new evaluations of the same task_ids."""
    kv = _seed_state(sampling_count=10)
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
    ])
    samples = _SamplesFake()
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store,
        deployer=_DeployTracker(), samples=samples,
    )
    await scheduler.tick(current_block=50)   # refresh @ 50
    await scheduler.tick(current_block=51)   # deploy champ
    task_state = await state.get_task_state()
    pool = task_state.task_ids["ENV_A"]
    # Plant champion samples but tagged with the WRONG refresh (say 0).
    samples.set_samples("champ_hk", "champ_rev", "ENV_A", pool,
                        score=0.5, refresh_block=0)
    samples.set_samples("champ_hk", "champ_rev", "ENV_B",
                        task_state.task_ids["ENV_B"],
                        score=0.5, refresh_block=0)

    # _samples_complete should NOT see these as ready.
    await scheduler.tick(current_block=52)
    assert await state.get_battle() is None, (
        "stale-refresh samples wrongly counted as current"
    )


@pytest.mark.asyncio
async def test_single_instance_clears_champion_deployment_at_refresh():
    """SSH-style single-instance provider: after a 7200-block refresh the
    inference host may still hold a stale model from the prior battle;
    clear ``champion.deployment_id`` so step 5 re-deploys before any new
    sampling."""
    from affine.src.scheduler.flow import FlowConfig
    kv = _seed_state()
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
    ])
    samples = _SamplesFake()
    state = StateStore(kv)
    queue = ChallengerQueue(miner_store)
    sampler = WindowSampler()
    comparator = WindowComparator()
    weight_writer = _WeightWriterFake()
    deployer = _DeployTracker()

    async def list_valid_miners_fn():
        return [r for r in miner_store.rows.values()
                if str(r.get("is_valid", "")).lower() == "true"]

    scheduler = FlowScheduler(
        config=FlowConfig(
            window_blocks=WINDOW_BLOCKS,
            task_pool_refresh_blocks=WINDOW_BLOCKS,
            single_instance_provider=True,
        ),
        state=state, queue=queue, sampler=sampler,
        comparator=comparator, weight_writer=weight_writer,
        deploy_fn=deployer.deploy, teardown_fn=deployer.teardown,
        sample_count_fn=samples.count_samples_for_tasks,
        scores_reader=samples.read_scores_for_tasks,
        list_valid_miners_fn=list_valid_miners_fn,
    )

    await scheduler.tick(current_block=50)   # refresh
    await scheduler.tick(current_block=51)   # deploy champion
    champ_before = await state.get_champion()
    assert champ_before.deployment_id == "wrk-001"

    # Force a refresh by advancing past the default refresh interval.
    # Battle is None, so refresh fires.
    await scheduler.tick(current_block=50 + WINDOW_BLOCKS + 1)
    champ_after = await state.get_champion()
    assert champ_after.deployment_id is None, (
        "single-instance refresh should clear champion.deployment_id"
    )
    assert champ_after.base_url is None


@pytest.mark.asyncio
async def test_single_instance_clears_champion_on_challenger_loss():
    """SSH-style single-instance provider: after a battle the host has
    served the challenger's model. On a loss, teardown empties the host;
    we must clear champion.deployment_id so step 5 fresh-deploys
    champion before next sampling."""
    from affine.src.scheduler.flow import FlowConfig
    kv = _seed_state()
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
        _make_miner(2, "chal_hk", 200, revision="chal_rev"),
    ])
    samples = _SamplesFake()
    state = StateStore(kv)
    queue = ChallengerQueue(miner_store)
    sampler = WindowSampler()
    comparator = WindowComparator()
    weight_writer = _WeightWriterFake()
    deployer = _DeployTracker()

    async def list_valid_miners_fn():
        return [r for r in miner_store.rows.values()
                if str(r.get("is_valid", "")).lower() == "true"]

    scheduler = FlowScheduler(
        config=FlowConfig(
            window_blocks=WINDOW_BLOCKS,
            task_pool_refresh_blocks=WINDOW_BLOCKS,
            single_instance_provider=True,
        ),
        state=state, queue=queue, sampler=sampler,
        comparator=comparator, weight_writer=weight_writer,
        deploy_fn=deployer.deploy, teardown_fn=deployer.teardown,
        sample_count_fn=samples.count_samples_for_tasks,
        scores_reader=samples.read_scores_for_tasks,
        list_valid_miners_fn=list_valid_miners_fn,
    )

    await scheduler.tick(current_block=50)
    await scheduler.tick(current_block=51)
    task_state = await state.get_task_state()
    rb = task_state.refreshed_at_block
    for env in ("ENV_A", "ENV_B"):
        samples.set_samples("champ_hk", "champ_rev", env,
                            task_state.task_ids[env], score=0.9, refresh_block=rb)
    await scheduler.tick(current_block=52)  # start battle
    for env in ("ENV_A", "ENV_B"):
        samples.set_samples("chal_hk", "chal_rev", env,
                            task_state.task_ids[env], score=0.3, refresh_block=rb)
    await scheduler.tick(current_block=53)  # decide — champion wins

    champ = await state.get_champion()
    # Champion held, but on single-instance host the container is gone;
    # invalidate the deployment so step 5 re-deploys next tick.
    assert champ.uid == 1
    assert champ.deployment_id is None
    assert champ.base_url is None


@pytest.mark.asyncio
async def test_decided_battle_snapshot_outcome_persists_battle_rules():
    """``score_snapshots`` is the audit record for each decided battle.
    The data the writer receives must carry every parameter that
    shaped the decision so the snapshot is self-describing:

    - global rule constants (margin / tolerance, partial-Pareto
      threshold) — passed via the ``rules`` kwarg, end up flat at the
      top of ``snapshot.config``
    - aggregate counts (dominant / not_worse / worse) — in ``outcome``
    - per-env detail (averages, sample counts, delta, margin, tolerance,
      verdict, reason) — in ``outcome.per_env``
    """
    from affine.src.scheduler.flow import (
        DEFAULT_MARGIN, DEFAULT_NOT_WORSE_TOLERANCE,
        WIN_MIN_DOMINANT_ENVS,
    )
    kv = _seed_state()
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
        _make_miner(2, "chal_hk", 200, revision="chal_rev"),
    ])
    samples = _SamplesFake()
    weight_writer = _WeightWriterFake()
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store,
        deployer=_DeployTracker(), samples=samples, weight_writer=weight_writer,
    )

    await scheduler.tick(current_block=50)  # refresh
    await scheduler.tick(current_block=51)  # deploy champ
    task_state = await state.get_task_state()
    rb = task_state.refreshed_at_block
    for env in ("ENV_A", "ENV_B"):
        samples.set_samples("champ_hk", "champ_rev", env,
                            task_state.task_ids[env], score=0.5, refresh_block=rb)
    await scheduler.tick(current_block=52)
    for env in ("ENV_A", "ENV_B"):
        samples.set_samples("chal_hk", "chal_rev", env,
                            task_state.task_ids[env], score=0.9, refresh_block=rb)
    await scheduler.tick(current_block=53)  # decide — challenger wins

    assert len(weight_writer.calls) == 1
    call = weight_writer.calls[0]

    # Rule constants are passed via the separate ``rules`` kwarg so the
    # writer can merge them flat at the top of ``snapshot.config``
    # (matching the pre-#449 path eg ``config.win_min_dominant_envs``).
    assert call["rules"] == {
        "win_margin": DEFAULT_MARGIN,
        "win_not_worse_tolerance": DEFAULT_NOT_WORSE_TOLERANCE,
        "win_min_dominant_envs": WIN_MIN_DOMINANT_ENVS,
    }

    outcome = call["outcome"]
    # Outcome carries the new decision detail only (no rules nested here).
    assert "rules" not in outcome
    # Aggregate counts: 2 envs both dominant for challenger (0.9 vs 0.5, delta=+0.4 > margin)
    assert outcome["dominant_count"] == 2
    assert outcome["not_worse_count"] == 2
    assert outcome["worse_count"] == 0
    assert outcome["winner"] == "challenger"
    assert outcome["champion_uid"] == 2  # new champion is the just-promoted challenger

    # Per-env detail must include ALL parameters that fed the decision
    per_env = outcome["per_env"]
    assert len(per_env) == 2
    for env_row in per_env:
        for required_field in (
            "env", "champion_avg", "challenger_avg",
            "champion_n", "challenger_n", "delta",
            "margin", "not_worse_tolerance", "verdict", "reason",
        ):
            assert required_field in env_row, f"per_env row missing {required_field!r}: {env_row}"
        # Verify values are floats/ints (not None / dataclass leaks)
        assert env_row["margin"] == DEFAULT_MARGIN
        assert env_row["not_worse_tolerance"] == DEFAULT_NOT_WORSE_TOLERANCE
        assert env_row["verdict"] == "dominant"
        assert env_row["reason"] == "challenger_better"
        assert abs(env_row["champion_avg"] - 0.5) < 1e-9
        assert abs(env_row["challenger_avg"] - 0.9) < 1e-9
        assert abs(env_row["delta"] - 0.4) < 1e-9


async def _single_endpoint_battle_ready_for_challenger_win(
    transition_results: List[DeploymentRoleTransitionResult],
):
    kv = _seed_state(sampling_count=4)
    miner_store = _InMemoryMinerStore([
        _make_miner(
            1, "champ_hk", 100,
            status=STATUS_CHAMPION, revision="champ_rev",
        ),
        _make_miner(2, "chal_hk", 200, revision="chal_rev"),
        _make_miner(3, "next_hk", 300, revision="next_rev"),
    ])
    samples = _SamplesFake()
    deployer = _SingleEndpointDeployTracker(
        champion_transition_results=list(transition_results),
    )
    scheduler, state, _ = _build_scheduler(
        kv=kv,
        miner_store=miner_store,
        deployer=deployer,
        samples=samples,
        config=FlowConfig(
            window_blocks=WINDOW_BLOCKS,
            single_instance_provider=True,
        ),
    )

    await scheduler.tick(current_block=50)
    await scheduler.tick(current_block=51)
    task_state = await state.get_task_state()
    refresh_block = task_state.refreshed_at_block
    for env in ("ENV_A", "ENV_B"):
        task_ids = task_state.task_ids[env]
        samples.set_samples(
            "champ_hk", "champ_rev", env, task_ids,
            score=0.5, refresh_block=refresh_block,
        )

    await scheduler.tick(current_block=52)
    assert deployer.endpoint_role == "challenger"

    for env in ("ENV_A", "ENV_B"):
        samples.set_samples(
            "chal_hk", "chal_rev", env, task_state.task_ids[env],
            score=0.9, refresh_block=refresh_block,
        )
    return scheduler, state, deployer, miner_store


@pytest.mark.asyncio
async def test_single_endpoint_win_relabels_runtime_before_next_battle():
    """A transient provider error retains the battle for a clean retry."""
    scheduler, state, deployer, _ = (
        await _single_endpoint_battle_ready_for_challenger_win([
            DeploymentRoleTransitionResult.RETRYABLE,
        ])
    )

    await scheduler.tick(current_block=53)

    champion = await state.get_champion()
    battle = await state.get_battle()
    assert champion.uid == 1
    assert battle is not None and battle.challenger.uid == 2
    assert deployer.endpoint_role == "challenger"

    await scheduler.tick(current_block=54)

    champion = await state.get_champion()
    assert champion.uid == 2
    assert await state.get_battle() is None
    assert deployer.endpoint_role == "champion"
    assert deployer.transitions == [
        (deployer.deployment_id, "champion"),
    ]

    await scheduler.tick(current_block=55)

    battle = await state.get_battle()
    assert battle is not None and battle.challenger.uid == 3
    assert deployer.endpoint_role == "challenger"


@pytest.mark.asyncio
async def test_stale_winner_runtime_does_not_block_next_battle():
    """A removed endpoint loses runtime reuse, not the evaluation result."""
    scheduler, state, deployer, miner_store = (
        await _single_endpoint_battle_ready_for_challenger_win([
            DeploymentRoleTransitionResult.STALE,
        ])
    )

    await scheduler.tick(current_block=53)

    champion = await state.get_champion()
    assert champion.uid == 2
    assert champion.deployment_id is None
    assert champion.base_url is None
    assert champion.deployments == []
    assert await state.get_battle() is None
    assert miner_store.rows[2]["challenge_status"] == STATUS_CHAMPION
    assert miner_store.rows[1]["challenge_status"] == STATUS_TERMINATED
    assert deployer.endpoint_role is None

    await scheduler.tick(current_block=54)

    battle = await state.get_battle()
    assert battle is not None and battle.challenger.uid == 3
    assert deployer.endpoint_role == "challenger"


@pytest.mark.asyncio
async def test_winning_challenger_must_top_up_samples_before_next_battle():
    """WIN path: when the challenger wins, it only needed ``sampling_count``
    overlap (the ``_battle_overlap_ready`` threshold) to be allowed to
    battle. But the champion threshold is the higher 95%-of-pool. The
    just-promoted champion must therefore complete supplementary
    sampling — step 6 must block step 7 until the new champion reaches
    the champion threshold."""
    kv = _seed_state(sampling_count=4)
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
        _make_miner(2, "chal_hk", 200, revision="chal_rev"),
        # uid=3 is queued so we can check whether step 7 fires prematurely.
        _make_miner(3, "next_hk", 300, revision="next_rev"),
    ])
    samples = _SamplesFake()
    deployer = _DeployTracker()
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store, deployer=deployer, samples=samples,
    )

    await scheduler.tick(current_block=50)  # refresh: pool of 5 per env
    await scheduler.tick(current_block=51)  # deploy champion
    task_state = await state.get_task_state()
    rb = task_state.refreshed_at_block
    # Champion is fully sampled — 5/5 per env. (pool = ceil(4*1.1) = 5,
    # threshold = ceil(5*0.95) = 5.)
    for env in ("ENV_A", "ENV_B"):
        samples.set_samples("champ_hk", "champ_rev", env,
                            task_state.task_ids[env], score=0.5, refresh_block=rb)
    await scheduler.tick(current_block=52)  # start battle uid=2

    # Challenger samples ONLY sampling_count tasks per env (4 out of 5):
    # enough to satisfy _battle_overlap_ready but NOT samples_complete.
    for env in ("ENV_A", "ENV_B"):
        partial_tasks = task_state.task_ids[env][:4]
        samples.set_samples("chal_hk", "chal_rev", env,
                            partial_tasks, score=0.9, refresh_block=rb)
    await scheduler.tick(current_block=53)  # decide — challenger wins

    # Promotion happened.
    new_champ = await state.get_champion()
    assert new_champ.uid == 2, "challenger should be the new champion"
    assert await state.get_battle() is None

    n_deploys_before = len(deployer.deploys)
    # Next tick: new champion has 4 samples per env, but threshold is 5.
    # _samples_complete must return False → no new battle, no deploy.
    await scheduler.tick(current_block=54)

    assert await state.get_battle() is None, (
        "step 6 must block step 7 until new champion reaches 95%-pool threshold"
    )
    assert len(deployer.deploys) == n_deploys_before, (
        "no challenger should be dispatched while new champion is still ramping samples"
    )

    # Now add the 5th sample to push new champion to the threshold.
    for env in ("ENV_A", "ENV_B"):
        all_tasks = task_state.task_ids[env]
        samples.set_samples("chal_hk", "chal_rev", env,
                            all_tasks, score=0.9, refresh_block=rb)

    # And now step 6 passes → step 7 dispatches uid=3.
    await scheduler.tick(current_block=55)
    battle = await state.get_battle()
    assert battle is not None and battle.challenger.uid == 3, (
        "once new champion samples are complete, next challenger should be dispatched"
    )


@pytest.mark.asyncio
async def test_single_instance_loss_skips_champion_redeploy_when_queue_nonempty():
    """Single-instance fast-path: after a challenger loses, the champion's
    samples are still valid (loss doesn't touch sample_results). If a next
    challenger is queued, step 5 should skip the ~2-minute champion
    redeploy and dispatch the next challenger directly — step 7 would
    overwrite the deployment within seconds anyway."""
    from affine.src.scheduler.flow import FlowConfig
    kv = _seed_state()
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
        _make_miner(2, "chal_hk", 200, revision="chal_rev"),
        _make_miner(3, "next_hk", 300, revision="next_rev"),
    ])
    samples = _SamplesFake()
    state = StateStore(kv)
    queue = ChallengerQueue(miner_store)
    sampler = WindowSampler()
    comparator = WindowComparator()
    weight_writer = _WeightWriterFake()
    deployer = _DeployTracker()

    async def list_valid_miners_fn():
        return [r for r in miner_store.rows.values()
                if str(r.get("is_valid", "")).lower() == "true"]

    scheduler = FlowScheduler(
        config=FlowConfig(
            window_blocks=WINDOW_BLOCKS,
            task_pool_refresh_blocks=WINDOW_BLOCKS,
            single_instance_provider=True,
        ),
        state=state, queue=queue, sampler=sampler,
        comparator=comparator, weight_writer=weight_writer,
        deploy_fn=deployer.deploy, teardown_fn=deployer.teardown,
        sample_count_fn=samples.count_samples_for_tasks,
        scores_reader=samples.read_scores_for_tasks,
        list_valid_miners_fn=list_valid_miners_fn,
    )

    # Drive through one full battle ending in a loss for uid=2.
    await scheduler.tick(current_block=50)  # refresh
    await scheduler.tick(current_block=51)  # deploy champion
    task_state = await state.get_task_state()
    rb = task_state.refreshed_at_block
    for env in ("ENV_A", "ENV_B"):
        samples.set_samples("champ_hk", "champ_rev", env,
                            task_state.task_ids[env], score=0.9, refresh_block=rb)
    await scheduler.tick(current_block=52)  # start battle uid=2
    for env in ("ENV_A", "ENV_B"):
        samples.set_samples("chal_hk", "chal_rev", env,
                            task_state.task_ids[env], score=0.3, refresh_block=rb)
    await scheduler.tick(current_block=53)  # decide — champion holds

    champ = await state.get_champion()
    assert champ.uid == 1
    assert champ.deployment_id is None  # cleared at battle start, not restored
    assert await state.get_battle() is None
    n_deploys_before = len(deployer.deploys)

    # Next tick: fast-path should pick uid=3 directly.
    await scheduler.tick(current_block=54)

    new_deploys = deployer.deploys[n_deploys_before:]
    assert len(new_deploys) == 1, (
        f"expected single deploy (challenger), got {[d.uid for d in new_deploys]}"
    )
    assert new_deploys[0].uid == 3, "fast-path must dispatch next challenger, not redeploy champion"

    battle = await state.get_battle()
    assert battle is not None and battle.challenger.uid == 3, "battle should be in flight with uid=3"

    champ = await state.get_champion()
    assert champ.uid == 1, "canonical champion unchanged"
    assert champ.deployment_id is None, "no wasted champion redeploy"


@pytest.mark.asyncio
async def test_single_instance_loss_redeploys_champion_when_queue_empty():
    """Single-instance fast-path corner: when the queue is empty after a
    loss, fall through to redeploying the champion so b300 isn't idle."""
    from affine.src.scheduler.flow import FlowConfig
    kv = _seed_state()
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
        _make_miner(2, "chal_hk", 200, revision="chal_rev"),
        # No uid=3 — once uid=2 loses, queue is empty.
    ])
    samples = _SamplesFake()
    state = StateStore(kv)
    queue = ChallengerQueue(miner_store)
    sampler = WindowSampler()
    comparator = WindowComparator()
    weight_writer = _WeightWriterFake()
    deployer = _DeployTracker()

    async def list_valid_miners_fn():
        return [r for r in miner_store.rows.values()
                if str(r.get("is_valid", "")).lower() == "true"]

    scheduler = FlowScheduler(
        config=FlowConfig(
            window_blocks=WINDOW_BLOCKS,
            task_pool_refresh_blocks=WINDOW_BLOCKS,
            single_instance_provider=True,
        ),
        state=state, queue=queue, sampler=sampler,
        comparator=comparator, weight_writer=weight_writer,
        deploy_fn=deployer.deploy, teardown_fn=deployer.teardown,
        sample_count_fn=samples.count_samples_for_tasks,
        scores_reader=samples.read_scores_for_tasks,
        list_valid_miners_fn=list_valid_miners_fn,
    )

    await scheduler.tick(current_block=50)
    await scheduler.tick(current_block=51)
    task_state = await state.get_task_state()
    rb = task_state.refreshed_at_block
    for env in ("ENV_A", "ENV_B"):
        samples.set_samples("champ_hk", "champ_rev", env,
                            task_state.task_ids[env], score=0.9, refresh_block=rb)
    await scheduler.tick(current_block=52)
    for env in ("ENV_A", "ENV_B"):
        samples.set_samples("chal_hk", "chal_rev", env,
                            task_state.task_ids[env], score=0.3, refresh_block=rb)
    await scheduler.tick(current_block=53)  # uid=2 loses, queue now empty

    n_deploys_before = len(deployer.deploys)
    await scheduler.tick(current_block=54)

    # Fast-path tried _start_battle, found empty queue, fell through.
    new_deploys = deployer.deploys[n_deploys_before:]
    assert len(new_deploys) == 1, (
        f"expected single deploy (champion fallback), got {[d.uid for d in new_deploys]}"
    )
    assert new_deploys[0].uid == 1, "fallback must redeploy champion when queue empty"

    champ = await state.get_champion()
    assert champ.deployment_id is not None, "champion redeployed so b300 has a live model"


@pytest.mark.asyncio
async def test_multi_instance_keeps_champion_deployment_on_loss():
    """Targon-style multi-instance (default): champion's deployment is
    independent of the challenger's. Don't touch it on a challenger loss."""
    kv = _seed_state()
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
        _make_miner(2, "chal_hk", 200, revision="chal_rev"),
    ])
    deployer = _DeployTracker()
    samples = _SamplesFake()
    scheduler, state, _ = _build_scheduler(   # default config, single_instance=False
        kv=kv, miner_store=miner_store, deployer=deployer, samples=samples,
    )

    await scheduler.tick(current_block=50)
    await scheduler.tick(current_block=51)
    task_state = await state.get_task_state()
    rb = task_state.refreshed_at_block
    for env in ("ENV_A", "ENV_B"):
        samples.set_samples("champ_hk", "champ_rev", env,
                            task_state.task_ids[env], score=0.9, refresh_block=rb)
    await scheduler.tick(current_block=52)
    for env in ("ENV_A", "ENV_B"):
        samples.set_samples("chal_hk", "chal_rev", env,
                            task_state.task_ids[env], score=0.3, refresh_block=rb)
    await scheduler.tick(current_block=53)

    champ = await state.get_champion()
    # Champion's deployment_id stays — Targon keeps both alive
    # independently until tearing down only the loser's.
    assert champ.deployment_id is not None
    assert champ.base_url is not None


# ---- new champion-completion threshold (95% of pool) ----------------------


@pytest.mark.asyncio
async def test_samples_complete_uses_pool_completion_ratio_not_sampling_count():
    """Champion completion now requires ~95% of the *pool* (eg 210/221
    for sampling_count=200), not just ``≥ sampling_count``.

    The old threshold made ``_battle_overlap_ready`` mathematically
    unsatisfiable because challenger overlap with a champion missing
    10% of pool averages ``sampling_count × 200/220 ≈ 182`` and never
    reaches the base sampling_count. The new threshold tightens
    champion's allowed misses to 5% so the math closes."""
    from affine.src.scorer.sampling_thresholds import champion_completion_threshold

    kv = _seed_state(sampling_count=200)
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
    ])
    threshold = champion_completion_threshold(200)

    class _Counter:
        def __init__(self, n):
            self.n = n

        async def count_samples_for_tasks(self, *a, **kw):
            return self.n

        async def read_scores_for_tasks(self, *a, **kw):
            return {}  # unused on this code path

    # 1) sample_count = old threshold (200) — under new rule, NOT enough.
    samples = _Counter(200)
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store, deployer=_DeployTracker(),
        samples=samples,
    )
    await state.set_task_state(TaskIdState(
        task_ids={"ENV_A": list(range(221)), "ENV_B": list(range(221))},
        refreshed_at_block=0,
    ))
    miner = MinerSnapshot(uid=1, hotkey="champ_hk", revision="champ_rev",
                          model="org/champ")
    envs = await state.get_environments()
    task_state = await state.get_task_state()
    assert not await scheduler._samples_complete(miner, envs=envs, task_state=task_state), (
        "200 samples (old threshold) must NOT be enough under new 95%-of-pool rule"
    )

    # 2) sample_count = new threshold — now enough.
    scheduler._sample_count = _Counter(threshold).count_samples_for_tasks
    assert await scheduler._samples_complete(miner, envs=envs, task_state=task_state), (
        f"{threshold} samples must satisfy the new threshold"
    )


# ---- early-regression fast-terminate ---------------------------------------
#
# Step 7.5 short-circuits the battle the moment a single scoring env
# reaches ``sampling_count`` overlap AND the challenger is below the
# not_worse threshold on that env — under partial-Pareto rules the rest
# of the envs can't rescue it. The three tests below pin the trigger
# decision matrix: must fire on regression, must NOT fire when overlap
# is insufficient, must NOT fire when sufficient overlap shows
# not_worse-or-better numbers.


async def _drive_through_start_of_battle(scheduler, state, samples,
                                          *, champion_score: float = 0.9):
    """Tick the scheduler through refresh → deploy → start_battle and
    return ``task_state``. Champion ends fully sampled so step 6 passes
    and step 7 fires; the caller sets challenger samples in whatever
    shape the test wants before the next tick.
    """
    await scheduler.tick(current_block=50)
    await scheduler.tick(current_block=51)
    ts = await state.get_task_state()
    for env in ("ENV_A", "ENV_B"):
        samples.set_samples(
            "champ_hk", "champ_rev", env, ts.task_ids[env],
            score=champion_score, refresh_block=ts.refreshed_at_block,
        )
    await scheduler.tick(current_block=52)
    return ts


@pytest.mark.asyncio
async def test_early_regression_triggers_lost_before_overlap_ready():
    """One env at sampling_count overlap + regression → challenger LOST
    immediately. The slower env's partial samples still get frozen onto
    the row so the rank UI keeps decide-time context."""
    kv = _seed_state(sampling_count=4)
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
        _make_miner(2, "chal_hk", 200, revision="chal_rev"),
    ])
    samples = _SamplesFake()
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store,
        deployer=_DeployTracker(), samples=samples,
        weight_writer=_WeightWriterFake(),
    )
    task_state = await _drive_through_start_of_battle(scheduler, state, samples)
    assert (await state.get_battle()) is not None

    # ENV_A: full overlap at score 0.5 vs champion's 0.9 — definitive
    # regression (0.5 < 0.9 * 0.98 = 0.882).
    samples.set_samples(
        "chal_hk", "chal_rev", "ENV_A", task_state.task_ids["ENV_A"],
        score=0.5, refresh_block=task_state.refreshed_at_block,
    )
    # ENV_B: only one overlapping sample — far below sampling_count=4.
    # The regression on ENV_A alone is sufficient under partial-Pareto.
    samples.set_samples(
        "chal_hk", "chal_rev", "ENV_B", task_state.task_ids["ENV_B"][:1],
        score=0.5, refresh_block=task_state.refreshed_at_block,
    )

    await scheduler.tick(current_block=77)

    row = miner_store.rows[2]
    assert row["challenge_status"] == STATUS_TERMINATED
    assert row["terminated_at_block"] == 77
    assert row["scores_refresh_block"] == task_state.refreshed_at_block
    assert "early_regression_in_ENV_A" in row["termination_reason"]

    frozen = row["scores_by_env"]
    assert frozen["ENV_A"]["count"] >= 4
    assert frozen["ENV_A"]["avg"] == pytest.approx(0.5)
    assert frozen["ENV_A"]["champion_overlap_avg"] == pytest.approx(0.9)
    # ENV_B's partial sample is preserved as context, even though it
    # wasn't the trigger and wouldn't have qualified on its own.
    assert frozen["ENV_B"]["count"] == 1
    assert frozen["ENV_B"]["avg"] == pytest.approx(0.5)
    assert frozen["ENV_B"]["champion_overlap_avg"] == pytest.approx(0.9)

    # Battle cleared; champion untouched.
    assert (await state.get_battle()) is None
    assert miner_store.rows[1]["challenge_status"] == STATUS_CHAMPION
    assert "terminated_at_block" not in miner_store.rows[1]


@pytest.mark.asyncio
async def test_system_miner_skips_early_regression_short_circuit():
    """System miners are evaluation probes: collect full metrics before
    final DECIDE instead of early-LOSTing on one regressed env."""
    kv = _seed_state(sampling_count=4)
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
        _make_miner(
            2000,
            "SYSTEM-1000",
            0,
            revision="sys_rev",
            model="Qwen/Qwen3.6-35B-A3B",
        ),
    ])
    samples = _SamplesFake()
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store,
        deployer=_DeployTracker(), samples=samples,
        weight_writer=_WeightWriterFake(),
    )
    task_state = await _drive_through_start_of_battle(scheduler, state, samples)
    battle = await state.get_battle()
    assert battle is not None
    assert battle.challenger.uid == 2000

    # ENV_A would be a definitive early regression for a normal miner.
    samples.set_samples(
        "SYSTEM-1000", "sys_rev", "ENV_A", task_state.task_ids["ENV_A"],
        score=0.5, refresh_block=task_state.refreshed_at_block,
    )
    # ENV_B is still below sampling_count, so without early-stop the
    # full comparator is not ready and the battle must remain active.
    samples.set_samples(
        "SYSTEM-1000", "sys_rev", "ENV_B", task_state.task_ids["ENV_B"][:1],
        score=0.5, refresh_block=task_state.refreshed_at_block,
    )

    await scheduler.tick(current_block=77)

    assert (await state.get_battle()) is not None
    assert miner_store.rows[2000]["challenge_status"] == STATUS_IN_PROGRESS
    assert "terminated_at_block" not in miner_store.rows[2000]
    assert "termination_reason" not in miner_store.rows[2000]


@pytest.mark.asyncio
async def test_no_early_decision_when_no_env_meets_sampling_count():
    """Every scoring env has overlap < sampling_count, even though the
    visible avg is far below threshold → must NOT short-circuit; tick
    waits for more samples (battle stays in flight)."""
    kv = _seed_state(sampling_count=4)
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
        _make_miner(2, "chal_hk", 200, revision="chal_rev"),
    ])
    samples = _SamplesFake()
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store,
        deployer=_DeployTracker(), samples=samples,
        weight_writer=_WeightWriterFake(),
    )
    task_state = await _drive_through_start_of_battle(scheduler, state, samples)
    assert (await state.get_battle()) is not None

    # Two samples per env (below sampling_count=4) at score 0.0 — a deep
    # apparent regression but on too small a sample to act on.
    for env in ("ENV_A", "ENV_B"):
        samples.set_samples(
            "chal_hk", "chal_rev", env, task_state.task_ids[env][:2],
            score=0.0, refresh_block=task_state.refreshed_at_block,
        )

    await scheduler.tick(current_block=77)

    assert (await state.get_battle()) is not None
    assert miner_store.rows[2]["challenge_status"] == STATUS_IN_PROGRESS
    assert "terminated_at_block" not in miner_store.rows[2]


@pytest.mark.asyncio
async def test_no_early_decision_when_overlap_complete_but_not_worse():
    """One env at full overlap with challenger matching champion (no
    regression) → must NOT short-circuit; tick waits for the slower env
    to finish so the full comparator can run."""
    kv = _seed_state(sampling_count=4)
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
        _make_miner(2, "chal_hk", 200, revision="chal_rev"),
    ])
    samples = _SamplesFake()
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store,
        deployer=_DeployTracker(), samples=samples,
        weight_writer=_WeightWriterFake(),
    )
    task_state = await _drive_through_start_of_battle(scheduler, state, samples)
    assert (await state.get_battle()) is not None

    # ENV_A: full overlap at 0.9 (matches champion). ENV_B: insufficient.
    # 0.9 >= 0.9 * 0.98 — not_worse, so no early decide.
    samples.set_samples(
        "chal_hk", "chal_rev", "ENV_A", task_state.task_ids["ENV_A"],
        score=0.9, refresh_block=task_state.refreshed_at_block,
    )
    samples.set_samples(
        "chal_hk", "chal_rev", "ENV_B", task_state.task_ids["ENV_B"][:1],
        score=0.9, refresh_block=task_state.refreshed_at_block,
    )

    await scheduler.tick(current_block=77)

    assert (await state.get_battle()) is not None
    assert miner_store.rows[2]["challenge_status"] == STATUS_IN_PROGRESS
    assert "terminated_at_block" not in miner_store.rows[2]


@pytest.mark.asyncio
async def test_early_regression_single_instance_clears_champion_deployment():
    """Single-instance provider mirror of the LOST teardown: when an
    early-LOST tears down the challenger, the shared host is empty so
    the champion's stored deployment_id is stale. Step 7.5 must clear
    it just like ``_decide``'s LOST branch does."""
    kv = _seed_state(sampling_count=4)
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
        _make_miner(2, "chal_hk", 200, revision="chal_rev"),
    ])
    samples = _SamplesFake()
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store,
        deployer=_DeployTracker(), samples=samples,
        weight_writer=_WeightWriterFake(),
    )
    scheduler.cfg = FlowConfig(
        window_blocks=WINDOW_BLOCKS,
        single_instance_provider=True,
    )

    task_state = await _drive_through_start_of_battle(scheduler, state, samples)
    # Standalone deployment record on the champion (the production
    # ``_start_battle`` clears this on entry under single_instance, but
    # we re-seed here so the assertion target is unambiguous).
    champ = await state.get_champion()
    champ.deployment_id = "wrk-champion"
    champ.base_url = "https://t/wrk-champion"
    await state.set_champion(champ)

    # Trigger early regression on ENV_A.
    samples.set_samples(
        "chal_hk", "chal_rev", "ENV_A", task_state.task_ids["ENV_A"],
        score=0.4, refresh_block=task_state.refreshed_at_block,
    )

    await scheduler.tick(current_block=80)

    champ_after = await state.get_champion()
    assert champ_after.deployment_id is None
    assert champ_after.base_url is None


@pytest.mark.asyncio
async def test_no_trigger_when_challenger_at_exact_not_worse_threshold():
    """Challenger sitting exactly at ``champion * (1 - tolerance)`` is
    boundary: comparator treats it as NOT_WORSE (the ``- 1e-9`` slack
    in ``comparator.py``'s ENV_WORSE branch). Step 7.5 mirrors that —
    a contest right on the line is not a definitive regression."""
    kv = _seed_state(sampling_count=4)
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
        _make_miner(2, "chal_hk", 200, revision="chal_rev"),
    ])
    samples = _SamplesFake()
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store,
        deployer=_DeployTracker(), samples=samples,
        weight_writer=_WeightWriterFake(),
    )
    task_state = await _drive_through_start_of_battle(
        scheduler, state, samples, champion_score=1.0,
    )

    # threshold = 1.0 * (1 - 0.02) = 0.98. Challenger sits exactly there.
    samples.set_samples(
        "chal_hk", "chal_rev", "ENV_A", task_state.task_ids["ENV_A"],
        score=0.98, refresh_block=task_state.refreshed_at_block,
    )
    # ENV_B kept short so step 8 (full-overlap_ready) doesn't fire and
    # we're testing 7.5 in isolation.
    samples.set_samples(
        "chal_hk", "chal_rev", "ENV_B", task_state.task_ids["ENV_B"][:1],
        score=0.98, refresh_block=task_state.refreshed_at_block,
    )

    await scheduler.tick(current_block=77)

    # No early-LOST: at-threshold is NOT_WORSE, battle continues.
    assert (await state.get_battle()) is not None
    assert miner_store.rows[2]["challenge_status"] == STATUS_IN_PROGRESS


@pytest.mark.asyncio
async def test_regression_in_sampling_only_env_does_not_trigger():
    """A non-scoring env (``enabled_for_scoring=False``) is excluded
    from the comparator's verdict — a regression on it must NOT fire
    step 7.5 either, regardless of overlap depth. Mirrors how
    ``_decide`` only sees scoring envs."""
    kv = InMemoryConfigStore()
    kv.data["environments"] = {
        "ENV_SCORING": {
            "display_name": "S", "enabled_for_sampling": True,
            "enabled_for_scoring": True,
            "sampling": {"sampling_count": 4, "dataset_range": [[0, 1000]],
                         "sampling_mode": "random"},
        },
        "ENV_SAMPLING_ONLY": {
            "display_name": "SO", "enabled_for_sampling": True,
            "enabled_for_scoring": False,  # excluded from verdict
            "sampling": {"sampling_count": 4, "dataset_range": [[0, 1000]],
                         "sampling_mode": "random"},
        },
    }
    kv.data["champion"] = {
        "uid": 1, "hotkey": "champ_hk", "revision": "champ_rev",
        "model": "org/champ",
        "deployment_id": None, "base_url": None, "since_block": 0,
    }
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
        _make_miner(2, "chal_hk", 200, revision="chal_rev"),
    ])
    samples = _SamplesFake()
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store,
        deployer=_DeployTracker(), samples=samples,
        weight_writer=_WeightWriterFake(),
    )

    # Warmup; champion fills both envs (sampling-only still gets sampled).
    await scheduler.tick(current_block=50)
    await scheduler.tick(current_block=51)
    task_state = await state.get_task_state()
    for env in ("ENV_SCORING", "ENV_SAMPLING_ONLY"):
        samples.set_samples(
            "champ_hk", "champ_rev", env, task_state.task_ids[env],
            score=0.9, refresh_block=task_state.refreshed_at_block,
        )
    await scheduler.tick(current_block=52)
    assert (await state.get_battle()) is not None

    # Heavy regression in the sampling-only env at FULL overlap;
    # scoring env clean and with only one sample (insufficient).
    samples.set_samples(
        "chal_hk", "chal_rev", "ENV_SAMPLING_ONLY",
        task_state.task_ids["ENV_SAMPLING_ONLY"],
        score=0.0, refresh_block=task_state.refreshed_at_block,
    )
    samples.set_samples(
        "chal_hk", "chal_rev", "ENV_SCORING",
        task_state.task_ids["ENV_SCORING"][:1],
        score=0.9, refresh_block=task_state.refreshed_at_block,
    )

    await scheduler.tick(current_block=77)

    # No early-LOST: the regression env isn't in scoring_envs.
    assert (await state.get_battle()) is not None
    assert miner_store.rows[2]["challenge_status"] == STATUS_IN_PROGRESS


@pytest.mark.asyncio
async def test_regression_detected_in_any_scoring_env_not_just_first():
    """Step 7.5 walks every scoring env — a regression in ENV_B (second
    in config order) must trigger even when ENV_A is healthy. Pins the
    "any single env regression is sufficient" rule."""
    kv = _seed_state(sampling_count=4)
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
        _make_miner(2, "chal_hk", 200, revision="chal_rev"),
    ])
    samples = _SamplesFake()
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store,
        deployer=_DeployTracker(), samples=samples,
        weight_writer=_WeightWriterFake(),
    )
    task_state = await _drive_through_start_of_battle(
        scheduler, state, samples, champion_score=0.9,
    )

    # ENV_A: full overlap, matches champion → NOT_WORSE.
    samples.set_samples(
        "chal_hk", "chal_rev", "ENV_A", task_state.task_ids["ENV_A"],
        score=0.9, refresh_block=task_state.refreshed_at_block,
    )
    # ENV_B: full overlap, deep regression.
    samples.set_samples(
        "chal_hk", "chal_rev", "ENV_B", task_state.task_ids["ENV_B"],
        score=0.4, refresh_block=task_state.refreshed_at_block,
    )

    await scheduler.tick(current_block=77)

    row = miner_store.rows[2]
    assert row["challenge_status"] == STATUS_TERMINATED
    assert "early_regression_in_ENV_B" in row["termination_reason"]


@pytest.mark.asyncio
async def test_no_trigger_when_regressing_env_below_sampling_count():
    """A regressing env that hasn't yet reached ``sampling_count``
    overlap is not yet decisive — the comparator wouldn't act on it
    either (its sample-count gate is identical). Tick must wait."""
    kv = _seed_state(sampling_count=4)
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
        _make_miner(2, "chal_hk", 200, revision="chal_rev"),
    ])
    samples = _SamplesFake()
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store,
        deployer=_DeployTracker(), samples=samples,
        weight_writer=_WeightWriterFake(),
    )
    task_state = await _drive_through_start_of_battle(
        scheduler, state, samples, champion_score=0.9,
    )

    # ENV_A: full overlap, matches champion (clean).
    samples.set_samples(
        "chal_hk", "chal_rev", "ENV_A", task_state.task_ids["ENV_A"],
        score=0.9, refresh_block=task_state.refreshed_at_block,
    )
    # ENV_B: deep regression but only 2 samples — well below
    # sampling_count=4, so neither 7.5 nor 8 fires.
    samples.set_samples(
        "chal_hk", "chal_rev", "ENV_B", task_state.task_ids["ENV_B"][:2],
        score=0.0, refresh_block=task_state.refreshed_at_block,
    )

    await scheduler.tick(current_block=77)

    assert (await state.get_battle()) is not None
    assert miner_store.rows[2]["challenge_status"] == STATUS_IN_PROGRESS


@pytest.mark.asyncio
async def test_early_regression_reason_format_pins_hotkey_prefix():
    """Frozen reason is exactly ``lost_to_champion:<hotkey[:10]>:
    early_regression_in_<env>``. Backfill scripts and the rank UI key
    off this format; changing it silently breaks downstream tooling."""
    kv = _seed_state(sampling_count=4)
    # Override the seeded champion's hotkey to a recognisable 16-char
    # string so the [:10] prefix is unambiguous in the assertion.
    kv.data["champion"]["hotkey"] = "champ_long_hotkey"
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_long_hotkey", 100,
                    status=STATUS_CHAMPION, revision="champ_rev"),
        _make_miner(2, "chal_hk", 200, revision="chal_rev"),
    ])
    samples = _SamplesFake()
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store,
        deployer=_DeployTracker(), samples=samples,
        weight_writer=_WeightWriterFake(),
    )

    # Inline warmup (the helper hardcodes ``champ_hk``).
    await scheduler.tick(current_block=50)
    await scheduler.tick(current_block=51)
    task_state = await state.get_task_state()
    rb = task_state.refreshed_at_block
    for env in ("ENV_A", "ENV_B"):
        samples.set_samples(
            "champ_long_hotkey", "champ_rev", env, task_state.task_ids[env],
            score=0.9, refresh_block=rb,
        )
    await scheduler.tick(current_block=52)
    assert (await state.get_battle()) is not None

    samples.set_samples(
        "chal_hk", "chal_rev", "ENV_A", task_state.task_ids["ENV_A"],
        score=0.4, refresh_block=rb,
    )

    await scheduler.tick(current_block=77)

    reason = miner_store.rows[2]["termination_reason"]
    assert reason == "lost_to_champion:champ_long:early_regression_in_ENV_A"


# ---- early-invalidation guard ----------------------------------------------
#
# Step 4.5 runs the mid-battle invalidation check BEFORE the
# overlap-ready gate, so a challenger flagged invalid by the monitor
# (anticopy / multi_commit / blacklist / repo-name) is torn down on
# the next tick rather than continuing to accumulate useless samples
# until ``_battle_overlap_ready`` is satisfied many minutes later.


@pytest.mark.asyncio
async def test_early_invalidation_fires_before_overlap_ready():
    """Challenger flipped to is_valid=false mid-battle is terminated
    even when neither side has produced enough samples to reach
    sampling_count overlap. Without the step 4.5 guard the executor
    would burn GPU time until step 8 finally fires."""
    kv = _seed_state(sampling_count=4)
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
        _make_miner(2, "chal_hk", 200, revision="chal_rev"),
    ])
    deployer = _DeployTracker()
    samples = _SamplesFake()
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store, deployer=deployer, samples=samples,
        weight_writer=_WeightWriterFake(),
    )
    task_state = await _drive_through_start_of_battle(scheduler, state, samples)
    assert (await state.get_battle()) is not None
    # Both miners only have a trickle of overlap — far below sampling_count;
    # step 8 would refuse to fire.
    for env in ("ENV_A", "ENV_B"):
        samples.set_samples(
            "chal_hk", "chal_rev", env, task_state.task_ids[env][:1],
            score=0.9, refresh_block=task_state.refreshed_at_block,
        )

    # Monitor flips uid=2 invalid mid-battle.
    miner_store.rows[2]["is_valid"] = "false"

    await scheduler.tick(current_block=80)

    # Battle cleared, challenger's Targon torn down, champion untouched.
    assert (await state.get_battle()) is None
    assert "wrk-002" in deployer.teardowns
    assert "wrk-001" not in deployer.teardowns

    # Lifecycle converges to TERMINATED; no frozen scores or
    # terminated_at_block because the comparator never decided.
    row = miner_store.rows[2]
    assert row["challenge_status"] == STATUS_TERMINATED
    assert row["termination_reason"].startswith("invalidated_mid_battle:")
    assert "scores_by_env" not in row
    assert "terminated_at_block" not in row


@pytest.mark.asyncio
async def test_early_invalidation_clears_champion_deployment_in_single_instance():
    """Single-instance provider host is shared by both miners — when
    the challenger's container is torn down on invalidation, the
    champion's stored deployment becomes stale. Step 4.5 must clear it
    so step 5 re-deploys champion next tick, symmetric to the regular
    LOST branch."""
    kv = _seed_state(sampling_count=4)
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
        _make_miner(2, "chal_hk", 200, revision="chal_rev"),
    ])
    samples = _SamplesFake()
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store,
        deployer=_DeployTracker(), samples=samples,
        weight_writer=_WeightWriterFake(),
    )
    scheduler.cfg = FlowConfig(
        window_blocks=WINDOW_BLOCKS,
        single_instance_provider=True,
    )
    await _drive_through_start_of_battle(scheduler, state, samples)
    champ = await state.get_champion()
    champ.deployment_id = "wrk-champion"
    champ.base_url = "https://t/wrk-champion"
    await state.set_champion(champ)

    miner_store.rows[2]["is_valid"] = "false"

    await scheduler.tick(current_block=80)

    champ_after = await state.get_champion()
    assert champ_after.deployment_id is None
    assert champ_after.base_url is None


@pytest.mark.asyncio
async def test_in_decide_invalidation_guard_catches_concurrent_flip():
    """Defense-in-depth: monitor can flip ``is_valid`` to false AFTER
    step 4.5 has read valid_miners but BEFORE ``_decide`` runs the
    comparator. Within a single tick there are many awaits between
    those two points; the in-decide invalidation guard catches the
    flip and refuses to promote.

    Wires a ``_list_valid_miners`` mock that returns the challenger
    valid on the first two calls (step 3 + step 4.5) and invalid
    thereafter (step-9 in-decide check), simulating the monitor
    writing concurrently during the tick.
    """
    kv = _seed_state(sampling_count=4)
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
        _make_miner(2, "chal_hk", 200, revision="chal_rev"),
    ])
    deployer = _DeployTracker()
    samples = _SamplesFake()
    weight_writer = _WeightWriterFake()
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store, deployer=deployer, samples=samples,
        weight_writer=weight_writer,
    )

    # Bring up the battle with full overlap and challenger==champion
    # avg, so step 7.5 (regression) doesn't fire and step 8
    # (overlap_ready) passes — execution will reach the in-decide
    # invalidation guard inside _decide.
    task_state = await _drive_through_start_of_battle(scheduler, state, samples)
    for env in ("ENV_A", "ENV_B"):
        samples.set_samples(
            "chal_hk", "chal_rev", env, task_state.task_ids[env],
            score=0.9, refresh_block=task_state.refreshed_at_block,
        )

    # Patch _list_valid_miners to flip ``is_valid`` on uid=2 after the
    # step-4.5 read sees the challenger valid; the in-decide guard sees
    # them invalid.
    call_count = [0]
    orig_fn = scheduler._list_valid_miners

    async def flipping_fn():
        call_count[0] += 1
        if call_count[0] >= 2:
            miner_store.rows[2]["is_valid"] = "false"
        return await orig_fn()

    scheduler._list_valid_miners = flipping_fn

    await scheduler.tick(current_block=77)

    # In-decide guard caught the flip: no promotion, no weight write,
    # battle cleared, challenger torn down.
    assert (await state.get_battle()) is None
    champ = await state.get_champion()
    assert champ.uid == 1, "in-decide guard failed to block promotion"
    assert weight_writer.calls == []
    assert "wrk-002" in deployer.teardowns
    # Same lifecycle-converge behavior as the step-4.5 path.
    assert miner_store.rows[2]["challenge_status"] == STATUS_TERMINATED
    assert miner_store.rows[2]["termination_reason"].startswith(
        "invalidated_mid_battle:"
    )


@pytest.mark.asyncio
async def test_no_early_invalidation_when_challenger_still_valid():
    """Sanity check: a healthy challenger doesn't get torn down. The
    guard must only fire on a real invalid signal, not on every tick."""
    kv = _seed_state(sampling_count=4)
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
        _make_miner(2, "chal_hk", 200, revision="chal_rev"),
    ])
    samples = _SamplesFake()
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store,
        deployer=_DeployTracker(), samples=samples,
        weight_writer=_WeightWriterFake(),
    )
    await _drive_through_start_of_battle(scheduler, state, samples)
    assert (await state.get_battle()) is not None
    assert miner_store.rows[2]["is_valid"] == "true"

    await scheduler.tick(current_block=80)

    assert (await state.get_battle()) is not None
    assert miner_store.rows[2]["challenge_status"] == STATUS_IN_PROGRESS


# ---- pre-deployed challengers (multi-host pre-sampling) -------------------


@pytest.mark.asyncio
async def test_predeploy_fills_available_capacity_in_fifo_order():
    """With pre-sample capacity, queued miners are pre-deployed in
    ``(first_block, uid)`` order. Champion is excluded; the active
    battle's challenger (if any) is also excluded."""
    kv = _seed_state()
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
        _make_miner(5, "early", 50, revision="r5"),    # earliest
        _make_miner(7, "mid", 200, revision="r7"),
        _make_miner(9, "late", 300, revision="r9"),
    ])
    deployer = _DeployTracker(pre_challenger_slots=2)
    samples = _SamplesFake()
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store, deployer=deployer, samples=samples,
    )

    await scheduler.tick(current_block=50)   # refresh task_ids
    await scheduler.tick(current_block=51)   # champion deploy + predeploy

    pre = await state.get_predeployed_challengers()
    assert [p.challenger.uid for p in pre] == [5, 7]
    # Champion was deployed once (1 entry); pre_challenger entries are
    # tracked separately on the deployer.
    assert [t.uid for t in deployer.predeploys] == [5, 7]


@pytest.mark.asyncio
async def test_single_model_provider_deploys_champion_before_prechallenger():
    """Dynamic placement must not let predeployment consume the only free
    endpoint before the champion has a serving runtime."""
    kv = _seed_state()
    miner_store = _InMemoryMinerStore([
        _make_miner(
            1, "champ_hk", 100,
            status=STATUS_CHAMPION, revision="champ_rev",
        ),
        _make_miner(2, "chal_hk", 200, revision="chal_rev"),
    ])
    deployer = _DeployTracker(pre_challenger_slots=1)
    scheduler, state, _ = _build_scheduler(
        kv=kv,
        miner_store=miner_store,
        deployer=deployer,
        samples=_SamplesFake(),
        config=FlowConfig(
            window_blocks=WINDOW_BLOCKS,
            single_instance_provider=True,
        ),
    )

    await scheduler.tick(current_block=50)

    assert deployer.predeploys == []
    assert (await state.get_predeployed_challengers()) == []

    await scheduler.tick(current_block=51)

    assert [target.uid for target in deployer.deploys] == [1]
    assert [target.uid for target in deployer.predeploys] == [2]


@pytest.mark.asyncio
async def test_predeploy_stops_cleanly_when_no_capacity():
    """Default single-machine config (``pre_challenger_slots=0``)
    surfaces ``RuntimeError`` on the first attempt; the scheduler must
    bail without leaving partial state."""
    kv = _seed_state()
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
        _make_miner(2, "chal", 200, revision="r2"),
    ])
    deployer = _DeployTracker()  # default: 0 pre-sample slots
    samples = _SamplesFake()
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store, deployer=deployer, samples=samples,
    )

    await scheduler.tick(current_block=50)
    await scheduler.tick(current_block=51)

    assert (await state.get_predeployed_challengers()) == []
    assert deployer.predeploys == []


@pytest.mark.asyncio
async def test_predeploy_invalidation_sweep_tears_down_without_status_write():
    """When a pre-deployed miner becomes invalid, the sweep tears down
    its deployment but leaves ``challenge_status`` untouched —
    ``invalid_reason`` on the miners row is the authoritative cause,
    same policy as :meth:`_decide_invalidation_lost` for the active
    battle."""
    kv = _seed_state()
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
        _make_miner(5, "early", 50, revision="r5"),
    ])
    deployer = _DeployTracker(pre_challenger_slots=4)
    samples = _SamplesFake()
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store, deployer=deployer, samples=samples,
    )

    await scheduler.tick(current_block=50)
    await scheduler.tick(current_block=51)
    assert len(await state.get_predeployed_challengers()) == 1

    # Monitor flips uid=5 to invalid.
    miner_store.rows[5]["is_valid"] = "false"
    pre_dep_id = (await state.get_predeployed_challengers())[0].deployment_id

    await scheduler.tick(current_block=52)

    assert (await state.get_predeployed_challengers()) == []
    assert pre_dep_id in deployer.teardowns
    # No status mutation — still in SAMPLING (or unchanged).
    assert miner_store.rows[5]["challenge_status"] == STATUS_SAMPLING


@pytest.mark.asyncio
async def test_predeploy_invalidation_sweep_drops_record_matching_champion():
    """Crash mid-``_adopt_predeployed_if_present`` can leave a stale
    record whose uid is the newly-installed champion. Sweep must tear
    it down so the endpoint can be filled again."""
    from affine.src.scorer.window_state import (
        BattleRecord, ChampionRecord, DeploymentRecord, MinerSnapshot,
    )

    kv = _seed_state()
    miner_store = _InMemoryMinerStore([
        _make_miner(7, "champ_hk", 50, status=STATUS_CHAMPION, revision="r7"),
    ])
    deployer = _DeployTracker(pre_challenger_slots=4)
    samples = _SamplesFake()
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store, deployer=deployer, samples=samples,
    )

    # State: uid=7 is current champion AND lingers in predeployed
    # records (the crash-recovery shape this fix targets).
    await state.set_champion(ChampionRecord(
        uid=7, hotkey="champ_hk", revision="r7", model="org/m7",
        deployment_id="wrk-champ", base_url="https://t/wrk-champ",
        since_block=0,
    ))
    stale_dep_id = "wrk-stale-pre"
    await state.set_predeployed_challengers([
        BattleRecord(
            challenger=MinerSnapshot(
                uid=7, hotkey="champ_hk", revision="r7", model="org/m7",
            ),
            deployment_id=stale_dep_id,
            base_url="https://t/wrk-stale-pre",
            started_at_block=0,
            deployments=[DeploymentRecord(
                endpoint_name="b300_2",
                deployment_id=stale_dep_id,
                base_url="https://t/wrk-stale-pre",
            )],
        ),
    ])
    await state.set_task_state(TaskIdState(
        task_ids={"ENV_A": [1, 2, 3], "ENV_B": [4, 5, 6]},
        refreshed_at_block=0,
    ))

    await scheduler.tick(current_block=10)

    assert (await state.get_predeployed_challengers()) == []
    assert stale_dep_id in deployer.teardowns


@pytest.mark.asyncio
async def test_predeploy_sweep_keeps_promoted_champion_runtime_alive():
    """A crash can leave the in-place promoted deployment in both champion
    and predeployed state after current_battle has been cleared."""
    from affine.src.scorer.window_state import (
        ChampionRecord, DeploymentRecord,
    )

    kv = _seed_state()
    miner_store = _InMemoryMinerStore([
        _make_miner(
            7, "winner_hk", 50,
            status=STATUS_CHAMPION, revision="r7",
        ),
    ])
    deployer = _DeployTracker()
    scheduler, state, _ = _build_scheduler(
        kv=kv,
        miner_store=miner_store,
        deployer=deployer,
        samples=_SamplesFake(),
    )
    deployment = DeploymentRecord(
        endpoint_name="endpoint-2",
        deployment_id="wrk-promoted",
        base_url="https://t/wrk-promoted",
    )
    await state.set_champion(ChampionRecord(
        uid=7,
        hotkey="winner_hk",
        revision="r7",
        model="org/m7",
        deployment_id=deployment.deployment_id,
        base_url=deployment.base_url,
        deployments=[deployment],
        since_block=50,
    ))
    await state.set_predeployed_challengers([
        BattleRecord(
            challenger=MinerSnapshot(
                uid=7,
                hotkey="winner_hk",
                revision="r7",
                model="org/m7",
            ),
            deployment_id=deployment.deployment_id,
            base_url=deployment.base_url,
            started_at_block=40,
            deployments=[deployment],
        ),
    ])

    await scheduler._predeploy_invalidation_sweep()

    assert (await state.get_predeployed_challengers()) == []
    assert deployer.teardowns == []
    assert (await state.get_champion()).deployment_id == "wrk-promoted"


@pytest.mark.asyncio
async def test_predeploy_early_loss_terminates_sure_loser():
    """Pre-deployed miner with a definitive env regression past the
    not_worse threshold is marked LOST + torn down before ever entering
    the formal battle — the user's 'fail fast' semantic."""
    kv = _seed_state(sampling_count=2)
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
        _make_miner(9, "loser", 80, revision="r9"),
    ])
    deployer = _DeployTracker(pre_challenger_slots=4)
    samples = _SamplesFake()
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store, deployer=deployer, samples=samples,
    )

    # Tick 1+2: refresh + champion deploy + predeploy uid=9
    await scheduler.tick(current_block=50)
    await scheduler.tick(current_block=51)

    task_state = await state.get_task_state()
    refresh = task_state.refreshed_at_block

    # Champion scored well, pre-deployed scored definitively below the
    # not_worse threshold on every overlap.
    for env in ("ENV_A", "ENV_B"):
        ids = task_state.task_ids[env][:2]   # sampling_count=2
        samples.set_samples(
            "champ_hk", "champ_rev", env, ids, score=0.9,
            refresh_block=refresh,
        )
        samples.set_samples(
            "r9_hk_unused", "r9", env, ids, score=0.1,
            refresh_block=refresh,
        )
        # The pre-deployed miner's hotkey in _make_miner is "loser",
        # set those rows too.
        samples.set_samples(
            "loser", "r9", env, ids, score=0.1,
            refresh_block=refresh,
        )

    await scheduler.tick(current_block=52)

    pre = await state.get_predeployed_challengers()
    assert pre == []
    assert miner_store.rows[9]["challenge_status"] == STATUS_TERMINATED
    reason = miner_store.rows[9].get("termination_reason") or ""
    assert "early_regression_in_" in reason
    assert reason.endswith(":predeploy")


@pytest.mark.asyncio
async def test_start_battle_adopts_predeployed_record():
    """When ``pick_next`` selects a miner that was pre-deployed, the
    running deployment is promoted in place. The old champion's distinct
    endpoint is released only after current_battle is durable."""
    kv = _seed_state()
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
        _make_miner(2, "chal_hk", 200, revision="chal_rev"),
    ])
    deployer = _DeployTracker(pre_challenger_slots=4)
    samples = _SamplesFake()
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store, deployer=deployer, samples=samples,
        config=FlowConfig(
            window_blocks=WINDOW_BLOCKS,
            single_instance_provider=True,
        ),
    )

    # Refresh, then champion deploy + predeploy uid=2 on a free endpoint.
    await scheduler.tick(current_block=50)
    await scheduler.tick(current_block=51)
    pre = await state.get_predeployed_challengers()
    assert [p.challenger.uid for p in pre] == [2]
    pre_dep_id = pre[0].deployment_id
    champion_dep_id = (await state.get_champion()).deployment_id
    active_deploys_before = len(deployer.deploys)
    teardowns_before = len(deployer.teardowns)

    # Complete champion baseline so step 7 fires _start_battle.
    task_state = await state.get_task_state()
    for env in ("ENV_A", "ENV_B"):
        samples.set_samples(
            "champ_hk", "champ_rev", env, task_state.task_ids[env],
            refresh_block=task_state.refreshed_at_block,
        )

    await scheduler.tick(current_block=52)

    battle = await state.get_battle()
    assert battle is not None and battle.challenger.uid == 2
    assert battle.deployment_id == pre_dep_id
    assert deployer.promotions == [(pre_dep_id, "challenger")]
    assert len(deployer.deploys) == active_deploys_before
    assert pre_dep_id not in deployer.teardowns
    assert champion_dep_id in deployer.teardowns
    assert len(deployer.teardowns) > teardowns_before
    assert (await state.get_predeployed_challengers()) == []
    champion = await state.get_champion()
    assert champion.deployment_id is None
    assert champion.base_url is None
    assert champion.deployments == []


@pytest.mark.asyncio
async def test_stale_predeployment_promotion_drops_state_without_teardown():
    from affine.src.scorer.window_state import ChampionRecord

    kv = _seed_state()
    miner_store = _InMemoryMinerStore([
        _make_miner(
            1, "champ_hk", 100,
            status=STATUS_CHAMPION, revision="champ_rev",
        ),
        _make_miner(2, "chal_hk", 200, revision="chal_rev"),
    ])
    deployer = _DeployTracker()
    samples = _SamplesFake()

    async def reject_stale_assignment(_record, _role):
        return DeploymentRoleTransitionResult.STALE

    scheduler, state, _ = _build_scheduler(
        kv=kv,
        miner_store=miner_store,
        deployer=deployer,
        samples=samples,
        transition_deployment_role_fn=reject_stale_assignment,
    )
    champion = ChampionRecord(
        uid=1,
        hotkey="champ_hk",
        revision="champ_rev",
        model="org/m1",
        deployment_id="wrk-champion",
        base_url="https://t/wrk-champion",
    )
    await state.set_champion(champion)
    stale = BattleRecord(
        challenger=MinerSnapshot(
            uid=2,
            hotkey="chal_hk",
            revision="chal_rev",
            model="org/m2",
        ),
        deployment_id="wrk-stale",
        base_url="https://t/wrk-stale",
        started_at_block=40,
    )
    await state.set_predeployed_challengers([stale])

    await scheduler._start_battle(champion, current_block=50)

    assert await state.get_battle() is None
    assert (await state.get_predeployed_challengers()) == []
    assert miner_store.rows[2]["challenge_status"] == STATUS_SAMPLING
    assert deployer.teardowns == []


@pytest.mark.asyncio
async def test_battle_retries_champion_runtime_release_after_teardown_error():
    kv = _seed_state()
    miner_store = _InMemoryMinerStore([
        _make_miner(
            1, "champ_hk", 100,
            status=STATUS_CHAMPION, revision="champ_rev",
        ),
        _make_miner(2, "chal_hk", 200, revision="chal_rev"),
    ])
    deployer = _DeployTracker(pre_challenger_slots=1)
    samples = _SamplesFake()
    scheduler, state, _ = _build_scheduler(
        kv=kv,
        miner_store=miner_store,
        deployer=deployer,
        samples=samples,
        config=FlowConfig(
            window_blocks=WINDOW_BLOCKS,
            single_instance_provider=True,
        ),
    )

    await scheduler.tick(current_block=50)
    await scheduler.tick(current_block=51)
    champion_dep_id = (await state.get_champion()).deployment_id
    deployer.teardown_failures_once.add(champion_dep_id)
    task_state = await state.get_task_state()
    for env in ("ENV_A", "ENV_B"):
        samples.set_samples(
            "champ_hk", "champ_rev", env, task_state.task_ids[env],
            refresh_block=task_state.refreshed_at_block,
        )

    await scheduler.tick(current_block=52)

    assert (await state.get_battle()) is not None
    assert (await state.get_champion()).deployment_id == champion_dep_id

    await scheduler.tick(current_block=53)

    assert (await state.get_champion()).deployment_id is None
    assert deployer.teardowns.count(champion_dep_id) == 2


@pytest.mark.asyncio
async def test_predeploy_waits_while_champion_runtime_release_is_pending():
    """A failed SSH teardown may clear DB assignment ownership before its
    retry succeeds. Persistent champion runtime state must keep that endpoint
    from being reassigned in the interim."""
    from affine.src.scorer.window_state import ChampionRecord

    kv = _seed_state()
    miner_store = _InMemoryMinerStore([
        _make_miner(
            1, "champ_hk", 100,
            status=STATUS_CHAMPION, revision="champ_rev",
        ),
        _make_miner(
            2, "active_hk", 200,
            status=STATUS_IN_PROGRESS, revision="active_rev",
        ),
        _make_miner(3, "queued_hk", 300, revision="queued_rev"),
    ])
    deployer = _DeployTracker(pre_challenger_slots=1)
    scheduler, state, _ = _build_scheduler(
        kv=kv,
        miner_store=miner_store,
        deployer=deployer,
        samples=_SamplesFake(),
        config=FlowConfig(
            window_blocks=WINDOW_BLOCKS,
            single_instance_provider=True,
        ),
    )
    champion = ChampionRecord(
        uid=1,
        hotkey="champ_hk",
        revision="champ_rev",
        model="org/m1",
        deployment_id="wrk-champ-pending-cleanup",
        base_url="https://t/wrk-champ-pending-cleanup",
    )
    await state.set_champion(champion)
    await state.set_battle(BattleRecord(
        challenger=MinerSnapshot(
            uid=2,
            hotkey="active_hk",
            revision="active_rev",
            model="org/m2",
        ),
        deployment_id="wrk-active",
        base_url="https://t/wrk-active",
        started_at_block=50,
    ))

    await scheduler._predeploy_fill_available(champion, current_block=51)

    assert deployer.predeploys == []
    assert (await state.get_predeployed_challengers()) == []


@pytest.mark.asyncio
async def test_single_endpoint_model_swap_does_not_teardown_active_challenger():
    from affine.src.scorer.window_state import ChampionRecord, DeploymentRecord

    kv = _seed_state()
    miner_store = _InMemoryMinerStore([
        _make_miner(
            1, "champ_hk", 100,
            status=STATUS_CHAMPION, revision="champ_rev",
        ),
        _make_miner(
            2, "chal_hk", 200,
            status=STATUS_IN_PROGRESS, revision="chal_rev",
        ),
    ])
    deployer = _DeployTracker()
    scheduler, state, _ = _build_scheduler(
        kv=kv,
        miner_store=miner_store,
        deployer=deployer,
        samples=_SamplesFake(),
        config=FlowConfig(
            window_blocks=WINDOW_BLOCKS,
            single_instance_provider=True,
        ),
    )
    shared = DeploymentRecord(
        endpoint_name="only-endpoint",
        deployment_id="ssh:only-endpoint:affine-sglang-current",
        base_url="https://only-endpoint/v1",
    )
    champion = ChampionRecord(
        uid=1,
        hotkey="champ_hk",
        revision="champ_rev",
        model="org/m1",
        deployment_id=shared.deployment_id,
        base_url=shared.base_url,
        deployments=[shared],
    )
    await state.set_champion(champion)
    battle = BattleRecord(
        challenger=MinerSnapshot(
            uid=2,
            hotkey="chal_hk",
            revision="chal_rev",
            model="org/m2",
        ),
        deployment_id=shared.deployment_id,
        base_url=shared.base_url,
        deployments=[shared],
        started_at_block=50,
    )

    released = await scheduler._release_champion_runtime_for_battle(
        champion, battle,
    )

    assert released is True
    assert deployer.teardowns == []
    persisted = await state.get_champion()
    assert persisted.deployment_id is None
    assert persisted.deployments == []


@pytest.mark.asyncio
async def test_predeploy_sweep_drops_active_battle_duplicate_without_teardown():
    from affine.src.scorer.window_state import DeploymentRecord

    kv = _seed_state()
    miner_store = _InMemoryMinerStore([
        _make_miner(
            1, "champ_hk", 100,
            status=STATUS_CHAMPION, revision="champ_rev",
        ),
        _make_miner(
            2, "chal_hk", 200,
            status=STATUS_IN_PROGRESS, revision="chal_rev",
        ),
    ])
    deployer = _DeployTracker()
    scheduler, state, _ = _build_scheduler(
        kv=kv,
        miner_store=miner_store,
        deployer=deployer,
        samples=_SamplesFake(),
    )
    deployment = DeploymentRecord(
        endpoint_name="endpoint-2",
        deployment_id="wrk-shared",
        base_url="https://t/wrk-shared",
    )
    record = BattleRecord(
        challenger=MinerSnapshot(
            uid=2, hotkey="chal_hk", revision="chal_rev", model="org/m2",
        ),
        deployment_id=deployment.deployment_id,
        base_url=deployment.base_url,
        started_at_block=50,
        deployments=[deployment],
    )
    await state.set_battle(record)
    await state.set_predeployed_challengers([record])

    await scheduler._predeploy_invalidation_sweep()

    assert (await state.get_predeployed_challengers()) == []
    assert deployer.teardowns == []
    assert (await state.get_battle()).deployment_id == "wrk-shared"


@pytest.mark.asyncio
async def test_predeploy_does_not_disturb_fifo_battle_order():
    """If a later-submitted miner has been pre-sampled while an
    earlier-submitted one hasn't, the next battle still picks the
    earlier one (FIFO). Pre-sampling order is independent of pick
    order — the queue's ``(first_block, uid)`` sort is authoritative."""
    kv = _seed_state()
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
        _make_miner(2, "early", 50, revision="r2"),    # earliest queued
        _make_miner(3, "late", 300, revision="r3"),    # later queued
    ])
    # Only 1 pre-sample slot; uid=2 (earliest) takes it first.
    deployer = _DeployTracker(pre_challenger_slots=1)
    samples = _SamplesFake()
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store, deployer=deployer, samples=samples,
    )

    await scheduler.tick(current_block=50)
    await scheduler.tick(current_block=51)
    pre = await state.get_predeployed_challengers()
    assert [p.challenger.uid for p in pre] == [2]

    # Complete champion baseline → _start_battle picks earliest = uid=2.
    task_state = await state.get_task_state()
    for env in ("ENV_A", "ENV_B"):
        samples.set_samples(
            "champ_hk", "champ_rev", env, task_state.task_ids[env],
            refresh_block=task_state.refreshed_at_block,
        )
    await scheduler.tick(current_block=52)
    battle = await state.get_battle()
    assert battle is not None
    assert battle.challenger.uid == 2  # FIFO preserved


@pytest.mark.asyncio
async def test_predeploy_phase_noop_without_champion():
    """Cold-start tick (no champion yet) must not pre-deploy anything —
    the gating samples don't exist yet."""
    kv = _seed_state(with_champion=False)
    miner_store = _InMemoryMinerStore([
        _make_miner(5, "early", 50, revision="r5"),
        _make_miner(7, "late", 200, revision="r7"),
    ])
    deployer = _DeployTracker(pre_challenger_slots=4)
    samples = _SamplesFake()
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store, deployer=deployer, samples=samples,
    )

    # Tick 1 refresh, tick 2 cold-start promotes uid=5.
    await scheduler.tick(current_block=50)
    await scheduler.tick(current_block=51)

    # Pre-deploy ran in tick 2 AFTER cold_start set the champion. uid=7
    # is the only remaining miner, so it lands as a pre-challenger.
    pre = await state.get_predeployed_challengers()
    assert [p.challenger.uid for p in pre] == [7]


@pytest.mark.asyncio
async def test_predeploy_excludes_active_battle_challenger():
    """The active ``current_battle.challenger`` must never appear in
    the pre-deployed list. The exclusion lives in
    :meth:`_predeploy_fill_available`."""
    kv = _seed_state()
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
        _make_miner(2, "active", 200, revision="r2"),
        _make_miner(3, "queued", 300, revision="r3"),
    ])
    deployer = _DeployTracker(pre_challenger_slots=4)
    samples = _SamplesFake()
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store, deployer=deployer, samples=samples,
    )

    # Get champion sampled, then start a battle with uid=2.
    await scheduler.tick(current_block=50)
    await scheduler.tick(current_block=51)
    task_state = await state.get_task_state()
    for env in ("ENV_A", "ENV_B"):
        samples.set_samples(
            "champ_hk", "champ_rev", env, task_state.task_ids[env],
            refresh_block=task_state.refreshed_at_block,
        )
    # First "complete" tick triggers start_battle with uid=2 (FIFO);
    # next tick's predeploy phase should pick uid=3, not uid=2.
    await scheduler.tick(current_block=52)
    battle = await state.get_battle()
    assert battle is not None and battle.challenger.uid == 2

    pre_uids = {p.challenger.uid for p in await state.get_predeployed_challengers()}
    assert 2 not in pre_uids
    assert pre_uids == {3}


@pytest.mark.asyncio
async def test_predeploy_deploy_failure_keeps_candidate_and_advances():
    """A pre-deploy failure (sglang OSError on transient HF blip,
    host issue, etc.) must NOT mark the miner FAILED — scheduler has
    no info to distinguish infra noise from a true model fault, and
    the monitor's ``hf_model_fetch`` check is the authoritative signal.
    The fill loop must continue with the next FIFO candidate so a
    single broken miner doesn't starve every pre-sample slot."""
    kv = _seed_state()
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
        _make_miner(5, "broken", 50, revision="r5"),    # earliest, will fail
        _make_miner(7, "fine", 200, revision="r7"),
        _make_miner(9, "alsofine", 300, revision="r9"),
    ])
    deployer = _DeployTracker(
        pre_challenger_slots=3,
        predeploy_failures={5},  # uid=5 fails to deploy
    )
    samples = _SamplesFake()
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store, deployer=deployer, samples=samples,
    )

    await scheduler.tick(current_block=50)
    await scheduler.tick(current_block=51)

    # uid=5 stays in queue (not marked FAILED); uid=7 and uid=9 take
    # the remaining slots.
    assert miner_store.rows[5]["challenge_status"] != STATUS_TERMINATED
    assert miner_store.rows[5].get("termination_reason") in (None, "")

    pre_uids = [p.challenger.uid for p in await state.get_predeployed_challengers()]
    assert pre_uids == [7, 9], (
        f"expected uid=5 failure not to starve later candidates, got {pre_uids}"
    )


@pytest.mark.asyncio
async def test_predeploy_no_capacity_does_not_mark_failed():
    """``NoEndpointCapacity`` is a clean termination signal — the
    candidate miner must NOT be marked FAILED (they're a valid
    candidate, just no host for them right now). This is the
    asymmetry between 'no capacity' and 'real deploy error' that
    motivated introducing ``NoEndpointCapacity`` as a distinct
    exception class."""
    kv = _seed_state()
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
        _make_miner(2, "queued", 50, revision="r2"),
    ])
    deployer = _DeployTracker()  # 0 pre-sample slots → NoEndpointCapacity
    samples = _SamplesFake()
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store, deployer=deployer, samples=samples,
    )

    await scheduler.tick(current_block=50)
    await scheduler.tick(current_block=51)

    # uid=2 was eligible but had no host. Status must stay SAMPLING.
    assert miner_store.rows[2]["challenge_status"] == STATUS_SAMPLING
    assert (await state.get_predeployed_challengers()) == []


# ---- orphan reaper ---------------------------------------------------------


@pytest.mark.asyncio
async def test_reaper_skips_old_orphan_without_endpoint_snapshot():
    """Without endpoint lifecycle evidence, the reaper must not blindly
    release or terminate an old in_progress row."""
    kv = _seed_state()
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
        _make_miner(99, "orphan_hk", 50, status=STATUS_IN_PROGRESS,
                    revision="orphan_rev"),
    ])
    # Force the orphan's claim well into the past so age >> grace.
    miner_store.rows[99]["challenge_claimed_at"] = 1
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store,
        deployer=_DeployTracker(), samples=_SamplesFake(),
    )

    await scheduler.tick(current_block=50)

    assert miner_store.rows[99]["challenge_status"] == STATUS_IN_PROGRESS
    assert miner_store.rows[99].get("termination_reason") in (None, "")


def test_orphan_release_reason_uses_endpoint_lifecycle():
    assert FlowScheduler._orphan_release_reason(
        claimed_at=100,
        endpoint_activations={"endpoint-1": 200},
    ).startswith("endpoint_reactivated_after_claim:")
    assert FlowScheduler._orphan_release_reason(
        claimed_at=100,
        endpoint_activations={},
    ) == "no_active_endpoint"
    assert FlowScheduler._orphan_release_reason(
        claimed_at=200,
        endpoint_activations={"endpoint-1": 100},
    ) == "stale_claim_endpoint_stable"
    assert FlowScheduler._orphan_release_reason(
        claimed_at=200,
        endpoint_activations=None,
    ) == "stale_claim_no_endpoint_snapshot"


@pytest.mark.asyncio
async def test_reaper_releases_orphan_when_no_endpoint_is_active():
    kv = _seed_state()
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
        _make_miner(99, "orphan_hk", 50, status=STATUS_IN_PROGRESS,
                    revision="orphan_rev"),
    ])
    miner_store.rows[99]["challenge_claimed_at"] = 100

    async def endpoint_activations():
        return {}

    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store,
        deployer=_DeployTracker(), samples=_SamplesFake(),
        list_active_endpoint_activations_fn=endpoint_activations,
    )

    await scheduler.tick(current_block=50)

    assert miner_store.rows[99]["challenge_status"] == STATUS_SAMPLING
    assert miner_store.rows[99].get("termination_reason") in (None, "")
    assert (await state.get_battle()) is None


@pytest.mark.asyncio
async def test_reaper_releases_orphan_after_endpoint_reactivation():
    kv = _seed_state()
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
        _make_miner(99, "orphan_hk", 50, status=STATUS_IN_PROGRESS,
                    revision="orphan_rev"),
    ])
    miner_store.rows[99]["challenge_claimed_at"] = 100

    async def endpoint_activations():
        return {"endpoint-1": 200}

    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store,
        deployer=_DeployTracker(), samples=_SamplesFake(),
        list_active_endpoint_activations_fn=endpoint_activations,
    )

    await scheduler.tick(current_block=50)

    assert miner_store.rows[99]["challenge_status"] == STATUS_SAMPLING
    assert (await state.get_battle()) is None


@pytest.mark.asyncio
async def test_reaper_releases_old_orphan_when_endpoint_stayed_stable():
    kv = _seed_state()
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
        _make_miner(99, "orphan_hk", 50, status=STATUS_IN_PROGRESS,
                    revision="orphan_rev"),
    ])
    miner_store.rows[99]["challenge_claimed_at"] = 200

    async def endpoint_activations():
        return {"endpoint-1": 100}

    scheduler, _, _ = _build_scheduler(
        kv=kv, miner_store=miner_store,
        deployer=_DeployTracker(), samples=_SamplesFake(),
        list_active_endpoint_activations_fn=endpoint_activations,
    )

    await scheduler.tick(current_block=50)

    assert miner_store.rows[99]["challenge_status"] == STATUS_SAMPLING
    assert miner_store.rows[99].get("termination_reason") in (None, "")


@pytest.mark.asyncio
async def test_reaper_skips_active_battle_challenger():
    """An in_progress row that IS the current battle challenger must
    never be reaped, regardless of age."""
    kv = _seed_state()
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
        _make_miner(2, "chal_hk", 200, revision="chal_rev"),
    ])
    samples = _SamplesFake()
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store,
        deployer=_DeployTracker(), samples=samples,
    )

    await scheduler.tick(current_block=50)   # refresh
    await scheduler.tick(current_block=51)   # deploy champ
    task_state = await state.get_task_state()
    for env in ("ENV_A", "ENV_B"):
        samples.set_samples(
            "champ_hk", "champ_rev", env, task_state.task_ids[env],
            score=0.3, refresh_block=task_state.refreshed_at_block,
        )
    await scheduler.tick(current_block=52)   # start battle with uid=2

    assert (await state.get_battle()).challenger.uid == 2
    # Age uid=2's claim into the past so the reaper would reap it
    # if not for the active-battle protection.
    miner_store.rows[2]["challenge_claimed_at"] = 1

    await scheduler.tick(current_block=53)

    assert miner_store.rows[2]["challenge_status"] == STATUS_IN_PROGRESS


@pytest.mark.asyncio
async def test_reaper_skips_champion_uid_for_cold_start_orphan():
    """If a cold_start crashed between set_champion and mark_terminated,
    the champion ends up with challenge_status=in_progress. The reaper
    must NOT reap them — that would terminate the running champion."""
    kv = _seed_state(with_champion=False)
    # Champion already set in system_config, but their miner_stats row
    # is still IN_PROGRESS (mid-cold-start crash scenario).
    kv.data["champion"] = {
        "uid": 5, "hotkey": "cold_champ", "revision": "cold_rev",
        "model": "org/cold",
        "deployment_id": None, "base_url": None,
        "since_block": 0,
    }
    miner_store = _InMemoryMinerStore([
        _make_miner(5, "cold_champ", 50,
                    status=STATUS_IN_PROGRESS, revision="cold_rev"),
    ])
    miner_store.rows[5]["challenge_claimed_at"] = 1
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store,
        deployer=_DeployTracker(), samples=_SamplesFake(),
    )

    await scheduler.tick(current_block=50)

    assert miner_store.rows[5]["challenge_status"] == STATUS_IN_PROGRESS


@pytest.mark.asyncio
async def test_reaper_respects_grace_window():
    """An in_progress row younger than ORPHAN_GRACE_SECONDS must not be
    reaped — guards the legitimate pick_next → set_battle window."""
    from affine.src.scheduler.flow import ORPHAN_GRACE_SECONDS
    import time as _time

    kv = _seed_state()
    miner_store = _InMemoryMinerStore([
        _make_miner(1, "champ_hk", 100, status=STATUS_CHAMPION, revision="champ_rev"),
        _make_miner(99, "fresh", 50, status=STATUS_IN_PROGRESS, revision="fresh_rev"),
    ])
    # Claimed just now — well within grace.
    miner_store.rows[99]["challenge_claimed_at"] = (
        int(_time.time()) - (ORPHAN_GRACE_SECONDS // 2)
    )
    scheduler, state, _ = _build_scheduler(
        kv=kv, miner_store=miner_store,
        deployer=_DeployTracker(), samples=_SamplesFake(),
    )

    await scheduler.tick(current_block=50)

    assert miner_store.rows[99]["challenge_status"] == STATUS_IN_PROGRESS


def test_format_cause_chain_linear():
    """Walks the full __cause__ chain, outermost-cause-first suffix."""
    from affine.src.scheduler.flow import _format_cause_chain
    root = ValueError("docker stderr")
    mid = RuntimeError("mid")
    top = Exception("wrapper")
    mid.__cause__ = root
    top.__cause__ = mid
    assert _format_cause_chain(top) == (
        " -> RuntimeError: mid -> ValueError: docker stderr"
    )


def test_format_cause_chain_none():
    """No __cause__ → empty suffix (the bare wrapper is logged on its own)."""
    from affine.src.scheduler.flow import _format_cause_chain
    assert _format_cause_chain(Exception("x")) == ""


def test_format_cause_chain_cycle_terminates():
    """A cyclic __cause__ (a <-> b) must not loop forever; the seen-set
    breaks it and the line stays bounded."""
    from affine.src.scheduler.flow import _format_cause_chain
    a = Exception("A")
    b = Exception("B")
    a.__cause__ = b
    b.__cause__ = a
    out = _format_cause_chain(a)
    assert out == " -> Exception: B -> Exception: A"


def test_format_cause_chain_depth_capped():
    """A chain deeper than max_depth truncates to ' -> ...' so a deep wrap
    can't produce an unbounded log line."""
    from affine.src.scheduler.flow import _format_cause_chain
    top = Exception("top")
    cur = top
    for i in range(20):
        nxt = Exception(f"e{i}")
        cur.__cause__ = nxt
        cur = nxt
    out = _format_cause_chain(top, max_depth=8)
    assert out.count(" -> ") == 9          # 8 causes + the truncation marker
    assert out.endswith(" -> ...")
