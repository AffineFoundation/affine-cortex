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
    FlowConfig,
    FlowScheduler,
)
from affine.src.scheduler.targon import DeployResult, DeployTarget
from affine.src.scorer.challenger_queue import (
    ChallengerQueue,
    STATUS_CHAMPION,
    STATUS_IN_PROGRESS,
    STATUS_SAMPLING,
    STATUS_TERMINATED,
)
from affine.src.scorer.comparator import WindowComparator
from affine.src.scorer.sampler import WindowSampler
from affine.src.scorer.weight_writer import WeightWriter
from affine.src.scorer.window_state import (
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
        if scores_refresh_block is not None:
            self.rows[uid]["scores_refresh_block"] = scores_refresh_block
        if terminated_at_block is not None:
            self.rows[uid]["terminated_at_block"] = terminated_at_block


@dataclass
class _DeployTracker:
    deploys: List[DeployTarget] = field(default_factory=list)
    teardowns: List[Optional[str]] = field(default_factory=list)
    fail_on_deploy: bool = False
    next_deployment_id: int = 0

    async def deploy(self, target: DeployTarget, role: str = "active") -> DeployResult:
        self.deploys.append(target)
        if self.fail_on_deploy:
            raise RuntimeError("simulated targon deploy failure")
        self.next_deployment_id += 1
        did = f"wrk-{self.next_deployment_id:03d}"
        return DeployResult(deployment_id=did, base_url=f"https://t/{did}")

    async def teardown(self, deployment_id: Optional[str]) -> None:
        self.teardowns.append(deployment_id)


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
    window_blocks=WINDOW_BLOCKS,
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
        config=FlowConfig(window_blocks=window_blocks),
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
    )
    return scheduler, state, weight_writer


# ---- tests -----------------------------------------------------------------


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

    # Jump forward 7200 blocks while battle still in flight — must NOT refresh.
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
async def test_deploy_failure_marks_challenger_failed():
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
    assert miner_store.rows[2]["challenge_status"] == STATUS_TERMINATED
    assert await state.get_battle() is None


@pytest.mark.asyncio
async def test_invalid_champion_dropped_and_workload_torn_down():
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
    await scheduler.tick(current_block=51)   # validation drops champion + cold-starts to next valid
    # The stale champion's Targon was torn down.
    assert "wrk-old" in deployer.teardowns
    # Cold-start path took over: uid=2 promoted as new champion.
    new_champ = await state.get_champion()
    assert new_champ is not None and new_champ.uid == 2


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
async def test_post_promotion_crash_recovery_does_not_self_demote():
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

    # Champion preserved as uid 2.
    champ = await state.get_champion()
    assert champ is not None and champ.uid == 2, (
        f"recovery wrongly demoted the just-promoted champion: got {champ}"
    )
    # Battle cleared.
    assert await state.get_battle() is None
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
        scores_refresh_block=None,
        terminated_at_block=None,
    ):
        if status == STATUS_CHAMPION:
            write_log.append(f"mark_won_uid={uid}")
        await orig_set_terminal(
            uid, status, reason=reason,
            hotkey=hotkey, revision=revision, model=model,
            scores_by_env=scores_by_env,
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
    # Challenger marked terminated (used their shot).
    assert miner_store.rows[2]["challenge_status"] in (STATUS_TERMINATED,)
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
        config=FlowConfig(window_blocks=WINDOW_BLOCKS, single_instance_provider=True),
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

    # Force a refresh by advancing block past window_blocks. Battle is
    # None, so refresh fires.
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
        config=FlowConfig(window_blocks=WINDOW_BLOCKS, single_instance_provider=True),
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


@pytest.mark.asyncio
async def test_winning_challenger_must_top_up_samples_before_next_battle():
    """WIN path: when the challenger wins, it only needed ``sampling_count``
    overlap (the ``_battle_overlap_ready`` threshold) to be allowed to
    battle. But the champion threshold is the higher 95%-of-pool. The
    just-promoted champion must therefore complete supplementary
    sampling — step 6 must block step 7 until the new champion reaches
    the champion threshold."""
    from affine.src.scheduler.flow import FlowConfig
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
        config=FlowConfig(window_blocks=WINDOW_BLOCKS, single_instance_provider=True),
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
        config=FlowConfig(window_blocks=WINDOW_BLOCKS, single_instance_provider=True),
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
