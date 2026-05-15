"""
Flow scheduler — replaces the old window-state-machine driver.

One block tick:

  1. Bootstrap task_ids if never refreshed.
  2. Refresh task_ids if 7200 blocks elapsed AND no battle in flight
     (don't disrupt a contest mid-flow).
  3. Validate champion against the valid-miner set; drop a stale one.
  4. No champion → bootstrap-promote the earliest pending miner (no
     contest, no Targon).
  5. Champion has no Targon workload → deploy it.
  6. Champion samples for the current task_ids not yet full → return
     (executors are filling them).
  7. No in-flight battle → pick next challenger, deploy, record battle.
  8. Battle challenger samples not yet full → return.
  9. Both subjects done → run comparator, transition champion (or drop
     challenger), write weights when the champion changes, clear battle.

The state machine is implicit — no ``phase`` / ``status`` enum. The
record shape itself tells us where we are.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional

from affine.core.setup import logger

from affine.src.scorer.challenger_queue import (
    ChallengerQueue,
    OUTCOME_LOST,
    OUTCOME_WON,
)
from affine.src.scorer.comparator import (
    EnvComparisonConfig,
    WindowComparator,
)
from affine.src.scorer.sampler import EnvSamplingConfig, WindowSampler
from affine.src.scorer.sampling_thresholds import (
    SAMPLE_BUFFER_RATIO,
    champion_completion_threshold,
)
from affine.src.scorer.weight_writer import WeightSubject, WeightWriter
from affine.src.scorer.window_state import (
    BattleRecord,
    ChampionRecord,
    DeploymentRecord,
    EnvConfig,
    MinerSnapshot,
    StateStore,
    TaskIdState,
)

from . import targon as targon_lifecycle


# ---- constants (no longer in system_config) --------------------------------


import math

WINDOW_BLOCKS = 7200
"""How often the per-env task_id pool is regenerated."""

DEFAULT_MARGIN = 0.03
"""Per-env additive margin the challenger must clear to be ``dominant``."""

DEFAULT_NOT_WORSE_TOLERANCE = 0.02
"""Multiplicative regression tolerance; challenger must keep >= champion * 0.98."""

WIN_MIN_DOMINANT_ENVS = 1
"""Partial Pareto: at least one env must be dominant; the rest must not regress."""

# ``SAMPLE_BUFFER_RATIO`` lives in ``sampling_thresholds`` and is
# re-exported above so downstream importers (and tests) keep working.


# ---- callable types --------------------------------------------------------


SampleCountFn = Callable[[str, str, str, List[int], int], Awaitable[int]]
"""(hotkey, revision, env, task_ids, refresh_block) → number of those task_ids
that have a sample_results row tagged with the given refresh_block."""

ScoresReader = Callable[[str, str, str, List[int], int],
                        Awaitable[Dict[int, float]]]
"""(hotkey, revision, env, task_ids, refresh_block) → {task_id: score} for
matches in the current refresh."""

ListValidMinersFn = Callable[[], Awaitable[List[Dict[str, Any]]]]
"""Return every is_valid=true row in the miners table."""

DeployFn = Callable[[targon_lifecycle.DeployTarget, str],
                    Awaitable[targon_lifecycle.DeployResult]]
TeardownFn = Callable[[Optional[str]], Awaitable[None]]


@dataclass
class FlowConfig:
    window_blocks: int = WINDOW_BLOCKS
    scorer_hotkey: str = "scheduler"
    single_instance_provider: bool = False
    """True when the inference provider hosts only one model at a time
    (e.g. SSH/sglang on a single GPU host that swaps models). In that
    mode, the champion's deployment becomes stale whenever:
      - a challenger is deployed (b300 now serves challenger's model)
      - a challenger is torn down on loss (b300 ends up empty)
      - a task-id refresh fires (b300 may be holding an old model from
        a previous battle)
    Flow clears ``champion.deployment_id`` at those points so step 5
    re-deploys champion fresh. Targon-style multi-instance providers
    leave this False — each deployment is independent and survives
    its sibling's teardown."""


class FlowScheduler:
    def __init__(
        self,
        *,
        config: FlowConfig,
        state: StateStore,
        queue: ChallengerQueue,
        sampler: WindowSampler,
        comparator: WindowComparator,
        weight_writer: WeightWriter,
        deploy_fn: DeployFn,
        teardown_fn: TeardownFn,
        sample_count_fn: SampleCountFn,
        scores_reader: ScoresReader,
        list_valid_miners_fn: ListValidMinersFn,
    ):
        self.cfg = config
        self.state = state
        self.queue = queue
        self.sampler = sampler
        self.comparator = comparator
        self.weight_writer = weight_writer
        self._deploy = deploy_fn
        self._teardown = teardown_fn
        self._sample_count = sample_count_fn
        self._scores_reader = scores_reader
        self._list_valid_miners = list_valid_miners_fn

    # ---- entry point ------------------------------------------------------

    async def tick(self, current_block: int) -> None:
        envs = await self.state.get_environments()
        if not envs:
            logger.warning("FlowScheduler: no sampling-enabled envs; skipping tick")
            return

        battle = await self.state.get_battle()
        task_state = await self.state.get_task_state()

        # 1 + 2. Task-id pool refresh (between battles only).
        if task_state is None or (
            battle is None
            and current_block - task_state.refreshed_at_block >= self.cfg.window_blocks
        ):
            task_state = await self._refresh_task_ids(current_block, envs)
            return  # let executors warm up on the new pool

        # 3. Validate champion against valid-miner set.
        champion = await self._read_validated_champion()

        # 4. Cold start.
        if champion is None:
            await self._cold_start(current_block)
            return

        # Crash-recovery must run before deployment/sample gates. If a
        # previous tick promoted the challenger then crashed before
        # ``clear_battle()``, the persisted state has the same UID in
        # champion and battle.challenger. Waiting for fresh samples here
        # can leave that stale battle stuck forever after threshold changes.
        if battle is not None and champion.uid == battle.challenger.uid:
            scoring_envs = await self.state.get_scoring_environments()
            await self._decide(
                champion, battle, scoring_envs, task_state, current_block,
            )
            return

        # 5. Champion needs inference. During a single-instance battle the
        # champion may intentionally have no live deployment because its
        # samples were completed before the challenger reused the machine.
        #
        # Single-instance loss-path fast-path: when the just-ended battle
        # was a loss, the champion is unchanged so its samples are still
        # complete (loss doesn't touch sample_results). Skip the ~2-minute
        # champion redeploy and dispatch the next challenger directly —
        # ``_start_battle`` would overwrite this deployment within seconds
        # anyway. When the queue is empty we fall through to the redeploy
        # so b300 isn't sitting empty.
        if battle is None and (not champion.deployment_id or not champion.base_url):
            if self.cfg.single_instance_provider and await self._samples_complete(
                MinerSnapshot(uid=champion.uid, hotkey=champion.hotkey,
                              revision=champion.revision, model=champion.model),
                envs=envs, task_state=task_state,
            ):
                await self._start_battle(champion, current_block)
                if await self.state.get_battle() is not None:
                    return
                # Queue empty — fall through to redeploy champion so the
                # endpoint isn't idle until a new challenger arrives.
            await self._deploy_champion(champion)
            return

        # 6. Champion samples not yet full.
        if not await self._samples_complete(
            MinerSnapshot(uid=champion.uid, hotkey=champion.hotkey,
                          revision=champion.revision, model=champion.model),
            envs=envs, task_state=task_state,
        ):
            return

        # 7. No battle — start one.
        if battle is None:
            await self._start_battle(champion, current_block)
            return

        # 8. Battle overlap not yet sufficient — wait for both miners to
        # have ≥ sampling_count current-refresh task_ids in common per env.
        if not await self._battle_overlap_ready(
            champion, battle.challenger, envs=envs, task_state=task_state,
        ):
            return

        # 9. Sufficient overlap — run the contest on overlap task_ids only.
        # The comparator sees only envs whose ``enabled_for_scoring`` is
        # true; sampling-only envs accumulate data without affecting
        # DECIDE. Same ``envs`` dict object stays in use for any tick
        # paths that need the full sampling set.
        scoring_envs = await self.state.get_scoring_environments()
        await self._decide(
            champion, battle, scoring_envs, task_state, current_block,
        )

    # ---- helpers ----------------------------------------------------------

    async def _read_validated_champion(self) -> Optional[ChampionRecord]:
        """Return the saved champion only if their hotkey is still
        is_valid=true. Monitor may invalidate a champion post-hoc
        (multi_commit / repo-name / blacklist) — the on-chain weight tx
        for an invalid hotkey would fail later, so drop the record now
        and fall through to the cold-start path."""
        champ = await self.state.get_champion()
        if champ is None:
            return None
        valid = await self._list_valid_miners()
        valid_uids = {int(r.get("uid", -1)) for r in valid}
        if champ.uid not in valid_uids:
            logger.warning(
                f"FlowScheduler: champion uid={champ.uid} no longer valid; "
                f"dropping. Tearing down any live workload."
            )
            await self._teardown_record(champ)
            await self.state.clear_champion()
            return None
        return champ

    async def _refresh_task_ids(
        self, current_block: int, envs: Mapping[str, EnvConfig],
    ) -> TaskIdState:
        """Generate a fresh per-env task_id pool, oversampled by
        ``SAMPLE_BUFFER_RATIO`` so the contest can decide as soon as
        the (champion ∩ challenger) overlap reaches the base
        ``sampling_count``. Slow-tail / errored task_ids in the buffer
        portion are simply abandoned.

        Envs with a ``dataset_range_source`` (SWE-INFINITE, DISTILL —
        datasets that grow over time) get their ``dataset_range``
        resolved against the remote metadata before sampling. Falls
        back to the static config range on any resolver failure.
        """
        from affine.src.scorer.dataset_range_resolver import resolve_dataset_range

        env_configs: Dict[str, EnvSamplingConfig] = {}
        for env, cfg in envs.items():
            resolved_range = cfg.dataset_range
            if cfg.dataset_range_source:
                fresh = await resolve_dataset_range(cfg.dataset_range_source)
                if fresh is not None and fresh != cfg.dataset_range:
                    logger.info(
                        f"FlowScheduler: dataset_range for {env} refreshed via "
                        f"source: {cfg.dataset_range} → {fresh}"
                    )
                    resolved_range = fresh
            env_configs[env] = EnvSamplingConfig(
                env=env,
                sampling_count=math.ceil(cfg.sampling_count * (1 + SAMPLE_BUFFER_RATIO)),
                dataset_range=resolved_range,
                mode=cfg.sampling_mode,
            )
        task_ids = self.sampler.generate(
            window_id=current_block // self.cfg.window_blocks,
            block_start=current_block,
            env_configs=env_configs,
        )
        new_state = TaskIdState(
            task_ids=task_ids, refreshed_at_block=current_block,
        )
        await self.state.set_task_state(new_state)

        # Single-instance provider: the inference host may be empty or
        # holding a stale challenger's model from before the refresh.
        # Clear champion.deployment_id so step 5 re-deploys champion onto
        # the host before any new sampling against the (possibly wrong)
        # base_url.
        if self.cfg.single_instance_provider:
            champ = await self.state.get_champion()
            if champ is not None and champ.deployment_id is not None:
                champ.deployment_id = None
                champ.base_url = None
                champ.deployments = []
                await self.state.set_champion(champ)
                logger.info(
                    "FlowScheduler: cleared champion.deployment_id at refresh "
                    "(single-instance provider — will redeploy next tick)"
                )

        logger.info(
            f"FlowScheduler: refreshed task_ids at block {current_block}, "
            f"{sum(len(v) for v in task_ids.values())} task units across "
            f"{len(task_ids)} envs (incl. {int(SAMPLE_BUFFER_RATIO * 100)}% buffer)"
        )
        return new_state

    async def _cold_start(self, current_block: int) -> None:
        """No champion yet — promote the earliest pending miner directly,
        no contest. Targon will be deployed on the next tick.

        Order: ``set_champion`` writes BEFORE ``mark_terminated``. If we
        crashed in between, recovery would read ``system_config.champion``
        on the next tick, see the right miner, and continue. The reverse
        order would leave us with ``miner_stats[X].challenge_status='champion'``
        but ``system_config.champion=None``; recovery's cold_start would
        then skip X (since queue.pick_next filters by status pending) and
        promote the next miner Y instead, stranding X permanently as
        STATUS_CHAMPION orphan.
        """
        candidate = await self.queue.pick_next(window_id=0, champion_uid=None)
        if candidate is None:
            return  # no eligible miners yet
        new_champ = ChampionRecord(
            uid=candidate.uid, hotkey=candidate.hotkey,
            revision=candidate.revision, model=candidate.model,
            since_block=current_block,
        )
        await self.state.set_champion(new_champ)
        await self.queue.mark_terminated(
            candidate.uid,
            OUTCOME_WON,
            hotkey=candidate.hotkey,
            revision=candidate.revision,
            model=candidate.model,
        )
        logger.info(
            f"FlowScheduler: bootstrap champion = uid {candidate.uid}"
        )

    async def _deploy_champion(self, champion: ChampionRecord) -> None:
        """Bring up the champion's Targon workload. Adopt an existing one
        if it matches the naming convention (idempotent across restarts)."""
        target = targon_lifecycle.DeployTarget(
            uid=champion.uid, hotkey=champion.hotkey,
            model=champion.model, revision=champion.revision,
        )
        try:
            result = await self._deploy(target, "champion")
        except Exception as e:
            logger.error(
                f"FlowScheduler: champion deploy failed for uid={champion.uid}: "
                f"{type(e).__name__}: {e}"
            )
            return
        deployments = _deployments_from_result(result)
        champion.deployments = deployments
        champion.deployment_id = result.deployment_id
        champion.base_url = result.base_url
        await self.state.set_champion(champion)
        logger.info(
            f"FlowScheduler: champion uid={champion.uid} deployed at {result.base_url}"
        )

    async def _samples_complete(
        self, miner: MinerSnapshot, *,
        envs: Mapping[str, EnvConfig], task_state: TaskIdState,
    ) -> bool:
        """True iff ``miner`` has ≥ ``champion_completion_threshold`` (95%
        of the pool) current-refresh samples in every sampling-enabled
        env. The pool is oversampled to ``ceil(sampling_count * 1.1)``;
        the 5% gap between the threshold and the full pool is the
        deliberately-abandoned long tail.

        Lower thresholds (eg the original ``sampling_count``) made
        ``_battle_overlap_ready`` mathematically unsatisfiable —
        challenger overlap with a champion missing 10% of the pool
        averages ``sampling_count × 200/220 ≈ 182`` and never hits the
        base ``sampling_count``. See
        ``affine/src/scorer/sampling_thresholds.py``."""
        for env, env_cfg in envs.items():
            tasks = task_state.task_ids.get(env, [])
            if not tasks:
                continue
            n = await self._sample_count(
                miner.hotkey, miner.revision, env, tasks,
                task_state.refreshed_at_block,
            )
            if n < champion_completion_threshold(env_cfg.sampling_count):
                return False
        return True

    async def _battle_overlap_ready(
        self,
        champion: ChampionRecord,
        challenger: MinerSnapshot,
        *,
        envs: Mapping[str, EnvConfig],
        task_state: TaskIdState,
    ) -> bool:
        """True iff for every env the (champion ∩ challenger) overlap of
        current-refresh sampled task_ids is ≥ ``sampling_count``.

        This is the gate for running the comparator. The user-facing
        guarantee: contests only run on samples both sides produced for
        the SAME tasks in the SAME refresh — no cross-window leakage."""
        for env, env_cfg in envs.items():
            tasks = task_state.task_ids.get(env, [])
            if not tasks:
                continue
            champ_scores = await self._scores_reader(
                champion.hotkey, champion.revision, env, tasks,
                task_state.refreshed_at_block,
            )
            chal_scores = await self._scores_reader(
                challenger.hotkey, challenger.revision, env, tasks,
                task_state.refreshed_at_block,
            )
            overlap = set(champ_scores) & set(chal_scores)
            if len(overlap) < env_cfg.sampling_count:
                return False
        return True

    async def _start_battle(
        self, champion: ChampionRecord, current_block: int,
    ) -> None:
        """Pick the next pending miner and stand up its Targon workload."""
        candidate = await self.queue.pick_next(
            window_id=current_block // self.cfg.window_blocks,
            champion_uid=champion.uid,
        )
        if candidate is None:
            return  # idle, no challengers
        target = targon_lifecycle.DeployTarget(
            uid=candidate.uid, hotkey=candidate.hotkey,
            model=candidate.model, revision=candidate.revision,
        )
        try:
            result = await self._deploy(target, "challenger")
        except Exception as e:
            logger.error(
                f"FlowScheduler: challenger deploy failed uid={candidate.uid}: "
                f"{type(e).__name__}: {e}"
            )
            # Used their one shot — mark FAILED so the queue doesn't keep
            # retrying a model the platform can't host. No frozen scores
            # to pass: this miner never sampled anything.
            from affine.src.scorer.challenger_queue import OUTCOME_FAILED
            await self.queue.mark_terminated(
                candidate.uid,
                OUTCOME_FAILED,
                reason=f"deployment_failed:{type(e).__name__}",
                hotkey=candidate.hotkey,
                revision=candidate.revision,
                model=candidate.model,
            )
            return
        if self.cfg.single_instance_provider:
            champion.deployment_id = None
            champion.base_url = None
            champion.deployments = []
            await self.state.set_champion(champion)
        battle = BattleRecord(
            challenger=MinerSnapshot(
                uid=candidate.uid, hotkey=candidate.hotkey,
                revision=candidate.revision, model=candidate.model,
            ),
            deployment_id=result.deployment_id,
            base_url=result.base_url,
            started_at_block=current_block,
            deployments=_deployments_from_result(result),
            # Captured BEFORE ``set_champion(new)`` ever runs so the
            # _decide recovery branch can still terminate the old
            # champion and re-emit weights after a mid-flow crash.
            previous_champion=MinerSnapshot(
                uid=champion.uid, hotkey=champion.hotkey,
                revision=champion.revision, model=champion.model,
            ),
        )
        await self.state.set_battle(battle)
        logger.info(
            f"FlowScheduler: battle started — challenger uid={candidate.uid} "
            f"vs champion uid={champion.uid}"
        )

    async def _decide(
        self,
        champion: ChampionRecord,
        battle: BattleRecord,
        envs: Mapping[str, EnvConfig],
        task_state: TaskIdState,
        current_block: int,
    ) -> None:
        """Pull both subjects' scores, run the comparator, transition.

        When the challenger wins:
          - Teardown the old champion's Targon.
          - Promote the challenger's Targon to the new champion's slot
            (no redeploy — same workload keeps serving).
          - Mark old champion ``LOST``, new champion ``WON``.
          - Re-write weights (champion changed).

        When the champion wins:
          - Teardown the challenger's Targon.
          - Mark challenger ``LOST``.
          - No weight write (champion stable).
        """
        # Recovery guard. If a prior tick crashed between ``set_champion(new)``
        # and ``clear_battle()``, the saved state has the just-promoted
        # miner stored both as the champion AND as the in-flight battle's
        # challenger. Naively running the comparator on (champion vs same
        # miner) would tie, hit the "challenger loses" branch, and tear
        # down + demote the freshly-crowned champion. Detect that case
        # explicitly and just finalize the bookkeeping.
        if champion.uid == battle.challenger.uid:
            logger.info(
                f"FlowScheduler: detected post-promotion crash recovery "
                f"for uid={champion.uid}; finalizing bookkeeping without "
                f"re-running comparator"
            )
            await self.queue.mark_terminated(
                champion.uid,
                OUTCOME_WON,
                hotkey=champion.hotkey,
                revision=champion.revision,
                model=champion.model,
            )
            # Old champion must also be flipped to ``terminated`` and the
            # on-chain weight tx re-emitted — both got skipped if the
            # crash happened between ``set_champion(new)`` and those
            # steps. ``previous_champion`` is captured at battle start so
            # we can still identify them after the canonical champion
            # record was overwritten. The reason string mirrors the
            # normal-path one in spirit but names the path explicitly:
            # if the crash hit after ``mark_terminated(LOST)`` already
            # ran, this write will overwrite the (more detailed) original
            # reason — acceptable since both leave the row terminated.
            prev = battle.previous_champion
            prev_ready = (
                prev is not None
                and prev.uid != champion.uid
                and prev.hotkey
                and prev.revision
            )
            if prev_ready:
                # No frozen scores: comparator ``result`` was lost in
                # the crash. Pre-crash freeze (if any) survives — DAO
                # SET is field-additive.
                await self.queue.mark_terminated(
                    prev.uid,
                    OUTCOME_LOST,
                    reason=f"dethroned_by:{champion.hotkey[:10]}:recovery",
                    hotkey=prev.hotkey,
                    revision=prev.revision,
                    model=prev.model,
                )
                # ``result=None`` is fine — the comparator-derived per-env
                # metadata becomes empty dicts; the actual weight tx
                # (1.0 for champion, 0 for the rest) still happens.
                try:
                    await self._write_weights(
                        champion, current_block, None, previous_champion=prev,
                    )
                except Exception as e:
                    logger.error(
                        f"FlowScheduler: recovery weight write failed: "
                        f"{type(e).__name__}: {e}"
                    )
            elif prev is None:
                logger.warning(
                    "FlowScheduler: recovery path lacked previous_champion "
                    "(pre-fix battle record); old champion not terminated "
                    "and weights not re-emitted — operator must verify."
                )
            else:
                logger.warning(
                    f"FlowScheduler: recovery saw malformed previous_champion "
                    f"(uid={prev.uid}, hotkey={prev.hotkey!r}, "
                    f"revision={prev.revision!r}); skipping termination + "
                    f"weight re-emit, operator must verify."
                )
            await self.state.clear_battle()
            return

        # Mid-battle invalidation guard. Monitor can flip ``is_valid`` to
        # false on the challenger between ``pick_next`` (start_battle) and
        # this decide tick (multi-commit / blacklist / repo-name mismatch
        # picked up late). Promoting an invalid hotkey would write
        # ``scores.overall_score = 1.0`` for them and the validator would
        # try to set on-chain weight for a banned miner. Treat as
        # "challenger loses" regardless of the comparator outcome.
        valid_uids = {int(r.get("uid", -1)) for r in await self._list_valid_miners()}
        if battle.challenger.uid not in valid_uids:
            logger.warning(
                f"FlowScheduler: challenger uid={battle.challenger.uid} "
                f"invalidated mid-battle; forcing LOST regardless of scores"
            )
            await self._teardown_record(battle)
            # No frozen scores: kicked for being invalid, not on score.
            await self.queue.mark_terminated(
                battle.challenger.uid,
                OUTCOME_LOST,
                reason="invalidated_mid_battle",
                hotkey=battle.challenger.hotkey,
                revision=battle.challenger.revision,
                model=battle.challenger.model,
            )
            await self.state.clear_battle()
            return

        env_configs = {
            env: EnvComparisonConfig(
                env=env,
                margin=DEFAULT_MARGIN,
                # Comparator's per-env sample-count gate. Aligned with
                # ``_battle_overlap_ready`` (which requires overlap ≥
                # ``sampling_count``) so the two gates can't disagree:
                # SWE=200, MEMORY=100, etc. — full coverage required.
                min_tasks_per_env=max(1, int(cfg.sampling_count)),
                not_worse_tolerance=DEFAULT_NOT_WORSE_TOLERANCE,
            )
            for env, cfg in envs.items()
        }
        # Read each side's current-refresh scores, then restrict the
        # comparator's view to the overlap (task_ids both miners have
        # sampled in this refresh). Asymmetric pool coverage gets
        # neutralised here — both sides see exactly the same task_ids.
        champ_scores: Dict[str, Dict[int, float]] = {}
        chal_scores: Dict[str, Dict[int, float]] = {}
        for env in env_configs:
            tasks = task_state.task_ids.get(env, [])
            champ_full = await self._scores_reader(
                champion.hotkey, champion.revision, env, tasks,
                task_state.refreshed_at_block,
            )
            chal_full = await self._scores_reader(
                battle.challenger.hotkey, battle.challenger.revision, env, tasks,
                task_state.refreshed_at_block,
            )
            overlap = set(champ_full) & set(chal_full)
            champ_scores[env] = {t: champ_full[t] for t in overlap}
            chal_scores[env] = {t: chal_full[t] for t in overlap}

        result = self.comparator.compare(
            champion_scores=champ_scores,
            challenger_scores=chal_scores,
            env_configs=env_configs,
            min_dominant_envs=WIN_MIN_DOMINANT_ENVS,
        )

        if result.winner == "challenger":
            await self._teardown_record(champion)
            new_champion = ChampionRecord(
                uid=battle.challenger.uid,
                hotkey=battle.challenger.hotkey,
                revision=battle.challenger.revision,
                model=battle.challenger.model,
                deployment_id=battle.deployment_id,
                base_url=battle.base_url,
                deployments=list(battle.deployments),
                since_block=current_block,
            )
            # Order: write the canonical champion record FIRST so any
            # concurrent reader sees the new identity before either miners
            # row flips.
            await self.state.set_champion(new_champion)
            await self.queue.mark_terminated(
                battle.challenger.uid,
                OUTCOME_WON,
                hotkey=battle.challenger.hotkey,
                revision=battle.challenger.revision,
                model=battle.challenger.model,
            )
            await self.queue.mark_terminated(
                champion.uid,
                OUTCOME_LOST,
                reason=f"dethroned_by:{battle.challenger.hotkey[:10]}",
                hotkey=champion.hotkey,
                revision=champion.revision,
                model=champion.model,
                scores_by_env=_final_scores_from_result(
                    result, role="champion",
                ),
                scores_refresh_block=task_state.refreshed_at_block,
                terminated_at_block=current_block,
            )
            await self._write_weights(
                new_champion, current_block, result,
                previous_champion=champion,
            )
            logger.info(
                f"FlowScheduler: champion uid {champion.uid} dethroned by "
                f"uid {battle.challenger.uid}"
            )
        else:
            await self._teardown_record(battle)
            await self.queue.mark_terminated(
                battle.challenger.uid,
                OUTCOME_LOST,
                reason=f"lost_to_champion:{champion.hotkey[:10]}:{result.reason}",
                hotkey=battle.challenger.hotkey,
                revision=battle.challenger.revision,
                model=battle.challenger.model,
                scores_by_env=_final_scores_from_result(
                    result, role="challenger",
                ),
                scores_refresh_block=task_state.refreshed_at_block,
                terminated_at_block=current_block,
            )
            # Single-instance provider: the teardown just emptied the
            # inference host (champion's container went away with the
            # challenger's — they're the same container). Champion's
            # deployment_id is now stale. Clear it so step 5 re-deploys
            # champion onto the host before the next ``_samples_complete``
            # would otherwise sample against a dead URL.
            if self.cfg.single_instance_provider:
                champion.deployment_id = None
                champion.base_url = None
                champion.deployments = []
                await self.state.set_champion(champion)
            logger.info(
                f"FlowScheduler: champion uid {champion.uid} held against "
                f"uid {battle.challenger.uid}: {result.reason}"
            )

        await self.state.clear_battle()

    async def _teardown_record(self, record: Any) -> None:
        deployments = list(getattr(record, "deployments", []) or [])
        if deployments:
            for dep in deployments:
                await self._teardown(dep.deployment_id)
            return
        await self._teardown(getattr(record, "deployment_id", None))

    async def _write_weights(
        self, champion: ChampionRecord, block_number: int, result: Any,
        previous_champion: Optional[ChampionRecord] = None,
    ) -> None:
        """Build per-uid subjects (1.0 for champion, 0.0 for everyone else)
        and persist via ``WeightWriter``."""
        champion_env_scores = _comparison_scores_by_env(result, role="challenger")
        previous_champion_env_scores = _comparison_scores_by_env(result, role="champion")
        valid = await self._list_valid_miners()
        subjects: List[WeightSubject] = []
        seen: set = set()
        for m in valid:
            uid = int(m.get("uid", -1))
            if uid < 0 or uid in seen:
                continue
            seen.add(uid)
            subjects.append(
                WeightSubject(
                    uid=uid,
                    hotkey=str(m.get("hotkey", "")),
                    revision=str(m.get("revision", "")),
                    model=str(m.get("model", "")),
                    first_block=int(m.get("first_block", 0)),
                    is_champion=(uid == champion.uid),
                    scores_by_env=(
                        champion_env_scores if uid == champion.uid
                        else (
                            previous_champion_env_scores
                            if previous_champion and uid == previous_champion.uid
                            else {}
                        )
                    ),
                    total_samples=int(m.get("total_samples", 0)),
                )
            )
        if not any(s.is_champion for s in subjects):
            subjects.append(
                WeightSubject(
                    uid=champion.uid, hotkey=champion.hotkey,
                    revision=champion.revision, model=champion.model,
                    first_block=0, is_champion=True,
                    scores_by_env=champion_env_scores, total_samples=0,
                )
            )
        await self.weight_writer.write(
            window_id=block_number,  # snapshots are keyed by block, not window
            block_number=block_number,
            scorer_hotkey=self.cfg.scorer_hotkey,
            subjects=subjects,
            outcome=_outcome_for_snapshot(result, champion),
            rules=_rules_for_snapshot(),
        )


def _deployments_from_result(result: targon_lifecycle.DeployResult) -> List[DeploymentRecord]:
    out: List[DeploymentRecord] = []
    for dep in getattr(result, "deployments", []) or []:
        out.append(
            DeploymentRecord(
                endpoint_name=getattr(dep, "endpoint_name", ""),
                deployment_id=dep.deployment_id,
                base_url=dep.base_url,
            )
        )
    if not out and result.deployment_id and result.base_url:
        out.append(
            DeploymentRecord(
                endpoint_name="",
                deployment_id=result.deployment_id,
                base_url=result.base_url,
            )
        )
    return out


def _as_float_or_none(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _final_scores_from_result(
    result: Any, *, role: str,
) -> Dict[str, Dict[str, float]]:
    """Build ``{env: {count, avg, champion_overlap_avg}}`` for one side of
    a finished contest. Used to freeze the comparator's decide-time view
    of the loser onto their miner_stats row, so rank UI keeps showing
    count / avg / threshold after the live cache forgets them.

    ``role`` is ``"champion"`` (record for a displaced champion) or
    ``"challenger"`` (record for a challenger that lost). The opposing
    side's avg is recorded as ``champion_overlap_avg`` — the basis the
    comparator actually used to decide this contest.
    """
    out: Dict[str, Dict[str, float]] = {}
    for e in getattr(result, "per_env", []) or []:
        if role == "champion":
            own_n = int(getattr(e, "champion_n", 0) or 0)
            own_avg = _as_float_or_none(getattr(e, "champion_avg", None))
            opp_avg = _as_float_or_none(getattr(e, "challenger_avg", None))
        else:
            own_n = int(getattr(e, "challenger_n", 0) or 0)
            own_avg = _as_float_or_none(getattr(e, "challenger_avg", None))
            opp_avg = _as_float_or_none(getattr(e, "champion_avg", None))
        if own_n <= 0 or own_avg is None:
            continue
        entry: Dict[str, float] = {"count": own_n, "avg": own_avg}
        if opp_avg is not None:
            entry["champion_overlap_avg"] = opp_avg
        out[str(getattr(e, "env", ""))] = entry
    return out


def _env_comparison_to_dict(e: Any) -> Dict[str, Any]:
    """Serialize one ``EnvComparison`` row into a plain dict for the
    snapshot. Captures every field the comparator considered for this
    env so the snapshot is self-describing for audit."""
    return {
        "env": getattr(e, "env", ""),
        "champion_avg": _as_float_or_none(getattr(e, "champion_avg", None)),
        "challenger_avg": _as_float_or_none(getattr(e, "challenger_avg", None)),
        "champion_n": int(getattr(e, "champion_n", 0) or 0),
        "challenger_n": int(getattr(e, "challenger_n", 0) or 0),
        "delta": _as_float_or_none(getattr(e, "delta", None)),
        "margin": float(getattr(e, "margin", 0.0) or 0.0),
        "not_worse_tolerance": float(getattr(e, "not_worse_tolerance", 0.0) or 0.0),
        "verdict": getattr(e, "verdict", ""),
        "reason": getattr(e, "reason", ""),
    }


def _rules_for_snapshot() -> Dict[str, Any]:
    """Rule constants persisted flat at the top of ``snapshot.config``.

    Pre-#449 the API returned ``ScorerConfig.to_dict()`` directly as
    ``snapshot.config``, so clients read ``config.win_min_dominant_envs``
    (etc.) at the top level. Keep them flat here so the response shape
    stays at the original path; the field names match the pre-refactor
    ones where the concept is preserved.

    - ``win_margin``: pre-refactor ``win_margin_end`` (the old system
      had a start/end ramp; the new system uses a single static value).
    - ``win_not_worse_tolerance``: same name as pre-refactor.
    - ``win_min_dominant_envs``: same name as pre-refactor.

    The comparator's per-env sample-count gate is now strictly
    ``sampling_count`` (no separate ratio), aligned with
    ``_battle_overlap_ready`` so the two gates can't disagree."""
    return {
        "win_margin": DEFAULT_MARGIN,
        "win_not_worse_tolerance": DEFAULT_NOT_WORSE_TOLERANCE,
        "win_min_dominant_envs": WIN_MIN_DOMINANT_ENVS,
    }


def _outcome_for_snapshot(result: Any, champion: ChampionRecord) -> Dict[str, Any]:
    """Build the ``outcome`` dict — the *decision detail* the snapshot
    keeps alongside the (flat) rule constants.

    Pre-refactor there was no equivalent on-snapshot record for
    per-decision detail (winner / per-env verdicts / counts) — those
    were either reconstructed from ``scores.scores_by_env`` rows or
    inferred from logs. The new ``config.outcome.*`` paths are
    additive and have no path conflict with the pre-#449 layout.

    ``result`` may be ``None`` on the crash-recovery path, in which
    case minimal identity fields are emitted (counts at 0, per_env
    empty)."""
    out: Dict[str, Any] = {
        "winner": "challenger" if result is not None else "recovery",
        "champion_uid": champion.uid,
        "reason": getattr(result, "reason", "") if result is not None else "",
    }
    if result is None:
        out["dominant_count"] = 0
        out["not_worse_count"] = 0
        out["worse_count"] = 0
        out["per_env"] = []
        return out
    out["dominant_count"] = int(getattr(result, "dominant_count", 0) or 0)
    out["not_worse_count"] = int(getattr(result, "not_worse_count", 0) or 0)
    out["worse_count"] = int(getattr(result, "worse_count", 0) or 0)
    out["per_env"] = [
        _env_comparison_to_dict(e) for e in (getattr(result, "per_env", []) or [])
    ]
    return out


def _comparison_scores_by_env(result: Any, *, role: str) -> Dict[str, Dict[str, Any]]:
    """Build rank-table metadata from a comparator result.

    ``role='challenger'`` carries the old get-rank threshold format:
    score-on-common, lose-below threshold, win-above threshold, and common
    task count. ``role='champion'`` carries the reference mean and count.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for env_result in getattr(result, "per_env", []) or []:
        env = getattr(env_result, "env", None)
        if not env:
            continue
        champion_avg = getattr(env_result, "champion_avg", None)
        champion_basis = champion_avg if isinstance(champion_avg, (int, float)) else 0.0
        if role == "challenger":
            challenger_avg = getattr(env_result, "challenger_avg", None)
            if not isinstance(challenger_avg, (int, float)):
                continue
            margin = float(getattr(env_result, "margin", 0.0) or 0.0)
            tolerance = float(getattr(env_result, "not_worse_tolerance", 0.0) or 0.0)
            common_tasks = int(getattr(env_result, "challenger_n", 0) or 0)
            out[str(env)] = {
                "score": challenger_avg,
                "score_on_common": challenger_avg,
                "sample_count": common_tasks,
                "common_tasks": common_tasks,
                "not_worse_threshold": champion_basis * (1.0 - tolerance),
                "dethrone_threshold": champion_basis + margin,
                "verdict": getattr(env_result, "verdict", ""),
                "reason": getattr(env_result, "reason", ""),
            }
        else:
            if not isinstance(champion_avg, (int, float)):
                continue
            sample_count = int(getattr(env_result, "champion_n", 0) or 0)
            out[str(env)] = {
                "score": champion_avg,
                "sample_count": sample_count,
                "historical_count": sample_count,
            }
    return out
