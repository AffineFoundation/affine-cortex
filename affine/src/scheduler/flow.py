"""
Flow scheduler — replaces the old window-state-machine driver.

One block tick:

  1. Bootstrap task_ids if never refreshed.
  2. Refresh task_ids if 7200 blocks elapsed AND no battle in flight
     (don't disrupt a contest mid-flow).
  3. Sync champion uid from their hotkey; if the hotkey is offline,
     keep the champion identity and use a burn sentinel uid.
  4. No champion because the system is truly uninitialized →
     bootstrap-promote the earliest pending miner (no contest, no Targon).
  4.5. If a battle is in flight and the challenger has been flipped
       to ``is_valid=false`` since ``_start_battle``, tear it down
       immediately — the executor would otherwise keep accumulating
       samples for a hotkey that can never win on chain.
  5. Champion has no Targon workload → deploy it.
  6. Champion samples for the current task_ids not yet full → return
     (executors are filling them).
  7. No in-flight battle → pick next challenger, deploy, record battle.
  7.5. Any scoring env reached ``sampling_count`` overlap AND the
       challenger is already worse on it under the not_worse rule →
       short-circuit LOST, freeing the host without waiting for slower
       envs to finish buffering.
  8. Battle challenger samples not yet full → return.
  9. Both subjects done → run comparator, transition champion (or drop
     challenger), write weights when the champion changes, clear battle.

The state machine is implicit — no ``phase`` / ``status`` enum. The
record shape itself tells us where we are.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Tuple

from affine.core.setup import logger
from affine.database.dao.miners import select_preferred_hotkey_row

from affine.src.scorer.challenger_queue import (
    ChallengerQueue,
    OUTCOME_FAILED,
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
from .champion_mirror import ChampionMirror


# ---- deploy_fn contract: signaling exception -------------------------------


class NoSpareEndpoint(RuntimeError):
    """``deploy_fn`` raises this when no host is free for the role.
    The pre-deploy fill loop treats it as 'stop filling, not a failure'."""


class TransientDeployError(RuntimeError):
    """``deploy_fn`` raises this when the deploy failure is an infra
    transient (SSH transport down, host unreachable, docker daemon
    dead) rather than a miner fault (bad model id, OOM-on-this-model,
    model never becomes ready).

    The flow scheduler treats it as 'retry next tick, do NOT mark
    miner FAILED' — the miner had no chance to fail and shouldn't
    burn their queue entry on a bad host."""


class DeploymentStateInvalidatedError(RuntimeError):
    """``deploy_fn`` raises this when a deployment attempt may have
    invalidated one or more existing deployment ids before failing.

    Example: SSH deploy starts with ``docker rm -f`` on the selected
    endpoint, then ``docker run`` fails. The scheduler must not trust
    any state record still pointing at that endpoint's stable
    deployment id."""

    def __init__(self, message: str, *, deployment_ids: List[str]):
        super().__init__(message)
        self.invalidated_deployment_ids = tuple(
            did for did in deployment_ids if did
        )


# ---- constants (no longer in system_config) --------------------------------


import math
import time

WINDOW_BLOCKS = 7200
"""How often the per-env task_id pool is regenerated."""

DEFAULT_MARGIN = 0.03
"""Per-env additive margin the challenger must clear to be ``dominant``."""

DEFAULT_NOT_WORSE_TOLERANCE = 0.02
"""Multiplicative regression tolerance; challenger must keep >= champion * 0.98."""

WIN_MIN_DOMINANT_ENVS = 1
"""Partial Pareto: at least one env must be dominant; the rest must not regress."""

_PREDEPLOY_PEEK_LIMIT = 32
"""Upper bound on candidates pulled per fill tick. Real terminator is
:exc:`NoSpareEndpoint`."""

ORPHAN_GRACE_SECONDS = 120
"""Minimum age of an ``in_progress`` row, missing from all in-flight
roles, before the reaper terminates it."""

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

ListCurrentMinersFn = Callable[[], Awaitable[List[Dict[str, Any]]]]
"""Return every current row in the miners table, valid or not."""

DeployFn = Callable[[targon_lifecycle.DeployTarget, str],
                    Awaitable[targon_lifecycle.DeployResult]]
TeardownFn = Callable[[Optional[str]], Awaitable[None]]

ListActiveEndpointNamesFn = Callable[[], Awaitable[set]]
"""Return the set of currently-active endpoint names. Used by the
pre-deploy invalidation sweep to drop records whose endpoint has been
removed from ``inference_endpoints`` while the scheduler was down
(``af db set-endpoint --no-active`` or row deletion). Optional —
non-ssh providers don't need it."""


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
        list_current_miners_fn: Optional[ListCurrentMinersFn] = None,
        list_active_endpoint_names_fn: Optional[ListActiveEndpointNamesFn] = None,
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
        self._list_current_miners = list_current_miners_fn or list_valid_miners_fn
        self._list_active_endpoint_names = list_active_endpoint_names_fn
        self._champion_mirror = ChampionMirror()

    # ---- entry point ------------------------------------------------------

    async def tick(self, current_block: int) -> None:
        """Battle phase, then pre-sample phase, then orphan reaper.
        The latter two are guarded so a flaky teardown / DDB scan
        can't block the next tick."""
        await self._tick_main(current_block)
        try:
            await self._predeploy_phase(current_block)
        except Exception as e:
            logger.warning(
                f"FlowScheduler: predeploy phase error "
                f"(non-fatal, will retry next tick): "
                f"{type(e).__name__}: {e}"
            )
        try:
            await self._reap_in_progress_orphans()
        except Exception as e:
            logger.warning(
                f"FlowScheduler: orphan reaper error "
                f"(non-fatal, will retry next tick): "
                f"{type(e).__name__}: {e}"
            )

    async def _tick_main(self, current_block: int) -> None:
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

        # 3. Read champion and reconcile uid from hotkey registration.
        champion = await self._read_synced_champion()

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

        # 4.5. Early-invalidation guard. Monitor can flip is_valid to
        # false on the challenger after ``_start_battle`` claimed them
        # (multi_commit / blacklist / repo-name / anticopy detected
        # late). If we don't act here, the executor will keep sampling
        # an unwinnable hotkey until ``_battle_overlap_ready`` (step 8)
        # is satisfied many minutes from now — pure waste, plus the
        # ⚡ marker on the rank UI confuses operators by suggesting an
        # active contest. Tear down on the first tick we see the miss.
        if battle is not None:
            valid_uids = {
                int(r.get("uid", -1))
                for r in await self._list_valid_miners()
            }
            if battle.challenger.uid not in valid_uids:
                await self._decide_invalidation_lost(champion, battle)
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

        # The comparator sees only envs whose ``enabled_for_scoring`` is
        # true; sampling-only envs accumulate data without affecting any
        # DECIDE path. Read once and share between the 7.5 short-circuit
        # and the step-9 full comparator.
        scoring_envs = await self.state.get_scoring_environments()

        # 7.5. Early-regression short-circuit. Under partial-Pareto a
        # single env that's both fully sampled AND showing a definitive
        # regression is enough to lose — no need to wait for the slower
        # envs to buffer their last samples.
        early = await self._check_early_regression(
            champion, battle.challenger,
            scoring_envs=scoring_envs, task_state=task_state,
        )
        if early is not None:
            regression_env, per_env_data = early
            await self._decide_early_lost(
                champion, battle,
                regression_env=regression_env,
                per_env_data=per_env_data,
                task_state=task_state,
                current_block=current_block,
            )
            return

        # 8. Battle overlap not yet sufficient — wait for both miners to
        # have ≥ sampling_count current-refresh task_ids in common per env.
        if not await self._battle_overlap_ready(
            champion, battle.challenger, envs=envs, task_state=task_state,
        ):
            return

        # 9. Sufficient overlap — run the contest on overlap task_ids only.
        await self._decide(
            champion, battle, scoring_envs, task_state, current_block,
        )

    # ---- helpers ----------------------------------------------------------

    async def _read_synced_champion(self) -> Optional[ChampionRecord]:
        """Return the saved champion, syncing only its current uid.

        The champion is a hotkey/revision/model identity. A hotkey can
        deregister and temporarily have no payable uid, but that does
        not mean the champion disappeared: challengers still have to
        beat the saved model/samples. Persist uid=-1 while offline so
        queue selection does not exclude the old recycled uid, and sync
        back to the current uid when the hotkey registers again.
        """
        champ = await self.state.get_champion()
        if champ is None:
            return None

        current_miners = await self._list_current_miners()
        matches = [
            row for row in current_miners
            if str(row.get("hotkey") or "") == champ.hotkey
        ]
        if matches:
            row = select_preferred_hotkey_row(matches) or matches[0]
            try:
                current_uid = int(row.get("uid"))
            except (TypeError, ValueError):
                current_uid = -1
            if current_uid != champ.uid:
                old_uid = champ.uid
                champ.uid = current_uid
                await self.state.set_champion(champ)
                logger.info(
                    f"FlowScheduler: champion hotkey={champ.hotkey[:10]} "
                    f"uid synced {old_uid} -> {current_uid}"
                )
            return await self._ensure_champion_mirrored(champ)

        if champ.uid != -1:
            old_uid = champ.uid
            champ.uid = -1
            await self.state.set_champion(champ)
            logger.warning(
                f"FlowScheduler: champion hotkey={champ.hotkey[:10]} "
                f"is currently deregistered; retaining champion identity "
                f"and burning weight until it re-registers "
                f"(old_uid={old_uid})"
            )
        return await self._ensure_champion_mirrored(champ)

    async def _ensure_champion_mirrored(
        self, champ: ChampionRecord,
    ) -> ChampionRecord:
        mirrored = await self._champion_mirror.ensure_mirrored(champ)
        if mirrored.model != champ.model or mirrored.revision != champ.revision:
            await self.state.set_champion(mirrored)
        return mirrored

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
        new_champ = await self._champion_mirror.ensure_mirrored(new_champ)
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
            await self._forget_invalidated_deployments_from_error(e)
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

    async def _check_early_regression(
        self,
        champion: ChampionRecord,
        challenger: MinerSnapshot,
        *,
        scoring_envs: Mapping[str, EnvConfig],
        task_state: TaskIdState,
    ) -> Optional[Tuple[str, Dict[str, Dict[str, float]]]]:
        """Detect a battle the challenger has already lost on one env
        before all envs finish buffering.

        Under partial-Pareto (``WIN_MIN_DOMINANT_ENVS=1`` + every env
        must be not_worse) a single env where the challenger sits below
        ``champion_avg * (1 - DEFAULT_NOT_WORSE_TOLERANCE)`` is a
        terminal verdict — no later env can rescue it. Once that env
        has the same overlap depth ``_decide`` would require
        (``cfg.sampling_count``), waiting for slower envs only burns
        GPU time.

        Sampling thresholds match ``_decide`` exactly (no looser early
        rule), so the short-circuit verdict is identical to what the
        full comparator pass would emit moments later — just earlier.

        Returns:
            * ``(env, per_env_data)`` when one env qualifies. ``env`` is
              the trigger env; ``per_env_data`` is the comparator-style
              view ``{env: {count, avg, champion_overlap_avg}}`` for
              every scoring env that already has at least one overlap
              sample, so the rank UI can show as much context as it
              would after a full decide.
            * ``None`` when no env has both enough overlap AND a
              regression.
        """
        per_env_data: Dict[str, Dict[str, float]] = {}
        regression_env: Optional[str] = None

        for env, env_cfg in scoring_envs.items():
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
            if not overlap:
                continue
            champ_overlap_avg = (
                sum(champ_scores[t] for t in overlap) / len(overlap)
            )
            chal_overlap_avg = (
                sum(chal_scores[t] for t in overlap) / len(overlap)
            )
            per_env_data[env] = {
                "count": len(overlap),
                "avg": chal_overlap_avg,
                "champion_overlap_avg": champ_overlap_avg,
            }
            # ``regression_env`` keeps the FIRST trigger env (iteration
            # order is the scoring_envs config order). Any one is
            # sufficient evidence; the others, regression or not, are
            # recorded into per_env_data anyway so the rank table shows
            # the full picture at decide-time.
            #
            # The ``- 1e-9`` slack mirrors the comparator's ENV_WORSE
            # branch (see ``comparator.py``) so the early verdict is
            # bit-identical to what the full pass would emit; without
            # it a ``chal_avg`` sitting exactly on the threshold would
            # tip WORSE here but NOT_WORSE later.
            not_worse_threshold = (
                champ_overlap_avg * (1.0 - DEFAULT_NOT_WORSE_TOLERANCE)
            )
            if (
                regression_env is None
                and len(overlap) >= int(env_cfg.sampling_count)
                and chal_overlap_avg < not_worse_threshold - 1e-9
            ):
                regression_env = env

        if regression_env is None:
            return None
        return regression_env, per_env_data

    async def _decide_early_lost(
        self,
        champion: ChampionRecord,
        battle: BattleRecord,
        *,
        regression_env: str,
        per_env_data: Dict[str, Dict[str, float]],
        task_state: TaskIdState,
        current_block: int,
    ) -> None:
        """LOST handler for the 7.5 short-circuit.

        Mirrors :meth:`_decide`'s challenger-loses branch — teardown,
        ``mark_terminated`` with frozen scores + freeze marker,
        single-instance host cleanup, ``clear_battle`` — but skips the
        full comparator pass because the single regressed env is
        already definitive under partial-Pareto.

        ``per_env_data`` already carries the same
        ``{env: {count, avg, champion_overlap_avg}}`` shape
        ``_final_scores_from_result(role="challenger")`` would produce,
        so the rank UI renders these rows identically to a
        normal-path LOST.
        """
        await self._teardown_record(battle)
        await self.queue.mark_terminated(
            battle.challenger.uid,
            OUTCOME_LOST,
            reason=(
                f"lost_to_champion:{champion.hotkey[:10]}:"
                f"early_regression_in_{regression_env}"
            ),
            hotkey=battle.challenger.hotkey,
            revision=battle.challenger.revision,
            model=battle.challenger.model,
            scores_by_env=per_env_data,
            scores_refresh_block=task_state.refreshed_at_block,
            terminated_at_block=current_block,
        )
        # Single-instance provider: the teardown just emptied the
        # inference host (challenger and champion share the same
        # container under sglang). Champion's deployment_id is now
        # stale — clear it so step 5 re-deploys champion next tick.
        # Symmetric to the ``_decide`` LOST branch.
        if self.cfg.single_instance_provider:
            champion.deployment_id = None
            champion.base_url = None
            champion.deployments = []
            await self.state.set_champion(champion)
        await self.state.clear_battle()
        logger.info(
            f"FlowScheduler: challenger uid={battle.challenger.uid} "
            f"early-LOST on {regression_env} regression vs champion "
            f"uid={champion.uid} — saved waiting on remaining envs"
        )

    async def _decide_invalidation_lost(
        self,
        champion: ChampionRecord,
        battle: BattleRecord,
    ) -> None:
        """Tear down the in-flight battle and terminate the challenger
        because monitor flipped its ``is_valid`` to false mid-contest.

        ``mark_terminated`` is written first so any crash later still
        leaves the lifecycle converged. ``is_valid`` is not durable
        (anticopy ``permanent=False`` can revert), so the lifecycle
        write — not ``is_valid`` — carries the one-shot guarantee.
        """
        logger.warning(
            f"FlowScheduler: challenger uid={battle.challenger.uid} "
            f"became invalid mid-battle; terminating lifecycle"
        )
        await self.queue.mark_terminated(
            battle.challenger.uid,
            OUTCOME_LOST,
            reason=f"invalidated_mid_battle:{battle.challenger.hotkey[:10]}",
            hotkey=battle.challenger.hotkey,
            revision=battle.challenger.revision,
            model=battle.challenger.model,
        )
        await self._teardown_record(battle)
        if self.cfg.single_instance_provider:
            champion.deployment_id = None
            champion.base_url = None
            champion.deployments = []
            await self.state.set_champion(champion)
        await self.state.clear_battle()

    # ---- invariant reaper -------------------------------------------------

    async def _reap_in_progress_orphans(self) -> None:
        """Terminate ``in_progress`` rows that don't match any in-flight
        role (active battle challenger, current champion, predeployed
        challenger) and are older than :data:`ORPHAN_GRACE_SECONDS`.

        Backstop for crash windows in ``pick_next``→``set_battle`` and
        ``_cold_start`` ``set_champion``→``mark_terminated(WON)``.
        """
        battle = await self.state.get_battle()
        champion = await self.state.get_champion()
        predeployed = await self.state.get_predeployed_challengers()
        protected: set = set()
        if battle is not None:
            protected.add(battle.challenger.uid)
        if champion is not None:
            protected.add(champion.uid)
        for record in predeployed:
            protected.add(record.challenger.uid)

        rows = await self.queue.list_in_progress()
        if not rows:
            return
        now = int(time.time())
        for row in rows:
            uid = row.get("uid")
            if uid is None or uid in protected:
                continue
            try:
                claimed_at = int(row.get("challenge_claimed_at") or 0)
            except (TypeError, ValueError):
                claimed_at = 0
            age = now - claimed_at if claimed_at > 0 else now
            if age < ORPHAN_GRACE_SECONDS:
                continue
            try:
                await self.queue.mark_terminated(
                    int(uid),
                    OUTCOME_FAILED,
                    reason=f"claim_orphan_reaped:age={age}s",
                    hotkey=str(row.get("hotkey") or ""),
                    revision=str(row.get("revision") or ""),
                    model=str(row.get("model") or ""),
                )
            except Exception as e:
                logger.warning(
                    f"FlowScheduler: orphan reaper failed to terminate "
                    f"uid={uid}: {type(e).__name__}: {e}"
                )
                continue
            logger.warning(
                f"FlowScheduler: reaped orphan in_progress uid={uid} "
                f"(age={age}s, protected_uids={sorted(protected)})"
            )

    # ---- pre-sample phase -------------------------------------------------

    async def _predeploy_phase(self, current_block: int) -> None:
        """Invalidation sweep, early-loss sweep, fill spare. No-op until
        champion + task_state exist."""
        champion = await self.state.get_champion()
        if champion is None:
            return
        task_state = await self.state.get_task_state()
        if task_state is None:
            return
        await self._predeploy_invalidation_sweep()
        await self._predeploy_early_loss_sweep(
            champion, task_state, current_block,
        )
        await self._predeploy_fill_spare(champion, current_block)

    async def _predeploy_invalidation_sweep(self) -> None:
        records = await self.state.get_predeployed_challengers()
        if not records:
            return
        valid_uids = {
            int(r.get("uid", -1)) for r in await self._list_valid_miners()
        }
        # Champion is read for the post-adoption stale-record check below;
        # ``None`` is fine for cold-start ticks.
        champion = await self.state.get_champion()
        champion_uid = champion.uid if champion is not None else None
        # An endpoint that's been removed from ``inference_endpoints``
        # (``af db set-endpoint --no-active`` or row deletion) is no
        # longer ours to teardown — the cfg may not even exist anymore.
        # Drop the record so the executor stops dispatching to a
        # base_url the operator removed.
        active_endpoints: Optional[set] = None
        if self._list_active_endpoint_names is not None:
            try:
                active_endpoints = await self._list_active_endpoint_names()
            except Exception as e:
                logger.warning(
                    f"FlowScheduler: list_active_endpoint_names failed; "
                    f"skipping endpoint reconciliation this tick: "
                    f"{type(e).__name__}: {e}"
                )
        kept: List[BattleRecord] = []
        changed = False
        for record in records:
            record_endpoints = {
                d.endpoint_name for d in (record.deployments or [])
                if getattr(d, "endpoint_name", None)
            }
            if (
                active_endpoints is not None
                and record_endpoints
                and not (record_endpoints & active_endpoints)
            ):
                logger.warning(
                    f"FlowScheduler: pre-deployed uid="
                    f"{record.challenger.uid} on endpoint(s) "
                    f"{sorted(record_endpoints)} no longer in active "
                    f"inference_endpoints; dropping record without "
                    f"teardown (endpoint is no longer ours to manage)"
                )
                changed = True
                continue
            # A record whose uid is now champion can only be left over
            # from a crash mid-``_adopt_predeployed_if_present``: the
            # champion write committed but the ``set_predeployed_challengers``
            # for the remaining list did not. The spare endpoint stays
            # assigned to that uid, blocking future fills. Tear down +
            # drop here so the fill loop can recover on the next tick.
            if champion_uid is not None and record.challenger.uid == champion_uid:
                logger.warning(
                    f"FlowScheduler: pre-deployed uid="
                    f"{record.challenger.uid} matches current champion; "
                    f"dropping stale crash-recovery record"
                )
                try:
                    await self._teardown_record(record)
                except Exception as e:
                    logger.error(
                        f"FlowScheduler: teardown failed for stale "
                        f"champion-matching pre-deployed uid="
                        f"{record.challenger.uid}: {type(e).__name__}: "
                        f"{e} — keeping for retry"
                    )
                    kept.append(record)
                    continue
                changed = True
                continue
            if record.challenger.uid in valid_uids:
                kept.append(record)
                continue
            logger.warning(
                f"FlowScheduler: pre-deployed uid="
                f"{record.challenger.uid} became invalid; tearing down "
                f"(challenge_status unchanged — invalid_reason is the "
                f"authoritative cause)"
            )
            try:
                await self._teardown_record(record)
            except Exception as e:
                # Keep the record so next tick retries — don't strand
                # the rest of the batch on one teardown's DDB hiccup.
                logger.error(
                    f"FlowScheduler: teardown failed for invalidated "
                    f"pre-deployed uid={record.challenger.uid}: "
                    f"{type(e).__name__}: {e} — keeping for retry"
                )
                kept.append(record)
                continue
            changed = True
        if changed:
            await self.state.set_predeployed_challengers(kept)

    async def _predeploy_early_loss_sweep(
        self,
        champion: ChampionRecord,
        task_state: TaskIdState,
        current_block: int,
    ) -> None:
        records = await self.state.get_predeployed_challengers()
        if not records:
            return
        scoring_envs = await self.state.get_scoring_environments()
        kept: List[BattleRecord] = []
        changed = False
        for record in records:
            early = await self._check_early_regression(
                champion, record.challenger,
                scoring_envs=scoring_envs, task_state=task_state,
            )
            if early is None:
                kept.append(record)
                continue
            regression_env, per_env_data = early
            try:
                await self._decide_predeployed_early_lost(
                    champion, record,
                    regression_env=regression_env,
                    per_env_data=per_env_data,
                    task_state=task_state,
                    current_block=current_block,
                )
            except Exception as e:
                # Same retry policy as the invalidation sweep — one
                # bad record shouldn't strand the rest. teardown +
                # mark_terminated are both idempotent so re-running
                # on the next tick is safe.
                logger.error(
                    f"FlowScheduler: early-loss decide failed for "
                    f"pre-deployed uid={record.challenger.uid}: "
                    f"{type(e).__name__}: {e} — keeping for retry"
                )
                kept.append(record)
                continue
            changed = True
        if changed:
            await self.state.set_predeployed_challengers(kept)

    async def _decide_predeployed_early_lost(
        self,
        champion: ChampionRecord,
        record: BattleRecord,
        *,
        regression_env: str,
        per_env_data: Dict[str, Dict[str, float]],
        task_state: TaskIdState,
        current_block: int,
    ) -> None:
        """Pre-deployed counterpart of :meth:`_decide_early_lost` — no
        battle teardown or champion cleanup, just LOST + free the slot.
        ``reason`` carries ``:predeploy`` for rank-UI provenance."""
        await self._teardown_record(record)
        await self.queue.mark_terminated(
            record.challenger.uid,
            OUTCOME_LOST,
            reason=(
                f"lost_to_champion:{champion.hotkey[:10]}:"
                f"early_regression_in_{regression_env}:predeploy"
            ),
            hotkey=record.challenger.hotkey,
            revision=record.challenger.revision,
            model=record.challenger.model,
            scores_by_env=per_env_data,
            scores_refresh_block=task_state.refreshed_at_block,
            terminated_at_block=current_block,
        )
        logger.info(
            f"FlowScheduler: pre-deployed uid={record.challenger.uid} "
            f"early-LOST on {regression_env} regression vs champion "
            f"uid={champion.uid} — terminated without entering battle"
        )

    async def _predeploy_fill_spare(
        self, champion: ChampionRecord, current_block: int,
    ) -> None:
        """Deploy queued miners on free non-primary endpoints, FIFO.
        Deploy failures leave the miner in queue and advance to the next
        candidate; ``NoSpareEndpoint`` ends the loop cleanly."""
        battle = await self.state.get_battle()
        records = await self.state.get_predeployed_challengers()
        exclude = {p.challenger.uid for p in records}
        if battle is not None:
            exclude.add(battle.challenger.uid)
        candidates = await self.queue.peek_next(
            n=_PREDEPLOY_PEEK_LIMIT,
            champion_uid=champion.uid,
            exclude_uids=exclude,
        )
        for cand in candidates:
            target = targon_lifecycle.DeployTarget(
                uid=cand.uid, hotkey=cand.hotkey,
                model=cand.model, revision=cand.revision,
            )
            try:
                result = await self._deploy(target, "pre_challenger")
            except NoSpareEndpoint:
                return
            except Exception as e:
                await self._forget_invalidated_deployments_from_error(e)
                # Any non-NoSpareEndpoint deploy failure (transient
                # transport, sglang container crash on startup, HF
                # 5xx, ``_wait_ready`` timeout) leaves the miner in
                # queue. Scheduler can't tell a true model fault apart
                # from infra noise; the monitor's ``hf_model_fetch``
                # check is the authoritative signal for "this model
                # is broken" — it flips ``is_valid=false`` and the
                # invalidation sweep tears the record down. Marking
                # FAILED here burns a miner's queue slot on every
                # infra blip.
                kind = (
                    "transport" if isinstance(e, TransientDeployError)
                    else "deploy"
                )
                logger.error(
                    f"FlowScheduler: pre-deploy {kind} error for "
                    f"uid={cand.uid}; leaving miner in queue: "
                    f"{type(e).__name__}: {e}"
                )
                continue
            deployments = _deployments_from_result(result)
            records.append(BattleRecord(
                challenger=MinerSnapshot(
                    uid=cand.uid, hotkey=cand.hotkey,
                    revision=cand.revision, model=cand.model,
                ),
                deployment_id=result.deployment_id,
                base_url=result.base_url,
                started_at_block=current_block,
                deployments=deployments,
            ))
            await self.state.set_predeployed_challengers(records)
            host = deployments[0].endpoint_name if deployments else "?"
            logger.info(
                f"FlowScheduler: pre-deployed uid={cand.uid} on {host} "
                f"(pre-sample slot {len(records)})"
            )

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
        await self._adopt_predeployed_if_present(candidate.uid)
        target = targon_lifecycle.DeployTarget(
            uid=candidate.uid, hotkey=candidate.hotkey,
            model=candidate.model, revision=candidate.revision,
        )
        try:
            result = await self._deploy(target, "challenger")
        except Exception as e:
            await self._forget_invalidated_deployments_from_error(e)
            # Any deploy failure (transient transport, sglang container
            # crash on startup, HF 5xx, ``_wait_ready`` timeout) leaves
            # the miner in queue. ``pick_next`` already flipped the row
            # to ``in_progress``; release it back so the same uid stays
            # re-pickable on the next tick (and may land on a healed
            # host). Pre-sample work, if any, was already torn down by
            # ``_adopt_predeployed_if_present`` — that work is lost,
            # but we'd rather lose pre-samples than a miner's queue
            # entry to an infra blip. The monitor's ``hf_model_fetch``
            # check is the authoritative signal for "model is broken";
            # scheduler doesn't have enough info to judge from a single
            # deploy failure.
            kind = (
                "transport" if isinstance(e, TransientDeployError)
                else "deploy"
            )
            logger.error(
                f"FlowScheduler: challenger {kind} error "
                f"uid={candidate.uid}; releasing claim so miner stays "
                f"re-pickable: {type(e).__name__}: {e}"
            )
            released = await self.queue.release_claim(
                candidate.uid,
                hotkey=candidate.hotkey, revision=candidate.revision,
            )
            if not released:
                logger.warning(
                    f"FlowScheduler: release_claim race on "
                    f"uid={candidate.uid} — row no longer in_progress; "
                    f"leaving as-is"
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

        # Mid-battle invalidation guard, defense-in-depth pair with
        # ``tick`` step 4.5. The early guard catches the cross-tick
        # window (most monitor updates land there), but ``is_valid``
        # can still flip while this tick is running — there are many
        # awaits between step 4.5 and here, each a point where the
        # monitor's DDB update can become visible. Re-check right
        # before the promotion path to keep on-chain weight from ever
        # going to a hotkey that just became invalid.
        valid_uids = {int(r.get("uid", -1)) for r in await self._list_valid_miners()}
        if battle.challenger.uid not in valid_uids:
            await self._decide_invalidation_lost(champion, battle)
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
            new_champion = await self._champion_mirror.ensure_mirrored(
                new_champion
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
            # Disabled until sampling budget grows: too frequent on rapid turnover.
            # await self._refresh_task_ids(current_block, envs)
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

    async def _forget_invalidated_deployments_from_error(
        self, error: Exception,
    ) -> None:
        deployment_ids = getattr(error, "invalidated_deployment_ids", None)
        if not deployment_ids:
            return
        await self._forget_deployments(deployment_ids)

    async def _forget_deployments(self, deployment_ids: Any) -> None:
        ids = {str(did) for did in deployment_ids if did}
        if not ids:
            return

        champion = await self.state.get_champion()
        if champion is not None:
            deployments = list(champion.deployments or [])
            kept_deployments = [
                dep for dep in deployments if dep.deployment_id not in ids
            ]
            changed = len(kept_deployments) != len(deployments)
            if champion.deployment_id in ids:
                champion.deployment_id = None
                champion.base_url = None
                changed = True
            if changed:
                champion.deployments = kept_deployments
                if not kept_deployments and champion.deployment_id is None:
                    champion.base_url = None
                await self.state.set_champion(champion)
                logger.warning(
                    f"FlowScheduler: forgot invalidated champion "
                    f"deployment(s) {sorted(ids)}"
                )

        battle = await self.state.get_battle()
        if battle is not None and (
            battle.deployment_id in ids
            or any(d.deployment_id in ids for d in battle.deployments)
        ):
            released = await self.queue.release_claim(
                battle.challenger.uid,
                hotkey=battle.challenger.hotkey,
                revision=battle.challenger.revision,
            )
            await self.state.clear_battle()
            logger.warning(
                f"FlowScheduler: cleared battle for invalidated "
                f"deployment(s) {sorted(ids)} (claim_released={released})"
            )

        records = await self.state.get_predeployed_challengers()
        if records:
            kept: List[BattleRecord] = []
            changed = False
            for record in records:
                if (
                    record.deployment_id in ids
                    or any(d.deployment_id in ids for d in record.deployments)
                ):
                    changed = True
                    logger.warning(
                        f"FlowScheduler: dropped pre-deployed uid="
                        f"{record.challenger.uid} for invalidated "
                        f"deployment(s) {sorted(ids)}"
                    )
                    continue
                kept.append(record)
            if changed:
                await self.state.set_predeployed_challengers(kept)

    async def _adopt_predeployed_if_present(self, uid: int) -> None:
        """Tear down the pre-sample slot for ``uid`` (if any) so the
        caller can deploy it fresh on the primary."""
        records = await self.state.get_predeployed_challengers()
        if not records:
            return
        remaining: List[BattleRecord] = []
        adopted: Optional[BattleRecord] = None
        for record in records:
            if adopted is None and record.challenger.uid == uid:
                adopted = record
                continue
            remaining.append(record)
        if adopted is None:
            return
        await self._teardown_record(adopted)
        await self.state.set_predeployed_challengers(remaining)
        # Pre-samples are keyed by ``refresh_block``; if the task-id pool
        # refreshed after this miner was pre-deployed, old-refresh samples
        # won't count toward current-refresh overlap and the battle
        # effectively starts from zero pre-sample value. Flag the wasted
        # work so operators can see it.
        task_state = await self.state.get_task_state()
        if (
            task_state is not None
            and adopted.started_at_block < task_state.refreshed_at_block
        ):
            logger.warning(
                f"FlowScheduler: adopted pre-deployed uid={uid} crosses "
                f"task-id refresh boundary (started_at_block="
                f"{adopted.started_at_block} < refreshed_at_block="
                f"{task_state.refreshed_at_block}); pre-samples from "
                f"earlier refresh are not counted toward current-refresh "
                f"overlap"
            )
        logger.info(
            f"FlowScheduler: adopted pre-deployed uid={uid} as active "
            f"challenger (pre-sample slot torn down; pre-samples retained "
            f"via sample_results)"
        )

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
