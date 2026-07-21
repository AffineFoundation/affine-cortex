"""
Flow scheduler — replaces the old window-state-machine driver.

One block tick:

  1. Bootstrap task_ids if never refreshed.
  2. Refresh task_ids if the configured refresh interval elapsed AND no
     battle is in flight (don't disrupt a contest mid-flow).
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
  7.5. For non-system miners, any scoring env reached ``sampling_count``
       overlap AND the challenger is already worse on it under the
       not_worse rule → short-circuit LOST, freeing the host without
       waiting for slower envs to finish buffering. System miners skip
       this so test runs collect full metrics before the final decision.
  8. Battle challenger samples not yet full → return.
  9. Both subjects done → run comparator, transition champion (or drop
     challenger), write weights when the champion changes, clear battle.

The state machine is implicit — no ``phase`` / ``status`` enum. The
record shape itself tells us where we are.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Tuple

from affine.core.setup import logger
from affine.database.dao.miners import select_preferred_hotkey_row
from affine.src.behavior_guard.gate import (
    GateSnapshot,
    read_gate_snapshot,
    record_deployment_fingerprint,
)
from affine.src.behavior_guard.models import (
    VerdictStatus,
    parse_behavior_gate_config,
)

from affine.src.scorer.challenger_queue import (
    ChallengerQueue,
    OUTCOME_LOST,
    OUTCOME_WON,
)
from affine.src.scorer.comparator import (
    ADDITIVE_MARGIN_ENVS,
    DEFAULT_ADDITIVE_MARGIN,
    EnvComparisonConfig,
    WindowComparator,
    not_worse_lower_bound,
)
from affine.src.scorer.sampler import EnvSamplingConfig, WindowSampler
from affine.src.scorer.sampling_thresholds import (
    SAMPLE_BUFFER_RATIO,
    champion_completion_threshold,
)
from affine.src.scorer.token_efficiency import (
    TOKEN_EFFICIENCY_ENV,
    SampleMetric,
    TokenEfficiencyComputation,
    compute_token_efficiency,
    load_token_efficiency_config,
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
    WindowRotationRequest,
)

from . import targon as targon_lifecycle
from .health import DeploymentHealthResult, DeploymentHealthState


# ---- deploy_fn contract: signaling exception -------------------------------


class NoEndpointCapacity(RuntimeError):
    """``deploy_fn`` raises this when no host is free for the role.
    The pre-deploy fill loop treats it as 'stop filling, not a failure'."""


# Compatibility for downstream imports while callers migrate away from the
# old fixed-primary/fixed-spare terminology.
NoSpareEndpoint = NoEndpointCapacity


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


class DeploymentRoleTransitionResult(str, Enum):
    """Outcome of fencing a runtime role change in provider state."""

    UPDATED = "updated"
    RETRYABLE = "retryable"
    STALE = "stale"


SYSTEM_MINER_MIN_UID = 1000
SYSTEM_MINER_HOTKEY_PREFIX = "SYSTEM-"


def _is_system_miner(miner: MinerSnapshot) -> bool:
    return (
        miner.uid >= SYSTEM_MINER_MIN_UID
        or miner.hotkey.startswith(SYSTEM_MINER_HOTKEY_PREFIX)
    )


def _runtime_record_miner(
    record: ChampionRecord | BattleRecord,
) -> MinerSnapshot:
    if isinstance(record, BattleRecord):
        return record.challenger
    return MinerSnapshot(
        uid=record.uid,
        hotkey=record.hotkey,
        revision=record.revision,
        model=record.model,
        model_type=record.model_type,
    )


def _format_cause_chain(exc: BaseException, *, max_depth: int = 8) -> str:
    """Render ``exc``'s ``__cause__`` chain as a `` -> Type: msg`` suffix so
    the real underlying error (docker stderr, sglang argv reject, HF auth,
    ...) lands in the log instead of just the wrapper. Bounded by
    ``max_depth`` and a seen-set: a pathological deep wrap can't blow up the
    log line and a cyclic chain (``a.__cause__ is b`` and back) can't loop
    forever — both truncate to `` -> ...``."""
    parts: List[str] = []
    seen: set = set()
    inner = exc.__cause__
    depth = 0
    while inner is not None and depth < max_depth:
        if id(inner) in seen:
            inner = None
            break
        seen.add(id(inner))
        parts.append(f" -> {type(inner).__name__}: {inner}")
        inner = inner.__cause__
        depth += 1
    if inner is not None:
        parts.append(" -> ...")
    return "".join(parts)


# ---- constants (no longer in system_config) --------------------------------


WINDOW_BLOCKS = 7200
"""Logical scheduler window size for window_id bucketing."""

TASK_POOL_REFRESH_BLOCKS_ENV = "SCHEDULER_TASK_POOL_REFRESH_BLOCKS"
"""Optional env var that controls how often the task_id pool refreshes."""


def _task_pool_refresh_blocks_from_env() -> Optional[int]:
    raw = os.getenv(TASK_POOL_REFRESH_BLOCKS_ENV)
    if raw is None or raw.strip() == "":
        return None
    return _positive_int_config(raw, source=TASK_POOL_REFRESH_BLOCKS_ENV)


def _positive_int_config(value: object, *, source: str) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{source} must be a positive integer, got {value!r}") from e
    if out < 1:
        raise ValueError(f"{source} must be a positive integer, got {value!r}")
    return out

DEFAULT_MARGIN = 0.03
"""Per-env additive margin the challenger must clear to be ``dominant``."""

DEFAULT_NOT_WORSE_TOLERANCE = 0.02
"""Multiplicative regression tolerance; challenger must keep >= champion * 0.98."""

WIN_MIN_DOMINANT_ENVS = 1
"""Partial Pareto: at least one env must be dominant; the rest must not regress."""

_PREDEPLOY_PEEK_LIMIT = 32
"""Upper bound on candidates pulled per fill tick. Real terminator is
:exc:`NoEndpointCapacity`."""

ORPHAN_GRACE_SECONDS = 120
"""Minimum age of an ``in_progress`` row, missing from all in-flight
roles, before the reaper releases it back to the queue."""

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

SampleMetricsReader = Callable[
    [str, str, str, List[int], int, bool],
    Awaitable[Dict[int, SampleMetric]],
]
"""(hotkey, revision, env, task_ids, refresh_block, include_usage) → metrics."""

ListValidMinersFn = Callable[[], Awaitable[List[Dict[str, Any]]]]
"""Return every is_valid=true row in the miners table."""

ListCurrentMinersFn = Callable[[], Awaitable[List[Dict[str, Any]]]]
"""Return every current row in the miners table, valid or not."""

DeployFn = Callable[[targon_lifecycle.DeployTarget, str],
                    Awaitable[targon_lifecycle.DeployResult]]
TeardownFn = Callable[[Optional[str]], Awaitable[None]]
TransitionDeploymentRoleFn = Callable[
    [BattleRecord, str], Awaitable[DeploymentRoleTransitionResult]
]
"""Move an already-running deployment to the requested runtime role.
The result distinguishes a durable update, a transient control-plane failure,
and a permanently stale provider assignment."""

ListActiveEndpointNamesFn = Callable[[], Awaitable[set]]
"""Return the set of currently-active endpoint names. Used by the
pre-deploy invalidation sweep to drop records whose endpoint has been
marked inactive while the scheduler was down. Optional — non-ssh providers
don't need it."""

ListActiveEndpointActivationsFn = Callable[[], Awaitable[Dict[str, int]]]
"""Return currently-active endpoint names mapped to ``activated_at`` unix
seconds. Used by the orphan reaper to tell whether an ``in_progress`` claim
belongs to an older endpoint generation. Optional for providers without
endpoint lifecycle metadata."""

RuntimeDeploymentRecord = ChampionRecord | BattleRecord

DeploymentHealthFn = Callable[
    [RuntimeDeploymentRecord], Awaitable[bool | DeploymentHealthResult]
]
"""Classify whether the active champion or challenger runtime still serves
the saved miner identity. Boolean callbacks remain supported for providers
using the legacy contract."""

DeploymentTransportRepairFn = Callable[
    [RuntimeDeploymentRecord, DeploymentHealthResult], Awaitable[bool]
]
"""Request repair of the transport in front of a healthy model runtime.
Return whether the transport is managed and a repair is pending."""

DeploymentEndpointRepairFn = Callable[
    [RuntimeDeploymentRecord, DeploymentHealthResult], Awaitable[bool]
]
"""Request replacement of an endpoint whose control plane and public path
remain unavailable. Return whether a fenced replacement is pending."""


@dataclass
class FlowConfig:
    window_blocks: int = WINDOW_BLOCKS
    task_pool_refresh_blocks: Optional[int] = None
    scorer_hotkey: str = "scheduler"
    single_instance_provider: bool = False
    """True when each provider endpoint can host only one model at a time.

    SSH/sglang endpoints use this mode. Endpoint positions have no fixed
    role: a pre-deployed challenger is promoted in place, then the previous
    champion's distinct runtime is released for the next queued model. A
    one-endpoint setup still works by replacing the champion deployment.
    Targon-style multi-instance providers leave this False because sibling
    deployments can coexist independently.
    """
    deployment_health_failure_threshold: int = 3
    """Consecutive inconclusive readiness failures required before repair."""
    deployment_unknown_failure_threshold: int = 5
    """Consecutive control-plane failures required before endpoint replacement."""

    def __post_init__(self) -> None:
        if self.task_pool_refresh_blocks is None:
            self.task_pool_refresh_blocks = _task_pool_refresh_blocks_from_env()
        if self.task_pool_refresh_blocks is None:
            self.task_pool_refresh_blocks = self.window_blocks
        self.task_pool_refresh_blocks = _positive_int_config(
            self.task_pool_refresh_blocks,
            source="FlowConfig.task_pool_refresh_blocks",
        )
        self.deployment_health_failure_threshold = _positive_int_config(
            self.deployment_health_failure_threshold,
            source="FlowConfig.deployment_health_failure_threshold",
        )
        self.deployment_unknown_failure_threshold = _positive_int_config(
            self.deployment_unknown_failure_threshold,
            source="FlowConfig.deployment_unknown_failure_threshold",
        )


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
        list_active_endpoint_activations_fn: Optional[
            ListActiveEndpointActivationsFn
        ] = None,
        deployment_health_fn: Optional[DeploymentHealthFn] = None,
        deployment_transport_repair_fn: Optional[
            DeploymentTransportRepairFn
        ] = None,
        deployment_endpoint_repair_fn: Optional[
            DeploymentEndpointRepairFn
        ] = None,
        sample_metrics_reader: Optional[SampleMetricsReader] = None,
        behavior_gate_dao: Any = None,
        transition_deployment_role_fn: Optional[
            TransitionDeploymentRoleFn
        ] = None,
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
        self._sample_metrics_reader = sample_metrics_reader
        self._list_valid_miners = list_valid_miners_fn
        self._list_current_miners = list_current_miners_fn or list_valid_miners_fn
        self._list_active_endpoint_names = list_active_endpoint_names_fn
        self._list_active_endpoint_activations = (
            list_active_endpoint_activations_fn
        )
        self._deployment_health = deployment_health_fn
        self._deployment_transport_repair = deployment_transport_repair_fn
        self._deployment_endpoint_repair = deployment_endpoint_repair_fn
        self._deployment_health_failure: Optional[
            Tuple[str, str, str, DeploymentHealthState, int]
        ] = None
        self._behavior_gate_dao = behavior_gate_dao
        self._transition_deployment_role = transition_deployment_role_fn
        self._last_behavior_gate_log: Dict[str, str] = {}
        self._last_no_endpoint_log_at = 0.0

    # ---- entry point ------------------------------------------------------

    async def tick(self, current_block: int) -> None:
        """Battle phase, then pre-sample phase, then orphan reaper.
        The latter two are guarded so a flaky teardown / DDB scan
        can't block the next tick."""
        # Run this before the main phase so a failed pre-deployed miner cannot
        # be promoted into the active battle before its gate verdict is
        # consumed.
        try:
            await self._predeploy_behavior_gate_sweep()
        except Exception as e:
            logger.warning(
                f"FlowScheduler: predeploy behavior-gate sweep error "
                f"(non-fatal, will retry next tick): "
                f"{type(e).__name__}: {e}"
            )
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
        battle = await self.state.get_battle()
        task_state = await self.state.get_task_state()
        rotation_request = await self.state.get_window_rotation_request()
        if rotation_request is not None:
            applied = await self._apply_window_rotation_request(
                rotation_request,
                battle=battle,
                task_state=task_state,
            )
            if not applied:
                return
            battle = await self.state.get_battle()
            task_state = await self.state.get_task_state()

        if not envs:
            logger.warning("FlowScheduler: no sampling-enabled envs; skipping tick")
            return

        # 1 + 2. Task-id pool refresh (between battles only).
        if task_state is None or (
            battle is None
            and current_block - task_state.refreshed_at_block
            >= self.cfg.task_pool_refresh_blocks
        ):
            task_state = await self._refresh_task_ids(current_block, envs)
            return  # let executors warm up on the new pool

        # 3. Read champion and reconcile uid from hotkey registration.
        champion = await self._read_synced_champion()

        # 4. Cold start.
        if champion is None:
            await self._cold_start(current_block)
            return

        # SSH endpoints host one model each. Once a battle is durable, the
        # old champion runtime is no longer needed: its baseline is already
        # complete and releasing its endpoint lets the next queued model
        # pre-sample. A crash after set_battle but before this cleanup retries
        # here. Do not clear the newly-promoted champion in the separate
        # set_champion(new) -> clear_battle recovery window.
        if (
            battle is not None
            and self.cfg.single_instance_provider
            and not _same_subject(champion, battle.challenger)
        ):
            if not await self._release_champion_runtime_for_battle(
                champion, battle,
            ):
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

        # Reconcile whichever model is actually running. During a battle the
        # challenger owns the single-instance endpoint, so checking only the
        # champion would leave a broken tunnel or exited challenger unnoticed.
        runtime_record: RuntimeDeploymentRecord = battle or champion
        if not await self._recover_unhealthy_runtime_deployment(runtime_record):
            return

        if battle is not None:
            gate_snapshot = await self._active_behavior_gate_snapshot(
                battle, envs,
            )
            if gate_snapshot is not None:
                if gate_snapshot.failed:
                    await self._decide_behavior_gate_lost(
                        champion, battle, gate_snapshot.reason,
                    )
                    return
                if not gate_snapshot.passed:
                    # pending/running/suspected/deferred all keep the model's
                    # lifecycle intact while benchmark fan-out remains zero.
                    return

        # 5. Champion needs inference. During a single-model-endpoint battle the
        # champion may intentionally have no live deployment because its
        # samples were completed before the challenger became active.
        #
        # Single-instance loss-path fast-path: when the just-ended battle
        # was a loss, the champion is unchanged so its samples are still
        # complete (loss doesn't touch sample_results). Skip the ~2-minute
        # champion redeploy and dispatch the next challenger directly; that
        # runtime would otherwise be released again as soon as the battle is
        # durable. When the queue is empty, fall through so an endpoint keeps
        # serving the champion.
        if battle is None and (not champion.deployment_id or not champion.base_url):
            if self.cfg.single_instance_provider and await self._samples_complete(
                MinerSnapshot(uid=champion.uid, hotkey=champion.hotkey,
                              revision=champion.revision, model=champion.model,
                              model_type=champion.model_type),
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
                          revision=champion.revision, model=champion.model,
                          model_type=champion.model_type),
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
        if not _is_system_miner(battle.challenger):
            early = await self._check_early_regression(
                champion, battle.challenger,
                scoring_envs=scoring_envs, task_state=task_state,
            )
            if early is not None:
                regression_env, per_env_data, overlap_task_ids = early
                await self._decide_early_lost(
                    champion, battle,
                    regression_env=regression_env,
                    per_env_data=per_env_data,
                    overlap_task_ids=overlap_task_ids,
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
            return champ

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

        # A single-model endpoint may be empty or hold stale runtime state
        # from before the refresh. Clear the persisted champion URL so step 5
        # reconciles its assignment before any new sampling.
        if self.cfg.single_instance_provider:
            champ = await self.state.get_champion()
            if champ is not None and (
                champ.deployment_id
                or champ.base_url
                or champ.deployments
            ):
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
            model_type=candidate.model_type,
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
        if not await self._deployment_capacity_available("champion"):
            return
        target = targon_lifecycle.DeployTarget(
            uid=champion.uid, hotkey=champion.hotkey,
            model=champion.model, revision=champion.revision,
            model_type=champion.model_type,
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

    async def _recover_unhealthy_runtime_deployment(
        self,
        record: RuntimeDeploymentRecord,
    ) -> bool:
        """Reconcile the champion or challenger runtime currently in use."""
        if self._deployment_health is None:
            return True
        if not record.deployment_id or not record.base_url:
            return True

        miner = _runtime_record_miner(record)
        role = "challenger" if isinstance(record, BattleRecord) else "champion"

        try:
            raw_health = await self._deployment_health(record)
            if isinstance(raw_health, bool):
                health = DeploymentHealthResult(
                    DeploymentHealthState.HEALTHY
                    if raw_health
                    else DeploymentHealthState.UNHEALTHY,
                    reason="legacy_boolean_health_check",
                )
            elif isinstance(raw_health, DeploymentHealthResult):
                health = raw_health
            else:
                raise TypeError(
                    "deployment health callback returned unsupported value "
                    f"{raw_health!r}"
                )
        except Exception as e:
            logger.warning(
                f"FlowScheduler: {role} deployment health check raised "
                f"for uid={miner.uid} deployment_id="
                f"{record.deployment_id!r}; leaving state unchanged "
                f"this tick: {type(e).__name__}: {e}"
            )
            return True

        deployment_id = record.deployment_id
        if health.state is DeploymentHealthState.HEALTHY:
            self._deployment_health_failure = None
            await self._sync_runtime_base_url(
                record,
                health.canonical_base_url,
            )
            return True

        if health.state is DeploymentHealthState.UNHEALTHY:
            self._deployment_health_failure = None
            return await self._recover_confirmed_runtime_failure(
                record,
                reason=health.reason,
            )

        previous = self._deployment_health_failure
        subject = f"{role}:{miner.uid}:{miner.hotkey}:{miner.revision}"
        marker = (subject, deployment_id, health.identity, health.state)
        failure_count = (
            previous[4] + 1
            if previous is not None and previous[:4] == marker
            else 1
        )
        self._deployment_health_failure = (*marker, failure_count)
        threshold = (
            self.cfg.deployment_unknown_failure_threshold
            if health.state is DeploymentHealthState.UNKNOWN
            else self.cfg.deployment_health_failure_threshold
        )
        if failure_count < threshold:
            logger.warning(
                f"FlowScheduler: {role} uid={miner.uid} deployment "
                f"{deployment_id!r} health suspected "
                f"({failure_count}/{threshold}, state={health.state.value}, "
                f"reason={health.reason or '-'}); preserving runtime state"
            )
            return True

        if health.state is DeploymentHealthState.UNKNOWN:
            if self._deployment_endpoint_repair is None:
                logger.error(
                    f"FlowScheduler: {role} uid={miner.uid} deployment "
                    f"{deployment_id!r} control plane remains unavailable "
                    "but no endpoint replacement callback is configured"
                )
                return True
            try:
                repair_pending = await self._deployment_endpoint_repair(
                    record,
                    health,
                )
            except Exception as e:
                logger.warning(
                    f"FlowScheduler: endpoint replacement request failed for "
                    f"{role} uid={miner.uid} deployment_id={deployment_id!r}; "
                    f"preserving runtime state: {type(e).__name__}: {e}"
                )
                return True
            if not repair_pending:
                logger.error(
                    f"FlowScheduler: endpoint replacement was not queued for "
                    f"{role} uid={miner.uid} deployment_id={deployment_id!r}; "
                    "preserving runtime state and rechecking next tick"
                )
                return True
            self._deployment_health_failure = None
            logger.warning(
                f"FlowScheduler: requested fenced endpoint replacement for "
                f"{role} uid={miner.uid} deployment_id={deployment_id!r} "
                f"after {failure_count} consecutive control-plane failures"
            )
            return True

        if health.state is DeploymentHealthState.TRANSPORT_UNHEALTHY:
            if self._deployment_transport_repair is None:
                logger.error(
                    f"FlowScheduler: {role} uid={miner.uid} deployment "
                    f"{deployment_id!r} transport is unhealthy but no repair "
                    "callback is configured; preserving the SGLang runtime"
                )
                return True
            try:
                repair_pending = await self._deployment_transport_repair(
                    record,
                    health,
                )
            except Exception as e:
                logger.warning(
                    f"FlowScheduler: transport repair request failed for "
                    f"{role} uid={miner.uid} deployment_id={deployment_id!r}; "
                    f"preserving runtime state: {type(e).__name__}: {e}"
                )
                return True
            if not repair_pending:
                logger.error(
                    f"FlowScheduler: {role} uid={miner.uid} deployment "
                    f"{deployment_id!r} transport repair was not queued; "
                    "preserving the SGLang runtime and rechecking next tick"
                )
                return True
            self._deployment_health_failure = None
            logger.warning(
                f"FlowScheduler: requested transport repair for {role} "
                f"uid={miner.uid} deployment_id={deployment_id!r} after "
                f"{failure_count} consecutive failures; preserving SGLang state"
            )
            return True

        self._deployment_health_failure = None
        return await self._recover_confirmed_runtime_failure(
            record,
            reason=(
                f"{health.reason or health.state.value}; "
                f"confirmed_after={threshold}"
            ),
        )

    async def _sync_runtime_base_url(
        self,
        record: RuntimeDeploymentRecord,
        canonical_base_url: str,
    ) -> None:
        """Publish a verified endpoint URL to the state executors consume."""
        if not canonical_base_url:
            return

        # The autoscaler may drain this deployment while the network probe is
        # in flight. Re-read state before writing so a late health result
        # cannot resurrect a deployment or battle that has already cleared.
        if isinstance(record, BattleRecord):
            current = await self.state.get_battle()
        else:
            current = await self.state.get_champion()
        if current is None or type(current) is not type(record):
            return
        expected_miner = _runtime_record_miner(record)
        current_miner = _runtime_record_miner(current)
        if (
            current.deployment_id != record.deployment_id
            or current_miner.uid != expected_miner.uid
            or current_miner.hotkey != expected_miner.hotkey
            or current_miner.revision != expected_miner.revision
        ):
            return

        deployments = list(current.deployments or [])
        changed = current.base_url != canonical_base_url
        for deployment in deployments:
            if (
                deployment.deployment_id == current.deployment_id
                or len(deployments) == 1
            ) and deployment.base_url != canonical_base_url:
                deployment.base_url = canonical_base_url
                changed = True
        if not changed:
            return
        current.base_url = canonical_base_url
        current.deployments = deployments
        if isinstance(current, BattleRecord):
            await self.state.set_battle(current)
        else:
            await self.state.set_champion(current)
        miner = _runtime_record_miner(current)
        logger.info(
            f"FlowScheduler: synchronized runtime URL for uid={miner.uid} "
            f"deployment_id={current.deployment_id!r} to {canonical_base_url!r}"
        )

    async def _recover_confirmed_runtime_failure(
        self,
        record: RuntimeDeploymentRecord,
        *,
        reason: str,
    ) -> bool:
        if isinstance(record, ChampionRecord):
            return await self._redeploy_unhealthy_champion(record, reason=reason)

        try:
            await self._teardown_record(record)
        except Exception as e:
            logger.warning(
                f"FlowScheduler: challenger uid={record.challenger.uid} "
                f"runtime teardown failed during infra retry: "
                f"{type(e).__name__}: {e}"
            )
        released = await self.queue.release_claim(
            record.challenger.uid,
            hotkey=record.challenger.hotkey,
            revision=record.challenger.revision,
        )
        await self.state.clear_battle()
        logger.warning(
            f"FlowScheduler: challenger uid={record.challenger.uid} runtime "
            f"is unhealthy; released it for an infra retry "
            f"(claim_released={released}, reason={reason or '-'})"
        )
        return False

    async def _redeploy_unhealthy_champion(
        self,
        champion: ChampionRecord,
        *,
        reason: str,
    ) -> bool:
        """Clear a confirmed stale runtime and run the normal deploy path."""

        stale_deployment_id = champion.deployment_id
        stale_base_url = champion.base_url
        champion.deployment_id = None
        champion.base_url = None
        champion.deployments = []
        await self.state.set_champion(champion)
        logger.warning(
            f"FlowScheduler: champion uid={champion.uid} deployment "
            f"{stale_deployment_id!r} at {stale_base_url!r} is unhealthy; "
            f"cleared stale state and redeploying (reason={reason or '-'})"
        )
        await self._deploy_champion(champion)
        return False

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
            if not env_cfg.enabled_for_scoring:
                continue  # shadow-run env: samples accumulate, DECIDE ignores
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
            if not env_cfg.enabled_for_scoring:
                continue  # shadow-run env: not part of DECIDE, no overlap req
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
              view ``{env: {count, avg, champion_overlap_avg}}`` and
              ``overlap_task_ids`` is ``{env: [task_id, ...]}`` for every
              scoring env that already has at least one overlap sample.
            * ``None`` when no env has both enough overlap AND a
              regression.
        """
        per_env_data: Dict[str, Dict[str, float]] = {}
        overlap_task_ids: Dict[str, List[int]] = {}
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
            overlap_ids = sorted(set(champ_scores) & set(chal_scores))
            if not overlap_ids:
                continue
            champ_overlap_avg = (
                sum(champ_scores[t] for t in overlap_ids) / len(overlap_ids)
            )
            chal_overlap_avg = (
                sum(chal_scores[t] for t in overlap_ids) / len(overlap_ids)
            )
            overlap_task_ids[env] = overlap_ids
            per_env_data[env] = {
                "count": len(overlap_ids),
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
            not_worse_threshold = not_worse_lower_bound(
                champ_overlap_avg, env,
                tolerance=DEFAULT_NOT_WORSE_TOLERANCE,
                additive_margin=DEFAULT_ADDITIVE_MARGIN,
            )
            if (
                regression_env is None
                and len(overlap_ids) >= int(env_cfg.sampling_count)
                and chal_overlap_avg < not_worse_threshold - 1e-9
            ):
                regression_env = env

        if regression_env is None:
            return None
        return regression_env, per_env_data, overlap_task_ids

    async def _sampling_only_freeze_scores(
        self,
        *,
        subject_hotkey: str,
        subject_revision: str,
        basis_hotkey: str,
        basis_revision: str,
        task_state: TaskIdState,
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
        """Freeze sampling-only envs onto a terminating miner.

        Sampling-only envs (``enabled_for_sampling=true,
        enabled_for_scoring=false`` — distill-v2) never enter the
        comparator, so ``_final_scores_from_result`` / the early-loss
        ``per_env_data`` omit them. The blind ``scores_by_env`` overwrite
        at termination would then drop the live distill entry
        :class:`LiveScoresMonitor` had parked on the row, leaving the env
        column blank for every terminated miner in ``af get-rank``.

        Reading the subject's samples here and merging them into the
        frozen payload keeps these envs visible after the live cache
        forgets the row — the same guarantee scoring envs already get.
        The subject's own full-set avg is frozen (matching the live
        display's basis, so a battling miner's number doesn't jump when
        it terminates); ``champion_overlap_avg`` is added as the
        threshold basis when a (subject ∩ basis) overlap exists.

        Returns ``(subject_scores, basis_scores)`` in the same shapes the
        scoring-env freeze produces (``{env: {count, avg,
        champion_overlap_avg?}}`` and ``{env: {count, avg}}``).
        """
        all_envs = await self.state.get_environments()
        scoring_envs = await self.state.get_scoring_environments()
        sampling_only = [e for e in all_envs if e not in scoring_envs]
        if not sampling_only:
            return {}, {}

        subject_out: Dict[str, Dict[str, float]] = {}
        basis_out: Dict[str, Dict[str, float]] = {}
        for env in sampling_only:
            tasks = task_state.task_ids.get(env, [])
            if not tasks:
                continue
            subj = await self._scores_reader(
                subject_hotkey, subject_revision, env, tasks,
                task_state.refreshed_at_block,
            )
            if not subj:
                continue
            entry: Dict[str, float] = {
                "count": len(subj),
                "avg": sum(subj.values()) / len(subj),
            }
            basis = await self._scores_reader(
                basis_hotkey, basis_revision, env, tasks,
                task_state.refreshed_at_block,
            )
            overlap = sorted(set(subj) & set(basis))
            if overlap:
                basis_avg = sum(basis[t] for t in overlap) / len(overlap)
                entry["champion_overlap_avg"] = basis_avg
                basis_out[env] = {"count": len(overlap), "avg": basis_avg}
            subject_out[env] = entry
        return subject_out, basis_out

    async def _decide_early_lost(
        self,
        champion: ChampionRecord,
        battle: BattleRecord,
        *,
        regression_env: str,
        per_env_data: Dict[str, Dict[str, float]],
        overlap_task_ids: Dict[str, List[int]],
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
        chal_so, chal_so_basis = await self._sampling_only_freeze_scores(
            subject_hotkey=battle.challenger.hotkey,
            subject_revision=battle.challenger.revision,
            basis_hotkey=champion.hotkey,
            basis_revision=champion.revision,
            task_state=task_state,
        )
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
            scores_by_env={**chal_so, **per_env_data},
            opponent_scores_by_env={
                **chal_so_basis,
                **_opponent_scores_from_early_lost(per_env_data),
            },
            battle_task_ids=overlap_task_ids,
            scores_refresh_block=task_state.refreshed_at_block,
            terminated_at_block=current_block,
        )
        # The champion runtime is normally released at battle start. Keep the
        # persisted state empty here as an idempotent recovery safeguard.
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

    async def _active_behavior_gate_snapshot(
        self,
        battle: BattleRecord,
        envs: Mapping[str, EnvConfig],
    ) -> Optional[GateSnapshot]:
        config = parse_behavior_gate_config(
            await self.state.get_behavior_gate_config()
        )
        if not config.enforces or not any(
            config.gates_environment(env) for env in envs
        ):
            return None
        if self._behavior_gate_dao is None:
            snapshot = GateSnapshot(
                status=VerdictStatus.PENDING,
                reason="behavior_gate_dao_unavailable",
                deployment_fingerprint=record_deployment_fingerprint(
                    battle, config,
                ),
            )
        else:
            try:
                snapshot = await read_gate_snapshot(
                    self._behavior_gate_dao, battle, config,
                )
            except Exception as exc:
                # A control-plane read error blocks progress but can never
                # turn into a model loss.
                snapshot = GateSnapshot(
                    status=VerdictStatus.PENDING,
                    reason=f"behavior_gate_read_error:{type(exc).__name__}",
                    deployment_fingerprint=record_deployment_fingerprint(
                        battle, config,
                    ),
                )
        self._log_behavior_gate_snapshot(battle, snapshot)
        return snapshot

    async def _seal_behavior_gate_for_promotion(
        self,
        battle: BattleRecord,
        envs: Mapping[str, EnvConfig],
    ) -> Tuple[bool, Optional[GateSnapshot]]:
        """Atomically fence a passed gate immediately before promotion.

        A plain strongly-consistent read cannot close the race with a runtime
        invariant failure: the verdict can flip after the read but before the
        challenger becomes the canonical champion.  ``seal_for_promotion``
        performs that final passed->sealed transition conditionally in DDB.

        ``False`` and transport errors are never interpreted as a model loss.
        They are followed by a strong verdict read so only a confirmed FAILED
        row loses; every other state pauses promotion for a later tick.
        """
        config = parse_behavior_gate_config(
            await self.state.get_behavior_gate_config()
        )
        if not config.enforces or not any(
            config.gates_environment(env) for env in envs
        ):
            return True, None

        fingerprint = record_deployment_fingerprint(battle, config)
        if self._behavior_gate_dao is None:
            snapshot = GateSnapshot(
                status=VerdictStatus.PENDING,
                reason="behavior_gate_dao_unavailable",
                deployment_fingerprint=fingerprint,
            )
            self._log_behavior_gate_snapshot(battle, snapshot)
            return False, snapshot

        try:
            sealed = await self._behavior_gate_dao.seal_for_promotion(
                battle.challenger.hotkey,
                battle.challenger.revision,
                config.policy_version,
                fingerprint,
            )
        except Exception as exc:
            sealed = False
            logger.warning(
                "FlowScheduler: behavior gate promotion seal error "
                f"uid={battle.challenger.uid}: {type(exc).__name__}"
            )

        if sealed:
            snapshot = GateSnapshot(
                status=VerdictStatus.PASSED,
                reason="promotion_sealed",
                deployment_fingerprint=fingerprint,
            )
            self._log_behavior_gate_snapshot(battle, snapshot)
            return True, snapshot

        try:
            snapshot = await read_gate_snapshot(
                self._behavior_gate_dao, battle, config,
            )
        except Exception as exc:
            snapshot = GateSnapshot(
                status=VerdictStatus.PENDING,
                reason=f"behavior_gate_read_error:{type(exc).__name__}",
                deployment_fingerprint=fingerprint,
            )
        self._log_behavior_gate_snapshot(battle, snapshot)
        return False, snapshot

    async def _clear_champion_deployment_state(
        self,
        champion: ChampionRecord,
    ) -> None:
        """Forget a champion workload immediately after it is torn down."""
        champion.deployment_id = None
        champion.base_url = None
        champion.deployments = []
        await self.state.set_champion(champion)

    async def _rollback_failed_promotion_recovery(
        self,
        promoted: ChampionRecord,
        battle: BattleRecord,
        *,
        reason: str,
        current_block: int,
    ) -> None:
        """Undo an unsealed promotion after a durable runtime failure.

        This handles the legacy/crash shape where ``set_champion(new)`` was
        persisted before the gate was sealed.  Teardown comes first so any
        transient failure leaves the battle marker intact and retryable.
        """
        logger.warning(
            "FlowScheduler: rolling back failed unsealed promotion "
            f"uid={promoted.uid} reason={reason}"
        )
        await self._teardown_record(battle)
        await self.queue.mark_terminated(
            promoted.uid,
            OUTCOME_LOST,
            reason=f"model_behavior:{reason}:promotion_recovery",
            hotkey=promoted.hotkey,
            revision=promoted.revision,
            model=promoted.model,
        )

        previous = battle.previous_champion
        previous_ready = (
            previous is not None
            and previous.uid != promoted.uid
            and bool(previous.hotkey)
            and bool(previous.revision)
        )
        if previous_ready:
            assert previous is not None
            restored = ChampionRecord(
                uid=previous.uid,
                hotkey=previous.hotkey,
                revision=previous.revision,
                model=previous.model,
                model_type=previous.model_type,
                # The old workload was torn down before the seal attempt.
                # Step 5 will deploy it again from this cleared state.
                deployment_id=None,
                base_url=None,
                deployments=[],
                since_block=current_block,
            )
            await self.queue.mark_terminated(
                previous.uid,
                OUTCOME_WON,
                hotkey=previous.hotkey,
                revision=previous.revision,
                model=previous.model,
            )
            # Write the rollback weight before changing canonical state.  If
            # the external write fails, champion==challenger still selects
            # this recovery branch on the next tick and retries it.
            await self._write_weights(restored, current_block, None)
            await self.state.set_champion(restored)
        else:
            logger.warning(
                "FlowScheduler: failed promotion recovery has no usable "
                "previous_champion; clearing canonical champion safely"
            )
            await self.state.clear_champion()

        await self.state.clear_battle()

    def _log_behavior_gate_snapshot(
        self,
        record: BattleRecord,
        snapshot: GateSnapshot,
    ) -> None:
        key = snapshot.deployment_fingerprint
        marker = f"{snapshot.status.value}:{snapshot.reason}"
        if self._last_behavior_gate_log.get(key) == marker:
            return
        self._last_behavior_gate_log[key] = marker
        log = logger.info if snapshot.passed else logger.warning
        log(
            f"FlowScheduler: behavior gate uid={record.challenger.uid} "
            f"status={snapshot.status.value} reason={snapshot.reason}"
        )

    async def _decide_behavior_gate_lost(
        self,
        champion: ChampionRecord,
        battle: BattleRecord,
        reason: str,
    ) -> None:
        """Consume a confirmed strong behavior violation as LOST."""
        logger.warning(
            f"FlowScheduler: challenger uid={battle.challenger.uid} "
            f"failed behavior preflight ({reason}); terminating lifecycle"
        )
        # Teardown first so a transient teardown error leaves the challenger
        # IN_PROGRESS with its battle record intact.  The next tick then reads
        # the same final gate verdict and retries without overwriting the
        # model-behavior termination reason through the invalidation path.
        await self._teardown_record(battle)
        await self.queue.mark_terminated(
            battle.challenger.uid,
            OUTCOME_LOST,
            reason=f"model_behavior:{reason}",
            hotkey=battle.challenger.hotkey,
            revision=battle.challenger.revision,
            model=battle.challenger.model,
        )
        if self.cfg.single_instance_provider:
            champion.deployment_id = None
            champion.base_url = None
            champion.deployments = []
            await self.state.set_champion(champion)
        await self.state.clear_battle()

    # ---- invariant reaper -------------------------------------------------

    async def _reap_in_progress_orphans(self) -> None:
        """Release ``in_progress`` rows that don't match any in-flight
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
        endpoint_activations: Optional[Dict[str, int]] = None
        if self._list_active_endpoint_activations is not None:
            try:
                endpoint_activations = (
                    await self._list_active_endpoint_activations()
                )
            except Exception as e:
                logger.warning(
                    f"FlowScheduler: active endpoint lifecycle lookup "
                    f"failed; orphan reaper will use claim age only this "
                    f"tick: {type(e).__name__}: {e}"
                )
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
            release_reason = self._orphan_release_reason(
                claimed_at=claimed_at,
                endpoint_activations=endpoint_activations,
            )
            if release_reason == "stale_claim_no_endpoint_snapshot":
                logger.warning(
                    f"FlowScheduler: skipped orphan in_progress uid={uid}; "
                    f"active endpoint lifecycle snapshot unavailable "
                    f"(age={age}s, protected_uids={sorted(protected)})"
                )
                continue
            try:
                released = await self.queue.release_claim(
                    int(uid),
                    hotkey=str(row.get("hotkey") or ""),
                    revision=str(row.get("revision") or ""),
                )
            except Exception as e:
                logger.warning(
                    f"FlowScheduler: orphan reaper failed to release "
                    f"uid={uid}: {type(e).__name__}: {e}"
                )
                continue
            if not released:
                logger.warning(
                    f"FlowScheduler: orphan reaper lost release race "
                    f"uid={uid} (age={age}s, "
                    f"protected_uids={sorted(protected)})"
                )
                continue
            logger.warning(
                f"FlowScheduler: released orphan in_progress uid={uid} "
                f"(age={age}s, reason={release_reason}, "
                f"protected_uids={sorted(protected)})"
            )

    @staticmethod
    def _orphan_release_reason(
        *,
        claimed_at: int,
        endpoint_activations: Optional[Dict[str, int]],
    ) -> str:
        if endpoint_activations is None:
            return "stale_claim_no_endpoint_snapshot"
        if not endpoint_activations:
            return "no_active_endpoint"
        latest = max(int(ts or 0) for ts in endpoint_activations.values())
        if claimed_at > 0 and latest > claimed_at:
            return f"endpoint_reactivated_after_claim:activated_at={latest}"
        return "stale_claim_endpoint_stable"

    # ---- pre-sample phase -------------------------------------------------

    async def _predeploy_behavior_gate_sweep(self) -> None:
        """Terminate pre-deployed records with a confirmed failed gate.

        Pending, suspected, and infrastructure-deferred rows stay deployed so
        the independent coordinator can retry.  They still dispatch no gated
        benchmark work in executor enforce mode.
        """
        records = await self.state.get_predeployed_challengers()
        if not records:
            return
        config = parse_behavior_gate_config(
            await self.state.get_behavior_gate_config()
        )
        envs = await self.state.get_environments()
        if not config.enforces or not any(
            config.gates_environment(env) for env in envs
        ):
            return

        battle = await self.state.get_battle()
        champion = await self.state.get_champion()
        kept: List[BattleRecord] = []
        changed = False
        for record in records:
            # set_battle is intentionally committed before the pre-list is
            # pruned during in-place promotion. A crash in that window leaves
            # two references to one deployment. Let the active-battle path
            # consume its gate verdict; tearing down via the pre-list would
            # kill the live battle (or its newly-promoted champion).
            if (
                battle is not None
                and _same_subject(record.challenger, battle.challenger)
                and _records_share_deployment(record, battle)
            ) or (
                champion is not None
                and _same_subject(record.challenger, champion)
                and _records_share_deployment(record, champion)
            ):
                kept.append(record)
                continue
            if self._behavior_gate_dao is None:
                kept.append(record)
                continue
            try:
                snapshot = await read_gate_snapshot(
                    self._behavior_gate_dao, record, config,
                )
            except Exception as exc:
                logger.warning(
                    f"FlowScheduler: behavior-gate read failed for "
                    f"pre-deployed uid={record.challenger.uid}; keeping "
                    f"blocked for retry: {type(exc).__name__}"
                )
                kept.append(record)
                continue
            self._log_behavior_gate_snapshot(record, snapshot)
            if not snapshot.failed:
                kept.append(record)
                continue

            try:
                # Keep the queue row IN_PROGRESS until teardown succeeds so a
                # transient provider error can be retried without replacing
                # the durable model-behavior reason via orphan recovery.
                await self._teardown_record(record)
                await self.queue.mark_terminated(
                    record.challenger.uid,
                    OUTCOME_LOST,
                    reason=f"model_behavior:{snapshot.reason}:predeploy",
                    hotkey=record.challenger.hotkey,
                    revision=record.challenger.revision,
                    model=record.challenger.model,
                )
            except Exception as exc:
                logger.error(
                    f"FlowScheduler: failed to consume behavior-gate loss "
                    f"for pre-deployed uid={record.challenger.uid}: "
                    f"{type(exc).__name__}: {exc} — keeping for retry"
                )
                kept.append(record)
                continue
            changed = True
            logger.warning(
                f"FlowScheduler: pre-deployed uid={record.challenger.uid} "
                f"LOST by behavior preflight ({snapshot.reason})"
            )
        if changed:
            await self.state.set_predeployed_challengers(kept)

    async def _predeploy_phase(self, current_block: int) -> None:
        """Invalidation sweep, early-loss sweep, fill free endpoints. No-op until
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
        await self._predeploy_fill_available(champion, current_block)

    async def _predeploy_invalidation_sweep(self) -> None:
        records = await self.state.get_predeployed_challengers()
        if not records:
            return
        valid_uids = {
            int(r.get("uid", -1)) for r in await self._list_valid_miners()
        }
        # Current-role records are read for crash recovery below; ``None`` is
        # fine for cold-start ticks.
        champion = await self.state.get_champion()
        battle = await self.state.get_battle()
        # An endpoint marked inactive is no longer ours to teardown.
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
            # Promotion writes current_battle before pruning the pre-list so
            # the running deployment is never absent from scheduler state.
            # A crash between those writes leaves a duplicate reference. Drop
            # it without teardown when both records point to the same runtime;
            # otherwise remove the genuinely stale, distinct pre-deployment.
            if (
                battle is not None
                and _same_subject(record.challenger, battle.challenger)
            ):
                if _records_share_deployment(record, battle):
                    logger.warning(
                        f"FlowScheduler: dropping duplicate pre-deployed "
                        f"record for active battle uid="
                        f"{record.challenger.uid} without teardown"
                    )
                    changed = True
                    continue
                logger.warning(
                    f"FlowScheduler: pre-deployed uid="
                    f"{record.challenger.uid} duplicates the active subject "
                    f"on a different deployment; tearing down stale runtime"
                )
                try:
                    await self._teardown_record(record)
                except Exception as e:
                    logger.error(
                        f"FlowScheduler: teardown failed for stale active-"
                        f"subject pre-deployment uid="
                        f"{record.challenger.uid}: {type(e).__name__}: "
                        f"{e} — keeping for retry"
                    )
                    kept.append(record)
                    continue
                changed = True
                continue

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
            # A matching champion record can remain after a crash during
            # promotion cleanup. In-place promotion means it may be the same
            # deployment now owned by the champion, in which case teardown
            # would kill valid inference. Legacy/distinct leftovers are still
            # safe to tear down.
            if (
                champion is not None
                and _same_subject(record.challenger, champion)
            ):
                shared = _records_share_deployment(record, champion)
                logger.warning(
                    f"FlowScheduler: pre-deployed uid="
                    f"{record.challenger.uid} matches current champion; "
                    f"dropping stale crash-recovery record"
                    + (" without teardown" if shared else "")
                )
                if not shared:
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
            if _is_system_miner(record.challenger):
                kept.append(record)
                continue
            early = await self._check_early_regression(
                champion, record.challenger,
                scoring_envs=scoring_envs, task_state=task_state,
            )
            if early is None:
                kept.append(record)
                continue
            regression_env, per_env_data, overlap_task_ids = early
            try:
                await self._decide_predeployed_early_lost(
                    champion, record,
                    regression_env=regression_env,
                    per_env_data=per_env_data,
                    overlap_task_ids=overlap_task_ids,
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
        overlap_task_ids: Dict[str, List[int]],
        task_state: TaskIdState,
        current_block: int,
    ) -> None:
        """Pre-deployed counterpart of :meth:`_decide_early_lost` — no
        battle teardown or champion cleanup, just LOST + free the slot.
        ``reason`` carries ``:predeploy`` for rank-UI provenance."""
        await self._teardown_record(record)
        chal_so, chal_so_basis = await self._sampling_only_freeze_scores(
            subject_hotkey=record.challenger.hotkey,
            subject_revision=record.challenger.revision,
            basis_hotkey=champion.hotkey,
            basis_revision=champion.revision,
            task_state=task_state,
        )
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
            scores_by_env={**chal_so, **per_env_data},
            opponent_scores_by_env={
                **chal_so_basis,
                **_opponent_scores_from_early_lost(per_env_data),
            },
            battle_task_ids=overlap_task_ids,
            scores_refresh_block=task_state.refreshed_at_block,
            terminated_at_block=current_block,
        )
        logger.info(
            f"FlowScheduler: pre-deployed uid={record.challenger.uid} "
            f"early-LOST on {regression_env} regression vs champion "
            f"uid={champion.uid} — terminated without entering battle"
        )

    async def _predeploy_fill_available(
        self, champion: ChampionRecord, current_block: int,
    ) -> None:
        """Deploy queued miners on any free endpoint, FIFO.

        Before the first battle, a champion without a runtime must deploy
        first. Otherwise a one-endpoint installation could let a pre-
        challenger consume its only machine and deadlock champion sampling.
        During a battle the champion runtime is intentionally absent and the
        endpoint it released is available for the next pre-challenger.

        Deploy failures leave the miner in queue and advance to the next
        candidate; ``NoEndpointCapacity`` ends the loop cleanly."""
        if not await self._deployment_capacity_available("predeploy"):
            return
        battle = await self.state.get_battle()
        if self.cfg.single_instance_provider:
            champion_has_runtime = bool(
                champion.deployment_id
                or champion.base_url
                or champion.deployments
            )
            if battle is None and not champion_has_runtime:
                return
            if battle is not None and champion_has_runtime:
                # Battle state is durable but old-runtime cleanup has not
                # converged. Do not reuse an endpoint whose stable SSH
                # deployment id may still be retried by that cleanup.
                return
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
                model_type=cand.model_type,
            )
            try:
                result = await self._deploy(target, "pre_challenger")
            except NoEndpointCapacity:
                return
            except Exception as e:
                await self._forget_invalidated_deployments_from_error(e)
                # Any non-capacity deploy failure (transient
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
                # Append e.__cause__ chain so the actual underlying error
                # (docker stderr, sglang argv reject, HF auth, ...) lands
                # in the log instead of just the DeploymentStateInvalidatedError
                # wrapper. Otherwise root-causing a consistently-failing
                # candidate requires repro outside the scheduler.
                logger.error(
                    f"FlowScheduler: pre-deploy {kind} error for "
                    f"uid={cand.uid}; leaving miner in queue: "
                    f"{type(e).__name__}: {e}{_format_cause_chain(e)}"
                )
                continue
            deployments = _deployments_from_result(result)
            records.append(BattleRecord(
                challenger=MinerSnapshot(
                    uid=cand.uid, hotkey=cand.hotkey,
                    revision=cand.revision, model=cand.model,
                    model_type=cand.model_type,
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
        """Pick the next pending miner and make its deployment active."""
        if not await self._deployment_capacity_available("challenger"):
            return
        candidate = await self.queue.pick_next(
            window_id=current_block // self.cfg.window_blocks,
            champion_uid=champion.uid,
        )
        if candidate is None:
            return  # idle, no challengers
        records = await self.state.get_predeployed_challengers()
        adopted: Optional[BattleRecord] = None
        remaining: List[BattleRecord] = []
        for record in records:
            if adopted is None and _record_matches_candidate(record, candidate):
                adopted = record
                continue
            remaining.append(record)

        if adopted is not None:
            transition = await self._transition_runtime_role(
                adopted, "challenger",
            )
            if transition is DeploymentRoleTransitionResult.RETRYABLE:
                released = await self.queue.release_claim(
                    candidate.uid,
                    hotkey=candidate.hotkey,
                    revision=candidate.revision,
                )
                logger.warning(
                    f"FlowScheduler: pre-deployment promotion temporarily "
                    f"failed for uid={candidate.uid}; retaining runtime and "
                    f"releasing claim={released} for retry"
                )
                return
            if transition is DeploymentRoleTransitionResult.STALE:
                # The provider assignment has moved to another owner.
                # Drop only the stale state reference; tearing it down by
                # endpoint-stable deployment id could kill that new owner.
                try:
                    await self.state.set_predeployed_challengers(remaining)
                except Exception as e:
                    logger.error(
                        f"FlowScheduler: failed to drop stale pre-"
                        f"deployment uid={candidate.uid}: "
                        f"{type(e).__name__}: {e}"
                    )
                released = await self.queue.release_claim(
                    candidate.uid,
                    hotkey=candidate.hotkey,
                    revision=candidate.revision,
                )
                logger.warning(
                    f"FlowScheduler: pre-deployed uid={candidate.uid} "
                    f"no longer owns its provider assignment; dropped "
                    f"stale record and released claim={released}"
                )
                return

            battle = BattleRecord(
                challenger=adopted.challenger,
                deployment_id=adopted.deployment_id,
                base_url=adopted.base_url,
                started_at_block=current_block,
                deployments=list(adopted.deployments),
                previous_champion=_champion_snapshot(champion),
            )
        else:
            target = targon_lifecycle.DeployTarget(
                uid=candidate.uid, hotkey=candidate.hotkey,
                model=candidate.model, revision=candidate.revision,
                model_type=candidate.model_type,
            )
            try:
                result = await self._deploy(target, "challenger")
            except Exception as e:
                await self._forget_invalidated_deployments_from_error(e)
                # Any deploy failure leaves the miner eligible for a retry.
                # The monitor's model checks, not a single provider failure,
                # are authoritative for deciding whether the model is broken.
                kind = (
                    "transport" if isinstance(e, TransientDeployError)
                    else "deploy"
                )
                logger.error(
                    f"FlowScheduler: challenger {kind} error "
                    f"uid={candidate.uid}; releasing claim so miner stays "
                    f"re-pickable: {type(e).__name__}: "
                    f"{e}{_format_cause_chain(e)}"
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
            battle = BattleRecord(
                challenger=MinerSnapshot(
                    uid=candidate.uid, hotkey=candidate.hotkey,
                    revision=candidate.revision, model=candidate.model,
                    model_type=candidate.model_type,
                ),
                deployment_id=result.deployment_id,
                base_url=result.base_url,
                started_at_block=current_block,
                deployments=_deployments_from_result(result),
                previous_champion=_champion_snapshot(champion),
            )

        # Persist the active role first. If pruning the pre-list fails, both
        # references temporarily point to one deployment; executor launch
        # dedupe and the invalidation sweep make that crash window harmless.
        try:
            await self.state.set_battle(battle)
        except Exception as e:
            released = await self.queue.release_claim(
                candidate.uid,
                hotkey=candidate.hotkey,
                revision=candidate.revision,
            )
            logger.error(
                f"FlowScheduler: failed to persist battle uid="
                f"{candidate.uid}; released claim={released} for retry: "
                f"{type(e).__name__}: {e}"
            )
            return
        if adopted is not None:
            try:
                await self.state.set_predeployed_challengers(remaining)
            except Exception as e:
                logger.warning(
                    f"FlowScheduler: battle uid={candidate.uid} is durable "
                    f"but pre-list pruning failed; recovery sweep will retry: "
                    f"{type(e).__name__}: {e}"
                )

            task_state = await self.state.get_task_state()
            if (
                task_state is not None
                and adopted.started_at_block < task_state.refreshed_at_block
            ):
                logger.warning(
                    f"FlowScheduler: promoted pre-deployed uid="
                    f"{candidate.uid} across task-id refresh boundary "
                    f"(started_at_block={adopted.started_at_block} < "
                    f"refreshed_at_block={task_state.refreshed_at_block}); "
                    f"earlier-refresh samples do not count toward overlap"
                )
            logger.info(
                f"FlowScheduler: promoted pre-deployed uid={candidate.uid} "
                f"in place as active challenger"
            )

        if self.cfg.single_instance_provider:
            await self._release_champion_runtime_for_battle(champion, battle)
        logger.info(
            f"FlowScheduler: battle started — challenger uid={candidate.uid} "
            f"vs champion uid={champion.uid}"
        )

    async def _transition_runtime_role(
        self, record: BattleRecord, role: str,
    ) -> DeploymentRoleTransitionResult:
        if self._transition_deployment_role is None:
            return DeploymentRoleTransitionResult.UPDATED
        try:
            result = await self._transition_deployment_role(record, role)
        except Exception as e:
            logger.error(
                f"FlowScheduler: provider role transition failed "
                f"for uid={record.challenger.uid} role={role}; "
                f"{type(e).__name__}: {e}"
            )
            return DeploymentRoleTransitionResult.RETRYABLE
        if not isinstance(result, DeploymentRoleTransitionResult):
            logger.error(
                f"FlowScheduler: provider role transition returned invalid "
                f"result={result!r} for uid={record.challenger.uid} "
                f"role={role}; retrying without changing scheduler state"
            )
            return DeploymentRoleTransitionResult.RETRYABLE
        return result

    async def _promote_runtime_to_champion(
        self, battle: BattleRecord,
    ) -> DeploymentRoleTransitionResult:
        """Fence runtime reuse behind the provider's persisted assignment."""
        result = await self._transition_runtime_role(battle, "champion")
        if result is DeploymentRoleTransitionResult.RETRYABLE:
            logger.error(
                "FlowScheduler: challenger-to-champion provider role "
                f"transition temporarily failed for uid="
                f"{battle.challenger.uid}; retaining battle for retry"
            )
        elif result is DeploymentRoleTransitionResult.STALE:
            logger.warning(
                "FlowScheduler: challenger-to-champion provider role "
                f"transition rejected for uid={battle.challenger.uid}; "
                "winner remains valid but its stale runtime will not be reused"
            )
        return result

    async def _deployment_capacity_available(self, role: str) -> bool:
        """Return whether an SSH/single-instance deployment can start now.

        Autoscaler-managed SSH endpoints legitimately disappear while the
        queue is idle. In that state scheduler should leave champion and
        challenger state untouched and wait for autoscaler to create a host,
        instead of calling deploy_fn every tick and logging the same
        ``no active ssh endpoints`` exception.
        """
        if (
            not self.cfg.single_instance_provider
            or self._list_active_endpoint_names is None
        ):
            return True
        try:
            active_endpoints = await self._list_active_endpoint_names()
        except Exception as e:
            logger.warning(
                f"FlowScheduler: active endpoint lookup failed before "
                f"{role} deploy; proceeding with deploy attempt: "
                f"{type(e).__name__}: {e}"
            )
            return True
        if active_endpoints:
            self._last_no_endpoint_log_at = 0.0
            return True
        now = time.time()
        if now - self._last_no_endpoint_log_at >= 60:
            logger.info(
                f"FlowScheduler: no active ssh endpoints for {role} deploy; "
                f"waiting for autoscaler"
            )
            self._last_no_endpoint_log_at = now
        return False

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
            gate_envs = await self.state.get_runtime_environments()
            sealed, gate_snapshot = (
                await self._seal_behavior_gate_for_promotion(
                    battle, gate_envs,
                )
            )
            if not sealed:
                if gate_snapshot is not None and gate_snapshot.failed:
                    await self._rollback_failed_promotion_recovery(
                        champion,
                        battle,
                        reason=gate_snapshot.reason,
                        current_block=current_block,
                    )
                else:
                    status = (
                        gate_snapshot.status.value
                        if gate_snapshot is not None
                        else VerdictStatus.PENDING.value
                    )
                    reason = (
                        gate_snapshot.reason
                        if gate_snapshot is not None
                        else "promotion_seal_not_committed"
                    )
                    logger.warning(
                        "FlowScheduler: post-promotion recovery paused by "
                        f"behavior gate uid={champion.uid} status={status} "
                        f"reason={reason}"
                    )
                return
            transition = await self._promote_runtime_to_champion(battle)
            if transition is DeploymentRoleTransitionResult.RETRYABLE:
                return
            if transition is DeploymentRoleTransitionResult.STALE:
                # The winner was already made canonical before the crash, but
                # its provider assignment has since disappeared or changed
                # owner. Preserve the evaluation result and force a clean
                # champion deploy on a later tick.
                await self._clear_champion_deployment_state(champion)
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
                # Sign-crossing envs (distill) use the additive margin for
                # both dominant and not_worse; natural [0,1] envs use the
                # default additive dominant margin + multiplicative not_worse.
                margin=(
                    DEFAULT_ADDITIVE_MARGIN if env in ADDITIVE_MARGIN_ENVS
                    else DEFAULT_MARGIN
                ),
                # Comparator's per-env sample-count gate. Aligned with
                # ``_battle_overlap_ready`` (which requires overlap ≥
                # ``sampling_count``) so the two gates can't disagree:
                # SWE=200, MEMORY=100, etc. — full coverage required.
                min_tasks_per_env=max(1, int(cfg.sampling_count)),
                not_worse_tolerance=DEFAULT_NOT_WORSE_TOLERANCE,
                additive_margin=DEFAULT_ADDITIVE_MARGIN,
            )
            for env, cfg in envs.items()
        }
        # Read each side's current-refresh scores, then restrict the
        # comparator's view to the overlap (task_ids both miners have
        # sampled in this refresh). Asymmetric pool coverage gets
        # neutralised here — both sides see exactly the same task_ids.
        champ_scores: Dict[str, Dict[int, float]] = {}
        chal_scores: Dict[str, Dict[int, float]] = {}
        overlap_task_ids: Dict[str, List[int]] = {}
        champ_metrics: Dict[str, Dict[int, SampleMetric]] = {}
        chal_metrics: Dict[str, Dict[int, SampleMetric]] = {}
        token_config = load_token_efficiency_config(
            await self.state.get_environment_payloads()
        )
        token_enabled = bool(token_config and token_config.enabled_for_sampling)
        for env in env_configs:
            tasks = task_state.task_ids.get(env, [])
            if token_enabled and self._sample_metrics_reader is not None:
                champ_metric_full = await self._sample_metrics_reader(
                    champion.hotkey, champion.revision, env, tasks,
                    task_state.refreshed_at_block, True,
                )
                chal_metric_full = await self._sample_metrics_reader(
                    battle.challenger.hotkey, battle.challenger.revision, env, tasks,
                    task_state.refreshed_at_block, True,
                )
                champ_full = {
                    task_id: metric.score
                    for task_id, metric in champ_metric_full.items()
                }
                chal_full = {
                    task_id: metric.score
                    for task_id, metric in chal_metric_full.items()
                }
            else:
                champ_full = await self._scores_reader(
                    champion.hotkey, champion.revision, env, tasks,
                    task_state.refreshed_at_block,
                )
                chal_full = await self._scores_reader(
                    battle.challenger.hotkey, battle.challenger.revision, env, tasks,
                    task_state.refreshed_at_block,
                )
                champ_metric_full = {
                    task_id: SampleMetric(score=score)
                    for task_id, score in champ_full.items()
                }
                chal_metric_full = {
                    task_id: SampleMetric(score=score)
                    for task_id, score in chal_full.items()
                }
            overlap_ids = sorted(set(champ_full) & set(chal_full))
            overlap_task_ids[env] = overlap_ids
            champ_scores[env] = {t: champ_full[t] for t in overlap_ids}
            chal_scores[env] = {t: chal_full[t] for t in overlap_ids}
            champ_metrics[env] = {t: champ_metric_full[t] for t in overlap_ids}
            chal_metrics[env] = {t: chal_metric_full[t] for t in overlap_ids}

        token_sidecar: TokenEfficiencyComputation | None = None
        if token_config and token_config.enabled_for_sampling:
            token_sidecar = compute_token_efficiency(
                env=TOKEN_EFFICIENCY_ENV,
                config=token_config,
                basis_metrics_by_runtime_env=champ_metrics,
                subject_metrics_by_runtime_env=chal_metrics,
                overlap_ids_by_runtime_env={
                    env: set(ids) for env, ids in overlap_task_ids.items()
                },
            )
            if (
                token_config.enabled_for_scoring
                and token_sidecar.available
                and token_sidecar.champion_score is not None
                and token_sidecar.challenger_score is not None
                and token_sidecar.comparison_config is not None
            ):
                champ_scores[TOKEN_EFFICIENCY_ENV] = {
                    0: token_sidecar.champion_score
                }
                chal_scores[TOKEN_EFFICIENCY_ENV] = {
                    0: token_sidecar.challenger_score
                }
                env_configs[TOKEN_EFFICIENCY_ENV] = token_sidecar.comparison_config

        result = self.comparator.compare(
            champion_scores=champ_scores,
            challenger_scores=chal_scores,
            env_configs=env_configs,
            min_dominant_envs=WIN_MIN_DOMINANT_ENVS,
        )
        if token_sidecar is not None:
            setattr(result, "token_efficiency", token_sidecar)

        if result.winner == "challenger":
            # Fence the promotion with a fresh strongly-consistent read.  A
            # real sample can invalidate a previously-passed preflight while
            # this tick is awaiting score reads/comparison; promoting from the
            # earlier snapshot would make that final failure unreachable once
            # the subject becomes champion.
            gate_envs = await self.state.get_runtime_environments()
            gate_snapshot = await self._active_behavior_gate_snapshot(
                battle, gate_envs,
            )
            if gate_snapshot is not None:
                if gate_snapshot.failed:
                    await self._decide_behavior_gate_lost(
                        champion, battle, gate_snapshot.reason,
                    )
                    return
                if not gate_snapshot.passed:
                    logger.warning(
                        "FlowScheduler: challenger promotion paused by "
                        f"behavior gate uid={battle.challenger.uid} "
                        f"status={gate_snapshot.status.value} "
                        f"reason={gate_snapshot.reason}"
                    )
                    return
            await self._teardown_record(champion)
            # The old workload is now gone.  Persist that fact before the
            # atomic seal so a failed/uncertain seal can never leave a stale
            # champion URL dispatchable while promotion is paused.
            await self._clear_champion_deployment_state(champion)
            sealed, gate_snapshot = (
                await self._seal_behavior_gate_for_promotion(
                    battle, gate_envs,
                )
            )
            if not sealed:
                if gate_snapshot is not None and gate_snapshot.failed:
                    await self._decide_behavior_gate_lost(
                        champion, battle, gate_snapshot.reason,
                    )
                    return
                status = (
                    gate_snapshot.status.value
                    if gate_snapshot is not None
                    else VerdictStatus.PENDING.value
                )
                reason = (
                    gate_snapshot.reason
                    if gate_snapshot is not None
                    else "promotion_seal_not_committed"
                )
                logger.warning(
                    "FlowScheduler: challenger promotion paused after old "
                    "champion teardown because behavior gate was not sealed "
                    f"uid={battle.challenger.uid} status={status} "
                    f"reason={reason}"
                )
                return
            transition = await self._promote_runtime_to_champion(battle)
            if transition is DeploymentRoleTransitionResult.RETRYABLE:
                return
            reuse_runtime = (
                transition is DeploymentRoleTransitionResult.UPDATED
            )
            new_champion = ChampionRecord(
                uid=battle.challenger.uid,
                hotkey=battle.challenger.hotkey,
                revision=battle.challenger.revision,
                model=battle.challenger.model,
                model_type=battle.challenger.model_type,
                deployment_id=(
                    battle.deployment_id if reuse_runtime else None
                ),
                base_url=battle.base_url if reuse_runtime else None,
                deployments=(
                    list(battle.deployments) if reuse_runtime else []
                ),
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
            # Old champion is being displaced — its basis is the winner.
            champ_so, champ_so_basis = await self._sampling_only_freeze_scores(
                subject_hotkey=champion.hotkey,
                subject_revision=champion.revision,
                basis_hotkey=battle.challenger.hotkey,
                basis_revision=battle.challenger.revision,
                task_state=task_state,
            )
            await self.queue.mark_terminated(
                champion.uid,
                OUTCOME_LOST,
                reason=f"dethroned_by:{battle.challenger.hotkey[:10]}",
                hotkey=champion.hotkey,
                revision=champion.revision,
                model=champion.model,
                scores_by_env={
                    **champ_so,
                    **_final_scores_from_result(result, role="champion"),
                },
                opponent_scores_by_env={
                    **champ_so_basis,
                    **_opponent_scores_from_result(result, role="champion"),
                },
                battle_task_ids=overlap_task_ids,
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
            chal_so, chal_so_basis = await self._sampling_only_freeze_scores(
                subject_hotkey=battle.challenger.hotkey,
                subject_revision=battle.challenger.revision,
                basis_hotkey=champion.hotkey,
                basis_revision=champion.revision,
                task_state=task_state,
            )
            await self.queue.mark_terminated(
                battle.challenger.uid,
                OUTCOME_LOST,
                reason=f"lost_to_champion:{champion.hotkey[:10]}:{result.reason}",
                hotkey=battle.challenger.hotkey,
                revision=battle.challenger.revision,
                model=battle.challenger.model,
                scores_by_env={
                    **chal_so,
                    **_final_scores_from_result(result, role="challenger"),
                },
                opponent_scores_by_env={
                    **chal_so_basis,
                    **_opponent_scores_from_result(result, role="challenger"),
                },
                battle_task_ids=overlap_task_ids,
                scores_refresh_block=task_state.refreshed_at_block,
                terminated_at_block=current_block,
            )
            # The champion runtime is normally released at battle start. Keep
            # the persisted state empty so the next tick either starts another
            # battle from completed samples or redeploys the champion.
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

    async def _apply_window_rotation_request(
        self,
        request: WindowRotationRequest,
        *,
        battle: Optional[BattleRecord],
        task_state: Optional[TaskIdState],
    ) -> bool:
        """Apply an operator-requested window rotation.

        The CLI only records the request. Scheduler consumes it here so
        active battle deployments are torn down through the provider-aware
        ``teardown_fn`` before ``current_battle`` is cleared.
        """
        if battle is not None:
            try:
                await self._teardown_record(battle)
            except Exception as e:
                logger.error(
                    f"FlowScheduler: window rotation request at block "
                    f"{request.requested_at_block} could not teardown "
                    f"battle deployment uid={battle.challenger.uid}: "
                    f"{type(e).__name__}: {e}; will retry next tick"
                )
                return False
            released = await self.queue.release_claim(
                battle.challenger.uid,
                hotkey=battle.challenger.hotkey,
                revision=battle.challenger.revision,
            )
            logger.warning(
                f"FlowScheduler: window rotation tore down battle "
                f"uid={battle.challenger.uid} "
                f"(claim_released={released})"
            )

        if task_state is not None:
            await self.state.clear_task_state()
            logger.warning(
                "FlowScheduler: window rotation cleared task pool; "
                "scheduler will refresh it before continuing"
            )

        await self.state.clear_battle()
        await self.state.clear_window_rotation_request()
        logger.warning(
            f"FlowScheduler: consumed window rotation request from block "
            f"{request.requested_at_block}"
        )
        return True

    async def _teardown_record(self, record: Any) -> None:
        deployments = list(getattr(record, "deployments", []) or [])
        if deployments:
            for dep in deployments:
                await self._teardown(dep.deployment_id)
            return
        await self._teardown(getattr(record, "deployment_id", None))

    async def _release_champion_runtime_for_battle(
        self,
        champion: ChampionRecord,
        battle: BattleRecord,
    ) -> bool:
        """Release the old champion's distinct endpoint after battle start.

        ``current_battle`` is persisted before this runs, so a teardown error
        can safely retry next tick. Deployment ids shared with the battle are
        skipped: that is the one-endpoint model-swap case where tearing down
        the old champion id would actually kill the active challenger.
        """
        if not (
            champion.deployment_id
            or champion.base_url
            or champion.deployments
        ):
            return True

        battle_ids = set(_record_deployment_ids(battle))
        champion_ids = _record_deployment_ids(champion)
        try:
            for deployment_id in champion_ids:
                if deployment_id in battle_ids:
                    continue
                await self._teardown(deployment_id)
            await self._clear_champion_deployment_state(champion)
        except Exception as e:
            logger.error(
                f"FlowScheduler: failed to release champion uid="
                f"{champion.uid} runtime after battle uid="
                f"{battle.challenger.uid} became durable: "
                f"{type(e).__name__}: {e}; will retry next tick"
            )
            return False

        logger.info(
            f"FlowScheduler: released champion uid={champion.uid} runtime "
            f"for active battle uid={battle.challenger.uid}"
        )
        return True

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
                    model_type=str(m.get("model_type") or ""),
                )
            )
        if not any(s.is_champion for s in subjects):
            subjects.append(
                WeightSubject(
                    uid=champion.uid, hotkey=champion.hotkey,
                    revision=champion.revision, model=champion.model,
                    first_block=0, is_champion=True,
                    scores_by_env=champion_env_scores, total_samples=0,
                    model_type=champion.model_type,
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


def _same_subject(left: Any, right: Any) -> bool:
    return (
        getattr(left, "hotkey", None) == getattr(right, "hotkey", None)
        and getattr(left, "revision", None) == getattr(right, "revision", None)
    )


def _record_matches_candidate(record: BattleRecord, candidate: Any) -> bool:
    challenger = record.challenger
    return (
        challenger.uid == candidate.uid
        and challenger.hotkey == candidate.hotkey
        and challenger.revision == candidate.revision
        and challenger.model == candidate.model
    )


def _champion_snapshot(champion: ChampionRecord) -> MinerSnapshot:
    return MinerSnapshot(
        uid=champion.uid,
        hotkey=champion.hotkey,
        revision=champion.revision,
        model=champion.model,
        model_type=champion.model_type,
    )


def _record_deployment_ids(record: Any) -> List[str]:
    ids = [
        str(dep.deployment_id)
        for dep in (getattr(record, "deployments", []) or [])
        if getattr(dep, "deployment_id", None)
    ]
    fallback = getattr(record, "deployment_id", None)
    if fallback:
        ids.append(str(fallback))
    return list(dict.fromkeys(ids))


def _records_share_deployment(left: Any, right: Any) -> bool:
    left_ids = set(_record_deployment_ids(left))
    right_ids = set(_record_deployment_ids(right))
    if left_ids and right_ids:
        return bool(left_ids & right_ids)
    left_url = getattr(left, "base_url", None)
    right_url = getattr(right, "base_url", None)
    return bool(left_url and right_url and left_url == right_url)


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
) -> Dict[str, Dict[str, Any]]:
    """Build ``{env: {count, avg, champion_overlap_avg}}`` for one side of
    a finished contest. Used to freeze the comparator's decide-time view
    of the loser onto their miner_stats row, so rank UI keeps showing
    count / avg / threshold after the live cache forgets them.

    ``role`` is ``"champion"`` (record for a displaced champion) or
    ``"challenger"`` (record for a challenger that lost). The opposing
    side's avg is recorded as ``champion_overlap_avg`` — the basis the
    comparator actually used to decide this contest.
    """
    out: Dict[str, Dict[str, Any]] = {}
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
    token = _token_sidecar(result)
    if token is not None:
        out[token.env] = (
            token.champion_payload if role == "champion"
            else token.challenger_payload
        )
    return out


def _opponent_scores_from_result(
    result: Any, *, role: str,
) -> Dict[str, Dict[str, float]]:
    opponent_role = "challenger" if role == "champion" else "champion"
    return _own_scores_only(
        _final_scores_from_result(result, role=opponent_role)
    )


def _opponent_scores_from_early_lost(
    per_env_data: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for env, data in per_env_data.items():
        count = data.get("count")
        avg = data.get("champion_overlap_avg")
        if count is None or avg is None:
            continue
        out[env] = {"count": int(count), "avg": float(avg)}
    return out


def _own_scores_only(
    scores_by_env: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for env, data in scores_by_env.items():
        if data.get("unit") == "tokens":
            continue
        if "count" not in data or "avg" not in data:
            continue
        out[env] = {"count": int(data["count"]), "avg": float(data["avg"])}
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


def _token_sidecar(result: Any) -> Optional[TokenEfficiencyComputation]:
    token = getattr(result, "token_efficiency", None)
    return token if isinstance(token, TokenEfficiencyComputation) else None


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
    per_env = [
        _env_comparison_to_dict(e) for e in (getattr(result, "per_env", []) or [])
    ]
    token = _token_sidecar(result)
    if token is not None:
        token_row = next(
            (row for row in per_env if row.get("env") == token.env),
            None,
        )
        metric = dict(token.snapshot_metric)
        if token_row is not None:
            token_row["metric"] = metric
        else:
            per_env.append({
                "env": token.env,
                "verdict": "unavailable",
                "reason": token.reason,
                "metric": metric,
            })
    out["per_env"] = per_env
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
                "not_worse_threshold": not_worse_lower_bound(
                    champion_basis, str(env),
                    tolerance=tolerance,
                    additive_margin=DEFAULT_ADDITIVE_MARGIN,
                ),
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
    token = _token_sidecar(result)
    if token is not None:
        out[token.env] = (
            token.challenger_payload if role == "challenger"
            else token.champion_payload
        )
    return out
