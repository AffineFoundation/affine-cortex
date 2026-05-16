"""
Executor worker — one per env, DB-poll mode.

Each subprocess owns one env (its own Docker container + SSH tunnel +
paramiko Transport). It polls ``system_config``; whenever the saved
state has a champion (with a deployment URL) it makes sure that champion
has a sample row for every task_id in the current pool. When there's
also an in-flight battle, the challenger gets the same treatment.

No phase enum, no window_id — just look at the state and fill in what's
missing.
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any, Dict, List, Optional

from affine.core.setup import logger
from affine.database.dao.sample_results import SampleResultsDAO
from affine.database.dao.system_config import SystemConfigDAO
from affine.src.executor.logging_utils import safe_log
from affine.src.executor.metrics import WorkerMetrics
from affine.src.scorer.dao_adapters import SampleResultsAdapter
from affine.src.scorer.sampling_thresholds import (
    champion_completion_threshold,
)
from affine.src.scorer.window_state import (
    DeploymentRecord,
    MinerSnapshot,
    StateStore,
    SystemConfigKVAdapter,
)


class ExecutorWorker:
    """One worker = one env. Polls state, runs sampling for its env."""

    def __init__(
        self,
        worker_id: int,
        env: str,
        *,
        max_concurrent: int = 60,
        poll_interval_sec: float = 5.0,
        idle_sleep_sec: float = 10.0,
        warmup_sec: float = 240.0,
        global_sem: Any = None,
        in_flight_value: Any = None,
        env_cap_value: Any = None,
    ):
        self.worker_id = worker_id
        self.env = env
        self.max_concurrent = max_concurrent
        # Cross-process ``BoundedSemaphore`` (or ``None`` in unit tests /
        # standalone runs). Real backpressure runs through this — see
        # ``_acquire_global_slot`` below. ``max_concurrent`` remains as a
        # standalone fallback for tests/local runs; production workers read
        # their live cap from ``_env_cap_value``.
        self._global_sem = global_sem
        # Manager-updated dynamic cap for this env. The worker reads this
        # before acquiring the global semaphore, so cap changes take effect
        # without restarting the subprocess.
        self._env_cap_value = env_cap_value
        # Per-env ``mp.Value(c_int, lock=False)`` for the manager's
        # ``[STATUS]`` printer to read live in-flight without IPC ping.
        # Single writer (this worker), single reader (manager); aligned
        # c_int reads are atomic on CPython so no lock needed.
        self._in_flight_value = in_flight_value
        self.poll_interval_sec = poll_interval_sec
        self.idle_sleep_sec = idle_sleep_sec
        # ``warmup_sec``: env containers report "ready" before they are
        # fully serving — affinetes finishes SSH tunnel + container startup
        # but the env's own app-level init (e.g. liveweb's Stooq lock /
        # cache build, memorygym worker boot, swebench:infinite's 3.5GB
        # image + python deps) can still be running. 60s tripped a
        # ReadError storm; 180s held for most envs but SWE-INFINITE
        # consistently needed ~240s before serving (observed 166-task
        # fail burst at the 180s mark on PR-#453 deploy). 240s clears
        # all observed env images at the cost of one extra minute on
        # cold restart.
        self.warmup_sec = warmup_sec

        self.running = False
        self.metrics = WorkerMetrics(worker_id=worker_id, env=env)

        self._env_executor = None
        self._state: Optional[StateStore] = None
        self._samples: Optional[SampleResultsAdapter] = None

    # ---- lifecycle ---------------------------------------------------------

    async def initialize(self) -> None:
        safe_log(f"[{self.env}] worker init", "INFO")
        # Each subprocess (spawn context) starts with a fresh Python module
        # state — the parent process's init_client() doesn't carry over.
        # Initialize the DynamoDB client here so the first DAO call doesn't
        # raise ``RuntimeError("DynamoDB client not initialized")``.
        from affine.database import init_client
        await init_client()
        self._state = StateStore(
            SystemConfigKVAdapter(
                SystemConfigDAO(), updated_by=f"executor-{self.env}",
            )
        )
        self._samples = SampleResultsAdapter(
            dao=SampleResultsDAO(),
            validator_hotkey=f"executor-{self.env}",
        )
        await self._init_env_executor()
        safe_log(f"[{self.env}] worker ready", "INFO")

    async def _init_env_executor(self) -> None:
        if self._env_executor is not None:
            return
        from affine.core.environments import SDKEnvironment

        self._env_executor = SDKEnvironment(self.env)
        _, mode = self._env_executor._get_hosts_and_mode()
        safe_log(f"[{self.env}] env mode={mode}", "INFO")

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False

    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics.to_dict()

    # ---- main loop ---------------------------------------------------------

    async def run(self) -> None:
        """Streaming dispatch loop.

        Unlike the older per-tick model (which gathered the entire pending
        list and blocked until every task finished before computing the
        next tick), this loop:

          - Computes new dispatchable (subject, task_id) keys on every
            poll.
          - Fires each new dispatch as a free-standing ``asyncio.Task``
            and immediately returns to the loop.
          - Tracks in-flight by key (subject_hotkey, subject_revision,
            task_id) so a slow task doesn't block its peers and the
            next poll can pull more work into the global semaphore the
            moment a peer completes.

        Net effect: a tick that contains 200 SWE-INFINITE tasks no
        longer waits for the slowest task to finish before dispatching
        peer task_ids. The per-env cap is applied before the shared
        global semaphore so fast-cycling envs cannot monopolize it.
        """
        assert self._state and self._samples, "call initialize() first"
        status_task = asyncio.create_task(self._publish_status_loop())
        # Tracks which (subject_hotkey, subject_revision, task_id) keys
        # are currently in flight so we don't dispatch the same key
        # twice while the first attempt is still running. Reset whenever
        # the task-id pool refreshes (new window).
        in_flight_keys: set = set()
        # Active asyncio.Task objects; kept so we can reap completed
        # ones and cancel everything on stop.
        in_flight_tasks: set = set()
        try:
            if self.warmup_sec > 0:
                logger.info(
                    f"[{self.env}] warmup: sleeping {self.warmup_sec:.0f}s before "
                    f"first dispatch (let env containers finish app-level boot)"
                )
                await asyncio.sleep(self.warmup_sec)
            while self.running:
                try:
                    # Reap completed dispatch tasks before launching new ones.
                    completed = {t for t in in_flight_tasks if t.done()}
                    in_flight_tasks -= completed
                    for t in completed:
                        # Surface any unexpected exception so silent crashes
                        # don't hide. ``_evaluate_and_persist`` catches its
                        # own exceptions; anything raised past it is a bug.
                        exc = t.exception()
                        if exc is not None:
                            logger.error(
                                f"[{self.env}] dispatch task error: "
                                f"{type(exc).__name__}: {exc}"
                            )
                    new = await self._dispatch_new(in_flight_keys, in_flight_tasks)
                except Exception as e:
                    logger.error(
                        f"[{self.env}] dispatch loop raised: {type(e).__name__}: {e}",
                        exc_info=True,
                    )
                    new = 0
                # Short poll when we just launched fresh work, longer when
                # idle so we're not hammering DB checks needlessly.
                await asyncio.sleep(self.poll_interval_sec if new else self.idle_sleep_sec)
        finally:
            status_task.cancel()
            for t in in_flight_tasks:
                t.cancel()
            if in_flight_tasks:
                await asyncio.gather(*in_flight_tasks, return_exceptions=True)

    async def _acquire_global_slot(self) -> None:
        """Block until the cross-process global dispatch semaphore yields
        a slot. Poll with non-blocking ``acquire`` + ``asyncio.sleep`` so
        we don't park a kernel thread per pending coroutine — there can
        be thousands of pending coroutines across all envs and we still
        want a single asyncio event loop per worker to stay responsive.
        """
        if self._global_sem is None:
            return
        while True:
            if self._global_sem.acquire(block=False):
                return
            await asyncio.sleep(0.05)

    def _release_global_slot(self) -> None:
        """Return a global slot. Swallow ``ValueError`` from
        ``BoundedSemaphore.release`` past its initial value — should not
        happen but it's not worth crashing the worker over."""
        if self._global_sem is None:
            return
        try:
            self._global_sem.release()
        except ValueError:
            logger.debug(f"[{self.env}] global sem release past bound")

    async def _publish_status_loop(self, interval_sec: float = 10.0) -> None:
        """Periodically write this worker's metrics to ``system_config`` under
        ``worker_status_<env>``. ``af db worker-status`` and
        ``af db sample-progress`` read these rows so operators can see live
        in-flight concurrency without docker-exec spelunking. Uses the
        asyncio loop directly — never multiprocessing.Queue (that path
        historically segfaulted under paramiko + nest_asyncio).

        Errors are caught and logged at WARNING (first occurrence) /
        DEBUG (subsequent) so a broken publish path surfaces in the
        executor's normal log without spamming it.
        """
        try:
            import time as _t
            from affine.src.scorer.window_state import SystemConfigKVAdapter
            from affine.database.dao.system_config import SystemConfigDAO
            kv = SystemConfigKVAdapter(SystemConfigDAO(), updated_by=f"executor-{self.env}")
        except Exception as e:
            logger.warning(
                f"[{self.env}] status publish setup failed; loop disabled: "
                f"{type(e).__name__}: {e}"
            )
            return

        # Confirm the loop actually entered iteration — this single line
        # is the diagnostic for "did the task get scheduled at all?".
        logger.info(f"[{self.env}] status publish loop entered (interval={interval_sec}s)")

        ever_failed = False
        while self.running:
            try:
                payload = self.metrics.to_dict()
                payload["reported_at"] = _t.time()
                await kv.set(f"worker_status_{self.env}", payload)
            except Exception as e:
                if not ever_failed:
                    logger.warning(
                        f"[{self.env}] status publish failed (first occurrence): "
                        f"{type(e).__name__}: {e}",
                        exc_info=True,
                    )
                    ever_failed = True
                else:
                    logger.debug(f"[{self.env}] status publish failed: {e}")
            await asyncio.sleep(interval_sec)

    async def _dispatch_new(
        self, in_flight_keys: set, in_flight_tasks: set,
    ) -> int:
        """Launch every (subject, task_id) that's still un-sampled and not
        already in flight. Each launched task is fire-and-forget — the
        run loop reaps completions on its next iteration.

        Returns the number of new dispatches so the loop knows whether
        to short-poll (more work to do) or idle-sleep.

        **Per-env sampling policy** (per-env stop here; the all-env
        gate that actually advances the battle lives in
        ``flow.FlowScheduler._samples_complete`` /
        ``_battle_overlap_ready``):

          - **Champion** drains the pool until ``len(champ_done) ≥
            champion_completion_threshold(sampling_count)`` (95% of the
            pool). The remaining 5% is the deliberately-abandoned long
            tail.
          - **Challenger** dispatches the full set of task_ids the
            *champion* has already sampled — never the raw pool — so a
            permanently-failing task in the long tail can't bias which
            task_ids the challenger ends up with. We early-stop once
            ``overlap (champ_done ∩ chal_done) ≥ sampling_count``;
            extra dispatches past that add no comparator signal.

        Why "challenger samples champion's set" rather than the raw
        pool: the contest gate (``_battle_overlap_ready``) requires
        ``overlap ≥ sampling_count``. If challenger sampled task_ids
        the champion never completed, those samples never count toward
        overlap and the math can't close — see
        ``affine/src/scorer/sampling_thresholds.py``.

        Concurrency: candidates pass the optional per-env semaphore first,
        then the shared cross-process global semaphore. This keeps all
        envs streaming while bounding how much of the global budget one
        env can occupy.
        """
        task_state = await self._state.get_task_state()
        if task_state is None:
            return 0
        task_ids = task_state.task_ids.get(self.env) or []
        if not task_ids:
            return 0
        refresh_block = task_state.refreshed_at_block

        # Pool refresh = new window. Clear in-flight tracking so the next
        # round of task_ids gets dispatched cleanly. Any tasks still in
        # the old refresh keep running; their result rows will be tagged
        # with the prior ``refresh_block`` and ignored by the comparator.
        if getattr(self, "_dispatch_refresh_block", 0) != refresh_block:
            in_flight_keys.clear()
            self._dispatch_refresh_block = refresh_block

        envs = await self._state.get_environments()
        env_cfg = envs.get(self.env)
        if env_cfg is None:
            return 0
        sampling_count = env_cfg.sampling_count

        champion = await self._state.get_champion()
        if champion is None:
            return 0
        champion_urls = _base_urls(
            champion.deployments,
            champion.base_url,
        )

        battle = await self._state.get_battle()
        battle_urls = _base_urls(
            battle.deployments if battle else [],
            battle.base_url if battle else None,
        )

        # Single-instance providers deliberately clear champion.deployment_id
        # at battle start (the host is now serving the challenger). In that
        # state we still need to dispatch challenger samples — bail out
        # only when nobody has a serving URL.
        if not champion_urls and not battle_urls:
            return 0

        # Read champion's current sample set once; both branches use it
        # (champion early-stop check + challenger candidate filter).
        # One Query (paginated) replaces the per-tid GetItems we used to
        # do via ``has_sample``.
        champ_done = await self._samples.read_scores_for_tasks(
            champion.hotkey, champion.revision, self.env,
            task_ids, refresh_block=refresh_block,
        )
        champion_threshold = champion_completion_threshold(sampling_count)

        # Each candidate carries the deployment_id we resolved from the
        # state read above. ``_evaluate_and_persist_gated`` re-reads
        # state at persist time and drops the result if this token is
        # no longer current — see ``_is_current_deployment``.
        candidates: List[tuple] = []
        if champion_urls and len(champ_done) < champion_threshold:
            champ_snap = MinerSnapshot(
                uid=champion.uid, hotkey=champion.hotkey,
                revision=champion.revision, model=champion.model,
            )
            for idx, tid in enumerate(task_ids):
                tid_int = int(tid)
                if tid_int in champ_done:
                    continue
                key = (champion.hotkey, champion.revision, tid_int)
                if key in in_flight_keys:
                    continue
                url = _pick_url(champion_urls, tid, idx)
                dep_id = _resolve_deployment_id(
                    url, champion.deployments, champion.deployment_id,
                )
                candidates.append((key, champ_snap, tid, url, dep_id))

        if battle is not None and battle_urls:
            chal_done = await self._samples.read_scores_for_tasks(
                battle.challenger.hotkey, battle.challenger.revision,
                self.env, task_ids, refresh_block=refresh_block,
            )
            overlap_count = len(champ_done.keys() & chal_done.keys())
            if overlap_count < sampling_count:
                for idx, tid in enumerate(task_ids):
                    tid_int = int(tid)
                    if tid_int not in champ_done:
                        continue              # champion hasn't sampled yet
                    if tid_int in chal_done:
                        continue              # challenger already done
                    key = (battle.challenger.hotkey, battle.challenger.revision, tid_int)
                    if key in in_flight_keys:
                        continue
                    url = _pick_url(battle_urls, tid, idx)
                    dep_id = _resolve_deployment_id(
                        url, battle.deployments, battle.deployment_id,
                    )
                    candidates.append(
                        (key, battle.challenger, tid, url, dep_id)
                    )

        # Pre-deployed challengers: same gating as ``battle.challenger``
        # but each one runs on its own non-primary endpoint. The
        # ``champion hasn't sampled yet`` guard naturally pauses their
        # dispatch while champion is still doing its baseline pass —
        # which is what makes "其他机器空闲等冠军" automatic without
        # any explicit wait logic. Their samples flow into
        # ``sample_results`` keyed by their own hotkey/revision, so
        # the scheduler's early-loss sweep and (eventually) the
        # comparator can read them just like a normal challenger's.
        for record in await self._state.get_predeployed_challengers():
            pre_urls = _base_urls(
                record.deployments, record.base_url,
            )
            if not pre_urls:
                continue
            pre_done = await self._samples.read_scores_for_tasks(
                record.challenger.hotkey, record.challenger.revision,
                self.env, task_ids, refresh_block=refresh_block,
            )
            if len(champ_done.keys() & pre_done.keys()) >= sampling_count:
                continue                  # enough overlap; let scheduler decide
            for idx, tid in enumerate(task_ids):
                tid_int = int(tid)
                if tid_int not in champ_done:
                    continue              # champion hasn't sampled yet
                if tid_int in pre_done:
                    continue              # this pre-challenger already done
                key = (
                    record.challenger.hotkey,
                    record.challenger.revision,
                    tid_int,
                )
                if key in in_flight_keys:
                    continue
                url = _pick_url(pre_urls, tid, idx)
                dep_id = _resolve_deployment_id(
                    url, record.deployments, record.deployment_id,
                )
                candidates.append(
                    (key, record.challenger, tid, url, dep_id)
                )

        if not candidates:
            return 0

        random.shuffle(candidates)

        # Launch each candidate as a free-standing task. The wrapper
        # removes the key from ``in_flight_keys`` on completion so a
        # failed-but-not-persisted task can be retried next poll.
        for key, miner, task_id, base_url, dep_id in candidates:
            in_flight_keys.add(key)
            t = asyncio.create_task(
                self._dispatch_one(
                    miner=miner, task_id=task_id, base_url=base_url,
                    refresh_block=refresh_block, key=key,
                    in_flight_keys=in_flight_keys,
                    expected_deployment_id=dep_id,
                )
            )
            in_flight_tasks.add(t)
        return len(candidates)

    async def _dispatch_one(
        self, *, miner: MinerSnapshot, task_id: int, base_url: str,
        refresh_block: int, key: tuple, in_flight_keys: set,
        expected_deployment_id: Optional[str] = None,
    ) -> None:
        try:
            await self._evaluate_and_persist(
                miner=miner, task_id=task_id, base_url=base_url,
                refresh_block=refresh_block,
                expected_deployment_id=expected_deployment_id,
            )
        finally:
            in_flight_keys.discard(key)

    async def _evaluate_and_persist(
        self, *, miner: MinerSnapshot, task_id: int, base_url: str,
        refresh_block: int,
        expected_deployment_id: Optional[str] = None,
    ) -> None:
        """Run one (env, task_id, miner) through affinetes, write result
        tagged with the task-id pool's ``refresh_block`` so the scheduler
        only counts current-refresh samples.

        Failure handling distinguishes three cases (pre-refactor parity,
        see PR #439):

          - **Miner-capability error** — the exception chain matches
            ``_ZERO_SCORE_ERROR_PATTERNS`` (context-length overflow).
            That's a property of the model and won't change on retry,
            so persist score=0.
          - **All other exceptions** (transport, parse, HTTP status,
            env crash mid-flight, env returning ``status=failed`` for
            non-capability reasons): never the miner's fault to know.
            **Don't persist** — let the next ``_tick`` re-attempt this
            task_id. The shuffle keeps us from always blocking on the
            same flaky task. The 10% buffer absorbs permanent infra-only
            failures; beyond that operators step in.
          - ``Result.success=False`` / ``error`` from a non-raising
            evaluate: the env returned a structured invalid attempt.
            **Don't persist** — rows with ``extra.error`` are not valid
            samples and must not count toward completion or comparison.
        """
        miner_obj = _Miner(
            hotkey=miner.hotkey, model=miner.model, revision=miner.revision,
            base_url=base_url,
        )
        await self._evaluate_and_persist_gated(
            miner=miner, task_id=task_id, base_url=base_url,
            refresh_block=refresh_block, miner_obj=miner_obj,
            expected_deployment_id=expected_deployment_id,
        )

    def _current_env_cap(self) -> int:
        if self._env_cap_value is None:
            return max(1, int(self.max_concurrent))
        return max(1, int(self._env_cap_value.value))

    def _current_env_in_flight(self) -> int:
        if self._in_flight_value is not None:
            return max(0, int(self._in_flight_value.value))
        return max(0, int(self.metrics.tasks_in_flight))

    async def _acquire_dispatch_slot(self) -> None:
        """Acquire this env's dynamic share, then one global slot.

        The cap is rechecked after acquiring the global slot because the
        manager can lower it while this coroutine is waiting.
        """
        while True:
            while self._current_env_in_flight() >= self._current_env_cap():
                await asyncio.sleep(0.05)
            await self._acquire_global_slot()
            if self._current_env_in_flight() < self._current_env_cap():
                return
            self._release_global_slot()
            await asyncio.sleep(0.05)

    async def _is_current_deployment(
        self, *, miner: MinerSnapshot, expected_deployment_id: Optional[str],
    ) -> bool:
        """True iff the deployment we dispatched against is still current
        for this miner in its current subject role.

        The drift this guards: between ``_dispatch_new`` and
        ``_samples.persist`` the scheduler may have swapped the inference
        deployment serving ``miner`` (e.g. champion lost, single-instance
        host re-tasked to a new challenger). The in-flight ``evaluate``
        call hits a base_url that's now serving a *different model*, so
        persisting its result would attribute new-model output to
        ``miner``. Drop instead.

        We accept the result iff the captured ``expected_deployment_id``
        still appears among the current champion's or current
        challenger's deployments. The challenger-wins case is naturally
        covered: when a challenger wins, its ``deployment_id`` is
        transferred into the new champion record, so the captured token
        still matches.

        ``expected_deployment_id is None`` means we couldn't resolve a
        token at dispatch (legacy record with neither ``deployments`` nor
        ``deployment_id``). We don't drop in that case — we have no way
        to detect drift, so preserve pre-validation behavior.
        """
        if expected_deployment_id is None:
            return True
        champion = await self._state.get_champion()
        if champion is not None and (
            champion.hotkey == miner.hotkey
            and champion.revision == miner.revision
        ):
            if champion.deployment_id == expected_deployment_id:
                return True
            if any(
                d.deployment_id == expected_deployment_id
                for d in champion.deployments
            ):
                return True
        battle = await self._state.get_battle()
        if battle is not None and (
            battle.challenger.hotkey == miner.hotkey
            and battle.challenger.revision == miner.revision
        ):
            if battle.deployment_id == expected_deployment_id:
                return True
            if any(
                d.deployment_id == expected_deployment_id
                for d in battle.deployments
            ):
                return True
        # Pre-deployed miners — same hotkey/revision identity check,
        # any deployment they currently hold counts as current. This
        # path is what keeps multi-host pre-sample dispatches from
        # being dropped as 'drift' even though they're attached to a
        # different (non-primary) endpoint than the active battle's.
        for record in await self._state.get_predeployed_challengers():
            if (
                record.challenger.hotkey != miner.hotkey
                or record.challenger.revision != miner.revision
            ):
                continue
            if record.deployment_id == expected_deployment_id:
                return True
            if any(
                d.deployment_id == expected_deployment_id
                for d in record.deployments
            ):
                return True
        return False

    async def _validate_or_drop(
        self, *, miner: MinerSnapshot, task_id: int,
        expected_deployment_id: Optional[str], latency_ms: int,
    ) -> bool:
        """Returns True if the persist should proceed, False to drop.

        On drop we increment ``tasks_dropped_drift`` and log; we do
        *not* call ``record_completion`` — the row was never written so
        the next dispatch tick will re-attempt this task_id against the
        new (current) deployment."""
        if await self._is_current_deployment(
            miner=miner, expected_deployment_id=expected_deployment_id,
        ):
            return True
        self.metrics.tasks_dropped_drift += 1
        logger.warning(
            f"[DRIFT] U{miner.uid:<4} │ {self.env:<20} │     DROPPED │ "
            f"task_id={task_id:<8} │ {latency_ms / 1000.0:6.3f}s │ "
            f"deployment {expected_deployment_id} no longer current"
        )
        return False

    async def _evaluate_and_persist_gated(
        self, *, miner: MinerSnapshot, task_id: int, base_url: str,
        refresh_block: int, miner_obj: "_Miner",
        expected_deployment_id: Optional[str] = None,
    ) -> None:
        await self._acquire_dispatch_slot()
        started = time.monotonic()
        # Track real in-flight concurrency. Two counters intentionally:
        # ``metrics.tasks_in_flight`` is in-process for ``af db worker-status``
        # back-compat; ``self._in_flight_value`` is shared with the manager
        # for the ``[STATUS]`` printer.
        self.metrics.tasks_in_flight += 1
        if self._in_flight_value is not None:
            self._in_flight_value.value += 1
        try:
            try:
                result = await self._env_executor.evaluate(
                    miner=miner_obj, task_id=task_id,
                )
            except Exception as e:
                latency_s = time.monotonic() - started
                latency_ms = int(latency_s * 1000)
                error_brief = str(e).replace("\n", " ").replace("\r", " ")[:200]
                zero_score = _is_zero_score_error(e)
                tag = "ZERO" if zero_score else "FAILED"
                logger.info(
                    f"[{tag}] U{miner.uid:<4} │ {self.env:<20} │     FAILED │ "
                    f"task_id={task_id:<8} │ {latency_s:6.3f}s │ {type(e).__name__}: {error_brief}"
                )
                if zero_score:
                    # Miner-capability limit (context overflow). Persist as
                    # score=0 — retrying would yield the same error.
                    if not await self._validate_or_drop(
                        miner=miner, task_id=task_id,
                        expected_deployment_id=expected_deployment_id,
                        latency_ms=latency_ms,
                    ):
                        return
                    await self._samples.persist(
                        miner_hotkey=miner.hotkey,
                        model_revision=miner.revision,
                        model=miner.model,
                        env=self.env,
                        task_id=task_id,
                        score=0.0,
                        latency_ms=latency_ms,
                        extra={
                            "error": error_brief,
                            "zero_score_reason": "context_overflow",
                        },
                        block_number=0,
                        refresh_block=refresh_block,
                    )
                self.metrics.record_completion(success=False, latency_ms=latency_ms)
                return
        finally:
            self.metrics.tasks_in_flight -= 1
            if self._in_flight_value is not None:
                self._in_flight_value.value -= 1
            self._release_global_slot()

        score = float(getattr(result, "score", 0.0))
        success = bool(getattr(result, "success", True))
        error = getattr(result, "error", None)
        extra = dict(getattr(result, "extra", {}) or {})
        latency_ms = int((time.monotonic() - started) * 1000)
        if error or extra.get("error"):
            error_brief = str(error or extra.get("error")).replace("\n", " ").replace("\r", " ")[:200]
            if _is_zero_score_error(Exception(error_brief)):
                if not await self._validate_or_drop(
                    miner=miner, task_id=task_id,
                    expected_deployment_id=expected_deployment_id,
                    latency_ms=latency_ms,
                ):
                    return
                await self._samples.persist(
                    miner_hotkey=miner.hotkey,
                    model_revision=miner.revision,
                    model=miner.model,
                    env=self.env,
                    task_id=task_id,
                    score=0.0,
                    latency_ms=latency_ms,
                    extra={
                        "error": error_brief,
                        "zero_score_reason": "context_overflow",
                    },
                    block_number=0,
                    refresh_block=refresh_block,
                )
                self.metrics.record_completion(success=False, latency_ms=latency_ms)
                return
            logger.info(
                f"[FAILED] U{miner.uid:<4} │ {self.env:<20} │     INVALID │ "
                f"task_id={task_id:<8} │ {latency_ms / 1000.0:6.3f}s │ {error_brief}"
            )
            self.metrics.record_completion(success=False, latency_ms=latency_ms)
            return
        if not await self._validate_or_drop(
            miner=miner, task_id=task_id,
            expected_deployment_id=expected_deployment_id,
            latency_ms=latency_ms,
        ):
            return
        await self._samples.persist(
            miner_hotkey=miner.hotkey,
            model_revision=miner.revision,
            model=miner.model,
            env=self.env,
            task_id=task_id,
            score=score,
            latency_ms=latency_ms,
            extra=extra,
            block_number=0,  # block tracking moved to scheduler/snapshot side
            refresh_block=refresh_block,
        )
        self.metrics.record_completion(success=success, latency_ms=latency_ms)
        # Observable success signal in the legacy executor's line format:
        # [RESULT] U<uid> │ <env>           │      0.500 │ task_id=12345 │   1.234s
        latency_s = latency_ms / 1000.0
        logger.info(
            f"[RESULT] U{miner.uid:<4} │ {self.env:<20} │ {score:10.3f} │ "
            f"task_id={task_id:<8} │ {latency_s:6.3f}s"
        )


class _Miner:
    """Duck-typed miner shim SDKEnvironment.evaluate expects."""

    __slots__ = (
        "hotkey", "model", "revision", "base_url",
        "inference_model", "public_base_url",
    )

    def __init__(self, *, hotkey: str, model: str, revision: str, base_url: str):
        self.hotkey = hotkey
        self.model = model
        self.revision = revision
        self.base_url = base_url
        self.inference_model = None
        self.public_base_url = None


def _base_urls(deployments: List[DeploymentRecord], fallback: Optional[str]) -> List[str]:
    urls = [d.base_url for d in deployments if d.base_url]
    if not urls and fallback:
        urls = [fallback]
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(urls))


def _pick_url(urls: List[str], task_id: int, index: int) -> str:
    if len(urls) == 1:
        return urls[0]
    return urls[(int(task_id) + index) % len(urls)]


def _resolve_deployment_id(
    base_url: str,
    deployments: List[DeploymentRecord],
    legacy_id: Optional[str],
) -> Optional[str]:
    """Map ``base_url`` → its serving ``deployment_id``.

    Multi-instance providers populate ``deployments`` (one entry per
    serving host, each with its own ``deployment_id``); legacy
    single-instance records leave that list empty and only set
    ``legacy_id`` + a single ``base_url``. We need a stable token to
    detect mid-evaluate model swaps (see ``_is_current_deployment``)."""
    for d in deployments:
        if d.base_url == base_url:
            return d.deployment_id
    return legacy_id


# Pre-refactor parity (see PR #439): only errors that point at a miner
# capability limit get persisted as a real zero-score sample. Everything
# else is transport / env-side flake and gets retried on the next tick.
#
# This is conservative on purpose. Random ``status=failed`` from the env,
# Targon 5xx, dropped connections — none of those are the miner's fault,
# and persisting them as 0 would let one flaky env round permanently
# disqualify a miner. The only sanctioned zero-score-on-error class is
# "the miner's response/prompt blew past the model context window",
# which is a model property and won't change on retry.
_ZERO_SCORE_ERROR_PATTERNS = (
    "context lengths of up to",
    "is longer than the model",
    "exceeds the maximum allowed length",
    "exceeds the maximum context length",
)


def _is_zero_score_error(exc: BaseException) -> bool:
    """Walk the cause chain and check each layer's message for a known
    miner-capability-limit pattern. We walk the chain because affinetes
    wraps the original error several times (ExecutionError → BackendError
    → EnvironmentError) and the substring of interest can live on any
    layer."""
    seen: List[BaseException] = []
    cur: Optional[BaseException] = exc
    while cur is not None and cur not in seen:
        msg = str(cur).lower()
        if any(p in msg for p in _ZERO_SCORE_ERROR_PATTERNS):
            return True
        seen.append(cur)
        cur = cur.__cause__ or cur.__context__
    return False
