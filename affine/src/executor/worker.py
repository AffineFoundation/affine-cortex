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
import hashlib
import json
import random
import time
from typing import Any, Dict, List, Optional

from affine.core.setup import logger
from affine.database.dao.sample_results import SampleResultsDAO
from affine.database.dao.system_config import SystemConfigDAO
from affine.src.behavior_guard.gate import (
    read_gate_snapshot,
    record_deployment_fingerprint,
)
from affine.src.behavior_guard.models import (
    SampleOutcomeEvidence,
    classify_sample_invariant,
    parse_behavior_gate_config,
)
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
        per_host_in_flight: Optional[Dict[str, Any]] = None,
        per_env_host_in_flight: Optional[Dict[str, Any]] = None,
        per_host_budget: int = 0,
        behavior_gate_dao: Any = None,
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
        # Per-host counters shared with manager + sibling workers.
        # Acquired atomically before each evaluate to cap a single
        # sglang's in-flight requests.
        self._per_host_in_flight: Dict[str, Any] = per_host_in_flight or {}
        self._per_env_host_in_flight: Dict[str, Any] = (
            per_env_host_in_flight or {}
        )
        self._per_host_budget = int(per_host_budget or 0)
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
        # Injected in focused tests; production initializes the DynamoDB
        # adapter after the subprocess has created its async client.
        self._behavior_gate_dao = behavior_gate_dao

        self._tasks_by_deployment: Dict[str, set] = {}
        self._last_behavior_gate_log: Dict[str, str] = {}
        # Repeated distinct-task runtime invariants can revoke a passed
        # preflight.  If that final control-plane write is temporarily
        # unavailable, keep the exact deployment fingerprint fail-closed and
        # retry before any later gate read can release traffic.
        self._runtime_gate_quarantine: Dict[str, Dict[str, Any]] = {}

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
        if self._behavior_gate_dao is None:
            from affine.database.dao.behavior_gate import BehaviorGateDAO
            self._behavior_gate_dao = BehaviorGateDAO()
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
                        if t.cancelled():
                            continue  # cancelled by _cancel_stale_dispatches
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

        predeployed = await self._state.get_predeployed_challengers()
        self._cancel_stale_dispatches(
            _collect_current_deployment_ids(champion, battle, predeployed)
        )

        # Challenger deployments must pass the independent, bounded protocol
        # probe before this env can fan out benchmark tasks.  Champion traffic
        # is deliberately unaffected: the gate is an admission check for a
        # newly deployed challenger, not a global inference kill switch.
        gate_config = await self._behavior_gate_config()
        battle_allowed = True
        blocked_deployment_ids: set[str] = set()
        if battle is not None:
            battle_allowed = await self._behavior_gate_allows(
                battle, gate_config,
            )
            if not battle_allowed:
                blocked_deployment_ids.update(_record_deployment_ids(battle))
        allowed_predeployed = []
        for record in predeployed:
            if await self._behavior_gate_allows(record, gate_config):
                allowed_predeployed.append(record)
            else:
                blocked_deployment_ids.update(_record_deployment_ids(record))
        if blocked_deployment_ids:
            self._cancel_dispatches_for_deployments(blocked_deployment_ids)

        # Single-instance providers deliberately clear champion.deployment_id
        # at battle start (the host is now serving the challenger). In that
        # state we still need to dispatch challenger samples. Still run the
        # stale-dispatch sweep first so endpoint scale-down can cancel
        # already-launched work before there are any new serving URLs.
        predeployed_urls = [
            url
            for record in allowed_predeployed
            for url in _base_urls(record.deployments, record.base_url)
        ]
        if not champion_urls and not battle_urls and not predeployed_urls:
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
                model_type=champion.model_type,
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

        # Shadow-run envs (enabled_for_scoring=false) don't feed DECIDE,
        # so they don't need challenger samples to overlap champion's.
        # Without this carve-out a shadow env injected mid-battle would
        # never dispatch — the current champion has already let go of
        # its endpoint (single-instance provider), so champ_done stays
        # empty and the overlap-gated loop below skips every task.
        is_shadow = not env_cfg.enabled_for_scoring

        if battle is not None and battle_urls and battle_allowed:
            chal_done = await self._samples.read_scores_for_tasks(
                battle.challenger.hotkey, battle.challenger.revision,
                self.env, task_ids, refresh_block=refresh_block,
            )
            chal_progress = (
                len(chal_done) if is_shadow
                else len(champ_done.keys() & chal_done.keys())
            )
            if chal_progress < sampling_count:
                for idx, tid in enumerate(task_ids):
                    tid_int = int(tid)
                    if not is_shadow and tid_int not in champ_done:
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

        # Pre-deployed challengers: same gating as ``battle.challenger``,
        # each on its currently assigned endpoint.
        for record in allowed_predeployed:
            pre_urls = _base_urls(
                record.deployments, record.base_url,
            )
            if not pre_urls:
                continue
            pre_done = await self._samples.read_scores_for_tasks(
                record.challenger.hotkey, record.challenger.revision,
                self.env, task_ids, refresh_block=refresh_block,
            )
            pre_progress = (
                len(pre_done) if is_shadow
                else len(champ_done.keys() & pre_done.keys())
            )
            if pre_progress >= sampling_count:
                continue                  # enough samples; let scheduler decide
            for idx, tid in enumerate(task_ids):
                tid_int = int(tid)
                if not is_shadow and tid_int not in champ_done:
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
        launched = 0
        for key, miner, task_id, base_url, dep_id in candidates:
            # In-place promotion persists current_battle before pruning the
            # pre-deployed list. A scheduler crash can therefore expose the
            # same subject through both collections for one poll. Candidate
            # construction is intentionally simple; fence duplicates at the
            # point where ownership of the in-flight key becomes authoritative.
            if key in in_flight_keys:
                continue
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
            if dep_id:
                self._tasks_by_deployment.setdefault(dep_id, set()).add(t)
            launched += 1
        return launched

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

    async def _behavior_gate_config(self):
        getter = getattr(self._state, "get_behavior_gate_config", None)
        raw = await getter() if getter is not None else {}
        return parse_behavior_gate_config(raw)

    async def _behavior_gate_allows(self, record, config) -> bool:
        """Return whether ``record`` may dispatch in this worker's env.

        Shadow and disabled modes are fail-open by definition.  Enforce mode
        is fail-closed for a configured env: a missing table row, transient DB
        read error, or non-final verdict keeps expensive tasks at zero rather
        than recreating the 600-way timeout fan-out.
        """
        if not config.enforces or not config.gates_environment(self.env):
            return True
        fingerprint = record_deployment_fingerprint(record, config)
        pending_failure = self._runtime_gate_quarantine.get(fingerprint)
        if pending_failure is not None:
            write_outcome = await self._write_runtime_invariant_failure(
                fingerprint,
                pending_failure,
            )
            if write_outcome == "sealed":
                self._log_behavior_gate_state(
                    record,
                    "sealed",
                    "promotion_sealed",
                )
                # The scheduler has established the challenger-evidence
                # cutoff.  Do not launch work whose result can no longer
                # affect that decision; the record will shortly become the
                # champion or remain blocked for crash recovery.
                return False
            self._log_behavior_gate_state(
                record,
                "failed" if write_outcome == "failed" else "quarantined",
                (
                    pending_failure["reason_code"]
                    if write_outcome == "failed"
                    else "runtime_invariant_write_pending"
                ),
            )
            # A successful retry durably changed the verdict to failed; an
            # unsuccessful retry remains locally quarantined.  Neither state
            # may release more benchmark work.
            return False
        if self._behavior_gate_dao is None:
            status = "dao_unavailable"
            self._log_behavior_gate_state(record, status, "awaiting_preflight")
            return False
        try:
            snapshot = await read_gate_snapshot(
                self._behavior_gate_dao, record, config,
            )
        except Exception as exc:
            self._log_behavior_gate_state(
                record,
                "read_error",
                type(exc).__name__,
            )
            return False
        self._log_behavior_gate_state(
            record, snapshot.status.value, snapshot.reason,
        )
        if snapshot.row and snapshot.row.get("promotion_sealed_at") is not None:
            self._log_behavior_gate_state(
                record, "sealed", "promotion_sealed",
            )
            return False
        return snapshot.passed

    def _log_behavior_gate_state(
        self, record, status: str, reason: str,
    ) -> None:
        key = (
            f"{record.challenger.hotkey}:{record.challenger.revision}:"
            f"{record.deployment_id}"
        )
        marker = f"{status}:{reason}"
        if self._last_behavior_gate_log.get(key) == marker:
            return
        self._last_behavior_gate_log[key] = marker
        log = logger.info if status == "passed" else logger.warning
        log(
            f"[{self.env}] behavior gate uid={record.challenger.uid} "
            f"status={status} reason={reason}; "
            f"dispatch={'allowed' if status == 'passed' else 'blocked'}"
        )

    def _cancel_dispatches_for_deployments(self, deployment_ids: set[str]) -> None:
        """Cancel shadow-launched calls when enforce mode becomes active."""
        for deployment_id in deployment_ids:
            bucket = self._tasks_by_deployment.pop(deployment_id, set())
            cancelled = 0
            for task in bucket:
                if not task.done():
                    task.cancel()
                    cancelled += 1
            if cancelled:
                logger.warning(
                    f"[{self.env}] cancelled {cancelled} dispatch"
                    f"{'es' if cancelled != 1 else ''} while behavior gate "
                    f"blocks deployment={deployment_id}"
                )

    def _cancel_stale_dispatches(self, current_ids: set) -> None:
        """Cancel tasks for any deployment_id not in ``current_ids``;
        GC done tasks from the live buckets."""
        for dep_id in list(self._tasks_by_deployment):
            bucket = self._tasks_by_deployment[dep_id]
            if dep_id not in current_ids:
                cancelled = 0
                for t in bucket:
                    if not t.done():
                        t.cancel()
                        cancelled += 1
                del self._tasks_by_deployment[dep_id]
                if cancelled:
                    logger.info(
                        f"[{self.env}] cancelled {cancelled} stale "
                        f"dispatch{'es' if cancelled != 1 else ''} for "
                        f"deployment={dep_id}"
                    )
                continue
            live = {t for t in bucket if not t.done()}
            if live:
                self._tasks_by_deployment[dep_id] = live
            else:
                del self._tasks_by_deployment[dep_id]

    async def _evaluate_and_persist(
        self, *, miner: MinerSnapshot, task_id: int, base_url: str,
        refresh_block: int,
        expected_deployment_id: Optional[str] = None,
    ) -> None:
        """Run one (env, task_id, miner) through affinetes, write result
        tagged with the task-id pool's ``refresh_block`` so the scheduler
        only counts current-refresh samples.

        Failure handling distinguishes two cases:

          - **Exceptions** (transport, parse, HTTP status, env crash
            mid-flight): they have no structured scoring disposition.
            **Don't persist** — let the next ``_tick`` re-attempt this
            task_id. The shuffle keeps us from always blocking on the
            same flaky task. The 10% buffer absorbs permanent infra-only
            failures; beyond that operators step in.
          - A non-raising evaluate with ``extra.valid_for_scoring=False``:
            the environment returned an invalid attempt. **Don't persist** —
            it must not count toward completion or comparison.
            ``success=False`` alone is not enough; environments also use it
            for legitimate zero-score model results.
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

    def _increment_env_slot(self) -> None:
        self.metrics.tasks_in_flight += 1
        if self._in_flight_value is not None:
            self._in_flight_value.value += 1

    def _decrement_env_slot(self) -> None:
        self.metrics.tasks_in_flight = max(
            0, self.metrics.tasks_in_flight - 1,
        )
        if self._in_flight_value is not None:
            self._in_flight_value.value = max(
                0, int(self._in_flight_value.value) - 1,
            )

    async def _acquire_dispatch_slot(
        self, *, expected_deployment_id: Optional[str] = None,
    ) -> Optional[str]:
        """env cap → per-host cap → global sem, re-checking both caps
        after the global acquire (manager may have lowered either).

        Returns the acquired host name, if a per-host slot was taken.
        """
        host = _host_from_deployment(expected_deployment_id)
        host_value = self._per_host_in_flight.get(host) if host else None
        while True:
            while self._current_env_in_flight() >= self._current_env_cap():
                await asyncio.sleep(0.05)
            while (
                host_value is not None
                and self._per_host_budget > 0
                and int(host_value.value) >= self._per_host_budget
            ):
                await asyncio.sleep(0.05)
            await self._acquire_global_slot()
            self._increment_env_slot()
            if self._current_env_in_flight() > self._current_env_cap():
                self._decrement_env_slot()
                self._release_global_slot()
                await asyncio.sleep(0.05)
                continue
            if (
                host_value is not None
                and self._per_host_budget > 0
                and not self._try_acquire_host_slot(host)
            ):
                self._decrement_env_slot()
                self._release_global_slot()
                await asyncio.sleep(0.05)
                continue
            if host_value is not None and self._per_host_budget > 0:
                return host
            return None

    def _try_acquire_host_slot(self, host: Optional[str]) -> bool:
        """Atomically reserve one per-host slot."""
        if not host or self._per_host_budget <= 0:
            return False
        host_value = self._per_host_in_flight.get(host)
        if host_value is None:
            return False
        env_host_value = self._per_env_host_in_flight.get(host)
        with host_value.get_lock():
            if int(host_value.value) >= self._per_host_budget:
                return False
            host_value.value += 1
            if env_host_value is not None:
                with env_host_value.get_lock():
                    env_host_value.value += 1
        return True

    def _release_host_slot(self, host: Optional[str]) -> None:
        if not host:
            return
        host_value = self._per_host_in_flight.get(host)
        if host_value is None:
            return
        env_host_value = self._per_env_host_in_flight.get(host)
        with host_value.get_lock():
            host_value.value = max(0, int(host_value.value) - 1)
            if env_host_value is not None:
                with env_host_value.get_lock():
                    env_host_value.value = max(0, int(env_host_value.value) - 1)

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

    async def _current_challenger_record(
        self,
        *,
        miner: MinerSnapshot,
        expected_deployment_id: Optional[str],
    ) -> Any:
        """Return the current battle/predeploy record for this exact rollout.

        Runtime invariants are admission evidence, so they must never mutate a
        champion's gate row.  Requiring a deployment token also keeps legacy
        URL-only dispatches from turning unverifiable telemetry into a loss.
        This lookup intentionally happens after ``_validate_or_drop`` and
        re-validates the role/token immediately before the gate write.
        """
        if expected_deployment_id is None:
            return None

        champion = await self._state.get_champion()
        if champion is not None and _same_subject(champion, miner):
            return None

        battle = await self._state.get_battle()
        if (
            battle is not None
            and _same_subject(battle.challenger, miner)
            and _record_has_deployment(battle, expected_deployment_id)
        ):
            return battle

        for record in await self._state.get_predeployed_challengers():
            if (
                _same_subject(record.challenger, miner)
                and _record_has_deployment(record, expected_deployment_id)
            ):
                return record
        return None

    async def _write_runtime_invariant_failure(
        self,
        fingerprint: str,
        failure: Dict[str, Any],
    ) -> str:
        """Commit a runtime failure or resolve a promotion-seal race.

        Returns ``failed`` when the terminal failure is durable, ``sealed``
        when promotion's atomic cutoff won, and ``pending`` when neither
        outcome can be confirmed and local quarantine must remain active.
        """
        try:
            if self._behavior_gate_dao is None:
                raise RuntimeError("behavior gate DAO unavailable")
            committed = await self._behavior_gate_dao.fail_runtime_invariant(
                failure["hotkey"],
                failure["revision"],
                failure["policy_version"],
                fingerprint,
                reason_code=failure["reason_code"],
                evidence=failure["evidence"],
                counts=failure["counts"],
            )
            if not committed:
                verdict = await self._behavior_gate_dao.get_verdict(
                    failure["hotkey"],
                    failure["revision"],
                    failure["policy_version"],
                    fingerprint,
                )
                if (
                    verdict
                    and verdict.get("status") == "passed"
                    and verdict.get("promotion_sealed_at") is not None
                ):
                    self._runtime_gate_quarantine.pop(fingerprint, None)
                    logger.info(
                        f"[{self.env}] runtime invariant cutoff already sealed "
                        f"uid={failure['uid']}"
                    )
                    return "sealed"
                raise RuntimeError("runtime invariant write was not confirmed")
        except Exception as exc:
            self._runtime_gate_quarantine[fingerprint] = failure
            # Evidence and exception text can contain model-controlled data;
            # expose only stable identifiers and the exception class.
            logger.error(
                f"[{self.env}] runtime invariant gate write failed "
                f"uid={failure['uid']}: {type(exc).__name__}"
            )
            return "pending"

        self._runtime_gate_quarantine.pop(fingerprint, None)
        return "failed"

    async def _runtime_invariant_blocks_persist(
        self,
        *,
        miner: MinerSnapshot,
        task_id: int,
        expected_deployment_id: Optional[str],
        score: float,
        extra: Dict[str, Any],
    ) -> bool:
        """Record a UID208-style impossible positive result.

        Missing or non-numeric telemetry is unknown rather than zero.  Every
        anomaly is recorded as a distinct-task observation; only a repeated
        identical signature closes the gate.  Shadow mode preserves the
        sample, while enforce mode suppresses invalid evidence immediately.
        """
        evidence = _sample_outcome_evidence(score=score, extra=extra)
        classification = classify_sample_invariant(evidence)
        if classification is None:
            return False

        config = await self._behavior_gate_config()
        if not config.gates_environment(self.env):
            return False
        record = await self._current_challenger_record(
            miner=miner,
            expected_deployment_id=expected_deployment_id,
        )
        if record is None:
            return False

        fingerprint = record_deployment_fingerprint(record, config)
        observation_evidence = {
            "environment": self.env,
            "score": score,
            "commands_executed": evidence.commands_executed,
            "llm_call_count": evidence.llm_call_count,
            "total_tokens": evidence.total_tokens,
            "output_bytes": evidence.output_bytes,
        }
        signature_hash = _runtime_signature_hash(
            environment=self.env,
            evidence=evidence,
            classification=classification.value,
        )
        task_hash = _runtime_task_hash(
            environment=self.env,
            task_id=task_id,
        )
        threshold = config.runtime_violations_to_fail
        try:
            if self._behavior_gate_dao is None:
                raise RuntimeError("behavior gate DAO unavailable")
            observation_count = int(
                await self._behavior_gate_dao.record_runtime_invariant_observation(
                    miner.hotkey,
                    miner.revision,
                    config.policy_version,
                    fingerprint,
                    signature_hash=signature_hash,
                    task_hash=task_hash,
                    classification=classification.value,
                    evidence=observation_evidence,
                    threshold=threshold,
                )
            )
        except Exception as exc:
            # The invalid sample still cannot become benchmark evidence in
            # enforce mode, but one uncertain observation is not sufficient
            # to quarantine or terminally fail the deployment.
            logger.error(
                f"[{self.env}] runtime invariant observation write failed "
                f"uid={miner.uid}: {type(exc).__name__}"
            )
            return config.enforces

        reason = "runtime_positive_score_zero_activity"
        if observation_count < threshold:
            logger.warning(
                f"[{self.env}] runtime invariant uid={miner.uid} "
                f"task_id={task_id} reason={reason} "
                f"observations={observation_count}/{threshold}; "
                f"mode={config.mode.value}; "
                f"persist={'blocked' if config.enforces else 'shadow-allowed'}"
            )
            return config.enforces

        failure = {
            "uid": miner.uid,
            "hotkey": miner.hotkey,
            "revision": miner.revision,
            "policy_version": config.policy_version,
            "reason_code": reason,
            # Keep the retry payload to the same compact, allowlisted scalar
            # evidence accepted by BehaviorGateDAO.  Raw output, errors,
            # prompts, and tool arguments never enter the quarantine.
            "evidence": observation_evidence,
            "counts": {
                "total": observation_count,
                "strikes": observation_count,
                classification.value: observation_count,
            },
        }
        await self._write_runtime_invariant_failure(fingerprint, failure)

        logger.warning(
            f"[{self.env}] runtime invariant uid={miner.uid} "
            f"task_id={task_id} reason={reason} "
            f"observations={observation_count}/{threshold}; "
            f"mode={config.mode.value}; "
            f"persist={'blocked' if config.enforces else 'shadow-allowed'}"
        )
        return config.enforces

    async def _evaluate_and_persist_gated(
        self, *, miner: MinerSnapshot, task_id: int, base_url: str,
        refresh_block: int, miner_obj: "_Miner",
        expected_deployment_id: Optional[str] = None,
    ) -> None:
        acquired_host = await self._acquire_dispatch_slot(
            expected_deployment_id=expected_deployment_id,
        )
        started = time.monotonic()
        try:
            try:
                result = await self._env_executor.evaluate(
                    miner=miner_obj, task_id=task_id,
                )
            except Exception as e:
                latency_s = time.monotonic() - started
                latency_ms = int(latency_s * 1000)
                error_brief = str(e).replace("\n", " ").replace("\r", " ")[:200]
                logger.info(
                    f"[FAILED] U{miner.uid:<4} │ {self.env:<20} │     FAILED │ "
                    f"task_id={task_id:<8} │ {latency_s:6.3f}s │ {type(e).__name__}: {error_brief}"
                )
                self.metrics.record_completion(success=False, latency_ms=latency_ms)
                return
        finally:
            self._decrement_env_slot()
            self._release_host_slot(acquired_host)
            self._release_global_slot()

        score = float(getattr(result, "score", 0.0))
        success = bool(getattr(result, "success", True))
        error = getattr(result, "error", None)
        extra = dict(getattr(result, "extra", {}) or {})
        valid_for_scoring = extra.get("valid_for_scoring")
        malformed_disposition = (
            "valid_for_scoring" in extra
            and not isinstance(valid_for_scoring, bool)
        )
        latency_ms = int((time.monotonic() - started) * 1000)
        if not await self._validate_or_drop(
            miner=miner,
            task_id=task_id,
            expected_deployment_id=expected_deployment_id,
            latency_ms=latency_ms,
        ):
            return
        if valid_for_scoring is False or malformed_disposition:
            detail = extra.get("failure_detail") or error or extra.get("error")
            error_brief = str(detail or "environment marked attempt invalid").replace(
                "\n", " "
            ).replace("\r", " ")[:200]
            disposition = (
                "malformed valid_for_scoring"
                if malformed_disposition
                else "valid_for_scoring=false"
            )
            logger.info(
                f"[FAILED] U{miner.uid:<4} │ {self.env:<20} │     INVALID │ "
                f"task_id={task_id:<8} │ {latency_ms / 1000.0:6.3f}s │ "
                f"{disposition}: {error_brief}"
            )
            self.metrics.record_completion(success=False, latency_ms=latency_ms)
            return
        if (error or extra.get("error")) and valid_for_scoring is not True:
            error_brief = str(error or extra.get("error")).replace(
                "\n", " "
            ).replace("\r", " ")[:200]
            logger.info(
                f"[FAILED] U{miner.uid:<4} │ {self.env:<20} │     INVALID │ "
                f"task_id={task_id:<8} │ {latency_ms / 1000.0:6.3f}s │ {error_brief}"
            )
            self.metrics.record_completion(success=False, latency_ms=latency_ms)
            return
        if await self._runtime_invariant_blocks_persist(
            miner=miner,
            task_id=task_id,
            expected_deployment_id=expected_deployment_id,
            score=score,
            extra=extra,
        ):
            self.metrics.record_completion(success=False, latency_ms=latency_ms)
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


def _host_from_deployment(dep_id: Optional[str]) -> Optional[str]:
    """``ssh:<host>:<container>`` → host. ``None`` for any other shape."""
    if not dep_id or not dep_id.startswith("ssh:"):
        return None
    parts = dep_id.split(":")
    return parts[1] if len(parts) >= 3 and parts[1] else None


def _collect_current_deployment_ids(champion, battle, predeployed) -> set:
    """Union of deployment_ids across champion + battle + pre-deployed."""
    ids: set = set()
    for record in (champion, battle, *predeployed):
        if record is None:
            continue
        if getattr(record, "deployment_id", None):
            ids.add(record.deployment_id)
        for d in getattr(record, "deployments", []) or []:
            if d.deployment_id:
                ids.add(d.deployment_id)
    return ids


def _record_deployment_ids(record) -> set[str]:
    return _collect_current_deployment_ids(None, record, [])


def _same_subject(left: Any, right: Any) -> bool:
    return (
        getattr(left, "hotkey", None) == getattr(right, "hotkey", None)
        and getattr(left, "revision", None) == getattr(right, "revision", None)
    )


def _record_has_deployment(record: Any, deployment_id: str) -> bool:
    if getattr(record, "deployment_id", None) == deployment_id:
        return True
    return any(
        getattr(deployment, "deployment_id", None) == deployment_id
        for deployment in (getattr(record, "deployments", ()) or ())
    )


def _sample_outcome_evidence(
    *, score: float, extra: Dict[str, Any],
) -> SampleOutcomeEvidence:
    usage = extra.get("usage")
    total_tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
    return SampleOutcomeEvidence(
        score=score,
        commands_executed=extra.get("commands_executed"),
        llm_call_count=extra.get("llm_call_count"),
        total_tokens=total_tokens,
        output_bytes=extra.get("output_bytes"),
        terminated_reason=extra.get("terminated_reason"),
    )


def _runtime_signature_hash(
    *,
    environment: str,
    evidence: SampleOutcomeEvidence,
    classification: str,
) -> str:
    """Hash the complete invariant signature without retaining raw fields."""
    return _stable_runtime_hash({
        "kind": "runtime-invariant-signature-v1",
        "environment": environment,
        "score": evidence.score,
        "terminated_reason": (
            str(evidence.terminated_reason)
            .replace("\n", " ")
            .replace("\r", " ")[:256]
            if evidence.terminated_reason is not None
            else None
        ),
        "commands_executed": evidence.commands_executed,
        "llm_call_count": evidence.llm_call_count,
        "total_tokens": evidence.total_tokens,
        "output_bytes": evidence.output_bytes,
        "classification": classification,
    })


def _runtime_task_hash(*, environment: str, task_id: int) -> str:
    """Hash task identity separately so retries dedupe without raw IDs."""
    return _stable_runtime_hash({
        "kind": "runtime-invariant-task-v1",
        "environment": environment,
        "task_id": int(task_id),
    })


def _stable_runtime_hash(payload: Dict[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


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
