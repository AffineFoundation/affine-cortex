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
        warmup_sec: float = 180.0,
        global_sem: Any = None,
        in_flight_value: Any = None,
    ):
        self.worker_id = worker_id
        self.env = env
        self.max_concurrent = max_concurrent
        # Cross-process ``BoundedSemaphore`` (or ``None`` in unit tests /
        # standalone runs). Real backpressure runs through this — see
        # ``_acquire_global_slot`` below. The per-tick local semaphore
        # is sized at ``max_concurrent`` and only exists as a defensive
        # floor against scheduling tens of thousands of coroutines.
        self._global_sem = global_sem
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
        # cache build, memorygym worker boot) can still be running. With
        # 60s the first tick after a restart historically slammed into the
        # tail of that init and produced a flood of empty ReadErrors;
        # 180s gives external data sources + uvicorn worker fanout enough
        # time to settle.
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
        assert self._state and self._samples, "call initialize() first"
        # Publish status to DB in the background so ``af db worker-status``
        # can read live in-flight / succeeded / failed counters.
        status_task = asyncio.create_task(self._publish_status_loop())
        try:
            if self.warmup_sec > 0:
                logger.info(
                    f"[{self.env}] warmup: sleeping {self.warmup_sec:.0f}s before "
                    f"first tick (let env containers finish app-level boot)"
                )
                await asyncio.sleep(self.warmup_sec)
            while self.running:
                try:
                    worked = await self._tick()
                except Exception as e:
                    logger.error(
                        f"[{self.env}] tick raised: {type(e).__name__}: {e}",
                        exc_info=True,
                    )
                    worked = False
                await asyncio.sleep(self.poll_interval_sec if worked else self.idle_sleep_sec)
        finally:
            status_task.cancel()

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

        First failure is logged at WARNING so a misconfigured DAO surfaces
        quickly; subsequent failures (likely the same cause) drop to DEBUG.
        """
        import time as _t
        from affine.src.scorer.window_state import SystemConfigKVAdapter
        from affine.database.dao.system_config import SystemConfigDAO
        kv = SystemConfigKVAdapter(SystemConfigDAO(), updated_by=f"executor-{self.env}")
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
                        f"{type(e).__name__}: {e}"
                    )
                    ever_failed = True
                else:
                    logger.debug(f"[{self.env}] status publish failed: {e}")
            await asyncio.sleep(interval_sec)

    async def _tick(self) -> bool:
        """One poll cycle. Returns True if any sampling work was attempted.

        Sampling policy is asymmetric between subjects:
          - **Champion** keeps draining the full pool (sampling_count *
            1.1 from Stage AF). The extra 10% gives later contests more
            comparison data and a buffer against task-ids that the env
            can't process (~5% expected); a champion sticks around for
            many battles, so investing in deeper samples pays off.
          - **Challenger** stops at the base ``sampling_count``. Once
            the overlap with champion can possibly clear the comparator
            threshold there's no value in continuing — the challenger
            either wins and the rotation moves on, or loses and gets
            torn down.

        The dispatch semaphore (``self.max_concurrent``) is a defensive
        hygiene cap, not a throughput knob — real backpressure comes
        from the env containers downstream. See ``executor/config.py``.
        """
        task_state = await self._state.get_task_state()
        if task_state is None:
            return False
        task_ids = task_state.task_ids.get(self.env) or []
        if not task_ids:
            return False
        refresh_block = task_state.refreshed_at_block

        envs = await self._state.get_environments()
        env_cfg = envs.get(self.env)
        if env_cfg is None:
            return False  # env disabled / removed from config
        sampling_count = env_cfg.sampling_count

        champion = await self._state.get_champion()
        champion_urls = _base_urls(champion.deployments if champion else [], champion.base_url if champion else None)
        if champion is None or not champion_urls:
            return False

        pending: List[tuple] = []  # (miner_snapshot, task_id, base_url)

        # Champion: drain the full pool (no cap). A persistently-failing
        # task_id keeps reappearing in ``pending`` each tick — that's
        # acceptable because env-level failures eat constant low-grade
        # capacity and the 10% buffer + 5% miss tolerance absorbs them.
        champ_snap = MinerSnapshot(
            uid=champion.uid, hotkey=champion.hotkey,
            revision=champion.revision, model=champion.model,
        )
        for idx, tid in enumerate(task_ids):
            if not await self._samples.has_sample(
                champion.hotkey, champion.revision, self.env, tid,
                refresh_block=refresh_block,
            ):
                pending.append((champ_snap, tid, _pick_url(champion_urls, tid, idx)))

        # Battle in flight → cap challenger at ``sampling_count``.
        battle = await self._state.get_battle()
        battle_urls = _base_urls(battle.deployments if battle else [], battle.base_url if battle else None)
        if battle is not None and battle_urls:
            chal_have = await self._samples.count_samples_for_tasks(
                battle.challenger.hotkey, battle.challenger.revision, self.env,
                task_ids, refresh_block=refresh_block,
            )
            if chal_have < sampling_count:
                remaining = sampling_count - chal_have
                for idx, tid in enumerate(task_ids):
                    if remaining <= 0:
                        break
                    if not await self._samples.has_sample(
                        battle.challenger.hotkey, battle.challenger.revision,
                        self.env, tid, refresh_block=refresh_block,
                    ):
                        pending.append((battle.challenger, tid, _pick_url(battle_urls, tid, idx)))
                        remaining -= 1

        if not pending:
            return False

        # Shuffle so the same slow-tail task_ids don't always end up last
        # across ticks / executor restarts. The task pool's deterministic
        # order would otherwise pin which task is the slowest to finish.
        # The contest's 10% buffer + overlap-readiness can then release
        # ``decide`` sooner once enough overlap appears anywhere in the pool.
        random.shuffle(pending)

        # Local asyncio semaphore is just a defensive floor — the real
        # gate is the cross-process ``self._global_sem`` acquired inside
        # ``_evaluate_and_persist``. Local cap >= pending size in normal
        # operation, so this is effectively a no-op except as protection
        # against runaway pending lists.
        local_sem = asyncio.Semaphore(self.max_concurrent)

        async def _run_one(miner: MinerSnapshot, task_id: int, base_url: str) -> None:
            async with local_sem:
                await self._evaluate_and_persist(
                    miner=miner, task_id=task_id, base_url=base_url,
                    refresh_block=refresh_block,
                )

        await asyncio.gather(
            *(_run_one(m, tid, url) for m, tid, url in pending),
            return_exceptions=True,
        )
        return True

    async def _evaluate_and_persist(
        self, *, miner: MinerSnapshot, task_id: int, base_url: str,
        refresh_block: int,
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
          - ``Result.success=False`` from a non-raising evaluate: the
            env returned a structured failure verdict alongside a score.
            Persist whatever score the env supplied (handled below).
        """
        miner_obj = _Miner(
            hotkey=miner.hotkey, model=miner.model, revision=miner.revision,
            base_url=base_url,
        )
        # Wait until the cross-process global semaphore yields a slot.
        # Envs with bigger pending lists submit more concurrent acquire
        # attempts and so win a proportional share of the b300 budget —
        # that's the cross-env priority, no explicit queue needed.
        await self._acquire_global_slot()
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
                    await self._samples.persist(
                        miner_hotkey=miner.hotkey,
                        model_revision=miner.revision,
                        model=miner.model,
                        env=self.env,
                        task_id=task_id,
                        score=0.0,
                        latency_ms=latency_ms,
                        extra={"error": error_brief, "zero_score_reason": "context_overflow"},
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
        if not success:
            extra.setdefault("error", error)
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
