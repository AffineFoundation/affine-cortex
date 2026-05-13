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
    ):
        self.worker_id = worker_id
        self.env = env
        self.max_concurrent = max_concurrent
        self.poll_interval_sec = poll_interval_sec
        self.idle_sleep_sec = idle_sleep_sec

        self.running = False
        self.metrics = WorkerMetrics(worker_id=worker_id, env=env)

        self._env_executor = None
        self._state: Optional[StateStore] = None
        self._samples: Optional[SampleResultsAdapter] = None

    # ---- lifecycle ---------------------------------------------------------

    async def initialize(self) -> None:
        safe_log(f"[{self.env}] worker init", "INFO")
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

    async def _tick(self) -> bool:
        """One poll cycle. Returns True if any sampling work was attempted."""
        task_state = await self._state.get_task_state()
        if task_state is None:
            return False
        task_ids = task_state.task_ids.get(self.env) or []
        if not task_ids:
            return False
        refresh_block = task_state.refreshed_at_block

        champion = await self._state.get_champion()
        champion_urls = _base_urls(champion.deployments if champion else [], champion.base_url if champion else None)
        if champion is None or not champion_urls:
            return False

        pending: List[tuple] = []  # (miner_snapshot, task_id, base_url)
        # Champion's missing samples for current pool (current refresh).
        for idx, tid in enumerate(task_ids):
            if not await self._samples.has_sample(
                champion.hotkey, champion.revision, self.env, tid,
                refresh_block=refresh_block,
            ):
                pending.append((
                    MinerSnapshot(
                        uid=champion.uid, hotkey=champion.hotkey,
                        revision=champion.revision, model=champion.model,
                    ),
                    tid, _pick_url(champion_urls, tid, idx),
                ))

        # Battle in flight → also fill challenger's missing samples.
        battle = await self._state.get_battle()
        battle_urls = _base_urls(battle.deployments if battle else [], battle.base_url if battle else None)
        if battle is not None and battle_urls:
            for idx, tid in enumerate(task_ids):
                if not await self._samples.has_sample(
                    battle.challenger.hotkey, battle.challenger.revision,
                    self.env, tid, refresh_block=refresh_block,
                ):
                    pending.append((battle.challenger, tid, _pick_url(battle_urls, tid, idx)))

        if not pending:
            return False

        # Shuffle so the same slow-tail task_ids don't always end up last
        # across ticks / executor restarts. The task pool's deterministic
        # order would otherwise pin which task is the slowest to finish.
        # The contest's 10% buffer + overlap-readiness can then release
        # ``decide`` sooner once enough overlap appears anywhere in the pool.
        random.shuffle(pending)

        sem = asyncio.Semaphore(self.max_concurrent)

        async def _run_one(miner: MinerSnapshot, task_id: int, base_url: str) -> None:
            async with sem:
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

        Failure handling distinguishes two cases:

          - ``evaluate`` raises an exception (network glitch, Targon 5xx,
            container restart mid-call): infrastructure-level failure.
            **Don't persist** — let the next ``_tick`` re-attempt this
            task_id. The shuffle keeps us from always blocking on the
            same flaky task. Permanent failures (>10% of pool) eventually
            exceed the refresh's buffer and operator intervention is
            required.
          - ``evaluate`` returns ``success=False``: semantic failure (the
            env's verdict that the miner's response was wrong). Persist
            with score=0 — that's the env's judgment, not a retry signal.
        """
        miner_obj = _Miner(
            hotkey=miner.hotkey, model=miner.model, revision=miner.revision,
            base_url=base_url,
        )
        started = time.monotonic()
        try:
            result = await self._env_executor.evaluate(
                miner=miner_obj, task_id=task_id,
            )
        except Exception as e:
            latency_ms = int((time.monotonic() - started) * 1000)
            logger.warning(
                f"[{self.env}] task_id={task_id} miner=uid{miner.uid} evaluate raised: "
                f"{type(e).__name__}: {e}; will retry next tick"
            )
            self.metrics.record_completion(success=False, latency_ms=latency_ms)
            return

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


class _Miner:
    """Duck-typed miner shim SDKEnvironment.evaluate expects."""

    __slots__ = ("hotkey", "model", "revision", "base_url",
                 "inference_model", "slug", "public_base_url")

    def __init__(self, *, hotkey: str, model: str, revision: str, base_url: str):
        self.hotkey = hotkey
        self.model = model
        self.revision = revision
        self.base_url = base_url
        self.inference_model = None
        self.slug = None
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
