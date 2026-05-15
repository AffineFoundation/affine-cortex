"""
Live scores monitor.

Periodically computes per-(uid, env) live count + running average from
the current refresh_block's sample_results and writes the result onto
each miner's ``miner_stats`` row (``scores_by_env`` field, keyed
by refresh_block).

Why a precompute pass: ``af get-rank`` needs per-(uid, env) score data
for every valid miner, and querying sample_results on every API call
would cost ``len(valid) × envs`` Queries per request. Computing on a
fixed cadence and parking the result on the miner_stats row instead
keeps the API path to one read per miner.

For each non-champion miner the row also carries the champion's avg on
the (champion ∩ miner) task overlap — matches the comparator's
decide-time basis (see ``flow.py:_decide``), so the CLI can render
per-row ``[lower, upper]`` thresholds that line up with what
``_decide`` would compute.

Per-miner storage on ``miner_stats`` (same attribute the comparator
freezes at termination — see :class:`MinerStatsDAO.update_challenge_status`)::

    {
      "scores_refresh_block": <int>,
      "scores_by_env": {
        "<env>": {
          "count": <int>,
          "avg": <float>,
          "champion_overlap_avg": <float>   // omitted on champion + no-overlap
        }
      }
    }

``scores_refresh_block`` lets readers detect "this is for the previous
task pool — drop it" after the scheduler refreshes the pool between
monitor cycles. The conditional write on the DAO side
(``attribute_not_exists(terminated_at_block)``) prevents live cycles
from clobbering the comparator's decide-time snapshot once a miner is
terminated.
"""

from __future__ import annotations

import asyncio
import time
from statistics import mean
from typing import Any, Dict, List, Optional

from affine.core.setup import logger
from affine.database.dao.miner_stats import MinerStatsDAO
from affine.database.dao.miners import MinersDAO
from affine.database.dao.system_config import SystemConfigDAO
from affine.src.scorer.dao_adapters import SampleResultsAdapter
from affine.src.scorer.window_state import (
    StateStore,
    SystemConfigKVAdapter,
    TaskIdState,
)


# Default cadence: 30 minutes. Long enough that 320+ Query calls per cycle
# is negligible cost; short enough that the rank UI stays useful within a
# single battle window (~hour scale).
DEFAULT_REFRESH_INTERVAL_SECONDS = 1800

# Bound the per-cycle fan-out so we don't open hundreds of concurrent
# DDB connections. Applies to both the sample-read pass and the
# per-miner miner_stats write pass.
MAX_CONCURRENCY = 32


class LiveScoresMonitor:
    """Refreshes per-miner ``scores_by_env`` on a fixed cadence."""

    _instance: Optional["LiveScoresMonitor"] = None
    _lock = asyncio.Lock()

    def __init__(self, refresh_interval_seconds: int = DEFAULT_REFRESH_INTERVAL_SECONDS):
        self.refresh_interval_seconds = refresh_interval_seconds
        self._config_dao = SystemConfigDAO()
        self._state = StateStore(
            SystemConfigKVAdapter(self._config_dao, updated_by="live_scores_monitor")
        )
        self._miners_dao = MinersDAO()
        self._stats_dao = MinerStatsDAO()
        self._samples = SampleResultsAdapter()
        self._background_task: Optional[asyncio.Task] = None

    @classmethod
    async def initialize(
        cls, refresh_interval_seconds: int = DEFAULT_REFRESH_INTERVAL_SECONDS,
    ) -> "LiveScoresMonitor":
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls(refresh_interval_seconds=refresh_interval_seconds)
                await cls._instance.start_background_tasks()
        return cls._instance

    async def start_background_tasks(self) -> None:
        if self._background_task and not self._background_task.done():
            return
        self._background_task = asyncio.create_task(self._refresh_loop())

    async def stop_background_tasks(self) -> None:
        if self._background_task and not self._background_task.done():
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
        self._background_task = None

    async def _refresh_loop(self) -> None:
        # Run once immediately at startup so the rank UI doesn't show
        # nothing for the first refresh interval after a deploy.
        while True:
            try:
                await self.refresh_once()
            except Exception as e:
                logger.error(
                    f"[LiveScoresMonitor] refresh failed: {e}", exc_info=True,
                )
            await asyncio.sleep(self.refresh_interval_seconds)

    async def refresh_once(self) -> Optional[int]:
        """Run one refresh cycle. Returns the number of miner_stats rows
        updated, or None if skipped (no task pool / no valid miners /
        no scoring envs)."""
        task_state = await self._state.get_task_state()
        if task_state is None or not task_state.task_ids:
            logger.info(
                "[LiveScoresMonitor] no task_state; skipping cycle"
            )
            return None

        env_configs = await self._state.get_scoring_environments()
        scoring_envs = [env for env in env_configs if task_state.task_ids.get(env)]
        if not scoring_envs:
            logger.info(
                "[LiveScoresMonitor] no scoring envs with task_ids; skipping"
            )
            return None

        valid_miners = await self._miners_dao.get_valid_miners()
        if not valid_miners:
            logger.info("[LiveScoresMonitor] no valid miners; skipping cycle")
            return None

        champion = await self._state.get_champion()
        champion_uid = champion.uid if champion is not None else None

        start = time.time()
        per_miner_scores = await self._compute_scores(
            valid_miners, scoring_envs, task_state,
            champion_uid=champion_uid,
        )
        compute_elapsed = time.time() - start

        # Per-miner write fanout into miner_stats. Concurrency bound
        # mirrors the sample-read pass.
        miner_by_uid = {
            int(m["uid"]): m for m in valid_miners
            if isinstance(m.get("uid"), int)
        }
        sem = asyncio.Semaphore(MAX_CONCURRENCY)
        refresh_block = int(task_state.refreshed_at_block)

        async def _persist(uid_str: str, scores_by_env: Dict[str, Any]) -> None:
            miner = miner_by_uid.get(int(uid_str))
            if miner is None:
                return
            async with sem:
                await self._stats_dao.update_live_scores(
                    hotkey=str(miner["hotkey"]),
                    revision=str(miner["revision"]),
                    scores_by_env=scores_by_env,
                    scores_refresh_block=refresh_block,
                )

        await asyncio.gather(*(
            _persist(uid, env_map) for uid, env_map in per_miner_scores.items()
        ))
        elapsed = time.time() - start
        logger.info(
            f"[LiveScoresMonitor] refreshed {len(per_miner_scores)} miners × "
            f"{len(scoring_envs)} envs in {elapsed:.1f}s "
            f"(compute={compute_elapsed:.1f}s, refresh_block={refresh_block})"
        )
        return len(per_miner_scores)

    async def _compute_scores(
        self,
        valid_miners: List[Dict[str, Any]],
        scoring_envs: List[str],
        task_state: TaskIdState,
        champion_uid: Optional[int] = None,
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Return ``{uid_str: {env: {"count", "avg", "champion_overlap_avg"?}}}``
        for every (valid miner × scoring env) with at least one sample
        row at the current refresh_block. ``champion_overlap_avg``
        omitted on the champion's row and when (champion ∩ miner) is
        empty."""

        sem = asyncio.Semaphore(MAX_CONCURRENCY)

        async def _one(uid: int, hotkey: str, revision: str, env: str):
            task_ids = task_state.task_ids.get(env) or []
            if not task_ids:
                return uid, env, None
            async with sem:
                scored = await self._samples.read_scores_for_tasks(
                    hotkey, revision, env, task_ids,
                    refresh_block=task_state.refreshed_at_block,
                )
            if not scored:
                return uid, env, None
            return uid, env, scored

        coros = []
        for m in valid_miners:
            uid = m.get("uid")
            hotkey = m.get("hotkey")
            revision = m.get("revision")
            if not isinstance(uid, int) or not hotkey or not revision:
                continue
            for env in scoring_envs:
                coros.append(_one(uid, hotkey, str(revision), env))

        # Two-pass: gather raw {task_id: score} maps first so the
        # second pass can intersect each challenger with the champion.
        per_miner_env: Dict[int, Dict[str, Dict[int, float]]] = {}
        for uid, env, scored in await asyncio.gather(*coros):
            if scored is None:
                continue
            per_miner_env.setdefault(uid, {})[env] = scored

        champ_per_env = (
            per_miner_env.get(int(champion_uid))
            if champion_uid is not None else None
        )
        out: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for uid, env_to_scored in per_miner_env.items():
            for env, scored in env_to_scored.items():
                entry: Dict[str, Any] = {
                    "count": len(scored),
                    "avg": float(mean(scored.values())),
                }
                if (
                    champ_per_env is not None
                    and uid != champion_uid
                    and env in champ_per_env
                ):
                    overlap = set(scored.keys()) & set(champ_per_env[env].keys())
                    if overlap:
                        entry["champion_overlap_avg"] = float(
                            mean(champ_per_env[env][t] for t in overlap)
                        )
                out.setdefault(str(uid), {})[env] = entry
        return out
