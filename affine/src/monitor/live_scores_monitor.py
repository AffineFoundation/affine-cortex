"""
Live scores monitor.

Periodically computes per-(uid, env) live count + running average from
the current refresh_block's sample_results, and writes the result into
``system_config['live_scores']`` for the rank API to read. Per-API-call
queries would cost ``len(valid) × envs`` DDB hits; precomputed on a
fixed cadence instead.

Also records, per non-champion miner, the champion's average on the
(champion ∩ miner) task overlap — matches the comparator's decide-time
basis (see ``flow.py:_decide``), so the CLI can render row-specific
``[lower, upper]`` thresholds that line up with what ``_decide`` would
compute, rather than one shared global number per env.

Storage shape::

    {
      "refreshed_at": <epoch seconds>,
      "refresh_block": <int — task_state.refreshed_at_block>,
      "champion_uid": <int | null>,
      "scores": {
        "<uid>": {
          "<env>": {
            "count": <int>,
            "avg": <float>,
            "champion_overlap_avg": <float>   // omitted on champion + no-overlap
          }
        }
      }
    }
"""

from __future__ import annotations

import asyncio
import time
from statistics import mean
from typing import Any, Dict, List, Optional

from affine.core.setup import logger
from affine.database.dao.miners import MinersDAO
from affine.database.dao.system_config import SystemConfigDAO
from affine.src.scorer.dao_adapters import SampleResultsAdapter
from affine.src.scorer.window_state import (
    StateStore,
    SystemConfigKVAdapter,
    TaskIdState,
)


# Storage key in system_config that holds the precomputed live-scores cache.
LIVE_SCORES_KEY = "live_scores"

# Default cadence: 30 minutes. Long enough that 320+ Query calls per cycle
# is negligible cost; short enough that the rank UI stays useful within a
# single battle window (~hour scale).
DEFAULT_REFRESH_INTERVAL_SECONDS = 1800

# Bound the per-cycle fan-out so we don't open hundreds of concurrent
# DDB connections.
MAX_CONCURRENCY = 32


class LiveScoresMonitor:
    """Refreshes ``system_config['live_scores']`` on a fixed cadence."""

    _instance: Optional["LiveScoresMonitor"] = None
    _lock = asyncio.Lock()

    def __init__(self, refresh_interval_seconds: int = DEFAULT_REFRESH_INTERVAL_SECONDS):
        self.refresh_interval_seconds = refresh_interval_seconds
        self._config_dao = SystemConfigDAO()
        self._state = StateStore(
            SystemConfigKVAdapter(self._config_dao, updated_by="live_scores_monitor")
        )
        self._miners_dao = MinersDAO()
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

    async def refresh_once(self) -> Optional[Dict[str, Any]]:
        """Run one refresh cycle. Returns the payload written, or None if
        skipped (no task pool / no valid miners / no scoring envs)."""
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
        scores = await self._compute_scores(
            valid_miners, scoring_envs, task_state,
            champion_uid=champion_uid,
        )
        elapsed = time.time() - start

        payload = {
            "refreshed_at": int(time.time()),
            "refresh_block": int(task_state.refreshed_at_block),
            "champion_uid": champion_uid,
            "scores": scores,
        }
        await self._config_dao.set_param(
            param_name=LIVE_SCORES_KEY,
            param_value=payload,
            param_type="dict",
            updated_by="live_scores_monitor",
        )
        logger.info(
            f"[LiveScoresMonitor] refreshed {len(scores)} miners × "
            f"{len(scoring_envs)} envs in {elapsed:.1f}s "
            f"(refresh_block={task_state.refreshed_at_block})"
        )
        return payload

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
