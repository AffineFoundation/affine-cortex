"""
Miner Slots Adjuster

Dynamically adjusts per-miner sampling slots based on success rate.
"""

import time
import asyncio
from typing import Optional, Dict, Any

from affine.core.setup import logger
from affine.database.dao.miner_stats import MinerStatsDAO
from affine.database.dao.miners import MinersDAO


class MinerSlotsAdjuster:
    """Conservative-backoff slots adjuster.

    Reads sampling_stats.last_1hour from MinerStats. Tiered, asymmetric
    update — fast multiplicative cuts on failure, slow additive grows
    on success — so an overloaded chute gets relief quickly while a
    recovering chute can't immediately re-trip itself.

      sr >= 0.90       cur + 2
      sr >= 0.70       hold (hysteresis)
      sr >= 0.50       cur - 3
      sr >= 0.30       int(cur * 0.7)
      sr <  0.30       int(cur * 0.5)
    """

    DEFAULT_SLOTS = 20
    MIN_SLOTS = 15
    MAX_SLOTS = 50

    # Per-miner cadence (gate) — each miner adjusted at most once per
    # ADJUSTMENT_INTERVAL. Outer poll runs more often so brand new
    # miners (last_adjusted_at == 0) don't have to wait the full
    # interval for their first adjustment.
    ADJUSTMENT_INTERVAL = 7200      # 2 h
    LOOP_INTERVAL_SECONDS = 600     # 10 min

    MIN_SAMPLES_FOR_ADJUSTMENT = 30

    def __init__(
        self,
        miner_stats_dao: Optional[MinerStatsDAO] = None,
        miners_dao: Optional[MinersDAO] = None,
    ):
        self.miner_stats_dao = miner_stats_dao or MinerStatsDAO()
        self.miners_dao = miners_dao or MinersDAO()
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        logger.info(
            f"Starting MinerSlotsAdjuster: "
            f"cadence={self.ADJUSTMENT_INTERVAL}s, "
            f"poll={self.LOOP_INTERVAL_SECONDS}s, "
            f"min_samples={self.MIN_SAMPLES_FOR_ADJUSTMENT}, "
            f"slots=[{self.MIN_SLOTS}, {self.MAX_SLOTS}]"
        )
        self._running = True
        self._task = asyncio.create_task(self._adjustment_loop())

    async def stop(self):
        logger.info("Stopping MinerSlotsAdjuster")
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _adjustment_loop(self):
        await asyncio.sleep(600)  # initial settle
        while self._running:
            try:
                await self._adjust_all_miners()
                await asyncio.sleep(self.LOOP_INTERVAL_SECONDS)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Slots adjustment error: {e}", exc_info=True)
                await asyncio.sleep(600)

    async def _adjust_all_miners(self):
        current_time = int(time.time())
        miners = await self.miners_dao.get_valid_miners()
        adjusted = 0
        for miner in miners:
            try:
                if await self._adjust_miner_slots(miner, current_time):
                    adjusted += 1
            except Exception as e:
                logger.error(
                    f"Error adjusting slots for miner {miner['hotkey'][:8]}...: {e}"
                )
        logger.info(
            f"Slots adjustment: adjusted={adjusted}, total={len(miners)}"
        )

    async def _adjust_miner_slots(
        self, miner: Dict[str, Any], current_time: int
    ) -> bool:
        hotkey = miner['hotkey']
        revision = miner['revision']

        stats = await self.miner_stats_dao.get_miner_stats(hotkey, revision)
        current_slots = self.DEFAULT_SLOTS
        last_adjusted = 0
        if stats:
            current_slots = stats.get('sampling_slots') or self.DEFAULT_SLOTS
            last_adjusted = stats.get('slots_last_adjusted_at') or 0

        # Floor stale below-MIN slots regardless of cadence/sample volume.
        if current_slots < self.MIN_SLOTS:
            await self.miner_stats_dao.update_sampling_slots(
                hotkey=hotkey, revision=revision,
                slots=self.MIN_SLOTS, adjusted_at=current_time,
            )
            logger.info(
                f"Miner {hotkey[:8]}... slots floored: "
                f"{current_slots} -> {self.MIN_SLOTS}"
            )
            return True

        # Cadence gate. last_adjusted == 0 → first-ever adjustment.
        if (last_adjusted > 0
                and current_time - last_adjusted < self.ADJUSTMENT_INTERVAL):
            return False

        # Use *inference_health* (success / (success + rate_limit_errors))
        # rather than the raw success_rate. Failures unrelated to chute
        # load — model quality (no_tools, empty output), billing (HTTP 402
        # zero balance), missing chute (404), scorer errors — should not
        # cause slots to shrink, since lowering concurrency does not fix
        # any of them. Only HTTP 429 ("Infrastructure at maximum capacity")
        # is a true overload signal: it means the chute literally cannot
        # absorb more requests right now. timeout_errors are intentionally
        # excluded because they are ambiguous (slow model vs. overload).
        sampling_stats = (stats or {}).get('sampling_stats', {}).get('last_1hour', {})
        successful_samples = int(sampling_stats.get('success', 0) or 0)
        rate_limit_errors = int(sampling_stats.get('rate_limit_errors', 0) or 0)
        load_relevant_samples = successful_samples + rate_limit_errors

        if load_relevant_samples < self.MIN_SAMPLES_FOR_ADJUSTMENT:
            return False

        success_rate = successful_samples / load_relevant_samples

        # Tiered backoff. Asymmetric on purpose.
        if success_rate >= 0.90:
            new_slots = current_slots + 2
        elif success_rate >= 0.70:
            new_slots = current_slots
        elif success_rate >= 0.50:
            new_slots = current_slots - 3
        elif success_rate >= 0.30:
            new_slots = int(current_slots * 0.7)
        else:
            new_slots = int(current_slots * 0.5)

        new_slots = max(self.MIN_SLOTS, min(self.MAX_SLOTS, new_slots))

        # Always write the timestamp so the cadence gate fires correctly
        # next cycle, even when slots stayed the same (hold zone).
        await self.miner_stats_dao.update_sampling_slots(
            hotkey=hotkey, revision=revision,
            slots=new_slots, adjusted_at=current_time,
        )

        if new_slots != current_slots:
            action = "increased" if new_slots > current_slots else "decreased"
            logger.info(
                f"Miner {hotkey[:8]}... slots {action}: "
                f"{current_slots} -> {new_slots} "
                f"(inference_health={success_rate:.1%}, "
                f"success={successful_samples}, rate_limit={rate_limit_errors})"
            )
            return True
        return False
