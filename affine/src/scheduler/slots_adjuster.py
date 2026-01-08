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
    """Dynamic slots adjuster based on sampling success rate.
    
    Uses sampling_stats from MinerStats table (last_1hour window) to determine
    success rate. No need to query sample_results table.
    
    Adjustment rules:
    - Only adjust miners with >10 samples in last 1 hour
    - Success rate >= 90%: slots + 1 (max 10)
    - Success rate < 50%: slots - 1 (min 3)
    - Adjustment runs every 2 hours
    
    Persistence:
    - sampling_slots stored in MinerStats table
    - slots_last_adjusted_at tracks last adjustment time
    """
    
    DEFAULT_SLOTS = 6
    MIN_SLOTS = 3
    MAX_SLOTS = 10
    ADJUSTMENT_INTERVAL = 21600  # 6 hours in seconds
    MIN_SAMPLES_FOR_ADJUSTMENT = 10
    HIGH_SUCCESS_THRESHOLD = 0.80
    LOW_SUCCESS_THRESHOLD = 0.50
    
    def __init__(
        self,
        miner_stats_dao: Optional[MinerStatsDAO] = None,
        miners_dao: Optional[MinersDAO] = None
    ):
        self.miner_stats_dao = miner_stats_dao or MinerStatsDAO()
        self.miners_dao = miners_dao or MinersDAO()
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the slots adjuster loop."""
        logger.info(
            f"Starting MinerSlotsAdjuster: interval={self.ADJUSTMENT_INTERVAL}s ({self.ADJUSTMENT_INTERVAL//3600}h), "
            f"min_samples={self.MIN_SAMPLES_FOR_ADJUSTMENT}, "
            f"thresholds=({self.LOW_SUCCESS_THRESHOLD}, {self.HIGH_SUCCESS_THRESHOLD}), "
            f"data_window=last_6hours"
        )
        self._running = True
        self._task = asyncio.create_task(self._adjustment_loop())
    
    async def stop(self):
        """Stop the slots adjuster loop."""
        logger.info("Stopping MinerSlotsAdjuster")
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
    
    async def _adjustment_loop(self):
        """Main adjustment loop - runs every 6 hours."""
        # Initial delay to let system stabilize (10 minutes)
        await asyncio.sleep(600)
        
        while self._running:
            try:
                await self._adjust_all_miners()
                await asyncio.sleep(self.ADJUSTMENT_INTERVAL)
            except asyncio.CancelledError:
                logger.info("Slots adjustment loop cancelled")
                break
            except Exception as e:
                logger.error(f"Slots adjustment error: {e}", exc_info=True)
                await asyncio.sleep(600)  # Retry after 10 minutes
    
    async def _adjust_all_miners(self):
        """Adjust slots for all eligible miners."""
        current_time = int(time.time())
        
        # Get all valid miners
        miners = await self.miners_dao.get_valid_miners()
        
        adjusted_count = 0
        skipped_count = 0
        
        for miner in miners:
            try:
                adjusted = await self._adjust_miner_slots(
                    miner,
                    current_time
                )
                if adjusted:
                    adjusted_count += 1
                else:
                    skipped_count += 1
            except Exception as e:
                logger.error(
                    f"Error adjusting slots for miner {miner['hotkey'][:8]}...: {e}"
                )
        
        logger.info(
            f"Slots adjustment completed: adjusted={adjusted_count}, "
            f"skipped={skipped_count}, total={len(miners)}"
        )
    
    async def _adjust_miner_slots(
        self,
        miner: Dict[str, Any],
        current_time: int
    ) -> bool:
        """Adjust slots for a single miner based on success rate.
        
        Uses sampling_stats.last_6hours from MinerStats table.
        
        Args:
            miner: Miner dict with hotkey, revision
            current_time: Current timestamp
            
        Returns:
            True if adjustment was made, False otherwise
        """
        hotkey = miner['hotkey']
        revision = miner['revision']
        
        # Get current stats including slots and sampling_stats
        stats = await self.miner_stats_dao.get_miner_stats(hotkey, revision)
        current_slots = self.DEFAULT_SLOTS
        last_adjusted = 0
        
        if stats:
            current_slots = stats.get('sampling_slots', self.DEFAULT_SLOTS)
            if current_slots is None:
                current_slots = self.DEFAULT_SLOTS
            last_adjusted = stats.get('slots_last_adjusted_at', 0)
            if last_adjusted is None:
                last_adjusted = 0
        
        # Check if adjustment is due (every 6 hours)
        if current_time - last_adjusted < self.ADJUSTMENT_INTERVAL:
            return False
        
        # Get sampling stats from MinerStats.sampling_stats.last_6hours
        sampling_stats = {}
        if stats:
            sampling_stats = stats.get('sampling_stats', {}).get('last_6hours', {})
        
        total_samples = sampling_stats.get('samples', 0)
        successful_samples = sampling_stats.get('success', 0)
        
        # Skip if not enough samples - don't update timestamp so we can check again next cycle
        # This ensures low-activity miners eventually get adjusted when they have enough samples
        if total_samples < self.MIN_SAMPLES_FOR_ADJUSTMENT:
            logger.debug(
                f"Miner {hotkey[:8]}... has {total_samples} samples in last 6 hours, "
                f"skipping adjustment (need >= {self.MIN_SAMPLES_FOR_ADJUSTMENT})"
            )
            return False
        
        # Use pre-calculated success_rate if available, otherwise calculate
        success_rate = sampling_stats.get('success_rate')
        if success_rate is None:
            success_rate = successful_samples / total_samples if total_samples > 0 else 0
        
        # Determine adjustment
        new_slots = current_slots
        action = "unchanged"
        
        if success_rate >= self.HIGH_SUCCESS_THRESHOLD:
            new_slots = min(current_slots + 1, self.MAX_SLOTS)
            if new_slots > current_slots:
                action = "increased"
        elif success_rate < self.LOW_SUCCESS_THRESHOLD:
            new_slots = max(current_slots - 1, self.MIN_SLOTS)
            if new_slots < current_slots:
                action = "decreased"
        
        # Apply adjustment only if slots actually changed
        if action != "unchanged":
            await self.miner_stats_dao.update_sampling_slots(
                hotkey=hotkey,
                revision=revision,
                slots=new_slots,
                adjusted_at=current_time
            )
            logger.info(
                f"Miner {hotkey[:8]}... slots {action}: {current_slots} -> {new_slots} "
                f"(success_rate={success_rate:.1%}, samples={total_samples})"
            )
            return True
        
        return False