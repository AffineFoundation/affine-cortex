"""
Sampling Statistics Collector

Collects and aggregates sampling statistics in memory with sliding windows,
then syncs to DynamoDB periodically.
"""

import time
import asyncio
from collections import defaultdict, deque
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from affine.core.setup import logger


@dataclass
class SampleEvent:
    """Single sampling event"""
    timestamp: int          # Unix timestamp (seconds)
    success: bool           # Whether the sample succeeded
    error_type: Optional[str]  # Error type: 'rate_limit', 'other', None


class SamplingStatsCollector:
    """Sampling statistics collector with in-memory sliding windows"""
    
    def __init__(self, sync_interval: int = 300, cleanup_interval: int = 3600, max_idle_time: int = 86400):
        """
        Args:
            sync_interval: Sync interval to database (seconds), default 5 minutes
            cleanup_interval: Cleanup interval for idle miners (seconds), default 1 hour
            max_idle_time: Maximum idle time before cleaning up miner events (seconds), default 24 hours
        """
        self.sync_interval = sync_interval
        self.cleanup_interval = cleanup_interval
        self.max_idle_time = max_idle_time
        
        # Data structure: {(hotkey, revision, env): deque[SampleEvent]}
        self._events: Dict[Tuple[str, str, str], deque] = defaultdict(
            lambda: deque(maxlen=10000)
        )
        
        # Track last activity time for each miner to enable cleanup
        self._last_activity: Dict[Tuple[str, str, str], int] = {}
        
        # Sync task
        self._sync_task: Optional[asyncio.Task] = None
        self._running = False
    
    def record_sample(
        self,
        hotkey: str,
        revision: str,
        env: str,
        success: bool,
        error_message: Optional[str] = None
    ):
        """Record a sampling event
        
        Args:
            hotkey: Miner hotkey
            revision: Model revision
            env: Environment name
            success: Whether the sample succeeded
            error_message: Error message (if failed)
        """
        # Classify error type
        error_type = None
        if not success and error_message:
            if "RateLimitError" in error_message or "429" in error_message:
                error_type = "rate_limit"
            else:
                error_type = "other"
        
        event = SampleEvent(
            timestamp=int(time.time()),
            success=success,
            error_type=error_type
        )
        
        key = (hotkey, revision, env)
        self._events[key].append(event)
        self._last_activity[key] = int(time.time())
    
    def _compute_window_stats(
        self,
        events: deque,
        window_seconds: int
    ) -> Dict[str, Any]:
        """Compute sliding window statistics
        
        Args:
            events: Event queue
            window_seconds: Window size (seconds)
            
        Returns:
            Statistics dictionary
        """
        current_time = int(time.time())
        cutoff_time = current_time - window_seconds
        
        samples = 0
        success = 0
        rate_limit_errors = 0
        other_errors = 0
        
        for event in events:
            if event.timestamp >= cutoff_time:
                samples += 1
                if event.success:
                    success += 1
                elif event.error_type == "rate_limit":
                    rate_limit_errors += 1
                elif event.error_type == "other":
                    other_errors += 1
        
        success_rate = success / samples if samples > 0 else 0.0
        samples_per_min = (samples / window_seconds) * 60 if window_seconds > 0 else 0.0
        
        return {
            "samples": samples,
            "success": success,
            "rate_limit_errors": rate_limit_errors,
            "other_errors": other_errors,
            "success_rate": success_rate,
            "samples_per_min": samples_per_min
        }
    
    async def compute_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Compute statistics for all miners
        
        Returns:
            Dict mapping "hotkey#revision" to env_stats
        """
        windows = {
            "last_15min": 15 * 60,
            "last_1hour": 60 * 60,
            "last_6hours": 6 * 60 * 60,
            "last_24hours": 24 * 60 * 60
        }
        
        all_stats = {}
        
        for (hotkey, revision, env), events in self._events.items():
            # Compute statistics for each time window
            window_stats = {}
            for window_name, window_seconds in windows.items():
                window_stats[window_name] = self._compute_window_stats(events, window_seconds)
            
            key = f"{hotkey}#{revision}"
            if key not in all_stats:
                all_stats[key] = {"envs": {}}
            
            all_stats[key]["envs"][env] = window_stats
        
        return all_stats
    
    def _cleanup_idle_miners(self):
        """Cleanup event queues for idle miners to prevent memory leak.
        
        Removes miners that haven't had any activity for max_idle_time seconds.
        """
        current_time = int(time.time())
        cutoff_time = current_time - self.max_idle_time
        
        keys_to_remove = [
            key for key, last_active in self._last_activity.items()
            if last_active < cutoff_time
        ]
        
        for key in keys_to_remove:
            self._events.pop(key, None)
            self._last_activity.pop(key, None)
        
        if keys_to_remove:
            logger.info(f"Cleaned up {len(keys_to_remove)} idle miner event queues")
    
    async def start_sync_loop(self):
        """Start background sync loop"""
        if self._running:
            logger.warning("Sync loop already running")
            return
        
        self._running = True
        self._sync_task = asyncio.create_task(self._sync_loop())
        logger.info(
            f"SamplingStatsCollector sync loop started "
            f"(sync_interval={self.sync_interval}s, cleanup_interval={self.cleanup_interval}s, "
            f"max_idle_time={self.max_idle_time}s)"
        )
    
    async def _sync_loop(self):
        """Background sync loop with periodic cleanup and retry logic"""
        from affine.database.dao.miner_stats import MinerStatsDAO
        dao = MinerStatsDAO()
        
        last_cleanup_time = int(time.time())
        consecutive_failures = 0
        max_consecutive_failures = 3
        
        while self._running:
            try:
                await asyncio.sleep(self.sync_interval)
                
                # Compute statistics
                all_stats = await self.compute_all_stats()
                
                # Batch sync to database with individual error handling
                sync_success = 0
                sync_failures = 0
                
                for miner_key, stats in all_stats.items():
                    try:
                        hotkey, revision = miner_key.split("#", 1)
                        await dao.update_sampling_stats(hotkey, revision, stats["envs"])
                        sync_success += 1
                    except Exception as e:
                        sync_failures += 1
                        logger.error(
                            f"Failed to sync stats for {miner_key}: {e}",
                            exc_info=False
                        )
                
                # Log sync summary
                if sync_success > 0:
                    logger.info(
                        f"Synced stats for {sync_success}/{len(all_stats)} miners to database"
                        + (f" ({sync_failures} failures)" if sync_failures > 0 else "")
                    )
                    consecutive_failures = 0  # Reset on partial success
                elif sync_failures > 0:
                    consecutive_failures += 1
                    logger.warning(
                        f"All {sync_failures} miner stats sync failed "
                        f"(consecutive failures: {consecutive_failures}/{max_consecutive_failures})"
                    )
                
                # Periodic cleanup of idle miners
                current_time = int(time.time())
                if current_time - last_cleanup_time >= self.cleanup_interval:
                    self._cleanup_idle_miners()
                    last_cleanup_time = current_time
                
            except asyncio.CancelledError:
                logger.info("Sync loop cancelled")
                break
            except Exception as e:
                consecutive_failures += 1
                logger.error(
                    f"Sync loop error: {e} "
                    f"(consecutive failures: {consecutive_failures}/{max_consecutive_failures})",
                    exc_info=True
                )
                
                # If too many consecutive failures, increase sleep time
                if consecutive_failures >= max_consecutive_failures:
                    backoff_time = min(self.sync_interval * 2, 600)  # Max 10 minutes
                    logger.warning(
                        f"Too many consecutive failures, backing off for {backoff_time}s"
                    )
                    await asyncio.sleep(backoff_time)
    
    async def stop(self):
        """Stop sync loop"""
        self._running = False
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
        logger.info("SamplingStatsCollector stopped")


# Singleton instance
_stats_collector: Optional[SamplingStatsCollector] = None


def get_stats_collector() -> SamplingStatsCollector:
    """Get singleton stats collector instance"""
    global _stats_collector
    if _stats_collector is None:
        _stats_collector = SamplingStatsCollector()
    return _stats_collector