"""
Sampling Statistics Collector

Collects and aggregates sampling statistics using local file-based persistence
with 5-minute granularity, then syncs to DynamoDB periodically.
"""

import time
import asyncio
from typing import Dict, Any, Optional
from affine.core.setup import logger
from affine.api.services.local_stats_store import LocalStatsStore


class SamplingStatsCollector:
    """Sampling statistics collector with local file-based persistence"""
    
    def __init__(self, sync_interval: int = 300, cleanup_interval: int = 3600):
        """
        Args:
            sync_interval: Sync interval to database (seconds), default 5 minutes
            cleanup_interval: Cleanup interval for old files (seconds), default 1 hour
        """
        self.sync_interval = sync_interval
        self.cleanup_interval = cleanup_interval
        
        # Local statistics store (5-minute granularity)
        self._local_store = LocalStatsStore()
        
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
        
        # Record to local store (accumulates in current 5-minute window)
        self._local_store.record_sample(hotkey, revision, env, success, error_type)
    
    async def compute_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Compute statistics for all miners from local files
        
        Returns:
            Dict mapping "hotkey#revision" to env_stats
        """
        # Flush current window before computing
        self._local_store.flush()
        
        windows = {
            "last_15min": 0.25,   # hours
            "last_1hour": 1,
            "last_6hours": 6,
            "last_24hours": 24
        }
        
        all_stats = {}
        
        # Load and aggregate statistics for each time window
        for window_name, hours in windows.items():
            window_stats = self._local_store.load_aggregated_stats(hours=hours)
            
            for miner_key, stats in window_stats.items():
                # Parse miner key: "hotkey#revision#env"
                parts = miner_key.split("#", 2)
                if len(parts) != 3:
                    continue
                
                hotkey, revision, env = parts
                key = f"{hotkey}#{revision}"
                
                if key not in all_stats:
                    all_stats[key] = {"envs": {}}
                
                if env not in all_stats[key]["envs"]:
                    all_stats[key]["envs"][env] = {}
                
                # Calculate derived metrics
                samples = stats["samples"]
                success_rate = stats["success"] / samples if samples > 0 else 0.0
                samples_per_min = (samples / (hours * 60)) if hours > 0 else 0.0
                
                all_stats[key]["envs"][env][window_name] = {
                    "samples": samples,
                    "success": stats["success"],
                    "rate_limit_errors": stats["rate_limit_errors"],
                    "other_errors": stats["other_errors"],
                    "success_rate": success_rate,
                    "samples_per_min": samples_per_min
                }
        
        return all_stats
    
    async def start_sync_loop(self):
        """Start background sync loop"""
        if self._running:
            logger.warning("Sync loop already running")
            return
        
        self._running = True
        self._sync_task = asyncio.create_task(self._sync_loop())
        logger.info(
            f"SamplingStatsCollector started "
            f"(sync_interval={self.sync_interval}s, cleanup_interval={self.cleanup_interval}s)"
        )
    
    async def _sync_loop(self):
        """Background sync loop with periodic cleanup and retry logic"""
        from affine.database.dao.miner_stats import MinerStatsDAO
        from affine.database.dao.miners import MinersDAO
        
        dao = MinerStatsDAO()
        miners_dao = MinersDAO()
        
        last_cleanup_time = int(time.time())
        consecutive_failures = 0
        max_consecutive_failures = 3
        
        while self._running:
            try:
                await asyncio.sleep(self.sync_interval)
                
                # Compute statistics from local files
                all_stats = await self.compute_all_stats()
                
                # Get all miners info for batch update
                all_miners = await miners_dao.get_all_miners()
                miners_dict = {f"{m['hotkey']}#{m['revision']}": m for m in all_miners}
                
                # Batch sync to database with individual error handling
                sync_success = 0
                sync_failures = 0
                
                for miner_key, stats in all_stats.items():
                    try:
                        hotkey, revision = miner_key.split("#", 1)
                        
                        # Update sampling stats
                        await dao.update_sampling_stats(hotkey, revision, stats["envs"])
                        
                        # Update miner basic info (model, rank, weight) if miner exists
                        miner_info = miners_dict.get(miner_key)
                        if miner_info:
                            await dao.update_miner_info(
                                hotkey=hotkey,
                                revision=revision,
                                model=miner_info.get('model', ''),
                                rank=miner_info.get('rank'),
                                weight=miner_info.get('weight'),
                                is_online=miner_info.get('is_valid', False)
                            )
                        
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
                    consecutive_failures = 0
                elif sync_failures > 0:
                    consecutive_failures += 1
                    logger.warning(
                        f"All {sync_failures} miner stats sync failed "
                        f"(consecutive failures: {consecutive_failures}/{max_consecutive_failures})"
                    )
                
                # Periodic cleanup of old files
                current_time = int(time.time())
                if current_time - last_cleanup_time >= self.cleanup_interval:
                    self._local_store.cleanup_old_files(keep_hours=25)
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
                    backoff_time = min(self.sync_interval * 2, 600)
                    logger.warning(
                        f"Too many consecutive failures, backing off for {backoff_time}s"
                    )
                    await asyncio.sleep(backoff_time)
    
    async def stop(self):
        """Stop sync loop and flush remaining data"""
        self._running = False
        
        # Flush current window before stopping
        try:
            self._local_store.flush()
        except Exception as e:
            logger.error(f"Failed to flush stats on stop: {e}")
        
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