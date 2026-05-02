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
    
    # Patterns where the chute literally cannot serve the request right
    # now — no warm replicas, billing hold, or deleted. Unlike
    # rate_limit, sending *fewer* concurrent requests does not help; the
    # underlying chute is offline / misconfigured / unfunded. The slots
    # adjuster should ignore this bucket entirely (treat like
    # other_errors, not like load).
    #
    # Checked first so that wrappers carrying both "Eval model
    # unreachable" and a 503 status get classified as unavailable, not
    # as load.
    _UNAVAILABLE_PATTERNS = (
        "402",                              # billing: zero balance
        "zero balance",
        "404",                              # missing chute
        "No matching chute",
        "503",                              # no instances available
        "No instances available",
        "ServiceUnavailableError",          # litellm wrapper for HTTP 503
        "No infrastructure available",      # HTTP 500 capacity-class message
    )

    # Patterns where the chute is online but signaling "too much" —
    # cutting concurrency genuinely helps. Calibrated against real
    # production error strings; intentionally conservative — generic
    # wrappers like ``BackendError`` and ``InternalServerError`` cover
    # too many non-load failures to be safe load signals.
    _LOAD_PATTERNS = (
        "429",                              # explicit rate-limit code
        "RateLimitError",                   # litellm / OpenAI wrapper
        "Infrastructure is at maximum capacity",
        "maximum capacity",
        "Eval model unreachable",           # NAVWORLD/MEMORY wrapper, default to load
        "model unreachable",
        # Empty-reply path raised by NAVWORLD (affinetes qqr env, commits
        # 8b8c2c9 / e75cf64): chute responds 200 but with no content and
        # no tool_calls, or all retries return non-200. Production scan
        # showed this only occurs on overloaded chutes (median 8s latency,
        # concentrated on miners with chute scaling lagging slots) — i.e.
        # cutting concurrency genuinely helps. Without this entry the
        # signal lands in other_errors and slots_adjuster never sees it.
        "Model returned empty reply",
        "Empty LLM response",               # earlier wording from 8b8c2c9
        "returned empty reply",             # broader fallback
    )

    # Patterns indicating a hard wall-clock timeout (model/agent didn't
    # finish within env proxy_timeout or HTTP client timeout). Ambiguous
    # signal — could be overload OR slow model OR slow agent loop — so
    # kept separate from rate_limit to avoid auto-cutting legitimately
    # slow inference. Note "timed out" is a broad match that covers
    # "Method 'evaluate' on environment X timed out" (env proxy timeout)
    # as well as nested LLM/HTTP timeouts.
    _TIMEOUT_PATTERNS = (
        "timed out",
        "ReadTimeout",
        "APITimeoutError",
        "TimeoutError",
        "Agent timeout",
        "agent_timeout",
        "Codex timed out",
        "Pre-fetch timeout",
    )

    @classmethod
    def _classify_error(cls, error_message: str) -> str:
        """Bucket an error_message into unavailable / rate_limit / timeout / other.

        Order matters:
          1. ``unavailable`` first — most specific. A wrapper like
             "Eval model unreachable after 10 retries: 503" should not be
             classified as load just because it nominally says
             "unreachable"; the embedded 503 means the chute has no
             warm replicas, which lower concurrency cannot fix.
          2. ``rate_limit`` next — chute online but at capacity.
          3. ``timeout`` next — wall-clock overrun, ambiguous.
          4. ``other`` — model behavior, scorer bugs, file errors, etc.
        """
        for pat in cls._UNAVAILABLE_PATTERNS:
            if pat in error_message:
                return "unavailable"
        for pat in cls._LOAD_PATTERNS:
            if pat in error_message:
                return "rate_limit"
        for pat in cls._TIMEOUT_PATTERNS:
            if pat in error_message:
                return "timeout"
        return "other"

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
        error_type = None
        if not success and error_message:
            error_type = self._classify_error(error_message)

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
                    "timeout_errors": stats.get("timeout_errors", 0),
                    "unavailable_errors": stats.get("unavailable_errors", 0),
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