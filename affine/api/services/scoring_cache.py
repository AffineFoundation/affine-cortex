"""
Scoring Cache Service

Proactive cache management for /scoring endpoint with full refresh strategy.
Simplified design: always performs full refresh every 5 minutes.
"""

import time
import asyncio
from typing import Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass

from affine.core.setup import logger


class CacheState(Enum):
    """Cache state machine."""
    EMPTY = "empty"
    WARMING = "warming"
    READY = "ready"
    REFRESHING = "refreshing"


@dataclass
class CacheConfig:
    """Cache configuration."""
    refresh_interval: int = 600  # 5 minutes


class ScoringCacheManager:
    """Manages scoring data cache with full refresh strategy."""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        
        # Dual cache data for scoring and sampling ranges
        self._data_scoring: Dict[str, Any] = {}
        self._data_sampling: Dict[str, Any] = {}
        self._state = CacheState.EMPTY
        self._lock = asyncio.Lock()
        
        # Timestamp for cache (both updated together)
        self._updated_at = 0
        
        # Background task
        self._refresh_task: Optional[asyncio.Task] = None
    
    @property
    def state(self) -> CacheState:
        return self._state
    
    async def warmup(self) -> None:
        """Warm up cache on startup."""
        logger.info("Warming up scoring cache (both range types)...")
        
        async with self._lock:
            self._state = CacheState.WARMING
            try:
                await self._full_refresh_both()
                self._state = CacheState.READY
                self._updated_at = int(time.time())
                logger.info(
                    f"Cache warmed up: scoring={len(self._data_scoring)} miners, "
                    f"sampling={len(self._data_sampling)} miners"
                )
            except Exception as e:
                logger.error(f"Failed to warm up cache: {e}", exc_info=True)
                self._state = CacheState.EMPTY
    
    async def get_data(self, range_type: str = "scoring") -> Dict[str, Any]:
        """Get cached scoring data with fallback logic.
        
        Args:
            range_type: Type of range to use ('scoring' or 'sampling')
        
        Non-blocking: Returns cached data immediately when READY or REFRESHING.
        Blocking: Waits for initial warmup when EMPTY or WARMING.
        """
        if range_type not in ("scoring", "sampling"):
            raise ValueError(f"Invalid range_type: {range_type}")
        
        # Select cache based on range_type
        data = self._data_scoring if range_type == "scoring" else self._data_sampling
        
        # Fast path: return cache if ready or refreshing (data can be empty dict)
        if self._state in [CacheState.READY, CacheState.REFRESHING]:
            return data
        
        # Slow path: cache not initialized yet
        if self._state == CacheState.EMPTY:
            async with self._lock:
                # Double check after acquiring lock
                if self._state == CacheState.EMPTY:
                    logger.warning(f"Cache miss for range_type={range_type} - computing synchronously")
                    self._state = CacheState.WARMING
                    try:
                        await self._full_refresh_both()
                        self._state = CacheState.READY
                        self._updated_at = int(time.time())
                        return self._data_scoring if range_type == "scoring" else self._data_sampling
                    except Exception as e:
                        self._state = CacheState.EMPTY
                        raise RuntimeError(f"Failed to compute scoring data: {e}") from e
        
        # Warming in progress - wait and recheck
        if self._state == CacheState.WARMING:
            for _ in range(60):
                await asyncio.sleep(1)
                # Recheck state - may have changed to READY
                if self._state == CacheState.READY:
                    return self._data_scoring if range_type == "scoring" else self._data_sampling
            # Timeout - return whatever we have
            logger.warning(f"Cache warming timeout, returning current data for range_type={range_type}")
            return self._data_scoring if range_type == "scoring" else self._data_sampling
        
        # Fallback: return any available data (should not reach here)
        logger.warning(f"Returning cache in unexpected state for range_type={range_type} (state={self._state})")
        return data
    
    async def start_refresh_loop(self) -> None:
        """Start background refresh loop."""
        self._refresh_task = asyncio.create_task(self._refresh_loop())
    
    async def stop_refresh_loop(self) -> None:
        """Stop background refresh loop."""
        if self._refresh_task:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
    
    async def _refresh_loop(self) -> None:
        """Background refresh loop with full refresh strategy."""
        while True:
            try:
                await asyncio.sleep(self.config.refresh_interval)
                
                # Set refreshing state (non-blocking for API access)
                async with self._lock:
                    if self._state == CacheState.READY:
                        self._state = CacheState.REFRESHING
                
                # Always perform full refresh for both range types
                await self._full_refresh_both()
                
                # Mark ready
                async with self._lock:
                    self._state = CacheState.READY
                    self._updated_at = int(time.time())
                
            except asyncio.CancelledError:
                logger.info("Cache refresh task cancelled")
                break
            except Exception as e:
                logger.error(f"Cache refresh failed: {e}", exc_info=True)
                async with self._lock:
                    if self._state == CacheState.REFRESHING:
                        self._state = CacheState.READY
    
    async def _full_refresh(self) -> None:
        """Execute full refresh (legacy method, now calls _full_refresh_both)."""
        await self._full_refresh_both()
    
    async def _full_refresh_both(self) -> None:
        """Execute full refresh for both scoring and sampling ranges with separate queries."""
        start_time = time.time()
        logger.info("Full refresh started (separate queries for scoring and sampling)")
        
        from affine.database.dao.system_config import SystemConfigDAO
        from affine.database.dao.miners import MinersDAO
        from affine.database.dao.sample_results import SampleResultsDAO
        
        system_config_dao = SystemConfigDAO()
        miners_dao = MinersDAO()
        sample_dao = SampleResultsDAO()
        
        # Get config - fetch ALL environments, not just active ones
        env_ranges_dict = await system_config_dao.get_env_task_ranges()
        
        # Get all environments from env_ranges_dict
        all_envs = list(env_ranges_dict.keys())
        
        if not all_envs:
            self._data_scoring = {}
            self._data_sampling = {}
            logger.info("Full refresh completed: no environments configured")
            return
        
        valid_miners = await miners_dao.get_valid_miners()
        if not valid_miners:
            self._data_scoring = {}
            self._data_sampling = {}
            logger.info("Full refresh completed: no valid miners")
            return
        
        miners_list = [
            {'hotkey': m['hotkey'], 'revision': m['revision']}
            for m in valid_miners
        ]
        
        # Build ranges for scoring and sampling separately
        env_ranges_scoring = {}
        env_ranges_sampling = {}
        
        for env in all_envs:
            scoring_range = env_ranges_dict[env]['scoring_range']
            sampling_range = env_ranges_dict[env]['sampling_range']
            
            scoring_start, scoring_end = scoring_range
            sampling_start, sampling_end = sampling_range
            
            # Only add non-empty ranges
            if scoring_start < scoring_end:
                env_ranges_scoring[env] = (scoring_start, scoring_end)
            if sampling_start < sampling_end:
                env_ranges_sampling[env] = (sampling_start, sampling_end)
        
        # Execute two separate queries to avoid superset range inefficiency
        samples_data_scoring = await sample_dao.get_scoring_samples_batch(
            miners=miners_list,
            env_ranges=env_ranges_scoring
        )
        samples_data_sampling = await sample_dao.get_scoring_samples_batch(
            miners=miners_list,
            env_ranges=env_ranges_sampling
        )
        
        # Assemble results using their respective query data
        self._data_scoring = self._assemble_result(
            valid_miners, all_envs, env_ranges_scoring, samples_data_scoring
        )
        self._data_sampling = self._assemble_result(
            valid_miners, all_envs, env_ranges_sampling, samples_data_sampling
        )
        
        # Log statistics
        elapsed = time.time() - start_time
        combo_count = len(valid_miners) * len(all_envs)
        logger.info(
            f"Full refresh completed: {len(valid_miners)} miners, "
            f"{len(all_envs)} environments, "
            f"{combo_count} miner×env combinations, "
            f"scoring={len(self._data_scoring)} entries, "
            f"sampling={len(self._data_sampling)} entries, "
            f"elapsed={elapsed:.2f}s"
        )
    
    def _assemble_result(
        self,
        miners: list,
        envs: list,
        env_ranges: dict,
        samples_data: dict
    ) -> Dict[str, Any]:
        """Assemble scoring result from query data."""
        result = {}
        
        for miner in miners:
            uid = miner['uid']
            hotkey = miner['hotkey']
            revision = miner['revision']
            key = f"{hotkey}#{revision}"
            
            miner_samples = samples_data.get(key, {})
            
            miner_entry = {
                'hotkey': hotkey,
                'model_revision': revision,
                'model_repo': miner.get('model'),
                'first_block': miner.get('first_block'),
                'env': {}
            }
            
            for env in envs:
                env_samples = miner_samples.get(env, [])
                start_id, end_id = env_ranges.get(env, (0, 0))
                
                if start_id >= end_id:
                    continue
                
                # Filter samples to only include those within the specified range
                samples_list = [
                    {
                        'task_id': int(s['task_id']),
                        'score': s['score'],
                        'task_uuid': s['timestamp'],
                        'timestamp': s['timestamp'],
                    }
                    for s in env_samples
                    if start_id <= int(s['task_id']) < end_id
                ]
                
                expected_count = end_id - start_id
                completed_count = len(samples_list)
                completeness = completed_count / expected_count if expected_count > 0 else 0.0
                
                completed_task_ids = {s['task_id'] for s in samples_list}
                all_task_ids = set(range(start_id, end_id))
                missing_task_ids = sorted(list(all_task_ids - completed_task_ids))[:100]
                
                miner_entry['env'][env] = {
                    'samples': samples_list,
                    'total_count': expected_count,
                    'completed_count': completed_count,
                    'missing_task_ids': missing_task_ids,
                    'completeness': round(completeness, 4)
                }
            
            result[str(uid)] = miner_entry
        
        return result


# Global cache manager instance
_cache_manager = ScoringCacheManager()


# Public API
async def warmup_cache() -> None:
    """Warm up cache on startup."""
    await _cache_manager.warmup()


async def refresh_cache_loop() -> None:
    """Start background refresh loop."""
    await _cache_manager.start_refresh_loop()


async def get_cached_data(range_type: str = "scoring") -> Dict[str, Any]:
    """Get cached scoring data.
    
    Args:
        range_type: Type of range to use ('scoring' or 'sampling')
    """
    return await _cache_manager.get_data(range_type=range_type)