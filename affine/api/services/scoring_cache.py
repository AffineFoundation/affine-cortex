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
        
        # Cache data for scoring and sampling environments
        self._scoring_data: Dict[str, Any] = {}  # enabled_for_scoring
        self._sampling_data: Dict[str, Any] = {}  # enabled_for_sampling
        self._state = CacheState.EMPTY
        self._lock = asyncio.Lock()
        
        # Timestamp for cache
        self._updated_at = 0
        
        # Background task
        self._refresh_task: Optional[asyncio.Task] = None
    
    @property
    def state(self) -> CacheState:
        return self._state
    
    async def warmup(self) -> None:
        """Warm up cache on startup."""
        logger.info("Warming up scoring cache (scoring and sampling environments)...")
        
        async with self._lock:
            self._state = CacheState.WARMING
            try:
                await self._full_refresh()
                self._state = CacheState.READY
                self._updated_at = int(time.time())
                logger.info(f"Cache warmed up: scoring={len(self._scoring_data)}, sampling={len(self._sampling_data)} miners")
            except Exception as e:
                logger.error(f"Failed to warm up cache: {e}", exc_info=True)
                self._state = CacheState.EMPTY
    
    async def get_data(self, range_type: str = "scoring") -> Dict[str, Any]:
        """Get cached data with fallback logic.
        
        Args:
            range_type: "scoring" or "sampling"
        
        Non-blocking: Returns cached data immediately when READY or REFRESHING.
        Blocking: Waits for initial warmup when EMPTY or WARMING.
        
        Returns:
            Data dict with hotkey#revision as keys (includes uid field in each entry)
        """
        # Fast path: return cache if ready or refreshing (data can be empty dict)
        if self._state in [CacheState.READY, CacheState.REFRESHING]:
            return self._scoring_data if range_type == "scoring" else self._sampling_data
        
        # Slow path: cache not initialized yet
        if self._state == CacheState.EMPTY:
            async with self._lock:
                # Double check after acquiring lock
                if self._state == CacheState.EMPTY:
                    logger.warning("Cache miss - computing synchronously")
                    self._state = CacheState.WARMING
                    try:
                        await self._full_refresh()
                        self._state = CacheState.READY
                        self._updated_at = int(time.time())
                        return self._scoring_data if range_type == "scoring" else self._sampling_data
                    except Exception as e:
                        self._state = CacheState.EMPTY
                        raise RuntimeError(f"Failed to compute cache data: {e}") from e
        
        # Warming in progress - wait and recheck
        if self._state == CacheState.WARMING:
            for _ in range(60):
                await asyncio.sleep(1)
                # Recheck state - may have changed to READY
                if self._state == CacheState.READY:
                    return self._scoring_data if range_type == "scoring" else self._sampling_data
            # Timeout - return whatever we have
            logger.warning("Cache warming timeout, returning current data")
            return self._scoring_data if range_type == "scoring" else self._sampling_data
        
        # Fallback: return any available data (should not reach here)
        logger.warning(f"Returning cache in unexpected state (state={self._state})")
        return self._scoring_data if range_type == "scoring" else self._sampling_data
    
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
                
                # Always perform full refresh
                await self._full_refresh()
                
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
        """Execute full refresh with NEW incremental update strategy.
        
        NEW DESIGN:
        - Uses sampling_list from sampling_config if available
        - Query by PK+SK for each miner+env+taskid (no range queries)
        - Incremental updates based on task ID differences
        - Detects miner changes and removes invalid cache
        """
        start_time = time.time()
        logger.info("Full refresh started (incremental update strategy)")
        
        from affine.database.dao.system_config import SystemConfigDAO
        from affine.database.dao.miners import MinersDAO
        from affine.database.dao.sample_results import SampleResultsDAO
        
        system_config_dao = SystemConfigDAO()
        miners_dao = MinersDAO()
        sample_dao = SampleResultsDAO()
        
        # 1. Get current valid miners
        valid_miners = await miners_dao.get_valid_miners()
        current_miner_keys = {
            (m['hotkey'], m['revision']) for m in valid_miners
        }
        
        # 2. Detect miner changes
        previous_miner_keys = getattr(self, '_previous_miner_keys', set())
        removed_miners = previous_miner_keys - current_miner_keys
        
        # 3. Remove invalid miner cache (from both scoring and sampling)
        if removed_miners:
            for hotkey, revision in removed_miners:
                key = f"{hotkey}#{revision}"
                
                # Remove from both caches
                self._scoring_data.pop(key, None)
                self._sampling_data.pop(key, None)
                
                logger.info(f"Removed cache for invalid miner {hotkey[:8]}...#{revision[:8]}...")
        
        # 4. Get environment configurations
        environments = await system_config_dao.get_param_value('environments', {})
        
        if not environments:
            self._scoring_data = {}
            self._sampling_data = {}
            logger.info("Full refresh completed: no environments configured")
            return
        
        if not valid_miners:
            self._scoring_data = {}
            self._sampling_data = {}
            logger.info("Full refresh completed: no valid miners")
            return
        
        # 5. Initialize all miner entries using hotkey#revision as key
        for miner in valid_miners:
            hotkey = miner['hotkey']
            revision = miner['revision']
            uid = miner['uid']
            key = f"{hotkey}#{revision}"
            
            # Initialize miner entry in both caches if not exists (include uid field)
            if key not in self._scoring_data:
                self._scoring_data[key] = {
                    'uid': uid,
                    'hotkey': hotkey,
                    'model_revision': revision,
                    'model_repo': miner.get('model'),
                    'first_block': miner.get('first_block'),
                    'env': {}
                }
            else:
                # Update UID if it changed
                self._scoring_data[key]['uid'] = uid
            
            if key not in self._sampling_data:
                self._sampling_data[key] = {
                    'uid': uid,
                    'hotkey': hotkey,
                    'model_revision': revision,
                    'model_repo': miner.get('model'),
                    'first_block': miner.get('first_block'),
                    'env': {}
                }
            else:
                # Update UID if it changed
                self._sampling_data[key]['uid'] = uid
        
        # 6. Build concurrent query tasks for ALL miner×env combinations
        async def query_and_populate(miner: dict, env_name: str, env_config: dict):
            """Query a single miner×env and populate caches."""
            hotkey = miner['hotkey']
            revision = miner['revision']
            key = f"{hotkey}#{revision}"
            
            enabled_for_scoring = env_config.get('enabled_for_scoring', False)
            enabled_for_sampling = env_config.get('enabled_for_sampling', False)
            
            if not enabled_for_scoring and not enabled_for_sampling:
                return
            
            # Query once
            env_cache_data = await self._query_miner_env_data(
                sample_dao=sample_dao,
                miner_info=miner,
                env=env_name,
                env_config=env_config
            )
            
            # Populate both caches if needed (using hotkey#revision as key)
            if enabled_for_scoring:
                self._scoring_data[key]['env'][env_name] = env_cache_data
            if enabled_for_sampling:
                self._sampling_data[key]['env'][env_name] = env_cache_data
        
        # Build all tasks
        tasks = []
        for miner in valid_miners:
            for env_name, env_config in environments.items():
                tasks.append(query_and_populate(miner, env_name, env_config))
        
        # Execute ALL miner×env queries concurrently
        logger.info(f"Starting concurrent refresh for {len(tasks)} miner×env combinations...")
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # 7. Update miner tracking
        self._previous_miner_keys = current_miner_keys
        
        # 8. Log statistics
        elapsed = time.time() - start_time
        enabled_envs = sum(
            1 for e in environments.values()
            if e.get('enabled_for_scoring') or e.get('enabled_for_sampling')
        )
        combo_count = len(valid_miners) * enabled_envs
        throughput = combo_count / elapsed if elapsed > 0 else 0
        logger.info(
            f"Full refresh completed: {len(valid_miners)} miners, "
            f"{enabled_envs}/{len(environments)} enabled environments, "
            f"{combo_count} miner×env combinations, "
            f"elapsed={elapsed:.2f}s, "
            f"throughput={throughput:.1f} combos/sec"
        )
    
    async def _query_miner_env_data(
        self,
        sample_dao,
        miner_info: dict,
        env: str,
        env_config: dict
    ) -> dict:
        """Query and build cache data for a single miner+env combination.

        Returns:
            Cache data dict with all_samples, sampling_task_ids, and statistics.
            No redundant 'samples' field — Stage1 derives task_scores from
            all_samples + sampling_task_ids.

        Strategy:
        1. Query full partition to get all_samples
        2. Compute completeness statistics against sampling_task_ids
        """
        from affine.core.sampling_list import get_task_id_set_from_config
        from affine.database.client import get_client

        hotkey = miner_info['hotkey']
        revision = miner_info['revision']

        # 1. Get target task IDs (sampling list)
        target_task_ids = get_task_id_set_from_config(env_config)

        if not target_task_ids:
            return {
                'all_samples': [],
                'sampling_task_ids': [],
                'total_count': 0,
                'completed_count': 0,
                'missing_task_ids': [],
                'completeness': 0.0
            }

        # 2. Query full partition (all samples for this miner×env)
        pk = sample_dao._make_pk(hotkey, revision, env)
        params = {
            'TableName': sample_dao.table_name,
            'KeyConditionExpression': 'pk = :pk',
            'ExpressionAttributeValues': {':pk': {'S': pk}},
            'ProjectionExpression': 'task_id,score,#ts',
            'ExpressionAttributeNames': {'#ts': 'timestamp'},
        }
        raw_items = await sample_dao._query_all_pages(get_client(), params)
        all_samples = [sample_dao._deserialize(item) for item in raw_items]

        # 3. Calculate completeness statistics against sampling list
        completed_ids = {s['task_id'] for s in all_samples if s.get('task_id') in target_task_ids}
        expected_count = len(target_task_ids)
        completed_count = len(completed_ids)
        completeness = completed_count / expected_count if expected_count > 0 else 0.0
        missing_ids = sorted(list(target_task_ids - completed_ids))[:100]

        return {
            'all_samples': all_samples,
            'sampling_task_ids': sorted(list(target_task_ids)),
            'total_count': expected_count,
            'completed_count': completed_count,
            'missing_task_ids': missing_ids,
            'completeness': round(completeness, 4)
        }
    

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
    """Get cached data.
    
    Args:
        range_type: "scoring" for enabled_for_scoring environments,
                   "sampling" for enabled_for_sampling environments
    """
    return await _cache_manager.get_data(range_type=range_type)