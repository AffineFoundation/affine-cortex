"""
Task Pool Manager

Implements weighted random task selection with UUID caching.

Key Features:
- Weighted random selection: probability proportional to pending task count per miner
- UUID location cache: fast O(1) task lookup during completion
- Idempotent completion: gracefully handle already-completed/deleted tasks

Optimizations:
- No locking: DynamoDB provides atomicity via delete+put
- UUID cache: avoid expensive Scan operations (50x speedup)
- Dependency injection: consistent with other DAOs
"""

import asyncio
import time
import random
import os
from typing import Dict, Any, Optional, List, Tuple, Callable, TypeVar, Generic

from affine.database.dao.task_pool import TaskPoolDAO
from affine.database.dao.execution_logs import ExecutionLogsDAO
from affine.database.dao.miners import MinersDAO
from affine.database.dao.sample_results import SampleResultsDAO
from affine.utils.subtensor import get_subtensor

from affine.core.setup import logger
from affine.core.environments import ENV_CONFIGS


T = TypeVar('T')


class AsyncCache(Generic[T]):
    """Generic async cache with background refresh support.
    
    Features:
    - TTL-based expiration
    - Non-blocking background refresh
    - Cold start handling (blocks only on first fetch)
    """
    
    def __init__(self, ttl: int, name: str = "cache"):
        """Initialize cache.
        
        Args:
            ttl: Time-to-live in seconds
            name: Cache name for logging
        """
        self.ttl = ttl
        self.name = name
        self._data: Optional[T] = None
        self._timestamp: float = 0
        self._refresh_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
    
    async def get(self, fetcher: Callable[[], T]) -> T:
        """Get cached data with background refresh.
        
        Args:
            fetcher: Async function to fetch fresh data
            
        Returns:
            Cached or fresh data
        """
        # Fast path: return cached data if available
        async with self._lock:
            if self._data is not None:
                age = time.time() - self._timestamp
                
                # Trigger background refresh if expired
                if age > self.ttl:
                    if self._refresh_task is None or self._refresh_task.done():
                        logger.debug(f"{self.name} cache expired (age={age:.1f}s), triggering refresh")
                        self._refresh_task = asyncio.create_task(
                            self._background_refresh(fetcher)
                        )
                
                # Return cached data (even if stale)
                return self._data
        
        data = await fetcher()
        
        async with self._lock:
            self._data = data
            self._timestamp = time.time()
        
        return data
    
    async def _background_refresh(self, fetcher: Callable[[], T]):
        """Background task to refresh cache."""
        try:
            logger.debug(f"{self.name} cache background refresh started")
            start_time = time.time()
            
            data = await fetcher()
            
            elapsed = time.time() - start_time
            logger.debug(f"{self.name} cache refreshed in {elapsed:.2f}s")
            
            async with self._lock:
                self._data = data
                self._timestamp = time.time()
                
        except Exception as e:
            logger.error(f"{self.name} cache refresh failed: {e}", exc_info=True)


class TaskPoolManager:
    """
    Manages task pool with weighted random selection and dual caching.
    
    Uses background refresh for miner counts to avoid blocking fetch requests.
    """
    
    def __init__(self, count_cache_ttl: int = 30, miners_cache_ttl: int = 60, stats_cache_ttl: int = 60, block_cache_ttl: int = 10, warmup: bool = True):
        """Initialize TaskPoolManager with caches.
        
        Args:
            count_cache_ttl: TTL for miner count cache (seconds)
            miners_cache_ttl: TTL for miners cache (seconds)
            stats_cache_ttl: TTL for pool stats cache (seconds)
            block_cache_ttl: TTL for block number cache (seconds)
            warmup: Whether to warmup caches on startup (default: True)
        """
        self.dao = TaskPoolDAO()
        self.logs_dao = ExecutionLogsDAO()
        self.miners_dao = MinersDAO()
        self.sample_dao = SampleResultsDAO()
        
        # Async caches with background refresh
        self._count_caches: Dict[str, AsyncCache[Dict[str, int]]] = {}  # {env: cache}
        self._count_cache_ttl = count_cache_ttl
        self._miners_cache = AsyncCache[Dict[str, Dict[str, Any]]](
            ttl=miners_cache_ttl,
            name="miners"
        )
        
        # Pool stats cache: {env: stats_dict}
        self._pool_stats_caches: Dict[str, AsyncCache[Dict[str, int]]] = {}
        self._stats_cache_ttl = stats_cache_ttl
        
        # Block number cache (10s TTL)
        self._block_cache = AsyncCache[int](
            ttl=block_cache_ttl,
            name="block_number"
        )
        
        # UUID location cache: task_uuid -> (pk, sk, assigned_at, env)
        # assigned_at and env are used for timeout detection without DB query
        self._uuid_cache: Dict[str, Tuple[str, str, int, str]] = {}
        self._cache_lock = asyncio.Lock()
        
        # Warmup flag
        self._warmup_enabled = warmup
        self._warmup_done = False
        
        # Timeout cleanup task
        self._timeout_cleanup_task: Optional[asyncio.Task] = None
        
        logger.info(f"TaskPoolManager initialized (count_cache_ttl={count_cache_ttl}s, miners_cache_ttl={miners_cache_ttl}s, stats_cache_ttl={stats_cache_ttl}s, block_cache_ttl={block_cache_ttl}s, warmup={warmup})")
    
    async def _get_miner_counts(self, env: str) -> Dict[str, int]:
        """Get task counts per miner with non-blocking cache refresh."""
        # Create cache for this env if not exists
        if env not in self._count_caches:
            self._count_caches[env] = AsyncCache[Dict[str, int]](
                ttl=self._count_cache_ttl,
                name=f"count[{env}]"
            )
        
        return await self._count_caches[env].get(
            lambda: self.dao.get_miner_task_counts(env)
        )
    
    async def _get_miners(self) -> Dict[str, Dict[str, Any]]:
        """Get all miners with non-blocking cache refresh."""
        async def fetch_miners():
            miners_list = await self.miners_dao.get_all_miners()
            return {miner['hotkey']: miner for miner in miners_list}
        
        return await self._miners_cache.get(fetch_miners)
    
    async def _get_current_block(self) -> int:
        """Get current block number with caching."""
        async def fetch_block():
            subtensor = await get_subtensor()
            return await subtensor.get_current_block()
        
        return await self._block_cache.get(fetch_block)
    
    async def reset_assigned_on_startup(self):
        """Reset all assigned tasks to pending on API server startup.
        
        Rationale:
        - When API server restarts, all executor processes are restarted/lost
        - Any tasks in 'assigned' status are orphaned (no executor running them)
        - We reset them to 'pending' so they can be reassigned to new executors
        
        This is different from timeout cleanup:
        - Startup reset: unconditional, all assigned -> pending (no timeout check)
        - Timeout cleanup: runtime check, only reset tasks that exceeded timeout
        
        This is called automatically during server startup if warmup=True.
        """
        if not self._warmup_enabled or self._warmup_done:
            return
        
        try:
            logger.info("TaskPoolManager startup reset: resetting all assigned tasks to pending...")
            start_time = time.time()
            
            # Reset all assigned tasks to pending via DAO
            reset_count = await self.dao.reset_all_assigned_to_pending()
            
            elapsed = time.time() - start_time
            logger.info(
                f"TaskPoolManager startup reset completed: "
                f"reset {reset_count} assigned tasks to pending in {elapsed:.2f}s"
            )
            
            self._warmup_done = True
            
        except Exception as e:
            logger.error(f"TaskPoolManager startup reset failed: {e}", exc_info=True)
            # Non-fatal: continue startup, tasks may be orphaned but will timeout eventually
    
    async def get_pool_stats(self, env: str) -> Dict[str, int]:
        """Get pool statistics for an environment with caching.
        
        Uses AsyncCache for automatic background refresh.
        
        Args:
            env: Environment name
            
        Returns:
            Dict with counts: pending, assigned, failed
        """
        # Create cache for this env if not exists
        if env not in self._pool_stats_caches:
            self._pool_stats_caches[env] = AsyncCache[Dict[str, int]](
                ttl=self._stats_cache_ttl,
                name=f"pool_stats[{env}]"
            )
        
        return await self._pool_stats_caches[env].get(
            lambda: self.dao.get_pool_stats(env)
        )
    
    async def reset_timeout_tasks(self) -> int:
        """Reset timeout assigned tasks during runtime (not on startup).
        
        This is for long-running scenarios where executors may crash or hang.
        Each task's timeout is determined by its environment's eval_params.timeout config.
        
        Uses UUID cache for fast timeout detection without DB scan.
        
        Returns:
            Number of tasks reset
        """
        current_time = int(time.time())
        
        # Find timeout tasks from cache
        timeout_tasks = []
        async with self._cache_lock:
            for task_uuid, (pk, sk, assigned_at, env) in list(self._uuid_cache.items()):
                # Skip if assigned_at is None or 0 (invalid timestamp)
                if not assigned_at:
                    continue
                
                # Get timeout from environment config
                env_config = ENV_CONFIGS.get(env)
                if not env_config:
                    logger.warning(f"Unknown environment {env} for task {task_uuid}, skipping")
                    continue
                
                timeout_seconds = env_config.eval_params['timeout']
                
                # Check if task has timed out
                timeout_threshold = current_time - timeout_seconds
                if assigned_at < timeout_threshold:
                    timeout_tasks.append((task_uuid, pk, sk))
        
        if not timeout_tasks:
            return 0
        
        # Reset tasks in DB with parallel processing (max 25 concurrent)
        from affine.database.client import get_client
        client = get_client()
        semaphore = asyncio.Semaphore(25)
        
        async def reset_single_task(task_uuid: str, pk: str, sk: str) -> bool:
            """Reset a single task. Returns True if successful."""
            async with semaphore:
                try:
                    # Get full task data
                    task = await self.dao.get(pk, sk)
                    if not task:
                        # Task already deleted
                        async with self._cache_lock:
                            self._uuid_cache.pop(task_uuid, None)
                        return False
                    
                    # Verify still assigned
                    if task.get('status') != 'assigned':
                        async with self._cache_lock:
                            self._uuid_cache.pop(task_uuid, None)
                        return False
                    
                    # Conditionally delete old assigned record
                    try:
                        await client.delete_item(
                            TableName=self.dao.table_name,
                            Key={
                                'pk': {'S': task['pk']},
                                'sk': {'S': task['sk']}
                            },
                            ConditionExpression='#status = :status',
                            ExpressionAttributeNames={'#status': 'status'},
                            ExpressionAttributeValues={':status': {'S': 'assigned'}}
                        )
                    except client.exceptions.ConditionalCheckFailedException:
                        # Task status changed (completed/reset by another process)
                        logger.debug(
                            f"Task {task_uuid} status changed during reset "
                            f"(race condition with complete_task or another reset)"
                        )
                        async with self._cache_lock:
                            self._uuid_cache.pop(task_uuid, None)
                        return False
                    
                    # Create new pending record
                    new_status = 'pending'
                    new_sk = self.dao._make_sk(task['env'], new_status, task['task_id'])
                    new_gsi1_pk = self.dao._make_gsi1_pk(task['env'], new_status)
                    new_gsi1_sk = self.dao._make_gsi1_sk(
                        task['miner_hotkey'],
                        task['model_revision'],
                        task['task_id']
                    )
                    
                    task['sk'] = new_sk
                    task['status'] = new_status
                    task['assigned_to'] = None
                    task['assigned_at'] = None
                    task['gsi1_pk'] = new_gsi1_pk
                    task['gsi1_sk'] = new_gsi1_sk
                    
                    await self.dao.put(task)
                    
                    # Remove from cache
                    async with self._cache_lock:
                        self._uuid_cache.pop(task_uuid, None)
                    
                    return True
                except Exception as e:
                    logger.error(f"Failed to reset task {task_uuid}: {e}", exc_info=True)
                    return False
        
        # Process all timeout tasks in parallel
        reset_tasks = [
            reset_single_task(task_uuid, pk, sk)
            for task_uuid, pk, sk in timeout_tasks
        ]
        
        results = await asyncio.gather(*reset_tasks, return_exceptions=True)
        
        # Count successful resets
        reset_count = sum(1 for r in results if r is True)
        
        if reset_count > 0:
            logger.info(f"Runtime timeout cleanup: reset {reset_count}/{len(timeout_tasks)} timeout assigned tasks")
        
        return reset_count
    
    async def start_timeout_cleanup_loop(self):
        """Start background timeout cleanup loop for runtime timeout detection.
        
        This runs continuously during API server operation to detect and reset
        tasks that have exceeded their execution timeout.
        
        Note: This is different from startup reset (reset_assigned_on_startup):
        - Startup: unconditional reset of all assigned tasks (executor processes lost)
        - Runtime: conditional reset of only timed-out tasks (executor may be running)
        """
        if self._timeout_cleanup_task is not None:
            logger.warning("Timeout cleanup loop already started")
            return
        
        async def cleanup_loop():
            """Background loop for runtime timeout task cleanup.
            
            Each task's timeout is determined by its environment's eval_params.timeout config.
            """
            cleanup_interval = int(os.getenv('TASK_TIMEOUT_CLEANUP_INTERVAL', '300'))  # 5 minutes
            
            logger.info(f"Runtime timeout cleanup loop started (interval={cleanup_interval}s, per-env timeout)")
            
            while True:
                try:
                    await self.reset_timeout_tasks()
                    await asyncio.sleep(cleanup_interval)
                except asyncio.CancelledError:
                    logger.info("Runtime timeout cleanup loop cancelled")
                    break
                except Exception as e:
                    logger.error(f"Runtime timeout cleanup error: {e}", exc_info=True)
        
        self._timeout_cleanup_task = asyncio.create_task(cleanup_loop())
        logger.info("Runtime timeout cleanup background task started")
    
    def _select_miners_weighted(
        self,
        miner_counts: Dict[str, int],
        count: int,
    ) -> List[str]:
        """Select miners using uniform random sampling.
        
        Simplified: randomly shuffle and take first `count` miners.
        
        Args:
            miner_counts: Dict mapping miner_key -> pending task count
            count: Number of miners to select
        
        Returns:
            List of selected miner_keys (no duplicates, length <= count)
        """
        if not miner_counts:
            return []

        miner_keys = list(miner_counts.keys())
        random.shuffle(miner_keys)
        return miner_keys[:count]
    
    async def _get_task_location(
        self, 
        task_uuid: str
    ) -> Optional[Tuple[str, str]]:
        """
        Get (PK, SK) for task UUID, with cache and DB fallback.
        
        Cache strategy:
        1. Check cache first (fast path)
        2. If miss, scan DB (cold start / evicted entry)
        3. Update cache for future lookups
        
        Args:
            task_uuid: Task UUID
            
        Returns:
            (pk, sk) tuple if found, None otherwise
        """
        # Fast path: check cache
        async with self._cache_lock:
            location = self._uuid_cache.get(task_uuid)
        
        if location:
            return location[:2]  # Return only (pk, sk)
        
        # Slow path: DB scan (cache miss)
        logger.debug(f"UUID cache miss for {task_uuid}, scanning DB")
        task = await self.dao.get_task_by_uuid(task_uuid)
        
        if not task:
            return None
        
        # Cache location
        async with self._cache_lock:
            assigned_at = task.get('assigned_at') or 0
            env = task.get('env', '')
            self._uuid_cache[task_uuid] = (task['pk'], task['sk'], assigned_at, env)
        
        return (task['pk'], task['sk'])
    
    async def fetch_task(
        self,
        executor_hotkey: str,
        env: Optional[str] = None,
        batch_size: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Fetch task(s) using weighted random selection with parallel queries.
        
        Algorithm:
        1. Select N miners randomly (weighted), where N = batch_size * 2-3
        2. Query all selected miners in parallel (asyncio.gather)
        3. Collect valid tasks and assign in parallel
        
        Args:
            executor_hotkey: Executor's hotkey
            env: Optional environment filter (if None, select from all envs)
            batch_size: Number of tasks to fetch (default: 1)
            
        Returns:
            List of task dicts (may be empty, length 0 to batch_size)
        """
        try:
            # Validate env parameter is provided
            if not env:
                logger.error("env parameter is required for fetch_task")
                return []
            
            # Get miner counts (cached)
            all_miner_counts = await self._get_miner_counts(env)
            
            if not all_miner_counts:
                logger.debug(f"No available tasks found for env={env}")
                return []
            
            # Weighted random selection of miners (without replacement)
            num_miners_to_query = min(int(batch_size * 1.5), len(all_miner_counts))
            
            # Select miners using weighted random sampling (anti-starvation: max_ratio=2.0)
            selected_miners = self._select_miners_weighted(
                all_miner_counts,
                num_miners_to_query,
            )
            
            # Parallel query all selected miners
            query_tasks = []
            for miner_key in selected_miners:
                hotkey, revision = miner_key.split('#', 1)
                query_tasks.append(
                    self.dao.get_pending_tasks_for_miner(env, hotkey, revision, limit=1)
                )
            
            # Execute all queries in parallel
            results = await asyncio.gather(*query_tasks, return_exceptions=True)
            
            # Collect valid tasks (filter out exceptions and empty results)
            candidate_tasks = []
            for result in results:
                if isinstance(result, Exception):
                    continue
                if isinstance(result, list) and result:
                    candidate_tasks.append(result[0])  # Take first task
            
            # Take first batch_size tasks and assign in parallel
            tasks_to_assign = candidate_tasks[:batch_size]
            
            if not tasks_to_assign:
                logger.debug(f"No valid tasks found for env={env} after querying {num_miners_to_query} miners")
                return []
            
            # Parallel assignment
            try:
                assigned_results = await self.dao.batch_assign_tasks(tasks_to_assign, executor_hotkey)
            except Exception as e:
                logger.error(f"Batch assign failed: {e}")
                assigned_results = []
            
            # Filter successful assignments, cache UUIDs, and enrich with miner data
            miners_dict = await self._get_miners()
            
            assigned_tasks = []
            for result in assigned_results:
                # Cache UUID location with assigned_at and env
                async with self._cache_lock:
                    assigned_at = result.get('assigned_at') or int(time.time())
                    task_env = result.get('env', '')
                    self._uuid_cache[result['task_uuid']] = (
                        result['pk'],
                        result['sk'],
                        assigned_at,
                        task_env
                    )
                
                # Enrich task with miner data from cache
                miner_hotkey = result['miner_hotkey']
                miner_record = miners_dict.get(miner_hotkey)
                
                if not miner_record:
                    logger.warning(f"Miner record not found for hotkey {miner_hotkey[:16]}..., skipping task")
                    continue
                
                miner_uid = miner_record.get('uid')
                if miner_uid is None:
                    logger.warning(f"UID not found for hotkey {miner_hotkey[:16]}..., skipping task")
                    continue
                
                chute_slug = miner_record.get('chute_slug')
                if not chute_slug:
                    logger.warning(f"chute_slug not found for hotkey {miner_hotkey[:16]}..., skipping task")
                    continue
                
                # Add miner_uid and chute_slug to task
                enriched_task = {
                    **result,
                    'miner_uid': miner_uid,
                    'chute_slug': chute_slug,
                }
                
                assigned_tasks.append(enriched_task)
                
                logger.debug(
                    f"Task {result['task_uuid']} assigned to {executor_hotkey} "
                    f"(miner={miner_hotkey[:12]}..., uid={miner_uid}, env={env}, task_id={result['task_id']})"
                )
            
            logger.info(
                f"TaskPoolManager.fetch_task({env}): "
                f"queried {num_miners_to_query} miners, assigned {len(assigned_tasks)}/{batch_size} tasks"
            )
            
            # Always return list
            return assigned_tasks
            
        except Exception as e:
            logger.error(f"Error fetching task(s): {e}", exc_info=True)
            return []
    
    async def complete_task(
        self,
        task_uuid: str,
        executor_hotkey: str,
        success: bool,
        result: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        error_code: Optional[str] = None,
        submission_signature: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Complete a task (success or failure).
        
        For successful tasks, also saves the sample to database.
        Idempotent: if task already completed/deleted, just log and return success.
        
        Args:
            task_uuid: Task UUID
            executor_hotkey: Executor's hotkey
            success: Whether task succeeded
            result: Task result (for success case, must include score, latency_ms, extra)
            error_message: Error message (for failure case)
            error_code: Error code (for failure case)
            submission_signature: Signature of submission (for success case)
            
        Returns:
            Status dict with 'status' and 'message' keys
        """
        try:
            # Step 1: Get task location (with cache)
            location = await self._get_task_location(task_uuid)
            
            if not location:
                logger.info(
                    f"Task {task_uuid} not found (completed/deleted), "
                    f"ignoring completion from {executor_hotkey}"
                )
                return {
                    'status': 'not_found',
                    'message': 'Task already completed or removed'
                }
            
            pk, sk = location
            
            # Step 2: Get full task data
            task = await self.dao.get(pk, sk)
            
            if not task:
                # Race condition: task deleted between cache check and get
                logger.warning(
                    f"Task {task_uuid} deleted after cache lookup "
                    f"(race condition, ignoring)"
                )
                
                # Clean cache
                async with self._cache_lock:
                    self._uuid_cache.pop(task_uuid, None)
                
                return {
                    'status': 'not_found',
                    'message': 'Task already completed or removed'
                }
            
            # Step 3: Handle successful completion
            if success:
                if not result:
                    raise ValueError(
                        f"Task {task_uuid} marked as success but result is None. "
                        "This indicates a bug in the caller."
                    )
                
                # Get current block number (cached)
                block_number = await self._get_current_block()
                
                # Save sample to database
                try:
                    await self.sample_dao.save_sample(
                        miner_hotkey=task["miner_hotkey"],
                        model_revision=task["model_revision"],
                        model=task["model"],
                        env=task["env"],
                        task_id=str(task["task_id"]),
                        score=result['score'],
                        latency_ms=result['latency_ms'],
                        extra=result.get('extra', {}),
                        validator_hotkey=executor_hotkey,
                        block_number=block_number,
                        signature=submission_signature or "",
                    )
                except Exception as e:
                    logger.error(f"Failed to save sample for task {task_uuid}: {e}", exc_info=True)
                    # Continue to log and complete task even if sample save fails
                
                # Log task completion
                await self.logs_dao.log_task_complete(
                    miner_hotkey=task['miner_hotkey'],
                    task_uuid=task_uuid,
                    dataset_task_id=task['task_id'],
                    env=task['env'],
                    executor_hotkey=executor_hotkey,
                    score=result['score'],
                    latency_ms=result['latency_ms'],
                    execution_time_ms=result.get('execution_time_ms', 0)
                )
            else:
                if not error_message:
                    raise ValueError(
                        f"Task {task_uuid} marked as failure but error_message is None. "
                        "This indicates a bug in the caller."
                    )
                
                await self.logs_dao.log_task_failure(
                    miner_hotkey=task['miner_hotkey'],
                    task_uuid=task_uuid,
                    dataset_task_id=task['task_id'],
                    env=task['env'],
                    executor_hotkey=executor_hotkey,
                    error_message=error_message,
                    error_code=error_code,
                    error_type='execution',
                    execution_time_ms=0
                )
            
            # Step 4: Complete or fail task
            if success:
                # Delete task from pool
                await self.dao.complete_task(task)
                
                # Remove from cache
                async with self._cache_lock:
                    self._uuid_cache.pop(task_uuid, None)
                
                logger.debug(
                    f"Task {task_uuid} completed successfully by {executor_hotkey} "
                    f"(miner={task['miner_hotkey']}, env={task['env']}, task_id={task['task_id']})"
                )
                
                return {
                    'status': 'completed',
                    'message': 'Task completed successfully'
                }
            
            # Handle task failure
            # error_message already validated above
            updated_task = await self.dao.fail_task(
                task,
                error_message,
                error_code
            )

            # fail_task() returns either 'paused' or 'pending' status
            if updated_task['status'] == 'paused':
                # Max retries reached, paused
                async with self._cache_lock:
                    self._uuid_cache.pop(task_uuid, None)
                
                logger.warning(
                    f"Task {task_uuid} paused"
                    f"{updated_task['retry_count']} retries (max={updated_task['max_retries']})"
                )
                return {
                    'status': 'paused',
                    'message': f"Task paused after {updated_task['retry_count']} retries"
                }
            
            # Status is 'pending', will retry (assigned_at is None for pending)
            # Remove from cache since pending tasks should not be cached
            async with self._cache_lock:
                self._uuid_cache.pop(task_uuid, None)
            
            logger.info(
                f"Task {task_uuid} will retry ({updated_task['retry_count']}/{updated_task['max_retries']})"
            )
            return {
                'status': 'retry',
                'message': f"Task will be retried ({updated_task['retry_count']}/{updated_task['max_retries']})"
            }
                
        except Exception as e:
            logger.error(f"Error completing task {task_uuid}: {e}", exc_info=True)
            return {
                'status': 'error',
                'message': f'Internal error: {str(e)}'
            }