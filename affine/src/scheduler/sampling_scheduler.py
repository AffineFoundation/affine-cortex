"""
Sampling Scheduler

Manages sampling list rotation and per-miner sampling pool allocation.
"""

import math
import time
import asyncio
import random
from typing import List, Optional, Dict, Set, Any

from affine.core.setup import logger
from affine.core.sampling_list import SamplingListManager
from affine.database.dao.system_config import SystemConfigDAO
from affine.database.dao.task_pool import TaskPoolDAO
from affine.database.dao.miners import MinersDAO
from affine.database.dao.sample_results import SampleResultsDAO
from affine.database.dao.miner_stats import MinerStatsDAO


class PerMinerSamplingScheduler:
    """Per-miner sampling pool scheduler with weighted allocation.

    Architecture:
    1. Each miner has dynamic slots (3-10, default 6), stored in MinerStats
    2. Weighted allocation across environments based on scheduling_weight config
    3. Minimum guarantee: each env gets at least 1 slot
    4. Incremental scheduling every 10s
    5. Priority-based task selection from sampling list tail
    6. Rate limiting: when rotation_enabled, actual sampling rate is limited to
       rotation_rate * RATE_MARGIN to prevent answer memorization attacks
    7. Minimum rate guarantee: allowed rate is at least sampling_count/48 per hour
       to ensure sampling completes within 2 days
    """

    DEFAULT_SLOTS = 6
    MIN_SLOTS = 3
    MAX_SLOTS = 10

    # Rate limiting: allow actual sampling rate to exceed rotation rate by this margin
    RATE_MARGIN = 1.2
    
    def __init__(
        self,
        system_config_dao: Optional[SystemConfigDAO] = None,
        task_pool_dao: Optional[TaskPoolDAO] = None,
        sample_results_dao: Optional[SampleResultsDAO] = None,
        miners_dao: Optional[MinersDAO] = None,
        miner_stats_dao: Optional[MinerStatsDAO] = None,
        scheduling_interval: int = 10
    ):
        self.config_dao = system_config_dao or SystemConfigDAO()
        self.task_pool_dao = task_pool_dao or TaskPoolDAO()
        self.sample_results_dao = sample_results_dao or SampleResultsDAO()
        self.miners_dao = miners_dao or MinersDAO()
        self.miner_stats_dao = miner_stats_dao or MinerStatsDAO()
        
        self.scheduling_interval = scheduling_interval
        
        # Cache for env weights (refreshed each scheduling cycle)
        self._env_weights: Dict[str, float] = {}
        
        # Track last known sampling lists per env
        self._last_sampling_lists: Dict[str, List[int]] = {}
        
        # Track last known valid miners for detecting additions/removals
        self._last_valid_miners: Set[tuple] = set()  # Set of (hotkey, revision) tuples

        # Rate limiting: track allocation timestamps per miner/env (sliding window)
        # Key: "{hotkey}#{revision}#{env}", Value: list of unix timestamps
        self._allocation_timestamps: Dict[str, List[float]] = {}

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start per-miner sampling scheduler."""
        logger.info(
            f"Starting per-miner sampling scheduler: "
            f"default_slots={self.DEFAULT_SLOTS}, interval={self.scheduling_interval}s"
        )
        self._running = True
        
        # Initialize sampling lists cache
        await self._initialize_sampling_lists()
        
        self._task = asyncio.create_task(self._scheduling_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def stop(self):
        """Stop sampling scheduler."""
        logger.info("Stopping per-miner sampling scheduler")
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
    
    async def _initialize_sampling_lists(self):
        """Initialize sampling lists cache from SystemConfig."""
        try:
            environments = await self.config_dao.get_param_value('environments', {})
            
            for env_name, env_config in environments.items():
                sampling_config = env_config.get('sampling_config', {})
                sampling_list = sampling_config.get('sampling_list', [])
                self._last_sampling_lists[env_name] = sampling_list
            
            logger.info(
                f"Initialized sampling lists for {len(self._last_sampling_lists)} environments"
            )
        except Exception as e:
            logger.error(f"Failed to initialize sampling lists: {e}", exc_info=True)
    
    async def _scheduling_loop(self):
        """Main scheduling loop - runs every 10s."""
        while self._running:
            try:
                await self._schedule_all_miners()
                await asyncio.sleep(self.scheduling_interval)
            except asyncio.CancelledError:
                logger.info("Scheduling loop cancelled")
                break
            except Exception as e:
                logger.error(f"Scheduling loop error: {e}", exc_info=True)
                await asyncio.sleep(10)
    
    async def _schedule_all_miners(self):
        """Schedule sampling tasks for all miners across all environments."""
        try:
            # Get all valid miners
            miners = await self.miners_dao.get_valid_miners()
            if not miners:
                logger.debug("No valid miners found")
                return
            
            # Build current valid miners set
            current_valid_miners = {
                (m['hotkey'], m['revision']) for m in miners
            }
            
            # Detect removed miners and cleanup their tasks
            removed_miners = self._last_valid_miners - current_valid_miners
            if removed_miners:
                await self._cleanup_removed_miners(removed_miners)
            
            # Detect new miners
            added_miners = current_valid_miners - self._last_valid_miners
            if added_miners:
                logger.info(f"Detected {len(added_miners)} new miners")
            
            # Update tracking
            self._last_valid_miners = current_valid_miners
            
            # Get sampling environments (only those with enabled_for_sampling=True)
            environments = await self.config_dao.get_param_value('environments', {})
            sampling_envs = [
                env_name for env_name, env_config in environments.items()
                if env_config.get('enabled_for_sampling', False)
                and env_config.get('sampling_config')
            ]
            
            if not sampling_envs:
                logger.warning("No sampling environments configured")
                return
            
            # Schedule each miner independently
            for miner in miners:
                try:
                    await self._schedule_miner(miner, sampling_envs, environments)
                except Exception as e:
                    logger.error(
                        f"Error scheduling miner {miner['hotkey'][:8]}...: {e}",
                        exc_info=True
                    )
        
        except Exception as e:
            logger.error(f"Error in schedule_all_miners: {e}", exc_info=True)
    
    async def _get_miner_slots(self, miner: Dict[str, Any]) -> int:
        """Get current slots allocation for a miner from MinerStats.
        
        Args:
            miner: Miner dict with hotkey, revision
            
        Returns:
            Number of slots (3-10, default 6)
        """
        try:
            hotkey = miner['hotkey']
            revision = miner['revision']
            stats = await self.miner_stats_dao.get_miner_slots(hotkey, revision)
            return stats if stats else self.DEFAULT_SLOTS
        except Exception as e:
            logger.debug(f"Error getting miner slots, using default: {e}")
            return self.DEFAULT_SLOTS
    
    def _get_env_weights(self, environments: Dict[str, Any]) -> Dict[str, float]:
        """Extract scheduling weights from environment configurations.

        Args:
            environments: Full environment configurations

        Returns:
            Dict mapping env name to weight (default 1.0 if not configured)
        """
        weights = {}
        for env_name, env_config in environments.items():
            if not env_config.get('enabled_for_sampling', False):
                continue
            sampling_config = env_config.get('sampling_config', {})
            # Default weight is 1.0 if not specified
            weights[env_name] = float(sampling_config.get('scheduling_weight', 1.0))
        return weights

    def _get_allocation_count(
        self,
        hotkey: str,
        revision: str,
        env: str,
        window_seconds: int = 3600
    ) -> int:
        """Get allocation count in the sliding window for rate limiting.

        Also cleans up expired timestamps.

        Args:
            hotkey: Miner hotkey
            revision: Model revision
            env: Environment name
            window_seconds: Time window in seconds (default 1 hour)

        Returns:
            Number of allocations in the time window
        """
        key = f"{hotkey}#{revision}#{env}"
        timestamps = self._allocation_timestamps.get(key, [])

        if not timestamps:
            return 0

        # Clean up expired timestamps
        cutoff = time.time() - window_seconds
        valid_timestamps = [t for t in timestamps if t > cutoff]

        # Update stored timestamps (cleanup)
        if len(valid_timestamps) != len(timestamps):
            if valid_timestamps:
                self._allocation_timestamps[key] = valid_timestamps
            else:
                self._allocation_timestamps.pop(key, None)

        return len(valid_timestamps)

    def _record_allocations(
        self,
        hotkey: str,
        revision: str,
        tasks: List[Dict[str, Any]]
    ):
        """Record task allocations for rate limiting.

        Args:
            hotkey: Miner hotkey
            revision: Model revision
            tasks: List of task specs with 'env' key
        """
        now = time.time()
        for task in tasks:
            env = task['env']
            key = f"{hotkey}#{revision}#{env}"
            if key not in self._allocation_timestamps:
                self._allocation_timestamps[key] = []
            self._allocation_timestamps[key].append(now)

    def _should_skip_env_for_miner(
        self,
        miner: Dict[str, Any],
        env: str,
        sampling_config: Dict[str, Any],
        allocation_count: int = 0
    ) -> bool:
        """Check if sampling should be skipped for this miner/env due to rate limiting.

        This prevents answer memorization attacks by limiting the allocation rate
        to slightly above the rotation rate when rotation is enabled.

        Uses internal allocation counter (real-time) to avoid 5-minute stats delay.

        Args:
            miner: Miner dict with hotkey, revision, uid
            env: Environment name
            sampling_config: Environment's sampling configuration
            allocation_count: Allocations in last hour from internal counter

        Returns:
            True if sampling should be skipped, False otherwise
        """
        # System models (uid <= 0) are not rate limited
        if miner.get('uid', 0) <= 0:
            return False

        # Only limit rate when rotation is enabled
        if not sampling_config.get('rotation_enabled', False):
            return False

        # Get rotation parameters
        rotation_count = sampling_config.get('rotation_count', 0)
        rotation_interval = sampling_config.get('rotation_interval', 0)

        if rotation_count <= 0 or rotation_interval <= 0:
            return False

        # Calculate allowed count per hour based on rotation rate
        allowed_per_hour = rotation_count * (3600 / rotation_interval) * self.RATE_MARGIN

        # Minimum rate guarantee: ensure sampling can complete within 48 hours
        sampling_count = sampling_config.get('sampling_count', 0)
        if sampling_count > 0:
            min_rate_per_hour = sampling_count / 48  # Complete in 2 days
            allowed_per_hour = max(allowed_per_hour, min_rate_per_hour)

        # Round up to make fractional rates effective
        allowed_per_hour = math.ceil(allowed_per_hour)

        # Compare allocation count with limit
        if allocation_count >= allowed_per_hour:
            logger.debug(
                f"Rate limit: miner={miner['hotkey'][:8]}... env={env} "
                f"allocations={allocation_count} allowed={allowed_per_hour}/hour"
            )
            return True

        return False

    async def _schedule_miner(
        self,
        miner: Dict[str, Any],
        sampling_envs: List[str],
        environments: Dict[str, Any]
    ):
        """Schedule sampling tasks for a single miner across all environments.

        Uses weighted allocation strategy with starvation prevention:
        1. Calculate target slots per env based on weights
        2. Ensure minimum 1 slot per env with missing tasks
        3. Allocate by deficit (target - active), highest first
        4. Remaining slots use round-robin
        5. **Anti-starvation**: If any env has missing tasks but active=0,
           allow allocation even if total_pool_count >= total_slots
        6. **Rate limiting**: Skip envs where actual sampling rate exceeds
           rotation_rate * RATE_MARGIN (anti-memorization)

        Args:
            miner: Miner dict with hotkey, revision, etc.
            sampling_envs: List of environment names
            environments: Full environment configurations
        """
        hotkey = miner['hotkey']
        revision = miner['revision']

        # Get miner's total slots from MinerStats
        total_slots = await self._get_miner_slots(miner)

        # Get current active task count per env (exclude paused tasks)
        env_active_counts: Dict[str, int] = {}
        total_pool_count = 0

        for env in sampling_envs:
            active_task_ids = await self.task_pool_dao.get_pending_task_ids_for_miner(
                miner_hotkey=hotkey,
                model_revision=revision,
                env=env,
                include_paused=False
            )
            env_active_counts[env] = len(active_task_ids)
            total_pool_count += len(active_task_ids)

        # Collect missing task IDs from all environments (before capacity check)
        env_missing_tasks: Dict[str, List[int]] = {}

        for env in sampling_envs:
            # Check rate limiting - skip env if allocation rate exceeds allowed rate
            sampling_config = environments.get(env, {}).get('sampling_config', {})
            allocation_count = self._get_allocation_count(hotkey, revision, env)
            if self._should_skip_env_for_miner(miner, env, sampling_config, allocation_count):
                continue

            try:
                missing_ids = await self._get_missing_task_ids(
                    miner=miner,
                    env=env,
                    environments=environments
                )
                
                if missing_ids:
                    env_missing_tasks[env] = missing_ids
            
            except Exception as e:
                logger.error(
                    f"Error getting missing tasks for {hotkey[:8]}...#{env}: {e}"
                )
        
        if not env_missing_tasks:
            return
        
        # Anti-starvation check: detect envs with missing tasks but no active tasks
        starving_envs = [
            env for env in env_missing_tasks
            if env_active_counts.get(env, 0) == 0
        ]
        
        # Capacity policy:
        # - Normal case: do not exceed total_slots
        # - Anti-starvation: allow a bounded temporary overflow of +1 per starving env
        #   (this overflow is meant to "un-starve" envs; it must not grow unbounded)
        if not starving_envs and total_pool_count >= total_slots:
            return
        
        # Only allow temporary overflow when the pool is already at capacity.
        # If the pool is not full, the scheduler can un-starve envs within total_slots,
        # so extra overflow would be unnecessary and would inflate concurrency.
        allowed_total = total_slots
        if total_pool_count >= total_slots and starving_envs:
            # Relative overflow cap: at most 1.5x total_slots (ceiling).
            # extra_cap = ceil(0.5 * total_slots)
            extra_cap = (total_slots + 1) // 2
            allowed_total = total_slots + min(len(starving_envs), extra_cap)
        if total_pool_count >= allowed_total:
            return
        
        slots_available = allowed_total - total_pool_count
        
        # Get env weights for weighted allocation
        env_weights = self._get_env_weights(environments)
        
        # Select tasks to create with weighted allocation strategy
        tasks_to_create = self._select_tasks_to_create(
            env_missing_tasks=env_missing_tasks,
            env_active_counts=env_active_counts,
            slots_available=slots_available,
            total_slots=total_slots,
            miner=miner,
            env_weights=env_weights,
        )
        
        # Create selected tasks and record allocations for rate limiting
        if tasks_to_create:
            await self._create_tasks(tasks_to_create, miner)
            self._record_allocations(hotkey, revision, tasks_to_create)

    async def _get_missing_task_ids(
        self,
        miner: Dict[str, Any],
        env: str,
        environments: Dict[str, Any]
    ) -> List[int]:
        """Get missing task IDs for a miner in an environment.
        
        Prioritizes tasks from the tail of sampling list.
        
        Returns:
            List of missing task IDs, ordered by priority (tail first)
        """
        hotkey = miner['hotkey']
        revision = miner['revision']
        
        # Get current sampling list
        env_config = environments.get(env, {})
        sampling_config = env_config.get('sampling_config', {})
        sampling_list = sampling_config.get('sampling_list', [])
        
        if not sampling_list:
            return []
        
        # Detect sampling list changes (compare lists directly to catch rotations)
        last_list = self._last_sampling_lists.get(env, [])
        if sampling_list != last_list:
            await self._handle_sampling_list_change(env, last_list, sampling_list)
            self._last_sampling_lists[env] = sampling_list
        
        # Get completed and pending task IDs
        completed_ids = await self.sample_results_dao.get_completed_task_ids(
            miner_hotkey=hotkey,
            model_revision=revision,
            env=env
        )
        
        pending_ids = await self.task_pool_dao.get_pending_task_ids_for_miner(
            miner_hotkey=hotkey,
            model_revision=revision,
            env=env
        )
        
        # Calculate missing IDs
        sampling_set = set(sampling_list)
        missing_ids = sampling_set - completed_ids - pending_ids
        
        if not missing_ids:
            return []
        
        # Prioritize tasks from tail (reverse order)
        # Tasks at the tail are less likely to be rotated out
        priority_order = []
        for task_id in reversed(sampling_list):
            if task_id in missing_ids:
                priority_order.append(task_id)
        
        return priority_order
    
    async def _handle_sampling_list_change(
        self,
        env: str,
        old_list: List[int],
        new_list: List[int]
    ):
        """Handle sampling list changes (rotation or resize).
        
        Note: Task cleanup is handled by SamplingScheduler's rotation logic.
        This method only logs the change for per-miner awareness.
        """
        old_set = set(old_list)
        new_set = set(new_list)
        
        removed_ids = old_set - new_set
        added_ids = new_set - old_set
        
        if removed_ids or added_ids:
            logger.info(
                f"Sampling list changed for {env}: "
                f"removed={len(removed_ids)}, added={len(added_ids)} "
                f"(cleanup handled by SamplingScheduler)"
            )
    
    def _select_tasks_to_create(
        self,
        env_missing_tasks: Dict[str, List[int]],
        env_active_counts: Dict[str, int],
        slots_available: int,
        total_slots: int,
        miner: Dict[str, Any],
        env_weights: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Select tasks to create using weighted allocation strategy.
        
        Strategy: Weighted target allocation with deficit-based scheduling
        
        Algorithm:
        1. Calculate target slots per env: total_slots * (weight / sum_weights)
        2. Calculate deficit: target - active_count (how many more each env needs)
        3. Allocate directly based on deficit (not proportionally to slots_available)
        4. Cap allocation by slots_available
        5. Ensure minimum 1 slot per env with positive deficit and missing tasks
        
        Example with total_slots=6, weights: game=2, lgc-v2=1, print=1 (sum=4):
        - game target: 6 * 2/4 = 3 slots
        - lgc-v2 target: 6 * 1/4 = 1.5 -> 2 slots (ceiling for fairness)
        - print target: 6 * 1/4 = 1.5 -> 1 slot (floor to not exceed total)
        
        If game has 1 active, lgc-v2 has 0, print has 2, slots_available=5:
        - game deficit: 3-1 = 2 -> allocate 2
        - lgc-v2 deficit: 2-0 = 2 -> allocate 2
        - print deficit: 1-2 = 0 -> allocate 0
        - Total allocation: 4 (within slots_available=5)
        - Remaining 1 slot: round-robin among envs with missing tasks
        
        Returns:
            List of task specs with env and task_id
        """
        if slots_available <= 0 or not env_missing_tasks:
            return []
        
        # Eligible envs are those that actually have missing tasks (i.e. tasks to create).
        # Envs with no tasks should not consume slots; this matches the original design goal.
        eligible_envs = [e for e, tasks in env_missing_tasks.items() if tasks]
        if not eligible_envs:
            return []
        
        # Deterministic ordering helpers
        def _weight(env: str) -> float:
            try:
                w = float(env_weights.get(env, 1.0))
            except Exception:
                w = 1.0
            return max(0.0, w)
        
        sorted_by_weight_desc = sorted(
            eligible_envs,
            key=lambda e: (-_weight(e), e),
        )
        
        # Stage 1: anti-starvation reservation.
        # If an eligible env has active==0, allocate 1 first to ensure it stops being starving.
        selected: List[Dict[str, Any]] = []
        planned_active_counts = {env: int(env_active_counts.get(env, 0)) for env in eligible_envs}
        
        starving_envs = [e for e in eligible_envs if planned_active_counts.get(e, 0) == 0]
        reserved = min(slots_available, len(starving_envs))
        
        if reserved > 0:
            # Uniform random sampling among starving envs (frequency fairness).
            # Note: tests should not rely on a fixed seed; in production we want true randomness.
            k = min(reserved, len(starving_envs))
            chosen_envs = random.sample(starving_envs, k) if k > 0 else []

            for env in chosen_envs:
                if not env_missing_tasks.get(env):
                    continue
                task_id = env_missing_tasks[env].pop(0)
                selected.append({'env': env, 'task_id': task_id})
                planned_active_counts[env] = planned_active_counts.get(env, 0) + 1
        
        remaining_slots = slots_available - len(selected)
        if remaining_slots <= 0:
            return selected
        
        # Stage 2: weighted fairness allocation within eligible envs.
        # Compute integer targets using floor + remainder distribution (fair under non-divisible slots).
        total_weight = sum(_weight(env) for env in eligible_envs)
        if total_weight <= 0:
            total_weight = float(len(eligible_envs))
        
        target_slots: Dict[str, int] = {}
        remainder_pool = []
        allocated_sum = 0
        for env in eligible_envs:
            raw_target = total_slots * (_weight(env) / total_weight)
            floor_target = int(raw_target)
            remainder = raw_target - floor_target
            target_slots[env] = floor_target
            allocated_sum += floor_target
            remainder_pool.append((env, remainder))
        
        # Remainder distribution tie-break:
        # 1) larger remainder first
        # 2) higher weight first
        # 3) env name ascending (deterministic & intuitive)
        remainder_pool.sort(key=lambda x: (-x[1], -_weight(x[0]), x[0]))
        for i in range(max(0, total_slots - allocated_sum)):
            if i >= len(remainder_pool):
                break
            env = remainder_pool[i][0]
            target_slots[env] += 1
        
        # Fill deficits first: allocate to envs that are below their targets.
        while remaining_slots > 0:
            deficit_envs = [
                env for env in eligible_envs
                if env_missing_tasks.get(env)
                and planned_active_counts.get(env, 0) < target_slots.get(env, 0)
            ]
            if not deficit_envs:
                break
            
            # Pick the env with largest deficit; tie-break by weight then name.
            def _deficit(env: str) -> int:
                return target_slots.get(env, 0) - planned_active_counts.get(env, 0)
            
            chosen = sorted(
                deficit_envs,
                key=lambda e: (_deficit(e), _weight(e), e),
                reverse=True,
            )[0]
            
            task_id = env_missing_tasks[chosen].pop(0)
            selected.append({'env': chosen, 'task_id': task_id})
            planned_active_counts[chosen] = planned_active_counts.get(chosen, 0) + 1
            remaining_slots -= 1
        
        # Stage 3: round-robin for any remaining slots.
        # Only used when under-quota envs have no available tasks; in that case, it is better
        # to allocate to any env with available tasks than to waste capacity.
        if remaining_slots > 0:
            envs_with_tasks = [e for e in sorted_by_weight_desc if env_missing_tasks.get(e)]
            env_index = 0
            while remaining_slots > 0 and envs_with_tasks:
                env = envs_with_tasks[env_index % len(envs_with_tasks)]
                if env_missing_tasks.get(env):
                    task_id = env_missing_tasks[env].pop(0)
                    selected.append({'env': env, 'task_id': task_id})
                    planned_active_counts[env] = planned_active_counts.get(env, 0) + 1
                    remaining_slots -= 1
                    if not env_missing_tasks[env]:
                        envs_with_tasks.remove(env)
                        continue
                env_index += 1
        
        if selected:
            env_distribution: Dict[str, int] = {}
            for task in selected:
                env_distribution[task['env']] = env_distribution.get(task['env'], 0) + 1
            logger.info(
                f"Selected {len(selected)} tasks for miner U{miner.get('uid', -1)}"
                f"({miner['hotkey'][:8]}...) - slots={total_slots}, distribution: {env_distribution}"
            )
        
        return selected
    
    async def _create_tasks(
        self,
        tasks_to_create: List[Dict[str, Any]],
        miner: Dict[str, Any]
    ):
        """Create tasks in the task pool."""
        task_list = [
            {
                'miner_hotkey': miner['hotkey'],
                'model_revision': miner['revision'],
                'model': miner['model'],
                'env': task_spec['env'],
                'task_id': task_spec['task_id'],
                'chute_id': miner['chute_id'],
            }
            for task_spec in tasks_to_create
        ]
        
        created_count = await self.task_pool_dao.batch_create_tasks(task_list)
        
        logger.info(
            f"Created {created_count} tasks for miner "
            f"U{miner.get('uid', -1)}({miner['hotkey'][:8]}...)"
        )
    
    
    async def _cleanup_removed_miners(self, removed_miners: Set[tuple]):
        """Cleanup all tasks for miners that have been removed.
        
        Args:
            removed_miners: Set of (hotkey, revision) tuples for removed miners
        """
        if not removed_miners:
            return
        
        logger.info(f"Cleaning up tasks for {len(removed_miners)} removed miners")
        
        total_deleted = 0
        for hotkey, revision in removed_miners:
            try:
                pk = self.task_pool_dao._make_pk(hotkey, revision)
                
                # Query all tasks for this miner (all envs, all statuses)
                from affine.database.client import get_client
                client = get_client()
                
                query_params = {
                    'TableName': self.task_pool_dao.table_name,
                    'KeyConditionExpression': 'pk = :pk',
                    'ExpressionAttributeValues': {':pk': {'S': pk}}
                }
                
                tasks_to_delete = []
                last_key = None
                
                while True:
                    if last_key:
                        query_params['ExclusiveStartKey'] = last_key
                    
                    response = await client.query(**query_params)
                    items = response.get('Items', [])
                    
                    # Delete only pending tasks, keep assigned tasks for executor to complete
                    for item in items:
                        task = self.task_pool_dao._deserialize(item)
                        if task.get('status') == 'pending':
                            tasks_to_delete.append(task)
                    
                    last_key = response.get('LastEvaluatedKey')
                    if not last_key:
                        break
                
                if tasks_to_delete:
                    deleted = await self.task_pool_dao._batch_delete_tasks(tasks_to_delete)
                    total_deleted += deleted
                    logger.info(
                        f"Deleted {deleted} pending tasks for removed miner "
                        f"{hotkey[:8]}...#{revision[:8]}..."
                    )
            
            except Exception as e:
                logger.error(
                    f"Error cleaning up tasks for removed miner {hotkey[:8]}...: {e}",
                    exc_info=True
                )
        
        if total_deleted > 0:
            logger.info(f"Total cleanup: removed {total_deleted} tasks for {len(removed_miners)} miners")
    
    async def _cleanup_loop(self):
        """Cleanup loop - runs every 5 minutes for all cleanup tasks.
        
        Handles:
        1. Invalid sampling tasks (env disabled or task_id not in sampling_list)
        2. Expired paused tasks (TTL exceeded)
        """
        # Wait 60s before first cleanup (let scheduler stabilize)
        await asyncio.sleep(60)
        
        while self._running:
            try:
                # Cleanup invalid sampling tasks
                await self._cleanup_invalid_sampling_tasks()
                
                # Cleanup expired paused tasks
                await self.task_pool_dao.cleanup_expired_paused_tasks()
                
                await asyncio.sleep(300)  # Run every 5 minutes
            except asyncio.CancelledError:
                logger.info("Cleanup loop cancelled")
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}", exc_info=True)
                await asyncio.sleep(60)
    
    async def _cleanup_invalid_sampling_tasks(self):
        """Cleanup all invalid sampling tasks from task pool.
        
        Valid task criteria:
        1. Environment has enabled_for_sampling=True
        2. task_id is in the environment's sampling_list
        
        All other pending tasks are considered invalid and will be deleted.
        """
        logger.info("Starting cleanup of invalid sampling tasks")
        
        # Build valid task set
        environments = await self.config_dao.get_param_value('environments', {})
        valid_tasks = set()  # Set of (env, task_id) tuples
        
        for env_name, env_config in environments.items():
            if not env_config.get('enabled_for_sampling', False):
                continue
            
            sampling_config = env_config.get('sampling_config', {})
            sampling_list = sampling_config.get('sampling_list', [])
            
            for task_id in sampling_list:
                valid_tasks.add((env_name, task_id))
        
        if not valid_tasks:
            logger.warning("No valid sampling tasks found in configuration")
            return
        
        # Count enabled environments
        enabled_env_count = sum(
            1 for env_config in environments.values()
            if env_config.get('enabled_for_sampling', False)
        )
        
        logger.info(
            f"Valid task set contains {len(valid_tasks)} tasks "
            f"across {enabled_env_count} enabled environments"
        )
        
        # Scan task pool and delete invalid tasks
        from affine.database.client import get_client
        client = get_client()
        
        # Get all valid miners
        valid_miners = await self.miners_dao.get_valid_miners()
        
        total_scanned = 0
        total_deleted = 0
        
        for miner in valid_miners:
            hotkey = miner['hotkey']
            revision = miner['revision']
            
            try:
                pk = self.task_pool_dao._make_pk(hotkey, revision)
                
                # Query all pending tasks for this miner
                query_params = {
                    'TableName': self.task_pool_dao.table_name,
                    'KeyConditionExpression': 'pk = :pk',
                    'FilterExpression': '#status = :status',
                    'ExpressionAttributeNames': {'#status': 'status'},
                    'ExpressionAttributeValues': {
                        ':pk': {'S': pk},
                        ':status': {'S': 'pending'}
                    }
                }
                
                tasks_to_delete = []
                last_key = None
                
                while True:
                    if last_key:
                        query_params['ExclusiveStartKey'] = last_key
                    
                    response = await client.query(**query_params)
                    items = response.get('Items', [])
                    
                    for item in items:
                        task = self.task_pool_dao._deserialize(item)
                        total_scanned += 1
                        
                        env = task.get('env')
                        task_id = task.get('task_id')
                        
                        # Check if this task is valid
                        if (env, task_id) not in valid_tasks:
                            tasks_to_delete.append(task)
                    
                    last_key = response.get('LastEvaluatedKey')
                    if not last_key:
                        break
                
                if tasks_to_delete:
                    deleted = await self.task_pool_dao._batch_delete_tasks(tasks_to_delete)
                    total_deleted += deleted
                    
                    # Log details for first few invalid tasks
                    if deleted > 0:
                        sample_tasks = tasks_to_delete[:3]
                        logger.info(
                            f"Deleted {deleted} invalid tasks for miner {hotkey[:8]}...#{revision[:8]}... "
                            f"(examples: {[(t['env'], t['task_id']) for t in sample_tasks]})"
                        )
            
            except Exception as e:
                logger.error(
                    f"Error cleaning up invalid tasks for miner {hotkey[:8]}...: {e}",
                    exc_info=True
                )
        
        logger.info(
            f"Cleanup completed: scanned {total_scanned} pending tasks, "
            f"deleted {total_deleted} invalid tasks"
        )


class SamplingScheduler:
    """Legacy sampling list rotation scheduler.
    
    Handles sampling list rotation and size adjustment.
    Works in conjunction with PerMinerSamplingScheduler.
    """
    
    def __init__(
        self,
        system_config_dao: Optional[SystemConfigDAO] = None,
        task_pool_dao: Optional[TaskPoolDAO] = None,
        sampling_list_manager: Optional[SamplingListManager] = None
    ):
        self.config_dao = system_config_dao or SystemConfigDAO()
        self.task_pool_dao = task_pool_dao or TaskPoolDAO()
        self.manager = sampling_list_manager or SamplingListManager()
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start rotation scheduler."""
        logger.info("Starting sampling list rotation scheduler")
        self._running = True
        self._task = asyncio.create_task(self._rotation_loop())
    
    async def stop(self):
        """Stop rotation scheduler."""
        logger.info("Stopping sampling list rotation scheduler")
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
    
    async def _rotation_loop(self):
        """Rotation loop - checks every 5 minutes."""
        while self._running:
            try:
                await self._check_and_rotate_all_envs()
                await asyncio.sleep(300)
            except asyncio.CancelledError:
                logger.info("Rotation loop cancelled")
                break
            except Exception as e:
                logger.error(f"Rotation loop error: {e}", exc_info=True)
                await asyncio.sleep(60)
    
    async def _check_and_rotate_all_envs(self):
        """Check all environments and rotate if needed."""
        from affine.core.dataset_range_resolver import resolve_dataset_range_source

        environments = await self.config_dao.get_param_value('environments', {})
        current_time = int(time.time())

        for env_name, env_config in environments.items():
            try:
                sampling_config = env_config.get('sampling_config')
                if not sampling_config:
                    continue

                # Re-resolve dynamic dataset_range from remote source
                range_source = sampling_config.get('dataset_range_source')
                if range_source:
                    old_range = sampling_config.get('dataset_range')
                    resolved_range = await resolve_dataset_range_source(
                        range_source, old_range=old_range
                    )
                    if resolved_range is not None:
                        sampling_config['dataset_range'] = resolved_range
                        environments[env_name]['sampling_config'] = sampling_config
                        await self.config_dao.set_param(
                            param_name='environments',
                            param_value=environments,
                            param_type='dict',
                            description='Environment configurations with dynamic sampling',
                            updated_by='sampling_scheduler_range_resolve'
                        )
                        logger.info(
                            f"Updated dataset_range for {env_name}: "
                            f"{old_range} -> {resolved_range}"
                        )

                # Check if list size needs adjustment
                current_size = len(sampling_config.get('sampling_list', []))
                target_size = sampling_config.get('sampling_count', 0)
                
                if current_size != target_size:
                    logger.info(
                        f"Detected size mismatch for {env_name}: "
                        f"current={current_size}, target={target_size}"
                    )
                    await self._adjust_sampling_list_size(env_name, sampling_config)
                    environments = await self.config_dao.get_param_value('environments', {})
                    sampling_config = environments[env_name]['sampling_config']
                
                # Check if rotation is needed
                rotation_enabled = sampling_config.get('rotation_enabled', True)
                if not rotation_enabled:
                    continue
                
                rotation_count = sampling_config.get('rotation_count', 0)
                if rotation_count == 0:
                    continue
                
                last_rotation = sampling_config.get('last_rotation_at', 0)
                rotation_interval = sampling_config.get('rotation_interval', 3600)
                
                if current_time - last_rotation >= rotation_interval:
                    await self._rotate_environment(env_name, sampling_config)
                    
            except Exception as e:
                logger.error(
                    f"Error checking rotation for {env_name}: {e}",
                    exc_info=True
                )
    
    async def _rotate_environment(self, env: str, sampling_config: dict):
        """Rotate sampling list for a single environment."""
        logger.info(f"Rotating sampling list for {env}")

        current_list = sampling_config['sampling_list']
        dataset_range = sampling_config['dataset_range']
        sampling_count = sampling_config['sampling_count']
        rotation_count = sampling_config['rotation_count']
        prioritize_new = 'dataset_range_source' in sampling_config

        new_list, removed_ids, added_ids = await self.manager.rotate_sampling_list(
            env=env,
            current_list=current_list,
            dataset_range=dataset_range,
            sampling_count=sampling_count,
            rotation_count=rotation_count,
            prioritize_new=prioritize_new
        )
        
        logger.info(
            f"Rotated {env}: removed={len(removed_ids)}, added={len(added_ids)}, "
            f"new_size={len(new_list)}"
        )
        
        await self._update_sampling_config(env, new_list)
        await self._cleanup_removed_tasks(env, removed_ids)
    
    async def _update_sampling_config(self, env: str, new_list: List[int]):
        """Update sampling_list in SystemConfig."""
        environments = await self.config_dao.get_param_value('environments', {})
        
        if env not in environments:
            logger.warning(f"Environment {env} not found in config during update")
            return
        
        environments[env]['sampling_config']['sampling_list'] = new_list
        environments[env]['sampling_config']['last_rotation_at'] = int(time.time())
        
        await self.config_dao.set_param(
            param_name='environments',
            param_value=environments,
            param_type='dict',
            description='Environment configurations with dynamic sampling',
            updated_by='sampling_scheduler'
        )
        
        logger.info(f"Updated sampling_list for {env} in SystemConfig")
    
    async def _adjust_sampling_list_size(self, env: str, sampling_config: dict):
        """Adjust sampling list size to match sampling_count."""
        current_list = sampling_config['sampling_list']
        current_size = len(current_list)
        target_size = sampling_config['sampling_count']
        
        if current_size == target_size:
            return
        
        logger.info(
            f"Adjusting sampling list size for {env}: {current_size} -> {target_size}"
        )
        
        prioritize_new = 'dataset_range_source' in sampling_config

        new_list, removed_ids, added_ids = await self.manager.rotate_sampling_list(
            env=env,
            current_list=current_list,
            dataset_range=sampling_config['dataset_range'],
            sampling_count=target_size,
            rotation_count=0,
            prioritize_new=prioritize_new
        )
        
        logger.info(
            f"Adjusted {env}: removed={len(removed_ids)}, added={len(added_ids)}, "
            f"new_size={len(new_list)}"
        )
        
        await self._update_sampling_config(env, new_list)
        await self._cleanup_removed_tasks(env, removed_ids)
    
    async def _cleanup_removed_tasks(self, env: str, removed_ids: List[int]):
        """Cleanup removed task IDs from TaskPool (pending only)."""
        if not removed_ids:
            return
        
        miners_dao = MinersDAO()
        valid_miners = await miners_dao.get_valid_miners()
        
        deleted_count = 0
        for miner in valid_miners:
            hotkey = miner['hotkey']
            revision = miner['revision']
            
            for task_id in removed_ids:
                pk = self.task_pool_dao._make_pk(hotkey, revision)
                sk = self.task_pool_dao._make_sk(env, 'pending', task_id)
                
                try:
                    deleted = await self.task_pool_dao.delete(pk, sk)
                    if deleted:
                        deleted_count += 1
                except Exception as e:
                    logger.debug(
                        f"Failed to delete task {env}/{task_id} for miner "
                        f"{hotkey[:8]}...#{revision[:8]}...: {e}"
                    )
        
        logger.info(
            f"Cleaned up {deleted_count} pending tasks for {len(removed_ids)} "
            f"removed task IDs in {env}"
        )