"""
Sampling Scheduler

Manages sampling list rotation and per-miner sampling pool allocation.
"""

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


class PerMinerSamplingScheduler:
    """Per-miner sampling pool scheduler.
    
    New architecture:
    1. Each miner has independent sampling concurrency limit N (default 5)
    2. Incremental scheduling every 10s (vs 300s full scan)
    3. Priority-based task selection from sampling list tail
    4. Supports dynamic sampling list rotation
    """
    
    def __init__(
        self,
        system_config_dao: Optional[SystemConfigDAO] = None,
        task_pool_dao: Optional[TaskPoolDAO] = None,
        sample_results_dao: Optional[SampleResultsDAO] = None,
        miners_dao: Optional[MinersDAO] = None,
        default_concurrency: int = 5,
        scheduling_interval: int = 10
    ):
        self.config_dao = system_config_dao or SystemConfigDAO()
        self.task_pool_dao = task_pool_dao or TaskPoolDAO()
        self.sample_results_dao = sample_results_dao or SampleResultsDAO()
        self.miners_dao = miners_dao or MinersDAO()
        
        self.default_concurrency = default_concurrency
        self.scheduling_interval = scheduling_interval
        
        # Track last known sampling lists per env
        self._last_sampling_lists: Dict[str, List[int]] = {}
        
        # Track last known valid miners for detecting additions/removals
        self._last_valid_miners: Set[tuple] = set()  # Set of (hotkey, revision) tuples
        
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start per-miner sampling scheduler."""
        logger.info(
            f"Starting per-miner sampling scheduler: "
            f"concurrency={self.default_concurrency}, interval={self.scheduling_interval}s"
        )
        self._running = True
        
        # Initialize sampling lists cache
        await self._initialize_sampling_lists()
        
        self._task = asyncio.create_task(self._scheduling_loop())
    
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
            
            # Get sampling environments
            environments = await self.config_dao.get_param_value('environments', {})
            sampling_envs = [
                env_name for env_name, env_config in environments.items()
                if env_config.get('sampling_config')
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
    
    async def _schedule_miner(
        self,
        miner: Dict[str, Any],
        sampling_envs: List[str],
        environments: Dict[str, Any]
    ):
        """Schedule sampling tasks for a single miner across all environments.
        
        Args:
            miner: Miner dict with hotkey, revision, etc.
            sampling_envs: List of environment names
            environments: Full environment configurations
        """
        hotkey = miner['hotkey']
        revision = miner['revision']
        
        # Get current pool count for this miner across all envs
        total_pool_count = 0
        for env in sampling_envs:
            pending_task_ids = await self.task_pool_dao.get_pending_task_ids_for_miner(
                miner_hotkey=hotkey,
                model_revision=revision,
                env=env
            )
            total_pool_count += len(pending_task_ids)
        
        # Check if pool is already at capacity
        if total_pool_count >= self.default_concurrency:
            return
        
        # Calculate how many tasks we can add
        slots_available = self.default_concurrency - total_pool_count
        
        # Collect missing task IDs from all environments
        env_missing_tasks: Dict[str, List[int]] = {}
        
        for env in sampling_envs:
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
        
        # Select tasks to create (randomly distributed across envs)
        tasks_to_create = self._select_tasks_to_create(
            env_missing_tasks=env_missing_tasks,
            slots_available=slots_available,
            miner=miner
        )
        
        # Create selected tasks
        if tasks_to_create:
            await self._create_tasks(tasks_to_create, miner)
    
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
        
        Removes tasks that are no longer in the sampling list.
        """
        old_set = set(old_list)
        new_set = set(new_list)
        
        removed_ids = old_set - new_set
        added_ids = new_set - old_set
        
        if removed_ids or added_ids:
            logger.info(
                f"Sampling list changed for {env}: "
                f"removed={len(removed_ids)}, added={len(added_ids)}"
            )
        
        # Clean up removed tasks from pool
        if removed_ids:
            await self._cleanup_removed_tasks(env, list(removed_ids))
    
    def _select_tasks_to_create(
        self,
        env_missing_tasks: Dict[str, List[int]],
        slots_available: int,
        miner: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Select tasks to create from missing tasks across environments.
        
        Strategy: Randomly distribute slots across environments with missing tasks.
        
        Returns:
            List of task specs with env and task_id
        """
        if slots_available <= 0:
            return []
        
        # Flatten all missing tasks with env info
        all_candidates = []
        for env, task_ids in env_missing_tasks.items():
            for task_id in task_ids:
                all_candidates.append({'env': env, 'task_id': task_id})
        
        # Randomly shuffle and select up to slots_available
        random.shuffle(all_candidates)
        selected = all_candidates[:slots_available]
        
        if selected:
            logger.info(
                f"Selected {len(selected)} tasks for miner U{miner.get('uid', -1)}"
                f"({miner['hotkey'][:8]}...) from {len(all_candidates)} candidates"
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
    
    async def _cleanup_removed_tasks(self, env: str, removed_ids: List[int]):
        """Cleanup removed task IDs from TaskPool (pending only)."""
        if not removed_ids:
            return
        
        valid_miners = await self.miners_dao.get_valid_miners()
        
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
                        f"{hotkey[:8]}...: {e}"
                    )
        
        if deleted_count > 0:
            logger.info(
                f"Cleaned up {deleted_count} pending tasks for {len(removed_ids)} "
                f"removed task IDs in {env}"
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
        environments = await self.config_dao.get_param_value('environments', {})
        current_time = int(time.time())
        
        for env_name, env_config in environments.items():
            try:
                sampling_config = env_config.get('sampling_config')
                if not sampling_config:
                    continue
                
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
        
        new_list, removed_ids, added_ids = await self.manager.rotate_sampling_list(
            env=env,
            current_list=current_list,
            dataset_range=dataset_range,
            sampling_count=sampling_count,
            rotation_count=rotation_count
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
        
        new_list, removed_ids, added_ids = await self.manager.rotate_sampling_list(
            env=env,
            current_list=current_list,
            dataset_range=sampling_config['dataset_range'],
            sampling_count=target_size,
            rotation_count=0
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