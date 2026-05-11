import time
import uuid
from typing import Dict, Any, List, Optional, Set
from affine.database.base_dao import BaseDAO
from affine.database.schema import get_table_name

from affine.core.setup import logger


class TaskPoolDAO(BaseDAO):
    """DAO for task_pool table.
    
    New Schema Design:
    - PK: MINER#{hotkey}#REV#{revision} - partition by miner
    - SK: ENV#{env}#STATUS#{status}#TASK_ID#{task_id}
    - GSI1: ENV#{env}#STATUS#{status} -> MINER#{hotkey}#REV#{revision}#TASK_ID#{task_id}
    
    Key improvements:
    - task_uuid retained as regular field (not indexed) for executor compatibility
    - MINER partition: enables O(m) cleanup instead of O(n)
    - GSI1 SK by MINER: supports efficient weighted counting
    """
    
    def __init__(self):
        self.table_name = get_table_name("task_pool")
        super().__init__()
    
    def _make_pk(self, miner_hotkey: str, model_revision: str) -> str:
        """Generate partition key by miner."""
        return f"MINER#{miner_hotkey}#REV#{model_revision}"
    
    def _make_sk(self, env: str, status: str, task_id: int) -> str:
        """Generate sort key with env, status, and task_id.

        Coerce to int defensively: sampling_list entries arriving from
        system_config are JSON-decoded and a value that was written as a
        string (e.g. via the admin CLI) flows through unchanged. A single
        str in the list crashes the whole batch_create_tasks call —
        which is the entire sampling pipeline — so we normalise here at
        the DAO boundary instead of trusting upstream typing.
        """
        return f"ENV#{env}#STATUS#{status}#TASK_ID#{int(task_id):06d}"

    def _make_gsi1_pk(self, env: str, status: str) -> str:
        """Generate GSI1 partition key for env+status queries."""
        return f"ENV#{env}#STATUS#{status}"

    def _make_gsi1_sk(self, miner_hotkey: str, model_revision: str, task_id: int) -> str:
        """Generate GSI1 sort key for miner+task_id queries."""
        return f"MINER#{miner_hotkey}#REV#{model_revision}#TASK_ID#{int(task_id):06d}"
    
    async def batch_create_tasks(
        self,
        tasks: List[Dict[str, Any]],
        ttl_days: int = 3
    ) -> int:
        """Batch create multiple tasks.
        
        Args:
            tasks: List of task dicts with keys:
                - miner_hotkey
                - model_revision
                - model
                - env
                - task_id (integer)
                - chute_id
            ttl_days: Days until tasks expire
            
        Returns:
            Number of tasks created
        """
        items = []
        created_at = int(time.time())
        status = 'pending'
        
        for task_info in tasks:
            # Coerce task_id to int once at the boundary so downstream
            # consumers (sk, gsi1_sk, executor formatting) all see a
            # consistent integer regardless of how upstream tagged it.
            task_id = int(task_info['task_id'])
            pk = self._make_pk(task_info['miner_hotkey'], task_info['model_revision'])
            sk = self._make_sk(task_info['env'], status, task_id)
            task_uuid = str(uuid.uuid4())

            item = {
                'pk': pk,
                'sk': sk,
                'task_uuid': task_uuid,
                'task_id': task_id,
                'miner_hotkey': task_info['miner_hotkey'],
                'model_revision': task_info['model_revision'],
                'model': task_info['model'],
                'env': task_info['env'],
                'chute_id': task_info['chute_id'],
                'status': status,
                'created_at': created_at,
                'assigned_to': None,
                'assigned_at': None,
                'retry_count': 0,
                # 1 = no retry. Tasks are partitioned per-miner, so retrying
                # only re-hits the same miner's chute — pointless when the
                # failure is persistent (overload, bad model, dead miner) and
                # noisy when it isn't (the rotator will recreate the task on
                # the next cycle and the scheduler can pick a different miner).
                'max_retries': 1,
                'last_error': None,
                'last_error_code': None,
                'last_failed_at': None,
                'ttl': self.get_ttl(ttl_days),
                'gsi1_pk': self._make_gsi1_pk(task_info['env'], status),
                'gsi1_sk': self._make_gsi1_sk(
                    task_info['miner_hotkey'],
                    task_info['model_revision'],
                    task_id,
                ),
            }
            items.append(item)
        
        await self.batch_write(items)
        return len(items)
    
    async def get_pending_tasks_by_env(
        self,
        env: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get pending tasks for a specific environment.
        
        Uses GSI1 for efficient querying.
        
        Args:
            env: Environment name
            limit: Maximum number of tasks to return
            
        Returns:
            List of pending tasks (sorted by miner for efficient grouping)
        """
        from affine.database.client import get_client
        client = get_client()
        
        gsi1_pk = self._make_gsi1_pk(env, 'pending')
        
        params = {
            'TableName': self.table_name,
            'IndexName': 'env-status-index',
            'KeyConditionExpression': 'gsi1_pk = :pk',
            'ExpressionAttributeValues': {':pk': {'S': gsi1_pk}}
        }
        
        if limit:
            params['Limit'] = limit
        
        all_tasks = []
        last_key = None
        
        while True:
            if last_key:
                params['ExclusiveStartKey'] = last_key
            
            response = await client.query(**params)
            items = response.get('Items', [])
            all_tasks.extend([self._deserialize(item) for item in items])
            
            last_key = response.get('LastEvaluatedKey')
            if not last_key or (limit and len(all_tasks) >= limit):
                break
        
        return all_tasks[:limit] if limit else all_tasks
    
    async def get_task_by_composite_key(
        self,
        miner_hotkey: str,
        model_revision: str,
        env: str,
        task_id: int
    ) -> Optional[Dict[str, Any]]:
        """Get a task by composite key (miner, env, task_id).
        
        Args:
            miner_hotkey: Miner's hotkey
            model_revision: Model revision
            env: Environment name
            task_id: Task ID
            
        Returns:
            Task if found, None otherwise
        """
        from affine.database.client import get_client
        client = get_client()
        
        pk = self._make_pk(miner_hotkey, model_revision)
        
        params = {
            'TableName': self.table_name,
            'KeyConditionExpression': 'pk = :pk AND begins_with(sk, :sk_prefix)',
            'ExpressionAttributeValues': {
                ':pk': {'S': pk},
                ':sk_prefix': {'S': f'ENV#{env}#STATUS#'}
            }
        }
        
        response = await client.query(**params)
        items = response.get('Items', [])
        
        for item in items:
            task = self._deserialize(item)
            if task.get('task_id') == task_id:
                return task
        
        return None
    
    async def get(self, pk: str, sk: str) -> Optional[Dict[str, Any]]:
        """Get task by primary key (PK, SK).
        
        Direct GetItem query - O(1) lookup.
        
        Args:
            pk: Partition key
            sk: Sort key
            
        Returns:
            Task dict if found, None otherwise
        """
        from affine.database.client import get_client
        client = get_client()
        
        params = {
            'TableName': self.table_name,
            'Key': {
                'pk': {'S': pk},
                'sk': {'S': sk}
            }
        }
        
        response = await client.get_item(**params)
        item = response.get('Item')
        
        return self._deserialize(item) if item else None
    
    async def get_task_by_uuid(self, task_uuid: str) -> Optional[Dict[str, Any]]:
        """Get a task by UUID.
        
        Uses Scan + Filter since UUID is not indexed.
        This is acceptable because:
        1. Only used during cache miss (low frequency after warmup)
        2. Task pool size is bounded (typically < 100k tasks)
        3. Alternative would be adding GSI2 (UUID index) which we want to avoid
        
        Args:
            task_uuid: Task UUID
            
        Returns:
            Task if found, None otherwise
        """
        from affine.database.client import get_client
        client = get_client()
        
        params = {
            'TableName': self.table_name,
            'FilterExpression': 'task_uuid = :uuid',
            'ExpressionAttributeValues': {':uuid': {'S': task_uuid}}
        }
        
        last_key = None
        
        while True:
            if last_key:
                params['ExclusiveStartKey'] = last_key
            
            response = await client.scan(**params)
            items = response.get('Items', [])
            
            if items:
                return self._deserialize(items[0])
            
            last_key = response.get('LastEvaluatedKey')
            if not last_key:
                break
        
        return None
    
    async def assign_task(
        self,
        task: Dict[str, Any],
        executor_hotkey: str
    ) -> Dict[str, Any]:
        """Assign a task to an executor.
        
        Changes status from 'pending' to 'assigned'.
        
        Args:
            task: Task dict
            executor_hotkey: Executor's hotkey
            
        Returns:
            Updated task
        """
        await self.delete(task['pk'], task['sk'])
        
        new_status = 'assigned'
        assigned_at = int(time.time())
        
        new_sk = self._make_sk(task['env'], new_status, task['task_id'])
        new_gsi1_pk = self._make_gsi1_pk(task['env'], new_status)
        new_gsi1_sk = self._make_gsi1_sk(
            task['miner_hotkey'],
            task['model_revision'],
            task['task_id']
        )
        
        task['sk'] = new_sk
        task['status'] = new_status
        task['assigned_to'] = executor_hotkey
        task['assigned_at'] = assigned_at
        task['gsi1_pk'] = new_gsi1_pk
        task['gsi1_sk'] = new_gsi1_sk
        
        await self.put(task)
        return task
    
    async def batch_assign_tasks(
        self,
        tasks: List[Dict[str, Any]],
        executor_hotkey: str
    ) -> List[Dict[str, Any]]:
        """Batch assign multiple tasks to an executor.
        
        Uses BatchWriteItem for efficient bulk operations.
        Each task requires 2 operations (delete + put), batched at 25 operations per request.
        
        Args:
            tasks: List of task dicts
            executor_hotkey: Executor's hotkey
            
        Returns:
            List of updated tasks
        """
        if not tasks:
            return []
        
        from affine.database.client import get_client
        client = get_client()
        
        new_status = 'assigned'
        assigned_at = int(time.time())
        
        # Prepare updated tasks
        updated_tasks = []
        for task in tasks:
            new_sk = self._make_sk(task['env'], new_status, task['task_id'])
            new_gsi1_pk = self._make_gsi1_pk(task['env'], new_status)
            new_gsi1_sk = self._make_gsi1_sk(
                task['miner_hotkey'],
                task['model_revision'],
                task['task_id']
            )
            
            updated_task = {
                **task,
                'sk': new_sk,
                'status': new_status,
                'assigned_to': executor_hotkey,
                'assigned_at': assigned_at,
                'gsi1_pk': new_gsi1_pk,
                'gsi1_sk': new_gsi1_sk,
            }
            updated_tasks.append(updated_task)
        
        # Build batch requests (delete old + put new for each task)
        all_requests = []
        for original_task, updated_task in zip(tasks, updated_tasks):
            # Delete old record
            all_requests.append({
                'DeleteRequest': {
                    'Key': {
                        'pk': {'S': original_task['pk']},
                        'sk': {'S': original_task['sk']}
                    }
                }
            })
            # Put new record
            all_requests.append({
                'PutRequest': {
                    'Item': self._serialize(updated_task)
                }
            })
        
        # Execute in batches of 25 (DynamoDB limit)
        batch_size = 25
        for i in range(0, len(all_requests), batch_size):
            batch = all_requests[i:i + batch_size]
            
            params = {
                'RequestItems': {
                    self.table_name: batch
                }
            }
            
            try:
                await client.batch_write_item(**params)
            except Exception as e:
                logger.error(f"Batch assign failed for batch starting at {i}: {e}")
                raise
        
        return updated_tasks
    
    async def complete_task(self, task: Dict[str, Any]) -> bool:
        """Mark task as completed and delete it from pool.
        
        Args:
            task: Task dict
            
        Returns:
            True if deleted successfully
        """
        return await self.delete(task['pk'], task['sk'])
    
    async def fail_task(
        self,
        task: Dict[str, Any],
        error_message: str,
        error_code: str = 'EXECUTION_ERROR'
    ) -> Optional[Dict[str, Any]]:
        """Record task failure and handle retry logic.

        If retry_count < max_retries, reset status to 'pending' (the
        scheduler will re-assign it). Otherwise DELETE the task and
        return None — the scheduler will either recreate it next cycle
        (if the sampling list still contains this task_id) or simply
        move on (rotation removes it within minutes). This replaces
        the old 4-hour 'paused' DB state, unnecessary given the small
        sampling windows and short rotation cycles.

        Args:
            task: Task dict
            error_message: Error description
            error_code: Error classification code

        Returns:
            Updated task if still retryable; None if deleted on max retries.
        """
        # Save old PK/SK for deletion
        old_pk = task['pk']
        old_sk = task['sk']

        retry_count = task.get('retry_count', 0) + 1
        max_retries = task.get('max_retries')

        if retry_count >= max_retries:
            logger.info(
                f"Task deleted after {retry_count} retries: "
                f"miner={task['miner_hotkey'][:12]}... env={task['env']} "
                f"task_id={task['task_id']} error={error_message[:100]}"
            )
            await self.delete(old_pk, old_sk)
            return None

        # Still have retries left, reset to pending
        new_status = 'pending'
        new_sk = self._make_sk(task['env'], new_status, task['task_id'])
        new_gsi1_pk = self._make_gsi1_pk(task['env'], new_status)
        new_gsi1_sk = self._make_gsi1_sk(
            task['miner_hotkey'],
            task['model_revision'],
            task['task_id']
        )
        
        task['sk'] = new_sk
        task['status'] = new_status
        task['retry_count'] = retry_count
        task['last_error'] = error_message
        task['last_error_code'] = error_code
        task['last_failed_at'] = int(time.time())
        task['assigned_to'] = None
        task['assigned_at'] = None
        task['gsi1_pk'] = new_gsi1_pk
        task['gsi1_sk'] = new_gsi1_sk
        
        await self.put(task)
        await self.delete(old_pk, old_sk)
        
        return task
    
    
    async def get_pending_task_ids_for_miner(
        self,
        miner_hotkey: str,
        model_revision: str,
        env: str,
    ) -> Set[int]:
        """Get set of task_ids in queue for a miner's env.

        Used by task generator to avoid creating duplicate tasks, and
        by the scheduler to count active tasks against the miner's slot
        budget. Only `pending` and `assigned` statuses are ever present
        since `fail_task` now deletes on max retries instead of marking
        tasks `paused`.
        """
        from affine.database.client import get_client
        client = get_client()

        pk = self._make_pk(miner_hotkey, model_revision)

        params = {
            'TableName': self.table_name,
            'KeyConditionExpression': 'pk = :pk AND begins_with(sk, :sk_prefix)',
            'ExpressionAttributeValues': {
                ':pk': {'S': pk},
                ':sk_prefix': {'S': f'ENV#{env}#STATUS#'}
            },
            'ProjectionExpression': 'task_id, #status',
            'ExpressionAttributeNames': {'#status': 'status'}
        }

        all_items = []
        last_key = None

        while True:
            if last_key:
                params['ExclusiveStartKey'] = last_key

            response = await client.query(**params)
            items = response.get('Items', [])
            all_items.extend([self._deserialize(item) for item in items])

            last_key = response.get('LastEvaluatedKey')
            if not last_key:
                break

        return {
            item['task_id'] for item in all_items
            if item.get('status') in ('pending', 'assigned')
        }
    
    async def cleanup_invalid_tasks(
        self,
        valid_miners: List[Dict[str, Any]]
    ) -> int:
        """Remove tasks for miners that are no longer valid.
        
        New efficient implementation: query by miner partition and batch delete.
        
        Args:
            valid_miners: List of valid miner dicts with 'hotkey' and 'revision'
            
        Returns:
            Number of tasks cleaned up
        """
        from affine.database.client import get_client
        client = get_client()
        
        valid_set = {
            (m['hotkey'], m.get('model_revision'))
            for m in valid_miners
        }
        
        params = {
            'TableName': self.table_name,
            'ProjectionExpression': 'pk, miner_hotkey, model_revision'
        }
        
        all_miner_keys = set()
        last_key = None
        
        while True:
            if last_key:
                params['ExclusiveStartKey'] = last_key
            
            response = await client.scan(**params)
            items = response.get('Items', [])
            
            for item in items:
                task = self._deserialize(item)
                all_miner_keys.add((task['miner_hotkey'], task['model_revision']))
            
            last_key = response.get('LastEvaluatedKey')
            if not last_key:
                break
        
        invalid_keys = all_miner_keys - valid_set
        
        if not invalid_keys:
            logger.info("No invalid miners found in task pool")
            return 0
        
        logger.info(f"Found {len(invalid_keys)} invalid miners to clean up, valid_set: {valid_set}, invalid_keys: {invalid_keys}")
        
        total_deleted = 0
        
        for hotkey, revision in invalid_keys:
            try:
                pk = self._make_pk(hotkey, revision)
                
                query_params = {
                    'TableName': self.table_name,
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
                    
                    # Only delete tasks with 'pending' status, skip 'assigned' tasks
                    for item in items:
                        task = self._deserialize(item)
                        if task.get('status') == 'pending':
                            tasks_to_delete.append(task)
                    
                    last_key = response.get('LastEvaluatedKey')
                    if not last_key:
                        break
                
                if tasks_to_delete:
                    deleted = await self._batch_delete_tasks(tasks_to_delete)
                    total_deleted += deleted
                    logger.info(
                        f"Deleted {deleted} pending tasks for invalid miner "
                        f"{hotkey[:12]}...#{revision} (skipped assigned tasks)"
                    )
            
            except Exception as e:
                logger.error(
                    f"Error cleaning up tasks for miner {hotkey[:12]}...#{revision}: {e}",
                    exc_info=True
                )
        
        logger.info(f"Cleanup complete: removed {total_deleted} tasks")
        return total_deleted
    
    async def _batch_delete_tasks(self, tasks: List[Dict[str, Any]]) -> int:
        """Batch delete tasks using DynamoDB BatchWriteItem."""
        from affine.database.client import get_client
        client = get_client()
        
        deleted_count = 0
        batch_size = 25
        
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            
            delete_requests = [
                {
                    'DeleteRequest': {
                        'Key': {
                            'pk': {'S': task['pk']},
                            'sk': {'S': task['sk']}
                        }
                    }
                }
                for task in batch
            ]
            
            params = {
                'RequestItems': {
                    self.table_name: delete_requests
                }
            }
            
            try:
                await client.batch_write_item(**params)
                deleted_count += len(batch)
            except Exception as e:
                logger.error(f"Batch delete failed: {e}")
        
        return deleted_count
    
    async def get_miner_task_counts(
        self,
        env: str,
        status: str = 'pending'
    ) -> Dict[str, int]:
        """Get task count per miner for an environment.
        
        Uses GSI1 to efficiently aggregate counts by miner.
        
        Algorithm:
        1. Query GSI1 by env+status (all pending/assigned tasks)
        2. Extract miner key from gsi1_sk
        3. Count tasks per miner
        
        Args:
            env: Environment name
            status: Task status (default: pending)
            
        Returns:
            Dict mapping "hotkey#revision" to task count
            Example: {'hotkey1#rev1': 100, 'hotkey2#rev2': 50}
        """
        from affine.database.client import get_client
        client = get_client()
        
        gsi1_pk = self._make_gsi1_pk(env, status)
        
        params = {
            'TableName': self.table_name,
            'IndexName': 'env-status-index',
            'KeyConditionExpression': 'gsi1_pk = :pk',
            'ExpressionAttributeValues': {':pk': {'S': gsi1_pk}},
            'ProjectionExpression': 'gsi1_sk'  # Only fetch gsi1_sk
        }
        
        miner_counts: Dict[str, int] = {}
        last_key = None
        
        while True:
            if last_key:
                params['ExclusiveStartKey'] = last_key
            
            response = await client.query(**params)
            items = response.get('Items', [])
            
            # Extract miner from gsi1_sk: MINER#{hotkey}#REV#{revision}#TASK_ID#{task_id}
            for item in items:
                gsi1_sk = item['gsi1_sk']['S']
                # Parse: MINER#xxx#REV#yyy#TASK_ID#zzz
                parts = gsi1_sk.split('#')
                if len(parts) >= 4:
                    miner_key = f"{parts[1]}#{parts[3]}"  # hotkey#revision
                    miner_counts[miner_key] = miner_counts.get(miner_key, 0) + 1
            
            last_key = response.get('LastEvaluatedKey')
            if not last_key:
                break
        
        return miner_counts
    
    async def get_pending_tasks_for_miner(
        self,
        env: str,
        miner_hotkey: str,
        model_revision: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get pending tasks for a specific miner in an environment.
        
        Uses primary key query for efficiency.
        
        Args:
            env: Environment name
            miner_hotkey: Miner's hotkey
            model_revision: Model revision
            limit: Maximum number of tasks
            
        Returns:
            List of pending tasks
        """
        from affine.database.client import get_client
        client = get_client()
        
        pk = self._make_pk(miner_hotkey, model_revision)
        sk_prefix = f"ENV#{env}#STATUS#pending#"
        
        params = {
            'TableName': self.table_name,
            'KeyConditionExpression': 'pk = :pk AND begins_with(sk, :sk_prefix)',
            'ExpressionAttributeValues': {
                ':pk': {'S': pk},
                ':sk_prefix': {'S': sk_prefix}
            },
            'Limit': limit
        }
        
        response = await client.query(**params)
        items = response.get('Items', [])
        
        return [self._deserialize(item) for item in items]
    
    async def get_tasks_by_miner(
        self,
        miner_hotkey: str,
        model_revision: str,
        env: str
    ) -> List[Dict[str, Any]]:
        """Get all tasks for a specific miner in an environment.
        
        Returns tasks regardless of status (pending, assigned, failed).
        
        Args:
            miner_hotkey: Miner's hotkey
            model_revision: Model revision
            env: Environment name
            
        Returns:
            List of all tasks for this miner in the environment
        """
        from affine.database.client import get_client
        client = get_client()
        
        pk = self._make_pk(miner_hotkey, model_revision)
        sk_prefix = f"ENV#{env}#STATUS#"
        
        params = {
            'TableName': self.table_name,
            'KeyConditionExpression': 'pk = :pk AND begins_with(sk, :sk_prefix)',
            'ExpressionAttributeValues': {
                ':pk': {'S': pk},
                ':sk_prefix': {'S': sk_prefix}
            }
        }
        
        all_tasks = []
        last_key = None
        
        while True:
            if last_key:
                params['ExclusiveStartKey'] = last_key
            
            response = await client.query(**params)
            items = response.get('Items', [])
            all_tasks.extend([self._deserialize(item) for item in items])
            
            last_key = response.get('LastEvaluatedKey')
            if not last_key:
                break
        
        return all_tasks

    async def get_pool_stats(self, env: str) -> Dict[str, int]:
        """Get statistics about the task pool for an environment.
        
        Uses GSI1 COUNT query for efficiency.
        
        Args:
            env: Environment name
            
        Returns:
            Dict with counts: pending, assigned, failed
        """
        from affine.database.client import get_client
        client = get_client()
        
        stats = {
            'pending': 0,
            'assigned': 0,
            'failed': 0
        }
        
        for status in stats.keys():
            gsi1_pk = self._make_gsi1_pk(env, status)
            
            total_count = 0
            last_key = None
            
            while True:
                params = {
                    'TableName': self.table_name,
                    'IndexName': 'env-status-index',
                    'KeyConditionExpression': 'gsi1_pk = :pk',
                    'ExpressionAttributeValues': {':pk': {'S': gsi1_pk}},
                    'Select': 'COUNT'
                }
                
                if last_key:
                    params['ExclusiveStartKey'] = last_key
                
                response = await client.query(**params)
                total_count += response.get('Count', 0)
                
                last_key = response.get('LastEvaluatedKey')
                if not last_key:
                    break
            
            stats[status] = total_count
        
        return stats
    
    async def delete_all_assigned_tasks(self) -> int:
        """Delete all assigned tasks on startup.
        
        Called by scheduler service on startup to clean up orphaned assigned tasks
        from previous runs (executor processes are lost on restart).
        
        Uses Scan to find all assigned tasks, then batch deletes them.
        
        Returns:
            Number of tasks deleted
        """
        from affine.database.client import get_client
        client = get_client()
        
        # Scan all assigned tasks
        params = {
            'TableName': self.table_name,
            'FilterExpression': '#status = :status',
            'ExpressionAttributeNames': {'#status': 'status'},
            'ExpressionAttributeValues': {':status': {'S': 'assigned'}}
        }
        
        all_assigned_tasks = []
        last_key = None
        
        while True:
            if last_key:
                params['ExclusiveStartKey'] = last_key
            
            response = await client.scan(**params)
            items = response.get('Items', [])
            
            for item in items:
                all_assigned_tasks.append(self._deserialize(item))
            
            last_key = response.get('LastEvaluatedKey')
            if not last_key:
                break
        
        if not all_assigned_tasks:
            logger.info("No assigned tasks to delete on startup")
            return 0
        
        logger.info(f"Found {len(all_assigned_tasks)} assigned tasks to delete on startup")
        
        # Batch delete all assigned tasks
        deleted_count = await self._batch_delete_tasks(all_assigned_tasks)
        
        logger.info(f"Deleted {deleted_count} assigned tasks on startup")
        return deleted_count
    
