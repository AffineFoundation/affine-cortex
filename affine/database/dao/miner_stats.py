"""
Miner Stats DAO

Manages historical miner metadata and real-time sampling statistics.
"""

import time
from typing import Dict, Any, List, Optional
from affine.database.base_dao import BaseDAO
from affine.database.schema import get_table_name
from affine.core.setup import logger


class MinerStatsDAO(BaseDAO):
    """DAO for miner_stats table.
    
    Schema Design:
    - PK: HOTKEY#{hotkey} - partition by hotkey
    - SK: REV#{revision} - each revision is a separate record
    - GSI: last-updated-index for cleanup queries
    
    Query Patterns:
    1. Get specific miner stats: get(hotkey, revision)
    2. Get all revisions for a hotkey: query by PK
    3. Get all historical miners: get_all_historical_miners()
    4. Cleanup inactive miners: cleanup_inactive_miners()
    """
    
    def __init__(self):
        self.table_name = get_table_name("miner_stats")
        super().__init__()
    
    def _make_pk(self, hotkey: str) -> str:
        """Generate partition key."""
        return f"HOTKEY#{hotkey}"
    
    def _make_sk(self, revision: str) -> str:
        """Generate sort key."""
        return f"REV#{revision}"
    
    async def update_miner_info(
        self,
        hotkey: str,
        revision: str,
        model: str,
        rank: Optional[int] = None,
        weight: Optional[float] = None,
        is_online: bool = True
    ) -> Dict[str, Any]:
        """Update miner basic information.
        
        Creates new record on first call, updates existing record on subsequent calls.
        Automatically updates first_seen_at and last_updated_at timestamps.
        Tracks historical best rank and weight.
        
        Args:
            hotkey: Miner SS58 hotkey
            revision: Model Git commit hash
            model: HuggingFace model repository
            rank: Current rank (1-256)
            weight: Current weight
            is_online: Whether miner is currently online
            
        Returns:
            Updated miner stats record
        """
        pk = self._make_pk(hotkey)
        sk = self._make_sk(revision)
        
        # Query existing record
        existing = await self.get(pk, sk)
        
        current_time = int(time.time())
        
        if existing:
            # Update existing record
            item = existing.copy()
            item['last_updated_at'] = current_time
            item['is_currently_online'] = is_online
            
            # Update historical best metrics
            if rank is not None and (item.get('best_rank', 999) > rank):
                item['best_rank'] = rank
            if weight is not None and (item.get('best_weight', 0) < weight):
                item['best_weight'] = weight
        else:
            # Create new record with all fields initialized
            item = {
                'pk': pk,
                'sk': sk,
                'hotkey': hotkey,
                'revision': revision,
                'model': model,
                'first_seen_at': current_time,
                'last_updated_at': current_time,
                'best_rank': rank if rank is not None else 999,
                'best_weight': weight if weight is not None else 0.0,
                'is_currently_online': is_online,
                'sampling_stats': {},
                'env_stats': {},
                'sampling_slots': 6,  # Default slots
                'slots_last_adjusted_at': 0  # Never adjusted
            }
        
        await self.put(item)
        return item
    
    async def update_sampling_stats(
        self,
        hotkey: str,
        revision: str,
        env_stats: Dict[str, Dict[str, Any]]
    ):
        """Update sampling statistics for a miner.
        
        Updates per-environment statistics and calculates global aggregated statistics.
        Uses atomic operations to avoid race conditions in concurrent updates.
        
        Args:
            hotkey: Miner hotkey
            revision: Model revision
            env_stats: Per-environment statistics
                {
                    "affine:ded-v2": {
                        "last_15min": {
                            "samples": 100,
                            "success": 95,
                            "rate_limit_errors": 3,
                            "timeout_errors": 1,
                            "other_errors": 1,
                            "success_rate": 0.95,
                            "samples_per_min": 6.67
                        },
                        "last_1hour": {...},
                        "last_6hours": {...},
                        "last_24hours": {...}
                    },
                    ...
                }
        """
        from affine.database.client import get_client
        client = get_client()
        
        pk = self._make_pk(hotkey)
        sk = self._make_sk(revision)
        
        # First attempt: try atomic update if record exists
        try:
            # Build update expression for env_stats
            update_parts = []
            expr_names = {'#last_updated': 'last_updated_at'}
            expr_values = {':timestamp': {'N': str(int(time.time()))}}
            
            # Update each environment's stats atomically
            for i, (env, stats) in enumerate(env_stats.items()):
                env_key = f'#env{i}'
                val_key = f':env{i}'
                expr_names[env_key] = env
                expr_names['#env_stats'] = 'env_stats'
                expr_values[val_key] = self._serialize({'dummy': stats})['dummy']
                update_parts.append(f'#env_stats.{env_key} = {val_key}')
            
            # Set update expression
            update_expr = f"SET {', '.join(update_parts)}, #last_updated = :timestamp"
            
            # Execute atomic update
            response = await client.update_item(
                TableName=self.table_name,
                Key={'pk': {'S': pk}, 'sk': {'S': sk}},
                UpdateExpression=update_expr,
                ExpressionAttributeNames=expr_names,
                ExpressionAttributeValues=expr_values,
                ConditionExpression='attribute_exists(pk)',
                ReturnValues='ALL_NEW'
            )
            
            # Deserialize updated item
            updated_item = self._deserialize(response['Attributes'])
            
        except client.exceptions.ConditionalCheckFailedException:
            # Record doesn't exist, create it with calculated global stats
            logger.warning(f"Miner stats not found: {hotkey}#{revision}, creating new record")
            
            # Calculate global stats from new env_stats before creating record
            global_stats = self._calculate_global_stats(env_stats)
            
            updated_item = {
                'pk': pk,
                'sk': sk,
                'hotkey': hotkey,
                'revision': revision,
                'model': '',
                'first_seen_at': int(time.time()),
                'last_updated_at': int(time.time()),
                'best_rank': 999,
                'best_weight': 0.0,
                'is_currently_online': True,
                'sampling_stats': global_stats,
                'env_stats': env_stats
            }
            await self.put(updated_item)
            return  # Exit early, no need for second update
        
        # Calculate global aggregated statistics from updated env_stats
        global_stats = self._calculate_global_stats(updated_item.get('env_stats', {}))
        
        # Update sampling_stats with another atomic operation
        await client.update_item(
            TableName=self.table_name,
            Key={'pk': {'S': pk}, 'sk': {'S': sk}},
            UpdateExpression='SET #sampling_stats = :stats',
            ExpressionAttributeNames={'#sampling_stats': 'sampling_stats'},
            ExpressionAttributeValues={':stats': self._serialize({'dummy': global_stats})['dummy']}
        )
    
    def _calculate_global_stats(self, env_stats: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate global aggregated statistics from per-environment stats.
        
        Optimized: Single pass through env_stats to aggregate all windows.
        
        Args:
            env_stats: Per-environment statistics
            
        Returns:
            Global aggregated statistics
        """
        windows = ["last_15min", "last_1hour", "last_6hours", "last_24hours"]
        
        # Initialize global stats for all windows
        global_stats = {
            window: {
                'samples': 0,
                'success': 0,
                'rate_limit_errors': 0,
                'timeout_errors': 0,
                'other_errors': 0,
                'samples_per_min_sum': 0.0,
                'env_count': 0
            }
            for window in windows
        }
        
        # Single pass aggregation
        for env, stats in env_stats.items():
            for window in windows:
                if window in stats:
                    wstats = stats[window]
                    global_stats[window]['samples'] += wstats.get('samples', 0)
                    global_stats[window]['success'] += wstats.get('success', 0)
                    global_stats[window]['rate_limit_errors'] += wstats.get('rate_limit_errors', 0)
                    global_stats[window]['timeout_errors'] += wstats.get('timeout_errors', 0)
                    global_stats[window]['other_errors'] += wstats.get('other_errors', 0)
                    global_stats[window]['samples_per_min_sum'] += wstats.get('samples_per_min', 0.0)
                    global_stats[window]['env_count'] += 1
        
        # Calculate derived metrics
        for window in windows:
            if global_stats[window]['samples'] > 0:
                global_stats[window]['success_rate'] = (
                    global_stats[window]['success'] / global_stats[window]['samples']
                )
            else:
                global_stats[window]['success_rate'] = 0.0
            
            if global_stats[window]['env_count'] > 0:
                global_stats[window]['samples_per_min'] = (
                    global_stats[window]['samples_per_min_sum'] / global_stats[window]['env_count']
                )
            else:
                global_stats[window]['samples_per_min'] = 0.0
            
            # Remove temporary fields
            del global_stats[window]['samples_per_min_sum']
            del global_stats[window]['env_count']
        
        return global_stats
    
    async def get_miner_stats(
        self,
        hotkey: str,
        revision: str
    ) -> Optional[Dict[str, Any]]:
        """Get miner statistics by hotkey and revision.
        
        Args:
            hotkey: Miner hotkey
            revision: Model revision
            
        Returns:
            Miner stats record or None if not found
        """
        pk = self._make_pk(hotkey)
        sk = self._make_sk(revision)
        return await self.get(pk, sk)
    
    async def get_all_historical_miners(self) -> List[Dict[str, Any]]:
        """Get all historical miner records.
        
        Performs full table scan to retrieve all miner stats.
        Useful for historical data queries and cleanup operations.
        
        Returns:
            List of all miner stats records
        """
        from affine.database.client import get_client
        client = get_client()
        
        params = {'TableName': self.table_name}
        
        all_miners = []
        last_key = None
        
        while True:
            if last_key:
                params['ExclusiveStartKey'] = last_key
            
            response = await client.scan(**params)
            items = response.get('Items', [])
            all_miners.extend([self._deserialize(item) for item in items])
            
            last_key = response.get('LastEvaluatedKey')
            if not last_key:
                break
        
        return all_miners
    
    async def cleanup_inactive_miners(
        self,
        inactive_days: int = 30,
        dry_run: bool = True
    ) -> List[Dict[str, Any]]:
        """Cleanup long-inactive miners.
        
        Finds miners that haven't been updated for the specified number of days
        and have never had any weight (best_weight == 0).
        
        Args:
            inactive_days: Inactive threshold in days
            dry_run: If True, only return list without deleting
            
        Returns:
            List of miners matching cleanup criteria
        """
        current_time = int(time.time())
        cutoff_time = current_time - (inactive_days * 86400)
        
        # Get all miners
        all_miners = await self.get_all_historical_miners()
        
        # Filter inactive miners
        inactive_miners = [
            m for m in all_miners
            if m.get('last_updated_at', 0) < cutoff_time
            and m.get('best_weight', 0) == 0
        ]
        
        if not dry_run:
            # Batch delete
            for miner in inactive_miners:
                await self.delete(miner['pk'], miner['sk'])
            
            logger.info(f"Cleaned up {len(inactive_miners)} inactive miners")
        
        return inactive_miners
    
    async def get_miner_slots(
        self,
        hotkey: str,
        revision: str
    ) -> Optional[int]:
        """Get sampling slots for a miner.
        
        Args:
            hotkey: Miner hotkey
            revision: Model revision
            
        Returns:
            Number of slots (None if not found, caller should use default)
        """
        pk = self._make_pk(hotkey)
        sk = self._make_sk(revision)
        item = await self.get(pk, sk)
        
        if not item:
            return None
        
        return item.get('sampling_slots')
    
    async def update_sampling_slots(
        self,
        hotkey: str,
        revision: str,
        slots: int,
        adjusted_at: int
    ) -> bool:
        """Update sampling slots for a miner.
        
        Args:
            hotkey: Miner's hotkey
            revision: Model revision
            slots: New slots value (3-10)
            adjusted_at: Timestamp of adjustment
            
        Returns:
            True if successful
        """
        from affine.database.client import get_client
        client = get_client()
        
        pk = self._make_pk(hotkey)
        sk = self._make_sk(revision)
        
        try:
            await client.update_item(
                TableName=self.table_name,
                Key={
                    'pk': {'S': pk},
                    'sk': {'S': sk}
                },
                UpdateExpression='SET sampling_slots = :slots, slots_last_adjusted_at = :adjusted_at, last_updated_at = :now',
                ExpressionAttributeValues={
                    ':slots': {'N': str(slots)},
                    ':adjusted_at': {'N': str(adjusted_at)},
                    ':now': {'N': str(int(time.time()))}
                },
                ConditionExpression='attribute_exists(pk)'
            )
            return True
        except client.exceptions.ConditionalCheckFailedException:
            logger.warning(f"Miner stats not found for {hotkey[:8]}..., cannot update slots")
            return False
        except Exception as e:
            logger.error(f"Failed to update sampling slots for {hotkey[:8]}...: {e}")
            return False