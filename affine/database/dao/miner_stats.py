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
    
    # A miner that has been cold for this many cumulative seconds (across
    # all causes — chute torn down, fetch failed, deployment unhealthy)
    # is auto-terminated. 48h = 172800s.
    COLD_TERMINATION_THRESHOLD_SECONDS = 48 * 3600

    # Upper bound on the per-call cold-time delta. Refresh cadence is ~5
    # minutes, so a single missed cycle adds ~10min; if the monitor is
    # down longer than this cap, we don't credit the gap as cold time.
    COLD_TRACKING_DELTA_CAP_SECONDS = 600

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
                'sampling_slots': 20,  # Default = MIN_SLOTS (slots_adjuster)
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
                'env_stats': env_stats,
                'sampling_slots': 20,  # Default = MIN_SLOTS (slots_adjuster)
                'slots_last_adjusted_at': 0  # Never adjusted
            }
            await self.put(updated_item)
            return  # Exit early, no need for second update
        except client.exceptions.ClientError as e:
            # Heal records that were created by update_challenge_state
            # before the if_not_exists initializers were added. Those
            # rows have challenge_* fields but no env_stats map, so the
            # nested `SET env_stats.GAME = ...` above fails with
            # ValidationException. Bootstrap the missing parent maps
            # with a single atomic update_item (no read, no race with
            # concurrent scorer writes), then retry the original SET.
            if e.response.get('Error', {}).get('Code') != 'ValidationException':
                raise
            logger.warning(
                f"Initializing miner_stats parent maps for "
                f"{hotkey[:8]}...#{revision[:7]} and retrying"
            )
            await client.update_item(
                TableName=self.table_name,
                Key={'pk': {'S': pk}, 'sk': {'S': sk}},
                UpdateExpression=(
                    'SET env_stats = if_not_exists(env_stats, :empty_map), '
                    'sampling_stats = if_not_exists(sampling_stats, :empty_map), '
                    'sampling_slots = if_not_exists(sampling_slots, :default_slots)'
                ),
                ExpressionAttributeValues={
                    ':empty_map': {'M': {}},
                    ':default_slots': {'N': '20'},
                },
            )
            response = await client.update_item(
                TableName=self.table_name,
                Key={'pk': {'S': pk}, 'sk': {'S': sk}},
                UpdateExpression=update_expr,
                ExpressionAttributeNames=expr_names,
                ExpressionAttributeValues=expr_values,
                ConditionExpression='attribute_exists(pk)',
                ReturnValues='ALL_NEW',
            )
            updated_item = self._deserialize(response['Attributes'])
        
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

    # Fields that make up a miner's challenge state (loss/win counters etc.)
    _CHALLENGE_FIELDS = (
        'challenge_consecutive_wins',
        'challenge_total_losses',
        'challenge_consecutive_losses',
        'challenge_checkpoints_passed',
        'challenge_status',
        'termination_reason',
    )

    @staticmethod
    def _challenge_defaults() -> Dict[str, Any]:
        return {
            'challenge_consecutive_wins': 0,
            'challenge_total_losses': 0,
            'challenge_consecutive_losses': 0,
            'challenge_checkpoints_passed': 0,
            'challenge_status': 'sampling',
            'termination_reason': '',
        }

    @classmethod
    def _extract_challenge_state(cls, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Pick only the challenge_* fields out of a stats row."""
        defaults = cls._challenge_defaults()
        return {k: stats.get(k, defaults[k]) for k in cls._CHALLENGE_FIELDS}

    @staticmethod
    def _has_challenge_state(stats: Dict[str, Any]) -> bool:
        """True if the stats row actually has any challenge_* counter set
        (not just defaults from a freshly-initialised sampling record).
        We ignore the default 'sampling' status + zero counters case, since
        that means the scorer has never written to this row yet."""
        if not stats:
            return False
        for k in ('challenge_total_losses', 'challenge_consecutive_losses',
                  'challenge_consecutive_wins', 'challenge_checkpoints_passed'):
            if stats.get(k, 0):
                return True
        if stats.get('challenge_status') == 'terminated':
            return True
        return False

    async def get_challenge_state(
        self,
        hotkey: str,
        revision: str
    ) -> Dict[str, Any]:
        """Get challenge state for a miner.

        Identity is hotkey: a miner that redeploys (revision change) or
        bounces cold/hot should not lose accumulated losses, wins, CP, or
        terminated status. Priority order:

        1. Direct row at (hotkey, revision) with real challenge state → use it.
        2. Fallback: most recently-updated row for this hotkey that carries
           real challenge state → inherit it (losses persist across
           revisions, terminated stays terminated forever).
        3. Nothing found → fresh defaults (brand-new hotkey).
        """
        direct = await self.get_miner_stats(hotkey, revision)
        if self._has_challenge_state(direct):
            return self._extract_challenge_state(direct)

        # Fallback: scan all revisions for this hotkey and pick the freshest
        # row that actually carries challenge state. DynamoDB PK query, not
        # a table scan, so this is an O(revisions-per-hotkey) operation.
        try:
            all_rows = await self.query(pk=self._make_pk(hotkey))
        except Exception as e:
            logger.warning(
                f"get_challenge_state fallback query failed for {hotkey[:8]}...: {e}"
            )
            all_rows = []

        candidates = [r for r in all_rows if self._has_challenge_state(r)]
        if candidates:
            latest = max(candidates, key=lambda r: r.get('last_updated_at', 0))
            logger.debug(
                f"get_challenge_state inherited state for {hotkey[:8]}... "
                f"(target rev={revision[:8]}..., source rev={latest.get('revision','?')[:8]}..., "
                f"losses={latest.get('challenge_total_losses', 0)}, "
                f"status={latest.get('challenge_status', 'sampling')})"
            )
            return self._extract_challenge_state(latest)

        # Direct row exists but has no challenge state (e.g., freshly created
        # by update_sampling_stats). Return its fields (all zeros/sampling).
        if direct:
            return self._extract_challenge_state(direct)
        return self._challenge_defaults()

    async def update_challenge_state(
        self,
        hotkey: str,
        revision: str,
        consecutive_wins: int,
        total_losses: int,
        consecutive_losses: int,
        checkpoints_passed: int,
        status: str,
        termination_reason: str = '',
    ) -> None:
        """Update miner's champion challenge state.

        Uses an upsert (update_item with no condition) — when a (hotkey,
        revision) pair has not been seen before, this is the very first
        write, so the record needs to be created.

        Initialize the rest of the schema with `if_not_exists` so a brand
        new record is created complete: empty env_stats / sampling_stats
        maps, default sampling_slots, etc. Without these initializers,
        the partial record breaks update_sampling_stats — its nested
        path update `SET env_stats.GAME = ...` raises
        ValidationException when env_stats does not exist, so per-window
        sampling stats never land in the row even though the underlying
        sample_results are written.
        """
        from affine.database.client import get_client
        client = get_client()

        pk = self._make_pk(hotkey)
        sk = self._make_sk(revision)

        now = int(time.time())

        await client.update_item(
            TableName=self.table_name,
            Key={'pk': {'S': pk}, 'sk': {'S': sk}},
            UpdateExpression=(
                'SET challenge_consecutive_wins = :cw, '
                'challenge_total_losses = :tl, '
                'challenge_consecutive_losses = :cl, '
                'challenge_checkpoints_passed = :cp, '
                'challenge_status = :cs, '
                'termination_reason = :tr, '
                'last_updated_at = :ts, '
                'hotkey = if_not_exists(hotkey, :hk), '
                'revision = if_not_exists(revision, :rev), '
                'model = if_not_exists(model, :empty_str), '
                'first_seen_at = if_not_exists(first_seen_at, :ts), '
                'best_rank = if_not_exists(best_rank, :default_rank), '
                'best_weight = if_not_exists(best_weight, :zero_f), '
                'is_currently_online = if_not_exists(is_currently_online, :true_b), '
                'env_stats = if_not_exists(env_stats, :empty_map), '
                'sampling_stats = if_not_exists(sampling_stats, :empty_map), '
                'sampling_slots = if_not_exists(sampling_slots, :default_slots), '
                'slots_last_adjusted_at = if_not_exists(slots_last_adjusted_at, :zero)'
            ),
            ExpressionAttributeValues={
                ':cw': {'N': str(consecutive_wins)},
                ':tl': {'N': str(total_losses)},
                ':cl': {'N': str(consecutive_losses)},
                ':cp': {'N': str(checkpoints_passed)},
                ':cs': {'S': status},
                ':tr': {'S': termination_reason},
                ':ts': {'N': str(now)},
                ':hk': {'S': hotkey},
                ':rev': {'S': revision},
                ':empty_str': {'S': ''},
                ':default_rank': {'N': '999'},
                ':zero_f': {'N': '0.0'},
                ':true_b': {'BOOL': True},
                ':empty_map': {'M': {}},
                ':default_slots': {'N': '20'},
                ':zero': {'N': '0'},
            },
        )

    async def update_cold_tracking(
        self,
        hotkey: str,
        revision: str,
        is_cold: bool,
        backfill_last_seen_ms: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Track cumulative cold time and auto-terminate at the threshold.

        Counted only after the miner has been observed sampling at least
        once (`has_been_hot`). On the first call for a record that lacks
        any prior tracking field, the caller may pass the most recent
        sample timestamp via ``backfill_last_seen_ms`` so accumulated
        cold time on long-cold miners is credited from that point.

        Args:
            hotkey: miner hotkey
            revision: model revision
            is_cold: True if the miner is currently not actively sampling
                (any cause: chute cold, fetch failed, deployment failed, …)
            backfill_last_seen_ms: epoch milliseconds of most recent
                sample. Used only when the record has no prior
                ``last_status_check_at``. Implies the miner has been hot
                in the past.

        Returns:
            Subset of fields actually written, or None if the record
            does not exist (never sampled — skip).
        """
        from affine.database.client import get_client
        client = get_client()

        pk = self._make_pk(hotkey)
        sk = self._make_sk(revision)

        existing = await self.get(pk, sk)
        if not existing:
            return None

        if existing.get('challenge_status') == 'terminated':
            return None

        now = int(time.time())
        cold_total = int(existing.get('cold_seconds_total', 0) or 0)
        has_been_hot = bool(existing.get('has_been_hot', False))
        last_check = existing.get('last_status_check_at')

        if last_check is None:
            if not is_cold:
                has_been_hot = True
                delta = 0
            elif backfill_last_seen_ms:
                has_been_hot = True
                delta = max(0, now - int(backfill_last_seen_ms) // 1000)
            else:
                delta = 0
        else:
            try:
                last_check_int = int(last_check)
            except (ValueError, TypeError):
                last_check_int = now
            delta = max(0, min(now - last_check_int, self.COLD_TRACKING_DELTA_CAP_SECONDS))
            if not is_cold:
                has_been_hot = True

        if is_cold and has_been_hot:
            cold_total += delta

        new_status = existing.get('challenge_status', 'sampling')
        new_reason = existing.get('termination_reason', '') or ''
        terminate_now = (
            cold_total >= self.COLD_TERMINATION_THRESHOLD_SECONDS
            and new_status != 'terminated'
        )
        if terminate_now:
            new_status = 'terminated'
            new_reason = 'cold_too_long'
            logger.info(
                f"[ColdTermination] Terminating miner hk={hotkey[:8]}... "
                f"rev={revision[:8]}... cold_seconds_total={cold_total} "
                f"(>= {self.COLD_TERMINATION_THRESHOLD_SECONDS})"
            )

        try:
            await client.update_item(
                TableName=self.table_name,
                Key={'pk': {'S': pk}, 'sk': {'S': sk}},
                UpdateExpression=(
                    'SET cold_seconds_total = :ct, '
                    'has_been_hot = :hbh, '
                    'last_status_check_at = :now, '
                    'last_updated_at = :now, '
                    'challenge_status = :cs, '
                    'termination_reason = :tr'
                ),
                ExpressionAttributeValues={
                    ':ct': {'N': str(cold_total)},
                    ':hbh': {'BOOL': has_been_hot},
                    ':now': {'N': str(now)},
                    ':cs': {'S': new_status},
                    ':tr': {'S': new_reason},
                },
                ConditionExpression='attribute_exists(pk)',
            )
        except client.exceptions.ConditionalCheckFailedException:
            return None

        return {
            'cold_seconds_total': cold_total,
            'has_been_hot': has_been_hot,
            'last_status_check_at': now,
            'challenge_status': new_status,
            'termination_reason': new_reason,
        }

