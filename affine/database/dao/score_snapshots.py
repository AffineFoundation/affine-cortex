"""
Score Snapshots DAO

Manages metadata for each scoring calculation.
"""

import time
from typing import Dict, Any, List, Optional
from affine.database.base_dao import BaseDAO
from affine.database.schema import get_table_name
from affine.database.client import get_client


class ScoreSnapshotsDAO(BaseDAO):
    """DAO for score_snapshots table.
    
    Stores metadata for each scoring calculation.
    
    PK: BLOCK#{block_number}
    SK: TIME#{timestamp}
    """
    
    def __init__(self):
        self.table_name = get_table_name("score_snapshots")
        super().__init__()
    
    def _make_pk(self, block_number: int) -> str:
        """Generate partition key."""
        return f"BLOCK#{block_number}"
    
    def _make_sk(self, timestamp: int) -> str:
        """Generate sort key."""
        return f"TIME#{timestamp}"
    
    async def save_snapshot(
        self,
        block_number: int,
        scorer_hotkey: str,
        config: Dict[str, Any],
        statistics: Dict[str, Any],
        ttl_days: int = 365,
    ) -> Dict[str, Any]:
        """Save a scoring snapshot metadata.
        
        Args:
            block_number: Block number for this snapshot
            scorer_hotkey: Hotkey of the scheduler service
            config: Configuration used for this scoring
                {
                    "environments": [env1, env2, ...],
                    "error_rate_reduction": float,
                    "burn_percentage": float
                }
            statistics: Statistics about this scoring run
                {
                    "total_miners": int,
                    "valid_miners": int,
                    "invalid_miners": int,
                    "final_weights": {"uid": "weight", ...}
                }
            ttl_days: Days until automatic deletion (default 365). The
                weights endpoint walks recent snapshots to find the past
                N distinct champions, so we keep at least a year of
                history to survive a long-tenured champion's win-streak.
            
        Returns:
            Saved item
        """
        timestamp = int(time.time())
        
        item = {
            'pk': self._make_pk(block_number),
            'sk': self._make_sk(timestamp),
            'block_number': block_number,
            'calculated_at': timestamp,
            'scorer_hotkey': scorer_hotkey,
            'config': config,
            'statistics': statistics,
            'timestamp': timestamp,
            'latest_marker': 'LATEST',  # For GSI queries
            'ttl': self.get_ttl(ttl_days),
        }
        
        return await self.put(item)
    
    async def get_snapshot_at_block(
        self,
        block_number: int
    ) -> Optional[Dict[str, Any]]:
        """Get the snapshot for a specific block.
        
        Returns the most recent snapshot for this block if multiple exist.
        
        Args:
            block_number: Block number
            
        Returns:
            Snapshot metadata if found, None otherwise
        """
        pk = self._make_pk(block_number)
        results = await self.query(pk=pk, limit=1, reverse=True)
        
        if not results:
            return None
        
        return results[0]
    
    async def get_latest_snapshot(self) -> Optional[Dict[str, Any]]:
        """Get the most recent snapshot.
        
        Uses latest-index GSI to find the latest snapshot.
        
        Returns:
            Latest snapshot metadata, or None if no snapshots exist
        """
        client = get_client()
        
        params = {
            'TableName': self.table_name,
            'IndexName': 'latest-index',
            'KeyConditionExpression': 'latest_marker = :marker',
            'ExpressionAttributeValues': {
                ':marker': {'S': 'LATEST'}
            },
            'ScanIndexForward': False,  # Descending order by timestamp
            'Limit': 1
        }
        
        response = await client.query(**params)
        items = response.get('Items', [])
        
        if not items:
            return None
        
        return self._deserialize(items[0])
    
    async def get_recent_snapshots(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent snapshots.
        
        Args:
            limit: Maximum number of snapshots to return
            
        Returns:
            List of recent snapshots (sorted by timestamp desc)
        """
        client = get_client()
        
        params = {
            'TableName': self.table_name,
            'IndexName': 'latest-index',
            'KeyConditionExpression': 'latest_marker = :marker',
            'ExpressionAttributeValues': {
                ':marker': {'S': 'LATEST'}
            },
            'ScanIndexForward': False,  # Descending
            'Limit': limit
        }
        
        response = await client.query(**params)
        items = [self._deserialize(item) for item in response.get('Items', [])]
        
        return items
    
    async def delete_snapshot(
        self,
        block_number: int,
        timestamp: int
    ) -> None:
        """Delete a specific snapshot.
        
        Args:
            block_number: Block number
            timestamp: Timestamp
        """
        pk = self._make_pk(block_number)
        sk = self._make_sk(timestamp)
        await self.delete(pk, sk)
    
    async def delete_snapshots_at_block(self, block_number: int) -> int:
        """Delete all snapshots for a specific block.
        
        Args:
            block_number: Block number
            
        Returns:
            Number of snapshots deleted
        """
        pk = self._make_pk(block_number)
        snapshots = await self.query(pk=pk)
        
        deleted_count = 0
        for snapshot in snapshots:
            await self.delete(snapshot['pk'], snapshot['sk'])
            deleted_count += 1
        
        return deleted_count