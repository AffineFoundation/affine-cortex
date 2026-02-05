"""
Scores DAO

Manages score snapshots organized by block number.
"""

import time
from typing import Dict, Any, List, Optional
from affine.database.base_dao import BaseDAO
from affine.database.schema import get_table_name
from affine.database.client import get_client


class ScoresDAO(BaseDAO):
    """DAO for scores table.
    
    Stores score snapshots per block with 30-day TTL.
    PK: SCORE#{block_number}
    SK: MINER#{hotkey}
    """
    
    def __init__(self):
        self.table_name = get_table_name("scores")
        super().__init__()
    
    def _make_pk(self, block_number: int) -> str:
        """Generate partition key."""
        return f"SCORE#{block_number}"
    
    def _make_sk(self, miner_hotkey: str) -> str:
        """Generate sort key."""
        return f"MINER#{miner_hotkey}"
    
    async def save_score(
        self,
        block_number: int,
        miner_hotkey: str,
        uid: int,
        model_revision: str,
        model: str,
        first_block: int,
        overall_score: float,
        average_score: float,
        scores_by_layer: Dict[str, float],
        scores_by_env: Dict[str, Any],
        total_samples: int,
        subset_contributions: Optional[Dict[str, Dict[str, Any]]] = None,
        cumulative_weight: Optional[float] = None,
        filter_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Save a score snapshot for a miner at a specific block.
        
        Merged from miner_scores table - now contains both summary and detailed data.
        
        Args:
            block_number: Current block number
            miner_hotkey: Miner's hotkey
            uid: Miner UID
            model_revision: Model revision
            model: Model repository identifier
            first_block: Block number when miner first registered
            overall_score: Overall score (normalized weight)
            average_score: Average score across environments
            scores_by_layer: Scores breakdown by layer
            scores_by_env: Detailed scores by environment (with score, sample_count, completeness)
            total_samples: Total number of samples
            subset_contributions: Detailed subset contributions (optional, from miner_scores)
            cumulative_weight: Cumulative weight before normalization (optional, from miner_scores)
            filter_info: Filtering information (optional, from miner_scores)
            
        Returns:
            Saved score item
        """
        calculated_at = int(time.time())
        
        item = {
            'pk': self._make_pk(block_number),
            'sk': self._make_sk(miner_hotkey),
            'block_number': block_number,
            'miner_hotkey': miner_hotkey,
            'uid': uid,
            'model_revision': model_revision,
            'model': model,
            'first_block': first_block,
            'calculated_at': calculated_at,
            'overall_score': overall_score,
            'average_score': average_score,
            'scores_by_layer': scores_by_layer,
            'scores_by_env': scores_by_env,
            'total_samples': total_samples,
            'latest_marker': 'LATEST',  # For GSI
        }
        
        # Add optional detailed fields (from miner_scores)
        if subset_contributions is not None:
            item['subset_contributions'] = subset_contributions
        
        if cumulative_weight is not None:
            item['cumulative_weight'] = cumulative_weight
        
        if filter_info is not None:
            item['filter_info'] = filter_info
        
        # Conditional TTL: only set TTL for miners with zero weight
        if overall_score == 0:
            item['ttl'] = self.get_ttl(7)  # 30 days for inactive miners
        # Miners with non-zero weight: no TTL (permanent storage)
        
        return await self.put(item)
    
    async def get_scores_at_block(
        self,
        block_number: int,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get all miner scores at a specific block.
        
        Args:
            block_number: Block number
            limit: Maximum number of scores to return
            
        Returns:
            List of score entries
        """
        pk = self._make_pk(block_number)
        
        return await self.query(pk=pk, limit=limit)
    
    async def get_latest_scores(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """Get the most recent score snapshot.
        
        Uses latest-block-index GSI to find the most recent block.
        
        Args:
            limit: Maximum number of scores to return
            
        Returns:
            Dictionary with block_number and scores list
        """
        client = get_client()
        
        # Query GSI to get latest block
        params = {
            'TableName': self.table_name,
            'IndexName': 'latest-block-index',
            'KeyConditionExpression': 'latest_marker = :marker',
            'ExpressionAttributeValues': {
                ':marker': {'S': 'LATEST'}
            },
            'ScanIndexForward': False,  # Descending order
            'Limit': 1
        }
        
        response = await client.query(**params)
        items = response.get('Items', [])
        
        if not items:
            return {'block_number': None, 'scores': []}
        
        # Get the latest block number
        latest_item = self._deserialize(items[0])
        latest_block = latest_item['block_number']
        
        # Get all scores for this block
        scores = await self.get_scores_at_block(latest_block, limit=limit)
        
        return {
            'block_number': latest_block,
            'calculated_at': latest_item['calculated_at'],
            'scores': scores
        }

    
    async def save_weight_snapshot(
        self,
        block_number: int,
        weights: Dict[str, float],
        calculation_details: Dict[str, Any],
        scorer_hotkey: str = "scorer_service"
    ) -> Dict[str, Any]:
        """Save a complete weight snapshot for all miners.
        
        This is a convenience method that saves weights for all miners at once.
        
        Args:
            block_number: Current block number
            weights: Dict mapping hotkey -> weight (0.0 to 1.0)
            calculation_details: Details about the calculation (method, params, etc.)
            scorer_hotkey: Service identifier
            
        Returns:
            Summary of saved snapshot
        """
        import uuid
        snapshot_id = str(uuid.uuid4())
        created_at = int(time.time())
        
        # Save each miner's weight
        saved_count = 0
        for hotkey, weight in weights.items():
            # Get additional info from calculation_details if available
            miner_details = calculation_details.get('miners', {}).get(hotkey, {})
            
            item = {
                'pk': self._make_pk(block_number),
                'sk': self._make_sk(hotkey),
                'block_number': block_number,
                'miner_hotkey': hotkey,
                'uid': miner_details.get('uid', -1),
                'model_revision': miner_details.get('model_revision', ''),
                'calculated_at': created_at,
                'overall_score': weight,
                'average_score': miner_details.get('average_score', weight),
                'scores_by_layer': miner_details.get('scores_by_layer', {}),
                'scores_by_env': miner_details.get('scores_by_env', {}),
                'total_samples': miner_details.get('total_samples', 0),
                'latest_marker': 'LATEST',
                'ttl': self.get_ttl(30),
                'snapshot_id': snapshot_id,
            }
            
            await self.put(item)
            saved_count += 1
        
        return {
            'snapshot_id': snapshot_id,
            'block_number': block_number,
            'created_at': created_at,
            'miners_count': saved_count,
            'calculation_details': calculation_details
        }
    
    async def get_weights_for_setting(self) -> Dict[str, Any]:
        """Get the latest weights in a format suitable for chain setting.

        Includes both regular miners (uid >= 0) and system miners (uid < 0).
        The validator's process_weights() method will handle allocating
        system miner weights to UID 0.

        Returns:
            Dict with:
                - block_number: Block at which weights were calculated
                - weights: Dict mapping hotkey -> weight
                - uids: Dict mapping uid -> weight (includes negative UIDs for system miners)
        """
        latest = await self.get_latest_scores()

        if not latest['block_number']:
            return {
                'block_number': None,
                'weights': {},
                'uids': {}
            }

        weights_by_hotkey = {}
        weights_by_uid = {}

        for score in latest['scores']:
            hotkey = score.get('miner_hotkey')
            uid = score.get('uid')
            weight = score.get('overall_score', 0.0)

            if hotkey:
                weights_by_hotkey[hotkey] = weight

            # Include all UIDs (both positive and negative)
            # Negative UIDs are system miners whose weights will be
            # allocated to UID 0 by the validator
            if uid is not None:
                weights_by_uid[uid] = weight

        return {
            'block_number': latest['block_number'],
            'calculated_at': latest.get('calculated_at'),
            'weights': weights_by_hotkey,
            'uids': weights_by_uid
        }