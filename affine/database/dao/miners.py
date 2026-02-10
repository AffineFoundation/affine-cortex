"""
Miners DAO

Manages miner validation state and anti-plagiarism tracking.
"""

import time
import logging
from typing import Dict, Any, List, Optional
from affine.database.base_dao import BaseDAO
from affine.database.schema import get_table_name


from affine.core.setup import logger


class MinersDAO(BaseDAO):
    """DAO for miners table.
    
    Schema Design:
    - PK: UID#{uid} - unique primary key, each UID has only one record
    - No SK needed - single record per UID
    - GSI1: is-valid-index for querying valid/invalid miners
    """
    
    def __init__(self):
        self.table_name = get_table_name("miners")
        super().__init__()
    
    def _make_pk(self, uid: int) -> str:
        """Generate partition key based on UID."""
        return f"UID#{uid}"
    
    async def save_miner(
        self,
        uid: int,
        hotkey: str,
        model: str,
        revision: str,
        chute_id: str,
        chute_slug: str,
        model_hash: str,
        chute_status: str,
        is_valid: bool,
        invalid_reason: Optional[str],
        block_number: int,
        first_block: int,
        template_check_result: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Save or update miner validation state.

        Directly updates the record for this UID (no history tracking).

        Args:
            uid: Miner UID (0-255 for regular miners, > 1000 for system miners)
            hotkey: Miner's SS58 hotkey (or virtual hotkey like "SYSTEM-1" for system miners)
            model: HuggingFace model repo
            revision: Git commit hash
            chute_id: Chutes deployment ID
            chute_slug: Chutes URL slug
            model_hash: SHA256 hash of all model weights
            chute_status: "hot" or "cold"
            is_valid: Overall validation result (boolean)
            invalid_reason: Reason if invalid (null if valid)
            block_number: Current block when this record was updated
            first_block: Block when miner first committed
            template_check_result: Template check result ("safe", "unsafe:reason", or null)

        Returns:
            Saved miner record
        """
        item = {
            'pk': self._make_pk(uid),
            'uid': uid,
            'hotkey': hotkey,
            'model': model,
            'revision': revision,
            'chute_id': chute_id,
            'chute_slug': chute_slug,
            'model_hash': model_hash,
            'chute_status': chute_status,
            'is_valid': 'true' if is_valid else 'false',  # Store as string for GSI
            'invalid_reason': invalid_reason,
            'block_number': block_number,
            'first_block': first_block,
            'template_check_result': template_check_result,
        }

        return await self.put(item)
    
    async def get_miner_by_uid(
        self,
        uid: int,
    ) -> Optional[Dict[str, Any]]:
        """Get miner by UID.
        
        Args:
            uid: Miner UID
            
        Returns:
            Miner record or None if not found
        """
        pk = self._make_pk(uid)
        return await self.get(pk)
    
    async def get_miner_by_hotkey(
        self,
        hotkey: str,
    ) -> Optional[Dict[str, Any]]:
        """Get miner by hotkey.
        
        Args:
            hotkey: Miner's SS58 hotkey
            
        Returns:
            Miner record or None if not found
        """
        from affine.database.client import get_client
        client = get_client()
        
        params = {
            'TableName': self.table_name,
            'IndexName': 'hotkey-index',
            'KeyConditionExpression': 'hotkey = :hotkey',
            'ExpressionAttributeValues': {':hotkey': {'S': hotkey}},
            'Limit': 1
        }
        
        response = await client.query(**params)
        items = [self._deserialize(item) for item in response.get('Items', [])]
        
        return items[0] if items else None
    
    async def get_valid_miners(self) -> List[Dict[str, Any]]:
        """Get all valid miners using GSI.
        
        Returns:
            List of valid miner records
        """
        from affine.database.client import get_client
        client = get_client()
        
        params = {
            'TableName': self.table_name,
            'IndexName': 'is-valid-index',
            'KeyConditionExpression': 'is_valid = :is_valid',
            'ExpressionAttributeValues': {':is_valid': {'S': 'true'}},
        }
        
        response = await client.query(**params)
        items = [self._deserialize(item) for item in response.get('Items', [])]
        
        return items
    
    async def get_invalid_miners(self) -> List[Dict[str, Any]]:
        """Get all invalid miners using GSI.
        
        Returns:
            List of invalid miner records
        """
        from affine.database.client import get_client
        client = get_client()
        
        params = {
            'TableName': self.table_name,
            'IndexName': 'is-valid-index',
            'KeyConditionExpression': 'is_valid = :is_valid',
            'ExpressionAttributeValues': {':is_valid': {'S': 'false'}},
        }
        
        response = await client.query(**params)
        items = [self._deserialize(item) for item in response.get('Items', [])]
        
        return items
    
    async def get_miners_by_model_hash(
        self,
        model_hash: str
    ) -> List[Dict[str, Any]]:
        """Get all miners with a specific model hash.
        
        Used for anti-plagiarism detection.
        
        Args:
            model_hash: Model weights SHA256 hash
            
        Returns:
            List of miners with this hash, sorted by first_block (earliest first)
        """
        from affine.database.client import get_client
        client = get_client()
        
        # Scan table for matching model_hash
        params = {
            'TableName': self.table_name,
            'FilterExpression': 'model_hash = :hash',
            'ExpressionAttributeValues': {':hash': {'S': model_hash}}
        }
        
        response = await client.scan(**params)
        items = [self._deserialize(item) for item in response.get('Items', [])]
        
        # Sort by first_block (earliest miner first)
        result = sorted(items, key=lambda x: x.get('first_block', float('inf')))
        
        return result
    
    async def get_all_miners(self) -> List[Dict[str, Any]]:
        """Get all miners (full table scan).
        
        Efficient for small tables (256 miners max).
        Returns all miners regardless of validation status.
        
        Returns:
            List of all miner records
        """
        from affine.database.client import get_client
        client = get_client()
        
        params = {
            'TableName': self.table_name,
        }
        
        response = await client.scan(**params)
        items = [self._deserialize(item) for item in response.get('Items', [])]
        
        return items