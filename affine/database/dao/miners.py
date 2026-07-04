"""
Miners DAO

Manages miner validation state and anti-plagiarism tracking.
"""

from typing import Dict, Any, List, Optional
from affine.database.base_dao import BaseDAO
from affine.database.schema import get_table_name


def _int_value(row: Dict[str, Any], key: str, default: int = -1) -> int:
    try:
        return int(row.get(key))
    except (TypeError, ValueError):
        return default


def select_preferred_hotkey_row(
    rows: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Pick the current miner row when stale duplicate hotkey rows exist.

    The miners table is keyed by uid and refreshed from the metagraph. If a
    hotkey deregisters and later returns on a different uid, an old uid row can
    survive until the monitor overwrites that uid. Prefer the freshest
    ``block_number`` first so callers resolve the current on-chain placement
    without requiring manual DB cleanup. Validity is only a tie-breaker:
    a newer invalid row should not be hidden behind an older valid stale row.
    """
    if not rows:
        return None
    return sorted(
        rows,
        key=lambda row: (
            _int_value(row, "block_number"),
            str(row.get("is_valid") or "").lower() == "true",
            _int_value(row, "first_block"),
            -_int_value(row, "uid", default=10**9),
        ),
        reverse=True,
    )[0]


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
        model_hash: str,
        is_valid: bool,
        invalid_reason: Optional[str],
        block_number: int,
        first_block: int,
        model_type: str = "",
        reg_block: int = 0,
        reg_hotkey: str = "",
    ) -> Dict[str, Any]:
        """Save or update miner validation state.

        Directly updates the record for this UID (no history tracking).
        Inference is provider-routed by the scheduler service per window.

        Args:
            uid: Miner UID (0-255 regular, > 1000 system).
            hotkey: SS58 hotkey (or virtual hotkey for system miners).
            model: HuggingFace model repo.
            revision: Git commit hash.
            model_hash: SHA256 hash of all model weights (for plagiarism check).
            is_valid: Overall validation result.
            invalid_reason: Reason if invalid.
            block_number: Current block when this record was updated.
            first_block: Block when this (hotkey, revision) was first committed.
            model_type: Raw HuggingFace config ``model_type`` when known.
        """
        if not (model and model_hash and model_type and first_block):
            existing = await self.get_miner_by_uid(uid)
            if (
                existing
                and existing.get("hotkey") == hotkey
                and existing.get("revision") == revision
            ):
                if not model and existing.get("model"):
                    model = str(existing["model"])
                if not model_hash and existing.get("model_hash"):
                    model_hash = str(existing["model_hash"])
                if not model_type and existing.get("model_type"):
                    model_type = str(existing["model_type"])
                if not first_block and existing.get("first_block") is not None:
                    try:
                        first_block = int(existing["first_block"])
                    except (TypeError, ValueError):
                        pass

        item = {
            'pk': self._make_pk(uid),
            'uid': uid,
            'hotkey': hotkey,
            'model': model,
            'revision': revision,
            'model_type': model_type,
            'model_hash': model_hash,
            'is_valid': 'true' if is_valid else 'false',  # Stored as string for GSI
            'invalid_reason': invalid_reason,
            'block_number': block_number,
            'first_block': first_block,
            'reg_block': reg_block,
            'reg_hotkey': reg_hotkey,
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
        }
        
        response = await client.query(**params)
        items = [self._deserialize(item) for item in response.get('Items', [])]

        return select_preferred_hotkey_row(items)
    
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
