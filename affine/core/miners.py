"""
Miner Information Query Module

Provides functionality to query miner information from the Bittensor blockchain.
This module is optimized for SDK usage and returns basic miner info from blockchain commits.
"""

import os
import json
import asyncio
from typing import Dict, List, Optional, Union, Any
from affine.core.models import Miner
from affine.core.setup import NETUID, logger
from affine.utils.subtensor import get_subtensor
from affine.utils.api_client import get_chute_info
from affine.utils.errors import NetworkError, ValidationError


class MinerQueryError(Exception):
    """Raised when miner query operations fail."""
    pass


async def miners(
    uids: Optional[Union[int, List[int]]] = None,
    netuid: int = NETUID,
    meta: Optional[Any] = None,
) -> Dict[int, Miner]:
    """
    Query miner information from blockchain.
    
    Simplified version for miner SDK usage - returns basic miner info from blockchain commits.
    For validator use cases with filtering logic, refer to affine.src.monitor.miners_monitor.
    
    This function:
    1. Fetches metagraph and commitment data from the blockchain
    2. Parses commit data to extract model, revision, and chute_id
    3. Validates chute information from Chutes API
    4. Returns only miners with valid commits and active chutes
    
    Args:
        uids: Miner UID(s) to query. If None, queries all UIDs in the metagraph.
            Can be a single int or a list of ints.
        netuid: Network UID (default: from NETUID config)
        meta: Optional metagraph object (will be fetched if not provided)
        
    Returns:
        Dict mapping UID to Miner info. Only includes miners with:
        - Valid blockchain commits
        - Valid model, revision, and chute_id in commit data
        - Active chute deployments (hot=True)
        
    Raises:
        MinerQueryError: If blockchain query fails or metagraph is invalid
        NetworkError: If network request to Chutes API fails
        
    Example:
        >>> miners_dict = await miners(7)
        >>> if 7 in miners_dict:
        >>>     print(miners_dict[7].model)
        
        >>> # Query multiple miners
        >>> miners_dict = await miners([1, 2, 3])
        >>> print(f"Found {len(miners_dict)} valid miners")
    """
    try:
        sub = await get_subtensor()
        meta = meta or await sub.metagraph(netuid)
        commits = await sub.get_all_revealed_commitments(netuid)
    except Exception as e:
        logger.error(f"Failed to fetch blockchain data: {e}")
        raise MinerQueryError(f"Failed to fetch blockchain data: {e}") from e
    
    if uids is None:
        uids = list(range(len(meta.hotkeys)))
    elif isinstance(uids, int):
        uids = [uids]
    
    # Validate UIDs are within metagraph bounds
    max_uid = len(meta.hotkeys) - 1
    invalid_uids = [uid for uid in uids if uid < 0 or uid > max_uid]
    if invalid_uids:
        logger.warning(f"Invalid UIDs requested: {invalid_uids}. Max UID: {max_uid}")
        uids = [uid for uid in uids if 0 <= uid <= max_uid]
    
    if not uids:
        logger.warning("No valid UIDs to query")
        return {}
    
    meta_sem = asyncio.Semaphore(int(os.getenv("AFFINE_META_CONCURRENCY", "12")))

    async def _fetch_miner(uid: int) -> Optional[Miner]:
        """
        Fetch miner information for a single UID.
        
        Args:
            uid: Miner UID to fetch
            
        Returns:
            Miner object if valid, None otherwise
        """
        try:
            if uid >= len(meta.hotkeys):
                logger.debug(f"UID {uid} out of bounds (metagraph size: {len(meta.hotkeys)})")
                return None
                
            hotkey = meta.hotkeys[uid]
            if hotkey not in commits:
                logger.debug(f"No commit found for UID {uid}")
                return None

            block, commit_data = commits[hotkey][-1]
            block = 0 if uid == 0 else block
            
            try:
                data = json.loads(commit_data)
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON in commit data for UID {uid}: {e}")
                return None
            
            model = data.get("model")
            miner_revision = data.get("revision")
            chute_id = data.get("chute_id")

            if not model or not miner_revision or not chute_id:
                logger.debug(
                    f"Incomplete commit data for UID {uid}: "
                    f"model={bool(model)}, revision={bool(miner_revision)}, chute_id={bool(chute_id)}"
                )
                return None

            # Fetch chute info with semaphore to limit concurrency
            try:
                async with meta_sem:
                    chute = await get_chute_info(chute_id)
            except NetworkError as e:
                logger.debug(f"Network error fetching chute info for UID {uid}: {e}")
                return None
            except Exception as e:
                logger.debug(f"Error fetching chute info for UID {uid}: {e}")
                return None

            if not chute or not chute.get("hot", False):
                logger.debug(f"Chute not active for UID {uid} (chute_id: {chute_id})")
                return None

            return Miner(
                uid=uid,
                hotkey=hotkey,
                model=model,
                block=int(block),
                revision=miner_revision,
                slug=chute.get("slug"),
                chute=chute,
            )
        except (KeyError, IndexError, ValueError) as e:
            logger.debug(f"Data parsing error for miner uid={uid}: {e}")
            return None
        except Exception as e:
            logger.trace(f"Unexpected error fetching miner uid={uid}: {e}", exc_info=True)
            return None

    try:
        results = await asyncio.gather(*(_fetch_miner(uid) for uid in uids), return_exceptions=True)
    except Exception as e:
        logger.error(f"Error during parallel miner fetch: {e}")
        raise MinerQueryError(f"Error during parallel miner fetch: {e}") from e
    
    # Filter out None values and exceptions
    output: Dict[int, Miner] = {}
    for uid, result in zip(uids, results):
        if isinstance(result, Exception):
            logger.debug(f"Exception fetching miner {uid}: {result}")
            continue
        if result is not None:
            output[uid] = result

    return output
