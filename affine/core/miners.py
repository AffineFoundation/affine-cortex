import os
import re
import json
import asyncio
from typing import Dict, List, Optional, Union
from affine.core.models import Miner
from affine.core.setup import NETUID
from affine.utils.subtensor import get_subtensor
from affine.utils.api_client import get_chute_info

logger = __import__("logging").getLogger("affine")

# Maximum allowed size for raw commit data (bytes)
MAX_COMMIT_DATA_SIZE = 10_000
# Maximum allowed length for individual string fields
MAX_FIELD_LENGTH = 500
# Pattern for valid model identifiers (org/model or model name, alphanumeric + hyphens/underscores/dots/slashes)
MODEL_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._\-/]{0,498}$")
# Pattern for valid revision identifiers (hex sha, branch name, tag)
REVISION_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._\-]{0,498}$")
# Required top-level keys in commit data
REQUIRED_COMMIT_FIELDS = {"model", "revision", "chute_id"}


def _validate_commit_data(commit_data: str, uid: int) -> Optional[dict]:
    """Validate and parse commit data JSON with structural checks.

    Returns parsed dict with only allowed fields, or None if invalid.
    """
    # Size limit check
    if len(commit_data) > MAX_COMMIT_DATA_SIZE:
        logger.warning(
            f"Commit data for uid={uid} exceeds max size "
            f"({len(commit_data)} > {MAX_COMMIT_DATA_SIZE}), skipping"
        )
        return None

    # Parse JSON with explicit error handling
    try:
        data = json.loads(commit_data)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Malformed JSON in commit data for uid={uid}: {e}")
        return None

    if not isinstance(data, dict):
        logger.warning(f"Commit data for uid={uid} is not a JSON object (got {type(data).__name__})")
        return None

    # Check required fields exist
    missing = REQUIRED_COMMIT_FIELDS - set(data.keys())
    if missing:
        logger.warning(f"Commit data for uid={uid} missing required fields: {missing}")
        return None

    # Validate field types — all required fields must be strings
    for field in REQUIRED_COMMIT_FIELDS:
        if not isinstance(data[field], str):
            logger.warning(
                f"Commit data for uid={uid}: field '{field}' must be str, "
                f"got {type(data[field]).__name__}"
            )
            return None
        if len(data[field]) > MAX_FIELD_LENGTH:
            logger.warning(
                f"Commit data for uid={uid}: field '{field}' exceeds max length "
                f"({len(data[field])} > {MAX_FIELD_LENGTH})"
            )
            return None

    # Validate model name format (prevent path traversal, injection)
    if not MODEL_PATTERN.match(data["model"]):
        logger.warning(
            f"Commit data for uid={uid}: invalid model name format: {data['model'][:80]!r}"
        )
        return None

    # Validate revision format
    if not REVISION_PATTERN.match(data["revision"]):
        logger.warning(
            f"Commit data for uid={uid}: invalid revision format: {data['revision'][:80]!r}"
        )
        return None

    # Strip to only known fields to prevent injection of unexpected keys
    return {k: data[k] for k in REQUIRED_COMMIT_FIELDS if k in data}


async def miners(
    uids: Optional[Union[int, List[int]]] = None,
    netuid: int = NETUID,
    meta: object = None,
) -> Dict[int, "Miner"]:
    """Query miner information from blockchain.

    Simplified version for miner SDK usage - returns basic miner info from blockchain commits.
    For validator use cases with filtering logic, refer to affine.src.monitor.miners_monitor.

    Args:
        uids: Miner UID(s) to query. If None, queries all UIDs.
        netuid: Network UID (default: from NETUID config)
        meta: Optional metagraph object (will be fetched if not provided)

    Returns:
        Dict mapping UID to Miner info. Only includes miners with valid commits.

    Example:
        >>> miner = await af.miners(7)
        >>> if miner:
        >>>     print(miner[7].model)
    """
    sub = await get_subtensor()
    meta = meta or await sub.metagraph(netuid)
    commits = await sub.get_all_revealed_commitments(netuid)

    if uids is None:
        uids = list(range(len(meta.hotkeys)))
    elif isinstance(uids, int):
        uids = [uids]

    # Log warning if commits contain hotkeys not present in metagraph
    metagraph_hotkeys = set(meta.hotkeys)
    unknown_hotkeys = set(commits.keys()) - metagraph_hotkeys
    if unknown_hotkeys:
        logger.warning(
            f"Found {len(unknown_hotkeys)} commit hotkeys not in metagraph — "
            f"possible stale data or injection attempt"
        )

    meta_sem = asyncio.Semaphore(int(os.getenv("AFFINE_META_CONCURRENCY", "12")))

    async def _fetch_miner(uid: int) -> Optional["Miner"]:
        try:
            hotkey = meta.hotkeys[uid]
            if hotkey not in commits:
                return None

            block, commit_data = commits[hotkey][-1]
            block = 0 if uid == 0 else block

            # Validate commit data structure and contents
            data = _validate_commit_data(commit_data, uid)
            if data is None:
                return None

            model = data["model"]
            miner_revision = data["revision"]
            chute_id = data["chute_id"]

            async with meta_sem:
                chute = await get_chute_info(chute_id)

            if not chute or not chute.get("hot", False):
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
        except Exception as e:
            logger.warning(f"Failed to fetch miner uid={uid}: {e}")
            return None

    results = await asyncio.gather(*(_fetch_miner(uid) for uid in uids))
    output = {uid: m for uid, m in zip(uids, results) if m is not None}

    return output
