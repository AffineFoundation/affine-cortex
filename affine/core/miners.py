"""SDK helper: read on-chain miner commits and return ``Miner`` objects.

Public face of the SDK — ``affine.miners(uid)`` returns the model + revision
a miner committed, nothing else. The queue-window scorer hosts inference
itself, so there is no chute / slug lookup here.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Union

from affine.core.models import Miner
from affine.core.setup import NETUID
from affine.utils.subtensor import get_subtensor


logger = logging.getLogger("affine")


async def miners(
    uids: Optional[Union[int, List[int]]] = None,
    netuid: int = NETUID,
    meta: object = None,
) -> Dict[int, Miner]:
    """Return ``{uid: Miner}`` for every UID with a valid on-chain commit.

    A commit is valid when its JSON has both ``model`` and ``revision``.
    Unknown extra fields (e.g. legacy ``chute_id``) are tolerated and
    ignored.

    Args:
        uids: A single UID, a list, or ``None`` for every UID on the
            subnet.
        netuid: Defaults to ``affine.core.setup.NETUID``.
        meta: Optional pre-fetched metagraph.
    """
    sub = await get_subtensor()
    meta = meta or await sub.metagraph(netuid)
    commits = await sub.get_all_revealed_commitments(netuid)

    if uids is None:
        uids = list(range(len(meta.hotkeys)))
    elif isinstance(uids, int):
        uids = [uids]

    out: Dict[int, Miner] = {}
    for uid in uids:
        try:
            hotkey = meta.hotkeys[uid]
        except IndexError:
            continue
        entries = commits.get(hotkey)
        if not entries:
            continue
        block, commit_data = entries[-1]
        block = 0 if uid == 0 else int(block)
        try:
            data = json.loads(commit_data)
        except (TypeError, ValueError):
            continue
        model = data.get("model")
        revision = data.get("revision")
        if not model or not revision:
            continue
        out[uid] = Miner(
            uid=uid, hotkey=hotkey, model=model, revision=revision, block=block,
        )
    return out
