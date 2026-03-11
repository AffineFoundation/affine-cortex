"""
Load logprobs samples from DynamoDB and parse into MinerLogprobs objects.

Fetches full extra_compressed data (including token logprobs) for the
logprobs environment for all active miners.
"""

import asyncio
import json
from typing import Dict, List, Optional

import numpy as np

from affine.core.setup import logger
from affine.database.base_dao import BaseDAO
from affine.database.client import get_client
from affine.database.schema import get_table_name
from affine.src.anticopy.models import MinerLogprobs


LOGPROBS_ENV = "LOGPROBS"
_TABLE = get_table_name("sample_results")
MIN_TOKENS = 10  # samples with fewer tokens are considered invalid
TOP_K = 3  # number of top-k logprobs to include per position


def _parse_tokens(tokens: list) -> tuple:
    """Parse tokens list from logprobs extra into numpy array and topk/greedy lists.

    Returns:
        logprob_vec: np.ndarray shape (n_tokens * TOP_K,)
                     Each position contributes TOP_K logprob values from top_k probs.
                     Positions with fewer than TOP_K candidates are padded with -100.
        topk_list:   list of [{token, prob}] per position
        greedy_list: list of top-1 token strings per position
    """
    positions = sorted(tokens, key=lambda t: t["position"])
    topk_list = [t.get("top_k", []) for t in positions]
    greedy_list = [t["token"] for t in positions]

    # Build extended logprob vector: TOP_K logprobs per position
    logprobs = []
    for t in positions:
        top_k = t.get("top_k", [])
        for k in range(TOP_K):
            if k < len(top_k) and top_k[k].get("prob", 0) > 0:
                logprobs.append(np.log(top_k[k]["prob"]))
            elif k == 0:
                # Fallback: use the greedy logprob for rank-0
                logprobs.append(t["logprob"])
            else:
                logprobs.append(-100.0)
    logprob_vec = np.array(logprobs, dtype=np.float32)

    return logprob_vec, topk_list, greedy_list


class LogprobsLoader(BaseDAO):
    """Loads logprobs-env samples with full extra (token data) for all miners."""

    def __init__(self):
        self.table_name = _TABLE
        super().__init__()

    def _make_pk(self, hotkey: str, revision: str) -> str:
        return f"MINER#{hotkey}#REV#{revision}#ENV#{LOGPROBS_ENV}"

    async def _fetch_miner_samples(
        self,
        client,
        hotkey: str,
        revision: str,
    ) -> List[dict]:
        """Fetch all logprobs samples for one miner, including extra_compressed."""
        pk = self._make_pk(hotkey, revision)
        params = {
            "TableName": self.table_name,
            "KeyConditionExpression": "pk = :pk",
            "ExpressionAttributeValues": {":pk": {"S": pk}},
            # No ProjectionExpression – fetch all attributes including extra_compressed
        }

        all_items = []
        last_key = None
        while True:
            if last_key:
                params["ExclusiveStartKey"] = last_key
            resp = await client.query(**params)
            all_items.extend(resp.get("Items", []))
            last_key = resp.get("LastEvaluatedKey")
            if not last_key:
                break

        return all_items

    async def load_all_miners(
        self,
        miners: List[Dict],  # list of {uid, hotkey, revision}
        batch_size: int = 30,
    ) -> Dict[int, MinerLogprobs]:
        """Load logprobs samples for all miners in parallel.

        Args:
            miners:     list of miner dicts with keys uid, hotkey, revision
            batch_size: concurrent DB queries per batch

        Returns:
            Dict mapping uid -> MinerLogprobs
        """
        client = get_client()
        result: Dict[int, MinerLogprobs] = {}

        # Build coroutines
        coros = [
            self._fetch_miner_samples(client, m["hotkey"], m["revision"])
            for m in miners
        ]

        # Execute in batches to avoid overwhelming DynamoDB
        all_raw: List[List[dict]] = []
        for i in range(0, len(coros), batch_size):
            batch_results = await asyncio.gather(
                *coros[i : i + batch_size], return_exceptions=True
            )
            all_raw.extend(batch_results)

        # Parse results
        for miner, raw_items in zip(miners, all_raw):
            uid = miner["uid"]
            hotkey = miner["hotkey"]

            if isinstance(raw_items, Exception):
                logger.warning(
                    f"anti_copy: failed to load samples for uid={uid}: {raw_items}"
                )
                continue

            miner_lp = MinerLogprobs(uid=uid, hotkey=hotkey)

            for raw in raw_items:
                item = self._deserialize(raw)
                task_id = item.get("task_id")
                if task_id is None:
                    continue

                # extra_compressed is bytes after _deserialize (DynamoDB B type)
                extra_compressed = item.get("extra_compressed")
                if not extra_compressed:
                    continue
                try:
                    extra_json = self.decompress_data(extra_compressed)
                    extra = json.loads(extra_json)
                except Exception as e:
                    logger.debug(f"anti_copy: decompress failed uid={uid} task={task_id}: {e}")
                    continue

                tokens = extra.get("tokens")
                if not tokens:
                    continue

                try:
                    lp_vec, topk, greedy = _parse_tokens(tokens)
                except Exception as e:
                    logger.debug(f"anti_copy: token parse failed uid={uid} task={task_id}: {e}")
                    continue

                miner_lp.task_logprobs[task_id] = lp_vec
                miner_lp.task_topk[task_id] = topk
                miner_lp.task_tokens[task_id] = greedy

                # Load hidden_states if available
                hs = extra.get("hidden_states")
                if hs and isinstance(hs, list) and len(hs) > 1:
                    miner_lp.task_hidden_states[task_id] = np.array(
                        hs, dtype=np.float32
                    )

            if miner_lp.task_logprobs or miner_lp.task_hidden_states:
                result[uid] = miner_lp
            else:
                logger.debug(f"anti_copy: no valid logprobs for uid={uid}, skipping")

        logger.info(
            f"anti_copy: loaded logprobs for {len(result)}/{len(miners)} miners"
        )
        return result
