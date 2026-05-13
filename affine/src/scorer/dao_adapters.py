"""
DynamoDB adapters used by the flow scheduler and executor.

  MinersQueueAdapter   wraps MinersDAO into ChallengerQueue's storage protocol
  SampleResultsAdapter wraps SampleResultsDAO for persist + count + read
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from botocore.exceptions import ClientError

from affine.core.setup import logger
from affine.database.client import get_client
from affine.database.dao.miners import MinersDAO
from affine.database.dao.sample_results import SampleResultsDAO

from .challenger_queue import STATUS_PENDING


# --------------------------------------------------------------------------- #
# Miner queue
# --------------------------------------------------------------------------- #


class MinersQueueAdapter:
    """Implements ``ChallengerQueue``'s storage protocol on top of ``MinersDAO``.

    ``claim_pending`` uses DynamoDB ``ConditionExpression`` to atomically
    transition a row from ``pending`` to ``in_progress`` — racing windows
    can't claim the same miner twice.
    """

    def __init__(self, dao: Optional[MinersDAO] = None):
        self._dao = dao or MinersDAO()
        self._table_name = self._dao.table_name

    async def list_valid_pending(self) -> List[Dict[str, Any]]:
        """Return every is_valid=true miner. Status filtering is done by
        the queue itself in memory — the subnet has ≤256 miners so a
        single GSI scan is cheap."""
        return await self._dao.get_valid_miners()

    async def claim_pending(
        self, uid: int, window_id: int, *, expected_status: str = STATUS_PENDING,
    ) -> bool:
        client = get_client()
        pk = self._dao._make_pk(uid)
        try:
            await client.update_item(
                TableName=self._table_name,
                Key={"pk": {"S": pk}},
                UpdateExpression=(
                    "SET challenge_status = :new, last_window_id = :wid, "
                    "challenge_claimed_at = :now"
                ),
                ConditionExpression=(
                    "attribute_exists(pk) AND ("
                    "attribute_not_exists(challenge_status) "
                    "OR challenge_status = :expected"
                    ")"
                ),
                ExpressionAttributeValues={
                    ":new": {"S": "in_progress"},
                    ":wid": {"N": str(window_id)},
                    ":now": {"N": str(int(time.time()))},
                    ":expected": {"S": expected_status},
                },
            )
            return True
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            if code == "ConditionalCheckFailedException":
                return False
            logger.warning(
                f"MinersQueueAdapter.claim_pending(uid={uid}) failed: {code}: {e}"
            )
            return False

    async def set_terminal(self, uid: int, new_status: str) -> None:
        client = get_client()
        pk = self._dao._make_pk(uid)
        try:
            await client.update_item(
                TableName=self._table_name,
                Key={"pk": {"S": pk}},
                UpdateExpression=(
                    "SET challenge_status = :new, challenge_terminated_at = :now"
                ),
                ConditionExpression="attribute_exists(pk)",
                ExpressionAttributeValues={
                    ":new": {"S": new_status},
                    ":now": {"N": str(int(time.time()))},
                },
            )
        except ClientError as e:
            logger.warning(
                f"MinersQueueAdapter.set_terminal(uid={uid}, status={new_status}) "
                f"failed: {e}"
            )


# --------------------------------------------------------------------------- #
# Sample-results reader / writer
# --------------------------------------------------------------------------- #


class SampleResultsAdapter:
    """Wraps ``SampleResultsDAO``.

    Rows are keyed by ``(hotkey, revision, env, task_id)`` only — there's
    no ``window_id`` filtering. The flow scheduler asks "how many of these
    task_ids does miner X have rows for?", which counts what's reusable
    across task-id refreshes. Stale tail-overlap samples (latest mode +
    static dataset_range → same task_ids recur) thus get reused, saving
    GPU time.
    """

    def __init__(
        self, *,
        dao: Optional[SampleResultsDAO] = None,
        validator_hotkey: str = "scheduler",
    ):
        self._dao = dao or SampleResultsDAO()
        self._table_name = self._dao.table_name
        self._validator_hotkey = validator_hotkey

    async def persist(
        self, *,
        miner_hotkey: str,
        model_revision: str,
        model: str,
        env: str,
        task_id: int,
        score: float,
        latency_ms: int,
        extra: Dict[str, Any],
        block_number: int,
        refresh_block: int,
    ) -> None:
        """Write one sample row tagged with ``refresh_block`` so the
        comparator can filter to current-task-batch samples only.

        Overwrite-on-collision is intentional: a re-sample within the
        same refresh (rare — executor's ``has_sample`` filters this out)
        or a fresh re-sample after a task-id pool refresh produces a
        semantically newer row that should replace the old one."""
        extra_json = json.dumps(extra, separators=(",", ":"))
        extra_compressed = self._dao.compress_data(extra_json)
        ttl_seconds = int(time.time()) + (30 * 86400)
        item = {
            "pk": self._dao._make_pk(miner_hotkey, model_revision, env),
            "sk": self._dao._make_sk(str(task_id)),
            "miner_hotkey": miner_hotkey,
            "model_revision": model_revision,
            "model": model,
            "env": env,
            "task_id": int(task_id),
            "score": score,
            "latency_ms": latency_ms,
            "timestamp": int(time.time() * 1000),
            "gsi_partition": "SAMPLE",
            "extra_compressed": extra_compressed,
            "validator_hotkey": self._validator_hotkey,
            "block_number": block_number,
            "signature": "",
            "ttl": ttl_seconds,
            "refresh_block": int(refresh_block),
        }
        client = get_client()
        serialized = self._dao._serialize(item)
        await client.put_item(TableName=self._table_name, Item=serialized)

    async def has_sample(
        self, miner_hotkey: str, model_revision: str, env: str, task_id: int,
        refresh_block: int,
    ) -> bool:
        """True iff a sample row exists for this (miner, env, task_id) AND
        its ``refresh_block`` matches the current task-id pool's refresh.

        Returning False for a row from a previous refresh forces the
        executor to re-sample, so the comparator sees only fresh data
        for the current contest."""
        item = await self._dao.get_sample_by_task_id(
            miner_hotkey=miner_hotkey,
            model_revision=model_revision,
            env=env,
            task_id=str(task_id),
            include_extra=False,
        )
        if not item:
            return False
        return int(item.get("refresh_block", -1)) == int(refresh_block)

    async def count_samples_for_tasks(
        self, hotkey: str, revision: str, env: str, task_ids: List[int],
        refresh_block: int,
    ) -> int:
        """Return how many of ``task_ids`` have a sample row tagged with
        the current ``refresh_block``."""
        if not task_ids:
            return 0
        scores = await self.read_scores_for_tasks(
            hotkey, revision, env, task_ids, refresh_block=refresh_block,
        )
        return len(scores)

    async def read_scores_for_tasks(
        self, hotkey: str, revision: str, env: str, task_ids: List[int],
        refresh_block: int,
    ) -> Dict[int, float]:
        """Return ``{task_id: score}`` for the requested task_ids restricted
        to rows whose ``refresh_block`` matches the current task-id pool.

        Single Query on the (pk = MINER#hk#REV#rev#ENV#env) partition,
        then filter in Python to the requested task_id subset AND to the
        current refresh.
        """
        if not task_ids:
            return {}
        wanted = {int(t) for t in task_ids}
        client = get_client()
        pk = self._dao._make_pk(hotkey, revision, env)
        out: Dict[int, float] = {}
        exclusive_start: Optional[Dict[str, Any]] = None
        target_refresh = int(refresh_block)
        while True:
            params: Dict[str, Any] = {
                "TableName": self._table_name,
                "KeyConditionExpression": "pk = :pk AND begins_with(sk, :sk)",
                "ExpressionAttributeValues": {
                    ":pk": {"S": pk},
                    ":sk": {"S": "TASK#"},
                },
                "ProjectionExpression": "task_id, score, refresh_block",
            }
            if exclusive_start:
                params["ExclusiveStartKey"] = exclusive_start
            resp = await client.query(**params)
            for raw in resp.get("Items", []):
                row = self._dao._deserialize(raw)
                tid = row.get("task_id")
                score = row.get("score")
                row_refresh = row.get("refresh_block")
                if tid is None or score is None or row_refresh is None:
                    continue
                if int(row_refresh) != target_refresh:
                    continue
                tid_int = int(tid)
                if tid_int in wanted:
                    out[tid_int] = float(score)
            exclusive_start = resp.get("LastEvaluatedKey")
            if not exclusive_start:
                break
        return out
