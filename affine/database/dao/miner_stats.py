"""Historical miner lifecycle state.

``miners`` is the current metagraph snapshot keyed by UID. This table is
keyed by ``(hotkey, revision)`` and keeps challenge lifecycle state even
after a miner leaves the active subnet.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Optional

from botocore.exceptions import ClientError

from affine.database.base_dao import BaseDAO
from affine.database.schema import get_table_name


class MinerStatsDAO(BaseDAO):
    """DAO for historical miner state."""

    STATUS_SAMPLING = "sampling"
    STATUS_IN_PROGRESS = "in_progress"
    STATUS_CHAMPION = "champion"
    STATUS_TERMINATED = "terminated"

    _CHALLENGE_FIELDS = (
        "challenge_consecutive_wins",
        "challenge_total_wins",
        "challenge_total_losses",
        "challenge_consecutive_losses",
        "challenge_checkpoints_passed",
        "challenge_status",
        "termination_reason",
        "admission_policy_identity",
        "admission_deferral_count",
        "admission_retry_after",
        "admission_deferral_exhausted",
        "admission_last_deployment_generation",
    )

    def __init__(self):
        self.table_name = get_table_name("miner_stats")
        super().__init__()

    def _make_pk(self, hotkey: str) -> str:
        return f"HOTKEY#{hotkey}"

    def _make_sk(self, revision: str) -> str:
        return f"REV#{revision}"

    async def get_miner_stats(
        self, hotkey: str, revision: str,
    ) -> Optional[Dict[str, Any]]:
        return await self.get(self._make_pk(hotkey), self._make_sk(revision))

    async def get_all_historical_miners(self) -> List[Dict[str, Any]]:
        from affine.database.client import get_client

        client = get_client()
        params = {"TableName": self.table_name}
        out: List[Dict[str, Any]] = []
        while True:
            response = await client.scan(**params)
            out.extend(
                self._deserialize(item)
                for item in response.get("Items", [])
            )
            last_key = response.get("LastEvaluatedKey")
            if not last_key:
                return out
            params["ExclusiveStartKey"] = last_key

    async def list_by_challenge_status(
        self, status: str,
    ) -> List[Dict[str, Any]]:
        """Scan all rows currently in the given lifecycle status."""
        from affine.database.client import get_client

        client = get_client()
        params = {
            "TableName": self.table_name,
            "FilterExpression": "challenge_status = :s",
            "ExpressionAttributeValues": {":s": {"S": status}},
        }
        out: List[Dict[str, Any]] = []
        while True:
            response = await client.scan(**params)
            out.extend(
                self._deserialize(item)
                for item in response.get("Items", [])
            )
            last_key = response.get("LastEvaluatedKey")
            if not last_key:
                return out
            params["ExclusiveStartKey"] = last_key

    async def update_miner_info(
        self,
        *,
        hotkey: str,
        revision: str,
        model: str,
        uid: Optional[int] = None,
        first_block: Optional[int] = None,
        block_number: Optional[int] = None,
        is_valid: Optional[bool] = None,
        invalid_reason: Optional[str] = None,
        model_hash: str = "",
        model_type: str = "",
        is_online: bool = True,
    ) -> None:
        """Persist current miner metadata without overwriting lifecycle state.

        ``miner_stats`` is the durable hotkey/revision history table. The
        monitor calls this on every refresh so a miner's identity and validity
        remain available after the UID leaves the current ``miners`` snapshot.
        Challenge fields are initialized only for brand-new rows and otherwise
        preserved.
        """
        from affine.database.client import get_client

        now = int(time.time())
        values = {
            ":hotkey": {"S": hotkey},
            ":revision": {"S": revision},
            ":now": {"N": str(now)},
            ":online": {"BOOL": is_online},
            ":sampling": {"S": self.STATUS_SAMPLING},
            ":empty": {"S": ""},
            ":zero": {"N": "0"},
        }
        update_parts = [
            "hotkey = :hotkey",
            "revision = :revision",
            "last_updated_at = :now",
            "first_seen_at = if_not_exists(first_seen_at, :now)",
            "is_currently_online = :online",
            "challenge_status = if_not_exists(challenge_status, :sampling)",
            "termination_reason = if_not_exists(termination_reason, :empty)",
            "challenge_consecutive_wins = if_not_exists(challenge_consecutive_wins, :zero)",
            "challenge_total_wins = if_not_exists(challenge_total_wins, :zero)",
            "challenge_total_losses = if_not_exists(challenge_total_losses, :zero)",
            "challenge_consecutive_losses = if_not_exists(challenge_consecutive_losses, :zero)",
            "challenge_checkpoints_passed = if_not_exists(challenge_checkpoints_passed, :zero)",
        ]
        if model:
            values[":model"] = {"S": model}
            update_parts.append("model = :model")
        if model_hash:
            values[":model_hash"] = {"S": model_hash}
            update_parts.append("model_hash = :model_hash")
        if model_type:
            values[":model_type"] = {"S": model_type}
            update_parts.append("model_type = :model_type")
        if uid is not None:
            values[":uid"] = {"N": str(uid)}
            update_parts.append("#uid = :uid")
        if first_block is not None:
            values[":first_block"] = {"N": str(first_block)}
            update_parts.append("first_block = if_not_exists(first_block, :first_block)")
        if block_number is not None:
            values[":block_number"] = {"N": str(block_number)}
            update_parts.append("block_number = :block_number")
        if is_valid is not None:
            values[":is_valid"] = {"BOOL": bool(is_valid)}
            update_parts.append("is_valid = :is_valid")
        values[":invalid_reason"] = (
            {"S": str(invalid_reason)}
            if invalid_reason is not None
            else {"NULL": True}
        )
        update_parts.append("invalid_reason = :invalid_reason")

        params = {
            "TableName": self.table_name,
            "Key": {
                "pk": {"S": self._make_pk(hotkey)},
                "sk": {"S": self._make_sk(revision)},
            },
            "UpdateExpression": f"SET {', '.join(update_parts)}",
            "ExpressionAttributeValues": values,
        }
        if uid is not None:
            params["ExpressionAttributeNames"] = {"#uid": "uid"}
        await get_client().update_item(**params)

    @classmethod
    def _extract_challenge_state(cls, row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        defaults = {
            "challenge_consecutive_wins": 0,
            "challenge_total_wins": 0,
            "challenge_total_losses": 0,
            "challenge_consecutive_losses": 0,
            "challenge_checkpoints_passed": 0,
            "challenge_status": cls.STATUS_SAMPLING,
            "termination_reason": "",
            "admission_policy_identity": "",
            "admission_deferral_count": 0,
            "admission_retry_after": 0,
            "admission_deferral_exhausted": False,
            "admission_last_deployment_generation": "",
        }
        if not row:
            return defaults
        return {field: row.get(field, defaults[field])
                for field in cls._CHALLENGE_FIELDS}

    @staticmethod
    def _has_challenge_state(row: Optional[Dict[str, Any]]) -> bool:
        if not row:
            return False
        if row.get("challenge_status") in {
            MinerStatsDAO.STATUS_TERMINATED,
            MinerStatsDAO.STATUS_IN_PROGRESS,
            MinerStatsDAO.STATUS_CHAMPION,
        }:
            return True
        for key in (
            "challenge_consecutive_wins",
            "challenge_total_wins",
            "challenge_total_losses",
            "challenge_consecutive_losses",
            "challenge_checkpoints_passed",
        ):
            if row.get(key, 0):
                return True
        return False

    async def get_challenge_state(
        self, hotkey: str, revision: str,
    ) -> Dict[str, Any]:
        """Return lifecycle state for ``(hotkey, revision)``.

        The direct row wins. If it has no lifecycle fields yet, inherit the
        latest state for the same hotkey. This fallback is policy, not just
        display behavior: one hotkey gets one lifetime challenge opportunity,
        so a terminated previous revision blocks claim_for_challenge for a
        later revision too.
        """
        direct = await self.get_miner_stats(hotkey, revision)
        if self._has_challenge_state(direct):
            return self._extract_challenge_state(direct)

        rows = await self.query(self._make_pk(hotkey))
        rows = [
            row for row in rows
            if row.get("revision") != revision and self._has_challenge_state(row)
        ]
        if rows:
            rows.sort(key=lambda r: int(r.get("last_updated_at", 0) or 0), reverse=True)
            return self._extract_challenge_state(rows[0])
        return self._extract_challenge_state(direct)

    async def build_challenge_state_map(
        self,
        miners: Iterable[Dict[str, Any]],
        *,
        concurrency: int = 32,
    ) -> Dict[tuple[str, str], Dict[str, Any]]:
        """Return lifecycle state for current miners keyed by (hotkey, revision).

        The rank API calls this on every request. Keep the reads bounded but
        parallel so the endpoint is not gated by 256 sequential DynamoDB gets.
        """
        import asyncio

        out: Dict[tuple[str, str], Dict[str, Any]] = {}
        pairs = [
            (str(miner["hotkey"]), str(miner["revision"]))
            for miner in miners
            if miner.get("hotkey") and miner.get("revision")
        ]
        sem = asyncio.Semaphore(max(1, int(concurrency)))

        async def _one(hotkey: str, revision: str):
            async with sem:
                state = await self.get_challenge_state(hotkey, revision)
            return (hotkey, revision), state

        results = await asyncio.gather(
            *(_one(hotkey, revision) for hotkey, revision in pairs)
        )
        for key, state in results:
            out[key] = state
        return out

    async def claim_for_challenge(
        self,
        *,
        hotkey: str,
        revision: str,
        model: str = "",
        window_id: int,
        admission_policy_identity: Optional[str] = None,
    ) -> bool:
        """Mark the historical row in-progress if it is still sampleable."""
        state = await self.get_challenge_state(hotkey, revision)
        if state.get("challenge_status") not in (None, "", self.STATUS_SAMPLING):
            return False
        policy_identity = str(admission_policy_identity or "").strip()
        now = int(time.time())
        if (
            policy_identity
            and state.get("admission_policy_identity") == policy_identity
            and (
                state.get("admission_deferral_exhausted") is True
                or int(state.get("admission_retry_after") or 0) > now
            )
        ):
            return False

        from affine.database.client import get_client

        client = get_client()
        condition = (
            "(attribute_not_exists(challenge_status) "
            "OR challenge_status = :sampling)"
        )
        values = {
            ":status": {"S": self.STATUS_IN_PROGRESS},
            ":wid": {"N": str(window_id)},
            ":now": {"N": str(now)},
            ":hotkey": {"S": hotkey},
            ":revision": {"S": revision},
            ":model": {"S": model},
            ":online": {"BOOL": True},
            ":sampling": {"S": self.STATUS_SAMPLING},
        }
        if policy_identity:
            condition += (
                " AND (attribute_not_exists(admission_policy_identity) "
                "OR admission_policy_identity <> :admission_policy "
                "OR ((attribute_not_exists(admission_deferral_exhausted) "
                "OR admission_deferral_exhausted = :false) "
                "AND (attribute_not_exists(admission_retry_after) "
                "OR admission_retry_after <= :now)))"
            )
            values.update({
                ":admission_policy": {"S": policy_identity},
                ":false": {"BOOL": False},
            })
        try:
            await client.update_item(
                TableName=self.table_name,
                Key={
                    "pk": {"S": self._make_pk(hotkey)},
                    "sk": {"S": self._make_sk(revision)},
                },
                UpdateExpression=(
                    "SET challenge_status = :status, last_window_id = :wid, "
                    "challenge_claimed_at = :now, last_updated_at = :now, "
                    "hotkey = if_not_exists(hotkey, :hotkey), "
                    "revision = if_not_exists(revision, :revision), "
                    "model = if_not_exists(model, :model), "
                    "first_seen_at = if_not_exists(first_seen_at, :now), "
                    "is_currently_online = :online"
                ),
                ConditionExpression=condition,
                ExpressionAttributeValues=values,
            )
            return True
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
                return False
            raise

    async def defer_admission_for_challenge(
        self,
        *,
        hotkey: str,
        revision: str,
        policy_identity: str,
        deployment_generation: str,
        base_delay_seconds: int = 300,
        max_attempts: int = 3,
        now: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Atomically requeue one expired admission with bounded backoff.

        The deployment generation is an idempotency token.  If the scheduler
        crashes after this write but before clearing ``current_battle``, the
        next tick observes the same token and treats the already-sampling row
        as success without incrementing the deferral counter again.  Transport
        errors propagate so callers never confuse an unconfirmed write with an
        idempotent retry.
        """

        policy = str(policy_identity).strip()
        generation = str(deployment_generation).strip()
        if not policy or not generation:
            raise ValueError(
                "policy_identity and deployment_generation are required"
            )
        base_delay = int(base_delay_seconds)
        maximum = int(max_attempts)
        if base_delay <= 0:
            raise ValueError("base_delay_seconds must be positive")
        if maximum < 1 or maximum > 10:
            raise ValueError("max_attempts must be between 1 and 10")

        from affine.database.client import get_client

        timestamp = int(time.time()) if now is None else int(now)
        client = get_client()
        key = {
            "pk": {"S": self._make_pk(hotkey)},
            "sk": {"S": self._make_sk(revision)},
        }
        for _ in range(5):
            response = await client.get_item(
                TableName=self.table_name,
                Key=key,
                ConsistentRead=True,
            )
            raw_item = response.get("Item")
            row = self._deserialize(raw_item) if raw_item else {}
            raw_status = row.get("challenge_status")
            status = str(raw_status or self.STATUS_SAMPLING)
            previous_generation = str(
                row.get("admission_last_deployment_generation") or ""
            )
            previous_policy = str(row.get("admission_policy_identity") or "")
            if (
                previous_policy == policy
                and previous_generation == generation
            ):
                return {
                    "released": status == self.STATUS_SAMPLING,
                    "idempotent": True,
                    "deferral_count": int(
                        row.get("admission_deferral_count") or 0
                    ),
                    "next_retry_at": int(row.get("admission_retry_after") or 0),
                    "exhausted": row.get("admission_deferral_exhausted") is True,
                    "status": status,
                }
            if status not in {self.STATUS_SAMPLING, self.STATUS_IN_PROGRESS}:
                return {
                    "released": False,
                    "idempotent": False,
                    "deferral_count": int(
                        row.get("admission_deferral_count") or 0
                    ),
                    "next_retry_at": int(row.get("admission_retry_after") or 0),
                    "exhausted": row.get("admission_deferral_exhausted") is True,
                    "status": status,
                }

            old_count = (
                int(row.get("admission_deferral_count") or 0)
                if previous_policy == policy
                else 0
            )
            count = min(maximum, old_count + 1)
            exhausted = count >= maximum
            delay = base_delay * (2 ** (count - 1))
            next_retry_at = timestamp + delay
            values = {
                ":sampling": {"S": self.STATUS_SAMPLING},
                ":policy": {"S": policy},
                ":generation": {"S": generation},
                ":count": {"N": str(count)},
                ":retry": {"N": str(next_retry_at)},
                ":exhausted": {"BOOL": exhausted},
                ":maximum": {"N": str(maximum)},
                ":now": {"N": str(timestamp)},
                ":hotkey": {"S": hotkey},
                ":revision": {"S": revision},
            }
            if "challenge_status" in row:
                status_condition = "challenge_status = :previous_status"
                values[":previous_status"] = {"S": str(raw_status or "")}
            else:
                status_condition = "attribute_not_exists(challenge_status)"
            if "admission_last_deployment_generation" in row:
                generation_condition = (
                    "admission_last_deployment_generation = "
                    ":previous_generation"
                )
                values[":previous_generation"] = {"S": previous_generation}
            else:
                generation_condition = (
                    "attribute_not_exists("
                    "admission_last_deployment_generation)"
                )
            if "admission_policy_identity" in row:
                policy_condition = (
                    "admission_policy_identity = :previous_policy"
                )
                values[":previous_policy"] = {"S": previous_policy}
            else:
                policy_condition = (
                    "attribute_not_exists(admission_policy_identity)"
                )
            try:
                await client.update_item(
                    TableName=self.table_name,
                    Key=key,
                    UpdateExpression=(
                        "SET challenge_status = :sampling, "
                        "admission_policy_identity = :policy, "
                        "admission_last_deployment_generation = :generation, "
                        "admission_deferral_count = :count, "
                        "admission_retry_after = :retry, "
                        "admission_deferral_exhausted = :exhausted, "
                        "admission_deferral_max_attempts = :maximum, "
                        "last_updated_at = :now, "
                        "hotkey = if_not_exists(hotkey, :hotkey), "
                        "revision = if_not_exists(revision, :revision)"
                    ),
                    ConditionExpression=(
                        f"({status_condition}) AND "
                        f"({generation_condition}) AND "
                        f"({policy_condition})"
                    ),
                    ExpressionAttributeValues=values,
                )
            except ClientError as exc:
                if exc.response.get("Error", {}).get("Code") == (
                    "ConditionalCheckFailedException"
                ):
                    continue
                raise
            return {
                "released": True,
                "idempotent": False,
                "deferral_count": count,
                "next_retry_at": next_retry_at,
                "exhausted": exhausted,
                "status": self.STATUS_SAMPLING,
            }
        raise RuntimeError("admission deferral update lost repeated write races")

    async def release_claim_for_challenge(
        self,
        *,
        hotkey: str,
        revision: str,
    ) -> bool:
        """Revert in_progress → sampling. Used when an active-battle
        deploy hits an infrastructure transient (SSH transport down,
        host unreachable): the miner had no chance to fail and must
        stay re-pickable. Atomic — succeeds only if the row is
        currently ``in_progress``."""
        from affine.database.client import get_client

        now = int(time.time())
        client = get_client()
        try:
            await client.update_item(
                TableName=self.table_name,
                Key={
                    "pk": {"S": self._make_pk(hotkey)},
                    "sk": {"S": self._make_sk(revision)},
                },
                UpdateExpression=(
                    "SET challenge_status = :sampling, "
                    "last_updated_at = :now"
                ),
                ConditionExpression="challenge_status = :in_progress",
                ExpressionAttributeValues={
                    ":sampling": {"S": self.STATUS_SAMPLING},
                    ":in_progress": {"S": self.STATUS_IN_PROGRESS},
                    ":now": {"N": str(now)},
                },
            )
            return True
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
                return False
            raise

    async def update_challenge_status(
        self,
        *,
        hotkey: str,
        revision: str,
        status: str,
        termination_reason: str = "",
        scores_by_env: Optional[Dict[str, Dict[str, float]]] = None,
        opponent_scores_by_env: Optional[Dict[str, Dict[str, float]]] = None,
        battle_task_ids: Optional[Dict[str, List[int]]] = None,
        scores_refresh_block: Optional[int] = None,
        terminated_at_block: Optional[int] = None,
    ) -> None:
        """Flip lifecycle state. When called at termination with score
        data, also freezes the comparator's decide-time view onto the
        row in the SAME atomic write: ``scores_by_env`` carries the
        miner's ``{env: {count, avg, champion_overlap_avg?}}`` view,
        ``opponent_scores_by_env`` carries the opponent's same-overlap
        ``{env: {count, avg}}`` view, ``battle_task_ids`` carries the
        common task ids used for that comparison, and
        ``terminated_at_block`` marks the row immutable (live writes
        from :class:`LiveScoresMonitor` skip rows where this attribute
        is set).
        """
        from affine.database.client import get_client

        now = int(time.time())
        values: Dict[str, Any] = {
            ":status": {"S": status},
            ":reason": {"S": termination_reason},
            ":now": {"N": str(now)},
            ":hotkey": {"S": hotkey},
            ":revision": {"S": revision},
        }
        update_parts = [
            "challenge_status = :status",
            "termination_reason = :reason",
            "last_updated_at = :now",
            "hotkey = if_not_exists(hotkey, :hotkey)",
            "revision = if_not_exists(revision, :revision)",
            "first_seen_at = if_not_exists(first_seen_at, :now)",
        ]
        if scores_by_env is not None:
            values[":sc"] = self._serialize({"_v": scores_by_env})["_v"]
            update_parts.append("scores_by_env = :sc")
        if opponent_scores_by_env is not None:
            values[":opp"] = self._serialize({"_v": opponent_scores_by_env})["_v"]
            update_parts.append("opponent_scores_by_env = :opp")
        if battle_task_ids is not None:
            values[":btids"] = self._serialize({"_v": battle_task_ids})["_v"]
            update_parts.append("battle_task_ids = :btids")
        if scores_refresh_block is not None:
            values[":srb"] = {"N": str(int(scores_refresh_block))}
            update_parts.append("scores_refresh_block = :srb")
        if terminated_at_block is not None:
            values[":tb"] = {"N": str(int(terminated_at_block))}
            update_parts.append("terminated_at_block = :tb")
        if status == self.STATUS_TERMINATED:
            # Wall-clock unix-second snapshot frozen on the first flip to
            # terminated. ``if_not_exists`` makes any later termination write
            # (recovery path, duplicate dethrone) idempotent so downstream
            # consumers see a stable timestamp.
            update_parts.append(
                "terminated_at = if_not_exists(terminated_at, :now)"
            )
        await get_client().update_item(
            TableName=self.table_name,
            Key={
                "pk": {"S": self._make_pk(hotkey)},
                "sk": {"S": self._make_sk(revision)},
            },
            UpdateExpression="SET " + ", ".join(update_parts),
            ExpressionAttributeValues=values,
        )

    async def terminate_if_sampling(
        self, *,
        hotkey: str,
        revision: str,
        reason: str,
    ) -> bool:
        """Flip the row to ``terminated`` iff it's still ``sampling``.

        Used by the monitor when a deterministic, miner-attributable invalid
        signal (HF repo deleted/privated, immutable commit-policy reject,
        malicious template, ...) arrives on a row the scheduler hasn't claimed
        yet. The conditional write protects in_progress / champion /
        already-terminated rows from being clobbered by a delayed monitor
        cycle. Returns True iff this call actually wrote the row.
        """
        from affine.database.client import get_client

        now = int(time.time())
        try:
            await get_client().update_item(
                TableName=self.table_name,
                Key={
                    "pk": {"S": self._make_pk(hotkey)},
                    "sk": {"S": self._make_sk(revision)},
                },
                UpdateExpression=(
                    "SET challenge_status = :terminated, "
                    "termination_reason = :reason, "
                    "last_updated_at = :now, "
                    "terminated_at = if_not_exists(terminated_at, :now), "
                    "hotkey = if_not_exists(hotkey, :hotkey), "
                    "revision = if_not_exists(revision, :revision), "
                    "first_seen_at = if_not_exists(first_seen_at, :now)"
                ),
                ConditionExpression=(
                    "attribute_not_exists(challenge_status) "
                    "OR challenge_status = :sampling"
                ),
                ExpressionAttributeValues={
                    ":terminated": {"S": self.STATUS_TERMINATED},
                    ":sampling": {"S": self.STATUS_SAMPLING},
                    ":reason": {"S": reason},
                    ":now": {"N": str(now)},
                    ":hotkey": {"S": hotkey},
                    ":revision": {"S": revision},
                },
            )
            return True
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
                return False
            raise

    async def update_live_scores(
        self, *,
        hotkey: str,
        revision: str,
        scores_by_env: Dict[str, Dict[str, float]],
        scores_refresh_block: int,
    ) -> None:
        """Overwrite the per-miner score snapshot for the current
        refresh_block. Called by :class:`LiveScoresMonitor` every cycle.

        Same ``scores_by_env`` attribute as
        :meth:`update_challenge_status` — the row stores ONE per-env
        snapshot, with ``terminated_at_block`` (if present) marking it
        as frozen-by-the-decision. The conditional write below skips
        any row that has already been frozen so a live refresh can't
        clobber the comparator's decide-time view.
        """
        from affine.database.client import get_client

        now = int(time.time())
        try:
            await get_client().update_item(
                TableName=self.table_name,
                Key={
                    "pk": {"S": self._make_pk(hotkey)},
                    "sk": {"S": self._make_sk(revision)},
                },
                UpdateExpression=(
                    "SET scores_by_env = :sc, "
                    "scores_refresh_block = :srb, "
                    "last_updated_at = :now, "
                    "hotkey = if_not_exists(hotkey, :hotkey), "
                    "revision = if_not_exists(revision, :revision), "
                    "first_seen_at = if_not_exists(first_seen_at, :now)"
                ),
                ConditionExpression="attribute_not_exists(terminated_at_block)",
                ExpressionAttributeValues={
                    ":sc": self._serialize({"_v": scores_by_env})["_v"],
                    ":srb": {"N": str(int(scores_refresh_block))},
                    ":now": {"N": str(now)},
                    ":hotkey": {"S": hotkey},
                    ":revision": {"S": revision},
                },
            )
        except ClientError as e:
            # Frozen rows (terminated_at_block already set) reject the
            # write — expected; the comparator's decision-time view is
            # the canonical record for those miners. Other errors
            # bubble up so the caller's retry/log path sees them.
            if e.response.get("Error", {}).get("Code") != "ConditionalCheckFailedException":
                raise

    async def build_display_scores_map(
        self,
        miners: Iterable[Dict[str, Any]],
        *,
        current_refresh_block: Optional[int] = None,
        concurrency: int = 32,
    ) -> Dict[str, Dict[str, Any]]:
        """Return ``{uid_str: {"scores": {env: ...}, "frozen": bool}}``
        for the rank API. One read per miner — parallelized so the
        /rank/current endpoint isn't gated by 256 sequential DDB gets.

        A row contributes when EITHER:
          * ``terminated_at_block`` is set → ``frozen=True``: always
            surfaced, regardless of ``current_refresh_block``.
          * ``scores_refresh_block`` matches ``current_refresh_block``
            → ``frozen=False``: fresh live aggregate.

        Stale-and-not-frozen rows are dropped — they were live snapshots
        from a previous task pool and the CLI must not show those numbers.

        Pass the full ``get_all_miners`` set, not just valid ones: a
        terminated miner that later turned invalid must still surface
        its frozen scores.
        """
        import asyncio

        miners_list = [
            m for m in miners
            if m.get("hotkey") and m.get("revision") and m.get("uid") is not None
        ]
        sem = asyncio.Semaphore(max(1, int(concurrency)))

        async def _one(miner):
            async with sem:
                row = await self.get_miner_stats(
                    str(miner["hotkey"]), str(miner["revision"]),
                )
            return int(miner["uid"]), row

        results = await asyncio.gather(*(_one(m) for m in miners_list))

        out: Dict[str, Dict[str, Any]] = {}
        for uid, row in results:
            if not row:
                continue
            scores = row.get("scores_by_env")
            if not isinstance(scores, dict) or not scores:
                continue
            terminated = row.get("terminated_at_block")
            if terminated is not None:
                out[str(uid)] = {"scores": scores, "frozen": True}
                continue
            refresh = row.get("scores_refresh_block")
            if (
                current_refresh_block is not None
                and refresh is not None
                and int(refresh) == int(current_refresh_block)
            ):
                out[str(uid)] = {"scores": scores, "frozen": False}
        return out
