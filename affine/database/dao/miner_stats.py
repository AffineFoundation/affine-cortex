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

    @classmethod
    def _extract_challenge_state(cls, row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not row:
            return {
                "challenge_consecutive_wins": 0,
                "challenge_total_wins": 0,
                "challenge_total_losses": 0,
                "challenge_consecutive_losses": 0,
                "challenge_checkpoints_passed": 0,
                "challenge_status": cls.STATUS_SAMPLING,
                "termination_reason": "",
            }
        return {field: row.get(field, cls._extract_challenge_state(None)[field])
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
        latest state for the same hotkey so old-system terminated state is
        still respected when the current snapshot is rebuilt.
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
        self, miners: Iterable[Dict[str, Any]],
    ) -> Dict[tuple[str, str], Dict[str, Any]]:
        out: Dict[tuple[str, str], Dict[str, Any]] = {}
        for miner in miners:
            hotkey = miner.get("hotkey")
            revision = miner.get("revision")
            if hotkey and revision:
                out[(str(hotkey), str(revision))] = await self.get_challenge_state(
                    str(hotkey), str(revision),
                )
        return out

    async def claim_for_challenge(
        self,
        *,
        hotkey: str,
        revision: str,
        model: str = "",
        window_id: int,
    ) -> bool:
        """Mark the historical row in-progress if it is still sampleable."""
        state = await self.get_challenge_state(hotkey, revision)
        if state.get("challenge_status") not in (None, "", self.STATUS_SAMPLING):
            return False

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
                    "SET challenge_status = :status, last_window_id = :wid, "
                    "challenge_claimed_at = :now, last_updated_at = :now, "
                    "hotkey = if_not_exists(hotkey, :hotkey), "
                    "revision = if_not_exists(revision, :revision), "
                    "model = if_not_exists(model, :model), "
                    "first_seen_at = if_not_exists(first_seen_at, :now), "
                    "is_currently_online = :online"
                ),
                ConditionExpression=(
                    "attribute_not_exists(challenge_status) "
                    "OR challenge_status = :sampling"
                ),
                ExpressionAttributeValues={
                    ":status": {"S": self.STATUS_IN_PROGRESS},
                    ":wid": {"N": str(window_id)},
                    ":now": {"N": str(now)},
                    ":hotkey": {"S": hotkey},
                    ":revision": {"S": revision},
                    ":model": {"S": model},
                    ":online": {"BOOL": True},
                    ":sampling": {"S": self.STATUS_SAMPLING},
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
    ) -> None:
        from affine.database.client import get_client

        now = int(time.time())
        await get_client().update_item(
            TableName=self.table_name,
            Key={
                "pk": {"S": self._make_pk(hotkey)},
                "sk": {"S": self._make_sk(revision)},
            },
            UpdateExpression=(
                "SET challenge_status = :status, termination_reason = :reason, "
                "last_updated_at = :now, hotkey = if_not_exists(hotkey, :hotkey), "
                "revision = if_not_exists(revision, :revision), "
                "first_seen_at = if_not_exists(first_seen_at, :now)"
            ),
            ExpressionAttributeValues={
                ":status": {"S": status},
                ":reason": {"S": termination_reason},
                ":now": {"N": str(now)},
                ":hotkey": {"S": hotkey},
                ":revision": {"S": revision},
            },
        )
