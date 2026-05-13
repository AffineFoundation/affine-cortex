"""
Challenger queue.

Each window picks at most one challenger to face the current champion. The
queue is materialized on demand from the ``miners`` table — there is no
separate queue table.

Selection rules:
  - challenge_status == "pending"
  - is_valid == true
  - uid != champion_uid (the current champion is never its own challenger)
  - order by (first_block ASC, uid ASC)
"""

from dataclasses import dataclass
from typing import Optional, Protocol


# Status values written into miners.challenge_status.
#
# Lifecycle:
#   pending  ─[claim_pending]→  in_progress
#   in_progress  ─[mark_terminated(WON)]→     champion
#                 ─[mark_terminated(LOST)]→    terminated_lost
#                 ─[mark_terminated(FAILED)]→  terminated_failed
#   champion  ─[mark_terminated(LOST)]→  terminated_lost   (dethroned)
#
# At any time the subnet has at most one row in ``champion`` state —
# kept in sync with ``system_config['champion'].uid``. The state's
# meaning differs from ``terminated_won`` (which doesn't exist in this
# state machine): the winner becomes the *active* champion, not a
# terminated bookmark.
STATUS_PENDING = "pending"
STATUS_IN_PROGRESS = "in_progress"
STATUS_CHAMPION = "champion"
STATUS_TERMINATED_LOST = "terminated_lost"
STATUS_TERMINATED_FAILED = "terminated_failed"

# What mark_terminated callers pass in.
OUTCOME_WON = "won"
OUTCOME_LOST = "lost"
OUTCOME_FAILED = "failed"

_OUTCOME_TO_STATUS = {
    OUTCOME_WON: STATUS_CHAMPION,
    OUTCOME_LOST: STATUS_TERMINATED_LOST,
    OUTCOME_FAILED: STATUS_TERMINATED_FAILED,
}


@dataclass(frozen=True)
class MinerCandidate:
    uid: int
    hotkey: str
    model: str
    revision: str
    first_block: int


class MinerQueueStore(Protocol):
    """Storage operations the queue needs.

    In production this wraps :class:`affine.database.dao.miners.MinersDAO`.
    In tests an in-memory implementation is used.
    """

    async def list_valid_pending(self) -> list[dict]: ...

    async def claim_pending(
        self, uid: int, window_id: int, *, expected_status: str = STATUS_PENDING
    ) -> bool:
        """Atomically transition ``uid`` from ``expected_status`` to ``in_progress``.

        Returns True iff the transition was applied (i.e. no race lost).
        """
        ...

    async def set_terminal(self, uid: int, new_status: str) -> None:
        """Set ``uid``'s ``challenge_status`` to a terminal value.

        Revision is effectively immutable per hotkey — the miners monitor
        rejects any second commit via the multi-commit rule — so we don't
        need a revision guard here.
        """
        ...


class ChallengerQueue:
    def __init__(self, store: MinerQueueStore):
        self._store = store

    async def pick_next(
        self, window_id: int, champion_uid: Optional[int]
    ) -> Optional[MinerCandidate]:
        """Return the earliest eligible challenger and mark it in_progress.

        A row is eligible if ``is_valid='true'`` and ``challenge_status`` is
        either ``'pending'`` or missing (the monitor seeds it on first
        refresh after a fresh on-chain commit; until then we still want
        the queue to claim — the conditional ``UpdateItem`` in the
        adapter tolerates the absent attribute too).

        Returns ``None`` if no eligible miner exists. On a lost race for
        the same candidate, the conditional write returns False and we
        walk on to the next row.
        """
        candidates = await self._store.list_valid_pending()
        ordered = sorted(
            candidates,
            key=lambda m: (m.get("first_block", float("inf")), m.get("uid", 0)),
        )
        for row in ordered:
            uid = row.get("uid")
            if uid is None or uid == champion_uid:
                continue
            status = row.get("challenge_status")
            if status is not None and status != STATUS_PENDING:
                continue
            if not _is_truthy(row.get("is_valid")):
                continue
            claimed = await self._store.claim_pending(uid, window_id)
            if claimed:
                return MinerCandidate(
                    uid=uid,
                    hotkey=row["hotkey"],
                    model=row.get("model", ""),
                    revision=row.get("revision", ""),
                    first_block=int(row.get("first_block", 0)),
                )
        return None

    async def mark_terminated(self, uid: int, outcome: str) -> None:
        """Transition ``uid`` to a terminal challenge state."""
        try:
            new_status = _OUTCOME_TO_STATUS[outcome]
        except KeyError:
            raise ValueError(f"unknown outcome {outcome!r}") from None
        await self._store.set_terminal(uid, new_status)


def _is_truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return False
