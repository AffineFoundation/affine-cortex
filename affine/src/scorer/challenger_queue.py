"""
Challenger queue.

Each window picks at most one challenger to face the current champion. The
queue is materialized from current online miners joined with historical
``miner_stats`` challenge state — there is no separate queue table.

Selection rules:
  - miner_stats.challenge_status is missing/"sampling"
  - is_valid == true
  - uid != champion_uid (the current champion is never its own challenger)
  - order by (first_block ASC, uid ASC)
"""

from dataclasses import dataclass
from typing import Optional, Protocol


# Status values read/written in miner_stats.challenge_status.
#
# Lifecycle:
#   sampling ─[claim_pending]→ in_progress
#   in_progress ─[mark_terminated(WON)]→  champion
#               ─[mark_terminated(LOST/FAILED)]→ terminated
#   champion ─[mark_terminated(LOST)]→ terminated
STATUS_SAMPLING = "sampling"
STATUS_IN_PROGRESS = "in_progress"
STATUS_CHAMPION = "champion"
STATUS_TERMINATED = "terminated"

# What mark_terminated callers pass in.
OUTCOME_WON = "won"
OUTCOME_LOST = "lost"
OUTCOME_FAILED = "failed"

_OUTCOME_TO_STATUS = {
    OUTCOME_WON: STATUS_CHAMPION,
    OUTCOME_LOST: STATUS_TERMINATED,
    OUTCOME_FAILED: STATUS_TERMINATED,
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
        self, uid: int, window_id: int, *, expected_status: str = STATUS_SAMPLING
    ) -> bool:
        """Atomically transition ``uid`` from ``expected_status`` to ``in_progress``.

        Returns True iff the transition was applied (i.e. no race lost).
        """
        ...

    async def set_terminal(
        self,
        uid: int,
        new_status: str,
        *,
        reason: str = "",
        hotkey: Optional[str] = None,
        revision: Optional[str] = None,
        model: str = "",
        scores_by_env: Optional[dict] = None,
        scores_refresh_block: Optional[int] = None,
        terminated_at_block: Optional[int] = None,
    ) -> None:
        """Set ``uid``'s historical ``challenge_status``."""
        ...


class ChallengerQueue:
    def __init__(self, store: MinerQueueStore):
        self._store = store

    async def pick_next(
        self, window_id: int, champion_uid: Optional[int]
    ) -> Optional[MinerCandidate]:
        """Return the earliest eligible challenger and mark it in_progress.

        A row is eligible if ``is_valid='true'`` and historical
        ``challenge_status`` is missing or ``'sampling'``.

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
            if status is not None and status != STATUS_SAMPLING:
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

    async def peek_next(
        self, n: int, *, champion_uid: Optional[int],
        exclude_uids: Optional[set] = None,
    ) -> list["MinerCandidate"]:
        """Return up to ``n`` next-up eligible candidates in pick order
        WITHOUT claiming any of them.

        Same filters and sort as :meth:`pick_next` so a candidate
        returned here is what ``pick_next`` will see next:
        ``is_valid=true``, ``challenge_status`` missing or
        ``'sampling'``, ``uid != champion_uid``, ordered by
        ``(first_block ASC, uid ASC)``. ``exclude_uids`` skips miners
        the caller already accounted for (e.g. the active
        ``current_battle.challenger`` and previously pre-deployed
        miners).

        Pure read — no DDB writes. Safe to call every scheduler tick
        from the pre-deploy decision path."""
        if n <= 0:
            return []
        skip = exclude_uids or set()
        candidates = await self._store.list_valid_pending()
        ordered = sorted(
            candidates,
            key=lambda m: (m.get("first_block", float("inf")), m.get("uid", 0)),
        )
        out: list[MinerCandidate] = []
        for row in ordered:
            uid = row.get("uid")
            if uid is None or uid == champion_uid or uid in skip:
                continue
            status = row.get("challenge_status")
            if status is not None and status != STATUS_SAMPLING:
                continue
            if not _is_truthy(row.get("is_valid")):
                continue
            out.append(MinerCandidate(
                uid=uid,
                hotkey=row["hotkey"],
                model=row.get("model", ""),
                revision=row.get("revision", ""),
                first_block=int(row.get("first_block", 0)),
            ))
            if len(out) >= n:
                break
        return out

    async def mark_terminated(
        self,
        uid: int,
        outcome: str,
        *,
        reason: Optional[str] = None,
        hotkey: Optional[str] = None,
        revision: Optional[str] = None,
        model: str = "",
        scores_by_env: Optional[dict] = None,
        scores_refresh_block: Optional[int] = None,
        terminated_at_block: Optional[int] = None,
    ) -> None:
        """Transition ``uid`` to a terminal challenge state."""
        try:
            new_status = _OUTCOME_TO_STATUS[outcome]
        except KeyError:
            raise ValueError(f"unknown outcome {outcome!r}") from None
        final_reason = reason or ""
        if outcome == OUTCOME_FAILED:
            final_reason = final_reason or "deployment_failed"
        elif outcome == OUTCOME_LOST:
            final_reason = final_reason or "lost"
        await self._store.set_terminal(
            uid,
            new_status,
            reason=final_reason,
            hotkey=hotkey,
            revision=revision,
            model=model,
            scores_by_env=scores_by_env,
            scores_refresh_block=scores_refresh_block,
            terminated_at_block=terminated_at_block,
        )


def _is_truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return False
