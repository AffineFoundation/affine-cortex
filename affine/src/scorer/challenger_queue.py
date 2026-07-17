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

import time
from dataclasses import dataclass
from typing import Optional, Protocol


# Status values read/written in miner_stats.challenge_status.
#
# Lifecycle:
#   sampling ─[claim_pending]→ in_progress
#   in_progress ─[release_claim]→ sampling
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
    model_type: str = ""


class MinerQueueStore(Protocol):
    """Storage operations the queue needs.

    In production this wraps :class:`affine.database.dao.miners.MinersDAO`.
    In tests an in-memory implementation is used.
    """

    async def list_valid_pending(self) -> list[dict]: ...

    async def claim_pending(
        self,
        uid: int,
        window_id: int,
        *,
        expected_status: str = STATUS_SAMPLING,
        admission_policy_identity: Optional[str] = None,
    ) -> bool:
        """Atomically transition ``uid`` from ``expected_status`` to ``in_progress``.

        Returns True iff the transition was applied (i.e. no race lost).
        """
        ...

    async def defer_admission(
        self,
        uid: int,
        *,
        hotkey: str,
        revision: str,
        policy_identity: str,
        deployment_generation: str,
        base_delay_seconds: int,
        max_attempts: int,
    ) -> dict:
        """Idempotently requeue an expired deployment with durable backoff."""
        ...

    async def release_claim(
        self, uid: int, *,
        hotkey: Optional[str] = None,
        revision: Optional[str] = None,
    ) -> bool:
        """Atomically revert ``uid`` from ``in_progress`` to
        ``sampling``. Used when a deploy hits a transport failure and
        the miner must remain re-pickable. Returns True iff the row
        was actually in_progress at the call site (else: race lost,
        someone already terminated it)."""
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
        opponent_scores_by_env: Optional[dict] = None,
        battle_task_ids: Optional[dict] = None,
        scores_refresh_block: Optional[int] = None,
        terminated_at_block: Optional[int] = None,
    ) -> None:
        """Set ``uid``'s historical ``challenge_status``."""
        ...

    async def list_in_progress(self) -> list[dict]:
        """Return all rows currently at ``challenge_status='in_progress'``."""
        ...


class ChallengerQueue:
    def __init__(self, store: MinerQueueStore):
        self._store = store

    async def pick_next(
        self,
        window_id: int,
        champion_uid: Optional[int],
        *,
        admission_policy_identity: Optional[str] = None,
        now: Optional[int] = None,
    ) -> Optional[MinerCandidate]:
        """Return the earliest eligible challenger and mark it in_progress.

        A row is eligible if ``is_valid='true'`` and historical
        ``challenge_status`` is missing or ``'sampling'``.

        Returns ``None`` if no eligible miner exists. On a lost race for
        the same candidate, the conditional write returns False and we
        walk on to the next row.
        """
        timestamp = int(time.time()) if now is None else int(now)
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
            if _admission_deferred(
                row,
                policy_identity=admission_policy_identity,
                now=timestamp,
            ):
                continue
            # Keep the queue protocol backwards-compatible for callers that do
            # not enable admission gating.  More importantly, this avoids
            # requiring unrelated stores to understand admission policy
            # metadata when there is no policy to enforce.
            if admission_policy_identity is None:
                claimed = await self._store.claim_pending(uid, window_id)
            else:
                claimed = await self._store.claim_pending(
                    uid,
                    window_id,
                    admission_policy_identity=admission_policy_identity,
                )
            if claimed:
                return MinerCandidate(
                    uid=uid,
                    hotkey=row["hotkey"],
                    model=row.get("model", ""),
                    revision=row.get("revision", ""),
                    first_block=int(row.get("first_block", 0)),
                    model_type=str(row.get("model_type") or ""),
                )
        return None

    async def peek_next(
        self, n: int, *, champion_uid: Optional[int],
        exclude_uids: Optional[set] = None,
        admission_policy_identity: Optional[str] = None,
        now: Optional[int] = None,
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
        timestamp = int(time.time()) if now is None else int(now)
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
            if _admission_deferred(
                row,
                policy_identity=admission_policy_identity,
                now=timestamp,
            ):
                continue
            out.append(MinerCandidate(
                uid=uid,
                hotkey=row["hotkey"],
                model=row.get("model", ""),
                revision=row.get("revision", ""),
                first_block=int(row.get("first_block", 0)),
                model_type=str(row.get("model_type") or ""),
            ))
            if len(out) >= n:
                break
        return out

    async def defer_admission(
        self,
        uid: int,
        *,
        hotkey: str,
        revision: str,
        policy_identity: str,
        deployment_generation: str,
        base_delay_seconds: int,
        max_attempts: int,
    ) -> dict:
        return await self._store.defer_admission(
            uid,
            hotkey=hotkey,
            revision=revision,
            policy_identity=policy_identity,
            deployment_generation=deployment_generation,
            base_delay_seconds=base_delay_seconds,
            max_attempts=max_attempts,
        )

    async def release_claim(
        self, uid: int, *,
        hotkey: Optional[str] = None,
        revision: Optional[str] = None,
    ) -> bool:
        """Atomically revert ``uid`` from ``in_progress`` to
        ``sampling`` after a transport-level deploy failure. Returns
        True iff the row was in_progress (else: race already
        terminated it, no action needed)."""
        return await self._store.release_claim(
            uid, hotkey=hotkey, revision=revision,
        )

    async def list_in_progress(self) -> list[dict]:
        return await self._store.list_in_progress()

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
        opponent_scores_by_env: Optional[dict] = None,
        battle_task_ids: Optional[dict] = None,
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
            opponent_scores_by_env=opponent_scores_by_env,
            battle_task_ids=battle_task_ids,
            scores_refresh_block=scores_refresh_block,
            terminated_at_block=terminated_at_block,
        )


def _is_truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return False


def _admission_deferred(
    row: dict,
    *,
    policy_identity: Optional[str],
    now: int,
) -> bool:
    """Filter only the current policy's cooldown; a policy rollout resets it."""

    policy = str(policy_identity or "")
    if not policy or str(row.get("admission_policy_identity") or "") != policy:
        return False
    if row.get("admission_deferral_exhausted") is True:
        return True
    try:
        return int(row.get("admission_retry_after") or 0) > int(now)
    except (TypeError, ValueError, OverflowError):
        # Malformed control data must not silently create a permanent hold.
        return False
