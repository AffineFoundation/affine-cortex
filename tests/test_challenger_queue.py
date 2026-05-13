"""Unit tests for ChallengerQueue."""

from typing import Optional

import pytest

from affine.src.scorer.challenger_queue import (
    ChallengerQueue,
    OUTCOME_FAILED,
    OUTCOME_LOST,
    OUTCOME_WON,
    STATUS_CHAMPION,
    STATUS_IN_PROGRESS,
    STATUS_PENDING,
    STATUS_TERMINATED_FAILED,
    STATUS_TERMINATED_LOST,
)


class InMemoryStore:
    """Simple substitute for MinersDAO used in tests."""

    def __init__(self, rows: list[dict]):
        # Index by uid for fast in-place mutation.
        self.rows: dict[int, dict] = {r["uid"]: dict(r) for r in rows}
        self.claim_calls: list[tuple[int, int]] = []
        self.terminal_calls: list[tuple[int, str]] = []
        # Simulate races: if uid is in this set, the first claim fails as if
        # another window beat us, then a subsequent claim would succeed.
        self.contended: set[int] = set()

    async def list_valid_pending(self) -> list[dict]:
        # In real DAO this is the GSI lookup of valid miners. We mimic that:
        # return everything that is_valid=true; the queue filters status.
        return [r for r in self.rows.values() if _truthy(r.get("is_valid"))]

    async def claim_pending(
        self, uid: int, window_id: int, *, expected_status: str = STATUS_PENDING
    ) -> bool:
        """Mirrors production: accept either attribute_not_exists OR matching expected_status."""
        self.claim_calls.append((uid, window_id))
        row = self.rows.get(uid)
        if row is None:
            return False
        current = row.get("challenge_status")
        if current is not None and current != expected_status:
            return False
        if uid in self.contended:
            self.contended.discard(uid)
            return False
        row["challenge_status"] = STATUS_IN_PROGRESS
        row["last_window_id"] = window_id
        return True

    async def set_terminal(self, uid: int, new_status: str) -> None:
        self.terminal_calls.append((uid, new_status))
        row = self.rows.setdefault(uid, {"uid": uid})
        row["challenge_status"] = new_status


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return False


def _miner(uid, hotkey, first_block, *, status=STATUS_PENDING, valid=True, revision="r1"):
    return {
        "uid": uid,
        "hotkey": hotkey,
        "model": f"org/{hotkey}",
        "revision": revision,
        "first_block": first_block,
        "is_valid": "true" if valid else "false",
        "challenge_status": status,
    }


@pytest.mark.asyncio
async def test_picks_earliest_first_block():
    store = InMemoryStore(
        [
            _miner(2, "B", first_block=200),
            _miner(1, "A", first_block=100),
            _miner(3, "C", first_block=300),
        ]
    )
    q = ChallengerQueue(store)
    cand = await q.pick_next(window_id=42, champion_uid=None)
    assert cand is not None
    assert cand.uid == 1
    assert store.rows[1]["challenge_status"] == STATUS_IN_PROGRESS
    assert store.rows[1]["last_window_id"] == 42


@pytest.mark.asyncio
async def test_uid_tiebreak_when_first_block_equal():
    store = InMemoryStore(
        [
            _miner(7, "G", first_block=100),
            _miner(3, "C", first_block=100),
            _miner(5, "E", first_block=100),
        ]
    )
    q = ChallengerQueue(store)
    cand = await q.pick_next(window_id=1, champion_uid=None)
    assert cand is not None and cand.uid == 3


@pytest.mark.asyncio
async def test_skips_champion_uid():
    store = InMemoryStore(
        [
            _miner(1, "A", first_block=100, status=STATUS_CHAMPION),
            _miner(2, "B", first_block=200, status=STATUS_PENDING),
        ]
    )
    q = ChallengerQueue(store)
    cand = await q.pick_next(window_id=1, champion_uid=1)
    assert cand is not None and cand.uid == 2


@pytest.mark.asyncio
async def test_skips_non_pending_status():
    store = InMemoryStore(
        [
            _miner(1, "A", first_block=100, status=STATUS_TERMINATED_LOST),
            _miner(2, "B", first_block=200, status=STATUS_CHAMPION),
            _miner(3, "C", first_block=300, status=STATUS_PENDING),
        ]
    )
    q = ChallengerQueue(store)
    cand = await q.pick_next(window_id=1, champion_uid=None)
    assert cand is not None and cand.uid == 3


@pytest.mark.asyncio
async def test_skips_invalid_miner():
    store = InMemoryStore(
        [
            _miner(1, "A", first_block=100, valid=False),
            _miner(2, "B", first_block=200, valid=True),
        ]
    )
    q = ChallengerQueue(store)
    cand = await q.pick_next(window_id=1, champion_uid=None)
    assert cand is not None and cand.uid == 2


@pytest.mark.asyncio
async def test_empty_queue_returns_none():
    store = InMemoryStore([])
    q = ChallengerQueue(store)
    cand = await q.pick_next(window_id=1, champion_uid=None)
    assert cand is None


@pytest.mark.asyncio
async def test_all_terminated_returns_none():
    store = InMemoryStore(
        [
            _miner(1, "A", first_block=100, status=STATUS_TERMINATED_LOST),
            _miner(2, "B", first_block=200, status=STATUS_CHAMPION),
        ]
    )
    q = ChallengerQueue(store)
    cand = await q.pick_next(window_id=1, champion_uid=2)
    assert cand is None


@pytest.mark.asyncio
async def test_lost_race_falls_through_to_next_candidate():
    store = InMemoryStore(
        [
            _miner(1, "A", first_block=100),
            _miner(2, "B", first_block=200),
        ]
    )
    # uid=1 has a contended flag → first claim returns False, queue should
    # fall through to uid=2.
    store.contended.add(1)
    q = ChallengerQueue(store)
    cand = await q.pick_next(window_id=1, champion_uid=None)
    assert cand is not None and cand.uid == 2
    # uid=1 was attempted, then released; nothing left for tests to assert
    # except that uid=1 is still pending (we treat a lost race as "someone
    # else got it" — the real DAO would reflect that downstream).
    assert (1, 1) in store.claim_calls
    assert (2, 1) in store.claim_calls


@pytest.mark.asyncio
async def test_mark_terminated_won():
    store = InMemoryStore([_miner(1, "A", 100, status=STATUS_IN_PROGRESS)])
    q = ChallengerQueue(store)
    await q.mark_terminated(1, OUTCOME_WON)
    assert store.rows[1]["challenge_status"] == STATUS_CHAMPION
    assert store.terminal_calls == [(1, STATUS_CHAMPION)]


@pytest.mark.asyncio
async def test_mark_terminated_lost_and_failed():
    store = InMemoryStore(
        [
            _miner(1, "A", 100, status=STATUS_IN_PROGRESS),
            _miner(2, "B", 200, status=STATUS_IN_PROGRESS),
        ]
    )
    q = ChallengerQueue(store)
    await q.mark_terminated(1, OUTCOME_LOST)
    await q.mark_terminated(2, OUTCOME_FAILED)
    assert store.rows[1]["challenge_status"] == STATUS_TERMINATED_LOST
    assert store.rows[2]["challenge_status"] == STATUS_TERMINATED_FAILED


@pytest.mark.asyncio
async def test_mark_terminated_unknown_outcome_raises():
    store = InMemoryStore([_miner(1, "A", 100)])
    q = ChallengerQueue(store)
    with pytest.raises(ValueError):
        await q.mark_terminated(1, "lol")


@pytest.mark.asyncio
async def test_pick_marks_status_in_db():
    store = InMemoryStore([_miner(7, "G", 100)])
    q = ChallengerQueue(store)
    assert store.rows[7]["challenge_status"] == STATUS_PENDING
    cand = await q.pick_next(window_id=99, champion_uid=None)
    assert cand is not None and cand.uid == 7
    assert store.rows[7]["challenge_status"] == STATUS_IN_PROGRESS
    assert store.rows[7]["last_window_id"] == 99


@pytest.mark.asyncio
async def test_picks_miner_missing_challenge_status():
    """A brand-new commit may land in miners table before the monitor seeds
    challenge_status. The queue should still claim such rows — the
    production adapter's UpdateItem condition tolerates the absent attr."""
    row = _miner(5, "F", first_block=50)
    row.pop("challenge_status")  # simulate pre-seed state
    store = InMemoryStore([row])
    q = ChallengerQueue(store)
    cand = await q.pick_next(window_id=1, champion_uid=None)
    assert cand is not None and cand.uid == 5
    # Claim flipped status to in_progress.
    assert store.rows[5]["challenge_status"] == STATUS_IN_PROGRESS
