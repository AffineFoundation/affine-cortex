"""``ForwardWorker._get_next_candidates`` — pull-based candidate
discovery.

The worker no longer reads from a jobs queue. Every iteration it
scans ``miners`` + ``miner_stats`` + ``scores_index`` itself, applies
the same active-bucket filter as ``af get-rank`` (valid + non-
terminated + unscored), and returns the next candidate by commit
time. These tests pin that contract.
"""

from __future__ import annotations

import pytest

from affine.src.anticopy.worker import ForwardWorker


class _FakeMinersDAO:
    def __init__(self, rows):
        self._rows = rows

    async def get_valid_miners(self):
        return list(self._rows)


class _FakeStatsDAO:
    def __init__(self, terminated_pairs=None):
        self._terminated = terminated_pairs or set()

    async def get_miner_stats(self, hotkey, revision):
        if (hotkey, revision) in self._terminated:
            return {"challenge_status": "terminated"}
        return None


class _FakeScoresDAO:
    """``scored`` = verdict fully written (row + ``verdict_at``).
    ``half_done`` = score blob persisted but ``update_verdict`` never
    ran (older-worker crash). Half-done rows must still surface as
    candidates so the worker can re-run verdict against the current
    peer set."""

    def __init__(self, scored_pairs=None, half_done_pairs=None):
        self._scored = scored_pairs or set()
        self._half_done = half_done_pairs or set()

    async def get_score(self, hotkey, revision):
        if (hotkey, revision) in self._scored:
            return {
                "hotkey": hotkey, "revision": revision,
                "verdict_at": 1234567890,
            }
        if (hotkey, revision) in self._half_done:
            return {"hotkey": hotkey, "revision": revision}
        return None


def _build_worker(*, miners, terminated=None, scored=None, half_done=None):
    return ForwardWorker(
        rollouts_dao=object(),
        scores_dao=_FakeScoresDAO(scored or set(), half_done or set()),
        miners_dao=_FakeMinersDAO(miners),
        miner_stats_dao=_FakeStatsDAO(terminated or set()),
        config_dao=object(),
    )


def _miner(uid, hk, rev, first_block):
    return {
        "uid": uid, "hotkey": hk, "revision": rev,
        "first_block": first_block, "is_valid": True,
        "model": f"org/m-{hk}",
    }


@pytest.mark.asyncio
async def test_orders_by_first_block_ascending_then_uid():
    """``af get-rank``-style ordering: earliest committer first, uid
    breaks ties."""
    worker = _build_worker(miners=[
        _miner(uid=5, hk="hk_late", rev="r_late", first_block=8500),
        _miner(uid=1, hk="hk_early", rev="r_early", first_block=8000),
        _miner(uid=2, hk="hk_mid", rev="r_mid", first_block=8200),
    ])
    rows = await worker._get_next_candidates(limit=3)
    assert [r["uid"] for r in rows] == [1, 2, 5]


@pytest.mark.asyncio
async def test_skips_terminated_miners():
    worker = _build_worker(
        miners=[
            _miner(uid=1, hk="hk_a", rev="r_a", first_block=8000),
            _miner(uid=2, hk="hk_term", rev="r_term", first_block=8200),
            _miner(uid=3, hk="hk_b", rev="r_b", first_block=8400),
        ],
        terminated={("hk_term", "r_term")},
    )
    rows = await worker._get_next_candidates(limit=3)
    assert [r["uid"] for r in rows] == [1, 3]


@pytest.mark.asyncio
async def test_skips_already_scored_miners():
    """scores_index is the durable done marker."""
    worker = _build_worker(
        miners=[
            _miner(uid=1, hk="hk_done", rev="r_done", first_block=8000),
            _miner(uid=2, hk="hk_todo", rev="r_todo", first_block=8200),
        ],
        scored={("hk_done", "r_done")},
    )
    rows = await worker._get_next_candidates(limit=3)
    assert [r["uid"] for r in rows] == [2]


@pytest.mark.asyncio
async def test_half_done_rows_resurface_for_verdict_only():
    """Score blob persisted but ``verdict_at`` empty (older-worker
    crash between upsert and update_verdict). Such rows must come back
    so ``_run_job`` can take the verdict-only fast path."""
    worker = _build_worker(
        miners=[
            _miner(uid=1, hk="hk_full",  rev="r_full",  first_block=8000),
            _miner(uid=2, hk="hk_half",  rev="r_half",  first_block=8200),
            _miner(uid=3, hk="hk_fresh", rev="r_fresh", first_block=8400),
        ],
        scored={("hk_full", "r_full")},
        half_done={("hk_half", "r_half")},
    )
    rows = await worker._get_next_candidates(limit=3)
    assert [r["uid"] for r in rows] == [2, 3]


@pytest.mark.asyncio
async def test_exclude_filters_inflight_candidate():
    """The prefetcher passes ``exclude={(in_flight_hotkey, rev)}`` to
    avoid re-downloading the currently-running candidate."""
    worker = _build_worker(miners=[
        _miner(uid=1, hk="hk_now", rev="r_now", first_block=8000),
        _miner(uid=2, hk="hk_next", rev="r_next", first_block=8200),
        _miner(uid=3, hk="hk_later", rev="r_later", first_block=8400),
    ])
    rows = await worker._get_next_candidates(
        limit=2, exclude={("hk_now", "r_now")},
    )
    assert [r["uid"] for r in rows] == [2, 3]


@pytest.mark.asyncio
async def test_validator_slot_uid_zero_skipped():
    worker = _build_worker(miners=[
        _miner(uid=0, hk="validator", rev="r0", first_block=7000),
        _miner(uid=1, hk="hk_a", rev="r_a", first_block=8000),
    ])
    rows = await worker._get_next_candidates(limit=3)
    assert [r["uid"] for r in rows] == [1]


@pytest.mark.asyncio
async def test_empty_when_nothing_pending():
    worker = _build_worker(
        miners=[_miner(uid=1, hk="hk", rev="r", first_block=8000)],
        scored={("hk", "r")},
    )
    rows = await worker._get_next_candidates(limit=3)
    assert rows == []
    cand = await worker._get_next_candidate()
    assert cand is None
