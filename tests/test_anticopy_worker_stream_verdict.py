"""Streaming verdict path — ``_stream_pick_origin`` holds at most one
peer R2 blob in memory at a time and early-exits on the earliest
threshold-crossing peer.

These tests pin the contract that the verdict-time memory footprint
is bounded regardless of how many peers exist in scores_index — the
previous in-memory ``peer_scores`` list OOM'd the worker's cgroup at
~90 peers.
"""

from __future__ import annotations

import pytest

from affine.src.anticopy.threshold import AntiCopyConfig
from affine.src.anticopy.worker import ForwardWorker


def _make_blob(hk, rev, first_block, lps):
    return {
        "schema": "ceac.score/v2",
        "hotkey": hk,
        "revision": rev,
        "model": f"org/m-{hk}",
        "first_block": first_block,
        "per_rollout": [{
            "rollout_key": f"champ#CDE#1",
            "env": "CDE",
            "resp_lp": list(lps),
            "resp_top": [],
        }],
    }


def _row(hk, rev, first_block):
    return {
        "hotkey": hk, "revision": rev,
        "r2_key": f"{hk}/{rev}.json.gz",
        "first_block": first_block,
    }


class _FakeR2:
    """Per-blob fetch counter so we can assert streaming, not bulk-load."""

    def __init__(self, blobs_by_key):
        self.blobs = blobs_by_key
        self.fetch_count = 0
        self.fetch_log = []

    def get_score_by_key(self, key):
        self.fetch_count += 1
        self.fetch_log.append(key)
        return self.blobs.get(key)


def _build_worker(r2):
    w = ForwardWorker(
        rollouts_dao=object(),
        scores_dao=object(),
        miners_dao=object(),
        miner_stats_dao=object(),
        config_dao=object(),
    )
    w.r2 = r2
    return w


@pytest.mark.asyncio
async def test_stream_picks_earliest_threshold_crosser_and_early_exits():
    """Three earlier peers, all identical to the candidate. The
    earliest-committed one wins; the stream stops there."""
    cfg = AntiCopyConfig(
        enabled=True, nll_threshold=0.05, min_overlap=10,
        agreement_ratio=0.5, verdict_lookback_days=0,
    )
    new_score = _make_blob("Z", "rz", first_block=1000, lps=[-0.7] * 60)

    rows = [
        _row("A", "ra", 200),   # earliest
        _row("B", "rb", 500),
        _row("C", "rc", 800),
    ]
    blobs = {
        "A/ra.json.gz": _make_blob("A", "ra", 200, [-0.7] * 60),
        "B/rb.json.gz": _make_blob("B", "rb", 500, [-0.7] * 60),
        "C/rc.json.gz": _make_blob("C", "rc", 800, [-0.7] * 60),
    }
    r2 = _FakeR2(blobs)
    worker = _build_worker(r2)

    decision = await worker._stream_pick_origin(
        cfg=cfg, ref_score=new_score, ref_first_block=1000,
        candidate_rows=rows,
    )
    assert decision.copy_of_hotkey == "A"   # earliest committer wins
    # Streaming + early-exit: only the earliest peer's blob was fetched.
    assert r2.fetch_count == 1
    assert r2.fetch_log == ["A/ra.json.gz"]


@pytest.mark.asyncio
async def test_stream_independent_when_no_peer_below_threshold():
    """No peer crosses threshold → independent verdict, closest peer
    info still surfaces, all peer blobs were fetched (no early-exit)."""
    cfg = AntiCopyConfig(
        enabled=True, nll_threshold=0.005, min_overlap=10,
        agreement_ratio=0.5, verdict_lookback_days=0,
    )
    new_score = _make_blob("Z", "rz", first_block=1000, lps=[-0.7] * 60)
    rows = [
        _row("A", "ra", 200),
        _row("B", "rb", 500),
    ]
    blobs = {
        "A/ra.json.gz": _make_blob("A", "ra", 200, [-2.0] * 60),  # very different
        "B/rb.json.gz": _make_blob("B", "rb", 500, [-1.5] * 60),  # also far
    }
    r2 = _FakeR2(blobs)
    worker = _build_worker(r2)

    decision = await worker._stream_pick_origin(
        cfg=cfg, ref_score=new_score, ref_first_block=1000,
        candidate_rows=rows,
    )
    assert decision.copy_of_hotkey == ""    # nobody crossed
    # closest_peer_model is whichever pair had smallest dec_med
    assert decision.closest_peer_model.startswith("org/m-")
    assert r2.fetch_count == 2              # scanned everything


@pytest.mark.asyncio
async def test_stream_lookback_skips_ancient_without_fetch():
    """Peer outside lookback window → not even fetched (saves I/O)."""
    cfg = AntiCopyConfig(
        enabled=True, nll_threshold=0.05, min_overlap=10,
        agreement_ratio=0.5, verdict_lookback_days=1,  # = 7200 blocks
    )
    new_score = _make_blob("Z", "rz", first_block=100_000, lps=[-0.7] * 60)
    rows = [
        _row("A", "ra", 50_000),    # 50k blocks earlier → way outside 1-day window
        _row("B", "rb", 95_000),    # 5k blocks earlier → inside
    ]
    blobs = {
        "A/ra.json.gz": _make_blob("A", "ra", 50_000, [-0.7] * 60),
        "B/rb.json.gz": _make_blob("B", "rb", 95_000, [-0.7] * 60),
    }
    r2 = _FakeR2(blobs)
    worker = _build_worker(r2)

    decision = await worker._stream_pick_origin(
        cfg=cfg, ref_score=new_score, ref_first_block=100_000,
        candidate_rows=rows,
    )
    assert decision.copy_of_hotkey == "B"
    # Ancient peer was never fetched.
    assert r2.fetch_log == ["B/rb.json.gz"]


@pytest.mark.asyncio
async def test_stream_skips_later_peers():
    """Peers with first_block >= candidate cannot be origin → not fetched."""
    cfg = AntiCopyConfig(
        enabled=True, nll_threshold=0.05, min_overlap=10,
        agreement_ratio=0.5, verdict_lookback_days=0,
    )
    new_score = _make_blob("Z", "rz", first_block=500, lps=[-0.7] * 60)
    rows = [
        _row("L", "rl", 900),   # later
    ]
    blobs = {"L/rl.json.gz": _make_blob("L", "rl", 900, [-0.7] * 60)}
    r2 = _FakeR2(blobs)
    worker = _build_worker(r2)

    decision = await worker._stream_pick_origin(
        cfg=cfg, ref_score=new_score, ref_first_block=500,
        candidate_rows=rows,
    )
    assert decision.copy_of_hotkey == ""
    assert r2.fetch_count == 0              # later peer never fetched


@pytest.mark.asyncio
async def test_stream_strips_resp_top_on_fetch():
    """Peer blobs carry the diagnostic ``resp_top`` matrix on disk;
    the streaming fetch must drop it before the blob is held in
    memory (10× memory saving). _fetch_peer_blob handles this."""
    big_top = [[[float(-0.1 * j), j] for j in range(5)] for _ in range(60)]
    blob_on_disk = _make_blob("A", "ra", 200, [-0.7] * 60)
    blob_on_disk["per_rollout"][0]["resp_top"] = big_top
    r2 = _FakeR2({"A/ra.json.gz": blob_on_disk})
    worker = _build_worker(r2)

    fetched = await worker._fetch_peer_blob(_row("A", "ra", 200))
    assert fetched is not None
    for ro in fetched["per_rollout"]:
        assert "resp_top" not in ro
