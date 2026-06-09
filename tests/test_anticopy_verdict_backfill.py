"""VerdictBackfillService — runs in the anticopy-refresh container.

Pins the contract that:
  - Verdict math is no longer the GPU-side worker's job; the backfill
    service computes it from cached R2 score blobs.
  - One R2 fetch per peer per tick (cache is reused across every
    pending candidate within that tick).
  - Pending rows (no ``verdict_at``) get their verdict written; rows
    with verdict_at populated are left alone.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest

from affine.src.anticopy.threshold import AntiCopyConfig
from affine.src.anticopy.verdict import (
    VerdictBackfillService,
    _pick_origin,
)


# ---------- helpers --------------------------------------------------


def _blob(hk: str, rev: str, first_block: int, lps, *, model=None) -> Dict[str, Any]:
    return {
        "schema": "ceac.score/v2",
        "hotkey": hk, "revision": rev,
        "model": model or f"org/m-{hk}",
        "first_block": first_block,
        "per_rollout": [{
            "rollout_key": "champ#CDE#1", "env": "CDE",
            "resp_lp": list(lps),
        }],
    }


def _row(hk, rev, first_block, *, verdict_at=None):
    r = {
        "hotkey": hk, "revision": rev,
        "r2_key": f"{hk}/{rev}.json.gz",
        "first_block": first_block,
    }
    if verdict_at is not None:
        r["verdict_at"] = verdict_at
    return r


class _FakeR2:
    def __init__(self, blobs):
        self.blobs = blobs
        self.fetch_log: List[str] = []

    def get_score_by_key(self, key):
        self.fetch_log.append(key)
        return self.blobs.get(key)


class _FakeScoresDAO:
    def __init__(self, rows):
        self.rows = list(rows)
        self.updates: List[Tuple[str, str, Dict[str, Any]]] = []

    async def list_all(self):
        return list(self.rows)

    async def update_verdict(self, hotkey, revision, **kwargs):
        self.updates.append((hotkey, revision, kwargs))


class _FakeConfigDAO:
    def __init__(self, cfg_dict):
        self._cfg = cfg_dict

    async def get_param_value(self, key, default=None):
        if key == "anticopy":
            return self._cfg
        return default


# ---------- _pick_origin (pure logic) -------------------------------


def test_pick_origin_earliest_committer_wins():
    cfg = AntiCopyConfig(
        enabled=True, nll_threshold=0.05, min_overlap=10,
        agreement_ratio=0.5, verdict_lookback_days=0,
    )
    ref = _blob("Z", "rz", 1000, [-0.7] * 60)
    peers = {
        ("A", "ra"): _blob("A", "ra", 200, [-0.7] * 60),   # earliest
        ("B", "rb"): _blob("B", "rb", 500, [-0.7] * 60),
        ("C", "rc"): _blob("C", "rc", 800, [-0.7] * 60),
    }
    dec = _pick_origin(
        cfg=cfg, ref_score=ref, ref_first_block=1000, peer_cache=peers,
    )
    assert dec.copy_of_hotkey == "A"
    assert dec.decision_median == 0.0


def test_pick_origin_independent_records_closest():
    cfg = AntiCopyConfig(
        enabled=True, nll_threshold=0.005, min_overlap=10,
        agreement_ratio=0.5, verdict_lookback_days=0,
    )
    ref = _blob("Z", "rz", 1000, [-0.7] * 60)
    peers = {
        ("A", "ra"): _blob("A", "ra", 200, [-2.0] * 60),
        ("B", "rb"): _blob("B", "rb", 500, [-1.5] * 60),
    }
    dec = _pick_origin(
        cfg=cfg, ref_score=ref, ref_first_block=1000, peer_cache=peers,
    )
    assert dec.copy_of_hotkey == ""
    # closest = whichever peer had smallest dec_med
    assert dec.closest_peer_model.startswith("org/m-")


def test_pick_origin_lookback_excludes_ancient():
    cfg = AntiCopyConfig(
        enabled=True, nll_threshold=0.05, min_overlap=10,
        agreement_ratio=0.5, verdict_lookback_days=1,  # 7200 blocks
    )
    ref = _blob("Z", "rz", 100_000, [-0.7] * 60)
    peers = {
        ("A", "ra"): _blob("A", "ra", 50_000, [-0.7] * 60),  # too old
        ("B", "rb"): _blob("B", "rb", 95_000, [-0.7] * 60),  # inside window
    }
    dec = _pick_origin(
        cfg=cfg, ref_score=ref, ref_first_block=100_000, peer_cache=peers,
    )
    assert dec.copy_of_hotkey == "B"


# ---------- VerdictBackfillService.tick -----------------------------


@pytest.mark.asyncio
async def test_backfill_only_processes_pending_rows():
    """Rows with ``verdict_at`` populated are skipped; rows without
    get their verdict written. Peer cache is built once."""
    blobs = {
        "Z/rz.json.gz": _blob("Z", "rz", 1000, [-0.7] * 60),
        "A/ra.json.gz": _blob("A", "ra", 200, [-0.7] * 60),
        "B/rb.json.gz": _blob("B", "rb", 500, [-0.7] * 60),
        "D/rd.json.gz": _blob("D", "rd", 1500, [-0.7] * 60),  # already verdicted
    }
    rows = [
        _row("Z", "rz", 1000),                              # pending
        _row("A", "ra", 200),                               # pending (will be first committer → indep)
        _row("B", "rb", 500),                               # pending
        _row("D", "rd", 1500, verdict_at=1700000000),       # done — skipped
    ]
    r2 = _FakeR2(blobs)
    scores = _FakeScoresDAO(rows)
    cfg_dao = _FakeConfigDAO({
        "enabled": True, "nll_threshold": 0.05, "min_overlap": 10,
        "agreement_ratio": 0.5, "verdict_lookback_days": 0,
    })
    svc = VerdictBackfillService(
        scores_dao=scores, config_dao=cfg_dao, r2=r2,
    )
    await svc._tick()

    # Three pending rows got an update_verdict call; the verdicted row didn't.
    updated_hks = sorted(hk for hk, _, _ in scores.updates)
    assert updated_hks == ["A", "B", "Z"]

    # Verdicts: A is earliest → independent; B's earlier peer is A → copy_of A;
    # Z's earliest below-threshold peer is A → copy_of A.
    by_hk = {hk: kw for hk, _, kw in scores.updates}
    assert by_hk["A"]["copy_of"] == ""
    assert by_hk["B"]["copy_of"] == "A"
    assert by_hk["Z"]["copy_of"] == "A"

    # Peer cache loaded each blob exactly once (4 fetches for 4 rows).
    assert sorted(r2.fetch_log) == sorted(blobs.keys())


@pytest.mark.asyncio
async def test_backfill_noop_when_disabled():
    cfg_dao = _FakeConfigDAO({"enabled": False})
    scores = _FakeScoresDAO([_row("A", "ra", 100)])
    r2 = _FakeR2({})
    svc = VerdictBackfillService(
        scores_dao=scores, config_dao=cfg_dao, r2=r2,
    )
    await svc._tick()
    assert scores.updates == []
    assert r2.fetch_log == []


@pytest.mark.asyncio
async def test_backfill_noop_when_no_pending_rows():
    """All rows already have verdict_at → no work, no R2 fetches."""
    cfg_dao = _FakeConfigDAO({
        "enabled": True, "nll_threshold": 0.05, "min_overlap": 10,
        "agreement_ratio": 0.5, "verdict_lookback_days": 0,
    })
    rows = [
        _row("A", "ra", 100, verdict_at=1700000000),
        _row("B", "rb", 200, verdict_at=1700000001),
    ]
    r2 = _FakeR2({})
    scores = _FakeScoresDAO(rows)
    svc = VerdictBackfillService(
        scores_dao=scores, config_dao=cfg_dao, r2=r2,
    )
    await svc._tick()
    assert scores.updates == []
    assert r2.fetch_log == []


@pytest.mark.asyncio
async def test_peer_cache_skips_rows_outside_lookback_window():
    """Peers whose ``first_block`` is older than the earliest pending
    candidate's lookback window are never fetched from R2 — they can't
    influence any verdict this tick anyway. Memory + latency win since
    ``scores_index`` has no TTL and accumulates dereg history.
    """
    # lookback_days=1 → 7200 blocks. Pending candidate at fb=120000 →
    # peer_fb_floor = 120000 - 7200 = 112800. Anything below that is
    # skipped at the cache-build step.
    blobs = {
        "ANCIENT/r0.json.gz":  _blob("ANCIENT",  "r0",  10_000, [-0.7] * 60),
        "OLD/r1.json.gz":      _blob("OLD",      "r1", 100_000, [-0.7] * 60),
        "EDGE/r2.json.gz":     _blob("EDGE",     "r2", 112_800, [-0.7] * 60),
        "FRESH/r3.json.gz":    _blob("FRESH",    "r3", 115_000, [-0.7] * 60),
        "PENDING/r4.json.gz":  _blob("PENDING",  "r4", 120_000, [-0.7] * 60),
    }
    rows = [
        _row("ANCIENT",  "r0",  10_000, verdict_at=1700000000),  # done, far outside window
        _row("OLD",      "r1", 100_000, verdict_at=1700000001),  # done, outside
        _row("EDGE",     "r2", 112_800, verdict_at=1700000002),  # done, on boundary
        _row("FRESH",    "r3", 115_000, verdict_at=1700000003),  # done, inside
        _row("PENDING",  "r4", 120_000),                          # pending
    ]
    r2 = _FakeR2(blobs)
    scores = _FakeScoresDAO(rows)
    cfg_dao = _FakeConfigDAO({
        "enabled": True, "nll_threshold": 0.05, "min_overlap": 10,
        "agreement_ratio": 0.5, "verdict_lookback_days": 1,
    })
    svc = VerdictBackfillService(
        scores_dao=scores, config_dao=cfg_dao, r2=r2,
    )
    await svc._tick()

    fetched = set(r2.fetch_log)
    assert "ANCIENT/r0.json.gz" not in fetched
    assert "OLD/r1.json.gz" not in fetched
    assert "EDGE/r2.json.gz" in fetched
    assert "FRESH/r3.json.gz" in fetched
    assert "PENDING/r4.json.gz" in fetched


@pytest.mark.asyncio
async def test_peer_cache_lookback_zero_disables_filter():
    """``verdict_lookback_days=0`` means no time-based pruning — every
    row's blob is loaded. Keeps backward compatibility with the
    original behaviour expected by older fixtures in this file.
    """
    blobs = {
        "VERY_OLD/r0.json.gz": _blob("VERY_OLD", "r0",   1_000, [-0.7] * 60),
        "PENDING/r1.json.gz":  _blob("PENDING",  "r1", 500_000, [-0.7] * 60),
    }
    rows = [
        _row("VERY_OLD", "r0",   1_000, verdict_at=1700000000),
        _row("PENDING",  "r1", 500_000),
    ]
    r2 = _FakeR2(blobs)
    scores = _FakeScoresDAO(rows)
    cfg_dao = _FakeConfigDAO({
        "enabled": True, "nll_threshold": 0.05, "min_overlap": 10,
        "agreement_ratio": 0.5, "verdict_lookback_days": 0,
    })
    svc = VerdictBackfillService(
        scores_dao=scores, config_dao=cfg_dao, r2=r2,
    )
    await svc._tick()

    assert sorted(r2.fetch_log) == sorted(blobs.keys())
