"""LiveScoresMonitor: per-challenger champion-overlap basis."""

from __future__ import annotations

from typing import Dict, List

import pytest

from affine.src.monitor.live_scores_monitor import LiveScoresMonitor
from affine.src.scorer.window_state import TaskIdState


class _FakeSamplesAdapter:
    def __init__(self, by_subject: Dict[tuple, Dict[int, float]]):
        self._by_subject = by_subject

    async def read_scores_for_tasks(
        self, hotkey: str, revision: str, env: str,
        task_ids: List[int], refresh_block: int,
    ) -> Dict[int, float]:
        bucket = self._by_subject.get((hotkey, revision, env), {})
        return {t: s for t, s in bucket.items() if t in task_ids}


def _make_monitor(samples_adapter) -> LiveScoresMonitor:
    monitor = LiveScoresMonitor.__new__(LiveScoresMonitor)
    monitor._samples = samples_adapter
    return monitor


@pytest.mark.asyncio
async def test_no_champion_omits_overlap_avg():
    samples = _FakeSamplesAdapter({
        ("hk1", "r1", "ENV_A"): {1: 1.0, 2: 0.0},
        ("hk2", "r2", "ENV_A"): {1: 0.5, 3: 0.5},
    })
    out = await _make_monitor(samples)._compute_scores(
        [{"uid": 1, "hotkey": "hk1", "revision": "r1"},
         {"uid": 2, "hotkey": "hk2", "revision": "r2"}],
        ["ENV_A"],
        TaskIdState(task_ids={"ENV_A": [1, 2, 3, 4]}, refreshed_at_block=10),
        champion_uid=None,
    )
    assert out["1"]["ENV_A"] == {"count": 2, "avg": 0.5}
    assert "champion_overlap_avg" not in out["2"]["ENV_A"]


@pytest.mark.asyncio
async def test_overlap_avg_uses_only_overlapping_tasks():
    """Two challengers with disjoint overlaps see DIFFERENT bases
    even though the champion's full-set avg is the same number — the
    whole point of this change."""
    samples = _FakeSamplesAdapter({
        ("champ_hk", "champ_rev", "ENV_A"): {1: 1.0, 2: 1.0, 3: 0.0, 4: 0.0},
        ("a_hk", "a_rev", "ENV_A"): {1: 0.5, 2: 0.5},
        ("b_hk", "b_rev", "ENV_A"): {3: 0.5, 4: 0.5},
    })
    out = await _make_monitor(samples)._compute_scores(
        [{"uid": 100, "hotkey": "champ_hk", "revision": "champ_rev"},
         {"uid": 1,   "hotkey": "a_hk",     "revision": "a_rev"},
         {"uid": 2,   "hotkey": "b_hk",     "revision": "b_rev"}],
        ["ENV_A"],
        TaskIdState(task_ids={"ENV_A": [1, 2, 3, 4]}, refreshed_at_block=10),
        champion_uid=100,
    )
    assert "champion_overlap_avg" not in out["100"]["ENV_A"]
    assert out["1"]["ENV_A"]["champion_overlap_avg"] == pytest.approx(1.0)
    assert out["2"]["ENV_A"]["champion_overlap_avg"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_empty_overlap_omits_avg():
    samples = _FakeSamplesAdapter({
        ("champ_hk", "champ_rev", "ENV_A"): {1: 1.0, 2: 1.0},
        ("a_hk", "a_rev", "ENV_A"): {3: 0.5, 4: 0.5},
    })
    out = await _make_monitor(samples)._compute_scores(
        [{"uid": 100, "hotkey": "champ_hk", "revision": "champ_rev"},
         {"uid": 1,   "hotkey": "a_hk",     "revision": "a_rev"}],
        ["ENV_A"],
        TaskIdState(task_ids={"ENV_A": [1, 2, 3, 4]}, refreshed_at_block=10),
        champion_uid=100,
    )
    assert "champion_overlap_avg" not in out["1"]["ENV_A"]


@pytest.mark.asyncio
async def test_champion_with_no_samples_skips_overlap():
    samples = _FakeSamplesAdapter({
        ("a_hk", "a_rev", "ENV_A"): {1: 0.5, 2: 0.5},
    })
    out = await _make_monitor(samples)._compute_scores(
        [{"uid": 100, "hotkey": "champ_hk", "revision": "champ_rev"},
         {"uid": 1,   "hotkey": "a_hk",     "revision": "a_rev"}],
        ["ENV_A"],
        TaskIdState(task_ids={"ENV_A": [1, 2, 3]}, refreshed_at_block=10),
        champion_uid=100,
    )
    assert "100" not in out
    assert "champion_overlap_avg" not in out["1"]["ENV_A"]


@pytest.mark.asyncio
async def test_empty_task_pool_returns_empty():
    out = await _make_monitor(_FakeSamplesAdapter({}))._compute_scores(
        [{"uid": 1, "hotkey": "hk", "revision": "r"}],
        ["ENV_A"],
        TaskIdState(task_ids={"ENV_A": []}, refreshed_at_block=10),
        champion_uid=None,
    )
    assert out == {}
