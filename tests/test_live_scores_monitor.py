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


class _CapturingStatsDAO:
    """Records ``update_live_scores`` calls without touching DDB."""
    def __init__(self):
        self.calls = []

    async def update_live_scores(
        self, *, hotkey, revision, scores_by_env, scores_refresh_block,
    ):
        self.calls.append({
            "hotkey": hotkey,
            "revision": revision,
            "scores_by_env": scores_by_env,
            "scores_refresh_block": scores_refresh_block,
        })


class _FakeMinersDAOForMonitor:
    def __init__(self, miners):
        self._miners = miners

    async def get_valid_miners(self):
        return list(self._miners)


class _FakeStateStore:
    def __init__(self, task_state, envs, champion_uid):
        self._task_state = task_state
        self._envs = envs
        self._champion_uid = champion_uid

    async def get_task_state(self):
        return self._task_state

    async def get_scoring_environments(self):
        return self._envs

    async def get_champion(self):
        if self._champion_uid is None:
            return None
        from collections import namedtuple
        Champ = namedtuple("Champ", ["uid"])
        return Champ(uid=self._champion_uid)


@pytest.mark.asyncio
async def test_refresh_once_persists_per_miner_to_miner_stats():
    """One refresh cycle calls ``MinerStatsDAO.update_live_scores`` for
    every valid miner that had samples this refresh_block. The cache no
    longer lives in ``system_config`` — every per-miner row carries its
    own live aggregates and refresh_block stamp."""
    samples = _FakeSamplesAdapter({
        ("champ_hk", "champ_rev", "ENV_A"): {1: 1.0, 2: 0.0},
        ("chal_hk", "chal_rev", "ENV_A"): {1: 0.5, 2: 0.5},
    })
    stats_dao = _CapturingStatsDAO()
    monitor = LiveScoresMonitor.__new__(LiveScoresMonitor)
    monitor._samples = samples
    monitor._stats_dao = stats_dao
    monitor._miners_dao = _FakeMinersDAOForMonitor([
        {"uid": 100, "hotkey": "champ_hk", "revision": "champ_rev", "is_valid": "true"},
        {"uid": 1,   "hotkey": "chal_hk",  "revision": "chal_rev",  "is_valid": "true"},
    ])
    monitor._state = _FakeStateStore(
        TaskIdState(task_ids={"ENV_A": [1, 2]}, refreshed_at_block=42),
        ["ENV_A"], champion_uid=100,
    )

    n = await monitor.refresh_once()

    assert n == 2
    by_hk = {c["hotkey"]: c for c in stats_dao.calls}
    assert set(by_hk) == {"champ_hk", "chal_hk"}
    assert by_hk["champ_hk"]["scores_refresh_block"] == 42
    # Champion's own row carries count + avg, no overlap-avg (we don't
    # compare the champion against themselves).
    champ_env = by_hk["champ_hk"]["scores_by_env"]["ENV_A"]
    assert champ_env["count"] == 2 and champ_env["avg"] == pytest.approx(0.5)
    assert "champion_overlap_avg" not in champ_env
    # Challenger's row carries champion-on-overlap as the basis.
    chal_env = by_hk["chal_hk"]["scores_by_env"]["ENV_A"]
    assert chal_env["count"] == 2 and chal_env["avg"] == pytest.approx(0.5)
    assert chal_env["champion_overlap_avg"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_refresh_once_skips_when_no_samples():
    """No miner samples at the current refresh_block → no writes at
    all (the per-miner ``_compute_scores`` returns empty)."""
    stats_dao = _CapturingStatsDAO()
    monitor = LiveScoresMonitor.__new__(LiveScoresMonitor)
    monitor._samples = _FakeSamplesAdapter({})  # no samples
    monitor._stats_dao = stats_dao
    monitor._miners_dao = _FakeMinersDAOForMonitor([
        {"uid": 1, "hotkey": "hk", "revision": "r", "is_valid": "true"},
    ])
    monitor._state = _FakeStateStore(
        TaskIdState(task_ids={"ENV_A": [1, 2]}, refreshed_at_block=42),
        ["ENV_A"], champion_uid=None,
    )

    n = await monitor.refresh_once()
    assert n == 0
    assert stats_dao.calls == []
