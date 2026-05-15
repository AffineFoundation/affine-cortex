"""miners_monitor step 8.0 (tokenizer reject) + step 8 (enqueue/verdict)
wiring tests. Uses dependency-injected fakes; no DDB / HF / network."""

from __future__ import annotations

import pytest

import affine.database.client as db_client
from affine.src.anticopy.threshold import AntiCopyConfig
from affine.src.monitor.miners_monitor import MinersMonitor


class _FakeDynamoClient:
    async def put_item(self, **kwargs):
        return {}
    async def get_item(self, **kwargs):
        return {}
    async def query(self, **kwargs):
        return {"Items": []}
    async def update_item(self, **kwargs):
        return {}


class _FakeConfigDAO:
    def __init__(self, champion_sig=""):
        self.champion_sig = champion_sig
        self.get_calls = []

    async def get_param_value(self, name, default=None):
        self.get_calls.append(name)
        if name == "anticopy_champion_tokenizer_sig":
            return self.champion_sig
        return default


class _FakeJobsDAO:
    def __init__(self):
        self.enqueued = []
    async def enqueue(self, **kwargs):
        self.enqueued.append(kwargs)


class _FakeScoresDAO:
    def __init__(self, score_row=None):
        self._row = score_row
    async def get_score(self, hotkey, revision):
        return self._row


class _FakeStateDAO:
    """Stubs the anticopy_state DAO; only the tokenizer-sig getter
    matters for the monitor step 8.0 path."""
    def __init__(self, champion_sig=""):
        self.champion_sig = champion_sig
    async def get_champion_tokenizer_sig(self):
        return self.champion_sig


def _build_monitor(
    *,
    cfg_enabled=True,
    champion_sig="champion_sig",
    score_row=None,
    cand_sig="champion_sig",
):
    """Wire a MinersMonitor with all anticopy dependencies stubbed."""
    monitor = MinersMonitor()
    monitor.config_dao = _FakeConfigDAO(champion_sig=champion_sig)
    monitor.anticopy_jobs_dao = _FakeJobsDAO()
    monitor.anticopy_scores_dao = _FakeScoresDAO(score_row=score_row)
    monitor.anticopy_state_dao = _FakeStateDAO(champion_sig=champion_sig)

    async def _fake_load(_dao):
        return AntiCopyConfig(enabled=cfg_enabled)
    monitor._safe_load_anticopy_config = (
        lambda: _fake_load(monitor.config_dao)
    )

    async def _fake_sig(_model, _rev):
        return cand_sig
    monitor._get_tokenizer_sig = _fake_sig
    return monitor


# ---- step 8.0 -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_tokenizer_mismatch_marks_permanent_invalid(monkeypatch):
    monkeypatch.setattr(db_client, "get_client", lambda: _FakeDynamoClient())
    monitor = _build_monitor(
        champion_sig="A" * 64, cand_sig="B" * 64, cfg_enabled=True,
    )
    # short-circuit the earlier validation steps
    monitor._get_model_info = _hot_get_model_info
    monkeypatch.setattr(
        "affine.src.monitor.miners_monitor.check_model_size",
        _async_pass_size,
    )
    monkeypatch.setattr(
        "affine.src.monitor.miners_monitor.check_template_safety",
        _async_safe,
    )

    info = await monitor._validate_miner(
        uid=42, hotkey="hk_test_anticopy_xyz",
        model="org/affine-model-hk_test_anticopy_xyz",
        revision="rev",
        block=99_999_999,             # past the suffix enforce block
        commit_count=1,
    )
    assert info.is_valid is False
    assert info.permanent_invalid is True
    assert info.invalid_reason.startswith("tokenizer_sig_mismatch:")


@pytest.mark.asyncio
async def test_tokenizer_match_then_enqueue_when_no_score(monkeypatch):
    monkeypatch.setattr(db_client, "get_client", lambda: _FakeDynamoClient())
    monitor = _build_monitor(
        champion_sig="S" * 64, cand_sig="S" * 64, cfg_enabled=True,
        score_row=None,
    )
    monitor._get_model_info = _hot_get_model_info
    monkeypatch.setattr(
        "affine.src.monitor.miners_monitor.check_model_size",
        _async_pass_size,
    )
    monkeypatch.setattr(
        "affine.src.monitor.miners_monitor.check_template_safety",
        _async_safe,
    )

    info = await monitor._validate_miner(
        uid=42, hotkey="hk_match_xyz",
        model="org/affine-model-hk_match_xyz",
        revision="rev",
        block=99_999_999,
        commit_count=1,
    )
    assert info.is_valid is True
    assert len(monitor.anticopy_jobs_dao.enqueued) == 1
    assert monitor.anticopy_jobs_dao.enqueued[0]["hotkey"] == "hk_match_xyz"


# ---- step 8 verdict does NOT touch miner state -------------------------------


@pytest.mark.asyncio
async def test_existing_copy_verdict_does_not_flip_miner(monkeypatch):
    """CEAC verdict lives in ``anticopy_scores_index`` only — even when
    the worker has already flagged this miner as ``copy_of`` someone
    else, ``miners.is_valid`` stays True. The similarity signal is a
    sandboxed advisory, not a kill switch.
    """
    monkeypatch.setattr(db_client, "get_client", lambda: _FakeDynamoClient())
    monitor = _build_monitor(
        champion_sig="S" * 64, cand_sig="S" * 64,
        score_row={"verdict_copy_of": "victim_hotkey", "computed_at": 1},
    )
    monitor._get_model_info = _hot_get_model_info
    monkeypatch.setattr(
        "affine.src.monitor.miners_monitor.check_model_size",
        _async_pass_size,
    )
    monkeypatch.setattr(
        "affine.src.monitor.miners_monitor.check_template_safety",
        _async_safe,
    )

    info = await monitor._validate_miner(
        uid=42, hotkey="hk_copy_xyz",
        model="org/affine-model-hk_copy_xyz",
        revision="rev",
        block=99_999_999,
        commit_count=1,
    )
    assert info.is_valid is True
    # already scored → no re-enqueue
    assert monitor.anticopy_jobs_dao.enqueued == []


# ---- step 8 gated off by anticopy.enabled = False -----------------------------


@pytest.mark.asyncio
async def test_anticopy_disabled_passes_through(monkeypatch):
    monkeypatch.setattr(db_client, "get_client", lambda: _FakeDynamoClient())
    monitor = _build_monitor(cfg_enabled=False)
    monitor._get_model_info = _hot_get_model_info
    monkeypatch.setattr(
        "affine.src.monitor.miners_monitor.check_model_size",
        _async_pass_size,
    )
    monkeypatch.setattr(
        "affine.src.monitor.miners_monitor.check_template_safety",
        _async_safe,
    )

    info = await monitor._validate_miner(
        uid=42, hotkey="hk_disabled_xyz",
        model="org/affine-model-hk_disabled_xyz",
        revision="rev",
        block=99_999_999,
        commit_count=1,
    )
    assert info.is_valid is True
    assert monitor.anticopy_jobs_dao.enqueued == []


# ---- shared helpers ----------------------------------------------------------


async def _hot_get_model_info(model_id, revision):
    return ("hash_abc", revision, "")


async def _async_pass_size(*_args, **_kw):
    return {"pass": True}


async def _async_safe(*_args, **_kw):
    return {"safe": True, "reason": ""}


# ---- stale-job prune --------------------------------------------------------


class _FakeJobsDAOWithPrune:
    """Fake jobs DAO that supports the prune contract: peek_pending +
    delete + STATE_PENDING/STATE_RUNNING constants."""
    STATE_PENDING = "pending"
    STATE_RUNNING = "running"

    def __init__(self, pending_rows):
        self._pending = pending_rows
        self.deleted = []

    async def peek_pending(self, *, limit=5, exclude_pk=None):
        return list(self._pending)

    async def delete(self, pk):
        self.deleted.append(pk)


@pytest.mark.asyncio
async def test_prune_stale_anticopy_jobs_removes_terminated():
    """Hotkey/revision pairs that aren't in the current metagraph
    snapshot must be deleted from the pending queue; live ones stay."""
    from affine.src.monitor.miners_monitor import MinersMonitor, MinerInfo

    monitor = MinersMonitor()
    monitor.anticopy_jobs_dao = _FakeJobsDAOWithPrune([
        # live (will stay)
        {"pk": "JOB#live_hk#rev1", "hotkey": "live_hk", "revision": "rev1"},
        # stale (will be deleted)
        {"pk": "JOB#dead_hk#rev2", "hotkey": "dead_hk", "revision": "rev2"},
        # live hotkey but stale revision — also deleted because the
        # hotkey re-committed to a new revision
        {"pk": "JOB#live_hk#oldrev", "hotkey": "live_hk", "revision": "oldrev"},
    ])

    current = [
        MinerInfo(uid=1, hotkey="live_hk", model="m1", revision="rev1"),
        MinerInfo(uid=2, hotkey="other_hk", model="m2", revision="rev_other"),
    ]

    await monitor._prune_stale_anticopy_jobs(current)

    assert sorted(monitor.anticopy_jobs_dao.deleted) == sorted([
        "JOB#dead_hk#rev2",
        "JOB#live_hk#oldrev",
    ])
