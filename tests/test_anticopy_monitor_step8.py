"""miners_monitor step 8.0 — tokenizer-signature early reject.

The full enqueue / prune flow no longer lives here: the CEAC worker
moved to a pull-based model that reads ``miners`` + ``miner_stats``
+ ``scores_index`` directly, so the monitor only needs to stamp
``info.tokenizer_sig`` and mark a mismatched candidate permanent-
invalid. These tests cover that single remaining gate.
"""

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


class _FakeStateDAO:
    """Stubs the anticopy_state DAO; only the tokenizer-sig getter
    matters for the monitor step 8.0 path."""
    def __init__(self, champion_sig=""):
        self.champion_sig = champion_sig
    async def get_champion_tokenizer_sig(self):
        return self.champion_sig


class _FakeScoresDAO:
    """Step 8.1 reads scores_index to surface anticopy verdicts. The
    fake takes an optional fixed return shape so tests can pre-load
    a 'copy_of' row for a hotkey/revision pair."""

    def __init__(self, return_row=None):
        self.return_row = return_row

    async def get_score(self, hotkey, revision):
        return self.return_row


def _build_monitor(
    *,
    cfg_enabled=True,
    champion_sig="champion_sig",
    cand_sig="champion_sig",
    score_row=None,
):
    """Wire a MinersMonitor with anticopy state/scores stubs."""
    monitor = MinersMonitor()
    monitor.anticopy_state_dao = _FakeStateDAO(champion_sig=champion_sig)
    monitor.anticopy_scores_dao = _FakeScoresDAO(return_row=score_row)

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
        block=99_999_999,
        commit_count=1,
    )
    assert info.is_valid is False
    assert info.permanent_invalid is True
    assert info.invalid_reason.startswith("tokenizer_sig_mismatch:")


@pytest.mark.asyncio
async def test_tokenizer_match_stays_valid(monkeypatch):
    """Candidate with matching tokenizer signature stays valid — the
    monitor no longer enqueues anything; the worker picks the miner up
    on its next pull-based scan."""
    monkeypatch.setattr(db_client, "get_client", lambda: _FakeDynamoClient())
    monitor = _build_monitor(
        champion_sig="S" * 64, cand_sig="S" * 64, cfg_enabled=True,
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


@pytest.mark.asyncio
async def test_anticopy_disabled_passes_through(monkeypatch):
    """When the anticopy config is disabled the monitor must NOT
    even attempt the tokenizer-sig lookup; step 8.0 is a no-op."""
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


# ---- step 8.1: anticopy verdict → invalid ------------------------------------


@pytest.mark.asyncio
async def test_anticopy_copy_verdict_marks_invalid(monkeypatch):
    """Monitor finds the miner flagged copy_of in scores_index → marks
    invalid with reason carrying the origin model name + dec_med so
    operators can see *why* without joining tables."""
    monkeypatch.setattr(db_client, "get_client", lambda: _FakeDynamoClient())
    monitor = _build_monitor(
        champion_sig="S" * 64, cand_sig="S" * 64, cfg_enabled=True,
        score_row={
            "hotkey": "hk_copier_xyz", "revision": "rev",
            "verdict_copy_of": "hk_origin_xyz",
            "closest_peer_model": "victim-org/Affine-original",
            "decision_median": 0.0123,
        },
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
        uid=42, hotkey="hk_copier_xyz",
        model="org/affine-model-hk_copier_xyz",
        revision="rev",
        block=99_999_999,
        commit_count=1,
    )
    assert info.is_valid is False
    # Not permanent — if the verdict gets re-evaluated and clears
    # (origin deregisters, threshold change, etc.) the next monitor
    # cycle should pick it back up.
    assert info.permanent_invalid is False
    assert info.invalid_reason.startswith("anticopy_copy:")
    assert "victim-org/Affine-original" in info.invalid_reason
    assert "dm=0.0123" in info.invalid_reason


@pytest.mark.asyncio
async def test_anticopy_independent_verdict_stays_valid(monkeypatch):
    """Score row exists but ``verdict_copy_of`` is empty → independent,
    miner stays valid."""
    monkeypatch.setattr(db_client, "get_client", lambda: _FakeDynamoClient())
    monitor = _build_monitor(
        champion_sig="S" * 64, cand_sig="S" * 64, cfg_enabled=True,
        score_row={
            "hotkey": "hk_indep_xyz", "revision": "rev",
            "verdict_copy_of": "",
            "decision_median": 0.5,
        },
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
        uid=42, hotkey="hk_indep_xyz",
        model="org/affine-model-hk_indep_xyz",
        revision="rev",
        block=99_999_999,
        commit_count=1,
    )
    assert info.is_valid is True


@pytest.mark.asyncio
async def test_anticopy_no_score_row_yet_stays_valid(monkeypatch):
    """Miner hasn't been verdicted yet (no row in scores_index) — must
    not block them, anticopy verdict is best-effort backfill."""
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
        uid=42, hotkey="hk_unscored_xyz",
        model="org/affine-model-hk_unscored_xyz",
        revision="rev",
        block=99_999_999,
        commit_count=1,
    )
    assert info.is_valid is True


# ---- shared helpers ----------------------------------------------------------


async def _hot_get_model_info(model_id, revision):
    return ("hash_abc", revision, "")


async def _async_pass_size(*_args, **_kw):
    return {"pass": True}


async def _async_safe(*_args, **_kw):
    return {"safe": True, "reason": ""}
