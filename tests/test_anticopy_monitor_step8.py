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
    """Step 8.0 doesn't need scores_index lookup, but the constructor
    expects one; this is a no-op stub."""
    async def get_score(self, hotkey, revision):
        return None


def _build_monitor(
    *,
    cfg_enabled=True,
    champion_sig="champion_sig",
    cand_sig="champion_sig",
):
    """Wire a MinersMonitor with anticopy state/scores stubs."""
    monitor = MinersMonitor()
    monitor.anticopy_state_dao = _FakeStateDAO(champion_sig=champion_sig)
    monitor.anticopy_scores_dao = _FakeScoresDAO()

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


# ---- shared helpers ----------------------------------------------------------


async def _hot_get_model_info(model_id, revision):
    return ("hash_abc", revision, "")


async def _async_pass_size(*_args, **_kw):
    return {"pass": True}


async def _async_safe(*_args, **_kw):
    return {"safe": True, "reason": ""}
