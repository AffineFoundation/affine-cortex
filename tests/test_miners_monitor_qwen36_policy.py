"""Qwen3.6 submission window policy in the miners monitor."""

import pytest

from affine.src.anticopy.threshold import AntiCopyConfig
from affine.src.monitor import miners_monitor as monitor_mod
from affine.src.monitor.miners_monitor import MinersMonitor


class _FakeMinersDAO:
    async def get_miner_by_uid(self, _uid):
        return None


def _build_monitor():
    monitor = MinersMonitor()
    monitor.dao = _FakeMinersDAO()
    monitor._get_model_info = _hot_get_model_info
    monitor._safe_load_anticopy_config = _anticopy_disabled
    return monitor


@pytest.mark.asyncio
async def test_qwen36_submission_before_allowed_block_is_rejected(monkeypatch):
    monkeypatch.setattr(monitor_mod, "QWEN36_ALLOWED_FROM_BLOCK", 1000)
    monkeypatch.setattr(monitor_mod, "check_model_size", _qwen36_size_ok)
    monkeypatch.setattr(monitor_mod, "check_template_safety", _template_safe)

    info = await _build_monitor()._validate_miner(
        uid=42,
        hotkey="hkqwen36",
        model="org/affine-model-hkqwen36",
        revision="rev",
        block=999,
        commit_count=1,
    )

    assert info.is_valid is False
    assert info.permanent_invalid is True
    assert info.terminate_stats is True
    assert info.model_type == monitor_mod.QWEN36_MODEL_TYPE
    assert info.invalid_reason == (
        "model_check:qwen36_not_allowed_until_block:1000:commit_block=999"
    )


@pytest.mark.asyncio
async def test_qwen36_submission_at_allowed_block_is_valid(monkeypatch):
    monkeypatch.setattr(monitor_mod, "QWEN36_ALLOWED_FROM_BLOCK", 1000)
    monkeypatch.setattr(monitor_mod, "check_model_size", _qwen36_size_ok)
    monkeypatch.setattr(monitor_mod, "check_template_safety", _template_safe)

    info = await _build_monitor()._validate_miner(
        uid=42,
        hotkey="hkqwen36",
        model="org/affine-model-hkqwen36",
        revision="rev",
        block=1000,
        commit_count=1,
    )

    assert info.is_valid is True
    assert info.model_type == monitor_mod.QWEN36_MODEL_TYPE


@pytest.mark.asyncio
async def test_qwen36_allow_block_does_not_reject_dense_qwen3(monkeypatch):
    monkeypatch.setattr(monitor_mod, "QWEN36_ALLOWED_FROM_BLOCK", 1000)
    monkeypatch.setattr(monitor_mod, "check_model_size", _qwen3_size_ok)
    monkeypatch.setattr(monitor_mod, "check_template_safety", _template_safe)

    info = await _build_monitor()._validate_miner(
        uid=42,
        hotkey="hkdense",
        model="org/affine-model-hkdense",
        revision="rev",
        block=999,
        commit_count=1,
    )

    assert info.is_valid is True
    assert info.model_type == "qwen3"


async def _hot_get_model_info(_model_id, revision):
    return ("hash_abc", revision, "")


async def _anticopy_disabled():
    return AntiCopyConfig(enabled=False)


async def _qwen36_size_ok(*_args, **_kw):
    return {"pass": True, "model_type": monitor_mod.QWEN36_MODEL_TYPE}


async def _qwen3_size_ok(*_args, **_kw):
    return {"pass": True, "model_type": "qwen3"}


async def _template_safe(*_args, **_kw):
    return {"safe": True, "reason": ""}
