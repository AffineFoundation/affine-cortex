"""miners_monitor step 8.0 — tokenizer-signature early reject.

The full enqueue / prune flow no longer lives here: the CEAC worker
moved to a pull-based model that reads ``miners`` + ``miner_stats``
+ ``scores_index`` directly, so the monitor only needs to stamp
``info.tokenizer_sig`` and mark a mismatched candidate permanent-
invalid. These tests cover that single remaining gate.
"""

from __future__ import annotations

import httpx
import pytest
from huggingface_hub.errors import RevisionNotFoundError

import affine.database.client as db_client
from affine.src.anticopy import tokenizer_sig as tokenizer_sig_mod
from affine.src.anticopy.tokenizer_sig import (
    TOKENIZER_SIG_REVISION_NOT_FOUND,
)
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
    """Stubs the anticopy_state DAO for the monitor step 8.0 path."""
    def __init__(self, champion_sig="", state=None):
        self.champion_sig = champion_sig
        self.state = state
    async def get_champion_tokenizer_sig(self):
        return self.champion_sig
    async def get_state(self):
        if self.state is not None:
            return dict(self.state)
        return {
            "active_champion_uid": 7,
            "active_champion_hotkey": "champ_hk",
            "active_champion_model": "champ/model",
            "active_champion_revision": "champ_rev",
            "champion_tokenizer_sig": self.champion_sig,
        }


class _FakeConfigDAO:
    def __init__(self, champion=None, legacy_sig=""):
        self.champion = (
            champion
            if champion is not None
            else {
                "uid": 7,
                "hotkey": "champ_hk",
                "model": "champ/model",
                "revision": "champ_rev",
            }
        )
        self.legacy_sig = legacy_sig

    async def get_param_value(self, name, default=None):
        if name == "champion":
            return self.champion
        if name == "anticopy_champion_tokenizer_sig":
            return self.legacy_sig
        return default


class _FakeScoresDAO:
    """Step 8.1 reads scores_index to surface anticopy verdicts. The
    fake takes an optional fixed return shape so tests can pre-load
    a 'copy_of' row for a hotkey/revision pair."""

    def __init__(self, return_row=None):
        self.return_row = return_row

    async def get_score(self, hotkey, revision):
        return self.return_row


class _FakeMinerDAO:
    def __init__(self, row=None):
        self.row = row

    async def get_miner_by_uid(self, _uid):
        return self.row


def _revision_not_found_error() -> RevisionNotFoundError:
    request = httpx.Request(
        "GET",
        "https://huggingface.co/api/models/org/model/revision/missing",
    )
    response = httpx.Response(404, request=request)
    return RevisionNotFoundError(
        "Revision not found",
        response=response,
    )


def _build_monitor(
    *,
    cfg_enabled=True,
    champion_sig="champion_sig",
    cand_sig="champion_sig",
    score_row=None,
    state=None,
    champion=None,
    legacy_sig="",
):
    """Wire a MinersMonitor with anticopy state/scores stubs."""
    monitor = MinersMonitor()
    monitor.config_dao = _FakeConfigDAO(
        champion=champion, legacy_sig=legacy_sig,
    )
    monitor.anticopy_state_dao = _FakeStateDAO(
        champion_sig=champion_sig, state=state,
    )
    monitor.anticopy_scores_dao = _FakeScoresDAO(return_row=score_row)

    async def _fake_load(_dao):
        return AntiCopyConfig(enabled=cfg_enabled)
    monitor._safe_load_anticopy_config = (
        lambda: _fake_load(monitor.config_dao)
    )

    async def _fake_sig(_model, _rev):
        reason = "" if cand_sig else "transient_fetch_failed"
        return cand_sig, reason
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
async def test_tokenizer_mismatch_skipped_when_anticopy_state_is_stale(monkeypatch):
    """A stale CEAC rollout anchor must not reject candidates for the
    live champion's tokenizer. The refresh cadence intentionally keeps
    rollout pools stable, so the monitor has to verify the anchor before
    using its signature as an early-reject gate."""
    monkeypatch.setattr(db_client, "get_client", lambda: _FakeDynamoClient())
    monitor = _build_monitor(
        champion_sig="A" * 64,
        cand_sig="B" * 64,
        cfg_enabled=True,
        state={
            "active_champion_uid": 83,
            "active_champion_hotkey": "old_hk",
            "active_champion_model": "old/model",
            "active_champion_revision": "old_rev",
            "champion_tokenizer_sig": "A" * 64,
        },
        champion={
            "uid": 2000,
            "hotkey": "SYSTEM-1000",
            "model": "Qwen/Qwen3.6-35B-A3B",
            "revision": "995ad96eacd98c81ed38be0c5b274b04031597b0",
        },
        legacy_sig="A" * 64,
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
        uid=42, hotkey="hk_stale_xyz",
        model="org/affine-model-hk_stale_xyz",
        revision="rev",
        block=99_999_999,
        commit_count=1,
    )
    assert info.is_valid is True
    assert info.invalid_reason is None


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


@pytest.mark.asyncio
async def test_retryable_model_check_failure_does_not_terminate(monkeypatch):
    monkeypatch.setattr(db_client, "get_client", lambda: _FakeDynamoClient())
    monitor = _build_monitor(cfg_enabled=True)
    monitor._get_model_info = _hot_get_model_info
    monkeypatch.setattr(
        "affine.src.monitor.miners_monitor.check_model_size",
        _async_retryable_size_failure,
    )
    monkeypatch.setattr(
        "affine.src.monitor.miners_monitor.check_template_safety",
        _async_safe,
    )

    info = await monitor._validate_miner(
        uid=42, hotkey="hk_retryable_xyz",
        model="org/affine-model-hk_retryable_xyz",
        revision="rev",
        block=99_999_999,
        commit_count=1,
    )

    assert info.is_valid is False
    assert info.invalid_reason == "model_check:config_fetch_failed"
    assert info.permanent_invalid is False
    assert info.terminate_stats is False


@pytest.mark.asyncio
async def test_retryable_model_check_preserves_previously_valid_miner(monkeypatch):
    monkeypatch.setattr(db_client, "get_client", lambda: _FakeDynamoClient())
    monitor = _build_monitor(cfg_enabled=True)
    monitor.dao = _FakeMinerDAO({
        "model": "org/affine-model-hk_retryable_xyz",
        "revision": "rev",
        "is_valid": "true",
        "template_check_result": "safe",
        "model_type": "qwen3_5_moe",
    })
    monitor._get_model_info = _hot_get_model_info
    monkeypatch.setattr(
        "affine.src.monitor.miners_monitor.check_model_size",
        _async_retryable_size_failure,
    )
    monkeypatch.setattr(
        "affine.src.monitor.miners_monitor.check_template_safety",
        _async_safe,
    )

    info = await monitor._validate_miner(
        uid=42, hotkey="hk_retryable_xyz",
        model="org/affine-model-hk_retryable_xyz",
        revision="rev",
        block=99_999_999,
        commit_count=1,
    )

    assert info.is_valid is True
    assert info.invalid_reason is None
    assert info.model_type == "qwen3_5_moe"
    assert info.permanent_invalid is False
    assert info.terminate_stats is False


@pytest.mark.asyncio
async def test_transient_hf_fetch_failure_preserves_previously_valid_miner(monkeypatch):
    monkeypatch.setattr(db_client, "get_client", lambda: _FakeDynamoClient())
    monitor = _build_monitor(cfg_enabled=True)
    monitor.dao = _FakeMinerDAO({
        "model": "org/affine-model-hk_hf_transient_xyz",
        "revision": "rev",
        "is_valid": "true",
        "template_check_result": "safe",
        "model_hash": "hash_cached",
        "model_type": "qwen3_5_moe",
    })

    async def _transient_hf_failure(_model, _revision):
        return None

    monitor._get_model_info = _transient_hf_failure

    info = await monitor._validate_miner(
        uid=238, hotkey="hk_hf_transient_xyz",
        model="org/affine-model-hk_hf_transient_xyz",
        revision="rev",
        block=99_999_999,
        commit_count=1,
    )

    assert info.is_valid is True
    assert info.invalid_reason is None
    assert info.model_hash == "hash_cached"
    assert info.model_type == "qwen3_5_moe"
    assert info.permanent_invalid is False
    assert info.terminate_stats is False


@pytest.mark.asyncio
async def test_transient_hf_fetch_failure_does_not_bypass_copy_verdict(monkeypatch):
    monkeypatch.setattr(db_client, "get_client", lambda: _FakeDynamoClient())
    monitor = _build_monitor(
        cfg_enabled=True,
        score_row={
            "hotkey": "hk_hf_copy_xyz",
            "revision": "rev",
            "verdict_copy_of": "hk_origin_xyz",
            "closest_peer_model": "victim-org/Affine-original",
            "decision_median": 0.0123,
        },
    )
    monitor.dao = _FakeMinerDAO({
        "model": "org/affine-model-hk_hf_copy_xyz",
        "revision": "rev",
        "is_valid": "true",
        "template_check_result": "safe",
        "model_hash": "hash_cached",
        "model_type": "qwen3_5_moe",
    })

    async def _transient_hf_failure(_model, _revision):
        return None

    monitor._get_model_info = _transient_hf_failure

    info = await monitor._validate_miner(
        uid=238, hotkey="hk_hf_copy_xyz",
        model="org/affine-model-hk_hf_copy_xyz",
        revision="rev",
        block=99_999_999,
        commit_count=1,
    )

    assert info.is_valid is False
    assert info.invalid_reason.startswith("anticopy_copy:")
    assert "victim-org/Affine-original" in info.invalid_reason
    assert info.permanent_invalid is False
    assert info.terminate_stats is False


@pytest.mark.asyncio
async def test_revision_not_found_does_not_preserve_previously_valid_miner(monkeypatch):
    monkeypatch.setattr(db_client, "get_client", lambda: _FakeDynamoClient())
    monitor = _build_monitor(cfg_enabled=True)
    monitor.dao = _FakeMinerDAO({
        "model": "org/affine-model-hk_revision_missing_xyz",
        "revision": "missing",
        "is_valid": "true",
        "template_check_result": "safe",
        "model_hash": "hash_cached",
        "model_type": "qwen3_5_moe",
    })

    class _RevisionMissingHfApi:
        def __init__(self, *args, **kwargs):
            pass

        def repo_info(self, *args, **kwargs):
            raise _revision_not_found_error()

    monkeypatch.setattr(
        "affine.src.monitor.miners_monitor.HfApi",
        _RevisionMissingHfApi,
    )

    info = await monitor._validate_miner(
        uid=238, hotkey="hk_revision_missing_xyz",
        model="org/affine-model-hk_revision_missing_xyz",
        revision="missing",
        block=99_999_999,
        commit_count=1,
    )

    assert info.is_valid is False
    assert info.invalid_reason == "hf_revision_not_found"
    assert info.permanent_invalid is True
    assert info.terminate_stats is True


@pytest.mark.asyncio
async def test_transient_tokenizer_sig_failure_preserves_previously_valid_miner(monkeypatch):
    monkeypatch.setattr(db_client, "get_client", lambda: _FakeDynamoClient())
    monitor = _build_monitor(cfg_enabled=True, champion_sig="champ_sig", cand_sig="")
    monitor.dao = _FakeMinerDAO({
        "model": "org/affine-model-hk_tokenizer_transient_xyz",
        "revision": "rev",
        "is_valid": "true",
        "template_check_result": "safe",
        "model_hash": "hash_cached",
        "model_type": "qwen3_5_moe",
    })
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
        uid=255, hotkey="hk_tokenizer_transient_xyz",
        model="org/affine-model-hk_tokenizer_transient_xyz",
        revision="rev",
        block=99_999_999,
        commit_count=1,
    )

    assert info.is_valid is True
    assert info.invalid_reason is None
    assert info.model_hash == "hash_abc"
    assert info.model_type == "qwen3_5_moe"
    assert info.permanent_invalid is False
    assert info.terminate_stats is False


@pytest.mark.asyncio
async def test_transient_tokenizer_sig_failure_does_not_bypass_copy_verdict(monkeypatch):
    monkeypatch.setattr(db_client, "get_client", lambda: _FakeDynamoClient())
    monitor = _build_monitor(
        cfg_enabled=True,
        champion_sig="champ_sig",
        cand_sig="",
        score_row={
            "hotkey": "hk_tokenizer_copy_xyz",
            "revision": "rev",
            "verdict_copy_of": "hk_origin_xyz",
            "closest_peer_model": "victim-org/Affine-original",
            "decision_median": 0.0123,
        },
    )
    monitor.dao = _FakeMinerDAO({
        "model": "org/affine-model-hk_tokenizer_copy_xyz",
        "revision": "rev",
        "is_valid": "true",
        "template_check_result": "safe",
        "model_hash": "hash_cached",
        "model_type": "qwen3_5_moe",
    })
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
        uid=255, hotkey="hk_tokenizer_copy_xyz",
        model="org/affine-model-hk_tokenizer_copy_xyz",
        revision="rev",
        block=99_999_999,
        commit_count=1,
    )

    assert info.is_valid is False
    assert info.invalid_reason.startswith("anticopy_copy:")
    assert "victim-org/Affine-original" in info.invalid_reason
    assert info.permanent_invalid is False
    assert info.terminate_stats is False


def test_tokenizer_sig_revision_not_found_is_deterministic(monkeypatch):
    class _RevisionMissingHfApi:
        def __init__(self, *args, **kwargs):
            pass

        def repo_info(self, *args, **kwargs):
            raise _revision_not_found_error()

    monkeypatch.setattr(tokenizer_sig_mod, "HfApi", _RevisionMissingHfApi)

    sig, src, reason = tokenizer_sig_mod.compute_tokenizer_signature_sync(
        "org/model",
        "missing",
    )

    assert sig is None
    assert src == ""
    assert reason == TOKENIZER_SIG_REVISION_NOT_FOUND


@pytest.mark.asyncio
async def test_missing_tokenizer_artifact_does_not_preserve_prior_valid(monkeypatch):
    monkeypatch.setattr(db_client, "get_client", lambda: _FakeDynamoClient())
    monitor = _build_monitor(cfg_enabled=True, champion_sig="champ_sig", cand_sig="")
    monitor.dao = _FakeMinerDAO({
        "model": "org/affine-model-hk_missing_tokenizer_xyz",
        "revision": "rev",
        "is_valid": "true",
        "template_check_result": "safe",
        "model_hash": "hash_cached",
        "model_type": "qwen3_5_moe",
    })
    monitor._get_model_info = _hot_get_model_info

    async def _missing_tokenizer(_model, _rev):
        return "", "missing_tokenizer_artifact"

    monitor._get_tokenizer_sig = _missing_tokenizer
    monkeypatch.setattr(
        "affine.src.monitor.miners_monitor.check_model_size",
        _async_pass_size,
    )
    monkeypatch.setattr(
        "affine.src.monitor.miners_monitor.check_template_safety",
        _async_safe,
    )

    info = await monitor._validate_miner(
        uid=255, hotkey="hk_missing_tokenizer_xyz",
        model="org/affine-model-hk_missing_tokenizer_xyz",
        revision="rev",
        block=99_999_999,
        commit_count=1,
    )

    assert info.is_valid is False
    assert info.invalid_reason == "tokenizer_sig_missing_tokenizer_artifact"
    assert info.permanent_invalid is False
    assert info.terminate_stats is False


@pytest.mark.asyncio
async def test_retryable_model_check_rejects_cached_type_outside_current_policy(monkeypatch):
    monkeypatch.setattr(db_client, "get_client", lambda: _FakeDynamoClient())
    monkeypatch.setattr(
        "affine.src.monitor.miners_monitor.QWEN36_ONLY_ENFORCE_BLOCK",
        0,
    )
    monitor = _build_monitor(cfg_enabled=True)
    monitor.dao = _FakeMinerDAO({
        "model": "org/affine-model-hk_retryable_xyz",
        "revision": "rev",
        "is_valid": "true",
        "template_check_result": "safe",
        "model_type": "qwen3",
    })
    monitor._get_model_info = _hot_get_model_info
    monkeypatch.setattr(
        "affine.src.monitor.miners_monitor.check_model_size",
        _async_retryable_size_failure,
    )
    monkeypatch.setattr(
        "affine.src.monitor.miners_monitor.check_template_safety",
        _async_safe,
    )

    info = await monitor._validate_miner(
        uid=42, hotkey="hk_retryable_xyz",
        model="org/affine-model-hk_retryable_xyz",
        revision="rev",
        block=99_999_999,
        commit_count=1,
    )

    assert info.is_valid is False
    assert info.invalid_reason == (
        "model_check:model_type=qwen3 (expected one of: qwen3_5_moe)"
    )
    assert info.model_type == "qwen3"
    assert info.permanent_invalid is True
    assert info.terminate_stats is True


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


async def _async_retryable_size_failure(*_args, **_kw):
    return {
        "pass": False,
        "reason": "config_fetch_failed",
        "model_type": "",
        "retryable": True,
    }


async def _async_safe(*_args, **_kw):
    return {"safe": True, "reason": ""}
