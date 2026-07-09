"""Gated-repo download-access probe in the miners monitor.

Miner models must be fully public. A gated HF repo returns full metadata via
``repo_info`` even when the validator isn't on its allow-list, so the
architecture/hash checks pass while the scheduler still can't download the
weights (403 at deploy). The monitor probes a real file resolve to surface that
403 — a deterministic ``GatedRepoError`` is a permanent, miner-attributable
reject (raised as ``HFRepoUnavailable``); rate limiting (429), 5xx, and network
blips must never terminate a good miner.
"""

import asyncio

import pytest
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

from affine.src.monitor.miners_monitor import HFRepoUnavailable, MinersMonitor


def _monitor():
    # The probe uses no instance state; skip the real __init__.
    return MinersMonitor.__new__(MinersMonitor)


def _run(monkeypatch, raises=None):
    def fake_metadata(url, token=None):
        if raises is not None:
            raise raises
        return object()

    monkeypatch.setattr(
        "affine.src.monitor.miners_monitor.get_hf_file_metadata", fake_metadata
    )
    return asyncio.run(_monitor()._assert_gated_downloadable("owner/model", "abc1234"))


def test_gated_403_is_permanent_reject(monkeypatch):
    # Deterministic gated 403 → HFRepoUnavailable → caller permanently terminates.
    with pytest.raises(HFRepoUnavailable) as exc:
        _run(monkeypatch, raises=GatedRepoError.__new__(GatedRepoError))
    assert exc.value.reason == "hf_repo_gated"


def test_public_repo_passes(monkeypatch):
    # Fully public (or access-accepted) repo → resolve succeeds → no raise.
    _run(monkeypatch, raises=None)


def test_rate_limit_is_swallowed(monkeypatch):
    # 429 is an HfHubHTTPError, NOT GatedRepoError → must not terminate the miner.
    _run(monkeypatch, raises=HfHubHTTPError.__new__(HfHubHTTPError))


def test_network_error_is_swallowed(monkeypatch):
    # Transient connection blip → must not terminate the miner.
    _run(monkeypatch, raises=ConnectionError("temporary"))


# --------------------------------------------------------------------------- #
# Grace window for transient HF 404 on freshly created/renamed repos.
# A brand-new repo can 404 for a few minutes while the hub propagates; the
# monitor must defer (not permanently terminate) until the grace window lapses.
# --------------------------------------------------------------------------- #

from affine.src.monitor.miners_monitor import HF_UNAVAILABLE_GRACE_SECONDS


def _grace_monitor():
    m = MinersMonitor.__new__(MinersMonitor)
    m._repo_unavailable_since = {}
    return m


def test_not_found_grace_defers_then_escalates(monkeypatch):
    m = _grace_monitor()
    clock = {"t": 1000.0}
    monkeypatch.setattr(
        "affine.src.monitor.miners_monitor.time.time", lambda: clock["t"]
    )
    # First sighting → within grace.
    assert m._repo_unavailable_grace_active("o/affine-hk", "rev", "hf_repo_not_found")
    # Still inside the window.
    clock["t"] += HF_UNAVAILABLE_GRACE_SECONDS - 1
    assert m._repo_unavailable_grace_active("o/affine-hk", "rev", "hf_repo_not_found")
    # Past the window → escalate (terminate for real).
    clock["t"] += 2
    assert not m._repo_unavailable_grace_active("o/affine-hk", "rev", "hf_repo_not_found")


def test_deliberate_states_never_graced():
    m = _grace_monitor()
    for reason in ("hf_repo_gated", "hf_repo_private", "hf_repo_disabled"):
        assert not m._repo_unavailable_grace_active("o/affine-hk", "rev", reason)
    # No grace bookkeeping should have been created for them.
    assert m._repo_unavailable_since == {}


def test_clear_resets_grace_on_recovery(monkeypatch):
    m = _grace_monitor()
    clock = {"t": 1000.0}
    monkeypatch.setattr(
        "affine.src.monitor.miners_monitor.time.time", lambda: clock["t"]
    )
    assert m._repo_unavailable_grace_active("o/affine-hk", "rev", "hf_repo_not_found")
    m._clear_repo_unavailable("o/affine-hk", "rev")
    # A later 404 gets a fresh full window rather than an already-expired one.
    clock["t"] += HF_UNAVAILABLE_GRACE_SECONDS + 100
    assert m._repo_unavailable_grace_active("o/affine-hk", "rev", "hf_repo_not_found")


def _validate_with_404(monkeypatch, m):
    from affine.src.monitor.miners_monitor import HFRepoUnavailable

    async def raise_404(model_id, revision):
        raise HFRepoUnavailable("hf_repo_not_found")

    monkeypatch.setattr(m, "_get_model_info", raise_404)
    return asyncio.run(
        m._validate_miner(
            uid=5,
            hotkey="hk",
            model="owner/affine-hk",
            revision="rev",
            block=0,
        )
    )


def test_validate_miner_defers_first_cycle_then_terminates(monkeypatch):
    m = _grace_monitor()
    clock = {"t": 1000.0}
    monkeypatch.setattr(
        "affine.src.monitor.miners_monitor.time.time", lambda: clock["t"]
    )
    # First cycle: transient 404 → invalid this cycle but NOT a permanent kill.
    info = _validate_with_404(monkeypatch, m)
    assert info.is_valid is False
    assert info.permanent_invalid is False
    assert info.terminate_stats is False
    # After the grace window: same 404 now escalates to permanent terminate.
    clock["t"] += HF_UNAVAILABLE_GRACE_SECONDS + 1
    info = _validate_with_404(monkeypatch, m)
    assert info.permanent_invalid is True
    assert info.terminate_stats is True
