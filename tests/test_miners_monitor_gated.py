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
