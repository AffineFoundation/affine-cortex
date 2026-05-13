"""Tests for the DB-driven provider dispatch resolver.

``_resolve_provider_kind`` is the single decision point that decides
whether the scheduler runs the SSH or Targon lifecycle. The previous
``AFFINE_PROVIDER_KIND`` env var was removed; this helper is now the
only thing between the active ``inference_endpoints`` rows and the
provider selection.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from affine.src.scheduler.main import _resolve_provider_kind


@dataclass
class _Ep:
    name: str
    kind: str


def test_empty_active_raises():
    with pytest.raises(RuntimeError, match="no active inference endpoints"):
        _resolve_provider_kind([])


def test_all_ssh_resolves_to_ssh():
    eps = [_Ep("b300", "ssh"), _Ep("b300-2", "ssh")]
    kind, ssh, targon = _resolve_provider_kind(eps)
    assert kind == "ssh"
    assert [ep.name for ep in ssh] == ["b300", "b300-2"]
    assert targon == []


def test_all_targon_resolves_to_targon():
    eps = [_Ep("prod", "targon")]
    kind, ssh, targon = _resolve_provider_kind(eps)
    assert kind == "targon"
    assert ssh == []
    assert [ep.name for ep in targon] == ["prod"]


def test_mixed_kinds_raises():
    eps = [_Ep("b300", "ssh"), _Ep("prod", "targon")]
    with pytest.raises(RuntimeError, match="mixed-kind"):
        _resolve_provider_kind(eps)


def test_unknown_kinds_raises():
    eps = [_Ep("weird", "vllm")]
    with pytest.raises(RuntimeError, match="none are kind=ssh"):
        _resolve_provider_kind(eps)
