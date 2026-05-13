"""Tests for affine.src.scheduler.targon helpers.

Network-touching helpers (probe_model_ready / wait_for_ready) are not
unit-tested here — they need a real Targon endpoint and live in the
deferred Targon-staging integration tests. This file covers the pure
list-filtering logic in find_existing_workload and orphan_sweep with a
``_FakeTargonClient`` substitute for ``TargonClient``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from affine.src.scheduler.targon import (
    DeployTarget,
    find_existing_workload,
    orphan_sweep,
)


class _FakeTargonClient:
    """Minimal stand-in for affine.core.providers.targon_client.TargonClient.

    Only implements the methods scheduler.targon actually calls.
    """

    WORKLOAD_NAME_PREFIX = "affine"

    def __init__(self, workloads: List[Dict[str, Any]], *, configured: bool = True):
        self._workloads = workloads
        self.configured = configured
        self.deleted: List[str] = []

    def _workload_name(self, model: str, revision: str, *, uid: int, hotkey: str) -> str:
        # Production uses a similar deterministic name. Truncate hotkey to
        # keep names short and DNS-safe.
        return f"{self.WORKLOAD_NAME_PREFIX}-{uid}-{hotkey[:8]}-{revision[:8]}"

    async def list_workloads(self, *, limit: int = 100) -> Dict[str, Any]:
        return {"items": self._workloads[:limit]}

    async def delete_deployment(self, wid: str) -> bool:
        self.deleted.append(wid)
        return True


def _make_target(uid: int = 1, hotkey: str = "abc123def456", revision: str = "rev01234") -> DeployTarget:
    return DeployTarget(
        uid=uid, hotkey=hotkey, model="org/m", revision=revision,
    )


# ---- find_existing_workload ------------------------------------------------


@pytest.mark.asyncio
async def test_find_existing_workload_returns_uid_when_match_running():
    target = _make_target()
    expected_name = f"affine-1-abc123de-rev01234"
    client = _FakeTargonClient([
        {"uid": "wrk-001", "name": expected_name,
         "state": {"status": "running"}},
        {"uid": "wrk-002", "name": "affine-other", "state": {"status": "running"}},
    ])
    found = await find_existing_workload(client, target)
    assert found == "wrk-001"


@pytest.mark.asyncio
async def test_find_existing_workload_returns_none_when_no_match():
    target = _make_target()
    client = _FakeTargonClient([
        {"uid": "wrk-XYZ", "name": "affine-99-xxxxxxxx-yyyyyyyy",
         "state": {"status": "running"}},
    ])
    assert await find_existing_workload(client, target) is None


@pytest.mark.asyncio
async def test_find_existing_workload_skips_terminated_states():
    target = _make_target()
    expected_name = f"affine-1-abc123de-rev01234"
    client = _FakeTargonClient([
        {"uid": "wrk-old", "name": expected_name, "state": {"status": "terminated"}},
        {"uid": "wrk-err", "name": expected_name, "state": {"status": "error"}},
    ])
    # Neither is in the "live" set {running, provisioning, deploying, rebuilding}.
    assert await find_existing_workload(client, target) is None


@pytest.mark.asyncio
async def test_find_existing_workload_accepts_deploying_state():
    target = _make_target()
    expected_name = f"affine-1-abc123de-rev01234"
    client = _FakeTargonClient([
        {"uid": "wrk-dep", "name": expected_name, "state": {"status": "deploying"}},
    ])
    assert await find_existing_workload(client, target) == "wrk-dep"


# ---- orphan_sweep ----------------------------------------------------------


@pytest.mark.asyncio
async def test_orphan_sweep_deletes_affine_workloads_not_in_known_set():
    client = _FakeTargonClient([
        {"uid": "wrk-keep", "name": "affine-1-abc-r1"},
        {"uid": "wrk-orphan", "name": "affine-2-def-r2"},
        {"uid": "wrk-other", "name": "someone-else-3"},  # non-affine prefix
    ])
    deleted = await orphan_sweep(client, known_workload_ids={"wrk-keep"})
    assert deleted == 1
    assert client.deleted == ["wrk-orphan"]


@pytest.mark.asyncio
async def test_orphan_sweep_never_touches_non_affine_prefixed_workloads():
    client = _FakeTargonClient([
        {"uid": "wrk-foreign-1", "name": "rival-corp-job-1"},
        {"uid": "wrk-foreign-2", "name": "production-llm-abc"},
    ])
    deleted = await orphan_sweep(client, known_workload_ids=set())
    assert deleted == 0
    assert client.deleted == []


@pytest.mark.asyncio
async def test_orphan_sweep_noop_when_client_unconfigured():
    client = _FakeTargonClient([
        {"uid": "wrk-orphan", "name": "affine-1-abc-r1"},
    ], configured=False)
    deleted = await orphan_sweep(client, known_workload_ids=set())
    assert deleted == 0
    assert client.deleted == []
