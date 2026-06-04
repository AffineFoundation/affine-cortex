"""RolloutRefreshService._maybe_tick scheduling.

Pins the contract that:
  * Refresh fires AT MOST once every ``refresh_interval_days`` (default
    7) — daily-tick churn made cross-day pair-comparisons depend on
    incidental overlap of refresh-window pools.
  * The schedule still respects ``refresh_utc_hour`` (don't fire in
    the middle of the night by mistake).
  * Disabled config short-circuits.
"""

from __future__ import annotations

import datetime as dt
from unittest.mock import patch

from affine.src.anticopy.refresh import RolloutRefreshService, _LAST_REFRESH_FIELD
from affine.src.anticopy.threshold import AntiCopyConfig


class _FakeStateDAO:
    def __init__(self, state=None):
        self._state = state or {}

    async def get_state(self):
        return dict(self._state)

    async def set_state(self, **fields):
        self._state.update(fields)


def _build_service(cfg, state):
    svc = RolloutRefreshService.__new__(RolloutRefreshService)
    svc.config_dao = object()
    svc.state_dao = _FakeStateDAO(state)
    svc.tick_calls = []

    async def fake_load(_dao):
        return cfg

    async def fake_tick(cfg_, *, day):
        svc.tick_calls.append(day)
        return {}

    svc.tick = fake_tick
    svc._load_cfg = fake_load
    return svc


def _run_at(svc, when_utc):
    """Helper: invoke _maybe_tick with a frozen wall clock. Only ``now``
    is overridden — ``strptime`` etc. delegate to the real class."""
    class _FrozenDt(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return when_utc

    import asyncio

    with patch("affine.src.anticopy.refresh.dt.datetime", _FrozenDt), \
         patch(
             "affine.src.anticopy.refresh.load_anticopy_config",
             svc._load_cfg,
         ):
        asyncio.run(svc._maybe_tick())


def test_disabled_config_short_circuits():
    svc = _build_service(
        AntiCopyConfig(enabled=False), state={},
    )
    _run_at(svc, dt.datetime(2026, 5, 17, 4, 0, tzinfo=dt.timezone.utc))
    assert svc.tick_calls == []


def test_first_run_fires_when_state_empty():
    svc = _build_service(
        AntiCopyConfig(
            enabled=True, refresh_interval_days=7, refresh_utc_hour=2,
        ),
        state={},
    )
    _run_at(svc, dt.datetime(2026, 5, 17, 4, 0, tzinfo=dt.timezone.utc))
    assert svc.tick_calls == ["2026-05-17"]


def test_skips_when_within_interval():
    svc = _build_service(
        AntiCopyConfig(
            enabled=True, refresh_interval_days=7, refresh_utc_hour=2,
        ),
        state={_LAST_REFRESH_FIELD: "2026-05-15"},
    )
    # 2 days after last → not yet due (interval=7).
    _run_at(svc, dt.datetime(2026, 5, 17, 4, 0, tzinfo=dt.timezone.utc))
    assert svc.tick_calls == []


def test_fires_when_interval_has_elapsed():
    svc = _build_service(
        AntiCopyConfig(
            enabled=True, refresh_interval_days=7, refresh_utc_hour=2,
        ),
        state={_LAST_REFRESH_FIELD: "2026-05-10"},
    )
    # 7 days after last → due.
    _run_at(svc, dt.datetime(2026, 5, 17, 4, 0, tzinfo=dt.timezone.utc))
    assert svc.tick_calls == ["2026-05-17"]


def test_skips_before_utc_hour_even_when_due():
    svc = _build_service(
        AntiCopyConfig(
            enabled=True, refresh_interval_days=7, refresh_utc_hour=2,
        ),
        state={_LAST_REFRESH_FIELD: "2026-05-10"},
    )
    # 7 days after last AND before refresh_utc_hour → skip.
    _run_at(svc, dt.datetime(2026, 5, 17, 1, 0, tzinfo=dt.timezone.utc))
    assert svc.tick_calls == []


def test_corrupt_last_refresh_falls_through_to_tick():
    """Garbage in the state field shouldn't wedge the service."""
    svc = _build_service(
        AntiCopyConfig(
            enabled=True, refresh_interval_days=7, refresh_utc_hour=2,
        ),
        state={_LAST_REFRESH_FIELD: "not-a-date"},
    )
    _run_at(svc, dt.datetime(2026, 5, 17, 4, 0, tzinfo=dt.timezone.utc))
    assert svc.tick_calls == ["2026-05-17"]
