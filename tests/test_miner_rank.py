"""``af get-rank`` renderer — partial / empty / full state.

The renderer is a pure function over dicts (no HTTP, no DB). We feed
it the same shapes the API would, capture stdout, and verify the
output makes sense and doesn't crash.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest

from affine.src.miner.rank import _print_queue, _print_window


def _render_window(state):
    buf = io.StringIO()
    with redirect_stdout(buf):
        _print_window(state)
    return buf.getvalue()


def _render_queue(queue):
    buf = io.StringIO()
    with redirect_stdout(buf):
        _print_queue(queue)
    return buf.getvalue()


# ---- _print_window --------------------------------------------------------


def test_window_renders_none_safely():
    out = _render_window(None)
    assert "no state" in out


def test_window_renders_empty_dict_safely():
    """An ``{}`` reply (which the API never actually produces — it always
    returns a dict with explicit None keys) collapses to the same
    'no state' branch as a None reply, because Python ``not {}`` is
    truthy. Documenting that behavior so a future API change doesn't
    silently break the renderer."""
    out = _render_window({})
    assert "no state" in out


def test_window_renders_champion_only():
    state = {
        "champion": {
            "uid": 5, "hotkey": "a" * 40, "revision": "r" * 40,
            "model": "org/m5",
        },
        "champion_base_url": "https://t/w1",
        "battle": None,
        "task_refresh_block": 7200,
    }
    out = _render_window(state)
    assert "uid=5" in out
    assert "org/m5" in out
    assert "https://t/w1" in out
    assert "(idle)" in out


def test_window_renders_battle_in_flight():
    state = {
        "champion": {"uid": 1, "hotkey": "champ", "revision": "rc", "model": "o/c"},
        "champion_base_url": "https://t/c",
        "battle": {
            "challenger": {"uid": 2, "hotkey": "chal", "revision": "rh", "model": "o/h"},
            "started_at_block": 42,
        },
        "task_refresh_block": 40,
    }
    out = _render_window(state)
    assert "uid=1" in out
    assert "uid=2" in out
    assert "started @ block 42" in out


def test_window_renders_when_champion_field_present_but_null():
    """API returns ``champion: None`` for empty-state currently — the
    renderer should treat that the same as missing."""
    state = {
        "champion": None,
        "champion_base_url": None,
        "battle": None,
        "task_refresh_block": None,
    }
    out = _render_window(state)
    assert "champion      : -" in out
    assert "battle        : - (idle)" in out


# ---- _print_queue ---------------------------------------------------------


def test_queue_renders_none_safely():
    out = _render_queue(None)
    assert "queue empty" in out


def test_queue_renders_empty_list_safely():
    out = _render_queue([])
    assert "queue empty" in out


def test_queue_renders_pending_entries():
    queue = [
        {"position": 1, "uid": 3, "hotkey": "hk_three", "revision": "r3", "model": "o/m3",
         "first_block": 100, "enqueued_at": 1000},
        {"position": 2, "uid": 7, "hotkey": "hk_seven", "revision": "r7", "model": "o/m7",
         "first_block": 200, "enqueued_at": 1100},
    ]
    out = _render_queue(queue)
    # Both rows make it onto the output.
    assert " 3 " in out  # uid 3
    assert " 7 " in out  # uid 7
    assert "100" in out
    assert "200" in out
