"""``af get-rank`` renderer."""

from __future__ import annotations

import asyncio
import io
from contextlib import redirect_stdout

import affine.src.miner.rank as rank
from affine.src.miner.rank import _print_rank_table


def _render_rank(window, queue, scores):
    buf = io.StringIO()
    with redirect_stdout(buf):
        _print_rank_table(window, queue, scores)
    return buf.getvalue()


class _FakeClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    async def get(self, path):
        self.calls.append(path)
        response = self.responses[path]
        if isinstance(response, Exception):
            raise response
        return response


class _FakeClientContext:
    def __init__(self, client):
        self.client = client

    async def __aenter__(self):
        return self.client

    async def __aexit__(self, exc_type, exc, tb):
        return False


def test_rank_table_renders_old_single_table_shape_with_sampling_marks():
    window = {
        "champion": {
            "uid": 1,
            "hotkey": "champ_hotkey",
            "revision": "rc",
            "model": "org/champion",
        },
        "battle": {
            "challenger": {
                "uid": 2,
                "hotkey": "challenger_hotkey",
                "revision": "rh",
                "model": "org/challenger",
            },
            "started_at_block": 42,
        },
        "task_refresh_block": 40,
        "sample_counts": {"1": {"SWE": 300}, "2": {"SWE": 77}},
    }
    queue = [
        {"position": 1, "uid": 3, "hotkey": "queued", "revision": "rq", "model": "org/q"},
    ]
    scores = {
        "block_number": 100,
        "calculated_at": 0,
        "scores": [
            {
                "uid": 1,
                "miner_hotkey": "champ_hotkey",
                "model": "org/champion",
                "overall_score": 1.0,
                "is_valid": True,
                "scores_by_env": {"SWE": {"score": 0.8, "sample_count": 10}},
            },
            {
                "uid": 2,
                "miner_hotkey": "challenger_hotkey",
                "model": "org/challenger",
                "overall_score": 0.0,
                "is_valid": True,
                "scores_by_env": {
                    "SWE": {
                        "score": 0.7,
                        "score_on_common": 0.7,
                        "common_tasks": 9,
                        "sample_count": 9,
                        "not_worse_threshold": 0.79,
                        "dethrone_threshold": 0.81,
                    }
                },
            },
            {
                "uid": 3,
                "miner_hotkey": "queued",
                "model": "org/q",
                "overall_score": 0.0,
                "is_valid": True,
                "scores_by_env": {"SWE": {"score": 0.6, "sample_count": 8}},
            },
        ],
    }

    out = _render_rank(window, queue, scores)

    assert "CHAMPION CHALLENGE RANKING - Block 100" in out
    assert "Champion:   UID 1" in out
    assert "Battle:     UID 2" in out
    assert "Hotkey" in out
    assert "⚡| Model" in out
    assert " Valid " not in out
    assert "CHAMPION" in out
    assert "BATTLING" in out
    assert "QUEUE #1" in out
    assert "80.00/300" in out
    assert "70.00[79.00,81.00]/9" in out
    assert out.count("⚡| org/") == 2
    assert "Sampling: ⚡ marks miners in the current live sampling set" in out


def test_rank_table_renders_empty_scores_safely():
    out = _render_rank(None, None, None)
    assert "No scores found" in out


def test_rank_table_groups_invalid_miners_below_valid_rows():
    scores = {
        "block_number": 100,
        "calculated_at": 0,
        "scores": [
            {
                "uid": 4,
                "miner_hotkey": "bad",
                "model": "org/bad",
                "overall_score": 0.0,
                "is_valid": False,
                "invalid_reason": "model_mismatch:detail",
                "scores_by_env": {},
            },
            {
                "uid": 5,
                "miner_hotkey": "good",
                "model": "org/good",
                "overall_score": 0.0,
                "is_valid": True,
                "scores_by_env": {},
            },
        ],
    }

    out = _render_rank(None, None, scores)

    assert out.index("good") < out.index("bad")
    assert "model_misma" in out


def test_rank_table_does_not_show_unknown_status_for_missing_validity():
    scores = {
        "block_number": 100,
        "calculated_at": 0,
        "scores": [
            {
                "uid": 4,
                "miner_hotkey": "hk",
                "model": "org/model",
                "overall_score": 0.0,
                "is_valid": None,
                "scores_by_env": {},
            },
        ],
    }

    out = _render_rank(None, None, scores)

    assert "UNKNOWN" not in out
    assert "VALID" in out


def test_rank_table_infers_active_champion_from_weight_when_window_is_empty():
    scores = {
        "block_number": 100,
        "calculated_at": 0,
        "scores": [
            {
                "uid": 7,
                "miner_hotkey": "champ",
                "model": "org/champ",
                "overall_score": 1.0,
                "is_valid": True,
                "scores_by_env": {},
            },
        ],
    }

    out = _render_rank({"champion": None, "battle": None}, [], scores)

    assert "Champion:   UID 7" in out
    assert "CHAMPION" in out
    assert "⚡| org/champ" in out


def test_rank_table_uses_live_count_only_for_non_threshold_cells():
    window = {
        "champion": {"uid": 1, "hotkey": "champ", "model": "org/champ"},
        "battle": None,
        "sample_counts": {"1": {"SWE": 321}},
    }
    scores = {
        "block_number": 100,
        "calculated_at": 0,
        "scores": [
            {
                "uid": 1,
                "miner_hotkey": "champ",
                "model": "org/champ",
                "overall_score": 1.0,
                "is_valid": True,
                "scores_by_env": {
                    "SWE": {
                        "score": 0.8,
                        "score_on_common": 0.8,
                        "common_tasks": 10,
                        "sample_count": 10,
                        "not_worse_threshold": 0.7,
                        "dethrone_threshold": 0.9,
                    }
                },
            },
        ],
    }

    out = _render_rank(window, [], scores)

    assert "80.00[70.00,90.00]/10" in out
    assert "80.00[70.00,90.00]/321" not in out


def test_rank_table_uses_color_on_tty(monkeypatch):
    class _TTYBuffer(io.StringIO):
        def isatty(self):
            return True

    scores = {
        "block_number": 100,
        "calculated_at": 0,
        "scores": [
            {
                "uid": 1,
                "miner_hotkey": "champ_hotkey",
                "model": "org/champion",
                "overall_score": 1.0,
                "is_valid": True,
                "scores_by_env": {},
            },
        ],
    }

    buf = _TTYBuffer()
    monkeypatch.delenv("NO_COLOR", raising=False)
    with redirect_stdout(buf):
        rank._print_rank_table(
            {"champion": {"uid": 1, "hotkey": "champ_hotkey", "model": "org/champion"}},
            [],
            scores,
        )

    out = buf.getvalue()
    assert "\033[" in out
    assert "\033[1;93m   CHAMPION\033[0m" in out


def test_rank_table_respects_no_color(monkeypatch):
    class _TTYBuffer(io.StringIO):
        def isatty(self):
            return True

    scores = {
        "block_number": 100,
        "calculated_at": 0,
        "scores": [
            {
                "uid": 1,
                "miner_hotkey": "champ_hotkey",
                "model": "org/champion",
                "overall_score": 1.0,
                "is_valid": True,
                "scores_by_env": {},
            },
        ],
    }

    buf = _TTYBuffer()
    monkeypatch.setenv("NO_COLOR", "1")
    with redirect_stdout(buf):
        rank._print_rank_table(
            {"champion": {"uid": 1, "hotkey": "champ_hotkey", "model": "org/champion"}},
            [],
            scores,
        )

    assert "\033[" not in buf.getvalue()


def test_get_rank_falls_back_to_scores_when_aggregate_endpoint_is_missing(monkeypatch):
    scores = {
        "block_number": 100,
        "calculated_at": 0,
        "scores": [
            {
                "uid": 1,
                "miner_hotkey": "champ_hotkey",
                "model": "org/champion",
                "overall_score": 1.0,
                "is_valid": True,
                "scores_by_env": {},
            },
        ],
    }
    client = _FakeClient({
        "/rank/current?top=256&queue_limit=10": RuntimeError("HTTP 404"),
        "/scores/latest?top=256": scores,
    })
    monkeypatch.setattr(rank, "cli_api_client", lambda: _FakeClientContext(client))

    buf = io.StringIO()
    with redirect_stdout(buf):
        asyncio.run(rank.get_rank_command())

    assert client.calls == [
        "/rank/current?top=256&queue_limit=10",
        "/scores/latest?top=256",
    ]
    out = buf.getvalue()
    assert "No scores found" not in out
    assert "CHAMPION CHALLENGE RANKING - Block 100" in out
    assert "CHAMPION" in out
