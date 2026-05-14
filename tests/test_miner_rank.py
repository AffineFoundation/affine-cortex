"""``af get-rank`` renderer."""

from __future__ import annotations

import asyncio
import io
from contextlib import redirect_stderr, redirect_stdout

import affine.src.miner.rank as rank
from affine.src.miner.rank import _print_rank_table


def _render_rank(window, queue, scores, *, show_reason=False):
    buf = io.StringIO()
    with redirect_stdout(buf):
        _print_rank_table(window, queue, scores, show_reason=show_reason)
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
        "live_sampling_uids": [1, 2],
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
    assert "Reason" not in out
    assert " Valid " not in out
    assert "CHAMPION" in out
    assert "BATTLING" in out
    assert "QUEUE #1" in out
    assert "80.00/300" in out
    assert "70.00[79.00,81.00]/9" in out
    assert out.count("⚡| org/") == 2
    assert "  | org/q" in out
    assert "Sampling: ⚡ marks miners in the current live sampling set" in out
    assert "Queue: #1 UID 3" not in out


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
    assert "model_mismatch:de" not in out


def test_rank_table_sorts_miners_by_status_then_uid_not_score():
    scores = {
        "block_number": 100,
        "calculated_at": 0,
        "scores": [
            {
                "uid": 9,
                "miner_hotkey": "uid9",
                "model": "org/uid9",
                "overall_score": 0.99,
                "is_valid": True,
                "scores_by_env": {},
            },
            {
                "uid": 3,
                "miner_hotkey": "uid3",
                "model": "org/uid3",
                "overall_score": 0.01,
                "is_valid": True,
                "scores_by_env": {},
            },
            {
                "uid": 4,
                "miner_hotkey": "terminated",
                "model": "org/terminated",
                "overall_score": 1.0,
                "is_valid": False,
                "invalid_reason": "hf_model_fetch_failed",
                "challenge_status": "terminated",
                "termination_reason": "lost_to_champion:abc",
                "scores_by_env": {},
            },
            {
                "uid": 2,
                "miner_hotkey": "invalid",
                "model": "org/invalid",
                "overall_score": 1.0,
                "is_valid": False,
                "invalid_reason": "model_check:bad",
                "scores_by_env": {},
            },
        ],
    }

    out = _render_rank(None, None, scores)

    assert out.index("uid3") < out.index("uid9")
    assert out.index("uid9") < out.index("invalid")
    assert out.index("uid9") < out.index("terminated")
    assert out.index("terminated") < out.index("invalid")
    assert "TERMINATED" in out
    assert "lost_to_champion:" not in out


def test_rank_table_can_show_reason_column_when_requested():
    scores = {
        "block_number": 100,
        "calculated_at": 0,
        "scores": [
            {
                "uid": 4,
                "miner_hotkey": "terminated",
                "model": "org/terminated",
                "overall_score": 1.0,
                "is_valid": False,
                "invalid_reason": "hf_model_fetch_failed",
                "challenge_status": "terminated",
                "termination_reason": "lost_to_champion:abc",
                "scores_by_env": {},
            },
            {
                "uid": 2,
                "miner_hotkey": "invalid",
                "model": "org/invalid",
                "overall_score": 1.0,
                "is_valid": False,
                "invalid_reason": "model_mismatch:detail",
                "scores_by_env": {},
            },
        ],
    }

    out = _render_rank(None, None, scores, show_reason=True)

    assert "Reason" in out
    assert "lost_to_champion:" in out
    assert "model_mismatch:de" in out


def test_rank_table_shows_terminated_for_terminated_status():
    """miner_stats challenge_status='terminated' renders TERMINATED."""
    scores = {
        "block_number": 100,
        "calculated_at": 0,
        "scores": [
            {
                "uid": 5,
                "miner_hotkey": "lost-prior",
                "model": "org/lost",
                "overall_score": 0.0,
                "is_valid": True,
                "challenge_status": "terminated",
                "termination_reason": "lost_to_champion:5GepM",
                "scores_by_env": {},
            },
            {
                "uid": 6,
                "miner_hotkey": "good",
                "model": "org/good",
                "overall_score": 0.0,
                "is_valid": True,
                "challenge_status": "sampling",
                "scores_by_env": {},
            },
        ],
    }

    out = _render_rank(None, None, scores)

    assert "TERMINATED" in out
    # Terminated row is rendered; UID column shows it's our miner.
    assert "5" in out


def test_rank_table_terminated_status_renders_terminated():
    scores = {
        "block_number": 100,
        "calculated_at": 0,
        "scores": [
            {
                "uid": 5,
                "miner_hotkey": "ex-champ",
                "model": "org/ex",
                "overall_score": 0.0,
                "is_valid": True,
                "challenge_status": "terminated",
                "scores_by_env": {},
            },
        ],
    }

    out = _render_rank(None, None, scores)
    assert "TERMINATED" in out


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
    assert "⚡| org/champ" not in out


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


def test_rank_table_keeps_thresholds_when_using_live_running_average():
    window = {
        "champion": {"uid": 1, "hotkey": "champ", "model": "org/champ"},
        "battle": {
            "challenger": {"uid": 2, "hotkey": "chal", "model": "org/chal"},
            "started_at_block": 42,
        },
        "sample_counts": {"2": {"SWE": 178}},
        "sample_averages": {"2": {"SWE": 0.4816}},
    }
    scores = {
        "block_number": 100,
        "calculated_at": 0,
        "scores": [
            {
                "uid": 2,
                "miner_hotkey": "chal",
                "model": "org/chal",
                "overall_score": 0.0,
                "is_valid": True,
                "scores_by_env": {
                    "SWE": {
                        "score": 0.0,
                        "score_on_common": 0.0,
                        "common_tasks": 0,
                        "not_worse_threshold": 0.4849,
                        "dethrone_threshold": 0.5248,
                    },
                },
            },
        ],
    }

    out = _render_rank(window, [], scores)

    assert "48.16[48.49,52.48]/178" in out


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


def test_get_rank_reports_aggregate_endpoint_errors_without_fallback(monkeypatch):
    client = _FakeClient({
        "/rank/current?top=256&queue_limit=10": RuntimeError("HTTP 404"),
    })
    monkeypatch.setattr(rank, "cli_api_client", lambda: _FakeClientContext(client))

    err = io.StringIO()
    with redirect_stderr(err):
        try:
            asyncio.run(rank.get_rank_command())
        except SystemExit as exc:
            assert exc.code == 1
        else:
            raise AssertionError("expected get_rank_command to exit")

    assert client.calls == ["/rank/current?top=256&queue_limit=10"]
    assert "failed to fetch /rank/current" in err.getvalue()


# ---- live-bracket thresholds (replaces stale snapshot thresholds) ---------
#
# The snapshot's ``not_worse_threshold`` / ``dethrone_threshold`` are
# baked from the LAST DECIDED battle. When a new battle is mid-flight
# against a stronger champion, those numbers can be wildly off — the
# challenger row would mix a live ``score_on_common`` with stale
# brackets, suggesting an easy win that isn't actually happening.
# The fix overrides the brackets with values computed live from the
# CURRENT champion's per-env average whenever that average is present
# in ``sample_averages``. See ``affine/src/scorer/comparator.py`` for
# ``DEFAULT_MARGIN`` / ``DEFAULT_NOT_WORSE_TOLERANCE``.


def test_rank_table_uses_live_brackets_from_current_champion_average():
    """Challenger row's brackets must come from CURRENT champion's
    live_avg (champion_live × 0.98, champion_live + 0.03), not from
    the stale snapshot ``not_worse_threshold`` / ``dethrone_threshold``
    that was written by the last decided battle.
    """
    window = {
        "champion": {"uid": 1, "hotkey": "champ", "model": "org/champ"},
        "battle": {
            "challenger": {"uid": 2, "hotkey": "chal", "model": "org/chal"},
            "started_at_block": 42,
        },
        "sample_counts": {
            "1": {"SWE": 213},
            "2": {"SWE": 78},
        },
        "sample_averages": {
            "1": {"SWE": 0.5455},   # current champion's live SWE avg
            "2": {"SWE": 0.5273},   # current challenger's live SWE avg
        },
    }
    scores = {
        "block_number": 100,
        "calculated_at": 0,
        "scores": [
            {
                "uid": 1, "miner_hotkey": "champ", "model": "org/champ",
                "overall_score": 1.0, "is_valid": True,
                "scores_by_env": {"SWE": {"score": 0.3662, "sample_count": 213}},
            },
            {
                "uid": 2, "miner_hotkey": "chal", "model": "org/chal",
                "overall_score": 0.0, "is_valid": True,
                "scores_by_env": {
                    "SWE": {
                        # Stale snapshot brackets from a prior battle
                        # against a much weaker champion.
                        "score": 0.0, "score_on_common": 0.0,
                        "common_tasks": 0,
                        "not_worse_threshold": 0.1983,
                        "dethrone_threshold": 0.2323,
                    },
                },
            },
        ],
    }

    out = _render_rank(window, [], scores)

    # Live brackets = champion_live (0.5455) × 0.98 and + 0.03.
    # not_worse = 53.46, dethrone = 57.55. NOT the stale 19.83 / 23.23.
    assert "52.73[53.46,57.55]/78" in out
    assert "[19.83," not in out, (
        "stale snapshot brackets must not appear when current champion "
        "has a live average"
    )


def test_rank_table_falls_back_to_snapshot_brackets_when_no_live_champion_avg():
    """Backward-compat: when the champion has no live_avg for this env
    (eg between battles, or championship was just inferred from
    weights), keep using the snapshot ``not_worse_threshold`` /
    ``dethrone_threshold`` rather than dropping them silently."""
    window = {
        "champion": {"uid": 1, "hotkey": "champ", "model": "org/champ"},
        "battle": {
            "challenger": {"uid": 2, "hotkey": "chal", "model": "org/chal"},
            "started_at_block": 42,
        },
        # Champion has no sample_averages entry — only challenger does.
        "sample_counts": {"2": {"SWE": 178}},
        "sample_averages": {"2": {"SWE": 0.4816}},
    }
    scores = {
        "block_number": 100,
        "calculated_at": 0,
        "scores": [
            {
                "uid": 2, "miner_hotkey": "chal", "model": "org/chal",
                "overall_score": 0.0, "is_valid": True,
                "scores_by_env": {
                    "SWE": {
                        "score": 0.0, "score_on_common": 0.0,
                        "common_tasks": 0,
                        "not_worse_threshold": 0.4849,
                        "dethrone_threshold": 0.5248,
                    },
                },
            },
        ],
    }

    out = _render_rank(window, [], scores)
    assert "48.16[48.49,52.48]/178" in out


def test_rank_table_champion_row_never_displays_live_brackets():
    """The champion's own row must NOT show live brackets — we'd be
    rendering thresholds against itself. Champion's payload typically
    has no ``not_worse_threshold`` either; the cell stays as plain
    ``score/count``."""
    window = {
        "champion": {"uid": 1, "hotkey": "champ", "model": "org/champ"},
        "battle": {
            "challenger": {"uid": 2, "hotkey": "chal", "model": "org/chal"},
            "started_at_block": 42,
        },
        "sample_counts": {"1": {"SWE": 213}, "2": {"SWE": 78}},
        "sample_averages": {"1": {"SWE": 0.5455}, "2": {"SWE": 0.5273}},
    }
    scores = {
        "block_number": 100,
        "calculated_at": 0,
        "scores": [
            {
                "uid": 1, "miner_hotkey": "champ", "model": "org/champ",
                "overall_score": 1.0, "is_valid": True,
                "scores_by_env": {"SWE": {"score": 0.5455, "sample_count": 213}},
            },
        ],
    }

    out = _render_rank(window, [], scores)
    # Champion row uses its live_avg (54.55) and live_count (213), no
    # brackets injected from its own average.
    assert "54.55/213" in out
    assert "[" not in out.split("|")[3]  # SWE cell has no brackets
