"""``af get-rank`` renderer."""

from __future__ import annotations

import io
from contextlib import redirect_stdout

from affine.src.miner.rank import _print_queue, _print_rank_table


def _render_rank(window, queue, scores):
    buf = io.StringIO()
    with redirect_stdout(buf):
        _print_rank_table(window, queue, scores)
    return buf.getvalue()


def _render_queue(queue):
    buf = io.StringIO()
    with redirect_stdout(buf):
        _print_queue(queue)
    return buf.getvalue()


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
    assert "CHAMPION" in out
    assert "BATTLING" in out
    assert "QUEUE #1" in out
    assert "80.00/10" in out
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
    assert "     -" in out


def test_rank_table_infers_champion_from_weight_without_sampling_mark():
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


def test_queue_renders_none_safely():
    out = _render_queue(None)
    assert "queue empty" in out


def test_queue_renders_pending_entries():
    queue = [
        {"position": 1, "uid": 3, "hotkey": "hk_three", "revision": "r3", "model": "o/m3",
         "first_block": 100, "enqueued_at": 1000},
        {"position": 2, "uid": 7, "hotkey": "hk_seven", "revision": "r7", "model": "o/m7",
         "first_block": 200, "enqueued_at": 1100},
    ]
    out = _render_queue(queue)
    assert " 3 " in out
    assert " 7 " in out
    assert "100" in out
    assert "200" in out
