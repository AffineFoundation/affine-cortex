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
        "sample_averages": {"1": {"SWE": 0.80}, "2": {"SWE": 0.70}},
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
    # Champion row: live count + live avg, no brackets (champion never
    # shows thresholds against itself).
    assert "80.00/300" in out
    # Challenger row: brackets computed from champion's live_avg (0.80)
    # via DEFAULT_NOT_WORSE_TOLERANCE=0.02 and DEFAULT_MARGIN=0.03 →
    # [0.784, 0.830], score is challenger's live_avg, count is its
    # live_count from sample_counts.
    assert "70.00[78.40,83.00]/77" in out
    assert out.count("⚡| org/") == 2
    assert "  | org/q" in out
    assert "Sampling: ⚡ marks miners in the current live sampling set" in out
    assert "Queue: #1 UID 3" not in out


def test_rank_table_renders_empty_scores_safely():
    out = _render_rank(None, None, None)
    assert "No scores found" in out


def test_rank_table_invalid_reason_is_rendered_but_truncated():
    """Invalid-miner ``invalid_reason`` shows up in the status column,
    truncated to fit the column width (no leak of the full message)."""
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
                "first_block": 100,
                "scores_by_env": {},
            },
            {
                "uid": 5,
                "miner_hotkey": "good",
                "model": "org/good",
                "overall_score": 0.0,
                "is_valid": True,
                "first_block": 200,
                "scores_by_env": {},
            },
        ],
    }

    out = _render_rank(None, None, scores)

    assert "model_misma" in out
    assert "model_mismatch:de" not in out


def test_rank_table_sorts_champion_active_inactive_buckets_by_commit_time():
    """Three buckets in order, each sorted by ``first_block`` ASC:

      1. champion (pinned)
      2. active rows (``is_valid=true`` and not terminated)
      3. inactive rows (terminated or ``is_valid=false`` — model check
         failure, no commit, blacklist, etc.)

    Within a bucket, earlier commits win. Rows missing ``first_block``
    sink to the bottom of their own bucket.
    """
    window = {
        "champion": {"uid": 9, "hotkey": "hkChamp", "model": "org/champ"},
    }
    scores = {
        "block_number": 100,
        "calculated_at": 0,
        "scores": [
            {
                "uid": 9, "miner_hotkey": "hkChamp", "model": "org/champ",
                "overall_score": 1.0, "is_valid": True,
                "first_block": 500, "scores_by_env": {},
            },
            # Active (valid, not terminated). first_block 50 / 250 / none.
            {
                "uid": 3, "miner_hotkey": "hkEarly", "model": "org/early",
                "overall_score": 0.0, "is_valid": True,
                "first_block": 50, "scores_by_env": {},
            },
            {
                "uid": 6, "miner_hotkey": "hkLate", "model": "org/late",
                "overall_score": 0.0, "is_valid": True,
                "first_block": 250, "scores_by_env": {},
            },
            {
                "uid": 7, "miner_hotkey": "hkNoFb", "model": "org/nofb",
                "overall_score": 0.0, "is_valid": True,
                "scores_by_env": {},
            },
            # Inactive: terminated and invalid. first_block 75 / 80.
            {
                "uid": 4, "miner_hotkey": "hkTerm", "model": "org/term",
                "overall_score": 0.0, "is_valid": False,
                "invalid_reason": "hf_model_fetch_failed",
                "challenge_status": "terminated",
                "termination_reason": "lost_to_champion:abc",
                "first_block": 75, "scores_by_env": {},
            },
            {
                "uid": 2, "miner_hotkey": "hkInval", "model": "org/inval",
                "overall_score": 0.0, "is_valid": False,
                "invalid_reason": "no_commit",
                "first_block": 80, "scores_by_env": {},
            },
        ],
    }

    out = _render_rank(window, None, scores)

    # Bucket 1: champion (uid 9) first.
    assert out.index("hkChamp") < out.index("hkEarly")
    # Bucket 2: active rows by first_block ASC, no-first_block row at
    # the bottom of the active group.
    assert out.index("hkEarly") < out.index("hkLate")
    assert out.index("hkLate") < out.index("hkNoFb")
    # Bucket 3: every inactive row comes after every active row.
    assert out.index("hkNoFb") < out.index("hkTerm")
    assert out.index("hkTerm") < out.index("hkInval")
    # Terminated / invalid status still renders in the status column.
    assert "TERMINATED" in out
    # Truncated termination reason should NOT leak into other columns.
    assert "lost_to_champion:" not in out


def test_rank_table_renders_past_champions_split():
    """When the reward-split feature is on, ``af get-rank`` should:

    * print a ``Reward:`` summary line listing each past-N champion +
      their share %.
    * mark each *non-active* past champion with status ``CO {pct}%``.
    * show the share (1/N) in the Weight column instead of the
      snapshot's stale 0.0 for past-champion rows AND for the active
      champion row (its on-chain weight is also 1/N now, not 1.0).
    * pin co-champions in bucket 2 (right after the active champion),
      above the rest of the valid set.
    """
    window = {
        "champion": {"uid": 9, "hotkey": "hkChamp", "model": "org/champ"},
        "past_champions": [
            # Active champion is always one of the N — included so the
            # frontend can display the full split in one place.
            {"uid": 9, "hotkey": "hkChamp",
             "model": "org/champ", "share": 0.5},
            {"uid": 3, "hotkey": "hkPast",
             "model": "org/past", "share": 0.5},
        ],
    }
    scores = {
        "block_number": 100,
        "calculated_at": 0,
        "scores": [
            {
                "uid": 9, "miner_hotkey": "hkChamp", "model": "org/champ",
                "overall_score": 1.0, "is_valid": True,
                "first_block": 500, "scores_by_env": {},
            },
            # Past champion. Its scores-table overall_score is 0.0
            # (snapshot was winner-takes-all) — we expect the Weight
            # column to surface the live 0.5 share instead.
            {
                "uid": 3, "miner_hotkey": "hkPast", "model": "org/past",
                "overall_score": 0.0, "is_valid": True,
                "first_block": 300, "scores_by_env": {},
            },
            # Plain valid miner — should sort below the co-champion.
            {
                "uid": 7, "miner_hotkey": "hkValid", "model": "org/v",
                "overall_score": 0.0, "is_valid": True,
                "first_block": 100, "scores_by_env": {},
            },
        ],
    }

    out = _render_rank(window, None, scores)

    # Summary header line.
    assert "Reward:" in out
    assert "2-way split (50% each" in out

    # The hotkeys also appear in the "Champion:" / "Reward:" header
    # block above the table. Pick the *table* row for each hotkey by
    # scanning lines and taking the first one that starts with the
    # hotkey (header lines start with "Champion:" / "Reward:" /
    # column-padding spaces, so the row line is the first match).
    def _row(out_text: str, hotkey: str) -> str:
        for line in out_text.splitlines():
            if line.startswith(hotkey + " "):
                return line
        raise AssertionError(f"row for {hotkey!r} not found in:\n{out_text}")

    champ_row = _row(out, "hkChamp")
    past_row = _row(out, "hkPast")
    valid_row = _row(out, "hkValid")

    # Champion's Weight column shows 0.5 (= split share), not the stale
    # snapshot's 1.0.
    assert "CHAMPION" in champ_row
    assert "0.5000" in champ_row
    # Past champion's status reads "CO 50%" and Weight is 0.5 (not 0).
    assert "CO 50%" in past_row
    assert "0.5000" in past_row
    # Plain valid row keeps its 0.0 overall_score.
    assert "0.0000" in valid_row

    # Sort order in the table: champion → co-champion → plain valid
    # (even though plain valid has the earlier first_block).
    lines = out.splitlines()
    champ_line_no = next(i for i, l in enumerate(lines) if l.startswith("hkChamp "))
    past_line_no = next(i for i, l in enumerate(lines) if l.startswith("hkPast "))
    valid_line_no = next(i for i, l in enumerate(lines) if l.startswith("hkValid "))
    assert champ_line_no < past_line_no < valid_line_no


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
        "sample_averages": {"1": {"SWE": 0.80}},
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

    # Champion row: live_avg + live_count from window; no brackets
    # (champion never shows thresholds against itself).
    assert "80.00/321" in out
    # Snapshot's own threshold metadata (not_worse_threshold,
    # dethrone_threshold) is never used for bracket rendering — only
    # the *champion's* live_avg drives brackets, and a champion row has
    # no brackets at all.
    assert "[70.00,90.00]" not in out


def test_rank_table_keeps_thresholds_when_using_live_running_average():
    # Champion live_avg drives the bracket math:
    #   lower = champ * (1 - DEFAULT_NOT_WORSE_TOLERANCE=0.02) = 0.4851
    #   upper = champ + DEFAULT_MARGIN=0.03                   = 0.5249
    window = {
        "champion": {"uid": 1, "hotkey": "champ", "model": "org/champ"},
        "battle": {
            "challenger": {"uid": 2, "hotkey": "chal", "model": "org/chal"},
            "started_at_block": 42,
        },
        "sample_counts": {"1": {"SWE": 213}, "2": {"SWE": 178}},
        "sample_averages": {"1": {"SWE": 0.4950}, "2": {"SWE": 0.4816}},
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
                "scores_by_env": {"SWE": {}},
            },
        ],
    }

    out = _render_rank(window, [], scores)

    assert "48.16[48.51,52.50]/178" in out


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


def test_rank_table_omits_brackets_when_no_live_champion_avg():
    """When the champion has no ``live_avg`` for this env (eg between
    battles, championship was just inferred from weights, or the
    live_scores cache hasn't been populated yet), the cell shows
    just ``score/count`` with no brackets. The snapshot's
    ``not_worse_threshold`` / ``dethrone_threshold`` fields are
    intentionally NOT consumed — they go stale across champion
    changes and would mislead readers."""
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
    assert "48.16/178" in out
    # Snapshot brackets are NOT fallback-rendered.
    assert "[48.49,52.48]" not in out


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
