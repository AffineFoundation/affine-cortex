"""
Rank display.

``af get-rank`` renders the public ranking surface as a single
champion-challenge table backed by the read-only API.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from affine.utils.api_client import cli_api_client


_RANK_FETCH_LIMIT = 256
_QUEUE_PREVIEW = 10


def _is_color_tty() -> bool:
    try:
        return sys.stdout.isatty() and os.getenv("NO_COLOR") is None
    except Exception:
        return False


def _ansi(text: str, code: str) -> str:
    if not _is_color_tty():
        return text
    return f"\033[{code}m{text}\033[0m"


async def _fetch_rank_payload(client) -> Dict[str, Any]:
    payload = await client.get(
        f"/rank/current?top={_RANK_FETCH_LIMIT}&queue_limit={_QUEUE_PREVIEW}",
    )
    if isinstance(payload, dict) and isinstance(payload.get("scores"), dict):
        return payload
    return payload if isinstance(payload, dict) else {}


def _short(value: Any, n: int) -> str:
    text = "" if value is None else str(value)
    return text[:n]


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_relative_time(epoch_seconds: Optional[int]) -> str:
    if not epoch_seconds:
        return "unknown"
    delta = int(time.time()) - int(epoch_seconds)
    if delta < 0:
        return "just now"
    if delta < 60:
        return f"{delta}s ago"
    if delta < 3600:
        return f"{delta // 60}m ago"
    if delta < 86400:
        hours, rem = divmod(delta, 3600)
        return f"{hours}h {rem // 60}m ago"
    return f"{delta // 86400}d ago"


def _format_iso(epoch_seconds: Optional[int]) -> str:
    if not epoch_seconds:
        return "unknown"
    return datetime.fromtimestamp(
        int(epoch_seconds), tz=timezone.utc,
    ).strftime("%Y-%m-%d %H:%M:%S UTC")


def _env_names(scores: List[Dict[str, Any]]) -> List[str]:
    envs: set[str] = set()
    for row in scores:
        scores_by_env = row.get("scores_by_env") or {}
        if isinstance(scores_by_env, dict):
            envs.update(str(k) for k in scores_by_env.keys())
    return sorted(envs)


def _env_cell(
    payload: Any,  # noqa: ARG001 — kept for signature stability with callers
    live_count: Optional[int] = None,
    live_avg: Optional[float] = None,
    *,
    champion_live_avg: Optional[float] = None,
) -> str:
    """Render one env cell as ``{score%}[{lower%},{upper%}]/{n}``.

    Data source is the live ``system_config['live_scores']`` cache only;
    the ``payload`` parameter (a ``scores_by_env[env]`` snapshot dict
    from the affine_scores table) is intentionally ignored — that
    snapshot is only refreshed on champion change, so it can be days
    stale for a miner that lost its last battle. Showing ``-`` is
    correct when the live cache has no entry for this miner.
    """
    if live_avg is None or live_count is None or live_count <= 0:
        return "-"
    if champion_live_avg is None:
        return f"{live_avg * 100:.2f}/{live_count}"
    from affine.src.scorer.comparator import (
        DEFAULT_MARGIN, DEFAULT_NOT_WORSE_TOLERANCE,
    )
    lower = champion_live_avg * (1.0 - DEFAULT_NOT_WORSE_TOLERANCE)
    upper = champion_live_avg + DEFAULT_MARGIN
    return (
        f"{live_avg * 100:.2f}"
        f"[{lower * 100:.2f},{upper * 100:.2f}]"
        f"/{live_count}"
    )


def _status_for(
    row: Dict[str, Any],
    *,
    champion_uid: Optional[int],
    battle_uid: Optional[int],
    queue_positions: Dict[int, int],
    co_champion_shares: Optional[Dict[int, float]] = None,
) -> str:
    uid = row.get("uid")
    if uid == champion_uid:
        return "CHAMPION"
    # Past champion still drawing a share of the on-chain weight. We
    # render this above the BATTLING / VALID / TERMINATED branches so
    # an active reward recipient is never hidden behind a queue marker.
    if co_champion_shares and uid in co_champion_shares:
        pct = int(round(co_champion_shares[uid] * 100))
        return f"CO {pct:>2d}%"
    # Terminated lifecycle state lives in miner_stats, not the current
    # miners snapshot.
    chal_status = str(row.get("challenge_status") or "")
    if chal_status == "terminated":
        return "TERMINATED"
    if row.get("is_valid") is False:
        reason = str(row.get("invalid_reason") or "invalid")
        return reason.split(":", 1)[0][:11]
    if uid == battle_uid:
        return "BATTLING"
    if uid in queue_positions:
        return f"QUEUE #{queue_positions[uid]}"
    return "VALID"


def _reason_for(row: Dict[str, Any], status: str) -> str:
    if status == "TERMINATED":
        return _short(row.get("termination_reason"), 18)
    if row.get("is_valid") is False:
        return _short(row.get("invalid_reason"), 18)
    return ""


def _colored_status(status: str, *, is_invalid: bool) -> str:
    text = f"{status:>11}"
    if status == "CHAMPION":
        return _ansi(text, "1;93")
    if status.startswith("CO "):  # past champion still drawing a share
        return _ansi(text, "1;93")
    if status == "BATTLING":
        return _ansi(text, "1;96")
    if status.startswith("QUEUE #"):
        return _ansi(text, "1;94")
    if status == "TERMINATED":
        return _ansi(text, "1;91")  # bright red — same as monitor-invalid
    if is_invalid:
        return _ansi(text, "1;91")
    if status == "VALID":
        return _ansi(text, "32")
    return text


def _sampling_mark(uid: Any, live_sampling_uids: set[int]) -> str:
    if uid in live_sampling_uids:
        return _ansi("⚡", "1;92")
    return "  "


def _sort_scores(
    rows: List[Dict[str, Any]],
    *,
    champion_uid: Optional[int],
    battle_uid: Optional[int],  # noqa: ARG001 — kept for signature stability
    queue_positions: Dict[int, int],  # noqa: ARG001 — kept for signature stability
    co_champion_uids: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """Champion at top, co-champions next, then active, then inactive.

    Buckets (each sorted by commit time ``first_block`` ASC):

      - **0 champion** (single row): the active title holder
      - **1 co-champions**: past champions still drawing a share of the
        on-chain weight via the reward-split feature — pinned high so
        operators see at a glance who's being paid
      - **2 active**: ``is_valid=true`` AND ``challenge_status != 'terminated'``
      - **3 inactive**: terminated, invalid (model check failure, no
        commit, blacklist, etc) — kept for visibility but pushed below
        the active set

    Within each bucket: ``first_block`` ASC (earliest committer first),
    then uid as a final tiebreaker. Rows missing ``first_block`` sink
    to the end of their bucket so they don't displace real
    chronological entries.
    """
    co_champion_uids = co_champion_uids or set()

    def key(row: Dict[str, Any]) -> tuple:
        uid = row.get("uid")
        is_champion = (uid == champion_uid) if champion_uid is not None else False
        is_co_champion = (
            (not is_champion) and uid in co_champion_uids
        )
        chal_status = str(row.get("challenge_status") or "")
        is_terminated = (chal_status == "terminated")
        is_invalid = (row.get("is_valid") is False)
        first_block = row.get("first_block")
        try:
            fb = int(first_block) if first_block is not None else None
        except (TypeError, ValueError):
            fb = None
        has_fb = fb is not None and fb > 0

        if is_champion:
            bucket = 0
        elif is_co_champion:
            bucket = 1
        elif is_terminated or is_invalid:
            bucket = 3
        else:
            bucket = 2
        return (
            bucket,
            (0, fb) if has_fb else (1, 0),
            int(uid) if isinstance(uid, int) else 9999,
        )

    return sorted(rows, key=key)


def _print_rank_table(
    window: Optional[Dict[str, Any]],
    queue: Optional[List[Dict[str, Any]]],
    scores_resp: Optional[Dict[str, Any]],
    *,
    show_reason: bool = False,
) -> None:
    if not scores_resp or not scores_resp.get("scores"):
        print("No scores found")
        return

    scores = list(scores_resp.get("scores") or [])
    envs = _env_names(scores)
    champion = (window or {}).get("champion") or {}
    battle = ((window or {}).get("battle") or {}).get("challenger") or {}
    live_champion_uid = champion.get("uid")
    champion_uid = live_champion_uid
    battle_uid = battle.get("uid")
    if champion_uid is None:
        weight_champions = [
            row for row in scores
            if _as_float(row.get("overall_score")) > 0.0
        ]
        if len(weight_champions) == 1:
            inferred = weight_champions[0]
            champion_uid = inferred.get("uid")
            live_champion_uid = champion_uid
            champion = {
                "uid": inferred.get("uid"),
                "hotkey": inferred.get("miner_hotkey"),
                "model": inferred.get("model"),
            }
    queue_positions = {
        int(row["uid"]): int(row.get("position") or idx + 1)
        for idx, row in enumerate(queue or [])
        if row.get("uid") is not None
    }
    live_sample_counts = (window or {}).get("sample_counts") or {}
    # Per-(uid, env) running average over the current refresh_block —
    # used to display battle subjects' real progress instead of the
    # last-decided snapshot's 0.00 placeholder. None for miners not
    # currently in the sampling set.
    live_sample_averages = (window or {}).get("sample_averages") or {}
    # Champion's per-env live averages — used to compute LIVE bracket
    # thresholds for challenger rows in ``_env_cell``. Without this, the
    # brackets fall back to the last-decided snapshot's stale values.
    champion_live_avgs: Dict[str, float] = (
        (live_sample_averages.get(str(champion_uid)) or {})
        if champion_uid is not None else {}
    )
    live_sampling_uids = {
        int(uid)
        for uid in ((window or {}).get("live_sampling_uids") or [])
        if isinstance(uid, int)
    }
    # Past-N champions currently sharing the on-chain weight. Map uid →
    # share (e.g. 0.20 for N=5) for fast lookup in row rendering and
    # bucket sorting. The active champion is included in the API payload
    # (it's one of the N) but we exclude it here so its row keeps the
    # plain CHAMPION status / weight rather than the CO-share label.
    co_champion_shares: Dict[int, float] = {}
    for entry in (window or {}).get("past_champions") or []:
        try:
            uid = int(entry.get("uid"))
            share = float(entry.get("share") or 0.0)
        except (TypeError, ValueError):
            continue
        if champion_uid is not None and uid == champion_uid:
            continue
        if share <= 0:
            continue
        co_champion_shares[uid] = share

    header_parts = ["Hotkey  ", " UID", "⚡| Model                    "]
    header_parts.extend(f"{env[:24]:>24}" for env in envs)
    header_parts.append("  Status   ")
    if show_reason:
        header_parts.append(" Reason           ")
    header_parts.append(" Weight ")
    header_line = " | ".join(header_parts)
    width = max(88, len(header_line))

    block_number = scores_resp.get("block_number")
    calculated_at = scores_resp.get("calculated_at")

    print(_ansi("=" * width, "2"))
    print(_ansi(f"CHAMPION CHALLENGE RANKING - Block {block_number}", "1;96"))
    print(
        f"Calculated: {_format_relative_time(calculated_at)} "
        f"({_format_iso(calculated_at)})"
    )
    if champion:
        champ_hk = _short(champion.get("hotkey"), 8)
        print(
            f"Champion:   UID {champion.get('uid')}  {champ_hk}...  "
            f"{champion.get('model', '-')}"
        )
    else:
        print("Champion:   (none)")
    # Reward-split: one summary line listing each past champion still
    # drawing a share. We render even when the only payee is the active
    # champion (so operators see the split is enabled) but format the
    # split level so it's distinct from the plain Champion: line above.
    past_champions = (window or {}).get("past_champions") or []
    if past_champions:
        total_share = sum(
            _as_float(entry.get("share")) for entry in past_champions
        )
        share_pct = (
            int(round((1.0 / len(past_champions)) * 100))
            if past_champions else 0
        )
        parts = []
        for entry in past_champions:
            uid = entry.get("uid")
            hk = _short(entry.get("hotkey"), 6)
            parts.append(f"UID {uid} {hk}…")
        print(
            f"Reward:     {len(past_champions)}-way split ({share_pct}% each, "
            f"total {total_share:.2f}) — {', '.join(parts)}"
        )
    if battle:
        battle_hk = _short(battle.get("hotkey"), 8)
        started = ((window or {}).get("battle") or {}).get("started_at_block")
        print(
            f"Battle:     UID {battle.get('uid')}  {battle_hk}...  "
            f"{battle.get('model', '-')}  started @ block {started}"
        )
    else:
        print("Battle:     idle")
    refresh = (window or {}).get("task_refresh_block")
    if refresh is not None:
        print(f"Task pool:  refreshed @ block {refresh}")
    print(_ansi("=" * width, "2"))
    print(_ansi(header_line, "1"))
    print(_ansi("-" * width, "2"))

    for row in _sort_scores(
        scores,
        champion_uid=champion_uid,
        battle_uid=battle_uid,
        queue_positions=queue_positions,
        co_champion_uids=set(co_champion_shares.keys()),
    ):
        status = _status_for(
            row,
            champion_uid=champion_uid,
            battle_uid=battle_uid,
            queue_positions=queue_positions,
            co_champion_shares=co_champion_shares,
        )
        row_parts = [
            f"{_short(row.get('miner_hotkey'), 8):8s}",
            f"{int(row.get('uid') or -1):4d}",
            f"{_sampling_mark(row.get('uid'), live_sampling_uids)}| "
            f"{_short(row.get('model'), 25):25s}",
        ]
        scores_by_env = row.get("scores_by_env") or {}
        uid_key = str(row.get("uid"))
        row_live_counts = live_sample_counts.get(uid_key) or {}
        row_live_avgs = live_sample_averages.get(uid_key) or {}
        is_champion_row = row.get("uid") == champion_uid
        for env in envs:
            live_count = row_live_counts.get(env)
            live_avg = row_live_avgs.get(env)
            # Only the *challenger* row needs live brackets — the
            # champion's own row shouldn't display thresholds against
            # itself.
            champ_live = (
                None if is_champion_row else champion_live_avgs.get(env)
            )
            row_parts.append(
                f"{_env_cell(scores_by_env.get(env), live_count, live_avg, champion_live_avg=champ_live):>24}"
            )
        row_parts.append(
            _colored_status(status, is_invalid=(row.get("is_valid") is False))
        )
        if show_reason:
            row_parts.append(f"{_reason_for(row, status):18s}")
        # ``overall_score`` is the snapshot's stale weight (1.0 for last
        # write's single champion). Override with the live split share
        # when this row is a current payee so the Weight column reflects
        # what the validator actually sets on chain.
        uid_int = row.get("uid")
        weight_to_show = (
            co_champion_shares[uid_int]
            if isinstance(uid_int, int) and uid_int in co_champion_shares
            else _as_float(row.get("overall_score"))
        )
        # Also override the active champion when a split is on (its
        # share is 1/N, not 1.0). past_champions includes the active
        # champion; look it up there.
        if (
            uid_int == champion_uid
            and (window or {}).get("past_champions")
        ):
            for entry in (window or {}).get("past_champions") or []:
                try:
                    if int(entry.get("uid")) == champion_uid:
                        weight_to_show = _as_float(entry.get("share"))
                        break
                except (TypeError, ValueError):
                    continue
        row_parts.append(f"{weight_to_show:>7.4f}")
        print(" | ".join(row_parts))

    print(_ansi("=" * width, "2"))
    total = len(scores)
    valid = sum(1 for row in scores if row.get("is_valid") is True)
    invalid = sum(1 for row in scores if row.get("is_valid") is False)
    queue_count = len(queue or [])
    print(
        f"Total: {total}  |  Champion: {champion_uid if champion_uid is not None else '-'}"
        f"  |  Battle: {battle_uid if battle_uid is not None else '-'}"
        f"  |  Queue head: {queue_count}  |  Valid: {valid}  |  Invalid: {invalid}"
    )
    print(f"Sampling: {_ansi('⚡', '1;92')} marks miners in the current live sampling set")
    print(_ansi("=" * width, "2"))

async def get_rank_command(*, show_reason: bool = False) -> None:
    try:
        async with cli_api_client() as client:
            payload = await _fetch_rank_payload(client)
    except Exception as e:
        print(f"Error: failed to fetch /rank/current: {e}", file=sys.stderr)
        sys.exit(1)

    _print_rank_table(
        payload.get("window") if isinstance(payload.get("window"), dict) else None,
        payload.get("queue") if isinstance(payload.get("queue"), list) else None,
        payload.get("scores") if isinstance(payload.get("scores"), dict) else None,
        show_reason=show_reason,
    )
