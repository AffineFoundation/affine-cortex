"""
Rank display.

``af get-rank`` renders the public ranking surface as a single
champion-challenge table backed by the read-only API.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from affine.core.setup import logger
from affine.utils.api_client import cli_api_client


_RANK_FETCH_LIMIT = 256
_QUEUE_PREVIEW = 10


async def _safe_get(client, path: str) -> Optional[Any]:
    try:
        return await client.get(path)
    except Exception as e:
        logger.warning(f"rank: GET {path} failed: {type(e).__name__}: {e}")
        return None


def _short(value: Any, n: int) -> str:
    text = "" if value is None else str(value)
    return text[:n]


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _number(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


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


def _env_cell(payload: Any) -> str:
    if not isinstance(payload, dict):
        return "-"
    score_on_common = _number(payload.get("score_on_common"))
    lower = _number(payload.get("not_worse_threshold"))
    upper = _number(payload.get("dethrone_threshold"))
    common_tasks = payload.get("common_tasks")
    if (
        score_on_common is not None
        and lower is not None
        and upper is not None
        and isinstance(common_tasks, int)
    ):
        return (
            f"{score_on_common * 100:.2f}"
            f"[{lower * 100:.2f},{upper * 100:.2f}]"
            f"/{common_tasks}"
        )

    score = None
    for key in ("score", "mean", "average_score", "score_on_common"):
        score = _number(payload.get(key))
        if score is not None:
            break
    if score is None:
        return "-"
    samples = None
    for key in ("sample_count", "historical_count", "common_tasks", "count"):
        if isinstance(payload.get(key), int):
            samples = int(payload[key])
            break
    if samples is None:
        return f"{score * 100:.2f}"
    return f"{score * 100:.2f}/{samples}"


def _valid_mark(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "-"


def _status_for(
    row: Dict[str, Any],
    *,
    champion_uid: Optional[int],
    battle_uid: Optional[int],
    queue_positions: Dict[int, int],
) -> str:
    uid = row.get("uid")
    if uid == champion_uid:
        return "CHAMPION"
    if row.get("is_valid") is False:
        reason = str(row.get("invalid_reason") or "invalid")
        return reason.split(":", 1)[0][:11]
    if uid == battle_uid:
        return "BATTLING"
    if uid in queue_positions:
        return f"QUEUE #{queue_positions[uid]}"
    return "VALID"


def _sampling_mark(uid: Any, champion_uid: Optional[int], battle_uid: Optional[int]) -> str:
    if uid == champion_uid or uid == battle_uid:
        return "⚡"
    return " "


def _sort_scores(
    rows: List[Dict[str, Any]],
    *,
    champion_uid: Optional[int],
    battle_uid: Optional[int],
    queue_positions: Dict[int, int],
) -> List[Dict[str, Any]]:
    def key(row: Dict[str, Any]) -> tuple:
        uid = row.get("uid")
        if uid == champion_uid:
            bucket = 0
        elif uid == battle_uid:
            bucket = 1
        elif uid in queue_positions:
            bucket = 2
        elif row.get("is_valid") is False:
            bucket = 4
        else:
            bucket = 3
        return (
            bucket,
            queue_positions.get(uid, 9999),
            -_as_float(row.get("overall_score")),
            int(uid if isinstance(uid, int) else 9999),
        )

    return sorted(rows, key=key)


def _print_rank_table(
    window: Optional[Dict[str, Any]],
    queue: Optional[List[Dict[str, Any]]],
    scores_resp: Optional[Dict[str, Any]],
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

    header_parts = ["Hotkey  ", " UID", "⚡| Model                    "]
    header_parts.extend(f"{env[:24]:>24}" for env in envs)
    header_parts.extend(["  Status   ", " Weight ", " Valid "])
    header_line = " | ".join(header_parts)
    width = max(88, len(header_line))

    block_number = scores_resp.get("block_number")
    calculated_at = scores_resp.get("calculated_at")

    print("=" * width)
    print(f"CHAMPION CHALLENGE RANKING - Block {block_number}")
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
    print("=" * width)
    print(header_line)
    print("-" * width)

    for row in _sort_scores(
        scores,
        champion_uid=champion_uid,
        battle_uid=battle_uid,
        queue_positions=queue_positions,
    ):
        status = _status_for(
            row,
            champion_uid=champion_uid,
            battle_uid=battle_uid,
            queue_positions=queue_positions,
        )
        row_parts = [
            f"{_short(row.get('miner_hotkey'), 8):8s}",
            f"{int(row.get('uid') or -1):4d}",
            f"{_sampling_mark(row.get('uid'), live_champion_uid, battle_uid)}| "
            f"{_short(row.get('model'), 25):25s}",
        ]
        scores_by_env = row.get("scores_by_env") or {}
        for env in envs:
            row_parts.append(f"{_env_cell(scores_by_env.get(env)):>24}")
        row_parts.extend([
            f"{status:>11}",
            f"{_as_float(row.get('overall_score')):>7.4f}",
            f"{_valid_mark(row.get('is_valid')):>6}",
        ])
        print(" | ".join(row_parts))

    print("=" * width)
    total = len(scores)
    valid = sum(1 for row in scores if row.get("is_valid") is True)
    invalid = sum(1 for row in scores if row.get("is_valid") is False)
    queue_count = len(queue or [])
    print(
        f"Total: {total}  |  Champion: {champion_uid if champion_uid is not None else '-'}"
        f"  |  Battle: {battle_uid if battle_uid is not None else '-'}"
        f"  |  Queue head: {queue_count}  |  Valid: {valid}  |  Invalid: {invalid}"
    )
    print("Sampling: ⚡ marks miners in the current live sampling set")
    if queue:
        head = ", ".join(
            f"#{row.get('position')} UID {row.get('uid')}"
            for row in queue[:_QUEUE_PREVIEW]
        )
        print(f"Queue: {head}")
    print("=" * width)


# Compatibility helpers used by unit tests and older callers.
def _print_window(state: Optional[Dict[str, Any]]) -> None:
    scores = {
        "block_number": "-",
        "calculated_at": None,
        "scores": [],
    }
    champion = (state or {}).get("champion")
    battle = ((state or {}).get("battle") or {}).get("challenger")
    if champion:
        scores["scores"].append({
            "uid": champion.get("uid"),
            "miner_hotkey": champion.get("hotkey"),
            "model": champion.get("model"),
            "overall_score": 1.0,
            "is_valid": True,
            "scores_by_env": {},
        })
    if battle:
        scores["scores"].append({
            "uid": battle.get("uid"),
            "miner_hotkey": battle.get("hotkey"),
            "model": battle.get("model"),
            "overall_score": 0.0,
            "is_valid": True,
            "scores_by_env": {},
        })
    _print_rank_table(state, [], scores if scores["scores"] else None)


def _print_queue(queue: Optional[List[Dict[str, Any]]]) -> None:
    print("=" * 88)
    print(f"CHALLENGER QUEUE (head {_QUEUE_PREVIEW})")
    print("=" * 88)
    if not queue:
        print("  (queue empty)")
        return
    print(f"  {'pos':<5}{'uid':<6}{'first_block':<14}{'hotkey':<20}{'revision':<12}{'model'}")
    for row in queue[:_QUEUE_PREVIEW]:
        print(
            f"  {row.get('position'):<5}{row.get('uid'):<6}"
            f"{str(row.get('first_block', '')):<14}"
            f"{_short(row.get('hotkey'), 18):<20}"
            f"{_short(row.get('revision'), 10):<12}"
            f"{_short(row.get('model'), 40)}"
        )


def _print_scores(scores_resp: Optional[Dict[str, Any]]) -> None:
    _print_rank_table(None, [], scores_resp)


async def get_rank_command() -> None:
    async with cli_api_client() as client:
        window = await _safe_get(client, "/windows/current")
        queue = await _safe_get(client, f"/windows/queue?limit={_QUEUE_PREVIEW}")
        scores = await _safe_get(client, f"/scores/latest?top={_RANK_FETCH_LIMIT}")

    _print_rank_table(
        window if isinstance(window, dict) else None,
        queue if isinstance(queue, list) else None,
        scores if isinstance(scores, dict) else None,
    )
