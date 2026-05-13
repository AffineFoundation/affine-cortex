"""
Rank display.

``af get-rank`` prints three blocks in order:

  1. CURRENT WINDOW     — block range, phase, champion, challenger, progress
  2. CHALLENGER QUEUE   — next N miners ordered by (first_block, uid)
  3. WEIGHT SNAPSHOT    — current scores table (one row per miner; the
                          champion sits at overall_score=1.0, everyone
                          else at 0.0)

Read-only; talks to the API only.
"""

from __future__ import annotations

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


def _print_window(state: Optional[Dict[str, Any]]) -> None:
    """Render ``GET /windows/current`` — champion + in-flight battle."""
    print("=" * 80)
    print("CURRENT STATE")
    print("=" * 80)
    if not state:
        print("  (no state)")
        return
    refresh = state.get("task_refresh_block")
    if refresh is not None:
        print(f"  task_refresh  : block {refresh}")
    champ = state.get("champion")
    if champ:
        base_url = state.get("champion_base_url") or "-"
        print(f"  champion      : uid={champ.get('uid')}  hk={(champ.get('hotkey') or '')[:14]}...  "
              f"{champ.get('model')}@{(champ.get('revision') or '')[:8]}...")
        print(f"  champion_url  : {base_url}")
    else:
        print("  champion      : -")
    battle = state.get("battle")
    if battle:
        chal = battle.get("challenger") or {}
        print(f"  battle        : uid={chal.get('uid')}  hk={(chal.get('hotkey') or '')[:14]}...  "
              f"{chal.get('model')}@{(chal.get('revision') or '')[:8]}...  "
              f"(started @ block {battle.get('started_at_block')})")
    else:
        print("  battle        : - (idle)")


def _print_queue(queue: Optional[List[Dict[str, Any]]]) -> None:
    print()
    print("=" * 80)
    print(f"CHALLENGER QUEUE (head {_QUEUE_PREVIEW})")
    print("=" * 80)
    if not queue:
        print("  (queue empty)")
        return
    print(f"  {'pos':<5}{'uid':<6}{'first_block':<14}{'hotkey':<20}{'revision':<12}{'model'}")
    for row in queue[:_QUEUE_PREVIEW]:
        hk = (row.get("hotkey") or "")[:18]
        rev = (row.get("revision") or "")[:10]
        model = (row.get("model") or "")[:40]
        print(
            f"  {row.get('position'):<5}{row.get('uid'):<6}"
            f"{row.get('first_block', ''):<14}{hk:<20}{rev:<12}{model}"
        )


def _print_scores(scores_resp: Optional[Dict[str, Any]]) -> None:
    print()
    print("=" * 80)
    print("WEIGHT SNAPSHOT")
    print("=" * 80)
    if not scores_resp or not scores_resp.get("scores"):
        print("  (no scores yet)")
        return
    print(f"  block_number  : {scores_resp.get('block_number')}")
    print(f"  calculated_at : {scores_resp.get('calculated_at')}")
    print()
    print(f"  {'uid':<6}{'weight':<10}{'valid':<8}{'hotkey':<18}{'revision':<12}{'model'}")
    scores = scores_resp["scores"]
    scores.sort(key=lambda s: -float(s.get("overall_score") or 0.0))
    for s in scores:
        weight = float(s.get("overall_score") or 0.0)
        is_valid = s.get("is_valid")
        valid_mark = "yes" if is_valid is True else ("no" if is_valid is False else "?")
        hk = (s.get("miner_hotkey") or "")[:16]
        rev = (s.get("model_revision") or "")[:10]
        model = (s.get("model") or "")[:35]
        print(f"  {s.get('uid'):<6}{weight:<10.4f}{valid_mark:<8}{hk:<18}{rev:<12}{model}")


async def get_rank_command() -> None:
    async with cli_api_client() as client:
        window = await _safe_get(client, "/windows/current")
        queue = await _safe_get(client, f"/windows/queue?limit={_QUEUE_PREVIEW}")
        scores = await _safe_get(client, f"/scores/latest?top={_RANK_FETCH_LIMIT}")

    _print_window(window if isinstance(window, dict) else None)
    _print_queue(queue if isinstance(queue, list) else None)
    _print_scores(scores if isinstance(scores, dict) else None)
