"""
Rank Display Module

Fetches and displays miner ranking information from the API in the
champion-challenge format. Read-only — runs against a deployed API.
"""

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

from affine.utils.api_client import cli_api_client
from affine.utils.errors import ApiResponseError
from affine.core.setup import logger


# Cap on miners returned per snapshot. Subnet currently well below this.
# Revisit if active miner count approaches the limit.
RANK_FETCH_LIMIT = 256


# ── Fetch helpers ───────────────────────────────────────────────────────────

async def fetch_latest_scores(client) -> Dict[str, Any]:
    """Fetch latest score snapshot from API."""
    return await client.get(f"/scores/latest?top={RANK_FETCH_LIMIT}")


async def fetch_environments(client) -> Tuple[List[str], Dict[str, Any]]:
    """Fetch enabled scoring environments and per-env config.

    Returns:
        (sorted env names, env_name -> env_config dict)
    """
    try:
        config = await client.get("/config/environments")
        if isinstance(config, dict):
            value = config.get("param_value")
            if isinstance(value, dict):
                enabled_envs = []
                env_configs = {}
                for env_name, env_config in value.items():
                    if isinstance(env_config, dict) and env_config.get("enabled_for_scoring", False):
                        enabled_envs.append(env_name)
                        env_configs[env_name] = env_config
                if enabled_envs:
                    return sorted(enabled_envs), env_configs
        logger.warning("Failed to parse environments config")
    except Exception as e:
        logger.error(f"Error fetching environments: {e}")
    return [], {}


async def fetch_scorer_config(client) -> dict:
    """Fetch scorer config from latest weights snapshot."""
    try:
        weights_data = await client.get("/scores/weights/latest")
        if isinstance(weights_data, dict):
            return weights_data.get("config", {}) or {}
    except Exception as e:
        logger.error(f"Error fetching scorer config: {e}")
    return {}


async def fetch_champion_state(client) -> Optional[Dict[str, Any]]:
    """Fetch the current champion record from /config/champion.

    Returns:
        Dict with hotkey, revision, uid, since_block, or None on cold
        start (no champion crowned yet) or fetch failure.
    """
    try:
        data = await client.get("/config/champion")
        if isinstance(data, dict):
            value = data.get("param_value")
            if isinstance(value, dict):
                return value
    except ApiResponseError as e:
        if e.status_code == 404:
            return None  # Cold start — expected.
        logger.warning(f"Could not fetch champion state: {e}")
    except Exception as e:
        logger.warning(f"Could not fetch champion state: {e}")
    return None


# ── Parse / sort ────────────────────────────────────────────────────────────

@dataclass
class RankedMiner:
    uid: int
    hotkey: str
    model: str
    scores_by_env: Dict[str, Dict[str, Any]]
    average_score: float
    is_champion: bool
    status: str                    # challenge_status — 'sampling' | 'terminated'
    consecutive_wins: int
    total_wins: int
    total_losses: int
    consecutive_losses: int
    checkpoints_passed: int
    total_samples: int             # Samples produced in current scoring cycle
    targon_status: Optional[str]   # 'active' | 'deploying' | None — Targon acceleration
    is_valid: Optional[bool]       # validator-side admissibility (anticopy etc.)
    invalid_reason: Optional[str]

    @property
    def is_cold(self) -> bool:
        # Challenge state preserved across cold/hot cycles, so a miner can sit
        # in 'sampling' indefinitely without producing samples. Flag that as
        # cold so operators can tell it apart from actively-sampled miners.
        return (
            not self.is_champion
            and self.status == "sampling"
            and self.is_valid is not False  # invalid miners get their own bucket
            and self.total_samples == 0
        )

    @property
    def is_invalid(self) -> bool:
        # validator-side invalidation (anticopy, model_mismatch, repo-name,
        # etc.) — sampling scheduler stops sampling these via is_valid filter,
        # so they shouldn't be displayed as 'sampling' even when their
        # challenge_status snapshot still says so.
        return self.is_valid is False


def parse_ranked_miners(scores_list: List[Dict[str, Any]]) -> List[RankedMiner]:
    out = []
    for s in scores_list:
        ci = s.get("challenge_info") or {}
        # is_valid arrives over the API as JSON bool (or None if the miners
        # row was missing); preserve None vs False distinction so the UI
        # only shows INVALID when we actually know the miner was rejected.
        is_valid = s.get("is_valid")
        if is_valid is not None:
            is_valid = bool(is_valid)
        out.append(RankedMiner(
            uid=s.get("uid"),
            hotkey=s.get("miner_hotkey") or "",
            model=s.get("model") or "",
            scores_by_env=s.get("scores_by_env") or {},
            average_score=s.get("average_score") or 0.0,
            is_champion=bool(ci.get("is_champion", False)),
            status=ci.get("status", "sampling"),
            consecutive_wins=int(ci.get("consecutive_wins", 0) or 0),
            total_wins=int(ci.get("total_wins", 0) or 0),
            total_losses=int(ci.get("total_losses", 0) or 0),
            consecutive_losses=int(ci.get("consecutive_losses", 0) or 0),
            checkpoints_passed=int(ci.get("checkpoints_passed", 0) or 0),
            total_samples=int(s.get("total_samples", 0) or 0),
            targon_status=s.get("targon_status"),
            is_valid=is_valid,
            invalid_reason=s.get("invalid_reason"),
        ))
    return out


_BOOST_GLYPHS = {
    "active":     "⚡",   # serving with dedicated capacity right now
    "deploying":  "⏳",   # warming up, joining shortly
}


def _is_color_tty() -> bool:
    """ANSI colors only if stdout is an interactive terminal."""
    import sys
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


def format_boost(status: Optional[str]) -> str:
    """Two-cell badge for the scheduling-acceleration column.

    ⚡ / ⏳ render as two terminal cells in CJK-aware fonts (East Asian
    Wide), but Python sees them as a single char — so empty rows need
    two literal spaces to match the rendered column width and keep the
    rest of the table aligned.
    """
    glyph = _BOOST_GLYPHS.get(status or "")
    if not glyph:
        return "  "
    if not _is_color_tty():
        return glyph
    color = "1;92" if status == "active" else "1;93"  # green / yellow
    return f"\033[{color}m{glyph}\033[0m"


# Backwards-compat shim — earlier callers / tests may still import this.
format_targon = format_boost


def sort_key(m: RankedMiner) -> tuple:
    """Champion → active sampling (highest CP first) → cold → invalid → terminated."""
    if m.is_champion:
        return (0,)
    if m.status == "terminated":
        return (4, -m.total_losses, -m.checkpoints_passed)
    if m.is_invalid:
        # validator-side rejection (anticopy / model_mismatch / repo-name /
        # …): not in the competition anymore even though challenge_status
        # might still read 'sampling' — group below cold but above
        # challenge-terminated so operators can spot them as a distinct
        # bucket.
        return (3, -m.checkpoints_passed, -m.average_score)
    if m.is_cold:
        return (2, -m.checkpoints_passed, -m.average_score)
    # Active sampling: closeness to dethrone, then checkpoint depth, then avg
    # Sort by checkpoint progress (closest to dethrone first), then avg score
    return (1, -m.checkpoints_passed, -m.average_score)


# ── Format helpers ──────────────────────────────────────────────────────────

def format_relative_time(epoch_seconds: Optional[int]) -> str:
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
        h, m = divmod(delta, 3600)
        return f"{h}h {m // 60}m ago"
    return f"{delta // 86400}d ago"


def format_iso(epoch_seconds: Optional[int]) -> str:
    if not epoch_seconds:
        return "unknown"
    return datetime.fromtimestamp(int(epoch_seconds), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def env_display_name(env: str, env_cfg: Dict[str, Any]) -> str:
    if isinstance(env_cfg, dict) and env_cfg.get("display_name"):
        return env_cfg["display_name"]
    if ":" in env:
        return env.split(":", 1)[1]
    return env


# ── Main render ─────────────────────────────────────────────────────────────

async def print_rank_table():
    """Fetch and print the champion-challenge ranking table."""
    async with cli_api_client() as client:
        scores_data, env_result, scorer_config, champion_state = await asyncio.gather(
            fetch_latest_scores(client),
            fetch_environments(client),
            fetch_scorer_config(client),
            fetch_champion_state(client),
        )
        environments, env_configs = env_result

        if not scores_data or not scores_data.get("block_number"):
            print("No scores found")
            return

        block_number = scores_data.get("block_number")
        calculated_at = scores_data.get("calculated_at")
        scores_list = scores_data.get("scores", [])

        if not scores_list:
            print(f"No miners scored at block {block_number}")
            return

        miners = parse_ranked_miners(scores_list)
        miners.sort(key=sort_key)

        dethrone_cp = scorer_config.get("champion_dethrone_min_checkpoint", 10)
        M = scorer_config.get("champion_termination_total_losses", 3)
        # Bounds shown in legend only. Actual per-cell thresholds come
        # from the scorer's real common-task computation (both bounds
        # stored in scores_by_env[env]), which differs per challenger.
        dethrone_margin = scorer_config.get("win_margin_end", 0.03)
        not_worse_tol = scorer_config.get("win_not_worse_tolerance", 0.015)

        # ── Header ────────────────────────────────────────────────────────
        # Boost column ("⚡") sits next to UID so it survives narrow
        # terminals — accelerated miners are the most operationally
        # interesting rows and the rightmost column is the first to clip.
        # ⚡ is glued to Model (no space between them) because the column
        # separator already provides space on the *left* of the glyph,
        # and a wide ⚡ visually filling its cell needs no right padding.
        header_parts = ["Hotkey  ", " UID", "⚡| Model                    "]
        for env in environments:
            disp = env_display_name(env, env_configs.get(env, {}))
            header_parts.append(f"{disp:>26}")
        header_parts.extend(["  Status   ", "  CP ", " Challenge "])
        header_line = " | ".join(header_parts)
        table_width = len(header_line)

        # Legend: every number in an env cell is on the *common-task*
        # basis with the current champion — the exact data Pareto uses.
        # This keeps score, lower, and upper directly comparable.
        # Champion row shows its historical average as a reference with [★].
        # "—" means no common tasks with the champion yet (new miner or
        # champion missing).
        legend = (
            f"Env cells: common-task-score% [lose-below, win-above] / common-tasks "
            f"(lose < champ_on_common×(1-{not_worse_tol * 100:.1f}%), "
            f"win > champ_on_common+{dethrone_margin * 100:.1f}%; "
            f"dethrone at CP {dethrone_cp}; champion row [★] shows historical avg)"
        )

        # ── Title block ───────────────────────────────────────────────────
        print("=" * table_width, flush=True)
        print(f"CHAMPION CHALLENGE RANKING — Block {block_number}", flush=True)
        print(
            f"Calculated: {format_relative_time(calculated_at)} ({format_iso(calculated_at)})",
            flush=True,
        )

        # Champion line
        champion_present_uid = next((m.uid for m in miners if m.is_champion), None)
        if champion_state:
            champ_hk = (champion_state.get("hotkey") or "")[:8]
            since_block = champion_state.get("since_block")
            tenure = (block_number - since_block) if (since_block is not None) else None
            tenure_str = f"Δ {tenure} blocks" if tenure is not None else "tenure unknown"
            if champion_present_uid is not None:
                print(
                    f"Champion:   {champ_hk}... reigning since block {since_block} ({tenure_str})",
                    flush=True,
                )
            else:
                print(
                    f"Champion:   {champ_hk}... reigning since block {since_block} "
                    f"({tenure_str}, offline this round)",
                    flush=True,
                )
        else:
            print("Champion:   (none — cold start)", flush=True)

        print(legend, flush=True)

        print("=" * table_width, flush=True)
        print(header_line, flush=True)
        print("-" * table_width, flush=True)

        # ── Rows ──────────────────────────────────────────────────────────
        for m in miners:
            row_parts = [
                f"{m.hotkey[:8]:8s}",
                f"{m.uid:4d}",
                f"{format_boost(m.targon_status)}| {m.model[:25]:25s}",
            ]

            for env in environments:
                if env not in m.scores_by_env:
                    row_parts.append(f"{'  -  ':>26}")
                    continue

                env_data = m.scores_by_env[env]

                if m.is_champion:
                    # Champion is the reference. No self-comparison, so
                    # fall back to historical average + total samples.
                    score_val = env_data.get("score", 0.0)
                    samples = env_data.get("historical_count",
                                            env_data.get("sample_count", 0))
                    score_str = f"{score_val * 100:.2f}[★]/{samples}"
                    row_parts.append(f"{score_str:>26}")
                    continue

                # Challenger: show numbers on the common-task basis only,
                # matching what Pareto actually evaluates.
                score_on_common = env_data.get("score_on_common")
                common_tasks = env_data.get("common_tasks")
                lower = env_data.get("not_worse_threshold")
                upper = env_data.get("dethrone_threshold")

                if (isinstance(score_on_common, (int, float))
                        and isinstance(common_tasks, int)
                        and common_tasks > 0
                        and isinstance(lower, (int, float))
                        and isinstance(upper, (int, float))):
                    score_str = (
                        f"{score_on_common * 100:.2f}"
                        f"[{lower * 100:.2f},{upper * 100:.2f}]"
                        f"/{common_tasks}"
                    )
                else:
                    # No overlap with champion (champion offline, brand-new
                    # miner, or old snapshot without vs_champion fields).
                    score_str = "—[—]/—"
                row_parts.append(f"{score_str:>26}")

            # Status / CP / Challenge
            if m.is_champion:
                status_str = "★ CHAMPION"
                cp_str = "—"
                challenge_str = "—"
            elif m.status == "terminated":
                status_str = "TERMINATED"
                cp_str = str(m.checkpoints_passed)
                if m.total_losses == 0:
                    challenge_str = "pairwise"
                else:
                    challenge_str = f"L:{m.total_losses}/{M}"
            elif m.is_invalid:
                # validator-side rejection: sampling has actually stopped
                # via get_valid_miners() filter, so showing 'sampling' or
                # 'cold' would be misleading. Surface the *specific* cause
                # in the Status column (a generic 'INVALID' loses the
                # signal an operator needs); the detail tail (similarity
                # scores, mismatched chute name, etc.) goes in Challenge.
                reason = m.invalid_reason or "invalid"
                kind, _, detail = reason.partition(":")
                # Map raw miners-monitor reason keys to compact labels
                # that fit the 11-char Status column without dropping
                # information. Unknown reasons fall through truncated.
                kind_label = {
                    "anticopy": "anticopy",
                    "chute_fetch_failed": "chute_fetch",
                    "chute_slug_empty": "chute_empty",
                    "chute_not_hot": "chute_cold",
                    "model_mismatch": "model_mis",
                    "model_name_missing_affine": "name_miss",
                    "repo_name_not_ending_with_hotkey": "repo_name",
                    "revision_mismatch": "rev_mis",
                    "hf_model_fetch_failed": "hf_fetch",
                    "multiple_commits": "multi_cmt",
                }.get(kind, kind[:11])
                status_str = kind_label
                cp_str = "—"
                # Detail (similarity scores, mismatched chute name, etc.)
                # is supplementary here — Status already names the kind.
                # Truncate to the 11-char column width so a long detail
                # doesn't shove the rest of the row out of alignment;
                # operators drill in via `af get-miner <uid>` for the
                # full invalid_reason if they need it.
                challenge_str = (detail[:11] if detail else "—")
            elif m.is_cold:
                status_str = "cold"
                cp_str = f"{m.checkpoints_passed}/{dethrone_cp}"
                challenge_str = "—"
            else:
                status_str = "sampling"
                cp_str = f"{m.checkpoints_passed}/{dethrone_cp}"
                parts = []
                if m.checkpoints_passed >= dethrone_cp and m.consecutive_wins > 0:
                    parts.append("READY")
                elif m.total_wins > 0:
                    parts.append(f"W:{m.total_wins}")
                if m.total_losses > 0:
                    parts.append(f"L:{m.total_losses}/{M}")
                challenge_str = " ".join(parts) if parts else "—"

            row_parts.append(f"{status_str:>11}")
            row_parts.append(f"{cp_str:>5}")
            row_parts.append(f"{challenge_str:>11}")

            print(" | ".join(row_parts), flush=True)

        # ── Footer ────────────────────────────────────────────────────────
        print("=" * table_width, flush=True)
        cold_count = sum(1 for m in miners if m.is_cold)
        sampling_count = sum(
            1 for m in miners
            if m.status == "sampling" and not m.is_champion and not m.is_cold
        )
        terminated_count = sum(1 for m in miners if m.status == "terminated")

        if champion_present_uid is not None:
            champ_summary = f"Champion: 1 (UID {champion_present_uid})"
        elif champion_state:
            champ_summary = f"Champion: 0 (last: UID {champion_state.get('uid')}, offline)"
        else:
            champ_summary = "Champion: 0 (cold start)"

        print(
            f"Total: {len(miners)}  |  {champ_summary}  |  "
            f"Sampling: {sampling_count}  |  Cold: {cold_count}  |  "
            f"Terminated: {terminated_count}",
            flush=True,
        )
        boost_active = sum(1 for m in miners if m.targon_status == "active")
        boost_deploying = sum(1 for m in miners if m.targon_status == "deploying")
        if boost_active or boost_deploying:
            on, off = ("\033[1;96m", "\033[0m") if _is_color_tty() else ("", "")
            zap, hourglass = ("⚡", "⏳")
            print(
                f"{on}Scheduler boost{off}: "
                f"{zap} {boost_active} miner(s) on dedicated capacity now, "
                f"{hourglass} {boost_deploying} warming up "
                f"— validator-accelerated sampling lane",
                flush=True,
            )
        print("=" * table_width, flush=True)


async def get_rank_command():
    """Command handler for `af get-rank`."""
    try:
        await print_rank_table()
    except Exception as e:
        logger.error(f"Failed to fetch and display ranks: {e}", exc_info=True)
        print(f"Error: {e}")
