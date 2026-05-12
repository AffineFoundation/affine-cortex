"""
Anti-Copy Detection Service - Main Entry Point

Runs as an independent background service that periodically fetches miners
from metagraph and runs copy detection using logprob/hidden-state signals.
"""

import asyncio
import json
import math
import time
import signal
import click
from collections import defaultdict
from typing import Dict, List
from affine.core.setup import logger, setup_logging, NETUID
from affine.database import init_client, close_client
from affine.utils.subtensor import get_subtensor
from affine.database.dao.anti_copy import AntiCopyDAO
from .detector import AntiCopyDetector
from .loader import LogprobsLoader


DEFAULT_INTERVAL = 86400  # 24 hours


class AntiCopyService:
    """Periodic anti-copy detection service."""

    _SEVERITY_RANK = {"clean": 0, "suspicious": 1, "cheat": 2}

    def __init__(self, interval: int = DEFAULT_INTERVAL):
        self.interval = interval
        self._running = False
        self._task = None

    async def _fetch_miners(self):
        """Fetch active miners from metagraph."""
        subtensor = await get_subtensor()
        meta = await subtensor.metagraph(NETUID)
        commits = await subtensor.get_all_revealed_commitments(NETUID)

        miners = []
        for uid in range(len(meta.hotkeys)):
            hotkey = meta.hotkeys[uid]
            if hotkey not in commits:
                continue
            try:
                block, commit_data = commits[hotkey][-1]
                data = json.loads(commit_data)
                revision = data.get("revision", "")
                model = data.get("model", "")
                if hotkey and revision and model:
                    miners.append({
                        "uid": uid,
                        "hotkey": hotkey,
                        "revision": revision,
                        "model": model,
                        "block": int(block) if uid != 0 else 0,
                    })
            except Exception:
                continue

        logger.info(f"[AntiCopy] Fetched {len(miners)} active miners from chain")
        return miners

    def _build_copy_records(
        self, flagged_pairs, miner_info: Dict[int, dict]
    ) -> List[dict]:
        """Build per-model anti-copy records.

        For each miner, find the strongest direct flagged relation that points
        to an earlier miner. Severity wins over recency; within the same
        severity, the earliest block wins.

        Returns:
            List of dicts ready for DAO.save_round() (only suspicious/cheat miners)
        """
        # Build adjacency: uid -> [(other_uid, CopyPair)]
        neighbors: Dict[int, list] = defaultdict(list)
        for pair in flagged_pairs:
            neighbors[pair.uid_a].append((pair.uid_b, pair))
            neighbors[pair.uid_b].append((pair.uid_a, pair))

        results = []
        for uid, peers in neighbors.items():
            info = miner_info[uid]
            my_block = info["block"]

            # Find the peer with the earliest block
            chosen_uid = None
            chosen_block = my_block
            chosen_pair = None
            chosen_severity = "clean"
            for peer_uid, pair in peers:
                peer_block = miner_info[peer_uid]["block"]
                if not (
                    peer_block < my_block or (peer_block == my_block and peer_uid < uid)
                ):
                    continue

                if (
                    self._SEVERITY_RANK.get(pair.verdict, 0)
                    > self._SEVERITY_RANK.get(chosen_severity, 0)
                ):
                    chosen_uid = peer_uid
                    chosen_block = peer_block
                    chosen_pair = pair
                    chosen_severity = pair.verdict
                    continue

                if (
                    pair.verdict == chosen_severity
                    and (
                        peer_block < chosen_block
                        or (
                            peer_block == chosen_block
                            and (chosen_uid is None or peer_uid < chosen_uid)
                        )
                    )
                ):
                    chosen_uid = peer_uid
                    chosen_block = peer_block
                    chosen_pair = pair

            # If no peer is earlier, this miner is an original
            if chosen_uid is None:
                continue

            orig_info = miner_info[chosen_uid]
            copy_entry = {
                "uid": chosen_uid,
                "hotkey": orig_info["hotkey"],
                "model": orig_info["model"],
            }
            if chosen_pair:
                for key, val in [
                    ("logprobs_cosine", chosen_pair.cosine_similarity),
                    ("hs_cosine", chosen_pair.hs_cosine),
                    ("js_div", chosen_pair.js_divergence),
                ]:
                    if val is not None and not (isinstance(val, float) and math.isnan(val)):
                        copy_entry[key] = val
                copy_entry["n_tasks"] = chosen_pair.n_tasks

            results.append({
                "uid": uid,
                "hotkey": info["hotkey"],
                "model": info["model"],
                "revision": info["revision"],
                "block": info["block"],
                "status": chosen_severity,
                "is_copy": chosen_severity == "cheat",
                "copy_of": [copy_entry],
            })

        return results

    async def _run_detection(self):
        """Run one round of copy detection."""
        # Detection is OFF by default while the LOGPROBS env is being
        # reworked (n_tokens=20 plus template-heavy prompts gives near-1.0
        # logprob cosine even for legitimate fine-tunes of the same base).
        # Set ANTICOPY_ENABLED=1 to re-enable after the env is updated and
        # detector thresholds are recalibrated. No DB writes happen while
        # disabled, so existing anti_copy_results rows are NOT refreshed —
        # they will age out via TTL or need manual cleanup.
        import os
        if os.getenv("ANTICOPY_ENABLED", "0") != "1":
            logger.info("[AntiCopy] disabled by default (set ANTICOPY_ENABLED=1 to enable), skipping detection")
            return

        miners = await self._fetch_miners()
        if len(miners) < 2:
            logger.warning("[AntiCopy] Not enough miners for comparison, skipping")
            return

        # Build uid -> miner info lookup
        miner_info = {m["uid"]: m for m in miners}

        loader = LogprobsLoader()
        miner_data = await loader.load_all_miners(miners)
        logger.info(f"[AntiCopy] Loaded logprobs for {len(miner_data)}/{len(miners)} miners")

        if len(miner_data) < 2:
            logger.warning("[AntiCopy] Not enough miners with logprob data, skipping")
            return

        detector = AntiCopyDetector()
        results = detector.detect(miner_data)

        flagged_pairs = [r for r in results if r.verdict != "clean"]
        logger.info(
            f"[AntiCopy] Detection complete: {len(flagged_pairs)} flagged pairs "
            f"out of {len(results)} pairs evaluated"
        )
        for r in flagged_pairs:
            logger.warning(f"[AntiCopy] {r}")

        # Save to DB: write results for ALL evaluated miners
        # so that previously-flagged miners get cleared when no longer copy
        flagged_records = self._build_copy_records(flagged_pairs, miner_info) if flagged_pairs else []
        flagged_uids = {r["uid"] for r in flagged_records}

        # Build clean records for non-copy miners that were evaluated
        clean_records = []
        for uid in miner_data:
            if uid not in flagged_uids:
                info = miner_info[uid]
                clean_records.append({
                    "uid": uid,
                    "hotkey": info["hotkey"],
                    "model": info["model"],
                    "revision": info["revision"],
                    "block": info["block"],
                    "status": "clean",
                    "is_copy": False,
                    "copy_of": [],
                })

        all_records = flagged_records + clean_records
        if all_records:
            dao = AntiCopyDAO()
            round_ts = int(time.time())
            await dao.save_round(all_records, round_timestamp=round_ts)
            cheat_count = sum(1 for r in all_records if r["status"] == "cheat")
            suspicious_count = sum(1 for r in all_records if r["status"] == "suspicious")
            logger.info(
                f"[AntiCopy] Saved {len(all_records)} records "
                f"({cheat_count} cheats, {suspicious_count} suspicious, {len(clean_records)} clean) to DB"
            )

    async def _loop(self):
        """Background detection loop."""
        while self._running:
            try:
                await self._run_detection()
            except Exception as e:
                logger.error(f"[AntiCopy] Error in detection loop: {e}", exc_info=True)
            await asyncio.sleep(self.interval)

    async def start(self):
        """Start background detection."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(f"[AntiCopy] Service started (interval={self.interval}s)")

    async def stop(self):
        """Stop background detection."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("[AntiCopy] Service stopped")


async def run_service(interval: int = DEFAULT_INTERVAL):
    """Run the anti-copy detection service."""
    logger.info("Starting Anti-Copy Detection Service")

    try:
        await init_client()
        logger.info("Database client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

    shutdown_event = asyncio.Event()

    def handle_shutdown(sig):
        logger.info(f"Received signal {sig}, initiating shutdown...")
        shutdown_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: handle_shutdown(s))

    service = None
    try:
        service = AntiCopyService(interval=interval)
        await service.start()
        await shutdown_event.wait()
    except Exception as e:
        logger.error(f"Error running AntiCopyService: {e}", exc_info=True)
        raise
    finally:
        if service:
            try:
                await service.stop()
            except Exception as e:
                logger.error(f"Error stopping AntiCopyService: {e}")
        try:
            await close_client()
            logger.info("Database client closed")
        except Exception as e:
            logger.error(f"Error closing database: {e}")

    logger.info("Anti-Copy Detection Service shut down successfully")


@click.command()
@click.option(
    "-v", "--verbosity",
    default=None,
    type=click.Choice(["0", "1", "2", "3"]),
    help="Logging verbosity: 0=CRITICAL, 1=INFO, 2=DEBUG, 3=TRACE"
)
@click.option(
    "--interval",
    default=DEFAULT_INTERVAL,
    type=int,
    help=f"Detection interval in seconds (default: {DEFAULT_INTERVAL})"
)
def main(verbosity, interval):
    """
    Affine Anti-Copy Detection Service.

    Periodically checks all miners for model copying using
    logprob cosine and hidden-state signals.
    """
    if verbosity is not None:
        setup_logging(int(verbosity))
    asyncio.run(run_service(interval=interval))


if __name__ == "__main__":
    main()
