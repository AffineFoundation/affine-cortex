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
from .models import MinerLogprobs
from .tensor_fingerprint import TensorFingerprinter, HAS_DEPS as HAS_TENSOR_DEPS


DEFAULT_INTERVAL = 86400  # 24 hours


class AntiCopyService:
    """Periodic anti-copy detection service."""

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
        self, copy_pairs, miner_info: Dict[int, dict]
    ) -> List[dict]:
        """Build per-model copy records.

        For each miner, find the earliest-block miner among all its
        direct similar pairs. If that miner committed earlier, the
        current miner is a copier pointing to it as the original.

        Returns:
            List of dicts ready for DAO.save_round() (only copiers)
        """
        # Build adjacency: uid -> [(other_uid, CopyPair)]
        neighbors: Dict[int, list] = defaultdict(list)
        for pair in copy_pairs:
            neighbors[pair.uid_a].append((pair.uid_b, pair))
            neighbors[pair.uid_b].append((pair.uid_a, pair))

        results = []
        for uid, peers in neighbors.items():
            info = miner_info[uid]
            my_block = info["block"]

            # Find the peer with the earliest block
            earliest_uid = None
            earliest_block = my_block
            earliest_pair = None
            for peer_uid, pair in peers:
                peer_block = miner_info[peer_uid]["block"]
                if peer_block < earliest_block or (
                    peer_block == earliest_block and peer_uid < uid
                ):
                    earliest_block = peer_block
                    earliest_uid = peer_uid
                    earliest_pair = pair

            # If no peer is earlier, this miner is an original
            if earliest_uid is None:
                continue

            orig_info = miner_info[earliest_uid]
            copy_entry = {
                "uid": earliest_uid,
                "hotkey": orig_info["hotkey"],
                "model": orig_info["model"],
            }
            if earliest_pair:
                for key, val in [
                    ("logprobs_cosine", earliest_pair.cosine_similarity),
                    ("hs_cosine", earliest_pair.hs_cosine),
                    ("tensor_cosine", earliest_pair.tensor_cosine),
                    ("js_div", earliest_pair.js_divergence),
                ]:
                    if val is not None and not (isinstance(val, float) and math.isnan(val)):
                        copy_entry[key] = val
                copy_entry["n_tasks"] = earliest_pair.n_tasks

            results.append({
                "uid": uid,
                "hotkey": info["hotkey"],
                "model": info["model"],
                "revision": info["revision"],
                "block": info["block"],
                "is_copy": True,
                "copy_of": [copy_entry],
            })

        return results

    async def _run_detection(self):
        """Run one round of copy detection."""
        miners = await self._fetch_miners()
        if len(miners) < 2:
            logger.warning("[AntiCopy] Not enough miners for comparison, skipping")
            return

        # Build uid -> miner info lookup
        miner_info = {m["uid"]: m for m in miners}

        loader = LogprobsLoader()
        miner_data = await loader.load_all_miners(miners)
        logger.info(f"[AntiCopy] Loaded logprobs for {len(miner_data)}/{len(miners)} miners")

        # Tensor fingerprinting (runs independently of logprobs)
        tensor_fingerprints = {}
        if HAS_TENSOR_DEPS:
            try:
                fingerprinter = TensorFingerprinter()
                tensor_fingerprints = fingerprinter.fingerprint_miners(miners)
                logger.info(
                    f"[AntiCopy] Tensor fingerprints for "
                    f"{len(tensor_fingerprints)}/{len(miners)} miners"
                )
            except Exception as e:
                logger.error(f"[AntiCopy] Tensor fingerprinting failed: {e}", exc_info=True)
        else:
            logger.info("[AntiCopy] Tensor fingerprinting unavailable (missing deps)")

        if len(miner_data) < 2 and len(tensor_fingerprints) < 2:
            logger.warning("[AntiCopy] Not enough miners with data, skipping")
            return

        # Ensure all miners with tensor fingerprints are in miner_data
        # (detector needs MinerLogprobs entries for pair enumeration)
        for uid in tensor_fingerprints:
            if uid not in miner_data:
                info = miner_info[uid]
                miner_data[uid] = MinerLogprobs(uid=uid, hotkey=info["hotkey"])

        detector = AntiCopyDetector()
        results = detector.detect(miner_data, tensor_fingerprints=tensor_fingerprints)

        copy_pairs = [r for r in results if r.is_copy]
        logger.info(
            f"[AntiCopy] Detection complete: {len(copy_pairs)} copy pairs "
            f"out of {len(results)} pairs evaluated"
        )
        for r in copy_pairs:
            logger.warning(f"[AntiCopy] {r}")

        # Save to DB: write results for ALL evaluated miners
        # so that previously-flagged miners get cleared when no longer copy
        copy_records = self._build_copy_records(copy_pairs, miner_info) if copy_pairs else []
        copy_uids = {r["uid"] for r in copy_records}

        # Build clean records for non-copy miners that were evaluated
        clean_records = []
        for uid in miner_data:
            if uid not in copy_uids:
                info = miner_info[uid]
                clean_records.append({
                    "uid": uid,
                    "hotkey": info["hotkey"],
                    "model": info["model"],
                    "revision": info["revision"],
                    "block": info["block"],
                    "is_copy": False,
                    "copy_of": [],
                })

        all_records = copy_records + clean_records
        if all_records:
            dao = AntiCopyDAO()
            round_ts = int(time.time())
            await dao.save_round(all_records, round_timestamp=round_ts)
            copy_count = sum(1 for r in all_records if r["is_copy"])
            logger.info(
                f"[AntiCopy] Saved {len(all_records)} records "
                f"({copy_count} copies, {len(clean_records)} clean) to DB"
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
