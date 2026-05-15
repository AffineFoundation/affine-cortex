"""
CEAC forward worker — runs alongside the anti-copy sglang server,
pulling its own work directly from the live miners table.

One iteration:

  1. scan ``miners`` + ``miner_stats`` + ``anticopy_scores_index``,
     pick the earliest-committed valid, non-terminated, unscored
     candidate (same ordering as ``af get-rank``'s active bucket)
  2. download the candidate ckpt into the local HF cache (if absent)
  3. ``/update_weights_from_disk`` the sglang server onto the ckpt
  4. for each rollout in the tokenizer-matching pool: teacher-force
     via sglang ``/generate`` with ``return_logprob=true``
  5. upload the score blob to R2 and write ``anticopy_scores_index``
  6. run pairwise vs every existing score; record any copy verdict
     on ``anticopy_scores_index.verdict_copy_of`` + retroactively
     refresh later peers whose origin candidate just landed

There is no separate jobs queue: ``scores_index`` IS the durable
"done" marker. A transient failure on candidate X just falls
through to the next iteration — next loop picks X up again because
its scores_index row still doesn't exist. CEAC writes only to its
own ``anticopy_*`` tables; it never mutates ``miners`` or
``miner_stats``.

Teacher-forcing follows sglang's native ``/generate`` contract
(``input_ids`` + ``max_new_tokens=0`` + ``logprob_start_len``);
that's the only API that gives us logprobs on a pre-existing
response without re-sampling.
"""

from __future__ import annotations

import asyncio
import json as _json
import os
import shlex
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional

import aiohttp
from huggingface_hub import scan_cache_dir, snapshot_download

from affine.core.setup import logger
from affine.database.dao.anticopy import (
    AntiCopyRolloutsDAO,
    AntiCopyScoresIndexDAO,
    AntiCopyStateDAO,
)
from affine.database.dao.miner_stats import MinerStatsDAO
from affine.database.dao.miners import MinersDAO
from affine.database.dao.system_config import SystemConfigDAO

from affine.src.anticopy.pairwise import (
    compare_scores,
    detect_copies,
    is_copy_verdict,
)
from affine.src.anticopy.r2 import AntiCopyR2
from affine.src.anticopy.threshold import AntiCopyConfig, load_anticopy_config
from affine.src.anticopy.tokenizer_sig import compute_tokenizer_signature


# sglang HTTP endpoint of the anticopy server. Defaults to localhost
# because the worker is deployed on the same GPU host.
SGLANG_URL = os.getenv("ANTICOPY_SGLANG_URL", "http://localhost:30000")

# Where snapshot_download lands models. Must match the path sglang sees
# inside its own container (we mount the same dir).
HF_CACHE_DIR = os.getenv("ANTICOPY_HF_CACHE", "/data")

# Remote SSH-control mode. When ``ANTICOPY_REMOTE_SSH_HOST`` is set the
# worker assumes it runs *off-host* (e.g. on a backend box with prod
# DDB/R2 credentials) while the sglang server + HF cache live on the
# named GPU host. In that mode ``_fetch_weights`` and the per-job
# ``_gc_hf_cache`` tick are executed remotely over SSH instead of
# locally — exactly so that no AWS keys ever land on the GPU host.
# The sglang HTTP traffic is expected to go through an SSH local-port
# forward the operator establishes alongside (so ``SGLANG_URL`` still
# points at localhost).
REMOTE_SSH_HOST = os.getenv("ANTICOPY_REMOTE_SSH_HOST", "")
REMOTE_SSH_KEY = os.getenv(
    "ANTICOPY_REMOTE_SSH_KEY", os.path.expanduser("~/.ssh/id_rsa"),
)
REMOTE_PYTHON = os.getenv(
    "ANTICOPY_REMOTE_PYTHON", "/workspace/anticopy/.venv/bin/python",
)
# SSH port for the GPU host. Most Targon-style deployments expose 22,
# but bare-metal rentals (e.g. B300 nodes) often run sshd on a
# non-standard port — the worker passes ``-p`` only when this is set.
REMOTE_SSH_PORT = os.getenv("ANTICOPY_REMOTE_SSH_PORT", "")

# Port the actual sglang HTTP server listens on inside the GPU host.
# When the worker speaks to sglang via an SSH local-port forward, the
# *worker* side is named by ``ANTICOPY_SGLANG_URL`` (e.g.
# ``http://localhost:33000``); the forward bridges that to
# ``127.0.0.1:<REMOTE_SGLANG_PORT>`` on the GPU host. Defaults to
# 30000 to match the historical Targon setup.
REMOTE_SGLANG_PORT = int(os.getenv("ANTICOPY_REMOTE_SGLANG_PORT", "30000"))

# Which GPU(s) sglang should pin to when the worker bootstraps it.
# Comma-separated indices (e.g. "0" or "0,1,2,3" for TP=4). For
# single-slot, single-GPU usage this is just the GPU index. The worker
# only uses this when it has to *launch* sglang from scratch — when
# sglang is already running we leave it alone.
SGLANG_BASE_GPU_ID = int(os.getenv("ANTICOPY_BASE_GPU_ID", "0"))

# Args passed to the bootstrap sglang launch (in addition to model
# path, port, host, base-gpu-id). Operators tune memory fraction etc.
# here without code changes.
#
# Why these flags by default:
#   * ``--disable-cuda-graph`` — sm_103 (B300) hosts often ship with a
#     CUDA toolkit too old to JIT-compile sglang's fused-rope kernel
#     for that target. Our use case is prefill-only (teacher-force,
#     ``max_new_tokens=0``) so CUDA graphs add ~zero value anyway.
#   * ``--attention-backend triton`` — sglang's default ``trtllm_mha``
#     ships pre-built CUDA kernels that do NOT include sm_103, so a
#     forward pass on B300 dies with ``cudaErrorNoKernelImageForDevice``
#     during the very first attention call. Triton's backend JIT-compiles
#     attention kernels at runtime via LLVM and works on any sm_X
#     the installed driver supports.
SGLANG_EXTRA_ARGS = os.getenv(
    "ANTICOPY_SGLANG_EXTRA_ARGS",
    "--mem-fraction-static 0.85 --context-length 32768 --dtype bfloat16",
).strip()

# When we hit a network blip mid-fetch, retry rather than fail the job
# outright.
MAX_FETCH_ATTEMPTS = 3

# Idle sleep between empty queue polls.
EMPTY_QUEUE_SLEEP_SEC = 30

# How many times a job is allowed to fail (counted by
# ``attempts``, which ``claim_next`` increments on every claim) before
# it is permanently marked ``failed``. Transient infra issues like an
# intermittent Targon SSH gateway or HF rate limits should not burn a
# candidate's verdict; we re-queue the job instead. Cap is small
# because the worker itself retries network ops at lower layers
# (``MAX_FETCH_ATTEMPTS`` etc.), so this is the outermost safety net.
MAX_JOB_ATTEMPTS = 5


class _SglangError(RuntimeError):
    pass


def _ssh_run(host: str, key_path: str, cmd: str, *, timeout: int = 600) -> tuple:
    """Run ``cmd`` over SSH; returns ``(rc, stdout, stderr)``.

    Used only when ``REMOTE_SSH_HOST`` is configured. The wrapper
    keeps stderr separate so we don't conflate sglang/HF banner output
    with the python helper's stdout, which we parse.
    """
    argv = [
        "ssh", "-i", key_path,
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=30",
        "-o", "ServerAliveInterval=30",
    ]
    if REMOTE_SSH_PORT:
        argv += ["-p", REMOTE_SSH_PORT]
    argv += [host, cmd]
    proc = subprocess.run(
        argv,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=timeout, text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


class ForwardWorker:
    """One-job-at-a-time loop. Public ``run`` is the service entry."""

    def __init__(
        self,
        *,
        rollouts_dao: Optional[AntiCopyRolloutsDAO] = None,
        scores_dao: Optional[AntiCopyScoresIndexDAO] = None,
        miners_dao: Optional[MinersDAO] = None,
        miner_stats_dao: Optional[MinerStatsDAO] = None,
        config_dao: Optional[SystemConfigDAO] = None,
        state_dao: Optional[AntiCopyStateDAO] = None,
        r2: Optional[AntiCopyR2] = None,
        sglang_url: str = SGLANG_URL,
    ):
        self.rollouts_dao = rollouts_dao or AntiCopyRolloutsDAO()
        self.scores_dao = scores_dao or AntiCopyScoresIndexDAO()
        self.miners_dao = miners_dao or MinersDAO()
        self.miner_stats_dao = miner_stats_dao or MinerStatsDAO()
        self.config_dao = config_dao or SystemConfigDAO()
        self.state_dao = state_dao or AntiCopyStateDAO()
        self.r2 = r2 or AntiCopyR2()
        self.sglang_url = sglang_url.rstrip("/")
        self._running = False
        # In-flight prefetcher task. Cancelled (or left to finish in
        # the background) when the next iteration starts.
        self._prefetch_task: Optional[asyncio.Task] = None
        # (model, revision) we've already kicked off a prefetch for —
        # snapshot_download is idempotent, but avoiding redundant
        # to_thread fan-out keeps logs sane and prevents stacking
        # multiple awaiting tasks on the same lock.
        self._prefetched: set = set()

    # ---- main loop ---------------------------------------------------

    async def run(self) -> None:
        """Pull-based loop: every iteration computes the next candidate
        from the live miners table (same ordering as ``af get-rank``)
        and runs it. There is no separate jobs queue — ``scores_index``
        is the single done marker.

        A miner is considered a candidate iff:
          * it is on the metagraph (has a row in ``miners``),
          * ``miners.is_valid`` is True,
          * ``miner_stats.challenge_status`` is not ``terminated``,
          * ``scores_index`` has no row for ``(hotkey, revision)`` yet.

        Candidates are sorted by ``(first_block ASC, uid ASC)`` so the
        earliest committer (and ultimately the active champion) is
        scored first. A transient failure on candidate X just falls
        through to the next iteration — next loop picks X up again
        because its ``scores_index`` row still doesn't exist. No
        backoff state machine, no queue cleanup.
        """
        self._running = True
        logger.info(f"[anticopy.worker] starting; sglang={self.sglang_url}")
        while self._running:
            try:
                cand = await self._get_next_candidate()
            except Exception as e:
                logger.error(
                    f"[anticopy.worker] candidate lookup failed: {e}",
                    exc_info=True,
                )
                await asyncio.sleep(EMPTY_QUEUE_SLEEP_SEC)
                continue

            if cand is None:
                await asyncio.sleep(EMPTY_QUEUE_SLEEP_SEC)
                continue

            try:
                await self._run_job(cand)
            finally:
                # GC after every job (success or fail). Always keeps
                # the most recently-used model directories so sglang's
                # current weights are never removed. In remote-SSH
                # mode the scan + delete happens on the GPU host.
                try:
                    cfg = await load_anticopy_config(self.config_dao)
                    if REMOTE_SSH_HOST:
                        await asyncio.to_thread(
                            _remote_gc_hf_cache,
                            REMOTE_SSH_HOST, REMOTE_SSH_KEY, REMOTE_PYTHON,
                            HF_CACHE_DIR, cfg.gc_keep_recent,
                        )
                    else:
                        await asyncio.to_thread(
                            _gc_hf_cache, HF_CACHE_DIR, cfg.gc_keep_recent,
                        )
                except Exception as e:
                    logger.debug(f"[anticopy.worker] gc skipped: {e}")

    async def _get_next_candidates(
        self, *, limit: int = 1,
        exclude: Optional[set] = None,
    ) -> List[Dict[str, Any]]:
        """Return the next ``limit`` candidates sorted by commit time
        (same ordering as ``af get-rank``'s active bucket: earliest
        ``first_block`` first, uid as tiebreaker). Excludes terminated
        miners and anything that already has a ``scores_index`` row.
        ``exclude`` is an optional set of ``(hotkey, revision)`` pairs
        to skip on top of those filters — used by the prefetcher to
        avoid the in-flight candidate's revision."""
        exclude = exclude or set()
        miners_rows = await self.miners_dao.get_valid_miners()
        candidates: List[Dict[str, Any]] = []
        for row in miners_rows:
            uid = int(row.get("uid", 0) or 0)
            if uid <= 0:
                continue                # validator slot / system reserved
            hk = str(row.get("hotkey") or "")
            rev = str(row.get("revision") or "")
            if not hk or not rev:
                continue
            if (hk, rev) in exclude:
                continue
            try:
                stats = await self.miner_stats_dao.get_miner_stats(hk, rev)
            except Exception as e:
                logger.debug(f"[anticopy.worker] stats lookup {hk[:10]}: {e}")
                stats = None
            if stats and stats.get("challenge_status") == "terminated":
                continue
            # ``scores_index`` is the done marker — any row there means
            # the candidate already has a verdict (independent or copy)
            # and shouldn't be re-scored on every loop iteration.
            try:
                score_row = await self.scores_dao.get_score(hk, rev)
            except Exception as e:
                logger.debug(f"[anticopy.worker] score lookup {hk[:10]}: {e}")
                score_row = None
            if score_row is not None:
                continue
            candidates.append(row)
        candidates.sort(
            key=lambda r: (
                int(r.get("first_block", 0) or 0),
                int(r.get("uid", 0) or 0),
            )
        )
        return candidates[:max(0, int(limit))]

    async def _get_next_candidate(self) -> Optional[Dict[str, Any]]:
        """Convenience wrapper around :meth:`_get_next_candidates`."""
        rows = await self._get_next_candidates(limit=1)
        return rows[0] if rows else None

    def stop(self) -> None:
        self._running = False

    # ---- one job -----------------------------------------------------

    async def _run_job(self, job: Dict[str, Any]) -> None:
        hotkey = str(job.get("hotkey", ""))
        revision = str(job.get("revision", ""))
        model = str(job.get("model", ""))
        uid = int(job.get("uid", 0) or 0)
        logger.info(
            f"[anticopy.worker] job claim hotkey={hotkey[:10]} "
            f"rev={revision[:8]} model={model}"
        )
        try:
            cfg = await load_anticopy_config(self.config_dao)
            if not cfg.enabled:
                logger.info("[anticopy.worker] anticopy disabled — sleeping")
                await asyncio.sleep(EMPTY_QUEUE_SLEEP_SEC)
                return

            # 1) Compute the candidate's tokenizer signature once so we
            # can stamp it on the score row for later filtering.
            cand_sig, _ = await compute_tokenizer_signature(model, revision)

            # 2) Stage weights locally + swap the sglang server onto them.
            local_path = await self._fetch_weights(model, revision)
            # Bootstrap sglang on the GPU host if it isn't running yet.
            # First-time setup launches it with this candidate as the
            # initial model so the subsequent ``update_weights`` is a
            # no-op on the very first job; later jobs hit the running
            # engine via ``/update_weights_from_disk``.
            await self._ensure_sglang_running(local_path)
            # Kick prefetch of the next candidate's ckpts now that our
            # own download is finished — sglang's reload is GPU-bound,
            # so the network is idle and prefetch won't contend with
            # the in-flight teacher-force.
            self._maybe_start_prefetch(
                current_hotkey=hotkey, current_revision=revision,
            )
            await self._update_weights(local_path)

            # 3) Teacher-force every eligible rollout.
            t0 = time.time()
            rollouts = await self.rollouts_dao.list_by_tokenizer(
                cand_sig, max_age_days=cfg.pool_days
            )
            logger.info(
                f"[anticopy.worker] {hotkey[:10]} rollout pool: "
                f"{len(rollouts)} rollouts (list_by_tokenizer {time.time()-t0:.1f}s)"
            )
            if not rollouts:
                logger.info(
                    f"[anticopy.worker] {hotkey[:10]} no eligible rollouts; "
                    f"writing empty score"
                )
            t0 = time.time()
            per_rollout = await self._score_all(rollouts)
            logger.info(
                f"[anticopy.worker] {hotkey[:10]} teacher-force complete: "
                f"{len(per_rollout)}/{len(rollouts)} rollouts in {time.time()-t0:.1f}s"
            )

            # 4) Persist score blob + index row.
            # ``ceac.score/v2`` indicates the logprobs were produced against
            # a v2 (full-trajectory + assistant_mask) rollout — i.e. the
            # resp_lp covers only the masked assistant tokens, not the
            # entire response array of the older v1 path. The on-disk shape
            # of per_rollout is the same as v1, but the semantics differ.
            score_payload = {
                "schema": "ceac.score/v2",
                "hotkey": hotkey,
                "revision": revision,
                "model": model,
                "tokenizer_sig": cand_sig,
                "computed_at": int(time.time()),
                "per_rollout": per_rollout,
            }
            r2_key = await asyncio.to_thread(
                self.r2.put_score,
                hotkey=hotkey, revision=revision, payload=score_payload,
            )
            first_block = await self._lookup_first_block(uid, hotkey, revision)
            rollout_keys = [r["rollout_key"] for r in per_rollout]
            await self.scores_dao.upsert(
                hotkey=hotkey,
                revision=revision,
                tokenizer_sig=cand_sig,
                r2_key=r2_key,
                rollout_keys=rollout_keys,
                first_block=first_block,
            )

            # 5) Pairwise vs every existing score; flag copy if found.
            (
                verdict_hotkey,
                decision_med,
                decision_per_env,
                closest_peer_model,
            ) = await self._run_verdict(
                cfg=cfg,
                new_score=score_payload,
                new_first_block=first_block,
            )
            await self.scores_dao.update_verdict(
                hotkey, revision,
                copy_of=verdict_hotkey,
                decision_median=decision_med,
                decision_per_env=decision_per_env,
                closest_peer_model=closest_peer_model,
            )
            per_env_str = " ".join(
                f"{env}={med:.4f}" for env, med in sorted(decision_per_env.items())
            ) or "(no peers)"
            if verdict_hotkey:
                logger.info(
                    f"[anticopy.worker] {hotkey[:10]} verdict "
                    f"copy_of={verdict_hotkey[:10]} dec_med={decision_med:.4f} "
                    f"per_env={{{per_env_str}}} recorded in scores_index"
                )
            elif closest_peer_model:
                logger.info(
                    f"[anticopy.worker] {hotkey[:10]} independent — "
                    f"closest peer {closest_peer_model[:40]} "
                    f"dec_med={decision_med:.4f}"
                )

            # Done — ``scores_index`` row above is the durable marker
            # next iteration uses to skip this miner. No separate
            # ``mark_done`` call needed.
            logger.info(
                f"[anticopy.worker] {hotkey[:10]} done "
                f"rollouts={len(per_rollout)} dec_med={decision_med:.4f} "
                f"per_env={{{per_env_str}}} "
                f"verdict={verdict_hotkey or 'independent'}"
            )
        except Exception as e:
            # No retry counter to bump — the loop will pick this same
            # candidate up next iteration as long as the failure didn't
            # write a ``scores_index`` row. A sleep here avoids
            # hot-looping on a permanently-broken miner; downstream
            # the operator can intervene by terminating the miner via
            # the scheduler.
            logger.error(
                f"[anticopy.worker] {hotkey[:10]} job failed: {e}",
                exc_info=True,
            )
            await asyncio.sleep(EMPTY_QUEUE_SLEEP_SEC)

    # ---- prefetcher --------------------------------------------------

    def _maybe_start_prefetch(
        self, *, current_hotkey: str, current_revision: str,
    ) -> None:
        """Fire-and-forget download of the next candidate's ckpt.

        Re-entrant: cancels any prior prefetch task that's still
        running (the previous "next" may now be the "current", so the
        old prefetch is moot or already done — either way the lock
        inside snapshot_download will serialise).
        """
        prior = self._prefetch_task
        if prior is not None and not prior.done():
            # Don't cancel: HF download is uninterruptible mid-shard
            # anyway. Just stop waiting on it; let it finish in the
            # background.
            pass
        self._prefetch_task = asyncio.create_task(
            self._prefetch_next_pending(
                current_hotkey=current_hotkey,
                current_revision=current_revision,
            )
        )

    async def _prefetch_next_pending(
        self, *, current_hotkey: str, current_revision: str,
    ) -> None:
        """Stage the next 1-2 candidates' ckpts in the background while
        the active job runs. Lookup uses the same filters as the main
        loop (valid + non-terminated + unscored) and excludes the
        current revision so we don't re-download what we just fetched.
        """
        try:
            rows = await self._get_next_candidates(
                limit=2,
                exclude={(current_hotkey, current_revision)},
            )
        except Exception as e:
            logger.debug(f"[anticopy.worker] prefetch lookup failed: {e}")
            return

        for nxt in rows:
            model = str(nxt.get("model", ""))
            revision = str(nxt.get("revision", ""))
            if not model or not revision:
                continue
            key = (model, revision)
            if key in self._prefetched:
                continue
            self._prefetched.add(key)
            logger.info(
                f"[anticopy.worker] prefetch start {model}@{revision[:8]}"
            )
            try:
                if REMOTE_SSH_HOST:
                    await asyncio.to_thread(
                        _remote_snapshot_download,
                        REMOTE_SSH_HOST, REMOTE_SSH_KEY, REMOTE_PYTHON,
                        model, revision, HF_CACHE_DIR,
                    )
                else:
                    await asyncio.to_thread(
                        snapshot_download,
                        repo_id=model,
                        revision=revision,
                        cache_dir=HF_CACHE_DIR,
                        local_files_only=False,
                        token=os.getenv("HF_TOKEN"),
                    )
                logger.info(
                    f"[anticopy.worker] prefetch done {model}@{revision[:8]}"
                )
            except asyncio.CancelledError:
                # main loop is moving on; drop marker so the next
                # iteration's prefetch can pick this back up.
                self._prefetched.discard(key)
                raise
            except Exception as e:
                logger.warning(
                    f"[anticopy.worker] prefetch failed "
                    f"{model}@{revision[:8]}: {e}"
                )
                self._prefetched.discard(key)
                # continue to next candidate — one repo's HF flake
                # shouldn't block downstream prefetches.

    # ---- ckpt staging ------------------------------------------------

    async def _fetch_weights(self, model: str, revision: str) -> str:
        """Idempotent: HF caches the snapshot under
        ``HF_CACHE_DIR/models--{org}--{repo}/snapshots/<sha>``. Returns
        the resolved snapshot path.

        In remote-SSH mode the download is driven over SSH on the GPU
        host, so the returned path is the path *on the GPU host*. The
        sglang ``/update_weights_from_disk`` POST then uses that path
        directly because sglang runs on the same host.
        """
        for attempt in range(1, MAX_FETCH_ATTEMPTS + 1):
            try:
                if REMOTE_SSH_HOST:
                    path = await asyncio.to_thread(
                        _remote_snapshot_download,
                        REMOTE_SSH_HOST, REMOTE_SSH_KEY, REMOTE_PYTHON,
                        model, revision, HF_CACHE_DIR,
                    )
                else:
                    path = await asyncio.to_thread(
                        snapshot_download,
                        repo_id=model,
                        revision=revision,
                        cache_dir=HF_CACHE_DIR,
                        local_files_only=False,
                        token=os.getenv("HF_TOKEN"),
                    )
                return path
            except Exception as e:
                logger.warning(
                    f"[anticopy.worker] snapshot_download attempt {attempt}/"
                    f"{MAX_FETCH_ATTEMPTS} for {model}@{revision[:8]}: {e}"
                )
                if attempt == MAX_FETCH_ATTEMPTS:
                    raise
                await asyncio.sleep(5 * attempt)
        raise RuntimeError("unreachable")

    async def _update_weights(self, model_path: str) -> None:
        """Tell sglang to load new weights from the given on-disk path,
        then wait until ``/model_info`` reports the new path.

        We use fire-and-poll because waiting synchronously on the POST
        is unreliable: sglang stops responding to HTTP keep-alives
        while loading 60 GB of shards (~5 min) so both the SSH tunnel
        and the GPU-host SSH session may tear the response channel
        down before sglang sends the 200, even though the swap
        completes successfully on the server side. Polling
        ``/model_info`` until ``model_path`` matches the requested
        path is the canonical "is the swap done?" check that works
        regardless of which transport flaked.
        """
        expected = model_path.rstrip("/")
        payload = {"model_path": model_path}

        # Fast path: if sglang already serves the expected path AND
        # actually responds to a tiny /generate probe, skip the POST.
        # ``model_path`` alone is not enough — sglang flips it to the
        # new value at the *start* of a reload, so it can appear to
        # match while the server is still 5 minutes away from being
        # responsive. The probe is what catches that state.
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(
                    f"{self.sglang_url}/model_info", timeout=10,
                ) as r:
                    if r.status == 200:
                        info = await r.json()
                        actual = (info.get("model_path") or "").rstrip("/")
                        if actual == expected and await self._sglang_responsive(s):
                            logger.info(
                                f"[anticopy.worker] sglang already serving "
                                f"{expected}, skipping update_weights"
                            )
                            return
        except Exception as e:
            logger.debug(f"[anticopy.worker] pre-update model_info: {e}")

        if REMOTE_SSH_HOST:
            payload_str = _json.dumps(payload)
            # Background the curl on the GPU host so the SSH session
            # itself returns instantly. setsid + nohup detach from the
            # parent so disconnect doesn't kill it; output is dropped
            # on the GPU host (we infer success via polling).
            cmd = (
                "setsid nohup bash -c "
                + shlex.quote(
                    f"curl -sS -X POST -H 'Content-Type: application/json' "
                    f"--max-time 1800 -d {shlex.quote(payload_str)} "
                    f"http://127.0.0.1:{REMOTE_SGLANG_PORT}/update_weights_from_disk "
                    f">/tmp/sglang_update_{REMOTE_SGLANG_PORT}.log 2>&1"
                )
                + " </dev/null >/dev/null 2>&1 &"
            )
            rc, _out, err = await asyncio.to_thread(
                lambda: _ssh_run(
                    REMOTE_SSH_HOST, REMOTE_SSH_KEY, cmd, timeout=60,
                )
            )
            if rc != 0:
                raise _SglangError(
                    f"remote launch update_weights rc={rc}: {err[-200:].strip()}"
                )
        else:
            # local mode: spawn the POST as a background asyncio task
            # so we can poll concurrently
            async def _post():
                try:
                    async with aiohttp.ClientSession() as s:
                        url = f"{self.sglang_url}/update_weights_from_disk"
                        async with s.post(url, json=payload, timeout=1800) as r:
                            return r.status, await r.text()
                except Exception as e:
                    logger.debug(f"[anticopy.worker] local update_weights post: {e}")
                    return None, str(e)
            asyncio.create_task(_post())

        # Now poll until sglang BOTH reports the expected path AND
        # responds to a tiny /generate probe. ``model_path`` flipping
        # alone isn't enough — sglang updates that field at reload
        # start, well before the new weights are usable.
        deadline = time.time() + 1800     # 30 minutes max
        last_observed = ""
        async with aiohttp.ClientSession() as s:
            while time.time() < deadline:
                try:
                    async with s.get(
                        f"{self.sglang_url}/model_info", timeout=10,
                    ) as r:
                        if r.status == 200:
                            info = await r.json()
                            actual = (info.get("model_path") or "").rstrip("/")
                            if actual != last_observed:
                                logger.info(
                                    f"[anticopy.worker] update_weights wait: "
                                    f"actual={actual} expected={expected}"
                                )
                                last_observed = actual
                            if actual == expected and await self._sglang_responsive(s):
                                logger.info(
                                    f"[anticopy.worker] sglang serving {expected}"
                                )
                                return
                except Exception as e:
                    logger.debug(f"[anticopy.worker] model_info poll: {e}")
                await asyncio.sleep(15)
        raise _SglangError(
            f"update_weights polling timed out after 30min; "
            f"never saw model_path={expected}"
        )

    async def _ensure_sglang_running(self, model_path: str) -> None:
        """Make sure an sglang server is up on the remote host. If
        ``/model_info`` already answers, return immediately (the running
        instance will be steered via ``update_weights_from_disk``); if
        the port is dead we SSH-exec ``python -m sglang.launch_server``
        with the supplied model as the initial weights so subsequent
        ``update_weights`` calls work against a live engine.

        Only effective in ``REMOTE_SSH_HOST`` mode; local-sglang
        deployments are expected to be brought up by the operator
        (compose / systemd).
        """
        # Probe first. We accept any 200 — even if it says a different
        # model is loaded, ``update_weights`` will swap correctly.
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(
                    f"{self.sglang_url}/model_info", timeout=10,
                ) as r:
                    if r.status == 200:
                        return
        except Exception:
            pass

        if not REMOTE_SSH_HOST:
            raise _SglangError(
                f"sglang not reachable at {self.sglang_url} and no "
                f"REMOTE_SSH_HOST configured — bring sglang up manually"
            )

        # SSH-exec the launcher under setsid+nohup so the engine
        # survives the SSH session that started it. Logs land in
        # ``/tmp/sglang_<port>.log`` for post-mortem.
        #
        # We intentionally do NOT touch ``CUDA_VISIBLE_DEVICES`` here —
        # ``--base-gpu-id`` already steers the process to the right GPU,
        # and any explicit env mask (especially an empty one, which
        # masks *all* GPUs) would clash with that and crash sglang at
        # ``get_device()`` with "no accelerator available".
        launch = (
            f"{shlex.quote(REMOTE_PYTHON)} -m sglang.launch_server "
            f"--model-path {shlex.quote(model_path)} "
            f"--host 127.0.0.1 --port {REMOTE_SGLANG_PORT} "
            f"--base-gpu-id {SGLANG_BASE_GPU_ID} "
            f"--trust-remote-code "
            f"{SGLANG_EXTRA_ARGS}"
        )
        cmd = (
            "setsid nohup bash -c "
            + shlex.quote(
                f"{launch} > /tmp/sglang_{REMOTE_SGLANG_PORT}.log 2>&1"
            )
            + " </dev/null >/dev/null 2>&1 &"
        )
        logger.info(
            f"[anticopy.worker] launching remote sglang gpu={SGLANG_BASE_GPU_ID} "
            f"port={REMOTE_SGLANG_PORT} model={os.path.basename(model_path)}"
        )
        rc, _out, err = await asyncio.to_thread(
            lambda: _ssh_run(
                REMOTE_SSH_HOST, REMOTE_SSH_KEY, cmd, timeout=60,
            )
        )
        if rc != 0:
            raise _SglangError(
                f"remote sglang launch rc={rc}: {err[-300:].strip()}"
            )

        # Poll until /model_info answers AND the engine actually
        # responds to a tiny /generate. Loading 60 GB shards on a B300
        # takes ~3-5 min, so allow generous deadline.
        deadline = time.time() + 900
        async with aiohttp.ClientSession() as s:
            while time.time() < deadline:
                try:
                    async with s.get(
                        f"{self.sglang_url}/model_info", timeout=10,
                    ) as r:
                        if r.status == 200 and await self._sglang_responsive(s):
                            logger.info(
                                f"[anticopy.worker] sglang up on "
                                f"{self.sglang_url} (gpu={SGLANG_BASE_GPU_ID})"
                            )
                            return
                except Exception as e:
                    logger.debug(f"[anticopy.worker] sglang startup poll: {e}")
                await asyncio.sleep(10)
        raise _SglangError(
            f"sglang launch on {self.sglang_url} never became ready after 15min"
        )

    async def _sglang_responsive(self, session: "aiohttp.ClientSession") -> bool:
        """Probe sglang with a 2-token prefill. Returns True iff the
        server completes the request in <30 s — proving the engine is
        actually serving the loaded weights, not just reporting the
        path while still loading shards."""
        try:
            async with session.post(
                f"{self.sglang_url}/generate",
                json={
                    "input_ids": [[1, 2]],
                    "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
                    "return_logprob": False,
                },
                timeout=30,
            ) as r:
                return r.status == 200
        except Exception:
            return False

    async def _wait_ready(self, timeout_s: int = 300) -> None:
        deadline = time.time() + timeout_s
        async with aiohttp.ClientSession() as s:
            while time.time() < deadline:
                try:
                    async with s.get(
                        f"{self.sglang_url}/v1/models", timeout=10
                    ) as r:
                        if r.status == 200:
                            return
                except aiohttp.ClientError:
                    pass
                await asyncio.sleep(2)
        raise _SglangError("sglang readiness timeout after weight swap")

    # ---- teacher forcing --------------------------------------------

    async def _score_all(
        self, rollouts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        n = len(rollouts)
        for idx, row in enumerate(rollouts, start=1):
            try:
                blob = await asyncio.to_thread(
                    self.r2.get_rollout_by_key, row["r2_key"]
                )
            except Exception as e:
                logger.warning(
                    f"[anticopy.worker] r2 get rollout failed {row.get('r2_key')}: {e}"
                )
                continue
            if not blob:
                continue
            t0 = time.time()
            try:
                schema = blob.get("schema", "ceac.rollout/v1")
                if schema == "ceac.rollout/v2":
                    lp, top = await self._teacher_force_v2(
                        tokens=blob["tokens"],
                        assistant_mask=blob["assistant_mask"],
                    )
                else:
                    lp, top = await self._teacher_force(
                        prompt_ids=blob["prompt_ids"],
                        response_ids=blob["response_ids"],
                    )
            except Exception as e:
                logger.warning(
                    f"[anticopy.worker] teacher_force failed "
                    f"{row.get('rollout_key')}: {e}"
                )
                continue
            out.append({
                "rollout_key": row["rollout_key"],
                "env": row.get("env", ""),
                "n_tokens": len(lp),
                "resp_lp": lp,
                "resp_top": top,
            })
            # Cheap heartbeat — every rollout, with timing so operators
            # can spot stalls (sglang slow / R2 slow) without a stack
            # dump.
            if idx == 1 or idx % 5 == 0 or idx == n:
                logger.info(
                    f"[anticopy.worker] tf progress {idx}/{n} "
                    f"env={row.get('env','?'):14s} n_tokens={len(lp)} "
                    f"dt={time.time()-t0:.1f}s"
                )
        return out

    async def _teacher_force_v2(
        self, *, tokens: List[int], assistant_mask: List[bool],
    ) -> tuple:
        """Schema-v2 teacher-force: prefill the entire trajectory and
        return per-token logprobs only at positions where
        ``assistant_mask`` is True.

        Resilience: ``sock_read=120`` detects SSH-tunnel silent stalls
        (TCP socket alive but no bytes flowing — happens occasionally
        on long-lived tunneled HTTP connections); each failed attempt
        opens a *fresh* ``ClientSession`` so the broken connection in
        the pool doesn't get reused.
        """
        body = {
            "input_ids": [list(tokens)],
            "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
            "return_logprob": True,
            "logprob_start_len": 0,
            "top_logprobs_num": 20,
        }
        last_err: Optional[Exception] = None
        for attempt in range(1, 4):
            try:
                connector = aiohttp.TCPConnector(force_close=True)
                timeout = aiohttp.ClientTimeout(total=600, sock_read=120)
                async with aiohttp.ClientSession(
                    connector=connector, timeout=timeout,
                ) as s:
                    async with s.post(
                        f"{self.sglang_url}/generate", json=body,
                    ) as r:
                        if r.status != 200:
                            txt = await r.text()
                            raise _SglangError(f"/generate {r.status}: {txt[:200]}")
                        payload = await r.json()
                return _parse_sglang_logprobs_v2(payload, assistant_mask)
            except (
                asyncio.TimeoutError,
                aiohttp.ClientError,
                _SglangError,
            ) as e:
                last_err = e
                logger.warning(
                    f"[anticopy.worker] _teacher_force_v2 attempt "
                    f"{attempt}/3 failed: {type(e).__name__}: {e}"
                )
                if attempt < 3:
                    await asyncio.sleep(2 * attempt)
        raise last_err or _SglangError("teacher_force_v2 exhausted retries")

    async def _teacher_force(
        self, *, prompt_ids: List[int], response_ids: List[int],
    ) -> tuple:
        """Prefill ``prompt_ids + response_ids`` on sglang and return
        per-token logprobs over the response positions. Returns
        ``(resp_lp, resp_top)``.

        resp_lp[i] = logprob the model assigned to ``response_ids[i]``.
        resp_top[i] = [[lp, token_id], ...] top-K alternatives at i.

        sglang protocol detail: ``logprob_start_len = N`` means return
        logprobs for input tokens ``[N, N+1, ..., len(input)-1]`` —
        and entry 0 (the boundary token at ``N``) comes back with
        ``lp=None``. To recover all ``len(response_ids)`` logprobs we
        ask for ``start_len = prompt_len - 1`` and drop the first
        boundary entry.
        """
        full_ids = list(prompt_ids) + list(response_ids)
        # sglang >= 0.5.10 expects ``input_ids`` as ``list[list[int]]``
        # (batched), even for one sequence. Wrap in a 1-element batch
        # and unwrap the response.
        body = {
            "input_ids": [full_ids],
            # max_new_tokens=1 is required by sglang; we discard the
            # one sampled token, all the data we need is in the prefill
            # logprobs returned via ``input_token_logprobs``.
            "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
            "return_logprob": True,
            "logprob_start_len": max(0, len(prompt_ids) - 1),
            "top_logprobs_num": 20,
        }
        last_err: Optional[Exception] = None
        for attempt in range(1, 4):
            try:
                connector = aiohttp.TCPConnector(force_close=True)
                timeout = aiohttp.ClientTimeout(total=600, sock_read=120)
                async with aiohttp.ClientSession(
                    connector=connector, timeout=timeout,
                ) as s:
                    async with s.post(
                        f"{self.sglang_url}/generate", json=body,
                    ) as r:
                        if r.status != 200:
                            txt = await r.text()
                            raise _SglangError(f"/generate {r.status}: {txt[:200]}")
                        payload = await r.json()
                return _parse_sglang_logprobs(payload, n_response=len(response_ids))
            except (asyncio.TimeoutError, aiohttp.ClientError, _SglangError) as e:
                last_err = e
                logger.warning(
                    f"[anticopy.worker] _teacher_force attempt "
                    f"{attempt}/3 failed: {type(e).__name__}: {e}"
                )
                if attempt < 3:
                    await asyncio.sleep(2 * attempt)
        raise last_err or _SglangError("teacher_force exhausted retries")

    # ---- verdict + invalid mark -------------------------------------

    async def _run_verdict(
        self,
        *,
        cfg: AntiCopyConfig,
        new_score: Dict[str, Any],
        new_first_block: int,
    ) -> tuple:
        """Compute the verdict for ``new_score`` AND retroactively refresh
        the verdicts of any later-committed peer that turns out to be a
        copy of it.

        Returns ``(copy_of_hotkey, overlap_max)`` for ``new_score``.

        Why the retroactive pass: candidates are not necessarily scored
        in commit order (priority overrides, retries, parallel workers).
        Without this pass, an earlier-committed candidate that arrives
        in the queue late would be flagged ``independent`` correctly,
        but the already-scored later peers that copied it would keep a
        stale ``verdict_copy_of`` pointing at whichever earlier peer
        was visible when they were scored — or at nothing.
        """
        index_rows = await self.scores_dao.list_all()
        peer_scores: List[Dict[str, Any]] = []
        for row in index_rows:
            if (
                row.get("hotkey") == new_score.get("hotkey")
                and row.get("revision") == new_score.get("revision")
            ):
                continue
            r2_key = row.get("r2_key")
            if not r2_key:
                continue
            try:
                blob = await asyncio.to_thread(self.r2.get_score_by_key, r2_key)
            except Exception as e:
                logger.debug(f"[anticopy.worker] peer score fetch failed {r2_key}: {e}")
                continue
            if not blob:
                continue
            blob["first_block"] = int(row.get("first_block", 0) or 0)
            peer_scores.append(blob)

        decision = detect_copies(
            new_score,
            new_first_block,
            peer_scores,
            nll_threshold=cfg.nll_threshold,
            min_overlap=cfg.min_overlap,
            agreement_ratio=cfg.agreement_ratio,
        )

        # Retroactive pass: any peer that committed AFTER new_score and
        # is a copy of it might now have an earlier origin (new_score
        # itself, or something detect_copies finds when we re-evaluate
        # with new_score included). Refresh those peers' scores_index
        # rows in place. We only re-evaluate peers we have evidence to
        # update, so this stays O(n_overlapping_copies).
        await self._refresh_later_peer_verdicts(
            cfg=cfg,
            new_score=new_score,
            new_first_block=new_first_block,
            peer_scores=peer_scores,
        )

        return (
            decision.copy_of_hotkey,
            decision.decision_median,
            dict(decision.decision_per_env),
            decision.closest_peer_model,
        )

    async def _refresh_later_peer_verdicts(
        self,
        *,
        cfg: AntiCopyConfig,
        new_score: Dict[str, Any],
        new_first_block: int,
        peer_scores: List[Dict[str, Any]],
    ) -> None:
        """For each peer committed strictly later than ``new_score`` and
        pairwise-similar to it, recompute the peer's full ``detect_copies``
        verdict (with ``new_score`` now included as a candidate origin)
        and write the refreshed result back to ``anticopy_scores_index``.

        Only peers that are *copies of* ``new_score`` are touched —
        independents stay independent, and copies pointed at an older
        peer than ``new_score`` are re-checked but typically unchanged.
        """
        new_hotkey = new_score.get("hotkey", "")
        for peer in peer_scores:
            peer_hk = peer.get("hotkey", "")
            peer_rev = peer.get("revision", "")
            peer_first = int(peer.get("first_block", 0) or 0)
            # Strict later-committer check (matches detect_copies' ordering).
            if (peer_first, peer_hk) <= (new_first_block, new_hotkey):
                continue
            # Quick prune: is peer a copy of new_score at all? If the
            # pair isn't above-threshold, peer's existing verdict can't
            # be affected by the new arrival.
            pair = compare_scores(peer, new_score)
            if pair.n_overlap_tokens < cfg.min_overlap:
                continue
            if not is_copy_verdict(
                pair,
                nll_threshold=cfg.nll_threshold,
                agreement_ratio=cfg.agreement_ratio,
            ):
                continue
            # Full re-evaluation: pick the earliest origin among ALL
            # other peers + new_score. (Excluding peer itself.)
            others = [p for p in peer_scores if p is not peer] + [new_score]
            refreshed = detect_copies(
                peer, peer_first, others,
                nll_threshold=cfg.nll_threshold,
                min_overlap=cfg.min_overlap,
                agreement_ratio=cfg.agreement_ratio,
            )
            try:
                await self.scores_dao.update_verdict(
                    peer_hk, peer_rev,
                    copy_of=refreshed.copy_of_hotkey,
                    decision_median=refreshed.decision_median,
                    decision_per_env=dict(refreshed.decision_per_env),
                    closest_peer_model=refreshed.closest_peer_model,
                )
                logger.info(
                    f"[anticopy.worker] retroactive verdict {peer_hk[:10]} "
                    f"copy_of={(refreshed.copy_of_hotkey or 'independent')[:14]} "
                    f"dec_med={refreshed.decision_median:.4f} "
                    f"(after {new_hotkey[:10]} arrived)"
                )
            except Exception as e:
                logger.warning(
                    f"[anticopy.worker] retroactive verdict update "
                    f"failed for {peer_hk[:10]}: {e}"
                )


    async def _lookup_first_block(
        self, uid: int, hotkey: str, revision: str,
    ) -> int:
        try:
            row = await self.miners_dao.get_miner_by_uid(uid)
        except Exception:
            row = None
        if not row:
            return 0
        if row.get("hotkey") != hotkey or row.get("revision") != revision:
            return 0
        try:
            return int(row.get("first_block", 0) or 0)
        except (TypeError, ValueError):
            return 0


def _remote_snapshot_download(
    host: str, key_path: str, py_path: str,
    model: str, revision: str, cache_dir: str,
) -> str:
    """Run ``huggingface_hub.snapshot_download`` on the SSH host and
    return the resolved snapshot path. Raises ``RuntimeError`` on any
    non-zero exit or empty stdout.

    Two-step strategy avoids the multi-minute cache-verify path
    huggingface_hub takes by default:

      1. Try ``local_files_only=True`` first. If the snapshot is
         already fully present on disk this returns in <1s — no etag
         round-trip, no sha256 verify. ~5 min saved per cached ckpt.
      2. On miss (LocalEntryNotFoundError), fall through to the real
         download with ``local_files_only=False``.
    """
    # Step 1: ``local_files_only=True`` + strict completeness check.
    # huggingface_hub returns a snapshot dir even when some shard
    # symlinks point at incomplete blobs; if we hand that to sglang's
    # ``/update_weights_from_disk`` it will 400 "Cannot find any model
    # weights". Verify locally that every safetensors shard referenced
    # by ``model.safetensors.index.json`` resolves to a non-empty
    # regular file before declaring the cache complete.
    probe = (
        "import os, json, sys; "
        "from huggingface_hub import snapshot_download; "
        "from huggingface_hub.errors import LocalEntryNotFoundError;\n"
        f"try:\n    p = snapshot_download(repo_id={model!r}, revision={revision!r}, "
        f"cache_dir={cache_dir!r}, local_files_only=True, "
        "token=os.getenv('HF_TOKEN'))\nexcept Exception:\n    print('MISS'); sys.exit(0)\n"
        "idx = os.path.join(p, 'model.safetensors.index.json')\n"
        "if not os.path.exists(idx):\n    print('MISS'); sys.exit(0)\n"
        "with open(idx) as fh:\n    m = json.load(fh).get('weight_map') or {}\n"
        "shards = set(m.values())\n"
        "ok = True\n"
        "for s in shards:\n"
        "    f = os.path.join(p, s); rp = os.path.realpath(f)\n"
        "    if not os.path.isfile(rp) or os.path.getsize(rp) < 100*1024*1024:\n"
        "        ok = False; break\n"
        "print(p if ok else 'MISS')"
    )
    cmd = f"{shlex.quote(py_path)} -c {shlex.quote(probe)}"
    rc, out, err = _ssh_run(host, key_path, cmd, timeout=60)
    if rc == 0:
        line = (out.strip().splitlines() or [""])[-1].strip()
        if line and line != "MISS":
            return line

    # Step 2: real download (handles resume + partial cache).
    body = (
        "from huggingface_hub import snapshot_download; "
        "import os; "
        f"p = snapshot_download(repo_id={model!r}, revision={revision!r}, "
        f"cache_dir={cache_dir!r}, local_files_only=False, "
        "token=os.getenv('HF_TOKEN')); "
        "print(p)"
    )
    cmd = f"{shlex.quote(py_path)} -c {shlex.quote(body)}"
    rc, out, err = _ssh_run(host, key_path, cmd, timeout=3600)
    if rc != 0:
        raise RuntimeError(
            f"remote snapshot_download rc={rc}: {err[-200:].strip()}"
        )
    lines = [ln for ln in out.strip().splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError(
            f"remote snapshot_download produced no path; stderr={err[-200:].strip()}"
        )
    return lines[-1]


def _remote_gc_hf_cache(
    host: str, key_path: str, py_path: str,
    cache_dir: str, keep_recent: int,
) -> Dict[str, int]:
    """Same contract as :func:`_gc_hf_cache` but executed on the SSH host."""
    keep_recent = max(2, int(keep_recent))
    body = (
        "from huggingface_hub import scan_cache_dir; import json; "
        f"info = scan_cache_dir({cache_dir!r}); "
        "repos = sorted(info.repos, key=lambda r: int(getattr(r,'last_modified',0) or 0), reverse=True); "
        f"survivors = repos[:{keep_recent}]; casualties = repos[{keep_recent}:]; "
        "ch = [rev.commit_hash for r in casualties for rev in (r.revisions or [])]; "
        "freed = 0;\n"
        "if ch:\n"
        "    strat = info.delete_revisions(*ch); "
        "    freed = int(getattr(strat,'expected_freed_size',0) or 0); "
        "    strat.execute()\n"
        "print(json.dumps({'kept': len(survivors), 'deleted_revisions': len(ch), 'freed_mb': freed//(1024*1024)}))"
    )
    cmd = f"{shlex.quote(py_path)} -c {shlex.quote(body)}"
    rc, out, err = _ssh_run(host, key_path, cmd, timeout=300)
    if rc != 0:
        logger.debug(f"[anticopy.worker] remote gc rc={rc} err={err[-200:]}")
        return {"kept": 0, "deleted_revisions": 0, "freed_mb": 0}
    try:
        return _json.loads((out.strip().splitlines() or ["{}"])[-1])
    except Exception:
        return {"kept": 0, "deleted_revisions": 0, "freed_mb": 0}


def _gc_hf_cache(cache_dir: str, keep_recent: int) -> Dict[str, int]:
    """Trim the HF cache so only the ``keep_recent`` most recently-used
    repos survive. The current sglang model is one of the most-recent
    by definition (its weights were just read for an /update_weights);
    the prefetched next ckpt is another. ``keep_recent`` should be at
    least 2.

    Uses ``huggingface_hub.scan_cache_dir`` so we touch the right
    bookkeeping files (blobs / refs / snapshots) instead of a blind
    ``rm -rf models--*``.

    Returns ``{"kept": n, "deleted_revisions": m, "freed_mb": k}``.
    """
    keep_recent = max(2, int(keep_recent))
    try:
        info = scan_cache_dir(cache_dir)
    except Exception:
        return {"kept": 0, "deleted_revisions": 0, "freed_mb": 0}

    repos = sorted(
        info.repos,
        key=lambda r: int(getattr(r, "last_modified", 0) or 0),
        reverse=True,
    )
    survivors = repos[:keep_recent]
    casualties = repos[keep_recent:]
    if not casualties:
        return {"kept": len(survivors), "deleted_revisions": 0, "freed_mb": 0}

    commit_hashes = []
    for repo in casualties:
        for rev in getattr(repo, "revisions", []) or []:
            ch = getattr(rev, "commit_hash", None)
            if ch:
                commit_hashes.append(ch)
    if not commit_hashes:
        return {"kept": len(survivors), "deleted_revisions": 0, "freed_mb": 0}

    try:
        strategy = info.delete_revisions(*commit_hashes)
        freed_bytes = int(getattr(strategy, "expected_freed_size", 0) or 0)
        strategy.execute()
    except Exception as e:
        logger.warning(f"[anticopy.worker] gc execute failed: {e}")
        return {"kept": len(survivors), "deleted_revisions": 0, "freed_mb": 0}

    return {
        "kept": len(survivors),
        "deleted_revisions": len(commit_hashes),
        "freed_mb": freed_bytes // (1024 * 1024),
    }


def _parse_sglang_logprobs_v2(
    payload: Dict[str, Any], assistant_mask: List[bool],
) -> tuple:
    """v2 parser: read ``input_token_logprobs`` aligned with the input
    tokens (length == len(tokens)), then select only positions where
    ``assistant_mask`` is True. Returns ``(resp_lp, resp_top)`` with
    same length == sum(assistant_mask) (skipping any non-finite
    boundary entries)."""
    src = payload
    if isinstance(payload, list):
        src = payload[0] if payload else {}
    meta = src.get("meta_info") or {}
    lp_raw = meta.get("input_token_logprobs") or src.get("input_token_logprobs") or []
    top_raw = meta.get("input_top_logprobs") or src.get("input_top_logprobs") or []

    resp_lp: List[Optional[float]] = []
    resp_top: List[List[List[Any]]] = []
    n = min(len(lp_raw), len(assistant_mask))
    for i in range(n):
        if not assistant_mask[i]:
            continue
        entry = lp_raw[i]
        if entry is None:                  # sglang's boundary slot (i=0)
            continue
        try:
            lp = float(entry[0]) if entry[0] is not None else None
        except (TypeError, ValueError, IndexError):
            lp = None
        if lp is None:
            continue
        resp_lp.append(lp)
        slot_out: List[List[Any]] = []
        if i < len(top_raw) and top_raw[i]:
            for e in top_raw[i]:
                try:
                    slot_out.append([float(e[0]), int(e[1])])
                except (TypeError, ValueError, IndexError):
                    continue
        resp_top.append(slot_out)
    return resp_lp, resp_top


def _parse_sglang_logprobs(
    payload: Dict[str, Any], *, n_response: Optional[int] = None,
) -> tuple:
    """sglang /generate logprob response → (resp_lp, resp_top).

    Reads ``meta_info.input_token_logprobs`` as a list of
    ``[logprob, token_id, str_or_null]`` aligned with
    ``input_top_logprobs``.

    When ``n_response`` is given, the function assumes the caller
    asked for ``logprob_start_len = prompt_len - 1``: it strips the
    first boundary entry (whose ``lp`` is ``None`` by sglang's
    contract) and slices the next ``n_response`` entries — these are
    exactly the response-position logprobs. When ``n_response`` is
    None we return the raw arrays as-is (legacy / debug path).
    """
    src = payload
    if isinstance(payload, list):                  # batched: take first
        src = payload[0] if payload else {}

    meta = src.get("meta_info") or {}
    lp_raw = meta.get("input_token_logprobs") or src.get("input_token_logprobs") or []
    top_raw = meta.get("input_top_logprobs") or src.get("input_top_logprobs") or []

    if n_response is not None:
        lp_raw = lp_raw[1:1 + int(n_response)]
        top_raw = top_raw[1:1 + int(n_response)]

    resp_lp: List[Optional[float]] = []
    for entry in lp_raw:
        if entry is None:
            resp_lp.append(None)
            continue
        try:
            resp_lp.append(float(entry[0]))
        except (TypeError, ValueError, IndexError):
            resp_lp.append(None)

    resp_top: List[List[List[Any]]] = []
    for slot in top_raw:
        if not slot:
            resp_top.append([])
            continue
        slot_out: List[List[Any]] = []
        for entry in slot:
            try:
                slot_out.append([float(entry[0]), int(entry[1])])
            except (TypeError, ValueError, IndexError):
                continue
        resp_top.append(slot_out)
    return resp_lp, resp_top
