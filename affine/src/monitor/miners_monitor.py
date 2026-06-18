"""
Miners monitor.

Reads metagraph commits and miner metadata. For every miner on the subnet:

  1. Read its on-chain commit ``{model, revision}``.
  2. Validate model on HuggingFace: revision exists, weight hashes
     computable, no duplicate-repo / suspicious commit history.
  3. Optional: model size + chat-template safety checks.
  4. Plagiarism detection by ``model_hash`` collision (earliest committer
     wins; later miners with the same hash are marked invalid).
  5. Persist to ``miners`` table. Challenge lifecycle is historical state
     owned by ``miner_stats`` and is not stored on the current snapshot.

Anti-copy checks run here; inference lifecycle is handled by the scheduler.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url
from huggingface_hub.errors import (
    DisabledRepoError,
    GatedRepoError,
    RepositoryNotFoundError,
)

from affine.core.setup import logger
from affine.database.dao.anticopy import (
    AntiCopyScoresIndexDAO,
    AntiCopyStateDAO,
)
from affine.database.dao.miner_stats import MinerStatsDAO
from affine.database.dao.miners import MinersDAO
from affine.database.dao.system_config import SystemConfigDAO
from affine.src.anticopy.threshold import load_anticopy_config
from affine.src.anticopy.tokenizer_sig import compute_tokenizer_signature
from affine.utils.model_size_checker import (
    QWEN36_MODEL_TYPE,
    QWEN36_ONLY_MODEL_TYPES,
    check_model_size,
)
from affine.utils.subtensor import get_subtensor
from affine.utils.template_checker import check_template_safety


NETUID = int(os.getenv("AFFINE_NETUID", "120"))

MULTI_COMMIT_ENFORCE_BLOCK = 7_710_000
REPO_HOTKEY_SUFFIX_ENFORCE_BLOCK = 7_290_000
# 8431035 = 2026-06-18 03:00:00 UTC / 2026-06-18 11:00:00 Beijing.
QWEN36_ONLY_ENFORCE_BLOCK = 8_431_035
QWEN36_ALLOWED_FROM_BLOCK = 8_432_280


@dataclass
class MinerInfo:
    uid: int
    hotkey: str
    model: str
    revision: str
    block: int = 0
    is_valid: bool = False
    invalid_reason: Optional[str] = None
    permanent_invalid: bool = False
    terminate_stats: bool = False
    model_hash: str = ""
    hf_revision: str = ""
    template_check_result: Optional[str] = None  # "safe" | "unsafe:<reason>" | None
    tokenizer_sig: str = ""
    model_type: str = ""

    def key(self) -> str:
        return f"{self.hotkey}#{self.revision}"

    def mark_invalid(
        self, reason: str, *, permanent: bool, terminate_stats: bool = False
    ) -> None:
        self.is_valid = False
        self.invalid_reason = reason
        self.permanent_invalid = permanent
        self.terminate_stats = terminate_stats


def _system_miner_info(uid: int, payload: Dict[str, object]) -> Optional[MinerInfo]:
    """Build the synthetic miner row for a configured system miner."""
    if uid <= 1000:
        return None
    suffix = uid - 1000
    revision = str(payload.get("revision") or f"SYSTEM-{suffix}")
    return MinerInfo(
        uid=uid,
        hotkey=f"SYSTEM-{suffix}",
        model=str(payload.get("model") or ""),
        revision=revision,
        block=0,
        is_valid=True,
        invalid_reason=None,
        model_hash="",
        hf_revision=revision,
        template_check_result="safe",
        model_type=str(payload.get("model_type") or ""),
    )


class HFRepoUnavailable(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class MinersMonitor:
    """Refreshes current miner snapshot and historical miner metadata."""

    _instance: Optional["MinersMonitor"] = None
    _lock = asyncio.Lock()

    def __init__(self, refresh_interval_seconds: int = 300):
        self.dao = MinersDAO()
        self.stats_dao = MinerStatsDAO()
        self.config_dao = SystemConfigDAO()
        self.anticopy_scores_dao = AntiCopyScoresIndexDAO()
        self.anticopy_state_dao = AntiCopyStateDAO()
        self.refresh_interval_seconds = refresh_interval_seconds
        self.last_update: int = 0
        # (model, revision) -> (cached result or None, fetched_at)
        self._weights_cache: Dict[Tuple[str, str], Tuple[Optional[Tuple[str, str, str]], float]] = {}
        self._weights_ttl_sec = 1800
        # (model, revision) -> (tokenizer_sig_or_empty, fetched_at)
        self._tokenizer_sig_cache: Dict[Tuple[str, str], Tuple[str, float]] = {}
        self._background_task: Optional[asyncio.Task] = None

    # ---- lifecycle -------------------------------------------------------

    @classmethod
    async def initialize(cls, refresh_interval_seconds: int = 300) -> "MinersMonitor":
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls(refresh_interval_seconds=refresh_interval_seconds)
                await cls._instance.start_background_tasks()
        return cls._instance

    @classmethod
    def get_instance(cls) -> "MinersMonitor":
        if cls._instance is None:
            raise RuntimeError("MinersMonitor.initialize() must be called first")
        return cls._instance

    async def start_background_tasks(self) -> None:
        if self._background_task and not self._background_task.done():
            return
        self._background_task = asyncio.create_task(self._refresh_loop())

    async def stop_background_tasks(self) -> None:
        if self._background_task and not self._background_task.done():
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
        self._background_task = None

    async def _refresh_loop(self) -> None:
        while True:
            try:
                await self.refresh_miners()
            except Exception as e:
                logger.error(f"[MinersMonitor] refresh failed: {e}", exc_info=True)
            await asyncio.sleep(self.refresh_interval_seconds)

    # ---- main refresh ----------------------------------------------------

    async def refresh_miners(self) -> Dict[str, MinerInfo]:
        logger.info("[MinersMonitor] refreshing miners from metagraph...")
        subtensor = await get_subtensor()
        meta = await subtensor.metagraph(NETUID)
        commits = await subtensor.get_all_revealed_commitments(NETUID)
        current_block = int(await subtensor.get_current_block())
        blacklist = await self._load_blacklist()

        miners: list[MinerInfo] = []
        for uid in range(len(meta.hotkeys)):
            hotkey = meta.hotkeys[uid]

            if hotkey in blacklist:
                m = MinerInfo(uid=uid, hotkey=hotkey, model="", revision="")
                m.mark_invalid("blacklisted", permanent=True)
                miners.append(m)
                continue

            if hotkey not in commits:
                m = MinerInfo(uid=uid, hotkey=hotkey, model="", revision="")
                m.mark_invalid("no_commit", permanent=False)
                miners.append(m)
                continue

            try:
                block, commit_data = commits[hotkey][-1]
                data = json.loads(commit_data)
                model = data.get("model", "")
                revision = data.get("revision", "")
                if not model or not revision:
                    m = MinerInfo(
                        uid=uid, hotkey=hotkey, model=model, revision=revision,
                        block=int(block) if uid != 0 else 0,
                    )
                    m.mark_invalid("incomplete_commit:missing_fields", permanent=False)
                    miners.append(m)
                    continue

                miners.append(
                    await self._validate_miner(
                        uid=uid,
                        hotkey=hotkey,
                        model=model,
                        revision=revision,
                        block=int(block) if uid != 0 else 0,
                        commit_count=len(commits[hotkey]),
                    )
                )
            except json.JSONDecodeError:
                m = MinerInfo(uid=uid, hotkey=hotkey, model="", revision="")
                m.mark_invalid("invalid_json_commit", permanent=True)
                miners.append(m)
            except Exception as e:
                logger.debug(f"[MinersMonitor] validate uid={uid} failed: {e}")
                m = MinerInfo(uid=uid, hotkey=hotkey, model="", revision="")
                m.mark_invalid(f"validation_error:{str(e)[:60]}", permanent=False)
                miners.append(m)

        miners = self._detect_plagiarism(miners)

        # System miners — bench models (uid > 1000). They short-circuit
        # every validation step and are simply persisted as valid.
        system_miners = await self.config_dao.get_system_miners()
        for uid_str, payload in system_miners.items():
            uid = int(uid_str)
            info = _system_miner_info(uid, payload)
            if info is not None:
                miners.append(info)

        # Persist current online snapshot. Historical challenge state lives
        # in miner_stats, keyed by (hotkey, revision).
        await self._persist_miners(miners, current_block=current_block)

        self.last_update = int(time.time())
        valid = {m.key(): m for m in miners if m.is_valid}
        logger.info(
            f"[MinersMonitor] refreshed {len(miners)} miners "
            f"({len(valid)} valid, {len(miners) - len(valid)} invalid)"
        )
        return valid

    # ---- validation -----------------------------------------------------

    async def _validate_miner(
        self,
        *,
        uid: int,
        hotkey: str,
        model: str,
        revision: str,
        block: int,
        commit_count: int = 1,
    ) -> MinerInfo:
        info = MinerInfo(uid=uid, hotkey=hotkey, model=model, revision=revision, block=block)

        # Inherit prior template_check_result so a one-cycle HF blip doesn't
        # nuke the cached safety verdict.
        try:
            existing = await self.dao.get_miner_by_uid(uid)
            if existing and existing.get("model") == model and existing.get("revision") == revision:
                info.template_check_result = existing.get("template_check_result")
        except Exception:
            pass

        # Multi-commit rule: a hotkey is only allowed one commit on this subnet.
        if uid != 0 and commit_count > 1 and block >= MULTI_COMMIT_ENFORCE_BLOCK:
            info.mark_invalid(
                f"multiple_commits:count={commit_count}",
                permanent=True,
                terminate_stats=True,
            )
            return info

        # "affine" must appear in the model name (system miners exempt).
        if uid != 0 and "affine" not in model.lower():
            info.mark_invalid(
                "model_name_missing_affine",
                permanent=True,
                terminate_stats=True,
            )
            return info

        # Repo name must end with the hotkey from a certain block onward —
        # makes plagiarism via repo rename impossible.
        if uid != 0 and block >= REPO_HOTKEY_SUFFIX_ENFORCE_BLOCK:
            repo_name = model.split("/")[-1] if "/" in model else model
            if not repo_name.lower().endswith(hotkey.lower()):
                info.mark_invalid(
                    f"repo_name_not_ending_with_hotkey:repo={repo_name}",
                    permanent=True,
                    terminate_stats=True,
                )
                return info

        # HuggingFace lookup → model_hash + sha + commit-history flags.
        try:
            model_info = await self._get_model_info(model, revision)
        except HFRepoUnavailable as e:
            info.mark_invalid(
                e.reason,
                permanent=True,
                terminate_stats=True,
            )
            return info
        if not model_info:
            info.mark_invalid("hf_model_fetch_failed", permanent=False)
            return info
        model_hash, hf_revision, duplicate_source = model_info
        info.model_hash = model_hash
        info.hf_revision = hf_revision

        if revision != hf_revision:
            info.mark_invalid(
                f"revision_mismatch:hf={hf_revision}",
                permanent=True,
                terminate_stats=True,
            )
            return info

        if uid != 0 and uid <= 1000:
            allowed_model_types = (
                QWEN36_ONLY_MODEL_TYPES
                if block >= QWEN36_ONLY_ENFORCE_BLOCK
                else None
            )
            size_result = await check_model_size(
                model,
                revision,
                allowed_model_types=allowed_model_types,
            )
            info.model_type = str(size_result.get("model_type") or "")
            if not size_result.get("pass"):
                info.mark_invalid(
                    f"model_check:{size_result.get('reason')}",
                    permanent=True,
                    terminate_stats=True,
                )
                return info
            if (
                info.model_type == QWEN36_MODEL_TYPE
                and block < QWEN36_ALLOWED_FROM_BLOCK
            ):
                info.mark_invalid(
                    "model_check:"
                    f"qwen36_not_allowed_until_block:{QWEN36_ALLOWED_FROM_BLOCK}"
                    f":commit_block={block}",
                    permanent=True,
                    terminate_stats=True,
                )
                return info

        if uid != 0 and duplicate_source:
            info.mark_invalid(
                f"duplicate_repo:from={duplicate_source}",
                permanent=True,
                terminate_stats=True,
            )
            return info

        # Template safety. Cached "safe" skips the check; cached "unsafe" is
        # honored (no second chances for malicious templates).
        if uid != 0:
            cached = info.template_check_result
            if cached == "safe":
                pass
            elif cached and cached.startswith("unsafe:"):
                info.mark_invalid(
                    f"malicious_template:{cached[7:]}",
                    permanent=True,
                    terminate_stats=True,
                )
                return info
            else:
                try:
                    tr = await check_template_safety(model, revision)
                    if not tr.get("safe"):
                        reason = tr.get("reason", "unknown")
                        transient = reason.startswith("template_fetch_failed:") or reason.startswith("check_error:")
                        info.mark_invalid(
                            f"malicious_template:{reason}",
                            permanent=not transient,
                            terminate_stats=not transient,
                        )
                        if not transient:
                            info.template_check_result = f"unsafe:{reason}"
                        return info
                    if not tr.get("reason", "").startswith("llm_audit_skipped:"):
                        info.template_check_result = "safe"
                except Exception as e:
                    logger.debug(f"[MinersMonitor] template check failed uid={uid}: {e}")

        # step 8.0: CEAC tokenizer-signature early reject. A candidate
        # whose tokenizer.json differs from the active champion's
        # cannot be teacher-forced apple-to-apple, so we never bother
        # putting it on the queue. Inactive (no champion sig yet) →
        # skip the check, anticopy is in cold-start.
        anticopy_cfg = await self._safe_load_anticopy_config()
        champion_sig = ""
        if anticopy_cfg is not None and anticopy_cfg.enabled:
            try:
                champion_sig = await self.anticopy_state_dao.get_champion_tokenizer_sig() or ""
                # legacy: fall back to system_config for backward-compat
                # in case ``anticopy_state`` hasn't been populated yet on
                # an older deployment. Drop this branch once all live
                # validators have rolled past the migration.
                if not champion_sig:
                    champion_sig = await self.config_dao.get_param_value(
                        "anticopy_champion_tokenizer_sig", default="",
                    ) or ""
            except Exception:
                champion_sig = ""

        if uid != 0 and anticopy_cfg is not None and anticopy_cfg.enabled and champion_sig:
            cand_sig = await self._get_tokenizer_sig(model, revision)
            info.tokenizer_sig = cand_sig
            if not cand_sig:
                info.mark_invalid("tokenizer_sig_fetch_failed", permanent=False)
                return info
            if cand_sig != champion_sig:
                info.mark_invalid(
                    f"tokenizer_sig_mismatch:cand={cand_sig[:12]}",
                    permanent=True,
                )
                return info

        # CEAC step 8.1: if anticopy backfill flagged this miner as a
        # copy of an earlier model, mark it invalid here so downstream
        # weight-setting drops it. Lookup is keyed by (hotkey, revision)
        # so a re-uploaded ckpt is re-evaluated fresh — we don't carry
        # over a stale verdict.
        # permanent=False so that if backfill re-runs (threshold change,
        # origin deregistration, etc.) and the verdict clears, the next
        # monitor cycle picks the miner back up.
        if (
            uid != 0
            and anticopy_cfg is not None
            and anticopy_cfg.enabled
        ):
            try:
                score_row = await self.anticopy_scores_dao.get_score(
                    hotkey, revision,
                )
            except Exception as e:
                logger.debug(
                    f"[MinersMonitor] anticopy score lookup failed "
                    f"{hotkey[:10]}: {e}"
                )
                score_row = None
            if score_row and (score_row.get("verdict_copy_of") or "").strip():
                origin_model = (
                    score_row.get("closest_peer_model") or "unknown"
                )
                dm = score_row.get("decision_median")
                try:
                    dm_str = f"{float(dm):.4f}"
                except (TypeError, ValueError):
                    dm_str = "?"
                info.mark_invalid(
                    f"anticopy_copy:copied_from={origin_model},dm={dm_str}",
                    permanent=False,
                )
                return info

        info.is_valid = True
        if not info.template_check_result:
            info.template_check_result = "safe"

        # CEAC step 8 (job enqueue) was removed when the worker switched
        # to a pull-based model: it now reads the live miners table
        # itself, sorts by commit time, and skips already-scored rows.
        # The monitor only needs to stamp the tokenizer signature here
        # (step 8.0 above) and persist ``miners.is_valid``; no queue
        # write is required.
        return info

    async def _safe_load_anticopy_config(self):
        """Resolve the anticopy config but never break a monitor cycle
        on a transient KV read failure."""
        try:
            return await load_anticopy_config(self.config_dao)
        except Exception as e:
            logger.debug(f"[MinersMonitor] load_anticopy_config failed: {e}")
            return None

    async def _get_tokenizer_sig(self, model_id: str, revision: str) -> str:
        """Cached sha256 of the candidate's ``tokenizer.json``. Empty
        string means HF couldn't supply it (treat as transient)."""
        key = (model_id, revision)
        now = time.time()
        cached = self._tokenizer_sig_cache.get(key)
        if cached and now - cached[1] < self._weights_ttl_sec:
            return cached[0]
        try:
            sig, _src = await compute_tokenizer_signature(model_id, revision)
        except Exception as e:
            logger.debug(
                f"[MinersMonitor] tokenizer_sig fetch failed {model_id}@{revision[:8]}: {e}"
            )
            sig = None
        sig_str = sig or ""
        self._tokenizer_sig_cache[key] = (sig_str, now)
        return sig_str

    async def _get_model_info(
        self, model_id: str, revision: str
    ) -> Optional[Tuple[str, str, str]]:
        """Return ``(model_hash, hf_revision, duplicate_source)`` or None.

        - ``model_hash``: sha256 over the sorted sha256s of every weight
          shard (.safetensors/.bin) listed by HF.
        - ``hf_revision``: the actual git SHA HF resolves the revision to.
        - ``duplicate_source``: empty string, ``blocked:too_many_commits``,
          ``blocked:commit_msg_too_long``, or the repo a "Duplicate from
          xxx" commit points at.
        """
        key = (model_id, revision)
        now = time.time()
        api = HfApi(token=os.getenv("HF_TOKEN"))
        cached = self._weights_cache.get(key)
        if cached and now - cached[1] < self._weights_ttl_sec:
            await self._verify_repo_accessible(api, model_id, revision)
            return cached[0]

        try:
            info = await asyncio.to_thread(
                lambda: api.repo_info(
                    repo_id=model_id, repo_type="model", revision=revision, files_metadata=True,
                )
            )
            if getattr(info, "gated", False):
                await self._assert_gated_downloadable(model_id, revision)
            hf_revision = getattr(info, "sha", None)
            siblings = getattr(info, "siblings", None) or []
            shas = {
                str(getattr(s, "lfs", {})["sha256"])
                for s in siblings
                if (
                    isinstance(getattr(s, "lfs", None), dict)
                    and (getattr(s, "rfilename", "") or getattr(s, "path", "")).endswith(
                        (".safetensors", ".bin")
                    )
                    and "sha256" in getattr(s, "lfs", {})
                )
            }
            if not shas or not hf_revision:
                self._weights_cache[key] = (None, now)
                return None
            import hashlib
            model_hash = hashlib.sha256("".join(sorted(shas)).encode()).hexdigest()

            duplicate_source = ""
            try:
                commits = list(
                    await asyncio.to_thread(
                        lambda: api.list_repo_commits(
                            repo_id=model_id, repo_type="model", revision=revision,
                        )
                    )
                )
                if len(commits) > 100:
                    duplicate_source = "blocked:too_many_commits"
                else:
                    for c in commits:
                        title = getattr(c, "title", "") or ""
                        if len(title) > 200:
                            duplicate_source = "blocked:commit_msg_too_long"
                            break
                        if title.lower().startswith("duplicate from"):
                            duplicate_source = title[len("Duplicate from"):].strip()
                            break
            except Exception as e:
                logger.debug(f"[MinersMonitor] list_repo_commits failed {model_id}@{revision[:8]}: {e}")

            result = (model_hash, hf_revision, duplicate_source)
            self._weights_cache[key] = (result, now)
            return result
        except HFRepoUnavailable:
            # Raised by the gated-access probe below; propagate to the caller's
            # permanent-reject handler instead of being swallowed as a soft
            # fetch failure by the generic ``except Exception``.
            self._weights_cache.pop(key, None)
            raise
        except DisabledRepoError as e:
            self._weights_cache.pop(key, None)
            self._raise_repo_unavailable(model_id, revision, "hf_repo_disabled", e)
        except GatedRepoError as e:
            self._weights_cache.pop(key, None)
            self._raise_repo_unavailable(model_id, revision, "hf_repo_private", e)
        except RepositoryNotFoundError as e:
            self._weights_cache.pop(key, None)
            self._raise_repo_unavailable(model_id, revision, "hf_repo_not_found", e)
        except Exception as e:
            logger.warning(
                f"[MinersMonitor] HF fetch failed {model_id}@{revision[:8]}: "
                f"{type(e).__name__}: {e}"
            )
            return None

    async def _assert_gated_downloadable(self, model_id: str, revision: str) -> None:
        """Miner models must be fully public. A gated repo still returns full
        metadata via ``repo_info`` even when the validator isn't on its
        allow-list, so ``_get_model_info`` (which hashes weights from metadata)
        and a cached ``config.json`` both pass — yet the scheduler can't
        download the weights to serve it (403 at deploy). Probe one real file
        resolve so that 403 surfaces here.

        Only a deterministic ``GatedRepoError`` (HTTP 403 + gated marker) is a
        permanent, miner-attributable reject. Everything else — rate limiting
        (429), 5xx, timeouts, DNS — is swallowed, so transient HF pressure can
        never wrongly terminate a good miner. Runs only on the 30-min
        weight-cache miss (gated repos only), so it adds no high-frequency load.
        """
        try:
            await asyncio.to_thread(
                lambda: get_hf_file_metadata(
                    hf_hub_url(model_id, "config.json", revision=revision),
                    token=os.getenv("HF_TOKEN"),
                )
            )
        except GatedRepoError as e:
            self._raise_repo_unavailable(model_id, revision, "hf_repo_gated", e)
        except Exception as e:
            logger.warning(
                f"[MinersMonitor] gated-access probe inconclusive "
                f"{model_id}@{revision[:8]}: {type(e).__name__}: {e}"
            )

    async def _verify_repo_accessible(
        self, api: HfApi, model_id: str, revision: str
    ) -> None:
        """Confirm the repo is still publicly reachable even on weight-cache hits.

        Deterministic visibility failures terminate the miner immediately.
        Generic HF/network failures are treated as transient and keep the
        cached weight metadata usable for this cycle.
        """
        try:
            await asyncio.to_thread(
                lambda: api.repo_info(
                    repo_id=model_id,
                    repo_type="model",
                    revision=revision,
                    files_metadata=False,
                )
            )
        except DisabledRepoError as e:
            self._weights_cache.pop((model_id, revision), None)
            self._raise_repo_unavailable(model_id, revision, "hf_repo_disabled", e)
        except GatedRepoError as e:
            self._weights_cache.pop((model_id, revision), None)
            self._raise_repo_unavailable(model_id, revision, "hf_repo_private", e)
        except RepositoryNotFoundError as e:
            self._weights_cache.pop((model_id, revision), None)
            self._raise_repo_unavailable(model_id, revision, "hf_repo_not_found", e)
        except Exception as e:
            logger.warning(
                f"[MinersMonitor] HF access check failed {model_id}@{revision[:8]}: "
                f"{type(e).__name__}: {e}"
            )

    def _raise_repo_unavailable(
        self, model_id: str, revision: str, reason: str, error: Exception
    ) -> None:
        logger.warning(
            f"[MinersMonitor] HF repo unavailable {model_id}@{revision[:8]}: "
            f"{type(error).__name__} reason={reason}"
        )
        raise HFRepoUnavailable(reason) from error

    def _detect_plagiarism(self, miners: list[MinerInfo]) -> list[MinerInfo]:
        """Flag later committers as invalid when their model_hash collides
        with an earlier committer's. Earliest wins."""
        by_hash: Dict[str, list[MinerInfo]] = {}
        for m in miners:
            if not m.is_valid or not m.model_hash or m.uid == 0 or m.uid > 1000:
                continue
            by_hash.setdefault(m.model_hash, []).append(m)
        for h, group in by_hash.items():
            if len(group) < 2:
                continue
            group.sort(key=lambda x: x.block)
            origin = group[0]
            for later in group[1:]:
                later.mark_invalid(
                    f"plagiarism:duplicate_of_uid={origin.uid}", permanent=True,
                )
                logger.info(
                    f"[MinersMonitor] plagiarism uid={later.uid} duplicates "
                    f"uid={origin.uid} (hash={h[:12]}...)"
                )
        return miners

    # ---- persistence -----------------------------------------------------

    async def _load_blacklist(self) -> set:
        env_bl = {
            hk.strip()
            for hk in os.getenv("AFFINE_MINER_BLACKLIST", "").split(",")
            if hk.strip()
        }
        db_bl = set(await self.config_dao.get_blacklist())
        return env_bl | db_bl

    async def _persist_miners(self, miners: list[MinerInfo], *, current_block: int) -> None:
        """Save the current online miner snapshot.

        Lifecycle (``challenge_status``) is owned by ``miner_stats`` and is
        written by the scheduler, with one exception: deterministic,
        miner-attributable rejects set ``terminate_stats=True`` at the verdict
        site. Those rows are conditionally terminated here so the rank queue
        stops showing them as "still sampling" and the rejection reason is
        visible in miner_stats. Transient or externally reversible rejects
        leave that flag false.
        """
        from affine.database.client import get_client

        client = get_client()

        for miner in miners:
            await self.dao.save_miner(
                uid=miner.uid,
                hotkey=miner.hotkey,
                model=miner.model,
                revision=miner.revision,
                model_hash=miner.model_hash,
                is_valid=miner.is_valid,
                invalid_reason=miner.invalid_reason,
                block_number=current_block,
                first_block=miner.block,
                model_type=miner.model_type,
            )
            if miner.hotkey and miner.revision:
                await self.stats_dao.update_miner_info(
                    hotkey=miner.hotkey,
                    revision=miner.revision,
                    model=miner.model,
                    uid=miner.uid,
                    first_block=miner.block,
                    block_number=current_block,
                    is_valid=miner.is_valid,
                    invalid_reason=miner.invalid_reason,
                    model_hash=miner.model_hash,
                    model_type=miner.model_type,
                    is_online=True,
                )
                await self._maybe_terminate_stats(miner)
            # template_check_result is a non-key attr we still want to keep
            # cached on the row — write it separately so the DAO signature
            # stays free of optional bric-à-brac.
            if miner.template_check_result is not None:
                try:
                    await client.update_item(
                        TableName=self.dao.table_name,
                        Key={"pk": {"S": self.dao._make_pk(miner.uid)}},
                        UpdateExpression="SET template_check_result = :v",
                        ExpressionAttributeValues={
                            ":v": {"S": miner.template_check_result},
                        },
                    )
                except Exception as e:
                    logger.debug(
                        f"[MinersMonitor] template_check_result write failed "
                        f"uid={miner.uid}: {e}"
                    )

    async def _maybe_terminate_stats(self, miner: MinerInfo) -> None:
        if not (miner.permanent_invalid and miner.terminate_stats):
            return
        reason = miner.invalid_reason or ""
        try:
            wrote = await self.stats_dao.terminate_if_sampling(
                hotkey=miner.hotkey,
                revision=miner.revision,
                reason=reason,
            )
        except Exception as e:
            logger.warning(
                f"[MinersMonitor] terminate_if_sampling failed "
                f"uid={miner.uid} reason={reason}: {e}"
            )
            return
        if wrote:
            logger.info(
                f"[MinersMonitor] terminated uid={miner.uid} reason={reason}"
            )

    # ---- public read ----------------------------------------------------

    async def get_valid_miners(self, force_refresh: bool = False) -> Dict[str, MinerInfo]:
        if force_refresh:
            return await self.refresh_miners()
        rows = await self.dao.get_valid_miners()
        out: Dict[str, MinerInfo] = {}
        for r in rows:
            info = MinerInfo(
                uid=int(r["uid"]),
                hotkey=r["hotkey"],
                model=r.get("model", ""),
                revision=r.get("revision", ""),
                block=int(r.get("first_block", 0) or 0),
                is_valid=True,
                model_hash=r.get("model_hash", ""),
                model_type=r.get("model_type", ""),
                template_check_result=r.get("template_check_result"),
            )
            out[info.key()] = info
        return out
