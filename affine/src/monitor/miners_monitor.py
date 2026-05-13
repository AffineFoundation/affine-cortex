"""
Miners monitor.

Reads metagraph commits and miner metadata. For every miner on the subnet:

  1. Read its on-chain commit ``{model, revision}``.
  2. Validate model on HuggingFace: revision exists, weight hashes
     computable, no duplicate-repo / suspicious commit history.
  3. Optional: model size + chat-template safety checks.
  4. Plagiarism detection by ``model_hash`` collision (earliest committer
     wins; later miners with the same hash are marked invalid).
  5. Persist to ``miners`` table.
  6. Seed ``miners.challenge_status='pending'`` and ``enqueued_at`` when a
     freshly-seen ``(hotkey, revision)`` lands — that's the only signal
     the challenger queue needs to enter the rotation.

Anti-copy checks run here; inference lifecycle is handled by the scheduler.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from botocore.exceptions import ClientError
from huggingface_hub import HfApi

from affine.core.setup import logger
from affine.database.dao.miners import MinersDAO
from affine.database.dao.system_config import SystemConfigDAO
from affine.utils.model_size_checker import check_model_size
from affine.utils.subtensor import get_subtensor
from affine.utils.template_checker import check_template_safety


NETUID = int(os.getenv("AFFINE_NETUID", "120"))

MULTI_COMMIT_ENFORCE_BLOCK = 7_710_000
REPO_HOTKEY_SUFFIX_ENFORCE_BLOCK = 7_290_000


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
    model_hash: str = ""
    hf_revision: str = ""
    template_check_result: Optional[str] = None  # "safe" | "unsafe:<reason>" | None

    def key(self) -> str:
        return f"{self.hotkey}#{self.revision}"

    def mark_invalid(self, reason: str, *, permanent: bool) -> None:
        self.is_valid = False
        self.invalid_reason = reason
        self.permanent_invalid = permanent


class MinersMonitor:
    """Refreshes the ``miners`` table from chain + HF, seeds challenge state."""

    _instance: Optional["MinersMonitor"] = None
    _lock = asyncio.Lock()

    def __init__(self, refresh_interval_seconds: int = 300):
        self.dao = MinersDAO()
        self.config_dao = SystemConfigDAO()
        self.refresh_interval_seconds = refresh_interval_seconds
        self.last_update: int = 0
        # (model, revision) -> (cached result or None, fetched_at)
        self._weights_cache: Dict[Tuple[str, str], Tuple[Optional[Tuple[str, str, str]], float]] = {}
        self._weights_ttl_sec = 1800
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
            if uid <= 1000:
                continue
            miners.append(
                MinerInfo(
                    uid=uid,
                    hotkey=f"SYSTEM-{uid - 1000}",
                    model=payload.get("model", ""),
                    revision=f"SYSTEM-{uid - 1000}",
                    block=0,
                    is_valid=True,
                    invalid_reason=None,
                    model_hash="",
                    hf_revision=f"SYSTEM-{uid - 1000}",
                    template_check_result="safe",
                )
            )

        # Persist + seed challenge_status for newly-seen (hotkey, revision).
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
            info.mark_invalid(f"multiple_commits:count={commit_count}", permanent=True)
            return info

        # "affine" must appear in the model name (system miners exempt).
        if uid != 0 and "affine" not in model.lower():
            info.mark_invalid("model_name_missing_affine", permanent=True)
            return info

        # Repo name must end with the hotkey from a certain block onward —
        # makes plagiarism via repo rename impossible.
        if uid != 0 and block >= REPO_HOTKEY_SUFFIX_ENFORCE_BLOCK:
            repo_name = model.split("/")[-1] if "/" in model else model
            if not repo_name.lower().endswith(hotkey.lower()):
                info.mark_invalid(
                    f"repo_name_not_ending_with_hotkey:repo={repo_name}", permanent=True,
                )
                return info

        # HuggingFace lookup → model_hash + sha + commit-history flags.
        model_info = await self._get_model_info(model, revision)
        if not model_info:
            info.mark_invalid("hf_model_fetch_failed", permanent=False)
            return info
        model_hash, hf_revision, duplicate_source = model_info
        info.model_hash = model_hash
        info.hf_revision = hf_revision

        if revision != hf_revision:
            info.mark_invalid(f"revision_mismatch:hf={hf_revision}", permanent=True)
            return info

        if uid != 0 and uid <= 1000:
            size_result = await check_model_size(model, revision)
            if not size_result.get("pass"):
                info.mark_invalid(f"model_check:{size_result.get('reason')}", permanent=True)
                return info

        if uid != 0 and duplicate_source:
            info.mark_invalid(f"duplicate_repo:from={duplicate_source}", permanent=True)
            return info

        # Template safety. Cached "safe" skips the check; cached "unsafe" is
        # honored (no second chances for malicious templates).
        if uid != 0:
            cached = info.template_check_result
            if cached == "safe":
                pass
            elif cached and cached.startswith("unsafe:"):
                info.mark_invalid(f"malicious_template:{cached[7:]}", permanent=True)
                return info
            else:
                try:
                    tr = await check_template_safety(model, revision)
                    if not tr.get("safe"):
                        reason = tr.get("reason", "unknown")
                        transient = reason.startswith("template_fetch_failed:") or reason.startswith("check_error:")
                        info.mark_invalid(f"malicious_template:{reason}", permanent=not transient)
                        if not transient:
                            info.template_check_result = f"unsafe:{reason}"
                        return info
                    if not tr.get("reason", "").startswith("llm_audit_skipped:"):
                        info.template_check_result = "safe"
                except Exception as e:
                    logger.debug(f"[MinersMonitor] template check failed uid={uid}: {e}")

        info.is_valid = True
        if not info.template_check_result:
            info.template_check_result = "safe"
        return info

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
        cached = self._weights_cache.get(key)
        if cached and now - cached[1] < self._weights_ttl_sec:
            return cached[0]

        try:
            api = HfApi(token=os.getenv("HF_TOKEN"))
            info = await asyncio.to_thread(
                lambda: api.repo_info(
                    repo_id=model_id, repo_type="model", revision=revision, files_metadata=True,
                )
            )
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
        except Exception as e:
            logger.warning(
                f"[MinersMonitor] HF fetch failed {model_id}@{revision[:8]}: "
                f"{type(e).__name__}: {e}"
            )
            self._weights_cache[key] = (None, now)
            return None

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
        """Save each miner row, and seed ``challenge_status='pending'``
        when a freshly-seen ``(hotkey, revision)`` is observed (or when the
        revision changes — each ``(hotkey, revision)`` gets exactly one
        chance to challenge the champion, regardless of prior history)."""
        from affine.database.client import get_client

        client = get_client()
        now = int(time.time())

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
            )
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

            # Seed challenge_status for new miners only. A miner can only
            # commit once on this subnet (the multi-commit rule in
            # _validate_miner rejects any second commit), so once the row
            # carries a challenge_status — pending, in_progress, champion,
            # or terminated_* — it stays. attribute_not_exists is enough.
            if not miner.is_valid or miner.uid == 0 or miner.uid > 1000:
                continue
            try:
                await client.update_item(
                    TableName=self.dao.table_name,
                    Key={"pk": {"S": self.dao._make_pk(miner.uid)}},
                    UpdateExpression=(
                        "SET challenge_status = :pending, enqueued_at = :now"
                    ),
                    ConditionExpression="attribute_not_exists(challenge_status)",
                    ExpressionAttributeValues={
                        ":pending": {"S": "pending"},
                        ":now": {"N": str(now)},
                    },
                )
            except ClientError as e:
                if e.response.get("Error", {}).get("Code") != "ConditionalCheckFailedException":
                    logger.debug(
                        f"[MinersMonitor] challenge_status seed failed "
                        f"uid={miner.uid}: {e}"
                    )
            except Exception as e:
                logger.debug(
                    f"[MinersMonitor] challenge_status seed unexpected error "
                    f"uid={miner.uid}: {type(e).__name__}: {e}"
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
                template_check_result=r.get("template_check_result"),
            )
            out[info.key()] = info
        return out
