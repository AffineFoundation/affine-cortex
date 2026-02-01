"""
Miners Monitor Service

Monitors and validates miners with anti-plagiarism detection.
Persists validation state to Miners table.
"""

import os
import json
import time
import asyncio
import aiohttp
import logging
from typing import Dict, Optional, Set
from dataclasses import dataclass
from huggingface_hub import HfApi

from affine.utils.subtensor import get_subtensor
from affine.utils.api_client import get_chute_info
from affine.utils.template_checker import check_template_safety
from affine.core.setup import NETUID
from affine.database.dao.miners import MinersDAO
from affine.database.dao.system_config import SystemConfigDAO
from affine.core.setup import logger


@dataclass
class MinerInfo:
    """Miner information data class"""
    uid: int
    hotkey: str
    model: str
    revision: str
    chute_id: str
    chute_slug: str = ""
    block: int = 0
    is_valid: bool = False
    invalid_reason: Optional[str] = None
    model_hash: str = ""  # HuggingFace model hash (cached)
    hf_revision: str = ""  # HuggingFace actual revision (cached)
    chute_status: str = ""
    template_check_result: Optional[str] = None  # "safe", "unsafe:reason", or None (unchecked)

    def key(self) -> str:
        """Generate unique key: hotkey#revision"""
        return f"{self.hotkey}#{self.revision}"


class MinersMonitor:
    """Miners monitor and validation service
    
    Responsibilities:
    1. Discover miners from metagraph
    2. Validate chute status, revision, and model weights
    3. Detect plagiarism via model hash comparison
    4. Persist validation results to database
    """
    
    _instance: Optional['MinersMonitor'] = None
    _lock = asyncio.Lock()
    
    def __init__(self, refresh_interval_seconds: int = 300):
        """Initialize monitor
        
        Args:
            refresh_interval_seconds: Auto-refresh interval in seconds
        """
        self.dao = MinersDAO()
        self.config_dao = SystemConfigDAO()
        self.refresh_interval_seconds = refresh_interval_seconds
        self.last_update: int = 0
        
        # Caches: (type, model, revision) -> (result, timestamp)
        # type: "model_info" | "duplicate"
        self.weights_cache: Dict[tuple, tuple] = {}
        self.weights_ttl = 1800  # 30 minutes
        
        # Background task management
        self._running = False
        self._refresh_task: Optional[asyncio.Task] = None
        
        logger.info("[MinersMonitor] Initialized")
    
    @classmethod
    def get_instance(cls) -> 'MinersMonitor':
        """Get global singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    async def initialize(cls, refresh_interval_seconds: int = 300) -> 'MinersMonitor':
        """Initialize global singleton and start background tasks"""
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls(refresh_interval_seconds=refresh_interval_seconds)
                await cls._instance.refresh_miners()
                await cls._instance.start_background_tasks()
        return cls._instance
    
    async def _refresh_loop(self):
        """Background refresh loop"""
        while self._running:
            try:
                await self.refresh_miners()
                await asyncio.sleep(self.refresh_interval_seconds)
            except Exception as e:
                logger.error(f"[MinersMonitor] Error in refresh loop: {e}", exc_info=True)
    
    async def start_background_tasks(self):
        """Start background refresh tasks"""
        if self._running:
            logger.warning("[MinersMonitor] Background tasks already running")
            return
        
        self._running = True
        self._refresh_task = asyncio.create_task(self._refresh_loop())
        logger.info(f"[MinersMonitor] Background refresh started (interval={self.refresh_interval_seconds}s)")
    
    async def stop_background_tasks(self):
        """Stop background refresh tasks"""
        self._running = False
        if self._refresh_task:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
        logger.info("[MinersMonitor] Background tasks stopped")
    
    async def _load_blacklist(self) -> set:
        """Load blacklisted hotkeys from database and environment, then merge them.
        
        Returns:
            Set of unique blacklisted hotkeys from both sources
        """
        # Load from environment variable
        env_blacklist_str = os.getenv("AFFINE_MINER_BLACKLIST", "").strip()
        env_blacklist = set()
        if env_blacklist_str:
            env_blacklist = {hk.strip() for hk in env_blacklist_str.split(",") if hk.strip()}
        
        # Load from database
        db_blacklist = set(await self.config_dao.get_blacklist())
        
        # Merge and deduplicate
        merged_blacklist = env_blacklist | db_blacklist
        
        if merged_blacklist:
            logger.debug(
                f"[MinersMonitor] Loaded blacklist: "
                f"{len(env_blacklist)} from env, {len(db_blacklist)} from db, "
                f"{len(merged_blacklist)} total after merge"
            )
        
        return merged_blacklist
    
    async def _get_model_info(self, model_id: str, revision: str) -> Optional[tuple[str, str, str]]:
        """Get model hash, actual revision and commit title from HuggingFace

        Args:
            model_id: HuggingFace model repo
            revision: Git commit hash

        Returns:
            Tuple of (model_hash, actual_revision, commit_title) or None if failed
        """
        key = (model_id, revision)
        now = time.time()
        cached = self.weights_cache.get(key)

        if cached and now - cached[1] < self.weights_ttl:
            return cached[0]

        try:
            hf_api = HfApi(token=os.getenv("HF_TOKEN"))

            def _repo_info():
                return hf_api.repo_info(
                    repo_id=model_id,
                    repo_type="model",
                    revision=revision,
                    files_metadata=True,
                )

            def _list_commits():
                return hf_api.list_repo_commits(
                    repo_id=model_id,
                    repo_type="model",
                    revision=revision,
                )

            info = await asyncio.to_thread(_repo_info)

            # Get actual revision (git SHA)
            actual_revision = getattr(info, "sha", None)

            # Get model weight hashes
            siblings = getattr(info, "siblings", None) or []

            def _name(s):
                return getattr(s, "rfilename", None) or getattr(s, "path", "") or ""

            shas = {
                str(getattr(s, "lfs", {})["sha256"])
                for s in siblings
                if (
                    isinstance(getattr(s, "lfs", None), dict)
                    and _name(s) is not None
                    and (_name(s).endswith(".safetensors") or _name(s).endswith(".bin"))
                    and "sha256" in getattr(s, "lfs", {})
                )
            }

            # Compute total hash
            model_hash = None
            if shas:
                import hashlib
                model_hash = hashlib.sha256("".join(sorted(shas)).encode()).hexdigest()

            # Get commit title for duplicate check
            commit_title = ""
            try:
                commits = await asyncio.to_thread(_list_commits)
                for commit in commits:
                    if getattr(commit, "commit_id", None) == revision:
                        commit_title = getattr(commit, "title", "") or ""
                        break
            except Exception as e:
                logger.debug(f"Failed to get commit title for {model_id}@{revision}: {e}")

            result = (model_hash, actual_revision, commit_title) if model_hash and actual_revision else None
            self.weights_cache[key] = (result, now)
            return result

        except Exception as e:
            logger.warning(
                f"Failed to fetch model info for {model_id}@{revision}: {type(e).__name__}: {e}",
                exc_info=True
            )
            self.weights_cache[key] = (None, now)
            return None

    async def _is_duplicate_commit(self, model_id: str, revision: str) -> Optional[str]:
        """Check if the commit is a 'Duplicate from xxx' commit

        Uses cached commit_title from _get_model_info().

        Args:
            model_id: HuggingFace model repo
            revision: Git commit hash to check

        Returns:
            Source repo name if it's a duplicate commit, None otherwise
        """
        key = (model_id, revision)
        cached = self.weights_cache.get(key)

        if not cached or not cached[0]:
            return None

        commit_title = cached[0][2] or ""
        if commit_title.lower().startswith("duplicate from"):
            return commit_title[len("Duplicate from"):].strip()

        return None
    
    async def _validate_miner(
        self,
        uid: int,
        hotkey: str,
        model: str,
        revision: str,
        chute_id: str,
        block: int,
    ) -> MinerInfo:
        """Validate a single miner

        Validation steps:
        1. Fetch chute info
        2. Validate chute_slug is not empty
        3. Check chute is hot
        4. Verify model name matches chute
        5. Verify model name contains "Affine" or "affine" (except uid 0)
        6. Verify repo name ends with hotkey
        7. Verify revision matches chute
        8. Fetch HuggingFace model info and verify revision
        9. Check if commit is "Duplicate from xxx" (plagiarism check)
        10. Check chat_template for malicious code

        Args:
            uid: Miner UID
            hotkey: Miner hotkey
            model: Model repo from commit
            revision: Git commit hash from commit
            chute_id: Chute deployment ID
            block: Block when miner committed

        Returns:
            MinerInfo with validation result and cached model_hash/hf_revision
        """
        info = MinerInfo(
            uid=uid,
            hotkey=hotkey,
            model=model,
            revision=revision,
            chute_id=chute_id,
            block=block,
        )

        # Inherit template_check_result from database if model/revision unchanged
        # This prevents losing the result when earlier steps fail
        try:
            existing = await self.dao.get_miner_by_uid(uid)
            if (existing and
                existing.get('model') == model and
                existing.get('revision') == revision):
                info.template_check_result = existing.get('template_check_result')
        except Exception:
            pass  # Ignore errors, will check template later if needed

        # Step 1: Fetch chute info
        chute = await get_chute_info(chute_id)
        if not chute:
            info.is_valid = False
            info.invalid_reason = "chute_fetch_failed"
            return info
        
        info.chute_slug = chute.get("slug", "")
        info.chute_status = "hot" if chute.get("hot", False) else "cold"
        
        # Step 2: Validate chute_slug is not empty
        if not info.chute_slug:
            info.is_valid = False
            info.invalid_reason = "chute_slug_empty"
            return info

        # Step 3: Check chute is hot
        if not chute.get("hot", False):
            info.is_valid = False
            info.invalid_reason = "chute_not_hot"
            return info
        
        # Step 4: Verify model name matches chute
        chute_model = chute.get("name", "")
        if model != chute_model:
            # Skip validation for uid 0
            if uid != 0:
                info.is_valid = False
                info.invalid_reason = f"model_mismatch:chute={chute_model}"
                return info

        # Step 5: Verify model name contains "Affine" or "affine" (except uid 0)
        if uid != 0:
            if "affine" not in model.lower():
                info.is_valid = False
                info.invalid_reason = "model_name_missing_affine"
                return info

        # Step 6: Verify repo name ends with hotkey
        if uid != 0 and block >= 7290000:
            # Extract repo name from model (format: owner/repo_name)
            repo_name = model.split('/')[-1] if '/' in model else model
            
            # Check if repo name ends with hotkey (case-insensitive)
            if not repo_name.lower().endswith(hotkey.lower()):
                info.is_valid = False
                info.invalid_reason = f"repo_name_not_ending_with_hotkey:repo={repo_name}"
                return info

        # Step 7: Verify revision matches chute
        chute_revision = chute.get("revision", "")
        if chute_revision and revision != chute_revision:
            info.is_valid = False
            info.invalid_reason = f"revision_mismatch:chute={chute_revision}"
            return info
        
        # Step 8: Fetch HuggingFace model info and verify revision
        model_info = await self._get_model_info(model, revision)
        if not model_info:
            info.is_valid = False
            info.invalid_reason = "hf_model_fetch_failed"
            return info
        
        model_hash, hf_revision, _ = model_info
        
        # Cache model info in MinerInfo
        info.model_hash = model_hash
        info.hf_revision = hf_revision
        
        # Verify revision matches
        if revision != hf_revision:
            info.is_valid = False
            info.invalid_reason = f"revision_mismatch:hf={hf_revision}"
            return info

        # Step 9: Check if commit is a "Duplicate from xxx" (except uid 0)
        if uid != 0:
            duplicate_source = await self._is_duplicate_commit(model, revision)
            if duplicate_source:
                info.is_valid = False
                info.invalid_reason = f"duplicate_repo:from={duplicate_source}"
                logger.info(
                    f"[MinersMonitor] Duplicate repo detected for uid={uid}: "
                    f"model={model} is duplicated from {duplicate_source}"
                )
                return info

        # Step 10: Check chat_template for malicious code (with database cache)
        # Skip for uid 0 (test/admin miner)
        if uid == 0:
            info.template_check_result = "safe"
            info.is_valid = True
            return info

        try:
            # Use inherited template_check_result as cache (already loaded at start)
            cached_result = info.template_check_result

            if cached_result == "safe":
                # Previously passed, skip check
                logger.debug(f"[MinersMonitor] Skipping template check for uid={uid} (cached: safe)")
            elif cached_result and cached_result.startswith("unsafe:"):
                # Previously failed, use cached result directly
                info.is_valid = False
                info.invalid_reason = f"malicious_template:{cached_result[7:]}"
                logger.debug(f"[MinersMonitor] Using cached template result for uid={uid}: {cached_result}")
                return info
            else:
                # No cache or model/revision changed, execute check
                template_result = await check_template_safety(model, revision)
                if not template_result["safe"]:
                    reason = template_result['reason']
                    info.is_valid = False
                    info.invalid_reason = f"malicious_template:{reason}"
                    info.template_check_result = f"unsafe:{reason}"
                    logger.warning(
                        f"[MinersMonitor] Malicious template detected for uid={uid}: {reason}"
                    )
                    return info

                # Check if audit was skipped (no API key, error, etc.)
                if template_result['reason'].startswith("llm_audit_skipped:"):
                    # Leave template_check_result as None, will retry next time
                    logger.debug(
                        f"[MinersMonitor] Template check skipped for uid={uid}: "
                        f"{template_result['reason']}"
                    )
                else:
                    info.template_check_result = "safe"
        except Exception as e:
            logger.warning(f"[MinersMonitor] Template check failed for uid={uid}: {e}")
            # Continue validation even if template check fails

        # All checks passed
        info.is_valid = True
        return info
    
    async def _detect_plagiarism(self, miners: list[MinerInfo]) -> list[MinerInfo]:
        """Detect plagiarism by checking duplicate model hashes
        
        Only valid miners are checked. For each unique model hash,
        only the miner with the earliest block is kept as valid.
        
        Note: model_hash is already cached in MinerInfo from _validate_miner()
        
        Args:
            miners: List of validated miners with cached model_hash
            
        Returns:
            Updated miners list with plagiarism detection
        """
        # Group valid miners by model hash (already cached in MinerInfo)
        hash_to_miners: Dict[str, list] = {}
        for miner in miners:
            if miner.is_valid and miner.model_hash:
                if miner.model_hash not in hash_to_miners:
                    hash_to_miners[miner.model_hash] = []
                hash_to_miners[miner.model_hash].append((miner.block, miner.uid, miner))
        
        # Keep only earliest miner for each hash
        for model_hash, group in hash_to_miners.items():
            if len(group) <= 1:
                continue
            
            # Sort by block (earliest first), then by UID
            group.sort(key=lambda x: (x[0], x[1]))
            earliest_block, earliest_uid, _ = group[0]
            
            # Mark duplicates as invalid
            for block, uid, miner in group[1:]:
                if miner.is_valid:
                    miner.is_valid = False
                    miner.invalid_reason = f"model_hash_duplicate:earliest_uid={earliest_uid}"
                    logger.info(
                        f"[MinersMonitor] Plagiarism detected: uid={uid} copied from uid={earliest_uid} "
                        f"(hash={model_hash[:16]}...)"
                    )
        
        return miners
    
    async def refresh_miners(self) -> Dict[str, MinerInfo]:
        """Refresh and validate all miners
        
        Returns:
            Dict of valid miners {key: MinerInfo}
        """
        try:
            logger.info("[MinersMonitor] Refreshing miners from metagraph...")
            
            # Get metagraph and commits
            subtensor = await get_subtensor()
            meta = await subtensor.metagraph(NETUID)
            commits = await subtensor.get_all_revealed_commitments(NETUID)
            
            current_block = await subtensor.get_current_block()
            
            # Load blacklist
            blacklist = await self._load_blacklist()
            
            # Discover and validate miners
            miners = []
            for uid in range(len(meta.hotkeys)):
                hotkey = meta.hotkeys[uid]
                
                # Check blacklist
                if hotkey in blacklist:
                    miners.append(MinerInfo(
                        uid=uid,
                        hotkey=hotkey,
                        model="",
                        revision="",
                        chute_id="",
                        block=0,
                        is_valid=False,
                        invalid_reason="blacklisted"
                    ))
                    continue
                
                # Check for commit
                if hotkey not in commits:
                    miners.append(MinerInfo(
                        uid=uid,
                        hotkey=hotkey,
                        model="",
                        revision="",
                        chute_id="",
                        block=0,
                        is_valid=False,
                        invalid_reason="no_commit"
                    ))
                    continue
                
                try:
                    block, commit_data = commits[hotkey][-1]
                    data = json.loads(commit_data)
                    
                    model = data.get("model", "")
                    revision = data.get("revision", "")
                    chute_id = data.get("chute_id", "")
                    
                    # Check if all required fields present
                    if not model or not revision or not chute_id:
                        miners.append(MinerInfo(
                            uid=uid,
                            hotkey=hotkey,
                            model=model,
                            revision=revision,
                            chute_id=chute_id,
                            block=int(block) if uid != 0 else 0,
                            is_valid=False,
                            invalid_reason="incomplete_commit:missing_fields"
                        ))
                        continue
                    
                    # Validate miner
                    miner_info = await self._validate_miner(
                        uid=uid,
                        hotkey=hotkey,
                        model=model,
                        revision=revision,
                        chute_id=chute_id,
                        block=int(block) if uid != 0 else 0,
                    )
                    
                    miners.append(miner_info)
                    
                except json.JSONDecodeError as e:
                    logger.debug(f"Invalid JSON in commit for uid={uid}: {e}")
                    miners.append(MinerInfo(
                        uid=uid,
                        hotkey=hotkey,
                        model="",
                        revision="",
                        chute_id="",
                        block=0,
                        is_valid=False,
                        invalid_reason="invalid_json_commit"
                    ))
                except Exception as e:
                    logger.debug(f"Failed to validate uid={uid}: {e}")
                    miners.append(MinerInfo(
                        uid=uid,
                        hotkey=hotkey,
                        model="",
                        revision="",
                        chute_id="",
                        block=0,
                        is_valid=False,
                        invalid_reason=f"validation_error:{str(e)[:50]}"
                    ))
            
            # Detect plagiarism
            miners = await self._detect_plagiarism(miners)
            
            # Persist to database
            for miner in miners:
                await self.dao.save_miner(
                    uid=miner.uid,
                    hotkey=miner.hotkey,
                    model=miner.model,
                    revision=miner.revision,
                    chute_id=miner.chute_id,
                    chute_slug=miner.chute_slug,
                    model_hash=miner.model_hash,
                    chute_status=miner.chute_status,
                    is_valid=miner.is_valid,
                    invalid_reason=miner.invalid_reason,
                    block_number=current_block,
                    first_block=miner.block,
                    template_check_result=miner.template_check_result,
                )
            
            valid_miners = {m.key(): m for m in miners if m.is_valid}
            
            self.last_update = int(time.time())
            
            logger.info(
                f"[MinersMonitor] Refreshed {len(miners)} miners "
                f"({len(valid_miners)} valid, {len(miners) - len(valid_miners)} invalid)"
            )
            
            return valid_miners
            
        except Exception as e:
            logger.error(f"[MinersMonitor] Failed to refresh miners: {e}", exc_info=True)
            return {}
    
    async def get_valid_miners(self, force_refresh: bool = False) -> Dict[str, MinerInfo]:
        """Get current valid miner list
        
        Args:
            force_refresh: Whether to force refresh
            
        Returns:
            Miners dictionary {key: MinerInfo}
        """
        # Query from database
        miners_data = await self.dao.get_valid_miners()
        
        # Convert to MinerInfo
        result = {}
        for data in miners_data:
            info = MinerInfo(
                uid=data['uid'],
                hotkey=data['hotkey'],
                model=data['model'],
                revision=data['revision'],
                chute_id=data['chute_id'],
                chute_slug=data.get('chute_slug', ''),
                block=data.get('first_block', 0),
                is_valid=True,
            )
            result[info.key()] = info
        
        return result