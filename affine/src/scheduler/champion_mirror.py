"""Mirror champion checkpoints into official HuggingFace repos."""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import tempfile
from dataclasses import replace
from typing import Optional
from urllib.parse import quote

from huggingface_hub import HfApi

from affine.core.setup import logger
from affine.src.scorer.window_state import ChampionRecord


OFFICIAL_HF_TOKEN_ENV = "AFFINE_OFFICIAL_HF_TOKEN"
OFFICIAL_HF_NAMESPACE_ENV = "AFFINE_OFFICIAL_HF_NAMESPACE"
OFFICIAL_HF_NAMESPACE_DEFAULT = "AffineFoundation"


class ChampionMirror:
    """Pushes the champion's exact git commit to an official affine repo."""

    def __init__(
        self,
        *,
        token: Optional[str] = None,
        namespace: Optional[str] = None,
    ):
        self.token = (token if token is not None else os.getenv(OFFICIAL_HF_TOKEN_ENV, "")).strip()
        self.namespace = (
            namespace
            if namespace is not None
            else os.getenv(OFFICIAL_HF_NAMESPACE_ENV, OFFICIAL_HF_NAMESPACE_DEFAULT)
        ).strip()

    def enabled(self) -> bool:
        return bool(self.token and self.namespace)

    async def ensure_mirrored(self, champion: ChampionRecord) -> ChampionRecord:
        if not self.enabled():
            return champion
        repo_id = _target_repo_id(champion.model, self.namespace)
        if champion.model == repo_id:
            return champion
        try:
            return await asyncio.to_thread(self._mirror_sync, champion, repo_id)
        except Exception as e:
            logger.warning(
                f"ChampionMirror: mirror failed uid={champion.uid} "
                f"{champion.model}@{champion.revision[:8]} -> {repo_id}: "
                f"{type(e).__name__}: {e}"
            )
            return champion

    def _mirror_sync(
        self, champion: ChampionRecord, target_repo_id: str
    ) -> ChampionRecord:
        api = HfApi(token=self.token)
        api.create_repo(
            repo_id=target_repo_id,
            repo_type="model",
            token=self.token,
            exist_ok=True,
        )
        with tempfile.TemporaryDirectory(prefix="affine-champion-") as tmp:
            repo_dir = os.path.join(tmp, "repo")
            source_url = _hf_git_url(champion.model, os.getenv("HF_TOKEN"))
            target_url = _hf_git_url(target_repo_id, self.token)
            self._run_git(["clone", source_url, repo_dir], cwd=tmp)
            self._run_git(["checkout", "--detach", champion.revision], cwd=repo_dir)
            self._run_git(["lfs", "install", "--local"], cwd=repo_dir)
            self._run_git(["lfs", "fetch", "origin", champion.revision], cwd=repo_dir)
            self._run_git(["lfs", "checkout"], cwd=repo_dir)
            self._run_git(["remote", "add", "official", target_url], cwd=repo_dir)
            branch = _branch_name(champion)
            self._run_git(
                ["push", "official", f"HEAD:refs/heads/{branch}"],
                cwd=repo_dir,
                timeout=3600,
            )
            self._run_git(
                ["push", "--force", "official", "HEAD:refs/heads/main"],
                cwd=repo_dir,
                timeout=3600,
            )
        logger.info(
            f"ChampionMirror: mirrored champion uid={champion.uid} "
            f"{champion.model}@{champion.revision[:8]} -> "
            f"{target_repo_id}@{champion.revision[:8]}"
        )
        return replace(champion, model=target_repo_id)

    def _run_git(
        self,
        args: list[str],
        *,
        cwd: str,
        timeout: int = 1800,
    ) -> None:
        cmd = ["git", *args]
        result = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"git {_mask_args(args)} failed with code {result.returncode}: "
                f"{_mask_text(result.stderr or result.stdout)}"
            )


def _hf_git_url(repo_id: str, token: Optional[str]) -> str:
    if token:
        return f"https://__token__:{quote(token, safe='')}@huggingface.co/{repo_id}"
    return f"https://huggingface.co/{repo_id}"


def _target_repo_id(source_repo_id: str, namespace: str) -> str:
    name = source_repo_id.rstrip("/").split("/")[-1]
    if not name.startswith("affine-"):
        name = f"affine-{name}"
    return f"{namespace}/{name}"


def _branch_name(champion: ChampionRecord) -> str:
    safe_hotkey = "".join(
        c if c.isalnum() or c in {"-", "_"} else "-"
        for c in champion.hotkey[:16]
    ).strip("-_") or "hotkey"
    return f"champions/uid-{champion.uid}-{safe_hotkey}-{champion.revision[:12]}"


def _mask_args(args: list[str]) -> str:
    return " ".join(_mask_text(arg) for arg in args)


def _mask_text(text: str) -> str:
    return re.sub(r"__token__:[^@\s]+", "__token__:***", text)
