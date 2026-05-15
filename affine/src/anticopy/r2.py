"""
R2 client for CEAC rollouts + scores. Both buckets are private; we
optionally tee through a local on-disk cache so the worker can re-read
recent rollouts without going to R2 on every job.

Reuses the same R2 endpoint + env credentials as ``teacher.mover``
(``R2_ENDPOINT``, ``R2_ACCESS_KEY``, ``R2_SECRET_KEY``).
"""

from __future__ import annotations

import gzip
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import boto3
from botocore.config import Config as BotoConfig

from affine.core.setup import logger


R2_ENDPOINT = os.getenv(
    "R2_ENDPOINT",
    "https://af76430a7056e37bd99ee03a4468d893.r2.cloudflarestorage.com",
)
R2_ACCESS_KEY = os.getenv("R2_ACCESS_KEY")
R2_SECRET_KEY = os.getenv("R2_SECRET_KEY")

# Buckets — keep the names env-overridable so staging deployments can
# point at parallel buckets without code edits.
ROLLOUTS_BUCKET = os.getenv("R2_ANTICOPY_ROLLOUTS_BUCKET", "affine-anticopy-rollouts")
SCORES_BUCKET = os.getenv("R2_ANTICOPY_SCORES_BUCKET", "affine-anticopy-scores")

# Optional local cache directory. When set, ``get_rollout`` / ``get_score``
# look here first and only fall back to R2 on miss; ``put_*`` always
# writes through to the local copy too.
LOCAL_CACHE_DIR = os.getenv("ANTICOPY_CACHE_DIR", "/var/cache/anticopy")


def _key_for_rollout(champion_hotkey: str, env: str, task_id: int) -> str:
    return f"{champion_hotkey}/{env}/{int(task_id)}.json.gz"


def _key_for_score(hotkey: str, revision: str) -> str:
    return f"{hotkey}/{revision}.json.gz"


class AntiCopyR2:
    """Thin sync wrapper over the boto3 S3 client.

    Methods are blocking by design — callers should wrap in
    ``asyncio.to_thread`` when running inside the async services. The
    refresh / worker entry points already do so.
    """

    def __init__(
        self,
        rollouts_bucket: str = ROLLOUTS_BUCKET,
        scores_bucket: str = SCORES_BUCKET,
        cache_dir: Optional[str] = LOCAL_CACHE_DIR,
    ):
        self.rollouts_bucket = rollouts_bucket
        self.scores_bucket = scores_bucket
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._s3 = None

    # -------- low-level S3 / cache ------------------------------------

    def _init_s3(self):
        if self._s3 is not None:
            return
        if not R2_ACCESS_KEY or not R2_SECRET_KEY:
            raise RuntimeError(
                "R2_ACCESS_KEY and R2_SECRET_KEY env vars are required"
            )
        self._s3 = boto3.client(
            "s3",
            endpoint_url=R2_ENDPOINT,
            aws_access_key_id=R2_ACCESS_KEY,
            aws_secret_access_key=R2_SECRET_KEY,
            region_name="auto",
            config=BotoConfig(
                connect_timeout=10,
                read_timeout=120,
                retries={"max_attempts": 3, "mode": "adaptive"},
            ),
        )

    def _cache_path(self, bucket: str, key: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        return self.cache_dir / bucket / key

    def _cache_read(self, bucket: str, key: str) -> Optional[bytes]:
        path = self._cache_path(bucket, key)
        if path is None or not path.exists():
            return None
        try:
            return path.read_bytes()
        except OSError as e:
            logger.debug(f"[anticopy.r2] cache read failed {path}: {e}")
            return None

    def _cache_write(self, bucket: str, key: str, data: bytes) -> None:
        path = self._cache_path(bucket, key)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_bytes(data)
            tmp.replace(path)
        except OSError as e:
            logger.debug(f"[anticopy.r2] cache write failed {path}: {e}")

    def _put_blob(self, bucket: str, key: str, payload: Dict[str, Any]) -> None:
        self._init_s3()
        raw = gzip.compress(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        self._s3.put_object(Bucket=bucket, Key=key, Body=raw)
        self._cache_write(bucket, key, raw)

    def _get_blob(self, bucket: str, key: str) -> Optional[Dict[str, Any]]:
        cached = self._cache_read(bucket, key)
        if cached is not None:
            try:
                return json.loads(gzip.decompress(cached).decode("utf-8"))
            except (OSError, ValueError) as e:
                logger.debug(f"[anticopy.r2] cache decode failed {key}: {e}")
        self._init_s3()
        try:
            resp = self._s3.get_object(Bucket=bucket, Key=key)
            raw = resp["Body"].read()
        except self._s3.exceptions.NoSuchKey:
            return None
        self._cache_write(bucket, key, raw)
        return json.loads(gzip.decompress(raw).decode("utf-8"))

    # -------- rollouts ------------------------------------------------

    def put_rollout(
        self,
        *,
        champion_hotkey: str,
        env: str,
        task_id: int,
        payload: Dict[str, Any],
    ) -> str:
        """Upload one rollout. Returns the bucket key for reference."""
        key = _key_for_rollout(champion_hotkey, env, task_id)
        self._put_blob(self.rollouts_bucket, key, payload)
        return key

    def get_rollout(
        self, *, champion_hotkey: str, env: str, task_id: int
    ) -> Optional[Dict[str, Any]]:
        return self._get_blob(
            self.rollouts_bucket, _key_for_rollout(champion_hotkey, env, task_id)
        )

    def get_rollout_by_key(self, r2_key: str) -> Optional[Dict[str, Any]]:
        return self._get_blob(self.rollouts_bucket, r2_key)

    # -------- scores --------------------------------------------------

    def put_score(
        self, *, hotkey: str, revision: str, payload: Dict[str, Any]
    ) -> str:
        key = _key_for_score(hotkey, revision)
        self._put_blob(self.scores_bucket, key, payload)
        return key

    def get_score(
        self, *, hotkey: str, revision: str
    ) -> Optional[Dict[str, Any]]:
        return self._get_blob(self.scores_bucket, _key_for_score(hotkey, revision))

    def get_score_by_key(self, r2_key: str) -> Optional[Dict[str, Any]]:
        return self._get_blob(self.scores_bucket, r2_key)
