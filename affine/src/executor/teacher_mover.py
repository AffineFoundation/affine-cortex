"""
Teacher Mover - Promotes teacher rollouts from the private to public bucket.

Runs alongside teacher_worker. Each tick:
  1. Reads the DISTILL env config from SystemConfig (DynamoDB) to pick up
     the current rotation_interval/rotation_count and enabled flag.
  2. Lists pending/{env}/* in the private bucket.
  3. Builds a candidate pool from all pending envs (with optional
     low-priority deprioritization via ``LOW_PRIORITY_ENVS``).
  4. Randomly samples target_count candidates without replacement.
  5. For each: copies the rollout to public/task_{next_id:011d}.json,
     moves the source from pending/ to promoted/, increments next_id.
  6. Updates public/metadata.json with the new completed_up_to.

Failed individual moves are skipped and replaced with another random
draw, so the batch always reaches target_count successful promotions
unless the candidate pool is exhausted.
"""

import asyncio
import json
import os
import random
from typing import Dict, List, Optional, Tuple

import boto3
from botocore.config import Config as BotoConfig

from affine.core.setup import logger
from affine.database.dao.system_config import SystemConfigDAO


# R2 endpoint and credentials are shared with teacher_worker
R2_ENDPOINT = os.getenv(
    "R2_ENDPOINT",
    "https://af76430a7056e37bd99ee03a4468d893.r2.cloudflarestorage.com",
)
R2_ACCESS_KEY = os.getenv("R2_ACCESS_KEY")
R2_SECRET_KEY = os.getenv("R2_SECRET_KEY")

# Source (private) and target (public) buckets
R2_DISTILL_PRIVATE_BUCKET = os.getenv("R2_DISTILL_PRIVATE_BUCKET", "affine-distill-private")
R2_DISTILL_PUBLIC_BUCKET = os.getenv(
    "R2_DISTILL_PUBLIC_BUCKET", "affine-distill-public"
)

# Fixed layout inside the private bucket
PENDING_PREFIX = "pending"
PROMOTED_PREFIX = "promoted"

# Envs deprioritized in random selection (empty = uniform over all envs)
LOW_PRIORITY_ENVS: set = set()

# SystemConfig env key the mover tracks
DISTILL_ENV_KEY = "DISTILL"

# Fallback sleep when the config lookup fails, so we retry soon
CONFIG_RETRY_SLEEP_SEC = 60


class TeacherMover:
    """Promotes random pending rollouts from private to public.

    Cadence and batch size are driven by the DISTILL entry in
    SystemConfig (DynamoDB), not by environment variables, so operators
    can retune the pipeline from the same place they tune the distill
    env's rotation behavior.
    """

    def __init__(
        self,
        private_bucket: str = R2_DISTILL_PRIVATE_BUCKET,
        public_bucket: str = R2_DISTILL_PUBLIC_BUCKET,
    ):
        self.private_bucket = private_bucket
        self.public_bucket = public_bucket
        self.pending_prefix = PENDING_PREFIX
        self.promoted_prefix = PROMOTED_PREFIX
        self._s3 = None
        self._config_dao = SystemConfigDAO()
        self.running = False

    def _init_s3(self):
        if self._s3 is not None:
            return
        if not R2_ACCESS_KEY or not R2_SECRET_KEY:
            raise RuntimeError("R2_ACCESS_KEY and R2_SECRET_KEY required")
        self._s3 = boto3.client(
            "s3",
            endpoint_url=R2_ENDPOINT,
            aws_access_key_id=R2_ACCESS_KEY,
            aws_secret_access_key=R2_SECRET_KEY,
            region_name="auto",
            config=BotoConfig(
                connect_timeout=10,
                read_timeout=60,
                retries={"max_attempts": 3, "mode": "adaptive"},
            ),
        )

    # ---------------- SystemConfig lookup ----------------

    async def _read_distill_config(self) -> Optional[Tuple[int, int, bool]]:
        """Return (interval_sec, count, enabled) from DISTILL SystemConfig.

        Returns None if the DISTILL entry is missing or malformed.
        """
        try:
            environments = await self._config_dao.get_param_value(
                "environments", default={}
            )
        except Exception as e:
            logger.warning(f"[MOVER] Failed to read SystemConfig: {e}")
            return None

        distill = environments.get(DISTILL_ENV_KEY)
        if not distill:
            return None

        enabled = bool(distill.get("enabled_for_sampling", False))
        sampling_config = distill.get("sampling_config") or {}
        try:
            interval_sec = int(sampling_config.get("rotation_interval", 0))
            count = int(sampling_config.get("rotation_count", 0))
        except (TypeError, ValueError):
            return None
        if interval_sec <= 0 or count <= 0:
            return None
        return interval_sec, count, enabled

    # ---------------- Listing & metadata ----------------

    def _list_pending(self) -> Dict[str, List[str]]:
        """Return {env_name: [keys]} of all pending rollouts."""
        by_env: Dict[str, List[str]] = {}
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(
            Bucket=self.private_bucket,
            Prefix=f"{self.pending_prefix}/",
        ):
            for obj in page.get("Contents", []) or []:
                key = obj["Key"]
                # key format: pending/{ENV}/{filename}
                parts = key.split("/", 2)
                if len(parts) < 3 or not parts[2]:
                    continue
                env = parts[1]
                by_env.setdefault(env, []).append(key)
        return by_env

    def _load_public_next_id(self) -> int:
        """Read the public metadata.json to get the next public task_id."""
        try:
            resp = self._s3.get_object(
                Bucket=self.public_bucket, Key="metadata.json"
            )
            meta = json.loads(resp["Body"].read())
            return int(meta.get("completed_up_to", 0)) + 1
        except self._s3.exceptions.NoSuchKey:
            return 1
        except Exception as e:
            logger.warning(
                f"[MOVER] Failed to load public metadata: {e}, starting from 1"
            )
            return 1

    def _write_public_metadata(self, completed_up_to: int) -> None:
        body = json.dumps({"completed_up_to": completed_up_to})
        self._s3.put_object(
            Bucket=self.public_bucket, Key="metadata.json", Body=body
        )

    # ---------------- Candidate selection ----------------

    @staticmethod
    def _build_candidate_pools(
        by_env: Dict[str, List[str]],
    ) -> Tuple[List[str], List[str]]:
        """Split candidates into (high_priority, low_priority) lists."""
        high: List[str] = []
        low: List[str] = []
        for env, keys in by_env.items():
            target = low if env.upper() in LOW_PRIORITY_ENVS else high
            target.extend(keys)
        random.shuffle(high)
        random.shuffle(low)
        return high, low

    def _draw_one(
        self, high: List[str], low: List[str]
    ) -> Optional[str]:
        """Pop a single candidate, preferring high priority."""
        if high:
            return high.pop()
        if low:
            return low.pop()
        return None

    # ---------------- Move a single rollout ----------------

    def _promote_one(self, src_key: str, public_id: int) -> bool:
        """Promote a single rollout. Returns True on success."""
        public_key = f"task_{public_id:011d}.json"
        try:
            # 1. Read source
            resp = self._s3.get_object(
                Bucket=self.private_bucket, Key=src_key
            )
            body = resp["Body"].read()

            # 2. Write to public
            self._s3.put_object(
                Bucket=self.public_bucket, Key=public_key, Body=body
            )

            # 3. Move source pending -> promoted (copy + delete)
            promoted_key = src_key.replace(
                f"{self.pending_prefix}/",
                f"{self.promoted_prefix}/",
                1,
            )
            self._s3.copy_object(
                Bucket=self.private_bucket,
                Key=promoted_key,
                CopySource={"Bucket": self.private_bucket, "Key": src_key},
            )
            self._s3.delete_object(Bucket=self.private_bucket, Key=src_key)

            logger.info(
                f"[MOVER] Promoted {src_key} -> {public_key}"
            )
            return True
        except Exception as e:
            logger.error(
                f"[MOVER] Failed to promote {src_key}: {e}"
            )
            return False

    # ---------------- One tick ----------------

    def _tick(self, target_count: int) -> int:
        """Run one promotion batch. Returns number of successful moves."""
        try:
            by_env = self._list_pending()
        except Exception as e:
            logger.error(f"[MOVER] Failed to list pending: {e}")
            return 0

        total_pending = sum(len(v) for v in by_env.values())
        if total_pending == 0:
            logger.info("[MOVER] No pending rollouts to promote")
            return 0

        logger.info(
            f"[MOVER] Pending: total={total_pending} "
            f"by_env={ {k: len(v) for k, v in by_env.items()} }"
        )

        high, low = self._build_candidate_pools(by_env)
        next_id = self._load_public_next_id()
        promoted = 0
        attempts = 0
        # Try up to target_count + total_pending so failed items can be replaced.
        max_attempts = target_count + total_pending
        while promoted < target_count and attempts < max_attempts:
            src = self._draw_one(high, low)
            if src is None:
                break
            attempts += 1
            if self._promote_one(src, next_id):
                next_id += 1
                promoted += 1

        if promoted > 0:
            try:
                self._write_public_metadata(next_id - 1)
            except Exception as e:
                logger.error(
                    f"[MOVER] Promoted {promoted} but failed to write public "
                    f"metadata: {e}"
                )
        logger.info(
            f"[MOVER] Tick complete: promoted={promoted}/{target_count} "
            f"public_completed_up_to={next_id - 1}"
        )
        return promoted

    # ---------------- Main loop ----------------

    async def _sleep_interruptible(self, seconds: int) -> None:
        """Sleep in small chunks so stop() is responsive."""
        slept = 0
        while self.running and slept < seconds:
            chunk = min(5, seconds - slept)
            await asyncio.sleep(chunk)
            slept += chunk

    async def run(self):
        """Main loop: read config each tick, promote a batch, sleep."""
        self._init_s3()
        logger.info(
            f"[MOVER] Starting: private={self.private_bucket} "
            f"public={self.public_bucket} (cadence driven by "
            f"SystemConfig.{DISTILL_ENV_KEY}.sampling_config)"
        )
        self.running = True
        while self.running:
            cfg = await self._read_distill_config()
            if cfg is None:
                logger.warning(
                    f"[MOVER] {DISTILL_ENV_KEY} config missing or invalid, "
                    f"retrying in {CONFIG_RETRY_SLEEP_SEC}s"
                )
                await self._sleep_interruptible(CONFIG_RETRY_SLEEP_SEC)
                continue

            interval_sec, target_count, enabled = cfg
            if not enabled:
                logger.info(
                    f"[MOVER] {DISTILL_ENV_KEY}.enabled_for_sampling is "
                    f"false, pausing for {interval_sec}s"
                )
                await self._sleep_interruptible(interval_sec)
                continue

            try:
                await asyncio.to_thread(self._tick, target_count)
            except Exception as e:
                logger.error(f"[MOVER] Tick error: {e}", exc_info=True)

            await self._sleep_interruptible(interval_sec)

    def stop(self):
        self.running = False


def run_mover_process(verbosity: int = 1):
    """Entry point for running the mover as a standalone subprocess."""
    from affine.core.setup import setup_logging

    setup_logging(verbosity, component="mover")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # DynamoDB client has to be initialized inside the subprocess event loop
    from affine.database.client import init_client
    loop.run_until_complete(init_client())

    mover = TeacherMover()

    try:
        loop.run_until_complete(mover.run())
    except KeyboardInterrupt:
        logger.info("[MOVER] Received interrupt")
    except Exception as e:
        logger.error(f"[MOVER] Fatal: {e}", exc_info=True)
    finally:
        mover.stop()
        try:
            pending = asyncio.all_tasks(loop)
            for t in pending:
                t.cancel()
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
        except Exception:
            pass
        loop.close()
