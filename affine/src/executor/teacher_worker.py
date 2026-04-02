"""
Teacher Worker - Generates teacher rollouts with logprobs for KL divergence.

Runs as an independent process alongside regular executor workers.
Picks random task_ids from each environment's sampling_list,
evaluates with the teacher model (collect_logprobs=True),
and uploads results to R2 as sequential KL task_ids.

R2 structure:
    teacher_rollouts/
      task_00000000001.json
      task_00000000002.json
      ...
      metadata.json  {"completed_up_to": N}
"""

import asyncio
import json
import os
import random
import time
from typing import Any, Dict, List, Optional

import boto3

from affine.core.setup import logger
from affine.database.dao.system_config import SystemConfigDAO


# Environments to generate teacher rollouts for
TEACHER_ENVS = ["GAME", "SWE-INFINITE", "NAVWORLD"]

# R2 configuration
R2_ENDPOINT = os.getenv(
    "R2_ENDPOINT",
    "https://af76430a7056e37bd99ee03a4468d893.r2.cloudflarestorage.com",
)
R2_BUCKET = os.getenv("R2_BUCKET", "affine-swe-infinite-public")
R2_PREFIX = os.getenv("R2_TEACHER_PREFIX", "teacher_rollouts")
R2_ACCESS_KEY = os.getenv("R2_ACCESS_KEY")
R2_SECRET_KEY = os.getenv("R2_SECRET_KEY")


class TeacherWorker:
    """Generates teacher rollouts and uploads to R2."""

    def __init__(
        self,
        teacher_model: str,
        teacher_base_url: str,
        api_key: str,
        envs: List[str] = None,
        concurrency: int = 2,
    ):
        """
        Args:
            teacher_model: Model name for teacher (e.g., "Qwen/Qwen3-235B-A22B")
            teacher_base_url: API base URL for teacher model
            api_key: API key for teacher model API calls
            envs: Environments to sample from (default: TEACHER_ENVS)
            concurrency: Number of concurrent rollout generators
        """
        self.teacher_model = teacher_model
        self.teacher_base_url = teacher_base_url
        self.api_key = api_key
        self.envs = envs or TEACHER_ENVS
        self.concurrency = concurrency

        self._config_dao = SystemConfigDAO()
        self._s3 = None
        self._next_id = 1
        self._env_instances = {}
        self.running = False

    def _init_s3(self):
        """Initialize S3 client for R2."""
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
        )

    async def _load_next_id(self) -> int:
        """Load next_id from R2 metadata.json."""
        try:
            resp = self._s3.get_object(
                Bucket=R2_BUCKET,
                Key=f"{R2_PREFIX}/metadata.json",
            )
            meta = json.loads(resp["Body"].read())
            return meta.get("completed_up_to", 0) + 1
        except self._s3.exceptions.NoSuchKey:
            return 1
        except Exception as e:
            logger.warning(f"[TEACHER] Failed to load metadata: {e}, starting from 1")
            return 1

    def _upload_rollout(self, task_id: int, data: Dict) -> None:
        """Upload rollout to R2."""
        key = f"{R2_PREFIX}/task_{task_id:011d}.json"
        body = json.dumps(data, separators=(",", ":"))
        self._s3.put_object(Bucket=R2_BUCKET, Key=key, Body=body)

    def _update_metadata(self, completed_up_to: int) -> None:
        """Update metadata.json in R2."""
        key = f"{R2_PREFIX}/metadata.json"
        body = json.dumps({"completed_up_to": completed_up_to})
        self._s3.put_object(Bucket=R2_BUCKET, Key=key, Body=body)

    async def _get_sampling_list(self, env: str) -> List[int]:
        """Get current sampling_list for an environment from SystemConfig."""
        environments = await self._config_dao.get_param_value("environments", default={})
        env_config = environments.get(env, {})
        sampling_config = env_config.get("sampling_config", {})
        return sampling_config.get("sampling_list", [])

    async def _get_env(self, env_name: str):
        """Get environment instance by connecting to existing containers.

        Uses connect_only=True to attach to containers already started by
        regular workers, without recreating them.
        """
        if env_name not in self._env_instances:
            import affinetes as af_env
            from affine.core.environments import SDKEnvironment, ENV_CONFIGS

            if env_name not in ENV_CONFIGS:
                logger.error(f"[TEACHER] Unknown environment: {env_name}")
                return None

            config = ENV_CONFIGS[env_name]

            # Discover hosts using the same config as regular workers
            tmp = SDKEnvironment.__new__(SDKEnvironment)
            tmp.config = config
            tmp._mode_override = None
            hosts, mode = tmp._get_hosts_and_mode()

            try:
                # connect_only requires replicas=1, pick first host
                env_instance = af_env.load_env(
                    image=config.docker_image,
                    mode=mode,
                    hosts=[hosts[0]] if hosts else None,
                    replicas=1,
                    container_name=env_name.lower().replace(":", "-"),
                    connect_only=True,
                )
                self._env_instances[env_name] = env_instance
                logger.info(f"[TEACHER] Connected to {env_name} ({mode}, hosts={hosts})")
            except Exception as e:
                logger.error(f"[TEACHER] Failed to connect to {env_name}: {e}")
                return None

        return self._env_instances[env_name]

    async def _generate_one_rollout(self) -> Optional[Dict]:
        """Pick a random env + task_id, run teacher evaluation, return rollout."""
        # Pick env with available sampling_list
        random.shuffle(self.envs)
        for env_name in self.envs:
            sampling_list = await self._get_sampling_list(env_name)
            if not sampling_list:
                continue

            task_id = random.choice(sampling_list)
            logger.info(f"[TEACHER] Generating rollout: env={env_name} task_id={task_id}")

            try:
                env = await self._get_env(env_name)
                if env is None:
                    continue

                result = await env.evaluate(
                    task_id=task_id,
                    model=self.teacher_model,
                    base_url=self.teacher_base_url,
                    api_key=self.api_key,
                    collect_logprobs=True,
                )

                # URL mode returns dict directly
                if isinstance(result, dict):
                    extra = result.get("extra", {})
                    score = result.get("score", 0.0)
                    error = result.get("error")
                else:
                    extra = result.extra or {}
                    score = result.score
                    error = result.error

                full_logprobs = extra.get("full_logprobs")

                if not full_logprobs:
                    logger.warning(
                        f"[TEACHER] No logprobs returned for {env_name}/{task_id}: "
                        f"error={extra.get('logprobs_error', error)}"
                    )
                    return None

                return {
                    "env": env_name,
                    "original_task_id": task_id,
                    "score": float(score),
                    "full_logprobs": full_logprobs,
                    "teacher_model": self.teacher_model,
                    "timestamp": time.time(),
                }

            except Exception as e:
                logger.error(f"[TEACHER] Failed {env_name}/{task_id}: {e}")
                return None

        logger.warning("[TEACHER] No environments with sampling_list available")
        return None

    async def _worker_loop(self, worker_id: int):
        """Single worker loop: generate rollouts continuously."""
        while self.running:
            try:
                rollout = await self._generate_one_rollout()

                if rollout:
                    async with self._upload_lock:
                        tid = self._next_id
                        self._upload_rollout(tid, rollout)
                        self._update_metadata(tid)
                        self._next_id += 1
                    logger.info(
                        f"[TEACHER-{worker_id}] Uploaded task_{tid:011d}.json "
                        f"(env={rollout['env']} original_task={rollout['original_task_id']} "
                        f"score={rollout['score']:.2f})"
                    )
                else:
                    await asyncio.sleep(10)

            except Exception as e:
                logger.error(f"[TEACHER-{worker_id}] Loop error: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def run(self):
        """Main loop: run concurrent workers generating rollouts."""
        self._init_s3()
        self._next_id = await self._load_next_id()
        self._upload_lock = asyncio.Lock()
        logger.info(
            f"[TEACHER] Starting: model={self.teacher_model} "
            f"envs={self.envs} next_id={self._next_id} concurrency={self.concurrency}"
        )
        self.running = True

        workers = [
            asyncio.create_task(self._worker_loop(i))
            for i in range(self.concurrency)
        ]
        await asyncio.gather(*workers, return_exceptions=True)

    def stop(self):
        self.running = False


def run_teacher_process(
    teacher_model: str,
    teacher_base_url: str,
    api_key: str,
    envs: List[str] = None,
    concurrency: int = 2,
    verbosity: int = 1,
):
    """Entry point for teacher worker subprocess."""
    from affine.core.setup import setup_logging
    setup_logging(verbosity, component="teacher")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Initialize DynamoDB client in subprocess
    from affine.database.client import init_client
    loop.run_until_complete(init_client())

    worker = TeacherWorker(
        teacher_model=teacher_model,
        teacher_base_url=teacher_base_url,
        api_key=api_key,
        envs=envs,
        concurrency=concurrency,
    )

    try:
        loop.run_until_complete(worker.run())
    except KeyboardInterrupt:
        logger.info("[TEACHER] Received interrupt")
    except Exception as e:
        logger.error(f"[TEACHER] Fatal: {e}", exc_info=True)
    finally:
        worker.stop()
        try:
            pending = asyncio.all_tasks(loop)
            for t in pending:
                t.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception:
            pass
        loop.close()
