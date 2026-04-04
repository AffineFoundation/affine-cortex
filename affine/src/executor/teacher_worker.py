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

import click
import boto3
from botocore.config import Config as BotoConfig

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
        self._env_connect_lock = asyncio.Lock()
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
            config=BotoConfig(
                connect_timeout=10,
                read_timeout=30,
                retries={"max_attempts": 3, "mode": "adaptive"},
            ),
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
        """Get or create a dedicated teacher container for the environment.

        Creates its own container (1 replica on the first available host)
        so the teacher runs independently from executor workers.
        """
        if env_name in self._env_instances:
            return self._env_instances[env_name]

        async with self._env_connect_lock:
            if env_name in self._env_instances:
                return self._env_instances[env_name]

            import affinetes as af_env
            from affine.core.environments import (
                SDKEnvironment, ENV_CONFIGS, convert_memory_format,
            )

            if env_name not in ENV_CONFIGS:
                logger.error(f"[TEACHER] Unknown environment: {env_name}")
                return None

            config = ENV_CONFIGS[env_name]

            # Reuse host/mode discovery from SDKEnvironment
            tmp = SDKEnvironment.__new__(SDKEnvironment)
            tmp.config = config
            tmp._mode_override = None
            hosts, mode = tmp._get_hosts_and_mode()

            # Collect env vars the same way SDKEnvironment does
            env_vars = dict(config.env_vars)
            api_key = os.getenv("CHUTES_API_KEY", "")
            if api_key:
                env_vars.update({"CHUTES_API_KEY": api_key, "API_KEY": api_key})
            if "task_type" in config.eval_params:
                env_vars["ENV_NAME"] = config.eval_params["task_type"]
            for key in config.required_env_vars:
                val = os.getenv(key, "")
                if val:
                    env_vars[key] = val

            mem_limit = convert_memory_format(config.mem_limit, mode)
            base_name = env_name.lower().replace(":", "-")

            load_kwargs = {
                "image": config.docker_image,
                "mode": mode,
                "env_vars": env_vars,
                "mem_limit": mem_limit,
                "pull": True,
                "replicas": 1,
                "hosts": [hosts[0]] if hosts else None,
                "container_name": f"teacher-{base_name}",
                "force_recreate": True,
            }
            if config.volumes:
                load_kwargs["volumes"] = config.volumes

            try:
                env_instance = af_env.load_env(**load_kwargs)
                self._env_instances[env_name] = env_instance
                logger.info(
                    f"[TEACHER] Created container teacher-{base_name} "
                    f"({mode}, host={hosts[0] if hosts else 'local'})"
                )
            except Exception as e:
                logger.error(f"[TEACHER] Failed to create container for {env_name}: {e}")
                return None

            return self._env_instances.get(env_name)

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
                logger.info(
                    f"[TEACHER-DEBUG] {env_name}/{task_id}: score={score} "
                    f"logprobs type={type(full_logprobs).__name__} "
                    f"len={len(full_logprobs) if isinstance(full_logprobs, (list, dict, str)) else full_logprobs} "
                    f"usage={extra.get('usage')}"
                )

                if not full_logprobs:
                    logger.warning(
                        f"[TEACHER] No logprobs returned for {env_name}/{task_id}: "
                        f"error={extra.get('logprobs_error', error)} "
                        f"extra_keys={list(extra.keys())} score={score}"
                    )
                    continue

                if score <= 0:
                    logger.info(
                        f"[TEACHER] Skipping failed rollout: env={env_name} "
                        f"task_id={task_id} score={score}"
                    )
                    continue

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
                continue

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
                        await asyncio.to_thread(self._upload_rollout, tid, rollout)
                        await asyncio.to_thread(self._update_metadata, tid)
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


async def run_service(
    teacher_model: str,
    teacher_base_url: str,
    api_key: str,
    envs: List[str],
    concurrency: int,
):
    """Run teacher worker as a standalone service."""
    import signal

    from affine.database.client import init_client
    await init_client()

    worker = TeacherWorker(
        teacher_model=teacher_model,
        teacher_base_url=teacher_base_url,
        api_key=api_key,
        envs=envs,
        concurrency=concurrency,
    )

    shutdown_event = asyncio.Event()

    def handle_shutdown(sig):
        logger.info(f"[TEACHER] Received signal {sig}, shutting down...")
        worker.stop()
        shutdown_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: handle_shutdown(s))

    try:
        await worker.run()
    except Exception as e:
        logger.error(f"[TEACHER] Fatal: {e}", exc_info=True)
        raise
    finally:
        worker.stop()


@click.command()
@click.option("--model", envvar="TEACHER_MODEL", required=True, help="Teacher model name")
@click.option("--base-url", envvar="TEACHER_BASE_URL", required=True, help="Teacher model API base URL")
@click.option("--api-key", envvar="TEACHER_API_KEY", default="", help="Teacher model API key")
@click.option("--envs", envvar="TEACHER_ENVS", default="GAME,SWE-INFINITE,NAVWORLD", help="Comma-separated environment list")
@click.option("--concurrency", envvar="TEACHER_CONCURRENCY", default=2, type=int, help="Number of concurrent workers")
@click.option("-v", "--verbosity", default=None, type=click.Choice(["0", "1", "2", "3"]), help="Logging verbosity")
def main(model, base_url, api_key, envs, concurrency, verbosity):
    """
    Affine Teacher Worker - Generate teacher rollouts with logprobs.

    Picks random tasks from environment sampling lists, evaluates with the
    teacher model (collecting logprobs), and uploads results to R2.
    """
    from affine.core.setup import setup_logging

    verbosity_val = int(verbosity) if verbosity is not None else 1
    setup_logging(verbosity_val, component="teacher")

    env_list = [e.strip() for e in envs.split(",") if e.strip()]
    logger.info(f"[TEACHER] Starting standalone: model={model} envs={env_list} concurrency={concurrency}")

    asyncio.run(run_service(
        teacher_model=model,
        teacher_base_url=base_url,
        api_key=api_key,
        envs=env_list,
        concurrency=concurrency,
    ))


if __name__ == "__main__":
    main()
