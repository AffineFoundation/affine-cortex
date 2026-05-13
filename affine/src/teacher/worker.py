"""
Teacher Worker - Generates teacher rollouts with logprobs for KL divergence.

Runs as an independent process alongside regular executor workers.
Picks random task_ids from each environment's sampling_list, evaluates
with the teacher model (collect_logprobs=True), and uploads results to a
private R2 bucket. A separate teacher_mover process periodically promotes
a random subset to the public bucket that the distill env reads from.

Private bucket layout:
    pending/{ENV}/{epoch_ms}.json   - new rollouts waiting to be promoted
    promoted/{ENV}/{epoch_ms}.json  - already promoted by the mover
"""

import asyncio
import json
import os
import random
import time
from typing import Dict, List, Optional, Sequence

import click
import boto3
from botocore.config import Config as BotoConfig

from affine.core.setup import logger
from affine.database.dao.system_config import SystemConfigDAO


# Environments to generate teacher rollouts for
TEACHER_ENVS = ["CORPUS-EVAL"]

# corpus-eval has an unbounded virtual task_id space (each id maps
# deterministically to a climbmix slice + teacher continuation), so we
# synthesize a large virtual sampling range when SystemConfig doesn't
# define one. This is the teacher-side fallback; the regular executor
# never goes through here.
_CORPUS_EVAL_VIRTUAL_RANGE = 1_000_000_000

# corpus_eval's ROW_STRIDE — the number of consecutive task_ids that
# share the same climbmix shard in the corpus_eval container's LRU
# cache. Matching this on the teacher side lets us walk a random
# segment sequentially so a burst of ~64 rollouts reuses the hot
# shard instead of triggering a fresh ~92 MB download each time.
_CORPUS_EVAL_SEGMENT_SIZE = 64

# R2 configuration
R2_ENDPOINT = os.getenv(
    "R2_ENDPOINT",
    "https://af76430a7056e37bd99ee03a4468d893.r2.cloudflarestorage.com",
)
# Private bucket the teacher writes to. The mover will promote a random
# subset to the corresponding public bucket.
R2_DISTILL_PRIVATE_BUCKET = os.getenv("R2_DISTILL_PRIVATE_BUCKET", "affine-distill-private")
PENDING_PREFIX = "pending"
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
        self._env_instances = {}
        self._env_connect_lock = asyncio.Lock()
        # Per-env segment state for corpus_eval shard-locality sampling:
        # env_name -> (segment_base, offset_within_segment). Protected
        # by _sampling_lock so concurrent workers don't race on advance.
        self._corpus_segment_state: Dict[str, tuple] = {}
        self._sampling_lock = asyncio.Lock()
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

    def _upload_rollout(self, env_name: str, data: Dict) -> str:
        """Upload a rollout to the private bucket under pending/{env}/.

        File name is the upload epoch in milliseconds, which is sortable
        and collision-free at the teacher's serial upload rate. Returns
        the key written.
        """
        epoch_ms = int(time.time() * 1000)
        key = f"{PENDING_PREFIX}/{env_name}/{epoch_ms}.json"
        body = json.dumps(data, separators=(",", ":"))
        self._s3.put_object(Bucket=R2_DISTILL_PRIVATE_BUCKET, Key=key, Body=body)
        return key

    async def _get_sampling_list(self, env: str) -> Sequence[int]:
        """Return a draw pool of task_ids for ``env``.

        The queue-window refactor swapped the legacy materialized
        ``sampling_list`` for a compact ``dataset_range`` (list of
        ``[start, end)`` intervals). For the teacher's purposes we
        collapse the first non-empty interval into a Python ``range``
        — ``random.choice`` works on it lazily.

        corpus-eval keeps its virtual fallback for the case where no
        config is defined for it (it has an unbounded task_id space;
        the env container synthesizes content on demand).
        """
        environments = await self._config_dao.get_param_value("environments", default={})
        env_config = environments.get(env, {})
        # Stage U shape: env_config['sampling']['dataset_range']. Fall back
        # to the older nested ``window_config`` key for forward-compat with
        # any partially-migrated system_config rows.
        sampling = env_config.get("sampling") or env_config.get("window_config") or {}
        intervals = sampling.get("dataset_range") or []
        for start, end in intervals:
            if end > start:
                return range(int(start), int(end))
        if env.upper() == "CORPUS-EVAL":
            # range is a lazy sequence; random.choice works on it
            # without materializing ~1B ints into memory.
            return range(_CORPUS_EVAL_VIRTUAL_RANGE)
        return []

    async def _pick_task_id(
        self, env_name: str, sampling_list: Sequence[int]
    ) -> int:
        """Draw a task_id from ``sampling_list`` for ``env_name``.

        For CORPUS-EVAL sampling over the virtual range, we do
        segment-random selection: pick a random 64-aligned base once,
        then walk ``_CORPUS_EVAL_SEGMENT_SIZE`` consecutive ids before
        rerolling. Consecutive ids share a shard in the corpus_eval
        container, so each segment gets served from the hot LRU entry
        instead of forcing a ~92 MB download per rollout. For all
        other envs (or for custom sampling_list overrides) fall back
        to uniform random.choice.
        """
        is_corpus_range = (
            env_name.upper() == "CORPUS-EVAL"
            and isinstance(sampling_list, range)
        )
        if not is_corpus_range:
            return random.choice(sampling_list)

        async with self._sampling_lock:
            state = self._corpus_segment_state.get(env_name)
            if state is None or state[1] >= _CORPUS_EVAL_SEGMENT_SIZE:
                n = len(sampling_list)
                # Align segment base to the stride so (task_id // 64)
                # matches corpus_eval's shard_idx computation exactly.
                max_base = max(0, n - _CORPUS_EVAL_SEGMENT_SIZE)
                base = random.randrange(
                    0, max_base + 1, _CORPUS_EVAL_SEGMENT_SIZE
                )
                offset = 0
            else:
                base, offset = state
            task_id = base + offset
            self._corpus_segment_state[env_name] = (base, offset + 1)
            return task_id

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

            # Collect env vars; override UVICORN_WORKERS=1 for teacher
            env_vars = dict(config.env_vars)
            env_vars["UVICORN_WORKERS"] = "1"
            if self.api_key:
                env_vars["API_KEY"] = self.api_key
            if "task_type" in config.eval_params:
                env_vars["ENV_NAME"] = config.eval_params["task_type"]
            for key in config.required_env_vars:
                val = os.getenv(key, "")
                if val:
                    env_vars[key] = val
            for key in getattr(config, "optional_env_vars", []):
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

            task_id = await self._pick_task_id(env_name, sampling_list)
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
                    key = await asyncio.to_thread(
                        self._upload_rollout, rollout["env"], rollout
                    )
                    logger.info(
                        f"[TEACHER-{worker_id}] Uploaded {key} "
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
        logger.info(
            f"[TEACHER] Starting: model={self.teacher_model} "
            f"envs={self.envs} concurrency={self.concurrency} "
            f"private_bucket={R2_DISTILL_PRIVATE_BUCKET} "
            f"pending_prefix={PENDING_PREFIX}"
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
    """Run teacher worker and mover together as a standalone service."""
    import signal

    from affine.database.client import init_client
    from affine.src.teacher.mover import TeacherMover

    await init_client()

    worker = TeacherWorker(
        teacher_model=teacher_model,
        teacher_base_url=teacher_base_url,
        api_key=api_key,
        envs=envs,
        concurrency=concurrency,
    )
    # Mover self-pauses when DISTILL.enabled_for_sampling is false.
    mover = TeacherMover()

    def handle_shutdown(sig):
        logger.info(f"[TEACHER] Received signal {sig}, shutting down...")
        worker.stop()
        mover.stop()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: handle_shutdown(s))

    tasks = [
        asyncio.create_task(worker.run(), name="teacher-worker"),
        asyncio.create_task(mover.run(), name="teacher-mover"),
    ]

    try:
        await asyncio.gather(*tasks)
    except Exception as e:
        logger.error(f"[TEACHER] Fatal: {e}", exc_info=True)
        raise
    finally:
        worker.stop()
        mover.stop()


@click.command()
@click.option("--model", envvar="TEACHER_MODEL", required=True, help="Teacher model name")
@click.option("--base-url", envvar="TEACHER_BASE_URL", required=True, help="Teacher model API base URL")
@click.option("--api-key", envvar="TEACHER_API_KEY", default="", help="Teacher model API key")
@click.option("--envs", envvar="TEACHER_ENVS", default="CORPUS-EVAL", help="Comma-separated environment list (override via TEACHER_ENVS env var)")
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
