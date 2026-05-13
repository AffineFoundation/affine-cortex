"""
SSH-based inference provider.

Runs sglang directly on an operator-managed host (b300 today) via SSH +
docker. The host is treated as a **single-instance** provider: only one
model loaded at a time, so swapping champion ↔ challenger means
``docker rm -f`` the old container before starting the new one.

The sglang command-line args mirror what
``affine.core.providers.targon_client.create_deployment`` sends to Targon
— same image (``lmsysorg/sglang:latest`` by default), same flags. Only
the orchestration layer differs (SSH + docker run, vs HTTP API).

Configuration is env-driven so operators can swap hosts without touching
code or system_config:

  AFFINE_SSH_PROVIDER_URL              ssh://[user@]host[:port]
  AFFINE_SSH_PROVIDER_KEY_PATH         optional, paramiko key_filename
  AFFINE_SSH_PROVIDER_PUBLIC_URL       full URL exposed to env containers
                                       (defaults to http://<host>:<port>/v1)
  AFFINE_SSH_PROVIDER_PORT             sglang listen port (30000)
  AFFINE_SSH_PROVIDER_DP               data-parallel size (8)
  AFFINE_SSH_PROVIDER_CACHE_DIR        HF cache mount point (/data)
  AFFINE_SSH_PROVIDER_DOCKER_IMAGE     (lmsysorg/sglang:latest)
  AFFINE_SSH_PROVIDER_CONTEXT_LEN      (65536)
  AFFINE_SSH_PROVIDER_MEM_FRACTION     (0.8)
  AFFINE_SSH_PROVIDER_CHUNKED_PREFILL  (4096)
  AFFINE_SSH_PROVIDER_TOOL_CALL_PARSER (qwen, set to 'none' to omit)
  AFFINE_SSH_PROVIDER_READY_TIMEOUT    seconds (1800)
  AFFINE_SSH_PROVIDER_POLL_INTERVAL    seconds between probes (15)
"""

from __future__ import annotations

import asyncio
import os
import shlex
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import aiohttp
import paramiko

from affine.core.setup import logger

from .targon import DeployResult, DeployTarget  # reuse the dataclasses


# All sglang defaults match what targon_client.py sends — same engine,
# same image, same perf knobs. Env vars let operators override per-host.
DEFAULT_DOCKER_IMAGE = os.getenv(
    "AFFINE_SSH_PROVIDER_DOCKER_IMAGE", "lmsysorg/sglang:latest",
)
DEFAULT_CACHE_DIR = os.getenv("AFFINE_SSH_PROVIDER_CACHE_DIR", "/data")
DEFAULT_PORT = int(os.getenv("AFFINE_SSH_PROVIDER_PORT", "30000"))
DEFAULT_DP = int(os.getenv("AFFINE_SSH_PROVIDER_DP", "8"))
DEFAULT_CONTEXT_LEN = os.getenv("AFFINE_SSH_PROVIDER_CONTEXT_LEN", "65536")
DEFAULT_MEM_FRACTION = os.getenv("AFFINE_SSH_PROVIDER_MEM_FRACTION", "0.8")
DEFAULT_CHUNKED_PREFILL = os.getenv("AFFINE_SSH_PROVIDER_CHUNKED_PREFILL", "4096")
DEFAULT_TOOL_CALL_PARSER = os.getenv("AFFINE_SSH_PROVIDER_TOOL_CALL_PARSER", "qwen")
DEFAULT_READY_TIMEOUT_SEC = int(os.getenv("AFFINE_SSH_PROVIDER_READY_TIMEOUT", "1800"))
DEFAULT_POLL_INTERVAL_SEC = float(os.getenv("AFFINE_SSH_PROVIDER_POLL_INTERVAL", "15"))

# Single container name — single-instance host means only one ever exists.
# ``docker rm -f`` is idempotent so start() always kicks off a fresh state.
CONTAINER_NAME = "affine-sglang-current"


@dataclass(frozen=True)
class SSHConfig:
    """Connection params for the remote host. Built from an
    ``inference_endpoints`` row via ``from_endpoint`` (preferred — the
    operator manages SSH targets in DynamoDB) or from env vars via
    ``from_env`` (fallback for first-boot / dev)."""
    host: str
    user: str = "root"
    port: int = 22
    key_path: Optional[str] = None
    public_inference_url: Optional[str] = None
    sglang_port: int = DEFAULT_PORT
    sglang_dp: int = DEFAULT_DP
    sglang_image: str = DEFAULT_DOCKER_IMAGE
    sglang_cache_dir: str = DEFAULT_CACHE_DIR
    connect_timeout: int = 30
    exec_timeout: int = 120

    @staticmethod
    def _parse_ssh_url(url: str) -> Tuple[str, str, int]:
        """``ssh://[user@]host[:port]`` → (user, host, port)."""
        if not url.startswith("ssh://"):
            raise ValueError(f"ssh URL must start with ssh://: {url!r}")
        body = url[len("ssh://"):]
        if "@" in body:
            user, body = body.split("@", 1)
        else:
            user = "root"
        if ":" in body:
            host, port_s = body.split(":", 1)
            port = int(port_s)
        else:
            host = body
            port = 22
        return user, host, port

    @classmethod
    def from_endpoint(cls, endpoint) -> "SSHConfig":
        """Build from an :class:`InferenceEndpointsDAO.Endpoint` row.

        ``endpoint.ssh_url`` is the SSH control plane (where we send
        ``docker run`` commands). ``endpoint.public_inference_url`` is
        the URL env containers actually call (may differ when there's
        an SSH tunnel hop, e.g. workers → val:port → tunnel → b300)."""
        if not endpoint.ssh_url:
            raise ValueError(f"endpoint {endpoint.name!r} has no ssh_url")
        user, host, port = cls._parse_ssh_url(endpoint.ssh_url)
        return cls(
            host=host, user=user, port=port,
            key_path=endpoint.ssh_key_path,
            public_inference_url=endpoint.public_inference_url,
            sglang_port=endpoint.sglang_port,
            sglang_dp=endpoint.sglang_dp,
            sglang_image=endpoint.sglang_image,
            sglang_cache_dir=endpoint.sglang_cache_dir,
        )

    @classmethod
    def from_env(cls) -> "SSHConfig":
        """Fallback: env-var driven config. Used when the
        ``inference_endpoints`` table has no active ssh row."""
        url = os.getenv("AFFINE_SSH_PROVIDER_URL")
        if not url:
            raise RuntimeError(
                "no ssh endpoint registered and "
                "AFFINE_SSH_PROVIDER_URL not set — register one via "
                "``af db set-endpoint`` or set the env var"
            )
        user, host, port = cls._parse_ssh_url(url)
        return cls(
            host=host, user=user, port=port,
            key_path=os.getenv("AFFINE_SSH_PROVIDER_KEY_PATH") or None,
            public_inference_url=os.getenv("AFFINE_SSH_PROVIDER_PUBLIC_URL") or None,
        )

    def inference_url(self) -> str:
        """Where env containers send their OpenAI chat completions."""
        return self.public_inference_url or f"http://{self.host}:{self.sglang_port}/v1"


def _build_sglang_args(target: DeployTarget, config: "SSHConfig") -> List[str]:
    """sglang launch args. Mirrors targon_client.py:382-403 verbatim."""
    args = [
        "--model-path", target.model,
        "--revision", target.revision,
        "--download-dir", config.sglang_cache_dir,
        "--host", "0.0.0.0",
        "--port", str(config.sglang_port),
        "--trust-remote-code",
        "--context-length", DEFAULT_CONTEXT_LEN,
        "--mem-fraction-static", DEFAULT_MEM_FRACTION,
        "--chunked-prefill-size", DEFAULT_CHUNKED_PREFILL,
    ]
    if DEFAULT_TOOL_CALL_PARSER and DEFAULT_TOOL_CALL_PARSER.lower() != "none":
        args += ["--tool-call-parser", DEFAULT_TOOL_CALL_PARSER]
    if config.sglang_dp > 1:
        args += ["--dp", str(config.sglang_dp)]
    return args


def _build_docker_run_cmd(target: DeployTarget, config: "SSHConfig") -> str:
    """Full shell command: rm any existing container, then docker run sglang."""
    sglang_args = " ".join(shlex.quote(str(a)) for a in _build_sglang_args(target, config))

    env_flags = (
        "-e HF_HOME=/data "
        "-e HF_HUB_CACHE=/data "
        "-e TRANSFORMERS_CACHE=/data "
        "-e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 "
    )
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        q = shlex.quote(hf_token)
        env_flags += f"-e HF_TOKEN={q} -e HUGGING_FACE_HUB_TOKEN={q} "

    return (
        f"docker rm -f {CONTAINER_NAME} 2>/dev/null; "
        f"docker run -d --name {CONTAINER_NAME} --gpus all "
        f"-p {config.sglang_port}:{config.sglang_port} "
        f"-v {shlex.quote(config.sglang_cache_dir)}:/data "
        f"{env_flags}"
        f"{shlex.quote(config.sglang_image)} "
        f"python -m sglang.launch_server {sglang_args}"
    )


# ---- low-level SSH (paramiko, sync; wrap in asyncio.to_thread for async) ----


def _ssh_exec_sync(config: SSHConfig, command: str) -> Tuple[int, str, str]:
    """Run ``command`` on ``config.host`` via SSH. Returns (rc, stdout, stderr)."""
    client = paramiko.SSHClient()
    # AutoAddPolicy is acceptable here — operator controls both ends and the
    # SSH endpoint is treated as configured-trust (key auth, fixed host in
    # env config). Production deployments may want a known_hosts file path
    # via a future config knob.
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            hostname=config.host, port=config.port, username=config.user,
            key_filename=config.key_path,
            timeout=config.connect_timeout,
            allow_agent=True, look_for_keys=True,
        )
        stdin, stdout, stderr = client.exec_command(
            command, timeout=config.exec_timeout,
        )
        rc = stdout.channel.recv_exit_status()
        out = stdout.read().decode("utf-8", errors="replace").strip()
        err = stderr.read().decode("utf-8", errors="replace").strip()
        return rc, out, err
    finally:
        client.close()


async def _ssh_exec(config: SSHConfig, command: str) -> Tuple[int, str, str]:
    """Async wrapper around the blocking paramiko call."""
    return await asyncio.to_thread(_ssh_exec_sync, config, command)


# ---- ready probe -----------------------------------------------------------


async def _probe_ready(base_url: str, *, timeout_sec: float = 5.0) -> bool:
    """True iff ``GET base_url/models`` returns 200 with non-empty ``data``."""
    url = base_url.rstrip("/") + "/models"
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout_sec)
        ) as sess:
            async with sess.get(url) as resp:
                if resp.status != 200:
                    return False
                payload = await resp.json(content_type=None)
    except Exception as e:
        logger.debug(f"ssh-provider: probe {url}: {type(e).__name__}: {e}")
        return False
    if not isinstance(payload, dict):
        return False
    data = payload.get("data")
    return isinstance(data, list) and len(data) > 0


async def _wait_ready(
    base_url: str, *,
    deadline_sec: int = DEFAULT_READY_TIMEOUT_SEC,
    poll_interval_sec: float = DEFAULT_POLL_INTERVAL_SEC,
) -> None:
    """Block until /v1/models reports the loaded model, or timeout."""
    deadline = time.monotonic() + deadline_sec
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        if await _probe_ready(base_url):
            logger.info(
                f"ssh-provider: model ready after {attempt} probes ({base_url})"
            )
            return
        await asyncio.sleep(poll_interval_sec)
    raise TimeoutError(
        f"ssh-provider: {base_url} did not become ready in {deadline_sec}s"
    )


# ---- public API: deploy / teardown -----------------------------------------


async def deploy(config: SSHConfig, target: DeployTarget) -> DeployResult:
    """Swap the remote host onto ``target``'s model.

    Steps:
      1. SSH in
      2. ``docker rm -f <CONTAINER_NAME>`` (idempotent — handles any stale
         container left from a previous deploy / crashed teardown)
      3. ``docker run -d --gpus all ... lmsysorg/sglang:latest python -m
         sglang.launch_server ...``
      4. Poll ``http://<host>:<port>/v1/models`` until ready

    On success returns ``DeployResult(deployment_id=CONTAINER_NAME, base_url=...)``.
    """
    cmd = _build_docker_run_cmd(target, config)
    logger.info(
        f"ssh-provider: deploying {target.model}@{target.revision[:8]} "
        f"on {config.host}:{config.port}"
    )
    rc, out, err = await _ssh_exec(config, cmd)
    if rc != 0:
        raise RuntimeError(
            f"ssh-provider: docker run failed on {config.host}: "
            f"rc={rc} stderr={err!r}"
        )
    if out:
        logger.info(f"ssh-provider: container id={out[:12]}...")

    base_url = config.inference_url()
    await _wait_ready(base_url)
    return DeployResult(deployment_id=CONTAINER_NAME, base_url=base_url)


async def teardown(config: SSHConfig, deployment_id: Optional[str]) -> None:
    """Stop and remove the sglang container on the remote host.

    Idempotent — ``docker rm -f`` swallows the "not found" case. We
    always target ``CONTAINER_NAME`` regardless of ``deployment_id``
    because the host is single-instance: at most one container exists,
    and that container's identity changes each deploy.
    """
    cmd = f"docker rm -f {CONTAINER_NAME} 2>/dev/null || true"
    try:
        rc, out, err = await _ssh_exec(config, cmd)
        if rc != 0:
            logger.warning(
                f"ssh-provider: teardown rc={rc} stderr={err!r}"
            )
    except Exception as e:
        logger.warning(
            f"ssh-provider: teardown raised {type(e).__name__}: {e}"
        )
