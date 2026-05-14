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

Endpoint configuration is DB-driven via the ``inference_endpoints`` table:

  ssh_url                 ssh://[user@]host[:port]
  ssh_key_path            optional, paramiko key_filename
  public_inference_url    full URL exposed to env containers
                          (defaults to http://<host>:<sglang_port>/v1)
  sglang_port             sglang listen port (30000)
  sglang_dp               data-parallel size (8)
  sglang_cache_dir        HF cache mount point (/data)
  sglang_image            (lmsysorg/sglang:latest)
  sglang_context_len      legacy field; deployment no longer passes --context-length
  sglang_mem_fraction     GPU memory fraction passed to sglang (0.85)
  sglang_chunked_prefill  chunked-prefill size (4096)
  sglang_tool_call_parser parser name, "none" to omit (qwen)
  ready_timeout_sec       seconds to wait for /v1/models (1800)
  poll_interval_sec       seconds between readiness probes (15)
"""

from __future__ import annotations

import asyncio
import os
import shlex
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import aiohttp
import paramiko

from affine.core.setup import logger

from .targon import DeployResult, DeployTarget, MachineDeployment  # reuse the dataclasses


# All sglang launch defaults match what targon_client.py sends. Endpoint-
# specific values come from the inference_endpoints table, not environment
# variables.
DEFAULT_DOCKER_IMAGE = "lmsysorg/sglang:latest"
DEFAULT_CACHE_DIR = "/data"
DEFAULT_PORT = 30000
DEFAULT_DP = 8
DEFAULT_CONTEXT_LEN = 65536
DEFAULT_MEM_FRACTION = 0.85
DEFAULT_CHUNKED_PREFILL = 4096
DEFAULT_TOOL_CALL_PARSER = "qwen"
DEFAULT_READY_TIMEOUT_SEC = 1800
DEFAULT_POLL_INTERVAL_SEC = 15.0

# Single container name — single-instance host means only one ever exists.
# ``docker rm -f`` is idempotent so start() always kicks off a fresh state.
CONTAINER_NAME = "affine-sglang-current"


@dataclass(frozen=True)
class SSHConfig:
    """Connection params for the remote host. Built from an
    ``inference_endpoints`` row via ``from_endpoint``. The operator
    manages SSH targets in DynamoDB so scheduler restarts do not depend
    on host-specific env vars."""
    host: str
    endpoint_name: str = ""
    user: str = "root"
    port: int = 22
    key_path: Optional[str] = None
    public_inference_url: Optional[str] = None
    sglang_port: int = DEFAULT_PORT
    sglang_dp: int = DEFAULT_DP
    sglang_image: str = DEFAULT_DOCKER_IMAGE
    sglang_cache_dir: str = DEFAULT_CACHE_DIR
    sglang_context_len: int = DEFAULT_CONTEXT_LEN
    sglang_mem_fraction: float = DEFAULT_MEM_FRACTION
    sglang_chunked_prefill: int = DEFAULT_CHUNKED_PREFILL
    sglang_tool_call_parser: str = DEFAULT_TOOL_CALL_PARSER
    ready_timeout_sec: int = DEFAULT_READY_TIMEOUT_SEC
    poll_interval_sec: float = DEFAULT_POLL_INTERVAL_SEC
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
            host=host, endpoint_name=endpoint.name, user=user, port=port,
            key_path=endpoint.ssh_key_path,
            public_inference_url=endpoint.public_inference_url,
            sglang_port=endpoint.sglang_port,
            sglang_dp=endpoint.sglang_dp,
            sglang_image=endpoint.sglang_image,
            sglang_cache_dir=endpoint.sglang_cache_dir,
            sglang_context_len=endpoint.sglang_context_len,
            sglang_mem_fraction=endpoint.sglang_mem_fraction,
            sglang_chunked_prefill=endpoint.sglang_chunked_prefill,
            sglang_tool_call_parser=endpoint.sglang_tool_call_parser,
            ready_timeout_sec=endpoint.ready_timeout_sec,
            poll_interval_sec=endpoint.poll_interval_sec,
        )

    def inference_url(self) -> str:
        """Where env containers send their OpenAI chat completions."""
        return self.public_inference_url or f"http://{self.host}:{self.sglang_port}/v1"

    def deployment_id(self) -> str:
        """Stable identifier persisted in scheduler state.

        Include endpoint name so a future multi-endpoint scheduler can tell
        which machine owns the single-instance container.
        """
        endpoint = self.endpoint_name or self.host
        return f"ssh:{endpoint}:{CONTAINER_NAME}"


def _build_sglang_args(target: DeployTarget, config: "SSHConfig") -> List[str]:
    """sglang launch args. Mirrors targon_client.py:382-403 verbatim."""
    args = [
        "--model-path", target.model,
        "--revision", target.revision,
        "--download-dir", config.sglang_cache_dir,
        "--host", "0.0.0.0",
        "--port", str(config.sglang_port),
        "--trust-remote-code",
        "--mem-fraction-static", str(config.sglang_mem_fraction),
        "--chunked-prefill-size", str(config.sglang_chunked_prefill),
    ]
    parser = config.sglang_tool_call_parser
    if parser and parser.lower() != "none":
        args += ["--tool-call-parser", parser]
    if config.sglang_dp > 1:
        args += ["--dp", str(config.sglang_dp)]
    return args


def _build_docker_run_cmd(target: DeployTarget, config: "SSHConfig") -> str:
    """Full shell command: rm any existing container, then docker run sglang."""
    sglang_args = " ".join(shlex.quote(str(a)) for a in _build_sglang_args(target, config))
    labels = {
        "io.affine.endpoint": config.endpoint_name or config.host,
        "io.affine.uid": str(target.uid),
        "io.affine.hotkey": target.hotkey,
        "io.affine.model": target.model,
        "io.affine.revision": target.revision,
    }
    label_flags = " ".join(
        f"--label {shlex.quote(k)}={shlex.quote(v)}"
        for k, v in labels.items()
    )

    env_flags = (
        "-e HF_HOME=/data "
        "-e HF_HUB_CACHE=/data "
        "-e TRANSFORMERS_CACHE=/data "
    )
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        q = shlex.quote(hf_token)
        env_flags += f"-e HF_TOKEN={q} -e HUGGING_FACE_HUB_TOKEN={q} "

    # --network host: avoids docker port mapping which triggers the
    #   "unsafe procfs detected: ip_unprivileged_port_start" OCI error on
    #   newer kernels (docker 29.x + cgroup v2). sglang binds 0.0.0.0:<port>
    #   on the host directly.
    # --ipc=host + --shm-size=32g: required for sglang multi-process
    #   (--dp/--tp) so the DP workers can share CUDA IPC handles and the
    #   tokenizer cache. The default 64 MiB /dev/shm is far too small.
    # --security-opt label=disable: matches the working production reference
    #   containers; needed when SELinux relabeling on the bind mount would
    #   prevent the container from reading /data.
    return (
        f"docker rm -f {CONTAINER_NAME} 2>/dev/null; "
        f"docker run -d --name {CONTAINER_NAME} --gpus all "
        f"--network host --ipc=host --shm-size=32g "
        f"--security-opt label=disable "
        f"{label_flags} "
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
    # endpoint config). Production deployments may want a known_hosts file path
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


def _parse_label_dump(text: str) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        labels[key.strip()] = value.strip()
    return labels


def _expected_labels(config: SSHConfig, target: DeployTarget) -> Dict[str, str]:
    return {
        "io.affine.endpoint": config.endpoint_name or config.host,
        "io.affine.uid": str(target.uid),
        "io.affine.hotkey": target.hotkey,
        "io.affine.model": target.model,
        "io.affine.revision": target.revision,
    }


async def _existing_container_matches(
    config: SSHConfig, target: DeployTarget,
) -> bool:
    """Return True when the current endpoint already serves ``target``.

    The scheduler persists assignment in system_config, but crash recovery
    can land after docker run and before state write. Labels on the remote
    container are the machine-local truth that lets the next scheduler adopt
    the workload instead of restarting the model.
    """
    fmt = "{{range $k,$v := .Config.Labels}}{{println $k \"=\" $v}}{{end}}"
    cmd = f"docker inspect --format {shlex.quote(fmt)} {CONTAINER_NAME}"
    rc, out, _ = await _ssh_exec(config, cmd)
    if rc != 0:
        return False
    labels = _parse_label_dump(out)
    expected = _expected_labels(config, target)
    return all(labels.get(k) == v for k, v in expected.items())


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

    On success returns ``DeployResult(deployment_id="ssh:<endpoint>:...", base_url=...)``.
    """
    base_url = config.inference_url()
    if await _existing_container_matches(config, target):
        await _wait_ready(
            base_url,
            deadline_sec=config.ready_timeout_sec,
            poll_interval_sec=config.poll_interval_sec,
        )
        logger.info(
            f"ssh-provider: adopted existing {config.deployment_id()} for "
            f"{target.model}@{target.revision[:8]} on "
            f"{config.endpoint_name or config.host}"
        )
        return DeployResult(
            deployment_id=config.deployment_id(),
            base_url=base_url,
            deployments=[
                MachineDeployment(
                    endpoint_name=config.endpoint_name,
                    deployment_id=config.deployment_id(),
                    base_url=base_url,
                )
            ],
        )

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

    await _wait_ready(
        base_url,
        deadline_sec=config.ready_timeout_sec,
        poll_interval_sec=config.poll_interval_sec,
    )
    return DeployResult(
        deployment_id=config.deployment_id(),
        base_url=base_url,
        deployments=[
            MachineDeployment(
                endpoint_name=config.endpoint_name,
                deployment_id=config.deployment_id(),
                base_url=base_url,
            )
        ],
    )


async def teardown(config: SSHConfig, deployment_id: Optional[str]) -> None:
    """Stop and remove the sglang container on the remote host.

    Idempotent — ``docker rm -f`` swallows the "not found" case. We
    only act when ``deployment_id`` belongs to this endpoint. That keeps
    endpoint ownership explicit once multiple SSH machines are active.
    """
    if deployment_id and deployment_id != config.deployment_id():
        logger.info(
            f"ssh-provider: skip teardown for deployment_id={deployment_id!r}; "
            f"this endpoint owns {config.deployment_id()!r}"
        )
        return
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
