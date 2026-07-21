"""
SSH-based inference provider.

Runs sglang directly on an operator-managed host (b300 today) via SSH +
docker. The host is treated as a **single-instance** provider: only one
model loaded at a time, so swapping champion ↔ challenger means
``docker rm -f`` the old container before starting the new one.

The sglang command-line args mirror what
``affine.core.providers.targon_client.create_deployment`` sends to Targon
— same image (``lmsysorg/sglang:v0.5.14`` by default), same flags. Only
the orchestration layer differs (SSH + docker run, vs HTTP API).

Endpoint configuration is DB-driven via the ``inference_endpoints`` table:

  ssh_url                 ssh://[user@]host[:port]
  ssh_key_path            optional, paramiko key_filename
  public_inference_url    full URL exposed to env containers
                          (defaults to http://<host>:<sglang_port>/v1)
  sglang_port             sglang listen port (default 10001)
  sglang_dp               data-parallel size (8)
  sglang_load_balance_method DP request routing strategy (total_tokens)
  sglang_cache_dir        HF cache mount point (/data)
  sglang_image            (lmsysorg/sglang:v0.5.14)
  sglang_context_len      legacy field; deployment no longer passes --context-length
  sglang_mem_fraction     GPU memory fraction passed to sglang (0.85)
  sglang_chunked_prefill  chunked-prefill size (4096)
  sglang_tool_call_parser parser name, "none" to omit (qwen)
  sglang_docker_args      optional extra docker-run args
  serving_mode            unified (this module) or pd (ssh_pd module)
  sglang_pd_*             4P4D worker/Gateway configuration
  ready_timeout_sec       seconds to wait for /v1/models (1800)
  poll_interval_sec       seconds between readiness probes (15)
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import paramiko
from huggingface_hub import HfApi

from affine.core.providers.targon_client import (
    QWEN36_REASONING_PARSER,
    QWEN36_TOOL_CALL_PARSER,
    is_qwen36,
)
from affine.core.sglang_runtime import (
    DEFAULT_SGLANG_IMAGE,
    DEFAULT_SGLANG_SERVER_ARGS,
)
from affine.core.setup import logger

from .flow import TransientDeployError
from .health import DeploymentHealthResult, DeploymentHealthState
from .targon import DeployResult, DeployTarget, MachineDeployment  # reuse the dataclasses


# Exceptions ``_ssh_exec`` re-raises as :class:`TransientDeployError` so
# the scheduler can tell "host's fault" apart from miner-fault deploy
# failures. ``OSError`` covers ``socket.error`` / ``socket.timeout`` /
# connection-refused / network-unreachable in Python 3. Paramiko's
# ``SSHException`` is the umbrella for auth + protocol + transport
# errors; we still re-raise these as transient because for the
# scheduler's purpose, the miner doesn't deserve a FAILED stamp from
# any of them.
_SSH_TRANSPORT_EXCEPTIONS = (
    paramiko.SSHException,
    OSError,
    EOFError,
)


# All sglang launch defaults match what targon_client.py sends. Endpoint-
# specific values come from the inference_endpoints table, not environment
# variables.
DEFAULT_DOCKER_IMAGE = DEFAULT_SGLANG_IMAGE
DEFAULT_CACHE_DIR = "/data"
DEFAULT_PORT = 10001
DEFAULT_DP = 8
DEFAULT_LOAD_BALANCE_METHOD = "total_tokens"
SUPPORTED_LOAD_BALANCE_METHODS = (
    "auto",
    "round_robin",
    "total_requests",
    "total_tokens",
)
DEFAULT_CONTEXT_LEN = 65536
DEFAULT_MEM_FRACTION = 0.85
DEFAULT_CHUNKED_PREFILL = 4096
DEFAULT_TOOL_CALL_PARSER = "qwen"
DEFAULT_READY_TIMEOUT_SEC = 1800
DEFAULT_POLL_INTERVAL_SEC = 15.0
DEFAULT_HF_METADATA_TIMEOUT_SEC = 30.0
HF_XET_MAX_NON_XET_FILE_BYTES = 20 * 1024 ** 3
HF_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt")
ACTIVE_HF_INCOMPLETE_CACHE_MAX_AGE_SECONDS = 6 * 60 * 60

# Single container name — single-instance host means only one ever exists.
# ``docker rm -f`` is idempotent so start() always kicks off a fresh state.
CONTAINER_NAME = "affine-sglang-current"
PD_COMPAT_CONTAINER_NAMES = (
    "affine-sglang-pd-gateway",
    *(f"affine-sglang-pd-decode-{index}" for index in range(4)),
    *(f"affine-sglang-pd-prefill-{index}" for index in range(4)),
)
RESTART_POLICY = "unless-stopped"

# HuggingFace snapshot cache layout: model "<owner>/<name>" lives at
# ``<cache>/models--<owner>--<name>/``. Both the prefix and the org/name
# separator are HF conventions — centralize so the cleanup glob, the
# Python-side dir name builder, and any future code stay in sync.
#
# ``HF_SNAPSHOT_PREFIX`` MUST be non-empty: the cleanup loop globs
# ``<cache>/<prefix>*/`` to scope deletion, and an empty prefix would
# match every directory under ``<cache>``. Assertion below makes that
# class of mistake impossible at module-load time rather than at deploy.
HF_SNAPSHOT_PREFIX = "models--"
HF_ORG_SEPARATOR = "--"
assert HF_SNAPSHOT_PREFIX, "HF_SNAPSHOT_PREFIX must be non-empty"


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
    sglang_load_balance_method: str = DEFAULT_LOAD_BALANCE_METHOD
    sglang_image: str = DEFAULT_DOCKER_IMAGE
    sglang_cache_dir: str = DEFAULT_CACHE_DIR
    sglang_context_len: int = DEFAULT_CONTEXT_LEN
    sglang_mem_fraction: float = DEFAULT_MEM_FRACTION
    sglang_chunked_prefill: int = DEFAULT_CHUNKED_PREFILL
    sglang_tool_call_parser: str = DEFAULT_TOOL_CALL_PARSER
    sglang_docker_args: Tuple[str, ...] = ()
    serving_mode: str = "unified"
    sglang_pd_prefill_replicas: int = 4
    sglang_pd_decode_replicas: int = 4
    sglang_pd_prefill_port_start: int = 11000
    sglang_pd_bootstrap_port_start: int = 12000
    sglang_pd_decode_port_start: int = 13000
    sglang_pd_transfer_backend: str = "mooncake"
    sglang_pd_ib_device: Optional[str] = None
    sglang_pd_gateway_image: str = ""
    sglang_pd_prefill_policy: str = "cache_aware"
    sglang_pd_decode_policy: str = "power_of_two"
    ready_timeout_sec: int = DEFAULT_READY_TIMEOUT_SEC
    poll_interval_sec: float = DEFAULT_POLL_INTERVAL_SEC
    connect_timeout: int = 30
    exec_timeout: int = 120
    autoscale_managed: bool = False
    autoscale_instance_id: Optional[str] = None
    endpoint_generation: int = 0

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
            sglang_load_balance_method=getattr(
                endpoint,
                "sglang_load_balance_method",
                DEFAULT_LOAD_BALANCE_METHOD,
            ),
            sglang_image=endpoint.sglang_image,
            sglang_cache_dir=endpoint.sglang_cache_dir,
            sglang_context_len=endpoint.sglang_context_len,
            sglang_mem_fraction=endpoint.sglang_mem_fraction,
            sglang_chunked_prefill=endpoint.sglang_chunked_prefill,
            sglang_tool_call_parser=endpoint.sglang_tool_call_parser,
            sglang_docker_args=_coerce_docker_args(
                getattr(endpoint, "sglang_docker_args", None)
            ),
            serving_mode=getattr(endpoint, "serving_mode", "unified"),
            sglang_pd_prefill_replicas=getattr(
                endpoint, "sglang_pd_prefill_replicas", 4,
            ),
            sglang_pd_decode_replicas=getattr(
                endpoint, "sglang_pd_decode_replicas", 4,
            ),
            sglang_pd_prefill_port_start=getattr(
                endpoint, "sglang_pd_prefill_port_start", 11000,
            ),
            sglang_pd_bootstrap_port_start=getattr(
                endpoint, "sglang_pd_bootstrap_port_start", 12000,
            ),
            sglang_pd_decode_port_start=getattr(
                endpoint, "sglang_pd_decode_port_start", 13000,
            ),
            sglang_pd_transfer_backend=getattr(
                endpoint, "sglang_pd_transfer_backend", "mooncake",
            ),
            sglang_pd_ib_device=getattr(
                endpoint, "sglang_pd_ib_device", None,
            ),
            sglang_pd_gateway_image=getattr(
                endpoint, "sglang_pd_gateway_image", "",
            ),
            sglang_pd_prefill_policy=getattr(
                endpoint, "sglang_pd_prefill_policy", "cache_aware",
            ),
            sglang_pd_decode_policy=getattr(
                endpoint, "sglang_pd_decode_policy", "power_of_two",
            ),
            ready_timeout_sec=endpoint.ready_timeout_sec,
            poll_interval_sec=endpoint.poll_interval_sec,
            autoscale_managed=bool(
                getattr(endpoint, "autoscale_managed", False)
            ),
            autoscale_instance_id=getattr(
                endpoint, "autoscale_instance_id", None
            ),
            endpoint_generation=int(getattr(endpoint, "generation", 0) or 0),
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
        container_name = (
            "affine-sglang-pd-gateway"
            if self.serving_mode == "pd"
            else CONTAINER_NAME
        )
        return f"ssh:{endpoint}:{container_name}"

    def health_identity(self) -> str:
        runtime = self.autoscale_instance_id or f"{self.host}:{self.port}"
        return (
            f"{self.endpoint_name or self.host}:{runtime}:"
            f"generation={self.endpoint_generation}"
        )


def _coerce_docker_args(value: Any) -> Tuple[str, ...]:
    if not value:
        return ()
    if isinstance(value, str):
        return tuple(shlex.split(value))
    if isinstance(value, (list, tuple)):
        return tuple(str(arg) for arg in value if str(arg))
    raise ValueError(
        "sglang_docker_args must be a string or list, "
        f"got {type(value).__name__}"
    )


def _build_sglang_args(target: DeployTarget, config: "SSHConfig") -> List[str]:
    """sglang launch args. Mirrors targon_client.create_deployment."""
    args = [
        "--model-path", target.model,
        "--revision", target.revision,
        "--download-dir", config.sglang_cache_dir,
        "--host", "0.0.0.0",
        "--port", str(config.sglang_port),
        "--trust-remote-code",
        "--mem-fraction-static", str(config.sglang_mem_fraction),
        "--chunked-prefill-size", str(config.sglang_chunked_prefill),
        *DEFAULT_SGLANG_SERVER_ARGS,
    ]
    parser = config.sglang_tool_call_parser
    # Qwen3.6 (qwen3_5_moe) needs a reasoning parser and the qwen3_coder
    # tool-call schema, or it serves unparsed <think> blocks and broken tool
    # calls. Dense qwen3 and everything else keep the legacy flags.
    if is_qwen36(target.model_type):
        args += ["--reasoning-parser", QWEN36_REASONING_PARSER]
        parser = QWEN36_TOOL_CALL_PARSER
    if parser and parser.lower() != "none":
        args += ["--tool-call-parser", parser]
    if config.sglang_dp > 1:
        load_balance_method = str(
            config.sglang_load_balance_method or ""
        ).strip().lower()
        if load_balance_method not in SUPPORTED_LOAD_BALANCE_METHODS:
            supported = ", ".join(SUPPORTED_LOAD_BALANCE_METHODS)
            raise ValueError(
                "unsupported SGLang load-balance method "
                f"{config.sglang_load_balance_method!r}; expected one of: "
                f"{supported}"
            )
        args += [
            "--dp", str(config.sglang_dp),
            "--load-balance-method", load_balance_method,
        ]
    return args


def _strip_env_from_docker_args(
    args: Tuple[str, ...], env_name: str,
) -> Tuple[str, ...]:
    """Drop ``-e NAME=...`` from operator args so deploy policy owns it."""
    out: List[str] = []
    i = 0
    while i < len(args):
        arg = str(args[i])
        if (
            arg in ("-e", "--env")
            and i + 1 < len(args)
            and str(args[i + 1]).startswith(f"{env_name}=")
        ):
            i += 2
            continue
        if arg.startswith("--env=") and arg[len("--env="):].startswith(
            f"{env_name}="
        ):
            i += 1
            continue
        if arg.startswith("-e") and arg != "-e" and arg[len("-e"):].startswith(
            f"{env_name}="
        ):
            i += 1
            continue
        out.append(arg)
        i += 1
    return tuple(out)


def _max_weight_file_size_bytes(siblings: Any) -> Optional[int]:
    sizes: List[int] = []
    for sibling in siblings or []:
        name = str(
            getattr(sibling, "rfilename", None)
            or getattr(sibling, "filename", None)
            or ""
        )
        if not name.endswith(HF_WEIGHT_SUFFIXES):
            continue
        size = getattr(sibling, "size", None)
        if size is None:
            continue
        try:
            sizes.append(int(size))
        except (TypeError, ValueError):
            continue
    return max(sizes) if sizes else None


def _hf_metadata_timeout_sec() -> float:
    raw = os.getenv("AFFINE_SSH_HF_METADATA_TIMEOUT_SEC", "")
    if not raw:
        return DEFAULT_HF_METADATA_TIMEOUT_SEC
    try:
        return max(1.0, float(raw))
    except ValueError:
        logger.warning(
            "ssh-provider: invalid AFFINE_SSH_HF_METADATA_TIMEOUT_SEC=%r; "
            "using %.1fs",
            raw,
            DEFAULT_HF_METADATA_TIMEOUT_SEC,
        )
        return DEFAULT_HF_METADATA_TIMEOUT_SEC


async def _should_disable_hf_xet(target: DeployTarget) -> bool:
    """Disable Xet only when the largest weight shard is below 50 GiB.

    If metadata is unavailable, keep Xet enabled. That is the conservative
    path for very large single-file weights, and avoids turning an inspection
    hiccup into a deploy policy change.
    """
    hf_token = os.getenv("HF_TOKEN")
    timeout_sec = _hf_metadata_timeout_sec()

    def _fetch_model_info():
        return HfApi(token=hf_token).model_info(
            repo_id=target.model,
            revision=target.revision,
            files_metadata=True,
            timeout=timeout_sec,
        )

    try:
        info = await asyncio.wait_for(
            asyncio.to_thread(_fetch_model_info),
            timeout=timeout_sec + 1.0,
        )
    except Exception as e:
        logger.warning(
            "ssh-provider: HF metadata lookup failed for %s@%s; "
            "leaving Xet enabled: %s: %s",
            target.model,
            target.revision[:8],
            type(e).__name__,
            e,
        )
        return False

    max_size = _max_weight_file_size_bytes(getattr(info, "siblings", None))
    if max_size is None:
        logger.warning(
            "ssh-provider: no HF weight file sizes found for %s@%s; "
            "leaving Xet enabled",
            target.model,
            target.revision[:8],
        )
        return False
    disable = max_size < HF_XET_MAX_NON_XET_FILE_BYTES
    logger.info(
        "ssh-provider: largest HF weight file for %s@%s is %.2f GiB; "
        "HF Xet %s",
        target.model,
        target.revision[:8],
        max_size / (1024 ** 3),
        "disabled" if disable else "enabled",
    )
    return disable


def _build_docker_run_cmd(
    target: DeployTarget,
    config: "SSHConfig",
    *,
    disable_hf_xet: bool = False,
) -> str:
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
    docker_args = _strip_env_from_docker_args(
        config.sglang_docker_args, "HF_HUB_DISABLE_XET",
    )
    extra_docker_flags = " ".join(
        shlex.quote(str(arg)) for arg in docker_args
    )

    env_flags = (
        "-e HF_HOME=/data "
        "-e HF_HUB_CACHE=/data "
        "-e TRANSFORMERS_CACHE=/data "
        "-e HF_HUB_DOWNLOAD_TIMEOUT=60 "
    )
    if disable_hf_xet:
        env_flags += "-e HF_HUB_DISABLE_XET=1 "
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
    stale_containers = " ".join((CONTAINER_NAME, *PD_COMPAT_CONTAINER_NAMES))
    return (
        f"docker rm -f {stale_containers} 2>/dev/null; "
        f"docker run -d --name {CONTAINER_NAME} --gpus all "
        f"--restart {RESTART_POLICY} "
        f"--network host --ipc=host --shm-size=32g "
        f"--security-opt label=disable "
        f"{extra_docker_flags} "
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
    """Async wrapper around the blocking paramiko call.

    Transport-level failures (SSH connect/timeout/network errors) are
    re-raised as :class:`TransientDeployError` so the flow scheduler
    can distinguish them from miner-fault deploy failures and skip the
    miner without stamping FAILED on a perfectly good model that just
    landed on a dead host."""
    try:
        return await asyncio.to_thread(_ssh_exec_sync, config, command)
    except _SSH_TRANSPORT_EXCEPTIONS as e:
        raise TransientDeployError(
            f"ssh-provider: transport error on {config.host}:{config.port}: "
            f"{type(e).__name__}: {e}"
        ) from e


async def probe_host(config: SSHConfig) -> int:
    """Verify that a static SSH endpoint can run GPU Docker workloads.

    This is intentionally read-only: registration should validate the same
    SSH, Docker, and NVIDIA prerequisites the scheduler needs without pulling
    an image or starting a container.
    """
    command = "docker info >/dev/null 2>&1 && nvidia-smi -L"
    rc, out, err = await _ssh_exec(config, command)
    if rc != 0:
        detail = err or out or "preflight command failed"
        raise RuntimeError(
            f"static endpoint preflight failed on {config.host}:{config.port}: "
            f"rc={rc} {detail}"
        )
    gpu_count = sum(
        1 for line in out.splitlines()
        if line.strip().lower().startswith("gpu ")
    )
    if gpu_count <= 0:
        raise RuntimeError(
            f"static endpoint preflight found no NVIDIA GPUs on "
            f"{config.host}:{config.port}"
        )
    return gpu_count


def _hf_cache_dir_name(model: str) -> str:
    """HuggingFace's snapshot dir convention: ``<prefix><owner><sep><name>``
    (the org separator ``/`` is flattened to ``--``)."""
    return f"{HF_SNAPSHOT_PREFIX}{model.replace('/', HF_ORG_SEPARATOR)}"


# Shell snippet that nukes every ``<cache>/<prefix>*`` directory except
# the new target's and recently active incomplete downloads. Run
# pre-deploy so b300 doesn't fill its disk with stale weights from every
# past challenger, while preserving HF's resumable download state across
# transient ready timeouts.
_CLEANUP_SCRIPT = r"""
TARGET_DIR={target_dir_q}
CACHE_DIR={cache_dir_q}
PREFIX={prefix_q}
ACTIVE_INCOMPLETE_MAX_AGE_MINUTES={active_incomplete_max_age_minutes_q}
for d in "$CACHE_DIR"/"$PREFIX"*/; do
    [ -d "$d" ] || continue
    name=$(basename "$d")
    [ "$name" = "$TARGET_DIR" ] && continue
    active_incomplete=$(find "$d" -type f -name '*.incomplete' -mmin "-$ACTIVE_INCOMPLETE_MAX_AGE_MINUTES" -print -quit 2>/dev/null)
    if [ -n "$active_incomplete" ]; then
        echo "kept-active: $name"
        continue
    fi
    echo "removed: $name"
    rm -rf "$d"
done
"""


def _build_cache_cleanup_cmd(target: DeployTarget, config: SSHConfig) -> str:
    """Return the shell snippet that purges stale HF caches before a deploy."""
    active_incomplete_max_age_minutes = max(
        1, (ACTIVE_HF_INCOMPLETE_CACHE_MAX_AGE_SECONDS + 59) // 60
    )
    return _CLEANUP_SCRIPT.format(
        target_dir_q=shlex.quote(_hf_cache_dir_name(target.model)),
        cache_dir_q=shlex.quote(config.sglang_cache_dir),
        prefix_q=shlex.quote(HF_SNAPSHOT_PREFIX),
        active_incomplete_max_age_minutes_q=shlex.quote(
            str(active_incomplete_max_age_minutes)
        ),
    )


def _is_valid_hf_model_id(model: str) -> bool:
    """A valid HuggingFace model id is ``<owner>/<name>`` with both
    parts non-empty. The cleanup loop's keep-check is by exact dir
    name match — a malformed id would translate to a target dir name
    that matches *zero* dirs on disk, and the loop would then delete
    every other ``models--*`` dir (including the currently-serving
    model). Guard against that by refusing to run cleanup when the
    id doesn't look like a real HF model."""
    if not isinstance(model, str) or not model:
        return False
    parts = model.split("/")
    if len(parts) != 2:
        return False
    owner, name = parts
    return bool(owner) and bool(name)


async def _cleanup_stale_caches(
    config: SSHConfig, target: DeployTarget,
) -> None:
    """Delete stale ``<prefix>*`` dirs on the remote host.

    Idempotent. Failures are logged but non-fatal — the deploy will
    still try to ``docker run``; sglang itself will surface
    out-of-disk later if the cleanup didn't free enough.

    Refuses to run when ``target.model`` is not a valid HuggingFace
    id (``<owner>/<name>``). A bogus id would produce a keep-dir name
    that matches no real cache entry, which would cause every other
    model dir — including the currently-serving model — to be wiped.
    Better to leave the cache alone and let the upstream HF download
    fail loudly than to silently nuke disk on a malformed commit."""
    if not _is_valid_hf_model_id(target.model):
        logger.warning(
            f"ssh-provider: skipping cache cleanup; "
            f"target.model={target.model!r} is not a valid HF id"
        )
        return
    cmd = _build_cache_cleanup_cmd(target, config)
    try:
        rc, out, err = await _ssh_exec(config, cmd)
    except Exception as e:
        logger.warning(
            f"ssh-provider: cache cleanup raised {type(e).__name__}: {e}"
        )
        return
    if rc != 0:
        logger.warning(
            f"ssh-provider: cache cleanup rc={rc} stderr={err!r}"
        )
        return
    for line in (out or "").splitlines():
        if line.startswith("removed:"):
            logger.info(f"ssh-provider: stale cache {line}")
        elif line.startswith("kept-active:"):
            logger.info(f"ssh-provider: active download cache {line}")


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


async def _container_exit_status(config: "SSHConfig") -> Optional[int]:
    """``int`` exit code if the sglang container is no longer running,
    ``None`` if it's running or absent (``docker inspect`` failure)."""
    fmt = "{{.State.Status}} {{.State.ExitCode}}"
    cmd = f"docker inspect --format {shlex.quote(fmt)} {CONTAINER_NAME}"
    rc, out, _ = await _ssh_exec(config, cmd)
    if rc != 0:
        return None
    return _parse_container_exit_status(out)


def _parse_container_exit_status(output: str) -> Optional[int]:
    """Parse ``docker inspect`` status from noisy SSH stdout.

    Targon's SSH wrapper prefixes stdout with lines such as
    ``Connecting to container ...``. Scan from the bottom so those banners don't
    hide the actual ``running 0`` / ``exited 1`` line.
    """
    for line in reversed((output or "").splitlines()):
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        status, exit_code_s = parts
        if status not in {
            "created", "restarting", "running", "removing",
            "paused", "exited", "dead",
        }:
            continue
        if status == "running":
            return None
        try:
            return int(exit_code_s)
        except ValueError:
            return None
    return None


def _parse_container_inspect_json(output: str) -> Optional[Dict[str, Any]]:
    """Parse ``docker inspect --format '{{json .}}'`` from noisy stdout."""
    for line in reversed((output or "").splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


class _ContainerNotFoundError(RuntimeError):
    pass


class _ContainerInspectUnavailableError(RuntimeError):
    pass


def _docker_object_not_found(*outputs: str) -> bool:
    text = "\n".join(outputs).lower()
    return "no such object" in text or "no such container" in text


async def _container_inspect(config: "SSHConfig") -> Dict[str, Any]:
    fmt = "{{json .}}"
    cmd = f"docker inspect --format {shlex.quote(fmt)} {CONTAINER_NAME}"
    rc, out, err = await _ssh_exec(config, cmd)
    if rc != 0:
        message = (
            f"docker inspect {CONTAINER_NAME!r} failed on {config.host}: "
            f"rc={rc} stderr={err!r}"
        )
        if _docker_object_not_found(out, err):
            raise _ContainerNotFoundError(message)
        raise _ContainerInspectUnavailableError(message)
    info = _parse_container_inspect_json(out)
    if info is None:
        raise _ContainerInspectUnavailableError(
            f"docker inspect {CONTAINER_NAME!r} on {config.host} returned "
            "no parseable JSON"
        )
    return info


async def _probe_remote_ready(
    config: "SSHConfig",
    *,
    timeout_sec: float = 5.0,
) -> Optional[bool]:
    """Probe SGLang from inside its host-networked container.

    ``None`` means the SSH/docker probe itself could not be executed, so the
    caller must not infer that SGLang is down from this observation.
    """
    url = f"http://127.0.0.1:{config.sglang_port}/v1/models"
    script = (
        "import json, urllib.request\n"
        "try:\n"
        f"    response = urllib.request.urlopen({url!r}, timeout={timeout_sec!r})\n"
        "    payload = json.load(response)\n"
        "    ready = response.status == 200 and isinstance(payload, dict) "
        "and isinstance(payload.get('data'), list) and bool(payload['data'])\n"
        "except Exception:\n"
        "    ready = False\n"
        "raise SystemExit(0 if ready else 1)\n"
    )
    cmd = (
        f"docker exec {shlex.quote(CONTAINER_NAME)} "
        f"python -c {shlex.quote(script)}"
    )
    try:
        rc, _out, err = await _ssh_exec(config, cmd)
    except Exception as e:
        logger.warning(
            f"ssh-provider: local readiness probe could not run on "
            f"{config.host}: {type(e).__name__}: {e}"
        )
        return None
    if rc == 0:
        return True
    if rc == 1:
        return False
    logger.warning(
        f"ssh-provider: local readiness probe could not run on {config.host}: "
        f"rc={rc} stderr={err!r}"
    )
    return None


async def deployment_health(
    config: "SSHConfig",
    target: DeployTarget,
    *,
    base_url: Optional[str] = None,
) -> DeploymentHealthResult:
    """Classify whether the remote container is still serving ``target``.

    This is a runtime reconcile check for the flow scheduler. ``/v1/models``
    alone is not enough on a single-instance host because the stable URL may
    be serving a different miner after manual intervention or a crash window;
    require both Docker labels and the OpenAI readiness probe to match.
    """
    probe_url = base_url or config.inference_url()

    def result(
        state: DeploymentHealthState,
        reason: str = "",
    ) -> DeploymentHealthResult:
        return DeploymentHealthResult(
            state,
            reason=reason,
            identity=config.health_identity(),
            canonical_base_url=config.inference_url(),
        )

    try:
        info = await _container_inspect(config)
    except _ContainerNotFoundError as e:
        logger.warning(f"ssh-provider: {e}")
        return result(
            DeploymentHealthState.UNHEALTHY,
            reason="container_missing",
        )
    except Exception as e:
        logger.warning(
            f"ssh-provider: docker inspect health check unavailable on "
            f"{config.host}; checking the public inference path before "
            f"preserving deployment state: "
            f"{type(e).__name__}: {e}"
        )
        if await _probe_ready(probe_url):
            return result(
                DeploymentHealthState.HEALTHY,
                reason="container_inspect_unavailable_public_ready",
            )
        return result(
            DeploymentHealthState.UNKNOWN,
            reason="container_inspect_and_public_probe_unavailable",
        )

    state = info.get("State") if isinstance(info.get("State"), dict) else {}
    status = str(state.get("Status") or "")
    container_cfg = (
        info.get("Config") if isinstance(info.get("Config"), dict) else {}
    )
    labels = container_cfg.get("Labels") or {}
    if not isinstance(labels, dict):
        labels = {}
    expected_labels = {
        "io.affine.endpoint": config.endpoint_name or config.host,
        "io.affine.uid": str(target.uid),
        "io.affine.hotkey": target.hotkey,
        "io.affine.model": target.model,
        "io.affine.revision": target.revision,
    }
    for key, expected in expected_labels.items():
        actual = str(labels.get(key) or "")
        if actual != str(expected):
            logger.warning(
                f"ssh-provider: {CONTAINER_NAME} label mismatch on "
                f"{config.host}: {key}={actual!r}, expected={expected!r}"
            )
            return result(
                DeploymentHealthState.UNHEALTHY,
                reason=f"container_label_mismatch:{key}",
            )

    if status in {"exited", "dead"}:
        logger.warning(
            f"ssh-provider: {CONTAINER_NAME} exited on {config.host} "
            f"(status={status!r}, exit={state.get('ExitCode')!r})"
        )
        return result(
            DeploymentHealthState.UNHEALTHY,
            reason=f"container_status:{status}",
        )
    if status in {"created", "restarting", "removing", "paused"}:
        logger.warning(
            f"ssh-provider: {CONTAINER_NAME} is in a transitional state on "
            f"{config.host} (status={status!r})"
        )
        return result(
            DeploymentHealthState.SUSPECTED,
            reason=f"container_status:{status or 'unknown'}",
        )
    if status != "running":
        return result(
            DeploymentHealthState.UNKNOWN,
            reason=f"container_status:{status or 'unknown'}",
        )

    if await _probe_ready(probe_url):
        return result(DeploymentHealthState.HEALTHY)

    local_ready = await _probe_remote_ready(config)
    if local_ready is True:
        logger.warning(
            f"ssh-provider: {CONTAINER_NAME} is ready on GPU-local port "
            f"{config.sglang_port}, but public probe {probe_url}/models failed"
        )
        return result(
            DeploymentHealthState.TRANSPORT_UNHEALTHY,
            reason="public_probe_failed_local_ready",
        )
    if local_ready is False:
        logger.warning(
            f"ssh-provider: {CONTAINER_NAME} labels match uid={target.uid}, "
            "but both public and GPU-local readiness probes failed"
        )
        return result(
            DeploymentHealthState.SUSPECTED,
            reason="public_and_local_probes_failed",
        )
    return result(
        DeploymentHealthState.UNKNOWN,
        reason="public_probe_failed_local_probe_unavailable",
    )


async def deployment_healthy(
    config: "SSHConfig",
    target: DeployTarget,
    *,
    base_url: Optional[str] = None,
) -> bool:
    """Compatibility wrapper for callers using the previous bool contract."""
    health = await deployment_health(config, target, base_url=base_url)
    return health.state is DeploymentHealthState.HEALTHY


async def _wait_ready(
    base_url: str, *,
    deadline_sec: int = DEFAULT_READY_TIMEOUT_SEC,
    poll_interval_sec: float = DEFAULT_POLL_INTERVAL_SEC,
    config: Optional["SSHConfig"] = None,
) -> None:
    """Block until /v1/models reports the loaded model. Raise early if
    the sglang container has exited (sglang crashed on startup —
    waiting the full ``deadline_sec`` is pure waste; the upstream
    deploy-failure handler will leave the miner in queue regardless)."""
    deadline = time.monotonic() + deadline_sec
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        if await _probe_ready(base_url):
            logger.info(
                f"ssh-provider: model ready after {attempt} probes ({base_url})"
            )
            return
        if config is not None:
            exit_code = await _container_exit_status(config)
            if exit_code is not None:
                raise RuntimeError(
                    f"ssh-provider: sglang container {CONTAINER_NAME!r} "
                    f"exited with code {exit_code} before /v1/models came up "
                    f"({base_url})"
                )
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
      3. ``docker run -d --gpus all ... lmsysorg/sglang:v0.5.14 python -m
         sglang.launch_server ...``
      4. Poll ``http://<host>:<port>/v1/models`` until ready

    On success returns ``DeployResult(deployment_id="ssh:<endpoint>:...", base_url=...)``.
    """
    base_url = config.inference_url()

    # Purge stale HF caches before the new container starts. The previous
    # container's open files keep its own model dir alive against the
    # ``rm -rf`` (Linux unlink semantics), so this is safe to do while
    # the old sglang is still running.
    await _cleanup_stale_caches(config, target)

    disable_hf_xet = await _should_disable_hf_xet(target)
    cmd = _build_docker_run_cmd(
        target, config, disable_hf_xet=disable_hf_xet,
    )
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
        config=config,
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
