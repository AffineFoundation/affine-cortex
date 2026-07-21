"""Single-host SGLang prefill/decode lifecycle for SSH endpoints.

The production-facing endpoint is one CPU-only SGLang Model Gateway. Four
single-GPU prefill workers (GPU 0..3) and four single-GPU decode workers
(GPU 4..7) listen on loopback only. Mooncake transfers KV cache directly
between workers; the gateway only coordinates the two HTTP request legs.

This module deliberately supports exactly 4P4D. Treating the topology as a
versioned deployment mode keeps a partially configured PD endpoint from
silently consuming the wrong GPUs. A future multi-host implementation should
be a separate topology instead of weakening these invariants.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from affine.core.providers.targon_client import (
    QWEN36_REASONING_PARSER,
    QWEN36_TOOL_CALL_PARSER,
    is_qwen36,
)
from affine.core.setup import logger

from .health import DeploymentHealthResult, DeploymentHealthState
from .ssh import (
    CONTAINER_NAME,
    RESTART_POLICY,
    SSHConfig,
    _cleanup_stale_caches,
    _probe_ready,
    _parse_container_exit_status,
    _should_disable_hf_xet,
    _ssh_exec,
    _strip_env_from_docker_args,
)
from .targon import DeployResult, DeployTarget, MachineDeployment


PREFILL_CONTAINER_PREFIX = "affine-sglang-pd-prefill"
DECODE_CONTAINER_PREFIX = "affine-sglang-pd-decode"
GATEWAY_CONTAINER_NAME = "affine-sglang-pd-gateway"
PREFILL_REPLICAS = 4
DECODE_REPLICAS = 4
GPU_COUNT = PREFILL_REPLICAS + DECODE_REPLICAS
GATEWAY_PROMETHEUS_PORT = 14000
GATEWAY_REQUEST_TIMEOUT_SEC = 3720
NUM_CONTINUOUS_DECODE_STEPS = 8

SUPPORTED_TRANSFER_BACKENDS = ("mooncake", "nixl")
SUPPORTED_ROUTING_POLICIES = (
    "random",
    "round_robin",
    "cache_aware",
    "power_of_two",
    "bucket",
    "manual",
    "consistent_hashing",
    "prefix_hash",
)
_DIGEST_IMAGE_RE = re.compile(r"^.+@sha256:[0-9a-f]{64}$")
_RESERVED_DOCKER_OPTIONS = (
    "--device",
    "--entrypoint",
    "--gpus",
    "--ipc",
    "--name",
    "--network",
    "--publish",
    "--restart",
    "--runtime",
    "--shm-size",
    "-p",
    "-P",
)


@dataclass(frozen=True)
class PDWorker:
    role: str
    replica: int
    gpu: int
    http_port: int
    bootstrap_port: Optional[int] = None

    @property
    def container_name(self) -> str:
        prefix = (
            PREFILL_CONTAINER_PREFIX
            if self.role == "prefill"
            else DECODE_CONTAINER_PREFIX
        )
        return f"{prefix}-{self.replica}"


def workers(config: SSHConfig) -> Tuple[PDWorker, ...]:
    prefill = tuple(
        PDWorker(
            role="prefill",
            replica=replica,
            gpu=replica,
            http_port=config.sglang_pd_prefill_port_start + replica,
            bootstrap_port=config.sglang_pd_bootstrap_port_start + replica,
        )
        for replica in range(config.sglang_pd_prefill_replicas)
    )
    decode = tuple(
        PDWorker(
            role="decode",
            replica=replica,
            gpu=config.sglang_pd_prefill_replicas + replica,
            http_port=config.sglang_pd_decode_port_start + replica,
        )
        for replica in range(config.sglang_pd_decode_replicas)
    )
    return prefill + decode


def managed_container_names(config: SSHConfig) -> Tuple[str, ...]:
    topology = workers(config)
    # Stop admission first, then decode, then prefill. Exact names keep
    # teardown recoverable and avoid wildcard removal on a shared host.
    ordered_workers = tuple(
        worker.container_name
        for role in ("decode", "prefill")
        for worker in topology
        if worker.role == role
    )
    return (GATEWAY_CONTAINER_NAME,) + ordered_workers


def required_gpu_count(config: SSHConfig) -> int:
    if config.serving_mode == "pd":
        return config.sglang_pd_prefill_replicas + config.sglang_pd_decode_replicas
    return max(1, int(config.sglang_dp))


def reservation_timeout_sec(config: SSHConfig) -> int:
    """Upper bound for two sequential readiness phases plus smoke test."""
    return max(60, int(config.ready_timeout_sec)) * 2 + 60


def _require_digest_image(image: str, field_name: str) -> None:
    if not _DIGEST_IMAGE_RE.fullmatch(str(image or "")):
        raise ValueError(
            f"{field_name} must be pinned as <image>@sha256:<64 hex chars> "
            "for serving_mode=pd"
        )


def validate_config(config: SSHConfig) -> None:
    if config.serving_mode != "pd":
        raise ValueError(
            f"ssh PD lifecycle requires serving_mode='pd', got {config.serving_mode!r}"
        )
    if config.sglang_pd_prefill_replicas != PREFILL_REPLICAS:
        raise ValueError(
            f"serving_mode=pd currently requires exactly {PREFILL_REPLICAS} "
            "prefill replicas"
        )
    if config.sglang_pd_decode_replicas != DECODE_REPLICAS:
        raise ValueError(
            f"serving_mode=pd currently requires exactly {DECODE_REPLICAS} "
            "decode replicas"
        )
    for arg in config.sglang_docker_args:
        if any(
            arg == option
            or arg.startswith(f"{option}=")
            or (option == "-p" and re.match(r"^-p\d", arg) is not None)
            for option in _RESERVED_DOCKER_OPTIONS
        ):
            raise ValueError(
                f"sglang_docker_args may not override PD topology option {arg!r}"
            )
    _require_digest_image(config.sglang_image, "sglang_image")
    _require_digest_image(
        config.sglang_pd_gateway_image,
        "sglang_pd_gateway_image",
    )

    backend = str(config.sglang_pd_transfer_backend or "")
    if backend not in SUPPORTED_TRANSFER_BACKENDS:
        raise ValueError(
            f"unsupported PD transfer backend {backend!r}; expected one of "
            f"{', '.join(SUPPORTED_TRANSFER_BACKENDS)}"
        )
    for field_name, policy in (
        ("sglang_pd_prefill_policy", config.sglang_pd_prefill_policy),
        ("sglang_pd_decode_policy", config.sglang_pd_decode_policy),
    ):
        if policy not in SUPPORTED_ROUTING_POLICIES:
            raise ValueError(
                f"unsupported {field_name}={policy!r}; expected one of "
                f"{', '.join(SUPPORTED_ROUTING_POLICIES)}"
            )

    ports = [config.sglang_port, GATEWAY_PROMETHEUS_PORT]
    for worker in workers(config):
        ports.append(worker.http_port)
        if worker.bootstrap_port is not None:
            ports.append(worker.bootstrap_port)
    if any(port < 1 or port > 65535 for port in ports):
        raise ValueError("all PD and gateway ports must be in [1, 65535]")
    if len(ports) != len(set(ports)):
        raise ValueError("gateway, metrics, worker, and bootstrap ports must be unique")


def _base_labels(
    target: DeployTarget,
    config: SSHConfig,
) -> Dict[str, str]:
    return {
        "io.affine.endpoint": config.endpoint_name or config.host,
        "io.affine.uid": str(target.uid),
        "io.affine.hotkey": target.hotkey,
        "io.affine.model": target.model,
        "io.affine.revision": target.revision,
        "io.affine.serving_mode": "pd",
    }


def _label_flags(labels: Dict[str, str]) -> str:
    return " ".join(
        f"--label {shlex.quote(key)}={shlex.quote(value)}"
        for key, value in labels.items()
    )


def _worker_args(
    target: DeployTarget,
    config: SSHConfig,
    worker: PDWorker,
) -> List[str]:
    args = [
        "--model-path",
        target.model,
        "--revision",
        target.revision,
        "--download-dir",
        "/data",
        "--host",
        "127.0.0.1",
        "--port",
        str(worker.http_port),
        "--trust-remote-code",
        "--mem-fraction-static",
        str(config.sglang_mem_fraction),
        "--chunked-prefill-size",
        str(config.sglang_chunked_prefill),
        "--disaggregation-mode",
        worker.role,
        "--disaggregation-transfer-backend",
        config.sglang_pd_transfer_backend,
        "--num-continuous-decode-steps",
        str(NUM_CONTINUOUS_DECODE_STEPS),
        "--enable-mixed-chunk",
        "--enable-metrics",
    ]
    if worker.bootstrap_port is not None:
        args += [
            "--disaggregation-bootstrap-port",
            str(worker.bootstrap_port),
        ]
    if config.sglang_pd_ib_device:
        args += [
            "--disaggregation-ib-device",
            config.sglang_pd_ib_device,
        ]

    parser = config.sglang_tool_call_parser
    if is_qwen36(target.model_type):
        args += ["--reasoning-parser", QWEN36_REASONING_PARSER]
        parser = QWEN36_TOOL_CALL_PARSER
    if parser and parser.lower() != "none":
        args += ["--tool-call-parser", parser]
    return args


def _operator_docker_flags(config: SSHConfig) -> str:
    args = config.sglang_docker_args
    for env_name in (
        "HF_HUB_DISABLE_XET",
        "SGLANG_MOONCAKE_CUSTOM_MEM_POOL",
        "MC_INTRANODE_NVLINK",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
    ):
        args = _strip_env_from_docker_args(args, env_name)
    return " ".join(shlex.quote(str(arg)) for arg in args)


def _hf_env_flags(*, disable_hf_xet: bool) -> str:
    env = [
        "-e HF_HOME=/data",
        "-e HF_HUB_CACHE=/data",
        "-e TRANSFORMERS_CACHE=/data",
        "-e HF_HUB_DOWNLOAD_TIMEOUT=60",
    ]
    if disable_hf_xet:
        env.append("-e HF_HUB_DISABLE_XET=1")
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        token = shlex.quote(hf_token)
        env.extend((f"-e HF_TOKEN={token}", f"-e HUGGING_FACE_HUB_TOKEN={token}"))
    return " ".join(env)


def _build_worker_docker_cmd(
    target: DeployTarget,
    config: SSHConfig,
    worker: PDWorker,
    *,
    disable_hf_xet: bool = False,
) -> str:
    labels = {
        **_base_labels(target, config),
        "io.affine.pd_role": worker.role,
        "io.affine.pd_replica": str(worker.replica),
        "io.affine.gpu": str(worker.gpu),
    }
    args = " ".join(
        shlex.quote(str(arg)) for arg in _worker_args(target, config, worker)
    )
    mooncake_env = ""
    if config.sglang_pd_transfer_backend == "mooncake":
        mooncake_env = (
            "-e SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK "
            "-e MC_INTRANODE_NVLINK=true "
        )
    return (
        f"docker run -d --name {worker.container_name} "
        f"--gpus {shlex.quote(f'device={worker.gpu}')} "
        f"--restart {RESTART_POLICY} --network host --ipc=host "
        "--shm-size=32g --security-opt label=disable --entrypoint python "
        f"{_operator_docker_flags(config)} {_label_flags(labels)} "
        f"-v {shlex.quote(config.sglang_cache_dir)}:/data "
        f"{_hf_env_flags(disable_hf_xet=disable_hf_xet)} "
        f"{mooncake_env}{shlex.quote(config.sglang_image)} "
        f"-m sglang.launch_server {args}"
    )


def _gateway_args(config: SSHConfig) -> List[str]:
    args = [
        "--pd-disaggregation",
        "--host",
        "0.0.0.0",
        "--port",
        str(config.sglang_port),
        "--prefill-policy",
        config.sglang_pd_prefill_policy,
        "--decode-policy",
        config.sglang_pd_decode_policy,
        "--worker-startup-timeout-secs",
        str(config.ready_timeout_sec),
        "--request-timeout-secs",
        str(GATEWAY_REQUEST_TIMEOUT_SEC),
        "--prometheus-host",
        "127.0.0.1",
        "--prometheus-port",
        str(GATEWAY_PROMETHEUS_PORT),
    ]
    for worker in workers(config):
        url = f"http://127.0.0.1:{worker.http_port}"
        if worker.role == "prefill":
            args += ["--prefill", url, str(worker.bootstrap_port)]
        else:
            args += ["--decode", url]
    return args


def _build_gateway_docker_cmd(
    target: DeployTarget,
    config: SSHConfig,
) -> str:
    labels = {
        **_base_labels(target, config),
        "io.affine.pd_role": "gateway",
    }
    args = " ".join(shlex.quote(arg) for arg in _gateway_args(config))
    return (
        f"docker run -d --name {GATEWAY_CONTAINER_NAME} "
        f"--restart {RESTART_POLICY} --network host "
        "--security-opt label=disable --entrypoint python "
        f"{_label_flags(labels)} "
        f"{shlex.quote(config.sglang_pd_gateway_image)} "
        f"-m sglang_router.launch_router {args}"
    )


def _remove_containers_cmd(config: SSHConfig) -> str:
    names = " ".join(
        shlex.quote(name) for name in (*managed_container_names(config), CONTAINER_NAME)
    )
    return f"docker rm -f {names} >/dev/null 2>&1 || true"


async def _remote_workers_ready(config: SSHConfig) -> bool:
    probe_script = (
        "import json,urllib.request; "
        "payload=json.load(urllib.request.urlopen("
        "'http://127.0.0.1:{port}/v1/models',timeout=5)); "
        "assert isinstance(payload.get('data'),list) and payload['data']"
    )
    probes = " && ".join(
        f"docker exec {shlex.quote(worker.container_name)} python -c "
        f"{shlex.quote(probe_script.format(port=worker.http_port))}"
        for worker in workers(config)
    )
    rc, _out, _err = await _ssh_exec(config, probes)
    return rc == 0


async def _container_exit_code(
    config: SSHConfig,
    container_name: str,
) -> Optional[int]:
    fmt = shlex.quote("{{.State.Status}} {{.State.ExitCode}}")
    rc, out, _err = await _ssh_exec(
        config,
        f"docker inspect --format {fmt} {shlex.quote(container_name)}",
    )
    if rc != 0:
        return None
    return _parse_container_exit_status(out)


async def _wait_workers_ready(config: SSHConfig) -> None:
    deadline = time.monotonic() + config.ready_timeout_sec
    while time.monotonic() < deadline:
        if await _remote_workers_ready(config):
            return
        for worker in workers(config):
            exit_code = await _container_exit_code(
                config,
                worker.container_name,
            )
            if exit_code is not None:
                raise RuntimeError(
                    f"ssh-pd: {worker.container_name} exited with code "
                    f"{exit_code} before readiness"
                )
        await asyncio.sleep(config.poll_interval_sec)
    raise TimeoutError(
        f"ssh-pd: workers did not become ready in {config.ready_timeout_sec}s"
    )


async def _wait_gateway_ready(config: SSHConfig) -> None:
    base_url = config.inference_url()
    deadline = time.monotonic() + config.ready_timeout_sec
    while time.monotonic() < deadline:
        if await _probe_ready(base_url):
            return
        exit_code = await _container_exit_code(
            config,
            GATEWAY_CONTAINER_NAME,
        )
        if exit_code is not None:
            raise RuntimeError(
                f"ssh-pd: {GATEWAY_CONTAINER_NAME} exited with code "
                f"{exit_code} before readiness"
            )
        await asyncio.sleep(config.poll_interval_sec)
    raise TimeoutError(
        f"ssh-pd: gateway {base_url} did not become ready in "
        f"{config.ready_timeout_sec}s"
    )


async def _smoke_test(
    base_url: str,
    target: DeployTarget,
    *,
    timeout_sec: float = 60.0,
) -> None:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": target.model,
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "max_tokens": 1,
        "temperature": 0,
        "stream": False,
    }
    timeout = aiohttp.ClientTimeout(total=timeout_sec)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=payload) as response:
            body = await response.text()
            if response.status != 200:
                raise RuntimeError(
                    f"ssh-pd: gateway smoke request failed with "
                    f"HTTP {response.status}: {body[:500]}"
                )
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    "ssh-pd: gateway smoke response was not JSON"
                ) from error
            if not isinstance(parsed.get("choices"), list):
                raise RuntimeError("ssh-pd: gateway smoke response has no choices")


def _parse_inspect_documents(output: str) -> List[Dict[str, Any]]:
    documents: List[Dict[str, Any]] = []
    for line in (output or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            documents.append(value)
    return documents


async def _inspect_containers(config: SSHConfig) -> List[Dict[str, Any]]:
    names = " ".join(shlex.quote(name) for name in managed_container_names(config))
    fmt = shlex.quote("{{json .}}")
    rc, out, err = await _ssh_exec(
        config,
        f"docker inspect --format {fmt} {names}",
    )
    if rc != 0:
        documents = _parse_inspect_documents(out)
        lowered = (err or "").lower()
        if "no such object" in lowered or "no such container" in lowered:
            return documents
        logger.warning(
            f"ssh-pd: docker inspect failed on {config.host}: rc={rc} stderr={err!r}"
        )
        raise RuntimeError(
            f"docker inspect unavailable: rc={rc} stderr={err!r}"
        )
    return _parse_inspect_documents(out)


async def _probe_remote_gateway_ready(config: SSHConfig) -> Optional[bool]:
    """Probe the gateway on the GPU host, bypassing its public transport."""
    url = f"http://127.0.0.1:{config.sglang_port}/v1/models"
    script = (
        "import json, urllib.request\n"
        "try:\n"
        f"    response = urllib.request.urlopen({url!r}, timeout=5)\n"
        "    payload = json.load(response)\n"
        "    ready = response.status == 200 and isinstance(payload, dict) "
        "and isinstance(payload.get('data'), list) and bool(payload['data'])\n"
        "except Exception:\n"
        "    ready = False\n"
        "raise SystemExit(0 if ready else 1)\n"
    )
    command = (
        f"docker exec {shlex.quote(GATEWAY_CONTAINER_NAME)} "
        f"python -c {shlex.quote(script)}"
    )
    try:
        rc, _out, err = await _ssh_exec(config, command)
    except Exception as error:
        logger.warning(
            f"ssh-pd: GPU-local gateway probe unavailable on {config.host}: "
            f"{type(error).__name__}: {error}"
        )
        return None
    if rc == 0:
        return True
    if rc == 1:
        return False
    logger.warning(
        f"ssh-pd: GPU-local gateway probe unavailable on {config.host}: "
        f"rc={rc} stderr={err!r}"
    )
    return None


def _expected_container_labels(
    target: DeployTarget,
    config: SSHConfig,
) -> Dict[str, Dict[str, str]]:
    expected = {
        GATEWAY_CONTAINER_NAME: {
            **_base_labels(target, config),
            "io.affine.pd_role": "gateway",
        }
    }
    for worker in workers(config):
        expected[worker.container_name] = {
            **_base_labels(target, config),
            "io.affine.pd_role": worker.role,
            "io.affine.pd_replica": str(worker.replica),
            "io.affine.gpu": str(worker.gpu),
        }
    return expected


async def deployment_health(
    config: SSHConfig,
    target: DeployTarget,
    *,
    base_url: Optional[str] = None,
) -> DeploymentHealthResult:
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
        validate_config(config)
        documents = await _inspect_containers(config)
    except Exception as error:
        logger.warning(
            f"ssh-pd: health inspection failed on {config.host}: "
            f"{type(error).__name__}: {error}"
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

    expected = _expected_container_labels(target, config)
    actual_by_name = {
        str(document.get("Name") or "").lstrip("/"): document for document in documents
    }
    if set(actual_by_name) != set(expected):
        logger.warning(
            f"ssh-pd: managed container set mismatch on {config.host}: "
            f"actual={sorted(actual_by_name)} expected={sorted(expected)}"
        )
        return result(
            DeploymentHealthState.UNHEALTHY,
            reason="container_set_mismatch",
        )
    for name, expected_labels in expected.items():
        document = actual_by_name[name]
        state = document.get("State") if isinstance(document.get("State"), dict) else {}
        status = str(state.get("Status") or "")
        if status in {"exited", "dead"}:
            return result(
                DeploymentHealthState.UNHEALTHY,
                reason=f"container_status:{name}:{status}",
            )
        if status in {"created", "restarting", "removing", "paused"}:
            return result(
                DeploymentHealthState.SUSPECTED,
                reason=f"container_status:{name}:{status}",
            )
        if status != "running":
            return result(
                DeploymentHealthState.UNKNOWN,
                reason=f"container_status:{name}:{status or 'unknown'}",
            )
        container_config = (
            document.get("Config") if isinstance(document.get("Config"), dict) else {}
        )
        labels = container_config.get("Labels") or {}
        for key, value in expected_labels.items():
            if str(labels.get(key) or "") != value:
                return result(
                    DeploymentHealthState.UNHEALTHY,
                    reason=f"container_label_mismatch:{name}:{key}",
                )

    if await _probe_ready(probe_url):
        return result(DeploymentHealthState.HEALTHY)
    local_ready = await _probe_remote_gateway_ready(config)
    if local_ready is True:
        return result(
            DeploymentHealthState.TRANSPORT_UNHEALTHY,
            reason="public_probe_failed_local_ready",
        )
    if local_ready is False:
        return result(
            DeploymentHealthState.SUSPECTED,
            reason="public_and_local_probes_failed",
        )
    return result(
        DeploymentHealthState.UNKNOWN,
        reason="public_probe_failed_local_probe_unavailable",
    )


async def deployment_healthy(
    config: SSHConfig,
    target: DeployTarget,
    *,
    base_url: Optional[str] = None,
) -> bool:
    """Compatibility wrapper for callers using the previous bool contract."""
    health = await deployment_health(config, target, base_url=base_url)
    return health.state is DeploymentHealthState.HEALTHY


async def deploy(config: SSHConfig, target: DeployTarget) -> DeployResult:
    validate_config(config)
    await _cleanup_stale_caches(config, target)
    disable_hf_xet = await _should_disable_hf_xet(target)
    await teardown(config, None)

    try:
        worker_commands = [
            _build_worker_docker_cmd(
                target,
                config,
                worker,
                disable_hf_xet=disable_hf_xet,
            )
            for worker in workers(config)
        ]
        rc, out, err = await _ssh_exec(config, " && ".join(worker_commands))
        if rc != 0:
            raise RuntimeError(
                f"ssh-pd: worker launch failed on {config.host}: rc={rc} stderr={err!r}"
            )
        if out:
            logger.info("ssh-pd: launched 4P4D workers")
        await _wait_workers_ready(config)

        rc, _out, err = await _ssh_exec(
            config,
            _build_gateway_docker_cmd(target, config),
        )
        if rc != 0:
            raise RuntimeError(
                f"ssh-pd: gateway launch failed on {config.host}: "
                f"rc={rc} stderr={err!r}"
            )
        base_url = config.inference_url()
        await _wait_gateway_ready(config)
        await _smoke_test(base_url, target)
    except Exception:
        await teardown(config, None)
        raise

    return DeployResult(
        deployment_id=config.deployment_id(),
        base_url=base_url,
        # Only the gateway is callable by Terminal. Worker URLs must never be
        # serialized into scorer deployment records.
        deployments=[
            MachineDeployment(
                endpoint_name=config.endpoint_name,
                deployment_id=config.deployment_id(),
                base_url=base_url,
            )
        ],
    )


async def teardown(
    config: SSHConfig,
    deployment_id: Optional[str],
) -> None:
    if deployment_id and deployment_id != config.deployment_id():
        logger.info(
            f"ssh-pd: skip teardown for deployment_id={deployment_id!r}; "
            f"this endpoint owns {config.deployment_id()!r}"
        )
        return
    try:
        rc, _out, err = await _ssh_exec(config, _remove_containers_cmd(config))
        if rc != 0:
            logger.warning(f"ssh-pd: teardown rc={rc} stderr={err!r}")
    except Exception as error:
        logger.warning(f"ssh-pd: teardown raised {type(error).__name__}: {error}")
