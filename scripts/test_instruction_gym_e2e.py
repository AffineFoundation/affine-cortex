#!/usr/bin/env python3
"""Exercise the Affine scheduler-to-Affinetes-to-InstructionGym image path.

This is intentionally a local-image gate.  Production enablement additionally
requires replacing the configured image with the published registry digest and
repeating the same probes after a clean pull.  The gate requires a reachable
OpenAI-compatible success endpoint and separately verifies the classified
connection-failure path.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import ipaddress
import json
import os
import random
import re
import secrets
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from affine.core.environments import (
    ENV_CONFIGS,
    INSTRUCTION_GYM_TASK_ID_END,
    _ENV_CACHE,
    SDKEnvironment,
)
from affine.src.scorer.sampler import (
    SAMPLING_MODE_RANDOM,
    EnvSamplingConfig,
    WindowSampler,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_CONTAINER_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_ENVIRONMENT_VARIABLE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_FAILURE_BASE_URL = "http://127.0.0.1:9"
_OWNER_LABEL = "io.affine.instruction-gym-e2e.owner"
_MAX_LOCAL_SAMPLING_COUNT = 1_000
_JUDGE_BASE_URL_ENV = "INSTRUCTION_GYM_JUDGE_BASE_URL"
_JUDGE_API_KEY_ENV = "INSTRUCTION_GYM_JUDGE_API_KEY"
_JUDGE_ENSEMBLE_JSON_ENV = "INSTRUCTION_GYM_JUDGE_ENSEMBLE_JSON"
_APPROVED_JUDGE_ENSEMBLE_MANIFEST_SHA256_ENV = (
    "INSTRUCTION_GYM_APPROVED_SEMANTIC_JUDGE_ENSEMBLE_MANIFEST_SHA256"
)
_LOCAL_JUDGE_BASE_URL = "http://127.0.0.1:8/v1"
_LOCAL_JUDGE_API_KEY = "affine-instruction-gym-e2e-local-judge-key"
_LOCAL_JUDGE_ENSEMBLE_JSON = json.dumps(
    {
        "aggregation_mode": "min",
        "members": [
            {
                "model_id": "affine-instruction-gym-e2e-judge-a",
                "model_revision": "affine-instruction-gym-e2e-judge-a-v1",
            },
            {
                "model_id": "affine-instruction-gym-e2e-judge-b",
                "model_revision": "affine-instruction-gym-e2e-judge-b-v1",
            },
        ],
    },
    separators=(",", ":"),
    sort_keys=True,
)
_JUDGE_MANIFEST_PROBE = (
    "from instruction_gym.semantic import SemanticJudgeEnsembleConfig; "
    "value = SemanticJudgeEnsembleConfig.from_env().manifest.manifest_sha256; "
    "assert value is not None; print(value)"
)


class IntegrationFailure(RuntimeError):
    """A bounded, user-safe end-to-end integration failure."""


class _DeterministicWindowSampler(WindowSampler):
    """Use a recorded RNG stream while exercising the production algorithm."""

    def __init__(self, seed: int) -> None:
        self._seed = seed

    def _rng(self) -> random.Random:
        return random.Random(self._seed)


def _is_loopback_host(hostname: str | None) -> bool:
    if hostname is None:
        return False
    normalized = hostname.rstrip(".").lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image", required=True, help="Already-built local image reference"
    )
    parser.add_argument(
        "--container-name",
        default="affine-instruction-gym-e2e",
        help="Dedicated temporary Docker container name",
    )
    parser.add_argument("--sampling-count", type=int, default=3)
    parser.add_argument("--sampling-seed", type=int, default=20260714)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument(
        "--success-base-url",
        required=True,
        help="OpenAI-compatible endpoint reachable from the temporary container",
    )
    parser.add_argument(
        "--success-api-key-env",
        help="Optional environment-variable name holding the per-call endpoint key",
    )
    parser.add_argument(
        "--host-network",
        action="store_true",
        help="Use Docker host networking so a localhost stub is reachable",
    )
    parser.add_argument(
        "--actor-host-port",
        type=int,
        default=18081,
        help="InstructionGym Actor port when --host-network is used",
    )
    parser.add_argument("--output", type=Path)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if (
        not isinstance(args.image, str)
        or not args.image
        or args.image != args.image.strip()
    ):
        raise IntegrationFailure("--image must be non-empty and trimmed")
    if _CONTAINER_NAME.fullmatch(args.container_name) is None:
        raise IntegrationFailure("--container-name is not a bounded Docker name")
    if (
        isinstance(args.sampling_count, bool)
        or not isinstance(args.sampling_count, int)
        or not 2 <= args.sampling_count <= _MAX_LOCAL_SAMPLING_COUNT
    ):
        raise IntegrationFailure(
            f"--sampling-count must be within [2, {_MAX_LOCAL_SAMPLING_COUNT}]"
        )
    if isinstance(args.sampling_seed, bool) or not isinstance(args.sampling_seed, int):
        raise IntegrationFailure("--sampling-seed must be an integer")
    if (
        isinstance(args.timeout, bool)
        or not isinstance(args.timeout, (int, float))
        or not 1 <= float(args.timeout) <= 120
    ):
        raise IntegrationFailure("--timeout must be within [1, 120] seconds")
    if not isinstance(args.success_base_url, str) or not args.success_base_url:
        raise IntegrationFailure("--success-base-url must be non-empty")
    if len(args.success_base_url) > 2048:
        raise IntegrationFailure("--success-base-url is too long")
    parsed_url = urlsplit(args.success_base_url)
    if (
        parsed_url.scheme not in {"http", "https"}
        or not parsed_url.netloc
        or parsed_url.username is not None
        or parsed_url.password is not None
        or parsed_url.fragment
    ):
        raise IntegrationFailure(
            "--success-base-url must be an HTTP(S) URL without userinfo or fragment"
        )
    if not _is_loopback_host(parsed_url.hostname):
        raise IntegrationFailure(
            "--success-base-url must use a loopback host for this local-only gate"
        )
    if not args.host_network:
        raise IntegrationFailure(
            "--host-network is required for a localhost success endpoint"
        )
    if args.success_api_key_env is not None and (
        not isinstance(args.success_api_key_env, str)
        or _ENVIRONMENT_VARIABLE.fullmatch(args.success_api_key_env) is None
    ):
        raise IntegrationFailure("--success-api-key-env is not a valid variable name")
    if (
        isinstance(args.actor_host_port, bool)
        or not isinstance(args.actor_host_port, int)
        or not 1 <= args.actor_host_port <= 65535
    ):
        raise IntegrationFailure("--actor-host-port must be within [1, 65535]")


def deterministic_task_handoff(
    *,
    count: int,
    seed: int,
) -> list[dict[str, int]]:
    """Run Cortex's shared random WindowSampler and expose direct task IDs."""

    sampled = _DeterministicWindowSampler(seed).generate(
        window_id=0,
        block_start=0,
        env_configs={
            "instruction-gym": EnvSamplingConfig(
                env="instruction-gym",
                dataset_range=[[0, INSTRUCTION_GYM_TASK_ID_END]],
                sampling_count=count,
                mode=SAMPLING_MODE_RANDOM,
            )
        },
    )["instruction-gym"]
    rows = [{"task_id": task_id} for task_id in sampled]
    if len({row["task_id"] for row in rows}) != count:
        raise IntegrationFailure("WindowSampler returned duplicate task IDs")
    if not all(0 <= row["task_id"] < INSTRUCTION_GYM_TASK_ID_END for row in rows):
        raise IntegrationFailure("WindowSampler returned an out-of-range task ID")
    return rows


def _prompt_sha256(result: Any) -> str:
    extra = getattr(result, "extra", None)
    value = extra.get("prompt_sha256") if isinstance(extra, dict) else None
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise IntegrationFailure("Actor result is missing a valid prompt_sha256")
    return value


def _catalog_sha256(result: Any) -> str:
    extra = getattr(result, "extra", None)
    value = extra.get("catalog_sha256") if isinstance(extra, dict) else None
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise IntegrationFailure("Actor result is missing a valid catalog_sha256")
    return value


def _plain_result(result: Any) -> Any:
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if hasattr(result, "dict"):
        return result.dict()
    if isinstance(result, dict):
        return result
    return vars(result)


def _normalized_field_name(key: Any) -> str:
    return "".join(character for character in str(key).lower() if character.isalnum())


def _sensitive_request_keys(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = _normalized_field_name(key)
            if (
                normalized == "baseurl"
                or normalized.endswith("apikey")
                or normalized.endswith("authorization")
            ):
                found.append(str(key))
            found.extend(_sensitive_request_keys(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.extend(_sensitive_request_keys(item))
    return found


def _assert_secret_safe_result(result: Any, *, secrets: tuple[str, ...]) -> None:
    payload = _plain_result(result)
    serialized = json.dumps(payload, sort_keys=True, default=str)
    for secret in secrets:
        if secret and secret in serialized:
            raise IntegrationFailure("Affine Result retained endpoint secret material")
    extra = getattr(result, "extra", None)
    request = extra.get("request") if isinstance(extra, dict) else None
    if not isinstance(request, dict):
        raise IntegrationFailure(
            "Affine Result is missing its sanitized request evidence"
        )
    sensitive_keys = _sensitive_request_keys(request)
    if sensitive_keys:
        raise IntegrationFailure(
            "Affine Result retained endpoint routing or credential fields"
        )


def _validate_success(
    result: Any,
    *,
    task_id: int,
    secrets: tuple[str, ...],
) -> dict[str, Any]:
    extra = getattr(result, "extra", None)
    score = float(getattr(result, "score", -1))
    if (
        getattr(result, "success", None) is not True
        or getattr(result, "error", None) is not None
        or score not in {0.0, 1.0}
        or getattr(result, "task_id", None) != task_id
        or not isinstance(extra, dict)
    ):
        raise IntegrationFailure(
            f"task_id={task_id} did not complete the successful scoring contract"
        )
    _assert_secret_safe_result(result, secrets=secrets)
    request = extra["request"]
    if request.get("task_id") != task_id:
        raise IntegrationFailure(
            "successful Affine Result changed the scheduler task_id"
        )

    case_pass = extra.get("case_pass")
    constraint_results = extra.get("constraint_results")
    scoring_evidence = extra.get("scoring_evidence")
    conversation = extra.get("conversation")
    if (
        not isinstance(case_pass, bool)
        or score != float(case_pass)
        or not isinstance(constraint_results, list)
        or not constraint_results
        or not isinstance(scoring_evidence, dict)
        or scoring_evidence.get("constraint_result_count") != len(constraint_results)
        or not isinstance(conversation, list)
        or len(conversation) != 2
        or conversation[0].get("role") != "user"
        or conversation[1].get("role") != "assistant"
        or not isinstance(conversation[0].get("content"), str)
        or not isinstance(conversation[1].get("content"), str)
    ):
        raise IntegrationFailure(
            "successful Actor result lacks coherent scoring evidence"
        )

    prompt_sha256 = _prompt_sha256(result)
    response_sha256 = extra.get("response_sha256")
    normalized_response_sha256 = extra.get("normalized_response_sha256")
    if (
        _SHA256.fullmatch(str(response_sha256)) is None
        or _SHA256.fullmatch(str(normalized_response_sha256)) is None
        or hashlib.sha256(conversation[0]["content"].encode()).hexdigest()
        != prompt_sha256
        or hashlib.sha256(conversation[1]["content"].encode()).hexdigest()
        != response_sha256
        or scoring_evidence.get("prompt_sha256") != prompt_sha256
        or scoring_evidence.get("response_sha256") != response_sha256
        or scoring_evidence.get("normalized_response_sha256")
        != normalized_response_sha256
    ):
        raise IntegrationFailure(
            "successful Actor result has inconsistent scoring hashes"
        )

    return {
        "task_id": task_id,
        "score": score,
        "case_pass": case_pass,
        "prompt_sha256": prompt_sha256,
        "response_sha256": response_sha256,
        "catalog_sha256": _catalog_sha256(result),
        "constraint_result_count": len(constraint_results),
    }


def _validate_endpoint_failure(
    result: Any,
    *,
    task_id: int,
    secrets: tuple[str, ...] = (),
) -> dict[str, Any]:
    extra = getattr(result, "extra", None)
    if (
        getattr(result, "success", None) is not False
        or float(getattr(result, "score", -1)) != 0.0
        or not isinstance(getattr(result, "error", None), str)
        or not isinstance(extra, dict)
        or extra.get("error_code") != "endpoint_connection_failed"
        or getattr(result, "task_id", None) != task_id
    ):
        raise IntegrationFailure(
            f"task_id={task_id} did not reach the materialized endpoint-failure contract"
        )
    _assert_secret_safe_result(result, secrets=secrets)
    request = extra.get("request")
    if not isinstance(request, dict) or request.get("task_id") != task_id:
        raise IntegrationFailure(
            "Affine Result did not retain the direct scheduler task_id"
        )
    return {
        "task_id": task_id,
        "prompt_sha256": _prompt_sha256(result),
        "catalog_sha256": _catalog_sha256(result),
        "error_code": extra["error_code"],
    }


def _lookup_container(client: Any, docker_module: Any, name: str) -> Any | None:
    try:
        return client.containers.get(name)
    except docker_module.errors.NotFound:
        return None
    except Exception as exc:
        raise IntegrationFailure(
            f"Docker container lookup failed with {type(exc).__name__}"
        ) from exc


def _validate_local_docker_client(client: Any) -> None:
    """Reject a daemon reached through a non-loopback Docker endpoint."""

    base_url = getattr(getattr(client, "api", None), "base_url", None)
    if not isinstance(base_url, str):
        raise IntegrationFailure(
            "Docker client did not expose a bounded local endpoint"
        )
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https", "http+docker"} or not _is_loopback_host(
        parsed.hostname
    ):
        raise IntegrationFailure(
            "local-only integration refuses a non-loopback Docker daemon"
        )


def _local_judge_environment(client: Any, *, image_id: str) -> dict[str, str]:
    """Bind a non-production Judge identity to the exact local image under test."""

    environment = {
        _JUDGE_BASE_URL_ENV: _LOCAL_JUDGE_BASE_URL,
        _JUDGE_API_KEY_ENV: _LOCAL_JUDGE_API_KEY,
        _JUDGE_ENSEMBLE_JSON_ENV: _LOCAL_JUDGE_ENSEMBLE_JSON,
    }
    try:
        output = client.containers.run(
            image_id,
            command=["-c", _JUDGE_MANIFEST_PROBE],
            entrypoint="python",
            environment=environment,
            network_disabled=True,
            remove=True,
            stdout=True,
            stderr=False,
        )
    except Exception as exc:
        raise IntegrationFailure(
            f"local Judge manifest probe failed with {type(exc).__name__}"
        ) from exc
    try:
        digest = output.decode("ascii").strip()
    except (AttributeError, UnicodeDecodeError) as exc:
        raise IntegrationFailure(
            "local Judge manifest probe returned invalid bytes"
        ) from exc
    if _SHA256.fullmatch(digest) is None:
        raise IntegrationFailure(
            "local Judge manifest probe returned an invalid digest"
        )
    return {
        **environment,
        _APPROVED_JUDGE_ENSEMBLE_MANIFEST_SHA256_ENV: digest,
    }


def _owned_container_identity(
    container: Any,
    *,
    owner_token: str,
    expected_image_id: str,
) -> str:
    """Bind cleanup to the exact container created by this invocation."""

    try:
        container.reload()
        container_id = container.id
        attributes = container.attrs
        labels = attributes.get("Config", {}).get("Labels", {}) or {}
        image_id = attributes.get("Image")
    except Exception as exc:
        raise IntegrationFailure(
            f"Docker container inspection failed with {type(exc).__name__}"
        ) from exc
    if not isinstance(container_id, str) or _SHA256.fullmatch(container_id) is None:
        raise IntegrationFailure("integration container has an invalid immutable ID")
    if not isinstance(labels, dict) or labels.get(_OWNER_LABEL) != owner_token:
        raise IntegrationFailure("integration container ownership label mismatch")
    if (
        not isinstance(image_id, str)
        or _IMAGE_ID.fullmatch(image_id) is None
        or image_id != expected_image_id
    ):
        raise IntegrationFailure("integration container image identity mismatch")
    return container_id


def _cleanup_owned_container(
    client: Any,
    docker_module: Any,
    *,
    container_name: str,
    owner_token: str,
    expected_image_id: str,
    owned_container_id: str | None,
) -> None:
    """Remove only this run's labeled container and never a name replacement."""

    candidate = None
    if owned_container_id is not None:
        candidate = _lookup_container(client, docker_module, owned_container_id)
    if candidate is None:
        by_name = _lookup_container(client, docker_module, container_name)
        if by_name is None:
            return
        if owned_container_id is not None and by_name.id != owned_container_id:
            raise IntegrationFailure(
                "integration container name was reused; replacement was not removed"
            )
        candidate = by_name

    observed_id = _owned_container_identity(
        candidate,
        owner_token=owner_token,
        expected_image_id=expected_image_id,
    )
    if owned_container_id is not None and observed_id != owned_container_id:
        raise IntegrationFailure("integration container immutable ID changed")
    try:
        candidate.remove(force=True)
    except Exception as exc:
        raise IntegrationFailure(
            f"owned integration container cleanup failed with {type(exc).__name__}"
        ) from exc
    if _lookup_container(client, docker_module, observed_id) is not None:
        raise IntegrationFailure("owned integration container remains after cleanup")
    by_name = _lookup_container(client, docker_module, container_name)
    if by_name is not None:
        raise IntegrationFailure(
            "integration container name was reused; replacement was not removed"
        )


def _validate_container_credential_isolation(
    container: Any,
    *,
    secrets: tuple[str, ...],
    expected_judge_environment: dict[str, str],
) -> None:
    try:
        container.reload()
        entries = container.attrs.get("Config", {}).get("Env", []) or []
    except Exception as exc:
        raise IntegrationFailure(
            f"Docker container environment inspection failed with {type(exc).__name__}"
        ) from exc
    if not isinstance(entries, list) or not all(
        isinstance(item, str) for item in entries
    ):
        raise IntegrationFailure("Docker container environment has an invalid shape")
    observed_environment: dict[str, str] = {}
    for entry in entries:
        key, separator, value = entry.partition("=")
        if not separator or key in observed_environment:
            raise IntegrationFailure("Docker container environment has duplicate keys")
        observed_environment[key] = value
    serialized = "\n".join(entries)
    if any(secret and secret in serialized for secret in secrets):
        raise IntegrationFailure(
            "InstructionGym container received endpoint secret material"
        )
    for key, value in expected_judge_environment.items():
        if observed_environment.get(key) != value:
            raise IntegrationFailure(
                "InstructionGym container received an unexpected local Judge identity"
            )
    for key, value in observed_environment.items():
        normalized = _normalized_field_name(key)
        if value and normalized.endswith("apikey") and key != _JUDGE_API_KEY_ENV:
            raise IntegrationFailure(
                "InstructionGym container received a non-Judge API credential"
            )


async def run_integration(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the real local Docker path and return machine-readable evidence."""

    _validate_args(args)
    handoff = deterministic_task_handoff(
        count=args.sampling_count,
        seed=args.sampling_seed,
    )
    success_api_key: str | None = None
    if args.success_api_key_env is not None:
        success_api_key = os.environ.get(args.success_api_key_env)
        if not success_api_key:
            raise IntegrationFailure(
                f"endpoint key variable {args.success_api_key_env!r} is unset or empty"
            )
    endpoint_secrets = tuple(
        value
        for value in (args.success_base_url, success_api_key, _FAILURE_BASE_URL)
        if value
    )
    try:
        import affinetes
        import docker
    except ImportError as exc:
        raise IntegrationFailure("Affinetes and Docker SDK must be installed") from exc

    try:
        docker_client = docker.from_env()
        _validate_local_docker_client(docker_client)
        image = docker_client.images.get(args.image)
        judge_environment = _local_judge_environment(
            docker_client,
            image_id=image.id,
        )
    except Exception as exc:
        if isinstance(exc, IntegrationFailure):
            raise
        raise IntegrationFailure(
            f"local image inspection failed with {type(exc).__name__}"
        ) from exc
    if _lookup_container(docker_client, docker, args.container_name) is not None:
        raise IntegrationFailure(
            "refusing to replace an existing integration container"
        )
    if "instruction-gym" in _ENV_CACHE:
        raise IntegrationFailure(
            "refusing to reuse an existing InstructionGym SDK instance"
        )

    original_config = ENV_CONFIGS["instruction-gym"]
    original_loader = affinetes.load_env
    owner_token = secrets.token_hex(32)
    sdk: SDKEnvironment | None = None
    owned_container_id: str | None = None
    operation_error: BaseException | None = None
    failure_evidence: list[dict[str, Any]] = []
    success_evidence: dict[str, Any] | None = None
    previous_judge_environment = {key: os.environ.get(key) for key in judge_environment}
    result_secrets = (
        *endpoint_secrets,
        judge_environment[_JUDGE_BASE_URL_ENV],
        judge_environment[_JUDGE_API_KEY_ENV],
        judge_environment[_JUDGE_ENSEMBLE_JSON_ENV],
    )

    def local_image_loader(**kwargs: Any) -> Any:
        kwargs.update(
            {
                "image": args.image,
                "pull": False,
                "container_name": args.container_name,
                "hosts": ["localhost"],
                "replicas": 1,
                "force_recreate": False,
                "create_only": True,
                "expected_owner": (_OWNER_LABEL, owner_token),
                "labels": {_OWNER_LABEL: owner_token},
            }
        )
        if args.host_network:
            kwargs.update(
                {
                    "host_network": True,
                    "host_port": args.actor_host_port,
                }
            )
        return original_loader(**kwargs)

    try:
        os.environ.update(judge_environment)
        ENV_CONFIGS["instruction-gym"] = replace(
            original_config,
            docker_image=args.image,
            env_vars={"UVICORN_WORKERS": "1"},
            mem_limit="1g",
            proxy_timeout=int(args.timeout) + 10,
        )
        affinetes.load_env = local_image_loader
        sdk = SDKEnvironment("instruction-gym", mode="docker")
        active_container = _lookup_container(
            docker_client,
            docker,
            args.container_name,
        )
        if active_container is None:
            raise IntegrationFailure("Affinetes did not create the expected container")
        owned_container_id = _owned_container_identity(
            active_container,
            owner_token=owner_token,
            expected_image_id=image.id,
        )
        _validate_container_credential_isolation(
            active_container,
            secrets=endpoint_secrets,
            expected_judge_environment=judge_environment,
        )
        first_task_id = handoff[0]["task_id"]
        second_task_id = handoff[1]["task_id"]

        success_kwargs = {
            "task_id": first_task_id,
            "model": "affine-instruction-gym-e2e",
            "base_url": args.success_base_url,
            "seed": 0,
            "temperature": 0.0,
            "timeout": float(args.timeout),
        }
        if success_api_key is not None:
            success_kwargs["api_key"] = success_api_key
        success_result = await sdk.evaluate(**success_kwargs)
        success_evidence = _validate_success(
            success_result,
            task_id=first_task_id,
            secrets=result_secrets,
        )

        for task_id, model_seed in (
            (first_task_id, 0),
            (first_task_id, 1),
            (second_task_id, 0),
        ):
            result = await sdk.evaluate(
                task_id=task_id,
                model="affine-instruction-gym-e2e",
                base_url=_FAILURE_BASE_URL,
                seed=model_seed,
                temperature=0.0,
                timeout=float(args.timeout),
            )
            failure_evidence.append(
                _validate_endpoint_failure(
                    result,
                    task_id=task_id,
                    secrets=result_secrets,
                )
            )
        if success_evidence["prompt_sha256"] != failure_evidence[0]["prompt_sha256"]:
            raise IntegrationFailure(
                "success and failure paths materialized different prompts"
            )
        if failure_evidence[0]["prompt_sha256"] != failure_evidence[1]["prompt_sha256"]:
            raise IntegrationFailure("model seed changed the materialized prompt")
        if (
            len(
                {row["catalog_sha256"] for row in (success_evidence, *failure_evidence)}
            )
            != 1
        ):
            raise IntegrationFailure(
                "Actor probes returned different catalog identities"
            )
    except (
        BaseException
    ) as exc:  # preserve cleanup across SDK/backend exception families
        operation_error = exc
    finally:
        affinetes.load_env = original_loader
        ENV_CONFIGS["instruction-gym"] = original_config
        for key, previous in previous_judge_environment.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous
        try:
            if sdk is not None:
                await sdk._env.cleanup()
        except BaseException as exc:
            if operation_error is None:
                operation_error = exc
        finally:
            _ENV_CACHE.pop("instruction-gym", None)

    try:
        _cleanup_owned_container(
            docker_client,
            docker,
            container_name=args.container_name,
            owner_token=owner_token,
            expected_image_id=image.id,
            owned_container_id=owned_container_id,
        )
    except BaseException as exc:
        if operation_error is None:
            operation_error = exc
    finally:
        docker_client.close()
    if operation_error is not None:
        if isinstance(operation_error, IntegrationFailure):
            raise operation_error
        if not isinstance(operation_error, Exception):
            raise operation_error
        raise IntegrationFailure(
            f"Affine integration failed with {type(operation_error).__name__}"
        ) from operation_error

    report = {
        "schema_version": "1.2",
        "passed": True,
        "image": args.image,
        "image_id": image.id,
        "sampling_mode": SAMPLING_MODE_RANDOM,
        "sampling_seed_sha256": hashlib.sha256(
            f"instruction-gym-e2e:{args.sampling_seed}".encode()
        ).hexdigest(),
        "scheduler_handoff": handoff,
        "successful_evaluation": success_evidence,
        "failure_evaluations": failure_evidence,
        "successful_scoring_verified": True,
        "local_judge_ensemble_manifest_sha256": judge_environment[
            _APPROVED_JUDGE_ENSEMBLE_MANIFEST_SHA256_ENV
        ],
        "local_judge_configuration_verified": True,
        "container_credential_isolation_verified": True,
        "result_secret_sanitization_verified": True,
        "task_seed_invariance_verified": True,
        "random_scheduler_handoff_verified": True,
        "direct_task_id_handoff_verified": True,
        "cleanup_verified": True,
    }
    return report


def main() -> int:
    args = _parser().parse_args()
    try:
        report = asyncio.run(run_integration(args))
        output = json.dumps(report, sort_keys=True, indent=2) + "\n"
        if args.output is None:
            print(output, end="")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output, encoding="utf-8")
    except (IntegrationFailure, OSError, ValueError) as exc:
        print(f"instruction-gym-e2e: error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
