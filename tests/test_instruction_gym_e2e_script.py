from __future__ import annotations

import importlib.util
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "test_instruction_gym_e2e.py"
SPEC = importlib.util.spec_from_file_location("affine_instruction_gym_e2e", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
e2e = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = e2e
SPEC.loader.exec_module(e2e)

STUB_SCRIPT = ROOT / "scripts" / "openai_stub.py"
STUB_SPEC = importlib.util.spec_from_file_location(
    "affine_instruction_gym_openai_stub", STUB_SCRIPT
)
assert STUB_SPEC is not None and STUB_SPEC.loader is not None
stub = importlib.util.module_from_spec(STUB_SPEC)
sys.modules[STUB_SPEC.name] = stub
STUB_SPEC.loader.exec_module(stub)


def test_scheduler_handoff_reuses_reproducible_direct_random_sampling() -> None:
    first = e2e.deterministic_task_handoff(count=100, seed=20260714)
    second = e2e.deterministic_task_handoff(count=100, seed=20260714)

    assert first == second
    assert len(first) == 100
    assert len({row["task_id"] for row in first}) == 100
    assert all(0 <= row["task_id"] < e2e.INSTRUCTION_GYM_TASK_ID_END for row in first)


def _result(*, task_id: int, request: dict | None = None):
    return SimpleNamespace(
        success=False,
        score=0.0,
        error="model endpoint connection failed",
        task_id=task_id,
        extra={
            "error_code": "endpoint_connection_failed",
            "prompt_sha256": "a" * 64,
            "catalog_sha256": "b" * 64,
            "request": {"task_id": task_id} if request is None else request,
        },
    )


def _successful_result(*, task_id: int, request: dict | None = None):
    prompt = "materialized prompt"
    response = "stub response"
    prompt_sha256 = hashlib.sha256(prompt.encode()).hexdigest()
    response_sha256 = hashlib.sha256(response.encode()).hexdigest()
    normalized_response_sha256 = "c" * 64
    return SimpleNamespace(
        success=True,
        score=0.0,
        error=None,
        task_id=task_id,
        extra={
            "case_pass": False,
            "prompt_sha256": prompt_sha256,
            "response_sha256": response_sha256,
            "normalized_response_sha256": normalized_response_sha256,
            "catalog_sha256": "b" * 64,
            "constraint_results": [{"strict_pass": False}],
            "scoring_evidence": {
                "prompt_sha256": prompt_sha256,
                "response_sha256": response_sha256,
                "normalized_response_sha256": normalized_response_sha256,
                "constraint_result_count": 1,
            },
            "conversation": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
            "request": {"task_id": task_id} if request is None else request,
        },
    )


def test_endpoint_failure_contract_retains_task_identity_without_routing_secrets() -> (
    None
):
    assert e2e._validate_endpoint_failure(_result(task_id=7), task_id=7) == {
        "task_id": 7,
        "prompt_sha256": "a" * 64,
        "catalog_sha256": "b" * 64,
        "error_code": "endpoint_connection_failed",
    }


@pytest.mark.parametrize(
    "request_payload",
    (
        {"task_id": 7, "base_url": "https://secret.invalid"},
        {"task_id": 7, "api_key": "secret"},
        {"task_id": 7, "headers": {"Authorization": "Bearer secret"}},
    ),
)
def test_endpoint_failure_contract_rejects_persisted_secrets(
    request_payload: dict,
) -> None:
    with pytest.raises(e2e.IntegrationFailure, match="routing or credential"):
        e2e._validate_endpoint_failure(
            _result(task_id=7, request=request_payload), task_id=7
        )


def test_success_contract_requires_coherent_real_scoring_evidence() -> None:
    result = _successful_result(task_id=7)

    evidence = e2e._validate_success(
        result,
        task_id=7,
        secrets=("https://endpoint.invalid/v1", "per-call-secret"),
    )

    assert evidence["task_id"] == 7
    assert evidence["score"] == 0.0
    assert evidence["case_pass"] is False
    assert evidence["constraint_result_count"] == 1
    assert evidence["response_sha256"] == result.extra["response_sha256"]


def test_success_contract_rejects_secret_material_outside_request() -> None:
    result = _successful_result(task_id=7)
    result.extra["endpoint_debug"] = "per-call-secret"

    with pytest.raises(e2e.IntegrationFailure, match="secret material"):
        e2e._validate_success(
            result,
            task_id=7,
            secrets=("per-call-secret",),
        )


@pytest.mark.parametrize("count", (0, 1, 1_001))
def test_e2e_arguments_bound_sampling_count(count: int) -> None:
    args = e2e._parser().parse_args(
        [
            "--image",
            "instruction-gym:test",
            "--sampling-count",
            str(count),
            "--success-base-url",
            "https://stub.invalid/v1",
        ]
    )
    with pytest.raises(e2e.IntegrationFailure, match="sampling-count"):
        e2e._validate_args(args)


def test_e2e_arguments_require_host_network_for_local_stub() -> None:
    args = e2e._parser().parse_args(
        [
            "--image",
            "instruction-gym:test",
            "--success-base-url",
            "http://127.0.0.1:19000/v1",
        ]
    )

    with pytest.raises(e2e.IntegrationFailure, match="host-network"):
        e2e._validate_args(args)


@pytest.mark.parametrize(
    "base_url",
    (
        "https://model.example/v1",
        "http://192.0.2.1:19000/v1",
        "http://localhost.example:19000/v1",
    ),
)
def test_e2e_arguments_reject_remote_success_endpoints(base_url: str) -> None:
    args = e2e._parser().parse_args(
        [
            "--image",
            "instruction-gym:test",
            "--success-base-url",
            base_url,
            "--host-network",
        ]
    )

    with pytest.raises(e2e.IntegrationFailure, match="loopback"):
        e2e._validate_args(args)


@pytest.mark.parametrize(
    ("base_url", "accepted"),
    (
        ("http+docker://localhost", True),
        ("http://127.0.0.1:2375", True),
        ("http://[::1]:2375", True),
        ("http://docker.example:2375", False),
        ("ssh://docker.example", False),
    ),
)
def test_e2e_rejects_nonlocal_docker_daemon(
    base_url: str,
    accepted: bool,
) -> None:
    client = SimpleNamespace(api=SimpleNamespace(base_url=base_url))

    if accepted:
        e2e._validate_local_docker_client(client)
    else:
        with pytest.raises(e2e.IntegrationFailure, match="non-loopback Docker"):
            e2e._validate_local_docker_client(client)


def test_cleanup_does_not_remove_same_name_replacement() -> None:
    class NotFound(Exception):
        pass

    replacement = SimpleNamespace(id="b" * 64)

    class Containers:
        @staticmethod
        def get(reference: str):
            if reference == "a" * 64:
                raise NotFound()
            assert reference == "instruction-gym-e2e"
            return replacement

    client = SimpleNamespace(containers=Containers())
    docker_module = SimpleNamespace(errors=SimpleNamespace(NotFound=NotFound))

    with pytest.raises(e2e.IntegrationFailure, match="replacement was not removed"):
        e2e._cleanup_owned_container(
            client,
            docker_module,
            container_name="instruction-gym-e2e",
            owner_token="c" * 64,
            expected_image_id="sha256:" + "d" * 64,
            owned_container_id="a" * 64,
        )
    assert not hasattr(replacement, "removed")


@pytest.mark.parametrize(
    ("host", "accepted"),
    (
        ("127.0.0.1", True),
        ("::1", True),
        ("localhost", True),
        ("0.0.0.0", False),
        ("model.example", False),
    ),
)
def test_openai_stub_is_loopback_only(host: str, accepted: bool) -> None:
    assert stub._is_loopback_host(host) is accepted
