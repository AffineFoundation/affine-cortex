import asyncio
import json
from typing import Any, Iterable

import httpx
import pytest

from affine.src.behavior_guard import probe as probe_module
from affine.src.behavior_guard.models import ProbeClassification
from affine.src.behavior_guard.probe import BehaviorProbeClient, ProbeConfig


BASE_URL = "https://challenger.test/v1"
MODEL = "owner/challenger"
NONCE = "nonce-test-1234"


def _event(
    delta: dict[str, Any],
    *,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "id": "chatcmpl-probe",
        "object": "chat.completion.chunk",
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def _sse(events: Iterable[dict[str, Any] | str], *, done: bool = True) -> bytes:
    frames: list[bytes] = []
    for event in events:
        payload = event if isinstance(event, str) else json.dumps(event)
        frames.append(f"data: {payload}\n\n".encode())
    if done:
        frames.append(b"data: [DONE]\n\n")
    return b"".join(frames)


def _response(content: bytes) -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream; charset=utf-8"},
        content=content,
    )


class _DelayedStream(httpx.AsyncByteStream):
    def __init__(self, delay: float, chunks: Iterable[bytes]):
        self.delay = delay
        self.chunks = list(chunks)

    async def __aiter__(self):
        for chunk in self.chunks:
            await asyncio.sleep(self.delay)
            yield chunk

    async def aclose(self) -> None:
        return None


class _ScriptedStream(httpx.AsyncByteStream):
    def __init__(self, chunks: Iterable[tuple[float, bytes]]):
        self.chunks = list(chunks)

    async def __aiter__(self):
        for delay, chunk in self.chunks:
            if delay:
                await asyncio.sleep(delay)
            yield chunk

    async def aclose(self) -> None:
        return None


class _HangingCloseStream(httpx.AsyncByteStream):
    def __init__(self, content: bytes):
        self.content = content
        self.close_cancelled = False

    async def __aiter__(self):
        yield self.content

    async def aclose(self) -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.close_cancelled = True
            raise


class _HangingCloseClient:
    def __init__(self):
        self.close_cancelled = False

    async def aclose(self) -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.close_cancelled = True
            raise


class _CancellingCloseStream(httpx.AsyncByteStream):
    def __init__(self, content: bytes):
        self.content = content

    async def __aiter__(self):
        yield self.content

    async def aclose(self) -> None:
        raise asyncio.CancelledError


@pytest.mark.asyncio
async def test_control_probe_accepts_exact_nonce_and_caps_server_tokens():
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return _response(
            _sse(
                [
                    _event({"role": "assistant"}),
                    _event({"content": NONCE}),
                    _event({}, finish_reason="stop"),
                    {"object": "chat.completion.chunk", "choices": [], "usage": {"completion_tokens": 4}},
                ]
            )
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        probe = BehaviorProbeClient(
            base_url=BASE_URL,
            model=MODEL,
            config=ProbeConfig(max_completion_tokens=17),
            client=http,
        )
        result = await probe.run_control_probe(nonce=NONCE, probe_id="control-1")

    assert result.classification is ProbeClassification.CLEAN
    assert result.reason == "control_nonce_matched"
    assert result.completion_tokens == 4
    assert result.output_bytes == len(NONCE)
    assert result.first_response_ms is not None
    assert result.first_action_ms is None
    assert captured["stream"] is True
    assert captured["stream_options"] == {"include_usage": True}
    assert captured["max_tokens"] == 17
    assert "tools" not in captured


@pytest.mark.asyncio
async def test_empty_stop_is_model_protocol_failure_without_raw_output():
    def handler(_request: httpx.Request) -> httpx.Response:
        return _response(_sse([_event({}, finish_reason="stop")]))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await BehaviorProbeClient(
            base_url=BASE_URL,
            model=MODEL,
            client=http,
        ).run_control_probe(nonce=NONCE, probe_id="empty-stop")

    assert result.classification is ProbeClassification.MODEL_PROTOCOL_FAILURE
    assert result.reason == "empty_stop"
    assert result.output_bytes == 0
    assert len(result.evidence_hash) == 64
    assert NONCE not in repr(result)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "reason"),
    [
        (b"data: {not-json}\n\n", "malformed_sse_json"),
        (
            _sse(
                [
                    _event({"content": NONCE}),
                    _event({}, finish_reason="stop"),
                ],
                done=False,
            ),
            "stream_missing_done",
        ),
    ],
)
async def test_malformed_or_incomplete_sse_is_model_protocol_failure(
    body: bytes,
    reason: str,
):
    def handler(_request: httpx.Request) -> httpx.Response:
        return _response(body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await BehaviorProbeClient(
            base_url=BASE_URL,
            model=MODEL,
            client=http,
        ).run_control_probe(nonce=NONCE, probe_id="bad-sse")

    assert result.classification is ProbeClassification.MODEL_PROTOCOL_FAILURE
    assert result.reason == reason


@pytest.mark.asyncio
async def test_nonce_action_accepts_fragmented_openai_tool_call():
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        tool_name = captured["tools"][0]["function"]["name"]
        return _response(
            _sse(
                [
                    _event({"role": "assistant"}),
                    _event(
                        {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_probe",
                                    "type": "function",
                                    "function": {
                                        "name": tool_name,
                                        "arguments": '{"non',
                                    },
                                }
                            ]
                        }
                    ),
                    _event(
                        {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "arguments": f'ce":"{NONCE}"}}',
                                    },
                                }
                            ]
                        },
                        finish_reason="tool_calls",
                    ),
                    {"object": "chat.completion.chunk", "choices": [], "usage": {"completion_tokens": 11}},
                ]
            )
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await BehaviorProbeClient(
            base_url=BASE_URL,
            model=MODEL,
            client=http,
        ).run_action_probe(nonce=NONCE, probe_id="action-ok")

    assert result.classification is ProbeClassification.CLEAN
    assert result.reason == "tool_nonce_matched"
    assert result.first_response_ms is not None
    assert result.first_action_ms is not None
    assert result.completion_tokens == 11
    assert captured["parallel_tool_calls"] is False
    assert captured["tool_choice"]["function"]["name"] == captured["tools"][0]["function"]["name"]


@pytest.mark.asyncio
async def test_malformed_tool_arguments_are_model_protocol_failure():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        tool_name = body["tools"][0]["function"]["name"]
        return _response(
            _sse(
                [
                    _event(
                        {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_bad",
                                    "type": "function",
                                    "function": {
                                        "name": tool_name,
                                        "arguments": '{"nonce":',
                                    },
                                }
                            ]
                        },
                        finish_reason="tool_calls",
                    )
                ]
            )
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await BehaviorProbeClient(
            base_url=BASE_URL,
            model=MODEL,
            client=http,
        ).run_action_probe(nonce=NONCE, probe_id="action-malformed")

    assert result.classification is ProbeClassification.MODEL_PROTOCOL_FAILURE
    assert result.reason == "malformed_tool_arguments"


@pytest.mark.asyncio
async def test_bounded_refusal_or_no_tool_call_is_quality_failure():
    refusal = "I cannot perform that action."

    def handler(_request: httpx.Request) -> httpx.Response:
        return _response(
            _sse(
                [
                    _event({"refusal": refusal}),
                    _event({}, finish_reason="stop"),
                ]
            )
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await BehaviorProbeClient(
            base_url=BASE_URL,
            model=MODEL,
            client=http,
        ).run_action_probe(nonce=NONCE, probe_id="bounded-refusal")

    assert result.classification is ProbeClassification.QUALITY_FAILURE
    assert result.reason == "bounded_refusal"
    assert result.is_admissible_completion is True
    assert refusal not in repr(result)


@pytest.mark.asyncio
async def test_forced_action_that_stops_without_tool_call_is_quality_failure():
    def handler(_request: httpx.Request) -> httpx.Response:
        return _response(
            _sse(
                [
                    _event({"content": "The action is complete."}),
                    _event({}, finish_reason="stop"),
                ]
            )
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await BehaviorProbeClient(
            base_url=BASE_URL,
            model=MODEL,
            client=http,
        ).run_action_probe(nonce=NONCE, probe_id="missing-tool")

    assert result.classification is ProbeClassification.QUALITY_FAILURE
    assert result.reason == "tool_action_not_called"
    assert result.is_strike is False


@pytest.mark.asyncio
async def test_first_response_deadline_after_http_headers_is_model_no_progress():
    config = ProbeConfig(
        first_response_timeout_s=0.01,
        first_action_timeout_s=0.04,
        total_timeout_s=0.05,
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_DelayedStream(0.1, [_sse([_event({"content": NONCE}, finish_reason="stop")])]),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await BehaviorProbeClient(
            base_url=BASE_URL,
            model=MODEL,
            config=config,
            client=http,
        ).run_control_probe(nonce=NONCE, probe_id="slow-first-response")

    assert result.classification is ProbeClassification.MODEL_NO_PROGRESS
    assert result.reason == "first_response_deadline_exceeded"
    assert result.first_response_ms is None


@pytest.mark.asyncio
async def test_first_action_and_total_deadlines_are_independently_enforced():
    no_action_config = ProbeConfig(
        first_response_timeout_s=0.01,
        first_action_timeout_s=0.02,
        total_timeout_s=0.2,
    )

    def no_action_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_ScriptedStream(
                [
                    (0, _sse([_event({"content": "thinking"})], done=False)),
                    (0.1, _sse([_event({}, finish_reason="stop")])),
                ]
            ),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(no_action_handler)) as http:
        no_action = await BehaviorProbeClient(
            base_url=BASE_URL,
            model=MODEL,
            config=no_action_config,
            client=http,
        ).run_action_probe(nonce=NONCE, probe_id="no-first-action")

    assert no_action.classification is ProbeClassification.MODEL_NO_PROGRESS
    assert no_action.reason == "first_action_deadline_exceeded"
    assert no_action.first_response_ms is not None
    assert no_action.first_action_ms is None

    total_config = ProbeConfig(
        first_response_timeout_s=0.01,
        first_action_timeout_s=0.02,
        total_timeout_s=0.04,
    )

    def total_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        tool_name = body["tools"][0]["function"]["name"]
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_ScriptedStream(
                [
                    (
                        0,
                        _sse(
                            [
                                _event(
                                    {
                                        "tool_calls": [
                                            {
                                                "index": 0,
                                                "id": "call_stall",
                                                "type": "function",
                                                "function": {
                                                    "name": tool_name,
                                                    "arguments": "",
                                                },
                                            }
                                        ]
                                    }
                                )
                            ],
                            done=False,
                        ),
                    ),
                    (0.1, _sse([_event({}, finish_reason="tool_calls")])),
                ]
            ),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(total_handler)) as http:
        total = await BehaviorProbeClient(
            base_url=BASE_URL,
            model=MODEL,
            config=total_config,
            client=http,
        ).run_action_probe(nonce=NONCE, probe_id="total-timeout")

    assert total.classification is ProbeClassification.MODEL_NO_PROGRESS
    assert total.reason == "total_deadline_exceeded"
    assert total.first_response_ms is not None
    assert total.first_action_ms is not None


@pytest.mark.asyncio
async def test_transport_timeout_and_http_error_are_infra_failures():
    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("secret upstream detail", request=request)

    def http_error_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, content=b"secret provider response")

    async with httpx.AsyncClient(transport=httpx.MockTransport(timeout_handler)) as http:
        timeout_result = await BehaviorProbeClient(
            base_url=BASE_URL,
            model=MODEL,
            client=http,
        ).run_control_probe(nonce=NONCE, probe_id="transport-timeout")
    async with httpx.AsyncClient(transport=httpx.MockTransport(http_error_handler)) as http:
        http_result = await BehaviorProbeClient(
            base_url=BASE_URL,
            model=MODEL,
            client=http,
        ).run_control_probe(nonce=NONCE, probe_id="http-error")

    assert timeout_result.classification is ProbeClassification.INFRA_FAILURE
    assert timeout_result.reason == "request_transport_error"
    assert "secret" not in repr(timeout_result)
    assert http_result.classification is ProbeClassification.INFRA_FAILURE
    assert http_result.reason == "http_status_503"
    assert "secret" not in repr(http_result)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 422])
async def test_deterministic_action_schema_rejection_is_protocol_failure(status):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=b"secret tool parser detail")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await BehaviorProbeClient(
            base_url=BASE_URL,
            model=MODEL,
            client=http,
        ).run_action_probe(nonce=NONCE, probe_id=f"action-http-{status}")

    assert result.classification is ProbeClassification.MODEL_PROTOCOL_FAILURE
    assert result.reason == f"action_http_status_{status}"
    assert "secret" not in repr(result)


@pytest.mark.asyncio
async def test_client_output_budget_stops_stream_without_storing_content():
    generated = "sensitive-junk-" * 20

    def handler(_request: httpx.Request) -> httpx.Response:
        return _response(_sse([_event({"content": generated})], done=False))

    config = ProbeConfig(max_output_bytes=32, max_wire_bytes=1_024)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await BehaviorProbeClient(
            base_url=BASE_URL,
            model=MODEL,
            config=config,
            client=http,
        ).run_control_probe(nonce=NONCE, probe_id="output-budget")

    assert result.classification is ProbeClassification.MODEL_NO_PROGRESS
    assert result.reason == "model_output_budget_exceeded"
    assert generated not in repr(result)


@pytest.mark.asyncio
async def test_many_data_lines_hit_incremental_event_budget():
    body = (b"data:\n" * 1_000) + b"data: {}\n\n"

    def handler(_request: httpx.Request) -> httpx.Response:
        return _response(body)

    config = ProbeConfig(
        max_wire_bytes=64 * 1024,
        max_sse_event_bytes=256,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await BehaviorProbeClient(
            base_url=BASE_URL,
            model=MODEL,
            config=config,
            client=http,
        ).run_control_probe(nonce=NONCE, probe_id="many-data-lines")

    assert result.classification is ProbeClassification.MODEL_NO_PROGRESS
    assert result.reason == "sse_event_budget_exceeded"


@pytest.mark.asyncio
async def test_large_chunk_checks_deadline_before_parsing_later_malformed_line(
    monkeypatch: pytest.MonkeyPatch,
):
    monotonic_calls = 0

    def fake_monotonic() -> float:
        nonlocal monotonic_calls
        monotonic_calls += 1
        return 0.0 if monotonic_calls <= 2 else 2.0

    monkeypatch.setattr(probe_module.time, "monotonic", fake_monotonic)
    body = (b": keepalive\n" * 64) + b"malformed-after-deadline\n"

    def handler(_request: httpx.Request) -> httpx.Response:
        return _response(body)

    config = ProbeConfig(
        first_response_timeout_s=0.5,
        first_action_timeout_s=0.75,
        total_timeout_s=1.0,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await BehaviorProbeClient(
            base_url=BASE_URL,
            model=MODEL,
            config=config,
            client=http,
        ).run_control_probe(nonce=NONCE, probe_id="chunk-deadline")

    assert result.classification is ProbeClassification.MODEL_NO_PROGRESS
    assert result.reason == "first_response_deadline_exceeded"


@pytest.mark.asyncio
async def test_hanging_response_and_owned_client_cleanup_are_bounded():
    stream = _HangingCloseStream(
        _sse(
            [
                _event({"content": NONCE}),
                _event({}, finish_reason="stop"),
            ]
        )
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=stream,
        )

    config = ProbeConfig(cleanup_timeout_s=0.01)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await asyncio.wait_for(
            BehaviorProbeClient(
                base_url=BASE_URL,
                model=MODEL,
                config=config,
                client=http,
            ).run_control_probe(nonce=NONCE, probe_id="hanging-response-close"),
            timeout=0.2,
        )

    assert result.classification is ProbeClassification.CLEAN
    assert stream.close_cancelled is True

    client = _HangingCloseClient()
    probe = BehaviorProbeClient(
        base_url=BASE_URL,
        model=MODEL,
        config=config,
        client=client,  # type: ignore[arg-type]
    )
    probe._owns_client = True
    await asyncio.wait_for(probe.aclose(), timeout=0.2)

    assert client.close_cancelled is True


@pytest.mark.asyncio
async def test_cleanup_does_not_swallow_cancellation():
    stream = _CancellingCloseStream(
        _sse(
            [
                _event({"content": NONCE}),
                _event({}, finish_reason="stop"),
            ]
        )
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=stream,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(asyncio.CancelledError):
            await BehaviorProbeClient(
                base_url=BASE_URL,
                model=MODEL,
                client=http,
            ).run_control_probe(nonce=NONCE, probe_id="cancelled-response-close")


@pytest.mark.asyncio
async def test_suite_runs_control_then_action_with_shared_nonce():
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if "tools" not in body:
            return _response(
                _sse(
                    [
                        _event({"content": NONCE}),
                        _event({}, finish_reason="stop"),
                    ]
                )
            )
        tool_name = body["tools"][0]["function"]["name"]
        return _response(
            _sse(
                [
                    _event(
                        {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_suite",
                                    "type": "function",
                                    "function": {
                                        "name": tool_name,
                                        "arguments": json.dumps({"nonce": NONCE}),
                                    },
                                }
                            ]
                        },
                        finish_reason="tool_calls",
                    )
                ]
            )
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        results = await BehaviorProbeClient(
            base_url=BASE_URL,
            model=MODEL,
            client=http,
        ).run_suite(nonce=NONCE, probe_id_prefix="suite")

    assert [result.probe_id for result in results] == ["suite-control", "suite-action"]
    assert [result.classification for result in results] == [
        ProbeClassification.CLEAN,
        ProbeClassification.CLEAN,
    ]
    assert "tools" not in seen[0]
    assert "tools" in seen[1]
