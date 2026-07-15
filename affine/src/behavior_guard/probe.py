"""Bounded, CPU-side probes for OpenAI-compatible challenger endpoints.

The probe intentionally talks to the inference endpoint directly instead of
creating an affinetes environment.  It exercises two small contracts:

* a tool-free control response containing an unpredictable nonce; and
* a forced tool call whose JSON arguments must return that nonce.

Only counters, timing information, reason codes, and a hash of that sanitized
evidence leave this module.  Assistant content and tool arguments are held in
bounded in-memory buffers only long enough to validate the response.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

import httpx

from affine.src.behavior_guard.models import ProbeClassification, ProbeResult


_NONCE_RE = re.compile(r"^[A-Za-z0-9_.:-]{8,128}$")
_TOOL_NAME = "affine_submit_probe_nonce"
_ALLOWED_FINISH_REASONS = {"stop", "tool_calls", "length", "content_filter"}
_DEADLINE_CHECK_INTERVAL_LINES = 64


class ProbeMode(str, Enum):
    """The two endpoint contracts exercised by :class:`BehaviorProbeClient`."""

    CONTROL = "control"
    NONCE_ACTION = "nonce_action"


@dataclass(frozen=True)
class ProbeConfig:
    """Resource and liveness limits for one probe request.

    ``max_completion_tokens`` is sent to the server as ``max_tokens``.  The
    byte and event limits are independent client-side guards, so an endpoint
    that ignores the token limit still cannot stream indefinitely.
    """

    first_response_timeout_s: float = 30.0
    first_action_timeout_s: float = 90.0
    total_timeout_s: float = 120.0
    max_completion_tokens: int = 512
    max_output_bytes: int = 32 * 1024
    max_wire_bytes: int = 256 * 1024
    max_sse_line_bytes: int = 256 * 1024
    max_sse_event_bytes: int = 256 * 1024
    max_sse_events: int = 2_048
    max_tool_argument_bytes: int = 4 * 1024
    cleanup_timeout_s: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "first_response_timeout_s",
            "first_action_timeout_s",
            "total_timeout_s",
            "cleanup_timeout_s",
        ):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.first_response_timeout_s > self.total_timeout_s:
            raise ValueError("first_response_timeout_s cannot exceed total_timeout_s")
        if self.first_action_timeout_s < self.first_response_timeout_s:
            raise ValueError(
                "first_action_timeout_s cannot be smaller than first_response_timeout_s"
            )
        if self.first_action_timeout_s > self.total_timeout_s:
            raise ValueError("first_action_timeout_s cannot exceed total_timeout_s")
        for name in (
            "max_completion_tokens",
            "max_output_bytes",
            "max_wire_bytes",
            "max_sse_line_bytes",
            "max_sse_event_bytes",
            "max_sse_events",
            "max_tool_argument_bytes",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_wire_bytes < self.max_output_bytes:
            raise ValueError("max_wire_bytes cannot be smaller than max_output_bytes")


@dataclass
class _ToolCall:
    call_id: str = ""
    call_type: str = ""
    name: str = ""
    arguments: str = ""


@dataclass
class _StreamState:
    start: float
    first_response_at: Optional[float] = None
    first_action_at: Optional[float] = None
    finish_reason: Optional[str] = None
    saw_done: bool = False
    visible_content: list[str] = field(default_factory=list)
    saw_reasoning: bool = False
    saw_refusal: bool = False
    tool_calls: dict[int, _ToolCall] = field(default_factory=dict)
    completion_tokens: int = 0
    output_bytes: int = 0
    wire_bytes: int = 0
    sse_events: int = 0


class _ProbeAbort(Exception):
    def __init__(self, classification: ProbeClassification, reason: str):
        super().__init__(reason)
        self.classification = classification
        self.reason = reason


class BehaviorProbeClient:
    """Run bounded streaming probes against an OpenAI-compatible base URL.

    ``base_url`` normally ends in ``/v1``.  A caller may inject an
    :class:`httpx.AsyncClient`, which makes transport ownership explicit and
    permits ``httpx.MockTransport`` in tests.  Injected clients are never
    closed by this class.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        config: Optional[ProbeConfig] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        if not base_url or not base_url.strip():
            raise ValueError("base_url is required")
        if not model or not model.strip():
            raise ValueError("model is required")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.config = config or ProbeConfig()
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=None, follow_redirects=False)

    async def __aenter__(self) -> "BehaviorProbeClient":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._bounded_aclose(self._client)

    async def run_suite(
        self,
        *,
        nonce: Optional[str] = None,
        probe_id_prefix: Optional[str] = None,
    ) -> tuple[ProbeResult, ProbeResult]:
        """Run the control and nonce-action probes sequentially.

        Sequential execution is deliberate: this is an admission probe and
        must not create the request fan-out that it is intended to prevent.
        """

        safe_nonce = _validated_nonce(nonce)
        prefix = probe_id_prefix or f"behavior-{secrets.token_hex(8)}"
        control = await self.run_control_probe(
            nonce=safe_nonce,
            probe_id=f"{prefix}-control",
        )
        action = await self.run_action_probe(
            nonce=safe_nonce,
            probe_id=f"{prefix}-action",
        )
        return control, action

    async def run_control_probe(
        self,
        *,
        nonce: Optional[str] = None,
        probe_id: Optional[str] = None,
    ) -> ProbeResult:
        safe_nonce = _validated_nonce(nonce)
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "This is a protocol liveness check. Follow the requested "
                        "response format exactly and do not add commentary."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Reply with exactly this nonce: {safe_nonce}",
                },
            ],
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": 0,
            "max_tokens": self.config.max_completion_tokens,
        }
        return await self._run_probe(
            mode=ProbeMode.CONTROL,
            probe_id=probe_id or f"control-{secrets.token_hex(8)}",
            nonce=safe_nonce,
            body=body,
        )

    async def run_action_probe(
        self,
        *,
        nonce: Optional[str] = None,
        probe_id: Optional[str] = None,
    ) -> ProbeResult:
        safe_nonce = _validated_nonce(nonce)
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "This is a tool-protocol liveness check. Call the provided "
                        "tool exactly once; do not answer with ordinary text."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Submit this nonce with the provided tool: {safe_nonce}",
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": _TOOL_NAME,
                        "description": "Return the admission-probe nonce.",
                        "parameters": {
                            "type": "object",
                            "properties": {"nonce": {"type": "string"}},
                            "required": ["nonce"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
            "tool_choice": {
                "type": "function",
                "function": {"name": _TOOL_NAME},
            },
            "parallel_tool_calls": False,
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": 0,
            "max_tokens": self.config.max_completion_tokens,
        }
        return await self._run_probe(
            mode=ProbeMode.NONCE_ACTION,
            probe_id=probe_id or f"action-{secrets.token_hex(8)}",
            nonce=safe_nonce,
            body=body,
        )

    async def _run_probe(
        self,
        *,
        mode: ProbeMode,
        probe_id: str,
        nonce: str,
        body: Mapping[str, Any],
    ) -> ProbeResult:
        state = _StreamState(start=time.monotonic())
        response: Optional[httpx.Response] = None
        classification = ProbeClassification.INFRA_FAILURE
        reason = "request_failed"

        try:
            request = self._client.build_request(
                "POST",
                self._chat_completions_url(),
                headers=self._headers(),
                json=dict(body),
                timeout=None,
            )
            open_timeout = min(
                self.config.first_response_timeout_s,
                self.config.total_timeout_s,
            )
            try:
                response = await asyncio.wait_for(
                    self._client.send(request, stream=True),
                    timeout=open_timeout,
                )
            except asyncio.TimeoutError:
                raise _ProbeAbort(
                    ProbeClassification.INFRA_FAILURE,
                    "request_headers_timeout",
                )
            except (httpx.TimeoutException, httpx.TransportError, OSError):
                raise _ProbeAbort(
                    ProbeClassification.INFRA_FAILURE,
                    "request_transport_error",
                )

            if response.status_code < 200 or response.status_code >= 300:
                if (
                    mode is ProbeMode.NONCE_ACTION
                    and response.status_code in {400, 422}
                ):
                    raise _ProbeAbort(
                        ProbeClassification.MODEL_PROTOCOL_FAILURE,
                        f"action_http_status_{response.status_code}",
                    )
                raise _ProbeAbort(
                    ProbeClassification.INFRA_FAILURE,
                    f"http_status_{response.status_code}",
                )
            content_type = response.headers.get("content-type", "").lower()
            if "text/event-stream" not in content_type:
                raise _ProbeAbort(
                    ProbeClassification.MODEL_PROTOCOL_FAILURE,
                    "invalid_content_type",
                )

            await self._consume_sse(response, state=state, mode=mode)
            classification, reason = self._classify_completed(
                mode=mode,
                nonce=nonce,
                state=state,
            )
        except _ProbeAbort as exc:
            classification, reason = exc.classification, exc.reason
        except asyncio.CancelledError:
            raise
        except (httpx.TimeoutException, httpx.TransportError, OSError):
            classification, reason = (
                ProbeClassification.INFRA_FAILURE,
                "response_transport_error",
            )
        finally:
            if response is not None:
                await self._bounded_aclose(response)

        return _build_result(
            probe_id=probe_id,
            mode=mode,
            state=state,
            classification=classification,
            reason=reason,
        )

    async def _consume_sse(
        self,
        response: httpx.Response,
        *,
        state: _StreamState,
        mode: ProbeMode,
    ) -> None:
        chunks = response.aiter_bytes().__aiter__()
        line_buffer = bytearray()
        event_data = bytearray()
        event_has_data = False
        lines_since_deadline_check = 0

        def append_event_data(value: bytes) -> None:
            nonlocal event_has_data

            additional_bytes = len(value) + (1 if event_has_data else 0)
            if len(event_data) + additional_bytes > self.config.max_sse_event_bytes:
                raise _ProbeAbort(
                    ProbeClassification.MODEL_NO_PROGRESS,
                    "sse_event_budget_exceeded",
                )
            if event_has_data:
                event_data.append(ord("\n"))
            event_data.extend(value)
            event_has_data = True

        def process_line(raw_line: bytes) -> None:
            nonlocal event_has_data

            if raw_line.endswith(b"\r"):
                raw_line = raw_line[:-1]
            if len(raw_line) > self.config.max_sse_line_bytes:
                raise _ProbeAbort(
                    ProbeClassification.MODEL_NO_PROGRESS,
                    "sse_line_budget_exceeded",
                )
            if not raw_line:
                if event_has_data:
                    self._raise_if_deadline_elapsed(state=state, mode=mode)
                    self._process_sse_event(
                        bytes(event_data),
                        state=state,
                        mode=mode,
                    )
                    event_data.clear()
                    event_has_data = False
                return
            if raw_line.startswith(b":"):
                return
            if raw_line.startswith(b"data:"):
                value = raw_line[5:]
                if value.startswith(b" "):
                    value = value[1:]
                append_event_data(value)
                return
            # Standard SSE metadata is harmless, but an arbitrary line is
            # not an OpenAI-compatible streaming frame.
            if raw_line.startswith((b"event:", b"id:", b"retry:")):
                return
            raise _ProbeAbort(
                ProbeClassification.MODEL_PROTOCOL_FAILURE,
                "malformed_sse_line",
            )

        while not state.saw_done:
            timeout, timeout_reason = self._next_wait(state=state, mode=mode)
            try:
                chunk = await asyncio.wait_for(anext(chunks), timeout=timeout)
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                raise _ProbeAbort(
                    ProbeClassification.MODEL_NO_PROGRESS,
                    timeout_reason,
                )

            state.wire_bytes += len(chunk)
            if state.wire_bytes > self.config.max_wire_bytes:
                raise _ProbeAbort(
                    ProbeClassification.MODEL_NO_PROGRESS,
                    "wire_output_budget_exceeded",
                )
            line_buffer.extend(chunk)

            line_start = 0
            while True:
                newline = line_buffer.find(b"\n", line_start)
                if newline < 0:
                    break
                process_line(bytes(line_buffer[line_start:newline]))
                line_start = newline + 1

                if state.saw_done:
                    return
                lines_since_deadline_check += 1
                if lines_since_deadline_check >= _DEADLINE_CHECK_INTERVAL_LINES:
                    self._raise_if_deadline_elapsed(state=state, mode=mode)
                    lines_since_deadline_check = 0

            # Compact once per received chunk rather than deleting the head for
            # every line.  The unconsumed suffix is at most one bounded SSE line.
            if line_start:
                del line_buffer[:line_start]
            max_buffered_line = self.config.max_sse_line_bytes + (
                1 if line_buffer.endswith(b"\r") else 0
            )
            if len(line_buffer) > max_buffered_line:
                raise _ProbeAbort(
                    ProbeClassification.MODEL_NO_PROGRESS,
                    "unterminated_sse_line_budget_exceeded",
                )

            self._raise_if_deadline_elapsed(state=state, mode=mode)

        # Be tolerant of a final SSE event without its trailing blank line,
        # while still requiring an explicit OpenAI ``[DONE]`` marker below.
        if not state.saw_done and line_buffer:
            raw_line = bytes(line_buffer)
            if raw_line.endswith(b"\r"):
                raw_line = raw_line[:-1]
            if len(raw_line) > self.config.max_sse_line_bytes:
                raise _ProbeAbort(
                    ProbeClassification.MODEL_NO_PROGRESS,
                    "sse_line_budget_exceeded",
                )
            if raw_line.startswith(b"data:"):
                value = raw_line[5:]
                if value.startswith(b" "):
                    value = value[1:]
                append_event_data(value)
            else:
                raise _ProbeAbort(
                    ProbeClassification.MODEL_PROTOCOL_FAILURE,
                    "truncated_sse_line",
                )
        if not state.saw_done and event_has_data:
            self._process_sse_event(
                bytes(event_data),
                state=state,
                mode=mode,
            )
        if not state.saw_done:
            raise _ProbeAbort(
                ProbeClassification.MODEL_PROTOCOL_FAILURE,
                "stream_missing_done",
            )

    async def _bounded_aclose(self, target: Any) -> None:
        try:
            await asyncio.wait_for(
                target.aclose(),
                timeout=self.config.cleanup_timeout_s,
            )
        except asyncio.TimeoutError:
            pass
        except Exception:
            # Cleanup of a broken transport must not leak provider details or
            # replace the already-sanitized probe outcome.  CancelledError is a
            # BaseException and intentionally remains visible to the caller.
            pass

    def _process_sse_event(
        self,
        raw_data: bytes,
        *,
        state: _StreamState,
        mode: ProbeMode,
    ) -> None:
        state.sse_events += 1
        if state.sse_events > self.config.max_sse_events:
            raise _ProbeAbort(
                ProbeClassification.MODEL_NO_PROGRESS,
                "sse_event_budget_exceeded",
            )
        if raw_data.strip() == b"[DONE]":
            if state.finish_reason is None:
                raise _ProbeAbort(
                    ProbeClassification.MODEL_PROTOCOL_FAILURE,
                    "done_before_finish_reason",
                )
            state.saw_done = True
            return

        try:
            payload = json.loads(raw_data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise _ProbeAbort(
                ProbeClassification.MODEL_PROTOCOL_FAILURE,
                "malformed_sse_json",
            )
        if not isinstance(payload, dict):
            raise _ProbeAbort(
                ProbeClassification.MODEL_PROTOCOL_FAILURE,
                "non_object_sse_payload",
            )
        if "error" in payload:
            raise _ProbeAbort(
                ProbeClassification.INFRA_FAILURE,
                "stream_error_frame",
            )
        object_name = payload.get("object")
        if object_name is not None and object_name != "chat.completion.chunk":
            raise _ProbeAbort(
                ProbeClassification.MODEL_PROTOCOL_FAILURE,
                "invalid_chunk_object",
            )

        self._record_usage(payload.get("usage"), state=state)
        choices = payload.get("choices")
        if choices == [] and payload.get("usage") is not None:
            return
        if not isinstance(choices, list) or len(choices) != 1:
            raise _ProbeAbort(
                ProbeClassification.MODEL_PROTOCOL_FAILURE,
                "invalid_choices",
            )
        choice = choices[0]
        if not isinstance(choice, dict) or choice.get("index", 0) != 0:
            raise _ProbeAbort(
                ProbeClassification.MODEL_PROTOCOL_FAILURE,
                "invalid_choice",
            )
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            raise _ProbeAbort(
                ProbeClassification.MODEL_PROTOCOL_FAILURE,
                "missing_delta",
            )

        for key in ("content", "refusal"):
            content = delta.get(key)
            if content is None:
                continue
            if not isinstance(content, str):
                raise _ProbeAbort(
                    ProbeClassification.MODEL_PROTOCOL_FAILURE,
                    f"invalid_{key}_delta",
                )
            if content:
                self._mark_first_response(state)
                if key == "refusal":
                    state.saw_refusal = True
                state.visible_content.append(content)
                self._add_output_bytes(content, state=state)

        for key in ("reasoning_content", "reasoning"):
            reasoning = delta.get(key)
            if reasoning is None:
                continue
            if not isinstance(reasoning, str):
                raise _ProbeAbort(
                    ProbeClassification.MODEL_PROTOCOL_FAILURE,
                    "invalid_reasoning_delta",
                )
            if reasoning:
                self._mark_first_response(state)
                state.saw_reasoning = True
                self._add_output_bytes(reasoning, state=state)

        tool_deltas = delta.get("tool_calls")
        if tool_deltas is not None:
            if mode is ProbeMode.CONTROL:
                raise _ProbeAbort(
                    ProbeClassification.MODEL_PROTOCOL_FAILURE,
                    "unexpected_tool_call",
                )
            if self._record_tool_deltas(tool_deltas, state=state):
                self._mark_first_response(state)

        finish_reason = choice.get("finish_reason")
        if finish_reason is not None:
            if not isinstance(finish_reason, str) or finish_reason not in _ALLOWED_FINISH_REASONS:
                raise _ProbeAbort(
                    ProbeClassification.MODEL_PROTOCOL_FAILURE,
                    "invalid_finish_reason",
                )
            if state.finish_reason is not None and state.finish_reason != finish_reason:
                raise _ProbeAbort(
                    ProbeClassification.MODEL_PROTOCOL_FAILURE,
                    "conflicting_finish_reason",
                )
            state.finish_reason = finish_reason
            self._mark_first_response(state)

    def _record_tool_deltas(self, value: Any, *, state: _StreamState) -> bool:
        if not isinstance(value, list) or not value:
            raise _ProbeAbort(
                ProbeClassification.MODEL_PROTOCOL_FAILURE,
                "invalid_tool_call_delta",
            )
        action_started = False
        for item in value:
            if not isinstance(item, dict):
                raise _ProbeAbort(
                    ProbeClassification.MODEL_PROTOCOL_FAILURE,
                    "invalid_tool_call_delta",
                )
            index = item.get("index")
            if not isinstance(index, int) or index < 0:
                raise _ProbeAbort(
                    ProbeClassification.MODEL_PROTOCOL_FAILURE,
                    "invalid_tool_call_index",
                )
            call = state.tool_calls.setdefault(index, _ToolCall())
            call_id = item.get("id")
            if call_id is not None:
                if not isinstance(call_id, str) or not call_id:
                    raise _ProbeAbort(
                        ProbeClassification.MODEL_PROTOCOL_FAILURE,
                        "invalid_tool_call_id",
                    )
                if call.call_id and call.call_id != call_id:
                    raise _ProbeAbort(
                        ProbeClassification.MODEL_PROTOCOL_FAILURE,
                        "conflicting_tool_call_id",
                    )
                call.call_id = call_id
                action_started = True
                if state.first_action_at is None:
                    state.first_action_at = time.monotonic()
            call_type = item.get("type")
            if call_type is not None:
                if call_type != "function":
                    raise _ProbeAbort(
                        ProbeClassification.MODEL_PROTOCOL_FAILURE,
                        "invalid_tool_call_type",
                    )
                call.call_type = call_type
            function = item.get("function")
            if function is not None:
                if not isinstance(function, dict):
                    raise _ProbeAbort(
                        ProbeClassification.MODEL_PROTOCOL_FAILURE,
                        "invalid_tool_function",
                    )
                name = function.get("name")
                if name is not None:
                    if not isinstance(name, str):
                        raise _ProbeAbort(
                            ProbeClassification.MODEL_PROTOCOL_FAILURE,
                            "invalid_tool_name",
                        )
                    call.name += name
                    if name:
                        action_started = True
                        if state.first_action_at is None:
                            state.first_action_at = time.monotonic()
                    self._add_output_bytes(name, state=state)
                arguments = function.get("arguments")
                if arguments is not None:
                    if not isinstance(arguments, str):
                        raise _ProbeAbort(
                            ProbeClassification.MODEL_PROTOCOL_FAILURE,
                            "invalid_tool_arguments",
                        )
                    call.arguments += arguments
                    if arguments:
                        action_started = True
                        if state.first_action_at is None:
                            state.first_action_at = time.monotonic()
                    self._add_output_bytes(arguments, state=state)
                    if len(call.arguments.encode("utf-8")) > self.config.max_tool_argument_bytes:
                        raise _ProbeAbort(
                            ProbeClassification.MODEL_NO_PROGRESS,
                            "tool_argument_budget_exceeded",
                        )
        return action_started

    @staticmethod
    def _mark_first_response(state: _StreamState) -> None:
        if state.first_response_at is None:
            state.first_response_at = time.monotonic()

    def _record_usage(self, usage: Any, *, state: _StreamState) -> None:
        if usage is None:
            return
        if not isinstance(usage, dict):
            raise _ProbeAbort(
                ProbeClassification.MODEL_PROTOCOL_FAILURE,
                "invalid_usage",
            )
        completion_tokens = usage.get("completion_tokens")
        if completion_tokens is None:
            return
        if not isinstance(completion_tokens, int) or completion_tokens < 0:
            raise _ProbeAbort(
                ProbeClassification.MODEL_PROTOCOL_FAILURE,
                "invalid_completion_tokens",
            )
        state.completion_tokens = max(state.completion_tokens, completion_tokens)
        if completion_tokens > self.config.max_completion_tokens:
            raise _ProbeAbort(
                ProbeClassification.MODEL_NO_PROGRESS,
                "completion_token_budget_exceeded",
            )

    def _add_output_bytes(self, value: str, *, state: _StreamState) -> None:
        state.output_bytes += len(value.encode("utf-8"))
        if state.output_bytes > self.config.max_output_bytes:
            raise _ProbeAbort(
                ProbeClassification.MODEL_NO_PROGRESS,
                "model_output_budget_exceeded",
            )

    def _next_wait(self, *, state: _StreamState, mode: ProbeMode) -> tuple[float, str]:
        now = time.monotonic()
        deadlines: list[tuple[float, str]] = [
            (state.start + self.config.total_timeout_s, "total_deadline_exceeded")
        ]
        if state.first_response_at is None:
            deadlines.append(
                (
                    state.start + self.config.first_response_timeout_s,
                    "first_response_deadline_exceeded",
                )
            )
        if mode is ProbeMode.NONCE_ACTION and state.first_action_at is None:
            deadlines.append(
                (
                    state.start + self.config.first_action_timeout_s,
                    "first_action_deadline_exceeded",
                )
            )
        deadline, reason = min(deadlines, key=lambda item: item[0])
        return max(deadline - now, 0.000_001), reason

    def _raise_if_deadline_elapsed(self, *, state: _StreamState, mode: ProbeMode) -> None:
        now = time.monotonic()
        if state.first_response_at is None and now >= state.start + self.config.first_response_timeout_s:
            raise _ProbeAbort(
                ProbeClassification.MODEL_NO_PROGRESS,
                "first_response_deadline_exceeded",
            )
        if (
            mode is ProbeMode.NONCE_ACTION
            and state.first_action_at is None
            and now >= state.start + self.config.first_action_timeout_s
        ):
            raise _ProbeAbort(
                ProbeClassification.MODEL_NO_PROGRESS,
                "first_action_deadline_exceeded",
            )
        if now >= state.start + self.config.total_timeout_s:
            raise _ProbeAbort(
                ProbeClassification.MODEL_NO_PROGRESS,
                "total_deadline_exceeded",
            )

    def _classify_completed(
        self,
        *,
        mode: ProbeMode,
        nonce: str,
        state: _StreamState,
    ) -> tuple[ProbeClassification, str]:
        if state.finish_reason is None or not state.saw_done:
            return ProbeClassification.MODEL_PROTOCOL_FAILURE, "incomplete_stream"
        if state.finish_reason == "length":
            return ProbeClassification.MODEL_NO_PROGRESS, "completion_token_budget_exhausted"
        if state.finish_reason == "content_filter":
            return ProbeClassification.QUALITY_FAILURE, "content_filtered"

        content = "".join(state.visible_content).strip()
        if mode is ProbeMode.CONTROL:
            if state.tool_calls:
                return ProbeClassification.MODEL_PROTOCOL_FAILURE, "unexpected_tool_call"
            if not content and not state.saw_reasoning:
                return ProbeClassification.MODEL_PROTOCOL_FAILURE, "empty_stop"
            if state.finish_reason != "stop":
                return ProbeClassification.MODEL_PROTOCOL_FAILURE, "control_finish_mismatch"
            if content == nonce:
                return ProbeClassification.CLEAN, "control_nonce_matched"
            return ProbeClassification.QUALITY_FAILURE, "control_nonce_mismatch"

        if not state.tool_calls:
            if not content and not state.saw_reasoning:
                return ProbeClassification.MODEL_PROTOCOL_FAILURE, "empty_stop"
            if state.finish_reason != "stop":
                return ProbeClassification.MODEL_PROTOCOL_FAILURE, "tool_call_missing"
            if state.saw_refusal:
                return ProbeClassification.QUALITY_FAILURE, "bounded_refusal"
            return ProbeClassification.QUALITY_FAILURE, "tool_action_not_called"
        if state.finish_reason != "tool_calls":
            return ProbeClassification.MODEL_PROTOCOL_FAILURE, "tool_finish_mismatch"
        if sorted(state.tool_calls) != [0]:
            return ProbeClassification.QUALITY_FAILURE, "multiple_tool_actions"

        call = state.tool_calls[0]
        if not call.call_id or call.call_type != "function" or not call.name:
            return ProbeClassification.MODEL_PROTOCOL_FAILURE, "incomplete_tool_call"
        if call.name != _TOOL_NAME:
            return ProbeClassification.QUALITY_FAILURE, "wrong_tool_name"
        try:
            arguments = json.loads(call.arguments)
        except json.JSONDecodeError:
            return ProbeClassification.MODEL_PROTOCOL_FAILURE, "malformed_tool_arguments"
        if not isinstance(arguments, dict):
            return ProbeClassification.MODEL_PROTOCOL_FAILURE, "non_object_tool_arguments"
        if set(arguments) != {"nonce"} or not isinstance(arguments.get("nonce"), str):
            return ProbeClassification.MODEL_PROTOCOL_FAILURE, "invalid_tool_argument_schema"
        if arguments["nonce"] != nonce:
            return ProbeClassification.QUALITY_FAILURE, "tool_nonce_mismatch"
        return ProbeClassification.CLEAN, "tool_nonce_matched"

    def _chat_completions_url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/chat/completions"
        return f"{self.base_url}/v1/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


# Explicit name for callers that want the transport contract in the type.
OpenAIBehaviorProbeClient = BehaviorProbeClient


def _validated_nonce(nonce: Optional[str]) -> str:
    value = nonce or f"affine.{secrets.token_hex(16)}"
    if not _NONCE_RE.fullmatch(value):
        raise ValueError(
            "nonce must be 8-128 characters using only letters, digits, '.', ':', '_' or '-'"
        )
    return value


def _build_result(
    *,
    probe_id: str,
    mode: ProbeMode,
    state: _StreamState,
    classification: ProbeClassification,
    reason: str,
) -> ProbeResult:
    ended = time.monotonic()
    duration_ms = max(0, round((ended - state.start) * 1_000))
    first_response_ms = (
        max(0, round((state.first_response_at - state.start) * 1_000))
        if state.first_response_at is not None
        else None
    )
    first_action_ms = (
        max(0, round((state.first_action_at - state.start) * 1_000))
        if state.first_action_at is not None
        else None
    )
    sanitized_evidence = {
        "classification": classification.value,
        "completion_tokens": state.completion_tokens,
        "finish_reason": state.finish_reason or "",
        "first_action_observed": state.first_action_at is not None,
        "first_response_observed": state.first_response_at is not None,
        "mode": mode.value,
        "output_bytes": state.output_bytes,
        "reason": reason,
        "refusal_observed": state.saw_refusal,
        "sse_events": state.sse_events,
        "saw_done": state.saw_done,
        "tool_call_count": len(state.tool_calls),
        "wire_bytes": state.wire_bytes,
    }
    evidence_hash = hashlib.sha256(
        json.dumps(
            sanitized_evidence,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return ProbeResult(
        probe_id=probe_id,
        classification=classification,
        reason=reason,
        duration_ms=duration_ms,
        first_response_ms=first_response_ms,
        first_action_ms=first_action_ms,
        completion_tokens=state.completion_tokens,
        output_bytes=state.output_bytes,
        evidence_hash=evidence_hash,
    )


__all__: Sequence[str] = (
    "BehaviorProbeClient",
    "OpenAIBehaviorProbeClient",
    "ProbeConfig",
    "ProbeMode",
)
