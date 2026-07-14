#!/usr/bin/env python3
"""Small secret-safe OpenAI Chat Completions stub for local integration gates."""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


MAX_REQUEST_BYTES = 1_048_576


def _is_loopback_host(hostname: str) -> bool:
    normalized = hostname.rstrip(".").lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19000)
    parser.add_argument("--model", default="affine-instruction-gym-e2e")
    parser.add_argument("--api-key-env")
    parser.add_argument(
        "--response",
        default="A deterministic local stub response.",
    )
    return parser


def _completion(model: str, content: str) -> bytes:
    return json.dumps(
        {
            "id": "instruction-gym-local-stub",
            "object": "chat.completion",
            "created": 1,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        },
        separators=(",", ":"),
    ).encode()


def _handler(*, model: str, response: str, api_key: str | None):
    completion = _completion(model, response)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: HTTPStatus, body: bytes = b"") -> None:
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            if body:
                self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            try:
                length = int(self.headers.get("content-length", ""))
            except ValueError:
                self._send(HTTPStatus.BAD_REQUEST)
                return
            if not 0 < length <= MAX_REQUEST_BYTES:
                self._send(HTTPStatus.BAD_REQUEST)
                return
            try:
                payload: Any = json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._send(HTTPStatus.BAD_REQUEST)
                return
            expected_authorization = f"Bearer {api_key}" if api_key else None
            if (
                self.path.rstrip("/") != "/v1/chat/completions"
                or not isinstance(payload, dict)
                or payload.get("model") != model
                or not isinstance(payload.get("messages"), list)
                or not payload["messages"]
                or expected_authorization is not None
                and self.headers.get("authorization") != expected_authorization
            ):
                self._send(HTTPStatus.BAD_REQUEST)
                return
            self._send(HTTPStatus.OK, completion)

        def log_message(self, _format: str, *args: Any) -> None:
            return

    return Handler


def main() -> int:
    args = _parser().parse_args()
    if not _is_loopback_host(args.host):
        raise SystemExit("--host must be a loopback address for this local-only stub")
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be within [1, 65535]")
    api_key = None
    if args.api_key_env is not None:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise SystemExit(f"API key variable {args.api_key_env!r} is unset or empty")
    server = ThreadingHTTPServer(
        (args.host, args.port),
        _handler(model=args.model, response=args.response, api_key=api_key),
    )
    print(f"openai-stub: listening on {args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
