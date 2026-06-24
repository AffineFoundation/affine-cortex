#!/usr/bin/env python3
"""Targon RENTAL wrapper for Affine GPU autoscaler.

Affine's autoscaler talks to a deliberately tiny provider API:

  POST /instances -> {"instance_id": "...", "ssh_url": "...", ...}
  POST /instances/{id}/renew -> {"instance_id": "...", "lease_expires_at": ...}
  DELETE /instances/{id}

Targon's native API is register -> deploy -> poll state. This wrapper bridges
those shapes without changing scheduler/executor logic. It creates a Targon
RENTAL workload that the scheduler can SSH into, then the scheduler continues
to deploy SGLang via Docker exactly as it does for static SSH endpoints.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterable, List, Mapping, Optional
from urllib.parse import urlparse

import requests


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_bool(name: str, default: str = "true") -> bool:
    value = _env(name, default).lower()
    return value not in {"0", "false", "no", "off", "none"}


API_BASE = _env("TARGON_API_URL", "https://api.targon.com/tha/v2").rstrip("/")
API_KEY = _env("TARGON_API_KEY")
AUTH_TOKEN = _env("AFFINE_GPU_PROVIDER_API_KEY")
RESOURCE_NAME = _env("TARGON_RESOURCE_NAME", "b200-xlarge")
RENTAL_IMAGE = _env("TARGON_RENTAL_IMAGE", _env("TARGON_AUTOSCALE_IMAGE"))
SSH_KEY_UID = _env("TARGON_SSH_KEY_UID")
SSH_KEY_NAME = _env("TARGON_SSH_KEY_NAME")
SSH_KEY_PATH = _env(
    "TARGON_SSH_KEY_PATH",
    "/root/.ssh/affine_validator_server",
)
SSH_HOST = _env("TARGON_SSH_HOST", "ssh.deployments.targon.com")
PROJECT_ID = _env("TARGON_PROJECT_ID")
APP_ID = _env("TARGON_APP_ID")
VOLUME_UID = _env("TARGON_VOLUME_UID")
VOLUME_MOUNT_PATH = _env("TARGON_VOLUME_MOUNT_PATH", "/data")
PORT = int(_env("TARGON_SGLANG_PORT", "10001") or "10001")
WAIT_TIMEOUT = int(_env("TARGON_CREATE_WAIT_TIMEOUT_SEC", "900") or "900")
POLL_SECONDS = int(_env("TARGON_CREATE_POLL_SECONDS", "15") or "15")
LEASE_HOURS = int(_env("TARGON_LEASE_HOURS", "0") or "0")
NAME_PREFIX = _env("TARGON_WORKLOAD_NAME_PREFIX", "affine-autoscale")
PUBLIC_INFERENCE_URL = _env("TARGON_PUBLIC_INFERENCE_URL")
REQUIRE_SSH_PROBE = _env_bool("TARGON_REQUIRE_SSH_PROBE", "true")
REQUIRE_DOCKER = _env_bool("TARGON_REQUIRE_DOCKER", "true")
REQUEST_TIMEOUT = int(_env("TARGON_API_TIMEOUT_SEC", "60") or "60")
RENEW_METHOD = _env("TARGON_RENEW_METHOD", "POST").upper()
RENEW_PATH_TEMPLATE = _env(
    "TARGON_RENEW_PATH_TEMPLATE",
    _env("TARGON_RENEW_PATH", ""),
)
RENEW_PAYLOAD_JSON = _env("TARGON_RENEW_PAYLOAD_JSON")


class TargonWrapperError(RuntimeError):
    pass


class TargonNotFound(TargonWrapperError):
    pass


class TargonAutoscaleClient:
    def __init__(self) -> None:
        if not API_BASE:
            raise TargonWrapperError("TARGON_API_URL is required")
        if not API_KEY:
            raise TargonWrapperError("TARGON_API_KEY is required")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            }
        )

    def request(self, method: str, path: str, **kwargs: Any) -> Any:
        resp = self.session.request(
            method,
            f"{API_BASE}{path}",
            timeout=REQUEST_TIMEOUT,
            **kwargs,
        )
        if resp.status_code in (204, 205):
            return {}
        if resp.status_code == 404:
            raise TargonNotFound(f"{method} {path} -> 404")
        if resp.status_code >= 400:
            raise TargonWrapperError(
                f"{method} {path} -> {resp.status_code}: {resp.text[:500]}"
            )
        if not resp.content:
            return {}
        content_type = resp.headers.get("Content-Type", "")
        if "json" not in content_type.lower():
            return {"_raw": resp.text}
        return resp.json()

    def create(self, name: str) -> dict:
        uid = ""
        try:
            result = self.request("POST", "/workloads", json=self._create_body(name))
            uid = _instance_id(result)
            if not uid:
                raise TargonWrapperError(
                    f"Targon create response did not include uid: {result}"
                )
            self.request("POST", f"/workloads/{uid}/deploy")
            return self.wait_ready(uid)
        except Exception:
            if uid:
                try:
                    self.delete(uid, force=True)
                except Exception as cleanup_error:
                    print(
                        f"warning: failed to cleanup Targon workload {uid}: "
                        f"{cleanup_error}",
                        flush=True,
                    )
            raise

    def _create_body(self, name: str) -> dict:
        if not RENTAL_IMAGE:
            raise TargonWrapperError(
                "TARGON_RENTAL_IMAGE is required. Use an image with SSH, "
                "Docker and NVIDIA container runtime available."
            )
        ssh_keys = self._ssh_key_uids()
        if not ssh_keys:
            raise TargonWrapperError(
                "Set TARGON_SSH_KEY_UID or TARGON_SSH_KEY_NAME so the "
                "autoscaler-created rental is reachable over SSH."
            )

        body: dict[str, Any] = {
            "name": name,
            "image": RENTAL_IMAGE,
            "resource_name": RESOURCE_NAME,
            "type": "RENTAL",
            "ports": [
                {"port": PORT, "protocol": "TCP", "routing": "PROXIED"},
            ],
            "ssh_keys": ssh_keys,
        }
        if PROJECT_ID:
            body["project_id"] = PROJECT_ID
        if APP_ID:
            body["app_id"] = APP_ID
        if VOLUME_UID:
            body["volumes"] = [
                {"uid": VOLUME_UID, "mount_path": VOLUME_MOUNT_PATH},
            ]

        commands = _json_list_env("TARGON_RENTAL_COMMANDS_JSON")
        args = _json_list_env("TARGON_RENTAL_ARGS_JSON")
        envs = _json_envs("TARGON_RENTAL_ENVS_JSON")
        if commands is not None:
            body["commands"] = commands
        if args is not None:
            body["args"] = args
        if envs:
            body["envs"] = envs
        return body

    def _ssh_key_uids(self) -> list[str]:
        if SSH_KEY_UID:
            return [SSH_KEY_UID]
        if not SSH_KEY_NAME:
            return []
        for key in self.items(self.request("GET", "/ssh-keys")):
            uid = str(key.get("uid") or key.get("id") or "")
            name = str(key.get("name") or key.get("key_name") or "")
            if uid and name == SSH_KEY_NAME:
                return [uid]
        raise TargonWrapperError(
            f"No Targon SSH key found for TARGON_SSH_KEY_NAME={SSH_KEY_NAME!r}"
        )

    def items(self, payload: Any) -> List[dict]:
        if isinstance(payload, list):
            return [x for x in payload if isinstance(x, dict)]
        if isinstance(payload, dict):
            for key in ("items", "data", "results", "ssh_keys", "workloads"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [x for x in value if isinstance(x, dict)]
        return []

    def wait_ready(self, uid: str) -> dict:
        start = time.time()
        last: dict[str, Any] = {}
        while time.time() - start < WAIT_TIMEOUT:
            workload = self._safe_get_workload(uid)
            state = self._safe_get_state(uid)
            merged = _merge_workload_state(workload, state)
            if merged:
                last = merged
            status = _status(merged)
            if status in {"failed", "error", "terminated", "deleted"}:
                raise TargonWrapperError(
                    f"Targon workload {uid} {status}: {str(_redacted(merged))[:800]}"
                )
            if _running(merged) and self._ssh_ready(uid):
                ssh_url = _ssh_url(uid)
                return {
                    "instance_id": uid,
                    "ssh_url": ssh_url,
                    "public_inference_url": (
                        _openai_endpoint(PUBLIC_INFERENCE_URL)
                        or _public_url(merged, uid, PORT)
                    ),
                    "lease_expires_at": _lease_expires_at(merged),
                    "status": status,
                    "raw": _redacted(merged),
                }
            time.sleep(POLL_SECONDS)
        raise TargonWrapperError(
            f"Targon workload {uid} did not become ready in {WAIT_TIMEOUT}s; "
            f"last={str(_redacted(last))[:1000]}"
        )

    def _safe_get_workload(self, uid: str) -> dict:
        try:
            workload = self.request("GET", f"/workloads/{uid}")
            return workload if isinstance(workload, dict) else {}
        except TargonNotFound:
            raise
        except Exception:
            return {}

    def _safe_get_state(self, uid: str) -> dict:
        try:
            state = self.request("GET", f"/workloads/{uid}/state")
            return state if isinstance(state, dict) else {}
        except TargonNotFound:
            raise
        except Exception:
            return {}

    def _ssh_ready(self, uid: str) -> bool:
        if not REQUIRE_SSH_PROBE:
            return True
        user, host, port = _parse_ssh_url(_ssh_url(uid))
        probe = "echo ready"
        if REQUIRE_DOCKER:
            probe = (
                "command -v docker >/dev/null "
                "&& docker version >/dev/null 2>&1 "
                "&& echo ready"
            )
        cmd = [
            "ssh",
            "-i",
            SSH_KEY_PATH,
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "ConnectTimeout=10",
            "-p",
            str(port),
            f"{user}@{host}",
            probe,
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=20,
            )
        except Exception:
            return False
        return result.returncode == 0 and "ready" in result.stdout

    def delete(self, uid: str, *, force: bool = False) -> bool:
        if not force:
            try:
                workload = self.request("GET", f"/workloads/{uid}")
            except TargonNotFound:
                return True
            name = str(workload.get("name") or "")
            if not _owned_workload_name(name):
                raise TargonWrapperError(
                    f"Refusing to delete non-autoscaler Targon workload "
                    f"{uid} name={name!r}"
                )
        try:
            self.request("DELETE", f"/workloads/{uid}")
            return True
        except TargonNotFound:
            return True

    def renew(self, uid: str) -> dict:
        workload = self.request("GET", f"/workloads/{uid}")
        name = str(workload.get("name") or "") if isinstance(workload, dict) else ""
        if name and not _owned_workload_name(name):
            raise TargonWrapperError(
                f"Refusing to renew non-autoscaler Targon workload "
                f"{uid} name={name!r}"
            )
        state = self._safe_get_state(uid)
        merged = _merge_workload_state(
            workload if isinstance(workload, dict) else {},
            state,
        )
        renewed_until = _default_lease_expires_at()
        renewal = self._renew_workload(uid, renewed_until)
        if isinstance(renewal, dict):
            merged.update(renewal)
        if RENEW_PATH_TEMPLATE:
            state = self._safe_get_state(uid)
            merged = _merge_workload_state(merged, state)
        return {
            "instance_id": uid,
            "lease_expires_at": _lease_expires_at(
                merged,
                fallback_expires_at=renewed_until,
            ),
            "status": _status(merged),
            "raw": _redacted(merged),
        }

    def _renew_workload(self, uid: str, expires_at: int) -> dict:
        if not RENEW_PATH_TEMPLATE:
            return {}
        variables = _lease_variables(uid, expires_at)
        path = _render_template(RENEW_PATH_TEMPLATE, variables)
        payload = _render_template(_json_payload_env(), variables)
        kwargs = {}
        if payload or RENEW_PAYLOAD_JSON:
            kwargs["json"] = payload
        result = self.request(RENEW_METHOD or "POST", path, **kwargs)
        return result if isinstance(result, dict) else {}


def _json_list_env(name: str) -> Optional[list[str]]:
    raw = _env(name)
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise TargonWrapperError(f"{name} must be a JSON array of strings: {e}")
    if not isinstance(parsed, list) or not all(isinstance(v, str) for v in parsed):
        raise TargonWrapperError(f"{name} must be a JSON array of strings")
    return parsed


def _json_envs(name: str) -> list[dict]:
    raw = _env(name)
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise TargonWrapperError(f"{name} must be a JSON object or env array: {e}")
    if isinstance(parsed, dict):
        return [{"name": str(k), "value": str(v)} for k, v in parsed.items()]
    if isinstance(parsed, list) and all(isinstance(v, dict) for v in parsed):
        return parsed
    raise TargonWrapperError(f"{name} must be a JSON object or env array")


def _instance_id(data: Mapping[str, Any]) -> str:
    nested = data.get("workload") if isinstance(data.get("workload"), dict) else {}
    return str(
        data.get("uid")
        or data.get("id")
        or data.get("workload_uid")
        or nested.get("uid")
        or nested.get("id")
        or ""
    )


def _status(data: Mapping[str, Any]) -> str:
    state = data.get("state") if isinstance(data.get("state"), dict) else {}
    return str(data.get("status") or state.get("status") or "").lower()


def _running(data: Mapping[str, Any]) -> bool:
    status = _status(data)
    state = data.get("state") if isinstance(data.get("state"), dict) else {}
    ready = int(data.get("ready_replicas") or state.get("ready_replicas") or 0)
    if ready > 0:
        return True
    return status in {"running", "ready", "active", "healthy"}


def _merge_workload_state(workload: Mapping[str, Any], state: Mapping[str, Any]) -> dict:
    merged: dict[str, Any] = dict(workload or {})
    if state:
        merged["state"] = dict(state)
        for key in ("status", "message", "ready_replicas", "total_replicas", "urls"):
            if key in state:
                merged[key] = state[key]
    return merged


def _ssh_url(uid: str) -> str:
    return f"ssh://{uid}@{SSH_HOST}:22"


def _public_url(data: Mapping[str, Any], uid: str, port: int) -> str:
    for url in _urls(data, port):
        return _openai_endpoint(url)
    return _openai_endpoint(f"https://{uid}-{port}.caas.targon.com")


def _urls(data: Mapping[str, Any], port: int) -> Iterable[str]:
    for value in data.get("urls") or []:
        if isinstance(value, dict) and int(value.get("port") or port) == port:
            url = value.get("url")
            if url:
                yield str(url)
        elif isinstance(value, str):
            yield value
    state = data.get("state") if isinstance(data.get("state"), dict) else {}
    for value in state.get("urls") or []:
        if isinstance(value, dict) and int(value.get("port") or port) == port:
            url = value.get("url")
            if url:
                yield str(url)
        elif isinstance(value, str):
            yield value


def _openai_endpoint(url: str) -> str:
    url = url.rstrip("/")
    if not url:
        return ""
    return url if url.endswith("/v1") else f"{url}/v1"


def _lease_expires_at(
    data: Mapping[str, Any],
    *,
    fallback_expires_at: int = 0,
) -> int:
    for key in (
        "removal_scheduled_at",
        "termination_scheduled_at",
        "scheduled_termination_at",
        "expires_at",
    ):
        parsed = _parse_time(data.get(key))
        if parsed:
            return parsed
    state = data.get("state") if isinstance(data.get("state"), dict) else {}
    for key in (
        "removal_scheduled_at",
        "termination_scheduled_at",
        "scheduled_termination_at",
        "expires_at",
    ):
        parsed = _parse_time(state.get(key))
        if parsed:
            return parsed
    if fallback_expires_at > 0:
        return fallback_expires_at
    default_expires_at = _default_lease_expires_at()
    if default_expires_at > 0:
        return default_expires_at
    return 0


def _default_lease_expires_at() -> int:
    if LEASE_HOURS <= 0:
        return 0
    return int(time.time()) + LEASE_HOURS * 60 * 60


def _lease_variables(uid: str, expires_at: int) -> dict[str, str]:
    now = int(time.time())
    return {
        "uid": uid,
        "workload_uid": uid,
        "lease_hours": str(LEASE_HOURS),
        "lease_expires_at": str(expires_at),
        "lease_expires_iso": _iso_utc(expires_at),
        "now": str(now),
        "now_iso": _iso_utc(now),
    }


def _iso_utc(timestamp: int) -> str:
    if timestamp <= 0:
        return ""
    return datetime.fromtimestamp(timestamp, timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _json_payload_env() -> dict:
    if not RENEW_PAYLOAD_JSON:
        return {}
    try:
        parsed = json.loads(RENEW_PAYLOAD_JSON)
    except json.JSONDecodeError as e:
        raise TargonWrapperError(
            f"TARGON_RENEW_PAYLOAD_JSON must be a JSON object: {e}"
        )
    if not isinstance(parsed, dict):
        raise TargonWrapperError("TARGON_RENEW_PAYLOAD_JSON must be a JSON object")
    return parsed


def _render_template(value: Any, variables: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        try:
            return value.format(**variables)
        except (KeyError, IndexError, ValueError):
            return value
    if isinstance(value, list):
        return [_render_template(v, variables) for v in value]
    if isinstance(value, dict):
        return {k: _render_template(v, variables) for k, v in value.items()}
    return value


def _parse_time(value: Any) -> int:
    if not value:
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    if not text:
        return 0
    if text.isdigit():
        return int(text)
    try:
        return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return 0


def _parse_ssh_url(url: str) -> tuple[str, str, int]:
    if not url.startswith("ssh://"):
        raise ValueError(f"ssh URL must start with ssh://: {url!r}")
    body = url[len("ssh://") :]
    user = ""
    if "@" in body:
        user, body = body.split("@", 1)
    if ":" in body:
        host, raw_port = body.rsplit(":", 1)
        return user, host, int(raw_port)
    return user, body, 22


def _safe_suffix(value: str) -> str:
    return "".join(
        c if c.isalnum() or c == "-" else "-"
        for c in value.lower()
    ).strip("-")


def _owned_workload_name(name: str) -> bool:
    return name == NAME_PREFIX or name.startswith(f"{NAME_PREFIX}-")


def _redacted(payload: Mapping[str, Any]) -> dict:
    out = dict(payload or {})
    if "envs" in out and isinstance(out["envs"], list):
        envs = []
        for item in out["envs"]:
            if not isinstance(item, dict):
                envs.append(item)
                continue
            redacted = dict(item)
            name = str(redacted.get("name") or "").upper()
            if any(token in name for token in ("TOKEN", "KEY", "SECRET", "PASSWORD")):
                redacted["value"] = "<redacted>"
            envs.append(redacted)
        out["envs"] = envs
    return out


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, payload: Any) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        if not AUTH_TOKEN:
            return True
        auth = self.headers.get("Authorization", "")
        return auth == f"Bearer {AUTH_TOKEN}"

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send(200, {"ok": True})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if not self._authorized():
            self._send(401, {"error": "unauthorized"})
            return
        parsed = urlparse(self.path)
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) == 3 and parts[0] == "instances" and parts[2] == "renew":
            try:
                result = TargonAutoscaleClient().renew(parts[1])
                self._send(200, result)
            except Exception as e:
                self._send(500, {"error": type(e).__name__, "message": str(e)})
            return
        if parsed.path != "/instances":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length).decode("utf-8") if length else "{}"
            data = json.loads(raw or "{}")
            suffix = _safe_suffix(str(data.get("endpoint_name") or int(time.time())))
            name = f"{NAME_PREFIX}-{suffix}"[:32].strip("-")
            result = TargonAutoscaleClient().create(name)
            self._send(200, result)
        except Exception as e:
            self._send(500, {"error": type(e).__name__, "message": str(e)})

    def do_DELETE(self) -> None:
        if not self._authorized():
            self._send(401, {"error": "unauthorized"})
            return
        parsed = urlparse(self.path)
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) != 2 or parts[0] != "instances":
            self._send(404, {"error": "not found"})
            return
        try:
            ok = TargonAutoscaleClient().delete(parts[1])
            self._send(200, {"ok": ok})
        except Exception as e:
            self._send(500, {"error": type(e).__name__, "message": str(e)})

    def log_message(self, fmt: str, *args: Any) -> None:
        print(
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            self.address_string(),
            fmt % args,
            flush=True,
        )


def main() -> None:
    host = _env("TARGON_WRAPPER_HOST", "0.0.0.0")
    port = int(_env("TARGON_WRAPPER_PORT", "8902") or "8902")
    print(
        f"targon-autoscale-wrapper listening on {host}:{port} "
        f"resource={RESOURCE_NAME} port={PORT}",
        flush=True,
    )
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
