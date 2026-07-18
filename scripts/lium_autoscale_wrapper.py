#!/usr/bin/env python3
"""Small Lium wrapper for Affine GPU autoscaler.

The Affine autoscaler expects a simple provider API:

  POST /instances -> {"instance_id": "...", "ssh_url": "...", ...}
  POST /instances/{id}/renew -> {"instance_id": "...", "lease_expires_at": ...}
  DELETE /instances/{id}

Lium's native API is multi-step: select an executor, resolve an SSH key and
template, rent a pod, then poll until SSH/public ports are assigned. This
wrapper bridges those shapes without changing the running Affine scheduler.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterable, List, Mapping, Optional
from urllib.parse import parse_qs, urlparse

import requests


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


API_BASE = _env("LIUM_API_BASE", "https://lium.io/api").rstrip("/")
API_KEY = _env("LIUM_API_KEY")
AUTH_TOKEN = _env("AFFINE_GPU_PROVIDER_API_KEY")
MACHINE_NAME = _env("LIUM_MACHINE_NAME", "NVIDIA B200")
GPU_COUNT = int(_env("LIUM_GPU_COUNT", "8") or "8")
SSH_KEY_NAME = _env("LIUM_SSH_KEY_NAME", "online")
SSH_KEY_UID = _env("LIUM_SSH_KEY_UID")
SSH_USER = _env("LIUM_SSH_USER", "root")
SSH_KEY_PATH = _env("LIUM_SSH_KEY_PATH", "/root/.ssh/id_ed25519")
TEMPLATE_ID = _env("LIUM_TEMPLATE_ID")
TERMINATION_HOURS = int(_env("LIUM_TERMINATION_HOURS", "8") or "8")
PORT = int(_env("LIUM_SGLANG_PORT", "40000") or "40000")
WAIT_TIMEOUT = int(_env("LIUM_CREATE_WAIT_TIMEOUT_SEC", "900") or "900")
POLL_SECONDS = int(_env("LIUM_CREATE_POLL_SECONDS", "15") or "15")
DELETE_VERIFY_ATTEMPTS = int(
    _env("LIUM_DELETE_VERIFY_ATTEMPTS", "10") or "10"
)
DELETE_VERIFY_INTERVAL_SECONDS = float(
    _env("LIUM_DELETE_VERIFY_INTERVAL_SECONDS", "2") or "2"
)
ENDPOINT_SCHEME = _env("LIUM_ENDPOINT_SCHEME", "http").rstrip(":/") or "http"
NAME_PREFIX = _env("LIUM_POD_NAME_PREFIX", "affine-autoscale")


class LiumWrapperError(RuntimeError):
    error_source = "wrapper"
    code = "wrapper_error"
    http_status = 500

    def __init__(
        self,
        message: str,
        *,
        code: str = "",
        http_status: int = 0,
    ):
        super().__init__(message)
        self.message = message
        if code:
            self.code = code
        if http_status:
            self.http_status = http_status


class LiumProviderError(LiumWrapperError):
    error_source = "provider"
    code = "provider_error"
    http_status = 502


class LiumHTTPError(LiumProviderError):
    code = "provider_http_error"

    def __init__(self, method: str, path: str, status_code: int, text: str):
        self.method = method
        self.path = path
        self.status_code = status_code
        self.text = text
        code = (
            "provider_instance_not_found"
            if status_code == 404
            else "provider_http_error"
        )
        super().__init__(
            f"{method} {path} -> {status_code}: {text[:500]}",
            code=code,
            http_status=status_code,
        )


class LiumRequestError(LiumProviderError):
    def __init__(
        self,
        method: str,
        path: str,
        error: Exception,
        *,
        code: str = "provider_transport_error",
        http_status: int = 502,
    ):
        self.method = method
        self.path = path
        self.original_error = error
        super().__init__(
            f"{method} {path} transport failed: {type(error).__name__}: {error}",
            code=code,
            http_status=http_status,
        )


class LiumClient:
    def __init__(self) -> None:
        if not API_BASE:
            raise LiumWrapperError(
                "LIUM_API_BASE is required",
                code="wrapper_config_error",
            )
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        if API_KEY:
            self.session.headers.update(
                {"Authorization": f"Bearer {API_KEY}", "X-API-Key": API_KEY}
            )

    def request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            resp = self.session.request(
                method, f"{API_BASE}{path}", timeout=60, **kwargs
            )
        except requests.Timeout as e:
            raise LiumRequestError(
                method,
                path,
                e,
                code="provider_timeout",
                http_status=504,
            ) from e
        except requests.RequestException as e:
            raise LiumRequestError(method, path, e) from e
        if resp.status_code in (204, 205):
            return {}
        if resp.status_code >= 400:
            raise LiumHTTPError(method, path, resp.status_code, resp.text)
        if not resp.content:
            return {}
        try:
            return resp.json()
        except ValueError as e:
            raise LiumProviderError(
                f"{method} {path} returned non-JSON response: {resp.text[:500]}",
                code="provider_invalid_response",
            ) from e

    def items(self, payload: Any) -> List[dict]:
        if isinstance(payload, list):
            return [x for x in payload if isinstance(x, dict)]
        if isinstance(payload, dict):
            for key in (
                "items",
                "instances",
                "workloads",
                "executors",
                "pods",
                "data",
                "results",
            ):
                value = payload.get(key)
                if isinstance(value, list):
                    return [x for x in value if isinstance(x, dict)]
        return []

    def select_executor(self) -> dict:
        data = self.request(
            "GET",
            "/executors",
            params={"size": 100, "machine_names": MACHINE_NAME},
        )
        candidates = [
            e
            for e in self.items(data)
            if int(e.get("available_gpu_count") or e.get("gpu_count") or 0) >= GPU_COUNT
        ]
        if not candidates:
            raise LiumProviderError(
                f"No Lium executor has {GPU_COUNT} available GPUs for {MACHINE_NAME}",
                code="provider_capacity_unavailable",
                http_status=503,
            )
        candidates.sort(
            key=lambda e: (
                float(e.get("price_per_gpu") or 999999),
                -(int(e.get("available_gpu_count") or 0)),
            )
        )
        return candidates[0]

    def resolve_ssh_public_key(self) -> str:
        keys = self.items(self.request("GET", "/ssh-keys"))
        for key in keys:
            uid = str(key.get("uid") or key.get("id") or key.get("key_id") or "")
            name = str(key.get("name") or key.get("key_name") or "")
            if (SSH_KEY_UID and uid == SSH_KEY_UID) or (
                SSH_KEY_NAME and name == SSH_KEY_NAME
            ):
                public_key = key.get("public_key")
                if public_key:
                    return str(public_key)
        raise LiumProviderError(
            f"No public key found for LIUM_SSH_KEY_NAME={SSH_KEY_NAME!r}",
            code="provider_configuration_error",
        )

    def resolve_template_id(self) -> str:
        if TEMPLATE_ID:
            return TEMPLATE_ID
        templates = self.items(self.request("GET", "/templates"))
        if not templates:
            raise LiumProviderError(
                "No Lium templates available",
                code="provider_configuration_error",
            )
        templates = [
            template for template in templates if PORT in _service_ports(template)
        ]
        if not templates:
            raise LiumProviderError(
                f"No Lium template exposes internal service port {PORT}",
                code="provider_configuration_error",
            )

        def score(template: Mapping[str, Any]) -> tuple:
            status = str(template.get("status") or "")
            name = str(template.get("name") or "").lower()
            image = (
                f"{template.get('docker_image') or ''}:"
                f"{template.get('docker_image_tag') or ''}"
            ).lower()
            supports_docker = bool(template.get("supports_docker"))
            is_pytorch = "pytorch" in name or "pytorch" in image
            is_dind = "dind" in name or "dind" in image
            return (
                0 if status in {"VERIFY_SUCCESS", "UPDATED"} else 1,
                0 if supports_docker else 1,
                0 if is_pytorch else 1,
                0 if is_dind else 1,
                str(template.get("updated_at") or ""),
            )

        templates.sort(key=score)
        template_id = str(templates[0].get("id") or "")
        if not template_id:
            raise LiumProviderError(
                f"Selected template has no id: {templates[0]}",
                code="provider_invalid_response",
            )
        return template_id

    def create(self, name: str) -> dict:
        existing_uid = self.find_pod_by_name(name, attempts=1)
        if existing_uid:
            return self.wait_ready(existing_uid)

        executor = self.select_executor()
        executor_id = _instance_id(executor)
        if not executor_id:
            raise LiumProviderError(
                f"Selected executor has no id: {executor}",
                code="provider_invalid_response",
            )
        payload = {
            "pod_name": name,
            "template_id": self.resolve_template_id(),
            "gpu_count": GPU_COUNT,
            "user_public_key": self.resolve_ssh_public_key(),
            "skip_add_agent_ssh_key": False,
            "termination_hours": TERMINATION_HOURS,
            "initial_port_count": 2,
            "enable_jupyter": False,
        }
        try:
            result = self.request(
                "POST", f"/executors/{executor_id}/rent", json=payload
            )
        except Exception:
            existing_uid = self.find_pod_by_name(name, attempts=5)
            if existing_uid:
                return self.wait_ready(existing_uid)
            raise
        uid = _instance_id(result)
        pod = result.get("pod") if isinstance(result, dict) else None
        if not uid and isinstance(pod, dict):
            uid = _instance_id(pod)
        if not uid:
            uid = self.find_pod_by_name(name, attempts=5)
        if not uid:
            raise LiumProviderError(
                f"Rent response did not include a pod id: {result}",
                code="provider_invalid_response",
            )
        try:
            return self.wait_ready(uid)
        except Exception:
            self._delete_owned_pod(uid)
            raise

    def find_pod_by_name(self, name: str, *, attempts: int = 5) -> str:
        for _ in range(max(1, attempts)):
            for pod in self.items(self.request("GET", "/pods")):
                if pod.get("pod_name") == name or pod.get("name") == name:
                    status = _status(pod)
                    if _terminal_status(status):
                        continue
                    uid = _instance_id(pod)
                    if uid:
                        return uid
            time.sleep(1)
        return ""

    def find_pod_by_uid(self, uid: str) -> Optional[dict]:
        try:
            pods = self.items(self.request("GET", "/pods"))
        except LiumHTTPError as e:
            if e.status_code == 404:
                return None
            raise
        for pod in pods:
            if _instance_id(pod) == uid:
                return pod
        return None

    def wait_ready(self, uid: str) -> dict:
        start = time.time()
        last: Optional[dict] = None
        while time.time() - start < WAIT_TIMEOUT:
            pod = self.get_pod(uid)
            if pod is None:
                time.sleep(POLL_SECONDS)
                continue
            if isinstance(pod, dict):
                last = pod
            status = _status(pod)
            if status in {"failed", "error", "terminated", "deleted"}:
                raise LiumProviderError(
                    f"Pod {uid} {status}: {str(pod)[:500]}",
                    code="provider_instance_failed",
                    http_status=502,
                )
            ssh_url = _ssh_url(pod)
            public_url, sglang_port = _inference_endpoint(pod, PORT)
            if ssh_url and public_url and self._ssh_ready(ssh_url):
                return {
                    "instance_id": uid,
                    "ssh_url": ssh_url,
                    "public_inference_url": public_url,
                    "sglang_port": sglang_port,
                    "lease_expires_at": (
                        _lease_expires_at(pod) or _default_lease_expires_at()
                    ),
                    "status": status,
                    "raw": _redacted_pod(pod),
                }
            time.sleep(POLL_SECONDS)
        raise LiumProviderError(
            f"Pod {uid} did not expose SSH and internal port {PORT} "
            f"within {WAIT_TIMEOUT}s; "
            f"last={str(_redacted_pod(last or {}))[:800]}",
            code="provider_instance_not_ready",
            http_status=504,
        )

    def get_pod(self, uid: str) -> Optional[dict]:
        try:
            pod = self.request("GET", f"/pods/{uid}")
        except LiumHTTPError as e:
            if e.status_code != 404:
                raise

            # Some Lium deployments list a running pod from ``GET /pods``
            # while their detail route returns 404 for the same UUID.  A
            # detail-only lookup would therefore make delete() report success
            # without ever issuing DELETE, leaving the rental bill running.
            return self.find_pod_by_uid(uid)
        return pod if isinstance(pod, dict) else {}

    def status(self, uid: str) -> dict:
        pod = self.get_pod(uid)
        if pod is None:
            raise LiumHTTPError("GET", f"/pods/{uid}", 404, "not found")
        public_url, sglang_port = _inference_endpoint(pod, PORT)
        return {
            "instance_id": uid,
            "lease_expires_at": _lease_expires_at(pod),
            "status": _status(pod),
            "ssh_url": _ssh_url(pod),
            "public_inference_url": public_url,
            "sglang_port": sglang_port,
            "raw": _redacted_pod(pod),
        }

    def delete(self, uid: str, *, expected_purpose: str = "") -> bool:
        return self._delete_owned_pod(uid, expected_purpose=expected_purpose)

    def _delete_owned_pod(
        self,
        uid: str,
        *,
        expected_purpose: str = "",
    ) -> bool:
        pod = self.get_pod(uid)
        if pod is None:
            return True
        pod_name = _pod_name(pod)
        if not _owned_pod_name(pod_name, expected_purpose=expected_purpose):
            raise LiumWrapperError(
                f"Refusing to delete pod {uid}: pod_name={pod_name!r} "
                f"does not match prefix={NAME_PREFIX!r} "
                f"purpose={expected_purpose or '*'}",
                code="wrapper_safety_violation",
                http_status=409,
            )
        try:
            self.request("DELETE", f"/pods/{uid}")
        except LiumHTTPError as e:
            if e.status_code != 404:
                raise

        for attempt in range(max(1, DELETE_VERIFY_ATTEMPTS)):
            remaining = self.find_pod_by_uid(uid)
            if remaining is None or _terminal_status(_status(remaining)):
                return True
            if attempt + 1 < DELETE_VERIFY_ATTEMPTS:
                time.sleep(max(0.0, DELETE_VERIFY_INTERVAL_SECONDS))
        raise LiumProviderError(
            f"Pod {uid} is still present after provider delete",
            code="provider_delete_not_confirmed",
            http_status=502,
        )

    def renew(self, uid: str) -> dict:
        if TERMINATION_HOURS <= 0:
            self.request("DELETE", f"/pods/{uid}/schedule-removal")
            pod = self.request("GET", f"/pods/{uid}")
            return {
                "instance_id": uid,
                "lease_expires_at": 0,
                "status": _status(pod),
                "raw": _redacted_pod(pod),
            }

        expires_at = _default_lease_expires_at()
        termination_time = datetime.fromtimestamp(
            expires_at,
            timezone.utc,
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.request(
            "POST",
            f"/pods/{uid}/schedule-removal",
            json={"removal_scheduled_at": termination_time},
        )
        pod = self.request("GET", f"/pods/{uid}")
        return {
            "instance_id": uid,
            "lease_expires_at": _lease_expires_at(pod) or expires_at,
            "status": _status(pod),
            "raw": _redacted_pod(pod),
        }

    def _ssh_ready(self, ssh_url: str) -> bool:
        try:
            user, host, port = _parse_ssh_url(ssh_url)
        except ValueError as e:
            raise LiumProviderError(
                str(e),
                code="provider_invalid_response",
            ) from e
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
            "echo ready",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        except Exception:
            return False
        return result.returncode == 0 and "ready" in result.stdout


def _instance_id(data: Mapping[str, Any]) -> str:
    return str(
        data.get("uid")
        or data.get("id")
        or data.get("pod_id")
        or data.get("instance_id")
        or ""
    )


def _status(data: Mapping[str, Any]) -> str:
    state = data.get("state") if isinstance(data.get("state"), dict) else {}
    return str(data.get("status") or state.get("status") or "").lower()


def _terminal_status(status: str) -> bool:
    return status in {"failed", "error", "terminated", "deleted"}


def _pod_name(data: Mapping[str, Any]) -> str:
    return str(data.get("pod_name") or data.get("name") or "")


def _safe_token(value: Any, *, default: str = "") -> str:
    text = str(value or "").strip().lower()
    safe = "".join(c if c.isalnum() or c == "-" else "-" for c in text)
    safe = "-".join(part for part in safe.split("-") if part)
    return safe or default


def _resource_name(*, purpose: str, suffix: str, max_len: int) -> str:
    purpose = _safe_token(purpose, default="eval")
    suffix = _safe_token(suffix, default=str(int(time.time())))
    raw = f"{NAME_PREFIX}-{purpose}-{suffix}".strip("-")
    if len(raw) <= max_len:
        return raw
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:6]
    fixed_len = len(NAME_PREFIX) + len(purpose) + len(digest) + 3
    suffix_budget = max(1, max_len - fixed_len)
    return f"{NAME_PREFIX}-{purpose}-{suffix[:suffix_budget]}-{digest}"[:max_len].strip(
        "-"
    )


def _owned_pod_name(name: str, *, expected_purpose: str = "") -> bool:
    if not (name == NAME_PREFIX or name.startswith(f"{NAME_PREFIX}-")):
        return False
    expected = _safe_token(expected_purpose)
    if not expected:
        return True
    purpose_prefix = f"{NAME_PREFIX}-{expected}"
    return name == purpose_prefix or name.startswith(f"{purpose_prefix}-")


def _default_lease_expires_at() -> int:
    if TERMINATION_HOURS <= 0:
        return 0
    return int(time.time()) + TERMINATION_HOURS * 60 * 60


def _lease_expires_at(data: Mapping[str, Any]) -> int:
    for key in (
        "removal_scheduled_at",
        "termination_scheduled_at",
        "scheduled_termination_at",
        "expires_at",
    ):
        value = data.get(key)
        if not value:
            continue
        parsed = _parse_time(value)
        if parsed:
            return parsed
    state = data.get("state") if isinstance(data.get("state"), dict) else {}
    value = state.get("removal_scheduled_at") or state.get("expires_at")
    return _parse_time(value)


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


def _parse_ssh_cmd(data: Mapping[str, Any]) -> dict:
    cmd = str(data.get("ssh_connect_cmd") or "")
    if not cmd:
        return {}
    try:
        parts = shlex.split(cmd)
    except ValueError:
        parts = cmd.split()
    host = ""
    port = ""
    user = ""
    for idx, part in enumerate(parts):
        if part == "-p" and idx + 1 < len(parts):
            port = parts[idx + 1]
        elif "@" in part and not part.startswith("-"):
            user, host = part.rsplit("@", 1)
    return {"user": user, "host": host, "port": port}


def _port_items(data: Mapping[str, Any]) -> List[dict]:
    items: List[dict] = []
    for key in ("ports", "exposed_ports", "services"):
        value = data.get(key)
        if isinstance(value, list):
            items.extend([x for x in value if isinstance(x, dict)])
    ports_mapping = data.get("ports_mapping")
    if isinstance(ports_mapping, dict):
        for internal, public in ports_mapping.items():
            try:
                items.append({"port": int(internal), "public_port": int(public)})
            except (TypeError, ValueError):
                pass
    network = data.get("network") if isinstance(data.get("network"), dict) else {}
    value = network.get("ports")
    if isinstance(value, list):
        items.extend([x for x in value if isinstance(x, dict)])
    return items


def _service_ports(data: Mapping[str, Any]) -> set[int]:
    ports = set()
    for value in data.get("internal_ports") or []:
        try:
            port = int(value)
        except (TypeError, ValueError):
            continue
        if port > 0 and port != 22:
            ports.add(port)
    return ports


def _ssh_url(data: Mapping[str, Any]) -> str:
    direct = str(data.get("ssh_url") or "").strip()
    if direct:
        return direct
    parsed = _parse_ssh_cmd(data)
    ssh = data.get("ssh") if isinstance(data.get("ssh"), dict) else {}
    network = data.get("network") if isinstance(data.get("network"), dict) else {}
    host = str(
        parsed.get("host")
        or ssh.get("host")
        or data.get("ssh_host")
        or data.get("public_ip")
        or data.get("ip")
        or data.get("host")
        or data.get("hostname")
        or network.get("public_ip")
        or ""
    )
    port = parsed.get("port") or ssh.get("port") or data.get("ssh_port")
    if not port:
        for item in _port_items(data):
            internal = (
                item.get("port")
                or item.get("target_port")
                or item.get("container_port")
            )
            if int(internal or 0) == 22:
                port = (
                    item.get("public_port")
                    or item.get("external_port")
                    or item.get("host_port")
                    or 22
                )
                break
    if not host:
        return ""
    user = parsed.get("user") or SSH_USER
    return f"ssh://{user}@{host}:{int(port or 22)}"


def _inference_endpoint(
    data: Mapping[str, Any],
    internal_port: int,
) -> tuple[str, int]:
    for url in _urls(data, internal_port):
        return _openai_endpoint(url), internal_port
    return "", 0


def _urls(data: Mapping[str, Any], port: int) -> Iterable[str]:
    for value in data.get("urls") or []:
        if not isinstance(value, dict):
            continue
        try:
            mapped_port = int(value.get("port") or 0)
        except (TypeError, ValueError):
            continue
        if mapped_port == port and value.get("url"):
            yield str(value["url"])
    state = data.get("state") if isinstance(data.get("state"), dict) else {}
    for value in state.get("urls") or []:
        if not isinstance(value, dict):
            continue
        try:
            mapped_port = int(value.get("port") or 0)
        except (TypeError, ValueError):
            continue
        if mapped_port == port and value.get("url"):
            yield str(value["url"])
    for item in _port_items(data):
        internal = (
            item.get("port") or item.get("target_port") or item.get("container_port")
        )
        if int(internal or 0) != port:
            continue
        if item.get("url"):
            yield str(item["url"])
            continue
        host = (
            item.get("host")
            or item.get("public_ip")
            or item.get("hostname")
            or _parse_host_from_ssh_url(_ssh_url(data))
        )
        public_port = (
            item.get("public_port")
            or item.get("external_port")
            or item.get("host_port")
        )
        if host and public_port:
            yield f"{ENDPOINT_SCHEME}://{host}:{int(public_port)}"


def _openai_endpoint(url: str) -> str:
    url = url.rstrip("/")
    if not url:
        return ""
    return url if url.endswith("/v1") else f"{url}/v1"


def _parse_ssh_url(url: str) -> tuple[str, str, int]:
    if not url.startswith("ssh://"):
        raise ValueError(f"ssh URL must start with ssh://: {url!r}")
    body = url[len("ssh://") :]
    user = SSH_USER
    if "@" in body:
        user, body = body.split("@", 1)
    if ":" in body:
        host, raw_port = body.rsplit(":", 1)
        return user, host, int(raw_port)
    return user, body, 22


def _parse_host_from_ssh_url(url: str) -> str:
    if not url:
        return ""
    try:
        return _parse_ssh_url(url)[1]
    except Exception:
        return ""


def _redacted_pod(pod: Mapping[str, Any]) -> dict:
    out = dict(pod or {})
    for key in ("user_public_key", "public_key"):
        if key in out:
            out[key] = "<redacted>"
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
        if not self._authorized():
            self._send(401, _wrapper_error("unauthorized", "unauthorized"))
            return
        parsed = urlparse(self.path)
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) == 2 and parts[0] == "instances":
            try:
                result = LiumClient().status(parts[1])
                self._send(200, result)
            except Exception as e:
                self._send(*_error_response(e))
            return
        self._send(404, _route_not_found("GET", parsed.path))

    def do_POST(self) -> None:
        if not self._authorized():
            self._send(401, _wrapper_error("unauthorized", "unauthorized"))
            return
        parsed = urlparse(self.path)
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) == 3 and parts[0] == "instances" and parts[2] == "renew":
            try:
                result = LiumClient().renew(parts[1])
                self._send(200, result)
            except Exception as e:
                self._send(*_error_response(e))
            return
        if parsed.path != "/instances":
            self._send(404, _route_not_found("POST", parsed.path))
            return
        try:
            data, error = _read_json_body(self)
            if error:
                self._send(400, error)
                return
            purpose = str(data.get("purpose") or "eval")
            suffix = str(data.get("endpoint_name") or int(time.time()))
            name = _resource_name(purpose=purpose, suffix=suffix, max_len=63)
            result = LiumClient().create(name)
            self._send(200, result)
        except Exception as e:
            self._send(*_error_response(e))

    def do_DELETE(self) -> None:
        if not self._authorized():
            self._send(401, _wrapper_error("unauthorized", "unauthorized"))
            return
        parsed = urlparse(self.path)
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) != 2 or parts[0] != "instances":
            self._send(404, _route_not_found("DELETE", parsed.path))
            return
        try:
            expected_purpose, error = _expected_purpose(self, parsed)
            if error:
                self._send(400, error)
                return
            ok = LiumClient().delete(
                parts[1],
                expected_purpose=expected_purpose,
            )
            self._send(200, {"ok": ok})
        except Exception as e:
            self._send(*_error_response(e))

    def log_message(self, fmt: str, *args: Any) -> None:
        print(
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            self.address_string(),
            fmt % args,
            flush=True,
        )


def _route_not_found(method: str, path: str) -> dict:
    payload = _wrapper_error("route_not_found", "wrapper route not found")
    payload.update(
        {
            "error": "not_found",
            "method": method,
            "path": path,
        }
    )
    return payload


def _wrapper_error(code: str, message: str, **extra: Any) -> dict:
    payload = {
        "error": code,
        "error_source": "wrapper",
        "code": code,
        "message": message,
    }
    payload.update(extra)
    return payload


def _provider_error(error: Exception, **extra: Any) -> dict:
    payload = {
        "error": type(error).__name__,
        "error_source": "provider",
        "code": getattr(error, "code", "provider_error"),
        "provider": "lium",
        "message": str(error),
    }
    payload.update(extra)
    return payload


def _provider_http_error(error: LiumHTTPError) -> dict:
    return _provider_error(
        error,
        provider_status_code=error.status_code,
        provider_method=error.method,
        provider_path=error.path,
        provider_response=error.text[:1000],
    )


def _provider_transport_error(error: LiumRequestError) -> dict:
    return _provider_error(
        error,
        provider_method=error.method,
        provider_path=error.path,
    )


def _provider_not_found(error: Exception) -> dict:
    if isinstance(error, LiumHTTPError):
        return _provider_http_error(error)
    return _provider_error(
        error,
        code="provider_instance_not_found",
        provider_status_code=404,
    )


def _bad_request(message: str, **extra: Any) -> dict:
    return _wrapper_error("bad_request", message, **extra)


def _error_response(error: Exception) -> tuple[int, dict]:
    if isinstance(error, LiumHTTPError):
        return error.status_code, _provider_http_error(error)
    if isinstance(error, LiumRequestError):
        return error.http_status, _provider_transport_error(error)
    if isinstance(error, LiumProviderError):
        return error.http_status, _provider_error(error)
    if isinstance(error, LiumWrapperError):
        return error.http_status, _wrapper_error(
            error.code,
            str(error),
            error=type(error).__name__,
        )
    return 500, _wrapper_error(
        "wrapper_internal_error",
        str(error),
        error=type(error).__name__,
    )


def _read_json_body(
    handler: BaseHTTPRequestHandler,
) -> tuple[dict, Optional[dict]]:
    try:
        length = int(handler.headers.get("Content-Length") or 0)
    except ValueError:
        return {}, _bad_request("Content-Length must be an integer")
    if length <= 0:
        return {}, None
    raw = handler.rfile.read(length)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return {}, _bad_request("request body must be UTF-8 JSON")
    try:
        data = json.loads(text or "{}")
    except json.JSONDecodeError as e:
        return {}, _bad_request(f"request body must be valid JSON: {e}")
    if not isinstance(data, dict):
        return {}, _bad_request("request body must be a JSON object")
    return data, None


def _expected_purpose(
    handler: BaseHTTPRequestHandler,
    parsed,
) -> tuple[str, Optional[dict]]:
    query = parse_qs(parsed.query)
    if query.get("purpose"):
        return _safe_token(query["purpose"][0]), None
    data, error = _read_json_body(handler)
    if error:
        return "", error
    return _safe_token(data.get("purpose")), None


def main() -> None:
    host = _env("LIUM_WRAPPER_HOST", "0.0.0.0")
    port = int(_env("LIUM_WRAPPER_PORT", "8901") or "8901")
    print(
        f"lium-autoscale-wrapper listening on {host}:{port} "
        f"machine={MACHINE_NAME} gpu_count={GPU_COUNT}",
        flush=True,
    )
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
