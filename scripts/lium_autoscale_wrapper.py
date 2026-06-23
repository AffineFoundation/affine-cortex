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

import json
import os
import shlex
import subprocess
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Iterable, List, Mapping, Optional
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
PORT = int(_env("LIUM_SGLANG_PORT", "10001") or "10001")
WAIT_TIMEOUT = int(_env("LIUM_CREATE_WAIT_TIMEOUT_SEC", "900") or "900")
POLL_SECONDS = int(_env("LIUM_CREATE_POLL_SECONDS", "15") or "15")
ENDPOINT_SCHEME = _env("LIUM_ENDPOINT_SCHEME", "http").rstrip(":/") or "http"
PUBLIC_INFERENCE_URL = _env("LIUM_PUBLIC_INFERENCE_URL")
NAME_PREFIX = _env("LIUM_POD_NAME_PREFIX", "affine-autoscale")


class LiumWrapperError(RuntimeError):
    pass


class LiumClient:
    def __init__(self) -> None:
        if not API_BASE:
            raise LiumWrapperError("LIUM_API_BASE is required")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        if API_KEY:
            self.session.headers.update(
                {"Authorization": f"Bearer {API_KEY}", "X-API-Key": API_KEY}
            )

    def request(self, method: str, path: str, **kwargs: Any) -> Any:
        resp = self.session.request(
            method, f"{API_BASE}{path}", timeout=60, **kwargs
        )
        if resp.status_code in (204, 205):
            return {}
        if resp.status_code >= 400:
            raise LiumWrapperError(
                f"{method} {path} -> {resp.status_code}: {resp.text[:500]}"
            )
        if not resp.content:
            return {}
        return resp.json()

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
            if int(e.get("available_gpu_count") or e.get("gpu_count") or 0)
            >= GPU_COUNT
        ]
        if not candidates:
            raise LiumWrapperError(
                f"No Lium executor has {GPU_COUNT} available GPUs for {MACHINE_NAME}"
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
        raise LiumWrapperError(
            f"No public key found for LIUM_SSH_KEY_NAME={SSH_KEY_NAME!r}"
        )

    def resolve_template_id(self) -> str:
        if TEMPLATE_ID:
            return TEMPLATE_ID
        templates = self.items(self.request("GET", "/templates"))
        if not templates:
            raise LiumWrapperError("No Lium templates available")

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
            raise LiumWrapperError(f"Selected template has no id: {templates[0]}")
        return template_id

    def create(self, name: str) -> dict:
        executor = self.select_executor()
        executor_id = _instance_id(executor)
        if not executor_id:
            raise LiumWrapperError(f"Selected executor has no id: {executor}")
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
        result = self.request(
            "POST", f"/executors/{executor_id}/rent", json=payload
        )
        uid = _instance_id(result)
        pod = result.get("pod") if isinstance(result, dict) else None
        if not uid and isinstance(pod, dict):
            uid = _instance_id(pod)
        if not uid:
            uid = self.find_pod_by_name(name)
        if not uid:
            raise LiumWrapperError(f"Rent response did not include a pod id: {result}")
        return self.wait_ready(uid)

    def find_pod_by_name(self, name: str) -> str:
        for _ in range(5):
            for pod in self.items(self.request("GET", "/pods")):
                if pod.get("pod_name") == name or pod.get("name") == name:
                    return _instance_id(pod)
            time.sleep(1)
        return ""

    def wait_ready(self, uid: str) -> dict:
        start = time.time()
        last: Optional[dict] = None
        while time.time() - start < WAIT_TIMEOUT:
            pod = self.request("GET", f"/pods/{uid}")
            if isinstance(pod, dict):
                last = pod
            status = _status(pod)
            if status in {"failed", "error", "terminated", "deleted"}:
                raise LiumWrapperError(f"Pod {uid} {status}: {str(pod)[:500]}")
            ssh_url = _ssh_url(pod)
            if ssh_url and self._ssh_ready(ssh_url):
                return {
                    "instance_id": uid,
                    "ssh_url": ssh_url,
                    "public_inference_url": (
                        _openai_endpoint(PUBLIC_INFERENCE_URL)
                        or _public_url(pod, PORT)
                    ),
                    "lease_expires_at": (
                        _lease_expires_at(pod) or _default_lease_expires_at()
                    ),
                    "status": status,
                    "raw": _redacted_pod(pod),
                }
            time.sleep(POLL_SECONDS)
        raise LiumWrapperError(
            f"Pod {uid} did not become SSH-ready in {WAIT_TIMEOUT}s; "
            f"last={str(_redacted_pod(last or {}))[:800]}"
        )

    def delete(self, uid: str) -> bool:
        self.request("DELETE", f"/pods/{uid}")
        return True

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
        user, host, port = _parse_ssh_url(ssh_url)
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
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=15
            )
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


def _ssh_url(data: Mapping[str, Any]) -> str:
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
            internal = item.get("port") or item.get("target_port") or item.get(
                "container_port"
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


def _public_url(data: Mapping[str, Any], port: int) -> str:
    for url in _urls(data, port):
        return _openai_endpoint(url)
    parsed = _parse_ssh_cmd(data)
    host = parsed.get("host") or _parse_host_from_ssh_url(_ssh_url(data))
    return f"{ENDPOINT_SCHEME}://{host}:{port}/v1" if host else ""


def _urls(data: Mapping[str, Any], port: int) -> Iterable[str]:
    for value in (data.get("urls") or []):
        if isinstance(value, dict) and int(value.get("port") or port) == port:
            url = value.get("url")
            if url:
                yield str(url)
    state = data.get("state") if isinstance(data.get("state"), dict) else {}
    for value in (state.get("urls") or []):
        if isinstance(value, dict) and int(value.get("port") or port) == port:
            url = value.get("url")
            if url:
                yield str(url)
    for item in _port_items(data):
        internal = item.get("port") or item.get("target_port") or item.get(
            "container_port"
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
            item.get("public_port") or item.get("external_port") or item.get("host_port")
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
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if not self._authorized():
            self._send(401, {"error": "unauthorized"})
            return
        parsed = urlparse(self.path)
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) == 3 and parts[0] == "instances" and parts[2] == "renew":
            try:
                result = LiumClient().renew(parts[1])
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
            suffix = str(data.get("endpoint_name") or int(time.time()))
            safe_suffix = "".join(
                c if c.isalnum() or c == "-" else "-" for c in suffix.lower()
            ).strip("-")
            name = f"{NAME_PREFIX}-{safe_suffix}"[:63].strip("-")
            result = LiumClient().create(name)
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
            ok = LiumClient().delete(parts[1])
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
