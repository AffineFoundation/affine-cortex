"""Config-driven GPU instance API client.

The scheduler already has a first-class Targon workload client for model
deployments. Autoscaling persistent SSH GPU machines is a different layer:
providers may expose different instance-create APIs, while Affine only needs
three normalized fields back: instance id, SSH URL, and optional public
inference URL.

This module keeps that provider surface declarative so Lium/Targon API shapes
can be configured without baking private response schemas into the codebase.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional

import aiohttp

from affine.core.setup import logger


DEFAULT_TIMEOUT_SEC = 20 * 60


class InstanceAPIError(RuntimeError):
    """Base class for provider API failures."""


class InstanceAPIHTTPError(InstanceAPIError):
    def __init__(self, method: str, path: str, status_code: int, text: str):
        self.method = method
        self.path = path
        self.status_code = status_code
        self.text = text
        super().__init__(f"{method} {path} -> HTTP {status_code}: {text[:400]}")


class InstanceAPINotFoundError(InstanceAPIHTTPError):
    """Provider reports that the managed instance no longer exists."""


class InstanceAPITimeoutError(InstanceAPIError):
    pass


class InstanceAPIRequestError(InstanceAPIError):
    pass


@dataclass(frozen=True)
class InstanceHandle:
    provider: str
    instance_id: str
    ssh_url: str = ""
    public_inference_url: str = ""
    lease_expires_at: int = 0
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InstanceAPIConfig:
    provider: str
    api_url: str = ""
    api_url_env: str = ""
    api_key_env: str = ""
    auth_header: str = "Authorization"
    auth_scheme: str = "Bearer"
    extra_headers: Dict[str, str] = field(default_factory=dict)
    create_method: str = "POST"
    create_path: str = ""
    delete_method: str = "DELETE"
    delete_path: str = ""
    renew_method: str = "POST"
    renew_path: str = ""
    renew_payload: Dict[str, Any] = field(default_factory=dict)
    status_method: str = "GET"
    status_path: str = ""
    create_payload: Dict[str, Any] = field(default_factory=dict)
    timeout_sec: int = DEFAULT_TIMEOUT_SEC
    response_paths: Dict[str, Iterable[str]] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls, provider: str, payload: Optional[Mapping[str, Any]]
    ) -> "InstanceAPIConfig":
        data = dict(payload or {})
        provider_upper = provider.upper()
        response_paths = data.get("response_paths")
        if not isinstance(response_paths, Mapping):
            response_paths = {}
        status_path = str(data.get("status_path") or "")
        if not status_path and _uses_normalized_instance_api(data):
            status_path = "/instances/{instance_id}"
        return cls(
            provider=provider,
            api_url=str(data.get("api_url") or ""),
            api_url_env=str(
                data.get("api_url_env") or f"{provider_upper}_AUTOSCALE_API_URL"
            ),
            api_key_env=str(data.get("api_key_env") or f"{provider_upper}_API_KEY"),
            auth_header=str(data.get("auth_header") or "Authorization"),
            auth_scheme=str(data.get("auth_scheme") or "Bearer"),
            extra_headers={
                str(k): str(v) for k, v in (data.get("extra_headers") or {}).items()
            },
            create_method=str(data.get("create_method") or "POST").upper(),
            create_path=str(data.get("create_path") or ""),
            delete_method=str(data.get("delete_method") or "DELETE").upper(),
            delete_path=str(data.get("delete_path") or ""),
            renew_method=str(data.get("renew_method") or "POST").upper(),
            renew_path=str(data.get("renew_path") or ""),
            renew_payload=(
                dict(data.get("renew_payload"))
                if isinstance(data.get("renew_payload"), Mapping)
                else {}
            ),
            status_method=str(data.get("status_method") or "GET").upper(),
            status_path=status_path,
            create_payload=(
                dict(data.get("create_payload"))
                if isinstance(data.get("create_payload"), Mapping)
                else {}
            ),
            timeout_sec=int(data.get("timeout_sec") or DEFAULT_TIMEOUT_SEC),
            response_paths={
                str(k): _coerce_paths(v) for k, v in response_paths.items()
            },
        )

    @property
    def resolved_api_url(self) -> str:
        fallback_env = f"{self.provider.upper()}_API_URL"
        return (
            self.api_url
            or os.getenv(self.api_url_env, "")
            or os.getenv(fallback_env, "")
        ).rstrip("/")

    @property
    def api_key(self) -> str:
        return os.getenv(self.api_key_env, "")


class InstanceAPIClient:
    """Small async REST wrapper for provider instance lifecycle APIs."""

    DEFAULT_RESPONSE_PATHS = {
        "instance_id": ("instance_id", "id", "uid", "instance.id", "data.id"),
        "ssh_url": ("ssh_url", "ssh.url", "ssh", "data.ssh_url"),
        "public_inference_url": (
            "public_inference_url",
            "inference_url",
            "public_url",
            "data.public_inference_url",
        ),
        "lease_expires_at": (
            "lease_expires_at",
            "expires_at",
            "lease.expires_at",
            "data.lease_expires_at",
        ),
    }

    def __init__(self, config: InstanceAPIConfig):
        self.config = config

    @property
    def configured(self) -> bool:
        return bool(self.config.resolved_api_url and self.config.create_path)

    async def create(
        self,
        *,
        variables: Optional[Mapping[str, Any]] = None,
        payload_overrides: Optional[Mapping[str, Any]] = None,
    ) -> Optional[InstanceHandle]:
        if not self.configured:
            logger.warning(
                "gpu-autoscaler: provider=%s missing api_url/create_path",
                self.config.provider,
            )
            return None
        variables = dict(variables or {})
        path = _render_template(self.config.create_path, variables)
        payload = _deep_merge(
            self.config.create_payload,
            dict(payload_overrides or {}),
        )
        payload = _render_template(payload, variables)
        try:
            result = await self._request(
                self.config.create_method,
                path,
                json=payload,
                timeout=self.config.timeout_sec,
            )
        except InstanceAPIError as e:
            logger.warning(
                "gpu-autoscaler: provider=%s create failed: %s: %s",
                self.config.provider,
                type(e).__name__,
                e,
            )
            return None
        if not isinstance(result, Mapping):
            return None
        instance_id = self._extract(result, "instance_id")
        if not instance_id:
            logger.warning(
                "gpu-autoscaler: provider=%s create response had no instance id",
                self.config.provider,
            )
            return None
        return InstanceHandle(
            provider=self.config.provider,
            instance_id=instance_id,
            ssh_url=self._extract(result, "ssh_url") or "",
            public_inference_url=(self._extract(result, "public_inference_url") or ""),
            lease_expires_at=_int_value(self._extract(result, "lease_expires_at")),
            raw=dict(result),
        )

    async def delete(
        self,
        instance_id: str,
        *,
        variables: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        if not instance_id:
            return False
        if not self.config.resolved_api_url or not self.config.delete_path:
            logger.warning(
                "gpu-autoscaler: provider=%s missing api_url/delete_path",
                self.config.provider,
            )
            return False
        render_vars = {"instance_id": instance_id, **dict(variables or {})}
        path = _render_template(self.config.delete_path, render_vars)
        delete_payload = {
            key: value
            for key, value in render_vars.items()
            if key != "instance_id" and value not in (None, "")
        }
        try:
            result = await self._request(
                self.config.delete_method,
                path,
                json=delete_payload or None,
                timeout=self.config.timeout_sec,
                expect_json=False,
            )
        except InstanceAPIHTTPError as e:
            if _instance_not_found_error(e):
                logger.info(
                    "gpu-autoscaler: provider=%s instance=%s already deleted",
                    self.config.provider,
                    instance_id,
                )
                return True
            logger.warning(
                "gpu-autoscaler: provider=%s delete failed for instance=%s: %s",
                self.config.provider,
                instance_id,
                e,
            )
            return False
        except InstanceAPIError as e:
            logger.warning(
                "gpu-autoscaler: provider=%s delete failed for instance=%s: %s: %s",
                self.config.provider,
                instance_id,
                type(e).__name__,
                e,
            )
            return False
        return result is not None

    async def renew(self, instance_id: str) -> Optional[InstanceHandle]:
        if not instance_id:
            return None
        if not self.config.resolved_api_url or not self.config.renew_path:
            logger.warning(
                "gpu-autoscaler: provider=%s missing api_url/renew_path",
                self.config.provider,
            )
            return None
        variables = {"instance_id": instance_id}
        path = _render_template(self.config.renew_path, variables)
        payload = _render_template(self.config.renew_payload, variables)
        try:
            result = await self._request(
                self.config.renew_method,
                path,
                json=payload,
                timeout=self.config.timeout_sec,
            )
        except InstanceAPIHTTPError as e:
            if _instance_not_found_error(e):
                logger.warning(
                    "gpu-autoscaler: provider=%s renew found missing instance=%s: %s",
                    self.config.provider,
                    instance_id,
                    e,
                )
                raise InstanceAPINotFoundError(
                    e.method,
                    e.path,
                    e.status_code,
                    e.text,
                ) from e
            logger.warning(
                "gpu-autoscaler: provider=%s renew failed for instance=%s: %s: %s",
                self.config.provider,
                instance_id,
                type(e).__name__,
                e,
            )
            return None
        except InstanceAPIError as e:
            logger.warning(
                "gpu-autoscaler: provider=%s renew failed for instance=%s: %s: %s",
                self.config.provider,
                instance_id,
                type(e).__name__,
                e,
            )
            return None
        if not isinstance(result, Mapping):
            return None
        return InstanceHandle(
            provider=self.config.provider,
            instance_id=self._extract(result, "instance_id") or instance_id,
            ssh_url=self._extract(result, "ssh_url") or "",
            public_inference_url=(self._extract(result, "public_inference_url") or ""),
            lease_expires_at=_int_value(self._extract(result, "lease_expires_at")),
            raw=dict(result),
        )

    async def status(self, instance_id: str) -> Optional[Dict[str, Any]]:
        if not instance_id or not self.config.status_path:
            return None
        path = _render_template(
            self.config.status_path,
            {"instance_id": instance_id},
        )
        try:
            result = await self._request(
                self.config.status_method,
                path,
                timeout=self.config.timeout_sec,
            )
        except InstanceAPIHTTPError as e:
            if _instance_not_found_error(e):
                logger.warning(
                    "gpu-autoscaler: provider=%s status found missing "
                    "instance=%s: %s",
                    self.config.provider,
                    instance_id,
                    e,
                )
                raise InstanceAPINotFoundError(
                    e.method,
                    e.path,
                    e.status_code,
                    e.text,
                ) from e
            logger.warning(
                "gpu-autoscaler: provider=%s status failed for instance=%s: %s: %s",
                self.config.provider,
                instance_id,
                type(e).__name__,
                e,
            )
            return None
        except InstanceAPIError as e:
            logger.warning(
                "gpu-autoscaler: provider=%s status failed for instance=%s: %s: %s",
                self.config.provider,
                instance_id,
                type(e).__name__,
                e,
            )
            return None
        return dict(result) if isinstance(result, Mapping) else None

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        headers.update(self.config.extra_headers)
        api_key = self.config.api_key
        if api_key and self.config.auth_header:
            value = api_key
            if self.config.auth_scheme:
                value = f"{self.config.auth_scheme} {api_key}"
            headers[self.config.auth_header] = value
        return headers

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        timeout: int = 60,
        expect_json: bool = True,
    ) -> Optional[Any]:
        url = f"{self.config.resolved_api_url}{path}"
        if path and not path.startswith("/"):
            url = f"{self.config.resolved_api_url}/{path}"
        client_timeout = aiohttp.ClientTimeout(total=timeout)
        try:
            async with aiohttp.ClientSession(timeout=client_timeout) as session:
                async with session.request(
                    method,
                    url,
                    headers=self._headers(),
                    json=json,
                ) as resp:
                    text = await resp.text()
                    if resp.status >= 400:
                        raise InstanceAPIHTTPError(
                            method,
                            path,
                            resp.status,
                            text,
                        )
                    if resp.status == 204 or not text:
                        return {}
                    if not expect_json:
                        return {"_raw": text}
                    try:
                        return await resp.json(content_type=None)
                    except Exception:
                        return {"_raw": text}
        except asyncio.TimeoutError as e:
            raise InstanceAPITimeoutError(
                f"{method} {path} timed out after {timeout}s"
            ) from e
        except aiohttp.ClientError as e:
            raise InstanceAPIRequestError(
                f"{method} {path} transport failed: {e}"
            ) from e

    def _extract(self, payload: Mapping[str, Any], field: str) -> Optional[str]:
        paths = self.config.response_paths.get(field)
        if not paths:
            paths = self.DEFAULT_RESPONSE_PATHS.get(field, ())
        for path in paths:
            value = _json_path(payload, path)
            if value is None or value == "":
                continue
            return str(value)
        return None


def _coerce_paths(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        return ()
    if isinstance(value, Iterable):
        return tuple(str(v) for v in value)
    return ()


def _uses_normalized_instance_api(data: Mapping[str, Any]) -> bool:
    create_path = str(data.get("create_path") or "").rstrip("/")
    delete_path = str(data.get("delete_path") or "").rstrip("/")
    renew_path = str(data.get("renew_path") or "").rstrip("/")
    return (
        create_path == "/instances"
        or delete_path == "/instances/{instance_id}"
        or renew_path == "/instances/{instance_id}/renew"
    )


def _int_value(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _instance_not_found_error(error: InstanceAPIHTTPError) -> bool:
    payload = _error_payload(error.text)
    if not payload:
        return False

    source = str(payload.get("error_source") or payload.get("source") or "").lower()
    code = str(payload.get("code") or "").lower()
    provider_status = _int_value(payload.get("provider_status_code"))

    if source == "wrapper" or code == "route_not_found":
        return False
    if source != "provider":
        return False
    return provider_status == 404 or code in {
        "provider_instance_not_found",
        "instance_not_found",
        "pod_not_found",
        "workload_not_found",
    }


def _error_payload(text: str) -> Optional[Mapping[str, Any]]:
    try:
        payload = json.loads(text or "{}")
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, Mapping) else None


def _json_path(payload: Mapping[str, Any], path: str) -> Any:
    current: Any = payload
    for part in str(path).split("."):
        if current is None:
            return None
        if isinstance(current, Mapping):
            current = current.get(part)
            continue
        if isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return None
            continue
        return None
    return current


def _render_template(value: Any, variables: Mapping[str, Any]) -> Any:
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


def _deep_merge(
    base: Mapping[str, Any], overrides: Mapping[str, Any]
) -> Dict[str, Any]:
    out: Dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out
