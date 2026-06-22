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

import copy
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional

import aiohttp

from affine.core.setup import logger


@dataclass(frozen=True)
class InstanceHandle:
    provider: str
    instance_id: str
    ssh_url: str = ""
    public_inference_url: str = ""
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
    status_method: str = "GET"
    status_path: str = ""
    create_payload: Dict[str, Any] = field(default_factory=dict)
    timeout_sec: int = 60
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
        return cls(
            provider=provider,
            api_url=str(data.get("api_url") or ""),
            api_url_env=str(
                data.get("api_url_env")
                or f"{provider_upper}_AUTOSCALE_API_URL"
            ),
            api_key_env=str(data.get("api_key_env") or f"{provider_upper}_API_KEY"),
            auth_header=str(data.get("auth_header") or "Authorization"),
            auth_scheme=str(data.get("auth_scheme") or "Bearer"),
            extra_headers={
                str(k): str(v)
                for k, v in (data.get("extra_headers") or {}).items()
            },
            create_method=str(data.get("create_method") or "POST").upper(),
            create_path=str(data.get("create_path") or ""),
            delete_method=str(data.get("delete_method") or "DELETE").upper(),
            delete_path=str(data.get("delete_path") or ""),
            status_method=str(data.get("status_method") or "GET").upper(),
            status_path=str(data.get("status_path") or ""),
            create_payload=(
                dict(data.get("create_payload"))
                if isinstance(data.get("create_payload"), Mapping)
                else {}
            ),
            timeout_sec=int(data.get("timeout_sec") or 60),
            response_paths={
                str(k): _coerce_paths(v)
                for k, v in response_paths.items()
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
        result = await self._request(
            self.config.create_method,
            path,
            json=payload,
            timeout=self.config.timeout_sec,
        )
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
            public_inference_url=(
                self._extract(result, "public_inference_url") or ""
            ),
            raw=dict(result),
        )

    async def delete(self, instance_id: str) -> bool:
        if not instance_id:
            return False
        if not self.config.resolved_api_url or not self.config.delete_path:
            logger.warning(
                "gpu-autoscaler: provider=%s missing api_url/delete_path",
                self.config.provider,
            )
            return False
        path = _render_template(
            self.config.delete_path,
            {"instance_id": instance_id},
        )
        result = await self._request(
            self.config.delete_method,
            path,
            timeout=self.config.timeout_sec,
            expect_json=False,
        )
        return result is not None

    async def status(self, instance_id: str) -> Optional[Dict[str, Any]]:
        if not instance_id or not self.config.status_path:
            return None
        path = _render_template(
            self.config.status_path,
            {"instance_id": instance_id},
        )
        result = await self._request(
            self.config.status_method,
            path,
            timeout=self.config.timeout_sec,
        )
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
                        logger.warning(
                            "gpu-autoscaler: %s %s -> HTTP %s: %s",
                            method,
                            path,
                            resp.status,
                            text[:400],
                        )
                        return None
                    if resp.status == 204 or not text:
                        return {}
                    if not expect_json:
                        return {"_raw": text}
                    try:
                        return await resp.json(content_type=None)
                    except Exception:
                        return {"_raw": text}
        except Exception as e:
            logger.warning(
                "gpu-autoscaler: %s %s failed: %s: %s",
                method,
                path,
                type(e).__name__,
                e,
            )
            return None

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


def _deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in overrides.items():
        if (
            isinstance(value, Mapping)
            and isinstance(out.get(key), Mapping)
        ):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out
