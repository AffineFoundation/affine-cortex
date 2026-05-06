"""Thin Targon REST wrapper.

Maps 1:1 onto the Targon Hosted Apps (tha/v2) REST API. Two primitives matter:

    - **App** (app-xxx): optional grouping, not required to run a workload.
    - **Workload** (wrk-xxx): the actual container/pod. Create -> Deploy ->
      Monitor via /state. Targon has no restart endpoint, so "restart" is
      modelled as "redeploy" (POST /workloads/{uid}/deploy again).

Only fields the rest of our code needs are exposed; everything else flows
through `_request` for opportunistic use (ping, logs, events).
"""

import os
from typing import Any, Dict, List, Optional

import aiohttp

from affine.core.setup import logger


DEFAULT_API_URL = "https://api.targon.com/tha/v2"
# Parity with Chutes: miners deploy via build_sglang_chute, so our Targon
# deployments default to sglang too (different vendor image, same engine).
DEFAULT_ENGINE = "sglang"
DEFAULT_IMAGE_BY_ENGINE = {
    "sglang": "lmsysorg/sglang:latest",
    "vllm": "vllm/vllm-openai:latest",
}
DEFAULT_RESOURCE_NAME = "h100-small"
DEFAULT_WORKLOAD_PORT = 8000
# 80 GB default is enough for most Affine miner models (up to ~32B fp16
# with HF cache + some headroom).
DEFAULT_VOLUME_SIZE_MB = 80_000


def external_url(workload_uid: str, port: int = DEFAULT_WORKLOAD_PORT) -> str:
    """Externally-routable URL for a Targon RENTAL workload."""
    return f"https://{workload_uid}-{port}.caas.targon.com"


# Targon resource name convention inferred from observed workloads:
#   h100-small=1, h200-xlarge=8. Middle sizes (medium/large) are best-guess
#   and may not exist for every GPU family. If Targon rejects a mapping,
#   operators can always pass an explicit --resource.
_GPU_SIZE_TABLE = {
    "h100": {1: "h100-small", 2: "h100-medium", 4: "h100-large", 8: "h100-xlarge"},
    "h200": {1: "h200-small", 2: "h200-medium", 4: "h200-large", 8: "h200-xlarge"},
    "b200": {1: "b200-small", 2: "b200-medium", 4: "b200-large", 8: "b200-xlarge"},
}


def resource_name_for(gpu_type: str, gpu_count: int) -> Optional[str]:
    """Map (gpu_type, gpu_count) to a Targon resource name. None if unknown."""
    table = _GPU_SIZE_TABLE.get((gpu_type or "").lower())
    if not table:
        return None
    return table.get(int(gpu_count))


TARGON_GPU_TYPE = "h200"


def fixed_gpu_count() -> Optional[int]:
    """Operator override for per-deployment GPU count.

    When set, the deployer ignores the miner's chute ``node_selector.gpu_count``
    and pins every Targon workload to this many GPUs (TP=count, DP=1). Lets the
    operator size the pool to their physical capacity without depending on
    whatever the miner happened to build for.

    Default 2 because the operator scenario this exists for is "I have N
    H200 boxes, run K miners on each" — 2 GPUs/miner doubles the parallel
    miners we can host versus the 4-GPU chute defaults common in the network.
    """
    raw = os.getenv("TARGON_FIXED_GPU_COUNT", "2").strip()
    if not raw or raw.lower() in ("0", "off", "false", "none", ""):
        return None
    try:
        v = int(raw)
        return v if v > 0 else None
    except ValueError:
        return None


def derive_deployment_args_from_chute(
    chute_info: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Mirror a miner's Chutes deployment shape on Targon.

    Reads the Chutes API response and returns kwargs to splat into
    ``TargonClient.create_deployment``. Silently drops any field the chute
    config doesn't expose — callers fall back to env-driven defaults.

    Fields mapped:
        node_selector.gpu_count    -> gpu_count (= tensor_parallel)
        image.name (sglang|vllm)   -> engine
        (gpu_type pinned to h200)  -> resource_name

    Not mapped (Chutes API doesn't expose them):
        --max-model-len / --mem-fraction / --chunked-prefill-size etc.
    """
    if not chute_info:
        return {}
    out: Dict[str, Any] = {}
    ns = chute_info.get("node_selector") or {}

    gpu_count = ns.get("gpu_count")
    if isinstance(gpu_count, int) and gpu_count > 0:
        out["gpu_count"] = gpu_count
        resolved = resource_name_for(TARGON_GPU_TYPE, gpu_count)
        if resolved:
            out["resource_name"] = resolved

    # Engine: image.name is the actual runtime. standard_template is always
    # "vllm" for Affine chutes regardless of what they actually run, so
    # prefer image.name.
    img_name = ((chute_info.get("image") or {}).get("name") or "").lower()
    if img_name in ("sglang", "vllm"):
        out["engine"] = img_name

    return out


# sglang tool-call parser names by model family. None = don't pass the flag.
_TOOL_CALL_PARSER_HINTS = [
    ("qwen", "qwen"),
    ("llama-3", "llama3"),
    ("llama3", "llama3"),
    ("llama-4", "llama3"),     # llama3 parser covers both 3.x and 4
    ("deepseek", "deepseekv3"),
    ("mistral", "mistral"),
]


def _infer_tool_call_parser(model_hf_repo: str) -> Optional[str]:
    name = (model_hf_repo or "").lower()
    for needle, parser in _TOOL_CALL_PARSER_HINTS:
        if needle in name:
            return parser
    # Affine miners predominantly fine-tune Qwen; default to qwen unless the
    # operator explicitly disables via TARGON_SGLANG_TOOL_CALL_PARSER=none.
    if "affine" in name:
        return "qwen"
    return None


class TargonClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        default_image: Optional[str] = None,
        data_volume_mount: str = "/data",
        default_resource_name: Optional[str] = None,
        default_engine: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("TARGON_API_KEY", "")
        self.api_url = (api_url or os.getenv("TARGON_API_URL", "") or DEFAULT_API_URL).rstrip("/")
        self.default_engine = (default_engine or os.getenv("TARGON_ENGINE", DEFAULT_ENGINE)).lower()
        self.default_image = (
            default_image
            or os.getenv("TARGON_DEFAULT_IMAGE")
            or DEFAULT_IMAGE_BY_ENGINE.get(self.default_engine, DEFAULT_IMAGE_BY_ENGINE["sglang"])
        )
        self.data_volume_mount = data_volume_mount or os.getenv(
            "TARGON_DATA_VOLUME_MOUNT", "/data"
        )
        self.default_resource_name = (
            default_resource_name
            or os.getenv("TARGON_RESOURCE_NAME", DEFAULT_RESOURCE_NAME)
        )

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.api_url)

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        timeout: int = 15,
        expect_json: bool = True,
    ) -> Optional[Any]:
        if not self.configured:
            logger.debug(f"TargonClient not configured, skipping {method} {path}")
            return None
        url = f"{self.api_url}{path}"
        client_timeout = aiohttp.ClientTimeout(total=timeout)
        try:
            async with aiohttp.ClientSession(timeout=client_timeout) as session:
                async with session.request(
                    method, url, headers=self._headers(), json=json, params=params,
                ) as resp:
                    body_text = await resp.text()
                    if resp.status >= 400:
                        logger.warning(
                            f"Targon {method} {path} -> HTTP {resp.status}: {body_text[:400]}"
                        )
                        return None
                    if resp.status == 204 or not body_text:
                        return {}
                    if expect_json:
                        try:
                            return await resp.json(content_type=None)
                        except Exception:
                            return {"_raw": body_text}
                    return body_text
        except Exception as e:
            logger.debug(f"Targon {method} {path} error: {e}")
            return None

    # ---------- connectivity ----------

    # ---------- volumes (RENTAL-only; used to persist model weights) ----------

    async def list_volumes(self, *, limit: int = 50) -> Optional[Dict[str, Any]]:
        return await self._request("GET", "/volumes", params={"limit": limit})

    async def get_volume(self, volume_uid: str) -> Optional[Dict[str, Any]]:
        return await self._request("GET", f"/volumes/{volume_uid}")

    async def find_volume_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Case-sensitive exact match by name. Paginates through pages."""
        cursor = None
        while True:
            params: Dict[str, Any] = {"limit": 50}
            if cursor:
                params["cursor"] = cursor
            page = await self._request("GET", "/volumes", params=params)
            if not page:
                return None
            for v in page.get("items", []) or []:
                if v.get("name") == name:
                    return v
            cursor = page.get("next_cursor")
            if not cursor:
                return None

    async def create_volume(
        self, *, name: str, size_in_mb: int,
        resource_name: Optional[str] = None,
    ) -> Optional[str]:
        body = {
            "name": name,
            "size_in_mb": int(size_in_mb),
            # Resource name: Targon may pick a sensible default if omitted;
            # volumes attach to a compatible GPU pool.
        }
        if resource_name:
            body["resource_name"] = resource_name
        result = await self._request("POST", "/volumes", json=body, timeout=30)
        if not result:
            return None
        return result.get("uid")

    async def ensure_volume(
        self, *, name: str, size_in_mb: int = 80_000,
        resource_name: Optional[str] = None,
    ) -> Optional[str]:
        """Idempotent: reuse an existing volume with this name, else create."""
        existing = await self.find_volume_by_name(name)
        if existing:
            return existing.get("uid")
        return await self.create_volume(
            name=name, size_in_mb=size_in_mb, resource_name=resource_name,
        )

    async def delete_volume(self, volume_uid: str) -> bool:
        result = await self._request(
            "DELETE", f"/volumes/{volume_uid}", expect_json=False,
        )
        return result is not None

    async def list_workloads(
        self, *, limit: int = 50, status: Optional[str] = None, project_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        params: Dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        if project_id:
            params["project_id"] = project_id
        return await self._request("GET", "/workloads", params=params)

    async def list_apps(self, *, limit: int = 50) -> Optional[Dict[str, Any]]:
        return await self._request("GET", "/apps", params={"limit": limit})

    # ---------- workload lifecycle ----------

    async def create_deployment(
        self,
        model_hf_repo: str,
        revision: str,
        *,
        uid: int,
        hotkey: str,
        image: Optional[str] = None,
        resource_name: Optional[str] = None,
        name: Optional[str] = None,
        volumes: Optional[List[Dict[str, Any]]] = None,
        port: int = DEFAULT_WORKLOAD_PORT,
        gpu_count: int = 1,
        tensor_parallel: Optional[int] = None,
        data_parallel: Optional[int] = None,
        engine: Optional[str] = None,
    ) -> Optional[str]:
        """Create + deploy a Targon RENTAL workload for `model_hf_repo`@`revision`.

        Returns the workload uid on success, or None on failure.

        SERVERLESS was tested but Targon's Knative edge kept sglang revisions
        stuck in provisioning (4/4 failed). RENTAL routes via caas.targon.com
        directly and works end-to-end with the public sglang image as PID 1.
        """
        eng = (engine or self.default_engine).lower()
        image = image or DEFAULT_IMAGE_BY_ENGINE.get(eng, self.default_image)
        resource = resource_name or self.default_resource_name
        mount = self.data_volume_mount

        envs = [
            {"name": "HF_HOME", "value": mount},
            {"name": "TRANSFORMERS_CACHE", "value": mount},
            {"name": "HF_HUB_CACHE", "value": mount},
            {"name": "MODEL_ID", "value": model_hf_repo},
            {"name": "MODEL_REVISION", "value": revision},
            # Allow sglang to honor a context_length larger than the model's
            # derived max_position_embeddings. Required for any miner whose
            # Chutes config sets a >40k context (most do, even when the HF
            # config caps at 40960). Chutes' nightly sglang images have this
            # ON by default; the public lmsysorg image needs the env flag.
            {"name": "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", "value": "1"},
        ]
        hf_token = os.getenv("HF_TOKEN", "")
        if hf_token:
            envs.append({"name": "HF_TOKEN", "value": hf_token})
            envs.append({"name": "HUGGING_FACE_HUB_TOKEN", "value": hf_token})

        # sglang's upstream image inherits the NVIDIA CUDA entrypoint wrapper,
        # which misparses CMDs starting with "--" (`exec --: invalid option`).
        # Explicit `commands` bypass the wrapper. vLLM's image has a working
        # ENTRYPOINT so we pass only args there.
        commands: Optional[List[str]] = (
            ["python", "-m", "sglang.launch_server"] if eng == "sglang" else None
        )
        # Default 65536: matches Chutes-side production deploys for current
        # Affine miners (Qwen3-based). sglang would reject context-length >
        # model's max_position_embeddings without the
        # SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 env var injected below.
        # 40k was too tight for SWE-INFINITE agent traces.
        max_model_len = os.getenv("TARGON_MAX_MODEL_LEN") or os.getenv("TARGON_VLLM_MAX_MODEL_LEN", "65536")
        # Default 0.8 is conservative enough to avoid KV cache OOM across the
        # diverse miner models (Qwen / Llama / Mistral). Operator overrides
        # win via env.
        mem_fraction = os.getenv("TARGON_MEM_FRACTION") or os.getenv("TARGON_VLLM_GPU_MEM_UTIL", "0.8")
        # sglang-specific perf knobs. chunked-prefill-size caps the per-step
        # prefill batch; tool-call-parser enables OpenAI tool_calls for Qwen-
        # family models (the predominant Affine miner architecture).
        chunked_prefill_size = os.getenv("TARGON_SGLANG_CHUNKED_PREFILL_SIZE", "4096")
        # Detect model family from HF path for tool-call-parser. sglang rejects
        # a wrong parser at startup so guessing is risky — infer from the repo
        # name when possible, else let operator override with the env var.
        tool_call_parser = os.getenv("TARGON_SGLANG_TOOL_CALL_PARSER") or _infer_tool_call_parser(model_hf_repo)

        # Determine tp / dp split. tp×dp must == gpu_count.
        # Fallback: if neither passed, default to all-tensor-parallel (tp=gpu_count, dp=1)
        # — this preserves prior behavior.
        if tensor_parallel is not None and data_parallel is not None:
            tp = int(tensor_parallel)
            dp = int(data_parallel)
        elif tensor_parallel is not None:
            tp = int(tensor_parallel)
            dp = max(1, gpu_count // max(tp, 1))
        elif data_parallel is not None:
            dp = int(data_parallel)
            tp = max(1, gpu_count // max(dp, 1))
        else:
            tp = gpu_count
            dp = 1

        if eng == "sglang":
            args = [
                "--model-path", model_hf_repo,
                "--revision", revision,
                "--download-dir", mount,
                "--host", "0.0.0.0",
                "--port", str(port),
                "--trust-remote-code",
                "--context-length", max_model_len,
                "--mem-fraction-static", mem_fraction,
                "--chunked-prefill-size", chunked_prefill_size,
            ]
            if tool_call_parser and tool_call_parser.lower() != "none":
                args += ["--tool-call-parser", tool_call_parser]
            # Use --tp / --dp (sglang accepts both --tp/--tp-size and --dp).
            # When dp > 1, each replica fits the full model on its own slice
            # of GPUs, freeing more KV-cache headroom per replica — required
            # for >40k context on smaller models that fit in a single GPU.
            if tp > 1:
                args += ["--tp", str(tp)]
            if dp > 1:
                args += ["--dp", str(dp)]
        else:  # vllm
            args = [
                "--model", model_hf_repo,
                "--revision", revision,
                "--download-dir", mount,
                "--host", "0.0.0.0",
                "--port", str(port),
                "--trust-remote-code",
                "--max-model-len", max_model_len,
                "--gpu-memory-utilization", mem_fraction,
            ]
            if tp > 1:
                args += ["--tensor-parallel-size", str(tp)]

        body: Dict[str, Any] = {
            "name": name or self._workload_name(
                model_hf_repo, revision, uid=uid, hotkey=hotkey,
            ),
            "image": image,
            "resource_name": resource,
            "type": "RENTAL",
            "ports": [{"port": port, "protocol": "TCP", "routing": "PROXIED"}],
            "envs": envs,
            "args": args,
        }
        if commands:
            body["commands"] = commands
        if volumes:
            body["volumes"] = volumes

        result = await self._request("POST", "/workloads", json=body, timeout=30)
        if not result:
            return None
        workload_uid = result.get("uid") or result.get("id")
        if not workload_uid:
            logger.warning(f"Targon create workload returned no uid: {result}")
            return None

        deployed = await self._request(
            "POST", f"/workloads/{workload_uid}/deploy", timeout=30,
        )
        if deployed is None:
            logger.warning(f"Targon workload {workload_uid} created but /deploy failed")

        return workload_uid

    async def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        state = await self._request("GET", f"/workloads/{deployment_id}/state")
        if state is None:
            return None
        return self._normalize_state(state)

    async def delete_deployment(self, deployment_id: str) -> bool:
        result = await self._request(
            "DELETE", f"/workloads/{deployment_id}", expect_json=False,
        )
        return result is not None

    async def restart_container(self, deployment_id: str) -> bool:
        """Targon has no restart endpoint. Redeploy the existing workload."""
        result = await self._request(
            "POST", f"/workloads/{deployment_id}/deploy", timeout=30,
        )
        return result is not None

    # ---------- helpers ----------

    # Every workload we create starts with this prefix so operators running
    # a shared Targon account can tell our rows apart from other tenants.
    WORKLOAD_NAME_PREFIX = "affine"

    @staticmethod
    def _sanitize_token(s: str, n: int) -> str:
        """Trim to n chars, lowercase, collapse non-[a-z0-9] into hyphens."""
        s = (s or "").lower()
        s = "".join(c if c.isalnum() or c == "-" else "-" for c in s)
        return s[:n].strip("-") or "x"

    @staticmethod
    def _workload_name(
        model_hf_repo: str, revision: str,
        *, uid: int, hotkey: str,
    ) -> str:
        """Build a Targon-valid workload name.

        Targon caps names at 32 chars, lowercase alphanumeric + hyphens.

        Format: affine-{modelname5}-{uid}-{hotkey5}-{revision5}
          e.g.  affine-affin-44-5fnfl-92540    (len=28)

        `modelname` = the part after the last `/` in the HF repo (or the
        whole thing if no slash). We take the first 5 chars — enough to
        distinguish most model families while staying inside the length
        budget. uid + hotkey5 disambiguate miners that share a model;
        rev5 disambiguates revisions of the same miner.
        """
        if uid is None or not hotkey:
            raise ValueError(
                "_workload_name requires uid and hotkey — they're the only "
                "tokens that disambiguate miners from each other"
            )
        prefix = TargonClient.WORKLOAD_NAME_PREFIX  # 'affine' (6)
        rev5 = TargonClient._sanitize_token((revision or "")[:5], 5) or "norev"
        repo_tail = (model_hf_repo or "").split("/")[-1]
        model5 = TargonClient._sanitize_token(repo_tail[:5], 5)
        hk5 = TargonClient._sanitize_token(hotkey[:5], 5)
        return f"{prefix}-{model5}-{uid}-{hk5}-{rev5}"

    @staticmethod
    def _normalize_state(state: Dict[str, Any]) -> Dict[str, Any]:
        """Map Targon's state shape onto BaseProvider expectations.

        Real state example (from /workloads list):
            {"status": "running", "message": "Running",
             "urls": [{"port": 8000, "url": "https://wrk-xxx-8000.caas.targon.com"}],
             "ready_replicas": 2, "total_replicas": 2}
        """
        status_str = str(state.get("status") or state.get("state") or "").lower()

        running = 0
        if "ready_replicas" in state:
            running = int(state.get("ready_replicas") or 0)
        else:
            replicas = state.get("replicas") or {}
            if isinstance(replicas, dict):
                running = int(replicas.get("running") or replicas.get("ready") or 0)
            elif isinstance(replicas, list):
                running = sum(
                    1 for r in replicas
                    if str(r.get("status", "")).lower() in {"running", "ready", "healthy"}
                )
        healthy = status_str in {"running", "ready", "active", "healthy"} and running > 0

        base_url = None
        urls = state.get("urls") or state.get("access_urls") or []
        if isinstance(urls, list) and urls:
            preferred = next(
                (u for u in urls if isinstance(u, dict) and int(u.get("port", 0) or 0) == DEFAULT_WORKLOAD_PORT),
                urls[0],
            )
            if isinstance(preferred, dict):
                base_url = preferred.get("url") or preferred.get("public_url")
            elif isinstance(preferred, str):
                base_url = preferred
        elif isinstance(urls, dict):
            base_url = urls.get("url") or urls.get("public_url")
        if base_url and not base_url.rstrip("/").endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"

        return {
            "running_instances": running,
            "healthy": healthy,
            "base_url": base_url,
            "model_identifier": state.get("model") or "",
            "raw": state,
        }


_singleton: Optional[TargonClient] = None


def get_targon_client() -> TargonClient:
    global _singleton
    if _singleton is None:
        _singleton = TargonClient()
    return _singleton
