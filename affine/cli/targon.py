"""`af targon ...` — admin-gated Targon provider management.

Six commands, one purpose each. The heavy lifting (Targon REST, DB state,
external probing) lives in small, composable helpers at the top of this
file so individual command bodies stay a few lines.

Commands
--------
  deploy      Create+deploy+wait+activate+smoke-test a miner's model on Targon
  list        List live Targon workloads (marks rows owned by this deploy)
  status      Unified health probe (Targon API, DB, and per-workload state)
  sync        Reconcile our DB with Targon's live workload list (both ways)
  restart     Redeploy a workload's revision (systemd auto-restart inside)
  teardown    Delete a Targon workload + mark the DB row deleted
"""

import asyncio
import contextlib
import json
import time
from typing import Any, Dict, Optional, Tuple

import aiohttp
import click

from affine.core.providers.targon_client import (
    DEFAULT_WORKLOAD_PORT,
    TargonClient,
    external_url,
    get_targon_client,
    resource_name_for,
)


# =========================================================================
# Constants
# =========================================================================

AFFINE_WORKLOAD_PREFIX = f"{TargonClient.WORKLOAD_NAME_PREFIX}-"  # "affine-"

# Fail-fast log signatures — any match aborts deploy wait immediately.
_FATAL_LOG_PATTERNS = (
    "Engine core initialization failed",
    "exec: --:",
    "ValueError: To serve at least one request",
    "RuntimeError: CUDA error",
    "torch.cuda.OutOfMemoryError",
    "AssertionError: No available memory",
    "OSError: No such file or directory",
    "huggingface_hub.errors.RepositoryNotFoundError",
    "401 Client Error: Unauthorized",
    "Image cannot be pulled",
)


# =========================================================================
# Shared helpers
# =========================================================================

def _run(coro):
    return asyncio.run(coro)


@contextlib.asynccontextmanager
async def _db_session():
    """Open DB client for the duration of the block; close on exit."""
    from affine.database import close_client, init_client
    await init_client()
    try:
        yield
    finally:
        await close_client()


async def _miner_info_by_uid(uid: int, api_base: str) -> Optional[Dict[str, Any]]:
    """Fetch a miner record from the Affine public API by UID."""
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as s:
        async with s.get(f"{api_base}/miners/uid/{uid}") as r:
            if r.status != 200:
                return None
            return await r.json()


async def _http_get(url: str, timeout: int = 10) -> Tuple[int, str]:
    """Return (status, body[:240]) for a GET. 0 on network error."""
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as s:
            async with s.get(url) as r:
                body = await r.text()
                return r.status, body[:240]
    except Exception as e:
        return 0, str(e)[:240]


async def _smoke_inference(endpoint: str, model: str) -> Tuple[bool, str]:
    """POST /v1/chat/completions with a trivial prompt; require completion tokens > 0."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 4,
        "temperature": 0,
    }
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        ) as s:
            async with s.post(
                f"{endpoint.rstrip('/')}/chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload,
            ) as r:
                if r.status != 200:
                    body = (await r.text())[:240]
                    return False, f"HTTP {r.status}  {body}"
                try:
                    data = await r.json(content_type=None)
                except Exception as e:
                    return False, f"HTTP 200 but invalid JSON: {e}"
                usage = (data or {}).get("usage", {}) if isinstance(data, dict) else {}
                if not usage.get("completion_tokens"):
                    return False, f"HTTP 200 but zero completion tokens: {str(data)[:240]}"
                return True, f"HTTP 200 tokens={usage.get('total_tokens','?')}"
    except Exception as e:
        return False, f"request error: {e}"


async def _get_workload(
    client: TargonClient, *,
    deployment_id: Optional[str] = None,
    uid: Optional[int] = None,
    affine_api: str = "https://api.affine.io/api/v1",
) -> Optional[Dict[str, Any]]:
    """Resolve either --deployment-id or --uid to a live Targon workload dict."""
    if deployment_id:
        state = await client.get_deployment_status(deployment_id)
        if state is None:
            return None
        return {"uid": deployment_id, "state": state.get("raw"), "base_url": external_url(deployment_id) + "/v1"}
    if uid is not None:
        miner = await _miner_info_by_uid(uid, affine_api)
        if not miner:
            return None
        # Find by name prefix (affine-{...}-{uid}-{hk5}-{rev5})
        wls = await client.list_workloads(limit=100) or {}
        for w in wls.get("items", []) or []:
            name = w.get("name", "")
            if name.startswith(AFFINE_WORKLOAD_PREFIX) and f"-{uid}-" in name:
                return {
                    "uid": w["uid"], "state": w.get("state"),
                    "base_url": external_url(w["uid"]) + "/v1",
                    "name": name,
                }
    return None


async def _extract_failure_reason(
    client: TargonClient, workload_uid: str,
) -> str:
    for params in ({"previous": "true", "tail": 500}, {"tail": 500}):
        logs = await client._request(
            "GET", f"/workloads/{workload_uid}/logs",
            params=params, expect_json=False, timeout=10,
        )
        if not logs:
            continue
        for line in str(logs).splitlines():
            low = line.lower()
            if any(k in low for k in ("error", "traceback", "exception", "exec: --:")):
                return line.strip()[:260]
    return "no error detail in logs"


async def _wait_for_healthy(
    client: TargonClient, workload_uid: str,
    *, timeout_sec: int, poll_sec: int,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Poll until workload is healthy, using external-probe + Targon-state + fail-fast.

    Returns (state_dict, failure_reason). state_dict is None on failure.
    """
    deadline = time.time() + timeout_sec
    start = time.time()
    last_printed: Optional[Tuple[str, str]] = None
    backoff_strikes = 0
    ext_base = external_url(workload_uid)

    click.echo(f"\nWaiting up to {timeout_sec}s for {workload_uid} to become healthy...")
    while time.time() < deadline:
        state = await client.get_deployment_status(workload_uid)
        if state:
            raw = state.get("raw") or {}
            status = raw.get("status") or raw.get("state") or "unknown"
            msg = (raw.get("message") or "")[:140]
            key = (status, msg)
            if key != last_printed:
                click.echo(
                    f"  [{int(time.time() - start)}s] status={status} "
                    f"ready={state['running_instances']} msg={msg}"
                )
                last_printed = key
            if state["healthy"] and state["running_instances"] > 0:
                return state, None
            if "back-off" in msg.lower() or "crashloop" in msg.lower():
                backoff_strikes += 1
                if backoff_strikes >= 2:
                    reason = await _extract_failure_reason(client, workload_uid)
                    return None, f"container crashlooping: {reason}"
            else:
                backoff_strikes = 0

        # Fatal log patterns from previous container.
        prev = await client._request(
            "GET", f"/workloads/{workload_uid}/logs",
            params={"previous": "true", "tail": 200}, expect_json=False, timeout=10,
        )
        if prev:
            txt = str(prev)
            for pat in _FATAL_LOG_PATTERNS:
                if pat in txt:
                    reason = await _extract_failure_reason(client, workload_uid)
                    return None, f"{pat} | {reason}"

        # External-probe fallback (bypasses Targon state aggregator lag).
        for path in ("/v1/models", "/health"):
            code, _ = await _http_get(f"{ext_base}{path}", timeout=5)
            if code == 200:
                click.echo(
                    f"  [{int(time.time() - start)}s] external {path} -> 200 "
                    f"(promoting to healthy)"
                )
                return {
                    "running_instances": 1, "healthy": True,
                    "base_url": f"{ext_base}/v1", "model_identifier": "",
                    "raw": {"status": "running", "source": "external_fallback"},
                }, None

        await asyncio.sleep(poll_sec)

    return None, f"timeout after {timeout_sec}s"


# =========================================================================
# Workload operations (reusable across commands)
# =========================================================================

async def _probe_workload_health(
    client: TargonClient, workload_uid: str, *,
    model_repo: Optional[str] = None,
) -> Tuple[bool, str]:
    """Is this workload serving?

    Probes: Targon /state + external /v1/models + optional /chat/completions
    smoke. Returns (healthy, one-line-detail).
    """
    state = await client.get_deployment_status(workload_uid)
    raw = (state or {}).get("raw") or {}
    state_status = raw.get("status") or "unknown"
    ready = raw.get("ready_replicas")
    ext = external_url(workload_uid)
    code_v1, _ = await _http_get(f"{ext}/v1/models", timeout=5)
    code_hx, _ = await _http_get(f"{ext}/health", timeout=5)
    external_ok = code_v1 == 200 or code_hx == 200

    if not external_ok:
        return False, (
            f"state={state_status} ready={ready} "
            f"/v1/models={code_v1} /health={code_hx}"
        )
    if model_repo:
        ok, smoke_detail = await _smoke_inference(f"{ext}/v1", model_repo)
        if not ok:
            return False, f"external OK but inference fails: {smoke_detail}"
        return True, f"state={state_status} ready={ready} inference={smoke_detail}"
    return True, f"state={state_status} ready={ready} external /v1/models=200"


async def _deploy_workload(
    uid: int, hotkey: str, revision: str, model_repo: str,
    *,
    gpu_type: str = "h100", gpu_count: int = 1,
    resource: Optional[str] = None,
    engine: Optional[str] = None,
    tensor_parallel: Optional[int] = None,
    data_parallel: Optional[int] = None,
    with_volume: bool = False, volume_size_mb: int = 80_000,
    no_wait: bool = False,
    timeout_sec: int = 1800, poll_sec: int = 15,
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Full deploy flow: create → register DB → wait → activate → smoke.

    Returns (workload_uid, healthy_state). On failure: auto-deletes the
    workload, marks DB deleted, returns (None, None).

    Shared by `af targon deploy` and `af targon check --recover`.
    """
    client = get_targon_client()
    if not client.configured:
        click.echo("FAIL: TARGON_API_KEY not set", err=True)
        return None, None

    resolved_resource = resource or resource_name_for(gpu_type, gpu_count)
    if not resolved_resource:
        click.echo(
            f"FAIL: no resource mapping for {gpu_type}/{gpu_count}",
            err=True,
        )
        return None, None
    client.default_resource_name = resolved_resource

    resolved_volumes = None
    if with_volume:
        vol_name = f"affine-model-{hotkey[:8]}-{revision[:8]}"
        click.echo(f"Ensuring volume {vol_name} ({volume_size_mb} MB)...")
        volume_uid = await client.ensure_volume(
            name=vol_name, size_in_mb=volume_size_mb,
            resource_name=resolved_resource,
        )
        if not volume_uid:
            click.echo("FAIL: volume ensure failed", err=True)
            return None, None
        resolved_volumes = [{
            "uid": volume_uid, "mount_path": client.data_volume_mount,
            "read_only": False,
        }]

    workload_name = client._workload_name(
        model_repo, revision, uid=uid, hotkey=hotkey,
    )
    workload_uid = await client.create_deployment(
        model_hf_repo=model_repo, revision=revision,
        gpu_count=gpu_count, tensor_parallel=tensor_parallel,
        data_parallel=data_parallel,
        engine=engine, volumes=resolved_volumes, name=workload_name,
    )
    if not workload_uid:
        click.echo("FAIL: create_deployment returned no uid", err=True)
        return None, None
    click.echo(f"OK: created workload {workload_uid}")

    base_url = external_url(workload_uid) + "/v1"
    db_ok = await _upsert_db_row(
        workload_uid, hotkey=hotkey, revision=revision,
        model_repo=model_repo, image=client.default_image,
        base_url=base_url, mount_path=client.data_volume_mount,
        status="deploying", instance_count=0,
    )
    click.echo("  DB: registered (status=deploying)" if db_ok else "  DB: skipped")

    if no_wait:
        return workload_uid, None

    state, failure = await _wait_for_healthy(
        client, workload_uid, timeout_sec=timeout_sec, poll_sec=poll_sec,
    )
    if not state:
        click.echo(f"\nFAILED: {failure}", err=True)
        click.echo("Cleaning up to avoid GPU spend...")
        await client.delete_deployment(workload_uid)
        if db_ok:
            await _mark_db_deleted(workload_uid)
        return None, None

    click.echo(f"\nHEALTHY: base_url={state.get('base_url') or base_url}")
    if db_ok:
        await _activate_db_row(
            workload_uid, instance_count=state["running_instances"],
            base_url=state.get("base_url") or base_url,
        )
        click.echo("  DB: activated (status=active)")
    ok, detail = await _smoke_inference(
        state.get("base_url") or base_url, model_repo,
    )
    click.echo(f"Inference smoke: {'OK' if ok else 'WARN'}: {detail}")
    return workload_uid, state


def _parse_uid_from_name(name: Optional[str]) -> Optional[int]:
    """Workload names follow affine-{model5}-{uid}-{hk5}-{rev5}."""
    if not name or not name.startswith(AFFINE_WORKLOAD_PREFIX):
        return None
    parts = name.split("-")
    if len(parts) < 5:
        return None
    with contextlib.suppress(ValueError):
        return int(parts[2])
    return None


# =========================================================================
# CLI commands (7)
# =========================================================================

@click.group()
def targon():
    """Manage Targon deployments."""


# -------------------- deploy --------------------

@targon.command("deploy")
@click.option("--uid", required=True, type=int, help="Miner UID")
@click.option("--affine-api", default="https://api.affine.io/api/v1",
              show_default=True)
@click.option("--gpu-type", default=None,
              type=click.Choice(["h100", "h200", "b200"], case_sensitive=False),
              help="Default: derived from Chutes config, else h200")
@click.option("--gpu-count", default=None, type=int,
              help="GPUs per replica. Default: derived from Chutes node_selector.gpu_count")
@click.option("--resource", default=None,
              help="Explicit Targon resource name (overrides gpu-type/count)")
@click.option("--engine", default=None,
              type=click.Choice(["sglang", "vllm"], case_sensitive=False),
              help="Default: derived from Chutes image.name")
@click.option("--tensor-parallel", default=None, type=int)
@click.option("--data-parallel", default=None, type=int,
              help="Replica count: each replica holds a full model copy. "
                   "Use with --tensor-parallel 1 to maximize per-replica "
                   "KV cache (lets long context fit on small models). "
                   "tp × dp must equal gpu_count.")
@click.option("--with-volume", is_flag=True,
              help="Attach persistent volume (for frequent redeploys of same revision)")
@click.option("--volume-size-mb", default=80_000, show_default=True, type=int)
@click.option("--model", default=None, help="Override HuggingFace repo")
@click.option("--max-model-len", default=None)
@click.option("--gpu-mem-util", default=None)
@click.option("--no-wait", is_flag=True, help="Return immediately; don't block until healthy")
@click.option("--timeout-sec", default=1800, show_default=True, type=int)
@click.option("--poll-sec", default=15, show_default=True, type=int)
@click.option("--dry-run", is_flag=True)
def deploy_cmd(
    uid: int, affine_api: str,
    gpu_type: Optional[str], gpu_count: Optional[int], resource: Optional[str],
    engine: Optional[str], tensor_parallel: Optional[int],
    data_parallel: Optional[int],
    with_volume: bool, volume_size_mb: int,
    model: Optional[str],
    max_model_len: Optional[str], gpu_mem_util: Optional[str],
    no_wait: bool, timeout_sec: int, poll_sec: int,
    dry_run: bool,
):
    """Deploy a miner's model to Targon (create + wait + activate + smoke-test).

    By default, gpu_count and engine are derived from the miner's Chutes config
    (same tensor parallelism and runtime as Chutes). gpu_type defaults to h200.
    All defaults can be overridden with explicit flags.
    """
    async def _go():
        miner = await _miner_info_by_uid(uid, affine_api)
        if not miner:
            click.echo(f"FAIL: could not fetch miner UID {uid}", err=True)
            return
        model_repo = model or miner.get("model")
        revision = miner.get("revision")
        hotkey = miner.get("hotkey")
        if not (model_repo and revision and hotkey):
            click.echo("FAIL: miner record missing model/revision/hotkey", err=True)
            return
        click.echo(
            f"UID {uid}: hotkey={hotkey[:12]}... rev={revision[:8]}... model={model_repo}"
        )

        # Mirror Chutes config unless the user pinned values with flags.
        from affine.core.providers.targon_client import (
            derive_deployment_args_from_chute, TARGON_GPU_TYPE,
        )
        from affine.utils.api_client import get_chute_info
        chute_args: Dict[str, Any] = {}
        if miner.get("chute_id"):
            chute_args = derive_deployment_args_from_chute(
                await get_chute_info(miner["chute_id"])
            )
        final_gpu_type = gpu_type or TARGON_GPU_TYPE
        final_gpu_count = gpu_count or chute_args.get("gpu_count", 1)
        final_engine = engine or chute_args.get("engine")

        import os
        if max_model_len:
            os.environ["TARGON_MAX_MODEL_LEN"] = str(max_model_len)
        if gpu_mem_util:
            os.environ["TARGON_MEM_FRACTION"] = str(gpu_mem_util)

        if dry_run:
            client = get_targon_client()
            resolved = resource or resource_name_for(final_gpu_type, final_gpu_count)
            click.echo(
                f"\nDRY-RUN: POST /workloads (RENTAL)\n"
                f"  name={client._workload_name(model_repo, revision, uid=uid, hotkey=hotkey)}\n"
                f"  engine={(final_engine or client.default_engine).lower()}\n"
                f"  resource={resolved}  gpu_count={final_gpu_count}  "
                f"tp={tensor_parallel or final_gpu_count}\n"
                f"  volume={'yes' if with_volume else '(none)'}"
            )
            return

        await _deploy_workload(
            uid, hotkey, revision, model_repo,
            gpu_type=final_gpu_type, gpu_count=final_gpu_count, resource=resource,
            engine=final_engine, tensor_parallel=tensor_parallel,
            data_parallel=data_parallel,
            with_volume=with_volume, volume_size_mb=volume_size_mb,
            no_wait=no_wait, timeout_sec=timeout_sec, poll_sec=poll_sec,
        )
    _run(_go())


# -------------------- check --------------------

@targon.command("check")
@click.option("--deployment-id", default=None, help="Workload UID to check")
@click.option("--uid", default=None, type=int,
              help="Resolve miner UID → our workload (by name prefix)")
@click.option("--affine-api", default="https://api.affine.io/api/v1", show_default=True)
@click.option("--recover", is_flag=True,
              help="On failure: restart → delete+redeploy, reusing deploy flow")
@click.option("--restart-wait-sec", default=300, show_default=True, type=int,
              help="How long to wait for a restart to recover before falling back "
                   "to delete+redeploy (--recover only)")
@click.option("--gpu-type", default="h100", show_default=True,
              type=click.Choice(["h100", "h200", "b200"], case_sensitive=False),
              help="GPU family for redeploy (--recover only)")
@click.option("--gpu-count", default=1, show_default=True, type=int,
              help="GPU count for redeploy (--recover only)")
@click.option("--engine", default=None,
              type=click.Choice(["sglang", "vllm"], case_sensitive=False))
@click.option("--with-volume", is_flag=True)
def check_cmd(
    deployment_id: Optional[str], uid: Optional[int], affine_api: str,
    recover: bool, restart_wait_sec: int,
    gpu_type: str, gpu_count: int,
    engine: Optional[str], with_volume: bool,
):
    """Verify a workload is serving. With --recover: restart → redeploy on failure.

    Use this when a RENTAL VM may have been recreated by Targon's infrastructure
    and the model isn't loading. --recover will first attempt an in-place restart;
    if the workload still isn't serving after --restart-wait-sec, it deletes the
    workload and runs the full deploy flow again.
    """
    if not deployment_id and uid is None:
        click.echo("FAIL: pass --deployment-id or --uid", err=True)
        return

    async def _go():
        client = get_targon_client()
        if not client.configured:
            click.echo("FAIL: TARGON_API_KEY not set", err=True)
            return

        info = await _get_workload(
            client, deployment_id=deployment_id, uid=uid, affine_api=affine_api,
        )
        if not info:
            click.echo("Workload not found on Targon.", err=True)
            if recover and uid is not None:
                click.echo("Running fresh deploy via --recover + --uid...")
                miner = await _miner_info_by_uid(uid, affine_api)
                if not miner:
                    click.echo(f"FAIL: cannot fetch miner UID {uid}", err=True); return
                await _deploy_workload(
                    uid, miner["hotkey"], miner["revision"], miner["model"],
                    gpu_type=gpu_type, gpu_count=gpu_count, engine=engine,
                    with_volume=with_volume,
                )
            return

        wid = info["uid"]
        # Resolve model_repo for inference smoke (from uid, or parse workload name).
        effective_uid = uid if uid is not None else _parse_uid_from_name(info.get("name"))
        miner_model: Optional[str] = None
        if effective_uid is not None:
            miner = await _miner_info_by_uid(effective_uid, affine_api)
            miner_model = (miner or {}).get("model")

        healthy, detail = await _probe_workload_health(
            client, wid, model_repo=miner_model,
        )
        click.echo(f"{wid}: {'OK' if healthy else 'UNHEALTHY'} — {detail}")
        if healthy or not recover:
            if not healthy and not recover:
                click.echo("Pass --recover to restart + redeploy on failure.")
            return

        # ----- Recovery: restart first, then delete+redeploy -----
        click.echo(f"\n[recover 1/2] restart_container {wid}")
        await client.restart_container(wid)
        deadline = time.time() + restart_wait_sec
        while time.time() < deadline:
            await asyncio.sleep(20)
            healthy, detail = await _probe_workload_health(
                client, wid, model_repo=miner_model,
            )
            click.echo(
                f"  [{int(time.time() - (deadline - restart_wait_sec))}s] "
                f"{'OK' if healthy else 'UNHEALTHY'} — {detail}"
            )
            if healthy:
                click.echo("recover complete via restart.")
                return

        click.echo(f"\n[recover 2/2] restart did not recover in {restart_wait_sec}s; "
                   f"delete + redeploy")
        if effective_uid is None:
            click.echo("FAIL: cannot parse UID from workload name; pass --uid to redeploy", err=True); return
        miner = await _miner_info_by_uid(effective_uid, affine_api)
        if not miner:
            click.echo(f"FAIL: cannot fetch miner UID {effective_uid}", err=True); return
        await client.delete_deployment(wid)
        await _mark_db_deleted(wid)
        await _deploy_workload(
            effective_uid, miner["hotkey"], miner["revision"], miner["model"],
            gpu_type=gpu_type, gpu_count=gpu_count, engine=engine,
            with_volume=with_volume,
        )
    _run(_go())


# -------------------- list --------------------

@targon.command("list")
@click.option("--from-db", is_flag=True,
              help="List DB rows instead of live Targon state")
@click.option("--all", "show_all", is_flag=True,
              help="Include status=deleted rows (only relevant with --from-db)")
def list_cmd(from_db: bool, show_all: bool):
    """List Targon workloads. Default = live; --from-db for DB view."""
    async def _go():
        if from_db:
            await _list_db(show_all)
        else:
            await _list_live()
    _run(_go())


async def _list_live():
    client = get_targon_client()
    if not client.configured:
        click.echo("FAIL: TARGON_API_KEY not set", err=True)
        return
    wls = await client.list_workloads(limit=100) or {}
    items = [
        w for w in wls.get("items", []) or []
        if (w.get("state") or {}).get("status") != "deleted"
    ]
    if not items:
        click.echo("No live Targon workloads.")
        return
    for w in items:
        st = w.get("state") or {}
        res = w.get("resource") or {}
        mark = "* " if (w.get("name") or "").startswith(AFFINE_WORKLOAD_PREFIX) else "  "
        click.echo(
            f"{mark}{w['uid']:<22}  status={st.get('status','?'):<11} "
            f"ready={st.get('ready_replicas','?')}/{st.get('total_replicas','?'):<3} "
            f"{res.get('name','?'):<14} ${w.get('cost_per_hour','?')}/h  "
            f"name={w.get('name','')}"
        )
    click.echo(f"  (* = owned by this deploy, prefix '{AFFINE_WORKLOAD_PREFIX}')")


async def _list_db(show_all: bool):
    async with _db_session():
        from affine.database.dao.targon_deployments import TargonDeploymentsDAO
        dao = TargonDeploymentsDAO()
        statuses = ("active", "deploying", "rebuilding", "failed")
        if show_all:
            statuses += ("deleted",)
        rows = []
        for s in statuses:
            rows.extend(await dao.list_by_status(s))
        if not rows:
            click.echo("No rows in targon_deployments.")
            return
        for d in rows:
            click.echo(
                f"  {d['deployment_id']:<22}  status={d['status']:<11} "
                f"instances={d.get('instance_count', 0):<3} "
                f"hotkey={d.get('hotkey','')[:12]}... "
                f"rev={d.get('revision','')[:8]}... "
                f"model={d.get('model_hf_repo','')}"
            )


# -------------------- status --------------------

@targon.command("status")
@click.option("--deployment-id", default=None, help="Specific workload UID to probe")
@click.option("--uid", default=None, type=int, help="Resolve miner UID to workload")
@click.option("--affine-api", default="https://api.affine.io/api/v1", show_default=True)
def status_cmd(deployment_id: Optional[str], uid: Optional[int], affine_api: str):
    """Health probe. No args = Targon API + DB + env check. Else focus on one workload."""
    async def _go():
        if not deployment_id and uid is None:
            await _status_all()
        else:
            await _status_workload(deployment_id, uid, affine_api)
    _run(_go())


async def _status_all():
    """Overall health: Targon API, env config, DB reachability."""
    client = get_targon_client()

    click.echo("== Targon API ==")
    click.echo(f"  URL: {client.api_url or '(unset)'}")
    click.echo(f"  Key: {'set ('+str(len(client.api_key))+'ch)' if client.api_key else 'UNSET'}")
    click.echo(f"  Resource default: {client.default_resource_name}")
    click.echo(f"  Engine default:   {client.default_engine}")
    if client.configured:
        wls = await client.list_workloads(limit=1)
        click.echo(f"  /workloads: {'OK' if wls is not None else 'ERR'}")

    click.echo("\n== DynamoDB ==")
    try:
        async with _db_session():
            from affine.database.dao.targon_deployments import TargonDeploymentsDAO
            rows = await TargonDeploymentsDAO().list_by_status("active")
            click.echo(f"  targon_deployments reachable; active rows: {len(rows)}")
    except Exception as e:
        click.echo(f"  ERR: {e.__class__.__name__}: {e}")


async def _status_workload(
    deployment_id: Optional[str], uid: Optional[int], affine_api: str,
):
    """Per-workload: Targon state + external /v1 probe + inference smoke."""
    client = get_targon_client()
    if not client.configured:
        click.echo("FAIL: TARGON_API_KEY not set", err=True)
        return
    info = await _get_workload(
        client, deployment_id=deployment_id, uid=uid, affine_api=affine_api,
    )
    if not info:
        click.echo("Not found.", err=True)
        return

    wid = info["uid"]
    click.echo(f"Workload: {wid}")
    click.echo(f"  state: {json.dumps(info.get('state'), default=str)[:400]}")
    ext = info["base_url"].rstrip("/v1")
    click.echo(f"\n  GET  {ext}/v1/models")
    code, body = await _http_get(f"{ext}/v1/models")
    click.echo(f"    HTTP {code}  {body[:120]}")

    # Smoke inference if miner model is known.
    if uid is not None:
        miner = await _miner_info_by_uid(uid, affine_api)
        model = (miner or {}).get("model")
        if model:
            click.echo(f"\n  POST {ext}/v1/chat/completions  model={model}")
            ok, detail = await _smoke_inference(f"{ext}/v1", model)
            click.echo(f"    {'OK' if ok else 'WARN'}: {detail}")


# -------------------- sync --------------------

@targon.command("sync")
@click.option("--dry-run", is_flag=True,
              help="Report what would change; don't write")
@click.option("--affine-api", default="https://api.affine.io/api/v1", show_default=True,
              help="Used to resolve hotkey+revision+model for rows we adopt")
def sync_cmd(dry_run: bool, affine_api: str):
    """Reconcile DB with live Targon: adopt our workloads + mark gone rows deleted."""
    async def _go():
        client = get_targon_client()
        if not client.configured:
            click.echo("FAIL: TARGON_API_KEY not set", err=True)
            return

        wls = await client.list_workloads(limit=200) or {}
        live = wls.get("items", []) or []
        live_ours = [
            w for w in live
            if (w.get("name") or "").startswith(AFFINE_WORKLOAD_PREFIX)
            and (w.get("state") or {}).get("status") != "deleted"
        ]
        live_uids = {w["uid"] for w in live_ours}

        async with _db_session():
            from affine.database.dao.targon_deployments import TargonDeploymentsDAO
            dao = TargonDeploymentsDAO()

            # 1. Mark DB rows deleted that no longer live on Targon.
            stale = []
            for s in ("active", "deploying", "rebuilding", "failed"):
                for d in await dao.list_by_status(s):
                    if d["deployment_id"] not in live_uids:
                        stale.append(d)

            # 2. Find live Affine workloads missing from DB.
            known = set()
            for s in ("active", "deploying", "rebuilding", "failed", "deleted"):
                for d in await dao.list_by_status(s):
                    known.add(d["deployment_id"])
            to_adopt = [w for w in live_ours if w["uid"] not in known]

            click.echo(
                f"sync summary: "
                f"{len(stale)} stale DB row(s) to mark deleted, "
                f"{len(to_adopt)} live workload(s) to adopt"
            )
            if dry_run:
                for d in stale:
                    click.echo(f"  [stale->delete] {d['deployment_id']}")
                for w in to_adopt:
                    click.echo(f"  [live->adopt]   {w['uid']}  name={w.get('name')}")
                click.echo("\n--dry-run: no writes.")
                return

            # Execute.
            for d in stale:
                await dao.mark_deleted(d["deployment_id"])
            for w in to_adopt:
                # Parse uid out of the name: affine-{model5}-{uid}-{hk5}-{rev5}
                parts = w.get("name", "").split("-")
                uid = None
                if len(parts) >= 5 and parts[0] == AFFINE_WORKLOAD_PREFIX.rstrip("-"):
                    with contextlib.suppress(ValueError):
                        uid = int(parts[2])
                miner = await _miner_info_by_uid(uid, affine_api) if uid is not None else None
                if not miner:
                    click.echo(
                        f"  skip {w['uid']}: cannot resolve miner info (uid={uid})",
                        err=True,
                    )
                    continue
                state = w.get("state") or {}
                ready = int(state.get("ready_replicas") or 0)
                base = external_url(w["uid"]) + "/v1"
                await dao.upsert_deployment(
                    deployment_id=w["uid"], hotkey=miner["hotkey"],
                    revision=miner["revision"], model_hf_repo=miner["model"],
                    image=client.default_image,
                    base_url=base,
                    instance_count=ready,
                    status="active" if ready > 0 else "deploying",
                    mount_path=client.data_volume_mount,
                )
            click.echo(
                f"sync complete: {len(stale)} marked deleted, {len(to_adopt)} adopted"
            )
    _run(_go())


# -------------------- restart --------------------

@targon.command("restart")
@click.option("--deployment-id", required=True)
def restart_cmd(deployment_id: str):
    """Redeploy a workload's revision (systemd + Targon restart it fresh)."""
    async def _go():
        client = get_targon_client()
        ok = await client.restart_container(deployment_id)
        click.echo(f"restart {deployment_id}: {'OK' if ok else 'FAILED'}")
        if ok:
            async with _db_session():
                from affine.database.dao.targon_deployments import TargonDeploymentsDAO
                await TargonDeploymentsDAO().set_status(deployment_id, "rebuilding")
    _run(_go())


# -------------------- delete --------------------

@targon.command("delete")
@click.option("--deployment-id", required=True)
@click.option("--yes", "-y", is_flag=True,
              help="Skip confirmation prompt (for scripting)")
def delete_cmd(deployment_id: str, yes: bool):
    """Delete a Targon workload and mark the DB row deleted."""
    async def _go():
        client = get_targon_client()

        # Show what we're about to delete.
        state = await client.get_deployment_status(deployment_id)
        if state:
            raw = state.get("raw") or {}
            click.echo(
                f"{deployment_id}  status={raw.get('status','?')}  "
                f"ready={raw.get('ready_replicas','?')}/{raw.get('total_replicas','?')}"
            )
        else:
            click.echo(f"{deployment_id}  (state unavailable)")

        if not yes and not click.confirm(
            f"Delete Targon workload {deployment_id}? This cannot be undone.",
            default=False,
        ):
            click.echo("Aborted.")
            return

        ok = await client.delete_deployment(deployment_id)
        click.echo(f"delete {deployment_id}: {'OK' if ok else 'FAILED'}")
        if ok:
            await _mark_db_deleted(deployment_id)
    _run(_go())


# =========================================================================
# DB helpers (small, reused)
# =========================================================================

async def _upsert_db_row(
    deployment_id: str, *,
    hotkey: str, revision: str, model_repo: str, image: str,
    base_url: str, mount_path: str,
    status: str, instance_count: int,
) -> bool:
    try:
        async with _db_session():
            from affine.database.dao.targon_deployments import TargonDeploymentsDAO
            await TargonDeploymentsDAO().upsert_deployment(
                deployment_id=deployment_id, hotkey=hotkey, revision=revision,
                model_hf_repo=model_repo, image=image,
                base_url=base_url, instance_count=instance_count, status=status,
                mount_path=mount_path,
            )
        return True
    except Exception:
        return False


async def _activate_db_row(
    deployment_id: str, *, instance_count: int, base_url: str,
) -> bool:
    try:
        async with _db_session():
            from affine.database.dao.targon_deployments import TargonDeploymentsDAO
            await TargonDeploymentsDAO().update_health(
                deployment_id, instance_count=instance_count, healthy=True,
                base_url=base_url,
            )
        return True
    except Exception:
        return False


async def _mark_db_deleted(deployment_id: str) -> bool:
    try:
        async with _db_session():
            from affine.database.dao.targon_deployments import TargonDeploymentsDAO
            await TargonDeploymentsDAO().mark_deleted(deployment_id)
        return True
    except Exception:
        return False
