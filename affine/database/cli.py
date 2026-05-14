"""
Database admin CLI.

Slim, focused on operations that still make sense post-refactor:
  - Bootstrap / inspect DynamoDB tables
  - Seed system_config from a JSON file (new flat shape: every top-level
    key is stored verbatim under ``CONFIG#PARAM#<key>``)
  - Seed/inspect the current champion
  - Maintain the miner blacklist and validator burn percentage
  - Drop sample_results rows by env+task range
  - Inspect, register, or remove individual miners (incl. system miners)

Removed: commands tied to the dead per-miner slot scheduler,
sampling-list rotation, TaskPoolDAO, AntiCopyDAO, or the targon_deployer
reconciler. Historical miner lifecycle state is still in MinerStatsDAO.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Optional

import click

from affine.cli.types import UID
from affine.database import close_client, init_client, init_tables
from affine.database.dao import MinerStatsDAO, MinersDAO, SampleResultsDAO, SystemConfigDAO
from affine.database.tables import delete_table, list_tables, reset_tables


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #


async def _cmd_init() -> None:
    await init_client()
    try:
        await init_tables()
        print("✓ Tables initialized")
    finally:
        await close_client()


async def _cmd_list() -> None:
    await init_client()
    try:
        tables = await list_tables()
        print(f"Found {len(tables)} tables:")
        for t in tables:
            print(f"  - {t}")
    finally:
        await close_client()


async def _cmd_reset(force: bool) -> None:
    if not force:
        if input("WARNING: this deletes ALL data. Type 'yes' to confirm: ").lower() != "yes":
            print("Aborted")
            return
    await init_client()
    try:
        await reset_tables()
        print("✓ Tables reset")
    finally:
        await close_client()


async def _cmd_reset_table(table_name: str, force: bool) -> None:
    from affine.database.schema import get_table_name

    full = get_table_name(table_name)
    if not force:
        if input(f"WARNING: delete all data in '{full}'. Type 'yes' to confirm: ").lower() != "yes":
            print("Aborted")
            return
    await init_client()
    try:
        await delete_table(full)
        await init_tables()
        print(f"✓ Reset '{full}'")
    finally:
        await close_client()


# --------------------------------------------------------------------------- #
# system_config load / get
# --------------------------------------------------------------------------- #


async def _cmd_load_config(json_file: str) -> None:
    """Upsert every top-level key from a JSON file into system_config.

    No sampling-list rotation, no smooth-transition logic — the new flow
    treats each entry as a plain KV pair. Existing keys are overwritten.
    """
    if not os.path.exists(json_file):
        print(f"Error: file '{json_file}' not found")
        sys.exit(1)
    try:
        with open(json_file) as f:
            payload = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON: {e}")
        sys.exit(1)
    if not isinstance(payload, dict):
        print("Error: top-level JSON must be an object")
        sys.exit(1)

    await init_client()
    try:
        dao = SystemConfigDAO()
        for key, value in payload.items():
            # When seeding ``environments``, resolve any
            # ``dataset_range_source`` so the static fallback in the
            # JSON file doesn't shrink an env that's accumulated new
            # task_ids since the file was last edited (SWE-INFINITE,
            # DISTILL grow over time).
            if key == "environments" and isinstance(value, dict):
                value = await _resolve_env_dataset_ranges(value)
            param_type = _param_type_of(value)
            await dao.set_param(
                param_name=key,
                param_value=value,
                param_type=param_type,
                description=f"Loaded from {os.path.basename(json_file)}",
                updated_by="cli:load-config",
            )
            if key == "environments" and isinstance(value, dict):
                print(f"  {key} ({len(value)} envs):")
                _print_env_summary(value)
            else:
                preview = _preview(value)
                print(f"  {key:<24} → {preview}")
        print(f"✓ Loaded {len(payload)} keys")
    finally:
        await close_client()


def _print_env_summary(environments: dict) -> None:
    """One line per env with the fields operators actually need to verify."""
    for env_name in sorted(environments):
        cfg = environments.get(env_name) or {}
        sampling = cfg.get("sampling") or {}
        sampling_on = cfg.get("enabled_for_sampling", False)
        scoring_on = cfg.get("enabled_for_scoring", True)
        count = sampling.get("sampling_count", "?")
        mode = sampling.get("sampling_mode", "?")
        rng = sampling.get("dataset_range")
        if sampling.get("dataset_range_source"):
            rng_str = "resolved-at-refresh"
        elif isinstance(rng, list) and rng:
            try:
                lo, hi = rng[0][0], rng[-1][1]
                rng_str = f"[{lo},{hi}]" if len(rng) == 1 else f"{len(rng)} ranges"
            except (IndexError, TypeError):
                rng_str = "?"
        else:
            rng_str = "—"
        print(
            f"      {env_name:<14} "
            f"sampling={'Y' if sampling_on else 'N'} "
            f"scoring={'Y' if scoring_on else 'N'} "
            f"count={count:<5} "
            f"mode={mode:<7} "
            f"range={rng_str}"
        )


async def _resolve_env_dataset_ranges(environments: dict) -> dict:
    """For each env with a ``dataset_range_source``, drop the static
    ``dataset_range`` from the seed payload. The scheduler resolves
    the range from the URL on every window refresh, so the seed value
    would only get overwritten anyway — and worse, would shrink the
    range below what's already accumulated in the DB if the JSON file
    is stale.

    No network call here; the URL fetch happens at window-refresh time
    inside ``FlowScheduler._refresh_task_ids``."""
    out = dict(environments)
    for env_name, env_cfg in out.items():
        if not isinstance(env_cfg, dict):
            continue
        sampling = env_cfg.get("sampling")
        if not isinstance(sampling, dict):
            continue
        if not sampling.get("dataset_range_source"):
            continue
        if "dataset_range" not in sampling:
            continue
        sampling = {k: v for k, v in sampling.items() if k != "dataset_range"}
        env_cfg = dict(env_cfg)
        env_cfg["sampling"] = sampling
        out[env_name] = env_cfg
        print(f"  · {env_name}: skipping static dataset_range "
              f"(dataset_range_source will resolve at window refresh)")
    return out


def _param_type_of(value) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return "str"


def _preview(value) -> str:
    s = json.dumps(value, separators=(",", ":"))
    return s if len(s) <= 80 else s[:77] + "..."


async def _cmd_get_config() -> None:
    await init_client()
    try:
        dao = SystemConfigDAO()
        params = await dao.list_all_configs()
        if not params:
            print("(empty)")
            return
        for p in sorted(params, key=lambda x: x.get("param_name", "")):
            name = p.get("param_name", "?")
            preview = _preview(p.get("param_value"))
            print(f"  {name:<24} = {preview}")
    finally:
        await close_client()


# --------------------------------------------------------------------------- #
# Champion
# --------------------------------------------------------------------------- #


async def _cmd_set_champion(hotkey: str, revision: str, since_block: Optional[int]) -> None:
    """Write the champion record. Looks up uid+model from miners table."""
    from affine.utils.subtensor import get_subtensor

    await init_client()
    try:
        miners = MinersDAO()
        all_miners = await miners.get_all_miners()
        matched = [m for m in all_miners
                   if m.get("hotkey") == hotkey and m.get("revision") == revision]
        if not matched:
            print(f"Error: no miner found with hotkey={hotkey[:16]}... revision={revision[:8]}...")
            sys.exit(1)
        m = matched[0]
        uid = int(m["uid"])
        model = m.get("model", "")

        if since_block is None:
            subtensor = await get_subtensor()
            since_block = int(await subtensor.get_current_block())

        cfg = SystemConfigDAO()
        existing = await cfg.get_param_value("champion", default=None)
        if existing:
            print(f"Replacing existing champion uid={existing.get('uid')} "
                  f"hotkey={existing.get('hotkey', '')[:12]}...")

        # ChampionRecord shape — Stage U dropped since_window_id (no
        # window-id concept anymore). deployment_id/base_url are populated
        # by the scheduler on the next tick when it brings up Targon.
        payload = {
            "uid": uid,
            "hotkey": hotkey,
            "revision": revision,
            "model": model,
            "deployment_id": None,
            "base_url": None,
            "since_block": since_block,
        }
        await cfg.set_param(
            param_name="champion",
            param_value=payload,
            param_type="dict",
            description="Current champion identity",
            updated_by="cli:set-champion",
        )
        print(f"✓ Champion set: uid={uid} since_block={since_block}")
    finally:
        await close_client()


# --------------------------------------------------------------------------- #
# Blacklist
# --------------------------------------------------------------------------- #


async def _cmd_blacklist_list() -> None:
    await init_client()
    try:
        items = await SystemConfigDAO().get_blacklist()
        if not items:
            print("(empty)")
            return
        for hk in items:
            print(f"  {hk}")
        print(f"({len(items)} entries)")
    finally:
        await close_client()


async def _cmd_blacklist_add(hotkeys: list) -> None:
    await init_client()
    try:
        await SystemConfigDAO().add_to_blacklist(hotkeys, updated_by="cli")
        print(f"✓ Added {len(hotkeys)} hotkey(s)")
    finally:
        await close_client()


async def _cmd_blacklist_remove(hotkeys: list) -> None:
    await init_client()
    try:
        await SystemConfigDAO().remove_from_blacklist(hotkeys, updated_by="cli")
        print(f"✓ Removed {len(hotkeys)} hotkey(s)")
    finally:
        await close_client()


async def _cmd_blacklist_clear() -> None:
    if input("Clear ALL blacklist entries. Type 'yes' to confirm: ").lower() != "yes":
        print("Aborted")
        return
    await init_client()
    try:
        await SystemConfigDAO().set_blacklist([], updated_by="cli")
        print("✓ Blacklist cleared")
    finally:
        await close_client()


# --------------------------------------------------------------------------- #
# Burn
# --------------------------------------------------------------------------- #


async def _cmd_set_burn(pct: float) -> None:
    if not 0.0 <= pct <= 1.0:
        print("Error: burn percentage must be in [0.0, 1.0]")
        sys.exit(1)
    await init_client()
    try:
        await SystemConfigDAO().set_param(
            param_name="validator_burn_percentage",
            param_value=pct,
            param_type="float",
            description="Validator burn percentage",
            updated_by="cli",
        )
        print(f"✓ Burn percentage set to {pct}")
    finally:
        await close_client()


async def _cmd_get_burn() -> None:
    await init_client()
    try:
        v = await SystemConfigDAO().get_param_value("validator_burn_percentage", default=0.0)
        print(f"validator_burn_percentage = {v}")
    finally:
        await close_client()


# --------------------------------------------------------------------------- #
# Sample maintenance
# --------------------------------------------------------------------------- #


async def _cmd_delete_samples_by_range(
    hotkey: Optional[str], revision: Optional[str], env: str,
    start_task_id: int, end_task_id: int, force: bool,
) -> None:
    if not force:
        scope = (f"miner={hotkey[:12]}... rev={revision[:8]}..."
                 if hotkey and revision else "ALL miners")
        prompt = (f"WARNING: delete samples for {scope} in env={env} "
                  f"task_id=[{start_task_id}, {end_task_id}). Type 'yes' to confirm: ")
        if input(prompt).lower() != "yes":
            print("Aborted")
            return
    await init_client()
    try:
        dao = SampleResultsDAO()
        if hotkey and revision:
            n = await dao.delete_samples_by_task_range(
                miner_hotkey=hotkey, model_revision=revision, env=env,
                start_task_id=start_task_id, end_task_id=end_task_id,
            )
        else:
            n = await dao.delete_all_samples_by_task_range(
                env=env, start_task_id=start_task_id, end_task_id=end_task_id,
            )
        print(f"✓ Deleted {n} samples")
    finally:
        await close_client()


# --------------------------------------------------------------------------- #
# Miner lookup / system miners
# --------------------------------------------------------------------------- #


async def _cmd_get_miner(hotkey: Optional[str], uid: Optional[int]) -> None:
    await init_client()
    try:
        dao = MinersDAO()
        if uid is not None:
            m = await dao.get_miner_by_uid(uid)
        elif hotkey:
            m = await dao.get_miner_by_hotkey(hotkey)
        else:
            print("Error: --hotkey or --uid required")
            sys.exit(1)
        if not m:
            print("(not found)")
            return
        state = {}
        if m.get("hotkey") and m.get("revision"):
            state = await MinerStatsDAO().get_challenge_state(
                m["hotkey"], m["revision"],
            )
        for k in ("uid", "hotkey", "model", "revision", "is_valid",
                  "first_block", "block_number", "invalid_reason", "model_hash"):
            if k in m and m[k] is not None:
                print(f"  {k:<18} = {m[k]}")
        for k in ("challenge_status", "termination_reason"):
            if state.get(k):
                print(f"  {k:<18} = {state[k]}")
    finally:
        await close_client()


async def _cmd_list_system_miners() -> None:
    await init_client()
    try:
        rows = await SystemConfigDAO().get_system_miners()
        if not rows:
            print("(empty)")
            return
        for uid_str, payload in sorted(rows.items(), key=lambda x: int(x[0])):
            print(f"  UID {uid_str:<6} → {payload.get('model', '?')}")
    finally:
        await close_client()


async def _cmd_set_miner(uid: int, model: str) -> None:
    """Register a system miner (UID > 1000)."""
    if uid <= 1000:
        # Auto-shift small UIDs into the system range — matches the old
        # behavior documented in the SystemConfigDAO.set_system_miner check.
        uid = 1000 + uid
        print(f"(auto-shifted UID into system range: {uid})")
    await init_client()
    try:
        await SystemConfigDAO().set_system_miner(uid=uid, model=model, updated_by="cli")
        print(f"✓ System miner UID={uid} set to model={model}")
    finally:
        await close_client()


async def _cmd_delete_miner(uid: int) -> None:
    if uid <= 1000:
        uid = 1000 + uid
    await init_client()
    try:
        deleted = await SystemConfigDAO().delete_system_miner(uid=uid, updated_by="cli")
        print("✓ Deleted" if deleted else "(not found)")
    finally:
        await close_client()


# --------------------------------------------------------------------------- #
# Click wiring
# --------------------------------------------------------------------------- #


@click.group()
def db():
    """Database admin commands."""


@db.command("init")
def init():
    """Initialize all DynamoDB tables."""
    asyncio.run(_cmd_init())


@db.command("list")
def list_cmd():
    """List all tables."""
    asyncio.run(_cmd_list())


@db.command("reset")
@click.option("--yes", "force", is_flag=True, help="Skip confirmation")
def reset(force):
    """Reset all tables (delete and recreate)."""
    asyncio.run(_cmd_reset(force))


@db.command("reset-table")
@click.option("--table", required=True, help="Base table name (e.g. miners)")
@click.option("--yes", "force", is_flag=True, help="Skip confirmation")
def reset_table(table, force):
    """Reset a single table."""
    asyncio.run(_cmd_reset_table(table, force))


@db.command("load-config")
@click.option(
    "--json-file",
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "system_config.json"),
    help="Path to JSON config file",
)
def load_config(json_file):
    """Upsert every top-level key into system_config."""
    asyncio.run(_cmd_load_config(json_file))


@db.command("get-config")
def get_config():
    """Print every system_config key."""
    asyncio.run(_cmd_get_config())


@db.command("set-champion")
@click.option("--hotkey", required=True, help="Champion miner's hotkey")
@click.option("--revision", required=True, help="Model revision hash")
@click.option("--since-block", type=int, default=None,
              help="Block to record as champion start (default: current block)")
def set_champion(hotkey, revision, since_block):
    """Set the initial champion before the first scorer run."""
    asyncio.run(_cmd_set_champion(hotkey, revision, since_block))


@db.group()
def blacklist():
    """Manage miner blacklist."""


@blacklist.command("list")
def blacklist_list():
    asyncio.run(_cmd_blacklist_list())


@blacklist.command("add")
@click.argument("hotkeys", nargs=-1, required=True)
def blacklist_add(hotkeys):
    asyncio.run(_cmd_blacklist_add(list(hotkeys)))


@blacklist.command("remove")
@click.argument("hotkeys", nargs=-1, required=True)
def blacklist_remove(hotkeys):
    asyncio.run(_cmd_blacklist_remove(list(hotkeys)))


@blacklist.command("clear")
def blacklist_clear():
    asyncio.run(_cmd_blacklist_clear())


@db.command("set-burn")
@click.argument("percentage", type=float)
def set_burn(percentage):
    """Set validator burn percentage (0.0 to 1.0)."""
    asyncio.run(_cmd_set_burn(percentage))


@db.command("get-burn")
def get_burn():
    asyncio.run(_cmd_get_burn())


@db.command("delete-samples-by-range")
@click.option("--hotkey", default=None)
@click.option("--revision", default=None)
@click.option("--env", required=True)
@click.option("--start-task-id", required=True, type=int)
@click.option("--end-task-id", required=True, type=int)
@click.option("--yes", "force", is_flag=True)
def delete_samples_by_range(hotkey, revision, env, start_task_id, end_task_id, force):
    """Delete sample_results rows by env + task_id range."""
    asyncio.run(_cmd_delete_samples_by_range(
        hotkey, revision, env, start_task_id, end_task_id, force))


@db.command("get-miner")
@click.option("--hotkey", default=None)
@click.option("--uid", type=UID, default=None)
def get_miner(hotkey, uid):
    """Look up a miner by hotkey or UID."""
    asyncio.run(_cmd_get_miner(hotkey, uid))


@db.command("list-system-miners")
def list_system_miners():
    """List configured system (benchmark) miners."""
    asyncio.run(_cmd_list_system_miners())


@db.command("set-miner")
@click.option("--uid", type=int, required=True)
@click.option("--model", required=True)
def set_miner(uid, model):
    """Register or update a system miner."""
    asyncio.run(_cmd_set_miner(uid, model))


@db.command("delete-miner")
@click.option("--uid", type=int, required=True)
def delete_miner(uid):
    """Delete a system miner registration."""
    asyncio.run(_cmd_delete_miner(uid))


# --------------------------------------------------------------------------- #
# inference_endpoints
# --------------------------------------------------------------------------- #


async def _cmd_list_endpoints() -> None:
    await init_client()
    try:
        from affine.database.dao.inference_endpoints import InferenceEndpointsDAO
        for ep in await InferenceEndpointsDAO().list_all():
            flag = "" if ep.active else "  (inactive)"
            print(f"{ep.name:<24} kind={ep.kind}{flag}")
            if ep.ssh_url:
                print(f"  ssh_url             : {ep.ssh_url}")
            if ep.public_inference_url:
                print(f"  public_inference_url: {ep.public_inference_url}")
            if ep.kind == "ssh":
                print(f"  sglang              : port={ep.sglang_port} dp={ep.sglang_dp} image={ep.sglang_image}")
                print(
                    "  sglang args         : "
                    f"context={ep.sglang_context_len} "
                    f"mem_fraction={ep.sglang_mem_fraction} "
                    f"chunked_prefill={ep.sglang_chunked_prefill} "
                    f"tool_parser={ep.sglang_tool_call_parser}"
                )
                print(
                    "  readiness           : "
                    f"timeout={ep.ready_timeout_sec}s "
                    f"poll={ep.poll_interval_sec}s"
                )
                print(f"  cache_dir           : {ep.sglang_cache_dir}")
            if ep.assigned_uid is not None:
                print(
                    "  assignment          : "
                    f"role={ep.assignment_role or '-'} "
                    f"uid={ep.assigned_uid} "
                    f"model={ep.assigned_model}@{(ep.assigned_revision or '')[:8]}"
                )
                if ep.deployment_id:
                    print(f"  deployment_id       : {ep.deployment_id}")
                if ep.base_url:
                    print(f"  base_url            : {ep.base_url}")
            if ep.notes:
                print(f"  notes               : {ep.notes}")
    finally:
        await close_client()


async def _cmd_set_endpoint(
    name: str, kind: str, ssh_url: Optional[str], ssh_key_path: Optional[str],
    public_inference_url: Optional[str], sglang_port: int, sglang_dp: int,
    sglang_image: str, sglang_cache_dir: str, sglang_context_len: int,
    sglang_mem_fraction: float, sglang_chunked_prefill: int,
    sglang_tool_call_parser: str, ready_timeout_sec: int,
    poll_interval_sec: float, active: bool, notes: Optional[str],
) -> None:
    await init_client()
    try:
        from affine.database.dao.inference_endpoints import (
            Endpoint, InferenceEndpointsDAO,
        )
        ep = Endpoint(
            name=name, kind=kind, active=active,
            ssh_url=ssh_url, ssh_key_path=ssh_key_path,
            public_inference_url=public_inference_url,
            sglang_port=sglang_port, sglang_dp=sglang_dp,
            sglang_image=sglang_image, sglang_cache_dir=sglang_cache_dir,
            sglang_context_len=sglang_context_len,
            sglang_mem_fraction=sglang_mem_fraction,
            sglang_chunked_prefill=sglang_chunked_prefill,
            sglang_tool_call_parser=sglang_tool_call_parser,
            ready_timeout_sec=ready_timeout_sec,
            poll_interval_sec=poll_interval_sec,
            notes=notes,
        )
        await InferenceEndpointsDAO().upsert(ep, updated_by="cli:set-endpoint")
        print(f"✓ wrote endpoint {name!r}")
    finally:
        await close_client()


async def _cmd_delete_endpoint(name: str) -> None:
    await init_client()
    try:
        from affine.database.dao.inference_endpoints import InferenceEndpointsDAO
        await InferenceEndpointsDAO().delete(name)
        print(f"✓ deleted endpoint {name!r}")
    finally:
        await close_client()


@db.command("list-endpoints")
def list_endpoints():
    """List registered inference endpoints."""
    asyncio.run(_cmd_list_endpoints())


@db.command("set-endpoint")
@click.option("--name", required=True, help="Unique label, e.g. 'ssh_b300'")
@click.option("--kind", required=True, type=click.Choice(["ssh", "targon"]))
@click.option("--ssh-url", default=None, help="ssh://user@host[:port] — required for kind=ssh")
@click.option("--ssh-key-path", default=None)
@click.option("--public-inference-url", default=None,
              help="What env containers actually connect to (e.g. http://val:30000/v1)")
@click.option("--sglang-port", type=int, default=30000)
@click.option("--sglang-dp", type=int, default=8)
@click.option("--sglang-image", default="lmsysorg/sglang:latest")
@click.option("--sglang-cache-dir", default="/data")
@click.option("--sglang-context-len", type=int, default=65536)
@click.option("--sglang-mem-fraction", type=float, default=0.85)
@click.option("--sglang-chunked-prefill", type=int, default=4096)
@click.option("--sglang-tool-call-parser", default="qwen",
              help='Tool-call parser name; set to "none" to omit')
@click.option("--ready-timeout-sec", type=int, default=1800)
@click.option("--poll-interval-sec", type=float, default=15.0)
@click.option("--active/--inactive", default=True)
@click.option("--notes", default=None)
def set_endpoint(
    name, kind, ssh_url, ssh_key_path, public_inference_url,
    sglang_port, sglang_dp, sglang_image, sglang_cache_dir,
    sglang_context_len, sglang_mem_fraction, sglang_chunked_prefill,
    sglang_tool_call_parser, ready_timeout_sec, poll_interval_sec,
    active, notes,
):
    """Register or update an inference endpoint."""
    if kind == "ssh" and not ssh_url:
        raise click.UsageError("--ssh-url is required when --kind=ssh")
    asyncio.run(_cmd_set_endpoint(
        name=name, kind=kind, ssh_url=ssh_url, ssh_key_path=ssh_key_path,
        public_inference_url=public_inference_url,
        sglang_port=sglang_port, sglang_dp=sglang_dp,
        sglang_image=sglang_image, sglang_cache_dir=sglang_cache_dir,
        sglang_context_len=sglang_context_len,
        sglang_mem_fraction=sglang_mem_fraction,
        sglang_chunked_prefill=sglang_chunked_prefill,
        sglang_tool_call_parser=sglang_tool_call_parser,
        ready_timeout_sec=ready_timeout_sec,
        poll_interval_sec=poll_interval_sec,
        active=active, notes=notes,
    ))


@db.command("delete-endpoint")
@click.option("--name", required=True)
def delete_endpoint(name):
    """Delete an inference endpoint registration."""
    asyncio.run(_cmd_delete_endpoint(name))


# --------------------------------------------------------------------------- #
# Sample progress
# --------------------------------------------------------------------------- #


def _parse_duration(spec: str) -> int:
    """Parse '5m', '2h', '30s', 'all' → seconds (0 = 'all')."""
    spec = spec.strip().lower()
    if spec in ("all", "0", ""):
        return 0
    unit = spec[-1]
    if unit not in "smh":
        return int(spec)
    n = int(spec[:-1])
    return n * (1 if unit == "s" else 60 if unit == "m" else 3600)


async def _cmd_sample_progress(window_spec: str, miner_filter: str) -> None:
    """Per-env progress + rate + ETA + live worker concurrency."""
    import time
    from datetime import datetime
    from affine.database.client import get_client
    from affine.database.schema import get_table_name

    await init_client()
    try:
        win_secs = _parse_duration(window_spec)
        sc = SystemConfigDAO()
        dao = SampleResultsDAO()
        client = get_client()
        table = get_table_name("sample_results")

        champion = await sc.get_param_value("champion") or {}
        battle = await sc.get_param_value("current_battle")
        tids = await sc.get_param_value("current_task_ids") or {}
        envs = await sc.get_param_value("environments") or {}
        rb = int(tids.get("refreshed_at_block", 0))

        subjects = []
        if miner_filter in ("active", "champion", "both") and champion.get("hotkey"):
            subjects.append(("champion", champion["uid"], champion["hotkey"], champion["revision"]))
        if miner_filter in ("active", "challenger", "both") and battle:
            chal = battle.get("challenger") or {}
            if chal.get("hotkey"):
                subjects.append(("challenger", chal["uid"], chal["hotkey"], chal["revision"]))

        if not subjects:
            click.echo("No subjects to report (no champion / battle, or filter excluded all).")
            return

        # Pull each env's live worker status (in_flight / cumulative counters).
        now = time.time()
        worker_status: dict = {}
        for env_name in envs:
            if envs[env_name].get("enabled_for_sampling", False):
                worker_status[env_name] = await sc.get_param_value(f"worker_status_{env_name}")

        win_label = window_spec if win_secs > 0 else "all-time"
        click.echo(f"refresh_block={rb}  window={win_label}")

        active_seen = False
        for role, uid, hk, rev in subjects:
            now_ms = int(now * 1000)
            cutoff_ms = now_ms - win_secs * 1000 if win_secs > 0 else 0
            env_lines = []
            subject_active = False

            for env_name in sorted(envs.keys()):
                cfg = envs[env_name]
                if not cfg.get("enabled_for_sampling", False):
                    env_lines.append(f"{env_name:<14}  (disabled)")
                    continue

                # --- progress columns ---
                target = (cfg.get("sampling") or {}).get("sampling_count", 0)

                pk = dao._make_pk(hk, rev, env_name)
                rows = []
                last_key = None
                while True:
                    params = {
                        "TableName": table,
                        "KeyConditionExpression": "pk = :pk AND begins_with(sk, :sk)",
                        "ExpressionAttributeValues": {":pk": {"S": pk}, ":sk": {"S": "TASK#"}},
                        "ProjectionExpression": "refresh_block, #t",
                        "ExpressionAttributeNames": {"#t": "timestamp"},
                    }
                    if last_key:
                        params["ExclusiveStartKey"] = last_key
                    resp = await client.query(**params)
                    rows.extend(resp.get("Items", []))
                    last_key = resp.get("LastEvaluatedKey")
                    if not last_key:
                        break

                current = [
                    int(r["timestamp"]["N"]) for r in rows
                    if "refresh_block" in r and int(r["refresh_block"]["N"]) == rb
                    and "timestamp" in r
                ]
                done = len(current)
                if target and done < target:
                    subject_active = True
                pct = (100 * done // target) if target else 0
                last_str = (
                    datetime.fromtimestamp(max(current) / 1000).strftime("%H:%M:%S")
                    if current else "—"
                )

                if win_secs > 0:
                    in_window = [t for t in current if t >= cutoff_ms]
                    rate_per_min = len(in_window) / (win_secs / 60.0)
                else:
                    rate_per_min = 0.0

                remaining = max(0, target - done)
                if remaining <= 0:
                    eta = "done"
                elif rate_per_min <= 0:
                    eta = "∞"
                else:
                    eta_min = remaining / rate_per_min
                    eta = f"{eta_min:.0f}m" if eta_min < 60 else f"{eta_min/60:.1f}h"

                # --- worker_status columns (live concurrency) — only meaningful
                # for the champion row; challenger sample writes share the env
                # workers so the columns are repeated but the numbers are env-wide.
                ws = worker_status.get(env_name) or {}
                in_flight = ws.get("tasks_in_flight", "—")
                succ = ws.get("tasks_succeeded", "—")
                fail = ws.get("tasks_failed", "—")
                total_ms = ws.get("total_execution_ms", 0)
                count = (ws.get("tasks_succeeded", 0) + ws.get("tasks_failed", 0)) if ws else 0
                avg_lat = f"{(total_ms / count) / 1000:.1f}s" if count else "—"
                reported_at = ws.get("reported_at", 0) if ws else 0
                age = int(now - reported_at) if reported_at else None
                age_str = f"{age}s" if age is not None else "—"

                env_lines.append(
                    f"{env_name:<14}{target:>5}{done:>6}{pct:>4}%  "
                    f"{rate_per_min:>6.1f}/m{eta:>8}  "
                    f"{in_flight:>7}{succ:>7}{fail:>6}{avg_lat:>9}  "
                    f"{last_str:>10}{age_str:>5}"
                )
            if not subject_active:
                continue

            active_seen = True
            click.echo("")
            click.echo(f"=== {role}  uid={uid}  ({hk[:14]}...@{rev[:8]}) ===")
            click.echo(
                f"{'env':<14}{'tgt':>5}{'done':>6}{'%':>5}  {'rate/m':>8}{'eta':>8}"
                f"  {'in_fly':>7}{'ok':>7}{'fail':>6}{'avg_lat':>9}  {'last':>10}{'age':>5}"
            )
            for line in env_lines:
                click.echo(line)

        if not active_seen:
            click.echo("")
            click.echo("No active sampling subjects to report.")
    finally:
        await close_client()


@db.command("sample-progress")
@click.option("--window", default="5m",
              help="Rate window: '30s', '5m', '2h', or 'all' (default: 5m)")
@click.option("--miner", default="active",
              type=click.Choice(["active", "champion", "challenger", "both"]),
              help="Which miner to report (default: active)")
def sample_progress(window, miner):
    """Per-env sample progress + rate + ETA + live executor concurrency.

    Combined view of:
      - ``done / target`` and ``%`` against ``sampling_count``
      - ``rate/m`` and ``eta`` over the requested rolling window
      - ``in_fly`` live in-flight evaluate() calls (from the
        ``worker_status_*`` rows each worker publishes every ~10s)
      - cumulative ``ok / fail`` and average per-task latency
      - timestamp of last write + freshness ``age`` of the worker status
    """
    asyncio.run(_cmd_sample_progress(window, miner))


def main():
    db()


if __name__ == "__main__":
    main()
