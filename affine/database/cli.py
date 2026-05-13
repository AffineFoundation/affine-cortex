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

Removed: every command tied to the dead per-miner slot scheduler, the
sampling-list rotation, MinerStatsDAO, TaskPoolDAO, AntiCopyDAO, or the
targon_deployer reconciler. Those concepts no longer exist.
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
from affine.database.dao import MinersDAO, SampleResultsDAO, SystemConfigDAO
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
            param_type = _param_type_of(value)
            await dao.set_param(
                param_name=key,
                param_value=value,
                param_type=param_type,
                description=f"Loaded from {os.path.basename(json_file)}",
                updated_by="cli:load-config",
            )
            preview = _preview(value)
            print(f"  {key:<24} → {preview}")
        print(f"✓ Loaded {len(payload)} keys")
    finally:
        await close_client()


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
        for k in ("uid", "hotkey", "model", "revision", "is_valid",
                  "challenge_status", "first_block", "block_number",
                  "invalid_reason", "model_hash"):
            if k in m and m[k] is not None:
                print(f"  {k:<18} = {m[k]}")
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
@click.option("--sglang-mem-fraction", type=float, default=0.8)
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


def main():
    db()


if __name__ == "__main__":
    main()
