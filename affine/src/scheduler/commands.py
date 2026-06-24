"""Operator commands for scheduler runtime state."""

from __future__ import annotations

import click

from affine.database import close_client, init_client
from affine.database.dao.system_config import SystemConfigDAO
from affine.src.scheduler.flow import FlowConfig
from affine.src.scorer.challenger_queue import ChallengerQueue
from affine.src.scorer.dao_adapters import MinersQueueAdapter
from affine.src.scorer.window_state import (
    StateStore,
    SystemConfigKVAdapter,
    TaskIdState,
)
from affine.utils.subtensor import get_subtensor


async def rotate_window_command(*, commit: bool = False) -> None:
    """Manually rotate the champion task window.

    Always performs a full rotation: it releases the active challenger
    claim (``in_progress`` -> ``sampling``), stales
    ``current_task_ids.refreshed_at_block`` to an old block, and clears
    ``current_battle`` so the next scheduler tick rotates immediately.

    Default mode is dry-run; pass ``commit=True`` to write. Because
    in-flight sample writes would otherwise land on the stale pool, the
    command asks the operator to confirm the sampling service is stopped
    before it writes anything.
    """
    await init_client()
    try:
        state = StateStore(
            SystemConfigKVAdapter(
                SystemConfigDAO(),
                updated_by="cli:rotate-window",
            )
        )

        champion = await state.get_champion()
        battle = await state.get_battle()
        predeployed = await state.get_predeployed_challengers()
        task_state = await state.get_task_state()

        subtensor = await get_subtensor()
        current_block = int(await subtensor.get_current_block())
        refresh_blocks = int(FlowConfig().task_pool_refresh_blocks)
        stale_block = current_block - refresh_blocks - 1

        click.echo(f"current_block      = {current_block}")
        click.echo(
            "champion           = "
            + (
                f"uid={champion.uid} {champion.hotkey[:12]}"
                if champion is not None else "None"
            )
        )
        click.echo(
            "battle in flight   = "
            + (
                f"uid={battle.challenger.uid} "
                f"{battle.challenger.hotkey[:12]} "
                f"rev={battle.challenger.revision[:8]}"
                if battle is not None else "None"
            )
        )
        if predeployed:
            click.echo(
                f"predeployed        = {[p.challenger.uid for p in predeployed]} "
                "(left as-is; they re-sample on the new pool)"
            )

        if task_state is None:
            click.echo(
                "current_task_ids   = (absent) - scheduler already rotates "
                "unconditionally next tick."
            )
        else:
            elapsed = current_block - task_state.refreshed_at_block
            click.echo(
                f"refreshed_at_block = {task_state.refreshed_at_block} "
                f"(elapsed {elapsed} / {refresh_blocks} blocks)"
            )
            for env, ids in sorted(task_state.task_ids.items()):
                click.echo(f"    {env:<14} {len(ids)} task_ids")

        click.echo("\n=== ROTATE plan ===")
        if battle is not None:
            click.echo(
                f"  1. release uid={battle.challenger.uid} claim: "
                "in_progress -> sampling"
            )
        else:
            click.echo("  1. (no battle in flight - nothing to release)")
        if task_state is not None:
            click.echo(f"  2. stale refreshed_at_block -> {stale_block}")
        else:
            click.echo("  2. (no task_state - rotation already unconditional)")
        click.echo("  3. clear current_battle <- opens the rotation gate")

        if not commit:
            click.echo("\n(dry-run) re-run with --commit to perform the rotation.")
            return

        click.echo(
            "\n⚠  Stop the sampling service first. In-flight sample writes "
            "would otherwise land on the stale task pool."
        )
        if not click.confirm("Has the sampling service been stopped?", default=False):
            click.echo("Aborted.")
            return

        if battle is not None:
            queue = ChallengerQueue(MinersQueueAdapter())
            released = await queue.release_claim(
                battle.challenger.uid,
                hotkey=battle.challenger.hotkey,
                revision=battle.challenger.revision,
            )
            suffix = "" if released else " (was not in_progress - check status)"
            click.echo(f"  1. release_claim(uid={battle.challenger.uid}) -> {released}{suffix}")

        if task_state is not None:
            await state.set_task_state(
                TaskIdState(
                    task_ids=task_state.task_ids,
                    refreshed_at_block=stale_block,
                )
            )
            click.echo(f"  2. refreshed_at_block -> {stale_block}")

        await state.clear_battle()
        click.echo("  3. current_battle cleared")
        click.echo("\nrotation armed - scheduler rotates on its next tick.")
    finally:
        await close_client()
