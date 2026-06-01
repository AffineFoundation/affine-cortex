"""Operator commands for scheduler runtime state."""

from __future__ import annotations

import click

from affine.database import close_client, init_client
from affine.database.dao.system_config import SystemConfigDAO
from affine.src.scheduler.flow import WINDOW_BLOCKS
from affine.src.scorer.challenger_queue import ChallengerQueue
from affine.src.scorer.dao_adapters import MinersQueueAdapter
from affine.src.scorer.window_state import (
    StateStore,
    SystemConfigKVAdapter,
    TaskIdState,
)
from affine.utils.subtensor import get_subtensor


async def rotate_champion_command(
    *,
    commit: bool = False,
    force: bool = False,
    full_rotate: bool = False,
) -> None:
    """Manually stale the active task pool so the scheduler rotates it.

    Default mode is stale-only: it rewrites
    ``current_task_ids.refreshed_at_block`` to an old block. The scheduler
    then performs its normal between-battles refresh on the next tick. With
    ``--full-rotate``, the command also releases the active challenger claim
    and clears ``current_battle`` so the next scheduler tick rotates
    immediately.
    """
    await init_client()
    try:
        state = StateStore(
            SystemConfigKVAdapter(
                SystemConfigDAO(),
                updated_by="cli:rotate-champion",
            )
        )

        champion = await state.get_champion()
        battle = await state.get_battle()
        predeployed = await state.get_predeployed_challengers()
        task_state = await state.get_task_state()

        subtensor = await get_subtensor()
        current_block = int(await subtensor.get_current_block())
        stale_block = current_block - WINDOW_BLOCKS - 1

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
                f"(elapsed {elapsed} / {WINDOW_BLOCKS} blocks)"
            )
            for env, ids in sorted(task_state.task_ids.items()):
                click.echo(f"    {env:<14} {len(ids)} task_ids")

        if not full_rotate:
            if task_state is None:
                click.echo("\nNothing to stale.")
                return
            if current_block - task_state.refreshed_at_block >= WINDOW_BLOCKS:
                click.echo(
                    "\nAlready past the window threshold - a no-battle tick "
                    "rotates on its own."
                )
                return
            click.echo(
                f"\nwould set refreshed_at_block: "
                f"{task_state.refreshed_at_block} -> {stale_block}"
            )
            if battle is not None and not force:
                click.echo(
                    "\nREFUSING: battle in flight. Use --full-rotate to "
                    "rotate anyway, or --force to just stale."
                )
                return
            if not commit:
                click.echo("\n(dry-run) re-run with --commit to write.")
                return
            await state.set_task_state(
                TaskIdState(
                    task_ids=task_state.task_ids,
                    refreshed_at_block=stale_block,
                )
            )
            click.echo("\ncommitted (stale-only).")
            return

        click.echo("\n=== FULL ROTATE plan ===")
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
        click.echo("\nfull rotation armed - scheduler rotates on its next tick.")
    finally:
        await close_client()
