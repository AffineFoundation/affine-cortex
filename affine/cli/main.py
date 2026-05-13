#!/usr/bin/env python3
"""
Affine CLI - Unified Command Line Interface

Provides a single entry point for all Affine components:

Server Services (af servers):
- af servers api       : Start API server
- af servers monitor   : Start monitor service (miners monitoring)
- af servers scheduler : Start flow scheduler (block-tick contest driver)
- af servers executor  : Start per-env executor manager (subprocess per env)
- af servers teacher   : Start teacher rollout worker + R2 mover (DISTILL)
- af servers validator : Start validator service

Miner Commands:
- af miner-deploy: One-command deployment (HF push → on-chain commit)
- af commit      : Commit model+revision to blockchain (miner)
- af pull        : Pull model from Hugging Face (miner)
- af get-weights : Query latest normalized weights
- af get-scores  : Query latest scores for top N miners
- af get-score   : Query score for a specific miner
- af get-miner   : Query public miner metadata
- af get-rank    : Query the public rank/status table

Docker Commands:
- af deploy : Deploy docker containers (validator/backend)
- af down   : Stop docker containers (validator/backend)

Database Commands:
- af db : Database management commands
"""

import sys
import os
import subprocess
import asyncio
import click
from affine.core.setup import setup_logging, logger

# Check if admin commands should be visible
SHOW_ADMIN_COMMANDS = os.getenv("AFFINE_SHOW_ADMIN_COMMANDS", "").lower() in ("1", "true", "yes")


@click.group()
@click.option(
    "-v", "--verbosity",
    count=True,
    help="Increase logging verbosity (-v=INFO, -vv=DEBUG, -vvv=TRACE)"
)
def cli(verbosity):
    """
    Affine CLI - Unified interface for all Affine components.
    
    Use -v, -vv, or -vvv for different logging levels.
    """
    # Convert count to verbosity level
    # -v -> 1, -vv -> 2, -vvv -> 3
    verbosity_level = min(verbosity, 3)
    setup_logging(verbosity_level)


# ============================================================================
# Server Services (Group)
# ============================================================================

@cli.group(hidden=not SHOW_ADMIN_COMMANDS)
def servers():
    """Start various backend server services."""
    pass


@servers.command()
def api():
    """Start API server."""
    from affine.api.server import config
    import uvicorn

    setup_logging(verbosity=1, component="api")

    uvicorn.run(
        "affine.api.server:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.RELOAD,
        log_level=config.LOG_LEVEL.lower(),
        workers=config.WORKERS,
        timeout_keep_alive=60,
    )


@servers.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def monitor(ctx):
    """Start monitor service."""
    from affine.src.monitor.main import main as monitor_main

    sys.argv = ["monitor"] + ctx.args
    monitor_main.main(standalone_mode=False)


@servers.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def scheduler(ctx):
    """Start the flow scheduler (block-tick driver of the whole contest)."""
    from affine.src.scheduler.main import main as scheduler_main

    sys.argv = ["scheduler"] + ctx.args
    scheduler_main.main(standalone_mode=False)


@servers.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def executor(ctx):
    """Start per-env executor manager (one subprocess per enabled env)."""
    from affine.src.executor.main import main as executor_main

    sys.argv = ["executor"] + ctx.args
    executor_main.main(standalone_mode=False)


@servers.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def teacher(ctx):
    """Start teacher rollout worker + R2 mover (feeds the DISTILL env)."""
    from affine.src.teacher.main import main as teacher_main

    sys.argv = ["teacher"] + ctx.args
    teacher_main.main(standalone_mode=False)


@servers.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def validator(ctx):
    """Start validator service."""
    from affine.src.validator.main import main as validator_main

    sys.argv = ["validator"] + ctx.args
    validator_main.main(standalone_mode=False)


@cli.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def validate(ctx):
    """Start validator service."""
    from affine.src.validator.main import main as validator_main
    
    sys.argv = ["validate"] + ctx.args
    validator_main.main(standalone_mode=False)


# ============================================================================
# Evaluation Command
# ============================================================================

from affine.src.miner.eval import eval_cmd
cli.add_command(eval_cmd, name="eval")


# ============================================================================
# Miner Commands
# ============================================================================

@cli.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def commit(ctx):
    """Commit model to blockchain."""
    from affine.src.miner.main import commit as miner_commit
    
    sys.argv = ["commit"] + ctx.args
    miner_commit.main(standalone_mode=False)


@cli.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def pull(ctx):
    """Pull model from Hugging Face."""
    from affine.src.miner.main import pull as miner_pull
    
    sys.argv = ["pull"] + ctx.args
    miner_pull.main(standalone_mode=False)


@cli.command("get-weights", context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def get_weights(ctx):
    """Query latest normalized weights for on-chain weight setting.
    
    Returns the most recent score snapshot with normalized weights
    for all miners, suitable for setting on-chain weights.
    
    Example:
        af get-weights
    """
    from affine.src.miner.main import get_weights as miner_get_weights
    
    sys.argv = ["get-weights"] + ctx.args
    miner_get_weights.main(standalone_mode=False)


@cli.command("get-scores", context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def get_scores(ctx):
    """Query latest scores for top N miners.
    
    Returns top N miners by score at the latest calculated block.
    
    Example:
        af get-scores
        af get-scores --top 10
    """
    from affine.src.miner.main import get_scores as miner_get_scores
    
    sys.argv = ["get-scores"] + ctx.args
    miner_get_scores.main(standalone_mode=False)


@cli.command("get-score", context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def get_score(ctx):
    """Query score for a specific miner by UID.
    
    Returns the score details for the specified miner from the latest snapshot.
    
    Example:
        af get-score 42
    """
    from affine.src.miner.main import get_score as miner_get_score
    
    sys.argv = ["get-score"] + ctx.args
    miner_get_score.main(standalone_mode=False)


@cli.command("get-miner")
@click.option("--uid", type=int, help="Miner UID")
@click.option("--hotkey", help="Miner hotkey")
def get_miner(uid, hotkey):
    """Query public miner metadata by UID or hotkey.

    Example:
        af get-miner --uid 42
        af get-miner --hotkey 5F...
    """
    from affine.src.miner.commands import get_miner_command

    asyncio.run(get_miner_command(uid=uid, hotkey=hotkey))


@cli.command("get-rank", context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def get_rank(ctx):
    """Query and display miner ranking table.

    Shows the live window state (champion/challenger/phase/progress),
    the public rank/status snapshot.

    Example:
        af get-rank
    """
    from affine.src.miner.main import get_rank as miner_get_rank

    sys.argv = ["get-rank"] + ctx.args
    miner_get_rank.main(standalone_mode=False)


@cli.command("miner-deploy", context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def miner_deploy(ctx):
    """One-command miner deployment: HF upload → on-chain commit.

    The scheduler service hosts inference per window, so miners only upload
    weights and commit the model snapshot.

    Examples:
        af miner-deploy -r myuser/model -p ./my_model
        af miner-deploy -r myuser/model --skip-upload --revision abc123
        af miner-deploy -r myuser/model -p ./my_model --dry-run
    """
    from affine.src.miner.main import deploy as miner_deploy_cmd

    sys.argv = ["miner-deploy"] + ctx.args
    miner_deploy_cmd.main(standalone_mode=False)


# ============================================================================
# Database Management Commands
# ============================================================================

# Import and register the db group from database.cli
from affine.database.cli import db
db.hidden = not SHOW_ADMIN_COMMANDS
cli.add_command(db)


# ============================================================================
# Docker Deployment Commands
# ============================================================================

@cli.command(hidden=not SHOW_ADMIN_COMMANDS)
@click.argument("service", type=click.Choice(["validator", "backend", "api"]))
@click.option("--local", is_flag=True, help="Use local build mode")
@click.option("--recreate", is_flag=True, help="Recreate containers")
@click.option("--restart", is_flag=True, help="Restart containers without recreating")
def deploy(service, local, recreate, restart):
    """Deploy docker containers for validator, backend, or api services.
    
    SERVICE: Either 'validator', 'backend', or 'api'
    
    Examples:
        af deploy validator --recreate --local
        af deploy backend --local
        af deploy api --local
        af deploy backend --restart --local
        af deploy validator
    """
    # Validate conflicting options
    if recreate and restart:
        logger.error("Cannot use both --recreate and --restart options")
        sys.exit(1)
    # Get the affine directory (where docker-compose files are located)
    affine_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Build docker-compose command based on service type
    if service == "validator":
        compose_files = ["-f", "docker-compose.yml"]
        if local:
            compose_files.extend(["-f", "docker-compose.local.yml"])
    elif service == "api":
        compose_files = ["-f", "compose/docker-compose.api.yml"]
        if local:
            compose_files.extend(["-f", "compose/docker-compose.api.local.yml"])
    else:  # backend
        compose_files = ["-f", "compose/docker-compose.backend.yml"]
        if local:
            compose_files.extend(["-f", "compose/docker-compose.backend.local.yml"])
    
    # Build the command with project directory
    if restart:
        # Use restart command instead of up
        cmd = ["docker", "compose", "--project-directory", affine_dir] + compose_files + ["restart"]
    else:
        cmd = ["docker", "compose", "--project-directory", affine_dir] + compose_files + ["up", "-d"]
        
        if recreate:
            cmd.append("--force-recreate")
        
        if local:
            cmd.append("--build")
    
    # Execute the command
    action = "Restarting" if restart else "Deploying"
    logger.info(f"{action} {service} services...")
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        subprocess.run(
            cmd,
            cwd=affine_dir,
            check=True,
            capture_output=False
        )
        success_msg = "restarted" if restart else "deployed"
        logger.info(f"✓ {service.capitalize()} services {success_msg} successfully")
    except subprocess.CalledProcessError as e:
        action_msg = "restart" if restart else "deploy"
        logger.error(f"✗ Failed to {action_msg} {service} services")
        sys.exit(e.returncode)


@cli.command(hidden=not SHOW_ADMIN_COMMANDS)
@click.argument("service", type=click.Choice(["validator", "backend", "api"]))
@click.option("--local", is_flag=True, help="Use local build mode")
@click.option("--volumes", "-v", is_flag=True, help="Remove volumes as well")
def down(service, local, volumes):
    """Stop and remove docker containers for validator, backend, or api services.
    
    SERVICE: Either 'validator', 'backend', or 'api'
    
    Examples:
        af down validator --local
        af down backend --local --volumes
        af down api
        af down backend -v
    """
    # Get the affine directory (where docker-compose files are located)
    affine_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Build docker-compose command based on service type
    if service == "validator":
        compose_files = ["-f", "docker-compose.yml"]
        if local:
            compose_files.extend(["-f", "docker-compose.local.yml"])
    elif service == "api":
        compose_files = ["-f", "compose/docker-compose.api.yml"]
        if local:
            compose_files.extend(["-f", "compose/docker-compose.api.local.yml"])
    else:  # backend
        compose_files = ["-f", "compose/docker-compose.backend.yml"]
        if local:
            compose_files.extend(["-f", "compose/docker-compose.backend.local.yml"])
    
    # Build the command with project directory
    cmd = ["docker", "compose", "--project-directory", affine_dir] + compose_files + ["down"]
    
    if volumes:
        cmd.append("--volumes")
    
    # Execute the command
    logger.info(f"Stopping {service} services...")
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        subprocess.run(
            cmd,
            cwd=affine_dir,
            check=True,
            capture_output=False
        )
        logger.info(f"✓ {service.capitalize()} services stopped successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Failed to stop {service} services")
        sys.exit(e.returncode)


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
