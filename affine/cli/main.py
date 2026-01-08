#!/usr/bin/env python3
"""
Affine CLI - Unified Command Line Interface

Provides a single entry point for all Affine components:

Server Services (af servers):
- af servers api       : Start API server
- af servers executor  : Start executor service
- af servers monitor   : Start monitor service (miners monitoring)
- af servers scorer    : Start scorer service
- af servers scheduler : Start scheduler service
- af servers validator : Start validator service

Miner Commands:
- af miner-deploy: One-command deployment (upload → deploy → commit)
- af commit      : Commit model to blockchain (miner)
- af pull        : Pull model from Hugging Face (miner)
- af chutes_push : Deploy model to Chutes (miner)
- af get-sample  : Query sample by UID, env, and task ID
- af get-miner   : Query miner status by UID
- af get-weights : Query latest normalized weights
- af get-scores  : Query latest scores for top N miners
- af get-rank    : Query and display miner ranking table

Docker Commands:
- af deploy : Deploy docker containers (validator/backend)
- af down   : Stop docker containers (validator/backend)

Database Commands:
- af db : Database management commands
"""

import sys
import os
import subprocess
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
    from affine.api.server import app, config
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
def executor(ctx):
    """Start executor service."""
    from affine.src.executor.main import main as executor_main
    
    # Forward to original executor CLI with captured -v flags
    sys.argv = ["executor"] + ctx.args
    executor_main.main(standalone_mode=False)


@servers.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def monitor(ctx):
    """Start monitor service."""
    from affine.src.monitor.main import main as monitor_main
    
    sys.argv = ["monitor"] + ctx.args
    monitor_main.main(standalone_mode=False)


@servers.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def scorer(ctx):
    """Start scorer service."""
    from affine.src.scorer.main import main as scorer_main
    
    sys.argv = ["scorer"] + ctx.args
    scorer_main.main(standalone_mode=False)


@servers.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def scheduler(ctx):
    """Start scheduler service."""
    from affine.src.scheduler.main import main as scheduler_main
    
    sys.argv = ["scheduler"] + ctx.args
    scheduler_main.main(standalone_mode=False)


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


@cli.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def chutes_push(ctx):
    """Deploy model to Chutes."""
    from affine.src.miner.main import chutes_push as miner_chutes_push
    
    sys.argv = ["chutes_push"] + ctx.args
    miner_chutes_push.main(standalone_mode=False)


@cli.command("get-sample", context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def get_sample(ctx):
    """Query sample result by UID, environment, and task ID.
    
    Example:
        af get-sample 42 affine task_123
    """
    from affine.src.miner.main import get_sample as miner_get_sample
    
    sys.argv = ["get-sample"] + ctx.args
    miner_get_sample.main(standalone_mode=False)


@cli.command("get-miner", context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def get_miner(ctx):
    """Query miner status and information by UID.
    
    Returns complete miner info including hotkey, model, revision,
    chute_id, validation status, and timestamps.
    
    Example:
        af get-miner 42
    """
    from affine.src.miner.main import get_miner as miner_get_miner
    
    sys.argv = ["get-miner"] + ctx.args
    miner_get_miner.main(standalone_mode=False)


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


# @cli.command("get-pool", context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
# @click.pass_context
# def get_pool(ctx):
#     """Query pending task IDs for a miner in an environment.
#
#     Returns the list of task IDs currently in the sampling queue.
#
#     Example:
#         af get-pool 100 agentgym:webshop
#     """
#     from affine.src.miner.main import get_pool as miner_get_pool
#
#     sys.argv = ["get-pool"] + ctx.args
#     miner_get_pool.main(standalone_mode=False)


@cli.command("get-rank", context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def get_rank(ctx):
    """Query and display miner ranking table.
    
    Fetches the latest score snapshot from the API and displays
    it in the same format as the scorer's detailed table output.
    
    Example:
        af get-rank
    """
    from affine.src.miner.main import get_rank as miner_get_rank
    
    sys.argv = ["get-rank"] + ctx.args
    miner_get_rank.main(standalone_mode=False)


@cli.command("get-envs", context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def get_envs(ctx):
    """Query current environment configurations.
    
    Returns all environment configurations including sampling settings,
    rotation settings, and enabled flags.
    
    Example:
        af get-envs
    """
    from affine.src.miner.main import get_envs as miner_get_envs
    
    sys.argv = ["get-envs"] + ctx.args
    miner_get_envs.main(standalone_mode=False)


@cli.command("miner-deploy", context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def miner_deploy(ctx):
    """One-command deployment: Upload -> Deploy -> Commit.
    
    Combines the three-step miner deployment process into a single command:
    1. Upload model to HuggingFace (skip with --skip-upload)
    2. Deploy to Chutes (skip with --skip-chutes)
    3. Commit on-chain (skip with --skip-commit)
    
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
# Miner Stats Management Commands
# ============================================================================

# Import and register miner_stats commands
from affine.cli.miner_stats import miner_stats
miner_stats.hidden = not SHOW_ADMIN_COMMANDS
cli.add_command(miner_stats)


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
        result = subprocess.run(
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
        result = subprocess.run(
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