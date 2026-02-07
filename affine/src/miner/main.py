"""
Miner CLI Commands

Provides command-line interface for miner operations:
- pull: Pull model from Hugging Face
- chutes_push: Deploy model to Chutes
- commit: Commit model to blockchain
- get-sample: Query sample by UID, environment, and task ID
- get-miner: Query miner status by UID
"""

import click
import asyncio

from affine.cli.types import UID

from affine.src.miner.commands import (
    pull_command,
    chutes_push_command,
    commit_command,
    deploy_command,
    get_sample_command,
    get_miner_command,
    get_weights_command,
    get_scores_command,
    get_score_command,
    get_pool_command,
    get_envs_command,
)
from affine.src.miner.rank import get_rank_command


@click.command()
@click.argument("uid", type=UID)
@click.option("--model-path", "-p", default="./model_path", type=click.Path(), help="Local directory to save the model")
@click.option("--hf-token", help="Hugging Face API token")
def pull(uid, model_path, hf_token):
    """Pull model from Hugging Face.

    UID can use 'n' prefix for negative values (e.g., n1 means -1).
    """
    asyncio.run(pull_command(
        uid=uid,
        model_path=model_path,
        hf_token=hf_token
    ))


@click.command()
@click.option("--repo", required=True, help="HF repo id")
@click.option("--revision", required=True, help="HF commit SHA")
@click.option("--chutes-api-key", help="Chutes API key")
@click.option("--chute-user", help="Chutes username")
def chutes_push(repo, revision, chutes_api_key, chute_user):
    """Deploy model to Chutes."""
    asyncio.run(chutes_push_command(
        repo=repo,
        revision=revision,
        chutes_api_key=chutes_api_key,
        chute_user=chute_user
    ))


@click.command()
@click.option("--repo", required=True, help="HF repo id")
@click.option("--revision", required=True, help="HF commit SHA")
@click.option("--chute-id", required=True, help="Chutes deployment id")
@click.option("--coldkey", help="Coldkey name")
@click.option("--hotkey", help="Hotkey name")
def commit(repo, revision, chute_id, coldkey, hotkey):
    """Commit model to blockchain."""
    asyncio.run(commit_command(
        repo=repo,
        revision=revision,
        chute_id=chute_id,
        coldkey=coldkey,
        hotkey=hotkey
    ))


@click.command("get-sample")
@click.argument("uid", type=UID)
@click.argument("env", type=str)
@click.argument("task_id", type=str)
def get_sample(uid, env, task_id):
    """Query sample result by UID, environment, and task ID.

    Examples:
        af get-sample 42 affine task_123
        af get-sample n1 affine task_123  (n1 means UID -1)
    """
    asyncio.run(get_sample_command(
        uid=uid,
        env=env,
        task_id=task_id,
    ))

@click.command("get-miner")
@click.argument("uid", type=UID)
def get_miner(uid):
    """Query miner status and information by UID.

    Returns complete miner info including hotkey, model, revision,
    chute_id, validation status, and timestamps.

    Examples:
        af get-miner 42
        af get-miner n1  (n1 means UID -1)
    """
    asyncio.run(get_miner_command(
        uid=uid,
    ))


@click.command("get-weights")
def get_weights():
    """Query latest normalized weights for on-chain weight setting.
    
    Returns the most recent score snapshot with normalized weights
    for all miners, suitable for setting on-chain weights.
    
    Example:
        af get-weights
    """
    asyncio.run(get_weights_command())

@click.command("get-scores")
@click.option("--top", "-t", default=10, type=int, help="Return top N miners by score (default: 256)")
def get_scores(top):
    """Query latest scores for top N miners.
    
    Returns top N miners by score at the latest calculated block.
    
    Example:
        af get-scores
        af get-scores --top 10
    """
    asyncio.run(get_scores_command(top=top))

@click.command("get-score")
@click.argument("uid", type=UID)
def get_score(uid):
    """Query score for a specific miner by UID.

    Returns the score details for the specified miner from the latest snapshot.

    Examples:
        af get-score 42
        af get-score n1  (n1 means UID -1)
    """
    asyncio.run(get_score_command(uid=uid))

@click.command("get-pool")
@click.argument("uid", type=UID)
@click.argument("env", type=str)
@click.option("--full", is_flag=True, help="Print full task_ids lists without truncation")
def get_pool(uid, env, full):
    """Query task pool status for a miner in an environment.

    Returns the list of task IDs currently in the sampling queue
    for the specified miner and environment.

    Examples:
        af get-pool 100 agentgym:webshop
        af get-pool n1 agentgym:webshop  (n1 means UID -1)
        af get-pool 100 agentgym:webshop --full
    """
    asyncio.run(get_pool_command(uid=uid, env=env, full=full))


@click.command("get-rank")
def get_rank():
    """Query and display miner ranking table.
    
    Fetches the latest score snapshot from the API and displays
    it in the same format as the scorer's detailed table output.
    
    Example:
        af get-rank
    """
    asyncio.run(get_rank_command())


@click.command("get-envs")
def get_envs():
    """Query current environment configurations.
    
    Returns all environment configurations including sampling settings,
    rotation settings, and enabled flags.
    
    Example:
        af get-envs
    """
    asyncio.run(get_envs_command())


@click.command("deploy")
@click.option("--repo", "-r", required=True, help="HuggingFace repository ID (e.g., username/model-name)")
@click.option("--model-path", "-p", type=click.Path(exists=True), help="Path to local model directory (required unless --skip-upload)")
@click.option("--revision", help="HuggingFace revision SHA (required if --skip-upload)")
@click.option("--chute-id", help="Chutes deployment ID (required if --skip-chutes)")
@click.option("--message", "-m", default="Model update", help="Commit message for HuggingFace upload")
@click.option("--dry-run", is_flag=True, help="Show what would be done without executing")
@click.option("--skip-upload", is_flag=True, help="Skip HuggingFace upload (requires --revision)")
@click.option("--skip-chutes", is_flag=True, help="Skip Chutes deployment (requires --chute-id)")
@click.option("--skip-commit", is_flag=True, help="Skip on-chain commit")
@click.option("--chutes-api-key", help="Chutes API key (optional, from env if not provided)")
@click.option("--chute-user", help="Chutes username (optional, from env if not provided)")
@click.option("--coldkey", help="Wallet coldkey name (optional, from env if not provided)")
@click.option("--hotkey", help="Wallet hotkey name (optional, from env if not provided)")
@click.option("--hf-token", help="HuggingFace token (optional, from env if not provided)")
@click.option("--private-repo", is_flag=True, help="Use private HF repo workflow (commit before deploy)")
def deploy(repo, model_path, revision, chute_id, message, dry_run, skip_upload, skip_chutes, skip_commit, chutes_api_key, chute_user, coldkey, hotkey, hf_token, private_repo):
    """One-command deployment: Upload -> Deploy -> Commit.
    
    Normal workflow:
    1. Upload model to HuggingFace (skip with --skip-upload)
    2. Deploy to Chutes (skip with --skip-chutes)
    3. Commit on-chain (skip with --skip-commit)
    
    Private Repo Workflow (--private-repo):
        Prevents cheaters from copying your model before your commit is on-chain.
        Uses a different order to ensure model stays hidden until committed:
        
        1. Upload to PRIVATE HuggingFace repo
        2. Pre-generate chute_id (deterministic from username + repo)
        3. COMMIT FIRST to blockchain with pre-generated chute_id
        4. Make HF repo PUBLIC after commit confirmed
        5. Deploy to Chutes (chute_id will match pre-generated one)
        
        This way, cheaters cannot see your model until after your commit is on-chain.
    
    Examples:
        # Full deployment (public)
        af miner-deploy -r myuser/model -p ./my_model
        
        # Private repo workflow (recommended for competitive advantage)
        af miner-deploy -r myuser/model -p ./my_model --private-repo
        
        # Skip upload (model already on HuggingFace)
        af miner-deploy -r myuser/model --skip-upload --revision abc123
        
        # Skip Chutes (already deployed)
        af miner-deploy -r myuser/model --skip-upload --revision abc123 --skip-chutes --chute-id xyz
        
        # Dry run
        af miner-deploy -r myuser/model -p ./my_model --dry-run
    """
    asyncio.run(deploy_command(
        repo=repo,
        model_path=model_path,
        revision=revision,
        chute_id=chute_id,
        message=message,
        dry_run=dry_run,
        skip_upload=skip_upload,
        skip_chutes=skip_chutes,
        skip_commit=skip_commit,
        chutes_api_key=chutes_api_key,
        chute_user=chute_user,
        coldkey=coldkey,
        hotkey=hotkey,
        hf_token=hf_token,
        private_repo=private_repo,
    ))




