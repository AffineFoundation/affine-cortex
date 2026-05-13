"""
Miner-side CLI entry points.

Surfaced via ``affine/cli/main.py``:
  - pull
  - commit
  - deploy        (alias: miner-deploy)
  - get-weights
  - get-scores / get-score
  - get-rank      (one-stop: window + queue + weight table)
"""

from __future__ import annotations

import asyncio

import click

from affine.cli.types import UID
from affine.src.miner.commands import (
    commit_command,
    deploy_command,
    get_score_command,
    get_scores_command,
    get_weights_command,
    pull_command,
)
from affine.src.miner.rank import get_rank_command


@click.command()
@click.argument("uid", type=UID)
@click.option("--model-path", "-p", default="./model_path", type=click.Path(),
              help="Local directory to save the model")
@click.option("--hf-token", help="Hugging Face API token")
def pull(uid, model_path, hf_token):
    """Pull a UID's committed model snapshot from HuggingFace."""
    asyncio.run(pull_command(uid=uid, model_path=model_path, hf_token=hf_token))


@click.command()
@click.option("--repo", required=True, help="HuggingFace repo id (user/model-name)")
@click.option("--revision", required=True, help="HuggingFace commit SHA")
@click.option("--coldkey", help="Wallet coldkey name (default: $BT_WALLET_COLD)")
@click.option("--hotkey", help="Wallet hotkey name (default: $BT_WALLET_HOT)")
def commit(repo, revision, coldkey, hotkey):
    """Commit {model, revision} to the chain. No chute_id — the validator
    hosts inference itself per window."""
    asyncio.run(commit_command(repo=repo, revision=revision, coldkey=coldkey, hotkey=hotkey))


@click.command("get-weights")
def get_weights():
    """Show the latest on-chain-bound weight snapshot."""
    asyncio.run(get_weights_command())


@click.command("get-scores")
@click.option("--top", "-t", default=10, type=int,
              help="Return top N miners by score")
def get_scores(top):
    """Show the top N miners from the latest scoring snapshot."""
    asyncio.run(get_scores_command(top=top))


@click.command("get-score")
@click.argument("uid", type=UID)
def get_score(uid):
    """Show one miner's score from the latest snapshot."""
    asyncio.run(get_score_command(uid=uid))


@click.command("get-rank")
def get_rank():
    """Show the current ranking table + live window state."""
    asyncio.run(get_rank_command())


@click.command("miner-deploy")
@click.option("--repo", "-r", required=True, help="HuggingFace repository id")
@click.option("--model-path", "-p", type=click.Path(exists=True),
              help="Local model directory (required unless --skip-upload)")
@click.option("--revision", help="HuggingFace commit SHA (required if --skip-upload)")
@click.option("--message", "-m", default="Model update", help="HF commit message")
@click.option("--dry-run", is_flag=True, help="Print actions, don't execute")
@click.option("--skip-upload", is_flag=True, help="Skip HF upload (requires --revision)")
@click.option("--skip-commit", is_flag=True, help="Skip on-chain commit")
@click.option("--hf-token", help="HuggingFace token (default: $HF_TOKEN)")
@click.option("--coldkey", help="Wallet coldkey name")
@click.option("--hotkey", help="Wallet hotkey name")
def deploy(repo, model_path, revision, message, dry_run, skip_upload, skip_commit,
           hf_token, coldkey, hotkey):
    """One-shot deployment: HF upload → on-chain commit."""
    asyncio.run(deploy_command(
        repo=repo,
        model_path=model_path,
        revision=revision,
        message=message,
        dry_run=dry_run,
        skip_upload=skip_upload,
        skip_commit=skip_commit,
        coldkey=coldkey,
        hotkey=hotkey,
        hf_token=hf_token,
    ))
