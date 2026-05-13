"""
Miner-side CLI command implementations.

After the queue-window refactor, the miner-side surface is purely:
  - pull              : fetch the on-chain model snapshot for a UID
  - commit            : write {model, revision} on chain
  - miner-deploy      : convenience wrapper (HF upload → commit)
  - get-weights       : pretty-print the latest weight snapshot
  - get-scores / get-score : score table / single-miner score
  - get-miner         : basic public miner metadata
  - get-rank          : one-stop status — rank/status table

The validator hosts inference per window via the scorer service; miners
upload to HuggingFace and commit their model snapshot.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Optional

from affine.core.setup import NETUID, logger
from affine.utils.api_client import cli_api_client
from affine.utils.subtensor import get_subtensor


def _conf(key: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(key, default)


# --------------------------------------------------------------------------- #
# pull
# --------------------------------------------------------------------------- #


async def pull_command(uid: int, model_path: str, hf_token: Optional[str] = None) -> None:
    """Fetch the model committed by ``uid`` from HuggingFace."""
    from huggingface_hub import snapshot_download

    hf_token = hf_token or _conf("HF_TOKEN")

    try:
        subtensor = await get_subtensor()
        meta = await subtensor.metagraph(NETUID)
        commits = await subtensor.get_all_revealed_commitments(NETUID)
    except Exception as e:
        logger.error(f"Failed to read chain: {e}")
        print(json.dumps({"success": False, "error": str(e)}))
        sys.exit(1)

    if uid >= len(meta.hotkeys):
        print(json.dumps({"success": False, "error": f"Invalid UID {uid}"}))
        sys.exit(1)
    hotkey = meta.hotkeys[uid]
    if hotkey not in commits:
        print(json.dumps({"success": False, "error": f"No commit for UID {uid}"}))
        sys.exit(1)

    _, commit_data = commits[hotkey][-1]
    try:
        data = json.loads(commit_data)
    except json.JSONDecodeError:
        print(json.dumps({"success": False, "error": "commit is not valid JSON"}))
        sys.exit(1)
    repo = data.get("model")
    revision = data.get("revision")
    if not repo or not revision:
        print(json.dumps({"success": False, "error": "commit missing model/revision"}))
        sys.exit(1)

    logger.info(f"Pulling {repo}@{revision[:8]}... → {model_path}")
    try:
        snapshot_download(
            repo_id=repo,
            repo_type="model",
            local_dir=model_path,
            token=hf_token,
            resume_download=True,
            revision=revision,
        )
    except Exception as e:
        logger.error(f"snapshot_download failed: {e}")
        print(json.dumps({"success": False, "error": str(e)}))
        sys.exit(1)

    print(json.dumps({
        "success": True, "uid": uid, "repo": repo, "revision": revision, "path": model_path,
    }))


# --------------------------------------------------------------------------- #
# commit
# --------------------------------------------------------------------------- #


async def commit_command(
    repo: str,
    revision: str,
    coldkey: Optional[str] = None,
    hotkey: Optional[str] = None,
) -> None:
    """Write ``{model, revision}`` to the chain commitment slot."""
    import bittensor as bt
    from bittensor.core.errors import MetadataError

    cold = coldkey or _conf("BT_WALLET_COLD", "default")
    hot = hotkey or _conf("BT_WALLET_HOT", "default")
    wallet = bt.Wallet(name=cold, hotkey=hot)
    payload = json.dumps({"model": repo, "revision": revision})

    logger.info(
        f"Committing {repo}@{revision[:8]}... as "
        f"{wallet.hotkey.ss58_address[:16]}..."
    )

    try:
        sub = await get_subtensor()
        while True:
            try:
                await sub.set_reveal_commitment(
                    wallet=wallet,
                    netuid=NETUID,
                    data=payload,
                    blocks_until_reveal=1,
                )
                break
            except MetadataError as e:
                if "SpaceLimitExceeded" in str(e):
                    logger.warning("commitment slot full, waiting next block...")
                    await sub.wait_for_block()
                else:
                    raise
    except Exception as e:
        logger.error(f"Commit failed: {e}")
        print(json.dumps({"success": False, "error": str(e)}))
        sys.exit(1)

    print(json.dumps({"success": True, "repo": repo, "revision": revision}))


# --------------------------------------------------------------------------- #
# miner-deploy
# --------------------------------------------------------------------------- #


async def deploy_command(
    repo: str,
    model_path: Optional[str],
    revision: Optional[str],
    message: str,
    dry_run: bool,
    skip_upload: bool,
    skip_commit: bool,
    coldkey: Optional[str],
    hotkey: Optional[str],
    hf_token: Optional[str],
) -> None:
    """One-shot: upload to HF (optional) → commit ``{model, revision}``.

    Inference is hosted by the scorer service per window, so the miner-side
    action is HuggingFace upload plus on-chain commit.
    """
    hf_token = hf_token or _conf("HF_TOKEN")

    if not skip_upload:
        if not model_path:
            print(json.dumps({
                "success": False,
                "error": "model_path required unless --skip-upload",
            }))
            sys.exit(1)
        if dry_run:
            logger.info(f"[dry-run] would upload {model_path} → {repo}")
            # Placeholder so subsequent dry-run messaging carries a value;
            # never actually commits anything.
            revision = revision or "<dry-run-pending-upload>"
        else:
            from huggingface_hub import HfApi

            try:
                api = HfApi(token=hf_token)
                api.create_repo(repo_id=repo, exist_ok=True, repo_type="model")
                api.upload_folder(folder_path=model_path, repo_id=repo, commit_message=message)
                info = api.repo_info(repo, repo_type="model")
                revision = info.sha
                logger.info(f"Uploaded {model_path} → {repo}@{revision[:8]}...")
            except Exception as e:
                logger.error(f"HF upload failed: {e}")
                print(json.dumps({"success": False, "error": str(e)}))
                sys.exit(1)
    elif not revision:
        print(json.dumps({
            "success": False,
            "error": "--skip-upload requires --revision",
        }))
        sys.exit(1)

    if skip_commit:
        print(json.dumps({
            "success": True, "repo": repo, "revision": revision, "committed": False,
        }))
        return

    if dry_run:
        logger.info(f"[dry-run] would commit {repo}@{revision}")
        print(json.dumps({
            "success": True, "repo": repo, "revision": revision, "dry_run": True,
        }))
        return

    await commit_command(repo=repo, revision=revision, coldkey=coldkey, hotkey=hotkey)


# --------------------------------------------------------------------------- #
# query commands
# --------------------------------------------------------------------------- #


async def get_weights_command() -> None:
    async with cli_api_client() as client:
        data = await client.get("/scores/weights/latest")
        if data:
            print(json.dumps(data, indent=2, ensure_ascii=False))


async def get_scores_command(top: int = 32) -> None:
    async with cli_api_client() as client:
        data = await client.get(f"/scores/latest?top={top}")
        if data:
            print(json.dumps(data, indent=2, ensure_ascii=False))


async def get_score_command(uid: int) -> None:
    async with cli_api_client() as client:
        data = await client.get(f"/scores/uid/{uid}")
        if data:
            print(json.dumps(data, indent=2, ensure_ascii=False))


async def get_miner_command(uid: Optional[int], hotkey: Optional[str]) -> None:
    if uid is None and not hotkey:
        print(json.dumps({
            "success": False,
            "error": "uid argument, --uid, or --hotkey required",
        }))
        sys.exit(1)
    if uid is not None and hotkey:
        print(json.dumps({
            "success": False,
            "error": "provide either uid or --hotkey, not both",
        }))
        sys.exit(1)
    endpoint = f"/miners/uid/{uid}" if uid is not None else f"/miners/hotkey/{hotkey}"
    async with cli_api_client() as client:
        data = await client.get(endpoint)
        if data:
            print(json.dumps(data, indent=2, ensure_ascii=False))
