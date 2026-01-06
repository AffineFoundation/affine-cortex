"""
Miner Commands Implementation

Provides command functions for miners:
- commit_command: Commit model to blockchain
- pull_command: Pull model from Hugging Face
- chutes_push_command: Deploy model to Chutes
- deploy_command: One-command deployment (upload → deploy → commit)
"""

import os
import sys
import json
import asyncio
import textwrap
from pathlib import Path
from typing import Optional
from affine.utils.api_client import cli_api_client
from affine.core.setup import logger, NETUID
from affine.utils.subtensor import get_subtensor


def get_conf(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get configuration value from environment variable."""
    return os.getenv(key, default)



# ============================================================================
# Command Implementations
# ============================================================================

async def pull_command(uid: int, model_path: str, hf_token: Optional[str] = None):
    """Pull model from Hugging Face.
    
    Args:
        uid: Miner UID
        model_path: Local directory to save model
        hf_token: Hugging Face API token (optional, from env if not provided)
    """
    from huggingface_hub import snapshot_download

    hf_token = hf_token or get_conf("HF_TOKEN")
    
    # Get miner info directly from subtensor
    try:
        subtensor = await get_subtensor()
        meta = await subtensor.metagraph(NETUID)
        commits = await subtensor.get_all_revealed_commitments(NETUID)
        
        if uid >= len(meta.hotkeys):
            logger.error(f"Invalid UID {uid}")
            print(json.dumps({"success": False, "error": f"Invalid UID {uid}"}))
            sys.exit(1)
        
        hotkey = meta.hotkeys[uid]
        
        if hotkey not in commits:
            logger.error(f"No commit found for UID {uid}")
            print(json.dumps({"success": False, "error": f"No commit found for UID {uid}"}))
            sys.exit(1)
        
        _, commit_data = commits[hotkey][-1]
        data = json.loads(commit_data)
        
        repo_name = data.get("model")
        revision = data.get("revision")
        
        if not repo_name:
            logger.error(f"Miner {uid} has no model configured")
            print(json.dumps({"success": False, "error": "No model configured"}))
            sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to get miner info: {e}")
        print(json.dumps({"success": False, "error": str(e)}))
        sys.exit(1)
    
    logger.info(f"Pulling model {repo_name}@{revision} for UID {uid} into {model_path}")
    
    try:
        snapshot_download(
            repo_id=repo_name,
            repo_type="model",
            local_dir=model_path,
            token=hf_token,
            resume_download=True,
            revision=revision,
        )
        
        result = {
            "success": True,
            "uid": uid,
            "repo": repo_name,
            "revision": revision,
            "path": model_path
        }
        print(json.dumps(result))
        logger.info(f"Model {repo_name} pulled successfully")
    
    except Exception as e:
        logger.error(f"Failed to download {repo_name}: {e}")
        print(json.dumps({"success": False, "error": str(e)}))
        sys.exit(1)


async def get_latest_chute_id(repo: str, api_key: str) -> Optional[str]:
    """Get latest chute ID for a repository.
    
    Args:
        repo: HF repository name
        api_key: Chutes API key
    
    Returns:
        Chute ID or None if not found
    """
    token = api_key or os.getenv("CHUTES_API_KEY", "")
    if not token:
        return None
    
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://api.chutes.ai/chutes/", headers={"Authorization": token}
            ) as r:
                if r.status != 200:
                    return None
                data = await r.json()
    except Exception:
        return None
    
    chutes = data.get("items", data) if isinstance(data, dict) else data
    if not isinstance(chutes, list):
        return None
    
    for chute in reversed(chutes):
        if any(chute.get(k) == repo for k in ("model_name", "name", "readme")):
            return chute.get("chute_id")
    return None


async def chutes_push_command(
    repo: str,
    revision: str,
    chutes_api_key: Optional[str] = None,
    chute_user: Optional[str] = None
):
    """Deploy model to Chutes.
    
    Args:
        repo: HF repository ID
        revision: HF commit SHA
        chutes_api_key: Chutes API key (optional, from env if not provided)
        chute_user: Chutes username (optional, from env if not provided)
    """
    chutes_api_key = chutes_api_key or get_conf("CHUTES_API_KEY")
    chute_user = chute_user or get_conf("CHUTE_USER")
    
    if not chutes_api_key:
        logger.error("CHUTES_API_KEY not configured")
        print(json.dumps({"success": False, "error": "CHUTES_API_KEY not configured"}))
        sys.exit(1)
    
    if not chute_user:
        logger.error("CHUTE_USER not configured")
        print(json.dumps({"success": False, "error": "CHUTE_USER not configured"}))
        sys.exit(1)
    
    logger.debug(f"Building Chute config for repo={repo} revision={revision}")
    
    # Generate Chute configuration
    chutes_config = textwrap.dedent(
        f"""
import os
from chutes.chute import NodeSelector
from chutes.chute.template.sglang import build_sglang_chute
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

chute = build_sglang_chute(
    username="{chute_user}",
    readme="{repo}",
    model_name="{repo}",
    image="chutes/sglang:nightly-2025081600",
    concurrency=40,
    revision="{revision}",
    node_selector=NodeSelector(
        gpu_count=4,
        include=["h200"],
    ),
    scaling_threshold=0.5,
    max_instances=2,
    shutdown_after_seconds=28800,
)
"""
    )
    
    tmp_file = Path("tmp_chute.py")
    tmp_file.write_text(chutes_config)
    logger.debug(f"Wrote Chute config to {tmp_file}")
    
    # Deploy to Chutes
    cmd = ["chutes", "deploy", f"{tmp_file.stem}:chute", "--accept-fee"]
    env = {**os.environ, "CHUTES_API_KEY": chutes_api_key}
    
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.PIPE,
        )
        
        if proc.stdin:
            proc.stdin.write(b"y\n")
            await proc.stdin.drain()
            proc.stdin.close()
        
        stdout, _ = await proc.communicate()
        output = stdout.decode(errors="ignore")
        logger.trace(output)
        
        # Check for errors
        import re
        match = re.search(
            r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3})\s+\|\s+(\w+)", output
        )
        if match and match.group(2) == "ERROR":
            logger.debug("Chutes deploy failed with error log")
            raise RuntimeError("Chutes deploy failed")
        
        if proc.returncode != 0:
            logger.debug(f"Chutes deploy failed with code {proc.returncode}")
            raise RuntimeError("Chutes deploy failed")
        
        tmp_file.unlink(missing_ok=True)
        logger.debug("Chute deployment successful")
        
        # Get chute info
        from affine.utils.api_client import get_chute_info
        chute_id = await get_latest_chute_id(repo, api_key=chutes_api_key)
        chute_info = await get_chute_info(chute_id) if chute_id else None
        
        result = {
            "success": bool(chute_id),
            "chute_id": chute_id,
            "chute": chute_info,
            "repo": repo,
            "revision": revision,
        }
        print(json.dumps(result))
        logger.info(f"Deployed to Chutes: {chute_id}")
    
    except Exception as e:
        logger.error(f"Chutes deployment failed: {e}")
        print(json.dumps({"success": False, "error": str(e)}))
        tmp_file.unlink(missing_ok=True)
        sys.exit(1)


async def commit_command(
    repo: str,
    revision: str,
    chute_id: str,
    coldkey: Optional[str] = None,
    hotkey: Optional[str] = None
):
    """Commit model to blockchain.

    Args:
        repo: HF repository ID
        revision: HF commit SHA
        chute_id: Chutes deployment ID
        coldkey: Wallet coldkey name (optional, from env if not provided)
        hotkey: Wallet hotkey name (optional, from env if not provided)
    """
    import bittensor as bt
    from bittensor.core.errors import MetadataError
    from affine.utils.subtensor import get_subtensor

    cold = coldkey or get_conf("BT_WALLET_COLD", "default")
    hot = hotkey or get_conf("BT_WALLET_HOT", "default")
    wallet = bt.Wallet(name=cold, hotkey=hot)
    
    logger.info(f"Committing: {repo}@{revision} (chute: {chute_id})")
    logger.info(f"Using wallet: {wallet.hotkey.ss58_address[:16]}...")

    async def _commit():
        sub = await get_subtensor()
        data = json.dumps({
            "model": repo,
            "revision": revision,
            "chute_id": chute_id
        })
        
        while True:
            try:
                await sub.set_reveal_commitment(
                    wallet=wallet,
                    netuid=NETUID,
                    data=data,
                    blocks_until_reveal=1
                )
                break
            except MetadataError as e:
                if "SpaceLimitExceeded" in str(e):
                    logger.warning("Space limit exceeded, waiting for next block...")
                    await sub.wait_for_block()
                else:
                    raise
    
    try:
        await _commit()
        
        result = {
            "success": True,
            "repo": repo,
            "revision": revision,
            "chute_id": chute_id,
        }
        print(json.dumps(result))
        logger.info("Commit successful")
    
    except Exception as e:
        logger.error(f"Commit failed: {e}")
        print(json.dumps({"success": False, "error": str(e)}))
        sys.exit(1)


async def get_sample_command(
    uid: int,
    env: str,
    task_id: str,
):
    """Query sample result by UID, environment, and task ID.
    
    Args:
        uid: Miner UID
        env: Environment name
        task_id: Task ID
    """
    
    async with cli_api_client() as client:
        endpoint = f"/samples/uid/{uid}/{env}/{task_id}"
        data = await client.get(endpoint)
        
        if data:
            print(json.dumps(data, indent=2, ensure_ascii=False))


async def get_miner_command(uid: int):
    """Query miner status and information by UID.
    
    Args:
        uid: Miner UID
    """
    async with cli_api_client() as client:
        endpoint = f"/miners/uid/{uid}"
        data = await client.get(endpoint)
        if data:
            print(json.dumps(data, indent=2, ensure_ascii=False))
            
            # Fetch and display sampling stats via API
            stats_endpoint = f"/miners/uid/{uid}/stats"
            stats_data = await client.get(stats_endpoint)
            
            if stats_data and stats_data.get('sampling_stats'):
                print("\n" + "="*80)
                print("GLOBAL SAMPLING STATISTICS")
                print("="*80)
                
                sampling_stats = stats_data['sampling_stats']
                
                for window in ['last_15min', 'last_1hour', 'last_6hours', 'last_24hours']:
                    if window in sampling_stats:
                        wstats = sampling_stats[window]
                        print(f"\n{window}:")
                        print(f"  Samples: {wstats.get('samples', 0)}")
                        print(f"  Success: {wstats.get('success', 0)}")
                        print(f"  Success rate: {wstats.get('success_rate', 0.0):.2%}")
                        print(f"  Samples/min: {wstats.get('samples_per_min', 0.0):.2f}")
                        print(f"  Rate limit errors: {wstats.get('rate_limit_errors', 0)}")
                        print(f"  Timeout errors: {wstats.get('timeout_errors', 0)}")
                        print(f"  Other errors: {wstats.get('other_errors', 0)}")
                
                # Display per-environment statistics
                if stats_data.get('env_stats'):
                    print("\n" + "="*80)
                    print("PER-ENVIRONMENT STATISTICS")
                    print("="*80)
                    
                    env_stats = stats_data['env_stats']
                    for env_name, env_data in env_stats.items():
                        print(f"\n[{env_name}]")
                        for window in ['last_15min', 'last_1hour', 'last_6hours', 'last_24hours']:
                            if window in env_data:
                                wstats = env_data[window]
                                print(f"  {window}:")
                                print(f"    Samples: {wstats.get('samples', 0)}")
                                print(f"    Success: {wstats.get('success', 0)}")
                                print(f"    Success rate: {wstats.get('success_rate', 0.0):.2%}")
                                print(f"    Samples/min: {wstats.get('samples_per_min', 0.0):.2f}")
                                print(f"    Rate limit errors: {wstats.get('rate_limit_errors', 0)}")
                                print(f"    Timeout errors: {wstats.get('timeout_errors', 0)}")
                                print(f"    Other errors: {wstats.get('other_errors', 0)}")
            else:
                print("\n" + "="*80)
                print("No sampling statistics available for this miner.")
                print("="*80)



async def get_weights_command():
    """Query latest normalized weights for on-chain weight setting.
    
    Returns the most recent score snapshot with normalized weights
    for all miners.
    """
    async with cli_api_client() as client:
        endpoint = "/scores/weights/latest"
        data = await client.get(endpoint)
        
        if data:
            print(json.dumps(data, indent=2, ensure_ascii=False))


async def get_scores_command(top: int = 32):
    """Query latest scores for top N miners.
    
    Args:
        top: Number of top miners to return (default: 256)
    """
    async with cli_api_client() as client:
        endpoint = f"/scores/latest?top={top}"
        data = await client.get(endpoint)
        
        if data:
            print(json.dumps(data, indent=2, ensure_ascii=False))


async def get_score_command(uid: int):
    """Query score for a specific miner by UID.
    
    Args:
        uid: Miner UID
    """
    async with cli_api_client() as client:
        endpoint = f"/scores/uid/{uid}"
        data = await client.get(endpoint)
        
        if data:
            print(json.dumps(data, indent=2, ensure_ascii=False))


async def get_pool_command(uid: int, env: str, full: bool = False):
    """Query task pool status for a miner in an environment.
    
    Args:
        uid: Miner UID
        env: Environment name (e.g., agentgym:webshop)
        full: If True, print full task_ids lists without truncation
    """
    async with cli_api_client() as client:
        endpoint = f"/samples/pool/uid/{uid}/{env}"
        data = await client.get(endpoint)
        
        if data:
            if data.get("success") is False:
                print(json.dumps({
                    "error": data.get("error"),
                    "status_code": data.get("status_code")
                }, indent=2, ensure_ascii=False))
                return
            if full:
                # Print full data without truncation
                print(json.dumps(data, indent=2, ensure_ascii=False))
            else:
                # Format output for better readability
                # Show summary first, then task_ids ranges instead of full lists
                summary = {
                    "uid": data.get("uid"),
                    "hotkey": data.get("hotkey"),
                    "model_revision": data.get("model_revision"),
                    "env": data.get("env"),
                    "sampling_config": data.get("sampling_config", {}),
                    "total_tasks": data.get("total_tasks"),
                    "sampled_count": data.get("sampled_count"),
                    "pool_count": data.get("pool_count"),
                    "missing_count": data.get("missing_count"),
                }
                
                # Helper function to format task_id list as ranges
                def format_task_ids(task_ids):
                    if not task_ids:
                        return "[]"
                    if len(task_ids) <= 10:
                        return str(task_ids)
                    # Show first 5 and last 5
                    return f"[{', '.join(map(str, task_ids[:5]))}, ..., {', '.join(map(str, task_ids[-5:]))}] (total: {len(task_ids)})"
                
                summary["sampled_task_ids"] = format_task_ids(data.get("sampled_task_ids", []))
                summary["pool_task_ids"] = format_task_ids(data.get("pool_task_ids", []))
                summary["missing_task_ids"] = format_task_ids(data.get("missing_task_ids", []))
                
                print(json.dumps(summary, indent=2, ensure_ascii=False))


async def get_envs_command():
    """Query current environment configurations.
    
    Returns all environment configurations including sampling settings,
    rotation settings, and enabled flags.
    """
    async with cli_api_client() as client:
        endpoint = "/config/environments"
        data = await client.get(endpoint)
        
        if data:
            print(json.dumps(data, indent=2, ensure_ascii=False))


async def deploy_command(
    repo: str,
    model_path: Optional[str] = None,
    revision: Optional[str] = None,
    chute_id: Optional[str] = None,
    message: str = "Model update",
    dry_run: bool = False,
    skip_upload: bool = False,
    skip_chutes: bool = False,
    skip_commit: bool = False,
    chutes_api_key: Optional[str] = None,
    chute_user: Optional[str] = None,
    coldkey: Optional[str] = None,
    hotkey: Optional[str] = None,
    hf_token: Optional[str] = None,
):
    """One-command deployment: Upload to HuggingFace -> Deploy to Chutes -> Commit on-chain.
    
    This combines the three-step deployment process into a single command:
    1. Upload model to HuggingFace (skip with --skip-upload if already uploaded)
    2. Deploy to Chutes (skip with --skip-chutes if already deployed)
    3. Commit on-chain (skip with --skip-commit to test without committing)
    
    Args:
        repo: HuggingFace repository ID (e.g., "username/model-name")
        model_path: Path to local model directory (required unless --skip-upload)
        revision: HuggingFace revision SHA (required if --skip-upload)
        chute_id: Chutes deployment ID (required if --skip-chutes)
        message: Commit message for HuggingFace upload
        dry_run: If True, show what would be done without executing
        skip_upload: Skip HuggingFace upload (requires --revision)
        skip_chutes: Skip Chutes deployment (requires --chute-id)
        skip_commit: Skip on-chain commit
        chutes_api_key: Chutes API key (optional, from env if not provided)
        chute_user: Chutes username (optional, from env if not provided)
        coldkey: Wallet coldkey name (optional, from env if not provided)
        hotkey: Wallet hotkey name (optional, from env if not provided)
        hf_token: HuggingFace token (optional, from env if not provided)
    """
    from huggingface_hub import HfApi
    
    chutes_api_key = chutes_api_key or get_conf("CHUTES_API_KEY")
    chute_user = chute_user or get_conf("CHUTE_USER")
    hf_token = hf_token or get_conf("HF_TOKEN")
    
    # Validate arguments based on skip flags
    if not skip_upload and not model_path:
        logger.error("--model-path is required unless --skip-upload is set")
        print(json.dumps({"success": False, "error": "--model-path is required unless --skip-upload is set"}))
        sys.exit(1)
    
    if skip_upload and not revision:
        logger.error("--revision is required when --skip-upload is set")
        print(json.dumps({"success": False, "error": "--revision is required when --skip-upload is set"}))
        sys.exit(1)
    
    if skip_chutes and not chute_id:
        logger.error("--chute-id is required when --skip-chutes is set")
        print(json.dumps({"success": False, "error": "--chute-id is required when --skip-chutes is set"}))
        sys.exit(1)
    
    # Validate required credentials
    if not dry_run:
        if not skip_upload and not hf_token:
            logger.error("HF_TOKEN not configured")
            print(json.dumps({"success": False, "error": "HF_TOKEN not configured"}))
            sys.exit(1)
        
        if not skip_chutes:
            if not chutes_api_key:
                logger.error("CHUTES_API_KEY not configured")
                print(json.dumps({"success": False, "error": "CHUTES_API_KEY not configured"}))
                sys.exit(1)
            
            if not chute_user:
                logger.error("CHUTE_USER not configured")
                print(json.dumps({"success": False, "error": "CHUTE_USER not configured"}))
                sys.exit(1)
    
    # Determine which steps to run
    steps = []
    if not skip_upload:
        steps.append("upload")
    if not skip_chutes:
        steps.append("chutes")
    if not skip_commit:
        steps.append("commit")
    
    total_steps = len(steps)
    current_step = 0
    
    logger.info("=" * 60)
    logger.info("AFFINE DEPLOYMENT")
    logger.info("=" * 60)
    logger.info(f"  Repository: {repo}")
    if model_path:
        logger.info(f"  Model Path: {model_path}")
    if revision:
        logger.info(f"  Revision: {revision}")
    if chute_id:
        logger.info(f"  Chute ID: {chute_id}")
    logger.info(f"  Steps: {' -> '.join(steps) if steps else 'none'}")
    if dry_run:
        logger.info("  Mode: DRY RUN")
    logger.info("=" * 60)
    
    # =========================================================================
    # Step 1: Upload to HuggingFace
    # =========================================================================
    if not skip_upload:
        current_step += 1
        logger.info(f"[{current_step}/{total_steps}] Uploading to HuggingFace ({repo})...")
        
        if dry_run:
            logger.info(f"  [DRY RUN] Would upload {model_path} to {repo}")
            revision = "dry-run-revision-sha"
        else:
            try:
                api = HfApi(token=hf_token)
                
                # Create repo if doesn't exist
                try:
                    api.create_repo(repo, exist_ok=True, repo_type="model")
                    logger.debug(f"Repository {repo} ready")
                except Exception as e:
                    logger.debug(f"Repo creation note: {e}")
                
                # Upload folder
                logger.info(f"  Uploading {model_path}...")
                api.upload_folder(
                    folder_path=model_path,
                    repo_id=repo,
                    commit_message=message
                )
                
                # Get latest commit SHA
                info = api.repo_info(repo, repo_type="model")
                revision = info.sha
                
                logger.info(f"  Upload complete. Revision: {revision[:12]}...")
                
            except Exception as e:
                logger.error(f"HuggingFace upload failed: {e}")
                print(json.dumps({"success": False, "error": f"HuggingFace upload failed: {str(e)}"}))
                sys.exit(1)
    else:
        logger.info(f"Skipping upload, using revision: {revision[:12]}...")
    
    # =========================================================================
    # Step 2: Deploy to Chutes
    # =========================================================================
    if not skip_chutes:
        current_step += 1
        logger.info(f"[{current_step}/{total_steps}] Deploying to Chutes...")
        
        if dry_run:
            logger.info(f"  [DRY RUN] Would deploy {repo}@{revision[:12]}...")
            chute_id = "dry-run-chute-id"
        else:
            try:
                # Generate Chute configuration (same as chutes_push_command)
                chutes_config = textwrap.dedent(
                    f"""
import os
from chutes.chute import NodeSelector
from chutes.chute.template.sglang import build_sglang_chute
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

chute = build_sglang_chute(
    username="{chute_user}",
    readme="{repo}",
    model_name="{repo}",
    image="chutes/sglang:nightly-2025081600",
    concurrency=40,
    revision="{revision}",
    node_selector=NodeSelector(
        gpu_count=4,
        include=["h200"],
    ),
    scaling_threshold=0.5,
    max_instances=2,
    shutdown_after_seconds=28800,
)
"""
                )
                
                tmp_file = Path("tmp_chute.py")
                tmp_file.write_text(chutes_config)
                logger.debug(f"Wrote Chute config to {tmp_file}")
                
                # Deploy to Chutes
                cmd = ["chutes", "deploy", f"{tmp_file.stem}:chute", "--accept-fee"]
                env = {**os.environ, "CHUTES_API_KEY": chutes_api_key}
                
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    stdin=asyncio.subprocess.PIPE,
                )
                
                if proc.stdin:
                    proc.stdin.write(b"y\n")
                    await proc.stdin.drain()
                    proc.stdin.close()
                
                stdout, _ = await proc.communicate()
                output = stdout.decode(errors="ignore")
                logger.trace(output)
                
                # Check for errors
                import re
                match = re.search(
                    r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3})\s+\|\s+(\w+)", output
                )
                if match and match.group(2) == "ERROR":
                    logger.debug("Chutes deploy failed with error log")
                    raise RuntimeError("Chutes deploy failed")
                
                if proc.returncode != 0:
                    logger.debug(f"Chutes deploy failed with code {proc.returncode}")
                    raise RuntimeError("Chutes deploy failed")
                
                tmp_file.unlink(missing_ok=True)
                logger.debug("Chute deployment successful")
                
                # Get chute ID
                chute_id = await get_latest_chute_id(repo, api_key=chutes_api_key)
                
                if not chute_id:
                    raise RuntimeError("Failed to get chute_id after deployment")
                
                logger.info(f"  Chutes deployment complete. Chute ID: {chute_id}")
                
            except Exception as e:
                logger.error(f"Chutes deployment failed: {e}")
                if 'tmp_file' in locals():
                    tmp_file.unlink(missing_ok=True)
                print(json.dumps({"success": False, "error": f"Chutes deployment failed: {str(e)}"}))
                sys.exit(1)
    else:
        logger.info(f"Skipping Chutes deployment, using chute_id: {chute_id}")
    
    # =========================================================================
    # Step 3: Commit on-chain
    # =========================================================================
    if not skip_commit:
        current_step += 1
        logger.info(f"[{current_step}/{total_steps}] Committing on-chain...")
        
        if dry_run:
            logger.info(f"  [DRY RUN] Would commit {repo}@{revision[:12]}... with chute {chute_id}")
        else:
            try:
                import bittensor as bt
                from bittensor.core.errors import MetadataError
                from affine.utils.subtensor import get_subtensor
                
                cold = coldkey or get_conf("BT_WALLET_COLD", "default")
                hot = hotkey or get_conf("BT_WALLET_HOT", "default")
                wallet = bt.Wallet(name=cold, hotkey=hot)
                
                logger.info(f"  Using wallet: {wallet.hotkey.ss58_address[:16]}...")
                
                sub = await get_subtensor()
                data = json.dumps({
                    "model": repo,
                    "revision": revision,
                    "chute_id": chute_id
                })
                
                while True:
                    try:
                        await sub.set_reveal_commitment(
                            wallet=wallet,
                            netuid=NETUID,
                            data=data,
                            blocks_until_reveal=1
                        )
                        break
                    except MetadataError as e:
                        if "SpaceLimitExceeded" in str(e):
                            logger.warning("Space limit exceeded, waiting for next block...")
                            await sub.wait_for_block()
                        else:
                            raise
                
                logger.info("  Commit successful")
                
            except Exception as e:
                logger.error(f"On-chain commit failed: {e}")
                print(json.dumps({"success": False, "error": f"On-chain commit failed: {str(e)}"}))
                sys.exit(1)
    else:
        logger.info("Skipping on-chain commit")
    
    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("=" * 60)
    if dry_run:
        logger.info("DRY RUN COMPLETE - No changes were made")
    else:
        logger.info("DEPLOYMENT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Repository: {repo}")
    logger.info(f"  Revision: {revision[:12] if revision and len(revision) > 12 else revision}...")
    logger.info(f"  Chute ID: {chute_id}")
    logger.info("=" * 60)
    
    result = {
        "success": True,
        "repo": repo,
        "revision": revision,
        "chute_id": chute_id,
        "dry_run": dry_run
    }
    print(json.dumps(result))