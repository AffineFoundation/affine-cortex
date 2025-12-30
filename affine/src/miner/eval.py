#!/usr/bin/env python3
"""
Affine Eval CLI Command

Unified evaluation command supporting multiple model source modes:
1. --uid: Evaluate a registered miner by UID (fetches chute info from metagraph)
2. --chute-slug: Evaluate via Chutes slug directly
3. --chute-id: Evaluate via Chutes deployment ID (fetches slug via API)
4. --base-url: Evaluate any model via custom base_url (local or remote)

Only one of these model source options can be specified at a time.

Execution modes (independent from model source):
- Default: Docker mode - uses local persistent Docker containers
- --basilica: Basilica mode - creates temporary cloud pods per evaluation

Examples:
    # Mode 1: By miner UID (default Docker execution)
    af eval --env affine:ded-v2 --uid 7 --samples 1

    # Mode 2: By Chutes slug (default Docker execution)
    af eval --env affine:ded-v2 --chute-slug my-model-abc123 --model deepseek-ai/DeepSeek-V3 --samples 1

    # Mode 3: By Chutes ID (default Docker execution)
    af eval --env affine:ded-v2 --chute-id abc-123-def --model deepseek-ai/DeepSeek-V3 --samples 1

    # Mode 4: By base URL with local inference server (requires --network-host)
    af eval --env affine:ded-v2 --base-url http://localhost:8000/v1 --model my-model --samples 1 --network-host
    
    # Using Basilica cloud execution (requires BASILICA_API_TOKEN)
    export BASILICA_API_TOKEN='your-token'
    af eval --env GAME --uid 7 --task-id 502284834 --basilica
"""

import asyncio
import json
import os
import sys
import time
import random
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import click
from dotenv import load_dotenv


def get_available_environments() -> List[str]:
    """Get list of available environment names."""
    from affine.core.environments import ENV_CONFIGS
    return sorted(set(ENV_CONFIGS.keys()))


@click.command("eval")
@click.option(
    "--env", "-e",
    required=False,
    default=None,
    help="Environment name (e.g., affine:ded-v2, agentgym:alfworld). Use --list-envs to see all."
)
@click.option(
    "--uid", "-u",
    type=int,
    default=None,
    help="Miner UID (fetches model/chute info from metagraph)"
)
@click.option(
    "--chute-slug",
    default=None,
    help="Chutes slug (e.g., my-model-abc123)"
)
@click.option(
    "--chute-id",
    default=None,
    help="Chutes deployment ID (fetches slug via API)"
)
@click.option(
    "--base-url", "-b",
    default=None,
    help="Base URL for inference (e.g., http://localhost:8000/v1)"
)
@click.option(
    "--model", "-m",
    default=None,
    help="Model name (required for --chute-slug, --chute-id, --base-url modes)"
)
@click.option(
    "--samples", "-n",
    type=int,
    default=1,
    help="Number of evaluation samples (default: 1)"
)
@click.option(
    "--task-id", "-t",
    type=int,
    default=None,
    help="Specific task ID for evaluation"
)
@click.option(
    "--task-id-range",
    nargs=2,
    type=int,
    default=None,
    help="Range of task IDs (start end), one sample per task"
)
@click.option(
    "--temperature",
    type=float,
    default=0.0,
    help="Sampling temperature (default: 0.0)"
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for evaluation"
)
@click.option(
    "--network-host",
    is_flag=True,
    default=False,
    help="Use host network mode (required for localhost access from container)"
)
@click.option(
    "--output", "-o",
    default=None,
    help="Output file path for JSON results (default: eval/results_yyyy_mm_dd_hh_mm_ss.json)"
)
@click.option(
    "--list-envs",
    is_flag=True,
    default=False,
    help="List all available environments and exit"
)
@click.option(
    "--basilica",
    is_flag=True,
    default=False,
    help="Use Basilica cloud execution (creates temporary pods, requires BASILICA_API_TOKEN). Default: Docker mode with persistent containers."
)
def eval_cmd(
    env: str,
    uid: Optional[int],
    chute_slug: Optional[str],
    chute_id: Optional[str],
    base_url: Optional[str],
    model: Optional[str],
    samples: int,
    task_id: Optional[int],
    task_id_range: Optional[Tuple[int, int]],
    temperature: float,
    seed: Optional[int],
    network_host: bool,
    output: Optional[str],
    list_envs: bool,
    basilica: bool,
):
    """Evaluate models on Affine environments.
    
    MODEL SOURCE MODES (mutually exclusive, choose one):
    
    \b
    1. --uid        : Miner UID (fetches model/chute from metagraph)
    2. --chute-slug : Direct Chutes slug
    3. --chute-id   : Chutes deployment ID (fetches slug via API)
    4. --base-url   : Any OpenAI-compatible endpoint (local or remote)
    
    For modes 2-4, --model is required to specify the model name.
    
    EXECUTION MODES (independent):
    
    \b
    - Default       : Docker mode (persistent local containers)
    - --basilica    : Basilica mode (temporary cloud pods, needs BASILICA_API_TOKEN)
    
    SPECIAL OPTIONS:
    
    \b
    - --network-host: Required for --base-url with localhost servers
    - --list-envs   : Show all available environments
    
    \b
    Examples:
        # Evaluate miner UID 7 with 10 random samples
        af eval --env GAME --uid 7 --samples 10
        
        # Evaluate specific task ID
        af eval --env GAME --uid 7 --task-id 502284834
        
        # Evaluate task ID range (one sample per task)
        af eval --env GAME --uid 7 --task-id-range 100 110
        
        # Evaluate using Chutes slug
        af eval --env GAME --chute-slug my-model-abc123 --model deepseek-ai/DeepSeek-V3 --samples 5
        
        # Evaluate local model server (requires --network-host)
        af eval --env GAME --base-url http://localhost:8000/v1 --model my-model --network-host
        
        # Evaluate using Basilica cloud execution
        export BASILICA_API_TOKEN='your-token'
        af eval --env GAME --uid 7 --task-id 502284834 --basilica
        
        # List available environments
        af eval --list-envs
    """
    # Handle --list-envs
    if list_envs:
        envs = get_available_environments()
        click.echo("Available environments:")
        for e in envs:
            click.echo(f"  - {e}")
        return
    
    # Validate --env is provided (required unless --list-envs)
    if env is None:
        raise click.UsageError("--env is required")
    
    # Validate mutually exclusive options
    mode_options = [
        ("--uid", uid),
        ("--chute-slug", chute_slug),
        ("--chute-id", chute_id),
        ("--base-url", base_url),
    ]
    specified = [(name, val) for name, val in mode_options if val is not None]
    
    if len(specified) == 0:
        raise click.UsageError(
            "One of --uid, --chute-slug, --chute-id, or --base-url is required"
        )
    
    if len(specified) > 1:
        names = [name for name, _ in specified]
        raise click.UsageError(
            f"Options {', '.join(names)} are mutually exclusive. Specify only one."
        )
    
    # Validate --model requirement for non-uid modes
    if uid is None and model is None:
        raise click.UsageError(
            "--model is required when using --chute-slug, --chute-id, or --base-url"
        )
    
    # Validate environment
    available_envs = get_available_environments()
    env_lower = env.lower()
    if env_lower not in [e.lower() for e in available_envs]:
        raise click.UsageError(
            f"Unknown environment: {env}\n"
            f"Use --list-envs to see available environments"
        )
    
    # Run async evaluation
    asyncio.run(_run_evaluation(
        env=env,
        uid=uid,
        chute_slug=chute_slug,
        chute_id=chute_id,
        base_url=base_url,
        model=model,
        samples=samples,
        task_id=task_id,
        task_id_range=task_id_range,
        temperature=temperature,
        seed=seed,
        network_host=network_host,
        output=output,
        basilica=basilica,
    ))


async def _resolve_endpoint(
    uid: Optional[int],
    chute_slug: Optional[str],
    chute_id: Optional[str],
    base_url: Optional[str],
    model: Optional[str],
) -> Tuple[str, str]:
    """Resolve the endpoint URL and model name based on the mode.
    
    Returns:
        Tuple of (base_url, model_name)
    """
    import affine as af
    from affine.utils.api_client import get_chute_info
    
    if uid is not None:
        # Mode 1: Fetch from metagraph
        click.echo(f"Fetching miner info for UID {uid}...")
        miner_dict = await af.miners(uid)
        if not miner_dict or uid not in miner_dict:
            raise click.ClickException(f"Unable to get miner info for UID {uid}")
        miner = miner_dict[uid]
        resolved_model = miner.model
        resolved_url = f"https://{miner.slug}.chutes.ai/v1"
        click.echo(f"  Model: {resolved_model}")
        click.echo(f"  Chute: {miner.slug}")
        return resolved_url, resolved_model
    
    elif chute_slug is not None:
        # Mode 2: Direct slug
        resolved_url = f"https://{chute_slug}.chutes.ai/v1"
        click.echo(f"Using Chutes slug: {chute_slug}")
        return resolved_url, model
    
    elif chute_id is not None:
        # Mode 3: Fetch slug from chute ID
        click.echo(f"Fetching chute info for ID {chute_id}...")
        chute_info = await get_chute_info(chute_id)
        if not chute_info:
            raise click.ClickException(f"Unable to get chute info for ID {chute_id}")
        slug = chute_info.get("slug")
        if not slug:
            raise click.ClickException(f"Chute {chute_id} has no slug")
        resolved_url = f"https://{slug}.chutes.ai/v1"
        click.echo(f"  Slug: {slug}")
        return resolved_url, model
    
    else:
        # Mode 4: Direct base URL
        click.echo(f"Using base URL: {base_url}")
        return base_url, model


async def _run_evaluation(
    env: str,
    uid: Optional[int],
    chute_slug: Optional[str],
    chute_id: Optional[str],
    base_url: Optional[str],
    model: Optional[str],
    samples: int,
    task_id: Optional[int],
    task_id_range: Optional[Tuple[int, int]],
    temperature: float,
    seed: Optional[int],
    network_host: bool,
    output: Optional[str],
    basilica: bool,
):
    """Run the evaluation asynchronously."""
    load_dotenv()
    
    from affine.core.setup import logger
    
    # Check API key requirement
    api_key = os.getenv("CHUTES_API_KEY")
    if uid is not None or chute_id is not None:
        if not api_key:
            click.echo("\n❌ CHUTES_API_KEY environment variable not set")
            click.echo("   Please set: export CHUTES_API_KEY='your-key'")
            click.echo("   Or create .env file with: CHUTES_API_KEY=your-key")
            sys.exit(1)
    
    # Set placeholder API key for local evaluation (required by affinetes Docker env)
    if not api_key:
        os.environ["CHUTES_API_KEY"] = "local-eval-placeholder"
        logger.debug("Set placeholder CHUTES_API_KEY for local evaluation")
    
    # Resolve endpoint
    click.echo("=" * 60)
    resolved_url, resolved_model = await _resolve_endpoint(
        uid=uid,
        chute_slug=chute_slug,
        chute_id=chute_id,
        base_url=base_url,
        model=model,
    )
    
    # Print configuration
    click.echo("\nEvaluation Configuration:")
    click.echo(f"  Environment: {env}")
    click.echo(f"  Model: {resolved_model}")
    click.echo(f"  Base URL: {resolved_url}")
    click.echo(f"  Execution Mode: {'Basilica (temporary pods)' if basilica else 'Docker (persistent containers)'}")
    if task_id is not None:
        click.echo(f"  Task ID: {task_id}")
    if task_id_range is not None:
        click.echo(f"  Task ID Range: {task_id_range[0]} - {task_id_range[1]}")
    click.echo(f"  Samples: {samples}")
    click.echo(f"  Temperature: {temperature}")
    if seed is not None:
        click.echo(f"  Seed: {seed}")
    if network_host:
        click.echo("  Network Mode: host")
    click.echo("=" * 60)
    
    try:
        # Create environment instance
        click.echo(f"\nLoading {env} environment...")
        env_instance = _create_environment(env, network_host, basilica)
        click.echo("✓ Environment loaded")
        
        # Run evaluation based on mode
        if task_id_range is not None:
            # Range mode: one sample per task
            results = await _evaluate_range(
                env_instance=env_instance,
                model=resolved_model,
                base_url=resolved_url,
                task_id_start=task_id_range[0],
                task_id_end=task_id_range[1],
                temperature=temperature,
            )
        else:
            # Single/multi-sample mode
            results = await _evaluate_samples(
                env_instance=env_instance,
                model=resolved_model,
                base_url=resolved_url,
                samples=samples,
                task_id=task_id,
                seed=seed,
                temperature=temperature,
            )
        
        # Calculate summary
        total_score = sum(r.get("score", 0.0) for r in results)
        total_time = sum(r.get("latency_seconds", 0.0) for r in results)
        total_samples = len(results)
        
        summary = {
            "environment": env,
            "model": resolved_model,
            "base_url": resolved_url,
            "samples": total_samples,
            "total_score": total_score,
            "average_score": total_score / total_samples if total_samples > 0 else 0,
            "total_time": total_time,
            "average_time": total_time / total_samples if total_samples > 0 else 0,
            "temperature": temperature,
            "results": results,
        }
        
        if uid is not None:
            summary["uid"] = uid
        if chute_slug is not None:
            summary["chute_slug"] = chute_slug
        if chute_id is not None:
            summary["chute_id"] = chute_id
        if seed is not None:
            summary["seed"] = seed
        
        # Generate default output path if not specified
        if output is None:
            timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            output = f"eval/results_{timestamp}.json"
        
        # Save to file
        try:
            # Ensure parent directory exists
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            click.echo(f"\n✓ Results saved to: {output}")
        except Exception as e:
            click.echo(f"\n✗ Failed to save results: {e}")
        
        # Print summary
        click.echo("\n" + "=" * 60)
        click.echo("Evaluation Summary:")
        click.echo(f"  Environment: {env}")
        click.echo(f"  Total Samples: {total_samples}")
        click.echo(f"  Total Score: {total_score:.4f}")
        click.echo(f"  Average Score: {summary['average_score']:.4f}")
        click.echo(f"  Total Time: {total_time:.2f} seconds")
        click.echo(f"  Average Time: {summary['average_time']:.2f} seconds/sample")
        
        # Show detailed results
        if total_samples > 1:
            click.echo("\nDetailed Results:")
            for idx, r in enumerate(results):
                status = "✓" if r.get("success", False) else "✗"
                score = r.get("score", 0.0)
                latency = r.get("latency_seconds", 0.0)
                task = r.get("task_id", "N/A")
                click.echo(f"  [{status}] Sample {idx}: task={task}, score={score:.4f}, time={latency:.2f}s")
                if r.get("error"):
                    click.echo(f"      Error: {r['error']}")
        
        click.echo("=" * 60)
        
    except KeyboardInterrupt:
        click.echo("\n\nEvaluation interrupted by user")
        sys.exit(0)
    except Exception as e:
        click.echo(f"\n✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def _create_environment(env_name: str, network_host: bool = False, basilica: bool = False):
    """Create environment instance with optional network_host and basilica mode."""
    from affine.core.environments import ENV_CONFIGS, convert_memory_format
    from affine.core.setup import logger
    import affinetes as af_env
    
    # Get config (try both original and lowercase)
    config = ENV_CONFIGS.get(env_name) or ENV_CONFIGS.get(env_name.lower())
    if config is None:
        raise ValueError(f"Unknown environment: {env_name}")
    
    # Build env_vars
    api_key = os.getenv("CHUTES_API_KEY", "")
    env_vars = {"CHUTES_API_KEY": api_key}
    
    # Add ENV_NAME for affine environments
    if "task_type" in config.eval_params:
        env_vars["ENV_NAME"] = config.eval_params["task_type"]
    
    env_vars.update(config.env_vars)
    
    # Choose execution mode
    mode = "basilica" if basilica else "docker"
    
    # Convert memory format for the selected mode
    mem_limit = convert_memory_format(config.mem_limit, mode)
    
    load_kwargs = {
        "image": config.docker_image,
        "mode": mode,
        "env_vars": env_vars,
        "mem_limit": mem_limit,
    }
    
    # Mode-specific parameters
    if mode == "docker":
        load_kwargs.update({
            "replicas": 1,
            "hosts": ["localhost"],
            "container_name": config.name.replace(":", "-") + "-eval",
            "pull": True,
            "force_recreate": True,
        })
        
        # Add volumes if configured
        if config.volumes:
            load_kwargs["volumes"] = config.volumes
        
        # Add network_mode for host network access
        if network_host:
            load_kwargs["network_mode"] = "host"
            logger.info(f"Using host network mode for {env_name}")
    
    elif mode == "basilica":
        # Basilica mode uses cpu_limit and requires BASILICA_API_TOKEN
        if not os.getenv("BASILICA_API_TOKEN"):
            raise ValueError(
                "BASILICA_API_TOKEN environment variable is required for --basilica mode. "
                "Set it with: export BASILICA_API_TOKEN='your-token'"
            )
        
        if hasattr(config, 'cpu_limit') and config.cpu_limit:
            load_kwargs["cpu_limit"] = config.cpu_limit
        
        logger.info(f"Using Basilica mode for {env_name} (temporary pods)")
    
    logger.info(f"Loading environment: {env_name} (mode={mode}, image={config.docker_image})")
    
    env = af_env.load_env(**load_kwargs)
    return _EnvironmentWrapper(env, config)


class _EnvironmentWrapper:
    """Wrapper for environment with config."""
    
    def __init__(self, env, config):
        self.env = env
        self.config = config
    
    async def evaluate(self, **kwargs):
        """Evaluate with config defaults."""
        # Merge config eval_params (kwargs take precedence)
        for key, value in self.config.eval_params.items():
            kwargs.setdefault(key, value)
        
        result = await self.env.evaluate(
            _timeout=self.config.proxy_timeout,
            **kwargs
        )
        return result


async def _evaluate_samples(
    env_instance: _EnvironmentWrapper,
    model: str,
    base_url: str,
    samples: int,
    task_id: Optional[int],
    seed: Optional[int],
    temperature: float,
) -> List[Dict[str, Any]]:
    """Evaluate multiple samples."""
    results = []
    
    click.echo(f"\nStarting evaluation ({samples} sample(s))...")
    
    for i in range(samples):
        click.echo(f"\rProgress: {i+1}/{samples}", nl=False)
        
        eval_kwargs = {
            "model": model,
            "base_url": base_url,
            "temperature": temperature,
        }
        
        # Set task_id (generate random if not specified)
        if task_id is not None:
            eval_kwargs["task_id"] = task_id
        else:
            eval_kwargs["task_id"] = random.randint(0, 10000)
        
        if seed is not None:
            eval_kwargs["seed"] = seed
        
        start_time = time.monotonic()
        result = await env_instance.evaluate(**eval_kwargs)
        latency = time.monotonic() - start_time
        
        # Convert result to dict
        if hasattr(result, "model_dump"):
            result_dict = result.model_dump()
        elif hasattr(result, "dict"):
            result_dict = result.dict()
        elif isinstance(result, dict):
            result_dict = result
        else:
            result_dict = {"raw": str(result)}
        
        result_dict["latency_seconds"] = latency
        result_dict["task_id"] = eval_kwargs["task_id"]
        results.append(result_dict)
    
    click.echo()  # New line after progress
    return results


async def _evaluate_range(
    env_instance: _EnvironmentWrapper,
    model: str,
    base_url: str,
    task_id_start: int,
    task_id_end: int,
    temperature: float,
) -> List[Dict[str, Any]]:
    """Evaluate across a range of task IDs (one sample per task)."""
    results = []
    
    task_count = task_id_end - task_id_start + 1
    click.echo(f"\nStarting evaluation ({task_count} tasks)...")
    
    for idx, task_id in enumerate(range(task_id_start, task_id_end + 1)):
        click.echo(f"\rProgress: {idx+1}/{task_count} (Task {task_id})", nl=False)
        
        eval_kwargs = {
            "model": model,
            "base_url": base_url,
            "temperature": temperature,
            "task_id": task_id,
        }
        
        start_time = time.monotonic()
        result = await env_instance.evaluate(**eval_kwargs)
        latency = time.monotonic() - start_time
        
        # Convert result to dict
        if hasattr(result, "model_dump"):
            result_dict = result.model_dump()
        elif hasattr(result, "dict"):
            result_dict = result.dict()
        elif isinstance(result, dict):
            result_dict = result
        else:
            result_dict = {"raw": str(result)}
        
        result_dict["latency_seconds"] = latency
        result_dict["task_id"] = task_id
        results.append(result_dict)
    
    click.echo()  # New line after progress
    return results


# For standalone testing
if __name__ == "__main__":
    eval_cmd()
