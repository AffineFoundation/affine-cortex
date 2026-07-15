"""
``af eval`` — local evaluator for an OpenAI-compatible inference endpoint.

This is a developer/debugging tool that exercises one Affine environment
against any URL serving the OpenAI Chat-Completions API. Use it to
sanity-check a fine-tune locally before committing on chain, or to spot-
test a model someone is hosting.

The scheduler service handles all production sampling — the queue-window
flow doesn't go through this CLI. Use ``--base-url`` + ``--model``.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click


def _available_environments() -> List[str]:
    from affine.core.environments import ENV_CONFIGS

    return sorted(ENV_CONFIGS.keys())


@click.command("eval")
@click.option(
    "--env",
    "-e",
    default=None,
    help="Environment name (e.g. affine:ded-v2). Use --list-envs to enumerate.",
)
@click.option(
    "--base-url",
    "-b",
    default=None,
    help="OpenAI-compatible base URL (e.g. http://localhost:8000/v1).",
)
@click.option(
    "--model",
    "-m",
    default=None,
    help="Model name the endpoint expects in chat-completions requests.",
)
@click.option(
    "--image",
    default=None,
    help="Use a local environment image override and skip pulling it.",
)
@click.option("--samples", "-n", type=int, default=1, help="Number of samples")
@click.option("--task-id", "-t", type=int, default=None, help="Pin a specific task_id")
@click.option(
    "--task-id-range",
    nargs=2,
    type=int,
    default=None,
    help="task_id range [start end) — one sample per task",
)
@click.option("--temperature", type=float, default=0.0)
@click.option("--seed", type=int, default=None)
@click.option(
    "--api-key",
    default=None,
    help="Explicit per-call model API key; never injected into the env container.",
)
@click.option(
    "--network-host",
    is_flag=True,
    default=False,
    help="Use Docker host networking (needed for localhost --base-url)",
)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output JSON path (default: eval/results_<ts>.json)",
)
@click.option("--list-envs", is_flag=True, default=False, help="List envs and exit")
@click.option(
    "--basilica",
    is_flag=True,
    default=False,
    help="Run env containers on Basilica (needs BASILICA_API_TOKEN)",
)
@click.option("--delay", type=float, default=0.0, help="Seconds between samples")
@click.option(
    "--max-retries", type=int, default=3, help="Retries on timeout / rate-limit errors"
)
def eval_cmd(
    env: Optional[str],
    base_url: Optional[str],
    model: Optional[str],
    image: Optional[str],
    samples: int,
    task_id: Optional[int],
    task_id_range: Optional[Tuple[int, int]],
    temperature: float,
    seed: Optional[int],
    api_key: Optional[str],
    network_host: bool,
    output: Optional[str],
    list_envs: bool,
    basilica: bool,
    delay: float,
    max_retries: int,
):
    """Evaluate an OpenAI endpoint on one Affine environment."""
    if list_envs:
        for e in _available_environments():
            click.echo(f"  - {e}")
        return

    if env is None:
        raise click.UsageError("--env is required (or use --list-envs)")
    if not base_url:
        raise click.UsageError("--base-url is required")
    if not model:
        raise click.UsageError("--model is required")

    available = _available_environments()
    resolved_key = next(
        (candidate for candidate in available if candidate.lower() == env.lower()),
        None,
    )
    if resolved_key is None:
        raise click.UsageError(
            f"Unknown environment: {env}\nUse --list-envs to see available envs"
        )
    from affine.core.environments import ENV_CONFIGS

    resolved_env = ENV_CONFIGS[resolved_key].name

    asyncio.run(
        _run(
            env=resolved_env,
            base_url=base_url,
            model=model,
            image=image,
            samples=samples,
            task_id=task_id,
            task_id_range=task_id_range,
            temperature=temperature,
            seed=seed,
            api_key=api_key,
            network_host=network_host,
            output=output,
            basilica=basilica,
            delay=delay,
            max_retries=max_retries,
        )
    )


# --------------------------------------------------------------------------- #
# execution
# --------------------------------------------------------------------------- #


async def _run(
    *,
    env: str,
    base_url: str,
    model: str,
    image: Optional[str],
    samples: int,
    task_id: Optional[int],
    task_id_range: Optional[Tuple[int, int]],
    temperature: float,
    seed: Optional[int],
    api_key: Optional[str],
    network_host: bool,
    output: Optional[str],
    basilica: bool,
    delay: float,
    max_retries: int,
) -> None:
    env_instance = _load_environment(
        env,
        network_host=network_host,
        basilica=basilica,
        image_override=image,
    )
    try:
        if task_id_range is not None:
            start, end = task_id_range
            results = await _evaluate_range(
                env_instance,
                base_url,
                model,
                temperature,
                max_retries,
                start,
                end,
                delay,
                seed=seed,
                api_key=api_key,
            )
        else:
            results = await _evaluate_samples(
                env_instance,
                base_url,
                model,
                samples,
                task_id,
                temperature,
                max_retries,
                delay,
                seed=seed,
                api_key=api_key,
            )
    finally:
        cleanup = getattr(getattr(env_instance, "env", None), "cleanup", None)
        if callable(cleanup):
            try:
                cleanup_result = cleanup()
                if inspect.isawaitable(cleanup_result):
                    await cleanup_result
            except Exception:
                pass

    _write_summary(env, base_url, model, temperature, seed, results, output)


def _load_environment(
    env_name: str,
    *,
    network_host: bool,
    basilica: bool,
    image_override: Optional[str] = None,
):
    """Bring up one affinetes env container and return a thin wrapper that
    forwards ``evaluate(**)`` plus the env's per-call timeout."""
    import affinetes as af_env
    from affine.core.environments import ENV_CONFIGS, validate_execution_mode

    config = ENV_CONFIGS[env_name]
    mode = validate_execution_mode(
        config,
        "basilica" if basilica else "docker",
    )
    env_vars = dict(config.env_vars or {})
    api_key = os.getenv("API_KEY")
    if config.forward_api_key:
        if api_key:
            env_vars["API_KEY"] = api_key
        for alias in ("CHUTES_API_KEY", "OPENAI_API_KEY"):
            value = os.getenv(alias) or api_key
            if value:
                env_vars[alias] = value
    for key in config.required_env_vars:
        value = os.getenv(key)
        if not value:
            raise click.ClickException(
                f"{key} environment variable is required for environment '{env_name}'"
            )
        env_vars[key] = value
    for key in getattr(config, "optional_env_vars", []):
        value = os.getenv(key)
        if value:
            env_vars[key] = value
    load_kwargs: Dict[str, Any] = {
        "image": image_override or config.docker_image,
        "mode": mode,
        "env_vars": env_vars,
        "mem_limit": config.mem_limit,
        "pull": image_override is None,
    }
    if mode == "docker":
        load_kwargs.update(
            {
                "replicas": 1,
                "hosts": ["localhost"],
                "container_name": config.name.replace(":", "-") + "-eval",
                "force_recreate": True,
            }
        )
        if config.volumes:
            load_kwargs["volumes"] = config.volumes
        if network_host:
            load_kwargs["host_network"] = True
    elif mode == "basilica":
        if not os.getenv("BASILICA_API_TOKEN"):
            raise click.ClickException(
                "BASILICA_API_TOKEN required for --basilica mode"
            )
        if getattr(config, "cpu_limit", None):
            load_kwargs["cpu_limit"] = config.cpu_limit

    env = af_env.load_env(**load_kwargs)
    return _EnvWrapper(env, config)


class _EnvWrapper:
    def __init__(self, env, config):
        self.env = env
        self.config = config

    async def evaluate(self, **kwargs):
        for k, v in self.config.eval_params.items():
            kwargs.setdefault(k, v)
        return await self.env.evaluate(_timeout=self.config.proxy_timeout, **kwargs)


# --------------------------------------------------------------------------- #
# evaluation loops
# --------------------------------------------------------------------------- #


_RETRYABLE_TOKENS = ("504", "timeout", "rate", "limit", "429", "too many", "upstream")


def _is_retryable(exc: Exception) -> bool:
    return any(tok in str(exc).lower() for tok in _RETRYABLE_TOKENS)


def _sanitize_result(value: Any) -> Any:
    from affine.core.environments import _remove_sensitive_request_fields

    return _remove_sensitive_request_fields(value)


async def _evaluate_one(
    env_instance: _EnvWrapper,
    eval_kwargs: Dict[str, Any],
    max_retries: int,
) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        start = time.monotonic()
        try:
            result = await env_instance.evaluate(**eval_kwargs)
            latency = time.monotonic() - start
            if hasattr(result, "model_dump"):
                out = result.model_dump()
            elif hasattr(result, "dict"):
                out = result.dict()
            elif isinstance(result, dict):
                out = dict(result)
            else:
                out = {"raw": str(result)}
            out = _sanitize_result(out)
            out["latency_seconds"] = latency
            out.setdefault("task_id", eval_kwargs.get("task_id"))
            return out
        except Exception as e:
            last_error = e
            if attempt >= max_retries or not _is_retryable(e):
                return {
                    "success": False,
                    "task_id": eval_kwargs.get("task_id"),
                    "error": f"{type(e).__name__}: evaluation_failed",
                    "latency_seconds": time.monotonic() - start,
                }
            await asyncio.sleep(min(2**attempt, 30))
    # Defensive — loop above always returns.
    error_type = type(last_error).__name__ if last_error is not None else "UnknownError"
    return {"success": False, "error": f"{error_type}: evaluation_failed"}


async def _evaluate_samples(
    env_instance: _EnvWrapper,
    base_url: str,
    model: str,
    samples: int,
    task_id: Optional[int],
    temperature: float,
    max_retries: int,
    delay: float,
    seed: Optional[int] = None,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i in range(samples):
        kwargs: Dict[str, Any] = {
            "base_url": base_url,
            "model": model,
            "temperature": temperature,
        }
        if task_id is not None:
            kwargs["task_id"] = task_id
        if seed is not None:
            kwargs["seed"] = seed
        if api_key is not None:
            kwargs["api_key"] = api_key
        click.echo(f"[{i + 1}/{samples}] evaluating...", nl=False)
        result = await _evaluate_one(env_instance, kwargs, max_retries)
        click.echo(
            f"  score={result.get('score', 0.0):.4f}  "
            f"latency={result.get('latency_seconds', 0.0):.2f}s  "
            f"ok={result.get('success', False)}"
        )
        out.append(result)
        if delay > 0 and i + 1 < samples:
            await asyncio.sleep(delay)
    return out


async def _evaluate_range(
    env_instance: _EnvWrapper,
    base_url: str,
    model: str,
    temperature: float,
    max_retries: int,
    start: int,
    end: int,
    delay: float,
    seed: Optional[int] = None,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    total = end - start
    for offset, tid in enumerate(range(start, end), start=1):
        click.echo(f"[{offset}/{total}] task_id={tid}...", nl=False)
        kwargs: Dict[str, Any] = {
            "base_url": base_url,
            "model": model,
            "temperature": temperature,
            "task_id": tid,
        }
        if seed is not None:
            kwargs["seed"] = seed
        if api_key is not None:
            kwargs["api_key"] = api_key
        result = await _evaluate_one(env_instance, kwargs, max_retries)
        click.echo(
            f"  score={result.get('score', 0.0):.4f}  "
            f"latency={result.get('latency_seconds', 0.0):.2f}s  "
            f"ok={result.get('success', False)}"
        )
        out.append(result)
        if delay > 0 and offset < total:
            await asyncio.sleep(delay)
    return out


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #


def _write_summary(
    env: str,
    base_url: str,
    model: str,
    temperature: float,
    seed: Optional[int],
    results: List[Dict[str, Any]],
    output: Optional[str],
) -> None:
    total_samples = len(results)
    total_score = sum(float(r.get("score", 0.0) or 0.0) for r in results)
    total_time = sum(float(r.get("latency_seconds", 0.0) or 0.0) for r in results)
    avg_score = total_score / total_samples if total_samples else 0.0
    avg_time = total_time / total_samples if total_samples else 0.0
    summary = {
        "environment": env,
        "base_url": base_url,
        "model": model,
        "samples": total_samples,
        "total_score": total_score,
        "average_score": avg_score,
        "total_time": total_time,
        "average_time": avg_time,
        "temperature": temperature,
        "results": results,
    }
    if seed is not None:
        summary["seed"] = seed
    if output is None:
        ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        output = f"eval/results_{ts}.json"
    try:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        click.echo(f"\n✓ Results saved to {output}")
    except Exception as e:
        click.echo(f"\n✗ Failed to save results: {e}")

    click.echo("=" * 60)
    click.echo(f"  env: {env}  model: {model}  base_url: {base_url}")
    click.echo(
        f"  samples: {total_samples}  avg_score: {avg_score:.4f}  avg_time: {avg_time:.2f}s"
    )
    click.echo("=" * 60)
