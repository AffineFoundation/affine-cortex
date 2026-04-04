#!/usr/bin/env python3
"""
Executor Main Entry Point - Multiprocess Architecture

Runs executor workers in separate subprocesses for better isolation and parallelism.
Each environment gets its own subprocess with independent event loop and resources.
"""

import signal
import asyncio
import os
import click
import time
import multiprocessing
import queue
from typing import List, Dict, Any, Optional
import bittensor as bt

from affine.core.setup import logger, setup_logging
from affine.utils.api_client import create_api_client
from affine.src.executor.worker_process import WorkerProcess
from affine.src.executor.config import get_max_concurrent


def _format_change(value: int) -> str:
    """Format change value with +/- prefix, or empty string if zero."""
    if value > 0:
        return f"+{value}"
    elif value < 0:
        return str(value)
    return ""


def _format_env_queue(env_name: str, queue_count: int, change: int) -> str:
    """Format environment queue with optional change indicator."""
    env_short = env_name.split(':')[-1]
    change_str = _format_change(change)
    return f"{env_short}={queue_count}({change_str})" if change_str else f"{env_short}={queue_count}"


def _format_env_stats(
    env_name: str,
    completed: int,
    success_rate: int,
    change: int,
    running: int,
    pending: int,
    fetch_avg_ms: float,
    submit_failed_change: int = 0,
    rate_per_hour: float = 0.0
) -> str:
    """Format environment completion stats."""
    env_short = env_name.split(':')[-1]
    change_str = f" finished:{_format_change(change)}" if change else " finished:0"
    submit_fail_str = f" submit_failed:{_format_change(submit_failed_change)}" if submit_failed_change else ""
    rate_str = f" rate:{rate_per_hour:.0f}/h" if rate_per_hour > 0 else ""
    return f"{env_short}@{completed}({success_rate}%{change_str}{submit_fail_str}{rate_str} running:{running} pending:{pending} fetch_avg:{fetch_avg_ms:.0f}ms)"


class ExecutorManager:
    """Main process manager for executor workers."""
    
    def __init__(
        self,
        envs: List[str],
        verbosity: int = 1,
    ):
        self.envs = envs
        self.verbosity = verbosity
        
        # IPC queue for stats
        self.stats_queue = multiprocessing.Queue()
        
        # Worker processes
        self.worker_processes: List[WorkerProcess] = []
        self.running = False
        
        # Stats tracking
        self.aggregated_stats: Dict[str, Dict] = {}
        self.last_status_time = None
        self.last_queue_stats = {}
        self.last_completed_stats = {}
        self.last_submit_failed_stats = {}
        
        # Wallet and API client (main process only)
        self.wallet = None
        self.api_client = None
        
        # Log environment concurrency configuration
        env_config_str = ", ".join([f"{env}={get_max_concurrent(env)}" for env in envs])
        logger.info(
            f"ExecutorManager initialized for {len(envs)} environments "
            f"({env_config_str})"
        )
    
    async def start(self):
        """Start all worker processes."""
        if self.running:
            logger.warning("ExecutorManager already running")
            return
        
        logger.info("Starting ExecutorManager...")
        
        # Load wallet
        coldkey = os.getenv("BT_WALLET_COLD", "default")
        hotkey = os.getenv("BT_WALLET_HOT", "default")
        self.wallet = bt.Wallet(name=coldkey, hotkey=hotkey)
        
        if not self.wallet:
            logger.error("No wallet configured. Set BT_WALLET_COLD and BT_WALLET_HOT environment variables.")
            raise RuntimeError("Wallet not configured")
        
        logger.info(f"Using wallet hotkey: {self.wallet.hotkey.ss58_address[:16]}...")
        
        # Create API client
        self.api_client = await create_api_client()
        
        # Create and start worker processes
        for idx, env in enumerate(self.envs):
            # Get max concurrent tasks for this specific environment
            max_concurrent = get_max_concurrent(env)
            worker_proc = WorkerProcess(
                worker_id=idx,
                env=env,
                wallet_name=coldkey,
                wallet_hotkey=hotkey,
                max_concurrent_tasks=max_concurrent,
                batch_size=20,
                stats_queue=self.stats_queue,
                verbosity=self.verbosity,
            )
            worker_proc.start()
            self.worker_processes.append(worker_proc)
        
        self.running = True

        # Start background tasks
        asyncio.create_task(self._stats_collector())
        asyncio.create_task(self._health_checker())

        logger.info(f"Started {len(self.worker_processes)} worker processes")
    
    async def _stats_collector(self):
        """Collect stats from worker processes."""
        while self.running:
            try:
                try:
                    stats = self.stats_queue.get_nowait()
                    env = stats['env']
                    self.aggregated_stats[env] = stats
                except queue.Empty:
                    pass
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Stats collector error: {e}")
                await asyncio.sleep(1)
    
    async def _health_checker(self):
        """Check worker process health and restart if needed."""
        while self.running:
            try:
                for worker_proc in self.worker_processes:
                    if not worker_proc.is_alive():
                        logger.warning(
                            f"Worker {worker_proc.env} died (PID: {worker_proc.process.pid}), "
                            "restarting..."
                        )
                        worker_proc.start()

                await asyncio.sleep(10)

            except Exception as e:
                logger.error(f"Health checker error: {e}")
                await asyncio.sleep(1)
    
    async def stop(self):
        """Stop all worker processes."""
        if not self.running:
            return
        
        logger.info("Stopping ExecutorManager...")
        self.running = False
        
        for worker_proc in self.worker_processes:
            worker_proc.terminate()
        
        if self.api_client:
            await self.api_client.close()
        
        logger.info("ExecutorManager stopped")
    
    async def wait(self):
        """Wait for shutdown signal."""
        while self.running:
            await asyncio.sleep(1)
    
    def get_all_metrics(self):
        """Get aggregated metrics from all workers."""
        return list(self.aggregated_stats.values())
    
    async def _fetch_queue_stats(self, metrics):
        """Fetch queue statistics from API for all environments."""
        async def fetch_env_stats(env: str):
            try:
                stats_response = await self.api_client.get(f"/tasks/pool/stats?env={env}")
                count = stats_response.get('pending_count', 0) if isinstance(stats_response, dict) else 0
                return (env, count)
            except Exception as e:
                logger.debug(f"Failed to fetch stats for {env}: {e}")
                return (env, 0)
        
        tasks = [fetch_env_stats(m['env']) for m in metrics]
        results = await asyncio.gather(*tasks)
        
        return {env: count for env, count in results}
    
    async def print_status(self):
        """Print status of all workers."""
        try:
            current_time = time.time()
            time_delta = int(current_time - self.last_status_time) if self.last_status_time else 0
            metrics = self.get_all_metrics()
            
            if not metrics:
                logger.info("[STATUS] No worker metrics available yet")
                return
            
            queue_stats = await self._fetch_queue_stats(metrics)
            total_queue = sum(queue_stats.values())
            
            env_queue_changes = {
                env: current - self.last_queue_stats.get(env, current)
                for env, current in queue_stats.items()
            }
            total_queue_change = sum(env_queue_changes.values())
            
            env_stats = {}
            for m in metrics:
                env_name = m['env']
                succeeded = m['tasks_succeeded']
                failed = m['tasks_failed']
                submit_failed = m['submit_failed']
                total = succeeded + failed
                success_rate = int(succeeded * 100 / total) if total > 0 else 0
                succeeded_change = succeeded - self.last_completed_stats.get(env_name, succeeded)
                submit_failed_change = submit_failed - self.last_submit_failed_stats.get(env_name, submit_failed)
                
                rate_per_hour = 0.0
                if time_delta > 0 and succeeded_change > 0:
                    rate_per_hour = (succeeded_change / time_delta) * 3600
                
                env_stats[env_name] = {
                    'succeeded': succeeded,
                    'failed': failed,
                    'submit_failed': submit_failed,
                    'success_rate': success_rate,
                    'queue': queue_stats.get(env_name, 0),
                    'queue_change': env_queue_changes.get(env_name, 0),
                    'succeeded_change': succeeded_change,
                    'submit_failed_change': submit_failed_change,
                    'rate_per_hour': rate_per_hour,
                    'running_tasks': m.get('running_tasks', 0),
                    'pending_tasks': m.get('pending_tasks', 0),
                    'fetch_avg_ms': m.get('avg_fetch_time_ms', 0),
                }
            
            queue_details = " ".join(
                _format_env_queue(env, stats['queue'], stats['queue_change'])
                for env, stats in sorted(env_stats.items())
            )
            
            env_stats_str = " ".join(
                _format_env_stats(
                    env,
                    stats['succeeded'],
                    stats['success_rate'],
                    stats['succeeded_change'],
                    stats['running_tasks'],
                    stats['pending_tasks'],
                    stats['fetch_avg_ms'],
                    stats['submit_failed_change'],
                    stats['rate_per_hour']
                )
                for env, stats in sorted(env_stats.items())
            )
            
            total_change_str = f"{_format_change(total_queue_change)}" if total_queue_change else ""
            
            logger.info(
                f"[STATUS] total_queue={total_queue}({total_change_str} in {time_delta}s) "
                f"({queue_details}) [{env_stats_str}]"
            )
            
            self.last_status_time = current_time
            self.last_queue_stats = queue_stats.copy()
            self.last_completed_stats = {m['env']: m['tasks_succeeded'] for m in metrics}
            self.last_submit_failed_stats = {m['env']: m['submit_failed'] for m in metrics}
            
        except Exception as e:
            logger.error(f"Error printing status: {e}", exc_info=True)
            metrics = self.get_all_metrics()
            logger.info(
                f"[STATUS] workers={len(metrics)} "
                f"succeeded={sum(m['tasks_succeeded'] for m in metrics)} "
                f"failed={sum(m['tasks_failed'] for m in metrics)} "
                f"submit_failed={sum(m['submit_failed'] for m in metrics)}"
            )


async def fetch_system_config() -> dict:
    """Fetch system configuration from API."""
    api_client = await create_api_client()
    config = await api_client.get("/config/environments")

    if isinstance(config, dict):
        value = config.get("param_value")
        if isinstance(value, dict):
            enabled_envs = [
                env_name for env_name, env_config in value.items()
                if isinstance(env_config, dict) and env_config.get("enabled_for_sampling", False)
            ]
            
            if enabled_envs:
                logger.info(f"Fetched environments from API: {enabled_envs}")
                return {"environments": enabled_envs}

    raise ValueError("Invalid or empty environments config from API")


async def run_service(
    envs: List[str] | None,
    verbosity: int,
):
    """Run the executor service in continuous mode."""
    
    if not envs:
        logger.info("No environments specified, fetching from API system config...")
        system_config = await fetch_system_config()
        envs = system_config.get("environments")
    else:
        logger.info(f"Using specified environments: {envs}")
    
    manager = ExecutorManager(
        envs=envs,
        verbosity=verbosity,
    )
    
    shutdown_event = asyncio.Event()
    
    def handle_shutdown(sig):
        logger.info(f"Received signal {sig}, initiating shutdown...")
        shutdown_event.set()
    
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: handle_shutdown(s))
    
    try:
        await manager.start()
        
        logger.info("Running in service mode (continuous). Press Ctrl+C to stop.")
        
        while not shutdown_event.is_set():
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=60)
            except asyncio.TimeoutError:
                await manager.print_status()
        
        await manager.print_status()
        
    except Exception as e:
        logger.error(f"Error running executor: {e}", exc_info=True)
        raise
    finally:
        await manager.stop()


@click.command()
@click.option(
    "--envs",
    multiple=True,
    help="Environments to execute tasks for (e.g., affine:sat). If not specified, fetches from API system config"
)
@click.option(
    "-v", "--verbosity",
    default=None,
    type=click.Choice(["0", "1", "2", "3"]),
    help="Logging verbosity: 0=CRITICAL, 1=INFO, 2=DEBUG, 3=TRACE"
)
def main(envs, verbosity):
    """
    Affine Executor - Execute sampling tasks for multiple environments.
    
    Each environment runs in its own subprocess with independent resources.
    Each environment has its own max concurrent tasks configuration (see affine/src/executor/config.py).
    Runs continuously in service mode.
    
    If --envs not specified, environments are fetched from API /api/v1/config/environments endpoint.
    """
    verbosity_val = int(verbosity) if verbosity is not None else 1
    setup_logging(verbosity_val, component="executor")
    
    selected_envs = list(envs) if envs else None
    
    # Set multiprocessing start method to 'spawn' for compatibility
    multiprocessing.set_start_method('spawn', force=True)
    
    asyncio.run(run_service(
        envs=selected_envs,
        verbosity=verbosity_val,
    ))


if __name__ == "__main__":
    main()
