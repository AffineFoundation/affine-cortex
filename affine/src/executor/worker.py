"""
Executor Worker - Individual Task Executor (Subprocess Mode)

Each worker runs in its own subprocess and executes tasks for a specific environment.
Environment creation and task execution happen entirely within the subprocess.
"""

import asyncio
import time
from typing import Optional, Dict, Any
import bittensor as bt
from affine.core.models import SampleSubmission
from affine.utils.api_client import create_api_client, APIClient
from affine.src.executor.metrics import WorkerMetrics
from affine.src.executor.logging_utils import safe_log


class ExecutorWorker:
    """Worker that executes sampling tasks for a specific environment in subprocess."""
    
    def __init__(
        self,
        worker_id: int,
        env: str,
        wallet: bt.Wallet,
        max_concurrent_tasks: int = 5,
        batch_size: int = 20,
    ):
        """Initialize executor worker.
        
        Args:
            worker_id: Unique worker ID
            env: Environment to execute tasks for (e.g., "affine:sat")
            wallet: Bittensor wallet for signing
            max_concurrent_tasks: Maximum number of concurrent task executions
            batch_size: Number of tasks to fetch per request
        """
        self.worker_id = worker_id
        self.env = env
        self.wallet = wallet
        self.hotkey = wallet.hotkey.ss58_address
        self.max_concurrent_tasks = max_concurrent_tasks
        self.batch_size = batch_size
        
        self.running = False
        self.metrics = WorkerMetrics(
            worker_id=worker_id,
            env=env,
        )
        
        self.api_client: Optional[APIClient] = None
        self.env_executor = None
        
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.execution_semaphore: Optional[asyncio.Semaphore] = None
        self.executor_tasks = []

    async def _init_env_executor(self):
        """Initialize environment executor in subprocess.
        
        Reuses SDKEnvironment's _load_environment logic for consistency.
        The executor mode is determined by affinetes_hosts.json configuration.
        """
        if self.env_executor is not None:
            return

        try:
            from affine.core.environments import SDKEnvironment

            self.env_executor = SDKEnvironment(self.env)
            
            # Log the mode being used
            _, execution_mode = self.env_executor._get_hosts_and_mode()
            safe_log(
                f"[{self.env}] Environment initialized using {execution_mode} mode "
                f"(via SDKEnvironment)",
                "INFO"
            )
            
        except Exception as e:
            safe_log(f"[{self.env}] Failed to initialize environment: {e}", "ERROR")
            raise
    
    async def initialize(self):
        """Initialize the worker (environment and API client)."""
        safe_log(f"[{self.env}] Initializing worker...", "INFO")

        if not self.wallet or not self.hotkey:
            raise RuntimeError("Wallet not configured for worker")
        
        self.api_client = await create_api_client()
        await self._init_env_executor()
        
        safe_log(f"[{self.env}] Worker initialized", "INFO")
    
    def start(self):
        """Start the worker fetch and execution loops."""
        self.running = True
        self.execution_semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        
        asyncio.create_task(self._fetch_loop())
        
        for i in range(self.max_concurrent_tasks):
            task = asyncio.create_task(self._execution_worker(i))
            self.executor_tasks.append(task)
        
        safe_log(
            f"[{self.env}] Started fetch loop (batch_size: {self.batch_size}) "
            f"and {self.max_concurrent_tasks} executor workers",
            "INFO"
        )
    
    async def stop(self):
        """Stop the worker and all executor tasks."""
        safe_log(f"[{self.env}] Stopping worker...", "INFO")
        self.running = False
        
        for task in self.executor_tasks:
            if not task.done():
                task.cancel()
        
        if self.executor_tasks:
            await asyncio.gather(*self.executor_tasks, return_exceptions=True)
        
        safe_log(f"[{self.env}] Worker stopped", "INFO")
    
    def _sign_message(self, message: str) -> str:
        """Sign a message using the wallet."""
        if not self.wallet:
            raise RuntimeError("Wallet not configured")
        
        signature = self.wallet.hotkey.sign(message.encode())
        return signature.hex()
    
    def _get_auth_headers(self, message: Optional[str] = None) -> Dict[str, str]:
        """Get authentication headers for API requests."""
        if message is None:
            message = str(int(time.time()))
        
        signature = self._sign_message(message)
        
        return {
            "X-Hotkey": self.hotkey,
            "X-Signature": signature,
            "X-Message": message,
        }
    
    async def _fetch_tasks_batch(self):
        """Fetch batch of tasks from API with latency tracking."""
        start_time = time.time()
        
        try:
            headers = self._get_auth_headers()
            response = await self.api_client.post(
                "/tasks/fetch",
                params={"env": self.env, "batch_size": self.batch_size},
                headers=headers
            )

            if not isinstance(response, dict):
                return []
            
            tasks = response.get("tasks", [])
            
            if not tasks:
                return []

            safe_log(
                f"[{self.env}] Fetched {len(tasks)} tasks (requested {self.batch_size})",
                "DEBUG"
            )
            return tasks

        except Exception as e:
            safe_log(f"[{self.env}] Failed to fetch tasks: {e}", "DEBUG")
            return []
        
        finally:
            fetch_time = (time.time() - start_time) * 1000
            self.metrics.fetch_count += 1
            self.metrics.total_fetch_time += fetch_time

    
    async def _execute_task(self, task: Dict) -> SampleSubmission:
        """Execute a sampling task."""
        start_time = time.time()
        
        try:
            model = task["model"]
            task_id = int(task["task_id"])
            task_uuid = task.get("task_uuid", "")
            miner_hotkey = task["miner_hotkey"]
            chute_slug = task.get("chute_slug", "")
            
            safe_log(
                f"[{self.env}] Executing task: "
                f"uuid={task_uuid[:8]}... miner={miner_hotkey[:12]}... model={model} task_id={task_id}",
                "DEBUG"
            )
            
            if not chute_slug:
                raise ValueError(
                    f"chute_slug is required but missing for task {task_uuid[:8]}... "
                    f"miner={miner_hotkey[:12]}..."
                )
            
            # Create a minimal miner object for evaluate()
            class MinimalMiner:
                def __init__(self, model, slug, hotkey):
                    self.model = model
                    self.slug = chute_slug.replace('.chutes.ai', '').replace('https://', '')
                    self.hotkey = hotkey
                    self.revision = ""
            
            miner = MinimalMiner(model, chute_slug, miner_hotkey)
            
            # Call SDKEnvironment.evaluate() which returns a Result object
            result = await self.env_executor.evaluate(
                miner=miner,
                task_id=task_id,
            )
            
            execution_time = time.time() - start_time
            self.metrics.total_execution_time += execution_time
            
            extra = result.extra or {}
            if result.error:
                extra["error"] = result.error
            
            submission = SampleSubmission(
                task_uuid=task_uuid,
                score=float(result.score),
                latency_ms=int(result.latency_seconds * 1000),
                extra=extra,
                signature="",
            )
            
            submission.sign(self.wallet)

            has_error = extra.get("error")
            if has_error:
                error_brief = str(has_error).replace('\n', ' ').replace('\r', ' ')[:300]
                safe_log(
                    f"[FAILED] U{task.get('miner_uid'):<4} │ {self.env:<20} │ {submission.score:10.3f} │ "
                    f"task_id={task_id:<6} │ {execution_time:6.3f}s │ error={error_brief}",
                    "INFO"
                )
            else:
                safe_log(
                    f"[RESULT] U{task.get('miner_uid'):<4} │ {self.env:<20} │ {submission.score:10.3f} │ "
                    f"task_id={task_id:<6} │ {execution_time:6.3f}s",
                    "INFO"
                )
            
            return submission
        
        except Exception:
            raise
    
    async def _submit_result(self, task: Dict, submission: SampleSubmission) -> bool:
        """Submit task result to API with authentication."""
        try:
            headers = self._get_auth_headers()
            
            submit_data = {
                "task_uuid": submission.task_uuid,
                "score": submission.score,
                "latency_ms": submission.latency_ms,
                "extra": submission.extra,
                "signature": submission.signature,
            }
            
            await self.api_client.post(
                "/tasks/submit",
                json=submit_data,
                headers=headers
            )
            
            has_error = submission.extra.get("error") if submission.extra else None
            if has_error:
                self.metrics.tasks_failed += 1
            else:
                self.metrics.tasks_succeeded += 1

            return True

        except Exception:
            self.metrics.submit_failed += 1
            raise
    
    async def _fetch_loop(self):
        """Fetch loop driven by queue size."""
        while self.running:
            try:
                current_queue_size = self.task_queue.qsize()
                
                if current_queue_size >= self.max_concurrent_tasks:
                    await asyncio.sleep(1)
                    continue

                tasks = await self._fetch_tasks_batch()
                
                num_tasks = len(tasks)
                if num_tasks > 0:
                    for task in tasks:
                        await self.task_queue.put(task)
                    
                    safe_log(
                        f"[{self.env}] Queued {num_tasks} tasks (queue_size={self.task_queue.qsize()})",
                        "DEBUG"
                    )
                else:
                    await asyncio.sleep(1)
            
            except asyncio.CancelledError:
                break
            
            except Exception as e:
                safe_log(f"[{self.env}] Error in fetch loop: {e}", "ERROR")
                await asyncio.sleep(1)
        
        safe_log(f"[{self.env}] Fetch loop stopped", "INFO")
    
    async def _execution_worker(self, worker_idx: int):
        """Execution worker that processes tasks from queue concurrently."""
        safe_log(f"[{self.env}] Execution worker {worker_idx} started", "DEBUG")
        
        while self.running:
            try:
                try:
                    task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                async with self.execution_semaphore:
                    safe_log(
                        f"[{self.env}] Worker {worker_idx} executing task "
                        f"uuid={task.get('task_uuid', 'unknown')[:8]}...",
                        "DEBUG"
                    )
                    
                    task_start_time = time.time()
                    submission = None
                    execution_error = None
                    
                    try:
                        submission = await self._execute_task(task)
                    except Exception as e:
                        execution_error = e
                        execution_time = time.time() - task_start_time
                        
                        # Create a failed submission with error details
                        error_brief = str(e).replace('\n', ' ').replace('\r', ' ')[:300]
                        safe_log(
                            f"[FAILED] U{task.get('miner_uid'):<4} │ {self.env:<20} │     FAILED │ "
                            f"task_id={task.get('task_id', 'N/A'):<6} │ {execution_time:6.3f}s │ {error_brief}",
                            "INFO"
                        )
                        
                        # Construct a failed submission
                        submission = SampleSubmission(
                            task_uuid=task.get('task_uuid', ''),
                            score=0.0,
                            latency_ms=int(execution_time * 1000),
                            extra={"error": str(e)},
                            signature="",
                        )
                        
                        submission.sign(self.wallet)
                    
                    # Always submit result if we have a submission
                    try:
                        if submission:
                            await self._submit_result(task, submission)
                    except Exception as submit_error:
                        safe_log(
                            f"[{self.env}] Failed to submit result for task {task.get('task_uuid', 'unknown')[:8]}...: {submit_error}",
                            "ERROR"
                        )
                    finally:
                        self.task_queue.task_done()
                        self.metrics.last_task_at = time.time()
            
            except asyncio.CancelledError:
                break
            
            except Exception as e:
                safe_log(f"[{self.env}] Error in execution worker {worker_idx}: {e}", "ERROR")
                await asyncio.sleep(1)
        
        safe_log(f"[{self.env}] Execution worker {worker_idx} stopped", "DEBUG")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get worker metrics."""
        total_tasks = self.metrics.tasks_succeeded + self.metrics.tasks_failed
        avg_time = (
            self.metrics.total_execution_time / total_tasks
            if total_tasks > 0
            else 0
        )
        
        avg_fetch_time = (
            self.metrics.total_fetch_time / self.metrics.fetch_count
            if self.metrics.fetch_count > 0
            else 0
        )
        
        running_tasks = 0
        if self.execution_semaphore is not None:
            running_tasks = self.max_concurrent_tasks - self.execution_semaphore._value
        
        pending_tasks = self.task_queue.qsize()
        
        self.metrics.running_tasks = running_tasks
        self.metrics.pending_tasks = pending_tasks
        
        metrics_dict = self.metrics.to_dict()
        metrics_dict['avg_execution_time'] = avg_time
        metrics_dict['avg_fetch_time_ms'] = avg_fetch_time
        
        return metrics_dict