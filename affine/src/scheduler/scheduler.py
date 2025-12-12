"""
Background Scheduler Service

Periodically executes task generation and cleanup operations.
"""

import asyncio
import logging
from typing import Optional

from .task_generator import TaskGeneratorService, MinerInfo
from affine.database.dao.miners import MinersDAO

from affine.core.setup import logger


class SchedulerService:
    """
    Background scheduler for periodic task generation.
    
    Responsibilities:
    1. Periodically fetch active miners
    2. Generate missing tasks for all miners
    3. Clean up invalid tasks
    """
    
    def __init__(
        self,
        task_generator: TaskGeneratorService,
        task_generation_interval: int = 300,
        cleanup_interval: int = 300,
        max_tasks_per_miner_env: int = 10,
        assigned_task_timeout: int = 600
    ):
        """
        Initialize SchedulerService.
        
        Args:
            task_generator: TaskGeneratorService instance
            task_generation_interval: Seconds between task generation runs
            cleanup_interval: Seconds between cleanup runs (includes timeout check)
            max_tasks_per_miner_env: Max tasks per miner/env per run
            assigned_task_timeout: Timeout for assigned tasks in seconds (default: 600 = 10 minutes)
        """
        self.task_generator = task_generator
        self.task_generation_interval = task_generation_interval
        self.cleanup_interval = cleanup_interval
        self.max_tasks_per_miner_env = max_tasks_per_miner_env
        self.assigned_task_timeout = assigned_task_timeout
        
        self._running = False
        self._task_generation_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the scheduler background tasks."""
        if self._running:
            logger.warning("Scheduler is already running")
            return
        
        self._running = True
        
        logger.info(
            f"Starting scheduler: "
            f"task_generation_interval={self.task_generation_interval}s, "
            f"cleanup_interval={self.cleanup_interval}s, "
            f"assigned_task_timeout={self.assigned_task_timeout}s"
        )
        
        self._task_generation_task = asyncio.create_task(
            self._task_generation_loop()
        )
        self._cleanup_task = asyncio.create_task(
            self._cleanup_loop()
        )
    
    async def stop(self):
        """Stop the scheduler background tasks."""
        if not self._running:
            return
        
        self._running = False
        
        if self._task_generation_task:
            self._task_generation_task.cancel()
            try:
                await self._task_generation_task
            except asyncio.CancelledError:
                pass
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Scheduler stopped")
    
    async def _fetch_active_miners(self) -> list[MinerInfo]:
        """Fetch active miners from database."""
        dao = MinersDAO()
        
        # Get valid miners from database
        miners_data = await dao.get_valid_miners()

        # Convert to MinerInfo list
        result = [
            MinerInfo(
                hotkey=miner['hotkey'],
                model_revision=miner['revision'],
                model=miner['model'],
                uid=miner['uid'],
                chute_id=miner['chute_id']
            )
            for miner in miners_data
        ]
        
        return result
    
    async def _task_generation_loop(self):
        """Background loop for task generation."""
        logger.info("Task generation loop started")
        
        while self._running:
            try:
                # Fetch active miners
                miners = await self._fetch_active_miners()
                logger.info(f"Fetched {len(miners)} active miners")
                
                if miners:
                    # Generate tasks
                    result = await self.task_generator.generate_all_tasks(
                        miners=miners,
                        max_tasks_per_miner_env=self.max_tasks_per_miner_env
                    )
                    
                    logger.info(
                        f"Task generation complete: "
                        f"created {result.total_tasks_created} tasks, "
                        f"{len(result.errors)} errors"
                    )
                else:
                    logger.warning("No active miners found")
                
            except Exception as e:
                logger.error(f"Task generation loop error: {e}")
            
            # Wait for next interval
            await asyncio.sleep(self.task_generation_interval)
    
    async def _cleanup_loop(self):
        """Background loop for cleanup operations and timeout check."""
        from affine.database.dao.task_pool import TaskPoolDAO
        
        logger.info("Cleanup loop started")
        
        while self._running:
            try:
                # Fetch active miners
                miners = await self._fetch_active_miners()
                
                if miners:
                    # Clean up invalid tasks
                    await self.task_generator.cleanup_invalid_tasks(miners)
                else:
                    logger.warning("No active miners found for cleanup")
                
                # Clean up expired paused tasks
                task_pool_dao = TaskPoolDAO()
                await task_pool_dao.cleanup_expired_paused_tasks()
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}", exc_info=True)
            
            # Wait for next interval
            await asyncio.sleep(self.cleanup_interval)
    
    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running


# Global scheduler instance
_scheduler: Optional[SchedulerService] = None


def get_scheduler() -> Optional[SchedulerService]:
    """Get the global scheduler instance."""
    return _scheduler


def create_scheduler(
    task_generator: TaskGeneratorService,
    task_generation_interval: int = 300,
    cleanup_interval: int = 3600,
    max_tasks_per_miner_env: int = 100,
    assigned_task_timeout: int = 600
) -> SchedulerService:
    """Create and set the global scheduler instance."""
    global _scheduler
    
    _scheduler = SchedulerService(
        task_generator=task_generator,
        task_generation_interval=task_generation_interval,
        cleanup_interval=cleanup_interval,
        max_tasks_per_miner_env=max_tasks_per_miner_env,
        assigned_task_timeout=assigned_task_timeout
    )
    
    return _scheduler