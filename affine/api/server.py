"""
Affine API Server

FastAPI application entry point.
"""

import os
import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from affine.api.config import config
from affine.api.middleware import setup_middleware
from affine.api.routers import (
    samples_router,
    tasks_router,
    miners_router,
    scores_router,
    config_router,
    logs_router,
)
from affine.database import init_client, close_client

from affine.core.setup import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    logger.info("Starting Affine API server...")
    try:
        await init_client()
        logger.info("Database client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database client: {e}")
        raise
    
    # Initialize TaskPoolManager (required for /fetch and /submit endpoints)
    if config.SERVICES_ENABLED:
        logger.info("Services mode enabled (SERVICES_ENABLED=true)")
        
        try:
            from affine.api.dependencies import get_task_pool_manager
            
            # Initialize TaskPoolManager singleton via dependency injection
            task_pool_manager = get_task_pool_manager()
            logger.info("TaskPoolManager initialized")
            
            # Start runtime timeout cleanup loop (for long-running operation)
            await task_pool_manager.start_timeout_cleanup_loop()
        except Exception as e:
            logger.error(f"Failed to initialize TaskPoolManager: {e}")
            raise
    else:
        logger.warning(
            "Services mode disabled (SERVICES_ENABLED=false). "
            "TaskPoolManager will not start. "
            "/fetch and /submit endpoints will return 503 Service Unavailable."
        )
    
    # Initialize scoring cache
    scoring_refresh_task = None
    try:
        from affine.api.services.scoring_cache import warmup_cache, refresh_cache_loop
        
        # Warmup cache on startup
        await warmup_cache()
        
        # Start background refresh task
        scoring_refresh_task = asyncio.create_task(refresh_cache_loop())
        logger.info("Scoring cache initialized with background refresh")
        
    except Exception as e:
        logger.error(f"Failed to initialize scoring cache: {e}")
        # Non-fatal: continue startup, cache will compute on first request
    
    # Initialize sampling statistics collector
    stats_collector = None
    if config.SERVICES_ENABLED:
        try:
            from affine.api.services.sampling_stats import get_stats_collector
            
            stats_collector = get_stats_collector()
            await stats_collector.start_sync_loop()
            logger.info("SamplingStatsCollector initialized with background sync")
            
        except Exception as e:
            logger.error(f"Failed to initialize SamplingStatsCollector: {e}")
            # Non-fatal: continue startup, stats will not be collected
    
    yield
    
    # Shutdown
    logger.info("Shutting down Affine API server...")
    
    # Stop sampling stats collector
    if stats_collector:
        try:
            await stats_collector.stop()
            logger.info("SamplingStatsCollector stopped")
        except Exception as e:
            logger.error(f"Error stopping SamplingStatsCollector: {e}")
    
    # Stop scoring cache refresh task
    if scoring_refresh_task:
        try:
            scoring_refresh_task.cancel()
            await scoring_refresh_task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error stopping scoring cache refresh: {e}")
    
    try:
        await close_client()
        logger.info("Database client closed")
    except Exception as e:
        logger.error(f"Error closing database client: {e}")


# Create FastAPI application
app = FastAPI(
    title=config.APP_NAME,
    description=config.APP_DESCRIPTION,
    version=config.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Setup middleware
setup_middleware(app)

# Include routers with /api/v1 prefix
app.include_router(samples_router, prefix="/api/v1")
app.include_router(tasks_router, prefix="/api/v1")
app.include_router(miners_router, prefix="/api/v1")
app.include_router(scores_router, prefix="/api/v1")
app.include_router(config_router, prefix="/api/v1")
app.include_router(logs_router, prefix="/api/v1")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error(
        f"Unhandled exception: {str(exc)}",
        exc_info=True,
        extra={
            "method": request.method,
            "path": request.url.path,
            "client": request.client.host if request.client else None,
        }
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An internal server error occurred",
                "timestamp": int(__import__("time").time()),
                "request_id": getattr(request.state, "request_id", "unknown"),
            }
        }
    )


@app.get("/")
async def root():
    """Root endpoint - redirect to docs."""
    return {
        "name": config.APP_NAME,
        "version": config.APP_VERSION,
        "docs": "/docs",
        "redoc": "/redoc",
    }


if __name__ == "__main__":
    import uvicorn
    
    # Note: workers=1 for development
    # In production, can use multiple workers (TaskPoolManager uses dependency injection)
    
    uvicorn.run(
        "affine.api.server:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.RELOAD,
        log_level=config.LOG_LEVEL.lower(),
        workers=config.WORKERS,
    )