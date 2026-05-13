"""
Affine API Server

FastAPI application entry point.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from affine.api.config import config
from affine.api.middleware import setup_middleware
from affine.api.routers import (
    config_router,
    miners_router,
    rank_router,
    scores_router,
)
from affine.core.setup import logger
from affine.database import close_client, init_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown — only the DB client; the scheduler service owns
    everything else (sample collection, scheduling, weight derivation)."""
    logger.info("Starting Affine API server...")
    try:
        await init_client()
        logger.info("Database client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database client: {e}")
        raise

    yield

    logger.info("Shutting down Affine API server...")
    try:
        await close_client()
        logger.info("Database client closed")
    except Exception as e:
        logger.error(f"Error closing database client: {e}")


app = FastAPI(
    title=config.APP_NAME,
    description=config.APP_DESCRIPTION,
    version=config.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

setup_middleware(app)

# Read-only public endpoints: rank/status, miner metadata, scores/weights,
# and public validator configuration.
app.include_router(rank_router, prefix="/api/v1")
app.include_router(miners_router, prefix="/api/v1")
app.include_router(scores_router, prefix="/api/v1")
app.include_router(config_router, prefix="/api/v1")
if config.INTERNAL_ENDPOINTS_ENABLED:
    from affine.api.routers import logs_router

    app.include_router(logs_router, prefix="/api/v1")
    logger.info("Internal API endpoints enabled")


@app.get("/", tags=["health"])
async def health():
    return {"status": "ok", "service": "affine-api"}


@app.get("/api/v1/health", tags=["health"])
async def api_health():
    return await health()


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        f"Unhandled exception: {exc}",
        exc_info=True,
        extra={
            "method": request.method,
            "path": request.url.path,
            "client": request.client.host if request.client else None,
        },
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


if __name__ == "__main__":
    import uvicorn

    # Note: workers=1 for development
    # In production, can use multiple workers.

    uvicorn.run(
        "affine.api.server:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.RELOAD,
        log_level=config.LOG_LEVEL.lower(),
        workers=config.WORKERS,
    )
