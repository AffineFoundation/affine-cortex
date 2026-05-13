"""
API Routers

Route definitions for all API endpoints.
"""

from affine.api.routers.config import router as config_router
from affine.api.routers.logs import router as logs_router
from affine.api.routers.miners import router as miners_router
from affine.api.routers.scores import router as scores_router
from affine.api.routers.windows import router as windows_router

__all__ = [
    "config_router",
    "logs_router",
    "miners_router",
    "scores_router",
    "windows_router",
]
