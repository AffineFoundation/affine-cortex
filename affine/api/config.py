"""
API Configuration

Environment variables and settings for the API layer.
"""

import os
from typing import List


class APIConfig:
    """API configuration from environment variables."""

    # Server settings
    HOST: str = os.getenv("API_HOST", "127.0.0.1")
    PORT: int = int(os.getenv("API_PORT", "8000"))
    WORKERS: int = int(os.getenv("API_WORKERS", "1"))
    RELOAD: bool = os.getenv("API_RELOAD", "false").lower() == "true"

    # Rate limiting
    RATE_LIMIT_ENABLED: bool = (
        os.getenv("API_RATE_LIMIT_ENABLED", "false").lower() == "true"
    )
    RATE_LIMIT_READ: int = int(os.getenv("API_RATE_LIMIT_READ", "10"))  # per min

    # CORS
    CORS_ORIGINS: List[str] = [
        origin.strip()
        for origin in os.getenv("API_CORS_ORIGINS", "*").split(",")
        if origin.strip()
    ]

    # Logging
    LOG_LEVEL: str = os.getenv("API_LOG_LEVEL", "INFO")

    # Optional operator-only diagnostics. The API process no longer owns
    # scheduler/executor services; this only mounts non-public read endpoints.
    INTERNAL_ENDPOINTS_ENABLED: bool = (
        os.getenv("API_INTERNAL_ENDPOINTS_ENABLED", "false").lower() == "true"
    )

    # App metadata
    APP_NAME: str = "Affine API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "RESTful API for Affine validator infrastructure"

# Singleton instance
config = APIConfig()
