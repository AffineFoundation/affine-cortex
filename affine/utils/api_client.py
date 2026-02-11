"""
API Client Utility

Provides a reusable HTTP client for making API requests to the Affine API server.
Handles common patterns like error handling, JSON response parsing, and logging.
"""

import json
import sys
import os
from typing import Optional, Dict, Any
from affine.core.setup import logger
import aiohttp
import asyncio
from affine.utils.errors import NetworkError, ApiResponseError


class GlobalSessionManager:
    """Singleton manager for shared aiohttp ClientSession across all workers.
    
    This ensures all HTTP requests share a single connection pool, minimizing
    file descriptor usage and improving performance.
    """
    
    _instance: Optional['GlobalSessionManager'] = None
    _lock: asyncio.Lock = asyncio.Lock()
    _session: Optional[aiohttp.ClientSession] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    async def get_session(cls) -> aiohttp.ClientSession:
        """Get or create the global shared session.
        
        Returns:
            Shared ClientSession instance
        """
        async with cls._lock:
            if cls._session is None or cls._session.closed:
                # Configure connector for shared connection pool
                # Need 8 workers × 60 concurrent tasks = 480 potential connections
                connector = aiohttp.TCPConnector(
                    limit=1000,  # Increased to handle 8 workers × 60 tasks
                    limit_per_host=0,  # No per-host limit (use total limit)
                    ttl_dns_cache=300,  # DNS cache TTL
                    force_close=False,  # Allow connection reuse
                    enable_cleanup_closed=True,  # Clean up closed connections
                    keepalive_timeout=30,  # Close idle connections after 30s (prevent stale connections)
                )
                
                # Connection timeout with safety limits
                timeout = aiohttp.ClientTimeout(
                    total=300,
                    connect=60,  # 60s connection timeout (wait for available connection)
                    sock_read=None
                )
                
                cls._session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout,
                    connector_owner=True  # Ensure connector is closed with session
                )
                
            
            return cls._session
    
    @classmethod
    async def close(cls):
        """Close the global shared session."""
        async with cls._lock:
            if cls._session and not cls._session.closed:
                await cls._session.close()
                cls._session = None
                logger.info("GlobalSessionManager: Closed shared session")

class CLIAPIClient:
    """CLI-specific API client context manager.
    
    Creates an independent session for one-time CLI commands,
    automatically closing it when done. This is separate from
    long-running services that use GlobalSessionManager.
    
    Usage:
        async with cli_api_client() as client:
            data = await client.get("/miners/uid/42")
    """
    
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or os.getenv("API_URL", "https://api.affine.io/api/v1")
        self._session: Optional[aiohttp.ClientSession] = None
        self._client: Optional['APIClient'] = None
    
    async def __aenter__(self) -> 'APIClient':
        """Enter context: create independent session and client"""
        connector = aiohttp.TCPConnector(
            limit=20,
            limit_per_host=0,
            force_close=False,
            keepalive_timeout=60,
            enable_cleanup_closed=True,
        )
        
        timeout = aiohttp.ClientTimeout(
            total=300,
            connect=30,  # connection timeout
            sock_read=None
        )
        
        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            connector_owner=True
        )
        
        self._client = APIClient(self.base_url, self._session)
        return self._client
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit context: close session"""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.debug("CLIAPIClient: Closed independent session")
        return False  # Don't suppress exceptions


class APIClient:
    """HTTP client for Affine API requests.
    
    Uses GlobalSessionManager's shared connection pool for all requests.
    """
    
    def __init__(self, base_url: str, session: aiohttp.ClientSession):
        """Initialize API client.
        
        Args:
            base_url: Base URL for API (e.g., "http://localhost:8000/api/v1")
            session: Shared ClientSession from GlobalSessionManager
        """
        self.base_url = base_url.rstrip("/")
        self._session = session
    
    async def close(self):
        """No-op: Session is managed by GlobalSessionManager."""
        pass
    
    async def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        """Make GET request to API endpoint.
        
        Args:
            endpoint: API endpoint path (e.g., "/miners/uid/123")
            params: Optional query parameters
            headers: Optional request headers
        
        Returns:
            Response data dict on success
            
        Raises:
            NetworkError: On network/connection errors
            ApiResponseError: On non-2xx response or invalid JSON
        """
        
        url = f"{self.base_url}{endpoint}"
        logger.debug(f"GET {url}")

        try:
            async with self._session.get(url, params=params, headers=headers) as response:
                if response.status >= 400:
                    body = await response.text()
                    raise ApiResponseError(f"HTTP {response.status}: {body[:200]}", response.status, url, body)
                
                try:
                    return await response.json()
                except Exception:
                    raw = await response.text()
                    raise ApiResponseError(f"Invalid JSON response: {raw[:200]}", response.status, url, raw)
                    
        except aiohttp.ClientError as e:
            raise NetworkError(f"Network error during GET {url}: {e}", url, e)

    
    async def post(
        self,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        output_json: bool = False,
    ) -> Any:
        """Make POST request to API endpoint.
        
        Args:
            endpoint: API endpoint path
            json: Request JSON payload
            params: Optional query parameters
            headers: Optional request headers
            output_json: Whether to print JSON response to stdout
        
        Returns:
            Response data dict on success
            
        Raises:
            NetworkError: On network/connection errors
            ApiResponseError: On non-2xx response or invalid JSON
        """
        
        url = f"{self.base_url}{endpoint}"
        logger.debug(f"POST {url}")
        
        try:
            async with self._session.post(url, json=json, params=params, headers=headers) as response:
                if response.status >= 400:
                    body = await response.text()
                    # Try to parse JSON error details if possible
                    try:
                        import json as json_lib
                        error_json = json_lib.loads(body)
                        msg = error_json.get("detail", str(error_json))
                    except Exception:
                        msg = body[:200]
                    
                    if output_json:
                         print(f'{{"success": false, "error": "{msg}"}}')
                    
                    raise ApiResponseError(f"HTTP {response.status}: {msg}", response.status, url, body)
                
                try:
                    data = await response.json()
                    if output_json:
                        import json as json_lib
                        print(json_lib.dumps({"success": True, "data": data}, indent=2, ensure_ascii=False))
                    return data
                except Exception:
                    raw = await response.text()
                    if output_json:
                        print(f'{{"success": false, "error": "Invalid JSON"}}')
                    raise ApiResponseError(f"Invalid JSON response: {raw[:200]}", response.status, url, raw)

        except aiohttp.ClientError as e:
            if output_json:
                 print(f'{{"success": false, "error": "{str(e)}"}}')
            raise NetworkError(f"Network error during POST {url}: {e}", url, e)


    async def put(
        self,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        """Make PUT request to API endpoint.
        
        Args:
            endpoint: API endpoint path
            json: Request JSON payload
            params: Optional query parameters
            headers: Optional request headers
        
        Returns:
            Response data dict on success, raises exception on error
        """
        
        url = f"{self.base_url}{endpoint}"
        logger.debug(f"PUT {url}")

        try:
            async with self._session.put(url, json=json, params=params, headers=headers) as response:
                if response.status >= 400:
                    body = await response.text()
                    raise ApiResponseError(f"HTTP {response.status}: {body[:200]}", response.status, url, body)
                
                if response.status == 204:
                    return {}
                
                try:
                    return await response.json()
                except Exception:
                    raw = await response.text()
                    raise ApiResponseError(f"Invalid JSON response: {raw[:200]}", response.status, url, raw)

        except aiohttp.ClientError as e:
            raise NetworkError(f"Network error during PUT {url}: {e}", url, e)
        

    async def get_chute_info(self, chute_id: str) -> Optional[Dict]:
        """Get chute info from Chutes API.
        
        Args:
            chute_id: Chute deployment ID
            
        Returns:
            Chute info dict or None if failed
        """
        url = f"https://api.chutes.ai/chutes/{chute_id}"
        token = os.getenv("CHUTES_API_KEY", "")
        
        if not token:
            logger.warning("CHUTES_API_KEY not configured")
            return None
        
        headers = {"Authorization": token}
        
        try:
            async with self._session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    return None
                
                info = await resp.json()
                # Remove unnecessary fields
                for k in ("readme", "cords", "tagline", "instances"):
                    info.pop(k, None)
                info.get("image", {}).pop("readme", None)
                
                return info
        except Exception as e:
            logger.debug(f"Failed to fetch chute {chute_id}: {e}")
            return None


def cli_api_client(base_url: Optional[str] = None) -> CLIAPIClient:
    """Create CLI-specific API client context manager.
    
    Args:
        base_url: Custom base URL (optional)
    
    Returns:
        CLIAPIClient context manager
        
    Example:
        async with cli_api_client() as client:
            data = await client.get("/miners/uid/42")
            print(json.dumps(data, indent=2))
    """
    return CLIAPIClient(base_url)


async def get_chute_info(chute_id: str) -> Optional[Dict]:
    """Legacy function for backward compatibility.
    
    Creates a temporary APIClient to fetch chute info.
    """
    async with cli_api_client() as client:
        return await client.get_chute_info(chute_id)


async def create_api_client(base_url: Optional[str] = None) -> APIClient:
    """Create API client with GlobalSessionManager's shared connection pool.
    
    Args:
        base_url: Custom base URL (optional, defaults to env or localhost)
    
    Returns:
        Configured APIClient instance using shared session
    """
    import os
    
    if base_url is None:
        base_url = os.getenv("API_URL", "https://api.affine.io/api/v1")
    
    # Always use GlobalSessionManager
    session = await GlobalSessionManager.get_session()
    return APIClient(base_url, session)