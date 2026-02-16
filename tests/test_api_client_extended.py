"""Extended tests for APIClient - DELETE method and edge cases."""

import pytest
import aiohttp
from unittest.mock import MagicMock, AsyncMock
from affine.utils.api_client import APIClient
from affine.utils.errors import NetworkError, ApiResponseError


class TestAPIClientDelete:
    """Test the DELETE method on APIClient."""

    @pytest.mark.asyncio
    async def test_delete_success_json(self):
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {"deleted": True}

        mock_session = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__aenter__.return_value = mock_response
        mock_session.delete.return_value = mock_ctx

        client = APIClient("http://test.com", mock_session)
        result = await client.delete("/items/123")
        assert result == {"deleted": True}

    @pytest.mark.asyncio
    async def test_delete_204_no_content(self):
        mock_response = AsyncMock()
        mock_response.status = 204

        mock_session = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__aenter__.return_value = mock_response
        mock_session.delete.return_value = mock_ctx

        client = APIClient("http://test.com", mock_session)
        result = await client.delete("/items/123")
        assert result == {}

    @pytest.mark.asyncio
    async def test_delete_404(self):
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_response.text.return_value = "Not Found"

        mock_session = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__aenter__.return_value = mock_response
        mock_session.delete.return_value = mock_ctx

        client = APIClient("http://test.com", mock_session)
        with pytest.raises(ApiResponseError) as exc:
            await client.delete("/items/999")
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_network_error(self):
        mock_session = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__aenter__.side_effect = aiohttp.ClientConnectionError("Connection refused")
        mock_session.delete.return_value = mock_ctx

        client = APIClient("http://test.com", mock_session)
        with pytest.raises(NetworkError):
            await client.delete("/items/1")


class TestAPIClientPut:
    """Test PUT method edge cases."""

    @pytest.mark.asyncio
    async def test_put_204_no_content(self):
        mock_response = AsyncMock()
        mock_response.status = 204

        mock_session = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__aenter__.return_value = mock_response
        mock_session.put.return_value = mock_ctx

        client = APIClient("http://test.com", mock_session)
        result = await client.put("/items/1", json={"name": "updated"})
        assert result == {}

    @pytest.mark.asyncio
    async def test_put_bad_json_response(self):
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.side_effect = ValueError("Bad JSON")
        mock_response.text.return_value = "not json"

        mock_session = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__aenter__.return_value = mock_response
        mock_session.put.return_value = mock_ctx

        client = APIClient("http://test.com", mock_session)
        with pytest.raises(ApiResponseError) as exc:
            await client.put("/items/1")
        assert "Invalid JSON" in str(exc.value)


class TestAPIClientPost:
    """Test POST method edge cases."""

    @pytest.mark.asyncio
    async def test_post_json_error_parsing(self):
        """POST with output_json=True and a JSON-formatted error body."""
        mock_response = AsyncMock()
        mock_response.status = 422
        mock_response.text.return_value = '{"detail": "Validation failed"}'

        mock_session = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__aenter__.return_value = mock_response
        mock_session.post.return_value = mock_ctx

        client = APIClient("http://test.com", mock_session)
        with pytest.raises(ApiResponseError) as exc:
            await client.post("/submit", output_json=True)
        assert exc.value.status_code == 422
