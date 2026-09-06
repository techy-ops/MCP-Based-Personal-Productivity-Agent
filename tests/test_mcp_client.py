import os
from pathlib import Path

import pytest

from app.config import BASE_DIR
from mcp_client import MCPClient
from mcp_client.exceptions import MCPConnectionError


@pytest.fixture
def client_factory():
    def _factory(**kwargs):
        env = {**os.environ, "DATABASE_URL": "sqlite:///:memory:"}
        return MCPClient(
            server_path=BASE_DIR / "mcp_servers" / "unified_server.py",
            env=env,
            **kwargs,
        )

    return _factory


def test_client_creation(client_factory):
    client = client_factory()
    assert client is not None
    assert client.is_connected is False
    assert client.session is None


def test_client_initial_state(client_factory):
    client = client_factory()
    assert client.is_connected is False
    assert client.session is None


@pytest.mark.asyncio
async def test_successful_connection(client_factory):
    client = client_factory()
    await client.connect()
    assert client.is_connected is True
    assert client.session is not None
    await client.close()
    assert client.is_connected is False


@pytest.mark.asyncio
async def test_successful_close(client_factory):
    client = client_factory()
    await client.connect()
    await client.close()
    assert client.is_connected is False
    assert client.session is None


@pytest.mark.asyncio
async def test_repeated_close(client_factory):
    client = client_factory()
    await client.connect()
    await client.close()
    await client.close()
    assert client.is_connected is False
    assert client.session is None


@pytest.mark.asyncio
async def test_repeated_connect_is_idempotent(client_factory):
    client = client_factory()
    await client.connect()
    await client.connect()
    assert client.is_connected is True
    assert client.session is not None
    await client.close()


@pytest.mark.asyncio
async def test_connection_failure_raises_client_exception():
    client = MCPClient(server_path=BASE_DIR / "mcp_servers" / "missing_server.py")
    with pytest.raises(MCPConnectionError):
        await client.connect()


@pytest.mark.asyncio
async def test_context_manager_lifecycle(client_factory):
    client = client_factory()
    async with client:
        assert client.is_connected is True
        assert client.session is not None
    assert client.is_connected is False
    assert client.session is None
