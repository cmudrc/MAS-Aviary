"""MCP connector — discovers and loads tools from real MCP servers.

Uses smolagents.ToolCollection.from_mcp() to connect to MCP servers
and discover available tools via the Model Context Protocol.
"""

from smolagents import Tool, ToolCollection

from src.config.loader import MCPConfig, MCPServerConfig


class MCPConnector:
    """Manages connections to one or more MCP servers and exposes their tools."""

    def __init__(self, config: MCPConfig):
        self._config = config
        self._collections: list = []  # active context managers
        self._tools: list[Tool] = []

    def connect(self) -> list[Tool]:
        """Connect to all configured MCP servers and return merged tool list.

        Each server connection is a context manager. Call disconnect() to
        cleanly close all connections when done.
        """
        self._tools = []
        for server in self._config.servers:
            tools = self._connect_server(server)
            self._tools.extend(tools)
        return list(self._tools)

    def _connect_server(self, server: MCPServerConfig) -> list[Tool]:
        """Connect to a single MCP server and return its tools."""
        mcp_config = {
            "url": server.url,
            "transport": server.transport,
        }
        collection_cm = ToolCollection.from_mcp(
            mcp_config, trust_remote_code=True,
        )
        collection = collection_cm.__enter__()
        self._collections.append(collection_cm)
        return list(collection.tools)

    def disconnect(self) -> None:
        """Close all MCP server connections."""
        for cm in self._collections:
            try:
                cm.__exit__(None, None, None)
            except Exception:
                pass
        self._collections.clear()
        self._tools.clear()

    @property
    def tools(self) -> list[Tool]:
        """Return the currently loaded tools."""
        return list(self._tools)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False
