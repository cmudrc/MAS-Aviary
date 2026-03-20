"""Tests for MCP connector and real-mode tool loading."""

from contextlib import contextmanager
from unittest.mock import patch

import pytest

from src.config.loader import AppConfig, MCPConfig, MCPServerConfig
from src.tools.mcp_connector import MCPConnector

# ---- Helpers ---------------------------------------------------------------


class FakeTool:
    """Minimal stand-in for a smolagents Tool discovered via MCP."""

    def __init__(self, name: str):
        self.name = name
        self.description = f"Fake MCP tool: {name}"


class FakeToolCollection:
    """Mimics the object returned by ToolCollection.from_mcp().__enter__()."""

    def __init__(self, tool_names: list[str]):
        self.tools = [FakeTool(n) for n in tool_names]


@contextmanager
def _fake_from_mcp(config, trust_remote_code=True):
    """Context manager that yields a FakeToolCollection based on the url."""
    url = config.get("url", "")
    if "server1" in url:
        yield FakeToolCollection(["tool_a", "tool_b"])
    elif "server2" in url:
        yield FakeToolCollection(["tool_c"])
    elif "empty" in url:
        yield FakeToolCollection([])
    elif "fail" in url:
        raise ConnectionError(f"Cannot connect to {url}")
    else:
        yield FakeToolCollection(["default_tool"])


def _make_config(*servers: tuple[str, str]) -> MCPConfig:
    """Create an MCPConfig with the given (url, transport) pairs."""
    return MCPConfig(
        mode="real",
        servers=[MCPServerConfig(url=u, transport=t) for u, t in servers],
    )


# ---- MCPConnector ----------------------------------------------------------


class TestMCPConnector:
    @patch("src.tools.mcp_connector.ToolCollection.from_mcp")
    def test_connect_single_server(self, mock_from_mcp):
        mock_from_mcp.side_effect = _fake_from_mcp
        config = _make_config(("http://server1/mcp", "streamable-http"))

        connector = MCPConnector(config)
        tools = connector.connect()

        assert len(tools) == 2
        assert tools[0].name == "tool_a"
        assert tools[1].name == "tool_b"
        mock_from_mcp.assert_called_once_with(
            {"url": "http://server1/mcp", "transport": "streamable-http"},
            trust_remote_code=True,
        )

    @patch("src.tools.mcp_connector.ToolCollection.from_mcp")
    def test_connect_multiple_servers(self, mock_from_mcp):
        mock_from_mcp.side_effect = _fake_from_mcp
        config = _make_config(
            ("http://server1/mcp", "streamable-http"),
            ("http://server2/mcp", "streamable-http"),
        )

        connector = MCPConnector(config)
        tools = connector.connect()

        assert len(tools) == 3
        names = [t.name for t in tools]
        assert names == ["tool_a", "tool_b", "tool_c"]

    @patch("src.tools.mcp_connector.ToolCollection.from_mcp")
    def test_tools_property(self, mock_from_mcp):
        mock_from_mcp.side_effect = _fake_from_mcp
        config = _make_config(("http://server1/mcp", "streamable-http"))

        connector = MCPConnector(config)
        connector.connect()
        assert len(connector.tools) == 2
        # Property returns a copy
        assert connector.tools is not connector._tools

    @patch("src.tools.mcp_connector.ToolCollection.from_mcp")
    def test_disconnect_clears_state(self, mock_from_mcp):
        mock_from_mcp.side_effect = _fake_from_mcp
        config = _make_config(("http://server1/mcp", "streamable-http"))

        connector = MCPConnector(config)
        connector.connect()
        assert len(connector.tools) == 2

        connector.disconnect()
        assert len(connector.tools) == 0
        assert connector._collections == []

    @patch("src.tools.mcp_connector.ToolCollection.from_mcp")
    def test_context_manager(self, mock_from_mcp):
        mock_from_mcp.side_effect = _fake_from_mcp
        config = _make_config(("http://server1/mcp", "streamable-http"))

        with MCPConnector(config) as connector:
            assert len(connector.tools) == 2

        # After exit, tools should be cleared
        assert len(connector.tools) == 0

    @patch("src.tools.mcp_connector.ToolCollection.from_mcp")
    def test_empty_server_list(self, mock_from_mcp):
        config = MCPConfig(mode="real", servers=[])
        connector = MCPConnector(config)
        tools = connector.connect()
        assert tools == []
        mock_from_mcp.assert_not_called()

    @patch("src.tools.mcp_connector.ToolCollection.from_mcp")
    def test_empty_tool_collection(self, mock_from_mcp):
        mock_from_mcp.side_effect = _fake_from_mcp
        config = _make_config(("http://empty/mcp", "streamable-http"))

        connector = MCPConnector(config)
        tools = connector.connect()
        assert tools == []

    @patch("src.tools.mcp_connector.ToolCollection.from_mcp")
    def test_connection_error_propagates(self, mock_from_mcp):
        mock_from_mcp.side_effect = _fake_from_mcp
        config = _make_config(("http://fail/mcp", "streamable-http"))

        connector = MCPConnector(config)
        with pytest.raises(ConnectionError, match="Cannot connect"):
            connector.connect()


# ---- tool_loader real mode -------------------------------------------------


class TestToolLoaderRealMode:
    @patch("src.tools.mcp_connector.ToolCollection.from_mcp")
    def test_load_all_mcp_tools(self, mock_from_mcp):
        from src.tools.tool_loader import load_tools_for_agent

        mock_from_mcp.side_effect = _fake_from_mcp
        config = AppConfig(mcp=_make_config(("http://server1/mcp", "streamable-http")))

        tools = load_tools_for_agent([], config)
        assert len(tools) == 2

    @patch("src.tools.mcp_connector.ToolCollection.from_mcp")
    def test_load_filtered_mcp_tools(self, mock_from_mcp):
        from src.tools.tool_loader import load_tools_for_agent

        mock_from_mcp.side_effect = _fake_from_mcp
        config = AppConfig(mcp=_make_config(("http://server1/mcp", "streamable-http")))

        tools = load_tools_for_agent(["tool_a"], config)
        assert len(tools) == 1
        assert tools[0].name == "tool_a"

    @patch("src.tools.mcp_connector.ToolCollection.from_mcp")
    def test_fallback_to_mock_tool(self, mock_from_mcp):
        from src.tools.tool_loader import load_tools_for_agent

        mock_from_mcp.side_effect = _fake_from_mcp
        config = AppConfig(mcp=_make_config(("http://server1/mcp", "streamable-http")))

        # echo_tool is not on the MCP server, should fall back to mock
        tools = load_tools_for_agent(["tool_a", "echo_tool"], config)
        assert len(tools) == 2
        assert tools[0].name == "tool_a"
        assert tools[1].name == "echo_tool"

    @patch("src.tools.mcp_connector.ToolCollection.from_mcp")
    def test_multiple_servers_merged(self, mock_from_mcp):
        from src.tools.tool_loader import load_tools_for_agent

        mock_from_mcp.side_effect = _fake_from_mcp
        config = AppConfig(
            mcp=_make_config(
                ("http://server1/mcp", "streamable-http"),
                ("http://server2/mcp", "streamable-http"),
            )
        )

        tools = load_tools_for_agent(["tool_a", "tool_c"], config)
        assert len(tools) == 2
        names = [t.name for t in tools]
        assert "tool_a" in names
        assert "tool_c" in names
