"""Unified tool loading interface — loads tools from mock definitions or real MCP."""

from smolagents import Tool

from src.config.loader import AppConfig
from src.tools.mock_tools import MOCK_TOOLS, create_mock_tool

# Keep MCP connectors alive so tool connections aren't garbage-collected.
_active_connectors: list = []


def load_tools_for_agent(tool_names: list[str], config: AppConfig) -> list[Tool]:
    """Load tool instances for an agent based on config mode.

    In 'mock' mode, tools are created from the local mock definitions.
    In 'real' mode, tools come from a shared MCPConnector instance.

    Args:
        tool_names: List of tool name strings from the agent config.
        config: Application config (used to determine mock vs real mode).

    Returns:
        List of smolagents Tool instances ready for agent use.
    """
    if config.mcp.mode == "mock":
        return [create_mock_tool(name) for name in tool_names]

    # If all requested tools are available as mocks, skip MCP connection.
    if tool_names and all(name in MOCK_TOOLS for name in tool_names):
        return [create_mock_tool(name) for name in tool_names]

    # Real MCP mode — connect and filter tools by requested names.
    from src.tools.mcp_connector import MCPConnector

    connector = MCPConnector(config.mcp)
    all_tools = connector.connect()
    _active_connectors.append(connector)

    if not tool_names:
        return all_tools

    # Build lookup by tool name.
    tool_map = {t.name: t for t in all_tools}
    result = []
    for name in tool_names:
        if name in tool_map:
            result.append(tool_map[name])
        elif name in MOCK_TOOLS:
            # Fall back to mock if a tool name isn't found on MCP servers.
            result.append(create_mock_tool(name))

    return result
