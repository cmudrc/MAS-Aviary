"""Mock tools for testing without real MCP servers.

Three mock tools demonstrating basic, computational, and stateful patterns:
- echo_tool: returns the input string unchanged
- calculator_tool: safely evaluates a math expression
- state_tool: maintains an internal counter across calls
"""

import ast
import operator
import time

from smolagents import Tool

# Invocation log — list of dicts, one per call.
# Any external system (e.g., InstrumentationLogger) can read or hook into this.
_invocation_log: list[dict] = []


def get_invocation_log() -> list[dict]:
    """Return the global invocation log (read-only snapshot)."""
    return list(_invocation_log)


def clear_invocation_log() -> None:
    """Clear the global invocation log (useful between tests)."""
    _invocation_log.clear()


def _log_invocation(tool_name: str, inputs: dict, output, duration: float) -> None:
    """Append an invocation record to the global log."""
    _invocation_log.append({
        "tool_name": tool_name,
        "inputs": inputs,
        "output": str(output),
        "timestamp": time.time(),
        "duration_seconds": duration,
    })


# ---- Echo Tool ----------------------------------------------------------------

class EchoTool(Tool):
    """Takes a string and returns it unchanged — for testing basic tool calling."""

    name = "echo_tool"
    description = "Takes a string and returns it unchanged."
    inputs = {
        "message": {"type": "string", "description": "The string to echo back."},
    }
    output_type = "string"

    def forward(self, message: str) -> str:
        start = time.monotonic()
        result = message
        _log_invocation(self.name, {"message": message}, result, time.monotonic() - start)
        return result


# ---- Calculator Tool ----------------------------------------------------------

# Safe operators for ast-based evaluation.
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node):
    """Recursively evaluate an AST node using only safe arithmetic ops."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        op_fn = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op_fn(_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op_fn = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_fn(_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


class CalculatorTool(Tool):
    """Safely evaluates a math expression string — for testing tool with computation."""

    name = "calculator_tool"
    description = "Evaluates a math expression string safely and returns the numeric result."
    inputs = {
        "expression": {"type": "string", "description": "A math expression, e.g. '2 + 3 * 4'."},
    }
    output_type = "string"

    def forward(self, expression: str) -> str:
        start = time.monotonic()
        try:
            tree = ast.parse(expression, mode="eval")
            result = _safe_eval(tree)
        except Exception as e:
            result = f"Error: {e}"
        _log_invocation(self.name, {"expression": expression}, result, time.monotonic() - start)
        return str(result)


# ---- State Tool ---------------------------------------------------------------

class StateTool(Tool):
    """Maintains an internal counter that increments on each call — for testing stateful tools."""

    name = "state_tool"
    description = "Increments an internal counter and returns the current count."
    inputs = {}
    output_type = "string"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._counter = 0

    def forward(self) -> str:
        start = time.monotonic()
        self._counter += 1
        result = self._counter
        _log_invocation(self.name, {}, result, time.monotonic() - start)
        return str(result)

    def reset(self) -> None:
        """Reset the counter to zero."""
        self._counter = 0


# ---- Registry of all mock tools -----------------------------------------------

MOCK_TOOLS = {
    "echo_tool": EchoTool,
    "calculator_tool": CalculatorTool,
    "state_tool": StateTool,
}


def create_mock_tool(name: str) -> Tool:
    """Create a mock tool instance by name."""
    if name not in MOCK_TOOLS:
        raise ValueError(f"Unknown mock tool: {name!r}. Available: {list(MOCK_TOOLS)}")
    return MOCK_TOOLS[name]()
