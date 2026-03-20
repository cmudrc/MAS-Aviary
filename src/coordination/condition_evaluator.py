"""Safe condition evaluator for graph-routed transition conditions.

Parses condition strings into a simple AST and evaluates against a
state dict.  This is NOT ``eval()`` — it only supports a restricted
set of comparison and logical operators.

Supported expressions
---------------------
- Equality:      ``complexity == 'simple'``
- Inequality:    ``passes_remaining > 0``, ``<``, ``>=``, ``<=``, ``!=``
- Membership:    ``error_type in ['SyntaxError', 'NameError']``
- Boolean:       ``execution_success == true``  (``true``/``false`` literals)
- Compound:      ``execution_success == true and stl_produced == true``
- Always:        ``always``  (always matches)

Missing keys
------------
If a condition references a state dict key that doesn't exist, the
comparison evaluates to ``False`` (non-matching) and a warning is
logged via the returned ``warnings`` list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Token types and lexer
# ---------------------------------------------------------------------------

_TOKEN_PATTERNS = [
    ("STRING_SQ", r"'[^']*'"),        # single-quoted string
    ("STRING_DQ", r'"[^"]*"'),        # double-quoted string
    ("LIST_OPEN", r"\["),
    ("LIST_CLOSE", r"\]"),
    ("COMMA", r","),
    ("OP", r"==|!=|>=|<=|>|<"),
    ("AND", r"\band\b"),
    ("OR", r"\bor\b"),
    ("IN", r"\bin\b"),
    ("TRUE", r"\btrue\b"),
    ("FALSE", r"\bfalse\b"),
    ("ALWAYS", r"\balways\b"),
    ("NUMBER", r"-?\d+(?:\.\d+)?"),
    ("IDENT", r"[a-zA-Z_][a-zA-Z0-9_]*"),
    ("WS", r"\s+"),
]

_TOKEN_RE = re.compile(
    "|".join(f"(?P<{name}>{pattern})" for name, pattern in _TOKEN_PATTERNS)
)


@dataclass
class _Token:
    type: str
    value: str


def _tokenize(expr: str) -> list[_Token]:
    """Tokenize a condition expression string."""
    tokens: list[_Token] = []
    for m in _TOKEN_RE.finditer(expr):
        kind = m.lastgroup
        val = m.group()
        if kind == "WS":
            continue
        tokens.append(_Token(type=kind, value=val))
    return tokens


# ---------------------------------------------------------------------------
# AST node types
# ---------------------------------------------------------------------------

@dataclass
class _Always:
    """The ``always`` literal — always matches."""


@dataclass
class _VarRef:
    """A reference to a state dict variable (unquoted identifier on the
    right side of a comparison, e.g. ``cycle_count >= escalation_threshold``)."""
    name: str


@dataclass
class _Comparison:
    """A binary comparison: ``left op right``."""
    left: str       # state dict key
    op: str         # ==, !=, >, <, >=, <=
    right: Any      # literal value or _VarRef


@dataclass
class _Membership:
    """An ``in`` check: ``left in [list]``."""
    left: str            # state dict key
    values: list[Any]    # list of literal values


@dataclass
class _And:
    """Logical AND of two sub-expressions."""
    left: Any
    right: Any


@dataclass
class _Or:
    """Logical OR of two sub-expressions."""
    left: Any
    right: Any


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class ConditionParseError(Exception):
    """Raised when a condition expression cannot be parsed."""


class _Parser:
    """Recursive-descent parser for condition expressions."""

    def __init__(self, tokens: list[_Token]):
        self._tokens = tokens
        self._pos = 0

    def _peek(self) -> _Token | None:
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None

    def _advance(self) -> _Token:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _expect(self, *types: str) -> _Token:
        tok = self._peek()
        if tok is None or tok.type not in types:
            expected = " or ".join(types)
            got = tok.type if tok else "end of input"
            raise ConditionParseError(
                f"Expected {expected}, got {got}"
            )
        return self._advance()

    def parse(self) -> Any:
        node = self._parse_or()
        if self._pos < len(self._tokens):
            raise ConditionParseError(
                f"Unexpected token: {self._tokens[self._pos].value!r}"
            )
        return node

    def _parse_or(self) -> Any:
        left = self._parse_and()
        while self._peek() and self._peek().type == "OR":
            self._advance()
            right = self._parse_and()
            left = _Or(left=left, right=right)
        return left

    def _parse_and(self) -> Any:
        left = self._parse_atom()
        while self._peek() and self._peek().type == "AND":
            self._advance()
            right = self._parse_atom()
            left = _And(left=left, right=right)
        return left

    def _parse_atom(self) -> Any:
        tok = self._peek()
        if tok is None:
            raise ConditionParseError("Unexpected end of expression")

        if tok.type == "ALWAYS":
            self._advance()
            return _Always()

        if tok.type == "IDENT":
            ident_tok = self._advance()
            ident = ident_tok.value

            next_tok = self._peek()
            if next_tok is None:
                raise ConditionParseError(
                    f"Expected operator after {ident!r}, got end of input"
                )

            # Membership: ident in [...]
            if next_tok.type == "IN":
                self._advance()
                values = self._parse_list()
                return _Membership(left=ident, values=values)

            # Comparison: ident op literal
            if next_tok.type == "OP":
                op_tok = self._advance()
                right = self._parse_literal()
                return _Comparison(left=ident, op=op_tok.value, right=right)

            raise ConditionParseError(
                f"Expected operator after {ident!r}, got {next_tok.value!r}"
            )

        raise ConditionParseError(f"Unexpected token: {tok.value!r}")

    def _parse_literal(self) -> Any:
        tok = self._peek()
        if tok is None:
            raise ConditionParseError("Expected literal, got end of input")

        if tok.type in ("STRING_SQ", "STRING_DQ"):
            self._advance()
            return tok.value[1:-1]  # strip quotes

        if tok.type == "NUMBER":
            self._advance()
            if "." in tok.value:
                return float(tok.value)
            return int(tok.value)

        if tok.type == "TRUE":
            self._advance()
            return True

        if tok.type == "FALSE":
            self._advance()
            return False

        if tok.type == "IDENT":
            # Bare identifier → variable reference into the state dict.
            self._advance()
            return _VarRef(name=tok.value)

        raise ConditionParseError(
            f"Expected literal value, got {tok.value!r}"
        )

    def _parse_list(self) -> list[Any]:
        self._expect("LIST_OPEN")
        values: list[Any] = []
        # Empty list.
        if self._peek() and self._peek().type == "LIST_CLOSE":
            self._advance()
            return values
        values.append(self._parse_literal())
        while self._peek() and self._peek().type == "COMMA":
            self._advance()
            values.append(self._parse_literal())
        self._expect("LIST_CLOSE")
        return values


def parse_condition(expr: str) -> Any:
    """Parse a condition expression string into an AST node.

    Raises:
        ConditionParseError: If the expression cannot be parsed.
    """
    tokens = _tokenize(expr)
    if not tokens:
        raise ConditionParseError(f"Empty condition expression: {expr!r}")
    parser = _Parser(tokens)
    return parser.parse()


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Result of evaluating a condition expression."""
    matched: bool
    warnings: list[str] = field(default_factory=list)


def _compare(left_val: Any, op: str, right_val: Any) -> bool:
    """Perform a comparison operation."""
    if op == "==":
        return left_val == right_val
    if op == "!=":
        return left_val != right_val
    # Numeric comparisons — coerce to float for safety.
    try:
        lv = float(left_val)
        rv = float(right_val)
    except (ValueError, TypeError):
        return False
    if op == ">":
        return lv > rv
    if op == "<":
        return lv < rv
    if op == ">=":
        return lv >= rv
    if op == "<=":
        return lv <= rv
    return False


def _eval_node(node: Any, state: dict) -> EvalResult:
    """Recursively evaluate an AST node against a state dict."""
    if isinstance(node, _Always):
        return EvalResult(matched=True)

    if isinstance(node, _Comparison):
        warnings: list[str] = []
        if node.left not in state:
            return EvalResult(
                matched=False,
                warnings=[f"Key {node.left!r} not found in state dict"],
            )
        left_val = state[node.left]

        # Resolve right side: _VarRef → state dict lookup.
        right_val = node.right
        if isinstance(right_val, _VarRef):
            if right_val.name not in state:
                return EvalResult(
                    matched=False,
                    warnings=[f"Key {right_val.name!r} not found in state dict"],
                )
            right_val = state[right_val.name]

        matched = _compare(left_val, node.op, right_val)
        return EvalResult(matched=matched, warnings=warnings)

    if isinstance(node, _Membership):
        if node.left not in state:
            return EvalResult(
                matched=False,
                warnings=[f"Key {node.left!r} not found in state dict"],
            )
        left_val = state[node.left]
        matched = left_val in node.values
        return EvalResult(matched=matched)

    if isinstance(node, _And):
        left_result = _eval_node(node.left, state)
        if not left_result.matched:
            return left_result
        right_result = _eval_node(node.right, state)
        return EvalResult(
            matched=right_result.matched,
            warnings=left_result.warnings + right_result.warnings,
        )

    if isinstance(node, _Or):
        left_result = _eval_node(node.left, state)
        if left_result.matched:
            return left_result
        right_result = _eval_node(node.right, state)
        return EvalResult(
            matched=right_result.matched,
            warnings=left_result.warnings + right_result.warnings,
        )

    raise ConditionParseError(f"Unknown AST node type: {type(node).__name__}")


def evaluate_condition(expr: str, state: dict) -> EvalResult:
    """Parse and evaluate a condition expression against a state dict.

    Args:
        expr: Condition string (e.g. ``"complexity == 'simple'"``).
        state: State dict to evaluate against.

    Returns:
        EvalResult with ``matched`` bool and any ``warnings``.

    Raises:
        ConditionParseError: If the expression cannot be parsed.
    """
    ast_node = parse_condition(expr)
    return _eval_node(ast_node, state)
