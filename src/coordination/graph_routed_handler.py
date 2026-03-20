"""Graph-Routed execution handler — state-machine-driven agent routing.

Implements the ``ExecutionHandler`` interface.  Given a graph definition
(states + transitions + conditions), routes execution through agents
based on condition evaluation against a mutable state dict.

Key behaviours (from PRD):
- Conditional branching via a safe expression evaluator (NOT eval)
- Complexity classification at the initial state
- Resource budgets tied to complexity level
- Cycle detection and complexity escalation
- Internal representations toggle (mental model context for agents)
- LLM graph generation toggle (predefined vs LLM-generated)
- Max transitions safety valve
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

from src.coordination.condition_evaluator import (
    ConditionParseError,
    evaluate_condition,
)
from src.coordination.execution_handler import Assignment, ExecutionHandler
from src.coordination.graph_definition import (
    GraphDefinition,
    load_graph,
    resolve_agent_for_role,
    validate_graph_strict,
)
from src.coordination.history import AgentMessage, ToolCallRecord
from src.coordination.resource_manager import ResourceManager
from src.logging.logger import InstrumentationLogger

# ---------------------------------------------------------------------------
# smolagents extraction helpers (Fix 6 + Fix 7)
# ---------------------------------------------------------------------------


def _extract_tool_calls(agent: Any) -> list[ToolCallRecord]:
    """Extract ToolCallRecords from a smolagents agent after agent.run()."""
    tool_calls: list[ToolCallRecord] = []
    logs = getattr(agent, "logs", None) or []
    if not logs:
        mem = getattr(agent, "memory", None)
        logs = getattr(mem, "steps", None) or []
    for step in logs:
        step_calls = getattr(step, "tool_calls", None)
        if not step_calls:
            continue
        obs = getattr(step, "observations", "") or ""
        for tc in step_calls:
            name = getattr(tc, "name", "") or getattr(tc, "tool_name", "")
            args = getattr(tc, "arguments", {}) or {}
            tool_calls.append(
                ToolCallRecord(
                    tool_name=name,
                    inputs=args if isinstance(args, dict) else {},
                    output=str(obs)[:2000] if obs else "",
                    duration_seconds=0.0,
                )
            )
    return tool_calls


def _extract_token_count(agent: Any, content: str) -> int | None:
    """Return token count, or char-based estimate, or None."""
    for attr in ("token_count", "total_tokens"):
        val = getattr(agent, attr, None)
        if isinstance(val, int) and val > 0:
            return val
    mem = getattr(agent, "memory", None)
    steps = getattr(mem, "steps", None) if mem else None
    if steps:
        total = 0
        for step in steps:
            usage = getattr(step, "token_usage", None)
            if usage:
                if isinstance(usage, dict):
                    total += usage.get("total_tokens", 0) or (
                        usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                    )
                elif isinstance(usage, int):
                    total += usage
        if total:
            return total
    if content:
        return max(1, len(content) // 4)
    return None


# ---------------------------------------------------------------------------
# Transition record for metrics
# ---------------------------------------------------------------------------


@dataclass
class TransitionRecord:
    """Record of a single state transition for metrics."""

    from_state: str
    to_state: str
    condition_matched: str
    agent_invoked: str | None
    passes_remaining: int
    context_used: int
    cycle_count: int
    timestamp: float = 0.0


# ---------------------------------------------------------------------------
# Error type regex
# ---------------------------------------------------------------------------

_ERROR_TYPE_RE = re.compile(r"\b([A-Z][a-zA-Z]*(?:Error|Exception|Warning|Fault))\b")

# Default error type patterns for extraction from agent output/errors.
_DEFAULT_ERROR_PATTERNS = [
    ("SyntaxError", "SyntaxError"),
    ("NameError", "NameError"),
    ("IndentationError", "IndentationError"),
    ("AttributeError", "AttributeError"),
    ("TypeError", "TypeError"),
    ("ValueError", "ValueError"),
    ("Standard_", "StdError"),
]


def _extract_error_type(
    text: str,
    patterns: list[dict] | None = None,
    default: str = "UnknownError",
) -> str:
    """Extract an error type from text using configurable patterns.

    Args:
        text: Error text to classify.
        patterns: List of dicts with ``pattern`` and ``type`` keys.
            If ``None``, uses default patterns.
        default: Fallback error type if no pattern matches.
    """
    if patterns:
        for p in patterns:
            if p["pattern"] in text:
                return p["type"]
    else:
        for pattern, etype in _DEFAULT_ERROR_PATTERNS:
            if pattern in text:
                return etype
    return default


# ---------------------------------------------------------------------------
# Complexity extraction
# ---------------------------------------------------------------------------

_COMPLEXITY_RE = re.compile(r"\b(simple|moderate|complex)\b", re.IGNORECASE)


def _extract_complexity(text: str) -> str | None:
    """Extract a complexity classification from agent output."""
    m = _COMPLEXITY_RE.search(text.lower())
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Review extraction
# ---------------------------------------------------------------------------


def _extract_review_result(text: str) -> dict:
    """Extract review pass/fail or verdict from agent output.

    Returns a dict with keys that can be merged into the state dict:
    ``review_passed`` (bool or None), ``review_verdict`` (str or None).
    """
    lower = text.lower()
    result: dict = {}

    # Binary pass/fail.
    if "review_passed" in lower:
        result["review_passed"] = True
    elif "review_failed" in lower:
        result["review_passed"] = False

    # Verdict keywords.
    if "major_issues" in lower or "major issues" in lower:
        result["review_verdict"] = "major_issues"
    elif "minor_issues" in lower or "minor issues" in lower:
        result["review_verdict"] = "minor_issues"
    elif "passed" in lower:
        result["review_verdict"] = "passed"
        if "review_passed" not in result:
            result["review_passed"] = True

    return result


# ---------------------------------------------------------------------------
# Execution result extraction
# ---------------------------------------------------------------------------


def _extract_execution_result(text: str) -> dict:
    """Extract execution success from agent output."""
    lower = text.lower()
    result: dict = {}

    if "success" in lower and "fail" not in lower:
        result["execution_success"] = True
    elif "fail" in lower or "error" in lower:
        result["execution_success"] = False

    return result


# ---------------------------------------------------------------------------
# Internal representations (mental model)
# ---------------------------------------------------------------------------


def _build_mental_model(
    graph: GraphDefinition,
    state_name: str,
    state_dict: dict,
) -> str:
    """Build a mental model context string for an agent.

    Describes the agent's position in the graph, what upstream/downstream
    states expect, and current resource status.
    """
    state = graph.states.get(state_name)
    if state is None:
        return ""

    lines = [f"WORKFLOW CONTEXT: You are at the {state_name} stage."]

    # Downstream info.
    if state.transitions:
        targets = [t.target for t in state.transitions]
        unique_targets = list(dict.fromkeys(targets))
        lines.append(f"Possible next states: {', '.join(unique_targets)}.")

    # Upstream info — which states lead here.
    upstream = []
    for s in graph.states.values():
        for t in s.transitions:
            if t.target == state_name:
                upstream.append(s.name)
    if upstream:
        unique_up = list(dict.fromkeys(upstream))
        lines.append(f"This state is reached from: {', '.join(unique_up)}.")

    # Resource status.
    pr = state_dict.get("passes_remaining", "?")
    cu = state_dict.get("context_used", 0)
    cb = state_dict.get("context_budget", "?")
    lines.append(f"The system has {pr} passes remaining and has used {cu}/{cb} of the context budget.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# GraphRoutedHandler
# ---------------------------------------------------------------------------


class GraphRoutedHandler(ExecutionHandler):
    """State-machine execution handler with conditional branching.

    Construction accepts a config dict with graph_routed settings.
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self._graph_mode: str = cfg.get("graph_mode", "predefined")
        self._predefined_graph_name: str = cfg.get("predefined_graph", "aviary")
        self._custom_graph_data: dict | None = cfg.get("custom_graph")
        self._allow_graph_modification: bool = cfg.get(
            "allow_graph_modification",
            False,
        )
        self._max_transitions: int = cfg.get("max_transitions", 50)
        self._internal_representations: bool = (
            cfg.get(
                "internal_representations",
                {},
            ).get("enabled", False)
            if isinstance(cfg.get("internal_representations"), dict)
            else bool(cfg.get("internal_representations", False))
        )

        # Error type extraction config.
        self._error_patterns: list[dict] | None = cfg.get("error_type_patterns")
        self._default_error_type: str = cfg.get("default_error_type", "UnknownError")

        # Predefined graph data (loaded externally or passed in config).
        self._graph_data: dict | None = cfg.get("_graph_data")

        # Transition log for metrics.
        self.transition_history: list[TransitionRecord] = []

        # Pre-hook session_id — set programmatically before execute() so
        # every graph node gets the session_id without relying on LLM output.
        self._session_id: str | None = None

    def set_session_id(self, session_id: str) -> None:
        """Inject session_id directly into the handler.

        Called by the coordinator or strategy when a pre-hook has already
        created the session.  The session_id is placed into the state dict
        at the start of execute(), making {session_id} placeholders in
        graph node prompts resolve correctly on every node.
        """
        self._session_id = session_id

        # Per-execution state (reset at start of each execute call).
        self._graph: GraphDefinition | None = None
        self._resource_mgr: ResourceManager | None = None
        self._state_dict: dict = {}

    # ------------------------------------------------------------------
    # ExecutionHandler interface
    # ------------------------------------------------------------------

    def execute(
        self,
        assignments: list[Assignment],
        agents: dict,
        logger: InstrumentationLogger | None,
        turn_offset: int = 0,
        action_metadata: dict | None = None,
    ) -> list[AgentMessage]:
        """Execute the graph-routed state machine.

        The handler ignores the assignments' task-per-agent structure
        and instead routes through its state machine graph.  The first
        assignment's task is used as the overall prompt.
        """
        _base_meta = dict(action_metadata or {})
        messages: list[AgentMessage] = []
        turn = turn_offset

        if not assignments:
            return messages

        task = assignments[0].task

        # 1. Load and validate graph.
        graph = self._load_graph(agents)
        self._graph = graph
        validate_graph_strict(graph)

        # 2. Initialize resource manager.
        # Auto-detect design states: any state with "DESIGN" in the name.
        design_states = frozenset(name for name in graph.states if "DESIGN" in name.upper())
        self._resource_mgr = ResourceManager(
            budgets=graph.resource_budgets,
            design_states=design_states or None,
        )

        # 3. Initialize state dict.
        self._state_dict = {
            "prompt_length": len(task.split()),
            "execution_success": False,
            "error_type": None,
            "error_message": None,
            "review_passed": None,
            "review_verdict": None,
            "last_agent_output": "",
            "last_error": "",
            "attempt_history": [],
            "states_visited": [],
        }

        # Inject pre-hook session_id if set via set_session_id().
        if self._session_id:
            self._state_dict["session_id"] = self._session_id

        current_state = graph.initial_state
        transition_count = 0

        # 4. Main loop.
        while current_state not in graph.terminal_states:
            if transition_count >= self._max_transitions:
                break

            state_def = graph.states.get(current_state)
            if state_def is None:
                msg = AgentMessage(
                    agent_name="graph_routed_handler",
                    content="",
                    turn_number=turn + 1,
                    timestamp=time.time(),
                    error=f"State {current_state!r} not found in graph",
                    metadata=self._build_msg_metadata(current_state, _base_meta),
                )
                messages.append(msg)
                if logger is not None:
                    logger.log_turn(msg)
                break

            self._state_dict["states_visited"].append(current_state)

            # Record design-state re-entry for cycle counting.
            self._resource_mgr.record_state_entry(current_state)

            # Sync resource state into state dict so routing-only
            # states (agent=null) can evaluate conditions like
            # ``cycle_count >= escalation_threshold``.
            self._sync_resource_state()

            # If this state has an agent, run it.
            if state_def.agent is not None:
                try:
                    agent = resolve_agent_for_role(
                        state_def.agent,
                        agents,
                        current_state,
                    )
                except ValueError as e:
                    msg = AgentMessage(
                        agent_name=state_def.agent or "unknown",
                        content="",
                        turn_number=turn + 1,
                        timestamp=time.time(),
                        error=str(e),
                        metadata=self._build_msg_metadata(current_state, _base_meta),
                    )
                    messages.append(msg)
                    if logger is not None:
                        logger.log_turn(msg)
                    break

                # Build context.
                context = self._build_agent_context(
                    task,
                    state_def,
                    current_state,
                )

                turn += 1
                start = time.monotonic()
                try:
                    result = agent.run(context)
                    content = str(result) if result is not None else ""
                except Exception as e:
                    content = ""
                    duration = time.monotonic() - start
                    msg = AgentMessage(
                        agent_name=state_def.agent,
                        content=content,
                        turn_number=turn,
                        timestamp=time.time(),
                        duration_seconds=duration,
                        error=str(e),
                        metadata=self._build_msg_metadata(current_state, _base_meta),
                    )
                    messages.append(msg)
                    if logger is not None:
                        logger.log_turn(msg)

                    # Update state dict with error info.
                    error_text = str(e)
                    self._state_dict["last_error"] = error_text
                    self._state_dict["error_type"] = _extract_error_type(
                        error_text,
                        self._error_patterns,
                        self._default_error_type,
                    )
                    self._state_dict["error_message"] = error_text
                    self._state_dict["execution_success"] = False
                    self._state_dict["attempt_history"].append(
                        {
                            "state": current_state,
                            "agent": state_def.agent,
                            "error": error_text,
                            "output": "",
                        }
                    )
                    self._resource_mgr.consume_pass()
                    self._sync_resource_state()

                    # Evaluate transitions from current state.
                    next_state = self._evaluate_transitions(state_def)
                    if next_state is None:
                        break

                    self.transition_history.append(
                        TransitionRecord(
                            from_state=current_state,
                            to_state=next_state,
                            condition_matched="error_fallback",
                            agent_invoked=state_def.agent,
                            passes_remaining=self._resource_mgr.state.passes_remaining,
                            context_used=self._resource_mgr.state.context_used,
                            cycle_count=self._resource_mgr.state.cycle_count,
                            timestamp=time.time(),
                        )
                    )
                    transition_count += 1
                    current_state = next_state
                    continue

                duration = time.monotonic() - start
                tool_calls = _extract_tool_calls(agent)
                token_count = _extract_token_count(agent, content)
                msg = AgentMessage(
                    agent_name=state_def.agent,
                    content=content,
                    turn_number=turn,
                    timestamp=time.time(),
                    duration_seconds=duration,
                    tool_calls=tool_calls,
                    token_count=token_count,
                    metadata=self._build_msg_metadata(current_state, _base_meta),
                )
                messages.append(msg)
                if logger is not None:
                    logger.log_turn(msg)

                # Update state dict from agent output.
                self._update_state_from_output(content, current_state)

                # Also extract session_id from tool call observations
                # (more reliable than agent text — the UUID comes directly
                # from the MCP server response, not the LLM's prose).
                if "session_id" not in self._state_dict:
                    for tc in tool_calls:
                        if tc.tool_name == "create_session" and tc.output:
                            uuid_match = re.search(
                                r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}"
                                r"-[0-9a-f]{4}-[0-9a-f]{12}",
                                tc.output,
                                re.IGNORECASE,
                            )
                            if uuid_match:
                                self._state_dict["session_id"] = uuid_match.group(0)
                                break

                self._state_dict["last_agent_output"] = content
                self._state_dict["attempt_history"].append(
                    {
                        "state": current_state,
                        "agent": state_def.agent,
                        "output": content,
                    }
                )

                self._resource_mgr.consume_pass()

                # Track code review cycles.
                if current_state == "CODE_REVIEWED":
                    self._resource_mgr.increment_code_review()

                # Estimate token count from content.
                token_estimate = len(content.split())
                self._resource_mgr.add_context(token_estimate)
                self._sync_resource_state()

            # Evaluate transitions.
            matched_condition = None
            next_state = None
            for trans in state_def.transitions:
                try:
                    result = evaluate_condition(trans.condition, self._state_dict)
                except ConditionParseError:
                    continue
                if result.matched:
                    next_state = trans.target
                    matched_condition = trans.condition
                    break

            if next_state is None:
                # No matching transition — stuck.
                break

            self.transition_history.append(
                TransitionRecord(
                    from_state=current_state,
                    to_state=next_state,
                    condition_matched=matched_condition or "none",
                    agent_invoked=state_def.agent,
                    passes_remaining=self._resource_mgr.state.passes_remaining,
                    context_used=self._resource_mgr.state.context_used,
                    cycle_count=self._resource_mgr.state.cycle_count,
                    timestamp=time.time(),
                )
            )
            transition_count += 1
            current_state = next_state

        # Signal graph completion in the last message's metadata so
        # callers (e.g. networked strategy) can detect when the full
        # graph traversal finished without relying on mark_task_done.
        if messages and current_state in graph.terminal_states:
            messages[-1].metadata["graph_complete"] = True

        return messages

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_graph(self, agents: dict) -> GraphDefinition:
        """Load the graph definition based on configuration."""
        if self._graph_mode == "llm_generated":
            return self._generate_graph_with_llm(agents)

        if self._graph_data is not None:
            return load_graph(self._graph_data)

        if self._custom_graph_data is not None:
            return load_graph(self._custom_graph_data)

        # Load predefined graph from YAML.
        graph_file = f"config/{self._predefined_graph_name}_graph.yaml"
        from src.coordination.graph_definition import load_graph_from_yaml

        return load_graph_from_yaml(graph_file)

    def _generate_graph_with_llm(self, agents: dict) -> GraphDefinition:
        """Generate a graph using an LLM agent.

        Requires a ``graph_designer`` agent in the agents dict.
        """
        try:
            designer = resolve_agent_for_role(
                "graph_designer",
                agents,
                "LLM_GRAPH_GENERATION",
            )
        except ValueError:
            raise ValueError("graph_mode is 'llm_generated' but no 'graph_designer' agent is available.")

        roles = list(agents.keys())
        prompt = (
            f"Design a state machine graph for task execution.\n"
            f"Available agent roles: {roles}\n"
            f"Output a valid JSON graph definition."
        )
        result = designer.run(prompt)
        text = str(result)

        # Parse JSON from output.
        try:
            graph_data = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code block.
            json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if json_match:
                graph_data = json.loads(json_match.group(1))
            else:
                raise ValueError(f"LLM graph designer did not produce valid JSON: {text[:200]}")

        graph = load_graph(graph_data)
        validate_graph_strict(graph)
        return graph

    def _build_agent_context(
        self,
        task: str,
        state_def: Any,
        current_state: str,
    ) -> str:
        """Build the context string for an agent at a given state."""
        parts: list[str] = [task]

        # Previous output as context.
        last_output = self._state_dict.get("last_agent_output", "")
        if last_output:
            parts.append(f"Context from previous step:\n{last_output}")

        # Error context if we're coming from an error.
        last_error = self._state_dict.get("last_error", "")
        if last_error:
            parts.append(f"Previous error:\n{last_error}")

        # Agent prompt from the state definition.
        if state_def.agent_prompt:
            try:
                formatted = state_def.agent_prompt.format(**self._state_dict)
            except (KeyError, IndexError):
                formatted = state_def.agent_prompt
            parts.append(formatted)

        # Internal representations (mental model).
        if self._internal_representations and self._graph is not None:
            mental = _build_mental_model(
                self._graph,
                current_state,
                self._state_dict,
            )
            if mental:
                parts.append(mental)

        return "\n\n".join(parts)

    def _update_state_from_output(
        self,
        content: str,
        current_state: str,
    ) -> None:
        """Update the state dict based on agent output content."""
        # Complexity extraction (for classifier states).
        complexity = _extract_complexity(content)
        if complexity is not None:
            self._state_dict["complexity"] = complexity
            if self._resource_mgr is not None:
                self._resource_mgr.set_complexity(complexity)

        # Review result extraction.
        review = _extract_review_result(content)
        if review:
            self._state_dict.update(review)

        # Execution result extraction.
        exec_result = _extract_execution_result(content)
        if exec_result:
            self._state_dict.update(exec_result)

        # Session ID extraction (domain-agnostic).
        session_match = re.search(
            r"(?:SESSION_ID|session_id)['\"\s:=]+([0-9a-f-]{36})",
            content,
            re.IGNORECASE,
        )
        if session_match:
            self._state_dict["session_id"] = session_match.group(1)

        # Convergence extraction (domain-agnostic).
        if "converged" in content.lower():
            if "not converge" in content.lower() or "failed to converge" in content.lower():
                self._state_dict["converged"] = False
            else:
                self._state_dict["converged"] = True

        # Extract error type from content when execution failed.
        if exec_result.get("execution_success") is False:
            etype = _extract_error_type(
                content,
                self._error_patterns,
                self._default_error_type,
            )
            self._state_dict["error_type"] = etype
            self._state_dict["error_message"] = content

        # Clear error state on successful execution.
        if exec_result.get("execution_success"):
            self._state_dict["error_type"] = None
            self._state_dict["error_message"] = None
            self._state_dict["last_error"] = ""

    def _evaluate_transitions(self, state_def: Any) -> str | None:
        """Evaluate transitions from a state and return the next state."""
        for trans in state_def.transitions:
            try:
                result = evaluate_condition(trans.condition, self._state_dict)
            except ConditionParseError:
                continue
            if result.matched:
                return trans.target
        return None

    def _sync_resource_state(self) -> None:
        """Sync resource manager state into the state dict."""
        if self._resource_mgr is not None:
            self._state_dict.update(self._resource_mgr.to_state_dict())

    def _build_msg_metadata(self, current_state: str, base_meta: dict | None = None) -> dict:
        """Build AgentMessage metadata from current graph execution state."""
        meta: dict = dict(base_meta or {})
        meta["graph_state"] = current_state
        if self._resource_mgr is not None:
            rs = self._resource_mgr.state
            meta["passes_remaining"] = rs.passes_remaining
            meta["passes_max"] = rs.passes_max
            meta["context_used"] = rs.context_used
            meta["context_budget"] = rs.context_budget
            meta["cycle_count"] = rs.cycle_count
            meta["escalation_threshold"] = rs.escalation_threshold
        if self._resource_mgr is not None and self._resource_mgr.complexity:
            meta["complexity"] = self._resource_mgr.complexity
        if self.transition_history:
            prev = self.transition_history[-1]
            meta["graph_transition_from"] = prev.from_state
            meta["graph_transition_condition"] = prev.condition_matched
        return meta
