"""Batch runner — runs prompts through all OS x handler combinations.

Runs Aviary aircraft design prompts through 12 combinations
(3 org structures x 4 handlers). Each combination runs sequentially
(shared GPU). Results are saved to logs/batch_results/{timestamp}/ as JSON.

Trace capture: after each agent run, smolagents step-by-step traces
(prompts, model outputs, tool calls, observations) are extracted and
saved alongside the batch results as {combo_name}_trace.json.

Usage:
    python -m src.runners.batch_runner
    python -m src.runners.batch_runner --config config/aviary_run.yaml
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.coordination.history import AgentMessage
from src.logging.cross_strategy_metrics import compute_cross_strategy_metrics
from src.logging.eval_classifier import (
    classify_aviary_eval,
    detect_aviary_agent_signals,
)
from src.logging.org_theory_metrics import compute_org_theory_metrics

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy-specific config overrides
# ---------------------------------------------------------------------------

# Maps org_structure -> (agents_config, coordination_config).
# When an org structure needs different config files than the default,
# the batch runner swaps these before calling Coordinator.from_config.
_STRATEGY_CONFIGS: dict[str, tuple[str, str]] = {
    "sequential": ("config/sequential_agents.yaml", "config/aviary_sequential.yaml"),
    "orchestrated": ("config/aviary_orchestrated_agents.yaml", "config/orchestrated.yaml"),
    "networked": ("config/aviary_networked_agents.yaml", "config/aviary_networked.yaml"),
}


# ---------------------------------------------------------------------------
# Combination definition
# ---------------------------------------------------------------------------


@dataclass
class CombinationConfig:
    """Configuration for one OS x handler combination."""

    name: str
    org_structure: str  # "orchestrated" | "networked" | "sequential"
    handler: str  # "placeholder" | "iterative_feedback" | "graph_routed" | "staged_pipeline"
    strategy_config: dict = field(default_factory=dict)
    handler_config: dict = field(default_factory=dict)


# Aviary: 12 combinations (sequential + orchestrated + networked) x handlers.
_AVIARY_STAGED_HANDLER_CONFIG: dict = {
    "pipeline_path": "config/aviary_staged_pipeline.yaml",
    "context_mode": "all_stages",
    "verdict_patterns": [
        "WEIGHT_PARAMETERS_SET",
        "VERDICT",
        "PASSED",
        "MINOR_ISSUES",
        "MAJOR_ISSUES",
    ],
}

ALL_COMBINATIONS: list[CombinationConfig] = [
    # Sequential x 3 handlers.
    CombinationConfig(
        "aviary_sequential_placeholder",
        "sequential",
        "placeholder",
        strategy_config={"pipeline_template": "aviary"},
    ),
    CombinationConfig(
        "aviary_sequential_iterative_feedback",
        "sequential",
        "iterative_feedback",
        strategy_config={"pipeline_template": "aviary"},
    ),
    CombinationConfig(
        "aviary_sequential_staged_pipeline",
        "sequential",
        "staged_pipeline",
        strategy_config={"pipeline_template": "aviary"},
        handler_config=_AVIARY_STAGED_HANDLER_CONFIG,
    ),
    # Orchestrated x 3 handlers.
    CombinationConfig("aviary_orchestrated_placeholder", "orchestrated", "placeholder"),
    CombinationConfig("aviary_orchestrated_iterative_feedback", "orchestrated", "iterative_feedback"),
    CombinationConfig(
        "aviary_orchestrated_staged_pipeline",
        "orchestrated",
        "staged_pipeline",
        handler_config=_AVIARY_STAGED_HANDLER_CONFIG,
    ),
    # Graph-routed: sequential + graph_routed with aviary graph.
    CombinationConfig(
        "aviary_sequential_graph_routed",
        "sequential",
        "graph_routed",
        strategy_config={"pipeline_template": "aviary"},
        handler_config={"predefined_graph": "aviary"},
    ),
    # Orchestrated + graph_routed with aviary graph.
    CombinationConfig(
        "aviary_orchestrated_graph_routed",
        "orchestrated",
        "graph_routed",
        handler_config={"predefined_graph": "aviary"},
    ),
    # Networked x 4 handlers.
    CombinationConfig("aviary_networked_placeholder", "networked", "placeholder"),
    CombinationConfig("aviary_networked_iterative_feedback", "networked", "iterative_feedback"),
    CombinationConfig(
        "aviary_networked_staged_pipeline",
        "networked",
        "staged_pipeline",
        handler_config=_AVIARY_STAGED_HANDLER_CONFIG,
    ),
    # Networked + graph_routed: disable workflow_phases (graph manages workflow).
    CombinationConfig(
        "aviary_networked_graph_routed",
        "networked",
        "graph_routed",
        strategy_config={"networked": {"workflow_phases": []}},
        handler_config={"predefined_graph": "aviary"},
    ),
]


# ---------------------------------------------------------------------------
# Per-combination result
# ---------------------------------------------------------------------------


@dataclass
class CombinationResult:
    """Result of running one OS x handler combination."""

    name: str
    org_structure: str
    handler: str
    status: str = "pending"  # "success" | "error" | "eval_skipped"
    error_message: str = ""
    duration_seconds: float = 0.0
    total_turns: int = 0
    total_tokens: int = 0
    messages: list[dict] = field(default_factory=list)
    eval_classification: dict = field(default_factory=dict)
    cross_strategy_metrics: dict = field(default_factory=dict)
    strategy_metrics: dict = field(default_factory=dict)
    handler_metrics: dict = field(default_factory=dict)
    org_theory_metrics: dict = field(default_factory=dict)
    gpu_memory_mb: float = 0.0
    # Transient: smolagents step traces (saved to separate file, excluded from summary).
    traces: dict = field(default_factory=dict, repr=False)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _msg_to_dict(msg: AgentMessage) -> dict:
    """Convert AgentMessage to JSON-serializable dict."""
    return {
        "agent_name": msg.agent_name,
        "content": msg.content,
        "turn_number": msg.turn_number,
        "timestamp": msg.timestamp,
        "duration_seconds": msg.duration_seconds,
        "token_count": msg.token_count,
        "error": msg.error,
        "is_retry": msg.is_retry,
        "retry_of_turn": msg.retry_of_turn,
        "metadata": msg.metadata,
        "tool_calls": [
            {
                "tool_name": tc.tool_name,
                "inputs": tc.inputs,
                "output": tc.output[:2000] if tc.output else "",
                "duration_seconds": tc.duration_seconds,
                "error": tc.error,
            }
            for tc in msg.tool_calls
        ],
    }


def _safe_dict(obj: Any) -> Any:
    """Convert dataclass/object to JSON-safe dict, excluding traces."""
    if hasattr(obj, "__dataclass_fields__"):
        d = asdict(obj)
        d.pop("traces", None)
        return d
    if isinstance(obj, dict):
        return {k: _safe_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_dict(x) for x in obj]
    return obj


# ---------------------------------------------------------------------------
# GPU memory management (Fix 2)
# ---------------------------------------------------------------------------


def _gpu_memory_mb() -> float:
    """Return current GPU memory allocated in MB, or 0 if CUDA unavailable."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
    except ImportError:
        pass
    return 0.0


def _gpu_cleanup() -> None:
    """Free GPU memory between combinations.

    Deletes all cached references, runs garbage collection, and empties
    the CUDA memory cache.
    """
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass
    gc.collect()


# ---------------------------------------------------------------------------
# Agent trace extraction (smolagents step-by-step traces)
# ---------------------------------------------------------------------------


def _serialize_chat_message(msg) -> dict:
    """Serialize a ChatMessage (object or dict) to JSON-safe format."""
    if isinstance(msg, dict):
        role = msg.get("role", "")
        content = msg.get("content", "")
    elif hasattr(msg, "role"):
        role = msg.role
        if hasattr(role, "value"):
            role = role.value
        content = msg.content
    else:
        return {"role": "unknown", "content": str(msg)[:5000]}

    # Handle multi-part content (list of dicts with type/text).
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                # Extract text, handling cases where make_json_serializable
                # may have parsed JSON strings into dicts/lists.
                text_val = part.get("text") or part.get("thinking") or part.get("content", "")
                if not isinstance(text_val, str):
                    text_val = json.dumps(text_val, default=str) if text_val else ""
                if text_val:
                    parts.append(text_val)
                elif part.get("type"):
                    parts.append(f"[{part['type']}]")
                else:
                    parts.append(str(part))
            else:
                parts.append(str(part))
        content = "\n".join(parts)

    content = str(content or "")
    if len(content) > 10000:
        content = content[:10000] + f"\n... [truncated, {len(content)} chars total]"
    return {"role": str(role), "content": content}


def _serialize_step(step: dict) -> dict:
    """Serialize a smolagents step dict to a clean JSON-safe format."""
    out: dict[str, Any] = {}
    out["step_number"] = step.get("step_number", 0)

    # Timing.
    timing = step.get("timing", {})
    if isinstance(timing, dict):
        out["start_time"] = timing.get("start_time")
        out["end_time"] = timing.get("end_time")
        out["duration_seconds"] = timing.get("duration", 0)
    elif hasattr(timing, "duration"):
        out["start_time"] = timing.start_time
        out["end_time"] = timing.end_time
        out["duration_seconds"] = timing.duration
    else:
        out["duration_seconds"] = 0

    # Model output (the agent's reasoning / response text).
    model_output = step.get("model_output", "")
    if isinstance(model_output, list):
        parts: list[str] = []
        for part in model_output:
            if isinstance(part, dict):
                # Prefer text/thinking keys; make_json_serializable may have
                # parsed JSON strings into dicts, so force str conversion.
                text_val = part.get("text") or part.get("thinking") or ""
                if not isinstance(text_val, str):
                    text_val = json.dumps(text_val, default=str)
                parts.append(text_val or str(part))
            else:
                parts.append(str(part))
        model_output = "\n".join(parts)
    out["model_output"] = str(model_output or "")

    # Tool calls.
    raw_calls = step.get("tool_calls") or []
    calls = []
    for tc in raw_calls:
        if isinstance(tc, dict):
            func = tc.get("function", tc)
            calls.append(
                {
                    "name": func.get("name", tc.get("name", "unknown")),
                    "arguments": func.get("arguments", tc.get("arguments", {})),
                    "id": tc.get("id", ""),
                }
            )
        elif hasattr(tc, "name"):
            calls.append(
                {
                    "name": tc.name,
                    "arguments": tc.arguments if hasattr(tc, "arguments") else {},
                    "id": tc.id if hasattr(tc, "id") else "",
                }
            )
    out["tool_calls"] = calls

    # Observations (tool results).
    out["observations"] = str(step.get("observations", "") or "")

    # Token usage.
    tu = step.get("token_usage")
    if isinstance(tu, dict):
        out["input_tokens"] = tu.get("input_tokens", 0)
        out["output_tokens"] = tu.get("output_tokens", 0)
    elif tu and hasattr(tu, "input_tokens"):
        out["input_tokens"] = tu.input_tokens
        out["output_tokens"] = tu.output_tokens
    else:
        out["input_tokens"] = 0
        out["output_tokens"] = 0

    # Error.
    error = step.get("error")
    if error:
        if isinstance(error, dict):
            out["error"] = error.get("message", str(error))
        else:
            out["error"] = str(error)

    out["is_final_answer"] = bool(step.get("is_final_answer", False))

    # Model input messages (full prompt context).
    input_msgs = step.get("model_input_messages") or []
    out["model_input_messages"] = [_serialize_chat_message(m) for m in input_msgs]

    return out


def _extract_agent_traces(coordinator) -> dict:
    """Extract smolagents step traces from all agents in a coordinator.

    Returns a dict mapping agent_name -> {system_prompt, steps: [...]}.

    Two-pass approach:
      1. Start with the accumulated traces from the trace-capture wrappers.
      2. For any agent whose accumulated entry has 0 steps (or is missing),
         attempt post-hoc extraction from the agent's current memory.  This
         covers dynamically created agents AND cases where the wrapper's
         capture silently failed.
    """
    traces: dict[str, dict] = {}

    # Start with accumulated traces if available.
    if hasattr(coordinator, "_trace_accumulator"):
        traces.update(coordinator._trace_accumulator)

    # Post-hoc extraction for agents with missing or empty steps.
    for name, agent in coordinator.agents.items():
        existing = traces.get(name)
        if existing and existing.get("steps"):
            continue  # already have meaningful data — skip
        if not hasattr(agent, "memory"):
            traces.setdefault(name, {"system_prompt": "", "steps": []})
            continue
        try:
            system_prompt = _get_system_prompt(agent)
            try:
                steps = agent.memory.get_full_steps()
            except Exception:
                steps = agent.memory.get_succinct_steps() if hasattr(agent.memory, "get_succinct_steps") else []

            if steps:
                traces[name] = {
                    "system_prompt": system_prompt,
                    "steps": [_serialize_step(s) for s in steps],
                }
                _log.debug("Post-hoc extraction for %r: %d steps", name, len(steps))
            else:
                traces.setdefault(name, {"system_prompt": system_prompt, "steps": []})
                _log.debug("Post-hoc extraction for %r: no steps in memory", name)
        except Exception as e:
            _log.warning("Post-hoc trace extraction failed for %r: %s", name, e)
            traces.setdefault(name, {"error": str(e), "steps": []})
    return traces


def _get_system_prompt(agent) -> str:
    """Extract system prompt from a smolagents agent."""
    if not hasattr(agent, "memory"):
        return ""
    mem = agent.memory
    if hasattr(mem, "system_prompt") and mem.system_prompt:
        sp = mem.system_prompt
        return sp.system_prompt if hasattr(sp, "system_prompt") else str(sp)
    return ""


def _install_trace_capture(coordinator) -> None:
    """Wrap agent.run() methods to capture step traces before memory reset.

    smolagents resets agent memory at the start of each run() call, so we
    capture steps after each invocation completes, accumulating them into
    coordinator._trace_accumulator.

    Also monitors ``coordinator.agents`` for dynamically created agents
    (e.g. workers spawned by the orchestrator's ``create_agent`` tool)
    and wraps them on-the-fly when they first appear.
    """
    accumulator: dict[str, dict] = {}
    coordinator._trace_accumulator = accumulator
    # Track which agents have already been wrapped to avoid double-wrapping.
    wrapped_names: set[str] = set()

    def _wrap_agent(name: str, agent) -> None:
        """Install a trace-capturing wrapper on a single agent."""
        if name in wrapped_names:
            return
        if not hasattr(agent, "run"):
            return
        wrapped_names.add(name)
        if name not in accumulator:
            accumulator[name] = {"system_prompt": "", "steps": []}

        original_run = agent.run

        def _make_traced_run(agent_name, orig_run, agent_ref):
            def traced_run(*args, **kwargs):
                result = orig_run(*args, **kwargs)
                try:
                    steps = agent_ref.memory.get_full_steps()
                    serialized = [_serialize_step(s) for s in steps]
                    accumulator[agent_name]["steps"].extend(serialized)
                    accumulator[agent_name]["system_prompt"] = _get_system_prompt(agent_ref)
                except Exception as exc:
                    _log.warning(
                        "Trace capture failed for agent %r: %s",
                        agent_name,
                        exc,
                    )
                return result

            return traced_run

        agent.run = _make_traced_run(name, original_run, agent)

    # Wrap all agents that exist at installation time.
    for name, agent in coordinator.agents.items():
        _wrap_agent(name, agent)

    # Store the wrapper function so it can be called for dynamic agents.
    coordinator._wrap_agent_for_trace = _wrap_agent


# ---------------------------------------------------------------------------
# Aviary eval extraction from agent messages
# ---------------------------------------------------------------------------

# Regex patterns for Aviary output metrics.
_FUEL_RE = re.compile(
    r"fuel[_ ]burned[_ ]kg['\":\s=]+([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
_GTOW_RE = re.compile(
    r"(?:gtow|gross[_ ]mass)[_ ]kg['\":\s=]+([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
_WING_MASS_RE = re.compile(
    r"wing[_ ]mass[_ ]kg['\":\s=]+([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
_RESERVE_FUEL_RE = re.compile(
    r"reserve[_ ]fuel[_ ]kg['\":\s=]+([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
_ZFW_RE = re.compile(
    r"zero[_ ]fuel[_ ]weight[_ ]kg['\":\s=]+([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
_CONVERGED_RE = re.compile(
    r"(?:converged|exit[_ ]code)['\":\s=]+(true|false|0|1)",
    re.IGNORECASE,
)
_OPT_GAP_RE = re.compile(
    r"optimality[_ ]gap[_ ]pct['\":\s=]+['\"]?([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
_VERDICT_RE = re.compile(
    r"VERDICT['\"\s:=]+['\"]?(COMPLETE|CONTINUE|RETRY|FAILED)",
    re.IGNORECASE,
)


_METRIC_TOOL_NAMES = frozenset(("get_results", "run_simulation", "validate_parameters"))

_METRIC_KEYS = ("fuel_burned_kg", "gtow_kg", "wing_mass_kg", "reserve_fuel_kg", "zero_fuel_weight_kg")


def _try_parse_json(raw: str) -> dict | None:
    """Parse JSON from a possibly-truncated tool output string.

    Uses ``JSONDecoder.raw_decode()`` (CMU DRC pattern) to recover
    partial JSON — finds the first valid JSON object in the string and
    returns whatever fields completed before truncation.
    """
    if not isinstance(raw, str):
        return raw if isinstance(raw, dict) else None
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(raw):
        try:
            obj, end = decoder.raw_decode(raw, idx)
            if isinstance(obj, dict):
                return obj
            idx = end
        except json.JSONDecodeError:
            idx += 1
    return None


def _extract_from_tool_outputs(messages: list[AgentMessage]) -> dict | None:
    """Extract Aviary metrics from MCP tool call outputs (ground truth).

    Scans tool calls for ``get_results``, ``run_simulation``, or
    ``validate_parameters`` outputs. Uses ``raw_decode()`` to recover
    partial data from truncated JSON. Returns ``None`` for metrics not
    found (never coerces to 0.0 — callers must distinguish missing from
    zero).
    """
    fuel = gtow = wing = reserve = zfw = None
    converged = None

    for msg in reversed(messages):
        for tc in msg.tool_calls:
            if tc.error is not None or not tc.output:
                continue
            if tc.tool_name not in _METRIC_TOOL_NAMES:
                continue
            data = _try_parse_json(tc.output)
            if not isinstance(data, dict) or not data.get("success", False):
                continue
            # run_simulation nests metrics inside a "summary" dict.
            if "summary" in data and isinstance(data["summary"], dict):
                data = {**data, **data["summary"]}
            # validate_parameters nests metrics inside "model_eval.outputs".
            model_eval = data.get("model_eval")
            if isinstance(model_eval, dict):
                outputs = model_eval.get("outputs")
                if isinstance(outputs, dict):
                    data = {**data, **outputs}
            # Extract metrics from the structured MCP response.
            if fuel is None and "fuel_burned_kg" in data:
                fuel = float(data["fuel_burned_kg"])
            if gtow is None and "gtow_kg" in data:
                gtow = float(data["gtow_kg"])
            if wing is None and "wing_mass_kg" in data:
                wing = float(data["wing_mass_kg"])
            if reserve is None and "reserve_fuel_kg" in data:
                reserve = float(data["reserve_fuel_kg"])
            if zfw is None and "zero_fuel_weight_kg" in data:
                zfw = float(data["zero_fuel_weight_kg"])
            if converged is None and "converged" in data:
                converged = bool(data["converged"])

    # Return None only if nothing at all was found.
    if fuel is None:
        return None

    # Return None for each metric not found — NOT 0.0.
    return {
        "fuel_burned_kg": fuel,
        "gtow_kg": gtow,
        "wing_mass_kg": wing,
        "reserve_fuel_kg": reserve,
        "zero_fuel_weight_kg": zfw,
        "converged": converged if converged is not None else True,
    }


def _regex_extract_metric(
    pattern: re.Pattern,
    messages: list[AgentMessage],
) -> float | None:
    """Scan agent message content for a metric using a regex."""
    for msg in reversed(messages):
        m = pattern.search(msg.content)
        if m:
            return float(m.group(1))
    return None


def _extract_aviary_eval_from_messages(messages: list[AgentMessage]) -> dict | None:
    """Extract Aviary optimization results from agent messages.

    Hybrid approach: extract each metric from tool call outputs first
    (ground truth), then fall through to regex on agent content for any
    metric still missing.  Never returns 0.0 for genuinely missing data —
    uses ``None`` so the classifier can distinguish missing from zero.

    Returns a dict with fuel_burned_kg, gtow_kg, wing_mass_kg, reserve_fuel_kg,
    zero_fuel_weight_kg, converged, optimality_gap_pct, verdict — or None if
    no eval data found.
    """
    # Primary: extract from tool call outputs (ground truth).
    tool_result = _extract_from_tool_outputs(messages)

    # Start with tool-output values (may have None for truncated fields).
    fuel = tool_result["fuel_burned_kg"] if tool_result else None
    gtow = tool_result["gtow_kg"] if tool_result else None
    wing = tool_result["wing_mass_kg"] if tool_result else None
    reserve = tool_result["reserve_fuel_kg"] if tool_result else None
    zfw = tool_result["zero_fuel_weight_kg"] if tool_result else None
    converged = tool_result["converged"] if tool_result else True

    # Fallback: regex on agent content for any metric still None.
    if fuel is None:
        fuel = _regex_extract_metric(_FUEL_RE, messages)
    if gtow is None:
        gtow = _regex_extract_metric(_GTOW_RE, messages)
    if wing is None:
        wing = _regex_extract_metric(_WING_MASS_RE, messages)
    if reserve is None:
        reserve = _regex_extract_metric(_RESERVE_FUEL_RE, messages)
    if zfw is None:
        zfw = _regex_extract_metric(_ZFW_RE, messages)

    # Convergence from agent content (if not found in tool output).
    if tool_result is None or not tool_result.get("converged", True):
        for msg in reversed(messages):
            m = _CONVERGED_RE.search(msg.content)
            if m:
                val = m.group(1).lower()
                converged = val in ("true", "0")
                break

    # Optimality gap and verdict always from agent content.
    opt_gap = _regex_extract_metric(_OPT_GAP_RE, messages) or 0.0
    verdict = None
    for msg in reversed(messages):
        m = _VERDICT_RE.search(msg.content)
        if m:
            verdict = m.group(1).upper()
            break

    # Need at least fuel_burned_kg to consider eval valid.
    if fuel is None:
        return None

    return {
        "fuel_burned_kg": fuel,
        "gtow_kg": gtow,
        "wing_mass_kg": wing,
        "reserve_fuel_kg": reserve,
        "zero_fuel_weight_kg": zfw,
        "converged": converged,
        "optimality_gap_pct": opt_gap,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge *override* into *base* in-place.

    Nested dicts are merged; all other values are replaced.
    """
    for key, val in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val


# ---------------------------------------------------------------------------
# Networked + graph_routed role aliasing
# ---------------------------------------------------------------------------


def _register_networked_graph_aliases(agents: dict, roles: list[str]) -> None:
    """Map graph role names to networked peer agents.

    Networked agents have generic names (agent_1, agent_2, ...) with no
    semantic overlap with graph roles (mission_architect, coder, executor, ...).
    This function creates aliases in the agents dict so the graph
    handler's ``resolve_agent_for_role()`` can find them.

    Strategy: tool-based heuristic first, then round-robin fallback.
    """
    _ROLE_TOOL_HINTS: dict[str, list[str]] = {
        # Aviary aircraft design roles.
        "simulation_executor": ["run_simulation", "get_results"],
        "mdo_integrator": ["check_constraints"],
        "mission_architect": ["create_session", "configure_mission"],
        "propulsion_analyst": ["set_aircraft_parameters"],
        "aerodynamics_analyst": ["set_aircraft_parameters", "get_design_space"],
        "weights_analyst": ["get_design_space"],
    }

    peer_names = [k for k in agents if k.startswith("agent_")]
    peers = [agents[n] for n in peer_names]
    if not peers:
        return

    # Build tool -> agent index.
    tool_index: dict[str, list] = {}
    for ag in peers:
        for tn in getattr(ag, "tools", {}):
            tool_index.setdefault(tn, []).append(ag)

    for idx, role in enumerate(roles):
        if role in agents:
            continue
        # 1. Tool-based heuristic.
        matched = False
        for ht in _ROLE_TOOL_HINTS.get(role, []):
            cands = tool_index.get(ht, [])
            if cands:
                agents[role] = cands[idx % len(cands)]
                matched = True
                break
        if matched:
            continue
        # 2. Round-robin fallback — all peers are interchangeable.
        agents[role] = peers[idx % len(peers)]


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------


def run_combination(
    combo: CombinationConfig,
    task: str,
    config,
    *,
    model=None,
    tools: list | None = None,
    session_id: str | None = None,
) -> CombinationResult:
    """Run a single OS x handler combination.

    Args:
        combo: The combination configuration.
        task: The task prompt to run.
        config: AppConfig instance.
        session_id: Pre-created session ID (from pre-hook). Passed to the
            coordinator so graph-routed handlers can inject it into node prompts.
        model: Pre-loaded LLM model (shared across combos).
        tools: Pre-loaded tools (shared across combos).

    Returns:
        CombinationResult with all metrics.
    """
    result = CombinationResult(
        name=combo.name,
        org_structure=combo.org_structure,
        handler=combo.handler,
    )

    start = time.monotonic()
    try:
        messages, traces = _execute_combination(combo, task, config, model=model, tools=tools, session_id=session_id)
        result.duration_seconds = time.monotonic() - start
        result.status = "success"
        result.total_turns = len(messages)
        result.total_tokens = sum(m.token_count or 0 for m in messages)
        result.messages = [_msg_to_dict(m) for m in messages]
        result.gpu_memory_mb = _gpu_memory_mb()
        result.traces = traces

        # Compute cross-strategy metrics.
        cs_metrics = compute_cross_strategy_metrics(
            messages,
            similarity_method="jaccard",
        )
        result.cross_strategy_metrics = cs_metrics

        # Detect agent approval/issue signals.
        outputs = [m.content for m in messages]
        approved, flagged = detect_aviary_agent_signals(outputs)
        eval_result = _extract_aviary_eval_from_messages(messages)
        converged = eval_result.get("converged", True) if eval_result else True
        classification = classify_aviary_eval(
            eval_result,
            converged=converged,
            agent_approved=approved,
            agent_flagged_issues=flagged,
        )
        result.eval_classification = _safe_dict(classification)

        # Compute org theory metrics.
        ot_config: dict = dict(combo.handler_config)
        ot_config["pipeline_template"] = combo.strategy_config.get("pipeline_template")
        ot_config["_eval_success"] = classification.result == "success" if classification else None

        # Inject stage allowed_tools for sequential modularity metrics.
        if combo.org_structure == "sequential":
            try:
                _inject_stage_allowed_tools(ot_config)
            except Exception:
                pass  # Non-critical — modularity metrics degrade gracefully
        result.org_theory_metrics = compute_org_theory_metrics(
            messages,
            os_name=combo.org_structure,
            handler_name=combo.handler,
            config=ot_config,
        )

    except Exception as e:
        result.duration_seconds = time.monotonic() - start
        result.status = "error"
        result.error_message = f"{type(e).__name__}: {e}"
        result.gpu_memory_mb = _gpu_memory_mb()

    return result


def _execute_combination(
    combo: CombinationConfig,
    task: str,
    config,
    *,
    model=None,
    tools: list | None = None,
    session_id: str | None = None,
) -> tuple[list[AgentMessage], dict]:
    """Execute one combination and return (messages, traces).

    This builds a Coordinator with the right strategy and handler,
    then runs the task. After execution, smolagents step traces are
    extracted from all agents. Config fields are temporarily swapped
    to point to the correct strategy-specific agents/coordination files.
    """
    from src.coordination.coordinator import Coordinator
    from src.logging.logger import InstrumentationLogger

    logger = InstrumentationLogger(config={})

    # Fix 1: Swap config paths for strategy-specific agents/coordination.
    original_agents = config.agents_config
    original_coord = config.coordination_config
    try:
        if combo.org_structure in _STRATEGY_CONFIGS:
            agents_path, coord_path = _STRATEGY_CONFIGS[combo.org_structure]
            config.agents_config = agents_path
            config.coordination_config = coord_path

        coordinator = Coordinator.from_config(
            config,
            logger=logger,
            strategy_override=combo.org_structure,
        )

        # Override execution handler if not the one set by config.
        if combo.handler != "placeholder":
            handler = _build_handler(combo.handler, combo.handler_config)
            if handler is not None:
                coordinator.execution_handler = handler

                # Inject handler name into coordinator config so strategies
                # can read it during initialize() (e.g. networked strategy
                # skips phase gating for staged_pipeline).
                coordinator.config["execution_handler"] = combo.handler

                # When combining orchestrated + graph_routed, extract graph
                # roles and wire them into the strategy so it can register
                # role aliases after the creation phase.  Also inject the
                # ListGraphRoles tool so the orchestrator can discover roles.
                if combo.org_structure == "orchestrated" and combo.handler == "graph_routed":
                    try:
                        graph = handler._load_graph({})
                        roles = sorted({s.agent for s in graph.states.values() if s.agent})
                        if roles:
                            # Set in config dict so initialize() picks it up
                            # (strategy.initialize() reads config["_graph_roles"]).
                            coordinator.config["_graph_roles"] = roles
                            # Inject ListGraphRoles tool into orchestrator.
                            from src.tools.orchestrator_tools import ListGraphRoles

                            orch_name = getattr(
                                coordinator.strategy,
                                "_orchestrator_name",
                                "orchestrator",
                            )
                            orch = coordinator.agents.get(orch_name)
                            if orch is not None:
                                tool = ListGraphRoles(graph_roles=roles)
                                orch.tools[tool.name] = tool
                    except Exception:
                        pass

                # When combining networked + graph_routed, store graph
                # roles so they can be aliased after strategy.initialize()
                # creates the peer agents (inside coordinator.run()).
                if combo.org_structure == "networked" and combo.handler == "graph_routed":
                    try:
                        graph = handler._load_graph({})
                        roles = sorted({s.agent for s in graph.states.values() if s.agent})
                        if roles:
                            coordinator.config["_graph_roles"] = roles
                        # Pass graph definition to strategy so it can drive
                        # the state machine one-state-per-turn instead of
                        # delegating to the handler's full traversal.
                        coordinator.config["_graph_def"] = graph
                    except Exception:
                        pass

        # Install trace capture BEFORE running so we catch all agent invocations.
        _install_trace_capture(coordinator)

        # Auto-wrap dynamically created agents (e.g. workers spawned by the
        # orchestrator's create_agent tool) so their run() calls are also
        # captured.  We replace the agents dict with a subclass that calls
        # the wrapper function on every new insertion.
        wrap_fn = coordinator._wrap_agent_for_trace
        _orig_agents = coordinator.agents

        class _AutoWrapDict(dict):
            """Dict that auto-wraps new agents for trace capture."""

            def __setitem__(self, key, value):
                super().__setitem__(key, value)
                try:
                    wrap_fn(key, value)
                except Exception:
                    pass

        auto_dict = _AutoWrapDict(_orig_agents)
        coordinator.agents = auto_dict
        # Ensure the strategy AND its orchestrator context all share the
        # same dict reference.  Without this, agents created dynamically
        # by the orchestrator (via CreateAgent) would be added to the old
        # dict and invisible to the coordinator and handler.
        if hasattr(coordinator, "strategy"):
            strat = coordinator.strategy
            if hasattr(strat, "_agents"):
                strat._agents = auto_dict
            if hasattr(strat, "_context") and strat._context is not None:
                strat._context.agents = auto_dict

        # For networked + graph_routed: wrap strategy.initialize() to
        # inject role aliases AFTER peer agents are created.  The graph
        # handler needs roles like 'mission_architect', 'simulation_executor'
        # etc. to map to actual agents, but networked agents have generic
        # names (agent_1, agent_2, ...).  We create aliases in the shared
        # agents dict so resolve_agent_for_role() succeeds.
        _graph_roles = coordinator.config.get("_graph_roles")
        if _graph_roles and combo.org_structure == "networked":
            _orig_init = coordinator.strategy.initialize

            def _init_with_aliases(agents, config, *, _orig=_orig_init, _roles=_graph_roles):
                _orig(agents, config)
                _register_networked_graph_aliases(agents, _roles)

            coordinator.strategy.initialize = _init_with_aliases

        # Merge strategy_config overrides into coordinator config so
        # strategy.initialize() sees them.  Deep-merges nested dicts
        # (e.g. {"networked": {"workflow_phases": []}}) into the
        # coordinator's config dict loaded from YAML.
        if combo.strategy_config:
            _deep_merge(coordinator.config, combo.strategy_config)

        result = coordinator.run(task, session_id=session_id)
        traces = _extract_agent_traces(coordinator)
        return result.history, traces
    finally:
        # Restore original config paths.
        config.agents_config = original_agents
        config.coordination_config = original_coord


def _build_handler(handler_name: str, handler_config: dict):
    """Build an execution handler by name."""
    if handler_name == "iterative_feedback":
        from src.coordination.iterative_feedback_handler import IterativeFeedbackHandler

        return IterativeFeedbackHandler(handler_config)

    if handler_name == "graph_routed":
        from src.coordination.graph_routed_handler import GraphRoutedHandler

        return GraphRoutedHandler(handler_config)

    if handler_name == "staged_pipeline":
        from src.coordination.staged_pipeline_handler import StagedPipelineHandler

        return StagedPipelineHandler(handler_config)

    return None


def _inject_stage_allowed_tools(ot_config: dict) -> None:
    """Load the pipeline template and inject _stage_allowed_tools into ot_config.

    Reads the sequential config files to resolve the template name and
    stage definitions, then builds a {stage_name: allowed_tools} mapping
    for use by modularity metrics in org_theory_metrics.
    """
    import yaml

    from src.coordination.pipeline_templates import load_template

    # Load coordination config (has pipeline_template name).
    coord_path = Path("config/sequential.yaml")
    if not coord_path.exists():
        return
    with open(coord_path) as f:
        coord_cfg = yaml.safe_load(f) or {}
    seq_cfg = coord_cfg.get("sequential", {})
    template_name = seq_cfg.get("pipeline_template", "linear")

    # Override from ot_config if already set (e.g. from strategy_config).
    if ot_config.get("pipeline_template"):
        template_name = ot_config["pipeline_template"]
    else:
        ot_config["pipeline_template"] = template_name

    # Load agents config (has template definitions).
    agents_path = Path("config/sequential_agents.yaml")
    templates_config = None
    if agents_path.exists():
        with open(agents_path) as f:
            agents_cfg = yaml.safe_load(f) or {}
        templates_config = agents_cfg.get("templates")

    custom_stages = seq_cfg.get("custom_stages") or None
    template = load_template(template_name, custom_stages, templates_config)

    stage_allowed: dict[str, list[str]] = {}
    for stage in template.stages:
        stage_allowed[stage.name] = list(stage.allowed_tools)
    ot_config["_stage_allowed_tools"] = stage_allowed


# ---------------------------------------------------------------------------
# Batch execution
# ---------------------------------------------------------------------------


def run_batch(
    task: str,
    config,
    combinations: list[CombinationConfig] | None = None,
    output_dir: str | None = None,
) -> list[CombinationResult]:
    """Run all combinations for a single task.

    Each combination runs sequentially. Failures are caught and logged.
    GPU memory is cleaned up between combinations.

    Args:
        task: The task prompt text.
        config: AppConfig instance.
        combinations: Which combinations to run (default: ALL_COMBINATIONS).
        output_dir: Where to save results (default: logs/batch_results/{ts}/).

    Returns:
        List of CombinationResult.
    """
    combos = combinations or ALL_COMBINATIONS
    results: list[CombinationResult] = []

    for i, combo in enumerate(combos):
        # Fix 2: Clean up GPU memory before each combination.
        mem_before = _gpu_memory_mb()
        if i > 0:
            _gpu_cleanup()
            mem_after = _gpu_memory_mb()
            print(f"  [GPU] cleanup: {mem_before:.0f}MB -> {mem_after:.0f}MB", flush=True)

        print(f"Running: {combo.name} ...", flush=True)
        try:
            result = run_combination(combo, task, config)
        except Exception as e:
            result = CombinationResult(
                name=combo.name,
                org_structure=combo.org_structure,
                handler=combo.handler,
                status="error",
                error_message=f"{type(e).__name__}: {e}",
            )
        results.append(result)

        eval_info = ""
        ec = result.eval_classification
        if ec:
            eval_info = f" eval={ec.get('result', 'N/A')}"
            fuel = ec.get("fuel_burned_kg")
            if fuel and fuel > 0:
                eval_info += f" fuel={fuel:.1f}kg"
            gap = ec.get("optimality_gap_pct")
            if gap:
                eval_info += f" gap={gap:.1f}%"
        print(
            f"  -> {result.status} ({result.total_turns} turns, "
            f"{result.duration_seconds:.1f}s, {result.gpu_memory_mb:.0f}MB{eval_info})"
        )

    # Save results.
    if output_dir is None:
        ts = int(time.time())
        output_dir = f"logs/batch_results/{ts}"
    save_batch_results(results, task, output_dir)

    return results


def save_batch_results(
    results: list[CombinationResult],
    task: str,
    output_dir: str,
) -> Path:
    """Save batch results and agent traces to JSON files.

    Creates:
      - batch_summary.json: overview of all combinations (no traces).
      - {combo_name}_trace.json: full smolagents step traces per combination.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save per-combination trace files.
    for r in results:
        if r.traces:
            trace_data = {
                "combination": r.name,
                "org_structure": r.org_structure,
                "handler": r.handler,
                "task": task,
                "status": r.status,
                "duration_seconds": r.duration_seconds,
                "total_turns": r.total_turns,
                "timestamp": time.time(),
                "agents": r.traces,
            }
            trace_path = out / f"{r.name}_trace.json"
            try:
                with open(trace_path, "w") as f:
                    json.dump(trace_data, f, indent=2, default=str)
            except Exception as e:
                print(f"  [WARN] Could not save trace for {r.name}: {e}")

    # Summary file (no traces).
    summary = {
        "task": task,
        "domain": "aviary",
        "timestamp": time.time(),
        "total_combinations": len(results),
        "results": [_safe_dict(r) for r in results],
    }
    summary_path = out / "batch_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Results saved to {out}/", flush=True)
    return out


def load_batch_results(path: str | Path) -> dict:
    """Load batch results from a batch_summary.json file."""
    path = Path(path)
    if path.is_dir():
        path = path / "batch_summary.json"
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_DEFAULT_AVIARY_TASK = (
    "Design a single-aisle commercial aircraft for 1,500 nmi range, "
    "162 passengers, cruise Mach 0.785, FL350. Optimize for minimum "
    "fuel burn. Constraints: fuel_burned_kg <= 8500, gtow_kg <= 72000."
)


def main():
    parser = argparse.ArgumentParser(description="Aviary batch runner")
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Custom task string (overrides default aviary task)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to AppConfig YAML (default: config/aviary_run.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--combinations",
        type=str,
        nargs="*",
        default=None,
        help="Specific combination names to run (default: all)",
    )
    args = parser.parse_args()

    config_path = args.config or "config/aviary_run.yaml"

    from src.config.loader import load_config

    config = load_config(config_path)

    task = args.task or _DEFAULT_AVIARY_TASK

    # Filter combinations if specified.
    combos = ALL_COMBINATIONS
    if args.combinations:
        combos = [c for c in ALL_COMBINATIONS if c.name in args.combinations]

    print(f"Running {len(combos)} combinations for aviary task")
    results = run_batch(task, config, combos, args.output_dir)

    successes = sum(1 for r in results if r.status == "success")
    errors = sum(1 for r in results if r.status == "error")
    print(f"\nDone: {successes} success, {errors} errors out of {len(results)}")


if __name__ == "__main__":
    main()
