"""Agent Trace Viewer — step-by-step visualization of multi-agent execution.

Shows each agent's reasoning, tool calls, and observations in a clean
conversational layout.  Much more readable than raw JSON.

Launch with:
    streamlit run src/ui/trace_viewer.py
"""

import json
import os
import sys
from pathlib import Path

# Ensure project root is on path when launched via `streamlit run`.
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_AGENT_ICONS = {
    "orchestrator": "🎯",
    "designer": "🎨",
    "coder": "💻",
    "reviewer": "🔍",
    "executor": "⚡",
    "planner": "📋",
    "decomposer": "🔧",
    "worker": "👷",
    "coordinator": "🗂️",
}
_DEFAULT_ICON = "🤖"

_AGENT_COLORS = [
    "#FF6B6B",  # red
    "#4ECDC4",  # teal
    "#45B7D1",  # blue
    "#96CEB4",  # green
    "#FFEAA7",  # yellow
    "#DDA0DD",  # plum
    "#98D8C8",  # mint
    "#F7DC6F",  # gold
]

_ROLE_COLORS = {
    "system": "#ff9800",
    "user": "#2196f3",
    "assistant": "#4caf50",
    "tool_call": "#9c27b0",
    "tool_response": "#795548",
    "tool-call": "#9c27b0",
    "tool-response": "#795548",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_color_cache: dict[str, str] = {}


def _agent_color(name: str) -> str:
    if name not in _color_cache:
        _color_cache[name] = _AGENT_COLORS[len(_color_cache) % len(_AGENT_COLORS)]
    return _color_cache[name]


def _agent_icon(name: str) -> str:
    for key, icon in _AGENT_ICONS.items():
        if key in name.lower():
            return icon
    return _DEFAULT_ICON


def _list_trace_files(base: str = "logs/batch_results/") -> list[str]:
    """Find all trace JSON files under batch results, newest first."""
    base_path = Path(base)
    if not base_path.exists():
        return []
    traces = []
    for d in sorted(base_path.iterdir(), reverse=True):
        if d.is_dir():
            for f in sorted(d.glob("*_trace.json")):
                traces.append(str(f))
    return traces


def _load_trace(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _format_args(args) -> str:
    """Format tool arguments for display."""
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, TypeError):
            return args
    if isinstance(args, dict):
        return json.dumps(args, indent=2, default=str)
    return str(args)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_tool_call(tc: dict, idx: int) -> None:
    """Render a single tool call as a styled expander."""
    name = tc.get("name", "unknown")
    args = tc.get("arguments", {})

    with st.expander(f"🔧 {name}", expanded=False):
        if isinstance(args, dict):
            # Pretty-print code for execute tools.
            code_keys = {"code", "python_code", "script"}
            code_key = code_keys & set(args.keys())
            if code_key:
                key = code_key.pop()
                st.code(args[key], language="python")
                other = {k: v for k, v in args.items() if k != key}
                if other:
                    for k, v in other.items():
                        st.markdown(f"**{k}:** `{v}`")
            else:
                for k, v in args.items():
                    val_str = str(v)
                    if len(val_str) > 300:
                        st.markdown(f"**{k}:**")
                        st.text(val_str[:1000])
                    else:
                        st.markdown(f"**{k}:** `{v}`")
        elif args:
            st.code(_format_args(args))


def _render_observations(obs: str) -> None:
    """Render observations (tool results) in a clean format."""
    if not obs or not obs.strip():
        return

    st.markdown("**Result:**")
    # Detect code / JSON in observations.
    stripped = obs.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            parsed = json.loads(stripped)
            st.json(parsed)
            return
        except (json.JSONDecodeError, TypeError):
            pass

    # Truncate very long observations.
    if len(obs) > 3000:
        st.code(obs[:3000] + f"\n... [{len(obs)} chars total]", language="text")
    else:
        st.code(obs, language="text")


def _render_step(step: dict, agent_name: str) -> None:
    """Render a single execution step as a styled card."""
    icon = _agent_icon(agent_name)
    color = _agent_color(agent_name)
    step_num = step.get("step_number", "?")
    duration = step.get("duration_seconds", 0)
    in_tok = step.get("input_tokens", 0)
    out_tok = step.get("output_tokens", 0)
    model_output = step.get("model_output", "")
    tool_calls = step.get("tool_calls", [])
    observations = step.get("observations", "")
    error = step.get("error")
    is_final = step.get("is_final_answer", False)

    with st.container(border=True):
        # ---- Header row ----
        cols = st.columns([5, 2, 3])
        with cols[0]:
            final_tag = "  ✅ **Final Answer**" if is_final else ""
            st.markdown(
                f"### {icon} <span style='color:{color}'>{agent_name}</span> — Step {step_num}{final_tag}",
                unsafe_allow_html=True,
            )
        with cols[1]:
            if duration:
                st.metric("Duration", f"{duration:.1f}s", label_visibility="collapsed")
        with cols[2]:
            if in_tok or out_tok:
                st.caption(f"📊 {in_tok:,} in → {out_tok:,} out tokens")

        # ---- Error ----
        if error:
            st.error(f"**Error:** {error}")

        # ---- Model output (the agent's reasoning / response) ----
        if model_output:
            # Show in a readable format — detect markdown / code.
            if "```" in model_output:
                st.markdown(model_output)
            elif len(model_output) > 2000:
                st.markdown(model_output[:2000] + f"\n\n*... [{len(model_output)} chars total]*")
            else:
                st.markdown(model_output)

        # ---- Tool calls ----
        if tool_calls:
            for i, tc in enumerate(tool_calls):
                _render_tool_call(tc, i)

        # ---- Observations (combined tool results) ----
        if observations:
            _render_observations(observations)

        # ---- Expandable: full prompt context ----
        input_msgs = step.get("model_input_messages", [])
        if input_msgs:
            with st.expander(f"📨 Prompt context ({len(input_msgs)} messages)"):
                for msg in input_msgs:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    role_color = _ROLE_COLORS.get(role, "#999")
                    st.markdown(
                        f"<span style='color:{role_color};font-weight:bold;font-size:0.85em'>[{role.upper()}]</span>",
                        unsafe_allow_html=True,
                    )
                    if content:
                        display = content[:3000]
                        if len(content) > 3000:
                            display += f"\n... [{len(content)} chars total]"
                        st.text(display)
                    st.markdown("<hr style='margin:4px 0;border-color:#333'>", unsafe_allow_html=True)


def _render_system_prompt(agent_name: str, prompt: str) -> None:
    """Show an agent's system prompt in a styled expander."""
    icon = _agent_icon(agent_name)
    color = _agent_color(agent_name)
    with st.expander(f"🧠 System Prompt — {icon} {agent_name}"):
        st.markdown(
            f"<div style='border-left:3px solid {color}; padding-left:12px'>",
            unsafe_allow_html=True,
        )
        st.text(prompt[:5000])
        st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def main():
    st.set_page_config(
        page_title="Agent Trace Viewer",
        page_icon="🔍",
        layout="wide",
    )

    st.title("🔍 Agent Trace Viewer")
    st.caption("Step-by-step visualization of multi-agent execution")

    # ---- Sidebar ----
    with st.sidebar:
        st.header("Select Trace")
        traces = _list_trace_files()

        if not traces:
            st.warning(
                "No trace files found.\n\n"
                "Run a combination first:\n"
                "```\npython -m src.runners.batch_runner \\\n"
                "  --prompt-index 1 \\\n"
                "  --combinations orchestrated_placeholder\n```"
            )
            st.stop()

        # Group by batch run directory.
        trace_labels = []
        for t in traces:
            p = Path(t)
            batch_id = p.parent.name
            combo = p.stem
            trace_labels.append(f"{batch_id} / {combo}")

        selected_idx = st.selectbox(
            "Trace file",
            range(len(traces)),
            format_func=lambda i: trace_labels[i],
        )
        selected_path = traces[selected_idx]
        data = _load_trace(selected_path)

        # Run info card.
        st.markdown("---")
        st.markdown("### Run Info")
        st.markdown(f"**Combination:** `{data.get('combination', 'N/A')}`")
        st.markdown(f"**Org Structure:** `{data.get('org_structure', 'N/A')}`")
        st.markdown(f"**Handler:** `{data.get('handler', 'N/A')}`")
        st.markdown(f"**Status:** `{data.get('status', 'N/A')}`")
        duration = data.get("duration_seconds", 0)
        st.markdown(f"**Duration:** `{duration:.1f}s`")
        st.markdown(f"**Turns:** `{data.get('total_turns', 'N/A')}`")

        # Agent list.
        agents = data.get("agents", {})
        st.markdown("---")
        st.markdown("### Agents")
        for name, agent_data in agents.items():
            icon = _agent_icon(name)
            n_steps = len(agent_data.get("steps", []))
            has_error = bool(agent_data.get("error"))
            badge = " ⚠️" if has_error else ""
            st.markdown(f"{icon} **{name}** — {n_steps} steps{badge}")

        # View mode.
        st.markdown("---")
        view_mode = st.radio(
            "View mode",
            ["Timeline", "Per Agent"],
            index=0,
        )

        selected_agent = None
        if view_mode == "Per Agent" and agents:
            selected_agent = st.selectbox("Select agent", list(agents.keys()))

    # ---- Main content ----
    task = data.get("task", "")
    agents = data.get("agents", {})

    # Task description.
    if task:
        with st.expander("📝 Task Description", expanded=False):
            if len(task) > 5000:
                st.markdown(task[:5000] + "\n\n*...[truncated]*")
            else:
                st.markdown(task)

    if view_mode == "Timeline":
        _render_timeline(agents)
    else:
        _render_per_agent(agents, selected_agent)


def _render_timeline(agents: dict) -> None:
    """Render all agent steps interleaved chronologically."""
    # Collect all steps with agent names.
    all_steps: list[tuple[str, dict]] = []
    for name, agent_data in agents.items():
        for step in agent_data.get("steps", []):
            all_steps.append((name, step))

    # Sort by start_time if available, else preserve order.
    def _sort_key(item):
        _, step = item
        t = step.get("start_time")
        if t is not None:
            return (t, step.get("step_number", 0))
        return (0, step.get("step_number", 0))

    all_steps.sort(key=_sort_key)

    if not all_steps:
        st.info("No execution steps recorded in this trace.")
        return

    # Summary stats.
    total_duration = sum(s.get("duration_seconds", 0) for _, s in all_steps)
    total_in_tok = sum(s.get("input_tokens", 0) for _, s in all_steps)
    total_out_tok = sum(s.get("output_tokens", 0) for _, s in all_steps)

    cols = st.columns(4)
    with cols[0]:
        st.metric("Total Steps", len(all_steps))
    with cols[1]:
        st.metric("Total Duration", f"{total_duration:.1f}s")
    with cols[2]:
        st.metric("Input Tokens", f"{total_in_tok:,}")
    with cols[3]:
        st.metric("Output Tokens", f"{total_out_tok:,}")

    st.markdown("---")

    # Agent flow visualization.
    agent_sequence = []
    for name, _ in all_steps:
        if not agent_sequence or agent_sequence[-1] != name:
            agent_sequence.append(name)

    flow_parts = []
    for name in agent_sequence:
        icon = _agent_icon(name)
        color = _agent_color(name)
        flow_parts.append(
            f"<span style='background-color:{color}; color:#000; "
            f"padding:3px 8px; border-radius:10px; font-weight:bold; "
            f"font-size:0.85em'>{icon} {name}</span>"
        )
    st.markdown("**Agent Flow:** " + " → ".join(flow_parts), unsafe_allow_html=True)
    st.markdown("")

    # Render each step.
    for name, step in all_steps:
        _render_step(step, name)


def _render_per_agent(agents: dict, selected_agent: str | None) -> None:
    """Render steps for a single selected agent."""
    if not selected_agent or selected_agent not in agents:
        st.info("Select an agent from the sidebar.")
        return

    agent_data = agents[selected_agent]

    # System prompt.
    sys_prompt = agent_data.get("system_prompt", "")
    if sys_prompt:
        _render_system_prompt(selected_agent, sys_prompt)

    # Steps.
    steps = agent_data.get("steps", [])
    if not steps:
        error = agent_data.get("error", "")
        if error:
            st.error(f"Agent error: {error}")
        else:
            st.info(f"No execution steps recorded for **{selected_agent}**.")
        return

    # Summary.
    total_duration = sum(s.get("duration_seconds", 0) for s in steps)
    total_in = sum(s.get("input_tokens", 0) for s in steps)
    total_out = sum(s.get("output_tokens", 0) for s in steps)

    cols = st.columns(4)
    with cols[0]:
        st.metric("Steps", len(steps))
    with cols[1]:
        st.metric("Duration", f"{total_duration:.1f}s")
    with cols[2]:
        st.metric("Input Tokens", f"{total_in:,}")
    with cols[3]:
        st.metric("Output Tokens", f"{total_out:,}")

    st.markdown("---")

    for step in steps:
        _render_step(step, selected_agent)


if __name__ == "__main__":
    main()
