"""Reusable Streamlit UI components for the dashboard."""

import streamlit as st

from src.ui.state import get_agent_color


def render_agent_message(msg: dict, index: int) -> None:
    """Render a single agent message in the activity feed."""
    agent = msg.get("agent_name", "unknown")
    color = get_agent_color(agent)
    content = msg.get("content", "")
    turn = msg.get("turn_number", "?")
    duration = msg.get("duration_seconds")
    token_count = msg.get("token_count")
    error = msg.get("error")
    is_retry = msg.get("is_retry", False)
    retry_of = msg.get("retry_of_turn")
    tool_calls = msg.get("tool_calls", []) or []

    # Header
    retry_badge = ""
    if is_retry and retry_of:
        retry_badge = f" &nbsp; :orange[Retry of turn #{retry_of}]"

    error_badge = ""
    if error:
        error_badge = " &nbsp; :red[ERROR]"

    st.markdown(
        f"**Turn {turn}** &mdash; "
        f"<span style='color:{color}; font-weight:bold'>{agent}</span>"
        f"{retry_badge}{error_badge}",
        unsafe_allow_html=True,
    )

    # Content
    if error:
        st.error(f"Error: {error}")
    if content:
        st.text(content[:500] + ("..." if len(content) > 500 else ""))

    # Tool calls (expandable)
    if tool_calls:
        with st.expander(f"Tool calls ({len(tool_calls)})"):
            for tc in tool_calls:
                tool_name = tc.get("tool_name", "unknown")
                inputs = tc.get("inputs", {})
                output = tc.get("output", "")
                tc_error = tc.get("error")
                tc_duration = tc.get("duration_seconds")

                st.markdown(f"**{tool_name}**")
                st.json(inputs)
                if tc_error:
                    st.error(tc_error)
                else:
                    st.code(str(output)[:200])
                if tc_duration is not None:
                    st.caption(f"Duration: {tc_duration:.3f}s")

    # Footer metadata
    meta_parts = []
    if duration is not None:
        meta_parts.append(f"Duration: {duration:.3f}s")
    if token_count is not None:
        meta_parts.append(f"Tokens: {token_count}")
    if meta_parts:
        st.caption(" | ".join(meta_parts))

    st.divider()


def render_strategy_panel(history: list[dict], strategy: str) -> None:
    """Render the strategy visualization panel."""
    if not history:
        st.info("No messages yet.")
        return

    # Show agent sequence as a flow
    agents_seen = []
    for msg in history:
        agent = msg.get("agent_name", "?")
        if not agents_seen or agents_seen[-1] != agent:
            agents_seen.append(agent)

    # Visual flow
    flow_parts = []
    for agent in agents_seen:
        color = get_agent_color(agent)
        flow_parts.append(
            f"<span style='background-color:{color}; color:#000; "
            f"padding:4px 10px; border-radius:12px; margin:2px; "
            f"display:inline-block; font-weight:bold'>{agent}</span>"
        )
    flow_html = " &rarr; ".join(flow_parts)
    st.markdown(f"**Agent Flow:** {flow_html}", unsafe_allow_html=True)

    # Agent participation breakdown
    st.markdown("**Agent Participation:**")
    agent_counts: dict[str, int] = {}
    for msg in history:
        agent = msg.get("agent_name", "?")
        agent_counts[agent] = agent_counts.get(agent, 0) + 1

    for agent, count in agent_counts.items():
        color = get_agent_color(agent)
        bar_width = int(count / max(agent_counts.values()) * 100)
        st.markdown(
            f"<div style='display:flex; align-items:center; margin:4px 0'>"
            f"<span style='width:100px; font-weight:bold'>{agent}</span>"
            f"<div style='background-color:{color}; height:20px; "
            f"width:{bar_width}%; border-radius:4px; margin-left:8px'></div>"
            f"<span style='margin-left:8px'>{count} turns</span></div>",
            unsafe_allow_html=True,
        )


def render_metrics_panel(metrics: dict) -> None:
    """Render the metrics dashboard panel."""
    if not metrics:
        st.info("No metrics available.")
        return

    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Messages", metrics.get("total_messages", 0))
    with col2:
        tokens = metrics.get("total_tokens")
        st.metric("Total Tokens", tokens if tokens is not None else "N/A")
    with col3:
        duration = metrics.get("total_duration_seconds", 0)
        st.metric("Total Duration", f"{duration:.2f}s")
    with col4:
        tool_calls = metrics.get("total_tool_calls", 0)
        st.metric("Tool Calls", tool_calls)

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("Errors", metrics.get("error_count", 0))
    with col6:
        error_rate = metrics.get("error_rate", 0)
        st.metric("Error Rate", f"{error_rate:.1%}")
    with col7:
        st.metric("Retries", metrics.get("retry_count", 0))
    with col8:
        efficiency = metrics.get("coordination_efficiency", 0)
        st.metric("Efficiency", f"{efficiency:.1%}")

    # Additional metrics
    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        redundancy = metrics.get("redundancy_rate", 0)
        st.metric("Redundancy Rate", f"{redundancy:.1%}")
    with col_b:
        tool_err = metrics.get("tool_error_rate", 0)
        st.metric("Tool Error Rate", f"{tool_err:.1%}")
