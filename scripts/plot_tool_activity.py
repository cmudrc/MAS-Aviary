"""Generate 8 high-DPI tool-activity plots showing individual tool calls over time.

Y-axis = individual Aviary tools, grouped by stage-gate role
X-axis = wall-clock time
Each marker/bar = one tool call, colored by AGENT IDENTITY

This makes the back-and-forth visible at the finest granularity: you see
agent_1 zigzagging across tool lanes, revisiting tools, etc.
"""

import json
import os

import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

BATCH_DIRS = {
    "1_sequential_IF": "logs/batch_results/1772821557",
    "2_sequential_SP": "logs/batch_results/1772823230",
    "3_orchestrated_IF": "logs/batch_results/1772823744",
    "4_orchestrated_SP": "logs/batch_results/1772824392",
    "5_orchestrated_GR": "logs/batch_results/1772824881",
    "6_networked_IF": "logs/batch_results/1772830187",
    "7_networked_SP": "logs/batch_results/1772833044",
    "8_networked_GR": "logs/batch_results/1772834243",
}

COMBO_LABELS = {
    "1_sequential_IF": "Sequential + Iterative Feedback",
    "2_sequential_SP": "Sequential + Staged Pipeline",
    "3_orchestrated_IF": "Orchestrated + Iterative Feedback",
    "4_orchestrated_SP": "Orchestrated + Staged Pipeline",
    "5_orchestrated_GR": "Orchestrated + Graph-Routed",
    "6_networked_IF": "Networked + Iterative Feedback",
    "7_networked_SP": "Networked + Staged Pipeline",
    "8_networked_GR": "Networked + Graph-Routed",
}

# Tools grouped by stage-gate role, in pipeline order
TOOL_GROUPS = [
    (
        "Orchestrator",
        [
            "list_available_tools",
            "list_graph_roles",
            "create_agent",
            "assign_task",
        ],
    ),
    (
        "Mission Architect",
        [
            "create_session",
            "configure_mission",
        ],
    ),
    (
        "Aero / Propulsion",
        [
            "get_design_space",
            "set_aircraft_parameters",
            "validate_parameters",
        ],
    ),
    (
        "Simulation Executor",
        [
            "run_simulation",
            "get_results",
        ],
    ),
    (
        "MDO Integrator",
        [
            "check_constraints",
            "mark_task_done",
        ],
    ),
    (
        "Blackboard",
        [
            "write_blackboard",
            "read_blackboard",
        ],
    ),
]

# Flatten to get canonical order
ALL_TOOLS_ORDERED = []
for _, tools in TOOL_GROUPS:
    ALL_TOOLS_ORDERED.extend(tools)

AGENT_COLORS = {
    "orchestrator": "#5B7FA5",
    "mission_architect": "#6AAB9C",
    "aerodynamics_analyst": "#E8A87C",
    "weights_analyst": "#D4A5A5",
    "propulsion_analyst": "#9B8EC0",
    "simulation_executor": "#7DB8D6",
    "mdo_integrator": "#C9B458",
    "agent_1": "#E07A5F",
    "agent_2": "#3D85C6",
    "agent_3": "#81B29A",
}
DEFAULT_COLOR = "#A0A0A0"

# Group divider colors
GROUP_COLORS = {
    "Orchestrator": "#E8EDF2",
    "Mission Architect": "#E8F4F0",
    "Aero / Propulsion": "#FDF0E6",
    "Simulation Executor": "#E6F2F8",
    "MDO Integrator": "#F5F0DC",
    "Blackboard": "#F0ECF5",
}


def load_batch(batch_dir):
    path = os.path.join(batch_dir, "batch_summary.json")
    with open(path) as f:
        return json.load(f)["results"][0]


def find_trace_file(batch_dir):
    """Find the trace JSON file in a batch directory, if it exists."""
    import glob as globmod

    traces = globmod.glob(os.path.join(batch_dir, "*_trace.json"))
    return traces[0] if traces else None


def extract_calls_from_trace(trace_path):
    """Extract tool calls from trace file using agent_N keys (original identities).

    Returns (calls, agents_seen, t0, total_duration, n_turns) or None if no agent_N data.
    """
    with open(trace_path) as f:
        trace = json.load(f)

    agents_data = trace.get("agents", {})
    agent_n_keys = sorted(k for k in agents_data if k.startswith("agent_"))
    if not agent_n_keys:
        return None

    # Collect all steps with timing from agent_N keys
    all_steps = []
    for agent_name in agent_n_keys:
        for step in agents_data[agent_name].get("steps", []):
            start = step.get("start_time")
            if start is None:
                continue
            dur = step.get("duration_seconds", 0) or 0
            for tc in step.get("tool_calls", []):
                tool_name = tc.get("name", "unknown")
                if tool_name == "final_answer":
                    continue
                all_steps.append(
                    {
                        "agent": agent_name,
                        "tool": tool_name,
                        "start_abs": start,
                        "dur": dur,
                        "step_num": step.get("step_number", 0),
                    }
                )

    if not all_steps:
        return None

    all_steps.sort(key=lambda s: s["start_abs"])
    t0 = all_steps[0]["start_abs"]

    calls = []
    agents_seen = []
    turn_counter = 0
    prev_agent = None
    for s in all_steps:
        if s["agent"] != prev_agent:
            turn_counter += 1
            prev_agent = s["agent"]
        if s["agent"] not in agents_seen:
            agents_seen.append(s["agent"])
        calls.append(
            {
                "tool": s["tool"],
                "agent": s["agent"],
                "turn": turn_counter,
                "start": s["start_abs"] - t0,
                "end": s["start_abs"] - t0 + s["dur"],
                "dur": s["dur"],
            }
        )

    total_duration = trace.get("duration_seconds", 0)
    if not total_duration and calls:
        total_duration = calls[-1]["end"]
    n_turns = turn_counter

    return calls, agents_seen, total_duration, n_turns


def extract_calls_from_messages(messages):
    """Extract tool calls from batch_summary messages (non-networked runs)."""
    t0 = messages[0]["timestamp"] - messages[0].get("duration_seconds", 0)

    calls = []
    agents_seen = []
    for msg in messages:
        agent = msg["agent_name"]
        if agent not in agents_seen:
            agents_seen.append(agent)
        turn_end = msg["timestamp"] - t0
        turn_dur = msg.get("duration_seconds", 0)
        turn_start = turn_end - turn_dur

        tool_calls = [tc for tc in msg.get("tool_calls", []) if tc["tool_name"] != "final_answer"]
        n_tc = len(tool_calls)
        for j, tc in enumerate(tool_calls):
            tool_name = tc["tool_name"]
            tc_dur = tc.get("duration_seconds", 0) or 0
            if n_tc > 1:
                tc_start = turn_start + j * (turn_dur / n_tc)
                tc_end = tc_start + turn_dur / n_tc
            else:
                tc_start = turn_start
                tc_end = turn_end
            if tc_dur > 0 and tc_dur < (tc_end - tc_start):
                tc_start = tc_end - tc_dur

            calls.append(
                {
                    "tool": tool_name,
                    "agent": agent,
                    "turn": msg["turn_number"],
                    "start": tc_start,
                    "end": tc_end,
                    "dur": tc_end - tc_start,
                }
            )

    return calls, agents_seen


def plot_tool_activity(key, result, output_dir, batch_dir):
    messages = result["messages"]
    if not messages:
        return

    label = COMBO_LABELS[key]
    fuel = result["eval_classification"]["fuel_burned_kg"]
    gtow = result["eval_classification"]["gtow_kg"]
    eval_result = result["eval_classification"]["result"]
    total_duration = result["duration_seconds"]
    n_turns = len(messages)

    # Try trace file first for original agent_N identities (networked runs)
    trace_path = find_trace_file(batch_dir)
    trace_result = None
    if trace_path:
        trace_result = extract_calls_from_trace(trace_path)

    if trace_result:
        calls, agents_seen, trace_dur, trace_turns = trace_result
        if trace_dur:
            total_duration = trace_dur
        if trace_turns:
            n_turns = trace_turns
    else:
        calls, agents_seen = extract_calls_from_messages(messages)

    if not calls:
        return

    # Determine which tools were actually used
    tools_used = {c["tool"] for c in calls}

    # Build Y-axis: only include tools that were actually called, keep group order
    active_tools = []
    group_spans = []  # (group_name, y_start, y_end)
    for group_name, group_tools in TOOL_GROUPS:
        group_start = len(active_tools)
        for t in group_tools:
            if t in tools_used:
                active_tools.append(t)
        group_end = len(active_tools)
        if group_end > group_start:
            group_spans.append((group_name, group_start, group_end))

    # Add any tools not in our predefined groups
    for t in sorted(tools_used):
        if t not in active_tools:
            active_tools.append(t)

    tool_to_y = {t: i for i, t in enumerate(active_tools)}
    n_tools = len(active_tools)

    # --- Figure ---
    bar_height = 0.55
    fig_height = max(3.0, 0.45 * n_tools + 2.2)
    fig, ax = plt.subplots(figsize=(14, fig_height), dpi=600)

    # Draw group background bands
    for group_name, y_start, y_end in group_spans:
        bg_color = GROUP_COLORS.get(group_name, "#f5f5f5")
        ax.axhspan(y_start - 0.5, y_end - 0.5, color=bg_color, zorder=0, alpha=0.7)

    # Draw group labels on right side
    for group_name, y_start, y_end in group_spans:
        y_mid = (y_start + y_end) / 2 - 0.5
        ax.text(
            total_duration * 1.07,
            y_mid,
            group_name,
            ha="left",
            va="center",
            fontsize=8,
            fontstyle="italic",
            color="#555",
            fontweight="medium",
            zorder=10,
        )

    # Draw tool calls as bars
    for c in calls:
        if c["tool"] not in tool_to_y:
            continue
        y = tool_to_y[c["tool"]]
        color = AGENT_COLORS.get(c["agent"], DEFAULT_COLOR)

        # Bar for this tool call
        bar_w = max(c["dur"], total_duration * 0.004)  # minimum visible width
        ax.barh(
            y,
            bar_w,
            left=c["start"],
            height=bar_height,
            color=color,
            alpha=0.85,
            edgecolor="#333",
            linewidth=0.4,
            zorder=3,
        )

        # Turn number inside bar if wide enough
        if bar_w > total_duration * 0.035:
            ax.text(
                c["start"] + bar_w * 0.5,
                y,
                f"T{c['turn']}",
                ha="center",
                va="center",
                fontsize=7,
                color="white",
                fontweight="bold",
                path_effects=[pe.withStroke(linewidth=1.4, foreground="#222")],
                zorder=6,
            )

    # Draw connection lines between consecutive tool calls per agent
    agent_calls = {}
    for c in calls:
        agent_calls.setdefault(c["agent"], []).append(c)

    for agent, acalls in agent_calls.items():
        color = AGENT_COLORS.get(agent, DEFAULT_COLOR)
        for k in range(len(acalls) - 1):
            c1 = acalls[k]
            c2 = acalls[k + 1]
            if c1["tool"] not in tool_to_y or c2["tool"] not in tool_to_y:
                continue
            y1 = tool_to_y[c1["tool"]]
            y2 = tool_to_y[c2["tool"]]
            x1 = c1["end"]
            x2 = c2["start"]
            if y1 != y2:
                ax.annotate(
                    "",
                    xy=(x2, y2),
                    xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle="->,head_width=0.1,head_length=0.06",
                        color=color,
                        lw=0.5,
                        alpha=0.35,
                        connectionstyle="arc3,rad=0.15",
                    ),
                    zorder=1,
                )

    # --- Y-axis ---
    display_names = [t.replace("_", " ") for t in active_tools]
    ax.set_yticks(range(n_tools))
    ax.set_yticklabels(display_names, fontsize=9, fontfamily="monospace")
    ax.invert_yaxis()

    # --- X-axis ---
    ax.set_xlabel("Time (seconds)", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", labelsize=9)
    ax.set_xlim(-total_duration * 0.02, total_duration * 1.18)

    # Grid
    ax.xaxis.grid(True, alpha=0.15, linestyle="--")
    ax.set_axisbelow(True)

    # --- Title ---
    subtitle = (
        f"fuel={fuel:.0f} kg  |  GTOW={gtow:.0f} kg  |  "
        f"eval={eval_result}  |  {n_turns} turns  |  "
        f"{len(calls)} tool calls  |  {total_duration:.0f}s"
    )
    ax.set_title(f"{label}  — Tool Call Activity\n{subtitle}", fontsize=12, fontweight="bold", pad=10, loc="left")

    # --- Agent legend ---
    agent_patches = []
    for a in agents_seen:
        c = AGENT_COLORS.get(a, DEFAULT_COLOR)
        agent_patches.append(mpatches.Patch(color=c, label=a, alpha=0.85))
    ax.legend(
        handles=agent_patches,
        loc="upper right",
        fontsize=8,
        framealpha=0.92,
        title="Agent Identity",
        title_fontsize=9,
        ncol=min(3, len(agent_patches)),
    )

    # --- Style ---
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)

    plt.tight_layout()

    out_png = os.path.join(output_dir, f"{key}_tool_activity.png")
    out_svg = os.path.join(output_dir, f"{key}_tool_activity.svg")
    fig.savefig(out_png, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    fig.savefig(out_svg, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {out_png}")
    print(f"  Saved: {out_svg}")


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base, "figures")
    os.makedirs(output_dir, exist_ok=True)

    for key, batch_rel in BATCH_DIRS.items():
        batch_dir = os.path.join(base, batch_rel)
        if not os.path.exists(batch_dir):
            print(f"  SKIP: {batch_dir} not found")
            continue
        print(f"Plotting {COMBO_LABELS[key]}...")
        result = load_batch(batch_dir)
        plot_tool_activity(key, result, output_dir, batch_dir)

    print(f"\nAll tool-activity plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
