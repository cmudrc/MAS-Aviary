"""Generate 8 high-DPI agent timeline plots from batch_summary.json files."""

import json
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# --- Configuration ---
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

# Professional color palette (colorblind-friendly, muted tones)
ROLE_COLORS = {
    "orchestrator": "#5B7FA5",  # steel blue
    "mission_architect": "#6AAB9C",  # sage green
    "aerodynamics_analyst": "#E8A87C",  # warm peach
    "weights_analyst": "#D4A5A5",  # dusty rose
    "propulsion_analyst": "#9B8EC0",  # soft purple
    "simulation_executor": "#7DB8D6",  # sky blue
    "mdo_integrator": "#C9B458",  # muted gold
}
ROLE_COLORS["idle (echo)"] = "#D0D0D0"  # light gray for wasted turns
DEFAULT_COLOR = "#A0A0A0"

TOOL_MARKERS = {
    "create_session": "s",  # square
    "configure_mission": "D",  # diamond
    "get_design_space": "^",  # triangle up
    "set_aircraft_parameters": "o",  # circle
    "validate_parameters": "P",  # plus (filled)
    "run_simulation": "*",  # star
    "get_results": "X",  # x (filled)
    "check_constraints": "h",  # hexagon
    "write_blackboard": "v",  # triangle down
    "read_blackboard": "<",  # triangle left
    "mark_task_done": ">",  # triangle right
    "list_available_tools": "p",  # pentagon
    "create_agent": "+",  # plus
    "assign_task": "1",  # tri_down
    "list_graph_roles": "2",  # tri_up
    "final_answer": "d",  # thin diamond
}


def resolve_role(msg):
    """Determine the canonical role name for a message.

    For networked agents (agent_1, agent_2, agent_3), check metadata
    for graph_role or stage_name to get the actual role.
    For orchestrated/sequential, the agent_name is already the role.
    """
    name = msg["agent_name"]
    meta = msg.get("metadata", {})

    # graph_role is the most reliable for networked graph-routed
    if meta.get("graph_role"):
        return meta["graph_role"]

    # stage_name for staged pipeline
    if meta.get("stage_name"):
        stage = meta["stage_name"]
        # Filter out generic "stage_N" names from overflow turns
        if not stage.startswith("stage_"):
            return stage

    # For orchestrated: role might be in metadata
    if meta.get("role") == "orchestrator_review":
        return "orchestrator"

    # If it's already a named role, keep it
    if not name.startswith("agent_"):
        return name

    # Fallback: tool-based classification using official stage-gate mapping
    #   mission_architect:    create_session, configure_mission
    #   aerodynamics_analyst: get_design_space, set_aircraft_parameters, validate_parameters (wing)
    #   weights_analyst:      (no tools — text-based analysis)
    #   propulsion_analyst:   get_design_space, set_aircraft_parameters, validate_parameters (engine)
    #   simulation_executor:  run_simulation, get_results
    #   mdo_integrator:       check_constraints, get_results, mark_task_done
    #   orchestrator:         list_available_tools, create_agent, assign_task, list_graph_roles
    tools_used = {tc["tool_name"] for tc in msg.get("tool_calls", []) if tc["tool_name"] != "final_answer"}
    if not tools_used:
        return "idle (echo)"

    ORCHESTRATOR_TOOLS = {"list_available_tools", "create_agent", "assign_task", "list_graph_roles"}
    if tools_used & ORCHESTRATOR_TOOLS:
        return "orchestrator"
    if "check_constraints" in tools_used or "mark_task_done" in tools_used:
        return "mdo_integrator"
    if "run_simulation" in tools_used:
        return "simulation_executor"
    if "configure_mission" in tools_used:
        return "mission_architect"
    if "create_session" in tools_used:
        return "mission_architect"
    if "set_aircraft_parameters" in tools_used:
        for tc in msg.get("tool_calls", []):
            if tc["tool_name"] == "set_aircraft_parameters":
                params = tc.get("inputs", {}).get("parameters", {})
                param_keys = " ".join(params.keys()) if params else ""
                if ("Engine" in param_keys or "SCALE_FACTOR" in param_keys) and not any(
                    w in param_keys for w in ("Wing", "ASPECT", "SWEEP", "TAPER", "AREA", "SPAN")
                ):
                    return "propulsion_analyst"
                return "aerodynamics_analyst"
    if "validate_parameters" in tools_used or "get_design_space" in tools_used:
        return "aerodynamics_analyst"
    if "get_results" in tools_used:
        return "simulation_executor"

    return name  # last resort


def get_color(role):
    return ROLE_COLORS.get(role, DEFAULT_COLOR)


def load_batch(batch_dir):
    path = os.path.join(batch_dir, "batch_summary.json")
    with open(path) as f:
        data = json.load(f)
    return data["results"][0]


def build_timeline_data(result):
    """Extract timeline bars and tool events from a batch result."""
    messages = result["messages"]
    if not messages:
        return [], []

    # Find global time origin
    t0 = messages[0]["timestamp"] - messages[0].get("duration_seconds", 0)

    bars = []  # (role, start_s, end_s, turn_number)
    events = []  # (role, time_s, tool_name)

    for msg in messages:
        role = resolve_role(msg)
        end_t = msg["timestamp"] - t0
        dur = msg.get("duration_seconds", 0)
        start_t = end_t - dur

        bars.append((role, start_t, end_t, msg["turn_number"]))

        # Extract individual tool call timestamps
        # Tools don't have individual timestamps, so distribute them evenly
        tool_calls = [tc for tc in msg.get("tool_calls", []) if tc["tool_name"] != "final_answer"]
        if tool_calls:
            n = len(tool_calls)
            for i, tc in enumerate(tool_calls):
                # Place tool calls evenly within the agent's active window
                t = start_t + (i + 0.5) * dur / max(n, 1)
                events.append((role, t, tc["tool_name"]))

    return bars, events


def plot_timeline(key, result, output_dir):
    """Create a single timeline plot for one combination."""
    bars, events = build_timeline_data(result)
    if not bars:
        return

    label = COMBO_LABELS[key]
    fuel = result["eval_classification"]["fuel_burned_kg"]
    gtow = result["eval_classification"]["gtow_kg"]
    eval_result = result["eval_classification"]["result"]
    duration = result["duration_seconds"]
    turns = result["total_turns"]

    # Determine unique roles in order of first appearance
    seen = {}
    for role, start, end, turn in bars:
        if role not in seen:
            seen[role] = len(seen)
    roles = list(seen.keys())
    n_roles = len(roles)

    # Create figure
    fig_height = max(2.5, 0.6 * n_roles + 1.8)
    fig, ax = plt.subplots(figsize=(12, fig_height), dpi=600)

    # Plot bars
    bar_height = 0.55
    for role, start, end, turn in bars:
        y = seen[role]
        color = get_color(role)
        is_idle = role == "idle (echo)"
        ax.barh(
            y,
            end - start,
            left=start,
            height=bar_height,
            color=color,
            alpha=0.45 if is_idle else 0.85,
            edgecolor="#999" if is_idle else "white",
            linewidth=0.8 if is_idle else 0.5,
            hatch="///" if is_idle else None,
            zorder=2,
        )
        # Turn number label inside bar
        mid = (start + end) / 2
        bar_width = end - start
        if bar_width > duration * 0.04:  # Only label if bar is wide enough
            ax.text(
                mid,
                y,
                f"T{turn}",
                ha="center",
                va="center",
                fontsize=6,
                fontweight="bold",
                color="#666" if is_idle else "white",
                zorder=3,
            )

    # Plot tool events as markers
    tool_legend = {}
    for role, t, tool_name in events:
        y = seen[role]
        marker = TOOL_MARKERS.get(tool_name, ".")
        color = get_color(role)
        # Darker version for markers
        ax.plot(
            t,
            y + bar_height / 2 + 0.08,
            marker=marker,
            color="black",
            markersize=5,
            markeredgewidth=0.5,
            zorder=4,
            alpha=0.7,
        )
        if tool_name not in tool_legend:
            tool_legend[tool_name] = marker

    # Y-axis
    ax.set_yticks(range(n_roles))
    ax.set_yticklabels(roles, fontsize=8, fontfamily="monospace")
    ax.invert_yaxis()

    # X-axis
    ax.set_xlabel("Time (seconds)", fontsize=9, fontweight="bold")
    ax.set_xlim(-duration * 0.02, duration * 1.05)

    # Grid
    ax.xaxis.grid(True, alpha=0.2, linestyle="--")
    ax.set_axisbelow(True)

    # Title with metrics
    title = f"{label}"
    subtitle = f"fuel={fuel:.0f} kg  |  GTOW={gtow:.0f} kg  |  eval={eval_result}  |  {turns} turns  |  {duration:.0f}s"
    ax.set_title(f"{title}\n{subtitle}", fontsize=10, fontweight="bold", pad=10, loc="left")

    # Legend for roles
    role_patches = [mpatches.Patch(color=get_color(r), label=r, alpha=0.85) for r in roles]
    leg1 = ax.legend(
        handles=role_patches,
        loc="upper right",
        fontsize=6,
        framealpha=0.9,
        title="Roles",
        title_fontsize=7,
        ncol=min(3, len(roles)),
    )

    # Tool marker legend (separate, below main legend)
    if tool_legend:
        tool_handles = []
        for tname, marker in sorted(tool_legend.items()):
            h = plt.Line2D(
                [0],
                [0],
                marker=marker,
                color="black",
                linestyle="None",
                markersize=4,
                label=tname.replace("_", " "),
                alpha=0.7,
            )
            tool_handles.append(h)
        ax.legend(
            handles=tool_handles,
            loc="lower right",
            fontsize=5,
            framealpha=0.9,
            title="Tools",
            title_fontsize=6,
            ncol=min(3, len(tool_handles)),
        )
        ax.add_artist(leg1)

    # Style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)

    plt.tight_layout()

    # Save
    out_path = os.path.join(output_dir, f"{key}_timeline.png")
    fig.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {out_path}")


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
        plot_timeline(key, result, output_dir)

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
