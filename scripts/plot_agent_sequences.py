"""Generate 8 high-DPI turn-sequence plots showing actual agent names and handoff order.

Unlike the swimlane timeline plots, these show:
  - Y-axis = turn number (execution order, top to bottom)
  - X-axis = wall-clock time
  - Each bar = one turn, colored by the RAW agent name
  - Role annotations on each bar show what role the agent played
  - A mapping legend links raw agent names to their resolved roles
  - Tool icons placed along each bar

This makes the back-and-forth handoff pattern clearly visible.
"""

import json
import os

import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

# Reuse config from the other script
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

# Colors keyed by RAW agent name — unique per identity
AGENT_COLORS = {
    "orchestrator": "#5B7FA5",
    "mission_architect": "#6AAB9C",
    "aerodynamics_analyst": "#E8A87C",
    "weights_analyst": "#D4A5A5",
    "propulsion_analyst": "#9B8EC0",
    "simulation_executor": "#7DB8D6",
    "mdo_integrator": "#C9B458",
    "agent_1": "#E07A5F",  # terracotta
    "agent_2": "#3D85C6",  # cobalt
    "agent_3": "#81B29A",  # seafoam
}
DEFAULT_COLOR = "#A0A0A0"

TOOL_ICONS = {
    "create_session": "s",
    "configure_mission": "D",
    "get_design_space": "^",
    "set_aircraft_parameters": "o",
    "validate_parameters": "P",
    "run_simulation": "*",
    "get_results": "X",
    "check_constraints": "h",
    "write_blackboard": "v",
    "read_blackboard": "<",
    "mark_task_done": ">",
    "list_available_tools": "p",
    "create_agent": "+",
    "assign_task": "1",
    "list_graph_roles": "2",
}


def resolve_role(msg):
    """Get the role this agent played (for annotation), same logic as timeline script."""
    meta = msg.get("metadata", {})
    if meta.get("graph_role"):
        return meta["graph_role"]
    if meta.get("stage_name"):
        stage = meta["stage_name"]
        if not stage.startswith("stage_"):
            return stage
    if meta.get("role") == "orchestrator_review":
        return "orchestrator"
    name = msg["agent_name"]
    if not name.startswith("agent_"):
        return name
    # Tool-based classification using official stage-gate mapping
    tools_used = {tc["tool_name"] for tc in msg.get("tool_calls", []) if tc["tool_name"] != "final_answer"}
    if not tools_used:
        return "idle"

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
    return "unknown"


def load_batch(batch_dir):
    path = os.path.join(batch_dir, "batch_summary.json")
    with open(path) as f:
        return json.load(f)["results"][0]


def plot_sequence(key, result, output_dir):
    messages = result["messages"]
    if not messages:
        return

    label = COMBO_LABELS[key]
    fuel = result["eval_classification"]["fuel_burned_kg"]
    gtow = result["eval_classification"]["gtow_kg"]
    eval_result = result["eval_classification"]["result"]
    total_duration = result["duration_seconds"]
    n_turns = len(messages)

    # Compute time origin
    t0 = messages[0]["timestamp"] - messages[0].get("duration_seconds", 0)

    # Build per-turn data
    turns = []
    agent_role_map = {}  # raw_name -> set of roles
    for msg in messages:
        raw_name = msg["agent_name"]
        role = resolve_role(msg)
        end_t = msg["timestamp"] - t0
        dur = msg.get("duration_seconds", 0)
        start_t = end_t - dur
        tool_calls = [tc for tc in msg.get("tool_calls", []) if tc["tool_name"] != "final_answer"]
        turns.append(
            {
                "turn": msg["turn_number"],
                "agent": raw_name,
                "role": role,
                "start": start_t,
                "end": end_t,
                "dur": dur,
                "tools": tool_calls,
            }
        )
        agent_role_map.setdefault(raw_name, set()).add(role)

    # --- Figure setup ---
    bar_height = 0.65
    fig_height = max(3.0, 0.55 * n_turns + 2.8)
    fig, ax = plt.subplots(figsize=(12, fig_height), dpi=600)

    # Unique agents in order of first appearance
    agent_order = []
    for t in turns:
        if t["agent"] not in agent_order:
            agent_order.append(t["agent"])

    # --- Draw turns ---
    tool_legend = {}
    for i, t in enumerate(turns):
        y = i
        color = AGENT_COLORS.get(t["agent"], DEFAULT_COLOR)
        is_idle = t["role"] == "idle"

        # Main bar
        ax.barh(
            y,
            t["dur"],
            left=t["start"],
            height=bar_height,
            color=color,
            alpha=0.40 if is_idle else 0.88,
            edgecolor="#888" if is_idle else "#444",
            linewidth=0.6,
            hatch="///" if is_idle else None,
            zorder=2,
        )

        # Role label inside bar (right-aligned)
        bar_w = t["end"] - t["start"]
        if bar_w > total_duration * 0.06:
            role_label = t["role"]
            ax.text(
                t["start"] + bar_w * 0.5,
                y,
                role_label,
                ha="center",
                va="center",
                fontsize=5.5,
                fontstyle="italic",
                color="white" if not is_idle else "#555",
                fontweight="medium",
                path_effects=[pe.withStroke(linewidth=1.5, foreground="#333" if not is_idle else "#ccc")],
                zorder=5,
            )

        # Tool markers along top edge of bar
        n_tools = len(t["tools"])
        for j, tc in enumerate(t["tools"]):
            tx = t["start"] + (j + 0.5) * t["dur"] / max(n_tools, 1)
            marker = TOOL_ICONS.get(tc["tool_name"], ".")
            ax.plot(
                tx,
                y - bar_height * 0.38,
                marker=marker,
                color="#222",
                markersize=4.5,
                markeredgewidth=0.4,
                zorder=6,
                alpha=0.75,
            )
            if tc["tool_name"] not in tool_legend:
                tool_legend[tc["tool_name"]] = marker

        # Handoff arrow to next turn
        if i < n_turns - 1:
            next_t = turns[i + 1]
            if t["agent"] != next_t["agent"]:
                # Draw a thin arrow from end of this bar to start of next
                ax.annotate(
                    "",
                    xy=(next_t["start"], i + 1),
                    xytext=(t["end"], i),
                    arrowprops=dict(
                        arrowstyle="->,head_width=0.15,head_length=0.1",
                        color="#999",
                        lw=0.6,
                        connectionstyle="arc3,rad=0.15",
                    ),
                    zorder=1,
                )

    # --- Y-axis: turn labels with raw agent name ---
    y_labels = [f"T{t['turn']}  {t['agent']}" for t in turns]
    ax.set_yticks(range(n_turns))
    ax.set_yticklabels(y_labels, fontsize=7, fontfamily="monospace")
    ax.invert_yaxis()

    # Color the y-tick labels to match agent color
    for i, t in enumerate(turns):
        color = AGENT_COLORS.get(t["agent"], DEFAULT_COLOR)
        ax.get_yticklabels()[i].set_color(color)
        ax.get_yticklabels()[i].set_fontweight("bold")

    # --- X-axis ---
    ax.set_xlabel("Time (seconds)", fontsize=9, fontweight="bold")
    ax.set_xlim(-total_duration * 0.02, total_duration * 1.05)

    # Grid
    ax.xaxis.grid(True, alpha=0.15, linestyle="--")
    ax.set_axisbelow(True)

    # --- Title ---
    subtitle = (
        f"fuel={fuel:.0f} kg  |  GTOW={gtow:.0f} kg  |  "
        f"eval={eval_result}  |  {n_turns} turns  |  {total_duration:.0f}s"
    )
    ax.set_title(f"{label}  — Turn Sequence\n{subtitle}", fontsize=10, fontweight="bold", pad=10, loc="left")

    # --- Agent color legend ---
    agent_patches = []
    for a in agent_order:
        c = AGENT_COLORS.get(a, DEFAULT_COLOR)
        agent_patches.append(mpatches.Patch(color=c, label=a, alpha=0.88))
    leg1 = ax.legend(
        handles=agent_patches,
        loc="upper right",
        fontsize=6,
        framealpha=0.92,
        title="Agents",
        title_fontsize=7,
        ncol=min(3, len(agent_patches)),
    )

    # --- Role mapping box ---
    mapping_lines = []
    for a in agent_order:
        roles = sorted(agent_role_map.get(a, set()) - {"idle"})
        if roles and roles != [a]:
            mapping_lines.append(f"{a} -> {', '.join(roles)}")
    if mapping_lines:
        mapping_text = "\n".join(mapping_lines)
        props = dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="#ccc", linewidth=0.5)
        # Place in lower-left
        ax.text(
            0.01,
            0.01,
            mapping_text,
            transform=ax.transAxes,
            fontsize=5,
            fontfamily="monospace",
            verticalalignment="bottom",
            bbox=props,
            zorder=10,
        )

    # --- Tool legend ---
    if tool_legend:
        tool_handles = []
        for tname, marker in sorted(tool_legend.items()):
            h = plt.Line2D(
                [0],
                [0],
                marker=marker,
                color="#222",
                linestyle="None",
                markersize=4,
                label=tname.replace("_", " "),
                alpha=0.75,
            )
            tool_handles.append(h)
        ax.legend(
            handles=tool_handles,
            loc="lower right",
            fontsize=5,
            framealpha=0.92,
            title="Tools",
            title_fontsize=6,
            ncol=min(3, len(tool_handles)),
        )
        ax.add_artist(leg1)

    # --- Style ---
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)

    plt.tight_layout()

    out_path = os.path.join(output_dir, f"{key}_sequence.png")
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
        plot_sequence(key, result, output_dir)

    print(f"\nAll sequence plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
