"""Generate 8 high-DPI role-activity plots showing agent movement across functional roles.

Y-axis = Aviary functional roles (mission_architect, aerodynamics_analyst, etc.)
X-axis = wall-clock time
Bars colored by AGENT IDENTITY — so you can see agent_1 jumping between role lanes,
making the back-and-forth handoff pattern clearly visible.

This is distinct from the swimlane timeline (which merges agents into roles) and the
turn-sequence view (which is ordered by turn number).
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

# Canonical Aviary roles in display order (top to bottom)
ROLE_ORDER = [
    "orchestrator",
    "mission_architect",
    "aerodynamics_analyst",
    "weights_analyst",
    "propulsion_analyst",
    "simulation_executor",
    "mdo_integrator",
]

ROLE_DISPLAY = {
    "orchestrator": "Orchestrator",
    "mission_architect": "Mission Architect",
    "aerodynamics_analyst": "Aerodynamics Analyst",
    "weights_analyst": "Weights Analyst",
    "propulsion_analyst": "Propulsion Analyst",
    "simulation_executor": "Simulation Executor",
    "mdo_integrator": "MDO Integrator",
}

# Colors keyed by AGENT IDENTITY — track who is doing what
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
    """Determine which functional role this turn belongs to.

    Uses the official Aviary stage-gate tool-to-role mapping:
      mission_architect:    create_session, configure_mission
      aerodynamics_analyst: get_design_space, set_aircraft_parameters, validate_parameters (wing)
      weights_analyst:      (no tools — text-based analysis)
      propulsion_analyst:   get_design_space, set_aircraft_parameters, validate_parameters (engine)
      simulation_executor:  run_simulation, get_results
      mdo_integrator:       check_constraints, get_results, mark_task_done
      orchestrator:         list_available_tools, create_agent, assign_task, list_graph_roles
    """
    meta = msg.get("metadata", {})

    # 1) Explicit metadata from graph-routed or staged-pipeline handlers
    if meta.get("graph_role"):
        return meta["graph_role"]
    if meta.get("stage_name"):
        stage = meta["stage_name"]
        if not stage.startswith("stage_"):
            return stage
    if meta.get("role") == "orchestrator_review":
        return "orchestrator"

    # 2) Named agents (not generic agent_1/2/3) are their own role
    name = msg["agent_name"]
    if not name.startswith("agent_"):
        return name

    # 3) Tool-based classification using official stage-gate mapping
    tools_used = {tc["tool_name"] for tc in msg.get("tool_calls", []) if tc["tool_name"] != "final_answer"}
    if not tools_used:
        return "idle"

    # Orchestrator management tools (highest priority — these are meta-tools)
    ORCHESTRATOR_TOOLS = {"list_available_tools", "create_agent", "assign_task", "list_graph_roles"}
    if tools_used & ORCHESTRATOR_TOOLS:
        return "orchestrator"

    # MDO Integrator: check_constraints or mark_task_done
    if "check_constraints" in tools_used or "mark_task_done" in tools_used:
        return "mdo_integrator"

    # Simulation Executor: run_simulation (get_results often co-occurs)
    if "run_simulation" in tools_used:
        return "simulation_executor"

    # Mission Architect: configure_mission (create_session often co-occurs)
    if "configure_mission" in tools_used:
        return "mission_architect"

    # create_session alone (without configure_mission) → mission_architect
    # since session init is the architect's first step
    if "create_session" in tools_used:
        return "mission_architect"

    # set_aircraft_parameters: distinguish aero vs propulsion by param names
    if "set_aircraft_parameters" in tools_used:
        for tc in msg.get("tool_calls", []):
            if tc["tool_name"] == "set_aircraft_parameters":
                params = tc.get("inputs", {}).get("parameters", {})
                param_keys = " ".join(params.keys()) if params else ""
                # Engine/SCALE_FACTOR → propulsion; everything else → aero
                if ("Engine" in param_keys or "SCALE_FACTOR" in param_keys) and not any(
                    w in param_keys for w in ("Wing", "ASPECT", "SWEEP", "TAPER", "AREA", "SPAN")
                ):
                    return "propulsion_analyst"
                return "aerodynamics_analyst"

    # validate_parameters or get_design_space alone (without set_aircraft_parameters)
    # → likely aero prep work, but could be propulsion; default to aero
    if "validate_parameters" in tools_used or "get_design_space" in tools_used:
        return "aerodynamics_analyst"

    # get_results alone (without run_simulation or check_constraints)
    if "get_results" in tools_used:
        return "simulation_executor"

    return "unknown"


def load_batch(batch_dir):
    path = os.path.join(batch_dir, "batch_summary.json")
    with open(path) as f:
        return json.load(f)["results"][0]


def plot_role_activity(key, result, output_dir):
    messages = result["messages"]
    if not messages:
        return

    label = COMBO_LABELS[key]
    fuel = result["eval_classification"]["fuel_burned_kg"]
    gtow = result["eval_classification"]["gtow_kg"]
    eval_result = result["eval_classification"]["result"]
    total_duration = result["duration_seconds"]
    n_turns = len(messages)

    t0 = messages[0]["timestamp"] - messages[0].get("duration_seconds", 0)

    # Build turn data with role assignment
    turns = []
    roles_seen = set()
    agents_seen = []
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
        if role != "idle" and role != "unknown":
            roles_seen.add(role)
        if raw_name not in agents_seen:
            agents_seen.append(raw_name)

    # Filter ROLE_ORDER to only roles that appear, keep canonical order
    active_roles = [r for r in ROLE_ORDER if r in roles_seen]
    # Add any roles not in ROLE_ORDER
    for r in sorted(roles_seen):
        if r not in active_roles:
            active_roles.append(r)

    role_to_y = {r: i for i, r in enumerate(active_roles)}
    n_roles = len(active_roles)

    # --- Figure ---
    bar_height = 0.55
    fig_height = max(3.5, 0.9 * n_roles + 2.5)
    fig, ax = plt.subplots(figsize=(14, fig_height), dpi=600)

    # Track which (agent, role) combos we've drawn for the connection lines

    # Draw bars and collect path points per agent
    agent_paths = {}  # agent -> list of (x_mid, y)
    tool_legend = {}

    for i, t in enumerate(turns):
        role = t["role"]
        if role in ("idle", "unknown"):
            continue
        if role not in role_to_y:
            continue

        y = role_to_y[role]
        color = AGENT_COLORS.get(t["agent"], DEFAULT_COLOR)

        # Main bar
        ax.barh(
            y,
            t["dur"],
            left=t["start"],
            height=bar_height,
            color=color,
            alpha=0.85,
            edgecolor="#333",
            linewidth=0.5,
            zorder=3,
        )

        # Turn number label inside bar
        bar_w = t["end"] - t["start"]
        if bar_w > total_duration * 0.03:
            ax.text(
                t["start"] + bar_w * 0.5,
                y,
                f"T{t['turn']}",
                ha="center",
                va="center",
                fontsize=5.5,
                color="white",
                fontweight="bold",
                path_effects=[pe.withStroke(linewidth=1.5, foreground="#222")],
                zorder=6,
            )

        # Tool markers below bar
        n_tools = len(t["tools"])
        for j, tc in enumerate(t["tools"]):
            tx = t["start"] + (j + 0.5) * t["dur"] / max(n_tools, 1)
            marker = TOOL_ICONS.get(tc["tool_name"], ".")
            ax.plot(
                tx,
                y + bar_height * 0.42,
                marker=marker,
                color="#222",
                markersize=4,
                markeredgewidth=0.3,
                zorder=7,
                alpha=0.7,
            )
            if tc["tool_name"] not in tool_legend:
                tool_legend[tc["tool_name"]] = marker

        # Track path for connection lines
        x_mid = (t["start"] + t["end"]) / 2
        agent_paths.setdefault(t["agent"], []).append((x_mid, y, t["start"], t["end"]))

    # Draw connection lines showing agent movement across roles
    for agent, points in agent_paths.items():
        color = AGENT_COLORS.get(agent, DEFAULT_COLOR)
        for k in range(len(points) - 1):
            x1_mid, y1, _, x1_end = points[k]
            x2_mid, y2, x2_start, _ = points[k + 1]
            if y1 != y2:
                # Agent jumped to a different role — draw a connecting line
                ax.annotate(
                    "",
                    xy=(x2_start, y2),
                    xytext=(x1_end, y1),
                    arrowprops=dict(
                        arrowstyle="->,head_width=0.12,head_length=0.08",
                        color=color,
                        lw=0.8,
                        alpha=0.5,
                        connectionstyle="arc3,rad=0.2",
                    ),
                    zorder=2,
                )
            else:
                # Same role, draw a thin connector
                ax.plot([x1_end, x2_start], [y1, y2], color=color, lw=0.5, alpha=0.3, zorder=1)

    # --- Y-axis: role names ---
    ax.set_yticks(range(n_roles))
    ax.set_yticklabels([ROLE_DISPLAY.get(r, r) for r in active_roles], fontsize=8, fontweight="bold")
    ax.invert_yaxis()

    # --- X-axis ---
    ax.set_xlabel("Time (seconds)", fontsize=9, fontweight="bold")
    ax.set_xlim(-total_duration * 0.02, total_duration * 1.05)

    # Grid
    ax.xaxis.grid(True, alpha=0.15, linestyle="--")
    ax.yaxis.grid(True, alpha=0.08, linestyle="-")
    ax.set_axisbelow(True)

    # --- Title ---
    subtitle = (
        f"fuel={fuel:.0f} kg  |  GTOW={gtow:.0f} kg  |  "
        f"eval={eval_result}  |  {n_turns} turns  |  {total_duration:.0f}s"
    )
    ax.set_title(f"{label}  — Agent Role Activity\n{subtitle}", fontsize=10, fontweight="bold", pad=10, loc="left")

    # --- Agent identity legend ---
    agent_patches = []
    for a in agents_seen:
        c = AGENT_COLORS.get(a, DEFAULT_COLOR)
        agent_patches.append(mpatches.Patch(color=c, label=a, alpha=0.85))
    leg1 = ax.legend(
        handles=agent_patches,
        loc="upper right",
        fontsize=6.5,
        framealpha=0.92,
        title="Agent Identity",
        title_fontsize=7,
        ncol=min(3, len(agent_patches)),
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

    # Add light horizontal band alternation for readability
    for i in range(n_roles):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#f5f5f5", zorder=0)

    plt.tight_layout()

    out_path = os.path.join(output_dir, f"{key}_role_activity.png")
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
        plot_role_activity(key, result, output_dir)

    print(f"\nAll role-activity plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
