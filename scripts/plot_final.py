"""Generate all 4 plot types for 8 combinations from logs/1Sample/ best runs.

Outputs to Final_Plots/ directory:
  - {N}_{combo}_tool_activity.png + .svg
  - {N}_{combo}_role_activity.png
  - {N}_{combo}_timeline.png
  - {N}_{combo}_sequence.png
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

# ============================================================
# Configuration
# ============================================================

SAMPLE_DIR = "logs/1Sample"

COMBOS = {
    "1_sequential_IF": "aviary_sequential_iterative_feedback",
    "2_sequential_SP": "aviary_sequential_staged_pipeline",
    "3_orchestrated_IF": "aviary_orchestrated_iterative_feedback",
    "4_orchestrated_SP": "aviary_orchestrated_staged_pipeline",
    "5_orchestrated_GR": "aviary_orchestrated_graph_routed",
    "6_networked_IF": "aviary_networked_iterative_feedback",
    "7_networked_SP": "aviary_networked_staged_pipeline",
    "8_networked_GR": "aviary_networked_graph_routed",
    "9_sequential_GR": "aviary_sequential_graph_routed",
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
    "9_sequential_GR": "Sequential + Graph-Routed",
}

# Tools grouped by stage-gate role
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
    ("Mission Architect", ["create_session", "configure_mission"]),
    ("Aero / Propulsion", ["get_design_space", "set_aircraft_parameters", "validate_parameters"]),
    ("Simulation Executor", ["run_simulation", "get_results"]),
    ("MDO Integrator", ["check_constraints", "mark_task_done"]),
    ("Blackboard", ["write_blackboard", "read_blackboard"]),
]

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
    "pre-hook": "#4A4A4A",
    "implicit": "#B0B0B0",
}
DEFAULT_COLOR = "#A0A0A0"

GROUP_COLORS = {
    "Orchestrator": "#E8EDF2",
    "Mission Architect": "#E8F4F0",
    "Aero / Propulsion": "#FDF0E6",
    "Simulation Executor": "#E6F2F8",
    "MDO Integrator": "#F5F0DC",
    "Blackboard": "#F0ECF5",
}

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

ROLE_COLORS = {
    "orchestrator": "#5B7FA5",
    "mission_architect": "#6AAB9C",
    "aerodynamics_analyst": "#E8A87C",
    "weights_analyst": "#D4A5A5",
    "propulsion_analyst": "#9B8EC0",
    "simulation_executor": "#7DB8D6",
    "mdo_integrator": "#C9B458",
    "idle (echo)": "#D0D0D0",
}

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
    "final_answer": "d",
}


# ============================================================
# Shared helpers
# ============================================================


def load_result(key):
    """Load best.json for a combination."""
    combo_dir = COMBOS[key]
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, SAMPLE_DIR, combo_dir, "best.json")
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        result = json.load(f)
    trace_path = os.path.join(base, SAMPLE_DIR, combo_dir, "best_trace.json")
    trace = None
    if os.path.exists(trace_path):
        with open(trace_path) as f:
            trace = json.load(f)
    return result, trace


def resolve_role(msg):
    """Determine functional role for a message."""
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

    tools_used = {tc["tool_name"] for tc in msg.get("tool_calls", []) if tc["tool_name"] != "final_answer"}
    if not tools_used:
        return "idle (echo)"

    ORCH_TOOLS = {"list_available_tools", "create_agent", "assign_task", "list_graph_roles"}
    if tools_used & ORCH_TOOLS:
        return "orchestrator"
    if "check_constraints" in tools_used or "mark_task_done" in tools_used:
        return "mdo_integrator"
    if "run_simulation" in tools_used:
        return "simulation_executor"
    if "configure_mission" in tools_used or "create_session" in tools_used:
        return "mission_architect"
    if "set_aircraft_parameters" in tools_used:
        for tc in msg.get("tool_calls", []):
            if tc["tool_name"] == "set_aircraft_parameters":
                params = tc.get("inputs", {}).get("parameters", {})
                pk = " ".join(params.keys()) if params else ""
                if ("Engine" in pk or "SCALE_FACTOR" in pk) and not any(
                    w in pk for w in ("Wing", "ASPECT", "SWEEP", "TAPER", "AREA", "SPAN")
                ):
                    return "propulsion_analyst"
                return "aerodynamics_analyst"
    if "validate_parameters" in tools_used or "get_design_space" in tools_used:
        return "aerodynamics_analyst"
    if "get_results" in tools_used:
        return "simulation_executor"
    return name


def extract_calls_from_trace(trace):
    """Extract tool calls from trace file.

    Prefers role-named keys (e.g. aerodynamics_analyst) over agent_N keys
    when both exist, because role names match the message agent_names and
    produce more informative legends.
    """
    agents_data = trace.get("agents", trace)

    # Detect which keys to use
    agent_n_keys = sorted(k for k in agents_data if k.startswith("agent_"))
    role_keys = sorted(
        k
        for k in agents_data
        if isinstance(agents_data[k], dict) and not k.startswith("agent_") and "steps" in agents_data[k]
    )

    # Count real tool calls per key set to pick the most complete one
    def _count_real_tools(keys):
        n = 0
        for k in keys:
            entry = agents_data[k]
            if not isinstance(entry, dict):
                continue
            for step in entry.get("steps", []):
                for tc in step.get("tool_calls", []):
                    if tc.get("name") != "final_answer":
                        n += 1
        return n

    n_agent = _count_real_tools(agent_n_keys)
    n_role = _count_real_tools(role_keys)

    # Prefer whichever set has more real tool calls (= more complete data).
    # When both exist with equal counts, prefer role keys for better labels.
    if agent_n_keys and role_keys:
        use_keys = role_keys if n_role >= n_agent else agent_n_keys
    elif agent_n_keys:
        use_keys = agent_n_keys
    elif role_keys:
        use_keys = role_keys
    else:
        return None

    all_steps = []
    for agent_name in use_keys:
        agent_entry = agents_data[agent_name]
        if not isinstance(agent_entry, dict):
            continue
        for step in agent_entry.get("steps", []):
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

    total_duration = calls[-1]["end"] if calls else 0
    return calls, agents_seen, total_duration, turn_counter


def extract_calls_from_messages(messages):
    """Extract tool calls from messages."""
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
                    "tool": tc["tool_name"],
                    "agent": agent,
                    "turn": msg["turn_number"],
                    "start": tc_start,
                    "end": tc_end,
                    "dur": tc_end - tc_start,
                }
            )
    return calls, agents_seen


def get_subtitle(result):
    ec = result.get("eval_classification", {})
    fuel = ec.get("fuel_burned_kg", 0)
    gtow = ec.get("gtow_kg", 0)
    eval_result = ec.get("result", "unknown")
    duration = result.get("duration_seconds", 0)
    n_turns = result.get("total_turns", len(result.get("messages", [])))
    return f"fuel={fuel:.0f} kg  |  GTOW={gtow:.0f} kg  |  eval={eval_result}  |  {n_turns} turns  |  {duration:.0f}s"


# ============================================================
# Plot 1: Tool Activity
# ============================================================

# ---- Y-axis tool lists ----

# Full superset: all tools from all groups (for uniform plots)
SUPERSET_TOOLS = []
SUPERSET_GROUP_SPANS = []
for _grp_name, _grp_tools in TOOL_GROUPS:
    _start = len(SUPERSET_TOOLS)
    SUPERSET_TOOLS.extend(_grp_tools)
    _end = len(SUPERSET_TOOLS)
    SUPERSET_GROUP_SPANS.append((_grp_name, _start, _end))

# Tiered: only the groups relevant to each org structure
# Sequential: Mission Architect + Aero/Propulsion + Simulation Executor + MDO Integrator
# Orchestrated: Orchestrator + Mission Architect + Aero/Propulsion + Sim Executor + MDO Integrator
# Networked: Mission Architect + Aero/Propulsion + Sim Executor + MDO Integrator + Blackboard

TIER_GROUPS = {
    "sequential": ["Mission Architect", "Aero / Propulsion", "Simulation Executor", "MDO Integrator"],
    "orchestrated": ["Orchestrator", "Mission Architect", "Aero / Propulsion", "Simulation Executor", "MDO Integrator"],
    "networked": ["Mission Architect", "Aero / Propulsion", "Simulation Executor", "MDO Integrator", "Blackboard"],
}

GROUP_TOOL_MAP = {name: tools for name, tools in TOOL_GROUPS}


def _build_tier(key):
    """Return (tool_list, group_spans) for the tier matching this combo key."""
    for tier_name in ("sequential", "orchestrated", "networked"):
        if tier_name in key:
            group_names = TIER_GROUPS[tier_name]
            break
    else:
        group_names = [name for name, _ in TOOL_GROUPS]

    tools = []
    spans = []
    for gn in group_names:
        start = len(tools)
        tools.extend(GROUP_TOOL_MAP[gn])
        spans.append((gn, start, len(tools)))
    return tools, spans


def _is_networked(key):
    return "networked" in key


def _remap_to_physical_agents(calls, agents_seen, messages):
    """Remap role-named agents back to physical agent_1/2/3 for networked combos.

    Uses rotation_index and total_agents from message metadata to determine
    which physical agent performed each role, making the rotation pattern visible.
    """
    total_agents = 3  # default
    # Build role_name -> physical agent mapping from metadata
    role_to_physical = {}
    for msg in messages:
        meta = msg.get("metadata", {})
        rot = meta.get("rotation_index")
        ta = meta.get("total_agents", total_agents)
        if rot is not None:
            total_agents = ta
            phys_idx = ((rot - 1) % total_agents) + 1
            agent_name = msg["agent_name"]
            role_to_physical[agent_name] = f"agent_{phys_idx}"

    if not role_to_physical:
        return calls, agents_seen

    # Remap all calls by agent name (role name -> physical agent)
    for c in calls:
        if c.get("synthetic") == "pre-hook":
            continue
        phys = role_to_physical.get(c["agent"])
        if phys:
            c["agent"] = phys

    # Rebuild agents_seen in order
    new_seen = []
    for c in calls:
        if c["agent"] not in new_seen:
            new_seen.append(c["agent"])
    return calls, new_seen


def _inject_prehook_calls(calls, agents_seen, global_x_max):
    """Add synthetic pre-hook bars for create_session + configure_mission at t<0."""
    prehook_w = global_x_max * 0.012
    synthetic = [
        {
            "tool": "create_session",
            "agent": "pre-hook",
            "turn": 0,
            "start": -prehook_w * 2.4,
            "end": -prehook_w * 1.4,
            "dur": prehook_w,
            "synthetic": "pre-hook",
        },
        {
            "tool": "configure_mission",
            "agent": "pre-hook",
            "turn": 0,
            "start": -prehook_w * 1.2,
            "end": -prehook_w * 0.2,
            "dur": prehook_w,
            "synthetic": "pre-hook",
        },
    ]
    if "pre-hook" not in agents_seen:
        agents_seen.insert(0, "pre-hook")
    return synthetic + calls


def _inject_implicit_reads(calls, messages, agents_seen, global_x_max):
    """Add synthetic read_blackboard bars at the start of each networked turn.

    Uses agent names already present in `calls` (already remapped to physical
    agent_1/2/3) so colors match the legend.
    """
    implicit_w = global_x_max * 0.008

    # Build a map: turn_number -> agent name used in calls for that turn
    # (already remapped to physical agents by _remap_to_physical_agents)
    turn_agent = {}
    for c in calls:
        if not c.get("synthetic"):
            turn_agent.setdefault(c["turn"], c["agent"])

    # Also build message-level timing
    t0 = messages[0]["timestamp"] - messages[0].get("duration_seconds", 0)
    synthetic = []
    for msg in messages:
        turn_end = msg["timestamp"] - t0
        turn_dur = msg.get("duration_seconds", 0)
        turn_start = turn_end - turn_dur
        # Prefer agent from calls (physical name); fallback to message agent_name
        agent = turn_agent.get(msg["turn_number"], msg["agent_name"])
        if agent not in agents_seen:
            agents_seen.append(agent)
        synthetic.append(
            {
                "tool": "read_blackboard",
                "agent": agent,
                "turn": msg["turn_number"],
                "start": turn_start,
                "end": turn_start + implicit_w,
                "dur": implicit_w,
                "synthetic": "implicit",
            }
        )
    return calls + synthetic


def _prepare_calls(key, result, trace, messages, global_x_max):
    """Extract tool calls and inject synthetic actions. Returns (calls, agents_seen)."""
    # Count real tool calls in messages for comparison
    msg_tool_count = sum(1 for m in messages for tc in m.get("tool_calls", []) if tc["tool_name"] != "final_answer")

    trace_result = None
    if trace:
        trace_result = extract_calls_from_trace(trace)

    # Use trace only if it has at least as many tool calls as messages
    if trace_result and len(trace_result[0]) >= msg_tool_count:
        calls, agents_seen, trace_dur, trace_turns = trace_result
    else:
        calls, agents_seen = extract_calls_from_messages(messages)

    if not calls:
        return None, None

    # For networked combos, remap role names to physical agent_1/2/3
    # so the rotation pattern is visible in the plot colors
    if _is_networked(key):
        calls, agents_seen = _remap_to_physical_agents(calls, agents_seen, messages)

    calls = _inject_prehook_calls(calls, agents_seen, global_x_max)

    if _is_networked(key):
        calls = _inject_implicit_reads(calls, messages, agents_seen, global_x_max)

    return calls, agents_seen


def _render_tool_activity(key, result, calls, agents_seen, tool_list, group_spans, global_x_max, output_dir, suffix=""):
    """Core rendering: plot tool activity with given y-axis tool list."""
    label = COMBO_LABELS[key]
    tools_used = {c["tool"] for c in calls}
    tool_to_y = {t: i for i, t in enumerate(tool_list)}
    n_tools = len(tool_list)

    bar_height = 0.55
    fig_height = max(3.0, 0.45 * n_tools + 2.2)
    fig, ax = plt.subplots(figsize=(14, fig_height), dpi=600)

    # Group background bands
    for group_name, y_start, y_end in group_spans:
        bg_color = GROUP_COLORS.get(group_name, "#f5f5f5")
        ax.axhspan(y_start - 0.5, y_end - 0.5, color=bg_color, zorder=0, alpha=0.7)

    # Group labels on the right
    for group_name, y_start, y_end in group_spans:
        y_mid = (y_start + y_end) / 2 - 0.5
        ax.text(
            global_x_max * 1.07,
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

    # Plot bars
    for c in calls:
        if c["tool"] not in tool_to_y:
            continue
        y = tool_to_y[c["tool"]]
        syn = c.get("synthetic")
        color = AGENT_COLORS.get(c["agent"], DEFAULT_COLOR)

        if syn == "pre-hook":
            bar_w = max(c["dur"], global_x_max * 0.008)
            ax.barh(
                y,
                bar_w,
                left=c["start"],
                height=bar_height,
                color=AGENT_COLORS["pre-hook"],
                alpha=0.6,
                edgecolor="#222",
                linewidth=0.6,
                hatch="///",
                zorder=4,
            )
        else:
            # Real and implicit reads use the SAME style: agent color, solid bar
            bar_w = max(c["dur"], global_x_max * 0.004)
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
            if bar_w > global_x_max * 0.025:
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

    # Connection lines (real calls only)
    agent_calls = {}
    for c in calls:
        if c.get("synthetic"):
            continue
        agent_calls.setdefault(c["agent"], []).append(c)
    for agent, acalls in agent_calls.items():
        color = AGENT_COLORS.get(agent, DEFAULT_COLOR)
        for k in range(len(acalls) - 1):
            c1, c2 = acalls[k], acalls[k + 1]
            if c1["tool"] not in tool_to_y or c2["tool"] not in tool_to_y:
                continue
            y1, y2 = tool_to_y[c1["tool"]], tool_to_y[c2["tool"]]
            if y1 != y2:
                ax.annotate(
                    "",
                    xy=(c2["start"], y2),
                    xytext=(c1["end"], y1),
                    arrowprops=dict(
                        arrowstyle="->,head_width=0.1,head_length=0.06",
                        color=color,
                        lw=0.5,
                        alpha=0.35,
                        connectionstyle="arc3,rad=0.15",
                    ),
                    zorder=1,
                )

    # Y-axis labels — grey out unused tools
    display_names = [t.replace("_", " ") for t in tool_list]
    ax.set_yticks(range(n_tools))
    ax.set_yticklabels(display_names, fontsize=9, fontfamily="monospace")
    for i, t in enumerate(tool_list):
        lbl = ax.get_yticklabels()[i]
        lbl.set_color("#222222" if t in tools_used else "#C0C0C0")
    ax.invert_yaxis()

    # Shared x-axis
    ax.set_xlabel("Time (seconds)", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", labelsize=9)
    ax.set_xlim(-global_x_max * 0.04, global_x_max * 1.18)
    ax.set_ylim(n_tools - 0.5, -0.5)
    ax.xaxis.grid(True, alpha=0.15, linestyle="--")
    ax.set_axisbelow(True)

    ax.set_title(
        f"{label}  — Tool Call Activity\n{get_subtitle(result)}", fontsize=12, fontweight="bold", pad=10, loc="left"
    )

    # Legend
    legend_agents = [a for a in agents_seen if a != "pre-hook"]
    agent_patches = [
        mpatches.Patch(color=AGENT_COLORS.get(a, DEFAULT_COLOR), label=a, alpha=0.85) for a in legend_agents
    ]
    prehook_patch = mpatches.Patch(
        facecolor=AGENT_COLORS["pre-hook"],
        alpha=0.6,
        edgecolor="#222",
        linewidth=0.6,
        hatch="///",
        label="pre-hook (framework)",
    )
    agent_patches.append(prehook_patch)

    ax.legend(
        handles=agent_patches,
        loc="upper right",
        fontsize=7.5,
        framealpha=0.92,
        title="Agent / Action Type",
        title_fontsize=8.5,
        ncol=min(4, len(agent_patches)),
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    plt.tight_layout()

    for ext in ("png", "svg"):
        fname = f"{key}_tool_activity{suffix}.{ext}"
        out = os.path.join(output_dir, fname)
        fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
        print(f"  Saved: {out}")
    plt.close(fig)


def plot_tool_activity(key, result, trace, output_dir, global_x_max):
    """Generate superset plot (all 15 tools)."""
    messages = result.get("messages", [])
    if not messages:
        return
    calls, agents_seen = _prepare_calls(key, result, trace, messages, global_x_max)
    if not calls:
        return
    _render_tool_activity(
        key, result, calls, agents_seen, SUPERSET_TOOLS, SUPERSET_GROUP_SPANS, global_x_max, output_dir
    )


def plot_tool_activity_tiered(key, result, trace, output_dir, global_x_max):
    """Generate tiered plot (only tools relevant to this org structure)."""
    messages = result.get("messages", [])
    if not messages:
        return
    calls, agents_seen = _prepare_calls(key, result, trace, messages, global_x_max)
    if not calls:
        return
    tier_tools, tier_spans = _build_tier(key)
    _render_tool_activity(
        key, result, calls, agents_seen, tier_tools, tier_spans, global_x_max, output_dir, suffix="_tiered"
    )


# ============================================================
# Plot 2: Role Activity
# ============================================================


def plot_role_activity(key, result, output_dir):
    messages = result.get("messages", [])
    if not messages:
        return

    label = COMBO_LABELS[key]
    total_duration = result.get("duration_seconds", 0)
    len(messages)
    t0 = messages[0]["timestamp"] - messages[0].get("duration_seconds", 0)

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
        if role not in ("idle", "idle (echo)", "unknown"):
            roles_seen.add(role)
        if raw_name not in agents_seen:
            agents_seen.append(raw_name)

    active_roles = [r for r in ROLE_ORDER if r in roles_seen]
    for r in sorted(roles_seen):
        if r not in active_roles:
            active_roles.append(r)

    role_to_y = {r: i for i, r in enumerate(active_roles)}
    n_roles = len(active_roles)

    bar_height = 0.55
    fig_height = max(3.5, 0.9 * n_roles + 2.5)
    fig, ax = plt.subplots(figsize=(14, fig_height), dpi=600)

    agent_paths = {}
    tool_legend = {}

    for t in turns:
        role = t["role"]
        if role in ("idle", "idle (echo)", "unknown") or role not in role_to_y:
            continue
        y = role_to_y[role]
        color = AGENT_COLORS.get(t["agent"], DEFAULT_COLOR)

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

        x_mid = (t["start"] + t["end"]) / 2
        agent_paths.setdefault(t["agent"], []).append((x_mid, y, t["start"], t["end"]))

    for agent, points in agent_paths.items():
        color = AGENT_COLORS.get(agent, DEFAULT_COLOR)
        for k in range(len(points) - 1):
            _, y1, _, x1_end = points[k]
            _, y2, x2_start, _ = points[k + 1]
            if y1 != y2:
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
                ax.plot([x1_end, x2_start], [y1, y2], color=color, lw=0.5, alpha=0.3, zorder=1)

    ax.set_yticks(range(n_roles))
    ax.set_yticklabels([ROLE_DISPLAY.get(r, r) for r in active_roles], fontsize=8, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Time (seconds)", fontsize=9, fontweight="bold")
    ax.set_xlim(-total_duration * 0.02, total_duration * 1.05)
    ax.xaxis.grid(True, alpha=0.15, linestyle="--")
    ax.yaxis.grid(True, alpha=0.08, linestyle="-")
    ax.set_axisbelow(True)

    ax.set_title(
        f"{label}  — Agent Role Activity\n{get_subtitle(result)}", fontsize=10, fontweight="bold", pad=10, loc="left"
    )

    agent_patches = [mpatches.Patch(color=AGENT_COLORS.get(a, DEFAULT_COLOR), label=a, alpha=0.85) for a in agents_seen]
    leg1 = ax.legend(
        handles=agent_patches,
        loc="upper right",
        fontsize=6.5,
        framealpha=0.92,
        title="Agent Identity",
        title_fontsize=7,
        ncol=min(3, len(agent_patches)),
    )

    if tool_legend:
        tool_handles = [
            plt.Line2D(
                [0], [0], marker=m, color="#222", linestyle="None", markersize=4, label=tn.replace("_", " "), alpha=0.75
            )
            for tn, m in sorted(tool_legend.items())
        ]
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

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)

    for i in range(n_roles):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#f5f5f5", zorder=0)

    plt.tight_layout()
    out = os.path.join(output_dir, f"{key}_role_activity.png")
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================
# Plot 3: Timeline
# ============================================================


def plot_timeline(key, result, output_dir):
    messages = result.get("messages", [])
    if not messages:
        return

    label = COMBO_LABELS[key]
    duration = result.get("duration_seconds", 0)
    result.get("total_turns", len(messages))
    t0 = messages[0]["timestamp"] - messages[0].get("duration_seconds", 0)

    bars = []
    events = []
    for msg in messages:
        role = resolve_role(msg)
        end_t = msg["timestamp"] - t0
        dur = msg.get("duration_seconds", 0)
        start_t = end_t - dur
        bars.append((role, start_t, end_t, msg["turn_number"]))
        tool_calls = [tc for tc in msg.get("tool_calls", []) if tc["tool_name"] != "final_answer"]
        for i, tc in enumerate(tool_calls):
            t = start_t + (i + 0.5) * dur / max(len(tool_calls), 1)
            events.append((role, t, tc["tool_name"]))

    seen = {}
    for role, _, _, _ in bars:
        if role not in seen:
            seen[role] = len(seen)
    roles = list(seen.keys())
    n_roles = len(roles)

    fig_height = max(2.5, 0.6 * n_roles + 1.8)
    fig, ax = plt.subplots(figsize=(12, fig_height), dpi=600)

    bar_height = 0.55
    for role, start, end, turn in bars:
        y = seen[role]
        color = ROLE_COLORS.get(role, DEFAULT_COLOR)
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
        mid = (start + end) / 2
        if (end - start) > duration * 0.04:
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

    tool_legend = {}
    for role, t, tool_name in events:
        y = seen[role]
        marker = TOOL_ICONS.get(tool_name, ".")
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

    ax.set_yticks(range(n_roles))
    ax.set_yticklabels(roles, fontsize=8, fontfamily="monospace")
    ax.invert_yaxis()
    ax.set_xlabel("Time (seconds)", fontsize=9, fontweight="bold")
    ax.set_xlim(-duration * 0.02, duration * 1.05)
    ax.xaxis.grid(True, alpha=0.2, linestyle="--")
    ax.set_axisbelow(True)

    ax.set_title(f"{label}\n{get_subtitle(result)}", fontsize=10, fontweight="bold", pad=10, loc="left")

    role_patches = [mpatches.Patch(color=ROLE_COLORS.get(r, DEFAULT_COLOR), label=r, alpha=0.85) for r in roles]
    leg1 = ax.legend(
        handles=role_patches,
        loc="upper right",
        fontsize=6,
        framealpha=0.9,
        title="Roles",
        title_fontsize=7,
        ncol=min(3, len(roles)),
    )

    if tool_legend:
        tool_handles = [
            plt.Line2D(
                [0], [0], marker=m, color="black", linestyle="None", markersize=4, label=tn.replace("_", " "), alpha=0.7
            )
            for tn, m in sorted(tool_legend.items())
        ]
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

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    plt.tight_layout()

    out = os.path.join(output_dir, f"{key}_timeline.png")
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================
# Plot 4: Sequence
# ============================================================


def plot_sequence(key, result, output_dir):
    messages = result.get("messages", [])
    if not messages:
        return

    label = COMBO_LABELS[key]
    total_duration = result.get("duration_seconds", 0)
    n_turns = len(messages)
    t0 = messages[0]["timestamp"] - messages[0].get("duration_seconds", 0)

    turns = []
    agent_role_map = {}
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

    bar_height = 0.65
    fig_height = max(3.0, 0.55 * n_turns + 2.8)
    fig, ax = plt.subplots(figsize=(12, fig_height), dpi=600)

    agent_order = []
    for t in turns:
        if t["agent"] not in agent_order:
            agent_order.append(t["agent"])

    tool_legend = {}
    for i, t in enumerate(turns):
        y = i
        color = AGENT_COLORS.get(t["agent"], DEFAULT_COLOR)
        is_idle = t["role"] in ("idle", "idle (echo)")

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

        bar_w = t["end"] - t["start"]
        if bar_w > total_duration * 0.06:
            ax.text(
                t["start"] + bar_w * 0.5,
                y,
                t["role"],
                ha="center",
                va="center",
                fontsize=5.5,
                fontstyle="italic",
                color="white" if not is_idle else "#555",
                fontweight="medium",
                path_effects=[pe.withStroke(linewidth=1.5, foreground="#333" if not is_idle else "#ccc")],
                zorder=5,
            )

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

        if i < n_turns - 1:
            next_t = turns[i + 1]
            if t["agent"] != next_t["agent"]:
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

    y_labels = [f"T{t['turn']}  {t['agent']}" for t in turns]
    ax.set_yticks(range(n_turns))
    ax.set_yticklabels(y_labels, fontsize=7, fontfamily="monospace")
    ax.invert_yaxis()

    for i, t in enumerate(turns):
        color = AGENT_COLORS.get(t["agent"], DEFAULT_COLOR)
        ax.get_yticklabels()[i].set_color(color)
        ax.get_yticklabels()[i].set_fontweight("bold")

    ax.set_xlabel("Time (seconds)", fontsize=9, fontweight="bold")
    ax.set_xlim(-total_duration * 0.02, total_duration * 1.05)
    ax.xaxis.grid(True, alpha=0.15, linestyle="--")
    ax.set_axisbelow(True)

    ax.set_title(
        f"{label}  — Turn Sequence\n{get_subtitle(result)}", fontsize=10, fontweight="bold", pad=10, loc="left"
    )

    agent_patches = [mpatches.Patch(color=AGENT_COLORS.get(a, DEFAULT_COLOR), label=a, alpha=0.88) for a in agent_order]
    leg1 = ax.legend(
        handles=agent_patches,
        loc="upper right",
        fontsize=6,
        framealpha=0.92,
        title="Agents",
        title_fontsize=7,
        ncol=min(3, len(agent_patches)),
    )

    mapping_lines = []
    for a in agent_order:
        roles = sorted(agent_role_map.get(a, set()) - {"idle", "idle (echo)"})
        if roles and roles != [a]:
            mapping_lines.append(f"{a} -> {', '.join(roles)}")
    if mapping_lines:
        props = dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="#ccc", linewidth=0.5)
        ax.text(
            0.01,
            0.01,
            "\n".join(mapping_lines),
            transform=ax.transAxes,
            fontsize=5,
            fontfamily="monospace",
            verticalalignment="bottom",
            bbox=props,
            zorder=10,
        )

    if tool_legend:
        tool_handles = [
            plt.Line2D(
                [0], [0], marker=m, color="#222", linestyle="None", markersize=4, label=tn.replace("_", " "), alpha=0.75
            )
            for tn, m in sorted(tool_legend.items())
        ]
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

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    plt.tight_layout()

    out = os.path.join(output_dir, f"{key}_sequence.png")
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================
# Main
# ============================================================


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base, "Final_Plots")
    os.makedirs(output_dir, exist_ok=True)

    # Pass 1: load all results and find global max duration for shared x-axis
    loaded = {}
    global_x_max = 0.0
    for key in COMBOS:
        result, trace = load_result(key)
        if result is None:
            print(f"  SKIP {key}: no best.json found")
            continue
        # Determine effective duration (take the larger of result vs trace)
        dur = result.get("duration_seconds", 0)
        if trace:
            tr = extract_calls_from_trace(trace)
            if tr and tr[2]:
                dur = max(dur, tr[2])
        global_x_max = max(global_x_max, dur)
        loaded[key] = (result, trace)

    print(f"Global x-axis max: {global_x_max:.0f}s")
    print(f"Superset y-axis: {len(SUPERSET_TOOLS)} tools")

    # Pass 2: generate both superset and tiered plots
    for key, (result, trace) in loaded.items():
        print(f"\n=== {COMBO_LABELS[key]} ===")
        plot_tool_activity(key, result, trace, output_dir, global_x_max)
        plot_tool_activity_tiered(key, result, trace, output_dir, global_x_max)

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
