"""Blackboard Viewer -- chat-room style visualization of agent coordination.

Shows which agent is doing what over time in a chronological, chat-room
layout.  Designed for networked strategy traces where agents coordinate
via read_blackboard / write_blackboard tool calls and MCP tools.

Structural blackboard writes (auto-posted by next_step() after each turn)
are recovered by parsing the 'Current Blackboard State:' section from
each agent's model_input_messages -- so the viewer shows the real board
state even when agents never called write_blackboard explicitly.

Launch with:
    streamlit run src/ui/blackboard_viewer.py
"""

import json
import os
import re
import sys
from dataclasses import dataclass
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

_ENTRY_TYPE_COLORS = {
    "status": "#3498db",
    "claim": "#e67e22",
    "result": "#2ecc71",
    "gap": "#e74c3c",
    "prediction": "#9b59b6",
}

_TOOL_ICONS = {
    "run_simulation": "🔧",
    "get_results": "📊",
    "check_output_files": "📁",
    "get_prompt": "📝",
    "list_prompts": "📋",
    "write_blackboard": "✏️",
    "read_blackboard": "👁️",
    "spawn_peer": "➕",
    "mark_task_done": "✅",
    "final_answer": "🏁",
}

# Regex to parse one blackboard entry from the context string.
# Format: [TYPE] key (by author, vN): value
_BB_ENTRY_RE = re.compile(
    r"\[(\w+)\]\s+(\S+)\s+\(by\s+(\S+),\s+v(\d+)\):\s*(.*?)(?=\n\[|\Z)",
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Event:
    """A single tool-call event extracted from a trace."""

    timestamp: float
    agent_name: str
    step_number: int
    tool_name: str
    arguments: dict
    result: dict | None
    result_raw: str
    duration: float
    success: bool | None
    model_output: str
    event_category: str


@dataclass
class ReconstructedEntry:
    """A single entry in the reconstructed blackboard state."""

    key: str
    value: str
    author: str
    entry_type: str
    version: int
    timestamp: float


@dataclass
class BlackboardSnapshot:
    """Blackboard state at the start of an agent's turn (from input context)."""

    agent_name: str
    timestamp: float
    entries: list  # list of dicts: {key, value, author, entry_type, version}


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


def _tool_icon(name: str) -> str:
    return _TOOL_ICONS.get(name, "🔹")


def _list_trace_files(base: str = "logs/batch_results/") -> list[str]:
    """Find all trace JSON files under batch results, newest first."""
    base_path = Path(_project_root) / base
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


def _try_parse_json(s: str) -> dict | None:
    """Try to parse a string as JSON, return None on failure."""
    if not s or not s.strip():
        return None
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None


def _format_time(timestamp: float, base_time: float) -> str:
    """Format timestamp as relative offset from base_time."""
    if timestamp is None or base_time is None:
        return ""
    delta = max(0, timestamp - base_time)
    mins = int(delta // 60)
    secs = int(delta % 60)
    return f"+{mins}:{secs:02d}"


def _categorize(tool_name: str) -> str:
    if tool_name == "write_blackboard":
        return "blackboard_write"
    if tool_name == "read_blackboard":
        return "blackboard_read"
    if tool_name == "spawn_peer":
        return "spawn"
    if tool_name == "mark_task_done":
        return "mark_done"
    if tool_name == "final_answer":
        return "final_answer"
    return "mcp_tool"


def _infer_success(result: dict | None, tool_name: str) -> bool | None:
    if result is None:
        return None
    if "success" in result:
        return bool(result["success"])
    if "return_code" in result:
        return result["return_code"] == 0
    return None


# ---------------------------------------------------------------------------
# Blackboard snapshot extraction (from model_input_messages)
# ---------------------------------------------------------------------------

def _parse_blackboard_section(text: str) -> list[dict]:
    """Parse the 'Current Blackboard State:' block from a context string."""
    marker = "Current Blackboard State:"
    bb_start = text.find(marker)
    if bb_start == -1:
        return []
    bb_text = text[bb_start + len(marker):]
    history_idx = bb_text.find("\nRecent History:")
    if history_idx != -1:
        bb_text = bb_text[:history_idx]
    entries = []
    for m in _BB_ENTRY_RE.finditer(bb_text):
        entries.append({
            "entry_type": m.group(1).lower(),
            "key": m.group(2),
            "author": m.group(3),
            "version": int(m.group(4)),
            "value": m.group(5).strip(),
        })
    return entries


def extract_blackboard_snapshots(trace_data: dict) -> list[BlackboardSnapshot]:
    """For each agent turn, extract the blackboard state they saw at start."""
    snapshots = []
    for agent_name, agent_data in trace_data.get("agents", {}).items():
        for step in agent_data.get("steps", []):
            if step.get("step_number") != 1:
                continue
            msgs = step.get("model_input_messages", [])
            user_msgs = [m for m in msgs if "USER" in str(m.get("role", ""))]
            if not user_msgs:
                continue
            content = str(user_msgs[0].get("content", ""))
            entries = _parse_blackboard_section(content)
            if not entries:
                continue
            snapshots.append(BlackboardSnapshot(
                agent_name=agent_name,
                timestamp=step.get("start_time", 0) - 0.5,
                entries=entries,
            ))
    snapshots.sort(key=lambda s: s.timestamp)
    return snapshots


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def extract_events(trace_data: dict) -> list[Event]:
    """Extract all tool-call events from trace data, sorted by time."""
    events: list[Event] = []
    agents = trace_data.get("agents", {})

    for agent_name, agent_data in agents.items():
        for step in agent_data.get("steps", []):
            timestamp = step.get("start_time") or 0
            step_num = step.get("step_number", 0)
            duration = step.get("duration_seconds", 0)
            observations = step.get("observations", "")
            model_output = step.get("model_output", "")
            tool_calls = step.get("tool_calls", [])
            if not tool_calls:
                continue
            result = _try_parse_json(observations)
            for tc in tool_calls:
                tool_name = tc.get("name", "unknown")
                arguments = tc.get("arguments", {})
                if isinstance(arguments, str):
                    arguments = {"answer": arguments}
                category = _categorize(tool_name)
                success = _infer_success(result, tool_name)
                events.append(Event(
                    timestamp=timestamp,
                    agent_name=agent_name,
                    step_number=step_num,
                    tool_name=tool_name,
                    arguments=arguments,
                    result=result,
                    result_raw=observations,
                    duration=duration,
                    success=success,
                    model_output=model_output,
                    event_category=category,
                ))

    events.sort(key=lambda e: (e.timestamp, e.step_number))
    return events


# ---------------------------------------------------------------------------
# Blackboard state reconstruction
# ---------------------------------------------------------------------------

def reconstruct_blackboard_state(
    events: list[Event],
    snapshots: list[BlackboardSnapshot],
) -> list[ReconstructedEntry]:
    """Build the final board state from snapshots + explicit tool calls."""
    entries: dict[str, ReconstructedEntry] = {}

    def _upsert(key, value, author, entry_type, timestamp):
        if key in entries:
            e = entries[key]
            e.value = value
            e.author = author
            e.entry_type = entry_type
            e.version += 1
            e.timestamp = timestamp
        else:
            entries[key] = ReconstructedEntry(
                key=key, value=value, author=author,
                entry_type=entry_type, version=1, timestamp=timestamp,
            )

    if snapshots:
        last_snap = snapshots[-1]
        for e in last_snap.entries:
            _upsert(e["key"], e["value"], e["author"], e["entry_type"], last_snap.timestamp)

    for ev in events:
        if ev.event_category == "blackboard_write":
            key = ev.arguments.get("key", "")
            value = ev.arguments.get("value", "")
            entry_type = ev.arguments.get("entry_type", "status")
        elif ev.event_category == "mark_done":
            key = "task_complete"
            value = f"DONE: {ev.arguments.get('summary', '')}"
            entry_type = "status"
        else:
            continue
        if not key:
            continue
        _upsert(key, value, ev.agent_name, entry_type, ev.timestamp)

    return sorted(entries.values(), key=lambda e: e.timestamp)


# ---------------------------------------------------------------------------
# Conversation tab — build ordered items
# ---------------------------------------------------------------------------

def _build_conversation(
    events: list[Event],
    snapshots: list[BlackboardSnapshot],
) -> list[dict]:
    """Build an ordered list of conversation items from snapshots + events.

    Returns items of types:
      "bb_auto_write"  — board was updated between turns (structural write)
      "agent_turn"     — an agent's full turn (read + tool calls)
      "task_complete"  — run ended via mark_task_done
    """
    if not snapshots:
        return []

    items = []
    prev_keys: set[str] = set()

    for i, snap in enumerate(snapshots):
        t_snap = snap.timestamp + 0.5  # actual start time
        t_next = (
            (snapshots[i + 1].timestamp + 0.5) if i + 1 < len(snapshots)
            else float("inf")
        )

        # Board entries that are NEW at this snapshot vs the previous one.
        curr_keys = {e["key"] for e in snap.entries}
        new_keys = curr_keys - prev_keys
        new_entries = [e for e in snap.entries if e["key"] in new_keys]

        # Show board auto-write that happened before this agent's turn
        # (written by Python's next_step() after the previous agent finished).
        if new_entries:
            prev_agent = snapshots[i - 1].agent_name if i > 0 else "system"
            items.append({
                "type": "bb_auto_write",
                "prev_agent": prev_agent,
                "timestamp": t_snap - 0.1,
                "entries": new_entries,
                "all_entries": snap.entries,
            })

        # Gather this agent's tool-call events within this turn's time window.
        turn_events = [
            e for e in events
            if t_snap <= e.timestamp < t_next
        ]

        prior_turns = sum(
            1 for j in range(i) if snapshots[j].agent_name == snap.agent_name
        )

        items.append({
            "type": "agent_turn",
            "agent": snap.agent_name,
            "timestamp": t_snap,
            "snapshot": snap,
            "new_keys": new_keys,
            "events": turn_events,
            "turn_index": i + 1,
            "is_repeat": prior_turns > 0,
        })

        prev_keys = curr_keys

    # mark_task_done appears as a special completion item at the end.
    for ev in events:
        if ev.event_category == "mark_done":
            items.append({
                "type": "task_complete",
                "agent": ev.agent_name,
                "timestamp": ev.timestamp + 0.05,
                "event": ev,
            })

    items.sort(key=lambda x: x["timestamp"])
    return items


# ---------------------------------------------------------------------------
# Conversation tab — rendering
# ---------------------------------------------------------------------------

def _render_conversation_tab(
    events: list[Event],
    snapshots: list[BlackboardSnapshot],
    trace_data: dict,
) -> None:
    """Chat-room conversation view using st.chat_message() bubbles."""
    if not snapshots and not events:
        st.info(
            "No conversation data available.\n\n"
            "This view works best with networked strategy traces that have "
            "blackboard state in agent input messages."
        )
        return

    base_time = (
        snapshots[0].timestamp + 0.5 if snapshots
        else (events[0].timestamp if events else 0)
    )

    if not snapshots:
        st.warning("No blackboard snapshots found — showing raw events only.")
        for ev in events:
            with st.chat_message(ev.agent_name, avatar=_agent_icon(ev.agent_name)):
                t = _format_time(ev.timestamp, base_time)
                st.markdown(f"**{ev.tool_name}** `{t}`")
        return

    # Summary bar.
    n_turns = len(snapshots)
    n_agents = len({s.agent_name for s in snapshots})
    final_entries = len(snapshots[-1].entries)
    mark_dones = [e for e in events if e.event_category == "mark_done"]
    dur = trace_data.get("duration_seconds", 0)

    cols = st.columns(5)
    cols[0].metric("Turns", n_turns)
    cols[1].metric("Agents", n_agents)
    cols[2].metric("Board entries", final_entries)
    cols[3].metric("Duration", f"{dur:.0f}s")
    cols[4].metric("Outcome", "✅ Done" if mark_dones else "⏱ Max turns")

    st.divider()

    items = _build_conversation(events, snapshots)

    for item in items:
        t = _format_time(item["timestamp"], base_time)

        # ── Board update (system post between turns) ───────────────────────
        if item["type"] == "bb_auto_write":
            entries = item["entries"]
            prev_agent = item["prev_agent"]

            with st.chat_message("assistant", avatar="📋"):
                st.markdown(
                    f"**Board Update** &nbsp; "
                    f"<span style='color:#888;font-size:0.85em'>"
                    f"auto-written after **{prev_agent}**'s turn · {t}</span>",
                    unsafe_allow_html=True,
                )
                for e in entries:
                    et_color = _ENTRY_TYPE_COLORS.get(e["entry_type"], "#888")
                    ac = _agent_color(e["author"])
                    value_preview = e["value"][:200]
                    if len(e["value"]) > 200:
                        value_preview += "…"
                    st.markdown(
                        f"<span style='background:{et_color};color:white;"
                        f"padding:1px 7px;border-radius:4px;font-size:0.78em'>"
                        f"{e['entry_type'].upper()}</span>"
                        f" &nbsp; `{e['key']}`"
                        f" &nbsp; by <span style='color:{ac}'>{e['author']}</span>",
                        unsafe_allow_html=True,
                    )
                    st.caption(value_preview)

        # ── Agent turn ─────────────────────────────────────────────────────
        elif item["type"] == "agent_turn":
            snap: BlackboardSnapshot = item["snapshot"]
            agent = snap.agent_name
            icon = _agent_icon(agent)
            new_keys: set = item["new_keys"]
            turn_events: list[Event] = item["events"]
            turn_num = item["turn_index"]
            is_repeat = item["is_repeat"]

            # Detect if agent skipped redundant work.
            mcp_calls = [e for e in turn_events if e.event_category == "mcp_tool"]
            fa_only = (
                len(turn_events) > 0
                and not mcp_calls
                and any(e.event_category == "final_answer" for e in turn_events)
            )

            with st.chat_message(agent, avatar=icon):
                # ── Turn header ──
                repeat_txt = " *(2nd visit)*" if is_repeat else ""
                smart_txt = " — **skipped duplicate work ✓**" if fa_only else ""
                st.markdown(
                    f"**{agent}** &nbsp; Turn {turn_num}{repeat_txt}"
                    f" &nbsp; `{t}`{smart_txt}"
                )

                # ── What the agent saw on the board (read receipt) ──
                pills = []
                for e in snap.entries:
                    et_color = _ENTRY_TYPE_COLORS.get(e["entry_type"], "#888")
                    star = " ✨" if e["key"] in new_keys else ""
                    short_key = e["key"].replace("_result", "")
                    pills.append(
                        f"<span style='background:{et_color}28;color:{et_color};"
                        f"border:1px solid {et_color}60;border-radius:4px;"
                        f"padding:1px 6px;font-size:0.78em'>"
                        f"{e['entry_type'].upper()} {short_key}{star}</span>"
                    )
                pills_html = " &nbsp; ".join(pills) or "<em style='color:#666'>empty</em>"
                st.markdown(
                    f"<span style='color:#888;font-size:0.85em'>📖 read board "
                    f"({len(snap.entries)} entr{'y' if len(snap.entries)==1 else 'ies'})"
                    f":</span><br>{pills_html}",
                    unsafe_allow_html=True,
                )

                # ── Tool calls (skip mark_done — shown separately) ──
                visible = [e for e in turn_events if e.event_category != "mark_done"]
                if visible:
                    st.markdown("---")
                for ev in visible:
                    ok = ev.success
                    ev_t = _format_time(ev.timestamp, base_time)
                    t_icon = _tool_icon(ev.tool_name)

                    if ev.event_category == "final_answer":
                        answer = ev.arguments.get("answer", "")
                        preview = (answer[:150] + "…") if len(answer) > 150 else answer
                        st.markdown(f"🏁 **final_answer** `{ev_t}`")
                        st.caption(preview)

                    elif ev.tool_name == "get_results" and ev.result:
                        fb = ev.result.get("fuel_burn", "?")
                        gtow = ev.result.get("gtow", "?")
                        wm = ev.result.get("wing_mass", "?")
                        badge = "✅" if ok else "❌"
                        col1, col2, col3, col4 = st.columns(4)
                        col1.markdown(f"📊 **get_results** {badge} `{ev_t}`")
                        col2.metric("Fuel Burn", fb)
                        col3.metric("Wing Mass", wm)
                        col4.metric("GTOW", gtow)

                    elif ev.tool_name == "run_simulation":
                        badge = "✅" if ok else "❌"
                        files = ""
                        if ev.result and ev.result.get("files_created"):
                            files = f" → `{'`, `'.join(ev.result['files_created'])}`"
                        st.markdown(f"🔧 **run_simulation** {badge} `{ev_t}`{files}")
                        if not ok and ev.result and ev.result.get("stderr"):
                            st.error(ev.result["stderr"][:300])
                        if ev.arguments.get("code"):
                            with st.expander("Code"):
                                st.code(ev.arguments["code"], language="python")

                    else:
                        badge = "✅" if ok is True else ("❌" if ok is False else "")
                        st.markdown(
                            f"{t_icon} **{ev.tool_name}** {badge} `{ev_t}`"
                        )

        # ── Task complete banner ───────────────────────────────────────────
        elif item["type"] == "task_complete":
            ev: Event = item["event"]
            summary = ev.arguments.get("summary", "")

            with st.chat_message("assistant", avatar="✅"):
                st.success(
                    f"**Task complete** — signalled by **{ev.agent_name}** at `{t}`\n\n"
                    f"{summary}\n\n"
                    f"*wrote `task_complete` to blackboard → coordinator stopped*"
                )


# ---------------------------------------------------------------------------
# Blackboard State tab
# ---------------------------------------------------------------------------

def _render_blackboard_state_tab(
    events: list[Event],
    snapshots: list[BlackboardSnapshot],
) -> None:
    has_tool_writes = any(
        e.event_category in ("blackboard_write", "mark_done") for e in events
    )
    has_snapshots = bool(snapshots)

    if not has_tool_writes and not has_snapshots:
        st.info(
            "No blackboard data found in this trace.\n\n"
            "Blackboard entries appear when agents call `write_blackboard`, "
            "`mark_task_done`, or when structural writes populate the board."
        )
        return

    entries = reconstruct_blackboard_state(events, snapshots)
    base_time = (
        snapshots[0].timestamp + 0.5 if snapshots
        else (events[0].timestamp if events else 0)
    )

    sources = []
    if has_snapshots:
        sources.append(f"{len(snapshots)} context snapshots (structural writes)")
    if has_tool_writes:
        n_tw = sum(1 for e in events if e.event_category in ("blackboard_write", "mark_done"))
        sources.append(f"{n_tw} explicit tool write(s)")
    st.caption(f"Reconstructed from: {', '.join(sources)}")

    st.subheader(f"Final Board State ({len(entries)} entries)")

    for entry in entries:
        et_color = _ENTRY_TYPE_COLORS.get(entry.entry_type, "#888")
        color = _agent_color(entry.author)
        time_str = _format_time(entry.timestamp, base_time)

        with st.container(border=True):
            st.markdown(
                f"<span style='background:{et_color};color:white;padding:1px 8px;"
                f"border-radius:10px;font-size:0.8em'>{entry.entry_type.upper()}</span>"
                f" &nbsp; `{entry.key}` &nbsp; v{entry.version}"
                f" &nbsp; by <span style='color:{color}'>{entry.author}</span>"
                f" &nbsp; <span style='color:#888;font-size:0.8em'>{time_str}</span>",
                unsafe_allow_html=True,
            )
            if len(entry.value) > 300:
                with st.expander("Value"):
                    st.text(entry.value)
            else:
                st.text(entry.value or "(empty)")

    if len(snapshots) > 1:
        st.divider()
        st.subheader("Board Evolution by Turn")
        prev_keys: set[str] = set()
        for snap in snapshots:
            color = _agent_color(snap.agent_name)
            icon = _agent_icon(snap.agent_name)
            time_str = _format_time(snap.timestamp + 0.5, base_time)
            current_keys = {e["key"] for e in snap.entries}
            new_keys = current_keys - prev_keys

            label = (
                f"{icon} **{snap.agent_name}** starts — "
                f"{len(snap.entries)} entries"
                + (f", **{len(new_keys)} new** 🆕" if new_keys else "")
                + f"   {time_str}"
            )
            with st.expander(label, expanded=False):
                for entry in snap.entries:
                    et_color = _ENTRY_TYPE_COLORS.get(entry["entry_type"], "#888")
                    is_new = entry["key"] in new_keys
                    new_badge = " 🆕" if is_new else ""
                    author_color = _agent_color(entry["author"])
                    st.markdown(
                        f"<span style='background:{et_color};color:white;padding:0 6px;"
                        f"border-radius:8px;font-size:0.75em'>{entry['entry_type'].upper()}</span>"
                        f" &nbsp; `{entry['key']}` v{entry['version']}"
                        f" &nbsp; by <span style='color:{author_color}'>{entry['author']}</span>"
                        f"{new_badge}",
                        unsafe_allow_html=True,
                    )
                    value = entry["value"]
                    if len(value) > 200:
                        with st.expander("Value"):
                            st.text(value)
                    else:
                        st.caption(value or "(empty)")
            prev_keys = current_keys


# ---------------------------------------------------------------------------
# All Tool Calls tab (old Chat Room)
# ---------------------------------------------------------------------------

def _render_all_tool_calls_tab(
    events: list[Event],
    snapshots: list[BlackboardSnapshot],
    selected_agents: list[str],
    show_reasoning: bool,
    show_snapshots: bool,
) -> None:
    """Detailed chronological tool-call log with optional snapshot cards."""
    if not events:
        st.info("No tool-call events match the current filters.")
        return

    base_time = events[0].timestamp if events else 0

    agents_active = len({e.agent_name for e in events})
    tools_used = len({e.tool_name for e in events})
    successes = sum(1 for e in events if e.success is True)
    failures = sum(1 for e in events if e.success is False)
    sum(1 for e in events if e.event_category == "blackboard_write")
    sum(1 for e in events if e.event_category == "mark_done")

    cols = st.columns(5)
    cols[0].metric("Events", len(events))
    cols[1].metric("Agents", agents_active)
    cols[2].metric("Tools", tools_used)
    cols[3].metric("✅ Passed", successes)
    cols[4].metric("❌ Failed", failures)

    st.divider()

    # Merged timeline with optional snapshots.
    timeline = [(ev.timestamp, "event", ev) for ev in events]
    if show_snapshots:
        for snap in snapshots:
            if snap.agent_name in selected_agents:
                timeline.append((snap.timestamp, "snapshot", snap))
    timeline.sort(key=lambda x: x[0])

    for _, item_type, item in timeline:
        if item_type == "snapshot":
            _render_snapshot_card(item, base_time)
        else:
            _render_event_card(item, base_time, show_reasoning)


def _render_snapshot_card(snap: BlackboardSnapshot, base_time: float) -> None:
    _agent_color(snap.agent_name)
    _agent_icon(snap.agent_name)
    time_str = _format_time(snap.timestamp + 0.5, base_time)
    n_results = sum(1 for e in snap.entries if e["entry_type"] == "result")
    detail = f"{len(snap.entries)} entries"
    if n_results:
        detail += f", {n_results} result{'s' if n_results > 1 else ''}"
    with st.expander(
        f"📋 Board when {snap.agent_name} started — {detail}   {time_str}",
        expanded=False,
    ):
        for entry in snap.entries:
            et_color = _ENTRY_TYPE_COLORS.get(entry["entry_type"], "#888")
            author_color = _agent_color(entry["author"])
            st.markdown(
                f"<span style='background:{et_color};color:white;padding:0 6px;"
                f"border-radius:8px;font-size:0.75em'>{entry['entry_type'].upper()}</span>"
                f" &nbsp; `{entry['key']}` v{entry['version']}"
                f" &nbsp; by <span style='color:{author_color}'>{entry['author']}</span>",
                unsafe_allow_html=True,
            )
            value = entry["value"]
            if len(value) > 250:
                with st.expander("Value"):
                    st.text(value)
            else:
                st.caption(value or "(empty)")


def _render_event_card(ev: Event, base_time: float, show_reasoning: bool) -> None:
    """Render one event as a detailed card."""
    color = _agent_color(ev.agent_name)
    icon = _agent_icon(ev.agent_name)
    t_icon = _tool_icon(ev.tool_name)
    time_str = _format_time(ev.timestamp, base_time)
    et = ev.arguments.get("entry_type", "") if ev.event_category == "blackboard_write" else ""
    et_color = _ENTRY_TYPE_COLORS.get(et, "#888") if et else ""

    if ev.event_category == "mark_done":
        summary = ev.arguments.get("summary", "")
        st.markdown(
            f"<div style='border:2px solid #2ecc71;border-radius:8px;padding:12px;"
            f"background:rgba(46,204,113,0.08);margin:8px 0'>"
            f"✅ <span style='color:{color};font-weight:bold'>{ev.agent_name}</span>"
            f" called <code>mark_task_done</code>"
            f" &nbsp; <span style='color:#888;font-size:0.8em'>{time_str}</span><br/>"
            f"<span style='color:#2ecc71;font-size:0.9em'>{summary}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        return

    if ev.event_category == "blackboard_write":
        key = ev.arguments.get("key", "")
        value = ev.arguments.get("value", "")
        with st.container(border=True):
            st.markdown(
                f"{icon} <span style='color:{color};font-weight:bold'>{ev.agent_name}</span>"
                f" &nbsp; ✏️"
                f" &nbsp; <span style='background:{et_color};color:white;padding:1px 8px;"
                f"border-radius:10px;font-size:0.8em'>{et.upper()}</span>"
                f" &nbsp; <span style='color:#888;font-size:0.8em'>{time_str}</span>",
                unsafe_allow_html=True,
            )
            st.markdown(f"**Key:** `{key}`")
            st.text(value[:300] + ("…" if len(value) > 300 else "") or "(empty)")
        return

    with st.container(border=True):
        if ev.success is True:
            status_badge = "<span style='color:#2ecc71'>✅</span>"
        elif ev.success is False:
            status_badge = "<span style='color:#e74c3c'>❌</span>"
        else:
            status_badge = ""

        st.markdown(
            f"{icon} <span style='color:{color};font-weight:bold'>{ev.agent_name}</span>"
            f" &nbsp; {t_icon} <code>{ev.tool_name}</code>"
            f" &nbsp; {status_badge}"
            f" &nbsp; <span style='color:#888;font-size:0.8em'>{time_str}</span>",
            unsafe_allow_html=True,
        )

        code_arg = ev.arguments.get("code")
        if code_arg:
            with st.expander("Code", expanded=False):
                st.code(code_arg, language="python")
        elif ev.arguments and ev.event_category != "final_answer":
            with st.expander("Arguments"):
                st.json(ev.arguments)

        if ev.event_category == "final_answer":
            answer = ev.arguments.get("answer", "")
            st.text(str(answer)[:300] + ("…" if len(str(answer)) > 300 else ""))
        elif ev.result:
            if ev.success is False and ev.result.get("stderr"):
                st.error(ev.result["stderr"][:400])
            elif ev.result.get("files_created"):
                st.caption(f"Files: {', '.join(ev.result['files_created'])}")
            if ev.tool_name == "get_results":
                metrics = {k: ev.result[k] for k in
                           ("fuel_burn", "gtow", "wing_mass", "converged")
                           if k in ev.result}
                if metrics:
                    cols = st.columns(len(metrics))
                    for col, (k, v) in zip(cols, metrics.items()):
                        col.metric(k.replace("_", " ").title(), f"{v}")
            with st.expander("Raw result"):
                st.json(ev.result)

        if show_reasoning and ev.model_output:
            with st.expander("Agent reasoning"):
                st.markdown(ev.model_output)


# ---------------------------------------------------------------------------
# Agent Activity tab
# ---------------------------------------------------------------------------

def _render_activity_tab(events: list[Event], trace_data: dict) -> None:
    if not events:
        st.info("No events to display.")
        return

    agents = trace_data.get("agents", {})
    agent_names = sorted(agents.keys())

    st.subheader("Events per Agent")
    agent_counts: dict[str, int] = {}
    for ev in events:
        agent_counts[ev.agent_name] = agent_counts.get(ev.agent_name, 0) + 1

    for name in agent_names:
        count = agent_counts.get(name, 0)
        if count == 0:
            continue
        color = _agent_color(name)
        icon = _agent_icon(name)
        pct = count / len(events) * 100
        st.markdown(
            f"{icon} **{name}** — {count} events ({pct:.0f}%)"
            f"<div style='background:#333;border-radius:4px;height:12px;margin:2px 0 8px 0'>"
            f"<div style='background:{color};width:{pct:.1f}%;height:100%;"
            f"border-radius:4px'></div></div>",
            unsafe_allow_html=True,
        )

    st.divider()

    st.subheader("Tool Usage per Agent")
    agent_tools: dict[str, dict[str, int]] = {}
    for ev in events:
        if ev.agent_name not in agent_tools:
            agent_tools[ev.agent_name] = {}
        agent_tools[ev.agent_name][ev.tool_name] = (
            agent_tools[ev.agent_name].get(ev.tool_name, 0) + 1
        )

    for name in agent_names:
        tools = agent_tools.get(name, {})
        if not tools:
            continue
        color = _agent_color(name)
        icon = _agent_icon(name)
        tool_str = ", ".join(
            f"{_tool_icon(t)} {t}: {c}" for t, c in sorted(tools.items())
        )
        st.markdown(
            f"{icon} <span style='color:{color}'>{name}</span>: {tool_str}",
            unsafe_allow_html=True,
        )

    st.divider()

    st.subheader("Success / Failure per Agent")
    for name in agent_names:
        agent_events = [e for e in events if e.agent_name == name]
        if not agent_events:
            continue
        successes = sum(1 for e in agent_events if e.success is True)
        failures = sum(1 for e in agent_events if e.success is False)
        unknown = sum(1 for e in agent_events if e.success is None)
        total = len(agent_events)
        color = _agent_color(name)
        icon = _agent_icon(name)
        parts = []
        if successes:
            parts.append(f"<span style='color:#2ecc71'>{successes} passed</span>")
        if failures:
            parts.append(f"<span style='color:#e74c3c'>{failures} failed</span>")
        if unknown:
            parts.append(f"<span style='color:#888'>{unknown} N/A</span>")
        st.markdown(
            f"{icon} <span style='color:{color}'>{name}</span> ({total} events): "
            + " | ".join(parts),
            unsafe_allow_html=True,
        )

    st.divider()

    st.subheader("Agent Timeline")
    base_time = events[0].timestamp if events else 0
    total_duration = (events[-1].timestamp - base_time) if len(events) > 1 else 1
    if total_duration <= 0:
        total_duration = 1

    for name in agent_names:
        agent_events = [e for e in events if e.agent_name == name]
        if not agent_events:
            continue
        color = _agent_color(name)
        icon = _agent_icon(name)
        markers = []
        for ev in agent_events:
            offset_pct = (ev.timestamp - base_time) / total_duration * 100
            marker_color = "#e74c3c" if ev.success is False else color
            markers.append(
                f"<div style='position:absolute;left:{offset_pct:.1f}%;"
                f"width:3px;height:100%;background:{marker_color}'></div>"
            )
        first_t = _format_time(agent_events[0].timestamp, base_time)
        last_t = _format_time(agent_events[-1].timestamp, base_time)
        st.markdown(
            f"{icon} **{name}** ({first_t} — {last_t})"
            f"<div style='position:relative;background:#333;border-radius:4px;"
            f"height:16px;margin:2px 0 8px 0;overflow:hidden'>"
            + "".join(markers)
            + "</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Blackboard Viewer",
        page_icon="📋",
        layout="wide",
    )
    st.title("📋 Blackboard Viewer")
    st.caption("Chat-room style visualization of networked agent coordination")

    # --- Sidebar ---
    with st.sidebar:
        st.header("Trace File")
        traces = _list_trace_files()
        if not traces:
            st.warning("No trace files found in logs/batch_results/")
            st.stop()

        labels = []
        for t in traces:
            parts = Path(t).parts
            batch_id = parts[-2] if len(parts) >= 2 else "?"
            name = Path(t).stem.replace("_trace", "")
            labels.append(f"{batch_id}/{name}")

        selected_idx = st.selectbox(
            "Select trace",
            range(len(traces)),
            format_func=lambda i: labels[i],
        )
        trace_path = traces[selected_idx]
        trace_data = _load_trace(trace_path)

        st.divider()
        st.subheader("Run Info")
        st.markdown(f"**Combination:** {trace_data.get('combination', '?')}")
        st.markdown(f"**Strategy:** {trace_data.get('org_structure', '?')}")
        st.markdown(f"**Handler:** {trace_data.get('handler', '?')}")
        st.markdown(f"**Status:** {trace_data.get('status', '?')}")
        dur = trace_data.get("duration_seconds", 0)
        st.markdown(f"**Duration:** {dur:.0f}s ({dur / 60:.1f}m)")
        st.markdown(f"**Turns:** {trace_data.get('total_turns', '?')}")

        st.divider()
        st.subheader("Agents")
        all_events = extract_events(trace_data)
        all_snapshots = extract_blackboard_snapshots(trace_data)
        agent_names = sorted(trace_data.get("agents", {}).keys())
        event_counts: dict[str, int] = {}
        for ev in all_events:
            event_counts[ev.agent_name] = event_counts.get(ev.agent_name, 0) + 1
        snap_counts: dict[str, int] = {}
        for snap in all_snapshots:
            snap_counts[snap.agent_name] = snap_counts.get(snap.agent_name, 0) + 1

        for name in agent_names:
            color = _agent_color(name)
            icon = _agent_icon(name)
            ev_c = event_counts.get(name, 0)
            sn_c = snap_counts.get(name, 0)
            detail = f"{ev_c} event{'s' if ev_c != 1 else ''}"
            if sn_c:
                detail += f", {sn_c} turn{'s' if sn_c != 1 else ''}"
            st.markdown(
                f"{icon} <span style='color:{color}'>{name}</span> ({detail})",
                unsafe_allow_html=True,
            )

        st.divider()
        st.subheader("Filters")
        categories = sorted({e.event_category for e in all_events})
        category_labels = {
            "blackboard_write": "✏️ Blackboard Writes",
            "blackboard_read": "👁️ Blackboard Reads",
            "spawn": "➕ Spawn Peer",
            "mark_done": "✅ Mark Task Done",
            "mcp_tool": "🔧 MCP Tools",
            "final_answer": "🏁 Final Answer",
        }
        selected_categories = st.multiselect(
            "Event categories (Tool Calls tab)",
            categories,
            default=categories,
            format_func=lambda c: category_labels.get(c, c),
        )
        selected_agents = st.multiselect(
            "Agents (Tool Calls tab)",
            agent_names,
            default=agent_names,
        )

        st.divider()
        st.subheader("Display")
        show_reasoning = st.checkbox("Show agent reasoning", value=False)
        show_snapshots = st.checkbox(
            "Show board snapshots in Tool Calls",
            value=True,
            help="Show board state cards in the Tool Calls tab.",
        )

    # Filter events for tool-calls tab.
    filtered = [
        e for e in all_events
        if e.event_category in selected_categories
        and e.agent_name in selected_agents
    ]

    # --- Tabs ---
    tab_convo, tab_state, tab_activity, tab_calls = st.tabs([
        "💬 Conversation",
        "📋 Blackboard State",
        "📊 Agent Activity",
        "🔧 All Tool Calls",
    ])

    with tab_convo:
        _render_conversation_tab(all_events, all_snapshots, trace_data)
    with tab_state:
        _render_blackboard_state_tab(all_events, all_snapshots)
    with tab_activity:
        _render_activity_tab(all_events, trace_data)
    with tab_calls:
        _render_all_tool_calls_tab(
            filtered, all_snapshots, selected_agents,
            show_reasoning, show_snapshots,
        )


if __name__ == "__main__":
    main()
