"""Peer tools for the networked coordination strategy.

Three Tool subclasses available to ALL peer agents (in addition to
whatever domain tools they have):
  - ReadBlackboard: read shared blackboard state
  - WriteBlackboard: post status, results, claims, gaps, predictions
  - SpawnPeer: create a new peer agent when a gap is identified

These tools operate on a shared NetworkedContext that holds mutable
state (blackboard, agent pool, model, config) across tool calls.
"""

import json
from dataclasses import dataclass, field

from smolagents import Tool, ToolCallingAgent
from smolagents.models import Model

from src.coordination.blackboard import Blackboard

PEER_TOOL_NAMES = frozenset({"read_blackboard", "write_blackboard", "spawn_peer", "mark_task_done"})


@dataclass
class NetworkedContext:
    """Shared mutable state for networked peer tools.

    Created by NetworkedStrategy during initialization and passed
    to each peer tool so they all operate on the same state.
    """

    blackboard: Blackboard
    agents: dict  # name -> ToolCallingAgent (all agents)
    model: Model
    all_tools: list  # full tool set (domain + peer tools) for new agents
    peer_prompt: str  # assembled system prompt for new agents
    agent_max_steps: int = 8
    max_agents: int = 10
    agent_counter: int = 0  # for auto-naming: agent_1, agent_2, ...
    spawned_agents: list = field(default_factory=list)  # names in spawn order
    config: dict = field(default_factory=dict)  # toggle config for filtering


class ReadBlackboard(Tool):
    """Let an agent see the current state of the shared blackboard."""

    name = "read_blackboard"
    description = (
        "Reads the shared blackboard to see what peers are working on, "
        "what's been completed, active claims, and identified gaps. "
        "Optionally filter by entry type."
    )
    inputs = {
        "entry_type": {
            "type": "string",
            "description": (
                'Filter entries by type: "status", "claim", "result", '
                '"gap", "prediction", or "all" (default "all").'
            ),
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, context: NetworkedContext, **kwargs):
        super().__init__(**kwargs)
        self._context = context

    def forward(self, entry_type: str | None = None) -> str:
        ctx = self._context
        bb = ctx.blackboard

        if entry_type and entry_type != "all":
            entries = bb.read_by_type(entry_type)
        else:
            entries = bb.read_all()

        # Apply toggle filtering via the blackboard's context renderer.
        # We re-filter manually to return structured JSON.
        filtered = []
        predictive = ctx.config.get("predictive_knowledge", False)
        trans_specialist = ctx.config.get("trans_specialist_knowledge", True)
        peer_monitoring = ctx.config.get("peer_monitoring_visible", True)

        from src.coordination.blackboard import _strip_metrics, _truncate_result

        for e in entries:
            if e.entry_type == "prediction" and not predictive:
                continue

            display_value = e.value
            if e.entry_type == "result" and not trans_specialist:
                display_value = _truncate_result(display_value)
            if not peer_monitoring:
                display_value = _strip_metrics(display_value)

            filtered.append({
                "key": e.key,
                "value": display_value,
                "author": e.author,
                "entry_type": e.entry_type,
                "timestamp": e.timestamp,
                "version": e.version,
            })

        # Summary info.
        claims = bb.get_claims()
        active_claims = [c.key for c in claims]
        gaps = bb.read_by_type("gap")
        identified_gaps = [g.key for g in gaps]

        return json.dumps({
            "entries": filtered,
            "total_entries": len(filtered),
            "active_claims": active_claims,
            "identified_gaps": identified_gaps,
        })


class WriteBlackboard(Tool):
    """Post status, results, claims, or identified gaps to the blackboard."""

    name = "write_blackboard"
    description = (
        "Write an entry to the shared blackboard. Use entry_type 'status' "
        "to report what you're doing, 'claim' to claim a subtask, 'result' "
        "to post completed work, 'gap' to flag unaddressed work, or "
        "'prediction' to predict what another agent will do."
    )
    inputs = {
        "key": {
            "type": "string",
            "description": (
                "Unique identifier for this entry "
                '(e.g., "geometry_planning", "agent_2_status").'
            ),
        },
        "value": {
            "type": "string",
            "description": "The content to post.",
        },
        "entry_type": {
            "type": "string",
            "description": (
                'One of: "status", "claim", "result", "gap", "prediction".'
            ),
        },
    }
    output_type = "string"

    def __init__(self, context: NetworkedContext, agent_name: str = "", **kwargs):
        super().__init__(**kwargs)
        self._context = context
        self._agent_name = agent_name

    def forward(self, key: str, value: str, entry_type: str) -> str:
        bb = self._context.blackboard
        entry, warning = bb.write(key, value, self._agent_name, entry_type)

        if entry is None:
            # Hard claim rejection.
            return json.dumps({
                "success": False,
                "key": key,
                "entry_type": entry_type,
                "error": warning,
            })

        return json.dumps({
            "success": True,
            "key": entry.key,
            "entry_type": entry.entry_type,
            "version": entry.version,
            "warning": warning,
        })


class SpawnPeer(Tool):
    """Create a new peer agent when a gap is identified."""

    name = "spawn_peer"
    description = (
        "Spawn a new peer agent to help with unaddressed work. The new "
        "agent gets the same tools and base prompt as all other peers. "
        "Provide a reason explaining why a new agent is needed."
    )
    inputs = {
        "reason": {
            "type": "string",
            "description": (
                "Why this new agent is needed "
                '(e.g., "No agent is handling fillet operations").'
            ),
        },
    }
    output_type = "string"

    def __init__(self, context: NetworkedContext, agent_name: str = "", **kwargs):
        super().__init__(**kwargs)
        self._context = context
        self._agent_name = agent_name

    def forward(self, reason: str) -> str:
        ctx = self._context

        # Check agent limit.
        total_agents = len(ctx.agents)
        if total_agents >= ctx.max_agents:
            return json.dumps({
                "success": False,
                "new_agent_name": None,
                "reason": reason,
                "total_agents": total_agents,
                "error": f"Maximum agent limit ({ctx.max_agents}) reached",
            })

        # Generate name.
        ctx.agent_counter += 1
        new_name = f"agent_{ctx.agent_counter}"
        # Ensure uniqueness (shouldn't collide, but be safe).
        while new_name in ctx.agents:
            ctx.agent_counter += 1
            new_name = f"agent_{ctx.agent_counter}"

        # Create agent with full tool set.
        agent = ToolCallingAgent(
            tools=list(ctx.all_tools),
            model=ctx.model,
            name=new_name,
            description=f"Peer agent spawned because: {reason}",
            instructions=ctx.peer_prompt,
            max_steps=ctx.agent_max_steps,
            add_base_tools=False,
        )

        # Register.
        ctx.agents[new_name] = agent
        ctx.spawned_agents.append(new_name)

        # Post gap entry to blackboard documenting why spawned.
        ctx.blackboard.write(
            key=f"spawn_{new_name}",
            value=f"New peer {new_name} spawned by {self._agent_name}: {reason}",
            author=self._agent_name or "system",
            entry_type="gap",
        )

        return json.dumps({
            "success": True,
            "new_agent_name": new_name,
            "reason": reason,
            "total_agents": len(ctx.agents),
            "error": None,
        })


class MarkTaskDone(Tool):
    """Signal that the overall task is fully complete.

    Writes a DONE status to the shared blackboard under the key
    "task_complete".  The NetworkedStrategy's is_complete() checks for
    this entry and breaks the coordinator loop on the next iteration.

    Call this once when you are certain all subtasks are finished and
    the results are on the blackboard.  The first agent to call it
    stops the run — no other agent needs to call it.
    """

    name = "mark_task_done"
    description = (
        "Signal that the overall task is fully complete. Call this when "
        "all subtasks are done and results are posted to the blackboard. "
        "Provide a brief summary of what was accomplished. The first agent "
        "to call this stops the entire run."
    )
    inputs = {
        "summary": {
            "type": "string",
            "description": (
                "Brief summary of what was accomplished "
                '(e.g., "STL generated and evaluated: PCD=0.035, eval=success").'
            ),
        },
    }
    output_type = "string"

    def __init__(self, context: NetworkedContext, agent_name: str = "", **kwargs):
        super().__init__(**kwargs)
        self._context = context
        self._agent_name = agent_name

    def forward(self, summary: str) -> str:
        bb = self._context.blackboard
        value = f"DONE: {summary}"
        entry, _ = bb.write("task_complete", value, self._agent_name, "status")
        return json.dumps({
            "success": entry is not None,
            "key": "task_complete",
            "summary": summary,
        })
