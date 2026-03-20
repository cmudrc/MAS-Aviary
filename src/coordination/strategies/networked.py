"""Networked coordination strategy — peer-based, blackboard-driven.

All agents are equals. No hierarchy, no orchestrator. Agents self-
organize by reading shared blackboard state to decide what work to do.
Communication happens through a shared Blackboard (mutable key-value
store) alongside the immutable SharedHistory.

Execution flow:
  1. initialize() creates N initial peer agents with full tool set +
     peer tools, initializes the blackboard with the task description.
  2. next_step() selects an agent via placeholder rotation, builds
     context from blackboard + history, returns a CoordinationAction.
  3. is_complete() checks for TASK_COMPLETE, max turns, or all agents
     reporting DONE on blackboard.

Toggles (all independent, any combination valid):
  - claiming_mode: "none" / "soft" / "hard"
  - peer_monitoring_visible: bool
  - trans_specialist_knowledge: bool
  - predictive_knowledge: bool
"""

from src.coordination.blackboard import Blackboard
from src.coordination.history import AgentMessage
from src.coordination.strategy import CoordinationAction, CoordinationStrategy
from src.tools.networked_tools import (
    PEER_TOOL_NAMES,
    MarkTaskDone,
    NetworkedContext,
    ReadBlackboard,
    SpawnPeer,
    WriteBlackboard,
)


class NetworkedStrategy(CoordinationStrategy):
    """Peer-based strategy where agents coordinate via shared blackboard."""

    def __init__(self):
        # Config (set during initialize).
        self._initial_agents: int = 5
        self._max_agents: int = 10
        self._agent_max_steps: int = 8
        self._claiming_mode: str = "soft"
        self._peer_monitoring_visible: bool = True
        self._trans_specialist_knowledge: bool = True
        self._predictive_knowledge: bool = False
        self._termination_keyword: str = "TASK_COMPLETE"
        self._max_turns: int = 40
        self._max_consecutive_errors: int = 3
        self._max_context_tokens: int = 4000
        self._max_recent_messages: int = 15

        # Runtime state.
        self._agents: dict = {}
        self._context: NetworkedContext | None = None
        self._blackboard: Blackboard | None = None
        self._agent_order: list[str] = []  # rotation order
        self._rotation_index: int = 0
        self._total_turns: int = 0
        self._consecutive_errors: int = 0
        self._task: str = ""

        # Peer prompt parts (loaded from config).
        self._base_prompt: str = ""
        self._soft_claiming_addition: str = ""
        self._hard_claiming_addition: str = ""
        self._prediction_addition: str = ""

        # Phase gating (optional, loaded from config).
        self._workflow_phases: list[dict] = []
        self._full_toolsets: dict[str, dict] = {}  # original tools per agent
        self._handler_is_staged_pipeline: bool = False

        # Graph-driven mode (optional, set when _graph_def is in config).
        self._graph = None           # GraphDefinition or None
        self._graph_current_state: str | None = None
        self._graph_state_dict: dict = {}
        self._graph_complete: bool = False

    def initialize(self, agents: dict, config: dict) -> None:
        """Set up strategy from agents dict and coordination config.

        The agents dict is used as the shared reference (same as
        orchestrated strategy) so dynamically spawned agents are
        visible to the Coordinator's run loop.
        """
        net_config = config.get("networked", {})

        # Read config.
        self._initial_agents = net_config.get("initial_agents", 5)
        self._max_agents = net_config.get("max_agents", 10)
        self._agent_max_steps = net_config.get("agent_max_steps", 8)
        self._claiming_mode = net_config.get("claiming_mode", "soft")
        self._peer_monitoring_visible = net_config.get("peer_monitoring_visible", True)
        self._trans_specialist_knowledge = net_config.get("trans_specialist_knowledge", True)
        self._predictive_knowledge = net_config.get("predictive_knowledge", False)

        term_config = config.get("termination", {})
        self._termination_keyword = term_config.get("keyword", "TASK_COMPLETE")
        self._max_turns = term_config.get("max_turns", 40)
        self._max_consecutive_errors = term_config.get("max_consecutive_errors", 3)

        ctx_config = config.get("context", {})
        self._max_context_tokens = ctx_config.get("max_context_tokens", 4000)
        self._max_recent_messages = ctx_config.get("max_recent_messages", 15)

        # Workflow phase gating (optional).
        # Skip phase gating when staged_pipeline handler is active — the
        # handler's stage definitions already enforce ordering through
        # completion criteria.  The two mechanisms are redundant and conflict.
        self._handler_is_staged_pipeline = (
            config.get("execution_handler") == "staged_pipeline"
        )
        self._workflow_phases = net_config.get("workflow_phases", [])

        # Load peer prompt templates from agent config.
        peer_template = config.get("peer_template", {})
        self._base_prompt = peer_template.get("base_system_prompt", _DEFAULT_BASE_PROMPT)
        self._soft_claiming_addition = peer_template.get("soft_claiming_addition", "")
        self._hard_claiming_addition = peer_template.get("hard_claiming_addition", "")
        self._prediction_addition = peer_template.get("prediction_prompt_addition", "")

        # Assemble the final peer prompt.
        peer_prompt = self._assemble_peer_prompt()

        # Collect worker tools from config (injected by Coordinator).
        worker_tools_dict = config.get("_worker_tools", {})
        worker_tools = list(worker_tools_dict.values()) if worker_tools_dict else []

        # Extract model from an existing agent, or from config.
        model = config.get("_model")
        if model is None:
            # Try to get model from first agent in dict.
            for agent in agents.values():
                if hasattr(agent, "model"):
                    model = agent.model
                    break

        # Initialize blackboard.
        self._blackboard = Blackboard(claiming_mode=self._claiming_mode)

        # Toggle config dict for context filtering.
        toggle_config = {
            "peer_monitoring_visible": self._peer_monitoring_visible,
            "trans_specialist_knowledge": self._trans_specialist_knowledge,
            "predictive_knowledge": self._predictive_knowledge,
        }

        # Use same dict reference so spawned agents are visible to Coordinator.
        self._agents = agents

        # Build shared context.
        self._context = NetworkedContext(
            blackboard=self._blackboard,
            agents=self._agents,
            model=model,
            all_tools=[],  # will be set after peer tools are created
            peer_prompt=peer_prompt,
            agent_max_steps=self._agent_max_steps,
            max_agents=self._max_agents,
            agent_counter=0,
            config=toggle_config,
        )

        # Create peer tools (need context reference).
        # Note: each agent gets its own WriteBlackboard and SpawnPeer
        # instances with agent_name set. ReadBlackboard is shared.
        # For tool creation, we'll create template instances;
        # per-agent instances are created in _create_peer_agent.

        # Build the full tool list for new agents (domain + peer tools).
        # Peer tools are created per-agent in _create_peer_agent.
        self._domain_tools = worker_tools

        # Create initial peer agents.
        self._agent_order = []
        for i in range(1, self._initial_agents + 1):
            name = f"agent_{i}"
            self._context.agent_counter = i
            self._create_peer_agent(name)

        # Set all_tools in context for future spawns (includes peer tools).
        # The peer tool instances in all_tools are templates; SpawnPeer
        # creates proper instances at spawn time. For now, use the
        # domain tools + template peer tools.
        self._context.all_tools = self._build_tool_list("template_agent")

        # Snapshot each agent's full toolset for phase-gating restore.
        if self._workflow_phases:
            for name, agent in self._agents.items():
                self._full_toolsets[name] = dict(agent.tools)

        # Graph-driven mode: load graph definition if provided.
        graph_def = config.get("_graph_def")
        if graph_def is not None:
            self._graph = graph_def
            self._graph_current_state = graph_def.initial_state
            self._graph_state_dict = {
                "execution_success": False,
                "review_verdict": None,
                "review_passed": None,
                "states_visited": [],
            }
            self._graph_complete = False
            # Snapshot toolsets for graph-driven tool filtering.
            for name, agent in self._agents.items():
                self._full_toolsets[name] = dict(agent.tools)

        # Reset state.
        self._rotation_index = 0
        self._total_turns = 0
        self._consecutive_errors = 0

    def set_session_id(self, session_id: str) -> None:
        """Inject a pre-created session_id into the blackboard.

        Called by the coordinator when a pre-hook has already created the
        MCP session.  Writes the session_id as a result entry and pre-
        completes Phase 1 (session_setup) so phase gating advances agents
        directly to Phase 2 (parameter_setting).  Without this, agents
        get stuck in Phase 1 because the task text tells them NOT to call
        create_session, but phase gating only offers Phase 1 tools.
        """
        if self._blackboard is None:
            return

        # Write session_id to blackboard so all agents can read it.
        self._blackboard.write(
            key="session_id",
            value=session_id,
            author="system",
            entry_type="result",
        )

        # Pre-complete Phase 1 (session_setup) — the pre-hook already
        # called create_session + configure_mission.
        self._blackboard.write(
            key="phase_session_setup",
            value=f"completed by pre-hook (session_id={session_id})",
            author="system",
            entry_type="status",
        )

    def next_step(self, history: list, current_state: dict) -> CoordinationAction:
        """Select next agent via rotation and build context."""
        self._total_turns = len(history)

        # Store task from state on first call.
        if "task" in current_state and not self._task:
            self._task = current_state["task"]
            # Write task to blackboard.
            if self._blackboard is not None:
                self._blackboard.write(
                    "task", self._task, "system", "status"
                )

        # Auto-complete workflow phases based on tool calls from last turn.
        self._auto_complete_phases(history)

        # Structural post-turn write: auto-post the previous agent's result to
        # the blackboard so subsequent agents see it in their context without
        # needing to call write_blackboard themselves.  Skips error turns and
        # entries already written (same agent, same key, same content).
        if history and self._blackboard:
            last = history[-1]
            if isinstance(last, AgentMessage) and last.content and not last.error:
                content = last.content
                if len(content) > 800:
                    content = content[:800] + "..."
                self._blackboard.write(
                    key=f"{last.agent_name}_result",
                    value=content,
                    author=last.agent_name,
                    entry_type="result",
                )

        # Track consecutive errors.
        if history:
            last = history[-1]
            if isinstance(last, AgentMessage) and last.error:
                self._consecutive_errors += 1
            else:
                self._consecutive_errors = 0

        if not self._agent_order:
            return CoordinationAction(
                action_type="error",
                agent_name=None,
                input_context="No agents available",
            )

        # ----- Graph-driven mode: one graph state per turn -----
        if self._graph is not None:
            return self._graph_driven_next_step(history, current_state)

        # Update agent order to include any spawned agents.
        current_names = [n for n in self._agent_order if n in self._agents]
        new_names = [
            n for n in self._agents
            if n not in current_names and n != "system"
        ]
        self._agent_order = current_names + new_names

        # Placeholder rotation: simple round-robin.
        if self._rotation_index >= len(self._agent_order):
            self._rotation_index = 0

        agent_name = self._agent_order[self._rotation_index]
        self._rotation_index += 1

        # Snapshot toolset for newly spawned agents (phase gating needs it).
        if self._workflow_phases and agent_name not in self._full_toolsets:
            agent = self._agents.get(agent_name)
            if agent:
                self._full_toolsets[agent_name] = dict(agent.tools)

        # Apply phase gating: filter agent's tools to current phase(s).
        active_phase = self._apply_phase_gate(agent_name)

        # Build input context.
        input_context = self._build_context(
            agent_name, history, current_state, active_phase=active_phase,
        )

        return CoordinationAction(
            action_type="invoke_agent",
            agent_name=agent_name,
            input_context=input_context,
            metadata={
                "turn": self._total_turns + 1,
                "rotation_index": self._rotation_index,
                "total_agents": len(self._agent_order),
            },
        )

    def is_complete(self, history: list, current_state: dict) -> bool:
        """Check if the task is finished."""
        # Check termination keyword in last message.
        if history:
            last = history[-1]
            content = last.content if isinstance(last, AgentMessage) else str(last)
            if self._termination_keyword and self._termination_keyword in content:
                return True

        # Check max turns.
        if len(history) >= self._max_turns:
            return True

        # Check consecutive errors.
        if self._consecutive_errors >= self._max_consecutive_errors:
            return True

        # Check if any agent called mark_task_done (single-signal completion).
        if self._blackboard:
            entry = self._blackboard.get("task_complete")
            if entry is not None and "DONE" in entry.value.upper():
                return True

        # Check if all agents report DONE on blackboard (fallback).
        if self._blackboard and self._agent_order:
            done_agents = set()
            for entry in self._blackboard.read_by_type("status"):
                if "DONE" in entry.value.upper():
                    done_agents.add(entry.author)
            if done_agents >= set(self._agent_order):
                return True

        # Check if graph-driven mode completed (strategy drove the graph).
        if self._graph_complete:
            return True

        # Check if graph-routed handler signalled completion (handler drove the graph).
        if history:
            for msg in reversed(history):
                if isinstance(msg, AgentMessage):
                    if msg.metadata.get("graph_complete"):
                        return True
                    if "graph_state" not in msg.metadata:
                        break

        return False

    # -- Phase gating -----------------------------------------------------------

    def _get_open_phases(self) -> list[dict]:
        """Return workflow phases that are currently available.

        A phase is "open" when:
        - All previous phases' board_keys exist on the blackboard.
        - This phase's board_key does NOT exist on the blackboard.
        """
        if not self._workflow_phases or not self._blackboard:
            return []

        open_phases = []
        for i, phase in enumerate(self._workflow_phases):
            board_key = phase["board_key"]
            # Already completed?
            if self._blackboard.get(board_key) is not None:
                continue
            # Check all previous phases are completed.
            prereqs_met = True
            for prev in self._workflow_phases[:i]:
                if self._blackboard.get(prev["board_key"]) is None:
                    prereqs_met = False
                    break
            if prereqs_met:
                open_phases.append(phase)
        return open_phases

    def _apply_phase_gate(self, agent_name: str) -> str | None:
        """Filter agent's tools to only the current open phase(s).

        Returns a short description of the active phase for context
        injection, or None if gating is not active.
        """
        if not self._workflow_phases:
            return None

        # When staged_pipeline handler is active, skip tool restriction.
        # The handler's completion criteria already enforce ordering.
        if self._handler_is_staged_pipeline:
            return None

        agent = self._agents.get(agent_name)
        if agent is None:
            return None

        # Restore full toolset first (in case it was filtered last turn).
        full_tools = self._full_toolsets.get(agent_name)
        if full_tools:
            agent.tools = dict(full_tools)

        open_phases = self._get_open_phases()

        if not open_phases:
            # All phases done — only keep peer tools + final_answer.
            agent.tools = {
                name: tool for name, tool in agent.tools.items()
                if name in PEER_TOOL_NAMES or name == "final_answer"
            }
            return "ALL_PHASES_COMPLETE"

        # Collect allowed domain tool names from open phases.
        allowed_domain_tools: set[str] = set()
        phase_names = []
        for phase in open_phases:
            allowed_domain_tools.update(phase.get("tools", []))
            phase_names.append(phase["name"])

        # Filter: keep allowed domain tools + peer tools + final_answer.
        # Exclude mark_task_done until all phases are complete — prevents
        # premature termination mid-pipeline.
        agent.tools = {
            name: tool for name, tool in agent.tools.items()
            if name in allowed_domain_tools
            or (name in PEER_TOOL_NAMES and name != "mark_task_done")
            or name == "final_answer"
        }

        return ", ".join(phase_names)

    def _auto_complete_phases(self, history: list) -> None:
        """After an agent's turn, check if any phase's tools were called
        successfully and auto-write the phase board_key to the blackboard."""
        if not self._workflow_phases or not self._blackboard or not history:
            return

        last = history[-1]
        if not isinstance(last, AgentMessage) or last.error:
            return

        # Collect tool names from the last agent's tool calls.
        called_tools: set[str] = set()
        tool_calls = getattr(last, "tool_calls", None) or []
        for tc in tool_calls:
            tool_name = getattr(tc, "tool_name", None) or getattr(tc, "name", "")
            if tool_name and not getattr(tc, "error", None):
                called_tools.add(tool_name)

        if not called_tools:
            return

        # Check each incomplete phase: if any of its tools were called,
        # mark the phase complete.  Special case: parameter_setting phase
        # requires validate_parameters to return valid:true before the
        # phase advances — calling set_aircraft_parameters alone is not
        # enough (params may be invalid, causing simulation failure).
        for phase in self._workflow_phases:
            board_key = phase["board_key"]
            if self._blackboard.get(board_key) is not None:
                continue  # already complete
            phase_tools = set(phase.get("tools", []))
            if not (called_tools & phase_tools):
                continue

            # Gate: if this phase includes validate_parameters, only
            # auto-complete when validation returned valid:true.
            if "validate_parameters" in phase_tools:
                validation_passed = False
                for tc in tool_calls:
                    tc_name = getattr(tc, "tool_name", None) or getattr(tc, "name", "")
                    tc_output = getattr(tc, "output", "") or ""
                    normalized = tc_output.replace(" ", "").replace("'", '"')
                    if tc_name == "validate_parameters" and '"valid":true' in normalized:
                        validation_passed = True
                        break
                if not validation_passed:
                    continue  # don't auto-complete until validation passes

            self._blackboard.write(
                key=board_key,
                value=f"completed by {last.agent_name}",
                author="system",
                entry_type="status",
            )

    # -- Internal helpers ------------------------------------------------------

    def _assemble_peer_prompt(self) -> str:
        """Build the final system prompt from base + toggle additions."""
        prompt = self._base_prompt

        if self._claiming_mode == "soft" and self._soft_claiming_addition:
            prompt += "\n" + self._soft_claiming_addition
        elif self._claiming_mode == "hard" and self._hard_claiming_addition:
            prompt += "\n" + self._hard_claiming_addition

        if self._predictive_knowledge and self._prediction_addition:
            prompt += "\n" + self._prediction_addition

        return prompt

    def _create_peer_agent(self, name: str) -> None:
        """Create a peer agent with full tools and register it."""
        tools = self._build_tool_list(name)

        from smolagents import ToolCallingAgent

        agent = ToolCallingAgent(
            tools=tools,
            model=self._context.model,
            name=name,
            description=f"Peer agent {name}",
            instructions=self._context.peer_prompt,
            max_steps=self._agent_max_steps,
            add_base_tools=False,
        )

        self._agents[name] = agent
        self._agent_order.append(name)

    def _build_tool_list(self, agent_name: str) -> list:
        """Build domain tools + per-agent peer tools."""
        peer_tools = [
            ReadBlackboard(self._context),
            WriteBlackboard(self._context, agent_name=agent_name),
            SpawnPeer(self._context, agent_name=agent_name),
            MarkTaskDone(self._context, agent_name=agent_name),
        ]
        return list(self._domain_tools) + peer_tools

    # -- Graph-driven mode ------------------------------------------------------

    def _graph_driven_next_step(
        self, history: list, current_state: dict,
    ) -> CoordinationAction:
        """Execute one graph state per turn, strategy-driven.

        Instead of delegating to the graph-routed handler (which runs the
        full graph in one call), the strategy drives the state machine:
        one state per coordinator turn.  This lets the networked rotation,
        blackboard, and peer tools operate between states.
        """
        # Advance graph from previous turn's output.
        if history:
            self._advance_graph_state(history)

        # Terminal check.
        if (self._graph_current_state is None
                or self._graph_current_state in self._graph.terminal_states):
            self._graph_complete = True
            return CoordinationAction(
                action_type="terminate",
                agent_name=None,
                input_context="Graph reached terminal state",
            )

        state_def = self._graph.states.get(self._graph_current_state)
        if state_def is None:
            return CoordinationAction(
                action_type="error",
                agent_name=None,
                input_context=f"Graph state {self._graph_current_state!r} not found",
            )

        # Skip routing-only states (agent=None) — evaluate transitions
        # immediately and recurse.
        if state_def.agent is None:
            self._graph_state_dict["states_visited"].append(
                self._graph_current_state,
            )
            next_state = self._evaluate_graph_transitions(state_def)
            if next_state is None:
                self._graph_complete = True
                return CoordinationAction(
                    action_type="terminate",
                    agent_name=None,
                    input_context="Graph stuck at routing state",
                )
            self._graph_current_state = next_state
            return self._graph_driven_next_step(history, current_state)

        # Resolve agent: look up role alias in agents dict.
        agent_name = state_def.agent
        if agent_name not in self._agents:
            # Try fuzzy resolution (the aliases may have been registered).
            from src.coordination.graph_definition import resolve_agent_for_role
            try:
                resolved = resolve_agent_for_role(
                    agent_name, self._agents, self._graph_current_state,
                )
                # Find the key in the agents dict for this resolved agent.
                for key, val in self._agents.items():
                    if val is resolved:
                        agent_name = key
                        break
            except ValueError:
                # Fallback: round-robin among peers.
                peer_names = [k for k in self._agent_order
                              if k.startswith("agent_")]
                if peer_names:
                    idx = self._rotation_index % len(peer_names)
                    agent_name = peer_names[idx]

        # Restore full toolset, but gate mark_task_done until graph reaches
        # a terminal state (prevents premature completion mid-pipeline).
        agent_obj = self._agents.get(agent_name)
        if agent_obj is not None:
            full_tools = self._full_toolsets.get(agent_name)
            if full_tools:
                agent_obj.tools = {
                    name: tool for name, tool in full_tools.items()
                    if name != "mark_task_done"
                }

        # Build context: focused graph state prompt only — do NOT include
        # the full task description, as that causes agents to run the
        # entire pipeline instead of just their assigned step.
        prompt = state_def.agent_prompt or ""
        try:
            prompt = prompt.format(**self._graph_state_dict)
        except KeyError:
            pass  # Missing keys left as-is

        parts = [
            "You are part of a networked team of agents. Each agent handles "
            "ONE step of the workflow. Complete ONLY the step below, write "
            "your result to the blackboard, then stop. Do NOT call tools "
            "for other steps — your peers will handle those.",
            f"GRAPH STATE: {self._graph_current_state}\n"
            f"Role: {state_def.agent}\n\n{prompt}",
        ]

        # Blackboard context.
        if self._blackboard:
            toggle_config = {
                "peer_monitoring_visible": self._peer_monitoring_visible,
                "trans_specialist_knowledge": self._trans_specialist_knowledge,
                "predictive_knowledge": self._predictive_knowledge,
            }
            bb_ctx = self._blackboard.to_context_string(
                agent_name, toggle_config,
            )
            parts.append(f"Current Blackboard State:\n{bb_ctx}")

        # Recent history.
        if history:
            recent = history[-self._max_recent_messages:]
            lines = []
            for msg in recent:
                if isinstance(msg, AgentMessage):
                    status = f" [ERROR: {msg.error}]" if msg.error else ""
                    lines.append(
                        f"[Turn {msg.turn_number}] {msg.agent_name}: "
                        f"{msg.content[:500]}{status}"
                    )
            if lines:
                parts.append("Recent History:\n" + "\n".join(lines))

        input_context = "\n\n".join(parts)

        self._rotation_index += 1

        # Resolve the actual peer agent name behind the role alias
        # so the viewer can distinguish real agents from role labels.
        peer_name = agent_name
        if agent_obj is not None:
            for key, val in self._agents.items():
                if val is agent_obj and key.startswith("agent_"):
                    peer_name = key
                    break

        # Build metadata with graph + resource state for SDA metrics.
        meta = {
            "turn": self._total_turns + 1,
            "rotation_index": self._rotation_index,
            "total_agents": len(self._agent_order),
            "graph_state": self._graph_current_state,
            "graph_role": state_def.agent,
            "peer_agent": peer_name,
            "bypass_handler": True,
        }
        # SDA fields: complexity and resource utilization.
        complexity = self._graph_state_dict.get("complexity")
        if complexity:
            meta["complexity"] = complexity
        visited = self._graph_state_dict.get("states_visited", [])
        total_states = len([
            s for s in self._graph.states.values() if s.agent is not None
        ])
        meta["passes_remaining"] = max(0, total_states - len(visited) - 1)
        meta["passes_max"] = total_states

        return CoordinationAction(
            action_type="invoke_agent",
            agent_name=agent_name,
            input_context=input_context,
            metadata=meta,
        )

    def _advance_graph_state(self, history: list) -> None:
        """Advance the graph state machine based on the last turn's output."""
        if not history or self._graph is None:
            return

        last = history[-1]
        if not isinstance(last, AgentMessage) or last.error:
            return

        content = last.content or ""

        # Import extraction helpers from the graph handler module.
        import re

        from src.coordination.graph_routed_handler import (
            _extract_complexity,
            _extract_execution_result,
            _extract_review_result,
        )

        # Extract structured data from agent output.
        complexity = _extract_complexity(content)
        if complexity is not None:
            self._graph_state_dict["complexity"] = complexity

        review = _extract_review_result(content)
        if review:
            self._graph_state_dict.update(review)

        exec_result = _extract_execution_result(content)
        if exec_result:
            self._graph_state_dict.update(exec_result)

        # Session ID extraction.
        session_match = re.search(
            r"(?:SESSION_ID|session_id)['\"\s:=]+([0-9a-f-]{36})",
            content, re.IGNORECASE,
        )
        if session_match:
            self._graph_state_dict["session_id"] = session_match.group(1)

        # Also try tool call outputs for session_id.
        if "session_id" not in self._graph_state_dict:
            tool_calls = getattr(last, "tool_calls", None) or []
            for tc in tool_calls:
                if getattr(tc, "tool_name", "") == "create_session":
                    output = getattr(tc, "output", "") or ""
                    uuid_match = re.search(
                        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}"
                        r"-[0-9a-f]{4}-[0-9a-f]{12}",
                        output, re.IGNORECASE,
                    )
                    if uuid_match:
                        self._graph_state_dict["session_id"] = (
                            uuid_match.group(0)
                        )
                        break

        # Convergence.
        lower = content.lower()
        if "converged" in lower:
            if "not converge" in lower or "failed to converge" in lower:
                self._graph_state_dict["converged"] = False
            else:
                self._graph_state_dict["converged"] = True

        # Record state visit.
        self._graph_state_dict["states_visited"].append(
            self._graph_current_state,
        )

        # Write graph state result to blackboard for peer visibility.
        if self._blackboard and content:
            self._blackboard.write(
                key=f"graph_{self._graph_current_state}",
                value=content[:800] if len(content) > 800 else content,
                author=last.agent_name,
                entry_type="result",
            )

        # Evaluate transitions.
        state_def = self._graph.states.get(self._graph_current_state)
        if state_def is not None:
            next_state = self._evaluate_graph_transitions(state_def)
            if next_state is not None:
                self._graph_current_state = next_state
            else:
                # No transition matched — stuck.  Mark complete to avoid loop.
                self._graph_complete = True

    def _evaluate_graph_transitions(self, state_def) -> str | None:
        """Evaluate transitions from a graph state, return next state."""
        from src.coordination.condition_evaluator import (
            ConditionParseError,
            evaluate_condition,
        )
        for trans in state_def.transitions:
            try:
                result = evaluate_condition(
                    trans.condition, self._graph_state_dict,
                )
            except ConditionParseError:
                continue
            if result.matched:
                return trans.target
        return None

    def _build_context(
        self,
        agent_name: str,
        history: list,
        current_state: dict,
        active_phase: str | None = None,
    ) -> str:
        """Build the input context string for an agent's turn."""
        parts = []

        # Task description.
        task = current_state.get("task", self._task)
        if task:
            parts.append(f"Task: {task}")

        # Phase gating context — tell agents which phase they're in AND
        # show the full workflow so they understand what comes next.
        if active_phase == "ALL_PHASES_COMPLETE":
            parts.append(
                "PHASE STATUS: All workflow phases are complete. "
                "If results look good, call mark_task_done with the final metrics."
            )
        elif active_phase and self._workflow_phases:
            phase_map = "\n".join(
                f"  {'>> ' if p['name'] in active_phase else '   '}"
                f"Phase {i+1} — {p['name']}: {', '.join(p.get('tools', []))}"
                f"{' [DONE]' if self._blackboard and self._blackboard.get(p['board_key']) else ''}"
                for i, p in enumerate(self._workflow_phases)
            )
            parts.append(
                f"YOUR CURRENT PHASE: {active_phase}\n"
                f"You can ONLY use tools for this phase. Complete it and "
                f"post results to the blackboard.\n\n"
                f"FULL WORKFLOW:\n{phase_map}\n"
                f"NOTE: parameter_setting phase requires validate_parameters "
                f"to return valid:true before advancing."
            )
        elif active_phase:
            parts.append(
                f"YOUR CURRENT PHASE: {active_phase}\n"
                "You can ONLY use tools for this phase. Complete it and "
                "post results to the blackboard. Do NOT attempt other phases."
            )

        # Blackboard state (filtered by toggles).
        if self._blackboard:
            toggle_config = {
                "peer_monitoring_visible": self._peer_monitoring_visible,
                "trans_specialist_knowledge": self._trans_specialist_knowledge,
                "predictive_knowledge": self._predictive_knowledge,
            }
            bb_context = self._blackboard.to_context_string(agent_name, toggle_config)
            parts.append(f"Current Blackboard State:\n{bb_context}")

        # Recent history (truncated).
        if history:
            recent = history[-self._max_recent_messages:]
            history_lines = []
            for msg in recent:
                if isinstance(msg, AgentMessage):
                    status = f" [ERROR: {msg.error}]" if msg.error else ""
                    history_lines.append(
                        f"[Turn {msg.turn_number}] {msg.agent_name}: "
                        f"{msg.content[:500]}{status}"
                    )
            if history_lines:
                parts.append("Recent History:\n" + "\n".join(history_lines))

        return "\n\n".join(parts)

    # -- Properties ------------------------------------------------------------

    @property
    def blackboard(self) -> Blackboard | None:
        return self._blackboard

    @property
    def context(self) -> NetworkedContext | None:
        return self._context

    @property
    def agent_order(self) -> list[str]:
        return list(self._agent_order)

    @property
    def peer_prompt(self) -> str:
        return self._context.peer_prompt if self._context else ""


# -- Default prompt (used when config doesn't provide one) ---------------------

_DEFAULT_BASE_PROMPT = """\
You are a peer agent in a collaborative team. There is no manager \
or leader — all agents are equals working together to solve a task.

You have access to a shared blackboard where you and your peers \
post status updates, share results, and flag gaps.

Your process each turn:
1. Call read_blackboard to see what's happening
2. Look at what's been completed and what gaps exist
3. Decide what you can contribute
4. Do your work using your available tools
5. Post your results to the blackboard with write_blackboard \
   using entry_type "result"
6. Update your status with write_blackboard using entry_type "status"
7. If you see a gap that no existing agent can fill and more agents \
   would help, call spawn_peer

When the overall task is fully solved (all subtasks completed and \
results posted), call mark_task_done with a brief summary of what \
was accomplished."""
