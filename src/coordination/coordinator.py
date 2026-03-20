"""Example coordinator runner — executes a CoordinationStrategy in a loop.

This is ONE way to run agents. You can replace this class entirely with a
different execution model — as long as your code writes AgentMessage entries
to the InstrumentationLogger, the UI and metrics will continue to work.
"""

import importlib
import time
from typing import Any

from src.coordination.history import AgentMessage, SharedHistory, ToolCallRecord
from src.coordination.strategy import CoordinationAction, CoordinationResult, CoordinationStrategy
from src.coordination.termination import TerminationChecker

# ---------------------------------------------------------------------------
# smolagents extraction helpers (Fix 6 + Fix 7)
# ---------------------------------------------------------------------------


def _extract_tool_calls(agent: Any) -> list[ToolCallRecord]:
    """Extract ToolCallRecords from a smolagents agent after agent.run()."""
    tool_calls: list[ToolCallRecord] = []
    logs = getattr(agent, "logs", None) or []
    if not logs:
        mem = getattr(agent, "memory", None)
        logs = getattr(mem, "steps", None) or []
    for step in logs:
        step_calls = getattr(step, "tool_calls", None)
        if not step_calls:
            continue
        obs = getattr(step, "observations", "") or ""
        for tc in step_calls:
            name = getattr(tc, "name", "") or getattr(tc, "tool_name", "")
            args = getattr(tc, "arguments", {}) or {}
            tool_calls.append(
                ToolCallRecord(
                    tool_name=name,
                    inputs=args if isinstance(args, dict) else {},
                    output=str(obs)[:2000] if obs else "",
                    duration_seconds=0.0,
                )
            )
    return tool_calls


def _extract_token_count(agent: Any, content: str) -> tuple[int | None, bool]:
    """Return (token_count, is_estimated). is_estimated=True means fallback."""
    for attr in ("token_count", "total_tokens"):
        val = getattr(agent, attr, None)
        if isinstance(val, int) and val > 0:
            return val, False
    mem = getattr(agent, "memory", None)
    steps = getattr(mem, "steps", None) if mem else None
    if steps:
        total = 0
        for step in steps:
            usage = getattr(step, "token_usage", None)
            if usage:
                if isinstance(usage, dict):
                    total += usage.get("total_tokens", 0) or (
                        usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                    )
                elif isinstance(usage, int):
                    total += usage
        if total:
            return total, False
    if content:
        return max(1, len(content) // 4), True
    return None, True


# Strategy name → fully-qualified class path.
_STRATEGY_MAP = {
    "sequential": "src.coordination.strategies.sequential.SequentialStrategy",
    "graph_routed": "src.coordination.strategies.graph_routed.GraphRoutedStrategy",
    "orchestrated": "src.coordination.strategies.orchestrated.OrchestratedStrategy",
    "networked": "src.coordination.strategies.networked.NetworkedStrategy",
}


def _load_strategy(name: str) -> CoordinationStrategy:
    """Instantiate a strategy by name from the registry."""
    if name not in _STRATEGY_MAP:
        raise ValueError(f"Unknown strategy {name!r}. Available: {list(_STRATEGY_MAP.keys())}")
    fqn = _STRATEGY_MAP[name]
    module_path, class_name = fqn.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls()


class Coordinator:
    """Runs a coordination strategy loop over a set of agents."""

    def __init__(self, agents: dict, strategy: CoordinationStrategy, config: dict, logger=None, execution_handler=None):
        """Create a Coordinator with explicit agents, strategy, and config.

        For convenience, use Coordinator.from_config() to build everything
        from YAML files.
        """
        self.agents = agents
        self.strategy = strategy
        self.history = SharedHistory()
        self.termination = TerminationChecker(config)
        self.config = config
        self.logger = logger
        self.execution_handler = execution_handler
        self._turn_counter = 0

    @classmethod
    def from_config(cls, config, logger=None, strategy_override: str | None = None):
        """Build a Coordinator from an AppConfig and YAML files.

        Args:
            config: AppConfig instance (from load_config).
            logger: Optional InstrumentationLogger.
            strategy_override: If set, overrides the strategy name from
                coordination.yaml (e.g. "sequential", "orchestrated").

        Returns:
            A fully wired Coordinator ready to .run().
        """
        from src.agents.agent_factory import create_agents_from_yaml
        from src.config.loader import load_yaml
        from src.llm.model_loader import load_model

        # Load LLM
        model = load_model(config.llm)

        # Load agents
        registry = create_agents_from_yaml(config.agents_config, model, config)
        agents = {name: registry.get(name) for name in registry.list_names()}

        # Load coordination config
        coord_config = load_yaml(config.coordination_config)
        strategy_name = strategy_override or coord_config.get("strategy", "sequential")
        strategy = _load_strategy(strategy_name)

        # Pass reliability config so strategies can enable guardrails
        # regardless of whether the model is ThinkingModel or OpenAIServerModel.
        coord_config["_reliability_config"] = config.llm.reliability or {}

        # For orchestrated strategy, pass all available worker tools so the
        # orchestrator can assign them to dynamically created agents.
        if strategy_name == "orchestrated":
            from src.tools.mock_tools import MOCK_TOOLS, create_mock_tool
            from src.tools.tool_loader import load_tools_for_agent

            if config.mcp.mode == "mock":
                all_tools = [create_mock_tool(n) for n in MOCK_TOOLS]
            else:
                all_tools = load_tools_for_agent([], config)
            coord_config["_worker_tools"] = {t.name: t for t in all_tools}

            # Pass required_tool_phases and required_result_signals from agents
            # config so the orchestrator validates coverage and result signals
            # before accepting DELEGATION_COMPLETE.
            agents_yaml = load_yaml(config.agents_config)
            phases = agents_yaml.get("required_tool_phases")
            if phases:
                coord_config["_required_tool_phases"] = phases
            result_signals = agents_yaml.get("required_result_signals")
            if result_signals:
                coord_config["_required_result_signals"] = result_signals

        # For sequential strategy, pass all available worker tools and model
        # so the strategy can create stage agents with tool restrictions.
        if strategy_name == "sequential":
            from src.tools.mock_tools import MOCK_TOOLS, create_mock_tool
            from src.tools.tool_loader import load_tools_for_agent

            if config.mcp.mode == "mock":
                all_tools = [create_mock_tool(n) for n in MOCK_TOOLS]
            else:
                all_tools = load_tools_for_agent([], config)
            coord_config["_worker_tools"] = {t.name: t for t in all_tools}
            coord_config["_model"] = model

            # Load stage defaults and templates from agents config if available.
            agents_yaml = load_yaml(config.agents_config)
            if "stage_defaults" in agents_yaml:
                coord_config["stage_defaults"] = agents_yaml["stage_defaults"]
            if "templates" in agents_yaml:
                coord_config["templates"] = agents_yaml["templates"]

        # For networked strategy, pass all available worker tools and model
        # so the strategy can create peer agents with full tool sets.
        if strategy_name == "networked":
            from src.tools.mock_tools import MOCK_TOOLS, create_mock_tool
            from src.tools.tool_loader import load_tools_for_agent

            if config.mcp.mode == "mock":
                all_tools = [create_mock_tool(n) for n in MOCK_TOOLS]
            else:
                all_tools = load_tools_for_agent([], config)
            coord_config["_worker_tools"] = {t.name: t for t in all_tools}
            coord_config["_model"] = model

            # Load peer template from agents config if available.
            agents_yaml = load_yaml(config.agents_config)
            if "peer_template" in agents_yaml:
                coord_config["peer_template"] = agents_yaml["peer_template"]

        # Build execution handler from config.
        handler_name = coord_config.get("execution_handler", "placeholder")
        execution_handler = None
        if handler_name == "iterative_feedback":
            from src.coordination.iterative_feedback_handler import (
                IterativeFeedbackHandler,
            )

            # Load iterative feedback config if available.
            ifb_config = coord_config.get("iterative_feedback", {})
            if not ifb_config:
                ifb_path = getattr(config, "iterative_feedback_config", None)
                if ifb_path:
                    ifb_yaml = load_yaml(ifb_path)
                    ifb_config = ifb_yaml.get("iterative_feedback", ifb_yaml)
            execution_handler = IterativeFeedbackHandler(ifb_config)

        if handler_name == "graph_routed":
            from src.coordination.graph_routed_handler import (
                GraphRoutedHandler,
            )

            gr_config = coord_config.get("graph_routed", {})
            if not gr_config:
                gr_path = getattr(config, "graph_routed_config", None)
                if gr_path:
                    gr_yaml = load_yaml(gr_path)
                    gr_config = gr_yaml.get("graph_routed", gr_yaml)
            execution_handler = GraphRoutedHandler(gr_config)

            # Extract unique agent roles from the graph so the orchestrated
            # strategy can inform the orchestrator what roles to create.
            try:
                graph = execution_handler._load_graph({})
                roles = sorted({s.agent for s in graph.states.values() if s.agent})
                if roles:
                    coord_config["_graph_roles"] = roles
            except Exception:
                pass

        if handler_name == "staged_pipeline":
            from src.coordination.staged_pipeline_handler import (
                StagedPipelineHandler,
            )

            sp_config = coord_config.get("staged_pipeline", {})
            if not sp_config:
                sp_path = getattr(config, "staged_pipeline_config", None)
                if sp_path:
                    sp_yaml = load_yaml(sp_path)
                    sp_config = sp_yaml.get("staged_pipeline", sp_yaml)
            execution_handler = StagedPipelineHandler(sp_config)

        return cls(
            agents,
            strategy,
            coord_config,
            logger=logger,
            execution_handler=execution_handler,
        )

    def run(self, task: str, *, session_id: str | None = None) -> CoordinationResult:
        """Execute the strategy loop until termination.

        Args:
            task: The user's task string to solve.
            session_id: Pre-created session ID (from pre-hook). If provided,
                injected into graph-routed handler so {session_id} placeholders
                resolve correctly on every node.

        Returns:
            CoordinationResult with final output, history, and metrics.
        """
        state = {"task": task}
        self.strategy.initialize(self.agents, self.config)

        # If a session_id was provided (pre-hook), inject it into both
        # the execution handler (for graph node prompt placeholders) and
        # the strategy (for orchestrated worker task context).
        if session_id:
            if self.execution_handler is not None and hasattr(self.execution_handler, "set_session_id"):
                self.execution_handler.set_session_id(session_id)
            if hasattr(self.strategy, "set_session_id"):
                self.strategy.set_session_id(session_id)

        while not self.termination.should_stop(self.history):
            # Let the strategy signal completion (e.g. orchestrated
            # strategy detects final_answer or delegation stall).
            if self.history and self.strategy.is_complete(
                self.history.get_all(),
                state,
            ):
                break

            action = self.strategy.next_step(
                self.history.get_all(),
                state,
            )

            if action.action_type == "terminate":
                break

            if action.action_type == "error":
                self._log_error(action)
                break

            if action.agent_name not in self.agents:
                self._log_error(
                    CoordinationAction(
                        action_type="error",
                        agent_name=action.agent_name,
                        input_context=f"Agent {action.agent_name!r} not found",
                    )
                )
                break

            messages = self._execute_agent(action)
            for message in messages:
                self.history.append(message)
                if self.logger is not None:
                    self.logger.log_turn(message)

        # Build result
        final_output = ""
        if len(self.history) > 0:
            final_output = self.history.get_recent(1)[0].content

        metrics = {}
        if self.logger is not None:
            metrics = self.logger.compute_metrics()

        return CoordinationResult(
            final_output=final_output,
            history=self.history.get_all(),
            metrics=metrics,
        )

    def _execute_agent(self, action: CoordinationAction) -> list[AgentMessage]:
        """Run a single agent and return AgentMessage(s).

        When an execution_handler is configured, delegates to it (which may
        retry the agent and produce multiple messages). Otherwise runs the
        agent directly once.

        The handler is bypassed during the orchestrated strategy's creation
        phase (metadata phase="creation") since the orchestrator needs to
        run its own ReAct loop to create agents, not be routed through
        the handler's execution model.  It is also bypassed when the action
        carries ``bypass_handler=True`` (e.g. orchestrator review turns in
        active execution mode).
        """
        phase = (action.metadata or {}).get("phase")
        bypass = (action.metadata or {}).get("bypass_handler", False)
        use_handler = self.execution_handler is not None and phase != "creation" and not bypass
        if use_handler:
            from src.coordination.execution_handler import Assignment

            assignment = Assignment(
                agent_name=action.agent_name,
                task=action.input_context,
                assigned_at_turn=self._turn_counter,
            )
            # Snapshot blackboard before handler.execute() for post-processing.
            bb = getattr(self.strategy, "_blackboard", None)
            net_ctx = getattr(self.strategy, "context", None)
            bb_writes_before = bb.write_count if bb is not None else 0
            bb_conflicts_before = bb.claim_conflicts if bb is not None else 0
            peers_before = len(getattr(net_ctx, "spawned_agents", [])) if net_ctx is not None else 0
            msgs = self.execution_handler.execute(
                [assignment],
                self.agents,
                logger=None,  # coordinator logs turns itself
                turn_offset=self._turn_counter,
                action_metadata=dict(action.metadata or {}),
            )
            # Post-process: inject blackboard deltas into handler messages.
            if bb is not None and msgs:
                total_writes = bb.write_count - bb_writes_before
                bb_size = len(bb.read_all())
                total_conflicts = bb.claim_conflicts - bb_conflicts_before
                for m in msgs:
                    m.metadata.setdefault("blackboard_writes", total_writes)
                    m.metadata.setdefault("blackboard_size", bb_size)
                    m.metadata.setdefault("claim_conflicts", total_conflicts)
            if net_ctx is not None and msgs:
                peers_now = len(getattr(net_ctx, "spawned_agents", []))
                for m in msgs:
                    m.metadata.setdefault("peers_spawned", peers_now - peers_before)
            if msgs:
                self._turn_counter = msgs[-1].turn_number
            return msgs

        self._turn_counter += 1
        agent = self.agents[action.agent_name]
        start = time.monotonic()

        # Base metadata from action (carries OS-level context: phase, stage,
        # rotation index, etc. — already populated by each strategy's next_step).
        msg_metadata = dict(action.metadata or {})

        # Snapshot blackboard counters for networked per-turn deltas (Fix 3).
        bb = getattr(self.strategy, "_blackboard", None)
        net_ctx = getattr(self.strategy, "context", None)
        bb_writes_before = bb.write_count if bb is not None else 0
        bb_conflicts_before = bb.claim_conflicts if bb is not None else 0
        peers_before = len(getattr(net_ctx, "spawned_agents", [])) if net_ctx is not None else 0

        try:
            result = agent.run(action.input_context)
            content = str(result) if result is not None else ""
            duration = time.monotonic() - start

            # Post-run extraction (Fix 6 + Fix 7).
            tool_calls = _extract_tool_calls(agent)
            token_count, token_estimated = _extract_token_count(agent, content)

            # Blackboard deltas for networked strategy (Fix 3).
            if bb is not None:
                msg_metadata["blackboard_writes"] = bb.write_count - bb_writes_before
                msg_metadata["blackboard_size"] = len(bb.read_all())
                msg_metadata["claim_conflicts"] = bb.claim_conflicts - bb_conflicts_before
            if net_ctx is not None:
                peers_now = len(getattr(net_ctx, "spawned_agents", []))
                msg_metadata["peers_spawned"] = peers_now - peers_before
            if token_estimated:
                msg_metadata["token_count_estimated"] = True

            return [
                AgentMessage(
                    agent_name=action.agent_name,
                    content=content,
                    turn_number=self._turn_counter,
                    timestamp=time.time(),
                    duration_seconds=duration,
                    tool_calls=tool_calls,
                    token_count=token_count,
                    metadata=msg_metadata,
                )
            ]
        except Exception as e:
            duration = time.monotonic() - start
            if bb is not None:
                msg_metadata["blackboard_writes"] = bb.write_count - bb_writes_before
                msg_metadata["blackboard_size"] = len(bb.read_all())
                msg_metadata["claim_conflicts"] = bb.claim_conflicts - bb_conflicts_before
            if net_ctx is not None:
                peers_now = len(getattr(net_ctx, "spawned_agents", []))
                msg_metadata["peers_spawned"] = peers_now - peers_before
            return [
                AgentMessage(
                    agent_name=action.agent_name,
                    content="",
                    turn_number=self._turn_counter,
                    timestamp=time.time(),
                    duration_seconds=duration,
                    error=str(e),
                    metadata=msg_metadata,
                )
            ]

    def _log_error(self, action: CoordinationAction) -> None:
        """Log a coordination-level error as an AgentMessage."""
        self._turn_counter += 1
        msg = AgentMessage(
            agent_name=action.agent_name or "coordinator",
            content=action.input_context,
            turn_number=self._turn_counter,
            timestamp=time.time(),
            error=action.input_context,
        )
        self.history.append(msg)
        if self.logger is not None:
            self.logger.log_turn(msg)
