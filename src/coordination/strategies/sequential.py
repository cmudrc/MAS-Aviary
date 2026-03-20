"""Sequential pipeline coordination strategy with template-based decomposition.

Replaces the old simple sequential strategy. Agents are organized into
ordered stages with explicit tool restrictions per stage, interface
contracts, and support for named pipeline templates (linear, v_model,
mbse, custom).

Decomposition modes:
  - "human": pipeline defined by template in config (default)
  - "llm": temporary planner agent generates the pipeline at runtime

Execution:
  - Stages run in template-defined order
  - Each stage's output is passed to the next stage
  - Interface contracts describe what passes between stages
  - Tool restrictions enforced per stage
  - PlaceholderExecutor handles actual execution
"""

import json
import re

from src.coordination.history import AgentMessage
from src.coordination.pipeline_templates import (
    PipelineStage,
    PipelineTemplate,
    load_template,
    resolve_tools,
)
from src.coordination.strategy import CoordinationAction, CoordinationStrategy


class SequentialStrategy(CoordinationStrategy):
    """Template-based sequential pipeline strategy."""

    def __init__(self):
        # Config (set during initialize).
        self._decomposition_mode: str = "human"
        self._pipeline_template_name: str = "linear"
        self._validate_interfaces: bool = False
        self._stage_max_steps: int = 8
        self._termination_keyword: str = "TASK_COMPLETE"
        self._max_turns: int = 30
        self._base_instructions: str = ""

        # Runtime state.
        self._template: PipelineTemplate | None = None
        self._agents: dict = {}
        self._stage_order: list[str] = []
        self._current_stage_index: int = 0
        self._total_turns: int = 0
        self._interface_results: list[dict] = []  # per-stage validation
        self._task: str = ""

        # Shared state extracted from stage outputs.  Keys like session_id
        # are captured once and injected into every subsequent stage's
        # context, preserving modularity (each stage only sees its
        # predecessor's handoff + the shared state header).
        self._shared_state: dict[str, str] = {}

        # When True, the graph-routed handler drives the full state
        # machine in a single execute() call.  The sequential strategy
        # emits one action (using the first stage agent) and then
        # terminates — the graph handler replaces the stage pipeline.
        self._graph_routed_mode: bool = False
        self._graph_routed_done: bool = False

    def initialize(self, agents: dict, config: dict) -> None:
        """Set up strategy from agents dict and coordination config."""
        seq_config = config.get("sequential", {})

        # Read config.
        self._decomposition_mode = seq_config.get("decomposition_mode", "human")
        self._pipeline_template_name = seq_config.get("pipeline_template", "linear")
        self._validate_interfaces = seq_config.get("validate_interfaces", False)
        self._stage_max_steps = seq_config.get("stage_max_steps", 8)

        term_config = config.get("termination", {})
        self._termination_keyword = term_config.get("keyword", "TASK_COMPLETE")
        self._max_turns = term_config.get("max_turns", 30)

        # Load base instructions from agent template config.
        stage_defaults = config.get("stage_defaults", {})
        self._base_instructions = stage_defaults.get("base_instructions", "")

        # Get templates from config (from agents YAML).
        templates_config = config.get("templates", {})

        # Get available domain tools.
        worker_tools_dict = config.get("_worker_tools", {})

        # Get model.
        model = config.get("_model")
        if model is None:
            for agent in agents.values():
                if hasattr(agent, "model"):
                    model = agent.model
                    break

        # Use same dict reference so Coordinator sees all agents.
        self._agents = agents

        if self._decomposition_mode == "human":
            self._init_human_mode(
                config, worker_tools_dict, model, templates_config,
            )
        elif self._decomposition_mode == "llm":
            self._init_llm_mode(
                config, worker_tools_dict, model, templates_config,
            )
        else:
            raise ValueError(
                f"Unknown decomposition_mode: {self._decomposition_mode!r}. "
                "Must be 'human' or 'llm'."
            )

        # Reset state.
        self._current_stage_index = 0
        self._total_turns = 0
        self._interface_results = []
        self._shared_state = {}

        # Detect graph-routed handler: the graph handler runs the full
        # state machine in one execute() call, so the sequential pipeline
        # should emit a single action and then terminate.
        self._graph_routed_mode = (
            config.get("execution_handler") == "graph_routed"
        )
        self._graph_routed_done = False

    def next_step(self, history: list, current_state: dict) -> CoordinationAction:
        """Advance to the current stage and return an action."""
        self._total_turns = len(history)

        # Store task on first call.
        if "task" in current_state and not self._task:
            self._task = current_state["task"]

        # Graph-routed mode: emit one action, then terminate.  The graph
        # handler runs the full state machine (TASK_CLASSIFIED → COMPLETE)
        # in a single execute() call — no need for per-stage iteration.
        if self._graph_routed_mode:
            if self._graph_routed_done:
                return CoordinationAction(
                    action_type="terminate",
                    agent_name=None,
                    input_context="",
                    metadata={"reason": "graph_routed_complete"},
                )
            self._graph_routed_done = True
            # Use the first stage agent as the entry point; the graph
            # handler ignores the agent name and routes by its own graph.
            first_stage = self._stage_order[0] if self._stage_order else "agent"
            return CoordinationAction(
                action_type="invoke_agent",
                agent_name=first_stage,
                input_context=self._task,
                metadata={
                    "stage_name": "graph_routed_full",
                    "pipeline_template": self._pipeline_template_name,
                },
            )

        # Check if pipeline complete.
        if self._current_stage_index >= len(self._stage_order):
            return CoordinationAction(
                action_type="terminate",
                agent_name=None,
                input_context="",
                metadata={
                    "reason": "pipeline_complete",
                    "stages_completed": len(self._stage_order),
                },
            )

        stage_name = self._stage_order[self._current_stage_index]
        stage = self._get_stage(stage_name)

        # Perform interface validation on previous stage's output.
        if self._validate_interfaces and history and self._current_stage_index > 0:
            prev_stage = self._get_stage(
                self._stage_order[self._current_stage_index - 1]
            )
            last_msg = history[-1]
            content = last_msg.content if isinstance(last_msg, AgentMessage) else str(last_msg)
            valid = _validate_interface(content, prev_stage.interface_output)
            self._interface_results.append({
                "stage": prev_stage.name,
                "valid": valid,
                "interface_output": prev_stage.interface_output,
            })

        # Extract shared state (e.g. session_id) from previous stage output.
        # Scans both agent content and tool call outputs so state is captured
        # even when the agent's final text fails parsing but tools succeeded.
        if history and self._current_stage_index > 0:
            last_msg = history[-1]
            prev_content = last_msg.content if isinstance(last_msg, AgentMessage) else str(last_msg)
            self._extract_shared_state(prev_content)
            if isinstance(last_msg, AgentMessage):
                for tc in last_msg.tool_calls:
                    if tc.output and tc.error is None:
                        self._extract_shared_state(tc.output)

        # Build input context.
        input_context = self._build_stage_context(
            stage, history, current_state,
        )

        self._current_stage_index += 1

        return CoordinationAction(
            action_type="invoke_agent",
            agent_name=stage_name,
            input_context=input_context,
            metadata={
                "stage_name": stage_name,
                "stage_index": self._current_stage_index - 1,
                "total_stages": len(self._stage_order),
                "pipeline_template": self._pipeline_template_name,
            },
        )

    def is_complete(self, history: list, current_state: dict) -> bool:
        """Check if the pipeline is finished."""
        # Graph-routed mode: done after the single execute() call.
        if self._graph_routed_mode and self._graph_routed_done:
            return True

        # All stages completed.
        if self._current_stage_index >= len(self._stage_order):
            return True

        # Termination keyword in last message.
        if history:
            last = history[-1]
            content = last.content if isinstance(last, AgentMessage) else str(last)
            if self._termination_keyword and self._termination_keyword in content:
                return True

        # Max turns.
        if len(history) >= self._max_turns:
            return True

        return False

    # -- Initialization modes --------------------------------------------------

    def _init_human_mode(self, config, worker_tools_dict, model, templates_config):
        """Create agents from the configured pipeline template."""
        custom_stages_raw = config.get("sequential", {}).get("custom_stages", [])
        self._template = load_template(
            self._pipeline_template_name,
            custom_stages=custom_stages_raw or None,
            templates_config=templates_config,
        )

        self._stage_order = []
        for stage in self._template.stages:
            tools = resolve_tools(stage.allowed_tools, worker_tools_dict)
            self._create_stage_agent(stage, tools, model)

    def _init_llm_mode(self, config, worker_tools_dict, model, templates_config):
        """Use the model to generate the pipeline decomposition."""
        # Build planner prompt.
        tool_names = sorted(worker_tools_dict.keys())
        tool_descriptions = []
        for name, tool in worker_tools_dict.items():
            desc = getattr(tool, "description", "")
            tool_descriptions.append(f"- {name}: {desc}")
        tools_info = "\n".join(tool_descriptions) if tool_descriptions else "(none)"

        task = config.get("_task", self._task or "")

        system_prompt = (
            "You are a pipeline decomposition planner. Given a task, break "
            "it down into sequential stages. For each stage, specify:\n"
            "1. stage_name: a short identifier (snake_case)\n"
            "2. role: a detailed description of what this stage does\n"
            "3. allowed_tools: which tools this stage needs (from the "
            "available set, or [] for pure reasoning stages)\n"
            "4. interface_output: what this stage passes to the next\n\n"
            "Output your decomposition as a JSON array:\n"
            '[{"stage_name": "...", "role": "...", "allowed_tools": '
            '["..."], "interface_output": "..."}]\n\n'
            f"Available tools:\n{tools_info}\n\n"
            "Guidelines:\n"
            "- Not every stage needs tools. Reasoning-only stages should "
            "have allowed_tools: []\n"
            "- The final stage should include instructions to output "
            "TASK_COMPLETE when done\n"
            "- Be specific in role descriptions\n"
            "- Typical pipelines have 3-6 stages"
        )

        user_prompt = f"Task to decompose: {task}\n\nAvailable tools: {tool_names}"

        # Single-shot model call — decomposition is a planning step,
        # not a multi-step agentic loop.
        from smolagents.models import ChatMessage as _ChatMessage

        messages = [
            _ChatMessage(role="system", content=system_prompt),
            _ChatMessage(role="user", content=user_prompt),
        ]
        response = model.generate(messages)
        raw_output = response.content if hasattr(response, "content") else str(response)

        # Parse the JSON output.
        stages_data = _parse_planner_output(raw_output)

        # Create agents from parsed stages.
        self._template = PipelineTemplate(
            name="llm_generated",
            stages=[
                PipelineStage(
                    name=s["stage_name"],
                    role=s["role"],
                    allowed_tools=s.get("allowed_tools", []),
                    interface_output=s.get("interface_output", ""),
                )
                for s in stages_data
            ],
        )
        self._pipeline_template_name = "llm_generated"

        self._stage_order = []
        for stage in self._template.stages:
            tools = resolve_tools(stage.allowed_tools, worker_tools_dict)
            self._create_stage_agent(stage, tools, model)

    # -- Agent creation --------------------------------------------------------

    def _create_stage_agent(self, stage: PipelineStage, tools: list, model) -> None:
        """Create a ToolCallingAgent for a pipeline stage."""
        # Build system prompt: base instructions + stage role.
        prompt_parts = []
        if self._base_instructions:
            prompt_parts.append(self._base_instructions.strip())
        prompt_parts.append(f"Your role: {stage.role}")
        if stage.interface_output:
            prompt_parts.append(
                f"Your output should be: {stage.interface_output}"
            )
        system_prompt = "\n\n".join(prompt_parts)

        from smolagents import ToolCallingAgent

        agent = ToolCallingAgent(
            tools=tools,
            model=model,
            name=stage.name,
            description=stage.role,
            instructions=system_prompt,
            max_steps=self._stage_max_steps,
            add_base_tools=False,
        )

        self._agents[stage.name] = agent
        self._stage_order.append(stage.name)

    # -- Context building ------------------------------------------------------

    # -- Shared state extraction ------------------------------------------------

    # UUID pattern — fallback for SESSION_ID when structured parsing fails.
    _UUID_RE = re.compile(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        re.IGNORECASE,
    )

    def _extract_shared_state(self, content: str) -> None:
        """Extract declared shared state keys from a stage's output.

        Primary mechanism: look for ``KEY: value`` lines in the output
        matching keys declared in the template's ``shared_state_keys``.
        Fallback for SESSION_ID: extract the first UUID if the structured
        ``SESSION_ID: <value>`` line was not found.

        Values are captured once — the first stage to produce a key is
        authoritative and later stages cannot overwrite it.
        """
        if not self._template:
            return

        for key in self._template.shared_state_keys:
            if key in self._shared_state:
                continue  # Already captured from an earlier stage.

            # Primary: structured ``KEY: value`` line.
            pattern = re.compile(
                rf"^\s*{re.escape(key)}\s*:\s*(.+)", re.MULTILINE,
            )
            m = pattern.search(content)
            if m:
                self._shared_state[key] = m.group(1).strip()
                continue

            # Fallback: for SESSION_ID, try UUID regex on full content.
            if key.upper() == "SESSION_ID":
                m = self._UUID_RE.search(content)
                if m:
                    self._shared_state[key] = m.group(0)

    def _format_shared_state_header(self) -> str | None:
        """Format shared state as a header block, or None if empty."""
        if not self._shared_state:
            return None
        lines = [f"  {k}: {v}" for k, v in sorted(self._shared_state.items())]
        return "SHARED STATE (from earlier stages):\n" + "\n".join(lines)

    # -- Context building ------------------------------------------------------

    def _build_stage_context(
        self, stage: PipelineStage, history: list, current_state: dict,
    ) -> str:
        """Build input context for a stage agent."""
        # First stage gets the task.
        if not history:
            return current_state.get("task", self._task)

        # Subsequent stages get previous stage's output framed with interface.
        last_msg = history[-1]
        prev_output = last_msg.content if isinstance(last_msg, AgentMessage) else str(last_msg)

        parts: list[str] = []

        # Inject shared state header (e.g. SESSION_ID) if available.
        header = self._format_shared_state_header()
        if header:
            parts.append(header)

        if self._current_stage_index > 0:
            prev_stage = self._get_stage(
                self._stage_order[self._current_stage_index - 1]
            )
            parts.append(
                f"Previous stage ({prev_stage.name}) output — "
                f"expected format: {prev_stage.interface_output}:\n"
                f"{prev_output}"
            )
        else:
            parts.append(prev_output)

        return "\n\n".join(parts)

    def _get_stage(self, name: str) -> PipelineStage:
        """Get a PipelineStage by name from the template."""
        if self._template:
            for stage in self._template.stages:
                if stage.name == name:
                    return stage
        # Fallback for dynamically created stages.
        return PipelineStage(name=name, role="", allowed_tools=[], interface_output="")

    # -- Properties ------------------------------------------------------------

    @property
    def template(self) -> PipelineTemplate | None:
        return self._template

    @property
    def stage_order(self) -> list[str]:
        return list(self._stage_order)

    @property
    def current_stage_index(self) -> int:
        return self._current_stage_index

    @property
    def interface_results(self) -> list[dict]:
        return list(self._interface_results)


# -- Helpers -------------------------------------------------------------------

def _validate_interface(output: str, expected_description: str) -> bool:
    """Lightweight interface validation: non-empty + keyword overlap."""
    if not output or not output.strip():
        return False
    # Simple keyword check: at least one word from the expected description
    # appears in the output.
    if not expected_description:
        return True
    keywords = set(re.findall(r'\w+', expected_description.lower()))
    output_words = set(re.findall(r'\w+', output.lower()))
    overlap = keywords & output_words
    # Pass if at least one meaningful keyword overlaps.
    return len(overlap) > 0


def _parse_planner_output(raw_output: str) -> list[dict]:
    """Parse the LLM planner's output into a list of stage dicts.

    Raises:
        ValueError: If the output cannot be parsed as JSON or doesn't
            contain valid stage definitions. Does NOT fall back silently.
    """
    # Try to find a JSON array in the output.
    # The LLM might output it wrapped in text or markdown.
    json_match = re.search(r'\[[\s\S]*\]', raw_output)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if isinstance(data, list) and len(data) > 0:
                # Validate required fields.
                for i, stage in enumerate(data):
                    if not isinstance(stage, dict):
                        raise ValueError(
                            f"Stage {i} is not a dict: {stage!r}"
                        )
                    if "stage_name" not in stage:
                        raise ValueError(
                            f"Stage {i} missing 'stage_name': {stage!r}"
                        )
                    if "role" not in stage:
                        raise ValueError(
                            f"Stage {i} missing 'role': {stage!r}"
                        )
                return data
        except json.JSONDecodeError as e:
            raise ValueError(
                f"LLM planner output contains invalid JSON: {e}\n"
                f"Raw output:\n{raw_output}"
            ) from e

    raise ValueError(
        f"LLM planner output could not be parsed as a JSON stage array.\n"
        f"Raw output:\n{raw_output}"
    )
