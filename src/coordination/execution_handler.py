"""Execution handler ABC and placeholder implementation.

Defines the interface between the orchestration layer (team creation)
and the execution layer (how assigned tasks are run). The placeholder
executor runs agents sequentially in creation order — it will be
replaced by Operational Methodology strategies later.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.coordination.history import AgentMessage
from src.logging.logger import InstrumentationLogger


@dataclass
class Assignment:
    """A single task assignment from the orchestrator to a worker agent."""

    agent_name: str
    task: str
    assigned_at_turn: int = 0


class ExecutionHandler(ABC):
    """Abstract interface for executing orchestrator-assigned tasks.

    Implementations define *how* agents run (sequential, iterative,
    graph-routed, etc.). The OrchestratedStrategy calls
    ``self.execution_handler.execute(...)`` without knowing which
    implementation it is.
    """

    @abstractmethod
    def execute(
        self,
        assignments: list[Assignment],
        agents: dict,
        logger: InstrumentationLogger | None,
        turn_offset: int = 0,
        action_metadata: dict | None = None,
    ) -> list[AgentMessage]:
        """Execute the assigned tasks and return resulting messages.

        Args:
            assignments: Ordered list of task assignments.
            agents: Dict mapping agent names to ToolCallingAgent instances.
            logger: Optional logger for per-turn logging.
            turn_offset: Starting turn number (to continue numbering
                from the orchestrator's last turn).
            action_metadata: Strategy-level metadata from the
                CoordinationAction (phase, rotation_index, turn, etc.).
                Handlers should merge this into each AgentMessage.metadata,
                with handler-specific keys taking precedence.

        Returns:
            List of AgentMessages produced during execution.
        """


class PlaceholderExecutor(ExecutionHandler):
    """Runs each assigned agent once in creation order.

    Each agent's output is passed as additional context to the next
    agent. Does NOT evaluate output quality, retry on failure, or
    make routing decisions.

    Design constraints (from PRD):
    - Runs each assigned agent exactly once in creation order
    - Passes output as context to next agent
    - Does NOT evaluate output quality
    - Does NOT retry on failure
    - Terminates when all assignments executed or TASK_COMPLETE found
      or max turns reached
    """

    def __init__(self, termination_keyword: str = "TASK_COMPLETE",
                 max_turns: int = 30):
        self._termination_keyword = termination_keyword
        self._max_turns = max_turns

    def execute(
        self,
        assignments: list[Assignment],
        agents: dict,
        logger: InstrumentationLogger | None,
        turn_offset: int = 0,
        action_metadata: dict | None = None,
    ) -> list[AgentMessage]:
        messages: list[AgentMessage] = []
        turn = turn_offset
        previous_output = ""
        _base_meta = dict(action_metadata or {})

        for assignment in assignments:
            if turn >= self._max_turns:
                break

            agent = agents.get(assignment.agent_name)
            if agent is None:
                turn += 1
                msg = AgentMessage(
                    agent_name=assignment.agent_name,
                    content="",
                    turn_number=turn,
                    timestamp=time.time(),
                    error=f"Agent '{assignment.agent_name}' not found",
                    metadata=dict(_base_meta),
                )
                messages.append(msg)
                if logger is not None:
                    logger.log_turn(msg)
                continue

            # Build input: task + context from previous agent.
            if previous_output:
                input_context = (
                    f"{assignment.task}\n\n"
                    f"Context from previous agent:\n{previous_output}"
                )
            else:
                input_context = assignment.task

            turn += 1
            start = time.monotonic()
            try:
                result = agent.run(input_context)
                content = str(result) if result is not None else ""
            except Exception as e:
                content = ""
                msg = AgentMessage(
                    agent_name=assignment.agent_name,
                    content=content,
                    turn_number=turn,
                    timestamp=time.time(),
                    duration_seconds=time.monotonic() - start,
                    error=str(e),
                    metadata=dict(_base_meta),
                )
                messages.append(msg)
                if logger is not None:
                    logger.log_turn(msg)
                previous_output = content
                continue

            duration = time.monotonic() - start
            msg = AgentMessage(
                agent_name=assignment.agent_name,
                content=content,
                turn_number=turn,
                timestamp=time.time(),
                duration_seconds=duration,
                metadata=dict(_base_meta),
            )
            messages.append(msg)
            if logger is not None:
                logger.log_turn(msg)

            previous_output = content

            # Check termination keyword.
            if self._termination_keyword and self._termination_keyword in content:
                break

        return messages
