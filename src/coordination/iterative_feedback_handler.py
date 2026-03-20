"""Iterative Feedback execution handler.

Implements the ExecutionHandler interface with a feedback-driven retry
loop.  When an agent finishes an attempt, the handler evaluates the tool
call outcome (success/failure, return code, stderr) and decides: retry
this agent or move on to the next agent.

Key behaviours (from PRD):
- Aspiration levels determine "good enough to move on"
- Feedback from tool call outcomes is injected into retry context
- Penultimate attempt (max_retries-1) warns the agent
- Final attempt forces a summary hand-off
- Three human feedback modes: none, real_time, between_prompt
- Toolless agents can optionally skip retries
- Cross-stage error forwarding: when a stage fails, an UPSTREAM_ERROR
  note is attached to context for all subsequent stages, allowing
  responsible upstream agents to self-correct on the next loop pass.
"""

import re
import time
from typing import Any, Callable

from src.coordination.execution_handler import Assignment, ExecutionHandler
from src.coordination.feedback_extraction import (
    AttemptFeedback,
    extract_feedback,
    format_feedback_for_retry,
)
from src.coordination.history import AgentMessage, ToolCallRecord
from src.logging.logger import InstrumentationLogger

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


def _extract_token_count(agent: Any, content: str) -> int | None:
    """Return token count, or char-based estimate, or None."""
    for attr in ("token_count", "total_tokens"):
        val = getattr(agent, attr, None)
        if isinstance(val, int) and val > 0:
            return val
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
            return total
    if content:
        return max(1, len(content) // 4)
    return None


# Absolute hard cap for max_retries (PRD §4).
_MAX_RETRIES_CAP = 20


def _set_total_attempts(
    messages: list,
    start_idx: int,
    attempt_feedbacks: list,
) -> None:
    """Backfill total_attempts and final_aspiration_met on this assignment's messages."""
    total = len(attempt_feedbacks)
    # final_aspiration_met: True if last feedback had no tool errors.
    final_met = bool(attempt_feedbacks and not attempt_feedbacks[-1].has_tool_errors)
    for m in messages[start_idx:]:
        if "total_attempts" not in m.metadata:
            m.metadata["total_attempts"] = total
        if "final_aspiration_met" not in m.metadata:
            m.metadata["final_aspiration_met"] = final_met


class IterativeFeedbackHandler(ExecutionHandler):
    """Feedback-driven retry loop for agent execution.

    Construction accepts the full iterative_feedback config dict.
    """

    def __init__(
        self,
        config: dict | None = None,
        human_feedback_callback: Callable[[AttemptFeedback], str | None] | None = None,
    ):
        cfg = config or {}
        self._max_retries: int = min(cfg.get("max_retries", 20), _MAX_RETRIES_CAP)
        self._feedback_window: int = cfg.get("feedback_window", 5)
        self._retry_toolless: bool = cfg.get("retry_toolless_agents", False)
        self._aspiration_mode: str = cfg.get("aspiration_mode", "tool_success")
        self._aspiration_threshold = cfg.get("aspiration_threshold")
        self._human_mode: str = cfg.get("human_feedback_mode", "none")
        self._human_guidance: str | None = cfg.get("human_guidance")
        self._human_skip_keyword: str = cfg.get("human_skip_keyword", "SKIP")
        self._termination_keyword: str = cfg.get("termination_keyword", "TASK_COMPLETE")
        self._stuck_threshold: int = cfg.get("stuck_threshold", 3)
        self._human_callback = human_feedback_callback

        # Per-assignment attempt histories (populated during execute).
        self.attempt_histories: list[list[AttemptFeedback]] = []

        # Track whether a prior execute() call already produced a successful
        # result for a given agent.  Prevents redundant agent invocations
        # when the coordinator loop keeps re-invoking the handler after the
        # task is done, but still allows different agents to run.
        self._last_successful_output: str | None = None
        self._last_successful_agent: str | None = None

        # Cross-stage error forwarding.  When a stage's output contains
        # simulation failure indicators, the error is recorded and
        # prepended to all subsequent stages so upstream agents can
        # self-correct when the pipeline loops back to them.
        self._upstream_errors: list[str] = []

    # ------------------------------------------------------------------
    # Cross-stage error forwarding
    # ------------------------------------------------------------------

    _FAILURE_RE = re.compile(
        r"(?:NaN|inf|not converge|did not converge|AVIARY_SETUP_ERROR|"
        r"simulation failed|failed to converge|residuals contain)",
        re.IGNORECASE,
    )

    def _detect_upstream_error(
        self,
        agent_name: str,
        content: str,
        tool_calls: list[ToolCallRecord],
    ) -> None:
        """If agent output signals a simulation or setup failure, record it.

        Scans both agent content and tool call outputs for failure
        keywords.  The recorded error is prepended to all subsequent
        stages' context.
        """
        texts = [content]
        for tc in tool_calls:
            if tc.error:
                texts.append(tc.error)
            if tc.output:
                texts.append(tc.output[:500])

        for text in texts:
            m = self._FAILURE_RE.search(text)
            if m:
                # Extract a short reason from the matched region.
                start = max(0, m.start() - 40)
                end = min(len(text), m.end() + 80)
                snippet = text[start:end].replace("\n", " ").strip()
                note = (
                    f"UPSTREAM_ERROR: {agent_name} failed — {snippet}\n"
                    "Downstream stages: if you are responsible for the "
                    "parameter that caused this failure, treat this as a "
                    "retry of your stage and adjust your parameters."
                )
                self._upstream_errors.append(note)
                return

    def _format_upstream_errors(self) -> str:
        """Format accumulated upstream errors for context injection."""
        if not self._upstream_errors:
            return ""
        return "\n\n".join(self._upstream_errors)

    # ------------------------------------------------------------------
    # Stuck detection
    # ------------------------------------------------------------------

    def _is_stuck(self, feedbacks: list[AttemptFeedback]) -> bool:
        """Return True if the last N feedbacks all have the same error type.

        Compares the sorted set of error_type strings from each feedback's
        tool_calls.  If the last ``_stuck_threshold`` feedbacks share the
        exact same error signature, the agent is stuck and further retries
        are unlikely to help.
        """
        n = self._stuck_threshold
        if len(feedbacks) < n:
            return False

        def _error_sig(fb: AttemptFeedback) -> tuple:
            """Hashable error signature from a feedback's tool outcomes.

            Compares only error types (not full messages) so that
            retries with the same failure category but different
            parameter values are correctly identified as stuck.
            """
            types: set[str] = set()
            for tc in fb.tool_calls:
                if not tc.success:
                    types.add(tc.error_type or "unknown")
            return tuple(sorted(types))

        recent = feedbacks[-n:]
        sig = _error_sig(recent[0])
        return all(_error_sig(fb) == sig for fb in recent[1:])

    # ------------------------------------------------------------------
    # ExecutionHandler interface
    # ------------------------------------------------------------------

    def execute(
        self,
        assignments: list[Assignment],
        agents: dict,
        logger: InstrumentationLogger | None,
        turn_offset: int = 0,
        action_metadata: dict | None = None,
    ) -> list[AgentMessage]:
        _base_meta = dict(action_metadata or {})

        # Defence-in-depth: if a previous execute() call already completed
        # successfully for the same agent, return the cached result instead
        # of re-invoking.  This guards against the coordinator loop
        # accidentally re-calling the handler after task completion.
        if (
            self._last_successful_output is not None
            and assignments
            and assignments[0].agent_name == self._last_successful_agent
        ):
            turn_offset += 1
            msg = AgentMessage(
                agent_name=assignments[0].agent_name,
                content=self._last_successful_output,
                turn_number=turn_offset,
                timestamp=time.time(),
                metadata={
                    **_base_meta,
                    "attempt_number": 0,
                    "aspiration_mode": self._aspiration_mode,
                    "aspiration_met": True,
                    "total_attempts": 1,
                    "final_aspiration_met": True,
                },
            )
            if logger is not None:
                logger.log_turn(msg)
            return [msg]

        messages: list[AgentMessage] = []
        turn = turn_offset
        previous_output = ""

        for assignment in assignments:
            agent = agents.get(assignment.agent_name)
            assignment_msg_start = len(messages)

            # Missing agent → error message, continue.
            if agent is None:
                turn += 1
                msg = AgentMessage(
                    agent_name=assignment.agent_name,
                    content="",
                    turn_number=turn,
                    timestamp=time.time(),
                    error=f"Agent '{assignment.agent_name}' not found",
                    metadata={
                        **_base_meta,
                        "attempt_number": 0,
                        "aspiration_mode": self._aspiration_mode,
                        "aspiration_met": False,
                        "total_attempts": 0,
                        "final_aspiration_met": False,
                    },
                )
                messages.append(msg)
                if logger is not None:
                    logger.log_turn(msg)
                self.attempt_histories.append([])
                continue

            # Determine if agent has tools.
            agent_has_tools = bool(getattr(agent, "tools", None))

            # If toolless and retry_toolless is false, run once.
            if not agent_has_tools and not self._retry_toolless:
                effective_max = 1
            else:
                effective_max = self._max_retries

            attempt_feedbacks: list[AttemptFeedback] = []
            human_feedback_text: str | None = None

            for attempt in range(effective_max):
                if turn >= turn_offset + 1000:  # safety valve
                    break

                # Build input context.
                context = self._build_context(
                    assignment.task,
                    previous_output,
                    attempt,
                    effective_max,
                    attempt_feedbacks,
                    human_feedback_text,
                )

                turn += 1
                start = time.monotonic()
                is_retry = attempt > 0

                try:
                    result = agent.run(context)
                    content = str(result) if result is not None else ""
                except Exception as e:
                    content = ""
                    duration = time.monotonic() - start
                    tool_calls = _extract_tool_calls(agent)
                    msg = AgentMessage(
                        agent_name=assignment.agent_name,
                        content=content,
                        turn_number=turn,
                        timestamp=time.time(),
                        duration_seconds=duration,
                        tool_calls=tool_calls,
                        is_retry=is_retry,
                        retry_of_turn=messages[-1].turn_number if is_retry and messages else None,
                        error=str(e),
                        metadata={
                            **_base_meta,
                            "attempt_number": attempt,
                            "aspiration_mode": self._aspiration_mode,
                        },
                    )
                    messages.append(msg)
                    if logger is not None:
                        logger.log_turn(msg)

                    fb = extract_feedback(msg, attempt_number=attempt)
                    msg.metadata["aspiration_met"] = False
                    attempt_feedbacks.append(fb)

                    # Exception always counts as failure — continue retrying.
                    if self._is_stuck(attempt_feedbacks):
                        msg.metadata["stuck_detected"] = True
                        break
                    human_feedback_text = self._get_human_feedback(fb)
                    if human_feedback_text == self._human_skip_keyword:
                        break
                    continue

                duration = time.monotonic() - start
                tool_calls = _extract_tool_calls(agent)
                token_count = _extract_token_count(agent, content)
                msg = AgentMessage(
                    agent_name=assignment.agent_name,
                    content=content,
                    turn_number=turn,
                    timestamp=time.time(),
                    duration_seconds=duration,
                    tool_calls=tool_calls,
                    token_count=token_count,
                    is_retry=is_retry,
                    retry_of_turn=messages[-1].turn_number if is_retry and messages else None,
                    metadata={
                        **_base_meta,
                        "attempt_number": attempt,
                        "aspiration_mode": self._aspiration_mode,
                    },
                )
                messages.append(msg)
                if logger is not None:
                    logger.log_turn(msg)

                fb = extract_feedback(msg, attempt_number=attempt)
                met = self._meets_aspiration(fb)
                msg.metadata["aspiration_met"] = met
                attempt_feedbacks.append(fb)

                # Check termination keyword.
                if self._termination_keyword and self._termination_keyword in content:
                    self.attempt_histories.append(attempt_feedbacks)
                    self._last_successful_output = content
                    _set_total_attempts(messages, assignment_msg_start, attempt_feedbacks)
                    return messages

                # Check aspiration.
                if self._meets_aspiration(fb):
                    break

                # Stuck detection: if the last N attempts all produced the
                # same error signature, stop retrying — the agent cannot
                # recover and further attempts just waste time.
                if self._is_stuck(attempt_feedbacks):
                    msg.metadata["stuck_detected"] = True
                    break

                # Final attempt — force summary if we haven't broken out.
                if attempt == effective_max - 1:
                    # Already on the last attempt; the summary instruction
                    # was injected via _build_context.  Accept whatever
                    # the agent produced.
                    break

                # Human feedback (real_time mode).
                human_feedback_text = self._get_human_feedback(fb)
                if human_feedback_text == self._human_skip_keyword:
                    break

            self.attempt_histories.append(attempt_feedbacks)
            _set_total_attempts(messages, assignment_msg_start, attempt_feedbacks)

            # Detect simulation/setup failures for cross-stage error forwarding.
            if messages:
                last_msg = messages[-1]
                self._detect_upstream_error(
                    last_msg.agent_name,
                    last_msg.content,
                    last_msg.tool_calls,
                )

            # Update previous_output for next assignment.
            if messages:
                previous_output = messages[-1].content

        # Cache the last successful output so subsequent calls short-circuit.
        if messages and not any(m.error for m in messages):
            self._last_successful_output = messages[-1].content
            self._last_successful_agent = messages[-1].agent_name

        return messages

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_context(
        self,
        task: str,
        previous_output: str,
        attempt: int,
        max_retries: int,
        feedback_history: list[AttemptFeedback],
        human_feedback: str | None,
    ) -> str:
        parts: list[str] = []

        # Human guidance (between_prompt mode — static for entire run).
        if self._human_guidance and self._human_mode == "between_prompt":
            parts.append(f'Human guidance for this run:\n"{self._human_guidance}"')

        # Cross-stage upstream errors (persists across all stages).
        upstream = self._format_upstream_errors()
        if upstream:
            parts.append(upstream)

        # Task + previous agent context.
        if previous_output:
            parts.append(f"{task}\n\nContext from previous agent:\n{previous_output}")
        else:
            parts.append(task)

        # Feedback from previous attempts (windowed).
        if feedback_history:
            window = feedback_history[-self._feedback_window :]
            for fb in window:
                parts.append(format_feedback_for_retry(fb, max_retries))

        # Human feedback from real_time mode.
        if human_feedback and human_feedback != self._human_skip_keyword:
            parts.append(f'Human feedback after attempt {attempt}:\n"{human_feedback}"')

        # Penultimate attempt warning.
        if attempt == max_retries - 2 and max_retries > 2:
            parts.append(
                "WARNING: This is your second-to-last attempt. If your next "
                "attempt also fails, you must summarize everything you have "
                "tried and what errors occurred so the next agent can continue "
                "from where you left off."
            )

        # Final attempt — force summary.
        if attempt == max_retries - 1 and max_retries > 1:
            n_tried = len(feedback_history)
            parts.append(
                f"This is your FINAL attempt. You have tried {n_tried} times "
                "and not succeeded. Instead of trying again, produce a summary of:\n"
                "1. What approaches you tried\n"
                "2. What errors occurred for each\n"
                "3. What you think the issue is\n"
                "4. What the next agent should try differently\n\n"
                "This summary will be passed to the next agent."
            )

        return "\n\n".join(parts)

    def _meets_aspiration(self, feedback: AttemptFeedback) -> bool:
        """Check if the feedback meets the configured aspiration level."""
        mode = self._aspiration_mode

        if mode == "tool_success":
            return not feedback.has_tool_errors

        if mode == "any_output":
            return len(feedback.output_content.strip()) > 0

        if mode == "no_tool_errors_or_max":
            # Always returns True — the loop naturally stops at max retries,
            # but this mode also passes if tools succeeded.
            return not feedback.has_tool_errors

        if mode == "custom":
            if callable(self._aspiration_threshold):
                return self._aspiration_threshold(feedback)
            return not feedback.has_tool_errors

        # Unknown mode → default to tool_success.
        return not feedback.has_tool_errors

    def _get_human_feedback(self, feedback: AttemptFeedback) -> str | None:
        """Get human feedback if in real_time mode and attempt failed."""
        if self._human_mode != "real_time":
            return None
        if not feedback.has_tool_errors:
            return None

        if self._human_callback is not None:
            return self._human_callback(feedback)

        # Default fallback: stdin input().
        return input(f"Attempt {feedback.attempt_number + 1} failed. Feedback (or SKIP): ")
