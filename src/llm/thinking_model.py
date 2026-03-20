"""Model wrapper with think-block stripping, robust JSON parsing, and reliability patterns.

Works with any model that emits chain-of-thought inside ``<think>...</think>``
tags (Qwen3, DeepSeek-R1, etc.).  Three layers of reliability:

1. **Think-block stripping** — ``<think>...</think>`` content is removed from
   the raw model output *before* tool-call parsing runs, so braces inside
   reasoning traces no longer confuse the JSON extractor.

2. **Robust JSON extraction** — instead of "first ``{`` to last ``}``", we
   find the JSON object that actually contains the expected tool-call keys
   (``name`` and ``arguments`` by default).  This handles models that emit
   analysis text with incidental braces before or around the real tool call.

3. **Prompt+validate retry** — on parse failure inside ``generate()``, the
   error is fed back to the model and generation is retried.  Retries are
   transparent to smolagents and do NOT consume ``max_steps`` budget.
   (Ported from CMU design-research-agents ``structured_output.py``.)
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

from smolagents import Tool, TransformersModel
from smolagents.models import (
    ChatMessage,
    ChatMessageToolCall,
    ChatMessageToolCallFunction,
    MessageRole,
    parse_json_if_needed,
)

from src.llm.reliability import ReliabilityConfig, add_strict_properties

logger = logging.getLogger(__name__)

# Pre-compiled pattern for <think>...</think> blocks (supports nested tags).
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _find_tool_call_json(
    text: str,
    name_key: str = "name",
    arguments_key: str = "arguments",
) -> dict:
    """Extract the JSON object containing *name_key* and *arguments_key*.

    Strategy:
      1. Try every ``{`` in the text as a candidate start.
      2. For each candidate, use a brace-depth counter to find the matching
         closing ``}``.
      3. Attempt ``json.loads`` on the substring.
      4. If the resulting dict contains both expected keys → return it.

    Fallback (CMU DRC pattern): if brace-depth scanning fails (e.g.
    truncated or malformed JSON), use ``json.JSONDecoder().raw_decode()``
    which can recover partial JSON objects from arbitrary text.

    Raises ``ValueError`` when no valid JSON object is found at all.
    """
    candidates: list[dict] = []

    i = 0
    while i < len(text):
        if text[i] != "{":
            i += 1
            continue

        # Walk forward counting brace depth.
        depth = 0
        for j in range(i, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            if depth == 0:
                candidate_str = text[i : j + 1]
                try:
                    obj = json.loads(candidate_str, strict=False)
                    if isinstance(obj, dict):
                        if name_key in obj and arguments_key in obj:
                            return obj
                        candidates.append(obj)
                except (json.JSONDecodeError, ValueError):
                    pass
                break
        i += 1

    # No object with the expected keys — return best candidate if any.
    if candidates:
        return candidates[0]

    # Fallback: raw_decode scanning (CMU DRC pattern).
    # JSONDecoder.raw_decode() can recover valid JSON objects from
    # positions where brace-depth counting failed (e.g. escaped braces,
    # strings containing braces, or partially truncated output).
    decoder = json.JSONDecoder(strict=False)
    raw_candidates: list[dict] = []
    pos = 0
    while pos < len(text):
        idx = text.find("{", pos)
        if idx == -1:
            break
        try:
            obj, end_idx = decoder.raw_decode(text, idx)
            if isinstance(obj, dict):
                if name_key in obj and arguments_key in obj:
                    return obj
                raw_candidates.append(obj)
            pos = end_idx
        except (json.JSONDecodeError, ValueError):
            pos = idx + 1

    if raw_candidates:
        return raw_candidates[0]

    raise ValueError("The model output does not contain any JSON blob.")


class ThinkingModel(TransformersModel):
    """TransformersModel subclass with think-block stripping, robust parsing,
    and CMU-style reliability patterns (retry + strict schemas).

    Drop-in replacement — accepts the same constructor arguments as
    ``TransformersModel`` plus an optional ``reliability`` config.
    """

    def __init__(self, *args, reliability: ReliabilityConfig | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._reliability = reliability or ReliabilityConfig()
        # Per-call thinking toggle.  When False, ``enable_thinking=False``
        # is injected into apply_chat_template_kwargs for the next
        # generate() call chain.  Strategies set this to False for agents
        # that need reliable JSON (e.g. orchestrator) and True for agents
        # that benefit from reasoning (e.g. simulation_executor).
        self._thinking_enabled: bool = True

    @property
    def thinking_enabled(self) -> bool:
        return self._thinking_enabled

    @thinking_enabled.setter
    def thinking_enabled(self, value: bool) -> None:
        self._thinking_enabled = value

    # Sentinel for "key was not present".
    _NO_KEY = object()

    def _set_thinking_kwarg(self, enabled: bool) -> Any:
        """Temporarily inject/remove ``enable_thinking`` in chat template kwargs.

        Returns the previous value (or ``_NO_KEY``) so it can be restored.
        """
        prev = self.apply_chat_template_kwargs.get("enable_thinking", self._NO_KEY)
        if enabled:
            self.apply_chat_template_kwargs.pop("enable_thinking", None)
        else:
            self.apply_chat_template_kwargs["enable_thinking"] = False
        return prev

    def _restore_thinking_kwarg(self, prev: Any) -> None:
        """Restore ``enable_thinking`` to its previous state."""
        if prev is self._NO_KEY:
            self.apply_chat_template_kwargs.pop("enable_thinking", None)
        else:
            self.apply_chat_template_kwargs["enable_thinking"] = prev

    # -- Truncation detection ---------------------------------------------------

    def _is_truncated(self, message: ChatMessage) -> bool:
        """Return True if the output was truncated at max_new_tokens.

        Truncated outputs have exactly max_new_tokens of generated text
        and typically contain an unclosed ``<think>`` block with no JSON.
        """
        content = message.content or ""
        # Heuristic: unclosed <think> block (opened but never closed).
        has_open_think = "<think>" in content and "</think>" not in content
        if has_open_think:
            return True
        # Heuristic: no braces at all in a supposedly-tool-calling response.
        stripped = strip_think_blocks(content)
        if not stripped or ("{" not in stripped and "}" not in stripped):
            # Only consider truncation if the raw content was substantial.
            if len(content) > 200:
                return True
        return False

    # -- Retry-aware generate --------------------------------------------------

    def generate(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs: Any,
    ) -> ChatMessage:
        """Generate with prompt+validate retry on parse failure.

        On each attempt: call ``super().generate()`` then
        ``self.parse_tool_calls()``.  If parsing fails, append the
        failed output and an error-feedback message to the conversation
        and retry.  Retries happen *inside* ``generate()`` so smolagents
        sees only one step regardless of retry count.

        Truncation detection (Fix 3): when the output is truncated
        (unclosed ``<think>`` block, no JSON), the retry disables
        thinking mode for that specific attempt instead of appending
        more context.
        """
        max_retries = self._reliability.max_retries
        last_error: Exception | None = None
        # Work on a copy so retries don't pollute the caller's list.
        msgs = list(messages)

        # Apply per-call thinking toggle (Fix 2).
        prev_thinking = self._set_thinking_kwarg(self._thinking_enabled)
        try:
            for attempt in range(max_retries + 1):
                raw_message = super().generate(
                    msgs,
                    stop_sequences=stop_sequences,
                    response_format=response_format,
                    tools_to_call_from=tools_to_call_from,
                    **kwargs,
                )
                try:
                    parsed = self.parse_tool_calls(raw_message)
                    if attempt > 0:
                        logger.info("Retry %d/%d succeeded", attempt, max_retries)
                    return parsed
                except (ValueError, AssertionError) as exc:
                    last_error = exc
                    if attempt == max_retries:
                        break

                    # Fix 3: truncation detection — if the output was
                    # truncated by max_new_tokens (unclosed <think> block,
                    # no JSON), disable thinking for the retry instead of
                    # appending more feedback context.
                    truncated = self._is_truncated(raw_message)
                    if truncated:
                        logger.warning(
                            "Output truncated (attempt %d/%d) — disabling "
                            "thinking for retry",
                            attempt + 1, max_retries + 1,
                        )
                        self._set_thinking_kwarg(False)
                        # Don't append error feedback — the model just
                        # needs more token budget for the actual JSON.
                        continue

                    logger.warning(
                        "Parse failed (attempt %d/%d): %s — retrying",
                        attempt + 1, max_retries + 1, exc,
                    )
                    # Append the failed assistant output and error feedback
                    # so the model can self-correct on the next attempt.
                    # Use content-block format ([{"type":"text","text":...}])
                    # to match smolagents' internal message representation —
                    # plain strings cause "string indices must be integers"
                    # when get_clean_message_list merges consecutive same-role
                    # messages.
                    error_feedback = (
                        "Your previous response could not be parsed as a "
                        f"valid tool call. Error: {exc}\n"
                        "Please respond with ONLY a valid JSON tool call "
                        "in the format: "
                        '{{"name": "<tool_name>", "arguments": {{...}}}}'
                    )
                    msgs = msgs + [
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": raw_message.content or ""}],
                        },
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": error_feedback}],
                        },
                    ]
        finally:
            # Always restore original thinking state.
            self._restore_thinking_kwarg(prev_thinking)

        # All retries exhausted.  Instead of raising (which smolagents
        # wraps as AgentGenerationError — a fatal error that kills run()),
        # return the raw message without tool_calls.  smolagents' own
        # _step_stream will then call parse_tool_calls() on our returned
        # message, get AgentParsingError (non-fatal), log it on the step,
        # and let the model self-correct on the next step.  This preserves
        # the natural error-recovery loop (CMU DRC "structured failure as
        # data" pattern) instead of escalating to an unrecoverable crash.
        logger.warning(
            "All %d retries exhausted: %s — returning raw message for "
            "smolagents error-recovery loop",
            max_retries + 1, last_error,
        )
        return raw_message  # type: ignore[possibly-undefined]

    # -- Strict tool schemas ---------------------------------------------------

    def _prepare_completion_kwargs(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Add ``additionalProperties: false`` to tool parameter schemas."""
        result = super()._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            **kwargs,
        )
        if self._reliability.strict_tool_schemas and "tools" in result:
            add_strict_properties(result["tools"])
        return result

    # -- Think-block-aware parse -----------------------------------------------

    def parse_tool_calls(self, message: ChatMessage) -> ChatMessage:
        """Strip ``<think>`` blocks, then parse tool calls with a robust JSON finder."""
        message.role = MessageRole.ASSISTANT

        if not message.tool_calls:
            assert message.content is not None, "Message contains no content and no tool calls"

            # 1. Strip <think>...</think> blocks.
            cleaned = strip_think_blocks(message.content)

            # 2. Extract the tool-call JSON using the robust parser.
            tool_dict = _find_tool_call_json(
                cleaned,
                name_key=self.tool_name_key,
                arguments_key=self.tool_arguments_key,
            )

            tool_name = tool_dict.get(self.tool_name_key)
            if tool_name is None:
                raise ValueError(
                    f"Tool call needs a '{self.tool_name_key}' key. "
                    f"Got keys: {list(tool_dict.keys())}"
                )

            tool_arguments = tool_dict.get(self.tool_arguments_key)
            if isinstance(tool_arguments, str):
                tool_arguments = parse_json_if_needed(tool_arguments)

            message.tool_calls = [
                ChatMessageToolCall(
                    id=str(uuid.uuid4()),
                    type="function",
                    function=ChatMessageToolCallFunction(
                        name=tool_name, arguments=tool_arguments
                    ),
                )
            ]

            # Update content to the cleaned version so downstream code
            # (prompt templates, logging) sees tidy output.
            message.content = cleaned

        assert len(message.tool_calls) > 0, "No tool call was found in the model output"

        for tool_call in message.tool_calls:
            tool_call.function.arguments = parse_json_if_needed(
                tool_call.function.arguments
            )
        return message


def strip_think_blocks(text: str) -> str:
    """Remove all ``<think>...</think>`` blocks from *text*."""
    return _THINK_RE.sub("", text).strip()
