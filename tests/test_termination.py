"""Tests for TerminationChecker."""

import time

import pytest

from src.coordination.history import AgentMessage, SharedHistory
from src.coordination.termination import TerminationChecker


def _msg(agent: str, content: str, turn: int, error: str | None = None) -> AgentMessage:
    return AgentMessage(
        agent_name=agent, content=content, turn_number=turn,
        timestamp=time.time(), error=error,
    )


@pytest.fixture
def default_checker():
    return TerminationChecker({
        "termination": {
            "keyword": "TASK_COMPLETE",
            "max_turns": 20,
            "max_consecutive_errors": 3,
        }
    })


# ---- Empty history (no termination) ------------------------------------------

def test_empty_history(default_checker):
    h = SharedHistory()
    assert default_checker.should_stop(h) is False
    assert default_checker.check_reason(h) is None


# ---- Keyword termination -----------------------------------------------------

def test_keyword_found(default_checker):
    h = SharedHistory()
    h.append(_msg("a", "The answer is 42. TASK_COMPLETE", 1))
    assert default_checker.should_stop(h) is True
    assert default_checker.check_reason(h) == "keyword:TASK_COMPLETE"


def test_keyword_not_in_earlier_messages(default_checker):
    h = SharedHistory()
    h.append(_msg("a", "TASK_COMPLETE", 1))
    h.append(_msg("b", "continuing work", 2))
    # Keyword only checked in the LAST message
    assert default_checker.should_stop(h) is False


def test_custom_keyword():
    checker = TerminationChecker({"termination": {"keyword": "DONE"}})
    h = SharedHistory()
    h.append(_msg("a", "DONE", 1))
    assert checker.should_stop(h) is True


# ---- Max turns ----------------------------------------------------------------

def test_max_turns_reached(default_checker):
    h = SharedHistory()
    for i in range(20):
        h.append(_msg("a", f"turn {i}", i))
    assert default_checker.should_stop(h) is True
    assert default_checker.check_reason(h) == "max_turns:20"


def test_max_turns_not_reached(default_checker):
    h = SharedHistory()
    for i in range(19):
        h.append(_msg("a", f"turn {i}", i))
    # only keyword or other conditions could trigger
    assert default_checker._max_turns_reached(h) is False


# ---- Consecutive errors -------------------------------------------------------

def test_consecutive_errors_trigger(default_checker):
    h = SharedHistory()
    h.append(_msg("a", "ok", 1))
    h.append(_msg("a", "err1", 2, error="fail"))
    h.append(_msg("a", "err2", 3, error="fail"))
    h.append(_msg("a", "err3", 4, error="fail"))
    assert default_checker.should_stop(h) is True
    assert "max_consecutive_errors" in default_checker.check_reason(h)


def test_consecutive_errors_broken_by_success(default_checker):
    h = SharedHistory()
    h.append(_msg("a", "err1", 1, error="fail"))
    h.append(_msg("a", "err2", 2, error="fail"))
    h.append(_msg("a", "ok", 3))  # breaks the streak
    assert default_checker._max_consecutive_errors(h) is False


# ---- Stuck detection ----------------------------------------------------------

def test_stuck_same_agent_same_output(default_checker):
    h = SharedHistory()
    h.append(_msg("a", "I am stuck", 1))
    h.append(_msg("a", "I am stuck", 2))
    assert default_checker.should_stop(h) is True
    assert default_checker.check_reason(h) == "stuck:identical_output"


def test_not_stuck_different_agents(default_checker):
    h = SharedHistory()
    h.append(_msg("a", "same text", 1))
    h.append(_msg("b", "same text", 2))
    assert default_checker._stuck_detected(h) is False


def test_not_stuck_different_content(default_checker):
    h = SharedHistory()
    h.append(_msg("a", "output1", 1))
    h.append(_msg("a", "output2", 2))
    assert default_checker._stuck_detected(h) is False


# ---- Default config -----------------------------------------------------------

def test_default_config_values():
    checker = TerminationChecker({})
    assert checker.keyword == "TASK_COMPLETE"
    assert checker.max_turns == 20
    assert checker.max_consecutive_errors == 3
