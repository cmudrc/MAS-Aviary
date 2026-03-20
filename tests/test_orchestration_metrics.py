"""Tests for orchestration-specific metrics computation."""


from src.coordination.history import AgentMessage, ToolCallRecord
from src.logging.orchestration_metrics import (
    compute_cross_prompt_metrics,
    compute_orchestration_metrics,
)

# ---- Per-prompt metrics ------------------------------------------------------

class TestOrchestrationMetrics:
    def test_empty_messages(self):
        result = compute_orchestration_metrics([])
        assert result["agents_spawned"] == 0
        assert result["orchestrator_turns"] == 0

    def test_counts_orchestrator_and_worker_turns(self):
        msgs = [
            AgentMessage(agent_name="orchestrator", content="plan", turn_number=1, timestamp=1.0),
            AgentMessage(agent_name="orchestrator", content="create", turn_number=2, timestamp=2.0),
            AgentMessage(agent_name="worker1", content="done", turn_number=3, timestamp=3.0),
        ]
        result = compute_orchestration_metrics(msgs)
        assert result["orchestrator_turns"] == 2
        assert result["worker_turns"] == 1

    def test_overhead_ratio(self):
        msgs = [
            AgentMessage(agent_name="orchestrator", content="x", turn_number=1, timestamp=1.0),
            AgentMessage(agent_name="w1", content="y", turn_number=2, timestamp=2.0),
            AgentMessage(agent_name="w2", content="z", turn_number=3, timestamp=3.0),
            AgentMessage(agent_name="w3", content="a", turn_number=4, timestamp=4.0),
        ]
        result = compute_orchestration_metrics(msgs)
        assert result["orchestrator_overhead_ratio"] == 0.25  # 1/4

    def test_agents_spawned(self):
        msgs = [
            AgentMessage(agent_name="orchestrator", content="x", turn_number=1, timestamp=1.0),
            AgentMessage(agent_name="w1", content="y", turn_number=2, timestamp=2.0),
            AgentMessage(agent_name="w2", content="z", turn_number=3, timestamp=3.0),
            AgentMessage(agent_name="w1", content="again", turn_number=4, timestamp=4.0),
        ]
        result = compute_orchestration_metrics(msgs)
        assert result["agents_spawned"] == 2  # w1, w2

    def test_token_growth(self):
        msgs = [
            AgentMessage(agent_name="orchestrator", content="x", turn_number=1,
                         timestamp=1.0, token_count=100),
            AgentMessage(agent_name="orchestrator", content="y", turn_number=2,
                         timestamp=2.0, token_count=200),
            AgentMessage(agent_name="w1", content="z", turn_number=3,
                         timestamp=3.0, token_count=50),
        ]
        result = compute_orchestration_metrics(msgs)
        assert result["orchestrator_token_growth"] == [100, 200]

    def test_tool_distribution(self):
        msgs = [
            AgentMessage(
                agent_name="w1", content="x", turn_number=1, timestamp=1.0,
                tool_calls=[
                    ToolCallRecord("calc", {}, "4", 0.1),
                    ToolCallRecord("echo", {}, "hi", 0.1),
                ],
            ),
            AgentMessage(
                agent_name="w2", content="y", turn_number=2, timestamp=2.0,
                tool_calls=[ToolCallRecord("calc", {}, "5", 0.1)],
            ),
        ]
        result = compute_orchestration_metrics(msgs)
        assert result["tool_distribution"]["w1"] == ["calc", "echo"]
        assert result["tool_distribution"]["w2"] == ["calc"]

    def test_information_ratio(self):
        msgs = [
            AgentMessage(agent_name="orchestrator", content="x", turn_number=1,
                         timestamp=1.0, token_count=300),
            AgentMessage(agent_name="w1", content="y", turn_number=2,
                         timestamp=2.0, token_count=100),
        ]
        result = compute_orchestration_metrics(msgs)
        assert result["information_ratio"] == 3.0  # 300/100

    def test_information_ratio_no_worker_tokens(self):
        msgs = [
            AgentMessage(agent_name="orchestrator", content="x", turn_number=1,
                         timestamp=1.0, token_count=100),
            AgentMessage(agent_name="w1", content="y", turn_number=2,
                         timestamp=2.0),  # no token_count
        ]
        result = compute_orchestration_metrics(msgs)
        assert result["information_ratio"] == 0.0

    def test_explicit_worker_names(self):
        msgs = [
            AgentMessage(agent_name="orchestrator", content="x", turn_number=1, timestamp=1.0),
            AgentMessage(agent_name="w1", content="y", turn_number=2, timestamp=2.0),
            AgentMessage(agent_name="system_agent", content="z", turn_number=3, timestamp=3.0),
        ]
        result = compute_orchestration_metrics(msgs, worker_names=["w1"])
        assert result["worker_turns"] == 1
        assert result["agents_spawned"] == 1


# ---- Cross-prompt metrics ----------------------------------------------------

class TestCrossPromptMetrics:
    def test_single_prompt(self):
        prompt1 = [
            AgentMessage(agent_name="w1", content="ok", turn_number=1, timestamp=1.0),
        ]
        result = compute_cross_prompt_metrics([prompt1])
        assert result["total_prompts"] == 1
        assert "w1" in result["per_agent"]
        assert result["per_agent"]["w1"]["prompts_participated"] == 1

    def test_multi_prompt_scores(self):
        prompt1 = [
            AgentMessage(agent_name="w1", content="ok", turn_number=1, timestamp=1.0,
                         tool_calls=[ToolCallRecord("calc", {}, "4", 0.1)]),
        ]
        prompt2 = [
            AgentMessage(agent_name="w1", content="ok", turn_number=1, timestamp=1.0,
                         tool_calls=[ToolCallRecord("calc", {}, "", 0.1, error="fail")]),
        ]
        result = compute_cross_prompt_metrics([prompt1, prompt2])
        agent = result["per_agent"]["w1"]
        assert agent["prompts_participated"] == 2
        # tool_error_rate: 1/2 = 0.5
        assert agent["tool_error_rate"] == 0.5
        # retry_rate: 0
        assert agent["retry_rate"] == 0.0
        # completion_rate: 1 error prompt / 2 total = 0.5
        assert agent["task_completion_rate"] == 0.5

    def test_authority_score_formula(self):
        prompt1 = [
            AgentMessage(agent_name="w1", content="ok", turn_number=1, timestamp=1.0),
        ]
        result = compute_cross_prompt_metrics([prompt1])
        agent = result["per_agent"]["w1"]
        # No errors: (1-0)*0.5 + (1-0)*0.3 + 1.0*0.2 = 1.0
        assert agent["authority_score"] == 1.0

    def test_empty_prompts(self):
        result = compute_cross_prompt_metrics([])
        assert result["total_prompts"] == 0
        assert result["per_agent"] == {}
