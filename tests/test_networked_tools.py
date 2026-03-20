"""Tests for networked peer tools — ReadBlackboard, WriteBlackboard, SpawnPeer.

No GPU needed. Uses DummyModel stub and mock blackboard.
"""

import json

import pytest
from smolagents import ToolCallingAgent
from smolagents.models import Model

from src.coordination.blackboard import Blackboard
from src.tools.mock_tools import CalculatorTool, EchoTool
from src.tools.networked_tools import (
    PEER_TOOL_NAMES,
    MarkTaskDone,
    NetworkedContext,
    ReadBlackboard,
    SpawnPeer,
    WriteBlackboard,
)

# ---- Fixtures ----------------------------------------------------------------

class DummyModel(Model):
    """Minimal model stub for agent construction."""

    def __init__(self):
        super().__init__(model_id="dummy")

    def generate(self, messages, stop_sequences=None, response_format=None,
                 tools_to_call_from=None, **kwargs):
        from smolagents.types import ChatMessage
        return ChatMessage(role="assistant", content="dummy response")


@pytest.fixture
def dummy_model():
    return DummyModel()


@pytest.fixture
def domain_tools():
    return [EchoTool(), CalculatorTool()]


@pytest.fixture
def context(dummy_model, domain_tools):
    """Fresh NetworkedContext with soft claiming."""
    bb = Blackboard(claiming_mode="soft")
    ctx = NetworkedContext(
        blackboard=bb,
        agents={"agent_1": "placeholder", "agent_2": "placeholder"},
        model=dummy_model,
        all_tools=domain_tools,
        peer_prompt="You are a peer agent.",
        agent_max_steps=5,
        max_agents=5,
        agent_counter=2,  # initial agents are agent_1, agent_2
        config={
            "peer_monitoring_visible": True,
            "trans_specialist_knowledge": True,
            "predictive_knowledge": False,
        },
    )
    return ctx


@pytest.fixture
def hard_context(dummy_model, domain_tools):
    """NetworkedContext with hard claiming."""
    bb = Blackboard(claiming_mode="hard")
    return NetworkedContext(
        blackboard=bb,
        agents={"agent_1": "placeholder"},
        model=dummy_model,
        all_tools=domain_tools,
        peer_prompt="You are a peer agent.",
        max_agents=5,
        agent_counter=1,
        config={
            "peer_monitoring_visible": True,
            "trans_specialist_knowledge": True,
            "predictive_knowledge": False,
        },
    )


@pytest.fixture
def none_context(dummy_model, domain_tools):
    """NetworkedContext with no claiming."""
    bb = Blackboard(claiming_mode="none")
    return NetworkedContext(
        blackboard=bb,
        agents={"agent_1": "placeholder"},
        model=dummy_model,
        all_tools=domain_tools,
        peer_prompt="You are a peer agent.",
        max_agents=5,
        agent_counter=1,
        config={
            "peer_monitoring_visible": True,
            "trans_specialist_knowledge": True,
            "predictive_knowledge": False,
        },
    )


# ---- ReadBlackboard tests ---------------------------------------------------

class TestReadBlackboard:
    def test_reads_all_entries(self, context):
        bb = context.blackboard
        bb.write("s1", "Working", "agent_1", "status")
        bb.write("r1", "Result data", "agent_2", "result")
        tool = ReadBlackboard(context)
        result = json.loads(tool.forward())
        assert result["total_entries"] == 2

    def test_filter_by_entry_type(self, context):
        bb = context.blackboard
        bb.write("s1", "Working", "agent_1", "status")
        bb.write("r1", "Data", "agent_2", "result")
        tool = ReadBlackboard(context)
        result = json.loads(tool.forward(entry_type="status"))
        assert result["total_entries"] == 1
        assert result["entries"][0]["entry_type"] == "status"

    def test_filter_all_returns_everything(self, context):
        bb = context.blackboard
        bb.write("s1", "Working", "agent_1", "status")
        bb.write("r1", "Data", "agent_2", "result")
        tool = ReadBlackboard(context)
        result = json.loads(tool.forward(entry_type="all"))
        assert result["total_entries"] == 2

    def test_returns_active_claims(self, context):
        bb = context.blackboard
        bb.write("geometry", "claimed", "agent_1", "claim")
        bb.write("coding", "claimed", "agent_2", "claim")
        tool = ReadBlackboard(context)
        result = json.loads(tool.forward())
        assert set(result["active_claims"]) == {"geometry", "coding"}

    def test_returns_identified_gaps(self, context):
        bb = context.blackboard
        bb.write("gap_1", "Nobody doing fillets", "agent_1", "gap")
        tool = ReadBlackboard(context)
        result = json.loads(tool.forward())
        assert result["identified_gaps"] == ["gap_1"]

    def test_predictions_filtered_when_disabled(self, context):
        context.config["predictive_knowledge"] = False
        bb = context.blackboard
        bb.write("pred_1", "agent_2 will do X", "agent_1", "prediction")
        bb.write("s1", "Working", "agent_1", "status")
        tool = ReadBlackboard(context)
        result = json.loads(tool.forward())
        assert result["total_entries"] == 1  # prediction excluded

    def test_predictions_included_when_enabled(self, context):
        context.config["predictive_knowledge"] = True
        bb = context.blackboard
        bb.write("pred_1", "agent_2 will do X", "agent_1", "prediction")
        bb.write("s1", "Working", "agent_1", "status")
        tool = ReadBlackboard(context)
        result = json.loads(tool.forward())
        assert result["total_entries"] == 2

    def test_trans_specialist_filters_results(self, context):
        context.config["trans_specialist_knowledge"] = False
        bb = context.blackboard
        long_val = "Output code\nReasoning: because I chose approach A for efficiency\n"
        bb.write("r1", long_val, "agent_1", "result")
        tool = ReadBlackboard(context)
        result = json.loads(tool.forward())
        entry = result["entries"][0]
        assert "because I chose approach A" not in entry["value"]

    def test_peer_monitoring_filters_metrics(self, context):
        context.config["peer_monitoring_visible"] = False
        bb = context.blackboard
        bb.write("s1", "Working\nerror_rate: 0.3\nDone", "agent_1", "status")
        tool = ReadBlackboard(context)
        result = json.loads(tool.forward())
        assert "error_rate" not in result["entries"][0]["value"]

    def test_empty_blackboard(self, context):
        tool = ReadBlackboard(context)
        result = json.loads(tool.forward())
        assert result["total_entries"] == 0
        assert result["entries"] == []


# ---- WriteBlackboard tests ---------------------------------------------------

class TestWriteBlackboard:
    def test_creates_entry(self, context):
        tool = WriteBlackboard(context, agent_name="agent_1")
        result = json.loads(tool.forward("task_1", "Working on it", "status"))
        assert result["success"] is True
        assert result["key"] == "task_1"
        assert result["entry_type"] == "status"
        assert result["version"] == 1

    def test_soft_claim_conflict_warning(self, context):
        bb = context.blackboard
        bb.write("subtask", "claimed", "agent_1", "claim")
        tool = WriteBlackboard(context, agent_name="agent_2")
        result = json.loads(tool.forward("subtask", "also claimed", "claim"))
        assert result["success"] is True
        assert result["warning"] is not None
        assert "already claimed" in result["warning"]

    def test_hard_claim_rejection(self, hard_context):
        bb = hard_context.blackboard
        bb.write("subtask", "claimed", "agent_1", "claim")
        tool = WriteBlackboard(hard_context, agent_name="agent_2")
        result = json.loads(tool.forward("subtask", "also claimed", "claim"))
        assert result["success"] is False
        assert "locked by agent_1" in result["error"]

    def test_hard_claim_success(self, hard_context):
        tool = WriteBlackboard(hard_context, agent_name="agent_1")
        result = json.loads(tool.forward("subtask", "claimed", "claim"))
        assert result["success"] is True

    def test_no_claim_mode_ignores(self, none_context):
        bb = none_context.blackboard
        bb.write("subtask", "claimed", "agent_1", "claim")
        tool = WriteBlackboard(none_context, agent_name="agent_2")
        result = json.loads(tool.forward("subtask", "also claimed", "claim"))
        assert result["success"] is True
        assert result.get("warning") is None

    def test_update_by_same_author(self, context):
        tool = WriteBlackboard(context, agent_name="agent_1")
        tool.forward("task", "v1", "status")
        result = json.loads(tool.forward("task", "v2", "status"))
        assert result["success"] is True
        assert result["version"] == 2


# ---- SpawnPeer tests ---------------------------------------------------------

class TestSpawnPeer:
    def test_spawns_new_agent(self, context):
        tool = SpawnPeer(context, agent_name="agent_1")
        result = json.loads(tool.forward("Need help with fillets"))
        assert result["success"] is True
        assert result["new_agent_name"] == "agent_3"
        assert result["total_agents"] == 3

    def test_new_agent_registered(self, context):
        tool = SpawnPeer(context, agent_name="agent_1")
        tool.forward("Need help")
        assert "agent_3" in context.agents
        assert isinstance(context.agents["agent_3"], ToolCallingAgent)

    def test_new_agent_has_correct_tools(self, context):
        tool = SpawnPeer(context, agent_name="agent_1")
        tool.forward("Need help")
        agent = context.agents["agent_3"]
        tool_names = set(agent.tools.keys())
        # Should have domain tools (echo_tool, calculator_tool).
        assert "echo_tool" in tool_names
        assert "calculator_tool" in tool_names

    def test_new_agent_has_peer_prompt(self, context):
        tool = SpawnPeer(context, agent_name="agent_1")
        tool.forward("Need help")
        agent = context.agents["agent_3"]
        assert agent.system_prompt is not None

    def test_rejects_when_max_reached(self, context):
        context.max_agents = 2  # already have 2 agents
        tool = SpawnPeer(context, agent_name="agent_1")
        result = json.loads(tool.forward("Need more agents"))
        assert result["success"] is False
        assert "Maximum agent limit" in result["error"]

    def test_posts_gap_entry_to_blackboard(self, context):
        tool = SpawnPeer(context, agent_name="agent_1")
        tool.forward("Nobody handling fillets")
        gaps = context.blackboard.read_by_type("gap")
        assert len(gaps) == 1
        assert "agent_3" in gaps[0].value
        assert "fillets" in gaps[0].value

    def test_spawned_agents_list(self, context):
        tool = SpawnPeer(context, agent_name="agent_1")
        tool.forward("Reason 1")
        tool.forward("Reason 2")
        assert context.spawned_agents == ["agent_3", "agent_4"]

    def test_auto_incrementing_names(self, context):
        tool = SpawnPeer(context, agent_name="agent_1")
        r1 = json.loads(tool.forward("Reason 1"))
        r2 = json.loads(tool.forward("Reason 2"))
        assert r1["new_agent_name"] == "agent_3"
        assert r2["new_agent_name"] == "agent_4"

    def test_new_agent_uses_shared_model(self, context):
        tool = SpawnPeer(context, agent_name="agent_1")
        tool.forward("Need help")
        agent = context.agents["agent_3"]
        assert agent.model is context.model


# ---- MarkTaskDone -----------------------------------------------------------

class TestMarkTaskDone:
    def test_writes_done_to_blackboard(self, context):
        tool = MarkTaskDone(context, agent_name="agent_1")
        tool.forward("STL generated and evaluated")
        entry = context.blackboard.get("task_complete")
        assert entry is not None
        assert "DONE" in entry.value.upper()

    def test_returns_success_json(self, context):
        tool = MarkTaskDone(context, agent_name="agent_1")
        raw = tool.forward("all done")
        result = json.loads(raw)
        assert result["success"] is True
        assert result["key"] == "task_complete"
        assert result["summary"] == "all done"

    def test_stores_calling_agent_as_author(self, context):
        tool = MarkTaskDone(context, agent_name="agent_2")
        tool.forward("finished")
        entry = context.blackboard.get("task_complete")
        assert entry.author == "agent_2"


# ---- PEER_TOOL_NAMES --------------------------------------------------------

class TestPeerToolNames:
    def test_contains_expected_names(self):
        assert PEER_TOOL_NAMES == {
            "read_blackboard", "write_blackboard", "spawn_peer", "mark_task_done"
        }
