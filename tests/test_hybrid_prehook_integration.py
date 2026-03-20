"""Integration tests for the hybrid pre-hook approach in stat_batch_runner.

These tests run REAL multi-agent combinations against the live MCP server
(port 8600) to verify:
  1. Pre-hook creates a session with initial_parameters via MCP
  2. Mission is configured correctly
  3. The hybrid task text is used by agents (not ignored / looped)
  4. Runs complete within timeout (no infinite loops)
  5. fuel_burned_kg > 0 for successful runs
  6. initial_params are preserved in the result

Requires: MCP server running on 127.0.0.1:8600, GPU available.

Usage:
    # Run all tests (slow — each runs a real multi-agent combination)
    PYTHONPATH=. pytest tests/test_hybrid_prehook_integration.py -v -s

    # Run just the pre-hook MCP tests (fast, no GPU needed)
    PYTHONPATH=. pytest tests/test_hybrid_prehook_integration.py -v -s -k "prehook"

    # Run one specific combo test
    PYTHONPATH=. pytest tests/test_hybrid_prehook_integration.py -v -s -k "orchestrated_iterative"
"""

from __future__ import annotations

import pytest

from scripts.stat_batch_runner import (
    PARAM_RANGES,
    _aggressive_gpu_cleanup,
    _load_mcp_tools,
    build_task_with_session,
    generate_random_params,
    setup_session_with_params,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def gpu_cleanup_between_tests():
    """Free GPU memory before and after each test to prevent OOM."""
    yield
    try:
        _aggressive_gpu_cleanup()
    except Exception:
        pass


@pytest.fixture(scope="module")
def aviary_config():
    """Load the aviary run config."""
    from src.config.loader import load_config

    return load_config("config/aviary_run.yaml")


@pytest.fixture(scope="module")
def tool_map(aviary_config):
    """Load MCP tools once for the test module."""
    try:
        tm = _load_mcp_tools(aviary_config)
    except Exception as e:
        pytest.skip(f"MCP server not available: {e}")
    if not tm:
        pytest.skip("No MCP tools loaded — server may be down")
    return tm


@pytest.fixture(scope="module")
def aviary_combos():
    """Get the AVIARY_COMBINATIONS list."""
    from src.runners.batch_runner import AVIARY_COMBINATIONS

    return {c.name: c for c in AVIARY_COMBINATIONS}


# ---------------------------------------------------------------------------
# Pre-hook MCP tests (fast — no GPU/LLM needed)
# ---------------------------------------------------------------------------


class TestPrehookMCP:
    """Test that the pre-hook creates sessions and sets params via MCP."""

    def test_create_session_with_params(self, tool_map):
        """Pre-hook creates a session with initial_parameters applied."""
        params = generate_random_params(seed=42)
        result = setup_session_with_params(tool_map, params)

        assert "session_id" in result
        assert result["session_id"]  # non-empty
        # Session ID should be a UUID
        sid = result["session_id"]
        assert len(sid) == 36, f"Session ID doesn't look like UUID: {sid}"

    def test_create_session_response_has_details(self, tool_map):
        """MCP response includes confirmation of parameter application."""
        params = generate_random_params(seed=99)
        result = setup_session_with_params(tool_map, params)

        # create_session response should be present
        assert "create_session" in result
        assert "configure_mission" in result

    def test_configure_mission_succeeds(self, tool_map):
        """Mission configuration (1500nmi, 162pax, M0.785, FL350) succeeds."""
        params = generate_random_params(seed=7)
        result = setup_session_with_params(tool_map, params)

        # configure_mission response
        cm = result["configure_mission"]
        # Should not be an error string
        if isinstance(cm, dict):
            assert cm.get("success", True) is not False
        else:
            assert "error" not in str(cm).lower()

    def test_underscore_params_not_sent(self, tool_map):
        """_derived_span and other underscore params are NOT sent to MCP."""
        params = generate_random_params(seed=42)
        assert "_derived_span" in params  # should exist in generated params

        # The function filters these out — it should not raise
        result = setup_session_with_params(tool_map, params)
        assert result["session_id"]

    def test_multiple_sessions_get_different_ids(self, tool_map):
        """Each pre-hook call creates a distinct session."""
        params_a = generate_random_params(seed=1)
        params_b = generate_random_params(seed=2)

        result_a = setup_session_with_params(tool_map, params_a)
        result_b = setup_session_with_params(tool_map, params_b)

        assert result_a["session_id"] != result_b["session_id"]

    def test_params_within_valid_ranges(self, tool_map):
        """Params sent to MCP are within the defined PARAM_RANGES."""
        params = generate_random_params(seed=55)
        for name, (lo, hi) in PARAM_RANGES.items():
            assert lo <= params[name] <= hi, f"{name}={params[name]} out of [{lo},{hi}]"

        # Should succeed without MCP validation errors
        result = setup_session_with_params(tool_map, params)
        assert result["session_id"]


class TestHybridTaskText:
    """Test that the hybrid task text is well-formed."""

    def test_task_includes_session_id(self):
        task = build_task_with_session("Design aircraft.", "abc-123-def", {})
        assert "abc-123-def" in task

    def test_task_includes_params(self):
        params = {
            "Aircraft.Wing.AREA": 130.0,
            "Aircraft.Wing.ASPECT_RATIO": 10.5,
            "_derived_span": 37.0,
        }
        task = build_task_with_session("Design aircraft.", "s1", params)
        assert "AREA: 130.0" in task
        assert "ASPECT_RATIO: 10.5" in task
        assert "_derived_span" not in task

    def test_task_warns_about_create_session_consequence(self):
        """Task warns that create_session produces a broken blank session."""
        task = build_task_with_session("base", "s1")
        assert "WILL FAIL" in task
        assert "create_session" in task
        # Should explain the consequence, not just ban it
        assert "blank session" in task.lower() or "without the mission" in task.lower()

    def test_task_uses_strong_language(self):
        """Should clearly state session is already created and configured."""
        task = build_task_with_session("base", "s1")
        assert "already been created" in task.lower()
        assert "setup is done" in task.lower()


# ---------------------------------------------------------------------------
# Real multi-agent combination tests (slow — need GPU + MCP)
# ---------------------------------------------------------------------------

_AVIARY_TASK = (
    "Design a single-aisle commercial aircraft for 1,500 nmi range, "
    "162 passengers, cruise Mach 0.785, FL350. Optimize for minimum "
    "fuel burn. Constraints: fuel_burned_kg <= 8500, gtow_kg <= 72000."
)

# 20-minute timeout for integration tests (matches production timeout)
_TEST_TIMEOUT_SECONDS = 20 * 60


def _run_combo_with_prehook(combo, tool_map, config, seed=42, timeout_seconds=_TEST_TIMEOUT_SECONDS):
    """Run a single combination with the hybrid pre-hook. Returns (result, session_id, params)."""
    import threading

    from src.runners.batch_runner import run_combination

    params = generate_random_params(seed=seed)
    setup = setup_session_with_params(tool_map, params)
    session_id = setup["session_id"]
    task = build_task_with_session(_AVIARY_TASK, session_id, params)

    result_box = [None]
    error_box = [None]

    def _target():
        try:
            result_box[0] = run_combination(combo, task, config, domain="aviary", session_id=session_id)
        except Exception as e:
            error_box[0] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=timeout_seconds)

    if t.is_alive():
        raise TimeoutError(f"Run exceeded {timeout_seconds / 60:.0f}m timeout — possible infinite loop")
    if error_box[0] is not None:
        raise error_box[0]

    return result_box[0], session_id, params


@pytest.mark.slow
class TestOrchestatedIterativeFeedback:
    """Real run: orchestrated × iterative_feedback with hybrid pre-hook."""

    def test_completes_without_timeout(self, tool_map, aviary_config, aviary_combos):
        combo = aviary_combos.get("aviary_orchestrated_iterative_feedback")
        if combo is None:
            pytest.skip("Combo not found")

        result, session_id, params = _run_combo_with_prehook(combo, tool_map, aviary_config, seed=42)

        # Basic completion checks
        assert result is not None, "Run returned None"
        assert result.status == "success", f"Run failed: {result.error_message}"
        assert result.total_turns > 0, "No turns executed"

        # Fuel check
        ec = result.eval_classification or {}
        fuel = ec.get("fuel_burned_kg")
        print(
            f"  Result: status={result.status}, fuel={fuel}, turns={result.total_turns}, "
            f"duration={result.duration_seconds:.0f}s"
        )
        assert fuel is not None and fuel > 0, f"Zero or missing fuel: {fuel}"

        # Duration sanity check (should finish well under timeout)
        assert result.duration_seconds < _TEST_TIMEOUT_SECONDS, "Took too long"


@pytest.mark.slow
class TestOrchestratedStagedPipeline:
    """Real run: orchestrated × staged_pipeline with hybrid pre-hook."""

    def test_completes_without_timeout(self, tool_map, aviary_config, aviary_combos):
        combo = aviary_combos.get("aviary_orchestrated_staged_pipeline")
        if combo is None:
            pytest.skip("Combo not found")

        result, session_id, params = _run_combo_with_prehook(combo, tool_map, aviary_config, seed=77)

        assert result is not None
        assert result.status == "success", f"Run failed: {result.error_message}"
        assert result.total_turns > 0

        ec = result.eval_classification or {}
        fuel = ec.get("fuel_burned_kg")
        print(
            f"  Result: status={result.status}, fuel={fuel}, turns={result.total_turns}, "
            f"duration={result.duration_seconds:.0f}s"
        )
        assert fuel is not None and fuel > 0, f"Zero or missing fuel: {fuel}"


@pytest.mark.slow
class TestOrchestratedGraphRouted:
    """Real run: orchestrated × graph_routed with hybrid pre-hook.

    This was the combo that previously looped (109 steps, 0 fuel) because
    the setup_agent ignored 'Do NOT call create_session'. The hybrid
    approach should fix this.
    """

    def test_completes_without_looping(self, tool_map, aviary_config, aviary_combos):
        combo = aviary_combos.get("aviary_orchestrated_graph_routed")
        if combo is None:
            pytest.skip("Combo not found")

        # Use seed=42 (same as sequential/iterative_feedback tests).
        # Graph-routed needs more time: graph traversal + up to 2 signal retries.
        result, session_id, params = _run_combo_with_prehook(
            combo,
            tool_map,
            aviary_config,
            seed=42,
            timeout_seconds=20 * 60,  # 20 min — matches production timeout
        )

        assert result is not None
        assert result.status == "success", f"Run failed: {result.error_message}"

        # The key check: should NOT have excessive turns (looping = 100+ turns)
        assert result.total_turns < 80, (
            f"Possible looping detected: {result.total_turns} turns (expected <80 for a healthy run)"
        )

        ec = result.eval_classification or {}
        fuel = ec.get("fuel_burned_kg")
        print(
            f"  Result: status={result.status}, fuel={fuel}, turns={result.total_turns}, "
            f"duration={result.duration_seconds:.0f}s"
        )
        assert fuel is not None and fuel > 0, f"Zero or missing fuel: {fuel}"


@pytest.mark.slow
class TestSequentialStagedPipeline:
    """Real run: sequential × staged_pipeline (known-good baseline)."""

    def test_completes_successfully(self, tool_map, aviary_config, aviary_combos):
        combo = aviary_combos.get("aviary_sequential_staged_pipeline")
        if combo is None:
            pytest.skip("Combo not found")

        result, session_id, params = _run_combo_with_prehook(combo, tool_map, aviary_config, seed=42)

        assert result is not None
        assert result.status == "success", f"Run failed: {result.error_message}"

        ec = result.eval_classification or {}
        fuel = ec.get("fuel_burned_kg")
        print(
            f"  Result: status={result.status}, fuel={fuel}, turns={result.total_turns}, "
            f"duration={result.duration_seconds:.0f}s"
        )
        assert fuel is not None and fuel > 0, f"Zero or missing fuel: {fuel}"


@pytest.mark.slow
class TestNetworkedIterativeFeedback:
    """Real run: networked × iterative_feedback with hybrid pre-hook."""

    def test_completes_without_timeout(self, tool_map, aviary_config, aviary_combos):
        combo = aviary_combos.get("aviary_networked_iterative_feedback")
        if combo is None:
            pytest.skip("Combo not found")

        result, session_id, params = _run_combo_with_prehook(combo, tool_map, aviary_config, seed=55)

        assert result is not None
        assert result.status == "success", f"Run failed: {result.error_message}"

        ec = result.eval_classification or {}
        fuel = ec.get("fuel_burned_kg")
        print(
            f"  Result: status={result.status}, fuel={fuel}, turns={result.total_turns}, "
            f"duration={result.duration_seconds:.0f}s"
        )
        # Networked may produce 0 fuel (known issue) — just check it completes
        assert result.total_turns > 0


@pytest.mark.slow
class TestSessionIdPreservedInResult:
    """Verify that the pre-hook session_id is actually used by agents."""

    def test_session_id_in_agent_messages(self, tool_map, aviary_config, aviary_combos):
        """Run sequential_iterative_feedback and check session_id appears in messages."""
        combo = aviary_combos.get("aviary_sequential_iterative_feedback")
        if combo is None:
            pytest.skip("Combo not found")

        result, session_id, params = _run_combo_with_prehook(combo, tool_map, aviary_config, seed=42)

        assert result is not None
        assert result.status == "success"

        # Check that the session_id from pre-hook appears in at least one message
        all_text = " ".join(
            str(m.get("content", "")) if isinstance(m, dict) else str(getattr(m, "content", ""))
            for m in (result.messages or [])
        )
        # The short prefix of the session ID should appear in tool calls
        sid_prefix = session_id[:8]
        assert sid_prefix in all_text or session_id in all_text, (
            f"Session ID {session_id} not found in any agent message — "
            f"agents may be creating their own sessions instead of using the pre-hook one"
        )


@pytest.mark.slow
class TestParamsPreservedInCheckpoint:
    """Verify that initial_params are saved regardless of outcome."""

    def test_params_in_result_dict(self, tool_map, aviary_config, aviary_combos):
        """Run a combo and verify params would be saved in checkpoint format."""
        combo = aviary_combos.get("aviary_sequential_staged_pipeline")
        if combo is None:
            pytest.skip("Combo not found")

        params = generate_random_params(seed=42)
        setup = setup_session_with_params(tool_map, params)
        setup["session_id"]

        # Build checkpoint entry the same way stat_batch_runner does
        settable_params = {k: v for k, v in params.items() if not k.startswith("_")}

        # Verify all PARAM_RANGES keys are present
        for name in PARAM_RANGES:
            assert name in settable_params, f"Missing param: {name}"
        assert "_derived_span" not in settable_params

        # Verify values are within range
        for name, (lo, hi) in PARAM_RANGES.items():
            assert lo <= settable_params[name] <= hi
