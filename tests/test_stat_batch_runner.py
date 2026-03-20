"""Unit tests for statistical batch runner (no MCP / GPU needed)."""

import math
from unittest.mock import MagicMock

import pytest

# Import the module functions we're testing
from scripts.stat_batch_runner import (
    PARAM_RANGES,
    _extract_session_id,
    build_task_with_session,
    generate_all_param_sets,
    generate_random_params,
    load_checkpoint,
    run_key,
    save_checkpoint,
    setup_session_with_params,
)

# ---------------------------------------------------------------------------
# Parameter generation
# ---------------------------------------------------------------------------

class TestGenerateRandomParams:
    def test_all_params_present(self):
        params = generate_random_params(seed=42)
        for name in PARAM_RANGES:
            assert name in params, f"Missing param: {name}"
        assert "_derived_span" in params

    def test_params_within_range(self):
        params = generate_random_params(seed=99)
        for name, (lo, hi) in PARAM_RANGES.items():
            assert lo <= params[name] <= hi, (
                f"{name}={params[name]} outside [{lo}, {hi}]"
            )

    def test_derived_span_correct(self):
        params = generate_random_params(seed=42)
        expected = round(
            math.sqrt(
                params["Aircraft.Wing.ASPECT_RATIO"]
                * params["Aircraft.Wing.AREA"]
            ),
            4,
        )
        assert params["_derived_span"] == expected

    def test_deterministic_with_same_seed(self):
        a = generate_random_params(seed=123)
        b = generate_random_params(seed=123)
        assert a == b

    def test_different_seeds_differ(self):
        a = generate_random_params(seed=1)
        b = generate_random_params(seed=2)
        assert a != b

    def test_rounding(self):
        params = generate_random_params(seed=42)
        for name in PARAM_RANGES:
            s = str(params[name])
            if "." in s:
                decimals = len(s.split(".")[1])
                assert decimals <= 4


class TestGenerateAllParamSets:
    def test_correct_count(self):
        sets = generate_all_param_sets(5, base_seed=0)
        assert len(sets) == 5

    def test_each_set_unique(self):
        sets = generate_all_param_sets(10, base_seed=0)
        # Each set should differ (different seeds)
        ar_values = [s["Aircraft.Wing.ASPECT_RATIO"] for s in sets]
        assert len(set(ar_values)) == 10

    def test_reproducible(self):
        a = generate_all_param_sets(3, base_seed=42)
        b = generate_all_param_sets(3, base_seed=42)
        assert a == b


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

class TestCheckpoint:
    def test_load_nonexistent(self, tmp_path):
        ckpt = load_checkpoint(tmp_path / "missing.json")
        assert ckpt == {"completed": {}, "failed": {}}

    def test_save_and_load(self, tmp_path):
        path = tmp_path / "ckpt.json"
        data = {
            "completed": {"r000_combo_a": {"status": "ok"}},
            "failed": {},
        }
        save_checkpoint(path, data)
        loaded = load_checkpoint(path)
        assert loaded == data

    def test_atomic_save(self, tmp_path):
        """Verify .tmp file is cleaned up after atomic save."""
        path = tmp_path / "ckpt.json"
        save_checkpoint(path, {"completed": {}, "failed": {}})
        assert path.exists()
        assert not path.with_suffix(".tmp").exists()


class TestRunKey:
    def test_format(self):
        assert run_key(0, "combo_a") == "r000_combo_a"
        assert run_key(29, "my_combo") == "r029_my_combo"


# ---------------------------------------------------------------------------
# Session ID extraction
# ---------------------------------------------------------------------------

class TestExtractSessionId:
    def test_from_dict(self):
        resp = {"success": True, "session_id": "abc-123-def"}
        assert _extract_session_id(resp) == "abc-123-def"

    def test_from_uuid_string(self):
        resp = "Session created: 12345678-1234-1234-1234-123456789abc"
        assert _extract_session_id(resp) == "12345678-1234-1234-1234-123456789abc"

    def test_from_dict_with_uuid(self):
        resp = {"session_id": "abcdef01-2345-6789-abcd-ef0123456789"}
        assert _extract_session_id(resp) == "abcdef01-2345-6789-abcd-ef0123456789"

    def test_failure(self):
        with pytest.raises(RuntimeError, match="no session_id"):
            _extract_session_id("no uuid here")

    def test_failure_on_empty_dict(self):
        with pytest.raises(RuntimeError, match="no session_id"):
            _extract_session_id({})


# ---------------------------------------------------------------------------
# setup_session_with_params (mocked MCP tools)
# ---------------------------------------------------------------------------

class TestSetupSessionWithParams:
    def _make_tool_map(self, create_resp, configure_resp=None):
        """Build a mock tool_map with create_session and configure_mission."""
        create = MagicMock()
        create.forward.return_value = create_resp

        configure = MagicMock()
        configure.forward.return_value = configure_resp or {"success": True}

        return {
            "create_session": create,
            "configure_mission": configure,
        }

    def test_happy_path_dict_response(self):
        tool_map = self._make_tool_map(
            create_resp={
                "success": True,
                "session_id": "aaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                "initial_parameters_applied": True,
            }
        )
        params = {"Aircraft.Wing.AREA": 130.0, "_derived_span": 35.0}
        result = setup_session_with_params(tool_map, params)

        assert result["session_id"] == "aaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        # Should pass only settable params (no _derived_span)
        tool_map["create_session"].forward.assert_called_once_with(
            initial_parameters={"Aircraft.Wing.AREA": 130.0}
        )
        # Should configure mission
        tool_map["configure_mission"].forward.assert_called_once()

    def test_filters_underscore_params(self):
        tool_map = self._make_tool_map(
            create_resp={"session_id": "11111111-2222-3333-4444-555555555555"}
        )
        params = {
            "Aircraft.Wing.AREA": 140.0,
            "Aircraft.Wing.SWEEP": 25.0,
            "_derived_span": 40.0,
            "_internal": 99,
        }
        setup_session_with_params(tool_map, params)

        call_kwargs = tool_map["create_session"].forward.call_args
        passed_params = call_kwargs.kwargs["initial_parameters"]
        assert "_derived_span" not in passed_params
        assert "_internal" not in passed_params
        assert "Aircraft.Wing.AREA" in passed_params

    def test_string_response_uuid_extraction(self):
        tool_map = self._make_tool_map(
            create_resp="Session 12345678-abcd-1234-abcd-123456789abc created OK"
        )
        result = setup_session_with_params(tool_map, {})
        assert result["session_id"] == "12345678-abcd-1234-abcd-123456789abc"

    def test_create_session_failure(self):
        tool_map = self._make_tool_map(create_resp="Error: server down")
        with pytest.raises(RuntimeError):
            setup_session_with_params(tool_map, {})


# ---------------------------------------------------------------------------
# build_task_with_session
# ---------------------------------------------------------------------------

class TestBuildTaskWithSession:
    def test_contains_session_id(self):
        task = build_task_with_session("Optimize fuel.", "sess-123")
        assert "sess-123" in task

    def test_contains_base_task(self):
        task = build_task_with_session("Optimize fuel.", "sess-123")
        assert "Optimize fuel." in task

    def test_warns_against_create_session(self):
        """Task warns that create_session will produce a broken session."""
        task = build_task_with_session("base", "s1")
        assert "WILL FAIL" in task or "will fail" in task.lower()
        assert "create_session" in task.lower()

    def test_includes_params_when_provided(self):
        params = {
            "Aircraft.Wing.ASPECT_RATIO": 10.5,
            "Aircraft.Wing.AREA": 130.0,
            "_derived_span": 36.9,  # should be excluded
        }
        task = build_task_with_session("base", "s1", params)
        assert "ASPECT_RATIO: 10.5" in task
        assert "AREA: 130.0" in task
        assert "_derived_span" not in task

    def test_no_params_section_when_none(self):
        task = build_task_with_session("base", "s1")
        assert "have already been applied" not in task

    def test_session_already_created_language(self):
        """Task uses strong language — 'already been created'."""
        task = build_task_with_session("base", "s1")
        assert "already been created" in task.lower()
