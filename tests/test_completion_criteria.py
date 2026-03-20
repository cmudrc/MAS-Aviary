"""Unit tests for completion criteria evaluation."""


from src.coordination.completion_criteria import (
    CompletionCriteria,
    evaluate_completion,
    load_completion_criteria,
)
from src.coordination.history import ToolCallRecord

# ---- Helpers ---------------------------------------------------------------

def _tc(name: str, error: str | None = None) -> ToolCallRecord:
    """Shortcut to create a ToolCallRecord."""
    return ToolCallRecord(
        tool_name=name, inputs={}, output="ok", duration_seconds=0.1,
        error=error,
    )


# ---- non_empty_output ------------------------------------------------------

class TestNonEmptyOutput:
    def test_passes_on_non_empty_string(self):
        c = CompletionCriteria(type="output_contains", check="non_empty_output")
        r = evaluate_completion(c, "Hello world")
        assert r.met is True

    def test_fails_on_empty_string(self):
        c = CompletionCriteria(type="output_contains", check="non_empty_output")
        r = evaluate_completion(c, "")
        assert r.met is False

    def test_fails_on_whitespace_only(self):
        c = CompletionCriteria(type="output_contains", check="non_empty_output")
        r = evaluate_completion(c, "   \n\t  ")
        assert r.met is False

    def test_passes_on_single_char(self):
        c = CompletionCriteria(type="output_contains", check="non_empty_output")
        r = evaluate_completion(c, "x")
        assert r.met is True

    def test_fails_on_none_content(self):
        c = CompletionCriteria(type="output_contains", check="non_empty_output")
        r = evaluate_completion(c, None)
        assert r.met is False


# ---- code_block ------------------------------------------------------------

class TestCodeBlock:
    def test_passes_on_triple_backtick_python(self):
        c = CompletionCriteria(type="output_contains", check="code_block")
        content = "Here is code:\n```python\nprint('hi')\n```"
        r = evaluate_completion(c, content)
        assert r.met is True

    def test_passes_on_import_statement(self):
        c = CompletionCriteria(type="output_contains", check="code_block")
        r = evaluate_completion(c, "import openmdao.api as om\nprob = om.Problem()")
        assert r.met is True

    def test_passes_on_api_call(self):
        c = CompletionCriteria(type="output_contains", check="code_block")
        r = evaluate_completion(c, "result = om.Problem()\nresult.setup()")
        assert r.met is True

    def test_passes_on_import_plus_def(self):
        c = CompletionCriteria(type="output_contains", check="code_block")
        content = "import math\ndef compute(): return math.pi"
        r = evaluate_completion(c, content)
        assert r.met is True

    def test_passes_on_import_plus_assignment(self):
        c = CompletionCriteria(type="output_contains", check="code_block")
        content = "import os\npath = os.getcwd()"
        r = evaluate_completion(c, content)
        assert r.met is True

    def test_fails_on_plain_text(self):
        c = CompletionCriteria(type="output_contains", check="code_block")
        r = evaluate_completion(c, "This is just a geometry plan with steps.")
        assert r.met is False

    def test_fails_on_empty_output(self):
        c = CompletionCriteria(type="output_contains", check="code_block")
        r = evaluate_completion(c, "")
        assert r.met is False

    def test_custom_patterns(self):
        c = CompletionCriteria(type="output_contains", check="code_block")
        r = evaluate_completion(
            c, "from numpy import array",
            code_block_patterns=["from numpy"],
        )
        assert r.met is True

    def test_custom_patterns_no_match(self):
        c = CompletionCriteria(type="output_contains", check="code_block")
        r = evaluate_completion(
            c, "No code here",
            code_block_patterns=["```python", "from numpy"],
        )
        assert r.met is False


# ---- tool_called -----------------------------------------------------------

class TestToolCalled:
    def test_passes_when_tool_called(self):
        c = CompletionCriteria(
            type="tool_attempted", check="tool_called",
            tool_name="run_simulation",
        )
        r = evaluate_completion(c, "done", [_tc("run_simulation")])
        assert r.met is True

    def test_passes_even_when_tool_failed(self):
        c = CompletionCriteria(
            type="tool_attempted", check="tool_called",
            tool_name="run_simulation",
        )
        tc = _tc("run_simulation", error="SyntaxError")
        r = evaluate_completion(c, "done", [tc])
        assert r.met is True  # attempted counts

    def test_fails_when_no_tool_calls(self):
        c = CompletionCriteria(
            type="tool_attempted", check="tool_called",
            tool_name="run_simulation",
        )
        r = evaluate_completion(c, "done", [])
        assert r.met is False

    def test_fails_when_different_tool_called(self):
        c = CompletionCriteria(
            type="tool_attempted", check="tool_called",
            tool_name="run_simulation",
        )
        r = evaluate_completion(c, "done", [_tc("get_design_space")])
        assert r.met is False

    def test_passes_with_multiple_tools_including_target(self):
        c = CompletionCriteria(
            type="tool_attempted", check="tool_called",
            tool_name="run_simulation",
        )
        tools = [_tc("get_design_space"), _tc("run_simulation")]
        r = evaluate_completion(c, "done", tools)
        assert r.met is True

    def test_any_tool_when_no_tool_name(self):
        c = CompletionCriteria(
            type="tool_attempted", check="tool_called",
            tool_name=None,
        )
        r = evaluate_completion(c, "done", [_tc("something")])
        assert r.met is True

    def test_dict_tool_calls(self):
        c = CompletionCriteria(
            type="tool_attempted", check="tool_called",
            tool_name="my_tool",
        )
        r = evaluate_completion(c, "done", [{"tool_name": "my_tool"}])
        assert r.met is True

    def test_fails_with_none_tool_calls(self):
        c = CompletionCriteria(
            type="tool_attempted", check="tool_called",
            tool_name="x",
        )
        r = evaluate_completion(c, "done", None)
        assert r.met is False


# ---- verdict_present -------------------------------------------------------

class TestVerdictPresent:
    def test_passes_on_acceptable(self):
        c = CompletionCriteria(type="output_contains", check="verdict_present")
        r = evaluate_completion(c, "The result is ACCEPTABLE.")
        assert r.met is True

    def test_passes_on_task_complete(self):
        c = CompletionCriteria(type="output_contains", check="verdict_present")
        r = evaluate_completion(c, "TASK_COMPLETE")
        assert r.met is True

    def test_passes_on_failed(self):
        c = CompletionCriteria(type="output_contains", check="verdict_present")
        r = evaluate_completion(c, "Execution failed entirely.")
        assert r.met is True

    def test_passes_on_issues(self):
        c = CompletionCriteria(type="output_contains", check="verdict_present")
        r = evaluate_completion(c, "There are ISSUES with the result.")
        assert r.met is True

    def test_fails_with_no_verdict_keywords(self):
        c = CompletionCriteria(type="output_contains", check="verdict_present")
        r = evaluate_completion(c, "Here is some general discussion.")
        assert r.met is False

    def test_custom_verdict_patterns(self):
        c = CompletionCriteria(type="output_contains", check="verdict_present")
        r = evaluate_completion(
            c, "Result: APPROVED",
            verdict_patterns=["APPROVED", "REJECTED"],
        )
        assert r.met is True


# ---- always ----------------------------------------------------------------

class TestAlways:
    def test_always_passes_regardless(self):
        c = CompletionCriteria(type="any", check="always")
        r = evaluate_completion(c, "")
        assert r.met is True

    def test_always_with_any_type(self):
        c = CompletionCriteria(type="any", check="something")
        r = evaluate_completion(c, "")
        assert r.met is True

    def test_always_check_on_non_any_type(self):
        c = CompletionCriteria(type="output_contains", check="always")
        r = evaluate_completion(c, "")
        assert r.met is True


# ---- Unknown types ---------------------------------------------------------

class TestUnknownTypes:
    def test_unknown_criteria_type(self):
        c = CompletionCriteria(type="magic", check="unicorn")
        r = evaluate_completion(c, "content")
        assert r.met is False
        assert "Unknown criteria type" in r.reason

    def test_unknown_check_for_output_contains(self):
        c = CompletionCriteria(type="output_contains", check="magic_check")
        r = evaluate_completion(c, "content")
        assert r.met is False
        assert "Unknown check" in r.reason


# ---- Loading ---------------------------------------------------------------

class TestLoadCompletionCriteria:
    def test_load_full(self):
        data = {
            "type": "tool_attempted",
            "check": "tool_called",
            "description": "Tool was called",
            "tool_name": "my_tool",
        }
        c = load_completion_criteria(data)
        assert c.type == "tool_attempted"
        assert c.check == "tool_called"
        assert c.description == "Tool was called"
        assert c.tool_name == "my_tool"

    def test_load_defaults(self):
        c = load_completion_criteria({})
        assert c.type == "any"
        assert c.check == "always"
        assert c.description == ""
        assert c.tool_name is None

    def test_load_partial(self):
        c = load_completion_criteria({"type": "output_contains", "check": "code_block"})
        assert c.type == "output_contains"
        assert c.check == "code_block"
        assert c.tool_name is None


# ---- CompletionResult fields -----------------------------------------------

class TestCompletionResultFields:
    def test_result_has_all_fields(self):
        c = CompletionCriteria(type="output_contains", check="non_empty_output")
        r = evaluate_completion(c, "hello")
        assert isinstance(r.met, bool)
        assert isinstance(r.reason, str)
        assert isinstance(r.evidence, str)

    def test_evidence_on_code_block_match(self):
        c = CompletionCriteria(type="output_contains", check="code_block")
        r = evaluate_completion(c, "```python\nprint(1)\n```")
        assert "```python" in r.evidence

    def test_evidence_on_tool_called(self):
        c = CompletionCriteria(
            type="tool_attempted", check="tool_called",
            tool_name="exec",
        )
        r = evaluate_completion(c, "done", [_tc("exec")])
        assert "exec" in r.evidence
