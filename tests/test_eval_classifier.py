"""Unit tests for eval classifier.

Tests Aviary metrics: fuel_burned_kg, gtow_kg, wing_mass_kg, etc.
"""

from src.logging.eval_classifier import (
    DEFAULT_AVIARY_THRESHOLDS,
    AviaryEvalClassification,
    AviaryEvalThresholds,
    classify_aviary_eval,
    detect_aviary_agent_signals,
    load_aviary_thresholds,
)

# ---- Thresholds loading ----------------------------------------------------


class TestLoadThresholds:
    def test_load_from_file(self):
        t = load_aviary_thresholds("config/eval_thresholds.yaml")
        assert isinstance(t, AviaryEvalThresholds)
        assert t.converged_required is True
        assert t.max_deviation_pct > 0

    def test_default_thresholds(self):
        t = DEFAULT_AVIARY_THRESHOLDS
        assert t.max_deviation_pct == 10.0
        assert "fuel_burned_kg" in t.reference
        assert "gtow_kg" in t.reference


# ---- Success classification ------------------------------------------------


class TestSuccess:
    def test_all_metrics_pass(self):
        result = classify_aviary_eval(
            {"fuel_burned_kg": 7000.65, "gtow_kg": 67365.86, "wing_mass_kg": 7466.35},
            converged=True,
        )
        assert result.result == "success"
        assert result.converged is True
        assert result.fuel_pass is True
        assert result.gtow_pass is True
        assert result.wing_mass_pass is True

    def test_exact_reference_pass(self):
        result = classify_aviary_eval(
            DEFAULT_AVIARY_THRESHOLDS.reference.copy(),
            converged=True,
        )
        assert result.result == "success"

    def test_within_deviation(self):
        # 5% deviation on fuel should still pass (threshold is 10%)
        ref = DEFAULT_AVIARY_THRESHOLDS.reference["fuel_burned_kg"]
        result = classify_aviary_eval(
            {"fuel_burned_kg": ref * 1.05, "gtow_kg": 67365.86, "wing_mass_kg": 7466.35},
            converged=True,
        )
        assert result.fuel_pass is True


# ---- Simulation failure ----------------------------------------------------


class TestSimFail:
    def test_sim_fail(self):
        result = classify_aviary_eval(
            {"fuel_burned_kg": 7000.0, "gtow_kg": 67000.0, "wing_mass_kg": 7400.0},
            converged=False,
        )
        assert result.result == "sim_fail"
        assert result.converged is False

    def test_sim_fail_ignores_good_metrics(self):
        result = classify_aviary_eval(
            DEFAULT_AVIARY_THRESHOLDS.reference.copy(),
            converged=False,
        )
        assert result.result == "sim_fail"


# ---- Eval skipped ----------------------------------------------------------


class TestEvalSkipped:
    def test_eval_none(self):
        result = classify_aviary_eval(None, converged=True)
        assert result.result == "eval_skipped"

    def test_eval_none_unconverged(self):
        result = classify_aviary_eval(None, converged=False)
        assert result.result == "eval_skipped"


# ---- Omission errors -------------------------------------------------------


class TestOmission:
    def test_fuel_over_threshold(self):
        # 50% deviation should fail
        ref = DEFAULT_AVIARY_THRESHOLDS.reference["fuel_burned_kg"]
        result = classify_aviary_eval(
            {"fuel_burned_kg": ref * 1.5, "gtow_kg": 67365.86, "wing_mass_kg": 7466.35},
            converged=True,
        )
        assert result.result == "omission"
        assert result.fuel_pass is False
        assert result.gtow_pass is True

    def test_gtow_over_threshold(self):
        ref = DEFAULT_AVIARY_THRESHOLDS.reference["gtow_kg"]
        result = classify_aviary_eval(
            {"fuel_burned_kg": 7000.65, "gtow_kg": ref * 2.0, "wing_mass_kg": 7466.35},
            converged=True,
        )
        assert result.result == "omission"
        assert result.gtow_pass is False

    def test_wing_mass_over_threshold(self):
        ref = DEFAULT_AVIARY_THRESHOLDS.reference["wing_mass_kg"]
        result = classify_aviary_eval(
            {"fuel_burned_kg": 7000.65, "gtow_kg": 67365.86, "wing_mass_kg": ref * 2.0},
            converged=True,
        )
        assert result.result == "omission"
        assert result.wing_mass_pass is False

    def test_all_metrics_fail(self):
        result = classify_aviary_eval(
            {"fuel_burned_kg": 99999.0, "gtow_kg": 999999.0, "wing_mass_kg": 99999.0},
            converged=True,
        )
        assert result.result == "omission"

    def test_agent_flagged_issues_still_omission(self):
        result = classify_aviary_eval(
            {"fuel_burned_kg": 99999.0, "gtow_kg": 999999.0, "wing_mass_kg": 99999.0},
            converged=True,
            agent_flagged_issues=True,
        )
        assert result.result == "omission"


# ---- Commission errors -----------------------------------------------------


class TestCommission:
    def test_agent_approved_but_failed(self):
        result = classify_aviary_eval(
            {"fuel_burned_kg": 99999.0, "gtow_kg": 999999.0, "wing_mass_kg": 99999.0},
            converged=True,
            agent_approved=True,
            agent_flagged_issues=False,
        )
        assert result.result == "commission"
        assert "Agent approved" in result.reason

    def test_agent_approved_and_flagged_not_commission(self):
        result = classify_aviary_eval(
            {"fuel_burned_kg": 99999.0, "gtow_kg": 999999.0, "wing_mass_kg": 99999.0},
            converged=True,
            agent_approved=True,
            agent_flagged_issues=True,
        )
        assert result.result == "omission"


# ---- Custom thresholds -----------------------------------------------------


class TestCustomThresholds:
    def test_stricter_thresholds(self):
        strict = AviaryEvalThresholds(
            reference=DEFAULT_AVIARY_THRESHOLDS.reference.copy(),
            max_deviation_pct=0.1,
            converged_required=True,
        )
        # 5% deviation should fail with 0.1% threshold
        ref = strict.reference["fuel_burned_kg"]
        result = classify_aviary_eval(
            {"fuel_burned_kg": ref * 1.05, "gtow_kg": 67365.86, "wing_mass_kg": 7466.35},
            converged=True,
            thresholds=strict,
        )
        assert result.result == "omission"

    def test_lenient_thresholds(self):
        lenient = AviaryEvalThresholds(
            reference=DEFAULT_AVIARY_THRESHOLDS.reference.copy(),
            max_deviation_pct=100.0,
            converged_required=True,
        )
        result = classify_aviary_eval(
            {"fuel_burned_kg": 10000.0, "gtow_kg": 80000.0, "wing_mass_kg": 10000.0},
            converged=True,
            thresholds=lenient,
        )
        assert result.result == "success"


# ---- Agent signal detection ------------------------------------------------


class TestDetectAgentSignals:
    def test_approved(self):
        approved, flagged = detect_aviary_agent_signals(["TASK_COMPLETE — done"])
        assert approved is True
        assert flagged is False

    def test_complete(self):
        approved, flagged = detect_aviary_agent_signals(["Output: COMPLETE"])
        assert approved is True

    def test_flagged_failed(self):
        approved, flagged = detect_aviary_agent_signals(["Execution FAILED entirely"])
        assert approved is False
        assert flagged is True

    def test_flagged_retry(self):
        approved, flagged = detect_aviary_agent_signals(["Need to RETRY this"])
        assert approved is False
        assert flagged is True

    def test_both_approved_and_flagged(self):
        approved, flagged = detect_aviary_agent_signals(
            [
                "TASK_COMPLETE",
                "RETRY needed later",
            ]
        )
        assert approved is True
        assert flagged is True

    def test_no_signals(self):
        approved, flagged = detect_aviary_agent_signals(["General discussion."])
        assert approved is False
        assert flagged is False

    def test_empty_outputs(self):
        approved, flagged = detect_aviary_agent_signals([])
        assert approved is False
        assert flagged is False


# ---- Classification result fields ------------------------------------------


class TestClassificationFields:
    def test_all_fields_present(self):
        result = classify_aviary_eval(
            DEFAULT_AVIARY_THRESHOLDS.reference.copy(),
            converged=True,
        )
        assert isinstance(result, AviaryEvalClassification)
        assert isinstance(result.result, str)
        assert isinstance(result.converged, bool)
        assert isinstance(result.fuel_burned_kg, float)
        assert isinstance(result.gtow_kg, float)
        assert isinstance(result.wing_mass_kg, float)
        assert isinstance(result.fuel_pass, bool)
        assert isinstance(result.gtow_pass, bool)
        assert isinstance(result.wing_mass_pass, bool)
        assert isinstance(result.reason, str)
