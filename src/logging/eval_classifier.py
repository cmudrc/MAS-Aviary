"""Eval result classifier for Aviary batch runs.

Given eval results and threshold config, classifies each run as one of:
  - success: all metrics within thresholds
  - omission: system produced output, no agent flagged issues, but
              eval metrics outside thresholds
  - commission: an agent reviewed/approved output during execution,
                but eval shows metrics outside thresholds
  - sim_fail: simulation didn't converge (Aviary)
  - eval_skipped: eval was not performed
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Aviary evaluation
# ---------------------------------------------------------------------------

_AVIARY_METRIC_NAMES = [
    "fuel_burned_kg",
    "gtow_kg",
    "wing_mass_kg",
    "reserve_fuel_kg",
    "zero_fuel_weight_kg",
]

# Aviary-specific approval/issue keywords (from mdo_integrator VERDICT output).
_AVIARY_APPROVAL_KEYWORDS = {"COMPLETE", "TASK_COMPLETE"}
_AVIARY_ISSUE_KEYWORDS = {"RETRY", "CONTINUE", "FAILED"}


@dataclass
class AviaryEvalThresholds:
    """Aviary evaluation thresholds — 5 output metrics vs reference benchmark."""

    reference: dict[str, float]
    max_deviation_pct: float
    converged_required: bool


def load_aviary_thresholds(
    path: str | Path = "config/eval_thresholds.yaml",
) -> AviaryEvalThresholds:
    """Load Aviary thresholds from YAML file."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    aviary = raw.get("aviary", {})
    reference = aviary.get("reference", {})
    return AviaryEvalThresholds(
        reference={k: float(v) for k, v in reference.items()},
        max_deviation_pct=float(aviary.get("max_deviation_pct", 10.0)),
        converged_required=bool(aviary.get("converged_required", True)),
    )


DEFAULT_AVIARY_THRESHOLDS = AviaryEvalThresholds(
    reference={
        "fuel_burned_kg": 7000.65,
        "gtow_kg": 67365.86,
        "wing_mass_kg": 7466.35,
        "reserve_fuel_kg": 1360.78,
        "zero_fuel_weight_kg": 58604.23,
    },
    max_deviation_pct=10.0,
    converged_required=True,
)


@dataclass
class AviaryEvalClassification:
    """Result of classifying an Aviary eval run."""

    result: str             # "success" | "omission" | "commission" | "sim_fail" | "eval_skipped"
    converged: bool
    fuel_burned_kg: float
    gtow_kg: float
    wing_mass_kg: float
    reserve_fuel_kg: float
    zero_fuel_weight_kg: float
    optimality_gap_pct: float
    fuel_pass: bool
    gtow_pass: bool
    wing_mass_pass: bool
    reserve_fuel_pass: bool
    zero_fuel_weight_pass: bool
    reason: str = ""


def classify_aviary_eval(
    eval_result: dict | None,
    *,
    converged: bool = True,
    agent_approved: bool = False,
    agent_flagged_issues: bool = False,
    thresholds: AviaryEvalThresholds | None = None,
) -> AviaryEvalClassification:
    """Classify an Aviary eval run result.

    Args:
        eval_result: Dict with keys: fuel_burned_kg, gtow_kg, wing_mass_kg,
            reserve_fuel_kg, zero_fuel_weight_kg, optimality_gap_pct.
            None if eval was skipped.
        converged: Whether the simulation converged.
        agent_approved: Whether mdo_integrator gave VERDICT: COMPLETE.
        agent_flagged_issues: Whether mdo_integrator gave VERDICT: RETRY/CONTINUE.
        thresholds: Aviary thresholds. Defaults to DEFAULT_AVIARY_THRESHOLDS.

    Returns:
        AviaryEvalClassification with result and metric details.
    """
    t = thresholds or DEFAULT_AVIARY_THRESHOLDS

    # Eval skipped.
    if eval_result is None:
        return AviaryEvalClassification(
            result="eval_skipped",
            converged=converged,
            fuel_burned_kg=0.0,
            gtow_kg=0.0,
            wing_mass_kg=0.0,
            reserve_fuel_kg=0.0,
            zero_fuel_weight_kg=0.0,
            optimality_gap_pct=0.0,
            fuel_pass=False,
            gtow_pass=False,
            wing_mass_pass=False,
            reserve_fuel_pass=False,
            zero_fuel_weight_pass=False,
            reason="Eval data not extracted from agent messages",
        )

    # Simulation didn't converge.
    if not converged:
        return AviaryEvalClassification(
            result="sim_fail",
            converged=False,
            fuel_burned_kg=eval_result.get("fuel_burned_kg") or 0.0,
            gtow_kg=eval_result.get("gtow_kg") or 0.0,
            wing_mass_kg=eval_result.get("wing_mass_kg") or 0.0,
            reserve_fuel_kg=eval_result.get("reserve_fuel_kg") or 0.0,
            zero_fuel_weight_kg=eval_result.get("zero_fuel_weight_kg") or 0.0,
            optimality_gap_pct=eval_result.get("optimality_gap_pct") or 0.0,
            fuel_pass=False,
            gtow_pass=False,
            wing_mass_pass=False,
            reserve_fuel_pass=False,
            zero_fuel_weight_pass=False,
            reason="Simulation did not converge — results unreliable",
        )

    # Extract metrics and check deviation from reference.
    # None means "metric not extracted" — skip that metric in pass/fail.
    def _check(name: str) -> tuple[float, bool, bool]:
        """Returns (actual_value, passed, was_found)."""
        raw = eval_result.get(name)
        if raw is None:
            return 0.0, True, False  # Missing → skip (don't penalise)
        actual = float(raw)
        ref = t.reference.get(name, 0.0)
        if ref == 0.0:
            return actual, True, True
        deviation = abs(actual - ref) / ref * 100.0
        return actual, deviation <= t.max_deviation_pct, True

    fuel, fuel_pass, fuel_found = _check("fuel_burned_kg")
    gtow, gtow_pass, gtow_found = _check("gtow_kg")
    wing, wing_pass, wing_found = _check("wing_mass_kg")
    reserve, reserve_pass, reserve_found = _check("reserve_fuel_kg")
    zfw, zfw_pass, zfw_found = _check("zero_fuel_weight_kg")
    gap = float(eval_result.get("optimality_gap_pct", 0.0))

    # Only count found metrics for pass/fail — missing metrics are skipped.
    all_pass = fuel_pass and gtow_pass and wing_pass and reserve_pass and zfw_pass

    if all_pass:
        return AviaryEvalClassification(
            result="success",
            converged=True,
            fuel_burned_kg=fuel,
            gtow_kg=gtow,
            wing_mass_kg=wing,
            reserve_fuel_kg=reserve,
            zero_fuel_weight_kg=zfw,
            optimality_gap_pct=gap,
            fuel_pass=True,
            gtow_pass=True,
            wing_mass_pass=True,
            reserve_fuel_pass=True,
            zero_fuel_weight_pass=True,
            reason="All metrics within thresholds",
        )

    # Failed — determine omission vs commission.
    failed = []
    for name, actual, passed, found in [
        ("fuel_burned_kg", fuel, fuel_pass, fuel_found),
        ("gtow_kg", gtow, gtow_pass, gtow_found),
        ("wing_mass_kg", wing, wing_pass, wing_found),
        ("reserve_fuel_kg", reserve, reserve_pass, reserve_found),
        ("zero_fuel_weight_kg", zfw, zfw_pass, zfw_found),
    ]:
        if not passed and found:
            ref = t.reference.get(name, 0.0)
            dev = abs(actual - ref) / ref * 100.0 if ref else 0.0
            failed.append(f"{name} {actual:.1f} ({dev:.1f}% from ref {ref:.1f})")
    detail = "; ".join(failed)

    result_type = "commission" if agent_approved and not agent_flagged_issues else "omission"
    reason = (
        f"Agent approved but metrics failed: {detail}"
        if result_type == "commission"
        else f"Metrics outside thresholds: {detail}"
    )

    return AviaryEvalClassification(
        result=result_type,
        converged=True,
        fuel_burned_kg=fuel,
        gtow_kg=gtow,
        wing_mass_kg=wing,
        reserve_fuel_kg=reserve,
        zero_fuel_weight_kg=zfw,
        optimality_gap_pct=gap,
        fuel_pass=fuel_pass,
        gtow_pass=gtow_pass,
        wing_mass_pass=wing_pass,
        reserve_fuel_pass=reserve_pass,
        zero_fuel_weight_pass=zfw_pass,
        reason=reason,
    )


def detect_aviary_agent_signals(
    agent_outputs: list[str],
) -> tuple[bool, bool]:
    """Detect Aviary-specific approval/issue signals from mdo_integrator.

    Returns:
        (agent_approved, agent_flagged_issues) booleans.
    """
    approved = False
    flagged = False
    for output in agent_outputs:
        if any(kw in output for kw in _AVIARY_APPROVAL_KEYWORDS):
            approved = True
        if any(kw in output for kw in _AVIARY_ISSUE_KEYWORDS):
            flagged = True
    return approved, flagged
