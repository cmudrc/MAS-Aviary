"""Streamlit components for the Batch Results tab.

Renders 4 panels:
  A. Combination Comparison Table
  B. Strategy-Specific Metrics (expandable)
  C. Cross-Strategy Comparison Charts
  D. Org Theory Concept Coverage
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

try:
    import altair as alt
    HAS_ALTAIR = True
except ImportError:
    HAS_ALTAIR = False


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def list_batch_dirs(base: str = "logs/batch_results/") -> list[str]:
    """List batch result directories, newest first."""
    base_path = Path(base)
    if not base_path.exists():
        return []
    dirs = [str(d) for d in base_path.iterdir() if d.is_dir() and (d / "batch_summary.json").exists()]
    return sorted(dirs, reverse=True)


def load_batch_data(batch_dir: str) -> dict | None:
    """Load batch summary JSON from a directory."""
    path = Path(batch_dir) / "batch_summary.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Panel A: Combination Comparison Table
# ---------------------------------------------------------------------------

_RESULT_COLORS = {
    "success": "#2ecc71",
    "omission": "#f39c12",
    "commission": "#f39c12",
    "compile_fail": "#e74c3c",
    "sim_fail": "#e74c3c",
    "eval_skipped": "#95a5a6",
    "error": "#e74c3c",
}


def render_comparison_table(data: dict) -> None:
    """Render Panel A: Combination comparison table.

    Renders Aviary columns (Converged, Fuel, GTOW, Wing, Reserve, ZFW, Opt. Gap).
    """
    st.subheader("Combination Comparison")

    results = data.get("results", [])
    if not results:
        st.info("No results available.")
        return

    rows = []
    for r in results:
        ec = r.get("eval_classification", {})
        cs = r.get("cross_strategy_metrics", {})
        eval_result = ec.get("result", r.get("status", "unknown"))

        row = {
            "OS": r.get("org_structure", ""),
            "Handler": r.get("handler", ""),
            "Status": r.get("status", ""),
            "Turns": r.get("total_turns", 0),
            "Tokens": r.get("total_tokens", 0),
            "Duration (s)": round(r.get("duration_seconds", 0), 2),
            "Converged": ec.get("converged", ""),
            "Fuel (kg)": round(ec.get("fuel_burned_kg", 0), 1) if ec.get("fuel_burned_kg") else "",
            "GTOW (kg)": round(ec.get("gtow_kg", 0), 1) if ec.get("gtow_kg") else "",
            "Wing (kg)": round(ec.get("wing_mass_kg", 0), 1) if ec.get("wing_mass_kg") else "",
            "Reserve (kg)": round(ec.get("reserve_fuel_kg", 0), 1) if ec.get("reserve_fuel_kg") else "",
            "ZFW (kg)": round(ec.get("zero_fuel_weight_kg", 0), 1) if ec.get("zero_fuel_weight_kg") else "",
            "Opt. Gap (%)": round(ec.get("optimality_gap_pct", 0), 1) if ec.get("optimality_gap_pct") else "",
            "Eval Result": eval_result,
        }
        row["Coord. Overhead"] = round(cs.get("coordination_overhead", 0), 1)
        row["Redundancy"] = round(cs.get("redundancy_rate", 0), 3)
        row["Efficiency"] = round(cs.get("coordination_efficiency", 0), 3)
        row["Error Amp."] = cs.get("error_amplification", 0)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Color-code the Eval Result column via styling.
    def _color_eval(val):
        color = _RESULT_COLORS.get(val, "#95a5a6")
        return f"background-color: {color}; color: white; font-weight: bold"

    styled = df.style.applymap(_color_eval, subset=["Eval Result"])
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Org theory metric concept groupings (for Panel B and Panel D)
# ---------------------------------------------------------------------------

# Maps (os_name, concept_name) -> list of metric keys in org_theory_metrics
_OS_CONCEPT_GROUPS: dict[str, list[tuple[str, list[str]]]] = {
    "orchestrated": [
        ("Span of Control", [
            "agents_spawned", "orchestrator_turns", "worker_turns",
            "orchestrator_overhead_ratio", "cost_per_agent",
            "reasoning_iterations", "orchestrator_token_growth",
        ]),
        ("Oversight", ["authority_holder", "authority_transfers"]),
        ("Information Asymmetry", ["information_ratio"]),
    ],
    "networked": [
        ("Common Ground", [
            "blackboard_utilization", "blackboard_size_final",
            "blackboard_reads_total", "blackboard_writes_total",
        ]),
        ("Peer Monitoring", ["claim_conflicts"]),
        ("Joint Myopia", ["convergence_score", "convergence_classification"]),
        ("Predictive Knowledge", ["prediction_count", "prediction_accuracy_mean"]),
        ("Self-Organization", [
            "agents_spawned", "self_selection_diversity", "duplicate_work_rate",
        ]),
    ],
    "sequential": [
        ("Decomposition", [
            "stage_count", "per_stage_duration", "per_stage_tokens", "stage_bottleneck",
        ]),
        ("Modularity", [
            "tool_utilization_per_stage", "stage_independence_score",
            "tool_restriction_violations",
        ]),
        ("Viscosity", ["propagation_time", "per_stage_propagation"]),
        ("Mirroring Hypothesis", ["template_used"]),
    ],
}

_HANDLER_CONCEPT_GROUPS: dict[str, list[tuple[str, list[str]]]] = {
    "iterative_feedback": [
        ("Aspiration Levels", [
            "aspiration_mode", "mean_attempts_to_success", "max_attempts_used",
        ]),
        ("Ambidexterity", ["ambidexterity_score"]),
        ("Escalation of Commitment", ["escalation_length", "escalation_detected"]),
        ("False Negative Avoidance", ["early_stopping_count"]),
    ],
    "graph_routed": [
        ("Structural Distribution of Attention", [
            "initial_complexity", "final_complexity", "resource_utilization",
            "context_utilization", "complexity_escalations",
        ]),
        ("Omission vs Commission", [
            "total_transitions", "misroute_rate", "missed_routes", "routing_accuracy",
        ]),
        ("Coupled Search", ["complexity_budget", "budget_constrained"]),
        ("Internal Representations", ["mental_model_enabled"]),
    ],
    "staged_pipeline": [
        ("Decomposition", ["stage_count", "per_stage_duration", "per_stage_tokens"]),
        ("Aspiration Levels (Completion Gates)", [
            "completion_rate", "per_stage_completion",
        ]),
        ("Error Propagation", [
            "propagation_rate", "recovery_rate", "propagation_depth",
            "first_failure_stage", "error_propagation_count",
        ]),
    ],
    "placeholder": [],
}


def _render_org_theory_metrics(ot: dict, os_name: str, handler_name: str) -> None:
    """Render org theory metrics grouped by theory concept."""
    if not ot:
        st.caption("No org theory metrics computed.")
        return

    all_groups = (
        _OS_CONCEPT_GROUPS.get(os_name, [])
        + _HANDLER_CONCEPT_GROUPS.get(handler_name, [])
    )

    for concept, keys in all_groups:
        st.markdown(f"**{concept}**")
        rows = []
        for k in keys:
            if k in ot:
                val = ot[k]
                if isinstance(val, float):
                    val_str = f"{val:.4f}"
                elif isinstance(val, list):
                    val_str = str(val)[:120]
                else:
                    val_str = "—" if val is None else str(val)
                rows.append({"Metric": k, "Value": val_str})
        if rows:
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("  (no keys found)")

    # Warnings section (collapsed)
    warnings = ot.get("warnings", [])
    if warnings:
        with st.expander(f"⚠ {len(warnings)} metric(s) set to null — missing fields"):
            for w in warnings:
                st.caption(f"• {w}")


# ---------------------------------------------------------------------------
# Panel B: Strategy-Specific Metrics (expandable)
# ---------------------------------------------------------------------------

def render_strategy_metrics(data: dict) -> None:
    """Render Panel B: Expandable strategy-specific metrics per combination."""
    st.subheader("Strategy-Specific Metrics")

    results = data.get("results", [])
    if not results:
        st.info("No results available.")
        return

    for r in results:
        name = r.get("name", "unknown")
        status = r.get("status", "")
        label = f"{name} ({status})"

        with st.expander(label):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Strategy Metrics:**")
                sm = r.get("strategy_metrics", {})
                if sm:
                    for k, v in sm.items():
                        st.text(f"  {k}: {v}")
                else:
                    st.caption("No strategy-specific metrics recorded.")

            with col2:
                st.markdown("**Handler Metrics:**")
                hm = r.get("handler_metrics", {})
                if hm:
                    for k, v in hm.items():
                        st.text(f"  {k}: {v}")
                else:
                    st.caption("No handler-specific metrics recorded.")

            # Cross-strategy metrics.
            cs = r.get("cross_strategy_metrics", {})
            if cs:
                st.markdown("**Cross-Strategy Metrics:**")
                cols = st.columns(5)
                metric_names = [
                    ("coordination_overhead", "Coord. Overhead"),
                    ("message_density", "Msg Density"),
                    ("redundancy_rate", "Redundancy"),
                    ("coordination_efficiency", "Efficiency"),
                    ("error_amplification", "Error Amp."),
                ]
                for i, (key, label) in enumerate(metric_names):
                    val = cs.get(key, 0)
                    with cols[i]:
                        if isinstance(val, float):
                            st.metric(label, f"{val:.3f}")
                        else:
                            st.metric(label, val)

            # Org theory metrics grouped by concept.
            ot = r.get("org_theory_metrics", {})
            if ot:
                st.markdown("---")
                st.markdown("**Org Theory Metrics (by concept):**")
                _render_org_theory_metrics(
                    ot,
                    os_name=r.get("org_structure", ""),
                    handler_name=r.get("handler", ""),
                )

            # Eval details.
            ec = r.get("eval_classification", {})
            if ec:
                st.markdown("**Eval Classification:**")
                st.text(f"  Result: {ec.get('result', 'N/A')}")
                st.text(f"  Reason: {ec.get('reason', 'N/A')}")

            # Error message if present.
            err = r.get("error_message", "")
            if err:
                st.error(f"Error: {err}")


# ---------------------------------------------------------------------------
# Panel C: Cross-Strategy Comparison Charts
# ---------------------------------------------------------------------------

def render_comparison_charts(data: dict) -> None:
    """Render Panel C: Charts comparing combinations."""
    st.subheader("Cross-Strategy Comparison")

    results = data.get("results", [])
    if not results:
        st.info("No results available.")
        return

    if not HAS_ALTAIR:
        st.warning("Altair not installed. Install with: pip install altair")
        _render_charts_fallback(results)
        return

    # Chart 1: Google paper metrics grouped by handler.
    st.markdown("#### Cross-Strategy Metrics by Combination")
    chart_rows = []
    for r in results:
        cs = r.get("cross_strategy_metrics", {})
        for metric in ["coordination_overhead", "redundancy_rate", "coordination_efficiency", "error_amplification"]:
            chart_rows.append({
                "Combination": r.get("name", ""),
                "OS": r.get("org_structure", ""),
                "Handler": r.get("handler", ""),
                "Metric": metric.replace("_", " ").title(),
                "Value": float(cs.get(metric, 0)),
            })

    if chart_rows:
        df_chart = pd.DataFrame(chart_rows)
        chart = alt.Chart(df_chart).mark_bar().encode(
            x=alt.X("Combination:N", sort=None, axis=alt.Axis(labelAngle=-45)),
            y="Value:Q",
            color="Handler:N",
            column="Metric:N",
        ).properties(width=200, height=200)
        st.altair_chart(chart)

    # Chart 2: Eval metrics by combination.
    st.markdown("#### Eval Metrics by Combination")
    eval_rows = []

    aviary_metrics = [
        ("Fuel (kg)", "fuel_burned_kg"),
        ("GTOW (kg)", "gtow_kg"),
        ("Wing (kg)", "wing_mass_kg"),
        ("Reserve (kg)", "reserve_fuel_kg"),
        ("ZFW (kg)", "zero_fuel_weight_kg"),
    ]
    for r in results:
        ec = r.get("eval_classification", {})
        if ec.get("result") not in ("eval_skipped", None):
            for metric, key in aviary_metrics:
                eval_rows.append({
                    "Combination": r.get("name", ""),
                    "Metric": metric,
                    "Value": float(ec.get(key, 0)),
                })

    if eval_rows:
        df_eval = pd.DataFrame(eval_rows)
        chart2 = alt.Chart(df_eval).mark_bar().encode(
            x=alt.X("Combination:N", sort=None, axis=alt.Axis(labelAngle=-45)),
            y="Value:Q",
            color="Metric:N",
            column="Metric:N",
        ).properties(width=200, height=200)
        st.altair_chart(chart2)
    else:
        st.caption("No eval metrics available (eval may have been skipped).")

    # Chart 3: Heatmap — OS × Handler colored by eval result.
    st.markdown("#### Eval Result Heatmap (OS × Handler)")
    heatmap_rows = []
    result_to_num = {
        "success": 3, "omission": 2, "commission": 1,
        "compile_fail": 0, "sim_fail": 0,
        "eval_skipped": -1, "error": -1,
    }
    for r in results:
        ec = r.get("eval_classification", {})
        eval_result = ec.get("result", r.get("status", "unknown"))
        heatmap_rows.append({
            "OS": r.get("org_structure", ""),
            "Handler": r.get("handler", ""),
            "Result": eval_result,
            "Score": result_to_num.get(eval_result, -1),
        })

    if heatmap_rows:
        df_hm = pd.DataFrame(heatmap_rows)
        heatmap = alt.Chart(df_hm).mark_rect().encode(
            x="Handler:N",
            y="OS:N",
            color=alt.Color("Score:Q", scale=alt.Scale(
                domain=[-1, 0, 1, 2, 3],
                range=["#95a5a6", "#e74c3c", "#f39c12", "#f39c12", "#2ecc71"],
            )),
            tooltip=["OS", "Handler", "Result"],
        ).properties(width=400, height=200)

        # Overlay text.
        text = alt.Chart(df_hm).mark_text(color="white", fontWeight="bold").encode(
            x="Handler:N",
            y="OS:N",
            text="Result:N",
        )
        st.altair_chart(heatmap + text, use_container_width=True)


def _render_charts_fallback(results: list[dict]) -> None:
    """Fallback table-based rendering when altair is not available."""
    st.markdown("#### Cross-Strategy Metrics")
    rows = []
    for r in results:
        cs = r.get("cross_strategy_metrics", {})
        rows.append({
            "Name": r.get("name", ""),
            "Overhead": round(cs.get("coordination_overhead", 0), 1),
            "Redundancy": round(cs.get("redundancy_rate", 0), 3),
            "Efficiency": round(cs.get("coordination_efficiency", 0), 3),
            "Error Amp.": cs.get("error_amplification", 0),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Panel D: Org Theory Concept Coverage
# ---------------------------------------------------------------------------

# Mapping: theory concept → (strategy, metric(s) that measure it)
_CONCEPT_MAP = [
    # Orchestration concepts.
    ("Span of Control", "Orchestrated", ["agents_spawned", "orchestrator_overhead_ratio"]),
    ("Oversight", "Orchestrated", ["orchestrator_turns", "authority_score"]),
    ("Information Asymmetry", "Orchestrated", ["information_ratio"]),
    # Networked concepts.
    ("Common Ground", "Networked", ["blackboard_utilization", "blackboard_size"]),
    ("Peer Monitoring", "Networked", ["duplicate_work_rate"]),
    ("Trans-specialist Knowledge", "Networked", ["self_selection_diversity"]),
    ("Joint Myopia", "Networked", ["mean_joint_myopia_score"]),
    ("Predictive Knowledge", "Networked", ["prediction_accuracy"]),
    # Sequential concepts.
    ("Decomposition", "Sequential", ["stage_count", "stage_independence_score"]),
    ("Modularity", "Sequential", ["tool_restriction_violations", "interface_pass_rate"]),
    ("Near-Decomposability", "Sequential", ["per_stage_tool_utilization"]),
    ("Mirroring Hypothesis", "Sequential", ["pipeline_template"]),
    ("Viscosity", "Sequential", ["propagation_time"]),
    # Iterative Feedback concepts.
    ("Aspiration Levels", "Iterative Feedback", ["mean_attempts_to_success"]),
    ("Problemistic Search", "Iterative Feedback", ["agents_exhausted_retries"]),
    ("Ambidexterity", "Iterative Feedback", ["ambidexterity_score"]),
    ("False Negative Avoidance", "Iterative Feedback", ["escalation_length"]),
    ("Escalation of Commitment", "Iterative Feedback", ["escalation_detected"]),
    # Graph-Routed concepts.
    ("Structural Distribution of Attention", "Graph Routed", ["resource_utilization", "path_efficiency"]),
    ("Coupled Search", "Graph Routed", ["routing_accuracy", "misroute_rate"]),
    ("Internal Representations", "Graph Routed", ["context_utilization"]),
    ("Omission vs Commission", "Graph Routed", ["omission_error_count", "commission_error_count"]),
    # Staged Pipeline concepts.
    ("Error Propagation", "Staged Pipeline", ["propagation_rate", "recovery_rate", "chain_length"]),
    ("Stage Coupling", "Staged Pipeline", ["error_propagation_count", "error_recovery_count"]),
    ("Assembly Line Efficiency", "Staged Pipeline", ["completion_rate", "total_duration"]),
    ("Completion Criteria Sensitivity", "Staged Pipeline", ["first_failure_stage", "propagation_depth"]),
]


def render_concept_coverage(data: dict) -> None:
    """Render Panel D: Org Theory Concept Coverage table.

    Values are taken from the org_theory_metrics dict computed by
    compute_org_theory_metrics(). Falls back to strategy_metrics /
    handler_metrics for combinations that pre-date the org theory metrics
    integration.
    """
    st.subheader("Org Theory Concept Coverage")

    results = data.get("results", [])

    rows = []
    for concept, strategy, metrics in _CONCEPT_MAP:
        # Collect computed values across all result combinations.
        values: dict[str, Any] = {}
        for r in results:
            # Primary source: org_theory_metrics
            ot = r.get("org_theory_metrics", {})
            for m in metrics:
                if m in ot and m not in values and ot[m] is not None:
                    values[m] = ot[m]

            # Fallback: legacy strategy_metrics / handler_metrics / cross_strategy
            fallback: dict = {}
            fallback.update(r.get("strategy_metrics", {}))
            fallback.update(r.get("handler_metrics", {}))
            fallback.update(r.get("cross_strategy_metrics", {}))
            fallback.update(r.get("eval_classification", {}))
            for m in metrics:
                if m not in values and m in fallback:
                    values[m] = fallback[m]

        def _fmt(v: Any) -> str:
            if v is None:
                return "null"
            if isinstance(v, float):
                return f"{v:.4f}"
            if isinstance(v, list):
                return str(v)[:80]
            return str(v)

        values_str = ", ".join(
            f"{m}={_fmt(values[m])}" if m in values else f"{m}=N/A"
            for m in metrics
        )
        # Simple "computed?" flag
        computed_count = sum(1 for m in metrics if m in values)
        coverage = f"{computed_count}/{len(metrics)}"

        rows.append({
            "Concept": concept,
            "Strategy/Handler": strategy,
            "Coverage": coverage,
            "Metrics": ", ".join(metrics),
            "Values": values_str,
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Main batch results renderer
# ---------------------------------------------------------------------------

def render_batch_results() -> None:
    """Top-level renderer for the Batch Results tab."""
    batch_dirs = list_batch_dirs()

    if not batch_dirs:
        st.info(
            "No batch results found. Run the batch runner first:\n\n"
            "```bash\npython -m src.runners.batch_runner --prompt-index 1\n```"
        )
        return

    selected = st.selectbox(
        "Select batch run",
        batch_dirs,
        format_func=lambda x: Path(x).name,
    )

    data = load_batch_data(selected)
    if data is None:
        st.error(f"Could not load batch data from {selected}")
        return

    st.caption(f"Task: {data.get('task', 'N/A')}")
    st.caption(f"Combinations: {data.get('total_combinations', 0)}")

    # Render all 4 panels.
    render_comparison_table(data)
    st.divider()
    render_comparison_charts(data)
    st.divider()
    render_strategy_metrics(data)
    st.divider()
    render_concept_coverage(data)
