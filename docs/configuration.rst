=============
Configuration
=============

MAS-Aviary is configured through a set of YAML files in the ``config/``
directory. This page documents every configuration file and its fields.

aviary_run.yaml
===============

The primary runtime configuration file. Controls LLM settings, MCP server
connection, and general run parameters.

.. code-block:: yaml

   # LLM Configuration
   llm:
     model_id: "..."          # HuggingFace model ID or API model name
     max_new_tokens: 4096     # Maximum tokens per LLM response
     temperature: 0.7         # Sampling temperature
     provider: "transformers" # LLM provider backend

   # MCP Server Configuration
   mcp:
     url: "http://localhost:8600"  # Aviary MCP server URL
     transport: "streamable-http"  # MCP transport protocol
     timeout: 120                  # Request timeout in seconds
     retries: 3                    # Number of retry attempts

   # Run Configuration
   run:
     combination: "aviary_sequential_iterative_feedback"
     output_dir: "results/"
     log_level: "INFO"

**LLM section**:

- ``model_id`` -- The model identifier. For local models, this is a
  HuggingFace repo path. For API models, the provider's model name.
- ``max_new_tokens`` -- Upper bound on generated tokens per call.
- ``temperature`` -- Controls randomness. Lower values produce more
  deterministic outputs.
- ``provider`` -- Backend to use (``transformers``, ``openai``, etc.).

**MCP section**:

- ``url`` -- Full URL to the `Aviary MCP server <https://github.com/Jezemba/Aviary>`_.
- ``transport`` -- Protocol transport. Use ``streamable-http`` for HTTP-based
  MCP.
- ``timeout`` -- Per-request timeout in seconds.
- ``retries`` -- Automatic retry count for transient failures.


aviary_sequential.yaml
======================

Configuration for the Sequential organizational structure.

.. code-block:: yaml

   pipeline:
     template: "default"
     stages:
       - name: "initial_sizing"
         agent: "sizing_agent"
         max_steps: 10
       - name: "aerodynamic_analysis"
         agent: "aero_agent"
         max_steps: 15
       - name: "propulsion_optimization"
         agent: "propulsion_agent"
         max_steps: 15
       - name: "final_evaluation"
         agent: "eval_agent"
         max_steps: 10

   context:
     pass_full_history: true   # Each stage sees all prior outputs
     summarize_prior: false    # If true, summarize instead of full pass

- ``pipeline.template`` -- Named template that defines the stage ordering.
- ``pipeline.stages`` -- Ordered list of stages. Each has a ``name``,
  assigned ``agent``, and ``max_steps`` limit.
- ``context.pass_full_history`` -- When true, each stage receives the
  concatenated outputs of all preceding stages.
- ``context.summarize_prior`` -- When true, prior outputs are summarized
  before being passed to the next stage.


aviary_networked.yaml
=====================

Configuration for the Networked organizational structure.

.. code-block:: yaml

   workflow:
     phases:
       - name: "explore"
         max_rounds: 5
         agents: ["sizing_agent", "aero_agent", "propulsion_agent"]
       - name: "refine"
         max_rounds: 10
         agents: ["sizing_agent", "aero_agent", "propulsion_agent"]
       - name: "converge"
         max_rounds: 3
         agents: ["eval_agent"]

   blackboard:
     claiming_mode: "shared"    # "shared" or "exclusive"
     sections:
       - "parameters"
       - "evaluation"
       - "proposals"
       - "consensus"

   termination:
     max_total_rounds: 30
     convergence_threshold: 0.01

- ``workflow.phases`` -- Named phases executed in order. Each phase lists
  participating agents and a ``max_rounds`` limit.
- ``blackboard.claiming_mode`` -- Controls task ownership:

  - ``shared`` -- Multiple agents can read and act on the same entry.
  - ``exclusive`` -- An entry is claimed by one agent; others skip it.

- ``blackboard.sections`` -- Named sections of the blackboard. Agents post
  and read from specific sections.
- ``termination.max_total_rounds`` -- Hard limit on total rounds across all
  phases.
- ``termination.convergence_threshold`` -- If metric change between rounds
  drops below this value, the workflow terminates early.


aviary_graph.yaml
=================

Configuration for the Graph-Routed operational handler.

.. code-block:: yaml

   graph:
     initial_state: "start"
     states:
       - name: "start"
         agent: "sizing_agent"
         action: "initial_sizing"
       - name: "refine_aero"
         agent: "aero_agent"
         action: "refine_aerodynamics"
       - name: "refine_propulsion"
         agent: "propulsion_agent"
         action: "refine_propulsion"
       - name: "evaluate"
         agent: "eval_agent"
         action: "evaluate_design"
       - name: "done"
         terminal: true

     transitions:
       - from: "start"
         to: "refine_aero"
         condition: "fuel_burned > 15000"
       - from: "start"
         to: "refine_propulsion"
         condition: "fuel_burned <= 15000"
       - from: "refine_aero"
         to: "evaluate"
         condition: "always"
       - from: "refine_propulsion"
         to: "evaluate"
         condition: "always"
       - from: "evaluate"
         to: "refine_aero"
         condition: "not converged"
       - from: "evaluate"
         to: "done"
         condition: "converged"

   resources:
     per_state_max_steps: 10
     per_state_timeout: 300     # seconds
     global_max_transitions: 50

- ``graph.initial_state`` -- The entry-point state.
- ``graph.states`` -- List of states. Each maps to an agent and action.
  A state with ``terminal: true`` ends the workflow.
- ``graph.transitions`` -- Directed edges between states. Each has a
  ``condition`` expression evaluated by the ``ConditionEvaluator``.
- ``resources.per_state_max_steps`` -- Maximum LLM steps per state visit.
- ``resources.per_state_timeout`` -- Wall-clock timeout per state in seconds.
- ``resources.global_max_transitions`` -- Hard limit on total state
  transitions.


aviary_staged_pipeline.yaml
===========================

Configuration for the Staged Pipeline operational handler.

.. code-block:: yaml

   stages:
     - name: "exploration"
       agents: ["sizing_agent", "aero_agent"]
       max_steps: 20
       completion_criteria:
         metric: "fuel_burned"
         threshold: 20000
         direction: "below"

     - name: "refinement"
       agents: ["propulsion_agent", "aero_agent"]
       max_steps: 30
       completion_criteria:
         metric: "gross_takeoff_weight"
         threshold: 80000
         direction: "below"

     - name: "finalization"
       agents: ["eval_agent"]
       max_steps: 10
       completion_criteria:
         all_metrics_within_threshold: true

   gates:
     strict: true              # Fail the run if a gate is not passed
     retry_on_failure: 2       # Retries before failing a gate

- ``stages`` -- Ordered list of stages. Each stage declares:

  - ``name`` -- Human-readable stage label.
  - ``agents`` -- List of agents active during this stage.
  - ``max_steps`` -- Maximum total LLM steps for the stage.
  - ``completion_criteria`` -- Quality gate that must be satisfied to
    advance. Can reference a single metric with a threshold and direction,
    or use ``all_metrics_within_threshold`` for the final stage.

- ``gates.strict`` -- When true, a failed gate aborts the entire run.
- ``gates.retry_on_failure`` -- Number of times to retry a stage before
  declaring gate failure.


eval_thresholds.yaml
====================

Defines the evaluation thresholds used to classify optimization results as
pass or fail.

.. code-block:: yaml

   thresholds:
     fuel_burned_kg:
       target: 12000
       acceptable: 14000
       unit: "kg"

     gross_takeoff_weight_kg:
       target: 75000
       acceptable: 82000
       unit: "kg"

     wing_mass_kg:
       target: 4200
       acceptable: 5000
       unit: "kg"

   classification:
     excellent: "all metrics at or below target"
     acceptable: "all metrics at or below acceptable"
     failed: "any metric above acceptable"

- ``thresholds`` -- Per-metric definitions:

  - ``target`` -- The ideal value. Results at or below this are classified
    as excellent.
  - ``acceptable`` -- The upper bound for a passing result.
  - ``unit`` -- Display unit for reporting.

- ``classification`` -- Rules for mapping metric values to overall result
  labels.
