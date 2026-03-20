============
Architecture
============

This page describes the full architecture of MAS-Aviary, covering the
end-to-end flow from user request to Aviary optimization, the three
organizational structures, the three operational handlers, and how they
compose into nine distinct coordination strategies.

Overview
========

The system follows a layered architecture where each layer has a single
responsibility:

.. code-block:: text

   +--------+     +--------------+     +------------+     +---------+
   |  User  | --> | Coordinator  | --> |  Strategy   | --> | Handler |
   +--------+     +--------------+     +------------+     +---------+
                                                               |
                                                               v
                                                          +---------+
                                                          |  Agent  |
                                                          +---------+
                                                               |
                                                               v
                                                     +-------------------+
                                                     | MCP Server (:8600)|
                                                     +-------------------+
                                                               |
                                                               v
                                                     +-------------------+
                                                     |   Aviary / OpenMDAO|
                                                     +-------------------+

**Coordinator** (``src.coordination.coordinator``)
    Entry point. Loads configuration, selects the strategy and handler,
    manages the agent registry, and orchestrates the full run lifecycle.

**Strategy** (``src.coordination.strategies``)
    Defines *how agents are organized*: who talks to whom and in what
    structure. Three strategies are available (see below).

**Handler** (``src.coordination.execution_handler``)
    Defines *how work proceeds*: the operational pattern that controls
    iteration, routing, and completion. Three handlers are available.

**Agent** (``src.agents``)
    An LLM-backed entity that reasons about aircraft design parameters and
    invokes MCP tools.

**MCP Server**
    Exposes Aviary design-space tools over the Model Context Protocol on
    port 8600.


Organizational Structures (Strategies)
======================================

The organizational structure determines the communication topology among
agents. Each structure is implemented as a subclass of the base ``Strategy``.

Sequential
----------

.. code-block:: text

   +----------+     +----------+     +----------+     +----------+
   | Agent A  | --> | Agent B  | --> | Agent C  | --> | Agent D  |
   +----------+     +----------+     +----------+     +----------+
       Stage 1          Stage 2          Stage 3          Stage 4

Agents execute one after another in a fixed pipeline order. Each agent
receives the accumulated context from all prior stages and appends its own
results before passing control to the next agent.

- **Module**: ``src.coordination.strategies.sequential``
- **Config**: ``config/aviary_sequential.yaml``
- **Key properties**:

  - Deterministic execution order.
  - Each agent sees the full history of prior stages.
  - Pipeline template defines stage names and assigned agents.
  - Stage-level ``max_steps`` limits per-agent iterations.

Orchestrated
------------

.. code-block:: text

                    +----------------+
                    |  Orchestrator  |
                    +----------------+
                   /        |        \
                  v         v         v
           +--------+ +--------+ +--------+
           | Agent A| | Agent B| | Agent C|
           +--------+ +--------+ +--------+

A central orchestrator agent delegates tasks to specialist agents. The
orchestrator decides which agent to invoke, what task to assign, and when
the overall objective is met.

- **Module**: ``src.coordination.strategies.orchestrated``
- **Config**: ``config/aviary_run.yaml`` (orchestrator section)
- **Key properties**:

  - Centralized decision-making via the orchestrator agent.
  - Dynamic task delegation -- the orchestrator chooses agents at runtime.
  - Specialist agents have narrow, well-defined roles.
  - The orchestrator synthesizes partial results into a final answer.

Networked
---------

.. code-block:: text

           +--------+       +--------+
           | Agent A| <---> | Agent B|
           +--------+       +--------+
               ^    \       /    ^
               |     v     v     |
               |   +---------+   |
               |   |Blackboard|  |
               |   +---------+   |
               |     ^     ^     |
               v    /       \    v
           +--------+       +--------+
           | Agent C| <---> | Agent D|
           +--------+       +--------+

Agents communicate through a shared **Blackboard** -- a central data store
where agents post findings, claim tasks, and read each other's results.
There is no fixed ordering; agents operate concurrently in workflow phases.

- **Module**: ``src.coordination.strategies.networked``
- **Config**: ``config/aviary_networked.yaml``
- **Key properties**:

  - Decentralized, peer-to-peer communication via the blackboard.
  - Workflow proceeds in named phases (e.g., ``explore``, ``refine``,
    ``converge``).
  - Claiming mode controls whether agents compete for or share tasks.
  - Agents read and react to other agents' blackboard entries.


Operational Handlers
====================

Handlers define the operational pattern -- *how* agents do their work within
the structure defined by the strategy.

Iterative Feedback Handler
--------------------------

.. code-block:: text

   +-------+     +----------+     +----------+     +--------+
   | Start | --> | Agent    | --> | Evaluate | --> | Better |--+
   +-------+     | executes |     | result   |     | than   |  |
                 +----------+     +----------+     | prev?  |  |
                      ^                            +--------+  |
                      |               No                |      |
                      +<--------------------------------+      |
                                                    Yes |      |
                                                        v      |
                                                   +--------+  |
                                                   |  Done  |<-+
                                                   +--------+
                                                  (max iters)

Each agent iterates on its task, receiving feedback from the evaluation of
its prior attempt. The loop continues until the result meets acceptance
criteria or the maximum iteration count is reached.

- **Module**: ``src.coordination.iterative_feedback_handler``
- **Config**: ``config/iterative_feedback.yaml``
- **Key components**:

  - ``FeedbackExtraction`` -- extracts actionable feedback from evaluation.
  - ``CompletionCriteria`` -- determines when an agent's work is sufficient.
  - ``TerminationChecker`` -- enforces global stopping conditions.

Staged Pipeline Handler
-----------------------

.. code-block:: text

   +-----------+     +------------+     +-----------+     +----------+
   | Stage 1   | --> | Gate 1     | --> | Stage 2   | --> | Gate 2   |
   | (explore) |     | (criteria) |     | (refine)  |     | (criteria)|
   +-----------+     +------------+     +-----------+     +----------+
                                                               |
                                                               v
                                                         +-----------+
                                                         | Stage 3   |
                                                         | (finalize)|
                                                         +-----------+

Work is divided into named stages, each with explicit completion criteria
that act as quality gates. An agent (or group of agents) must satisfy the
gate before the pipeline advances to the next stage.

- **Module**: ``src.coordination.staged_pipeline_handler``
- **Config**: ``config/aviary_staged_pipeline.yaml``
- **Key components**:

  - ``StageDefinition`` -- declares stage name, assigned agents, and goals.
  - ``CompletionCriteria`` -- per-stage quality gates.
  - ``PipelineTemplate`` -- ordered list of stages.

Graph-Routed Handler
--------------------

.. code-block:: text

   +----------+          +----------+          +----------+
   |  State A | --T1---> |  State B | --T2---> |  State C |
   +----------+          +----------+          +----------+
        |                     |                     |
        +------T3----------->-+------T4----------->-+
        |                                           |
        +-------------------T5--------------------->+

   T1..T5 = transitions (condition-guarded edges)

The handler models the optimization as a **state machine**. Each state maps
to an agent action, and transitions are guarded by conditions evaluated
against the current results. This allows non-linear, adaptive workflows
where the next step depends on runtime data.

- **Module**: ``src.coordination.graph_routed_handler``
- **Config**: ``config/aviary_graph.yaml``
- **Key components**:

  - ``GraphDefinition`` -- declares states and transitions.
  - ``ConditionEvaluator`` -- evaluates transition guard expressions.
  - ``ResourceManager`` -- enforces per-state resource budgets (steps, time).


Strategy x Handler Combinations
===============================

Every organizational structure can be paired with every operational handler,
producing a 3 x 3 matrix of nine combinations:

.. list-table:: Combination Matrix
   :header-rows: 1
   :widths: 25 25 25 25

   * - Structure \\ Handler
     - Iterative Feedback
     - Staged Pipeline
     - Graph-Routed
   * - **Sequential**
     - sequential_iterative_feedback
     - sequential_staged_pipeline
     - sequential_graph_routed
   * - **Orchestrated**
     - orchestrated_iterative_feedback
     - orchestrated_staged_pipeline
     - orchestrated_graph_routed
   * - **Networked**
     - networked_iterative_feedback
     - networked_staged_pipeline
     - networked_graph_routed

Each combination has a unique identifier (shown in the cells) used in
configuration files and the ``stat_batch_runner.py`` ``--combinations`` flag.


Blackboard Communication (Networked)
=====================================

The blackboard is the shared memory substrate for the Networked strategy.

.. code-block:: text

   +-----------------------------------------------------------+
   |                       Blackboard                          |
   |-----------------------------------------------------------|
   | Section        | Writer   | Content                       |
   |----------------|----------|-------------------------------|
   | parameters     | Agent A  | {"wing_area": 120.5, ...}     |
   | evaluation     | Agent B  | {"fuel_burned": 13200, ...}   |
   | proposals      | Agent C  | {"change": "increase AR",...} |
   | consensus      | All      | {"agreed_params": {...}}      |
   +-----------------------------------------------------------+

- Agents **post** entries tagged with their identity and a section name.
- Agents **read** entries from other sections to inform their reasoning.
- The **claiming mode** (``exclusive`` or ``shared``) controls whether a
  posted task can be picked up by one agent or many.
- The blackboard is persisted across workflow phases within a single run.


Orchestrator Delegation Flow
=============================

In the Orchestrated strategy, the orchestrator agent follows this decision
loop:

.. code-block:: text

   +---------------------+
   | Orchestrator starts  |
   +---------------------+
            |
            v
   +---------------------+
   | Assess current state |<--------------------------+
   +---------------------+                           |
            |                                         |
            v                                         |
   +---------------------+     No                     |
   | Objective met?       |-------+                   |
   +---------------------+       |                   |
            | Yes                 v                   |
            v            +------------------+         |
   +--------+            | Select specialist |         |
   |  Done  |            +------------------+         |
   +--------+                    |                    |
                                 v                    |
                        +------------------+          |
                        | Delegate subtask  |          |
                        +------------------+          |
                                 |                    |
                                 v                    |
                        +------------------+          |
                        | Collect result    |----------+
                        +------------------+

The orchestrator uses special tools (``src.tools.orchestrator_tools``) to
delegate work and collect results from specialist agents.


Sequential Pipeline Stages
===========================

In the Sequential strategy, the pipeline template defines an ordered list of
stages. Each stage specifies:

1. **Stage name** -- a human-readable label (e.g., ``initial_sizing``).
2. **Assigned agent** -- which agent executes this stage.
3. **Max steps** -- upper bound on LLM calls within the stage.
4. **Input mapping** -- how prior-stage outputs feed into this stage.

.. code-block:: text

   Pipeline Template (example):

   Stage 1: initial_sizing
     Agent: sizing_agent
     Max steps: 10
     Inputs: user requirements
         |
         v
   Stage 2: aerodynamic_analysis
     Agent: aero_agent
     Max steps: 15
     Inputs: Stage 1 outputs
         |
         v
   Stage 3: propulsion_optimization
     Agent: propulsion_agent
     Max steps: 15
     Inputs: Stage 1 + Stage 2 outputs
         |
         v
   Stage 4: final_evaluation
     Agent: eval_agent
     Max steps: 10
     Inputs: All prior stage outputs


Agent Lifecycle and MCP Interaction
====================================

Each agent follows a consistent lifecycle regardless of the strategy or
handler it operates under:

.. code-block:: text

   +--------------+     +----------------+     +---------------+
   | 1. Receive   | --> | 2. Reason      | --> | 3. Select     |
   |    task       |     |    (LLM call)  |     |    MCP tool   |
   +--------------+     +----------------+     +---------------+
                                                      |
                                                      v
                                               +---------------+
                                               | 4. Call MCP   |
                                               |    server     |
                                               +---------------+
                                                      |
                                                      v
                                               +---------------+
                                               | 5. Process    |
                                               |    result     |
                                               +---------------+
                                                      |
                                                      v
                                               +---------------+
                                               | 6. Report     |
                                               |    back       |
                                               +---------------+

**Step 1 -- Receive task**: The handler assigns a task description and any
relevant context (prior results, feedback, blackboard state).

**Step 2 -- Reason**: The agent's LLM processes the task and context,
producing a chain-of-thought plan.

**Step 3 -- Select MCP tool**: Based on its reasoning, the agent selects one
or more tools exposed by the Aviary MCP server.

**Step 4 -- Call MCP server**: The agent invokes the tool via HTTP on
``localhost:8600``. The MCP connector (``src.tools.mcp_connector``) handles
serialization, retries, and error mapping.

**Step 5 -- Process result**: The agent interprets the tool's response
(e.g., new metric values, parameter sensitivities) and updates its internal
state.

**Step 6 -- Report back**: The agent returns its results to the handler,
which decides the next action (iterate, advance stage, transition state, or
terminate).

The agent factory (``src.agents.agent_factory``) constructs agents with the
correct LLM model, system prompt, and tool set. The agent registry
(``src.agents.agent_registry``) tracks all active agents for the coordinator.
