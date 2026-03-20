==========
Quickstart
==========

This guide walks you through setting up MAS-Aviary, running your first
optimization, and viewing results.

Prerequisites
=============

- Python 3.12+
- A running Aviary MCP server (port 8600)
- Access to a supported LLM provider

Installation
============

1. **Clone the repository**

   .. code-block:: bash

      git clone <repo-url> MAS-Aviary
      cd MAS-Aviary

2. **Create and activate a virtual environment**

   .. code-block:: bash

      python -m venv .venv
      source .venv/bin/activate   # Linux / macOS
      .venv\Scripts\activate      # Windows

3. **Install dependencies**

   .. code-block:: bash

      pip install -e .

Starting the Aviary MCP Server
==============================

The Aviary MCP server must be running before you launch any optimization. It
exposes the aircraft design tools that agents interact with.

.. code-block:: bash

   # Start the MCP server on port 8600
   python -m aviary_mcp_server --port 8600

Verify the server is reachable:

.. code-block:: bash

   curl http://localhost:8600/health

Running a Single Optimization
==============================

The ``stat_batch_runner.py`` script executes one or more strategy combinations.
To run a single optimization with the Sequential + Iterative Feedback strategy:

.. code-block:: bash

   python scripts/stat_batch_runner.py \
       --repeats 1 \
       --combinations aviary_sequential_iterative_feedback

This will:

1. Load the configuration from ``config/aviary_run.yaml`` and the
   strategy-specific YAML files.
2. Instantiate the coordinator, strategy, and handler.
3. Spawn agents that connect to the Aviary MCP server.
4. Execute the optimization loop.
5. Write results to the ``results/`` directory.

Understanding the Output
========================

Results are written as JSON files under ``results/``. Each file contains:

.. code-block:: json

   {
     "combination": "aviary_sequential_iterative_feedback",
     "run_id": "20260319_143022_abc123",
     "status": "completed",
     "metrics": {
       "fuel_burned_kg": 12345.6,
       "gross_takeoff_weight_kg": 78900.1,
       "wing_mass_kg": 4567.8
     },
     "agent_logs": [ "..." ],
     "duration_seconds": 120.5
   }

Key fields:

- **combination** -- The strategy combination that was executed.
- **metrics** -- The final aircraft design metrics produced by Aviary.
- **agent_logs** -- Detailed logs of each agent's tool calls and reasoning.
- **duration_seconds** -- Wall-clock time for the full run.

Running the Dashboard
=====================

MAS-Aviary includes a Streamlit dashboard for visualizing results across
strategy combinations and repeat runs.

.. code-block:: bash

   streamlit run src/ui/app.py

The dashboard opens in your browser and provides:

- **Strategy Comparison** -- Side-by-side metrics for all nine combinations.
- **Run History** -- Time-series view of repeated runs.
- **Agent Traces** -- Step-by-step replay of agent reasoning and tool calls.
- **Metric Distributions** -- Box plots and histograms for fuel, GTOW, and
  wing mass across repeats.

Next Steps
==========

- Read the :doc:`architecture` page to understand how strategies and handlers
  compose.
- See :doc:`configuration` for a full reference of every YAML config file.
- Browse the :doc:`api/index` for module-level documentation.
