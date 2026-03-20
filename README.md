# MAS-Aviary

**Multi-Agent Coordination for Aircraft Design Optimization**

MAS-Aviary is a multi-agent LLM framework for aircraft design optimization
using NASA's OpenMDAO/Aviary simulation. Autonomous agents powered by Qwen3-8B
coordinate via the Model Context Protocol (MCP) to optimize aircraft parameters
such as fuel burn, gross weight, and wing mass. The framework systematically
explores eight distinct coordination strategies spanning three organizational
structures and three task-handling paradigms, enabling rigorous comparison of
multi-agent approaches on a real-world engineering optimization problem.

| Repository | Description |
|------------|-------------|
| **MAS-Aviary** (this repo) | Multi-agent LLM framework (client) |
| [Aviary](https://github.com/Jezemba/Aviary) | MCP server wrapping NASA OpenMDAO/Aviary (backend) |

---

## Architecture

```
User
  |
  v
Coordinator ──── selects strategy and dispatches work
  |
  v
Strategy ──────── Sequential | Orchestrated | Networked
  |
  v
Handler ────────── Iterative Feedback | Staged Pipeline | Graph-Routed
  |
  v
Agent(s) ───────── Qwen3-8B via smolagents ToolCallingAgent
  |
  v
MCP Server ─────── Streamable HTTP, tool-based interface
  |
  v
Aviary ──────────── NASA OpenMDAO aircraft design simulation
```

---

## Coordination Strategies

The framework evaluates **eight** combinations of three organizational
structures and three task-handling paradigms:

| # | Structure | Handler | Combination Name |
|---|-----------|---------|------------------|
| 1 | Sequential | Iterative Feedback | `aviary_sequential_iterative_feedback` |
| 2 | Sequential | Staged Pipeline | `aviary_sequential_staged_pipeline` |
| 3 | Sequential | Graph-Routed | `aviary_sequential_graph_routed` |
| 4 | Orchestrated | Iterative Feedback | `aviary_orchestrated_iterative_feedback` |
| 5 | Orchestrated | Staged Pipeline | `aviary_orchestrated_staged_pipeline` |
| 6 | Orchestrated | Graph-Routed | `aviary_orchestrated_graph_routed` |
| 7 | Networked | Iterative Feedback | `aviary_networked_iterative_feedback` |
| 8 | Networked | Graph-Routed | `aviary_networked_graph_routed` |

> **Note:** Networked + Staged Pipeline is **not a valid combination** and is
> excluded from the experiment matrix. The Staged Pipeline handler assumes a
> fixed sequence of pipeline stages with deterministic agent-to-stage assignment,
> which conflicts with the Networked strategy's peer-based role rotation and
> blackboard claiming model. Specifically, the Staged Pipeline requires each
> stage to be assigned to a single named agent in a predetermined order, while
> the Networked strategy dynamically assigns roles to whichever peer agent
> claims the next available task from the blackboard. Adapting the Networked
> strategy to enforce the rigid stage ordering required by the Staged Pipeline
> handler would negate the collaborative, decentralized properties that define
> the Networked approach. The combination is defined in the code
> (`ALL_COMBINATIONS` in `src/runners/batch_runner.py`) for completeness but is
> filtered out of all experiment runs.

---

## Prerequisites

- **Python**: 3.12 or later
- **GPU**: NVIDIA GPU with CUDA 12.x (for Qwen3-8B inference)
- **OS**: Linux (tested on Ubuntu 22.04)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Jezemba/MAS-Aviary.git
cd MAS-Aviary
```

### 2. Create a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Install PyTorch with CUDA support

Install the version matching your CUDA driver. For CUDA 12.8:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

For CUDA 12.4:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Check your CUDA version with `nvidia-smi`.

### 4. Install MAS-Aviary and dependencies

```bash
pip install -e ".[dev,ui,tracking]"
```

This installs:

| Group | Packages |
|-------|----------|
| **Core** | torch, transformers, smolagents, scikit-learn, matplotlib, pyyaml, requests |
| **dev** | pytest, pytest-cov, ruff, sphinx, sphinx-rtd-theme, sphinx-autodoc-typehints |
| **ui** | streamlit, altair |
| **tracking** | wandb |

### 5. Verify the installation

```bash
# Lint check
ruff check src/ tests/ scripts/

# Run unit tests (no GPU or MCP server needed)
pytest -m "not slow and not gpu and not mcp" -v

# Check GPU access
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}')"
```

---

## MCP Server Configuration

The MCP server URL is configurable at three levels (highest priority wins):

1. **CLI flag**: `--mcp-url http://your-host:port/mcp`
2. **Environment variable**: `export MAS_AVIARY_MCP_URL=http://your-host:port/mcp`
3. **YAML config**: edit `mcp.servers[0].url` in `config/aviary_run.yaml`

The default config ships with `http://127.0.0.1:8600/mcp`. Override it for
remote servers, non-standard ports, or containerized deployments:

```bash
# Environment variable (persists for the shell session)
export MAS_AVIARY_MCP_URL=http://192.168.1.50:9000/mcp

# Or pass directly to the runner
python scripts/stat_batch_runner.py --mcp-url http://192.168.1.50:9000/mcp --repeats 1
```

Other environment variable overrides:

| Variable | Overrides | Example |
|----------|-----------|---------|
| `MAS_AVIARY_MCP_URL` | MCP server URL | `http://10.0.0.5:8600/mcp` |
| `MAS_AVIARY_MCP_MODE` | MCP mode (`mock` or `real`) | `mock` |
| `MAS_AVIARY_MODEL_ID` | LLM model | `Qwen/Qwen3-4B` |
| `MAS_AVIARY_API_BASE` | vLLM API endpoint | `http://localhost:8000/v1` |

---

## Quickstart

### 1. Start the Aviary MCP server

The MCP server ([Aviary repo](https://github.com/Jezemba/Aviary)) provides
tool-based access to NASA OpenMDAO/Aviary simulations. It must be running
before launching agents:

```bash
# Clone and install the Aviary MCP server (requires a separate conda env)
git clone https://github.com/Jezemba/Aviary.git
cd Aviary
conda create -n aviary python=3.11 -y
conda activate aviary
pip install -r requirements.txt

# Start the server (default: http://localhost:8600/mcp)
python server/aviary_mcp_server.py
```

The server exposes 9 MCP tools: `get_design_space`, `create_session`,
`set_aircraft_parameters`, `configure_mission`, `validate_parameters`,
`run_simulation`, `get_results`, `get_trajectory`, `check_constraints`.

See the [Aviary README](https://github.com/Jezemba/Aviary#readme) for full
server documentation.

### 2. Run a single optimization

```bash
python scripts/stat_batch_runner.py \
    --repeats 1 \
    --combinations aviary_sequential_iterative_feedback
```

### 3. Run all eight combinations

```bash
python scripts/stat_batch_runner.py --repeats 1
```

### 4. Run the full statistical batch (30 repeats)

```bash
python scripts/stat_batch_runner.py --repeats 30
```

The runner supports crash recovery — re-run the same command to resume from the checkpoint.

---

## Dashboard

Launch the Streamlit dashboard to visualize optimization results:

```bash
pip install -e ".[ui]"  # if not already installed
streamlit run src/ui/app.py
```

---

## Testing

Run the unit test suite (no GPU or MCP server required):

```bash
pytest -m "not slow and not gpu and not mcp" -v
```

Run with coverage:

```bash
pytest -m "not slow and not gpu and not mcp" --cov=src --cov-report=term-missing -v
```

Run integration tests (requires GPU + MCP server):

```bash
pytest -m "slow or gpu" -v --timeout=600
```

---

## Documentation

Full documentation is hosted at **https://jezemba.github.io/MAS-Aviary/**
(auto-deployed on every push to main).

To build locally:

```bash
cd docs
make html
# Open _build/html/index.html in your browser
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Citation

If you use MAS-Aviary in your research, please cite:

```bibtex
@article{ezemba2026masaviary,
  title={MAS-Aviary: Multi-Agent Coordination for Aircraft Design Optimization},
  author={Ezemba, Jessica},
  year={2026}
}
```
