# MAS-Aviary

**Multi-Agent Coordination for Aircraft Design Optimization**

MAS-Aviary is a multi-agent LLM framework for aircraft design optimization
using NASA's OpenMDAO/Aviary simulation. Autonomous agents powered by Qwen3-8B
coordinate via the Model Context Protocol (MCP) to optimize aircraft parameters
such as fuel burn, gross weight, and wing mass. The framework systematically
explores nine distinct coordination strategies spanning three organizational
structures and three task-handling paradigms, enabling rigorous comparison of
multi-agent approaches on a real-world engineering optimization problem.

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

The framework evaluates all nine combinations of three organizational
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
| 8 | Networked | Staged Pipeline | `aviary_networked_staged_pipeline` |
| 9 | Networked | Graph-Routed | `aviary_networked_graph_routed` |

---

## Installation

```bash
git clone https://github.com/Jezemba/MAS-Aviary.git
cd MAS-Aviary
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

---

## Quickstart

Run a single optimization with one of the nine combinations:

```bash
python scripts/stat_batch_runner.py --repeats 1 --combinations aviary_sequential_iterative_feedback
```

---

## Dashboard

Launch the Streamlit dashboard to visualize optimization results:

```bash
streamlit run src/ui/app.py
```

---

## Key Dependencies

| Package | Purpose |
|---------|---------|
| torch | Tensor computation and GPU acceleration |
| transformers | Qwen3-8B model loading and inference |
| smolagents | Agent framework with tool-calling support |
| scikit-learn | Statistical analysis of optimization results |
| matplotlib | Result plotting and visualization |
| streamlit | Interactive dashboard |
| pyyaml | Configuration file parsing |
| requests | MCP server communication |

---

## Testing

Run the unit test suite (excludes slow, GPU, and MCP-dependent tests):

```bash
pytest -m "not slow and not gpu and not mcp" -v
```

---

## Documentation

Full Sphinx documentation is available in the [`docs/`](docs/) directory.

To build locally:

```bash
cd docs
make html
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
