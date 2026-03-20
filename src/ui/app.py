"""Main Streamlit dashboard for the multi-agent system.

Launch with: streamlit run src/ui/app.py

Reads from InstrumentationLogger output (JSON files in logs/).
Can also launch new runs directly from the UI.
"""

import os
import sys
import threading
import time

# Ensure project root is on the path when launched via `streamlit run`.
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st  # noqa: E402

from src.config.loader import load_config  # noqa: E402
from src.coordination.coordinator import Coordinator  # noqa: E402
from src.logging.logger import InstrumentationLogger  # noqa: E402
from src.ui.batch_components import render_batch_results  # noqa: E402
from src.ui.components import render_agent_message, render_metrics_panel, render_strategy_panel  # noqa: E402
from src.ui.state import RunState, list_run_files, load_run_file  # noqa: E402


def _init_session_state():
    """Initialize Streamlit session state defaults."""
    if "run_state" not in st.session_state:
        st.session_state.run_state = RunState()
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    if "logger" not in st.session_state:
        st.session_state.logger = None
    if "run_thread" not in st.session_state:
        st.session_state.run_thread = None
    if "stop_requested" not in st.session_state:
        st.session_state.stop_requested = False


def _run_task(task: str, strategy: str, config_path: str):
    """Execute a coordination run in the background."""
    try:
        config = load_config(config_path)
        logger = InstrumentationLogger(
            {"logging": {"output_dir": config.logging.output_dir}},
        )
        st.session_state.logger = logger
        st.session_state.is_running = True

        coordinator = Coordinator.from_config(
            config,
            logger=logger,
            strategy_override=strategy,
        )

        result = coordinator.run(task)
        logger.export_json()

        st.session_state.run_state = RunState(
            history=[_msg_to_dict(m) for m in result.history],
            metrics=result.metrics,
            task=task,
            strategy=strategy,
        )
    except Exception as e:
        st.session_state.run_state = RunState(
            error=str(e),
            task=task,
            strategy=strategy,
        )
    finally:
        st.session_state.is_running = False


def _msg_to_dict(msg) -> dict:
    """Convert an AgentMessage to a plain dict for display."""
    import dataclasses

    if dataclasses.is_dataclass(msg):
        return dataclasses.asdict(msg)
    return dict(msg) if hasattr(msg, "__iter__") else {"content": str(msg)}


def main():
    st.set_page_config(
        page_title="Multi-Agent Dashboard",
        page_icon="🤖",
        layout="wide",
    )
    _init_session_state()

    st.title("Multi-Agent System Dashboard")

    # ---- Sidebar: Controls ---------------------------------------------------
    with st.sidebar:
        st.header("Controls")

        config_path = st.text_input("Config path", value="config/default.yaml")
        strategy = st.selectbox(
            "Coordination Strategy",
            ["sequential", "orchestrated", "networked", "graph_routed"],
        )
        task = st.text_area("Task", placeholder="Enter a task for the agents...")

        col_run, col_stop = st.columns(2)
        with col_run:
            run_clicked = st.button(
                "Run",
                disabled=st.session_state.is_running or not task,
                use_container_width=True,
            )
        with col_stop:
            stop_clicked = st.button(
                "Stop",
                disabled=not st.session_state.is_running,
                use_container_width=True,
            )

        if run_clicked and task:
            st.session_state.stop_requested = False
            thread = threading.Thread(
                target=_run_task,
                args=(task, strategy, config_path),
                daemon=True,
            )
            thread.start()
            st.session_state.run_thread = thread
            st.rerun()

        if stop_clicked:
            st.session_state.stop_requested = True

        # Status indicator
        if st.session_state.is_running:
            st.info("Running...")
        elif st.session_state.run_state.error:
            st.error(f"Error: {st.session_state.run_state.error}")

        # Load from file
        st.markdown("---")
        st.header("Load Previous Run")
        run_files = list_run_files()
        if run_files:
            selected_file = st.selectbox(
                "Select run file",
                run_files,
                format_func=lambda x: x.split("/")[-1],
            )
            if st.button("Load", use_container_width=True):
                st.session_state.run_state = load_run_file(selected_file)
                st.rerun()
        else:
            st.caption("No previous runs found in logs/.")

    # ---- Main content --------------------------------------------------------
    state = st.session_state.run_state

    # Show live data from logger if running
    if st.session_state.is_running and st.session_state.logger:
        logger = st.session_state.logger
        msgs = logger.get_messages()
        state = RunState(
            history=[_msg_to_dict(m) for m in msgs],
            metrics=logger.compute_metrics() if msgs else {},
            is_running=True,
            task=st.session_state.run_state.task,
            strategy=st.session_state.run_state.strategy,
        )

    # Four-panel layout
    tab_feed, tab_strategy, tab_metrics, tab_batch = st.tabs(
        [
            "Agent Activity Feed",
            "Strategy Visualization",
            "Metrics Dashboard",
            "Batch Results",
        ]
    )

    with tab_feed:
        if not state.history:
            st.info("No activity yet. Enter a task and click Run, or load a previous run.")
        else:
            for i, msg in enumerate(state.history):
                render_agent_message(msg, i)

    with tab_strategy:
        render_strategy_panel(state.history, state.strategy)

    with tab_metrics:
        render_metrics_panel(state.metrics)

    with tab_batch:
        render_batch_results()

    # Auto-refresh while running
    if st.session_state.is_running:
        time.sleep(1)
        st.rerun()


if __name__ == "__main__":
    main()
