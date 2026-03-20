"""CLI entry point — run a multi-agent coordination task from the command line.

Usage:
    python main.py "Your task here"
    python main.py --strategy sequential "Calculate 15 times 7"
    python main.py --config config/custom.yaml "Your task here"
"""

import argparse
import json
import sys

from src.config.loader import load_config
from src.coordination.coordinator import Coordinator
from src.logging.logger import InstrumentationLogger


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a multi-agent coordination task.",
    )
    parser.add_argument(
        "task",
        help="The task for the agents to solve.",
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to the main YAML config file (default: config/default.yaml).",
    )
    parser.add_argument(
        "--strategy",
        default=None,
        choices=["sequential", "graph_routed", "orchestrated", "networked"],
        help="Override the coordination strategy from config.",
    )
    parser.add_argument(
        "--export",
        default=None,
        metavar="PATH",
        help="Export run results to a JSON file at PATH.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main entry point. Returns 0 on success, 1 on error."""
    args = parse_args(argv)

    # Load config
    config = load_config(args.config)
    logger = InstrumentationLogger({"logging": {"output_dir": config.logging.output_dir}})

    # Build coordinator
    coordinator = Coordinator.from_config(
        config,
        logger=logger,
        strategy_override=args.strategy,
    )

    # Run
    print(f"Task: {args.task}")
    print(f"Strategy: {args.strategy or '(from config)'}")
    print("-" * 60)

    result = coordinator.run(args.task)

    # Display results
    print("-" * 60)
    print(f"Final output:\n{result.final_output}")
    print("-" * 60)

    if result.metrics:
        print("Metrics:")
        for key, value in result.metrics.items():
            print(f"  {key}: {value}")

    # Export if requested
    if args.export:
        from src.logging.exporter import export_run

        export_run(result.history, result.metrics, args.export)
        print(f"\nResults exported to {args.export}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
