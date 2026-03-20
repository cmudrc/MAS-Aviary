"""Generate a list of pre-validated parameter sets for Aviary runs.

Generates random parameter combinations, creates MCP sessions, validates
each via validate_parameters, and keeps only those that pass validation.
Outputs a JSON file that stat_batch_runner can consume.

Connects to MCP directly via HTTP (no smolagents/GPU needed), so it can
run concurrently with active agent runs.

Usage:
    .venv/bin/python scripts/generate_valid_params.py \
        --count 30 --output config/valid_param_sets.json

    # Dry run — just show how many attempts it takes:
    .venv/bin/python scripts/generate_valid_params.py \
        --count 5 --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import requests

# ---------------------------------------------------------------------------
# Parameter ranges (same as stat_batch_runner)
# ---------------------------------------------------------------------------

PARAM_RANGES = {
    "Aircraft.Wing.ASPECT_RATIO": (7.0, 14.0),
    "Aircraft.Wing.AREA": (100.0, 160.0),
    "Aircraft.Wing.SWEEP": (15.0, 40.0),
    "Aircraft.Wing.TAPER_RATIO": (0.15, 0.45),
    "Aircraft.Fuselage.LENGTH": (28.0, 50.0),
    "Aircraft.Fuselage.MAX_HEIGHT": (3.0, 5.5),
    "Aircraft.Fuselage.MAX_WIDTH": (3.0, 5.5),
    "Aircraft.Engine.SCALE_FACTOR": (0.8, 1.5),
}

MCP_URL = "http://127.0.0.1:8600/mcp"


def generate_random_params(seed: int) -> dict[str, float]:
    """Generate one random parameter set within valid ranges."""
    rng = np.random.default_rng(seed)
    params = {}
    for name, (lo, hi) in PARAM_RANGES.items():
        params[name] = round(float(rng.uniform(lo, hi)), 4)
    ar = params["Aircraft.Wing.ASPECT_RATIO"]
    area = params["Aircraft.Wing.AREA"]
    params["_derived_span"] = round(math.sqrt(ar * area), 4)
    return params


# ---------------------------------------------------------------------------
# Direct MCP HTTP client (no smolagents dependency)
# ---------------------------------------------------------------------------

class MCPClient:
    """Minimal MCP streamable-http client for tool calls."""

    def __init__(self, url: str = MCP_URL):
        self.url = url
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        self._req_id = 0
        self._initialize()

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def _initialize(self):
        """Perform MCP handshake."""
        resp = requests.post(self.url, json={
            "jsonrpc": "2.0", "id": self._next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "param-validator", "version": "1.0"},
            },
        }, headers=self.headers, timeout=10)
        resp.raise_for_status()
        mcp_session = resp.headers.get("mcp-session-id", "")
        if mcp_session:
            self.headers["mcp-session-id"] = mcp_session

        # Send initialized notification
        requests.post(self.url, json={
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        }, headers=self.headers, timeout=10)

    def call_tool(self, name: str, arguments: dict, timeout: int = 60) -> dict:
        """Call an MCP tool and return parsed result."""
        resp = requests.post(self.url, json={
            "jsonrpc": "2.0", "id": self._next_id(),
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }, headers=self.headers, timeout=timeout)
        resp.raise_for_status()

        # Parse SSE response — find the data: line with the result
        for line in resp.text.split("\n"):
            if line.startswith("data:"):
                data = json.loads(line[5:].strip())
                if "result" in data:
                    # Extract text content from MCP result
                    content = data["result"].get("content", [])
                    for item in content:
                        if item.get("type") == "text":
                            try:
                                return json.loads(item["text"])
                            except json.JSONDecodeError:
                                return {"raw": item["text"]}
                    return data["result"]
                if "error" in data:
                    return {"error": data["error"]}
        return {"error": "No result in response"}


def validate_params(
    client: MCPClient,
    params: dict[str, float],
    timeout: int = 30,
) -> tuple[bool, str, str | None]:
    """Create a session with params, configure mission, validate.

    Returns (valid, summary, session_id).
    """
    settable = {k: v for k, v in params.items() if not k.startswith("_")}

    # 1. Create session with initial parameters
    resp = client.call_tool("create_session", {
        "initial_parameters": settable,
    })
    session_id = resp.get("session_id")
    if not session_id:
        m = re.search(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            str(resp),
        )
        if m:
            session_id = m.group(0)
        else:
            raise RuntimeError(f"create_session failed: {resp}")

    # 2. Configure mission
    client.call_tool("configure_mission", {
        "session_id": session_id,
        "range_nmi": 1500,
        "num_passengers": 162,
        "cruise_mach": 0.785,
        "cruise_altitude_ft": 35000,
    })

    # 3. Validate
    vresp = client.call_tool("validate_parameters", {
        "session_id": session_id,
        "timeout_seconds": timeout,
    }, timeout=timeout + 10)

    valid = vresp.get("valid", False)
    summary = vresp.get("summary", str(vresp)[:200])

    return valid, summary, session_id


def main():
    parser = argparse.ArgumentParser(
        description="Generate pre-validated Aviary parameter sets",
    )
    parser.add_argument(
        "--count", type=int, default=30,
        help="Number of valid parameter sets to generate (default: 30)",
    )
    parser.add_argument(
        "--output", type=str, default="config/valid_param_sets.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--base-seed", type=int, default=42,
        help="Starting seed for random generation (default: 42)",
    )
    parser.add_argument(
        "--max-attempts", type=int, default=500,
        help="Max total attempts before giving up (default: 500)",
    )
    parser.add_argument(
        "--mcp-url", type=str, default=MCP_URL,
        help=f"MCP server URL (default: {MCP_URL})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print results but don't save",
    )
    args = parser.parse_args()

    print(f"Connecting to MCP at {args.mcp_url}...")
    client = MCPClient(args.mcp_url)
    print("  Connected.")

    valid_sets: list[dict] = []
    seeds_used: list[int] = []
    attempt = 0
    start_time = time.time()

    print(f"\nGenerating {args.count} valid parameter sets "
          f"(max {args.max_attempts} attempts)...\n")

    while len(valid_sets) < args.count and attempt < args.max_attempts:
        seed = args.base_seed + attempt
        params = generate_random_params(seed)
        attempt += 1

        try:
            valid, summary, session_id = validate_params(client, params)
        except Exception as e:
            print(f"  [{attempt:3d}] seed={seed} ERROR: {e}")
            # Re-initialize MCP connection on error
            try:
                client = MCPClient(args.mcp_url)
            except Exception:
                pass
            continue

        status = "VALID" if valid else "invalid"
        print(f"  [{attempt:3d}] seed={seed} {status}: {summary[:80]}")

        if valid:
            params["_seed"] = seed
            valid_sets.append(params)
            seeds_used.append(seed)

    elapsed = time.time() - start_time
    hit_rate = len(valid_sets) / attempt * 100 if attempt > 0 else 0

    print(f"\n{'='*60}")
    print(f"Results: {len(valid_sets)}/{args.count} valid sets found")
    print(f"Attempts: {attempt}, hit rate: {hit_rate:.1f}%")
    print(f"Time: {elapsed:.1f}s ({elapsed/attempt:.1f}s per attempt)")
    print(f"Seeds: {seeds_used}")

    if args.dry_run:
        print("\nDry run — not saving.")
        return

    if not valid_sets:
        print("\nNo valid sets found — nothing to save.")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "description": "Pre-validated Aviary parameter sets",
        "base_seed": args.base_seed,
        "attempts": attempt,
        "hit_rate_pct": round(hit_rate, 1),
        "generation_time_s": round(elapsed, 1),
        "count": len(valid_sets),
        "params": valid_sets,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
