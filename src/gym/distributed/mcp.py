"""Minimal MCP stdio server exposing zen lab tools.

Lets hanzo-dev / claude desktop / any MCP client read lab status and run
self-tests without learning the HTTP API. Speaks MCP over stdin/stdout
using plain JSON-RPC — no SDK dependency.

Register in hanzo-dev / claude desktop config:

    [mcp-servers.zen-lab]
    command = "python3"
    args = ["-m", "gym.distributed.mcp"]
    env = { ZEN_LAB_COORDINATOR = "http://spark.local:8443" }

Tools exposed:
    lab_status           — current metrics (topology + recent rounds + losses)
    lab_topology         — static lab layout
    lab_self_test        — probe local backend + optional coordinator handshake
    lab_round            — drill into a specific round
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from typing import Any

COORDINATOR = os.environ.get("ZEN_LAB_COORDINATOR", "http://spark.local:8443")


# ── tool registry ───────────────────────────────────────────────────────────


def _fetch(path: str) -> Any:
    with urllib.request.urlopen(COORDINATOR.rstrip("/") + path, timeout=10) as r:
        return json.loads(r.read())


TOOLS = [
    {
        "name": "lab_status",
        "description": "Return zen lab status: topology, last 50 rounds, current losses.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "lab_topology",
        "description": "Return the declared lab topology: workers, data weights, expert pins.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "lab_self_test",
        "description": "Run the self-test on this box: probe backend + verify canonical round-trip.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "lab_round",
        "description": "Return details of a specific federation round by ID.",
        "inputSchema": {
            "type": "object",
            "properties": {"round_id": {"type": "integer"}},
            "required": ["round_id"],
        },
    },
]


def call_tool(name: str, args: dict) -> dict:
    if name == "lab_status":
        return _fetch("/v1/metrics")
    if name == "lab_topology":
        return _fetch("/v1/topology")
    if name == "lab_self_test":
        from .selftest import selftest
        # Capture stdout
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = selftest(COORDINATOR)
        return {"exit_code": rc, "output": buf.getvalue()}
    if name == "lab_round":
        rid = int(args["round_id"])
        metrics = _fetch("/v1/metrics")
        for r in metrics.get("rounds", []):
            if r["round_id"] == rid:
                return r
        return {"error": f"round {rid} not found"}
    raise ValueError(f"unknown tool: {name}")


# ── JSON-RPC over stdio ─────────────────────────────────────────────────────


def _send(msg: dict) -> None:
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def _handle(msg: dict) -> dict | None:
    method = msg.get("method")
    mid = msg.get("id")
    params = msg.get("params") or {}

    if method == "initialize":
        return {
            "jsonrpc": "2.0", "id": mid,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "zen-lab", "version": "0.1.0"},
            },
        }

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}}

    if method == "tools/call":
        try:
            result = call_tool(params["name"], params.get("arguments") or {})
            return {
                "jsonrpc": "2.0", "id": mid,
                "result": {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]},
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0", "id": mid,
                "error": {"code": -32000, "message": str(e)},
            }

    # notifications/initialized and other notifications don't expect a response
    if mid is None:
        return None

    return {"jsonrpc": "2.0", "id": mid,
            "error": {"code": -32601, "message": f"method not found: {method}"}}


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        reply = _handle(msg)
        if reply is not None:
            _send(reply)
    return 0


if __name__ == "__main__":
    sys.exit(main())
