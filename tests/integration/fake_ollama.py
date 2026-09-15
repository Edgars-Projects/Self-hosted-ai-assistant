"""A deterministic stand-in for the Ollama HTTP API.

The "model" follows a tiny script so tests can drive real tool calls:

* ``TOOL <name> <json-args>`` in a user message -> calls that tool
* a tool result -> replies ``Result from tool: <output>``
* anything else -> a canned reply
"""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

MODELS = ["qwen3:14b", "qwen3:30b-a3b", "qwen3:4b", "nomic-embed-text:latest"]


def scripted_reply(payload: dict[str, Any]) -> dict[str, Any]:
    msgs = payload.get("messages") or []
    last = msgs[-1] if msgs else {}
    tools = {t["function"]["name"] for t in payload.get("tools") or []}
    content = str(last.get("content", ""))
    if last.get("role") == "tool":
        return {"role": "assistant", "content": "Result from tool: " + content.strip()[:200]}
    m = re.search(r"TOOL (\w+) (\{.*\})", content)
    if m and m.group(1) in tools and last.get("role") == "user":
        call = {"function": {"name": m.group(1), "arguments": json.loads(m.group(2))}}
        return {"role": "assistant", "content": "", "tool_calls": [call]}
    if "hello" in content.lower():
        return {"role": "assistant", "content": "Hi! How can I help?"}
    return {"role": "assistant", "content": "OK"}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:  # keep test output quiet
        pass

    def _json(self, obj: Any) -> None:
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.startswith("/api/tags"):
            return self._json({"models": [{"name": n} for n in MODELS]})
        self._json({"models": []} if self.path.startswith("/api/ps") else {"version": "fake"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/api/embed":
            return self._json({"embeddings": [[0.1] * 16]})
        if self.path != "/api/chat":
            return self._json({})
        msg = scripted_reply(payload)
        if not payload.get("stream", True):
            return self._json({"message": msg, "done": True})
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        for chunk in (
            {"message": msg, "done": False},
            {"message": {"role": "assistant", "content": ""}, "done": True},
        ):
            self.wfile.write((json.dumps(chunk) + "\n").encode())


def start() -> tuple[ThreadingHTTPServer, str]:
    """Start the server on a free port; returns the server and its base URL."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"
