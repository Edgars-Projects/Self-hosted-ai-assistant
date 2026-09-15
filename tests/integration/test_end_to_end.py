"""End-to-end: real handlers, agent loop, tools and database; fake network edges."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("telegram")
pytest.importorskip("apscheduler")

from . import fake_ollama  # noqa: E402

SCENARIO = os.path.join(os.path.dirname(__file__), "scenario.py")


@pytest.fixture(scope="module")
def transcript(tmp_path_factory):
    server, url = fake_ollama.start()
    tmp = tmp_path_factory.mktemp("e2e")
    out = tmp / "sent.json"
    messages = [
        "/start",
        "hello",
        "STRANGER hello",
        'TOOL remember {"key": "colour", "value": "blue"}',
        'TOOL recall {"key": "colour"}',
        'TOOL forget {"key": "colour"}',
        'TOOL recall {"key": "colour"}',
        'TOOL run_python {"code": "print(6*7)"}',
        'TOOL make_pdf {"filename": "report", "title": "T", "content": "# H\\n- item"}',
        'TOOL make_spreadsheet {"filename": "sheet", "sheets": {"S": [["a"], [1]]}}',
        "TOOL email_inbox {}",
        'TOOL crypto {"action": "help"}',
        "/stats",
        "/reset",
    ]
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    subprocess.run(
        [sys.executable, SCENARIO, url, str(tmp / "state"), str(out), *messages],
        check=True,
        timeout=240,
        env=env,
        capture_output=True,
        text=True,
    )
    server.shutdown()
    return json.loads(out.read_text())


def replies_to(transcript, prompt):
    """Final bot messages sent after ``prompt`` and before the next user message."""
    start = next(i for i, m in enumerate(transcript) if m["api"] == "USER" and m["text"] == prompt)
    out = []
    for m in transcript[start + 1 :]:
        if m["api"] == "USER":
            break
        out.append(m["text"] or "")
    return out


def test_bot_answers_the_owner(transcript):
    assert "Hi! How can I help?" in replies_to(transcript, "hello")


def test_strangers_are_ignored(transcript):
    assert not [m for m in transcript if m["api"] != "USER" and m["chat"] == 7]


def test_memory_round_trip(transcript):
    assert "Result from tool: saved 'colour'" in replies_to(
        transcript, 'TOOL remember {"key": "colour", "value": "blue"}'
    )
    recalls = [
        r
        for r in transcript
        if r["api"] == "sendMessage" and (r["text"] or "").startswith("Result from tool:")
    ]
    texts = [r["text"] for r in recalls]
    assert "Result from tool: blue" in texts
    assert "Result from tool: nothing saved under 'colour'" in texts


def test_python_tool_executes(transcript):
    assert "Result from tool: 42" in replies_to(
        transcript, 'TOOL run_python {"code": "print(6*7)"}'
    )


def test_documents_are_written(transcript):
    texts = " ".join(m["text"] or "" for m in transcript)
    assert "report.pdf" in texts and "sheet.xlsx" in texts


def test_tools_fail_gracefully_without_network_or_setup(transcript):
    assert any("no mailbox yet" in t for t in replies_to(transcript, "TOOL email_inbox {}"))
    assert any(
        "actions: price" in t for t in replies_to(transcript, 'TOOL crypto {"action": "help"}')
    )


def test_commands_respond(transcript):
    assert any("turns" in t for t in replies_to(transcript, "/stats"))
    assert "conversation cleared (memory kept)" in replies_to(transcript, "/reset")
