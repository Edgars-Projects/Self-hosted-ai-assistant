"""SQLite schema and connection setup. All state lives in one file."""

from __future__ import annotations

import os
import sqlite3
import threading
from typing import Any, Literal, Sequence

from .config import DB_PATH

Fetch = Literal["one", "all"] | None

SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
  chat_id INTEGER, key TEXT, value TEXT,
  ts TEXT DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (chat_id, key));

CREATE TABLE IF NOT EXISTS turns (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  chat_id INTEGER, role TEXT, content TEXT,
  ts TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE INDEX IF NOT EXISTS turns_chat ON turns(chat_id, id);

CREATE TABLE IF NOT EXISTS vectors (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  chat_id INTEGER, kind TEXT, source TEXT, text TEXT, vec BLOB,
  ts TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE INDEX IF NOT EXISTS vec_chat ON vectors(chat_id, kind);

CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY, chat_id INTEGER, kind TEXT,
  spec TEXT, payload TEXT, state TEXT,
  ts TEXT DEFAULT CURRENT_TIMESTAMP);

CREATE TABLE IF NOT EXISTS rules (
  id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id INTEGER, text TEXT,
  source TEXT, score REAL DEFAULT 0, active INTEGER DEFAULT 1,
  ts TEXT DEFAULT CURRENT_TIMESTAMP);

CREATE TABLE IF NOT EXISTS examples (
  id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id INTEGER, task TEXT,
  approach TEXT, vec BLOB, score REAL DEFAULT 1,
  ts TEXT DEFAULT CURRENT_TIMESTAMP);

CREATE TABLE IF NOT EXISTS tool_stats (
  name TEXT PRIMARY KEY, calls INTEGER DEFAULT 0, fails INTEGER DEFAULT 0,
  total_ms REAL DEFAULT 0, last_error TEXT);

CREATE TABLE IF NOT EXISTS feedback (
  id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id INTEGER, verdict TEXT,
  note TEXT, context TEXT, ts TEXT DEFAULT CURRENT_TIMESTAMP);

CREATE TABLE IF NOT EXISTS model_scores (
  model TEXT, kind TEXT, uses INTEGER DEFAULT 0, wins REAL DEFAULT 0,
  losses REAL DEFAULT 0, ms REAL DEFAULT 0, last_used TEXT,
  last_note TEXT, PRIMARY KEY (model, kind));

CREATE TABLE IF NOT EXISTS model_probe (
  id INTEGER PRIMARY KEY AUTOINCREMENT, question TEXT, kind TEXT,
  known_answer TEXT, ts TEXT DEFAULT CURRENT_TIMESTAMP);

CREATE TABLE IF NOT EXISTS goals (
  id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id INTEGER, text TEXT,
  kind TEXT DEFAULT 'inform', cadence_min INTEGER DEFAULT 1440,
  success TEXT, active INTEGER DEFAULT 1, runs INTEGER DEFAULT 0,
  wins INTEGER DEFAULT 0, last_run TEXT, last_result TEXT,
  ts TEXT DEFAULT CURRENT_TIMESTAMP);

CREATE TABLE IF NOT EXISTS goal_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT, goal_id INTEGER, score INTEGER,
  verdict TEXT, summary TEXT, delivered INTEGER DEFAULT 0,
  ts TEXT DEFAULT CURRENT_TIMESTAMP);

CREATE TABLE IF NOT EXISTS settings (
  chat_id INTEGER, key TEXT, value TEXT, PRIMARY KEY (chat_id, key));
"""


def open_db() -> sqlite3.Connection:
    """Open (creating if needed) the state database in WAL mode."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    con.commit()
    return con


class Store:
    """Thread-safe access to the state database.

    SQLite connections are shared across the bot's worker threads, so every
    statement runs under one lock and commits immediately.
    """

    def __init__(self, con: sqlite3.Connection, lock: threading.Lock | None = None) -> None:
        self.con = con
        self.lock = lock or threading.Lock()

    def q(self, sql: str, args: Sequence[Any] = (), fetch: Fetch = None) -> Any:
        """Run one statement. ``fetch`` is ``"one"``, ``"all"`` or ``None``."""
        with self.lock:
            cur = self.con.execute(sql, args)
            out = cur.fetchall() if fetch == "all" else (cur.fetchone() if fetch == "one" else None)
            self.con.commit()
            return out

    def setting(self, chat_id: int, key: str, default: str | None = None) -> str | None:
        """Read a per-chat setting."""
        r = self.q(
            "SELECT value FROM settings WHERE chat_id=? AND key=?",
            (chat_id, key),
            "one",
        )
        return r[0] if r else default

    def set_setting(self, chat_id: int, key: str, value: object) -> None:
        """Write a per-chat setting; values are stored as strings."""
        self.q("INSERT OR REPLACE INTO settings VALUES (?,?,?)", (chat_id, key, str(value)))
