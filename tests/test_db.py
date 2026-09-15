from assistant import db

EXPECTED_TABLES = {
    "facts",
    "turns",
    "vectors",
    "jobs",
    "rules",
    "examples",
    "tool_stats",
    "feedback",
    "model_scores",
    "model_probe",
    "goals",
    "goal_log",
    "settings",
}


def test_open_db_creates_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "state" / "bot.db"))
    con = db.open_db()
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert EXPECTED_TABLES <= tables
    assert con.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"


def test_open_db_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "bot.db"))
    db.open_db().close()
    db.open_db().close()
