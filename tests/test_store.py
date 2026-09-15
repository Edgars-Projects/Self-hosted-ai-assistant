from assistant import db
from assistant.db import Store


def make_store(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "bot.db"))
    return Store(db.open_db())


def test_settings_round_trip(tmp_path, monkeypatch):
    store = make_store(tmp_path, monkeypatch)
    assert store.setting(1, "voice") is None
    assert store.setting(1, "voice", "0") == "0"
    store.set_setting(1, "voice", 1)
    assert store.setting(1, "voice") == "1"
    assert store.setting(2, "voice") is None  # settings are per chat


def test_query_fetch_modes(tmp_path, monkeypatch):
    store = make_store(tmp_path, monkeypatch)
    store.q("INSERT INTO facts (chat_id,key,value) VALUES (?,?,?)", (1, "a", "x"))
    store.q("INSERT INTO facts (chat_id,key,value) VALUES (?,?,?)", (1, "b", "y"))
    assert store.q("SELECT COUNT(*) FROM facts", (), "one") == (2,)
    assert len(store.q("SELECT * FROM facts", (), "all")) == 2
    assert store.q("DELETE FROM facts") is None
