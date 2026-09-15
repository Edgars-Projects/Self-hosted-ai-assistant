import json

import pytest

from assistant import db
from assistant.db import Store
from assistant.tools import crypto, documents, mail
from assistant.tools.clock import t_now
from assistant.tools.memory import MemoryTools


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "bot.db"))
    return Store(db.open_db())


# --- memory -------------------------------------------------------------
def test_memory_remember_recall_forget(store):
    embedded = []
    mem = MemoryTools(store, lambda *a: embedded.append(a))
    assert mem.remember(1, "colour", "blue") == "saved 'colour'"
    assert embedded == [(1, "fact", "colour", "colour: blue")]
    assert mem.recall(1, "colour") == "blue"
    assert mem.recall(1) == "colour: blue"
    assert mem.recall(2) == "(empty)"
    assert mem.forget(1, "colour") == "forgot 'colour'"
    assert mem.recall(1, "colour") == "nothing saved under 'colour'"


# --- documents ----------------------------------------------------------
def test_make_pdf_writes_a_pdf(tmp_path, monkeypatch):
    pytest.importorskip("reportlab")
    monkeypatch.setattr(documents, "DL_DIR", str(tmp_path))
    out = documents.t_make_pdf("../../report", "Title", "# H1\n## H2\n- bullet\n\nbody")
    path = tmp_path / "report.pdf"  # directory traversal is stripped
    assert str(path) in out
    assert path.read_bytes().startswith(b"%PDF")


def test_make_xlsx_accepts_json_and_keeps_formulas(tmp_path, monkeypatch):
    openpyxl = pytest.importorskip("openpyxl")
    monkeypatch.setattr(documents, "DL_DIR", str(tmp_path))
    sheets = json.dumps({"Sales": [["item", "qty"], ["a", 2], ["total", "=SUM(B2:B2)"]]})
    documents.t_make_xlsx("sales", sheets)
    ws = openpyxl.load_workbook(tmp_path / "sales.xlsx")["Sales"]
    assert ws["A1"].font.bold
    assert ws["B3"].value == "=SUM(B2:B2)"


# --- mail ---------------------------------------------------------------
class FakeResponse:
    def __init__(self, data, status=200):
        self._data, self.status_code, self.text = data, status, json.dumps(data)

    def json(self):
        return self._data


def test_inbox_requires_a_mailbox(store):
    assert "no mailbox yet" in mail.InboxTools(store).inbox(1)


def test_inbox_create_and_list(store, monkeypatch):
    def fake_get(url, **kw):
        if url.endswith("/domains"):
            return FakeResponse({"hydra:member": [{"domain": "example.test"}]})
        assert kw["headers"] == {"Authorization": "Bearer tok"}
        return FakeResponse(
            {
                "hydra:member": [
                    {
                        "id": "m1",
                        "from": {"address": "a@b.c"},
                        "subject": "Hi",
                        "createdAt": "2026-09-15T10:00:00",
                    }
                ]
            }
        )

    def fake_post(url, **kw):
        return FakeResponse({"token": "tok"} if url.endswith("/token") else {}, 201)

    monkeypatch.setattr(mail.requests, "get", fake_get)
    monkeypatch.setattr(mail.requests, "post", fake_post)
    tools = mail.InboxTools(store, api="https://mail.test")
    assert tools.create(1).startswith("mailbox ready: bot")
    assert "@example.test" in store.setting(1, "mailbox")
    assert tools.inbox(1) == "[m1] a@b.c — Hi (2026-09-15T10:00)"


def test_mail_send_explains_missing_smtp(monkeypatch):
    for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASS"):
        monkeypatch.delenv(k, raising=False)
    assert "needs SMTP_HOST" in mail.t_mail_send("x@y.z", "s", "b")


# --- crypto -------------------------------------------------------------
def test_crypto_unknown_action_lists_actions():
    assert crypto.t_crypto("nope").startswith("actions: price, market, balance, gas")


def test_crypto_balance_requires_address():
    assert crypto.t_crypto("balance") == "give an address"


def test_crypto_network_errors_are_reported(monkeypatch):
    def boom(*a, **kw):
        raise crypto.requests.ConnectionError("offline")

    monkeypatch.setattr(crypto.requests, "get", boom)
    assert crypto.t_crypto("price", "bitcoin").startswith("lookup failed: ConnectionError")


# --- clock --------------------------------------------------------------
def test_now_has_a_date():
    import re

    assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", t_now())
