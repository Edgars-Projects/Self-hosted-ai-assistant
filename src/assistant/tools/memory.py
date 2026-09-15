"""Explicit key/value memory: facts the user asks the assistant to remember."""

from __future__ import annotations

from typing import Callable

from ..db import Store
from ..util import clip

VecAdd = Callable[[int, str, str, str], object]


class MemoryTools:
    """``remember`` / ``recall`` / ``forget`` backed by the ``facts`` table.

    Each saved fact is also embedded via ``vec_add`` so it can be found later
    by semantic search, not just by exact key.
    """

    def __init__(self, store: Store, vec_add: VecAdd) -> None:
        self.store = store
        self.vec_add = vec_add

    def remember(self, chat_id: int, key: str, value: str) -> str:
        self.store.q(
            "INSERT OR REPLACE INTO facts (chat_id,key,value) VALUES (?,?,?)",
            (chat_id, key, value),
        )
        self.vec_add(chat_id, "fact", key, f"{key}: {value}")
        return f"saved {key!r}"

    def recall(self, chat_id: int, key: str | None = None) -> str:
        if key:
            r = self.store.q(
                "SELECT value FROM facts WHERE chat_id=? AND key=?",
                (chat_id, key),
                "one",
            )
            return r[0] if r else f"nothing saved under {key!r}"
        rows = self.store.q("SELECT key,value FROM facts WHERE chat_id=?", (chat_id,), "all")
        return clip("\n".join(f"{k}: {v}" for k, v in rows) or "(empty)")

    def forget(self, chat_id: int, key: str) -> str:
        self.store.q("DELETE FROM facts WHERE chat_id=? AND key=?", (chat_id, key))
        return f"forgot {key!r}"
