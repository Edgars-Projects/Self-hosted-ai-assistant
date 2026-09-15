"""Run the real ``assistant.app.main`` against fake Telegram and Ollama.

Usage: python scenario.py <ollama-url> <state-dir> <output.json> <message>...

Only the network edges are faked: Telegram HTTP calls are answered locally
and model installs are skipped. Handlers, the agent loop, tool dispatch and
the database all run for real. Every message the bot sends is recorded.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Any

ollama_url, state_dir, out_path, *script = sys.argv[1:]
os.environ.update(BOT_STATE=state_dir, TELEGRAM_TOKEN="123456:TEST")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import telegram  # noqa: E402
from telegram.ext import Application  # noqa: E402
from telegram.request import HTTPXRequest  # noqa: E402

OWNER_CHAT = {"id": 42, "type": "private", "first_name": "Owner"}
STRANGER_CHAT = {"id": 7, "type": "private", "first_name": "Stranger"}
BOT_USER = {"id": 1, "is_bot": True, "first_name": "Bot", "username": "test_bot"}
SENT: list[dict[str, Any]] = []
_ids = iter(range(1000, 10**6))


async def fake_telegram(self: HTTPXRequest, url: str, method: str, request_data=None, **_: Any):
    api = url.rsplit("/", 1)[-1]
    params = request_data.parameters if request_data else {}
    result: Any = True
    if api == "getMe":
        result = BOT_USER
    elif api in ("sendMessage", "editMessageText", "sendDocument", "sendVoice", "sendPhoto"):
        text = params.get("text") or params.get("caption")
        SENT.append({"api": api, "chat": params.get("chat_id"), "text": text})
        result = {
            "message_id": next(_ids),
            "date": int(time.time()),
            "chat": OWNER_CHAT,
            "from": BOT_USER,
            "text": text or "",
        }
    return 200, json.dumps({"ok": True, "result": result}).encode()


async def _no_network(self: HTTPXRequest) -> None:
    return None


def _update(n: int, text: str, chat: dict[str, Any]) -> dict[str, Any]:
    msg: dict[str, Any] = {
        "message_id": n,
        "date": int(time.time()),
        "chat": chat,
        "from": {"id": chat["id"], "is_bot": False, "first_name": "U"},
        "text": text,
    }
    if text.startswith("/"):
        msg["entities"] = [{"type": "bot_command", "offset": 0, "length": len(text.split()[0])}]
    return {"update_id": n, "message": msg}


def run_script(self: Application, **_: Any) -> None:
    async def go() -> None:
        await self.initialize()
        if self.post_init:
            await self.post_init(self)
        for n, text in enumerate(script, 1):
            chat = STRANGER_CHAT if text.startswith("STRANGER ") else OWNER_CHAT
            text = text.removeprefix("STRANGER ")
            SENT.append({"api": "USER", "chat": chat["id"], "text": text})
            await self.process_update(telegram.Update.de_json(_update(n, text, chat), self.bot))
            await asyncio.sleep(0.2)
        await asyncio.sleep(1)
        await self.shutdown()

    asyncio.run(go())


HTTPXRequest.do_request = fake_telegram
HTTPXRequest.initialize = _no_network
Application.run_polling = run_script

from assistant import app  # noqa: E402

vars(app).update(
    bootstrap=lambda: None,
    OLLAMA_URL=ollama_url,
    ENABLE_TUNNEL=False,
    IDLE_ENABLED=False,
    SMART_MODEL="qwen3:30b-a3b",
    UTILITY_MODEL="qwen3:4b",
)
app.main()
with open(out_path, "w") as fh:
    json.dump(SENT, fh, indent=1)
