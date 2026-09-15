"""Email: a disposable inbox (mail.tm) for receiving, and SMTP for sending."""

from __future__ import annotations

import json
import os
import secrets as _secrets
import smtplib
from email.message import EmailMessage

import requests

from ..config import MAIL_API
from ..db import Store
from ..util import clip


def _smtp_credentials() -> tuple[str | None, str | None, str | None]:
    """SMTP settings from env vars, falling back to Kaggle secrets."""
    host = os.environ.get("SMTP_HOST")
    user = os.environ.get("SMTP_USER")
    pw = os.environ.get("SMTP_PASS")
    if not (host and user and pw):
        try:
            from kaggle_secrets import UserSecretsClient

            u = UserSecretsClient()
            host = host or u.get_secret("SMTP_HOST")
            user = user or u.get_secret("SMTP_USER")
            pw = pw or u.get_secret("SMTP_PASS")
        except Exception:  # noqa: BLE001 - not on Kaggle, or secret missing
            pass
    return host, user, pw


def t_mail_send(to: str, subject: str, body: str) -> str:
    """Send a plain-text email over SMTP with STARTTLS."""
    host, user, pw = _smtp_credentials()
    if not (host and user and pw):
        return (
            "sending needs SMTP_HOST, SMTP_USER and SMTP_PASS as "
            "secrets. Receiving via the throwaway inbox works "
            "without them."
        )
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = user, to, subject
    msg.set_content(body)
    port = int(os.environ.get("SMTP_PORT", "587"))
    with smtplib.SMTP(host, port, timeout=60) as srv:
        srv.starttls()
        srv.login(user, pw)
        srv.send_message(msg)
    return f"sent to {to}"


class InboxTools:
    """A per-chat disposable mailbox whose credentials live in chat settings."""

    def __init__(self, store: Store, api: str = MAIL_API) -> None:
        self.store = store
        self.api = api

    def _account(self, chat_id: int) -> dict | None:
        raw = self.store.setting(chat_id, "mailbox")
        return json.loads(raw) if raw else None

    def _headers(self, chat_id: int) -> tuple[dict | None, str | None]:
        a = self._account(chat_id)
        if not a:
            return None, "no mailbox yet — call email_create_inbox first"
        return {"Authorization": f"Bearer {a['token']}"}, None

    def create(self, chat_id: int) -> str:
        dom = requests.get(f"{self.api}/domains", timeout=30).json()
        domain = (dom.get("hydra:member") or dom)[0]["domain"]
        addr = f"bot{_secrets.token_hex(4)}@{domain}"
        pw = _secrets.token_urlsafe(12)
        r = requests.post(
            f"{self.api}/accounts", timeout=30, json={"address": addr, "password": pw}
        )
        if r.status_code >= 300:
            return f"could not create mailbox: {r.status_code} {r.text[:200]}"
        tok = requests.post(
            f"{self.api}/token", timeout=30, json={"address": addr, "password": pw}
        ).json()
        self.store.set_setting(
            chat_id,
            "mailbox",
            json.dumps({"address": addr, "password": pw, "token": tok["token"]}),
        )
        return f"mailbox ready: {addr}"

    def inbox(self, chat_id: int, limit: int = 10) -> str:
        h, err = self._headers(chat_id)
        if err:
            return err
        r = requests.get(f"{self.api}/messages", headers=h, timeout=30).json()
        msgs = (r.get("hydra:member") or r)[: int(limit)]
        if not msgs:
            return "inbox empty"
        return clip(
            "\n".join(
                f"[{m['id']}] {m['from']['address']} — {m['subject']} ({m['createdAt'][:16]})"
                for m in msgs
            )
        )

    def read(self, chat_id: int, message_id: str) -> str:
        from bs4 import BeautifulSoup

        h, err = self._headers(chat_id)
        if err:
            return err
        m = requests.get(f"{self.api}/messages/{message_id}", headers=h, timeout=30).json()
        body = m.get("text") or BeautifulSoup(
            " ".join(m.get("html") or []), "html.parser"
        ).get_text(" ")
        return clip(f"From: {m['from']['address']}\nSubject: {m['subject']}\n\n{body}")
