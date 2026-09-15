"""Current local time."""

from __future__ import annotations

import time


def t_now() -> str:
    """Return the current local date and time, e.g. ``Tuesday 2026-09-15 17:04:00 BST``."""
    return time.strftime("%A %Y-%m-%d %H:%M:%S %Z", time.localtime())
