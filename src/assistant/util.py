"""Small helpers shared by the app and tool modules."""

from __future__ import annotations

from .config import MAX_TOOL_OUT


def clip(s: object, n: int = MAX_TOOL_OUT) -> str:
    """Truncate tool output for the model, never silently.

    A silent truncation is a bug: the model would guess at the missing part.
    The marker tells it exactly how to fetch the rest instead.
    """
    s = str(s)
    if len(s) <= n:
        return s
    return (
        s[:n] + f"\n\n[TRUNCATED: {len(s):,} characters total, "
        f"{n:,} shown. To get the rest: call the same tool with a "
        f"more specific target, or use run_python / python_session "
        f"to fetch and extract only the part you need (e.g. slice "
        f"the text, or search it for a keyword). Do NOT guess at "
        f"the missing content — go and get it.]"
    )
