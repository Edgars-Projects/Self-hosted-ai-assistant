"""Tool implementations the assistant can call.

Tools that need shared state receive it explicitly (a :class:`~assistant.db.Store`
or a callable) instead of reaching into the application's closures, which keeps
them independently testable.
"""
