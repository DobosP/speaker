"""Dormant optional host facade and retained Python audio rollback.

``serving/`` is the real Go HTTP/auth/JWT/static boundary. Its ordinary image is
Python-free. ``text_backend.py`` is an explicitly selected, one-turn private
stdin/stdout JSON adapter to the retained Python assistant text core; it does
not host HTTP. Without an adapter, the Go server returns unavailable for chat.

``worker.py`` retains the LiveKit audio rollback and Python ``VoiceRuntime``.
``token_server.py`` retains import-safe compatibility helpers for that worker;
it is no longer the web listener. No package import activates audio or a server.

Current status and promotion gates live in ``STATUS.md`` and ADR-0096/0097/0164;
see ``docs/go_serving_boundary.md`` for migration scope and qualification.
"""
