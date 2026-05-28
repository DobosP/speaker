"""Parallel cloud-LLM dev tool.

Fire multiple prompts at a single OpenAI-compatible cloud endpoint
(DeepSeek, OpenAI, Together, Groq, ...), stream their responses in
parallel, and **hard-close** any stream the moment you don't need the
rest -- which is the only way to stop the provider billing you for the
tokens still in flight (the openai SDK's stream is exited as a context
manager, which closes the underlying HTTP connection).

Use it during research while building the assistant: ask N angles in
parallel, kill the ones that aren't going anywhere, and keep a smart
overview of what's running so you don't burn tokens by accident.

Run:

    python -m tools.cloudchat

It reads ``config.json``'s ``llm.cloud`` block for ``base_url`` /
``model`` / ``api_key_env``; no separate config. The ``openai`` Python
package is imported lazily so the rest of the repo stays unaffected;
the tool prints an install hint if it's missing.

This module is dev tooling -- it does NOT touch the voice runtime or
the brain. It only depends on ``config.json`` for the endpoint.
"""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Protocol


# --- session model -----------------------------------------------------------

@dataclass
class QueryState:
    """Single in-flight (or completed) cloud query."""

    qid: int
    prompt: str
    started: float
    finished: Optional[float] = None
    tokens: int = 0
    chars: int = 0
    text: str = ""
    status: str = "running"  # 'running' | 'done' | 'cancelled' | 'error'
    error: Optional[str] = None
    cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def elapsed(self) -> float:
        return (self.finished or time.time()) - self.started


class CloudStream(Protocol):
    """Context-managed token stream. Exiting the ``with`` block MUST close
    the underlying HTTP connection (so a cancel actually stops the bill)."""

    def __enter__(self) -> Iterator[str]: ...
    def __exit__(self, *exc: Any) -> None: ...


class CloudClient(Protocol):
    """Anything that can serve a streaming chat completion via a context-
    managed token stream. ``OpenAICloudClient`` is the real impl; tests
    inject a fake."""

    def stream_chat(self, model: str, messages: list[dict]) -> CloudStream: ...


class CloudSession:
    """Owns the parallel queries. Each :meth:`fire` spawns a daemon thread
    that streams into the matching :class:`QueryState`; :meth:`cancel`
    sets the state's cancel event AND closes the upstream HTTP stream
    on the next iteration boundary."""

    def __init__(
        self,
        client: CloudClient,
        model: str,
        *,
        system: Optional[str] = None,
    ) -> None:
        self._client = client
        self._model = model
        self._system = system
        self._queries: dict[int, QueryState] = {}
        self._lock = threading.Lock()
        self._next_id = 1
        # Tokens land here in arrival order so the CLI printer thread can
        # interleave them with [Qn] prefixes without polling state.
        self.output: "queue.Queue[tuple[int, str]]" = queue.Queue()

    @property
    def model(self) -> str:
        return self._model

    def fire(self, prompt: str) -> QueryState:
        """Spawn a new parallel query. Returns the freshly-allocated state."""
        with self._lock:
            qid = self._next_id
            self._next_id += 1
            state = QueryState(qid=qid, prompt=prompt, started=time.time())
            self._queries[qid] = state
        threading.Thread(
            target=self._run, args=(state,), daemon=True, name=f"cloudchat-Q{qid}"
        ).start()
        return state

    def cancel(self, qid: int) -> bool:
        """Signal cancellation. The query thread sees this on its next
        iteration boundary, marks the state ``cancelled``, and exits the
        context manager (which closes the HTTP connection). Returns True
        if a cancel was actually sent (i.e. the query was still running)."""
        with self._lock:
            state = self._queries.get(qid)
        if state is None or state.status != "running":
            return False
        state.cancel.set()
        return True

    def cancel_all(self) -> int:
        with self._lock:
            states = list(self._queries.values())
        sent = 0
        for s in states:
            if s.status == "running":
                s.cancel.set()
                sent += 1
        return sent

    def snapshot(self) -> list[QueryState]:
        with self._lock:
            return list(self._queries.values())

    def total_tokens(self) -> int:
        return sum(s.tokens for s in self.snapshot())

    def _run(self, state: QueryState) -> None:
        try:
            messages: list[dict] = []
            if self._system:
                messages.append({"role": "system", "content": self._system})
            messages.append({"role": "user", "content": state.prompt})
            with self._client.stream_chat(self._model, messages) as tokens:
                for piece in tokens:
                    if state.cancel.is_set():
                        state.status = "cancelled"
                        return
                    state.text += piece
                    state.tokens += 1  # chunks ~= tokens for SSE deltas
                    state.chars += len(piece)
                    self.output.put((state.qid, piece))
            if state.status == "running":
                state.status = "done"
        except Exception as exc:  # noqa: BLE001
            state.status = "error"
            state.error = repr(exc)
        finally:
            state.finished = time.time()


# --- openai SDK adapter (lazy) -----------------------------------------------

class _OpenAIStream:
    """Wraps the openai SDK's chat completion stream as a :class:`CloudStream`.
    The SDK object already supports ``.close()``; we call it on context exit
    so cancel really does close the HTTP connection."""

    def __init__(self, sdk_stream: Any) -> None:
        self._stream = sdk_stream

    def __enter__(self) -> Iterator[str]:
        return self._iter()

    def __exit__(self, *exc: Any) -> None:
        try:
            self._stream.close()
        except Exception:  # noqa: BLE001
            pass

    def _iter(self) -> Iterator[str]:
        for chunk in self._stream:
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            piece = getattr(choices[0].delta, "content", None)
            if piece:
                yield piece


class OpenAICloudClient:
    """Real CloudClient using the openai SDK. Import is lazy so the rest
    of the repo never pulls openai unless this tool actually runs."""

    def __init__(
        self,
        *,
        base_url: Optional[str],
        api_key: Optional[str],
        timeout: float = 60.0,
    ) -> None:
        try:
            from openai import OpenAI  # lazy
        except ImportError as exc:  # noqa: BLE001
            raise SystemExit(
                "tools.cloudchat needs the 'openai' Python package.\n"
                "  pip install openai\n"
                f"(import error: {exc})"
            )
        self._client = OpenAI(
            base_url=base_url, api_key=api_key or "not-needed", timeout=timeout
        )

    def stream_chat(self, model: str, messages: list[dict]) -> CloudStream:
        return _OpenAIStream(
            self._client.chat.completions.create(
                model=model, messages=messages, stream=True
            )
        )


# --- config + CLI ------------------------------------------------------------

def load_cloud_config(path: str = "config.json") -> dict:
    """Read ``llm.cloud`` from the project config. Same block ``core/app.py``
    uses, so the tool reuses whatever the user already configured."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return (raw.get("llm", {}) or {}).get("cloud", {}) or {}


def parse_command(line: str) -> tuple[str, list[str]]:
    """Parse a CLI line. ``/cmd arg arg`` -> ``('cmd', ['arg', 'arg'])``;
    anything else -> ``('prompt', [line])``. Empty -> ``('noop', [])``."""
    stripped = line.strip()
    if not stripped:
        return ("noop", [])
    if stripped.startswith("/"):
        head, *rest = stripped[1:].split(maxsplit=1)
        args = rest[0].split() if rest else []
        return (head.lower(), args)
    return ("prompt", [stripped])


def format_status(states: list[QueryState]) -> str:
    """Render the 'smart overview' table. Showns: id, status, tokens,
    elapsed, prompt preview. Counts at the bottom."""
    if not states:
        return "(no queries yet)"
    rows = []
    rows.append(f"{'ID':<4} {'STATUS':<10} {'TOKENS':>6} {'TIME':>7}  PROMPT")
    rows.append("-" * 60)
    counts: dict[str, int] = {}
    total_tokens = 0
    for s in sorted(states, key=lambda x: x.qid):
        counts[s.status] = counts.get(s.status, 0) + 1
        total_tokens += s.tokens
        preview = s.prompt if len(s.prompt) <= 38 else s.prompt[:35] + "..."
        rows.append(
            f"Q{s.qid:<3} {s.status:<10} {s.tokens:>6} {s.elapsed:>6.1f}s  {preview}"
        )
    summary = " | ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
    rows.append("-" * 60)
    rows.append(f"total: {total_tokens} tokens | {summary}")
    return "\n".join(rows)


_HELP = """\
type a query and Enter   fire a parallel query
/cancel <n>              cancel query Q<n>
/cancel *                cancel ALL running queries (hard-close streams)
/status                  print the overview table
/help                    this help
/quit                    exit
"""


def _printer_loop(session: CloudSession, out, stop: threading.Event) -> None:
    """Pulls tokens from the session output queue and prints them with a
    ``[Qn]`` prefix. Also prints a ``[Qn] [done/cancelled/error]`` footer
    when a query finishes, so the user always sees the terminal state."""
    seen_finished: set[int] = set()
    current_qid: Optional[int] = None
    while not stop.is_set():
        try:
            qid, piece = session.output.get(timeout=0.1)
        except queue.Empty:
            for s in session.snapshot():
                if s.status != "running" and s.qid not in seen_finished:
                    if current_qid is not None:
                        out.write("\n")
                        current_qid = None
                    seen_finished.add(s.qid)
                    detail = f" ({s.error})" if s.error else ""
                    out.write(
                        f"[Q{s.qid}] [{s.status} at {s.elapsed:.1f}s, {s.tokens} tokens]"
                        f"{detail}\n"
                    )
                    out.flush()
            continue
        if qid != current_qid:
            if current_qid is not None:
                out.write("\n")
            out.write(f"[Q{qid}] ")
            current_qid = qid
        out.write(piece)
        out.flush()


def run_cli(
    session: CloudSession,
    *,
    in_stream=None,
    out_stream=None,
    err_stream=None,
) -> None:
    """Main interactive loop. Reads commands from ``in_stream`` (default
    stdin), dispatches them, and prints streamed tokens to ``out_stream``.
    The printer runs on a daemon thread; the main thread blocks on input."""
    in_stream = in_stream or sys.stdin
    out_stream = out_stream or sys.stdout
    err_stream = err_stream or sys.stderr

    out_stream.write(_HELP)
    out_stream.write(f"connected: model={session.model}\n")
    out_stream.flush()

    stop = threading.Event()
    printer = threading.Thread(
        target=_printer_loop,
        args=(session, out_stream, stop),
        daemon=True,
        name="cloudchat-printer",
    )
    printer.start()

    try:
        while True:
            try:
                line = in_stream.readline()
            except KeyboardInterrupt:
                err_stream.write("\n(interrupt; cancelling all running)\n")
                session.cancel_all()
                continue
            if not line:  # EOF
                break
            cmd, args = parse_command(line)
            if cmd == "noop":
                continue
            if cmd == "quit" or cmd == "exit":
                break
            if cmd == "help":
                out_stream.write(_HELP)
                out_stream.flush()
            elif cmd == "status":
                out_stream.write(format_status(session.snapshot()) + "\n")
                out_stream.flush()
            elif cmd == "cancel":
                if not args:
                    err_stream.write("usage: /cancel <n> | /cancel *\n")
                    continue
                target = args[0]
                if target == "*":
                    n = session.cancel_all()
                    err_stream.write(f"cancelling {n} running quer{'y' if n == 1 else 'ies'}\n")
                else:
                    try:
                        qid = int(target.lstrip("Qq"))
                    except ValueError:
                        err_stream.write(f"bad query id: {target!r}\n")
                        continue
                    if session.cancel(qid):
                        err_stream.write(f"cancelling Q{qid}\n")
                    else:
                        err_stream.write(f"Q{qid} not running\n")
            elif cmd == "prompt":
                state = session.fire(args[0])
                err_stream.write(f"fired Q{state.qid}\n")
            else:
                err_stream.write(f"unknown command: /{cmd} (try /help)\n")
    finally:
        # Cancel anything still running so we don't keep paying for tokens
        # after the user has walked away.
        session.cancel_all()
        # Let the printer flush the last status footers before we exit.
        time.sleep(0.2)
        stop.set()
        printer.join(timeout=1.0)
        snap = session.snapshot()
        if snap:
            out_stream.write("\n" + format_status(snap) + "\n")
            out_stream.flush()


def main(argv: Optional[list[str]] = None) -> int:
    cfg = load_cloud_config()
    model = cfg.get("model")
    if not model:
        sys.stderr.write(
            "tools.cloudchat needs config.json -> llm.cloud.model set.\n"
            "Example:\n"
            '  "llm": {"cloud": {"enabled": true, "model": "deepseek-chat",\n'
            '    "base_url": "https://api.deepseek.com",\n'
            '    "api_key_env": "DEEPSEEK_API_KEY"}}\n'
        )
        return 2
    api_key = os.environ.get(cfg.get("api_key_env", "OPENAI_API_KEY"))
    client = OpenAICloudClient(
        base_url=cfg.get("base_url"),
        api_key=api_key,
        timeout=float(cfg.get("timeout", 60.0)),
    )
    session = CloudSession(client, model, system=cfg.get("system_prompt"))
    run_cli(session)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
