"""Tests for the parallel cloud-LLM dev tool (tools.cloudchat).

Uses an injected fake client (no real cloud, no openai SDK dep) so tests
run on the same logic suite as everything else. Covers:

* :class:`CloudSession` lifecycle: fire / cancel / cancel_all / parallel
  execution / error capture;
* hard-close semantics: cancel really exits the upstream stream's context
  manager (so a real HTTP connection would close);
* CLI helpers: ``parse_command`` and ``format_status``.
"""
from __future__ import annotations

import io
import threading
import time
from typing import Any, Iterator

from tools.cloudchat import (
    CloudSession,
    QueryState,
    format_status,
    parse_command,
    run_cli,
)


# --- fakes -------------------------------------------------------------------

class FakeStream:
    """Minimal CloudStream fake: yields scripted tokens with a tiny delay so
    cancellation has time to land mid-stream. Records its __exit__ call so a
    test can assert the HTTP stream would have been closed."""

    def __init__(self, tokens: list[str], delay: float = 0.005) -> None:
        self._tokens = list(tokens)
        self._delay = delay
        self.closed = False
        self.iter_count = 0

    def __enter__(self) -> Iterator[str]:
        return self._iter()

    def __exit__(self, *exc: Any) -> None:
        self.closed = True

    def _iter(self) -> Iterator[str]:
        for t in self._tokens:
            if self.closed:
                break
            if self._delay:
                time.sleep(self._delay)
            self.iter_count += 1
            yield t


class FakeCloudClient:
    """Maps a prompt to a scripted list of tokens. Records every stream
    handed out so a test can inspect close-state and iteration count."""

    def __init__(
        self,
        scripts: dict[str, list[str]],
        *,
        default: list[str] | None = None,
        delay: float = 0.005,
    ) -> None:
        self._scripts = scripts
        self._default = default if default is not None else ["[unscripted]"]
        self._delay = delay
        self.streams: list[FakeStream] = []
        self.last_messages: list[dict] | None = None

    def stream_chat(self, model: str, messages: list[dict]) -> FakeStream:
        self.last_messages = messages
        prompt = messages[-1]["content"]
        tokens = self._scripts.get(prompt, self._default)
        stream = FakeStream(tokens, delay=self._delay)
        self.streams.append(stream)
        return stream


def _wait_for(predicate, *, timeout: float = 2.0, interval: float = 0.005) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


# --- parse_command -----------------------------------------------------------

def test_parse_command_routes_slash_commands_and_prompts():
    assert parse_command("") == ("noop", [])
    assert parse_command("   ") == ("noop", [])
    assert parse_command("hello world") == ("prompt", ["hello world"])
    assert parse_command("/quit") == ("quit", [])
    assert parse_command("/cancel 3") == ("cancel", ["3"])
    assert parse_command("/cancel *") == ("cancel", ["*"])
    assert parse_command("/STATUS") == ("status", [])
    assert parse_command("/cancel  Q2 ") == ("cancel", ["Q2"])


# --- format_status -----------------------------------------------------------

def test_format_status_empty():
    assert "no queries" in format_status([])


def test_format_status_table_includes_totals_and_each_row():
    states = [
        QueryState(qid=1, prompt="hi", started=time.time() - 1.5,
                   finished=time.time() - 0.5, tokens=42, chars=120,
                   text="hello", status="done"),
        QueryState(qid=2, prompt="long prompt " * 5, started=time.time() - 2.0,
                   finished=time.time() - 1.0, tokens=10, status="cancelled"),
    ]
    out = format_status(states)
    assert "Q1" in out and "Q2" in out
    assert "done" in out and "cancelled" in out
    assert "total: 52 tokens" in out
    # Long prompt gets a preview truncation.
    assert "..." in out


# --- CloudSession.fire ------------------------------------------------------

def test_fire_assigns_monotonic_ids():
    client = FakeCloudClient({"a": ["x"], "b": ["y"]})
    session = CloudSession(client, "fake-model")
    s1 = session.fire("a")
    s2 = session.fire("b")
    assert s1.qid == 1 and s2.qid == 2


def test_query_completes_with_accumulated_text():
    client = FakeCloudClient({"hello": ["he", "ll", "o"]})
    session = CloudSession(client, "fake-model")
    state = session.fire("hello")
    assert _wait_for(lambda: state.status == "done")
    assert state.text == "hello"
    assert state.tokens == 3
    assert state.chars == 5
    assert state.finished is not None


def test_system_prompt_is_forwarded():
    client = FakeCloudClient({"q": ["a"]})
    session = CloudSession(client, "fake-model", system="be terse")
    state = session.fire("q")
    assert _wait_for(lambda: state.status == "done")
    assert client.last_messages[0] == {"role": "system", "content": "be terse"}
    assert client.last_messages[-1] == {"role": "user", "content": "q"}


# --- CloudSession.cancel ----------------------------------------------------

def test_cancel_marks_state_and_exits_stream_context():
    """Hard-close semantics: cancel ends the iteration AND the stream's
    __exit__ runs (which on a real client closes the HTTP connection)."""
    client = FakeCloudClient({"slow": ["x"] * 200}, delay=0.005)
    session = CloudSession(client, "fake-model")
    state = session.fire("slow")
    # Let it stream a few tokens.
    assert _wait_for(lambda: state.tokens >= 3, timeout=1.0)
    assert state.status == "running"
    assert session.cancel(state.qid)
    assert _wait_for(lambda: state.status == "cancelled")
    # The stream's __exit__ ran -> on a real openai stream this would have
    # called .close() and severed the HTTP connection.
    assert client.streams[0].closed
    # And the iterator did NOT drain all 200 tokens.
    assert client.streams[0].iter_count < 200


def test_cancel_returns_false_after_completion():
    client = FakeCloudClient({"a": ["b"]})
    session = CloudSession(client, "fake-model")
    state = session.fire("a")
    assert _wait_for(lambda: state.status == "done")
    assert session.cancel(state.qid) is False


def test_cancel_all_targets_only_running_queries():
    client = FakeCloudClient({"a": ["x"] * 100, "b": ["y"] * 100, "c": ["z"]}, delay=0.005)
    session = CloudSession(client, "fake-model")
    session.fire("a")
    session.fire("b")
    c = session.fire("c")
    assert _wait_for(lambda: c.status == "done", timeout=1.0)
    n = session.cancel_all()
    assert n == 2  # a + b were still running; c was done
    assert _wait_for(lambda: all(s.status != "running" for s in session.snapshot()))


# --- parallelism + isolation ------------------------------------------------

def test_parallel_queries_run_concurrently():
    client = FakeCloudClient({"a": ["x"] * 50, "b": ["y"] * 50}, delay=0.005)
    session = CloudSession(client, "fake-model")
    session.fire("a")
    session.fire("b")
    # Both should be running shortly after spawn.
    assert _wait_for(
        lambda: sum(1 for s in session.snapshot() if s.status == "running") == 2,
        timeout=1.0,
    )


def test_total_tokens_sums_across_queries():
    client = FakeCloudClient({"a": ["x", "y"], "b": ["z"]})
    session = CloudSession(client, "fake-model")
    s1 = session.fire("a")
    s2 = session.fire("b")
    assert _wait_for(lambda: s1.status == "done" and s2.status == "done")
    assert session.total_tokens() == 3


# --- error path -------------------------------------------------------------

def test_stream_exception_marks_query_error():
    class _BoomClient:
        def stream_chat(self, model: str, messages: list[dict]) -> FakeStream:
            class _Boom:
                def __enter__(self):  # noqa: D401
                    raise RuntimeError("upstream 503")

                def __exit__(self, *exc): pass

            return _Boom()  # type: ignore[return-value]

    session = CloudSession(_BoomClient(), "fake-model")
    state = session.fire("hi")
    assert _wait_for(lambda: state.status == "error")
    assert state.error and "503" in state.error


# --- CLI end-to-end (in-memory streams) -------------------------------------

def test_run_cli_fires_and_then_quits():
    """Drive run_cli with a scripted input stream, assert tokens land in
    the output and the smart-overview table is printed on exit."""
    client = FakeCloudClient({"what is 2+2": ["4", "2"]}, delay=0.002)
    session = CloudSession(client, "fake-model")
    out = io.StringIO()
    err = io.StringIO()
    # Drip the input lines so the query thread has time to stream between
    # them (mirrors a real user typing).
    inp = _DripStream(["what is 2+2\n", "/status\n", "/quit\n"], pause=0.05)
    run_cli(session, in_stream=inp, out_stream=out, err_stream=err)
    out_text = out.getvalue()
    # The query streamed.
    assert "[Q1] 42" in out_text
    # The status table was printed on /status AND again on exit.
    assert "Q1" in out_text and "done" in out_text
    # The 'fired Q1' notification went to stderr.
    assert "fired Q1" in err.getvalue()


def test_run_cli_cancel_command_stops_stream():
    client = FakeCloudClient({"slow": ["x"] * 500}, delay=0.005)
    session = CloudSession(client, "fake-model")
    out = io.StringIO()
    err = io.StringIO()
    # Fire, sleep briefly via /status, then cancel, then quit. The /status
    # call gives the query thread time to start streaming before we cancel.
    inp = _DripStream(["slow\n", "/status\n", "/cancel 1\n", "/quit\n"], pause=0.05)
    run_cli(session, in_stream=inp, out_stream=out, err_stream=err)
    snap = session.snapshot()
    assert snap and snap[0].status == "cancelled"
    assert client.streams[0].closed
    assert "cancelling Q1" in err.getvalue()


class _DripStream:
    """Test stream that returns lines one at a time with a small pause
    between them, so the printer thread has time to interleave tokens."""

    def __init__(self, lines: list[str], pause: float = 0.02) -> None:
        self._lines = list(lines)
        self._pause = pause

    def readline(self) -> str:
        if not self._lines:
            return ""
        if self._pause:
            time.sleep(self._pause)
        return self._lines.pop(0)
