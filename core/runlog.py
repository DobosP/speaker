"""Run logging: every run writes a committable log file plus a summarized JSON
digest -- created even when the run crashes -- so a failure on someone else's
machine can be shipped back for debugging.

Concurrency: the ``speaker`` logger gets a non-blocking ``QueueHandler``; a
``QueueListener`` on a background thread does the actual formatting + disk
writes (and summary aggregation). So logging from the real-time audio/LLM
threads is just an enqueue -- the hot path never blocks on I/O.

Artifacts under ``logs/runs/`` (``.txt``/``.json`` are not gitignored):

    run-<id>.txt           full DEBUG log (devices, ASR, decisions, prompts,
                           timings, tracebacks)
    run-<id>.summary.json  condensed: durations, LLM requests, per-turn
                           latencies, the conversation transcript with stage
                           timing, error tally, and "where it got stuck" hints

Components log to ``speaker.*`` and may attach structured payloads via
``extra={"llm_request": {...}}`` or ``extra={"transcript": {...}}``; the
:class:`_SummaryHandler` folds those into the digest. ``finalize()`` is
idempotent and also runs at interpreter exit, so the summary is written on a
clean stop, a Ctrl-C, or an unhandled exception alike.
"""
from __future__ import annotations

import atexit
import json
import logging
import queue
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from logging.handlers import QueueHandler, QueueListener
from pathlib import Path
from typing import Optional

_FMT = "%(asctime)s.%(msecs)03d %(levelname)-5s %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"


def _hhmmss(epoch: float) -> str:
    return datetime.fromtimestamp(epoch).strftime("%H:%M:%S")


@dataclass
class RunSummary:
    """Structured digest of one run, built from log records + metrics."""

    run_id: str
    log_path: str
    started_at: float = field(default_factory=time.time)
    meta: dict = field(default_factory=dict)
    level_counts: dict = field(default_factory=lambda: defaultdict(int))
    llm_requests: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    transcript: list = field(default_factory=list)
    turns: list = field(default_factory=list)

    def note(self, **meta) -> None:
        self.meta.update({k: v for k, v in meta.items() if v is not None})

    def attach_metrics(self, records: list) -> None:
        self.turns = records

    def to_dict(self) -> dict:
        llm_times = [r.get("duration_sec", 0.0) for r in self.llm_requests]
        total_llm = round(sum(llm_times), 2)
        errors = [e for e in self.errors if e["level"] in ("ERROR", "CRITICAL")]
        stuck = []
        if self.llm_requests and all(r.get("cancelled") for r in self.llm_requests):
            stuck.append("every LLM request was cancelled (barge-in storm or no audio settle?)")
        if errors:
            stuck.append(f"{len(errors)} error(s) -- see 'errors' below and the .txt traceback")
        if not self.llm_requests and self.meta.get("engine") in ("sherpa", "livekit"):
            stuck.append("no LLM request was ever issued (ASR never produced a final?)")
        if not self.transcript and self.meta.get("engine") in ("sherpa", "livekit"):
            stuck.append("empty transcript (nothing was recognized or spoken)")
        return {
            "run_id": self.run_id,
            "log_path": self.log_path,
            "duration_sec": round(time.time() - self.started_at, 2),
            "meta": self.meta,
            "counts": {
                "llm_requests": len(self.llm_requests),
                "turns": len(self.turns),
                "transcript_entries": len(self.transcript),
                "errors": len(errors),
                "warnings": self.level_counts.get("WARNING", 0),
                "log_lines_by_level": dict(self.level_counts),
            },
            "transcript": self.transcript,
            "llm": {
                "total_time_sec": total_llm,
                "avg_time_sec": round(total_llm / len(llm_times), 2) if llm_times else None,
                "requests": self.llm_requests,
            },
            "turns": self.turns,
            "stuck_hints": stuck,
            "errors": self.errors[-50:],
        }

    def write(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


class _SummaryHandler(logging.Handler):
    """Folds log records (and their structured ``extra`` payloads) into a
    :class:`RunSummary`. Runs on the listener thread, off the hot path."""

    def __init__(self, summary: RunSummary):
        super().__init__(level=logging.DEBUG)
        self._summary = summary

    def emit(self, record: logging.LogRecord) -> None:
        s = self._summary
        s.level_counts[record.levelname] += 1
        req = getattr(record, "llm_request", None)
        if isinstance(req, dict):
            s.llm_requests.append(req)
        tr = getattr(record, "transcript", None)
        if isinstance(tr, dict):
            entry = dict(tr)
            entry["at_sec"] = round(record.created - s.started_at, 2)
            s.transcript.append(entry)
        if record.levelno >= logging.WARNING:
            exc = logging.Formatter().formatException(record.exc_info) if record.exc_info else None
            s.errors.append(
                {
                    "t": _hhmmss(record.created),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                    "exc": exc,
                }
            )


@dataclass
class RunLog:
    run_id: str
    log_path: str
    summary_path: str
    summary: RunSummary
    logger: logging.Logger
    listener: QueueListener
    handlers: list
    _finalized: bool = False

    def finalize(self, metrics_records: Optional[list] = None) -> None:
        """Flush async logging, then write the summary. Idempotent and safe to
        call from a finally block, a signal path, or atexit."""
        if self._finalized:
            return
        self._finalized = True
        if metrics_records is not None:
            self.summary.attach_metrics(metrics_records)
        # Drain the queue so every record reaches the summary handler first.
        try:
            self.listener.stop()
        except Exception:  # noqa: BLE001 - never mask the original failure
            pass
        try:
            self.summary.write(self.summary_path)
        except Exception:  # noqa: BLE001
            pass
        for h in self.handlers:
            try:
                h.close()
            except Exception:  # noqa: BLE001
                pass


def setup_logging(
    debug: bool = False,
    *,
    log_dir: str = "logs/runs",
    run_id: Optional[str] = None,
    console: bool = True,
) -> RunLog:
    """Wire the ``speaker`` logger to a non-blocking queue feeding a background
    listener (rich DEBUG file + console + summary aggregation). Returns a
    :class:`RunLog`; its ``finalize()`` also runs at interpreter exit."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = str(Path(log_dir) / f"run-{run_id}.txt")
    summary_path = str(Path(log_dir) / f"run-{run_id}.summary.json")

    fmt = logging.Formatter(_FMT, _DATEFMT)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)  # the file is always rich, regardless of --debug
    fh.setFormatter(fmt)
    handlers: list[logging.Handler] = [fh]
    if console:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG if debug else logging.INFO)
        ch.setFormatter(fmt)
        handlers.append(ch)
    summary = RunSummary(run_id=run_id, log_path=log_path)
    handlers.append(_SummaryHandler(summary))

    log_q: "queue.Queue" = queue.Queue(-1)  # unbounded; enqueue is cheap
    listener = QueueListener(log_q, *handlers, respect_handler_level=True)
    listener.start()

    root = logging.getLogger("speaker")
    root.setLevel(logging.DEBUG)
    root.propagate = False
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(QueueHandler(log_q))

    runlog = RunLog(run_id, log_path, summary_path, summary, root, listener, handlers)
    atexit.register(runlog.finalize)
    root.info("run %s started (debug=%s) -> %s", run_id, debug, log_path)
    return runlog
