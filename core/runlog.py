"""Run logging: every run writes a committable log file plus a summarized JSON
digest so a failure on someone else's machine can be shipped back for debugging.

Two artifacts per run, under ``logs/runs/`` (``.txt`` / ``.json`` are not
gitignored, unlike ``*.log``):

    logs/runs/run-<id>.txt           full DEBUG log (devices, ASR, decisions,
                                     LLM prompts, task timings, tracebacks)
    logs/runs/run-<id>.summary.json  condensed: durations, LLM requests, per-turn
                                     latencies, error tally, "where it got stuck"

Components just log to the ``speaker.*`` loggers (optionally with an
``extra={"llm_request": {...}}`` payload); a :class:`_SummaryHandler` aggregates
those into the summary, so nothing is coupled to a global object.
"""
from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

_FMT = "%(asctime)s.%(msecs)03d %(levelname)-5s %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"


def _hhmmss(epoch: float) -> str:
    return datetime.fromtimestamp(epoch).strftime("%H:%M:%S")


@dataclass
class RunSummary:
    """Structured digest of one run, built up from log records + metrics."""

    run_id: str
    log_path: str
    started_at: float = field(default_factory=time.time)
    meta: dict = field(default_factory=dict)
    level_counts: dict = field(default_factory=lambda: defaultdict(int))
    llm_requests: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    turns: list = field(default_factory=list)

    def note(self, **meta) -> None:
        self.meta.update({k: v for k, v in meta.items() if v is not None})

    def attach_metrics(self, records: list) -> None:
        self.turns = records

    def to_dict(self) -> dict:
        llm_times = [r.get("duration_sec", 0.0) for r in self.llm_requests]
        total_llm = round(sum(llm_times), 2)
        errors = [e for e in self.errors if e["level"] in ("ERROR", "CRITICAL")]
        # "Where did it get stuck?" heuristics for the reader.
        stuck = []
        if self.llm_requests and all(r.get("cancelled") for r in self.llm_requests):
            stuck.append("every LLM request was cancelled (barge-in storm or no audio settle?)")
        if errors:
            stuck.append(f"{len(errors)} error(s) -- see 'errors' below and the .txt traceback")
        if not self.llm_requests and self.meta.get("engine") in ("sherpa", "livekit"):
            stuck.append("no LLM request was ever issued (ASR never produced a final?)")
        return {
            "run_id": self.run_id,
            "log_path": self.log_path,
            "duration_sec": round(time.time() - self.started_at, 2),
            "meta": self.meta,
            "counts": {
                "llm_requests": len(self.llm_requests),
                "turns": len(self.turns),
                "errors": len(errors),
                "warnings": self.level_counts.get("WARNING", 0),
                "log_lines_by_level": dict(self.level_counts),
            },
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
    """A log handler that folds records into a :class:`RunSummary`."""

    def __init__(self, summary: RunSummary):
        super().__init__(level=logging.DEBUG)
        self._summary = summary

    def emit(self, record: logging.LogRecord) -> None:
        s = self._summary
        s.level_counts[record.levelname] += 1
        req = getattr(record, "llm_request", None)
        if isinstance(req, dict):
            s.llm_requests.append(req)
        if record.levelno >= logging.WARNING:
            exc = None
            if record.exc_info:
                exc = logging.Formatter().formatException(record.exc_info)
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

    def finalize(self, metrics_records: Optional[list] = None) -> None:
        if metrics_records is not None:
            self.summary.attach_metrics(metrics_records)
        self.summary.write(self.summary_path)
        self.logger.info("run summary written -> %s", self.summary_path)


def setup_logging(
    debug: bool = False,
    *,
    log_dir: str = "logs/runs",
    run_id: Optional[str] = None,
    console: bool = True,
) -> RunLog:
    """Configure the ``speaker`` logger tree to write a rich DEBUG file plus an
    aggregating summary handler, and (optionally) a console handler at INFO
    (or DEBUG when ``debug``). Returns a :class:`RunLog` to finalize at the end."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = str(Path(log_dir) / f"run-{run_id}.txt")
    summary_path = str(Path(log_dir) / f"run-{run_id}.summary.json")

    fmt = logging.Formatter(_FMT, _DATEFMT)
    root = logging.getLogger("speaker")
    root.setLevel(logging.DEBUG)
    root.propagate = False
    for h in list(root.handlers):
        root.removeHandler(h)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)  # the file is always rich, regardless of --debug
    fh.setFormatter(fmt)
    root.addHandler(fh)

    if console:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG if debug else logging.INFO)
        ch.setFormatter(fmt)
        root.addHandler(ch)

    summary = RunSummary(run_id=run_id, log_path=log_path)
    root.addHandler(_SummaryHandler(summary))

    root.info("run %s started (debug=%s) -> %s", run_id, debug, log_path)
    return RunLog(run_id, log_path, summary_path, summary, root)
