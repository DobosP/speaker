"""Run logging: every run writes a private local log file plus a summarized JSON
digest, including when the run crashes, so a failure can be replayed and
diagnosed on the same machine.

Concurrency: the ``speaker`` logger gets a non-blocking ``QueueHandler``; a
``QueueListener`` on a background thread does the actual formatting + disk
writes (and summary aggregation). So logging from the real-time audio/LLM
threads is just an enqueue -- the hot path never blocks on I/O.

Artifacts live under ``$SPEAKER_RUN_LOG_DIR`` (default ``logs/runs/``). New
bundles are ignored by git and may contain raw voice, transcripts, and prompts:

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
import threading
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
    system: dict = field(default_factory=dict)

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
        if (
            not self.llm_requests
            and not self.transcript
            and self.meta.get("engine") in ("sherpa", "livekit")
        ):
            stuck.append("no LLM request was ever issued (ASR never produced a final?)")
        if not self.transcript and self.meta.get("engine") in ("sherpa", "livekit"):
            stuck.append("empty transcript (nothing was recognized or spoken)")
        # Promote watchdog warnings (logger='speaker.watchdog') to named hints.
        # The watchdog emits a WARNING the moment it detects a stalled stage,
        # so this surfaces real-time stuck states that the post-hoc checks
        # above can't see (clean run, but the LLM hung for 15s mid-turn).
        wd_msgs = [
            e.get("message", "") for e in self.errors
            if e.get("logger") == "speaker.watchdog"
        ]
        if any("llm stuck" in m for m in wd_msgs):
            stuck.append("LLM stalled mid-turn (asr_final fired but no first token; see watchdog warnings)")
        if any("tts stuck" in m for m in wd_msgs):
            stuck.append("TTS stalled mid-turn (LLM streamed tokens but no audio; see watchdog warnings)")
        if any("capture silent" in m for m in wd_msgs):
            stuck.append("capture thread went silent (audio loop crashed or stalled; see watchdog warnings)")
        if any("barge-in storm" in m for m in wd_msgs):
            stuck.append("barge-in gate flapping (many detections in <2s; TTS likely leaking into mic)")
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
            "system": self.system,
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


class _ThreadQueueHandler(QueueHandler):
    """QueueHandler for an in-process *thread* queue.

    The stock ``QueueHandler.prepare()`` formats the message and strips
    ``args``/``exc_info`` on the **calling** thread (so the record is picklable
    for a multiprocessing queue). We use a plain thread queue, so we skip that:
    returning the record untouched defers all string formatting and traceback
    rendering to the listener thread. Logging on the hot path is then just a
    record allocation + a lock-free-ish ``deque.append`` -- no interpolation,
    no I/O. Safe because our log args are immutable (str/num) or freshly-built
    dicts that are never mutated after the call.

    Backpressure (backlog: unbounded runlog queue): the queue is BOUNDED now, so
    a debug-log storm can no longer grow memory without limit while the listener
    is behind. Overflow policy: DEBUG/INFO records are dropped and COUNTED; a
    WARNING+ record first tries a short blocking put (never dropped if the
    listener recovers within the grace window). The next successful enqueue
    injects one coalesced ``runlog dropped N record(s)`` WARNING — the storm is
    visible in the log without amplifying it.
    """

    _WARN_PUT_TIMEOUT_SEC = 0.25

    def __init__(self, log_queue: "queue.Queue") -> None:
        super().__init__(log_queue)
        self._dropped = 0
        self._drop_lock = threading.Lock()

    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        return record

    def enqueue(self, record: logging.LogRecord) -> None:
        if record.levelno >= logging.WARNING:
            try:
                self.queue.put(record, timeout=self._WARN_PUT_TIMEOUT_SEC)
            except queue.Full:
                with self._drop_lock:
                    self._dropped += 1
                return
            self._flush_drop_summary()
            return
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            with self._drop_lock:
                self._dropped += 1
            return
        self._flush_drop_summary()

    def _flush_drop_summary(self) -> None:
        with self._drop_lock:
            n, self._dropped = self._dropped, 0
        if not n:
            return
        summary = logging.LogRecord(
            "speaker.runlog", logging.WARNING, __file__, 0,
            "runlog dropped %d record(s) under backpressure", (n,), None,
        )
        try:
            self.queue.put_nowait(summary)
        except queue.Full:
            with self._drop_lock:
                self._dropped += n  # still saturated -- retry on a later enqueue


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


def _git_tracked_run_stems(log_dir: str) -> set:
    """Run stems under ``log_dir`` that are COMMITTED to git -- the curated
    regression / dev corpus the owner deliberately keeps (e.g. the barge-in
    replay WAVs). These must never be auto-pruned. Returns an empty set when not
    in a git repo or git is unavailable, so non-git deployments fall back to
    plain size-capping (where there is no committed corpus to protect anyway)."""
    try:
        import subprocess

        out = subprocess.run(
            ["git", "ls-files", "-z", "--", log_dir],
            capture_output=True, timeout=5,
        )
        if out.returncode != 0:
            return set()
        names = out.stdout.decode("utf-8", "replace").split("\0")
        return {Path(n).name.split(".", 1)[0] for n in names if n.strip()}
    except Exception:  # noqa: BLE001 -- git missing / not a repo / timeout
        return set()


def prune_old_runs(log_dir: str, keep: int, *, protected: "Optional[set]" = None) -> int:
    """Keep only the newest ``keep`` EPHEMERAL run bundles in ``log_dir`` so it
    doesn't grow without bound. A bundle is all files sharing a ``run-<id>``
    stem (.txt/.summary.json/.wav). Run ids are timestamps, so a name sort is
    chronological. Returns how many bundles were removed.

    Git-tracked bundles are a curated corpus the owner committed on purpose (the
    barge-in replay WAVs, kept debugging runs) and are NEVER auto-pruned -- only
    ephemeral (untracked) local runs count toward ``keep``. This is what lets the
    committed regression corpus survive the per-startup prune. Pass ``protected``
    to override the tracked-stem lookup (tests)."""
    if keep <= 0:
        return 0
    d = Path(log_dir)
    if protected is None:
        protected = _git_tracked_run_stems(log_dir)
    stems = sorted({p.name.split(".", 1)[0] for p in d.glob("run-*.*")}, reverse=True)
    prunable = [s for s in stems if s not in protected]
    removed = 0
    for stem in prunable[keep:]:
        for f in d.glob(stem + ".*"):
            try:
                f.unlink()
            except OSError:
                pass
        removed += 1
    return removed


def setup_logging(
    debug: bool = False,
    *,
    log_dir: Optional[str] = None,
    run_id: Optional[str] = None,
    console: bool = True,
) -> RunLog:
    """Wire the ``speaker`` logger to a non-blocking queue feeding a background
    listener (rich DEBUG file + console + summary aggregation). Returns a
    :class:`RunLog`; its ``finalize()`` also runs at interpreter exit.

    ``log_dir`` defaults to ``$SPEAKER_RUN_LOG_DIR`` or ``logs/runs`` (the env
    override lets tests redirect the bundle to a tmp dir)."""
    import os

    log_dir = log_dir or os.environ.get("SPEAKER_RUN_LOG_DIR", "logs/runs")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    # Keep the bundle dir condensed: retain only the newest N runs (env override).
    prune_old_runs(log_dir, int(os.environ.get("SPEAKER_KEEP_RUNS", "20")))
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

    # Bounded (backlog: log-storm backpressure): deep enough that only a genuine
    # storm with a stalled listener ever fills it; overflow policy lives in
    # _ThreadQueueHandler.enqueue (count + coalesce, WARNING+ gets a grace put).
    log_q: "queue.Queue" = queue.Queue(maxsize=8192)
    listener = QueueListener(log_q, *handlers, respect_handler_level=True)
    listener.start()

    root = logging.getLogger("speaker")
    root.setLevel(logging.DEBUG)
    root.propagate = False
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(_ThreadQueueHandler(log_q))

    runlog = RunLog(run_id, log_path, summary_path, summary, root, listener, handlers)
    atexit.register(runlog.finalize)
    root.info("run %s started (debug=%s) -> %s", run_id, debug, log_path)
    return runlog
