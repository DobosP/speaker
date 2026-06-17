"""Live watchdog: warns when the pipeline appears stuck mid-turn.

The post-hoc ``stuck_hints`` in :mod:`core.runlog` only flag things visible
after the run ended (no LLM requests, all cancelled, errors present). The
watchdog runs while the session is live, inspects the metrics anchors and
engine heartbeat once a second, and emits WARNING log lines when:

* an ``asr_final`` has fired but ``llm_first_token`` hasn't, after a deadline
  (the LLM took the prompt but never produced a token);
* ``llm_first_token`` has fired but ``tts_first_audio`` hasn't, after a deadline
  (the LLM streamed tokens but TTS never produced audio);
* the engine's capture heartbeat has been silent for too long
  (capture thread crashed or stalled);
* many ``barge-in detected`` events fire in a short window
  (barge-in gate is flapping; TTS likely leaking through the mic, or the
  cancel path doesn't actually stop playback fast enough).

WARNINGs land in the run bundle's ``errors[]`` already (folded in by
:class:`core.runlog._SummaryHandler`), and the named hints in
:meth:`core.runlog.RunSummary.to_dict` promote each of these to a top-level
``stuck_hints`` entry -- so the next stuck reproduction leaves visible
evidence in ``summary.json`` instead of looking clean.

The watchdog diagnoses; it does not heal. A self-healing strategy without
evidence of the failure mode would just be guessing.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Callable, Optional

from .metrics import (
    ASR_FINAL,
    BARGE_IN,
    BARGE_IN_STOP,
    HANDLED_LOCAL,
    LLM_FIRST_TOKEN,
    MetricsRecorder,
    SUPERSEDED,
    TTS_FIRST_AUDIO,
)

log = logging.getLogger("speaker.watchdog")


class StuckWatchdog:
    """Background daemon that periodically inspects pipeline state and warns
    when a turn appears stalled. See module docstring for the heuristics.

    The watchdog shares a clock with :class:`MetricsRecorder` so deltas
    against its stamps are well-defined. Tests can pass a fake clock + drive
    :meth:`tick` directly without sleeping.
    """

    # Public knobs (overridable per-instance from tests).
    LLM_FIRST_TOKEN_DEADLINE_SEC = 10.0
    TTS_FIRST_AUDIO_DEADLINE_SEC = 5.0
    CAPTURE_SILENT_DEADLINE_SEC = 5.0
    BARGE_IN_STORM_WINDOW_SEC = 1.5
    BARGE_IN_STORM_THRESHOLD = 3

    def __init__(
        self,
        recorder: MetricsRecorder,
        *,
        interval_sec: float = 1.0,
        clock: Optional[Callable[[], float]] = None,
        on_storm: Optional[Callable[[], None]] = None,
        on_tick: Optional[Callable[[], None]] = None,
    ) -> None:
        self._recorder = recorder
        self._interval = interval_sec
        # Default matches MetricsRecorder's clock so per-turn deltas are
        # comparable. Pass a controlled clock from tests.
        self._clock = clock or time.perf_counter
        # Optional reaction hook fired once per detected barge-in storm. The
        # engine can wire its brief barge-in debounce here so a flapping VAD
        # gate (TTS leaking into the mic, no AEC) collapses into one interrupt
        # instead of a rattling string of them. Diagnosis still logs as before.
        self._on_storm = on_storm
        # Periodic maintenance hook fired once per tick (the runtime wires it to
        # the supervisor's overdue-task reap, so a hung task is killed on the same
        # 1 s cadence the watchdog already runs -- the controller "heals" instead
        # of only diagnosing). Guarded so a hook error never kills the loop.
        self._on_tick = on_tick
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._warned: set[tuple[int, str]] = set()
        self._last_heartbeat: Optional[float] = None
        self._heartbeat_warned: bool = False
        self._barge_ins: deque = deque()
        self._storm_warned_at: Optional[float] = None
        # Last reported capture-stream state from the engine ("open" /
        # "recovering" / "fatal"). The heartbeat check skips its warning
        # while we're known-recovering -- a 6 s reopen is intentional, not
        # a stall.
        self._capture_state: str = "open"
        self._capture_state_at: float = self._clock()

    # --- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        t = threading.Thread(target=self._loop, name="speaker-watchdog", daemon=True)
        self._thread = t
        t.start()

    def stop(self) -> None:
        self._stop.set()
        t, self._thread = self._thread, None
        if t is not None and t.is_alive():
            t.join(timeout=2.0)

    # --- hooks called from the engine / runtime threads ----------------------
    def note_heartbeat(self) -> None:
        """Engine reports the capture loop is alive (called from sherpa's
        existing 2s heartbeat)."""
        self._last_heartbeat = self._clock()
        # A fresh heartbeat re-arms the silence check so we'd warn again if
        # the thread goes silent a second time.
        self._heartbeat_warned = False

    def note_barge_in(self) -> None:
        """Runtime reports a barge-in event; the watchdog tracks the rate."""
        self._barge_ins.append(self._clock())

    def note_capture_state(self, state: str, message: str = "") -> None:
        """Runtime forwards the engine's capture-stream lifecycle.

        Recording this state has two effects:

        - the heartbeat check is suppressed while ``state == "recovering"``
          (a legitimate 6 s reopen is not a stalled audio thread);
        - on ``state == "fatal"`` we emit a one-shot ``log.error`` so the
          run bundle records that capture has truly stopped.
        """
        self._capture_state = state
        self._capture_state_at = self._clock()
        if state == "fatal":
            log.error("capture lost (device hardware): %s", message or "unknown")
        # Returning from "recovering" -> "open" re-arms the silence check.
        if state == "open":
            self._heartbeat_warned = False

    # --- inspection (one pass; tests drive this directly) --------------------
    def tick(self) -> None:
        now = self._clock()
        self._check_turns(now)
        self._check_heartbeat(now)
        self._check_barge_in_storm(now)
        if self._on_tick is not None:
            try:
                self._on_tick()
            except Exception:  # noqa: BLE001 - a maintenance hook must never kill the loop
                log.exception("watchdog on_tick hook raised")

    def _check_turns(self, now: float) -> None:
        for i, rec in enumerate(self._recorder.records()):
            stamps = rec.stamps
            # A turn the user barged into (or stopped via a "stop" command that
            # aborted playback), one resolved with no LLM at all (HANDLED_LOCAL:
            # the intent fast-path), or one PREEMPTED by a newer final
            # (SUPERSEDED: newest-input-wins cancelled it pre-answer) may
            # legitimately never reach llm_first_token or tts_first_audio -- that
            # is an interrupt / no-LLM / cancelled turn, not a stall. Skip both
            # stuck checks so it isn't mis-flagged as stuck (the live run surfaced
            # this false positive on a cancelled turn; rc-5 covers the no-LLM +
            # superseded cases).
            if (
                BARGE_IN in stamps or BARGE_IN_STOP in stamps
                or HANDLED_LOCAL in stamps or SUPERSEDED in stamps
            ):
                continue
            if ASR_FINAL in stamps and LLM_FIRST_TOKEN not in stamps:
                elapsed = now - stamps[ASR_FINAL]
                if elapsed >= self.LLM_FIRST_TOKEN_DEADLINE_SEC:
                    self._warn_once(
                        i, "llm_first_token",
                        "llm stuck: turn %d had asr_final but no llm_first_token after %.1fs"
                        % (i, elapsed),
                    )
            if LLM_FIRST_TOKEN in stamps and TTS_FIRST_AUDIO not in stamps:
                elapsed = now - stamps[LLM_FIRST_TOKEN]
                if elapsed >= self.TTS_FIRST_AUDIO_DEADLINE_SEC:
                    self._warn_once(
                        i, "tts_first_audio",
                        "tts stuck: turn %d had llm_first_token but no tts_first_audio after %.1fs"
                        % (i, elapsed),
                    )

    def _check_heartbeat(self, now: float) -> None:
        if self._last_heartbeat is None:
            return  # engine hasn't reported any heartbeats; not applicable
        if self._heartbeat_warned:
            return
        # While the engine is mid-recover, the absence of heartbeats is
        # expected -- the capture loop is sleeping during backoff. Don't
        # mis-attribute the gap as a stalled thread.
        if self._capture_state == "recovering":
            return
        elapsed = now - self._last_heartbeat
        if elapsed >= self.CAPTURE_SILENT_DEADLINE_SEC:
            self._heartbeat_warned = True
            log.warning(
                "capture silent: no heartbeat for %.1fs (audio thread crashed or stalled?)",
                elapsed,
            )

    def _check_barge_in_storm(self, now: float) -> None:
        cutoff = now - self.BARGE_IN_STORM_WINDOW_SEC
        while self._barge_ins and self._barge_ins[0] < cutoff:
            self._barge_ins.popleft()
        if len(self._barge_ins) < self.BARGE_IN_STORM_THRESHOLD:
            return
        # Suppress repeat warnings while the storm continues so we don't spam.
        if (
            self._storm_warned_at is not None
            and now - self._storm_warned_at < self.BARGE_IN_STORM_WINDOW_SEC
        ):
            return
        self._storm_warned_at = now
        log.warning(
            "barge-in storm: %d detections in the last %.1fs "
            "(gate flapping; TTS likely leaking into mic, or stop_speaking too slow)",
            len(self._barge_ins), self.BARGE_IN_STORM_WINDOW_SEC,
        )
        # Let a wired engine briefly debounce its barge-in gate. Best-effort:
        # the watchdog's job is diagnosis, so a misbehaving hook must not take
        # the loop down (and must not block it -- the hook is expected to be a
        # cheap, non-blocking flag set).
        if self._on_storm is not None:
            try:
                self._on_storm()
            except Exception:  # noqa: BLE001 - hook failure must not stop the watchdog
                log.exception("watchdog on_storm hook raised")

    @property
    def in_storm(self) -> bool:
        """Whether a barge-in storm was warned within the last window (cheap
        read for tests / a runtime that wants to poll rather than be called)."""
        if self._storm_warned_at is None:
            return False
        return (self._clock() - self._storm_warned_at) < self.BARGE_IN_STORM_WINDOW_SEC

    def _warn_once(self, turn_idx: int, key: str, msg: str) -> None:
        token = (turn_idx, key)
        if token in self._warned:
            return
        self._warned.add(token)
        log.warning(msg)

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self.tick()
            except Exception:  # noqa: BLE001
                log.exception("watchdog tick failed")
