"""RealUsageRun: drive the user's recordings through the REAL assistant with the
REAL laptop audio OUTPUT, capturing the three live-failure signals.

WHY this is NOT FileReplayEngine: FileReplayEngine (core/engines/file_replay.py)
is headless -- its speak() synthesizes with the real TTS but DISCARDS the clip
and never opens an sd.OutputStream. So it exercises real STT + real LLM + real
TTS-synthesis but NOT (a) the real ALSA OUTPUT path, (b) the playback thread's
blocking out.write(), or (c) shutdown of that thread. All three live failures
live exactly in the parts FileReplayEngine skips.

WHAT this does instead: it reuses tools/live_session's REAL-runtime build
(core.app.build_runtime over a recording SherpaOnnxEngine subclass) and does the
OPPOSITE of live_session's --inject path -- it keeps the REAL sd.OutputStream
(so ALSA + the playback thread + shutdown ARE exercised and the assistant speaks
aloud) but patches ONLY sd.InputStream with an InjectingInputStream that feeds
the user's recorded WAV into the recognizer in place of the live mic.

The three signals it observes (without editing sherpa.py, which another agent
owns):
  1. SHUTDOWN HANG  -- runtime.stop() is called on a worker thread and joined
     with a hard timeout; if the join times out, stop() is hung in the ALSA
     out.write() inside the play-thread (sherpa.py stop()/_playback_loop).
  2. BARGE-IN STORM -- on_barge_in fires are counted at the engine-callback
     boundary (every fire, even when cancel_all has nothing to cancel).
  3. broken OUTPUT  -- a logging.Handler captures ALSA/PortAudio warnings, and
     engine._running.is_set() is checked (cleared == playback loop crashed).
plus the ASR finals (STT quality) and the assistant's spoken responses.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

log = logging.getLogger("speaker.real_usage")

# Log lines that mean the real audio OUTPUT path is broken. Best-effort regex --
# paired with the engine._running liveness check so a crashed playback loop is
# caught even if the wording changes.
_PLAYBACK_TROUBLE_RE = re.compile(
    r"alsa|portaudio|playback loop crashed|underrun|mmap|output stream|out\.write",
    re.IGNORECASE,
)


class _PlaybackErrorWatcher(logging.Handler):
    """A logging handler installed on the 'speaker' logger for the duration of a
    fixture run; it records any WARNING/ERROR matching the playback-trouble regex
    into ``self.errors``. Observation-only -- never edits the engine."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.errors: list[str] = []
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001 - a bad format string must not crash logging
            return
        if _PLAYBACK_TROUBLE_RE.search(msg or ""):
            with self._lock:
                self.errors.append(msg)

    def snapshot(self) -> list[str]:
        with self._lock:
            return list(self.errors)


def run_stop_with_timeout(stop_callable, timeout: float) -> tuple[bool, float]:
    """Call ``stop_callable()`` on a daemon worker thread and join it with a hard
    ``timeout``. Returns ``(completed_ok, elapsed_seconds)``.

    This is the regression test for the SHUTDOWN HANG: if stop() is stuck in a
    blocking C-level ALSA out.write() inside the play thread, runtime.stop()'s
    own join(timeout=1.0) times out but stop() still never returns -- so OUR
    join here also times out and we report ``completed_ok=False``. We cannot
    force-kill the C-level write from Python, so on timeout the worker is left
    as a wedged daemon thread (the caller decides whether to os._exit). This
    function is PURE (no audio/models) so it is unit-testable with a fake stop
    that hangs."""
    error: dict = {}

    def _worker() -> None:
        try:
            stop_callable()
        except Exception as exc:  # noqa: BLE001 - record, don't crash the harness
            error["exc"] = exc

    t = threading.Thread(target=_worker, name="real-usage-stop", daemon=True)
    start = time.perf_counter()
    t.start()
    t.join(timeout)
    elapsed = time.perf_counter() - start
    completed = not t.is_alive()
    if completed and "exc" in error:
        log.warning("runtime.stop() raised: %r", error["exc"])
    return completed, elapsed


def _make_signal_engine(config):
    """A SherpaOnnxEngine subclass that records what it speaks/stops (like
    live_session._RecordingEngine) AND taps the EngineCallbacks the runtime wires
    so we count on_barge_in fires and collect on_final transcripts at the engine
    boundary."""
    from core.engine import EngineCallbacks
    from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine

    class _SignalEngine(SherpaOnnxEngine):
        def __init__(self, cfg):
            super().__init__(cfg)
            self._rec_lock = threading.Lock()
            self.spoken: list[str] = []
            self._stops: list[float] = []
            self.barge_in_count = 0
            self.asr_finals: list[str] = []

        def start(self, callbacks: EngineCallbacks) -> None:
            # Wrap the runtime's callbacks so we tap the live signal stream
            # (runtime.start wires these). Each tap increments our counters then
            # delegates so the runtime sees the exact same events.
            orig_barge = callbacks.on_barge_in
            orig_final = callbacks.on_final

            def _on_barge_in() -> None:
                with self._rec_lock:
                    self.barge_in_count += 1
                if orig_barge is not None:
                    orig_barge()

            def _on_final(text: str) -> None:
                with self._rec_lock:
                    self.asr_finals.append((text or "").strip())
                if orig_final is not None:
                    orig_final(text)

            wrapped = EngineCallbacks(
                on_partial=callbacks.on_partial,
                on_final=_on_final,
                on_barge_in=_on_barge_in,
                on_command=callbacks.on_command,
                on_metric=callbacks.on_metric,
                on_heartbeat=callbacks.on_heartbeat,
                on_capture_state=callbacks.on_capture_state,
            )
            super().start(wrapped)

        def speak(self, text, on_done=None):
            with self._rec_lock:
                self.spoken.append((text or "").strip())
            super().speak(text, on_done)

        def stop_speaking(self):
            with self._rec_lock:
                self._stops.append(time.perf_counter())
            super().stop_speaking()

        def signals(self) -> dict:
            with self._rec_lock:
                return {
                    "spoken": list(self.spoken),
                    "asr_finals": list(self.asr_finals),
                    "barge_in_count": self.barge_in_count,
                }

    cfg = config if isinstance(config, SherpaConfig) else SherpaConfig.from_dict(config.get("sherpa", {}))
    return _SignalEngine(cfg), cfg


class RealUsageRun:
    """Drive ONE recording through the real assistant with real audio output.

    One RealUsageRun owns one VoiceRuntime -- the caller builds a fresh
    RealUsageRun per fixture (clean, independent shutdown test per recording).
    """

    # Quiet period (no task/speech) after the assistant engages before the turn
    # counts as idle. Bridges the brief task-complete -> engine.speak() gap.
    _IDLE_SETTLE = 0.8
    # How long to wait for the assistant to even BEGIN responding.
    _START_TIMEOUT = 12.0

    def __init__(
        self,
        config: dict,
        *,
        llm_backend: str = "ollama",
        main_model: Optional[str] = None,
        fast_model: Optional[str] = None,
        response_timeout: float = 60.0,
        shutdown_timeout: float = 8.0,
        open_mic: bool = False,
        warm_timeout: float = 20.0,
    ) -> None:
        from always_on_agent.events import Mode
        from core.app import build_runtime
        from core.llm_factory import build_llms
        from core.routing import build_router

        self._response_timeout = response_timeout
        self._shutdown_timeout = shutdown_timeout
        self._open_mic = open_mic
        self._warm_timeout = warm_timeout
        self._orig_input_stream = None
        self._inject_holder: dict = {}
        self._inject_stream = None
        self._watcher = _PlaybackErrorWatcher()
        self._error: Optional[str] = None
        # Liveness of the engine's playback loop, snapshotted AFTER the response
        # wait but BEFORE shutdown. _running is cleared both on a normal stop()
        # AND on a playback-loop crash (sherpa.py:1006), so it is only a crash
        # signal while we still expect the engine to be alive -- hence we sample
        # it here, not after shutdown.
        self._playback_alive_pre_shutdown: Optional[bool] = None

        self.engine, self._sherpa_cfg = _make_signal_engine(config)
        if not getattr(self._sherpa_cfg, "asr_encoder", "") or not getattr(self._sherpa_cfg, "asr_tokens", ""):
            raise RuntimeError(
                "sherpa ASR model paths are not configured (sherpa.asr_encoder / "
                "asr_tokens) -- the real-usage harness needs real models. See config.json."
            )

        args = SimpleNamespace(llm=llm_backend, model=main_model, fast_model=fast_model)
        llm, fast_llm = build_llms(args, config)
        router = build_router(config)
        self.runtime = build_runtime(
            config, engine=self.engine, llm=llm, fast_llm=fast_llm, router=router,
            start_mode=Mode.ASSISTANT, force_planner=False, force_stream_tts=False,
        )

    # --- lifecycle ---
    def start(self) -> None:
        # Install the playback-error log watcher for the run.
        logging.getLogger("speaker").addHandler(self._watcher)

        if not self._open_mic:
            # The OPPOSITE of live_session's --inject: patch ONLY sd.InputStream
            # so the WAV feeds the recognizer in place of the mic, but leave
            # sd.OutputStream REAL so the assistant speaks aloud and the real
            # ALSA output + playback thread + shutdown ARE exercised.
            import sounddevice as sd

            from tools.live_session.driver import InjectingInputStream

            holder: dict = {}

            def _input_factory(*args, samplerate=16000, **kwargs):
                stream = InjectingInputStream(int(samplerate) or 16000)
                holder["stream"] = stream  # last opened wins
                return stream

            self._orig_input_stream = sd.InputStream
            sd.InputStream = _input_factory
            self._inject_holder = holder

        self.runtime.start(run_bus=True)
        ready = getattr(self.runtime, "warm_ready", None)
        if ready is not None:
            ready.wait(timeout=self._warm_timeout)

        if not self._open_mic:
            self._inject_stream = self._inject_holder.get("stream")
            if self._inject_stream is None:
                raise RuntimeError("real-usage: the engine never opened an input stream to inject into")
            log.info("WAV-inject mode: feeding the recording into the recognizer at %d Hz "
                     "(real mic never opened; real speaker output IS used)", self._inject_stream._sr)
        log.info("assistant ready")

    def feed_wav(self, samples, sample_rate: int) -> None:
        """Push the recorded waveform into the recognizer (WAV-inject mode), or
        play it over the speakers (open-mic mode, so the real mic hears it)."""
        import numpy as np

        if self._open_mic:
            # Play the recording aloud so the real (open) mic captures it. This
            # is the only variant that can reproduce the acoustic-echo barge-in
            # STORM organically -- it is flakier and needs real speakers + mic.
            self._play_over_air(samples, sample_rate)
            return

        from tools.live_session.synthetic_user import _resample

        stream = self._inject_stream
        sr2 = stream._sr
        resampled = _resample(samples, sample_rate, sr2)
        # Prepend a ~0.4s noise-floor lead-in so the streaming recognizer settles
        # before the first word (else the onset is mis-decoded).
        lead = stream._noise[: int(0.4 * sr2)]
        stream.push(np.concatenate([lead, resampled]))

    def _play_over_air(self, samples, sample_rate: int) -> None:
        import numpy as np
        import sounddevice as sd

        out_dev = getattr(self._sherpa_cfg, "output_device", None)
        data = np.asarray(samples, dtype="float32").reshape(-1)
        try:
            sd.play(data, samplerate=sample_rate, device=out_dev)
        except Exception:  # noqa: BLE001 - over-air is best-effort
            log.exception("over-air playback of the recording failed")

    def wait_for_response(self) -> None:
        """Poll the supervisor + engine until the assistant has answered and gone
        idle (or the response timeout fires). This is where the assistant SPEAKS
        ALOUD and the real ALSA write path runs."""
        engaged = False
        last_activity = time.time()
        deadline = time.time() + self._response_timeout
        start_deadline = time.time() + self._START_TIMEOUT
        last_speak = self._spoken_count()
        while time.time() < deadline:
            active = self._has_work()
            speaking = self._is_speaking()
            sc = self._spoken_count()
            new_speech = sc > last_speak
            last_speak = sc
            if active or speaking or sc > 0:
                engaged = True
            if active or speaking or new_speech:
                last_activity = time.time()
            if engaged and not active and not speaking and (time.time() - last_activity) >= self._IDLE_SETTLE:
                self._snapshot_playback_liveness()
                return
            if not engaged and time.time() >= start_deadline:
                log.warning("no response observed within %.0fs; moving on", self._START_TIMEOUT)
                self._snapshot_playback_liveness()
                return
            time.sleep(0.03)
        log.warning("response timeout (%.0fs)", self._response_timeout)
        self._snapshot_playback_liveness()

    def _snapshot_playback_liveness(self) -> None:
        """Sample engine._running BEFORE shutdown clears it. Cleared here ==
        the playback loop crashed mid-run (sherpa.py:1006), i.e. the real
        out.write() failure path that goes mute. Observation-only."""
        running = getattr(self.engine, "_running", None)
        try:
            self._playback_alive_pre_shutdown = bool(running.is_set()) if running is not None else None
        except Exception:  # noqa: BLE001
            self._playback_alive_pre_shutdown = None

    def shutdown(self) -> tuple[bool, float]:
        """Stop the runtime under a hard timeout (catches the SHUTDOWN HANG).
        Returns ``(ok, seconds)``. Restores sd.InputStream and removes the log
        watcher regardless of whether stop() returned."""
        ok, seconds = run_stop_with_timeout(self.runtime.stop, self._shutdown_timeout)
        if not ok:
            log.error("SHUTDOWN HANG: runtime.stop() did not return within %.0fs "
                      "(play-thread join blocked in ALSA out.write)", self._shutdown_timeout)
        # Restore the patched input stream + remove the log watcher even on hang.
        if self._orig_input_stream is not None:
            import sounddevice as sd

            sd.InputStream = self._orig_input_stream
            self._orig_input_stream = None
        try:
            logging.getLogger("speaker").removeHandler(self._watcher)
        except Exception:  # noqa: BLE001
            pass
        return ok, seconds

    # --- signal collection ---
    def collect(self, shutdown_ok: bool, shutdown_seconds: float, *, fixture: str) -> dict:
        sig = self.engine.signals()
        return {
            "fixture": fixture,
            "asr_finals": [t for t in sig["asr_finals"] if t],
            "spoken": sig["spoken"],
            "first_audio_latencies": self._first_audio_latencies(),
            "barge_in_count": sig["barge_in_count"],
            "playback_errors": self._watcher.snapshot(),
            "playback_loop_dead": self._playback_loop_dead(),
            "shutdown_ok": shutdown_ok,
            "shutdown_seconds": round(shutdown_seconds, 3),
            "shutdown_timeout": self._shutdown_timeout,
            "error": self._error,
        }

    def _first_audio_latencies(self) -> list[float]:
        out: list[float] = []
        try:
            for rec in self.runtime.metrics.records():
                fa = rec.first_audio_latency
                if fa is not None:
                    out.append(round(fa, 4))
        except Exception:  # noqa: BLE001
            pass
        return out

    def _playback_loop_dead(self) -> bool:
        """True iff the engine's playback loop was already dead while the run was
        still live -- i.e. the real out.write() failure path crashed the loop
        (sherpa.py:1006 clears _running) BEFORE we ever asked it to stop. Read
        from the pre-shutdown snapshot, since a clean stop() also clears _running
        (so reading it post-shutdown is useless). Paired with the log watcher's
        'playback loop crashed' line for redundancy if the wording changes."""
        return self._playback_alive_pre_shutdown is False

    # --- observation helpers ---
    def _has_work(self) -> bool:
        try:
            st = self.runtime.supervisor.state
            return bool(st.active_tasks) or bool(st.queued_tasks)
        except Exception:  # noqa: BLE001
            return False

    def _is_speaking(self) -> bool:
        try:
            return bool(self.engine.is_speaking)
        except Exception:  # noqa: BLE001
            return False

    def _spoken_count(self) -> int:
        try:
            with self.engine._rec_lock:
                return len(self.engine.spoken)
        except Exception:  # noqa: BLE001
            return 0


def run_fixture(
    config: dict,
    wav_path: Path,
    *,
    llm_backend: str = "ollama",
    main_model: Optional[str] = None,
    fast_model: Optional[str] = None,
    response_timeout: float = 60.0,
    shutdown_timeout: float = 8.0,
    open_mic: bool = False,
) -> dict:
    """Run ONE recording end-to-end in this process and return its result dict.

    Build runtime -> patch sd.InputStream -> start -> feed the WAV -> wait for the
    spoken response -> shutdown under timeout -> collect signals. A fresh runtime
    per fixture isolates the shutdown test. The caller (CLI) typically runs this
    in a SUBPROCESS so a single hung fixture can't wedge the batch.
    """
    from core.engines.file_replay import load_waveform

    fixture = Path(wav_path).name
    run = RealUsageRun(
        config,
        llm_backend=llm_backend,
        main_model=main_model,
        fast_model=fast_model,
        response_timeout=response_timeout,
        shutdown_timeout=shutdown_timeout,
        open_mic=open_mic,
    )
    shutdown_ok, shutdown_seconds = True, 0.0
    try:
        run.start()
        samples, sr = load_waveform(str(wav_path))
        run.feed_wav(samples, sr)
        run.wait_for_response()
    except Exception as exc:  # noqa: BLE001 - record the harness error, still try to shut down
        log.exception("fixture %s failed mid-run", fixture)
        run._error = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            run.runtime.metrics.close_turn()
        except Exception:  # noqa: BLE001
            pass
        shutdown_ok, shutdown_seconds = run.shutdown()
    return run.collect(shutdown_ok, shutdown_seconds, fixture=fixture)
