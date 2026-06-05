"""LiveConversation: builds the REAL assistant (the same VoiceRuntime the CLI
builds, via core.app.build_runtime), drives a scenario by speaking the synthetic
user's lines through the speakers, observes the assistant's spoken responses +
per-turn latency, and records an attributed timeline.

Observing the assistant: a RecordingEngine subclass captures every sentence the
engine actually ``speak()``s (and every ``stop_speaking()``), so streamed AND
interrupted speech is captured exactly -- not just the completed-task text. An
assistant "turn" is the run of sentences spoken between user lines; it is flushed
into one attributed event (with latency + an interrupted flag) at the next
boundary.

Capture timing (decouples speaking from observing): after a user line is spoken,
poll until a condition set by the NEXT line's timing -- ``immediately`` -> don't
wait; ``barge_in`` -> until the assistant STARTS speaking (so the next line
interrupts mid-answer); else -> until the assistant is idle. "Idle" is honored
only AFTER the turn has been observed to BEGIN (a new transcript/metrics turn, a
task, or speech), with a start-timeout so a no-response line still terminates.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from .report import summarize_capture
from .synthetic_user import SyntheticUser, save_wav

log = logging.getLogger("speaker.live.driver")

_CONTROL_PREFIXES = (
    "Mode:", "Queued ", "[cancelled]", "Confirm command:", "Nothing to ",
    "Command cancelled.", "I can help with:",
)

# How long to wait for the assistant to even BEGIN responding before treating the
# line as un-answered (so a silent line still terminates within response_timeout).
_START_TIMEOUT = 8.0
# Quiet period (no task/speech) after engagement before a turn counts as idle --
# bridges the brief task-complete -> engine.speak() gap so it isn't read as done.
_IDLE_SETTLE = 0.6
# Max wait for a barge to actually register a stop after it's injected (the VAD
# accumulation lags the inject by ~1-1.5s). The interrupted answer is flushed as
# soon as the stop lands, or at this bound if it never does (a real no-stop).
_BARGE_STOP_TIMEOUT = 4.0


def _is_answer(text: str) -> bool:
    text = (text or "").strip()
    return bool(text) and not text.startswith(_CONTROL_PREFIXES)


class InjectingInputStream:
    """A fake capture stream that feeds pre-generated audio into the recognizer
    in place of the mic, paced at real-time (so endpoint timing + latency stay
    realistic). Used by --inject to test STT->LLM->TTS on CLEAN audio without the
    noisy over-the-air acoustic path (which on built-in laptop speaker+mic garbles
    STT and feeds back). Matches the read()/start/stop/close/abort surface the
    capture loop + recovering wrapper touch."""

    def __init__(self, sample_rate: int):
        import numpy as np

        self._np = np
        self._sr = int(sample_rate)
        self._buf = np.zeros(0, dtype="float32")
        self._lock = threading.Lock()
        # Between utterances, feed a low-level noise floor -- NOT digital zeros.
        # A real mic never delivers perfect silence; the streaming zipformer is
        # out-of-distribution on pure zeros and hallucinates filler words ("And")
        # on every endpoint cycle. ~3e-4 RMS sits above the "~silent" heartbeat
        # threshold (1e-4) and well below speech, so it decodes to nothing.
        # Precomputed + cycled (deterministic, no per-read RNG).
        self._noise = (np.random.default_rng(20260530).standard_normal(self._sr) * 3e-4).astype("float32")
        self._npos = 0

    def push(self, samples) -> None:
        s = self._np.asarray(samples, dtype="float32").reshape(-1)
        with self._lock:
            self._buf = self._np.concatenate([self._buf, s])

    def _floor(self, n: int):
        if n <= 0:
            return self._np.zeros(0, dtype="float32")
        reps = -(-n // len(self._noise))  # ceil
        pool = self._np.tile(self._noise, reps) if reps > 1 else self._noise
        start = self._npos % len(self._noise)
        self._npos = (self._npos + n) % len(self._noise)
        return pool[start:start + n].copy()

    def read(self, frames):
        frames = int(frames)
        with self._lock:
            if len(self._buf) >= frames:
                out, self._buf = self._buf[:frames], self._buf[frames:]
            else:
                out = self._np.concatenate([self._buf, self._floor(frames - len(self._buf))])
                self._buf = self._np.zeros(0, dtype="float32")
        time.sleep(frames / float(self._sr))  # pace like a real mic block read
        return out, False

    def start(self): pass
    def stop(self): pass
    def close(self): pass
    def abort(self): pass


class _NullOutputStream:
    """A drop-in for sd.OutputStream that discards audio but FAITHFULLY drives the
    PortAudio callback, paced at real-time. Used by --inject so the engine's
    callback-driven playback path runs exactly as on real hardware -- the audio
    callback drains the PlaybackFIFO and stamps TTS_FIRST_AUDIO -- without opening
    a finicky/blocking real output device (which on this box can stall for tens of
    seconds on a short clip).

    Emulating the callback is REQUIRED, not optional: the engine no longer pushes
    audio via ``out.write()``; the 2026-06-02 rewrite has PortAudio PULL audio from
    ``callback=self._audio_cb`` (which is where TTS_FIRST_AUDIO -- and thus every
    latency metric, since the driver gates latency on that stamp -- is recorded).
    A fake that only implements ``write()`` leaves the callback unfired, so the
    FIFO never drains, TTS_FIRST_AUDIO never stamps, and the run reports null
    latencies + a 'tts stuck' watchdog stall. So this fake spawns a thread that
    calls the callback with a zero buffer, paced at ``blocksize/samplerate``,
    exactly like a low-latency PortAudio device pulling from the FIFO."""

    def __init__(self, *args, samplerate: int = 48000, channels: int = 1,
                 callback=None, blocksize: int = 0, **kwargs):
        self._sr = int(samplerate) or 48000
        self._channels = int(channels) or 1
        self._callback = callback
        # ~10ms blocks: fine enough that TTS_FIRST_AUDIO stamps within a frame of
        # audio becoming available, paced like a low-latency device.
        self._block = int(blocksize) or max(1, self._sr // 100)
        self.active = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self.active = True
        if self._callback is None:
            return  # no callback to pull -> behaves as a pure sink
        self._stop.clear()
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(
                target=self._pump, name="inject-null-output", daemon=True)
            self._thread.start()

    def _pump(self):
        import numpy as np

        period = self._block / float(self._sr)
        buf = np.zeros((self._block, self._channels), dtype="float32")
        while not self._stop.is_set():
            buf[:] = 0.0  # PortAudio hands the callback a fresh buffer to fill
            try:
                self._callback(buf, self._block, None, None)
            except Exception:  # noqa: BLE001 - mirror PortAudio: a callback raise
                pass           # is swallowed, never kills the pump thread
            time.sleep(period)

    def write(self, samples):
        # Legacy blocking path (the callback engine never calls this); keep it a
        # real-time-paced no-op for any caller that still pushes.
        n = len(samples) if hasattr(samples, "__len__") else 0
        if n:
            time.sleep(n / float(self._sr))

    def _halt(self):
        self.active = False
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=0.5)
        self._thread = None

    def stop(self):
        self._halt()

    def abort(self):
        self._halt()

    def close(self):
        self._halt()


def make_recording_engine(config):
    """A SherpaOnnxEngine that records exactly what it speaks + stops, so the
    harness sees streamed and interrupted assistant speech."""
    from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine

    class _RecordingEngine(SherpaOnnxEngine):
        def __init__(self, cfg):
            super().__init__(cfg)
            self._rec_lock = threading.Lock()
            self._spoken: list[tuple[str, float]] = []
            self._stops: list[float] = []

        def speak(self, text, on_done=None):
            with self._rec_lock:
                self._spoken.append(((text or "").strip(), time.perf_counter()))
            super().speak(text, on_done)

        def stop_speaking(self):
            with self._rec_lock:
                self._stops.append(time.perf_counter())
            super().stop_speaking()

        def spoken_count(self) -> int:
            with self._rec_lock:
                return len(self._spoken)

        def spoken_since(self, n: int) -> list[tuple[str, float]]:
            with self._rec_lock:
                return list(self._spoken[n:])

        def stopped_after(self, t: float) -> bool:
            with self._rec_lock:
                return any(s >= t for s in self._stops)

    cfg = config if isinstance(config, SherpaConfig) else SherpaConfig.from_dict(config.get("sherpa", {}))
    return _RecordingEngine(cfg), cfg


class LiveConversation:
    def __init__(
        self,
        config: dict,
        *,
        llm_backend: str = "ollama",
        main_model: Optional[str] = None,
        fast_model: Optional[str] = None,
        out_dir: Path,
        user_speaker_id: Optional[int] = None,
        user_speed: Optional[float] = None,
        capture_assistant_audio: bool = True,
        response_timeout: float = 45.0,
        inject: bool = False,
        user_volume: Optional[float] = None,
        noise_snr_db: Optional[float] = None,
    ) -> None:
        from always_on_agent.events import Mode
        from core.app import build_runtime
        from core.llm_factory import build_llms
        from core.routing import build_router

        self._out = Path(out_dir)
        self._out.mkdir(parents=True, exist_ok=True)
        self._timeout = response_timeout
        self._inject = inject
        self._llm_backend = llm_backend
        self._llm_host = ((config.get("llm", {}) or {}).get("host") or "http://127.0.0.1:11434")
        # Driver-side post-speech cooldown. config.sherpa.post_speech_cooldown_sec
        # is DEAD config in core (never read), so the harness enforces the settle
        # itself: after the assistant goes idle we wait this long before the NEXT
        # user line so the assistant tail + room reverb decays and two voices never
        # mix into one capture segment. Skipped on overlapping timings.
        self._post_speech_cooldown = float(
            (config.get("sherpa", {}) or {}).get("post_speech_cooldown_sec", 0.6) or 0.0
        )
        self._inject_stream: Optional[InjectingInputStream] = None
        self._orig_output_stream = None  # saved sd.OutputStream while inject patches it
        self._orig_input_stream = None   # saved sd.InputStream while inject patches it
        self.engine, sherpa_cfg = make_recording_engine(config)
        if not getattr(sherpa_cfg, "asr_encoder", "") or not getattr(sherpa_cfg, "asr_tokens", ""):
            raise RuntimeError(
                "sherpa ASR model paths are not configured (sherpa.asr_encoder / "
                "asr_tokens) -- the live harness needs real models. See config.json."
            )
        if hasattr(self.engine, "set_record_path"):
            self.engine.set_record_path(str(self._out / "heard_over_air.wav"))

        args = SimpleNamespace(llm=llm_backend, model=main_model, fast_model=fast_model)
        llm, fast_llm = build_llms(args, config)
        router = build_router(config)
        self.runtime = build_runtime(
            config, engine=self.engine, llm=llm, fast_llm=fast_llm, router=router,
            start_mode=Mode.ASSISTANT, force_planner=False, force_stream_tts=False,
        )

        self.user = SyntheticUser(
            sherpa_cfg, speaker_id=user_speaker_id, speed=user_speed,
            output_device=getattr(sherpa_cfg, "output_device", None),
            volume=1.0 if user_volume is None else float(user_volume),
            noise_snr_db=noise_snr_db,
        )
        self._assistant_voice = None
        if capture_assistant_audio:
            try:
                self._assistant_voice = SyntheticUser(
                    sherpa_cfg,
                    speaker_id=int(getattr(sherpa_cfg, "tts_speaker_id", 0) or 0),
                    speed=float(getattr(sherpa_cfg, "tts_speed", 1.0) or 1.0),
                )
            except Exception:  # noqa: BLE001
                log.warning("could not build assistant-voice TTS; skipping assistant audio")

        self.events: list[dict] = []
        self._t0 = 0.0
        self._uidx = 0
        self._aidx = 0
        self._flush_base = 0          # engine speak()s already flushed into events
        self._metrics_consumed = 0    # metrics turns already paired to an answer
        # per-line baselines (snapshotted when a user line is spoken)
        self._ln_metrics = 0
        self._ln_transcript = 0
        self._ln_speak = 0
        self._cur_user_event: Optional[dict] = None
        # The timing of the most-recently-spoken user line. The barge-in grader
        # needs to know whether the line PRECEDING an assistant turn was an
        # intended barge (timing == "barge_in"); a barge that fires with no
        # intended barge before it is a self-interrupt. Carried forward here so
        # _flush_assistant can stamp it onto the assistant event.
        self._last_user_timing: Optional[str] = None
        # The timing of the line ABOUT to be spoken -- so a barge_in turn flags the
        # PRECEDING assistant answer it interrupts (the answer is barged by the NEXT
        # line, not the one before it).
        self._upcoming_user_timing: Optional[str] = None
        # Continuous-capture observation. We OBSERVE the engine's existing 2 s
        # heartbeat instead of instrumenting core: the runtime feeds every beat to
        # the watchdog's note_heartbeat, so we wrap that callable with a thin
        # harness-owned recorder (a public read, per the plan's risk note -- no
        # reach into watchdog internals). Each beat stamps a monotonic time and a
        # counter; a turn's max gap and the global beat count grade always-on
        # capture. _hb_silent_warned latches the watchdog's >5 s "capture silent".
        self._hb_lock = threading.Lock()
        self._hb_last: Optional[float] = None
        self._hb_count = 0
        self._hb_silent_warned = False
        self._wrap_heartbeat()
        self._wall_t0 = 0.0  # wall-clock session start (set in start())
        self.capture_verdict: Optional[dict] = None

    # --- continuous-capture observation (no core instrumentation) ---
    def _wrap_heartbeat(self) -> None:
        """Wrap the watchdog's note_heartbeat so every engine 2 s beat also stamps
        a harness-owned monotonic time + counter. This is the ALWAYS-ON proof: the
        beat fires from the engine's _capture_loop, so its freshness == the capture
        thread is alive and recording. We never read the watchdog's private state."""
        wd = getattr(self.runtime, "_watchdog", None)
        if wd is None or not hasattr(wd, "note_heartbeat"):
            return
        inner = wd.note_heartbeat

        def _wrapped() -> None:
            with self._hb_lock:
                self._hb_last = time.monotonic()
                self._hb_count += 1
            try:
                inner()
            except Exception:  # noqa: BLE001 - never break the engine's heartbeat
                log.debug("inner heartbeat hook raised", exc_info=True)

        wd.note_heartbeat = _wrapped

    def _hb_snapshot(self) -> tuple[Optional[float], int]:
        with self._hb_lock:
            return self._hb_last, self._hb_count

    # --- lifecycle ---
    def _free_llm_vram(self) -> None:
        """Unload any resident Ollama models so the CPU-side sherpa ONNX models
        build into a clean memory state.

        On a 16GB-VRAM laptop, two pinned gemma tiers (keep_alive=-1, ~14GB) leave
        the Windows GPU near-full, which commits enough *shared* system memory that
        the sherpa ASR/TTS onnxruntime load fails with 'bad allocation'. The harness
        builds a FRESH runtime per scenario, so models pinned by the PREVIOUS
        scenario would starve this scenario's engine build -- scenario 1 passes
        (empty GPU) and every later scenario in a suite dies. Freeing them here
        lets the engine load first; ``warm_on_start`` re-loads the LLM tiers AFTER
        the engine is built (the proven-working order). Best-effort: a missing /
        unreachable Ollama, or any error, is a silent no-op."""
        if self._llm_backend != "ollama":
            return
        try:
            import requests

            host = self._llm_host.rstrip("/")
            ps = requests.get(f"{host}/api/ps", timeout=3).json()
            models = ps.get("models", [])
            resident_gb = sum(m.get("size_vram", 0) for m in models) / 1e9
            # Only EVICT a heavy resident set. On this 16GB Windows box the
            # CPU-side sherpa ONNX load `bad alloc`s FLAKILY when the GPU is more
            # than ~half full (Windows commits shared system memory for a near-full
            # GPU and the commit charge varies run to run). A light set (e.g. 4b+1b
            # ~5.7GB) coexists with sherpa fine, so leaving it WARM avoids a slow
            # cold reload (and the >12GB churn that degraded a long suite). A heavy
            # set (12b-resident, ~10GB+) is unloaded so the engine builds into a
            # clean GPU; warm_on_start reloads the tiers after, and start() waits
            # the warmup out so turn 1 isn't stalled.
            if resident_gb <= 8.0:
                return
            for m in models:
                name = m.get("name")
                if name:
                    requests.post(
                        f"{host}/api/generate",
                        json={"model": name, "keep_alive": 0, "prompt": ""},
                        timeout=5,
                    )
            # Wait (bounded) for the driver to actually RELEASE the VRAM before the
            # engine's onnxruntime models allocate -- ``keep_alive=0`` evicts
            # asynchronously, so loading sherpa immediately would still race a
            # near-full GPU.
            for _ in range(20):
                if not requests.get(f"{host}/api/ps", timeout=3).json().get("models"):
                    break
                time.sleep(0.25)
        except Exception:  # noqa: BLE001 - VRAM hygiene is strictly best-effort
            log.debug("could not free Ollama VRAM before start (continuing)", exc_info=True)

    def start(self) -> None:
        # Free any VRAM pinned by a previous scenario BEFORE the engine builds its
        # CPU-side sherpa models (see _free_llm_vram); warm_on_start reloads the
        # LLM tiers afterwards.
        self._free_llm_vram()
        if self._inject:
            # Patch the audio device seams BEFORE start() so the engine never
            # touches the real hardware:
            #  - sd.InputStream -> an InjectingInputStream we keep a handle to, so
            #    the mic is never opened (its flaky open/close crashes this box and
            #    inject doesn't need it) and user audio feeds straight to the ASR.
            #  - sd.OutputStream -> a null sink (assistant audio isn't needed; we
            #    re-synthesize the track + read latency from metrics) so a blocking
            #    ALSA output can't stall the run. Patched here so warm-up uses it too.
            import sounddevice as sd

            holder: dict = {}

            def _input_factory(*args, samplerate=16000, **kwargs):
                stream = InjectingInputStream(int(samplerate) or 16000)
                holder["stream"] = stream  # last opened wins == the one that sticks
                return stream

            self._orig_input_stream = sd.InputStream
            self._orig_output_stream = sd.OutputStream
            sd.InputStream = _input_factory
            sd.OutputStream = _NullOutputStream
            self._inject_holder = holder
        self.runtime.start(run_bus=True)
        self._t0 = time.perf_counter()
        # Wall-clock anchor for the continuous-capture verdict: the engine's
        # WavRecorder begins accumulating frames inside runtime.start(), so the
        # recording duration vs (now - _wall_t0) is the ground truth that capture
        # never paused (the recorder writes EVERY block, even during playback).
        self._wall_t0 = time.monotonic()
        ready = getattr(self.runtime, "warm_ready", None)
        if ready is not None:
            # Generous wait: _free_llm_vram evicted the LLM tiers, so warm_on_start
            # COLD-reloads them here (main ~10GB). Under concurrent load that warmup
            # can take tens of seconds; waiting it out means turn 1 runs on warm
            # models instead of stalling behind a still-loading main tier (which
            # blew past the old 20s wait and broke first-turn latency attribution).
            ready.wait(timeout=120.0)
        if self._inject:
            self._inject_stream = self._inject_holder.get("stream")
            if self._inject_stream is None:
                raise RuntimeError("inject mode: the engine never opened an input stream")
            log.info("INJECT mode: feeding synthetic-user audio into the recognizer "
                     "at %d Hz (real mic never opened)", self._inject_stream._sr)
        log.info("assistant ready; starting conversation")

    def stop(self) -> None:
        try:
            self.runtime.stop()
        except Exception:  # noqa: BLE001
            log.exception("runtime stop failed")
        if self._orig_output_stream is not None or self._orig_input_stream is not None:
            import sounddevice as sd

            if self._orig_output_stream is not None:
                sd.OutputStream = self._orig_output_stream
                self._orig_output_stream = None
            if self._orig_input_stream is not None:
                sd.InputStream = self._orig_input_stream
                self._orig_input_stream = None

    def _now(self) -> float:
        return time.perf_counter() - self._t0

    # --- driving ---
    def run_scenario(self, scenario) -> list[dict]:
        turns = scenario.turns
        for i, turn in enumerate(turns):
            self._upcoming_user_timing = turn.timing
            # For a NORMAL line, flush the prior assistant turn before speaking. For
            # a barge_in line we DEFER the flush until AFTER the barge plays, so the
            # interrupted answer is captured WITH its interrupted flag set (the barge
            # calls stop_speaking only once it's injected -- flushing first would
            # always record interrupted=False).
            if turn.timing != "barge_in":
                self._flush_assistant()
            # Driver-side post-speech cooldown: after the PRIOR turn settled to
            # idle, let the assistant tail + room reverb decay before the next
            # user audio so two voices never mix into one capture segment. Only on
            # non-overlapping timings (barge_in/immediate are DESIGNED to overlap),
            # and never before the very first line.
            if (i > 0 and self._post_speech_cooldown > 0
                    and turn.timing not in ("barge_in", "immediately")):
                time.sleep(self._post_speech_cooldown)
            delay = _parse_pause(turn.timing)
            if delay:
                time.sleep(delay)
            barge_t = time.perf_counter() if turn.timing == "barge_in" else None
            self._speak_user(turn)
            if turn.timing == "barge_in":
                # The barge LAGS its injection by the VAD accumulation (~1-1.5s),
                # so a fixed short wait flushes the answer BEFORE the barge lands
                # (recording interrupted=False even though it did stop). Poll for
                # the engine to actually register a stop after the barge, up to a
                # bound, THEN flush -- so interrupted reflects reality.
                deadline = time.time() + _BARGE_STOP_TIMEOUT
                while time.time() < deadline and not self.engine.stopped_after(barge_t):
                    time.sleep(0.05)
                self._flush_assistant()
            self._capture_until(self._mode_for_next(turns, i))
            self._finalize_capture_probe()
        self._upcoming_user_timing = None
        self._flush_assistant()
        self.capture_verdict = self._compute_capture_verdict()
        return self.events

    @staticmethod
    def _mode_for_next(turns, i) -> str:
        nxt = turns[i + 1] if i + 1 < len(turns) else None
        if nxt is None:
            return "idle"
        if nxt.timing == "immediately":
            return "none"
        if nxt.timing == "barge_in":
            return "speaking"
        return "idle"

    def _speak_user(self, turn) -> None:
        self._uidx += 1
        self._last_user_timing = turn.timing
        self._ln_metrics = len(self.runtime.metrics.records())
        self._ln_transcript = len(self.runtime.supervisor.state.transcript_log)
        self._ln_speak = self.engine.spoken_count()
        # Continuous-capture probe baselines for THIS turn.
        hb_last, hb_count = self._hb_snapshot()
        # last_partial is never cleared between turns (supervisor only ever
        # overwrites it on a new partial), so snapshot it as a per-turn baseline
        # -- otherwise a stale value from an earlier turn reads as a live partial
        # on every poll even when the mic heard nothing this turn.
        st0 = self.runtime.supervisor.state
        self._probe = {
            "hb_count0": hb_count,
            "hb_last0": hb_last,
            "transcript0": self._ln_transcript,
            "partial0": (getattr(st0, "last_partial", "") or "").strip(),
            "partials_during_user": 0,   # partials seen WHILE the user audio plays
            "hb_gap_max_s": 0.0,         # worst heartbeat gap observed this turn
        }
        t_start = self._now()
        log.info("USER%s: %r", " (barge-in)" if turn.timing == "barge_in" else "", turn.text)
        if self._inject and self._inject_stream is not None:
            # Feed clean audio straight into the recognizer at the capture rate,
            # paced real-time by the injecting stream. No speakers, no mic.
            # Prepend a short noise-floor lead-in (like a real speaker's pre-speech
            # pause) so the streaming recognizer settles before the onset -- without
            # it the first word is mis-decoded ("What's" -> "Once"/"Was").
            import numpy as np

            from .synthetic_user import _resample

            samples, sr = self.user.synthesize(turn.text)
            sr2 = self._inject_stream._sr
            # Slice the precomputed noise pool directly (no _npos mutation -> no
            # race with the capture loop's read()/_floor).
            lead = self._inject_stream._noise[: int(0.4 * sr2)]
            played = np.concatenate([lead, _resample(samples, sr, sr2)])
            # Observe the recognizer WHILE the injected audio is being consumed:
            # the injecting stream paces itself real-time, so a partial appearing
            # before the buffer drains proves live transcription DURING input.
            # Non-blocking (auto-stops after the play duration) so inject's
            # push-and-return-immediately timing -- which the immediate/barge_in
            # merge cases depend on -- is unchanged.
            play_dur = len(played) / float(sr2)
            self._inject_stream.push(played)
            self._start_user_observer(duration_s=play_dur)
        else:
            # The acoustic path blocks until playback finishes; observe in parallel
            # so a partial produced while the speaker is playing the user line is
            # proof the mic transcribed live (full-duplex), not after the fact.
            stop = self._start_user_observer()
            try:
                samples, sr = self.user.say(turn.text)
            finally:
                stop()
        audio_path = self._out / "user" / f"{self._uidx:02d}.wav"
        save_wav(samples, sr, audio_path)
        self._cur_user_event = {
            "idx": self._uidx, "speaker": "user", "text": turn.text, "timing": turn.timing,
            "audio": str(audio_path.relative_to(self._out)),
            "t_start": round(t_start, 3), "t_end": round(self._now(), 3), "asr_final": None,
        }
        self.events.append(self._cur_user_event)

    # --- continuous-capture probing during the user's audio ---
    def _poll_capture_once(self, ref_last: Optional[float]) -> None:
        """One observation pass: count a live partial / transcript growth and
        track the worst heartbeat gap. ``ref_last`` is the heartbeat time at the
        previous pass (or line start) used to measure the gap to the latest beat."""
        st = self.runtime.supervisor.state
        partial = (getattr(st, "last_partial", "") or "").strip()
        # Only a partial that DIFFERS from this turn's baseline is a live partial;
        # an unchanged value is the stale carry-over from a previous turn.
        fresh_partial = bool(partial) and partial != self._probe["partial0"]
        grew = len(st.transcript_log) > self._probe["transcript0"]
        if fresh_partial or grew:
            self._probe["partials_during_user"] += 1
        hb_last, _ = self._hb_snapshot()
        now = time.monotonic()
        # Gap = time since the most recent beat. A capture loop alive on its 2 s
        # cadence keeps this small; a stalled thread lets it grow unbounded.
        last = hb_last if hb_last is not None else ref_last
        if last is not None:
            self._probe["hb_gap_max_s"] = max(self._probe["hb_gap_max_s"], now - last)

    def _start_user_observer(self, duration_s: Optional[float] = None):
        """Background observer of capture state while the user line is in flight.

        A daemon thread polls partial/transcript growth + heartbeat freshness so a
        partial appearing WHILE the user audio plays proves live (full-duplex)
        transcription. Two modes:

        * acoustic (``say()`` blocks): called with no duration; returns a stop()
          callable the caller invokes when playback finishes.
        * inject (push-and-return-immediately): called with ``duration_s``; the
          thread self-terminates after that, so inject's timing is unchanged and
          no stop() handle is needed."""
        stop_evt = threading.Event()
        ref0 = self._probe.get("hb_last0")
        end = (time.monotonic() + duration_s) if duration_s is not None else None

        def _run() -> None:
            while not stop_evt.is_set():
                self._poll_capture_once(ref0)
                if end is not None and time.monotonic() >= end:
                    return
                stop_evt.wait(0.03)

        th = threading.Thread(target=_run, name="live-capture-probe", daemon=True)
        th.start()

        def _stop() -> None:
            stop_evt.set()
            th.join(timeout=1.0)

        return _stop

    def _finalize_capture_probe(self) -> None:
        """Attach the per-turn capture block to the current user event once its
        turn has finished (assistant idle / barge / timeout)."""
        if self._cur_user_event is None or not getattr(self, "_probe", None):
            return
        _, hb_count = self._hb_snapshot()
        finals = len(self.runtime.supervisor.state.transcript_log) - self._probe["transcript0"]
        self._cur_user_event["capture"] = {
            "partials_during_user": int(self._probe["partials_during_user"]),
            "finals": int(max(0, finals)),
            "heartbeats": int(hb_count - self._probe["hb_count0"]),
            "heartbeat_gap_max_s": round(float(self._probe["hb_gap_max_s"]), 3),
            "capture_silent_warned": bool(self._hb_silent_warned),
        }

    def _compute_capture_verdict(self) -> dict:
        """Session-level full-duplex / continuous-capture verdict. PASS when the
        recorder covered ~the whole wall-clock session AND no >5 s capture-silent
        gap was seen AND at least one partial appeared while a user line played."""
        rec = getattr(self.engine, "_recorder", None)
        rec_secs = float(getattr(rec, "seconds", 0.0) or 0.0)
        wall = max(0.0, time.monotonic() - (self._wall_t0 or time.monotonic()))
        partials_total = sum(
            (e.get("capture") or {}).get("partials_during_user", 0)
            for e in self.events if e.get("speaker") == "user"
        )
        return summarize_capture(
            rec_seconds=rec_secs,
            wall_seconds=wall,
            partials_during_user_total=int(partials_total),
            capture_silent_warned=bool(self._hb_silent_warned),
        )

    def _capture_until(self, mode: str) -> None:
        if mode == "none":
            return
        # "Engaged" must mean the assistant actually picked up the work -- a task
        # in flight or speech -- NOT just a transcript (which also appears for an
        # INGEST'd / not-addressed turn, and BEFORE the task is created). Honor
        # idle only after engagement AND a short settle, so the brief task->speak
        # gap doesn't read as idle. A turn that never engages (INGEST) falls out
        # on the start timeout.
        engaged = False
        last_seen_speak = self._ln_speak
        last_activity = time.time()
        deadline = time.time() + self._timeout
        start_deadline = time.time() + _START_TIMEOUT
        while time.time() < deadline:
            self._attach_heard_transcript()
            # Keep observing capture through the turn (heartbeat gap + a latch for
            # the watchdog's >5 s "capture silent" condition). The recorder + the
            # ASR gate keep running during the assistant's OWN playback, so this
            # also proves capture never paused while the assistant was speaking.
            self._observe_capture_health()
            active = self._has_work()
            speaking = self._is_speaking()
            sc = self.engine.spoken_count()
            # NEW speech is an EDGE -- spoken_count only ever grows, so comparing
            # to the line baseline (spoke-since-line) is a LEVEL that would never
            # let last_activity settle once the assistant has spoken at all. Refresh
            # activity on a fresh sentence only; keep "engaged" on the level.
            new_speech = sc > last_seen_speak
            last_seen_speak = sc
            spoke_since_line = sc > self._ln_speak
            if active or speaking or spoke_since_line:
                engaged = True
            if active or speaking or new_speech:
                last_activity = time.time()
            if mode == "speaking" and (speaking or spoke_since_line):
                return
            if mode == "idle":
                if engaged and not active and not speaking and (time.time() - last_activity) >= _IDLE_SETTLE:
                    return
                if not engaged and time.time() >= start_deadline:
                    log.warning("no response observed within %.0fs; moving on", _START_TIMEOUT)
                    return
            time.sleep(0.03)
        log.warning("response timeout (%.0fs, mode=%s)", self._timeout, mode)

    # --- observation helpers ---
    def _has_work(self) -> bool:
        st = self.runtime.supervisor.state
        return bool(st.active_tasks) or bool(st.queued_tasks)

    def _is_speaking(self) -> bool:
        try:
            return bool(self.engine.is_speaking)
        except Exception:  # noqa: BLE001
            return False

    def _idle(self) -> bool:
        return not self._has_work() and not self._is_speaking()

    # Mirror the watchdog's silence deadline (CAPTURE_SILENT_DEADLINE_SEC=5.0)
    # without importing it, so the harness latches the same "capture silent"
    # condition the watchdog would log.
    _CAPTURE_SILENT_DEADLINE_SEC = 5.0

    def _observe_capture_health(self) -> None:
        """Track the worst heartbeat gap during a turn and latch the capture-silent
        condition. Once a beat has been seen, a gap past the deadline means the
        capture thread stalled -- the negation of the continuous-capture proof."""
        hb_last, _ = self._hb_snapshot()
        if hb_last is None:
            return
        gap = time.monotonic() - hb_last
        probe = getattr(self, "_probe", None)
        if probe is not None:
            probe["hb_gap_max_s"] = max(probe["hb_gap_max_s"], gap)
        if gap >= self._CAPTURE_SILENT_DEADLINE_SEC:
            self._hb_silent_warned = True

    def _attach_heard_transcript(self) -> None:
        log_ = self.runtime.supervisor.state.transcript_log
        if len(log_) <= self._ln_transcript or self._cur_user_event is None:
            return
        heard = log_[self._ln_transcript:]
        self._ln_transcript = len(log_)
        prev = self._cur_user_event.get("asr_final")
        merged = (prev + " | " if prev else "") + " | ".join(heard)
        self._cur_user_event["asr_final"] = merged

    def _flush_assistant(self) -> None:
        """Record everything the engine spoke since the last flush as ONE
        attributed assistant event (joined text, first-audio time, latency,
        interrupted flag). No-op when the assistant said nothing."""
        spoken = self.engine.spoken_since(self._flush_base)
        spoken = [(t, ts) for (t, ts) in spoken if _is_answer(t)]
        if not spoken:
            self._flush_base = self.engine.spoken_count()
            return
        self._flush_base = self.engine.spoken_count()
        self._aidx += 1
        text = " ".join(t for t, _ in spoken)
        t_first = round(spoken[0][1] - self._t0, 3)
        interrupted = self.engine.stopped_after(spoken[0][1])
        latency = self._consume_latency()
        audio_rel = None
        if self._assistant_voice is not None:
            try:
                a_samples, a_sr = self._assistant_voice.synthesize(text)
                a_path = self._out / "assistant" / f"{self._aidx:02d}.wav"
                save_wav(a_samples, a_sr, a_path)
                audio_rel = str(a_path.relative_to(self._out))
            except Exception:  # noqa: BLE001
                log.debug("assistant-audio re-synth failed", exc_info=True)
        log.info("ASSISTANT%s: %r (first_audio=%s)",
                 " [interrupted]" if interrupted else "", text[:80],
                 latency.get("first_audio_latency") if latency else None)
        self.events.append({
            "idx": self._aidx, "speaker": "assistant", "text": text, "audio": audio_rel,
            "t_start": t_first, "interrupted": interrupted, "latency": latency,
            # Did the scenario INTEND to barge before this answer? The barge-in
            # grader uses this to separate a real interrupt (intended) from a
            # self-interrupt (a barge fired with no intended barge). barge_in_ms
            # is derivable in report.py from latency.barge_in_latency, so no extra
            # field is surfaced here -- keep the driver surface minimal.
            "barge_intended": getattr(self, "_upcoming_user_timing", None) == "barge_in",
        })

    def _consume_latency(self) -> Optional[dict]:
        """Pair this answer with the next unconsumed metrics turn that the model
        actually answered (in order).

        Pairs on LLM_FIRST_TOKEN, not TTS_FIRST_AUDIO: every answered turn stamps
        a first token, but the first-audio stamp is TTS-side and -- in --inject
        mode, where a Python pump thread emulates the PortAudio callback -- can be
        missed under load (the pump gets GIL-starved past the watchdog deadline).
        Gating on first-audio there silently DROPPED the whole turn's latency
        (endpoint + final->token, the reliable ASR/LLM response-speed numbers) just
        because the racy audio stamp didn't land. as_dict() still carries
        first_audio_latency when present and None when not."""
        from core.metrics import LLM_FIRST_TOKEN

        recs = self.runtime.metrics.records()
        for i in range(self._metrics_consumed, len(recs)):
            if LLM_FIRST_TOKEN in recs[i].stamps:
                self._metrics_consumed = i + 1
                return recs[i].as_dict()
        return None


def _parse_pause(timing: str) -> float:
    if isinstance(timing, str) and timing.startswith("pause:"):
        try:
            return float(timing.split(":", 1)[1].rstrip("s"))
        except (ValueError, IndexError):
            return 0.0
    return 0.0
