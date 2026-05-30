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
    """A drop-in for sd.OutputStream that discards audio, paced at real-time so
    ``is_speaking`` still reflects a realistic speaking duration. Used by --inject
    so the assistant's TTS metrics (TTS_FIRST_AUDIO) still fire but a blocking /
    finicky real audio OUTPUT can't stall the harness (on this machine ALSA
    out.write() blocks for tens of seconds on a short clip)."""

    def __init__(self, *args, samplerate: int = 48000, **kwargs):
        self._sr = int(samplerate) or 48000
        self.active = True

    def start(self):
        self.active = True

    def write(self, samples):
        n = len(samples) if hasattr(samples, "__len__") else 0
        if n:
            time.sleep(n / float(self._sr))

    def stop(self):
        self.active = False

    def abort(self):
        self.active = False

    def close(self):
        self.active = False


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
    ) -> None:
        from always_on_agent.events import Mode
        from core.app import build_runtime
        from core.llm_factory import build_llms
        from core.routing import build_router

        self._out = Path(out_dir)
        self._out.mkdir(parents=True, exist_ok=True)
        self._timeout = response_timeout
        self._inject = inject
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

    # --- lifecycle ---
    def start(self) -> None:
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
        ready = getattr(self.runtime, "warm_ready", None)
        if ready is not None:
            ready.wait(timeout=20.0)
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
            # Record the prior assistant turn (possibly interrupted) BEFORE we
            # speak the next line -- so a barged/partial answer isn't lost.
            self._flush_assistant()
            delay = _parse_pause(turn.timing)
            if delay:
                time.sleep(delay)
            self._speak_user(turn)
            self._capture_until(self._mode_for_next(turns, i))
        self._flush_assistant()
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
        self._ln_metrics = len(self.runtime.metrics.records())
        self._ln_transcript = len(self.runtime.supervisor.state.transcript_log)
        self._ln_speak = self.engine.spoken_count()
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
            self._inject_stream.push(np.concatenate([lead, _resample(samples, sr, sr2)]))
        else:
            samples, sr = self.user.say(turn.text)
        audio_path = self._out / "user" / f"{self._uidx:02d}.wav"
        save_wav(samples, sr, audio_path)
        self._cur_user_event = {
            "idx": self._uidx, "speaker": "user", "text": turn.text, "timing": turn.timing,
            "audio": str(audio_path.relative_to(self._out)),
            "t_start": round(t_start, 3), "t_end": round(self._now(), 3), "asr_final": None,
        }
        self.events.append(self._cur_user_event)

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
        })

    def _consume_latency(self) -> Optional[dict]:
        """Pair this answer with the next unconsumed metrics turn that produced
        first audio (in order)."""
        from core.metrics import TTS_FIRST_AUDIO

        recs = self.runtime.metrics.records()
        for i in range(self._metrics_consumed, len(recs)):
            if TTS_FIRST_AUDIO in recs[i].stamps:
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
