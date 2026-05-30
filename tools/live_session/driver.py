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


def _is_answer(text: str) -> bool:
    text = (text or "").strip()
    return bool(text) and not text.startswith(_CONTROL_PREFIXES)


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
    ) -> None:
        from always_on_agent.events import Mode
        from core.app import build_runtime
        from core.llm_factory import build_llms
        from core.routing import build_router

        self._out = Path(out_dir)
        self._out.mkdir(parents=True, exist_ok=True)
        self._timeout = response_timeout
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
        self.runtime.start(run_bus=True)
        self._t0 = time.perf_counter()
        ready = getattr(self.runtime, "warm_ready", None)
        if ready is not None:
            ready.wait(timeout=20.0)
        log.info("assistant ready; starting conversation")

    def stop(self) -> None:
        try:
            self.runtime.stop()
        except Exception:  # noqa: BLE001
            log.exception("runtime stop failed")

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
        started = False
        deadline = time.time() + self._timeout
        start_deadline = time.time() + _START_TIMEOUT
        while time.time() < deadline:
            self._attach_heard_transcript()
            if not started:
                started = self._turn_began()
            if mode == "speaking" and self._started_speaking():
                return
            if mode == "idle":
                if started and self._idle():
                    return
                if not started and time.time() > start_deadline:
                    log.warning("no response observed within %.0fs; moving on", _START_TIMEOUT)
                    return
            time.sleep(0.03)
        log.warning("response timeout (%.0fs, mode=%s)", self._timeout, mode)

    # --- observation helpers ---
    def _turn_began(self) -> bool:
        st = self.runtime.supervisor.state
        try:
            speaking = self.engine.is_speaking
        except Exception:  # noqa: BLE001
            speaking = False
        return (
            self.engine.spoken_count() > self._ln_speak
            or len(st.transcript_log) > self._ln_transcript
            or len(self.runtime.metrics.records()) > self._ln_metrics
            or bool(st.active_tasks) or bool(st.queued_tasks) or speaking
        )

    def _started_speaking(self) -> bool:
        try:
            if self.engine.is_speaking:
                return True
        except Exception:  # noqa: BLE001
            pass
        return self.engine.spoken_count() > self._ln_speak

    def _idle(self) -> bool:
        st = self.runtime.supervisor.state
        try:
            speaking = self.engine.is_speaking
        except Exception:  # noqa: BLE001
            speaking = False
        return not st.active_tasks and not st.queued_tasks and not speaking

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
