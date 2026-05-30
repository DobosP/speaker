"""NoiseStressSession: drives the REAL assistant under generated noise + an
intruder voice, in two delivery modes, and observes the engine's own signals to
grade voice isolation honestly.

It COMPOSES ``tools.live_session.driver.LiveConversation`` (it does not edit it):
LiveConversation builds the same real ``VoiceRuntime`` the CLI builds, owns the
``InjectingInputStream`` seam, the synthetic user, and the metrics recorder. We:

  * wrap ``runtime.metrics.mark`` BEFORE start() with a thin counter that tallies
    ``speaker_rejected_final`` (gate drops) and ``SPEECH_END`` / ``ASR_FINAL`` /
    ``TTS_FIRST_AUDIO`` (a delivered, answered turn) -- the on_metric callback the
    engine fires is ``runtime.metrics.mark``, captured at start(), so replacing the
    attribute first makes the wrapper authoritative without touching core;

  * drive turns OURSELVES (not LiveConversation.run_scenario) so we control the
    EXACT audio per turn -- the mock-user clip MIXED with white/pink/babble at a
    target SNR, an intruder clip, or a pure-noise window -- and snapshot the metric
    counters around each turn's window to attribute answered-vs-rejected.

INJECT mode is the trustworthy filtering measurement (real mic/speaker never
touched). ACOUSTIC mode plays the mock user + looping background noise through the
REAL output device while the REAL mic captures -- the realism demo, flagged as
acoustic-confounded in the report.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("speaker.noise_stress.driver")

# How long to wait, after pushing a turn's audio, for the runtime to either
# answer it or drop it before we snapshot the window's outcome.
TURN_SETTLE_SEC = 6.0
# Quiet gap injected between turns so two turns never merge into one segment.
INTER_TURN_GAP_SEC = 1.0


class _MetricCounter:
    """Wraps ``MetricsRecorder.mark`` to tally the signals the grader needs.

    The engine's ``on_metric`` callback IS ``runtime.metrics.mark`` (see
    ``core/runtime.py``), and ``speaker_rejected_final`` is fired with no open
    turn, so the recorder itself may drop it -- we count it here, at the callback
    seam, where every fire is visible. All other stages pass straight through."""

    def __init__(self, inner_mark):
        self._inner = inner_mark
        self._lock = threading.Lock()
        self.counts: dict[str, int] = {}

    def __call__(self, stage, *args, **kwargs):
        with self._lock:
            self.counts[stage] = self.counts.get(stage, 0) + 1
        return self._inner(stage, *args, **kwargs)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self.counts)


class NoiseStressSession:
    """Owns a LiveConversation for one scenario run + the generated-noise mix."""

    def __init__(
        self,
        config: dict,
        *,
        mode: str,
        out_dir: Path,
        sherpa_overrides: dict,
        user_speaker_id: int,
        intruder_speaker_id: int,
        seed: int,
        llm_backend: str = "ollama",
        main_model: Optional[str] = None,
        fast_model: Optional[str] = None,
        response_timeout: float = 45.0,
        output_device=None,
        noise_volume: float = 1.0,
    ) -> None:
        import copy

        import numpy as np

        from tools.live_session.driver import LiveConversation

        if mode not in ("inject", "acoustic"):
            raise ValueError(f"mode must be inject|acoustic, got {mode!r}")
        self.mode = mode
        self._out = Path(out_dir)
        self._out.mkdir(parents=True, exist_ok=True)
        self._seed = int(seed)
        self._intruder_sid = int(intruder_speaker_id)
        self._noise_volume = float(noise_volume)
        self._np = np

        # Apply the speaker-gate overrides IN MEMORY only (never written back).
        cfg = copy.deepcopy(config)
        cfg.setdefault("sherpa", {})
        # Strip our private bookkeeping key before it reaches SherpaConfig.from_dict.
        clean = {k: v for k, v in sherpa_overrides.items() if not k.startswith("_")}
        cfg["sherpa"].update(clean)

        self.convo = LiveConversation(
            cfg,
            llm_backend=llm_backend,
            main_model=main_model,
            fast_model=fast_model,
            out_dir=self._out,
            user_speaker_id=user_speaker_id,
            capture_assistant_audio=False,
            response_timeout=response_timeout,
            inject=(mode == "inject"),
        )
        self._sherpa_cfg = self.convo.user  # has the sherpa config baked into TTS
        self._gate_rate = int(getattr(self.convo.engine.config, "sample_rate", 16000) or 16000)

        # Install the metric counter BEFORE start() so the engine's on_metric
        # (== runtime.metrics.mark, bound at start()) routes through it.
        self._metrics = _MetricCounter(self.convo.runtime.metrics.mark)
        self.convo.runtime.metrics.mark = self._metrics  # type: ignore[assignment]

        # A second TTS voice for the intruder (a DIFFERENT speaker id the gate
        # should reject). Built lazily on first use.
        self._intruder = None
        self._output_device = output_device

        # Acoustic-mode background-noise player (started per noisy turn).
        self._noise_stop = threading.Event()
        self._noise_thread: Optional[threading.Thread] = None

    # --- lifecycle ---
    def start(self) -> None:
        self.convo.start()
        # Confirm the input gate is actually live (the whole test depends on it).
        gate = getattr(self.convo.engine, "_speaker_gate", None)
        enrolled = bool(gate is not None and gate.is_enrolled)
        gate_on = bool(getattr(self.convo.engine.config, "speaker_gate_input", False))
        if not (enrolled and gate_on):
            log.warning(
                "speaker-ID INPUT GATE IS NOT ACTIVE (enrolled=%s, gate_input=%s) -- "
                "intruder/noise rejection numbers are MEANINGLESS (the gate fails open). "
                "This is a setup error, not an app finding.",
                enrolled, gate_on,
            )
        else:
            log.info("speaker-ID input gate active (enrolled, gate_input=True) -- "
                     "competing-voice rejection is being measured")
        self.gate_active = enrolled and gate_on
        time.sleep(0.5)

    def stop(self) -> None:
        self._stop_noise_player()
        try:
            self.convo.stop()
        except Exception:  # noqa: BLE001
            log.exception("convo stop failed")

    # --- audio construction ---
    def _user_clip(self, text: str):
        """Mock-user clip resampled to the gate/capture rate."""
        from tools.live_session.synthetic_user import _resample

        samples, sr = self.convo.user.synthesize(text)
        return _resample(samples, sr, self._gate_rate)

    def _intruder_clip(self, text: str):
        from tools.live_session.synthetic_user import SyntheticUser, _resample

        if self._intruder is None:
            self._intruder = SyntheticUser(
                self.convo.engine.config, speaker_id=self._intruder_sid
            )
        samples, sr = self._intruder.synthesize(text)
        return _resample(samples, sr, self._gate_rate)

    def _noise_clip(self, n: int, kind: str, rng):
        from . import noise as ng

        if kind == "white":
            return ng.white(n, rng)
        if kind == "pink":
            return ng.pink(n, rng)
        if kind == "babble":
            clips = [
                self._intruder_clip(t)
                for t in (
                    "the meeting is at three this afternoon",
                    "could you pass me the salt please",
                    "i think it might rain later today",
                )
            ]
            return ng.babble(n, self._gate_rate, clips, rng)
        return self._np.zeros(n, dtype="float32")

    def _mix_for_turn(self, nt) -> "tuple":
        """Return (mixed_clip_at_gate_rate, scripted_text, measured_snr) for one
        NoiseTurn. ``speaker`` decides whose voice (if any) is the foreground."""
        from . import noise as ng

        rng = self._np.random.default_rng(self._seed + abs(hash(nt.text)) % 100000)
        spec = nt.noise
        if nt.speaker == "user":
            voice = self._user_clip(nt.text)
        elif nt.speaker == "intruder":
            voice = self._intruder_clip(nt.text)
        else:  # noise-only window: a stretch of voiced/unvoiced energy, no line
            n = int(self._gate_rate * 2.5)
            noise = self._noise_clip(n, spec.kind if spec.kind != "none" else "babble", rng)
            return noise.astype("float32"), "", None

        if spec.kind == "none" or spec.snr_db is None:
            return self._np.asarray(voice, dtype="float32"), nt.text, None

        noise = self._noise_clip(len(voice), spec.kind, rng)
        scaled = ng.scaled_noise_for_snr(voice, noise, spec.snr_db)
        mixed = ng.mix_at_snr(voice, noise, spec.snr_db)
        measured = ng.measured_snr_db(voice, scaled)
        return mixed.astype("float32"), nt.text, round(measured, 2)

    # --- driving one scenario ---
    def run_scenario(self, scenario) -> list[dict]:
        """Drive each NoiseTurn, observing the metric window to attribute
        answered-vs-rejected. Returns the per-turn observation dicts the grader
        consumes."""
        observations: list[dict] = []
        for i, nt in enumerate(scenario.noise_turns):
            if i > 0:
                time.sleep(INTER_TURN_GAP_SEC)
            obs = self._run_turn(nt)
            observations.append(obs)
        return observations

    def _run_turn(self, nt) -> dict:
        mixed, scripted, measured_snr = self._mix_for_turn(nt)
        before = self._metrics.snapshot()
        transcript0 = len(self.convo.runtime.supervisor.state.transcript_log)

        if self.mode == "inject":
            self._inject(mixed)
        else:
            self._acoustic(nt, mixed)

        # Wait for the runtime to settle (answer or drop) this turn.
        time.sleep(TURN_SETTLE_SEC)
        after = self._metrics.snapshot()

        rejected = after.get("speaker_rejected_final", 0) - before.get("speaker_rejected_final", 0)
        # An ANSWERED turn delivered an ASR_FINAL via on_final (the engine only
        # stamps ASR_FINAL through runtime._on_final, i.e. for a turn that passed
        # the gate) and produced first audio.
        asr_finals = after.get("asr_final", 0) - before.get("asr_final", 0)
        first_audio = after.get("tts_first_audio", 0) - before.get("tts_first_audio", 0)
        answered = asr_finals > 0 and first_audio > 0

        # What the recognizer heard for this window (best-effort, for STT score).
        log_ = self.convo.runtime.supervisor.state.transcript_log
        heard = " | ".join(log_[transcript0:]) if len(log_) > transcript0 else None

        log.info(
            "turn[%s/%s]: answered=%s rejected_finals=%d asr_finals=%d first_audio=%d heard=%r",
            nt.speaker, nt.noise.label(), answered, rejected, asr_finals, first_audio, heard,
        )
        return {
            "speaker": nt.speaker,
            "scripted": scripted,
            "heard": heard,
            "answered": bool(answered),
            "expected_answered": bool(nt.expected_answered),
            "rejected_finals": int(max(0, rejected)),
            "asr_finals": int(max(0, asr_finals)),
            "noise": nt.noise.label(),
            "measured_snr_db": measured_snr,
        }

    # --- delivery seams ---
    def _inject(self, mixed) -> None:
        """Feed the mixed clip into the recognizer via the injecting stream,
        with a short noise-floor lead-in (the same pattern LiveConversation uses
        so the streaming recognizer settles before onset)."""
        stream = self.convo._inject_stream
        if stream is None:
            raise RuntimeError("inject mode: the injecting input stream is not available")
        sr2 = stream._sr
        from tools.live_session.synthetic_user import _resample

        played = _resample(mixed, self._gate_rate, sr2)
        lead = stream._noise[: int(0.4 * sr2)]
        stream.push(self._np.concatenate([lead, played]))

    def _acoustic(self, nt, mixed) -> None:
        """ACOUSTIC mode: play background noise (looping) through the real output
        device WHILE the mock-user / intruder line plays over the same speaker,
        and let the real app capture via the real mic. Confounded on a laptop
        (shared speaker+mic, no AEC) -- a realism demo only."""
        import numpy as np

        spec = nt.noise
        # Start a looping noise bed for the duration of the line, if any.
        bed = None
        if spec.kind != "none":
            rng = np.random.default_rng(self._seed + 7)
            bed = self._noise_clip(int(self._gate_rate * 4), spec.kind, rng)
        self._start_noise_player(bed)
        try:
            if nt.speaker == "noise":
                # No spoken line: just let the noise bed play for a window.
                time.sleep(2.5)
            elif nt.speaker == "intruder":
                self._intruder_say(nt.text)
            else:
                self.convo.user.say(nt.text)
        finally:
            self._stop_noise_player()

    def _intruder_say(self, text: str) -> None:
        from tools.live_session.synthetic_user import SyntheticUser

        if self._intruder is None:
            self._intruder = SyntheticUser(
                self.convo.engine.config, speaker_id=self._intruder_sid,
                output_device=self._output_device,
            )
        self._intruder.say(text)

    def _start_noise_player(self, bed) -> None:
        if bed is None or len(bed) == 0:
            return
        import numpy as np
        import sounddevice as sd

        self._noise_stop.clear()
        play = (np.asarray(bed, dtype="float32") * self._noise_volume).astype("float32")

        def _loop():
            try:
                while not self._noise_stop.is_set():
                    sd.play(play, self._gate_rate, device=self._output_device)
                    # Poll for stop while the block plays.
                    end = time.monotonic() + len(play) / float(self._gate_rate)
                    while time.monotonic() < end and not self._noise_stop.is_set():
                        time.sleep(0.05)
                    sd.stop()
            except Exception:  # noqa: BLE001
                log.debug("noise player stopped on error", exc_info=True)

        self._noise_thread = threading.Thread(target=_loop, name="noise-bed", daemon=True)
        self._noise_thread.start()

    def _stop_noise_player(self) -> None:
        self._noise_stop.set()
        th = self._noise_thread
        if th is not None:
            th.join(timeout=1.0)
            self._noise_thread = None
        try:
            import sounddevice as sd

            sd.stop()
        except Exception:  # noqa: BLE001
            pass
