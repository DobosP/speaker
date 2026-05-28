from __future__ import annotations

import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

log = logging.getLogger("speaker.sherpa")


def _auto_threads() -> int:
    """Sensible CPU thread count for one ONNX model on a laptop.

    STT/TTS run on the CPU (the GPU is reserved for the LLM), so we use about
    half the logical cores, clamped to 2..4 -- sherpa-onnx rarely benefits from
    more, and we leave headroom for the capture loop and the rest of the system.
    """
    cores = os.cpu_count() or 4
    return max(2, min(4, cores // 2))

from ..engine import AudioEngine, EngineCallbacks
from ..metrics import BARGE_IN_STOP, SPEECH_END, TTS_FIRST_AUDIO
from ._sherpa_models import build_keyword_spotter, build_recognizer, build_tts, build_vad
from .speaker_gate import SpeakerGate, sherpa_speaker_gate

# Production audio engine built on sherpa-onnx (k2-fsa) + sounddevice.
#
# sherpa-onnx is the cross-platform, on-device replacement for the hand-rolled
# utils/audio.py: it provides VAD, streaming ASR, endpointing, TTS, and speaker
# ID from ONNX models on Linux/Windows/macOS/Android/iOS. We import it (and
# sounddevice) lazily so the rest of the runtime and the test suite work in
# environments without the native deps or a sound card.
#
# NOTE on echo / self-barge-in: sherpa-onnx does not do acoustic echo
# cancellation. For robust full-duplex barge-in either (a) use a headset, or
# (b) gate barge-in on speaker identity (sherpa-onnx ships speaker-ID models)
# so the assistant's own TTS isn't mistaken for the user. v1 ships VAD-gated
# barge-in; the gate hook is left as `_looks_like_user`. Tune on hardware.


@dataclass
class SherpaConfig:
    sample_rate: int = 16000
    # Streaming transducer model dir (zipformer). Required for ASR.
    asr_tokens: str = ""
    asr_encoder: str = ""
    asr_decoder: str = ""
    asr_joiner: str = ""
    # Silero VAD model (.onnx). Required for endpointing / barge-in.
    vad_model: str = ""
    # Command fast-path: a streaming keyword spotter (separate small transducer)
    # that runs continuously -- even during playback -- so control phrases like
    # "stop" trigger an action instantly without waiting for ASR + LLM. All
    # empty -> the spotter is disabled and only the normal ASR path runs.
    kws_tokens: str = ""
    kws_encoder: str = ""
    kws_decoder: str = ""
    kws_joiner: str = ""
    kws_keywords_file: str = ""
    kws_threshold: float = 0.25
    kws_score: float = 1.0
    # Offline TTS (e.g. vits / kokoro export). Required for speech output.
    tts_model: str = ""
    tts_tokens: str = ""
    tts_data_dir: str = ""
    tts_speaker_id: int = 0
    tts_speed: float = 1.0
    # Barge-in: seconds of detected voice during playback before we interrupt.
    barge_in_min_speech_sec: float = 0.2
    # Speaker-ID gate: only treat playback-time voice as barge-in if it matches
    # the enrolled user (keeps the assistant's own TTS from self-interrupting).
    speaker_embedding_model: str = ""
    speaker_enroll_wav: str = ""
    # Persisted enrollment embedding (JSON written by ``python -m core --enroll``).
    # Preferred over re-extracting from ``speaker_enroll_wav`` every boot: cheaper
    # startup and it can average several passes. Empty -> fall back to the WAV.
    speaker_enroll_embedding: str = ""
    speaker_threshold: float = 0.5
    # When enrolled, also gate the *normal* ASR finals on speaker identity (not
    # just barge-in), so ambient voices / a TV / a read-aloud quotation aren't
    # answered as if addressed to the assistant. Fail-open when unenrolled.
    speaker_gate_input: bool = True
    # Audio device selection (sounddevice index or name; None/"" = system
    # default). List them with `python -m core --list-devices`. Use these when
    # the default output is e.g. an HDMI monitor with no speakers.
    input_device: object = None
    output_device: object = None
    # Software gain applied to captured audio before ASR -- a quick boost for a
    # quiet mic (1.0 = off). Prefer raising the OS mic level; this is a stopgap.
    input_gain: float = 1.0
    # ONNX execution provider for STT/TTS/speaker-ID. Keep "cpu" so the GPU
    # stays free for the LLM; sherpa-onnx also supports "cuda"/"coreml".
    provider: str = "cpu"
    # CPU threads. ``num_threads`` is the base; ``asr_num_threads`` /
    # ``tts_num_threads`` override per-model. 0 means auto-detect from cores.
    num_threads: int = 0
    asr_num_threads: int = 0
    tts_num_threads: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> "SherpaConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    @property
    def base_threads(self) -> int:
        return self.num_threads if self.num_threads > 0 else _auto_threads()

    @property
    def resolved_asr_threads(self) -> int:
        return self.asr_num_threads if self.asr_num_threads > 0 else self.base_threads

    @property
    def resolved_tts_threads(self) -> int:
        return self.tts_num_threads if self.tts_num_threads > 0 else self.base_threads


def _norm_device(dev):
    """Normalize a device setting: '', None -> None (system default); a digit
    string -> int index; anything else (a device name) is returned as-is."""
    if dev in (None, ""):
        return None
    if isinstance(dev, str) and dev.lstrip("-").isdigit():
        return int(dev)
    return dev


def _resample_linear(samples, src_sr: int, dst_sr: int):
    """Linear-resample mono float32 audio. Bridges a sound card that only opens
    at its native rate to sherpa's fixed 16 kHz (and TTS output back the other
    way), so capture/playback work even when the device rejects the target rate."""
    import numpy as np

    x = np.asarray(samples, dtype="float32").reshape(-1)
    if src_sr == dst_sr or x.size == 0:
        return x
    n_out = int(round(x.shape[0] * float(dst_sr) / float(src_sr)))
    if n_out <= 0:
        return np.zeros(0, dtype="float32")
    idx = np.linspace(0.0, x.shape[0] - 1, num=n_out)
    return np.interp(idx, np.arange(x.shape[0]), x).astype("float32")


class SherpaOnnxEngine(AudioEngine):
    """Real-time on-device engine. Requires model files in :class:`SherpaConfig`.

    Lifecycle: ``start`` opens the mic and runs a capture loop that streams
    audio into the recognizer; partial hypotheses fire ``on_partial`` and an
    endpoint fires ``on_final``. ``speak`` synthesizes via sherpa-onnx TTS and
    plays it while watching for barge-in.
    """

    def __init__(self, config: SherpaConfig):
        self.config = config
        self._cb = EngineCallbacks()
        self._recognizer = None
        self._vad = None
        self._tts = None
        self._kws = None
        self._kws_stream = None
        self._stream_in = None
        # Optional session recording (the 16 kHz audio fed to the recognizer,
        # written to WAV so the run can be replayed and frozen into a test).
        self._record_path: Optional[str] = None
        self._recorder = None
        # Actual mic capture rate (may differ from config.sample_rate when the
        # device won't open at 16 kHz); captured audio is resampled to 16 kHz.
        self._capture_sr = config.sample_rate
        # TTS native rate vs. the rate the speaker actually opened at.
        self._tts_sr = 0
        self._play_sr = 0
        self._capture_thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._speaking = threading.Event()
        self._stop_speaking = threading.Event()
        self._speaker_gate: Optional[SpeakerGate] = None
        # Single playback sink: every utterance is queued onto one worker thread
        # so sentences play in order and never overlap (streaming TTS emits many
        # short sentences in quick succession). Bounded so a runaway producer
        # can't grow memory; oldest is dropped under backpressure to stay fresh.
        self._play_q: "queue.Queue[tuple[Optional[str], Optional[Callable[[], None]]]]" = (
            queue.Queue(maxsize=64)
        )
        self._play_thread: Optional[threading.Thread] = None
        # Whether this sherpa-onnx build supports the streaming TTS callback
        # (play audio as it is synthesized). Flipped off on the first build that
        # rejects the ``callback`` kwarg, after which we chunk the finished wave.
        self._tts_can_stream = True

    def set_record_path(self, path: Optional[str]) -> None:
        """Record this session's recognizer-rate audio to ``path`` (WAV)."""
        self._record_path = path

    # --- lazy model construction ---
    def _build(self) -> None:
        c = self.config
        self._recognizer = build_recognizer(c)
        self._vad = build_vad(c)
        self._tts = build_tts(c)
        self._kws = build_keyword_spotter(c)
        if self._kws is not None:
            self._kws_stream = self._kws.create_stream()
        if c.speaker_embedding_model:
            self._speaker_gate = sherpa_speaker_gate(
                c.speaker_embedding_model,
                threshold=c.speaker_threshold,
                num_threads=c.resolved_asr_threads,
                provider=c.provider,
            )
            self._enroll_speaker_gate()
        if self._speaker_gate is not None and self._speaker_gate.is_enrolled:
            log.info(
                "speaker-ID gate enrolled (threshold=%.2f, input gating=%s)",
                c.speaker_threshold, c.speaker_gate_input,
            )
        elif c.speaker_embedding_model:
            log.warning(
                "speaker-ID model loaded but no enrollment found -- gate is fail-open "
                "(everything passes). Run `python -m core --enroll` to enroll your voice."
            )

    def _enroll_speaker_gate(self) -> None:
        """Load the enrolled reference into the gate.

        Prefer the persisted embedding JSON (cheap, multi-pass averaged); fall
        back to extracting from ``speaker_enroll_wav``. A reference produced by a
        different embedding model is skipped (incomparable vector space) rather
        than trusted. Any failure leaves the gate unenrolled (fail-open)."""
        c = self.config
        gate = self._speaker_gate
        if gate is None:
            return
        if c.speaker_enroll_embedding and os.path.exists(c.speaker_enroll_embedding):
            from ..enroll import enrollment_matches_model, load_enrollment

            try:
                enrollment = load_enrollment(c.speaker_enroll_embedding)
            except Exception as exc:  # noqa: BLE001 - corrupt/old file shouldn't crash boot
                log.warning("could not load enrollment %s: %s", c.speaker_enroll_embedding, exc)
            else:
                if enrollment_matches_model(enrollment, c.speaker_embedding_model):
                    gate.enroll_embedding(enrollment.embedding)
                    return
                log.warning(
                    "enrollment %s was made with a different model (%s); ignoring it -- "
                    "re-run `python -m core --enroll`.",
                    c.speaker_enroll_embedding, enrollment.model or "?",
                )
        if c.speaker_enroll_wav:
            import sherpa_onnx  # lazy

            samples, sr = sherpa_onnx.read_wave(c.speaker_enroll_wav)
            gate.enroll(samples, sr)

    # --- AudioEngine ---
    def start(self, callbacks: EngineCallbacks) -> None:
        import sounddevice as sd  # lazy

        self._cb = callbacks
        self._build()
        self._running.set()
        in_dev = _norm_device(self.config.input_device)
        out_dev = _norm_device(self.config.output_device)
        try:
            log.info("input device: %s", sd.query_devices(in_dev, kind="input").get("name", "?"))
            log.info("output device: %s", sd.query_devices(out_dev, kind="output").get("name", "?"))
        except Exception as exc:  # noqa: BLE001 - diagnostics only
            log.warning("could not query audio devices: %s", exc)
        log.info(
            "models: recognizer=%s vad=%s tts=%s kws=%s",
            self._recognizer is not None,
            self._vad is not None,
            self._tts is not None,
            self._kws is not None,
        )
        # Build the fallback chain: (preferred_device, preferred_sr) ->
        # (preferred_device, native_sr) -> (system_default, preferred_sr)
        # -> (system_default, 16000). The recovering wrapper walks this
        # list on every (re)open, so the same logic that covers initial
        # bring-up also covers reopen after a mid-session PortAudio error.
        from ._recovering_input import _RecoveringInputStream, OpenAttempt

        preferred_sr = self.config.sample_rate
        try:
            dev_sr_in = int(
                sd.query_devices(in_dev, kind="input")["default_samplerate"]
            )
        except Exception:
            dev_sr_in = preferred_sr
        attempts: list[OpenAttempt] = []
        # 1) preferred device at preferred rate
        attempts.append(OpenAttempt(device=in_dev, samplerate=preferred_sr))
        # 2) preferred device at its native rate (deduped)
        if dev_sr_in != preferred_sr:
            attempts.append(OpenAttempt(device=in_dev, samplerate=dev_sr_in))
        # 3) system-default device at preferred rate
        if in_dev is not None:
            attempts.append(OpenAttempt(device=None, samplerate=preferred_sr))
        # 4) system-default device at 16k (lowest common denominator)
        if preferred_sr != 16000:
            attempts.append(OpenAttempt(device=None, samplerate=16000))

        def _open(device, samplerate):
            return sd.InputStream(
                channels=1,
                samplerate=samplerate,
                dtype="float32",
                blocksize=int(samplerate * 0.1),
                device=device,
            )

        self._stream_in = _RecoveringInputStream(
            attempts,
            opener=_open,
            on_state=self._on_capture_state,
            channels=1,
            block_seconds=0.1,
        )
        self._stream_in.open()
        self._capture_sr = self._stream_in.actual_samplerate
        if self._capture_sr != preferred_sr:
            print(
                f"[sherpa] mic rejected {preferred_sr} Hz; "
                f"capturing at {self._capture_sr} Hz and resampling to {preferred_sr} Hz"
            )
        if self._record_path:
            from ..recorder import WavRecorder

            self._recorder = WavRecorder(self._record_path, self.config.sample_rate)
            log.info("recording session audio -> %s", self._record_path)
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        self._play_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._play_thread.start()

    def stop(self) -> None:
        self._running.clear()
        self._stop_speaking.set()
        self._drain_play_q()
        self._play_q.put((None, None))  # sentinel: wake the worker so it exits
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=1.0)
        if self._play_thread is not None:
            self._play_thread.join(timeout=1.0)
        if self._stream_in is not None:
            # The recovering wrapper handles its own internal close-best-effort;
            # we just need to release its handle so the engine can be GC'd.
            try:
                self._stream_in.close()
            except Exception:  # noqa: BLE001 - device may already be gone
                pass
            self._stream_in = None
        if self._recorder is not None:
            log.info("recorded %.1fs of session audio -> %s",
                     self._recorder.seconds, self._recorder.path)
            self._recorder.close()
            self._recorder = None

    def speak(self, text: str, on_done: Optional[Callable[[], None]] = None) -> None:
        # Non-blocking: hand the utterance to the single playback worker. Keeping
        # one sink (instead of a thread per call) is what makes sentence-level
        # streaming play in order rather than on top of itself.
        if self._tts is None:
            if on_done:
                on_done()
            return
        self._enqueue_play(text, on_done)

    def stop_speaking(self) -> None:
        # Cut the current utterance and discard whatever is queued behind it, so
        # a barge-in flushes pending speech instead of letting it play out.
        self._stop_speaking.set()
        self._drain_play_q()

    @property
    def is_speaking(self) -> bool:
        return self._speaking.is_set()

    # --- internals ---
    def _capture_loop(self) -> None:
        import numpy as np

        last_partial = ""
        recognizer = self._recognizer
        if recognizer is None:
            log.error("no recognizer built (ASR model paths missing in config?); "
                      "capture loop idle -- the assistant will never hear you")
            return
        stream = recognizer.create_stream()
        voiced_run = 0.0
        block_sec = 0.1
        # Rolling buffer of the current (non-speaking) ASR segment, used to embed
        # the speaker when an endpoint fires so input can be gated on identity.
        # Capped so a long monologue can't grow memory; the tail is what the
        # speaker model needs. Reset on every endpoint and on decode-recovery.
        utterance: list = []
        utterance_len = 0
        max_utterance = int(self.config.sample_rate * 10)
        # Diagnostics: cumulative + per-interval counters for the 2 s heartbeat.
        total_blocks = partials = finals = 0
        beat_blocks = 0
        beat_level = 0.0
        asr_errors = 0
        last_beat = time.monotonic()
        log.info("capture loop started (capture_sr=%d -> asr_sr=%d)",
                 self._capture_sr, self.config.sample_rate)
        try:
            while self._running.is_set():
                audio, _ = self._stream_in.read(int(self._capture_sr * block_sec))
                samples = np.asarray(audio, dtype="float32").reshape(-1)
                if self._capture_sr != self.config.sample_rate:
                    samples = _resample_linear(samples, self._capture_sr, self.config.sample_rate)
                if self.config.input_gain != 1.0:
                    samples = np.clip(samples * self.config.input_gain, -1.0, 1.0)

                if self._recorder is not None:
                    self._recorder.write(samples)

                total_blocks += 1
                beat_blocks += 1
                if samples.size:
                    beat_level += float(np.sqrt(np.mean(samples * samples)))
                now = time.monotonic()
                if now - last_beat >= 2.0:
                    avg = beat_level / max(beat_blocks, 1)
                    log.debug(
                        "capture heartbeat: blocks=%d avg_rms=%.4f partials=%d finals=%d speaking=%s",
                        total_blocks, avg, partials, finals, self._speaking.is_set(),
                    )
                    if avg < 1e-4:
                        log.warning(
                            "input is ~silent (avg_rms=%.6f) -- wrong mic, muted, or no "
                            "permission? run `python -m sounddevice` to list devices", avg,
                        )
                    self._cb.on_heartbeat()
                    last_beat, beat_blocks, beat_level = now, 0, 0.0

                # Command fast-path: keyword spotting runs every block, including
                # during playback, so phrases like "stop" act with the lowest
                # possible latency and never wait on ASR endpointing or the LLM.
                self._poll_keywords(samples)

                # Barge-in watch while the assistant is speaking.
                if self._speaking.is_set() and self._vad is not None:
                    self._vad.accept_waveform(samples)
                    if self._vad.is_speech_detected() and self._looks_like_user(samples):
                        voiced_run += block_sec
                        if voiced_run >= self.config.barge_in_min_speech_sec:
                            voiced_run = 0.0
                            log.info("barge-in detected")
                            self._cb.on_barge_in()
                    else:
                        voiced_run = 0.0
                    continue

                try:
                    utterance.append(samples)
                    utterance_len += samples.size
                    while utterance_len > max_utterance and len(utterance) > 1:
                        utterance_len -= utterance[0].size
                        utterance.pop(0)
                    stream.accept_waveform(self.config.sample_rate, samples)
                    while recognizer.is_ready(stream):
                        recognizer.decode_stream(stream)
                    text = recognizer.get_result(stream)
                    if text and text != last_partial:
                        last_partial = text
                        partials += 1
                        log.debug("asr partial: %r", text)
                        self._cb.on_partial(text)
                    if recognizer.is_endpoint(stream):
                        final_text = recognizer.get_result(stream)
                        recognizer.reset(stream)
                        last_partial = ""
                        seg = np.concatenate(utterance) if utterance else samples
                        utterance, utterance_len = [], 0
                        if final_text.strip():
                            finals += 1
                            log.info("asr final: %r", final_text)
                            if self._should_act_on_final(seg):
                                self._cb.on_metric(SPEECH_END)
                                self._cb.on_final(final_text)
                            else:
                                log.info(
                                    "dropping final %r -- speaker is not the enrolled user",
                                    final_text,
                                )
                                self._cb.on_metric("speaker_rejected_final")
                    asr_errors = 0
                except Exception:
                    # Recover from a transient decode error by resetting the
                    # stream; bail with a clear message if it keeps failing
                    # (almost always an ASR model/tokens mismatch -- re-run
                    # `python -m tools.setup_models`).
                    asr_errors += 1
                    log.warning("ASR decode error #%d (resetting stream)", asr_errors,
                                exc_info=(asr_errors == 1))
                    last_partial = ""
                    utterance, utterance_len = [], 0
                    try:
                        recognizer.reset(stream)
                    except Exception:
                        stream = recognizer.create_stream()
                    if asr_errors >= 10:
                        log.error(
                            "ASR failed %d times in a row -- likely an ASR model/tokens "
                            "mismatch. Re-run `python -m tools.setup_models` to refetch.",
                            asr_errors,
                        )
                        self._running.clear()
                        break
        except Exception:
            # A daemon thread that dies silently is exactly what makes the app
            # look "stuck": surface the traceback instead of vanishing.
            log.exception("capture loop crashed -- the assistant has stopped listening")
            self._running.clear()

    def _on_capture_state(self, state, message: str) -> None:
        """Forward the recovering wrapper's state changes up to the runtime.

        ``state`` is a :class:`core.engines._recovering_input.StreamState`
        but we marshal it to its string value so the engine callback
        contract stays plain-Python -- no cross-module enum imports
        required of shells. The runtime publishes it as an AgentEvent;
        the watchdog uses it to suppress its 'audio thread stalled'
        warning during legitimate reopens."""
        state_str = getattr(state, "value", str(state))
        try:
            self._cb.on_capture_state(state_str, message)
        except Exception:  # noqa: BLE001
            log.exception("on_capture_state callback raised")
        # Metric for the run summary: every reopen leaves a trace.
        if state_str == "recovering":
            self._cb.on_metric("capture_recovery_start")
        elif state_str == "open":
            self._cb.on_metric("capture_recovery_end")

    def _poll_keywords(self, samples) -> None:
        kws = self._kws
        ks = self._kws_stream
        if kws is None or ks is None:
            return
        ks.accept_waveform(self.config.sample_rate, samples)
        while kws.is_ready(ks):
            kws.decode_stream(ks)
        keyword = kws.get_result(ks)
        if keyword:
            kws.reset_stream(ks)
            self._cb.on_command(keyword)

    def _enqueue_play(self, text: str, on_done: Optional[Callable[[], None]]) -> None:
        try:
            self._play_q.put_nowait((text, on_done))
        except queue.Full:
            # Drop the oldest queued sentence rather than block or lag playback.
            try:
                self._play_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._play_q.put_nowait((text, on_done))
            except queue.Full:
                if on_done:
                    on_done()

    def _drain_play_q(self) -> None:
        try:
            while True:
                self._play_q.get_nowait()
        except queue.Empty:
            pass

    def _playback_loop(self) -> None:
        import sounddevice as sd

        out = None
        try:
            while self._running.is_set():
                try:
                    text, on_done = self._play_q.get(timeout=0.1)
                except queue.Empty:
                    continue
                if text is None:  # shutdown sentinel from stop()
                    break
                self._stop_speaking.clear()
                self._speaking.set()
                self._cb.on_speech_start()
                log.debug("speaking: %r (queue depth=%d)", text, self._play_q.qsize())
                played = {"any": False}

                def write(samples) -> None:
                    if self._stop_speaking.is_set():
                        return
                    if not played["any"]:
                        played["any"] = True
                        self._cb.on_metric(TTS_FIRST_AUDIO)
                    if self._play_sr != self._tts_sr:
                        samples = _resample_linear(samples, self._tts_sr, self._play_sr)
                    out.write(samples)

                try:
                    if out is None:
                        out_dev = _norm_device(self.config.output_device)
                        self._tts_sr = int(getattr(self._tts, "sample_rate", 0)) or 22050
                        try:
                            out = sd.OutputStream(
                                channels=1, samplerate=self._tts_sr, dtype="float32",
                                device=out_dev,
                            )
                            out.start()
                            self._play_sr = self._tts_sr
                            log.info("playback opened at %d Hz on device %s",
                                     self._play_sr, out_dev if out_dev is not None else "default")
                        except sd.PortAudioError:
                            dev_sr = int(sd.query_devices(out_dev, kind="output")["default_samplerate"])
                            out = sd.OutputStream(
                                channels=1, samplerate=dev_sr, dtype="float32",
                                device=out_dev,
                            )
                            out.start()
                            self._play_sr = dev_sr
                            log.warning(
                                "speaker rejected %d Hz; playing at %d Hz and resampling",
                                self._tts_sr, dev_sr,
                            )
                    self._synthesize(text, write)
                    if self._stop_speaking.is_set() and played["any"]:
                        self._cb.on_metric(BARGE_IN_STOP)
                finally:
                    if on_done:
                        on_done()
                    # Only fall idle once the queue drains, so the capture loop
                    # doesn't flap ASR/barge-in on/off between adjacent sentences.
                    if self._play_q.empty():
                        self._speaking.clear()
                        self._cb.on_speech_end()
        except Exception:
            log.exception("playback loop crashed -- the assistant has gone mute")
            self._running.clear()
        finally:
            if out is not None:
                out.stop()
                out.close()

    def _synthesize(self, text: str, write: Callable[[object], None]) -> None:
        """Synthesize ``text``, handing each audio chunk to ``write`` as it is
        produced. sherpa-onnx ``OfflineTts.generate`` streams via a ``callback``,
        so the first samples play before the whole sentence is synthesized; a
        build without that param falls back to chunking the finished waveform."""
        import numpy as np

        tts = self._tts
        sid = self.config.tts_speaker_id
        speed = self.config.tts_speed
        if self._tts_can_stream:

            def on_chunk(samples, *_progress) -> int:
                write(np.asarray(samples, dtype="float32").reshape(-1))
                return 0 if self._stop_speaking.is_set() else 1

            try:
                tts.generate(text, sid=sid, speed=speed, callback=on_chunk)
                return
            except TypeError:
                self._tts_can_stream = False  # this build has no streaming callback

        audio = tts.generate(text, sid=sid, speed=speed)
        samples = np.asarray(audio.samples, dtype="float32").reshape(-1)
        sr = int(getattr(audio, "sample_rate", 0)) or 22050
        chunk = max(1, int(sr * 0.1))
        for i in range(0, len(samples), chunk):
            if self._stop_speaking.is_set():
                break
            write(samples[i : i + chunk])

    def _looks_like_user(self, samples) -> bool:
        # Speaker-ID gate: when enrolled, only the user's own voice counts as
        # barge-in, so the assistant's TTS bleeding into the mic can't
        # self-interrupt. Fail-open when no gate/enrollment is configured.
        gate = self._speaker_gate
        if gate is None or not gate.is_enrolled:
            return True
        return gate.accept(samples, self.config.sample_rate)

    def _should_act_on_final(self, samples) -> bool:
        """Whether a completed ASR utterance should be delivered to the runtime.

        With the gate enrolled and ``speaker_gate_input`` on, only the enrolled
        user's speech is answered -- a TV, another person, or a read-aloud
        quotation is dropped instead of being treated as a request. Fail-open
        when gating is off, there's no gate, or no enrollment, so an
        unconfigured setup behaves exactly as before."""
        if not self.config.speaker_gate_input:
            return True
        gate = self._speaker_gate
        if gate is None or not gate.is_enrolled:
            return True
        return gate.accept(samples, self.config.sample_rate)
