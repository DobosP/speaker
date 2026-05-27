from __future__ import annotations

import os
import queue
import threading
from dataclasses import dataclass
from typing import Callable, Optional


def _auto_threads() -> int:
    """Sensible CPU thread count for one ONNX model on a laptop.

    STT/TTS run on the CPU (the GPU is reserved for the LLM), so we use about
    half the logical cores, clamped to 2..4 -- sherpa-onnx rarely benefits from
    more, and we leave headroom for the capture loop and the rest of the system.
    """
    cores = os.cpu_count() or 4
    return max(2, min(4, cores // 2))

from ..engine import AudioEngine, EngineCallbacks
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
    speaker_threshold: float = 0.5
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
            import sherpa_onnx  # lazy

            self._speaker_gate = sherpa_speaker_gate(
                c.speaker_embedding_model,
                threshold=c.speaker_threshold,
                num_threads=c.resolved_asr_threads,
                provider=c.provider,
            )
            if c.speaker_enroll_wav:
                samples, sr = sherpa_onnx.read_wave(c.speaker_enroll_wav)
                self._speaker_gate.enroll(samples, sr)

    # --- AudioEngine ---
    def start(self, callbacks: EngineCallbacks) -> None:
        import sounddevice as sd  # lazy

        self._cb = callbacks
        self._build()
        self._running.set()
        self._stream_in = sd.InputStream(
            channels=1,
            samplerate=self.config.sample_rate,
            dtype="float32",
            blocksize=int(self.config.sample_rate * 0.1),
        )
        self._stream_in.start()
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
            self._stream_in.stop()
            self._stream_in.close()
            self._stream_in = None

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
            return
        stream = recognizer.create_stream()
        voiced_run = 0.0
        block_sec = 0.1
        while self._running.is_set():
            audio, _ = self._stream_in.read(int(self.config.sample_rate * block_sec))
            samples = np.asarray(audio, dtype="float32").reshape(-1)

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
                        self._cb.on_barge_in()
                else:
                    voiced_run = 0.0
                continue

            stream.accept_waveform(self.config.sample_rate, samples)
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)
            text = recognizer.get_result(stream)
            if text and text != last_partial:
                last_partial = text
                self._cb.on_partial(text)
            if recognizer.is_endpoint(stream):
                final_text = recognizer.get_result(stream)
                recognizer.reset(stream)
                last_partial = ""
                if final_text.strip():
                    self._cb.on_final(final_text)

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
                try:
                    if out is None:
                        sr = int(getattr(self._tts, "sample_rate", 0)) or 22050
                        out = sd.OutputStream(channels=1, samplerate=sr, dtype="float32")
                        out.start()
                    self._synthesize(text, lambda s: self._write_chunk(out, s))
                finally:
                    if on_done:
                        on_done()
                    # Only fall idle once the queue drains, so the capture loop
                    # doesn't flap ASR/barge-in on/off between adjacent sentences.
                    if self._play_q.empty():
                        self._speaking.clear()
                        self._cb.on_speech_end()
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

    def _write_chunk(self, out, samples) -> None:
        if self._stop_speaking.is_set():
            return
        out.write(samples)

    def _looks_like_user(self, samples) -> bool:
        # Speaker-ID gate: when enrolled, only the user's own voice counts as
        # barge-in, so the assistant's TTS bleeding into the mic can't
        # self-interrupt. Fail-open when no gate/enrollment is configured.
        gate = self._speaker_gate
        if gate is None or not gate.is_enrolled:
            return True
        return gate.accept(samples, self.config.sample_rate)
