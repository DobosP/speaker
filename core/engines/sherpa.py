from __future__ import annotations

import os
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
        self._stream_in = None
        self._capture_thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._speaking = threading.Event()
        self._stop_speaking = threading.Event()
        self._speaker_gate: Optional[SpeakerGate] = None

    # --- lazy model construction ---
    def _build(self) -> None:
        import sherpa_onnx  # lazy

        c = self.config
        if c.asr_encoder:
            self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens=c.asr_tokens,
                encoder=c.asr_encoder,
                decoder=c.asr_decoder,
                joiner=c.asr_joiner,
                num_threads=c.resolved_asr_threads,
                provider=c.provider,
                sample_rate=c.sample_rate,
                feature_dim=80,
                enable_endpoint_detection=True,
            )
        if c.vad_model:
            vad_config = sherpa_onnx.VadModelConfig()
            vad_config.silero_vad.model = c.vad_model
            vad_config.sample_rate = c.sample_rate
            vad_config.num_threads = c.resolved_asr_threads
            vad_config.provider = c.provider
            self._vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=30)
        if c.tts_model:
            tts_config = sherpa_onnx.OfflineTtsConfig()
            tts_config.model.vits.model = c.tts_model
            tts_config.model.vits.tokens = c.tts_tokens
            if c.tts_data_dir:
                tts_config.model.vits.data_dir = c.tts_data_dir
            tts_config.model.num_threads = c.resolved_tts_threads
            tts_config.model.provider = c.provider
            self._tts = sherpa_onnx.OfflineTts(tts_config)
        if c.speaker_embedding_model:
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

    def stop(self) -> None:
        self._running.clear()
        self._stop_speaking.set()
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=1.0)
        if self._stream_in is not None:
            self._stream_in.stop()
            self._stream_in.close()
            self._stream_in = None

    def speak(self, text: str, on_done: Optional[Callable[[], None]] = None) -> None:
        if self._tts is None:
            if on_done:
                on_done()
            return
        threading.Thread(target=self._speak_blocking, args=(text, on_done), daemon=True).start()

    def stop_speaking(self) -> None:
        self._stop_speaking.set()

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

    def _speak_blocking(self, text: str, on_done: Optional[Callable[[], None]]) -> None:
        import numpy as np
        import sounddevice as sd

        self._stop_speaking.clear()
        self._speaking.set()
        self._cb.on_speech_start()
        try:
            audio = self._tts.generate(
                text, sid=self.config.tts_speaker_id, speed=self.config.tts_speed
            )
            samples = np.asarray(audio.samples, dtype="float32")
            with sd.OutputStream(channels=1, samplerate=audio.sample_rate, dtype="float32") as out:
                chunk = int(audio.sample_rate * 0.1)
                for i in range(0, len(samples), chunk):
                    if self._stop_speaking.is_set():
                        break
                    out.write(samples[i : i + chunk])
        finally:
            self._speaking.clear()
            self._cb.on_speech_end()
            if on_done:
                on_done()

    def _looks_like_user(self, samples) -> bool:
        # Speaker-ID gate: when enrolled, only the user's own voice counts as
        # barge-in, so the assistant's TTS bleeding into the mic can't
        # self-interrupt. Fail-open when no gate/enrollment is configured.
        gate = self._speaker_gate
        if gate is None or not gate.is_enrolled:
            return True
        return gate.accept(samples, self.config.sample_rate)
