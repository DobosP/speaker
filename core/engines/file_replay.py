from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from ..engine import AudioEngine, EngineCallbacks
from ..metrics import BARGE_IN_STOP, SPEECH_END, TTS_FIRST_AUDIO
from ._sherpa_models import build_recognizer, build_tts
from .sherpa import SherpaConfig

# Headless engine: runs the REAL sherpa-onnx recognizer and TTS over a recorded
# waveform instead of a live mic + sound card, so the full ASR -> LLM -> TTS
# pipeline can run on a server/CI CPU and be measured. It reuses the same
# SherpaConfig and model builders as SherpaOnnxEngine -- only the audio
# transport differs (an in-memory array in, a null sink out).
#
# This is the production path's measurement twin: it shares the recognition
# config and the model objects, but is driven by ``replay_samples`` from the
# bench harness rather than by a capture thread.


class FileReplayEngine(AudioEngine):
    """Replay recorded audio through the real recognizer + TTS, no audio device.

    ``replay_samples`` feeds a waveform in 0.1s blocks through the streaming
    recognizer (firing ``on_partial``/``on_final`` exactly as the live engine
    does) and then a tail of silence so the endpointer declares end-of-speech.
    ``speak`` synthesizes with the real TTS model but discards the audio,
    stamping ``tts_first_audio`` when the clip is ready -- which, for the
    non-streaming offline TTS, is when playback would begin.
    """

    def __init__(self, config: SherpaConfig, *, trailing_silence_sec: float = 0.6):
        self.config = config
        self.trailing_silence_sec = trailing_silence_sec
        self._cb = EngineCallbacks()
        self._recognizer = None
        self._tts = None
        self._stream = None
        self._speaking = threading.Event()
        self._stop_speaking = threading.Event()
        # Observability for the bench harness: what the assistant tried to say
        # this run, and the most recent recognized utterance.
        self.spoken: list[str] = []
        self.last_final: str = ""

    # --- AudioEngine ---
    def start(self, callbacks: EngineCallbacks) -> None:
        self._cb = callbacks
        self._recognizer = build_recognizer(self.config)
        if self._recognizer is None:
            raise SystemExit(
                "FileReplayEngine needs an ASR model: set sherpa.asr_encoder/"
                "decoder/joiner/tokens (see tools/bench, which fetches them)."
            )
        self._tts = build_tts(self.config)
        self._stream = self._recognizer.create_stream()

    def stop(self) -> None:
        self._stop_speaking.set()

    def speak(self, text: str, on_done: Optional[Callable[[], None]] = None) -> None:
        # Synchronous: the bench drives one turn at a time, and synthesizing on
        # the calling (bus) thread keeps the metric stamp ordering deterministic.
        if self._tts is None:
            if on_done:
                on_done()
            return
        self._stop_speaking.clear()
        self._speaking.set()
        self.spoken.append(text)
        self._cb.on_speech_start()
        try:
            if self._stop_speaking.is_set():
                self._cb.on_metric(BARGE_IN_STOP)
                return
            self._tts.generate(
                text, sid=self.config.tts_speaker_id, speed=self.config.tts_speed
            )
            # Offline TTS returns the whole clip; "first audio" is now (the point
            # at which a player would start). No sound card -- we discard it.
            self._cb.on_metric(TTS_FIRST_AUDIO)
        finally:
            self._speaking.clear()
            self._cb.on_speech_end()
            if on_done:
                on_done()

    def stop_speaking(self) -> None:
        self._stop_speaking.set()

    @property
    def is_speaking(self) -> bool:
        return self._speaking.is_set()

    # --- replay driver (called by the bench harness) ---
    def replay_samples(self, samples, sample_rate: int) -> None:
        """Feed a float32 mono waveform through the recognizer, then silence."""
        import numpy as np

        recognizer = self._recognizer
        stream = self._stream
        if recognizer is None or stream is None:
            raise RuntimeError("replay_samples called before start()")

        samples = np.asarray(samples, dtype="float32").reshape(-1)
        block = max(1, int(sample_rate * 0.1))
        last_partial = ""
        emitted_final = False
        tail = np.zeros(int(sample_rate * self.trailing_silence_sec), dtype="float32")
        full = np.concatenate([samples, tail])

        for i in range(0, len(full), block):
            chunk = full[i : i + block]
            stream.accept_waveform(sample_rate, chunk)
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
                    self.last_final = final_text
                    self._cb.on_metric(SPEECH_END)
                    self._cb.on_final(final_text)
                    emitted_final = True

        # Endpoint never fired (very short clip): flush what we have as final so
        # the turn still drives the brain and produces a metrics record.
        if not emitted_final:
            text = recognizer.get_result(stream)
            recognizer.reset(stream)
            if text.strip():
                self.last_final = text
                self._cb.on_metric(SPEECH_END)
                self._cb.on_final(text)

    def barge_in(self) -> None:
        """Simulate the user talking over playback (for barge-in benchmarks)."""
        self._cb.on_barge_in()


def load_waveform(path: str) -> tuple["object", int]:
    """Load a fixture as ``(float32 mono samples, sample_rate)``.

    Supports the ``.npy`` arrays in ``tests/fixture_audio`` (16 kHz mono float32)
    and plain PCM ``.wav`` files. Returns a numpy array.
    """
    import numpy as np

    if path.endswith(".npy"):
        arr = np.load(path).astype("float32").reshape(-1)
        return arr, 16000
    if path.endswith(".wav"):
        import wave

        with wave.open(path, "rb") as wf:
            sample_rate = wf.getframerate()
            n = wf.getnframes()
            raw = wf.readframes(n)
            width = wf.getsampwidth()
            channels = wf.getnchannels()
        if width != 2:
            raise ValueError(f"{path}: only 16-bit PCM WAV is supported")
        data = np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0
        if channels > 1:
            data = data.reshape(-1, channels).mean(axis=1)
        return data, sample_rate
    raise ValueError(f"Unsupported fixture format: {path} (use .npy or .wav)")
