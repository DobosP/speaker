from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from numbers import Real
from typing import Callable, Optional

from ..audio_frontend import AudioResampler
from ..engine import (
    NO_PLAYBACK_CAPABILITIES,
    AudioEngine,
    EngineCallbacks,
    PlaybackCapabilities,
    PlaybackOutcome,
    PlaybackReceipt,
    SpeechStyle,
    TrackedSpeech,
)
from ..metrics import BARGE_IN_STOP, SPEECH_END, TTS_FIRST_AUDIO
from ..tts_markup import (
    prepare_speech_style,
    resolve_tts_params,
    style_to_directives,
)
from ._sherpa_models import (
    build_final_recognizer,
    build_punctuation,
    build_recognizer,
    build_tts,
)
from .sherpa import (
    SherpaConfig,
    _attested_interrupt_repair,
    _transcribe_final_text,
)

log = logging.getLogger("speaker.file_replay")

_FILE_REPLAY_PLAYBACK_CAPABILITIES = PlaybackCapabilities(
    tracked_terminal=True,
    exact_started=True,
    speech_style_hints=True,
)


@dataclass(eq=False)
class _TrackedReplayCall:
    speech: TrackedSpeech
    on_terminal: Callable[[PlaybackReceipt], None]
    on_started: Optional[Callable[[str], None]]
    terminal: bool = False


@dataclass(frozen=True)
class _TrackedReplayDelivery:
    on_started: Optional[Callable[[str], None]]
    on_terminal: Callable[[PlaybackReceipt], None]
    receipt: PlaybackReceipt


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
        self._final_recognizer = None
        self._punct = None
        self._tts = None
        self._stream = None
        self._speaking = threading.Event()
        self._stop_speaking = threading.Event()
        self._receipt_lock = threading.RLock()
        self._generate_lock = threading.Lock()
        self._started = False
        self._stopping = False
        self._play_generation = 0
        self._tracked_busy = False
        self._active_tracked: Optional[_TrackedReplayCall] = None
        # Observability for the bench harness: what the assistant tried to say
        # this run, and the most recent recognized utterance.
        self.spoken: list[str] = []
        self.finals: list[str] = []
        self.last_final: str = ""

    # --- AudioEngine ---
    def _tts_params(self, tts, style: Optional[SpeechStyle]) -> tuple[int, float]:
        if not self.config.tts_markup or style is None:
            return self.config.tts_speaker_id, self.config.tts_speed
        return resolve_tts_params(
            style_to_directives(style),
            default_sid=self.config.tts_speaker_id,
            default_speed=self.config.tts_speed,
            voice_map=self.config.tts_speaker_voices,
            emotion_speed_map=self.config.tts_emotion_speed_map,
            num_speakers=int(getattr(tts, "num_speakers", 0) or 0),
            speed_min=self.config.tts_speed_min,
            speed_max=self.config.tts_speed_max,
            lock_speaker_id=self.config.tts_lock_speaker_id,
        )

    def start(self, callbacks: EngineCallbacks) -> None:
        recognizer = build_recognizer(self.config)
        if recognizer is None:
            raise SystemExit(
                "FileReplayEngine needs an ASR model: set sherpa.asr_encoder/"
                "decoder/joiner/tokens (see tools/bench, which fetches them)."
            )
        tts = build_tts(self.config)
        final_recognizer = build_final_recognizer(self.config)
        punctuation = build_punctuation(self.config)
        stream = recognizer.create_stream()
        with self._receipt_lock:
            self._stop_speaking.set()
            delivery = self._claim_active_tracked_locked(
                PlaybackOutcome.INTERRUPTED
            )
            self._cb = callbacks
            self._recognizer = recognizer
            self._final_recognizer = final_recognizer
            self._punct = punctuation
            self._tts = tts
            self._stream = stream
            self._play_generation += 1
            self._stopping = False
            self._started = True
        if delivery is not None:
            fatal_error = self._deliver_tracked(delivery)
            if fatal_error is not None:
                raise fatal_error

    def stop(self) -> None:
        with self._receipt_lock:
            self._play_generation += 1
            self._stop_speaking.set()
            self._stopping = True
            self._started = False
            delivery = self._claim_active_tracked_locked(
                PlaybackOutcome.INTERRUPTED
            )
        if delivery is not None:
            fatal_error = self._deliver_tracked(delivery)
            if fatal_error is not None:
                raise fatal_error

    def speak(self, text: str, on_done: Optional[Callable[[], None]] = None) -> None:
        # Synchronous: the bench drives one turn at a time, and synthesizing on
        # the calling (bus) thread keeps the metric stamp ordering deterministic.
        with self._receipt_lock:
            generation = self._play_generation
        try:
            style = None
            if self.config.tts_markup:
                prepared = prepare_speech_style(
                    text,
                    style=None,
                    voices=self.config.tts_speaker_voices.keys(),
                    emotions=self.config.tts_emotion_speed_map.keys(),
                )
                text = prepared.text
                style = prepared.style
                if not text or not text.strip():
                    return
            # FileReplay owns one non-thread-safe OfflineTts model.  Dropping a
            # concurrent legacy request is safer than queueing stale work past a
            # later cut, and makes callback re-entry non-blocking.
            if not self._generate_lock.acquire(blocking=False):
                return
            try:
                with self._receipt_lock:
                    if (
                        generation != self._play_generation
                        or self._stopping
                        or not self._started
                        or self._tts is None
                        or self._tracked_busy
                    ):
                        return
                    tts = self._tts
                    callbacks = self._cb
                    self._stop_speaking.clear()

                self._speaking.set()
                try:
                    self.spoken.append(text)
                    callbacks.on_speech_start()
                    try:
                        with self._receipt_lock:
                            cut = (
                                generation != self._play_generation
                                or self._stop_speaking.is_set()
                            )
                        if cut:
                            callbacks.on_metric(BARGE_IN_STOP)
                            return
                        sid, speed = self._tts_params(tts, style)
                        tts.generate(
                            text,
                            sid=sid,
                            speed=speed,
                        )
                        with self._receipt_lock:
                            cut = (
                                generation != self._play_generation
                                or self._stop_speaking.is_set()
                            )
                        if cut:
                            callbacks.on_metric(BARGE_IN_STOP)
                            return
                        # Offline TTS returns the whole clip; "first audio" is
                        # now the point where a player would start. No sound card.
                        callbacks.on_metric(TTS_FIRST_AUDIO)
                    finally:
                        callbacks.on_speech_end()
                finally:
                    self._speaking.clear()
            finally:
                self._generate_lock.release()
        finally:
            if on_done:
                on_done()

    @property
    def playback_capabilities(self) -> PlaybackCapabilities:
        if (
            type(self).speak is not FileReplayEngine.speak
            and type(self).speak_tracked is FileReplayEngine.speak_tracked
        ):
            return NO_PLAYBACK_CAPABILITIES
        return _FILE_REPLAY_PLAYBACK_CAPABILITIES

    def speak_tracked(
        self,
        speech: TrackedSpeech,
        *,
        on_terminal: Callable[[PlaybackReceipt], None],
        on_started: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Synthesize one fragment into the deterministic in-memory null sink.

        FileReplay has no acoustic output callback or output sample domain. A
        non-empty OfflineTts clip accepted by its null sink is one exact
        start/completion boundary, but carries no acoustic sample counts. Stops
        claim the active ticket immediately while native generation unwinds; a
        concurrent call is dropped rather than overlapping the non-thread-safe
        TTS model.
        """

        if not speech.fragment_id:
            raise ValueError("tracked speech requires a fragment_id")
        if self.config.tts_markup:
            prepared = prepare_speech_style(
                speech.text,
                style=speech.style,
                voices=self.config.tts_speaker_voices.keys(),
                emotions=self.config.tts_emotion_speed_map.keys(),
            )
            speech = TrackedSpeech(
                fragment_id=speech.fragment_id,
                text=prepared.text,
                style=prepared.style,
            )
        ticket = _TrackedReplayCall(
            speech=speech,
            on_terminal=on_terminal,
            on_started=on_started,
        )
        if not speech.text.strip():
            self._finish_tracked(ticket, PlaybackOutcome.DROPPED)
            return
        with self._receipt_lock:
            generation = self._play_generation
        if not self._generate_lock.acquire(blocking=False):
            self._finish_tracked(ticket, PlaybackOutcome.DROPPED)
            return
        rejection: Optional[PlaybackOutcome] = None
        activated = False
        reservation_started = False
        try:
            with self._receipt_lock:
                if (
                    generation != self._play_generation
                    or self._stopping
                    or not self._started
                ):
                    rejection = PlaybackOutcome.DROPPED
                elif self._tts is None:
                    rejection = PlaybackOutcome.FAILED
                elif self._tracked_busy:
                    rejection = PlaybackOutcome.DROPPED
                else:
                    reservation_started = True
                    tts = self._tts
                    callbacks = self._cb
                    self._stop_speaking.clear()
                    self._active_tracked = ticket
                    self._tracked_busy = True
                    activated = True
        except BaseException:
            with self._receipt_lock:
                if reservation_started:
                    if self._active_tracked is ticket:
                        self._active_tracked = None
                    self._tracked_busy = False
            activated = False
            raise
        finally:
            if not activated:
                self._generate_lock.release()
        if not activated:
            assert rejection is not None
            self._finish_tracked(ticket, rejection)
            return

        outcome = PlaybackOutcome.FAILED
        fatal_error: Optional[BaseException] = None
        speech_lifecycle_invoked = False
        try:
            try:
                try:
                    self._speaking.set()
                    self.spoken.append(speech.text)
                except Exception as exc:  # internal bookkeeping must not leak
                    log.exception("FileReplay tracked admission bookkeeping failed")
                    fatal_error = exc
                except BaseException as exc:
                    fatal_error = exc
                if fatal_error is None:
                    speech_lifecycle_invoked = True
                    fatal_error = self._invoke_tracked_callback(
                        callbacks.on_speech_start,
                        log_message="FileReplay on_speech_start callback raised",
                    )
                if fatal_error is None:
                    if self._stop_speaking.is_set():
                        metric_fatal = self._invoke_tracked_callback(
                            callbacks.on_metric,
                            BARGE_IN_STOP,
                            log_message=(
                                "FileReplay barge-stop metric callback raised"
                            ),
                        )
                        if fatal_error is None:
                            fatal_error = metric_fatal
                        outcome = PlaybackOutcome.INTERRUPTED
                    else:
                        try:
                            sid, speed = self._tts_params(tts, speech.style)
                            generated = tts.generate(
                                speech.text,
                                sid=sid,
                                speed=speed,
                            )
                            if self._valid_generated_clip(generated):
                                outcome = PlaybackOutcome.COMPLETED
                        except Exception:  # noqa: BLE001 - receipt owns failure
                            log.exception("FileReplay tracked TTS generation failed")
                        except BaseException as exc:
                            fatal_error = exc
                        if (
                            self._stop_speaking.is_set()
                            and outcome is PlaybackOutcome.COMPLETED
                        ):
                            outcome = PlaybackOutcome.INTERRUPTED
            finally:
                try:
                    self._speaking.clear()
                except BaseException as exc:
                    if fatal_error is None:
                        fatal_error = exc
                try:
                    with self._receipt_lock:
                        delivery = self._claim_tracked_locked(ticket, outcome)
                finally:
                    self._generate_lock.release()

            # External callbacks run after the ticket is immutable and the model
            # lock is free. ``_tracked_busy`` stays set until delivery finishes,
            # so callback re-entry is rejected instead of overtaking this receipt.
            if delivery is not None:
                if outcome is PlaybackOutcome.COMPLETED:
                    metric_fatal = self._invoke_tracked_callback(
                        callbacks.on_metric,
                        TTS_FIRST_AUDIO,
                        log_message="FileReplay first-audio metric callback raised",
                    )
                    if fatal_error is None:
                        fatal_error = metric_fatal
                started_fatal = self._deliver_tracked_started(delivery)
                if fatal_error is None:
                    fatal_error = started_fatal
            if speech_lifecycle_invoked:
                end_fatal = self._invoke_tracked_callback(
                    callbacks.on_speech_end,
                    log_message="FileReplay on_speech_end callback raised",
                )
                if fatal_error is None:
                    fatal_error = end_fatal
            if delivery is not None:
                terminal_fatal = self._deliver_tracked_terminal(delivery)
                if fatal_error is None:
                    fatal_error = terminal_fatal
        finally:
            with self._receipt_lock:
                self._tracked_busy = False
        if fatal_error is not None:
            raise fatal_error

    def _claim_tracked(
        self,
        ticket: Optional[_TrackedReplayCall],
        outcome: PlaybackOutcome,
    ) -> Optional[_TrackedReplayDelivery]:
        if ticket is None:
            return None
        with self._receipt_lock:
            return self._claim_tracked_locked(ticket, outcome)

    def _claim_tracked_locked(
        self,
        ticket: _TrackedReplayCall,
        outcome: PlaybackOutcome,
    ) -> Optional[_TrackedReplayDelivery]:
        if ticket.terminal:
            return None
        receipt = PlaybackReceipt(
            fragment_id=ticket.speech.fragment_id,
            outcome=outcome,
            safe_text_prefix=(
                ticket.speech.text if outcome is PlaybackOutcome.COMPLETED else ""
            ),
        )
        ticket.terminal = True
        if self._active_tracked is ticket:
            self._active_tracked = None
        return _TrackedReplayDelivery(
            on_started=(
                ticket.on_started if outcome is PlaybackOutcome.COMPLETED else None
            ),
            on_terminal=ticket.on_terminal,
            receipt=receipt,
        )

    def _claim_active_tracked_locked(
        self,
        outcome: PlaybackOutcome,
    ) -> Optional[_TrackedReplayDelivery]:
        active = self._active_tracked
        if active is None:
            return None
        return self._claim_tracked_locked(active, outcome)

    @staticmethod
    def _valid_generated_clip(generated) -> bool:
        samples = getattr(generated, "samples", None)
        sample_rate = getattr(generated, "sample_rate", None)
        try:
            if samples is None or int(sample_rate) <= 0:
                return False
            shape = getattr(samples, "shape", None)
            if shape is not None:
                if len(shape) != 1 or int(shape[0]) <= 0:
                    return False
                dtype_kind = getattr(getattr(samples, "dtype", None), "kind", None)
                if dtype_kind is not None:
                    return dtype_kind in {"f", "i", "u"}
            if len(samples) == 0:
                return False
            return all(
                isinstance(sample, Real) and not isinstance(sample, bool)
                for sample in samples
            )
        except (IndexError, TypeError, ValueError):
            return False

    @staticmethod
    def _invoke_tracked_callback(
        callback: Callable[..., None],
        *args,
        log_message: str,
    ) -> Optional[BaseException]:
        try:
            callback(*args)
        except Exception:  # noqa: BLE001 - tracked receipts must still resolve
            log.exception(log_message)
        except BaseException as exc:
            return exc
        return None

    @classmethod
    def _deliver_tracked_started(
        cls,
        delivery: _TrackedReplayDelivery,
    ) -> Optional[BaseException]:
        if delivery.on_started is None:
            return None
        return cls._invoke_tracked_callback(
            delivery.on_started,
            delivery.receipt.fragment_id,
            log_message="FileReplay tracked on_started callback raised",
        )

    @classmethod
    def _deliver_tracked_terminal(
        cls,
        delivery: _TrackedReplayDelivery,
    ) -> Optional[BaseException]:
        return cls._invoke_tracked_callback(
            delivery.on_terminal,
            delivery.receipt,
            log_message="FileReplay tracked terminal callback raised",
        )

    @classmethod
    def _deliver_tracked(
        cls,
        delivery: _TrackedReplayDelivery,
    ) -> Optional[BaseException]:
        started_fatal = cls._deliver_tracked_started(delivery)
        terminal_fatal = cls._deliver_tracked_terminal(delivery)
        return started_fatal if started_fatal is not None else terminal_fatal

    def _finish_tracked(
        self,
        ticket: Optional[_TrackedReplayCall],
        outcome: PlaybackOutcome,
    ) -> None:
        delivery = self._claim_tracked(ticket, outcome)
        if delivery is not None:
            fatal_error = self._deliver_tracked(delivery)
            if fatal_error is not None:
                raise fatal_error

    def stop_speaking(self) -> None:
        with self._receipt_lock:
            self._play_generation += 1
            self._stop_speaking.set()
            delivery = self._claim_active_tracked_locked(
                PlaybackOutcome.INTERRUPTED
            )
        if delivery is not None:
            fatal_error = self._deliver_tracked(delivery)
            if fatal_error is not None:
                raise fatal_error

    @property
    def is_speaking(self) -> bool:
        return self._speaking.is_set()

    # --- replay driver (called by the bench harness) ---
    def replay_samples(
        self,
        samples,
        sample_rate: int,
        *,
        speech_sec: Optional[float] = None,
    ) -> None:
        """Feed a waveform through streaming endpointing and production finals.

        ``speech_sec`` is optional independent ownership evidence from a corpus
        manifest. Raw array duration is never promoted to attested timing.
        """
        import numpy as np

        recognizer = self._recognizer
        stream = self._stream
        if recognizer is None or stream is None:
            raise RuntimeError("replay_samples called before start()")

        sample_rate = int(sample_rate)
        if sample_rate <= 0:
            raise ValueError("replay sample_rate must be positive")
        samples = np.asarray(samples, dtype="float32").reshape(-1)
        if sample_rate != self.config.sample_rate:
            # Match the live capture contract: both streaming and offline ASR
            # see one configured-rate, anti-aliased waveform. Resampling the
            # caller-owned PCM before block/endpoint accounting also keeps the
            # two passes on exactly the same sample timeline.
            samples = AudioResampler(
                sample_rate,
                self.config.sample_rate,
                quality=self.config.resampler_quality,
            ).process(samples, last=True)
            sample_rate = self.config.sample_rate
        block = max(1, int(sample_rate * 0.1))
        last_partial = ""
        segment_parts = []
        endpoint_count = 0
        tail = np.zeros(int(sample_rate * self.trailing_silence_sec), dtype="float32")
        full = np.concatenate([samples, tail])

        for i in range(0, len(full), block):
            chunk = full[i : i + block]
            # Retain only caller-owned PCM for the offline final. The synthetic
            # endpoint-flush tail is transport scaffolding, not utterance audio.
            # Reset after each endpoint so a multi-utterance replay cannot feed
            # an earlier utterance into a later second pass.
            source_end = min(i + len(chunk), len(samples))
            if i < source_end:
                segment_parts.append(samples[i:source_end])
            stream.accept_waveform(sample_rate, chunk)
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)
            text = recognizer.get_result(stream)
            if text and text != last_partial:
                last_partial = text
                self._cb.on_partial(text)
            if recognizer.is_endpoint(stream):
                raw_final = recognizer.get_result(stream)
                segment = (
                    np.concatenate(segment_parts)
                    if segment_parts
                    else np.zeros(0, dtype="float32")
                )
                endpoint_count += 1
                if speech_sec is not None and endpoint_count > 1:
                    raise RuntimeError(
                        "speech_sec attestation is valid for one replayed "
                        "utterance, but this waveform produced multiple endpoints"
                    )
                final_text = ""
                if raw_final.strip():
                    final_text = _transcribe_final_text(
                        self.config,
                        self._final_recognizer,
                        self._punct,
                        segment,
                        raw_final,
                        speech_sec=speech_sec,
                        segment_sample_rate=sample_rate,
                        attested_repair=_attested_interrupt_repair,
                    )
                recognizer.reset(stream)
                segment_parts.clear()
                last_partial = ""
                if final_text.strip():
                    self.last_final = final_text
                    self.finals.append(final_text)
                    self._cb.on_metric(SPEECH_END)
                    self._cb.on_final(final_text)

        # Always inspect the current post-endpoint stream. A very short clip may
        # never endpoint, and a second caller-owned segment can remain after an
        # earlier endpoint. Gating this flush on whether *any* prior final was
        # emitted silently dropped that trailing segment and let a one-utterance
        # duration attestation escape its reuse check.
        raw_final = recognizer.get_result(stream)
        segment = (
            np.concatenate(segment_parts)
            if segment_parts
            else np.zeros(0, dtype="float32")
        )
        text = ""
        if raw_final.strip():
            if speech_sec is not None and endpoint_count:
                raise RuntimeError(
                    "speech_sec attestation was already consumed by an "
                    "earlier replay endpoint"
                )
            text = _transcribe_final_text(
                self.config,
                self._final_recognizer,
                self._punct,
                segment,
                raw_final,
                speech_sec=speech_sec,
                segment_sample_rate=sample_rate,
                attested_repair=_attested_interrupt_repair,
            )
        recognizer.reset(stream)
        if text.strip():
            self.last_final = text
            self.finals.append(text)
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
