"""Device-free full-duplex cancellation through the production audio workers.

This is the missing bridge between pure word-cut tests and live owner-mic runs:
the real Sherpa capture thread, playback worker/FIFO/audio callback, and threaded
VoiceRuntime all run concurrently, but paced device streams and deterministic
model doubles keep the test hardware/model/network independent.

The committed ``user_short_stop_command.npy`` is only a high-energy speech
carrier here; fake ASR scripts its text as ``stop``. This validates concurrent
I/O and cancellation, not echo cancellation, recognition accuracy, or bare-
speaker acoustics. Promoted-PCM finalization has separate focused coverage.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator, Optional, Sequence

import numpy as np

from always_on_agent.events import EventKind, Mode
from core.llm import capability_context
from core.metrics import BARGE_IN, BARGE_IN_STOP, TTS_FIRST_AUDIO
from core.runtime import VoiceRuntime
from tools.live_session.driver import (
    InjectingInputStream,
    _NullOutputStream,
    make_recording_engine,
)


_FIXTURES = Path(__file__).parent / "fixture_audio" / "failure_discovery"


def _wait_until(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


class _EnergyVad:
    """One-block deterministic speech verdict over the injected carrier."""

    def __init__(self, threshold: float = 0.01) -> None:
        self.threshold = float(threshold)
        self._active = False

    def accept_waveform(self, samples) -> None:
        block = np.asarray(samples, dtype="float32").reshape(-1)
        level = float(np.sqrt(np.mean(block * block))) if block.size else 0.0
        self._active = level > self.threshold

    def is_speech_detected(self) -> bool:
        return self._active

    def reset(self) -> None:
        self._active = False


class _AsrStream:
    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self.text = ""

    def accept_waveform(self, _sample_rate: int, samples) -> None:
        block = np.asarray(samples, dtype="float32").reshape(-1)
        level = float(np.sqrt(np.mean(block * block))) if block.size else 0.0
        if level > self.threshold:
            self.text = "stop"


class _EnergyStopRecognizer:
    """Scripts ``stop`` only after speech-like injected PCM reaches a stream."""

    def __init__(self, threshold: float = 0.01) -> None:
        self.threshold = float(threshold)

    def create_stream(self, **_kwargs):
        return _AsrStream(self.threshold)

    def is_ready(self, _stream) -> bool:
        return False

    def decode_stream(self, _stream) -> None:  # pragma: no cover - never ready
        pass

    def get_result(self, stream) -> str:
        return stream.text

    def reset(self, stream) -> None:
        stream.text = ""

    def is_endpoint(self, _stream) -> bool:
        # The test owns only the immediate playback-time word cut. Do not turn
        # the promoted command PCM into a second, endpointed STOP event.
        return False


class _FixtureTts:
    """Non-streaming TTS double returning the committed clean reference clip."""

    sample_rate = 16000

    def __init__(self, samples) -> None:
        self._samples = np.asarray(samples, dtype="float32").reshape(-1)
        self._lock = threading.Lock()
        self._texts: list[str] = []

    def generate(self, text: str, *, sid: int = 0, speed: float = 1.0):
        del sid, speed
        with self._lock:
            self._texts.append(text)
        return SimpleNamespace(samples=self._samples.copy(), sample_rate=self.sample_rate)

    def texts(self) -> list[str]:
        with self._lock:
            return list(self._texts)


class _FirstThenBlockedLlm:
    """Emits one audible sentence, then ignores cancellation until released."""

    def __init__(self) -> None:
        self.waiting = threading.Event()
        self.release = threading.Event()
        self.stale_attempted = threading.Event()
        self.finished = threading.Event()
        self.cancel_event = None

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        images: Optional[Sequence[object]] = None,
        history: Optional[Sequence[object]] = None,
    ) -> str:  # pragma: no cover - answering uses stream()
        del prompt, system, images, history
        return "First sentence. STALE post-cut sentence."

    def stream(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        images: Optional[Sequence[object]] = None,
        history: Optional[Sequence[object]] = None,
    ) -> Iterator[str]:
        del prompt, system, images, history
        self.cancel_event = capability_context.get().get("cancel_event")
        try:
            yield "First sentence. "
            self.waiting.set()
            if not self.release.wait(timeout=10.0):
                raise TimeoutError("test did not release the blocked provider")
            self.stale_attempted.set()
            yield "STALE post-cut sentence. "
        finally:
            self.finished.set()


class _ObservedNullOutput(_NullOutputStream):
    """Callback sink that records blocks containing actually played audio."""

    def __init__(self, *args, first_audio: threading.Event, **kwargs) -> None:
        callback = kwargs.get("callback")
        self._first_audio = first_audio
        self._played_lock = threading.Lock()
        self._nonzero_blocks = 0

        def observed(outdata, frames, time_info, status):
            if callback is not None:
                callback(outdata, frames, time_info, status)
            if bool(np.any(np.abs(outdata) > 1e-7)):
                with self._played_lock:
                    self._nonzero_blocks += 1
                self._first_audio.set()

        kwargs["callback"] = observed
        super().__init__(*args, **kwargs)

    def nonzero_blocks(self) -> int:
        with self._played_lock:
            return self._nonzero_blocks


def _fake_sounddevice(monkeypatch, holder: dict, first_audio: threading.Event):
    def input_stream(*_args, samplerate=16000, **_kwargs):
        stream = InjectingInputStream(int(samplerate) or 16000)
        holder["input"] = stream
        return stream

    def output_stream(*args, **kwargs):
        stream = _ObservedNullOutput(*args, first_audio=first_audio, **kwargs)
        holder["output"] = stream
        return stream

    def query_devices(_device=None, *, kind=None):
        name = "Headless EC input" if kind == "input" else "Headless output"
        return {"name": name, "hostapi": 0, "default_samplerate": 16000}

    fake = SimpleNamespace(
        InputStream=input_stream,
        OutputStream=output_stream,
        PortAudioError=RuntimeError,
        query_devices=query_devices,
        query_hostapis=lambda _index=0: {"name": "headless"},
        check_input_settings=lambda **_kwargs: None,
        check_output_settings=lambda **_kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake)


def test_real_capture_and_playback_workers_cancel_stale_stream_on_word_cut(monkeypatch):
    """A playback-time command cuts real FIFO audio and retires stale work once."""
    from core import enroll
    from core.engines import sherpa as sherpa_module
    from core.engines.sherpa import SherpaConfig

    carrier = np.load(_FIXTURES / "user_short_stop_command.npy").astype("float32")
    reference = np.load(_FIXTURES / "reference_clean_tts.npy").astype("float32")
    assert carrier.size / 16000 >= 0.5
    assert reference.size / 16000 >= 0.8

    recognizer = _EnergyStopRecognizer()
    vad = _EnergyVad()
    tts = _FixtureTts(reference)
    monkeypatch.setattr(sherpa_module, "build_recognizer", lambda _cfg: recognizer)
    monkeypatch.setattr(sherpa_module, "build_vad", lambda _cfg: vad)
    monkeypatch.setattr(sherpa_module, "build_tts", lambda _cfg: tts)
    monkeypatch.setattr(
        enroll,
        "verify_required_os_echo_route",
        lambda _cfg: "headless-verified-echo-route",
    )

    holder: dict = {}
    first_audio = threading.Event()
    _fake_sounddevice(monkeypatch, holder, first_audio)

    config = SherpaConfig(
        sample_rate=16000,
        block_sec=0.1,
        input_device="headless-echo-source",
        output_device="headless-output",
        barge_in_enabled=True,
        barge_word_cut_enabled=True,
        # This fixture isolates FIFO/cancellation plumbing. Production
        # enrolled-speaker authority has a separate end-to-end regression.
        barge_word_cut_require_speaker=False,
        # The bridge injects only after actual first audio. Onset-grace policy
        # has dedicated tests; disabling it removes an unrelated clock wait.
        barge_in_playback_onset_grace_sec=0.0,
        aec_enabled=False,
        coherence_barge_in_enabled=False,
        dtd_enabled=False,
        input_calibrate=False,
        tts_target_rms=0.0,
        tts_output_leveler=False,
        tts_output_lowpass_hz=0.0,
    )
    engine, _ = make_recording_engine(config)
    llm = _FirstThenBlockedLlm()
    runtime = VoiceRuntime(
        engine,
        llm,
        start_mode=Mode.ASSISTANT,
        stream_tts=True,
        warm_on_start=False,
    )

    barge_count = 0
    barge_lock = threading.Lock()
    task_id = None
    try:
        runtime.start(run_bus=True)
        input_stream = holder.get("input")
        assert input_stream is not None
        assert engine._word_cut_route_verified is True  # noqa: SLF001
        assert (
            engine._capture_thread is not None  # noqa: SLF001
            and engine._capture_thread.is_alive()  # noqa: SLF001
        )
        assert (
            engine._play_thread is not None  # noqa: SLF001
            and engine._play_thread.is_alive()  # noqa: SLF001
        )

        original_barge = engine._cb.on_barge_in  # noqa: SLF001 - engine callback seam

        def counted_barge() -> None:
            nonlocal barge_count
            with barge_lock:
                barge_count += 1
            original_barge()

        engine._cb.on_barge_in = counted_barge  # noqa: SLF001

        # Seed an ordinary turn directly; only the overlapping interrupt below
        # is under test for capture-path behavior.
        engine._cb.on_final("tell me a story")  # noqa: SLF001
        assert llm.waiting.wait(timeout=3.0), "LLM never emitted sentence one"
        assert first_audio.wait(timeout=3.0), "playback callback never emitted audio"
        assert engine.is_speaking
        assert engine._first_audio_pending is False  # noqa: SLF001
        assert any(TTS_FIRST_AUDIO in record.stamps for record in runtime.metrics.records())

        assert _wait_until(lambda: bool(runtime.supervisor.state.active_tasks))
        task_id = next(iter(runtime.supervisor.state.active_tasks))
        task_runtime = runtime.supervisor.tasks
        with task_runtime._threads_lock:  # noqa: SLF001 - capture real coordinator
            coordinator = task_runtime._threads[task_id]  # noqa: SLF001

        # Inject only after callback-driven playback has emitted real audio.
        assert engine.is_speaking
        assert llm.cancel_event is not None and not llm.cancel_event.is_set()
        inject_at = time.perf_counter()
        input_stream.push(carrier)

        assert _wait_until(lambda: engine.stopped_after(inject_at), timeout=2.0)
        coordinator.join(timeout=2.0)
        assert not coordinator.is_alive(), "task coordinator waited on stale provider"
        assert _wait_until(
            lambda: (
                not runtime.supervisor.state.active_tasks
                and not runtime.supervisor.state.queued_tasks
                and runtime.supervisor.tasks.active_count == 0
            )
        )

        # Cancellation and the real FIFO cut must win while the adversarial
        # provider is still blocked and owned by the test.
        assert llm.cancel_event is not None and llm.cancel_event.is_set()
        assert not llm.release.is_set() and not llm.finished.is_set()
        with barge_lock:
            assert barge_count == 1
        records = runtime.metrics.records()
        assert sum(BARGE_IN in record.stamps for record in records) == 1
        assert sum(BARGE_IN_STOP in record.stamps for record in records) == 1
        [barge_record] = [
            record
            for record in records
            if BARGE_IN in record.stamps and BARGE_IN_STOP in record.stamps
        ]
        assert barge_record.barge_in_latency is not None
        assert barge_record.barge_in_latency >= 0.0
        output = holder["output"]
        nonzero_at_cut = output.nonzero_blocks()

        # Wake the cancelled provider and make it try a complete stale sentence.
        # Generation/task fences must leave no route back to speak/FIFO.
        llm.release.set()
        assert llm.stale_attempted.wait(timeout=2.0)
        assert llm.finished.wait(timeout=2.0)
        assert _wait_until(runtime.bus.idle)
        time.sleep(0.15)  # allow the bounded de-click fade callback to drain
        assert runtime.wait_idle(timeout=2.0)  # receipt + memory handoff settled

        spoken = [text for text, _stamp in engine.spoken_since(0)]
        assert spoken == ["First sentence."]
        assert tts.texts() == ["First sentence."]
        assert output.nonzero_blocks() <= nonzero_at_cut + 2
        assert [
            item.text
            for item in runtime.memory.all()
            if "assistant_output" in item.tags
        ] == []

        terminal = {
            event.kind
            for event in runtime.supervisor.state.event_log
            if event.payload.get("task_id") == task_id
        }
        assert EventKind.TASK_CANCELLED in terminal
        assert EventKind.TASK_COMPLETED not in terminal
        assert EventKind.TASK_FAILED not in terminal
    finally:
        llm.release.set()
        runtime.stop()
