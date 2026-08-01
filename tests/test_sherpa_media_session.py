from __future__ import annotations

import threading
import time
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from core.engine import EngineCallbacks, TranscriptAbortReason
from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine
from core.engines import _recovering_input as recovering_input_module
from core.engines._sherpa_streaming_decode_owner import DecodeOwnerState
from core.engines._sherpa_streaming_decode_owner import (
    DecodeStreamRole,
    SherpaStreamingDecodeOwner,
)
from core.realtime_media_stage import CaptureScope
from core.engines._aec import FarEndRing
from core.engines._asr_segment import ASRSegment
from core.media_session import (
    CaptureBlockContext,
    CaptureControlLane,
    CaptureGap,
    CaptureGapReason,
    CaptureMailbox,
    CaptureMediaSession,
    CaptureStamp,
    CapturedBlock,
)


def test_sherpa_defaults_to_bounded_capture_with_inline_opt_out() -> None:
    config = SherpaConfig()

    assert config.media_pcm_queue_ms == 300.0
    assert config.media_pcm_queue_max_frames == 8
    assert config.barge_word_cut_require_speaker is False
    assert SherpaConfig(media_pcm_queue_ms=0.0).media_pcm_queue_ms == 0.0


def test_public_start_uses_ready_owner_before_default_reader_and_restarts(
    monkeypatch,
) -> None:
    engine = SherpaOnnxEngine(
        SherpaConfig(
            input_calibrate=False,
            endpoint_enabled=False,
            final_speech_evidence_enabled=False,
            aec_enabled=False,
            barge_in_enabled=False,
            coherence_barge_in_enabled=False,
        )
    )
    recognizers = []
    inputs = []
    order: list[tuple[str, int, str]] = []

    class _Recognizer:
        def __init__(self, run: int) -> None:
            self.run = run
            self.native_threads: list[threading.Thread] = []

        def _note(self, operation: str) -> None:
            thread = threading.current_thread()
            self.native_threads.append(thread)
            order.append((operation, self.run, thread.name))

        def create_stream(self):
            self._note("create_stream")
            return object()

        def is_ready(self, _stream) -> bool:
            self._note("is_ready")
            return False

        def decode_stream(self, _stream) -> None:
            self._note("decode_stream")

        def get_result(self, _stream) -> str:
            self._note("get_result")
            return ""

        def is_endpoint(self, _stream) -> bool:
            self._note("is_endpoint")
            return False

        def reset(self, _stream) -> None:
            self._note("reset")

    class _Input:
        actual_samplerate = 16_000
        actual_device = "ready-before-reader"
        generation = 1

        def __init__(self, *_args, **_kwargs) -> None:
            self.run = len(inputs) + 1
            self.read_entered = threading.Event()
            self.release = threading.Event()
            self.read_exited = threading.Event()
            inputs.append(self)

        def open(self) -> None:
            pass

        def read(self, frames: int):
            owner = engine._streaming_decode_owner
            assert owner is not None
            snapshot = owner.snapshot()
            assert snapshot.ready
            assert snapshot.state is DecodeOwnerState.RUNNING
            assert any(
                operation == "create_stream" and run == self.run
                for operation, run, _thread in order
            )
            order.append(("read", self.run, threading.current_thread().name))
            self.read_entered.set()
            assert self.release.wait(2.0)
            self.read_exited.set()
            return np.zeros(frames, dtype="float32"), False

        def request_close(self) -> None:
            self.release.set()

        def abort_read(self, *, timeout=None) -> bool:
            self.release.set()
            return self.read_exited.wait(timeout)

        def close(self, **_kwargs) -> bool:
            assert self.read_exited.is_set()
            return True

    sounddevice = SimpleNamespace(
        query_devices=lambda *_args, **_kwargs: {
            "name": "ready-before-reader",
            "default_samplerate": 16_000,
        },
        check_input_settings=lambda **_kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "sounddevice", sounddevice)
    monkeypatch.setattr(recovering_input_module, "_RecoveringInputStream", _Input)
    monkeypatch.setattr(engine, "_resolve_capture_domain", lambda *_a, **_k: True)
    monkeypatch.setattr(engine, "_rebind_speech_evidence_domain", lambda: None)

    def build() -> None:
        recognizer = _Recognizer(len(recognizers) + 1)
        recognizers.append(recognizer)
        engine._recognizer = recognizer
        engine._tts = object()

    monkeypatch.setattr(engine, "_build", build)

    for run in (1, 2):
        engine.start(EngineCallbacks())
        source = inputs[-1]
        assert source.read_entered.wait(1.0)
        owner = engine._streaming_decode_owner
        media = engine._capture_media_session
        assert owner is not None and media is not None
        assert engine._capture_thread is owner.thread
        assert media.start_observed
        assert owner.snapshot().state is DecodeOwnerState.RUNNING
        create_index = next(
            index
            for index, item in enumerate(order)
            if item[0] == "create_stream" and item[1] == run
        )
        read_index = next(
            index
            for index, item in enumerate(order)
            if item[0] == "read" and item[1] == run
        )
        assert create_index < read_index

        engine.stop()

        terminal = owner.snapshot()
        assert terminal.state is DecodeOwnerState.CLOSED
        assert terminal.session_closed
        assert not media.may_start_or_is_alive
        assert engine._stream_in is None
        assert not engine._capture_resource_hold.is_set()

    assert len(recognizers) == 2
    for recognizer in recognizers:
        assert recognizer.native_threads
        assert {thread.name for thread in recognizer.native_threads} == {
            "speaker-streaming-decode-owner"
        }


def test_partial_callback_can_reenter_full_stop_on_decode_owner(monkeypatch) -> None:
    engine = SherpaOnnxEngine(
        SherpaConfig(
            input_calibrate=False,
            endpoint_enabled=False,
            final_speech_evidence_enabled=False,
            aec_enabled=False,
            barge_in_enabled=False,
            coherence_barge_in_enabled=False,
        )
    )
    inputs = []
    callback_done = threading.Event()
    callback_errors: list[BaseException] = []
    callback_threads: list[threading.Thread] = []

    class _Recorder:
        seconds = 0.1
        path = "legacy-reentrant.wav"

        def __init__(self) -> None:
            self.close_calls = 0

        def write(self, _samples) -> None:
            pass

        def close(self) -> None:
            self.close_calls += 1

    recorder = _Recorder()
    engine.set_record_path("legacy-reentrant.wav")

    class _Stream:
        accepted = False
        decoded = False

        def accept_waveform(self, _sample_rate: int, _samples) -> None:
            self.accepted = True

    class _Recognizer:
        def create_stream(self):
            return _Stream()

        def is_ready(self, stream: _Stream) -> bool:
            return stream.accepted and not stream.decoded

        def decode_stream(self, stream: _Stream) -> None:
            stream.decoded = True

        def get_result(self, stream: _Stream) -> str:
            return "stop from callback" if stream.decoded else ""

        def is_endpoint(self, _stream: _Stream) -> bool:
            return False

        def reset(self, stream: _Stream) -> None:
            stream.accepted = False
            stream.decoded = False

    class _Input:
        actual_samplerate = 16_000
        actual_device = "reentrant-stop-mic"
        generation = 1

        def __init__(self, *_args, **_kwargs) -> None:
            self.release_first = threading.Event()
            self.close_requested = threading.Event()
            self.second_read_entered = threading.Event()
            self.second_read_exited = threading.Event()
            self.closed = threading.Event()
            self.read_count = 0
            inputs.append(self)

        def open(self) -> None:
            pass

        def read(self, frames: int):
            self.read_count += 1
            if self.read_count == 1:
                assert self.release_first.wait(2.0)
                return np.full(frames, 0.1, dtype="float32"), False
            self.second_read_entered.set()
            assert self.close_requested.wait(2.0)
            self.second_read_exited.set()
            return np.zeros(frames, dtype="float32"), False

        def request_close(self) -> None:
            self.close_requested.set()

        def abort_read(self, *, timeout=None) -> bool:
            self.close_requested.set()
            return self.second_read_exited.wait(timeout)

        def close(self, **_kwargs) -> bool:
            assert self.second_read_exited.is_set()
            self.closed.set()
            return True

    sounddevice = SimpleNamespace(
        query_devices=lambda *_args, **_kwargs: {
            "name": "reentrant-stop-mic",
            "default_samplerate": 16_000,
        },
        check_input_settings=lambda **_kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "sounddevice", sounddevice)
    monkeypatch.setattr(recovering_input_module, "_RecoveringInputStream", _Input)
    monkeypatch.setattr(engine, "_resolve_capture_domain", lambda *_a, **_k: True)
    monkeypatch.setattr(engine, "_rebind_speech_evidence_domain", lambda: None)

    def build() -> None:
        engine._recognizer = _Recognizer()
        engine._tts = object()

    monkeypatch.setattr(engine, "_build", build)

    def open_recorders() -> None:
        engine._recorder = recorder

    monkeypatch.setattr(engine, "_open_recorders", open_recorders)

    def stop_from_partial(_result) -> None:
        callback_threads.append(threading.current_thread())
        try:
            engine.stop()
        except BaseException as exc:  # pragma: no cover - assertion captures it
            callback_errors.append(exc)
        finally:
            callback_done.set()

    engine.start(EngineCallbacks(on_partial_result=stop_from_partial))
    source = inputs[0]
    owner = engine._streaming_decode_owner
    media = engine._capture_media_session
    assert owner is not None and media is not None
    source.release_first.set()
    try:
        assert source.second_read_entered.wait(1.0)
        assert callback_done.wait(3.0)
        owner.thread.join(timeout=2.0)

        assert callback_errors == []
        assert callback_threads == [owner.thread]
        assert not owner.thread.is_alive()
        assert owner.snapshot().state is DecodeOwnerState.CLOSED
        assert owner.snapshot().session_closed
        assert not media.may_start_or_is_alive
        assert source.second_read_exited.is_set()
        assert source.closed.wait(1.0)
        assert engine._stream_in is None
        assert not engine._capture_resource_hold.is_set()
        assert recorder.close_calls == 1
        assert engine._recorder is None

        # A later caller may collect metadata again without reopening or
        # double-closing any native resource.
        engine.stop()
        assert not engine._capture_resource_hold.is_set()
    finally:
        source.release_first.set()
        source.close_requested.set()
        if owner.thread.is_alive():
            engine.stop()


def test_same_domain_gap_preserves_optional_speaker_and_route_authority() -> None:
    class _SpeakerEvidence:
        cleared = False

        def clear_enrollment(self) -> None:
            self.cleared = True

    class _Resettable:
        resets = 0

        def reset(self) -> None:
            self.resets += 1

    class _Clearable:
        clears = 0

        def clear(self) -> None:
            self.clears += 1

    class _Delay:
        continuity_resets = 0
        full_resets = 0

        def reset_continuity(self) -> None:
            self.continuity_resets += 1

        def reset(self) -> None:
            self.full_resets += 1

    engine = SherpaOnnxEngine(SherpaConfig())
    evidence = _SpeakerEvidence()
    denoiser = _Resettable()
    agc = _Resettable()
    apm = _Resettable()
    relaxed_apm = _Resettable()
    vad = _Resettable()
    far_ref = _Clearable()
    delay = _Delay()
    engine._speaker_gate = evidence
    engine._input_agc = agc
    engine._denoiser = denoiser
    engine._aec = apm
    engine._aec_asr = relaxed_apm
    engine._vad = vad
    engine._far_ref = far_ref
    engine._aec_delay_cal = delay
    engine._aec_ref_delay = 321
    engine._word_cut_route_verified = True
    engine._os_echo_route_verified = True
    engine._ambient_rms = 0.25
    engine._playback_floor_rms = 0.5
    engine._raw_playback_floor_rms = 0.75

    engine._reset_capture_frontends_after_gap(capture_sample_rate=16000)

    assert (
        agc.resets,
        denoiser.resets,
        apm.resets,
        relaxed_apm.resets,
        vad.resets,
    ) == (
        1,
        1,
        1,
        1,
        1,
    )
    assert far_ref.clears == 0
    assert engine._ambient_rms == 0.25
    assert engine._playback_floor_rms == 0.5
    assert engine._raw_playback_floor_rms == 0.75
    assert engine._aec_ref_delay == 321
    assert delay.continuity_resets == 1
    assert delay.full_resets == 0
    assert not evidence.cleared
    assert engine._word_cut_route_verified
    assert engine._os_echo_route_verified


class _AsrStream:
    def __init__(self, recognizer) -> None:
        self.recognizer = recognizer
        self.samples: list[np.ndarray] = []
        self.decoded = False

    def accept_waveform(self, _sample_rate: int, samples) -> None:
        note_native_thread = getattr(self.recognizer, "note_native_thread", None)
        if callable(note_native_thread):
            note_native_thread()
        block = np.asarray(samples, dtype="float32").copy()
        self.samples.append(block)
        self.recognizer.trace.append(("accept", float(np.mean(block))))
        if self.recognizer.resets == 0:
            self.recognizer.first_accepted.set()
        else:
            self.recognizer.post_gap_accepted.set()
            self.recognizer.engine._running.clear()


class _StallingRecognizer:
    def __init__(self, engine: SherpaOnnxEngine) -> None:
        self.engine = engine
        self.trace: list[tuple[str, object]] = []
        self.native_threads: list[threading.Thread] = []
        self.resets = 0
        self.first_accepted = threading.Event()
        self.decode_stalled = threading.Event()
        self.release_decode = threading.Event()
        self.post_gap_accepted = threading.Event()

    def create_stream(self):
        self.note_native_thread()
        return _AsrStream(self)

    def is_ready(self, stream: _AsrStream) -> bool:
        self.note_native_thread()
        return bool(stream.samples and not stream.decoded)

    def decode_stream(self, stream: _AsrStream) -> None:
        self.note_native_thread()
        if self.resets == 0:
            self.decode_stalled.set()
            assert self.release_decode.wait(2.0)
        stream.decoded = True

    def get_result(self, stream: _AsrStream) -> str:
        self.note_native_thread()
        if self.resets == 0 and stream.decoded:
            return "before gap"
        return ""

    def is_endpoint(self, _stream: _AsrStream) -> bool:
        self.note_native_thread()
        return False

    def reset(self, stream: _AsrStream) -> None:
        self.note_native_thread()
        self.trace.append(("reset", self.resets))
        self.resets += 1
        stream.samples.clear()
        stream.decoded = False

    def note_native_thread(self) -> None:
        self.native_threads.append(threading.current_thread())


class _BurstingInput:
    actual_samplerate = 16000
    generation = 1

    def __init__(self, first_accepted: threading.Event) -> None:
        self.first_accepted = first_accepted
        self.index = 0
        self.burst_finished = threading.Event()
        self.release = threading.Event()

    def read(self, frames: int):
        self.index += 1
        if self.index == 2:
            assert self.first_accepted.wait(2.0)
        if self.index > 12:
            self.burst_finished.set()
            self.release.wait(2.0)
        return np.full(frames, float(self.index), dtype="float32"), False


def test_bounded_reader_aborts_and_resets_before_post_gap_pcm() -> None:
    engine = SherpaOnnxEngine(
        SherpaConfig(
            endpoint_enabled=False,
            media_pcm_queue_ms=300.0,
            media_pcm_queue_max_frames=8,
        )
    )
    recognizer = _StallingRecognizer(engine)
    source = _BurstingInput(recognizer.first_accepted)
    media = CaptureMediaSession(
        source,
        block_seconds=engine.config.block_sec,
        buffer_ms=engine.config.media_pcm_queue_ms,
        max_frames=engine.config.media_pcm_queue_max_frames,
    )
    engine._recognizer = recognizer
    engine._stream_in = source
    engine._capture_sr = source.actual_samplerate
    engine._capture_media_session = media
    partials = []
    aborts = []
    engine._cb = EngineCallbacks(
        on_partial_result=lambda result: (
            recognizer.trace.append(("partial", result.text)),
            partials.append(result),
        ),
        on_transcript_abort=lambda result: (
            recognizer.trace.append(("abort", result.reason)),
            aborts.append(result),
        ),
    )

    engine._running.set()
    media.start()
    capture = threading.Thread(target=engine._capture_loop)
    capture.start()
    try:
        assert recognizer.decode_stalled.wait(1.0)
        assert source.burst_finished.wait(1.0)
        assert media.generation > source.generation
        recognizer.release_decode.set()
        capture.join(timeout=3.0)

        assert not capture.is_alive()
        assert engine._recognizer is None
        assert engine._streaming_decode_session is not None
        assert engine._streaming_decode_session.closed
        assert recognizer.native_threads
        assert all(thread is capture for thread in recognizer.native_threads)
        assert recognizer.post_gap_accepted.is_set()
        assert [item.text for item in partials] == ["Before gap"]
        assert [item.reason for item in aborts] == [TranscriptAbortReason.BACKPRESSURE]
        labels = [name for name, _value in recognizer.trace]
        assert labels.index("partial") < labels.index("abort")
        assert labels.index("abort") < labels.index("reset")
        post_gap_accept = next(
            index
            for index, item in enumerate(recognizer.trace)
            if item[0] == "accept" and float(item[1]) > 1.0
        )
        assert labels.index("reset") < post_gap_accept
        assert partials[0].acoustic.spans[0].key == aborts[0].acoustic.spans[0].key
        assert aborts[0].revision > partials[0].revision
    finally:
        engine._running.clear()
        media.request_stop()
        source.release.set()
        media.join(timeout=1.0)
        recognizer.release_decode.set()
        capture.join(timeout=1.0)


class _OwnerGapStream:
    def __init__(self, recognizer: "_OwnerGapRecognizer") -> None:
        self.recognizer = recognizer
        self.samples: list[np.ndarray] = []
        self.decoded = False

    def accept_waveform(self, _sample_rate: int, samples) -> None:
        self.recognizer.note_native_thread()
        block = np.asarray(samples, dtype="float32").copy()
        self.samples.append(block)
        self.recognizer.accepts += 1
        self.recognizer.trace.append(("accept", float(np.mean(block))))
        if self.recognizer.accepts == 1:
            self.recognizer.first_accepted.set()
        else:
            self.recognizer.post_gap_accepted.set()
            self.recognizer.engine._running.clear()


class _OwnerGapRecognizer:
    def __init__(self, engine: SherpaOnnxEngine) -> None:
        self.engine = engine
        self.trace: list[tuple[str, object]] = []
        self.native_threads: list[threading.Thread] = []
        self.accepts = 0
        self.first_accepted = threading.Event()
        self.decode_stalled = threading.Event()
        self.release_decode = threading.Event()
        self.post_gap_accepted = threading.Event()

    def create_stream(self):
        self.note_native_thread()
        self.trace.append(("create", len(self.native_threads)))
        return _OwnerGapStream(self)

    def is_ready(self, stream: _OwnerGapStream) -> bool:
        self.note_native_thread()
        return bool(stream.samples and not stream.decoded)

    def decode_stream(self, stream: _OwnerGapStream) -> None:
        self.note_native_thread()
        if self.accepts == 1:
            self.decode_stalled.set()
            assert self.release_decode.wait(2.0)
        stream.decoded = True

    def get_result(self, stream: _OwnerGapStream) -> str:
        self.note_native_thread()
        return "before gap" if self.accepts == 1 and stream.decoded else ""

    def is_endpoint(self, _stream: _OwnerGapStream) -> bool:
        self.note_native_thread()
        return False

    def reset(self, stream: _OwnerGapStream) -> None:
        self.note_native_thread()
        stream.samples.clear()
        stream.decoded = False

    def note_native_thread(self) -> None:
        self.native_threads.append(threading.current_thread())


def test_complete_owner_rotates_all_roles_before_post_gap_pcm() -> None:
    engine = SherpaOnnxEngine(
        SherpaConfig(
            endpoint_enabled=False,
            media_pcm_queue_ms=300.0,
            media_pcm_queue_max_frames=8,
        )
    )
    recognizer = _OwnerGapRecognizer(engine)
    source = _BurstingInput(recognizer.first_accepted)
    media = CaptureMediaSession(
        source,
        block_seconds=engine.config.block_sec,
        buffer_ms=engine.config.media_pcm_queue_ms,
        max_frames=engine.config.media_pcm_queue_max_frames,
        capture_epoch=engine._capture_epoch,
    )
    engine._recognizer = recognizer
    engine._stream_in = source
    engine._capture_sr = source.actual_samplerate
    engine._capture_media_session = media
    session = engine._claim_streaming_decode_session()
    assert session is not None
    partials = []
    aborts = []
    engine._cb = EngineCallbacks(
        on_partial_result=lambda result: (
            recognizer.trace.append(("partial", result.text)),
            partials.append(result),
        ),
        on_transcript_abort=lambda result: (
            recognizer.trace.append(("abort", result.reason)),
            aborts.append(result),
        ),
    )
    initial_scope = CaptureScope(
        capture_epoch=engine._capture_epoch,
        capture_generation=media.generation,
    )
    rotations = []

    def process(owner: SherpaStreamingDecodeOwner) -> None:
        owner.create_stream(role=DecodeStreamRole.WORD_CUT)
        rotate = owner.rotate_capture_scope

        def traced_rotate(scope: CaptureScope):
            recognizer.trace.append(("rotate", scope))
            rotations.append(scope)
            return rotate(scope)

        owner.rotate_capture_scope = traced_rotate
        engine._streaming_decode_owner_loop(owner)

    owner = SherpaStreamingDecodeOwner(
        session,
        run_id=1,
        capture_scope=initial_scope,
        processor=process,
    )
    engine._streaming_decode_owner = owner
    engine._capture_thread = owner.thread
    engine._running.set()
    owner.start()
    media.start()
    try:
        assert recognizer.decode_stalled.wait(1.0)
        assert source.burst_finished.wait(1.0)
        recognizer.release_decode.set()
        assert recognizer.post_gap_accepted.wait(2.0)
        source.release.set()
        assert owner.close(timeout=2.0)
        media.join(timeout=1.0)

        snapshot = owner.snapshot()
        assert snapshot.state is DecodeOwnerState.CLOSED
        assert snapshot.capture_rotations == 1
        assert snapshot.decode_rotations == 0
        assert snapshot.streams_created == 3
        assert snapshot.streams_retired == 3
        assert rotations == [
            CaptureScope(
                capture_epoch=engine._capture_epoch,
                capture_generation=media.generation,
            )
        ]
        assert not media.mailbox.has_in_flight
        assert recognizer.native_threads
        assert set(recognizer.native_threads) == {owner.thread}
        assert [item.text for item in partials] == ["Before gap"]
        assert [item.reason for item in aborts] == [
            TranscriptAbortReason.BACKPRESSURE
        ]
        labels = [name for name, _value in recognizer.trace]
        assert labels.index("partial") < labels.index("abort")
        assert labels.index("abort") < labels.index("rotate")
        post_gap_accept = next(
            index
            for index, item in enumerate(recognizer.trace)
            if item[0] == "accept" and float(item[1]) > 1.0
        )
        assert labels.index("rotate") < post_gap_accept
    finally:
        engine._running.clear()
        media.request_stop()
        source.release.set()
        recognizer.release_decode.set()
        media.join(timeout=1.0)
        owner.close(timeout=1.0)


def test_cancelled_gap_carrier_cannot_orphan_a_published_partial() -> None:
    engine = SherpaOnnxEngine(
        SherpaConfig(
            endpoint_enabled=False,
            media_pcm_queue_ms=100.0,
            media_pcm_queue_max_frames=1,
        )
    )
    recognizer = _OwnerGapRecognizer(engine)
    recognizer.release_decode.set()
    capture_epoch = engine._capture_epoch

    class _CancellingGapMedia:
        error = None

        def __init__(self) -> None:
            self.mailbox = CaptureMailbox(
                buffer_ms=100.0,
                max_frames=1,
                source_generation=1,
                source_sample_rate_hz=16_000,
                source_device=None,
            )
            self.allow_reads = threading.Event()
            self.take_calls = 0
            self.cancelled_carrier = None
            self.cancel_result = None
            self._offer(1.0)

        @property
        def generation(self) -> int:
            return self.mailbox.generation

        def _offer(self, value: float):
            return self.mailbox.offer(
                np.full(1_600, value, dtype="float32"),
                sample_rate_hz=16_000,
                source_generation=1,
                capture_epoch=capture_epoch,
                source_device=None,
                context=CaptureBlockContext(),
            )

        def take(self, timeout=None):
            assert self.allow_reads.wait(2.0)
            self.take_calls += 1
            if self.take_calls == 2:
                # Manufacture one gap carrier, pop it into the exact in-flight
                # slot, then overflow again before returning it to Sherpa. This
                # deterministically recreates reader cancellation between
                # take() and the processor's first stateful action.
                self._offer(2.0)
                self._offer(3.0)
                carrier = self.mailbox.take(timeout=0.0)
                assert carrier is not None and carrier.gap_before is not None
                self._offer(4.0)
                self.cancel_result = self._offer(5.0)
                self.cancelled_carrier = carrier
                assert carrier.effect_lease is not None
                assert carrier.effect_lease.cancelled
                assert self.cancel_result.in_flight_cancelled == 1
                return carrier
            return self.mailbox.take(timeout=timeout)

        def finish(self, captured: CapturedBlock) -> bool:
            return self.mailbox.finish(captured)

        def request_stop(self) -> None:
            self.mailbox.close()

    media = _CancellingGapMedia()
    source = SimpleNamespace(
        actual_samplerate=16_000,
        actual_device=None,
        generation=1,
        request_close=lambda: None,
    )
    engine._recognizer = recognizer
    engine._stream_in = source
    engine._capture_sr = source.actual_samplerate
    engine._capture_media_session = media
    session = engine._claim_streaming_decode_session()
    assert session is not None
    partials = []
    aborts = []
    engine._cb = EngineCallbacks(
        on_partial_result=lambda result: (
            recognizer.trace.append(("partial", result.text)),
            partials.append(result),
        ),
        on_transcript_abort=lambda result: (
            recognizer.trace.append(("abort", result.reason)),
            aborts.append(result),
        ),
    )
    owner = SherpaStreamingDecodeOwner(
        session,
        run_id=1,
        capture_scope=CaptureScope(
            capture_epoch=capture_epoch,
            capture_generation=media.generation,
        ),
        processor=engine._streaming_decode_owner_loop,
    )
    engine._streaming_decode_owner = owner
    engine._capture_thread = owner.thread
    engine._running.set()
    owner.start(timeout=1.0)
    media.allow_reads.set()
    try:
        assert recognizer.post_gap_accepted.wait(2.0)
        assert owner.close(timeout=2.0)

        snapshot = owner.snapshot()
        assert snapshot.state is DecodeOwnerState.CLOSED
        assert snapshot.capture_rotations == 1
        assert snapshot.identity.capture_scope.capture_generation == media.generation
        assert media.cancelled_carrier is not None
        assert media.cancelled_carrier.capture_generation == 2
        assert media.cancel_result is not None
        assert media.cancel_result.capture_generation == 3
        assert not media.mailbox.has_in_flight
        assert [item.text for item in partials] == ["Before gap"]
        assert [item.reason for item in aborts] == [
            TranscriptAbortReason.BACKPRESSURE
        ]
        assert partials[0].acoustic.spans[0].key == aborts[0].acoustic.spans[0].key
        assert aborts[0].revision > partials[0].revision
        labels = [name for name, _value in recognizer.trace]
        assert labels.index("partial") < labels.index("abort")
    finally:
        engine._running.clear()
        media.request_stop()
        owner.close(timeout=1.0)


class _QuietVad:
    def __init__(self) -> None:
        self.resets = 0

    def accept_waveform(self, _samples) -> None:
        pass

    def is_speech_detected(self) -> bool:
        return False

    def reset(self) -> None:
        self.resets += 1


class _NativeFaultStream:
    def __init__(self, recognizer: "_NativeFaultRecognizer", sequence: int) -> None:
        self.recognizer = recognizer
        self.sequence = sequence
        self.accepted = False

    def accept_waveform(self, _sample_rate: int, samples) -> None:
        self.recognizer.note_native_thread()
        if (
            self.recognizer.failure_phase == "primary_accept"
            and self.sequence == 1
            and not self.recognizer.failed
        ) or (
            self.recognizer.failure_phase == "word_cut_accept"
            and self.sequence == 2
            and not self.recognizer.failed
        ):
            self.recognizer.failed = True
            self.recognizer.trace.append(("fault_accept", self.sequence))
            raise RuntimeError("injected native accept failure")
        self.accepted = True
        self.recognizer.decode_identities.append(
            getattr(
                self.recognizer.engine._capture_callback_context,
                "decode_identity",
                None,
            )
        )
        self.recognizer.trace.append(
            ("accept", self.sequence, float(np.mean(samples)))
        )


class _NativeFaultRecognizer:
    def __init__(self, failure_phase: str) -> None:
        self.failure_phase = failure_phase
        self.failed = False
        self.streams_created = 0
        self.trace: list[tuple] = []
        self.native_threads: list[threading.Thread] = []
        self.engine = None
        self.decode_identities = []

    def note_native_thread(self) -> None:
        self.native_threads.append(threading.current_thread())

    def create_stream(self):
        self.note_native_thread()
        self.streams_created += 1
        self.trace.append(("create", self.streams_created))
        return _NativeFaultStream(self, self.streams_created)

    def is_ready(self, _stream: _NativeFaultStream) -> bool:
        self.note_native_thread()
        return False

    def decode_stream(self, _stream: _NativeFaultStream) -> None:
        self.note_native_thread()

    def get_result(self, _stream: _NativeFaultStream) -> str:
        self.note_native_thread()
        return ""

    def is_endpoint(self, _stream: _NativeFaultStream) -> bool:
        self.note_native_thread()
        return False

    def reset(self, stream: _NativeFaultStream) -> None:
        self.note_native_thread()
        if (
            self.failure_phase == "command_reset"
            and stream.sequence == 1
            and not self.failed
        ):
            self.failed = True
            self.trace.append(("fault_reset", stream.sequence))
            raise RuntimeError("injected native reset failure")
        stream.accepted = False
        self.trace.append(("reset", stream.sequence))


@pytest.mark.parametrize(
    "failure_phase, expected_streams, expected_fault",
    [
        ("primary_accept", 2, ("fault_accept", 1)),
        ("command_reset", 2, ("fault_reset", 1)),
        ("word_cut_accept", 3, ("fault_accept", 2)),
    ],
)
def test_complete_owner_rotates_whole_continuity_after_native_fault(
    failure_phase: str,
    expected_streams: int,
    expected_fault: tuple[str, int],
) -> None:
    engine = SherpaOnnxEngine(
        SherpaConfig(
            endpoint_enabled=False,
            final_speech_evidence_enabled=False,
            barge_in_enabled=(failure_phase == "word_cut_accept"),
            barge_word_cut_enabled=(failure_phase == "word_cut_accept"),
            aec_enabled=False,
        )
    )
    recognizer = _NativeFaultRecognizer(failure_phase)
    source = _IdentityInput()
    first_release = threading.Event()
    recognizer.engine = engine

    def wait_until_owner_started() -> None:
        assert first_release.wait(2.0)

    if failure_phase == "word_cut_accept":
        engine._vad = _QuietVad()
        engine._speaking.set()
        engine._barge_sustain_reset_pending = True
        first_context = CaptureBlockContext(
            authority_source_generation=1,
            authority_source_device="mic-a",
            word_cut_route_verified=True,
            speaking=True,
            speak_generation=0,
            playback_generation=0,
        )

        def word_cut_step(_recognizer, stream, samples, *_args, **_kwargs):
            stream.accept_waveform(engine.config.sample_rate, samples)

        engine._barge_word_cut_step = word_cut_step
    else:
        first_context = CaptureBlockContext()

    first = _captured(0.1, at=20.0, context=first_context, sequence=1)
    second = _captured(
        0.2,
        at=20.1,
        context=CaptureBlockContext(),
        sequence=2,
    )
    media = _ScriptedMedia(
        engine,
        [(first, wait_until_owner_started), (second, None)],
    )
    engine._recognizer = recognizer
    engine._stream_in = source
    engine._capture_sr = source.actual_samplerate
    engine._capture_media_session = media
    if failure_phase == "command_reset":
        keyword_attempts = iter((True, False))
        engine._poll_keywords = lambda *_args, **_kwargs: next(keyword_attempts)

    session = engine._claim_streaming_decode_session()
    assert session is not None
    def process(owner: SherpaStreamingDecodeOwner) -> None:
        engine._streaming_decode_owner_loop(owner)

    owner = SherpaStreamingDecodeOwner(
        session,
        run_id=1,
        capture_scope=CaptureScope(capture_epoch=0, capture_generation=1),
        processor=process,
    )
    engine._streaming_decode_owner = owner
    engine._capture_thread = owner.thread
    engine._running.set()
    owner.start(timeout=1.0)
    first_release.set()
    try:
        assert owner.close(timeout=2.0)
        snapshot = owner.snapshot()
        assert snapshot.state is DecodeOwnerState.CLOSED
        assert snapshot.capture_rotations == 0
        assert snapshot.decode_rotations == 1
        assert snapshot.identity.continuity_generation == 1
        assert snapshot.streams_created == expected_streams
        assert snapshot.streams_retired == expected_streams
        assert expected_fault in recognizer.trace
        assert any(
            item[:2] == ("accept", expected_streams)
            for item in recognizer.trace
        )
        assert recognizer.native_threads
        assert set(recognizer.native_threads) == {owner.thread}
        assert recognizer.decode_identities[-1] == snapshot.identity
    finally:
        engine._running.clear()
        first_release.set()
        owner.close(timeout=1.0)


def test_capture_reference_snapshot_is_irreversibly_readonly() -> None:
    source = np.arange(8, dtype="float32")

    reference = SherpaOnnxEngine._readonly_capture_reference(source)
    source[:] = -1.0

    assert reference.tolist() == list(map(float, range(8)))
    assert not reference.flags.writeable
    with pytest.raises(ValueError):
        reference.setflags(write=True)


class _KeywordStream:
    def __init__(self) -> None:
        self.low = False
        self.high = False

    def accept_waveform(self, _sample_rate: int, samples) -> None:
        mean = float(np.mean(samples))
        self.low |= mean <= 1.0
        self.high |= mean > 1.0


class _SplitKeywordSpotter:
    def __init__(self) -> None:
        self.streams: list[_KeywordStream] = []
        self.resets = 0

    def create_stream(self) -> _KeywordStream:
        stream = _KeywordStream()
        self.streams.append(stream)
        return stream

    def is_ready(self, _stream: _KeywordStream) -> bool:
        return False

    def decode_stream(self, _stream: _KeywordStream) -> None:
        pass

    def get_result(self, stream: _KeywordStream) -> str:
        return "stop" if stream.low and stream.high else ""

    def reset_stream(self, stream: _KeywordStream) -> None:
        self.resets += 1
        stream.low = False
        stream.high = False


def test_capture_gap_recreates_kws_before_post_gap_pcm() -> None:
    engine = SherpaOnnxEngine(
        SherpaConfig(
            endpoint_enabled=False,
            media_pcm_queue_ms=300.0,
            media_pcm_queue_max_frames=8,
        )
    )
    recognizer = _StallingRecognizer(engine)
    source = _BurstingInput(recognizer.first_accepted)
    media = CaptureMediaSession(
        source,
        block_seconds=engine.config.block_sec,
        buffer_ms=engine.config.media_pcm_queue_ms,
        max_frames=engine.config.media_pcm_queue_max_frames,
    )
    spotter = _SplitKeywordSpotter()
    engine._kws = spotter
    engine._kws_stream = spotter.create_stream()
    engine._recognizer = recognizer
    engine._stream_in = source
    engine._capture_sr = source.actual_samplerate
    engine._capture_media_session = media
    commands: list[str] = []
    engine._cb = EngineCallbacks(on_command=commands.append)

    engine._running.set()
    media.start()
    capture = threading.Thread(target=engine._capture_loop)
    capture.start()
    try:
        assert recognizer.decode_stalled.wait(1.0)
        assert source.burst_finished.wait(1.0)
        recognizer.release_decode.set()
        capture.join(timeout=3.0)

        assert not capture.is_alive()
        assert commands == []
        assert len(spotter.streams) >= 2
        assert spotter.streams[0] is not spotter.streams[1]
        assert spotter.streams[1].high
        assert not spotter.streams[1].low
    finally:
        engine._running.clear()
        media.request_stop()
        source.release.set()
        media.join(timeout=1.0)
        recognizer.release_decode.set()
        capture.join(timeout=1.0)


class _InlineOverflowInput:
    actual_samplerate = 16000
    generation = 3

    def __init__(self, engine: SherpaOnnxEngine) -> None:
        self.engine = engine
        self.reads = 0

    def read(self, frames: int):
        self.reads += 1
        if self.reads == 1:
            return np.ones(frames, dtype="float32"), False
        self.engine._running.clear()
        return np.zeros(frames, dtype="float32"), True


class _InlineRecognizer:
    def __init__(self) -> None:
        self.resets = 0
        self.trace: list[tuple[str, object]] = []
        self.first_accepted = threading.Event()
        self.post_gap_accepted = threading.Event()
        self.engine = None

    def create_stream(self):
        return _AsrStream(self)

    def is_ready(self, stream: _AsrStream) -> bool:
        return bool(stream.samples and not stream.decoded)

    def decode_stream(self, stream: _AsrStream) -> None:
        stream.decoded = True

    def get_result(self, stream: _AsrStream) -> str:
        return "hello" if self.resets == 0 and stream.decoded else ""

    def is_endpoint(self, _stream: _AsrStream) -> bool:
        return False

    def reset(self, stream: _AsrStream) -> None:
        self.resets += 1
        stream.samples.clear()
        stream.decoded = False


def test_inline_compatibility_still_terminalizes_native_overflow() -> None:
    engine = SherpaOnnxEngine(
        SherpaConfig(endpoint_enabled=False, media_pcm_queue_ms=0.0)
    )
    recognizer = _InlineRecognizer()
    recognizer.engine = engine
    engine._recognizer = recognizer
    engine._stream_in = _InlineOverflowInput(engine)
    engine._capture_sr = 16000
    partials = []
    aborts = []
    engine._cb = EngineCallbacks(
        on_partial_result=partials.append,
        on_transcript_abort=aborts.append,
    )

    engine._running.set()
    engine._capture_loop()

    assert [item.text for item in partials] == ["Hello"]
    assert [item.reason for item in aborts] == [TranscriptAbortReason.BACKPRESSURE]
    assert recognizer.resets == 1
    assert partials[0].acoustic.spans[0].capture_generation == 3


class _StopAwareInput:
    actual_samplerate = 16000
    generation = 1

    def __init__(self) -> None:
        self.read_entered = threading.Event()
        self.release = threading.Event()
        self.read_exited = threading.Event()
        self.order: list[str] = []

    def read(self, frames: int):
        self.read_entered.set()
        self.release.wait(2.0)
        self.order.append("read_exit")
        self.read_exited.set()
        return np.ones(frames, dtype="float32"), False

    def request_close(self) -> None:
        self.order.append("request_close")
        self.release.set()

    def abort_read(self, *, timeout: float) -> bool:
        self.order.append("abort_read")
        self.release.set()
        return self.read_exited.wait(timeout)

    def close(self, **_kwargs) -> bool:
        assert self.read_exited.is_set()
        self.order.append("close")
        return True


def test_stop_wakes_reader_before_close_and_runs_no_queued_callback() -> None:
    engine = SherpaOnnxEngine(SherpaConfig(endpoint_enabled=False))
    recognizer = _InlineRecognizer()
    recognizer.engine = engine
    source = _StopAwareInput()
    media = CaptureMediaSession(
        source,
        block_seconds=engine.config.block_sec,
        buffer_ms=engine.config.media_pcm_queue_ms,
        max_frames=engine.config.media_pcm_queue_max_frames,
    )
    partials = []
    engine._recognizer = recognizer
    engine._stream_in = source
    engine._capture_sr = source.actual_samplerate
    engine._capture_media_session = media
    engine._cb = EngineCallbacks(on_partial_result=partials.append)
    engine._running.set()
    media.start()
    engine._capture_thread = threading.Thread(target=engine._capture_loop)
    engine._capture_thread.start()
    assert source.read_entered.wait(1.0)

    engine.stop()

    assert not media.is_alive
    assert not engine._capture_thread.is_alive()
    assert partials == []
    assert source.order == ["request_close", "read_exit", "close"]
    assert engine._stream_in is None
    assert engine._capture_media_session is None


class _IdentityInput:
    actual_samplerate = 16000

    def __init__(self, generation: int = 1, device="mic-a") -> None:
        self.generation = generation
        self.actual_device = device


class _ClosedMailbox:
    closed = True


class _ScriptedMedia:
    error = None
    mailbox = _ClosedMailbox()

    def __init__(self, engine, entries, *, generation: int = 1) -> None:
        self.engine = engine
        self.entries = list(entries)
        self.generation = generation

    def take(self, timeout=None):
        del timeout
        if not self.entries:
            self.engine._running.clear()
            return None
        block, action = self.entries.pop(0)
        if action is not None:
            action()
        return block

    def request_stop(self) -> None:
        pass


class _CollectingStream:
    def __init__(self, owner) -> None:
        self.owner = owner

    def accept_waveform(self, _sample_rate, samples) -> None:
        self.owner.accepted.append(np.asarray(samples).copy())


class _CollectingRecognizer:
    def __init__(self) -> None:
        self.accepted: list[np.ndarray] = []

    def create_stream(self):
        return _CollectingStream(self)

    def is_ready(self, _stream) -> bool:
        return False

    def decode_stream(self, _stream) -> None:
        pass

    def get_result(self, _stream) -> str:
        return ""

    def is_endpoint(self, _stream) -> bool:
        return False

    def reset(self, _stream) -> None:
        pass


def _captured(
    value: float,
    *,
    at: float,
    context: CaptureBlockContext,
    source_generation: int = 1,
    sequence: int = 1,
    capture_generation: int | None = None,
    gap_before: CaptureGap | None = None,
) -> CapturedBlock:
    pcm = np.full(1600, value, dtype="float32")
    return CapturedBlock(
        pcm=pcm,
        sample_rate_hz=16000,
        sequence=sequence,
        captured_at=at,
        captured_started_at=at - 0.1,
        capture_epoch=0,
        source_generation=source_generation,
        source_device="mic-a",
        source_sample_start=(sequence - 1) * 1600,
        source_sample_end=sequence * 1600,
        capture_generation=(
            source_generation if capture_generation is None else capture_generation
        ),
        context=context,
        gap_before=gap_before,
    )


def test_backlog_uses_captured_playback_state_across_start_and_end() -> None:
    engine = SherpaOnnxEngine(
        SherpaConfig(endpoint_enabled=False, barge_in_enabled=False)
    )
    recognizer = _CollectingRecognizer()
    source = _IdentityInput()
    engine._recognizer = recognizer
    engine._stream_in = source
    engine._capture_sr = 16000
    engine._playback_generation = 2
    engine._speaking.set()
    command_attempts: list[tuple[float, bool]] = []

    def poll(samples, **kwargs):
        command_attempts.append(
            (float(np.mean(samples)), bool(kwargs["allow_publish"]))
        )
        return bool(kwargs["allow_publish"])

    engine._poll_keywords = poll

    before_start = _captured(
        1.0,
        at=10.0,
        sequence=1,
        context=CaptureBlockContext(
            speaking=False,
            speak_generation=0,
            playback_generation=1,
        ),
    )
    old_playback = _captured(
        2.0,
        at=10.1,
        sequence=2,
        context=CaptureBlockContext(
            speaking=True,
            speak_generation=0,
            playback_generation=1,
        ),
    )
    after_end = _captured(
        3.0,
        at=10.2,
        sequence=3,
        context=CaptureBlockContext(
            speaking=True,
            speak_generation=0,
            playback_generation=2,
        ),
    )
    media = _ScriptedMedia(
        engine,
        [
            (before_start, None),
            (old_playback, None),
            (after_end, engine._speaking.clear),
        ],
    )
    engine._capture_media_session = media
    engine._running.set()

    engine._capture_loop()

    assert [float(np.mean(block)) for block in recognizer.accepted] == [1.0]
    assert command_attempts == [(1.0, False)]


def test_first_eligible_post_gap_block_cannot_publish_or_raise_ambient_floor() -> None:
    engine = SherpaOnnxEngine(
        SherpaConfig(
            endpoint_enabled=False,
            barge_in_enabled=False,
            input_loudness_margin_db=3.0,
        )
    )
    recognizer = _CollectingRecognizer()
    source = _IdentityInput()
    gap = CaptureGap(
        reason=CaptureGapReason.MEDIA_BACKPRESSURE,
        prior_generation=1,
        generation=2,
        first_dropped_sequence=1,
        last_dropped_sequence=1,
        dropped_frames=1,
        dropped_samples=1600,
    )
    block = _captured(
        0.15,
        at=11.0,
        context=CaptureBlockContext(),
        capture_generation=2,
        gap_before=gap,
    )
    publish_permissions: list[bool] = []

    def poll(_samples, **kwargs):
        publish_permissions.append(bool(kwargs["allow_publish"]))
        return False

    engine._recognizer = recognizer
    engine._stream_in = source
    engine._capture_sr = 16000
    engine._ambient_rms = 0.1
    engine._poll_keywords = poll
    engine._capture_media_session = _ScriptedMedia(
        engine,
        [(block, None)],
        generation=1,
    )
    engine._running.set()

    engine._capture_loop()

    assert publish_permissions == [False]
    assert engine._ambient_rms == 0.1
    assert len(recognizer.accepted) == 1


def test_stale_playback_block_does_not_consume_post_gap_guard() -> None:
    engine = SherpaOnnxEngine(
        SherpaConfig(
            endpoint_enabled=False,
            barge_in_enabled=False,
            input_loudness_margin_db=3.0,
        )
    )
    recognizer = _CollectingRecognizer()
    source = _IdentityInput()
    gap = CaptureGap(
        reason=CaptureGapReason.MEDIA_BACKPRESSURE,
        prior_generation=1,
        generation=2,
        first_dropped_sequence=1,
        last_dropped_sequence=1,
        dropped_frames=1,
        dropped_samples=1600,
    )
    stale = _captured(
        0.2,
        at=12.0,
        context=CaptureBlockContext(
            speaking=True,
            speak_generation=0,
            playback_generation=1,
        ),
        capture_generation=2,
        gap_before=gap,
    )
    current = _captured(
        0.15,
        at=12.1,
        sequence=2,
        context=CaptureBlockContext(
            speaking=False,
            speak_generation=0,
            playback_generation=2,
        ),
        capture_generation=2,
    )
    publish_permissions: list[bool] = []

    def poll(_samples, **kwargs):
        publish_permissions.append(bool(kwargs["allow_publish"]))
        return False

    engine._recognizer = recognizer
    engine._stream_in = source
    engine._capture_sr = 16000
    engine._ambient_rms = 0.1
    engine._playback_generation = 2
    engine._speaking.set()
    engine._poll_keywords = poll
    engine._capture_media_session = _ScriptedMedia(
        engine,
        [
            (stale, None),
            (current, engine._speaking.clear),
        ],
        generation=1,
    )
    engine._running.set()

    engine._capture_loop()

    assert publish_permissions == [False]
    assert engine._ambient_rms == 0.1
    assert len(recognizer.accepted) == 1
    assert np.isclose(float(np.mean(recognizer.accepted[0])), 0.15)


class _RecordingAec:
    always_on = True
    suppresses_noise = False
    suppresses_nearend = False

    def __init__(self) -> None:
        self.references: list[np.ndarray] = []

    def process_16k(self, near, far):
        self.references.append(np.asarray(far).copy())
        return np.asarray(near).copy()

    def reset(self) -> None:
        pass


class _RecordingDelay:
    def __init__(self) -> None:
        self.references: list[np.ndarray] = []

    def observe(self, _mic, far0) -> None:
        self.references.append(np.asarray(far0).copy())

    def current_delay_samples(self) -> int:
        return 77


class _RecordingCoherence:
    def __init__(self) -> None:
        self.references: list[np.ndarray] = []

    def note_playback(self, samples, _sample_rate) -> None:
        self.references.append(np.asarray(samples).copy())

    def reset(self) -> None:
        pass


def test_aec_and_delay_calibrator_use_reader_captured_reference() -> None:
    engine = SherpaOnnxEngine(
        SherpaConfig(endpoint_enabled=False, barge_in_enabled=False)
    )
    recognizer = _CollectingRecognizer()
    source = _IdentityInput()
    aec = _RecordingAec()
    delay = _RecordingDelay()
    captured_zero = np.full(1600, 0.25, dtype="float32")
    captured_delayed = np.full(1600, 0.5, dtype="float32")
    context = CaptureBlockContext(
        far_reference_zero=captured_zero,
        far_reference_delayed=captured_delayed,
        coherence_reference=captured_zero,
    )
    block = _captured(0.1, at=20.0, context=context)
    engine._recognizer = recognizer
    engine._stream_in = source
    engine._capture_sr = 16000
    engine._aec = aec
    engine._apm_always_on = True
    engine._aec_delay_cal = delay
    engine._far_ref = FarEndRing()
    engine._far_ref.push(np.full(1600, 0.9, dtype="float32"))
    engine._capture_media_session = _ScriptedMedia(engine, [(block, None)])
    engine._running.set()

    engine._capture_loop()

    np.testing.assert_array_equal(delay.references[0], captured_zero)
    np.testing.assert_array_equal(aec.references[0], captured_delayed)
    assert engine._aec_ref_delay == 77


def test_coherence_ingest_uses_reader_captured_reference() -> None:
    engine = SherpaOnnxEngine(
        SherpaConfig(endpoint_enabled=False, barge_in_enabled=False)
    )
    recognizer = _CollectingRecognizer()
    source = _IdentityInput()
    captured_reference = np.full(1600, 0.35, dtype="float32")
    context = CaptureBlockContext(
        speaking=True,
        barge_watch_active=False,
        speak_generation=0,
        playback_generation=4,
        coherence_reference=captured_reference,
    )
    block = _captured(0.1, at=21.0, context=context)
    coherence = _RecordingCoherence()
    engine._recognizer = recognizer
    engine._stream_in = source
    engine._capture_sr = 16000
    engine._speaking.set()
    engine._playback_generation = 4
    engine._echo_coherence = coherence
    engine._playback_ref = FarEndRing()
    engine._playback_ref.push(np.full(1600, 0.9, dtype="float32"))
    engine._capture_media_session = _ScriptedMedia(engine, [(block, None)])
    engine._running.set()

    engine._capture_loop()

    np.testing.assert_array_equal(coherence.references[0], captured_reference)


class _LevelVad:
    def __init__(self) -> None:
        self.speech = False

    def accept_waveform(self, samples) -> None:
        self.speech = bool(float(np.mean(np.abs(samples))) > 0.01)

    def is_speech_detected(self) -> bool:
        return self.speech

    def reset(self) -> None:
        self.speech = False


def test_vad_and_endpoint_clocks_use_captured_time(
    monkeypatch,
) -> None:
    observed_vad: list[float] = []
    observed_decisions: list[float] = []
    original_observe = ASRSegment.observe_vad
    original_trailing = ASRSegment.trailing_silence

    def observe(self, active, now):
        observed_vad.append(now)
        return original_observe(self, active, now)

    def trailing(self, now):
        observed_decisions.append(now)
        return original_trailing(self, now)

    monkeypatch.setattr(ASRSegment, "observe_vad", observe)
    monkeypatch.setattr(ASRSegment, "trailing_silence", trailing)
    engine = SherpaOnnxEngine(
        SherpaConfig(endpoint_enabled=False, barge_in_enabled=False)
    )
    recognizer = _CollectingRecognizer()
    source = _IdentityInput()
    engine._recognizer = recognizer
    engine._vad = _LevelVad()
    engine._stream_in = source
    engine._capture_sr = 16000
    quiet_context = CaptureBlockContext()
    blocks = [
        _captured(0.2, at=100.0, sequence=1, context=quiet_context),
        _captured(0.0, at=100.7, sequence=2, context=quiet_context),
    ]
    engine._capture_media_session = _ScriptedMedia(
        engine, [(block, None) for block in blocks]
    )
    engine._running.set()

    engine._capture_loop()

    assert observed_vad == [100.0, 100.7]
    assert observed_decisions == [100.0, 100.7, 100.7]


def test_capture_context_snapshots_both_reference_windows_atomically() -> None:
    class _AtomicReference:
        def __init__(self) -> None:
            self.calls: list[tuple[int, tuple[int, ...]]] = []

        def read_windows(self, n: int, delays):
            normalized = tuple(delays)
            self.calls.append((n, normalized))
            return (
                np.full(n, 0.25, dtype="float32"),
                np.full(n, 0.5, dtype="float32"),
            )

        def read(self, _n: int, _delay: int):
            raise AssertionError("separate ring-head reads are forbidden")

    engine = SherpaOnnxEngine(SherpaConfig())
    reference = _AtomicReference()
    engine._playback_ref = reference
    engine._aec_ref_delay = 23
    stamp = CaptureStamp(
        sample_rate_hz=48000,
        sample_count=4800,
        captured_started_at=1.0,
        captured_at=1.1,
        capture_epoch=1,
        source_generation=1,
        source_device="mic-a",
        source_sample_start=0,
        source_sample_end=4800,
    )

    context = engine._snapshot_capture_context(stamp)

    assert reference.calls == [(1600, (0, 23))]
    np.testing.assert_array_equal(
        context.far_reference_zero,
        np.full(1600, 0.25, dtype="float32"),
    )
    np.testing.assert_array_equal(
        context.far_reference_delayed,
        np.full(1600, 0.5, dtype="float32"),
    )


def test_capture_context_fails_closed_until_new_source_authority_is_bound() -> None:
    class _Gate:
        is_enrolled = True

    engine = SherpaOnnxEngine(SherpaConfig())
    domain = object()
    engine._capture_media_session = object()
    engine._capture_resolution = domain
    engine._capture_authority_source_generation = 1
    engine._capture_authority_source_device = "mic-a"
    engine._word_cut_route_verified = True
    engine._os_echo_route_verified = True
    engine._speaker_gate = _Gate()
    engine._speaker_gate_warmed = True
    engine._playback_ref = FarEndRing()

    old_stamp = CaptureStamp(
        sample_rate_hz=16000,
        sample_count=1600,
        captured_started_at=1.0,
        captured_at=1.1,
        capture_epoch=1,
        source_generation=1,
        source_device="mic-a",
        source_sample_start=0,
        source_sample_end=1600,
    )
    new_stamp = CaptureStamp(
        sample_rate_hz=16000,
        sample_count=1600,
        captured_started_at=1.1,
        captured_at=1.2,
        capture_epoch=1,
        source_generation=2,
        source_device="mic-b",
        source_sample_start=0,
        source_sample_end=1600,
    )

    old = engine._snapshot_capture_context(old_stamp)
    unbound = engine._snapshot_capture_context(new_stamp)
    engine._capture_authority_source_generation = 2
    engine._capture_authority_source_device = "mic-b"
    rebound = engine._snapshot_capture_context(new_stamp)

    assert old.source_domain is domain
    assert old.word_cut_route_verified
    assert old.os_echo_route_verified
    assert old.speaker_authority_available
    assert not unbound.word_cut_route_verified
    assert not unbound.os_echo_route_verified
    assert not unbound.speaker_authority_available
    assert rebound.word_cut_route_verified
    assert rebound.os_echo_route_verified
    assert rebound.speaker_authority_available


class _BlockingDenoiser:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def process_16k(self, samples):
        self.entered.set()
        assert self.release.wait(2.0)
        return np.asarray(samples).copy()

    def reset(self) -> None:
        pass


class _DoubleRecoverySource:
    actual_samplerate = 16000
    actual_device = "mic-a"

    def __init__(self, denoiser: _BlockingDenoiser) -> None:
        self.generation = 1
        self.denoiser = denoiser
        self.reads = 0
        self.recovered = threading.Event()
        self.release = threading.Event()

    def read(self, frames: int):
        self.reads += 1
        if self.reads == 1:
            return np.full(frames, 1.0, dtype="float32"), False
        if self.reads == 2:
            assert self.denoiser.entered.wait(2.0)
            self.generation = 2
            self.actual_device = "mic-b"
            self.recovered.set()
            return np.full(frames, 2.0, dtype="float32"), False
        self.release.wait(2.0)
        return np.full(frames, 3.0, dtype="float32"), False


def test_second_recovery_discards_already_popped_stale_pcm() -> None:
    engine = SherpaOnnxEngine(
        SherpaConfig(endpoint_enabled=False, barge_in_enabled=False)
    )
    recognizer = _CollectingRecognizer()
    denoiser = _BlockingDenoiser()
    source = _DoubleRecoverySource(denoiser)
    media = CaptureMediaSession(
        source,
        block_seconds=engine.config.block_sec,
        buffer_ms=engine.config.media_pcm_queue_ms,
        max_frames=engine.config.media_pcm_queue_max_frames,
        context_provider=engine._snapshot_capture_context,
    )
    engine._recognizer = recognizer
    engine._denoiser = denoiser
    engine._stream_in = source
    engine._capture_sr = 16000
    engine._capture_media_session = media
    engine._reset_capture_frontends_after_reopen = lambda *, capture_sample_rate=None: (
        setattr(engine, "_capture_sr", int(capture_sample_rate or 16000))
    )
    engine._running.set()
    media.start()
    capture = threading.Thread(target=engine._capture_loop)
    capture.start()
    try:
        assert denoiser.entered.wait(1.0)
        assert source.recovered.wait(1.0)
        denoiser.release.set()
        deadline = time.monotonic() + 2.0
        while not recognizer.accepted and time.monotonic() < deadline:
            time.sleep(0.01)
        engine._running.clear()
        media.request_stop()
        source.release.set()
        capture.join(timeout=2.0)

        assert recognizer.accepted
        assert all(float(np.mean(block)) != 1.0 for block in recognizer.accepted)
        assert float(np.mean(recognizer.accepted[0])) == 2.0
    finally:
        engine._running.clear()
        denoiser.release.set()
        media.request_stop()
        source.release.set()
        media.join(timeout=1.0)
        capture.join(timeout=1.0)


def test_recovery_callback_runs_off_reader_and_stop_fences_followups() -> None:
    engine = SherpaOnnxEngine(SherpaConfig())
    engine._capture_control_lane = CaptureControlLane()
    entered = threading.Event()
    release = threading.Event()
    calls: list[str] = []

    def on_state(state: str, _message: str) -> None:
        calls.append(state)
        entered.set()
        release.wait(2.0)

    engine._cb = EngineCallbacks(on_capture_state=on_state)
    engine._running.set()
    reader = threading.Thread(
        target=engine._on_capture_state,
        args=("recovering", "device lost"),
    )
    reader.start()
    reader.join(timeout=0.2)
    assert not reader.is_alive()

    processor = threading.Thread(
        target=engine._drain_capture_control_events,
        args=(engine._capture_epoch,),
    )
    processor.start()
    assert entered.wait(1.0)
    stopper = threading.Thread(target=engine.stop)
    stopper.start()
    stopper.join(timeout=1.5)
    assert not stopper.is_alive()
    release.set()
    processor.join(timeout=1.0)

    assert calls == ["recovering"]
