"""Capture-loop regressions for the load-bearing normal-final VAD gate."""
from __future__ import annotations

import threading
import time
from dataclasses import replace

import numpy as np
import pytest

from core.engine import EngineCallbacks
from core.engines._asr_segment import ASRSegment
from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine
from core.metrics import SPEECH_END


class _Input:
    def __init__(self, engine, blocks=1):
        self.engine = engine
        self.left = blocks

    def read(self, n):
        self.left -= 1
        if self.left <= 0:
            self.engine._running.clear()
        return np.zeros(n, dtype="float32"), False


class _Stream:
    def accept_waveform(self, sample_rate, samples):
        pass


class _HallucinatingRecognizer:
    """The exact live idle shape: stable raw ``AND`` + an endpoint."""
    def create_stream(self):
        return _Stream()

    def is_ready(self, stream):
        return False

    def decode_stream(self, stream):  # pragma: no cover - never ready
        pass

    def get_result(self, stream):
        return "AND"

    def is_endpoint(self, stream):
        return True

    def reset(self, stream):
        pass


class _Vad:
    def __init__(self, speech: bool):
        self.speech = speech
        self.accepted = 0

    def accept_waveform(self, samples):
        self.accepted += 1

    def is_speech_detected(self):
        return self.speech


def _run(*, vad_speech: bool | None):
    engine = SherpaOnnxEngine(SherpaConfig(endpoint_enabled=False))
    engine._recognizer = _HallucinatingRecognizer()
    engine._vad = _Vad(vad_speech) if vad_speech is not None else None
    engine._stream_in = _Input(engine)
    finals: list[str] = []
    metrics: list[str] = []
    engine._cb = EngineCallbacks(
        on_final=finals.append,
        on_metric=lambda name, **kwargs: metrics.append(name),
    )
    engine._running.set()
    thread = threading.Thread(target=engine._capture_loop)
    thread.start()
    thread.join(timeout=5.0)
    assert not thread.is_alive()
    return engine, finals, metrics


def test_idle_and_hallucination_is_dropped_when_vad_saw_no_speech():
    engine, finals, metrics = _run(vad_speech=False)
    assert finals == []
    assert metrics.count("vad_rejected_final") == 1
    assert engine._vad.accepted == 1


def test_same_recognizer_final_is_admitted_after_vad_speech():
    engine, finals, metrics = _run(vad_speech=True)
    assert finals == ["And"]
    assert "vad_rejected_final" not in metrics
    assert engine._vad.accepted == 1


def test_evidence_enabled_without_vad_explicitly_bypasses():
    _engine, finals, metrics = _run(vad_speech=None)
    assert finals == ["And"]
    assert "speech_evidence_rejected_final" not in metrics


_EVIDENCE_FRAME = 320


def _normalize_evidence(values, target_rms: float):
    values = np.asarray(values, dtype="float32")
    level = float(np.sqrt(np.mean(values.astype("float64") ** 2)))
    return (values * (target_rms / level)).astype("float32")


def _evidence_ambient_frame(rms: float = 0.001):
    t = np.arange(_EVIDENCE_FRAME, dtype="float64") / 16000.0
    return _normalize_evidence(
        np.sin(2 * np.pi * 5100 * t)
        + 0.7 * np.sin(2 * np.pi * 6200 * t + 0.3)
        + 0.4 * np.sin(2 * np.pi * 7100 * t + 0.8),
        rms,
    )


def _evidence_voice_frame(rms: float = 0.0022):
    t = np.arange(_EVIDENCE_FRAME, dtype="float64") / 16000.0
    envelope = np.sin(
        np.pi * (np.arange(_EVIDENCE_FRAME) + 0.5) / _EVIDENCE_FRAME
    ) ** 0.2
    return _normalize_evidence(
        envelope
        * (
            np.sin(2 * np.pi * 180 * t)
            + 0.55 * np.sin(2 * np.pi * 360 * t + 0.2)
            + 0.25 * np.sin(2 * np.pi * 540 * t + 0.5)
        ),
        rms,
    )


def _evidence_ambient_block(rms: float = 0.001):
    return np.tile(_evidence_ambient_frame(rms), 5)


def _short_yes_block(*, voice_rms: float = 0.0022):
    t = np.arange(_EVIDENCE_FRAME * 5, dtype="float64") / 16000.0
    frame_index = np.arange(_EVIDENCE_FRAME * 5) // _EVIDENCE_FRAME
    second = np.asarray((0.2, 0.75, 0.35, 0.8, 0.25))[frame_index]
    third = np.asarray((0.6, 0.2, 0.7, 0.25, 0.65))[frame_index]
    signal = (
        np.sin(2 * np.pi * 180 * t)
        + second * np.sin(2 * np.pi * 360 * t + 0.2)
        + third * np.sin(2 * np.pi * 540 * t + 0.5)
    )
    return np.concatenate(
        [
            _normalize_evidence(
                signal[
                    index * _EVIDENCE_FRAME : (index + 1) * _EVIDENCE_FRAME
                ],
                voice_rms,
            )
            for index in range(5)
        ]
    )


def _run_calibrated_evidence(
    *,
    raw_blocks: list[np.ndarray],
    text: str,
    ambient_rms: float | None,
    input_gain: float = 1.0,
    capture_sample_rate: int = 16000,
    pre_gain_resampler=None,
    live_resampler=None,
    enqueue_sink: list[str] | None = None,
    calibration_pcm: list[np.ndarray] | None = None,
    confirmed_barge_handoff: bool = False,
    mismatch_confirm_generation: bool = False,
    expire_confirm_handoff: bool = False,
):
    class _EvidenceInput:
        generation = 0

        def __init__(self, engine):
            self.engine = engine
            self.blocks = [np.asarray(block, dtype="float32") for block in raw_blocks]

        def read(self, n):
            block = self.blocks.pop(0)
            assert block.shape == (n,)
            if not self.blocks:
                self.engine._running.clear()
            return block.copy(), False

    class _EvidenceStream(_Stream):
        def __init__(self):
            self.blocks = 0

        def accept_waveform(self, _sample_rate, _samples):
            self.blocks += 1

    class _EvidenceRecognizer:
        def __init__(self):
            self.stream = _EvidenceStream()

        def create_stream(self):
            return self.stream

        def is_ready(self, _stream):
            return False

        def decode_stream(self, _stream):  # pragma: no cover - never ready
            pass

        def get_result(self, stream):
            return text if stream.blocks else ""

        def is_endpoint(self, stream):
            return stream.blocks >= len(raw_blocks)

        def reset(self, stream):
            stream.blocks = 0

    engine = SherpaOnnxEngine(
        SherpaConfig(
            endpoint_enabled=False,
            input_calibrate=False,
            input_gain=input_gain,
            final_speech_evidence_enabled=True,
        )
    )
    engine._recognizer = _EvidenceRecognizer()
    engine._vad = _Vad(True)
    engine._stream_in = _EvidenceInput(engine)
    engine._capture_sr = int(capture_sample_rate)
    engine._pre_gain_resampler = pre_gain_resampler
    engine._resampler = live_resampler
    if confirmed_barge_handoff:
        handoff_at = time.perf_counter()
        assert engine._publish_confirm_handoff_if_current(
            [raw_blocks[0]],
            [raw_blocks[0]],
            speech_at=handoff_at,
            speech_end_at=handoff_at,
        )
        if mismatch_confirm_generation:
            engine._stream_in.generation = 1
        if expire_confirm_handoff:
            engine._confirm_handoff_pending = replace(
                engine._confirm_handoff_pending,
                expires_at=0.0,
            )
    if ambient_rms is not None:
        engine._install_speech_evidence_profile(
            {"ambient_rms": ambient_rms, "clipping_fraction": 0.0},
            calibration_pcm
            if calibration_pcm is not None
            else [_evidence_ambient_block(), _evidence_ambient_block()],
        )
    partials: list[str] = []
    finals: list[str] = []
    metrics: list[str] = []
    engine._cb = EngineCallbacks(
        on_partial=partials.append,
        on_final=finals.append,
        on_metric=lambda name, **_kwargs: metrics.append(name),
    )
    if enqueue_sink is not None:
        engine._final_q = object()
        engine._enqueue_final = (
            lambda _seg, raw, *_args, **_kwargs: enqueue_sink.append(raw)
        )

    engine._running.set()
    thread = threading.Thread(target=engine._capture_loop)
    thread.start()
    thread.join(timeout=5.0)
    assert not thread.is_alive()
    return partials, finals, metrics


def test_one_frame_impulse_cannot_publish_partial_or_final():
    impulse = _evidence_ambient_block()
    impulse[_EVIDENCE_FRAME // 2] = 0.5
    partials, finals, metrics = _run_calibrated_evidence(
        raw_blocks=[impulse],
        text="AND",
        ambient_rms=0.001,
    )

    assert partials == []
    assert finals == []
    assert metrics.count("speech_evidence_rejected_final") == 1


def test_quiet_short_yes_pattern_publishes_within_one_capture_block():
    partials, finals, metrics = _run_calibrated_evidence(
        raw_blocks=[_short_yes_block()],
        text="YES",
        ambient_rms=0.001,
    )

    assert partials == ["Yes"]
    assert finals == ["Yes"]
    assert "speech_evidence_rejected_final" not in metrics


def test_missing_calibration_explicitly_fails_open():
    partials, finals, metrics = _run_calibrated_evidence(
        raw_blocks=[np.zeros(1600, dtype="float32")],
        text="AND",
        ambient_rms=None,
    )

    assert partials == ["And"]
    assert finals == ["And"]
    assert metrics.count("speech_evidence_unavailable_fail_open") == 1


def test_confirmed_barge_handoff_bypasses_armed_ordinary_evidence_gate():
    partials, finals, metrics = _run_calibrated_evidence(
        raw_blocks=[_evidence_ambient_block()],
        text="STOP",
        ambient_rms=0.001,
        confirmed_barge_handoff=True,
    )

    assert partials == ["Stop"]
    assert finals == ["Stop"]
    assert "speech_evidence_rejected_final" not in metrics


def test_confirmed_barge_handoff_cannot_cross_capture_generation():
    partials, finals, metrics = _run_calibrated_evidence(
        raw_blocks=[_evidence_ambient_block()],
        text="STOP",
        ambient_rms=0.001,
        confirmed_barge_handoff=True,
        mismatch_confirm_generation=True,
    )

    assert partials == []
    assert finals == []
    assert metrics.count("barge_confirm_handoff_stale") == 1
    assert metrics.count("speech_evidence_rejected_final") == 1


def test_confirmed_barge_handoff_expires_before_unrelated_turn():
    partials, finals, metrics = _run_calibrated_evidence(
        raw_blocks=[_evidence_ambient_block()],
        text="STOP",
        ambient_rms=0.001,
        confirmed_barge_handoff=True,
        expire_confirm_handoff=True,
    )

    assert partials == []
    assert finals == []
    assert metrics.count("barge_confirm_handoff_stale") == 1
    assert metrics.count("speech_evidence_rejected_final") == 1


def test_short_stop_transcript_passes_an_armed_profile():
    partials, finals, metrics = _run_calibrated_evidence(
        raw_blocks=[_short_yes_block()],
        text="STOP",
        ambient_rms=0.001,
    )

    assert partials == ["Stop"]
    assert finals == ["Stop"]
    assert "speech_evidence_rejected_final" not in metrics


def test_static_gain_cannot_manufacture_pre_gain_speech_evidence():
    partials, finals, metrics = _run_calibrated_evidence(
        raw_blocks=[_evidence_ambient_block(rms=0.003)],
        text="AND",
        ambient_rms=0.001,
        input_gain=12.0,
    )

    assert partials == []
    assert finals == []
    assert metrics.count("speech_evidence_rejected_final") == 1


def test_static_gain_cannot_lift_subthreshold_voice_into_evidence():
    partials, finals, metrics = _run_calibrated_evidence(
        raw_blocks=[_short_yes_block(voice_rms=0.0015)],
        text="YES",
        ambient_rms=0.001,
        input_gain=12.0,
    )

    assert partials == []
    assert finals == []
    assert metrics.count("speech_evidence_rejected_final") == 1


def test_native_rate_pre_gain_and_live_paths_use_independent_resamplers_once():
    class _Decimator:
        kind = "test-decimator"

        def __init__(self):
            self.calls: list[int] = []

        def process(self, samples):
            block = np.asarray(samples, dtype="float32")
            self.calls.append(block.size)
            return block[::3]

    evidence_resampler = _Decimator()
    live_resampler = _Decimator()
    partials, finals, metrics = _run_calibrated_evidence(
        raw_blocks=[np.repeat(_short_yes_block(), 3)],
        text="YES",
        ambient_rms=0.001,
        capture_sample_rate=48000,
        pre_gain_resampler=evidence_resampler,
        live_resampler=live_resampler,
    )

    assert evidence_resampler.calls == [4800]
    assert live_resampler.calls == [4800]
    assert partials == ["Yes"]
    assert finals == ["Yes"]
    assert "speech_evidence_rejected_final" not in metrics


def test_speech_evidence_rejects_before_async_enqueue():
    enqueued: list[str] = []
    impulse = _evidence_ambient_block()
    impulse[_EVIDENCE_FRAME // 2] = 0.5
    _run_calibrated_evidence(
        raw_blocks=[impulse],
        text="AND",
        ambient_rms=0.001,
        enqueue_sink=enqueued,
    )
    assert enqueued == []

    _run_calibrated_evidence(
        raw_blocks=[_short_yes_block()],
        text="YES",
        ambient_rms=0.001,
        enqueue_sink=enqueued,
    )
    assert enqueued == ["YES"]


def test_vad_onset_rebase_replays_alternate_asr_domain_not_primary():
    class _RecordingStream(_Stream):
        def __init__(self):
            self.blocks: list[np.ndarray] = []

        def accept_waveform(self, _sample_rate, samples):
            self.blocks.append(np.asarray(samples, dtype="float32").copy())

    class _Recognizer:
        def __init__(self):
            self.resets = 0

        def reset(self, _stream):
            self.resets += 1

        def is_ready(self, _stream):
            return False

        def decode_stream(self, _stream):  # pragma: no cover - never ready
            pass

    segment = ASRSegment(
        sample_rate=16000,
        pre_roll_sec=0.2,
        max_utterance_sec=2.0,
        vad_available=True,
        block_sec=0.1,
    )
    segment.append(
        np.full(1600, 0.11, dtype="float32"),
        np.full(1600, 0.71, dtype="float32"),
    )
    segment.append(
        np.full(1600, 0.22, dtype="float32"),
        np.full(1600, 0.82, dtype="float32"),
    )
    engine = SherpaOnnxEngine(SherpaConfig())
    recognizer = _Recognizer()
    stream = _RecordingStream()

    replayed = engine._rebase_normal_asr_stream(
        recognizer, stream, segment
    )

    assert recognizer.resets == 1
    assert replayed == 3200
    assert len(stream.blocks) == 1
    np.testing.assert_array_equal(
        stream.blocks[0],
        np.concatenate(
            [
                np.full(1600, 0.71, dtype="float32"),
                np.full(1600, 0.82, dtype="float32"),
            ]
        ),
    )
    primary, alternate = segment.arrays()
    assert np.any(np.isclose(primary, 0.11))
    assert alternate is not None and np.any(np.isclose(alternate, 0.71))


def test_mid_utterance_pause_resume_does_not_rebase_a_second_time():
    class _PauseInput:
        generation = 0

        def __init__(self, engine):
            self.engine = engine
            self.values = [0.11, 0.0, 0.22, 0.0]

        def read(self, n):
            value = self.values.pop(0)
            if not self.values:
                self.engine._running.clear()
            return np.full(n, value, dtype="float32"), False

    class _PauseStream:
        def __init__(self):
            self.generation = 0
            self.accepted = {0: []}

        def accept_waveform(self, _sample_rate, samples):
            self.accepted.setdefault(self.generation, []).append(
                np.asarray(samples, dtype="float32").copy()
            )

    class _PauseRecognizer:
        def __init__(self):
            self.stream = _PauseStream()
            self.resets = 0

        def create_stream(self):
            return self.stream

        def is_ready(self, _stream):
            return False

        def decode_stream(self, _stream):  # pragma: no cover - never ready
            pass

        def _values(self, stream):
            blocks = stream.accepted.get(stream.generation, [])
            return np.concatenate(blocks) if blocks else np.zeros(0, dtype="float32")

        def get_result(self, stream):
            values = self._values(stream)
            if np.any(np.isclose(values, 0.22)):
                return "HELLO AGAIN"
            return "HELLO" if np.any(np.isclose(values, 0.11)) else ""

        def is_endpoint(self, stream):
            values = self._values(stream)
            return (
                np.any(np.isclose(values, 0.22))
                and np.count_nonzero(np.isclose(values, 0.0)) >= 3200
            )

        def reset(self, stream):
            self.resets += 1
            stream.generation += 1
            stream.accepted.setdefault(stream.generation, [])

    class _PauseVad(_Vad):
        def __init__(self):
            super().__init__(False)
            self.states = iter([True, False, True, False])

        def accept_waveform(self, samples):
            super().accept_waveform(samples)
            self.speech = next(self.states)

        def reset(self):
            pass

    engine = SherpaOnnxEngine(SherpaConfig(endpoint_enabled=False))
    recognizer = _PauseRecognizer()
    engine._recognizer = recognizer
    engine._vad = _PauseVad()
    engine._stream_in = _PauseInput(engine)
    partials: list[str] = []
    finals: list[str] = []
    engine._cb = EngineCallbacks(
        on_partial=partials.append,
        on_final=finals.append,
    )

    engine._running.set()
    thread = threading.Thread(target=engine._capture_loop)
    thread.start()
    thread.join(timeout=5.0)

    assert not thread.is_alive()
    assert partials == ["Hello", "Hello again"]
    assert finals == ["Hello again"]
    assert recognizer.resets == 2  # one onset rebase + one endpoint reset
    continuous = np.concatenate(recognizer.stream.accepted[1])
    assert np.any(np.isclose(continuous, 0.11))
    assert np.any(np.isclose(continuous, 0.22))


@pytest.mark.parametrize(("command", "expected"), [("YES", "Yes"), ("STOP", "Stop")])
def test_first_vad_onset_rebases_stale_decoder_and_replays_bounded_preroll_once(
    command, expected
):
    """Run-200747 regression: pre-VAD text cannot join a later speech epoch."""

    class _ScriptedInput:
        generation = 0

        def __init__(self, engine):
            self.engine = engine
            self.blocks = [0.01, 0.02, 0.03, 0.11, 0.22, 0.0]

        def read(self, n):
            value = self.blocks.pop(0)
            if not self.blocks:
                self.engine._running.clear()
            return np.full(n, value, dtype="float32"), False

    class _EpochStream:
        def __init__(self):
            self.generation = 0
            self.accepted = {0: []}

        def accept_waveform(self, _sample_rate, samples):
            self.accepted.setdefault(self.generation, []).append(
                np.asarray(samples, dtype="float32").copy()
            )

    class _EpochRecognizer:
        def __init__(self):
            self.stream = _EpochStream()
            self.resets = 0

        def create_stream(self):
            return self.stream

        def is_ready(self, _stream):
            return False

        def decode_stream(self, _stream):  # pragma: no cover - never ready
            pass

        def _values(self, stream):
            blocks = stream.accepted.get(stream.generation, [])
            return np.concatenate(blocks) if blocks else np.zeros(0, dtype="float32")

        def get_result(self, stream):
            values = self._values(stream)
            if stream.generation == 0 and np.any(np.isclose(values, 0.02)):
                return "MA"
            if (
                stream.generation == 1
                and np.any(np.isclose(values, 0.11))
                and np.any(np.isclose(values, 0.22))
            ):
                return command
            return ""

        def is_endpoint(self, stream):
            values = self._values(stream)
            return (
                stream.generation == 1
                and values.size > 0
                and np.any(np.isclose(values, 0.0))
            )

        def reset(self, stream):
            self.resets += 1
            stream.generation += 1
            stream.accepted.setdefault(stream.generation, [])

    class _ScriptedVad:
        def __init__(self):
            self.states = iter([False, False, False, False, True, False])
            self.active = False
            self.resets = 0

        def accept_waveform(self, _samples):
            self.active = next(self.states)

        def is_speech_detected(self):
            return self.active

        def reset(self):
            self.resets += 1

    engine = SherpaOnnxEngine(
        SherpaConfig(
            endpoint_enabled=False,
            asr_final_preroll_sec=0.2,
        )
    )
    recognizer = _EpochRecognizer()
    engine._recognizer = recognizer
    engine._vad = _ScriptedVad()
    engine._stream_in = _ScriptedInput(engine)
    partials: list[str] = []
    finals: list[str] = []
    metrics: list[str] = []
    finalized_pcm: list[np.ndarray] = []
    original_finalize = engine._finalize_and_dispatch

    def _capture_finalize(seg, *args, **kwargs):
        finalized_pcm.append(np.asarray(seg, dtype="float32").copy())
        return original_finalize(seg, *args, **kwargs)

    engine._finalize_and_dispatch = _capture_finalize
    engine._cb = EngineCallbacks(
        on_partial=partials.append,
        on_final=finals.append,
        on_metric=lambda name, **_kwargs: metrics.append(name),
    )

    engine._running.set()
    thread = threading.Thread(target=engine._capture_loop)
    thread.start()
    thread.join(timeout=5.0)

    assert not thread.is_alive()
    assert partials == [expected]
    assert finals == [expected]
    assert all("ma" not in text.lower() for text in partials + finals)
    assert metrics.count(SPEECH_END) == 1
    assert "vad_rejected_final" not in metrics
    assert recognizer.resets == 2  # first onset rebase + ordinary endpoint reset

    fresh = np.concatenate(recognizer.stream.accepted[1])
    # Only the bounded 200 ms lookback is replayed; old 0.01/0.02 history is gone.
    assert not np.any(np.isclose(fresh, 0.01))
    assert not np.any(np.isclose(fresh, 0.02))
    for value in (0.03, 0.11, 0.22, 0.0):
        assert np.count_nonzero(np.isclose(fresh, value)) == 1600
    assert len(finalized_pcm) == 1
    for value in (0.03, 0.11, 0.22, 0.0):
        assert np.count_nonzero(np.isclose(finalized_pcm[0], value)) == 1600


def test_empty_vad_blip_is_reset_after_endpoint_ceiling_without_a_final():
    class _SlowInput:
        generation = 0

        def __init__(self, engine):
            self.engine = engine
            self.left = 4

        def read(self, n):
            time.sleep(0.03)
            self.left -= 1
            if self.left <= 0:
                self.engine._running.clear()
            return np.full(n, 0.01, dtype="float32"), False

    class _EmptyRecognizer(_HallucinatingRecognizer):
        def __init__(self):
            self.resets = 0

        def get_result(self, stream):
            return ""

        def is_endpoint(self, stream):
            return False

        def reset(self, stream):
            self.resets += 1

    class _BlipVad(_Vad):
        def __init__(self):
            super().__init__(False)
            self.states = iter([True, False, False, False])
            self.resets = 0

        def accept_waveform(self, samples):
            super().accept_waveform(samples)
            self.speech = next(self.states)

        def reset(self):
            self.resets += 1

    engine = SherpaOnnxEngine(
        SherpaConfig(
            endpoint_enabled=False,
            block_sec=0.01,
            endpoint_max_silence_sec=0.02,
        )
    )
    recognizer = _EmptyRecognizer()
    engine._recognizer = recognizer
    engine._vad = _BlipVad()
    engine._stream_in = _SlowInput(engine)
    finals: list[str] = []
    metrics: list[str] = []
    engine._cb = EngineCallbacks(
        on_final=finals.append,
        on_metric=lambda name, **_kwargs: metrics.append(name),
    )

    engine._running.set()
    thread = threading.Thread(target=engine._capture_loop)
    thread.start()
    thread.join(timeout=5.0)

    assert not thread.is_alive()
    assert finals == []
    assert metrics.count("vad_abandoned_epoch_reset") == 1
    assert recognizer.resets == 2  # onset rebase + abandoned episode reset


def test_confirmed_barge_stream_is_adopted_instead_of_erased_at_vad_onset():
    """Bounded confirm PCM is adopted even when the next VAD block is quiet."""

    class _TwoBlockInput:
        generation = 0

        def __init__(self, engine):
            self.engine = engine
            self.values = [0.41, 0.22]

        def read(self, n):
            value = self.values.pop(0)
            if not self.values:
                self.engine._running.clear()
            return np.full(n, value, dtype="float32"), False

    class _ConfirmStream:
        def __init__(self):
            self.generation = 0
            self.accepted = {0: []}

        def accept_waveform(self, _sample_rate, samples):
            self.accepted.setdefault(self.generation, []).append(
                np.asarray(samples, dtype="float32").copy()
            )

    class _ConfirmRecognizer:
        def __init__(self):
            self.stream = _ConfirmStream()
            self.resets = 0

        def create_stream(self):
            return self.stream

        def is_ready(self, _stream):
            return False

        def decode_stream(self, _stream):  # pragma: no cover - never ready
            pass

        def _values(self, stream):
            blocks = stream.accepted.get(stream.generation, [])
            return np.concatenate(blocks) if blocks else np.zeros(0, dtype="float32")

        def get_result(self, stream):
            return "STOP" if np.any(np.isclose(self._values(stream), 0.41)) else ""

        def is_endpoint(self, stream):
            values = self._values(stream)
            return (
                np.any(np.isclose(values, 0.41))
                and np.any(np.isclose(values, 0.22))
            )

        def reset(self, stream):
            self.resets += 1
            stream.generation += 1
            stream.accepted.setdefault(stream.generation, [])

    engine = SherpaOnnxEngine(
        SherpaConfig(
            endpoint_enabled=False,
            barge_in_enabled=True,
            barge_confirm_enabled=True,
        )
    )
    recognizer = _ConfirmRecognizer()
    engine._recognizer = recognizer
    engine._vad = _Vad(False)
    engine._stream_in = _TwoBlockInput(engine)
    engine._install_speech_evidence_profile(
        {"ambient_rms": 0.001, "clipping_fraction": 0.0},
        [_evidence_ambient_block(), _evidence_ambient_block()],
    )
    engine._barge_watch_active = lambda: True
    barges: list[str] = []
    partials: list[str] = []
    finals: list[str] = []

    def _barge():
        barges.append("barge")
        engine._speaking.clear()

    engine._cb = EngineCallbacks(
        on_barge_in=_barge,
        on_partial=partials.append,
        on_final=finals.append,
    )
    engine._speaking.set()
    engine._begin_barge_confirm(
        recognizer, recognizer.stream, time.monotonic()
    )

    engine._running.set()
    thread = threading.Thread(target=engine._capture_loop)
    thread.start()
    thread.join(timeout=5.0)

    assert not thread.is_alive()
    assert barges == ["barge"]
    assert partials == ["Stop"]
    assert finals == ["Stop"]
    assert recognizer.resets == 1  # ordinary endpoint only; no onset rebase
    confirmed = np.concatenate(recognizer.stream.accepted[0])
    assert np.count_nonzero(np.isclose(confirmed, 0.41)) == 1600
    assert np.count_nonzero(np.isclose(confirmed, 0.22)) == 1600
    assert not engine._confirm_handoff_stream_live
    assert engine._confirm_handoff_pending is None


def test_stop_drops_block_that_returns_after_capture_shutdown_signal():
    """An abort-unblocked stale block cannot re-enter ASR during teardown."""
    read_started = threading.Event()
    release_read = threading.Event()

    class _BlockingInput:
        generation = 0

        def read(self, n):
            read_started.set()
            assert release_read.wait(timeout=2.0)
            return np.ones(n, dtype="float32"), False

        def request_close(self):
            release_read.set()

        def close(self):
            return True

    engine = SherpaOnnxEngine(SherpaConfig(endpoint_enabled=False, block_sec=0.02))
    engine._recognizer = _HallucinatingRecognizer()
    engine._vad = _Vad(True)
    engine._stream_in = _BlockingInput()
    engine._cb = EngineCallbacks()
    engine._running.set()
    engine._capture_thread = threading.Thread(target=engine._capture_loop, daemon=True)
    engine._capture_thread.start()
    assert read_started.wait(timeout=1.0)

    engine.stop()

    assert not engine._capture_thread.is_alive()
    assert engine._vad.accepted == 0


def test_stop_after_initial_read_check_fences_post_dsp_side_effects():
    """Shutdown during DSP drops the block before recorder/KWS/ASR callbacks."""
    dsp_entered = threading.Event()
    release_dsp = threading.Event()

    class _Input:
        generation = 0

        def read(self, n):
            return np.ones(n, dtype="float32"), False

        def request_close(self):
            pass

        def close(self):
            return True

    class _Denoiser:
        def process_16k(self, samples):
            dsp_entered.set()
            assert release_dsp.wait(timeout=2.0)
            return samples

        def reset(self):
            pass

    class _Recorder:
        seconds = 0.1
        path = "headless.wav"

        def __init__(self):
            self.writes = 0
            self.closes = 0

        def write(self, _samples):
            self.writes += 1

        def close(self):
            self.closes += 1

    events = []
    recorder = _Recorder()
    engine = SherpaOnnxEngine(SherpaConfig(endpoint_enabled=False, block_sec=0.02))
    engine._recognizer = _HallucinatingRecognizer()
    engine._vad = _Vad(True)
    engine._denoiser = _Denoiser()
    engine._stream_in = _Input()
    engine._recorder = recorder
    engine._poll_keywords = lambda _samples: events.append("kws")
    engine._cb = EngineCallbacks(
        on_partial=lambda _text: events.append("partial"),
        on_final=lambda _text: events.append("final"),
        on_command=lambda _text: events.append("command"),
        on_heartbeat=lambda: events.append("heartbeat"),
        on_barge_in=lambda: events.append("barge"),
    )
    engine._running.set()
    engine._capture_thread = threading.Thread(target=engine._capture_loop, daemon=True)
    engine._capture_thread.start()
    assert dsp_entered.wait(timeout=1.0)

    stopper = threading.Thread(target=engine.stop, daemon=True)
    stopper.start()
    assert engine._capture_stopping.wait(timeout=1.0)
    release_dsp.set()
    stopper.join(timeout=2.0)

    assert not stopper.is_alive()
    assert not engine._capture_thread.is_alive()
    assert recorder.writes == 0
    assert recorder.closes == 1
    assert events == []


def test_stop_retains_resources_while_admitted_capture_effect_is_stuck(monkeypatch):
    """A block admitted before stop keeps every resource it may still touch."""
    from core.engines import sherpa as sherpa_module

    monkeypatch.setattr(sherpa_module, "_CAPTURE_FORCE_JOIN_TIMEOUT_SEC", 0.02)
    write_entered = threading.Event()
    release_write = threading.Event()

    class _Input:
        generation = 0

        def __init__(self):
            self.close_calls = 0

        def read(self, n):
            return np.ones(n, dtype="float32"), False

        def request_close(self):
            pass

        def abort_read(self, *, timeout):
            return False

        def close(self):
            self.close_calls += 1
            return True

    class _Recorder:
        seconds = 0.1
        path = "headless.wav"

        def __init__(self):
            self.close_calls = 0

        def write(self, _samples):
            write_entered.set()
            assert release_write.wait(timeout=2.0)

        def close(self):
            self.close_calls += 1

    class _Output:
        def __init__(self):
            self.stop_calls = 0
            self.close_calls = 0

        def stop(self):
            self.stop_calls += 1

        def close(self):
            self.close_calls += 1

    stream = _Input()
    recorder = _Recorder()
    output = _Output()
    engine = SherpaOnnxEngine(SherpaConfig(endpoint_enabled=False, block_sec=0.01))
    engine._recognizer = _HallucinatingRecognizer()
    engine._vad = _Vad(True)
    engine._stream_in = stream
    engine._recorder = recorder
    engine._out_stream = output
    engine._cb = EngineCallbacks()
    engine._running.set()
    engine._capture_thread = threading.Thread(target=engine._capture_loop, daemon=True)
    engine._capture_thread.start()
    assert write_entered.wait(timeout=1.0)

    started = time.monotonic()
    engine.stop()
    elapsed = time.monotonic() - started

    assert elapsed < 0.5
    assert engine._capture_resource_hold.is_set()
    assert engine._stream_in is stream
    assert stream.close_calls == 0
    assert recorder.close_calls == 0
    assert output.stop_calls == 0 and output.close_calls == 0

    release_write.set()
    engine._capture_thread.join(timeout=1.0)
    assert not engine._capture_thread.is_alive()
    engine.stop()
    assert not engine._capture_resource_hold.is_set()
    assert engine._stream_in is None
    assert stream.close_calls == 1
    assert recorder.close_calls == 1
    assert output.stop_calls == 1 and output.close_calls == 1
    assert engine._out_stream is None


def test_active_vad_rule3_endpoint_finalizes_and_resets_bounded_segment():
    from core.endpointing import ScriptedTurnCompletionDetector

    class _Rule3Recognizer(_HallucinatingRecognizer):
        def __init__(self):
            self.resets = 0

        def get_result(self, stream):
            return "and then"

        def reset(self, stream):
            self.resets += 1

    detector = ScriptedTurnCompletionDetector({"and then": 0.05})
    engine = SherpaOnnxEngine(
        SherpaConfig(endpoint_enabled=True, endpoint_max_silence_sec=1.6),
        turn_detector=detector,
    )
    recognizer = _Rule3Recognizer()
    engine._recognizer = recognizer
    engine._vad = _Vad(True)  # continuous speech: no semantic-silence clock
    engine._stream_in = _Input(engine)
    finals: list[str] = []
    engine._cb = EngineCallbacks(on_final=finals.append)

    engine._running.set()
    thread = threading.Thread(target=engine._capture_loop)
    thread.start()
    thread.join(timeout=5.0)

    assert not thread.is_alive()
    assert finals == ["And then"]
    assert recognizer.resets == 2  # first VAD-onset rebase + rule-3 endpoint
    assert detector.calls == []


def test_capture_reopen_preserves_recovered_block_after_rebinding_domain():
    from core.engines.speaker_gate import SpeakerGate

    class _RecoveringInput:
        def __init__(self, engine):
            self.engine = engine
            self.generation = 1
            self.actual_samplerate = 16000
            self.actual_device = "preferred"
            self.read_sizes = []
            self.calls = 0

        def read(self, n):
            self.read_sizes.append(n)
            self.calls += 1
            if self.calls == 1:
                return np.full(n, 0.11, dtype="float32"), False
            if self.calls == 2:
                self.generation = 2
                self.actual_samplerate = 48000
                self.actual_device = None
                # Mirror _RecoveringInputStream: the read was initiated with the
                # old frame count, but its internal retry returns one correctly
                # timed 100 ms block at the recovered rate.
                return np.full(4800, 0.22, dtype="float32"), False
            self.engine._running.clear()
            return np.full(n, 0.33, dtype="float32"), False

    class _Recognizer:
        def __init__(self):
            self.accepted = []
            self.resets = 0

        def create_stream(self):
            return _Stream()

        def is_ready(self, stream):
            return False

        def decode_stream(self, stream):  # pragma: no cover
            pass

        def get_result(self, stream):
            return "hello" if self.accepted else ""

        def is_endpoint(self, stream):
            # The first reset now binds the initial stream to the first VAD
            # episode. Only a subsequent recovery-domain reset may make this
            # fixture endpoint; do not mistake onset rebasing for recovery.
            return self.resets >= 2 and bool(self.accepted)

        def reset(self, stream):
            self.resets += 1
            self.accepted.clear()

    class _SpeechVad(_Vad):
        def __init__(self):
            super().__init__(True)
            self.resets = 0

        def reset(self):
            self.resets += 1

    recognizer = _Recognizer()

    def accept(_sr, samples):
        recognizer.accepted.append(np.asarray(samples).copy())

    engine = SherpaOnnxEngine(SherpaConfig(endpoint_enabled=False))
    stream = _Stream()
    stream.accept_waveform = accept
    recognizer.create_stream = lambda: stream
    engine._recognizer = recognizer
    engine._vad = _SpeechVad()
    engine._stream_in = _RecoveringInput(engine)
    engine._capture_sr = 16000
    gate = SpeakerGate(threshold=0.5, embed_fn=lambda _s, _sr: [1.0])
    gate.enroll_embedding([1.0])
    engine._speaker_gate = gate
    resolved = []
    engine._resolve_capture_domain = (
        lambda _sd, selector, **_kwargs: resolved.append(selector) or False
    )
    finals = []
    engine._finalize_and_dispatch = (
        lambda seg, raw, speech_end, asr_seg=None, speech_sec=None: finals.append(
            np.asarray(seg).copy()
        )
    )
    engine._cb = EngineCallbacks()

    engine._running.set()
    thread = threading.Thread(target=engine._capture_loop)
    thread.start()
    thread.join(timeout=5.0)
    assert not thread.is_alive()

    assert engine._stream_in.read_sizes == [1600, 1600, 4800]
    assert engine._capture_sr == 48000
    assert engine._resampler is not None and engine._resampler.src_sr == 48000
    assert recognizer.resets >= 1
    assert resolved == [None]
    assert not gate.is_enrolled
    assert finals and not np.any(np.isclose(finals[0], 0.11))
    assert np.any(np.isclose(finals[0], 0.22, atol=1e-3))
