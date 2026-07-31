from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading
import time

import numpy as np
import pytest

from tools.capture_replay import (
    FRAME_SAMPLES,
    SAMPLE_RATE_HZ,
    ReplayCaptureSession,
    load_corpus,
)


class _ManualClock:
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _track(path: Path, values: np.ndarray) -> dict[str, object]:
    raw = np.asarray(values, dtype="<f4").reshape(-1).tobytes()
    path.write_bytes(raw)
    return {
        "file": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "encoding": "f32le",
        "samples": len(raw) // 4,
    }


def _case(
    tmp_path: Path,
    *,
    references: bool = True,
    explicit_delayed: bool = True,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    samples = 2 * FRAME_SAMPLES
    mic_values = np.concatenate(
        (
            np.full(FRAME_SAMPLES, 0.1, dtype="float32"),
            np.full(FRAME_SAMPLES, 0.2, dtype="float32"),
        )
    )
    tracks: dict[str, object] = {
        "mic": _track(tmp_path / "mic.f32le", mic_values)
    }
    assertion = "transcript"
    delay = 0
    if references:
        assertion = "double_talk"
        delay = 80
        far_zero = np.concatenate(
            (
                np.full(FRAME_SAMPLES, 0.25, dtype="float32"),
                np.full(FRAME_SAMPLES, 0.5, dtype="float32"),
            )
        )
        tracks["far_reference_zero"] = _track(
            tmp_path / "far-zero.f32le",
            far_zero,
        )
        tracks["coherence_reference"] = _track(
            tmp_path / "coherence.f32le",
            np.full(samples, -0.2, dtype="float32"),
        )
        if explicit_delayed:
            tracks["far_reference_delayed"] = _track(
                tmp_path / "far-delayed.f32le",
                np.full(samples, 0.75, dtype="float32"),
            )
    payload = {
        "schema_version": 1,
        "purpose": "Replay session unit test.",
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "frame_samples": FRAME_SAMPLES,
        "cases": [
            {
                "id": "replay-case",
                "tracks": tracks,
                "speech_intervals": [
                    {"start_sample": 0, "end_sample": samples}
                ],
                "speaker_intervals": [
                    {
                        "speaker_id": "speaker-a",
                        "role": "target",
                        "start_sample": 0,
                        "end_sample": samples,
                    }
                ],
                "word_intervals": [
                    {
                        "speaker_id": "speaker-a",
                        "text": "hello",
                        "start_sample": 100,
                        "end_sample": 600,
                    },
                    {
                        "speaker_id": "speaker-a",
                        "text": "there",
                        "start_sample": 1_700,
                        "end_sample": 2_300,
                    },
                ],
                "assertion": assertion,
                "expected_text": "hello there",
                "commands": ["hello"],
                "tags": ["unit-test"],
                "aec_delay_samples": delay,
            }
        ],
    }
    manifest = tmp_path / "corpus.json"
    manifest.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return load_corpus(manifest).cases[0]


def test_replay_yields_exact_causal_frames_and_coordinates(tmp_path: Path) -> None:
    case = _case(tmp_path)
    clock = _ManualClock(100.0)
    running = threading.Event()
    running.set()
    holder: dict[str, ReplayCaptureSession] = {}
    exhausted_calls: list[float] = []

    def exhausted() -> None:
        session = holder["session"]
        assert not session.mailbox.closed
        exhausted_calls.append(clock())
        running.clear()

    session = ReplayCaptureSession(
        case,
        clock=clock,
        on_exhausted=exhausted,
        capture_epoch=7,
        source_generation=3,
        capture_generation=9,
        source_device="mic-replay",
    )
    holder["session"] = session
    session.start()

    assert session.timeline_base == 100.0
    assert session.take(timeout=0.0) is None
    assert not session.mailbox.closed
    clock.advance(0.1)
    first = session.take(timeout=0.0)

    assert first is not None
    assert first.sequence == 1
    assert first.capture_epoch == 7
    assert first.source_generation == 3
    assert first.capture_generation == 9
    assert first.source_device == "mic-replay"
    assert first.source_sample_start == 0
    assert first.source_sample_end == FRAME_SAMPLES
    assert first.captured_started_at == pytest.approx(100.0)
    assert first.captured_at == pytest.approx(100.1)
    assert first.duration_ms == pytest.approx(100.0)
    np.testing.assert_array_equal(
        first.pcm,
        np.full(FRAME_SAMPLES, 0.1, dtype="float32"),
    )
    assert session.frames_taken == 1
    assert session.frames_total == 2
    assert running.is_set()

    clock.advance(0.1)
    second = session.take(timeout=0.0)

    assert second is not None
    assert second.sequence == 2
    assert second.source_sample_start == FRAME_SAMPLES
    assert second.source_sample_end == 2 * FRAME_SAMPLES
    assert second.captured_started_at == pytest.approx(100.1)
    assert second.captured_at == pytest.approx(100.2)
    assert exhausted_calls == []
    assert running.is_set()
    assert not session.mailbox.closed

    assert session.take(timeout=0.0) is None
    assert exhausted_calls == [pytest.approx(100.2)]
    assert not running.is_set()
    assert session.naturally_exhausted
    assert session.mailbox.closed
    assert not session.is_alive
    assert session.take(timeout=0.0) is None
    assert session.error is None


def test_replay_attaches_owned_aligned_reference_context(tmp_path: Path) -> None:
    case = _case(tmp_path)
    clock = _ManualClock(20.0)
    session = ReplayCaptureSession(case, clock=clock)
    session.start()
    clock.advance(0.1)

    block = session.take(timeout=0.0)

    assert block is not None and block.context is not None
    context = block.context
    assert context.source_domain == "capture-replay"
    assert context.authority_source_generation == 1
    assert context.authority_source_device == "capture-replay"
    assert context.speaking
    assert context.barge_watch_active
    assert context.speak_generation == 1
    assert context.playback_generation == 1
    assert context.playback_level == pytest.approx(0.25)
    assert context.playback_onset_at == pytest.approx(20.0)
    assert context.last_playback_at == pytest.approx(20.1)
    assert context.aec_delay_samples == 80
    np.testing.assert_array_equal(
        context.far_reference_zero,
        np.full(FRAME_SAMPLES, 0.25, dtype="float32"),
    )
    np.testing.assert_array_equal(
        context.far_reference_delayed,
        np.full(FRAME_SAMPLES, 0.75, dtype="float32"),
    )
    np.testing.assert_array_equal(
        context.coherence_reference,
        np.full(FRAME_SAMPLES, -0.2, dtype="float32"),
    )
    assert context.far_reference_zero is not None
    assert not context.far_reference_zero.flags.writeable
    with pytest.raises(ValueError):
        context.far_reference_zero[0] = 1.0

    block.pcm.fill(9.0)
    clock.advance(0.1)
    second = session.take(timeout=0.0)
    assert second is not None
    np.testing.assert_array_equal(
        second.pcm,
        np.full(FRAME_SAMPLES, 0.2, dtype="float32"),
    )


def test_missing_optional_tracks_are_zero_fenced_and_delay_is_derived(
    tmp_path: Path,
) -> None:
    no_references = _case(tmp_path / "none", references=False)
    clock = _ManualClock(1.0)
    session = ReplayCaptureSession(no_references, clock=clock)
    session.start()
    clock.advance(0.1)
    plain = session.take(timeout=0.0)

    assert plain is not None and plain.context is not None
    assert not plain.context.speaking
    assert plain.context.far_reference_zero is not None
    assert plain.context.far_reference_delayed is not None
    assert plain.context.coherence_reference is not None
    assert np.count_nonzero(plain.context.far_reference_zero) == 0
    assert np.count_nonzero(plain.context.far_reference_delayed) == 0
    assert np.count_nonzero(plain.context.coherence_reference) == 0

    derived_case = _case(
        tmp_path / "derived",
        references=True,
        explicit_delayed=False,
    )
    clock = _ManualClock(5.0)
    derived_session = ReplayCaptureSession(derived_case, clock=clock)
    derived_session.start()
    clock.advance(0.1)
    derived = derived_session.take(timeout=0.0)

    assert derived is not None and derived.context is not None
    delayed = derived.context.far_reference_delayed
    assert delayed is not None
    np.testing.assert_array_equal(delayed[:80], np.zeros(80, dtype="float32"))
    np.testing.assert_array_equal(
        delayed[80:],
        np.full(FRAME_SAMPLES - 80, 0.25, dtype="float32"),
    )


def test_request_stop_interrupts_a_paced_take_without_exhaustion(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path, references=False)
    exhausted: list[bool] = []
    session = ReplayCaptureSession(
        case,
        on_exhausted=lambda: exhausted.append(True),
    )
    session.start()
    result: list[object] = []
    entered = threading.Event()

    def consume() -> None:
        entered.set()
        result.append(session.take(timeout=1.0))

    thread = threading.Thread(target=consume)
    thread.start()
    assert entered.wait(1.0)
    time.sleep(0.01)
    session.request_stop()
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert result == [None]
    assert session.mailbox.closed
    assert not session.naturally_exhausted
    assert exhausted == []
    assert session.error is None
    session.close()
    session.join(timeout=0.0)


def test_exhaustion_callback_failure_is_reported_as_session_error(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path, references=False)
    clock = _ManualClock(10.0)

    def fail() -> None:
        raise RuntimeError("runner lifecycle failed")

    session = ReplayCaptureSession(case, clock=clock, on_exhausted=fail)
    session.start()
    clock.advance(0.1)
    assert session.take(timeout=0.0) is not None
    clock.advance(0.1)

    assert session.take(timeout=0.0) is not None
    assert session.take(timeout=0.0) is None
    assert isinstance(session.error, RuntimeError)
    assert session.mailbox.closed
    assert not session.naturally_exhausted
    assert session.take(timeout=0.0) is None


def test_invalid_clock_closes_with_error_and_lifecycle_validation(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path, references=False)
    bad = ReplayCaptureSession(case, clock=lambda: float("nan"))

    assert bad.take(timeout=0.0) is None
    assert isinstance(bad.error, RuntimeError)
    assert bad.mailbox.closed

    clock = _ManualClock(3.0)
    session = ReplayCaptureSession(case, clock=clock)
    with pytest.raises(ValueError):
        session.take(timeout=-1.0)
    with pytest.raises(ValueError):
        session.take(timeout=float("nan"))
    session.start()
    with pytest.raises(RuntimeError):
        session.start()
    with pytest.raises(ValueError):
        session.join(timeout=-1.0)
