from __future__ import annotations

from pytest import approx

from core.metrics import (
    ASR_FINAL,
    BARGE_IN,
    BARGE_IN_STOP,
    LLM_FIRST_TOKEN,
    SPEECH_END,
    TTS_FIRST_AUDIO,
    MetricsRecorder,
    TurnRecord,
    mark_first_token,
)


class FakeClock:
    """Deterministic monotonic clock advanced by hand."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def tick(self, dt: float) -> float:
        self.t += dt
        return self.t


def test_turn_record_computes_deltas():
    rec = TurnRecord(
        stamps={
            SPEECH_END: 1.0,
            ASR_FINAL: 1.2,
            LLM_FIRST_TOKEN: 1.7,
            TTS_FIRST_AUDIO: 2.0,
        }
    )
    assert rec.endpoint_latency == approx(0.2)
    assert rec.final_to_first_token == approx(0.5)
    assert rec.first_token_to_audio == approx(0.3)
    # first audio is measured from speech_end (the user-perceived anchor).
    assert rec.first_audio_latency == approx(1.0)
    assert rec.barge_in_latency is None


def test_first_audio_falls_back_to_asr_final_without_speech_end():
    rec = TurnRecord(stamps={ASR_FINAL: 5.0, TTS_FIRST_AUDIO: 6.5})
    assert rec.first_audio_latency == approx(1.5)


def test_recorder_separates_turns_on_repeated_start_stage():
    clock = FakeClock()
    rec = MetricsRecorder(clock=clock)

    clock.tick(0.0); rec.mark(SPEECH_END)
    clock.tick(0.2); rec.mark(ASR_FINAL)
    clock.tick(0.6); rec.mark(TTS_FIRST_AUDIO)
    # second utterance -- a repeat speech_end banks the first turn.
    clock.tick(1.0); rec.mark(SPEECH_END)
    clock.tick(0.1); rec.mark(ASR_FINAL)

    records = rec.records()
    assert len(records) == 2
    assert records[0].first_audio_latency == approx(0.8)
    assert records[1].endpoint_latency == approx(0.1)


def test_recorder_marks_once_and_ignores_stray_midturn_events():
    clock = FakeClock()
    rec = MetricsRecorder(clock=clock)
    # No open turn yet: a mid-turn stamp is dropped.
    rec.mark(TTS_FIRST_AUDIO)
    assert rec.records() == []

    rec.mark(ASR_FINAL)
    clock.tick(0.3); rec.mark(TTS_FIRST_AUDIO)
    clock.tick(0.3); rec.mark(TTS_FIRST_AUDIO)  # second one ignored
    [record] = rec.records()
    assert record.first_audio_latency == approx(0.3)


def test_recorder_barge_in_latency():
    clock = FakeClock()
    rec = MetricsRecorder(clock=clock)
    rec.mark(ASR_FINAL)
    clock.tick(0.4); rec.mark(BARGE_IN)
    clock.tick(0.25); rec.mark(BARGE_IN_STOP)
    [record] = rec.records()
    assert record.barge_in_latency == approx(0.25)


def test_mark_first_token_stamps_only_first():
    clock = FakeClock()
    rec = MetricsRecorder(clock=clock)
    rec.mark(ASR_FINAL)

    def tokens():
        clock.tick(0.5)
        yield "hello"
        clock.tick(0.5)
        yield " world"

    out = list(mark_first_token(tokens(), rec))
    assert out == ["hello", " world"]
    [record] = rec.records()
    assert record.final_to_first_token == approx(0.5)


def test_mark_first_token_passes_through_without_recorder():
    assert list(mark_first_token(iter(["a", "b"]), None)) == ["a", "b"]
