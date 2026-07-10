from __future__ import annotations

from pytest import approx

from core.metrics import (
    ASR_FINAL,
    BARGE_IN,
    BARGE_IN_STOP,
    LLM_FIRST_TOKEN,
    MERGED,
    SPEECH_END,
    SUPERSEDED,
    TTS_FIRST_AUDIO,
    MetricsRecorder,
    TurnRecord,
    mark_first_token,
)


def test_mark_superseded_turn_stamps_the_just_banked_turn():
    # Mirrors newest-input-wins: turn 1's ASR_FINAL banks turn 0, then turn 0
    # (now _completed[-1]) is marked superseded -- NOT the open turn 1 (rc-5).
    rec = MetricsRecorder(clock=lambda: 1.0)
    rec.mark(ASR_FINAL)          # turn 0
    rec.mark(ASR_FINAL)          # turn 1 -> banks turn 0
    rec.mark_superseded_turn()
    records = rec.records()
    assert SUPERSEDED in records[0].stamps      # the preempted turn
    assert SUPERSEDED not in records[1].stamps  # the new (open) turn untouched


def test_mark_superseded_turn_is_a_noop_with_nothing_banked():
    rec = MetricsRecorder(clock=lambda: 1.0)
    rec.mark_superseded_turn()   # no completed turn yet -> no crash, no stamp
    rec.mark(ASR_FINAL)
    assert SUPERSEDED not in rec.records()[0].stamps


def test_mark_merged_turn_stamps_the_just_banked_turn():
    rec = MetricsRecorder(clock=lambda: 1.0)
    rec.mark(ASR_FINAL)       # original turn, cancelled by continuation merge
    rec.mark(ASR_FINAL)       # add-on turn banks the original
    rec.mark_merged_turn()
    records = rec.records()
    assert MERGED in records[0].stamps
    assert SUPERSEDED in records[0].stamps
    assert MERGED not in records[1].stamps


def test_continuation_metrics_target_original_token_across_intermediate_turns():
    rec = MetricsRecorder(clock=lambda: 1.0)
    rec.mark(ASR_FINAL)
    victim_token = rec.current_turn_token()
    assert victim_token is not None
    rec.mark(SPEECH_END)

    # Later speech starts bank/open intermediate records before the delayed
    # replacement task admits. Token targeting must still mark the true victim.
    rec.mark(SPEECH_END)
    rec.mark_arrival_superseded_turn(victim_token)
    rec.mark(SPEECH_END)
    rec.mark_merged_turn(victim_token)

    records = rec.records()
    victim = next(record for record in records if record.turn_token == victim_token)
    assert MERGED in victim.stamps
    assert SUPERSEDED in victim.stamps
    assert all(
        MERGED not in record.stamps
        for record in records
        if record.turn_token != victim_token
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


def test_mark_first_token_close_propagates_to_underlying_iterator():
    class CloseAwareIterator:
        def __init__(self):
            self.closed = False

        def __iter__(self):
            return self

        def __next__(self):
            return "token"

        def close(self):
            self.closed = True

    underlying = CloseAwareIterator()
    wrapped = mark_first_token(underlying, None)

    assert next(wrapped) == "token"
    wrapped.close()

    assert underlying.closed is True


def test_task_scoped_first_token_cannot_stamp_a_replacement_turn():
    rec = MetricsRecorder()
    rec.mark(ASR_FINAL)
    old_turn = rec.current_turn_token()
    rec.mark(ASR_FINAL)  # bank old + open replacement

    rec.mark(LLM_FIRST_TOKEN, turn_token=old_turn)

    assert LLM_FIRST_TOKEN not in rec.records()[-1].stamps
    assert rec.recent_ttft_ms() is None


def test_reset_does_not_reuse_turn_token_held_by_abandoned_provider():
    rec = MetricsRecorder()
    rec.mark(ASR_FINAL)
    stale_token = rec.current_turn_token()
    rec.reset()
    rec.mark(ASR_FINAL)

    assert rec.current_turn_token() != stale_token
    rec.mark(LLM_FIRST_TOKEN, turn_token=stale_token)
    assert LLM_FIRST_TOKEN not in rec.records()[-1].stamps


# --- control-plane-3: the TTS first-audio EWMA the watchdog scales off ---------


def test_recent_tts_ms_folds_first_token_to_audio():
    clock = FakeClock()
    rec = MetricsRecorder(clock=clock)
    assert rec.recent_tts_ms() is None            # cold: no sample yet
    rec.mark(ASR_FINAL)
    rec.mark(LLM_FIRST_TOKEN)
    clock.tick(0.4)                                # 400ms of TTS synth for sentence 1
    rec.mark(TTS_FIRST_AUDIO)
    assert rec.recent_tts_ms() == approx(400.0, abs=1.0)


def test_recent_tts_ms_none_without_a_first_token_anchor():
    clock = FakeClock()
    rec = MetricsRecorder(clock=clock)
    rec.mark(ASR_FINAL)
    clock.tick(0.4)
    rec.mark(TTS_FIRST_AUDIO)   # no LLM_FIRST_TOKEN -> nothing to measure against
    assert rec.recent_tts_ms() is None


def test_reset_clears_the_tts_ewma():
    clock = FakeClock()
    rec = MetricsRecorder(clock=clock)
    rec.mark(ASR_FINAL); rec.mark(LLM_FIRST_TOKEN); clock.tick(0.3); rec.mark(TTS_FIRST_AUDIO)
    assert rec.recent_tts_ms() is not None
    rec.reset()
    assert rec.recent_tts_ms() is None
