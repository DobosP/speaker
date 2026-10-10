"""Long-session watchdog behavior without scanning settled history each tick."""

from __future__ import annotations

import logging

from core.metrics import (
    ASR_FINAL,
    HANDLED_LOCAL,
    LLM_FIRST_TOKEN,
    TTS_FIRST_AUDIO,
    TTS_REQUESTED,
    MetricsRecorder,
)
from core.watchdog import StuckWatchdog


def setup():
    now = [0.0]
    recorder = MetricsRecorder(clock=lambda: now[0])
    watcher = StuckWatchdog(recorder, clock=lambda: now[0])
    watcher.ADAPTIVE_DEADLINES = False
    return now, recorder, watcher


def healthy(recorder):
    recorder.mark(ASR_FINAL)
    recorder.mark(LLM_FIRST_TOKEN)
    recorder.mark(TTS_FIRST_AUDIO)
    recorder.close_turn()


def test_settled_history_is_preserved_but_not_rescanned(monkeypatch):
    now, recorder, watcher = setup()
    for _ in range(10000):
        healthy(recorder)
    before = [record.as_dict() for record in recorder.records()]
    counts = []
    original = recorder.watchdog_snapshot

    def observe(*args):
        snapshot = original(*args)
        counts.append(len(snapshot.records))
        return snapshot

    monkeypatch.setattr(recorder, "watchdog_snapshot", observe)
    monkeypatch.setattr(
        recorder,
        "records",
        lambda: (_ for _ in ()).throw(AssertionError("full history copied")),
    )
    watcher.tick()
    watcher.tick()
    recorder.mark(ASR_FINAL)
    watcher.tick()
    watcher.tick()
    assert counts == [0, 0, 1, 1]
    assert len(watcher._pending_metric_tokens) == 1
    assert not watcher._warned
    assert [record.as_dict() for record in recorder._completed] == before
    now[0] = 11
    watcher.tick()
    watcher.tick()
    assert len(watcher._warned) == 1  # current turn still needs phase deduplication
    healthy(recorder)
    watcher.tick()
    watcher.tick()
    assert not watcher._warned
    assert not watcher._pending_metric_tokens


def test_banked_unresolved_turn_warns_later_once(caplog):
    now, recorder, watcher = setup()
    recorder.mark(ASR_FINAL)
    watcher.tick()
    healthy(recorder)  # banks old final without manufacturing a supersede mark
    with caplog.at_level(logging.WARNING, logger="speaker.watchdog"):
        now[0] = 9
        watcher.tick()
        assert not caplog.records
        now[0] = 11
        watcher.tick()
        now[0] = 12
        watcher.tick()
    assert len(caplog.records) == 1
    assert "llm stuck: turn 0" in caplog.text
    assert not watcher._warned
    assert not watcher._pending_metric_tokens


def test_retrospective_supersede_retires_a_pending_old_turn(caplog):
    now, recorder, watcher = setup()
    recorder.mark(ASR_FINAL)
    token = recorder.current_turn_token()
    watcher.tick()
    healthy(recorder)
    recorder.mark_arrival_superseded_turn(token)
    now[0] = 11
    with caplog.at_level(logging.WARNING, logger="speaker.watchdog"):
        watcher.tick()
        watcher.tick()
    assert not caplog.records
    assert not watcher._pending_metric_tokens


def test_current_turn_can_progress_to_tts_after_llm_warning(caplog):
    now, recorder, watcher = setup()
    recorder.mark(ASR_FINAL)
    with caplog.at_level(logging.WARNING, logger="speaker.watchdog"):
        now[0] = 11
        watcher.tick()
        watcher.tick()
        now[0] = 12
        recorder.mark(LLM_FIRST_TOKEN)
        watcher.tick()
        now[0] = 18
        watcher.tick()
        watcher.tick()
        recorder.mark(TTS_FIRST_AUDIO)
        watcher.tick()
    assert len(caplog.records) == 2
    assert "llm stuck" in caplog.records[0].message
    assert "tts stuck" in caplog.records[1].message
    assert not watcher._pending_metric_tokens


def test_two_pending_phases_on_closed_turn_keep_independent_deduplication(caplog):
    now, recorder, watcher = setup()
    recorder.mark(ASR_FINAL)
    recorder.mark(TTS_REQUESTED)
    with caplog.at_level(logging.WARNING, logger="speaker.watchdog"):
        now[0] = 6
        watcher.tick()  # TTS only reaches its deadline
        healthy(recorder)
        now[0] = 7
        watcher.tick()
        now[0] = 11
        watcher.tick()  # old LLM phase now reaches its deadline
        watcher.tick()
    assert len(caplog.records) == 2
    assert "tts stuck: turn 0" in caplog.records[0].message
    assert "llm stuck: turn 0" in caplog.records[1].message
    assert not watcher._warned
    assert not watcher._pending_metric_tokens


def test_reset_does_not_reuse_warning_identity_or_keep_old_pending_turns(caplog):
    now, recorder, watcher = setup()
    recorder.mark(ASR_FINAL)
    with caplog.at_level(logging.WARNING, logger="speaker.watchdog"):
        now[0] = 11
        watcher.tick()
        recorder.reset()
        recorder.mark(ASR_FINAL)
        now[0] = 22
        watcher.tick()
        watcher.tick()
    assert len(caplog.records) == 2
    assert all("llm stuck: turn 0" in row.message for row in caplog.records)
    assert len(watcher._warned) == 1


def test_snapshot_is_detached_and_keeps_original_indices():
    _, recorder, _ = setup()
    recorder.mark(ASR_FINAL)
    first_token = recorder.current_turn_token()
    recorder.close_turn()
    healthy(recorder)
    recorder.mark(ASR_FINAL)
    latest = recorder.current_turn_token()
    snapshot = recorder.watchdog_snapshot(latest, {first_token})
    assert [(i, row.turn_token) for i, row in snapshot.records] == [
        (0, first_token),
        (2, latest),
    ]
    snapshot.records[0][1].stamps[HANDLED_LOCAL] = 999
    assert HANDLED_LOCAL not in recorder.records()[0].stamps
    recorder.reset()
    empty = recorder.watchdog_snapshot(latest, {first_token})
    assert empty.records == ()
    assert empty.epoch == snapshot.epoch + 1
    assert empty.through_token == latest


def test_independent_watchdogs_do_not_consume_each_others_observations(caplog):
    now, recorder, first = setup()
    second = StuckWatchdog(recorder, clock=lambda: now[0])
    second.ADAPTIVE_DEADLINES = False
    recorder.mark(ASR_FINAL)
    now[0] = 11
    with caplog.at_level(logging.WARNING, logger="speaker.watchdog"):
        first.tick()
        second.tick()
        first.tick()
        second.tick()
    assert len(caplog.records) == 2


def test_closed_warning_deduplication_does_not_accumulate_over_session(caplog):
    now, recorder, watcher = setup()
    with caplog.at_level(logging.CRITICAL, logger="speaker.watchdog"):
        for _ in range(1000):
            recorder.mark(ASR_FINAL)
            now[0] += 11
            watcher.tick()
            recorder.close_turn()
            watcher.tick()
            assert not watcher._warned
            assert not watcher._pending_metric_tokens
    assert len(recorder.records()) == 1000
