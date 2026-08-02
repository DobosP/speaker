"""Engine-level tests for ASR final post-processing and stream creation.

No sherpa-onnx, no models, no audio: the recognizer / punctuation objects are
fakes assigned onto the engine, exercising the casing + punctuation + hotword
wiring directly (the threaded capture loop is out of scope).
"""
from __future__ import annotations

from threading import Event

import pytest

from core.engines._sherpa_streaming_decode import (
    SherpaStreamingDecodeSession,
    SherpaStreamingDecodeStream,
)
from core.engines._sherpa_streaming_decode_owner import (
    DecodeStreamRole,
    SherpaStreamingDecodeOwner,
)
from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine
from core.realtime_media_stage import CaptureScope


class _FakePunct:
    def __init__(self):
        self.seen = []

    def add_punctuation(self, text):
        self.seen.append(text)
        return text + "."


def test_postprocess_final_casing_only():
    eng = SherpaOnnxEngine(SherpaConfig(asr_restore_casing=True))
    # No punctuation model -> casing restoration only.
    assert eng._postprocess_final("HE MURMURED HIS MURDERING") == "He murmured his murdering"


def test_postprocess_final_casing_disabled_is_passthrough():
    eng = SherpaOnnxEngine(SherpaConfig(asr_restore_casing=False))
    assert eng._postprocess_final("HE MURMURED") == "HE MURMURED"


def test_postprocess_final_with_punctuation_then_casing():
    eng = SherpaOnnxEngine(SherpaConfig(asr_restore_casing=True))
    eng._punct = _FakePunct()
    # Punctuation runs first (sees raw text), then casing (force-lowercases the
    # all-caps source before re-capitalizing) -> readable sentence.
    out = eng._postprocess_final("WHAT TIME IS IT")
    assert eng._punct.seen == ["WHAT TIME IS IT"]
    assert out == "What time is it."


def test_postprocess_final_survives_punctuation_failure():
    class _Boom:
        def add_punctuation(self, text):
            raise RuntimeError("punct model died")

    eng = SherpaOnnxEngine(SherpaConfig(asr_restore_casing=True))
    eng._punct = _Boom()
    # Falls back to casing-only rather than losing the turn.
    assert eng._postprocess_final("SET A TIMER") == "Set a timer"


class _FakeRecognizer:
    def __init__(self):
        self.hotwords_arg = "UNSET"
        self.hotword_args = []
        self.plain_calls = 0

    def create_stream(self, hotwords=None):
        self.hotwords_arg = hotwords
        self.hotword_args.append(hotwords)
        return object()


class _NoHotwordRecognizer:
    """A build whose create_stream() rejects the hotwords kwarg."""

    def __init__(self):
        self.plain_calls = 0

    def create_stream(self):
        self.plain_calls += 1
        return object()


def test_new_asr_stream_passes_hotwords_with_beam_search():
    eng = SherpaOnnxEngine(
        SherpaConfig(asr_decoding_method="modified_beam_search", asr_hotwords="FLURRY\nPARIS")
    )
    rec = _FakeRecognizer()
    eng._recognizer = rec
    eng._hotwords = ["FLURRY", "PARIS"]
    eng._new_asr_stream()
    assert rec.hotwords_arg == "FLURRY\nPARIS"


def test_new_asr_stream_rejects_active_hotwords_under_greedy():
    eng = SherpaOnnxEngine(
        SherpaConfig(asr_decoding_method="greedy_search", asr_hotwords="Flurry")
    )
    rec = _FakeRecognizer()
    eng._recognizer = rec
    eng._hotwords = ["Flurry"]
    with pytest.raises(ValueError, match="modified_beam_search"):
        eng._new_asr_stream()
    assert rec.hotwords_arg == "UNSET"


def test_new_asr_stream_fails_closed_when_build_rejects_hotwords():
    eng = SherpaOnnxEngine(
        SherpaConfig(asr_decoding_method="modified_beam_search", asr_hotwords="Flurry")
    )
    rec = _NoHotwordRecognizer()
    eng._recognizer = rec
    eng._hotwords = ["Flurry"]
    with pytest.raises(RuntimeError, match="per-stream hotword"):
        eng._new_asr_stream()
    assert rec.plain_calls == 0


def test_decode_session_preserves_hotword_stream_creation():
    eng = SherpaOnnxEngine(
        SherpaConfig(
            asr_decoding_method="modified_beam_search",
            asr_hotwords="FLURRY\nPARIS",
        )
    )
    rec = _FakeRecognizer()
    session = SherpaStreamingDecodeSession(rec)
    session.bind_current_thread()
    eng._hotwords = ["FLURRY", "PARIS"]

    stream = eng._new_asr_stream(session)

    assert type(stream) is SherpaStreamingDecodeStream
    assert rec.hotwords_arg == "FLURRY\nPARIS"
    session.close()


def test_decode_session_fails_closed_without_hotword_stream_support():
    eng = SherpaOnnxEngine(
        SherpaConfig(
            asr_decoding_method="modified_beam_search",
            asr_hotwords="Flurry",
        )
    )
    rec = _NoHotwordRecognizer()
    session = SherpaStreamingDecodeSession(rec)
    session.bind_current_thread()
    eng._hotwords = ["Flurry"]

    with pytest.raises(RuntimeError, match="per-stream hotword"):
        eng._new_asr_stream(session)

    assert rec.plain_calls == 0
    assert session.snapshot().native_calls_failed == 1
    session.close()


def _run_owner_stream_creation(eng, rec):
    session = SherpaStreamingDecodeSession(rec)
    release = Event()
    streams = []

    def process(owner):
        streams.append(
            eng._new_asr_stream(owner, role=DecodeStreamRole.PRIMARY)
        )
        streams.append(
            eng._new_asr_stream(owner, role=DecodeStreamRole.WORD_CUT)
        )
        owner.publish_processor_ready()
        assert release.wait(2.0)

    owner = SherpaStreamingDecodeOwner(
        session,
        run_id=1,
        capture_scope=CaptureScope(capture_epoch=0, capture_generation=1),
        processor=process,
    )
    owner.start(timeout=1.0)
    release.set()
    assert owner.close(timeout=2.0)
    return owner.snapshot(), session.snapshot(), streams


def test_dedicated_owner_preserves_hotwords_for_both_stream_roles():
    eng = SherpaOnnxEngine(
        SherpaConfig(
            asr_decoding_method="modified_beam_search",
            asr_hotwords="FLURRY\nPARIS",
        )
    )
    rec = _FakeRecognizer()
    eng._hotwords = ["FLURRY", "PARIS"]

    owner, session, streams = _run_owner_stream_creation(eng, rec)

    assert all(type(stream) is SherpaStreamingDecodeStream for stream in streams)
    assert rec.hotword_args == ["FLURRY\nPARIS", "FLURRY\nPARIS"]
    assert owner.streams_created == 2
    assert owner.streams_retired == 2
    assert session.native_calls_failed == 0


def test_dedicated_owner_fails_startup_without_hotword_stream_support():
    eng = SherpaOnnxEngine(
        SherpaConfig(
            asr_decoding_method="modified_beam_search",
            asr_hotwords="Flurry",
        )
    )
    rec = _NoHotwordRecognizer()
    eng._hotwords = ["Flurry"]

    session = SherpaStreamingDecodeSession(rec)

    def process(owner):
        eng._new_asr_stream(owner, role=DecodeStreamRole.PRIMARY)

    owner = SherpaStreamingDecodeOwner(
        session,
        run_id=1,
        capture_scope=CaptureScope(capture_epoch=0, capture_generation=1),
        processor=process,
    )

    with pytest.raises(RuntimeError, match="startup failed: RuntimeError"):
        owner.start(timeout=1.0)
    assert owner.close(timeout=2.0)
    assert rec.plain_calls == 0
    snapshot = owner.snapshot()
    assert snapshot.streams_created == 0
    assert snapshot.streams_retired == 0
    assert session.snapshot().native_calls_failed == 1
