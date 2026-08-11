from __future__ import annotations

from dataclasses import replace
import gc
import sys
import threading
import time
import types
import weakref

import numpy as np
import pytest

import core.engines.sherpa as sherpa_module
from core.engine import EngineCallbacks
from core.engines._acoustic_turn import AcousticTurnTracker
from core.engines._kws_speaker_inference_owner import (
    KwsSpeakerInferenceOutcome,
    KwsSpeakerInferenceResult,
)
from core.engines.sherpa import (
    SherpaConfig,
    SherpaOnnxEngine,
    _CaptureBlockRejected,
    _KwsControlAuthority,
    _KwsFeedRoute,
    _MAX_KWS_PCM_LEDGER_CHUNKS,
    _SpeakerPcmStatus,
)
from core.engines.speaker_gate import (
    SpeakerGate,
    SpeakerSimilarityBatchReceipt,
    runtime_speaker_inference_permit,
)
from core.kws_contract import KWS_STOP_PHRASES, KwsPhraseBinding
from core.media_session import CaptureBlockContext, CapturedBlock


class _Binder:
    def __init__(self):
        self.capture_bound = False
        self.playback_bound = False
        self.contract = types.SimpleNamespace(provenance="autotest-virtual-delay:test")
        self.fail_duplex = False

    @property
    def fully_bound(self):
        return self.capture_bound and self.playback_bound

    def bind_capture(self, *, timeout_sec):
        assert timeout_sec > 0
        self.capture_bound = True

    def bind_playback(self, *, timeout_sec):
        assert timeout_sec > 0
        self.playback_bound = True

    def verify(self, *, require_playback):
        if require_playback and (self.fail_duplex or not self.fully_bound):
            return False, "playback wrong"
        return self.capture_bound, "exact virtual route"


def _engine(binder):
    return SherpaOnnxEngine(
        SherpaConfig(
            barge_in_enabled=True,
            barge_word_cut_enabled=True,
            aec_enabled=False,
        ),
        virtual_audio_binder=binder,
    )


def test_virtual_word_cut_authority_is_two_phase():
    binder = _Binder()
    engine = _engine(binder)

    engine._bind_virtual_capture_route()
    assert binder.capture_bound is True
    assert engine._word_cut_route_verified is False
    assert engine._os_echo_route_verified is False

    engine._prepare_virtual_playback_route()
    assert engine._word_cut_route_verified is False
    assert engine._os_echo_route_verified is False
    # Capture may have consumed the reply reset while playback was still
    # unbound. Successful authorization must re-arm this same reply.
    engine._barge_sustain_reset_pending = False
    engine._authorize_virtual_playback_route()
    assert binder.fully_bound is True
    assert engine._word_cut_route_verified is True
    assert engine._os_echo_route_verified is True
    assert engine._barge_sustain_reset_pending is True


def test_loss_started_before_duplex_grant_cannot_reauthorize_route():
    binder = _Binder()
    binder.capture_bound = True
    binder.playback_bound = True
    engine = _engine(binder)
    engine._virtual_route_failure_in_progress = True

    with pytest.raises(RuntimeError, match="revocation already started"):
        engine._authorize_virtual_playback_route()

    assert engine._word_cut_route_verified is False
    assert engine._os_echo_route_verified is False
    assert engine._barge_sustain_reset_pending is False


def test_rearmed_duplex_grant_initializes_word_cut_in_same_reply():
    class _Stream:
        def __init__(self):
            self.blocks = []

        def accept_waveform(self, _sample_rate, samples):
            self.blocks.append(np.asarray(samples).copy())

    class _Recognizer:
        def __init__(self):
            self.streams = []

        def create_stream(self, **_kwargs):
            stream = _Stream()
            self.streams.append(stream)
            return stream

        def is_ready(self, _stream):
            return False

        def get_result(self, _stream):
            return ""

        def reset(self, _stream):
            pass

        def is_endpoint(self, _stream):
            return False

    class _Vad:
        def accept_waveform(self, _samples):
            pass

        def is_speech_detected(self):
            return True

    class _OneBlockInput:
        generation = 0
        actual_device = None

        def __init__(self, engine):
            self.engine = engine

        def read(self, frames):
            self.engine._running.clear()
            return np.zeros(frames, dtype="float32"), False

    binder = _Binder()
    engine = _engine(binder)
    recognizer = _Recognizer()
    engine._recognizer = recognizer
    engine._vad = _Vad()
    engine._capture_sr = engine.config.sample_rate
    engine._stream_in = _OneBlockInput(engine)
    # This fixture bypasses the normal capture-domain resolution that binds the
    # live source identity before virtual-route authorization.
    engine._capture_authority_source_generation = 0
    engine._capture_authority_source_device = None
    engine._speaking.set()
    # Model the race: the reply reset was already consumed with capture-only
    # authority, so no detector stream exists yet.
    engine._barge_sustain_reset_pending = False
    engine._bind_virtual_capture_route()
    engine._prepare_virtual_playback_route()
    engine._authorize_virtual_playback_route()

    engine._running.set()
    engine._capture_loop()

    assert len(recognizer.streams) == 2
    assert len(recognizer.streams[0].blocks) == 0
    assert len(recognizer.streams[1].blocks) == 1


def test_private_kws_rechecks_route_at_callback_seam():
    class _KwsStream:
        def accept_waveform(self, _sample_rate, _samples):
            pass

    class _Kws:
        def __init__(self, engine):
            self.engine = engine
            self.resets = 0

        def is_ready(self, _stream):
            return False

        def get_result(self, _stream):
            # Revocation begins after the outer eligibility read and decode but
            # before the callback seam protected by the route lock.
            self.engine._virtual_route_failure_in_progress = True
            self.engine._speaking.clear()
            return "stop"

        def reset_stream(self, _stream):
            self.resets += 1

    binder = _Binder()
    engine = _engine(binder)
    commands = []
    engine._kws_stream = _KwsStream()
    engine._kws = _Kws(engine)
    engine._cb = EngineCallbacks(on_command=commands.append)
    engine._os_echo_route_verified = True

    consumed = engine._poll_keywords(
        np.zeros(1600, dtype="float32"),
        guard_private_route=True,
        require_os_echo_route=True,
    )

    assert engine._kws.resets == 1
    assert commands == []
    assert consumed is False


class _FixedKws:
    """Minimal spotter that reports one fixed keyword hit per poll."""

    def __init__(self, keyword):
        self.keyword = keyword
        self.resets = 0

    def is_ready(self, _stream):
        return False

    def get_result(self, _stream):
        return self.keyword

    def reset_stream(self, _stream):
        self.resets += 1

    def create_stream(self):
        return _NullKwsStream()


_PLAYBACK_KWS_TOKENS = tuple(phrase.tokens for phrase in KWS_STOP_PHRASES)
_PLAYBACK_KWS_BINDINGS = tuple(
    KwsPhraseBinding(
        tokens=tokens,
        word_token_counts=phrase.word_token_counts,
        surface=phrase.surface,
        result_label=phrase.result_label,
    )
    for tokens, phrase in zip(
        _PLAYBACK_KWS_TOKENS,
        KWS_STOP_PHRASES,
        strict=True,
    )
)


class _TokenKws(_FixedKws):
    def __init__(
        self,
        keyword,
        tokens,
        *,
        timestamps=(),
        before_result=None,
        before_tokens=None,
        before_timestamps=None,
    ):
        super().__init__(keyword)
        self._tokens = tokens
        self._timestamps = timestamps
        self._before_result = before_result
        self._before_tokens = before_tokens
        self._before_timestamps = before_timestamps
        self.token_calls = 0
        self.timestamp_calls = 0

    def get_result(self, stream):
        if self._before_result is not None:
            self._before_result()
        return super().get_result(stream)

    def tokens(self, _stream):
        self.token_calls += 1
        if self._before_tokens is not None:
            self._before_tokens()
        return self._tokens

    def timestamps(self, _stream):
        self.timestamp_calls += 1
        if self._before_timestamps is not None:
            self._before_timestamps()
        return self._timestamps


class _NullKwsStream:
    def __init__(self):
        self.sample_rates: list[int] = []
        self.blocks: list[np.ndarray] = []

    def accept_waveform(self, _sample_rate, _samples):
        self.sample_rates.append(_sample_rate)
        self.blocks.append(np.asarray(_samples, dtype="float32").copy())


class _BoundedKwsStream:
    def __init__(self, spotter: "_BoundedKws", sequence: int) -> None:
        self.spotter = spotter
        self.sequence = sequence
        self.decode_calls = 0

    def accept_waveform(self, _sample_rate, samples):
        self.spotter.accept_calls += 1
        self.spotter.accepted.append(np.asarray(samples).copy())
        if self.spotter.failure_phase == "accept":
            raise RuntimeError("injected KWS accept failure")


class _BoundedKws:
    """Keyword fake with exact native-call accounting and stream recreation."""

    def __init__(
        self,
        *,
        ready_decodes: int | None,
        keyword: str = "",
        failure_phase: str | None = None,
    ) -> None:
        self.ready_decodes = ready_decodes
        self.keyword = keyword
        self.failure_phase = failure_phase
        self.create_calls = 0
        self.accept_calls = 0
        self.ready_calls = 0
        self.decode_calls = 0
        self.result_calls = 0
        self.reset_calls = 0
        self.accepted: list[np.ndarray] = []
        self.streams: list[_BoundedKwsStream] = []

    def create_stream(self):
        self.create_calls += 1
        stream = _BoundedKwsStream(self, self.create_calls)
        self.streams.append(stream)
        return stream

    def is_ready(self, stream):
        self.ready_calls += 1
        if self.failure_phase == "is_ready":
            raise RuntimeError("injected KWS readiness failure")
        return self.ready_decodes is None or stream.decode_calls < self.ready_decodes

    def decode_stream(self, stream):
        self.decode_calls += 1
        stream.decode_calls += 1
        if self.failure_phase == "decode":
            raise RuntimeError("injected KWS decode failure")

    def get_result(self, _stream):
        self.result_calls += 1
        if self.failure_phase == "result":
            raise RuntimeError("injected KWS result failure")
        return self.keyword

    def reset_stream(self, _stream):
        self.reset_calls += 1


def _kws_engine(keyword):
    engine = SherpaOnnxEngine(SherpaConfig())
    engine._kws = _FixedKws(keyword)
    engine._kws_stream = _NullKwsStream()
    commands: list[str] = []
    engine._cb = EngineCallbacks(on_command=commands.append)
    return engine, commands


def _authority_kws_engine(
    keyword: str,
    *,
    phrase_index: int = 0,
    speaking: bool = True,
    os_echo_route_verified: bool = False,
    before_result=None,
    before_tokens=None,
    timestamps=(),
    before_timestamps=None,
):
    engine = SherpaOnnxEngine(SherpaConfig())
    spotter = _TokenKws(
        keyword,
        _PLAYBACK_KWS_TOKENS[phrase_index],
        timestamps=timestamps,
        before_result=before_result,
        before_tokens=before_tokens,
        before_timestamps=before_timestamps,
    )
    engine._kws = spotter
    engine._kws_stream = _NullKwsStream()
    engine._kws_phrase_bindings = _PLAYBACK_KWS_BINDINGS
    engine._stream_in = types.SimpleNamespace(
        generation=3,
        actual_device="mic-a",
    )
    engine._capture_epoch = 2
    engine._capture_callback_context.epoch = 2
    engine._capture_callback_context.capture_generation = 5
    engine._capture_media_session = types.SimpleNamespace(generation=5)
    engine._capture_authority_source_generation = 3
    engine._capture_authority_source_device = "mic-a"
    engine._os_echo_route_verified = os_echo_route_verified
    engine._speak_gen = 7
    engine._playback_generation = 11
    if speaking:
        engine._speaking.set()
    else:
        engine._speaking.clear()
    route_owner = None
    if not speaking:
        route = _KwsFeedRoute.IDLE
    elif os_echo_route_verified:
        route = _KwsFeedRoute.VERIFIED_OS_ECHO
    else:
        route = _KwsFeedRoute.IN_APP_AEC
        route_owner = object()
        engine._aec = route_owner
    authority = _KwsControlAuthority(
        capture_epoch=2,
        capture_generation=5,
        capture_sequence=1,
        source_generation=3,
        source_device="mic-a",
        source_sample_rate_hz=16_000,
        source_sample_count=1600,
        source_sample_start=0,
        source_sample_end=1600,
        source_domain=engine._capture_resolution,
        kws_sample_rate_hz=16_000,
        authority_source_generation=3,
        authority_source_device="mic-a",
        os_echo_route_verified=os_echo_route_verified,
        route=route,
        route_owner=route_owner,
        speaker_authority_available=False,
        speaker_authority=None,
        speaking=speaking,
        speak_generation=7,
        playback_generation=11,
    )
    return engine, spotter, authority


def _bind_kws_test_pcm(
    engine: SherpaOnnxEngine,
    authority: _KwsControlAuthority,
    pcm,
) -> _KwsControlAuthority:
    if authority.speaker_authority is None:
        gate = SpeakerGate(embed_fn=lambda _samples, _sample_rate: [1.0, 0.0])
        gate.enroll_embedding([1.0, 0.0])
        engine._replace_speaker_gate(gate)
        assert engine._warm_speaker_gate()
        speaker_authority = engine._snapshot_kws_speaker_authority()
        assert speaker_authority is not None
        authority = replace(
            authority,
            speaker_authority_available=True,
            speaker_authority=speaker_authority,
        )
    owned = np.array(pcm, dtype="float32", order="C", copy=True).reshape(-1)
    authority = replace(
        authority,
        source_sample_count=owned.size,
        source_sample_end=authority.source_sample_start + owned.size,
    )
    assert engine._prepare_kws_submission(authority)
    engine._append_kws_submitted_pcm(owned, authority)
    return authority


def _array_storage_root(array: np.ndarray) -> np.ndarray:
    root = array
    while isinstance(root.base, np.ndarray):
        root = root.base
    return root


def _speaker_available_idle_kws_case(
    *,
    block_sec: float | None = None,
    window_sec: float | None = None,
):
    engine, spotter, authority = _authority_kws_engine("", speaking=False)
    if block_sec is not None:
        engine.config.block_sec = block_sec
    if window_sec is not None:
        engine.config.barge_word_cut_speaker_window_sec = window_sec
    gate = SpeakerGate(embed_fn=lambda _samples, _sample_rate: [1.0, 0.0])
    gate.enroll_embedding([1.0, 0.0])
    engine._replace_speaker_gate(gate)
    assert engine._warm_speaker_gate()
    speaker_authority = engine._snapshot_kws_speaker_authority()
    assert speaker_authority is not None
    return (
        engine,
        spotter,
        replace(
            authority,
            source_sample_count=1,
            source_sample_end=1,
            speaker_authority_available=True,
            speaker_authority=speaker_authority,
        ),
    )


def test_kws_word_regions_share_one_owned_read_only_phrase_copy() -> None:
    engine, _spotter, authority = _authority_kws_engine(
        "stop",
        phrase_index=1,
    )
    pcm = np.arange(8_000, dtype="float32")
    _bind_kws_test_pcm(engine, authority, pcm)
    retained = tuple(chunk.pcm for chunk in engine._kws_pcm_ledger.chunks)
    timestamps = [0.00, 0.04, 0.08, 0.12, 0.20, 0.24, 0.28, 0.32, 0.36]

    regions = engine._kws_word_pcm_regions(
        _PLAYBACK_KWS_BINDINGS[1],
        timestamps,
    )

    assert regions is not None
    assert len(regions) == 2
    np.testing.assert_array_equal(regions[0], pcm[:3_200])
    np.testing.assert_array_equal(regions[1], pcm[3_200:6_400])
    phrase_root = _array_storage_root(regions[0])
    assert phrase_root is _array_storage_root(regions[1])
    assert phrase_root.base is None
    assert phrase_root.flags.owndata
    assert phrase_root.size == 6_400
    np.testing.assert_array_equal(phrase_root, pcm[:6_400])
    assert not phrase_root.flags.writeable
    assert all(not region.flags.writeable for region in regions)
    assert not np.shares_memory(regions[0], regions[1])
    assert not np.shares_memory(phrase_root, pcm)
    assert all(not np.shares_memory(phrase_root, chunk) for chunk in retained)


def test_kws_interval_rejects_forged_geometry_before_output_allocation(
    monkeypatch,
) -> None:
    engine, _spotter, authority = _authority_kws_engine("stop")
    _bind_kws_test_pcm(engine, authority, np.arange(3_200, dtype="float32"))
    ledger = engine._kws_pcm_ledger
    original_chunk = ledger.chunks[0]
    allocations = 0

    def unexpected_empty(*_args, **_kwargs):
        nonlocal allocations
        allocations += 1
        raise AssertionError("invalid ledger geometry must not allocate output")

    monkeypatch.setattr(np, "empty", unexpected_empty)
    ledger.chunks[0] = replace(original_chunk, end=original_chunk.end - 1)

    assert engine._copy_kws_pcm_interval(0, 640) is None
    assert allocations == 0

    ledger.chunks[0] = original_chunk
    speaker_authority = ledger.scope.speaker_authority
    short_policy = replace(speaker_authority.policy, window_sec=0.10)
    ledger.scope = replace(
        ledger.scope,
        speaker_authority=replace(speaker_authority, policy=short_policy),
    )

    assert engine._copy_kws_pcm_interval(0, 3_200) is None
    assert allocations == 0


def test_kws_word_regions_accept_float32_timestamp_rounding_within_half_sample():
    engine, _spotter, authority = _authority_kws_engine("stop")
    pcm = np.arange(30_000, dtype="float32")
    _bind_kws_test_pcm(engine, authority, pcm)
    timestamps = [float(np.float32(value)) for value in (1.44, 1.48, 1.52, 1.56)]

    regions = engine._kws_word_pcm_regions(
        _PLAYBACK_KWS_BINDINGS[0],
        timestamps,
    )

    assert regions is not None
    assert len(regions) == 1
    np.testing.assert_array_equal(regions[0], pcm[23_040:25_600])


@pytest.mark.parametrize("offset_samples", [0.51, 1.0])
def test_kws_word_regions_reject_more_than_half_sample_off_grid(
    offset_samples,
) -> None:
    engine, _spotter, authority = _authority_kws_engine("stop")
    _bind_kws_test_pcm(engine, authority, np.arange(3_200, dtype="float32"))
    timestamps = [0.00, 0.04, 0.08, 0.12 + offset_samples / 16_000]

    assert (
        engine._kws_word_pcm_regions(
            _PLAYBACK_KWS_BINDINGS[0],
            timestamps,
        )
        is None
    )


@pytest.mark.parametrize(
    "timestamps",
    [
        (0.00, 0.04, 0.08, 0.12),
        [0.00, 0.04, 0.08],
        [0, 0.04, 0.08, 0.12],
        [np.float64(0.00), 0.04, 0.08, 0.12],
        [float("nan"), 0.04, 0.08, 0.12],
        [float("inf"), 0.04, 0.08, 0.12],
        [0.00, 0.04, 0.08, 1e308],
        [-0.04, 0.00, 0.04, 0.08],
        [0.00, 0.04, 0.04, 0.08],
        [0.00, 0.08, 0.04, 0.12],
        [0.00, 0.04, 0.08, 0.121],
        [0.20, 0.24, 0.28, 0.32],
    ],
    ids=(
        "not-list",
        "wrong-count",
        "integer-timestamp",
        "numpy-float",
        "nan",
        "infinity",
        "finite-overflow",
        "negative",
        "duplicate",
        "decreasing",
        "off-grid",
        "outside-ledger",
    ),
)
def test_kws_word_regions_reject_malformed_or_unretained_timestamps(
    timestamps,
) -> None:
    engine, _spotter, authority = _authority_kws_engine("stop")
    _bind_kws_test_pcm(engine, authority, np.arange(3_200, dtype="float32"))

    assert (
        engine._kws_word_pcm_regions(
            _PLAYBACK_KWS_BINDINGS[0],
            timestamps,
        )
        is None
    )


def test_kws_word_regions_reject_token_frames_trimmed_from_bounded_ledger() -> None:
    engine, _spotter, authority = _authority_kws_engine("stop")
    _bind_kws_test_pcm(engine, authority, np.ones(40_000, dtype="float32"))

    assert engine._kws_pcm_ledger.retained_start == 8_000
    assert (
        engine._kws_word_pcm_regions(
            _PLAYBACK_KWS_BINDINGS[0],
            [0.00, 0.04, 0.08, 0.12],
        )
        is None
    )


@pytest.mark.parametrize("cold_gate", [False, True], ids=("unavailable", "cold"))
def test_kws_unavailable_scope_keeps_clock_without_pcm_or_timestamps(
    cold_gate,
) -> None:
    engine, spotter, authority = _authority_kws_engine(
        "",
        timestamps=[0.00, 0.04, 0.08, 0.12],
    )
    if cold_gate:
        gate = SpeakerGate(embed_fn=lambda _samples, _sample_rate: [1.0, 0.0])
        gate.enroll_embedding([1.0, 0.0])
        engine._replace_speaker_gate(gate)
        assert not engine._speaker_gate_warmed
    stop_results = []
    engine._cb = EngineCallbacks(on_control_stop_result=stop_results.append)

    def unexpected_score(*_args, **_kwargs):
        raise AssertionError("unavailable/cold KWS must not score speaker PCM")

    engine._score_explicit_speaker_pcm_batch = unexpected_score

    assert not engine._poll_keywords(
        np.full(1_600, 0.25, dtype="float32"),
        control_authority=authority,
    )

    ledger = engine._kws_pcm_ledger
    assert ledger.scope is authority
    assert ledger.submitted_end == 1_600
    assert ledger.retained_start == 1_600
    assert ledger.last_capture_sequence == 1
    assert ledger.last_source_sample_end == 1_600
    assert tuple(ledger.chunks) == ()

    engine._now_playing = "I will stop now."
    spotter.keyword = "stop"
    second_authority = replace(
        authority,
        capture_sequence=2,
        source_sample_start=1_600,
        source_sample_end=3_200,
    )
    assert not engine._poll_keywords(
        np.full(1_600, 0.25, dtype="float32"),
        control_authority=second_authority,
    )

    assert stop_results == []
    assert spotter.timestamp_calls == 0
    assert engine._kws_own_echo_suppressions == 1


def test_kws_speaker_availability_transition_starts_fresh_retained_scope() -> None:
    engine, spotter, authority = _authority_kws_engine("")
    engine._cb = EngineCallbacks(on_control_stop_result=lambda _result: None)
    unavailable_stream = engine._kws_stream
    unavailable_pcm = np.full(1_600, 0.25, dtype="float32")

    assert not engine._poll_keywords(
        unavailable_pcm,
        control_authority=authority,
    )
    assert len(unavailable_stream.blocks) == 1
    assert tuple(engine._kws_pcm_ledger.chunks) == ()
    assert engine._kws_pcm_ledger.submitted_end == 1_600

    gate = SpeakerGate(embed_fn=lambda _samples, _sample_rate: [1.0, 0.0])
    gate.enroll_embedding([1.0, 0.0])
    engine._replace_speaker_gate(gate)
    assert engine._warm_speaker_gate()
    speaker_authority = engine._snapshot_kws_speaker_authority()
    assert speaker_authority is not None
    available_authority = replace(
        authority,
        capture_sequence=2,
        source_sample_start=1_600,
        source_sample_end=3_200,
        speaker_authority_available=True,
        speaker_authority=speaker_authority,
    )
    available_pcm = np.full(1_600, 0.75, dtype="float32")

    assert not engine._poll_keywords(
        available_pcm,
        control_authority=available_authority,
    )

    available_stream = engine._kws_stream
    assert available_stream is not unavailable_stream
    assert spotter.resets == 1
    assert len(available_stream.blocks) == 1
    np.testing.assert_array_equal(available_stream.blocks[0], available_pcm)
    ledger = engine._kws_pcm_ledger
    assert ledger.scope is available_authority
    assert ledger.submitted_end == 1_600
    assert ledger.retained_start == 0
    assert len(ledger.chunks) == 1
    np.testing.assert_array_equal(ledger.chunks[0].pcm, available_pcm)


def test_kws_allow_publish_false_skips_prefix_then_rotates_for_fresh_suffix() -> None:
    engine, spotter, authority = _authority_kws_engine("")
    gate = SpeakerGate(embed_fn=lambda _samples, _sample_rate: [1.0, 0.0])
    gate.enroll_embedding([1.0, 0.0])
    engine._replace_speaker_gate(gate)
    assert engine._warm_speaker_gate()
    speaker_authority = engine._snapshot_kws_speaker_authority()
    assert speaker_authority is not None
    authority = replace(
        authority,
        speaker_authority_available=True,
        speaker_authority=speaker_authority,
    )
    engine._cb = EngineCallbacks(on_control_stop_result=lambda _result: None)
    old_stream = engine._kws_stream
    prefix = np.full(1_600, 0.25, dtype="float32")
    suffix = np.full(1_600, 0.75, dtype="float32")

    assert not engine._poll_keywords(
        prefix,
        allow_publish=False,
        control_authority=authority,
    )
    assert old_stream.blocks == []
    assert spotter.resets == 0
    assert engine._kws_continuity_dirty
    assert engine._kws_pcm_ledger.scope is None

    second_authority = replace(
        authority,
        capture_sequence=2,
        source_sample_start=1_600,
        source_sample_end=3_200,
    )
    assert not engine._poll_keywords(
        suffix,
        control_authority=second_authority,
    )

    fresh_stream = engine._kws_stream
    assert fresh_stream is not old_stream
    assert spotter.resets == 1
    assert old_stream.blocks == []
    assert len(fresh_stream.blocks) == 1
    np.testing.assert_array_equal(fresh_stream.blocks[0], suffix)
    ledger = engine._kws_pcm_ledger
    assert ledger.scope is second_authority
    assert ledger.submitted_end == 1_600
    assert ledger.retained_start == 0
    assert len(ledger.chunks) == 1
    np.testing.assert_array_equal(ledger.chunks[0].pcm, suffix)


@pytest.mark.parametrize(
    "discontinuity",
    [
        "capture-epoch",
        "capture-generation",
        "source-generation",
        "source-device",
        "source-rate",
        "source-domain",
        "kws-rate",
        "authority-generation",
        "authority-device",
        "os-route",
        "feed-route",
        "speaker-authority",
        "speaking",
        "speak-generation",
        "playback-generation",
        "capture-sequence-gap",
        "source-sample-gap",
    ],
)
def test_kws_every_scope_or_adjacency_change_rotates_native_and_ledger(
    discontinuity,
) -> None:
    engine, spotter, authority = _authority_kws_engine("")
    authority = _bind_kws_test_pcm(
        engine,
        authority,
        np.full(1_600, 0.25, dtype="float32"),
    )
    old_stream = engine._kws_stream
    next_authority = replace(
        authority,
        capture_sequence=2,
        source_sample_start=1_600,
        source_sample_end=3_200,
        source_sample_count=1_600,
    )
    changes = {
        "capture-epoch": {"capture_epoch": 3},
        "capture-generation": {"capture_generation": 6},
        "source-generation": {"source_generation": 4},
        "source-device": {"source_device": "mic-b"},
        "source-rate": {"source_sample_rate_hz": 48_000},
        "source-domain": {"source_domain": object()},
        "kws-rate": {"kws_sample_rate_hz": 8_000},
        "authority-generation": {"authority_source_generation": 4},
        "authority-device": {"authority_source_device": "mic-b"},
        "os-route": {"os_echo_route_verified": True},
        "feed-route": {
            "route": _KwsFeedRoute.IDLE,
            "route_owner": None,
        },
        "speaker-authority": {
            "speaker_authority_available": False,
            "speaker_authority": None,
        },
        "speaking": {"speaking": False},
        "speak-generation": {"speak_generation": 8},
        "playback-generation": {"playback_generation": 12},
        "capture-sequence-gap": {"capture_sequence": 3},
        "source-sample-gap": {
            "source_sample_start": 1_601,
            "source_sample_end": 3_201,
        },
    }
    next_authority = replace(next_authority, **changes[discontinuity])

    assert engine._prepare_kws_submission(next_authority)

    assert spotter.resets == 1
    assert engine._kws_stream is not old_stream
    ledger = engine._kws_pcm_ledger
    assert ledger.scope is next_authority
    assert ledger.submitted_end == 0
    assert ledger.retained_start == 0
    assert tuple(ledger.chunks) == ()
    assert ledger.last_capture_sequence is None
    assert ledger.last_source_sample_end is None


def test_kws_tiny_feed_chunk_cap_keeps_current_without_native_reset() -> None:
    engine, spotter, authority = _speaker_available_idle_kws_case()
    callbacks = []
    engine._cb = EngineCallbacks(on_command_result=callbacks.append)
    stream = engine._kws_stream

    for index in range(_MAX_KWS_PCM_LEDGER_CHUNKS):
        block_authority = replace(
            authority,
            capture_sequence=index + 1,
            source_sample_start=index,
            source_sample_end=index + 1,
        )
        assert not engine._poll_keywords(
            np.array([index], dtype="float32"),
            control_authority=block_authority,
        )

    ledger = engine._kws_pcm_ledger
    assert engine._kws_stream is stream
    assert len(ledger.chunks) == _MAX_KWS_PCM_LEDGER_CHUNKS
    assert ledger.retained_start == 0
    assert ledger.submitted_end == _MAX_KWS_PCM_LEDGER_CHUNKS
    assert spotter.resets == 0

    overflow_index = _MAX_KWS_PCM_LEDGER_CHUNKS
    overflow_authority = replace(
        authority,
        capture_sequence=overflow_index + 1,
        source_sample_start=overflow_index,
        source_sample_end=overflow_index + 1,
    )
    assert not engine._poll_keywords(
        np.array([overflow_index], dtype="float32"),
        control_authority=overflow_authority,
    )

    ledger = engine._kws_pcm_ledger
    assert engine._kws_stream is stream
    assert len(stream.blocks) == _MAX_KWS_PCM_LEDGER_CHUNKS + 1
    assert spotter.resets == 0
    assert callbacks == []
    assert len(ledger.chunks) == _MAX_KWS_PCM_LEDGER_CHUNKS
    assert ledger.retained_start == 1
    assert ledger.submitted_end == _MAX_KWS_PCM_LEDGER_CHUNKS + 1
    assert (ledger.chunks[0].start, ledger.chunks[0].end) == (1, 2)
    assert (ledger.chunks[-1].start, ledger.chunks[-1].end) == (
        _MAX_KWS_PCM_LEDGER_CHUNKS,
        _MAX_KWS_PCM_LEDGER_CHUNKS + 1,
    )
    np.testing.assert_array_equal(
        ledger.chunks[-1].pcm,
        np.array([overflow_index], dtype="float32"),
    )
    assert ledger.last_capture_sequence == _MAX_KWS_PCM_LEDGER_CHUNKS + 1
    assert ledger.last_source_sample_end == _MAX_KWS_PCM_LEDGER_CHUNKS + 1


def test_kws_sample_window_expiry_precedes_chunk_cap_eviction() -> None:
    engine, spotter, authority = _speaker_available_idle_kws_case(
        block_sec=_MAX_KWS_PCM_LEDGER_CHUNKS / 16_000,
        window_sec=_MAX_KWS_PCM_LEDGER_CHUNKS / 16_000,
    )
    stream = engine._kws_stream
    assert engine._kws_pcm_window_sample_limit(authority) == 256

    for index in range(_MAX_KWS_PCM_LEDGER_CHUNKS + 1):
        block_authority = replace(
            authority,
            capture_sequence=index + 1,
            source_sample_start=index,
            source_sample_end=index + 1,
        )
        assert not engine._poll_keywords(
            np.ones(1, dtype="float32"),
            control_authority=block_authority,
        )

    ledger = engine._kws_pcm_ledger
    assert engine._kws_stream is stream
    assert spotter.resets == 0
    assert ledger.retained_start == 1
    assert ledger.submitted_end == 257
    assert len(ledger.chunks) == 256
    assert (ledger.chunks[0].start, ledger.chunks[0].end) == (1, 2)


def test_kws_word_regions_reject_chunk_cap_prefix_but_keep_retained_tail() -> None:
    engine, spotter, authority = _speaker_available_idle_kws_case(window_sec=20.0)
    block_samples = 640

    for index in range(_MAX_KWS_PCM_LEDGER_CHUNKS + 1):
        start = index * block_samples
        block_authority = replace(
            authority,
            capture_sequence=index + 1,
            source_sample_start=start,
            source_sample_end=start + block_samples,
            source_sample_count=block_samples,
        )
        assert not engine._poll_keywords(
            np.arange(start, start + block_samples, dtype="float32"),
            control_authority=block_authority,
        )

    ledger = engine._kws_pcm_ledger
    assert spotter.resets == 0
    assert len(ledger.chunks) == _MAX_KWS_PCM_LEDGER_CHUNKS
    assert ledger.retained_start == block_samples
    assert ledger.submitted_end == ((_MAX_KWS_PCM_LEDGER_CHUNKS + 1) * block_samples)
    assert (
        engine._kws_word_pcm_regions(
            _PLAYBACK_KWS_BINDINGS[0],
            [0.00, 0.04, 0.08, 0.12],
        )
        is None
    )

    retained = engine._kws_word_pcm_regions(
        _PLAYBACK_KWS_BINDINGS[0],
        [0.04, 0.08, 0.12, 0.16],
    )

    assert retained is not None
    assert len(retained) == 1
    np.testing.assert_array_equal(
        retained[0],
        np.arange(640, 3_200, dtype="float32"),
    )
    assert retained[0].flags.owndata
    assert not retained[0].flags.writeable


def test_kws_forged_over_cap_abstains_then_recovers_at_fresh_origin() -> None:
    engine, spotter, authority = _speaker_available_idle_kws_case()
    callbacks = []
    engine._cb = EngineCallbacks(on_command_result=callbacks.append)
    assert not engine._poll_keywords(
        np.ones(1, dtype="float32"),
        control_authority=authority,
    )
    stale_stream = engine._kws_stream
    chunk = engine._kws_pcm_ledger.chunks[0]
    engine._kws_pcm_ledger.chunks.extend(
        chunk for _ in range(_MAX_KWS_PCM_LEDGER_CHUNKS)
    )
    assert len(engine._kws_pcm_ledger.chunks) == 257

    forged_authority = replace(
        authority,
        capture_sequence=2,
        source_sample_start=1,
        source_sample_end=2,
    )
    assert not engine._poll_keywords(
        np.ones(1, dtype="float32"),
        control_authority=forged_authority,
    )

    fresh_stream = engine._kws_stream
    assert fresh_stream is not stale_stream
    assert fresh_stream.blocks == []
    assert spotter.resets == 1
    assert callbacks == []
    assert engine._kws_pcm_ledger.scope is None
    assert engine._kws_pcm_ledger.submitted_end == 0
    assert tuple(engine._kws_pcm_ledger.chunks) == ()

    recovery_authority = replace(
        authority,
        capture_sequence=3,
        source_sample_start=2,
        source_sample_end=3,
    )
    assert not engine._poll_keywords(
        np.array([3.0], dtype="float32"),
        control_authority=recovery_authority,
    )

    recovered = engine._kws_pcm_ledger
    assert engine._kws_stream is fresh_stream
    assert len(fresh_stream.blocks) == 1
    assert callbacks == []
    assert recovered.scope is recovery_authority
    assert recovered.retained_start == 0
    assert recovered.submitted_end == 1
    assert len(recovered.chunks) == 1
    assert (recovered.chunks[0].start, recovered.chunks[0].end) == (0, 1)
    assert recovered.last_capture_sequence == 3
    assert recovered.last_source_sample_end == 3


def test_kws_submitted_clock_overflow_abstains_then_recovers() -> None:
    engine, spotter, authority = _speaker_available_idle_kws_case()
    callbacks = []
    engine._cb = EngineCallbacks(on_command_result=callbacks.append)
    assert not engine._poll_keywords(
        np.ones(1, dtype="float32"),
        control_authority=authority,
    )
    stale_stream = engine._kws_stream
    engine._kws_pcm_ledger.submitted_end = sys.maxsize - 1

    boundary_authority = replace(
        authority,
        capture_sequence=2,
        source_sample_start=1,
        source_sample_end=2,
    )
    assert not engine._poll_keywords(
        np.ones(1, dtype="float32"),
        control_authority=boundary_authority,
    )
    assert engine._kws_stream is stale_stream
    assert engine._kws_pcm_ledger.submitted_end == sys.maxsize
    assert spotter.resets == 0

    overflow_authority = replace(
        authority,
        capture_sequence=3,
        source_sample_start=2,
        source_sample_end=3,
    )
    assert not engine._poll_keywords(
        np.ones(1, dtype="float32"),
        control_authority=overflow_authority,
    )

    fresh_stream = engine._kws_stream
    assert fresh_stream is not stale_stream
    assert fresh_stream.blocks == []
    assert spotter.resets == 1
    assert callbacks == []
    assert engine._kws_pcm_ledger.scope is None
    assert engine._kws_pcm_ledger.submitted_end == 0
    assert tuple(engine._kws_pcm_ledger.chunks) == ()

    recovery_authority = replace(
        authority,
        capture_sequence=4,
        source_sample_start=3,
        source_sample_end=4,
    )
    assert not engine._poll_keywords(
        np.array([3.0], dtype="float32"),
        control_authority=recovery_authority,
    )

    recovered = engine._kws_pcm_ledger
    assert engine._kws_stream is fresh_stream
    assert len(fresh_stream.blocks) == 1
    assert callbacks == []
    assert recovered.scope is recovery_authority
    assert recovered.retained_start == 0
    assert recovered.submitted_end == 1
    assert len(recovered.chunks) == 1
    assert (recovered.chunks[0].start, recovered.chunks[0].end) == (0, 1)
    assert recovered.last_capture_sequence == 4
    assert recovered.last_source_sample_end == 4


def _warmed_speaker_scoring_engine(embed_fn):
    engine = SherpaOnnxEngine(SherpaConfig())
    gate = SpeakerGate(threshold=0.5, embed_fn=embed_fn)
    gate.enroll_embedding([1.0, 0.0])
    engine._replace_speaker_gate(gate)
    assert engine._warm_speaker_gate()
    authority = engine._snapshot_kws_speaker_authority()
    assert authority is not None
    return engine, gate, authority


def _prepared_kws_inference_case(embed_fn):
    engine, _spotter, control_authority = _authority_kws_engine(
        "stop",
        timestamps=[0.00, 0.04, 0.08, 0.12],
    )
    gate = SpeakerGate(threshold=0.5, embed_fn=embed_fn)
    gate.enroll_embedding([1.0, 0.0])
    engine._replace_speaker_gate(gate)
    assert engine._warm_speaker_gate()
    speaker_authority = engine._snapshot_kws_speaker_authority()
    assert speaker_authority is not None
    control_authority = replace(
        control_authority,
        speaker_authority_available=True,
        speaker_authority=speaker_authority,
    )
    prepared = engine._prepare_kws_speaker_pcm_batch(
        (np.full(1_600, 0.25, dtype="float32"),),
        speaker_authority,
    )
    assert tuple(item.status for item in prepared) == (_SpeakerPcmStatus.DEFER,)
    return engine, gate, control_authority, speaker_authority, prepared


def _ambiguous_one_word_kws_case(embed_fn):
    engine, spotter, control_authority = _authority_kws_engine(
        "stop",
        timestamps=[0.00, 0.04, 0.08, 0.12],
    )
    gate = SpeakerGate(threshold=0.5, embed_fn=embed_fn)
    gate.enroll_embedding([1.0, 0.0])
    engine._replace_speaker_gate(gate)
    assert engine._warm_speaker_gate()
    speaker_authority = engine._snapshot_kws_speaker_authority()
    assert speaker_authority is not None
    control_authority = replace(
        control_authority,
        source_sample_count=3_200,
        source_sample_end=3_200,
        speaker_authority_available=True,
        speaker_authority=speaker_authority,
    )
    engine._now_playing = "I will stop after this sentence."
    pcm = np.full(3_200, 0.25, dtype="float32")
    return engine, spotter, control_authority, speaker_authority, pcm


def test_stop_wins_capture_condition_before_kws_worker_start(monkeypatch) -> None:
    engine, _gate, control_authority, speaker_authority, prepared = (
        _prepared_kws_inference_case(lambda _samples, _sample_rate: [1.0, 0.0])
    )
    claimed = threading.Event()
    outcomes = []

    class _StartProbeOwner:
        ticket = object()

        def __init__(self) -> None:
            self.start_calls = 0
            self.receive_calls = 0
            self.abandoned = []
            self.reap_calls = 0

        def start(self):
            self.start_calls += 1
            return self.ticket

        def receive(self, _ticket):
            self.receive_calls += 1
            raise AssertionError("stop-first owner must never be received")

        def abandon(self, outcome):
            self.abandoned.append(outcome)
            return True

        def try_reap(self):
            self.reap_calls += 1
            return True

    owner = _StartProbeOwner()

    def claim(*_args, **_kwargs):
        claimed.set()
        return owner

    monkeypatch.setattr(
        sherpa_module,
        "try_claim_kws_speaker_inference_owner",
        claim,
    )

    def run() -> None:
        outcomes.extend(
            engine._run_kws_speaker_inference(
                prepared,
                expected_authority=speaker_authority,
                control_authority=control_authority,
                deadline=time.monotonic() + 5.0,
            )
        )

    capture_worker = threading.Thread(target=run)
    with engine._capture_effect_condition:
        capture_worker.start()
        assert claimed.wait(timeout=2.0)
        engine._capture_epoch += 1
        engine._capture_stopping.set()
        engine._capture_effect_condition.notify_all()
    capture_worker.join(timeout=2.0)

    assert not capture_worker.is_alive()
    assert tuple(item.status for item in outcomes) == (_SpeakerPcmStatus.STALE,)
    assert owner.start_calls == 0
    assert owner.receive_calls == 0
    assert owner.abandoned == [KwsSpeakerInferenceOutcome.STOPPED]
    assert owner.reap_calls == 1
    assert engine._kws_speaker_inference_owner is None
    assert engine._kws_speaker_inference_disabled_epoch is None


def test_current_epoch_deadline_expiry_before_worker_start_latches_timeout(
    monkeypatch,
) -> None:
    engine, _gate, control_authority, speaker_authority, prepared = (
        _prepared_kws_inference_case(lambda _samples, _sample_rate: [1.0, 0.0])
    )
    clock = [0.0]
    monkeypatch.setattr(sherpa_module.time, "monotonic", lambda: clock[0])

    class _DeadlineOwner:
        ticket = object()

        def __init__(self) -> None:
            self.start_calls = 0
            self.receive_calls = 0
            self.abandoned = []
            self.reap_calls = 0

        def start(self):
            self.start_calls += 1
            return self.ticket

        def receive(self, _ticket):
            self.receive_calls += 1
            raise AssertionError("expired owner must never be received")

        def abandon(self, outcome):
            self.abandoned.append(outcome)
            return True

        def try_reap(self):
            self.reap_calls += 1
            return True

    owner = _DeadlineOwner()

    def claim(*_args, **_kwargs):
        clock[0] = 1.0
        return owner

    monkeypatch.setattr(
        sherpa_module,
        "try_claim_kws_speaker_inference_owner",
        claim,
    )

    decisions = engine._run_kws_speaker_inference(
        prepared,
        expected_authority=speaker_authority,
        control_authority=control_authority,
        deadline=1.0,
    )

    assert tuple(item.status for item in decisions) == (_SpeakerPcmStatus.ERROR,)
    assert owner.start_calls == 0
    assert owner.receive_calls == 0
    assert owner.abandoned == [KwsSpeakerInferenceOutcome.TIMEOUT]
    assert owner.reap_calls == 1
    assert engine._kws_speaker_inference_owner is None
    assert engine._kws_speaker_inference_is_disabled(control_authority.capture_epoch)


def test_kws_worker_start_wins_before_stop_then_stop_wakes_without_join() -> None:
    native_entered = threading.Event()
    release_native = threading.Event()
    block_native = False
    native_calls = []

    def embed(_samples, _sample_rate):
        if block_native:
            native_calls.append("native")
            native_entered.set()
            assert release_native.wait(timeout=2.0)
        return [1.0, 0.0]

    engine, _gate, control_authority, speaker_authority, prepared = (
        _prepared_kws_inference_case(embed)
    )
    block_native = True
    outcomes = []
    capture_worker = threading.Thread(
        target=lambda: outcomes.extend(
            engine._run_kws_speaker_inference(
                prepared,
                expected_authority=speaker_authority,
                control_authority=control_authority,
                deadline=time.monotonic() + 5.0,
            )
        )
    )
    capture_worker.start()
    assert native_entered.wait(timeout=2.0)
    with engine._kws_speaker_inference_lock:
        owner = engine._kws_speaker_inference_owner
    assert owner is not None

    with engine._capture_effect_condition:
        engine._capture_epoch += 1
        engine._capture_stopping.set()
        engine._capture_effect_condition.notify_all()
    engine._abandon_owned_kws_speaker_inference()
    capture_worker.join(timeout=2.0)

    assert not capture_worker.is_alive()
    assert tuple(item.status for item in outcomes) == (_SpeakerPcmStatus.ERROR,)
    assert owner.snapshot().abandoned
    assert owner.snapshot().thread_alive
    assert native_calls == ["native"]

    release_native.set()
    owner._thread.join(timeout=2.0)
    assert not owner.snapshot().thread_alive
    engine._require_kws_speaker_inference_idle_before_mutation()
    assert engine._kws_speaker_inference_owner is None


@pytest.mark.parametrize(
    ("receipt_case", "expected_status", "expected_latch"),
    [
        ("nonfinite", _SpeakerPcmStatus.ERROR, True),
        ("stale", _SpeakerPcmStatus.STALE, False),
        ("reject", _SpeakerPcmStatus.REJECT, False),
        ("defer", _SpeakerPcmStatus.DEFER, False),
        ("busy", _SpeakerPcmStatus.BUSY, False),
    ],
)
def test_kws_capture_reduction_latches_only_error_receipts(
    monkeypatch,
    receipt_case,
    expected_status,
    expected_latch,
) -> None:
    engine, _gate, control_authority, speaker_authority, prepared = (
        _prepared_kws_inference_case(lambda _samples, _sample_rate: [1.0, 0.0])
    )
    if receipt_case == "busy":
        result = KwsSpeakerInferenceResult(KwsSpeakerInferenceOutcome.BUSY)
    else:
        receipt_authority = speaker_authority.gate_authority
        if receipt_case == "stale":
            other_gate = SpeakerGate(
                threshold=0.5,
                embed_fn=lambda _samples, _sample_rate: [1.0, 0.0],
            )
            other_gate.enroll_embedding([1.0, 0.0])
            receipt_authority = SpeakerGate.authority_state(other_gate)
        if receipt_case == "nonfinite":
            similarity = float("nan")
        elif receipt_case == "reject":
            similarity = 0.0
        elif receipt_case == "defer":
            similarity = float(
                (
                    speaker_authority.policy.accept_threshold
                    + speaker_authority.policy.reject_threshold
                )
                / 2.0
            )
        else:
            similarity = 1.0
        result = KwsSpeakerInferenceResult(
            KwsSpeakerInferenceOutcome.COMPLETED,
            receipt=SpeakerSimilarityBatchReceipt(
                authority=receipt_authority,
                similarities=(similarity,),
            ),
        )

    class _ResultOwner:
        ticket = object()

        def __init__(self) -> None:
            self.start_calls = 0
            self.receive_calls = 0
            self.reap_calls = 0

        def start(self):
            self.start_calls += 1
            return self.ticket

        def receive(self, ticket):
            self.receive_calls += 1
            assert ticket is self.ticket
            return result

        def abandon(self, _outcome):
            raise AssertionError("completed fake owner must not be abandoned")

        def try_reap(self):
            self.reap_calls += 1
            return True

    owner = _ResultOwner()
    monkeypatch.setattr(
        sherpa_module,
        "try_claim_kws_speaker_inference_owner",
        lambda *_args, **_kwargs: owner,
    )

    decisions = engine._run_kws_speaker_inference(
        prepared,
        expected_authority=speaker_authority,
        control_authority=control_authority,
        deadline=time.monotonic() + 5.0,
    )

    assert tuple(item.status for item in decisions) == (expected_status,)
    assert (
        engine._kws_speaker_inference_is_disabled(control_authority.capture_epoch)
        is expected_latch
    )
    assert owner.start_calls == 1
    assert owner.receive_calls == 1
    assert owner.reap_calls == 1
    assert engine._kws_speaker_inference_owner is None


def test_successful_kws_speaker_callback_runs_on_capture_caller_not_worker() -> None:
    scoring = False
    speaker_worker_threads = []

    def embed(_samples, _sample_rate):
        if scoring:
            speaker_worker_threads.append(threading.get_ident())
        return [1.0, 0.0]

    engine, _spotter, authority, _speaker_authority, pcm = _ambiguous_one_word_kws_case(
        embed
    )
    scoring = True
    callback_threads = []
    engine._cb = EngineCallbacks(
        on_control_stop_result=lambda _result: callback_threads.append(
            threading.get_ident()
        )
    )
    capture_caller = threading.get_ident()

    assert engine._poll_keywords(pcm, control_authority=authority)

    assert callback_threads == [capture_caller]
    assert len(speaker_worker_threads) == 1
    assert speaker_worker_threads[0] != capture_caller
    assert engine._kws_speaker_inference_owner is None


def test_stop_abandon_wakes_capture_and_late_native_return_never_callbacks() -> None:
    native_entered = threading.Event()
    release_native = threading.Event()
    block_native = False

    def embed(_samples, _sample_rate):
        if block_native:
            native_entered.set()
            assert release_native.wait(timeout=2.0)
        return [1.0, 0.0]

    engine, _spotter, authority, _speaker_authority, pcm = _ambiguous_one_word_kws_case(
        embed
    )
    block_native = True
    callbacks = []
    capture_results = []
    engine._cb = EngineCallbacks(on_control_stop_result=callbacks.append)
    capture_caller = threading.Thread(
        target=lambda: capture_results.append(
            engine._poll_keywords(pcm, control_authority=authority)
        )
    )

    capture_caller.start()
    assert native_entered.wait(timeout=2.0)
    with engine._kws_speaker_inference_lock:
        owner = engine._kws_speaker_inference_owner
    assert owner is not None
    with engine._capture_effect_condition:
        engine._capture_epoch += 1
        engine._capture_stopping.set()
        engine._capture_effect_condition.notify_all()
    engine._abandon_owned_kws_speaker_inference()
    capture_caller.join(timeout=2.0)

    assert not capture_caller.is_alive()
    assert capture_results == [False]
    assert callbacks == []
    assert owner.snapshot().abandoned
    assert owner.snapshot().thread_alive

    release_native.set()
    owner._thread.join(timeout=2.0)
    assert not owner.snapshot().thread_alive
    assert engine._reap_finished_owned_kws_speaker_inference()
    assert engine._kws_speaker_inference_owner is None
    assert callbacks == []


def test_speaker_action_deadline_at_callback_admission_never_terminalizes(
    monkeypatch,
) -> None:
    engine, _spotter, authority, _speaker_authority, pcm = _ambiguous_one_word_kws_case(
        lambda _samples, _sample_rate: [1.0, 0.0]
    )
    tracker = AcousticTurnTracker("stream-a")
    tracker.rotate_capture(capture_epoch=2, capture_generation=5)
    before, _revision = tracker.partial(emitted_at=10.0)
    engine._acoustic_turn_tracker = tracker
    callbacks = []
    engine._cb = EngineCallbacks(on_control_stop_result=callbacks.append)
    clock = [100.0]
    monkeypatch.setattr(sherpa_module.time, "monotonic", lambda: clock[0])
    original_claim = engine._claim_playback_kws_if_current

    def claim_then_expire(control_authority, *, stop_only=None):
        claim = original_claim(control_authority, stop_only=stop_only)
        assert claim is not None
        clock[0] = 101.0
        return claim

    engine._claim_playback_kws_if_current = claim_then_expire

    assert not engine._poll_keywords(pcm, control_authority=authority)

    after = tracker.current(emitted_at=11.0)
    assert after is not None
    assert after.spans[-1].key == before.spans[-1].key
    assert tracker.active
    assert callbacks == []
    assert engine._playback_kws_claim is None
    assert engine._kws_speaker_inference_is_disabled(authority.capture_epoch)


@pytest.mark.parametrize(
    ("levels", "statuses"),
    [
        ((0.25,), (_SpeakerPcmStatus.ACCEPT,)),
        (
            (0.25, 0.5),
            (_SpeakerPcmStatus.ACCEPT, _SpeakerPcmStatus.ACCEPT),
        ),
        (
            (0.25, -0.5),
            (_SpeakerPcmStatus.ACCEPT, _SpeakerPcmStatus.REJECT),
        ),
        (
            (-0.25, 0.5),
            (_SpeakerPcmStatus.REJECT, _SpeakerPcmStatus.ACCEPT),
        ),
    ],
    ids=("one-owner", "two-owner", "owner-then-nonowner", "nonowner-then-owner"),
)
def test_explicit_speaker_batch_scores_every_word_independently(levels, statuses):
    embedded: list[np.ndarray] = []

    def embed(samples, _sample_rate):
        clip = np.asarray(samples, dtype="float32").copy()
        embedded.append(clip)
        return [1.0, 0.0] if float(np.mean(clip)) > 0.0 else [0.0, 1.0]

    engine, _gate, authority = _warmed_speaker_scoring_engine(embed)
    embedded.clear()
    regions = tuple(np.full(1_920, level, dtype="float32") for level in levels)

    decisions = engine._score_explicit_speaker_pcm_batch(
        regions,
        min_sec=0.10,
        expected_authority=authority,
    )

    assert tuple(decision.status for decision in decisions) == statuses
    assert len(embedded) == len(regions) <= 2
    for decision, region, scored in zip(
        decisions,
        regions,
        embedded,
        strict=True,
    ):
        np.testing.assert_array_equal(decision.identity_pcm, region)
        np.testing.assert_array_equal(scored, region)


def test_explicit_speaker_batch_unavailable_cold_and_insufficient_abstain() -> None:
    regions = (
        np.full(1_600, 0.25, dtype="float32"),
        np.full(1_600, 0.5, dtype="float32"),
    )
    unavailable = SherpaOnnxEngine(SherpaConfig())
    assert (
        tuple(
            decision.status
            for decision in unavailable._score_explicit_speaker_pcm_batch(
                regions,
                min_sec=0.10,
            )
        )
        == (_SpeakerPcmStatus.UNAVAILABLE,) * 2
    )

    cold = SherpaOnnxEngine(SherpaConfig())
    cold_gate = SpeakerGate(embed_fn=lambda _samples, _sample_rate: [1.0, 0.0])
    cold_gate.enroll_embedding([1.0, 0.0])
    cold._replace_speaker_gate(cold_gate)
    assert (
        tuple(
            decision.status
            for decision in cold._score_explicit_speaker_pcm_batch(
                regions,
                min_sec=0.10,
            )
        )
        == (_SpeakerPcmStatus.COLD,) * 2
    )

    warmed, _gate, authority = _warmed_speaker_scoring_engine(
        lambda _samples, _sample_rate: [1.0, 0.0]
    )
    insufficient = warmed._score_explicit_speaker_pcm_batch(
        (
            np.full(1_599, 0.25, dtype="float32"),
            np.zeros(1_600, dtype="float32"),
        ),
        min_sec=0.10,
        expected_authority=authority,
    )
    assert tuple(item.status for item in insufficient) == (
        _SpeakerPcmStatus.INSUFFICIENT,
        _SpeakerPcmStatus.INSUFFICIENT,
    )


def test_explicit_speaker_batch_busy_error_and_stale_fail_closed() -> None:
    fail_embedding = False

    def embed(_samples, _sample_rate):
        if fail_embedding:
            raise RuntimeError("injected KWS speaker failure")
        return [1.0, 0.0]

    engine, gate, authority = _warmed_speaker_scoring_engine(embed)
    region = (np.full(1_600, 0.25, dtype="float32"),)

    assert gate._inference_lock.acquire(blocking=False)
    try:
        busy = engine._score_explicit_speaker_pcm_batch(
            region,
            min_sec=0.10,
            expected_authority=authority,
        )
    finally:
        gate._inference_lock.release()
    assert tuple(item.status for item in busy) == (_SpeakerPcmStatus.BUSY,)

    fail_embedding = True
    error = engine._score_explicit_speaker_pcm_batch(
        region,
        min_sec=0.10,
        expected_authority=authority,
    )
    assert tuple(item.status for item in error) == (_SpeakerPcmStatus.ERROR,)

    fail_embedding = False
    gate.enroll_embedding([1.0, 0.0])
    stale = engine._score_explicit_speaker_pcm_batch(
        region,
        min_sec=0.10,
        expected_authority=authority,
    )
    assert tuple(item.status for item in stale) == (_SpeakerPcmStatus.STALE,)


@pytest.mark.parametrize(
    "similarity",
    [float("nan"), float("inf"), float("-inf"), 1.001, -1.001, np.float64(1.0)],
    ids=("nan", "positive-inf", "negative-inf", "high", "low", "numpy-float"),
)
def test_explicit_speaker_batch_rejects_nonfinite_out_of_range_or_typed_scores(
    monkeypatch,
    similarity,
) -> None:
    engine, gate, authority = _warmed_speaker_scoring_engine(
        lambda _samples, _sample_rate: [1.0, 0.0]
    )

    def forged_batch(_gate, _samples, _sample_rate):
        return SpeakerSimilarityBatchReceipt(
            authority=authority.gate_authority,
            similarities=(similarity,),
        )

    monkeypatch.setattr(SpeakerGate, "try_similarity_batch", forged_batch)

    decisions = engine._score_explicit_speaker_pcm_batch(
        (np.full(1_600, 0.25, dtype="float32"),),
        min_sec=0.10,
        expected_authority=authority,
    )

    assert tuple(item.status for item in decisions) == (_SpeakerPcmStatus.ERROR,)
    assert engine._kws_speaker_authority_is_current(authority)
    assert gate.authority_is_current(authority.gate_authority)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("barge_word_cut_speaker_threshold", float("nan")),
        ("barge_word_cut_speaker_threshold", float("inf")),
        ("barge_word_cut_speaker_reject_threshold", float("nan")),
        ("barge_word_cut_speaker_reject_threshold", float("-inf")),
    ],
)
def test_explicit_speaker_batch_marks_nonfinite_mutated_policy_stale(field, value):
    engine, _gate, authority = _warmed_speaker_scoring_engine(
        lambda _samples, _sample_rate: [1.0, 0.0]
    )
    setattr(engine.config, field, value)

    decisions = engine._score_explicit_speaker_pcm_batch(
        (np.full(1_600, 0.25, dtype="float32"),),
        min_sec=0.10,
        expected_authority=authority,
    )

    assert tuple(item.status for item in decisions) == (_SpeakerPcmStatus.STALE,)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("barge_word_cut_speaker_threshold", 0.75),
        ("barge_word_cut_speaker_reject_threshold", 0.10),
    ],
)
def test_explicit_speaker_batch_policy_mutation_during_inference_is_stale(
    field,
    value,
) -> None:
    inference_started = threading.Event()
    release_inference = threading.Event()
    block_inference = False

    def embed(_samples, _sample_rate):
        if block_inference:
            inference_started.set()
            assert release_inference.wait(timeout=2.0)
        return [1.0, 0.0]

    engine, _gate, authority = _warmed_speaker_scoring_engine(embed)
    block_inference = True
    decisions = []
    worker = threading.Thread(
        target=lambda: decisions.extend(
            engine._score_explicit_speaker_pcm_batch(
                (np.full(1_600, 0.25, dtype="float32"),),
                min_sec=0.10,
                expected_authority=authority,
            )
        )
    )

    worker.start()
    assert inference_started.wait(timeout=2.0)
    setattr(engine.config, field, value)
    release_inference.set()
    worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert tuple(item.status for item in decisions) == (_SpeakerPcmStatus.STALE,)
    assert not engine._kws_speaker_authority_is_current(authority)


def _ambiguous_two_word_kws_case(levels: tuple[float, float]):
    timestamps = [0.00, 0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32]
    engine, spotter, authority = _authority_kws_engine(
        "",
        phrase_index=1,
        timestamps=timestamps,
    )
    embedded: list[np.ndarray] = []

    def embed(samples, _sample_rate):
        clip = np.asarray(samples, dtype="float32").copy()
        embedded.append(clip)
        return [1.0, 0.0] if float(np.mean(clip)) > 0.0 else [0.0, 1.0]

    gate = SpeakerGate(threshold=0.5, embed_fn=embed)
    gate.enroll_embedding([1.0, 0.0])
    engine._replace_speaker_gate(gate)
    assert engine._warm_speaker_gate()
    speaker_authority = engine._snapshot_kws_speaker_authority()
    assert speaker_authority is not None
    embedded.clear()
    authority = replace(
        authority,
        source_sample_count=3_200,
        source_sample_end=3_200,
        speaker_authority_available=True,
        speaker_authority=speaker_authority,
    )
    pcm = np.concatenate(
        (
            np.full(2_560, levels[0], dtype="float32"),
            np.full(3_200, levels[1], dtype="float32"),
            np.full(640, levels[1], dtype="float32"),
        )
    )
    return engine, spotter, gate, authority, pcm, embedded


@pytest.mark.parametrize(
    ("levels", "accepted"),
    [
        ((0.25, 0.50), True),
        ((0.25, -0.50), False),
        ((-0.25, 0.50), False),
    ],
    ids=("both-owner", "owner-then-nonowner", "nonowner-then-owner"),
)
def test_ambiguous_two_word_kws_requires_every_word_from_same_owner(
    levels,
    accepted,
) -> None:
    engine, spotter, _gate, authority, pcm, embedded = _ambiguous_two_word_kws_case(
        levels
    )
    stop_results = []
    engine._cb = EngineCallbacks(on_control_stop_result=stop_results.append)
    engine._now_playing = "Please stop talking while I finish this sentence."
    engine._word_cut_base = "untouched word-cut base"
    engine._word_cut_candidate_samples = 37
    engine._wc_stats = {"sentinel": 9}
    word_cut_state = (
        engine._word_cut_base,
        engine._word_cut_candidate_samples,
        dict(engine._wc_stats),
    )

    assert not engine._poll_keywords(
        pcm[:3_200],
        control_authority=authority,
    )
    spotter.keyword = "stop"
    second_authority = replace(
        authority,
        capture_sequence=2,
        source_sample_start=3_200,
        source_sample_end=6_400,
    )
    consumed = engine._poll_keywords(
        pcm[3_200:],
        control_authority=second_authority,
    )

    assert consumed is accepted
    assert [result.text for result in stop_results] == (["stop"] if accepted else [])
    assert len(embedded) == 2
    np.testing.assert_array_equal(embedded[0], pcm[:2_560])
    np.testing.assert_array_equal(embedded[1], pcm[2_560:5_760])
    assert engine._playback_kws_claim is None
    assert engine._kws_own_echo_suppressions == (0 if accepted else 1)
    assert (
        engine._word_cut_base,
        engine._word_cut_candidate_samples,
        engine._wc_stats,
    ) == word_cut_state


def test_ambiguous_kws_releases_pcm_and_native_stream_before_stop_callback() -> None:
    engine, spotter, _gate, authority, pcm, _embedded = _ambiguous_two_word_kws_case(
        (0.25, 0.50)
    )
    fed_pcm_refs: list[weakref.ReferenceType[np.ndarray]] = []
    heavy_pcm_refs: list[weakref.ReferenceType[np.ndarray]] = []

    class _RetainingKwsStream:
        def __init__(self) -> None:
            self.blocks: list[np.ndarray] = []

        def accept_waveform(self, _sample_rate, samples) -> None:
            fed_pcm_refs.append(weakref.ref(samples))
            self.blocks.append(samples)

    retaining_stream = _RetainingKwsStream()
    engine._kws_stream = retaining_stream
    old_stream_ref = weakref.ref(retaining_stream)
    del retaining_stream

    original_prepare = engine._prepare_kws_speaker_pcm_batch

    def observe_heavy_pcm(pcm_regions, expected_authority):
        prepared = original_prepare(pcm_regions, expected_authority)
        observed: list[np.ndarray] = [*pcm_regions]
        roots = {
            id(root): root
            for root in (_array_storage_root(region) for region in pcm_regions)
        }
        observed.extend(roots.values())
        observed.extend(
            decision.identity_pcm
            for decision in prepared
            if decision.identity_pcm is not None
        )
        heavy_pcm_refs.extend(weakref.ref(array) for array in observed)
        return prepared

    engine._prepare_kws_speaker_pcm_batch = observe_heavy_pcm
    stop_results = []

    def assert_released_then_stop(result) -> None:
        gc.collect()
        assert old_stream_ref() is None
        assert len(fed_pcm_refs) == 2
        assert all(reference() is None for reference in fed_pcm_refs)
        assert len(heavy_pcm_refs) >= 5
        assert all(reference() is None for reference in heavy_pcm_refs)
        ledger = engine._kws_pcm_ledger
        assert ledger.scope is None
        assert ledger.submitted_end == 0
        assert tuple(ledger.chunks) == ()
        stop_results.append(result)

    engine._cb = EngineCallbacks(on_control_stop_result=assert_released_then_stop)
    engine._now_playing = "Please stop talking while I finish this sentence."
    assert not engine._poll_keywords(
        pcm[:3_200],
        control_authority=authority,
    )
    spotter.keyword = "stop"
    second_authority = replace(
        authority,
        capture_sequence=2,
        source_sample_start=3_200,
        source_sample_end=6_400,
    )

    assert engine._poll_keywords(
        pcm[3_200:],
        control_authority=second_authority,
    )

    assert [result.text for result in stop_results] == ["stop"]
    assert engine._playback_kws_claim is None


def test_post_score_speaker_mutation_at_claim_install_abstains_and_releases() -> None:
    ledger_state_during_score = []
    scoring = False

    def embed(_samples, _sample_rate):
        if scoring:
            ledger = engine._kws_pcm_ledger
            ledger_state_during_score.append(
                (ledger.scope, tuple(ledger.chunks), ledger.submitted_end)
            )
        return [1.0, 0.0]

    engine, spotter, authority = _authority_kws_engine(
        "stop",
        timestamps=[0.00, 0.04, 0.08, 0.12],
    )
    gate = SpeakerGate(threshold=0.5, embed_fn=embed)
    gate.enroll_embedding([1.0, 0.0])
    engine._replace_speaker_gate(gate)
    assert engine._warm_speaker_gate()
    speaker_authority = engine._snapshot_kws_speaker_authority()
    assert speaker_authority is not None
    authority = replace(
        authority,
        source_sample_count=3_200,
        source_sample_end=3_200,
        speaker_authority_available=True,
        speaker_authority=speaker_authority,
    )
    engine._now_playing = "I will stop after this sentence."
    stop_results = []
    engine._cb = EngineCallbacks(on_control_stop_result=stop_results.append)
    original_current = engine._kws_control_authority_is_current
    mutated = False

    def mutate_after_claim(candidate):
        nonlocal mutated
        if engine._playback_kws_claim is not None and not mutated:
            mutated = True
            gate.enroll_embedding([1.0, 0.0])
        return original_current(candidate)

    engine._kws_control_authority_is_current = mutate_after_claim
    scoring = True

    assert not engine._poll_keywords(
        np.full(3_200, 0.25, dtype="float32"),
        control_authority=authority,
    )

    assert spotter.timestamp_calls == 1
    assert mutated
    assert ledger_state_during_score == [(None, (), 0)]
    assert stop_results == []
    assert engine._playback_kws_claim is None


def test_kws_own_echo_suppressed_when_now_playing_contains_keyword():
    # The assistant's own TTS ("I'll stop now") leaks back through imperfect AEC
    # and the spotter fires on it -- the own-echo guard must drop the hit so it
    # cannot self-interrupt.
    engine, commands = _kws_engine("stop")
    engine._speaking.set()
    engine._now_playing = "Okay, I will stop now."
    stop_results = []
    engine._cb = EngineCallbacks(
        on_command=commands.append,
        on_control_stop_result=stop_results.append,
    )

    consumed = engine._poll_keywords(np.zeros(1600, dtype="float32"))

    assert commands == []
    assert stop_results == []
    assert consumed is False
    assert engine._kws_own_echo_suppressions == 1
    assert engine._kws.resets == 1  # stream is still reset after the hit


def test_kws_hit_passes_when_not_speaking_even_if_now_playing_matches(
    monkeypatch,
):
    # Not speaking -> the guard is inert; a genuine user "stop" must reach the
    # command callback even if the last-played line contained the word.
    engine, commands = _kws_engine("stop")
    engine._now_playing = "Okay, I will stop now."
    permit_before = runtime_speaker_inference_permit().snapshot()
    monkeypatch.setattr(
        sherpa_module,
        "try_claim_kws_speaker_inference_owner",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("idle KWS must not allocate a speaker task")
        ),
    )

    consumed = engine._poll_keywords(np.zeros(1600, dtype="float32"))

    assert commands == ["stop"]
    assert consumed is True
    assert engine._kws_own_echo_suppressions == 0
    assert runtime_speaker_inference_permit().snapshot() == permit_before


def test_kws_hit_passes_when_now_playing_does_not_contain_keyword():
    # Speaking, but the playing sentence does not contain the spotted word ->
    # this is the user talking over the assistant, not its own echo. Pass it.
    engine, commands = _kws_engine("stop")
    engine._speaking.set()
    engine._now_playing = "The weather is nice today."
    stop_commands = []
    engine._cb = EngineCallbacks(
        on_command=commands.append,
        on_control_stop_result=stop_commands.append,
    )

    consumed = engine._poll_keywords(np.zeros(1600, dtype="float32"))

    assert commands == []
    assert [result.text for result in stop_commands] == ["stop"]
    assert consumed is True
    assert engine._kws_own_echo_suppressions == 0


@pytest.mark.parametrize(
    "keyword,phrase_index",
    [("stop", 0), ("wait", 4)],
)
def test_playback_kws_uses_only_typed_stop_callback_and_canonicalizes_wait(
    keyword,
    phrase_index,
    monkeypatch,
) -> None:
    engine, spotter, authority = _authority_kws_engine(
        keyword,
        phrase_index=phrase_index,
    )
    engine._now_playing = "The weather is pleasant today."
    generic_commands = []
    generic_results = []
    stop_results = []
    engine._cb = EngineCallbacks(
        on_command=generic_commands.append,
        on_command_result=generic_results.append,
        on_control_stop_result=stop_results.append,
    )

    permit_before = runtime_speaker_inference_permit().snapshot()
    monkeypatch.setattr(
        sherpa_module,
        "try_claim_kws_speaker_inference_owner",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("novel STOP must not allocate a speaker task")
        ),
    )

    consumed = engine._poll_keywords(
        np.zeros(1600, dtype="float32"),
        control_authority=authority,
    )

    assert consumed is True
    assert generic_commands == []
    assert generic_results == []
    assert [result.text for result in stop_results] == ["stop"]
    assert spotter.token_calls == 1
    assert spotter.timestamp_calls == 0
    assert engine._playback_kws_claim is None
    assert runtime_speaker_inference_permit().snapshot() == permit_before


def test_playback_non_stop_abstains_while_idle_generic_label_is_unchanged(
    monkeypatch,
) -> None:
    permit_before = runtime_speaker_inference_permit().snapshot()
    monkeypatch.setattr(
        sherpa_module,
        "try_claim_kws_speaker_inference_owner",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("non-ambiguous KWS must not allocate a speaker task")
        ),
    )
    playback, playback_spotter, playback_authority = _authority_kws_engine(
        "command mode"
    )
    playback_generic = []
    playback_stop = []
    playback._cb = EngineCallbacks(
        on_command_result=playback_generic.append,
        on_control_stop_result=playback_stop.append,
    )

    assert not playback._poll_keywords(
        np.zeros(1600, dtype="float32"),
        control_authority=playback_authority,
    )
    assert playback_generic == []
    assert playback_stop == []
    assert playback_spotter.token_calls == 1

    idle, idle_spotter, idle_authority = _authority_kws_engine(
        "command mode",
        speaking=False,
    )
    idle_generic = []
    idle_stop = []
    idle._cb = EngineCallbacks(
        on_command_result=idle_generic.append,
        on_control_stop_result=idle_stop.append,
    )

    assert idle._poll_keywords(
        np.zeros(1600, dtype="float32"),
        control_authority=idle_authority,
    )
    assert [result.text for result in idle_generic] == ["command mode"]
    assert idle_stop == []
    assert idle_spotter.token_calls == 0
    assert idle_spotter.timestamp_calls == 0
    assert runtime_speaker_inference_permit().snapshot() == permit_before


def test_playback_kws_uses_stop_callback_captured_before_native_tokens() -> None:
    original_results = []
    replacement_results = []
    engine = None

    def replace_callback() -> None:
        engine._cb = EngineCallbacks(
            on_control_stop_result=replacement_results.append,
        )

    engine, spotter, authority = _authority_kws_engine(
        "stop",
        before_tokens=replace_callback,
    )
    engine._now_playing = "A sentence with no control words."
    engine._cb = EngineCallbacks(on_control_stop_result=original_results.append)

    assert engine._poll_keywords(
        np.zeros(1_600, dtype="float32"),
        control_authority=authority,
    )

    assert [result.text for result in original_results] == ["stop"]
    assert replacement_results == []
    assert spotter.token_calls == 1
    assert spotter.timestamp_calls == 0


def test_ambiguous_kws_uses_stop_callback_captured_before_timestamps() -> None:
    engine, spotter, _gate, authority, pcm, _embedded = _ambiguous_two_word_kws_case(
        (0.25, 0.50)
    )
    original_results = []
    replacement_results = []

    def replace_callback() -> None:
        engine._cb = EngineCallbacks(
            on_control_stop_result=replacement_results.append,
        )

    spotter._before_timestamps = replace_callback
    engine._now_playing = "Please stop talking while I finish this sentence."
    engine._cb = EngineCallbacks(on_control_stop_result=original_results.append)
    assert not engine._poll_keywords(
        pcm[:3_200],
        control_authority=authority,
    )
    spotter.keyword = "stop"
    second_authority = replace(
        authority,
        capture_sequence=2,
        source_sample_start=3_200,
        source_sample_end=6_400,
    )

    assert engine._poll_keywords(
        pcm[3_200:],
        control_authority=second_authority,
    )

    assert [result.text for result in original_results] == ["stop"]
    assert replacement_results == []
    assert spotter.timestamp_calls == 1


@pytest.mark.parametrize(
    "stop_callback", [None, object()], ids=("missing", "noncallable")
)
def test_playback_kws_without_callable_typed_stop_callback_abstains_before_native(
    stop_callback,
) -> None:
    engine, spotter, authority = _authority_kws_engine("stop")
    engine._now_playing = "A sentence with no control words."
    tracker = AcousticTurnTracker("stream-a")
    tracker.rotate_capture(capture_epoch=2, capture_generation=5)
    before, _revision = tracker.partial(emitted_at=10.0)
    engine._acoustic_turn_tracker = tracker
    generic = []
    engine._cb = EngineCallbacks(
        on_command_result=generic.append,
        on_control_stop_result=stop_callback,
    )

    def unexpected_claim(*_args, **_kwargs):
        raise AssertionError("missing STOP callback must not claim playback")

    engine._claim_playback_kws_if_current = unexpected_claim

    assert not engine._poll_keywords(
        np.zeros(1600, dtype="float32"),
        control_authority=authority,
    )

    after = tracker.current(emitted_at=11.0)
    assert after is not None
    assert after.spans[-1].key == before.spans[-1].key
    assert tracker.active
    assert generic == []
    assert spotter.resets == 0
    assert spotter.token_calls == 0
    assert engine._playback_kws_claim is None
    engine._enqueue_play("new speech remains admissible", None)
    assert engine._play_q.get_nowait()[0] == "new speech remains admissible"


@pytest.mark.parametrize(
    "phrase_index,own_speech",
    [
        (3, "Please be quiet while I explain."),
        (0, "Please be quiet while I explain."),
        (4, "I need you to hold on for a moment."),
    ],
    ids=("exact-be-quiet", "collapsed-stop-sibling", "collapsed-wait-sibling"),
)
def test_playback_kws_collapsed_alias_ambiguity_abstains(
    phrase_index,
    own_speech,
) -> None:
    keyword = KWS_STOP_PHRASES[phrase_index].result_label
    engine, _spotter, authority = _authority_kws_engine(
        keyword,
        phrase_index=phrase_index,
    )
    engine._now_playing = own_speech
    stop_results = []
    engine._cb = EngineCallbacks(on_control_stop_result=stop_results.append)

    assert not engine._poll_keywords(
        np.zeros(1600, dtype="float32"),
        control_authority=authority,
    )

    assert stop_results == []
    assert engine._kws_own_echo_suppressions == 1
    assert _spotter.timestamp_calls == 0


def test_playback_kws_token_label_mismatch_abstains() -> None:
    engine, _spotter, authority = _authority_kws_engine(
        "stop",
        phrase_index=4,
    )
    engine._now_playing = "A sentence with no control words."
    stop_results = []
    engine._cb = EngineCallbacks(on_control_stop_result=stop_results.append)

    assert not engine._poll_keywords(
        np.zeros(1600, dtype="float32"),
        control_authority=authority,
    )
    assert stop_results == []


@pytest.mark.parametrize("native_label", ["STOP", "STOP!!!", " stop", "wait "])
def test_playback_kws_requires_raw_exact_native_label(native_label) -> None:
    phrase_index = 4 if native_label.startswith("wait") else 0
    engine, _spotter, authority = _authority_kws_engine(
        native_label,
        phrase_index=phrase_index,
    )
    engine._now_playing = "A sentence with no control words."
    generic = []
    stop_results = []
    engine._cb = EngineCallbacks(
        on_command_result=generic.append,
        on_control_stop_result=stop_results.append,
    )

    assert not engine._poll_keywords(
        np.zeros(1600, dtype="float32"),
        control_authority=authority,
    )

    assert generic == []
    assert stop_results == []
    assert engine._playback_kws_claim is None


def test_playback_kws_without_exact_phrase_binding_abstains() -> None:
    engine, spotter, authority = _authority_kws_engine("stop")
    engine._kws_phrase_bindings = ()
    engine._now_playing = "A sentence with no control words."
    stop_results = []
    engine._cb = EngineCallbacks(on_control_stop_result=stop_results.append)

    assert not engine._poll_keywords(
        np.zeros(1600, dtype="float32"),
        control_authority=authority,
    )

    assert spotter.token_calls == 0
    assert stop_results == []
    assert engine._playback_kws_claim is None


@pytest.mark.parametrize(
    "stale_case",
    [
        "capture_epoch",
        "capture_generation",
        "media_generation",
        "source_generation",
        "source_device",
        "authority_source_generation",
        "authority_source_device",
        "os_echo_route_authority",
        "speak_generation",
        "playback_generation",
        "speaking_state",
    ],
)
def test_stale_kws_authority_cannot_callback_or_terminal_lineage(stale_case) -> None:
    holder = {}

    def make_stale() -> None:
        engine = holder["engine"]
        if stale_case == "capture_epoch":
            engine._capture_epoch += 1
        elif stale_case == "capture_generation":
            engine._capture_callback_context.capture_generation += 1
        elif stale_case == "media_generation":
            engine._capture_media_session.generation += 1
        elif stale_case == "source_generation":
            engine._stream_in.generation += 1
        elif stale_case == "source_device":
            engine._stream_in.actual_device = "mic-b"
        elif stale_case == "authority_source_generation":
            engine._capture_authority_source_generation += 1
        elif stale_case == "authority_source_device":
            engine._capture_authority_source_device = "mic-b"
        elif stale_case == "os_echo_route_authority":
            engine._os_echo_route_verified = True
        elif stale_case == "speak_generation":
            engine._speak_gen += 1
        elif stale_case == "playback_generation":
            engine._playback_generation += 1
        else:
            engine._speaking.clear()

    engine, spotter, authority = _authority_kws_engine(
        "stop",
        before_tokens=make_stale,
    )
    holder["engine"] = engine
    engine._now_playing = "A sentence with no control words."
    tracker = AcousticTurnTracker("stream-a")
    tracker.rotate_capture(capture_epoch=2, capture_generation=5)
    before, _revision = tracker.partial(emitted_at=10.0)
    engine._acoustic_turn_tracker = tracker
    generic = []
    stop_results = []
    engine._cb = EngineCallbacks(
        on_command_result=generic.append,
        on_control_stop_result=stop_results.append,
    )

    assert not engine._poll_keywords(
        np.zeros(1600, dtype="float32"),
        control_authority=authority,
    )

    after = tracker.current(emitted_at=11.0)
    assert after is not None
    assert after.spans[-1].key == before.spans[-1].key
    assert tracker.active
    assert generic == []
    assert stop_results == []
    assert spotter.resets == 1
    assert engine._playback_kws_claim is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("capture_epoch", True),
        ("capture_generation", np.int64(5)),
        ("source_generation", True),
        ("authority_source_generation", np.int64(3)),
        ("os_echo_route_verified", 0),
        ("speaking", 1),
        ("speak_generation", np.int64(7)),
        ("playback_generation", True),
    ],
)
def test_forged_coercible_kws_authority_abstains_without_terminal(
    field,
    value,
) -> None:
    engine, _spotter, authority = _authority_kws_engine("stop")
    authority = replace(authority, **{field: value})
    engine._now_playing = "A sentence with no control words."
    tracker = AcousticTurnTracker("stream-a")
    tracker.rotate_capture(capture_epoch=2, capture_generation=5)
    before, _revision = tracker.partial(emitted_at=10.0)
    engine._acoustic_turn_tracker = tracker
    stop_results = []
    engine._cb = EngineCallbacks(on_control_stop_result=stop_results.append)

    assert not engine._poll_keywords(
        np.zeros(1600, dtype="float32"),
        control_authority=authority,
    )

    after = tracker.current(emitted_at=11.0)
    assert after is not None
    assert after.spans[-1].key == before.spans[-1].key
    assert tracker.active
    assert stop_results == []
    assert engine._playback_kws_claim is None


def test_kws_authority_snapshot_does_not_coerce_forged_capture_types() -> None:
    engine, _spotter, _authority = _authority_kws_engine("stop")
    captured = CapturedBlock(
        pcm=np.zeros(1, dtype="float32"),
        sample_rate_hz=16_000,
        sequence=1,
        captured_at=1.0,
        capture_epoch=True,
        source_generation=np.int64(3),
        capture_generation=np.int64(5),
        source_device="mic-a",
    )
    context = CaptureBlockContext(
        authority_source_generation=np.int64(3),
        authority_source_device="mic-a",
        os_echo_route_verified=1,
        speaking=1,
        speak_generation=np.int64(7),
        playback_generation=np.int64(11),
    )

    authority = engine._kws_control_authority(
        captured,
        context,
        route=_KwsFeedRoute.IDLE,
    )

    assert authority.capture_epoch is True
    assert type(authority.source_generation) is np.int64
    assert type(authority.capture_generation) is np.int64
    assert type(authority.authority_source_generation) is np.int64
    assert authority.os_echo_route_verified == 1
    assert type(authority.os_echo_route_verified) is int
    assert authority.speaking == 1
    assert type(authority.speaking) is int
    assert type(authority.speak_generation) is np.int64
    assert type(authority.playback_generation) is np.int64
    assert not engine._kws_control_authority_is_current(authority)


@pytest.mark.parametrize("route_case", ["virtual-route", "os-echo-route"])
def test_revoked_playback_route_cannot_callback_or_terminal_lineage(route_case) -> None:
    holder = {}

    def revoke_route() -> None:
        engine = holder["engine"]
        if route_case == "virtual-route":
            engine._virtual_route_failure_in_progress = True
        else:
            engine._os_echo_route_verified = False

    engine, _spotter, authority = _authority_kws_engine(
        "stop",
        os_echo_route_verified=True,
        before_tokens=revoke_route,
    )
    holder["engine"] = engine
    engine._now_playing = "A sentence with no control words."
    tracker = AcousticTurnTracker("stream-a")
    tracker.rotate_capture(capture_epoch=2, capture_generation=5)
    before, _revision = tracker.partial(emitted_at=10.0)
    engine._acoustic_turn_tracker = tracker
    stop_results = []
    engine._cb = EngineCallbacks(on_control_stop_result=stop_results.append)

    assert not engine._poll_keywords(
        np.zeros(1600, dtype="float32"),
        guard_private_route=route_case == "virtual-route",
        require_os_echo_route=route_case == "os-echo-route",
        os_echo_route_verified=True,
        control_authority=authority,
    )

    after = tracker.current(emitted_at=11.0)
    assert after is not None
    assert after.spans[-1].key == before.spans[-1].key
    assert tracker.active
    assert stop_results == []
    assert engine._playback_kws_claim is None


@pytest.mark.parametrize("initially_speaking", [False, True], ids=("idle", "active"))
def test_kws_authority_rejects_playback_state_aba(initially_speaking) -> None:
    holder = {}

    def cycle_playback_state() -> None:
        engine = holder["engine"]
        if initially_speaking:
            engine.stop_speaking()
            assert engine._begin_playback_run_if_current(engine._speak_gen) is False
        else:
            assert engine._begin_playback_run_if_current(engine._speak_gen) is False
            engine.stop_speaking()
        assert engine._speaking.is_set() is initially_speaking

    keyword = "stop" if initially_speaking else "command mode"
    engine, _spotter, authority = _authority_kws_engine(
        keyword,
        speaking=initially_speaking,
        before_result=cycle_playback_state,
    )
    holder["engine"] = engine
    engine._now_playing = "A sentence with no control words."
    generic = []
    stop_results = []
    engine._cb = EngineCallbacks(
        on_command_result=generic.append,
        on_control_stop_result=stop_results.append,
    )

    assert not engine._poll_keywords(
        np.zeros(1600, dtype="float32"),
        control_authority=authority,
    )

    assert generic == []
    assert stop_results == []
    assert engine._playback_kws_claim is None


def test_capture_callback_admission_failure_does_not_terminalize_or_leak_claim() -> (
    None
):
    engine, _spotter, authority = _authority_kws_engine("stop")
    engine._now_playing = "A sentence with no control words."
    tracker = AcousticTurnTracker("stream-a")
    tracker.rotate_capture(capture_epoch=2, capture_generation=5)
    before, _revision = tracker.partial(emitted_at=10.0)
    engine._acoustic_turn_tracker = tracker
    stop_results = []
    engine._cb = EngineCallbacks(on_control_stop_result=stop_results.append)
    claim_current = engine._claim_playback_kws_if_current

    def claim_then_stop_capture(control_authority, *, stop_only=None):
        claim = claim_current(control_authority, stop_only=stop_only)
        assert claim is not None
        engine._capture_epoch += 1
        return claim

    engine._claim_playback_kws_if_current = claim_then_stop_capture

    assert not engine._poll_keywords(
        np.zeros(1600, dtype="float32"),
        control_authority=authority,
    )

    after = tracker.current(emitted_at=11.0)
    assert after is not None
    assert after.spans[-1].key == before.spans[-1].key
    assert tracker.active
    assert stop_results == []
    assert engine._playback_kws_claim is None


def test_kws_callback_exception_releases_playback_claim() -> None:
    class _CallbackFailure(RuntimeError):
        pass

    engine, _spotter, authority = _authority_kws_engine("stop")
    engine._now_playing = "A sentence with no control words."

    def fail_callback(_result):
        assert engine._playback_kws_claim is not None
        raise _CallbackFailure("typed STOP callback failed")

    engine._cb = EngineCallbacks(on_control_stop_result=fail_callback)

    with pytest.raises(_CallbackFailure, match="typed STOP callback failed"):
        engine._poll_keywords(
            np.zeros(1600, dtype="float32"),
            control_authority=authority,
        )

    assert engine._playback_kws_claim is None


def test_typed_kws_hit_closes_source_acoustic_turn() -> None:
    engine = SherpaOnnxEngine(SherpaConfig())
    engine._kws = _FixedKws("stop")
    engine._kws_stream = _NullKwsStream()
    tracker = AcousticTurnTracker("stream-a")
    tracker.rotate_capture(capture_epoch=2, capture_generation=3)
    engine._acoustic_turn_tracker = tracker
    commands = []
    engine._cb = EngineCallbacks(on_command_result=commands.append)

    consumed = engine._poll_keywords(np.zeros(1600, dtype="float32"))

    assert consumed is True
    assert len(commands) == 1
    assert commands[0].revision == 0
    assert commands[0].acoustic is not None
    assert not tracker.active


def test_legacy_kws_callback_still_closes_a_typed_partial_turn() -> None:
    engine = SherpaOnnxEngine(SherpaConfig())
    engine._kws = _FixedKws("stop")
    engine._kws_stream = _NullKwsStream()
    tracker = AcousticTurnTracker("stream-a")
    tracker.rotate_capture(capture_epoch=2, capture_generation=3)
    engine._acoustic_turn_tracker = tracker
    first_lineage, first_revision = tracker.partial(emitted_at=10.0)
    commands = []
    engine._cb = EngineCallbacks(
        on_command=commands.append,
        on_partial_result=lambda _result: None,
    )

    consumed = engine._poll_keywords(np.zeros(1600, dtype="float32"))
    next_lineage, next_revision = tracker.partial(emitted_at=11.0)

    assert consumed is True
    assert commands == ["stop"]
    assert first_revision == 0
    assert next_revision == 0
    assert first_lineage.spans[0].key != next_lineage.spans[0].key
    assert next_lineage.spans[0].turn_index == 2


def _bounded_kws_engine(spotter: _BoundedKws) -> SherpaOnnxEngine:
    engine = SherpaOnnxEngine(SherpaConfig())
    engine._kws = spotter
    engine._kws_stream = spotter.create_stream()
    return engine


def test_kws_exactly_sixty_four_decodes_runs_final_probe_and_result_path() -> None:
    spotter = _BoundedKws(ready_decodes=64)
    engine = _bounded_kws_engine(spotter)

    assert not engine._poll_keywords(np.zeros(1600, dtype="float32"))

    assert spotter.accept_calls == 1
    assert spotter.ready_calls == 65
    assert spotter.decode_calls == 64
    assert spotter.result_calls == 1
    assert spotter.reset_calls == 0
    assert spotter.create_calls == 1


def test_kws_decode_exhaustion_abstains_before_result_and_preserves_lineage() -> None:
    spotter = _BoundedKws(ready_decodes=None, keyword="stop")
    engine = _bounded_kws_engine(spotter)
    old_stream = engine._kws_stream
    tracker = AcousticTurnTracker("stream-a")
    tracker.rotate_capture(capture_epoch=2, capture_generation=3)
    before, revision = tracker.partial(emitted_at=10.0)
    engine._acoustic_turn_tracker = tracker
    commands: list[str] = []
    engine._cb = EngineCallbacks(on_command=commands.append)

    assert not engine._poll_keywords(np.zeros(1600, dtype="float32"))

    after = tracker.current(emitted_at=11.0)
    assert after is not None
    assert revision == 0
    assert before.spans[0].key == after.spans[0].key
    assert tracker.active
    assert commands == []
    assert spotter.accept_calls == 1
    assert spotter.ready_calls == 65
    assert spotter.decode_calls == 64
    assert spotter.result_calls == 0
    assert spotter.reset_calls == 1
    assert spotter.create_calls == 2
    assert engine._kws_stream is spotter.streams[-1]
    assert engine._kws_stream is not old_stream


@pytest.mark.parametrize("native_result", [None, False, 0, (), []])
def test_kws_falsey_non_string_result_is_native_fault_and_preserves_lineage(
    native_result,
) -> None:
    spotter = _BoundedKws(ready_decodes=0, keyword=native_result)
    engine = _bounded_kws_engine(spotter)
    old_stream = engine._kws_stream
    tracker = AcousticTurnTracker("stream-a")
    tracker.rotate_capture(capture_epoch=2, capture_generation=3)
    before, _revision = tracker.partial(emitted_at=10.0)
    engine._acoustic_turn_tracker = tracker
    commands = []
    engine._cb = EngineCallbacks(on_command_result=commands.append)

    assert not engine._poll_keywords(np.zeros(1600, dtype="float32"))

    after = tracker.current(emitted_at=11.0)
    assert after is not None
    assert after.spans[-1].key == before.spans[-1].key
    assert tracker.active
    assert commands == []
    assert spotter.result_calls == 1
    assert spotter.reset_calls == 1
    assert spotter.create_calls == 2
    assert engine._kws_stream is not old_stream


@pytest.mark.parametrize("failure_phase", ["accept", "is_ready", "decode", "result"])
def test_kws_native_fault_abstains_recreates_stream_and_preserves_lineage(
    failure_phase,
) -> None:
    spotter = _BoundedKws(
        ready_decodes=1,
        keyword="stop",
        failure_phase=failure_phase,
    )
    engine = _bounded_kws_engine(spotter)
    old_stream = engine._kws_stream
    tracker = AcousticTurnTracker("stream-a")
    tracker.rotate_capture(capture_epoch=2, capture_generation=3)
    before, _revision = tracker.partial(emitted_at=10.0)
    engine._acoustic_turn_tracker = tracker
    commands: list[str] = []
    engine._cb = EngineCallbacks(on_command=commands.append)

    assert not engine._poll_keywords(np.zeros(1600, dtype="float32"))

    after = tracker.current(emitted_at=11.0)
    assert after is not None
    assert before.spans[0].key == after.spans[0].key
    assert tracker.active
    assert commands == []
    assert spotter.reset_calls == 1
    assert spotter.create_calls == 2
    assert engine._kws_stream is not old_stream


def test_kws_recovery_recreates_even_when_best_effort_reset_fails() -> None:
    class _ResetFailureKws(_BoundedKws):
        def reset_stream(self, stream):
            super().reset_stream(stream)
            raise RuntimeError("injected KWS reset failure")

    spotter = _ResetFailureKws(ready_decodes=None)
    engine = _bounded_kws_engine(spotter)
    old_stream = engine._kws_stream

    assert not engine._poll_keywords(np.zeros(1600, dtype="float32"))

    assert spotter.reset_calls == 1
    assert spotter.create_calls == 2
    assert engine._kws_stream is spotter.streams[-1]
    assert engine._kws_stream is not old_stream


def test_kws_recovery_sets_stream_none_when_recreation_fails() -> None:
    class _RecreateFailureKws(_BoundedKws):
        def create_stream(self):
            if self.create_calls:
                self.create_calls += 1
                raise RuntimeError("injected KWS recreate failure")
            return super().create_stream()

    spotter = _RecreateFailureKws(ready_decodes=None)
    engine = _bounded_kws_engine(spotter)

    assert not engine._poll_keywords(np.zeros(1600, dtype="float32"))

    assert spotter.reset_calls == 1
    assert spotter.create_calls == 2
    assert engine._kws_stream is None


@pytest.mark.parametrize(
    "config_override, samples, detected_at",
    [
        ({}, np.zeros(0, dtype="float32"), None),
        (
            {
                "sample_rate": 100,
                "block_sec": 0.1,
                "media_pcm_queue_ms": 300.0,
            },
            np.zeros(31, dtype="float32"),
            None,
        ),
        ({"block_sec": float("nan")}, np.zeros(1, dtype="float32"), None),
        ({"sample_rate": True}, np.zeros(1, dtype="float32"), None),
        ({"block_sec": True}, np.zeros(1, dtype="float32"), None),
        ({"media_pcm_queue_ms": True}, np.zeros(1, dtype="float32"), None),
        ({}, np.zeros(1, dtype="float32"), -1.0),
        ({}, np.zeros(1, dtype="float32"), float("nan")),
        ({}, np.zeros(1, dtype="float32"), True),
        pytest.param(
            {},
            np.zeros(1, dtype="float32"),
            10**10_000,
            id="giant-detected-at",
        ),
    ],
)
def test_kws_rejects_empty_oversize_or_invalid_timing_before_native_work(
    config_override,
    samples,
    detected_at,
) -> None:
    spotter = _BoundedKws(ready_decodes=0, keyword="stop")
    engine = SherpaOnnxEngine(SherpaConfig(**config_override))
    engine._kws = spotter
    initial_stream = spotter.create_stream()
    engine._kws_stream = initial_stream
    commands: list[str] = []
    engine._cb = EngineCallbacks(on_command=commands.append)

    with pytest.raises(_CaptureBlockRejected):
        engine._poll_keywords(samples, detected_at=detected_at)

    assert commands == []
    assert spotter.accept_calls == 0
    assert spotter.ready_calls == 0
    assert spotter.decode_calls == 0
    assert spotter.result_calls == 0
    assert spotter.reset_calls == 0
    assert spotter.create_calls == 1
    assert engine._kws_stream is initial_stream


def test_kws_callback_exception_is_not_reclassified_as_native_failure() -> None:
    class _CallbackFailure(RuntimeError):
        pass

    spotter = _BoundedKws(ready_decodes=0, keyword="stop")
    engine = _bounded_kws_engine(spotter)

    def fail_callback(_keyword):
        raise _CallbackFailure("command callback failed")

    engine._cb = EngineCallbacks(on_command=fail_callback)

    with pytest.raises(_CallbackFailure, match="command callback failed"):
        engine._poll_keywords(np.zeros(1600, dtype="float32"))

    assert spotter.result_calls == 1
    assert spotter.reset_calls == 1  # the ordinary hit reset, not recovery
    assert spotter.create_calls == 2


def _one_block_kws_capture() -> tuple[SherpaOnnxEngine, object]:
    class _Stream:
        def __init__(self) -> None:
            self.blocks = []

        def accept_waveform(self, _sample_rate, samples) -> None:
            self.blocks.append(np.asarray(samples).copy())

    class _Recognizer:
        def __init__(self) -> None:
            self.stream = _Stream()
            self.reset_calls = 0

        def create_stream(self, **_kwargs):
            return self.stream

        def is_ready(self, _stream) -> bool:
            return False

        def get_result(self, _stream) -> str:
            return ""

        def reset(self, _stream) -> None:
            self.reset_calls += 1

        def is_endpoint(self, _stream) -> bool:
            return False

    class _OneBlockInput:
        generation = 0
        actual_device = None

        def __init__(self, engine) -> None:
            self.engine = engine

        def read(self, frames):
            self.engine._running.clear()
            return np.ones(frames, dtype="float32"), False

    engine = SherpaOnnxEngine(
        SherpaConfig(
            barge_in_enabled=False,
            aec_enabled=False,
        )
    )
    recognizer = _Recognizer()
    engine._recognizer = recognizer
    engine._capture_sr = engine.config.sample_rate
    engine._stream_in = _OneBlockInput(engine)
    # Inline fixture equivalent of production's capture-domain authority bind.
    engine._capture_authority_source_generation = 0
    engine._capture_authority_source_device = None
    return engine, recognizer


def test_capture_kws_consumed_frame_never_reaches_normal_asr() -> None:
    engine, recognizer = _one_block_kws_capture()
    engine._poll_keywords = lambda _samples, **_kwargs: True

    engine._running.set()
    engine._capture_loop()

    assert recognizer.reset_calls == 1
    assert recognizer.stream.blocks == []


def test_legacy_kws_exhaustion_recreates_stream_then_preserves_same_block_asr():
    engine, recognizer = _one_block_kws_capture()
    spotter = _BoundedKws(ready_decodes=None, keyword="stop")
    engine._kws = spotter
    engine._kws_stream = spotter.create_stream()
    commands: list[str] = []
    engine._cb = EngineCallbacks(on_command=commands.append)

    engine._running.set()
    engine._capture_loop()

    assert commands == []
    assert spotter.ready_calls == 65
    assert spotter.decode_calls == 64
    assert spotter.result_calls == 0
    assert spotter.reset_calls == 2
    assert spotter.create_calls == 2
    assert len(recognizer.stream.blocks) == 1
    assert np.array_equal(
        recognizer.stream.blocks[0],
        np.ones(1600, dtype="float32"),
    )


def test_oversize_kws_capture_block_is_rejected_without_normal_asr_laundering():
    engine, recognizer = _one_block_kws_capture()
    spotter = _BoundedKws(ready_decodes=0, keyword="stop")
    engine._kws = spotter
    engine._kws_stream = spotter.create_stream()

    class _OversizeInput:
        generation = 0

        def read(self, _frames):
            engine._running.clear()
            return np.ones(4801, dtype="float32"), False

    engine._stream_in = _OversizeInput()
    engine._running.set()
    engine._capture_loop()

    assert spotter.accept_calls == 0
    assert spotter.ready_calls == 0
    assert spotter.result_calls == 0
    assert recognizer.stream.blocks == []


def test_stale_no_hit_skips_asr_and_forces_fresh_native_time_origin() -> None:
    engine, recognizer = _one_block_kws_capture()

    class _TwoBlockInput:
        generation = 0
        actual_device = None

        def __init__(self) -> None:
            self.reads = 0

        def read(self, _frames):
            self.reads += 1
            if self.reads == 1:
                return np.full(1_600, 0.25, dtype="float32"), False
            self.generation = 0
            engine._running.clear()
            return np.full(1_600, 2.0, dtype="float32"), False

    class _StaleStream:
        def __init__(self) -> None:
            self.low = False
            self.high = False
            self.accepted: list[float] = []

        def accept_waveform(self, _sample_rate, samples) -> None:
            mean = float(np.mean(samples))
            self.accepted.append(mean)
            self.low |= mean <= 1.0
            self.high |= mean > 1.0

    class _StalingNoHitSpotter:
        def __init__(self) -> None:
            self.streams: list[_StaleStream] = []
            self.result_calls = 0
            self.resets = 0

        def create_stream(self) -> _StaleStream:
            stream = _StaleStream()
            self.streams.append(stream)
            return stream

        def is_ready(self, _stream) -> bool:
            return False

        def decode_stream(self, _stream) -> None:
            pass

        def get_result(self, stream: _StaleStream) -> str:
            self.result_calls += 1
            if self.result_calls == 1:
                engine._stream_in.generation = 1
            return "stop" if stream.low and stream.high else ""

        def reset_stream(self, stream: _StaleStream) -> None:
            self.resets += 1
            stream.low = False
            stream.high = False

    source = _TwoBlockInput()
    spotter = _StalingNoHitSpotter()
    engine._stream_in = source
    engine._kws = spotter
    engine._kws_stream = spotter.create_stream()
    dirty_marks = []
    mark_dirty = engine._mark_keyword_continuity_dirty

    def record_dirty() -> None:
        mark_dirty()
        dirty_marks.append(
            (
                engine._kws_continuity_dirty,
                engine._kws_pcm_ledger.scope,
                tuple(engine._kws_pcm_ledger.chunks),
            )
        )

    engine._mark_keyword_continuity_dirty = record_dirty
    commands = []
    engine._cb = EngineCallbacks(on_command=commands.append)

    engine._running.set()
    engine._capture_loop()

    assert commands == []
    assert spotter.result_calls == 2
    assert len(spotter.streams) == 2
    assert spotter.streams[0].accepted == [0.25]
    assert spotter.streams[1].accepted == [2.0]
    assert spotter.resets == 2  # stale-prefix rotation, then capture retirement
    assert dirty_marks == [(True, None, ()), (True, None, ())]
    assert len(recognizer.stream.blocks) == 1
    np.testing.assert_array_equal(
        recognizer.stream.blocks[0],
        np.full(1_600, 2.0, dtype="float32"),
    )


def test_kws_ledger_and_speaker_regions_use_exact_post_frontend_pcm() -> None:
    engine, recognizer = _one_block_kws_capture()
    raw = np.linspace(0.20, 0.40, 3_200, dtype="float32")

    class _OneExactBlock:
        generation = 0
        actual_device = None

        def read(self, _frames):
            engine._running.clear()
            return raw.copy(), False

    class _Scale:
        def process(self, samples):
            return np.asarray(samples, dtype="float32") * np.float32(2.0)

    class _Offset:
        def process_16k(self, samples):
            return np.asarray(samples, dtype="float32") + np.float32(0.125)

    scored: list[np.ndarray] = []

    def embed(samples, _sample_rate):
        scored.append(np.asarray(samples, dtype="float32").copy())
        return [1.0, 0.0]

    gate = SpeakerGate(threshold=0.5, embed_fn=embed)
    gate.enroll_embedding([1.0, 0.0])
    engine._replace_speaker_gate(gate)
    assert engine._warm_speaker_gate()
    scored.clear()
    stream = _NullKwsStream()
    spotter = _TokenKws(
        "stop",
        _PLAYBACK_KWS_TOKENS[0],
        timestamps=[0.00, 0.04, 0.08, 0.12],
    )
    engine._stream_in = _OneExactBlock()
    engine._input_agc = _Scale()
    engine._denoiser = _Offset()
    engine._kws = spotter
    engine._kws_stream = stream
    engine._kws_phrase_bindings = _PLAYBACK_KWS_BINDINGS
    engine._os_echo_route_verified = True
    engine._speaking.set()
    engine._now_playing = "I will stop after this sentence."
    stop_results = []
    engine._cb = EngineCallbacks(on_control_stop_result=stop_results.append)
    expected = raw * np.float32(2.0) + np.float32(0.125)

    engine._running.set()
    engine._capture_loop()

    assert recognizer.stream.blocks == []
    assert len(stream.blocks) == 1
    np.testing.assert_array_equal(stream.blocks[0], expected)
    assert len(scored) == 1
    np.testing.assert_array_equal(scored[0], expected[:2_560])
    assert [result.text for result in stop_results] == ["stop"]
    assert spotter.timestamp_calls == 1


def test_capture_kws_terminal_closes_open_confirm_window() -> None:
    engine, recognizer = _one_block_kws_capture()
    engine._capture_authority_source_generation = 0
    engine._capture_authority_source_device = None
    engine._os_echo_route_verified = True
    engine._kws = _TokenKws("stop", _PLAYBACK_KWS_TOKENS[0])
    engine._kws_stream = _NullKwsStream()
    engine._kws_phrase_bindings = _PLAYBACK_KWS_BINDINGS
    stop_results = []
    barges: list[str] = []
    engine._cb = EngineCallbacks(
        on_control_stop_result=stop_results.append,
        on_barge_in=lambda: barges.append("barge"),
    )
    engine._speaking.set()
    engine._begin_barge_confirm(recognizer, recognizer.stream, time.monotonic())
    engine._confirm_base_text = "old partial"
    engine._confirm_handoff_stream_live = True
    engine._confirm_handoff_pending = object()
    engine._confirm_primary_pcm.append(np.ones(3, dtype="float32"))
    engine._confirm_alternate_pcm.append(np.ones(3, dtype="float32"))
    engine._confirm_audio_started_at = 1.0
    engine._confirm_audio_ended_at = 2.0
    engine._confirm_echo_obs.append((1.0, 2.0, 0.0))
    assert engine._barge_confirm_active()
    assert engine._duck_gain < 1.0

    engine._running.set()
    engine._capture_loop()

    assert recognizer.reset_calls == 1
    assert recognizer.stream.blocks == []
    assert engine._kws.resets == 2
    assert [result.text for result in stop_results] == ["stop"]
    assert barges == []
    assert not engine._barge_confirm_active()
    assert engine._duck_gain == 1.0
    assert engine._confirm_base_text == ""
    assert engine._confirm_speak_generation is None
    assert engine._confirm_playback_generation is None
    assert not engine._confirm_handoff_stream_live
    assert engine._confirm_handoff_pending is None
    assert engine._confirm_primary_pcm == []
    assert engine._confirm_alternate_pcm == []
    assert engine._confirm_audio_started_at is None
    assert engine._confirm_audio_ended_at is None
    assert engine._confirm_echo_obs == []


def test_failed_virtual_playback_proof_never_grants_authority():
    binder = _Binder()
    binder.capture_bound = True
    binder.fail_duplex = True
    engine = _engine(binder)
    engine._word_cut_route_verified = True
    engine._os_echo_route_verified = True

    with pytest.raises(RuntimeError, match="duplex route"):
        engine._prepare_virtual_playback_route()

    assert engine._word_cut_route_verified is False
    assert engine._os_echo_route_verified is False


def test_production_engine_has_no_virtual_route_authority():
    engine = SherpaOnnxEngine(SherpaConfig())

    engine._bind_virtual_capture_route()
    engine._prepare_virtual_playback_route()
    engine._authorize_virtual_playback_route()

    assert engine._virtual_audio_binder is None
    assert engine._word_cut_route_verified is False


def test_route_loss_hard_flushes_and_closes_before_publishing_failure():
    engine = _engine(_Binder())
    engine._running.set()
    engine._word_cut_route_verified = True
    engine._os_echo_route_verified = True

    class _FIFO:
        def __init__(self):
            self.calls = []

        def interrupt_tags(self, status, fade_samples=0):
            self.calls.append(("interrupt", status, fade_samples))
            return ()

        def flush(self, fade_samples=0):
            self.calls.append(("flush", fade_samples))

    class _Output:
        def __init__(self):
            self.calls = []

        def abort(self):
            assert engine.virtual_route_failure == ""
            self.calls.append("abort")

        def close(self):
            assert engine.virtual_route_failure == ""
            self.calls.append("close")

    fifo = _FIFO()
    output = _Output()
    engine._fifo = fifo
    engine._out_stream = output

    engine._fail_virtual_route("target drift")

    assert [call[0] for call in fifo.calls] == ["interrupt", "flush"]
    assert all(call[-1] == 0 for call in fifo.calls)
    assert output.calls == ["abort", "close"]
    assert engine._fifo is None and engine._out_stream is None
    assert engine._running.is_set() is False
    assert engine._word_cut_route_verified is False
    assert engine._os_echo_route_verified is False
    assert engine.virtual_route_failure == "target drift"


def test_concurrent_route_failure_cannot_publish_before_blocked_abort_finishes():
    engine = _engine(_Binder())
    engine._running.set()
    abort_entered = threading.Event()
    release_abort = threading.Event()

    class _Output:
        def __init__(self):
            self.abort_calls = 0

        def abort(self):
            self.abort_calls += 1
            abort_entered.set()
            assert release_abort.wait(2.0)

        def close(self):
            pass

    output = _Output()
    engine._out_stream = output
    first = threading.Thread(
        target=engine._fail_virtual_route,
        args=("first failure",),
    )
    first.start()
    assert abort_entered.wait(1.0)

    engine._fail_virtual_route("second failure")
    assert engine.virtual_route_failure == ""
    assert output.abort_calls == 1

    release_abort.set()
    first.join(timeout=1.0)
    assert first.is_alive() is False
    assert engine.virtual_route_failure == "first failure"
    assert output.abort_calls == 1


def test_unexpected_private_capture_exit_publishes_fatal_but_stop_does_not():
    engine = _engine(_Binder())
    engine._running.set()

    assert engine._fail_virtual_capture_if_unexpected("capture failed") is True
    assert engine.virtual_route_failure == "capture failed"
    assert engine._running.is_set() is False

    stopping = _engine(_Binder())
    stopping._running.set()
    stopping._capture_stopping.set()
    assert stopping._fail_virtual_capture_if_unexpected("normal stop") is False
    assert stopping.virtual_route_failure == ""


def test_route_loss_fences_an_inflight_capture_block_before_callbacks():
    read_entered = threading.Event()
    release_read = threading.Event()
    commands = []

    class _Stream:
        def accept_waveform(self, _sample_rate, _samples):
            pass

    class _Recognizer:
        def create_stream(self, **_kwargs):
            return _Stream()

        def is_ready(self, _stream):
            return False

        def get_result(self, _stream):
            return ""

        def reset(self, _stream):
            pass

        def is_endpoint(self, _stream):
            return False

    class _BlockingInput:
        generation = 0

        def read(self, frames):
            read_entered.set()
            assert release_read.wait(2.0)
            return np.ones(frames, dtype="float32"), False

    engine = _engine(_Binder())
    engine._recognizer = _Recognizer()
    engine._capture_sr = engine.config.sample_rate
    engine._stream_in = _BlockingInput()
    engine._poll_keywords = lambda _samples, **_kwargs: commands.append("stop")
    engine._running.set()
    worker = threading.Thread(target=engine._capture_loop)
    worker.start()
    assert read_entered.wait(1.0)

    engine._fail_virtual_route("proof lost during read")
    release_read.set()
    worker.join(timeout=1.0)

    assert worker.is_alive() is False
    assert commands == []
    assert engine._capture_stopping.is_set()
    assert engine.virtual_route_failure == "proof lost during read"


def test_private_capture_without_recognizer_fails_parent_instead_of_hanging():
    engine = _engine(_Binder())
    engine._recognizer = None
    engine._running.set()

    engine._capture_loop()

    assert engine._running.is_set() is False
    assert engine.virtual_route_failure == "virtual capture loop stopped unexpectedly"


def test_restart_refuses_retained_virtual_route_monitor_before_audio_open():
    engine = _engine(_Binder())
    engine._virtual_route_thread = types.SimpleNamespace(is_alive=lambda: True)

    with pytest.raises(RuntimeError, match="virtual-route monitor"):
        engine.start(EngineCallbacks())
