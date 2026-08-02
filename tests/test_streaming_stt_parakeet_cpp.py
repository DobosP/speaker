from __future__ import annotations

from array import array
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import struct

import pytest

import tools.streaming_stt.adapters.parakeet_cpp as adapter_module
from tools.streaming_stt.adapters.parakeet_cpp import (
    ParakeetCppAdapter,
    ParakeetCppAdapterError,
)
from tools.streaming_stt.manifest import ManifestError, ParakeetCppConfig
from tools.streaming_stt.protocol import (
    MAX_PARAKEET_CPP_OBSERVED_EVENTS,
    PARAKEET_CPP_ENDPOINT_PROTOCOL_VERSION,
    PcmInput,
    ResourceUsage,
    StreamConfig,
    TranscribeRequest,
)


class _Clock:
    def __init__(self) -> None:
        self.value = 10.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        self.value += 0.001
        return self.value

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.value += seconds


class _ArtifactVerifier:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, str, int, int]] = []

    def __call__(
        self,
        path: Path,
        *,
        expected_sha256: str,
        expected_size_bytes: int,
        maximum_size_bytes: int,
    ) -> object:
        payload = path.read_bytes()
        metadata = path.stat()
        self.calls.append(
            (path, expected_sha256, expected_size_bytes, maximum_size_bytes)
        )
        return (
            path,
            hashlib.sha256(payload).hexdigest(),
            len(payload),
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )


class _Api:
    def __init__(
        self,
        feed_documents: list[bytes],
        *,
        finalize_document: bytes | None = None,
        bridge_abi: int = 1,
        upstream_abi: int = 6,
        actual_device: str = "cpu",
        fail_feed: bool = False,
        noisy: bool = False,
    ) -> None:
        self.feed_documents = list(feed_documents)
        self.finalize_document = finalize_document or _document()
        self.bridge_abi = bridge_abi
        self.upstream_abi = upstream_abi
        self.selected_device = actual_device
        self.fail_feed = fail_feed
        self.noisy = noisy
        self.calls: list[tuple[object, ...]] = []
        self.chunks: list[tuple[float, ...]] = []
        self.active = False
        self.closed = False
        self.handle = object()

    def _noise(self) -> None:
        if self.noisy:
            os.write(1, b"private-native-stdout\n")
            os.write(2, b"private-native-stderr\n")

    def bridge_abi_version(self) -> int:
        self._noise()
        return self.bridge_abi

    def upstream_abi_version(self) -> int:
        self._noise()
        return self.upstream_abi

    def open(self, model_path: Path, requested_device: str, num_threads: int) -> object:
        self._noise()
        self.calls.append(("open", model_path, requested_device, num_threads))
        return self.handle

    def actual_device(self, handle: object) -> str:
        assert handle is self.handle
        self._noise()
        return self.selected_device

    def begin(self, handle: object) -> None:
        assert handle is self.handle and not self.closed and not self.active
        self._noise()
        self.calls.append(("begin",))
        self.active = True

    def feed_json(self, handle: object, samples: array[float]) -> bytes:
        assert handle is self.handle and self.active
        self._noise()
        self.calls.append(("feed", len(samples)))
        self.chunks.append(tuple(samples))
        if self.fail_feed or not self.feed_documents:
            raise RuntimeError("private native detail")
        return self.feed_documents.pop(0)

    def finalize_json(self, handle: object) -> bytes:
        assert handle is self.handle and self.active
        self._noise()
        self.calls.append(("finalize",))
        self.active = False
        return self.finalize_document

    def close(self, handle: object) -> None:
        assert handle is self.handle and not self.closed
        self._noise()
        self.calls.append(("close",))
        self.active = False
        self.closed = True


def _document(
    text: str = "",
    *,
    events: tuple[tuple[str, int], ...] = (),
    words: list[dict[str, object]] | None = None,
    eou: int | None = None,
    eob: int | None = None,
    frame_sec: float = 0.08,
) -> bytes:
    value = {
        "text": text,
        "eou": (
            int(any(event_type == "eou" for event_type, _frame in events))
            if eou is None
            else eou
        ),
        "eob": (
            int(any(event_type == "eob" for event_type, _frame in events))
            if eob is None
            else eob
        ),
        "frame_sec": frame_sec,
        "events": [
            {"type": event_type, "frame": frame, "t": frame * frame_sec}
            for event_type, frame in events
        ],
        "words": [] if words is None else words,
    }
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _timed_document(
    events: tuple[tuple[str, int, float], ...],
) -> bytes:
    value = {
        "text": "",
        "eou": int(any(event_type == "eou" for event_type, _frame, _time in events)),
        "eob": int(any(event_type == "eob" for event_type, _frame, _time in events)),
        "frame_sec": 0.08,
        "events": [
            {"type": event_type, "frame": frame, "t": event_time}
            for event_type, frame, event_time in events
        ],
        "words": [],
    }
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    native = tmp_path / "native"
    model = tmp_path / "model"
    native.mkdir()
    model.mkdir()
    library = (native / "libparakeet.so").resolve()
    bridge = (native / "libspeaker_parakeet_bridge.so").resolve()
    gguf = (model / "realtime_eou_120m-v1-f16.gguf").resolve()
    library.write_bytes(b"libparakeet-fixture")
    bridge.write_bytes(b"bridge-fixture")
    gguf.write_bytes(b"model-fixture")
    return gguf, library, bridge


def _request(
    tmp_path: Path,
    *,
    source_samples: int = 2_560,
    tail_samples: int = 1_280,
    pace: str = "burst",
    chunk_samples: int = 1_280,
    protocol_version: int = PARAKEET_CPP_ENDPOINT_PROTOCOL_VERSION,
    sample_rate: int = 16_000,
    values: list[float] | None = None,
) -> TranscribeRequest:
    samples = values if values is not None else [0.25] * source_samples
    payload = b"".join(struct.pack("<f", value) for value in samples)
    pcm_path = (tmp_path / "case.f32le").resolve()
    pcm_path.write_bytes(payload)
    return TranscribeRequest(
        request_id="case-r0",
        pcm=PcmInput(
            path=pcm_path,
            sha256=hashlib.sha256(payload).hexdigest(),
            samples=source_samples,
            sample_rate=sample_rate,
        ),
        stream=StreamConfig(
            chunk_samples=chunk_samples,
            pace=pace,
            partial_interval_ms=80,
            tail_padding_samples=tail_samples,
        ),
        protocol_version=protocol_version,
    )


def _adapter(
    tmp_path: Path,
    api: _Api,
    *,
    clock: _Clock | None = None,
    verifier: _ArtifactVerifier | None = None,
) -> tuple[ParakeetCppAdapter, _ArtifactVerifier]:
    gguf, library, bridge = _paths(tmp_path)
    selected_verifier = verifier or _ArtifactVerifier()
    selected_clock = clock or _Clock()
    adapter = ParakeetCppAdapter(
        gguf,
        library,
        bridge,
        config=ParakeetCppConfig(),
        library_sha256="1" * 64,
        library_size_bytes=library.stat().st_size,
        bridge_sha256="2" * 64,
        bridge_size_bytes=bridge.stat().st_size,
        api=api,
        clock=selected_clock,
        sleeper=selected_clock.sleep,
        resource_reader=lambda: ResourceUsage(
            rss_mb=64.0,
            threads=1,
            vram_mb=None,
        ),
        artifact_verifier=selected_verifier,
    )
    return adapter, selected_verifier


def test_manifest_config_is_exact_and_adapter_constant_is_exported() -> None:
    config = ParakeetCppConfig()
    assert adapter_module.PARAKEET_CPP_ADAPTER == "parakeet-cpp-realtime-eou-v1"
    assert config.model_dtype == "F16"
    assert config.c_api_version == 6
    assert config.bridge_abi_version == 1
    assert config.native_chunk_samples == 1_280
    with pytest.raises(ManifestError):
        replace(config, requested_device="CUDA0")


def test_concatenates_deltas_and_complete_source_eou_is_authoritative(tmp_path: Path) -> None:
    api = _Api(
        [
            _document("find"),
            _document(" in my"),
            _document(" vault", events=(("eou", 3),)),
        ]
    )
    adapter, verifier = _adapter(tmp_path, api)
    partials = []
    case = adapter.transcribe(
        _request(tmp_path, source_samples=3_840, tail_samples=1_280),
        emit_partial=lambda seq, partial: partials.append((seq, partial)),
    )
    adapter.close()

    assert case.final == "find in my vault"
    assert [partial.text for _seq, partial in partials] == [
        "find",
        "find in my",
        "find in my vault",
    ]
    assert case.source_samples_offered == case.source_samples_consumed == 3_840
    assert case.tail_samples_offered == 1_280
    assert case.tail_samples_consumed == 0
    assert case.endpoint_reason == "eou"
    assert case.event_origin == "feed"
    assert case.endpoint_sample == 3_840
    assert case.encoder_frame == 3
    assert case.encoder_time_seconds == pytest.approx(0.24)
    assert case.endpoint_latency_ms == 0.0
    assert case.authoritative is True
    assert [
        (
            event.event_type,
            event.event_origin,
            event.observed_sample,
            event.encoder_frame,
            event.encoder_time_seconds,
        )
        for event in case.observed_events
    ] == [("eou", "feed", 3_840, 3, pytest.approx(0.24))]
    assert case.model_padding_samples == 0
    assert api.calls[-2:] == [("finalize",), ("close",)]
    assert len(verifier.calls) == 9


def test_marker_only_eou_retains_accumulated_text(tmp_path: Path) -> None:
    api = _Api(
        [
            _document("find in my vault"),
            _document("", events=(("eou", 2),)),
        ]
    )
    adapter, _verifier = _adapter(tmp_path, api)
    case = adapter.transcribe(_request(tmp_path), emit_partial=lambda *_: None)
    adapter.close()
    assert case.final == "find in my vault"
    assert case.authoritative is True


def test_complete_source_eou_stops_optional_tail_at_offered_chunk_boundary(
    tmp_path: Path,
) -> None:
    api = _Api(
        [
            _document("command"),
            _document(" complete", events=(("eou", 2),)),
        ]
    )
    adapter, _verifier = _adapter(tmp_path, api)
    case = adapter.transcribe(
        _request(tmp_path, source_samples=1_600, tail_samples=1_280),
        emit_partial=lambda *_: None,
    )
    adapter.close()

    assert case.final == "command complete"
    assert case.source_samples_consumed == case.source_samples_offered == 1_600
    assert case.tail_samples_consumed == 960
    assert case.endpoint_sample == 2_560
    assert case.endpoint_latency_ms == pytest.approx(60.0)
    assert case.observed_events[0].observed_sample == 2_560
    assert case.observed_events[0].observed_sample != case.source_samples_offered
    assert case.observed_events[0].observed_sample % 1_280 == 0
    assert [len(chunk) for chunk in api.chunks] == [1_280, 1_280]


def test_native_event_time_is_canonicalized_from_encoder_frame(
    tmp_path: Path,
) -> None:
    api = _Api([_timed_document((("eou", 1, 0.0799995),))])
    adapter, _verifier = _adapter(tmp_path, api)
    case = adapter.transcribe(
        _request(tmp_path, source_samples=1_280, tail_samples=0),
        emit_partial=lambda *_: None,
    )
    adapter.close()

    assert case.encoder_frame == 1
    assert case.encoder_time_seconds == pytest.approx(0.08)
    assert case.observed_events[0].encoder_time_seconds == pytest.approx(0.08)


def test_early_first_eou_freezes_text_and_blocks_later_eou_authority(
    tmp_path: Path,
) -> None:
    api = _Api(
        [
            _document("first", events=(("eou", 1),)),
            _document(" ack", events=(("eob", 2),)),
            _document(" later", events=(("eou", 3),)),
            _document(" tail", events=(("eob", 4),)),
        ],
        finalize_document=_document(" finalize", events=(("eou", 5),)),
    )
    adapter, _verifier = _adapter(tmp_path, api)
    partials = []
    case = adapter.transcribe(
        _request(tmp_path, source_samples=3_840, tail_samples=1_280),
        emit_partial=lambda _seq, partial: partials.append(partial),
    )
    adapter.close()

    assert case.final == "first"
    assert [partial.text for partial in partials] == ["first"]
    assert [call for call in api.calls if call[0] == "feed"] == [
        ("feed", 1_280),
        ("feed", 1_280),
        ("feed", 1_280),
        ("feed", 1_280),
    ]
    assert case.source_samples_consumed == case.source_samples_offered == 3_840
    assert case.tail_samples_consumed == case.tail_samples_offered == 1_280
    assert case.endpoint_reason == "tail_exhausted"
    assert case.event_origin == "none"
    assert case.endpoint_sample is None
    assert case.encoder_frame is None
    assert case.authoritative is False
    assert [event.event_type for event in case.observed_events] == [
        "eou",
        "eob",
        "eou",
        "eob",
        "eou",
    ]
    assert [event.observed_sample for event in case.observed_events] == [
        1_280,
        2_560,
        3_840,
        5_120,
        5_120,
    ]


def test_eob_never_freezes_and_later_complete_eou_is_authoritative(
    tmp_path: Path,
) -> None:
    api = _Api(
        [
            _document("ack", events=(("eob", 1),)),
            _document(" command", events=(("eou", 2),)),
        ]
    )
    adapter, _verifier = _adapter(tmp_path, api)
    partials = []
    case = adapter.transcribe(
        _request(tmp_path),
        emit_partial=lambda _seq, partial: partials.append(partial),
    )
    adapter.close()

    assert case.final == "ack command"
    assert [partial.text for partial in partials] == ["ack", "ack command"]
    assert case.source_samples_consumed == case.source_samples_offered == 2_560
    assert case.tail_samples_consumed == 0
    assert case.endpoint_reason == "eou"
    assert case.event_origin == "feed"
    assert case.endpoint_sample == 2_560
    assert case.encoder_frame == 2
    assert case.authoritative is True
    assert [event.event_type for event in case.observed_events] == ["eob", "eou"]


def test_only_early_eou_eob_and_finalize_events_still_exhaust_full_clip(
    tmp_path: Path,
) -> None:
    api = _Api(
        [
            _document("early", events=(("eou", 1),)),
            _document(" backchannel", events=(("eob", 2),)),
            _document(" tail"),
        ],
        finalize_document=_document(" done", events=(("eou", 3),)),
    )
    adapter, _verifier = _adapter(tmp_path, api)
    case = adapter.transcribe(
        _request(tmp_path, source_samples=2_560, tail_samples=1_280),
        emit_partial=lambda *_: None,
    )
    adapter.close()

    assert case.final == "early"
    assert case.source_samples_consumed == case.source_samples_offered == 2_560
    assert case.tail_samples_consumed == case.tail_samples_offered == 1_280
    assert case.endpoint_reason == "tail_exhausted"
    assert case.event_origin == "none"
    assert case.native_endpoint is False
    assert case.endpoint_sample is None
    assert case.encoder_frame is None
    assert case.encoder_time_seconds is None
    assert case.authoritative is False
    assert case.endpoint_latency_ms is None
    assert [
        (event.event_type, event.event_origin, event.observed_sample)
        for event in case.observed_events
    ] == [
        ("eou", "feed", 1_280),
        ("eob", "feed", 2_560),
        ("eou", "finalize", 3_840),
    ]


def test_finalize_event_is_observation_only_and_never_terminal(tmp_path: Path) -> None:
    api = _Api(
        [_document("hello"), _document(" world")],
        finalize_document=_document(" done", events=(("eou", 2),)),
    )
    adapter, _verifier = _adapter(tmp_path, api)
    case = adapter.transcribe(
        _request(tmp_path, source_samples=1_280, tail_samples=1_280),
        emit_partial=lambda *_: None,
    )
    adapter.close()
    assert case.final == "hello world done"
    assert case.endpoint_reason == "tail_exhausted"
    assert case.event_origin == "none"
    assert case.native_endpoint is False
    assert case.endpoint_sample is None
    assert case.encoder_frame is None
    assert case.encoder_time_seconds is None
    assert case.authoritative is False
    assert case.endpoint_latency_ms is None
    assert len(case.observed_events) == 1
    assert case.observed_events[0].event_type == "eou"
    assert case.observed_events[0].event_origin == "finalize"
    assert case.observed_events[0].observed_sample == 2_560


def test_encoder_time_beyond_feed_boundary_fails_but_finalize_tail_is_allowed(
    tmp_path: Path,
) -> None:
    feed_root = tmp_path / "feed"
    feed_root.mkdir()
    feed_api = _Api(
        [_document("early", events=(("eob", 1), ("eou", 3)))]
    )
    feed_adapter, _verifier = _adapter(feed_root, feed_api)
    with pytest.raises(ParakeetCppAdapterError):
        feed_adapter.transcribe(
            _request(feed_root, source_samples=1_280, tail_samples=0),
            emit_partial=lambda *_: None,
        )
    assert feed_api.closed is True

    finalize_root = tmp_path / "finalize"
    finalize_root.mkdir()
    finalize_api = _Api(
        [_document("complete"), _document("")],
        finalize_document=_document("", events=(("eou", 3),)),
    )
    finalize_adapter, _verifier = _adapter(finalize_root, finalize_api)
    case = finalize_adapter.transcribe(
        _request(
            finalize_root,
            source_samples=1_280,
            tail_samples=1_280,
        ),
        emit_partial=lambda *_: None,
    )
    finalize_adapter.close()
    assert case.source_samples_consumed == case.source_samples_offered == 1_280
    assert case.tail_samples_consumed == case.tail_samples_offered == 1_280
    assert case.encoder_time_seconds is None
    assert case.event_origin == "none"
    assert case.authoritative is False
    assert case.endpoint_reason == "tail_exhausted"
    assert case.observed_events[0].event_origin == "finalize"
    assert case.observed_events[0].encoder_time_seconds == pytest.approx(0.24)


def test_tail_exhaustion_consumes_short_final_chunk_without_model_padding(
    tmp_path: Path,
) -> None:
    api = _Api(
        [_document("all"), _document(" deltas"), _document(" remain")],
        finalize_document=_document(" final"),
    )
    adapter, _verifier = _adapter(tmp_path, api)
    case = adapter.transcribe(
        _request(tmp_path, source_samples=1_600, tail_samples=1_000),
        emit_partial=lambda *_: None,
    )
    adapter.close()
    assert [len(chunk) for chunk in api.chunks] == [1_280, 1_280, 40]
    assert case.final == "all deltas remain final"
    assert case.endpoint_reason == "tail_exhausted"
    assert case.event_origin == "none"
    assert case.native_endpoint is False
    assert case.source_samples_consumed == 1_600
    assert case.tail_samples_consumed == 1_000
    assert case.endpoint_sample is None
    assert case.model_padding_samples == 0


def test_complete_document_eou_freezes_before_nonempty_finalize_delta(
    tmp_path: Path,
) -> None:
    api = _Api(
        [_document("ack", events=(("eob", 1), ("eou", 1)))],
        finalize_document=_document(" ignored", events=(("eou", 2),)),
    )
    adapter, _verifier = _adapter(tmp_path, api)
    case = adapter.transcribe(
        _request(tmp_path, source_samples=1_280),
        emit_partial=lambda *_: None,
    )
    adapter.close()
    assert case.final == "ack"
    assert case.endpoint_reason == "eou"
    assert case.event_origin == "feed"
    assert case.authoritative is True
    assert case.endpoint_sample == 1_280
    assert case.encoder_frame == 1
    assert [
        (event.event_type, event.event_origin, event.observed_sample)
        for event in case.observed_events
    ] == [
        ("eob", "feed", 1_280),
        ("eou", "feed", 1_280),
        ("eou", "finalize", 1_280),
    ]


def test_zero_confidence_word_is_accepted(tmp_path: Path) -> None:
    words = [{"w": "quiet", "start": 0.0, "end": 0.08, "conf": 0.0}]
    api = _Api([_document("quiet", events=(("eou", 1),), words=words)])
    adapter, _verifier = _adapter(tmp_path, api)
    case = adapter.transcribe(
        _request(tmp_path, source_samples=1_280), emit_partial=lambda *_: None
    )
    adapter.close()
    assert case.final == "quiet"


@pytest.mark.parametrize(
    "bad_document",
    [
        _document(events=(("eou", 1),), eou=0),
        _document(events=(("eou", 1),), frame_sec=0.04),
        b'{"text":"secret","text":"changed","eou":0,"eob":0,'
        b'"frame_sec":0.08,"events":[],"words":[]}',
        b'{"text":"secret","eou":0,"eob":0,"frame_sec":NaN,'
        b'"events":[],"words":[]}',
    ],
)
def test_malformed_or_conflicting_json_fails_detail_free_and_poisons(
    tmp_path: Path, bad_document: bytes
) -> None:
    api = _Api([bad_document])
    adapter, _verifier = _adapter(tmp_path, api)
    with pytest.raises(ParakeetCppAdapterError) as captured:
        adapter.transcribe(
            _request(tmp_path, source_samples=1_280),
            emit_partial=lambda *_: None,
        )
    assert str(captured.value) == ""
    assert api.closed is True
    with pytest.raises(ParakeetCppAdapterError):
        adapter.transcribe(
            _request(tmp_path, source_samples=1_280),
            emit_partial=lambda *_: None,
        )


def test_nonmonotonic_or_duplicate_events_fail_closed(tmp_path: Path) -> None:
    api = _Api(
        [_document("a"), _document("b", events=(("eou", 2),))],
        finalize_document=_document("", events=(("eou", 2),)),
    )
    adapter, _verifier = _adapter(tmp_path, api)
    with pytest.raises(ParakeetCppAdapterError):
        adapter.transcribe(_request(tmp_path), emit_partial=lambda *_: None)
    assert api.closed is True


def test_same_type_frame_near_time_duplicates_fail_before_authority(
    tmp_path: Path,
) -> None:
    api = _Api(
        [
            _timed_document(
                (
                    ("eou", 1, 0.0799995),
                    ("eou", 1, 0.08),
                )
            )
        ]
    )
    adapter, _verifier = _adapter(tmp_path, api)

    with pytest.raises(ParakeetCppAdapterError):
        adapter.transcribe(
            _request(tmp_path, source_samples=1_280, tail_samples=0),
            emit_partial=lambda *_: None,
        )
    assert api.closed is True


def test_observed_event_limit_counts_all_continued_feed_events(
    tmp_path: Path,
) -> None:
    event_count = MAX_PARAKEET_CPP_OBSERVED_EVENTS + 1
    api = _Api(
        [
            _document(events=(("eob", frame),))
            for frame in range(1, event_count + 1)
        ]
    )
    adapter, _verifier = _adapter(tmp_path, api)
    with pytest.raises(ParakeetCppAdapterError):
        adapter.transcribe(
            _request(
                tmp_path,
                source_samples=event_count * 1_280,
                tail_samples=0,
            ),
            emit_partial=lambda *_: None,
        )
    assert api.closed is True


def test_discarded_post_eou_deltas_still_count_toward_total_text_byte_budget(
    tmp_path: Path,
) -> None:
    api = _Api(
        [
            _document("é" * 2_048, events=(("eou", 1),)),
            _document("x"),
        ]
    )
    adapter, _verifier = _adapter(tmp_path, api)

    with pytest.raises(ParakeetCppAdapterError):
        adapter.transcribe(
            _request(tmp_path, source_samples=2_560, tail_samples=0),
            emit_partial=lambda *_: None,
        )
    assert api.closed is True


def test_duplicate_and_empty_deltas_do_not_emit_duplicate_partials(tmp_path: Path) -> None:
    api = _Api(
        [
            _document("hello"),
            _document(""),
            _document("", events=(("eou", 3),)),
        ]
    )
    adapter, _verifier = _adapter(tmp_path, api)
    partials = []
    case = adapter.transcribe(
        _request(tmp_path, source_samples=3_840),
        emit_partial=lambda _seq, partial: partials.append(partial),
    )
    adapter.close()
    assert case.final == "hello"
    assert [partial.text for partial in partials] == ["hello"]


def test_realtime_pacing_is_deterministic_and_reports_backlog(tmp_path: Path) -> None:
    clock = _Clock()
    api = _Api([_document(), _document(events=(("eou", 2),))])
    adapter, _verifier = _adapter(tmp_path, api, clock=clock)
    case = adapter.transcribe(
        _request(
            tmp_path,
            source_samples=2_560,
            tail_samples=0,
            pace="realtime",
        ),
        emit_partial=lambda *_: None,
    )
    adapter.close()
    assert clock.sleeps
    assert case.deadline_misses >= 0
    assert case.max_backlog_ms >= 0.0


@pytest.mark.parametrize(
    "request_kwargs",
    [
        {"protocol_version": 3},
        {"chunk_samples": 640},
        {"sample_rate": 8_000},
    ],
)
def test_wrong_protocol_or_pcm_geometry_is_rejected_before_native_begin(
    tmp_path: Path, request_kwargs: dict[str, object]
) -> None:
    api = _Api([_document()])
    adapter, _verifier = _adapter(tmp_path, api)
    with pytest.raises(ParakeetCppAdapterError):
        adapter.transcribe(
            _request(tmp_path, **request_kwargs),
            emit_partial=lambda *_: None,
        )
    adapter.close()
    assert ("begin",) not in api.calls


def test_nonfinite_pcm_is_rejected_before_native_begin(tmp_path: Path) -> None:
    api = _Api([_document()])
    adapter, _verifier = _adapter(tmp_path, api)
    with pytest.raises(ParakeetCppAdapterError):
        adapter.transcribe(
            _request(
                tmp_path,
                source_samples=2,
                tail_samples=0,
                values=[0.0, float("nan")],
            ),
            emit_partial=lambda *_: None,
        )
    adapter.close()
    assert ("begin",) not in api.calls


def test_native_failure_is_detail_free_and_closes_model(tmp_path: Path) -> None:
    api = _Api([_document()], fail_feed=True)
    adapter, _verifier = _adapter(tmp_path, api)
    with pytest.raises(ParakeetCppAdapterError) as captured:
        adapter.transcribe(
            _request(tmp_path, source_samples=1_280),
            emit_partial=lambda *_: None,
        )
    assert str(captured.value) == ""
    assert api.calls[-1] == ("close",)


def test_native_output_is_suppressed_and_repr_hides_text_and_paths(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    secret = "private transcript phrase"
    api = _Api([_document(secret, events=(("eou", 1),))], noisy=True)
    adapter, _verifier = _adapter(tmp_path, api)
    case = adapter.transcribe(
        _request(tmp_path, source_samples=1_280),
        emit_partial=lambda *_: None,
    )
    adapter.close()
    captured = capfd.readouterr()
    assert captured.out == captured.err == ""
    assert secret not in repr(case)
    assert str(tmp_path) not in repr(adapter.ready)


def test_abi_or_device_mismatch_fails_closed_after_open(tmp_path: Path) -> None:
    for index, api in enumerate(
        (
            _Api([], bridge_abi=2),
            _Api([], upstream_abi=5),
            _Api([], actual_device="CUDA0"),
        )
    ):
        case_root = tmp_path / str(index)
        case_root.mkdir()
        with pytest.raises(ParakeetCppAdapterError):
            _adapter(case_root, api)
        if api.bridge_abi == 1 and api.upstream_abi == 6:
            assert api.closed is True
        else:
            assert api.closed is False


def test_close_rechecks_all_artifacts_and_detects_mutation(tmp_path: Path) -> None:
    api = _Api([])
    adapter, verifier = _adapter(tmp_path, api)
    adapter.library_path.write_bytes(b"changed-library")
    with pytest.raises(ParakeetCppAdapterError):
        adapter.close()
    assert api.closed is True
    assert len(verifier.calls) == 9


def test_real_artifact_verifier_rejects_symlink_hardlink_and_bad_digest(
    tmp_path: Path,
) -> None:
    source = (tmp_path / "source.so").resolve()
    source.write_bytes(b"native")
    digest = hashlib.sha256(b"native").hexdigest()
    identity = adapter_module._verify_artifact(
        source,
        expected_sha256=digest,
        expected_size_bytes=6,
        maximum_size_bytes=16,
    )
    assert identity.sha256 == digest

    symlink = (tmp_path / "symlink.so").resolve(strict=False)
    symlink.symlink_to(source)
    with pytest.raises(ParakeetCppAdapterError):
        adapter_module._verify_artifact(
            symlink,
            expected_sha256=digest,
            expected_size_bytes=6,
            maximum_size_bytes=16,
        )
    hardlink = (tmp_path / "hardlink.so").resolve(strict=False)
    os.link(source, hardlink)
    with pytest.raises(ParakeetCppAdapterError):
        adapter_module._verify_artifact(
            source,
            expected_sha256=digest,
            expected_size_bytes=6,
            maximum_size_bytes=16,
        )
    hardlink.unlink()
    with pytest.raises(ParakeetCppAdapterError):
        adapter_module._verify_artifact(
            source,
            expected_sha256="0" * 64,
            expected_size_bytes=6,
            maximum_size_bytes=16,
        )


def test_bridge_source_binds_exact_abi_device_lifecycle_and_private_errors() -> None:
    source = (
        Path(adapter_module.__file__).parents[1]
        / "native"
        / "parakeet_cpp_bridge.cpp"
    ).read_text(encoding="utf-8")
    assert "kBridgeAbiVersion = 1" in source
    assert "kRequiredParakeetCapiVersion = 6" in source
    assert source.index("pk::set_num_threads(num_threads)") < source.index(
        "parakeet_capi_load(model_path)"
    )
    assert "backend.device_name()" in source
    assert "equal_ascii_casefold" in source
    assert source.index("parakeet_capi_free(handle->context)") < source.index(
        "pk::shutdown_backend()"
    )
    assert "parakeet_capi_last_error" not in source
    assert "handle->lifecycle != Lifecycle::kStreaming" in source
