from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.streaming_stt.protocol import (
    MAX_HYPOTHESIS_CHARS,
    MAX_LINE_BYTES,
    MAX_STREAM_CHUNK_SAMPLES,
    PROTOCOL_VERSION,
    FinalEvent,
    PcmInput,
    ProtocolError,
    StreamConfig,
    TranscribeRequest,
    encode_message,
    parse_request,
    parse_response,
)


def _request(tmp_path: Path) -> TranscribeRequest:
    return TranscribeRequest(
        request_id="case-0000-r0",
        pcm=PcmInput(
            path=(tmp_path / "case.f32le").resolve(),
            sha256="a" * 64,
            samples=1600,
        ),
        stream=StreamConfig(
            chunk_samples=160,
            pace="burst",
            partial_interval_ms=200,
            tail_padding_samples=80,
        ),
    )


def test_transcribe_request_round_trips_the_exact_v2_contract(tmp_path):
    request = _request(tmp_path)

    parsed = parse_request(encode_message(request.as_dict()))

    assert parsed == request
    assert parsed.pcm.size_bytes == 6400


def test_request_allows_la13_native_stride_but_keeps_a_bounded_chunk_limit(
    tmp_path,
):
    request = _request(tmp_path)
    payload = request.as_dict()
    payload["stream"]["chunk_samples"] = 17_920

    parsed = parse_request(encode_message(payload))

    assert isinstance(parsed, TranscribeRequest)
    assert parsed.stream.chunk_samples == 17_920
    payload["stream"]["chunk_samples"] = MAX_STREAM_CHUNK_SAMPLES + 1
    with pytest.raises(ProtocolError):
        parse_request(encode_message(payload))


@pytest.mark.parametrize(
    "raw",
    [
        b'{"v":2,"v":2,"id":"x","op":"shutdown"}\n',
        b'{"v":NaN,"id":"x","op":"shutdown"}\n',
        b'{"v":1,"id":"x","op":"shutdown"}\n',
        b'{"v":2,"id":"X","op":"shutdown"}\n',
        b'{"v":2,"id":"x","op":"shutdown","extra":1}\n',
    ],
)
def test_request_rejects_duplicates_nonfinite_version_and_unknown_fields(raw):
    with pytest.raises(ProtocolError):
        parse_request(raw)


def test_protocol_rejects_oversized_line():
    with pytest.raises(ProtocolError):
        parse_request(b"{" + b"x" * MAX_LINE_BYTES + b"}")


def test_response_rejects_oversized_hypothesis():
    payload = {
        "v": PROTOCOL_VERSION,
        "id": "case-1",
        "type": "partial",
        "seq": 0,
        "text": "x" * (MAX_HYPOTHESIS_CHARS + 1),
        "samples_seen": 1,
        "elapsed_ms": 1.0,
        "decode_ms": 1.0,
    }

    with pytest.raises(ProtocolError):
        parse_response(json.dumps(payload).encode())


def test_final_response_is_strict_typed_and_text_is_repr_private():
    private = "SENTINEL_PRIVATE_HYPOTHESIS"
    payload = {
        "v": PROTOCOL_VERSION,
        "id": "case-1",
        "type": "final",
        "seq": 1,
        "text": private,
        "samples_seen": 1600,
        "elapsed_ms": 200.0,
        "finalization_ms": 50.0,
        "compute_ms": 30.0,
        "audio_seconds": 0.1,
        "chunks": 10,
        "deadline_misses": 0,
        "max_backlog_ms": 2.0,
        "model_padding_samples": 37,
        "resources": {"rss_mb": 50.0, "threads": 2, "vram_mb": None},
    }

    result = parse_response(json.dumps(payload).encode())

    assert isinstance(result, FinalEvent)
    assert result.text == private
    assert result.model_padding_samples == 37
    assert private not in repr(result)


@pytest.mark.parametrize("padding", [-1, 32_001, True])
def test_final_response_rejects_invalid_model_padding(padding):
    payload = {
        "v": PROTOCOL_VERSION,
        "id": "case-1",
        "type": "final",
        "seq": 0,
        "text": "x",
        "samples_seen": 1,
        "elapsed_ms": 1.0,
        "finalization_ms": 1.0,
        "compute_ms": 1.0,
        "audio_seconds": 0.1,
        "chunks": 1,
        "deadline_misses": 0,
        "max_backlog_ms": 0.0,
        "model_padding_samples": padding,
        "resources": {"rss_mb": 1.0, "threads": 1, "vram_mb": None},
    }

    with pytest.raises(ProtocolError):
        parse_response(json.dumps(payload).encode())


def test_ready_requires_a_valid_source_bundle_digest():
    payload = {
        "v": PROTOCOL_VERSION,
        "type": "ready",
        "model_id": "fake-stream-v1",
        "manifest_sha256": "a" * 64,
        "source_bundle_sha256": "b" * 64,
        "adapter": "fake-json-v1",
        "model_load_ms": 1.0,
        "resources": {"rss_mb": 50.0, "threads": 2, "vram_mb": None},
        "runtime": {"python": "3.12", "platform": "linux"},
    }

    ready = parse_response(json.dumps(payload).encode())
    assert ready.source_bundle_sha256 == "b" * 64

    payload["source_bundle_sha256"] = "not-a-digest"
    with pytest.raises(ProtocolError):
        parse_response(json.dumps(payload).encode())


def test_message_encoder_rejects_nonfinite_and_bounds_output():
    with pytest.raises(ProtocolError):
        encode_message({"value": float("inf")})


def test_response_rejects_numeric_overflow_as_a_protocol_error():
    payload = {
        "v": PROTOCOL_VERSION,
        "id": "case-1",
        "type": "partial",
        "seq": 0,
        "text": "x",
        "samples_seen": 1,
        "elapsed_ms": 10**400,
        "decode_ms": 1.0,
    }

    with pytest.raises(ProtocolError):
        parse_response(json.dumps(payload).encode())


@pytest.mark.parametrize(
    ("event_type", "field"),
    [
        ("partial", "elapsed_ms"),
        ("partial", "decode_ms"),
        ("final", "compute_ms"),
        ("final", "max_backlog_ms"),
    ],
)
def test_response_rejects_finite_but_unrealistic_numeric_values(event_type, field):
    if event_type == "partial":
        payload = {
            "v": PROTOCOL_VERSION,
            "id": "case-1",
            "type": "partial",
            "seq": 0,
            "text": "x",
            "samples_seen": 1,
            "elapsed_ms": 1.0,
            "decode_ms": 1.0,
        }
    else:
        payload = {
            "v": PROTOCOL_VERSION,
            "id": "case-1",
            "type": "final",
            "seq": 0,
            "text": "x",
            "samples_seen": 1,
            "elapsed_ms": 1.0,
            "finalization_ms": 1.0,
            "compute_ms": 1.0,
            "audio_seconds": 0.1,
            "chunks": 1,
            "deadline_misses": 0,
            "max_backlog_ms": 1.0,
            "model_padding_samples": 0,
            "resources": {"rss_mb": 1.0, "threads": 1, "vram_mb": None},
        }
    payload[field] = 1e308

    with pytest.raises(ProtocolError):
        parse_response(json.dumps(payload).encode())


def test_response_rejects_unrealistic_worker_resource_values():
    payload = {
        "v": PROTOCOL_VERSION,
        "type": "ready",
        "model_id": "fake-stream-v1",
        "manifest_sha256": "a" * 64,
        "source_bundle_sha256": "b" * 64,
        "adapter": "fake-json-v1",
        "model_load_ms": 1.0,
        "resources": {"rss_mb": 1e308, "threads": 1, "vram_mb": None},
        "runtime": {"python": "3.12", "platform": "linux"},
    }

    with pytest.raises(ProtocolError):
        parse_response(json.dumps(payload).encode())
