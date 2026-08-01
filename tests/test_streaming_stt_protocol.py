from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.streaming_stt.protocol import (
    MAX_HYPOTHESIS_CHARS,
    MAX_LEGACY_STREAM_SAMPLES,
    MAX_LINE_BYTES,
    MAX_NATIVE_STREAM_SAMPLES,
    MAX_STREAM_CHUNK_SAMPLES,
    NATIVE_ENDPOINT_PROTOCOL_VERSION,
    PROTOCOL_VERSION,
    FinalEvent,
    NativeFinalEvent,
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


@pytest.mark.parametrize(
    ("version", "accepted_tail", "rejected_tail"),
    [
        (PROTOCOL_VERSION, 16_000, 16_001),
        (NATIVE_ENDPOINT_PROTOCOL_VERSION, 48_000, 48_001),
    ],
)
def test_request_tail_bound_is_exact_for_each_protocol_version(
    tmp_path,
    version,
    accepted_tail,
    rejected_tail,
):
    payload = _request(tmp_path).as_dict()
    payload["v"] = version
    payload["stream"]["tail_padding_samples"] = accepted_tail

    parsed = parse_request(encode_message(payload))

    assert isinstance(parsed, TranscribeRequest)
    assert parsed.protocol_version == version
    assert parsed.stream.tail_padding_samples == accepted_tail

    payload["stream"]["tail_padding_samples"] = rejected_tail
    with pytest.raises(ProtocolError):
        parse_request(encode_message(payload))


@pytest.mark.parametrize(
    ("version", "accepted_samples", "rejected_samples"),
    [
        (
            PROTOCOL_VERSION,
            MAX_LEGACY_STREAM_SAMPLES,
            MAX_LEGACY_STREAM_SAMPLES + 1,
        ),
        (
            NATIVE_ENDPOINT_PROTOCOL_VERSION,
            MAX_NATIVE_STREAM_SAMPLES,
            MAX_NATIVE_STREAM_SAMPLES + 1,
        ),
    ],
)
def test_partial_progress_bound_is_exact_for_each_protocol_version(
    version,
    accepted_samples,
    rejected_samples,
):
    payload = {
        "v": version,
        "id": "case-1",
        "type": "partial",
        "seq": 0,
        "text": "x",
        "samples_seen": accepted_samples,
        "elapsed_ms": 1.0,
        "decode_ms": 1.0,
    }

    assert parse_response(encode_message(payload)).samples_seen == accepted_samples

    payload["samples_seen"] = rejected_samples
    with pytest.raises(ProtocolError):
        parse_response(encode_message(payload))


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


def _native_final_payload() -> dict[str, object]:
    return {
        "v": NATIVE_ENDPOINT_PROTOCOL_VERSION,
        "id": "case-1",
        "type": "native_final",
        "seq": 1,
        "text": "stop now",
        "samples_seen": 1920,
        "elapsed_ms": 150.0,
        "finalization_ms": 4.0,
        "compute_ms": 30.0,
        "audio_seconds": 0.1,
        "chunks": 2,
        "deadline_misses": 0,
        "max_backlog_ms": 2.0,
        "model_padding_samples": 640,
        "resources": {"rss_mb": 50.0, "threads": 2, "vram_mb": 512.0},
        "source_samples": 1600,
        "source_samples_consumed": 1600,
        "declared_tail_samples": 48_000,
        "tail_samples_consumed": 320,
        "endpoint_reason": "eou",
        "native_endpoint": True,
        "endpoint_probability": 0.75,
        "endpoint_sample": 2560,
        "endpoint_latency_ms": 60.0,
        "authoritative": True,
    }


def test_native_final_response_is_a_distinct_exact_v3_contract():
    payload = _native_final_payload()

    result = parse_response(json.dumps(payload).encode())

    assert isinstance(result, NativeFinalEvent)
    assert not isinstance(result, FinalEvent)
    assert result.source_samples_consumed == 1600
    assert result.tail_samples_consumed == 320
    assert result.endpoint_reason == "eou"
    assert result.endpoint_probability == 0.75
    assert result.protocol_version == NATIVE_ENDPOINT_PROTOCOL_VERSION
    assert "stop now" not in repr(result)


def test_complete_source_eob_is_native_but_never_authoritative():
    payload = _native_final_payload()
    payload.update(
        {
            "endpoint_reason": "eob",
            "endpoint_latency_ms": None,
            "authoritative": False,
        }
    )

    result = parse_response(encode_message(payload))

    assert isinstance(result, NativeFinalEvent)
    assert result.endpoint_reason == "eob"
    assert result.native_endpoint is True
    assert result.source_samples_consumed == result.source_samples
    assert result.endpoint_latency_ms is None
    assert result.authoritative is False


@pytest.mark.parametrize(
    "updates",
    [
        {"endpoint_reason": "eob", "authoritative": True},
        {"endpoint_reason": "eob", "endpoint_latency_ms": 60.0},
    ],
)
def test_complete_source_eob_rejects_response_authority_and_latency(updates):
    payload = _native_final_payload()
    payload.update(
        {
            "endpoint_reason": "eob",
            "endpoint_latency_ms": None,
            "authoritative": False,
        }
    )
    payload.update(updates)

    with pytest.raises(ProtocolError):
        parse_response(encode_message(payload))


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("version", PROTOCOL_VERSION),
        ("type", "final"),
        ("endpoint_reason", "silence"),
        ("native_endpoint", 1),
        ("endpoint_probability", 1.01),
        ("source_samples_consumed", True),
        ("declared_tail_samples", 48_001),
    ],
)
def test_native_final_rejects_wrong_version_type_and_unbounded_fields(
    mutation,
    value,
):
    payload = _native_final_payload()
    if mutation == "version":
        payload["v"] = value
    elif mutation == "type":
        payload["type"] = value
    else:
        payload[mutation] = value

    with pytest.raises(ProtocolError):
        parse_response(json.dumps(payload).encode())


@pytest.mark.parametrize(
    "updates",
    [
        {"samples_seen": 1919},
        {"source_samples_consumed": 1599, "tail_samples_consumed": 321},
        {"endpoint_sample": 1919},
        {"endpoint_probability": None},
        {"endpoint_latency_ms": None},
        {"authoritative": False},
        {
            "native_endpoint": False,
            "endpoint_reason": "eou",
        },
        {
            "native_endpoint": False,
            "endpoint_reason": "tail_exhausted",
            "endpoint_probability": None,
            "endpoint_sample": None,
            "endpoint_latency_ms": None,
            "authoritative": False,
        },
    ],
)
def test_native_final_rejects_relationally_inconsistent_evidence(updates):
    payload = _native_final_payload()
    payload.update(updates)

    with pytest.raises(ProtocolError):
        parse_response(json.dumps(payload).encode())


def test_v2_final_rejects_native_fields_instead_of_relaxing_legacy_schema():
    payload = _native_final_payload()
    payload["v"] = PROTOCOL_VERSION
    payload["type"] = "final"

    with pytest.raises(ProtocolError):
        parse_response(json.dumps(payload).encode())


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
