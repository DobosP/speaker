from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.streaming_stt.protocol import (
    MAX_HYPOTHESIS_CHARS,
    MAX_LEGACY_STREAM_SAMPLES,
    MAX_LINE_BYTES,
    MAX_NATIVE_STREAM_SAMPLES,
    MAX_PARAKEET_CPP_OBSERVED_EVENTS,
    MAX_STREAM_CHUNK_SAMPLES,
    MOBILE_ZIPFORMER_ENDPOINT_PROTOCOL_VERSION,
    NATIVE_ENDPOINT_PROTOCOL_VERSION,
    PARAKEET_CPP_ENDPOINT_PROTOCOL_VERSION,
    PROTOCOL_VERSION,
    FinalEvent,
    NativeFinalEvent,
    ParakeetCppFinalEvent,
    ParakeetCppObservedEvent,
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
        (PARAKEET_CPP_ENDPOINT_PROTOCOL_VERSION, 48_000, 48_001),
        (MOBILE_ZIPFORMER_ENDPOINT_PROTOCOL_VERSION, 48_000, 48_001),
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
        (
            PARAKEET_CPP_ENDPOINT_PROTOCOL_VERSION,
            MAX_NATIVE_STREAM_SAMPLES,
            MAX_NATIVE_STREAM_SAMPLES + 1,
        ),
        (
            MOBILE_ZIPFORMER_ENDPOINT_PROTOCOL_VERSION,
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


def _parakeet_cpp_final_payload() -> dict[str, object]:
    return {
        "v": PARAKEET_CPP_ENDPOINT_PROTOCOL_VERSION,
        "id": "case-1",
        "type": "parakeet_cpp_final",
        "seq": 1,
        "text": "stop now",
        "samples_seen": 16_640,
        "elapsed_ms": 150.0,
        "finalization_ms": 4.0,
        "compute_ms": 30.0,
        "audio_seconds": 1.0,
        "chunks": 13,
        "deadline_misses": 0,
        "max_backlog_ms": 2.0,
        "model_padding_samples": 0,
        "resources": {"rss_mb": 50.0, "threads": 2, "vram_mb": 512.0},
        "source_samples_offered": 16_000,
        "source_samples_consumed": 16_000,
        "tail_samples_offered": 48_000,
        "tail_samples_consumed": 640,
        "observed_events": [
            {
                "event_type": "eou",
                "event_origin": "feed",
                "observed_sample": 16_640,
                "encoder_frame": 13,
                "encoder_time_seconds": 1.04,
            }
        ],
        "endpoint_reason": "eou",
        "native_endpoint": True,
        "encoder_frame": 13,
        "encoder_time_seconds": 1.04,
        "event_origin": "feed",
        "endpoint_sample": 16_640,
        "endpoint_latency_ms": 40.0,
        "authoritative": True,
    }


def _tail_exhausted_parakeet_cpp_payload() -> dict[str, object]:
    payload = _parakeet_cpp_final_payload()
    payload.update(
        {
            "samples_seen": 64_000,
            "chunks": 50,
            "tail_samples_consumed": 48_000,
            "observed_events": [],
            "endpoint_reason": "tail_exhausted",
            "native_endpoint": False,
            "encoder_frame": None,
            "encoder_time_seconds": None,
            "event_origin": "none",
            "endpoint_sample": None,
            "endpoint_latency_ms": None,
            "authoritative": False,
        }
    )
    return payload


def test_parakeet_cpp_final_is_a_distinct_exact_v4_contract():
    payload = _parakeet_cpp_final_payload()

    result = parse_response(encode_message(payload))

    assert isinstance(result, ParakeetCppFinalEvent)
    assert not isinstance(result, (FinalEvent, NativeFinalEvent))
    assert result.source_samples_offered == 16_000
    assert result.source_samples_consumed == 16_000
    assert result.tail_samples_offered == 48_000
    assert result.tail_samples_consumed == 640
    assert result.observed_events == (
        ParakeetCppObservedEvent("eou", "feed", 16_640, 13, 1.04),
    )
    assert result.endpoint_reason == "eou"
    assert result.encoder_frame == 13
    assert result.encoder_time_seconds == 1.04
    assert result.event_origin == "feed"
    assert result.authoritative is True
    assert result.protocol_version == PARAKEET_CPP_ENDPOINT_PROTOCOL_VERSION
    assert "stop now" not in repr(result)


def test_early_eou_eob_and_finalize_events_are_telemetry_not_terminals():
    payload = _tail_exhausted_parakeet_cpp_payload()
    payload["observed_events"] = [
        {
            "event_type": "eou",
            "event_origin": "feed",
            "observed_sample": 1_280,
            "encoder_frame": 1,
            "encoder_time_seconds": 0.08,
        },
        {
            "event_type": "eob",
            "event_origin": "feed",
            "observed_sample": 15_360,
            "encoder_frame": 12,
            "encoder_time_seconds": 0.96,
        },
        {
            "event_type": "eou",
            "event_origin": "finalize",
            "observed_sample": 64_000,
            "encoder_frame": 50,
            "encoder_time_seconds": 4.0,
        },
    ]

    result = parse_response(encode_message(payload))

    assert isinstance(result, ParakeetCppFinalEvent)
    assert [event.event_type for event in result.observed_events] == [
        "eou",
        "eob",
        "eou",
    ]
    assert result.source_samples_consumed == result.source_samples_offered
    assert result.tail_samples_consumed == result.tail_samples_offered
    assert result.native_endpoint is False
    assert result.endpoint_reason == "tail_exhausted"
    assert result.event_origin == "none"
    assert result.encoder_frame is None
    assert result.encoder_time_seconds is None
    assert result.endpoint_sample is None


def test_early_first_eou_prevents_later_complete_source_eou_from_authorizing():
    payload = _tail_exhausted_parakeet_cpp_payload()
    payload["observed_events"] = [
        {
            "event_type": "eou",
            "event_origin": "feed",
            "observed_sample": 1_280,
            "encoder_frame": 1,
            "encoder_time_seconds": 0.08,
        },
        {
            "event_type": "eob",
            "event_origin": "feed",
            "observed_sample": 15_360,
            "encoder_frame": 12,
            "encoder_time_seconds": 0.96,
        },
        {
            "event_type": "eou",
            "event_origin": "feed",
            "observed_sample": 16_640,
            "encoder_frame": 13,
            "encoder_time_seconds": 1.04,
        },
    ]

    result = parse_response(encode_message(payload))

    assert isinstance(result, ParakeetCppFinalEvent)
    assert result.observed_events[0].observed_sample == 1_280
    assert result.source_samples_consumed == result.source_samples_offered
    assert result.tail_samples_consumed == result.tail_samples_offered
    assert result.endpoint_reason == "tail_exhausted"
    assert result.endpoint_sample is None
    assert result.encoder_frame is None
    assert result.authoritative is False


def test_eob_before_complete_source_eou_does_not_block_acceptance():
    payload = _parakeet_cpp_final_payload()
    payload["observed_events"] = [
        {
            "event_type": "eob",
            "event_origin": "feed",
            "observed_sample": 1_280,
            "encoder_frame": 1,
            "encoder_time_seconds": 0.08,
        },
        {
            "event_type": "eou",
            "event_origin": "feed",
            "observed_sample": 16_640,
            "encoder_frame": 13,
            "encoder_time_seconds": 1.04,
        },
    ]

    result = parse_response(encode_message(payload))

    assert isinstance(result, ParakeetCppFinalEvent)
    assert result.observed_events[0].event_type == "eob"
    assert result.endpoint_reason == "eou"
    assert result.endpoint_sample == 16_640
    assert result.encoder_frame == 13
    assert result.authoritative is True


@pytest.mark.parametrize(
    "updates",
    [
        {"authoritative": False},
        {"endpoint_reason": "eob", "authoritative": True},
        {"event_origin": "finalize"},
        {"native_endpoint": False},
        {"encoder_frame": 12},
        {"encoder_time_seconds": 0.96},
        {"endpoint_sample": 16_000},
        {"endpoint_latency_ms": 79.0},
        {
            "observed_events": [
                {
                    "event_type": "eou",
                    "event_origin": "feed",
                    "observed_sample": 1_280,
                    "encoder_frame": 1,
                    "encoder_time_seconds": 0.08,
                }
            ],
        },
    ],
)
def test_parakeet_cpp_terminal_must_match_first_accepted_observation(updates):
    payload = _parakeet_cpp_final_payload()
    payload.update(updates)

    with pytest.raises(ProtocolError):
        parse_response(encode_message(payload))


def test_first_observed_complete_source_eou_is_the_only_accepted_terminal():
    payload = _parakeet_cpp_final_payload()
    payload["observed_events"] = [
        {
            "event_type": "eou",
            "event_origin": "feed",
            "observed_sample": 16_640,
            "encoder_frame": 12,
            "encoder_time_seconds": 0.96,
        },
        {
            "event_type": "eou",
            "event_origin": "feed",
            "observed_sample": 16_640,
            "encoder_frame": 13,
            "encoder_time_seconds": 1.04,
        },
    ]

    with pytest.raises(ProtocolError):
        parse_response(encode_message(payload))

    payload.update({"encoder_frame": 12, "encoder_time_seconds": 0.96})
    parsed = parse_response(encode_message(payload))
    assert isinstance(parsed, ParakeetCppFinalEvent)
    assert parsed.encoder_frame == 12


def test_exact_observed_event_maximum_is_accepted_and_one_more_is_rejected():
    payload = _tail_exhausted_parakeet_cpp_payload()
    events = [
        {
            "event_type": "eob",
            "event_origin": "finalize",
            "observed_sample": 64_000,
            "encoder_frame": frame,
            "encoder_time_seconds": frame * 0.08,
        }
        for frame in range(MAX_PARAKEET_CPP_OBSERVED_EVENTS)
    ]
    payload["observed_events"] = events

    parsed = parse_response(encode_message(payload))
    assert isinstance(parsed, ParakeetCppFinalEvent)
    assert len(parsed.observed_events) == MAX_PARAKEET_CPP_OBSERVED_EVENTS

    events.append(
        {
            "event_type": "eob",
            "event_origin": "finalize",
            "observed_sample": 64_000,
            "encoder_frame": MAX_PARAKEET_CPP_OBSERVED_EVENTS,
            "encoder_time_seconds": MAX_PARAKEET_CPP_OBSERVED_EVENTS * 0.08,
        }
    )
    with pytest.raises(ProtocolError):
        parse_response(encode_message(payload))


@pytest.mark.parametrize(
    "events",
    [
        [
            {
                "event_type": "eob",
                "event_origin": "feed",
                "observed_sample": 1_281,
                "encoder_frame": 1,
                "encoder_time_seconds": 0.08,
            }
        ],
        [
            {
                "event_type": "eob",
                "event_origin": "feed",
                "observed_sample": 16_640,
                "encoder_frame": 13,
                "encoder_time_seconds": 1.0399995,
            },
            {
                "event_type": "eob",
                "event_origin": "feed",
                "observed_sample": 17_920,
                "encoder_frame": 13,
                "encoder_time_seconds": 1.04,
            },
        ],
    ],
)
def test_parakeet_cpp_rejects_impossible_feed_geometry_and_near_duplicates(
    events,
):
    payload = _tail_exhausted_parakeet_cpp_payload()
    payload["observed_events"] = events

    with pytest.raises(ProtocolError):
        parse_response(encode_message(payload))


@pytest.mark.parametrize(
    "events",
    [
        None,
        {},
        [
            {
                "event_type": "eou",
                "event_origin": "feed",
                "observed_sample": 16_640,
                "encoder_frame": 13,
            }
        ],
        [
            {
                "event_type": "eou",
                "event_origin": "feed",
                "observed_sample": 16_640,
                "encoder_frame": 13,
                "encoder_time_seconds": 1.04,
                "extra": 0,
            }
        ],
        [
            {
                "event_type": "silence",
                "event_origin": "feed",
                "observed_sample": 16_640,
                "encoder_frame": 13,
                "encoder_time_seconds": 1.04,
            }
        ],
        [
            {
                "event_type": "eou",
                "event_origin": "callback",
                "observed_sample": 16_640,
                "encoder_frame": 13,
                "encoder_time_seconds": 1.04,
            }
        ],
        [
            {
                "event_type": "eou",
                "event_origin": "feed",
                "observed_sample": True,
                "encoder_frame": 13,
                "encoder_time_seconds": 1.04,
            }
        ],
        [
            {
                "event_type": "eou",
                "event_origin": "feed",
                "observed_sample": 16_640,
                "encoder_frame": True,
                "encoder_time_seconds": 1.04,
            }
        ],
        [
            {
                "event_type": "eou",
                "event_origin": "feed",
                "observed_sample": 16_640,
                "encoder_frame": 13,
                "encoder_time_seconds": float("inf"),
            }
        ],
    ],
)
def test_parakeet_cpp_observed_events_have_an_exact_bounded_shape(events):
    payload = _parakeet_cpp_final_payload()
    payload["observed_events"] = events

    with pytest.raises(ProtocolError):
        parse_response(json.dumps(payload).encode())


@pytest.mark.parametrize(
    "events",
    [
        [
            {
                "event_type": "eou",
                "event_origin": "feed",
                "observed_sample": 0,
                "encoder_frame": 0,
                "encoder_time_seconds": 0.0,
            }
        ],
        [
            {
                "event_type": "eou",
                "event_origin": "feed",
                "observed_sample": 17_281,
                "encoder_frame": 13,
                "encoder_time_seconds": 1.04,
            }
        ],
        [
            {
                "event_type": "eou",
                "event_origin": "feed",
                "observed_sample": 1_280,
                "encoder_frame": 2,
                "encoder_time_seconds": 0.16,
            }
        ],
        [
            {
                "event_type": "eou",
                "event_origin": "feed",
                "observed_sample": 16_640,
                "encoder_frame": 13,
                "encoder_time_seconds": 1.03,
            }
        ],
        [
            {
                "event_type": "eob",
                "event_origin": "feed",
                "observed_sample": 2_560,
                "encoder_frame": 2,
                "encoder_time_seconds": 0.16,
            },
            {
                "event_type": "eou",
                "event_origin": "feed",
                "observed_sample": 1_280,
                "encoder_frame": 3,
                "encoder_time_seconds": 0.24,
            },
        ],
        [
            {
                "event_type": "eob",
                "event_origin": "feed",
                "observed_sample": 1_280,
                "encoder_frame": 2,
                "encoder_time_seconds": 0.16,
            },
            {
                "event_type": "eou",
                "event_origin": "feed",
                "observed_sample": 2_560,
                "encoder_frame": 1,
                "encoder_time_seconds": 0.08,
            },
        ],
        [
            {
                "event_type": "eob",
                "event_origin": "finalize",
                "observed_sample": 16_640,
                "encoder_frame": 12,
                "encoder_time_seconds": 0.96,
            },
            {
                "event_type": "eou",
                "event_origin": "feed",
                "observed_sample": 16_640,
                "encoder_frame": 13,
                "encoder_time_seconds": 1.04,
            },
        ],
        [
            {
                "event_type": "eob",
                "event_origin": "finalize",
                "observed_sample": 16_000,
                "encoder_frame": 13,
                "encoder_time_seconds": 1.04,
            }
        ],
        [
            {
                "event_type": "eob",
                "event_origin": "feed",
                "observed_sample": 1_280,
                "encoder_frame": 1,
                "encoder_time_seconds": 0.08,
            },
            {
                "event_type": "eob",
                "event_origin": "feed",
                "observed_sample": 2_560,
                "encoder_frame": 1,
                "encoder_time_seconds": 0.08,
            },
        ],
    ],
)
def test_parakeet_cpp_observations_reject_bad_order_boundaries_and_duplicates(
    events,
):
    payload = _parakeet_cpp_final_payload()
    payload["observed_events"] = events

    with pytest.raises(ProtocolError):
        parse_response(encode_message(payload))


@pytest.mark.parametrize(
    "updates",
    [
        {"v": NATIVE_ENDPOINT_PROTOCOL_VERSION},
        {"type": "native_final"},
        {"endpoint_reason": "silence"},
        {"endpoint_reason": {}},
        {"event_origin": "callback"},
        {"event_origin": []},
        {"native_endpoint": 1},
        {"source_samples_offered": True},
        {"tail_samples_offered": 48_001},
        {"encoder_frame": True},
        {"encoder_time_seconds": float("inf")},
        {"observed_events": "events"},
        {"endpoint_probability": 0.75},
    ],
)
def test_parakeet_cpp_final_rejects_wrong_version_shape_and_unbounded_fields(
    updates,
):
    payload = _parakeet_cpp_final_payload()
    payload.update(updates)

    with pytest.raises(ProtocolError):
        parse_response(json.dumps(payload).encode())


@pytest.mark.parametrize(
    "updates",
    [
        {"samples_seen": 17_279},
        {"source_samples_offered": 15_999},
        {"tail_samples_offered": 639},
        {
            "source_samples_consumed": 15_999,
            "tail_samples_consumed": 1_281,
        },
        {"endpoint_sample": 17_279},
        {"model_padding_samples": 1, "endpoint_sample": 17_281},
        {"encoder_frame": None},
        {"encoder_time_seconds": None},
        {"encoder_frame": 17_281},
        {"encoder_time_seconds": 1.081},
        {"encoder_frame": 0},
        {"event_origin": "none"},
        {"endpoint_latency_ms": None},
        {"endpoint_latency_ms": 79.0},
        {"native_endpoint": False},
    ],
)
def test_parakeet_cpp_final_rejects_relationally_inconsistent_evidence(updates):
    payload = _parakeet_cpp_final_payload()
    payload.update(updates)

    with pytest.raises(ProtocolError):
        parse_response(encode_message(payload))


@pytest.mark.parametrize(
    "updates",
    [
        {"source_samples_consumed": 15_999, "samples_seen": 63_999},
        {"tail_samples_consumed": 47_999, "samples_seen": 63_999},
        {"native_endpoint": True},
        {"authoritative": True},
        {"endpoint_reason": "eob"},
        {"event_origin": "feed"},
        {"encoder_frame": 1},
        {"encoder_time_seconds": 0.08},
        {"endpoint_sample": 64_000},
        {"endpoint_latency_ms": 0.0},
    ],
)
def test_tail_exhaustion_remains_terminal_when_events_are_only_informational(
    updates,
):
    payload = _tail_exhausted_parakeet_cpp_payload()
    payload["observed_events"] = [
        {
            "event_type": "eou",
            "event_origin": "feed",
            "observed_sample": 1_280,
            "encoder_frame": 1,
            "encoder_time_seconds": 0.08,
        },
        {
            "event_type": "eou",
            "event_origin": "finalize",
            "observed_sample": 64_000,
            "encoder_frame": 50,
            "encoder_time_seconds": 4.0,
        },
    ]
    payload.update(updates)

    with pytest.raises(ProtocolError):
        parse_response(encode_message(payload))


def test_v4_rejects_v3_probability_bearing_native_final():
    payload = _native_final_payload()
    payload["v"] = PARAKEET_CPP_ENDPOINT_PROTOCOL_VERSION

    with pytest.raises(ProtocolError):
        parse_response(encode_message(payload))


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
