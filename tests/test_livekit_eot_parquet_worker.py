from __future__ import annotations

import json
import math
import struct

import pytest

from tools import livekit_eot_parquet_worker as worker


def _wav(
    samples: int = 16_000,
    *,
    audio_format: int = 1,
    channels: int = 1,
    sample_rate: int = 16_000,
    bits: int = 16,
    block_align: int = 2,
    byte_rate: int = 32_000,
) -> bytes:
    data = b"\x00" * (samples * block_align)
    fmt = struct.pack(
        "<HHIIHH",
        audio_format,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits,
    )
    payload = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    payload += b"data" + struct.pack("<I", len(data)) + data
    return b"RIFF" + struct.pack("<I", len(payload) + 4) + b"WAVE" + payload


def _row(
    *,
    silence_spans: list[dict[str, float]] | None = None,
    audio_path: str = "synthetic.wav",
    duration: float = 1.0,
    audio_payload: bytes | None = None,
) -> dict[str, object]:
    return {
        "audio": {"bytes": _wav() if audio_payload is None else audio_payload, "path": audio_path},
        "language": "en",
        "duration": duration,
        "silence_spans": silence_spans
        or [
            {"start": 0.25, "end": 0.40},
            {"start": 0.75, "end": 1.0},
        ],
    }


def _features() -> dict[str, object]:
    value_string = {"dtype": "string", "_type": "Value"}
    value_float = {"dtype": "float64", "_type": "Value"}
    return {
        "id": dict(value_string),
        "audio": {"sampling_rate": 16_000, "_type": "Audio"},
        "language": dict(value_string),
        "duration": dict(value_float),
        "silence_spans": {
            "feature": {"start": dict(value_float), "end": dict(value_float)},
            "_type": "List",
        },
        "words": {
            "feature": {
                "start": dict(value_float),
                "end": dict(value_float),
                "word": dict(value_string),
            },
            "_type": "List",
        },
        "messages": {
            "feature": {"role": dict(value_string), "content": dict(value_string)},
            "_type": "List",
        },
    }


def test_synthetic_pcm_row_yields_only_aggregate_values():
    summary = worker._validate_row(_row(audio_path="absent-but-never-opened.wav"))

    assert worker.SCHEMA_CONTRACT["audio"] == {
        "container": "riff-wave",
        "audio_format": 1,
        "encoding": "pcm-signed-16-little-endian",
        "sample_rate_hz": 16_000,
        "channels": 1,
        "bits_per_sample": 16,
        "block_align_bytes": 2,
        "byte_rate": 32_000,
        "embedded_bytes_only": True,
    }
    assert summary["audio_samples"] == 16_000
    assert summary["silence_spans"] == 2
    assert summary["hold_labels"] == 1
    assert summary["eot_labels"] == 1
    assert summary["duration_quantized"] is False
    assert summary["final_gap_samples"] == 0
    assert summary["minimum_eot_silence_samples"] == 4_000
    assert set(summary) == {
        "audio_bytes",
        "audio_samples",
        "silence_spans",
        "hold_labels",
        "eot_labels",
        "minimum_eot_silence_samples",
        "maximum_silence_span_samples",
        "duration_quantized",
        "final_gap_samples",
        "audio_sha256",
        "silence_sha256",
    }


def test_all_silence_spans_are_at_least_100ms_and_final_is_at_least_200ms():
    worker._validate_row(
        _row(
            silence_spans=[
                {"start": 0.0, "end": 0.1},
                {"start": 0.75, "end": 1.0},
            ]
        )
    )
    worker._validate_row(
        _row(
            duration=0.2,
            audio_payload=_wav(samples=3_200),
            silence_spans=[{"start": 0.0, "end": 0.2}],
        )
    )
    with pytest.raises(worker.WorkerError):
        worker._validate_row(
            _row(
                silence_spans=[
                    {"start": 0.0, "end": 0.1 - 1e-9},
                    {"start": 0.75, "end": 1.0},
                ]
            )
        )
    with pytest.raises(worker.WorkerError):
        worker._validate_row(
            _row(silence_spans=[{"start": 0.8 + 1e-9, "end": 1.0}])
        )


def test_silence_span_five_second_maximum_is_exact():
    six_second_wav = _wav(samples=6 * 16_000)
    worker._validate_row(
        _row(
            duration=6.0,
            audio_payload=six_second_wav,
            silence_spans=[
                {"start": 0.0, "end": 5.0},
                {"start": 5.8, "end": 6.0},
            ],
        )
    )
    with pytest.raises(worker.WorkerError):
        worker._validate_row(
            _row(
                duration=6.0,
                audio_payload=six_second_wav,
                silence_spans=[
                    {"start": 0.0, "end": 5.0 + 1e-9},
                    {"start": 5.8, "end": 6.0},
                ],
            )
        )


def test_final_end_accepts_exact_one_sample_gap_but_rejects_two_samples():
    one_sample = 1.0 - 1.0 / 16_000
    summary = worker._validate_row(
        _row(silence_spans=[{"start": 0.75, "end": one_sample}])
    )
    assert summary["eot_labels"] == 1

    with pytest.raises(worker.WorkerError):
        worker._validate_row(
            _row(
                silence_spans=[
                    {"start": 0.75, "end": 1.0 - 2.0 / 16_000},
                ]
            )
        )


def test_every_silence_end_must_be_within_the_audio():
    with pytest.raises(worker.WorkerError):
        worker._validate_row(
            _row(silence_spans=[{"start": 0.75, "end": 1.0 + 0.1 / 16_000}])
        )


@pytest.mark.parametrize("gap_samples", [1.000001, 1.25, 1.49, 2.0])
def test_final_end_rejects_every_gap_greater_than_one_sample(gap_samples: float):
    with pytest.raises(worker.WorkerError):
        worker._validate_row(
            _row(
                silence_spans=[
                    {"start": 0.75, "end": 1.0 - gap_samples / 16_000},
                ]
            )
        )


def test_duration_publisher_microsecond_quantization_envelope_is_exact():
    summary = worker._validate_row(_row(duration=1.0000005))
    assert summary["duration_quantized"] is True
    with pytest.raises(worker.WorkerError):
        worker._validate_row(_row(duration=1.0000005000625))


@pytest.mark.parametrize("duration", [0.9999375, 1.0000625])
def test_duration_nearest_sample_must_equal_audio_samples(duration: float):
    with pytest.raises(worker.WorkerError):
        worker._validate_row(_row(duration=duration))


@pytest.mark.parametrize(
    "silence_spans",
    [
        [
            {"start": 0.250000001, "end": 0.40},
            {"start": 0.75, "end": 1.0},
        ],
        [
            {"start": 0.25, "end": 0.400000001},
            {"start": 0.75, "end": 1.0},
        ],
    ],
)
def test_every_span_endpoint_must_be_an_exact_sample(
    silence_spans: list[dict[str, float]],
):
    with pytest.raises(worker.WorkerError):
        worker._validate_row(_row(silence_spans=silence_spans))


@pytest.mark.parametrize("final_end", [0.999875, 1.0000625])
def test_final_gap_rejects_two_samples_and_negative_one_sample(final_end: float):
    with pytest.raises(worker.WorkerError):
        worker._validate_row(
            _row(silence_spans=[{"start": 0.75, "end": final_end}])
        )


@pytest.mark.parametrize("overlap_samples", [0.25, 0.75, 1.0])
def test_silence_spans_reject_every_positive_overlap(overlap_samples: float):
    with pytest.raises(worker.WorkerError):
        worker._validate_row(
            _row(
                silence_spans=[
                    {"start": 0.25, "end": 0.50},
                    {
                        "start": 0.50 - overlap_samples / 16_000,
                        "end": 1.0,
                    },
                ]
            )
        )


@pytest.mark.parametrize(
    "payload",
    [
        _wav(audio_format=3, bits=32, block_align=4, byte_rate=64_000),
        _wav(bits=8, block_align=1, byte_rate=16_000),
        _wav(bits=24, block_align=3, byte_rate=48_000),
        _wav(channels=2, block_align=4, byte_rate=64_000),
        _wav(block_align=4, byte_rate=64_000),
    ],
)
def test_embedded_wav_must_be_exact_pcm16_mono_16khz(payload: bytes):
    with pytest.raises(worker.WorkerError):
        worker._validate_row(_row(audio_payload=payload))


@pytest.mark.parametrize(
    "audio_path",
    [
        "",
        ".",
        "..",
        "../escape.wav",
        "directory/audio.wav",
        "directory\\audio.wav",
        "x\x00.wav",
    ],
)
def test_audio_path_must_be_a_flat_nonempty_value_but_is_never_opened(audio_path: str):
    with pytest.raises(worker.WorkerError):
        worker._validate_row(_row(audio_path=audio_path))


def test_huggingface_metadata_requires_exact_pyarrow25_list_feature_shape():
    metadata = {
        b"huggingface": json.dumps({"info": {"features": _features()}}).encode()
    }
    worker._validate_huggingface_metadata(metadata)

    wrong = _features()
    wrong["silence_spans"] = [wrong["silence_spans"]["feature"]]
    with pytest.raises(worker.WorkerError):
        worker._validate_huggingface_metadata(
            {b"huggingface": json.dumps({"info": {"features": wrong}}).encode()}
        )


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_or_out_of_order_silence_is_rejected(bad: float):
    with pytest.raises(worker.WorkerError):
        worker._validate_row(
            _row(silence_spans=[{"start": 0.75, "end": bad}])
        )
    with pytest.raises(worker.WorkerError):
        worker._validate_row(
            _row(
                silence_spans=[
                    {"start": 0.4, "end": 0.6},
                    {"start": 0.5, "end": 1.0},
                ]
            )
        )
