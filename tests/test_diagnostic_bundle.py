from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import threading
import time
import wave

import numpy as np
import pytest

import core.diagnostic_bundle as diagnostic_bundle
from core.diagnostic_bundle import (
    CaptureCoordinate,
    DiagnosticManifestStatus,
    DiagnosticObservation,
    DiagnosticStage,
    DiagnosticTrack,
    EndpointReplayConfig,
    FinalModelInputRole,
    SynchronizedDiagnosticBundle,
    validate_manifest,
)


def _paths(tmp_path: Path) -> tuple[dict[DiagnosticTrack, Path], Path, Path]:
    tracks = {
        role: tmp_path / f"run.{role.value}.wav" for role in DiagnosticTrack
    }
    return tracks, tmp_path / "run.timeline.jsonl", tmp_path / "run.manifest.json"


def _coordinate(
    sequence: int,
    source_start: int,
    source_end: int,
    *,
    source_generation: int = 2,
    capture_generation: int = 4,
    gap: bool = False,
) -> CaptureCoordinate:
    duration = (source_end - source_start) / 48_000.0
    captured_at = 100.0 + sequence * duration
    fields: dict[str, object] = {}
    if gap:
        fields.update(
            gap_reason="media_backpressure",
            gap_prior_generation=capture_generation - 1,
            gap_generation=capture_generation,
            gap_first_dropped_sequence=sequence - 2,
            gap_last_dropped_sequence=sequence - 1,
            gap_dropped_frames=2,
            gap_dropped_samples=960,
        )
    return CaptureCoordinate(
        sequence=sequence,
        sample_rate_hz=48_000,
        captured_started_at=captured_at - duration,
        captured_at=captured_at,
        capture_epoch=3,
        source_generation=source_generation,
        capture_generation=capture_generation,
        source_sample_start=source_start,
        source_sample_end=source_end,
        **fields,
    )


def _frame(length: int, base: float = 0.1) -> dict[DiagnosticTrack, np.ndarray]:
    return {
        role: np.full(length, base + index * 0.1, dtype="float32")
        for index, role in enumerate(DiagnosticTrack)
    }


def _bundle(
    tmp_path: Path, *, queue_max: int = 16, flush_sec: float = 0.05
) -> SynchronizedDiagnosticBundle:
    tracks, timeline, manifest = _paths(tmp_path)
    return SynchronizedDiagnosticBundle(
        tracks,
        timeline,
        manifest,
        sample_rate=16_000,
        queue_max=queue_max,
        flush_sec=flush_sec,
    )


def _complete_turn(
    bundle: SynchronizedDiagnosticBundle,
    span,
    *,
    utterance_id: str,
    revision: int,
    samples: np.ndarray,
    role: FinalModelInputRole,
):
    stream_id = "sherpa-11111111111111111111111111111111"
    observations = (
        DiagnosticObservation(
            stage=DiagnosticStage.ENDPOINT_EVALUATED,
            monotonic_ns=10 + revision * 10,
            span=span,
            stream_id=stream_id,
            utterance_id=utterance_id,
            reason="asr",
            basis="acoustic",
            completion_state="unavailable",
            acoustic_endpoint=True,
            vad_active=False,
            early_endpoint_allowed=True,
            trailing_silence_sec=0.8,
            boolean_value=True,
        ),
        DiagnosticObservation(
            stage=DiagnosticStage.ENDPOINT_COMMITTED,
            monotonic_ns=10 + revision * 10,
            span=span,
            stream_id=stream_id,
            utterance_id=utterance_id,
            revision=revision,
            reason="asr",
            basis="acoustic",
            completion_state="unavailable",
        ),
        DiagnosticObservation(
            stage=DiagnosticStage.ASR_STREAMING_FINAL,
            monotonic_ns=12 + revision * 10,
            span=span,
            stream_id=stream_id,
            utterance_id=utterance_id,
            revision=revision,
            outcome="nonempty",
        ),
        DiagnosticObservation(
            stage=DiagnosticStage.FINALIZER_STARTED,
            monotonic_ns=13 + revision * 10,
            span=span,
            stream_id=stream_id,
            utterance_id=utterance_id,
            revision=revision,
        ),
    )
    for observation in observations:
        assert bundle.observe(observation)
    receipt = bundle.write_final_model_input(
        samples,
        stream_id=stream_id,
        utterance_id=utterance_id,
        capture_epoch=span.coordinate.capture_epoch,
        capture_generation=span.coordinate.capture_generation,
        revision=revision,
        role=role,
    )
    assert receipt is not None
    for observation in (
        DiagnosticObservation(
            stage=DiagnosticStage.FINAL_SELECTION,
            monotonic_ns=14 + revision * 10,
            span=span,
            stream_id=stream_id,
            utterance_id=utterance_id,
            revision=revision,
            selected_source="streaming",
            outcome="unavailable",
            support=0,
            boolean_value=False,
        ),
        DiagnosticObservation(
            stage=DiagnosticStage.FINAL_DISPATCHED,
            monotonic_ns=15 + revision * 10,
            span=span,
            stream_id=stream_id,
            utterance_id=utterance_id,
            revision=revision,
            outcome="callback_returned",
        ),
    ):
        assert bundle.observe(observation)
    return receipt


def _timeline(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _rewrite_timeline_and_bind(
    timeline: Path,
    manifest: Path,
    records: list[dict[str, object]],
) -> None:
    encoded = b"".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n"
        for record in records
    )
    timeline.write_bytes(encoded)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["timeline"]["sha256"] = hashlib.sha256(encoded).hexdigest()
    payload["timeline"]["bytes"] = len(encoded)
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _downgrade_to_schema_v1(timeline: Path, manifest: Path) -> None:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["schema_version"] = 1
    payload.pop("final_model_input")
    payload["provenance"] = {
        "native_host_pcm_present": False,
        "physical_raw_mic_present": False,
        "model_pre_gain_tap_may_include_host_processing": True,
        "model_pre_gain_tap_includes_application_resampling": True,
        "audio_storage_is_pcm16_quantized": True,
        "model_asr_track_is_continuous_selected_tap": True,
        "recognizer_reset_replay_receipts_present": False,
        "turn_selection_ranges_present": False,
        "frame_track_lengths_may_differ": True,
        "observation_spans_use_model_gate_track": True,
        "playback_reference_preserves_reader_snapshot": True,
        "playback_timestamps_are_receipt_dispatch_time": True,
        "physical_playback_audibility_present": False,
    }
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    records = _timeline(timeline)
    records[0]["schema_version"] = 1
    records[0].pop("final_model_input")
    records[-1].pop("final_model_input_count")
    records[-1].pop("final_model_input_samples")
    _rewrite_timeline_and_bind(timeline, manifest, records)


def test_clean_bundle_writes_equal_tracks_typed_observation_hashes_and_footer(
    tmp_path: Path,
) -> None:
    tracks, timeline, manifest = _paths(tmp_path)
    bundle = _bundle(tmp_path)

    first = bundle.write_frame(_frame(8, 0.05), _coordinate(1, 0, 24))
    second = bundle.write_frame(_frame(12, 0.15), _coordinate(2, 24, 60))
    assert first is not None and second is not None
    assert (first.bundle_sample_start, first.bundle_sample_end) == (0, 8)
    assert (second.bundle_sample_start, second.bundle_sample_end) == (8, 20)
    assert bundle.observe(
        DiagnosticObservation(
            stage=DiagnosticStage.BARGE_DETECTED,
            monotonic_ns=101_500_000_000,
            span=second,
            stream_id="sherpa-11111111111111111111111111111111",
            utterance_id="u2",
            revision=3,
        )
    )

    assert bundle.close() == manifest
    assert bundle.complete
    assert bundle.manifest_status is DiagnosticManifestStatus.COMPLETE
    assert set(bundle.paths) == {
        *(role.value for role in DiagnosticTrack),
        "final_model_input",
        "timeline",
        "manifest",
    }
    assert validate_manifest(manifest)

    counts = []
    first_values = []
    for role in DiagnosticTrack:
        with wave.open(str(tracks[role]), "rb") as recording:
            assert recording.getnchannels() == 1
            assert recording.getsampwidth() == 2
            assert recording.getframerate() == 16_000
            counts.append(recording.getnframes())
            first_values.append(
                int.from_bytes(recording.readframes(1), "little", signed=True)
            )
    assert counts == [20, 20, 20, 20]
    assert len(set(first_values)) == len(DiagnosticTrack)

    records = _timeline(timeline)
    assert [record["kind"] for record in records] == [
        "header",
        "frame",
        "frame",
        "observation",
        "footer",
    ]
    assert records[-1] == {
        "clean_shutdown": True,
        "complete": True,
        "failure_codes": [],
        "frame_count": 2,
        "kind": "footer",
        "sample_count": 20,
        "sample_count_by_track": {
            role.value: 20 for role in DiagnosticTrack
        },
        "final_model_input_count": 0,
        "final_model_input_samples": 0,
    }
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["complete"] is True
    assert payload["sample_count"] == 20
    assert payload["provenance"] == {
        "continuous_audio_storage_is_pcm16_quantized": True,
        "final_model_input_storage_is_lossless_f32le": True,
        "final_model_input_receipts_present": True,
        "frame_track_lengths_may_differ": True,
        "model_asr_track_is_continuous_selected_tap": True,
        "model_pre_gain_tap_includes_application_resampling": True,
        "model_pre_gain_tap_may_include_host_processing": True,
        "native_host_pcm_present": False,
        "observation_spans_use_model_gate_track": True,
        "physical_playback_audibility_present": False,
        "physical_raw_mic_present": False,
        "playback_reference_preserves_reader_snapshot": True,
        "playback_timestamps_are_receipt_dispatch_time": True,
        "recognizer_reset_replay_receipts_present": False,
        "turn_selection_ranges_present": True,
    }
    for entry in [
        payload["timeline"],
        *payload["tracks"].values(),
        payload["final_model_input"],
    ]:
        assert Path(entry["file"]).name == entry["file"]
        artifact = tmp_path / entry["file"]
        raw = artifact.read_bytes()
        assert entry["sha256"] == hashlib.sha256(raw).hexdigest()
        assert entry["bytes"] == len(raw)


def test_constructor_preserves_legacy_positional_optional_arguments(
    tmp_path: Path,
) -> None:
    tracks, timeline, manifest = _paths(tmp_path)
    replay = EndpointReplayConfig(enabled=True)

    bundle = SynchronizedDiagnosticBundle(
        tracks,
        timeline,
        manifest,
        16_000,
        replay,
        7,
        0.02,
        1.5,
    )

    assert bundle.endpoint_replay_config is replay
    assert bundle._queue.maxsize == 7
    assert bundle._flush_sec == 0.02
    assert bundle._close_timeout_sec == 1.5
    assert bundle.write_frame(_frame(4), _coordinate(1, 0, 12)) is not None
    bundle.close()
    assert validate_manifest(manifest)


def test_schema_v1_bundle_remains_read_only_validatable(tmp_path: Path) -> None:
    _tracks, timeline, manifest = _paths(tmp_path)
    bundle = _bundle(tmp_path)
    assert bundle.write_frame(_frame(8), _coordinate(1, 0, 24)) is not None
    bundle.close()
    _downgrade_to_schema_v1(timeline, manifest)

    assert validate_manifest(manifest)
    track = bundle.path_by_role[DiagnosticTrack.MODEL_ASR_TAP]
    with track.open("r+b") as handle:
        handle.seek(-1, os.SEEK_END)
        value = handle.read(1)
        handle.seek(-1, os.SEEK_END)
        handle.write(bytes([value[0] ^ 0x01]))
    assert not validate_manifest(manifest)


def test_exact_final_inputs_are_lossless_packed_identity_bound_and_immutable(
    tmp_path: Path,
) -> None:
    _tracks, timeline, manifest = _paths(tmp_path)
    bundle = _bundle(tmp_path)
    span = bundle.write_frame(_frame(8), _coordinate(1, 0, 24))
    assert span is not None
    first = np.array([1.25, -1.5, 0.1234567], dtype="float32")
    second = np.array([-0.33333334, 0.75], dtype="float32")
    expected = first.astype("<f4", copy=True).tobytes() + second.astype(
        "<f4", copy=True
    ).tobytes()

    first_receipt = _complete_turn(
        bundle,
        span,
        utterance_id="u1",
        revision=1,
        samples=first,
        role=FinalModelInputRole.MODEL_GATE_SEGMENT,
    )
    first[:] = 0.0
    second_receipt = _complete_turn(
        bundle,
        span,
        utterance_id="u2",
        revision=2,
        samples=second,
        role=FinalModelInputRole.SELECTED_ASR_SEGMENT,
    )
    bundle.close()

    assert bundle.final_model_input_path.read_bytes() == expected
    assert (first_receipt.sample_start, first_receipt.sample_end) == (0, 3)
    assert (first_receipt.byte_start, first_receipt.byte_end) == (0, 12)
    assert (second_receipt.sample_start, second_receipt.sample_end) == (3, 5)
    assert (second_receipt.byte_start, second_receipt.byte_end) == (12, 20)
    receipts = [
        record
        for record in _timeline(timeline)
        if record["kind"] == "final_model_input"
    ]
    assert [record["role"] for record in receipts] == [
        FinalModelInputRole.MODEL_GATE_SEGMENT.value,
        FinalModelInputRole.SELECTED_ASR_SEGMENT.value,
    ]
    assert [record["capture_epoch"] for record in receipts] == [3, 3]
    assert [record["capture_generation"] for record in receipts] == [4, 4]
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["final_model_input"] == {
        "file": bundle.final_model_input_path.name,
        "sha256": hashlib.sha256(expected).hexdigest(),
        "bytes": len(expected),
        "samples": 5,
        "input_count": 2,
        "encoding": "f32le",
    }
    assert validate_manifest(manifest)


def test_per_input_digest_rejects_spool_mutation_even_when_outer_hash_is_rebound(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    span = bundle.write_frame(_frame(8), _coordinate(1, 0, 24))
    assert span is not None
    _complete_turn(
        bundle,
        span,
        utterance_id="u1",
        revision=1,
        samples=np.array([0.1234567, -0.7654321], dtype="float32"),
        role=FinalModelInputRole.MODEL_GATE_SEGMENT,
    )
    manifest = bundle.close()
    assert validate_manifest(manifest)

    mutated = bytearray(bundle.final_model_input_path.read_bytes())
    mutated[0] ^= 0x01
    bundle.final_model_input_path.write_bytes(mutated)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["final_model_input"]["sha256"] = hashlib.sha256(mutated).hexdigest()
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    assert not validate_manifest(manifest)


def test_final_input_byte_backpressure_rejects_atomically_and_fails_closed(
    tmp_path: Path,
) -> None:
    tracks, timeline, manifest = _paths(tmp_path)
    bundle = SynchronizedDiagnosticBundle(
        tracks,
        timeline,
        manifest,
        sample_rate=16_000,
        final_input_queue_max_bytes=8,
        flush_sec=10.0,
    )
    assert bundle.write_frame(_frame(4), _coordinate(1, 0, 12)) is not None
    entered = threading.Event()
    release = threading.Event()
    original = bundle._write_final_model_input_item

    def blocked(item) -> None:
        entered.set()
        assert release.wait(timeout=2.0)
        original(item)

    bundle._write_final_model_input_item = blocked  # type: ignore[method-assign]
    first = bundle.write_final_model_input(
        np.array([0.1, 0.2], dtype="float32"),
        stream_id="sherpa-11111111111111111111111111111111",
        utterance_id="u1",
        capture_epoch=3,
        capture_generation=4,
        revision=1,
        role=FinalModelInputRole.MODEL_GATE_SEGMENT,
    )
    assert first is not None
    assert entered.wait(timeout=1.0)
    second = bundle.write_final_model_input(
        np.array([0.3], dtype="float32"),
        stream_id="sherpa-11111111111111111111111111111111",
        utterance_id="u2",
        capture_epoch=3,
        capture_generation=4,
        revision=2,
        role=FinalModelInputRole.MODEL_GATE_SEGMENT,
    )
    assert second is None
    assert bundle.failure_codes == ("final_input_backpressure",)
    assert bundle._next_final_input_index == 1
    assert bundle._next_final_input_sample == 2
    release.set()
    bundle.close()
    assert not validate_manifest(manifest)


@pytest.mark.parametrize(
    "samples",
    (
        np.array([], dtype="float32"),
        np.array([0.1], dtype="float64"),
        np.array([[0.1]], dtype="float32"),
        np.array([np.nan], dtype="float32"),
        np.array([np.inf], dtype="float32"),
    ),
)
def test_invalid_final_input_never_advances_spool_cursor(
    tmp_path: Path, samples: np.ndarray
) -> None:
    bundle = _bundle(tmp_path)
    assert bundle.write_frame(_frame(4), _coordinate(1, 0, 12)) is not None

    assert bundle.write_final_model_input(
        samples,
        stream_id="sherpa-11111111111111111111111111111111",
        utterance_id="u1",
        capture_epoch=3,
        capture_generation=4,
        revision=1,
        role=FinalModelInputRole.MODEL_GATE_SEGMENT,
    ) is None
    assert bundle._next_final_input_index == 0
    assert bundle._next_final_input_sample == 0
    bundle.close()
    assert bundle.failure_codes == ("invalid_final_input",)


def test_timeline_maps_packed_samples_to_source_gap_and_generation(tmp_path: Path) -> None:
    _tracks, timeline, manifest = _paths(tmp_path)
    bundle = _bundle(tmp_path)
    before = bundle.write_frame(_frame(4), _coordinate(1, 0, 12))
    after = bundle.write_frame(
        _frame(6),
        _coordinate(
            4,
            7_000,
            7_018,
            source_generation=3,
            capture_generation=5,
            gap=True,
        ),
    )
    assert before is not None and after is not None
    bundle.close()

    frame_records = [record for record in _timeline(timeline) if record["kind"] == "frame"]
    assert frame_records[0]["bundle_sample_start"] == 0
    assert frame_records[0]["bundle_sample_end"] == 4
    assert frame_records[1]["bundle_sample_start"] == 4
    assert frame_records[1]["bundle_sample_end"] == 10
    coordinate = frame_records[1]["coordinate"]
    assert coordinate["sequence"] == 4
    assert coordinate["source_generation"] == 3
    assert coordinate["capture_generation"] == 5
    assert coordinate["source_sample_start"] == 7_000
    assert coordinate["source_sample_end"] == 7_018
    assert coordinate["gap_reason"] == "media_backpressure"
    assert coordinate["gap_dropped_frames"] == 2
    assert coordinate["gap_dropped_samples"] == 960
    assert validate_manifest(manifest)


@pytest.mark.parametrize("mutation", ("sequence", "source_range", "generation"))
def test_manifest_rejects_broken_cross_frame_capture_continuity(
    tmp_path: Path, mutation: str
) -> None:
    _tracks, timeline, manifest = _paths(tmp_path)
    bundle = _bundle(tmp_path)
    assert bundle.write_frame(_frame(4), _coordinate(1, 0, 12)) is not None
    assert bundle.write_frame(_frame(4), _coordinate(2, 12, 24)) is not None
    bundle.close()
    assert validate_manifest(manifest)

    records = _timeline(timeline)
    second = records[2]["coordinate"]
    assert isinstance(second, dict)
    if mutation == "sequence":
        second["sequence"] = 1
    elif mutation == "source_range":
        second["source_sample_start"] = 13
    else:
        second["capture_generation"] = 5
    _rewrite_timeline_and_bind(timeline, manifest, records)

    assert not validate_manifest(manifest)


def test_queue_saturation_drops_whole_frame_and_marks_manifest_incomplete(
    tmp_path: Path,
) -> None:
    tracks, timeline, manifest = _paths(tmp_path)
    bundle = _bundle(tmp_path, queue_max=1)
    entered = threading.Event()
    release = threading.Event()
    original = bundle._write_frame_item

    def blocked(item) -> None:
        entered.set()
        assert release.wait(timeout=3.0)
        original(item)

    bundle._write_frame_item = blocked  # type: ignore[method-assign]
    assert bundle.write_frame(_frame(8), _coordinate(1, 0, 24)) is not None
    assert entered.wait(timeout=1.0)
    assert bundle.write_frame(_frame(8), _coordinate(2, 24, 48)) is not None
    assert bundle.write_frame(_frame(8), _coordinate(3, 48, 72)) is None
    release.set()
    bundle.close()

    assert not bundle.complete
    assert bundle.manifest_status is DiagnosticManifestStatus.INCOMPLETE
    assert "queue_drop" in bundle.failure_codes
    assert not validate_manifest(manifest)
    assert _timeline(timeline)[-1]["failure_codes"] == ["queue_drop"]
    counts = []
    for path in tracks.values():
        with wave.open(str(path), "rb") as recording:
            counts.append(recording.getnframes())
    assert counts == [16, 16, 16, 16]


def test_writer_error_marks_manifest_incomplete(tmp_path: Path) -> None:
    _tracks, timeline, manifest = _paths(tmp_path)
    bundle = _bundle(tmp_path)

    def fail_write(_item) -> None:
        raise OSError("synthetic diagnostic write failure")

    bundle._write_frame_item = fail_write  # type: ignore[method-assign]
    assert bundle.write_frame(_frame(8), _coordinate(1, 0, 24)) is not None
    bundle.close()

    assert bundle.manifest_status is DiagnosticManifestStatus.INCOMPLETE
    assert bundle.failure_codes == ("write_error",)
    assert _timeline(timeline)[-1]["failure_codes"] == ["write_error"]
    assert not validate_manifest(manifest)


@pytest.mark.parametrize("fault", ["missing", "dtype"])
def test_invalid_frame_is_all_tracks_or_none_and_incomplete(
    tmp_path: Path, fault: str
) -> None:
    _tracks, _timeline_path, manifest = _paths(tmp_path)
    bundle = _bundle(tmp_path)
    tracks = _frame(8)
    expected = "invalid_track"
    if fault == "missing":
        tracks.pop(DiagnosticTrack.MODEL_ASR_TAP)
        expected = "missing_track"
    else:
        tracks[DiagnosticTrack.MODEL_ASR_TAP] = np.zeros(8, dtype="float64")

    assert bundle.write_frame(tracks, _coordinate(1, 0, 24)) is None
    bundle.close()
    assert expected in bundle.failure_codes
    assert not validate_manifest(manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["frame_count"] == 0
    assert payload["sample_count"] == 0


def test_per_track_cursors_preserve_stateful_dsp_length_differences(
    tmp_path: Path,
) -> None:
    tracks, timeline, manifest = _paths(tmp_path)
    bundle = _bundle(tmp_path)
    frame = _frame(8)
    frame[DiagnosticTrack.MODEL_PRE_GAIN_TAP] = np.zeros(1468, dtype="float32")
    frame[DiagnosticTrack.MODEL_GATE_PCM] = np.zeros(1440, dtype="float32")
    frame[DiagnosticTrack.MODEL_ASR_TAP] = np.zeros(1440, dtype="float32")
    frame[DiagnosticTrack.PLAYBACK_REFERENCE_READER_SNAPSHOT] = np.zeros(
        1440, dtype="float32"
    )

    span = bundle.write_frame(frame, _coordinate(1, 0, 4800))
    assert span is not None
    assert (span.bundle_sample_start, span.bundle_sample_end) == (0, 1440)
    bundle.close()

    assert validate_manifest(manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["sample_count"] == 1440
    assert payload["tracks"]["model_pre_gain_tap"]["samples"] == 1468
    frame_record = _timeline(timeline)[1]
    assert frame_record["track_ranges"]["model_pre_gain_tap"] == {
        "sample_start": 0,
        "sample_end": 1468,
    }
    assert frame_record["track_ranges"]["model_gate_pcm"] == {
        "sample_start": 0,
        "sample_end": 1440,
    }
    with wave.open(str(tracks[DiagnosticTrack.MODEL_PRE_GAIN_TAP]), "rb") as rec:
        assert rec.getnframes() == 1468


def test_periodic_flush_leaves_readable_wavs_and_timeline_before_close(
    tmp_path: Path,
) -> None:
    tracks, timeline, _manifest = _paths(tmp_path)
    bundle = _bundle(tmp_path, flush_sec=0.02)
    assert bundle.write_frame(_frame(32), _coordinate(1, 0, 96)) is not None

    deadline = time.monotonic() + 2.0
    observed = False
    while time.monotonic() < deadline:
        time.sleep(0.02)
        try:
            with wave.open(str(next(iter(tracks.values()))), "rb") as recording:
                wav_frames = recording.getnframes()
            kinds = [record["kind"] for record in _timeline(timeline)]
            if wav_frames == 32 and "frame" in kinds:
                observed = True
                break
        except (EOFError, json.JSONDecodeError, wave.Error):
            continue
    assert observed, "periodic flush must make the in-progress evidence readable"
    bundle.close()


def test_files_are_private_and_leaf_symlinks_or_existing_paths_are_never_followed(
    tmp_path: Path,
) -> None:
    tracks, timeline, manifest = _paths(tmp_path)
    previous = os.umask(0)
    try:
        bundle = _bundle(tmp_path)
        assert bundle.write_frame(_frame(4), _coordinate(1, 0, 12)) is not None
        bundle.close()
    finally:
        os.umask(previous)
    for path in [
        *tracks.values(),
        bundle.final_model_input_path,
        timeline,
        manifest,
    ]:
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    occupied_tracks, occupied_timeline, occupied_manifest = _paths(occupied)
    sentinel = occupied_tracks[DiagnosticTrack.MODEL_GATE_PCM]
    sentinel.write_bytes(b"do-not-overwrite")
    with pytest.raises(FileExistsError):
        SynchronizedDiagnosticBundle(
            occupied_tracks,
            occupied_timeline,
            occupied_manifest,
            16_000,
        )
    assert sentinel.read_bytes() == b"do-not-overwrite"

    symlinked = tmp_path / "symlinked"
    symlinked.mkdir()
    linked_tracks, linked_timeline, linked_manifest = _paths(symlinked)
    target = symlinked / "target.wav"
    target.write_bytes(b"target-stays")
    linked_tracks[DiagnosticTrack.MODEL_ASR_TAP].symlink_to(target)
    with pytest.raises(FileExistsError):
        SynchronizedDiagnosticBundle(
            linked_tracks,
            linked_timeline,
            linked_manifest,
            16_000,
        )
    assert target.read_bytes() == b"target-stays"


def test_manifest_publication_is_no_clobber_even_if_path_appears_during_run(
    tmp_path: Path,
) -> None:
    _tracks, _timeline_path, manifest = _paths(tmp_path)
    bundle = _bundle(tmp_path)
    assert bundle.write_frame(_frame(4), _coordinate(1, 0, 12)) is not None
    manifest.write_bytes(b"foreign-manifest")

    with pytest.raises(FileExistsError):
        bundle.close()

    assert manifest.read_bytes() == b"foreign-manifest"
    assert bundle.manifest_status is DiagnosticManifestStatus.PUBLISH_FAILED
    with pytest.raises(FileExistsError):
        bundle.close()


def test_manifest_late_publish_error_accepts_exact_committed_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tracks, _timeline_path, manifest = _paths(tmp_path)
    bundle = _bundle(tmp_path)
    assert bundle.write_frame(_frame(4), _coordinate(1, 0, 12)) is not None
    original = diagnostic_bundle._publish_manifest

    def publish_then_raise(path, payload) -> None:
        original(path, payload)
        raise OSError("late cleanup failure")

    monkeypatch.setattr(
        diagnostic_bundle, "_publish_manifest", publish_then_raise
    )

    assert bundle.close() == manifest
    assert bundle.manifest_status is DiagnosticManifestStatus.COMPLETE
    assert "manifest_publish_error" not in bundle.failure_codes
    assert validate_manifest(manifest)


def test_manifest_precommit_failure_cleans_temp_and_stays_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tracks, _timeline_path, manifest = _paths(tmp_path)
    bundle = _bundle(tmp_path)
    assert bundle.write_frame(_frame(4), _coordinate(1, 0, 12)) is not None

    def fail_fdopen(_fd, _mode, *args, **kwargs):
        raise OSError("temp open failure")

    monkeypatch.setattr(diagnostic_bundle.os, "fdopen", fail_fdopen)

    with pytest.raises(OSError, match="temp open failure"):
        bundle.close()
    assert bundle.manifest_status is DiagnosticManifestStatus.PUBLISH_FAILED
    assert not manifest.exists()
    assert not list(tmp_path.glob(f".{manifest.name}.*.tmp"))
    with pytest.raises(OSError, match="temp open failure"):
        bundle.close()


def test_manifest_destination_race_at_atomic_commit_never_clobbers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tracks, _timeline_path, manifest = _paths(tmp_path)
    bundle = _bundle(tmp_path)
    assert bundle.write_frame(_frame(4), _coordinate(1, 0, 12)) is not None
    foreign = b"foreign-at-commit"

    if os.name == "nt":
        original_commit = diagnostic_bundle.os.rename

        def collide(source, destination, *args, **kwargs):
            Path(destination).write_bytes(foreign)
            return original_commit(source, destination, *args, **kwargs)

        monkeypatch.setattr(diagnostic_bundle.os, "rename", collide)
    else:
        original_commit = diagnostic_bundle.os.link

        def collide(source, destination, *args, **kwargs):
            Path(destination).write_bytes(foreign)
            return original_commit(source, destination, *args, **kwargs)

        monkeypatch.setattr(diagnostic_bundle.os, "link", collide)

    with pytest.raises(FileExistsError):
        bundle.close()
    assert manifest.read_bytes() == foreign
    assert bundle.manifest_status is DiagnosticManifestStatus.PUBLISH_FAILED
    assert not list(tmp_path.glob(f".{manifest.name}.*.tmp"))


def test_manifest_temp_cleanup_failure_after_commit_keeps_success_authoritative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tracks, _timeline_path, manifest = _paths(tmp_path)
    bundle = _bundle(tmp_path)
    assert bundle.write_frame(_frame(4), _coordinate(1, 0, 12)) is not None
    original_unlink = Path.unlink

    def fail_manifest_temp_unlink(path: Path, *args, **kwargs):
        if path.name.startswith(f".{manifest.name}.") and path.name.endswith(
            ".tmp"
        ):
            raise OSError("temp cleanup failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_manifest_temp_unlink)

    assert bundle.close() == manifest
    assert bundle.manifest_status is DiagnosticManifestStatus.COMPLETE
    assert validate_manifest(manifest)
    residues = list(tmp_path.glob(f".{manifest.name}.*.tmp"))
    assert len(residues) == 1
    assert residues[0].is_file()
    assert stat.S_IMODE(residues[0].stat().st_mode) == 0o600


def test_writer_failure_before_close_is_terminal_and_never_reopens(
    tmp_path: Path,
) -> None:
    _tracks, _timeline_path, manifest = _paths(tmp_path)
    bundle = _bundle(tmp_path)

    def fail_get(*args, **kwargs):
        raise OSError("writer loop failure")

    bundle._queue.get = fail_get  # type: ignore[method-assign]
    assert bundle._close_done.wait(timeout=1.0)
    assert bundle.manifest_status is DiagnosticManifestStatus.PUBLISH_FAILED

    with pytest.raises(OSError, match="writer loop failure"):
        bundle.close()
    assert bundle.manifest_status is DiagnosticManifestStatus.PUBLISH_FAILED
    assert not manifest.exists()


def test_context_exit_preserves_application_exception_over_publish_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ApplicationFailure(RuntimeError):
        pass

    bundle = _bundle(tmp_path)

    def fail_publish(_path, _payload) -> None:
        raise OSError("publish failure")

    monkeypatch.setattr(diagnostic_bundle, "_publish_manifest", fail_publish)

    with pytest.raises(ApplicationFailure, match="application sentinel"):
        with bundle:
            assert (
                bundle.write_frame(_frame(4), _coordinate(1, 0, 12))
                is not None
            )
            raise ApplicationFailure("application sentinel")
    assert bundle.manifest_status is DiagnosticManifestStatus.PUBLISH_FAILED
    with pytest.raises(OSError, match="publish failure"):
        bundle.close()


def test_delayed_async_publish_failure_transitions_from_finalizing_to_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracks, timeline, manifest = _paths(tmp_path)
    bundle = SynchronizedDiagnosticBundle(
        tracks,
        timeline,
        manifest,
        sample_rate=16_000,
        close_timeout_sec=0.05,
    )
    assert bundle.write_frame(_frame(4), _coordinate(1, 0, 12)) is not None
    entered = threading.Event()
    release = threading.Event()

    def blocked_failure(_path, _payload) -> None:
        entered.set()
        assert release.wait(timeout=2.0)
        raise OSError("delayed publish failure")

    monkeypatch.setattr(
        diagnostic_bundle, "_publish_manifest", blocked_failure
    )
    try:
        assert bundle.close() == manifest
        assert entered.wait(timeout=1.0)
        assert bundle.manifest_status is DiagnosticManifestStatus.FINALIZING
        assert not manifest.exists()
    finally:
        release.set()
    assert bundle._close_done.wait(timeout=1.0)
    assert bundle.manifest_status is DiagnosticManifestStatus.PUBLISH_FAILED
    with pytest.raises(OSError, match="delayed publish failure"):
        bundle.close()


def test_manifest_validation_fails_closed_when_missing_or_artifact_mutates(
    tmp_path: Path,
) -> None:
    tracks, _timeline_path, manifest = _paths(tmp_path)
    assert not validate_manifest(manifest)
    bundle = _bundle(tmp_path)
    assert bundle.write_frame(_frame(8), _coordinate(1, 0, 24)) is not None
    bundle.close()
    assert validate_manifest(manifest)

    path = tracks[DiagnosticTrack.MODEL_GATE_PCM]
    with path.open("r+b") as handle:
        handle.seek(-1, os.SEEK_END)
        original = handle.read(1)
        handle.seek(-1, os.SEEK_END)
        handle.write(bytes([original[0] ^ 0x01]))
    assert not validate_manifest(manifest)


def test_oversized_timeline_is_rejected_before_opening_a_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    timeline = tmp_path / "oversized.timeline.jsonl"
    with timeline.open("wb") as handle:
        handle.truncate((64 << 20) + 1)
    timeline.chmod(0o600)
    reader_opened = False

    def forbidden_reader(*_args, **_kwargs):
        nonlocal reader_opened
        reader_opened = True
        raise AssertionError("oversized timeline must be rejected before reading")

    monkeypatch.setattr(diagnostic_bundle.os, "fdopen", forbidden_reader)
    monkeypatch.setattr("builtins.open", forbidden_reader)

    assert not diagnostic_bundle._validate_timeline(
        timeline,
        frame_count=1,
        sample_count=1,
        track_sample_counts={role.value: 1 for role in DiagnosticTrack},
        sample_rate_hz=16_000,
        endpoint_replay_config=EndpointReplayConfig(),
    )
    assert reader_opened is False


def test_schema_v2_timeline_replacement_after_hash_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tracks, timeline, manifest = _paths(tmp_path)
    bundle = _bundle(tmp_path)
    assert bundle.write_frame(_frame(4), _coordinate(1, 0, 12)) is not None
    bundle.close()
    assert validate_manifest(manifest)
    replacement = tmp_path / "replacement.timeline.jsonl"
    replacement.write_bytes(timeline.read_bytes())
    replacement.chmod(0o600)
    original_hash = diagnostic_bundle._hash_private_regular_snapshot
    replaced = False

    def hash_then_replace(path: Path):
        nonlocal replaced
        result = original_hash(path)
        if path == timeline and not replaced:
            os.replace(replacement, timeline)
            replaced = True
        return result

    monkeypatch.setattr(
        diagnostic_bundle, "_hash_private_regular_snapshot", hash_then_replace
    )

    assert not validate_manifest(manifest)
    assert replaced


def test_same_size_valid_wav_replacement_after_hash_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracks, _timeline_path, manifest = _paths(tmp_path)
    bundle = _bundle(tmp_path)
    assert bundle.write_frame(_frame(4), _coordinate(1, 0, 12)) is not None
    bundle.close()
    assert validate_manifest(manifest)
    target = tracks[DiagnosticTrack.MODEL_GATE_PCM]
    original_payload = target.read_bytes()
    replacement_payload = bytearray(original_payload)
    replacement_payload[-1] ^= 0x01
    replacement = tmp_path / "replacement.wav"
    replacement.write_bytes(replacement_payload)
    replacement.chmod(0o600)
    assert replacement.stat().st_size == target.stat().st_size
    with wave.open(str(replacement), "rb") as recording:
        assert recording.getnframes() == 4
    original_hash = diagnostic_bundle._hash_private_regular_snapshot
    replaced = False

    def hash_then_replace(path: Path):
        nonlocal replaced
        result = original_hash(path)
        if path == target and not replaced:
            os.replace(replacement, target)
            replaced = True
        return result

    monkeypatch.setattr(
        diagnostic_bundle, "_hash_private_regular_snapshot", hash_then_replace
    )

    assert not validate_manifest(manifest)
    assert replaced


def test_timeline_replacement_after_wav_validation_is_rejected_by_final_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tracks, timeline, manifest = _paths(tmp_path)
    bundle = _bundle(tmp_path)
    assert bundle.write_frame(_frame(4), _coordinate(1, 0, 12)) is not None
    bundle.close()
    assert validate_manifest(manifest)
    replacement = tmp_path / "late-replacement.timeline.jsonl"
    replacement.write_bytes(timeline.read_bytes())
    replacement.chmod(0o600)
    original_validator = diagnostic_bundle._validate_wav_artifacts
    replaced = False

    def validate_then_replace(*args, **kwargs):
        nonlocal replaced
        result = original_validator(*args, **kwargs)
        assert result
        os.replace(replacement, timeline)
        replaced = True
        return result

    monkeypatch.setattr(
        diagnostic_bundle, "_validate_wav_artifacts", validate_then_replace
    )

    assert not validate_manifest(manifest)
    assert replaced


@pytest.mark.parametrize("race", ("mutate", "replace"))
def test_manifest_change_during_last_artifact_parser_is_rejected_by_final_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, race: str
) -> None:
    _tracks, _timeline_path, manifest = _paths(tmp_path)
    bundle = _bundle(tmp_path)
    assert bundle.write_frame(_frame(4), _coordinate(1, 0, 12)) is not None
    bundle.close()
    assert validate_manifest(manifest)
    original_manifest = manifest.read_bytes()
    replacement = tmp_path / "replacement.manifest.json"
    replacement.write_bytes(original_manifest)
    replacement.chmod(0o600)
    initial_metadata = manifest.stat()
    original_validator = diagnostic_bundle._validate_wav_artifacts
    raced = False

    def validate_during_manifest_race(*args, **kwargs):
        nonlocal raced
        result = original_validator(*args, **kwargs)
        assert result
        if race == "replace":
            os.replace(replacement, manifest)
        else:
            assert original_manifest.endswith(b"\n")
            manifest.write_bytes(original_manifest[:-1] + b" ")
            os.utime(
                manifest,
                ns=(
                    initial_metadata.st_atime_ns,
                    initial_metadata.st_mtime_ns + 1_000_000_000,
                ),
            )
        raced = True
        return result

    monkeypatch.setattr(
        diagnostic_bundle,
        "_validate_wav_artifacts",
        validate_during_manifest_race,
    )

    assert not validate_manifest(manifest)
    assert raced


def test_timeline_record_count_cap_rejects_an_otherwise_valid_schema_v2_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tracks, _timeline_path, manifest = _paths(tmp_path)
    bundle = _bundle(tmp_path)
    assert bundle.write_frame(_frame(4), _coordinate(1, 0, 12)) is not None
    bundle.close()
    assert validate_manifest(manifest)

    monkeypatch.setattr(diagnostic_bundle, "_TIMELINE_MAX_RECORDS", 2)

    assert not validate_manifest(manifest)


def test_schema_v1_timeline_replacement_after_hash_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tracks, timeline, manifest = _paths(tmp_path)
    bundle = _bundle(tmp_path)
    assert bundle.write_frame(_frame(4), _coordinate(1, 0, 12)) is not None
    bundle.close()
    _downgrade_to_schema_v1(timeline, manifest)
    assert validate_manifest(manifest)
    replacement = tmp_path / "legacy-replacement.timeline.jsonl"
    replacement.write_bytes(timeline.read_bytes())
    replacement.chmod(0o600)
    original_hash = diagnostic_bundle._hash_private_regular_snapshot
    replaced = False

    def hash_then_replace(path: Path):
        nonlocal replaced
        result = original_hash(path)
        if path == timeline and not replaced:
            os.replace(replacement, timeline)
            replaced = True
        return result

    monkeypatch.setattr(
        diagnostic_bundle, "_hash_private_regular_snapshot", hash_then_replace
    )

    assert not validate_manifest(manifest)
    assert replaced


def test_schema_v1_timeline_size_cap_precedes_artifact_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tracks, timeline, manifest = _paths(tmp_path)
    bundle = _bundle(tmp_path)
    assert bundle.write_frame(_frame(4), _coordinate(1, 0, 12)) is not None
    bundle.close()
    _downgrade_to_schema_v1(timeline, manifest)
    assert validate_manifest(manifest)
    monkeypatch.setattr(
        diagnostic_bundle, "_TIMELINE_MAX_BYTES", timeline.stat().st_size - 1
    )
    hashed = False

    def forbidden_hash(_path: Path):
        nonlocal hashed
        hashed = True
        raise AssertionError("oversized legacy timeline must fail before hashing")

    monkeypatch.setattr(
        diagnostic_bundle, "_hash_private_regular_snapshot", forbidden_hash
    )

    assert not validate_manifest(manifest)
    assert not hashed


def test_schema_v1_timeline_record_count_cap_is_enforced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tracks, timeline, manifest = _paths(tmp_path)
    bundle = _bundle(tmp_path)
    assert bundle.write_frame(_frame(4), _coordinate(1, 0, 12)) is not None
    bundle.close()
    _downgrade_to_schema_v1(timeline, manifest)
    assert validate_manifest(manifest)

    monkeypatch.setattr(diagnostic_bundle, "_TIMELINE_MAX_RECORDS", 2)

    assert not validate_manifest(manifest)


def test_observation_schema_rejects_canary_text_and_never_persists_it(
    tmp_path: Path,
) -> None:
    _tracks, timeline, manifest = _paths(tmp_path)
    bundle = _bundle(tmp_path)
    span = bundle.write_frame(_frame(4), _coordinate(1, 0, 12))
    assert span is not None
    canary = "CANARY private transcript sentence must not persist"

    with pytest.raises(TypeError):
        DiagnosticObservation(  # type: ignore[call-arg]
            stage=DiagnosticStage.ASR_PARTIAL,
            monotonic_ns=1,
            span=span,
            text=canary,
        )
    with pytest.raises(ValueError, match="not free-form text"):
        DiagnosticObservation(
            stage=DiagnosticStage.FINAL_SELECTION,
            monotonic_ns=2,
            span=span,
            outcome=canary,
        )
    with pytest.raises(ValueError, match="recognized closed symbol"):
        DiagnosticObservation(
            stage=DiagnosticStage.FINAL_SELECTION,
            monotonic_ns=2,
            outcome="private_canary_transcript",
        )
    with pytest.raises(ValueError, match="opaque token"):
        DiagnosticObservation(
            stage=DiagnosticStage.ASR_PARTIAL,
            monotonic_ns=3,
            span=span,
            utterance_id=canary,
        )

    assert bundle.observe(
        DiagnosticObservation(
            stage=DiagnosticStage.VAD_TRANSITION,
            monotonic_ns=4,
            span=span,
            stream_id="sherpa-11111111111111111111111111111111",
            utterance_id="u1",
            boolean_value=True,
        )
    )
    bundle.close()
    persisted = timeline.read_text(encoding="utf-8") + manifest.read_text(
        encoding="utf-8"
    )
    assert canary not in persisted
    lowered = persisted.lower()
    assert '"text"' not in lowered
    assert "transcript_hash" not in lowered
    assert "transcript_length" not in lowered
    assert validate_manifest(manifest)


def test_required_finalizer_playback_and_barge_stages_are_closed_and_persisted(
    tmp_path: Path,
) -> None:
    _tracks, timeline, manifest = _paths(tmp_path)
    bundle = _bundle(tmp_path)
    span = bundle.write_frame(_frame(4), _coordinate(1, 0, 12))
    assert span is not None
    observations = (
        DiagnosticObservation(
            stage=DiagnosticStage.ENDPOINT_EVALUATED,
            monotonic_ns=8,
            span=span,
            stream_id="sherpa-11111111111111111111111111111111",
            utterance_id="u1",
            reason="asr",
            basis="acoustic",
            completion_state="unavailable",
            acoustic_endpoint=True,
            vad_active=False,
            early_endpoint_allowed=True,
            trailing_silence_sec=0.8,
            boolean_value=True,
        ),
        DiagnosticObservation(
            stage=DiagnosticStage.ENDPOINT_COMMITTED,
            monotonic_ns=8,
            span=span,
            stream_id="sherpa-11111111111111111111111111111111",
            utterance_id="u1",
            revision=1,
            reason="asr",
            basis="acoustic",
            completion_state="unavailable",
        ),
        DiagnosticObservation(
            stage=DiagnosticStage.ASR_STREAMING_FINAL,
            monotonic_ns=9,
            span=span,
            stream_id="sherpa-11111111111111111111111111111111",
            utterance_id="u1",
            revision=1,
            outcome="nonempty",
        ),
        DiagnosticObservation(
            stage=DiagnosticStage.FINALIZER_QUEUED,
            monotonic_ns=10,
            span=span,
            stream_id="sherpa-11111111111111111111111111111111",
            utterance_id="u1",
            revision=1,
        ),
        DiagnosticObservation(
            stage=DiagnosticStage.FINALIZER_STARTED,
            monotonic_ns=11,
            span=span,
            stream_id="sherpa-11111111111111111111111111111111",
            utterance_id="u1",
            revision=1,
        ),
        DiagnosticObservation(
            stage=DiagnosticStage.FINAL_SELECTION,
            monotonic_ns=12,
            span=span,
            stream_id="sherpa-11111111111111111111111111111111",
            utterance_id="u1",
            revision=1,
            selected_source="streaming",
            outcome="unavailable",
            support=0,
            boolean_value=False,
        ),
        DiagnosticObservation(
            stage=DiagnosticStage.FINAL_DISPATCHED,
            monotonic_ns=13,
            span=span,
            stream_id="sherpa-11111111111111111111111111111111",
            utterance_id="u1",
            revision=1,
            outcome="callback_returned",
        ),
        DiagnosticObservation(
            stage=DiagnosticStage.PLAYBACK_ADMITTED,
            monotonic_ns=14,
            correlation_id="playback-7",
            agent_session_id="session-22222222222222222222222222222222",
            agent_turn_id="turn-1",
            agent_revision=0,
            playback_kind="auxiliary",
        ),
        DiagnosticObservation(
            stage=DiagnosticStage.PLAYBACK_STARTED,
            monotonic_ns=15,
            correlation_id="playback-7",
            agent_session_id="session-22222222222222222222222222222222",
            agent_turn_id="turn-1",
            agent_revision=0,
        ),
        DiagnosticObservation(
            stage=DiagnosticStage.PLAYBACK_TERMINAL,
            monotonic_ns=16,
            correlation_id="playback-7",
            agent_session_id="session-22222222222222222222222222222222",
            agent_turn_id="turn-1",
            agent_revision=0,
            outcome="completed",
        ),
        DiagnosticObservation(
            stage=DiagnosticStage.BARGE_DETECTED,
            monotonic_ns=17,
            span=span,
            stream_id="sherpa-11111111111111111111111111111111",
            utterance_id="u1",
            revision=2,
        ),
    )
    for observation in observations:
        assert bundle.observe(observation)
        if observation.stage is DiagnosticStage.FINALIZER_STARTED:
            assert bundle.write_final_model_input(
                np.array([0.125, -0.25], dtype="float32"),
                stream_id="sherpa-11111111111111111111111111111111",
                utterance_id="u1",
                capture_epoch=3,
                capture_generation=4,
                revision=1,
                role=FinalModelInputRole.MODEL_GATE_SEGMENT,
            ) is not None
    bundle.close()

    observed = [
        record for record in _timeline(timeline) if record["kind"] == "observation"
    ]
    assert [record["stage"] for record in observed] == [
        observation.stage.value for observation in observations
    ]
    assert validate_manifest(manifest)


@pytest.mark.parametrize("mutation", ("missing", "after_started"))
def test_finalizer_lifecycle_requires_streaming_final_before_worker_events(
    tmp_path: Path, mutation: str
) -> None:
    _tracks, timeline, manifest = _paths(tmp_path)
    bundle = _bundle(tmp_path)
    span = bundle.write_frame(_frame(4), _coordinate(1, 0, 12))
    assert span is not None
    _complete_turn(
        bundle,
        span,
        utterance_id="u1",
        revision=1,
        samples=np.array([0.125, -0.25], dtype="float32"),
        role=FinalModelInputRole.MODEL_GATE_SEGMENT,
    )
    bundle.close()
    assert validate_manifest(manifest)

    records = _timeline(timeline)
    streaming_final = next(
        record
        for record in records
        if record.get("stage") == DiagnosticStage.ASR_STREAMING_FINAL.value
    )
    records.remove(streaming_final)
    if mutation == "after_started":
        started_index = next(
            index
            for index, record in enumerate(records)
            if record.get("stage") == DiagnosticStage.FINALIZER_STARTED.value
        )
        records.insert(started_index + 1, streaming_final)
    _rewrite_timeline_and_bind(timeline, manifest, records)

    assert not validate_manifest(manifest)


def test_async_queue_observation_may_follow_worker_terminal(tmp_path: Path) -> None:
    _tracks, _timeline_path, manifest = _paths(tmp_path)
    bundle = _bundle(tmp_path)
    span = bundle.write_frame(_frame(4), _coordinate(1, 0, 12))
    assert span is not None
    _complete_turn(
        bundle,
        span,
        utterance_id="u1",
        revision=1,
        samples=np.array([0.125, -0.25], dtype="float32"),
        role=FinalModelInputRole.MODEL_GATE_SEGMENT,
    )
    # The capture thread records this after offer_nowait() returns. The worker
    # can serialize its complete lifecycle first without violating queue causality.
    assert bundle.observe(
        DiagnosticObservation(
            stage=DiagnosticStage.FINALIZER_QUEUED,
            monotonic_ns=26,
            span=span,
            stream_id="sherpa-11111111111111111111111111111111",
            utterance_id="u1",
            revision=1,
        )
    )
    bundle.close()

    assert validate_manifest(manifest)


def test_endpoint_policy_decision_is_replayed_and_tampering_fails_closed(
    tmp_path: Path,
) -> None:
    _tracks, timeline, manifest = _paths(tmp_path)
    bundle = _bundle(tmp_path)
    span = bundle.write_frame(_frame(4), _coordinate(1, 0, 12))
    assert span is not None
    assert bundle.observe(
        DiagnosticObservation(
            stage=DiagnosticStage.ENDPOINT_EVALUATED,
            monotonic_ns=8,
            span=span,
            reason="unknown",
            basis="acoustic",
            completion_state="unavailable",
            acoustic_endpoint=False,
            vad_active=False,
            early_endpoint_allowed=True,
            trailing_silence_sec=0.1,
            boolean_value=False,
        )
    )
    bundle.close()
    assert validate_manifest(manifest)

    records = _timeline(timeline)
    evaluation = next(
        record
        for record in records
        if record.get("stage") == DiagnosticStage.ENDPOINT_EVALUATED.value
    )
    evaluation["boolean_value"] = True
    _rewrite_timeline_and_bind(timeline, manifest, records)
    assert not validate_manifest(manifest)


def test_adaptive_endpoint_snapshot_and_pause_sequence_replay_exactly(
    tmp_path: Path,
) -> None:
    tracks, timeline, manifest = _paths(tmp_path)
    replay = EndpointReplayConfig(
        enabled=True,
        min_silence_sec=0.8,
        max_silence_sec=1.6,
        complete_threshold=0.6,
        incomplete_threshold=0.3,
        high_confidence_floor=0.2,
        high_confidence_score=0.75,
        adaptive_floor=True,
        pause_window=2,
        pause_quantile=1.0,
        pause_margin=0.0,
        pause_min_samples=2,
        initial_pause_samples_sec=(0.3, 0.4),
    )
    bundle = SynchronizedDiagnosticBundle(
        tracks,
        timeline,
        manifest,
        sample_rate=16_000,
        endpoint_replay_config=replay,
    )
    span = bundle.write_frame(_frame(4), _coordinate(1, 0, 12))
    assert span is not None
    stream_id = "sherpa-11111111111111111111111111111111"
    observations = (
        DiagnosticObservation(
            stage=DiagnosticStage.ENDPOINT_PAUSE_OBSERVED,
            monotonic_ns=7,
            span=span,
            numeric_value=0.45,
        ),
        DiagnosticObservation(
            stage=DiagnosticStage.ENDPOINT_EVALUATED,
            monotonic_ns=8,
            span=span,
            reason="semantic",
            basis="semantic_early",
            completion_state="scored",
            numeric_value=0.9,
            acoustic_endpoint=False,
            vad_active=False,
            early_endpoint_allowed=True,
            trailing_silence_sec=0.5,
            boolean_value=True,
        ),
        DiagnosticObservation(
            stage=DiagnosticStage.ENDPOINT_COMMITTED,
            monotonic_ns=8,
            span=span,
            stream_id=stream_id,
            utterance_id="u1",
            revision=1,
            reason="semantic",
            basis="semantic_early",
            completion_state="scored",
            numeric_value=0.9,
        ),
        DiagnosticObservation(
            stage=DiagnosticStage.ASR_STREAMING_FINAL,
            monotonic_ns=9,
            span=span,
            stream_id=stream_id,
            utterance_id="u1",
            revision=1,
            outcome="nonempty",
        ),
        DiagnosticObservation(
            stage=DiagnosticStage.FINALIZER_STARTED,
            monotonic_ns=10,
            span=span,
            stream_id=stream_id,
            utterance_id="u1",
            revision=1,
        ),
        DiagnosticObservation(
            stage=DiagnosticStage.FINAL_SELECTION,
            monotonic_ns=11,
            span=span,
            stream_id=stream_id,
            utterance_id="u1",
            revision=1,
            selected_source="streaming",
            outcome="unavailable",
            support=0,
            boolean_value=False,
        ),
        DiagnosticObservation(
            stage=DiagnosticStage.FINAL_DISPATCHED,
            monotonic_ns=12,
            span=span,
            stream_id=stream_id,
            utterance_id="u1",
            revision=1,
            outcome="callback_returned",
        ),
    )
    for observation in observations:
        assert bundle.observe(observation)
        if observation.stage is DiagnosticStage.FINALIZER_STARTED:
            assert bundle.write_final_model_input(
                np.array([0.125, -0.25], dtype="float32"),
                stream_id=stream_id,
                utterance_id="u1",
                capture_epoch=3,
                capture_generation=4,
                revision=1,
                role=FinalModelInputRole.MODEL_GATE_SEGMENT,
            ) is not None
    bundle.close()

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["endpoint_replay_config"] == replay.to_payload()
    assert _timeline(timeline)[0]["endpoint_replay_config"] == replay.to_payload()
    assert validate_manifest(manifest)

    # With the two-sample window, changing the observed pause changes the
    # learned floor from 0.45 to 0.9. The persisted true decision must then fail
    # replay. This distinguishes both initial-state restoration and event order
    # from the 0.8-second cold start.
    records = _timeline(timeline)
    pause = next(
        record
        for record in records
        if record.get("stage")
        == DiagnosticStage.ENDPOINT_PAUSE_OBSERVED.value
    )
    pause["numeric_value"] = 0.9
    _rewrite_timeline_and_bind(timeline, manifest, records)
    assert not validate_manifest(manifest)


def test_semantic_hold_reset_v2_validates_through_full_manifest(
    tmp_path: Path,
) -> None:
    tracks, _timeline_path, manifest = _paths(tmp_path)
    replay = EndpointReplayConfig(
        algorithm="adaptive_endpoint_v2",
        enabled=True,
        min_silence_sec=0.7,
        max_silence_sec=1.6,
        complete_threshold=0.6,
        incomplete_threshold=0.3,
        semantic_hold_reset_enabled=True,
        rule3_min_utterance_length_sec=20.0,
    )
    bundle = SynchronizedDiagnosticBundle(
        tracks,
        tmp_path / "run.timeline.jsonl",
        manifest,
        sample_rate=16_000,
        endpoint_replay_config=replay,
    )
    hold_span = bundle.write_frame(_frame(4), _coordinate(1, 0, 12))
    commit_span = bundle.write_frame(_frame(4), _coordinate(2, 12, 24))
    assert hold_span is not None and commit_span is not None
    stream_id = "sherpa-11111111111111111111111111111111"
    utterance_id = "u1"
    observations = (
        DiagnosticObservation(
            stage=DiagnosticStage.ENDPOINT_EVALUATED,
            monotonic_ns=10,
            span=hold_span,
            stream_id=stream_id,
            utterance_id=utterance_id,
            reason="unknown",
            basis="semantic_hold",
            completion_state="scored",
            numeric_value=0.05,
            acoustic_endpoint=True,
            native_acoustic_endpoint=True,
            semantic_hold_active=False,
            hard_boundary=False,
            vad_active=False,
            early_endpoint_allowed=True,
            trailing_silence_sec=0.8,
            utterance_elapsed_sec=1.0,
            boolean_value=False,
        ),
        DiagnosticObservation(
            stage=DiagnosticStage.ENDPOINT_HELD_RESET,
            monotonic_ns=10,
            span=hold_span,
            stream_id=stream_id,
            utterance_id=utterance_id,
            ordinal=1,
            outcome="reset",
        ),
        DiagnosticObservation(
            stage=DiagnosticStage.ENDPOINT_EVALUATED,
            monotonic_ns=20,
            span=commit_span,
            stream_id=stream_id,
            utterance_id=utterance_id,
            reason="asr",
            basis="acoustic",
            completion_state="scored",
            numeric_value=0.05,
            acoustic_endpoint=True,
            native_acoustic_endpoint=False,
            semantic_hold_active=True,
            hard_boundary=False,
            vad_active=False,
            early_endpoint_allowed=True,
            trailing_silence_sec=1.6,
            utterance_elapsed_sec=1.8,
            boolean_value=True,
        ),
        DiagnosticObservation(
            stage=DiagnosticStage.ENDPOINT_COMMITTED,
            monotonic_ns=20,
            span=commit_span,
            stream_id=stream_id,
            utterance_id=utterance_id,
            revision=1,
            reason="asr",
            basis="acoustic",
            completion_state="scored",
            numeric_value=0.05,
        ),
        DiagnosticObservation(
            stage=DiagnosticStage.ASR_STREAMING_FINAL,
            monotonic_ns=21,
            span=commit_span,
            stream_id=stream_id,
            utterance_id=utterance_id,
            revision=1,
            outcome="nonempty",
        ),
        DiagnosticObservation(
            stage=DiagnosticStage.FINALIZER_STARTED,
            monotonic_ns=22,
            span=commit_span,
            stream_id=stream_id,
            utterance_id=utterance_id,
            revision=1,
        ),
    )
    for observation in observations:
        assert bundle.observe(observation)
    assert bundle.write_final_model_input(
        np.array([0.125, -0.25], dtype="float32"),
        stream_id=stream_id,
        utterance_id=utterance_id,
        capture_epoch=3,
        capture_generation=4,
        revision=1,
        role=FinalModelInputRole.MODEL_GATE_SEGMENT,
    ) is not None
    for observation in (
        DiagnosticObservation(
            stage=DiagnosticStage.FINAL_SELECTION,
            monotonic_ns=23,
            span=commit_span,
            stream_id=stream_id,
            utterance_id=utterance_id,
            revision=1,
            selected_source="streaming",
            outcome="unavailable",
            support=0,
            boolean_value=False,
        ),
        DiagnosticObservation(
            stage=DiagnosticStage.FINAL_DISPATCHED,
            monotonic_ns=24,
            span=commit_span,
            stream_id=stream_id,
            utterance_id=utterance_id,
            revision=1,
            outcome="callback_returned",
        ),
    ):
        assert bundle.observe(observation)
    bundle.close()

    assert bundle.manifest_status is DiagnosticManifestStatus.COMPLETE
    assert validate_manifest(manifest)


def test_impossible_adaptive_pause_snapshot_fails_closed(tmp_path: Path) -> None:
    tracks, _timeline_path, manifest = _paths(tmp_path)
    bundle = SynchronizedDiagnosticBundle(
        tracks,
        tmp_path / "run.timeline.jsonl",
        manifest,
        sample_rate=16_000,
        endpoint_replay_config=EndpointReplayConfig(
            enabled=True,
            adaptive_floor=True,
            max_silence_sec=1.6,
            pause_window=2,
            initial_pause_samples_sec=(0.1, 0.4),
        ),
    )
    assert bundle.write_frame(_frame(4), _coordinate(1, 0, 12)) is not None

    bundle.close()

    assert bundle.manifest_status is DiagnosticManifestStatus.INCOMPLETE
    assert "timeline_validation_error" in bundle.failure_codes
    assert not validate_manifest(manifest)


def test_endpoint_replay_rejects_a_second_evaluation_before_pending_commit() -> None:
    def endpoint(
        stage: DiagnosticStage,
        at: int,
        stream: str,
    ) -> dict[str, object]:
        common: dict[str, object] = {
            "kind": "observation",
            "stage": stage.value,
            "monotonic_ns": at,
            "frame_index": at,
            "bundle_sample_start": at * 4,
            "bundle_sample_end": at * 4 + 4,
            "stream_id": stream,
            "utterance_id": "u1",
            "reason": "asr",
            "basis": "acoustic",
            "completion_state": "unavailable",
        }
        if stage is DiagnosticStage.ENDPOINT_EVALUATED:
            common.update(
                acoustic_endpoint=True,
                vad_active=False,
                early_endpoint_allowed=True,
                trailing_silence_sec=0.8,
                boolean_value=True,
            )
        else:
            common["revision"] = 1
        return common

    first = "sherpa-11111111111111111111111111111111"
    second = "sherpa-22222222222222222222222222222222"
    reordered = [
        endpoint(DiagnosticStage.ENDPOINT_EVALUATED, 1, first),
        endpoint(DiagnosticStage.ENDPOINT_EVALUATED, 2, second),
        endpoint(DiagnosticStage.ENDPOINT_COMMITTED, 2, second),
        endpoint(DiagnosticStage.ENDPOINT_COMMITTED, 1, first),
    ]

    assert not diagnostic_bundle._validate_endpoint_replay(
        reordered, EndpointReplayConfig()
    )


def test_missing_turn_terminal_marks_clean_close_incomplete(tmp_path: Path) -> None:
    _tracks, _timeline_path, manifest = _paths(tmp_path)
    bundle = _bundle(tmp_path)
    span = bundle.write_frame(_frame(4), _coordinate(1, 0, 12))
    assert span is not None
    stream_id = "sherpa-11111111111111111111111111111111"
    for observation in (
        DiagnosticObservation(
            stage=DiagnosticStage.ENDPOINT_EVALUATED,
            monotonic_ns=8,
            span=span,
            stream_id=stream_id,
            utterance_id="u1",
            reason="asr",
            basis="acoustic",
            completion_state="unavailable",
            acoustic_endpoint=True,
            vad_active=False,
            early_endpoint_allowed=True,
            trailing_silence_sec=0.8,
            boolean_value=True,
        ),
        DiagnosticObservation(
            stage=DiagnosticStage.ENDPOINT_COMMITTED,
            monotonic_ns=8,
            span=span,
            stream_id=stream_id,
            utterance_id="u1",
            revision=1,
            reason="asr",
            basis="acoustic",
            completion_state="unavailable",
        ),
        DiagnosticObservation(
            stage=DiagnosticStage.ASR_STREAMING_FINAL,
            monotonic_ns=9,
            span=span,
            stream_id=stream_id,
            utterance_id="u1",
            revision=1,
            outcome="nonempty",
        ),
    ):
        assert bundle.observe(observation)
    bundle.close()

    assert bundle.manifest_status is DiagnosticManifestStatus.INCOMPLETE
    assert "timeline_validation_error" in bundle.failure_codes
    assert not validate_manifest(manifest)


def test_reply_playback_must_link_to_a_dispatched_acoustic_turn(tmp_path: Path) -> None:
    _tracks, _timeline_path, manifest = _paths(tmp_path)
    bundle = _bundle(tmp_path)
    assert bundle.write_frame(_frame(4), _coordinate(1, 0, 12)) is not None
    agent = {
        "agent_session_id": "session-22222222222222222222222222222222",
        "agent_turn_id": "turn-1",
        "agent_revision": 0,
    }
    for observation in (
        DiagnosticObservation(
            stage=DiagnosticStage.PLAYBACK_ADMITTED,
            monotonic_ns=10,
            correlation_id="playback-1",
            playback_kind="reply",
            stream_id="sherpa-11111111111111111111111111111111",
            utterance_id="u1",
            revision=1,
            **agent,
        ),
        DiagnosticObservation(
            stage=DiagnosticStage.PLAYBACK_TERMINAL,
            monotonic_ns=11,
            correlation_id="playback-1",
            outcome="dropped",
            **agent,
        ),
    ):
        assert bundle.observe(observation)
    bundle.close()

    assert bundle.manifest_status is DiagnosticManifestStatus.INCOMPLETE
    assert "timeline_validation_error" in bundle.failure_codes
    assert not validate_manifest(manifest)


def test_playback_identity_cannot_change_after_admission(tmp_path: Path) -> None:
    _tracks, timeline, manifest = _paths(tmp_path)
    bundle = _bundle(tmp_path)
    assert bundle.write_frame(_frame(4), _coordinate(1, 0, 12)) is not None
    agent = {
        "agent_session_id": "session-22222222222222222222222222222222",
        "agent_turn_id": "turn-1",
        "agent_revision": 0,
    }
    for observation in (
        DiagnosticObservation(
            stage=DiagnosticStage.PLAYBACK_ADMITTED,
            monotonic_ns=10,
            correlation_id="playback-1",
            playback_kind="auxiliary",
            **agent,
        ),
        DiagnosticObservation(
            stage=DiagnosticStage.PLAYBACK_TERMINAL,
            monotonic_ns=11,
            correlation_id="playback-1",
            outcome="dropped",
            **agent,
        ),
    ):
        assert bundle.observe(observation)
    bundle.close()
    assert validate_manifest(manifest)

    records = _timeline(timeline)
    terminal = next(
        record
        for record in records
        if record.get("stage") == DiagnosticStage.PLAYBACK_TERMINAL.value
    )
    terminal["agent_turn_id"] = "turn-2"
    _rewrite_timeline_and_bind(timeline, manifest, records)
    assert not validate_manifest(manifest)


def test_unclean_shutdown_publishes_a_bound_but_invalid_manifest(tmp_path: Path) -> None:
    _tracks, timeline, manifest = _paths(tmp_path)
    bundle = _bundle(tmp_path)
    assert bundle.write_frame(_frame(4), _coordinate(1, 0, 12)) is not None
    bundle.close(clean_shutdown=False)

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["complete"] is False
    assert payload["clean_shutdown"] is False
    assert payload["failure_codes"] == ["unclean_shutdown"]
    assert _timeline(timeline)[-1]["complete"] is False
    assert bundle.manifest_status is DiagnosticManifestStatus.INCOMPLETE
    assert not validate_manifest(manifest)


def test_clean_close_without_capture_frames_is_never_valid_evidence(
    tmp_path: Path,
) -> None:
    _tracks, _timeline_path, manifest = _paths(tmp_path)
    bundle = _bundle(tmp_path)

    bundle.close(clean_shutdown=True)

    assert bundle.failure_codes == ("no_frames",)
    assert not bundle.complete
    assert not validate_manifest(manifest)


def test_controlled_invalidation_is_private_and_fails_closed(tmp_path: Path) -> None:
    _tracks, timeline, manifest = _paths(tmp_path)
    bundle = _bundle(tmp_path)
    assert bundle.write_frame(_frame(4), _coordinate(1, 0, 12)) is not None

    bundle.invalidate("finalizer_timeout")
    with pytest.raises(ValueError, match="unknown diagnostic invalidation code"):
        bundle.invalidate("private transcript words")
    bundle.close()

    assert bundle.failure_codes == ("finalizer_timeout",)
    assert _timeline(timeline)[-1]["failure_codes"] == ["finalizer_timeout"]
    assert not validate_manifest(manifest)
    with pytest.raises(RuntimeError, match="finalizing or finalized"):
        bundle.invalidate("receipt_timeout")


def test_stalled_writer_cannot_make_close_or_voice_shutdown_unbounded(
    tmp_path: Path,
) -> None:
    tracks, timeline, manifest = _paths(tmp_path)
    bundle = SynchronizedDiagnosticBundle(
        tracks,
        timeline,
        manifest,
        sample_rate=16_000,
        queue_max=1,
        flush_sec=10.0,
        close_timeout_sec=0.05,
    )
    entered = threading.Event()
    release = threading.Event()
    original = bundle._write_frame_item

    def blocked(item) -> None:
        entered.set()
        assert release.wait(timeout=2.0)
        original(item)

    bundle._write_frame_item = blocked  # type: ignore[method-assign]
    assert bundle.write_frame(_frame(8), _coordinate(1, 0, 24)) is not None
    assert entered.wait(timeout=1.0)

    started = time.monotonic()
    assert bundle.close() == manifest
    elapsed = time.monotonic() - started

    assert elapsed < 0.5
    assert bundle.manifest_status is DiagnosticManifestStatus.FINALIZING
    assert bundle.failure_codes == ()
    assert not manifest.exists()
    assert not validate_manifest(manifest)
    release.set()
    assert bundle._close_done.wait(timeout=1.0)
    bundle._thread.join(timeout=1.0)
    assert not bundle._thread.is_alive()
    assert bundle.manifest_status is DiagnosticManifestStatus.COMPLETE
    assert validate_manifest(manifest)


def test_stalled_fsync_keeps_close_bounded_and_eventually_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracks, timeline, manifest = _paths(tmp_path)
    bundle = SynchronizedDiagnosticBundle(
        tracks,
        timeline,
        manifest,
        sample_rate=16_000,
        flush_sec=10.0,
        close_timeout_sec=0.05,
    )
    assert bundle.write_frame(_frame(8), _coordinate(1, 0, 24)) is not None
    deadline = time.monotonic() + 1.0
    while bundle._written_frames != 1 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert bundle._written_frames == 1

    entered = threading.Event()
    release = threading.Event()
    original = diagnostic_bundle.os.fsync
    blocked_once = False

    def blocked_fsync(fd: int) -> None:
        nonlocal blocked_once
        if not blocked_once:
            blocked_once = True
            entered.set()
            assert release.wait(timeout=2.0)
        original(fd)

    monkeypatch.setattr(diagnostic_bundle.os, "fsync", blocked_fsync)
    try:
        started = time.monotonic()
        assert bundle.close() == manifest
        assert time.monotonic() - started < 0.5
        assert entered.wait(timeout=1.0)
        assert bundle.manifest_status is DiagnosticManifestStatus.FINALIZING
        assert not manifest.exists()
    finally:
        release.set()
    assert bundle._close_done.wait(timeout=1.0)
    assert bundle.manifest_status is DiagnosticManifestStatus.COMPLETE
    assert validate_manifest(manifest)


def test_concurrent_close_waiters_share_one_bounded_manifest_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracks, timeline, manifest = _paths(tmp_path)
    bundle = SynchronizedDiagnosticBundle(
        tracks,
        timeline,
        manifest,
        sample_rate=16_000,
        close_timeout_sec=0.05,
    )
    assert bundle.write_frame(_frame(8), _coordinate(1, 0, 24)) is not None
    entered = threading.Event()
    release = threading.Event()
    original = diagnostic_bundle._publish_manifest
    publish_calls = 0

    def blocked_publish(path, payload) -> None:
        nonlocal publish_calls
        publish_calls += 1
        entered.set()
        assert release.wait(timeout=2.0)
        original(path, payload)

    monkeypatch.setattr(diagnostic_bundle, "_publish_manifest", blocked_publish)
    results: list[Path] = []
    waiters = [
        threading.Thread(target=lambda: results.append(bundle.close()))
        for _ in range(2)
    ]
    try:
        for waiter in waiters:
            waiter.start()
        assert entered.wait(timeout=1.0)
        for waiter in waiters:
            waiter.join(timeout=0.5)
            assert not waiter.is_alive()
        assert results == [manifest, manifest]
        assert publish_calls == 1
        assert bundle.manifest_status is DiagnosticManifestStatus.FINALIZING
        assert not manifest.exists()
    finally:
        release.set()
    assert bundle._close_done.wait(timeout=1.0)
    assert publish_calls == 1
    assert bundle.manifest_status is DiagnosticManifestStatus.COMPLETE
    assert validate_manifest(manifest)


@pytest.mark.parametrize(
    ("first_clean", "second_clean", "expected_status"),
    [
        (True, False, DiagnosticManifestStatus.COMPLETE),
        (False, True, DiagnosticManifestStatus.INCOMPLETE),
    ],
)
def test_first_close_request_owns_shutdown_semantics(
    tmp_path: Path,
    first_clean: bool,
    second_clean: bool,
    expected_status: DiagnosticManifestStatus,
) -> None:
    _tracks, _timeline_path, manifest = _paths(tmp_path)
    bundle = _bundle(tmp_path)
    assert bundle.write_frame(_frame(4), _coordinate(1, 0, 12)) is not None

    assert bundle.close(clean_shutdown=first_clean) == manifest
    assert bundle.close(clean_shutdown=second_clean) == manifest

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["clean_shutdown"] is first_clean
    assert bundle.manifest_status is expected_status
    assert validate_manifest(manifest) is first_clean
