from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import stat
import struct

import numpy as np
import pytest

import tools.prepare_notsofar1_natural_overlap_fixture as subject


@dataclass(frozen=True)
class _SyntheticSource:
    root: Path
    injection: subject.TestSourceInjection
    speakers: tuple[str, ...]
    references: tuple[str, ...]


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _wav(*, seconds: int, offset: int) -> bytes:
    samples = seconds * subject.SAMPLE_RATE_HZ
    values = np.arange(samples, dtype=np.int32)
    values = ((values + offset) % 20_001 - 10_000).astype("<i2")
    data = values.tobytes()
    return (
        b"RIFF"
        + struct.pack("<I", 36 + len(data))
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, 16_000, 32_000, 2, 16)
        + b"data"
        + struct.pack("<I", len(data))
        + data
    )


def _segment(
    *,
    speaker: str,
    device: str,
    start: str,
    end: str,
    reference: str,
) -> dict[str, object]:
    first = Decimal(start)
    last = Decimal(end)
    tokens = reference.split()
    step = (last - first) / Decimal(len(tokens) + 1)
    timings = []
    for index, token in enumerate(tokens):
        token_start = first + step * Decimal(index)
        token_end = min(last, token_start + min(step, Decimal("0.050000")))
        timings.append([token, float(token_start), float(token_end)])
    return {
        "ct_wav_file_name": f"close_talk/{device}.wav",
        "end_time": float(last),
        "speaker_id": speaker,
        "start_time": float(first),
        "text": reference,
        "word_timing": timings,
    }


def _write_private(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    path.write_bytes(raw)
    os.chmod(path, 0o600)


def _make_source(
    tmp_path: Path,
    *,
    extra_pair: bool = False,
    first_marker: str | None = None,
    first_segment_duration: Decimal | None = None,
) -> _SyntheticSource:
    root = tmp_path / "source"
    root.mkdir(parents=True, mode=0o700)
    speakers = tuple(f"raw-speaker-private-{index}" for index in range(4))
    devices = tuple(f"raw-close-device-private-{index}" for index in range(4))
    pair_times = [
        ("1.000000", "2.200000", "1.700000", "2.700000"),
        ("4.000000", "5.300000", "4.800000", "5.900000"),
        ("7.000000", "8.400000", "7.900000", "9.000000"),
        ("10.000000", "11.200000", "10.700000", "11.800000"),
        ("13.000000", "14.400000", "13.900000", "15.000000"),
        ("16.000000", "17.200000", "16.700000", "17.900000"),
        ("19.000000", "20.400000", "19.900000", "21.000000"),
    ]
    if first_segment_duration is not None:
        pair_times = [
            (
                "1.000001",
                str(Decimal("1.000001") + first_segment_duration),
                "1.300001",
                "2.000001",
            ),
            ("11.000000", "12.300000", "11.800000", "12.900000"),
            ("14.000000", "15.400000", "14.900000", "16.000000"),
            ("17.000000", "18.200000", "17.700000", "18.800000"),
            ("20.000000", "21.400000", "20.900000", "22.000000"),
            ("23.000000", "24.200000", "23.700000", "24.900000"),
            ("26.000000", "27.400000", "26.900000", "28.000000"),
        ]
    if extra_pair:
        pair_times.append(("22.000000", "23.200000", "22.700000", "23.900000"))
    references: list[str] = []
    segments: list[dict[str, object]] = []
    for index, (a_start, a_end, b_start, b_end) in enumerate(pair_times):
        a_speaker = speakers[index % 4]
        b_speaker = speakers[(index + 1) % 4]
        marker = f" {first_marker}" if index == 0 and first_marker is not None else ""
        a_reference = f"sentinel private{marker} alpha reference {index}"
        b_reference = f"sentinel private bravo reference {index}"
        references.extend((a_reference, b_reference))
        segments.extend(
            (
                _segment(
                    speaker=a_speaker,
                    device=devices[index % 4],
                    start=a_start,
                    end=a_end,
                    reference=a_reference,
                ),
                _segment(
                    speaker=b_speaker,
                    device=devices[(index + 1) % 4],
                    start=b_start,
                    end=b_end,
                    reference=b_reference,
                ),
            )
        )
    if first_segment_duration is not None:
        duration = Decimal("30.000000")
    else:
        duration = Decimal("25.000000") if extra_pair else Decimal("22.000000")
    metadata = {
        "Hashtags": "#DebateOverlaps",
        "MeetingDurationSec": float(duration),
        "MtgType": "mtg",
        "NumParticipants": 4,
        "ParticipantAliasToCtDevice": dict(zip(speakers, devices, strict=True)),
        "ParticipantAliases": list(speakers),
        "Room": "raw-private-room",
        "Topic": "raw-private-topic",
        "meeting_id": "MTG_32006",
    }
    device_rows = [
        {
            "channels_num": 1,
            "device_name": "rockfall_2",
            "is_close_talk": False,
            "is_mc": False,
            "wav_file_names": "sc_rockfall_2/ch0.wav",
        },
        {
            "channels_num": 1,
            "device_name": "meetup_0",
            "is_close_talk": False,
            "is_mc": False,
            "wav_file_names": "sc_meetup_0/ch0.wav",
        },
    ]
    payloads = {
        "gt_transcription.json": _json_bytes(segments),
        "gt_meeting_metadata.json": _json_bytes(metadata),
        "devices.json": _json_bytes(device_rows),
        "sc_meetup_0/ch0.wav": _wav(seconds=int(duration), offset=0),
        "sc_rockfall_2/ch0.wav": _wav(seconds=int(duration), offset=211),
    }
    for relative, raw in payloads.items():
        _write_private(root / relative, raw)
    injection = subject.TestSourceInjection(
        fixture_id="synthetic-notsofar-overlap-v1",
        file_pins={
            relative: (len(raw), hashlib.sha256(raw).hexdigest())
            for relative, raw in payloads.items()
        },
        meeting_duration_seconds=duration,
    )
    return _SyntheticSource(root, injection, speakers, tuple(references))


def _prepare(
    tmp_path: Path, *, extra_pair: bool = False
) -> tuple[_SyntheticSource, Path, subject.PreparedNotsofarNaturalOverlapFixture]:
    source = _make_source(tmp_path, extra_pair=extra_pair)
    output = tmp_path / "published"
    prepared = subject.prepare_notsofar1_natural_overlap_fixture(
        source_dir=source.root,
        output_dir=output,
        accepted_license=subject.LICENSE_ID,
        test_source_injection=source.injection,
    )
    return source, output, prepared


def _manifest(output: Path) -> dict[str, object]:
    return json.loads((output / subject.MANIFEST_FILENAME).read_bytes())


def _replace_private(path: Path, raw: bytes) -> None:
    path.unlink()
    path.write_bytes(raw)
    os.chmod(path, 0o600)


def _set_nested(
    value: object, path: tuple[str | int, ...], replacement: object
) -> None:
    current = value
    for component in path[:-1]:
        current = current[component]  # type: ignore[index]
    current[path[-1]] = replacement  # type: ignore[index]


def _scalar_paths(
    value: object, prefix: tuple[str | int, ...] = ()
) -> list[tuple[tuple[str | int, ...], object]]:
    rows: list[tuple[tuple[str | int, ...], object]] = []
    if type(value) is dict:
        for key, item in value.items():
            rows.extend(_scalar_paths(item, (*prefix, key)))
    elif type(value) is list:
        for index, item in enumerate(value):
            rows.extend(_scalar_paths(item, (*prefix, index)))
    elif type(value) in {str, int, bool}:
        rows.append((prefix, value))
    return rows


def _rebind_receipt_to_manifest(output: Path, manifest: dict[str, object]) -> None:
    manifest_raw = subject._canonical_json(manifest, newline=True)
    manifest_path = output / subject.MANIFEST_FILENAME
    _replace_private(manifest_path, manifest_raw)
    receipt_path = output / subject.PREPARATION_RECEIPT_FILENAME
    receipt = json.loads(receipt_path.read_bytes())
    receipt["fixture_id"] = manifest["fixture_id"]
    receipt["lock_recipe_sha256"] = manifest["lock_recipe_sha256"]
    receipt["overlap_metric"] = manifest["overlap_metric"]
    receipt["production_evidence"] = manifest["production_evidence"]
    receipt["source_contract_sha256"] = manifest["source_contract_sha256"]
    receipt["manifest"] = {
        "file": subject.MANIFEST_FILENAME,
        "sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "size_bytes": len(manifest_raw),
    }
    _replace_private(
        receipt_path,
        subject._canonical_json(receipt, newline=True),
    )


def _assert_private_tree(root: Path) -> None:
    root_metadata = root.stat()
    assert stat.S_IMODE(root_metadata.st_mode) == 0o700
    assert root_metadata.st_uid == os.getuid()
    for path in root.iterdir():
        metadata = path.stat()
        assert path.is_file()
        assert stat.S_IMODE(metadata.st_mode) == 0o600
        assert metadata.st_uid == os.getuid()
        assert metadata.st_nlink == 1


def _selector_segments(
    overlap: Decimal, *, third_annotated_intersection: bool = False
) -> tuple[object, ...]:
    rows = []
    ordinal = 0
    for index in range(7):
        start = Decimal(index * 3 + 1)
        left_end = start + Decimal("1.000000")
        right_start = left_end - overlap
        right_end = right_start + Decimal("1.000000")
        rows.extend(
            (
                subject.inherited._Segment(
                    f"speaker-{index}-a",
                    f"alpha lexical reference {index}",
                    start,
                    left_end,
                    ordinal,
                ),
                subject.inherited._Segment(
                    f"speaker-{index}-b",
                    f"bravo lexical reference {index}",
                    right_start,
                    right_end,
                    ordinal + 1,
                ),
            )
        )
        ordinal += 2
    if third_annotated_intersection:
        rows.append(
            subject.inherited._Segment(
                "speaker-third",
                None,
                Decimal("0.900000"),
                Decimal("1.100000"),
                ordinal,
            )
        )
    return tuple(rows)


def _private_selection_rows(
    windows: tuple[dict[str, object], ...] | list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for window in windows:
        envelope = window["envelope"]
        overlap = window["overlap"]
        role_a = window["role_a"]
        role_b = window["role_b"]
        assert isinstance(envelope, dict)
        assert isinstance(overlap, dict)
        assert isinstance(role_a, dict)
        assert isinstance(role_b, dict)
        rows.append(
            {
                "end_sample": envelope["source_end_sample"],
                "overlap_relative_end_sample": overlap["relative_end_sample"],
                "overlap_relative_start_sample": overlap["relative_start_sample"],
                "reference_a": role_a["reference"],
                "reference_b": role_b["reference"],
                "role_a_relative_end_sample": role_a["activity_relative_end_sample"],
                "role_a_relative_start_sample": role_a[
                    "activity_relative_start_sample"
                ],
                "role_b_relative_end_sample": role_b["activity_relative_end_sample"],
                "role_b_relative_start_sample": role_b[
                    "activity_relative_start_sample"
                ],
                "start_sample": envelope["source_start_sample"],
            }
        )
    return rows


def _independent_private_selection_commitment(
    fixture_id: str,
    rows: list[dict[str, object]],
) -> str:
    payload = {
        "domain": "speaker-notsofar1-natural-overlap-private-selection-commitment-v1",
        "fixture_id": fixture_id,
        "ordered_private_selection_rows": rows,
        "version": 1,
    }
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def test_selector_freezes_decimal_threshold_and_any_third_annotation_veto() -> None:
    exact = subject._select(_selector_segments(Decimal(".300000")))
    assert len(exact) == 7
    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject._select(_selector_segments(Decimal(".299999")))
    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject._select(
            _selector_segments(Decimal(".300000"), third_annotated_intersection=True)
        )


def test_exact_eight_second_unaligned_role_prepares_and_reopens_but_excess_rejects(
    tmp_path: Path,
) -> None:
    source = _make_source(
        tmp_path / "exact",
        first_segment_duration=Decimal("8.000000"),
    )
    output = tmp_path / "exact-output"
    prepared = subject.prepare_notsofar1_natural_overlap_fixture(
        source_dir=source.root,
        output_dir=output,
        accepted_license=subject.LICENSE_ID,
        test_source_injection=source.injection,
    )
    manifest = _manifest(output)
    boundary = next(
        window
        for window in manifest["windows"]
        if window["envelope"]["samples"] == subject.MAX_QUANTIZED_INTERVAL_SAMPLES
    )
    assert boundary["role_a"]["activity_relative_end_sample"] == (
        subject.MAX_QUANTIZED_INTERVAL_SAMPLES
    )
    assert subject.load_notsofar1_natural_overlap_bundle(output) == prepared

    excessive = _make_source(
        tmp_path / "excessive",
        first_segment_duration=Decimal("8.000001"),
    )
    excessive_output = tmp_path / "excessive-output"
    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject.prepare_notsofar1_natural_overlap_fixture(
            source_dir=excessive.root,
            output_dir=excessive_output,
            accepted_license=subject.LICENSE_ID,
            test_source_injection=excessive.injection,
        )
    assert not excessive_output.exists()


def test_rank_preimage_is_ordinal_free_nul_joined_utf8() -> None:
    left = subject.inherited._Segment(
        "speaker-z",
        "alpha reference",
        Decimal("1.000000"),
        Decimal("2.000000"),
        91,
    )
    right = subject.inherited._Segment(
        "speaker-a",
        "bravo reference",
        Decimal("1.200000"),
        Decimal("2.200000"),
        17,
    )
    expected = "18aedfdadda39119c464934c398f6bb4c4897b1cf3399cdc52a8b37c684b179f"
    assert subject._rank(left, right) == expected
    assert subject._rank(right, left) == expected
    assert (
        subject._rank(
            subject.inherited._Segment(
                left.speaker, left.reference, left.start, left.end, 1
            ),
            subject.inherited._Segment(
                right.speaker, right.reference, right.start, right.end, 2
            ),
        )
        == expected
    )


def test_known_marker_is_scoreable_but_unknown_marker_is_ineligible(
    tmp_path: Path,
) -> None:
    known = _make_source(tmp_path / "known", first_marker="<FILL/>")
    prepared = subject.prepare_notsofar1_natural_overlap_fixture(
        source_dir=known.root,
        output_dir=tmp_path / "known-output",
        accepted_license=subject.LICENSE_ID,
        test_source_injection=known.injection,
    )
    assert len(prepared.windows) == 7

    unknown = _make_source(tmp_path / "unknown", first_marker="<UNKNOWN/>")
    output = tmp_path / "unknown-output"
    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject.prepare_notsofar1_natural_overlap_fixture(
            source_dir=unknown.root,
            output_dir=output,
            accepted_license=subject.LICENSE_ID,
            test_source_injection=unknown.injection,
        )
    assert not output.exists()


def test_publishes_exact_grouped_windows_and_original_device_pcm(
    tmp_path: Path,
) -> None:
    source, output, prepared = _prepare(tmp_path)
    manifest = _manifest(output)
    original = {
        "channel-a": np.frombuffer(
            (source.root / "sc_meetup_0/ch0.wav").read_bytes()[44:], dtype="<i2"
        ),
        "channel-b": np.frombuffer(
            (source.root / "sc_rockfall_2/ch0.wav").read_bytes()[44:], dtype="<i2"
        ),
    }

    assert manifest["kind"] == "notsofar1-natural-overlap-paired-device-v1"
    assert manifest["schema_version"] == 1
    assert manifest["production_evidence"] is False
    assert manifest["overlap_metric"] == "not-implemented"
    assert type(manifest["overlap_metric"]) is str
    assert manifest["sample_format"] == {
        "channels": 1,
        "encoding": "f32le",
        "sample_conversion": "pcm-s16le-to-f32le-divide-32768-v1",
        "sample_rate_hz": 16_000,
    }
    assert manifest["evidence_scope"] == {
        "diagnostic_only": True,
        "natural_overlap": True,
        "ordinary_wer": False,
        "original_device_audio": True,
        "promotion_authority": False,
        "qualification_authority": False,
        "speaker_authority": False,
    }
    receipt = json.loads((output / subject.PREPARATION_RECEIPT_FILENAME).read_bytes())
    lock = json.loads(subject.DEFAULT_LOCK.read_bytes())
    assert receipt["overlap_metric"] == manifest["overlap_metric"]
    assert type(receipt["overlap_metric"]) is str
    assert lock["overlap_metric"] == manifest["overlap_metric"]
    assert type(lock["overlap_metric"]) is str
    assert manifest["evidence_scope"]["ordinary_wer"] is False
    assert receipt["evidence_scope"]["ordinary_wer"] is False
    assert lock["evidence_scope"]["ordinary_wer"] is False
    windows = manifest["windows"]
    assert len(windows) == 7
    total_samples = 0
    for index, window in enumerate(windows):
        assert window["window_id"] == f"notsofar-overlap-w{index:02d}"
        assert list(row["channel"] for row in window["channels"]) == [
            "channel-a",
            "channel-b",
        ]
        envelope = window["envelope"]
        overlap = window["overlap"]
        samples = envelope["samples"]
        total_samples += samples
        assert (
            envelope["source_end_sample"] - envelope["source_start_sample"] == samples
        )
        assert (
            overlap["relative_end_sample"] - overlap["relative_start_sample"]
            == overlap["samples"]
        )
        assert overlap["samples"] >= 4_800
        for role in (window["role_a"], window["role_b"]):
            assert (
                0
                <= role["activity_relative_start_sample"]
                < role["activity_relative_end_sample"]
                <= samples
            )
            assert (
                hashlib.sha256(role["reference"].encode()).hexdigest()
                == role["reference_sha256"]
            )
        for channel in window["channels"]:
            assert channel["case_id"] == f"{window['window_id']}-{channel['channel']}"
            assert channel["file"] == f"{channel['case_id']}.f32le"
            raw = (output / channel["file"]).read_bytes()
            assert len(raw) == channel["size_bytes"] == samples * 4
            assert channel["samples"] == samples
            assert hashlib.sha256(raw).hexdigest() == channel["sha256"]
            actual = np.frombuffer(raw, dtype="<f4")
            expected = original[channel["channel"]][
                envelope["source_start_sample"] : envelope["source_end_sample"]
            ].astype(np.float32) / np.float32(32_768)
            assert np.array_equal(actual, expected)
            assert np.isfinite(actual).all()
    assert total_samples * 2 * 4 <= subject.MAX_DERIVED_PCM_BYTES
    assert prepared.path == output / subject.MANIFEST_FILENAME
    assert prepared.digest == hashlib.sha256(prepared.path.read_bytes()).hexdigest()
    loaded = subject.load_notsofar1_natural_overlap_bundle(output)
    assert prepared == loaded
    subject.verify_notsofar1_natural_overlap_bundle(prepared)
    subject.verify_notsofar1_natural_overlap_bundle(loaded)
    _assert_private_tree(output)


def test_public_preparer_has_no_commit_state_authority_parameter() -> None:
    assert tuple(
        inspect.signature(subject.prepare_notsofar1_natural_overlap_fixture).parameters
    ) == (
        "source_dir",
        "output_dir",
        "accepted_license",
        "lock_path",
        "test_source_injection",
    )


def test_bundle_root_named_like_manifest_supports_both_api_shapes(
    tmp_path: Path,
) -> None:
    source = _make_source(tmp_path / "source")
    output = tmp_path / subject.MANIFEST_FILENAME
    prepared = subject.prepare_notsofar1_natural_overlap_fixture(
        source_dir=source.root,
        output_dir=output,
        accepted_license=subject.LICENSE_ID,
        test_source_injection=source.injection,
    )
    assert output.is_dir()
    from_root = subject.load_notsofar1_natural_overlap_bundle(output)
    from_manifest = subject.load_notsofar1_natural_overlap_bundle(prepared.path)
    assert from_root == prepared
    assert from_manifest == prepared
    subject.verify_notsofar1_natural_overlap_bundle(prepared)


def test_rank_selection_is_order_independent_bounded_and_ordinal_free(
    tmp_path: Path,
) -> None:
    source, output, prepared = _prepare(tmp_path, extra_pair=True)
    first = _manifest(output)
    assert len(first["windows"]) == 7

    transcript = source.root / "gt_transcription.json"
    rows = json.loads(transcript.read_bytes())
    transcript.write_bytes(_json_bytes(list(reversed(rows))))
    os.chmod(transcript, 0o600)
    pins = dict(source.injection.file_pins)
    raw = transcript.read_bytes()
    pins["gt_transcription.json"] = (len(raw), hashlib.sha256(raw).hexdigest())
    reordered = subject.TestSourceInjection(
        fixture_id="synthetic-notsofar-overlap-reordered-v1",
        file_pins=pins,
        meeting_duration_seconds=source.injection.meeting_duration_seconds,
    )
    second_output = tmp_path / "reordered"
    second = subject.prepare_notsofar1_natural_overlap_fixture(
        source_dir=source.root,
        output_dir=second_output,
        accepted_license=subject.LICENSE_ID,
        test_source_injection=reordered,
    )
    second_manifest = _manifest(second_output)
    assert first["windows"] == second_manifest["windows"]
    assert prepared.production_evidence is second.production_evidence is False


def test_private_selection_commitment_uses_actual_ordered_reference_strings(
    tmp_path: Path,
) -> None:
    _source, output, _prepared = _prepare(tmp_path)
    manifest = _manifest(output)
    rows = _private_selection_rows(manifest["windows"])
    expected = _independent_private_selection_commitment(
        manifest["fixture_id"],
        rows,
    )
    assert (
        subject._selection_private_commitment_sha256(
            manifest["fixture_id"],
            rows,
        )
        == expected
    )

    forged = json.loads(json.dumps(rows))
    forged[0]["reference_a"] += " forged"
    forged_commitment = _independent_private_selection_commitment(
        manifest["fixture_id"],
        forged,
    )
    assert forged_commitment != expected
    assert (
        subject._selection_private_commitment_sha256(
            manifest["fixture_id"],
            forged,
        )
        == forged_commitment
    )


def test_receipt_stdout_repr_and_public_metadata_are_private(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source, output, prepared = _prepare(tmp_path)
    receipt_raw = (output / subject.PREPARATION_RECEIPT_FILENAME).read_text()
    manifest = _manifest(output)
    public = (
        receipt_raw + repr(prepared) + repr(source.injection) + capsys.readouterr().out
    )
    forbidden = (
        *source.speakers,
        *source.references,
        str(source.root),
        "raw-private-room",
        "raw-private-topic",
        "sc_meetup_0",
        "sc_rockfall_2",
        subject.SELECTION_SEED,
        "ordinal",
        "rank",
        "speaker-slot",
    )
    assert all(value not in public for value in forbidden)
    encoded_manifest = json.dumps(manifest, sort_keys=True)
    assert all(value not in encoded_manifest for value in source.speakers)
    assert "ordinal" not in encoded_manifest
    assert "rank" not in encoded_manifest
    assert "speaker_slot" not in encoded_manifest
    assert (
        prepared.receipt_sha256
        == hashlib.sha256(
            (output / subject.PREPARATION_RECEIPT_FILENAME).read_bytes()
        ).hexdigest()
    )


@pytest.mark.parametrize("bad_license", ["", "CC0-1.0", "cc-by-4.0"])
def test_rejects_wrong_license_before_publication(
    tmp_path: Path, bad_license: str
) -> None:
    source = _make_source(tmp_path)
    output = tmp_path / "published"
    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject.prepare_notsofar1_natural_overlap_fixture(
            source_dir=source.root,
            output_dir=output,
            accepted_license=bad_license,
            test_source_injection=source.injection,
        )
    assert not output.exists()


def test_rejects_hostile_license_equality_object_before_publication(
    tmp_path: Path,
) -> None:
    class _HostileLicense:
        def __eq__(self, _other: object) -> bool:
            return True

        def __ne__(self, _other: object) -> bool:
            return False

    source = _make_source(tmp_path)
    output = tmp_path / "published"
    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject.prepare_notsofar1_natural_overlap_fixture(
            source_dir=source.root,
            output_dir=output,
            accepted_license=_HostileLicense(),  # type: ignore[arg-type]
            test_source_injection=source.injection,
        )
    assert not output.exists()


def test_production_rejects_byte_identical_alternate_lock_path(tmp_path: Path) -> None:
    copied = tmp_path / "copied.lock.json"
    _write_private(copied, subject.DEFAULT_LOCK.read_bytes())
    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject._load_lock(copied, test_source_injection=None)


def test_no_clobber_output_is_rejected(tmp_path: Path) -> None:
    source = _make_source(tmp_path)
    occupied = tmp_path / "occupied"
    occupied.mkdir(mode=0o700)
    marker = occupied / "keep"
    marker.write_text("untouched")
    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject.prepare_notsofar1_natural_overlap_fixture(
            source_dir=source.root,
            output_dir=occupied,
            accepted_license=subject.LICENSE_ID,
            test_source_injection=source.injection,
        )
    assert marker.read_text() == "untouched"


@pytest.mark.parametrize("marker_kind", ["directory", "regular", "symlink", "fifo"])
def test_any_git_marker_type_rejects_output_ancestor(
    tmp_path: Path,
    marker_kind: str,
) -> None:
    source = _make_source(tmp_path / "source")
    git_parent = tmp_path / f"private-git-{marker_kind}"
    git_parent.mkdir(mode=0o700)
    marker = git_parent / ".git"
    if marker_kind == "directory":
        marker.mkdir(mode=0o700)
    elif marker_kind == "regular":
        _write_private(marker, b"gitdir: elsewhere\n")
    elif marker_kind == "symlink":
        marker.symlink_to(tmp_path / "deliberately-absent-git-marker-target")
    else:
        os.mkfifo(marker, mode=0o600)
    output = git_parent / "published"
    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject.prepare_notsofar1_natural_overlap_fixture(
            source_dir=source.root,
            output_dir=output,
            accepted_license=subject.LICENSE_ID,
            test_source_injection=source.injection,
        )
    assert not output.exists()


@pytest.mark.parametrize("marker_kind", ["directory", "regular", "symlink", "fifo"])
def test_loader_and_verify_reject_copied_valid_bundle_under_any_git_marker(
    tmp_path: Path,
    marker_kind: str,
) -> None:
    _source, original, prepared = _prepare(tmp_path / "original")
    git_parent = tmp_path / f"copied-git-{marker_kind}"
    git_parent.mkdir(mode=0o700)
    marker = git_parent / ".git"
    if marker_kind == "directory":
        marker.mkdir(mode=0o700)
    elif marker_kind == "regular":
        _write_private(marker, b"gitdir: elsewhere\n")
    elif marker_kind == "symlink":
        marker.symlink_to(tmp_path / "deliberately-absent-git-target")
    else:
        os.mkfifo(marker, mode=0o600)
    copied = git_parent / "bundle"
    shutil.copytree(original, copied)
    os.chmod(copied, 0o700)
    for path in copied.iterdir():
        os.chmod(path, 0o600)

    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject.load_notsofar1_natural_overlap_bundle(copied)
    copied_loaded = replace(
        prepared,
        path=copied / subject.MANIFEST_FILENAME,
    )
    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject.verify_notsofar1_natural_overlap_bundle(copied_loaded)


def test_self_consistent_lock_tamper_and_production_test_identity_are_rejected(
    tmp_path: Path,
) -> None:
    source = _make_source(tmp_path)
    original = json.loads(subject.DEFAULT_LOCK.read_bytes())
    for label in ("selection", "artifact"):
        lock = json.loads(_json_bytes(original))
        if label == "selection":
            lock["selection"]["minimum_overlap_seconds"] = "0.299999"
        else:
            lock["output_artifacts"][0]["sha256"] = "0" * 64
        body = dict(lock)
        body.pop("recipe_sha256")
        lock["recipe_sha256"] = hashlib.sha256(_json_bytes(body)).hexdigest()
        tampered = tmp_path / f"tampered-{label}.lock.json"
        _write_private(tampered, _json_bytes(lock))
        with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
            subject.prepare_notsofar1_natural_overlap_fixture(
                source_dir=source.root,
                output_dir=tmp_path / f"tampered-{label}-output",
                accepted_license=subject.LICENSE_ID,
                lock_path=tampered,
                test_source_injection=source.injection,
            )

    forged = subject.TestSourceInjection(
        fixture_id=subject.FIXTURE_ID,
        file_pins=source.injection.file_pins,
        meeting_duration_seconds=source.injection.meeting_duration_seconds,
    )
    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject.prepare_notsofar1_natural_overlap_fixture(
            source_dir=source.root,
            output_dir=tmp_path / "forged-output",
            accepted_license=subject.LICENSE_ID,
            test_source_injection=forged,
        )


def test_fully_reanchored_production_lock_with_zero_artifact_pin_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = json.loads(subject.DEFAULT_LOCK.read_bytes())
    lock["output_artifacts"][0]["sha256"] = "0" * 64
    body = dict(lock)
    body.pop("recipe_sha256")
    recipe = hashlib.sha256(_json_bytes(body)).hexdigest()
    lock["recipe_sha256"] = recipe
    zero_pin_lock = tmp_path / "zero-pin.lock.json"
    _write_private(zero_pin_lock, _json_bytes(lock))
    monkeypatch.setattr(subject, "DEFAULT_LOCK", zero_pin_lock)
    monkeypatch.setattr(subject, "EXPECTED_LOCK_RECIPE_SHA256", recipe)

    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject._load_lock(zero_pin_lock, test_source_injection=None)


@pytest.mark.parametrize(
    ("relative", "oversized"),
    [
        ("gt_transcription.json", 1024 * 1024 + 1),
        ("sc_meetup_0/ch0.wav", 16 * 1024 * 1024 + 1),
    ],
)
def test_test_source_injection_keeps_metadata_and_wav_hard_caps(
    tmp_path: Path, relative: str, oversized: int
) -> None:
    source = _make_source(tmp_path)
    pins = dict(source.injection.file_pins)
    pins[relative] = (oversized, pins[relative][1])
    injection = subject.TestSourceInjection(
        fixture_id="synthetic-oversized-source-v1",
        file_pins=pins,
        meeting_duration_seconds=source.injection.meeting_duration_seconds,
    )
    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject._load_lock(
            subject.DEFAULT_LOCK,
            test_source_injection=injection,
        )


def test_test_source_injection_keeps_24_mib_total_source_cap(tmp_path: Path) -> None:
    source = _make_source(tmp_path)
    pins = dict(source.injection.file_pins)
    for relative in ("sc_meetup_0/ch0.wav", "sc_rockfall_2/ch0.wav"):
        pins[relative] = (13 * 1024 * 1024, pins[relative][1])
    injection = subject.TestSourceInjection(
        fixture_id="synthetic-oversized-source-set-v1",
        file_pins=pins,
        meeting_duration_seconds=source.injection.meeting_duration_seconds,
    )
    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject._load_lock(subject.DEFAULT_LOCK, test_source_injection=injection)


def test_source_hash_mode_link_and_close_time_mutation_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _make_source(tmp_path)
    transcript = source.root / "gt_transcription.json"
    transcript.write_bytes(transcript.read_bytes() + b" ")
    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject.prepare_notsofar1_natural_overlap_fixture(
            source_dir=source.root,
            output_dir=tmp_path / "hash-output",
            accepted_license=subject.LICENSE_ID,
            test_source_injection=source.injection,
        )

    mode_source = _make_source(tmp_path / "mode")
    os.chmod(mode_source.root / "devices.json", 0o644)
    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject.prepare_notsofar1_natural_overlap_fixture(
            source_dir=mode_source.root,
            output_dir=tmp_path / "mode-output",
            accepted_license=subject.LICENSE_ID,
            test_source_injection=mode_source.injection,
        )

    linked_source = _make_source(tmp_path / "link")
    os.link(
        linked_source.root / "devices.json",
        linked_source.root / "devices-alias.json",
    )
    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject.prepare_notsofar1_natural_overlap_fixture(
            source_dir=linked_source.root,
            output_dir=tmp_path / "link-output",
            accepted_license=subject.LICENSE_ID,
            test_source_injection=linked_source.injection,
        )

    closing_source = _make_source(tmp_path / "closing")
    original = subject._commit_terminal_receipt

    def mutate_before_link(*args: object, **kwargs: object):
        target = closing_source.root / "devices.json"
        raw = bytearray(target.read_bytes())
        raw[-1] ^= 1
        target.write_bytes(raw)
        os.chmod(target, 0o600)
        return original(*args, **kwargs)

    monkeypatch.setattr(subject, "_commit_terminal_receipt", mutate_before_link)
    output = tmp_path / "closing-output"
    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject.prepare_notsofar1_natural_overlap_fixture(
            source_dir=closing_source.root,
            output_dir=output,
            accepted_license=subject.LICENSE_ID,
            test_source_injection=closing_source.injection,
        )
    assert output.exists()
    assert not (output / subject.PREPARATION_RECEIPT_FILENAME).exists()


def test_source_rejects_symlinked_intermediate_device_directory(tmp_path: Path) -> None:
    source = _make_source(tmp_path)
    directory = source.root / "sc_meetup_0"
    wav = directory / "ch0.wav"
    raw = wav.read_bytes()
    wav.unlink()
    directory.rmdir()
    outside = tmp_path / "outside-private-device"
    outside.mkdir(mode=0o700)
    _write_private(outside / "ch0.wav", raw)
    directory.symlink_to(outside, target_is_directory=True)
    output = tmp_path / "published"
    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject.prepare_notsofar1_natural_overlap_fixture(
            source_dir=source.root,
            output_dir=output,
            accepted_license=subject.LICENSE_ID,
            test_source_injection=source.injection,
        )
    assert not output.exists()


def test_git_ancestor_appearing_before_terminal_receipt_link_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _make_source(tmp_path)
    output = tmp_path / "published"
    original = subject._commit_terminal_receipt

    def add_git_marker_before_link(*args: object, **kwargs: object):
        output_path = args[0]
        assert type(output_path) is type(Path("."))
        _write_private(output_path.parent / ".git", b"gitdir: late\n")
        return original(*args, **kwargs)

    monkeypatch.setattr(subject, "_commit_terminal_receipt", add_git_marker_before_link)
    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject.prepare_notsofar1_natural_overlap_fixture(
            source_dir=source.root,
            output_dir=output,
            accepted_license=subject.LICENSE_ID,
            test_source_injection=source.injection,
        )
    assert output.exists()
    assert not (output / subject.PREPARATION_RECEIPT_FILENAME).exists()


def test_source_root_replaced_during_last_close_guard_read_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _make_source(tmp_path)
    output = tmp_path / "published"
    original = subject._verify_source_root
    root_checks = 0
    renamed = tmp_path / "renamed-source"

    def replace_root_before_last_close_check(*args: object, **kwargs: object) -> None:
        nonlocal root_checks
        root_checks += 1
        if root_checks == len(subject._SOURCE_PATHS) * 4:
            source.root.rename(renamed)
            source.root.mkdir(mode=0o700)
        original(*args, **kwargs)

    monkeypatch.setattr(
        subject,
        "_verify_source_root",
        replace_root_before_last_close_check,
    )
    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject.prepare_notsofar1_natural_overlap_fixture(
            source_dir=source.root,
            output_dir=output,
            accepted_license=subject.LICENSE_ID,
            test_source_injection=source.injection,
        )
    assert root_checks == len(subject._SOURCE_PATHS) * 4
    assert output.exists()
    assert not (output / subject.PREPARATION_RECEIPT_FILENAME).exists()


def test_source_and_loader_reject_fifo_without_blocking(tmp_path: Path) -> None:
    source = _make_source(tmp_path / "source-fifo")
    special = source.root / "devices.json"
    special.unlink()
    os.mkfifo(special, mode=0o600)
    output = tmp_path / "source-fifo-output"
    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject.prepare_notsofar1_natural_overlap_fixture(
            source_dir=source.root,
            output_dir=output,
            accepted_license=subject.LICENSE_ID,
            test_source_injection=source.injection,
        )
    assert not output.exists()

    _source, bundle_root, _prepared = _prepare(tmp_path / "bundle-fifo")
    manifest = _manifest(bundle_root)
    artifact = bundle_root / manifest["windows"][0]["channels"][0]["file"]
    artifact.unlink()
    os.mkfifo(artifact, mode=0o600)
    assert "O_NONBLOCK" in inspect.getsource(subject._read_private)
    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject.load_notsofar1_natural_overlap_bundle(bundle_root)


def test_loader_rejects_artifact_reference_and_receipt_tamper(tmp_path: Path) -> None:
    _source, output, _prepared = _prepare(tmp_path)
    manifest = _manifest(output)
    first_file = output / manifest["windows"][0]["channels"][0]["file"]
    raw = bytearray(first_file.read_bytes())
    raw[0] ^= 1
    first_file.write_bytes(raw)
    os.chmod(first_file, 0o600)
    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject.load_notsofar1_natural_overlap_bundle(output)

    _source, output, _prepared = _prepare(tmp_path / "reference")
    manifest_path = output / subject.MANIFEST_FILENAME
    manifest = _manifest(output)
    manifest["windows"][0]["role_a"]["reference"] += " forged"
    manifest_path.write_bytes(_json_bytes(manifest) + b"\n")
    os.chmod(manifest_path, 0o600)
    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject.load_notsofar1_natural_overlap_bundle(output)

    _source, output, _prepared = _prepare(tmp_path / "receipt")
    receipt = output / subject.PREPARATION_RECEIPT_FILENAME
    raw = bytearray(receipt.read_bytes())
    raw[-2] ^= 1
    receipt.write_bytes(raw)
    os.chmod(receipt, 0o600)
    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject.load_notsofar1_natural_overlap_bundle(output)


def test_overlap_metric_missing_tampered_or_extra_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def mutate(value: dict[str, object], mutation: str) -> None:
        if mutation == "missing":
            value.pop("overlap_metric")
        elif mutation == "tampered":
            value["overlap_metric"] = "implemented"
        else:
            value["unexpected_overlap_metric"] = "not-implemented"

    for mutation in ("missing", "tampered", "extra"):
        _source, output, _prepared = _prepare(tmp_path / f"manifest-{mutation}")
        manifest = _manifest(output)
        mutate(manifest, mutation)
        manifest_raw = subject._canonical_json(manifest, newline=True)
        _replace_private(output / subject.MANIFEST_FILENAME, manifest_raw)
        receipt_path = output / subject.PREPARATION_RECEIPT_FILENAME
        receipt = json.loads(receipt_path.read_bytes())
        receipt["manifest"] = {
            "file": subject.MANIFEST_FILENAME,
            "sha256": hashlib.sha256(manifest_raw).hexdigest(),
            "size_bytes": len(manifest_raw),
        }
        _replace_private(
            receipt_path,
            subject._canonical_json(receipt, newline=True),
        )
        with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
            subject.load_notsofar1_natural_overlap_bundle(output)

        _source, output, _prepared = _prepare(tmp_path / f"receipt-{mutation}")
        receipt_path = output / subject.PREPARATION_RECEIPT_FILENAME
        receipt = json.loads(receipt_path.read_bytes())
        mutate(receipt, mutation)
        _replace_private(
            receipt_path,
            subject._canonical_json(receipt, newline=True),
        )
        with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
            subject.load_notsofar1_natural_overlap_bundle(output)

        lock = json.loads(subject.DEFAULT_LOCK.read_bytes())
        mutate(lock, mutation)
        body = dict(lock)
        body.pop("recipe_sha256")
        recipe = hashlib.sha256(_json_bytes(body)).hexdigest()
        lock["recipe_sha256"] = recipe
        lock_path = tmp_path / f"lock-{mutation}.json"
        _write_private(lock_path, _json_bytes(lock))
        with monkeypatch.context() as scoped:
            scoped.setattr(subject, "DEFAULT_LOCK", lock_path)
            scoped.setattr(subject, "EXPECTED_LOCK_RECIPE_SHA256", recipe)
            with pytest.raises(
                subject.NotsofarNaturalOverlapPreparationError,
                match="^$",
            ):
                subject._load_lock(lock_path, test_source_injection=None)


def test_lock_rejects_every_bool_int_schema_family_before_source_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _make_source(tmp_path / "source")
    original = json.loads(subject.DEFAULT_LOCK.read_bytes())
    mutations = [
        (
            path,
            1
            if type(value) is str
            else (0 if value is False else (1 if value is True else True)),
        )
        for path, value in _scalar_paths(original)
        if path != ("recipe_sha256",)
    ]
    source_opens = 0

    def forbidden_source_open(*_args: object, **_kwargs: object) -> object:
        nonlocal source_opens
        source_opens += 1
        raise AssertionError("invalid lock reached source open")

    monkeypatch.setattr(subject, "_source_root", forbidden_source_open)
    for index, (path, replacement) in enumerate(mutations):
        lock = json.loads(_json_bytes(original))
        _set_nested(lock, path, replacement)
        body = dict(lock)
        body.pop("recipe_sha256")
        recipe = hashlib.sha256(_json_bytes(body)).hexdigest()
        lock["recipe_sha256"] = recipe
        lock_path = tmp_path / f"typed-lock-{index}.json"
        _write_private(lock_path, _json_bytes(lock))
        with monkeypatch.context() as scoped:
            scoped.setattr(subject, "DEFAULT_LOCK", lock_path)
            scoped.setattr(subject, "EXPECTED_LOCK_RECIPE_SHA256", recipe)
            scoped.setattr(subject, "_source_root", forbidden_source_open)
            with pytest.raises(
                subject.NotsofarNaturalOverlapPreparationError, match="^$"
            ):
                subject.prepare_notsofar1_natural_overlap_fixture(
                    source_dir=source.root,
                    output_dir=tmp_path / f"typed-lock-output-{index}",
                    accepted_license=subject.LICENSE_ID,
                    lock_path=lock_path,
                    test_source_injection=source.injection,
                )
    assert source_opens == 0


def test_manifest_and_receipt_reject_all_typed_families_before_pcm_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _source, output, _prepared = _prepare(tmp_path)
    manifest_path = output / subject.MANIFEST_FILENAME
    receipt_path = output / subject.PREPARATION_RECEIPT_FILENAME
    manifest_raw = manifest_path.read_bytes()
    receipt_raw = receipt_path.read_bytes()
    original_manifest = json.loads(manifest_raw)
    original_receipt = json.loads(receipt_raw)

    def replacement_for(value: object) -> object:
        if type(value) is str:
            return 1
        if value is False:
            return 0
        if value is True:
            return 1
        return True

    manifest_mutations = [
        (path, replacement_for(value))
        for path, value in _scalar_paths(original_manifest)
    ]
    receipt_mutations = [
        (path, replacement_for(value))
        for path, value in _scalar_paths(original_receipt)
    ]
    reads: list[str] = []
    original_read = subject._read_private

    def recording_read(*args: object, **kwargs: object):
        name = args[1] if len(args) > 1 else kwargs["name"]
        reads.append(name)
        return original_read(*args, **kwargs)

    monkeypatch.setattr(subject, "_read_private", recording_read)
    for path, replacement in manifest_mutations:
        manifest = json.loads(manifest_raw)
        receipt = json.loads(receipt_raw)
        _set_nested(manifest, path, replacement)
        forged_manifest_raw = subject._canonical_json(manifest, newline=True)
        receipt["manifest"] = {
            "file": subject.MANIFEST_FILENAME,
            "sha256": hashlib.sha256(forged_manifest_raw).hexdigest(),
            "size_bytes": len(forged_manifest_raw),
        }
        if path[0] in {
            "fixture_id",
            "lock_recipe_sha256",
            "overlap_metric",
            "production_evidence",
            "source_contract_sha256",
        }:
            receipt[path[0]] = manifest[path[0]]
        _replace_private(manifest_path, forged_manifest_raw)
        _replace_private(receipt_path, subject._canonical_json(receipt, newline=True))
        reads.clear()
        with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
            subject.load_notsofar1_natural_overlap_bundle(output)
        assert reads == [subject.MANIFEST_FILENAME]
        _replace_private(manifest_path, manifest_raw)
        _replace_private(receipt_path, receipt_raw)

    for path, replacement in receipt_mutations:
        receipt = json.loads(receipt_raw)
        _set_nested(receipt, path, replacement)
        _replace_private(receipt_path, subject._canonical_json(receipt, newline=True))
        reads.clear()
        with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
            subject.load_notsofar1_natural_overlap_bundle(output)
        assert reads == [
            subject.MANIFEST_FILENAME,
            subject.PREPARATION_RECEIPT_FILENAME,
        ]
        assert not any(name.endswith(".f32le") for name in reads)
        _replace_private(receipt_path, receipt_raw)


def test_wrong_builtin_container_shapes_fail_before_source_or_pcm_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_lock = json.loads(subject.DEFAULT_LOCK.read_bytes())
    lock_mutations = (
        (("evidence_scope",), []),
        (("privacy",), []),
        (("sample_format",), []),
        (("source",), []),
        (("source", "files"), {}),
        (("selection",), []),
        (("selection", "cases"), {}),
        (("output_artifacts",), {}),
    )
    for index, (path, replacement) in enumerate(lock_mutations):
        lock = json.loads(_json_bytes(original_lock))
        _set_nested(lock, path, replacement)
        body = dict(lock)
        body.pop("recipe_sha256")
        recipe = hashlib.sha256(_json_bytes(body)).hexdigest()
        lock["recipe_sha256"] = recipe
        lock_path = tmp_path / f"container-lock-{index}.json"
        _write_private(lock_path, _json_bytes(lock))
        with monkeypatch.context() as scoped:
            scoped.setattr(subject, "DEFAULT_LOCK", lock_path)
            scoped.setattr(subject, "EXPECTED_LOCK_RECIPE_SHA256", recipe)
            with pytest.raises(
                subject.NotsofarNaturalOverlapPreparationError, match="^$"
            ):
                subject._load_lock(lock_path, test_source_injection=None)

    _source, output, _prepared = _prepare(tmp_path / "bundle")
    manifest_path = output / subject.MANIFEST_FILENAME
    receipt_path = output / subject.PREPARATION_RECEIPT_FILENAME
    manifest_raw = manifest_path.read_bytes()
    receipt_raw = receipt_path.read_bytes()
    reads: list[str] = []
    original_read = subject._read_private

    def recording_read(*args: object, **kwargs: object):
        name = args[1] if len(args) > 1 else kwargs["name"]
        reads.append(name)
        return original_read(*args, **kwargs)

    monkeypatch.setattr(subject, "_read_private", recording_read)
    manifest_mutations = (
        (("evidence_scope",), []),
        (("sample_format",), []),
        (("windows",), {}),
        (("windows", 0), []),
        (("windows", 0, "envelope"), []),
        (("windows", 0, "overlap"), []),
        (("windows", 0, "role_a"), []),
        (("windows", 0, "channels"), {}),
    )
    for path, replacement in manifest_mutations:
        manifest = json.loads(manifest_raw)
        _set_nested(manifest, path, replacement)
        _replace_private(
            manifest_path,
            subject._canonical_json(manifest, newline=True),
        )
        reads.clear()
        with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
            subject.load_notsofar1_natural_overlap_bundle(output)
        assert reads == [subject.MANIFEST_FILENAME]
        _replace_private(manifest_path, manifest_raw)

    receipt_mutations = (
        (("evidence_scope",), []),
        (("privacy",), []),
        (("manifest",), []),
        (("totals",), []),
        (("preparer_files",), {}),
        (("source_files",), {}),
    )
    for path, replacement in receipt_mutations:
        receipt = json.loads(receipt_raw)
        _set_nested(receipt, path, replacement)
        _replace_private(
            receipt_path,
            subject._canonical_json(receipt, newline=True),
        )
        reads.clear()
        with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
            subject.load_notsofar1_natural_overlap_bundle(output)
        assert reads == [
            subject.MANIFEST_FILENAME,
            subject.PREPARATION_RECEIPT_FILENAME,
        ]
        assert not any(name.endswith(".f32le") for name in reads)
        _replace_private(receipt_path, receipt_raw)


def test_schema_helpers_reject_subclasses_and_hostile_hooks_without_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Hostile:
        calls = 0

        def _called(self) -> None:
            type(self).calls += 1
            raise AssertionError("hostile schema hook invoked")

        def __bool__(self) -> bool:
            self._called()

        def __eq__(self, _other: object) -> bool:
            self._called()

        def __int__(self) -> int:
            self._called()

        def __iter__(self):
            self._called()

        def __len__(self) -> int:
            self._called()

        def __str__(self) -> str:
            self._called()

    class _HostileDict(dict):
        calls = 0

        def __iter__(self):
            type(self).calls += 1
            raise AssertionError("hostile dict hook invoked")

        def values(self):
            type(self).calls += 1
            raise AssertionError("hostile dict values hook invoked")

    class _HostileList(list):
        calls = 0

        def __iter__(self):
            type(self).calls += 1
            raise AssertionError("hostile list hook invoked")

    class _HostileStr(str):
        calls = 0
        __hash__ = str.__hash__

        def _called(self) -> None:
            type(self).calls += 1
            raise AssertionError("hostile string hook invoked")

        def __eq__(self, _other: object) -> bool:
            self._called()

        def startswith(self, *_args: object, **_kwargs: object) -> bool:
            self._called()

        def strip(self, *_args: object, **_kwargs: object) -> str:
            self._called()

        def encode(self, *_args: object, **_kwargs: object) -> bytes:
            self._called()

    class _HostileInt(int):
        calls = 0

        def __eq__(self, _other: object) -> bool:
            type(self).calls += 1
            raise AssertionError("hostile int hook invoked")

        def __int__(self) -> int:
            type(self).calls += 1
            raise AssertionError("hostile int conversion invoked")

    class _HostileBytes(bytes):
        calls = 0

        def __bool__(self) -> bool:
            type(self).calls += 1
            raise AssertionError("hostile bytes truth hook invoked")

        def __len__(self) -> int:
            type(self).calls += 1
            raise AssertionError("hostile bytes length hook invoked")

    hostile_values: list[object] = [
        _Hostile(),
        _HostileDict(),
        _HostileList(),
        _HostileStr("schema_version"),
        _HostileInt(1),
    ]
    hostile_key = _HostileStr("schema_version")
    hostile_values.append({hostile_key: 1})
    for cls in (
        _Hostile,
        _HostileDict,
        _HostileList,
        _HostileStr,
        _HostileInt,
        _HostileBytes,
    ):
        cls.calls = 0
    for hostile in hostile_values:
        with monkeypatch.context() as scoped:
            scoped.setattr(subject.json, "loads", lambda *_args, **_kwargs: hostile)
            with pytest.raises(
                subject.NotsofarNaturalOverlapPreparationError, match="^$"
            ):
                subject._strict_json(b"{}", maximum=16)
    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject._strict_json(_HostileBytes(b"{}"), maximum=16)
    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject._strict_json(b"{}", maximum=_HostileInt(16))
    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject._strict_json(b"{}", maximum=True)

    hostile = _Hostile()
    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject._expect_exact_json(hostile, 1)
    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject._safe_selection_rows([hostile])
    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject._selection_private_commitment_sha256("synthetic-safe-v1", hostile)  # type: ignore[arg-type]

    source = _make_source(tmp_path / "source")
    loaded_lock = subject._load_lock(
        subject.DEFAULT_LOCK,
        test_source_injection=source.injection,
    )
    hostile_lock = replace(loaded_lock, production_evidence=hostile)  # type: ignore[arg-type]
    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject._validate_against_lock(hostile_lock, (), (), {})
    hostile_contract_lock = replace(loaded_lock, fixture_id=hostile)  # type: ignore[arg-type]
    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject._source_contract_sha256(hostile_contract_lock, [], (), ())
    hostile_injection = subject.TestSourceInjection(
        fixture_id=hostile,  # type: ignore[arg-type]
        file_pins=source.injection.file_pins,
        meeting_duration_seconds=source.injection.meeting_duration_seconds,
    )
    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject._load_lock(
            subject.DEFAULT_LOCK,
            test_source_injection=hostile_injection,
        )
    hostile_pins_injection = subject.TestSourceInjection(
        fixture_id="synthetic-hostile-pins-v1",
        file_pins=hostile,  # type: ignore[arg-type]
        meeting_duration_seconds=source.injection.meeting_duration_seconds,
    )
    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject._load_lock(
            subject.DEFAULT_LOCK,
            test_source_injection=hostile_pins_injection,
        )
    hostile_key_pins = dict(source.injection.file_pins)
    first_key = next(iter(hostile_key_pins))
    first_value = hostile_key_pins.pop(first_key)
    hostile_key_pins[_HostileStr(first_key)] = first_value
    hostile_key_injection = subject.TestSourceInjection(
        fixture_id="synthetic-hostile-key-v1",
        file_pins=hostile_key_pins,
        meeting_duration_seconds=source.injection.meeting_duration_seconds,
    )
    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject._load_lock(
            subject.DEFAULT_LOCK,
            test_source_injection=hostile_key_injection,
        )
    manifest = {"schema_version": hostile}
    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject._validate_authority_inputs(manifest, {}, ())
    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject._source_budget({hostile_key: ("role", 1, "0" * 64)})  # type: ignore[dict-item]
    loaded = subject.load_notsofar1_natural_overlap_bundle(
        _prepare(tmp_path / "bundle")[1]
    )
    valid_manifest = _manifest(loaded.path.parent)
    valid_receipt = json.loads(
        (loaded.path.parent / subject.PREPARATION_RECEIPT_FILENAME).read_bytes()
    )
    valid_manifest_raw = loaded.path.read_bytes()
    valid_receipt_raw = (
        loaded.path.parent / subject.PREPARATION_RECEIPT_FILENAME
    ).read_bytes()
    manifest_identity = subject._PrivateFile(
        name=subject.MANIFEST_FILENAME,
        size_bytes=len(valid_manifest_raw),
        sha256=hashlib.sha256(valid_manifest_raw).hexdigest(),
        snapshot=(0,) * 9,
    )
    hostile_manifest_header = json.loads(json.dumps(valid_manifest))
    schema_version = hostile_manifest_header.pop("schema_version")
    hostile_manifest_header[_HostileStr("schema_version")] = schema_version
    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject._parse_receipt(
            valid_receipt_raw,
            manifest=manifest_identity,
            manifest_value=hostile_manifest_header,
        )
    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject._parse_receipt(
            valid_receipt_raw,
            manifest=replace(manifest_identity, name=hostile),  # type: ignore[arg-type]
            manifest_value=valid_manifest,
        )
    hostile_windows = json.loads(json.dumps(loaded.windows))
    hostile_windows[0]["role_b"]["reference"] = hostile
    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject._validate_authority_inputs(
            valid_manifest,
            valid_receipt,
            tuple(hostile_windows),
        )
    for field in ("recipe_sha256", "overlap_metric", "production_evidence"):
        hostile_field_lock = replace(loaded_lock, **{field: hostile})  # type: ignore[arg-type]
        with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
            subject._source_contract_sha256(hostile_field_lock, [], (), ())
    valid_source_rows = json.loads(json.dumps(valid_receipt["source_files"]))
    valid_selection_rows = subject._loaded_selection_rows(loaded.windows)
    valid_closure = tuple(json.loads(json.dumps(valid_receipt["preparer_files"])))
    for target in (
        "source-role",
        "source-hash",
        "source-size",
        "closure-path",
        "closure-hash",
        "closure-size",
    ):
        source_rows = json.loads(json.dumps(valid_source_rows))
        closure_rows = list(json.loads(json.dumps(valid_closure)))
        if target.startswith("source-"):
            source_rows[0][
                target.removeprefix("source-")
                .replace("hash", "sha256")
                .replace("size", "size_bytes")
            ] = hostile
        else:
            closure_rows[0][
                target.removeprefix("closure-")
                .replace("hash", "sha256")
                .replace("size", "size_bytes")
            ] = hostile
        with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
            subject._source_contract_sha256(
                loaded_lock,
                source_rows,
                valid_selection_rows,
                closure_rows,
            )
    verifier_calls = 0

    def forbidden_reopen(*_args: object, **_kwargs: object) -> object:
        nonlocal verifier_calls
        verifier_calls += 1
        raise AssertionError("invalid loaded metadata reached bundle reopen")

    bool_windows = json.loads(json.dumps(loaded.windows))
    assert bool_windows[0]["role_a"]["activity_relative_start_sample"] == 0
    bool_windows[0]["role_a"]["activity_relative_start_sample"] = False
    hostile_value_windows = json.loads(json.dumps(loaded.windows))
    hostile_value_windows[0]["role_b"]["reference"] = hostile
    hostile_key_windows = json.loads(json.dumps(loaded.windows))
    window_id = hostile_key_windows[0].pop("window_id")
    hostile_key_windows[0][_HostileStr("window_id")] = window_id
    for invalid_windows in (
        bool_windows,
        hostile_value_windows,
        hostile_key_windows,
    ):
        with monkeypatch.context() as scoped:
            scoped.setattr(
                subject,
                "load_notsofar1_natural_overlap_bundle",
                forbidden_reopen,
            )
            with pytest.raises(
                subject.NotsofarNaturalOverlapPreparationError, match="^$"
            ):
                subject.verify_notsofar1_natural_overlap_bundle(
                    replace(loaded, windows=tuple(invalid_windows))
                )
    assert verifier_calls == 0
    assert _Hostile.calls == 0
    assert _HostileDict.calls == 0
    assert _HostileList.calls == 0
    assert _HostileStr.calls == 0
    assert _HostileInt.calls == 0
    assert _HostileBytes.calls == 0


@pytest.mark.parametrize("mutation", ["in-place", "replacement"])
def test_loader_rejects_leaf_changed_after_initial_read_before_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    _source, output, _prepared = _prepare(tmp_path)
    manifest = _manifest(output)
    target = output / manifest["windows"][0]["channels"][0]["file"]
    target_name = target.name
    original = subject._read_private
    target_reads = 0

    def mutate_after_read(*args: object, **kwargs: object):
        nonlocal target_reads
        name = args[1] if len(args) > 1 else kwargs.get("name")
        if name == target_name:
            target_reads += 1
        result = original(*args, **kwargs)
        if name == target_name:
            if target_reads == 1:
                raw = bytearray(target.read_bytes())
                if mutation == "in-place":
                    raw[0] ^= 1
                    target.write_bytes(raw)
                    os.chmod(target, 0o600)
                else:
                    _replace_private(target, bytes(raw))
        return result

    monkeypatch.setattr(subject, "_read_private", mutate_after_read)
    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject.load_notsofar1_natural_overlap_bundle(output)
    assert target_reads == 2


def test_exact_entry_enumeration_stops_at_expected_plus_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = subject._expected_entries(receipt=True)

    class _Entry:
        def __init__(self, name: str) -> None:
            self.name = name

    class _UnboundedEntries:
        calls = 0

        def __enter__(self) -> _UnboundedEntries:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def __iter__(self) -> _UnboundedEntries:
            return self

        def __next__(self) -> _Entry:
            type(self).calls += 1
            return _Entry(f"extra-{type(self).calls}")

    monkeypatch.setattr(subject.os, "scandir", lambda _fd: _UnboundedEntries())
    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject._exact_entries(-1, expected)
    assert _UnboundedEntries.calls == len(expected) + 1


def test_loader_rejects_fully_rebound_forged_production_authority(
    tmp_path: Path,
) -> None:
    _source, output, _prepared = _prepare(tmp_path)
    lock_value = json.loads(subject.DEFAULT_LOCK.read_bytes())
    production_lock = subject._Lock(
        fixture_id=subject.FIXTURE_ID,
        recipe_sha256=lock_value["recipe_sha256"],
        overlap_metric=subject.OVERLAP_METRIC,
        source_lock_recipe_sha256=subject.inherited.EXPECTED_RECIPE_SHA256,
        source_pins=subject.inherited._SOURCE_PINS,
        selection_rows=subject._safe_selection_rows(lock_value["selection"]["cases"]),
        selection_private_commitment_sha256=lock_value["selection"][
            "selection_private_commitment_sha256"
        ],
        artifact_pins=tuple(dict(row) for row in lock_value["output_artifacts"]),
        production_evidence=True,
    )
    manifest = _manifest(output)
    first_window = manifest["windows"][0]
    first_role = first_window["role_a"]
    first_role["reference"] += " forged-reference"
    first_role["reference_sha256"] = hashlib.sha256(
        first_role["reference"].encode("utf-8")
    ).hexdigest()
    first_channel = first_window["channels"][0]
    audio_path = output / first_channel["file"]
    values = np.frombuffer(audio_path.read_bytes(), dtype="<f4").copy()
    values[0] = np.float32(0.123)
    audio_raw = values.astype("<f4").tobytes()
    _replace_private(audio_path, audio_raw)
    first_channel["sha256"] = hashlib.sha256(audio_raw).hexdigest()

    forged_sources = [
        {
            "role": production_lock.source_pins[relative][0],
            "sha256": production_lock.source_pins[relative][2],
            "size_bytes": production_lock.source_pins[relative][1],
        }
        for relative in subject._SOURCE_PATHS
    ]
    forged_sources[0]["sha256"] = "1" * 64
    forged_closure = [dict(row) for row in subject._preparer_closure()]
    forged_closure[0]["sha256"] = "2" * 64
    forged_contract = subject._source_contract_sha256(
        production_lock,
        forged_sources,
        production_lock.selection_rows,
        forged_closure,
    )
    manifest.update(
        {
            "fixture_id": subject.FIXTURE_ID,
            "lock_recipe_sha256": production_lock.recipe_sha256,
            "production_evidence": True,
            "source_contract_sha256": forged_contract,
        }
    )
    manifest_raw = subject._canonical_json(manifest, newline=True)
    _replace_private(output / subject.MANIFEST_FILENAME, manifest_raw)
    receipt_path = output / subject.PREPARATION_RECEIPT_FILENAME
    receipt = json.loads(receipt_path.read_bytes())
    receipt.update(
        {
            "fixture_id": subject.FIXTURE_ID,
            "lock_recipe_sha256": production_lock.recipe_sha256,
            "preparer_files": forged_closure,
            "production_evidence": True,
            "source_contract_sha256": forged_contract,
            "source_files": forged_sources,
        }
    )
    receipt["manifest"] = {
        "file": subject.MANIFEST_FILENAME,
        "sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "size_bytes": len(manifest_raw),
    }
    _replace_private(receipt_path, subject._canonical_json(receipt, newline=True))

    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject.load_notsofar1_natural_overlap_bundle(output)


def test_loader_rejects_self_consistent_production_bundle_with_nonexact_total(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _source, output, _prepared = _prepare(tmp_path)
    manifest = _manifest(output)
    receipt_path = output / subject.PREPARATION_RECEIPT_FILENAME
    receipt = json.loads(receipt_path.read_bytes())
    windows = tuple(manifest["windows"])
    private_selection_rows = subject._loaded_selection_rows(windows)
    public_selection_rows = subject._loaded_selection_rows(
        windows,
        include_references=False,
    )
    artifact_rows = subject._loaded_artifact_rows(windows)
    compact_total = sum(row["size_bytes"] for row in artifact_rows)
    assert 0 < compact_total < subject.PRODUCTION_DERIVED_PCM_BYTES

    source_pins = {
        relative: (
            receipt["source_files"][index]["role"],
            receipt["source_files"][index]["size_bytes"],
            receipt["source_files"][index]["sha256"],
        )
        for index, relative in enumerate(subject._SOURCE_PATHS)
    }
    production_lock = subject._Lock(
        fixture_id=subject.FIXTURE_ID,
        recipe_sha256=subject.EXPECTED_LOCK_RECIPE_SHA256,
        overlap_metric=subject.OVERLAP_METRIC,
        source_lock_recipe_sha256=subject.inherited.EXPECTED_RECIPE_SHA256,
        source_pins=source_pins,
        selection_rows=public_selection_rows,
        selection_private_commitment_sha256=(
            subject._selection_private_commitment_sha256(
                subject.FIXTURE_ID,
                subject._loaded_commitment_rows(windows),
            )
        ),
        artifact_pins=artifact_rows,
        production_evidence=True,
    )
    source_contract = subject._source_contract_sha256(
        production_lock,
        receipt["source_files"],
        private_selection_rows,
        tuple(receipt["preparer_files"]),
    )
    manifest.update(
        {
            "fixture_id": subject.FIXTURE_ID,
            "lock_recipe_sha256": production_lock.recipe_sha256,
            "production_evidence": True,
            "source_contract_sha256": source_contract,
        }
    )
    manifest_raw = subject._canonical_json(manifest, newline=True)
    _replace_private(output / subject.MANIFEST_FILENAME, manifest_raw)
    receipt.update(
        {
            "fixture_id": subject.FIXTURE_ID,
            "lock_recipe_sha256": production_lock.recipe_sha256,
            "manifest": {
                "file": subject.MANIFEST_FILENAME,
                "sha256": hashlib.sha256(manifest_raw).hexdigest(),
                "size_bytes": len(manifest_raw),
            },
            "production_evidence": True,
            "source_contract_sha256": source_contract,
        }
    )
    assert receipt["totals"]["audio_bytes"] == compact_total
    _replace_private(receipt_path, subject._canonical_json(receipt, newline=True))
    monkeypatch.setattr(
        subject, "_load_lock", lambda *_args, **_kwargs: production_lock
    )

    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject.load_notsofar1_natural_overlap_bundle(output)


@pytest.mark.parametrize(
    "mutation",
    ["missing", "extra", "symlink", "hardlink", "file-mode", "root-mode"],
)
def test_loader_rejects_leaf_and_directory_filesystem_aliases(
    tmp_path: Path, mutation: str
) -> None:
    _source, output, _prepared = _prepare(tmp_path)
    manifest = _manifest(output)
    artifact = output / manifest["windows"][0]["channels"][0]["file"]
    if mutation == "missing":
        artifact.unlink()
    elif mutation == "extra":
        _write_private(output / "unexpected.bin", b"unexpected")
    elif mutation == "symlink":
        target = output / manifest["windows"][0]["channels"][1]["file"]
        artifact.unlink()
        artifact.symlink_to(target.name)
    elif mutation == "hardlink":
        os.link(artifact, tmp_path / "outside-hardlink")
    elif mutation == "file-mode":
        os.chmod(artifact, 0o644)
    else:
        os.chmod(output, 0o755)
    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject.load_notsofar1_natural_overlap_bundle(output)


def test_loader_rejects_aggregate_cap_before_opening_any_audio_leaf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _source, output, _prepared = _prepare(tmp_path)
    manifest = _manifest(output)
    for window in manifest["windows"]:
        for channel in window["channels"]:
            channel["size_bytes"] = subject.MAX_DERIVED_PCM_BYTES
    _replace_private(
        output / subject.MANIFEST_FILENAME,
        subject._canonical_json(manifest, newline=True),
    )

    opened: list[str] = []
    original = subject._read_private

    def recording_read(*args: object, **kwargs: object):
        name = args[1] if len(args) > 1 else kwargs.get("name")
        assert isinstance(name, str)
        opened.append(name)
        return original(*args, **kwargs)

    monkeypatch.setattr(subject, "_read_private", recording_read)
    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject.load_notsofar1_natural_overlap_bundle(output)
    assert opened == [subject.MANIFEST_FILENAME]


@pytest.mark.parametrize("leaf", ["manifest", "receipt", "audio"])
def test_loader_rejects_oversized_leaves(tmp_path: Path, leaf: str) -> None:
    _source, output, _prepared = _prepare(tmp_path)
    if leaf == "manifest":
        target = output / subject.MANIFEST_FILENAME
        size = subject._MAX_MANIFEST_BYTES + 1
    elif leaf == "receipt":
        target = output / subject.PREPARATION_RECEIPT_FILENAME
        size = subject._MAX_RECEIPT_BYTES + 1
    else:
        manifest = _manifest(output)
        target = output / manifest["windows"][0]["channels"][0]["file"]
        size = subject._MAX_AUDIO_BYTES + 1
    _replace_private(target, b"x" * size)
    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject.load_notsofar1_natural_overlap_bundle(output)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"schema_version":1,"schema_version":1}\n',
        b"[" * 18 + b"0" + b"]" * 18,
    ],
)
def test_loader_rejects_duplicate_and_deep_json(tmp_path: Path, raw: bytes) -> None:
    _source, output, _prepared = _prepare(tmp_path)
    _replace_private(output / subject.MANIFEST_FILENAME, raw)
    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject.load_notsofar1_natural_overlap_bundle(output)


def test_verify_rejects_mutated_loaded_nested_window(tmp_path: Path) -> None:
    _source, output, _prepared = _prepare(tmp_path)
    loaded = subject.load_notsofar1_natural_overlap_bundle(output)
    role = loaded.windows[0]["role_a"]
    assert isinstance(role, dict)
    role["reference"] += " mutated-after-load"
    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject.verify_notsofar1_natural_overlap_bundle(loaded)


def test_verify_rejects_constructed_exact_type_with_forged_fields(
    tmp_path: Path,
) -> None:
    _source, output, _prepared = _prepare(tmp_path)
    loaded = subject.load_notsofar1_natural_overlap_bundle(output)
    forged = subject.LoadedNotsofarNaturalOverlapBundle(
        path=loaded.path,
        digest="0" * 64,
        receipt_sha256=loaded.receipt_sha256,
        source_contract_sha256=loaded.source_contract_sha256,
        production_evidence=loaded.production_evidence,
        windows=loaded.windows,
        identities=loaded.identities,
    )
    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject.verify_notsofar1_natural_overlap_bundle(forged)

    missing = subject.LoadedNotsofarNaturalOverlapBundle(
        path=tmp_path / "absent" / subject.MANIFEST_FILENAME,
        digest=loaded.digest,
        receipt_sha256=loaded.receipt_sha256,
        source_contract_sha256=loaded.source_contract_sha256,
        production_evidence=loaded.production_evidence,
        windows=loaded.windows,
        identities=loaded.identities,
    )
    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject.verify_notsofar1_natural_overlap_bundle(missing)


@pytest.mark.parametrize(
    "field",
    [
        "path",
        "digest",
        "receipt_sha256",
        "source_contract_sha256",
        "production_evidence",
        "windows",
        "identities",
        "nested_reference",
        "nested_identity",
    ],
)
def test_verify_rejects_hostile_loaded_fields_without_invoking_hooks(
    tmp_path: Path,
    field: str,
) -> None:
    _source, output, prepared = _prepare(tmp_path)

    class _HostileField:
        calls = 0

        def _called(self) -> None:
            type(self).calls += 1
            raise AssertionError("hostile loaded-field hook invoked")

        def __eq__(self, _other: object) -> bool:
            self._called()

        def __fspath__(self) -> str:
            self._called()

        def __str__(self) -> str:
            self._called()

    values: dict[str, object] = {
        "path": prepared.path,
        "digest": prepared.digest,
        "receipt_sha256": prepared.receipt_sha256,
        "source_contract_sha256": prepared.source_contract_sha256,
        "production_evidence": prepared.production_evidence,
        "windows": prepared.windows,
        "identities": prepared.identities,
    }
    if field == "nested_reference":
        windows = json.loads(json.dumps(prepared.windows))
        windows[0]["role_a"]["reference"] = _HostileField()
        values["windows"] = tuple(windows)
    elif field == "nested_identity":
        identities = [list(identity) for identity in prepared.identities]
        identities[0][0] = _HostileField()
        values["identities"] = tuple(tuple(identity) for identity in identities)
    else:
        values[field] = _HostileField()
    forged = subject.LoadedNotsofarNaturalOverlapBundle(**values)  # type: ignore[arg-type]
    with pytest.raises(subject.NotsofarNaturalOverlapPreparationError, match="^$"):
        subject.verify_notsofar1_natural_overlap_bundle(forged)
    assert _HostileField.calls == 0


def test_production_selector_vector_is_frozen_without_opening_retained_wavs() -> None:
    expected = (
        (528_320, 552_800, 9_600, 23_040),
        (5_870_720, 5_901_600, 9_280, 25_920),
        (569_280, 611_360, 1_600, 10_720),
        (5_574_880, 5_642_720, 39_680, 57_920),
        (2_278_720, 2_317_600, 7_360, 37_920),
        (2_321_440, 2_378_240, 40_960, 56_000),
        (3_919_200, 3_960_320, 16_640, 36_480),
    )
    assert subject.PRODUCTION_SELECTION_GEOMETRY == expected
    assert (
        sum(end - start for start, end, _overlap_start, _overlap_end in expected)
        == 302_080
    )
    assert subject.PRODUCTION_DERIVED_PCM_BYTES == 2_416_640
    assert subject.SELECTION_SEED == "speaker-notsofar1-natural-overlap-v1-2026-08-12"
    assert subject.MIN_OVERLAP_SECONDS == Decimal(".300000")


def test_production_lock_recipe_source_selection_and_artifact_pins_are_anchored() -> (
    None
):
    loaded = subject._load_lock(subject.DEFAULT_LOCK, test_source_injection=None)
    lock_value = json.loads(subject.DEFAULT_LOCK.read_bytes())
    assert subject.EXPECTED_LOCK_RECIPE_SHA256 == (
        "ec6bf59b21e7fe59d7ea432fd4a6ac17e965fb3289e2523121576f71636150a5"
    )
    assert lock_value["recipe_sha256"] == subject.EXPECTED_LOCK_RECIPE_SHA256
    assert loaded.recipe_sha256 == subject.EXPECTED_LOCK_RECIPE_SHA256
    assert loaded.production_evidence is True
    assert all(row["sha256"] != "0" * 64 for row in lock_value["output_artifacts"])
    selection = lock_value["selection"]
    commitment = selection["selection_private_commitment_sha256"]
    assert commitment == (
        "fec0323b7e677e373cfd7b8332258a930581ef87707f5a211a509967985b9beb"
    )
    assert loaded.selection_private_commitment_sha256 == commitment
    assert [key for key in selection if "commitment" in key] == [
        "selection_private_commitment_sha256"
    ]
    case_keys = {
        "end_sample",
        "overlap_relative_end_sample",
        "overlap_relative_start_sample",
        "role_a_relative_end_sample",
        "role_a_relative_start_sample",
        "role_b_relative_end_sample",
        "role_b_relative_start_sample",
        "start_sample",
    }
    assert all(set(row) == case_keys for row in selection["cases"])
    encoded_selection = json.dumps(selection, ensure_ascii=True, sort_keys=True)
    forbidden = (
        "reference_a",
        "reference_b",
        "reference_sha256",
        "transcript",
        "speaker_id",
        "device_id",
        "local_path",
        "ordinal",
        "rank_sha256",
    )
    assert all(value not in encoded_selection for value in forbidden)
    selection_rows = subject._safe_selection_rows(lock_value["selection"]["cases"])
    assert (
        tuple(
            (
                row["start_sample"],
                row["end_sample"],
                row["overlap_relative_start_sample"],
                row["overlap_relative_end_sample"],
            )
            for row in selection_rows
        )
        == subject.PRODUCTION_SELECTION_GEOMETRY
    )
    assert (
        len(lock_value["output_artifacts"])
        == subject.WINDOW_COUNT * subject.CHANNEL_COUNT
    )
    assert (
        sum(row["size_bytes"] for row in lock_value["output_artifacts"])
        == subject.PRODUCTION_DERIVED_PCM_BYTES
    )
