from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import struct

import pytest

import tools.prepare_notsofar1_far_field_fixture as subject


@dataclass(frozen=True)
class _SyntheticSource:
    root: Path
    injection: subject.TestSourceInjection
    speaker_secrets: tuple[str, ...]
    references: tuple[str, ...]


def _wav(*, seconds: int, value: int) -> bytes:
    samples = seconds * subject.SAMPLE_RATE_HZ
    data = struct.pack("<h", value) * samples
    header = (
        b"RIFF"
        + struct.pack("<I", 36 + len(data))
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, 16_000, 32_000, 2, 16)
        + b"data"
        + struct.pack("<I", len(data))
    )
    return header + data


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _make_source(tmp_path: Path) -> _SyntheticSource:
    root = tmp_path / "source"
    (root / "sc_meetup_0").mkdir(parents=True)
    (root / "sc_rockfall_2").mkdir()
    speakers = tuple(f"raw-speaker-secret-{letter}" for letter in "abcd")
    close_devices = tuple(f"private-device-{index}" for index in range(4))
    references: list[str] = []
    segments: list[dict[str, object]] = []
    starts = (1, 3, 5, 7, 9, 11, 13, 15, 17)
    for index, start in enumerate(starts):
        speaker = speakers[index // 3]
        lexical_tokens = ["private", "transcript", "number", str(index)]
        if index == 0:
            reference = f"private <FILL/> transcript number {index}"
            timed_tokens = ["private", "<fill/>", "transcript", "number", str(index)]
        elif index == 1:
            reference = f"<PName>private</PName> transcript number {index}"
            timed_tokens = [
                "<PName>private</PName>",
                "transcript",
                "number",
                str(index),
            ]
        else:
            reference = " ".join(lexical_tokens)
            timed_tokens = lexical_tokens
        references.append(reference)
        timings = []
        for token_index, token in enumerate(timed_tokens):
            token_start = start + 0.1 * token_index
            token_end = token_start if token == "<fill/>" else token_start + 0.08
            timings.append([token, token_start, token_end])
        segments.append(
            {
                "ct_wav_file_name": (f"close_talk/{close_devices[index // 3]}.wav"),
                "end_time": start + 0.75,
                "speaker_id": speaker,
                "start_time": float(start),
                "text": reference,
                "word_timing": timings,
            }
        )
    segments.append(
        {
            "ct_wav_file_name": f"close_talk/{close_devices[3]}.wav",
            "end_time": 19.3,
            "speaker_id": speakers[3],
            "start_time": 19.0,
            "text": "<BA/>",
            "word_timing": [["<BA/>", 19.1, 19.1]],
        }
    )
    metadata = {
        "Hashtags": "#DebateOverlaps",
        "MeetingDurationSec": 20.0,
        "MtgType": "mtg",
        "NumParticipants": 4,
        "ParticipantAliasToCtDevice": dict(zip(speakers, close_devices, strict=True)),
        "ParticipantAliases": list(speakers),
        "Room": "private-room-secret",
        "Topic": "private-topic-secret",
        "meeting_id": "MTG_32006",
    }
    devices = [
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
        "devices.json": _json_bytes(devices),
        "sc_meetup_0/ch0.wav": _wav(seconds=20, value=1000),
        "sc_rockfall_2/ch0.wav": _wav(seconds=20, value=-1000),
    }
    for relative, payload in payloads.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    pins = {
        relative: (len(payload), hashlib.sha256(payload).hexdigest())
        for relative, payload in payloads.items()
    }
    injection = subject.TestSourceInjection(
        fixture_id="synthetic-notsofar-v1",
        file_pins=pins,
        meeting_duration_seconds=Decimal("20.0"),
    )
    return _SyntheticSource(root, injection, speakers, tuple(references))


def _prepare(tmp_path: Path) -> tuple[_SyntheticSource, subject.PreparedNotsofarCorpus]:
    source = _make_source(tmp_path)
    prepared = subject.prepare_notsofar1_far_field_fixture(
        source_dir=source.root,
        output_dir=tmp_path / "published",
        accepted_license="CC-BY-4.0",
        test_source_injection=source.injection,
    )
    return source, prepared


def test_selects_nine_windows_and_exact_channel_pairs(tmp_path: Path) -> None:
    _source, prepared = _prepare(tmp_path)
    assert prepared.corpus.schema_version == 2
    assert len(prepared.corpus.cases) == 18
    assert prepared.production_evidence is False
    for index in range(9):
        left = prepared.corpus.cases[index * 2]
        right = prepared.corpus.cases[index * 2 + 1]
        assert left.case_id == f"notsofar-w{index:02d}-channel-a"
        assert right.case_id == f"notsofar-w{index:02d}-channel-b"
        assert left.expected_text == right.expected_text
        assert left.samples == right.samples
        assert left.tags[:-1] == right.tags[:-1]
        assert left.tags[-1:] == ("channel-a",)
        assert right.tags[-1:] == ("channel-b",)
    slots = [
        tag
        for case in prepared.corpus.cases
        for tag in case.tags
        if tag.startswith("speaker-slot-")
    ]
    assert {slots.count(f"speaker-slot-{index}") for index in range(3)} == {6}


def test_receipt_is_private_and_scopes_evidence(tmp_path: Path) -> None:
    source, prepared = _prepare(tmp_path)
    raw = (prepared.corpus.path.parent / "preparation-receipt.json").read_text()
    receipt = json.loads(raw)
    assert receipt["production_evidence"] is False
    assert receipt["evidence_scope"] == {
        "hard_wer_nonoverlap": True,
        "overlap_cases": 0,
        "overlap_evidence": False,
        "speaker_authority": False,
    }
    assert len(receipt["preparer_files"]) == 5
    assert all(not Path(row["path"]).is_absolute() for row in receipt["preparer_files"])
    forbidden = (
        *source.speaker_secrets,
        *source.references,
        str(source.root),
        "sc_meetup_0",
        "sc_rockfall_2",
        "private-device",
    )
    assert all(value not in raw for value in forbidden)
    assert "raw-speaker-secret" not in repr(source.injection)


@pytest.mark.parametrize("bad_license", ["", "CC0-1.0", "cc-by-4.0"])
def test_rejects_license_without_publication(tmp_path: Path, bad_license: str) -> None:
    source = _make_source(tmp_path)
    output = tmp_path / "published"
    with pytest.raises(subject.NotsofarPreparationError, match="^$"):
        subject.prepare_notsofar1_far_field_fixture(
            source_dir=source.root,
            output_dir=output,
            accepted_license=bad_license,
            test_source_injection=source.injection,
        )
    assert not output.exists()


def test_rejects_hash_tamper_and_semantic_metadata_tamper(tmp_path: Path) -> None:
    source = _make_source(tmp_path)
    metadata = source.root / "gt_meeting_metadata.json"
    metadata.write_bytes(metadata.read_bytes() + b" ")
    with pytest.raises(subject.NotsofarPreparationError, match="^$"):
        subject.prepare_notsofar1_far_field_fixture(
            source_dir=source.root,
            output_dir=tmp_path / "hash-output",
            accepted_license="CC-BY-4.0",
            test_source_injection=source.injection,
        )

    source = _make_source(tmp_path / "semantic")
    metadata = source.root / "gt_meeting_metadata.json"
    value = json.loads(metadata.read_bytes())
    value["NumParticipants"] = 5
    raw = _json_bytes(value)
    metadata.write_bytes(raw)
    pins = dict(source.injection.file_pins)
    pins["gt_meeting_metadata.json"] = (len(raw), hashlib.sha256(raw).hexdigest())
    injection = subject.TestSourceInjection(
        fixture_id="semantic-tamper-v1",
        file_pins=pins,
        meeting_duration_seconds=Decimal("20.0"),
    )
    with pytest.raises(subject.NotsofarPreparationError, match="^$"):
        subject.prepare_notsofar1_far_field_fixture(
            source_dir=source.root,
            output_dir=tmp_path / "semantic-output",
            accepted_license="CC-BY-4.0",
            test_source_injection=injection,
        )


def test_no_clobber_and_git_output_rejected(tmp_path: Path) -> None:
    source = _make_source(tmp_path)
    output = tmp_path / "published"
    output.mkdir()
    marker = output / "keep"
    marker.write_text("untouched")
    with pytest.raises(subject.NotsofarPreparationError, match="^$"):
        subject.prepare_notsofar1_far_field_fixture(
            source_dir=source.root,
            output_dir=output,
            accepted_license="CC-BY-4.0",
            test_source_injection=source.injection,
        )
    assert marker.read_text() == "untouched"

    repo_output = Path(__file__).resolve().parents[1] / "must-not-publish-notsofar"
    assert not repo_output.exists()
    with pytest.raises(subject.NotsofarPreparationError, match="^$"):
        subject.prepare_notsofar1_far_field_fixture(
            source_dir=source.root,
            output_dir=repo_output,
            accepted_license="CC-BY-4.0",
            test_source_injection=source.injection,
        )
    assert not repo_output.exists()


def test_source_mutation_after_publication_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _make_source(tmp_path)
    original = subject.publish_private_corpus

    def mutate_after_publish(**kwargs: object):
        corpus = original(**kwargs)
        target = source.root / "devices.json"
        target.write_bytes(target.read_bytes() + b" ")
        return corpus

    monkeypatch.setattr(subject, "publish_private_corpus", mutate_after_publish)
    output = tmp_path / "published"
    with pytest.raises(subject.NotsofarPreparationError, match="^$"):
        subject.prepare_notsofar1_far_field_fixture(
            source_dir=source.root,
            output_dir=output,
            accepted_license="CC-BY-4.0",
            test_source_injection=source.injection,
        )
    assert (output / "corpus.json").is_file()


def test_lock_is_self_bound_and_license_tamper_is_rejected(tmp_path: Path) -> None:
    source = _make_source(tmp_path)
    lock = json.loads(subject.DEFAULT_LOCK.read_bytes())
    lock["license_id"] = "CC0-1.0"
    body = dict(lock)
    body.pop("recipe_sha256")
    lock["recipe_sha256"] = hashlib.sha256(_json_bytes(body)).hexdigest()
    tampered = tmp_path / "tampered.lock.json"
    tampered.write_bytes(_json_bytes(lock))
    with pytest.raises(subject.NotsofarPreparationError, match="^$"):
        subject.prepare_notsofar1_far_field_fixture(
            source_dir=source.root,
            output_dir=tmp_path / "published",
            accepted_license="CC-BY-4.0",
            lock_path=tampered,
            test_source_injection=source.injection,
        )


def test_frozen_marker_transform_and_unknown_marker_rejection() -> None:
    assert subject._strip_notsofar_markers(
        "  hello <FILL/> <PName>private name</PName> world  "
    ) == ("hello private name world", False, False)
    assert subject._strip_notsofar_markers("<unknown/>") == (None, True, True)
    for malformed in ("<noise>", "<PName>open", "stray>", "<PName><PName>x</PName>"):
        with pytest.raises(subject.NotsofarPreparationError, match="^$"):
            subject._strip_notsofar_markers(malformed)
