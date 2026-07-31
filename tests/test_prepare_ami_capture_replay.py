from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
import struct
import warnings
import zipfile

import numpy as np
import pytest

from tools.capture_replay import FRAME_SAMPLES, SAMPLE_RATE_HZ, load_corpus
from tools import prepare_ami_capture_replay as prepare


def _xml(events: list[tuple[object, ...]]) -> bytes:
    rows = [
        '<?xml version="1.0" encoding="ISO-8859-1"?>',
        '<nite:root xmlns:nite="http://nite.sourceforge.net/">',
    ]
    previous_end: object | None = None
    for index, event in enumerate(events):
        if event[0] == "w":
            _kind, start, end, text = event
            rows.append(
                f'<w nite:id="word{index}" starttime="{start}" '
                f'endtime="{end}">{text}</w>'
            )
            previous_end = end
        elif event[0] == "activity":
            _kind, name, start, end = event
            rows.append(
                f'<{name} nite:id="activity{index}" starttime="{start}" '
                f'endtime="{end}" />'
            )
        else:
            _kind, text = event
            assert previous_end is not None
            rows.append(
                f'<w nite:id="punc{index}" starttime="{previous_end}" '
                f'endtime="{previous_end}" punc="true">{text}</w>'
            )
    rows.append("</nite:root>")
    return "".join(rows).encode("iso-8859-1")


def _default_annotations() -> dict[str, bytes]:
    return {
        "A": _xml(
            [
                ("w", "1.000", "1.500", "Alpha"),
                ("p", ","),
                ("w", "1.600", "2.200", "solo"),
                ("p", "."),
                ("activity", "vocalsound", "4.000", "7.000"),
                ("w", "8.000", "9.500", "overlap"),
                ("w", "15.000", "15.800", "turn"),
            ]
        ),
        "B": _xml(
            [
                ("w", "8.500", "9.200", "together"),
                ("p", "!"),
                ("w", "16.000", "16.800", "changed"),
            ]
        ),
        "C": _xml([("w", "21.000", "21.600", "charlie")]),
        "D": _xml([("w", "25.000", "25.500", "delta")]),
    }


def _annotation_zip(
    path: Path,
    *,
    annotations: dict[str, bytes] | None = None,
    extras: list[tuple[str | zipfile.ZipInfo, bytes]] = (),
) -> bytes:
    values = _default_annotations() if annotations is None else annotations
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("ami_public_manual_1.6.2/words/", b"")
        for speaker in "ABCD":
            archive.writestr(
                f"ami_public_manual_1.6.2/words/ES2004a.{speaker}.words.xml",
                values[speaker],
            )
        for name, payload in extras:
            archive.writestr(name, payload)
    return path.read_bytes()


def _wav_bytes(
    *,
    seconds: int = 30,
    sample_rate: int = SAMPLE_RATE_HZ,
    channels: int = 1,
    bits: int = 16,
) -> bytes:
    frames = seconds * sample_rate
    wave = (
        np.sin(np.arange(frames, dtype=np.float64) * (2.0 * np.pi * 173.0 / sample_rate))
        * 8_000.0
    ).astype("<i2")
    if channels > 1:
        wave = np.repeat(wave[:, None], channels, axis=1).reshape(-1)
    payload = wave.tobytes(order="C")
    block_align = channels * (bits // 8)
    fmt = struct.pack(
        "<HHIIHH",
        1,
        channels,
        sample_rate,
        sample_rate * block_align,
        block_align,
        bits,
    )
    body = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", len(payload)) + payload
    return b"RIFF" + struct.pack("<I", len(body) + 4) + b"WAVE" + body


def _sources(
    tmp_path: Path,
    *,
    annotations: dict[str, bytes] | None = None,
    wav: bytes | None = None,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    annotation_path = tmp_path / "private-annotations.zip"
    annotation_raw = _annotation_zip(annotation_path, annotations=annotations)
    audio_path = tmp_path / "private-room-recording.wav"
    audio_raw = _wav_bytes() if wav is None else wav
    audio_path.write_bytes(audio_raw)
    return {
        "annotations_zip": annotation_path,
        "annotations_sha256": hashlib.sha256(annotation_raw).hexdigest(),
        "annotations_bytes": len(annotation_raw),
        "audio_wav": audio_path,
        "audio_sha256": hashlib.sha256(audio_raw).hexdigest(),
        "audio_bytes": len(audio_raw),
    }


def _prepare(tmp_path: Path, *, output_name: str = "prepared", **source_overrides):
    source = _sources(tmp_path, **source_overrides)
    corpus = prepare.prepare_ami_capture_replay(
        **source,
        output_dir=tmp_path / output_name,
    )
    return corpus, source


def _different_speaker_overlap(case) -> int:
    words = case.word_intervals
    return sum(
        max(
            0,
            min(left.end_sample, right.end_sample)
            - max(left.start_sample, right.start_sample),
        )
        for index, left in enumerate(words)
        for right in words[index + 1 :]
        if left.speaker_id != right.speaker_id
    )


def test_prepares_four_bounded_schema_v1_cases_with_exact_annotations(tmp_path):
    corpus, source = _prepare(tmp_path)

    assert corpus.schema_version == 1
    assert corpus.sample_rate_hz == 16_000
    assert corpus.frame_samples == 1_600
    assert [case.case_id for case in corpus.cases] == [
        "ami-es2004a-single",
        "ami-es2004a-overlap",
        "ami-es2004a-transition",
        "ami-es2004a-silence",
    ]
    assert all(8.0 <= case.duration_seconds <= 15.0 for case in corpus.cases)
    assert all(case.samples % FRAME_SAMPLES == 0 for case in corpus.cases)
    assert all(set(case.tracks) == {"mic"} for case in corpus.cases)

    single, overlap, transition, silence = corpus.cases
    assert len({row.speaker_id for row in single.word_intervals}) == 1
    assert [
        (word.text, word.start_sample, word.end_sample)
        for word in single.word_intervals
    ] == [
        ("Alpha,", 32_000, 40_000),
        ("solo.", 41_600, 51_200),
    ]
    assert _different_speaker_overlap(overlap) > 0
    assert len({row.speaker_id for row in transition.word_intervals}) >= 2
    assert _different_speaker_overlap(transition) == 0
    assert any(
        previous.speaker_id != current.speaker_id
        and previous.end_sample <= current.start_sample
        for previous, current in zip(
            transition.word_intervals,
            transition.word_intervals[1:],
        )
    )
    assert silence.assertion.value == "silence"
    assert silence.expected_text == ""
    assert silence.speech_intervals == ()
    assert silence.speaker_intervals == ()
    assert silence.word_intervals == ()

    all_words = [
        word.text
        for case in corpus.cases
        for word in case.word_intervals
    ]
    assert "Alpha," in all_words
    assert "solo." in all_words
    assert "together!" in all_words
    source_i16 = np.frombuffer(
        source["audio_wav"].read_bytes()[44:],
        dtype="<i2",
    )
    single_pcm = np.frombuffer(single.mic.pcm_bytes, dtype="<f4")
    np.testing.assert_array_equal(
        single_pcm[:32_000],
        source_i16[180_000:212_000].astype(np.float32)
        * np.float32(1.0 / 32768.0),
    )
    np.testing.assert_array_equal(
        single_pcm[32_000:51_200],
        source_i16[16_000:35_200].astype(np.float32)
        * np.float32(1.0 / 32768.0),
    )
    for case in corpus.cases[:3]:
        assert case.expected_text == " ".join(
            word.text for word in case.word_intervals
        )
        first = min(word.start_sample for word in case.word_intervals)
        last = max(word.end_sample for word in case.word_intervals)
        assert first >= 2 * SAMPLE_RATE_HZ
        assert case.samples - last >= 2 * SAMPLE_RATE_HZ
    calibration_prefixes = {
        case.mic.pcm_bytes[: 2 * SAMPLE_RATE_HZ * 4]
        for case in corpus.cases
    }
    assert len(calibration_prefixes) == 1
    roomtone = np.frombuffer(next(iter(calibration_prefixes)), "<f4")
    assert np.count_nonzero(roomtone)
    for case in corpus.cases:
        assert "calibration-roomtone-2s" in case.tags
        assert "postroll-roomtone-2s" in case.tags
        pcm = np.frombuffer(case.mic.pcm_bytes, dtype="<f4")
        if case.word_intervals:
            tail_start = max(word.end_sample for word in case.word_intervals)
            tail = pcm[tail_start:]
            np.testing.assert_array_equal(tail, np.resize(roomtone, tail.size))
        else:
            tail = pcm[-2 * SAMPLE_RATE_HZ :]
            np.testing.assert_array_equal(tail, roomtone)
        assert np.count_nonzero(tail)
        assert np.sqrt(np.mean(tail.astype("float64") ** 2)) > 0.0

    output = tmp_path / "prepared"
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o600
        for path in output.iterdir()
    )
    manifest = (output / "capture-replay.json").read_text(encoding="utf-8")
    assert source["annotations_sha256"] in manifest
    assert source["audio_sha256"] in manifest
    assert str(source["annotations_bytes"]) in corpus.purpose
    assert str(source["audio_bytes"]) in corpus.purpose
    assert load_corpus(output / "capture-replay.json").digest == corpus.digest


def test_selection_and_bytes_are_deterministic_across_destinations(tmp_path):
    source = _sources(tmp_path)
    first = prepare.prepare_ami_capture_replay(
        **source,
        output_dir=tmp_path / "first",
    )
    second = prepare.prepare_ami_capture_replay(
        **source,
        output_dir=tmp_path / "second",
    )

    assert first.digest == second.digest
    assert first.path.read_bytes() == second.path.read_bytes()
    assert [case.mic.sha256 for case in first.cases] == [
        case.mic.sha256 for case in second.cases
    ]
    assert [case.mic.pcm_bytes for case in first.cases] == [
        case.mic.pcm_bytes for case in second.cases
    ]
    safe = prepare._safe_result(first)
    assert safe["cases"] == 4
    assert 32.0 <= safe["total_seconds"] <= 60.0
    assert safe["audio_bytes"] == sum(case.samples * 4 for case in first.cases)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("annotations_sha256", "0" * 64),
        ("audio_sha256", "f" * 64),
        ("annotations_bytes", 1),
        ("audio_bytes", 1),
    ],
)
def test_explicit_source_hash_and_size_are_required(
    tmp_path,
    field,
    replacement,
):
    source = _sources(tmp_path)
    source[field] = replacement
    output = tmp_path / "must-not-exist"

    with pytest.raises(prepare.AmiPreparationError):
        prepare.prepare_ami_capture_replay(**source, output_dir=output)

    assert not output.exists()


@pytest.mark.parametrize(
    "wav",
    [
        _wav_bytes(sample_rate=8_000),
        _wav_bytes(channels=2),
        _wav_bytes() + b"trailing",
    ],
)
def test_only_exact_mono_pcm16_16k_riff_is_accepted(tmp_path, wav):
    source = _sources(tmp_path, wav=wav)
    output = tmp_path / "must-not-exist"

    with pytest.raises(prepare.AmiPreparationError):
        prepare.prepare_ami_capture_replay(**source, output_dir=output)

    assert not output.exists()


@pytest.mark.parametrize(
    "unsafe_name",
    [
        "../escape.xml",
        "/absolute.xml",
        "a\\b.xml",
        "a//b.xml",
        "a/./b.xml",
        "C:/drive.xml",
    ],
)
def test_zip_rejects_unsafe_member_names(tmp_path, unsafe_name):
    archive = tmp_path / "annotations.zip"
    raw = _annotation_zip(archive, extras=[(unsafe_name, b"<root />")])
    audio = _wav_bytes()
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(audio)

    with pytest.raises(prepare.AmiPreparationError):
        prepare.prepare_ami_capture_replay(
            annotations_zip=archive,
            annotations_sha256=hashlib.sha256(raw).hexdigest(),
            annotations_bytes=len(raw),
            audio_wav=audio_path,
            audio_sha256=hashlib.sha256(audio).hexdigest(),
            audio_bytes=len(audio),
            output_dir=tmp_path / "output",
        )


def test_zip_rejects_symlinks_duplicates_and_per_member_budget(tmp_path, monkeypatch):
    symlink = zipfile.ZipInfo("linked.xml")
    symlink.create_system = 3
    symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
    archive = tmp_path / "symlink.zip"
    raw = _annotation_zip(archive, extras=[(symlink, b"target")])
    audio = _wav_bytes()
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(audio)
    base = {
        "audio_wav": audio_path,
        "audio_sha256": hashlib.sha256(audio).hexdigest(),
        "audio_bytes": len(audio),
    }
    with pytest.raises(prepare.AmiPreparationError):
        prepare.prepare_ami_capture_replay(
            annotations_zip=archive,
            annotations_sha256=hashlib.sha256(raw).hexdigest(),
            annotations_bytes=len(raw),
            output_dir=tmp_path / "symlink-output",
            **base,
        )

    duplicate = tmp_path / "duplicate.zip"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(duplicate, "w") as writer:
            for speaker, payload in _default_annotations().items():
                name = f"words/ES2004a.{speaker}.words.xml"
                writer.writestr(name, payload)
                if speaker == "A":
                    writer.writestr(name, payload)
    duplicate_raw = duplicate.read_bytes()
    with pytest.raises(prepare.AmiPreparationError):
        prepare.prepare_ami_capture_replay(
            annotations_zip=duplicate,
            annotations_sha256=hashlib.sha256(duplicate_raw).hexdigest(),
            annotations_bytes=len(duplicate_raw),
            output_dir=tmp_path / "duplicate-output",
            **base,
        )

    ordinary = tmp_path / "ordinary.zip"
    ordinary_raw = _annotation_zip(ordinary)
    monkeypatch.setattr(prepare, "_MAX_ZIP_MEMBER_BYTES", 32)
    with pytest.raises(prepare.AmiPreparationError):
        prepare.prepare_ami_capture_replay(
            annotations_zip=ordinary,
            annotations_sha256=hashlib.sha256(ordinary_raw).hexdigest(),
            annotations_bytes=len(ordinary_raw),
            output_dir=tmp_path / "budget-output",
            **base,
        )


@pytest.mark.parametrize(
    "bad_a",
    [
        _xml(
            [
                ("w", "NaN", "1.500", "secret"),
                ("w", "1.600", "2.000", "later"),
            ]
        ),
        _xml(
            [
                ("w", "2.000", "2.500", "secret"),
                ("w", "1.000", "1.500", "backwards"),
            ]
        ),
        _xml([("w", "1.000", "1.500", "secret"), ("p", "not-punctuation")]),
        b'<!DOCTYPE root [<!ENTITY x "secret">]><root><w starttime="1" '
        b'endtime="2">&x;</w></root>',
    ],
)
def test_word_xml_requires_finite_ordered_timing_and_punctuation(tmp_path, bad_a):
    annotations = _default_annotations()
    annotations["A"] = bad_a
    source = _sources(tmp_path, annotations=annotations)
    output = tmp_path / "must-not-exist"

    with pytest.raises(prepare.AmiPreparationError):
        prepare.prepare_ami_capture_replay(**source, output_dir=output)

    assert not output.exists()


def test_output_is_no_overwrite_and_existing_files_are_unchanged(tmp_path):
    corpus, source = _prepare(tmp_path)
    manifest_before = corpus.path.read_bytes()
    audio_before = {
        case.mic.path.name: case.mic.path.read_bytes()
        for case in corpus.cases
    }

    with pytest.raises(prepare.AmiPreparationError):
        prepare.prepare_ami_capture_replay(
            **source,
            output_dir=tmp_path / "prepared",
        )

    assert corpus.path.read_bytes() == manifest_before
    assert {
        case.mic.path.name: case.mic.path.read_bytes()
        for case in corpus.cases
    } == audio_before


def test_cli_receipts_and_failures_are_aggregate_only(tmp_path, capsys):
    annotations = _default_annotations()
    secret_transcript = "do-not-leak-private-phrase"
    annotations["A"] = _xml(
        [
            ("w", "2.000", "2.500", secret_transcript),
            ("w", "1.000", "1.500", "backwards"),
        ]
    )
    source = _sources(tmp_path, annotations=annotations)
    private_path = str(source["annotations_zip"])
    argv = [
        "--annotations-zip",
        private_path,
        "--annotations-sha256",
        str(source["annotations_sha256"]),
        "--annotations-bytes",
        str(source["annotations_bytes"]),
        "--audio-wav",
        str(source["audio_wav"]),
        "--audio-sha256",
        str(source["audio_sha256"]),
        "--audio-bytes",
        str(source["audio_bytes"]),
        "--output-dir",
        str(tmp_path / "output"),
    ]

    assert prepare.main(argv) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "ok": False,
        "error": "ami_capture_replay_prerequisites_unavailable",
    }
    assert captured.err == ""
    assert secret_transcript not in captured.out
    assert private_path not in captured.out

    good = _sources(tmp_path / "good")
    success_argv = [
        "--annotations-zip",
        str(good["annotations_zip"]),
        "--annotations-sha256",
        str(good["annotations_sha256"]),
        "--annotations-bytes",
        str(good["annotations_bytes"]),
        "--audio-wav",
        str(good["audio_wav"]),
        "--audio-sha256",
        str(good["audio_sha256"]),
        "--audio-bytes",
        str(good["audio_bytes"]),
        "--output-dir",
        str(tmp_path / "good-output"),
    ]
    assert prepare.main(success_argv) == 0
    captured = capsys.readouterr()
    receipt = json.loads(captured.out)
    assert set(receipt) == {
        "ok",
        "schema_version",
        "manifest_sha256",
        "cases",
        "audio_bytes",
        "total_seconds",
    }
    assert receipt["ok"] is True
    assert receipt["cases"] == 4
    assert captured.err == ""
    assert str(good["audio_wav"]) not in captured.out
    assert "Alpha" not in captured.out

    assert prepare.main(["--unknown", private_path]) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "ok": False,
        "error": "ami_capture_replay_prerequisites_unavailable",
    }
    assert captured.err == ""
    assert private_path not in captured.out


def test_help_makes_no_download_or_model_boundary_explicit():
    help_text = " ".join(prepare._parser().format_help().split())

    assert "no download, model, network, or audio-device access" in help_text
