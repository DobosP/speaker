from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools.capture_replay import (
    FRAME_SAMPLES,
    SAMPLE_RATE_HZ,
    ReplayAssertion,
    ReplayCorpusError,
    SpeakerRole,
    load_corpus,
    verify_corpus_snapshot,
)
from tools.capture_replay import corpus as corpus_module


def _track(path: Path, samples: np.ndarray) -> dict[str, object]:
    raw = np.asarray(samples, dtype="<f4").reshape(-1).tobytes()
    path.write_bytes(raw)
    return {
        "file": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "encoding": "f32le",
        "samples": len(raw) // 4,
    }


def _manifest(
    tmp_path: Path,
    *,
    assertion: str = "transcript",
    with_reference: bool = False,
) -> tuple[Path, dict[str, object]]:
    samples = 2 * FRAME_SAMPLES
    mic = np.zeros(samples, dtype="float32")
    mic[100:1_200] = 0.1
    tracks: dict[str, object] = {
        "mic": _track(tmp_path / "case.mic.f32le", mic)
    }
    if with_reference:
        reference = np.zeros(samples, dtype="float32")
        reference[800:2_400] = 0.2
        tracks["far_reference_zero"] = _track(
            tmp_path / "case.far0.f32le",
            reference,
        )

    annotated = assertion not in {"silence", "far_end_only"}
    expected_text = "hello there" if assertion in {
        "transcript",
        "near_end_only",
        "double_talk",
    } else ""
    case = {
        "id": "case-001",
        "tracks": tracks,
        "speech_intervals": (
            [{"start_sample": 100, "end_sample": 1_200}] if annotated else []
        ),
        "speaker_intervals": (
            [
                {
                    "speaker_id": "speaker-a",
                    "role": "unknown",
                    "start_sample": 100,
                    "end_sample": 1_200,
                }
            ]
            if annotated
            else []
        ),
        "word_intervals": (
            [
                {
                    "speaker_id": "speaker-a",
                    "text": "hello",
                    "start_sample": 200,
                    "end_sample": 500,
                },
                {
                    "speaker_id": "speaker-a",
                    "text": "there",
                    "start_sample": 700,
                    "end_sample": 1_000,
                },
            ]
            if annotated
            else []
        ),
        "assertion": assertion,
        "expected_text": expected_text,
        "commands": ["hello"] if expected_text else [],
        "tags": ["meeting", "unit-test"],
        "aec_delay_samples": 80 if with_reference else 0,
    }
    payload: dict[str, object] = {
        "schema_version": 1,
        "purpose": "Deterministic unit-test capture replay.",
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "frame_samples": FRAME_SAMPLES,
        "cases": [case],
    }
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path, payload


def _rewrite(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _case(payload: dict[str, object]) -> dict[str, object]:
    cases = payload["cases"]
    assert isinstance(cases, list)
    case = cases[0]
    assert isinstance(case, dict)
    return case


def test_loads_hash_bound_frame_aligned_manifest(tmp_path: Path) -> None:
    path, _payload = _manifest(tmp_path)

    corpus = load_corpus(path)

    assert corpus.schema_version == 1
    assert corpus.sample_rate_hz == SAMPLE_RATE_HZ
    assert corpus.frame_samples == FRAME_SAMPLES
    assert corpus.audio_bytes == 2 * FRAME_SAMPLES * 4
    assert len(corpus.digest) == 64
    assert len(corpus.cases) == 1
    case = corpus.cases[0]
    assert case.case_id == "case-001"
    assert case.assertion is ReplayAssertion.TRANSCRIPT
    assert case.speaker_intervals[0].role is SpeakerRole.UNKNOWN
    assert case.samples == 2 * FRAME_SAMPLES
    assert case.expected_text == "hello there"
    assert case.commands == ("hello",)
    assert case.duration_seconds == pytest.approx(0.2)
    assert case.track("far_reference_zero") is None
    verify_corpus_snapshot(corpus)


@pytest.mark.parametrize(
    ("assertion", "with_reference"),
    [
        ("transcript", False),
        ("silence", False),
        ("near_end_only", False),
        ("far_end_only", True),
        ("double_talk", True),
        ("event_only", False),
    ],
)
def test_closed_assertion_contracts_are_loadable(
    tmp_path: Path,
    assertion: str,
    with_reference: bool,
) -> None:
    path, _payload = _manifest(
        tmp_path,
        assertion=assertion,
        with_reference=with_reference,
    )

    case = load_corpus(path).cases[0]

    assert case.assertion.value == assertion


def test_rejects_duplicate_json_keys_and_unknown_fields(tmp_path: Path) -> None:
    path, payload = _manifest(tmp_path)
    duplicate = path.read_text(encoding="utf-8").replace(
        '"schema_version": 1',
        '"schema_version": 1, "schema_version": 1',
        1,
    )
    path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(ReplayCorpusError):
        load_corpus(path)

    _rewrite(path, payload)
    payload["unexpected"] = True
    _rewrite(path, payload)
    with pytest.raises(ReplayCorpusError):
        load_corpus(path)


def test_rejects_manifest_or_track_symlinks(tmp_path: Path) -> None:
    path, payload = _manifest(tmp_path)
    manifest_link = tmp_path / "manifest-link.json"
    manifest_link.symlink_to(path)
    with pytest.raises(ReplayCorpusError):
        load_corpus(manifest_link)

    mic_link = tmp_path / "mic-link.f32le"
    mic_link.symlink_to(tmp_path / "case.mic.f32le")
    tracks = _case(payload)["tracks"]
    assert isinstance(tracks, dict)
    mic = tracks["mic"]
    assert isinstance(mic, dict)
    mic["file"] = mic_link.name
    _rewrite(path, payload)
    with pytest.raises(ReplayCorpusError):
        load_corpus(path)


def test_rejects_hash_mismatch_nonfinite_pcm_and_non_frame_audio(
    tmp_path: Path,
) -> None:
    path, payload = _manifest(tmp_path)
    tracks = _case(payload)["tracks"]
    assert isinstance(tracks, dict)
    mic = tracks["mic"]
    assert isinstance(mic, dict)
    mic["sha256"] = "0" * 64
    _rewrite(path, payload)
    with pytest.raises(ReplayCorpusError):
        load_corpus(path)

    invalid = np.zeros(2 * FRAME_SAMPLES, dtype="<f4")
    invalid[0] = np.nan
    mic.update(_track(tmp_path / "case.mic.f32le", invalid))
    _rewrite(path, payload)
    with pytest.raises(ReplayCorpusError):
        load_corpus(path)

    unaligned = np.zeros(FRAME_SAMPLES + 1, dtype="<f4")
    mic.update(_track(tmp_path / "case.mic.f32le", unaligned))
    _rewrite(path, payload)
    with pytest.raises(ReplayCorpusError):
        load_corpus(path)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda case: case["speech_intervals"].append(
            {"start_sample": 500, "end_sample": 900}
        ),
        lambda case: case["speaker_intervals"][0].update(
            {"start_sample": 0, "end_sample": 2_000}
        ),
        lambda case: case["word_intervals"][0].update({"speaker_id": "speaker-b"}),
        lambda case: case.update({"expected_text": "different words"}),
        lambda case: case["word_intervals"].reverse(),
    ],
)
def test_rejects_inconsistent_or_unsorted_ground_truth(
    tmp_path: Path,
    mutation,
) -> None:
    path, payload = _manifest(tmp_path)
    mutation(_case(payload))
    _rewrite(path, payload)

    with pytest.raises(ReplayCorpusError):
        load_corpus(path)


@pytest.mark.parametrize(
    ("assertion", "with_reference", "mutation"),
    [
        (
            "silence",
            False,
            lambda case: case["speech_intervals"].append(
                {"start_sample": 0, "end_sample": 100}
            ),
        ),
        (
            "near_end_only",
            True,
            lambda _case: None,
        ),
        (
            "double_talk",
            False,
            lambda _case: None,
        ),
        (
            "far_end_only",
            False,
            lambda _case: None,
        ),
    ],
)
def test_rejects_assertion_track_or_annotation_mismatch(
    tmp_path: Path,
    assertion: str,
    with_reference: bool,
    mutation,
) -> None:
    path, payload = _manifest(
        tmp_path,
        assertion=assertion,
        with_reference=with_reference,
    )
    mutation(_case(payload))
    _rewrite(path, payload)

    with pytest.raises(ReplayCorpusError):
        load_corpus(path)


def test_rejects_duplicate_case_ids_files_and_unsafe_paths(tmp_path: Path) -> None:
    path, payload = _manifest(tmp_path, with_reference=True)
    case = _case(payload)
    tracks = case["tracks"]
    assert isinstance(tracks, dict)
    far = tracks["far_reference_zero"]
    mic = tracks["mic"]
    assert isinstance(far, dict) and isinstance(mic, dict)
    far.update(mic)
    _rewrite(path, payload)
    with pytest.raises(ReplayCorpusError):
        load_corpus(path)

    path, payload = _manifest(tmp_path)
    cases = payload["cases"]
    assert isinstance(cases, list)
    cases.append(dict(cases[0]))
    _rewrite(path, payload)
    with pytest.raises(ReplayCorpusError):
        load_corpus(path)

    cases.pop()
    mic = _case(payload)["tracks"]["mic"]
    assert isinstance(mic, dict)
    mic["file"] = "../case.mic.f32le"
    _rewrite(path, payload)
    with pytest.raises(ReplayCorpusError):
        load_corpus(path)


def test_enforces_total_audio_and_case_count_bounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, payload = _manifest(tmp_path)
    monkeypatch.setattr(corpus_module, "MAX_CORPUS_AUDIO_BYTES", 1)
    with pytest.raises(ReplayCorpusError):
        load_corpus(path)

    monkeypatch.setattr(corpus_module, "MAX_CORPUS_AUDIO_BYTES", 64 * 1024 * 1024)
    monkeypatch.setattr(corpus_module, "MAX_CASES", 0)
    _rewrite(path, payload)
    with pytest.raises(ReplayCorpusError):
        load_corpus(path)


def test_snapshot_verification_detects_late_manifest_or_pcm_change(
    tmp_path: Path,
) -> None:
    path, payload = _manifest(tmp_path)
    corpus = load_corpus(path)
    payload["purpose"] = "Changed after load."
    _rewrite(path, payload)
    with pytest.raises(ReplayCorpusError):
        verify_corpus_snapshot(corpus)

    path, _payload = _manifest(tmp_path)
    corpus = load_corpus(path)
    mic = tmp_path / "case.mic.f32le"
    raw = bytearray(mic.read_bytes())
    raw[-1] ^= 1
    mic.write_bytes(raw)
    with pytest.raises(ReplayCorpusError):
        verify_corpus_snapshot(corpus)
