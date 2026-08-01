from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import stat
import struct
import zipfile

import numpy as np
import pytest

from tools import prepare_ami_capture_replay as ami_source
from tools import prepare_ami_conversation_fixture as prepare
from tools import public_conversation_fixture as fixture
from tools.streaming_stt.corpus import load_corpus


def _xml(
    events: list[tuple[float, float, str]],
    *,
    activities: tuple[tuple[float, float, str], ...] = (),
) -> bytes:
    rows = [
        '<?xml version="1.0" encoding="ISO-8859-1"?>',
        '<nite:root xmlns:nite="http://nite.sourceforge.net/">',
    ]
    ordered = [
        (start, end, "w", text, index)
        for index, (start, end, text) in enumerate(events)
    ] + [
        (start, end, kind, "", index)
        for index, (start, end, kind) in enumerate(activities)
    ]
    for start, end, kind, text, index in sorted(ordered):
        if kind == "w":
            rows.append(
                f'<w nite:id="word{index}" starttime="{start:.3f}" '
                f'endtime="{end:.3f}">{text}</w>'
            )
        else:
            rows.append(
                f'<{kind} nite:id="activity{index}" starttime="{start:.3f}" '
                f'endtime="{end:.3f}" />'
            )
    rows.append("</nite:root>")
    return "".join(rows).encode("iso-8859-1")


def _annotation_rows() -> dict[str, list[tuple[float, float, str]]]:
    result: dict[str, list[tuple[float, float, str]]] = {
        speaker: [] for speaker in "ABCD"
    }
    cursor = 2.0
    # Twelve isolated regions cannot become transitions because their gaps are
    # greater than the transition bound.
    for index in range(12):
        speaker = "ABCD"[index % 4]
        result[speaker].extend(
            [
                (cursor, cursor + 0.38, f"isolated{index}a"),
                (cursor + 0.48, cursor + 0.98, f"isolated{index}b"),
            ]
        )
        cursor += 3.0

    # Eight disjoint two-speaker exchanges provide ample transition candidates.
    for index in range(8):
        left = "ABCD"[index % 4]
        right = "ABCD"[(index + 1) % 4]
        result[left].extend(
            [
                (cursor, cursor + 0.35, f"transition{index}lefta"),
                (cursor + 0.45, cursor + 0.90, f"transition{index}leftb"),
            ]
        )
        result[right].extend(
            [
                (cursor + 1.10, cursor + 1.45, f"transition{index}righta"),
                (cursor + 1.55, cursor + 2.00, f"transition{index}rightb"),
            ]
        )
        cursor += 4.0

    # This genuine cross-speaker overlap must never enter hard-WER candidates.
    result["A"].append((cursor, cursor + 1.0, "forbiddenoverlapa"))
    result["B"].append((cursor + 0.3, cursor + 1.2, "forbiddenoverlapb"))
    return {speaker: sorted(rows) for speaker, rows in result.items()}


def _annotation_zip(path: Path) -> bytes:
    rows = _annotation_rows()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("ami_public_manual_1.6.2/words/", b"")
        for speaker in "ABCD":
            archive.writestr(
                f"ami_public_manual_1.6.2/words/ES2004a.{speaker}.words.xml",
                _xml(
                    rows[speaker],
                    activities=(
                        ((2.20, 2.30, "vocalsound"),)
                        if speaker == "A"
                        else ()
                    ),
                ),
            )
    return path.read_bytes()


def _wav_bytes(*, seconds: int = 74, offset: int = 0, sample_rate: int = 16_000) -> bytes:
    frames = seconds * sample_rate
    values = (((np.arange(frames, dtype=np.int64) + offset) % 20_003) - 10_001).astype(
        "<i2"
    )
    payload = values.tobytes(order="C")
    fmt = struct.pack(
        "<HHIIHH",
        1,
        1,
        sample_rate,
        sample_rate * 2,
        2,
        16,
    )
    body = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", len(payload)) + payload
    return b"RIFF" + struct.pack("<I", len(body) + 4) + b"WAVE" + body


def _sources(tmp_path: Path, *, far_sample_rate: int = 16_000) -> dict[str, object]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    annotations_path = tmp_path / "private-annotations-canary.zip"
    annotations = _annotation_zip(annotations_path)
    far_path = tmp_path / "private-far-canary.wav"
    far = _wav_bytes(offset=17, sample_rate=far_sample_rate)
    far_path.write_bytes(far)
    close_path = tmp_path / "private-close-canary.wav"
    close = _wav_bytes(offset=991)
    close_path.write_bytes(close)
    injection = prepare.TestSourceIdentityInjection(
        annotations_sha256=hashlib.sha256(annotations).hexdigest(),
        annotations_bytes=len(annotations),
        far_sha256=hashlib.sha256(far).hexdigest(),
        far_bytes=len(far),
        fixture_id="synthetic-ami-source-v1",
    )
    return {
        "annotations_zip": annotations_path,
        "far_wav": far_path,
        "close_wav": close_path,
        "close_sha256": hashlib.sha256(close).hexdigest(),
        "close_bytes": len(close),
        "accepted_terms": frozenset({"CC-BY-4.0"}),
        "test_source_identity_injection": injection,
    }


def _prepare(tmp_path: Path, *, output_name: str = "prepared"):
    sources = _sources(tmp_path)
    result = prepare.prepare_ami_conversation_fixture(
        **sources,
        output_dir=tmp_path / output_name,
    )
    return result, sources


def _receipt(corpus_path: Path) -> dict[str, object]:
    return json.loads(
        (corpus_path.parent / "preparation-receipt.json").read_text(encoding="ascii")
    )


def _semantic_drift_lock(tmp_path: Path) -> Path:
    root = json.loads(fixture.DEFAULT_LOCK.read_text(encoding="ascii"))
    source = next(
        item for item in root["sources"] if item["source_id"] == prepare.SOURCE_ID
    )
    source["selection_recipe"]["eligibility"][0] = "manual_word_timings_drifted"
    source["selection_recipe_sha256"] = fixture._canonical_sha256(
        source["selection_recipe"]
    )
    digest_input = dict(root)
    digest_input.pop("recipe_sha256")
    root["recipe_sha256"] = fixture._canonical_sha256(digest_input)
    path = tmp_path / "semantic-drift.lock.json"
    path.write_text(
        json.dumps(root, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    return path


def test_lock_binds_real_ami_materializer_and_decoder_contract():
    lock = fixture.load_fixture_lock()
    source = lock.source_by_id(prepare.SOURCE_ID)
    annotations, far, production = prepare._expected_source_identities(None)

    assert source.decoder_contract == prepare.DECODER_CONTRACT
    assert source.preparer_contract == prepare.PREPARER_CONTRACT
    assert dict(source.selection_recipe) == prepare._expected_source_recipe(
        seed=lock.selection_seed
    )
    assert prepare.REQUIRED_PREPARER_FILES <= set(source.preparer_files)
    assert annotations == (prepare.ANNOTATIONS_SHA256, prepare.ANNOTATIONS_BYTES)
    assert far == (prepare.FAR_SHA256, prepare.FAR_BYTES)
    assert production is True


def test_fully_redigested_semantic_recipe_drift_fails_before_publication(tmp_path):
    sources = _sources(tmp_path / "sources")
    output = tmp_path / "must-not-publish-drifted-recipe"
    with pytest.raises(prepare.AmiConversationPreparationError):
        prepare.prepare_ami_conversation_fixture(
            **sources,
            output_dir=output,
            lock_path=_semantic_drift_lock(tmp_path),
        )
    assert not output.exists()


def test_raw_annotations_derive_only_unambiguous_deterministic_windows(tmp_path):
    sources = _sources(tmp_path)
    annotation_raw = Path(sources["annotations_zip"]).read_bytes()
    members = ami_source._annotation_members(annotation_raw)
    far = ami_source._riff_pcm16_mono(Path(sources["far_wav"]).read_bytes())
    words, activities = ami_source._ordered_annotations(
        members,
        audio_samples=int(far.size),
    )
    seed = fixture.load_fixture_lock().selection_seed

    first = prepare._derive_candidates(
        words,
        activities,
        audio_samples=int(far.size),
        seed=seed,
    )
    second = prepare._derive_candidates(
        words,
        activities,
        audio_samples=int(far.size),
        seed=seed,
    )
    selected = prepare._select_windows(first, seed=seed)

    assert first == second
    assert Counter(item.kind for item in first)["isolated"] >= 8
    assert Counter(item.kind for item in first)["turn_transition"] >= 4
    assert [item.kind for item in selected] == ["isolated"] * 8 + [
        "turn_transition"
    ] * 4
    assert all(ami_source._different_speaker_overlap(item.words) == 0 for item in first)
    assert all("forbiddenoverlap" not in item.reference for item in first)
    assert activities
    assert all(
        not prepare._activity_intersects(
            activity,
            start_sample=item.start_sample,
            end_sample=item.end_sample,
        )
        for item in first
        for activity in activities
    )
    assert len({item.identity_sha256 for item in selected}) == 12
    assert all(
        not prepare._overlaps(left, right)
        for index, left in enumerate(selected)
        for right in selected[index + 1 :]
    )


def test_publishes_exact_paired_schema_v2_pcm_and_redacted_receipt(tmp_path):
    result, sources = _prepare(tmp_path)
    corpus = result.corpus
    receipt = _receipt(corpus.path)

    assert corpus.schema_version == 2
    assert len(corpus.cases) == 24
    assert receipt["decoder"]["contract"] == prepare.DECODER_CONTRACT
    assert receipt["decoder"]["production_evidence"] is False
    assert receipt["selection"]["parent_receipt_sha256"] is None
    assert receipt["selection"]["parent_corpus_manifest_sha256"] is None
    assert receipt["selection"]["selected_cases"] == 24
    assert receipt["totals"]["pcm_bytes"] == corpus.audio_bytes
    assert receipt["artifacts"][2]["local_sha256"] == sources["close_sha256"]
    assert receipt["artifacts"][2]["upstream_digest"] == sources["close_sha256"]

    cases = receipt["cases"]
    for index in range(0, 24, 2):
        close_case = corpus.cases[index]
        far_case = corpus.cases[index + 1]
        close_receipt = cases[index]
        far_receipt = cases[index + 1]
        assert close_receipt["attributes"]["channel"] == "close"
        assert far_receipt["attributes"]["channel"] == "far"
        assert close_case.expected_text == far_case.expected_text
        assert close_case.samples == far_case.samples
        assert close_case.audio_bytes != far_case.audio_bytes
        assert close_receipt["reference_sha256"] == far_receipt["reference_sha256"]
        assert close_receipt["attributes"]["window_sha256"] == far_receipt[
            "attributes"
        ]["window_sha256"]
        for field in ("window_index", "window_kind", "start_sample", "end_sample"):
            assert close_receipt["attributes"][field] == far_receipt["attributes"][field]
        assert close_case.samples == (
            close_receipt["attributes"]["end_sample"]
            - close_receipt["attributes"]["start_sample"]
        )

    first = cases[0]
    start = first["attributes"]["start_sample"]
    end = first["attributes"]["end_sample"]
    close_i16 = ami_source._riff_pcm16_mono(Path(sources["close_wav"]).read_bytes())
    expected = close_i16[start:end].astype(np.float32) * np.float32(1.0 / 32768.0)
    np.testing.assert_array_equal(
        np.frombuffer(corpus.cases[0].audio_bytes, dtype="<f4"),
        expected,
    )
    assert first["source_audio_sha256"] == hashlib.sha256(
        close_i16[start:end].astype("<i2", copy=False).tobytes(order="C")
    ).hexdigest()

    receipt_raw = (corpus.path.parent / "preparation-receipt.json").read_bytes()
    assert b"isolated0a" not in receipt_raw
    assert b"transition0lefta" not in receipt_raw
    assert str(sources["annotations_zip"]).encode() not in receipt_raw
    assert str(sources["close_wav"]).encode() not in receipt_raw
    assert stat.S_IMODE(corpus.path.parent.stat().st_mode) == 0o700
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o600
        for path in corpus.path.parent.iterdir()
    )
    assert load_corpus(corpus.path).digest == corpus.digest


def test_materialization_is_byte_deterministic_and_test_evidence_fails_production_gate(
    tmp_path,
):
    sources = _sources(tmp_path)
    first = prepare.prepare_ami_conversation_fixture(
        **sources,
        output_dir=tmp_path / "first",
    )
    second = prepare.prepare_ami_conversation_fixture(
        **sources,
        output_dir=tmp_path / "second",
    )

    assert first.corpus.path.read_bytes() == second.corpus.path.read_bytes()
    assert first.receipt_sha256 == second.receipt_sha256
    assert first.selected_windows_sha256 == second.selected_windows_sha256
    assert [case.audio_bytes for case in first.corpus.cases] == [
        case.audio_bytes for case in second.corpus.cases
    ]
    with pytest.raises(fixture.FixtureError):
        fixture.validate_private_source(
            first.corpus.path,
            lock=fixture.load_fixture_lock(),
            expected_source_id=prepare.SOURCE_ID,
        )


@pytest.mark.parametrize("target", ["annotations", "far", "close-receipt"])
def test_raw_source_or_local_receipt_tamper_fails_before_publication(tmp_path, target):
    sources = _sources(tmp_path)
    if target == "annotations":
        path = Path(sources["annotations_zip"])
        raw = bytearray(path.read_bytes())
        raw[-1] ^= 1
        path.write_bytes(raw)
    elif target == "far":
        path = Path(sources["far_wav"])
        raw = bytearray(path.read_bytes())
        raw[-1] ^= 1
        path.write_bytes(raw)
    else:
        sources["close_sha256"] = "0" * 64
    output = tmp_path / "must-not-exist"

    with pytest.raises(prepare.AmiConversationPreparationError):
        prepare.prepare_ami_conversation_fixture(**sources, output_dir=output)

    assert not output.exists()


def test_wrong_wav_contract_and_terms_fail_closed(tmp_path):
    bad_sources = _sources(tmp_path / "bad-wav", far_sample_rate=8_000)
    with pytest.raises(prepare.AmiConversationPreparationError):
        prepare.prepare_ami_conversation_fixture(
            **bad_sources,
            output_dir=tmp_path / "bad-wav-output",
        )

    sources = _sources(tmp_path / "bad-terms")
    sources["accepted_terms"] = frozenset()
    with pytest.raises(prepare.AmiConversationPreparationError):
        prepare.prepare_ami_conversation_fixture(
            **sources,
            output_dir=tmp_path / "bad-terms-output",
        )

    with pytest.raises(prepare.AmiConversationPreparationError):
        prepare._validated_output(
            Path.cwd() / "must-stay-outside-git",
            production_evidence=False,
        )


def test_close_and_far_must_be_distinct_source_recordings(tmp_path):
    sources = _sources(tmp_path)
    far = Path(sources["far_wav"])
    sources["close_wav"] = far
    sources["close_sha256"] = hashlib.sha256(far.read_bytes()).hexdigest()
    sources["close_bytes"] = far.stat().st_size

    with pytest.raises(prepare.AmiConversationPreparationError):
        prepare.prepare_ami_conversation_fixture(
            **sources,
            output_dir=tmp_path / "same-close-far",
        )


def test_safe_cli_failure_does_not_echo_private_paths(tmp_path, capsys):
    private = tmp_path / "private-path-canary"
    code = prepare.main(["--annotations-zip", str(private)])
    output = capsys.readouterr().out

    assert code == 2
    assert str(private) not in output
    assert json.loads(output) == dict(prepare._SAFE_ERROR)
