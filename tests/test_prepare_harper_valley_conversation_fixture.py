from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import wave

import pytest

from tools import prepare_harper_valley_conversation_fixture as prepare
from tools import public_conversation_fixture as fixture
from tools.streaming_stt.corpus import load_corpus


TERMS = frozenset({"CC-BY-4.0"})
MACHINE_SENTINEL = "machine-only-private-sentinel"
DIALOG_SENTINEL = "dialog-only-private-sentinel"
EMOTION_SENTINEL = "emotion-only-private-sentinel"
TIMESTAMP_SENTINEL = 1_712_345_678_901


def _run_git(root: Path, *arguments: str) -> bytes:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Fixture Test",
        "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
        "GIT_COMMITTER_NAME": "Fixture Test",
        "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
        "GIT_AUTHOR_DATE": "2026-08-02T00:00:00+00:00",
        "GIT_COMMITTER_DATE": "2026-08-02T00:00:00+00:00",
    }
    completed = subprocess.run(
        ("/usr/bin/git", "-C", str(root), *arguments),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    return completed.stdout


def _party(*, speaker_id: int, caller: bool, name: str) -> dict[str, object]:
    return {
        "arrival_time_ms": TIMESTAMP_SENTINEL - 10_000,
        "hangup_time_ms": TIMESTAMP_SENTINEL + 10_000,
        "metadata": {"first and last name": name} if caller else {"employee": "test"},
        "responses": [{"submit_time_ms": TIMESTAMP_SENTINEL}],
        "speaker_id": speaker_id,
        "survey_response": {"data": {}, "submit_time_ms": TIMESTAMP_SENTINEL},
    }


def _metadata(sid: str, *, task_type: str, speaker_id: int, name: str) -> bytes:
    value = {
        "agent": _party(speaker_id=10_000 + speaker_id, caller=False, name="agent"),
        "caller": _party(speaker_id=speaker_id, caller=True, name=name),
        "end_time_ms": TIMESTAMP_SENTINEL + 2_000,
        "labels": {"agent_mos": 5, "caller_mos": 5, "lhvb_script": "test"},
        "session": "synthetic-private-session",
        "sid": sid,
        "start_time_ms": TIMESTAMP_SENTINEL,
        # Multiple records deliberately prove that the authoritative contract
        # is "all task_type values agree", not "exactly one task record".
        "tasks": [
            {"task_type": task_type, "synthetic slot": "a"},
            {"task_type": task_type, "synthetic slot": "b"},
        ],
    }
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _turn(
    *,
    human: str,
    index: int = 1,
    role: str = "caller",
    channel: int = 1,
) -> dict[str, object]:
    return {
        "channel_index": channel,
        "dialog_acts": [DIALOG_SENTINEL],
        "duration_ms": 500,
        "emotion": {"positive": EMOTION_SENTINEL},
        "human_transcript": human,
        "index": index,
        "offset_ms": 100,
        "speaker_role": role,
        "start_ms": 100,
        "start_timestamp_ms": TIMESTAMP_SENTINEL,
        "transcript": MACHINE_SENTINEL,
        "word_durations_ms": [100],
        "word_offsets_ms": [0],
    }


def _transcript(human: str) -> bytes:
    rows = [
        _turn(
            human="synthetic agent reference",
            index=1,
            role="agent",
            channel=2,
        ),
        _turn(human=human, index=2),
    ]
    return (json.dumps(rows, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _wav(path: Path, *, tone: int) -> None:
    samples = [0] * 800
    samples.extend((tone if index % 2 == 0 else -tone) for index in range(4_000))
    samples.extend([0] * 800)
    raw = bytearray()
    for sample in samples:
        raw.extend(int(sample).to_bytes(2, "little", signed=True))
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(8_000)
        writer.writeframes(bytes(raw))


def _make_repository(tmp_path: Path) -> tuple[Path, tuple[str, ...]]:
    root = tmp_path / "private-harper-source"
    for relative in (
        "data/metadata",
        "data/transcript",
        "data/audio/caller",
        "data/audio/agent",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    private_values: list[str] = []
    ordinal = 0
    for task_index, task_type in enumerate(prepare.TASK_TYPES):
        cases = (
            ("clean", "known human reference for banking request"),
            ("clean", "please help complete this spoken banking request"),
            ("clean", "could you handle this natural conversation request"),
            ("clean", "i need assistance with this account request"),
            ("pii", "confidentialname needs help with this request"),
            ("timestamp", "please arrange this request at 12:30 pm"),
            ("marker", "[noise]"),
        )
        for kind, human in cases:
            sid = f"{ordinal:016x}"
            speaker_id = 20_000 + ordinal
            name = f"Confidentialname{ordinal} Privatesurname{ordinal}"
            metadata_path = root / "data" / "metadata" / f"{sid}.json"
            transcript_path = root / "data" / "transcript" / f"{sid}.json"
            caller_path = root / "data" / "audio" / "caller" / f"{sid}.wav"
            agent_path = root / "data" / "audio" / "agent" / f"{sid}.wav"
            metadata_path.write_bytes(
                _metadata(sid, task_type=task_type, speaker_id=speaker_id, name=name)
            )
            transcript_path.write_bytes(_transcript(human))
            _wav(caller_path, tone=500 + task_index * 100 + ordinal)
            _wav(agent_path, tone=700 + task_index * 100 + ordinal)
            if kind != "clean":
                private_values.extend((human, name))
            ordinal += 1

    _run_git(root, "init", "-q")
    _run_git(root, "add", ".")
    _run_git(root, "commit", "-q", "-m", "synthetic exact Harper fixture")
    for directory in (root, *(path for path in root.rglob("*") if path.is_dir())):
        if directory.name != ".git" and ".git" not in directory.parts:
            directory.chmod(0o700)
    for path in root.rglob("*"):
        if path.is_file() and ".git" not in path.parts:
            path.chmod(0o600)
    assert not _run_git(root, "status", "--porcelain")
    return root, tuple(private_values)


def _test_source_injection(root: Path) -> prepare.TestSourceInjection:
    return prepare.TestSourceInjection(
        fixture_id="synthetic-harper-v1",
        commit=_run_git(root, "rev-parse", "HEAD").decode("ascii").strip(),
        root_tree=_run_git(root, "rev-parse", "HEAD^{tree}").decode("ascii").strip(),
        metadata_tree=_run_git(root, "rev-parse", "HEAD:data/metadata")
        .decode("ascii")
        .strip(),
        transcript_tree=_run_git(root, "rev-parse", "HEAD:data/transcript")
        .decode("ascii")
        .strip(),
        caller_audio_tree=_run_git(root, "rev-parse", "HEAD:data/audio/caller")
        .decode("ascii")
        .strip(),
        agent_audio_tree=_run_git(root, "rev-parse", "HEAD:data/audio/agent")
        .decode("ascii")
        .strip(),
        expected_sessions=56,
    )


def _prepare(tmp_path: Path, *, output_name: str = "prepared"):
    source, private_values = _make_repository(tmp_path)
    injection = _test_source_injection(source)
    result = prepare.prepare_harper_valley_conversation_fixture(
        repository=source,
        output_dir=tmp_path / output_name,
        accepted_terms=TERMS,
        test_source_injection=injection,
    )
    return result, source, private_values


def _tree_payloads(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.iterdir())
        if path.is_file()
    }


def test_materializes_exact_private_24_case_source_without_production_claim(tmp_path):
    result, _source, private_values = _prepare(tmp_path)
    corpus = result.corpus
    receipt_path = corpus.path.parent / "preparation-receipt.json"
    receipt_raw = receipt_path.read_bytes()
    receipt = json.loads(receipt_raw)

    assert corpus.schema_version == 2
    assert len(corpus.cases) == 24
    assert corpus.audio_bytes == 24 * 8_000 * 4
    assert corpus.path.stat().st_mode & 0o777 == 0o600
    assert corpus.path.parent.stat().st_mode & 0o777 == 0o700
    assert receipt_path.stat().st_mode & 0o777 == 0o600
    assert hashlib.sha256(receipt_raw).hexdigest() == result.receipt_sha256
    assert receipt["source_id"] == "harper-valley"
    assert receipt["decoder"]["contract"] == (
        "test-source-synthetic-harper-v1-wave-f32le-v1"
    )
    assert receipt["decoder"]["production_evidence"] is False
    assert receipt["selection"]["eligible_rows"] == 32
    assert receipt["selection"]["selected_cases"] == 24
    assert receipt["selection"]["parent_receipt_sha256"] is None
    assert receipt["selection"]["parent_corpus_manifest_sha256"] is None
    assert Counter(
        item["attributes"]["task_type_index"] for item in receipt["cases"]
    ) == {index: 3 for index in range(8)}
    assert len({item["speaker_sha256"] for item in receipt["cases"]}) == 24
    assert all(item["attributes"]["pii_scan_passed"] for item in receipt["cases"])
    assert load_corpus(corpus.path).digest == corpus.digest
    with pytest.raises(fixture.FixtureError):
        fixture.validate_private_source(
            corpus.path,
            lock=fixture.load_fixture_lock(),
            expected_source_id="harper-valley",
        )

    encoded_receipt = receipt_raw.decode("ascii")
    encoded_corpus = corpus.path.read_text(encoding="utf-8")
    for private in (
        *private_values,
        MACHINE_SENTINEL,
        DIALOG_SENTINEL,
        EMOTION_SENTINEL,
        str(tmp_path),
        str(TIMESTAMP_SENTINEL),
        ".wav",
        "private-harper-source",
    ):
        assert private not in encoded_receipt
        assert private not in encoded_corpus
    assert all(
        case.expected_text.startswith(
            (
                "known human",
                "please help",
                "could you",
                "i need",
            )
        )
        for case in corpus.cases
    )


def test_selection_receipt_pcm_and_manifest_are_byte_exact_across_runs(tmp_path):
    first, source, _private = _prepare(tmp_path, output_name="first")
    second = prepare.prepare_harper_valley_conversation_fixture(
        repository=source,
        output_dir=tmp_path / "second",
        accepted_terms=TERMS,
        test_source_injection=_test_source_injection(source),
    )

    assert first.receipt_sha256 == second.receipt_sha256
    assert first.metadata_sha256 == second.metadata_sha256
    assert first.selected_rows_sha256 == second.selected_rows_sha256
    assert first.corpus.digest == second.corpus.digest
    assert _tree_payloads(first.corpus.path.parent) == _tree_payloads(
        second.corpus.path.parent
    )


@pytest.mark.parametrize(
    "marker",
    (
        "baby",
        "ringing",
        "laughter",
        "kids",
        "music",
        "noise",
        "unintelligible",
        "dogs",
        "cough",
    ),
)
def test_known_marker_only_references_are_not_hard_wer_eligible(marker):
    names = ("privatecaller", "privatesurname")

    assert not prepare._reference_is_safe(f"[{marker}]", names)
    assert not prepare._reference_is_safe(f"[{marker}] [{marker.upper()}]", names)
    assert prepare._reference_is_safe(f"spoken words [{marker}]", names)


def test_authoritative_channel_and_start_ms_override_role_and_offset_metadata():
    caller_on_channel_one = _turn(
        human="lexical caller words",
        role="agent",
        channel=1,
    )
    caller_on_channel_one["start_ms"] = 175
    caller_on_channel_one["offset_ms"] = 925
    agent_on_channel_two = _turn(
        human="must not become a caller case",
        index=2,
        role="caller",
        channel=2,
    )
    candidates = prepare._transcript_candidates(
        [caller_on_channel_one, agent_on_channel_two],
        sid="0" * 16,
        task_type=prepare.TASK_TYPES[0],
        speaker_id="int:7",
        names=("privatecaller", "privatesurname"),
        audio_entry=prepare.GitEntry(
            relative="data/audio/caller/0000000000000000.wav",
            oid="0" * 40,
        ),
    )

    assert len(candidates) == 1
    assert candidates[0].speaker_id == "int:7"
    assert candidates[0].start_ms == 175
    assert candidates[0].duration_ms == 500
    assert candidates[0].audio_relative.startswith("data/audio/caller/")


def test_dirty_or_blob_tampered_source_fails_before_publication(tmp_path):
    source, _private = _make_repository(tmp_path)
    injection = _test_source_injection(source)
    transcript = next((source / "data" / "transcript").iterdir())
    transcript.write_bytes(transcript.read_bytes() + b" ")
    transcript.chmod(0o600)

    output = tmp_path / "must-not-exist"
    with pytest.raises(prepare.HarperPreparationError):
        prepare.prepare_harper_valley_conversation_fixture(
            repository=source,
            output_dir=output,
            accepted_terms=TERMS,
            test_source_injection=injection,
        )
    assert not output.exists()


@pytest.mark.parametrize(
    "mutator",
    (
        lambda metadata, transcript: metadata["tasks"].append(
            {"task_type": "pay bill"}
        ),
        lambda metadata, transcript: transcript[1].__setitem__(
            "human_transcript", None
        ),
        lambda metadata, transcript: transcript[1].__setitem__(
            "channel_index", 3
        ),
    ),
)
def test_authoritative_task_human_reference_and_timing_contracts_fail_closed(
    tmp_path, mutator
):
    source, _private = _make_repository(tmp_path)
    metadata_path = next((source / "data" / "metadata").iterdir())
    sid = metadata_path.stem
    transcript_path = source / "data" / "transcript" / f"{sid}.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    mutator(metadata, transcript)
    metadata_path.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
    transcript_path.write_text(json.dumps(transcript, sort_keys=True), encoding="utf-8")
    metadata_path.chmod(0o600)
    transcript_path.chmod(0o600)
    _run_git(source, "add", ".")
    _run_git(source, "commit", "-q", "-m", "synthetic invalid source")
    injection = _test_source_injection(source)

    with pytest.raises(prepare.HarperPreparationError):
        prepare.prepare_harper_valley_conversation_fixture(
            repository=source,
            output_dir=tmp_path / "rejected",
            accepted_terms=TERMS,
            test_source_injection=injection,
        )


def test_private_checkout_terms_and_outside_git_output_are_required(
    tmp_path,
):
    source, _private = _make_repository(tmp_path)
    injection = _test_source_injection(source)

    source.chmod(0o755)
    with pytest.raises(prepare.HarperPreparationError):
        prepare.prepare_harper_valley_conversation_fixture(
            repository=source,
            output_dir=tmp_path / "public-source-rejected",
            accepted_terms=TERMS,
            test_source_injection=injection,
        )
    source.chmod(0o700)
    with pytest.raises(prepare.HarperPreparationError):
        prepare.prepare_harper_valley_conversation_fixture(
            repository=source,
            output_dir=source / "inside-git",
            accepted_terms=TERMS,
            test_source_injection=injection,
        )
    with pytest.raises(prepare.HarperPreparationError):
        prepare.prepare_harper_valley_conversation_fixture(
            repository=source,
            output_dir=tmp_path / "terms-rejected",
            accepted_terms=frozenset(),
            test_source_injection=injection,
        )


def test_cli_failure_is_detail_free(tmp_path, capsys):
    private_path = tmp_path / "private-source-name"
    assert (
        prepare.main(
            [
                "--repository",
                str(private_path),
                "--output-dir",
                str(tmp_path / "private-output-name"),
                "--accept-term",
                "CC-BY-4.0",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "ok": False,
        "error": "harper_valley_fixture_prerequisites_unavailable",
    }
    assert "private-source-name" not in captured.out + captured.err
    assert "private-output-name" not in captured.out + captured.err


def test_cli_parse_failure_does_not_echo_unrecognized_private_value(capsys):
    assert prepare.main(["--unexpected-private-value"]) == 2

    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "ok": False,
        "error": "harper_valley_fixture_prerequisites_unavailable",
    }
    assert "unexpected-private-value" not in captured.out + captured.err


def test_executable_production_pins_match_matrix_and_lock():
    dataset = prepare.dataset_by_id(prepare.DATASET_ID)
    prepare._validate_catalog_contract(dataset)
    source = fixture.load_fixture_lock().source_by_id("harper-valley")
    assert source.preparer_contract == prepare.PREPARER_CONTRACT
    assert source.preparer_files == prepare.PREPARER_FILES
    assert dict(source.selection_recipe) == prepare._expected_source_recipe(dataset)
    assert source.selection_recipe["seed"] == dataset.selection.seed == (
        "speaker-public-eval-v1-2026-08-01"
    )
    assert source.selection_recipe["seed"] != fixture.load_fixture_lock().selection_seed
    assert "no_nonlexical_marker_only" in source.selection_recipe["eligibility"]
    assert prepare.EXPECTED_ELIGIBLE_ROWS == 8_000
    assert prepare.EXPECTED_METADATA_SHA256 == (
        "f3f1333be2ec6f35186e1bb791c39eb4eb2ddf32493ce5bdaad3cb8ff0e2a260"
    )
    assert prepare.EXPECTED_ELIGIBLE_SET_SHA256 == (
        "6b3774c65feb03ae1b65a3456dd23c7fec452fd60d19a98da136ece3db2cb59a"
    )
    assert prepare.EXPECTED_SELECTED_ROWS_SHA256 == (
        "b4bdc5ad4cfded0551b7feb0699dc626168cddf8c5dfd38b44dff5c44ad83820"
    )
    assert prepare.DECODER_CONTRACT.startswith("python-stdlib-wave-pcm")
    assert stat.S_ISREG(Path(prepare.__file__).stat().st_mode)
