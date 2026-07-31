from __future__ import annotations

import hashlib
import io
import json
import stat
from itertools import count
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import public_stt_eval as public_eval
from tools.public_voice_fixtures import (
    DEFAULT_MANIFEST,
    EVIDENCE_SCOPE,
    load_manifest,
    source_records_for_suite,
)


class _FakeRecognizer:
    def __init__(
        self,
        outputs: list[object],
        *,
        mutate_path: Path | None = None,
    ) -> None:
        self._outputs = iter(outputs)
        self._mutate_path = mutate_path
        self.warm_calls = 0
        self.sample_types: list[type[object]] = []

    def warm(self) -> None:
        self.warm_calls += 1

    def transcribe(self, pcm: object, sample_rate: int) -> object:
        assert sample_rate > 0
        self.sample_types.append(type(pcm))
        if self._mutate_path is not None:
            self._mutate_path.write_bytes(b"changed")
            self._mutate_path = None
        output = next(self._outputs)
        if isinstance(output, BaseException):
            raise output
        return SimpleNamespace(text=output)


def _model(root: Path) -> Path:
    model = root / "model"
    (model / "nested").mkdir(parents=True)
    (model / "config.json").write_text('{"model":"test"}\n', encoding="utf-8")
    (model / "nested" / "weights.bin").write_bytes(b"deterministic test weights")
    return model


def _clock():
    ticks = count(step=1_000_000)
    return ticks.__next__


def _selection(
    fixture: dict[str, object],
    *,
    seed: str,
    expected_text: str,
) -> dict[str, object]:
    extract = fixture["extract"]
    assert isinstance(extract, dict)
    extractor = extract["type"]
    if extractor == "speech_commands_tar":
        target = str(extract["target"])
        label = "bird" if target == "_unknown_" else target
        filename = "noise.wav" if target == "_silence_" else "speaker_nohash_0.wav"
        member = f"{label}/{filename}"
        return {
            "archive_member": member,
            "archive_label": label,
            "speaker_id": None if target == "_silence_" else "speaker",
            "selection_rank_sha256": hashlib.sha256(
                seed.encode("utf-8") + b"\0" + member.encode("utf-8")
            ).hexdigest(),
        }
    if extractor == "direct_wav":
        return {}
    if extractor == "tar_member":
        return {"archive_member": extract["member"]}
    annotation = extract["annotation"]
    assert isinstance(annotation, dict)
    tokens = expected_text.split()
    vocal = annotation.get("required_vocal_type")
    return {
        "start_sec": float(extract["start_sec"]),
        "end_sec": float(extract["end_sec"]),
        "active_speakers": int(annotation["min_active_speakers"]),
        "max_concurrent_speakers": int(annotation["min_concurrent_speakers"]),
        "vocal_types": [] if vocal is None else [vocal],
        "annotation_tokens": tokens,
        "annotation_token_count": len(tokens),
        "annotation_transcript": " ".join(tokens),
    }


def _write_prepared_suite(
    root: Path,
    *,
    suite: str = "commands",
    manifest_path: Path = DEFAULT_MANIFEST,
) -> tuple[Path, list[dict[str, object]]]:
    manifest = load_manifest(manifest_path)
    fixtures = [
        fixture for fixture in manifest.fixtures if fixture["suite"] == suite
    ]
    rows: list[dict[str, object]] = []
    for index, raw_fixture in enumerate(fixtures):
        fixture = dict(raw_fixture)
        name = str(fixture["id"])
        path = root / f"{name}.npy"
        samples = np.full(100 + index, 0.01 * (index + 1), dtype=np.float32)
        np.save(path, samples)
        assertion = str(fixture["assertion"])
        expected = str(fixture.get("expected_text") or "")
        if assertion == "unknown_word":
            expected = "bird"
        source_ids = [str(fixture["source_id"])] + [
            str(value) for value in fixture.get("required_source_ids", [])
        ]
        rows.append(
            {
                "name": name,
                "file": path.name,
                "expectation": expected or assertion,
                "expected_text": expected,
                "assertion": assertion,
                "tags": list(fixture.get("tags", [])),
                "sample_rate": int(manifest.data["output_sample_rate"]),
                "sample_count": int(samples.size),
                "duration_sec": round(
                    samples.size / int(manifest.data["output_sample_rate"]), 6
                ),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "source_sha256": {
                    source_id: manifest.source_by_id[source_id]["sha256"]
                    for source_id in source_ids
                },
                "selection": _selection(
                    fixture,
                    seed=str(manifest.data["selection_seed"]),
                    expected_text=expected,
                ),
            }
        )
    metadata = {
        "schema_version": 2,
        "suite": suite,
        "purpose": manifest.data["purpose"],
        "evidence_scope": dict(EVIDENCE_SCOPE),
        "audio_committed": False,
        "manifest_sha256": manifest.digest,
        "selection_seed": manifest.data["selection_seed"],
        "cases": rows,
        "groups": [
            group for group in manifest.groups if group.get("suite") == suite
        ],
        "sources": list(source_records_for_suite(manifest, suite)),
    }
    path = root / "metadata.json"
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path, rows


def _factory_for(
    outputs: list[object],
    *,
    mutate_path: Path | None = None,
) -> tuple[public_eval.RecognizerFactory, list[_FakeRecognizer], list[Path]]:
    recognizers: list[_FakeRecognizer] = []
    model_paths: list[Path] = []

    def factory(model_path: Path) -> _FakeRecognizer:
        model_paths.append(model_path)
        recognizer = _FakeRecognizer(outputs, mutate_path=mutate_path)
        recognizers.append(recognizer)
        return recognizer

    return factory, recognizers, model_paths


def _eligible_outputs(rows: list[dict[str, object]]) -> list[str]:
    return [
        str(row["expected_text"])
        for row in rows
        if row["assertion"] in {"transcript", "empty_reference"}
    ]


def test_report_binds_exact_corpus_model_decoder_and_uses_raw_empty_metric(
    tmp_path, monkeypatch
):
    model = _model(tmp_path)
    metadata, rows = _write_prepared_suite(tmp_path)
    outputs = _eligible_outputs(rows)
    outputs[-1] = "background speech"
    factory, recognizers, model_paths = _factory_for(outputs)
    snapshot_root = tmp_path / "snapshots"
    snapshot_root.mkdir()
    real_load = np.load
    load_options: list[tuple[object, object]] = []

    def tracked_load(*args, **kwargs):
        load_options.append((kwargs.get("allow_pickle"), kwargs.get("mmap_mode")))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(np, "load", tracked_load)
    payload = public_eval.evaluate_public_stt(
        metadata,
        model,
        label="turbo-v3",
        snapshot_root=snapshot_root,
        recognizer_factory=factory,
        allow_test_recognizer_override=True,
        clock_ns=_clock(),
    )

    assert len(recognizers) == 1
    assert recognizers[0].warm_calls == 1
    assert len(model_paths) == 1
    assert model_paths[0] != model.resolve()
    assert model_paths[0].parent.parent == snapshot_root.resolve()
    assert not model_paths[0].exists()
    assert recognizers[0].sample_types == [np.ndarray] * 11
    assert load_options == [(False, None)] * 23  # 12 validation + 11 decode loads
    assert payload["evidence"] == {
        "kind": "whole_clip_decoder_smoke",
        "frame_replay": False,
        "aec_behavior": False,
        "live_hardware": False,
        "owner_voice": False,
        "held_out": False,
    }
    assert payload["corpus"]["cases"] == 12
    assert payload["corpus"]["sources"] == 1
    for field in (
        "manifest_sha256",
        "metadata_sha256",
        "source_set_sha256",
        "case_set_sha256",
    ):
        assert len(payload["corpus"][field]) == 64
    assert payload["model"]["files"] == 2
    assert payload["model"]["directories"] == 1
    assert len(payload["model"]["tree_sha256"]) == 64
    assert payload["model"]["evaluation_input"] == {
        "kind": "private_materialized_snapshot",
        "copy_strategy": "reflink_first_file_copy_fallback",
        "logical_bytes": payload["model"]["bytes"],
    }
    assert payload["decoder"]["kind"] == "test_recognizer_override"
    assert payload["decoder"]["production"] is False
    assert len(payload["decoder"]["callable_source_sha256"]) == 64
    assert "runtime" not in payload["decoder"]
    assert "model_loader" not in payload["decoder"]
    assert "transcribe" not in payload["decoder"]
    assert payload["evaluator"]["schema_version"] == 1
    assert payload["evaluator"]["kind"] == "public_stt_aggregate_evaluator"
    assert payload["evaluator"]["production_report"] is False
    assert set(payload["evaluator"]["files"]) == {
        "tools/public_stt_eval.py",
        "tools/recorded_stt_eval.py",
        "core/wer.py",
    }
    assert all(
        len(digest) == 64
        and set(digest) <= set("0123456789abcdef")
        for digest in payload["evaluator"]["files"].values()
    )
    assert payload["cases"] == {
        "all": 12,
        "eligible": 11,
        "transcript": 10,
        "empty_reference": 1,
        "excluded_assertions": {"unknown_word": 1},
    }
    assert payload["accuracy"]["empty_reference"] == {
        "clips": 1,
        "exact_empty": 0,
        "nonempty_transcripts": 1,
        "wer": 1.0,
        "cer": 1.0,
        "word_errors": 2,
        "char_errors": 16,
    }
    encoded = json.dumps(payload)
    for private_value in ("gsc-yes", "background speech", str(tmp_path)):
        assert private_value not in encoded
    assert "activation" not in encoded
    assert "no_action" not in encoded


def test_audio_change_restore_cannot_change_decoded_snapshot(tmp_path):
    model = _model(tmp_path)
    metadata, rows = _write_prepared_suite(tmp_path)
    first = next(row for row in rows if row["assertion"] == "transcript")
    audio_path = tmp_path / str(first["file"])
    original_bytes = audio_path.read_bytes()
    alternate = np.full(int(first["sample_count"]), -0.75, dtype=np.float32)
    alternate_buffer = io.BytesIO()
    np.save(alternate_buffer, alternate)
    alternate_bytes = alternate_buffer.getvalue()
    assert len(alternate_bytes) == len(original_bytes)
    outputs = iter(_eligible_outputs(rows))
    observed_first_samples: list[float] = []

    class ChangeRestoreRecognizer:
        def warm(self):
            audio_path.write_bytes(alternate_bytes)

        def transcribe(self, pcm, _sample_rate):
            if not observed_first_samples:
                observed_first_samples.append(float(pcm[0]))
                audio_path.write_bytes(original_bytes)
            return SimpleNamespace(text=next(outputs))

    def factory(_model_path):
        return ChangeRestoreRecognizer()

    public_eval.evaluate_public_stt(
        metadata,
        model,
        label="test",
        recognizer_factory=factory,
        allow_test_recognizer_override=True,
        clock_ns=_clock(),
    )

    assert observed_first_samples == [pytest.approx(0.01)]
    assert audio_path.read_bytes() == original_bytes


def test_model_change_restore_cannot_change_private_loaded_path(tmp_path):
    model = _model(tmp_path)
    metadata, rows = _write_prepared_suite(tmp_path)
    source_weights = model / "nested" / "weights.bin"
    original_bytes = source_weights.read_bytes()
    replacement = b"x" * len(original_bytes)
    outputs = iter(_eligible_outputs(rows))
    loaded_model_bytes: list[bytes] = []
    snapshot_write_errors: list[type[BaseException]] = []
    received_paths: list[Path] = []

    class LazyModelRecognizer:
        def __init__(self, model_path):
            self.model_path = model_path

        def warm(self):
            source_weights.write_bytes(replacement)

        def transcribe(self, _pcm, _sample_rate):
            if not loaded_model_bytes:
                snapshot_weights = self.model_path / "nested" / "weights.bin"
                loaded_model_bytes.append(snapshot_weights.read_bytes())
                try:
                    snapshot_weights.write_bytes(b"mutate")
                except OSError as exc:
                    snapshot_write_errors.append(type(exc))
                source_weights.write_bytes(original_bytes)
            return SimpleNamespace(text=next(outputs))

    def factory(model_path):
        received_paths.append(model_path)
        return LazyModelRecognizer(model_path)

    public_eval.evaluate_public_stt(
        metadata,
        model,
        label="test",
        recognizer_factory=factory,
        allow_test_recognizer_override=True,
        clock_ns=_clock(),
    )

    assert loaded_model_bytes == [original_bytes]
    assert snapshot_write_errors == [PermissionError]
    assert received_paths[0] != model.resolve()
    assert not received_paths[0].exists()
    assert source_weights.read_bytes() == original_bytes


def test_punctuation_only_empty_reference_is_raw_nonempty_but_normalized_exact(
    tmp_path,
):
    model = _model(tmp_path)
    metadata, rows = _write_prepared_suite(tmp_path)
    outputs = _eligible_outputs(rows)
    outputs[-1] = "..."
    factory, _, _ = _factory_for(outputs)

    payload = public_eval.evaluate_public_stt(
        metadata,
        model,
        label="test",
        recognizer_factory=factory,
        allow_test_recognizer_override=True,
        clock_ns=_clock(),
    )

    assert payload["accuracy"]["empty_reference"] == {
        "clips": 1,
        "exact_empty": 0,
        "nonempty_transcripts": 1,
        "wer": 0.0,
        "cer": 0.0,
        "word_errors": 0,
        "char_errors": 0,
    }


@pytest.mark.parametrize(
    "mutation",
    ["missing", "extra", "duplicate", "malformed", "float-rate", "float-source-size"],
)
def test_case_set_is_exact_and_malformed_rows_fail_before_model_load(
    tmp_path, mutation
):
    model = _model(tmp_path)
    metadata, _ = _write_prepared_suite(tmp_path)
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    if mutation == "missing":
        payload["cases"].pop()
    elif mutation == "extra":
        extra = dict(payload["cases"][0])
        extra["name"] = "extra"
        extra["file"] = "extra.npy"
        payload["cases"].append(extra)
    elif mutation == "duplicate":
        payload["cases"][-1]["name"] = payload["cases"][0]["name"]
    elif mutation == "malformed":
        payload["cases"][0]["unexpected"] = True
    elif mutation == "float-rate":
        payload["cases"][0]["sample_rate"] = 16000.0
    else:
        payload["sources"][0]["size_bytes"] = float(
            payload["sources"][0]["size_bytes"]
        )
    metadata.write_text(json.dumps(payload), encoding="utf-8")
    factory, recognizers, _ = _factory_for([])

    with pytest.raises(public_eval.PublicSttPrerequisiteError):
        public_eval.evaluate_public_stt(
            metadata,
            model,
            label="small",
            recognizer_factory=factory,
            allow_test_recognizer_override=True,
        )

    assert recognizers == []


def test_duplicate_json_keys_are_rejected_before_model_load(tmp_path):
    model = _model(tmp_path)
    metadata, _ = _write_prepared_suite(tmp_path)
    raw = metadata.read_text(encoding="utf-8")
    metadata.write_text(
        raw.replace('"schema_version": 2,', '"schema_version": 2, "schema_version": 2,'),
        encoding="utf-8",
    )
    factory, recognizers, _ = _factory_for([])

    with pytest.raises(public_eval.PublicSttPrerequisiteError):
        public_eval.evaluate_public_stt(
            metadata,
            model,
            label="small",
            recognizer_factory=factory,
            allow_test_recognizer_override=True,
        )

    assert recognizers == []


def test_every_case_hash_is_verified_even_when_assertion_is_excluded(tmp_path):
    model = _model(tmp_path)
    metadata, rows = _write_prepared_suite(tmp_path)
    excluded = next(row for row in rows if row["assertion"] == "unknown_word")
    (tmp_path / str(excluded["file"])).write_bytes(b"tampered")
    factory, recognizers, _ = _factory_for([])

    with pytest.raises(public_eval.PublicSttPrerequisiteError):
        public_eval.evaluate_public_stt(
            metadata,
            model,
            label="small",
            recognizer_factory=factory,
            allow_test_recognizer_override=True,
        )

    assert recognizers == []


@pytest.mark.parametrize("limit", ["case", "corpus"])
def test_audio_snapshot_memory_ceiling_fails_before_model_load(
    tmp_path, monkeypatch, limit
):
    model = _model(tmp_path)
    metadata, _ = _write_prepared_suite(tmp_path)
    if limit == "case":
        monkeypatch.setattr(public_eval, "_MAX_CASE_AUDIO_BYTES", 64)
    else:
        monkeypatch.setattr(public_eval, "_MAX_CORPUS_AUDIO_BYTES", 600)
    factory, recognizers, _ = _factory_for([])

    with pytest.raises(public_eval.PublicSttPrerequisiteError):
        public_eval.evaluate_public_stt(
            metadata,
            model,
            label="small",
            recognizer_factory=factory,
            allow_test_recognizer_override=True,
        )

    assert recognizers == []


def test_manifest_metadata_and_source_set_must_match_exactly(tmp_path):
    model = _model(tmp_path)
    metadata, _ = _write_prepared_suite(tmp_path)
    copied_manifest = tmp_path / "manifest.json"
    copied_manifest.write_bytes(DEFAULT_MANIFEST.read_bytes() + b"\n")
    factory, recognizers, _ = _factory_for([])

    with pytest.raises(public_eval.PublicSttPrerequisiteError):
        public_eval.evaluate_public_stt(
            metadata,
            model,
            label="small",
            manifest_path=copied_manifest,
            recognizer_factory=factory,
            allow_test_recognizer_override=True,
        )

    assert recognizers == []


@pytest.mark.parametrize("mutation", ["tokens", "impossible-concurrency"])
def test_annotation_selection_must_match_transcript_and_speaker_bounds(
    tmp_path, mutation
):
    model = _model(tmp_path)
    metadata, _ = _write_prepared_suite(tmp_path, suite="conversation")
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    if mutation == "tokens":
        row = next(item for item in payload["cases"] if item["assertion"] == "transcript")
        row["selection"]["annotation_tokens"] = ["candy"]
        row["selection"]["annotation_token_count"] = 1
        row["selection"]["annotation_transcript"] = "candy"
    else:
        row = next(item for item in payload["cases"] if item["name"].endswith("overlap"))
        row["selection"]["active_speakers"] = 1
        row["selection"]["max_concurrent_speakers"] = 4
    metadata.write_text(json.dumps(payload), encoding="utf-8")
    factory, recognizers, _ = _factory_for([])

    with pytest.raises(public_eval.PublicSttPrerequisiteError):
        public_eval.evaluate_public_stt(
            metadata,
            model,
            label="small",
            recognizer_factory=factory,
            allow_test_recognizer_override=True,
        )

    assert recognizers == []


def test_recognizer_override_is_rejected_without_explicit_test_opt_in(
    tmp_path, capsys
):
    model = _model(tmp_path)
    metadata, rows = _write_prepared_suite(tmp_path)
    factory, recognizers, _ = _factory_for(_eligible_outputs(rows))

    with pytest.raises(public_eval.PublicSttPrerequisiteError):
        public_eval.evaluate_public_stt(
            metadata,
            model,
            label="small",
            recognizer_factory=factory,
        )
    assert recognizers == []

    assert (
        public_eval.main(
            [
                "--metadata",
                str(metadata),
                "--model-path",
                str(model),
                "--label",
                "small",
            ],
            recognizer_factory=factory,
        )
        == 2
    )
    assert json.loads(capsys.readouterr().out) == public_eval._SAFE_ERROR
    assert recognizers == []


def test_production_decoder_binding_contains_only_production_contract():
    binding = public_eval._decoder_binding(
        public_eval.FasterWhisperEndpointRecognizer,
        allow_test_recognizer_override=False,
    )

    assert binding["kind"] == "production_faster_whisper"
    assert binding["production"] is True
    assert binding["runtime"] == {
        "platform": "linux",
        "accelerator": "nvidia-cuda",
        "device_index": 0,
        "compute_type": "float16",
    }
    assert binding["model_loader"]["local_files_only"] is True
    assert binding["transcribe"]["beam_size"] == 5
    assert "callable_source_sha256" not in binding


def test_evaluator_binding_is_versioned_deterministic_and_hashes_exact_files():
    first = public_eval._evaluator_binding(production_report=True)
    second = public_eval._evaluator_binding(production_report=True)
    repo_root = Path(public_eval.__file__).resolve().parents[1]

    assert first == second
    assert first["schema_version"] == 1
    assert first["production_report"] is True
    assert first["files"] == {
        relative: hashlib.sha256((repo_root / relative).read_bytes()).hexdigest()
        for relative in (
            "tools/public_stt_eval.py",
            "tools/recorded_stt_eval.py",
            "core/wer.py",
        )
    }


@pytest.mark.parametrize("target", ["case", "metadata", "model"])
def test_corpus_and_model_snapshots_are_reverified_after_decode(tmp_path, target):
    model = _model(tmp_path)
    metadata, rows = _write_prepared_suite(tmp_path)
    targets = {
        "case": tmp_path / str(rows[0]["file"]),
        "metadata": metadata,
        "model": model / "nested" / "weights.bin",
    }
    factory, _, _ = _factory_for(
        _eligible_outputs(rows),
        mutate_path=targets[target],
    )

    with pytest.raises(public_eval.PublicSttPrerequisiteError):
        public_eval.evaluate_public_stt(
            metadata,
            model,
            label="small",
            recognizer_factory=factory,
            allow_test_recognizer_override=True,
            clock_ns=_clock(),
        )


def test_empty_model_tree_is_rejected_before_factory(tmp_path):
    metadata, _ = _write_prepared_suite(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    factory, recognizers, _ = _factory_for([])

    with pytest.raises(public_eval.PublicSttPrerequisiteError):
        public_eval.evaluate_public_stt(
            metadata,
            empty,
            label="small",
            recognizer_factory=factory,
            allow_test_recognizer_override=True,
        )
    assert recognizers == []



def test_model_fingerprint_binds_symlink_target_and_content(tmp_path):
    model = tmp_path / "linked-model"
    model.mkdir()
    target = tmp_path / "target.bin"
    target.write_bytes(b"first")
    (model / "weights.bin").symlink_to(target)

    first = public_eval._fingerprint_model_tree(model)
    target.write_bytes(b"second")
    second = public_eval._fingerprint_model_tree(model)

    assert first["files"] == second["files"] == 1
    assert first["tree_sha256"] != second["tree_sha256"]


def test_cli_failure_is_detail_free_and_success_report_is_mode_600(
    tmp_path, capsys
):
    model = _model(tmp_path)
    metadata, rows = _write_prepared_suite(tmp_path)
    outputs: list[object] = _eligible_outputs(rows)
    outputs[0] = RuntimeError("SENTINEL PRIVATE TRANSCRIPT AND PATH")
    report = tmp_path / "SENTINEL_REPORT_PATH.json"
    factory, _, _ = _factory_for(outputs)

    exit_code = public_eval.main(
        [
            "--manifest",
            str(DEFAULT_MANIFEST),
            "--metadata",
            str(metadata),
            "--model-path",
            str(model),
            "--label",
            "small",
            "--output",
            str(report),
        ],
        recognizer_factory=factory,
        allow_test_recognizer_override=True,
        clock_ns=_clock(),
    )

    output = capsys.readouterr()
    stdout_payload = json.loads(output.out)
    assert exit_code == 1
    assert output.err == ""
    assert stdout_payload == json.loads(report.read_text(encoding="utf-8"))
    assert stdout_payload["accuracy"]["decode_errors"] == 1
    assert stat.S_IMODE(report.stat().st_mode) == 0o600
    assert "SENTINEL" not in output.out
    assert str(tmp_path) not in output.out

    corrupt = json.loads(metadata.read_text(encoding="utf-8"))
    corrupt["manifest_sha256"] = "0" * 64
    metadata.write_text(json.dumps(corrupt), encoding="utf-8")
    assert (
        public_eval.main(
            [
                "--metadata",
                str(metadata),
                "--model-path",
                str(model),
                "--label",
                "small",
            ],
            recognizer_factory=factory,
            allow_test_recognizer_override=True,
        )
        == 2
    )
    failure = capsys.readouterr()
    assert json.loads(failure.out) == public_eval._SAFE_ERROR
    assert failure.err == ""


@pytest.mark.parametrize("label", ["", "A", "bad/label", "a" * 65])
def test_label_is_bounded_before_model_construction(tmp_path, label):
    model = _model(tmp_path)
    metadata, _ = _write_prepared_suite(tmp_path)
    factory, recognizers, _ = _factory_for([])

    with pytest.raises(public_eval.PublicSttPrerequisiteError):
        public_eval.evaluate_public_stt(
            metadata,
            model,
            label=label,
            recognizer_factory=factory,
            allow_test_recognizer_override=True,
        )

    assert recognizers == []
