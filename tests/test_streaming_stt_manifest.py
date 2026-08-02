from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import sys

import pytest

from tests.streaming_stt_helpers import (
    REPO_ROOT,
    bound_file,
    scripted_case,
    write_fixture,
)
from tools.streaming_stt import bounded_io
from tools.streaming_stt.bounded_io import BoundedReadError, hash_regular_bounded
from tools.streaming_stt import corpus as corpus_module
from tools.streaming_stt.corpus import (
    CorpusError,
    load_corpus,
    verify_corpus_snapshot,
)
from tools.streaming_stt import manifest as manifest_module
from tools.streaming_stt.manifest import (
    MOONSHINE_ADAPTER,
    MOONSHINE_ARTIFACT_NAMES,
    MOONSHINE_EXTERNAL_ENDPOINT_ADAPTER,
    ManifestError,
    MoonshineExternalEndpointConfig,
    artifact_maximum_bytes,
    load_worker_manifest,
)
from tools.streaming_stt.manifest import MAX_FAKE_ARTIFACT_BYTES
from tools.streaming_stt.source_bundle import (
    WORKER_SOURCE_FILES,
    SourceBundleError,
    stage_worker_source_bundle,
    verify_source_bundle,
)


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    manifest, corpus, _ = write_fixture(
        tmp_path,
        [
            {
                "values": [0.0, 0.1, -0.1, 0.0],
                "expected_text": "stop now",
                "commands": ["stop"],
            }
        ],
        [
            scripted_case(
                partials=[(2, "stop", 100.0, 2.0)],
                final="stop now",
            )
        ],
    )
    return manifest, corpus


def _schema_v4_corpus(tmp_path: Path) -> Path:
    audio = tmp_path / "negative.f32le"
    audio.write_bytes(bytes(16))
    receipt = tmp_path / "preparation-receipt.json"
    receipt.write_bytes(b'{"schema_version":1,"kind":"command-noise"}\n')
    corpus = tmp_path / "command-noise-corpus.json"
    corpus.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "purpose": "schema v4 loader contract",
                "provenance": {
                    "kind": "public-command-noise-v1",
                    "suite": "short-command-noise",
                    "manifest_sha256": "a" * 64,
                    "metadata_sha256": hashlib.sha256(
                        receipt.read_bytes()
                    ).hexdigest(),
                    "source_set_sha256": "b" * 64,
                },
                "cases": [
                    {
                        "id": "negative",
                        "file": audio.name,
                        "sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
                        "samples": 4,
                        "expected_text": "",
                        "assertion": "speech_negative",
                        "commands": [],
                        "forbidden_commands": ["turn on", "turn off"],
                        "tags": ["speech-negative"],
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return corpus


def _moonshine_manifest(
    tmp_path: Path,
    monkeypatch,
    *,
    model_arch: str = "tiny-streaming",
) -> Path:
    venv = tmp_path / "candidate-venv"
    python = venv / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.symlink_to(Path(sys.executable).resolve(strict=True))
    marker = venv / "pyvenv.cfg"
    marker.write_text(
        "home = /usr/bin\ninclude-system-site-packages = false\n",
        encoding="utf-8",
    )

    receipt = tmp_path / "runtime-receipt.json"
    receipt.write_text('{"schema_version":1}\n', encoding="utf-8")
    wheel = tmp_path / "moonshine_voice-0.1.0-py3-none-manylinux_2_34_x86_64.whl"
    wheel.write_bytes(b"exact-test-wheel")
    model_root = tmp_path / "model"
    model_root.mkdir()
    model_files = {
        "model-adapter": model_root / "adapter.ort",
        "model-cross-kv": model_root / "cross_kv.ort",
        "model-decoder-kv": model_root / "decoder_kv.ort",
        "model-encoder": model_root / "encoder.ort",
        "model-frontend": model_root / "frontend.ort",
        "model-config": model_root / "streaming_config.json",
        "model-tokenizer": model_root / "tokenizer.bin",
    }
    for index, path in enumerate(model_files.values(), start=1):
        path.write_bytes(bytes([index]))

    monkeypatch.setattr(
        manifest_module,
        "_MOONSHINE_RELEASE_RECEIPT",
        (hashlib.sha256(wheel.read_bytes()).hexdigest(), wheel.stat().st_size),
    )
    monkeypatch.setattr(
        manifest_module,
        "_MOONSHINE_MODEL_RECEIPTS",
        {
            model_arch: {
                name: (
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    path.stat().st_size,
                )
                for name, path in model_files.items()
            }
        },
    )
    artifacts_by_name = {
        "runtime-receipt": receipt,
        "venv-marker": marker,
        "release-wheel": wheel,
        **model_files,
    }
    payload = {
        "schema_version": 2,
        "model_id": "moonshine-tiny-streaming-en-0.1.0",
        "adapter": MOONSHINE_ADAPTER,
        "adapter_config": {
            "package_version": "0.1.0",
            "api_version": 30000,
            "model_arch": model_arch,
            "provider": "cpu",
            "language": "en",
        },
        "python": bound_file(python.resolve(strict=True), allow_path=python),
        "worker": bound_file(REPO_ROOT / "tools" / "streaming_stt" / "worker.py"),
        "artifacts": [
            {"name": name, **bound_file(artifacts_by_name[name])}
            for name in MOONSHINE_ARTIFACT_NAMES
        ],
        "limits": {
            "startup_timeout_sec": 30.0,
            "case_timeout_sec": 60.0,
        },
    }
    manifest = tmp_path / "moonshine-worker-manifest.json"
    manifest.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return manifest


def _moonshine_external_endpoint_manifest(
    tmp_path: Path,
    monkeypatch,
    *,
    segmentation_mode: str = "external-presegmented",
    vad_threshold: float = 0.0,
) -> Path:
    manifest = _moonshine_manifest(tmp_path, monkeypatch)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload.update(
        {
            "schema_version": 7,
            "model_id": "moonshine-tiny-streaming-external-endpoint-en-0.1.0",
            "adapter": MOONSHINE_EXTERNAL_ENDPOINT_ADAPTER,
        }
    )
    payload["adapter_config"].update(
        {
            "segmentation_mode": segmentation_mode,
            "endpoint_owner": "external-input-boundary",
            "vad_threshold": vad_threshold,
            "vad_max_segment_duration_sec": 136.0,
            "vad_hop_size_samples": 512,
            "streaming_chunk_samples": 1280,
            "online_partial_interval_ms": 500,
            "authoritative_alignment_samples": 2560,
            "tail_alignment_policy": "zero-pad-to-vad-model-lcm",
            "finalization_policy": "verified-native-free-authoritative-batch-v2",
            "maximum_source_samples": 2_097_152,
        }
    )
    manifest.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return manifest


def test_manifest_binds_exact_interpreter_worker_and_artifact(tmp_path):
    manifest_path, _ = _fixture(tmp_path)

    manifest = load_worker_manifest(manifest_path)

    assert manifest.model_id == "fake-stream-v1"
    assert manifest.adapter == "fake-json-v1"
    assert (
        manifest.worker.sha256
        == hashlib.sha256(manifest.worker.path.resolve().read_bytes()).hexdigest()
    )
    assert set(manifest.artifact_by_name) == {"fake-script"}
    assert len(manifest.digest) == 64


def test_manifest_v2_binds_closed_moonshine_runtime_and_retains_venv_launcher(
    tmp_path,
    monkeypatch,
):
    path = _moonshine_manifest(tmp_path, monkeypatch)

    manifest = load_worker_manifest(path)

    assert manifest.schema_version == 2
    assert manifest.adapter == MOONSHINE_ADAPTER
    assert manifest.adapter_config is not None
    assert manifest.adapter_config.as_dict() == {
        "package_version": "0.1.0",
        "api_version": 30000,
        "model_arch": "tiny-streaming",
        "provider": "cpu",
        "language": "en",
    }
    assert tuple(manifest.artifact_by_name) == MOONSHINE_ARTIFACT_NAMES
    assert manifest.python.path == tmp_path / "candidate-venv" / "bin" / "python"
    assert manifest.python.path.is_symlink()


def test_manifest_v7_binds_exact_external_presegmented_moonshine_receipt(
    tmp_path,
    monkeypatch,
):
    path = _moonshine_external_endpoint_manifest(tmp_path, monkeypatch)

    manifest = load_worker_manifest(path)

    assert manifest.schema_version == 7
    assert manifest.adapter == MOONSHINE_EXTERNAL_ENDPOINT_ADAPTER
    assert isinstance(
        manifest.adapter_config,
        MoonshineExternalEndpointConfig,
    )
    assert manifest.adapter_config.as_dict() == {
        "package_version": "0.1.0",
        "api_version": 30000,
        "model_arch": "tiny-streaming",
        "provider": "cpu",
        "language": "en",
        "segmentation_mode": "external-presegmented",
        "endpoint_owner": "external-input-boundary",
        "vad_threshold": 0.0,
        "vad_max_segment_duration_sec": 136.0,
        "vad_hop_size_samples": 512,
        "streaming_chunk_samples": 1280,
        "online_partial_interval_ms": 500,
        "authoritative_alignment_samples": 2560,
        "tail_alignment_policy": "zero-pad-to-vad-model-lcm",
        "finalization_policy": "verified-native-free-authoritative-batch-v2",
        "maximum_source_samples": 2_097_152,
    }
    assert tuple(manifest.artifact_by_name) == MOONSHINE_ARTIFACT_NAMES
    assert all(
        artifact_maximum_bytes(MOONSHINE_EXTERNAL_ENDPOINT_ADAPTER, name)
        == artifact_maximum_bytes(MOONSHINE_ADAPTER, name)
        for name in MOONSHINE_ARTIFACT_NAMES
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("segmentation_mode", "legacy"),
        ("endpoint_owner", "worker-vad"),
        ("vad_threshold", 0),
        ("vad_threshold", -0.0),
        ("vad_max_segment_duration_sec", 136),
        ("vad_hop_size_samples", True),
        ("vad_hop_size_samples", 256),
        ("streaming_chunk_samples", True),
        ("streaming_chunk_samples", 2560),
        ("online_partial_interval_ms", True),
        ("online_partial_interval_ms", 80),
        ("authoritative_alignment_samples", False),
        ("authoritative_alignment_samples", 1280),
        ("tail_alignment_policy", "drop-tail"),
        ("finalization_policy", "stop-only"),
        ("maximum_source_samples", False),
        ("maximum_source_samples", 2_097_153),
    ],
)
def test_manifest_v7_rejects_wrong_external_endpoint_receipt_value_or_type(
    tmp_path,
    monkeypatch,
    field,
    value,
):
    path = _moonshine_external_endpoint_manifest(tmp_path, monkeypatch)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["adapter_config"][field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ManifestError):
        load_worker_manifest(path)


@pytest.mark.parametrize(
    ("segmentation_mode", "vad_threshold"),
    [("internal-vad", 0.5), ("internal-vad", 0.0), ("external-presegmented", 0.5)],
)
def test_manifest_v7_rejects_every_non_external_zero_threshold_profile(
    tmp_path,
    monkeypatch,
    segmentation_mode,
    vad_threshold,
):
    path = _moonshine_external_endpoint_manifest(
        tmp_path,
        monkeypatch,
        segmentation_mode=segmentation_mode,
        vad_threshold=vad_threshold,
    )

    with pytest.raises(ManifestError):
        load_worker_manifest(path)


@pytest.mark.parametrize("mutation", ["missing", "extra", "null"])
def test_manifest_v7_rejects_open_or_null_external_endpoint_config(
    tmp_path,
    monkeypatch,
    mutation,
):
    path = _moonshine_external_endpoint_manifest(tmp_path, monkeypatch)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "missing":
        del payload["adapter_config"]["tail_alignment_policy"]
    elif mutation == "extra":
        payload["adapter_config"]["unbound_option"] = "unsafe"
    else:
        payload["adapter_config"]["endpoint_owner"] = None
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ManifestError):
        load_worker_manifest(path)


@pytest.mark.parametrize(
    ("schema_version", "adapter"),
    [
        (2, MOONSHINE_EXTERNAL_ENDPOINT_ADAPTER),
        (7, MOONSHINE_ADAPTER),
    ],
)
def test_manifest_rejects_moonshine_adapter_schema_cross_pair(
    tmp_path,
    monkeypatch,
    schema_version,
    adapter,
):
    path = (
        _moonshine_manifest(tmp_path, monkeypatch)
        if schema_version == 2
        else _moonshine_external_endpoint_manifest(tmp_path, monkeypatch)
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["adapter"] = adapter
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ManifestError):
        load_worker_manifest(path)


@pytest.mark.parametrize("model_arch", ["small-streaming", "medium-streaming"])
def test_manifest_v2_accepts_exact_pinned_nondefault_streaming_model(
    tmp_path,
    monkeypatch,
    model_arch,
):
    path = _moonshine_manifest(
        tmp_path,
        monkeypatch,
        model_arch=model_arch,
    )

    manifest = load_worker_manifest(path)

    assert manifest.adapter_config is not None
    assert manifest.adapter_config.model_arch == model_arch


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("package_version", "0.0.69"),
        ("api_version", 29999),
        ("model_arch", "base"),
        ("provider", "cuda"),
        ("language", "ro"),
    ],
)
def test_manifest_v2_rejects_open_or_unpinned_moonshine_config(
    tmp_path,
    monkeypatch,
    field,
    value,
):
    path = _moonshine_manifest(tmp_path, monkeypatch)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["adapter_config"][field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ManifestError):
        load_worker_manifest(path)


@pytest.mark.parametrize(
    "marker_text",
    [
        "home = /usr/bin\ninclude-system-site-packages = true\n",
        "home = /usr/bin\n",
        (
            "include-system-site-packages = false\n"
            "include-system-site-packages = false\n"
        ),
    ],
)
def test_manifest_v2_requires_one_disabled_system_site_setting(
    tmp_path,
    monkeypatch,
    marker_text,
):
    path = _moonshine_manifest(tmp_path, monkeypatch)
    payload = json.loads(path.read_text(encoding="utf-8"))
    marker = tmp_path / "candidate-venv" / "pyvenv.cfg"
    marker.write_text(marker_text, encoding="utf-8")
    artifact = next(
        item for item in payload["artifacts"] if item["name"] == "venv-marker"
    )
    artifact.update(bound_file(marker))
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ManifestError):
        load_worker_manifest(path)


@pytest.mark.parametrize("mutation", ["reorder", "basename", "model-parent", "wheel"])
def test_manifest_v2_rejects_changed_moonshine_provenance_or_layout(
    tmp_path,
    monkeypatch,
    mutation,
):
    path = _moonshine_manifest(tmp_path, monkeypatch)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "reorder":
        payload["artifacts"][-1], payload["artifacts"][-2] = (
            payload["artifacts"][-2],
            payload["artifacts"][-1],
        )
    else:
        selected = {
            "basename": "model-adapter",
            "model-parent": "model-encoder",
            "wheel": "release-wheel",
        }[mutation]
        artifact = next(
            item for item in payload["artifacts"] if item["name"] == selected
        )
        original = Path(artifact["path"])
        if mutation == "basename":
            replacement = original.with_name("wrong.ort")
            replacement.write_bytes(original.read_bytes())
        elif mutation == "model-parent":
            replacement = tmp_path / "other-model" / original.name
            replacement.parent.mkdir()
            replacement.write_bytes(original.read_bytes())
        else:
            original.write_bytes(b"changed-wheel")
            replacement = original
        artifact.update(bound_file(replacement))
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ManifestError):
        load_worker_manifest(path)


def test_private_source_bundle_is_deterministic_exact_and_single_link(tmp_path):
    first = stage_worker_source_bundle(REPO_ROOT, tmp_path / "bundle-a")
    second = stage_worker_source_bundle(REPO_ROOT, tmp_path / "bundle-b")

    assert first.tree_sha256 == second.tree_sha256
    assert tuple(item.relative_path for item in first.files) == WORKER_SOURCE_FILES
    assert stat.S_IMODE(first.root.stat().st_mode) == 0o700
    for item in first.files:
        metadata = (first.root / item.relative_path).lstat()
        assert stat.S_ISREG(metadata.st_mode)
        assert stat.S_IMODE(metadata.st_mode) == 0o400
        assert metadata.st_nlink == 1

    extra = first.root / "unexpected.py"
    extra.write_text("raise RuntimeError\n", encoding="utf-8")
    with pytest.raises(SourceBundleError):
        verify_source_bundle(first, required_files=WORKER_SOURCE_FILES)


@pytest.mark.parametrize("target", ["worker", "artifact"])
def test_manifest_rejects_changed_bound_file(tmp_path, target):
    manifest_path, _ = _fixture(tmp_path)
    manifest = load_worker_manifest(manifest_path)
    if target == "worker":
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["worker"]["sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ManifestError):
            load_worker_manifest(manifest_path)
    else:
        path = manifest.artifact_by_name["fake-script"].path
        path.write_bytes(path.read_bytes() + b"x")
        with pytest.raises(ManifestError):
            load_worker_manifest(manifest_path)


def test_manifest_rejects_duplicate_keys_and_unknown_adapter(tmp_path):
    manifest_path, _ = _fixture(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["adapter"] = "moonshine-unresolved"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ManifestError):
        load_worker_manifest(manifest_path)

    manifest_path.write_text(
        '{"schema_version":1,"schema_version":1}',
        encoding="utf-8",
    )
    with pytest.raises(ManifestError):
        load_worker_manifest(manifest_path)


def test_manifest_rejects_symlinked_artifact(tmp_path):
    manifest_path, _ = _fixture(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    original = Path(payload["artifacts"][0]["path"])
    link = tmp_path / "artifact-link.json"
    link.symlink_to(original)
    payload["artifacts"][0]["path"] = str(link)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ManifestError):
        load_worker_manifest(manifest_path)


@pytest.mark.parametrize("target_name", ["manifest", "artifact", "corpus", "pcm"])
def test_stable_input_readers_reject_hardlinks(tmp_path, target_name):
    manifest_path, corpus_path = _fixture(tmp_path)
    corpus_payload = json.loads(corpus_path.read_text(encoding="utf-8"))
    targets = {
        "manifest": manifest_path,
        "artifact": Path(
            json.loads(manifest_path.read_text(encoding="utf-8"))["artifacts"][0][
                "path"
            ]
        ),
        "corpus": corpus_path,
        "pcm": tmp_path / corpus_payload["cases"][0]["file"],
    }
    target = targets[target_name]
    os.link(target, tmp_path / f"{target_name}-hardlink")

    loader = (
        load_worker_manifest if target_name in {"manifest", "artifact"} else load_corpus
    )
    selected = manifest_path if target_name in {"manifest", "artifact"} else corpus_path
    with pytest.raises((ManifestError, CorpusError)):
        loader(selected)


def test_manifest_bound_paths_must_be_lexically_absolute(tmp_path):
    manifest_path, _ = _fixture(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["python"]["path"] = "~/ambient-python"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ManifestError):
        load_worker_manifest(manifest_path)


def test_manifest_rejects_oversized_artifact_before_unbounded_loading(
    tmp_path,
    monkeypatch,
):
    manifest_path, _ = _fixture(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = Path(payload["artifacts"][0]["path"])
    with artifact.open("r+b") as handle:
        handle.truncate(MAX_FAKE_ARTIFACT_BYTES + 1)
    payload["artifacts"][0]["size_bytes"] = MAX_FAKE_ARTIFACT_BYTES + 1
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    def forbid_read_bytes(_path):
        raise AssertionError("bound artifacts must never use Path.read_bytes()")

    monkeypatch.setattr(Path, "read_bytes", forbid_read_bytes)
    with pytest.raises(ManifestError):
        load_worker_manifest(manifest_path)


def test_manifest_rejects_numeric_overflow_as_a_stable_manifest_error(tmp_path):
    manifest_path, _ = _fixture(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["limits"]["startup_timeout_sec"] = 10**400
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ManifestError):
        load_worker_manifest(manifest_path)


def test_bound_file_hash_rejects_in_place_mutation_during_streaming_read(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "large-bound-file"
    payload = b"a" * (2 * 1024 * 1024)
    target.write_bytes(payload)
    real_read = bounded_io.os.read
    calls = 0

    def racing_read(descriptor, size):
        nonlocal calls
        chunk = real_read(descriptor, size)
        calls += 1
        if calls == 1:
            target.write_bytes(b"b" * len(payload))
        return chunk

    monkeypatch.setattr(bounded_io.os, "read", racing_read)

    with pytest.raises(BoundedReadError):
        hash_regular_bounded(
            target,
            maximum_bytes=len(payload),
            expected_bytes=len(payload),
        )


def test_corpus_loads_finite_exact_f32_and_hides_text_from_repr(tmp_path):
    _, corpus_path = _fixture(tmp_path)

    corpus = load_corpus(corpus_path)

    assert corpus.audio_bytes == 16
    assert corpus.cases[0].samples == 4
    assert "stop now" not in repr(corpus)


def test_schema_v4_loader_binds_fixed_preparation_receipt_and_forbidden_targets(
    tmp_path,
):
    corpus_path = _schema_v4_corpus(tmp_path)

    corpus = load_corpus(corpus_path)

    assert corpus.schema_version == 4
    assert corpus.provenance is not None
    assert corpus.provenance.kind == "public-command-noise-v1"
    assert corpus.cases[0].assertion == "speech_negative"
    assert corpus.cases[0].expected_text == ""
    assert corpus.cases[0].commands == ()
    assert corpus.cases[0].forbidden_commands == ("turn on", "turn off")
    assert "turn on" not in repr(corpus)
    verify_corpus_snapshot(corpus)


def test_schema_v4_transcript_negative_keeps_wer_reference_without_positive_target(
    tmp_path,
):
    corpus_path = _schema_v4_corpus(tmp_path)
    payload = json.loads(corpus_path.read_text(encoding="utf-8"))
    payload["cases"][0].update(
        {
            "expected_text": "the dog is barking",
            "assertion": "transcript",
            "commands": [],
            "forbidden_commands": ["dark"],
        }
    )
    corpus_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    corpus = load_corpus(corpus_path)

    assert corpus.cases[0].expected_text == "the dog is barking"
    assert corpus.cases[0].commands == ()
    assert corpus.cases[0].forbidden_commands == ("dark",)


def test_schema_v4_receipt_tamper_fails_load_and_loaded_snapshot_verification(
    tmp_path,
):
    corpus_path = _schema_v4_corpus(tmp_path)
    corpus = load_corpus(corpus_path)
    receipt = tmp_path / "preparation-receipt.json"
    original = receipt.read_bytes()
    receipt.write_bytes(original.replace(b"command-noise", b"command-voice"))
    assert receipt.stat().st_size == len(original)

    with pytest.raises(CorpusError):
        verify_corpus_snapshot(corpus)
    with pytest.raises(CorpusError):
        load_corpus(corpus_path)


@pytest.mark.parametrize(
    "mutation",
    (
        "missing-forbidden-field",
        "normalized-duplicate",
        "positive-on-negative",
        "empty-negative-forbidden",
        "missing-positive-reference",
        "forbidden-present-in-reference",
        "legacy-extra-field",
    ),
)
def test_corpus_schema_versions_reject_cross_schema_or_ambiguous_command_rows(
    tmp_path,
    mutation,
):
    corpus_path = _schema_v4_corpus(tmp_path)
    payload = json.loads(corpus_path.read_text(encoding="utf-8"))
    row = payload["cases"][0]
    if mutation == "missing-forbidden-field":
        row.pop("forbidden_commands")
    elif mutation == "normalized-duplicate":
        row["forbidden_commands"] = ["Turn on!", "turn-on"]
    elif mutation == "positive-on-negative":
        row["commands"] = ["turn on"]
    elif mutation == "empty-negative-forbidden":
        row["forbidden_commands"] = []
    elif mutation == "missing-positive-reference":
        row.update(
            {
                "expected_text": "turn on the light",
                "assertion": "transcript",
                "commands": ["turn off"],
                "forbidden_commands": [],
            }
        )
    elif mutation == "forbidden-present-in-reference":
        row.update(
            {
                "expected_text": "turn on the light",
                "assertion": "transcript",
                "commands": [],
                "forbidden_commands": ["turn on"],
            }
        )
    else:
        payload["schema_version"] = 3
        payload["provenance"]["kind"] = "private-diagnostic-v1"
    corpus_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(CorpusError):
        load_corpus(corpus_path)


def test_schema_v4_loader_reference_binding_uses_token_not_substring_matches(
    tmp_path,
):
    corpus_path = _schema_v4_corpus(tmp_path)
    payload = json.loads(corpus_path.read_text(encoding="utf-8"))
    payload["cases"][0].update(
        {
            "expected_text": "only proceed now",
            "assertion": "transcript",
            "commands": ["PROCEED!"],
            "forbidden_commands": ["on", "off"],
        }
    )
    corpus_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    corpus = load_corpus(corpus_path)

    assert corpus.cases[0].commands == ("PROCEED!",)
    assert corpus.cases[0].forbidden_commands == ("on", "off")


@pytest.mark.parametrize("mutation", ["hash", "unknown-field", "duplicate-key"])
def test_corpus_rejects_changed_or_non_strict_metadata(tmp_path, mutation):
    _, corpus_path = _fixture(tmp_path)
    payload = json.loads(corpus_path.read_text(encoding="utf-8"))
    if mutation == "hash":
        payload["cases"][0]["sha256"] = "0" * 64
        corpus_path.write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "unknown-field":
        payload["cases"][0]["extra"] = True
        corpus_path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        corpus_path.write_text(
            '{"schema_version":1,"schema_version":1}',
            encoding="utf-8",
        )

    with pytest.raises(CorpusError):
        load_corpus(corpus_path)


def test_corpus_rejects_nonfinite_pcm(tmp_path):
    _, corpus_path = _fixture(tmp_path)
    payload = json.loads(corpus_path.read_text(encoding="utf-8"))
    audio = tmp_path / payload["cases"][0]["file"]
    raw = bytearray(audio.read_bytes())
    raw[:4] = bytes.fromhex("0000c07f")
    audio.write_bytes(raw)
    payload["cases"][0]["sha256"] = hashlib.sha256(raw).hexdigest()
    corpus_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CorpusError):
        load_corpus(corpus_path)


@pytest.mark.parametrize(
    ("expected_text", "commands"),
    [
        ("!!!", []),
        ("stop now", ["!!!"]),
    ],
)
def test_corpus_rejects_normalized_empty_references_and_commands(
    tmp_path,
    expected_text,
    commands,
):
    _, corpus_path = _fixture(tmp_path)
    payload = json.loads(corpus_path.read_text(encoding="utf-8"))
    payload["cases"][0]["expected_text"] = expected_text
    payload["cases"][0]["commands"] = commands
    corpus_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CorpusError):
        load_corpus(corpus_path)


@pytest.mark.parametrize(
    ("target_name", "expected_error"),
    [
        ("worker-manifest", ManifestError),
        ("corpus-manifest", CorpusError),
        ("pcm", CorpusError),
    ],
)
def test_sparse_oversized_inputs_are_rejected_without_unbounded_read(
    tmp_path,
    monkeypatch,
    target_name,
    expected_error,
):
    manifest_path, corpus_path = _fixture(tmp_path)
    payload = json.loads(corpus_path.read_text(encoding="utf-8"))
    targets = {
        "worker-manifest": manifest_path,
        "corpus-manifest": corpus_path,
        "pcm": tmp_path / payload["cases"][0]["file"],
    }
    with targets[target_name].open("r+b") as handle:
        handle.truncate(1 << 30)

    def forbid_read_bytes(_path):
        raise AssertionError("unbounded Path.read_bytes() must not be used")

    monkeypatch.setattr(Path, "read_bytes", forbid_read_bytes)
    loader = load_worker_manifest if target_name == "worker-manifest" else load_corpus
    selected = manifest_path if target_name == "worker-manifest" else corpus_path
    with pytest.raises(expected_error):
        loader(selected)


@pytest.mark.parametrize("target_name", ["corpus-manifest", "pcm"])
def test_snapshot_verification_rejects_post_load_growth_without_unbounded_read(
    tmp_path,
    monkeypatch,
    target_name,
):
    _, corpus_path = _fixture(tmp_path)
    corpus = load_corpus(corpus_path)
    target = (
        corpus.path if target_name == "corpus-manifest" else corpus.cases[0].source_path
    )
    with target.open("r+b") as handle:
        handle.truncate(1 << 30)

    def forbid_read_bytes(_path):
        raise AssertionError("snapshot verification must remain byte bounded")

    monkeypatch.setattr(Path, "read_bytes", forbid_read_bytes)
    with pytest.raises(CorpusError):
        verify_corpus_snapshot(corpus)


def test_corpus_total_budget_is_checked_before_reading_next_case(
    tmp_path,
    monkeypatch,
):
    payload = bytes(8)
    digest = hashlib.sha256(payload).hexdigest()
    rows = []
    for index in range(3):
        audio = tmp_path / f"bounded-{index}.f32le"
        audio.write_bytes(payload)
        rows.append(
            {
                "id": f"bounded-{index}",
                "file": audio.name,
                "sha256": digest,
                "samples": 2,
                "expected_text": "x",
                "assertion": "transcript",
                "commands": [],
                "tags": [],
            }
        )
    corpus_path = tmp_path / "bounded-corpus.json"
    corpus_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "purpose": "total budget regression",
                "cases": rows,
            }
        ),
        encoding="utf-8",
    )
    original_read = corpus_module.read_regular_bounded

    def guarded_read(path, **kwargs):
        if Path(path).name == "bounded-2.f32le":
            raise AssertionError("over-budget case must be rejected before read")
        return original_read(path, **kwargs)

    monkeypatch.setattr(corpus_module, "MAX_PCM_BYTES", 8)
    monkeypatch.setattr(corpus_module, "MAX_CORPUS_BYTES", 16)
    monkeypatch.setattr(corpus_module, "read_regular_bounded", guarded_read)

    with pytest.raises(CorpusError):
        load_corpus(corpus_path)


def _parakeet_cpp_manifest_fixture(tmp_path, monkeypatch):
    from tests.test_provision_parakeet_cpp_candidate import _inputs
    from tools import provision_parakeet_cpp_candidate as provision

    paths = _inputs(tmp_path, monkeypatch)
    manifest = provision.provision_candidate(
        python=paths["python"],
        native_root=paths["native_root"],
        model_root=paths["model_root"],
        output_dir=paths["output_dir"],
    )
    return manifest.path, paths


def test_manifest_v8_binds_exact_parakeet_cpp_cpu_cell(tmp_path, monkeypatch):
    from tools.streaming_stt.manifest import (
        PARAKEET_CPP_ADAPTER,
        PARAKEET_CPP_ARTIFACT_NAMES,
        ParakeetCppConfig,
    )

    path, _paths = _parakeet_cpp_manifest_fixture(tmp_path, monkeypatch)

    manifest = load_worker_manifest(path)

    assert manifest.schema_version == 8
    assert manifest.adapter == PARAKEET_CPP_ADAPTER
    assert isinstance(manifest.adapter_config, ParakeetCppConfig)
    assert manifest.adapter_config.requested_device == "cpu"
    assert manifest.adapter_config.actual_device == "cpu"
    assert manifest.adapter_config.ggml_native is False
    assert manifest.adapter_config.num_threads == 1
    assert tuple(manifest.artifact_by_name) == PARAKEET_CPP_ARTIFACT_NAMES


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_threads", True),
        ("ggml_cuda", 0),
        ("frame_sec", 0.0800001),
        ("model_dtype", "Q8_0"),
    ],
)
def test_manifest_v8_rejects_nonexact_config_value_or_type(
    tmp_path,
    monkeypatch,
    field,
    value,
):
    path, _paths = _parakeet_cpp_manifest_fixture(tmp_path, monkeypatch)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["adapter_config"][field] = value
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ManifestError):
        load_worker_manifest(path)


def test_manifest_v8_rejects_boolean_patch_order_in_rebound_source_receipt(
    tmp_path,
    monkeypatch,
):
    path, paths = _parakeet_cpp_manifest_fixture(tmp_path, monkeypatch)
    receipt = paths["source_receipt"]
    receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
    receipt_payload["ordered_patches"][0]["order"] = True
    receipt.write_text(
        json.dumps(receipt_payload, sort_keys=True),
        encoding="utf-8",
    )
    manifest_payload = json.loads(path.read_text(encoding="utf-8"))
    source_artifact = next(
        artifact
        for artifact in manifest_payload["artifacts"]
        if artifact["name"] == "source-receipt"
    )
    source_artifact["sha256"] = hashlib.sha256(receipt.read_bytes()).hexdigest()
    source_artifact["size_bytes"] = receipt.stat().st_size
    path.write_text(json.dumps(manifest_payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ManifestError):
        load_worker_manifest(path)


def test_manifest_v8_rejects_changed_native_dependency_closure(
    tmp_path,
    monkeypatch,
):
    path, _paths = _parakeet_cpp_manifest_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        manifest_module,
        "PARAKEET_CPP_BRIDGE_NEEDED",
        ("libparakeet.so", "ld-linux-x86-64.so.2"),
    )

    with pytest.raises(ManifestError):
        load_worker_manifest(path)
