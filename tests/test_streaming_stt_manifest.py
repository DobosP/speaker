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
    ManifestError,
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


def test_manifest_v2_accepts_exact_pinned_small_streaming_model(
    tmp_path,
    monkeypatch,
):
    path = _moonshine_manifest(
        tmp_path,
        monkeypatch,
        model_arch="small-streaming",
    )

    manifest = load_worker_manifest(path)

    assert manifest.adapter_config is not None
    assert manifest.adapter_config.model_arch == "small-streaming"


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
