from __future__ import annotations

from copy import deepcopy
import csv
from dataclasses import dataclass, field, replace
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import struct
import subprocess
import sys
from typing import Callable, Mapping

import numpy as np
import pytest

import tools.prepare_microsoft_aec_fixture as subject


_FIXTURE_ID = "synthetic-microsoft-aec-test-v1"
_SELF_CANONICALIZATION = (
    "UTF-8 JSON with sort_keys=true, separators=(',', ':'), "
    "ensure_ascii=false, allow_nan=false"
)


@dataclass(frozen=True)
class _SyntheticSource:
    root: Path
    injection: subject.TestFixtureInjection
    metadata_path: Path = field(repr=False)
    artifacts: Mapping[tuple[int, str], Path] = field(repr=False)
    private_markers: tuple[str, ...] = field(repr=False)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _git_blob_sha1(raw: bytes) -> str:
    header = b"blob " + str(len(raw)).encode("ascii") + b"\0"
    return hashlib.sha1(header + raw).hexdigest()


def _lfs_pointer_blob_sha1(raw: bytes) -> str:
    pointer = (
        b"version https://git-lfs.github.com/spec/v1\n"
        + b"oid sha256:"
        + hashlib.sha256(raw).hexdigest().encode("ascii")
        + b"\nsize "
        + str(len(raw)).encode("ascii")
        + b"\n"
    )
    return _git_blob_sha1(pointer)


def _seal_lock(lock: dict[str, object]) -> dict[str, object]:
    self_digest = lock["self_digest"]
    assert isinstance(self_digest, dict)
    self_digest.pop("value", None)
    self_digest["value"] = hashlib.sha256(_canonical_json(lock)).hexdigest()
    return lock


def _selected_rows_sha256(fileids: tuple[str, ...]) -> str:
    identities = [
        json.dumps([fileid], ensure_ascii=False, separators=(",", ":"))
        for fileid in fileids
    ]
    encoded = json.dumps(
        identities,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _wav(
    *,
    samples: int,
    value: int = 1000,
    rate: int = subject.SAMPLE_RATE_HZ,
    channels: int = 1,
    bits: int = 16,
    riff: bytes = b"RIFF",
) -> bytes:
    if bits == 16:
        data = struct.pack("<h", value) * samples * channels
    else:
        data = bytes([value & 0xFF]) * samples * channels
    block_align = channels * bits // 8
    byte_rate = rate * block_align
    return (
        riff
        + struct.pack("<I", 36 + len(data))
        + b"WAVEfmt "
        + struct.pack(
            "<IHHIIHH",
            16,
            1,
            channels,
            rate,
            byte_rate,
            block_align,
            bits,
        )
        + b"data"
        + struct.pack("<I", len(data))
        + data
    )


def _lfs_pointer(*, size: int = 3_244) -> bytes:
    return (
        b"version https://git-lfs.github.com/spec/v1\n"
        + b"oid sha256:"
        + (b"1" * 64)
        + b"\nsize "
        + str(size).encode("ascii")
        + b"\n"
    )


def _metadata_rows(fileids: tuple[str, ...]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index, fileid in enumerate(fileids):
        rows.append(
            {
                "nearend_speaker": f"private-near-speaker-{fileid}",
                "nearend_wav_path": f"private-near-clean-{fileid}.wav",
                "nearend_wav_path_noisy": f"private-near-noisy-{fileid}.wav",
                "farend_speaker": f"private-far-speaker-{fileid}",
                "farend_wav_path": f"private-far-clean-{fileid}.wav",
                "farend_wav_path_noisy": f"private-far-noisy-{fileid}.wav",
                "ser": str(index - 1),
                "is_farend_nonlinear": str(index % 2),
                "is_farend_noisy": str((index + 1) % 2),
                "is_nearend_noisy": str(index % 2),
                "split": "train" if index == 0 else "test",
                "fileid": fileid,
                "nearend_scale": "0.75" if index == 0 else "1.25",
            }
        )
    return rows


def _metadata_csv(rows: list[dict[str, str]], *, header: tuple[str, ...]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow([row[column] for column in subject.META_COLUMNS])
    return stream.getvalue().encode("utf-8")


def _make_source(
    tmp_path: Path,
    *,
    payload_overrides: Mapping[tuple[int, str], bytes] | None = None,
    sample_overrides: Mapping[tuple[int, str], int] | None = None,
    source_header: tuple[str, ...] | None = None,
    source_row_overrides: Mapping[int, Mapping[str, str]] | None = None,
) -> _SyntheticSource:
    tmp_path.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp_path.chmod(0o700)
    root = tmp_path / "private-source-location-never-publish"
    root.mkdir(mode=0o700)
    (root / ".git").mkdir(mode=0o700)
    head = root / ".git" / "HEAD"
    head.write_bytes((subject.REVISION + "\n").encode("ascii"))

    fileids = ("101", "202")
    lock_rows = _metadata_rows(fileids)
    source_rows = deepcopy(lock_rows)
    for index, replacements in (source_row_overrides or {}).items():
        source_rows[index].update(replacements)
    meta_raw = _metadata_csv(
        source_rows,
        header=source_header or tuple(subject.META_COLUMNS),
    )
    metadata_path = root / subject.META_RELATIVE_PATH
    metadata_path.parent.mkdir(mode=0o700, parents=True)
    metadata_path.write_bytes(meta_raw)

    payload_overrides = payload_overrides or {}
    sample_overrides = sample_overrides or {}
    artifacts: dict[tuple[int, str], Path] = {}
    lock_cases: list[dict[str, object]] = []
    audio_bytes = 0
    for case_index, (fileid, row) in enumerate(zip(fileids, lock_rows, strict=True)):
        lock_artifacts: list[dict[str, object]] = []
        for role_index, role in enumerate(subject.SIGNAL_ROLES):
            relative = subject._expected_artifact_path(role, fileid)
            samples = sample_overrides.get(
                (case_index, role), 1_600 if case_index == 0 else 1_599
            )
            raw = payload_overrides.get(
                (case_index, role),
                _wav(
                    samples=samples,
                    value=100 * (case_index + 1) + role_index + 1,
                ),
            )
            path = root.joinpath(*relative.split("/"))
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            path.write_bytes(raw)
            artifacts[(case_index, role)] = path
            digest = hashlib.sha256(raw).hexdigest()
            lock_artifacts.append(
                {
                    "git_pointer_blob_sha1": _lfs_pointer_blob_sha1(raw),
                    "lfs_oid_sha256": digest,
                    "relative_path": relative,
                    "role": role,
                    "size_bytes": len(raw),
                }
            )
            audio_bytes += len(raw)
        lock_cases.append(
            {
                "artifacts": lock_artifacts,
                "fileid": fileid,
                "metadata": row,
                "ordinal": case_index + 1,
            }
        )

    selected_digest = _selected_rows_sha256(fileids)
    lock: dict[str, object] = {
        "cases": lock_cases,
        "fixture_id": _FIXTURE_ID,
        "layout": {
            "artifact_count": len(fileids) * len(subject.SIGNAL_ROLES),
            "audio_bytes": audio_bytes,
            "case_count": len(fileids),
            "metadata_bytes": len(meta_raw),
            "signal_roles": list(subject.SIGNAL_ROLES),
            "total_pinned_bytes": audio_bytes + len(meta_raw),
        },
        "license": {
            "acceptance_ids": list(subject.LICENSE_ACCEPTANCE_IDS),
            "gate_id": "synthetic-test-license-gate",
            "redistribution": "cache-only",
        },
        "metadata": {
            "columns": list(subject.META_COLUMNS),
            "git_blob_sha1": _git_blob_sha1(meta_raw),
            "relative_path": subject.META_RELATIVE_PATH,
            "row_count": len(fileids),
            "sha256": hashlib.sha256(meta_raw).hexdigest(),
            "size_bytes": len(meta_raw),
        },
        "metric_evidence": {"scope": "synthetic-test-only"},
        "schema_version": subject.SCHEMA_VERSION,
        "selection": {
            "algorithm": "test-fixed-v1",
            "description": "Two fixed generated cases for unit tests.",
            "expected_examples": len(fileids),
            "fixed_ids": list(fileids),
            "identity_fields": ["fileid"],
            "policy_sha256": hashlib.sha256(b"synthetic-policy").hexdigest(),
            "seed": "speaker-public-eval-v1-2026-08-01",
            "selected_rows_sha256": selected_digest,
            "selected_upstream_split_counts": {"synthetic": len(fileids)},
            "source_row_domain": "test_injection",
            "speaker_disjoint": False,
            "speaker_field": None,
            "split": "synthetic",
            "strata_fields": [],
        },
        "self_digest": {
            "algorithm": "sha256",
            "canonicalization": _SELF_CANONICALIZATION,
            "excluded_json_pointer": "/self_digest/value",
        },
        "source": {
            "commit": subject.REVISION,
            "repository_url": "https://github.com/microsoft/AEC-Challenge",
            "root_tree_sha1": hashlib.sha1(b"synthetic-root-tree").hexdigest(),
        },
        "usage_constraints": ["evaluation_only", "private_cache_only"],
    }
    _seal_lock(lock)

    for entry in root.rglob("*"):
        entry.chmod(0o700 if entry.is_dir() else 0o600)
    return _SyntheticSource(
        root=root,
        injection=subject.TestFixtureInjection(_FIXTURE_ID, lock),
        metadata_path=metadata_path,
        artifacts=artifacts,
        private_markers=(
            str(root),
            "private-source-location-never-publish",
            "private-near-speaker",
            "private-far-speaker",
            "private-near-clean",
            "private-far-clean",
        ),
    )


def _prepare(
    tmp_path: Path,
    source: _SyntheticSource,
    *,
    output_name: str = "private-output",
) -> subject.PreparedMicrosoftAecFixture:
    return subject.prepare_microsoft_aec_fixture(
        source.root,
        tmp_path / output_name,
        accepted_terms=subject.LICENSE_ACCEPTANCE_IDS,
        _test_source=source.injection,
    )


def _assert_prepare_rejected(
    tmp_path: Path,
    source: _SyntheticSource,
    *,
    output_name: str = "rejected-output",
) -> None:
    with pytest.raises(subject.MicrosoftAecFixtureError, match="^$"):
        _prepare(tmp_path, source, output_name=output_name)


def test_prepare_load_verify_are_opaque_private_and_receipt_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _make_source(tmp_path)
    output = tmp_path / "private-output"
    observed: dict[str, object] = {}

    def inspect_before_commit(callback: Callable[[], None]) -> None:
        entries = {entry.name for entry in output.iterdir()}
        assert subject.PREPARATION_RECEIPT_FILENAME not in entries
        assert entries == {
            subject.MANIFEST_FILENAME,
            *(
                f"case-{index:02d}-{suffix}.wav"
                for index in range(2)
                for suffix in ("echo", "far", "mic", "near")
            ),
        }
        observed["receipt_was_absent"] = True
        callback()

    monkeypatch.setattr(subject, "_terminal_commit_guard", inspect_before_commit)
    prepared = _prepare(tmp_path, source)
    loaded = subject.load_microsoft_aec_bundle(prepared.path)
    subject.verify_microsoft_aec_bundle(loaded)

    assert observed == {"receipt_was_absent": True}
    assert prepared.fixture_id == loaded.fixture_id == _FIXTURE_ID
    assert prepared.production_evidence is loaded.production_evidence is False
    assert prepared.artifact_count == 8
    assert [case.artifacts[0].samples for case in loaded.cases] == [1_600, 1_599]
    manifest = json.loads((prepared.path / subject.MANIFEST_FILENAME).read_bytes())
    assert [case["terminal_padding_samples"] for case in manifest["cases"]] == [0, 1]
    assert stat.S_IMODE(prepared.path.stat().st_mode) == 0o700
    expected_entries = {
        subject.MANIFEST_FILENAME,
        subject.PREPARATION_RECEIPT_FILENAME,
        *(
            f"case-{index:02d}-{suffix}.wav"
            for index in range(2)
            for suffix in ("echo", "far", "mic", "near")
        ),
    }
    assert {entry.name for entry in prepared.path.iterdir()} == expected_entries
    for entry in prepared.path.iterdir():
        metadata = entry.stat()
        assert stat.S_IMODE(metadata.st_mode) == 0o600
        assert metadata.st_nlink == 1

    decoded = subject.decode_microsoft_aec_wav(
        (prepared.path / "case-00-echo.wav").read_bytes()
    )
    assert decoded.dtype == np.float32
    assert decoded.flags.owndata
    assert decoded.shape == (1_600,)
    assert np.all(np.isfinite(decoded))

    published_text = b"".join(
        (prepared.path / name).read_bytes()
        for name in (subject.MANIFEST_FILENAME, subject.PREPARATION_RECEIPT_FILENAME)
    ).decode("ascii")
    for marker in source.private_markers:
        assert marker not in published_text
        assert marker not in repr(prepared)
        assert marker not in repr(loaded)
    assert all("101" not in name and "202" not in name for name in expected_entries)


def test_missing_term_rejects_before_source_or_output_access(tmp_path: Path) -> None:
    source = _make_source(tmp_path)
    output = tmp_path / "private-output"
    with pytest.raises(subject.MicrosoftAecFixtureError, match="^$"):
        subject.prepare_microsoft_aec_fixture(
            source.root,
            output,
            accepted_terms=subject.LICENSE_ACCEPTANCE_IDS[:-1],
            _test_source=source.injection,
        )
    assert not output.exists()


@pytest.mark.parametrize("mutation", ["head", "root-mode", "file-mode"])
def test_rejects_wrong_head_and_nonprivate_source_modes(
    tmp_path: Path, mutation: str
) -> None:
    source = _make_source(tmp_path)
    if mutation == "head":
        (source.root / ".git" / "HEAD").write_text(
            "ref: refs/heads/main\n", encoding="ascii"
        )
    elif mutation == "root-mode":
        source.root.chmod(0o755)
    else:
        source.artifacts[(0, subject.SIGNAL_ROLES[0])].chmod(0o644)
    _assert_prepare_rejected(tmp_path, source)


@pytest.mark.parametrize("tamper", ["metadata", "audio"])
def test_rejects_bound_source_hash_tamper(tmp_path: Path, tamper: str) -> None:
    source = _make_source(tmp_path)
    target = (
        source.metadata_path
        if tamper == "metadata"
        else source.artifacts[(0, subject.SIGNAL_ROLES[0])]
    )
    raw = bytearray(target.read_bytes())
    raw[-2] ^= 1
    target.write_bytes(raw)
    target.chmod(0o600)
    _assert_prepare_rejected(tmp_path, source)


@pytest.mark.parametrize("kind", ["semantic-row", "header"])
def test_rejects_pinned_metadata_semantics_and_wrong_header(
    tmp_path: Path, kind: str
) -> None:
    if kind == "semantic-row":
        source = _make_source(
            tmp_path,
            source_row_overrides={0: {"nearend_scale": "0.76"}},
        )
    else:
        header = list(subject.META_COLUMNS)
        header[0] = "wrong_nearend_speaker"
        source = _make_source(tmp_path, source_header=tuple(header))
    _assert_prepare_rejected(tmp_path, source)


@pytest.mark.parametrize(
    ("name", "payload"),
    [
        ("lfs-pointer", _lfs_pointer()),
        ("riff", _wav(samples=1_600, riff=b"RIFX")),
        ("rate", _wav(samples=1_600, rate=8_000)),
        ("channels", _wav(samples=1_600, channels=2)),
        ("width", _wav(samples=1_600, bits=8)),
    ],
)
def test_rejects_lfs_pointer_and_noncanonical_wav(
    tmp_path: Path, name: str, payload: bytes
) -> None:
    role = subject.SIGNAL_ROLES[0]
    source = _make_source(tmp_path, payload_overrides={(0, role): payload})
    _assert_prepare_rejected(tmp_path, source, output_name=f"rejected-{name}")


def test_rejects_mismatched_synchronized_role_lengths(tmp_path: Path) -> None:
    source = _make_source(
        tmp_path,
        sample_overrides={(0, subject.SIGNAL_ROLES[1]): 1_599},
    )
    _assert_prepare_rejected(tmp_path, source)


@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_rejects_linked_source_artifacts(tmp_path: Path, kind: str) -> None:
    source = _make_source(tmp_path)
    target = source.artifacts[(0, subject.SIGNAL_ROLES[0])]
    if kind == "symlink":
        other = source.artifacts[(0, subject.SIGNAL_ROLES[1])]
        target.unlink()
        target.symlink_to(other)
    else:
        os.link(target, tmp_path / "second-hardlink")
    _assert_prepare_rejected(tmp_path, source)


@pytest.mark.parametrize("kind", ["directory", "file", "symlink"])
def test_existing_output_is_never_clobbered(tmp_path: Path, kind: str) -> None:
    source = _make_source(tmp_path)
    output = tmp_path / "private-output"
    if kind == "directory":
        output.mkdir(mode=0o700)
        marker = output / "keep"
        marker.write_bytes(b"keep")
        marker.chmod(0o600)
    elif kind == "file":
        output.write_bytes(b"keep")
        output.chmod(0o600)
    else:
        target = tmp_path / "symlink-target"
        target.mkdir(mode=0o700)
        output.symlink_to(target, target_is_directory=True)
    with pytest.raises(subject.MicrosoftAecFixtureError, match="^$"):
        _prepare(tmp_path, source)
    if kind == "directory":
        assert (output / "keep").read_bytes() == b"keep"
    elif kind == "file":
        assert output.read_bytes() == b"keep"
    else:
        assert output.is_symlink()


def test_source_mutation_before_terminal_commit_leaves_no_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _make_source(tmp_path)
    target = source.artifacts[(0, subject.SIGNAL_ROLES[0])]
    output = tmp_path / "private-output"

    def mutate_then_close(callback: Callable[[], None]) -> None:
        raw = bytearray(target.read_bytes())
        raw[-2] ^= 1
        target.write_bytes(raw)
        target.chmod(0o600)
        callback()

    monkeypatch.setattr(subject, "_terminal_commit_guard", mutate_then_close)
    with pytest.raises(subject.MicrosoftAecFixtureError, match="^$"):
        _prepare(tmp_path, source)
    assert output.is_dir()
    assert (output / subject.MANIFEST_FILENAME).is_file()
    assert not (output / subject.PREPARATION_RECEIPT_FILENAME).exists()


def test_lock_drift_before_terminal_commit_leaves_no_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _make_source(tmp_path)
    output = tmp_path / "private-output"
    original_load_lock = subject._load_lock

    def drift_then_close(callback: Callable[[], None]) -> None:
        def drifted_lock(*args: object, **kwargs: object):
            loaded = original_load_lock(*args, **kwargs)
            return replace(loaded, raw_sha256="f" * 64)

        monkeypatch.setattr(subject, "_load_lock", drifted_lock)
        callback()

    monkeypatch.setattr(subject, "_terminal_commit_guard", drift_then_close)
    with pytest.raises(subject.MicrosoftAecFixtureError, match="^$"):
        _prepare(tmp_path, source)
    assert output.is_dir()
    assert not (output / subject.PREPARATION_RECEIPT_FILENAME).exists()


@pytest.mark.parametrize(
    "tamper",
    ["file", "manifest", "receipt", "mode", "symlink", "hardlink", "extra"],
)
def test_loader_and_loaded_identity_reject_bundle_tamper(
    tmp_path: Path, tamper: str
) -> None:
    source = _make_source(tmp_path)
    prepared = _prepare(tmp_path, source)
    loaded = subject.load_microsoft_aec_bundle(prepared.path)
    artifact = prepared.path / "case-00-echo.wav"
    if tamper == "file":
        raw = bytearray(artifact.read_bytes())
        raw[-2] ^= 1
        artifact.write_bytes(raw)
    elif tamper == "manifest":
        manifest = prepared.path / subject.MANIFEST_FILENAME
        manifest.write_bytes(manifest.read_bytes() + b" ")
    elif tamper == "receipt":
        receipt = prepared.path / subject.PREPARATION_RECEIPT_FILENAME
        receipt.write_bytes(receipt.read_bytes() + b" ")
    elif tamper == "mode":
        artifact.chmod(0o640)
    elif tamper == "symlink":
        artifact.unlink()
        artifact.symlink_to(prepared.path / "case-00-far.wav")
    elif tamper == "hardlink":
        os.link(artifact, tmp_path / "bundle-artifact-hardlink")
    else:
        extra = prepared.path / "extra"
        extra.write_bytes(b"extra")
        extra.chmod(0o600)
    with pytest.raises(subject.MicrosoftAecFixtureError, match="^$"):
        subject.load_microsoft_aec_bundle(prepared.path)
    with pytest.raises(subject.MicrosoftAecFixtureError, match="^$"):
        subject.verify_microsoft_aec_bundle(loaded)


def test_loader_rejects_preparer_closure_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _make_source(tmp_path)
    prepared = _prepare(tmp_path, source)
    loaded = subject.load_microsoft_aec_bundle(prepared.path)
    original = subject._preparer_closure()
    drifted = (
        *original,
        {"path": "synthetic-drift", "sha256": "0" * 64, "size_bytes": 1},
    )
    monkeypatch.setattr(subject, "_preparer_closure", lambda: drifted)
    with pytest.raises(subject.MicrosoftAecFixtureError, match="^$"):
        subject.load_microsoft_aec_bundle(prepared.path)
    with pytest.raises(subject.MicrosoftAecFixtureError, match="^$"):
        subject.verify_microsoft_aec_bundle(loaded)


def test_loader_rechecks_preparer_closure_after_artifact_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _make_source(tmp_path)
    prepared = _prepare(tmp_path, source)
    original = subject._preparer_closure()
    drifted = (
        *original,
        {"path": "synthetic-drift", "sha256": "0" * 64, "size_bytes": 1},
    )
    calls = 0

    def closure_then_drift():
        nonlocal calls
        calls += 1
        return original if calls == 1 else drifted

    monkeypatch.setattr(subject, "_preparer_closure", closure_then_drift)
    with pytest.raises(subject.MicrosoftAecFixtureError, match="^$"):
        subject.load_microsoft_aec_bundle(prepared.path)
    assert calls >= 2


def test_test_injection_cannot_claim_the_production_fixture_id(tmp_path: Path) -> None:
    source = _make_source(tmp_path)
    lock = deepcopy(source.injection.lock_value)
    assert isinstance(lock, dict)
    lock["fixture_id"] = subject.FIXTURE_ID
    _seal_lock(lock)
    injection = subject.TestFixtureInjection(subject.FIXTURE_ID, lock)
    with pytest.raises(subject.MicrosoftAecFixtureError, match="^$"):
        subject.prepare_microsoft_aec_fixture(
            source.root,
            tmp_path / "private-output",
            accepted_terms=subject.LICENSE_ACCEPTANCE_IDS,
            _test_source=injection,
        )


def test_cli_failure_is_detail_free_and_does_not_disclose_paths(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    secret_source = tmp_path / "private-source-cli-secret"
    output = tmp_path / "private-output-cli-secret"
    argv = ["--source-dir", str(secret_source), "--output-dir", str(output)]
    for term in subject.LICENSE_ACCEPTANCE_IDS:
        argv.extend(("--accept-term", term))
    assert subject.main(argv) == 2
    captured = capfd.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "error": "microsoft_aec_fixture_prerequisites_unavailable",
        "ok": False,
    }
    assert captured.out == (
        '{"error":"microsoft_aec_fixture_prerequisites_unavailable","ok":false}\n'
    )
    assert str(secret_source) not in captured.out
    assert str(output) not in captured.out


def test_clean_process_eager_local_imports_equal_the_receipt_closure() -> None:
    expected = (
        "tools/__init__.py",
        "tools/prepare_microsoft_aec_fixture.py",
        "tools/streaming_stt/__init__.py",
        "tools/streaming_stt/bounded_io.py",
    )
    assert subject._PREPARER_FILES == expected
    assert "tools/public_voice_eval_matrix.py" not in expected
    script = """
import json
from pathlib import Path
import sys

root = Path.cwd().resolve()
import tools.prepare_microsoft_aec_fixture as subject

observed = set()
for module in tuple(sys.modules.values()):
    raw_path = getattr(module, "__file__", None)
    if not isinstance(raw_path, str):
        continue
    try:
        relative = Path(raw_path).resolve(strict=True).relative_to(root).as_posix()
    except (OSError, RuntimeError, ValueError):
        continue
    observed.add(relative)
print(json.dumps(sorted(observed), separators=(",", ":")))
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert tuple(json.loads(completed.stdout)) == expected


def test_same_size_staged_receipt_tamper_is_precommit_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _make_source(tmp_path)
    output = tmp_path / "private-output"
    original_stage = subject._stage_terminal_receipt
    staged: dict[str, int] = {}

    def capture_stage(directory_fd: int, raw: bytes):
        result = original_stage(directory_fd, raw)
        staged["fd"] = result[0]
        return result

    def tamper_then_close(callback: Callable[[], None]) -> None:
        descriptor = staged["fd"]
        os.lseek(descriptor, 0, os.SEEK_SET)
        original = os.read(descriptor, 1)
        replacement = b"[" if original != b"[" else b"{"
        os.lseek(descriptor, 0, os.SEEK_SET)
        assert os.write(descriptor, replacement) == 1
        os.fsync(descriptor)
        assert os.fstat(descriptor).st_size > 1
        callback()

    monkeypatch.setattr(subject, "_stage_terminal_receipt", capture_stage)
    monkeypatch.setattr(subject, "_terminal_commit_guard", tamper_then_close)
    with pytest.raises(subject.MicrosoftAecFixtureError, match="^$"):
        _prepare(tmp_path, source)
    assert output.is_dir()
    assert not (output / subject.PREPARATION_RECEIPT_FILENAME).exists()


def test_unexpected_output_leaf_during_close_guard_is_precommit_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _make_source(tmp_path)
    output = tmp_path / "private-output"

    def add_leaf_then_close(callback: Callable[[], None]) -> None:
        unexpected = output / "unexpected"
        unexpected.write_bytes(b"unexpected")
        unexpected.chmod(0o600)
        callback()

    monkeypatch.setattr(subject, "_terminal_commit_guard", add_leaf_then_close)
    with pytest.raises(subject.MicrosoftAecFixtureError, match="^$"):
        _prepare(tmp_path, source)
    assert (output / "unexpected").read_bytes() == b"unexpected"
    assert not (output / subject.PREPARATION_RECEIPT_FILENAME).exists()


def test_source_root_mode_drift_during_close_guard_is_precommit_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _make_source(tmp_path)
    output = tmp_path / "private-output"

    def relax_source_then_close(callback: Callable[[], None]) -> None:
        source.root.chmod(0o755)
        callback()

    monkeypatch.setattr(subject, "_terminal_commit_guard", relax_source_then_close)
    with pytest.raises(subject.MicrosoftAecFixtureError, match="^$"):
        _prepare(tmp_path, source)
    assert stat.S_IMODE(source.root.stat().st_mode) == 0o755
    assert not (output / subject.PREPARATION_RECEIPT_FILENAME).exists()


def test_real_keyboard_interrupt_before_link_propagates_without_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _make_source(tmp_path)
    output = tmp_path / "private-output"

    def interrupt_before_close(_callback: Callable[[], None]) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(subject, "_terminal_commit_guard", interrupt_before_close)
    with pytest.raises(KeyboardInterrupt):
        _prepare(tmp_path, source)
    assert output.is_dir()
    assert not (output / subject.PREPARATION_RECEIPT_FILENAME).exists()


@pytest.mark.parametrize("fault_type", [RuntimeError, KeyboardInterrupt])
def test_api_returns_published_success_when_link_raises_after_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault_type: type[BaseException],
) -> None:
    source = _make_source(tmp_path)
    output = tmp_path / "private-output"
    original_link = subject.os.link
    original_fsync = subject.os.fsync
    observed = {"link_created": False, "post_link_fsync": False}

    def link_then_raise(*args: object, **kwargs: object) -> None:
        original_link(*args, **kwargs)
        observed["link_created"] = True
        raise fault_type

    def track_fsync(descriptor: int) -> None:
        if observed["link_created"]:
            observed["post_link_fsync"] = True
        original_fsync(descriptor)

    monkeypatch.setattr(subject.os, "link", link_then_raise)
    monkeypatch.setattr(subject.os, "fsync", track_fsync)
    prepared = _prepare(tmp_path, source)
    assert prepared.path == output
    assert observed == {"link_created": True, "post_link_fsync": True}
    assert (output / subject.PREPARATION_RECEIPT_FILENAME).is_file()
    subject.verify_microsoft_aec_bundle(subject.load_microsoft_aec_bundle(output))


@pytest.mark.parametrize("fault_type", [OSError, KeyboardInterrupt])
def test_cli_returns_success_when_stdout_fails_after_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    fault_type: type[BaseException],
) -> None:
    source = _make_source(tmp_path)
    output = tmp_path / "private-output"
    real_prepare = subject.prepare_microsoft_aec_fixture
    real_write = subject.os.write

    def prepare_synthetic(
        source_dir: Path,
        output_dir: Path,
        *,
        accepted_terms: tuple[str, ...] | list[str],
        lock_path: Path,
        _commit_state: subject._CommitState,
    ) -> subject.PreparedMicrosoftAecFixture:
        assert Path(source_dir) == source.root
        assert Path(output_dir) == output
        assert tuple(accepted_terms) == subject.LICENSE_ACCEPTANCE_IDS
        assert Path(lock_path) == subject.DEFAULT_LOCK
        return real_prepare(
            source.root,
            output,
            accepted_terms=subject.LICENSE_ACCEPTANCE_IDS,
            _test_source=source.injection,
            _commit_state=_commit_state,
        )

    def fail_stdout(descriptor: int, raw: bytes) -> int:
        if descriptor == 1:
            raise fault_type
        return real_write(descriptor, raw)

    monkeypatch.setattr(subject, "prepare_microsoft_aec_fixture", prepare_synthetic)
    monkeypatch.setattr(subject.os, "write", fail_stdout)
    argv = ["--source-dir", str(source.root), "--output-dir", str(output)]
    for term in subject.LICENSE_ACCEPTANCE_IDS:
        argv.extend(("--accept-term", term))
    assert subject.main(argv) == 0
    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert (output / subject.PREPARATION_RECEIPT_FILENAME).is_file()
    subject.verify_microsoft_aec_bundle(subject.load_microsoft_aec_bundle(output))


@pytest.mark.parametrize(
    "fault",
    [KeyboardInterrupt(), subject._LifecycleSignal(subject.signal.SIGTERM)],
)
def test_cli_returns_committed_success_when_lifecycle_hits_after_api_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    fault: BaseException,
) -> None:
    source = _make_source(tmp_path)
    output = tmp_path / "private-output"
    real_prepare = subject.prepare_microsoft_aec_fixture

    def publish_then_interrupt(
        source_dir: Path,
        output_dir: Path,
        *,
        accepted_terms: tuple[str, ...] | list[str],
        lock_path: Path,
        _commit_state: subject._CommitState,
    ) -> subject.PreparedMicrosoftAecFixture:
        assert Path(source_dir) == source.root
        assert Path(output_dir) == output
        prepared = real_prepare(
            source.root,
            output,
            accepted_terms=subject.LICENSE_ACCEPTANCE_IDS,
            _test_source=source.injection,
            _commit_state=_commit_state,
        )
        assert prepared.path == output
        raise fault

    monkeypatch.setattr(
        subject,
        "prepare_microsoft_aec_fixture",
        publish_then_interrupt,
    )
    argv = ["--source-dir", str(source.root), "--output-dir", str(output)]
    for term in subject.LICENSE_ACCEPTANCE_IDS:
        argv.extend(("--accept-term", term))
    assert subject.main(argv) == 0
    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    subject.verify_microsoft_aec_bundle(subject.load_microsoft_aec_bundle(output))
