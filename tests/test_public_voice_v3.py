from __future__ import annotations

import copy
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import stat
import sys
import time

import numpy as np
import pytest
import soundfile as sf

from tools import public_stt_eval
from tools import public_voice_fixtures as public
from tools import public_voice_parquet_worker as worker


_MANIFEST = Path("tests/fixtures/public_voice_manifest.v3.json")


def _manifest_payload() -> dict[str, object]:
    payload = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_manifest(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _ulaw_wav(frames: int = 800) -> bytes:
    output = io.BytesIO()
    sf.write(
        output,
        np.zeros(frames, dtype=np.float32),
        8000,
        format="WAV",
        subtype="ULAW",
    )
    return output.getvalue()


def _private_write(path: Path, payload: bytes) -> None:
    if path.exists():
        path.chmod(0o600)
    path.write_bytes(payload)
    path.chmod(0o400)


def _worker_result(
    output_dir: Path,
    manifest: public.PublicVoiceManifest,
    *,
    wav: bytes | None = None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    audio = _ulaw_wav() if wav is None else wav
    cases: list[dict[str, object]] = []
    for fixture in manifest.fixtures:
        extract = fixture["extract"]
        assert isinstance(extract, dict)
        fixture_id = str(fixture["id"])
        intent_id = int(extract["intent_id"])
        intent_label = str(extract["intent_label"])
        transcript = f"known transcript for {intent_label}"
        source_path = f"en-US~{intent_label.upper()}/{intent_id:03d}.wav"
        filename = f"{fixture_id}.wav"
        _private_write(output_dir / filename, audio)
        cases.append(
            {
                "fixture_id": fixture_id,
                "intent_id": intent_id,
                "intent_label": intent_label,
                "language_id": 4,
                "language": "en-US",
                "row_index": intent_id,
                "source_path": source_path,
                "transcription": transcript,
                "transcription_sha256": hashlib.sha256(
                    transcript.encode("utf-8")
                ).hexdigest(),
                "audio_file": filename,
                "audio_sha256": hashlib.sha256(audio).hexdigest(),
                "audio_size_bytes": len(audio),
                "selection_rank_sha256": public._minds_selection_rank(
                    str(manifest.data["selection_seed"]),
                    public._MINDS_SOURCE_SHA256,
                    intent_label,
                    source_path,
                ),
            }
        )
    result: dict[str, object] = {
        "schema_version": 1,
        "source_sha256": public._MINDS_SOURCE_SHA256,
        "source_size_bytes": public._MINDS_SOURCE_SIZE_BYTES,
        "source_row_count": 563,
        "source_row_groups": 6,
        "pyarrow_version": "25.0.0",
        "cases": cases,
    }
    _private_write(
        output_dir / "result.json",
        (json.dumps(result, sort_keys=True) + "\n").encode("utf-8"),
    )
    return result


def _rewrite_result(output_dir: Path, result: dict[str, object]) -> None:
    _private_write(
        output_dir / "result.json",
        (json.dumps(result, sort_keys=True) + "\n").encode("utf-8"),
    )


def _isolated_interpreter(tmp_path: Path) -> Path:
    root = tmp_path / "isolated"
    binary = root / "bin" / "python"
    binary.parent.mkdir(parents=True)
    binary.symlink_to(Path(sys.executable))
    (root / "pyvenv.cfg").write_text("include-system-site-packages = false\n")
    package = (
        root
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "pyarrow"
    )
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('__version__ = "25.0.0"\n')
    return binary


def _valid_probe_payload(
    root: Path,
    **overrides: object,
) -> bytes:
    production = public._resolved_production_venv()
    payload: dict[str, object] = {
        "schema_version": public._PARQUET_PROBE_SCHEMA_VERSION,
        "prefix": str(root.resolve()),
        "base_prefix": str(Path(sys.base_prefix).resolve()),
        "enable_user_site": False,
        "sys_path": [str(Path(sys.base_prefix).resolve())],
        "pyarrow_version": public._PARQUET_PYARROW_VERSION,
        "pyarrow_file": str(
            (
                root
                / "lib"
                / f"python{sys.version_info.major}.{sys.version_info.minor}"
                / "site-packages"
                / "pyarrow"
                / "__init__.py"
            ).resolve()
        ),
    }
    assert not any(
        public._is_equal_or_within(Path(value), production)
        for value in payload["sys_path"]
    )
    payload.update(overrides)
    return json.dumps(payload, sort_keys=True).encode("utf-8")


def _valid_worker_request(source_path: Path) -> dict[str, object]:
    return {
        "schema_version": worker._PROTOCOL_VERSION,
        "source_path": str(source_path),
        "source_sha256": worker._SOURCE_SHA256,
        "source_size_bytes": worker._SOURCE_SIZE_BYTES,
        "selection_seed": worker._SELECTION_SEED,
        "source_id": worker._SOURCE_ID,
        "cases": [
            {
                "fixture_id": f"case-{intent_id}",
                "intent_id": intent_id,
                "intent_label": intent_label,
            }
            for intent_id, intent_label in enumerate(worker._INTENTS)
        ],
    }


def _prepared_v3_metadata(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    manifest = public.load_manifest(_MANIFEST)
    worker_output = tmp_path / "worker"
    _worker_result(worker_output, manifest)
    prepared = public._validate_parquet_worker_output(
        manifest,
        manifest.fixtures,
        worker_output,
        worker_sha256=public._sha256_file(public._PARQUET_WORKER),
        output_sample_rate=16000,
    )
    output = tmp_path / "prepared"
    output.mkdir()
    rows: list[dict[str, object]] = []
    for fixture in manifest.fixtures:
        fixture_id = str(fixture["id"])
        item = prepared[fixture_id]
        audio_path = output / f"{fixture_id}.npy"
        np.save(audio_path, item.samples)
        rows.append(
            {
                "name": fixture_id,
                "file": audio_path.name,
                "expectation": item.expected_text,
                "expected_text": item.expected_text,
                "assertion": "transcript",
                "tags": [],
                "sample_rate": 16000,
                "sample_count": len(item.samples),
                "duration_sec": round(len(item.samples) / 16000, 6),
                "sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest(),
                "source_sha256": {"minds14-en-us-train": public._MINDS_SOURCE_SHA256},
                "selection": dict(item.selection),
            }
        )
    metadata: dict[str, object] = {
        "schema_version": 3,
        "suite": "intent-asr",
        "purpose": manifest.data["purpose"],
        "evidence_scope": dict(public.EVIDENCE_SCOPE),
        "audio_committed": False,
        "manifest_sha256": manifest.digest,
        "selection_seed": manifest.data["selection_seed"],
        "cases": rows,
        "groups": [],
        "sources": list(public.source_records_for_suite(manifest, "intent-asr")),
    }
    metadata_path = output / "metadata.json"
    _write_manifest(metadata_path, metadata)
    return metadata_path, metadata


def test_v3_manifest_exact_provenance_and_v2_default_are_locked():
    manifest = public.load_manifest(_MANIFEST)
    source = manifest.sources[0]

    assert public.DEFAULT_MANIFEST.name == "public_voice_manifest.v2.json"
    assert public.SUITES == ("commands", "conversation", "echo")
    assert manifest.data["schema_version"] == 3
    assert manifest.data["selection_seed"] == "speaker-public-minds14-v1"
    assert len(manifest.fixtures) == 14
    assert [fixture["extract"]["intent_id"] for fixture in manifest.fixtures] == list(
        range(14)
    )
    assert source["upstream_version"] == ("40ce77cb32a384e4d50a568e1ec39ac804019d33")
    assert source["size_bytes"] == 34_196_221
    assert source["sha256"] == (
        "37004471dc896ce20771b3fdda0ee8fb33ec7a030fb4e2fd047c561ec0a1ee30"
    )
    assert source["citation"].endswith("doi:10.18653/v1/2021.emnlp-main.591.")
    assert source["license_id"] == "CC-BY-4.0"
    assert source["acquisition"] == "verified-override-only"


def test_manifest_reader_rejects_sparse_symlink_and_hardlink_inputs(tmp_path: Path):
    oversized = tmp_path / "oversized.json"
    with oversized.open("wb") as handle:
        handle.truncate(public._MAX_MANIFEST_BYTES + 1)
    target = tmp_path / "manifest.json"
    target.write_bytes(_MANIFEST.read_bytes())
    symlink = tmp_path / "manifest-link.json"
    symlink.symlink_to(target)
    hardlink = tmp_path / "manifest-hard.json"
    os.link(target, hardlink)

    for candidate in (oversized, symlink, hardlink):
        with pytest.raises(public.PublicFixtureError):
            public.load_manifest(candidate)


def test_stable_source_reader_detects_growth_retarget_and_hardlinks(tmp_path: Path):
    payload = b"pinned-source"
    source = tmp_path / "source.bin"
    source.write_bytes(payload)

    with pytest.raises(public.PublicFixtureError, match="changed while reading"):
        with public._stable_regular_reader(
            source,
            field="source file",
            maximum=len(payload) + 1,
        ) as (handle, _opened):
            assert handle.read() == payload
            source.write_bytes(payload + b"x")

    source.write_bytes(payload)
    with pytest.raises(public.PublicFixtureError, match="changed while reading"):
        with public._stable_regular_reader(
            source,
            field="source file",
            maximum=len(payload),
        ) as (handle, _opened):
            source.unlink()
            source.write_bytes(b"replacement")
            assert handle.read() == payload
    assert source.read_bytes() == b"replacement"

    source.write_bytes(payload)
    hardlink = tmp_path / "source-hard.bin"
    os.link(source, hardlink)
    record = {
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    with pytest.raises(public.PublicFixtureError, match="single-link"):
        public._verify_source_file(hardlink, record)
    destination = tmp_path / "snapshot.bin"
    with pytest.raises(public.PublicFixtureError):
        public._clone_or_copy_file(
            hardlink,
            destination,
            expected_size=len(payload),
            require_single_link=True,
        )
    assert not destination.exists()


@pytest.mark.parametrize("boundary", ["repo-worker", "worker-output", "venv-marker"])
def test_final_parent_read_boundaries_reject_same_size_preopen_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
):
    interpreter = _isolated_interpreter(tmp_path)
    if boundary == "venv-marker":
        target = interpreter.parent.parent / "pyvenv.cfg"
    else:
        target = tmp_path / f"{boundary}.py"
        target.write_bytes(b"original worker bytes\n")
        if boundary == "worker-output":
            target.chmod(0o400)

    def invoke():
        if boundary == "venv-marker":
            return public._validate_parquet_python(interpreter)
        if boundary == "worker-output":
            return public._read_stable_worker_file(target, maximum=4096)
        return public._read_stable_repo_worker(target)

    original_size = target.stat().st_size
    real_open = public.os.open
    swapped = False

    def racing_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if Path(path) == target and not swapped:
            swapped = True
            replacement = target.with_name(f".{target.name}.replacement")
            replacement.write_bytes(b"x" * original_size)
            if boundary == "worker-output":
                replacement.chmod(0o400)
            os.replace(replacement, target)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(public.os, "open", racing_open)

    with pytest.raises(public.PublicFixtureError):
        invoke()
    assert swapped is True


def test_worker_request_rejects_same_size_preopen_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    request_path = tmp_path / "request.json"
    payload = _valid_worker_request(tmp_path / "source.parquet")
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    request_path.write_bytes(encoded)
    replacement_payload = copy.deepcopy(payload)
    replacement_payload["source_id"] = "x" * len(worker._SOURCE_ID)
    replacement = json.dumps(replacement_payload, sort_keys=True).encode("utf-8")
    assert len(replacement) == len(encoded)
    real_open = worker.os.open
    swapped = False

    def racing_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if Path(path) == request_path and not swapped:
            swapped = True
            alternate = tmp_path / "replacement-request.json"
            alternate.write_bytes(replacement)
            os.replace(alternate, request_path)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(worker.os, "open", racing_open)

    with pytest.raises(worker.WorkerError):
        worker._load_request(request_path)
    assert swapped is True


def test_worker_parquet_read_stays_on_verified_fd_and_rejects_path_retarget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = tmp_path / "source.parquet"
    original = b"verified parquet bytes"
    replacement = b"replacement bytes!!!!!"
    assert len(replacement) == len(original)
    source.write_bytes(original)
    monkeypatch.setattr(worker, "_SOURCE_SIZE_BYTES", len(original))
    monkeypatch.setattr(
        worker,
        "_SOURCE_SHA256",
        hashlib.sha256(original).hexdigest(),
    )
    monkeypatch.setattr(worker, "_SOURCE_ROWS", 1)
    monkeypatch.setattr(worker, "_SOURCE_ROW_GROUPS", 1)
    monkeypatch.setattr(worker, "_validate_arrow_schema", lambda _schema: None)

    class FakeTable:
        def to_pylist(self):
            return [{}]

    class FakeParquet:
        def __init__(self, handle):
            self.metadata = type("Metadata", (), {"num_rows": 1, "num_row_groups": 1})()
            self.schema_arrow = object()
            backup = tmp_path / "verified-source-backup"
            source.replace(backup)
            source.write_bytes(replacement)
            assert handle.read() == original

        def read(self, **_kwargs):
            return FakeTable()

    class FakePq:
        @staticmethod
        def ParquetFile(handle):
            return FakeParquet(handle)

    with pytest.raises(worker.WorkerError):
        worker._read_verified_rows(FakePq, source, ("path",))


def test_production_evaluator_binds_material_public_fixture_logic():
    binding = public_stt_eval._evaluator_binding(production_report=True)
    files = binding["files"]

    assert files["tools/public_voice_fixtures.py"] == public_stt_eval._sha256_file(
        Path(public.__file__).resolve(strict=True)
    )


def test_v2_and_v3_list_outputs_are_schema_specific_and_exact(capsys):
    assert public.main(["list"]) == 0
    assert capsys.readouterr().out == (
        "commands: 12 fixtures, up to 119.2 MiB of pinned sources\n"
        "conversation: 3 fixtures, up to 60.7 MiB of pinned sources\n"
        "development/regression only; not final held-out or live evidence\n"
    )

    assert public.main(["--manifest", str(_MANIFEST), "list"]) == 0
    assert capsys.readouterr().out == (
        "intent-asr: 14 fixtures, up to 32.6 MiB of pinned sources\n"
        "development/regression only; not final held-out or live evidence\n"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "revision",
        "hash",
        "size",
        "seed",
        "acquisition",
        "row-count",
        "intent-order",
        "source-transcript",
        "product-scenario",
    ],
)
def test_v3_manifest_rejects_provenance_or_coverage_drift(
    tmp_path: Path, mutation: str
):
    payload = _manifest_payload()
    source = payload["sources"][0]
    fixtures = payload["fixtures"]
    if mutation == "revision":
        source["upstream_version"] = "main"
    elif mutation == "hash":
        source["sha256"] = "0" * 64
    elif mutation == "size":
        source["size_bytes"] += 1
    elif mutation == "seed":
        payload["selection_seed"] = "another-seed"
    elif mutation == "acquisition":
        source["acquisition"] = "cache"
    elif mutation == "row-count":
        fixtures[0]["extract"]["row_count"] = 562
    elif mutation == "intent-order":
        fixtures[0], fixtures[1] = fixtures[1], fixtures[0]
    elif mutation == "source-transcript":
        del fixtures[0]["expected_text_from_source"]
    else:
        payload["product_scenarios"] = [{}]
    path = tmp_path / "manifest.json"
    _write_manifest(path, payload)

    with pytest.raises(public.PublicFixtureError):
        public.load_manifest(path)


@pytest.mark.parametrize(
    "mutation",
    ["source-field", "source-transcript", "parquet-extractor"],
)
def test_v2_schema_does_not_accept_v3_fields(tmp_path: Path, mutation: str):
    payload = json.loads(public.DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    fixture = payload["fixtures"][0]
    if mutation == "source-field":
        payload["sources"][0]["acquisition"] = "verified-override-only"
    elif mutation == "source-transcript":
        fixture["expected_text_from_source"] = True
    else:
        fixture["extract"] = {
            "type": "hf_audio_parquet",
            "config": "en-US",
            "split": "train",
            "row_count": 563,
            "row_groups": 6,
            "language_id": 4,
            "intent_id": 0,
            "intent_label": "abroad",
        }
    path = tmp_path / "manifest.json"
    _write_manifest(path, payload)

    with pytest.raises(public.PublicFixtureError):
        public.load_manifest(path)


def test_minds_rank_matches_the_approved_nul_delimited_bytes():
    seed = "speaker-public-minds14-v1"
    source = public._MINDS_SOURCE_SHA256
    intent = "balance"
    path = "en-US/example.wav"

    assert (
        public._minds_selection_rank(seed, source, intent, path)
        == hashlib.sha256(
            seed.encode()
            + b"\0"
            + source.encode("ascii")
            + b"\0"
            + intent.encode()
            + b"\0"
            + path.encode()
        ).hexdigest()
    )
    assert public._minds_selection_rank(seed, source, intent, path) != (
        public._minds_selection_rank(seed, source, intent, path + ".other")
    )


def test_v3_source_is_override_only_even_when_download_is_accepted(
    tmp_path: Path,
):
    manifest = public.load_manifest(_MANIFEST)
    opened = False

    def opener(_request, _timeout):
        nonlocal opened
        opened = True
        raise AssertionError("network must not be called")

    with pytest.raises(public.PublicFixtureError, match="explicit hash-verified"):
        public._resolve_sources(
            manifest,
            "intent-asr",
            tmp_path,
            allow_download=True,
            offline=False,
            opener=opener,
        )
    assert opened is False


def test_parquet_interpreter_must_be_absolute_isolated_and_not_production(
    tmp_path: Path,
):
    isolated = _isolated_interpreter(tmp_path)

    assert public._validate_parquet_python(isolated) == isolated
    with pytest.raises(public.PublicFixtureError, match="absolute"):
        public._validate_parquet_python(Path("relative/python"))
    with pytest.raises(public.PublicFixtureError, match="production"):
        public._validate_parquet_python(Path(sys.executable))
    (isolated.parent.parent / "pyvenv.cfg").write_bytes(b"x" * 4097)
    with pytest.raises(public.PublicFixtureError, match="invalid isolated"):
        public._validate_parquet_python(isolated)


def test_parquet_interpreter_rejects_production_root_directory_alias(
    tmp_path: Path,
):
    alias = tmp_path / "production-alias"
    alias.symlink_to(public._resolved_production_venv(), target_is_directory=True)

    with pytest.raises(public.PublicFixtureError, match="production"):
        public._validate_parquet_python(alias / "bin" / "python")


def test_parquet_interpreter_rejects_root_inside_production_venv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    production = tmp_path / "production"
    production.mkdir()
    isolated = _isolated_interpreter(production)
    monkeypatch.setattr(
        public,
        "_resolved_production_venv",
        lambda: production.resolve(),
    )

    with pytest.raises(public.PublicFixtureError, match="production"):
        public._validate_parquet_python(isolated)


def test_parquet_interpreter_matches_effective_marker_precedence_and_ambiguity(
    tmp_path: Path,
):
    isolated = _isolated_interpreter(tmp_path)
    root_marker = isolated.parent.parent / "pyvenv.cfg"
    executable_marker = isolated.parent / "pyvenv.cfg"
    root_marker.unlink()
    executable_marker.write_text(" Include-System-Site-Packages = FALSE \n")

    assert public._validate_parquet_python(isolated) == isolated

    root_marker.write_text("include-system-site-packages = false\n")
    with pytest.raises(public.PublicFixtureError, match="invalid isolated"):
        public._validate_parquet_python(isolated)


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_parquet_interpreter_marker_must_be_regular_single_link(
    tmp_path: Path,
    link_kind: str,
):
    isolated = _isolated_interpreter(tmp_path)
    marker = isolated.parent.parent / "pyvenv.cfg"
    marker.unlink()
    target = isolated.parent.parent / "marker-target"
    target.write_text("include-system-site-packages = false\n")
    if link_kind == "symlink":
        marker.symlink_to(target)
    else:
        os.link(target, marker)

    with pytest.raises(public.PublicFixtureError, match="invalid isolated"):
        public._validate_parquet_python(isolated)


@pytest.mark.parametrize(
    "marker",
    [
        b"include-system-site-packages = true\n",
        (
            b"include-system-site-packages = false\n"
            b"include-system-site-packages = true\n"
        ),
        b"x-include-system-site-packages = false\n",
        b"include-system-site-packages = false-ish\n",
        b"\xffinclude-system-site-packages = false\n",
        b"x" * (public._MAX_VENV_MARKER_BYTES + 1),
    ],
)
def test_parquet_interpreter_rejects_unsafe_or_decoy_markers(
    tmp_path: Path,
    marker: bytes,
):
    isolated = _isolated_interpreter(tmp_path)
    (isolated.parent.parent / "pyvenv.cfg").write_bytes(marker)

    with pytest.raises(public.PublicFixtureError, match="invalid isolated"):
        public._validate_parquet_python(isolated)


@pytest.mark.parametrize(
    ("field", "value_factory"),
    [
        ("prefix", lambda root, _production: str(root.parent)),
        ("base_prefix", lambda root, _production: str(root)),
        ("enable_user_site", lambda _root, _production: True),
        ("sys_path", lambda _root, production: [str(production)]),
        ("pyarrow_version", lambda _root, _production: "24.0.0"),
        ("pyarrow_file", lambda _root, _production: "/usr/lib/pyarrow.py"),
    ],
)
def test_parquet_probe_rejects_environment_identity_drift(
    tmp_path: Path,
    field: str,
    value_factory,
):
    isolated = _isolated_interpreter(tmp_path)
    root = isolated.parent.parent.resolve()
    production = public._resolved_production_venv()
    raw = _valid_probe_payload(
        root,
        **{field: value_factory(root, production)},
    )

    with pytest.raises(public.PublicFixtureError, match="invalid isolated"):
        public._validate_parquet_probe_result(
            raw,
            candidate_root=root,
            production_venv=production,
        )


def test_parquet_probe_rejects_malformed_or_oversized_output(
    tmp_path: Path,
):
    isolated = _isolated_interpreter(tmp_path)
    root = isolated.parent.parent.resolve()
    production = public._resolved_production_venv()

    invalid_schema = _valid_probe_payload(root, schema_version=True)
    for raw in (
        b"{",
        b'{"schema_version":1,"schema_version":1}',
        b"x" * (public._MAX_PARQUET_PROBE_BYTES + 1),
        invalid_schema,
    ):
        with pytest.raises(public.PublicFixtureError, match="invalid isolated"):
            public._validate_parquet_probe_result(
                raw,
                candidate_root=root,
                production_venv=production,
            )


def test_parquet_probe_runner_bounds_stdout_and_timeout(tmp_path: Path):
    overflow = f"import os\nos.write(1, b'x' * {public._MAX_PARQUET_PROBE_BYTES + 1})\n"
    with pytest.raises(public.PublicFixtureError, match="invalid isolated"):
        public._run_parquet_python_probe(
            Path(sys.executable),
            candidate_root=tmp_path,
            code=overflow,
            timeout_sec=1.0,
        )

    started = time.monotonic()
    with pytest.raises(public.PublicFixtureError, match="invalid isolated"):
        public._run_parquet_python_probe(
            Path(sys.executable),
            candidate_root=tmp_path,
            code="import time; time.sleep(5)",
            timeout_sec=0.05,
        )
    assert time.monotonic() - started < 1.0


def test_parquet_probe_runner_is_isolated_and_one_thread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("SHOULD_NOT_REACH_PROBE", "secret")
    code = (
        "import json, os, sys\n"
        "payload = {\n"
        "    'isolated': sys.flags.isolated,\n"
        "    'dont_write_bytecode': sys.flags.dont_write_bytecode,\n"
        "    'stdin_eof': os.read(0, 1) == b'',\n"
        "    'home': os.environ.get('HOME'),\n"
        "    'tmpdir': os.environ.get('TMPDIR'),\n"
        "    'ambient': os.environ.get('SHOULD_NOT_REACH_PROBE'),\n"
        "    'threads': [os.environ.get(name) for name in (\n"
        "        'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',\n"
        "        'NUMEXPR_NUM_THREADS', 'ARROW_NUM_THREADS')],\n"
        "}\n"
        "os.write(2, b'discarded')\n"
        "os.write(1, json.dumps(payload, sort_keys=True).encode('ascii'))\n"
    )

    raw = public._run_parquet_python_probe(
        Path(sys.executable),
        candidate_root=tmp_path,
        code=code,
        timeout_sec=1.0,
    )
    payload = json.loads(raw)

    assert payload == {
        "ambient": None,
        "dont_write_bytecode": 1,
        "home": str(tmp_path),
        "isolated": 1,
        "stdin_eof": True,
        "threads": ["1", "1", "1", "1", "1"],
        "tmpdir": str(tmp_path),
    }


def test_parquet_probe_runner_reaps_original_group_descendants_before_return(
    tmp_path: Path,
):
    marker = tmp_path / "late-probe-write"
    code = (
        "import os, pathlib, time\n"
        "child = os.fork()\n"
        "if child == 0:\n"
        "    time.sleep(0.25)\n"
        f"    pathlib.Path({str(marker)!r}).write_text('raced')\n"
        "    os._exit(0)\n"
        "os.write(1, b'{}')\n"
    )

    assert (
        public._run_parquet_python_probe(
            Path(sys.executable),
            candidate_root=tmp_path,
            code=code,
            timeout_sec=1.0,
        )
        == b"{}"
    )
    time.sleep(0.35)
    assert not marker.exists()


def test_parquet_probe_runner_does_not_wait_for_escaped_stdout_holder(
    tmp_path: Path,
):
    pid_file = tmp_path / "escaped-probe-pid"
    code = (
        "import os, pathlib, time\n"
        "ready_read, ready_write = os.pipe()\n"
        "child = os.fork()\n"
        "if child == 0:\n"
        "    os.close(ready_read)\n"
        "    os.setsid()\n"
        f"    pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()))\n"
        "    os.write(ready_write, b'1')\n"
        "    os.close(ready_write)\n"
        "    time.sleep(0.75)\n"
        "    os._exit(0)\n"
        "os.close(ready_write)\n"
        "os.read(ready_read, 1)\n"
        "os.close(ready_read)\n"
        "os.write(1, b'{}')\n"
        "os._exit(0)\n"
    )
    escaped_pid: int | None = None
    started = time.monotonic()
    try:
        assert (
            public._run_parquet_python_probe(
                Path(sys.executable),
                candidate_root=tmp_path,
                code=code,
                timeout_sec=0.1,
            )
            == b"{}"
        )
        elapsed = time.monotonic() - started
        escaped_pid = int(pid_file.read_text())

        assert elapsed < 0.5
        os.kill(escaped_pid, 0)
    finally:
        if escaped_pid is None and pid_file.exists():
            escaped_pid = int(pid_file.read_text())
        if escaped_pid is not None:
            try:
                os.kill(escaped_pid, public.signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize("boundary", ["marker", "path", "root"])
def test_parquet_interpreter_rejects_post_probe_retarget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
):
    isolated = _isolated_interpreter(tmp_path)
    root = isolated.parent.parent

    def racing_probe(_interpreter, *, candidate_root, **_kwargs):
        if boundary == "marker":
            marker = root / "pyvenv.cfg"
            replacement = root / ".replacement-marker"
            replacement.write_bytes(marker.read_bytes())
            os.replace(replacement, marker)
        elif boundary == "path":
            replacement = isolated.parent / ".replacement-python"
            replacement.symlink_to(Path(sys.executable))
            os.replace(replacement, isolated)
        else:
            old_root = root.with_name("old-isolated")
            root.rename(old_root)
            _isolated_interpreter(tmp_path)
        return _valid_probe_payload(candidate_root)

    monkeypatch.setattr(public, "_run_parquet_python_probe", racing_probe)

    with pytest.raises(public.PublicFixtureError, match="invalid isolated"):
        public._validate_parquet_python(isolated)


def test_parent_accepts_exact_private_worker_output_without_importing_pyarrow(
    tmp_path: Path,
):
    manifest = public.load_manifest(_MANIFEST)
    output = tmp_path / "output"
    _worker_result(output, manifest)

    prepared = public._validate_parquet_worker_output(
        manifest,
        manifest.fixtures,
        output,
        worker_sha256=public._sha256_file(public._PARQUET_WORKER),
        output_sample_rate=16000,
    )

    assert len(prepared) == 14
    assert all(len(item.samples) == 1600 for item in prepared.values())
    assert all(item.expected_text for item in prepared.values())
    assert "pyarrow" not in sys.modules


def test_private_cleanup_never_chmods_symlink_or_hardlink_targets(tmp_path: Path):
    external = tmp_path / "external-sentinel"
    external.write_bytes(b"sentinel")
    external.chmod(0o400)
    root = tmp_path / "private-root"
    nested = root / "nested"
    nested.mkdir(parents=True)
    local = nested / "local"
    local.write_bytes(b"local")
    local.chmod(0o400)
    (nested / "symlink").symlink_to(external)
    os.link(external, nested / "hardlink")
    nested.chmod(0o500)
    root.chmod(0o500)

    public._unlock_private_tree(root)

    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert stat.S_IMODE(nested.stat().st_mode) == 0o700
    assert stat.S_IMODE(local.stat().st_mode) == 0o400
    assert stat.S_IMODE(external.stat().st_mode) == 0o400
    assert external.read_bytes() == b"sentinel"
    public.shutil.rmtree(root)
    assert stat.S_IMODE(external.stat().st_mode) == 0o400
    assert external.read_bytes() == b"sentinel"


@pytest.mark.parametrize(
    "mutation",
    [
        "protocol",
        "pyarrow",
        "rank",
        "audio-hash",
        "duplicate-row",
        "traversal",
        "extra-file",
        "broad-mode",
        "hard-link",
        "malformed-audio",
        "over-duration",
    ],
)
def test_parent_rejects_malformed_or_unsafe_worker_output(
    tmp_path: Path, mutation: str
):
    manifest = public.load_manifest(_MANIFEST)
    output = tmp_path / "output"
    result = _worker_result(output, manifest)
    cases = result["cases"]
    assert isinstance(cases, list)
    first = cases[0]
    assert isinstance(first, dict)
    if mutation == "protocol":
        result["schema_version"] = 2
    elif mutation == "pyarrow":
        result["pyarrow_version"] = "24.0.0"
    elif mutation == "rank":
        first["selection_rank_sha256"] = "0" * 64
    elif mutation == "audio-hash":
        first["audio_sha256"] = "0" * 64
    elif mutation == "duplicate-row":
        first["row_index"] = cases[1]["row_index"]
    elif mutation == "traversal":
        first["audio_file"] = "../outside.wav"
    elif mutation == "extra-file":
        _private_write(output / "extra", b"x")
    elif mutation == "broad-mode":
        (output / str(first["audio_file"])).chmod(0o644)
    elif mutation == "hard-link":
        os.link(output / str(first["audio_file"]), tmp_path / "second-link.wav")
    elif mutation == "malformed-audio":
        audio = b"not a wav"
        _private_write(output / str(first["audio_file"]), audio)
        first["audio_size_bytes"] = len(audio)
        first["audio_sha256"] = hashlib.sha256(audio).hexdigest()
    elif mutation == "over-duration":
        audio = _ulaw_wav(30 * 8000 + 1)
        _private_write(output / str(first["audio_file"]), audio)
        first["audio_size_bytes"] = len(audio)
        first["audio_sha256"] = hashlib.sha256(audio).hexdigest()
    if mutation not in {"extra-file", "broad-mode", "hard-link"}:
        _rewrite_result(output, result)

    with pytest.raises(public.PublicFixtureError):
        public._validate_parquet_worker_output(
            manifest,
            manifest.fixtures,
            output,
            worker_sha256=public._sha256_file(public._PARQUET_WORKER),
            output_sample_rate=16000,
        )


def test_worker_request_reader_is_bounded_and_duplicate_key_strict(tmp_path: Path):
    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b"x" * (worker._MAX_REQUEST_BYTES + 1))
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}')

    with pytest.raises(worker.WorkerError):
        worker._load_request(oversized)
    with pytest.raises(worker.WorkerError):
        worker._load_request(duplicate)


@pytest.mark.parametrize(
    "source_path",
    [
        "/en-US~BALANCE/example.wav",
        "other/example.wav",
        "en-US~ADDRESS/example.wav",
        "en-US~BALANCE/nested/example.wav",
        "en-US~BALANCE/.hidden.wav",
        "en-US~BALANCE/example.mp3",
        "en-US~BALANCE/../example.wav",
        "en-US~BALANCE/./example.wav",
        "en-US~BALANCE//example.wav",
    ],
)
def test_worker_never_treats_unpinned_path_shapes_as_audio_paths(source_path: str):
    assert (
        worker._source_audio_basename("en-US~BALANCE/example.wav", "balance")
        == "example.wav"
    )
    assert worker._embedded_audio_basename("example.wav") == "example.wav"

    with pytest.raises(worker.WorkerError):
        worker._source_audio_basename(source_path, "balance")
    with pytest.raises(worker.WorkerError):
        worker._embedded_audio_basename(source_path)


def test_worker_has_one_unthreaded_read_and_pyarrow_is_not_a_runtime_dependency():
    source = inspect.getsource(worker._read_verified_rows)

    assert source.count("parquet.read(") == 1
    assert "use_threads=False" in source
    assert worker._PYARROW_VERSION == "25.0.0"
    dependency_files = [
        Path("requirements.txt"),
        Path("requirements-agent.txt"),
        Path("requirements-ondevice.txt"),
        Path("requirements-remote.txt"),
    ]
    assert all(
        "pyarrow" not in path.read_text(encoding="utf-8").casefold()
        for path in dependency_files
    )


def test_worker_runner_uses_exact_isolated_file_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manifest = public.load_manifest(_MANIFEST)
    isolated = _isolated_interpreter(tmp_path)
    monkeypatch.setattr(
        public,
        "_run_parquet_python_probe",
        lambda *_args, **_kwargs: _valid_probe_payload(isolated.parent.parent),
    )
    source = tmp_path / "source.parquet"
    source.write_bytes(b"synthetic parent-runner source")
    fake_worker = tmp_path / "fake worker;literal.py"
    fake_worker.write_text("# synthetic worker path\n")
    fake_worker.chmod(0o664)
    monkeypatch.setattr(public, "_PARQUET_WORKER", fake_worker)
    monkeypatch.setenv("SHOULD_NOT_REACH_WORKER", "secret")
    captured: dict[str, object] = {}
    events: list[object] = []

    class FakeProcess:
        pid = 999_999

        def __init__(self, argv, **kwargs):
            captured["argv"] = argv
            captured["kwargs"] = kwargs
            self.returncode = None
            private_worker = Path(argv[3])
            captured["worker_mode"] = stat.S_IMODE(private_worker.stat().st_mode)
            captured["worker_hash"] = hashlib.sha256(
                private_worker.read_bytes()
            ).hexdigest()
            output_index = argv.index("--output-dir") + 1
            _worker_result(Path(argv[output_index]), manifest)

        def wait(self, timeout=None):
            captured.setdefault("timeouts", []).append(timeout)
            events.append(("wait", timeout))
            self.returncode = 0
            return 0

        def poll(self):
            return self.returncode

    group_alive = True

    def kill_group(pid, signal_number):
        nonlocal group_alive
        events.append(("signal", signal_number))
        assert pid == FakeProcess.pid
        if signal_number == public.signal.SIGKILL:
            group_alive = False
            return
        if signal_number == 0 and not group_alive:
            raise ProcessLookupError

    real_validate = public._validate_parquet_worker_output

    def validate_after_reap(*args, **kwargs):
        events.append("validate")
        return real_validate(*args, **kwargs)

    monkeypatch.setattr(public.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(public.os, "killpg", kill_group)
    monkeypatch.setattr(public, "_validate_parquet_worker_output", validate_after_reap)
    prepared = public._extract_hf_audio_parquet(
        source,
        manifest,
        manifest.fixtures,
        parquet_python=isolated,
        output_sample_rate=16000,
        temporary_parent=tmp_path / "workers",
    )

    argv = captured["argv"]
    kwargs = captured["kwargs"]
    assert argv[:3] == [str(isolated), "-I", "-B"]
    assert Path(argv[3]).name == "public_voice_parquet_worker.py"
    assert Path(argv[3]) != fake_worker
    assert Path(argv[3]).parent == kwargs["cwd"]
    assert captured["worker_mode"] == 0o400
    assert (
        captured["worker_hash"] == hashlib.sha256(fake_worker.read_bytes()).hexdigest()
    )
    assert kwargs["stdin"] is public.subprocess.DEVNULL
    assert kwargs["stdout"] is public.subprocess.DEVNULL
    assert kwargs["stderr"] is public.subprocess.DEVNULL
    assert kwargs["start_new_session"] is True
    assert kwargs["close_fds"] is True
    assert "shell" not in kwargs
    assert "SHOULD_NOT_REACH_WORKER" not in kwargs["env"]
    assert captured["timeouts"] == [
        public._PARQUET_WORKER_TIMEOUT_SEC,
        public._PARQUET_GROUP_REAP_TIMEOUT_SEC,
    ]
    assert events[-1] == "validate"
    assert events.index(("signal", public.signal.SIGKILL)) < events.index("validate")
    assert len(prepared) == 14
    assert list((tmp_path / "workers").iterdir()) == []


@pytest.mark.parametrize("failure", ["nonzero", "partial", "timeout"])
def test_worker_runner_failure_cleans_private_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
):
    manifest = public.load_manifest(_MANIFEST)
    isolated = _isolated_interpreter(tmp_path)
    monkeypatch.setattr(
        public,
        "_run_parquet_python_probe",
        lambda *_args, **_kwargs: _valid_probe_payload(isolated.parent.parent),
    )
    source = tmp_path / "source.parquet"
    source.write_bytes(b"synthetic")
    fake_worker = tmp_path / "worker.py"
    fake_worker.write_text("# worker\n")
    fake_worker.chmod(0o664)
    monkeypatch.setattr(public, "_PARQUET_WORKER", fake_worker)
    killed: list[tuple[int, int]] = []

    class FakeProcess:
        pid = 999_998

        def __init__(self, argv, **_kwargs):
            self.returncode = None
            self.initial_wait = True
            if failure == "partial":
                output = Path(argv[argv.index("--output-dir") + 1])
                _private_write(output / "result.json", b"{}")

        def wait(self, timeout=None):
            if failure == "timeout" and self.initial_wait:
                self.initial_wait = False
                raise public.subprocess.TimeoutExpired("worker", timeout)
            self.returncode = 2 if failure == "nonzero" else 0
            return self.returncode

        def poll(self):
            return self.returncode

    group_alive = True

    def kill_group(pid, signal_number):
        nonlocal group_alive
        killed.append((pid, signal_number))
        if signal_number == public.signal.SIGKILL:
            group_alive = False
            return
        if signal_number == 0 and not group_alive:
            raise ProcessLookupError

    monkeypatch.setattr(public.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(public.os, "killpg", kill_group)
    with pytest.raises(public.PublicFixtureError):
        public._extract_hf_audio_parquet(
            source,
            manifest,
            manifest.fixtures,
            parquet_python=isolated,
            output_sample_rate=16000,
            temporary_parent=tmp_path / "workers",
        )

    assert list((tmp_path / "workers").iterdir()) == []
    assert killed[0] == (FakeProcess.pid, public.signal.SIGKILL)
    assert killed[-1] == (FakeProcess.pid, 0)


def test_worker_group_descendant_is_dead_before_output_can_race(tmp_path: Path):
    marker = tmp_path / "late-write"
    code = (
        "import os, pathlib, sys, time\n"
        "child = os.fork()\n"
        "if child == 0:\n"
        "    time.sleep(0.25)\n"
        "    pathlib.Path(sys.argv[1]).write_text('raced')\n"
        "    os._exit(0)\n"
        "os._exit(0)\n"
    )
    process = public.subprocess.Popen(
        [sys.executable, "-c", code, str(marker)],
        stdin=public.subprocess.DEVNULL,
        stdout=public.subprocess.DEVNULL,
        stderr=public.subprocess.DEVNULL,
        start_new_session=True,
    )
    assert process.wait(timeout=2.0) == 0

    public._terminate_worker_process_group(process)
    time.sleep(0.35)

    assert not marker.exists()
    with pytest.raises(ProcessLookupError):
        os.killpg(process.pid, 0)


def test_v3_evaluator_validates_worker_provenance_before_model_construction(
    tmp_path: Path,
):
    metadata_path, metadata = _prepared_v3_metadata(tmp_path)
    manifest = public.load_manifest(_MANIFEST)

    corpus = public_stt_eval._load_corpus(metadata_path, manifest)

    assert len(corpus.all_cases) == 14
    assert len(corpus.eligible_cases) == 14
    rows = metadata["cases"]
    assert isinstance(rows, list)
    tampered = copy.deepcopy(metadata)
    tampered_rows = tampered["cases"]
    tampered_rows[0]["selection"]["worker_protocol_version"] = 2
    _write_manifest(metadata_path, tampered)
    with pytest.raises(public_stt_eval.PublicSttPrerequisiteError):
        public_stt_eval._load_corpus(metadata_path, manifest)

    tampered = copy.deepcopy(metadata)
    tampered["cases"][0]["selection"]["row_index"] = []
    _write_manifest(metadata_path, tampered)
    with pytest.raises(public_stt_eval.PublicSttPrerequisiteError):
        public_stt_eval._load_corpus(metadata_path, manifest)

    tampered = copy.deepcopy(metadata)
    for row in tampered["cases"]:
        row["selection"]["audio_size_bytes"] = 8 * 1024 * 1024
    _write_manifest(metadata_path, tampered)
    with pytest.raises(public_stt_eval.PublicSttPrerequisiteError):
        public_stt_eval._load_corpus(metadata_path, manifest)


def test_evaluator_metadata_reader_is_bounded_and_link_safe(tmp_path: Path):
    oversized = tmp_path / "oversized-metadata.json"
    with oversized.open("wb") as handle:
        handle.truncate(public_stt_eval._MAX_METADATA_BYTES + 1)
    target = tmp_path / "metadata.json"
    target.write_text("{}")
    symlink = tmp_path / "metadata-link.json"
    symlink.symlink_to(target)
    hardlink = tmp_path / "metadata-hard.json"
    os.link(target, hardlink)

    for candidate in (oversized, symlink, hardlink):
        with pytest.raises(public_stt_eval.PublicSttPrerequisiteError):
            public_stt_eval._strict_json_file(candidate)

    race = tmp_path / "metadata-race.json"
    race.write_text("{}")
    with pytest.raises(public_stt_eval.PublicSttPrerequisiteError):
        with public_stt_eval._stable_regular_reader(
            race,
            maximum=64,
            require_single_link=True,
        ) as (handle, _opened):
            assert handle.read() == b"{}"
            race.write_text('{"grew":true}')

    race.write_text("{}")
    with pytest.raises(public_stt_eval.PublicSttPrerequisiteError):
        with public_stt_eval._stable_regular_reader(
            race,
            maximum=64,
            require_single_link=True,
        ) as (handle, _opened):
            race.unlink()
            race.write_text('{"replacement":true}')
            assert handle.read() == b"{}"
