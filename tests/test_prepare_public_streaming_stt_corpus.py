from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.test_public_voice_v3 import _prepared_v3_metadata
from tools import prepare_public_streaming_stt_corpus as prepare
from tools import streaming_stt_eval
from tools.streaming_stt.corpus import CorpusError, load_corpus


_V3_MANIFEST = Path("tests/fixtures/public_voice_manifest.v3.json")


def _public_case(tmp_path: Path, *, assertion: str = "transcript"):
    samples = np.asarray([0.0, 0.25, -0.25, 0.0], dtype=np.float32)
    buffer = io.BytesIO()
    np.save(buffer, samples, allow_pickle=False)
    return SimpleNamespace(
        assertion=assertion,
        sample_rate=16_000,
        sample_count=samples.size,
        reference="find my balance",
        audio_path=tmp_path / "minds14-balance.npy",
        audio_bytes=buffer.getvalue(),
    ), samples


def _install_public_loader(monkeypatch, public):
    monkeypatch.setattr(prepare, "load_manifest", lambda _path: object())
    monkeypatch.setattr(
        prepare,
        "_load_public_corpus",
        lambda _metadata, _manifest: public,
    )
    calls: list[tuple[object, ...]] = []

    def verify(cases):
        calls.append(tuple(cases))

    monkeypatch.setattr(prepare, "_verify_case_snapshot", verify)
    return calls


def test_verified_public_npy_is_converted_to_private_exact_f32le(
    tmp_path,
    monkeypatch,
):
    case, samples = _public_case(tmp_path)
    public = SimpleNamespace(
        all_cases=(case,),
        eligible_cases=(case,),
        binding={
            "manifest_sha256": "a" * 64,
            "metadata_sha256": "b" * 64,
            "source_set_sha256": "c" * 64,
            "suite": "intent-asr",
        },
    )
    calls = _install_public_loader(monkeypatch, public)
    metadata = tmp_path / "metadata.json"
    metadata.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "streaming-corpus"

    corpus = prepare.prepare_public_corpus(
        public_manifest=tmp_path / "public-manifest.json",
        metadata=metadata,
        output_dir=output,
    )

    assert calls == [(case,), (case,)]
    assert len(corpus.cases) == 1
    assert corpus.cases[0].audio_bytes == samples.astype("<f4").tobytes()
    assert corpus.cases[0].expected_text == "find my balance"
    assert corpus.cases[0].tags == ("public-voice", "intent-asr")
    assert corpus.schema_version == 2
    assert corpus.provenance is not None
    assert corpus.provenance.as_dict() == {
        "kind": "public-voice-v1",
        "suite": "intent-asr",
        "manifest_sha256": "a" * 64,
        "metadata_sha256": "b" * 64,
        "source_set_sha256": "c" * 64,
    }
    assert (output / "corpus.json").stat().st_mode & 0o777 == 0o600
    assert output.stat().st_mode & 0o777 == 0o700
    safe = prepare._safe_result(corpus)
    assert safe["provenance"] == corpus.provenance.as_dict()
    assert (
        streaming_stt_eval._corpus_binding(corpus)["provenance"] == safe["provenance"]
    )
    assert str(tmp_path) not in json.dumps(safe)


def test_public_conversion_is_no_overwrite(tmp_path, monkeypatch):
    case, _samples = _public_case(tmp_path)
    public = SimpleNamespace(
        all_cases=(case,),
        eligible_cases=(case,),
        binding={
            "manifest_sha256": "a" * 64,
            "metadata_sha256": "b" * 64,
            "source_set_sha256": "c" * 64,
            "suite": "intent-asr",
        },
    )
    _install_public_loader(monkeypatch, public)
    metadata = tmp_path / "metadata.json"
    metadata.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "streaming-corpus"
    first = prepare.prepare_public_corpus(
        public_manifest=tmp_path / "public-manifest.json",
        metadata=metadata,
        output_dir=output,
    )
    before = first.path.read_bytes()

    with pytest.raises(prepare.PublicStreamingCorpusError):
        prepare.prepare_public_corpus(
            public_manifest=tmp_path / "public-manifest.json",
            metadata=metadata,
            output_dir=output,
        )

    assert first.path.read_bytes() == before


def test_nontranscript_public_case_is_rejected_before_output(tmp_path, monkeypatch):
    case, _samples = _public_case(tmp_path, assertion="silence")
    public = SimpleNamespace(
        all_cases=(case,),
        eligible_cases=(case,),
        binding={
            "manifest_sha256": "a" * 64,
            "metadata_sha256": "b" * 64,
            "source_set_sha256": "c" * 64,
            "suite": "intent-asr",
        },
    )
    _install_public_loader(monkeypatch, public)
    metadata = tmp_path / "metadata.json"
    metadata.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "streaming-corpus"

    with pytest.raises(prepare.PublicStreamingCorpusError):
        prepare.prepare_public_corpus(
            public_manifest=tmp_path / "public-manifest.json",
            metadata=metadata,
            output_dir=output,
        )

    assert not output.exists()


def test_real_v3_metadata_converts_with_exact_public_provenance(tmp_path):
    metadata, _payload = _prepared_v3_metadata(tmp_path)
    manifest = prepare.load_manifest(_V3_MANIFEST)
    public = prepare._load_public_corpus(metadata, manifest)

    corpus = prepare.prepare_public_corpus(
        public_manifest=_V3_MANIFEST,
        metadata=metadata,
        output_dir=tmp_path / "streaming-v3",
    )

    assert len(corpus.cases) == 14
    assert corpus.schema_version == 2
    assert corpus.provenance is not None
    assert corpus.provenance.as_dict() == {
        "kind": "public-voice-v1",
        "suite": "intent-asr",
        "manifest_sha256": public.binding["manifest_sha256"],
        "metadata_sha256": public.binding["metadata_sha256"],
        "source_set_sha256": public.binding["source_set_sha256"],
    }
    report_binding = streaming_stt_eval._corpus_binding(corpus)
    assert report_binding["provenance"] == corpus.provenance.as_dict()
    assert report_binding["schema_version"] == 2


def test_structured_public_provenance_is_strict(tmp_path, monkeypatch):
    case, _samples = _public_case(tmp_path)
    public = SimpleNamespace(
        all_cases=(case,),
        eligible_cases=(case,),
        binding={
            "manifest_sha256": "a" * 64,
            "metadata_sha256": "b" * 64,
            "source_set_sha256": "c" * 64,
            "suite": "intent-asr",
        },
    )
    _install_public_loader(monkeypatch, public)
    metadata = tmp_path / "metadata.json"
    metadata.write_text("{}\n", encoding="utf-8")
    corpus = prepare.prepare_public_corpus(
        public_manifest=tmp_path / "public-manifest.json",
        metadata=metadata,
        output_dir=tmp_path / "streaming-corpus",
    )
    payload = json.loads(corpus.path.read_text(encoding="utf-8"))
    payload["provenance"]["source_set_sha256"] = "C" * 64
    corpus.path.write_text(json.dumps(payload), encoding="utf-8")
    corpus.path.chmod(0o600)

    with pytest.raises(CorpusError):
        load_corpus(corpus.path)


def test_converter_help_excludes_download_and_model_execution():
    help_text = " ".join(prepare._parser().format_help().split())

    assert "no download or model execution" in help_text
