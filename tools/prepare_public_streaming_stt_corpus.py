"""Convert a verified prepared public-voice suite into private f32le replay.

The existing public fixture loader performs provenance and NPY validation.
This tool only changes the container format needed by the streaming worker;
it does not download data, run a recognizer, or emit transcript rows.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

from tools.public_stt_eval import (
    _load_corpus as _load_public_corpus,
    _verify_case_snapshot,
)
from tools.public_voice_fixtures import load_manifest
from tools.streaming_stt.corpus import CorpusProvenance, LoadedCorpus
from tools.streaming_stt.corpus_writer import (
    CorpusWriteCase,
    CorpusWriterError,
    publish_private_corpus,
)


_SAFE_ID_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ERROR = {
    "ok": False,
    "error": "public_streaming_corpus_prerequisites_unavailable",
}


class PublicStreamingCorpusError(RuntimeError):
    """A detail-free invalid source, conversion, or destination failure."""


def _case_id(case: object) -> str:
    audio_path = getattr(case, "audio_path", None)
    if not isinstance(audio_path, Path):
        raise PublicStreamingCorpusError()
    value = audio_path.stem
    if _SAFE_ID_RE.fullmatch(value) is None:
        raise PublicStreamingCorpusError()
    return value


def _raw_case(case: object) -> tuple[str, bytes, str]:
    import numpy as np

    if (
        getattr(case, "assertion", None) != "transcript"
        or getattr(case, "sample_rate", None) != 16_000
        or type(getattr(case, "sample_count", None)) is not int
        or not isinstance(getattr(case, "reference", None), str)
    ):
        raise PublicStreamingCorpusError()
    try:
        samples = np.load(
            io.BytesIO(getattr(case, "audio_bytes")),
            allow_pickle=False,
        )
        if (
            not isinstance(samples, np.ndarray)
            or samples.ndim != 1
            or samples.dtype != np.float32
            or samples.size != case.sample_count
            or not np.isfinite(samples).all()
        ):
            raise PublicStreamingCorpusError()
        raw = samples.astype("<f4", copy=False).tobytes(order="C")
    except PublicStreamingCorpusError:
        raise
    except Exception:
        raise PublicStreamingCorpusError() from None
    if len(raw) != case.sample_count * 4:
        raise PublicStreamingCorpusError()
    return _case_id(case), raw, case.reference


def _provenance(public: object) -> CorpusProvenance:
    binding = getattr(public, "binding", None)
    if not isinstance(binding, Mapping):
        raise PublicStreamingCorpusError()
    suite = binding.get("suite")
    manifest_sha256 = binding.get("manifest_sha256")
    metadata_sha256 = binding.get("metadata_sha256")
    source_set_sha256 = binding.get("source_set_sha256")
    if (
        not isinstance(suite, str)
        or _SAFE_ID_RE.fullmatch(suite) is None
        or not isinstance(manifest_sha256, str)
        or _SHA256_RE.fullmatch(manifest_sha256) is None
        or not isinstance(metadata_sha256, str)
        or _SHA256_RE.fullmatch(metadata_sha256) is None
        or not isinstance(source_set_sha256, str)
        or _SHA256_RE.fullmatch(source_set_sha256) is None
    ):
        raise PublicStreamingCorpusError()
    return CorpusProvenance(
        kind="public-voice-v1",
        suite=suite,
        manifest_sha256=manifest_sha256,
        metadata_sha256=metadata_sha256,
        source_set_sha256=source_set_sha256,
    )


def prepare_public_corpus(
    *,
    public_manifest: Path | str,
    metadata: Path | str,
    output_dir: Path | str,
) -> LoadedCorpus:
    """Verify a prepared public suite and create one no-overwrite raw corpus."""

    manifest = load_manifest(Path(public_manifest))
    metadata_path = Path(metadata).expanduser().resolve(strict=True)
    public = _load_public_corpus(metadata_path, manifest)
    _verify_case_snapshot(public.all_cases)
    if len(public.eligible_cases) != len(public.all_cases):
        raise PublicStreamingCorpusError()
    converted = tuple(_raw_case(case) for case in public.eligible_cases)
    if not converted or len({case_id for case_id, _raw, _text in converted}) != len(
        converted
    ):
        raise PublicStreamingCorpusError()
    provenance = _provenance(public)

    purpose = "verified public voice streaming conversion v1"
    try:
        loaded = publish_private_corpus(
            cases=tuple(
                CorpusWriteCase(
                    case_id=case_id,
                    audio_bytes=raw,
                    reference=reference,
                    tags=("public-voice", provenance.suite),
                )
                for case_id, raw, reference in converted
            ),
            provenance=provenance,
            output_dir=output_dir,
            purpose=purpose,
        )
    except CorpusWriterError:
        raise PublicStreamingCorpusError() from None
    _verify_case_snapshot(public.all_cases)
    return loaded


def _safe_result(corpus: LoadedCorpus) -> dict[str, object]:
    if corpus.schema_version != 2 or corpus.provenance is None:
        raise PublicStreamingCorpusError()
    return {
        "ok": True,
        "corpus_sha256": corpus.digest,
        "cases": len(corpus.cases),
        "audio_bytes": corpus.audio_bytes,
        "purpose_sha256": hashlib.sha256(corpus.purpose.encode("utf-8")).hexdigest(),
        "provenance": corpus.provenance.as_dict(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert one already verified prepared public-voice suite into a "
            "private raw streaming-STT corpus; no download or model execution."
        )
    )
    parser.add_argument("--public-manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        corpus = prepare_public_corpus(
            public_manifest=args.public_manifest,
            metadata=args.metadata,
            output_dir=args.output_dir,
        )
        result: Mapping[str, object] = _safe_result(corpus)
        code = 0
    except Exception:  # noqa: BLE001 - paths and transcript rows stay private
        result = _SAFE_ERROR
        code = 2
    print(
        json.dumps(
            result,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
