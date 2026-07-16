"""Aggregate-only local-GPU STT and choice-only selector bake-off.

This evaluator compares endpointed Faster-Whisper candidates with the real
streaming Zipformer, SenseVoice second pass, and production-selected final on
identical hash-pinned PCM.  It starts no chatbot, TTS, tools, audio device, or
live recorder.  An optional Ollama model may choose only an existing ASR source
index; it cannot return corrected text.

References, hypotheses, prompts, clip identifiers, and paths remain private in
memory.  Standard output and optional mode-600 reports contain aggregate metrics
and artifact digests only.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter_ns
from typing import Sequence

from core.engine import EngineCallbacks
from core.engines._faster_whisper import FasterWhisperEndpointRecognizer
from core.engines.file_replay import FileReplayEngine
from core.wer import normalize
import tools.recorded_stt_eval as recorded_stt_eval
from tools.recorded_stt_eval import (
    EvaluationPrerequisiteError,
    _config_digest,
    _load_corpus,
    _model_digest,
    _update_artifact_digest,
    _write_report,
)
from tools.stt_selector_eval import (
    ConsensusSelector,
    LocalOllamaChoiceSelector,
    SingleSourceSelector,
    SourceHypothesis,
    VerifierCase,
    evaluate_variants,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_MANIFEST = _REPO_ROOT / "tests" / "fixtures" / "recorded_voice_manifest.json"
_DEFAULT_SMALL_MODEL = "Systran/faster-whisper-small"
_DEFAULT_TURBO_MODEL = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
_DEFAULT_SMALL_REVISION = "536b0662742c02347bc0e980a01041f333bce120"
_DEFAULT_TURBO_REVISION = "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf"
_SAFE_ERROR = {"ok": False, "error": "local_gpu_stt_prerequisites_unavailable"}


@dataclass(frozen=True)
class _PrivateSourceRow:
    reference: str = field(repr=False)
    production: str = field(repr=False)
    streaming: str = field(repr=False)
    offline: str = field(repr=False)


@dataclass(frozen=True)
class _DecodeBatch:
    texts: tuple[str, ...] = field(repr=False)
    cold_first_ms: float
    warm_p50_ms: float
    warm_p95_ms: float
    warm_max_ms: float
    repeat_disagreements: int
    empty_outputs: int
    confidence_min: float
    confidence_median: float
    avg_logprob_min: float | None
    no_speech_probability_max: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "cold_first_decode_ms": self.cold_first_ms,
            "warm_decode_ms": {
                "p50": self.warm_p50_ms,
                "p95": self.warm_p95_ms,
                "max": self.warm_max_ms,
            },
            "repeat_disagreements": self.repeat_disagreements,
            "empty_outputs": self.empty_outputs,
            "confidence_diagnostic": {
                "calibrated_probability": False,
                "min": self.confidence_min,
                "median": self.confidence_median,
                "avg_logprob_min": self.avg_logprob_min,
                "no_speech_probability_max": self.no_speech_probability_max,
            },
        }


def _joined(decisions: Sequence[object], field_name: str) -> str:
    values: list[str] = []
    for decision in decisions:
        value = getattr(decision, field_name, None)
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
    return " ".join(values)


def _load_machine_config(config_root: Path):
    from core.config import apply_device_profile, load_config
    from core.engines.sherpa import SherpaConfig

    root = config_root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise EvaluationPrerequisiteError()
    main_path = root / "config.json"
    local_path = root / "config.local.json"
    raw = load_config(str(main_path), local=str(local_path))
    raw = apply_device_profile(raw, raw.get("device", "desktop"))
    effective = SherpaConfig.from_dict(raw.get("sherpa", {}))
    if not effective.asr_encoder:
        raise EvaluationPrerequisiteError()
    # A bake-off baseline must remain the established streaming/SenseVoice
    # production selector even after a machine has opted into this verifier.
    # Keep the effective config separately for provenance, but never decode it
    # here or add the same GPU model to both sides of the comparison.
    baseline = replace(
        effective,
        asr_final_verifier_backend="",
        asr_final_verifier_model="",
    )
    return baseline, effective, root


def _production_rows(config, config_root: Path, corpus) -> tuple[_PrivateSourceRow, ...]:
    engine = FileReplayEngine(config, asr_only=True)
    previous_cwd = Path.cwd()
    try:
        os.chdir(config_root)
        engine.start(EngineCallbacks())
    finally:
        os.chdir(previous_cwd)
    rows: list[_PrivateSourceRow] = []
    try:
        for item in corpus:
            result = engine.evaluate_samples(
                item.samples,
                item.sample_rate,
                speech_sec=item.speech_sec,
            )
            decisions = tuple(result.decisions)
            rows.append(
                _PrivateSourceRow(
                    reference=item.reference,
                    production=_joined(decisions, "selected"),
                    streaming=_joined(decisions, "streaming_raw"),
                    offline=_joined(decisions, "offline_raw"),
                )
            )
    finally:
        engine.stop()
    return tuple(rows)


def _production_model_digest(config, config_root: Path) -> str:
    previous_cwd = Path.cwd()
    try:
        os.chdir(config_root)
        return _model_digest(config)
    finally:
        os.chdir(previous_cwd)


def _local_snapshot(model: str, *, revision: str | None = None) -> Path:
    if not isinstance(model, str) or not model.strip():
        raise EvaluationPrerequisiteError()
    candidate = Path(model).expanduser()
    if candidate.is_dir():
        return candidate.resolve(strict=True)
    if revision is None or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise EvaluationPrerequisiteError()
    try:
        from huggingface_hub import snapshot_download

        resolved = Path(
            snapshot_download(
                repo_id=model,
                revision=revision,
                local_files_only=True,
            )
        ).resolve(strict=True)
    except Exception as exc:
        raise EvaluationPrerequisiteError() from exc
    if not resolved.is_dir():
        raise EvaluationPrerequisiteError()
    return resolved


def _tree_digest(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        _update_artifact_digest(digest, path)
    except Exception as exc:
        raise EvaluationPrerequisiteError() from exc
    return digest.hexdigest()


def _percentile_ms(values_ns: Sequence[int], percentile: float) -> float:
    if not values_ns:
        return 0.0
    ordered = sorted(values_ns)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return round(ordered[index] / 1_000_000, 3)


def _finite_summary(values: Sequence[float], *, mode: str) -> float | None:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return None
    if mode == "min":
        return round(finite[0], 6)
    if mode == "max":
        return round(finite[-1], 6)
    return round(finite[len(finite) // 2], 6)


def _decode_model(
    model_path: Path,
    corpus,
    *,
    repeats: int,
    recognizer_factory=FasterWhisperEndpointRecognizer,
) -> _DecodeBatch:
    recognizer = recognizer_factory(model_path)
    outputs: list[str] = []
    warm_latencies: list[int] = []
    cold_first_ns = 0
    disagreements = 0
    confidences: list[float] = []
    avg_logprobs: list[float] = []
    no_speech_values: list[float] = []

    for clip_index, item in enumerate(corpus):
        normalized_outputs: set[tuple[str, ...]] = set()
        first_text = ""
        for repeat_index in range(repeats):
            started = perf_counter_ns()
            result = recognizer.transcribe(item.samples, item.sample_rate)
            elapsed = perf_counter_ns() - started
            if clip_index == 0 and repeat_index == 0:
                cold_first_ns = elapsed
            else:
                warm_latencies.append(elapsed)
            if repeat_index == 0:
                first_text = result.text
                confidences.append(float(result.confidence))
                if result.avg_logprob is not None:
                    avg_logprobs.append(float(result.avg_logprob))
                if result.no_speech_probability is not None:
                    no_speech_values.append(float(result.no_speech_probability))
            normalized_outputs.add(tuple(normalize(result.text)))
        disagreements += len(normalized_outputs) > 1
        outputs.append(first_text)

    return _DecodeBatch(
        texts=tuple(outputs),
        cold_first_ms=round(cold_first_ns / 1_000_000, 3),
        warm_p50_ms=_percentile_ms(warm_latencies, 0.50),
        warm_p95_ms=_percentile_ms(warm_latencies, 0.95),
        warm_max_ms=_percentile_ms(warm_latencies, 1.0),
        repeat_disagreements=disagreements,
        empty_outputs=sum(not normalize(text) for text in outputs),
        confidence_min=_finite_summary(confidences, mode="min") or 0.0,
        confidence_median=_finite_summary(confidences, mode="median") or 0.0,
        avg_logprob_min=_finite_summary(avg_logprobs, mode="min"),
        no_speech_probability_max=_finite_summary(no_speech_values, mode="max"),
    )


def _cases(
    rows: Sequence[_PrivateSourceRow],
    small: Sequence[str],
    turbo: Sequence[str],
) -> tuple[VerifierCase, ...]:
    if not (len(rows) == len(small) == len(turbo)):
        raise EvaluationPrerequisiteError()
    return tuple(
        VerifierCase(
            row.reference,
            (
                SourceHypothesis("production", row.production),
                SourceHypothesis("streaming", row.streaming),
                SourceHypothesis("offline", row.offline),
                SourceHypothesis("gpu_small", small_text),
                SourceHypothesis("gpu_turbo", turbo_text),
            ),
            "production",
        )
        for row, small_text, turbo_text in zip(rows, small, turbo)
    )


def _selector_variants(llm_models: Sequence[str], repeats: int):
    variants: dict[str, object] = {
        "gpu_small_direct": SingleSourceSelector("gpu_small"),
        "gpu_turbo_direct": SingleSourceSelector("gpu_turbo"),
        "exact_consensus_small": ConsensusSelector(
            source_ids=("streaming", "offline", "gpu_small"),
            min_support=2,
            min_similarity=1.0,
        ),
        "exact_consensus_turbo": ConsensusSelector(
            source_ids=("streaming", "offline", "gpu_turbo"),
            min_support=2,
            min_similarity=1.0,
        ),
    }
    for index, model in enumerate(llm_models):
        variants[f"llm_choice_{index}"] = LocalOllamaChoiceSelector(
            model,
            source_ids=("streaming", "offline", "gpu_small"),
            repeats=repeats,
            max_sources=3,
        )
    return variants


def _ollama_identities(models: Sequence[str]) -> list[dict[str, object]]:
    from tools.conversation_eval.identity import verify_ollama_blob_identity

    records = []
    for model in models:
        identity = verify_ollama_blob_identity(
            model,
            host="http://127.0.0.1:11434",
            timeout_sec=5.0,
        )
        records.append(
            {
                "model": model,
                "ok": identity.ok,
                "blob_sha256": identity.blob_sha256,
                "effective_config_sha256": identity.effective_config_sha256,
            }
        )
    return records


def _contract_digests(*, decode_repeats: int, llm_repeats: int) -> dict[str, str]:
    from tools.conversation_eval.provenance import json_sha256

    return {
        "gpu_decode": json_sha256(
            {
                "device": "cuda:0",
                "compute_type": "float16",
                "language": "en",
                "task": "transcribe",
                "beam_size": 5,
                "patience": 1.0,
                "temperature": 0.0,
                "vad_filter": False,
                "condition_on_previous_text": False,
                "prompt": None,
                "hotwords": None,
                "decode_repeats": decode_repeats,
            }
        ),
        "llm_choice": json_sha256(
            {
                "system_prompt": LocalOllamaChoiceSelector._SYSTEM_PROMPT,
                "source_ids": ["streaming", "offline", "gpu_small"],
                "temperature": 0.0,
                "seed": 0,
                "top_k": 1,
                "num_predict": 8,
                "permutations": "all",
                "repeats": llm_repeats,
                "output": "existing_source_index_or_abstain",
            }
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate-only local-GPU STT selector bake-off."
    )
    parser.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST)
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=_REPO_ROOT,
        help="root used for relative private manifest clip_dir (never reported)",
    )
    parser.add_argument(
        "--config-root",
        type=Path,
        default=_REPO_ROOT,
        help="speaker config/model root (never reported)",
    )
    parser.add_argument("--small-model", default=_DEFAULT_SMALL_MODEL)
    parser.add_argument("--turbo-model", default=_DEFAULT_TURBO_MODEL)
    parser.add_argument("--small-revision", default=_DEFAULT_SMALL_REVISION)
    parser.add_argument("--turbo-revision", default=_DEFAULT_TURBO_REVISION)
    parser.add_argument(
        "--llm-model",
        action="append",
        default=[],
        help="optional local Ollama choice-only arbiter; repeat to compare models",
    )
    parser.add_argument("--decode-repeats", type=int, default=3)
    parser.add_argument("--llm-repeats", type=int, default=2)
    parser.add_argument("--keyword", action="append", default=[])
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if not 2 <= args.decode_repeats <= 8 or not 1 <= args.llm_repeats <= 8:
            raise EvaluationPrerequisiteError()
        recorded_stt_eval._REPO_ROOT = args.corpus_root.expanduser().resolve(strict=True)
        corpus, corpus_digest = _load_corpus(args.manifest)
        config, effective_config, config_root = _load_machine_config(
            args.config_root
        )
        rows = _production_rows(config, config_root, corpus)

        small_path = _local_snapshot(
            args.small_model,
            revision=args.small_revision,
        )
        turbo_path = _local_snapshot(
            args.turbo_model,
            revision=args.turbo_revision,
        )
        small = _decode_model(small_path, corpus, repeats=args.decode_repeats)
        turbo = _decode_model(turbo_path, corpus, repeats=args.decode_repeats)
        gc.collect()

        llm_identities = _ollama_identities(args.llm_model)
        variants = _selector_variants(tuple(args.llm_model), args.llm_repeats)
        report = evaluate_variants(
            _cases(rows, small.texts, turbo.texts),
            variants,
            keywords=tuple(args.keyword),
        )
        report.update(
            corpus_digest=corpus_digest,
            production_config_digest=_config_digest(config),
            production_model_digest=_production_model_digest(config, config_root),
            effective_machine_config_digest=_config_digest(effective_config),
            effective_machine_model_digest=_production_model_digest(
                effective_config,
                config_root,
            ),
            gpu_models={
                "small": {
                    "artifact_digest": _tree_digest(small_path),
                    "revision": args.small_revision,
                    **small.as_dict(),
                },
                "turbo": {
                    "artifact_digest": _tree_digest(turbo_path),
                    "revision": args.turbo_revision,
                    **turbo.as_dict(),
                },
            },
            ollama_models=llm_identities,
            runtime={
                "python": platform.python_version(),
                "faster_whisper": __import__("faster_whisper").__version__,
                "ctranslate2": __import__("ctranslate2").__version__,
                "device": "cuda:0",
                "compute_type": "float16",
            },
            contract_digests=_contract_digests(
                decode_repeats=args.decode_repeats,
                llm_repeats=args.llm_repeats,
            ),
        )

        deterministic = (
            small.repeat_disagreements == 0
            and turbo.repeat_disagreements == 0
            and all(record["ok"] is True for record in llm_identities)
        )
        if not deterministic:
            report["ok"] = False
        for name, variant in report["variants"].items():
            if name.startswith("gpu_small") or name.endswith("_small"):
                model_deterministic = small.repeat_disagreements == 0
            elif name.startswith("gpu_turbo") or name.endswith("_turbo"):
                model_deterministic = turbo.repeat_disagreements == 0
            else:
                model_deterministic = deterministic
            variant["decode_deterministic"] = model_deterministic
            if not model_deterministic:
                variant["comparison"]["promotable"] = False

        if args.output is not None:
            _write_report(args.output, report)
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))

        promotable = [
            value["comparison"]["promotable"]
            for value in report["variants"].values()
        ]
        return 0 if report["ok"] and any(promotable) else 3
    except Exception:  # noqa: BLE001 - never expose private/native detail
        print(json.dumps(_SAFE_ERROR, sort_keys=True, separators=(",", ":")))
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
