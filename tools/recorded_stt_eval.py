"""Privacy-safe, recording-driven STT evaluation.

This command replays a labelled local corpus through the real streaming and
second-pass recognizers without constructing the chatbot, TTS, tools, or audio
devices.  References and hypotheses are reduced immediately to aggregate
accuracy counters; stdout and optional reports contain no transcript, clip id,
or filesystem path.

Examples::

    python -m tools.recorded_stt_eval
    python -m tools.recorded_stt_eval --set asr_max_active_paths=8
    python -m tools.recorded_stt_eval --manifest /private/labelled.json \
        --keyword vault --set asr_max_active_paths=8

Supplying one or more ``--set`` values runs both the machine's current baseline
and the candidate.  The candidate is promotable only when every labelled clip
is covered, word and character errors do not regress, target-keyword recall
does not regress, and at least one of those measures improves.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass, field, fields, replace
from numbers import Real
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from core.wer import normalize, word_error_rate


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_MANIFEST = _REPO_ROOT / "tests" / "fixtures" / "recorded_voice_manifest.json"
_SAFE_ERROR = {"ok": False, "error": "recorded_stt_prerequisites_unavailable"}
_VERIFIER_COMPLETED_OUTCOMES = frozenset(
    {
        "consensus",
        "empty",
        "tie",
        "no_quorum",
        "control_guard",
        "attested_control",
        "empty_veto",
        "empty_streaming_guard",
    }
)
_ARTIFACT_FIELDS = (
    "asr_tokens",
    "asr_encoder",
    "asr_decoder",
    "asr_joiner",
    "asr_final_model",
    "asr_final_tokens",
    "asr_final_decoder",
    "asr_final_joiner",
    "asr_final_verifier_model",
    "punct_model",
    "asr_final_hr_dict_dir",
    "asr_final_hr_lexicon",
    "asr_final_hr_rule_fsts",
    "asr_final_rule_fsts",
)
_ARTIFACT_LIST_FIELDS = {"asr_final_hr_rule_fsts", "asr_final_rule_fsts"}


class EvaluationPrerequisiteError(RuntimeError):
    """A deliberately detail-free local corpus/model prerequisite failure."""


@dataclass(frozen=True)
class _CorpusItem:
    reference: str
    samples: object
    sample_rate: int
    speech_sec: float | None

    # A failed assertion must not echo a private label or waveform.
    def __repr__(self) -> str:
        return "_CorpusItem(<private>)"


@dataclass(frozen=True)
class AccuracyTotals:
    clips: int = 0
    nonempty: int = 0
    exact: int = 0
    substitutions: int = 0
    insertions: int = 0
    deletions: int = 0
    ref_words: int = 0
    hyp_words: int = 0
    char_edits: int = 0
    ref_chars: int = 0
    hyp_chars: int = 0
    keyword_attempts: int = 0
    keyword_hits: int = 0

    @property
    def word_errors(self) -> int:
        return self.substitutions + self.insertions + self.deletions

    @property
    def wer(self) -> float:
        if self.ref_words == 0:
            return 0.0 if self.hyp_words == 0 else 1.0
        return self.word_errors / self.ref_words

    @property
    def cer(self) -> float:
        if self.ref_chars == 0:
            return 0.0 if self.hyp_chars == 0 else 1.0
        return self.char_edits / self.ref_chars

    def as_dict(self) -> dict[str, int | float]:
        return {
            "clips": self.clips,
            "nonempty": self.nonempty,
            "exact": self.exact,
            "wer": round(self.wer, 4),
            "cer": round(self.cer, 4),
            "word_errors": self.word_errors,
            "substitutions": self.substitutions,
            "insertions": self.insertions,
            "deletions": self.deletions,
            "ref_words": self.ref_words,
            "hyp_words": self.hyp_words,
            "char_edits": self.char_edits,
            "ref_chars": self.ref_chars,
            "hyp_chars": self.hyp_chars,
            "keyword_attempts": self.keyword_attempts,
            "keyword_hits": self.keyword_hits,
        }


@dataclass(frozen=True)
class EvaluationTotals:
    streaming: AccuracyTotals
    offline: AccuracyTotals
    selected: AccuracyTotals
    clips: int
    decisions: int
    offline_outcomes: Mapping[str, int]
    verifier_outcomes: Mapping[str, int] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return self.clips > 0 and self.selected.clips == self.clips

    def as_dict(self) -> dict[str, object]:
        return {
            "clips": self.clips,
            "decisions": self.decisions,
            "complete": self.complete,
            "offline_outcomes": dict(sorted(self.offline_outcomes.items())),
            "verifier_outcomes": dict(sorted(self.verifier_outcomes.items())),
            "streaming": self.streaming.as_dict(),
            "offline": self.offline.as_dict(),
            "selected": self.selected.as_dict(),
        }


def _enabled_verifier_evaluation_ok(config, totals: EvaluationTotals) -> bool:
    """Require a selected verifier to complete at least one decode without error.

    Empty output and safe consensus abstentions are completed verifier decisions,
    not failures. ``skipped`` and ``unavailable`` do not prove that the explicitly
    selected verifier decoded any recording.
    """
    backend = str(
        getattr(config, "asr_final_verifier_backend", "") or ""
    ).strip()
    if not backend:
        return True
    outcomes = totals.verifier_outcomes
    if int(outcomes.get("error", 0)) > 0:
        return False
    return any(
        int(outcomes.get(outcome, 0)) > 0
        for outcome in _VERIFIER_COMPLETED_OUTCOMES
    )


def _enabled_offline_evaluation_ok(config, totals: EvaluationTotals) -> bool:
    """Require an explicitly selected offline final to actually decode."""
    backend = str(getattr(config, "asr_final_backend", "") or "").strip()
    if not backend:
        return True
    outcomes = totals.offline_outcomes
    if int(outcomes.get("error", 0)) > 0:
        return False
    return any(int(outcomes.get(outcome, 0)) > 0 for outcome in ("decoded", "empty"))


@dataclass(frozen=True)
class CandidateComparison:
    wins: int
    ties: int
    losses: int
    promotable: bool

    def as_dict(self) -> dict[str, int | bool]:
        return {
            "wins": self.wins,
            "ties": self.ties,
            "losses": self.losses,
            "promotable": self.promotable,
        }


def _normalized_characters(text: str) -> str:
    """Case/punctuation-insensitive characters on the same tokens as WER."""
    return "".join(normalize(text))


def _edit_distance(reference: Sequence[object], hypothesis: Sequence[object]) -> int:
    """Memory-bounded Levenshtein distance for aggregate CER/comparison."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, 1):
        current = [i]
        for j, hyp_item in enumerate(hypothesis, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (ref_item != hyp_item),
                )
            )
        previous = current
    return previous[-1]


def _measure(
    pairs: Iterable[tuple[str, str]],
    *,
    keywords: Sequence[str] = (),
) -> AccuracyTotals:
    totals = Counter()
    normalized_keywords = [normalize(keyword) for keyword in keywords]
    normalized_keywords = [tokens for tokens in normalized_keywords if tokens]
    for reference, hypothesis in pairs:
        wer = word_error_rate(reference, hypothesis)
        ref_tokens = normalize(reference)
        hyp_tokens = normalize(hypothesis)
        ref_chars = _normalized_characters(reference)
        hyp_chars = _normalized_characters(hypothesis)
        totals.update(
            clips=1,
            nonempty=bool(hyp_tokens),
            exact=ref_tokens == hyp_tokens,
            substitutions=wer.substitutions,
            insertions=wer.insertions,
            deletions=wer.deletions,
            ref_words=wer.ref_words,
            hyp_words=wer.hyp_words,
            char_edits=_edit_distance(ref_chars, hyp_chars),
            ref_chars=len(ref_chars),
            hyp_chars=len(hyp_chars),
        )
        for keyword_tokens in normalized_keywords:
            if _contains_tokens(ref_tokens, keyword_tokens):
                totals["keyword_attempts"] += 1
                if _contains_tokens(hyp_tokens, keyword_tokens):
                    totals["keyword_hits"] += 1
    return AccuracyTotals(**{field.name: int(totals[field.name]) for field in fields(AccuracyTotals)})


def _sum_accuracy(left: AccuracyTotals, right: AccuracyTotals) -> AccuracyTotals:
    return AccuracyTotals(
        **{
            field.name: int(getattr(left, field.name) + getattr(right, field.name))
            for field in fields(AccuracyTotals)
        }
    )


def _pair_errors(reference: str, hypothesis: str) -> tuple[int, int]:
    measured = word_error_rate(reference, hypothesis)
    word_errors = measured.substitutions + measured.insertions + measured.deletions
    char_errors = _edit_distance(
        _normalized_characters(reference),
        _normalized_characters(hypothesis),
    )
    return word_errors, char_errors


def _contains_tokens(haystack: Sequence[str], needle: Sequence[str]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    width = len(needle)
    return any(
        list(haystack[i : i + width]) == list(needle)
        for i in range(len(haystack) - width + 1)
    )


def compare_candidate(
    baseline: EvaluationTotals,
    candidate: EvaluationTotals,
    baseline_selected_errors: Sequence[tuple[int, int]],
    candidate_selected_errors: Sequence[tuple[int, int]],
) -> CandidateComparison:
    """Compare per-clip selected errors, then enforce aggregate no-regression."""
    if len(baseline_selected_errors) != len(candidate_selected_errors):
        return CandidateComparison(
            0,
            0,
            max(len(baseline_selected_errors), len(candidate_selected_errors)),
            False,
        )
    wins = ties = losses = 0
    for baseline_errors, candidate_errors in zip(
        baseline_selected_errors, candidate_selected_errors
    ):
        if candidate_errors < baseline_errors:
            wins += 1
        elif candidate_errors > baseline_errors:
            losses += 1
        else:
            ties += 1

    b = baseline.selected
    c = candidate.selected
    no_regression = (
        candidate.complete
        and c.nonempty == candidate.clips
        and c.word_errors <= b.word_errors
        and c.char_edits <= b.char_edits
        and c.keyword_hits >= b.keyword_hits
        and losses <= wins
    )
    improvement = (
        c.word_errors < b.word_errors
        or c.char_edits < b.char_edits
        or c.keyword_hits > b.keyword_hits
    )
    return CandidateComparison(wins, ties, losses, no_regression and improvement)


def _parse_override(raw: str, known_fields: set[str]) -> tuple[str, object]:
    if "=" not in raw:
        raise EvaluationPrerequisiteError()
    name, value = raw.split("=", 1)
    if name not in known_fields:
        raise EvaluationPrerequisiteError()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = value
    return name, parsed


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_corpus(manifest_path: Path) -> tuple[list[_CorpusItem], str]:
    """Load hash-pinned private audio while keeping all details out of errors."""
    from core.engines.file_replay import load_waveform

    try:
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw)
        entries = manifest.get("clips")
        if not isinstance(entries, list) or not entries:
            raise ValueError
        clip_dir = Path(str(manifest.get("clip_dir") or "logs/fixture_audio"))
        if not clip_dir.is_absolute():
            clip_dir = _REPO_ROOT / clip_dir
        items: list[_CorpusItem] = []
        corpus_hash = hashlib.sha256(raw)
        seen: set[str] = set()
        for entry in entries:
            clip_id = entry["id"]
            reference = entry["expected_text"]
            expected_hash = entry["sha256"]
            if (
                not isinstance(clip_id, str)
                or not clip_id
                or clip_id in seen
                or not isinstance(reference, str)
                or not reference.strip()
                or re.fullmatch(r"[0-9a-f]{64}", str(expected_hash)) is None
            ):
                raise ValueError
            seen.add(clip_id)
            path = clip_dir / f"{clip_id}.wav"
            if _sha256_file(path) != expected_hash:
                raise ValueError
            samples, sample_rate = load_waveform(str(path))
            sample_rate = int(sample_rate)
            if sample_rate <= 0:
                raise ValueError
            speech_sec = entry.get("speech_sec")
            if speech_sec is not None:
                if isinstance(speech_sec, bool) or not isinstance(speech_sec, Real):
                    raise ValueError
                speech_sec = float(speech_sec)
                duration_sec = len(samples) / sample_rate
                if (
                    not math.isfinite(speech_sec)
                    or speech_sec <= 0.0
                    or speech_sec > duration_sec
                ):
                    raise ValueError
            items.append(_CorpusItem(reference, samples, sample_rate, speech_sec))
            corpus_hash.update(bytes.fromhex(expected_hash))
        return items, corpus_hash.hexdigest()
    except Exception as exc:  # noqa: BLE001 - public error must stay detail-free
        raise EvaluationPrerequisiteError() from exc


def _load_config():
    try:
        import sherpa_onnx  # noqa: F401 - explicit native prerequisite
        from core.config import apply_device_profile, load_config
        from core.engines.sherpa import SherpaConfig

        cfg = load_config()
        cfg = apply_device_profile(cfg, cfg.get("device", "desktop"))
        sherpa = SherpaConfig.from_dict(cfg.get("sherpa", {}))
        if not sherpa.asr_encoder or not Path(sherpa.asr_encoder).is_file():
            raise ValueError
        return sherpa
    except Exception as exc:  # noqa: BLE001 - public error must stay detail-free
        raise EvaluationPrerequisiteError() from exc


def _config_digest(config) -> str:
    payload = {field.name: getattr(config, field.name) for field in fields(config)}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _update_artifact_digest(digest, path: Path) -> None:
    """Hash one configured file/tree without exposing any local name."""
    if path.is_file():
        digest.update(b"file\0")
        digest.update(bytes.fromhex(_sha256_file(path)))
        return
    if path.is_dir():
        digest.update(b"dir\0")
        children = sorted(
            candidate for candidate in path.rglob("*") if candidate.is_file()
        )
        for child in children:
            relative = child.relative_to(path).as_posix().encode("utf-8")
            digest.update(hashlib.sha256(relative).digest())
            digest.update(bytes.fromhex(_sha256_file(child)))
        return
    raise ValueError


def _model_digest(config) -> str:
    digest = hashlib.sha256()
    try:
        found = 0
        for name in _ARTIFACT_FIELDS:
            raw_value = str(getattr(config, name, "") or "").strip()
            if not raw_value:
                continue
            if (
                name == "asr_final_verifier_model"
                and hasattr(config, "asr_final_verifier_backend")
                and not str(
                    getattr(config, "asr_final_verifier_backend", "") or ""
                ).strip()
            ):
                # Match readiness/runtime semantics: a disabled backend makes a
                # stale local path inert and it must not poison baseline proof.
                continue
            digest.update(name.encode("ascii"))
            values = (
                [part.strip() for part in raw_value.split(",") if part.strip()]
                if name in _ARTIFACT_LIST_FIELDS
                else [raw_value]
            )
            for value in values:
                _update_artifact_digest(digest, Path(value))
                found += 1
        if found == 0:
            raise ValueError
        return digest.hexdigest()
    except Exception as exc:  # noqa: BLE001 - public error must stay detail-free
        raise EvaluationPrerequisiteError() from exc


def _evaluate(config, corpus: Sequence[_CorpusItem], keywords: Sequence[str]):
    from core.engine import EngineCallbacks
    from core.engines.file_replay import FileReplayEngine

    engine = FileReplayEngine(config, asr_only=True)
    try:
        engine.start(EngineCallbacks())
        streaming_totals = AccuracyTotals()
        offline_totals = AccuracyTotals()
        selected_totals = AccuracyTotals()
        selected_errors: list[tuple[int, int]] = []
        decisions_total = 0
        outcomes: Counter[str] = Counter()
        verifier_outcomes: Counter[str] = Counter()
        for item in corpus:
            result = engine.evaluate_samples(
                item.samples,
                item.sample_rate,
                speech_sec=item.speech_sec,
            )
            decisions = tuple(result.decisions)
            decisions_total += len(decisions)
            outcomes.update(decision.offline_outcome for decision in decisions)
            verifier_outcomes.update(
                decision.verifier_outcome for decision in decisions
            )
            streaming = " ".join(
                decision.streaming_raw.strip()
                for decision in decisions
                if decision.streaming_raw.strip()
            )
            offline = " ".join(
                (decision.offline_raw or "").strip()
                for decision in decisions
                if (decision.offline_raw or "").strip()
            )
            selected = " ".join(
                decision.selected.strip()
                for decision in decisions
                if decision.selected.strip()
            )
            streaming_totals = _sum_accuracy(
                streaming_totals,
                _measure([(item.reference, streaming)], keywords=keywords),
            )
            offline_totals = _sum_accuracy(
                offline_totals,
                _measure([(item.reference, offline)], keywords=keywords),
            )
            selected_totals = _sum_accuracy(
                selected_totals,
                _measure([(item.reference, selected)], keywords=keywords),
            )
            selected_errors.append(_pair_errors(item.reference, selected))
    except Exception as exc:  # noqa: BLE001 - never expose native/private detail
        raise EvaluationPrerequisiteError() from exc
    finally:
        engine.stop()

    totals = EvaluationTotals(
        streaming=streaming_totals,
        offline=offline_totals,
        selected=selected_totals,
        clips=len(corpus),
        decisions=decisions_total,
        offline_outcomes=dict(outcomes),
        verifier_outcomes=dict(verifier_outcomes),
    )
    return totals, tuple(selected_errors)


def _write_report(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically write aggregate-only JSON as a private regular file."""
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".stt-eval-", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate-only STT evaluation over local labelled recordings."
    )
    parser.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="candidate SherpaConfig override; JSON values are accepted",
    )
    parser.add_argument(
        "--keyword",
        action="append",
        default=[],
        help="target phrase measured only as aggregate attempts/hits",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional aggregate-only JSON report (atomically written mode 600)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        corpus, corpus_digest = _load_corpus(args.manifest)
        baseline_config = _load_config()
        baseline, baseline_selected_errors = _evaluate(
            baseline_config, corpus, tuple(args.keyword)
        )
        payload: dict[str, object] = {
            "ok": (
                baseline.complete
                and baseline.selected.nonempty == baseline.clips
                and _enabled_offline_evaluation_ok(baseline_config, baseline)
                and _enabled_verifier_evaluation_ok(baseline_config, baseline)
            ),
            "corpus_digest": corpus_digest,
            "baseline_config_digest": _config_digest(baseline_config),
            "baseline_model_digest": _model_digest(baseline_config),
            "baseline": baseline.as_dict(),
        }
        exit_code = 0 if payload["ok"] else 2

        if args.overrides:
            known = {field.name for field in fields(baseline_config)}
            overrides = dict(_parse_override(raw, known) for raw in args.overrides)
            candidate_config = replace(baseline_config, **overrides)
            candidate, candidate_selected_errors = _evaluate(
                candidate_config, corpus, tuple(args.keyword)
            )
            comparison = compare_candidate(
                baseline,
                candidate,
                baseline_selected_errors,
                candidate_selected_errors,
            )
            if not (
                _enabled_offline_evaluation_ok(baseline_config, baseline)
                and _enabled_offline_evaluation_ok(candidate_config, candidate)
                and _enabled_verifier_evaluation_ok(baseline_config, baseline)
                and _enabled_verifier_evaluation_ok(candidate_config, candidate)
            ):
                comparison = replace(comparison, promotable=False)
            payload.update(
                candidate_config_digest=_config_digest(candidate_config),
                candidate_model_digest=_model_digest(candidate_config),
                candidate=candidate.as_dict(),
                comparison=comparison.as_dict(),
                ok=comparison.promotable,
            )
            exit_code = 0 if comparison.promotable else 3

        if args.output is not None:
            _write_report(args.output, payload)
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return exit_code
    except Exception:  # noqa: BLE001 - CLI failures must remain detail-free
        print(json.dumps(_SAFE_ERROR, sort_keys=True, separators=(",", ":")))
        return 2


if __name__ == "__main__":  # pragma: no cover - exercised through main tests
    raise SystemExit(main())
