"""Strict aggregate-only AMI forced-alignment endpoint proxy metrics.

The labels consumed here are a diagnostic proxy derived from AMI's forced
word alignment.  They are deliberately not exposed through the generic
corpus schema and are not acoustic speech-end ground truth.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Sequence

from .corpus import LoadedCorpus
from .manifest import PARAKEET_CPP_ADAPTER, PARAKEET_REALTIME_EOU_ADAPTER
from .metrics import RunRecord, aggregate_metrics
from .protocol import (
    NATIVE_ENDPOINT_PROTOCOL_VERSION,
    PARAKEET_CPP_ENDPOINT_PROTOCOL_VERSION,
    CaseTrace,
    NativeFinalEvent,
    ParakeetCppFinalEvent,
    PartialEvent,
    ReadyEvent,
)


AMI_ENDPOINT_PROXY_SCHEMA_VERSION = 1
AMI_ENDPOINT_PROXY_KIND = "ami-forced-aligned-last-word-end-proxy-v1"
AMI_ENDPOINT_PROXY_LABEL_SET_KIND = (
    "ami-forced-aligned-last-word-end-label-set-v1"
)
AMI_ENDPOINT_PROXY_SIDECAR_KIND = "ami-speech-end-proxy-labels-v1"
AMI_ENDPOINT_PROXY_LABEL_CONTRACT = "ami-forced-aligned-last-word-end-v1"
AMI_ENDPOINT_PROXY_FIXTURE_ID = "public-conversation-96-v1"
AMI_ENDPOINT_PROXY_SOURCE_ID = "ami-es2004a-close-far"

_SAMPLE_RATE_HZ = 16_000
_EXPECTED_CASES = 24
_EXPECTED_ISOLATED_CASES = 16
_EXPECTED_TRANSITION_CASES = 8
_MAX_REPEATS = 8
_MAX_POST_PROXY_CONTEXT_SAMPLES = _SAMPLE_RATE_HZ // 10
_SAFE_ID_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SIDECAR_PRIVACY = {
    "contains_audio": False,
    "contains_transcripts": False,
    "contains_local_paths": False,
    "contains_raw_speaker_identifiers": False,
}
_SIDECAR_EVIDENCE_SCOPE = {
    "after_pcm": True,
    "forced_alignment_proxy": True,
    "acoustic_ground_truth": False,
    "capture": False,
    "vad": False,
    "device": False,
    "live_latency": False,
    "default_promotion": False,
}


class AmiEndpointProxyError(ValueError):
    """A detail-free AMI endpoint-proxy validation failure."""


@dataclass(frozen=True, slots=True)
class AmiEndpointProxyLabel:
    """One private, exact corpus-row label."""

    case_id: str = field(repr=False)
    pcm_sha256: str = field(repr=False)
    samples: int = field(repr=False)
    forced_aligned_last_word_end_sample: int = field(repr=False)
    window_kind: str = field(repr=False)
    channel: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class AmiEndpointProxyBinding:
    """Exact corpus/receipt/label-set binding for one private AMI fixture."""

    corpus_manifest_sha256: str
    receipt_sha256: str
    sidecar_sha256: str
    annotation_metadata_sha256: str
    label_set_sha256: str
    labels: tuple[AmiEndpointProxyLabel, ...] = field(repr=False)


def _fail() -> object:
    raise AmiEndpointProxyError("invalid AMI endpoint proxy evidence")


def _valid_sha256(value: object) -> bool:
    return type(value) is str and _SHA256_RE.fullmatch(value) is not None


def _validated_labels(
    labels: Sequence[AmiEndpointProxyLabel],
) -> tuple[AmiEndpointProxyLabel, ...]:
    if type(labels) is not tuple or not 1 <= len(labels) <= _EXPECTED_CASES:
        _fail()
    seen_ids: set[str] = set()
    result: list[AmiEndpointProxyLabel] = []
    for label in labels:
        if (
            type(label) is not AmiEndpointProxyLabel
            or type(label.case_id) is not str
            or _SAFE_ID_RE.fullmatch(label.case_id) is None
            or label.case_id in seen_ids
            or not _valid_sha256(label.pcm_sha256)
            or type(label.samples) is not int
            or label.samples <= 1
            or type(label.forced_aligned_last_word_end_sample) is not int
            or not 0
            < label.forced_aligned_last_word_end_sample
            < label.samples
            or label.samples - label.forced_aligned_last_word_end_sample
            > _MAX_POST_PROXY_CONTEXT_SAMPLES
            or label.window_kind not in {"isolated", "turn_transition"}
            or label.channel not in {"close", "far"}
        ):
            _fail()
        seen_ids.add(label.case_id)
        result.append(label)
    return tuple(result)


def _canonical_label_rows(
    labels: Sequence[AmiEndpointProxyLabel],
) -> tuple[dict[str, object], ...]:
    validated = _validated_labels(labels)
    return tuple(
        {
            "case_id": label.case_id,
            "pcm_sha256": label.pcm_sha256,
            "samples": label.samples,
            "forced_aligned_last_word_end_sample": (
                label.forced_aligned_last_word_end_sample
            ),
            "window_kind": label.window_kind,
            "channel": label.channel,
        }
        for label in validated
    )


def canonical_label_set_sha256(
    labels: Sequence[AmiEndpointProxyLabel],
) -> str:
    """Hash exact ordered private rows using one domain-separated encoding."""

    try:
        payload = {
            "kind": AMI_ENDPOINT_PROXY_LABEL_SET_KIND,
            "labels": _canonical_label_rows(labels),
        }
        raw = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
        return hashlib.sha256(raw).hexdigest()
    except AmiEndpointProxyError:
        raise
    except (OverflowError, TypeError, UnicodeError, ValueError):
        raise AmiEndpointProxyError(
            "invalid AMI endpoint proxy evidence"
        ) from None


def canonical_sidecar_sha256(
    *,
    annotation_metadata_sha256: str,
    label_set_sha256: str,
    labels: Sequence[AmiEndpointProxyLabel],
) -> str:
    """Reconstruct the exact canonical AMI sidecar digest from typed rows."""

    try:
        rows = _canonical_label_rows(labels)
        if (
            not _valid_sha256(annotation_metadata_sha256)
            or not _valid_sha256(label_set_sha256)
            or canonical_label_set_sha256(labels) != label_set_sha256
        ):
            _fail()
        payload = {
            "schema_version": 1,
            "kind": AMI_ENDPOINT_PROXY_SIDECAR_KIND,
            "fixture_id": AMI_ENDPOINT_PROXY_FIXTURE_ID,
            "source_id": AMI_ENDPOINT_PROXY_SOURCE_ID,
            "corpus_suite": "conversation96-ami",
            "sample_rate_hz": _SAMPLE_RATE_HZ,
            "label_contract": AMI_ENDPOINT_PROXY_LABEL_CONTRACT,
            "scoreable_scope": "isolated_nonoverlap_only",
            "annotation_metadata_sha256": annotation_metadata_sha256,
            "label_set_sha256": label_set_sha256,
            "labels": rows,
            "privacy": _SIDECAR_PRIVACY,
            "evidence_scope": _SIDECAR_EVIDENCE_SCOPE,
        }
        raw = (
            json.dumps(
                payload,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("ascii")
            + b"\n"
        )
        return hashlib.sha256(raw).hexdigest()
    except AmiEndpointProxyError:
        raise
    except (OverflowError, TypeError, UnicodeError, ValueError):
        raise AmiEndpointProxyError(
            "invalid AMI endpoint proxy evidence"
        ) from None


def validate_ami_endpoint_proxy_binding(
    corpus: LoadedCorpus,
    binding: AmiEndpointProxyBinding,
) -> tuple[AmiEndpointProxyLabel, ...]:
    """Fail closed unless one exact AMI label surface matches the corpus."""

    try:
        if (
            type(corpus) is not LoadedCorpus
            or type(binding) is not AmiEndpointProxyBinding
            or corpus.schema_version != 2
            or corpus.provenance is None
            or corpus.provenance.kind != "public-voice-v1"
            or corpus.provenance.suite != "conversation96-ami"
            or not _valid_sha256(corpus.digest)
            or not _valid_sha256(binding.corpus_manifest_sha256)
            or corpus.digest != binding.corpus_manifest_sha256
            or not _valid_sha256(binding.receipt_sha256)
            or corpus.provenance.source_set_sha256 != binding.receipt_sha256
            or not _valid_sha256(binding.sidecar_sha256)
            or corpus.provenance.metadata_sha256 != binding.sidecar_sha256
            or not _valid_sha256(binding.annotation_metadata_sha256)
            or not _valid_sha256(binding.label_set_sha256)
        ):
            _fail()
        labels = _validated_labels(binding.labels)
        if (
            len(corpus.cases) != _EXPECTED_CASES
            or len(labels) != _EXPECTED_CASES
            or canonical_label_set_sha256(labels) != binding.label_set_sha256
            or canonical_sidecar_sha256(
                annotation_metadata_sha256=(
                    binding.annotation_metadata_sha256
                ),
                label_set_sha256=binding.label_set_sha256,
                labels=labels,
            )
            != binding.sidecar_sha256
        ):
            _fail()

        isolated = 0
        transitions = 0
        for index, (case, label) in enumerate(
            zip(corpus.cases, labels, strict=True)
        ):
            if (
                case.case_id != label.case_id
                or case.sha256 != label.pcm_sha256
                or case.samples != label.samples
                or label.window_kind not in case.tags
                or label.channel not in case.tags
            ):
                _fail()
            if label.window_kind == "isolated":
                isolated += 1
            else:
                transitions += 1

            pair_offset = index % 2
            if (pair_offset == 0 and label.channel != "close") or (
                pair_offset == 1 and label.channel != "far"
            ):
                _fail()

        for index in range(0, _EXPECTED_CASES, 2):
            close = labels[index]
            far = labels[index + 1]
            close_suffix = "-close"
            far_suffix = "-far"
            if (
                not close.case_id.endswith(close_suffix)
                or not far.case_id.endswith(far_suffix)
                or close.case_id[: -len(close_suffix)]
                != far.case_id[: -len(far_suffix)]
                or close.samples != far.samples
                or close.forced_aligned_last_word_end_sample
                != far.forced_aligned_last_word_end_sample
                or close.window_kind != far.window_kind
            ):
                _fail()
        if (
            isolated != _EXPECTED_ISOLATED_CASES
            or transitions != _EXPECTED_TRANSITION_CASES
        ):
            _fail()
        return labels
    except AmiEndpointProxyError:
        raise
    except (AttributeError, OverflowError, TypeError, ValueError):
        raise AmiEndpointProxyError(
            "invalid AMI endpoint proxy evidence"
        ) from None


def _nearest_rank(values: Sequence[int], quantile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def _sample_summary(values: Sequence[int]) -> dict[str, int | None]:
    return {
        "p50": _nearest_rank(values, 0.50),
        "p95": _nearest_rank(values, 0.95),
        "max": max(values) if values else None,
    }


def _milliseconds_summary(
    values: Sequence[int],
) -> dict[str, float | None]:
    samples = _sample_summary(values)
    return {
        key: None if value is None else round(value * 1000.0 / _SAMPLE_RATE_HZ, 3)
        for key, value in samples.items()
    }


def _record_sequence(
    corpus: LoadedCorpus,
    records: Sequence[RunRecord],
    *,
    repeats: int,
) -> tuple[RunRecord, ...]:
    if type(repeats) is not int or not 1 <= repeats <= _MAX_REPEATS:
        _fail()
    if type(records) not in {list, tuple}:
        _fail()
    expected = len(corpus.cases) * repeats
    if len(records) != expected:
        _fail()
    result: list[RunRecord] = []
    for position, record in enumerate(records):
        expected_repeat, expected_case = divmod(position, len(corpus.cases))
        if (
            type(record) is not RunRecord
            or record.repeat != expected_repeat
            or record.case_index != expected_case
            or type(record.trace) is not CaseTrace
            or type(record.trace.partials) is not tuple
            or any(type(item) is not PartialEvent for item in record.trace.partials)
        ):
            _fail()
        result.append(record)
    return tuple(result)


def aggregate_ami_endpoint_proxy(
    corpus: LoadedCorpus,
    binding: AmiEndpointProxyBinding,
    records: Sequence[RunRecord],
    ready: ReadyEvent,
    *,
    repeats: int,
) -> dict[str, object]:
    """Reduce isolated AMI proxy timing without exposing row-level evidence."""

    try:
        labels = validate_ami_endpoint_proxy_binding(corpus, binding)
        ordered_records = _record_sequence(corpus, records, repeats=repeats)
        if type(ready) is not ReadyEvent:
            _fail()
        if ready.adapter == PARAKEET_REALTIME_EOU_ADAPTER:
            if ready.protocol_version != NATIVE_ENDPOINT_PROTOCOL_VERSION:
                _fail()
            final_type: type[NativeFinalEvent] | type[ParakeetCppFinalEvent] = (
                NativeFinalEvent
            )
            endpoint_field = "endpoint_sample"
            sample_definition = (
                "source_consumed_plus_tail_consumed_plus_model_padding"
            )
            selection = "native_terminal_eou"
        elif ready.adapter == PARAKEET_CPP_ADAPTER:
            if ready.protocol_version != PARAKEET_CPP_ENDPOINT_PROTOCOL_VERSION:
                _fail()
            final_type = ParakeetCppFinalEvent
            endpoint_field = "observed_sample"
            sample_definition = "source_pcm_plus_fed_tail_samples"
            selection = "first_feed_eou"
        else:
            _fail()

        if any(type(record.trace.final) is not final_type for record in ordered_records):
            _fail()
        validation_metrics = aggregate_metrics(
            corpus,
            ordered_records,
            ready,
            repeats=repeats,
        )
        if validation_metrics.get("coverage_complete") is not True:
            _fail()

        deltas: list[int] = []
        feed_eou_observations = 0
        non_feed_eou_observations = 0
        evaluations_with_non_feed_eou = 0
        missing_eou_evaluations = 0
        missing_feed_eou_evaluations = 0
        non_feed_only_eou_evaluations = 0
        tail_exhaustion_evaluations = 0

        for record in ordered_records:
            label = labels[record.case_index]
            if label.window_kind != "isolated":
                continue
            final = record.trace.final
            endpoint_sample: int | None
            has_any_eou = False
            has_non_feed_eou = False
            if type(final) is NativeFinalEvent:
                has_any_eou = final.endpoint_reason == "eou"
                endpoint_sample = (
                    final.endpoint_sample if has_any_eou else None
                )
                if final.endpoint_reason == "tail_exhausted":
                    tail_exhaustion_evaluations += 1
                if has_any_eou:
                    feed_eou_observations += 1
            elif type(final) is ParakeetCppFinalEvent:
                eou_events = tuple(
                    event
                    for event in final.observed_events
                    if event.event_type == "eou"
                )
                feed_events = tuple(
                    event for event in eou_events if event.event_origin == "feed"
                )
                non_feed_events = tuple(
                    event
                    for event in eou_events
                    if event.event_origin != "feed"
                )
                has_any_eou = bool(eou_events)
                has_non_feed_eou = bool(non_feed_events)
                feed_eou_observations += len(feed_events)
                non_feed_eou_observations += len(non_feed_events)
                endpoint_sample = (
                    feed_events[0].observed_sample if feed_events else None
                )
                if final.endpoint_reason == "tail_exhausted":
                    tail_exhaustion_evaluations += 1
            else:  # Defends this reducer if validation is ever refactored.
                _fail()

            if has_non_feed_eou:
                evaluations_with_non_feed_eou += 1
            if not has_any_eou:
                missing_eou_evaluations += 1
            if endpoint_sample is None:
                missing_feed_eou_evaluations += 1
                if has_any_eou:
                    non_feed_only_eou_evaluations += 1
                continue
            if type(endpoint_sample) is not int or endpoint_sample <= 0:
                _fail()
            deltas.append(
                endpoint_sample
                - label.forced_aligned_last_word_end_sample
            )

        scored_evaluations = _EXPECTED_ISOLATED_CASES * repeats
        premature = tuple(value for value in deltas if value < 0)
        at_or_after = tuple(value for value in deltas if value >= 0)
        early = tuple(-value for value in premature)
        absolute = tuple(abs(value) for value in deltas)
        if (
            len(deltas) + missing_feed_eou_evaluations != scored_evaluations
            or len(premature) + len(at_or_after) != len(deltas)
        ):
            _fail()

        sample_groups = {
            "signed": tuple(deltas),
            "early": early,
            "post_proxy": at_or_after,
            "absolute": absolute,
        }
        return {
            "schema_version": AMI_ENDPOINT_PROXY_SCHEMA_VERSION,
            "kind": AMI_ENDPOINT_PROXY_KIND,
            "ground_truth": False,
            "binding": {
                "corpus_manifest_sha256": binding.corpus_manifest_sha256,
                "receipt_sha256": binding.receipt_sha256,
                "sidecar_sha256": binding.sidecar_sha256,
                "annotation_metadata_sha256": (
                    binding.annotation_metadata_sha256
                ),
                "label_set_sha256": binding.label_set_sha256,
            },
            "scope": "isolated_nonoverlap_diagnostic",
            "sample_domain": {
                "adapter": ready.adapter,
                "sample_rate_hz": _SAMPLE_RATE_HZ,
                "endpoint_field": endpoint_field,
                "definition": sample_definition,
                "selection": selection,
            },
            "denominators": {
                "corpus_cases": _EXPECTED_CASES,
                "isolated_cases": _EXPECTED_ISOLATED_CASES,
                "excluded_turn_transition_cases": _EXPECTED_TRANSITION_CASES,
                "repeats": repeats,
                "all_evaluations": _EXPECTED_CASES * repeats,
                "scored_evaluations": scored_evaluations,
            },
            "endpoint_observations": {
                "feed_eou_observations": feed_eou_observations,
                "non_feed_eou_observations": non_feed_eou_observations,
                "evaluations_with_non_feed_eou": (
                    evaluations_with_non_feed_eou
                ),
                "missing_eou_evaluations": missing_eou_evaluations,
                "missing_feed_eou_evaluations": (
                    missing_feed_eou_evaluations
                ),
                "non_feed_only_eou_evaluations": (
                    non_feed_only_eou_evaluations
                ),
                "tail_exhaustion_evaluations": tail_exhaustion_evaluations,
                "premature_feed_eou_evaluations": len(premature),
                "at_or_after_proxy_feed_eou_evaluations": len(at_or_after),
            },
            "sample_deltas": {
                name: _sample_summary(values)
                for name, values in sample_groups.items()
            },
            "milliseconds": {
                name: _milliseconds_summary(values)
                for name, values in sample_groups.items()
            },
        }
    except AmiEndpointProxyError:
        raise
    except (AttributeError, KeyError, OverflowError, TypeError, ValueError):
        raise AmiEndpointProxyError(
            "invalid AMI endpoint proxy evidence"
        ) from None
