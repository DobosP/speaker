"""AMI-only forced-alignment endpoint-proxy evaluation envelope.

The generic schema-v2 corpus stays unchanged.  This entry point first validates
the existing public-conversation AMI corpus and receipt, then loads their fixed
private sidecar and passes only a typed in-memory binding to the isolated STT
evaluator.  It revalidates every source after inference and publishes only
aggregate, explicitly non-authoritative diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
import stat

from tools import public_conversation_fixture as fixture
from tools import streaming_stt_eval
from tools.prepare_ami_conversation_fixture import (
    ENDPOINT_PROXY_CONTRACT,
    ENDPOINT_PROXY_LABELS_FILENAME,
    ENDPOINT_PROXY_LABELS_KIND,
    SOURCE_ID,
)
from tools.streaming_stt.ami_endpoint_proxy import (
    AMI_ENDPOINT_PROXY_FIXTURE_ID,
    AMI_ENDPOINT_PROXY_KIND,
    AMI_ENDPOINT_PROXY_LABEL_CONTRACT,
    AMI_ENDPOINT_PROXY_SCHEMA_VERSION,
    AMI_ENDPOINT_PROXY_SIDECAR_KIND,
    AMI_ENDPOINT_PROXY_SOURCE_ID,
    AmiEndpointProxyBinding,
    AmiEndpointProxyLabel,
    canonical_sidecar_sha256,
    validate_ami_endpoint_proxy_binding,
)
from tools.streaming_stt.bounded_io import (
    opened_directory_nofollow,
    read_regular_bounded,
)
from tools.streaming_stt.corpus import LoadedCorpus, load_corpus
from tools.streaming_stt.manifest import (
    PARAKEET_CPP_ADAPTER,
    PARAKEET_REALTIME_EOU_ADAPTER,
    load_worker_manifest,
)


SCHEMA_VERSION = 1
KIND = "ami-endpoint-proxy-evaluation-v1"
_CORPUS_SUITE = "conversation96-ami"
_SAMPLE_RATE_HZ = 16_000
_MAX_LABEL_BYTES = 64 * 1024
_MAX_IMPLEMENTATION_BYTES = 512 * 1024
_MAX_REPORT_BYTES = 4 * 1024 * 1024
_MAX_PATH_CHARS = 4096
_SHA256_CHARS = frozenset("0123456789abcdef")
_SAFE_ERROR = {"error": "ami_endpoint_proxy_evaluation_unavailable", "ok": False}
_PRODUCTION_RUNNER = streaming_stt_eval.run_benchmark
_LABEL_FIELDS = {
    "case_id",
    "pcm_sha256",
    "samples",
    "forced_aligned_last_word_end_sample",
    "window_kind",
    "channel",
}
_ROOT_FIELDS = {
    "schema_version",
    "kind",
    "fixture_id",
    "source_id",
    "corpus_suite",
    "sample_rate_hz",
    "label_contract",
    "scoreable_scope",
    "annotation_metadata_sha256",
    "label_set_sha256",
    "labels",
    "privacy",
    "evidence_scope",
}
_PRIVACY = {
    "contains_audio": False,
    "contains_transcripts": False,
    "contains_local_paths": False,
    "contains_raw_speaker_identifiers": False,
}
_EVIDENCE_SCOPE = {
    "after_pcm": True,
    "forced_alignment_proxy": True,
    "acoustic_ground_truth": False,
    "capture": False,
    "vad": False,
    "device": False,
    "live_latency": False,
    "default_promotion": False,
}


class AmiEndpointProxyEvaluationError(RuntimeError):
    """A detail-free source, label, execution, or publication failure."""


@dataclass(frozen=True, slots=True)
class LoadedAmiEndpointProxy:
    corpus: LoadedCorpus = field(repr=False)
    binding: AmiEndpointProxyBinding
    sidecar_sha256: str
    annotation_metadata_sha256: str
    sidecar_path: Path = field(repr=False)


def _bad() -> object:
    raise AmiEndpointProxyEvaluationError()


def _sha256(value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_CHARS for character in value)
    ):
        raise AmiEndpointProxyEvaluationError()
    return value


def _strict_json(raw: bytes) -> object:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise AmiEndpointProxyEvaluationError()
            result[key] = value
        return result

    try:
        return json.loads(
            raw.decode("ascii"),
            object_pairs_hook=pairs,
            parse_constant=lambda _value: _bad(),
        )
    except (
        AmiEndpointProxyEvaluationError,
        UnicodeError,
        ValueError,
        OverflowError,
        RecursionError,
    ):
        raise AmiEndpointProxyEvaluationError() from None


def _directory_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        stat.S_IFMT(value.st_mode),
        stat.S_IMODE(value.st_mode),
        value.st_uid,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _file_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        stat.S_IFMT(value.st_mode),
        stat.S_IMODE(value.st_mode),
        value.st_uid,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _private_directory(value: os.stat_result) -> bool:
    return (
        stat.S_ISDIR(value.st_mode)
        and stat.S_IMODE(value.st_mode) & 0o077 == 0
        and (
            not hasattr(os, "geteuid")
            or value.st_uid == os.geteuid()
        )
    )


def _private_file(value: os.stat_result, *, maximum_bytes: int) -> bool:
    return (
        stat.S_ISREG(value.st_mode)
        and value.st_nlink == 1
        and stat.S_IMODE(value.st_mode) == 0o600
        and 0 < value.st_size <= maximum_bytes
        and (
            not hasattr(os, "geteuid")
            or value.st_uid == os.geteuid()
        )
    )


def _read_private_regular_at(
    parent: Path,
    directory_fd: int,
    name: str,
    *,
    maximum_bytes: int,
) -> tuple[bytes, Path]:
    descriptor = -1
    try:
        parent_before = os.fstat(directory_fd)
        if not _private_directory(parent_before):
            raise AmiEndpointProxyEvaluationError()
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if not _private_file(before, maximum_bytes=maximum_bytes):
            raise AmiEndpointProxyEvaluationError()
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(name, flags, dir_fd=directory_fd)
        opened = os.fstat(descriptor)
        if _file_identity(opened) != _file_identity(before):
            raise AmiEndpointProxyEvaluationError()
        data = os.read(descriptor, maximum_bytes + 1)
        if len(data) != opened.st_size or len(data) > maximum_bytes:
            raise AmiEndpointProxyEvaluationError()
        after = os.fstat(descriptor)
        current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        parent_after = os.fstat(directory_fd)
        candidate = parent / name
        resolved = candidate.resolve(strict=True)
        if (
            _file_identity(after) != _file_identity(opened)
            or _file_identity(current) != _file_identity(opened)
            or resolved != candidate
            or _file_identity(resolved.stat()) != _file_identity(opened)
            or _directory_identity(parent_after)
            != _directory_identity(parent_before)
            or _directory_identity(parent.lstat())
            != _directory_identity(parent_before)
            or not _private_directory(parent_after)
            or parent.resolve(strict=True) != parent
        ):
            raise AmiEndpointProxyEvaluationError()
        return data, resolved
    except AmiEndpointProxyEvaluationError:
        raise
    except Exception:
        raise AmiEndpointProxyEvaluationError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _typed_json_equal(value: object, expected: object) -> bool:
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(value) == set(expected) and all(  # type: ignore[arg-type]
            _typed_json_equal(value[key], item)  # type: ignore[index]
            for key, item in expected.items()
        )
    if isinstance(expected, list):
        return len(value) == len(expected) and all(  # type: ignore[arg-type]
            _typed_json_equal(left, right)
            for left, right in zip(value, expected, strict=True)  # type: ignore[arg-type]
        )
    return value == expected


def _labels_from_sidecar(value: object) -> tuple[AmiEndpointProxyLabel, ...]:
    if not isinstance(value, list) or len(value) != 24:
        raise AmiEndpointProxyEvaluationError()
    labels: list[AmiEndpointProxyLabel] = []
    for row in value:
        if not isinstance(row, dict) or set(row) != _LABEL_FIELDS:
            raise AmiEndpointProxyEvaluationError()
        try:
            labels.append(AmiEndpointProxyLabel(**row))
        except (TypeError, ValueError, OverflowError):
            raise AmiEndpointProxyEvaluationError() from None
    return tuple(labels)


def _load_sidecar(
    corpus: LoadedCorpus,
    *,
    receipt_sha256: str,
) -> LoadedAmiEndpointProxy:
    try:
        if (
            type(corpus) is not LoadedCorpus
            or corpus.schema_version != 2
            or corpus.provenance is None
            or corpus.provenance.kind != "public-voice-v1"
            or corpus.provenance.suite != _CORPUS_SUITE
            or corpus.provenance.source_set_sha256 != _sha256(receipt_sha256)
        ):
            raise AmiEndpointProxyEvaluationError()
        parent = corpus.path.parent
        with opened_directory_nofollow(parent, require_private=True) as (
            bound_parent,
            directory_fd,
        ):
            if bound_parent != parent:
                raise AmiEndpointProxyEvaluationError()
            sidecar_raw, stable_sidecar_path = _read_private_regular_at(
                parent,
                directory_fd,
                ENDPOINT_PROXY_LABELS_FILENAME,
                maximum_bytes=_MAX_LABEL_BYTES,
            )
        sidecar_sha256 = hashlib.sha256(sidecar_raw).hexdigest()
        if sidecar_sha256 != corpus.provenance.metadata_sha256:
            raise AmiEndpointProxyEvaluationError()
        root = _strict_json(sidecar_raw)
        if (
            not isinstance(root, dict)
            or set(root) != _ROOT_FIELDS
            or not _typed_json_equal(root.get("schema_version"), 1)
            or not _typed_json_equal(
                root.get("kind"),
                ENDPOINT_PROXY_LABELS_KIND,
            )
            or not _typed_json_equal(root.get("fixture_id"), fixture.FIXTURE_ID)
            or not _typed_json_equal(root.get("source_id"), SOURCE_ID)
            or not _typed_json_equal(root.get("corpus_suite"), _CORPUS_SUITE)
            or not _typed_json_equal(root.get("sample_rate_hz"), _SAMPLE_RATE_HZ)
            or not _typed_json_equal(
                root.get("label_contract"),
                ENDPOINT_PROXY_CONTRACT,
            )
            or not _typed_json_equal(
                root.get("scoreable_scope"),
                "isolated_nonoverlap_only",
            )
            or not _typed_json_equal(root.get("privacy"), _PRIVACY)
            or not _typed_json_equal(
                root.get("evidence_scope"),
                _EVIDENCE_SCOPE,
            )
            or ENDPOINT_PROXY_LABELS_KIND != AMI_ENDPOINT_PROXY_SIDECAR_KIND
            or ENDPOINT_PROXY_CONTRACT != AMI_ENDPOINT_PROXY_LABEL_CONTRACT
            or fixture.FIXTURE_ID != AMI_ENDPOINT_PROXY_FIXTURE_ID
            or SOURCE_ID != AMI_ENDPOINT_PROXY_SOURCE_ID
        ):
            raise AmiEndpointProxyEvaluationError()
        labels = _labels_from_sidecar(root.get("labels"))
        annotation_metadata_sha256 = _sha256(
            root.get("annotation_metadata_sha256")
        )
        label_set_sha256 = _sha256(root.get("label_set_sha256"))
        binding = AmiEndpointProxyBinding(
            corpus_manifest_sha256=corpus.digest,
            receipt_sha256=receipt_sha256,
            sidecar_sha256=sidecar_sha256,
            annotation_metadata_sha256=annotation_metadata_sha256,
            label_set_sha256=label_set_sha256,
            labels=labels,
        )
        validate_ami_endpoint_proxy_binding(corpus, binding)
        if (
            canonical_sidecar_sha256(
                annotation_metadata_sha256=annotation_metadata_sha256,
                label_set_sha256=label_set_sha256,
                labels=labels,
            )
            != sidecar_sha256
        ):
            raise AmiEndpointProxyEvaluationError()
        return LoadedAmiEndpointProxy(
            corpus=corpus,
            binding=binding,
            sidecar_sha256=sidecar_sha256,
            annotation_metadata_sha256=annotation_metadata_sha256,
            sidecar_path=stable_sidecar_path,
        )
    except AmiEndpointProxyEvaluationError:
        raise
    except Exception:
        raise AmiEndpointProxyEvaluationError() from None


def load_ami_endpoint_proxy(
    corpus_path: Path | str,
    *,
    lock_path: Path | str = fixture.DEFAULT_LOCK,
) -> LoadedAmiEndpointProxy:
    """Validate the official AMI fixture and its fixed private label sidecar."""

    try:
        lock = fixture.load_fixture_lock(lock_path)
        source = fixture.validate_private_source(
            corpus_path,
            lock=lock,
            expected_source_id=SOURCE_ID,
        )
        corpus = load_corpus(source.corpus_path)
        if (
            corpus.digest != source.corpus_manifest_sha256
            or corpus.path != source.corpus_path
            or source.receipt_path.parent != corpus.path.parent
        ):
            raise AmiEndpointProxyEvaluationError()
        return _load_sidecar(corpus, receipt_sha256=source.receipt_sha256)
    except AmiEndpointProxyEvaluationError:
        raise
    except Exception:
        raise AmiEndpointProxyEvaluationError() from None


def _implementation_sha256() -> str:
    try:
        path = Path(__file__).resolve(strict=True)
        snapshot = read_regular_bounded(
            path,
            maximum_bytes=_MAX_IMPLEMENTATION_BYTES,
        )
        return hashlib.sha256(snapshot.data).hexdigest()
    except Exception:
        raise AmiEndpointProxyEvaluationError() from None


def _is_absolute_path_text(value: str) -> bool:
    return (
        value.startswith(("/", "\\", "file:"))
        or PureWindowsPath(value).is_absolute()
    )


def _assert_path_free(value: object) -> None:
    if value is None or isinstance(value, (bool, int, float)):
        return
    if isinstance(value, str):
        if len(value) > _MAX_PATH_CHARS or _is_absolute_path_text(value):
            raise AmiEndpointProxyEvaluationError()
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise AmiEndpointProxyEvaluationError()
            _assert_path_free(key)
            _assert_path_free(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _assert_path_free(item)
        return
    raise AmiEndpointProxyEvaluationError()


def _canonical_digest(value: object) -> str:
    try:
        raw = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        return hashlib.sha256(raw).hexdigest()
    except Exception:
        raise AmiEndpointProxyEvaluationError() from None


def _plain_json(value: object) -> object:
    try:
        raw = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        return _strict_json(raw)
    except AmiEndpointProxyEvaluationError:
        raise
    except Exception:
        raise AmiEndpointProxyEvaluationError() from None


def _count(value: object) -> int:
    if type(value) is not int or value < 0:
        raise AmiEndpointProxyEvaluationError()
    return value


def _number_or_none(value: object, *, nonnegative: bool = True) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AmiEndpointProxyEvaluationError()
    try:
        selected = float(value)
    except (OverflowError, ValueError):
        raise AmiEndpointProxyEvaluationError() from None
    if not math.isfinite(selected) or (nonnegative and selected < 0):
        raise AmiEndpointProxyEvaluationError()
    return value


def _accuracy_snapshot(value: object, *, evaluations: int) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise AmiEndpointProxyEvaluationError()
    transcript = value.get("transcript")
    fields = {
        "clips",
        "nonempty",
        "exact",
        "wer",
        "cer",
        "word_errors",
        "substitutions",
        "insertions",
        "deletions",
        "ref_words",
        "hyp_words",
        "char_edits",
        "ref_chars",
        "hyp_chars",
        "keyword_attempts",
        "keyword_hits",
    }
    if not isinstance(transcript, Mapping) or set(transcript) != fields:
        raise AmiEndpointProxyEvaluationError()
    counts = {
        key: _count(transcript.get(key))
        for key in fields - {"wer", "cer"}
    }
    wer = _number_or_none(transcript.get("wer"))
    cer = _number_or_none(transcript.get("cer"))
    if (
        wer is None
        or cer is None
        or counts["clips"] != evaluations
        or counts["nonempty"] > evaluations
        or counts["exact"] > evaluations
        or counts["word_errors"]
        != counts["substitutions"]
        + counts["insertions"]
        + counts["deletions"]
        or counts["keyword_hits"] > counts["keyword_attempts"]
    ):
        raise AmiEndpointProxyEvaluationError()
    snapshot = _plain_json(dict(transcript))
    if not isinstance(snapshot, dict):
        raise AmiEndpointProxyEvaluationError()
    return snapshot


def _sample_totals(value: object, *, fields: set[str]) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise AmiEndpointProxyEvaluationError()
    return {key: _count(value.get(key)) for key in fields}


def _endpointing_snapshot(
    value: object,
    *,
    adapter: str,
    expected_evaluations: int | None = None,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise AmiEndpointProxyEvaluationError()
    if adapter == PARAKEET_REALTIME_EOU_ADAPTER:
        count_fields = {
            "evaluations",
            "native_endpoints",
            "native_eou_events",
            "native_eob_backchannels",
            "authoritative_endpoints",
            "early_source_endpoints",
            "source_early_native_events",
            "source_early_eou_events",
            "source_early_eob_backchannels",
            "tail_exhaustions",
            "terminal_model_padding_samples_total",
            "model_input_samples_consumed_total",
        }
        optional_number_fields = {
            "endpoint_sample_p50",
            "endpoint_sample_p95",
            "authoritative_endpoint_latency_p50_ms",
            "authoritative_endpoint_latency_p95_ms",
            "authoritative_endpoint_latency_max_ms",
            "endpoint_probability_p50",
            "endpoint_probability_p95",
        }
        expected = count_fields | optional_number_fields | {
            "reasons",
            "source_samples",
            "declared_tail_samples",
        }
        if set(value) != expected:
            raise AmiEndpointProxyEvaluationError()
        counts = {key: _count(value.get(key)) for key in count_fields}
        reasons = _sample_totals(
            value.get("reasons"),
            fields={"eou", "eob", "tail_exhausted"},
        )
        source = _sample_totals(
            value.get("source_samples"),
            fields={"offered_total", "consumed_total", "dropped_total"},
        )
        tail = _sample_totals(
            value.get("declared_tail_samples"),
            fields={"offered_total", "consumed_total", "unconsumed_total"},
        )
        for key in optional_number_fields:
            _number_or_none(value.get(key))
        if (
            (
                expected_evaluations is not None
                and counts["evaluations"] != expected_evaluations
            )
            or counts["native_eou_events"] != reasons["eou"]
            or counts["native_eob_backchannels"] != reasons["eob"]
            or reasons["eou"] + reasons["eob"]
            != counts["native_endpoints"]
            or sum(reasons.values()) != counts["evaluations"]
            or counts["tail_exhaustions"] != reasons["tail_exhausted"]
            or counts["authoritative_endpoints"] > reasons["eou"]
            or source["offered_total"]
            != source["consumed_total"] + source["dropped_total"]
            or tail["offered_total"]
            != tail["consumed_total"] + tail["unconsumed_total"]
        ):
            raise AmiEndpointProxyEvaluationError()
    elif adapter == PARAKEET_CPP_ADAPTER:
        count_fields = {
            "evaluations",
            "accepted_endpoints",
            "tail_exhaustions",
            "terminal_model_padding_samples_total",
            "model_input_samples_consumed_total",
        }
        optional_number_fields = {
            "accepted_endpoint_sample_p50",
            "accepted_endpoint_sample_p95",
            "observed_encoder_frame_p50",
            "observed_encoder_frame_p95",
            "observed_encoder_time_p50_seconds",
            "observed_encoder_time_p95_seconds",
            "accepted_endpoint_latency_p50_ms",
            "accepted_endpoint_latency_p95_ms",
            "accepted_endpoint_latency_max_ms",
        }
        expected = count_fields | optional_number_fields | {
            "terminal_reasons",
            "terminal_origins",
            "observations",
            "source_samples",
            "tail_samples",
        }
        if set(value) != expected:
            raise AmiEndpointProxyEvaluationError()
        counts = {key: _count(value.get(key)) for key in count_fields}
        reasons = _sample_totals(
            value.get("terminal_reasons"),
            fields={"eou", "tail_exhausted"},
        )
        origins = _sample_totals(
            value.get("terminal_origins"),
            fields={"feed", "none"},
        )
        observations = value.get("observations")
        if not isinstance(observations, Mapping) or set(observations) != {
            "total",
            "eou",
            "eob",
            "origins",
            "source_early",
            "post_source_feed",
        }:
            raise AmiEndpointProxyEvaluationError()
        observation_counts = {
            key: _count(observations.get(key))
            for key in {"total", "eou", "eob"}
        }
        observation_origins = _sample_totals(
            observations.get("origins"),
            fields={"feed", "finalize"},
        )
        source_early = _sample_totals(
            observations.get("source_early"),
            fields={"total", "eou", "eob"},
        )
        post_source_feed = _sample_totals(
            observations.get("post_source_feed"),
            fields={"total", "eou", "eob"},
        )
        source = _sample_totals(
            value.get("source_samples"),
            fields={"offered_total", "consumed_total", "dropped_total"},
        )
        tail = _sample_totals(
            value.get("tail_samples"),
            fields={"offered_total", "consumed_total", "unconsumed_total"},
        )
        for key in optional_number_fields:
            _number_or_none(value.get(key))
        if (
            (
                expected_evaluations is not None
                and counts["evaluations"] != expected_evaluations
            )
            or sum(reasons.values()) != counts["evaluations"]
            or sum(origins.values()) != counts["evaluations"]
            or counts["accepted_endpoints"] != reasons["eou"]
            or counts["accepted_endpoints"] != origins["feed"]
            or counts["tail_exhaustions"] != reasons["tail_exhausted"]
            or counts["tail_exhaustions"] != origins["none"]
            or observation_counts["eou"] + observation_counts["eob"]
            != observation_counts["total"]
            or sum(observation_origins.values()) != observation_counts["total"]
            or source_early["eou"] + source_early["eob"]
            != source_early["total"]
            or post_source_feed["eou"] + post_source_feed["eob"]
            != post_source_feed["total"]
            or source_early["total"] > observation_counts["total"]
            or post_source_feed["total"] > observation_counts["total"]
            or source["offered_total"]
            != source["consumed_total"] + source["dropped_total"]
            or tail["offered_total"]
            != tail["consumed_total"] + tail["unconsumed_total"]
        ):
            raise AmiEndpointProxyEvaluationError()
    else:
        raise AmiEndpointProxyEvaluationError()
    snapshot = _plain_json(dict(value))
    if not isinstance(snapshot, dict):
        raise AmiEndpointProxyEvaluationError()
    return snapshot


def _proxy_summary(
    value: object,
    *,
    allow_negative: bool,
    integer_only: bool = False,
    require_positive: bool = False,
) -> dict[str, int | float | None]:
    if not isinstance(value, Mapping) or set(value) != {"p50", "p95", "max"}:
        raise AmiEndpointProxyEvaluationError()
    selected = {
        key: _number_or_none(value.get(key), nonnegative=not allow_negative)
        for key in ("p50", "p95", "max")
    }
    if any(
        item is not None
        and (
            (integer_only and type(item) is not int)
            or (require_positive and item <= 0)
        )
        for item in selected.values()
    ):
        raise AmiEndpointProxyEvaluationError()
    present = tuple(
        selected[key]
        for key in ("p50", "p95", "max")
        if selected[key] is not None
    )
    if len(present) not in {0, 3} or (
        len(present) == 3
        and not (float(present[0]) <= float(present[1]) <= float(present[2]))
    ):
        raise AmiEndpointProxyEvaluationError()
    return selected


def _proxy_metrics_snapshot(
    value: object,
    *,
    binding: AmiEndpointProxyBinding,
    adapter: str,
    repeats: int,
) -> dict[str, object]:
    root_fields = {
        "schema_version",
        "kind",
        "ground_truth",
        "binding",
        "scope",
        "sample_domain",
        "denominators",
        "endpoint_observations",
        "sample_deltas",
        "milliseconds",
    }
    if not isinstance(value, Mapping) or set(value) != root_fields:
        raise AmiEndpointProxyEvaluationError()
    expected_binding = {
        "corpus_manifest_sha256": binding.corpus_manifest_sha256,
        "receipt_sha256": binding.receipt_sha256,
        "sidecar_sha256": binding.sidecar_sha256,
        "annotation_metadata_sha256": binding.annotation_metadata_sha256,
        "label_set_sha256": binding.label_set_sha256,
    }
    sample_domain = value.get("sample_domain")
    expected_domain = {
        "adapter": adapter,
        "sample_rate_hz": _SAMPLE_RATE_HZ,
        "endpoint_field": (
            "endpoint_sample"
            if adapter == PARAKEET_REALTIME_EOU_ADAPTER
            else "observed_sample"
        ),
        "definition": (
            "source_consumed_plus_tail_consumed_plus_model_padding"
            if adapter == PARAKEET_REALTIME_EOU_ADAPTER
            else "source_pcm_plus_fed_tail_samples"
        ),
        "selection": (
            "native_terminal_eou"
            if adapter == PARAKEET_REALTIME_EOU_ADAPTER
            else "first_feed_eou"
        ),
    }
    denominators = value.get("denominators")
    expected_denominators = {
        "corpus_cases": 24,
        "isolated_cases": 16,
        "excluded_turn_transition_cases": 8,
        "repeats": repeats,
        "all_evaluations": 24 * repeats,
        "scored_evaluations": 16 * repeats,
    }
    observations = value.get("endpoint_observations")
    observation_fields = {
        "feed_eou_observations",
        "non_feed_eou_observations",
        "evaluations_with_non_feed_eou",
        "missing_eou_evaluations",
        "missing_feed_eou_evaluations",
        "non_feed_only_eou_evaluations",
        "tail_exhaustion_evaluations",
        "premature_feed_eou_evaluations",
        "at_or_after_proxy_feed_eou_evaluations",
    }
    if (
        not _typed_json_equal(value.get("schema_version"), 1)
        or not _typed_json_equal(value.get("kind"), AMI_ENDPOINT_PROXY_KIND)
        or not _typed_json_equal(value.get("ground_truth"), False)
        or not _typed_json_equal(value.get("binding"), expected_binding)
        or not _typed_json_equal(
            value.get("scope"),
            "isolated_nonoverlap_diagnostic",
        )
        or not _typed_json_equal(sample_domain, expected_domain)
        or not _typed_json_equal(denominators, expected_denominators)
        or not isinstance(observations, Mapping)
        or set(observations) != observation_fields
    ):
        raise AmiEndpointProxyEvaluationError()
    counts = {key: _count(observations.get(key)) for key in observation_fields}
    scored = 16 * repeats
    observed = scored - counts["missing_feed_eou_evaluations"]
    if (
        counts["premature_feed_eou_evaluations"]
        + counts["at_or_after_proxy_feed_eou_evaluations"]
        != observed
        or counts["missing_feed_eou_evaluations"]
        != counts["missing_eou_evaluations"]
        + counts["non_feed_only_eou_evaluations"]
        or counts["missing_eou_evaluations"]
        > counts["missing_feed_eou_evaluations"]
        or counts["non_feed_only_eou_evaluations"]
        > counts["missing_feed_eou_evaluations"]
        or counts["non_feed_only_eou_evaluations"]
        > counts["evaluations_with_non_feed_eou"]
        or counts["evaluations_with_non_feed_eou"]
        > counts["non_feed_only_eou_evaluations"] + observed
        or counts["evaluations_with_non_feed_eou"] > scored
        or counts["non_feed_eou_observations"]
        < counts["evaluations_with_non_feed_eou"]
        or counts["feed_eou_observations"] < observed
        or counts["tail_exhaustion_evaluations"] > scored
        or (
            adapter == PARAKEET_REALTIME_EOU_ADAPTER
            and (
                counts["non_feed_eou_observations"] != 0
                or counts["evaluations_with_non_feed_eou"] != 0
                or counts["non_feed_only_eou_evaluations"] != 0
                or counts["feed_eou_observations"] != observed
            )
        )
    ):
        raise AmiEndpointProxyEvaluationError()
    samples = value.get("sample_deltas")
    milliseconds = value.get("milliseconds")
    groups = {"signed", "early", "post_proxy", "absolute"}
    if (
        not isinstance(samples, Mapping)
        or set(samples) != groups
        or not isinstance(milliseconds, Mapping)
        or set(milliseconds) != groups
    ):
        raise AmiEndpointProxyEvaluationError()
    sample_summaries = {
        name: _proxy_summary(
            samples.get(name),
            allow_negative=name == "signed",
            integer_only=True,
            require_positive=name == "early",
        )
        for name in groups
    }
    millisecond_summaries = {
        name: _proxy_summary(
            milliseconds.get(name),
            allow_negative=name == "signed",
        )
        for name in groups
    }
    expected_counts = {
        "signed": observed,
        "absolute": observed,
        "early": counts["premature_feed_eou_evaluations"],
        "post_proxy": counts["at_or_after_proxy_feed_eou_evaluations"],
    }
    for name in groups:
        sample_summary = sample_summaries[name]
        millisecond_summary = millisecond_summaries[name]
        if (expected_counts[name] == 0) != all(
            item is None for item in sample_summary.values()
        ):
            raise AmiEndpointProxyEvaluationError()
        for key in ("p50", "p95", "max"):
            sample = sample_summary[key]
            milliseconds_value = millisecond_summary[key]
            expected_ms = (
                None
                if sample is None
                else round(float(sample) * 1000.0 / _SAMPLE_RATE_HZ, 3)
            )
            if not _typed_json_equal(milliseconds_value, expected_ms):
                raise AmiEndpointProxyEvaluationError()
    snapshot = _plain_json(dict(value))
    if not isinstance(snapshot, dict):
        raise AmiEndpointProxyEvaluationError()
    return snapshot


def _generic_metrics_snapshot(
    value: object,
    *,
    binding: AmiEndpointProxyBinding,
    adapter: str,
    repeats: int,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise AmiEndpointProxyEvaluationError()
    evaluations = _count(value.get("evaluations"))
    coverage = value.get("coverage_complete")
    if evaluations != 24 * repeats or type(coverage) is not bool:
        raise AmiEndpointProxyEvaluationError()
    accuracy = _accuracy_snapshot(value.get("accuracy"), evaluations=evaluations)
    endpointing = _endpointing_snapshot(
        value.get("endpointing"),
        adapter=adapter,
        expected_evaluations=evaluations,
    )
    proxy = _proxy_metrics_snapshot(
        value.get("ami_endpoint_proxy"),
        binding=binding,
        adapter=adapter,
        repeats=repeats,
    )
    strata = value.get("strata")
    expected_strata = (("close", 12), ("far", 12), ("isolated", 16))
    if not isinstance(strata, list) or len(strata) != len(expected_strata):
        raise AmiEndpointProxyEvaluationError()
    safe_strata: list[dict[str, object]] = []
    for row, (tag, cases) in zip(strata, expected_strata, strict=True):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"tag", "cases", "metrics"}
            or not _typed_json_equal(row.get("tag"), tag)
            or not _typed_json_equal(row.get("cases"), cases)
        ):
            raise AmiEndpointProxyEvaluationError()
        nested = row.get("metrics")
        nested_evaluations = cases * repeats
        if (
            not isinstance(nested, Mapping)
            or _count(nested.get("evaluations")) != nested_evaluations
            or type(nested.get("coverage_complete")) is not bool
        ):
            raise AmiEndpointProxyEvaluationError()
        safe_strata.append(
            {
                "tag": tag,
                "cases": cases,
                "metrics": {
                    "evaluations": nested_evaluations,
                    "coverage_complete": nested["coverage_complete"],
                    "accuracy": {
                        "transcript": _accuracy_snapshot(
                            nested.get("accuracy"),
                            evaluations=nested_evaluations,
                        )
                    },
                    "endpointing": _endpointing_snapshot(
                        nested.get("endpointing"),
                        adapter=adapter,
                        expected_evaluations=nested_evaluations,
                    ),
                },
            }
        )
    return {
        "evaluations": evaluations,
        "coverage_complete": coverage,
        "accuracy": {"transcript": accuracy},
        "endpointing": endpointing,
        "strata": safe_strata,
        "ami_endpoint_proxy": proxy,
    }


def _config_snapshot(
    value: object,
    *,
    binding: AmiEndpointProxyBinding,
    repeats: int,
    chunk_samples: int | None,
    pace: str | None,
    partial_interval_ms: int | None,
    tail_padding_samples: int | None,
) -> dict[str, object]:
    fields = {
        "chunk_samples",
        "pace",
        "partial_interval_ms",
        "tail_padding_samples",
        "repeats",
        "stratum_tags",
        "ami_endpoint_proxy_label_set_sha256",
        "ami_endpoint_proxy_sidecar_sha256",
        "contract_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise AmiEndpointProxyEvaluationError()
    without_contract = {key: item for key, item in value.items() if key != "contract_sha256"}
    if (
        type(value.get("chunk_samples")) is not int
        or value["chunk_samples"] <= 0
        or type(value.get("pace")) is not str
        or value.get("pace") not in {"burst", "realtime"}
        or type(value.get("partial_interval_ms")) is not int
        or value["partial_interval_ms"] <= 0
        or type(value.get("tail_padding_samples")) is not int
        or value["tail_padding_samples"] < 0
        or not _typed_json_equal(value.get("repeats"), repeats)
        or not _typed_json_equal(
            value.get("stratum_tags"),
            ["close", "far", "isolated"],
        )
        or not _typed_json_equal(
            value.get("ami_endpoint_proxy_label_set_sha256"),
            binding.label_set_sha256,
        )
        or not _typed_json_equal(
            value.get("ami_endpoint_proxy_sidecar_sha256"),
            binding.sidecar_sha256,
        )
        or not _typed_json_equal(
            value.get("contract_sha256"),
            _canonical_digest(without_contract),
        )
        or (
            chunk_samples is not None
            and value.get("chunk_samples") != chunk_samples
        )
        or (pace is not None and value.get("pace") != pace)
        or (
            partial_interval_ms is not None
            and value.get("partial_interval_ms") != partial_interval_ms
        )
        or (
            tail_padding_samples is not None
            and value.get("tail_padding_samples") != tail_padding_samples
        )
    ):
        raise AmiEndpointProxyEvaluationError()
    snapshot = _plain_json(dict(value))
    if not isinstance(snapshot, dict):
        raise AmiEndpointProxyEvaluationError()
    return snapshot


def _validate_nested_report(
    value: object,
    *,
    binding: AmiEndpointProxyBinding,
    adapter: str,
    repeats: int,
    worker_manifest_sha256: str,
    chunk_samples: int | None = None,
    pace: str | None = None,
    partial_interval_ms: int | None = None,
    tail_padding_samples: int | None = None,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "ok",
        "evidence",
        "worker",
        "corpus",
        "config",
        "metrics",
        "worker_stderr",
        "evaluator",
    }:
        raise AmiEndpointProxyEvaluationError()
    evidence = value.get("evidence")
    worker = value.get("worker")
    corpus = value.get("corpus")
    config = value.get("config")
    metrics = value.get("metrics")
    evaluator = value.get("evaluator")
    expected_corpus_proxy = {
        "receipt_sha256": binding.receipt_sha256,
        "sidecar_sha256": binding.sidecar_sha256,
        "annotation_metadata_sha256": binding.annotation_metadata_sha256,
        "label_set_sha256": binding.label_set_sha256,
    }
    expected_evidence = {
        "schema_version": AMI_ENDPOINT_PROXY_SCHEMA_VERSION,
        "kind": AMI_ENDPOINT_PROXY_KIND,
        "sidecar_sha256": binding.sidecar_sha256,
        "label_set_sha256": binding.label_set_sha256,
        "scoreable_scope": "isolated_nonoverlap_only",
        "ground_truth": False,
        "aggregate_only": True,
        "quality_thresholds": "none",
        "capture_included": False,
        "vad_included": False,
        "device_included": False,
        "live_latency": False,
        "promotional": False,
    }
    if (
        type(value.get("ok")) is not bool
        or not isinstance(evidence, Mapping)
        or not _typed_json_equal(
            evidence.get("ami_endpoint_proxy"),
            expected_evidence,
        )
        or not isinstance(worker, Mapping)
        or not _typed_json_equal(worker.get("adapter"), adapter)
        or not _typed_json_equal(
            worker.get("manifest_sha256"),
            worker_manifest_sha256,
        )
        or not isinstance(corpus, Mapping)
        or not _typed_json_equal(
            corpus.get("manifest_sha256"),
            binding.corpus_manifest_sha256,
        )
        or not _typed_json_equal(
            corpus.get("ami_endpoint_proxy"),
            expected_corpus_proxy,
        )
        or not _typed_json_equal(
            evaluator,
            streaming_stt_eval._evaluator_binding(adapter),
        )
    ):
        raise AmiEndpointProxyEvaluationError()
    safe_config = _config_snapshot(
        config,
        binding=binding,
        repeats=repeats,
        chunk_samples=chunk_samples,
        pace=pace,
        partial_interval_ms=partial_interval_ms,
        tail_padding_samples=tail_padding_samples,
    )
    safe_metrics = _generic_metrics_snapshot(
        metrics,
        binding=binding,
        adapter=adapter,
        repeats=repeats,
    )
    if value["ok"] is not safe_metrics["coverage_complete"]:
        raise AmiEndpointProxyEvaluationError()
    expected_evaluator = streaming_stt_eval._evaluator_binding(adapter)
    files = expected_evaluator.get("files")
    if not isinstance(files, Mapping):
        raise AmiEndpointProxyEvaluationError()
    result = {
        "ok": value["ok"],
        "adapter": adapter,
        "worker_manifest_sha256": worker_manifest_sha256,
        "evaluator": {
            "schema_version": expected_evaluator["schema_version"],
            "kind": expected_evaluator["kind"],
            "files": len(files),
            "file_set_sha256": _canonical_digest(files),
        },
        "corpus": {
            "manifest_sha256": binding.corpus_manifest_sha256,
            "ami_endpoint_proxy": expected_corpus_proxy,
        },
        "config": safe_config,
        "metrics": safe_metrics,
    }
    _assert_path_free(result)
    return result


def run_ami_endpoint_proxy_evaluation(
    *,
    worker_manifest: Path | str,
    corpus_path: Path | str,
    scratch_parent: Path | str,
    lock_path: Path | str = fixture.DEFAULT_LOCK,
    repeats: int = 3,
    chunk_samples: int | None = None,
    pace: str | None = None,
    partial_interval_ms: int | None = None,
    tail_padding_samples: int | None = None,
) -> dict[str, object]:
    """Run one AMI proxy diagnostic with validation on both sides of inference."""

    try:
        implementation_sha256 = _implementation_sha256()
        manifest_before = load_worker_manifest(worker_manifest)
        if manifest_before.adapter not in {
            PARAKEET_CPP_ADAPTER,
            PARAKEET_REALTIME_EOU_ADAPTER,
        }:
            raise AmiEndpointProxyEvaluationError()
        before = load_ami_endpoint_proxy(corpus_path, lock_path=lock_path)
    except AmiEndpointProxyEvaluationError:
        raise
    except Exception:
        raise AmiEndpointProxyEvaluationError() from None
    nested: object | None = None
    run_failed = False
    try:
        nested = _PRODUCTION_RUNNER(
            worker_manifest,
            before.corpus.path,
            scratch_parent=scratch_parent,
            repeats=repeats,
            chunk_samples=chunk_samples,
            pace=pace,
            partial_interval_ms=partial_interval_ms,
            tail_padding_samples=tail_padding_samples,
            stratum_tags=("isolated", "close", "far"),
            ami_endpoint_proxy=before.binding,
        )
    except BaseException:  # Revalidate even for unsafe runner exits.
        run_failed = True
    try:
        after = load_ami_endpoint_proxy(corpus_path, lock_path=lock_path)
        manifest_after = load_worker_manifest(worker_manifest)
        implementation_after = _implementation_sha256()
    except BaseException:
        raise AmiEndpointProxyEvaluationError() from None
    if (
        run_failed
        or nested is None
        or implementation_after != implementation_sha256
        or manifest_after.digest != manifest_before.digest
        or manifest_after.adapter != manifest_before.adapter
        or before.binding != after.binding
        or before.sidecar_sha256 != after.sidecar_sha256
        or before.annotation_metadata_sha256 != after.annotation_metadata_sha256
    ):
        raise AmiEndpointProxyEvaluationError()
    try:
        report = _validate_nested_report(
            nested,
            binding=before.binding,
            adapter=manifest_before.adapter,
            repeats=repeats,
            worker_manifest_sha256=manifest_before.digest,
            chunk_samples=chunk_samples,
            pace=pace,
            partial_interval_ms=partial_interval_ms,
            tail_padding_samples=tail_padding_samples,
        )
        result = {
            "ok": report["ok"],
            "schema_version": SCHEMA_VERSION,
            "kind": KIND,
            "execution_complete": True,
            "quality_decision": "not_evaluated",
            "development_only": True,
            "promotional": False,
            "binding": {
                "implementation_sha256": implementation_sha256,
                "corpus_manifest_sha256": (
                    before.binding.corpus_manifest_sha256
                ),
                "receipt_sha256": before.binding.receipt_sha256,
                "sidecar_sha256": before.sidecar_sha256,
                "annotation_metadata_sha256": (
                    before.annotation_metadata_sha256
                ),
                "label_set_sha256": before.binding.label_set_sha256,
            },
            "report": report,
        }
        _assert_path_free(result)
        return result
    except AmiEndpointProxyEvaluationError:
        raise
    except BaseException:
        raise AmiEndpointProxyEvaluationError() from None


def _write_new_private(path: Path | str, value: Mapping[str, object]) -> str:
    output = -1
    try:
        selected = Path(os.path.abspath(Path(path).expanduser()))
        payload = (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
            + b"\n"
        )
        if not payload or len(payload) > _MAX_REPORT_BYTES:
            raise AmiEndpointProxyEvaluationError()
        with opened_directory_nofollow(
            selected.parent,
            require_private=True,
        ) as (bound_parent, descriptor):
            parent_before = os.fstat(descriptor)
            if (
                bound_parent != selected.parent
                or not _private_directory(parent_before)
            ):
                raise AmiEndpointProxyEvaluationError()
            flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            output = os.open(selected.name, flags, 0o600, dir_fd=descriptor)
            parent_stable = os.fstat(descriptor)
            if (
                _directory_identity(parent_stable)[:3]
                != _directory_identity(parent_before)[:3]
                or not _private_directory(parent_stable)
            ):
                raise AmiEndpointProxyEvaluationError()
            os.fchmod(output, 0o600)
            written = 0
            while written < len(payload):
                count = os.write(output, payload[written:])
                if count <= 0:
                    raise AmiEndpointProxyEvaluationError()
                written += count
            os.fsync(output)
            opened = os.fstat(output)
            if not _private_file(opened, maximum_bytes=_MAX_REPORT_BYTES):
                raise AmiEndpointProxyEvaluationError()
            os.lseek(output, 0, os.SEEK_SET)
            observed = os.read(output, len(payload) + 1)
            after = os.fstat(output)
            current = os.stat(
                selected.name,
                dir_fd=descriptor,
                follow_symlinks=False,
            )
            parent_after = os.fstat(descriptor)
            if (
                observed != payload
                or _file_identity(after) != _file_identity(opened)
                or _file_identity(current) != _file_identity(opened)
                or _directory_identity(parent_after)
                != _directory_identity(parent_stable)
                or _directory_identity(selected.parent.lstat())
                != _directory_identity(parent_stable)
                or not _private_directory(parent_after)
                or selected.parent.resolve(strict=True) != selected.parent
            ):
                raise AmiEndpointProxyEvaluationError()
            os.fsync(descriptor)
        return hashlib.sha256(payload).hexdigest()
    except AmiEndpointProxyEvaluationError:
        raise
    except Exception:
        raise AmiEndpointProxyEvaluationError() from None
    finally:
        if output >= 0:
            try:
                os.close(output)
            except OSError:
                pass


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise AmiEndpointProxyEvaluationError()


def _positive_int(value: str) -> int:
    try:
        selected = int(value, 10)
    except ValueError:
        raise argparse.ArgumentTypeError("expected positive integer") from None
    if selected <= 0:
        raise argparse.ArgumentTypeError("expected positive integer")
    return selected


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        description=(
            "Run one aggregate-only AMI forced-alignment endpoint-proxy "
            "diagnostic without changing app/runtime defaults."
        )
    )
    parser.add_argument("--worker-manifest", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=fixture.DEFAULT_LOCK)
    parser.add_argument("--repeats", type=_positive_int, default=3)
    parser.add_argument("--chunk-samples", type=_positive_int)
    parser.add_argument("--pace", choices=("burst", "realtime"))
    parser.add_argument("--partial-interval-ms", type=_positive_int)
    parser.add_argument("--tail-padding-samples", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        report = run_ami_endpoint_proxy_evaluation(
            worker_manifest=args.worker_manifest,
            corpus_path=args.corpus,
            scratch_parent=args.scratch_root,
            lock_path=args.lock,
            repeats=args.repeats,
            chunk_samples=args.chunk_samples,
            pace=args.pace,
            partial_interval_ms=args.partial_interval_ms,
            tail_padding_samples=args.tail_padding_samples,
        )
        report_sha256 = _write_new_private(args.output, report)
        result: Mapping[str, object] = {
            "ok": report["ok"],
            "execution_complete": True,
            "kind": KIND,
            "quality_decision": "not_evaluated",
            "report_sha256": report_sha256,
            "corpus_sha256": report["binding"]["corpus_manifest_sha256"],
            "label_set_sha256": report["binding"]["label_set_sha256"],
        }
        code = 0 if report["ok"] is True else 1
    except Exception:  # noqa: BLE001 - never disclose private source details
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


__all__ = [
    "AmiEndpointProxyEvaluationError",
    "KIND",
    "LoadedAmiEndpointProxy",
    "load_ami_endpoint_proxy",
    "run_ami_endpoint_proxy_evaluation",
]
