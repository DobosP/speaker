"""Evaluate MobileWhisper revisions on exact mobile-Zipformer endpoint epochs.

Within this offline comparison, the current exact Zipformer rerun supplies the
endpoint-epoch boundaries.  A MobileWhisper pass receives one complete-PCM
request with no evaluator- or adapter-appended tail for each reset-committed
Zipformer epoch.  Candidate text is reduced only into separate aggregate
diagnostic cells; it is never admitted to the application agent.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from typing import Final, Mapping, Sequence

from core.wer import normalize
from tools import mobile_asr_evidence_eval as baseline_api
from tools.streaming_stt.bounded_io import (
    BoundedReadError,
    hash_regular_bounded,
    opened_directory_nofollow,
    read_regular_bounded,
)
from tools.streaming_stt.protocol import (
    MOBILE_ZIPFORMER_ENDPOINT_PROTOCOL_VERSION,
    PROTOCOL_VERSION as FINAL_PROTOCOL_VERSION,
    FinalEvent,
    MobileZipformerEndpointObservation,
    MobileZipformerFinalEvent,
    PcmInput,
    StreamConfig,
    TranscribeRequest,
)
from tools.streaming_stt.source_bundle import (
    WORKER_SOURCE_FILES,
    SourceBundle,
    verify_source_bundle,
)


SCHEMA_VERSION: Final = 1
REPORT_KIND: Final = "mobile-asr-two-pass-eval-report-v1"
SYNTHETIC_REPORT_KIND: Final = "synthetic-mobile-asr-two-pass-eval-report-v1"
BASELINE_MODEL_ID: Final = baseline_api.MODEL_ID
BASELINE_ADAPTER: Final = baseline_api.ADAPTER
CANDIDATE_MODEL_ID: Final = "sherpa-onnx-whisper-base.en-mobile-int8-v1"
CANDIDATE_ADAPTER: Final = "sherpa-onnx-mobile-whisper-final-v1"
CANDIDATE_SOURCE_REVISION: Final = "59eea950fc76df2453efb57e6c0fd334548e8ffe"
CANDIDATE_ARTIFACT_SET_SHA256: Final = (
    "3a3e907d65aced6f76a40afaabee86c12e7d281de3397bbfed33ad90f227af34"
)
CANDIDATE_MOBILE_CONFIG_SHA256: Final = (
    "f3555a3ffaaa2569bb12cb71c39f31883d27d759b35ec85024ca057fbc5e34f1"
)
CANDIDATE_MODEL_TOTAL_SIZE_BYTES: Final = 160_626_066
CANDIDATE_WORKER_LIFETIME_PCM_LIMIT_BYTES: Final = 62 * 1024 * 1024
ACCEPTED_BASELINE_REPORT_SHA256: Final = (
    "264f439652064ca78821abdcfed77d03a9a637fb82d6ed336b7a6112d083e796"
)
ACCEPTED_BASELINE_AGGREGATE_SHA256: Final = (
    "e485da1a658bb010c0edf03707d875f667e80a9a44b8572099239e07b74aa188"
)
EXPECTED_SOURCE_EVALUATIONS: Final = 129
EXPECTED_LOGICAL_CASES: Final = 123
EXPECTED_ENDPOINT_EPOCHS: Final = 198
EXPECTED_SOURCE_PCM_BYTES: Final = 39_299_500
EXPECTED_SOURCE_SAMPLES: Final = 9_824_875
MAXIMUM_DERIVED_EPOCH_PCM_BYTES: Final = (
    EXPECTED_SOURCE_PCM_BYTES + EXPECTED_SOURCE_EVALUATIONS * 48_000 * 4
)
MAXIMUM_DERIVED_EPOCH_SAMPLES: Final = MAXIMUM_DERIVED_EPOCH_PCM_BYTES // 4
_MAX_REPORT_BYTES: Final = 256 * 1024
_MAX_PUBLIC_OUTPUT_BYTES: Final = 4 * 1024
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_BASELINE_STREAM = baseline_api._STREAM
_CANDIDATE_STREAM = StreamConfig(
    chunk_samples=1_600,
    pace="burst",
    partial_interval_ms=100,
    tail_padding_samples=0,
)
_SAFE_ERROR = {
    "error": "mobile_asr_two_pass_eval_prerequisites_unavailable",
    "ok": False,
}
_METRIC_AGGREGATION = {
    "component_domains_are_independent": True,
    "cross_model_pooled_metric": None,
    "pooled_metric": None,
    "pooled_wer": False,
}
_CANDIDATE_AUTHORITY = {
    "agent_session_authority": False,
    "app_admission_authority": False,
    "control_authority": False,
    "endpoint_authority": False,
    "endpoint_epoch_boundary_authority": False,
    "semantic_turn_boundary_authority": False,
    "tool_authority": False,
}
_GENERAL_EVIDENCE_SCOPE = {
    "default_authority": False,
    "device_authority": False,
    "held_out_authority": False,
    "latency_authority": False,
    "live_authority": False,
    "microphone_authority": False,
    "model_quality_authority": False,
    "promotion_authority": False,
    "qualification_authority": False,
    "training_disjoint_authority": False,
}
_EXPECTED_BASELINE_ENDPOINTS = {
    "committed_endpoints": 198,
    "empty_committed_endpoints": 59,
    "empty_hypothesis_evaluations": 19,
    "endpoint_terminated_evaluations": 129,
    "evaluations": 129,
    "multiple_endpoint_evaluations": 40,
    "nonempty_committed_endpoints": 139,
    "nonempty_hypothesis_evaluations": 110,
    "pre_source_complete_endpoints": 69,
    "single_endpoint_evaluations": 89,
    "source_complete_endpoints": 129,
    "source_complete_evaluations": 129,
    "tail_exhausted_evaluations": 0,
    "zero_endpoint_evaluations": 0,
}
_EXPECTED_CANDIDATE_CONFIG = {
    "control_authority": False,
    "core_package_version": "1.13.3",
    "debug": True,
    "decoding_method": "greedy_search",
    "enable_segment_timestamps": False,
    "enable_token_timestamps": False,
    "endpoint_owner": "mobile-zipformer",
    "execution_mode": "complete-endpoint-epoch-final-only",
    "feature_dim": 80,
    "hr_dict_dir": "",
    "hr_lexicon": "",
    "hr_rule_fsts": "",
    "language": "en",
    "maximum_tail_padding_samples": 0,
    "model_type": "whisper",
    "native_chunk_samples": 1_600,
    "num_threads": 1,
    "numpy_version": "2.4.6",
    "package_version": "1.13.3",
    "provider": "cpu",
    "rule_fars": "",
    "rule_fsts": "",
    "sample_rate": 16_000,
    "segmentation_mode": "external-mobile-zipformer-endpoint-epoch",
    "source_repo_id": "csukuangfj/sherpa-onnx-whisper-base.en",
    "source_revision": CANDIDATE_SOURCE_REVISION,
    "tail_paddings": -1,
    "task": "transcribe",
    "variant": "base.en-int8",
}
_EVALUATOR_FILES = (
    "core/wer.py",
    "mobile/lib/agent_decision.dart",
    "mobile/lib/contract.dart",
    "tests/golden/commands.json",
    "tools/__init__.py",
    "tools/mobile_asr_evidence_eval.py",
    "tools/mobile_asr_two_pass_eval.py",
    "tools/prepare_mobile_asr_evidence_packet.py",
    "tools/prepare_primock57_conversation_fixture.py",
    "tools/provision_mobile_whisper.py",
    "tools/provision_mobile_zipformer.py",
    "tools/streaming_stt/__init__.py",
    "tools/streaming_stt/bounded_io.py",
    "tools/streaming_stt/corpus.py",
    "tools/streaming_stt/mobile-asr-evidence-packet-v2.lock.json",
    "tools/streaming_stt/mobile-whisper-base-en-v1.lock.json",
    "tools/streaming_stt/mobile-zipformer-en-2023-06-26-v1.lock.json",
    "tools/streaming_stt/overlap_metrics.py",
    "tools/streaming_stt/private_diagnostic_receipt.py",
)
_CONTROL_SOURCE_SHA256 = {
    "mobile/lib/agent_decision.dart": (
        "f639f4c2445897e9c79824473b7f4f8c21b2511b3d415a68491b47a2347a238a"
    ),
    "mobile/lib/contract.dart": (
        "6b4058074d396efac59cc25beba37fbd19f032df1e78f1dee7e62d3f41648e21"
    ),
    "tests/golden/commands.json": (
        "998fba2389c9bcd451346d9b8f36cf8048c650c7aa41c9e9fca6fd7e118196c7"
    ),
}
_STOP_LEXEMES = frozenset(
    {
        "stop",
        "cancel",
        "cancel that",
        "quiet",
        "stop talking",
        "stop speaking",
        "be quiet",
    }
)
_CONFIRM_LEXEMES = frozenset({"yes", "confirm", "approve", "do it", "ok do it"})
_DENY_LEXEMES = frozenset({"no", "deny", "cancel command", "do not", "dont"})
_MODE_LEXEMES = {
    "passive mode": "passive",
    "assistant mode": "assistant",
    "command mode": "command",
    "search mode": "search",
    "research mode": "research",
    "dictation mode": "dictation",
    "meeting mode": "meeting",
}
_CONTROL_CLASSES = (
    "stop",
    "confirm_lexeme",
    "deny_lexeme",
    "mode_lexeme",
    "none",
)
_ENDPOINT_EPOCH_GEOMETRY_DOMAIN = (
    "speaker/mobile-asr-two-pass/reset-committed-endpoint-epoch-geometry/v1"
)
_TWO_PASS_SCRATCH_CHILDREN = (
    "candidate-worker",
    "source-bundle",
    "worker",
)


class MobileAsrTwoPassEvalError(RuntimeError):
    """A detail-free two-pass evidence, lifecycle, or publication failure."""


@dataclass(frozen=True, slots=True)
class _CandidateAuthority:
    provision: object = field(repr=False)
    manifest: object = field(repr=False)
    manifest_path: Path = field(repr=False)
    manifest_sha256: str
    provision_receipt_sha256: str
    artifact_set_sha256: str
    mobile_config_sha256: str
    total_size_bytes: int


@dataclass(frozen=True, slots=True)
class _BaselineEvaluation:
    outcome: baseline_api._EndpointOutcome = field(repr=False)
    observations: tuple[MobileZipformerEndpointObservation, ...] = field(repr=False)

    def __post_init__(self) -> None:
        if (
            type(self.outcome) is not baseline_api._EndpointOutcome
            or type(self.observations) is not tuple
            or any(
                type(value) is not MobileZipformerEndpointObservation
                for value in self.observations
            )
            or tuple(value.text for value in self.observations)
            != self.outcome.endpoint_texts
            or tuple(value.source_complete for value in self.observations)
            != self.outcome.endpoint_source_complete
        ):
            raise MobileAsrTwoPassEvalError()


@dataclass(frozen=True, slots=True)
class _Epoch:
    leaf_index: int
    observation_index: int
    start_sample: int
    end_sample: int
    samples: int
    sha256: str
    path: Path = field(repr=False)

    def __post_init__(self) -> None:
        if (
            type(self.leaf_index) is not int
            or self.leaf_index < 0
            or type(self.observation_index) is not int
            or self.observation_index < 0
            or type(self.start_sample) is not int
            or type(self.end_sample) is not int
            or not 0 <= self.start_sample < self.end_sample
            or self.samples != self.end_sample - self.start_sample
            or self.samples <= 0
            or _SHA256_RE.fullmatch(self.sha256) is None
            or not isinstance(self.path, Path)
        ):
            raise MobileAsrTwoPassEvalError()


@dataclass(frozen=True, slots=True)
class LoadedMobileAsrTwoPassEvalReport:
    path: Path = field(repr=False)
    digest: str
    packet_index_sha256: str
    packet_receipt_sha256: str
    baseline_model_manifest_sha256: str
    candidate_model_manifest_sha256: str
    source_bundle_sha256: str
    value: Mapping[str, object] = field(repr=False)
    snapshot: tuple[int, ...] = field(repr=False)


def _canonical_json(value: object, *, newline: bool = False) -> bytes:
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (MemoryError, OverflowError, TypeError, UnicodeError, ValueError):
        raise MobileAsrTwoPassEvalError() from None
    return raw + (b"\n" if newline else b"")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256(value: object) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise MobileAsrTwoPassEvalError()
    return value


def _strict_json(raw: bytes) -> object:
    if not raw or len(raw) > _MAX_REPORT_BYTES:
        raise MobileAsrTwoPassEvalError()

    def pairs(rows: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in rows:
            if key in result:
                raise MobileAsrTwoPassEvalError()
            result[key] = value
        return result

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                MobileAsrTwoPassEvalError()
            ),
        )
    except (MemoryError, OverflowError, UnicodeError, ValueError):
        raise MobileAsrTwoPassEvalError() from None


def _snapshot(metadata: os.stat_result) -> tuple[int, ...]:
    return baseline_api._snapshot(metadata)


def _stable_two_pass_scratch_snapshot(scratch: Path) -> tuple[int, ...]:
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        with opened_directory_nofollow(scratch, require_private=True) as (
            _bound,
            root_descriptor,
        ):
            before = os.fstat(root_descriptor)
            children: list[str] = []
            with os.scandir(root_descriptor) as entries:
                for entry in entries:
                    if len(children) >= len(_TWO_PASS_SCRATCH_CHILDREN):
                        raise MobileAsrTwoPassEvalError()
                    children.append(entry.name)
            if (
                _bound != scratch
                or _snapshot(scratch.lstat()) != _snapshot(before)
                or not stat.S_ISDIR(before.st_mode)
                or stat.S_IMODE(before.st_mode) != 0o700
                or before.st_uid != os.getuid()
                or tuple(sorted(children)) != _TWO_PASS_SCRATCH_CHILDREN
            ):
                raise MobileAsrTwoPassEvalError()
            for child_name in _TWO_PASS_SCRATCH_CHILDREN:
                child_before = os.stat(
                    child_name,
                    dir_fd=root_descriptor,
                    follow_symlinks=False,
                )
                child_descriptor = os.open(
                    child_name,
                    flags,
                    dir_fd=root_descriptor,
                )
                try:
                    child_opened = os.fstat(child_descriptor)
                    child_after = os.stat(
                        child_name,
                        dir_fd=root_descriptor,
                        follow_symlinks=False,
                    )
                    if (
                        _snapshot(child_before) != _snapshot(child_opened)
                        or _snapshot(child_opened) != _snapshot(child_after)
                        or not stat.S_ISDIR(child_opened.st_mode)
                        or stat.S_IMODE(child_opened.st_mode) != 0o700
                        or child_opened.st_uid != os.getuid()
                    ):
                        raise MobileAsrTwoPassEvalError()
                finally:
                    os.close(child_descriptor)
            after = os.fstat(root_descriptor)
            if _snapshot(before) != _snapshot(after) or _snapshot(after) != _snapshot(
                scratch.lstat()
            ):
                raise MobileAsrTwoPassEvalError()
            return _snapshot(after)
    except MobileAsrTwoPassEvalError:
        raise
    except Exception:
        raise MobileAsrTwoPassEvalError() from None


def _advance_scratch_snapshot_for_candidate(
    scratch: Path,
    previous_snapshot: tuple[int, ...],
) -> tuple[int, ...]:
    if (
        type(previous_snapshot) is not tuple
        or len(previous_snapshot) != 9
        or any(type(value) is not int for value in previous_snapshot)
    ):
        raise MobileAsrTwoPassEvalError()
    current = _stable_two_pass_scratch_snapshot(scratch)
    if current[:5] != previous_snapshot[:5] or current[5] != previous_snapshot[5] + 1:
        raise MobileAsrTwoPassEvalError()
    return current


def _verify_two_pass_scratch_snapshot(
    scratch: Path,
    expected_snapshot: tuple[int, ...],
) -> None:
    if (
        type(expected_snapshot) is not tuple
        or len(expected_snapshot) != 9
        or any(type(value) is not int for value in expected_snapshot)
        or _stable_two_pass_scratch_snapshot(scratch) != expected_snapshot
    ):
        raise MobileAsrTwoPassEvalError()


def _mobile_config_dict(value: object) -> dict[str, object]:
    try:
        result = getattr(value, "as_dict")()
        if not isinstance(result, dict) or not result:
            raise MobileAsrTwoPassEvalError()
        _canonical_json(result)
        return dict(result)
    except MobileAsrTwoPassEvalError:
        raise
    except Exception:
        raise MobileAsrTwoPassEvalError() from None


def _bind_candidate_configs(
    provision_config: object,
    manifest_config: object,
    *,
    provision_type: type[object],
    manifest_type: type[object],
    source_revision: str,
) -> str:
    if (
        type(provision_config) is not provision_type
        or type(manifest_config) is not manifest_type
        or type(source_revision) is not str
        or re.fullmatch(r"[0-9a-f]{40}", source_revision) is None
        or source_revision != CANDIDATE_SOURCE_REVISION
    ):
        raise MobileAsrTwoPassEvalError()
    expected = dict(_EXPECTED_CANDIDATE_CONFIG)
    provision_value = _mobile_config_dict(provision_config)
    manifest_value = _mobile_config_dict(manifest_config)
    if provision_value != expected or manifest_value != expected:
        raise MobileAsrTwoPassEvalError()
    if provision_value != manifest_value:
        raise MobileAsrTwoPassEvalError()
    return _canonical_sha256(manifest_value)


def _load_candidate_authority(path: Path | str) -> _CandidateAuthority:
    try:
        from tools.provision_mobile_whisper import (
            MobileWhisperConfig as ProvisionMobileWhisperConfig,
            SOURCE_REVISION as MOBILE_WHISPER_SOURCE_REVISION,
            load_mobile_whisper_provision,
        )
        from tools.streaming_stt.manifest import (
            MobileWhisperConfig,
            load_worker_manifest,
        )

        provision = load_mobile_whisper_provision(path)
        manifest_path = Path(getattr(provision, "path"))
        manifest = load_worker_manifest(manifest_path)
        config_sha256 = _bind_candidate_configs(
            getattr(provision, "mobile_config"),
            getattr(manifest, "adapter_config"),
            provision_type=ProvisionMobileWhisperConfig,
            manifest_type=MobileWhisperConfig,
            source_revision=MOBILE_WHISPER_SOURCE_REVISION,
        )
        if (
            getattr(provision, "model_id") != CANDIDATE_MODEL_ID
            or getattr(provision, "adapter") != CANDIDATE_ADAPTER
            or getattr(manifest, "schema_version") != 11
            or getattr(manifest, "model_id") != CANDIDATE_MODEL_ID
            or getattr(manifest, "adapter") != CANDIDATE_ADAPTER
            or getattr(manifest, "digest") != getattr(provision, "digest")
            or Path(getattr(manifest, "path")) != manifest_path
            or getattr(manifest, "python") != getattr(provision, "python")
            or getattr(manifest, "worker") != getattr(provision, "worker")
            or tuple(getattr(manifest, "artifacts"))
            != tuple(getattr(provision, "artifacts"))
            or getattr(provision, "artifact_set_sha256")
            != CANDIDATE_ARTIFACT_SET_SHA256
            or getattr(provision, "total_size_bytes")
            != CANDIDATE_MODEL_TOTAL_SIZE_BYTES
            or config_sha256 != CANDIDATE_MOBILE_CONFIG_SHA256
        ):
            raise MobileAsrTwoPassEvalError()
        return _CandidateAuthority(
            provision=provision,
            manifest=manifest,
            manifest_path=manifest_path,
            manifest_sha256=_sha256(getattr(provision, "digest")),
            provision_receipt_sha256=_sha256(getattr(provision, "receipt_digest")),
            artifact_set_sha256=_sha256(getattr(provision, "artifact_set_sha256")),
            mobile_config_sha256=config_sha256,
            total_size_bytes=int(getattr(provision, "total_size_bytes")),
        )
    except MobileAsrTwoPassEvalError:
        raise
    except Exception:
        raise MobileAsrTwoPassEvalError() from None


def _verify_candidate_authority(authority: _CandidateAuthority) -> None:
    current = _load_candidate_authority(authority.manifest_path)
    if current != authority:
        raise MobileAsrTwoPassEvalError()


def _evaluator_closure_sha256() -> str:
    root = Path(__file__).resolve().parents[1]
    rows: list[dict[str, object]] = []
    try:
        for relative in _EVALUATOR_FILES:
            loaded = read_regular_bounded(
                root.joinpath(*PurePosixPath(relative).parts),
                maximum_bytes=4 * 1024 * 1024,
            )
            rows.append(
                {
                    "name": relative,
                    "sha256": hashlib.sha256(loaded.data).hexdigest(),
                    "size_bytes": len(loaded.data),
                }
            )
        return _canonical_sha256(rows)
    except (BoundedReadError, OSError, RuntimeError, ValueError):
        raise MobileAsrTwoPassEvalError() from None


def _verify_control_projection_sources() -> None:
    """Fail closed if the Dart lexeme contract drifts from this projection."""

    root = Path(__file__).resolve().parents[1]
    try:
        for relative, expected in _CONTROL_SOURCE_SHA256.items():
            loaded = read_regular_bounded(
                root.joinpath(*PurePosixPath(relative).parts),
                maximum_bytes=512 * 1024,
            )
            if hashlib.sha256(loaded.data).hexdigest() != expected:
                raise MobileAsrTwoPassEvalError()
    except MobileAsrTwoPassEvalError:
        raise
    except (BoundedReadError, OSError, RuntimeError, ValueError):
        raise MobileAsrTwoPassEvalError() from None


def _candidate_binding(
    authority: _CandidateAuthority, source_bundle: SourceBundle
) -> dict[str, object]:
    return {
        "adapter": CANDIDATE_ADAPTER,
        "artifact_set_sha256": authority.artifact_set_sha256,
        "mobile_config_sha256": authority.mobile_config_sha256,
        "model_id": CANDIDATE_MODEL_ID,
        "model_manifest_sha256": authority.manifest_sha256,
        "model_total_size_bytes": authority.total_size_bytes,
        "provision_receipt_sha256": authority.provision_receipt_sha256,
        "source_bundle_files": len(source_bundle.files),
        "source_bundle_sha256": source_bundle.tree_sha256,
    }


def _verify_common_worker_binding(
    value: object, *, adapter: str, model_id: str
) -> None:
    fields = {
        "adapter",
        "artifact_set_sha256",
        "mobile_config_sha256",
        "model_id",
        "model_manifest_sha256",
        "model_total_size_bytes",
        "provision_receipt_sha256",
        "source_bundle_files",
        "source_bundle_sha256",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or value.get("adapter") != adapter
        or value.get("model_id") != model_id
        or type(value.get("model_total_size_bytes")) is not int
        or value["model_total_size_bytes"] <= 0
        or type(value.get("source_bundle_files")) is not int
        or value["source_bundle_files"] <= 0
    ):
        raise MobileAsrTwoPassEvalError()
    for key in (
        "artifact_set_sha256",
        "mobile_config_sha256",
        "model_manifest_sha256",
        "provision_receipt_sha256",
        "source_bundle_sha256",
    ):
        _sha256(value.get(key))


def _epoch_payload(
    leaf: baseline_api._Leaf,
    *,
    start_sample: int,
    end_sample: int,
    maximum_tail_samples: int,
) -> bytes:
    try:
        if (
            type(leaf) is not baseline_api._Leaf
            or type(start_sample) is not int
            or type(end_sample) is not int
            or type(maximum_tail_samples) is not int
            or not 0 <= start_sample < end_sample <= leaf.samples + maximum_tail_samples
            or len(leaf.audio) != leaf.samples * 4
        ):
            raise MobileAsrTwoPassEvalError()
        source_start = min(start_sample, leaf.samples)
        source_end = min(end_sample, leaf.samples)
        source = leaf.audio[source_start * 4 : source_end * 4]
        tail_samples = max(0, end_sample - max(start_sample, leaf.samples))
        payload = source + b"\x00" * (tail_samples * 4)
        if len(payload) != (end_sample - start_sample) * 4:
            raise MobileAsrTwoPassEvalError()
        return payload
    except MobileAsrTwoPassEvalError:
        raise
    except (MemoryError, OverflowError, TypeError, ValueError):
        raise MobileAsrTwoPassEvalError() from None


def _extract_baseline_evaluation(
    trace: object, leaf: baseline_api._Leaf
) -> _BaselineEvaluation:
    try:
        outcome = baseline_api._extract_endpoint_outcome(trace, leaf)
        final = getattr(trace, "final")
        if type(final) is not MobileZipformerFinalEvent:
            raise MobileAsrTwoPassEvalError()
        observations = final.endpoint_observations
        previous = 0
        for observation in observations:
            if (
                type(observation) is not MobileZipformerEndpointObservation
                or not previous
                < observation.observed_sample
                <= leaf.samples + _BASELINE_STREAM.tail_padding_samples
                or observation.source_complete
                != (observation.observed_sample >= leaf.samples)
            ):
                raise MobileAsrTwoPassEvalError()
            previous = observation.observed_sample
        return _BaselineEvaluation(outcome=outcome, observations=observations)
    except MobileAsrTwoPassEvalError:
        raise
    except Exception:
        raise MobileAsrTwoPassEvalError() from None


def _write_epoch_files(
    root: Path,
    inputs: baseline_api._Inputs,
    evaluations: Sequence[_BaselineEvaluation],
) -> tuple[_Epoch, ...]:
    leaves = tuple(leaf for component in inputs.components for leaf in component.leaves)
    if len(leaves) != len(evaluations) or any(
        type(evaluation) is not _BaselineEvaluation for evaluation in evaluations
    ):
        raise MobileAsrTwoPassEvalError()
    try:
        root.mkdir(mode=0o700)
        os.chmod(root, 0o700)
        epochs: list[_Epoch] = []
        total_bytes = 0
        for leaf_index, (leaf, evaluation) in enumerate(
            zip(leaves, evaluations, strict=True)
        ):
            previous = 0
            for observation_index, observation in enumerate(evaluation.observations):
                payload = _epoch_payload(
                    leaf,
                    start_sample=previous,
                    end_sample=observation.observed_sample,
                    maximum_tail_samples=_BASELINE_STREAM.tail_padding_samples,
                )
                total_bytes += len(payload)
                if total_bytes > MAXIMUM_DERIVED_EPOCH_PCM_BYTES:
                    raise MobileAsrTwoPassEvalError()
                path = root / f"epoch-{len(epochs):03d}.f32le"
                baseline_api._write_private(path, payload)
                epochs.append(
                    _Epoch(
                        leaf_index=leaf_index,
                        observation_index=observation_index,
                        start_sample=previous,
                        end_sample=observation.observed_sample,
                        samples=len(payload) // 4,
                        sha256=hashlib.sha256(payload).hexdigest(),
                        path=path,
                    )
                )
                previous = observation.observed_sample
        if total_bytes != sum(epoch.samples * 4 for epoch in epochs):
            raise MobileAsrTwoPassEvalError()
        return tuple(epochs)
    except MobileAsrTwoPassEvalError:
        raise
    except Exception:
        raise MobileAsrTwoPassEvalError() from None


def _verify_epochs(
    root: Path,
    inputs: baseline_api._Inputs,
    evaluations: Sequence[_BaselineEvaluation],
    epochs: tuple[_Epoch, ...],
) -> None:
    leaves = tuple(leaf for component in inputs.components for leaf in component.leaves)
    expected_count = sum(len(evaluation.observations) for evaluation in evaluations)
    if len(leaves) != len(evaluations) or len(epochs) != expected_count:
        raise MobileAsrTwoPassEvalError()
    try:
        metadata = root.lstat()
        if (
            root.resolve(strict=True) != root
            or not stat.S_ISDIR(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o700
            or metadata.st_uid != os.getuid()
        ):
            raise MobileAsrTwoPassEvalError()
        cursor = 0
        total_bytes = 0
        for leaf_index, (leaf, evaluation) in enumerate(
            zip(leaves, evaluations, strict=True)
        ):
            previous = 0
            for observation_index, observation in enumerate(evaluation.observations):
                epoch = epochs[cursor]
                payload = _epoch_payload(
                    leaf,
                    start_sample=previous,
                    end_sample=observation.observed_sample,
                    maximum_tail_samples=_BASELINE_STREAM.tail_padding_samples,
                )
                expected_path = root / f"epoch-{cursor:03d}.f32le"
                current = epoch.path.lstat()
                if (
                    epoch.leaf_index != leaf_index
                    or epoch.observation_index != observation_index
                    or epoch.start_sample != previous
                    or epoch.end_sample != observation.observed_sample
                    or epoch.samples * 4 != len(payload)
                    or epoch.sha256 != hashlib.sha256(payload).hexdigest()
                    or epoch.path != expected_path
                    or epoch.path.resolve(strict=True) != epoch.path
                    or not stat.S_ISREG(current.st_mode)
                    or stat.S_IMODE(current.st_mode) != 0o600
                    or current.st_nlink != 1
                    or current.st_uid != os.getuid()
                    or current.st_size != len(payload)
                ):
                    raise MobileAsrTwoPassEvalError()
                digest = hash_regular_bounded(
                    epoch.path,
                    maximum_bytes=baseline_api.packet_api.MAX_PCM_BYTES,
                    expected_bytes=len(payload),
                )
                if digest.sha256 != epoch.sha256:
                    raise MobileAsrTwoPassEvalError()
                total_bytes += len(payload)
                previous = observation.observed_sample
                cursor += 1
        if (
            cursor != len(epochs)
            or total_bytes > MAXIMUM_DERIVED_EPOCH_PCM_BYTES
            or total_bytes != sum(epoch.samples * 4 for epoch in epochs)
        ):
            raise MobileAsrTwoPassEvalError()
    except MobileAsrTwoPassEvalError:
        raise
    except Exception:
        raise MobileAsrTwoPassEvalError() from None


def _extract_candidate_text(trace: object, epoch: _Epoch, request_id: str) -> str:
    try:
        partials = getattr(trace, "partials")
        final = getattr(trace, "final")
        if (
            partials != ()
            or type(final) is not FinalEvent
            or final.protocol_version != FINAL_PROTOCOL_VERSION
            or final.request_id != request_id
            or type(final.text) is not str
            or final.samples_seen != epoch.samples
            or final.model_padding_samples != 0
            or not math.isclose(
                final.audio_seconds,
                epoch.samples / 16_000,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        ):
            raise MobileAsrTwoPassEvalError()
        return final.text
    except MobileAsrTwoPassEvalError:
        raise
    except Exception:
        raise MobileAsrTwoPassEvalError() from None


def _candidate_outcomes(
    evaluations: Sequence[_BaselineEvaluation], candidate_texts: Sequence[str]
) -> tuple[baseline_api._EndpointOutcome, ...]:
    if any(type(value) is not str for value in candidate_texts):
        raise MobileAsrTwoPassEvalError()
    cursor = 0
    outcomes: list[baseline_api._EndpointOutcome] = []
    for evaluation in evaluations:
        count = len(evaluation.observations)
        selected = tuple(candidate_texts[cursor : cursor + count])
        if len(selected) != count:
            raise MobileAsrTwoPassEvalError()
        outcomes.append(
            baseline_api._EndpointOutcome(
                endpoint_texts=selected,
                endpoint_source_complete=evaluation.outcome.endpoint_source_complete,
                endpoint_reason=evaluation.outcome.endpoint_reason,
                source_complete=evaluation.outcome.source_complete,
            )
        )
        cursor += count
    if cursor != len(candidate_texts):
        raise MobileAsrTwoPassEvalError()
    return tuple(outcomes)


def _baseline_endpoint_epoch_geometry(
    evaluations: Sequence[_BaselineEvaluation],
) -> tuple[int, str]:
    rows: list[dict[str, object]] = []
    for source_ordinal, evaluation in enumerate(evaluations):
        if type(evaluation) is not _BaselineEvaluation:
            raise MobileAsrTwoPassEvalError()
        for observation_ordinal, observation in enumerate(evaluation.observations):
            rows.append(
                {
                    "observation_ordinal": observation_ordinal,
                    "observed_sample": observation.observed_sample,
                    "source_complete": observation.source_complete,
                    "source_ordinal": source_ordinal,
                }
            )
    if len(rows) > EXPECTED_ENDPOINT_EPOCHS:
        raise MobileAsrTwoPassEvalError()
    return len(rows), _canonical_sha256(
        {"domain": _ENDPOINT_EPOCH_GEOMETRY_DOMAIN, "rows": rows}
    )


def _reduce(
    inputs: baseline_api._Inputs,
    outcomes: Sequence[baseline_api._EndpointOutcome],
    *,
    include_endpoint_integrity: bool,
) -> list[dict[str, object]]:
    selected = tuple(outcomes)
    if len(selected) != EXPECTED_SOURCE_EVALUATIONS:
        raise MobileAsrTwoPassEvalError()
    reports: list[dict[str, object]] = []
    cursor = 0
    for component in inputs.components:
        subset = selected[cursor : cursor + len(component.leaves)]
        if len(subset) != len(component.leaves):
            raise MobileAsrTwoPassEvalError()
        row = baseline_api._component_report(component, subset)
        if not include_endpoint_integrity:
            totals = baseline_api._endpoint_totals(subset)
            row = {
                "component": row["component"],
                "epoch_text_accounting": {
                    "empty_candidate_text_epochs": totals.empty_committed_endpoints,
                    "inherited_endpoint_epochs": totals.committed_endpoints,
                    "nonempty_candidate_text_epochs": totals.nonempty_committed_endpoints,
                    "source_evaluations": totals.evaluations,
                },
                "evaluations": row["evaluations"],
                "metric_domain": row["metric_domain"],
                "metrics": row["metrics"],
            }
        reports.append(row)
        cursor += len(component.leaves)
    if cursor != len(selected):
        raise MobileAsrTwoPassEvalError()
    return reports


def _accepted_baseline_contract(
    components: Sequence[Mapping[str, object]],
    endpoints: baseline_api._EndpointTotals,
    inputs: baseline_api._Inputs,
) -> dict[str, object]:
    return {
        "components": list(components),
        "execution": {
            "components": len(inputs.components),
            "endpoint_integrity": endpoints.as_dict(),
            "evaluations": EXPECTED_SOURCE_EVALUATIONS,
            "logical_cases": sum(value.logical_cases for value in inputs.components),
            "pcm_bytes": inputs.pcm_bytes,
            "pcm_inputs": EXPECTED_SOURCE_EVALUATIONS,
            "samples": inputs.samples,
            "stream": {
                "chunk_samples": _BASELINE_STREAM.chunk_samples,
                "pace": _BASELINE_STREAM.pace,
                "partial_report_cadence_ms": _BASELINE_STREAM.partial_interval_ms,
                "protocol_version": MOBILE_ZIPFORMER_ENDPOINT_PROTOCOL_VERSION,
                "repeats": 1,
                "tail_padding_samples": _BASELINE_STREAM.tail_padding_samples,
            },
        },
    }


def _require_accepted_baseline(
    components: Sequence[Mapping[str, object]],
    endpoints: baseline_api._EndpointTotals,
    inputs: baseline_api._Inputs,
) -> str:
    contract = _accepted_baseline_contract(components, endpoints, inputs)
    digest = _canonical_sha256(contract)
    if (
        not inputs.production_inputs
        or digest != ACCEPTED_BASELINE_AGGREGATE_SHA256
        or endpoints.as_dict() != _EXPECTED_BASELINE_ENDPOINTS
    ):
        raise MobileAsrTwoPassEvalError()
    return digest


def _normalize_mobile_control(text: str) -> str:
    """Match Dart `normalizeCommand`: delete non-ASCII-letter/space bytes."""

    if type(text) is not str:
        raise MobileAsrTwoPassEvalError()
    lowered = text.lower()
    retained = "".join(
        character
        for character in lowered
        if character == " " or "a" <= character <= "z"
    )
    return " ".join(retained.split())


def _control_lexeme_projection(text: str) -> tuple[str, str]:
    """Project text only; this is not a stateful mobile AgentDecision."""

    normalized = _normalize_mobile_control(text)
    if normalized in _STOP_LEXEMES:
        return "stop", "stop"
    if normalized in _CONFIRM_LEXEMES:
        return "confirm_lexeme", "confirm_lexeme"
    if normalized in _DENY_LEXEMES:
        return "deny_lexeme", "deny_lexeme"
    target = _MODE_LEXEMES.get(normalized)
    if target is not None:
        return "mode_lexeme", f"mode_lexeme:{target}"
    return "none", "none"


def _command_hit_vector(
    leaf: baseline_api._Leaf, outcome: baseline_api._EndpointOutcome
) -> tuple[tuple[bool, ...], tuple[bool, ...]]:
    decisions = tuple(tuple(normalize(value)) for value in outcome.endpoint_texts)

    def hits(commands: Sequence[str]) -> tuple[bool, ...]:
        return tuple(
            any(
                baseline_api._contains_tokens(decision, normalize(command))
                for decision in decisions
            )
            for command in commands
        )

    return hits(leaf.commands), hits(leaf.forbidden_commands)


def _comparison(
    inputs: baseline_api._Inputs,
    baseline_outcomes: Sequence[baseline_api._EndpointOutcome],
    candidate_outcomes: Sequence[baseline_api._EndpointOutcome],
) -> dict[str, object]:
    baseline_selected = tuple(baseline_outcomes)
    candidate_selected = tuple(candidate_outcomes)
    if (
        len(baseline_selected) != EXPECTED_SOURCE_EVALUATIONS
        or len(candidate_selected) != EXPECTED_SOURCE_EVALUATIONS
    ):
        raise MobileAsrTwoPassEvalError()
    baseline_texts: list[str] = []
    candidate_texts: list[str] = []
    for baseline, candidate in zip(baseline_selected, candidate_selected, strict=True):
        if (
            len(baseline.endpoint_texts) != len(candidate.endpoint_texts)
            or baseline.endpoint_source_complete != candidate.endpoint_source_complete
            or baseline.endpoint_reason != candidate.endpoint_reason
        ):
            raise MobileAsrTwoPassEvalError()
        baseline_texts.extend(baseline.endpoint_texts)
        candidate_texts.extend(candidate.endpoint_texts)
    endpoint_epochs = len(baseline_texts)
    baseline_empty = sum(value == "" for value in baseline_texts)
    candidate_empty = sum(value == "" for value in candidate_texts)
    exact = sum(
        normalize(left) == normalize(right)
        for left, right in zip(baseline_texts, candidate_texts, strict=True)
    )

    transition_counts = {
        baseline_class: {candidate_class: 0 for candidate_class in _CONTROL_CLASSES}
        for baseline_class in _CONTROL_CLASSES
    }
    exact_projection_agreements = mode_alias_disagreements = 0
    for baseline_text, candidate_text in zip(
        baseline_texts, candidate_texts, strict=True
    ):
        baseline_class, baseline_exact = _control_lexeme_projection(baseline_text)
        candidate_class, candidate_exact = _control_lexeme_projection(candidate_text)
        transition_counts[baseline_class][candidate_class] += 1
        exact_projection_agreements += baseline_exact == candidate_exact
        mode_alias_disagreements += (
            baseline_class == candidate_class == "mode_lexeme"
            and baseline_exact != candidate_exact
        )
    matrix = [
        {
            "baseline_class": baseline_class,
            "candidate_counts": dict(transition_counts[baseline_class]),
        }
        for baseline_class in _CONTROL_CLASSES
    ]
    class_agreements = sum(
        transition_counts[value][value] for value in _CONTROL_CLASSES
    )
    gained = sum(
        transition_counts["none"][value]
        for value in _CONTROL_CLASSES
        if value != "none"
    )
    lost = sum(
        transition_counts[value]["none"]
        for value in _CONTROL_CLASSES
        if value != "none"
    )
    changed_nonempty = sum(
        transition_counts[left][right]
        for left in _CONTROL_CLASSES
        for right in _CONTROL_CLASSES
        if left != "none" and right != "none" and left != right
    )
    command_component = inputs.components[0]
    if command_component.component_id != "command-noise":
        raise MobileAsrTwoPassEvalError()
    target_agreements = forbidden_agreements = 0
    target_gained = target_lost = forbidden_gained = forbidden_lost = 0
    for leaf, baseline, candidate in zip(
        command_component.leaves,
        baseline_selected[: len(command_component.leaves)],
        candidate_selected[: len(command_component.leaves)],
        strict=True,
    ):
        base_target, base_forbidden = _command_hit_vector(leaf, baseline)
        cand_target, cand_forbidden = _command_hit_vector(leaf, candidate)
        target_agreements += sum(
            left == right for left, right in zip(base_target, cand_target, strict=True)
        )
        forbidden_agreements += sum(
            left == right
            for left, right in zip(base_forbidden, cand_forbidden, strict=True)
        )
        target_gained += sum(
            not left and right
            for left, right in zip(base_target, cand_target, strict=True)
        )
        target_lost += sum(
            left and not right
            for left, right in zip(base_target, cand_target, strict=True)
        )
        forbidden_gained += sum(
            not left and right
            for left, right in zip(base_forbidden, cand_forbidden, strict=True)
        )
        forbidden_lost += sum(
            left and not right
            for left, right in zip(base_forbidden, cand_forbidden, strict=True)
        )
    target_attempts = sum(len(leaf.commands) for leaf in command_component.leaves)
    forbidden_attempts = sum(
        len(leaf.forbidden_commands) for leaf in command_component.leaves
    )
    return {
        "candidate_empty_on_nonempty_baseline_epochs": sum(
            left != "" and right == ""
            for left, right in zip(baseline_texts, candidate_texts, strict=True)
        ),
        "candidate_empty_epochs": candidate_empty,
        "candidate_nonempty_on_empty_baseline_epochs": sum(
            left == "" and right != ""
            for left, right in zip(baseline_texts, candidate_texts, strict=True)
        ),
        "endpoint_epochs": endpoint_epochs,
        "exact_normalized_epoch_agreements": exact,
        "baseline_empty_epochs": baseline_empty,
        "control_lexeme_projection": {
            "actual_agent_decision_authority": False,
            "candidate_gained_control_lexeme_epochs": gained,
            "candidate_lost_control_lexeme_epochs": lost,
            "changed_nonempty_control_class_epochs": changed_nonempty,
            "endpoint_epochs": endpoint_epochs,
            "exact_class_agreements": class_agreements,
            "exact_class_disagreements": endpoint_epochs - class_agreements,
            "exact_projection_agreements": exact_projection_agreements,
            "exact_projection_disagreements": (
                endpoint_epochs - exact_projection_agreements
            ),
            "mode_alias_disagreements": mode_alias_disagreements,
            "pending_confirmation_state_applied": False,
            "protocol": "mobile-text-control-lexeme-projection-v1",
            "transition_matrix": matrix,
        },
        "packet_command_assertion_disagreement": {
            "evaluations": len(command_component.leaves),
            "forbidden_attempts": forbidden_attempts,
            "forbidden_exact_agreements": forbidden_agreements,
            "forbidden_exact_disagreements": (
                forbidden_attempts - forbidden_agreements
            ),
            "forbidden_gained_hits": forbidden_gained,
            "forbidden_lost_hits": forbidden_lost,
            "routing_authority": False,
            "target_attempts": target_attempts,
            "target_exact_agreements": target_agreements,
            "target_exact_disagreements": target_attempts - target_agreements,
            "target_gained_hits": target_gained,
            "target_lost_hits": target_lost,
        },
    }


def _worker_stderr(value: Mapping[str, object]) -> dict[str, object]:
    result = dict(value)
    if (
        set(result) != {"bytes", "sha256", "truncated"}
        or type(result.get("bytes")) is not int
        or result["bytes"] < 0
        or _sha256(result.get("sha256")) != result["sha256"]
        or type(result.get("truncated")) is not bool
    ):
        raise MobileAsrTwoPassEvalError()
    return result


def _report(
    inputs: baseline_api._Inputs,
    *,
    baseline_evaluations: Sequence[_BaselineEvaluation],
    baseline_components: list[dict[str, object]],
    candidate_components: list[dict[str, object]],
    baseline_endpoints: baseline_api._EndpointTotals,
    candidate_outcomes: Sequence[baseline_api._EndpointOutcome],
    comparison: Mapping[str, object],
    baseline_binding: Mapping[str, object],
    candidate_binding: Mapping[str, object],
    evaluator_closure_sha256: str,
    baseline_aggregate_sha256: str,
    derived_epoch_pcm_bytes_total: int,
    baseline_stderr: Mapping[str, object],
    candidate_stderr: Mapping[str, object],
    model_executed: bool,
) -> dict[str, object]:
    endpoint_epochs = int(comparison["endpoint_epochs"])
    geometry_count, geometry_sha256 = _baseline_endpoint_epoch_geometry(
        baseline_evaluations
    )
    candidate_geometry = baseline_api._endpoint_totals(candidate_outcomes).as_dict()
    if geometry_count != endpoint_epochs:
        raise MobileAsrTwoPassEvalError()
    value: dict[str, object] = {
        "baseline": {
            "components": baseline_components,
            "endpoint_integrity": baseline_endpoints.as_dict(),
            "role": "reset-committed-endpoint-epoch-segmentation",
        },
        "bindings": {
            "accepted_baseline_aggregate_sha256": baseline_aggregate_sha256,
            "accepted_baseline_report_sha256": ACCEPTED_BASELINE_REPORT_SHA256,
            "current_rerun_endpoint_epoch_geometry_count": geometry_count,
            "current_rerun_endpoint_epoch_geometry_sha256": geometry_sha256,
            "baseline_worker": dict(baseline_binding),
            "candidate_worker": dict(candidate_binding),
            "evaluator_closure_sha256": evaluator_closure_sha256,
            "packet_index_sha256": inputs.packet_index_sha256,
            "packet_lock_sha256": inputs.packet_lock_sha256,
            "packet_receipt_sha256": inputs.packet_receipt_sha256,
        },
        "candidate": {
            "authority": dict(_CANDIDATE_AUTHORITY),
            "components": candidate_components,
            "role": "revision-text-only-on-baseline-epochs",
        },
        "comparison": dict(comparison),
        "complete": True,
        "evidence_scope": {
            **_GENERAL_EVIDENCE_SCOPE,
            "baseline_endpoint_epoch_boundary_authority": inputs.production_inputs,
            "semantic_turn_boundary_authority": False,
            "candidate_model_executed": model_executed,
            "candidate_revision_text_evidence": inputs.production_inputs,
            "production_packet": inputs.production_inputs,
            "unpooled_component_result_authority": inputs.production_inputs,
        },
        "execution": {
            "baseline_stream": {
                "chunk_samples": _BASELINE_STREAM.chunk_samples,
                "pace": _BASELINE_STREAM.pace,
                "partial_report_cadence_ms": _BASELINE_STREAM.partial_interval_ms,
                "protocol_version": MOBILE_ZIPFORMER_ENDPOINT_PROTOCOL_VERSION,
                "repeats": 1,
                "tail_padding_samples": _BASELINE_STREAM.tail_padding_samples,
            },
            "baseline_worker_stderr": _worker_stderr(baseline_stderr),
            "candidate_epoch_geometry": {
                "baseline_endpoint_observations": endpoint_epochs,
                "candidate_app_admissions": 0,
                "candidate_requests": endpoint_epochs,
                "candidate_worker_lifetime_pcm_limit_bytes": (
                    CANDIDATE_WORKER_LIFETIME_PCM_LIMIT_BYTES
                ),
                "derived_epoch_pcm_bytes_total": derived_epoch_pcm_bytes_total,
                "derived_epoch_samples_total": derived_epoch_pcm_bytes_total // 4,
                "empty_baseline_epochs_create_requests": True,
                "maximum_derived_epoch_pcm_bytes_total": (
                    MAXIMUM_DERIVED_EPOCH_PCM_BYTES
                ),
                "maximum_derived_epoch_samples_total": (MAXIMUM_DERIVED_EPOCH_SAMPLES),
                "preserves_baseline_order_and_boundaries": True,
                "reported_adapter_or_evaluator_appended_pcm_padding_samples": 0,
                "sherpa_internal_tail_padding_observation": (
                    "unobserved-and-unreported"
                ),
                "source_evaluations": EXPECTED_SOURCE_EVALUATIONS,
            },
            "candidate_stream": {
                "chunk_samples": _CANDIDATE_STREAM.chunk_samples,
                "pace": _CANDIDATE_STREAM.pace,
                "partial_report_cadence_ms": _CANDIDATE_STREAM.partial_interval_ms,
                "protocol_version": FINAL_PROTOCOL_VERSION,
                "repeats": 1,
                "tail_padding_samples": _CANDIDATE_STREAM.tail_padding_samples,
            },
            "candidate_worker_stderr": _worker_stderr(candidate_stderr),
            "components": len(inputs.components),
            "logical_cases": sum(value.logical_cases for value in inputs.components),
            "source_pcm_bytes": inputs.pcm_bytes,
            "source_pcm_inputs": EXPECTED_SOURCE_EVALUATIONS,
            "source_samples": inputs.samples,
            "workers_sequential": True,
        },
        "kind": REPORT_KIND if inputs.production_inputs else SYNTHETIC_REPORT_KIND,
        "metric_aggregation": dict(_METRIC_AGGREGATION),
        "ok": True,
        "production_inputs": inputs.production_inputs,
        "schema_version": SCHEMA_VERSION,
    }
    # Candidate geometry is used only to reject accidental regrouping.  It is
    # intentionally not serialized as candidate endpoint authority.
    if (
        candidate_geometry["committed_endpoints"] != endpoint_epochs
        or candidate_geometry["evaluations"] != EXPECTED_SOURCE_EVALUATIONS
    ):
        raise MobileAsrTwoPassEvalError()
    value["binding_sha256"] = _canonical_sha256(value)
    _validate_report(value, production_inputs=inputs.production_inputs)
    return value


def _validate_component_rows(
    value: object,
    *,
    baseline: bool,
) -> list[Mapping[str, object]]:
    if not isinstance(value, list) or len(value) != len(
        baseline_api._EXPECTED_COMPONENTS
    ):
        raise MobileAsrTwoPassEvalError()
    rows: list[Mapping[str, object]] = []
    for row, expected in zip(value, baseline_api._EXPECTED_COMPONENTS, strict=True):
        fields = (
            {
                "component",
                "endpoint_integrity",
                "evaluations",
                "metric_domain",
                "metrics",
            }
            if baseline
            else {
                "component",
                "epoch_text_accounting",
                "evaluations",
                "metric_domain",
                "metrics",
            }
        )
        if (
            not isinstance(row, Mapping)
            or set(row) != fields
            or row.get("component") != expected[0]
            or row.get("metric_domain") != expected[1]
            or row.get("evaluations") != expected[3]
        ):
            raise MobileAsrTwoPassEvalError()
        endpoint = row.get("endpoint_integrity") if baseline else None
        if baseline:
            baseline_api._validate_endpoint(endpoint, evaluations=expected[3])
        metric_endpoint = endpoint
        if not baseline:
            accounting = row.get("epoch_text_accounting")
            if (
                not isinstance(accounting, Mapping)
                or set(accounting)
                != {
                    "empty_candidate_text_epochs",
                    "inherited_endpoint_epochs",
                    "nonempty_candidate_text_epochs",
                    "source_evaluations",
                }
                or any(
                    type(accounting.get(key)) is not int or accounting[key] < 0
                    for key in accounting
                )
                or accounting["source_evaluations"] != expected[3]
                or accounting["empty_candidate_text_epochs"]
                + accounting["nonempty_candidate_text_epochs"]
                != accounting["inherited_endpoint_epochs"]
            ):
                raise MobileAsrTwoPassEvalError()
            # Only command-noise validation consumes endpoint-derived context,
            # and it needs this one actual per-component text count.  Do not
            # fabricate candidate endpoint ownership or endpoint geometry.
            metric_endpoint = {
                "nonempty_committed_endpoints": accounting[
                    "nonempty_candidate_text_epochs"
                ]
            }
        baseline_api._validate_component_metrics(
            expected[0], row.get("metrics"), endpoint=metric_endpoint
        )
        rows.append(row)
    return rows


def _validate_report(value: object, *, production_inputs: bool) -> dict[str, object]:
    root_fields = {
        "baseline",
        "binding_sha256",
        "bindings",
        "candidate",
        "comparison",
        "complete",
        "evidence_scope",
        "execution",
        "kind",
        "metric_aggregation",
        "ok",
        "production_inputs",
        "schema_version",
    }
    if (
        not isinstance(value, dict)
        or set(value) != root_fields
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("kind")
        != (REPORT_KIND if production_inputs else SYNTHETIC_REPORT_KIND)
        or value.get("ok") is not True
        or value.get("complete") is not True
        or value.get("production_inputs") is not production_inputs
        or value.get("metric_aggregation") != _METRIC_AGGREGATION
    ):
        raise MobileAsrTwoPassEvalError()
    unsigned = dict(value)
    observed_binding = unsigned.pop("binding_sha256")
    if _sha256(observed_binding) != _canonical_sha256(unsigned):
        raise MobileAsrTwoPassEvalError()

    bindings = value.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != {
        "accepted_baseline_aggregate_sha256",
        "accepted_baseline_report_sha256",
        "current_rerun_endpoint_epoch_geometry_count",
        "current_rerun_endpoint_epoch_geometry_sha256",
        "baseline_worker",
        "candidate_worker",
        "evaluator_closure_sha256",
        "packet_index_sha256",
        "packet_lock_sha256",
        "packet_receipt_sha256",
    }:
        raise MobileAsrTwoPassEvalError()
    for key in (
        "accepted_baseline_aggregate_sha256",
        "accepted_baseline_report_sha256",
        "current_rerun_endpoint_epoch_geometry_sha256",
        "evaluator_closure_sha256",
        "packet_index_sha256",
        "packet_lock_sha256",
        "packet_receipt_sha256",
    ):
        _sha256(bindings.get(key))
    if (
        type(bindings.get("current_rerun_endpoint_epoch_geometry_count")) is not int
        or not 0
        <= bindings["current_rerun_endpoint_epoch_geometry_count"]
        <= EXPECTED_ENDPOINT_EPOCHS
    ):
        raise MobileAsrTwoPassEvalError()
    _verify_common_worker_binding(
        bindings.get("baseline_worker"),
        adapter=BASELINE_ADAPTER if production_inputs else "synthetic-test-only",
        model_id=BASELINE_MODEL_ID if production_inputs else "synthetic-test-only",
    )
    _verify_common_worker_binding(
        bindings.get("candidate_worker"),
        adapter=CANDIDATE_ADAPTER if production_inputs else "synthetic-test-only",
        model_id=CANDIDATE_MODEL_ID if production_inputs else "synthetic-test-only",
    )
    baseline_binding = bindings["baseline_worker"]
    candidate_binding = bindings["candidate_worker"]
    if not isinstance(baseline_binding, Mapping) or not isinstance(
        candidate_binding, Mapping
    ):
        raise MobileAsrTwoPassEvalError()
    if (
        baseline_binding["source_bundle_sha256"]
        != candidate_binding["source_bundle_sha256"]
    ):
        raise MobileAsrTwoPassEvalError()
    if production_inputs and (
        bindings.get("accepted_baseline_aggregate_sha256")
        != ACCEPTED_BASELINE_AGGREGATE_SHA256
        or bindings.get("accepted_baseline_report_sha256")
        != ACCEPTED_BASELINE_REPORT_SHA256
        or bindings.get("packet_index_sha256") != baseline_api.PACKET_INDEX_SHA256
        or bindings.get("packet_lock_sha256") != baseline_api.PACKET_LOCK_SHA256
        or bindings.get("packet_receipt_sha256") != baseline_api.PACKET_RECEIPT_SHA256
        or baseline_binding.get("artifact_set_sha256")
        != baseline_api.MODEL_ARTIFACT_SET_SHA256
        or baseline_binding.get("mobile_config_sha256")
        != baseline_api.MOBILE_CONFIG_SHA256
        or baseline_binding.get("model_total_size_bytes")
        != baseline_api.MODEL_TOTAL_SIZE_BYTES
        or candidate_binding.get("artifact_set_sha256") != CANDIDATE_ARTIFACT_SET_SHA256
        or candidate_binding.get("mobile_config_sha256")
        != CANDIDATE_MOBILE_CONFIG_SHA256
        or candidate_binding.get("model_total_size_bytes")
        != CANDIDATE_MODEL_TOTAL_SIZE_BYTES
    ):
        raise MobileAsrTwoPassEvalError()

    baseline = value.get("baseline")
    if (
        not isinstance(baseline, Mapping)
        or set(baseline) != {"components", "endpoint_integrity", "role"}
        or baseline.get("role") != "reset-committed-endpoint-epoch-segmentation"
    ):
        raise MobileAsrTwoPassEvalError()
    baseline_rows = _validate_component_rows(baseline.get("components"), baseline=True)
    baseline_endpoint_integrity = baseline.get("endpoint_integrity")
    baseline_api._validate_endpoint(
        baseline_endpoint_integrity,
        evaluations=EXPECTED_SOURCE_EVALUATIONS,
    )
    if production_inputs:
        contract = {
            "components": list(baseline_rows),
            "execution": {
                "components": 5,
                "endpoint_integrity": baseline["endpoint_integrity"],
                "evaluations": EXPECTED_SOURCE_EVALUATIONS,
                "logical_cases": EXPECTED_LOGICAL_CASES,
                "pcm_bytes": EXPECTED_SOURCE_PCM_BYTES,
                "pcm_inputs": EXPECTED_SOURCE_EVALUATIONS,
                "samples": EXPECTED_SOURCE_SAMPLES,
                "stream": {
                    "chunk_samples": 1_600,
                    "pace": "burst",
                    "partial_report_cadence_ms": 100,
                    "protocol_version": MOBILE_ZIPFORMER_ENDPOINT_PROTOCOL_VERSION,
                    "repeats": 1,
                    "tail_padding_samples": 48_000,
                },
            },
        }
        if (
            _canonical_sha256(contract) != ACCEPTED_BASELINE_AGGREGATE_SHA256
            or baseline.get("endpoint_integrity") != _EXPECTED_BASELINE_ENDPOINTS
        ):
            raise MobileAsrTwoPassEvalError()

    candidate = value.get("candidate")
    if (
        not isinstance(candidate, Mapping)
        or set(candidate) != {"authority", "components", "role"}
        or candidate.get("authority") != _CANDIDATE_AUTHORITY
        or candidate.get("role") != "revision-text-only-on-baseline-epochs"
    ):
        raise MobileAsrTwoPassEvalError()
    candidate_rows = _validate_component_rows(
        candidate.get("components"), baseline=False
    )
    for baseline_row, candidate_row in zip(baseline_rows, candidate_rows, strict=True):
        endpoint = baseline_row.get("endpoint_integrity")
        accounting = candidate_row.get("epoch_text_accounting")
        if (
            not isinstance(endpoint, Mapping)
            or not isinstance(accounting, Mapping)
            or accounting.get("inherited_endpoint_epochs")
            != endpoint.get("committed_endpoints")
            or accounting.get("source_evaluations") != endpoint.get("evaluations")
        ):
            raise MobileAsrTwoPassEvalError()

    comparison = value.get("comparison")
    comparison_fields = {
        "baseline_empty_epochs",
        "candidate_empty_epochs",
        "candidate_empty_on_nonempty_baseline_epochs",
        "candidate_nonempty_on_empty_baseline_epochs",
        "control_lexeme_projection",
        "endpoint_epochs",
        "exact_normalized_epoch_agreements",
        "packet_command_assertion_disagreement",
    }
    if not isinstance(comparison, Mapping) or set(comparison) != comparison_fields:
        raise MobileAsrTwoPassEvalError()
    integer_fields = comparison_fields - {
        "control_lexeme_projection",
        "packet_command_assertion_disagreement",
    }
    if any(
        type(comparison.get(key)) is not int or comparison[key] < 0
        for key in integer_fields
    ):
        raise MobileAsrTwoPassEvalError()
    endpoint_epochs = comparison["endpoint_epochs"]
    if (
        endpoint_epochs > EXPECTED_ENDPOINT_EPOCHS
        or bindings["current_rerun_endpoint_epoch_geometry_count"] != endpoint_epochs
        or not isinstance(baseline_endpoint_integrity, Mapping)
        or comparison["baseline_empty_epochs"]
        != baseline_endpoint_integrity.get("empty_committed_endpoints")
        or comparison["baseline_empty_epochs"] > endpoint_epochs
        or comparison["candidate_empty_epochs"] > endpoint_epochs
        or comparison["exact_normalized_epoch_agreements"] > endpoint_epochs
        or comparison["candidate_nonempty_on_empty_baseline_epochs"]
        > comparison["baseline_empty_epochs"]
        or comparison["candidate_empty_on_nonempty_baseline_epochs"]
        > endpoint_epochs - comparison["baseline_empty_epochs"]
        or comparison["candidate_empty_epochs"]
        - comparison["candidate_empty_on_nonempty_baseline_epochs"]
        != comparison["baseline_empty_epochs"]
        - comparison["candidate_nonempty_on_empty_baseline_epochs"]
        or comparison["exact_normalized_epoch_agreements"]
        < comparison["baseline_empty_epochs"]
        - comparison["candidate_nonempty_on_empty_baseline_epochs"]
        or (
            production_inputs
            and (
                endpoint_epochs != EXPECTED_ENDPOINT_EPOCHS
                or comparison["baseline_empty_epochs"] != 59
            )
        )
    ):
        raise MobileAsrTwoPassEvalError()

    assertion = comparison.get("packet_command_assertion_disagreement")
    assertion_fields = {
        "evaluations",
        "forbidden_attempts",
        "forbidden_exact_agreements",
        "forbidden_exact_disagreements",
        "forbidden_gained_hits",
        "forbidden_lost_hits",
        "routing_authority",
        "target_attempts",
        "target_exact_agreements",
        "target_exact_disagreements",
        "target_gained_hits",
        "target_lost_hits",
    }
    if (
        not isinstance(assertion, Mapping)
        or set(assertion) != assertion_fields
        or assertion.get("routing_authority") is not False
        or any(
            type(assertion.get(key)) is not int or assertion[key] < 0
            for key in assertion_fields - {"routing_authority"}
        )
        or assertion["evaluations"] != 57
        or (production_inputs and assertion["target_attempts"] != 26)
        or (production_inputs and assertion["forbidden_attempts"] != 355)
        or assertion["target_exact_agreements"]
        + assertion["target_exact_disagreements"]
        != assertion["target_attempts"]
        or assertion["forbidden_exact_agreements"]
        + assertion["forbidden_exact_disagreements"]
        != assertion["forbidden_attempts"]
        or assertion["target_gained_hits"] + assertion["target_lost_hits"]
        != assertion["target_exact_disagreements"]
        or assertion["forbidden_gained_hits"] + assertion["forbidden_lost_hits"]
        != assertion["forbidden_exact_disagreements"]
    ):
        raise MobileAsrTwoPassEvalError()
    baseline_command_metrics = baseline_rows[0].get("metrics")
    candidate_command_metrics = candidate_rows[0].get("metrics")
    if (
        baseline_rows[0].get("component") != "command-noise"
        or candidate_rows[0].get("component") != "command-noise"
        or not isinstance(baseline_command_metrics, Mapping)
        or not isinstance(candidate_command_metrics, Mapping)
        or assertion["evaluations"] != baseline_rows[0].get("evaluations")
        or assertion["evaluations"] != candidate_rows[0].get("evaluations")
        or assertion["target_attempts"]
        != baseline_command_metrics.get("command_attempts")
        or assertion["target_attempts"]
        != candidate_command_metrics.get("command_attempts")
        or assertion["forbidden_attempts"]
        != baseline_command_metrics.get("forbidden_target_attempts")
        or assertion["forbidden_attempts"]
        != candidate_command_metrics.get("forbidden_target_attempts")
        or candidate_command_metrics.get("command_hits")
        != baseline_command_metrics.get("command_hits")
        + assertion["target_gained_hits"]
        - assertion["target_lost_hits"]
        or candidate_command_metrics.get("forbidden_target_hits")
        != baseline_command_metrics.get("forbidden_target_hits")
        + assertion["forbidden_gained_hits"]
        - assertion["forbidden_lost_hits"]
    ):
        raise MobileAsrTwoPassEvalError()
    control = comparison.get("control_lexeme_projection")
    control_fields = {
        "actual_agent_decision_authority",
        "candidate_gained_control_lexeme_epochs",
        "candidate_lost_control_lexeme_epochs",
        "changed_nonempty_control_class_epochs",
        "endpoint_epochs",
        "exact_class_agreements",
        "exact_class_disagreements",
        "exact_projection_agreements",
        "exact_projection_disagreements",
        "mode_alias_disagreements",
        "pending_confirmation_state_applied",
        "protocol",
        "transition_matrix",
    }
    if (
        not isinstance(control, Mapping)
        or set(control) != control_fields
        or control.get("actual_agent_decision_authority") is not False
        or control.get("pending_confirmation_state_applied") is not False
        or control.get("protocol") != "mobile-text-control-lexeme-projection-v1"
        or any(
            type(control.get(key)) is not int or control[key] < 0
            for key in {
                "candidate_gained_control_lexeme_epochs",
                "candidate_lost_control_lexeme_epochs",
                "changed_nonempty_control_class_epochs",
                "endpoint_epochs",
                "exact_class_agreements",
                "exact_class_disagreements",
                "exact_projection_agreements",
                "exact_projection_disagreements",
                "mode_alias_disagreements",
            }
        )
        or control["endpoint_epochs"] != endpoint_epochs
        or control["exact_class_agreements"] + control["exact_class_disagreements"]
        != endpoint_epochs
        or control["exact_projection_agreements"]
        + control["exact_projection_disagreements"]
        != endpoint_epochs
        or control["mode_alias_disagreements"]
        > control["exact_projection_disagreements"]
        or control["exact_projection_disagreements"]
        != control["exact_class_disagreements"] + control["mode_alias_disagreements"]
    ):
        raise MobileAsrTwoPassEvalError()
    matrix = control.get("transition_matrix")
    if not isinstance(matrix, list) or len(matrix) != len(_CONTROL_CLASSES):
        raise MobileAsrTwoPassEvalError()
    matrix_total = matrix_diagonal = 0
    normalized_matrix: dict[str, Mapping[str, object]] = {}
    for row, expected_class in zip(matrix, _CONTROL_CLASSES, strict=True):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"baseline_class", "candidate_counts"}
            or row.get("baseline_class") != expected_class
        ):
            raise MobileAsrTwoPassEvalError()
        counts = row.get("candidate_counts")
        if (
            not isinstance(counts, Mapping)
            or set(counts) != set(_CONTROL_CLASSES)
            or any(
                type(counts.get(key)) is not int or counts[key] < 0 for key in counts
            )
        ):
            raise MobileAsrTwoPassEvalError()
        normalized_matrix[expected_class] = counts
        matrix_total += sum(int(counts[key]) for key in _CONTROL_CLASSES)
        matrix_diagonal += int(counts[expected_class])
    if (
        matrix_total != endpoint_epochs
        or matrix_diagonal != control["exact_class_agreements"]
        or sum(
            int(normalized_matrix["none"][key])
            for key in _CONTROL_CLASSES
            if key != "none"
        )
        != control["candidate_gained_control_lexeme_epochs"]
        or sum(
            int(normalized_matrix[key]["none"])
            for key in _CONTROL_CLASSES
            if key != "none"
        )
        != control["candidate_lost_control_lexeme_epochs"]
        or sum(
            int(normalized_matrix[left][right])
            for left in _CONTROL_CLASSES
            for right in _CONTROL_CLASSES
            if left != "none" and right != "none" and left != right
        )
        != control["changed_nonempty_control_class_epochs"]
    ):
        raise MobileAsrTwoPassEvalError()

    accounting_rows = [row["epoch_text_accounting"] for row in candidate_rows]
    if any(not isinstance(row, Mapping) for row in accounting_rows):
        raise MobileAsrTwoPassEvalError()
    if (
        sum(int(row["inherited_endpoint_epochs"]) for row in accounting_rows)
        != endpoint_epochs
        or sum(int(row["empty_candidate_text_epochs"]) for row in accounting_rows)
        != comparison["candidate_empty_epochs"]
        or sum(int(row["nonempty_candidate_text_epochs"]) for row in accounting_rows)
        != endpoint_epochs - comparison["candidate_empty_epochs"]
        or sum(int(row["source_evaluations"]) for row in accounting_rows)
        != EXPECTED_SOURCE_EVALUATIONS
    ):
        raise MobileAsrTwoPassEvalError()

    execution = value.get("execution")
    execution_fields = {
        "baseline_stream",
        "baseline_worker_stderr",
        "candidate_epoch_geometry",
        "candidate_stream",
        "candidate_worker_stderr",
        "components",
        "logical_cases",
        "source_pcm_bytes",
        "source_pcm_inputs",
        "source_samples",
        "workers_sequential",
    }
    if not isinstance(execution, Mapping) or set(execution) != execution_fields:
        raise MobileAsrTwoPassEvalError()
    if (
        execution.get("components") != 5
        or execution.get("logical_cases") != EXPECTED_LOGICAL_CASES
        or execution.get("source_pcm_inputs") != EXPECTED_SOURCE_EVALUATIONS
        or type(execution.get("source_pcm_bytes")) is not int
        or type(execution.get("source_samples")) is not int
        or execution["source_pcm_bytes"] != execution["source_samples"] * 4
        or execution.get("workers_sequential") is not True
        or execution.get("baseline_stream")
        != {
            "chunk_samples": 1_600,
            "pace": "burst",
            "partial_report_cadence_ms": 100,
            "protocol_version": MOBILE_ZIPFORMER_ENDPOINT_PROTOCOL_VERSION,
            "repeats": 1,
            "tail_padding_samples": 48_000,
        }
        or execution.get("candidate_stream")
        != {
            "chunk_samples": 1_600,
            "pace": "burst",
            "partial_report_cadence_ms": 100,
            "protocol_version": FINAL_PROTOCOL_VERSION,
            "repeats": 1,
            "tail_padding_samples": 0,
        }
        or (
            production_inputs
            and (
                execution["source_pcm_bytes"] != EXPECTED_SOURCE_PCM_BYTES
                or execution["source_samples"] != EXPECTED_SOURCE_SAMPLES
            )
        )
    ):
        raise MobileAsrTwoPassEvalError()
    _worker_stderr(execution.get("baseline_worker_stderr"))  # type: ignore[arg-type]
    _worker_stderr(execution.get("candidate_worker_stderr"))  # type: ignore[arg-type]
    geometry = execution.get("candidate_epoch_geometry")
    if (
        not isinstance(geometry, Mapping)
        or set(geometry)
        != {
            "baseline_endpoint_observations",
            "candidate_app_admissions",
            "candidate_requests",
            "candidate_worker_lifetime_pcm_limit_bytes",
            "derived_epoch_pcm_bytes_total",
            "derived_epoch_samples_total",
            "empty_baseline_epochs_create_requests",
            "maximum_derived_epoch_pcm_bytes_total",
            "maximum_derived_epoch_samples_total",
            "preserves_baseline_order_and_boundaries",
            "reported_adapter_or_evaluator_appended_pcm_padding_samples",
            "sherpa_internal_tail_padding_observation",
            "source_evaluations",
        }
        or geometry.get("baseline_endpoint_observations") != endpoint_epochs
        or geometry.get("candidate_app_admissions") != 0
        or geometry.get("candidate_requests") != endpoint_epochs
        or geometry.get("candidate_worker_lifetime_pcm_limit_bytes")
        != CANDIDATE_WORKER_LIFETIME_PCM_LIMIT_BYTES
        or geometry.get("empty_baseline_epochs_create_requests") is not True
        or geometry.get("preserves_baseline_order_and_boundaries") is not True
        or geometry.get("reported_adapter_or_evaluator_appended_pcm_padding_samples")
        != 0
        or geometry.get("sherpa_internal_tail_padding_observation")
        != "unobserved-and-unreported"
        or geometry.get("source_evaluations") != EXPECTED_SOURCE_EVALUATIONS
        or geometry.get("maximum_derived_epoch_pcm_bytes_total")
        != MAXIMUM_DERIVED_EPOCH_PCM_BYTES
        or geometry.get("maximum_derived_epoch_samples_total")
        != MAXIMUM_DERIVED_EPOCH_SAMPLES
        or geometry["maximum_derived_epoch_pcm_bytes_total"]
        > geometry["candidate_worker_lifetime_pcm_limit_bytes"]
        or type(geometry.get("derived_epoch_pcm_bytes_total")) is not int
        or not 0
        <= geometry["derived_epoch_pcm_bytes_total"]
        <= MAXIMUM_DERIVED_EPOCH_PCM_BYTES
        or geometry["derived_epoch_pcm_bytes_total"]
        > CANDIDATE_WORKER_LIFETIME_PCM_LIMIT_BYTES
        or geometry.get("derived_epoch_samples_total")
        != geometry["derived_epoch_pcm_bytes_total"] // 4
        or geometry["derived_epoch_pcm_bytes_total"] % 4
    ):
        raise MobileAsrTwoPassEvalError()

    evidence = value.get("evidence_scope")
    expected_evidence = {
        **_GENERAL_EVIDENCE_SCOPE,
        "baseline_endpoint_epoch_boundary_authority": production_inputs,
        "semantic_turn_boundary_authority": False,
        "candidate_model_executed": production_inputs,
        "candidate_revision_text_evidence": production_inputs,
        "production_packet": production_inputs,
        "unpooled_component_result_authority": production_inputs,
    }
    if evidence != expected_evidence:
        raise MobileAsrTwoPassEvalError()

    forbidden_exact_keys = {
        "case_id",
        "endpoint_text",
        "hypothesis",
        "path",
        "reference",
        "request_id",
        "selected_text",
        "speaker",
        "transcript",
    }

    def walk(item: object) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                if key in forbidden_exact_keys or type(key) is not str:
                    raise MobileAsrTwoPassEvalError()
                walk(nested)
        elif isinstance(item, list):
            for nested in item:
                walk(nested)
        elif type(item) is float and not math.isfinite(item):
            raise MobileAsrTwoPassEvalError()

    walk(value)
    if len(_canonical_json(value, newline=True)) > _MAX_REPORT_BYTES:
        raise MobileAsrTwoPassEvalError()
    return value


def _load_report_common(
    path: Path | str,
    *,
    expected_sha256: str,
    production_inputs: bool,
) -> LoadedMobileAsrTwoPassEvalReport:
    try:
        expected = _sha256(expected_sha256)
        snapshot = read_regular_bounded(Path(path), maximum_bytes=_MAX_REPORT_BYTES)
        metadata = snapshot.path.lstat()
        if (
            stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_nlink != 1
            or metadata.st_uid != os.getuid()
            or hashlib.sha256(snapshot.data).hexdigest() != expected
        ):
            raise MobileAsrTwoPassEvalError()
        value = _strict_json(snapshot.data)
        if _canonical_json(value, newline=True) != snapshot.data:
            raise MobileAsrTwoPassEvalError()
        report = _validate_report(value, production_inputs=production_inputs)
        bindings = report["bindings"]
        if not isinstance(bindings, Mapping):
            raise MobileAsrTwoPassEvalError()
        baseline = bindings["baseline_worker"]
        candidate = bindings["candidate_worker"]
        if not isinstance(baseline, Mapping) or not isinstance(candidate, Mapping):
            raise MobileAsrTwoPassEvalError()
        loaded = LoadedMobileAsrTwoPassEvalReport(
            path=snapshot.path,
            digest=expected,
            packet_index_sha256=str(bindings["packet_index_sha256"]),
            packet_receipt_sha256=str(bindings["packet_receipt_sha256"]),
            baseline_model_manifest_sha256=str(baseline["model_manifest_sha256"]),
            candidate_model_manifest_sha256=str(candidate["model_manifest_sha256"]),
            source_bundle_sha256=str(baseline["source_bundle_sha256"]),
            value=report,
            snapshot=_snapshot(metadata),
        )
        closing = read_regular_bounded(
            loaded.path,
            maximum_bytes=_MAX_REPORT_BYTES,
            expected_bytes=len(snapshot.data),
        )
        if (
            closing.data != snapshot.data
            or _snapshot(loaded.path.lstat()) != loaded.snapshot
        ):
            raise MobileAsrTwoPassEvalError()
        return loaded
    except MobileAsrTwoPassEvalError:
        raise
    except Exception:
        raise MobileAsrTwoPassEvalError() from None


def load_mobile_asr_two_pass_eval_report(
    path: Path | str,
    *,
    expected_sha256: str,
    packet_index_sha256: str,
    packet_receipt_sha256: str,
    baseline_model_manifest_sha256: str,
    candidate_model_manifest_sha256: str,
) -> LoadedMobileAsrTwoPassEvalReport:
    loaded = _load_report_common(
        path, expected_sha256=expected_sha256, production_inputs=True
    )
    if (
        loaded.packet_index_sha256 != _sha256(packet_index_sha256)
        or loaded.packet_receipt_sha256 != _sha256(packet_receipt_sha256)
        or loaded.baseline_model_manifest_sha256
        != _sha256(baseline_model_manifest_sha256)
        or loaded.candidate_model_manifest_sha256
        != _sha256(candidate_model_manifest_sha256)
    ):
        raise MobileAsrTwoPassEvalError()
    return loaded


def verify_mobile_asr_two_pass_eval_report(
    loaded: LoadedMobileAsrTwoPassEvalReport,
) -> None:
    if type(loaded) is not LoadedMobileAsrTwoPassEvalReport:
        raise MobileAsrTwoPassEvalError()
    current = load_mobile_asr_two_pass_eval_report(
        loaded.path,
        expected_sha256=loaded.digest,
        packet_index_sha256=loaded.packet_index_sha256,
        packet_receipt_sha256=loaded.packet_receipt_sha256,
        baseline_model_manifest_sha256=loaded.baseline_model_manifest_sha256,
        candidate_model_manifest_sha256=loaded.candidate_model_manifest_sha256,
    )
    if current != loaded or _snapshot(loaded.path.lstat()) != loaded.snapshot:
        raise MobileAsrTwoPassEvalError()


def _ready_matches(
    ready: object,
    *,
    protocol_version: int,
    model_id: str,
    adapter: str,
    manifest_sha256: str,
    source_bundle: SourceBundle,
) -> bool:
    return (
        getattr(ready, "protocol_version", None) == protocol_version
        and getattr(ready, "model_id", None) == model_id
        and getattr(ready, "adapter", None) == adapter
        and getattr(ready, "manifest_sha256", None) == manifest_sha256
        and getattr(ready, "source_bundle_sha256", None) == source_bundle.tree_sha256
    )


def _candidate_venv_root(manifest: object) -> Path:
    """Derive the complete candidate venv from its exact ``bin/python`` path."""

    try:
        raw_path = getattr(getattr(manifest, "python"), "path")
        if not isinstance(raw_path, Path):
            raise MobileAsrTwoPassEvalError()
        python_path = raw_path
        if (
            not python_path.is_absolute()
            or python_path.name != "python"
            or python_path.parent.name != "bin"
            or python_path != python_path.parent.parent / "bin" / "python"
        ):
            raise MobileAsrTwoPassEvalError()
        venv_root = python_path.parent.parent.resolve(strict=True)
        bin_root = python_path.parent.resolve(strict=True)
        if (
            venv_root == venv_root.parent
            or not venv_root.is_dir()
            or not bin_root.is_dir()
            or bin_root.parent != venv_root
            or not python_path.is_file()
        ):
            raise MobileAsrTwoPassEvalError()
        python_path.resolve(strict=True)
        return venv_root
    except MobileAsrTwoPassEvalError:
        raise
    except Exception:
        raise MobileAsrTwoPassEvalError() from None


def _validate_destinations(
    inputs: baseline_api._Inputs,
    baseline_model: baseline_api._ModelAuthority,
    candidate_model: _CandidateAuthority,
    *,
    scratch_root: Path | str,
    output_path: Path | str,
) -> tuple[Path, Path, tuple[int, ...], tuple[int, ...]]:
    try:
        scratch, output, scratch_parent, output_parent = (
            baseline_api._validate_destinations(
                inputs,
                baseline_model,
                scratch_root=scratch_root,
                output_path=output_path,
            )
        )
        candidate_python = Path(getattr(candidate_model.manifest, "python").path)
        candidate_roots = {
            candidate_model.manifest_path.parent.resolve(strict=True),
            _candidate_venv_root(candidate_model.manifest),
            candidate_python.resolve(strict=True).parent,
            Path(getattr(candidate_model.manifest, "worker").path)
            .resolve(strict=True)
            .parent,
            *(
                Path(artifact.path).resolve(strict=True).parent
                for artifact in getattr(candidate_model.manifest, "artifacts")
            ),
        }
        for root in candidate_roots:
            if baseline_api._paths_overlap(
                scratch, root
            ) or baseline_api._paths_overlap(output.parent, root):
                raise MobileAsrTwoPassEvalError()
        return scratch, output, scratch_parent, output_parent
    except MobileAsrTwoPassEvalError:
        raise
    except Exception:
        raise MobileAsrTwoPassEvalError() from None


def preflight_mobile_asr_two_pass_eval(
    packet_root: Path | str,
    *,
    packet_index_sha256: str,
    packet_receipt_sha256: str,
    baseline_worker_manifest: Path | str,
    candidate_worker_manifest: Path | str,
    scratch_root: Path | str,
    output_path: Path | str,
) -> dict[str, object]:
    try:
        inputs = baseline_api._load_production_inputs(
            packet_root,
            packet_index_sha256=packet_index_sha256,
            packet_receipt_sha256=packet_receipt_sha256,
        )
        baseline_model = baseline_api._load_model_authority(baseline_worker_manifest)
        candidate_model = _load_candidate_authority(candidate_worker_manifest)
        _validate_destinations(
            inputs,
            baseline_model,
            candidate_model,
            scratch_root=scratch_root,
            output_path=output_path,
        )
        _verify_control_projection_sources()
        closure = _evaluator_closure_sha256()
        if inputs.packet is None:
            raise MobileAsrTwoPassEvalError()
        baseline_api.packet_api.verify_mobile_asr_evidence_packet(inputs.packet)
        baseline_api._verify_model_authority(baseline_model)
        _verify_candidate_authority(candidate_model)
        return {
            "candidate_model_manifest_sha256": candidate_model.manifest_sha256,
            "components": 5,
            "evaluator_closure_sha256": closure,
            "kind": "mobile-asr-two-pass-eval-preflight-v1",
            "logical_cases": EXPECTED_LOGICAL_CASES,
            "ok": True,
            "packet_index_sha256": inputs.packet_index_sha256,
            "packet_lock_sha256": inputs.packet_lock_sha256,
            "packet_receipt_sha256": inputs.packet_receipt_sha256,
            "pcm_bytes": inputs.pcm_bytes,
            "pcm_inputs": EXPECTED_SOURCE_EVALUATIONS,
            "production_inputs": True,
            "samples": inputs.samples,
            "baseline_model_manifest_sha256": baseline_model.manifest_sha256,
        }
    except MobileAsrTwoPassEvalError:
        raise
    except Exception:
        raise MobileAsrTwoPassEvalError() from None


def run_mobile_asr_two_pass_eval(
    packet_root: Path | str,
    *,
    packet_index_sha256: str,
    packet_receipt_sha256: str,
    baseline_worker_manifest: Path | str,
    candidate_worker_manifest: Path | str,
    scratch_root: Path | str,
    output_path: Path | str,
) -> LoadedMobileAsrTwoPassEvalReport:
    """Run the fixed baseline-first, revision-only endpoint-epoch comparison."""

    state = baseline_api._CommitState()
    try:
        inputs = baseline_api._load_production_inputs(
            packet_root,
            packet_index_sha256=packet_index_sha256,
            packet_receipt_sha256=packet_receipt_sha256,
        )
        baseline_model = baseline_api._load_model_authority(baseline_worker_manifest)
        candidate_model = _load_candidate_authority(candidate_worker_manifest)
        scratch, output, scratch_parent_snapshot, output_parent_snapshot = (
            _validate_destinations(
                inputs,
                baseline_model,
                candidate_model,
                scratch_root=scratch_root,
                output_path=output_path,
            )
        )
        _verify_control_projection_sources()
        closure = _evaluator_closure_sha256()
        worker_root, pcm_paths, source_bundle, scratch_snapshot = (
            baseline_api._prepare_scratch(scratch, inputs)
        )
        if _snapshot(scratch.parent.lstat())[:5] != scratch_parent_snapshot[:5]:
            raise MobileAsrTwoPassEvalError()
        scratch_parent_snapshot = _snapshot(scratch.parent.lstat())

        from tools import streaming_stt_eval
        from tools.streaming_stt.supervisor import StreamingWorker

        run_lock = streaming_stt_eval._BenchmarkRunLock.acquire(
            streaming_stt_eval._HOST_LOCK_PATH
        )
        baseline_worker = None
        candidate_worker = None
        try:
            run_lock.assert_bound()
            baseline_worker = StreamingWorker(
                baseline_model.manifest, worker_root, source_bundle
            )
            ready = baseline_worker.start()
            if not _ready_matches(
                ready,
                protocol_version=MOBILE_ZIPFORMER_ENDPOINT_PROTOCOL_VERSION,
                model_id=BASELINE_MODEL_ID,
                adapter=BASELINE_ADAPTER,
                manifest_sha256=baseline_model.manifest_sha256,
                source_bundle=source_bundle,
            ):
                raise MobileAsrTwoPassEvalError()
            leaves = tuple(
                leaf for component in inputs.components for leaf in component.leaves
            )
            evaluations: list[_BaselineEvaluation] = []
            for index, leaf in enumerate(leaves):
                request = TranscribeRequest(
                    request_id=f"baseline-{index:03d}",
                    pcm=PcmInput(
                        path=pcm_paths[index].resolve(strict=True),
                        sha256=leaf.sha256,
                        samples=leaf.samples,
                    ),
                    stream=_BASELINE_STREAM,
                    protocol_version=MOBILE_ZIPFORMER_ENDPOINT_PROTOCOL_VERSION,
                )
                evaluations.append(
                    _extract_baseline_evaluation(
                        baseline_worker.transcribe(request), leaf
                    )
                )
            baseline_worker.close()
            if not baseline_worker.fully_stopped or not _ready_matches(
                baseline_worker.ready,
                protocol_version=MOBILE_ZIPFORMER_ENDPOINT_PROTOCOL_VERSION,
                model_id=BASELINE_MODEL_ID,
                adapter=BASELINE_ADAPTER,
                manifest_sha256=baseline_model.manifest_sha256,
                source_bundle=source_bundle,
            ):
                raise MobileAsrTwoPassEvalError()
            baseline_stderr = dict(baseline_worker.stderr_summary)
            baseline_worker = None

            baseline_outcomes = tuple(evaluation.outcome for evaluation in evaluations)
            baseline_components = _reduce(
                inputs, baseline_outcomes, include_endpoint_integrity=True
            )
            baseline_endpoints = baseline_api._endpoint_totals(baseline_outcomes)
            baseline_aggregate_sha256 = _require_accepted_baseline(
                baseline_components, baseline_endpoints, inputs
            )

            baseline_api._verify_scratch(
                scratch, scratch_snapshot, source_bundle, pcm_paths, inputs
            )
            epoch_root = scratch / "candidate-worker"
            epochs = _write_epoch_files(epoch_root, inputs, evaluations)
            if len(epochs) != EXPECTED_ENDPOINT_EPOCHS:
                raise MobileAsrTwoPassEvalError()
            scratch_snapshot = _advance_scratch_snapshot_for_candidate(
                scratch, scratch_snapshot
            )
            baseline_api._verify_scratch(
                scratch, scratch_snapshot, source_bundle, pcm_paths, inputs
            )
            _verify_two_pass_scratch_snapshot(scratch, scratch_snapshot)
            _verify_epochs(epoch_root, inputs, evaluations, epochs)

            # The candidate is constructed only after baseline close and exact
            # ADR-0207 aggregate acceptance.
            candidate_worker = StreamingWorker(
                candidate_model.manifest, epoch_root, source_bundle
            )
            ready = candidate_worker.start()
            if not _ready_matches(
                ready,
                protocol_version=FINAL_PROTOCOL_VERSION,
                model_id=CANDIDATE_MODEL_ID,
                adapter=CANDIDATE_ADAPTER,
                manifest_sha256=candidate_model.manifest_sha256,
                source_bundle=source_bundle,
            ):
                raise MobileAsrTwoPassEvalError()
            candidate_texts: list[str] = []
            for index, epoch in enumerate(epochs):
                request_id = f"candidate-{index:03d}"
                request = TranscribeRequest(
                    request_id=request_id,
                    pcm=PcmInput(
                        path=epoch.path.resolve(strict=True),
                        sha256=epoch.sha256,
                        samples=epoch.samples,
                    ),
                    stream=_CANDIDATE_STREAM,
                    protocol_version=FINAL_PROTOCOL_VERSION,
                )
                candidate_texts.append(
                    _extract_candidate_text(
                        candidate_worker.transcribe(request), epoch, request_id
                    )
                )
            candidate_worker.close()
            if not candidate_worker.fully_stopped or not _ready_matches(
                candidate_worker.ready,
                protocol_version=FINAL_PROTOCOL_VERSION,
                model_id=CANDIDATE_MODEL_ID,
                adapter=CANDIDATE_ADAPTER,
                manifest_sha256=candidate_model.manifest_sha256,
                source_bundle=source_bundle,
            ):
                raise MobileAsrTwoPassEvalError()
            candidate_stderr = dict(candidate_worker.stderr_summary)
            candidate_worker = None
        finally:
            for worker in (candidate_worker, baseline_worker):
                if worker is not None:
                    try:
                        worker.close()
                    except Exception:
                        pass
            run_lock.close()

        evaluations_tuple = tuple(evaluations)
        candidate_outcomes = _candidate_outcomes(evaluations_tuple, candidate_texts)
        candidate_components = _reduce(
            inputs, candidate_outcomes, include_endpoint_integrity=False
        )
        comparison = _comparison(inputs, baseline_outcomes, candidate_outcomes)
        derived_epoch_pcm_bytes_total = sum(epoch.samples * 4 for epoch in epochs)
        report = _report(
            inputs,
            baseline_evaluations=evaluations_tuple,
            baseline_components=baseline_components,
            candidate_components=candidate_components,
            baseline_endpoints=baseline_endpoints,
            candidate_outcomes=candidate_outcomes,
            comparison=comparison,
            baseline_binding=baseline_api._worker_binding(
                baseline_model, source_bundle
            ),
            candidate_binding=_candidate_binding(candidate_model, source_bundle),
            evaluator_closure_sha256=closure,
            baseline_aggregate_sha256=baseline_aggregate_sha256,
            derived_epoch_pcm_bytes_total=derived_epoch_pcm_bytes_total,
            baseline_stderr=baseline_stderr,
            candidate_stderr=candidate_stderr,
            model_executed=True,
        )

        def guard() -> None:
            if inputs.packet is None:
                raise MobileAsrTwoPassEvalError()
            baseline_api.packet_api.verify_mobile_asr_evidence_packet(inputs.packet)
            baseline_api._verify_model_authority(baseline_model)
            _verify_candidate_authority(candidate_model)
            _verify_control_projection_sources()
            if _evaluator_closure_sha256() != closure:
                raise MobileAsrTwoPassEvalError()
            _verify_two_pass_scratch_snapshot(scratch, scratch_snapshot)
            baseline_api._verify_scratch(
                scratch, scratch_snapshot, source_bundle, pcm_paths, inputs
            )
            _verify_epochs(epoch_root, inputs, evaluations_tuple, epochs)
            geometry_count, geometry_sha256 = _baseline_endpoint_epoch_geometry(
                evaluations_tuple
            )
            report_bindings = report.get("bindings")
            report_execution = report.get("execution")
            report_geometry = (
                report_execution.get("candidate_epoch_geometry")
                if isinstance(report_execution, Mapping)
                else None
            )
            if (
                not isinstance(report_bindings, Mapping)
                or not isinstance(report_geometry, Mapping)
                or report_bindings.get("current_rerun_endpoint_epoch_geometry_count")
                != geometry_count
                or report_bindings.get("current_rerun_endpoint_epoch_geometry_sha256")
                != geometry_sha256
                or report_geometry.get("derived_epoch_pcm_bytes_total")
                != sum(epoch.samples * 4 for epoch in epochs)
            ):
                raise MobileAsrTwoPassEvalError()
            verify_source_bundle(source_bundle, required_files=WORKER_SOURCE_FILES)
            if (
                _snapshot(scratch.parent.lstat()) != scratch_parent_snapshot
                or _snapshot(output.parent.lstat()) != output_parent_snapshot
            ):
                raise MobileAsrTwoPassEvalError()
            _validate_report(report, production_inputs=True)

        digest = baseline_api._publish_report(output, report, guard=guard, state=state)
        loaded = load_mobile_asr_two_pass_eval_report(
            output,
            expected_sha256=digest,
            packet_index_sha256=inputs.packet_index_sha256,
            packet_receipt_sha256=inputs.packet_receipt_sha256,
            baseline_model_manifest_sha256=baseline_model.manifest_sha256,
            candidate_model_manifest_sha256=candidate_model.manifest_sha256,
        )
        verify_mobile_asr_two_pass_eval_report(loaded)
        return loaded
    except BaseException as error:
        if state.committed:
            try:
                return load_mobile_asr_two_pass_eval_report(
                    output_path,
                    expected_sha256=state.digest,
                    packet_index_sha256=packet_index_sha256,
                    packet_receipt_sha256=packet_receipt_sha256,
                    baseline_model_manifest_sha256=getattr(
                        locals().get("baseline_model"), "manifest_sha256"
                    ),
                    candidate_model_manifest_sha256=getattr(
                        locals().get("candidate_model"), "manifest_sha256"
                    ),
                )
            except Exception:
                pass
        if isinstance(error, KeyboardInterrupt):
            raise
        if isinstance(error, MobileAsrTwoPassEvalError):
            raise
        raise MobileAsrTwoPassEvalError() from None


def _synthetic_binding(authority: str) -> dict[str, object]:
    return {
        "adapter": "synthetic-test-only",
        "artifact_set_sha256": authority,
        "mobile_config_sha256": authority,
        "model_id": "synthetic-test-only",
        "model_manifest_sha256": authority,
        "model_total_size_bytes": 1,
        "provision_receipt_sha256": authority,
        "source_bundle_files": 1,
        "source_bundle_sha256": authority,
    }


def _run_synthetic_test_evaluation(
    baseline_outcomes: tuple[baseline_api._EndpointOutcome, ...],
    candidate_texts: tuple[str, ...],
    output_path: Path | str,
) -> LoadedMobileAsrTwoPassEvalReport:
    """Exercise aggregate/privacy/publication paths without model authority."""

    inputs = baseline_api._synthetic_inputs()
    if len(baseline_outcomes) != EXPECTED_SOURCE_EVALUATIONS or any(
        type(outcome) is not baseline_api._EndpointOutcome
        for outcome in baseline_outcomes
    ):
        raise MobileAsrTwoPassEvalError()
    evaluations = tuple(
        _BaselineEvaluation(
            outcome=outcome,
            observations=tuple(
                MobileZipformerEndpointObservation(
                    text=text,
                    observed_sample=index + 1,
                    source_complete=complete,
                )
                for index, (text, complete) in enumerate(
                    zip(
                        outcome.endpoint_texts,
                        outcome.endpoint_source_complete,
                        strict=True,
                    )
                )
            ),
        )
        for outcome in baseline_outcomes
    )
    candidate_outcomes = _candidate_outcomes(evaluations, candidate_texts)
    baseline_components = _reduce(
        inputs, baseline_outcomes, include_endpoint_integrity=True
    )
    candidate_components = _reduce(
        inputs, candidate_outcomes, include_endpoint_integrity=False
    )
    baseline_endpoints = baseline_api._endpoint_totals(baseline_outcomes)
    comparison = _comparison(inputs, baseline_outcomes, candidate_outcomes)
    baseline_contract = _canonical_sha256(
        _accepted_baseline_contract(baseline_components, baseline_endpoints, inputs)
    )
    authority = _canonical_sha256(
        {
            "baseline": [len(value.endpoint_texts) for value in baseline_outcomes],
            "candidate": [len(value.endpoint_texts) for value in candidate_outcomes],
        }
    )
    report = _report(
        inputs,
        baseline_evaluations=evaluations,
        baseline_components=baseline_components,
        candidate_components=candidate_components,
        baseline_endpoints=baseline_endpoints,
        candidate_outcomes=candidate_outcomes,
        comparison=comparison,
        baseline_binding=_synthetic_binding(authority),
        candidate_binding=_synthetic_binding(authority),
        evaluator_closure_sha256=authority,
        baseline_aggregate_sha256=baseline_contract,
        derived_epoch_pcm_bytes_total=len(candidate_texts) * 4,
        baseline_stderr={
            "bytes": 0,
            "sha256": hashlib.sha256(b"").hexdigest(),
            "truncated": False,
        },
        candidate_stderr={
            "bytes": 0,
            "sha256": hashlib.sha256(b"").hexdigest(),
            "truncated": False,
        },
        model_executed=False,
    )
    try:
        output, _parent, _parent_snapshot = baseline_api._new_synthetic_output(
            output_path
        )
        state = baseline_api._CommitState()
        digest = baseline_api._publish_report(
            output,
            report,
            guard=lambda: _validate_report(report, production_inputs=False),
            state=state,
        )
        return _load_report_common(
            output, expected_sha256=digest, production_inputs=False
        )
    except MobileAsrTwoPassEvalError:
        raise
    except Exception:
        raise MobileAsrTwoPassEvalError() from None


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise MobileAsrTwoPassEvalError()


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        description="Run fixed MobileWhisper revisions on mobile endpoint epochs."
    )
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--packet-root", type=Path, required=True)
    parser.add_argument("--packet-index-sha256", required=True)
    parser.add_argument("--packet-receipt-sha256", required=True)
    parser.add_argument("--baseline-worker-manifest", type=Path, required=True)
    parser.add_argument("--candidate-worker-manifest", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _emit_public(value: object) -> bool:
    """Attempt one bounded binary stdout write and flush without retrying."""

    try:
        raw = _canonical_json(value) + b"\n"
        if len(raw) > _MAX_PUBLIC_OUTPUT_BYTES:
            return False
        if sys.stdout.buffer.write(raw) != len(raw):
            return False
        sys.stdout.buffer.flush()
        return True
    except Exception:
        replacement = -1
        try:
            replacement = os.open(
                os.devnull,
                os.O_WRONLY | getattr(os, "O_CLOEXEC", 0),
            )
            os.dup2(
                replacement,
                sys.stdout.buffer.fileno(),
                inheritable=False,
            )
        except Exception:
            pass
        finally:
            if replacement >= 0:
                os.close(replacement)
        return False


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        kwargs = {
            "packet_index_sha256": args.packet_index_sha256,
            "packet_receipt_sha256": args.packet_receipt_sha256,
            "baseline_worker_manifest": args.baseline_worker_manifest,
            "candidate_worker_manifest": args.candidate_worker_manifest,
            "scratch_root": args.scratch_root,
            "output_path": args.output,
        }
        if args.preflight:
            result = preflight_mobile_asr_two_pass_eval(args.packet_root, **kwargs)
        else:
            loaded = run_mobile_asr_two_pass_eval(args.packet_root, **kwargs)
            result = {
                "candidate_requests": EXPECTED_ENDPOINT_EPOCHS,
                "kind": "mobile-asr-two-pass-eval-publication-v1",
                "ok": True,
                "report_sha256": loaded.digest,
                "source_evaluations": EXPECTED_SOURCE_EVALUATIONS,
            }
        return 0 if _emit_public(result) else 2
    except KeyboardInterrupt:
        return 130
    except (
        MobileAsrTwoPassEvalError,
        baseline_api.MobileAsrEvidenceEvalError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        _emit_public(_SAFE_ERROR)
        return 2


__all__ = [
    "ACCEPTED_BASELINE_AGGREGATE_SHA256",
    "ACCEPTED_BASELINE_REPORT_SHA256",
    "BASELINE_ADAPTER",
    "BASELINE_MODEL_ID",
    "CANDIDATE_ADAPTER",
    "CANDIDATE_MODEL_ID",
    "CANDIDATE_SOURCE_REVISION",
    "CANDIDATE_WORKER_LIFETIME_PCM_LIMIT_BYTES",
    "LoadedMobileAsrTwoPassEvalReport",
    "MAXIMUM_DERIVED_EPOCH_PCM_BYTES",
    "MobileAsrTwoPassEvalError",
    "load_mobile_asr_two_pass_eval_report",
    "main",
    "preflight_mobile_asr_two_pass_eval",
    "run_mobile_asr_two_pass_eval",
    "verify_mobile_asr_two_pass_eval_report",
]


if __name__ == "__main__":
    raise SystemExit(main())
