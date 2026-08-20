"""Run the exact mobile Zipformer over the five-component evidence packet.

This controller deliberately keeps the packet's five metric domains separate.
It consumes only a terminal-receipt-bound packet, a terminal-receipt-bound
mobile model provision, and a source snapshot of the worker implementation.
References and endpoint text remain process-private; the retained report is an
aggregate-only, single-link file.
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
import signal
import stat
import sys
from typing import Callable, Final, Mapping, Sequence

from core.wer import normalize, word_error_rate
from tools import prepare_mobile_asr_evidence_packet as packet_api
from tools.streaming_stt.bounded_io import (
    BoundedReadError,
    hash_regular_bounded,
    opened_directory_nofollow,
    read_regular_bounded,
)
from tools.streaming_stt.corpus import LoadedCorpus, load_corpus, verify_corpus_snapshot
from tools.streaming_stt.overlap_metrics import (
    PROTOCOL_VERSION as OVERLAP_METRIC,
    OverlapMetricResult,
    aggregate_two_utterance_min_order_wer,
    score_two_utterance_min_order_wer,
)
from tools.streaming_stt.protocol import (
    MOBILE_ZIPFORMER_ENDPOINT_PROTOCOL_VERSION,
    PcmInput,
    StreamConfig,
    TranscribeRequest,
)
from tools.streaming_stt.source_bundle import (
    WORKER_SOURCE_FILES,
    SourceBundle,
    stage_worker_source_bundle,
    verify_source_bundle,
)


SCHEMA_VERSION: Final = 1
REPORT_KIND: Final = "mobile-asr-evidence-eval-report-v1"
SYNTHETIC_REPORT_KIND: Final = "synthetic-mobile-asr-evidence-eval-report-v1"
MODEL_ID: Final = "sherpa-onnx-streaming-zipformer-en-2023-06-26-mobile-hybrid-v1"
ADAPTER: Final = "sherpa-onnx-mobile-zipformer-endpoint-v1"
PROTOCOL_VERSION: Final = MOBILE_ZIPFORMER_ENDPOINT_PROTOCOL_VERSION
PACKET_SCHEMA_VERSION: Final = 2
PACKET_ID: Final = "mobile-asr-english-five-component-v2"
PACKET_LOCK_SHA256: Final = (
    "c81a89a7a7df534ec0a23dc20c0a915c4aa5eedec6270a13ec08456a0683cc11"
)
PACKET_INDEX_SHA256: Final = (
    "232e4230d16b0014baff38e2daf4156aaef3c4e129541079248fac0b2f9e835c"
)
PACKET_RECEIPT_SHA256: Final = (
    "36eec947383069fb15bc998209ae4aaf76ea82f7031666b972fe98f8c7c5f51f"
)
MODEL_ARTIFACT_SET_SHA256: Final = (
    "eebca4036773b78642c846acdc966d1bab8f377a8b8ab632f9d3aac7ab38bcc5"
)
MODEL_TOTAL_SIZE_BYTES: Final = 74_207_237
MOBILE_CONFIG_SHA256: Final = (
    "749fa50525a08fc612bc82fb4e05b4fbfa62f3e762b621acceaf05f9223fa883"
)
_EXPECTED_MOBILE_CONFIG = {
    "core_package_version": "1.13.3",
    "debug": True,
    "decoding_method": "greedy_search",
    "enable_endpoint_detection": True,
    "feature_dim": 80,
    "language": "en",
    "max_active_paths": 4,
    "maximum_tail_padding_samples": 48_000,
    "model_type": "zipformer2",
    "native_chunk_samples": 1_600,
    "num_threads": 1,
    "numpy_version": "2.4.6",
    "package_version": "1.13.3",
    "provider": "cpu",
    "rule1_min_trailing_silence": 2.4,
    "rule2_min_trailing_silence": 0.8,
    "rule3_min_utterance_length": 20.0,
    "sample_rate": 16_000,
    "source_repo_id": ("csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26"),
    "source_revision": "672fbf1b30579d6585301139bb363f42a0ad4a24",
    "variant": "epoch-99-avg-1-chunk-16-left-128-mobile-hybrid",
}
_STREAM = StreamConfig(
    chunk_samples=1_600,
    pace="burst",
    partial_interval_ms=100,
    tail_padding_samples=48_000,
)
_EXPECTED_COMPONENTS = (
    ("command-noise", "command-assertion-v1", 57, 57),
    ("demand-noise", "stratified-wer-cer-v1", 42, 42),
    ("notsofar-far-field", "paired-channel-wer-cer-v1", 18, 18),
    ("primock-isolated", "ordinary-wer-cer-v1", 3, 3),
    ("primock-overlap", "two-utterance-min-order-wer-v1", 3, 9),
)
_DEMAND_STRATA = ("dkitchen", "dliving", "dwashing", "snr0", "snr10", "snr20")
_MAX_REPORT_BYTES = 256 * 1024
_MAX_PUBLIC_OUTPUT_BYTES = 4 * 1024
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ERROR = {
    "error": "mobile_asr_evidence_eval_prerequisites_unavailable",
    "ok": False,
}
_METRIC_AGGREGATION = {
    "component_domains_are_independent": True,
    "pooled_metric": None,
    "pooled_wer": False,
}
_EVIDENCE_SCOPE = {
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
_EVALUATOR_FILES = (
    "core/wer.py",
    "tools/__init__.py",
    "tools/mobile_asr_evidence_eval.py",
    "tools/prepare_mobile_asr_evidence_packet.py",
    "tools/prepare_primock57_conversation_fixture.py",
    "tools/provision_mobile_zipformer.py",
    "tools/streaming_stt/__init__.py",
    "tools/streaming_stt/bounded_io.py",
    "tools/streaming_stt/corpus.py",
    "tools/streaming_stt/mobile-asr-evidence-packet-v2.lock.json",
    "tools/streaming_stt/mobile-zipformer-en-2023-06-26-v1.lock.json",
    "tools/streaming_stt/overlap_metrics.py",
    "tools/streaming_stt/private_diagnostic_receipt.py",
)


class MobileAsrEvidenceEvalError(RuntimeError):
    """A detail-free packet, model, worker, metric, or publication failure."""


@dataclass(frozen=True, slots=True)
class _Leaf:
    component_id: str
    kind: str
    sha256: str
    samples: int
    audio: bytes = field(repr=False)
    expected_text: str = field(repr=False)
    reference_a: str = field(default="", repr=False)
    reference_b: str = field(default="", repr=False)
    assertion: str = "transcript"
    commands: tuple[str, ...] = field(default=(), repr=False)
    forbidden_commands: tuple[str, ...] = field(default=(), repr=False)
    tags: tuple[str, ...] = ()
    group_index: int = 0

    def __repr__(self) -> str:
        return f"_Leaf(component_id={self.component_id!r}, kind={self.kind!r})"


@dataclass(frozen=True, slots=True)
class _Component:
    component_id: str
    metric_domain: str
    logical_cases: int
    leaves: tuple[_Leaf, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class _Inputs:
    components: tuple[_Component, ...] = field(repr=False)
    production_inputs: bool
    packet_lock_sha256: str
    packet_index_sha256: str
    packet_receipt_sha256: str
    pcm_bytes: int
    samples: int
    packet: object | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class _EndpointOutcome:
    endpoint_texts: tuple[str, ...] = field(repr=False)
    endpoint_source_complete: tuple[bool, ...]
    endpoint_reason: str
    source_complete: bool

    def __post_init__(self) -> None:
        if (
            type(self.endpoint_texts) is not tuple
            or type(self.endpoint_source_complete) is not tuple
            or len(self.endpoint_texts) != len(self.endpoint_source_complete)
            or any(type(value) is not str for value in self.endpoint_texts)
            or any(type(value) is not bool for value in self.endpoint_source_complete)
            or self.endpoint_reason not in {"endpoint", "tail_exhausted"}
            or type(self.source_complete) is not bool
        ):
            raise MobileAsrEvidenceEvalError()

    @property
    def joined_nonempty(self) -> str:
        return " ".join(value for value in self.endpoint_texts if value)

    def __repr__(self) -> str:
        return "_EndpointOutcome(<private>)"


@dataclass(frozen=True, slots=True)
class _EndpointTotals:
    evaluations: int
    source_complete_evaluations: int
    committed_endpoints: int
    empty_committed_endpoints: int
    nonempty_committed_endpoints: int
    zero_endpoint_evaluations: int
    single_endpoint_evaluations: int
    multiple_endpoint_evaluations: int
    empty_hypothesis_evaluations: int
    nonempty_hypothesis_evaluations: int
    pre_source_complete_endpoints: int
    source_complete_endpoints: int
    endpoint_terminated_evaluations: int
    tail_exhausted_evaluations: int

    def as_dict(self) -> dict[str, int]:
        return {
            "committed_endpoints": self.committed_endpoints,
            "empty_committed_endpoints": self.empty_committed_endpoints,
            "empty_hypothesis_evaluations": self.empty_hypothesis_evaluations,
            "endpoint_terminated_evaluations": self.endpoint_terminated_evaluations,
            "evaluations": self.evaluations,
            "multiple_endpoint_evaluations": self.multiple_endpoint_evaluations,
            "nonempty_committed_endpoints": self.nonempty_committed_endpoints,
            "nonempty_hypothesis_evaluations": self.nonempty_hypothesis_evaluations,
            "pre_source_complete_endpoints": self.pre_source_complete_endpoints,
            "single_endpoint_evaluations": self.single_endpoint_evaluations,
            "source_complete_endpoints": self.source_complete_endpoints,
            "source_complete_evaluations": self.source_complete_evaluations,
            "tail_exhausted_evaluations": self.tail_exhausted_evaluations,
            "zero_endpoint_evaluations": self.zero_endpoint_evaluations,
        }


@dataclass(frozen=True, slots=True)
class _ModelAuthority:
    provision: object = field(repr=False)
    manifest: object = field(repr=False)
    manifest_path: Path = field(repr=False)
    manifest_sha256: str
    provision_receipt_sha256: str
    artifact_set_sha256: str
    mobile_config_sha256: str
    total_size_bytes: int


@dataclass(frozen=True, slots=True)
class LoadedMobileAsrEvidenceEvalReport:
    path: Path = field(repr=False)
    digest: str
    packet_index_sha256: str
    packet_receipt_sha256: str
    model_manifest_sha256: str
    source_bundle_sha256: str
    value: Mapping[str, object] = field(repr=False)
    snapshot: tuple[int, ...] = field(repr=False)


@dataclass(slots=True)
class _CommitState:
    linked: bool = False
    committed: bool = False
    digest: str = ""


@dataclass(frozen=True, slots=True)
class _SyntheticWorkerContract:
    """Explicit nonproduction worker outputs available only to private tests."""

    outputs: tuple[_EndpointOutcome, ...] = field(repr=False)
    production_inputs: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.outputs) is not tuple
            or len(self.outputs) != 129
            or any(type(value) is not _EndpointOutcome for value in self.outputs)
            or self.production_inputs is not False
        ):
            raise MobileAsrEvidenceEvalError()


class _SyntheticNonproductionWorker:
    def __init__(self, contract: _SyntheticWorkerContract) -> None:
        if type(contract) is not _SyntheticWorkerContract:
            raise MobileAsrEvidenceEvalError()
        self._outputs = contract.outputs
        self._index = 0

    def transcribe(self, _leaf: _Leaf) -> _EndpointOutcome:
        if self._index >= len(self._outputs):
            raise MobileAsrEvidenceEvalError()
        result = self._outputs[self._index]
        self._index += 1
        return result

    def finish(self) -> None:
        if self._index != len(self._outputs):
            raise MobileAsrEvidenceEvalError()


def _canonical_json(value: object, *, newline: bool = False) -> bytes:
    try:
        raw = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (MemoryError, OverflowError, TypeError, UnicodeError, ValueError):
        raise MobileAsrEvidenceEvalError() from None
    return raw + (b"\n" if newline else b"")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256(value: object) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise MobileAsrEvidenceEvalError()
    return value


def _strict_json(raw: bytes) -> object:
    if not raw or len(raw) > _MAX_REPORT_BYTES:
        raise MobileAsrEvidenceEvalError()

    def pairs(rows: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in rows:
            if key in result:
                raise MobileAsrEvidenceEvalError()
            result[key] = value
        return result

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                MobileAsrEvidenceEvalError()
            ),
        )
    except (MemoryError, OverflowError, UnicodeError, ValueError):
        raise MobileAsrEvidenceEvalError() from None


def _snapshot(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        stat.S_IMODE(metadata.st_mode),
        metadata.st_uid,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _has_git_ancestor(path: Path) -> bool:
    current = path
    while True:
        if (current / ".git").exists() or (current / ".git").is_symlink():
            return True
        if current.parent == current:
            return False
        current = current.parent


def _stable_private_directory(path: Path | str) -> Path:
    try:
        candidate = Path(os.path.abspath(Path(path).expanduser()))
        if not candidate.is_absolute() or candidate.resolve(strict=True) != candidate:
            raise MobileAsrEvidenceEvalError()
        with opened_directory_nofollow(candidate, require_private=True) as (
            stable,
            descriptor,
        ):
            metadata = os.fstat(descriptor)
            if (
                stable != candidate
                or stat.S_IMODE(metadata.st_mode) != 0o700
                or metadata.st_uid != os.getuid()
            ):
                raise MobileAsrEvidenceEvalError()
        return candidate
    except MobileAsrEvidenceEvalError:
        raise
    except (BoundedReadError, OSError, RuntimeError, TypeError, ValueError):
        raise MobileAsrEvidenceEvalError() from None


def _new_path(path: Path | str) -> tuple[Path, Path, tuple[int, ...]]:
    candidate = Path(os.path.abspath(Path(path).expanduser()))
    try:
        if (
            not candidate.is_absolute()
            or not candidate.name
            or candidate.name in {".", ".."}
            or candidate.exists()
            or candidate.is_symlink()
        ):
            raise MobileAsrEvidenceEvalError()
        parent = _stable_private_directory(candidate.parent)
        if _has_git_ancestor(parent):
            raise MobileAsrEvidenceEvalError()
        return candidate, parent, _snapshot(parent.lstat())
    except MobileAsrEvidenceEvalError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError):
        raise MobileAsrEvidenceEvalError() from None


def _new_synthetic_output(path: Path | str) -> tuple[Path, Path, tuple[int, ...]]:
    """Test-only destination check; `/tmp` may carry a harness `.git` marker."""

    candidate = Path(os.path.abspath(Path(path).expanduser()))
    try:
        if (
            not candidate.is_absolute()
            or not candidate.name
            or candidate.name in {".", ".."}
            or candidate.exists()
            or candidate.is_symlink()
        ):
            raise MobileAsrEvidenceEvalError()
        parent = _stable_private_directory(candidate.parent)
        return candidate, parent, _snapshot(parent.lstat())
    except MobileAsrEvidenceEvalError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError):
        raise MobileAsrEvidenceEvalError() from None


def _component_root(packet: object, directory: str) -> Path:
    try:
        root = Path(getattr(packet, "root")) / "components" / directory
        if root.resolve(strict=True) != root:
            raise MobileAsrEvidenceEvalError()
        return root
    except MobileAsrEvidenceEvalError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError):
        raise MobileAsrEvidenceEvalError() from None


def _packet_component_snapshot(
    packet: object,
    lock: object,
    spec: object,
    root: Path,
) -> object:
    """Select one exact-v2 component already admitted by the packet loader."""

    try:
        if (
            type(packet) is not packet_api.LoadedMobileAsrEvidencePacket
            or type(lock) is not packet_api.PacketLock
            or type(spec) is not packet_api.ComponentSpec
            or getattr(lock, "raw_sha256") != PACKET_LOCK_SHA256
        ):
            raise MobileAsrEvidenceEvalError()
        authoritative_specs = tuple(getattr(lock, "components"))
        components = getattr(packet, "_validated_components")
        if (
            type(authoritative_specs) is not tuple
            or type(components) is not tuple
            or len(components) != len(authoritative_specs)
            or any(
                type(value) is not packet_api._ValidatedComponent
                for value in components
            )
            or tuple(value.spec for value in components) != authoritative_specs
            or tuple(value for value in authoritative_specs if value == spec) != (spec,)
        ):
            raise MobileAsrEvidenceEvalError()
        matches = tuple(value for value in components if value.spec == spec)
        if len(matches) != 1 or matches[0].root != root:
            raise MobileAsrEvidenceEvalError()
        return matches[0]
    except MobileAsrEvidenceEvalError:
        raise
    except Exception:
        raise MobileAsrEvidenceEvalError() from None


def _packet_artifacts(spec: object, component: object) -> dict[str, object]:
    """Rebind an admitted component's immutable in-memory artifact snapshot."""

    try:
        if (
            type(spec) is not packet_api.ComponentSpec
            or type(component) is not packet_api._ValidatedComponent
            or component.spec != spec
            or type(component.artifacts) is not tuple
            or len(component.artifacts) != spec.pcm_inputs + 2
        ):
            raise MobileAsrEvidenceEvalError()
        result: dict[str, object] = {}
        folded: set[str] = set()
        for artifact in component.artifacts:
            if type(artifact) is not packet_api._Artifact:
                raise MobileAsrEvidenceEvalError()
            name = artifact.name
            pure = PurePosixPath(name)
            if (
                type(name) is not str
                or not name
                or pure.is_absolute()
                or len(pure.parts) != 1
                or pure.name != name
                or name in result
                or name.casefold() in folded
                or type(artifact.payload) is not bytes
                or hashlib.sha256(artifact.payload).hexdigest()
                != _sha256(artifact.sha256)
            ):
                raise MobileAsrEvidenceEvalError()
            result[name] = artifact
            folded.add(name.casefold())
        manifest = result.get(spec.manifest_file)
        receipt = result.get(spec.receipt_file)
        if (
            type(manifest) is not packet_api._Artifact
            or manifest.sha256 != spec.manifest_sha256
            or type(receipt) is not packet_api._Artifact
            or receipt.sha256 != spec.receipt_sha256
        ):
            raise MobileAsrEvidenceEvalError()
        return result
    except MobileAsrEvidenceEvalError:
        raise
    except Exception:
        raise MobileAsrEvidenceEvalError() from None


def _manifest_strings(
    value: object,
    *,
    maximum_items: int,
    maximum_chars: int,
) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or len(value) > maximum_items
        or any(
            not isinstance(item, str) or not item or len(item) > maximum_chars
            for item in value
        )
    ):
        raise MobileAsrEvidenceEvalError()
    return tuple(value)


def _manifest_int(value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise MobileAsrEvidenceEvalError()
    return value


def _isolated_component_from_snapshot(spec: object, component: object) -> _Component:
    """Build private evaluator leaves only from the exact packet-v2 snapshot."""

    try:
        if (
            type(spec) is not packet_api.ComponentSpec
            or spec.component_id != "primock-isolated"
            or spec.metric_domain != "ordinary-wer-cer-v1"
            or spec.logical_cases != 3
            or spec.pcm_inputs != 3
        ):
            raise MobileAsrEvidenceEvalError()
        artifacts = _packet_artifacts(spec, component)
        manifest = artifacts.pop(spec.manifest_file)
        artifacts.pop(spec.receipt_file)
        if type(manifest) is not packet_api._Artifact:
            raise MobileAsrEvidenceEvalError()
        value = _strict_json(manifest.payload)
        if (
            not isinstance(value, dict)
            or set(value) != {"cases", "provenance", "purpose", "schema_version"}
            or value.get("schema_version") != 2
            or _canonical_json(value, newline=True) != manifest.payload
        ):
            raise MobileAsrEvidenceEvalError()
        cases = value.get("cases")
        if not isinstance(cases, list) or len(cases) != spec.logical_cases:
            raise MobileAsrEvidenceEvalError()
        leaves: list[_Leaf] = []
        for index, row in enumerate(cases):
            if not isinstance(row, dict) or set(row) != {
                "assertion",
                "commands",
                "expected_text",
                "file",
                "id",
                "samples",
                "sha256",
                "tags",
            }:
                raise MobileAsrEvidenceEvalError()
            name = row.get("file")
            samples = row.get("samples")
            expected_text = row.get("expected_text")
            tags = _manifest_strings(
                row.get("tags"), maximum_items=32, maximum_chars=64
            )
            commands = _manifest_strings(
                row.get("commands"), maximum_items=16, maximum_chars=128
            )
            artifact = artifacts.pop(name, None) if isinstance(name, str) else None
            if (
                row.get("id") != f"primock57-isolated-{index:02d}"
                or name != f"primock57-isolated-{index:02d}.f32le"
                or row.get("assertion") != "transcript"
                or commands
                or len(tags) != 3
                or tags[:2] != ("primock57", "isolated")
                or tags[2] not in {"role-a", "role-b"}
                or isinstance(samples, bool)
                or not isinstance(samples, int)
                or samples <= 0
                or not isinstance(expected_text, str)
                or not expected_text.strip()
                or len(expected_text) > 4096
                or type(artifact) is not packet_api._Artifact
                or artifact.sha256 != _sha256(row.get("sha256"))
                or len(artifact.payload) != samples * 4
            ):
                raise MobileAsrEvidenceEvalError()
            leaves.append(
                _Leaf(
                    component_id=spec.component_id,
                    kind="corpus",
                    sha256=artifact.sha256,
                    samples=samples,
                    audio=artifact.payload,
                    expected_text=expected_text,
                    assertion="transcript",
                    commands=commands,
                    tags=tags,
                    group_index=index,
                )
            )
        if (
            artifacts
            or len(leaves) != spec.pcm_inputs
            or sum(len(leaf.audio) for leaf in leaves) != spec.pcm_bytes
        ):
            raise MobileAsrEvidenceEvalError()
        return _Component(
            component_id=spec.component_id,
            metric_domain=spec.metric_domain,
            logical_cases=spec.logical_cases,
            leaves=tuple(leaves),
        )
    except MobileAsrEvidenceEvalError:
        raise
    except Exception:
        raise MobileAsrEvidenceEvalError() from None


def _overlap_component_from_snapshot(spec: object, component: object) -> _Component:
    """Build overlap leaves from the packet-admitted manifest and PCM bytes."""

    try:
        if (
            type(spec) is not packet_api.ComponentSpec
            or spec.component_id != "primock-overlap"
            or spec.metric_domain != OVERLAP_METRIC
            or spec.logical_cases != 3
            or spec.pcm_inputs != 9
        ):
            raise MobileAsrEvidenceEvalError()
        artifacts = _packet_artifacts(spec, component)
        manifest = artifacts.pop(spec.manifest_file)
        artifacts.pop(spec.receipt_file)
        if type(manifest) is not packet_api._Artifact:
            raise MobileAsrEvidenceEvalError()
        value = _strict_json(manifest.payload)
        if (
            not isinstance(value, dict)
            or set(value)
            != {
                "cases",
                "evidence_scope",
                "fixture_id",
                "kind",
                "lock_recipe_sha256",
                "production_evidence",
                "sample_format",
                "schema_version",
                "source_contract_sha256",
            }
            or value.get("schema_version") != 1
            or value.get("kind") != "primock57-two-role-overlap-diagnostic-v1"
            or value.get("production_evidence") is not True
            or _canonical_json(value, newline=True) != manifest.payload
        ):
            raise MobileAsrEvidenceEvalError()
        cases = value.get("cases")
        if not isinstance(cases, list) or len(cases) != spec.logical_cases:
            raise MobileAsrEvidenceEvalError()
        leaves: list[_Leaf] = []
        for group_index, case in enumerate(cases):
            if not isinstance(case, dict) or set(case) != {
                "case_id",
                "envelope",
                "mix",
                "overlap",
                "role_a",
                "role_b",
            }:
                raise MobileAsrEvidenceEvalError()
            envelope = case.get("envelope")
            overlap = case.get("overlap")
            role_a = case.get("role_a")
            role_b = case.get("role_b")
            mix = case.get("mix")
            if (
                case.get("case_id") != f"primock57-overlap-{group_index:02d}"
                or not isinstance(envelope, dict)
                or set(envelope)
                != {"samples", "source_end_sample", "source_start_sample"}
                or not isinstance(overlap, dict)
                or set(overlap)
                != {"relative_end_sample", "relative_start_sample", "samples"}
                or not isinstance(role_a, dict)
                or not isinstance(role_b, dict)
                or set(role_a)
                != {
                    "activity_end_sample",
                    "activity_start_sample",
                    "bytes",
                    "file",
                    "interval_index",
                    "reference",
                    "reference_sha256",
                    "sha256",
                }
                or set(role_b) != set(role_a)
                or not isinstance(mix, dict)
                or set(mix) != {"arithmetic", "bytes", "file", "sha256"}
            ):
                raise MobileAsrEvidenceEvalError()
            samples = _manifest_int(envelope.get("samples"), minimum=1)
            source_start = _manifest_int(envelope.get("source_start_sample"))
            source_end = _manifest_int(envelope.get("source_end_sample"), minimum=1)
            overlap_start = _manifest_int(overlap.get("relative_start_sample"))
            overlap_end = _manifest_int(overlap.get("relative_end_sample"), minimum=1)
            overlap_samples = _manifest_int(overlap.get("samples"), minimum=1)
            reference_a = role_a.get("reference")
            reference_b = role_b.get("reference")
            if (
                source_end - source_start != samples
                or overlap_end - overlap_start != overlap_samples
                or not 0 <= overlap_start < overlap_end <= samples
                or not isinstance(reference_a, str)
                or not reference_a
                or len(reference_a) > 4096
                or hashlib.sha256(reference_a.encode("utf-8")).hexdigest()
                != _sha256(role_a.get("reference_sha256"))
                or not isinstance(reference_b, str)
                or not reference_b
                or len(reference_b) > 4096
                or hashlib.sha256(reference_b.encode("utf-8")).hexdigest()
                != _sha256(role_b.get("reference_sha256"))
            ):
                raise MobileAsrEvidenceEvalError()
            for kind, row in (("role_a", role_a), ("role_b", role_b), ("mix", mix)):
                name = row.get("file")
                size_bytes = row.get("bytes")
                suffix = kind.replace("_", "-")
                artifact = artifacts.pop(name, None) if isinstance(name, str) else None
                if (
                    name != f"primock57-overlap-{group_index:02d}-{suffix}.f32le"
                    or isinstance(size_bytes, bool)
                    or not isinstance(size_bytes, int)
                    or size_bytes != samples * 4
                    or type(artifact) is not packet_api._Artifact
                    or artifact.sha256 != _sha256(row.get("sha256"))
                    or len(artifact.payload) != size_bytes
                    or (
                        kind != "mix"
                        and (
                            _manifest_int(row.get("activity_start_sample"))
                            >= _manifest_int(row.get("activity_end_sample"), minimum=1)
                            or _manifest_int(row.get("activity_end_sample"), minimum=1)
                            > samples
                            or _manifest_int(row.get("interval_index"), minimum=1) < 1
                        )
                    )
                    or (
                        kind == "mix"
                        and row.get("arithmetic")
                        != "float32-add-then-multiply-float32-0.5-v1"
                    )
                ):
                    raise MobileAsrEvidenceEvalError()
                expected = (
                    reference_a
                    if kind == "role_a"
                    else reference_b
                    if kind == "role_b"
                    else ""
                )
                leaves.append(
                    _Leaf(
                        component_id=spec.component_id,
                        kind=kind,
                        sha256=artifact.sha256,
                        samples=samples,
                        audio=artifact.payload,
                        expected_text=expected,
                        reference_a=reference_a,
                        reference_b=reference_b,
                        tags=("primock57", "overlap"),
                        group_index=group_index,
                    )
                )
        if (
            artifacts
            or len(leaves) != spec.pcm_inputs
            or sum(len(leaf.audio) for leaf in leaves) != spec.pcm_bytes
        ):
            raise MobileAsrEvidenceEvalError()
        return _Component(
            component_id=spec.component_id,
            metric_domain=spec.metric_domain,
            logical_cases=spec.logical_cases,
            leaves=tuple(leaves),
        )
    except MobileAsrEvidenceEvalError:
        raise
    except Exception:
        raise MobileAsrEvidenceEvalError() from None


def _corpus_component(spec: object, root: Path) -> _Component:
    try:
        corpus = load_corpus(root / str(getattr(spec, "manifest_file")))
        verify_corpus_snapshot(corpus)
        if (
            type(corpus) is not LoadedCorpus
            or corpus.digest != getattr(spec, "manifest_sha256")
            or len(corpus.cases) != getattr(spec, "logical_cases")
            or corpus.audio_bytes != getattr(spec, "pcm_bytes")
        ):
            raise MobileAsrEvidenceEvalError()
        leaves = tuple(
            _Leaf(
                component_id=str(getattr(spec, "component_id")),
                kind="corpus",
                sha256=case.sha256,
                samples=case.samples,
                audio=case.audio_bytes,
                expected_text=case.expected_text,
                assertion=case.assertion,
                commands=case.commands,
                forbidden_commands=case.forbidden_commands,
                tags=case.tags,
                group_index=index,
            )
            for index, case in enumerate(corpus.cases)
        )
        verify_corpus_snapshot(corpus)
        return _Component(
            component_id=str(getattr(spec, "component_id")),
            metric_domain=str(getattr(spec, "metric_domain")),
            logical_cases=int(getattr(spec, "logical_cases")),
            leaves=leaves,
        )
    except MobileAsrEvidenceEvalError:
        raise
    except Exception:
        raise MobileAsrEvidenceEvalError() from None


def _load_production_inputs(
    packet_root: Path | str,
    *,
    packet_index_sha256: str,
    packet_receipt_sha256: str,
) -> _Inputs:
    try:
        expected_index = _sha256(packet_index_sha256)
        expected_receipt = _sha256(packet_receipt_sha256)
        packet = packet_api.load_mobile_asr_evidence_packet(packet_root)
        lock = packet_api.load_packet_lock()
        if (
            packet.schema_version != PACKET_SCHEMA_VERSION
            or packet.packet_id != PACKET_ID
            or lock.raw_sha256 != PACKET_LOCK_SHA256
            or packet.packet_lock_sha256 != PACKET_LOCK_SHA256
            or expected_index != PACKET_INDEX_SHA256
            or packet.packet_index_sha256 != PACKET_INDEX_SHA256
            or expected_receipt != PACKET_RECEIPT_SHA256
            or packet.packet_receipt_sha256 != PACKET_RECEIPT_SHA256
            or tuple(spec.component_id for spec in lock.components)
            != tuple(row[0] for row in _EXPECTED_COMPONENTS)
        ):
            raise MobileAsrEvidenceEvalError()
        components: list[_Component] = []
        for spec in lock.components:
            root = _component_root(packet, spec.directory)
            if spec.component_id in {"primock-isolated", "primock-overlap"}:
                snapshot = _packet_component_snapshot(packet, lock, spec, root)
                if spec.component_id == "primock-isolated":
                    component = _isolated_component_from_snapshot(spec, snapshot)
                else:
                    component = _overlap_component_from_snapshot(spec, snapshot)
            else:
                component = _corpus_component(spec, root)
            components.append(component)
        observed = tuple(
            (
                value.component_id,
                value.metric_domain,
                value.logical_cases,
                len(value.leaves),
            )
            for value in components
        )
        if (
            observed != _EXPECTED_COMPONENTS
            or sum(len(value.leaves) for value in components) != 129
            or sum(len(leaf.audio) for value in components for leaf in value.leaves)
            != packet.pcm_bytes
            or sum(leaf.samples for value in components for leaf in value.leaves)
            != packet.samples
        ):
            raise MobileAsrEvidenceEvalError()
        packet_api.verify_mobile_asr_evidence_packet(packet)
        return _Inputs(
            components=tuple(components),
            production_inputs=True,
            packet_lock_sha256=packet.packet_lock_sha256,
            packet_index_sha256=packet.packet_index_sha256,
            packet_receipt_sha256=packet.packet_receipt_sha256,
            pcm_bytes=packet.pcm_bytes,
            samples=packet.samples,
            packet=packet,
        )
    except MobileAsrEvidenceEvalError:
        raise
    except Exception:
        raise MobileAsrEvidenceEvalError() from None


def _mobile_config_dict(value: object) -> dict[str, object]:
    try:
        method = getattr(value, "as_dict")
        result = method()
        if not isinstance(result, dict) or not result:
            raise MobileAsrEvidenceEvalError()
        _canonical_json(result)
        return dict(result)
    except MobileAsrEvidenceEvalError:
        raise
    except Exception:
        raise MobileAsrEvidenceEvalError() from None


def _bind_mobile_configs(
    provision_config: object,
    manifest_config: object,
    *,
    provision_type: type[object],
    manifest_type: type[object],
) -> str:
    """Bind two separately owned config types by their exact closed mappings."""

    if (
        type(provision_config) is not provision_type
        or type(manifest_config) is not manifest_type
    ):
        raise MobileAsrEvidenceEvalError()
    provision_value = _mobile_config_dict(provision_config)
    manifest_value = _mobile_config_dict(manifest_config)
    if (
        provision_value != _EXPECTED_MOBILE_CONFIG
        or manifest_value != _EXPECTED_MOBILE_CONFIG
        or provision_value != manifest_value
        or _canonical_sha256(manifest_value) != MOBILE_CONFIG_SHA256
    ):
        raise MobileAsrEvidenceEvalError()
    return MOBILE_CONFIG_SHA256


def _load_model_authority(path: Path | str) -> _ModelAuthority:
    try:
        from tools.provision_mobile_zipformer import (
            MobileZipformerConfig as ProvisionMobileZipformerConfig,
            load_mobile_zipformer_provision,
        )
        from tools.streaming_stt.manifest import (
            MobileZipformerConfig,
            load_worker_manifest,
        )

        provision = load_mobile_zipformer_provision(path)
        manifest_path = Path(getattr(provision, "path"))
        manifest = load_worker_manifest(manifest_path)
        provision_config = getattr(provision, "mobile_config")
        manifest_config = getattr(manifest, "adapter_config")
        config_sha256 = _bind_mobile_configs(
            provision_config,
            manifest_config,
            provision_type=ProvisionMobileZipformerConfig,
            manifest_type=MobileZipformerConfig,
        )
        if (
            getattr(provision, "model_id") != MODEL_ID
            or getattr(provision, "adapter") != ADAPTER
            or getattr(manifest, "schema_version") != 10
            or getattr(manifest, "model_id") != MODEL_ID
            or getattr(manifest, "adapter") != ADAPTER
            or getattr(manifest, "digest") != getattr(provision, "digest")
            or Path(getattr(manifest, "path")) != manifest_path
            or getattr(manifest, "python") != getattr(provision, "python")
            or getattr(manifest, "worker") != getattr(provision, "worker")
            or tuple(getattr(manifest, "artifacts"))
            != tuple(getattr(provision, "artifacts"))
            or getattr(provision, "artifact_set_sha256") != MODEL_ARTIFACT_SET_SHA256
            or getattr(provision, "total_size_bytes") != MODEL_TOTAL_SIZE_BYTES
        ):
            raise MobileAsrEvidenceEvalError()
        return _ModelAuthority(
            provision=provision,
            manifest=manifest,
            manifest_path=manifest_path,
            manifest_sha256=_sha256(getattr(provision, "digest")),
            provision_receipt_sha256=_sha256(getattr(provision, "receipt_digest")),
            artifact_set_sha256=_sha256(getattr(provision, "artifact_set_sha256")),
            mobile_config_sha256=config_sha256,
            total_size_bytes=int(getattr(provision, "total_size_bytes")),
        )
    except MobileAsrEvidenceEvalError:
        raise
    except Exception:
        raise MobileAsrEvidenceEvalError() from None


def _verify_model_authority(authority: _ModelAuthority) -> None:
    current = _load_model_authority(authority.manifest_path)
    if (
        current.manifest_sha256 != authority.manifest_sha256
        or current.provision_receipt_sha256 != authority.provision_receipt_sha256
        or current.artifact_set_sha256 != authority.artifact_set_sha256
        or current.mobile_config_sha256 != authority.mobile_config_sha256
        or current.total_size_bytes != authority.total_size_bytes
        or current.provision != authority.provision
        or current.manifest != authority.manifest
    ):
        raise MobileAsrEvidenceEvalError()


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
        raise MobileAsrEvidenceEvalError() from None


def _validate_destinations(
    inputs: _Inputs,
    model: _ModelAuthority,
    *,
    scratch_root: Path | str,
    output_path: Path | str,
) -> tuple[Path, Path, tuple[int, ...], tuple[int, ...]]:
    scratch, scratch_parent, scratch_parent_snapshot = _new_path(scratch_root)
    output, output_parent, output_parent_snapshot = _new_path(output_path)
    try:
        protected = {
            Path(getattr(inputs.packet, "root")).resolve(strict=True),
            model.manifest_path.parent.resolve(strict=True),
            Path(getattr(model.manifest, "python").path).resolve(strict=True).parent,
            Path(getattr(model.manifest, "worker").path).resolve(strict=True).parent,
            *(
                Path(artifact.path).resolve(strict=True).parent
                for artifact in getattr(model.manifest, "artifacts")
            ),
        }
        if _paths_overlap(scratch, output_parent):
            raise MobileAsrEvidenceEvalError()
        for root in protected:
            if _paths_overlap(scratch, root) or _paths_overlap(output_parent, root):
                raise MobileAsrEvidenceEvalError()
        if scratch_parent == output_parent:
            raise MobileAsrEvidenceEvalError()
        return scratch, output, scratch_parent_snapshot, output_parent_snapshot
    except MobileAsrEvidenceEvalError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError):
        raise MobileAsrEvidenceEvalError() from None


def preflight_mobile_asr_evidence_eval(
    packet_root: Path | str,
    *,
    packet_index_sha256: str,
    packet_receipt_sha256: str,
    worker_manifest: Path | str,
    scratch_root: Path | str,
    output_path: Path | str,
) -> dict[str, object]:
    """Validate all exact production inputs without creating or executing."""

    try:
        inputs = _load_production_inputs(
            packet_root,
            packet_index_sha256=packet_index_sha256,
            packet_receipt_sha256=packet_receipt_sha256,
        )
        model = _load_model_authority(worker_manifest)
        _validate_destinations(
            inputs,
            model,
            scratch_root=scratch_root,
            output_path=output_path,
        )
        closure = _evaluator_closure_sha256()
        if inputs.packet is None:
            raise MobileAsrEvidenceEvalError()
        packet_api.verify_mobile_asr_evidence_packet(inputs.packet)
        _verify_model_authority(model)
        return {
            "components": 5,
            "evaluator_closure_sha256": closure,
            "kind": "mobile-asr-evidence-eval-preflight-v1",
            "logical_cases": 123,
            "model_manifest_sha256": model.manifest_sha256,
            "ok": True,
            "packet_index_sha256": inputs.packet_index_sha256,
            "packet_lock_sha256": inputs.packet_lock_sha256,
            "packet_receipt_sha256": inputs.packet_receipt_sha256,
            "pcm_bytes": inputs.pcm_bytes,
            "pcm_inputs": 129,
            "production_inputs": True,
            "samples": inputs.samples,
        }
    except MobileAsrEvidenceEvalError:
        raise
    except Exception:
        raise MobileAsrEvidenceEvalError() from None


def _write_private(path: Path, payload: bytes) -> None:
    descriptor = -1
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags, 0o600)
        os.fchmod(descriptor, 0o600)
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if type(count) is not int or count <= 0:
                raise MobileAsrEvidenceEvalError()
            written += count
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_nlink != 1
            or metadata.st_uid != os.getuid()
            or metadata.st_size != len(payload)
        ):
            raise MobileAsrEvidenceEvalError()
    except MobileAsrEvidenceEvalError:
        raise
    except OSError:
        raise MobileAsrEvidenceEvalError() from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _prepare_scratch(
    scratch: Path,
    inputs: _Inputs,
) -> tuple[Path, tuple[Path, ...], SourceBundle, tuple[int, ...]]:
    try:
        scratch.mkdir(mode=0o700)
        os.chmod(scratch, 0o700)
        worker_root = scratch / "worker"
        worker_root.mkdir(mode=0o700)
        os.chmod(worker_root, 0o700)
        source_bundle = stage_worker_source_bundle(
            Path(__file__).resolve().parents[1],
            scratch / "source-bundle",
        )
        paths: list[Path] = []
        index = 0
        for component in inputs.components:
            for leaf in component.leaves:
                path = worker_root / f"input-{index:03d}.f32le"
                _write_private(path, leaf.audio)
                paths.append(path)
                index += 1
        if len(paths) != 129:
            raise MobileAsrEvidenceEvalError()
        root_snapshot = _snapshot(scratch.lstat())
        _verify_scratch(scratch, root_snapshot, source_bundle, tuple(paths), inputs)
        return worker_root, tuple(paths), source_bundle, root_snapshot
    except MobileAsrEvidenceEvalError:
        raise
    except Exception:
        raise MobileAsrEvidenceEvalError() from None


def _verify_scratch(
    scratch: Path,
    root_snapshot: tuple[int, ...],
    source_bundle: SourceBundle,
    paths: tuple[Path, ...],
    inputs: _Inputs,
) -> None:
    leaves = tuple(leaf for component in inputs.components for leaf in component.leaves)
    if len(paths) != len(leaves):
        raise MobileAsrEvidenceEvalError()
    try:
        metadata = scratch.lstat()
        if (
            _snapshot(metadata)[:7] != root_snapshot[:7]
            or stat.S_IMODE(metadata.st_mode) != 0o700
            or metadata.st_uid != os.getuid()
        ):
            raise MobileAsrEvidenceEvalError()
        verify_source_bundle(source_bundle, required_files=WORKER_SOURCE_FILES)
        worker_root = scratch / "worker"
        for path, leaf in zip(paths, leaves, strict=True):
            current = path.lstat()
            if (
                path.parent != worker_root
                or path.resolve(strict=True) != path
                or not stat.S_ISREG(current.st_mode)
                or stat.S_IMODE(current.st_mode) != 0o600
                or current.st_nlink != 1
                or current.st_uid != os.getuid()
                or current.st_size != len(leaf.audio)
            ):
                raise MobileAsrEvidenceEvalError()
            digest = hash_regular_bounded(
                path,
                maximum_bytes=packet_api.MAX_PCM_BYTES,
                expected_bytes=len(leaf.audio),
            )
            if digest.sha256 != leaf.sha256:
                raise MobileAsrEvidenceEvalError()
    except MobileAsrEvidenceEvalError:
        raise
    except Exception:
        raise MobileAsrEvidenceEvalError() from None


def _extract_endpoint_outcome(trace: object, leaf: _Leaf) -> _EndpointOutcome:
    try:
        from tools.streaming_stt.protocol import (
            MobileZipformerEndpointObservation,
            MobileZipformerFinalEvent,
        )

        final = getattr(trace, "final")
        if (
            type(final) is not MobileZipformerFinalEvent
            or final.protocol_version != PROTOCOL_VERSION
            or final.source_samples != leaf.samples
            or final.source_samples_consumed != leaf.samples
            or final.declared_tail_samples != _STREAM.tail_padding_samples
            or not 0 <= final.tail_samples_consumed <= _STREAM.tail_padding_samples
            or final.endpoint_reason not in {"endpoint", "tail_exhausted"}
            or final.native_endpoint != (final.endpoint_reason == "endpoint")
            or final.authoritative is not final.native_endpoint
            or type(final.endpoint_observations) is not tuple
            or any(
                type(value) is not MobileZipformerEndpointObservation
                for value in final.endpoint_observations
            )
        ):
            raise MobileAsrEvidenceEvalError()
        texts = tuple(value.text for value in final.endpoint_observations)
        source_complete = tuple(
            value.source_complete for value in final.endpoint_observations
        )
        joined = " ".join(value for value in texts if value)
        if final.text != joined:
            raise MobileAsrEvidenceEvalError()
        return _EndpointOutcome(
            endpoint_texts=texts,
            endpoint_source_complete=source_complete,
            endpoint_reason=final.endpoint_reason,
            source_complete=True,
        )
    except MobileAsrEvidenceEvalError:
        raise
    except Exception:
        raise MobileAsrEvidenceEvalError() from None


def _endpoint_totals(outcomes: Sequence[_EndpointOutcome]) -> _EndpointTotals:
    selected = tuple(outcomes)
    if not selected or any(type(value) is not _EndpointOutcome for value in selected):
        raise MobileAsrEvidenceEvalError()
    endpoint_counts = [len(value.endpoint_texts) for value in selected]
    committed = sum(endpoint_counts)
    empty = sum(text == "" for value in selected for text in value.endpoint_texts)
    source_complete = sum(
        complete for value in selected for complete in value.endpoint_source_complete
    )
    return _EndpointTotals(
        evaluations=len(selected),
        source_complete_evaluations=sum(value.source_complete for value in selected),
        committed_endpoints=committed,
        empty_committed_endpoints=empty,
        nonempty_committed_endpoints=committed - empty,
        zero_endpoint_evaluations=sum(count == 0 for count in endpoint_counts),
        single_endpoint_evaluations=sum(count == 1 for count in endpoint_counts),
        multiple_endpoint_evaluations=sum(count > 1 for count in endpoint_counts),
        empty_hypothesis_evaluations=sum(
            not value.joined_nonempty for value in selected
        ),
        nonempty_hypothesis_evaluations=sum(
            bool(value.joined_nonempty) for value in selected
        ),
        pre_source_complete_endpoints=committed - source_complete,
        source_complete_endpoints=source_complete,
        endpoint_terminated_evaluations=sum(
            value.endpoint_reason == "endpoint" for value in selected
        ),
        tail_exhausted_evaluations=sum(
            value.endpoint_reason == "tail_exhausted" for value in selected
        ),
    )


def _edit_distance(reference: Sequence[object], hypothesis: Sequence[object]) -> int:
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for row, ref_item in enumerate(reference, 1):
        current = [row]
        for column, hyp_item in enumerate(hypothesis, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (ref_item != hyp_item),
                )
            )
        previous = current
    return previous[-1]


def _accuracy(pairs: Sequence[tuple[str, str]]) -> dict[str, object]:
    substitutions = insertions = deletions = 0
    reference_words = hypothesis_words = 0
    character_errors = reference_characters = hypothesis_characters = 0
    exact = nonempty = 0
    for reference, hypothesis in pairs:
        measured = word_error_rate(reference, hypothesis)
        substitutions += measured.substitutions
        insertions += measured.insertions
        deletions += measured.deletions
        reference_words += measured.ref_words
        hypothesis_words += measured.hyp_words
        ref_tokens = normalize(reference)
        hyp_tokens = normalize(hypothesis)
        exact += ref_tokens == hyp_tokens
        nonempty += bool(hyp_tokens)
        ref_chars = "".join(ref_tokens)
        hyp_chars = "".join(hyp_tokens)
        character_errors += _edit_distance(ref_chars, hyp_chars)
        reference_characters += len(ref_chars)
        hypothesis_characters += len(hyp_chars)
    word_errors = substitutions + insertions + deletions
    wer = (
        (0.0 if hypothesis_words == 0 else 1.0)
        if reference_words == 0
        else word_errors / reference_words
    )
    cer = (
        (0.0 if hypothesis_characters == 0 else 1.0)
        if reference_characters == 0
        else character_errors / reference_characters
    )
    if not math.isfinite(wer) or not math.isfinite(cer):
        raise MobileAsrEvidenceEvalError()
    return {
        "cer": cer,
        "character_errors": character_errors,
        "evaluations": len(pairs),
        "exact": exact,
        "hypothesis_characters": hypothesis_characters,
        "hypothesis_words": hypothesis_words,
        "nonempty": nonempty,
        "reference_characters": reference_characters,
        "reference_words": reference_words,
        "substitutions": substitutions,
        "insertions": insertions,
        "deletions": deletions,
        "wer": wer,
        "word_errors": word_errors,
    }


def _contains_tokens(haystack: Sequence[str], needle: Sequence[str]) -> bool:
    return (
        bool(needle)
        and len(needle) <= len(haystack)
        and any(
            tuple(haystack[index : index + len(needle)]) == tuple(needle)
            for index in range(len(haystack) - len(needle) + 1)
        )
    )


def _command_metrics(
    component: _Component,
    outcomes: Sequence[_EndpointOutcome],
) -> dict[str, object]:
    transcript_pairs: list[tuple[str, str]] = []
    command_attempts = command_hits = 0
    forbidden_attempts = forbidden_hits = 0
    monitored_negative_evaluations = monitored_negative_hits = 0
    silence_nonempty_endpoints = speech_negative_nonempty_endpoints = 0
    for leaf, outcome in zip(component.leaves, outcomes, strict=True):
        decisions = tuple(tuple(normalize(value)) for value in outcome.endpoint_texts)
        if leaf.assertion == "transcript":
            transcript_pairs.append((leaf.expected_text, outcome.joined_nonempty))
        for command in leaf.commands:
            command_attempts += 1
            command_hits += any(
                _contains_tokens(decision, normalize(command)) for decision in decisions
            )
        case_forbidden = False
        for command in leaf.forbidden_commands:
            forbidden_attempts += 1
            hit = any(
                _contains_tokens(decision, normalize(command)) for decision in decisions
            )
            forbidden_hits += hit
            case_forbidden |= hit
        monitored = not leaf.commands and bool(leaf.forbidden_commands)
        monitored_negative_evaluations += monitored
        monitored_negative_hits += monitored and case_forbidden
        nonempty = sum(bool(normalize(value)) for value in outcome.endpoint_texts)
        if leaf.assertion == "silence":
            silence_nonempty_endpoints += nonempty
        elif leaf.assertion == "speech_negative":
            speech_negative_nonempty_endpoints += nonempty
    return {
        "command_attempts": command_attempts,
        "command_hits": command_hits,
        "command_recall": None
        if command_attempts == 0
        else command_hits / command_attempts,
        "forbidden_target_attempts": forbidden_attempts,
        "forbidden_target_hits": forbidden_hits,
        "forbidden_target_false_positive_rate": (
            None if forbidden_attempts == 0 else forbidden_hits / forbidden_attempts
        ),
        "monitored_negative_evaluations": monitored_negative_evaluations,
        "monitored_negative_evaluations_with_hit": monitored_negative_hits,
        "silence_nonempty_endpoints": silence_nonempty_endpoints,
        "speech_negative_nonempty_endpoints": speech_negative_nonempty_endpoints,
        "transcript_accuracy": _accuracy(transcript_pairs),
    }


def _demand_metrics(
    component: _Component,
    outcomes: Sequence[_EndpointOutcome],
) -> dict[str, object]:
    pairs = [
        (leaf.expected_text, outcome.joined_nonempty)
        for leaf, outcome in zip(component.leaves, outcomes, strict=True)
    ]
    strata: list[dict[str, object]] = []
    for tag in _DEMAND_STRATA:
        selected = [
            pair
            for leaf, pair in zip(component.leaves, pairs, strict=True)
            if tag in leaf.tags
        ]
        if len(selected) != 14:
            raise MobileAsrEvidenceEvalError()
        strata.append({"accuracy": _accuracy(selected), "stratum": tag})
    return {"accuracy": _accuracy(pairs), "strata": strata}


def _notsofar_metrics(
    component: _Component,
    outcomes: Sequence[_EndpointOutcome],
) -> dict[str, object]:
    pairs = [
        (leaf.expected_text, outcome.joined_nonempty)
        for leaf, outcome in zip(component.leaves, outcomes, strict=True)
    ]
    channels: list[dict[str, object]] = []
    for public_name, tag in (("a", "channel-a"), ("b", "channel-b")):
        selected = [
            pair
            for leaf, pair in zip(component.leaves, pairs, strict=True)
            if tag in leaf.tags
        ]
        if len(selected) != 9:
            raise MobileAsrEvidenceEvalError()
        channels.append({"accuracy": _accuracy(selected), "channel": public_name})
    agreements = 0
    for index in range(0, 18, 2):
        left_leaf = component.leaves[index]
        right_leaf = component.leaves[index + 1]
        left_outcome = outcomes[index]
        right_outcome = outcomes[index + 1]
        if (
            "channel-a" not in left_leaf.tags
            or "channel-b" not in right_leaf.tags
            or left_leaf.expected_text != right_leaf.expected_text
            or left_leaf.samples != right_leaf.samples
        ):
            raise MobileAsrEvidenceEvalError()
        agreements += normalize(left_outcome.joined_nonempty) == normalize(
            right_outcome.joined_nonempty
        )
    return {
        "accuracy": _accuracy(pairs),
        "channels": channels,
        "paired_channel": {
            "exact_normalized_output_agreements": agreements,
            "pairs": 9,
        },
    }


def _overlap_metrics(
    component: _Component,
    outcomes: Sequence[_EndpointOutcome],
) -> dict[str, object]:
    measured: list[OverlapMetricResult] = []
    for leaf, outcome in zip(component.leaves, outcomes, strict=True):
        if leaf.kind == "mix":
            measured.append(
                score_two_utterance_min_order_wer(
                    leaf.reference_a,
                    leaf.reference_b,
                    outcome.joined_nonempty,
                )
            )
        elif leaf.kind not in {"role_a", "role_b"}:
            raise MobileAsrEvidenceEvalError()
    aggregate = aggregate_two_utterance_min_order_wer(measured).as_dict()
    aggregate.pop("cases")
    return {
        "custom_accuracy": aggregate,
        "scored_mix_evaluations": 3,
        "unscored_stem_executions": 6,
    }


def _component_report(
    component: _Component,
    outcomes: Sequence[_EndpointOutcome],
) -> dict[str, object]:
    if len(outcomes) != len(component.leaves):
        raise MobileAsrEvidenceEvalError()
    endpoint = _endpoint_totals(outcomes).as_dict()
    if component.component_id == "command-noise":
        metrics = _command_metrics(component, outcomes)
    elif component.component_id == "demand-noise":
        metrics = _demand_metrics(component, outcomes)
    elif component.component_id == "notsofar-far-field":
        metrics = _notsofar_metrics(component, outcomes)
    elif component.component_id == "primock-isolated":
        metrics = {
            "accuracy": _accuracy(
                [
                    (leaf.expected_text, outcome.joined_nonempty)
                    for leaf, outcome in zip(component.leaves, outcomes, strict=True)
                ]
            )
        }
    elif component.component_id == "primock-overlap":
        metrics = _overlap_metrics(component, outcomes)
    else:
        raise MobileAsrEvidenceEvalError()
    return {
        "component": component.component_id,
        "endpoint_integrity": endpoint,
        "evaluations": len(component.leaves),
        "metric_domain": component.metric_domain,
        "metrics": metrics,
    }


def _reduce_components(
    inputs: _Inputs,
    transcribe: Callable[[_Leaf, int], _EndpointOutcome],
) -> tuple[list[dict[str, object]], _EndpointTotals]:
    reports: list[dict[str, object]] = []
    all_outcomes: list[_EndpointOutcome] = []
    request_index = 0
    for component in inputs.components:
        outcomes: list[_EndpointOutcome] = []
        for leaf in component.leaves:
            outcome = transcribe(leaf, request_index)
            if type(outcome) is not _EndpointOutcome:
                raise MobileAsrEvidenceEvalError()
            outcomes.append(outcome)
            all_outcomes.append(outcome)
            request_index += 1
        reports.append(_component_report(component, outcomes))
    if request_index != 129:
        raise MobileAsrEvidenceEvalError()
    return reports, _endpoint_totals(all_outcomes)


def _worker_binding(
    model: _ModelAuthority,
    source_bundle: SourceBundle,
) -> dict[str, object]:
    return {
        "adapter": ADAPTER,
        "artifact_set_sha256": model.artifact_set_sha256,
        "mobile_config_sha256": model.mobile_config_sha256,
        "model_id": MODEL_ID,
        "model_manifest_sha256": model.manifest_sha256,
        "model_total_size_bytes": model.total_size_bytes,
        "provision_receipt_sha256": model.provision_receipt_sha256,
        "source_bundle_files": len(source_bundle.files),
        "source_bundle_sha256": source_bundle.tree_sha256,
    }


def _report_evidence_scope(
    *, production_inputs: bool, model_executed: bool
) -> dict[str, bool]:
    """Grant only exact-model identity and five-domain unpooled result authority."""

    if (
        type(production_inputs) is not bool
        or type(model_executed) is not bool
        or model_executed is not production_inputs
    ):
        raise MobileAsrEvidenceEvalError()
    return {
        **_EVIDENCE_SCOPE,
        "mobile_model_identity_authority": production_inputs,
        "model_executed": model_executed,
        "production_packet": production_inputs,
        "unpooled_component_result_authority": production_inputs,
    }


def _report(
    inputs: _Inputs,
    components: list[dict[str, object]],
    endpoints: _EndpointTotals,
    *,
    binding: Mapping[str, object],
    evaluator_closure_sha256: str,
    worker_stderr: Mapping[str, object],
    model_executed: bool,
) -> dict[str, object]:
    kind = REPORT_KIND if inputs.production_inputs else SYNTHETIC_REPORT_KIND
    value: dict[str, object] = {
        "bindings": {
            **dict(binding),
            "evaluator_closure_sha256": evaluator_closure_sha256,
            "packet_index_sha256": inputs.packet_index_sha256,
            "packet_lock_sha256": inputs.packet_lock_sha256,
            "packet_receipt_sha256": inputs.packet_receipt_sha256,
        },
        "complete": True,
        "components": components,
        "evidence_scope": _report_evidence_scope(
            production_inputs=inputs.production_inputs,
            model_executed=model_executed,
        ),
        "execution": {
            "components": len(inputs.components),
            "endpoint_integrity": endpoints.as_dict(),
            "evaluations": 129,
            "logical_cases": sum(value.logical_cases for value in inputs.components),
            "pcm_bytes": inputs.pcm_bytes,
            "pcm_inputs": 129,
            "samples": inputs.samples,
            "stream": {
                "chunk_samples": _STREAM.chunk_samples,
                "pace": _STREAM.pace,
                "partial_report_cadence_ms": _STREAM.partial_interval_ms,
                "protocol_version": PROTOCOL_VERSION,
                "repeats": 1,
                "tail_padding_samples": _STREAM.tail_padding_samples,
            },
            "worker_stderr": dict(worker_stderr),
        },
        "kind": kind,
        "metric_aggregation": dict(_METRIC_AGGREGATION),
        "ok": True,
        "production_inputs": inputs.production_inputs,
        "schema_version": SCHEMA_VERSION,
    }
    value["binding_sha256"] = _canonical_sha256(value)
    _validate_report(value, production_inputs=inputs.production_inputs)
    return value


def _validate_endpoint(value: object, *, evaluations: int) -> None:
    expected = set(_EndpointTotals(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).as_dict())
    if not isinstance(value, Mapping) or set(value) != expected:
        raise MobileAsrEvidenceEvalError()
    if any(type(item) is not int or item < 0 for item in value.values()):
        raise MobileAsrEvidenceEvalError()
    if (
        value["evaluations"] != evaluations
        or value["source_complete_evaluations"] != evaluations
        or value["zero_endpoint_evaluations"]
        + value["single_endpoint_evaluations"]
        + value["multiple_endpoint_evaluations"]
        != evaluations
        or value["empty_hypothesis_evaluations"]
        + value["nonempty_hypothesis_evaluations"]
        != evaluations
        or value["empty_committed_endpoints"] + value["nonempty_committed_endpoints"]
        != value["committed_endpoints"]
        or value["pre_source_complete_endpoints"] + value["source_complete_endpoints"]
        != value["committed_endpoints"]
        or value["source_complete_endpoints"]
        != value["endpoint_terminated_evaluations"]
        or value["endpoint_terminated_evaluations"]
        > value["single_endpoint_evaluations"] + value["multiple_endpoint_evaluations"]
        or value["endpoint_terminated_evaluations"]
        + value["tail_exhausted_evaluations"]
        != evaluations
    ):
        raise MobileAsrEvidenceEvalError()


def _walk_scalars(value: object) -> Sequence[object]:
    result: list[object] = []
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, Mapping):
            stack.extend(current.keys())
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
        else:
            result.append(current)
    return result


_ACCURACY_INTEGER_FIELDS = (
    "character_errors",
    "deletions",
    "evaluations",
    "exact",
    "hypothesis_characters",
    "hypothesis_words",
    "insertions",
    "nonempty",
    "reference_characters",
    "reference_words",
    "substitutions",
    "word_errors",
)
_ACCURACY_FIELDS = {*_ACCURACY_INTEGER_FIELDS, "cer", "wer"}


def _expected_rate(errors: int, reference: int, hypothesis: int) -> float:
    if reference == 0:
        return 0.0 if hypothesis == 0 else 1.0
    return errors / reference


def _validate_finite_ratio(value: object, *, expected: float) -> None:
    if type(value) is not float or not math.isfinite(value) or value != expected:
        raise MobileAsrEvidenceEvalError()


def _validate_accuracy(value: object, *, evaluations: int) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != _ACCURACY_FIELDS:
        raise MobileAsrEvidenceEvalError()
    if any(
        type(value.get(key)) is not int or value[key] < 0
        for key in _ACCURACY_INTEGER_FIELDS
    ):
        raise MobileAsrEvidenceEvalError()
    if (
        value["evaluations"] != evaluations
        or value["exact"] > evaluations
        or value["nonempty"] > evaluations
        or value["word_errors"]
        != value["substitutions"] + value["insertions"] + value["deletions"]
        or value["deletions"] > value["reference_words"]
        or value["insertions"] > value["hypothesis_words"]
        or value["substitutions"] > value["reference_words"] - value["deletions"]
        or value["substitutions"] > value["hypothesis_words"] - value["insertions"]
        or value["hypothesis_words"]
        != value["reference_words"] - value["deletions"] + value["insertions"]
    ):
        raise MobileAsrEvidenceEvalError()
    _validate_finite_ratio(
        value.get("wer"),
        expected=_expected_rate(
            int(value["word_errors"]),
            int(value["reference_words"]),
            int(value["hypothesis_words"]),
        ),
    )
    _validate_finite_ratio(
        value.get("cer"),
        expected=_expected_rate(
            int(value["character_errors"]),
            int(value["reference_characters"]),
            int(value["hypothesis_characters"]),
        ),
    )
    return value


def _integer_accuracy_sum(
    values: Sequence[Mapping[str, object]],
) -> dict[str, int]:
    return {
        key: sum(int(value[key]) for value in values)
        for key in _ACCURACY_INTEGER_FIELDS
    }


def _validate_optional_rate(
    value: object,
    *,
    numerator: int,
    denominator: int,
) -> None:
    if denominator == 0:
        if value is not None:
            raise MobileAsrEvidenceEvalError()
        return
    _validate_finite_ratio(value, expected=numerator / denominator)


def _validate_command_metrics(
    value: object,
    *,
    endpoint: Mapping[str, object],
) -> None:
    fields = {
        "command_attempts",
        "command_hits",
        "command_recall",
        "forbidden_target_attempts",
        "forbidden_target_false_positive_rate",
        "forbidden_target_hits",
        "monitored_negative_evaluations",
        "monitored_negative_evaluations_with_hit",
        "silence_nonempty_endpoints",
        "speech_negative_nonempty_endpoints",
        "transcript_accuracy",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise MobileAsrEvidenceEvalError()
    counts = (
        "command_attempts",
        "command_hits",
        "forbidden_target_attempts",
        "forbidden_target_hits",
        "monitored_negative_evaluations",
        "monitored_negative_evaluations_with_hit",
        "silence_nonempty_endpoints",
        "speech_negative_nonempty_endpoints",
    )
    if any(type(value.get(key)) is not int or value[key] < 0 for key in counts):
        raise MobileAsrEvidenceEvalError()
    if (
        value["command_hits"] > value["command_attempts"]
        or value["forbidden_target_hits"] > value["forbidden_target_attempts"]
        or value["monitored_negative_evaluations"] > 57
        or value["monitored_negative_evaluations_with_hit"]
        > value["monitored_negative_evaluations"]
        or value["monitored_negative_evaluations_with_hit"]
        > value["forbidden_target_hits"]
        or value["silence_nonempty_endpoints"]
        + value["speech_negative_nonempty_endpoints"]
        > endpoint["nonempty_committed_endpoints"]
    ):
        raise MobileAsrEvidenceEvalError()
    _validate_optional_rate(
        value.get("command_recall"),
        numerator=int(value["command_hits"]),
        denominator=int(value["command_attempts"]),
    )
    _validate_optional_rate(
        value.get("forbidden_target_false_positive_rate"),
        numerator=int(value["forbidden_target_hits"]),
        denominator=int(value["forbidden_target_attempts"]),
    )
    accuracy = _validate_accuracy(
        value.get("transcript_accuracy"),
        evaluations=int(value["transcript_accuracy"].get("evaluations", -1))
        if isinstance(value.get("transcript_accuracy"), Mapping)
        else -1,
    )
    if not 1 <= accuracy["evaluations"] <= 57:
        raise MobileAsrEvidenceEvalError()


def _validate_demand_metrics(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != {"accuracy", "strata"}:
        raise MobileAsrEvidenceEvalError()
    overall = _validate_accuracy(value.get("accuracy"), evaluations=42)
    strata = value.get("strata")
    if not isinstance(strata, list) or len(strata) != 6:
        raise MobileAsrEvidenceEvalError()
    rows: list[Mapping[str, object]] = []
    names: list[object] = []
    for row in strata:
        if not isinstance(row, Mapping) or set(row) != {"accuracy", "stratum"}:
            raise MobileAsrEvidenceEvalError()
        names.append(row.get("stratum"))
        rows.append(_validate_accuracy(row.get("accuracy"), evaluations=14))
    if tuple(names) != _DEMAND_STRATA:
        raise MobileAsrEvidenceEvalError()
    expected = {key: int(overall[key]) for key in _ACCURACY_INTEGER_FIELDS}
    if (
        _integer_accuracy_sum(rows[:3]) != expected
        or _integer_accuracy_sum(rows[3:]) != expected
    ):
        raise MobileAsrEvidenceEvalError()


def _validate_notsofar_metrics(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "accuracy",
        "channels",
        "paired_channel",
    }:
        raise MobileAsrEvidenceEvalError()
    overall = _validate_accuracy(value.get("accuracy"), evaluations=18)
    channels = value.get("channels")
    if not isinstance(channels, list) or len(channels) != 2:
        raise MobileAsrEvidenceEvalError()
    rows: list[Mapping[str, object]] = []
    names: list[object] = []
    for row in channels:
        if not isinstance(row, Mapping) or set(row) != {"accuracy", "channel"}:
            raise MobileAsrEvidenceEvalError()
        names.append(row.get("channel"))
        rows.append(_validate_accuracy(row.get("accuracy"), evaluations=9))
    if tuple(names) != ("a", "b") or _integer_accuracy_sum(rows) != {
        key: int(overall[key]) for key in _ACCURACY_INTEGER_FIELDS
    }:
        raise MobileAsrEvidenceEvalError()
    paired = value.get("paired_channel")
    if (
        not isinstance(paired, Mapping)
        or set(paired) != {"exact_normalized_output_agreements", "pairs"}
        or paired.get("pairs") != 9
        or type(paired.get("exact_normalized_output_agreements")) is not int
        or not 0 <= paired["exact_normalized_output_agreements"] <= 9
    ):
        raise MobileAsrEvidenceEvalError()


def _validate_overlap_metrics(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "custom_accuracy",
        "scored_mix_evaluations",
        "unscored_stem_executions",
    }:
        raise MobileAsrEvidenceEvalError()
    if (
        value.get("scored_mix_evaluations") != 3
        or value.get("unscored_stem_executions") != 6
    ):
        raise MobileAsrEvidenceEvalError()
    custom = value.get("custom_accuracy")
    fields = {
        "deletions",
        "hypothesis_words",
        "insertions",
        "protocol",
        "reference_words",
        "substitutions",
        "wer",
        "word_errors",
    }
    if not isinstance(custom, Mapping) or set(custom) != fields:
        raise MobileAsrEvidenceEvalError()
    counts = (
        "deletions",
        "hypothesis_words",
        "insertions",
        "reference_words",
        "substitutions",
        "word_errors",
    )
    if any(type(custom.get(key)) is not int or custom[key] < 0 for key in counts):
        raise MobileAsrEvidenceEvalError()
    if (
        custom.get("protocol") != OVERLAP_METRIC
        or custom["word_errors"]
        != custom["substitutions"] + custom["insertions"] + custom["deletions"]
        or custom["deletions"] > custom["reference_words"]
        or custom["insertions"] > custom["hypothesis_words"]
        or custom["substitutions"] > custom["reference_words"] - custom["deletions"]
        or custom["substitutions"] > custom["hypothesis_words"] - custom["insertions"]
        or custom["hypothesis_words"]
        != custom["reference_words"] - custom["deletions"] + custom["insertions"]
    ):
        raise MobileAsrEvidenceEvalError()
    _validate_finite_ratio(
        custom.get("wer"),
        expected=_expected_rate(
            int(custom["word_errors"]),
            int(custom["reference_words"]),
            int(custom["hypothesis_words"]),
        ),
    )


def _validate_component_metrics(
    component: str,
    value: object,
    *,
    endpoint: Mapping[str, object],
) -> None:
    if component == "command-noise":
        _validate_command_metrics(value, endpoint=endpoint)
    elif component == "demand-noise":
        _validate_demand_metrics(value)
    elif component == "notsofar-far-field":
        _validate_notsofar_metrics(value)
    elif component == "primock-isolated":
        if not isinstance(value, Mapping) or set(value) != {"accuracy"}:
            raise MobileAsrEvidenceEvalError()
        _validate_accuracy(value.get("accuracy"), evaluations=3)
    elif component == "primock-overlap":
        _validate_overlap_metrics(value)
    else:
        raise MobileAsrEvidenceEvalError()


def _validate_report(value: object, *, production_inputs: bool) -> dict[str, object]:
    root_fields = {
        "binding_sha256",
        "bindings",
        "complete",
        "components",
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
        raise MobileAsrEvidenceEvalError()
    unsigned = dict(value)
    observed_binding = unsigned.pop("binding_sha256")
    if _sha256(observed_binding) != _canonical_sha256(unsigned):
        raise MobileAsrEvidenceEvalError()
    bindings = value.get("bindings")
    required_bindings = {
        "adapter",
        "artifact_set_sha256",
        "evaluator_closure_sha256",
        "mobile_config_sha256",
        "model_id",
        "model_manifest_sha256",
        "model_total_size_bytes",
        "packet_index_sha256",
        "packet_lock_sha256",
        "packet_receipt_sha256",
        "provision_receipt_sha256",
        "source_bundle_files",
        "source_bundle_sha256",
    }
    if not production_inputs:
        required_bindings = {
            "adapter",
            "artifact_set_sha256",
            "evaluator_closure_sha256",
            "mobile_config_sha256",
            "model_id",
            "model_manifest_sha256",
            "model_total_size_bytes",
            "packet_index_sha256",
            "packet_lock_sha256",
            "packet_receipt_sha256",
            "provision_receipt_sha256",
            "source_bundle_files",
            "source_bundle_sha256",
        }
    if not isinstance(bindings, Mapping) or set(bindings) != required_bindings:
        raise MobileAsrEvidenceEvalError()
    for key in (
        "artifact_set_sha256",
        "evaluator_closure_sha256",
        "mobile_config_sha256",
        "model_manifest_sha256",
        "packet_index_sha256",
        "packet_lock_sha256",
        "packet_receipt_sha256",
        "provision_receipt_sha256",
        "source_bundle_sha256",
    ):
        _sha256(bindings.get(key))
    if (
        type(bindings.get("source_bundle_files")) is not int
        or bindings["source_bundle_files"] <= 0
        or type(bindings.get("model_total_size_bytes")) is not int
        or bindings["model_total_size_bytes"] <= 0
    ):
        raise MobileAsrEvidenceEvalError()
    if production_inputs:
        if (
            bindings.get("adapter") != ADAPTER
            or bindings.get("artifact_set_sha256") != MODEL_ARTIFACT_SET_SHA256
            or bindings.get("mobile_config_sha256") != MOBILE_CONFIG_SHA256
            or bindings.get("model_id") != MODEL_ID
            or bindings.get("model_total_size_bytes") != MODEL_TOTAL_SIZE_BYTES
            or bindings.get("packet_index_sha256") != PACKET_INDEX_SHA256
            or bindings.get("packet_lock_sha256") != PACKET_LOCK_SHA256
            or bindings.get("packet_receipt_sha256") != PACKET_RECEIPT_SHA256
            or bindings.get("source_bundle_files") != len(WORKER_SOURCE_FILES)
        ):
            raise MobileAsrEvidenceEvalError()
    elif (
        bindings.get("adapter") != "synthetic-test-only"
        or bindings.get("model_id") != "synthetic-test-only"
        or bindings.get("model_total_size_bytes") != 1
        or bindings.get("source_bundle_files") != 1
        or len(
            {
                bindings.get("artifact_set_sha256"),
                bindings.get("evaluator_closure_sha256"),
                bindings.get("mobile_config_sha256"),
                bindings.get("model_manifest_sha256"),
                bindings.get("provision_receipt_sha256"),
                bindings.get("source_bundle_sha256"),
            }
        )
        != 1
    ):
        raise MobileAsrEvidenceEvalError()
    execution = value.get("execution")
    if not isinstance(execution, Mapping) or set(execution) != {
        "components",
        "endpoint_integrity",
        "evaluations",
        "logical_cases",
        "pcm_bytes",
        "pcm_inputs",
        "samples",
        "stream",
        "worker_stderr",
    }:
        raise MobileAsrEvidenceEvalError()
    if (
        execution.get("components") != 5
        or execution.get("evaluations") != 129
        or execution.get("logical_cases") != 123
        or execution.get("pcm_inputs") != 129
        or type(execution.get("pcm_bytes")) is not int
        or type(execution.get("samples")) is not int
        or execution["samples"] * 4 != execution["pcm_bytes"]
        or (
            production_inputs
            and (
                execution["pcm_bytes"] != 39_299_500
                or execution["samples"] != 9_824_875
            )
        )
        or (
            not production_inputs
            and (execution["pcm_bytes"] != 516 or execution["samples"] != 129)
        )
        or execution.get("stream")
        != {
            "chunk_samples": 1_600,
            "pace": "burst",
            "partial_report_cadence_ms": 100,
            "protocol_version": 5,
            "repeats": 1,
            "tail_padding_samples": 48_000,
        }
    ):
        raise MobileAsrEvidenceEvalError()
    _validate_endpoint(execution.get("endpoint_integrity"), evaluations=129)
    stderr = execution.get("worker_stderr")
    if (
        not isinstance(stderr, Mapping)
        or set(stderr) != {"bytes", "sha256", "truncated"}
        or type(stderr.get("bytes")) is not int
        or stderr["bytes"] < 0
        or _sha256(stderr.get("sha256")) != stderr.get("sha256")
        or type(stderr.get("truncated")) is not bool
    ):
        raise MobileAsrEvidenceEvalError()
    components = value.get("components")
    if not isinstance(components, list) or len(components) != 5:
        raise MobileAsrEvidenceEvalError()
    component_endpoints: list[Mapping[str, object]] = []
    for row, expected in zip(components, _EXPECTED_COMPONENTS, strict=True):
        if not isinstance(row, Mapping) or set(row) != {
            "component",
            "endpoint_integrity",
            "evaluations",
            "metric_domain",
            "metrics",
        }:
            raise MobileAsrEvidenceEvalError()
        if (
            row.get("component") != expected[0]
            or row.get("metric_domain") != expected[1]
            or row.get("evaluations") != expected[3]
        ):
            raise MobileAsrEvidenceEvalError()
        endpoint = row.get("endpoint_integrity")
        _validate_endpoint(endpoint, evaluations=expected[3])
        if not isinstance(endpoint, Mapping):
            raise MobileAsrEvidenceEvalError()
        component_endpoints.append(endpoint)
        _validate_component_metrics(
            expected[0],
            row.get("metrics"),
            endpoint=endpoint,
        )
    combined_endpoints = {
        key: sum(int(endpoint[key]) for endpoint in component_endpoints)
        for key in execution["endpoint_integrity"]
    }
    if combined_endpoints != execution["endpoint_integrity"]:
        raise MobileAsrEvidenceEvalError()
    evidence = value.get("evidence_scope")
    if not isinstance(evidence, Mapping) or set(evidence) != {
        *_EVIDENCE_SCOPE,
        "mobile_model_identity_authority",
        "model_executed",
        "production_packet",
        "unpooled_component_result_authority",
    }:
        raise MobileAsrEvidenceEvalError()
    if (
        any(
            evidence.get(key) is not expected
            for key, expected in _EVIDENCE_SCOPE.items()
        )
        or evidence.get("mobile_model_identity_authority") is not production_inputs
        or evidence.get("model_executed") is not production_inputs
        or evidence.get("production_packet") is not production_inputs
        or evidence.get("unpooled_component_result_authority") is not production_inputs
    ):
        raise MobileAsrEvidenceEvalError()
    scalars = _walk_scalars(value)
    forbidden_keys = {"case_id", "path", "reference", "transcript"}
    if any(type(item) is float and not math.isfinite(item) for item in scalars) or any(
        item in forbidden_keys for item in scalars if type(item) is str
    ):
        raise MobileAsrEvidenceEvalError()
    encoded = _canonical_json(value, newline=True)
    if len(encoded) > _MAX_REPORT_BYTES:
        raise MobileAsrEvidenceEvalError()
    return value


def _publication_signal_numbers() -> tuple[int, ...]:
    raw = (
        getattr(signal, "SIGINT", None),
        getattr(signal, "SIGHUP", None),
        getattr(signal, "SIGTERM", None),
    )
    if any(isinstance(value, bool) or not isinstance(value, int) for value in raw):
        raise MobileAsrEvidenceEvalError()
    normalized = tuple(int(value) for value in raw)
    if (
        not normalized
        or len(set(normalized)) != len(normalized)
        or any(type(value) is not int or value <= 0 for value in normalized)
    ):
        raise MobileAsrEvidenceEvalError()
    return normalized


def _publication_parent_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return _snapshot(metadata)[:6]


def _linked_report_matches(
    *,
    path: Path,
    parent: Path,
    directory_fd: int,
    descriptor: int,
    parent_identity: tuple[int, ...],
    encoded: bytes,
    digest: str,
) -> bool:
    try:
        directory = os.fstat(directory_fd)
        lexical_parent = parent.lstat()
        current = os.fstat(descriptor)
        published = os.stat(
            path.name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        lexical = path.lstat()
        os.lseek(descriptor, 0, os.SEEK_SET)
        observed = os.read(descriptor, len(encoded) + 1)
        return (
            _publication_parent_identity(directory) == parent_identity
            and _publication_parent_identity(lexical_parent) == parent_identity
            and path.parent == parent
            and (current.st_dev, current.st_ino)
            == (published.st_dev, published.st_ino)
            == (lexical.st_dev, lexical.st_ino)
            and stat.S_ISREG(current.st_mode)
            and stat.S_ISREG(published.st_mode)
            and stat.S_ISREG(lexical.st_mode)
            and stat.S_IMODE(current.st_mode) == 0o600
            and stat.S_IMODE(published.st_mode) == 0o600
            and stat.S_IMODE(lexical.st_mode) == 0o600
            and current.st_uid == os.getuid()
            and published.st_uid == os.getuid()
            and lexical.st_uid == os.getuid()
            and current.st_nlink == 1
            and published.st_nlink == 1
            and lexical.st_nlink == 1
            and current.st_size == len(encoded)
            and published.st_size == len(encoded)
            and lexical.st_size == len(encoded)
            and observed == encoded
            and hashlib.sha256(observed).hexdigest() == digest
        )
    except (OSError, RuntimeError, ValueError):
        return False


def _publish_report(
    path: Path,
    report: Mapping[str, object],
    *,
    guard: Callable[[], None],
    state: _CommitState,
) -> str:
    descriptor = -1
    try:
        if state.linked or state.committed or state.digest:
            raise MobileAsrEvidenceEvalError()
        parent = _stable_private_directory(path.parent)
        if path.parent != parent or path.exists() or path.is_symlink():
            raise MobileAsrEvidenceEvalError()
        encoded = _canonical_json(report, newline=True)
        digest = hashlib.sha256(encoded).hexdigest()
        with opened_directory_nofollow(parent, require_private=True) as (
            stable,
            directory_fd,
        ):
            if stable != parent:
                raise MobileAsrEvidenceEvalError()
            before_parent = _snapshot(os.fstat(directory_fd))
            parent_identity = _publication_parent_identity(os.fstat(directory_fd))
            temporary = getattr(os, "O_TMPFILE", 0)
            if not temporary:
                raise MobileAsrEvidenceEvalError()
            descriptor = os.open(
                ".",
                os.O_RDWR | os.O_CLOEXEC | temporary,
                0o600,
                dir_fd=directory_fd,
            )
            os.fchmod(descriptor, 0o600)
            written = 0
            while written < len(encoded):
                count = os.write(descriptor, encoded[written:])
                if type(count) is not int or count <= 0:
                    raise MobileAsrEvidenceEvalError()
                written += count
            os.fsync(descriptor)
            guard()
            os.lseek(descriptor, 0, os.SEEK_SET)
            observed = os.read(descriptor, len(encoded) + 1)
            metadata = os.fstat(descriptor)
            if (
                observed != encoded
                or hashlib.sha256(observed).hexdigest() != digest
                or not stat.S_ISREG(metadata.st_mode)
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or metadata.st_nlink != 0
                or metadata.st_uid != os.getuid()
                or metadata.st_size != len(encoded)
                or _snapshot(parent.lstat()) != before_parent
            ):
                raise MobileAsrEvidenceEvalError()
            previous = signal.pthread_sigmask(
                signal.SIG_BLOCK, _publication_signal_numbers()
            )
            try:
                try:
                    os.link(
                        f"/proc/self/fd/{descriptor}",
                        path.name,
                        dst_dir_fd=directory_fd,
                        follow_symlinks=True,
                    )
                except BaseException:
                    if not _linked_report_matches(
                        path=path,
                        parent=parent,
                        directory_fd=directory_fd,
                        descriptor=descriptor,
                        parent_identity=parent_identity,
                        encoded=encoded,
                        digest=digest,
                    ):
                        raise
                if not _linked_report_matches(
                    path=path,
                    parent=parent,
                    directory_fd=directory_fd,
                    descriptor=descriptor,
                    parent_identity=parent_identity,
                    encoded=encoded,
                    digest=digest,
                ):
                    raise MobileAsrEvidenceEvalError()
                state.linked = True
                os.fsync(directory_fd)
                if not _linked_report_matches(
                    path=path,
                    parent=parent,
                    directory_fd=directory_fd,
                    descriptor=descriptor,
                    parent_identity=parent_identity,
                    encoded=encoded,
                    digest=digest,
                ):
                    raise MobileAsrEvidenceEvalError()
                state.committed = True
                state.digest = digest
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous)
            if not state.committed:
                raise MobileAsrEvidenceEvalError()
        return digest
    except BaseException as error:
        if state.committed:
            return state.digest
        if isinstance(error, KeyboardInterrupt):
            raise
        raise MobileAsrEvidenceEvalError() from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _load_report_common(
    path: Path | str,
    *,
    expected_sha256: str,
    production_inputs: bool,
) -> LoadedMobileAsrEvidenceEvalReport:
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
            raise MobileAsrEvidenceEvalError()
        value = _strict_json(snapshot.data)
        if _canonical_json(value, newline=True) != snapshot.data:
            raise MobileAsrEvidenceEvalError()
        report = _validate_report(value, production_inputs=production_inputs)
        bindings = report["bindings"]
        if not isinstance(bindings, Mapping):
            raise MobileAsrEvidenceEvalError()
        loaded = LoadedMobileAsrEvidenceEvalReport(
            path=snapshot.path,
            digest=expected,
            packet_index_sha256=str(bindings["packet_index_sha256"]),
            packet_receipt_sha256=str(bindings["packet_receipt_sha256"]),
            model_manifest_sha256=str(bindings["model_manifest_sha256"]),
            source_bundle_sha256=str(bindings["source_bundle_sha256"]),
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
            raise MobileAsrEvidenceEvalError()
        return loaded
    except MobileAsrEvidenceEvalError:
        raise
    except Exception:
        raise MobileAsrEvidenceEvalError() from None


def load_mobile_asr_evidence_eval_report(
    path: Path | str,
    *,
    expected_sha256: str,
    packet_index_sha256: str,
    packet_receipt_sha256: str,
    model_manifest_sha256: str,
) -> LoadedMobileAsrEvidenceEvalReport:
    """Load only a pinned production report and recheck its exact bindings."""

    loaded = _load_report_common(
        path,
        expected_sha256=expected_sha256,
        production_inputs=True,
    )
    if (
        loaded.packet_index_sha256 != _sha256(packet_index_sha256)
        or loaded.packet_receipt_sha256 != _sha256(packet_receipt_sha256)
        or loaded.model_manifest_sha256 != _sha256(model_manifest_sha256)
    ):
        raise MobileAsrEvidenceEvalError()
    return loaded


def verify_mobile_asr_evidence_eval_report(
    report: LoadedMobileAsrEvidenceEvalReport,
) -> None:
    """Close-revalidate one previously loaded production report."""

    if type(report) is not LoadedMobileAsrEvidenceEvalReport:
        raise MobileAsrEvidenceEvalError()
    current = load_mobile_asr_evidence_eval_report(
        report.path,
        expected_sha256=report.digest,
        packet_index_sha256=report.packet_index_sha256,
        packet_receipt_sha256=report.packet_receipt_sha256,
        model_manifest_sha256=report.model_manifest_sha256,
    )
    if current != report:
        raise MobileAsrEvidenceEvalError()


def _ready_matches(ready: object, model: _ModelAuthority, bundle: SourceBundle) -> bool:
    return (
        getattr(ready, "protocol_version", None) == PROTOCOL_VERSION
        and getattr(ready, "model_id", None) == MODEL_ID
        and getattr(ready, "adapter", None) == ADAPTER
        and getattr(ready, "manifest_sha256", None) == model.manifest_sha256
        and getattr(ready, "source_bundle_sha256", None) == bundle.tree_sha256
    )


def run_mobile_asr_evidence_eval(
    packet_root: Path | str,
    *,
    packet_index_sha256: str,
    packet_receipt_sha256: str,
    worker_manifest: Path | str,
    scratch_root: Path | str,
    output_path: Path | str,
) -> LoadedMobileAsrEvidenceEvalReport:
    """Run one fixed sequential 129-leaf production evaluation."""

    state = _CommitState()
    try:
        inputs = _load_production_inputs(
            packet_root,
            packet_index_sha256=packet_index_sha256,
            packet_receipt_sha256=packet_receipt_sha256,
        )
        model = _load_model_authority(worker_manifest)
        scratch, output, scratch_parent_snapshot, output_parent_snapshot = (
            _validate_destinations(
                inputs,
                model,
                scratch_root=scratch_root,
                output_path=output_path,
            )
        )
        closure = _evaluator_closure_sha256()
        worker_root, pcm_paths, source_bundle, scratch_snapshot = _prepare_scratch(
            scratch, inputs
        )
        if _snapshot(scratch.parent.lstat())[:5] != scratch_parent_snapshot[:5]:
            raise MobileAsrEvidenceEvalError()
        scratch_parent_snapshot = _snapshot(scratch.parent.lstat())
        from tools import streaming_stt_eval
        from tools.streaming_stt.supervisor import StreamingWorker

        run_lock = streaming_stt_eval._BenchmarkRunLock.acquire(
            streaming_stt_eval._HOST_LOCK_PATH
        )
        worker = None
        try:
            run_lock.assert_bound()
            worker = StreamingWorker(model.manifest, worker_root, source_bundle)
            ready = worker.start()
            if not _ready_matches(ready, model, source_bundle):
                raise MobileAsrEvidenceEvalError()

            leaves = tuple(
                leaf for component in inputs.components for leaf in component.leaves
            )

            def transcribe(leaf: _Leaf, index: int) -> _EndpointOutcome:
                if leaves[index] is not leaf:
                    raise MobileAsrEvidenceEvalError()
                request = TranscribeRequest(
                    request_id=f"mobile-{index:03d}",
                    pcm=PcmInput(
                        path=pcm_paths[index].resolve(strict=True),
                        sha256=leaf.sha256,
                        samples=leaf.samples,
                    ),
                    stream=_STREAM,
                    protocol_version=PROTOCOL_VERSION,
                )
                return _extract_endpoint_outcome(worker.transcribe(request), leaf)

            components, endpoints = _reduce_components(inputs, transcribe)
            worker.close()
            if not worker.fully_stopped or not _ready_matches(
                worker.ready, model, source_bundle
            ):
                raise MobileAsrEvidenceEvalError()
            stderr = dict(worker.stderr_summary)
            worker = None
        finally:
            if worker is not None:
                try:
                    worker.close()
                except Exception:
                    pass
            run_lock.close()
        binding = _worker_binding(model, source_bundle)
        report = _report(
            inputs,
            components,
            endpoints,
            binding=binding,
            evaluator_closure_sha256=closure,
            worker_stderr=stderr,
            model_executed=True,
        )

        def guard() -> None:
            if inputs.packet is None:
                raise MobileAsrEvidenceEvalError()
            packet_api.verify_mobile_asr_evidence_packet(inputs.packet)
            _verify_model_authority(model)
            if _evaluator_closure_sha256() != closure:
                raise MobileAsrEvidenceEvalError()
            _verify_scratch(scratch, scratch_snapshot, source_bundle, pcm_paths, inputs)
            run_lock_path = output.parent
            if (
                _snapshot(scratch.parent.lstat()) != scratch_parent_snapshot
                or _snapshot(run_lock_path.lstat()) != output_parent_snapshot
            ):
                raise MobileAsrEvidenceEvalError()
            _validate_report(report, production_inputs=True)

        digest = _publish_report(output, report, guard=guard, state=state)
        loaded = load_mobile_asr_evidence_eval_report(
            output,
            expected_sha256=digest,
            packet_index_sha256=inputs.packet_index_sha256,
            packet_receipt_sha256=inputs.packet_receipt_sha256,
            model_manifest_sha256=model.manifest_sha256,
        )
        verify_mobile_asr_evidence_eval_report(loaded)
        return loaded
    except BaseException as error:
        if state.committed:
            try:
                return load_mobile_asr_evidence_eval_report(
                    output_path,
                    expected_sha256=state.digest,
                    packet_index_sha256=packet_index_sha256,
                    packet_receipt_sha256=packet_receipt_sha256,
                    model_manifest_sha256=getattr(
                        locals().get("model"), "manifest_sha256"
                    ),
                )
            except Exception:
                pass
        if isinstance(error, KeyboardInterrupt):
            raise
        if isinstance(error, MobileAsrEvidenceEvalError):
            raise
        raise MobileAsrEvidenceEvalError() from None


def _synthetic_inputs() -> _Inputs:
    audio = b"\x00\x00\x00\x00"
    digest = hashlib.sha256(audio).hexdigest()
    components: list[_Component] = []
    command: list[_Leaf] = []
    for index in range(57):
        if index == 0:
            assertion = "transcript"
            reference = "alpha beta"
            commands = ("alpha beta",)
            forbidden = ("gamma",)
        elif index % 3 == 0:
            assertion = "silence"
            reference = ""
            commands = ()
            forbidden = ("alpha beta",)
        else:
            assertion = "speech_negative"
            reference = ""
            commands = ()
            forbidden = ("alpha beta",)
        command.append(
            _Leaf(
                "command-noise",
                "corpus",
                digest,
                1,
                audio,
                reference,
                assertion=assertion,
                commands=commands,
                forbidden_commands=forbidden,
                group_index=index,
            )
        )
    components.append(
        _Component("command-noise", "command-assertion-v1", 57, tuple(command))
    )
    demand: list[_Leaf] = []
    for index in range(42):
        tags = (
            _DEMAND_STRATA[index % 3],
            _DEMAND_STRATA[3 + (index % 3)],
        )
        demand.append(
            _Leaf(
                "demand-noise",
                "corpus",
                digest,
                1,
                audio,
                "delta echo",
                tags=tags,
                group_index=index,
            )
        )
    components.append(
        _Component("demand-noise", "stratified-wer-cer-v1", 42, tuple(demand))
    )
    notsofar: list[_Leaf] = []
    for index in range(9):
        for channel in ("a", "b"):
            notsofar.append(
                _Leaf(
                    "notsofar-far-field",
                    "corpus",
                    digest,
                    1,
                    audio,
                    "foxtrot golf",
                    tags=(f"pair-{index:02d}", f"channel-{channel}"),
                    group_index=index,
                )
            )
    components.append(
        _Component(
            "notsofar-far-field", "paired-channel-wer-cer-v1", 18, tuple(notsofar)
        )
    )
    isolated = tuple(
        _Leaf(
            "primock-isolated",
            "corpus",
            digest,
            1,
            audio,
            "hotel india",
            group_index=index,
        )
        for index in range(3)
    )
    components.append(
        _Component("primock-isolated", "ordinary-wer-cer-v1", 3, isolated)
    )
    overlap: list[_Leaf] = []
    for index in range(3):
        for kind in ("role_a", "role_b", "mix"):
            overlap.append(
                _Leaf(
                    "primock-overlap",
                    kind,
                    digest,
                    1,
                    audio,
                    "juliet"
                    if kind == "role_a"
                    else "kilo"
                    if kind == "role_b"
                    else "",
                    reference_a="juliet",
                    reference_b="kilo",
                    group_index=index,
                )
            )
    components.append(_Component("primock-overlap", OVERLAP_METRIC, 3, tuple(overlap)))
    return _Inputs(
        components=tuple(components),
        production_inputs=False,
        packet_lock_sha256=hashlib.sha256(b"synthetic-lock").hexdigest(),
        packet_index_sha256=hashlib.sha256(b"synthetic-index").hexdigest(),
        packet_receipt_sha256=hashlib.sha256(b"synthetic-receipt").hexdigest(),
        pcm_bytes=129 * 4,
        samples=129,
    )


def _run_synthetic_test_evaluation(
    contract: _SyntheticWorkerContract,
    output_path: Path | str,
) -> LoadedMobileAsrEvidenceEvalReport:
    """Exercise the reducer/publication lifecycle with nonproduction authority."""

    if type(contract) is not _SyntheticWorkerContract:
        raise MobileAsrEvidenceEvalError()
    inputs = _synthetic_inputs()
    worker = _SyntheticNonproductionWorker(contract)

    def transcribe(leaf: _Leaf, _index: int) -> _EndpointOutcome:
        return worker.transcribe(leaf)

    components, endpoints = _reduce_components(inputs, transcribe)
    worker.finish()
    authority = _canonical_sha256(
        [
            {
                "endpoint_count": len(value.endpoint_texts),
                "endpoint_reason": value.endpoint_reason,
                "source_complete": value.source_complete,
            }
            for value in contract.outputs
        ]
    )
    binding = {
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
    report = _report(
        inputs,
        components,
        endpoints,
        binding=binding,
        evaluator_closure_sha256=authority,
        worker_stderr={
            "bytes": 0,
            "sha256": hashlib.sha256(b"").hexdigest(),
            "truncated": False,
        },
        model_executed=False,
    )
    output, _parent, _snapshot_parent = _new_synthetic_output(output_path)
    state = _CommitState()
    digest = _publish_report(
        output,
        report,
        guard=lambda: _validate_report(report, production_inputs=False),
        state=state,
    )
    return _load_report_common(
        output,
        expected_sha256=digest,
        production_inputs=False,
    )


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise MobileAsrEvidenceEvalError()


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        description="Run the fixed five-component mobile-ASR evidence evaluation."
    )
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--packet-root", type=Path, required=True)
    parser.add_argument("--packet-index-sha256", required=True)
    parser.add_argument("--packet-receipt-sha256", required=True)
    parser.add_argument("--worker-manifest", type=Path, required=True)
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
            "worker_manifest": args.worker_manifest,
            "scratch_root": args.scratch_root,
            "output_path": args.output,
        }
        if args.preflight:
            result = preflight_mobile_asr_evidence_eval(args.packet_root, **kwargs)
        else:
            loaded = run_mobile_asr_evidence_eval(args.packet_root, **kwargs)
            result = {
                "evaluations": 129,
                "kind": "mobile-asr-evidence-eval-publication-v1",
                "ok": True,
                "report_sha256": loaded.digest,
            }
        return 0 if _emit_public(result) else 2
    except KeyboardInterrupt:
        return 130
    except (MobileAsrEvidenceEvalError, OSError, RuntimeError, TypeError, ValueError):
        _emit_public(_SAFE_ERROR)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LoadedMobileAsrEvidenceEvalReport",
    "MobileAsrEvidenceEvalError",
    "load_mobile_asr_evidence_eval_report",
    "main",
    "preflight_mobile_asr_evidence_eval",
    "run_mobile_asr_evidence_eval",
    "verify_mobile_asr_evidence_eval_report",
]
