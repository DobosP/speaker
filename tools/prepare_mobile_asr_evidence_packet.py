"""Prepare the exact five-component English mobile-ASR evidence packet.

The packet is a byte-for-byte copy of five already prepared, locally retained
components.  This command never downloads, rematerializes, or evaluates audio.
Component metric domains remain independent: no packet-level WER is defined.

The committed component receipt hashes are input authority.  In particular,
the DEMAND preparation receipt binds the runtime that created it.  A newly
materialized receipt is not learned or accepted at the command line; accepting
one requires review and a new packet lock.  The newly written packet receipt is
a runtime result, not component-selection authority.  Any later evaluator must
pin both the packet index and packet receipt hashes.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import sys
from typing import Callable, Final, Mapping, Sequence

from tools.streaming_stt.bounded_io import BoundedReadError, read_regular_bounded
from tools.streaming_stt.corpus import (
    CorpusError,
    LoadedCorpus,
    _strict_json,
    load_corpus,
    verify_corpus_snapshot,
)
from tools.streaming_stt.corpus_writer import (
    _new_private_output,
    _read_exact_written_private,
    _verify_output_binding,
    _verify_written_private_metadata,
    _write_new_private,
)
from tools.streaming_stt.protocol import MAX_PCM_BYTES


SCHEMA_VERSION: Final = 2
LOCK_KIND: Final = "mobile-asr-evidence-packet-lock-v2"
PACKET_ID: Final = "mobile-asr-english-five-component-v2"
LOCK_RECIPE_SHA256: Final = (
    "3bd6600b914876caeea3c58e82b00f0dfe0c8b8951edc0f4affb28b152e48c9d"
)
LOCK_FILE_SHA256: Final = (
    "c81a89a7a7df534ec0a23dc20c0a915c4aa5eedec6270a13ec08456a0683cc11"
)
LOCK_FILE_BYTES: Final = 8481
DEFAULT_LOCK: Final = (
    Path(__file__).resolve().parent
    / "streaming_stt"
    / "mobile-asr-evidence-packet-v2.lock.json"
)

_MAX_LOCK_BYTES = 64 * 1024
_MAX_MANIFEST_BYTES = 256 * 1024
_MAX_RECEIPT_BYTES = 256 * 1024
_MAX_PACKET_METADATA_BYTES = 256 * 1024
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_LEAF_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,127}\Z")
_EXPECTED_ORDER = (
    "command-noise",
    "demand-noise",
    "notsofar-far-field",
    "primock-isolated",
    "primock-overlap",
)
_EXPECTED_CONTRACT = {
    "command-noise": (
        "command-noise",
        "production-final-command-noise-schema-v4",
        "streaming-stt-corpus-v4",
        "corpus.json",
        "command-assertion-v1",
        57,
        57,
        3_714_904,
    ),
    "demand-noise": (
        "demand-noise",
        "streaming-stt-corpus-schema-v2",
        "streaming-stt-corpus-v2",
        "corpus.json",
        "stratified-wer-cer-v1",
        42,
        42,
        26_230_128,
    ),
    "notsofar-far-field": (
        "notsofar-far-field",
        "streaming-stt-corpus-schema-v2",
        "streaming-stt-corpus-v2",
        "corpus.json",
        "paired-channel-wer-cer-v1",
        18,
        18,
        2_309_120,
    ),
    "primock-isolated": (
        "primock-isolated",
        "primock57-isolated-bundle-v1",
        "streaming-stt-corpus-v2",
        "corpus.json",
        "ordinary-wer-cer-v1",
        3,
        3,
        284_668,
    ),
    "primock-overlap": (
        "primock-overlap",
        "primock57-overlap-bundle-v1",
        "primock57-overlap-custom-v1",
        "primock57-two-role-overlap-diagnostic-v1.json",
        "two-utterance-min-order-wer-v1",
        3,
        9,
        6_760_680,
    ),
}
_EXPECTED_TOTALS = {
    "components": 5,
    "logical_cases": 123,
    "pcm_bytes": 39_299_500,
    "pcm_inputs": 129,
    "samples": 9_824_875,
}
_EXPECTED_METRIC_AGGREGATION = {
    "component_domains_are_independent": True,
    "pooled_metric": None,
    "pooled_wer": False,
}
_EXPECTED_RUNTIME_RECEIPT_POLICY = {
    "component_hash_authority": "committed-lock-manifest-and-receipt-digests-only",
    "computed_component_hashes": "compare-only-never-cli-supplied-or-learned",
    "packet_receipt": "runtime-result-not-component-selection-authority",
    "packet_reuse": "future-evaluator-must-pin-packet-index-and-packet-receipt-sha256",
}
_EXPECTED_LANGUAGE = {"bcp47": "en", "english_only": True}
_EXPECTED_PUBLICATION = {
    "component_materialization": "exact-byte-copy-v1",
    "external_path_references": False,
    "files": "mode-0600-single-link",
    "redistribution": False,
    "root": "fresh-outside-git-mode-0700",
    "storage": "private-cache",
    "terminal_receipt": "last-no-clobber-completion-marker-v1",
}
_EXPECTED_PRIVACY = {
    "packet_index": "aggregate-path-transcript-error-free-v1",
    "packet_receipt": "aggregate-path-transcript-error-free-v1",
    "stdout": "aggregate-path-transcript-error-free-v1",
}
_PRIMOCK_FIXTURE_ID = "primock57-consultation01-v1"
_PRIMOCK_LICENSE_ID = "CC-BY-4.0"
_PRIMOCK_LOCK_RECIPE_SHA256 = (
    "2bc14c4114959fe1323d843ed6bb1b17cf81aa46f620b6805a98dd380a1e17ef"
)
_PRIMOCK_ISOLATED_SELECTION_SHA256 = (
    "9de94461889750796720f658b7ef3109179d075ef86d2b7cda2fc608ddae640a"
)
_PRIMOCK_OVERLAP_SELECTION_SHA256 = (
    "f91c56ec722a8958802fedac92ba45d04920df534b4c6a9bd90cb2f1cb8bb05e"
)
_PRIMOCK_ISOLATED_SOURCE_CONTRACT_SHA256 = (
    "6c098e29916a98073cfa89b546e9eb85d140392e20e4aeb0bfc5ad490bdf139b"
)
_PRIMOCK_OVERLAP_SOURCE_CONTRACT_SHA256 = (
    "506a1e24e44e26f724952a18de95a5f7f862c65c910c3943671d2b0d90806646"
)
_PRIMOCK_SOURCE_ROWS_SHA256 = (
    "fc2edca789bc69a6ce5f2ea5530782091685469415565051033c6f13c1e0d641"
)
_PRIMOCK_PREPARER_ROWS_SHA256 = (
    "f4a032fb8595686897183f16ed75e8c62de2714b76a48feac1ee0e370199b116"
)
_PRIMOCK_SOURCE_LOCK_BYTES = 8_181
_PRIMOCK_SOURCE_LOCK_SHA256 = (
    "058da5caf533a771b276f81323c5f011d35ac406fa4f91c51cec5a0ed25c2031"
)
_PRIMOCK_SOURCE_LOCK_PROJECTION_SHA256 = (
    "8d55bc03251933e9cd89280d22b658733c127019c9568c93ea08388934f99e43"
)
_PRIMOCK_SOURCE_LOCK = (
    Path(__file__).resolve().parent
    / "streaming_stt"
    / "primock57-consultation01-v1.lock.json"
)
_PRIMOCK_SOURCE_ROLES = (
    "role-a-audio",
    "role-b-audio",
    "role-a-annotation",
    "role-b-annotation",
    "license",
)
_PRIMOCK_SOURCE_PATHS = (
    "audio/day1_consultation01_doctor.wav",
    "audio/day1_consultation01_patient.wav",
    "transcripts/day1_consultation01_doctor.TextGrid",
    "transcripts/day1_consultation01_patient.TextGrid",
    "LICENSE.md",
)
_PRIMOCK_PREPARER_ROWS = (
    (
        "tools/__init__.py",
        63,
        "ca607a54b592e2879ed4f3daac2fd7aad1a8223087f74986b59cc9d98534c426",
    ),
    (
        "tools/streaming_stt/__init__.py",
        1_682,
        "db337b8d55d685071cee59b0984b3dc1da8505587a78faded46770f6d671e923",
    ),
    (
        "tools/prepare_primock57_conversation_fixture.py",
        119_417,
        "41f498d1b1f4160e160f697bbcd0c48a3f7b5a4d44eeb2cebac934239ff3fbf6",
    ),
    (
        "core/__init__.py",
        1_535,
        "ca29bb32269447ee053a0aec163b2105e7450cd9be5328c460412f03e1ccaa03",
    ),
    (
        "core/wer.py",
        2_816,
        "0e58dabb21985a56322646bab9d89e71a726a73a733593238ed3e4bf792a03d2",
    ),
    (
        "tools/streaming_stt/bounded_io.py",
        10_409,
        "cedf3e81b9fb58ba1f1dc978525a0672db280f52fe0c4648c849af2752a3843e",
    ),
    (
        "tools/streaming_stt/corpus.py",
        15_834,
        "b0f25101f9dad861b66d13582cf80e3a995c241e929a75ee7cdbc035dbe63618",
    ),
    (
        "tools/streaming_stt/corpus_writer.py",
        26_555,
        "42eba5892d6291bfef4685d39ab0c9f380246dbcfb9dc6abba2dc854b910081a",
    ),
    (
        "tools/streaming_stt/private_diagnostic_receipt.py",
        8_996,
        "5b54ae30660279e38f0ce22595cbb8e687cfd804873dabcb142f887d714f5194",
    ),
    (
        "tools/streaming_stt/protocol.py",
        37_701,
        "2f776b24df02dd02a7d9b9c9520afac148d2bddfad2257002227250311ec746c",
    ),
)
_PRIMOCK_RECEIPT_PRIVACY = {
    "local_paths_in_receipt": False,
    "raw_role_labels_in_receipt": False,
    "transcripts_in_receipt": False,
}
_PRIMOCK_ISOLATED_PURPOSE = (
    "PriMock57 consultation 01 marker-free, zero-other-role-intersection "
    "isolated hard WER"
)
_PRIMOCK_OVERLAP_MANIFEST = "primock57-two-role-overlap-diagnostic-v1.json"
_PRIMOCK_OVERLAP_KIND = "primock57-two-role-overlap-diagnostic-v1"
_PRIMOCK_ISOLATED_RECEIPT_KIND = "primock57-isolated-preparation-receipt-v1"
_PRIMOCK_OVERLAP_RECEIPT_KIND = "primock57-overlap-preparation-receipt-v1"
_EXPECTED_EVIDENCE_SCOPE = {
    "default_authority": False,
    "device_authority": False,
    "endpoint_authority": False,
    "evaluation_result_authority": False,
    "gpu_authority": False,
    "held_out_authority": False,
    "latency_authority": False,
    "live_authority": False,
    "microphone_authority": False,
    "mobile_model_identity_authority": False,
    "model_quality_authority": False,
    "multilingual_authority": False,
    "promotion_authority": False,
    "qualification_authority": False,
    "romanian_authority": False,
    "training_disjoint_authority": False,
}
_SOURCE_FLAGS = {
    "--command-noise-root": "command-noise",
    "--demand-noise-root": "demand-noise",
    "--notsofar-far-field-root": "notsofar-far-field",
    "--primock-isolated-root": "primock-isolated",
    "--primock-overlap-root": "primock-overlap",
}
_OUTPUT_FLAG = "--output-root"
_SAFE_ERROR = {
    "error": "mobile_asr_evidence_packet_prerequisites_unavailable",
    "ok": False,
}


class MobileAsrEvidencePacketError(RuntimeError):
    """A detail-free input, validation, or publication failure."""


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    component_id: str
    directory: str
    loader: str
    schema: str
    manifest_file: str
    manifest_sha256: str
    receipt_file: str
    receipt_sha256: str
    licenses: tuple[str, ...]
    logical_cases: int
    pcm_inputs: int
    pcm_bytes: int
    metric_domain: str


@dataclass(frozen=True, slots=True)
class PacketLock:
    raw_sha256: str
    recipe_sha256: str
    components: tuple[ComponentSpec, ...]


@dataclass(frozen=True, slots=True)
class _Artifact:
    name: str
    payload: bytes = field(repr=False)
    sha256: str


@dataclass(frozen=True, slots=True)
class _ValidatedComponent:
    spec: ComponentSpec
    root: Path = field(repr=False)
    artifacts: tuple[_Artifact, ...] = field(repr=False)
    inventory: tuple[tuple[str, tuple[int, ...]], ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class _ParsedCommand:
    action: str
    sources: Mapping[str, str]
    output_root: str | None


@dataclass(frozen=True, slots=True)
class LoadedMobileAsrEvidencePacket:
    """A fully reloaded packet; private bytes and paths stay out of repr."""

    schema_version: int
    packet_id: str
    packet_lock_sha256: str
    packet_index_sha256: str
    packet_receipt_sha256: str
    components: int
    logical_cases: int
    pcm_inputs: int
    pcm_bytes: int
    samples: int
    root: Path = field(repr=False)
    _validated_components: tuple[_ValidatedComponent, ...] = field(repr=False)
    _tree_snapshot: tuple[tuple[str, tuple[int, ...]], ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class _StagedTerminalReceipt:
    name: str
    size_bytes: int
    sha256: str
    snapshot: tuple[int, ...]


@dataclass(slots=True)
class _TerminalCommitState:
    committed: bool = False


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
        raise MobileAsrEvidencePacketError() from None
    return raw + (b"\n" if newline else b"")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256(value: object) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise MobileAsrEvidencePacketError()
    return value


def _safe_leaf(value: object) -> str:
    if not isinstance(value, str) or _SAFE_LEAF_RE.fullmatch(value) is None:
        raise MobileAsrEvidencePacketError()
    return value


def _positive_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MobileAsrEvidencePacketError()
    return value


def _parse_component(value: object) -> ComponentSpec:
    expected_fields = {
        "directory",
        "id",
        "licenses",
        "loader",
        "logical_cases",
        "manifest",
        "metric_domain",
        "pcm_bytes",
        "pcm_inputs",
        "prepared_output_classification",
        "receipt",
        "schema",
        "source_classification",
        "upstream",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise MobileAsrEvidencePacketError()
    component_id = _safe_leaf(value.get("id"))
    manifest = value.get("manifest")
    receipt = value.get("receipt")
    licenses = value.get("licenses")
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"file", "sha256"}
        or not isinstance(receipt, dict)
        or set(receipt) != {"file", "sha256"}
        or not isinstance(licenses, list)
        or not licenses
        or any(not isinstance(item, str) or not item for item in licenses)
        or len(set(licenses)) != len(licenses)
        or value.get("source_classification") != "public-derived"
        or value.get("prepared_output_classification") != "private-prepared-output"
        or not isinstance(value.get("upstream"), dict)
    ):
        raise MobileAsrEvidencePacketError()
    spec = ComponentSpec(
        component_id=component_id,
        directory=_safe_leaf(value.get("directory")),
        loader=_safe_leaf(value.get("loader")),
        schema=_safe_leaf(value.get("schema")),
        manifest_file=_safe_leaf(manifest.get("file")),
        manifest_sha256=_sha256(manifest.get("sha256")),
        receipt_file=_safe_leaf(receipt.get("file")),
        receipt_sha256=_sha256(receipt.get("sha256")),
        licenses=tuple(licenses),
        logical_cases=_positive_int(value.get("logical_cases")),
        pcm_inputs=_positive_int(value.get("pcm_inputs")),
        pcm_bytes=_positive_int(value.get("pcm_bytes")),
        metric_domain=_safe_leaf(value.get("metric_domain")),
    )
    expected = _EXPECTED_CONTRACT.get(component_id)
    if expected != (
        spec.directory,
        spec.loader,
        spec.schema,
        spec.manifest_file,
        spec.metric_domain,
        spec.logical_cases,
        spec.pcm_inputs,
        spec.pcm_bytes,
    ):
        raise MobileAsrEvidencePacketError()
    return spec


def _parse_lock(raw: bytes, *, expected_raw_sha256: str) -> PacketLock:
    try:
        if (
            not raw
            or len(raw) > _MAX_LOCK_BYTES
            or hashlib.sha256(raw).hexdigest() != expected_raw_sha256
        ):
            raise MobileAsrEvidencePacketError()
        value = _strict_json(raw)
        expected_fields = {
            "components",
            "evidence_scope",
            "kind",
            "language",
            "metric_aggregation",
            "order",
            "packet_id",
            "privacy",
            "publication",
            "recipe_digest_rule",
            "recipe_sha256",
            "runtime_receipt_policy",
            "sample_format",
            "schema_version",
            "totals",
        }
        if (
            not isinstance(value, dict)
            or set(value) != expected_fields
            or value.get("schema_version") != SCHEMA_VERSION
            or value.get("kind") != LOCK_KIND
            or value.get("packet_id") != PACKET_ID
            or value.get("recipe_digest_rule")
            != "sha256-canonical-json-without-recipe_sha256-v1"
            or value.get("recipe_sha256") != LOCK_RECIPE_SHA256
            or value.get("order") != list(_EXPECTED_ORDER)
            or value.get("totals") != _EXPECTED_TOTALS
            or value.get("metric_aggregation") != _EXPECTED_METRIC_AGGREGATION
            or value.get("runtime_receipt_policy") != _EXPECTED_RUNTIME_RECEIPT_POLICY
            or value.get("language") != _EXPECTED_LANGUAGE
            or value.get("publication") != _EXPECTED_PUBLICATION
            or value.get("privacy") != _EXPECTED_PRIVACY
            or value.get("evidence_scope") != _EXPECTED_EVIDENCE_SCOPE
            or value.get("sample_format")
            != {"channels": 1, "encoding": "f32le", "sample_rate_hz": 16_000}
        ):
            raise MobileAsrEvidencePacketError()
        digest_value = dict(value)
        digest_value.pop("recipe_sha256")
        if _canonical_sha256(digest_value) != LOCK_RECIPE_SHA256:
            raise MobileAsrEvidencePacketError()
        component_values = value.get("components")
        if not isinstance(component_values, list):
            raise MobileAsrEvidencePacketError()
        components = tuple(_parse_component(item) for item in component_values)
        if (
            tuple(item.component_id for item in components) != _EXPECTED_ORDER
            or len({item.directory for item in components}) != len(components)
            or len({item.metric_domain for item in components}) != len(components)
            or sum(item.logical_cases for item in components) != 123
            or sum(item.pcm_inputs for item in components) != 129
            or sum(item.pcm_bytes for item in components) != 39_299_500
            or any(item.pcm_bytes % 4 for item in components)
        ):
            raise MobileAsrEvidencePacketError()
        demand = components[1]
        demand_value = component_values[1]
        if (
            demand.component_id != "demand-noise"
            or not isinstance(demand_value, dict)
            or not isinstance(demand_value.get("upstream"), dict)
            or demand_value["upstream"].get("receipt_runtime_binding")
            != "exact-current-runtime-receipt-reviewed-v2"
        ):
            raise MobileAsrEvidencePacketError()
        return PacketLock(
            raw_sha256=expected_raw_sha256,
            recipe_sha256=LOCK_RECIPE_SHA256,
            components=components,
        )
    except MobileAsrEvidencePacketError:
        raise
    except (CorpusError, MemoryError, OverflowError, TypeError, ValueError):
        raise MobileAsrEvidencePacketError() from None


def load_packet_lock(path: Path | str = DEFAULT_LOCK) -> PacketLock:
    """Load only the repository's exact committed packet lock."""

    try:
        selected = Path(os.path.abspath(Path(path).expanduser()))
        if (
            selected != DEFAULT_LOCK
            or selected.resolve(strict=True) != selected
            or selected.name != DEFAULT_LOCK.name
        ):
            raise MobileAsrEvidencePacketError()
        snapshot = read_regular_bounded(
            selected,
            maximum_bytes=_MAX_LOCK_BYTES,
            expected_bytes=LOCK_FILE_BYTES,
        )
        return _parse_lock(snapshot.data, expected_raw_sha256=LOCK_FILE_SHA256)
    except MobileAsrEvidencePacketError:
        raise
    except (BoundedReadError, OSError, RuntimeError, TypeError, ValueError):
        raise MobileAsrEvidencePacketError() from None


def _absolute_private_root(path: Path | str) -> Path:
    try:
        from tools import prepare_demand_noise_streaming_stt_corpus as demand

        supplied = Path(path).expanduser()
        if not supplied.is_absolute():
            raise MobileAsrEvidencePacketError()
        candidate = Path(os.path.abspath(supplied))
        metadata = candidate.lstat()
        if (
            candidate.resolve(strict=True) != candidate
            or not stat.S_ISDIR(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o700
            or demand._has_git_ancestor(candidate)
            or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
        ):
            raise MobileAsrEvidencePacketError()
        return candidate
    except MobileAsrEvidencePacketError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError):
        raise MobileAsrEvidencePacketError() from None


def _metadata_snapshot(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        metadata.st_nlink,
        stat.S_IMODE(metadata.st_mode),
        metadata.st_uid,
    )


def _inventory(
    root: Path,
    artifacts: tuple[_Artifact, ...],
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    try:
        root_metadata = root.lstat()
        if (
            root.resolve(strict=True) != root
            or not stat.S_ISDIR(root_metadata.st_mode)
            or stat.S_IMODE(root_metadata.st_mode) != 0o700
            or (hasattr(os, "geteuid") and root_metadata.st_uid != os.geteuid())
        ):
            raise MobileAsrEvidencePacketError()
        names = [item.name for item in artifacts]
        if len(set(names)) != len(names) or set(os.listdir(root)) != set(names):
            raise MobileAsrEvidencePacketError()
        rows: list[tuple[str, tuple[int, ...]]] = [
            (".", _metadata_snapshot(root_metadata))
        ]
        identities: set[tuple[int, int]] = set()
        by_name = {item.name: item for item in artifacts}
        for name in sorted(names):
            if _safe_leaf(name) != name:
                raise MobileAsrEvidencePacketError()
            path = root / name
            metadata = path.lstat()
            artifact = by_name[name]
            identity = (metadata.st_dev, metadata.st_ino)
            if (
                path.parent != root
                or path.resolve(strict=True) != path
                or not stat.S_ISREG(metadata.st_mode)
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or metadata.st_nlink != 1
                or metadata.st_size != len(artifact.payload)
                or identity in identities
                or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
            ):
                raise MobileAsrEvidencePacketError()
            identities.add(identity)
            rows.append((name, _metadata_snapshot(metadata)))
        return tuple(rows)
    except MobileAsrEvidencePacketError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError):
        raise MobileAsrEvidencePacketError() from None


def _locked_file(root: Path, name: str, digest: str, maximum: int) -> _Artifact:
    try:
        snapshot = read_regular_bounded(root / name, maximum_bytes=maximum)
        if (
            snapshot.path.parent != root
            or snapshot.path.name != name
            or hashlib.sha256(snapshot.data).hexdigest() != digest
        ):
            raise MobileAsrEvidencePacketError()
        return _Artifact(name=name, payload=snapshot.data, sha256=digest)
    except MobileAsrEvidencePacketError:
        raise
    except (AttributeError, BoundedReadError, OSError, RuntimeError, ValueError):
        raise MobileAsrEvidencePacketError() from None


def _corpus_artifacts(
    root: Path,
    spec: ComponentSpec,
    corpus: LoadedCorpus,
) -> tuple[_Artifact, ...]:
    manifest = _locked_file(
        root,
        spec.manifest_file,
        spec.manifest_sha256,
        _MAX_MANIFEST_BYTES,
    )
    receipt = _locked_file(
        root,
        spec.receipt_file,
        spec.receipt_sha256,
        _MAX_RECEIPT_BYTES,
    )
    pcm: list[_Artifact] = []
    for case in corpus.cases:
        if (
            case.source_path.parent != root
            or case.source_path.name in {spec.manifest_file, spec.receipt_file}
            or len(case.audio_bytes) != case.samples * 4
        ):
            raise MobileAsrEvidencePacketError()
        pcm.append(
            _Artifact(
                name=_safe_leaf(case.source_path.name),
                payload=case.audio_bytes,
                sha256=_sha256(case.sha256),
            )
        )
    if (
        corpus.digest != spec.manifest_sha256
        or len(corpus.cases) != spec.logical_cases
        or len(pcm) != spec.pcm_inputs
        or corpus.audio_bytes != spec.pcm_bytes
        or sum(len(item.payload) for item in pcm) != spec.pcm_bytes
        or any(hashlib.sha256(item.payload).hexdigest() != item.sha256 for item in pcm)
    ):
        raise MobileAsrEvidencePacketError()
    return (manifest, receipt, *pcm)


def _validate_generic(root: Path, spec: ComponentSpec) -> tuple[_Artifact, ...]:
    try:
        corpus = load_corpus(root / spec.manifest_file)
        expected_schema = 4 if spec.component_id == "command-noise" else 2
        if corpus.schema_version != expected_schema:
            raise MobileAsrEvidencePacketError()
        if spec.component_id == "command-noise":
            from tools import production_final_stt_eval as command_eval

            command_eval._validate_exact_corpus(corpus)
            command_eval._verify_private_corpus_snapshot(corpus)
        else:
            verify_corpus_snapshot(corpus)
        return _corpus_artifacts(root, spec, corpus)
    except MobileAsrEvidencePacketError:
        raise
    except Exception:
        raise MobileAsrEvidencePacketError() from None


def _load_locked_primock_source_rows() -> list[dict[str, object]]:
    try:
        snapshot = read_regular_bounded(
            _PRIMOCK_SOURCE_LOCK,
            maximum_bytes=_MAX_LOCK_BYTES,
            expected_bytes=_PRIMOCK_SOURCE_LOCK_BYTES,
        )
        if (
            snapshot.path != _PRIMOCK_SOURCE_LOCK
            or hashlib.sha256(snapshot.data).hexdigest() != _PRIMOCK_SOURCE_LOCK_SHA256
        ):
            raise MobileAsrEvidencePacketError()
        value = _strict_json(snapshot.data)
        expected_fields = {
            "schema_version",
            "kind",
            "fixture_id",
            "license_id",
            "repository",
            "revision",
            "sample_format",
            "selection",
            "output_artifacts",
            "source",
            "privacy",
            "evidence_scope",
            "recipe_digest_rule",
            "recipe_sha256",
        }
        if not isinstance(value, dict) or set(value) != expected_fields:
            raise MobileAsrEvidencePacketError()
        projection = {
            name: value[name]
            for name in (
                "fixture_id",
                "license_id",
                "output_artifacts",
                "recipe_sha256",
                "selection",
                "source",
            )
        }
        recipe_body = dict(value)
        recipe_body.pop("recipe_sha256")
        source = value.get("source")
        if (
            value.get("schema_version") != 1
            or value.get("kind") != "primock57-conversation-fixture-lock-v1"
            or value.get("fixture_id") != _PRIMOCK_FIXTURE_ID
            or value.get("license_id") != _PRIMOCK_LICENSE_ID
            or value.get("recipe_sha256") != _PRIMOCK_LOCK_RECIPE_SHA256
            or value.get("recipe_digest_rule")
            != "sha256-canonical-json-without-recipe_sha256-v1"
            or _canonical_sha256(recipe_body) != _PRIMOCK_LOCK_RECIPE_SHA256
            or _canonical_sha256(projection) != _PRIMOCK_SOURCE_LOCK_PROJECTION_SHA256
            or not isinstance(source, dict)
            or not isinstance(source.get("files"), list)
            or len(source["files"]) != len(_PRIMOCK_SOURCE_PATHS)
        ):
            raise MobileAsrEvidencePacketError()
        rows: list[dict[str, object]] = []
        for raw, path, role in zip(
            source["files"],
            _PRIMOCK_SOURCE_PATHS,
            _PRIMOCK_SOURCE_ROLES,
            strict=True,
        ):
            if (
                not isinstance(raw, dict)
                or set(raw) != {"git_blob", "path", "role", "sha256", "size_bytes"}
                or raw.get("path") != path
                or raw.get("role") != role
                or type(raw.get("size_bytes")) is not int
                or int(raw["size_bytes"]) <= 0
            ):
                raise MobileAsrEvidencePacketError()
            rows.append(
                {
                    "role": role,
                    "sha256": _sha256(raw.get("sha256")),
                    "size_bytes": int(raw["size_bytes"]),
                }
            )
        if _canonical_sha256(rows) != _PRIMOCK_SOURCE_ROWS_SHA256:
            raise MobileAsrEvidencePacketError()
        return rows
    except MobileAsrEvidencePacketError:
        raise
    except (BoundedReadError, OSError, RuntimeError, TypeError, ValueError):
        raise MobileAsrEvidencePacketError() from None


def _primock_source_rows(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list) or len(value) != len(_PRIMOCK_SOURCE_ROLES):
        raise MobileAsrEvidencePacketError()
    rows: list[dict[str, object]] = []
    for index, (row, role) in enumerate(zip(value, _PRIMOCK_SOURCE_ROLES, strict=True)):
        maximum = 16 * 1024 * 1024 if index < 2 else 64 * 1024
        if (
            not isinstance(row, dict)
            or set(row) != {"role", "sha256", "size_bytes"}
            or row.get("role") != role
            or type(row.get("size_bytes")) is not int
            or not 0 < int(row["size_bytes"]) <= maximum
        ):
            raise MobileAsrEvidencePacketError()
        rows.append(
            {
                "role": role,
                "sha256": _sha256(row.get("sha256")),
                "size_bytes": int(row["size_bytes"]),
            }
        )
    if _canonical_sha256(rows) != _PRIMOCK_SOURCE_ROWS_SHA256:
        raise MobileAsrEvidencePacketError()
    if rows != _load_locked_primock_source_rows():
        raise MobileAsrEvidencePacketError()
    return rows


def _primock_preparer_rows(value: object) -> tuple[dict[str, object], ...]:
    if not isinstance(value, list) or len(value) != len(_PRIMOCK_PREPARER_ROWS):
        raise MobileAsrEvidencePacketError()
    rows: list[dict[str, object]] = []
    for row, expected in zip(value, _PRIMOCK_PREPARER_ROWS, strict=True):
        path, size_bytes, sha256 = expected
        if (
            not isinstance(row, dict)
            or set(row) != {"path", "sha256", "size_bytes"}
            or row.get("path") != path
            or row.get("size_bytes") != size_bytes
            or row.get("sha256") != sha256
        ):
            raise MobileAsrEvidencePacketError()
        rows.append(
            {
                "path": path,
                "sha256": sha256,
                "size_bytes": size_bytes,
            }
        )
    if _canonical_sha256(rows) != _PRIMOCK_PREPARER_ROWS_SHA256:
        raise MobileAsrEvidencePacketError()
    return tuple(rows)


def _primock_source_contract_sha256(
    *,
    source_rows: list[dict[str, object]],
    preparer_rows: tuple[dict[str, object], ...],
    selection_sha256: str,
) -> str:
    return _canonical_sha256(
        {
            "accepted_license": _PRIMOCK_LICENSE_ID,
            "fixture_id": _PRIMOCK_FIXTURE_ID,
            "lock_recipe_sha256": _PRIMOCK_LOCK_RECIPE_SHA256,
            "preparer_files": list(preparer_rows),
            "production_evidence": True,
            "selection_sha256": selection_sha256,
            "source_files": source_rows,
        }
    )


def _parse_locked_primock_receipt(
    raw: bytes,
    *,
    overlap: bool,
) -> dict[str, object]:
    value = _strict_json(raw)
    common_fields = {
        "accepted_license",
        "fixture_id",
        "kind",
        "lock_recipe_sha256",
        "preparer_files",
        "privacy",
        "production_evidence",
        "schema_version",
        "selection_sha256",
        "source_files",
        "totals",
    }
    expected_fields = (
        common_fields | {"manifest", "source_contract_sha256"}
        if overlap
        else common_fields
    )
    expected_selection = (
        _PRIMOCK_OVERLAP_SELECTION_SHA256
        if overlap
        else _PRIMOCK_ISOLATED_SELECTION_SHA256
    )
    expected_kind = (
        _PRIMOCK_OVERLAP_RECEIPT_KIND if overlap else _PRIMOCK_ISOLATED_RECEIPT_KIND
    )
    expected_totals = {"cases": 3, "pcm_files": 9} if overlap else {"cases": 3}
    if (
        not isinstance(value, dict)
        or set(value) != expected_fields
        or _canonical_json(value, newline=True) != raw
        or value.get("schema_version") != 1
        or value.get("kind") != expected_kind
        or value.get("accepted_license") != _PRIMOCK_LICENSE_ID
        or value.get("fixture_id") != _PRIMOCK_FIXTURE_ID
        or value.get("lock_recipe_sha256") != _PRIMOCK_LOCK_RECIPE_SHA256
        or value.get("production_evidence") is not True
        or value.get("privacy") != _PRIMOCK_RECEIPT_PRIVACY
        or value.get("selection_sha256") != expected_selection
        or value.get("totals") != expected_totals
    ):
        raise MobileAsrEvidencePacketError()
    source_rows = _primock_source_rows(value.get("source_files"))
    preparer_rows = _primock_preparer_rows(value.get("preparer_files"))
    source_contract = _primock_source_contract_sha256(
        source_rows=source_rows,
        preparer_rows=preparer_rows,
        selection_sha256=expected_selection,
    )
    expected_source_contract = (
        _PRIMOCK_OVERLAP_SOURCE_CONTRACT_SHA256
        if overlap
        else _PRIMOCK_ISOLATED_SOURCE_CONTRACT_SHA256
    )
    if source_contract != expected_source_contract:
        raise MobileAsrEvidencePacketError()
    if overlap:
        manifest = value.get("manifest")
        if (
            value.get("source_contract_sha256") != expected_source_contract
            or not isinstance(manifest, dict)
            or set(manifest) != {"bytes", "file", "sha256"}
            or manifest.get("file") != _PRIMOCK_OVERLAP_MANIFEST
            or type(manifest.get("bytes")) is not int
            or int(manifest.get("bytes", 0)) <= 0
        ):
            raise MobileAsrEvidencePacketError()
        _sha256(manifest.get("sha256"))
    value["source_files"] = source_rows
    value["preparer_files"] = list(preparer_rows)
    return value


def _primock_int(value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise MobileAsrEvidencePacketError()
    return value


def _primock_overlap_artifact(value: object) -> dict[str, object]:
    expected = {
        "activity_end_sample",
        "activity_start_sample",
        "bytes",
        "file",
        "interval_index",
        "reference",
        "reference_sha256",
        "sha256",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise MobileAsrEvidencePacketError()
    _safe_leaf(value.get("file"))
    _sha256(value.get("sha256"))
    _primock_int(value.get("bytes"), minimum=1)
    _primock_int(value.get("activity_start_sample"))
    _primock_int(value.get("activity_end_sample"), minimum=1)
    _primock_int(value.get("interval_index"), minimum=1)
    reference = value.get("reference")
    if (
        not isinstance(reference, str)
        or not reference
        or len(reference) > 4096
        or hashlib.sha256(reference.encode("utf-8")).hexdigest()
        != _sha256(value.get("reference_sha256"))
    ):
        raise MobileAsrEvidencePacketError()
    return value


def _require_committed_component_spec(
    spec: ComponentSpec,
    *,
    component_id: str,
) -> None:
    """Keep historical compatibility subordinate to the exact v2 lock."""

    selected = load_packet_lock()
    matches = tuple(
        item for item in selected.components if item.component_id == component_id
    )
    if type(spec) is not ComponentSpec or matches != (spec,):
        raise MobileAsrEvidencePacketError()


def _validate_primock_isolated(
    root: Path,
    spec: ComponentSpec,
) -> tuple[_Artifact, ...]:
    try:
        _require_committed_component_spec(spec, component_id="primock-isolated")
        receipt = _locked_file(
            root,
            spec.receipt_file,
            spec.receipt_sha256,
            _MAX_RECEIPT_BYTES,
        )
        manifest = _locked_file(
            root,
            spec.manifest_file,
            spec.manifest_sha256,
            _MAX_MANIFEST_BYTES,
        )
        receipt_value = _parse_locked_primock_receipt(
            receipt.payload,
            overlap=False,
        )
        corpus = load_corpus(root / spec.manifest_file)
        verify_corpus_snapshot(corpus)
        source_rows = receipt_value["source_files"]
        if (
            manifest.sha256 != corpus.digest
            or corpus.schema_version != 2
            or len(corpus.cases) != spec.logical_cases
            or corpus.purpose != _PRIMOCK_ISOLATED_PURPOSE
            or corpus.provenance is None
            or corpus.provenance.kind != "public-voice-v1"
            or corpus.provenance.suite != _PRIMOCK_FIXTURE_ID
            or corpus.provenance.manifest_sha256 != _PRIMOCK_LOCK_RECIPE_SHA256
            or corpus.provenance.metadata_sha256 != _canonical_sha256(source_rows[2:4])
            or corpus.provenance.source_set_sha256 != receipt.sha256
        ):
            raise MobileAsrEvidencePacketError()
        for index, case in enumerate(corpus.cases):
            if (
                case.case_id != f"primock57-isolated-{index:02d}"
                or case.source_path.name != f"{case.case_id}.f32le"
                or case.assertion != "transcript"
                or case.commands
                or case.forbidden_commands
                or len(case.tags) != 3
                or case.tags[:2] != ("primock57", "isolated")
                or case.tags[2] not in {"role-a", "role-b"}
            ):
                raise MobileAsrEvidencePacketError()
        return _corpus_artifacts(root, spec, corpus)
    except MobileAsrEvidencePacketError:
        raise
    except Exception:
        raise MobileAsrEvidencePacketError() from None


def _load_declared_overlap_pcm(
    root: Path,
    cases: Sequence[object],
    *,
    expected_inputs: int,
    expected_bytes: int,
    sidecars: frozenset[str],
) -> tuple[_Artifact, ...]:
    """Read only the exact leaf map already admitted by an outer authority."""

    declared: list[tuple[str, str, int]] = []
    try:
        for raw_case in cases:
            if not isinstance(raw_case, dict):
                raise MobileAsrEvidencePacketError()
            for field_name in ("role_a", "role_b", "mix"):
                row = raw_case.get(field_name)
                if not isinstance(row, dict):
                    raise MobileAsrEvidencePacketError()
                name = _safe_leaf(row.get("file"))
                digest = _sha256(row.get("sha256"))
                size_bytes = row.get("bytes")
                if (
                    isinstance(size_bytes, bool)
                    or not isinstance(size_bytes, int)
                    or size_bytes <= 0
                    or size_bytes > MAX_PCM_BYTES
                    or size_bytes % 4
                ):
                    raise MobileAsrEvidencePacketError()
                declared.append((name, digest, size_bytes))
        names = [name for name, _digest, _size in declared]
        if (
            len(declared) != expected_inputs
            or len(set(names)) != len(names)
            or set(os.listdir(root)) != {*sidecars, *names}
        ):
            raise MobileAsrEvidencePacketError()
        pcm: list[_Artifact] = []
        for name, digest, size_bytes in declared:
            artifact = _locked_file(root, name, digest, MAX_PCM_BYTES)
            if len(artifact.payload) != size_bytes:
                raise MobileAsrEvidencePacketError()
            pcm.append(artifact)
        if sum(len(item.payload) for item in pcm) != expected_bytes:
            raise MobileAsrEvidencePacketError()
        return tuple(pcm)
    except MobileAsrEvidencePacketError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError):
        raise MobileAsrEvidencePacketError() from None


def _validate_primock_overlap(
    root: Path,
    spec: ComponentSpec,
) -> tuple[_Artifact, ...]:
    try:
        _require_committed_component_spec(spec, component_id="primock-overlap")
        import numpy as np

        manifest = _locked_file(
            root,
            spec.manifest_file,
            spec.manifest_sha256,
            _MAX_MANIFEST_BYTES,
        )
        receipt = _locked_file(
            root,
            spec.receipt_file,
            spec.receipt_sha256,
            _MAX_RECEIPT_BYTES,
        )
        receipt_value = _parse_locked_primock_receipt(
            receipt.payload,
            overlap=True,
        )
        receipt_manifest = receipt_value["manifest"]
        if (
            not isinstance(receipt_manifest, dict)
            or receipt_manifest["file"] != spec.manifest_file
            or receipt_manifest["bytes"] != len(manifest.payload)
            or receipt_manifest["sha256"] != manifest.sha256
        ):
            raise MobileAsrEvidencePacketError()
        manifest_value = _strict_json(manifest.payload)
        if not isinstance(manifest_value, dict) or set(manifest_value) != {
            "cases",
            "evidence_scope",
            "fixture_id",
            "kind",
            "lock_recipe_sha256",
            "production_evidence",
            "sample_format",
            "schema_version",
            "source_contract_sha256",
        }:
            raise MobileAsrEvidencePacketError()
        cases = manifest_value.get("cases")
        if (
            _canonical_json(manifest_value, newline=True) != manifest.payload
            or manifest_value.get("schema_version") != 1
            or manifest_value.get("kind") != _PRIMOCK_OVERLAP_KIND
            or manifest_value.get("fixture_id") != _PRIMOCK_FIXTURE_ID
            or manifest_value.get("production_evidence") is not True
            or manifest_value.get("source_contract_sha256")
            != receipt_value["source_contract_sha256"]
            or manifest_value.get("lock_recipe_sha256") != _PRIMOCK_LOCK_RECIPE_SHA256
            or manifest_value.get("sample_format")
            != {
                "channels": 1,
                "encoding": "f32le",
                "sample_conversion": "pcm-s16le-to-f32le-divide-32768-v1",
                "sample_rate_hz": 16_000,
            }
            or manifest_value.get("evidence_scope")
            != {
                "diagnostic_only": True,
                "natural_conversation_alignment": True,
                "ordinary_wer": False,
                "original_device_mix": False,
                "overlap_metric": "not-implemented",
                "promotion_authority": False,
                "qualification_authority": False,
            }
            or not isinstance(cases, list)
            or len(cases) != spec.logical_cases
        ):
            raise MobileAsrEvidencePacketError()
        pcm = _load_declared_overlap_pcm(
            root,
            cases,
            expected_inputs=spec.pcm_inputs,
            expected_bytes=spec.pcm_bytes,
            sidecars=frozenset({spec.manifest_file, spec.receipt_file}),
        )
        by_name = {item.name: item for item in pcm}
        observed_artifact_pins: dict[str, tuple[int, str]] = {}
        selection_rows: list[dict[str, object]] = []
        previous_source_end = -1
        for index, case in enumerate(cases):
            if not isinstance(case, dict) or set(case) != {
                "case_id",
                "envelope",
                "mix",
                "overlap",
                "role_a",
                "role_b",
            }:
                raise MobileAsrEvidencePacketError()
            case_id = f"primock57-overlap-{index:02d}"
            envelope = case.get("envelope")
            overlap = case.get("overlap")
            mix = case.get("mix")
            if (
                case.get("case_id") != case_id
                or not isinstance(envelope, dict)
                or set(envelope)
                != {"samples", "source_end_sample", "source_start_sample"}
                or not isinstance(overlap, dict)
                or set(overlap)
                != {"relative_end_sample", "relative_start_sample", "samples"}
                or not isinstance(mix, dict)
                or set(mix) != {"arithmetic", "bytes", "file", "sha256"}
            ):
                raise MobileAsrEvidencePacketError()
            samples = _primock_int(envelope.get("samples"), minimum=1)
            source_start = _primock_int(envelope.get("source_start_sample"))
            source_end = _primock_int(envelope.get("source_end_sample"), minimum=1)
            overlap_start = _primock_int(overlap.get("relative_start_sample"))
            overlap_end = _primock_int(overlap.get("relative_end_sample"), minimum=1)
            overlap_samples = _primock_int(overlap.get("samples"), minimum=1)
            mix_name = _safe_leaf(mix.get("file"))
            mix_sha256 = _sha256(mix.get("sha256"))
            if (
                source_end - source_start != samples
                or source_start < previous_source_end
                or overlap_end - overlap_start != overlap_samples
                or not 0 <= overlap_start < overlap_end <= samples
                or mix.get("arithmetic") != "float32-add-then-multiply-float32-0.5-v1"
                or _primock_int(mix.get("bytes"), minimum=1) != samples * 4
                or mix_name != f"{case_id}-mix.f32le"
            ):
                raise MobileAsrEvidencePacketError()
            previous_source_end = source_end
            role_a = _primock_overlap_artifact(case.get("role_a"))
            role_b = _primock_overlap_artifact(case.get("role_b"))
            arrays: dict[str, np.ndarray] = {}
            for key, row, suffix in (
                ("role_a", role_a, "role-a"),
                ("role_b", role_b, "role-b"),
            ):
                name = _safe_leaf(row["file"])
                artifact = by_name.get(name)
                start = _primock_int(row["activity_start_sample"])
                end = _primock_int(row["activity_end_sample"], minimum=1)
                if (
                    name != f"{case_id}-{suffix}.f32le"
                    or artifact is None
                    or len(artifact.payload) != samples * 4
                    or row["bytes"] != samples * 4
                    or artifact.sha256 != row["sha256"]
                    or not 0 <= start < end <= samples
                ):
                    raise MobileAsrEvidencePacketError()
                array = np.frombuffer(artifact.payload, dtype="<f4")
                if (
                    array.size != samples
                    or not np.isfinite(array).all()
                    or np.any(array[:start] != 0)
                    or np.any(array[end:] != 0)
                ):
                    raise MobileAsrEvidencePacketError()
                arrays[key] = array
                observed_artifact_pins[name] = (samples, artifact.sha256)
            if (
                max(
                    _primock_int(role_a["activity_start_sample"]),
                    _primock_int(role_b["activity_start_sample"]),
                )
                != overlap_start
                or min(
                    _primock_int(role_a["activity_end_sample"], minimum=1),
                    _primock_int(role_b["activity_end_sample"], minimum=1),
                )
                != overlap_end
            ):
                raise MobileAsrEvidencePacketError()
            mix_artifact = by_name.get(mix_name)
            recomputed = np.multiply(
                np.add(arrays["role_a"], arrays["role_b"], dtype=np.float32),
                np.float32(0.5),
                dtype=np.float32,
            )
            if (
                mix_artifact is None
                or len(mix_artifact.payload) != samples * 4
                or mix_artifact.sha256 != mix_sha256
                or mix_artifact.payload != np.asarray(recomputed, dtype="<f4").tobytes()
            ):
                raise MobileAsrEvidencePacketError()
            observed_artifact_pins[mix_name] = (samples, mix_artifact.sha256)
            role_a_start = source_start + _primock_int(role_a["activity_start_sample"])
            role_a_end = source_start + _primock_int(
                role_a["activity_end_sample"], minimum=1
            )
            role_b_start = source_start + _primock_int(role_b["activity_start_sample"])
            role_b_end = source_start + _primock_int(
                role_b["activity_end_sample"], minimum=1
            )
            selection_rows.append(
                {
                    "case_id": case_id,
                    "overlap_end_sample": source_start + overlap_end,
                    "overlap_relative_end_sample": overlap_end,
                    "overlap_relative_start_sample": overlap_start,
                    "overlap_samples": overlap_samples,
                    "overlap_start_sample": source_start + overlap_start,
                    "role_a": {
                        "end_sample": role_a_end,
                        "interval_index": _primock_int(
                            role_a["interval_index"], minimum=1
                        ),
                        "reference_sha256": _sha256(role_a["reference_sha256"]),
                        "role": "role-a",
                        "start_sample": role_a_start,
                    },
                    "role_b": {
                        "end_sample": role_b_end,
                        "interval_index": _primock_int(
                            role_b["interval_index"], minimum=1
                        ),
                        "reference_sha256": _sha256(role_b["reference_sha256"]),
                        "role": "role-b",
                        "start_sample": role_b_start,
                    },
                    "source_end_sample": source_end,
                    "source_start_sample": source_start,
                }
            )
        selection_sha256 = hashlib.sha256(_canonical_json(selection_rows)).hexdigest()
        if (
            selection_sha256 != receipt_value["selection_sha256"]
            or set(by_name) != set(observed_artifact_pins)
            or len(pcm) != spec.pcm_inputs
        ):
            raise MobileAsrEvidencePacketError()
        return (manifest, receipt, *pcm)
    except MobileAsrEvidencePacketError:
        raise
    except Exception:
        raise MobileAsrEvidencePacketError() from None


def _validate_component(
    path: Path | str,
    spec: ComponentSpec,
) -> _ValidatedComponent:
    root = _absolute_private_root(path)
    if spec.loader in {
        "production-final-command-noise-schema-v4",
        "streaming-stt-corpus-schema-v2",
    }:
        artifacts = _validate_generic(root, spec)
    elif spec.loader == "primock57-isolated-bundle-v1":
        artifacts = _validate_primock_isolated(root, spec)
    elif spec.loader == "primock57-overlap-bundle-v1":
        artifacts = _validate_primock_overlap(root, spec)
    else:
        raise MobileAsrEvidencePacketError()
    return _ValidatedComponent(
        spec=spec,
        root=root,
        artifacts=artifacts,
        inventory=_inventory(root, artifacts),
    )


def _component_fingerprint(
    component: _ValidatedComponent,
) -> tuple[tuple[str, int, str], ...]:
    return tuple(
        sorted(
            (item.name, len(item.payload), item.sha256) for item in component.artifacts
        )
    )


def _close_validate(component: _ValidatedComponent) -> None:
    current = _validate_component(component.root, component.spec)
    if current.inventory != component.inventory or _component_fingerprint(
        current
    ) != _component_fingerprint(component):
        raise MobileAsrEvidencePacketError()


def preflight_packet(
    sources: Mapping[str, Path | str],
) -> dict[str, object]:
    """Fully validate exact production inputs and return aggregate facts only."""

    selected = load_packet_lock()
    components = _validate_all(sources, selected)
    for component in components:
        _close_validate(component)
    return {
        "components": 5,
        "kind": "mobile-asr-evidence-packet-preflight-v2",
        "logical_cases": 123,
        "ok": True,
        "packet_lock_sha256": selected.raw_sha256,
        "pcm_bytes": 39_299_500,
        "pcm_inputs": 129,
        "production_inputs": True,
        "samples": 9_824_875,
    }


def _validate_all(
    sources: Mapping[str, Path | str],
    lock: PacketLock,
) -> tuple[_ValidatedComponent, ...]:
    try:
        if type(lock) is not PacketLock or set(sources) != set(_EXPECTED_ORDER):
            raise MobileAsrEvidencePacketError()
        components = tuple(
            _validate_component(sources[spec.component_id], spec)
            for spec in lock.components
        )
        roots = [item.root for item in components]
        identities = [(root.lstat().st_dev, root.lstat().st_ino) for root in roots]
        if len(set(identities)) != len(identities):
            raise MobileAsrEvidencePacketError()
        for index, left in enumerate(roots):
            for right in roots[index + 1 :]:
                if left.is_relative_to(right) or right.is_relative_to(left):
                    raise MobileAsrEvidencePacketError()
        return components
    except MobileAsrEvidencePacketError:
        raise
    except Exception:
        raise MobileAsrEvidencePacketError() from None


def _packet_index(lock: PacketLock) -> dict[str, object]:
    return {
        "components": [
            {
                "directory": spec.directory,
                "id": spec.component_id,
                "licenses": list(spec.licenses),
                "logical_cases": spec.logical_cases,
                "manifest_sha256": spec.manifest_sha256,
                "metric_domain": spec.metric_domain,
                "pcm_bytes": spec.pcm_bytes,
                "pcm_inputs": spec.pcm_inputs,
                "receipt_sha256": spec.receipt_sha256,
                "schema": spec.schema,
            }
            for spec in lock.components
        ],
        "kind": "mobile-asr-evidence-packet-index-v2",
        "metric_aggregation": _EXPECTED_METRIC_AGGREGATION,
        "packet_id": PACKET_ID,
        "packet_lock_sha256": lock.raw_sha256,
        "schema_version": SCHEMA_VERSION,
        "totals": _EXPECTED_TOTALS,
    }


def _artifact_tree(component: _ValidatedComponent) -> dict[str, object]:
    rows = [
        {"name": name, "sha256": digest, "size_bytes": size}
        for name, size, digest in _component_fingerprint(component)
    ]
    return {
        "files": len(rows),
        "id": component.spec.component_id,
        "payload_bytes": sum(int(row["size_bytes"]) for row in rows),
        "tree_sha256": _canonical_sha256(rows),
    }


def _packet_receipt(
    lock: PacketLock,
    index_sha256: str,
    components: tuple[_ValidatedComponent, ...],
) -> dict[str, object]:
    return {
        "complete": True,
        "components": [_artifact_tree(item) for item in components],
        "kind": "mobile-asr-evidence-packet-receipt-v2",
        "packet_id": PACKET_ID,
        "packet_index_sha256": index_sha256,
        "packet_lock_sha256": lock.raw_sha256,
        "privacy": {
            "local_paths": False,
            "raw_errors": False,
            "transcripts": False,
        },
        "runtime_receipt_policy": _EXPECTED_RUNTIME_RECEIPT_POLICY,
        "schema_version": SCHEMA_VERSION,
        "totals": _EXPECTED_TOTALS,
    }


def _validated_output(path: Path | str, sources: Sequence[Path]) -> Path:
    try:
        from tools import prepare_demand_noise_streaming_stt_corpus as demand

        candidate = demand._validated_output_path(path)
        for root in sources:
            if candidate.is_relative_to(root) or root.is_relative_to(candidate):
                raise MobileAsrEvidencePacketError()
        return candidate
    except MobileAsrEvidencePacketError:
        raise
    except Exception:
        raise MobileAsrEvidencePacketError() from None


def _verify_output_layout(
    root: Path,
    components: tuple[_ValidatedComponent, ...],
    *,
    terminal: bool,
) -> None:
    try:
        root_names = {"components", "packet-index.json"}
        if terminal:
            root_names.add("packet-receipt.json")
        if set(os.listdir(root)) != root_names:
            raise MobileAsrEvidencePacketError()
        directories = [root, root / "components"]
        if set(os.listdir(root / "components")) != {
            item.spec.directory for item in components
        }:
            raise MobileAsrEvidencePacketError()
        file_identities: set[tuple[int, int]] = set()
        for component in components:
            directory = root / "components" / component.spec.directory
            directories.append(directory)
            if set(os.listdir(directory)) != {
                item.name for item in component.artifacts
            }:
                raise MobileAsrEvidencePacketError()
            for artifact in component.artifacts:
                path = directory / artifact.name
                metadata = path.lstat()
                identity = (metadata.st_dev, metadata.st_ino)
                if (
                    path.resolve(strict=True) != path
                    or not stat.S_ISREG(metadata.st_mode)
                    or stat.S_IMODE(metadata.st_mode) != 0o600
                    or metadata.st_nlink != 1
                    or metadata.st_size != len(artifact.payload)
                    or identity in file_identities
                    or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
                ):
                    raise MobileAsrEvidencePacketError()
                file_identities.add(identity)
        for name in ("packet-index.json", "packet-receipt.json"):
            path = root / name
            if not path.exists():
                if name == "packet-receipt.json" and not terminal:
                    continue
                raise MobileAsrEvidencePacketError()
            metadata = path.lstat()
            identity = (metadata.st_dev, metadata.st_ino)
            if (
                path.resolve(strict=True) != path
                or not stat.S_ISREG(metadata.st_mode)
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or metadata.st_nlink != 1
                or identity in file_identities
                or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
            ):
                raise MobileAsrEvidencePacketError()
            file_identities.add(identity)
        for directory in directories:
            metadata = directory.lstat()
            if (
                directory.resolve(strict=True) != directory
                or not stat.S_ISDIR(metadata.st_mode)
                or stat.S_IMODE(metadata.st_mode) != 0o700
                or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
            ):
                raise MobileAsrEvidencePacketError()
    except MobileAsrEvidencePacketError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError):
        raise MobileAsrEvidencePacketError() from None


def _stage_terminal_receipt(
    directory_fd: int,
    name: str,
    raw: bytes,
) -> tuple[int, _StagedTerminalReceipt]:
    descriptor = -1
    prepared = False
    try:
        if _safe_leaf(name) != name or not raw or len(raw) > _MAX_PACKET_METADATA_BYTES:
            raise MobileAsrEvidencePacketError()
        try:
            os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise MobileAsrEvidencePacketError()
        temporary = getattr(os, "O_TMPFILE", 0)
        if not temporary:
            raise MobileAsrEvidencePacketError()
        descriptor = os.open(
            ".",
            os.O_RDWR | temporary | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=directory_fd,
        )
        os.fchmod(descriptor, 0o600)
        view = memoryview(raw)
        while view:
            count = os.write(descriptor, view)
            if count <= 0:
                raise MobileAsrEvidencePacketError()
            view = view[count:]
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_nlink != 0
            or metadata.st_size != len(raw)
            or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
        ):
            raise MobileAsrEvidencePacketError()
        staged = _StagedTerminalReceipt(
            name=name,
            size_bytes=len(raw),
            sha256=hashlib.sha256(raw).hexdigest(),
            snapshot=_metadata_snapshot(metadata),
        )
        _verify_staged_terminal_receipt(descriptor, staged, raw)
        prepared = True
        return descriptor, staged
    except MobileAsrEvidencePacketError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError):
        raise MobileAsrEvidencePacketError() from None
    finally:
        if descriptor >= 0 and not prepared:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _verify_staged_terminal_receipt(
    descriptor: int,
    staged: _StagedTerminalReceipt,
    expected: bytes,
) -> None:
    try:
        before = os.fstat(descriptor)
        if (
            _metadata_snapshot(before) != staged.snapshot
            or before.st_nlink != 0
            or staged.size_bytes != len(expected)
            or staged.sha256 != hashlib.sha256(expected).hexdigest()
        ):
            raise MobileAsrEvidencePacketError()
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        consumed = 0
        while consumed <= staged.size_bytes:
            chunk = os.read(
                descriptor,
                min(65_536, staged.size_bytes + 1 - consumed),
            )
            if not chunk:
                break
            chunks.append(chunk)
            consumed += len(chunk)
        after = os.fstat(descriptor)
        if (
            b"".join(chunks) != expected
            or consumed != staged.size_bytes
            or _metadata_snapshot(after) != staged.snapshot
        ):
            raise MobileAsrEvidencePacketError()
    except MobileAsrEvidencePacketError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError):
        raise MobileAsrEvidencePacketError() from None


def _terminal_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        stat.S_IMODE(metadata.st_mode),
        metadata.st_uid,
        metadata.st_size,
    )


def _commit_signal_numbers() -> set[int]:
    return {
        int(signum)
        for signum in (
            getattr(signal, "SIGHUP", None),
            getattr(signal, "SIGINT", None),
            getattr(signal, "SIGTERM", None),
        )
        if isinstance(signum, int)
    }


def _commit_terminal_receipt(
    directory_fd: int,
    descriptor: int,
    staged: _StagedTerminalReceipt,
    expected: bytes,
    state: _TerminalCommitState,
    *,
    commit_guard: Callable[[], None],
) -> None:
    if not hasattr(signal, "pthread_sigmask") or state.committed:
        raise MobileAsrEvidencePacketError()
    try:
        opened = os.fstat(descriptor)
        expected_identity = _terminal_identity(opened)
        if opened.st_nlink != 0 or _metadata_snapshot(opened) != staged.snapshot:
            raise MobileAsrEvidencePacketError()
    except MobileAsrEvidencePacketError:
        raise
    except OSError:
        raise MobileAsrEvidencePacketError() from None
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, _commit_signal_numbers())
    link_returned = False
    try:
        try:
            commit_guard()
            _verify_staged_terminal_receipt(descriptor, staged, expected)
            os.link(
                f"/proc/self/fd/{descriptor}",
                staged.name,
                dst_dir_fd=directory_fd,
                follow_symlinks=True,
            )
            link_returned = True
            published = os.stat(
                staged.name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
            if (
                _terminal_identity(published) != expected_identity
                or published.st_nlink != 1
                or not stat.S_ISREG(published.st_mode)
            ):
                raise MobileAsrEvidencePacketError()
            os.fsync(directory_fd)
        finally:
            if link_returned:
                state.committed = True
            else:
                try:
                    opened = os.fstat(descriptor)
                    published = os.stat(
                        staged.name,
                        dir_fd=directory_fd,
                        follow_symlinks=False,
                    )
                except OSError:
                    pass
                else:
                    if (
                        _terminal_identity(opened) == expected_identity
                        and _terminal_identity(published) == expected_identity
                        and opened.st_nlink == 1
                        and published.st_nlink == 1
                    ):
                        state.committed = True
    except MobileAsrEvidencePacketError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError):
        raise MobileAsrEvidencePacketError() from None
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def _packet_tree_snapshot(
    root: Path,
    components: tuple[_ValidatedComponent, ...],
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    paths = [
        root,
        root / "components",
        root / "packet-index.json",
        root / "packet-receipt.json",
    ]
    for component in components:
        directory = root / "components" / component.spec.directory
        paths.append(directory)
        paths.extend(directory / artifact.name for artifact in component.artifacts)
    try:
        rows = tuple(
            (
                "." if path == root else path.relative_to(root).as_posix(),
                _metadata_snapshot(path.lstat()),
            )
            for path in paths
        )
        if len({name for name, _snapshot in rows}) != len(rows):
            raise MobileAsrEvidencePacketError()
        return rows
    except MobileAsrEvidencePacketError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError):
        raise MobileAsrEvidencePacketError() from None


def _close_validate_packet_tree(
    root: Path,
    components: tuple[_ValidatedComponent, ...],
    index_raw: bytes,
    receipt_raw: bytes,
    *,
    component_guard: Callable[[], None],
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    _verify_output_layout(root, components, terminal=True)
    tree_snapshot = _packet_tree_snapshot(root, components)
    component_guard()
    closed_index = read_regular_bounded(
        root / "packet-index.json",
        maximum_bytes=_MAX_PACKET_METADATA_BYTES,
        expected_bytes=len(index_raw),
    )
    closed_receipt = read_regular_bounded(
        root / "packet-receipt.json",
        maximum_bytes=_MAX_PACKET_METADATA_BYTES,
        expected_bytes=len(receipt_raw),
    )
    _verify_output_layout(root, components, terminal=True)
    if (
        closed_index.data != index_raw
        or closed_receipt.data != receipt_raw
        or _packet_tree_snapshot(root, components) != tree_snapshot
    ):
        raise MobileAsrEvidencePacketError()
    return tree_snapshot


def load_mobile_asr_evidence_packet(
    packet_root: Path | str,
) -> LoadedMobileAsrEvidencePacket:
    """Strictly load the complete exact-copy packet and its terminal receipt."""

    try:
        lock = load_packet_lock()
        root = _absolute_private_root(packet_root)
        if set(os.listdir(root)) != {
            "components",
            "packet-index.json",
            "packet-receipt.json",
        }:
            raise MobileAsrEvidencePacketError()
        index_snapshot = read_regular_bounded(
            root / "packet-index.json",
            maximum_bytes=_MAX_PACKET_METADATA_BYTES,
        )
        receipt_snapshot = read_regular_bounded(
            root / "packet-receipt.json",
            maximum_bytes=_MAX_PACKET_METADATA_BYTES,
        )
        index_value = _strict_json(index_snapshot.data)
        expected_index = _packet_index(lock)
        if (
            index_value != expected_index
            or _canonical_json(index_value, newline=True) != index_snapshot.data
        ):
            raise MobileAsrEvidencePacketError()
        components = tuple(
            _validate_component(
                root / "components" / spec.directory,
                spec,
            )
            for spec in lock.components
        )
        index_sha256 = hashlib.sha256(index_snapshot.data).hexdigest()
        receipt_value = _strict_json(receipt_snapshot.data)
        expected_receipt = _packet_receipt(lock, index_sha256, components)
        if (
            receipt_value != expected_receipt
            or _canonical_json(receipt_value, newline=True) != receipt_snapshot.data
        ):
            raise MobileAsrEvidencePacketError()

        def close_components() -> None:
            for component in components:
                _close_validate(component)

        tree_snapshot = _close_validate_packet_tree(
            root,
            components,
            index_snapshot.data,
            receipt_snapshot.data,
            component_guard=close_components,
        )
        packet = LoadedMobileAsrEvidencePacket(
            schema_version=SCHEMA_VERSION,
            packet_id=PACKET_ID,
            packet_lock_sha256=lock.raw_sha256,
            packet_index_sha256=index_sha256,
            packet_receipt_sha256=hashlib.sha256(receipt_snapshot.data).hexdigest(),
            components=5,
            logical_cases=123,
            pcm_inputs=129,
            pcm_bytes=39_299_500,
            samples=9_824_875,
            root=root,
            _validated_components=components,
            _tree_snapshot=tree_snapshot,
        )
        return packet
    except MobileAsrEvidencePacketError:
        raise
    except Exception:
        raise MobileAsrEvidencePacketError() from None


def verify_mobile_asr_evidence_packet(
    packet: LoadedMobileAsrEvidencePacket,
) -> None:
    """Reopen an exact packet and reject any byte, inode, mode, or entry change."""

    if type(packet) is not LoadedMobileAsrEvidencePacket:
        raise MobileAsrEvidencePacketError()
    current = load_mobile_asr_evidence_packet(packet.root)
    if (
        current.schema_version != packet.schema_version
        or current.packet_id != packet.packet_id
        or current.packet_lock_sha256 != packet.packet_lock_sha256
        or current.packet_index_sha256 != packet.packet_index_sha256
        or current.packet_receipt_sha256 != packet.packet_receipt_sha256
        or current.components != packet.components
        or current.logical_cases != packet.logical_cases
        or current.pcm_inputs != packet.pcm_inputs
        or current.pcm_bytes != packet.pcm_bytes
        or current.samples != packet.samples
        or current.root != packet.root
        or current._tree_snapshot != packet._tree_snapshot
        or tuple(_component_fingerprint(item) for item in current._validated_components)
        != tuple(_component_fingerprint(item) for item in packet._validated_components)
        or tuple(item.inventory for item in current._validated_components)
        != tuple(item.inventory for item in packet._validated_components)
    ):
        raise MobileAsrEvidencePacketError()


def _publish_tree_lifecycle(
    *,
    candidate: Path,
    source_components: tuple[_ValidatedComponent, ...],
    index_raw: bytes,
    receipt_builder: Callable[[tuple[_ValidatedComponent, ...]], bytes],
    copy_loader: Callable[[Path, ComponentSpec], _ValidatedComponent],
    source_guard: Callable[[_ValidatedComponent], None],
    copy_guard: Callable[[_ValidatedComponent], None],
    complete_loader: Callable[[Path], object],
    complete_verifier: Callable[[object], None],
    loaded_guard: Callable[[object, str, str], None],
    result_builder: Callable[[object], dict[str, object]],
    post_link_hook: Callable[[], None] | None = None,
) -> dict[str, object]:
    state = _TerminalCommitState()
    staged_fd = -1
    loaded: object | None = None
    index_sha256 = hashlib.sha256(index_raw).hexdigest()
    staged_sha256 = ""
    try:
        with _new_private_output(candidate) as root_output:
            with ExitStack() as stack:
                component_parent = stack.enter_context(
                    _new_private_output(candidate / "components")
                )
                output_directories = {}
                written = {}
                for component in source_components:
                    output = stack.enter_context(
                        _new_private_output(
                            candidate / "components" / component.spec.directory
                        )
                    )
                    output_directories[component.spec.component_id] = output
                    written[component.spec.component_id] = tuple(
                        _write_new_private(
                            output.directory_fd,
                            artifact.name,
                            artifact.payload,
                        )
                        for artifact in component.artifacts
                    )
                index_written = _write_new_private(
                    root_output.directory_fd,
                    "packet-index.json",
                    index_raw,
                )
                copied = tuple(
                    copy_loader(
                        candidate / "components" / component.spec.directory,
                        component.spec,
                    )
                    for component in source_components
                )
                if any(
                    _component_fingerprint(source)
                    != _component_fingerprint(destination)
                    for source, destination in zip(
                        source_components,
                        copied,
                        strict=True,
                    )
                ):
                    raise MobileAsrEvidencePacketError()
                receipt_raw = receipt_builder(copied)
                staged_fd, staged = _stage_terminal_receipt(
                    root_output.directory_fd,
                    "packet-receipt.json",
                    receipt_raw,
                )
                staged_sha256 = staged.sha256

                def commit_guard() -> None:
                    for output in (
                        root_output,
                        component_parent,
                        *output_directories.values(),
                    ):
                        _verify_output_binding(output)
                        os.fsync(output.directory_fd)
                    for component in source_components:
                        output = output_directories[component.spec.component_id]
                        retained = written[component.spec.component_id]
                        for artifact, private_file in zip(
                            component.artifacts,
                            retained,
                            strict=True,
                        ):
                            _verify_written_private_metadata(
                                output.directory_fd,
                                private_file,
                            )
                            _read_exact_written_private(
                                output.directory_fd,
                                private_file,
                                artifact.payload,
                                maximum_bytes=max(
                                    MAX_PCM_BYTES,
                                    _MAX_PACKET_METADATA_BYTES,
                                ),
                            )
                    _verify_written_private_metadata(
                        root_output.directory_fd,
                        index_written,
                    )
                    _read_exact_written_private(
                        root_output.directory_fd,
                        index_written,
                        index_raw,
                        maximum_bytes=_MAX_PACKET_METADATA_BYTES,
                    )
                    if set(os.listdir(component_parent.directory_fd)) != {
                        item.spec.directory for item in source_components
                    }:
                        raise MobileAsrEvidencePacketError()
                    for component in source_components:
                        source_guard(component)
                    for component in copied:
                        copy_guard(component)
                    _verify_output_layout(candidate, copied, terminal=False)

                _commit_terminal_receipt(
                    root_output.directory_fd,
                    staged_fd,
                    staged,
                    receipt_raw,
                    state,
                    commit_guard=commit_guard,
                )
                if not state.committed:
                    raise MobileAsrEvidencePacketError()
                if post_link_hook is not None:
                    post_link_hook()
        loaded = complete_loader(candidate)
        loaded_guard(loaded, index_sha256, staged_sha256)
        complete_verifier(loaded)
        return result_builder(loaded)
    except BaseException as error:
        if state.committed:
            try:
                recovered = complete_loader(candidate)
                loaded_guard(recovered, index_sha256, staged_sha256)
                complete_verifier(recovered)
            except Exception:
                if isinstance(error, Exception):
                    raise MobileAsrEvidencePacketError() from None
                raise error
            return result_builder(recovered)
        if isinstance(error, MobileAsrEvidencePacketError):
            raise
        if isinstance(error, Exception):
            raise MobileAsrEvidencePacketError() from None
        raise
    finally:
        if staged_fd >= 0:
            try:
                os.close(staged_fd)
            except OSError:
                pass


def _publication_result(
    packet: LoadedMobileAsrEvidencePacket,
) -> dict[str, object]:
    return {
        "components": packet.components,
        "kind": "mobile-asr-evidence-packet-publication-v2",
        "logical_cases": packet.logical_cases,
        "ok": True,
        "packet_index_sha256": packet.packet_index_sha256,
        "packet_lock_sha256": packet.packet_lock_sha256,
        "packet_receipt_sha256": packet.packet_receipt_sha256,
        "pcm_bytes": packet.pcm_bytes,
        "pcm_inputs": packet.pcm_inputs,
        "production_inputs": True,
        "samples": packet.samples,
    }


def publish_packet(
    sources: Mapping[str, Path | str],
    output_root: Path | str,
) -> dict[str, object]:
    """Validate exact inputs, then atomically link the terminal receipt last."""

    selected = load_packet_lock()
    components = _validate_all(sources, selected)
    candidate = _validated_output(output_root, [item.root for item in components])
    index_raw = _canonical_json(_packet_index(selected), newline=True)
    index_sha256 = hashlib.sha256(index_raw).hexdigest()

    def receipt_builder(copied: tuple[_ValidatedComponent, ...]) -> bytes:
        return _canonical_json(
            _packet_receipt(selected, index_sha256, copied),
            newline=True,
        )

    def complete_verifier(value: object) -> None:
        if type(value) is not LoadedMobileAsrEvidencePacket:
            raise MobileAsrEvidencePacketError()
        verify_mobile_asr_evidence_packet(value)

    def loaded_guard(value: object, expected_index: str, expected_receipt: str) -> None:
        if (
            type(value) is not LoadedMobileAsrEvidencePacket
            or value.packet_index_sha256 != expected_index
            or value.packet_receipt_sha256 != expected_receipt
        ):
            raise MobileAsrEvidencePacketError()

    def result_builder(value: object) -> dict[str, object]:
        if type(value) is not LoadedMobileAsrEvidencePacket:
            raise MobileAsrEvidencePacketError()
        return _publication_result(value)

    return _publish_tree_lifecycle(
        candidate=candidate,
        source_components=components,
        index_raw=index_raw,
        receipt_builder=receipt_builder,
        copy_loader=_validate_component,
        source_guard=_close_validate,
        copy_guard=_close_validate,
        complete_loader=load_mobile_asr_evidence_packet,
        complete_verifier=complete_verifier,
        loaded_guard=loaded_guard,
        result_builder=result_builder,
    )


@dataclass(frozen=True, slots=True)
class _SyntheticTestPacketContract:
    """Unmistakably nonproduction authority for small filesystem tests only."""

    packet_id: str
    artifacts: tuple[_Artifact, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class _LoadedSyntheticTestPacket:
    packet_id: str
    authority_sha256: str
    packet_index_sha256: str
    packet_receipt_sha256: str
    production_inputs: bool
    root: Path = field(repr=False)
    component: _ValidatedComponent = field(repr=False)
    tree_snapshot: tuple[tuple[str, tuple[int, ...]], ...] = field(repr=False)


def _synthetic_test_authority(
    contract: _SyntheticTestPacketContract,
) -> tuple[ComponentSpec, str]:
    if (
        type(contract) is not _SyntheticTestPacketContract
        or not isinstance(contract.packet_id, str)
        or not contract.packet_id.startswith("synthetic-")
        or contract.packet_id == PACKET_ID
        or type(contract.artifacts) is not tuple
        or len(contract.artifacts) != 3
    ):
        raise MobileAsrEvidencePacketError()
    expected_names = {
        "synthetic-audio.f32le",
        "synthetic-component-receipt.json",
        "synthetic-manifest.json",
    }
    if (
        {item.name for item in contract.artifacts} != expected_names
        or any(type(item) is not _Artifact for item in contract.artifacts)
        or any(
            not item.payload or item.sha256 != hashlib.sha256(item.payload).hexdigest()
            for item in contract.artifacts
        )
    ):
        raise MobileAsrEvidencePacketError()
    audio = next(
        item for item in contract.artifacts if item.name == "synthetic-audio.f32le"
    )
    if len(audio.payload) % 4:
        raise MobileAsrEvidencePacketError()
    authority_sha256 = _canonical_sha256(
        {
            "artifacts": [
                {"name": name, "sha256": digest, "size_bytes": size}
                for name, size, digest in sorted(
                    (item.name, len(item.payload), item.sha256)
                    for item in contract.artifacts
                )
            ],
            "kind": "synthetic-mobile-asr-packet-test-authority-v1",
            "packet_id": contract.packet_id,
            "production_inputs": False,
        }
    )
    by_name = {item.name: item for item in contract.artifacts}
    spec = ComponentSpec(
        component_id="synthetic-component",
        directory="synthetic-component",
        loader="synthetic-test-only",
        schema="synthetic-test-only",
        manifest_file="synthetic-manifest.json",
        manifest_sha256=by_name["synthetic-manifest.json"].sha256,
        receipt_file="synthetic-component-receipt.json",
        receipt_sha256=by_name["synthetic-component-receipt.json"].sha256,
        licenses=("synthetic-test-only",),
        logical_cases=1,
        pcm_inputs=1,
        pcm_bytes=len(audio.payload),
        metric_domain="synthetic-test-only",
    )
    return spec, authority_sha256


def _synthetic_test_index(
    contract: _SyntheticTestPacketContract,
) -> dict[str, object]:
    spec, authority_sha256 = _synthetic_test_authority(contract)
    return {
        "authority_sha256": authority_sha256,
        "component": {
            "directory": spec.directory,
            "files": len(contract.artifacts),
            "pcm_bytes": spec.pcm_bytes,
        },
        "kind": "synthetic-mobile-asr-packet-index-v1",
        "packet_id": contract.packet_id,
        "production_inputs": False,
        "schema_version": 1,
    }


def _synthetic_test_receipt(
    contract: _SyntheticTestPacketContract,
    index_sha256: str,
) -> dict[str, object]:
    _spec, authority_sha256 = _synthetic_test_authority(contract)
    rows = [
        {"name": name, "sha256": digest, "size_bytes": size}
        for name, size, digest in sorted(
            (item.name, len(item.payload), item.sha256) for item in contract.artifacts
        )
    ]
    return {
        "authority_sha256": authority_sha256,
        "complete": True,
        "component_tree_sha256": _canonical_sha256(rows),
        "kind": "synthetic-mobile-asr-packet-receipt-v1",
        "packet_id": contract.packet_id,
        "packet_index_sha256": index_sha256,
        "production_inputs": False,
        "schema_version": 1,
    }


def _load_synthetic_test_packet(
    root_path: Path | str,
    contract: _SyntheticTestPacketContract,
) -> _LoadedSyntheticTestPacket:
    spec, authority_sha256 = _synthetic_test_authority(contract)
    try:
        root = _absolute_private_root(root_path)
        if set(os.listdir(root)) != {
            "components",
            "packet-index.json",
            "packet-receipt.json",
        }:
            raise MobileAsrEvidencePacketError()
        component_root = _absolute_private_root(
            root / "components" / "synthetic-component"
        )
        artifacts = tuple(
            _locked_file(
                component_root,
                item.name,
                item.sha256,
                max(MAX_PCM_BYTES, _MAX_PACKET_METADATA_BYTES),
            )
            for item in contract.artifacts
        )
        component = _ValidatedComponent(
            spec=spec,
            root=component_root,
            artifacts=artifacts,
            inventory=_inventory(component_root, artifacts),
        )
        index = read_regular_bounded(
            root / "packet-index.json",
            maximum_bytes=_MAX_PACKET_METADATA_BYTES,
        )
        receipt = read_regular_bounded(
            root / "packet-receipt.json",
            maximum_bytes=_MAX_PACKET_METADATA_BYTES,
        )
        index_sha256 = hashlib.sha256(index.data).hexdigest()
        if (
            _strict_json(index.data) != _synthetic_test_index(contract)
            or index.data
            != _canonical_json(_synthetic_test_index(contract), newline=True)
            or _strict_json(receipt.data)
            != _synthetic_test_receipt(contract, index_sha256)
            or receipt.data
            != _canonical_json(
                _synthetic_test_receipt(contract, index_sha256),
                newline=True,
            )
        ):
            raise MobileAsrEvidencePacketError()

        def close_component() -> None:
            current_artifacts = tuple(
                _locked_file(
                    component_root,
                    item.name,
                    item.sha256,
                    max(MAX_PCM_BYTES, _MAX_PACKET_METADATA_BYTES),
                )
                for item in contract.artifacts
            )
            current = _ValidatedComponent(
                spec=spec,
                root=component_root,
                artifacts=current_artifacts,
                inventory=_inventory(component_root, current_artifacts),
            )
            if (
                _component_fingerprint(component) != _component_fingerprint(current)
                or component.inventory != current.inventory
            ):
                raise MobileAsrEvidencePacketError()

        before = _close_validate_packet_tree(
            root,
            (component,),
            index.data,
            receipt.data,
            component_guard=close_component,
        )
        return _LoadedSyntheticTestPacket(
            packet_id=contract.packet_id,
            authority_sha256=authority_sha256,
            packet_index_sha256=index_sha256,
            packet_receipt_sha256=hashlib.sha256(receipt.data).hexdigest(),
            production_inputs=False,
            root=root,
            component=component,
            tree_snapshot=before,
        )
    except MobileAsrEvidencePacketError:
        raise
    except Exception:
        raise MobileAsrEvidencePacketError() from None


def _verify_synthetic_test_packet(
    loaded: _LoadedSyntheticTestPacket,
    contract: _SyntheticTestPacketContract,
) -> None:
    if type(loaded) is not _LoadedSyntheticTestPacket:
        raise MobileAsrEvidencePacketError()
    current = _load_synthetic_test_packet(loaded.root, contract)
    if (
        current.packet_id != loaded.packet_id
        or current.authority_sha256 != loaded.authority_sha256
        or current.packet_index_sha256 != loaded.packet_index_sha256
        or current.packet_receipt_sha256 != loaded.packet_receipt_sha256
        or current.production_inputs is not False
        or loaded.production_inputs is not False
        or current.root != loaded.root
        or current.tree_snapshot != loaded.tree_snapshot
        or current.component.inventory != loaded.component.inventory
        or _component_fingerprint(current.component)
        != _component_fingerprint(loaded.component)
    ):
        raise MobileAsrEvidencePacketError()


def _synthetic_test_result(
    loaded: _LoadedSyntheticTestPacket,
) -> dict[str, object]:
    return {
        "authority_sha256": loaded.authority_sha256,
        "kind": "synthetic-mobile-asr-packet-publication-v1",
        "ok": True,
        "packet_index_sha256": loaded.packet_index_sha256,
        "packet_receipt_sha256": loaded.packet_receipt_sha256,
        "production_inputs": False,
    }


def _publish_synthetic_test_packet(
    contract: _SyntheticTestPacketContract,
    output_root: Path | str,
    *,
    post_link_hook: Callable[[], None] | None = None,
) -> dict[str, object]:
    spec, _authority_sha256 = _synthetic_test_authority(contract)
    candidate = _validated_output(output_root, [])
    index_raw = _canonical_json(_synthetic_test_index(contract), newline=True)
    index_sha256 = hashlib.sha256(index_raw).hexdigest()
    source = _ValidatedComponent(
        spec=spec,
        root=Path("/synthetic-test-authority-has-no-source-root"),
        artifacts=contract.artifacts,
        inventory=(),
    )

    def receipt_builder(copied: tuple[_ValidatedComponent, ...]) -> bytes:
        if len(copied) != 1 or _component_fingerprint(
            copied[0]
        ) != _component_fingerprint(source):
            raise MobileAsrEvidencePacketError()
        return _canonical_json(
            _synthetic_test_receipt(contract, index_sha256),
            newline=True,
        )

    def copy_loader(root: Path, selected_spec: ComponentSpec) -> _ValidatedComponent:
        if selected_spec != spec:
            raise MobileAsrEvidencePacketError()
        artifacts = tuple(
            _locked_file(
                root,
                item.name,
                item.sha256,
                max(MAX_PCM_BYTES, _MAX_PACKET_METADATA_BYTES),
            )
            for item in contract.artifacts
        )
        return _ValidatedComponent(
            spec=spec,
            root=root,
            artifacts=artifacts,
            inventory=_inventory(root, artifacts),
        )

    def source_guard(component: _ValidatedComponent) -> None:
        _synthetic_test_authority(contract)
        if component is not source:
            raise MobileAsrEvidencePacketError()

    def copy_guard(component: _ValidatedComponent) -> None:
        current = copy_loader(component.root, component.spec)
        if current.inventory != component.inventory or _component_fingerprint(
            current
        ) != _component_fingerprint(component):
            raise MobileAsrEvidencePacketError()

    def complete_loader(root: Path) -> object:
        return _load_synthetic_test_packet(root, contract)

    def complete_verifier(value: object) -> None:
        if type(value) is not _LoadedSyntheticTestPacket:
            raise MobileAsrEvidencePacketError()
        _verify_synthetic_test_packet(value, contract)

    def loaded_guard(value: object, expected_index: str, expected_receipt: str) -> None:
        if (
            type(value) is not _LoadedSyntheticTestPacket
            or value.packet_index_sha256 != expected_index
            or value.packet_receipt_sha256 != expected_receipt
            or value.production_inputs is not False
        ):
            raise MobileAsrEvidencePacketError()

    def result_builder(value: object) -> dict[str, object]:
        if type(value) is not _LoadedSyntheticTestPacket:
            raise MobileAsrEvidencePacketError()
        return _synthetic_test_result(value)

    return _publish_tree_lifecycle(
        candidate=candidate,
        source_components=(source,),
        index_raw=index_raw,
        receipt_builder=receipt_builder,
        copy_loader=copy_loader,
        source_guard=source_guard,
        copy_guard=copy_guard,
        complete_loader=complete_loader,
        complete_verifier=complete_verifier,
        loaded_guard=loaded_guard,
        result_builder=result_builder,
        post_link_hook=post_link_hook,
    )


def _parse_cli(argv: Sequence[str]) -> _ParsedCommand | None:
    if list(argv) in (["--help"], ["help"]):
        return None
    if not argv or argv[0] not in {"preflight", "publish"}:
        raise MobileAsrEvidencePacketError()
    action = argv[0]
    tail = list(argv[1:])
    if len(tail) % 2:
        raise MobileAsrEvidencePacketError()
    values: dict[str, str] = {}
    for index in range(0, len(tail), 2):
        flag, value = tail[index : index + 2]
        allowed = {*_SOURCE_FLAGS, *({_OUTPUT_FLAG} if action == "publish" else set())}
        if (
            flag not in allowed
            or flag in values
            or not value
            or "\x00" in value
            or value.startswith("--")
        ):
            raise MobileAsrEvidencePacketError()
        values[flag] = value
    expected = set(_SOURCE_FLAGS)
    if action == "publish":
        expected.add(_OUTPUT_FLAG)
    if set(values) != expected:
        raise MobileAsrEvidencePacketError()
    return _ParsedCommand(
        action=action,
        sources={
            component_id: values[flag] for flag, component_id in _SOURCE_FLAGS.items()
        },
        output_root=values.get(_OUTPUT_FLAG),
    )


def _help_receipt() -> dict[str, object]:
    return {
        "actions": ["preflight", "publish"],
        "component_flags": list(_SOURCE_FLAGS),
        "kind": "mobile-asr-evidence-packet-help-v2",
        "ok": True,
        "publish_only_flag": _OUTPUT_FLAG,
    }


def _emit(value: object) -> None:
    sys.stdout.buffer.write(_canonical_json(value, newline=True))
    sys.stdout.buffer.flush()


def main(argv: Sequence[str] | None = None) -> int:
    """CLI with aggregate-only receipts and detail-free failures."""

    try:
        command = _parse_cli(tuple(sys.argv[1:] if argv is None else argv))
        if command is None:
            _emit(_help_receipt())
            return 0
        if command.action == "preflight":
            result = preflight_packet(command.sources)
        else:
            if command.output_root is None:
                raise MobileAsrEvidencePacketError()
            result = publish_packet(
                command.sources,
                command.output_root,
            )
        _emit(result)
        return 0
    except Exception:
        _emit(_SAFE_ERROR)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
