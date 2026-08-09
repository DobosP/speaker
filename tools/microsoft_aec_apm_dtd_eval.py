"""Receipt-bound aggregate Microsoft AEC APM/DTD component replay.

This evaluator is intentionally narrower than a live voice-agent test.  It
replays the exact synchronized component tracks from the private Microsoft AEC
fixture through the committed ``open_speaker`` WebRTC APM configuration, then
drives the production DTD math with a fixed energy oracle.  It opens no audio
device, VAD, recognizer, endpoint, tool, or network surface and publishes no
per-case information.
"""

from __future__ import annotations

import argparse
import base64
import csv
import ctypes
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import hashlib
import importlib
from importlib import metadata as importlib_metadata
from importlib import machinery as importlib_machinery
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import stat
import sys
from types import CodeType, FunctionType, ModuleType
from typing import Callable, Mapping, Sequence

import numpy as np

from core.config import apply_device_profile
from core.engines._aec import build_aec
from core.engines._dtd import AdaptiveDTD, BargeSustain
from core.engines.echo_coherence import EchoCoherenceDetector
from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine
from tools.audio_eval.metrics import (
    absolute_cosine_similarity,
    capped_db_ratio,
    echo_attenuation_db,
    interference_reduction_db,
    oracle_activity_mask,
    percentile_summary,
    projection_gain,
    si_sdr_db,
    signal_power,
    signal_rms,
)
from tools.prepare_microsoft_aec_fixture import (
    LoadedMicrosoftAecBundle,
    MicrosoftAecCase,
    decode_microsoft_aec_wav,
    load_microsoft_aec_bundle,
    verify_microsoft_aec_bundle,
)
from tools.streaming_stt.bounded_io import (
    BoundedReadError,
    opened_directory_nofollow,
    read_regular_bounded,
)


SCHEMA_VERSION = 1
REPORT_KIND = "microsoft-aec-apm-dtd-eval-v1"
SAMPLE_RATE_HZ = 16_000
BLOCK_SAMPLES = 1_600
CONVERGENCE_SAMPLES = 16_000
ECHO_PREFIX_SAMPLES = 160_000
MAX_TERMINAL_PADDING_SAMPLES = 1
PRODUCTION_CASES = 32
SIGNAL_ROLES = (
    "echo_signal",
    "farend_speech",
    "nearend_mic_signal",
    "nearend_speech",
)
_CLIP_LEVEL = 32767.0 / 32768.0
_MAX_CONFIG_BYTES = 4 * 1024 * 1024
_MAX_SOURCE_BYTES = 2 * 1024 * 1024
_MAX_AUDIO_BYTES = 512 * 1024
_MAX_REPORT_BYTES = 256 * 1024
_LIVEKIT_RECORD_MAX_BYTES = 2 * 1024 * 1024
_LIVEKIT_FILE_MAX_BYTES = 64 * 1024 * 1024
_LIVEKIT_TOTAL_MAX_BYTES = 128 * 1024 * 1024
_LIVEKIT_FILE_MAX_COUNT = 10_000
_LIVEKIT_NATIVE_NAMES = {
    "liblivekit_ffi.so",
    "liblivekit_ffi.dylib",
    "livekit_ffi.dll",
}
_SAFE_LABEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+_-]{0,63}\Z")
_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _REPO_ROOT / "config.json"
_REMOTE_REQUIREMENTS_PATH = _REPO_ROOT / "requirements-remote.txt"
_EAGER_LOCAL_IMPORT_FILES = (
    "always_on_agent/__init__.py",
    "always_on_agent/acoustic.py",
    "always_on_agent/capabilities.py",
    "always_on_agent/continuation.py",
    "always_on_agent/diagnostics.py",
    "always_on_agent/event_bus.py",
    "always_on_agent/events.py",
    "always_on_agent/followups.py",
    "always_on_agent/memory.py",
    "always_on_agent/models.py",
    "always_on_agent/origin.py",
    "always_on_agent/planner.py",
    "always_on_agent/recall.py",
    "always_on_agent/runtime.py",
    "always_on_agent/session_actor.py",
    "always_on_agent/speech_analyzer.py",
    "always_on_agent/supervisor.py",
    "always_on_agent/tasks.py",
    "always_on_agent/text.py",
    "core/__init__.py",
    "core/asr_text.py",
    "core/asr_verifier.py",
    "core/audio_frontend.py",
    "core/config.py",
    "core/contract.py",
    "core/engine.py",
    "core/engines/__init__.py",
    "core/engines/_acoustic_turn.py",
    "core/engines/_aec.py",
    "core/engines/_asr_segment.py",
    "core/engines/_denoiser.py",
    "core/engines/_dtd.py",
    "core/engines/_semantic_hold.py",
    "core/engines/_sherpa_models.py",
    "core/engines/_sherpa_streaming_decode.py",
    "core/engines/_sherpa_streaming_decode_owner.py",
    "core/engines/_speech_evidence.py",
    "core/engines/echo_coherence.py",
    "core/engines/sherpa.py",
    "core/engines/speaker_gate.py",
    "core/media_session.py",
    "core/metrics.py",
    "core/realtime_media_stage.py",
    "core/tts_markup.py",
    "tools/__init__.py",
    "tools/audio_eval/__init__.py",
    "tools/audio_eval/metrics.py",
    "tools/microsoft_aec_apm_dtd_eval.py",
    "tools/prepare_microsoft_aec_fixture.py",
    "tools/streaming_stt/__init__.py",
    "tools/streaming_stt/bounded_io.py",
)
_CLOSURE_FILES = tuple(
    sorted(
        (
            *_EAGER_LOCAL_IMPORT_FILES,
            "core/engines/_apm.py",
            "requirements-remote.txt",
        )
    )
)
_CONFIG_KEYS = (
    "sample_rate",
    "block_sec",
    "provider",
    "aec_enabled",
    "aec_backend",
    "aec_ref_delay_ms",
    "aec_auto_delay",
    "apm_noise_suppression",
    "apm_high_pass_filter",
    "apm_gain_control",
    "apm_always_on",
    "apm_stream_delay_ms",
    "barge_in_enabled",
    "barge_in_min_speech_sec",
    "barge_in_sustain_window_sec",
    "coherence_barge_in_enabled",
    "coherence_voiced_band_hz",
    "coherence_ring_ms",
    "coherence_max_delay_ms",
    "coherence_nperseg",
    "coherence_margin_delta",
    "coherence_confirm_frames",
    "coherence_warmup_frames",
    "coherence_sigma_k",
    "coherence_baseline_alpha",
    "coherence_var_alpha",
    "coherence_provisional_baseline",
    "dtd_enabled",
    "dtd_k",
    "dtd_weight_raw",
    "dtd_weight_resid",
    "dtd_weight_coh",
    "dtd_coherence_echo_veto",
    "dtd_confirm_frames",
    "dtd_warmup_frames",
    "dtd_chart_rel_floor",
    "dtd_chart_z_freeze",
    "dtd_chart_persist",
    "dtd_chart_robust_seed",
    "dtd_chart_freeze_limit",
    "dtd_residual_floor_margin_db",
)
_SAFE_ERROR = {
    "error": "microsoft_aec_apm_dtd_evaluation_prerequisites_unavailable",
    "ok": False,
}


class MicrosoftAecApmDtdEvalError(RuntimeError):
    """Detail-free fixture, runtime, replay, or publication failure."""


class _LifecycleSignal(BaseException):
    def __init__(self, signum: int) -> None:
        self.signum = int(signum)


@dataclass(frozen=True, slots=True)
class _CaseAudio:
    case: MicrosoftAecCase = field(repr=False)
    tracks: Mapping[str, np.ndarray] = field(repr=False)


@dataclass(frozen=True, slots=True)
class _BundleSnapshot:
    bundle: LoadedMicrosoftAecBundle = field(repr=False)


@dataclass(frozen=True, slots=True)
class _ClosureSnapshot:
    files: tuple[tuple[str, str, int], ...] = field(repr=False)
    sha256: str
    total_bytes: int


@dataclass(frozen=True, slots=True)
class _ConfigurationSnapshot:
    config: SherpaConfig = field(repr=False)
    source_sha256: str
    sherpa_sha256: str
    values: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _RuntimeFile:
    relative: str = field(repr=False)
    sha256: str
    size_bytes: int
    identity: tuple[int, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class _RuntimeModule:
    name: str
    relative: str
    module_identity: int = field(repr=False)
    spec_identity: int = field(repr=False)
    loader_identity: int = field(repr=False)
    module_path_identity: int = field(repr=False)
    spec_locations_identity: int = field(repr=False)


@dataclass(slots=True)
class _RuntimeLibraryState:
    initialization_attempted: bool = False
    client: object | None = field(default=None, repr=False)
    queue: object | None = field(default=None, repr=False)
    cdll: object | None = field(default=None, repr=False)
    resolved_name: str = ""
    native_handle: int = 0
    native_functions: tuple[object, ...] = field(default=(), repr=False)
    native_signatures: tuple[tuple[int, tuple[int, ...], int], ...] = field(
        default=(),
        repr=False,
    )


@dataclass(frozen=True, slots=True)
class _RuntimeSnapshot:
    binding: Mapping[str, object]
    files: tuple[_RuntimeFile, ...] = field(repr=False)
    modules: tuple[_RuntimeModule, ...] = field(default=(), repr=False)
    execution_identity: tuple[int, ...] = field(default=(), repr=False)
    library_state: _RuntimeLibraryState = field(
        default_factory=_RuntimeLibraryState,
        compare=False,
        repr=False,
    )
    record_identity: tuple[int, ...] = field(default=(), repr=False)


@dataclass(slots=True)
class _ReportCommitState:
    committed: bool = False
    digest: str = ""


@dataclass(slots=True)
class _ReplayTotals:
    source_samples: int = 0
    processed_samples: int = 0
    padding_samples: int = 0
    clipped_samples: int = 0
    near_projection_db: list[float] = field(default_factory=list)
    near_cosine: list[float] = field(default_factory=list)
    near_si_sdr: list[float] = field(default_factory=list)
    echo_erle: list[float] = field(default_factory=list)
    echo_energy: float = 0.0
    residual_energy: float = 0.0
    double_projection_db: list[float] = field(default_factory=list)
    double_cosine: list[float] = field(default_factory=list)
    double_si_sdr: list[float] = field(default_factory=list)
    double_interference: list[float] = field(default_factory=list)
    echo_prefix_frames: int = 0
    echo_false_cut_cases: int = 0
    eligible_active_frames: int = 0
    detected_active_frames: int = 0
    detected_cases: int = 0
    cut_latency_ms: list[float] = field(default_factory=list)
    d_score_echo: list[float] = field(default_factory=list)
    d_score_double: list[float] = field(default_factory=list)


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
        raise MicrosoftAecApmDtdEvalError() from None
    return raw + (b"\n" if newline else b"")


def _strict_json(raw: bytes) -> object:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise MicrosoftAecApmDtdEvalError()
            result[key] = value
        return result

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_float=float,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                MicrosoftAecApmDtdEvalError()
            ),
        )
    except MicrosoftAecApmDtdEvalError:
        raise
    except (OverflowError, UnicodeError, ValueError):
        raise MicrosoftAecApmDtdEvalError() from None


def _digest_labeled(items: Sequence[tuple[str, bytes]]) -> str:
    digest = hashlib.sha256()
    for label, raw in items:
        digest.update(label.encode("ascii"))
        digest.update(b"\0")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _execution_closure() -> _ClosureSnapshot:
    items: list[tuple[str, bytes]] = []
    files: list[tuple[str, str, int]] = []
    try:
        for relative in _CLOSURE_FILES:
            loaded = read_regular_bounded(
                _REPO_ROOT / relative,
                maximum_bytes=_MAX_SOURCE_BYTES,
            )
            raw = loaded.data
            digest = hashlib.sha256(raw).hexdigest()
            items.append((relative, raw))
            files.append((relative, digest, len(raw)))
    except BoundedReadError:
        raise MicrosoftAecApmDtdEvalError() from None
    return _ClosureSnapshot(
        files=tuple(files),
        sha256=_digest_labeled(items),
        total_bytes=sum(len(raw) for _label, raw in items),
    )


def _json_scalar(value: object) -> object:
    if isinstance(value, tuple):
        return [_json_scalar(item) for item in value]
    if type(value) in {bool, int, float, str}:
        if isinstance(value, float) and not math.isfinite(value):
            raise MicrosoftAecApmDtdEvalError()
        return value
    raise MicrosoftAecApmDtdEvalError()


def _configuration_snapshot() -> _ConfigurationSnapshot:
    try:
        loaded = read_regular_bounded(
            _CONFIG_PATH,
            maximum_bytes=_MAX_CONFIG_BYTES,
        )
        value = _strict_json(loaded.data)
        if not isinstance(value, dict):
            raise MicrosoftAecApmDtdEvalError()
        effective = apply_device_profile(value, "open_speaker", strict=True)
        sherpa = effective.get("sherpa")
        if not isinstance(sherpa, dict):
            raise MicrosoftAecApmDtdEvalError()
        config = SherpaConfig.from_dict(sherpa)
        values = {key: _json_scalar(getattr(config, key)) for key in _CONFIG_KEYS}
        if (
            config.sample_rate != SAMPLE_RATE_HZ
            or config.block_sec != 0.1
            or str(config.provider).casefold() != "cpu"
            or config.aec_enabled is not True
            or str(config.aec_backend).casefold() != "apm"
            or config.apm_always_on is not True
            or config.apm_noise_suppression is not True
            or config.apm_gain_control is not False
            or config.barge_in_enabled is not True
            or config.coherence_barge_in_enabled is not True
            or config.dtd_enabled is not True
        ):
            raise MicrosoftAecApmDtdEvalError()
        sherpa_sha256 = hashlib.sha256(_canonical_json(values)).hexdigest()
        return _ConfigurationSnapshot(
            config=config,
            source_sha256=hashlib.sha256(loaded.data).hexdigest(),
            sherpa_sha256=sherpa_sha256,
            values=values,
        )
    except MicrosoftAecApmDtdEvalError:
        raise
    except (BoundedReadError, InvalidOperation, KeyError, TypeError, ValueError):
        raise MicrosoftAecApmDtdEvalError() from None


def _file_identity(metadata: os.stat_result) -> tuple[int, ...]:
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


def _required_livekit_version() -> str:
    try:
        loaded = read_regular_bounded(
            _REMOTE_REQUIREMENTS_PATH,
            maximum_bytes=256 * 1024,
        )
        text = loaded.data.decode("utf-8")
        matches: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            match = re.fullmatch(
                r"livekit==([A-Za-z0-9][A-Za-z0-9.+_-]{0,63})(?:\s+#.*)?",
                line,
            )
            if match is not None:
                matches.append(match.group(1))
        if len(matches) != 1:
            raise MicrosoftAecApmDtdEvalError()
        return matches[0]
    except MicrosoftAecApmDtdEvalError:
        raise
    except (BoundedReadError, UnicodeError):
        raise MicrosoftAecApmDtdEvalError() from None


def _safe_distribution_relative(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or "\x00" in value
        or "\r" in value
        or "\n" in value
    ):
        raise MicrosoftAecApmDtdEvalError()
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise MicrosoftAecApmDtdEvalError()
    return value


def _record_sha256(value: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256="):
        raise MicrosoftAecApmDtdEvalError()
    encoded = value.removeprefix("sha256=")
    if not encoded or "=" in encoded:
        raise MicrosoftAecApmDtdEvalError()
    try:
        raw = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
    except (ValueError, TypeError):
        raise MicrosoftAecApmDtdEvalError() from None
    if (
        len(raw) != hashlib.sha256().digest_size
        or base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=") != encoded
    ):
        raise MicrosoftAecApmDtdEvalError()
    return raw.hex()


def _runtime_member(
    distribution: object,
    relative: str,
    *,
    expected_sha256: str,
    expected_size: int,
) -> _RuntimeFile:
    try:
        located = Path(distribution.locate_file(relative))  # type: ignore[attr-defined]
        if not located.is_absolute():
            raise MicrosoftAecApmDtdEvalError()
        before = located.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or before.st_nlink != 1
            or before.st_size != expected_size
            or expected_size > _LIVEKIT_FILE_MAX_BYTES
        ):
            raise MicrosoftAecApmDtdEvalError()
        loaded = read_regular_bounded(
            located,
            maximum_bytes=max(1, expected_size),
            expected_bytes=expected_size,
            allow_empty=True,
        )
        after = located.lstat()
        digest = hashlib.sha256(loaded.data).hexdigest()
        if (
            digest != expected_sha256
            or _file_identity(before) != _file_identity(after)
            or loaded.path != located.resolve(strict=True)
        ):
            raise MicrosoftAecApmDtdEvalError()
        return _RuntimeFile(
            relative=relative,
            sha256=digest,
            size_bytes=expected_size,
            identity=_file_identity(after),
        )
    except MicrosoftAecApmDtdEvalError:
        raise
    except (BoundedReadError, OSError, RuntimeError, TypeError, ValueError):
        raise MicrosoftAecApmDtdEvalError() from None


def _loaded_livekit_names() -> tuple[str, ...]:
    return tuple(
        sorted(
            name
            for name in sys.modules
            if name == "livekit" or name.startswith("livekit.")
        )
    )


def _require_default_livekit_library_env() -> None:
    if os.environ.get("LIVEKIT_LIB_PATH") not in {None, ""}:
        raise MicrosoftAecApmDtdEvalError()


def _exact_module_locations(value: object, expected: Path) -> bool:
    if value is None:
        return False
    try:
        locations = tuple(
            Path(item).resolve(strict=True)
            for item in value  # type: ignore[union-attr]
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        return False
    return locations == (expected,)


def _livekit_namespace_root(distribution: object) -> tuple[Path, Path]:
    try:
        root = Path(distribution.locate_file("livekit"))  # type: ignore[attr-defined]
        rtc_root = root / "rtc"
        if not root.is_absolute():
            raise MicrosoftAecApmDtdEvalError()
        with opened_directory_nofollow(root) as (stable_root, root_fd):
            root_metadata = os.fstat(root_fd)
            if stable_root != root or root_metadata.st_uid != os.getuid():
                raise MicrosoftAecApmDtdEvalError()
        with opened_directory_nofollow(rtc_root) as (stable_rtc, rtc_fd):
            rtc_metadata = os.fstat(rtc_fd)
            if stable_rtc != rtc_root or rtc_metadata.st_uid != os.getuid():
                raise MicrosoftAecApmDtdEvalError()
        for forbidden in (
            root / "__init__.py",
            root / "__init__.pyc",
            root / "__pycache__",
        ):
            try:
                forbidden.lstat()
            except FileNotFoundError:
                continue
            raise MicrosoftAecApmDtdEvalError()
        return root, rtc_root
    except MicrosoftAecApmDtdEvalError:
        raise
    except (BoundedReadError, OSError, RuntimeError, TypeError, ValueError):
        raise MicrosoftAecApmDtdEvalError() from None


def _validate_livekit_tree(
    root: Path,
    rtc_root: Path,
    files: Sequence[_RuntimeFile],
) -> None:
    retained = {item.relative for item in files}
    try:
        for current_root, directories, names in os.walk(rtc_root, followlinks=False):
            current = Path(current_root)
            for directory_name in tuple(directories):
                directory = current / directory_name
                metadata = directory.lstat()
                if (
                    directory_name == "__pycache__"
                    or stat.S_ISLNK(metadata.st_mode)
                    or not stat.S_ISDIR(metadata.st_mode)
                    or metadata.st_uid != os.getuid()
                ):
                    raise MicrosoftAecApmDtdEvalError()
            for name in names:
                member = current / name
                relative = member.relative_to(root.parent).as_posix()
                metadata = member.lstat()
                if (
                    name.endswith(".pyc")
                    or not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_uid != os.getuid()
                    or metadata.st_nlink != 1
                    or relative not in retained
                ):
                    raise MicrosoftAecApmDtdEvalError()
    except MicrosoftAecApmDtdEvalError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise MicrosoftAecApmDtdEvalError() from None


def _namespace_spec(root: Path, *, loaded: bool) -> object:
    try:
        spec = importlib.util.find_spec("livekit")
        loader = None if spec is None else spec.loader
        if (
            spec is None
            or type(spec) is not importlib_machinery.ModuleSpec
            or spec.name != "livekit"
            or spec.origin is not None
            or not _exact_module_locations(spec.submodule_search_locations, root)
            or (loaded and type(loader) is not importlib_machinery.NamespaceLoader)
            or (not loaded and loader is not None)
        ):
            raise MicrosoftAecApmDtdEvalError()
        return spec
    except MicrosoftAecApmDtdEvalError:
        raise
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        raise MicrosoftAecApmDtdEvalError() from None


def _source_spec(name: str, expected: Path, *, package: bool) -> object:
    try:
        spec = importlib.util.find_spec(name)
        if (
            spec is None
            or type(spec) is not importlib_machinery.ModuleSpec
            or spec.name != name
            or type(spec.loader) is not importlib_machinery.SourceFileLoader
            or not isinstance(spec.origin, str)
            or Path(spec.origin) != expected
            or Path(spec.origin).resolve(strict=True) != expected
            or Path(spec.loader.path) != expected
            or spec.loader.name != name
            or set(vars(spec.loader)) != {"name", "path"}
            or bool(spec.submodule_search_locations is not None) is not package
            or (
                package
                and not _exact_module_locations(
                    spec.submodule_search_locations,
                    expected.parent,
                )
            )
        ):
            raise MicrosoftAecApmDtdEvalError()
        return spec
    except MicrosoftAecApmDtdEvalError:
        raise
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        raise MicrosoftAecApmDtdEvalError() from None


def _namespace_package_spec(name: str, expected: Path, *, loaded: bool) -> object:
    try:
        spec = importlib.util.find_spec(name)
        loader = None if spec is None else spec.loader
        if (
            spec is None
            or type(spec) is not importlib_machinery.ModuleSpec
            or spec.name != name
            or spec.origin is not None
            or not _exact_module_locations(spec.submodule_search_locations, expected)
            or (loaded and type(loader) is not importlib_machinery.NamespaceLoader)
            or (not loaded and loader is not None)
        ):
            raise MicrosoftAecApmDtdEvalError()
        return spec
    except MicrosoftAecApmDtdEvalError:
        raise
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        raise MicrosoftAecApmDtdEvalError() from None


def _bound_python_function(value: object, *, module: str) -> tuple[object, object]:
    if (
        type(value) is not FunctionType
        or getattr(value, "__module__", None) != module
        or type(getattr(value, "__code__", None)) is not CodeType
    ):
        raise MicrosoftAecApmDtdEvalError()
    return value, value.__code__  # type: ignore[union-attr]


def _bound_class_function(
    owner: object,
    name: str,
    *,
    module: str,
) -> tuple[object, object]:
    try:
        return _bound_python_function(vars(owner)[name], module=module)
    except (KeyError, TypeError):
        raise MicrosoftAecApmDtdEvalError() from None


def _bound_property_getter(
    owner: object,
    name: str,
    *,
    module: str,
) -> tuple[object, object, object]:
    try:
        descriptor = vars(owner)[name]
        if (
            type(descriptor) is not property
            or descriptor.fget is None
            or descriptor.fset is not None
            or descriptor.fdel is not None
        ):
            raise MicrosoftAecApmDtdEvalError()
        function, code = _bound_python_function(descriptor.fget, module=module)
        return descriptor, function, code
    except MicrosoftAecApmDtdEvalError:
        raise
    except (AttributeError, KeyError, TypeError):
        raise MicrosoftAecApmDtdEvalError() from None


def _surface_name_token(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest(), "big")


def _surface_value_identity(value: object) -> tuple[int, ...]:
    identity = [id(value)]
    if type(value) is FunctionType:
        code = getattr(value, "__code__", None)
        if type(code) is not CodeType:
            raise MicrosoftAecApmDtdEvalError()
        identity.append(id(code))
        globals_map = value.__globals__
        if type(globals_map) is not dict:
            raise MicrosoftAecApmDtdEvalError()
        identity.append(id(globals_map))

        defaults = value.__defaults__
        identity.append(id(defaults))
        if defaults is None:
            identity.append(-1)
        elif type(defaults) is tuple:
            identity.append(len(defaults))
            identity.extend(id(item) for item in defaults)
        else:
            raise MicrosoftAecApmDtdEvalError()

        keyword_defaults = value.__kwdefaults__
        identity.append(id(keyword_defaults))
        if keyword_defaults is None:
            identity.append(-1)
        elif type(keyword_defaults) is dict:
            keyword_items = tuple(sorted(keyword_defaults.items()))
            identity.append(len(keyword_items))
            for name, item in keyword_items:
                if type(name) is not str:
                    raise MicrosoftAecApmDtdEvalError()
                identity.extend((_surface_name_token(name), id(item)))
        else:
            raise MicrosoftAecApmDtdEvalError()

        closure = value.__closure__
        identity.append(id(closure))
        if closure is None:
            identity.append(-1)
        elif type(closure) is tuple:
            identity.append(len(closure))
            for cell in closure:
                identity.append(id(cell))
                try:
                    contents = cell.cell_contents
                except ValueError:
                    identity.append(0)
                else:
                    identity.extend((1, id(contents)))
        else:
            raise MicrosoftAecApmDtdEvalError()

        builtins_object = value.__builtins__
        identity.append(id(builtins_object))
        if type(builtins_object) is dict:
            builtins_map = builtins_object
        elif type(builtins_object) is ModuleType:
            builtins_map = vars(builtins_object)
        else:
            raise MicrosoftAecApmDtdEvalError()
        referenced_names = tuple(sorted(set(code.co_names)))
        identity.append(len(referenced_names))
        for name in referenced_names:
            if type(name) is not str:
                raise MicrosoftAecApmDtdEvalError()
            identity.append(_surface_name_token(name))
            if name in globals_map:
                identity.extend((1, id(globals_map[name])))
            elif name in builtins_map:
                identity.extend((2, id(builtins_map[name])))
            else:
                identity.append(0)
    elif type(value) in {classmethod, staticmethod}:
        identity.extend(_surface_value_identity(value.__func__))
    elif type(value) is property:
        for function in (value.fget, value.fset, value.fdel):
            identity.append(id(function))
            if function is not None:
                identity.extend(_surface_value_identity(function))
    return tuple(identity)


def _immutable_namespace_surface(namespace: Mapping[str, object]) -> tuple[int, ...]:
    try:
        items = tuple(sorted(namespace.items()))
        identity: list[int] = [len(items)]
        for name, value in items:
            if type(name) is not str:
                raise MicrosoftAecApmDtdEvalError()
            identity.append(_surface_name_token(name))
            identity.extend(_surface_value_identity(value))
        return tuple(identity)
    except MicrosoftAecApmDtdEvalError:
        raise
    except (TypeError, UnicodeError):
        raise MicrosoftAecApmDtdEvalError() from None


def _immutable_class_surface(
    owner: object,
    *,
    inherited: Sequence[str] = (),
) -> tuple[int, ...]:
    if not isinstance(owner, type):
        raise MicrosoftAecApmDtdEvalError()
    identity = list(_immutable_namespace_surface(vars(owner)))
    for name in inherited:
        if type(name) is not str:
            raise MicrosoftAecApmDtdEvalError()
        try:
            value = getattr(owner, name)
        except AttributeError:
            raise MicrosoftAecApmDtdEvalError() from None
        identity.append(_surface_name_token(name))
        identity.extend(_surface_value_identity(value))
    return tuple(identity)


def _livekit_apm_execution_identity(
    rtc: object,
    apm: object,
    *,
    expected_version: str,
) -> tuple[int, ...]:
    """Bind the complete package-local execution graph used by the APM path."""

    try:
        ffi = sys.modules.get("livekit.rtc._ffi_client")
        utils = sys.modules.get("livekit.rtc._utils")
        audio_frame = sys.modules.get("livekit.rtc.audio_frame")
        proto = sys.modules.get("livekit.rtc._proto.ffi_pb2")
        version = sys.modules.get("livekit.rtc.version")
        named_modules = {
            "livekit.rtc": rtc,
            "livekit.rtc.apm": apm,
            "livekit.rtc._ffi_client": ffi,
            "livekit.rtc._utils": utils,
            "livekit.rtc.audio_frame": audio_frame,
            "livekit.rtc._proto.ffi_pb2": proto,
            "livekit.rtc.version": version,
        }
        if any(
            type(module) is not ModuleType or getattr(module, "__name__", None) != name
            for name, module in named_modules.items()
        ):
            raise MicrosoftAecApmDtdEvalError()

        rtc_values = vars(rtc)
        apm_values = vars(apm)
        ffi_values = vars(ffi)
        utils_values = vars(utils)
        frame_values = vars(audio_frame)
        proto_values = vars(proto)
        version_values = vars(version)

        apm_class = apm_values.get("AudioProcessingModule")
        frame_class = frame_values.get("AudioFrame")
        ffi_client = ffi_values.get("FfiClient")
        ffi_handle = ffi_values.get("FfiHandle")
        ffi_queue = ffi_values.get("FfiQueue")
        classproperty_class = utils_values.get("classproperty")
        get_address = utils_values.get("get_address")
        ensure_buffer = utils_values.get("_ensure_compatible_buffer")
        proto_request = proto_values.get("FfiRequest")
        proto_response = proto_values.get("FfiResponse")
        proto_event = proto_values.get("FfiEvent")
        proto_log_level = proto_values.get("LogLevel")
        version_value = version_values.get("__version__")
        if (
            not isinstance(apm_class, type)
            or getattr(apm_class, "__module__", None) != "livekit.rtc.apm"
            or not isinstance(frame_class, type)
            or getattr(frame_class, "__module__", None) != "livekit.rtc.audio_frame"
            or not isinstance(ffi_client, type)
            or getattr(ffi_client, "__module__", None) != "livekit.rtc._ffi_client"
            or not isinstance(ffi_handle, type)
            or getattr(ffi_handle, "__module__", None) != "livekit.rtc._ffi_client"
            or not isinstance(ffi_queue, type)
            or getattr(ffi_queue, "__module__", None) != "livekit.rtc._ffi_client"
            or not isinstance(classproperty_class, type)
            or getattr(classproperty_class, "__module__", None) != "livekit.rtc._utils"
            or not isinstance(expected_version, str)
            or version_value != expected_version
            or ffi_values.get("__version__") is not version_value
            or rtc_values.get("__version__") is not version_value
        ):
            raise MicrosoftAecApmDtdEvalError()

        if (
            rtc_values.get("AudioProcessingModule") is not apm_class
            or rtc_values.get("AudioFrame") is not frame_class
            or apm_values.get("FfiClient") is not ffi_client
            or apm_values.get("FfiHandle") is not ffi_handle
            or apm_values.get("AudioFrame") is not frame_class
            or apm_values.get("get_address") is not get_address
            or apm_values.get("proto_ffi") is not proto
            or ffi_values.get("proto_ffi") is not proto
            or ffi_values.get("classproperty") is not classproperty_class
            or frame_values.get("FfiHandle") is not ffi_handle
            or frame_values.get("get_address") is not get_address
            or frame_values.get("_ensure_compatible_buffer") is not ensure_buffer
            or frame_values.get("ctypes") is not ffi_values.get("ctypes")
            or utils_values.get("ctypes") is not ffi_values.get("ctypes")
        ):
            raise MicrosoftAecApmDtdEvalError()

        functions: list[object] = []
        for owner, names, module_name in (
            (
                apm_class,
                (
                    "__init__",
                    "process_stream",
                    "process_reverse_stream",
                    "set_stream_delay_ms",
                ),
                "livekit.rtc.apm",
            ),
            (
                ffi_client,
                ("__init__", "request"),
                "livekit.rtc._ffi_client",
            ),
            (
                ffi_handle,
                ("__init__", "__del__", "dispose"),
                "livekit.rtc._ffi_client",
            ),
            (
                ffi_queue,
                ("__init__", "put"),
                "livekit.rtc._ffi_client",
            ),
            (
                classproperty_class,
                ("__init__", "__get__"),
                "livekit.rtc._utils",
            ),
            (
                frame_class,
                ("__init__",),
                "livekit.rtc.audio_frame",
            ),
        ):
            for name in names:
                functions.extend(_bound_class_function(owner, name, module=module_name))
        for function, module_name in (
            (ffi_values.get("get_ffi_lib"), "livekit.rtc._ffi_client"),
            (ffi_values.get("_lib_name"), "livekit.rtc._ffi_client"),
            (ffi_values.get("to_python_level"), "livekit.rtc._ffi_client"),
            (get_address, "livekit.rtc._utils"),
            (ensure_buffer, "livekit.rtc._utils"),
        ):
            functions.extend(_bound_python_function(function, module=module_name))

        descriptors: list[object] = []
        instance_descriptor = vars(ffi_client).get("instance")
        if type(instance_descriptor) is not classproperty_class:
            raise MicrosoftAecApmDtdEvalError()
        instance_classmethod = vars(instance_descriptor).get("f")
        if type(instance_classmethod) is not classmethod:
            raise MicrosoftAecApmDtdEvalError()
        instance_getter = instance_classmethod.__func__
        instance_function, instance_code = _bound_python_function(
            instance_getter,
            module="livekit.rtc._ffi_client",
        )
        descriptors.extend(
            (
                instance_descriptor,
                instance_classmethod,
                instance_function,
                instance_code,
            )
        )
        descriptors.extend(
            _bound_property_getter(
                ffi_client,
                "queue",
                module="livekit.rtc._ffi_client",
            )
        )
        for name in ("data", "sample_rate", "num_channels"):
            descriptors.extend(
                _bound_property_getter(
                    frame_class,
                    name,
                    module="livekit.rtc.audio_frame",
                )
            )

        ffi_callback = ffi_values.get("ffi_event_callback")
        ffi_callback_type = ffi_values.get("ffi_cb_fnc")
        ffi_resource_files = ffi_values.get("_resource_files")
        ffi_logger = ffi_values.get("logger")
        if (
            ffi_values.get("INVALID_HANDLE") != 0
            or ffi_callback is None
            or ffi_callback_type is None
            or ffi_resource_files is None
            or ffi_logger is None
            or proto_request is None
            or proto_response is None
            or proto_event is None
            or proto_log_level is None
        ):
            raise MicrosoftAecApmDtdEvalError()
        direct_globals = tuple(
            ffi_values.get(name)
            for name in (
                "atexit",
                "ctypes",
                "importlib",
                "logging",
                "os",
                "platform",
                "signal",
                "sys",
                "threading",
            )
        )
        if any(type(value) is not ModuleType for value in direct_globals):
            raise MicrosoftAecApmDtdEvalError()

        bound = (
            *named_modules.values(),
            apm_class,
            frame_class,
            ffi_client,
            ffi_handle,
            ffi_queue,
            classproperty_class,
            get_address,
            ensure_buffer,
            proto_request,
            proto_response,
            proto_event,
            proto_log_level,
            version_value,
            ffi_callback,
            ffi_callback_type,
            ffi_resource_files,
            ffi_logger,
            ffi_values.get("INVALID_HANDLE"),
            *direct_globals,
            *functions,
            *descriptors,
        )
        proto_surface: list[int] = []
        message_methods = (
            "__new__",
            "__init__",
            "SerializeToString",
            "ParseFromString",
        )
        for message_class in (proto_request, proto_response, proto_event):
            proto_surface.extend(
                _immutable_class_surface(
                    message_class,
                    inherited=message_methods,
                )
            )
        proto_surface.extend(_immutable_namespace_surface(vars(proto_log_level)))
        proto_surface.extend(_immutable_class_surface(type(proto_log_level)))
        return (*tuple(id(value) for value in bound), *proto_surface)
    except MicrosoftAecApmDtdEvalError:
        raise
    except (AttributeError, KeyError, TypeError):
        raise MicrosoftAecApmDtdEvalError() from None


def _bind_livekit_execution_surface(
    distribution: object,
    files: Sequence[_RuntimeFile],
    *,
    retained_modules: tuple[_RuntimeModule, ...] | None,
    retained_execution: tuple[int, ...] | None,
) -> tuple[tuple[_RuntimeModule, ...], tuple[int, ...]]:
    _require_default_livekit_library_env()
    root, rtc_root = _livekit_namespace_root(distribution)
    retained_files = {item.relative for item in files}
    rtc_init = rtc_root / "__init__.py"
    apm_source = rtc_root / "apm.py"
    resources_root = rtc_root / "resources"
    resources_init = resources_root / "__init__.py"
    resources_init_relative = resources_init.relative_to(root.parent).as_posix()
    resources_namespace = resources_init_relative not in retained_files
    try:
        with opened_directory_nofollow(resources_root) as (
            stable_resources,
            resources_fd,
        ):
            resources_metadata = os.fstat(resources_fd)
            if (
                stable_resources != resources_root
                or resources_metadata.st_uid != os.getuid()
            ):
                raise MicrosoftAecApmDtdEvalError()
    except (BoundedReadError, OSError, RuntimeError, ValueError):
        raise MicrosoftAecApmDtdEvalError() from None
    if not {
        rtc_init.relative_to(root.parent).as_posix(),
        apm_source.relative_to(root.parent).as_posix(),
    } <= retained_files or not any(
        PurePosixPath(relative).parent.as_posix() == "livekit/rtc/resources"
        and PurePosixPath(relative).name in _LIVEKIT_NATIVE_NAMES
        for relative in retained_files
    ):
        raise MicrosoftAecApmDtdEvalError()
    _validate_livekit_tree(root, rtc_root, files)
    importlib.invalidate_caches()
    preloaded = _loaded_livekit_names()
    expected_names = (
        ()
        if retained_modules is None
        else tuple(item.name for item in retained_modules)
    )
    if (retained_modules is None and preloaded) or (
        retained_modules is not None and preloaded != expected_names
    ):
        raise MicrosoftAecApmDtdEvalError()
    _namespace_spec(root, loaded=retained_modules is not None)

    initial_import = retained_modules is None
    previous_dont_write = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        namespace = importlib.import_module("livekit")
        _namespace_spec(root, loaded=True)
        _source_spec("livekit.rtc", rtc_init, package=True)
        rtc = importlib.import_module("livekit.rtc")
        _source_spec("livekit.rtc.apm", apm_source, package=False)
        apm = importlib.import_module("livekit.rtc.apm")
        if resources_namespace:
            _namespace_package_spec(
                "livekit.rtc.resources",
                resources_root,
                loaded=retained_modules is not None,
            )
        else:
            _source_spec(
                "livekit.rtc.resources",
                resources_init,
                package=True,
            )
        resources = importlib.import_module("livekit.rtc.resources")
        if resources_namespace:
            _namespace_package_spec(
                "livekit.rtc.resources",
                resources_root,
                loaded=True,
            )
        else:
            _source_spec(
                "livekit.rtc.resources",
                resources_init,
                package=True,
            )
    except BaseException as error:
        if initial_import:
            for name in _loaded_livekit_names():
                sys.modules.pop(name, None)
        if isinstance(error, (KeyboardInterrupt, _LifecycleSignal)):
            raise
        if isinstance(error, MicrosoftAecApmDtdEvalError):
            raise
        raise MicrosoftAecApmDtdEvalError() from None
    finally:
        sys.dont_write_bytecode = previous_dont_write

    if (
        sys.modules.get("livekit") is not namespace
        or sys.modules.get("livekit.rtc") is not rtc
        or sys.modules.get("livekit.rtc.apm") is not apm
        or sys.modules.get("livekit.rtc.resources") is not resources
        or getattr(rtc, "resources", None) is not resources
        or getattr(namespace, "__spec__", None) is None
        or getattr(namespace, "__loader__", None) is not namespace.__spec__.loader
        or not _exact_module_locations(getattr(namespace, "__path__", None), root)
        or getattr(rtc, "AudioProcessingModule", None)
        is not getattr(apm, "AudioProcessingModule", None)
        or getattr(getattr(apm, "AudioProcessingModule", None), "__module__", None)
        != "livekit.rtc.apm"
    ):
        raise MicrosoftAecApmDtdEvalError()
    execution_identity = _livekit_apm_execution_identity(
        rtc,
        apm,
        expected_version=distribution.version,  # type: ignore[attr-defined]
    )
    if retained_execution is not None and execution_identity != retained_execution:
        raise MicrosoftAecApmDtdEvalError()

    observed: list[_RuntimeModule] = []
    for name in _loaded_livekit_names():
        module = sys.modules.get(name)
        spec = getattr(module, "__spec__", None)
        loader = None if spec is None else spec.loader
        if name == "livekit":
            if (
                module is not namespace
                or type(module) is not ModuleType
                or spec is None
                or type(spec) is not importlib_machinery.ModuleSpec
                or type(loader) is not importlib_machinery.NamespaceLoader
                or spec.origin is not None
                or not _exact_module_locations(spec.submodule_search_locations, root)
            ):
                raise MicrosoftAecApmDtdEvalError()
            relative = "livekit"
        elif name == "livekit.rtc.resources" and resources_namespace:
            if (
                module is not resources
                or type(module) is not ModuleType
                or spec is None
                or type(spec) is not importlib_machinery.ModuleSpec
                or type(loader) is not importlib_machinery.NamespaceLoader
                or spec.origin is not None
                or getattr(module, "__loader__", None) is not loader
                or not _exact_module_locations(
                    spec.submodule_search_locations,
                    resources_root,
                )
                or not _exact_module_locations(
                    getattr(module, "__path__", None),
                    resources_root,
                )
            ):
                raise MicrosoftAecApmDtdEvalError()
            relative = "livekit/rtc/resources"
        else:
            if not name.startswith("livekit.rtc"):
                raise MicrosoftAecApmDtdEvalError()
            origin = None if spec is None else spec.origin
            module_file = getattr(module, "__file__", None)
            if (
                spec is None
                or type(module) is not ModuleType
                or type(spec) is not importlib_machinery.ModuleSpec
                or spec.name != name
                or type(loader) is not importlib_machinery.SourceFileLoader
                or not isinstance(origin, str)
                or not isinstance(module_file, str)
                or Path(origin) != Path(module_file)
                or Path(origin).resolve(strict=True) != Path(origin)
                or getattr(module, "__loader__", None) is not loader
                or Path(loader.path) != Path(origin)
                or loader.name != name
                or set(vars(loader)) != {"name", "path"}
            ):
                raise MicrosoftAecApmDtdEvalError()
            try:
                relative = Path(origin).relative_to(root.parent).as_posix()
            except ValueError:
                raise MicrosoftAecApmDtdEvalError() from None
            if relative not in retained_files or relative.endswith(".pyc"):
                raise MicrosoftAecApmDtdEvalError()
            if spec.submodule_search_locations is not None and not (
                Path(origin).name == "__init__.py"
                and _exact_module_locations(
                    spec.submodule_search_locations,
                    Path(origin).parent,
                )
            ):
                raise MicrosoftAecApmDtdEvalError()
        module_path = getattr(module, "__path__", None)
        spec_locations = spec.submodule_search_locations
        if spec_locations is None:
            if module_path is not None:
                raise MicrosoftAecApmDtdEvalError()
            module_path_identity = 0
            spec_locations_identity = 0
        else:
            if module_path is not spec_locations:
                raise MicrosoftAecApmDtdEvalError()
            module_path_identity = id(module_path)
            spec_locations_identity = id(spec_locations)
        observed.append(
            _RuntimeModule(
                name=name,
                relative=relative,
                module_identity=id(module),
                spec_identity=id(spec),
                loader_identity=id(loader),
                module_path_identity=module_path_identity,
                spec_locations_identity=spec_locations_identity,
            )
        )
    result = tuple(observed)
    if retained_modules is not None and result != retained_modules:
        raise MicrosoftAecApmDtdEvalError()
    _require_default_livekit_library_env()
    _validate_livekit_tree(root, rtc_root, files)
    return result, execution_identity


def _bind_livekit_library(
    root: Path,
    files: Sequence[_RuntimeFile],
    state: _RuntimeLibraryState,
) -> None:
    """Initialize and retain the verified process-global FFI singleton.

    Accessing ``FfiClient.instance`` can initialize native process state and
    register native callbacks.  That action has no rollback.  Once attempted,
    any later evaluator failure remains terminal and must not be represented as
    restoring a pre-initialization process.
    """

    sentinel = object()
    try:
        ffi_module = sys.modules.get("livekit.rtc._ffi_client")
        apm_module = sys.modules.get("livekit.rtc.apm")
        if type(ffi_module) is not ModuleType or type(apm_module) is not ModuleType:
            raise MicrosoftAecApmDtdEvalError()
        ffi_client = vars(ffi_module).get("FfiClient", sentinel)
        ffi_queue = vars(ffi_module).get("FfiQueue", sentinel)
        if (
            not isinstance(ffi_client, type)
            or getattr(ffi_client, "__module__", None) != "livekit.rtc._ffi_client"
            or not isinstance(ffi_queue, type)
            or getattr(ffi_queue, "__module__", None) != "livekit.rtc._ffi_client"
            or vars(apm_module).get("FfiClient", sentinel) is not ffi_client
        ):
            raise MicrosoftAecApmDtdEvalError()
        instance = vars(ffi_client).get("_instance", sentinel)
        if instance is sentinel:
            raise MicrosoftAecApmDtdEvalError()
        if state.client is None and (
            state.queue is not None
            or state.cdll is not None
            or state.resolved_name
            or state.native_handle
            or state.native_functions
            or state.native_signatures
        ):
            raise MicrosoftAecApmDtdEvalError()
        if state.client is not None and instance is not state.client:
            raise MicrosoftAecApmDtdEvalError()
        if instance is not None and type(instance) is not ffi_client:
            raise MicrosoftAecApmDtdEvalError()

        state.initialization_attempted = True
        _require_default_livekit_library_env()
        observed_instance = getattr(ffi_client, "instance")
        _require_default_livekit_library_env()
        instance = vars(ffi_client).get("_instance", sentinel)
        if (
            instance is sentinel
            or instance is None
            or observed_instance is not instance
            or type(instance) is not ffi_client
        ):
            raise MicrosoftAecApmDtdEvalError()
        library = vars(instance).get("_ffi_lib", sentinel)
        queue = vars(instance).get("_queue", sentinel)
        if type(library) is not ctypes.CDLL or type(queue) is not ffi_queue:
            raise MicrosoftAecApmDtdEvalError()
        raw_name = vars(library).get("_name", sentinel)
        if isinstance(raw_name, bytes):
            raw_name = os.fsdecode(raw_name)
        if not isinstance(raw_name, str) or not raw_name or "\x00" in raw_name:
            raise MicrosoftAecApmDtdEvalError()
        library_path = Path(raw_name)
        resolved = library_path.resolve(strict=True)
        if not library_path.is_absolute() or resolved != library_path:
            raise MicrosoftAecApmDtdEvalError()
        relative = resolved.relative_to(root.parent).as_posix()
        matching = tuple(item for item in files if item.relative == relative)
        before = resolved.lstat()
        if (
            len(matching) != 1
            or PurePosixPath(relative).parent.as_posix() != "livekit/rtc/resources"
            or PurePosixPath(relative).name not in _LIVEKIT_NATIVE_NAMES
            or not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or before.st_nlink != 1
            or _file_identity(before) != matching[0].identity
        ):
            raise MicrosoftAecApmDtdEvalError()
        loaded = read_regular_bounded(
            resolved,
            maximum_bytes=max(1, matching[0].size_bytes),
            expected_bytes=matching[0].size_bytes,
            allow_empty=True,
        )
        after = resolved.lstat()
        if (
            loaded.path != resolved
            or hashlib.sha256(loaded.data).hexdigest() != matching[0].sha256
            or _file_identity(before) != _file_identity(after)
            or _file_identity(after) != matching[0].identity
        ):
            raise MicrosoftAecApmDtdEvalError()
        resolved_name = str(resolved)
        native_handle = vars(library).get("_handle", sentinel)
        if type(native_handle) is not int or native_handle <= 0:
            raise MicrosoftAecApmDtdEvalError()
        native_functions = tuple(
            vars(library).get(name, sentinel)
            for name in (
                "livekit_ffi_initialize",
                "livekit_ffi_request",
                "livekit_ffi_drop_handle",
                "livekit_ffi_dispose",
            )
        )
        if any(function is sentinel for function in native_functions):
            raise MicrosoftAecApmDtdEvalError()
        native_signatures: list[tuple[int, tuple[int, ...], int]] = []
        for function in native_functions:
            argtypes = getattr(function, "argtypes", sentinel)
            restype = getattr(function, "restype", sentinel)
            if type(argtypes) is not list or restype is sentinel:
                raise MicrosoftAecApmDtdEvalError()
            native_signatures.append(
                (id(argtypes), tuple(id(value) for value in argtypes), id(restype))
            )
        signature_tuple = tuple(native_signatures)
        if state.client is not None and (
            instance is not state.client
            or queue is not state.queue
            or library is not state.cdll
            or resolved_name != state.resolved_name
            or native_handle != state.native_handle
            or len(native_functions) != len(state.native_functions)
            or any(
                observed is not retained
                for observed, retained in zip(
                    native_functions,
                    state.native_functions,
                    strict=True,
                )
            )
            or signature_tuple != state.native_signatures
        ):
            raise MicrosoftAecApmDtdEvalError()
        if state.client is None and (
            state.queue is not None
            or state.cdll is not None
            or state.resolved_name
            or state.native_handle
            or state.native_functions
            or state.native_signatures
        ):
            raise MicrosoftAecApmDtdEvalError()
        if state.client is None:
            state.client = instance
            state.queue = queue
            state.cdll = library
            state.resolved_name = resolved_name
            state.native_handle = native_handle
            state.native_functions = native_functions
            state.native_signatures = signature_tuple
    except MicrosoftAecApmDtdEvalError:
        raise
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, _LifecycleSignal)):
            raise
        raise MicrosoftAecApmDtdEvalError() from None


def _livekit_runtime_snapshot(
    *,
    retained: _RuntimeSnapshot | None = None,
) -> _RuntimeSnapshot:
    preexisting_modules = _loaded_livekit_names()
    library_state = (
        _RuntimeLibraryState() if retained is None else retained.library_state
    )
    try:
        _require_default_livekit_library_env()
        distribution = importlib_metadata.distribution("livekit")
        version = distribution.version
        if (
            not isinstance(version, str)
            or _SAFE_LABEL_RE.fullmatch(version) is None
            or version != _required_livekit_version()
        ):
            raise MicrosoftAecApmDtdEvalError()
        distribution_files = distribution.files
        if distribution_files is None:
            raise MicrosoftAecApmDtdEvalError()
        record_candidates = [
            item
            for item in distribution_files
            if item.name == "RECORD"
            and item.parent.name.casefold().endswith(".dist-info")
        ]
        if len(record_candidates) != 1:
            raise MicrosoftAecApmDtdEvalError()
        record_relative = _safe_distribution_relative(str(record_candidates[0]))
        record_path = Path(distribution.locate_file(record_relative))
        record_before = record_path.lstat()
        if (
            not record_path.is_absolute()
            or not stat.S_ISREG(record_before.st_mode)
            or record_before.st_uid != os.getuid()
            or record_before.st_nlink != 1
        ):
            raise MicrosoftAecApmDtdEvalError()
        record_loaded = read_regular_bounded(
            record_path,
            maximum_bytes=_LIVEKIT_RECORD_MAX_BYTES,
        )
        record_after = record_path.lstat()
        if _file_identity(record_before) != _file_identity(record_after):
            raise MicrosoftAecApmDtdEvalError()
        record_raw = record_loaded.data
        record_text = record_raw.decode("utf-8")
        rows = list(csv.reader(io.StringIO(record_text, newline=""), strict=True))
        if not rows or len(rows) > _LIVEKIT_FILE_MAX_COUNT * 2:
            raise MicrosoftAecApmDtdEvalError()
        seen: set[str] = set()
        selected: list[tuple[str, str, int]] = []
        for row in rows:
            if len(row) != 3:
                raise MicrosoftAecApmDtdEvalError()
            relative = _safe_distribution_relative(row[0])
            if relative in seen:
                raise MicrosoftAecApmDtdEvalError()
            seen.add(relative)
            parts = PurePosixPath(relative).parts
            if (
                not parts
                or parts[0] != "livekit"
                or "__pycache__" in parts
                or relative.endswith(".pyc")
            ):
                continue
            expected_sha256 = _record_sha256(row[1])
            if not row[2].isascii() or not row[2].isdigit():
                raise MicrosoftAecApmDtdEvalError()
            expected_size = int(row[2])
            if expected_size > _LIVEKIT_FILE_MAX_BYTES:
                raise MicrosoftAecApmDtdEvalError()
            selected.append((relative, expected_sha256, expected_size))
        if not selected or len(selected) > _LIVEKIT_FILE_MAX_COUNT:
            raise MicrosoftAecApmDtdEvalError()
        selected.sort(key=lambda item: item[0])
        files = tuple(
            _runtime_member(
                distribution,
                relative,
                expected_sha256=expected_sha256,
                expected_size=expected_size,
            )
            for relative, expected_sha256, expected_size in selected
        )
        total_bytes = sum(item.size_bytes for item in files)
        maximum_file_bytes = max(item.size_bytes for item in files)
        if total_bytes > _LIVEKIT_TOTAL_MAX_BYTES:
            raise MicrosoftAecApmDtdEvalError()
        modules, execution_identity = _bind_livekit_execution_surface(
            distribution,
            files,
            retained_modules=None if retained is None else retained.modules,
            retained_execution=(
                None if retained is None else retained.execution_identity
            ),
        )
        post_files = tuple(
            _runtime_member(
                distribution,
                relative,
                expected_sha256=expected_sha256,
                expected_size=expected_size,
            )
            for relative, expected_sha256, expected_size in selected
        )
        if post_files != files or _file_identity(record_path.lstat()) != _file_identity(
            record_after
        ):
            raise MicrosoftAecApmDtdEvalError()
        content_set_sha256 = _digest_labeled(
            tuple(
                (
                    item.relative,
                    bytes.fromhex(item.sha256) + item.size_bytes.to_bytes(8, "big"),
                )
                for item in files
            )
        )
        binding = {
            "configuration_profile": "open_speaker",
            "cpu_only": True,
            "implementation": "livekit.rtc.AudioProcessingModule",
            "livekit": {
                "content_set_sha256": content_set_sha256,
                "distribution": "livekit",
                "file_count": len(files),
                "maximum_file_bytes": maximum_file_bytes,
                "record_sha256": hashlib.sha256(record_raw).hexdigest(),
                "record_size_bytes": len(record_raw),
                "total_bytes": total_bytes,
                "version": version,
            },
            "production_evidence": True,
        }
        root, _rtc_root = _livekit_namespace_root(distribution)
        _bind_livekit_library(root, post_files, library_state)
        _require_default_livekit_library_env()
        snapshot = _RuntimeSnapshot(
            binding=binding,
            files=post_files,
            modules=modules,
            execution_identity=execution_identity,
            library_state=library_state,
            record_identity=_file_identity(record_after),
        )
        return snapshot
    except MicrosoftAecApmDtdEvalError:
        if (
            retained is None
            and not preexisting_modules
            and not library_state.initialization_attempted
        ):
            for name in _loaded_livekit_names():
                sys.modules.pop(name, None)
            importlib.invalidate_caches()
        raise
    except BaseException as error:
        if (
            retained is None
            and not preexisting_modules
            and not library_state.initialization_attempted
        ):
            for name in _loaded_livekit_names():
                sys.modules.pop(name, None)
            importlib.invalidate_caches()
        if isinstance(error, (KeyboardInterrupt, _LifecycleSignal)):
            raise
        raise MicrosoftAecApmDtdEvalError() from None


def _runtime_binding(
    *,
    injected: bool,
    retained: _RuntimeSnapshot | None = None,
) -> _RuntimeSnapshot:
    if injected:
        return _RuntimeSnapshot(
            binding={
                "configuration_profile": "open_speaker",
                "cpu_only": True,
                "implementation": "injected-test-double",
                "livekit": None,
                "production_evidence": False,
            },
            files=(),
        )
    return _livekit_runtime_snapshot(retained=retained)


def _read_case_audio(
    bundle: LoadedMicrosoftAecBundle,
    case: MicrosoftAecCase,
) -> _CaseAudio:
    try:
        if (
            case not in bundle.cases
            or tuple(artifact.role for artifact in case.artifacts) != SIGNAL_ROLES
        ):
            raise MicrosoftAecApmDtdEvalError()
        tracks: dict[str, np.ndarray] = {}
        for artifact in case.artifacts:
            loaded = read_regular_bounded(
                bundle.root / artifact.name,
                maximum_bytes=_MAX_AUDIO_BYTES,
                expected_bytes=artifact.size_bytes,
            )
            if hashlib.sha256(loaded.data).hexdigest() != artifact.sha256:
                raise MicrosoftAecApmDtdEvalError()
            samples = decode_microsoft_aec_wav(loaded.data)
            if (
                samples.size != artifact.samples
                or samples.ndim != 1
                or samples.dtype.kind != "f"
                or not bool(np.all(np.isfinite(samples)))
            ):
                raise MicrosoftAecApmDtdEvalError()
            samples = np.asarray(samples, dtype=np.float32).copy()
            samples.setflags(write=False)
            tracks[artifact.role] = samples
        if (
            set(tracks) != set(SIGNAL_ROLES)
            or len({track.size for track in tracks.values()}) != 1
        ):
            raise MicrosoftAecApmDtdEvalError()
        return _CaseAudio(case=case, tracks=tracks)
    except MicrosoftAecApmDtdEvalError:
        raise
    except (BoundedReadError, OSError, RuntimeError, ValueError):
        raise MicrosoftAecApmDtdEvalError() from None


def _load_bundle_snapshot(path: Path | str) -> _BundleSnapshot:
    try:
        bundle = load_microsoft_aec_bundle(path)
        verify_microsoft_aec_bundle(bundle)
        if not bundle.cases:
            raise MicrosoftAecApmDtdEvalError()
        if bundle.production_evidence and len(bundle.cases) != PRODUCTION_CASES:
            raise MicrosoftAecApmDtdEvalError()
        return _BundleSnapshot(bundle=bundle)
    except MicrosoftAecApmDtdEvalError:
        raise
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, _LifecycleSignal)):
            raise
        raise MicrosoftAecApmDtdEvalError() from None


def _verify_bound_state(
    bundle: _BundleSnapshot,
    closure: _ClosureSnapshot,
    configuration: _ConfigurationSnapshot,
    runtime: _RuntimeSnapshot,
    *,
    injected: bool,
    run_guard: Callable[[], None],
) -> None:
    try:
        run_guard()
        verify_microsoft_aec_bundle(bundle.bundle)
        if _execution_closure() != closure:
            raise MicrosoftAecApmDtdEvalError()
        if _configuration_snapshot() != configuration:
            raise MicrosoftAecApmDtdEvalError()
        if _runtime_binding(injected=injected, retained=runtime) != runtime:
            raise MicrosoftAecApmDtdEvalError()
    except MicrosoftAecApmDtdEvalError:
        raise
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, _LifecycleSignal)):
            raise
        raise MicrosoftAecApmDtdEvalError() from None


def _scaled_near(case: _CaseAudio) -> np.ndarray:
    try:
        decimal_scale = Decimal(case.case.nearend_scale)
        if not decimal_scale.is_finite() or decimal_scale <= 0:
            raise MicrosoftAecApmDtdEvalError()
        scale = np.float32(str(decimal_scale))
    except (InvalidOperation, OverflowError, TypeError, ValueError):
        raise MicrosoftAecApmDtdEvalError() from None
    source = case.tracks["nearend_speech"]
    if not bool(np.isfinite(scale)) or scale <= np.float32(0.0):
        raise MicrosoftAecApmDtdEvalError()
    target = np.asarray(source, dtype=np.float32) * scale
    if not bool(np.all(np.isfinite(target))) or signal_power(target) == 0.0:
        raise MicrosoftAecApmDtdEvalError()
    return np.asarray(target, dtype=np.float32)


def _terminal_padding(samples: int) -> int:
    padding = (-samples) % BLOCK_SAMPLES
    if padding > MAX_TERMINAL_PADDING_SAMPLES:
        raise MicrosoftAecApmDtdEvalError()
    return padding


def _padded(samples: np.ndarray) -> tuple[np.ndarray, int]:
    padding = _terminal_padding(int(samples.size))
    if padding:
        return np.concatenate(
            (np.asarray(samples), np.zeros(padding, dtype=samples.dtype))
        ), padding
    return np.asarray(samples).copy(), 0


def _processor(
    factory: Callable[[SherpaConfig], object],
    config: SherpaConfig,
    *,
    require_production_apm: bool,
) -> object:
    try:
        processor = factory(config)
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, _LifecycleSignal)):
            raise
        raise MicrosoftAecApmDtdEvalError() from None
    if processor is None or not callable(getattr(processor, "process_16k", None)):
        raise MicrosoftAecApmDtdEvalError()
    if require_production_apm and (
        getattr(processor, "sample_rate", None) != SAMPLE_RATE_HZ
        or getattr(processor, "always_on", None) is not True
        or getattr(processor, "suppresses_noise", None) is not True
        or getattr(processor, "suppresses_nearend", None) is not True
    ):
        raise MicrosoftAecApmDtdEvalError()
    return processor


def _retire_processor(processor: object) -> None:
    reset = getattr(processor, "reset", None)
    if reset is None:
        return
    if not callable(reset):
        raise MicrosoftAecApmDtdEvalError()
    try:
        reset()
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, _LifecycleSignal)):
            raise
        raise MicrosoftAecApmDtdEvalError() from None


def _process_phase(
    processor: object,
    near: np.ndarray,
    far: np.ndarray,
    *,
    observe: Callable[[int, np.ndarray, np.ndarray, np.ndarray], None] | None = None,
) -> tuple[np.ndarray, int, int]:
    if (
        not isinstance(near, np.ndarray)
        or not isinstance(far, np.ndarray)
        or near.ndim != 1
        or far.ndim != 1
        or near.size == 0
        or near.size != far.size
        or near.dtype.kind != "f"
        or far.dtype.kind != "f"
        or not bool(np.all(np.isfinite(near)))
        or not bool(np.all(np.isfinite(far)))
    ):
        raise MicrosoftAecApmDtdEvalError()
    near_padded, padding = _padded(near)
    far_padded, far_padding = _padded(far)
    if padding != far_padding or near_padded.size % BLOCK_SAMPLES:
        raise MicrosoftAecApmDtdEvalError()
    outputs: list[np.ndarray] = []
    clipped = 0
    for frame_index, start in enumerate(range(0, near_padded.size, BLOCK_SAMPLES)):
        near_block = np.asarray(
            near_padded[start : start + BLOCK_SAMPLES], dtype=np.float32
        ).copy()
        far_block = np.asarray(
            far_padded[start : start + BLOCK_SAMPLES], dtype=np.float32
        ).copy()
        try:
            raw_output = processor.process_16k(near_block, far_block)
        except BaseException as error:
            if isinstance(error, (KeyboardInterrupt, _LifecycleSignal)):
                raise
            raise MicrosoftAecApmDtdEvalError() from None
        if (
            not isinstance(raw_output, np.ndarray)
            or raw_output.ndim != 1
            or raw_output.shape != (BLOCK_SAMPLES,)
            or raw_output.dtype.kind != "f"
        ):
            raise MicrosoftAecApmDtdEvalError()
        output = np.asarray(raw_output, dtype=np.float64).copy()
        if not bool(np.all(np.isfinite(output))):
            raise MicrosoftAecApmDtdEvalError()
        clipped += int(np.count_nonzero(np.abs(output) >= _CLIP_LEVEL))
        if observe is not None:
            observe(frame_index, near_block, far_block, output)
        outputs.append(output)
    combined = np.concatenate(outputs)
    if combined.size != near.size + padding:
        raise MicrosoftAecApmDtdEvalError()
    return combined[: near.size].copy(), padding, clipped


def _partial_dtd_engine(config: SherpaConfig, processor: object) -> SherpaOnnxEngine:
    detector = EchoCoherenceDetector(
        config.sample_rate,
        voiced_band=tuple(config.coherence_voiced_band_hz),
        ring_ms=config.coherence_ring_ms,
        max_delay_ms=config.coherence_max_delay_ms,
        nperseg=config.coherence_nperseg,
        margin_delta=config.coherence_margin_delta,
        confirm_frames=config.coherence_confirm_frames,
        warmup_frames=config.coherence_warmup_frames,
        sigma_k=config.coherence_sigma_k,
        baseline_alpha=config.coherence_baseline_alpha,
        var_alpha=config.coherence_var_alpha,
        provisional_baseline=config.coherence_provisional_baseline,
    )
    if detector.available is not True:
        raise MicrosoftAecApmDtdEvalError()
    dtd = AdaptiveDTD(
        k=config.dtd_k,
        weights=(
            config.dtd_weight_raw,
            config.dtd_weight_resid,
            config.dtd_weight_coh,
        ),
        confirm_frames=config.dtd_confirm_frames,
        warmup_frames=config.dtd_warmup_frames,
        chart_rel_floor=config.dtd_chart_rel_floor,
        chart_z_freeze=config.dtd_chart_z_freeze,
        chart_robust_seed=config.dtd_chart_robust_seed,
        chart_freeze_limit=config.dtd_chart_freeze_limit,
        persistent_charts=config.dtd_chart_persist,
    )
    engine = SherpaOnnxEngine.__new__(SherpaOnnxEngine)
    engine.config = config
    engine._aec = processor
    engine._echo_coherence = detector
    engine._dtd = dtd
    engine._playback_floor_rms = 0.0
    engine._raw_playback_floor_rms = 0.0
    engine._apm_owns_ns = bool(
        getattr(processor, "always_on", False)
        and getattr(processor, "suppresses_noise", False)
    )
    engine._resid_blind = bool(getattr(processor, "suppresses_nearend", False))
    engine._aec_ref_delay = 0
    return engine


def _gain_db(estimate: np.ndarray, target: np.ndarray) -> float:
    gain = projection_gain(estimate, target)
    magnitude = abs(gain)
    if not math.isfinite(magnitude):
        raise MicrosoftAecApmDtdEvalError()
    if magnitude == 0.0:
        return -120.0
    return min(120.0, max(-120.0, 20.0 * math.log10(magnitude)))


def _absolute_cosine(estimate: np.ndarray, target: np.ndarray) -> float:
    if signal_power(estimate) == 0.0:
        if signal_power(target) == 0.0:
            raise MicrosoftAecApmDtdEvalError()
        return 0.0
    return absolute_cosine_similarity(estimate, target)


def _si_sdr(estimate: np.ndarray, target: np.ndarray) -> float:
    if signal_power(estimate) == 0.0:
        if signal_power(target) == 0.0:
            raise MicrosoftAecApmDtdEvalError()
        return -120.0
    return si_sdr_db(estimate, target)


def _activity_masks(target: np.ndarray) -> tuple[object, np.ndarray]:
    target_padded, _padding = _padded(target)
    try:
        oracle = oracle_activity_mask(target_padded)
    except ValueError:
        raise MicrosoftAecApmDtdEvalError() from None
    sample_mask = np.repeat(
        np.asarray(oracle.active_mask, dtype=np.bool_),
        BLOCK_SAMPLES,
    )[: target.size]
    if (
        sample_mask.shape != target.shape
        or sample_mask.dtype != np.bool_
        or not bool(np.any(sample_mask))
        or signal_power(target[sample_mask]) == 0.0
    ):
        raise MicrosoftAecApmDtdEvalError()
    sample_mask.setflags(write=False)
    return oracle, sample_mask


def _double_talk_metrics(
    processed: np.ndarray,
    mixture: np.ndarray,
    target: np.ndarray,
    active_samples: np.ndarray,
) -> tuple[float, float, float, float]:
    if (
        not isinstance(active_samples, np.ndarray)
        or active_samples.dtype != np.bool_
        or active_samples.ndim != 1
        or active_samples.shape != target.shape
        or not bool(np.any(active_samples))
    ):
        raise MicrosoftAecApmDtdEvalError()
    try:
        scored_output = np.asarray(processed[active_samples], dtype=np.float64)
        scored_mixture = np.asarray(mixture[active_samples], dtype=np.float64)
        scored_target = np.asarray(target[active_samples], dtype=np.float64)
        return (
            _gain_db(scored_output, scored_target),
            _absolute_cosine(scored_output, scored_target),
            _si_sdr(scored_output, scored_target),
            interference_reduction_db(
                scored_mixture,
                scored_output,
                scored_target,
            ),
        )
    except (MicrosoftAecApmDtdEvalError, ValueError):
        raise MicrosoftAecApmDtdEvalError() from None


def _observe_oracle_inactive(
    engine: SherpaOnnxEngine,
    processed: np.ndarray,
    raw: np.ndarray,
) -> None:
    detector = engine._echo_coherence
    dtd = engine._dtd
    if detector is None or dtd is None:
        raise MicrosoftAecApmDtdEvalError()
    detector.decide(raw)
    incoherent = float(detector.last_incoherent_fraction)
    residual = float(engine._dtd_residual_level(processed, raw))
    raw_level = signal_rms(raw)
    if not all(math.isfinite(value) for value in (incoherent, residual, raw_level)):
        raise MicrosoftAecApmDtdEvalError()
    dtd.observe_echo(raw_level, residual, incoherent)


def _summary(values: Sequence[float]) -> dict[str, int | float]:
    try:
        return percentile_summary(values).as_dict()
    except ValueError:
        raise MicrosoftAecApmDtdEvalError() from None


def _optional_summary(
    values: Sequence[float],
) -> dict[str, int | float | None]:
    if values:
        return _summary(values)
    return {"count": 0, "min": None, "p50": None, "p95": None, "max": None}


def _evaluate_case(
    case: _CaseAudio,
    config: SherpaConfig,
    factory: Callable[[SherpaConfig], object],
    totals: _ReplayTotals,
    *,
    production_geometry: bool,
    require_production_apm: bool,
) -> None:
    tracks = case.tracks
    echo = tracks["echo_signal"]
    far = tracks["farend_speech"]
    mixture = tracks["nearend_mic_signal"]
    target = _scaled_near(case)
    if len({echo.size, far.size, mixture.size, target.size}) != 1:
        raise MicrosoftAecApmDtdEvalError()
    padded_samples = echo.size + _terminal_padding(int(echo.size))
    if production_geometry and padded_samples != ECHO_PREFIX_SAMPLES:
        raise MicrosoftAecApmDtdEvalError()

    near_processor = _processor(
        factory,
        config,
        require_production_apm=require_production_apm,
    )
    try:
        near_output, near_padding, near_clipped = _process_phase(
            near_processor,
            target,
            np.zeros_like(target),
        )
    finally:
        _retire_processor(near_processor)

    echo_processor = _processor(
        factory,
        config,
        require_production_apm=require_production_apm,
    )
    if echo_processor is near_processor:
        _retire_processor(echo_processor)
        raise MicrosoftAecApmDtdEvalError()
    try:
        engine = _partial_dtd_engine(config, echo_processor)
        sustain = BargeSustain(
            window_sec=config.barge_in_sustain_window_sec,
            block_sec=config.block_sec,
            min_voiced_sec=config.barge_in_min_speech_sec,
        )
    except BaseException:
        _retire_processor(echo_processor)
        raise

    echo_false_cut = False

    def observe_echo(
        _frame_index: int,
        raw: np.ndarray,
        reference: np.ndarray,
        processed: np.ndarray,
    ) -> None:
        nonlocal echo_false_cut
        engine._echo_coherence.note_playback(reference, SAMPLE_RATE_HZ)
        engine._update_playback_floor(signal_rms(processed))
        engine._update_raw_playback_floor(signal_rms(raw))
        eligible = engine._looks_like_user(processed, raw)
        score = float(engine._dtd.last_D)
        if not math.isfinite(score):
            raise MicrosoftAecApmDtdEvalError()
        totals.d_score_echo.append(score)
        totals.echo_prefix_frames += 1
        if sustain.update(eligible):
            echo_false_cut = True
            sustain.reset()

    try:
        echo_output, echo_padding, echo_clipped = _process_phase(
            echo_processor,
            echo,
            far,
            observe=observe_echo,
        )

        oracle, active_samples = _activity_masks(target)
        first_active = next(
            (index for index, active in enumerate(oracle.active_mask) if active),
            None,
        )
        if first_active is None:
            raise MicrosoftAecApmDtdEvalError()
        first_cut_latency: float | None = None

        def observe_double(
            frame_index: int,
            raw: np.ndarray,
            reference: np.ndarray,
            processed: np.ndarray,
        ) -> None:
            nonlocal first_cut_latency
            engine._echo_coherence.note_playback(reference, SAMPLE_RATE_HZ)
            engine._update_playback_floor(signal_rms(processed))
            engine._update_raw_playback_floor(signal_rms(raw))
            active = oracle.active_mask[frame_index]
            if not active:
                _observe_oracle_inactive(engine, processed, raw)
                sustain.update(False)
                return
            eligible = engine._looks_like_user(processed, raw)
            totals.eligible_active_frames += 1
            totals.detected_active_frames += int(eligible)
            score = float(engine._dtd.last_D)
            if not math.isfinite(score):
                raise MicrosoftAecApmDtdEvalError()
            totals.d_score_double.append(score)
            if sustain.update(eligible):
                sustain.reset()
                if first_cut_latency is None:
                    first_cut_latency = float(
                        (frame_index - first_active + 1)
                        * BLOCK_SAMPLES
                        * 1000
                        / SAMPLE_RATE_HZ
                    )

        double_output, double_padding, double_clipped = _process_phase(
            echo_processor,
            mixture,
            far,
            observe=observe_double,
        )
        if first_cut_latency is not None:
            totals.detected_cases += 1
            totals.cut_latency_ms.append(first_cut_latency)
        totals.echo_false_cut_cases += int(echo_false_cut)
    finally:
        _retire_processor(echo_processor)

    try:
        totals.near_projection_db.append(_gain_db(near_output, target))
        totals.near_cosine.append(_absolute_cosine(near_output, target))
        totals.near_si_sdr.append(_si_sdr(near_output, target))
        totals.echo_erle.append(
            echo_attenuation_db(
                echo,
                echo_output,
                convergence_samples=CONVERGENCE_SAMPLES,
            )
        )
        echo_tail = np.asarray(echo[CONVERGENCE_SAMPLES:], dtype=np.float64)
        residual_tail = np.asarray(echo_output[CONVERGENCE_SAMPLES:], dtype=np.float64)
        if echo_tail.size == 0 or residual_tail.size != echo_tail.size:
            raise MicrosoftAecApmDtdEvalError()
        totals.echo_energy += signal_power(echo_tail) * echo_tail.size
        totals.residual_energy += signal_power(residual_tail) * residual_tail.size
        double_metrics = _double_talk_metrics(
            double_output,
            mixture,
            target,
            active_samples,
        )
        totals.double_projection_db.append(double_metrics[0])
        totals.double_cosine.append(double_metrics[1])
        totals.double_si_sdr.append(double_metrics[2])
        totals.double_interference.append(double_metrics[3])
    except (MicrosoftAecApmDtdEvalError, ValueError):
        raise MicrosoftAecApmDtdEvalError() from None

    source = int(target.size + echo.size + mixture.size)
    padding = int(near_padding + echo_padding + double_padding)
    totals.source_samples += source
    totals.padding_samples += padding
    totals.processed_samples += source + padding
    totals.clipped_samples += near_clipped + echo_clipped + double_clipped


def _configuration_report(snapshot: _ConfigurationSnapshot) -> dict[str, object]:
    return {
        "profile": "open_speaker",
        "sherpa_sha256": snapshot.sherpa_sha256,
        "source_sha256": snapshot.source_sha256,
        "values": dict(snapshot.values),
    }


def _bound_coverage(bundle: _BundleSnapshot) -> dict[str, int | bool]:
    source_samples = 0
    padding_samples = 0
    for case in bundle.bundle.cases:
        sample_counts = {artifact.samples for artifact in case.artifacts}
        if (
            tuple(artifact.role for artifact in case.artifacts) != SIGNAL_ROLES
            or len(sample_counts) != 1
        ):
            raise MicrosoftAecApmDtdEvalError()
        samples = next(iter(sample_counts))
        source_samples += samples * 3
        padding_samples += _terminal_padding(samples) * 3
    cases = len(bundle.bundle.cases)
    return {
        "cases": cases,
        "dropped_samples": 0,
        "nonfinite_samples": 0,
        "padding_samples": padding_samples,
        "phase_evaluations": cases * 3,
        "processed_samples": source_samples + padding_samples,
        "scored_samples": source_samples,
        "source_complete": True,
        "source_samples": source_samples,
        "tracks": cases * len(SIGNAL_ROLES),
    }


def _build_report(
    bundle: _BundleSnapshot,
    closure: _ClosureSnapshot,
    configuration: _ConfigurationSnapshot,
    runtime: _RuntimeSnapshot,
    totals: _ReplayTotals,
    *,
    injected: bool,
) -> dict[str, object]:
    case_count = len(bundle.bundle.cases)
    missed_cases = case_count - totals.detected_cases
    report: dict[str, object] = {
        "binding_sha256": "",
        "coverage": {
            "cases": case_count,
            "clipped_samples": totals.clipped_samples,
            "dropped_samples": 0,
            "nonfinite_samples": 0,
            "padding_samples": totals.padding_samples,
            "phase_evaluations": case_count * 3,
            "processed_samples": totals.processed_samples,
            "scored_samples": totals.source_samples,
            "source_complete": True,
            "source_samples": totals.source_samples,
            "tracks": case_count * len(SIGNAL_ROLES),
        },
        "double_talk": {
            "absolute_cosine": _summary(totals.double_cosine),
            "interference_reduction_db": _summary(totals.double_interference),
            "projection_gain_db": _summary(totals.double_projection_db),
            "si_sdr_db": _summary(totals.double_si_sdr),
        },
        "dtd": {
            "d_score_double_talk": _summary(totals.d_score_double),
            "d_score_echo": _summary(totals.d_score_echo),
            "detected_active_frames": totals.detected_active_frames,
            "detected_cases": totals.detected_cases,
            "echo_false_cut_cases": totals.echo_false_cut_cases,
            "echo_false_cut_rate": totals.echo_false_cut_cases / case_count,
            "echo_prefix_frames": totals.echo_prefix_frames,
            "eligible_active_frames": totals.eligible_active_frames,
            "first_cut_latency_ms": _optional_summary(totals.cut_latency_ms),
            "missed_cases": missed_cases,
            "recall": totals.detected_cases / case_count,
        },
        "echo_only": {
            "convergence_samples": CONVERGENCE_SAMPLES,
            "energy_pooled_erle_db": capped_db_ratio(
                totals.echo_energy,
                totals.residual_energy,
            ),
            "erle_db": _summary(totals.echo_erle),
        },
        "evaluator": {
            "closure_sha256": closure.sha256,
            "file_count": len(closure.files),
            "total_bytes": closure.total_bytes,
        },
        "evidence": {
            "apm_component_replay": not injected,
            "conversation_latency": False,
            "default_change": False,
            "device": False,
            "endpoint": False,
            "identity": False,
            "live_capture": False,
            "oracle_gated_dtd_component_replay": True,
            "promotion": False,
            "receipt_bound_official_data": bundle.bundle.production_evidence,
            "stt": False,
            "tool": False,
            "vad": False,
        },
        "fixture": {
            "case_count": case_count,
            "lock_recipe_sha256": bundle.bundle.lock_recipe_sha256,
            "manifest_sha256": bundle.bundle.manifest_sha256,
            "production_evidence": bundle.bundle.production_evidence,
            "receipt_sha256": bundle.bundle.receipt_sha256,
            "source_contract_sha256": bundle.bundle.source_contract_sha256,
            "track_count": case_count * len(SIGNAL_ROLES),
        },
        "kind": REPORT_KIND,
        "near_retention": {
            "absolute_cosine": _summary(totals.near_cosine),
            "projection_gain_db": _summary(totals.near_projection_db),
            "si_sdr_db": _summary(totals.near_si_sdr),
        },
        "protocol": {
            "activity_oracle": "fixed-100ms-near-energy-v1",
            "block_samples": BLOCK_SAMPLES,
            "case_execution": "sequential-one-at-a-time",
            "convergence_samples": CONVERGENCE_SAMPLES,
            "echo_prefix_samples_per_case": ECHO_PREFIX_SAMPLES,
            "near_target": "nearend-scale-f32-v1",
            "phase_order": [
                "near-only-fresh",
                "echo-prefix-fresh",
                "double-talk-same-state",
            ],
            "processor_instances_per_case": 2,
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "scoring": "original-samples-only",
            "terminal_padding": "explicit-zero",
            "terminal_padding_max_samples": MAX_TERMINAL_PADDING_SAMPLES,
        },
        "runtime": {
            **dict(runtime.binding),
            "configuration": _configuration_report(configuration),
        },
        "schema_version": SCHEMA_VERSION,
    }
    unsigned = dict(report)
    unsigned.pop("binding_sha256")
    report["binding_sha256"] = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
    return report


def _mapping(value: object, fields: set[str]) -> Mapping[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise MicrosoftAecApmDtdEvalError()
    return value


def _integer(value: object, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise MicrosoftAecApmDtdEvalError()
    return value


def _finite(value: object) -> float:
    if type(value) not in {int, float} or isinstance(value, bool):
        raise MicrosoftAecApmDtdEvalError()
    result = float(value)
    if not math.isfinite(result):
        raise MicrosoftAecApmDtdEvalError()
    return result


def _sha256(value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MicrosoftAecApmDtdEvalError()
    return value


def _validate_summary(
    value: object,
    *,
    count: int,
    allow_empty: bool = False,
) -> Mapping[str, object]:
    summary = _mapping(value, {"count", "min", "p50", "p95", "max"})
    if summary["count"] != count:
        raise MicrosoftAecApmDtdEvalError()
    if count == 0:
        if not allow_empty or any(
            summary[key] is not None for key in ("min", "p50", "p95", "max")
        ):
            raise MicrosoftAecApmDtdEvalError()
        return summary
    ordered = tuple(_finite(summary[key]) for key in ("min", "p50", "p95", "max"))
    if ordered != tuple(sorted(ordered)):
        raise MicrosoftAecApmDtdEvalError()
    return summary


def _validate_report(
    value: object,
    *,
    bundle: _BundleSnapshot | None = None,
    closure: _ClosureSnapshot | None = None,
    configuration: _ConfigurationSnapshot | None = None,
    runtime: _RuntimeSnapshot | None = None,
) -> dict[str, object]:
    root = _mapping(
        value,
        {
            "binding_sha256",
            "coverage",
            "double_talk",
            "dtd",
            "echo_only",
            "evaluator",
            "evidence",
            "fixture",
            "kind",
            "near_retention",
            "protocol",
            "runtime",
            "schema_version",
        },
    )
    if root["schema_version"] != SCHEMA_VERSION or root["kind"] != REPORT_KIND:
        raise MicrosoftAecApmDtdEvalError()
    coverage = _mapping(
        root["coverage"],
        {
            "cases",
            "clipped_samples",
            "dropped_samples",
            "nonfinite_samples",
            "padding_samples",
            "phase_evaluations",
            "processed_samples",
            "scored_samples",
            "source_complete",
            "source_samples",
            "tracks",
        },
    )
    cases = _integer(coverage["cases"], minimum=1)
    tracks = _integer(coverage["tracks"], minimum=1)
    source_samples = _integer(coverage["source_samples"], minimum=1)
    padding = _integer(coverage["padding_samples"])
    processed = _integer(coverage["processed_samples"], minimum=1)
    clipped = _integer(coverage["clipped_samples"])
    if (
        tracks != cases * len(SIGNAL_ROLES)
        or coverage["phase_evaluations"] != cases * 3
        or coverage["scored_samples"] != source_samples
        or processed != source_samples + padding
        or coverage["dropped_samples"] != 0
        or coverage["nonfinite_samples"] != 0
        or coverage["source_complete"] is not True
        or clipped > processed
        or padding > cases * 3 * MAX_TERMINAL_PADDING_SAMPLES
    ):
        raise MicrosoftAecApmDtdEvalError()
    if bundle is not None:
        expected_coverage = _bound_coverage(bundle)
        if any(
            coverage[key] != expected for key, expected in expected_coverage.items()
        ):
            raise MicrosoftAecApmDtdEvalError()

    fixture_value = _mapping(
        root["fixture"],
        {
            "case_count",
            "lock_recipe_sha256",
            "manifest_sha256",
            "production_evidence",
            "receipt_sha256",
            "source_contract_sha256",
            "track_count",
        },
    )
    if (
        fixture_value["case_count"] != cases
        or fixture_value["track_count"] != tracks
        or type(fixture_value["production_evidence"]) is not bool
    ):
        raise MicrosoftAecApmDtdEvalError()
    for key in (
        "lock_recipe_sha256",
        "manifest_sha256",
        "receipt_sha256",
        "source_contract_sha256",
    ):
        _sha256(fixture_value[key])
    if bundle is not None and fixture_value != {
        "case_count": len(bundle.bundle.cases),
        "lock_recipe_sha256": bundle.bundle.lock_recipe_sha256,
        "manifest_sha256": bundle.bundle.manifest_sha256,
        "production_evidence": bundle.bundle.production_evidence,
        "receipt_sha256": bundle.bundle.receipt_sha256,
        "source_contract_sha256": bundle.bundle.source_contract_sha256,
        "track_count": len(bundle.bundle.cases) * len(SIGNAL_ROLES),
    }:
        raise MicrosoftAecApmDtdEvalError()
    if fixture_value["production_evidence"] and (
        cases != PRODUCTION_CASES
        or tracks != 128
        or coverage["phase_evaluations"] != 96
    ):
        raise MicrosoftAecApmDtdEvalError()

    protocol = _mapping(
        root["protocol"],
        {
            "activity_oracle",
            "block_samples",
            "case_execution",
            "convergence_samples",
            "echo_prefix_samples_per_case",
            "near_target",
            "phase_order",
            "processor_instances_per_case",
            "sample_rate_hz",
            "scoring",
            "terminal_padding",
            "terminal_padding_max_samples",
        },
    )
    if protocol != {
        "activity_oracle": "fixed-100ms-near-energy-v1",
        "block_samples": BLOCK_SAMPLES,
        "case_execution": "sequential-one-at-a-time",
        "convergence_samples": CONVERGENCE_SAMPLES,
        "echo_prefix_samples_per_case": ECHO_PREFIX_SAMPLES,
        "near_target": "nearend-scale-f32-v1",
        "phase_order": [
            "near-only-fresh",
            "echo-prefix-fresh",
            "double-talk-same-state",
        ],
        "processor_instances_per_case": 2,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "scoring": "original-samples-only",
        "terminal_padding": "explicit-zero",
        "terminal_padding_max_samples": MAX_TERMINAL_PADDING_SAMPLES,
    }:
        raise MicrosoftAecApmDtdEvalError()

    near = _mapping(
        root["near_retention"],
        {"absolute_cosine", "projection_gain_db", "si_sdr_db"},
    )
    double = _mapping(
        root["double_talk"],
        {
            "absolute_cosine",
            "interference_reduction_db",
            "projection_gain_db",
            "si_sdr_db",
        },
    )
    for summary in near.values():
        _validate_summary(summary, count=cases)
    for summary in double.values():
        _validate_summary(summary, count=cases)
    echo = _mapping(
        root["echo_only"],
        {"convergence_samples", "energy_pooled_erle_db", "erle_db"},
    )
    if echo["convergence_samples"] != CONVERGENCE_SAMPLES:
        raise MicrosoftAecApmDtdEvalError()
    _finite(echo["energy_pooled_erle_db"])
    _validate_summary(echo["erle_db"], count=cases)

    dtd = _mapping(
        root["dtd"],
        {
            "d_score_double_talk",
            "d_score_echo",
            "detected_active_frames",
            "detected_cases",
            "echo_false_cut_cases",
            "echo_false_cut_rate",
            "echo_prefix_frames",
            "eligible_active_frames",
            "first_cut_latency_ms",
            "missed_cases",
            "recall",
        },
    )
    echo_frames = _integer(dtd["echo_prefix_frames"], minimum=1)
    false_cuts = _integer(dtd["echo_false_cut_cases"])
    active = _integer(dtd["eligible_active_frames"], minimum=1)
    detected_frames = _integer(dtd["detected_active_frames"])
    detected_cases = _integer(dtd["detected_cases"])
    missed_cases = _integer(dtd["missed_cases"])
    if (
        echo_frames != cases * (ECHO_PREFIX_SAMPLES // BLOCK_SAMPLES)
        or false_cuts > cases
        or detected_frames > active
        or detected_cases + missed_cases != cases
        or detected_cases > cases
        or _finite(dtd["echo_false_cut_rate"]) != false_cuts / cases
        or _finite(dtd["recall"]) != detected_cases / cases
    ):
        raise MicrosoftAecApmDtdEvalError()
    _validate_summary(dtd["d_score_echo"], count=echo_frames)
    _validate_summary(dtd["d_score_double_talk"], count=active)
    _validate_summary(
        dtd["first_cut_latency_ms"],
        count=detected_cases,
        allow_empty=True,
    )

    evaluator = _mapping(
        root["evaluator"],
        {"closure_sha256", "file_count", "total_bytes"},
    )
    _sha256(evaluator["closure_sha256"])
    _integer(evaluator["file_count"], minimum=1)
    _integer(evaluator["total_bytes"], minimum=1)
    if closure is not None and evaluator != {
        "closure_sha256": closure.sha256,
        "file_count": len(closure.files),
        "total_bytes": closure.total_bytes,
    }:
        raise MicrosoftAecApmDtdEvalError()

    evidence = _mapping(
        root["evidence"],
        {
            "apm_component_replay",
            "conversation_latency",
            "default_change",
            "device",
            "endpoint",
            "identity",
            "live_capture",
            "oracle_gated_dtd_component_replay",
            "promotion",
            "receipt_bound_official_data",
            "stt",
            "tool",
            "vad",
        },
    )
    if any(type(item) is not bool for item in evidence.values()):
        raise MicrosoftAecApmDtdEvalError()
    if (
        evidence["receipt_bound_official_data"]
        is not fixture_value["production_evidence"]
        or evidence["oracle_gated_dtd_component_replay"] is not True
        or any(
            evidence[key]
            for key in (
                "conversation_latency",
                "default_change",
                "device",
                "endpoint",
                "identity",
                "live_capture",
                "promotion",
                "stt",
                "tool",
                "vad",
            )
        )
    ):
        raise MicrosoftAecApmDtdEvalError()

    runtime_value = _mapping(
        root["runtime"],
        {
            "configuration",
            "configuration_profile",
            "cpu_only",
            "implementation",
            "livekit",
            "production_evidence",
        },
    )
    if (
        runtime_value["configuration_profile"] != "open_speaker"
        or runtime_value["cpu_only"] is not True
        or type(runtime_value["production_evidence"]) is not bool
        or evidence["apm_component_replay"] is not runtime_value["production_evidence"]
    ):
        raise MicrosoftAecApmDtdEvalError()
    configuration_value = _mapping(
        runtime_value["configuration"],
        {"profile", "sherpa_sha256", "source_sha256", "values"},
    )
    if configuration_value["profile"] != "open_speaker" or set(
        _mapping(configuration_value["values"], set(_CONFIG_KEYS))
    ) != set(_CONFIG_KEYS):
        raise MicrosoftAecApmDtdEvalError()
    sherpa_sha256 = _sha256(configuration_value["sherpa_sha256"])
    _sha256(configuration_value["source_sha256"])
    if (
        hashlib.sha256(_canonical_json(configuration_value["values"])).hexdigest()
        != sherpa_sha256
    ):
        raise MicrosoftAecApmDtdEvalError()
    if configuration is not None and configuration_value != _configuration_report(
        configuration
    ):
        raise MicrosoftAecApmDtdEvalError()
    if runtime_value["production_evidence"]:
        if runtime_value["implementation"] != "livekit.rtc.AudioProcessingModule":
            raise MicrosoftAecApmDtdEvalError()
        livekit = _mapping(
            runtime_value["livekit"],
            {
                "content_set_sha256",
                "distribution",
                "file_count",
                "maximum_file_bytes",
                "record_sha256",
                "record_size_bytes",
                "total_bytes",
                "version",
            },
        )
        if (
            livekit["distribution"] != "livekit"
            or type(livekit["version"]) is not str
            or _SAFE_LABEL_RE.fullmatch(livekit["version"]) is None
        ):
            raise MicrosoftAecApmDtdEvalError()
        _sha256(livekit["content_set_sha256"])
        _sha256(livekit["record_sha256"])
        _integer(livekit["record_size_bytes"], minimum=1)
        file_count = _integer(livekit["file_count"], minimum=1)
        total_bytes = _integer(livekit["total_bytes"], minimum=1)
        maximum_file_bytes = _integer(livekit["maximum_file_bytes"], minimum=1)
        if (
            file_count > _LIVEKIT_FILE_MAX_COUNT
            or total_bytes > _LIVEKIT_TOTAL_MAX_BYTES
            or maximum_file_bytes > _LIVEKIT_FILE_MAX_BYTES
            or maximum_file_bytes > total_bytes
        ):
            raise MicrosoftAecApmDtdEvalError()
    elif (
        runtime_value["implementation"] != "injected-test-double"
        or runtime_value["livekit"] is not None
    ):
        raise MicrosoftAecApmDtdEvalError()
    if runtime is not None:
        retained_runtime = dict(runtime_value)
        retained_runtime.pop("configuration")
        if retained_runtime != dict(runtime.binding):
            raise MicrosoftAecApmDtdEvalError()

    unsigned = dict(root)
    binding = _sha256(unsigned.pop("binding_sha256"))
    if hashlib.sha256(_canonical_json(unsigned)).hexdigest() != binding:
        raise MicrosoftAecApmDtdEvalError()
    encoded = _canonical_json(root)
    if len(encoded) > _MAX_REPORT_BYTES:
        raise MicrosoftAecApmDtdEvalError()
    if bundle is not None:
        forbidden = {
            str(bundle.bundle.root),
            bundle.bundle.fixture_id,
            *(case.case_id for case in bundle.bundle.cases),
            *(case.rank_sha256 for case in bundle.bundle.cases),
            *(
                artifact.name
                for case in bundle.bundle.cases
                for artifact in case.artifacts
            ),
        }
        lowered = encoded.decode("ascii").casefold()
        if any(text and text.casefold() in lowered for text in forbidden):
            raise MicrosoftAecApmDtdEvalError()
    return dict(root)


def _private_output_parent(path: Path) -> Path:
    candidate = Path(os.path.abspath(path.expanduser()))
    try:
        if candidate != path.expanduser() or candidate.name in {"", ".", ".."}:
            raise MicrosoftAecApmDtdEvalError()
        parent = candidate.parent
        with opened_directory_nofollow(parent, require_private=True) as (
            stable,
            descriptor,
        ):
            metadata = os.fstat(descriptor)
            if (
                stable != parent
                or stat.S_IMODE(metadata.st_mode) != 0o700
                or metadata.st_uid != os.getuid()
            ):
                raise MicrosoftAecApmDtdEvalError()
        for ancestor in (parent, *parent.parents):
            marker = ancestor / ".git"
            try:
                marker_metadata = os.lstat(marker)
            except FileNotFoundError:
                continue
            if not stat.S_ISDIR(marker_metadata.st_mode):
                raise MicrosoftAecApmDtdEvalError()
            try:
                head_metadata = os.lstat(marker / "HEAD")
            except FileNotFoundError:
                continue
            if stat.S_ISREG(head_metadata.st_mode):
                raise MicrosoftAecApmDtdEvalError()
            raise MicrosoftAecApmDtdEvalError()
        return candidate
    except MicrosoftAecApmDtdEvalError:
        raise
    except (BoundedReadError, OSError, RuntimeError, ValueError):
        raise MicrosoftAecApmDtdEvalError() from None


def _validate_output_destination(
    path: Path,
    bundle: _BundleSnapshot,
) -> Path:
    output = _private_output_parent(path)
    try:
        bundle_root = bundle.bundle.root.resolve(strict=True)
    except (OSError, RuntimeError):
        raise MicrosoftAecApmDtdEvalError() from None
    if output == bundle_root or bundle_root in output.parents:
        raise MicrosoftAecApmDtdEvalError()
    return output


def _rebind_output_parent(parent: Path, retained: os.stat_result) -> None:
    try:
        with opened_directory_nofollow(parent, require_private=True) as (
            stable,
            descriptor,
        ):
            current = os.fstat(descriptor)
            if (
                stable != parent
                or (current.st_dev, current.st_ino)
                != (retained.st_dev, retained.st_ino)
                or not stat.S_ISDIR(current.st_mode)
                or current.st_uid != os.getuid()
                or stat.S_IMODE(current.st_mode) != 0o700
            ):
                raise MicrosoftAecApmDtdEvalError()
    except MicrosoftAecApmDtdEvalError:
        raise
    except (BoundedReadError, OSError, RuntimeError, ValueError):
        raise MicrosoftAecApmDtdEvalError() from None


def _verify_staged_report(
    descriptor: int,
    encoded: bytes,
    digest: str,
) -> None:
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        observed = bytearray()
        while len(observed) <= len(encoded):
            chunk = os.read(descriptor, len(encoded) + 1 - len(observed))
            if not chunk:
                break
            observed.extend(chunk)
        metadata = os.fstat(descriptor)
        if (
            bytes(observed) != encoded
            or hashlib.sha256(observed).hexdigest() != digest
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size != len(encoded)
            or metadata.st_uid != os.getuid()
            or metadata.st_nlink != 0
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise MicrosoftAecApmDtdEvalError()
    except MicrosoftAecApmDtdEvalError:
        raise
    except (OSError, OverflowError, TypeError, ValueError):
        raise MicrosoftAecApmDtdEvalError() from None


def _commit_signal_numbers() -> tuple[int, ...]:
    return tuple(
        signum
        for signum in dict.fromkeys(
            (
                getattr(signal, "SIGINT", None),
                getattr(signal, "SIGHUP", None),
                getattr(signal, "SIGTERM", None),
            )
        )
        if isinstance(signum, int)
    )


def _publish_report(
    path: Path,
    report: Mapping[str, object],
    *,
    commit_guard: Callable[[], None],
    state: _ReportCommitState,
) -> str:
    output = _private_output_parent(path)
    encoded = _canonical_json(report, newline=True)
    digest = hashlib.sha256(encoded).hexdigest()
    descriptor = -1
    try:
        if state.committed or state.digest:
            raise MicrosoftAecApmDtdEvalError()
        with opened_directory_nofollow(output.parent, require_private=True) as (
            stable_parent,
            directory_fd,
        ):
            if stable_parent != output.parent:
                raise MicrosoftAecApmDtdEvalError()
            parent_before = os.fstat(directory_fd)
            if stat.S_IMODE(parent_before.st_mode) != 0o700:
                raise MicrosoftAecApmDtdEvalError()
            try:
                os.stat(output.name, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise MicrosoftAecApmDtdEvalError()
            temporary = getattr(os, "O_TMPFILE", 0)
            if not temporary or not hasattr(signal, "pthread_sigmask"):
                raise MicrosoftAecApmDtdEvalError()
            descriptor = os.open(
                ".",
                os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | temporary,
                0o600,
                dir_fd=directory_fd,
            )
            os.fchmod(descriptor, 0o600)
            created = os.fstat(descriptor)
            if (
                not stat.S_ISREG(created.st_mode)
                or created.st_uid != os.getuid()
                or created.st_nlink != 0
                or stat.S_IMODE(created.st_mode) != 0o600
            ):
                raise MicrosoftAecApmDtdEvalError()
            view = memoryview(encoded)
            written = 0
            while written < len(view):
                count = os.write(descriptor, view[written:])
                if type(count) is not int or count <= 0:
                    raise MicrosoftAecApmDtdEvalError()
                written += count
            os.fsync(descriptor)
            _verify_staged_report(descriptor, encoded, digest)
            commit_guard()
            parent_now = os.fstat(directory_fd)
            if (parent_now.st_dev, parent_now.st_ino) != (
                parent_before.st_dev,
                parent_before.st_ino,
            ) or (
                parent_now.st_uid != os.getuid()
                or stat.S_IMODE(parent_now.st_mode) != 0o700
            ):
                raise MicrosoftAecApmDtdEvalError()
            _rebind_output_parent(output.parent, parent_before)
            _verify_staged_report(descriptor, encoded, digest)
            previous_mask = signal.pthread_sigmask(
                signal.SIG_BLOCK,
                _commit_signal_numbers(),
            )
            link_returned = False
            try:
                try:
                    os.link(
                        f"/proc/self/fd/{descriptor}",
                        output.name,
                        dst_dir_fd=directory_fd,
                        follow_symlinks=True,
                    )
                    link_returned = True
                finally:
                    if link_returned:
                        state.committed = True
                        state.digest = digest
                    else:
                        try:
                            published = os.stat(
                                output.name,
                                dir_fd=directory_fd,
                                follow_symlinks=False,
                            )
                            current = os.fstat(descriptor)
                        except OSError:
                            pass
                        else:
                            if (
                                (published.st_dev, published.st_ino)
                                == (current.st_dev, current.st_ino)
                                and published.st_nlink == 1
                                and published.st_uid == os.getuid()
                                and stat.S_IMODE(published.st_mode) == 0o600
                                and published.st_size == len(encoded)
                            ):
                                state.committed = True
                                state.digest = digest
            finally:
                try:
                    if state.committed:
                        os.fsync(directory_fd)
                finally:
                    signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            if not state.committed:
                raise MicrosoftAecApmDtdEvalError()
            published = os.stat(
                output.name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
            current = os.fstat(descriptor)
            if (
                (published.st_dev, published.st_ino) != (current.st_dev, current.st_ino)
                or published.st_nlink != 1
                or published.st_uid != os.getuid()
                or stat.S_IMODE(published.st_mode) != 0o600
                or published.st_size != len(encoded)
            ):
                raise MicrosoftAecApmDtdEvalError()
            return digest
    except BaseException as error:
        if state.committed:
            return state.digest
        if isinstance(error, (KeyboardInterrupt, _LifecycleSignal)):
            raise
        if isinstance(error, MicrosoftAecApmDtdEvalError):
            raise
        raise MicrosoftAecApmDtdEvalError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def run_microsoft_aec_apm_dtd_eval(
    bundle_dir: Path,
    *,
    output_path: Path | None = None,
    run_guard: Callable[[], None] = lambda: None,
    _processor_factory: Callable[[SherpaConfig], object] | None = None,
) -> dict:
    """Run the fixed replay with a private publication-state owner."""

    return _run_microsoft_aec_apm_dtd_eval(
        bundle_dir,
        output_path=output_path,
        run_guard=run_guard,
        _processor_factory=_processor_factory,
        _commit_state=_ReportCommitState(),
    )


def _run_microsoft_aec_apm_dtd_eval(
    bundle_dir: Path,
    *,
    output_path: Path | None,
    run_guard: Callable[[], None],
    _processor_factory: Callable[[SherpaConfig], object] | None,
    _commit_state: _ReportCommitState,
) -> dict:
    """Run the fixed sequential replay and optionally publish one private report."""

    if (
        not isinstance(_commit_state, _ReportCommitState)
        or _commit_state.committed
        or _commit_state.digest
        or not callable(run_guard)
        or (_processor_factory is not None and not callable(_processor_factory))
    ):
        raise MicrosoftAecApmDtdEvalError()
    injected = _processor_factory is not None
    factory = build_aec if _processor_factory is None else _processor_factory
    bundle = _load_bundle_snapshot(bundle_dir)
    output = (
        None
        if output_path is None
        else _validate_output_destination(Path(output_path), bundle)
    )
    closure = _execution_closure()
    configuration = _configuration_snapshot()
    runtime = _runtime_binding(injected=injected)
    _verify_bound_state(
        bundle,
        closure,
        configuration,
        runtime,
        injected=injected,
        run_guard=run_guard,
    )
    totals = _ReplayTotals()
    report: dict[str, object] | None = None
    state: _ReportCommitState | None = _commit_state if output is not None else None
    try:
        for bound_case in bundle.bundle.cases:
            case = _read_case_audio(bundle.bundle, bound_case)
            _evaluate_case(
                case,
                configuration.config,
                factory,
                totals,
                production_geometry=bundle.bundle.production_evidence,
                require_production_apm=not injected,
            )
            del case
        _verify_bound_state(
            bundle,
            closure,
            configuration,
            runtime,
            injected=injected,
            run_guard=run_guard,
        )
        report = _build_report(
            bundle,
            closure,
            configuration,
            runtime,
            totals,
            injected=injected,
        )
        report = _validate_report(
            report,
            bundle=bundle,
            closure=closure,
            configuration=configuration,
            runtime=runtime,
        )
        _verify_bound_state(
            bundle,
            closure,
            configuration,
            runtime,
            injected=injected,
            run_guard=run_guard,
        )
        if output is not None:
            if state is None:
                raise MicrosoftAecApmDtdEvalError()

            def commit_guard() -> None:
                _verify_bound_state(
                    bundle,
                    closure,
                    configuration,
                    runtime,
                    injected=injected,
                    run_guard=run_guard,
                )
                _validate_report(
                    report,
                    bundle=bundle,
                    closure=closure,
                    configuration=configuration,
                    runtime=runtime,
                )
                if _validate_output_destination(output, bundle) != output:
                    raise MicrosoftAecApmDtdEvalError()

            _publish_report(
                output,
                report,
                commit_guard=commit_guard,
                state=state,
            )
        return report
    except BaseException as error:
        if state is not None and state.committed and report is not None:
            return report
        if isinstance(error, (KeyboardInterrupt, _LifecycleSignal)):
            raise
        if isinstance(error, MicrosoftAecApmDtdEvalError):
            raise
        raise MicrosoftAecApmDtdEvalError() from None


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise MicrosoftAecApmDtdEvalError()


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        description="Run the private Microsoft AEC APM/DTD component replay."
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def _lifecycle_handler(signum: int, _frame: object) -> None:
    raise _LifecycleSignal(signum)


def main(argv: Sequence[str] | None = None) -> int:
    previous: dict[int, object] = {}
    commit_state = _ReportCommitState()
    try:
        for signum in (
            getattr(signal, "SIGHUP", None),
            getattr(signal, "SIGTERM", None),
        ):
            if isinstance(signum, int):
                previous[signum] = signal.signal(signum, _lifecycle_handler)
        args = _parser().parse_args(argv)
        report = _run_microsoft_aec_apm_dtd_eval(
            args.bundle,
            output_path=args.output,
            run_guard=lambda: None,
            _processor_factory=None,
            _commit_state=commit_state,
        )
        terminal = (
            report
            if args.output is None
            else {
                "binding_sha256": report["binding_sha256"],
                "kind": REPORT_KIND,
                "ok": True,
            }
        )
        os.write(1, _canonical_json(terminal, newline=True))
        return 0
    except KeyboardInterrupt:
        if commit_state.committed:
            return 0
        return 130
    except _LifecycleSignal as interrupted:
        if commit_state.committed:
            return 0
        return 128 + interrupted.signum
    except BaseException:
        if commit_state.committed:
            return 0
        try:
            os.write(1, _canonical_json(_SAFE_ERROR, newline=True))
        except BaseException:
            pass
        return 2
    finally:
        for signum, handler in reversed(tuple(previous.items())):
            try:
                signal.signal(signum, handler)
            except BaseException:
                pass


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "MicrosoftAecApmDtdEvalError",
    "REPORT_KIND",
    "main",
    "run_microsoft_aec_apm_dtd_eval",
]
