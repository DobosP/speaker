"""Strict aggregate attestation for two guided live-STT capture bundles.

The controller fixes capture roles, profile order, route availability, and
execution geometry.  It invokes no audio device or tool capability.  Private
labels, PCM, and child reports remain under a new private scratch directory;
the only separately published artifact is a path-free aggregate report.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import stat
import subprocess
import sys
from typing import Callable

from core.guided_stt_plan import (
    GUIDED_STT_CAPTURE_PROTOCOL,
    GUIDED_STT_PLAN_ID,
    built_in_guided_stt_plan,
    guided_stt_route_availability_sha256,
)
from core.wer import normalize
from tools import public_conversation_qualification as qualification
from tools.streaming_stt.bounded_io import (
    BoundedReadError,
    opened_directory_nofollow,
    read_regular_bounded,
)
from tools.streaming_stt.corpus import LoadedCorpus, load_corpus, verify_corpus_snapshot
from tools.streaming_stt.protocol import MAX_CORPUS_BYTES
from tools.tool_route_gate import (
    EXPECTED_ROUTES,
    ToolRouteGateProfile,
    ToolRouteGateTotals,
    no_regression as route_no_regression,
    profile_digest as route_profile_digest,
)


SCHEMA_VERSION = 1
KIND = "guided-stt-pair-attestation-v1"
CONTROL_PROFILE = "sense-voice"
CANDIDATE_PROFILE = "parakeet-faster-whisper"
CASES_PER_BUNDLE = 16
PREPARE_TIMEOUT_SEC = 120.0
EVALUATE_TIMEOUT_SEC = 1_800.0
MAX_CHILD_REPORT_BYTES = 4 * 1024 * 1024
MAX_REPORT_BYTES = 16 * 1024 * 1024
MAX_CHILD_FILE_BYTES = 64 * 1024 * 1024
MAX_CHILD_FDS = 256
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SAFE_ERROR = {"error": "guided STT pair attestation failed", "ok": False}
_SHA256_CHARS = frozenset("0123456789abcdef")
_OFFLINE_OUTCOMES = frozenset({"unavailable", "skipped", "error", "decoded", "empty"})
_VERIFIER_OUTCOMES = frozenset(
    {
        "unavailable",
        "skipped",
        "error",
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
_COMPLETED_VERIFIER_OUTCOMES = _VERIFIER_OUTCOMES - {
    "unavailable",
    "skipped",
    "error",
}
_SELECTED_SOURCES = frozenset(
    {"none", "streaming", "offline", "verifier_consensus", "established_override"}
)
_ROUTE_ATTEMPTS = {
    "none": 5,
    "vault.search": 5,
    "web.search": 2,
    "reminder.create": 1,
    "reminder.list": 1,
    "reminder.cancel": 1,
    "app.open": 1,
}
_ACCURACY_INT_FIELDS = (
    "clips",
    "nonempty",
    "exact",
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
)
_ACCURACY_FIELDS = frozenset((*_ACCURACY_INT_FIELDS, "wer", "cer"))
_ROUTE_COUNTER_FIELDS = (
    "annotated_cases",
    "decisions",
    "single_decision_cases",
    "empty_decisions",
    "expected_positive_cases",
    "expected_none_cases",
    "exact_cases",
    "misses",
    "wrong_tool",
    "unexpected_tool",
    "unexpected_control",
    "unexpected_action",
    "multi_decision_cases",
)


class GuidedSttPairAttestationError(RuntimeError):
    """A deliberately detail-free source, execution, or report failure."""


class _LifecycleSignal(BaseException):
    """A process lifecycle signal translated into bounded Python cleanup."""

    def __init__(self, signum: int) -> None:
        super().__init__()
        self.signum = signum


@dataclass(frozen=True, slots=True)
class _BundleState:
    source: object = field(repr=False, compare=False)
    run_dir: Path = field(repr=False)
    plan_path: Path = field(repr=False)
    diagnostic_manifest_path: Path = field(repr=False)
    receipt_sha256: str
    plan_sha256: str
    contract_sha256: str
    summary_sha256: str
    diagnostic_manifest_sha256: str
    profile: str
    profile_sha256: str
    capture_config_sha256: str
    effective_sherpa_sha256: str
    device_profile: str
    effective_input_gain: float
    case_order_sha256: str


@dataclass(frozen=True, slots=True)
class _PreparedState:
    role: str
    labels_path: Path = field(repr=False)
    corpus_path: Path = field(repr=False)
    labels_sha256: str
    corpus_sha256: str
    cases: int
    audio_bytes: int
    receipt_roles: tuple[int, int]
    loaded_corpus: LoadedCorpus = field(repr=False, compare=False)
    labels_identity: tuple[int, int, int, int, int, int] = field(repr=False)
    corpus_identities: tuple[tuple[int, int, int, int, int, int], ...] = field(
        repr=False
    )


@dataclass(frozen=True, slots=True)
class AttestationPublication:
    """The already-published result of one successful attestation."""

    report: Mapping[str, object] = field(repr=False)
    report_sha256: str


@dataclass(slots=True)
class _ReportCommitState:
    """Share the exact terminal commit boundary with the CLI frame."""

    pending_publication: AttestationPublication | None = None
    committed: bool = False


def _default_bundle_loader(path: Path | str) -> object:
    from tools.guided_stt_capture import load_verified_guided_capture_bundle

    return load_verified_guided_capture_bundle(path)


def _default_bundle_verifier(bundle: object) -> None:
    from tools.guided_stt_capture import verify_guided_capture_bundle

    verify_guided_capture_bundle(bundle)  # type: ignore[arg-type]


_BUNDLE_LOADER: Callable[[Path | str], object] = _default_bundle_loader
_BUNDLE_VERIFIER: Callable[[object], None] = _default_bundle_verifier


def _default_current_binder(bundle: _BundleState) -> object:
    from tools.live_launcher import _selected_live_config

    return _selected_live_config(
        _REPO_ROOT,
        bundle.device_profile,
        bundle.profile,
        True,
        guided_capture=True,
        input_gain=bundle.effective_input_gain,
    )


_CURRENT_BINDER: Callable[[_BundleState], object] = _default_current_binder


def _sha256(value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_CHARS for character in value)
    ):
        raise GuidedSttPairAttestationError()
    return value


def _exact_int(value: object, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise GuidedSttPairAttestationError()
    return value


def _strict_json(raw: bytes) -> object:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise GuidedSttPairAttestationError()
            result[key] = value
        return result

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                GuidedSttPairAttestationError()
            ),
        )
    except GuidedSttPairAttestationError:
        raise
    except (UnicodeError, ValueError, OverflowError, RecursionError):
        raise GuidedSttPairAttestationError() from None


def _canonical_json(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
            + b"\n"
        )
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise GuidedSttPairAttestationError() from None


def _fixed_case_order_digest() -> str:
    plan = built_in_guided_stt_plan()
    return hashlib.sha256(
        b"speaker-guided-stt-case-order-v1\0"
        + _canonical_json([case.case_id for case in plan.cases])
    ).hexdigest()


def _fixed_reference_totals(clips: int) -> tuple[int, int]:
    if type(clips) is not int or clips <= 0 or clips % CASES_PER_BUNDLE:
        raise GuidedSttPairAttestationError()
    plan = built_in_guided_stt_plan()
    words = sum(len(normalize(case.expected_text)) for case in plan.cases)
    characters = sum(len("".join(normalize(case.expected_text))) for case in plan.cases)
    multiplier = clips // CASES_PER_BUNDLE
    return words * multiplier, characters * multiplier


def _safe_path(value: object) -> Path:
    try:
        path = Path(value)  # type: ignore[arg-type]
        if not path.is_absolute() or not path.name:
            raise GuidedSttPairAttestationError()
        return path.resolve(strict=True)
    except GuidedSttPairAttestationError:
        raise
    except Exception:
        raise GuidedSttPairAttestationError() from None


def _bundle_state(raw: object, expected_profile: str) -> _BundleState:
    try:
        run_dir = _safe_path(getattr(raw, "run_dir"))
        plan_path = _safe_path(getattr(raw, "plan_path"))
        diagnostic_manifest_path = _safe_path(getattr(raw, "diagnostic_manifest_path"))
        profile = getattr(raw, "final_stt_profile")
        gain = getattr(raw, "effective_input_gain")
        device = getattr(raw, "device_profile")
        case_order_sha256 = _sha256(getattr(raw, "case_order_sha256"))
        if (
            profile != expected_profile
            or getattr(raw, "final_stt_profile_schema_version") != 1
            or getattr(raw, "case_count") != CASES_PER_BUNDLE
            or plan_path.parent != run_dir
            or diagnostic_manifest_path.parent != run_dir
            or type(device) is not str
            or not device
            or "/" in device
            or "\\" in device
            or isinstance(gain, bool)
            or not isinstance(gain, (int, float))
            or not math.isfinite(float(gain))
            or float(gain) <= 0.0
            or case_order_sha256 != _fixed_case_order_digest()
        ):
            raise GuidedSttPairAttestationError()
        for name in ("receipt_path", "contract_path", "summary_path"):
            if _safe_path(getattr(raw, name)).parent != run_dir:
                raise GuidedSttPairAttestationError()
        return _BundleState(
            source=raw,
            run_dir=run_dir,
            plan_path=plan_path,
            diagnostic_manifest_path=diagnostic_manifest_path,
            receipt_sha256=_sha256(getattr(raw, "receipt_sha256")),
            plan_sha256=_sha256(getattr(raw, "plan_sha256")),
            contract_sha256=_sha256(getattr(raw, "contract_sha256")),
            summary_sha256=_sha256(getattr(raw, "summary_sha256")),
            diagnostic_manifest_sha256=_sha256(
                getattr(raw, "diagnostic_manifest_sha256")
            ),
            profile=profile,
            profile_sha256=_sha256(getattr(raw, "final_stt_profile_sha256")),
            capture_config_sha256=_sha256(getattr(raw, "capture_config_sha256")),
            effective_sherpa_sha256=_sha256(getattr(raw, "effective_sherpa_sha256")),
            device_profile=device,
            effective_input_gain=float(gain),
            case_order_sha256=case_order_sha256,
        )
    except GuidedSttPairAttestationError:
        raise
    except Exception:
        raise GuidedSttPairAttestationError() from None


def _load_bundle(path: Path | str, expected_profile: str) -> _BundleState:
    try:
        raw = _BUNDLE_LOADER(path)
        return _bundle_state(raw, expected_profile)
    except GuidedSttPairAttestationError:
        raise
    except Exception:
        raise GuidedSttPairAttestationError() from None


def _recheck_bundle(bundle: _BundleState) -> None:
    try:
        _BUNDLE_VERIFIER(bundle.source)
        current = _bundle_state(_BUNDLE_LOADER(bundle.run_dir), bundle.profile)
        if current != bundle:
            raise GuidedSttPairAttestationError()
    except GuidedSttPairAttestationError:
        raise
    except Exception:
        raise GuidedSttPairAttestationError() from None


def _verify_current_binding(bundle: _BundleState) -> None:
    """Recompute the launcher-owned effective capture/profile binding."""

    try:
        current = _CURRENT_BINDER(bundle)
        if (
            getattr(current, "final_stt_profile") != bundle.profile
            or getattr(current, "final_stt_profile_sha256") != bundle.profile_sha256
            or getattr(current, "final_stt_profile_schema_version") != 1
            or getattr(current, "effective_device") != bundle.device_profile
            or float(getattr(current, "effective_input_gain"))
            != bundle.effective_input_gain
            or getattr(current, "capture_config_sha256") != bundle.capture_config_sha256
            or getattr(current, "effective_sherpa_sha256")
            != bundle.effective_sherpa_sha256
        ):
            raise GuidedSttPairAttestationError()
    except GuidedSttPairAttestationError:
        raise
    except Exception:
        raise GuidedSttPairAttestationError() from None


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _new_private_paths(
    scratch_root: Path | str,
    output_path: Path | str,
) -> tuple[Path, Path]:
    try:
        scratch_value = Path(scratch_root).expanduser()
        if not scratch_value.is_absolute() or not scratch_value.name:
            raise GuidedSttPairAttestationError()
        scratch = Path(os.path.abspath(scratch_value))
        output = qualification._new_private_output_path(output_path)
        if (
            scratch == output
            or _is_within(output, scratch)
            or _is_within(scratch, output)
            or qualification._has_git_ancestor(scratch)
            or qualification._has_git_ancestor(output)
        ):
            raise GuidedSttPairAttestationError()
        with opened_directory_nofollow(scratch.parent, require_private=True) as (
            stable_parent,
            parent_fd,
        ):
            if stable_parent != scratch.parent:
                raise GuidedSttPairAttestationError()
            try:
                os.stat(scratch.name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise GuidedSttPairAttestationError()
        return scratch, output
    except GuidedSttPairAttestationError:
        raise
    except Exception:
        raise GuidedSttPairAttestationError() from None


def _create_private_scratch(path: Path) -> tuple[int, int]:
    try:
        with opened_directory_nofollow(path.parent, require_private=True) as (
            stable_parent,
            parent_fd,
        ):
            if stable_parent != path.parent:
                raise GuidedSttPairAttestationError()
            os.mkdir(path.name, mode=0o700, dir_fd=parent_fd)
            os.fsync(parent_fd)
        metadata = path.lstat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o700
            or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
            or path.resolve(strict=True) != path
        ):
            raise GuidedSttPairAttestationError()
        return metadata.st_dev, metadata.st_ino
    except GuidedSttPairAttestationError:
        raise
    except Exception:
        raise GuidedSttPairAttestationError() from None


def _verify_scratch(path: Path, identity: tuple[int, int]) -> None:
    try:
        metadata = path.lstat()
        if (
            (metadata.st_dev, metadata.st_ino) != identity
            or not stat.S_ISDIR(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o700
            or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
            or path.resolve(strict=True) != path
        ):
            raise GuidedSttPairAttestationError()
    except GuidedSttPairAttestationError:
        raise
    except Exception:
        raise GuidedSttPairAttestationError() from None


def _private_file_identity(path: Path) -> tuple[int, int, int, int, int, int]:
    try:
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_nlink != 1
            or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
            or path.resolve(strict=True) != path
        ):
            raise GuidedSttPairAttestationError()
        return (
            metadata.st_dev,
            metadata.st_ino,
            stat.S_IFMT(metadata.st_mode),
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )
    except GuidedSttPairAttestationError:
        raise
    except Exception:
        raise GuidedSttPairAttestationError() from None


def _corpus_identities(
    corpus: LoadedCorpus,
) -> tuple[tuple[int, int, int, int, int, int], ...]:
    paths = (
        corpus.path,
        corpus.path.parent / "preparation-receipt.json",
        *(case.source_path for case in corpus.cases),
    )
    if len(set(paths)) != len(paths):
        raise GuidedSttPairAttestationError()
    return tuple(_private_file_identity(path) for path in paths)


def _unblock_child_lifecycle_signals() -> None:
    if hasattr(signal, "pthread_sigmask"):
        signal.pthread_sigmask(signal.SIG_UNBLOCK, _child_signal_numbers())


def _apply_child_limits() -> None:
    _unblock_child_lifecycle_signals()
    os.nice(15)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    for kind, maximum in (
        (resource.RLIMIT_NOFILE, MAX_CHILD_FDS),
        (resource.RLIMIT_FSIZE, MAX_CHILD_FILE_BYTES),
    ):
        _soft, hard = resource.getrlimit(kind)
        selected = maximum if hard == resource.RLIM_INFINITY else min(maximum, hard)
        resource.setrlimit(kind, (selected, selected))


def _lifecycle_signal_numbers() -> tuple[int, ...]:
    return tuple(
        signum
        for signum in (
            getattr(signal, "SIGHUP", None),
            getattr(signal, "SIGTERM", None),
        )
        if type(signum) is int
    )


def _child_signal_numbers() -> set[int]:
    return {
        signum
        for signum in (
            getattr(signal, "SIGHUP", None),
            getattr(signal, "SIGINT", None),
            getattr(signal, "SIGTERM", None),
        )
        if type(signum) is int
    }


def _lifecycle_signal_handler(signum: int, _frame: object) -> None:
    raise _LifecycleSignal(signum)


def _child_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_child_group(process: subprocess.Popen[object]) -> None:
    if not _child_group_exists(process.pid):
        try:
            process.wait(timeout=0.0)
        except BaseException:
            pass
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except OSError:
        pass
    try:
        process.wait(timeout=5.0)
    except (OSError, subprocess.TimeoutExpired):
        pass
    except BaseException:
        pass
    if _child_group_exists(process.pid):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            process.wait(timeout=5.0)
        except BaseException:
            pass


def _run_child(command: Sequence[str], *, timeout_sec: float) -> int:
    env = dict(os.environ)
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONHASHSEED": "0",
        }
    )
    env.pop("PYTHONHOME", None)
    process: subprocess.Popen[object] | None = None
    previous_mask: set[signal.Signals] | None = None
    try:
        if hasattr(signal, "pthread_sigmask"):
            previous_mask = signal.pthread_sigmask(
                signal.SIG_BLOCK,
                _child_signal_numbers(),
            )
        process = subprocess.Popen(
            list(command),
            cwd=_REPO_ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
            preexec_fn=_apply_child_limits,
        )
        if previous_mask is not None:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            previous_mask = None
        return_code = process.wait(timeout=timeout_sec)
        if _child_group_exists(process.pid):
            _terminate_child_group(process)
            raise GuidedSttPairAttestationError()
        return return_code
    except BaseException:
        if previous_mask is not None:
            try:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            except BaseException:
                pass
        if process is not None:
            _terminate_child_group(process)
        raise


_CHILD_RUNNER: Callable[..., int] = _run_child


def _private_json(path: Path, *, maximum_bytes: int) -> Mapping[str, object]:
    try:
        snapshot = read_regular_bounded(path, maximum_bytes=maximum_bytes)
        metadata = snapshot.path.lstat()
        if (
            stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_nlink != 1
            or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
        ):
            raise GuidedSttPairAttestationError()
        value = _strict_json(snapshot.data)
        if not isinstance(value, Mapping):
            raise GuidedSttPairAttestationError()
        return value
    except GuidedSttPairAttestationError:
        raise
    except (BoundedReadError, OSError):
        raise GuidedSttPairAttestationError() from None


def _prepared_state(
    *,
    role: str,
    labels_path: Path,
    corpus_path: Path,
    bundle: _BundleState,
) -> _PreparedState:
    try:
        labels_identity = _private_file_identity(labels_path)
        labels = read_regular_bounded(labels_path, maximum_bytes=256 * 1024)
        labels_metadata = labels.path.lstat()
        if (
            stat.S_IMODE(labels_metadata.st_mode) != 0o600
            or labels_metadata.st_nlink != 1
            or (hasattr(os, "geteuid") and labels_metadata.st_uid != os.geteuid())
            or _private_file_identity(labels.path) != labels_identity
        ):
            raise GuidedSttPairAttestationError()
        loaded = load_corpus(corpus_path / "corpus.json")
        if (
            loaded.schema_version != 3
            or loaded.provenance is None
            or loaded.provenance.kind != "private-diagnostic-v1"
            or loaded.provenance.suite != "final-model-input"
            or loaded.provenance.manifest_sha256 != bundle.diagnostic_manifest_sha256
            or loaded.provenance.metadata_sha256
            != hashlib.sha256(labels.data).hexdigest()
            or len(loaded.cases) != CASES_PER_BUNDLE
        ):
            raise GuidedSttPairAttestationError()
        plan = built_in_guided_stt_plan()
        role_counts: Counter[str] = Counter()
        for case, expected in zip(loaded.cases, plan.cases, strict=True):
            if (
                case.case_id != expected.case_id
                or case.expected_text != expected.expected_text
                or len(case.tags) != len(expected.tags) + 2
                or case.tags[0] != "private-diagnostic"
                or case.tags[1] not in {"model_gate_segment", "selected_asr_segment"}
                or tuple(case.tags[2:]) != expected.tags
            ):
                raise GuidedSttPairAttestationError()
            role_counts[case.tags[1]] += 1
        receipt_roles = (
            role_counts["model_gate_segment"],
            role_counts["selected_asr_segment"],
        )
        if receipt_roles != (0, CASES_PER_BUNDLE):
            raise GuidedSttPairAttestationError()
        corpus_identities = _corpus_identities(loaded)
        verify_corpus_snapshot(loaded)
        if _corpus_identities(loaded) != corpus_identities:
            raise GuidedSttPairAttestationError()
        from tools import recorded_stt_eval

        evaluator_load = recorded_stt_eval._load_corpus(loaded.path)
        recorded_stt_eval._verify_loaded_corpus(evaluator_load)
        if (
            _private_file_identity(labels.path) != labels_identity
            or _corpus_identities(loaded) != corpus_identities
        ):
            raise GuidedSttPairAttestationError()
        return _PreparedState(
            role=role,
            labels_path=labels.path,
            corpus_path=loaded.path,
            labels_sha256=hashlib.sha256(labels.data).hexdigest(),
            corpus_sha256=loaded.digest,
            cases=len(loaded.cases),
            audio_bytes=loaded.audio_bytes,
            receipt_roles=receipt_roles,
            loaded_corpus=loaded,
            labels_identity=labels_identity,
            corpus_identities=corpus_identities,
        )
    except GuidedSttPairAttestationError:
        raise
    except Exception:
        raise GuidedSttPairAttestationError() from None


def _recheck_prepared(state: _PreparedState, bundle: _BundleState) -> None:
    try:
        verify_corpus_snapshot(state.loaded_corpus)
        if (
            _private_file_identity(state.labels_path) != state.labels_identity
            or _corpus_identities(state.loaded_corpus) != state.corpus_identities
        ):
            raise GuidedSttPairAttestationError()
    except GuidedSttPairAttestationError:
        raise
    except Exception:
        raise GuidedSttPairAttestationError() from None
    current = _prepared_state(
        role=state.role,
        labels_path=state.labels_path,
        corpus_path=state.corpus_path.parent,
        bundle=bundle,
    )
    if current != state:
        raise GuidedSttPairAttestationError()
    try:
        verify_corpus_snapshot(state.loaded_corpus)
        if (
            _private_file_identity(state.labels_path) != state.labels_identity
            or _corpus_identities(state.loaded_corpus) != state.corpus_identities
        ):
            raise GuidedSttPairAttestationError()
    except GuidedSttPairAttestationError:
        raise
    except Exception:
        raise GuidedSttPairAttestationError() from None


def _closed_counts(
    value: object, allowed: frozenset[str], decisions: int
) -> dict[str, int]:
    if not isinstance(value, Mapping) or any(
        type(key) is not str
        or key not in allowed
        or type(count) is not int
        or count < 0
        for key, count in value.items()
    ):
        raise GuidedSttPairAttestationError()
    result = {str(key): int(count) for key, count in value.items()}
    if sum(result.values()) != decisions:
        raise GuidedSttPairAttestationError()
    return result


def _accuracy(value: object, clips: int) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _ACCURACY_FIELDS:
        raise GuidedSttPairAttestationError()
    result = dict(value)
    for name in _ACCURACY_INT_FIELDS:
        _exact_int(result.get(name))
    expected_ref_words, expected_ref_chars = _fixed_reference_totals(clips)
    if (
        result["clips"] != clips
        or result["nonempty"] > clips
        or result["exact"] > clips
        or result["ref_words"] != expected_ref_words
        or result["ref_chars"] != expected_ref_chars
        or result["word_errors"]
        != result["substitutions"] + result["insertions"] + result["deletions"]
        or result["deletions"] > result["ref_words"]
        or result["insertions"] > result["hyp_words"]
        or result["substitutions"]
        > min(
            result["ref_words"] - result["deletions"],
            result["hyp_words"] - result["insertions"],
        )
        or result["hyp_words"]
        != result["ref_words"] - result["deletions"] + result["insertions"]
        or result["nonempty"] > result["hyp_words"]
        or result["hyp_words"] > result["hyp_chars"]
        or result["word_errors"] < clips - result["exact"]
        or (result["word_errors"] == 0) is not (result["exact"] == clips)
        or (result["exact"] == clips and result["char_edits"] != 0)
        or abs(result["ref_chars"] - result["hyp_chars"]) > result["char_edits"]
        or result["char_edits"] > max(result["ref_chars"], result["hyp_chars"])
        or result["keyword_hits"] > result["keyword_attempts"]
        or result["keyword_attempts"] != 0
    ):
        raise GuidedSttPairAttestationError()
    expected_wer = (
        result["word_errors"] / result["ref_words"]
        if result["ref_words"]
        else (0.0 if result["hyp_words"] == 0 else 1.0)
    )
    expected_cer = (
        result["char_edits"] / result["ref_chars"]
        if result["ref_chars"]
        else (0.0 if result["hyp_chars"] == 0 else 1.0)
    )
    if result.get("wer") != round(expected_wer, 4) or result.get("cer") != round(
        expected_cer, 4
    ):
        raise GuidedSttPairAttestationError()
    return result


def _route_totals(value: object, *, cases: int, decisions: int) -> ToolRouteGateTotals:
    expected_fields = frozenset((*_ROUTE_COUNTER_FIELDS, "per_expected", "complete"))
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise GuidedSttPairAttestationError()
    counters = {name: _exact_int(value.get(name)) for name in _ROUTE_COUNTER_FIELDS}
    rows = value.get("per_expected")
    if not isinstance(rows, list) or len(rows) != len(EXPECTED_ROUTES):
        raise GuidedSttPairAttestationError()
    attempts: dict[str, int] = {}
    hits: dict[str, int] = {}
    for row, route in zip(rows, EXPECTED_ROUTES, strict=True):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"route", "attempts", "hits"}
            or row.get("route") != route
        ):
            raise GuidedSttPairAttestationError()
        attempts[route] = _exact_int(row.get("attempts"))
        hits[route] = _exact_int(row.get("hits"))
    totals = ToolRouteGateTotals(**counters, attempts=attempts, hits=hits)
    if (
        totals.annotated_cases != cases
        or totals.decisions != decisions
        or cases % CASES_PER_BUNDLE
        or attempts
        != {
            route: count * (cases // CASES_PER_BUNDLE)
            for route, count in _ROUTE_ATTEMPTS.items()
        }
        or value.get("complete") is not totals.complete
        or totals.as_dict() != dict(value)
    ):
        raise GuidedSttPairAttestationError()
    return totals


def _evaluation(
    value: object,
    profile: str,
    *,
    expected_clips: int = CASES_PER_BUNDLE,
) -> dict[str, object]:
    expected = {
        "clips",
        "decisions",
        "complete",
        "selected_sources_attested",
        "selected_source_accounting_complete",
        "offline_outcomes",
        "verifier_outcomes",
        "selected_sources",
        "streaming",
        "offline",
        "selected",
        "tool_route_gate",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise GuidedSttPairAttestationError()
    clips = _exact_int(value.get("clips"), minimum=1)
    decisions = _exact_int(value.get("decisions"), minimum=1)
    if (
        clips != expected_clips
        or value.get("complete") is not True
        or value.get("selected_sources_attested") is not True
        or value.get("selected_source_accounting_complete") is not True
    ):
        raise GuidedSttPairAttestationError()
    offline = _closed_counts(
        value.get("offline_outcomes"), _OFFLINE_OUTCOMES, decisions
    )
    verifier = _closed_counts(
        value.get("verifier_outcomes"), _VERIFIER_OUTCOMES, decisions
    )
    _closed_counts(value.get("selected_sources"), _SELECTED_SOURCES, decisions)
    if offline.get("error", 0) or not (
        offline.get("decoded", 0) or offline.get("empty", 0)
    ):
        raise GuidedSttPairAttestationError()
    if profile == CONTROL_PROFILE:
        if verifier != {"unavailable": decisions}:
            raise GuidedSttPairAttestationError()
    elif verifier.get("error", 0) or not any(
        verifier.get(outcome, 0) for outcome in _COMPLETED_VERIFIER_OUTCOMES
    ):
        raise GuidedSttPairAttestationError()
    accuracies = {
        name: _accuracy(value.get(name), clips)
        for name in ("streaming", "offline", "selected")
    }
    if (
        any(
            accuracy["exact"] > accuracy["nonempty"] for accuracy in accuracies.values()
        )
        or len(
            {
                (accuracy["ref_words"], accuracy["ref_chars"])
                for accuracy in accuracies.values()
            }
        )
        != 1
    ):
        raise GuidedSttPairAttestationError()
    _route_totals(value.get("tool_route_gate"), cases=clips, decisions=decisions)
    return dict(value)


def _child_report(
    value: object,
    *,
    corpus: _PreparedState,
    control: _BundleState,
    candidate: _BundleState,
    child_rc: int,
) -> dict[str, object]:
    expected = {
        "ok",
        "corpus_digest",
        "baseline_config_digest",
        "baseline_model_digest",
        "baseline",
        "baseline_final_stt_profile",
        "baseline_final_stt_profile_digest",
        "tool_route_profile_digest",
        "candidate_config_digest",
        "candidate_model_digest",
        "candidate",
        "comparison",
        "candidate_final_stt_profile",
        "candidate_final_stt_profile_digest",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise GuidedSttPairAttestationError()
    comparison = value.get("comparison")
    if not isinstance(comparison, Mapping) or set(comparison) != {
        "wins",
        "ties",
        "losses",
        "promotable",
    }:
        raise GuidedSttPairAttestationError()
    wins = _exact_int(comparison.get("wins"))
    ties = _exact_int(comparison.get("ties"))
    losses = _exact_int(comparison.get("losses"))
    promotable = comparison.get("promotable")
    if (
        wins + ties + losses != CASES_PER_BUNDLE
        or type(promotable) is not bool
        or value.get("ok") is not promotable
        or child_rc != (0 if promotable else 3)
        or value.get("corpus_digest") != corpus.corpus_sha256
        or value.get("baseline_final_stt_profile") != CONTROL_PROFILE
        or value.get("candidate_final_stt_profile") != CANDIDATE_PROFILE
        or value.get("baseline_final_stt_profile_digest") != control.profile_sha256
        or value.get("candidate_final_stt_profile_digest") != candidate.profile_sha256
        or value.get("tool_route_profile_digest") != _fixed_route_digest()
    ):
        raise GuidedSttPairAttestationError()
    for name in (
        "baseline_config_digest",
        "baseline_model_digest",
        "candidate_config_digest",
        "candidate_model_digest",
    ):
        _sha256(value.get(name))
    result = dict(value)
    result["baseline"] = _evaluation(value.get("baseline"), CONTROL_PROFILE)
    result["candidate"] = _evaluation(value.get("candidate"), CANDIDATE_PROFILE)
    if promotable is not _expected_child_promotable(
        result["baseline"],  # type: ignore[arg-type]
        result["candidate"],  # type: ignore[arg-type]
        wins=wins,
        losses=losses,
    ):
        raise GuidedSttPairAttestationError()
    result["comparison"] = dict(comparison)
    return result


def _fixed_route_digest() -> str:
    return route_profile_digest(
        ToolRouteGateProfile(
            vault_enabled=True,
            reminders_enabled=True,
            app_aliases=("obsidian",),
        )
    )


def _expected_child_promotable(
    control: Mapping[str, object],
    candidate: Mapping[str, object],
    *,
    wins: int,
    losses: int,
) -> bool:
    """Reproduce the aggregate portion of recorded_stt_eval's decision."""

    control_selected = control["selected"]
    candidate_selected = candidate["selected"]
    control_route = _route_totals(
        control["tool_route_gate"],
        cases=int(control["clips"]),
        decisions=int(control["decisions"]),
    )
    candidate_route = _route_totals(
        candidate["tool_route_gate"],
        cases=int(candidate["clips"]),
        decisions=int(candidate["decisions"]),
    )
    route_safe = route_no_regression(control_route, candidate_route)
    route_improvement = bool(
        route_safe and candidate_route.complete and not control_route.complete
    )
    no_regression = bool(
        candidate_selected["nonempty"] == candidate["clips"]  # type: ignore[index]
        and candidate_selected["word_errors"]  # type: ignore[index]
        <= control_selected["word_errors"]  # type: ignore[index]
        and candidate_selected["char_edits"]  # type: ignore[index]
        <= control_selected["char_edits"]  # type: ignore[index]
        and candidate_selected["keyword_hits"]  # type: ignore[index]
        >= control_selected["keyword_hits"]  # type: ignore[index]
        and losses <= wins
    )
    improvement = bool(
        candidate_selected["word_errors"]  # type: ignore[index]
        < control_selected["word_errors"]  # type: ignore[index]
        or candidate_selected["char_edits"]  # type: ignore[index]
        < control_selected["char_edits"]  # type: ignore[index]
        or candidate_selected["keyword_hits"]  # type: ignore[index]
        > control_selected["keyword_hits"]  # type: ignore[index]
        or route_improvement
    )
    return bool(
        no_regression
        and improvement
        and control_selected["nonempty"] == control["clips"]  # type: ignore[index]
        and candidate_selected["nonempty"] == candidate["clips"]  # type: ignore[index]
        and candidate_route.complete
        and route_safe
    )


def _sum_accuracy(values: Sequence[Mapping[str, object]]) -> dict[str, object]:
    integers = {
        name: sum(int(value[name]) for value in values) for name in _ACCURACY_INT_FIELDS
    }
    integers["wer"] = round(
        integers["word_errors"] / integers["ref_words"]
        if integers["ref_words"]
        else (0.0 if integers["hyp_words"] == 0 else 1.0),
        4,
    )
    integers["cer"] = round(
        integers["char_edits"] / integers["ref_chars"]
        if integers["ref_chars"]
        else (0.0 if integers["hyp_chars"] == 0 else 1.0),
        4,
    )
    return integers


def _sum_maps(values: Sequence[Mapping[str, int]]) -> dict[str, int]:
    result: Counter[str] = Counter()
    for value in values:
        result.update(value)
    return dict(sorted(result.items()))


def _sum_routes(values: Sequence[Mapping[str, object]]) -> dict[str, object]:
    attempts = Counter()
    hits = Counter()
    counters = Counter()
    for value in values:
        for name in _ROUTE_COUNTER_FIELDS:
            counters[name] += int(value[name])
        for row in value["per_expected"]:  # type: ignore[index]
            attempts[row["route"]] += int(row["attempts"])
            hits[row["route"]] += int(row["hits"])
    return ToolRouteGateTotals(
        **{name: counters[name] for name in _ROUTE_COUNTER_FIELDS},
        attempts={route: attempts[route] for route in EXPECTED_ROUTES},
        hits={route: hits[route] for route in EXPECTED_ROUTES},
    ).as_dict()


def _sum_evaluations(values: Sequence[Mapping[str, object]]) -> dict[str, object]:
    clips = sum(int(value["clips"]) for value in values)
    decisions = sum(int(value["decisions"]) for value in values)
    return {
        "clips": clips,
        "decisions": decisions,
        "complete": all(value["complete"] is True for value in values),
        "selected_sources_attested": True,
        "selected_source_accounting_complete": True,
        "offline_outcomes": _sum_maps(
            [value["offline_outcomes"] for value in values]  # type: ignore[list-item]
        ),
        "verifier_outcomes": _sum_maps(
            [value["verifier_outcomes"] for value in values]  # type: ignore[list-item]
        ),
        "selected_sources": _sum_maps(
            [value["selected_sources"] for value in values]  # type: ignore[list-item]
        ),
        "streaming": _sum_accuracy(
            [value["streaming"] for value in values]  # type: ignore[list-item]
        ),
        "offline": _sum_accuracy(
            [value["offline"] for value in values]  # type: ignore[list-item]
        ),
        "selected": _sum_accuracy(
            [value["selected"] for value in values]  # type: ignore[list-item]
        ),
        "tool_route_gate": _sum_routes(
            [value["tool_route_gate"] for value in values]  # type: ignore[list-item]
        ),
    }


def _comparison(
    reports: Sequence[Mapping[str, object]],
    control: Mapping[str, object],
    candidate: Mapping[str, object],
) -> dict[str, object]:
    wins = sum(int(report["comparison"]["wins"]) for report in reports)  # type: ignore[index]
    ties = sum(int(report["comparison"]["ties"]) for report in reports)  # type: ignore[index]
    losses = sum(int(report["comparison"]["losses"]) for report in reports)  # type: ignore[index]
    control_route = _route_totals(
        control["tool_route_gate"],
        cases=int(control["clips"]),
        decisions=int(control["decisions"]),
    )
    candidate_route = _route_totals(
        candidate["tool_route_gate"],
        cases=int(candidate["clips"]),
        decisions=int(candidate["decisions"]),
    )
    c_selected = candidate["selected"]
    b_selected = control["selected"]
    candidate_accuracy_safe = bool(
        c_selected["nonempty"] == candidate["clips"]  # type: ignore[index]
        and c_selected["word_errors"] <= b_selected["word_errors"]  # type: ignore[index]
        and c_selected["char_edits"] <= b_selected["char_edits"]  # type: ignore[index]
        and losses <= wins
    )
    candidate_route_safe = bool(
        candidate_route.complete and route_no_regression(control_route, candidate_route)
    )
    candidate_improvement = bool(
        c_selected["word_errors"] < b_selected["word_errors"]  # type: ignore[index]
        or c_selected["char_edits"] < b_selected["char_edits"]  # type: ignore[index]
        or candidate_route.exact_cases > control_route.exact_cases
    )
    control_accuracy_safe = bool(
        b_selected["nonempty"] == control["clips"]  # type: ignore[index]
        and b_selected["word_errors"] <= c_selected["word_errors"]  # type: ignore[index]
        and b_selected["char_edits"] <= c_selected["char_edits"]  # type: ignore[index]
        and wins <= losses
    )
    control_route_safe = bool(
        control_route.complete and route_no_regression(candidate_route, control_route)
    )
    control_improvement = bool(
        b_selected["word_errors"] < c_selected["word_errors"]  # type: ignore[index]
        or b_selected["char_edits"] < c_selected["char_edits"]  # type: ignore[index]
        or control_route.exact_cases > candidate_route.exact_cases
    )
    if candidate_accuracy_safe and candidate_route_safe and candidate_improvement:
        verdict = "candidate_preferred"
    elif control_accuracy_safe and control_route_safe and control_improvement:
        verdict = "control_preferred"
    elif (
        wins == 0
        and losses == 0
        and dict(b_selected) == dict(c_selected)  # type: ignore[arg-type]
        and control_route.as_dict() == candidate_route.as_dict()
    ):
        verdict = "tie"
    else:
        verdict = "mixed_inconclusive"
    return {
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "accuracy_no_regression": candidate_accuracy_safe,
        "tool_route_no_regression": candidate_route_safe,
        "improvement": candidate_improvement,
        "verdict": verdict,
    }


def _implementation_sha256() -> str:
    try:
        snapshot = read_regular_bounded(Path(__file__), maximum_bytes=2 * 1024 * 1024)
        return hashlib.sha256(snapshot.data).hexdigest()
    except Exception:
        raise GuidedSttPairAttestationError() from None


def _report(
    control_bundle: _BundleState,
    candidate_bundle: _BundleState,
    corpora: Sequence[_PreparedState],
    evaluations: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    control_evaluations = [value["baseline"] for value in evaluations]
    candidate_evaluations = [value["candidate"] for value in evaluations]
    combined_control = _sum_evaluations(control_evaluations)  # type: ignore[arg-type]
    combined_candidate = _sum_evaluations(candidate_evaluations)  # type: ignore[arg-type]
    comparison = _comparison(
        evaluations,
        combined_control,
        combined_candidate,
    )
    profile_bindings = (
        (
            "control",
            control_bundle,
            evaluations[0]["baseline_config_digest"],
            evaluations[0]["baseline_model_digest"],
        ),
        (
            "candidate",
            candidate_bundle,
            evaluations[0]["candidate_config_digest"],
            evaluations[0]["candidate_model_digest"],
        ),
    )
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "ok": True,
        "execution_complete": True,
        "quality_verdict": comparison["verdict"],
        "contract": {
            "capture_protocol": GUIDED_STT_CAPTURE_PROTOCOL,
            "plan_id": GUIDED_STT_PLAN_ID,
            "plan_sha256": control_bundle.plan_sha256,
            "case_order_sha256": _fixed_case_order_digest(),
            "cases_per_bundle": CASES_PER_BUNDLE,
            "capture_environment": {
                "capture_config_sha256": control_bundle.capture_config_sha256,
                "device_profile": control_bundle.device_profile,
                "effective_input_gain": control_bundle.effective_input_gain,
            },
            "route_availability_sha256": guided_stt_route_availability_sha256(),
            "tool_route_profile_sha256": _fixed_route_digest(),
            "profile_order": [CONTROL_PROFILE, CANDIDATE_PROFILE],
            "corpus_order": ["control-capture", "candidate-capture"],
            "execution_order": [
                ["control-capture", CONTROL_PROFILE],
                ["control-capture", CANDIDATE_PROFILE],
                ["candidate-capture", CONTROL_PROFILE],
                ["candidate-capture", CANDIDATE_PROFILE],
            ],
            "implementation_sha256": _implementation_sha256(),
        },
        "profiles": [
            {
                "role": role,
                "name": bundle.profile,
                "profile_sha256": bundle.profile_sha256,
                "config_sha256": config_sha,
                "model_sha256": model_sha,
            }
            for role, bundle, config_sha, model_sha in profile_bindings
        ],
        "bundles": [
            {
                "role": corpus.role,
                "capture_profile": bundle.profile,
                "receipt_sha256": bundle.receipt_sha256,
                "contract_sha256": bundle.contract_sha256,
                "summary_sha256": bundle.summary_sha256,
                "diagnostic_manifest_sha256": bundle.diagnostic_manifest_sha256,
                "corpus_sha256": corpus.corpus_sha256,
                "cases": corpus.cases,
                "audio_bytes": corpus.audio_bytes,
                "receipt_roles": {
                    "model_gate_segment": corpus.receipt_roles[0],
                    "selected_asr_segment": corpus.receipt_roles[1],
                },
            }
            for bundle, corpus in zip(
                (control_bundle, candidate_bundle), corpora, strict=True
            )
        ],
        "results": {
            "by_capture": [
                {
                    "role": corpus.role,
                    "control": evaluation["baseline"],
                    "candidate": evaluation["candidate"],
                    "comparison": evaluation["comparison"],
                }
                for corpus, evaluation in zip(corpora, evaluations, strict=True)
            ],
            "combined": {
                "control": combined_control,
                "candidate": combined_candidate,
                "comparison": comparison,
            },
        },
        "limits": {
            "aggregate_only": True,
            "sequential_execution": True,
            "same_pcm_within_each_profile_pair": True,
            "same_pcm_across_capture_bundles": False,
            "tool_invocation": False,
            "runtime_default_promotion": False,
            "owner_review_required": True,
            "hard_ram_limit": False,
            "hard_vram_limit": False,
        },
    }
    qualification._assert_path_free(result)
    return result


def validate_attestation_report(value: object) -> dict[str, object]:
    """Strictly validate the closed retained aggregate report shape."""

    try:
        top_fields = {
            "schema_version",
            "kind",
            "ok",
            "execution_complete",
            "quality_verdict",
            "contract",
            "profiles",
            "bundles",
            "results",
            "limits",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != top_fields
            or value.get("schema_version") != SCHEMA_VERSION
            or value.get("kind") != KIND
            or value.get("ok") is not True
            or value.get("execution_complete") is not True
            or value.get("quality_verdict")
            not in {
                "candidate_preferred",
                "control_preferred",
                "tie",
                "mixed_inconclusive",
            }
        ):
            raise GuidedSttPairAttestationError()
        contract = value.get("contract")
        profiles = value.get("profiles")
        bundles = value.get("bundles")
        results = value.get("results")
        limits = value.get("limits")
        contract_fields = {
            "capture_protocol",
            "plan_id",
            "plan_sha256",
            "case_order_sha256",
            "cases_per_bundle",
            "capture_environment",
            "route_availability_sha256",
            "tool_route_profile_sha256",
            "profile_order",
            "corpus_order",
            "execution_order",
            "implementation_sha256",
        }
        execution_order = [
            ["control-capture", CONTROL_PROFILE],
            ["control-capture", CANDIDATE_PROFILE],
            ["candidate-capture", CONTROL_PROFILE],
            ["candidate-capture", CANDIDATE_PROFILE],
        ]
        if (
            not isinstance(contract, Mapping)
            or set(contract) != contract_fields
            or contract.get("capture_protocol") != GUIDED_STT_CAPTURE_PROTOCOL
            or contract.get("plan_id") != GUIDED_STT_PLAN_ID
            or contract.get("plan_sha256") != built_in_guided_stt_plan().sha256
            or contract.get("case_order_sha256") != _fixed_case_order_digest()
            or contract.get("cases_per_bundle") != CASES_PER_BUNDLE
            or contract.get("profile_order") != [CONTROL_PROFILE, CANDIDATE_PROFILE]
            or contract.get("corpus_order") != ["control-capture", "candidate-capture"]
            or contract.get("execution_order") != execution_order
            or contract.get("route_availability_sha256")
            != guided_stt_route_availability_sha256()
            or contract.get("tool_route_profile_sha256") != _fixed_route_digest()
            or not isinstance(profiles, list)
            or len(profiles) != 2
            or not isinstance(bundles, list)
            or len(bundles) != 2
            or not isinstance(results, Mapping)
            or set(results) != {"by_capture", "combined"}
            or not isinstance(results.get("by_capture"), list)
            or len(results["by_capture"]) != 2
            or not isinstance(results.get("combined"), Mapping)
        ):
            raise GuidedSttPairAttestationError()

        capture_environment = contract.get("capture_environment")
        if not isinstance(capture_environment, Mapping) or set(capture_environment) != {
            "capture_config_sha256",
            "device_profile",
            "effective_input_gain",
        }:
            raise GuidedSttPairAttestationError()
        device = capture_environment.get("device_profile")
        gain = capture_environment.get("effective_input_gain")
        if (
            type(device) is not str
            or not device
            or "/" in device
            or "\\" in device
            or isinstance(gain, bool)
            or not isinstance(gain, (int, float))
            or not math.isfinite(float(gain))
            or float(gain) <= 0.0
        ):
            raise GuidedSttPairAttestationError()
        _sha256(capture_environment.get("capture_config_sha256"))
        for name in (
            "case_order_sha256",
            "implementation_sha256",
        ):
            _sha256(contract.get(name))

        profile_fields = {
            "role",
            "name",
            "profile_sha256",
            "config_sha256",
            "model_sha256",
        }
        profile_roles = (
            ("control", CONTROL_PROFILE),
            ("candidate", CANDIDATE_PROFILE),
        )
        for row, (role, name) in zip(profiles, profile_roles, strict=True):
            if (
                not isinstance(row, Mapping)
                or set(row) != profile_fields
                or row.get("role") != role
                or row.get("name") != name
            ):
                raise GuidedSttPairAttestationError()
            for digest_name in (
                "profile_sha256",
                "config_sha256",
                "model_sha256",
            ):
                _sha256(row.get(digest_name))

        bundle_fields = {
            "role",
            "capture_profile",
            "receipt_sha256",
            "contract_sha256",
            "summary_sha256",
            "diagnostic_manifest_sha256",
            "corpus_sha256",
            "cases",
            "audio_bytes",
            "receipt_roles",
        }
        bundle_roles = (
            ("control-capture", CONTROL_PROFILE),
            ("candidate-capture", CANDIDATE_PROFILE),
        )
        for row, (role, profile) in zip(bundles, bundle_roles, strict=True):
            if (
                not isinstance(row, Mapping)
                or set(row) != bundle_fields
                or row.get("role") != role
                or row.get("capture_profile") != profile
                or row.get("cases") != CASES_PER_BUNDLE
                or type(row.get("audio_bytes")) is not int
                or not 0 < row["audio_bytes"] <= MAX_CORPUS_BYTES
                or row.get("receipt_roles")
                != {
                    "model_gate_segment": 0,
                    "selected_asr_segment": CASES_PER_BUNDLE,
                }
            ):
                raise GuidedSttPairAttestationError()
            for digest_name in (
                "receipt_sha256",
                "contract_sha256",
                "summary_sha256",
                "diagnostic_manifest_sha256",
                "corpus_sha256",
            ):
                _sha256(row.get(digest_name))

        by_capture = results["by_capture"]
        validated_rows: list[dict[str, object]] = []
        for row, (role, _profile) in zip(by_capture, bundle_roles, strict=True):
            if (
                not isinstance(row, Mapping)
                or set(row) != {"role", "control", "candidate", "comparison"}
                or row.get("role") != role
            ):
                raise GuidedSttPairAttestationError()
            control_result = _evaluation(row.get("control"), CONTROL_PROFILE)
            candidate_result = _evaluation(row.get("candidate"), CANDIDATE_PROFILE)
            comparison = row.get("comparison")
            if not isinstance(comparison, Mapping) or set(comparison) != {
                "wins",
                "ties",
                "losses",
                "promotable",
            }:
                raise GuidedSttPairAttestationError()
            wins = _exact_int(comparison.get("wins"))
            ties = _exact_int(comparison.get("ties"))
            losses = _exact_int(comparison.get("losses"))
            if (
                wins + ties + losses != CASES_PER_BUNDLE
                or type(comparison.get("promotable")) is not bool
                or comparison.get("promotable")
                is not _expected_child_promotable(
                    control_result,
                    candidate_result,
                    wins=wins,
                    losses=losses,
                )
            ):
                raise GuidedSttPairAttestationError()
            validated_rows.append(
                {
                    "role": role,
                    "control": control_result,
                    "candidate": candidate_result,
                    "comparison": dict(comparison),
                }
            )

        combined = results["combined"]
        if not isinstance(combined, Mapping) or set(combined) != {
            "control",
            "candidate",
            "comparison",
        }:
            raise GuidedSttPairAttestationError()
        combined_control = _evaluation(
            combined.get("control"),
            CONTROL_PROFILE,
            expected_clips=CASES_PER_BUNDLE * 2,
        )
        combined_candidate = _evaluation(
            combined.get("candidate"),
            CANDIDATE_PROFILE,
            expected_clips=CASES_PER_BUNDLE * 2,
        )
        expected_control = _sum_evaluations(
            [row["control"] for row in validated_rows]  # type: ignore[list-item]
        )
        expected_candidate = _sum_evaluations(
            [row["candidate"] for row in validated_rows]  # type: ignore[list-item]
        )
        expected_comparison = _comparison(
            validated_rows,
            expected_control,
            expected_candidate,
        )
        if (
            combined_control != expected_control
            or combined_candidate != expected_candidate
            or combined.get("comparison") != expected_comparison
            or value.get("quality_verdict") != expected_comparison["verdict"]
        ):
            raise GuidedSttPairAttestationError()

        expected_limits = {
            "aggregate_only": True,
            "sequential_execution": True,
            "same_pcm_within_each_profile_pair": True,
            "same_pcm_across_capture_bundles": False,
            "tool_invocation": False,
            "runtime_default_promotion": False,
            "owner_review_required": True,
            "hard_ram_limit": False,
            "hard_vram_limit": False,
        }
        if not isinstance(limits, Mapping) or dict(limits) != expected_limits:
            raise GuidedSttPairAttestationError()

        qualification._assert_path_free(value)
        encoded = _canonical_json(value)
        if len(encoded) > MAX_REPORT_BYTES:
            raise GuidedSttPairAttestationError()
        snapshot = _strict_json(encoded)
        if not isinstance(snapshot, dict):
            raise GuidedSttPairAttestationError()
        return snapshot
    except GuidedSttPairAttestationError:
        raise
    except Exception:
        raise GuidedSttPairAttestationError() from None


def _report_entry_identity(metadata: os.stat_result) -> tuple[int, int, int]:
    return metadata.st_dev, metadata.st_ino, stat.S_IFMT(metadata.st_mode)


def _report_parent_identity(
    metadata: os.stat_result,
) -> tuple[int, int, int, int, int]:
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
    ):
        raise GuidedSttPairAttestationError()
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        stat.S_IMODE(metadata.st_mode),
        metadata.st_uid,
    )


def _verify_report_parent_binding(
    path: Path,
    directory_fd: int,
    identity: tuple[int, int, int, int, int],
) -> None:
    try:
        opened = os.fstat(directory_fd)
        lexical = path.lstat()
        if (
            _report_parent_identity(opened) != identity
            or _report_parent_identity(lexical) != identity
            or path.resolve(strict=True) != path
        ):
            raise GuidedSttPairAttestationError()
    except GuidedSttPairAttestationError:
        raise
    except Exception:
        raise GuidedSttPairAttestationError() from None


def _commit_report_link(
    directory_fd: int,
    descriptor: int,
    name: str,
    identity: tuple[int, int, int],
    state: _ReportCommitState,
) -> None:
    if not hasattr(signal, "pthread_sigmask"):
        raise GuidedSttPairAttestationError()
    previous_mask = signal.pthread_sigmask(
        signal.SIG_BLOCK,
        _child_signal_numbers(),
    )
    link_returned = False
    try:
        try:
            os.link(
                f"/proc/self/fd/{descriptor}",
                name,
                dst_dir_fd=directory_fd,
                follow_symlinks=True,
            )
            link_returned = True
        finally:
            if link_returned:
                state.committed = True
            else:
                try:
                    opened = os.fstat(descriptor)
                    published = os.stat(
                        name,
                        dir_fd=directory_fd,
                        follow_symlinks=False,
                    )
                except OSError:
                    pass
                else:
                    if (
                        _report_entry_identity(opened) == identity
                        and _report_entry_identity(published) == identity
                        and opened.st_nlink == 1
                        and published.st_nlink == 1
                        and stat.S_IMODE(opened.st_mode) == 0o600
                        and stat.S_IMODE(published.st_mode) == 0o600
                        and opened.st_size == published.st_size
                        and (
                            not hasattr(os, "geteuid")
                            or (
                                opened.st_uid == os.geteuid()
                                and published.st_uid == os.geteuid()
                            )
                        )
                    ):
                        state.committed = True
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def _read_report_descriptor(descriptor: int, maximum_bytes: int) -> bytes:
    result = bytearray()
    while len(result) <= maximum_bytes:
        chunk = os.read(
            descriptor,
            min(1024 * 1024, maximum_bytes + 1 - len(result)),
        )
        if not chunk:
            break
        result.extend(chunk)
    if len(result) > maximum_bytes:
        raise GuidedSttPairAttestationError()
    return bytes(result)


def write_new_report(
    path: Path | str,
    value: Mapping[str, object],
    *,
    _commit_state: _ReportCommitState | None = None,
    _publication: AttestationPublication | None = None,
    _commit_guard: Callable[[], bool] | None = None,
) -> str:
    """Publish one private report and preserve its exact interrupt boundary."""

    state = _commit_state or _ReportCommitState()
    descriptor = -1
    report_sha256 = ""
    try:
        if state.committed:
            raise GuidedSttPairAttestationError()
        if _commit_guard is not None and not callable(_commit_guard):
            raise GuidedSttPairAttestationError()
        selected = qualification._new_private_output_path(path)
        if qualification._has_git_ancestor(selected):
            raise GuidedSttPairAttestationError()
        validated = validate_attestation_report(value)
        encoded = _canonical_json(validated)
        if len(encoded) > MAX_REPORT_BYTES:
            raise GuidedSttPairAttestationError()
        report_sha256 = hashlib.sha256(encoded).hexdigest()
        if _publication is not None and (
            _publication.report_sha256 != report_sha256
            or dict(_publication.report) != validated
        ):
            raise GuidedSttPairAttestationError()

        with opened_directory_nofollow(
            selected.parent,
            require_private=True,
        ) as (stable_parent, directory_fd):
            if stable_parent != selected.parent:
                raise GuidedSttPairAttestationError()
            parent_identity = _report_parent_identity(os.fstat(directory_fd))
            try:
                os.stat(
                    selected.name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                raise GuidedSttPairAttestationError()

            temporary_flag = getattr(os, "O_TMPFILE", 0)
            if not temporary_flag:
                raise GuidedSttPairAttestationError()
            flags = os.O_RDWR | temporary_flag
            flags |= getattr(os, "O_CLOEXEC", 0)
            descriptor = os.open(
                ".",
                flags,
                0o600,
                dir_fd=directory_fd,
            )
            created = os.fstat(descriptor)
            if (
                not stat.S_ISREG(created.st_mode)
                or created.st_nlink != 0
                or (hasattr(os, "geteuid") and created.st_uid != os.geteuid())
            ):
                raise GuidedSttPairAttestationError()
            staged_identity = _report_entry_identity(created)
            os.fchmod(descriptor, 0o600)

            view = memoryview(encoded)
            written = 0
            while written < len(view):
                count = os.write(descriptor, view[written:])
                if type(count) is not int or count <= 0:
                    raise GuidedSttPairAttestationError()
                written += count
            os.fsync(descriptor)
            before_read = os.fstat(descriptor)
            if (
                _report_entry_identity(before_read) != staged_identity
                or not stat.S_ISREG(before_read.st_mode)
                or stat.S_IMODE(before_read.st_mode) != 0o600
                or before_read.st_nlink != 0
                or before_read.st_size != len(encoded)
                or (hasattr(os, "geteuid") and before_read.st_uid != os.geteuid())
            ):
                raise GuidedSttPairAttestationError()
            os.lseek(descriptor, 0, os.SEEK_SET)
            observed = _read_report_descriptor(descriptor, len(encoded))
            after_read = os.fstat(descriptor)
            if (
                observed != encoded
                or hashlib.sha256(observed).hexdigest() != report_sha256
                or _report_entry_identity(after_read) != staged_identity
                or after_read.st_nlink != 0
            ):
                raise GuidedSttPairAttestationError()

            # O_TMPFILE leaves no pathname to race or clean up. The one final
            # no-clobber link is the terminal commit and is signal-masked together
            # with the shared commit marker.
            if _commit_guard is not None and _commit_guard() is not True:
                raise GuidedSttPairAttestationError()
            _verify_report_parent_binding(
                selected.parent,
                directory_fd,
                parent_identity,
            )
            _commit_report_link(
                directory_fd,
                descriptor,
                selected.name,
                staged_identity,
                state,
            )
            if not state.committed:
                raise GuidedSttPairAttestationError()
            try:
                os.fsync(directory_fd)
            except BaseException:
                pass
        return report_sha256
    except BaseException as error:
        if state.committed:
            return report_sha256
        if isinstance(error, (KeyboardInterrupt, _LifecycleSignal)):
            raise
        raise GuidedSttPairAttestationError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _prepare_command(bundle: _BundleState, state_root: Path, role: str) -> list[str]:
    return [
        sys.executable,
        "-B",
        "-m",
        "tools.prepare_live_stt_corpus",
        "--diagnostic-manifest",
        str(bundle.diagnostic_manifest_path),
        "--reference-plan",
        str(bundle.plan_path),
        "--labels-output",
        str(state_root / f"{role}-labels.json"),
        "--output-dir",
        str(state_root / f"{role}-corpus"),
    ]


def _evaluation_command(
    corpus: _PreparedState,
    device: str,
    output: Path,
) -> list[str]:
    return [
        sys.executable,
        "-B",
        "-m",
        "tools.recorded_stt_eval",
        "--manifest",
        str(corpus.corpus_path),
        "--device",
        device,
        "--baseline-final-stt-profile",
        CONTROL_PROFILE,
        "--candidate-final-stt-profile",
        CANDIDATE_PROFILE,
        "--tool-route-gate",
        "--tool-route-vault-enabled",
        "--tool-route-reminders-enabled",
        "--tool-route-app-alias",
        "obsidian",
        "--output",
        str(output),
    ]


def run_attestation(
    *,
    control_bundle: Path | str,
    candidate_bundle: Path | str,
    scratch_root: Path | str,
    output_path: Path | str,
    _commit_state: _ReportCommitState | None = None,
) -> AttestationPublication:
    """Run the fixed two-bundle/two-profile cross-over exactly once."""

    try:
        scratch, output = _new_private_paths(scratch_root, output_path)
        control = _load_bundle(control_bundle, CONTROL_PROFILE)
        candidate = _load_bundle(candidate_bundle, CANDIDATE_PROFILE)
        if (
            control.run_dir == candidate.run_dir
            or _is_within(control.run_dir, candidate.run_dir)
            or _is_within(candidate.run_dir, control.run_dir)
            or any(
                _is_within(selected, bundle.run_dir)
                or _is_within(bundle.run_dir, selected)
                for selected in (scratch, output)
                for bundle in (control, candidate)
            )
            or control.plan_sha256 != candidate.plan_sha256
            or control.case_order_sha256 != candidate.case_order_sha256
            or control.capture_config_sha256 != candidate.capture_config_sha256
            or control.device_profile != candidate.device_profile
            or control.effective_input_gain != candidate.effective_input_gain
        ):
            raise GuidedSttPairAttestationError()
        for bundle in (control, candidate):
            _verify_current_binding(bundle)
        scratch_identity = _create_private_scratch(scratch)
        bundles = (control, candidate)
        roles = ("control-capture", "candidate-capture")
        prepared: list[_PreparedState] = []
        for bundle, role in zip(bundles, roles, strict=True):
            command = _prepare_command(bundle, scratch, role)
            if _CHILD_RUNNER(command, timeout_sec=PREPARE_TIMEOUT_SEC) != 0:
                raise GuidedSttPairAttestationError()
            _verify_scratch(scratch, scratch_identity)
            for current in bundles:
                _recheck_bundle(current)
                _verify_current_binding(current)
            for state, source in zip(prepared, bundles, strict=False):
                _recheck_prepared(state, source)
            prepared.append(
                _prepared_state(
                    role=role,
                    labels_path=scratch / f"{role}-labels.json",
                    corpus_path=scratch / f"{role}-corpus",
                    bundle=bundle,
                )
            )

        for bundle in bundles:
            _verify_current_binding(bundle)
        evaluations: list[dict[str, object]] = []
        for corpus, bundle in zip(prepared, bundles, strict=True):
            for current in bundles:
                _verify_current_binding(current)
            child_output = scratch / f"{corpus.role}-evaluation.json"
            command = _evaluation_command(
                corpus,
                control.device_profile,
                child_output,
            )
            child_rc = _CHILD_RUNNER(command, timeout_sec=EVALUATE_TIMEOUT_SEC)
            if child_rc not in {0, 3}:
                raise GuidedSttPairAttestationError()
            child = _child_report(
                _private_json(child_output, maximum_bytes=MAX_CHILD_REPORT_BYTES),
                corpus=corpus,
                control=control,
                candidate=candidate,
                child_rc=child_rc,
            )
            if evaluations and any(
                child[name] != evaluations[0][name]
                for name in (
                    "baseline_config_digest",
                    "baseline_model_digest",
                    "candidate_config_digest",
                    "candidate_model_digest",
                    "baseline_final_stt_profile_digest",
                    "candidate_final_stt_profile_digest",
                    "tool_route_profile_digest",
                )
            ):
                raise GuidedSttPairAttestationError()
            evaluations.append(child)
            _verify_scratch(scratch, scratch_identity)
            for current in bundles:
                _recheck_bundle(current)
                _verify_current_binding(current)
            for state, source in zip(prepared, bundles, strict=True):
                _recheck_prepared(state, source)

        if len(evaluations) != 2:
            raise GuidedSttPairAttestationError()
        report = validate_attestation_report(
            _report(control, candidate, prepared, evaluations)
        )
        _verify_scratch(scratch, scratch_identity)
        for current in bundles:
            _recheck_bundle(current)
            _verify_current_binding(current)
        for state, source in zip(prepared, bundles, strict=True):
            _recheck_prepared(state, source)
        if qualification._new_private_output_path(output) != output:
            raise GuidedSttPairAttestationError()
        report_sha256 = hashlib.sha256(_canonical_json(report)).hexdigest()
        publication = AttestationPublication(
            report=report,
            report_sha256=report_sha256,
        )

        def commit_guard() -> bool:
            _verify_scratch(scratch, scratch_identity)
            for current in bundles:
                _recheck_bundle(current)
                _verify_current_binding(current)
            for state, source in zip(prepared, bundles, strict=True):
                _recheck_prepared(state, source)
            return True

        commit_state = _commit_state or _ReportCommitState()
        if commit_state.pending_publication is not None or commit_state.committed:
            raise GuidedSttPairAttestationError()
        commit_state.pending_publication = publication
        try:
            observed_sha256 = write_new_report(
                output,
                report,
                _commit_state=commit_state,
                _publication=publication,
                _commit_guard=commit_guard,
            )
            if observed_sha256 != report_sha256:
                raise GuidedSttPairAttestationError()
            return publication
        except BaseException:
            if commit_state.committed:
                return publication
            raise
    except GuidedSttPairAttestationError:
        raise
    except Exception:
        raise GuidedSttPairAttestationError() from None


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise GuidedSttPairAttestationError()


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        description=(
            "Cross-replay one fixed SenseVoice and one fixed Parakeet/Faster-Whisper "
            "guided capture; publish aggregate-only evidence."
        )
    )
    parser.add_argument("--control-bundle", type=Path, required=True)
    parser.add_argument("--candidate-bundle", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _main(argv: Sequence[str] | None = None) -> int:
    commit_state = _ReportCommitState()
    try:
        args = _parser().parse_args(argv)
        publication = run_attestation(
            control_bundle=args.control_bundle,
            candidate_bundle=args.candidate_bundle,
            scratch_root=args.scratch_root,
            output_path=args.output,
            _commit_state=commit_state,
        )
    except _LifecycleSignal as interrupted:
        if commit_state.committed and commit_state.pending_publication is not None:
            publication = commit_state.pending_publication
        else:
            try:
                print(_canonical_json(_SAFE_ERROR).decode("ascii"), end="")
            except BaseException:
                pass
            return 128 + interrupted.signum
    except KeyboardInterrupt:
        if commit_state.committed and commit_state.pending_publication is not None:
            publication = commit_state.pending_publication
        else:
            try:
                print(_canonical_json(_SAFE_ERROR).decode("ascii"), end="")
            except BaseException:
                pass
            return 130
    except Exception:
        if commit_state.committed and commit_state.pending_publication is not None:
            publication = commit_state.pending_publication
        else:
            try:
                print(_canonical_json(_SAFE_ERROR).decode("ascii"), end="")
            except BaseException:
                pass
            return 2

    try:
        assert publication is not None
    except BaseException:
        try:
            print(_canonical_json(_SAFE_ERROR).decode("ascii"), end="")
        except BaseException:
            pass
        return 2

    # Publication is the terminal commit.  Once it returns successfully, a
    # broken stdout consumer or late interrupt must not claim that no report
    # exists by changing the process status to a failure code.
    try:
        summary = {
            "execution_complete": True,
            "kind": KIND,
            "ok": True,
            "quality_verdict": publication.report["quality_verdict"],
            "report_sha256": publication.report_sha256,
            "report_written": True,
        }
        print(_canonical_json(summary).decode("ascii"), end="")
    except BaseException:
        pass
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    previous_handlers: dict[int, object] = {}
    try:
        for signum in _lifecycle_signal_numbers():
            previous_handlers[signum] = signal.signal(
                signum,
                _lifecycle_signal_handler,
            )
    except _LifecycleSignal as interrupted:
        result = 128 + interrupted.signum
    except KeyboardInterrupt:
        result = 130
    except Exception:
        result = 2
    else:
        try:
            return _main(argv)
        finally:
            for signum, previous in reversed(tuple(previous_handlers.items())):
                try:
                    signal.signal(signum, previous)
                except BaseException:
                    pass

    for signum, previous in reversed(tuple(previous_handlers.items())):
        try:
            signal.signal(signum, previous)
        except BaseException:
            pass
    try:
        print(_canonical_json(_SAFE_ERROR).decode("ascii"), end="")
    except BaseException:
        pass
    return result


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "AttestationPublication",
    "GuidedSttPairAttestationError",
    "KIND",
    "SCHEMA_VERSION",
    "main",
    "run_attestation",
    "validate_attestation_report",
    "write_new_report",
]
