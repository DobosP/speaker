"""Bounded adapter for NVIDIA Parakeet Realtime EOU 120M.

The candidate runtime is imported lazily inside the isolated worker. The
adapter uses the exact NeMo 2.7.3 ``NemoStreamingASRService`` with a local
``.nemo`` file, never a model identifier or download path. It stops on the
first native ``<EOU>``/``<EOB>`` marker and reports the source and declared
silence tail actually delivered to the model. Service text is a detokenized
delta per call, so the adapter preserves raw delta boundaries while building
the cumulative hypothesis exposed to the worker.
"""

from __future__ import annotations

from array import array
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import importlib
from importlib import metadata, util
import math
import os
from pathlib import Path
import struct
import sys
import threading
import time
from types import ModuleType
from typing import TypeVar

from ..bounded_io import (
    BoundedReadError,
    hash_regular_bounded,
    read_regular_bounded,
)
from ..manifest import (
    PARAKEET_REALTIME_EOU_MODEL_RECEIPT,
    ParakeetRealtimeEouConfig,
)
from ..protocol import (
    MAX_HYPOTHESIS_CHARS,
    MAX_PARTIALS_PER_CASE,
    MAX_PCM_BYTES,
    MAX_RESOURCE_MB,
    MAX_STREAM_SAMPLES,
    NATIVE_ENDPOINT_PROTOCOL_VERSION,
    ResourceUsage,
    TranscribeRequest,
)


_NEMO_DISTRIBUTION = "nemo-toolkit"
_TORCH_DISTRIBUTION = "torch"
_NUMPY_DISTRIBUTION = "numpy"
_PIPECAT_PACKAGE = "nemo.agents.voice_agent.pipecat"
_PIPECAT_SERVICES_PACKAGE = f"{_PIPECAT_PACKAGE}.services"
_PIPECAT_NEMO_PACKAGE = f"{_PIPECAT_SERVICES_PACKAGE}.nemo"
_UTILS_MODULE = f"{_PIPECAT_NEMO_PACKAGE}.utils"
_SERVICE_MODULE = "nemo.agents.voice_agent.pipecat.services.nemo.streaming_asr"
_NEMO_PIPECAT_FILES = {
    _PIPECAT_PACKAGE: (
        "nemo/agents/voice_agent/pipecat/__init__.py",
        "31a8b51d0dc8a34a45a2d2c63d44395e8607c0d57995ae33d10e643ce56846c2",
        755,
    ),
    _PIPECAT_SERVICES_PACKAGE: (
        "nemo/agents/voice_agent/pipecat/services/__init__.py",
        "05b7a7dd9408434f37d08ddfbcf9dce2534d2b8319a0b0df44d85f2a887e68b8",
        610,
    ),
    _PIPECAT_NEMO_PACKAGE: (
        "nemo/agents/voice_agent/pipecat/services/nemo/__init__.py",
        "54da6d75a1797458e9bfd1257ee2a9addf9c7fe105a826fc3955c05e4e665653",
        811,
    ),
    _SERVICE_MODULE: (
        "nemo/agents/voice_agent/pipecat/services/nemo/streaming_asr.py",
        "db89c405aacf39e6aa2251e48d443a5fa65e4db949ceeac2b134cdf203578ee7",
        13_527,
    ),
    _UTILS_MODULE: (
        "nemo/agents/voice_agent/pipecat/services/nemo/utils.py",
        "455835e5fce84cd81356ac8f3f4dbf45cb59e163cccd8b0ebd4ed617318e6c94",
        7_070,
    ),
}
_SYNTHETIC_PACKAGES = (
    _PIPECAT_PACKAGE,
    _PIPECAT_SERVICES_PACKAGE,
    _PIPECAT_NEMO_PACKAGE,
)
_OPTIONAL_DISTRIBUTIONS = ("pipecat-ai", "pipecat", "wget")
_OPTIONAL_MODULES = ("pipecat", "wget")
_MEBIBYTE = 1024.0 * 1024.0
_VRAM_FRACTION = 0.25
_T = TypeVar("_T")


class ParakeetRealtimeEouAdapterError(RuntimeError):
    """A detail-free candidate load, state, or streaming failure."""


@dataclass(frozen=True)
class _AuthenticatedSource:
    path: Path
    data: bytes = field(repr=False)
    sha256: str
    size_bytes: int


def _deny_wget_download(*_args: object, **_kwargs: object) -> None:
    raise ParakeetRealtimeEouAdapterError()


def _optional_imports_are_absent() -> None:
    try:
        blocked_prefixes = (*_OPTIONAL_MODULES, _PIPECAT_PACKAGE)
        if any(
            name == prefix or name.startswith(f"{prefix}.")
            for name in tuple(sys.modules)
            for prefix in blocked_prefixes
        ):
            raise ParakeetRealtimeEouAdapterError()
        voice_agent = sys.modules.get("nemo.agents.voice_agent")
        if isinstance(voice_agent, ModuleType) and "pipecat" in vars(voice_agent):
            raise ParakeetRealtimeEouAdapterError()
        for distribution_name in _OPTIONAL_DISTRIBUTIONS:
            try:
                metadata.distribution(distribution_name)
            except metadata.PackageNotFoundError:
                continue
            raise ParakeetRealtimeEouAdapterError()
        if any(util.find_spec(name) is not None for name in _OPTIONAL_MODULES):
            raise ParakeetRealtimeEouAdapterError()
    except ParakeetRealtimeEouAdapterError:
        raise
    except Exception:
        raise ParakeetRealtimeEouAdapterError() from None


def _authenticated_nemo_pipecat_sources(
    config: ParakeetRealtimeEouConfig,
) -> dict[str, _AuthenticatedSource]:
    try:
        distribution = metadata.distribution(_NEMO_DISTRIBUTION)
        if distribution.version != config.nemo_version:
            raise ParakeetRealtimeEouAdapterError()
        root = Path(distribution.locate_file(""))
        if not root.is_absolute() or root.resolve(strict=True) != root:
            raise ParakeetRealtimeEouAdapterError()
        authenticated: dict[str, _AuthenticatedSource] = {}
        for module_name, (relative, expected_sha256, expected_size) in (
            _NEMO_PIPECAT_FILES.items()
        ):
            expected = root / relative
            located = Path(distribution.locate_file(relative))
            if not located.is_absolute() or located != expected:
                raise ParakeetRealtimeEouAdapterError()
            snapshot = read_regular_bounded(
                located,
                maximum_bytes=expected_size,
                expected_bytes=expected_size,
            )
            digest = hashlib.sha256(snapshot.data).hexdigest()
            if (
                snapshot.path != expected
                or digest != expected_sha256
                or len(snapshot.data) != expected_size
                or Path(distribution.locate_file(relative)) != expected
            ):
                raise ParakeetRealtimeEouAdapterError()
            authenticated[module_name] = _AuthenticatedSource(
                path=expected,
                data=snapshot.data,
                sha256=digest,
                size_bytes=len(snapshot.data),
            )
        if Path(distribution.locate_file("")) != root:
            raise ParakeetRealtimeEouAdapterError()
        return authenticated
    except (
        AttributeError,
        BoundedReadError,
        ImportError,
        metadata.PackageNotFoundError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        raise ParakeetRealtimeEouAdapterError() from None


def _execute_authenticated_module(
    name: str,
    source: _AuthenticatedSource,
    parent: ModuleType,
) -> ModuleType:
    """Execute one exact in-memory source snapshot without reopening its path."""

    child_name = name.rsplit(".", 1)[-1]
    if name in sys.modules or child_name in vars(parent):
        raise ParakeetRealtimeEouAdapterError()
    module = ModuleType(name)
    module.__file__ = str(source.path)
    module.__package__ = name.rpartition(".")[0]
    module.__loader__ = None
    module.__spec__ = None
    module.__speaker_source_sha256__ = source.sha256
    module.__speaker_source_size_bytes__ = source.size_bytes
    sys.modules[name] = module
    setattr(parent, child_name, module)
    try:
        code = compile(
            source.data,
            str(source.path),
            "exec",
            flags=0,
            dont_inherit=True,
            optimize=0,
        )
        exec(code, module.__dict__)
    except Exception:
        raise ParakeetRealtimeEouAdapterError() from None
    return module


def _exact_module_source(module: object, expected: _AuthenticatedSource) -> bool:
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str):
        return False
    try:
        lexical = Path(os.path.abspath(module_file))
        return (
            lexical == expected.path
            and lexical.resolve(strict=True) == expected.path
            and getattr(module, "__speaker_source_sha256__", None)
            == expected.sha256
            and getattr(module, "__speaker_source_size_bytes__", None)
            == expected.size_bytes
        )
    except (OSError, RuntimeError, ValueError):
        return False


def _import_authenticated_nemo_service(
    config: ParakeetRealtimeEouConfig,
) -> ModuleType:
    """Import the two authenticated NeMo modules without optional Pipecat."""

    _optional_imports_are_absent()
    authenticated = _authenticated_nemo_pipecat_sources(config)
    packages: dict[str, ModuleType] = {}
    for name in _SYNTHETIC_PACKAGES:
        package = ModuleType(name)
        package.__package__ = name
        package.__path__ = [str(authenticated[name].path.parent)]
        packages[name] = package
    wget_stub = ModuleType("wget")
    wget_stub.download = _deny_wget_download  # type: ignore[attr-defined]

    imported: dict[str, object] = {}
    candidate: ModuleType | None = None
    verified_cloud: ModuleType | None = None
    import_succeeded = False
    cleanup_succeeded = True
    try:
        for name in _SYNTHETIC_PACKAGES:
            if name in sys.modules:
                raise ParakeetRealtimeEouAdapterError()
            sys.modules[name] = packages[name]
        if "wget" in sys.modules:
            raise ParakeetRealtimeEouAdapterError()
        sys.modules["wget"] = wget_stub

        utils = _execute_authenticated_module(
            _UTILS_MODULE,
            authenticated[_UTILS_MODULE],
            packages[_PIPECAT_NEMO_PACKAGE],
        )
        imported[_UTILS_MODULE] = utils
        if (
            not isinstance(utils, ModuleType)
            or sys.modules.get(_UTILS_MODULE) is not utils
            or not _exact_module_source(utils, authenticated[_UTILS_MODULE])
        ):
            raise ParakeetRealtimeEouAdapterError()

        service = _execute_authenticated_module(
            _SERVICE_MODULE,
            authenticated[_SERVICE_MODULE],
            packages[_PIPECAT_NEMO_PACKAGE],
        )
        imported[_SERVICE_MODULE] = service
        if (
            not isinstance(service, ModuleType)
            or sys.modules.get(_SERVICE_MODULE) is not service
            or not _exact_module_source(service, authenticated[_SERVICE_MODULE])
        ):
            raise ParakeetRealtimeEouAdapterError()
        if any(
            sys.modules.get(name) is not packages[name]
            or getattr(packages[name], "__path__", None)
            != [str(authenticated[name].path.parent)]
            for name in packages
        ):
            raise ParakeetRealtimeEouAdapterError()
        retained_names = {
            name
            for name in tuple(sys.modules)
            if name == _PIPECAT_PACKAGE
            or name.startswith(f"{_PIPECAT_PACKAGE}.")
        }
        if retained_names != {*packages, _UTILS_MODULE, _SERVICE_MODULE}:
            raise ParakeetRealtimeEouAdapterError()
        if sys.modules.get("wget") is not wget_stub:
            raise ParakeetRealtimeEouAdapterError()
        cloud = sys.modules.get("nemo.utils.cloud")
        if (
            not isinstance(cloud, ModuleType)
            or getattr(cloud, "wget", None) is not wget_stub
            or getattr(wget_stub, "download", None) is not _deny_wget_download
        ):
            raise ParakeetRealtimeEouAdapterError()
        verified_cloud = cloud
        candidate = service
        import_succeeded = True
    except Exception:
        import_succeeded = False

    created_prefix_modules = {
        name: module
        for name, module in tuple(sys.modules.items())
        if name == _PIPECAT_PACKAGE or name.startswith(f"{_PIPECAT_PACKAGE}.")
    }
    if import_succeeded:
        retained = {_UTILS_MODULE, _SERVICE_MODULE}
    else:
        retained = set()
    for name in sorted(
        created_prefix_modules,
        key=lambda candidate_name: candidate_name.count("."),
        reverse=True,
    ):
        module = created_prefix_modules[name]
        parent_name, _, child_name = name.rpartition(".")
        parent = created_prefix_modules.get(parent_name) or sys.modules.get(parent_name)
        if isinstance(parent, ModuleType):
            current_child = vars(parent).get(child_name)
            if current_child is module:
                try:
                    delattr(parent, child_name)
                except (AttributeError, RuntimeError, TypeError):
                    cleanup_succeeded = False
            elif current_child is not None:
                cleanup_succeeded = False
        if name in retained:
            continue
        current = sys.modules.get(name)
        if current is module:
            del sys.modules[name]
        elif current is not None:
            cleanup_succeeded = False
    if sys.modules.get("wget") is wget_stub:
        del sys.modules["wget"]
    else:
        cleanup_succeeded = False

    if import_succeeded and (
        sys.modules.get("nemo.utils.cloud") is not verified_cloud
        or getattr(verified_cloud, "wget", None) is not wget_stub
        or getattr(wget_stub, "download", None) is not _deny_wget_download
    ):
        cleanup_succeeded = False
    if import_succeeded:
        retained_names = {
            name
            for name in tuple(sys.modules)
            if name == _PIPECAT_PACKAGE
            or name.startswith(f"{_PIPECAT_PACKAGE}.")
        }
        if retained_names != {_UTILS_MODULE, _SERVICE_MODULE} or any(
            sys.modules.get(name) is not imported.get(name)
            or not _exact_module_source(sys.modules.get(name), authenticated[name])
            for name in (_UTILS_MODULE, _SERVICE_MODULE)
        ):
            cleanup_succeeded = False
    elif any(
        name == _PIPECAT_PACKAGE or name.startswith(f"{_PIPECAT_PACKAGE}.")
        for name in tuple(sys.modules)
    ):
        cleanup_succeeded = False
    if any(
        value is wget_stub or any(value is package for package in packages.values())
        for value in tuple(sys.modules.values())
    ):
        cleanup_succeeded = False

    if not import_succeeded or not cleanup_succeeded:
        remaining = {
            name: module
            for name, module in tuple(sys.modules.items())
            if name == _PIPECAT_PACKAGE
            or name.startswith(f"{_PIPECAT_PACKAGE}.")
        }
        for name, module in sorted(
            remaining.items(),
            key=lambda item: item[0].count("."),
            reverse=True,
        ):
            parent_name, _, child_name = name.rpartition(".")
            parent = remaining.get(parent_name) or sys.modules.get(parent_name)
            if isinstance(parent, ModuleType) and child_name in vars(parent):
                try:
                    delattr(parent, child_name)
                except (AttributeError, RuntimeError, TypeError):
                    pass
            if name in sys.modules:
                del sys.modules[name]
        raise ParakeetRealtimeEouAdapterError()
    if candidate is None:
        raise ParakeetRealtimeEouAdapterError()
    return candidate


@dataclass(frozen=True)
class ParakeetRealtimeEouReady:
    model_load_ms: float
    resources: ResourceUsage


@dataclass(frozen=True)
class ParakeetRealtimeEouPartial:
    after_samples: int
    text: str = field(repr=False)
    elapsed_ms: float
    decode_ms: float


@dataclass(frozen=True)
class ParakeetRealtimeEouCase:
    partials: tuple[ParakeetRealtimeEouPartial, ...] = field(repr=False)
    final: str = field(repr=False)
    elapsed_ms: float
    finalization_ms: float
    compute_ms: float
    deadline_misses: int
    max_backlog_ms: float
    resources: ResourceUsage
    model_padding_samples: int
    chunks_yielded: int
    source_samples: int
    source_samples_consumed: int
    declared_tail_samples: int
    tail_samples_consumed: int
    endpoint_reason: str
    native_endpoint: bool
    endpoint_probability: float | None
    endpoint_sample: int | None
    endpoint_latency_ms: float | None
    authoritative: bool


@dataclass(frozen=True)
class _ServiceResult:
    text: str = field(repr=False)
    is_final: bool
    endpoint_reason: str | None
    endpoint_probability: float | None


def _read_pcm(request: TranscribeRequest) -> array[float]:
    try:
        snapshot = read_regular_bounded(
            request.pcm.path,
            maximum_bytes=MAX_PCM_BYTES,
            expected_bytes=request.pcm.size_bytes,
        )
    except BoundedReadError:
        raise ParakeetRealtimeEouAdapterError() from None
    raw = snapshot.data
    if (
        len(raw) != request.pcm.size_bytes
        or hashlib.sha256(raw).hexdigest() != request.pcm.sha256
    ):
        raise ParakeetRealtimeEouAdapterError()
    samples = array("f")
    samples.frombytes(raw)
    if sys.byteorder != "little":
        samples.byteswap()
    if len(samples) != request.pcm.samples or any(
        not math.isfinite(sample) for sample in samples
    ):
        raise ParakeetRealtimeEouAdapterError()
    return samples


def _host_resources(torch_api: object | None) -> ResourceUsage:
    rss_mb: float | None = None
    threads: int | None = None
    try:
        status = Path("/proc/self/status").read_text(
            encoding="utf-8",
            errors="strict",
        )
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                fields = line.split()
                if len(fields) == 3 and fields[2] == "kB":
                    rss_mb = int(fields[1]) / 1024.0
            elif line.startswith("Threads:"):
                fields = line.split()
                if len(fields) == 2:
                    threads = int(fields[1])
    except (OSError, UnicodeError, ValueError):
        pass
    if rss_mb is None:
        try:
            import resource

            maximum_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            rss_mb = maximum_rss / (1024.0 if sys.platform != "darwin" else 1024**2)
        except (ImportError, OSError, ValueError):
            rss_mb = 0.0
    if threads is None:
        threads = max(1, threading.active_count())

    vram_mb: float | None = None
    if torch_api is not None:
        try:
            cuda = getattr(torch_api, "cuda")
            allocated = float(cuda.max_memory_allocated(0)) / _MEBIBYTE
            reserved = float(cuda.max_memory_reserved(0)) / _MEBIBYTE
            vram_mb = max(allocated, reserved)
        except (AttributeError, RuntimeError, TypeError, ValueError, OverflowError):
            raise ParakeetRealtimeEouAdapterError() from None
    if (
        not math.isfinite(rss_mb)
        or not 0.0 <= rss_mb <= MAX_RESOURCE_MB
        or not 1 <= threads <= 4096
        or (
            vram_mb is not None
            and (not math.isfinite(vram_mb) or not 0.0 <= vram_mb <= MAX_RESOURCE_MB)
        )
    ):
        raise ParakeetRealtimeEouAdapterError()
    return ResourceUsage(rss_mb=rss_mb, threads=threads, vram_mb=vram_mb)


def _flush_standard_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        flush = getattr(stream, "flush", None)
        if not callable(flush):
            raise ParakeetRealtimeEouAdapterError()
        try:
            flush()
        except (OSError, RuntimeError, ValueError):
            raise ParakeetRealtimeEouAdapterError() from None


def _restore_output_descriptors(
    saved: tuple[tuple[int, int], ...],
    sink_descriptor: int,
) -> None:
    failed = False
    for target, descriptor in saved:
        try:
            os.dup2(descriptor, target, inheritable=False)
        except OSError:
            failed = True
    for _target, descriptor in saved:
        try:
            os.close(descriptor)
        except OSError:
            failed = True
    if sink_descriptor >= 0:
        try:
            os.close(sink_descriptor)
        except OSError:
            failed = True
    try:
        _flush_standard_streams()
    except ParakeetRealtimeEouAdapterError:
        failed = True
    if failed:
        raise ParakeetRealtimeEouAdapterError()


@contextmanager
def _suppressed_process_output() -> Iterator[None]:
    """Keep third-party fd output away from the worker JSONL protocol."""

    saved: list[tuple[int, int]] = []
    sink_descriptor = -1
    try:
        _flush_standard_streams()
        for target in (1, 2):
            descriptor = os.dup(target)
            os.set_inheritable(descriptor, False)
            saved.append((target, descriptor))
        flags = os.O_WRONLY | getattr(os, "O_CLOEXEC", 0)
        sink_descriptor = os.open(os.devnull, flags)
        for target, _descriptor in saved:
            os.dup2(sink_descriptor, target, inheritable=False)
    except (OSError, RuntimeError, ValueError, ParakeetRealtimeEouAdapterError):
        try:
            _restore_output_descriptors(tuple(saved), sink_descriptor)
        except ParakeetRealtimeEouAdapterError:
            pass
        raise ParakeetRealtimeEouAdapterError() from None
    try:
        yield
    finally:
        suppressed_flush_failed = False
        try:
            _flush_standard_streams()
        except ParakeetRealtimeEouAdapterError:
            suppressed_flush_failed = True
        _restore_output_descriptors(tuple(saved), sink_descriptor)
        if suppressed_flush_failed:
            raise ParakeetRealtimeEouAdapterError()


def _suppressed_call(function: Callable[[], _T]) -> _T:
    with _suppressed_process_output():
        return function()


def _cuda_synchronize(torch_api: object) -> None:
    try:
        synchronize = getattr(getattr(torch_api, "cuda"), "synchronize")
        if not callable(synchronize):
            raise ParakeetRealtimeEouAdapterError()
        synchronize(0)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        raise ParakeetRealtimeEouAdapterError() from None


def _bind_torch_threads(torch_api: object) -> None:
    try:
        set_threads = getattr(torch_api, "set_num_threads")
        set_interop_threads = getattr(torch_api, "set_num_interop_threads")
        get_threads = getattr(torch_api, "get_num_threads")
        get_interop_threads = getattr(torch_api, "get_num_interop_threads")
        if not all(
            callable(function)
            for function in (
                set_threads,
                set_interop_threads,
                get_threads,
                get_interop_threads,
            )
        ):
            raise ParakeetRealtimeEouAdapterError()
        set_threads(1)
        set_interop_threads(1)
        intra_threads = get_threads()
        interop_threads = get_interop_threads()
        if (
            type(intra_threads) is not int
            or intra_threads != 1
            or type(interop_threads) is not int
            or interop_threads != 1
        ):
            raise ParakeetRealtimeEouAdapterError()
    except (AttributeError, RuntimeError, TypeError, ValueError):
        raise ParakeetRealtimeEouAdapterError() from None


def _verify_exact_model(path: Path) -> None:
    expected_sha256, expected_size = PARAKEET_REALTIME_EOU_MODEL_RECEIPT
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size <= 0
    ):
        raise ParakeetRealtimeEouAdapterError()
    try:
        digest = hash_regular_bounded(
            path,
            maximum_bytes=expected_size,
            expected_bytes=expected_size,
        )
    except (BoundedReadError, OSError, RuntimeError, TypeError, ValueError):
        raise ParakeetRealtimeEouAdapterError() from None
    if (
        digest.path != path
        or digest.sha256 != expected_sha256
        or digest.size_bytes != expected_size
    ):
        raise ParakeetRealtimeEouAdapterError()


def _release_torch(torch_api: object) -> None:
    def release() -> None:
        cuda = getattr(torch_api, "cuda")
        synchronize = getattr(cuda, "synchronize", None)
        empty_cache = getattr(cuda, "empty_cache", None)
        if callable(synchronize):
            synchronize(0)
        if callable(empty_cache):
            empty_cache()

    try:
        _suppressed_call(release)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        raise ParakeetRealtimeEouAdapterError() from None


def _verify_loaded_service(
    service: object,
    torch_api: object,
    config: ParakeetRealtimeEouConfig,
) -> None:
    model = getattr(service, "asr_model", None)
    parameters = getattr(model, "parameters", None)
    expected_dtype = getattr(torch_api, "float32", None)
    if (
        not callable(getattr(service, "transcribe", None))
        or not callable(getattr(service, "reset_state", None))
        or getattr(service, "device", None) != config.device
        or getattr(service, "decoder_type", None) != "rnnt"
        or getattr(service, "eou_string", None) != config.eou_token
        or getattr(service, "eob_string", None) != config.eob_token
        or expected_dtype is None
        or not callable(parameters)
    ):
        raise ParakeetRealtimeEouAdapterError()
    try:
        model_parameters = iter(parameters())
        first = next(model_parameters)
        if (
            getattr(first, "dtype", None) != expected_dtype
            or str(getattr(first, "device", "")) != config.device
        ):
            raise ParakeetRealtimeEouAdapterError()
        for parameter in model_parameters:
            if (
                getattr(parameter, "dtype", None) != expected_dtype
                or str(getattr(parameter, "device", "")) != config.device
            ):
                raise ParakeetRealtimeEouAdapterError()
    except (StopIteration, TypeError, RuntimeError, ValueError):
        raise ParakeetRealtimeEouAdapterError() from None


def _load_service(
    model_path: Path,
    config: ParakeetRealtimeEouConfig,
) -> tuple[object, object]:
    """Import the exact candidate closure and restore only the local model."""

    try:
        if metadata.version(_NEMO_DISTRIBUTION) != config.nemo_version:
            raise ParakeetRealtimeEouAdapterError()
        if metadata.version(_TORCH_DISTRIBUTION) != config.torch_version:
            raise ParakeetRealtimeEouAdapterError()
        if metadata.version(_NUMPY_DISTRIBUTION) != config.numpy_version:
            raise ParakeetRealtimeEouAdapterError()
        with _suppressed_process_output():
            torch = importlib.import_module("torch")
            if (
                getattr(torch, "__version__", None) != config.torch_version
                or getattr(getattr(torch, "version", None), "cuda", None)
                != config.cuda_version
            ):
                raise ParakeetRealtimeEouAdapterError()
            _bind_torch_threads(torch)
            module = _import_authenticated_nemo_service(config)
            cuda = getattr(torch, "cuda", None)
            service_type = getattr(module, "NemoStreamingASRService", None)
            if (
                cuda is None
                or not callable(getattr(cuda, "is_available", None))
                or not cuda.is_available()
                or not callable(
                    getattr(cuda, "set_per_process_memory_fraction", None)
                )
                or not callable(service_type)
            ):
                raise ParakeetRealtimeEouAdapterError()
            cuda.set_per_process_memory_fraction(_VRAM_FRACTION, 0)
            _cuda_synchronize(torch)
            service = service_type(
                model=str(model_path),
                att_context_size=[
                    config.attention_context_left,
                    config.attention_context_right,
                ],
                device=config.device,
                eou_string=config.eou_token,
                eob_string=config.eob_token,
                decoder_type="rnnt",
                sample_rate=config.sample_rate,
                use_amp=config.use_amp,
                chunk_size_in_secs=(
                    config.native_chunk_samples / config.sample_rate
                ),
            )
            _cuda_synchronize(torch)
            _verify_loaded_service(service, torch, config)
    except Exception:
        raise ParakeetRealtimeEouAdapterError() from None
    return service, torch


def _pcm16_bytes(samples: list[float]) -> bytes:
    payload = bytearray(len(samples) * 2)
    for index, raw in enumerate(samples):
        bounded = min(32767, max(-32768, int(float(raw) * 32768.0)))
        struct.pack_into("<h", payload, index * 2, bounded)
    return bytes(payload)


def _service_result(value: object, config: ParakeetRealtimeEouConfig) -> _ServiceResult:
    text = getattr(value, "text", None)
    is_final = getattr(value, "is_final", None)
    eou_probability = getattr(value, "eou_prob", None)
    eob_probability = getattr(value, "eob_prob", None)
    if not isinstance(text, str) or len(text) > MAX_HYPOTHESIS_CHARS:
        raise ParakeetRealtimeEouAdapterError()
    if type(is_final) is not bool:
        raise ParakeetRealtimeEouAdapterError()
    marker_positions: dict[str, int] = {
        "eou": text.find(config.eou_token),
        "eob": text.find(config.eob_token),
    }
    probabilities = {
        "eou": eou_probability,
        "eob": eob_probability,
    }
    present = [reason for reason, position in marker_positions.items() if position >= 0]
    if not is_final:
        if present or eou_probability is not None or eob_probability is not None:
            raise ParakeetRealtimeEouAdapterError()
        return _ServiceResult(
            text=text,
            is_final=False,
            endpoint_reason=None,
            endpoint_probability=None,
        )
    if not present:
        raise ParakeetRealtimeEouAdapterError()
    validated_probabilities: dict[str, float] = {}
    for candidate, probability in probabilities.items():
        marker_present = marker_positions[candidate] >= 0
        if probability is None:
            if marker_present:
                raise ParakeetRealtimeEouAdapterError()
            continue
        if (
            not marker_present
            or isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(float(probability))
            or not 0.0 <= float(probability) <= 1.0
        ):
            raise ParakeetRealtimeEouAdapterError()
        validated_probabilities[candidate] = float(probability)
    reason = min(present, key=lambda candidate: marker_positions[candidate])
    before = text[: marker_positions[reason]]
    return _ServiceResult(
        text=before,
        is_final=True,
        endpoint_reason=reason,
        endpoint_probability=validated_probabilities[reason],
    )


class ParakeetRealtimeEouAdapter:
    """One batch-one native-EOU stream with explicit terminal consumption."""

    def __init__(
        self,
        model_path: Path,
        *,
        config: ParakeetRealtimeEouConfig,
        service: object | None = None,
        torch_api: object | None = None,
        clock: Callable[[], float] = time.perf_counter,
        sleeper: Callable[[float], None] = time.sleep,
        resource_reader: Callable[[object | None], ResourceUsage] = _host_resources,
    ) -> None:
        if not isinstance(config, ParakeetRealtimeEouConfig):
            raise ParakeetRealtimeEouAdapterError()
        try:
            selected_model = Path(model_path).resolve(strict=True)
        except (OSError, RuntimeError):
            raise ParakeetRealtimeEouAdapterError() from None
        if (
            not selected_model.is_file()
            or selected_model.name != config.model_filename
            or (service is None) != (torch_api is None)
        ):
            raise ParakeetRealtimeEouAdapterError()
        self.config = config
        self.model_path = selected_model
        self._clock = clock
        self._sleeper = sleeper
        self._resource_reader = resource_reader
        self._closed = False
        self._poisoned = False
        self._state_clean = True
        self._active_stream_id: str | None = None
        load_started = self._now()
        candidate_service: object
        candidate_torch: object
        if service is None:
            _verify_exact_model(selected_model)
            loaded_torch: object | None = None
            try:
                candidate_service, loaded_torch = _load_service(selected_model, config)
                _verify_exact_model(selected_model)
            except Exception:
                if loaded_torch is not None:
                    try:
                        _release_torch(loaded_torch)
                    except ParakeetRealtimeEouAdapterError:
                        pass
                raise ParakeetRealtimeEouAdapterError() from None
            candidate_torch = loaded_torch
        else:
            candidate_service, candidate_torch = service, torch_api
        if (
            not callable(getattr(candidate_service, "transcribe", None))
            or not callable(getattr(candidate_service, "reset_state", None))
        ):
            raise ParakeetRealtimeEouAdapterError()
        self._service = candidate_service
        self._torch = candidate_torch
        load_finished = self._now()
        self.ready = ParakeetRealtimeEouReady(
            model_load_ms=self._duration_ms(load_started, load_finished),
            resources=self._resources(),
        )

    def __enter__(self) -> ParakeetRealtimeEouAdapter:
        if self._closed:
            raise ParakeetRealtimeEouAdapterError()
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()

    def _now(self) -> float:
        try:
            value = float(self._clock())
        except (TypeError, ValueError, OverflowError):
            raise ParakeetRealtimeEouAdapterError() from None
        if not math.isfinite(value):
            raise ParakeetRealtimeEouAdapterError()
        return value

    @staticmethod
    def _duration_ms(start: float, end: float) -> float:
        duration = end - start
        if not math.isfinite(duration) or duration < 0.0:
            raise ParakeetRealtimeEouAdapterError()
        return duration * 1000.0

    def _timed(self, function: Callable[[], _T]) -> tuple[_T, float]:
        with _suppressed_process_output():
            self._cuda_synchronize()
            started = self._now()
            result = function()
            self._cuda_synchronize()
            finished = self._now()
        return result, self._duration_ms(started, finished)

    def _cuda_synchronize(self) -> None:
        if self._torch is None:
            raise ParakeetRealtimeEouAdapterError()
        _cuda_synchronize(self._torch)

    def _resources(self) -> ResourceUsage:
        try:
            value = self._resource_reader(self._torch)
        except Exception:
            raise ParakeetRealtimeEouAdapterError() from None
        if not isinstance(value, ResourceUsage):
            raise ParakeetRealtimeEouAdapterError()
        return value

    def _reset(self, stream_id: str) -> None:
        if not isinstance(stream_id, str) or not stream_id:
            self._poisoned = True
            raise ParakeetRealtimeEouAdapterError()
        try:
            _suppressed_call(
                lambda: self._service.reset_state(stream_id=stream_id)
            )
        except Exception:
            self._poisoned = True
            raise ParakeetRealtimeEouAdapterError() from None
        self._state_clean = True
        self._active_stream_id = None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not self._state_clean and not self._poisoned:
            try:
                if self._active_stream_id is None:
                    raise ParakeetRealtimeEouAdapterError()
                self._reset(self._active_stream_id)
            except ParakeetRealtimeEouAdapterError:
                pass
        torch_api = self._torch
        self._service = None
        self._torch = None
        if torch_api is not None:
            try:
                _release_torch(torch_api)
            except Exception:
                self._poisoned = True

    def transcribe(
        self,
        request: TranscribeRequest,
        *,
        emit_partial: Callable[[int, ParakeetRealtimeEouPartial], None],
    ) -> ParakeetRealtimeEouCase:
        if (
            self._closed
            or self._poisoned
            or not self._state_clean
            or request.protocol_version != NATIVE_ENDPOINT_PROTOCOL_VERSION
            or request.stream.chunk_samples != self.config.native_chunk_samples
            or request.stream.tail_padding_samples
            > self.config.maximum_tail_padding_samples
            or request.pcm.sample_rate != self.config.sample_rate
            or request.pcm.samples + request.stream.tail_padding_samples
            < 2 * self.config.native_chunk_samples
        ):
            raise ParakeetRealtimeEouAdapterError()
        source = _read_pcm(request)
        declared_source = request.pcm.samples
        declared_tail = request.stream.tail_padding_samples
        declared_total = declared_source + declared_tail
        if declared_total > MAX_STREAM_SAMPLES:
            raise ParakeetRealtimeEouAdapterError()

        case_started = self._now()
        stream_started = case_started
        self._state_clean = False
        self._active_stream_id = request.request_id
        partials: list[ParakeetRealtimeEouPartial] = []
        cumulative_text = ""
        latest_text = ""
        compute_ms = 0.0
        finalization_ms = 0.0
        deadline_misses = 0
        max_backlog_ms = 0.0
        samples_seen = 0
        source_seen = 0
        tail_seen = 0
        chunks = 0
        model_padding_samples = 0
        endpoint_reason = "tail_exhausted"
        endpoint_probability: float | None = None
        endpoint_sample: int | None = None
        endpoint_latency_ms: float | None = None
        native_endpoint = False
        partial_interval_samples = (
            request.stream.partial_interval_ms * self.config.sample_rate + 999
        ) // 1000
        next_partial_samples = partial_interval_samples

        try:
            for offset in range(
                0,
                declared_total,
                self.config.native_chunk_samples,
            ):
                actual_count = min(
                    self.config.native_chunk_samples,
                    declared_total - offset,
                )
                if request.stream.pace == "realtime":
                    target = stream_started + offset / self.config.sample_rate
                    before_wait = self._now()
                    if before_wait < target:
                        self._sleeper(target - before_wait)
                    if self._now() < target:
                        raise ParakeetRealtimeEouAdapterError()

                source_start = min(offset, declared_source)
                source_end = min(offset + actual_count, declared_source)
                source_count = max(0, source_end - source_start)
                tail_count = actual_count - source_count
                chunk = list(source[source_start:source_end])
                chunk.extend([0.0] * tail_count)
                model_padding_samples = (
                    self.config.native_chunk_samples - actual_count
                )
                chunk.extend([0.0] * model_padding_samples)
                if len(chunk) != self.config.native_chunk_samples:
                    raise ParakeetRealtimeEouAdapterError()

                value, decode_ms = self._timed(
                    lambda chunk=chunk: self._service.transcribe(
                        _pcm16_bytes(chunk),
                        stream_id=request.request_id,
                    )
                )
                compute_ms += decode_ms
                chunks += 1
                source_seen += source_count
                tail_seen += tail_count
                samples_seen += actual_count
                parsed = _service_result(value, self.config)
                if len(parsed.text) > MAX_HYPOTHESIS_CHARS - len(cumulative_text):
                    raise ParakeetRealtimeEouAdapterError()
                cumulative_text += parsed.text
                latest_text = cumulative_text.strip()

                if request.stream.pace == "realtime":
                    model_samples_seen = samples_seen + model_padding_samples
                    deadline = (
                        stream_started
                        + model_samples_seen / self.config.sample_rate
                    )
                    now = self._now()
                    backlog_ms = self._duration_ms(deadline, max(deadline, now))
                    max_backlog_ms = max(max_backlog_ms, backlog_ms)
                    if backlog_ms > 0.0:
                        deadline_misses += 1

                if parsed.is_final:
                    native_endpoint = True
                    endpoint_reason = str(parsed.endpoint_reason)
                    endpoint_probability = parsed.endpoint_probability
                    endpoint_sample = samples_seen + model_padding_samples
                    endpoint_latency_ms = (
                        (tail_seen + model_padding_samples)
                        * 1000.0
                        / self.config.sample_rate
                        if source_seen == declared_source
                        else None
                    )
                    finalization_ms = decode_ms
                    # NVIDIA's released service resets immediately on EOU/EOB.
                    self._state_clean = True
                    self._active_stream_id = None
                    break

                if samples_seen >= next_partial_samples and latest_text:
                    if len(partials) >= MAX_PARTIALS_PER_CASE:
                        raise ParakeetRealtimeEouAdapterError()
                    if not partials or partials[-1].text != latest_text:
                        partial = ParakeetRealtimeEouPartial(
                            after_samples=samples_seen,
                            text=latest_text,
                            elapsed_ms=self._duration_ms(case_started, self._now()),
                            decode_ms=decode_ms,
                        )
                        emit_partial(len(partials), partial)
                        partials.append(partial)
                    while next_partial_samples <= samples_seen:
                        next_partial_samples += partial_interval_samples

            if not native_endpoint:
                if (
                    source_seen != declared_source
                    or tail_seen != declared_tail
                    or samples_seen != declared_total
                ):
                    raise ParakeetRealtimeEouAdapterError()
                _, reset_ms = self._timed(
                    lambda: self._reset(request.request_id)
                )
                compute_ms += reset_ms
                finalization_ms = decode_ms + reset_ms
        except ParakeetRealtimeEouAdapterError:
            self._poisoned = True
            raise
        except Exception:
            self._poisoned = True
            raise ParakeetRealtimeEouAdapterError() from None

        finished = self._now()
        authoritative = (
            native_endpoint
            and endpoint_reason == "eou"
            and source_seen == declared_source
        )
        if not authoritative:
            endpoint_latency_ms = None
        return ParakeetRealtimeEouCase(
            partials=tuple(partials),
            final=latest_text,
            elapsed_ms=self._duration_ms(case_started, finished),
            finalization_ms=finalization_ms,
            compute_ms=compute_ms,
            deadline_misses=deadline_misses,
            max_backlog_ms=max_backlog_ms,
            resources=self._resources(),
            model_padding_samples=model_padding_samples,
            chunks_yielded=chunks,
            source_samples=declared_source,
            source_samples_consumed=source_seen,
            declared_tail_samples=declared_tail,
            tail_samples_consumed=tail_seen,
            endpoint_reason=endpoint_reason,
            native_endpoint=native_endpoint,
            endpoint_probability=endpoint_probability,
            endpoint_sample=endpoint_sample,
            endpoint_latency_ms=endpoint_latency_ms,
            authoritative=authoritative,
        )


__all__ = [
    "ParakeetRealtimeEouAdapter",
    "ParakeetRealtimeEouAdapterError",
    "ParakeetRealtimeEouCase",
    "ParakeetRealtimeEouPartial",
    "ParakeetRealtimeEouReady",
]
