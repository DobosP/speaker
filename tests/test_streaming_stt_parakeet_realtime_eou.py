from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import os
from pathlib import Path
import struct
import sys
from types import ModuleType, SimpleNamespace

import pytest

import tools.streaming_stt.adapters.parakeet_realtime_eou as adapter_module
from tools.streaming_stt.adapters.parakeet_realtime_eou import (
    ParakeetRealtimeEouAdapter,
    ParakeetRealtimeEouAdapterError,
)
from tools.streaming_stt.manifest import ManifestError, ParakeetRealtimeEouConfig
from tools.streaming_stt.protocol import (
    NATIVE_ENDPOINT_PROTOCOL_VERSION,
    PcmInput,
    ResourceUsage,
    StreamConfig,
    TranscribeRequest,
)


@dataclass
class _Result:
    text: str
    is_final: bool
    eou_prob: float | None = None
    eob_prob: float | None = None


class _Service:
    def __init__(self, results, *, events=None):
        self.results = list(results)
        self.chunks: list[bytes] = []
        self.stream_ids: list[str] = []
        self.reset_ids: list[str] = []
        self.reset_calls = 0
        self.events = events if events is not None else []

    def transcribe(self, audio, *, stream_id):
        self.events.append(("transcribe", stream_id))
        self.chunks.append(audio)
        self.stream_ids.append(stream_id)
        if not self.results:
            raise RuntimeError
        result = self.results.pop(0)
        if result.is_final:
            # The real NeMo 2.7.3 service resets internally on EOU/EOB.
            self.reset_calls += 1
            self.reset_ids.append(stream_id)
        return result

    def reset_state(self, *, stream_id):
        self.events.append(("reset", stream_id))
        self.reset_calls += 1
        self.reset_ids.append(stream_id)


class _StatefulService(_Service):
    def __init__(self, results, *, events=None):
        super().__init__(results, events=events)
        self.clean = True
        self.active_stream_id = None
        self.stream_starts: list[tuple[str, bool]] = []

    def transcribe(self, audio, *, stream_id):
        if self.active_stream_id is None:
            self.stream_starts.append((stream_id, self.clean))
            if not self.clean:
                raise RuntimeError
            self.active_stream_id = stream_id
        elif self.active_stream_id != stream_id:
            raise RuntimeError
        result = super().transcribe(audio, stream_id=stream_id)
        self.clean = False
        if result.is_final:
            self.clean = True
            self.active_stream_id = None
        return result

    def reset_state(self, *, stream_id):
        super().reset_state(stream_id=stream_id)
        self.clean = True
        self.active_stream_id = None


class _Cuda:
    def __init__(self, events):
        self.events = events

    def synchronize(self, device):
        self.events.append(("synchronize", device))


class _Torch:
    def __init__(self, events):
        self.cuda = _Cuda(events)


class _Clock:
    def __init__(self):
        self.value = 1.0

    def __call__(self):
        self.value += 0.001
        return self.value


def _config() -> ParakeetRealtimeEouConfig:
    return ParakeetRealtimeEouConfig()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("python_version", "3.12.4"),
        ("nemo_version", "2.7.2"),
        ("torch_version", "2.7.1"),
        ("cuda_version", "12.5"),
        ("numpy_version", "2.2.5"),
        ("wheel_lock_sha256", "0" * 64),
        ("runtime_content_sha256", "0" * 64),
        ("runtime_file_count", 49_601),
        ("runtime_total_size_bytes", 6_430_910_099),
        ("runtime_maximum_file_bytes", 984_633_130),
    ],
)
def test_runtime_identity_is_repository_frozen(field, value):
    with pytest.raises(ManifestError):
        replace(_config(), **{field: value})


def _request(
    tmp_path: Path,
    *,
    source_samples: int = 2_560,
    tail_samples: int = 2_560,
) -> TranscribeRequest:
    values = [0.25] * source_samples
    payload = b"".join(struct.pack("<f", value) for value in values)
    path = tmp_path / "case.f32le"
    path.write_bytes(payload)
    return TranscribeRequest(
        request_id="case-r0",
        pcm=PcmInput(
            path=path.resolve(),
            sha256=hashlib.sha256(payload).hexdigest(),
            samples=source_samples,
        ),
        stream=StreamConfig(
            chunk_samples=1_280,
            pace="burst",
            partial_interval_ms=80,
            tail_padding_samples=tail_samples,
        ),
        protocol_version=NATIVE_ENDPOINT_PROTOCOL_VERSION,
    )


def _adapter(
    tmp_path: Path,
    service: _Service,
    *,
    events=None,
) -> ParakeetRealtimeEouAdapter:
    model = tmp_path / "parakeet_realtime_eou_120m-v1.nemo"
    model.write_bytes(b"model-fixture")
    selected_events = service.events if events is None else events
    return ParakeetRealtimeEouAdapter(
        model,
        config=_config(),
        service=service,
        torch_api=_Torch(selected_events),
        clock=_Clock(),
        resource_reader=lambda _torch: ResourceUsage(
            rss_mb=128.0,
            threads=1,
            vram_mb=512.0,
        ),
    )


def _bind_fake_load(
    monkeypatch,
    *,
    model_fault: str | None = None,
    constructor_failure: bool = False,
):
    events = []
    captured = {}
    float32_dtype = object()

    class Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def set_per_process_memory_fraction(fraction, device):
            events.append(("memory_fraction", fraction, device))

        @staticmethod
        def synchronize(device):
            events.append(("synchronize", device))

    class Torch:
        __version__ = "2.7.1+cu126"
        version = SimpleNamespace(cuda="12.6")
        cuda = Cuda()
        float32 = float32_dtype
        intra_threads = 4
        interop_threads = 4

        @classmethod
        def set_num_threads(cls, value):
            events.append(("set_num_threads", value))
            cls.intra_threads = value

        @classmethod
        def set_num_interop_threads(cls, value):
            events.append(("set_num_interop_threads", value))
            cls.interop_threads = value

        @classmethod
        def get_num_threads(cls):
            return cls.intra_threads

        @classmethod
        def get_num_interop_threads(cls):
            return cls.interop_threads

    class Model:
        def parameters(self):
            if model_fault == "empty":
                return iter(())
            return iter(
                (
                    SimpleNamespace(
                        dtype=(
                            object()
                            if model_fault == "dtype"
                            else float32_dtype
                        ),
                        device=("cpu" if model_fault == "device" else "cuda:0"),
                    ),
                )
            )

    class Service:
        def __init__(self, **kwargs):
            os.write(1, b"hidden-constructor-stdout\n")
            os.write(2, b"hidden-constructor-stderr\n")
            events.append(("construct", kwargs["model"]))
            if constructor_failure:
                raise RuntimeError
            captured.update(kwargs)
            self.device = kwargs["device"]
            self.decoder_type = (
                "ctc" if model_fault == "decoder" else kwargs["decoder_type"]
            )
            self.eou_string = kwargs["eou_string"]
            self.eob_string = kwargs["eob_string"]
            self.asr_model = Model()

        @staticmethod
        def transcribe(_audio, *, stream_id):
            raise AssertionError(stream_id)

        @staticmethod
        def reset_state(*, stream_id):
            raise AssertionError(stream_id)

    torch_api = Torch()

    def import_module(name):
        os.write(1, f"hidden-import-stdout:{name}\n".encode())
        os.write(2, f"hidden-import-stderr:{name}\n".encode())
        events.append(("import", name))
        if name == "torch":
            return torch_api
        if name == adapter_module._SERVICE_MODULE:
            return SimpleNamespace(NemoStreamingASRService=Service)
        raise ImportError

    versions = {
        "nemo-toolkit": "2.7.3",
        "torch": "2.7.1+cu126",
        "numpy": "2.2.6",
    }
    monkeypatch.setattr(adapter_module.metadata, "version", versions.__getitem__)
    monkeypatch.setattr(adapter_module.importlib, "import_module", import_module)
    monkeypatch.setattr(
        adapter_module,
        "_import_authenticated_nemo_service",
        lambda _config: import_module(adapter_module._SERVICE_MODULE),
    )
    return events, captured, torch_api


def _bind_authenticated_import(
    tmp_path: Path,
    monkeypatch,
    *,
    bad_digest_module: str | None = None,
    redirected_module: str | None = None,
    installed_distribution: str | None = None,
    fail_service_import: bool = False,
    partial_side_effect: bool = False,
):
    root = (tmp_path / "site-packages").resolve()
    root.mkdir()
    original_receipts = adapter_module._NEMO_PIPECAT_FILES
    source_by_module = {
        name: f"# authenticated fixture: {index}\n".encode()
        for index, name in enumerate(original_receipts)
    }
    fixture_receipts = {
        name: (
            receipt[0],
            (
                "0" * 64
                if name == bad_digest_module
                else hashlib.sha256(source_by_module[name]).hexdigest()
            ),
            len(source_by_module[name]),
        )
        for name, receipt in original_receipts.items()
    }
    monkeypatch.setattr(adapter_module, "_NEMO_PIPECAT_FILES", fixture_receipts)
    relative_by_module = {
        name: receipt[0] for name, receipt in fixture_receipts.items()
    }
    for name, relative in relative_by_module.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source_by_module[name])
    redirected = (tmp_path / "redirected.py").resolve()
    redirected.write_bytes(b"redirected")

    class Distribution:
        version = "2.7.3"

        @staticmethod
        def locate_file(relative):
            if (
                redirected_module is not None
                and relative == relative_by_module[redirected_module]
            ):
                return redirected
            return root / relative

    def distribution(name):
        if name == adapter_module._NEMO_DISTRIBUTION:
            return Distribution()
        if name == installed_distribution:
            return SimpleNamespace(version="installed")
        raise adapter_module.metadata.PackageNotFoundError(name)

    read_calls = []
    original_read = adapter_module.read_regular_bounded

    def observed_read(path, *, maximum_bytes, expected_bytes):
        candidate = Path(path)
        match = next(
            (
                (name, receipt)
                for name, receipt in fixture_receipts.items()
                if root / receipt[0] == candidate
            ),
            None,
        )
        assert match is not None
        name, receipt = match
        assert maximum_bytes == receipt[2]
        assert expected_bytes == receipt[2]
        read_calls.append((name, receipt[1], receipt[2]))
        return original_read(
            candidate,
            maximum_bytes=maximum_bytes,
            expected_bytes=expected_bytes,
        )

    cloud = ModuleType("nemo.utils.cloud")
    voice_agent = ModuleType("nemo.agents.voice_agent")
    monkeypatch.setitem(sys.modules, "nemo.utils.cloud", cloud)
    monkeypatch.setitem(sys.modules, "nemo.agents.voice_agent", voice_agent)
    state = {"packages": None, "utils": None, "service": None}

    def execute_module(name, source, parent):
        root_package = sys.modules[adapter_module._PIPECAT_PACKAGE]
        services_package = sys.modules[adapter_module._PIPECAT_SERVICES_PACKAGE]
        nemo_package = sys.modules[adapter_module._PIPECAT_NEMO_PACKAGE]
        voice_agent.pipecat = root_package
        root_package.services = services_package
        services_package.nemo = nemo_package
        state["packages"] = (
            root_package,
            services_package,
            nemo_package,
        )
        module = ModuleType(name)
        module.__file__ = str(source.path)
        module.__package__ = name.rpartition(".")[0]
        module.__speaker_source_sha256__ = source.sha256
        module.__speaker_source_size_bytes__ = source.size_bytes
        monkeypatch.setitem(sys.modules, name, module)
        setattr(parent, name.rsplit(".", 1)[-1], module)
        if name == adapter_module._UTILS_MODULE:
            cloud.wget = sys.modules["wget"]
            state["utils"] = module
            return module
        if name == adapter_module._SERVICE_MODULE:
            if fail_service_import:
                if partial_side_effect:
                    side_effect_name = (
                        adapter_module._PIPECAT_NEMO_PACKAGE
                        + ".partial_side_effect"
                    )
                    side_effect = ModuleType(side_effect_name)
                    monkeypatch.setitem(sys.modules, side_effect_name, side_effect)
                    nemo_package.partial_side_effect = side_effect
                raise ImportError
            module.NemoStreamingASRService = object
            state["service"] = module
            return module
        raise ImportError

    monkeypatch.setattr(adapter_module.metadata, "distribution", distribution)
    monkeypatch.setattr(adapter_module.util, "find_spec", lambda _name: None)
    monkeypatch.setattr(adapter_module, "read_regular_bounded", observed_read)
    monkeypatch.setattr(adapter_module, "_execute_authenticated_module", execute_module)
    return cloud, voice_agent, state, read_calls


def test_authenticated_optional_import_retains_only_exact_modules(
    tmp_path,
    monkeypatch,
):
    cloud, voice_agent, state, hash_calls = _bind_authenticated_import(
        tmp_path,
        monkeypatch,
    )

    service = adapter_module._import_authenticated_nemo_service(_config())

    assert service is state["service"]
    assert sys.modules[adapter_module._UTILS_MODULE] is state["utils"]
    assert sys.modules[adapter_module._SERVICE_MODULE] is service
    assert all(name not in sys.modules for name in adapter_module._SYNTHETIC_PACKAGES)
    assert "wget" not in sys.modules
    assert not hasattr(voice_agent, "pipecat")
    assert hash_calls == [
        (name, receipt[1], receipt[2])
        for name, receipt in adapter_module._NEMO_PIPECAT_FILES.items()
    ]
    assert cloud.wget.download is adapter_module._deny_wget_download


@pytest.mark.parametrize(
    ("bad_digest_module", "redirected_module"),
    [
        (adapter_module._SERVICE_MODULE, None),
        (None, adapter_module._UTILS_MODULE),
    ],
)
def test_authenticated_optional_import_rejects_changed_file_or_path(
    tmp_path,
    monkeypatch,
    bad_digest_module,
    redirected_module,
):
    _bind_authenticated_import(
        tmp_path,
        monkeypatch,
        bad_digest_module=bad_digest_module,
        redirected_module=redirected_module,
    )

    with pytest.raises(ParakeetRealtimeEouAdapterError):
        adapter_module._import_authenticated_nemo_service(_config())

    assert all(name not in sys.modules for name in adapter_module._SYNTHETIC_PACKAGES)
    assert "wget" not in sys.modules


@pytest.mark.parametrize(
    "module_name",
    ("pipecat", "wget", adapter_module._SERVICE_MODULE),
)
def test_authenticated_optional_import_rejects_preloaded_modules_without_removal(
    tmp_path,
    monkeypatch,
    module_name,
):
    _bind_authenticated_import(tmp_path, monkeypatch)
    existing = ModuleType(module_name)
    monkeypatch.setitem(sys.modules, module_name, existing)

    with pytest.raises(ParakeetRealtimeEouAdapterError):
        adapter_module._import_authenticated_nemo_service(_config())

    assert sys.modules[module_name] is existing


@pytest.mark.parametrize("distribution_name", adapter_module._OPTIONAL_DISTRIBUTIONS)
def test_authenticated_optional_import_rejects_installed_optional_distribution(
    tmp_path,
    monkeypatch,
    distribution_name,
):
    _bind_authenticated_import(
        tmp_path,
        monkeypatch,
        installed_distribution=distribution_name,
    )

    with pytest.raises(ParakeetRealtimeEouAdapterError):
        adapter_module._import_authenticated_nemo_service(_config())

    assert all(name not in sys.modules for name in adapter_module._SYNTHETIC_PACKAGES)
    assert "wget" not in sys.modules


def test_authenticated_optional_import_cleans_up_after_service_import_failure(
    tmp_path,
    monkeypatch,
):
    cloud, voice_agent, state, _hash_calls = _bind_authenticated_import(
        tmp_path,
        monkeypatch,
        fail_service_import=True,
    )

    with pytest.raises(ParakeetRealtimeEouAdapterError):
        adapter_module._import_authenticated_nemo_service(_config())

    assert adapter_module._UTILS_MODULE not in sys.modules
    assert adapter_module._SERVICE_MODULE not in sys.modules
    assert all(name not in sys.modules for name in adapter_module._SYNTHETIC_PACKAGES)
    assert "wget" not in sys.modules
    assert not hasattr(voice_agent, "pipecat")
    root_package, services_package, nemo_package = state["packages"]
    assert not hasattr(root_package, "services")
    assert not hasattr(services_package, "nemo")
    assert not hasattr(nemo_package, "utils")
    assert cloud.wget.download is adapter_module._deny_wget_download
    with pytest.raises(ParakeetRealtimeEouAdapterError):
        cloud.wget.download("https://example.invalid/model")


def test_authenticated_optional_import_removes_partial_descendant_side_effects(
    tmp_path,
    monkeypatch,
):
    _cloud, _voice_agent, state, _read_calls = _bind_authenticated_import(
        tmp_path,
        monkeypatch,
        fail_service_import=True,
        partial_side_effect=True,
    )
    side_effect_name = (
        adapter_module._PIPECAT_NEMO_PACKAGE + ".partial_side_effect"
    )

    with pytest.raises(ParakeetRealtimeEouAdapterError):
        adapter_module._import_authenticated_nemo_service(_config())

    assert side_effect_name not in sys.modules
    _root_package, _services_package, nemo_package = state["packages"]
    assert not hasattr(nemo_package, "partial_side_effect")
    assert all(
        name != adapter_module._PIPECAT_PACKAGE
        and not name.startswith(f"{adapter_module._PIPECAT_PACKAGE}.")
        for name in sys.modules
    )


def test_authenticated_module_executes_snapshot_not_replaced_path(
    tmp_path,
):
    path = (tmp_path / "authenticated_leaf.py").resolve()
    snapshot = b"selected_value = 'authenticated'\n"
    path.write_bytes(snapshot)
    source = adapter_module._AuthenticatedSource(
        path=path,
        data=snapshot,
        sha256=hashlib.sha256(snapshot).hexdigest(),
        size_bytes=len(snapshot),
    )
    path.write_bytes(b"raise RuntimeError('replacement executed')\n")
    parent = ModuleType("speaker_test_parent")
    name = "speaker_test_parent.authenticated_leaf"
    try:
        module = adapter_module._execute_authenticated_module(name, source, parent)
        assert module.selected_value == "authenticated"
        assert module.__speaker_source_sha256__ == source.sha256
        assert module.__speaker_source_size_bytes__ == source.size_bytes
    finally:
        sys.modules.pop(name, None)
        if hasattr(parent, "authenticated_leaf"):
            delattr(parent, "authenticated_leaf")


def test_authenticated_optional_import_leaves_wget_as_offline_deny_only(
    tmp_path,
    monkeypatch,
):
    cloud, _voice_agent, _state, _hash_calls = _bind_authenticated_import(
        tmp_path,
        monkeypatch,
    )

    adapter_module._import_authenticated_nemo_service(_config())

    assert "wget" not in sys.modules
    assert cloud.wget.download is adapter_module._deny_wget_download
    with pytest.raises(ParakeetRealtimeEouAdapterError):
        cloud.wget.download("https://example.invalid/model")


def test_mocked_load_contract_binds_threads_cuda_rnnt_and_restores_output(
    tmp_path,
    monkeypatch,
    capfd,
):
    model = tmp_path / "parakeet_realtime_eou_120m-v1.nemo"
    model.write_bytes(b"model-fixture")
    events, captured, torch_api = _bind_fake_load(monkeypatch)

    service, selected_torch = adapter_module._load_service(model, _config())
    os.write(1, b"visible-stdout\n")
    os.write(2, b"visible-stderr\n")
    stdout, stderr = capfd.readouterr()

    assert selected_torch is torch_api
    assert service.decoder_type == "rnnt"
    assert captured["decoder_type"] == "rnnt"
    assert captured["device"] == "cuda:0"
    assert captured["chunk_size_in_secs"] == 0.08
    assert events == [
        ("import", "torch"),
        ("set_num_threads", 1),
        ("set_num_interop_threads", 1),
        ("import", adapter_module._SERVICE_MODULE),
        ("memory_fraction", 0.25, 0),
        ("synchronize", 0),
        ("construct", str(model)),
        ("synchronize", 0),
    ]
    assert stdout == "visible-stdout\n"
    assert stderr == "visible-stderr\n"


@pytest.mark.parametrize("model_fault", ("decoder", "dtype", "device", "empty"))
def test_mocked_load_contract_rejects_non_rnnt_fp32_cuda_model(
    tmp_path,
    monkeypatch,
    model_fault,
):
    model = tmp_path / "parakeet_realtime_eou_120m-v1.nemo"
    model.write_bytes(b"model-fixture")
    _bind_fake_load(monkeypatch, model_fault=model_fault)

    with pytest.raises(ParakeetRealtimeEouAdapterError):
        adapter_module._load_service(model, _config())


def test_mocked_load_failure_still_restores_process_output(
    tmp_path,
    monkeypatch,
    capfd,
):
    model = tmp_path / "parakeet_realtime_eou_120m-v1.nemo"
    model.write_bytes(b"model-fixture")
    _bind_fake_load(monkeypatch, constructor_failure=True)

    with pytest.raises(ParakeetRealtimeEouAdapterError):
        adapter_module._load_service(model, _config())
    os.write(1, b"restored-stdout\n")
    os.write(2, b"restored-stderr\n")
    stdout, stderr = capfd.readouterr()

    assert stdout == "restored-stdout\n"
    assert stderr == "restored-stderr\n"


def test_adapter_rehashes_exact_model_before_and_after_load(
    tmp_path,
    monkeypatch,
):
    payload = b"exact-model-receipt"
    model = tmp_path / "parakeet_realtime_eou_120m-v1.nemo"
    model.write_bytes(payload)
    monkeypatch.setattr(
        adapter_module,
        "PARAKEET_REALTIME_EOU_MODEL_RECEIPT",
        (hashlib.sha256(payload).hexdigest(), len(payload)),
    )
    load_calls = []
    service = _Service([_Result("ok<EOU>", True, eou_prob=0.9)])
    torch_api = _Torch(service.events)

    def load(selected, config):
        load_calls.append((selected, config))
        return service, torch_api

    monkeypatch.setattr(adapter_module, "_load_service", load)
    adapter = ParakeetRealtimeEouAdapter(
        model,
        config=_config(),
        clock=_Clock(),
        resource_reader=lambda _torch: ResourceUsage(
            rss_mb=128.0,
            threads=1,
            vram_mb=512.0,
        ),
    )

    assert load_calls == [(model.resolve(strict=True), _config())]
    adapter.close()


def test_adapter_rejects_model_replaced_during_load(
    tmp_path,
    monkeypatch,
):
    payload = b"authenticated-model"
    model = tmp_path / "parakeet_realtime_eou_120m-v1.nemo"
    model.write_bytes(payload)
    monkeypatch.setattr(
        adapter_module,
        "PARAKEET_REALTIME_EOU_MODEL_RECEIPT",
        (hashlib.sha256(payload).hexdigest(), len(payload)),
    )
    load_calls = []

    def load(selected, _config):
        load_calls.append(selected)
        replacement = tmp_path / "replacement.nemo"
        replacement.write_bytes(b"x" * len(payload))
        os.replace(replacement, selected)
        events = []
        return _Service([]), _Torch(events)

    monkeypatch.setattr(adapter_module, "_load_service", load)

    with pytest.raises(ParakeetRealtimeEouAdapterError):
        ParakeetRealtimeEouAdapter(
            model,
            config=_config(),
            clock=_Clock(),
            resource_reader=lambda _torch: ResourceUsage(
                rss_mb=128.0,
                threads=1,
                vram_mb=512.0,
            ),
        )
    assert load_calls == [model.resolve(strict=True)]


def test_adapter_rejects_wrong_model_before_load(tmp_path, monkeypatch):
    expected = b"authenticated-model"
    model = tmp_path / "parakeet_realtime_eou_120m-v1.nemo"
    model.write_bytes(b"x" * len(expected))
    monkeypatch.setattr(
        adapter_module,
        "PARAKEET_REALTIME_EOU_MODEL_RECEIPT",
        (hashlib.sha256(expected).hexdigest(), len(expected)),
    )
    load_calls = []
    monkeypatch.setattr(
        adapter_module,
        "_load_service",
        lambda *_args: load_calls.append(True),
    )

    with pytest.raises(ParakeetRealtimeEouAdapterError):
        ParakeetRealtimeEouAdapter(model, config=_config())
    assert load_calls == []


def test_service_transcribe_and_reset_raw_fd_output_is_suppressed(
    tmp_path,
    capfd,
):
    class NoisyService(_Service):
        def transcribe(self, audio, *, stream_id):
            os.write(1, b"hidden-transcribe-stdout\n")
            os.write(2, b"hidden-transcribe-stderr\n")
            return super().transcribe(audio, stream_id=stream_id)

        def reset_state(self, *, stream_id):
            os.write(1, b"hidden-reset-stdout\n")
            os.write(2, b"hidden-reset-stderr\n")
            return super().reset_state(stream_id=stream_id)

    service = NoisyService([_Result("text", False)] * 4)
    adapter = _adapter(tmp_path, service)

    adapter.transcribe(
        _request(tmp_path),
        emit_partial=lambda _seq, _partial: None,
    )
    os.write(1, b"visible-after-service-stdout\n")
    os.write(2, b"visible-after-service-stderr\n")
    stdout, stderr = capfd.readouterr()

    assert stdout == "visible-after-service-stdout\n"
    assert stderr == "visible-after-service-stderr\n"


def test_close_cleanup_raw_fd_output_is_suppressed(tmp_path, capfd):
    events = []

    class NoisyCuda(_Cuda):
        def synchronize(self, device):
            os.write(1, b"hidden-close-sync-stdout\n")
            os.write(2, b"hidden-close-sync-stderr\n")
            super().synchronize(device)

        @staticmethod
        def empty_cache():
            os.write(1, b"hidden-empty-cache-stdout\n")
            os.write(2, b"hidden-empty-cache-stderr\n")

    torch_api = _Torch(events)
    torch_api.cuda = NoisyCuda(events)
    model = tmp_path / "parakeet_realtime_eou_120m-v1.nemo"
    model.write_bytes(b"model-fixture")
    adapter = ParakeetRealtimeEouAdapter(
        model,
        config=_config(),
        service=_Service([], events=events),
        torch_api=torch_api,
        clock=_Clock(),
        resource_reader=lambda _torch: ResourceUsage(
            rss_mb=128.0,
            threads=1,
            vram_mb=512.0,
        ),
    )

    adapter.close()
    os.write(1, b"visible-after-close-stdout\n")
    os.write(2, b"visible-after-close-stderr\n")
    stdout, stderr = capfd.readouterr()

    assert stdout == "visible-after-close-stdout\n"
    assert stderr == "visible-after-close-stderr\n"


def test_output_restoration_failure_poisoning_fails_closed(
    tmp_path,
    monkeypatch,
):
    adapter = _adapter(
        tmp_path,
        _Service([_Result("stop<EOU>", True, eou_prob=0.9)]),
    )
    original_restore = adapter_module._restore_output_descriptors

    def restore_then_fail(saved, sink_descriptor):
        original_restore(saved, sink_descriptor)
        raise ParakeetRealtimeEouAdapterError()

    monkeypatch.setattr(
        adapter_module,
        "_restore_output_descriptors",
        restore_then_fail,
    )
    request = _request(tmp_path, tail_samples=48_000)

    with pytest.raises(ParakeetRealtimeEouAdapterError):
        adapter.transcribe(request, emit_partial=lambda _seq, _partial: None)
    assert adapter._poisoned is True
    with pytest.raises(ParakeetRealtimeEouAdapterError):
        adapter.transcribe(request, emit_partial=lambda _seq, _partial: None)


def test_stops_on_first_eou_and_reports_actual_source_and_tail(tmp_path):
    service = _Service(
        [
            _Result("stop", False),
            _Result(" now", False),
            _Result("<EOU>", True, eou_prob=0.75),
            _Result("must not run", False),
        ]
    )
    adapter = _adapter(tmp_path, service)
    emitted = []

    result = adapter.transcribe(
        _request(tmp_path),
        emit_partial=lambda seq, partial: emitted.append((seq, partial)),
    )

    assert result.final == "stop now"
    assert result.endpoint_reason == "eou"
    assert result.native_endpoint is True
    assert result.endpoint_probability == 0.75
    assert result.source_samples == 2_560
    assert result.source_samples_consumed == 2_560
    assert result.declared_tail_samples == 2_560
    assert result.tail_samples_consumed == 1_280
    assert result.endpoint_sample == 3_840
    assert result.endpoint_latency_ms == 80.0
    assert result.authoritative is True
    assert result.chunks_yielded == 3
    assert len(service.chunks) == 3
    assert all(len(chunk) == 2_560 for chunk in service.chunks)
    assert service.reset_calls == 1
    assert service.reset_ids == ["case-r0"]
    assert [item[0] for item in emitted] == list(range(len(emitted)))


def test_marker_only_terminal_retains_accumulated_delta_text(tmp_path):
    service = _Service(
        [
            _Result("listen", False),
            _Result(" carefully", False),
            _Result("<EOU>", True, eou_prob=0.91),
        ]
    )
    adapter = _adapter(tmp_path, service)
    emitted = []

    result = adapter.transcribe(
        _request(tmp_path),
        emit_partial=lambda seq, partial: emitted.append((seq, partial)),
    )

    assert result.final == "listen carefully"
    assert [partial.text for _seq, partial in emitted] == [
        "listen",
        "listen carefully",
    ]
    assert result.endpoint_reason == "eou"
    assert result.endpoint_probability == 0.91


def test_word_start_and_subword_deltas_join_before_outer_trim(tmp_path):
    service = _Service(
        [
            _Result("  turn", False),
            _Result("ing", False),
            _Result(" point  ", False),
            _Result("<EOU>", True, eou_prob=0.88),
        ]
    )
    adapter = _adapter(tmp_path, service)
    emitted = []

    result = adapter.transcribe(
        _request(tmp_path),
        emit_partial=lambda seq, partial: emitted.append((seq, partial)),
    )

    assert result.final == "turning point"
    assert [partial.text for _seq, partial in emitted] == [
        "turn",
        "turning",
        "turning point",
    ]


def test_terminal_delta_text_is_appended_before_marker(tmp_path):
    service = _Service(
        [
            _Result("voice", False),
            _Result(" agent<EOU>", True, eou_prob=0.79),
        ]
    )
    adapter = _adapter(tmp_path, service)

    result = adapter.transcribe(
        _request(tmp_path),
        emit_partial=lambda _seq, _partial: None,
    )

    assert result.final == "voice agent"
    assert result.endpoint_reason == "eou"
    assert result.authoritative is True


def test_cumulative_delta_hypothesis_overflow_poisoning_fails_closed(tmp_path):
    service = _Service(
        [
            _Result("x" * (adapter_module.MAX_HYPOTHESIS_CHARS - 1), False),
            _Result("yz", False),
        ]
    )
    adapter = _adapter(tmp_path, service)
    request = _request(tmp_path)

    with pytest.raises(ParakeetRealtimeEouAdapterError):
        adapter.transcribe(request, emit_partial=lambda _seq, _partial: None)
    assert adapter._poisoned is True
    with pytest.raises(ParakeetRealtimeEouAdapterError):
        adapter.transcribe(request, emit_partial=lambda _seq, _partial: None)


def test_repeated_partial_cadence_emits_changed_cumulative_text_only(tmp_path):
    service = _Service(
        [
            _Result("one", False),
            _Result(" two", False),
            _Result("", False),
            _Result(" three", False),
        ]
    )
    adapter = _adapter(tmp_path, service)
    emitted = []

    result = adapter.transcribe(
        _request(tmp_path),
        emit_partial=lambda seq, partial: emitted.append((seq, partial)),
    )

    assert result.final == "one two three"
    assert [seq for seq, _partial in emitted] == [0, 1, 2]
    assert [partial.after_samples for _seq, partial in emitted] == [
        1_280,
        2_560,
        5_120,
    ]
    assert [partial.text for _seq, partial in emitted] == [
        "one",
        "one two",
        "one two three",
    ]


def test_tail_exhaustion_is_non_authoritative_and_resets_once(tmp_path):
    service = _Service(
        [
            _Result("hello", False),
            _Result(" world", False),
            _Result("", False),
            _Result("", False),
        ]
    )
    adapter = _adapter(tmp_path, service)

    result = adapter.transcribe(
        _request(tmp_path),
        emit_partial=lambda _seq, _partial: None,
    )

    assert result.final == "hello world"
    assert result.endpoint_reason == "tail_exhausted"
    assert result.native_endpoint is False
    assert result.endpoint_probability is None
    assert result.endpoint_sample is None
    assert result.endpoint_latency_ms is None
    assert result.source_samples_consumed == 2_560
    assert result.tail_samples_consumed == 2_560
    assert result.authoritative is False
    assert result.finalization_ms == pytest.approx(2.0)
    assert result.compute_ms == pytest.approx(5.0)
    assert service.reset_calls == 1
    assert service.reset_ids == ["case-r0"]
    assert service.events[-3:] == [
        ("synchronize", 0),
        ("reset", "case-r0"),
        ("synchronize", 0),
    ]


def test_sequential_native_eou_and_eob_begin_from_clean_service_state(tmp_path):
    service = _StatefulService(
        [
            _Result("first", False),
            _Result("<EOU>", True, eou_prob=0.9),
            _Result("second", False),
            _Result("<EOB>", True, eob_prob=0.8),
        ]
    )
    adapter = _adapter(tmp_path, service)
    first = replace(
        _request(tmp_path, tail_samples=48_000),
        request_id="native-first",
    )
    second = replace(
        _request(tmp_path, tail_samples=48_000),
        request_id="native-second",
    )

    first_result = adapter.transcribe(
        first,
        emit_partial=lambda _seq, _partial: None,
    )
    second_result = adapter.transcribe(
        second,
        emit_partial=lambda _seq, _partial: None,
    )

    assert first_result.endpoint_reason == "eou"
    assert second_result.endpoint_reason == "eob"
    assert first_result.final == "first"
    assert second_result.final == "second"
    assert service.stream_starts == [
        ("native-first", True),
        ("native-second", True),
    ]
    assert service.reset_ids == ["native-first", "native-second"]
    assert service.clean is True


def test_sequential_tail_reset_cleans_state_before_next_request(tmp_path):
    service = _StatefulService(
        [
            _Result("first", False),
            _Result("", False),
            _Result("", False),
            _Result("", False),
            _Result("second", False),
            _Result("<EOU>", True, eou_prob=0.9),
        ]
    )
    adapter = _adapter(tmp_path, service)
    first = replace(_request(tmp_path), request_id="tail-first")
    second = replace(
        _request(tmp_path, tail_samples=48_000),
        request_id="tail-second",
    )

    first_result = adapter.transcribe(
        first,
        emit_partial=lambda _seq, _partial: None,
    )
    second_result = adapter.transcribe(
        second,
        emit_partial=lambda _seq, _partial: None,
    )

    assert first_result.endpoint_reason == "tail_exhausted"
    assert second_result.endpoint_reason == "eou"
    assert first_result.final == "first"
    assert second_result.final == "second"
    assert service.stream_starts == [
        ("tail-first", True),
        ("tail-second", True),
    ]
    assert service.reset_ids == ["tail-first", "tail-second"]
    assert service.clean is True


def test_early_source_endpoint_is_retained_as_premature_evidence(tmp_path):
    service = _Service([_Result("wrong<EOB>", True, eob_prob=0.6)])
    adapter = _adapter(tmp_path, service)

    result = adapter.transcribe(
        _request(tmp_path, tail_samples=48_000),
        emit_partial=lambda _seq, _partial: None,
    )

    assert result.endpoint_reason == "eob"
    assert result.source_samples_consumed == 1_280
    assert result.tail_samples_consumed == 0
    assert result.endpoint_sample == 1_280
    assert result.endpoint_latency_ms is None
    assert result.authoritative is False


def test_complete_source_eob_is_native_but_never_authoritative(tmp_path):
    service = _Service(
        [
            _Result("yes", False),
            _Result("", False),
            _Result("<EOB>", True, eob_prob=0.6),
        ]
    )
    adapter = _adapter(tmp_path, service)

    result = adapter.transcribe(
        _request(tmp_path),
        emit_partial=lambda _seq, _partial: None,
    )

    assert result.endpoint_reason == "eob"
    assert result.native_endpoint is True
    assert result.source_samples_consumed == result.source_samples
    assert result.endpoint_sample == 3_840
    assert result.endpoint_latency_ms is None
    assert result.authoritative is False


def test_last_short_chunk_is_zero_padded_but_not_counted_as_tail(tmp_path):
    service = _Service(
        [
            _Result("a", False),
            _Result("", False),
            _Result("", False),
        ]
    )
    adapter = _adapter(tmp_path, service)

    result = adapter.transcribe(
        _request(tmp_path, source_samples=2_600, tail_samples=1),
        emit_partial=lambda _seq, _partial: None,
    )

    assert result.source_samples_consumed == 2_600
    assert result.tail_samples_consumed == 1
    assert result.model_padding_samples == 1_239
    assert result.chunks_yielded == 3
    assert len(service.chunks[-1]) == 2_560


def test_endpoint_boundary_includes_terminal_model_zero_fill(tmp_path):
    service = _Service(
        [
            _Result("a", False),
            _Result("", False),
            _Result("<EOU>", True, eou_prob=0.8),
        ]
    )
    adapter = _adapter(tmp_path, service)

    result = adapter.transcribe(
        _request(tmp_path, source_samples=2_600, tail_samples=1),
        emit_partial=lambda _seq, _partial: None,
    )

    assert result.source_samples_consumed == 2_600
    assert result.tail_samples_consumed == 1
    assert result.model_padding_samples == 1_239
    assert result.endpoint_sample == 3_840
    assert result.endpoint_latency_ms == 77.5
    assert result.authoritative is True


@pytest.mark.parametrize(
    ("terminal", "final", "reason", "probability"),
    [
        (_Result("stop<EOU>after", True, eou_prob=0.9), "stop", "eou", 0.9),
        (
            _Result("stop<EOB>noise<EOU>", True, eou_prob=0.9, eob_prob=0.8),
            "stop",
            "eob",
            0.8,
        ),
        (
            _Result("stop<EOU>noise<EOB>", True, eou_prob=0.9, eob_prob=0.8),
            "stop",
            "eou",
            0.9,
        ),
    ],
)
def test_service_valid_suffix_and_both_markers_select_earliest(
    tmp_path,
    terminal,
    final,
    reason,
    probability,
):
    adapter = _adapter(tmp_path, _Service([terminal]))

    result = adapter.transcribe(
        _request(tmp_path, tail_samples=48_000),
        emit_partial=lambda _seq, _partial: None,
    )

    assert result.final == final
    assert result.endpoint_reason == reason
    assert result.endpoint_probability == probability


@pytest.mark.parametrize(
    "terminal",
    [
        _Result("stop<EOU>", False),
        _Result("stop<EOU>", True, eou_prob=1.1),
        _Result(
            "stop<EOU>noise<EOB>",
            True,
            eou_prob=0.9,
            eob_prob=1.1,
        ),
    ],
)
def test_malformed_terminal_poisoning_fails_closed(tmp_path, terminal):
    service = _Service([terminal])
    adapter = _adapter(tmp_path, service)
    request = _request(tmp_path, tail_samples=48_000)

    with pytest.raises(ParakeetRealtimeEouAdapterError):
        adapter.transcribe(request, emit_partial=lambda _seq, _partial: None)
    with pytest.raises(ParakeetRealtimeEouAdapterError):
        adapter.transcribe(request, emit_partial=lambda _seq, _partial: None)


def test_cuda_synchronization_brackets_each_chunk_call(tmp_path):
    events = []
    service = _Service(
        [_Result("stop<EOU>", True, eou_prob=0.9)],
        events=events,
    )
    adapter = _adapter(tmp_path, service, events=events)

    adapter.transcribe(
        _request(tmp_path, tail_samples=48_000),
        emit_partial=lambda _seq, _partial: None,
    )

    assert events == [
        ("synchronize", 0),
        ("transcribe", "case-r0"),
        ("synchronize", 0),
    ]


def test_rejects_legacy_protocol_and_non_native_chunk_geometry(tmp_path):
    service = _Service([_Result("", False), _Result("", False)])
    adapter = _adapter(tmp_path, service)
    request = _request(tmp_path)

    legacy = TranscribeRequest(
        request_id=request.request_id,
        pcm=request.pcm,
        stream=request.stream,
    )
    with pytest.raises(ParakeetRealtimeEouAdapterError):
        adapter.transcribe(legacy, emit_partial=lambda _seq, _partial: None)

    wrong_chunk = TranscribeRequest(
        request_id=request.request_id,
        pcm=request.pcm,
        stream=StreamConfig(
            chunk_samples=2_560,
            pace="burst",
            partial_interval_ms=80,
            tail_padding_samples=2_560,
        ),
        protocol_version=NATIVE_ENDPOINT_PROTOCOL_VERSION,
    )
    with pytest.raises(ParakeetRealtimeEouAdapterError):
        adapter.transcribe(wrong_chunk, emit_partial=lambda _seq, _partial: None)
