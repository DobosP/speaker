"""Local-file native Moonshine 0.1.5 adapter for English after-PCM comparison.

No downloader, microphone, streaming endpoint or agent is constructed here.
The benchmark owner hashes these public artifacts and runs local-file inference
in a bounded worker; OS network isolation is a caller-reported property.
Moonshine exposes no native thread-count option; the requested CPU
budget is enforced by the caller's process affinity and verified before load.
"""

from __future__ import annotations

from importlib.metadata import version
import os
from pathlib import Path
import stat
from typing import Any


RUNTIME_VERSION = "0.1.5"
RUNTIME_SOURCE_REVISION = "234f60faa0eb388b01cdf7e60aca232af37aefda"
MODEL_SOURCE_REPO = "moonshine-ai/moonshine-voice-assets"
MODEL_SOURCE_REVISION = "0bf2f2e5aff22e6fbba4300b00a4e00bbc4f8aae"
EXPORT_DIRECTORY = "quantized_26_08_21"
WHEEL_FILENAME = "moonshine_voice-0.1.5-py3-none-manylinux_2_34_x86_64.whl"
WHEEL_BYTES = 19_897_307
WHEEL_SHA256 = "1ed9e0ccf94be4845d69e7fa862dcf4c8b5801a758796966494c694f48184496"
WHEEL_URL = (
    "https://files.pythonhosted.org/packages/a9/9d/"
    "228f738b48e0e7cc1de97c3f842b5470c6a3c5f7a0bffbf13c3d00eefb87/" + WHEEL_FILENAME
)
MODEL_FILES = (
    "adapter.ort",
    "cross_kv.ort",
    "decoder_kv.ort",
    "encoder.ort",
    "frontend.model.ort",
    "frontend.weights.ort",
    "streaming_config.json",
    "tokenizer.bin",
)
ARCHITECTURES = {"tiny-streaming": 2, "small-streaming": 4}
# Six SHA-256 values come from the pinned publisher commit's LFS pointers.
# The two small, non-LFS files carry size and MD5/CRC32C in its FILES.tsv.
# Provisioning verifies those, then records their observed SHA-256 separately.
PUBLIC_ARTIFACTS = {
    "tiny-streaming": {
        "adapter.ort": (
            1_319_664,
            "22ecc949e146c49667fda28d102d4e30749a107dc88a396292aa8f277ef1347c",
            None,
        ),
        "cross_kv.ort": (
            1_287_544,
            "143a36667b8d05fd9d04e8c337b7ee121f37ef299aea6b3d82bdb3d3401950b4",
            None,
        ),
        "decoder_kv.ort": (
            32_583_720,
            "8852553f312adb6c9aa4d17418015049b30f412209ee569d336548c0044627de",
            None,
        ),
        "encoder.ort": (
            7_675_440,
            "a8414e1a5dedf9f2093d7680601dd8a9b0433e7020260eafe0e370ead91134ca",
            None,
        ),
        "frontend.model.ort": (23_344, None, "6c5e6287b3a34eee5f269e3d97ca72a6"),
        "frontend.weights.ort": (
            2_093_464,
            "217da24ac6f522ebf02da8ef288e77d1ac68d50d4a6821433182e4fbf4204bbd",
            None,
        ),
        "streaming_config.json": (509, None, "d022b54c561ca6009f5949c09a0377c7"),
        "tokenizer.bin": (
            249_974,
            "6884b35fd6377d4c4d32336a0bc152f36b64d1e45b6503683cdc238250a8472d",
            None,
        ),
    },
    "small-streaming": {
        "adapter.ort": (
            2_870_368,
            "c665f742364febad597cc9ac1e0b341ffbee0e24a1466e2f3bde95e6e4771762",
            None,
        ),
        "cross_kv.ort": (
            5_356_536,
            "e2d3417144e9514055ebfefe8dcc4c0a55a55adcb8530435844c75c53e352bf6",
            None,
        ),
        "decoder_kv.ort": (
            81_878_600,
            "1a05465b1dd955858dfcbee039c0020fb5dd982b0f5094c34e61735d518d771b",
            None,
        ),
        "encoder.ort": (
            44_148_576,
            "2d4d973e91e8aca08c51e7e7efa28a46ab265b63d809d5294d18b86bcd85b993",
            None,
        ),
        "frontend.model.ort": (26_944, None, "43c2d6eb445ed9a56665be659d9a6e8e"),
        "frontend.weights.ort": (
            7_769_464,
            "7ef97521bd4bad3928f5bb6808586f4fcc6e92bd5990394112eed7d4052ec338",
            None,
        ),
        "streaming_config.json": (512, None, "c987419d7fbd825ace3a36e76c413c4c"),
        "tokenizer.bin": (
            249_974,
            "6884b35fd6377d4c4d32336a0bc152f36b64d1e45b6503683cdc238250a8472d",
            None,
        ),
    },
}
MAX_SAMPLES = 30 * 16_000
MAX_RESULT_LINES = 256
MAX_RESULT_CHARS = 4096


class MoonshineAdapterError(RuntimeError):
    """A fixed, content-free adapter failure."""


def _model_directory(model_dir: Path, arch: str) -> Path:
    try:
        root = Path(model_dir)
        if not root.is_absolute() or root.resolve(strict=True) != root:
            raise ValueError
        if not root.is_dir() or set(item.name for item in root.iterdir()) != set(
            MODEL_FILES
        ):
            raise ValueError
        for name, (size, _sha256, _md5) in PUBLIC_ARTIFACTS[arch].items():
            metadata = (root / name).lstat()
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_size != size
            ):
                raise ValueError
        return root
    except (OSError, RuntimeError, TypeError, ValueError):
        raise MoonshineAdapterError("moonshine_model_layout_invalid") from None


def _check_cpu_budget(threads: int) -> None:
    if type(threads) is not int or not 1 <= threads <= 8:
        raise MoonshineAdapterError("moonshine_cpu_budget_invalid")
    try:
        affinity = os.sched_getaffinity(0)
    except (AttributeError, OSError):
        raise MoonshineAdapterError("moonshine_cpu_budget_unverified") from None
    if not 1 <= len(affinity) <= threads:
        raise MoonshineAdapterError("moonshine_cpu_budget_unverified")


def _load_native_transcriber(model_dir: Path, arch: str) -> Any:
    try:
        if version("moonshine-voice") != RUNTIME_VERSION:
            raise ValueError
        from moonshine_voice.moonshine_api import ModelArch
        from moonshine_voice.transcriber import Transcriber

        return Transcriber(
            model_path=str(model_dir),
            model_arch=ModelArch(ARCHITECTURES[arch]),
            options={
                "ort_providers": "cpu",
                "log_api_calls": "false",
                "log_ort_run": "false",
                "log_output_text": "false",
                "return_audio_data": "false",
                "identify_speakers": "false",
                "word_timestamps": "false",
            },
        )
    except Exception:
        raise MoonshineAdapterError("moonshine_native_load_failed") from None


class NativeMoonshineDecoder:
    """One persistent native model, complete-clip transcription, exact close."""

    cpu_budget_kind = "process_cpu_affinity"
    native_thread_count_configured = None

    def __init__(self, transcriber: Any, threads: int, single_thread_env: bool):
        self._transcriber = transcriber
        self._threads = threads
        self._closed = False
        self._failed = False
        self._close_failed = False
        self._single_thread_env = single_thread_env

    def metadata(self) -> dict[str, object]:
        return {
            "runtime_version": RUNTIME_VERSION,
            "cpu_budget_kind": self.cpu_budget_kind,
            "requested_cpu_affinity_budget": self._threads,
            "native_thread_budget_supported": False,
            "configured_native_threads": None,
            "single_thread_env": self._single_thread_env,
        }

    def __call__(self, samples: Any) -> str:
        if self._closed or self._failed:
            raise MoonshineAdapterError("moonshine_decoder_unavailable")
        _check_cpu_budget(self._threads)
        import numpy as np

        if (
            type(samples) is not np.ndarray
            or samples.dtype != np.dtype("float32")
            or samples.ndim != 1
            or not 1 <= samples.size <= MAX_SAMPLES
            or not np.isfinite(samples).all()
            or (np.abs(samples) > 1.0).any()
        ):
            raise MoonshineAdapterError("moonshine_pcm_invalid")
        try:
            transcript = self._transcriber.transcribe_without_streaming(
                samples.tolist(),
                sample_rate=16000,
                flags=0,
            )
            lines = transcript.lines
            if type(lines) is not list or len(lines) > MAX_RESULT_LINES:
                raise ValueError
            texts = []
            count = 0
            for line in lines:
                text = line.text
                if type(text) is not str:
                    raise ValueError
                count += len(text) + 1
                if count > MAX_RESULT_CHARS + 1:
                    raise ValueError
                if text.strip():
                    texts.append(text.strip())
            return " ".join(texts)
        except Exception:
            self._failed = True
            raise MoonshineAdapterError("moonshine_native_decode_failed") from None

    def close(self) -> None:
        if self._closed:
            if self._close_failed:
                raise MoonshineAdapterError("moonshine_native_close_failed")
            return
        self._closed = True
        try:
            self._transcriber.close()
        except Exception:
            self._close_failed = True
            raise MoonshineAdapterError("moonshine_native_close_failed") from None


def build_decoder(model_dir: Path, arch: str, threads: int) -> NativeMoonshineDecoder:
    """Load only the explicit local eight-file tuple under an enforced CPU mask."""
    if type(arch) is not str or arch not in ARCHITECTURES:
        raise MoonshineAdapterError("moonshine_architecture_invalid")
    _check_cpu_budget(threads)
    root = _model_directory(model_dir, arch)
    flag = os.environ.get("MOONSHINE_ORT_SINGLE_THREAD", "")
    single_thread_env = bool(flag) and flag != "0"
    return NativeMoonshineDecoder(
        _load_native_transcriber(root, arch),
        threads,
        single_thread_env,
    )
