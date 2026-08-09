from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import pytest

import tools.streaming_stt.manifest as manifest_module
from tools.streaming_stt.manifest import (
    KYUTAI_ADAPTER,
    KYUTAI_ARTIFACT_NAMES,
    KYUTAI_MODEL_RECEIPTS,
    KyutaiConfig,
    ManifestError,
    load_worker_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _bound(path: Path, *, lexical: Path | None = None) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": str(lexical if lexical is not None else path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _fixture(tmp_path: Path, monkeypatch) -> Path:
    venv = tmp_path / "venv"
    python = venv / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.symlink_to(Path(sys.executable).resolve(strict=True))
    marker = venv / "pyvenv.cfg"
    marker.write_text(
        "home = /usr/bin\ninclude-system-site-packages = false\n",
        encoding="utf-8",
    )

    control = tmp_path / "control"
    control.mkdir()
    receipt = control / "runtime-receipt.json"
    receipt.write_text('{"schema_version":1}\n', encoding="utf-8")
    wheel_lock = control / "kyutai-stt-runtime-wheels.lock.json"
    wheel_lock.write_text('{"schema_version":1}\n', encoding="utf-8")

    model = tmp_path / "model"
    model.mkdir()
    model_paths = {
        "model-config": model / "config.json",
        "model-weights": model / "model.safetensors",
        "model-mimi": model / "mimi-pytorch-e351c8d8@125.safetensors",
        "model-tokenizer": model / "tokenizer_en_fr_audio_8000.model",
    }
    for index, path in enumerate(model_paths.values(), start=1):
        path.write_bytes(bytes([index]))
    fixture_receipts = {
        name: (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_size,
        )
        for name, path in model_paths.items()
    }
    monkeypatch.setattr(
        manifest_module,
        "KYUTAI_MODEL_RECEIPTS",
        fixture_receipts,
    )
    wheel_digest = hashlib.sha256(wheel_lock.read_bytes()).hexdigest()
    monkeypatch.setattr(
        manifest_module,
        "KYUTAI_RUNTIME_WHEEL_LOCK_SHA256",
        wheel_digest,
    )
    monkeypatch.setattr(
        manifest_module,
        "KYUTAI_RUNTIME_WHEEL_LOCK_SIZE_BYTES",
        wheel_lock.stat().st_size,
    )
    config = KyutaiConfig(wheel_lock_sha256=wheel_digest)
    by_name = {
        "runtime-receipt": receipt,
        "runtime-wheel-lock": wheel_lock,
        "venv-marker": marker,
        **model_paths,
    }
    payload = {
        "schema_version": 9,
        "model_id": "kyutai-stt-1b-en-fr-candle-095e",
        "adapter": KYUTAI_ADAPTER,
        "adapter_config": config.as_dict(),
        "python": _bound(python.resolve(strict=True), lexical=python),
        "worker": _bound(REPO_ROOT / "tools" / "streaming_stt" / "worker.py"),
        "artifacts": [
            {"name": name, **_bound(by_name[name])} for name in KYUTAI_ARTIFACT_NAMES
        ],
        "limits": {
            "startup_timeout_sec": 300.0,
            "case_timeout_sec": 300.0,
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def test_schema_v9_binds_exact_candle_model_and_diagnostic_stream(
    tmp_path,
    monkeypatch,
):
    loaded = load_worker_manifest(_fixture(tmp_path, monkeypatch))

    assert loaded.schema_version == 9
    assert loaded.adapter == KYUTAI_ADAPTER
    assert tuple(loaded.artifact_by_name) == KYUTAI_ARTIFACT_NAMES
    assert isinstance(loaded.adapter_config, KyutaiConfig)
    config = loaded.adapter_config
    assert config.model_revision == "095e38f6242006a93c2541149b181988397f5c7c"
    assert config.moshi_version == "0.2.11"
    assert config.julius_version == "0.2.8"
    assert (
        config.input_sample_rate_hz,
        config.input_chunk_samples,
        config.mimi_sample_rate_hz,
        config.mimi_frame_samples,
    ) == (16_000, 1_280, 24_000, 1_920)
    assert config.partial_interval_ms == 160
    assert config.resampling_mode == "whole-buffer-noncausal"
    assert config.initial_frame_policy == "duplicate-first-frame-prime"
    assert config.initial_frame_prime_steps == 1
    assert config.terminal_tail_samples == 16_000
    assert config.semantic_head_policy == "diagnostic-finite-only"
    assert config.semantic_head_count == 4
    assert config.semantic_head_dim == 6
    assert config.endpoint_owner == "none"
    assert config.early_stop is False
    assert config.temperature == config.text_temperature == 0.0
    assert config.torch_compile is False
    assert config.no_torch_compile_env == "1"
    assert config.cuda_graph is False
    assert config.no_cuda_graph_env == "1"
    assert config.maximum_vram_fraction == 0.5
    assert config.minimum_free_vram_mb == 8_192
    assert config.minimum_host_available_bytes == 12 * 1024**3


def test_four_public_candle_receipts_are_repository_frozen():
    assert KYUTAI_MODEL_RECEIPTS == {
        "model-config": (
            "a3f1c6f7a39fca1fb1bbff68eaabc560b8037d2cdc68aa1f489859949a4223de",
            1_315,
        ),
        "model-weights": (
            "b9e97c53229dce728d65c76bfa892f7b563c69d671899f0ebc6518582dddec6f",
            1_978_522_200,
        ),
        "model-mimi": (
            "09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50",
            384_644_900,
        ),
        "model-tokenizer": (
            "cd87dd5d17169151782ac700280ec057e5d658a9afbe238a048ea5ff318cce69",
            120_378,
        ),
    }


def test_measured_runtime_content_is_repository_frozen():
    assert manifest_module.KYUTAI_RUNTIME_WHEEL_LOCK_SHA256 == (
        "8d0e41563bb5e91500af42a912c5e6f825f16a67537ddc350556303e88609e25"
    )
    assert manifest_module.KYUTAI_RUNTIME_WHEEL_LOCK_SIZE_BYTES == 25_818
    assert manifest_module.KYUTAI_RUNTIME_CONTENT_SHA256 == (
        "9edc9c42b8c718d0e4b17d917c04c05acc0b5ecca4ec9504866ad34f1f62dbf0"
    )
    assert manifest_module.KYUTAI_RUNTIME_FILE_COUNT == 16_547
    assert manifest_module.KYUTAI_RUNTIME_TOTAL_SIZE_BYTES == 5_694_993_765
    assert manifest_module.KYUTAI_RUNTIME_MAXIMUM_FILE_BYTES == 984_633_129


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("julius_version", "0.2.7"),
        ("model_revision", "main"),
        ("dtype", "float16"),
        ("initial_frame_policy", "none"),
        ("initial_frame_prime_steps", 0),
        ("input_chunk_samples", 640),
        ("mimi_frame_samples", 960),
        ("resampling_mode", "online-causal"),
        ("terminal_tail_samples", 15_999),
        ("partial_interval_ms", 500),
        ("temperature", -0.0),
        ("text_temperature", -0.0),
        ("semantic_head_count", 3),
        ("semantic_head_policy", "endpoint"),
        ("endpoint_owner", "semantic-head"),
        ("early_stop", True),
        ("num_threads", True),
        ("maximum_vram_fraction", 1.0),
        ("minimum_free_vram_mb", 4_096),
        ("minimum_host_available_bytes", 1),
        ("torch_compile", True),
        ("no_torch_compile_env", "0"),
        ("cuda_graph", True),
        ("no_cuda_graph_env", "0"),
    ],
)
def test_kyutai_config_rejects_semantic_or_resource_drift(field, value):
    with pytest.raises(ManifestError):
        replace(KyutaiConfig(), **{field: value})


def test_schema_v9_rejects_wrong_model_basename_or_config_field(
    tmp_path,
    monkeypatch,
):
    path = _fixture(tmp_path, monkeypatch)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["adapter_config"]["early_stop"] = True
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ManifestError):
        load_worker_manifest(path)
