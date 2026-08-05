"""Exact-artifact and atomic-acquisition tests for optional Smart Turn setup."""
from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path

import pytest

import tools.setup_models as setup_models


@pytest.fixture
def pinned_payload(monkeypatch):
    payload = b"deterministic-smart-turn-test-model"
    monkeypatch.setattr(setup_models, "SMART_TURN_MODEL_BYTES", len(payload))
    monkeypatch.setattr(
        setup_models,
        "SMART_TURN_MODEL_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )
    return payload


def _model_path(root: Path) -> Path:
    return root / setup_models.SMART_TURN_MODEL_FILENAME


def _part_files(root: Path) -> list[Path]:
    return list(root.glob(f".{setup_models.SMART_TURN_MODEL_FILENAME}.*.part"))


def test_smart_turn_release_identity_is_immutable_and_complete():
    assert setup_models.SMART_TURN_MODEL_REVISION == (
        "f766f81d3cfdf7737ac64aad813d91bbfd56bf93"
    )
    assert setup_models.SMART_TURN_MODEL_FILENAME == "smart-turn-v3.2-cpu.onnx"
    assert setup_models.SMART_TURN_MODEL_BYTES == 8_679_182
    assert setup_models.SMART_TURN_MODEL_SHA256 == (
        "2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f"
    )
    assert setup_models.SMART_TURN_MODEL_LICENSE == "BSD-2-Clause"
    assert setup_models.SMART_TURN_MODEL_REVISION in setup_models.SMART_TURN_MODEL_URL
    assert "/resolve/main/" not in setup_models.SMART_TURN_MODEL_URL
    assert setup_models.SMART_TURN_MODEL_URL.endswith(
        "/" + setup_models.SMART_TURN_MODEL_FILENAME
    )


def test_smart_turn_fetch_is_atomic_and_revalidates_existing(
    tmp_path, pinned_payload
):
    calls = []

    def open_payload(url):
        calls.append(url)
        return io.BytesIO(pinned_payload)

    path = setup_models.fetch_smart_turn_model(
        str(tmp_path), "https://mirror.invalid/model", opener=open_payload
    )

    assert Path(path) == _model_path(tmp_path)
    assert Path(path).read_bytes() == pinned_payload
    assert os.lstat(path).st_nlink == 1
    assert _part_files(tmp_path) == []
    assert calls == ["https://mirror.invalid/model"]

    def unexpected_download(_url):
        pytest.fail("idempotent setup downloaded instead of revalidating")

    assert setup_models.fetch_smart_turn_model(
        str(tmp_path), opener=unexpected_download
    ) == str(_model_path(tmp_path))


def test_smart_turn_existing_same_length_tamper_is_rejected(
    tmp_path, pinned_payload
):
    destination = _model_path(tmp_path)
    destination.write_bytes(b"x" * len(pinned_payload))

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        setup_models.fetch_smart_turn_model(
            str(tmp_path), opener=lambda _url: pytest.fail("downloaded")
        )


def test_smart_turn_symlink_is_rejected(tmp_path, pinned_payload):
    source = tmp_path / "source.onnx"
    source.write_bytes(pinned_payload)
    destination = _model_path(tmp_path)
    destination.symlink_to(source)

    with pytest.raises(ValueError, match="regular file, not a symlink"):
        setup_models.validate_smart_turn_model(str(destination))


def test_smart_turn_hardlink_is_rejected(tmp_path, pinned_payload):
    source = tmp_path / "source.onnx"
    source.write_bytes(pinned_payload)
    destination = _model_path(tmp_path)
    os.link(source, destination)

    with pytest.raises(ValueError, match="must not be hardlinked"):
        setup_models.validate_smart_turn_model(str(destination))


def test_invalid_forced_refresh_preserves_valid_active_model(
    tmp_path, pinned_payload
):
    destination = Path(
        setup_models.fetch_smart_turn_model(
            str(tmp_path), opener=lambda _url: io.BytesIO(pinned_payload)
        )
    )
    before = destination.read_bytes()
    invalid = bytes([pinned_payload[0] ^ 1]) + pinned_payload[1:]

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        setup_models.fetch_smart_turn_model(
            str(tmp_path), force=True, opener=lambda _url: io.BytesIO(invalid)
        )

    assert destination.read_bytes() == before == pinned_payload
    assert setup_models.validate_smart_turn_model(str(destination)) == str(destination)
    assert _part_files(tmp_path) == []


def test_interrupted_smart_turn_fetch_leaves_no_model_or_part(
    tmp_path, pinned_payload
):
    class InterruptedResponse(io.BytesIO):
        def __init__(self, payload):
            super().__init__(payload)
            self.read_count = 0

        def read(self, size=-1):
            self.read_count += 1
            if self.read_count > 1:
                raise OSError("synthetic interrupted transfer")
            return super().read(size)

    with pytest.raises(OSError, match="interrupted transfer"):
        setup_models.fetch_smart_turn_model(
            str(tmp_path), opener=lambda _url: InterruptedResponse(pinned_payload)
        )

    assert not _model_path(tmp_path).exists()
    assert _part_files(tmp_path) == []


def test_oversized_smart_turn_fetch_fails_before_publish(tmp_path, pinned_payload):
    with pytest.raises(ValueError, match="exceeds pinned size"):
        setup_models.fetch_smart_turn_model(
            str(tmp_path), opener=lambda _url: io.BytesIO(pinned_payload + b"x")
        )

    assert not _model_path(tmp_path).exists()
    assert _part_files(tmp_path) == []
