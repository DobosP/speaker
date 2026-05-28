"""Unit tests for the speaker enrollment flow (core.enroll).

Pure / injected fakes -- no microphone, no sherpa-onnx, no model files. The
recorder and the gate are both injected, so ``run_enrollment`` exercises the
real averaging + persistence + self-check logic with synthetic audio.
"""
from __future__ import annotations

import json

import pytest

from core.enroll import (
    Enrollment,
    average_embeddings,
    enroll_from_recordings,
    enrollment_matches_model,
    l2_normalize,
    load_enrollment,
    run_enrollment,
    save_enrollment,
)
from core.engines.speaker_gate import SpeakerGate

USER = [1.0, 0.0, 0.0]


def _gate(embed):
    """A SpeakerGate whose embed() returns ``embed`` regardless of input."""
    return SpeakerGate(threshold=0.5, embed_fn=lambda samples, sr: embed)


# --- pure embedding math -----------------------------------------------------


def test_l2_normalize_unit_and_zero():
    assert l2_normalize([3.0, 0.0, 0.0]) == [1.0, 0.0, 0.0]
    assert l2_normalize([0.0, 0.0]) == [0.0, 0.0]  # zero vector unchanged


def test_average_embeddings_normalizes_each_then_renormalizes():
    out = average_embeddings([[2.0, 0.0], [0.0, 5.0]])
    # Each normalized to a unit axis; mean is the diagonal, renormalized.
    assert pytest.approx(out[0], abs=1e-9) == 0.7071067811865476
    assert pytest.approx(out[1], abs=1e-9) == 0.7071067811865476


def test_average_embeddings_rejects_empty_and_mismatched():
    with pytest.raises(ValueError):
        average_embeddings([])
    with pytest.raises(ValueError):
        average_embeddings([[]])
    with pytest.raises(ValueError):
        average_embeddings([[1.0, 0.0], [1.0]])


# --- persistence -------------------------------------------------------------


def test_save_load_round_trip(tmp_path):
    path = tmp_path / "sub" / "enroll.json"  # parent dir created on save
    enr = Enrollment(model="/m/spk.onnx", embedding=[0.6, 0.8], sample_rate=16000, passes=3)
    save_enrollment(str(path), enr)
    loaded = load_enrollment(str(path))
    assert loaded.model == "/m/spk.onnx"
    assert loaded.embedding == [0.6, 0.8]
    assert loaded.sample_rate == 16000
    assert loaded.passes == 3
    assert loaded.dim == 2


def test_load_enrollment_rejects_missing_embedding(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"model": "/m/spk.onnx", "embedding": []}))
    with pytest.raises(ValueError):
        load_enrollment(str(path))


def test_enrollment_matches_model():
    enr = Enrollment(model="/m/spk.onnx", embedding=USER)
    assert enrollment_matches_model(enr, "/m/spk.onnx") is True
    assert enrollment_matches_model(enr, "/m/other.onnx") is False
    # Empty recorded model (hand-written / legacy) is trusted.
    assert enrollment_matches_model(Enrollment(model="", embedding=USER), "/m/x.onnx") is True


# --- enroll_from_recordings --------------------------------------------------


def test_enroll_from_recordings_averages_and_records_provenance():
    gate = _gate([3.0, 0.0, 0.0])  # un-normalized; averaging must normalize it
    enr = enroll_from_recordings(
        gate, [[0.0], [0.0]], model_path="/m/spk.onnx", sample_rate=16000
    )
    assert enr is not None
    assert enr.passes == 2
    assert enr.embedding == [1.0, 0.0, 0.0]
    assert enr.model.endswith("spk.onnx")


def test_enroll_from_recordings_returns_none_when_no_usable_embedding():
    gate = _gate(None)  # model can't embed any recording
    assert enroll_from_recordings(gate, [[0.0]], model_path="/m/spk.onnx") is None


# --- run_enrollment (CLI orchestration) --------------------------------------


def _config(tmp_path, **sherpa):
    base = {
        "speaker_embedding_model": "/m/spk.onnx",
        "speaker_enroll_embedding": str(tmp_path / "enroll.json"),
        "sample_rate": 16000,
    }
    base.update(sherpa)
    return {"sherpa": base}


def test_run_enrollment_saves_embedding_and_wires_config(tmp_path):
    config = _config(tmp_path)
    config_path = tmp_path / "config.local.json"
    msgs: list[str] = []
    code = run_enrollment(
        config,
        passes=3,
        seconds=1.0,
        config_path=str(config_path),
        recorder=lambda secs: [0.1, 0.2, 0.3],
        gate=_gate(USER),
        out=msgs.append,
    )
    assert code == 0
    # The averaged embedding landed on disk.
    enr = load_enrollment(str(tmp_path / "enroll.json"))
    assert enr.embedding == USER
    assert enr.passes == 3
    # The machine-local config now points at the model + the saved embedding.
    written = json.loads(config_path.read_text())
    assert written["sherpa"]["speaker_embedding_model"].endswith("spk.onnx")
    assert written["sherpa"]["speaker_enroll_embedding"].endswith("enroll.json")
    # Self-check reported a clean (high-similarity) reference, no warning.
    text = "\n".join(msgs)
    assert "Enrolled from 3 clip(s)" in text
    assert "WARNING" not in text


def test_run_enrollment_without_model_is_actionable(tmp_path):
    config = {"sherpa": {}}  # no speaker_embedding_model
    msgs: list[str] = []
    code = run_enrollment(
        config, config_path=str(tmp_path / "c.json"),
        recorder=lambda secs: [0.0], gate=_gate(USER), out=msgs.append,
    )
    assert code == 2
    assert "setup_models" in "\n".join(msgs)


def test_run_enrollment_reports_failure_when_no_embedding(tmp_path):
    config = _config(tmp_path)
    msgs: list[str] = []
    code = run_enrollment(
        config, passes=2, config_path=str(tmp_path / "c.json"),
        recorder=lambda secs: [0.0], gate=_gate(None), out=msgs.append,
    )
    assert code == 3
    assert "Enrollment failed" in "\n".join(msgs)
