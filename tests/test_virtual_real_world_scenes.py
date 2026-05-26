"""Virtual real-world tests for background human speech and mitigations."""

from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path

import numpy as np
import pytest

from tests.fixtures import silence
from tests.harness import AudioHarness, MockWakewordService, make_recorder
from tests.virtual_scenes import (
    VirtualScene,
    VirtualSceneBuilder,
    pad_with_silence,
    passive_intent_gate,
)

pytestmark = [pytest.mark.audio]

CORPUS_DIR = Path(__file__).parent / "fixture_audio" / "virtual_real_world"
SCENES = VirtualSceneBuilder().build()
METADATA = VirtualSceneBuilder().save(CORPUS_DIR, SCENES)
BACKGROUND_SCENES = [scene for scene in SCENES if scene.kind == "background_human"]
TARGET_SCENES = [scene for scene in SCENES if scene.kind == "target_user"]


def _finish_recording_callbacks(audio: np.ndarray) -> list[np.ndarray]:
    callbacks: list[np.ndarray] = []
    rec = make_recorder(
        callback=lambda captured: callbacks.append(captured),
        silence_duration=0.05,
        aec_enabled=False,
    )
    rec._audio_buffer = pad_with_silence(audio, trailing_sec=0.12)
    rec._is_speaking = True
    rec._finish_recording()
    return callbacks


def _run_wakeword_scene(scene: VirtualScene, *, arm_wakeword: bool) -> list[np.ndarray]:
    callbacks: list[np.ndarray] = []
    rec = make_recorder(
        callback=lambda captured: callbacks.append(captured),
        wakeword_enabled=True,
        wakeword_policy="strict_required",
        silence_duration=0.05,
        aec_enabled=False,
    )
    wakeword = MockWakewordService()
    rec._wakeword_service = wakeword
    with AudioHarness(rec) as harness:
        if arm_wakeword:
            wakeword.arm()
        harness.inject(scene.audio, inter_chunk_delay=0.02)
        harness.inject(silence(0.35), inter_chunk_delay=0.02)
        harness.drain(timeout=5.0)
    return callbacks


def test_virtual_real_world_corpus_has_required_metadata_and_coverage():
    names = [scene.name for scene in SCENES]
    hashes = [
        hashlib.sha256((CORPUS_DIR / f"{scene.name}.npy").read_bytes()).hexdigest()
        for scene in SCENES
    ]
    by_kind = Counter(scene.kind for scene in SCENES)
    rooms = {scene.room_profile for scene in SCENES}
    devices = {scene.device_artifact for scene in SCENES}

    assert METADATA["case_count"] == len(SCENES)
    assert len(SCENES) >= 100
    assert len(names) == len(set(names))
    assert len(hashes) == len(set(hashes))
    assert by_kind["target_user"] >= 40
    assert by_kind["background_human"] >= 40
    assert by_kind["assistant_echo"] >= 4
    assert len(rooms) >= 5
    assert len(devices) >= 5
    assert (CORPUS_DIR / "metadata.json").exists()


@pytest.mark.parametrize("scene", BACKGROUND_SCENES[:12], ids=lambda scene: scene.name)
def test_virtual_background_human_speech_documents_ungated_limit(scene: VirtualScene):
    callbacks = _finish_recording_callbacks(scene.audio)

    assert scene.category == "unsolvable_without_intent_gate"
    assert scene.solvable is False
    assert callbacks, (
        f"{scene.description} should demonstrate the ungated limitation: "
        "raw human speech can look like a directed user turn."
    )


@pytest.mark.parametrize("scene", BACKGROUND_SCENES[:12], ids=lambda scene: scene.name)
def test_virtual_background_human_speech_blocked_by_wakeword_gate(scene: VirtualScene):
    callbacks = _run_wakeword_scene(scene, arm_wakeword=False)

    assert callbacks == [], (
        f"{scene.description} must be blocked when strict wakeword gating "
        "supplies the missing intent signal."
    )


@pytest.mark.parametrize("scene", TARGET_SCENES[:12], ids=lambda scene: scene.name)
def test_virtual_target_user_allowed_after_wakeword(scene: VirtualScene):
    callbacks = _run_wakeword_scene(scene, arm_wakeword=True)

    assert callbacks, f"{scene.description} should start a turn after wakeword."


@pytest.mark.parametrize("scene", SCENES[:60], ids=lambda scene: scene.name)
def test_virtual_intent_gate_accepts_only_assistant_directed_scenes(scene: VirtualScene):
    accepted = passive_intent_gate(scene)

    if scene.kind == "target_user":
        assert accepted, scene.description
    else:
        assert not accepted, scene.description
