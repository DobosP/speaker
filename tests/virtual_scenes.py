"""Virtual real-world acoustic scenes for background-speech testing."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from tests.fixtures import (
    SR,
    apply_gain_db,
    babble_noise,
    hard_clip,
    human_voice,
    mix,
    music_noise,
    nonstationary_noise,
    real_tts_echo,
    reverberant_echo,
    sample_rate_roundtrip,
    silence,
    tv_noise,
)


@dataclass(frozen=True)
class VirtualScene:
    name: str
    kind: str
    expectation: str
    audio: np.ndarray
    source: str
    speaker: str
    distance_m: float
    snr_db: float | None
    room_profile: str
    device_artifact: str
    transcript_type: str
    category: str
    solvable: bool
    mitigation: str
    description: str
    importance: str
    echo_reference: np.ndarray | None = None

    def metadata(self) -> dict[str, object]:
        audio = self.audio.astype(np.float32)
        return {
            "name": self.name,
            "kind": self.kind,
            "expectation": self.expectation,
            "source": self.source,
            "speaker": self.speaker,
            "distance_m": self.distance_m,
            "snr_db": self.snr_db,
            "room_profile": self.room_profile,
            "device_artifact": self.device_artifact,
            "transcript_type": self.transcript_type,
            "category": self.category,
            "solvable": self.solvable,
            "mitigation": self.mitigation,
            "description": self.description,
            "importance": self.importance,
            "samples": int(len(audio)),
            "sample_rate": SR,
            "rms": round(float(np.sqrt(np.mean(audio**2))), 6),
            "sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
        }

    def save(self, directory: Path) -> dict[str, object]:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / f"{self.name}.npy", self.audio.astype(np.float32))
        return self.metadata()


class VirtualSceneBuilder:
    """Build repeatable real-world scenes from open speech and synthetic rooms."""

    speakers = ("jackson", "nicolas", "george", "theo")
    digits = (0, 1, 2)

    def __init__(self, duration_sec: float = 1.0):
        self.duration_sec = duration_sec

    def build(self) -> list[VirtualScene]:
        scenes: list[VirtualScene] = []
        for speaker in self.speakers:
            for digit in self.digits:
                base = human_voice(
                    self.duration_sec,
                    amplitude=0.20,
                    speaker=speaker,
                    digit=digit,
                )
                scenes.extend(self._target_user_scenes(speaker, digit, base))
                scenes.extend(self._background_human_scenes(speaker, digit, base))
        scenes.extend(self._assistant_echo_scenes())
        return scenes

    def save(self, directory: Path, scenes: list[VirtualScene]) -> dict[str, object]:
        directory.mkdir(parents=True, exist_ok=True)
        for stale in directory.glob("*.npy"):
            stale.unlink()
        metadata = {
            "version": 1,
            "dataset": "FSDD human speech with deterministic virtual rooms/devices",
            "case_count": len(scenes),
            "sample_rate": SR,
            "cases": [scene.save(directory) for scene in scenes],
        }
        (directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        return metadata

    def _target_user_scenes(
        self,
        speaker: str,
        digit: int,
        base: np.ndarray,
    ) -> list[VirtualScene]:
        transforms = {
            "near_clean": (base, 0.4, None, "dry", "none"),
            "kitchen_tv": (
                mix(base, tv_noise(self.duration_sec, amplitude=0.035, seed=digit), 8.0),
                1.2,
                8.0,
                "kitchen_tile",
                "none",
            ),
            "car_resampled": (
                sample_rate_roundtrip(apply_gain_db(base, -3.0), device_sr=48_000),
                0.8,
                None,
                "car_cabin",
                "48khz_roundtrip",
            ),
            "over_babble": (
                mix(base, babble_noise(self.duration_sec, amplitude=0.035), 7.0),
                1.0,
                7.0,
                "living_room",
                "none",
            ),
        }
        scenes = []
        for label, (audio, distance_m, snr_db, room, device) in transforms.items():
            name = f"virtual_target_{speaker}_{digit}_{label}"
            scenes.append(
                VirtualScene(
                    name=name,
                    kind="target_user",
                    expectation="callback",
                    audio=self._unique_floor(name, audio),
                    source=f"FSDD:{digit}_{speaker}_0.wav",
                    speaker=speaker,
                    distance_m=distance_m,
                    snr_db=snr_db,
                    room_profile=room,
                    device_artifact=device,
                    transcript_type="assistant_directed_command",
                    category="target_user",
                    solvable=True,
                    mitigation="normal_listening_or_wakeword_gate",
                    description=f"{speaker} digit {digit} intentionally addresses the assistant in {label}.",
                    importance="Real user speech must still start a turn across normal room and device variation.",
                )
            )
        return scenes

    def _background_human_scenes(
        self,
        speaker: str,
        digit: int,
        base: np.ndarray,
    ) -> list[VirtualScene]:
        far = apply_gain_db(base, -12.0)
        transforms = {
            "podcast_across_room": (
                mix(far, tv_noise(self.duration_sec, amplitude=0.025, seed=digit + 40), 4.0),
                3.0,
                4.0,
                "living_room",
                "phone_speaker",
            ),
            "roommate_kitchen": (
                mix(far, nonstationary_noise(self.duration_sec, 0.012, 0.035), 5.0),
                4.0,
                5.0,
                "kitchen_tile",
                "none",
            ),
            "video_call_speaker": (
                hard_clip(apply_gain_db(far, 7.0), limit=0.12),
                2.4,
                None,
                "office",
                "clipped_speaker",
            ),
            "music_and_speech": (
                mix(far, music_noise(self.duration_sec, amplitude=0.035), 3.0),
                3.6,
                3.0,
                "bedroom",
                "bluetooth_speaker",
            ),
        }
        scenes = []
        for label, (audio, distance_m, snr_db, room, device) in transforms.items():
            name = f"virtual_background_{speaker}_{digit}_{label}"
            scenes.append(
                VirtualScene(
                    name=name,
                    kind="background_human",
                    expectation="no_callback",
                    audio=self._unique_floor(name, audio),
                    source=f"FSDD:{digit}_{speaker}_0.wav",
                    speaker=speaker,
                    distance_m=distance_m,
                    snr_db=snr_db,
                    room_profile=room,
                    device_artifact=device,
                    transcript_type="passive_background_speech",
                    category="unsolvable_without_intent_gate",
                    solvable=False,
                    mitigation="wakeword_or_speaker_or_intent_gate",
                    description=f"Background human speech from {speaker} digit {digit} in {label}.",
                    importance=(
                        "Raw ungated audio cannot prove whether this human speech "
                        "is directed at the assistant; an external intent signal is required."
                    ),
                )
            )
        return scenes

    def _assistant_echo_scenes(self) -> list[VirtualScene]:
        ref = real_tts_echo(1.2, amplitude=0.10)
        transforms = {
            "living_room_wall": reverberant_echo(ref, direct_delay_ms=18.0, rt60_ms=320.0),
            "car_dashboard": reverberant_echo(ref, direct_delay_ms=8.0, rt60_ms=180.0),
            "phone_table": hard_clip(reverberant_echo(ref, direct_delay_ms=4.0), limit=0.18),
            "sample_rate_artifact": sample_rate_roundtrip(reverberant_echo(ref), device_sr=44_100),
        }
        scenes = []
        for label, audio in transforms.items():
            name = f"virtual_echo_{label}"
            scenes.append(
                VirtualScene(
                    name=name,
                    kind="assistant_echo",
                    expectation="no_interrupt",
                    audio=self._unique_floor(name, audio),
                    source="TTS fixture",
                    speaker="assistant",
                    distance_m=1.5,
                    snr_db=None,
                    room_profile=label,
                    device_artifact="speaker_to_mic_path",
                    transcript_type="assistant_self_audio",
                    category="echo_alignment_gap",
                    solvable=True,
                    mitigation="lag_aware_echo_reference_or_aec",
                    description=f"Assistant TTS echo through virtual {label}.",
                    importance="Assistant playback should not trigger self barge-in.",
                    echo_reference=ref,
                )
            )
        return scenes

    @staticmethod
    def _unique_floor(name: str, audio: np.ndarray) -> np.ndarray:
        seed = int(hashlib.sha256(name.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal(len(audio)).astype(np.float32) * 1e-6
        return (audio.astype(np.float32) + noise).astype(np.float32)


def passive_intent_gate(scene: VirtualScene) -> bool:
    """Test-local intent gate: only assistant-directed scenes are accepted."""
    return scene.transcript_type == "assistant_directed_command"


def pad_with_silence(audio: np.ndarray, trailing_sec: float = 0.35) -> np.ndarray:
    return np.concatenate([audio.astype(np.float32), silence(trailing_sec)])
