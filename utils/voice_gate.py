"""
Optional wakeword and speaker-identity gates for noisy environments.

These gates are dependency-safe:
- If optional libraries are not installed, the gate reports unavailable.
- The caller can choose whether to disable the gate or fail closed.
"""

from __future__ import annotations

import tempfile
import wave
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from openwakeword.model import Model as OpenWakeWordModel

    OPENWAKEWORD_AVAILABLE = True
except ImportError:
    OPENWAKEWORD_AVAILABLE = False

try:
    from speechbrain.inference.speaker import SpeakerRecognition

    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False

try:
    import soundfile as sf

    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    import sounddevice as sd

    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

KNOWN_OPENWAKEWORD_MODELS = (
    "alexa",
    "hey_jarvis",
    "hey_mycroft",
    "hey_rhasspy",
    "timer",
    "weather",
)


def _resample(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return audio.astype(np.float32)
    if LIBROSA_AVAILABLE:
        return librosa.resample(
            audio.astype(np.float32), orig_sr=source_sr, target_sr=target_sr
        ).astype(np.float32)
    # Lightweight fallback: linear interpolation.
    if len(audio) < 2:
        return audio.astype(np.float32)
    ratio = float(target_sr) / float(source_sr)
    n_out = max(int(round(len(audio) * ratio)), 1)
    x_old = np.linspace(0.0, 1.0, num=len(audio), dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=n_out, dtype=np.float32)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def _sample_rate_candidates(preferred: int, device_default: Optional[int]) -> list[int]:
    """Return de-duplicated sample-rate fallbacks in priority order."""
    ordered: list[int] = []
    for rate in (
        preferred,
        device_default,
        16000,
        48000,
        44100,
        32000,
        24000,
        22050,
        8000,
    ):
        if rate is None:
            continue
        try:
            value = int(rate)
        except Exception:
            continue
        if value <= 0:
            continue
        if value not in ordered:
            ordered.append(value)
    return ordered


def list_known_wakewords() -> list[str]:
    """Return known openWakeWord model labels."""
    return list(KNOWN_OPENWAKEWORD_MODELS)


def validate_wakeword_name(name: Optional[str], available_labels: list[str]) -> tuple[bool, str]:
    """Validate a requested wakeword against discovered labels."""
    if not name:
        return True, ""
    requested = name.strip().lower()
    if not requested:
        return True, ""
    labels = [str(x).strip().lower() for x in available_labels if str(x).strip()]
    if requested in labels:
        return True, ""
    for label in labels:
        if requested in label:
            return True, ""
    if not labels:
        return False, (
            f"Wakeword '{name}' not validated because no labels were discovered. "
            "Run with --list-wakewords or set --wakeword-model-path explicitly."
        )
    return False, (
        f"Unknown wakeword '{name}'. Available labels: {', '.join(labels)}"
    )


@dataclass
class OpenWakeWordGate:
    threshold: float = 0.5
    model_path: Optional[str] = None
    wakeword: Optional[str] = None

    def __post_init__(self):
        self.available = False
        self._model = None
        self.last_score = 0.0
        self.available_labels: list[str] = []
        if not OPENWAKEWORD_AVAILABLE:
            return
        kwargs = {}
        if self.model_path:
            kwargs["wakeword_models"] = [self.model_path]
        try:
            self._model = OpenWakeWordModel(**kwargs)
            self.available_labels = self._discover_labels()
            self.available = True
        except Exception:
            self._model = None
            self.available = False

    def detect(self, audio_chunk: np.ndarray, sample_rate: int) -> bool:
        if not self.available or self._model is None:
            return False
        audio_16k = _resample(audio_chunk.flatten(), sample_rate, 16000)
        # openWakeWord expects PCM-like float values in [-1, 1].
        audio_16k = np.clip(audio_16k.astype(np.float32), -1.0, 1.0)
        try:
            scores = self._model.predict(audio_16k)
        except Exception:
            return False
        top = self._extract_top_score(scores, self.wakeword)
        self.last_score = top
        return top >= self.threshold

    def _discover_labels(self) -> list[str]:
        if self._model is None:
            return []
        try:
            # Prime a single inference to get output keys.
            sample = np.zeros(1600, dtype=np.float32)
            scores = self._model.predict(sample)
            if isinstance(scores, dict):
                labels = [str(k).strip() for k in scores.keys() if str(k).strip()]
                if labels:
                    return labels
        except Exception:
            pass
        # Fallback: expose curated known labels for UX.
        return list_known_wakewords()

    @staticmethod
    def _extract_top_score(scores, wakeword: Optional[str] = None) -> float:
        wakeword_l = wakeword.lower().strip() if wakeword else None
        if isinstance(scores, dict):
            values = []
            for key, value in scores.items():
                if wakeword_l and wakeword_l not in str(key).lower():
                    continue
                if isinstance(value, dict):
                    if "score" in value:
                        values.append(float(value["score"]))
                    else:
                        for nested_key, nested in value.items():
                            if wakeword_l and wakeword_l not in str(nested_key).lower():
                                continue
                            try:
                                values.append(float(nested))
                            except Exception:
                                continue
                else:
                    try:
                        values.append(float(value))
                    except Exception:
                        continue
            return max(values) if values else 0.0
        try:
            return float(scores)
        except Exception:
            return 0.0


@dataclass
class SpeechBrainSpeakerVerifier:
    enrollment_wav: str
    threshold: float = 0.55

    def __post_init__(self):
        self.available = False
        self._verifier = None
        self.last_score = 0.0
        self._enrollment = Path(self.enrollment_wav)
        if not SPEECHBRAIN_AVAILABLE or not self._enrollment.exists():
            return
        try:
            self._verifier = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
            )
            self.available = True
        except Exception:
            self._verifier = None
            self.available = False

    def verify(self, audio: np.ndarray, sample_rate: int) -> bool:
        if not self.available or self._verifier is None:
            return False
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            self._write_wav(tmp_path, audio, sample_rate)
            score, _ = self._verifier.verify_files(str(self._enrollment), tmp_path)
            try:
                value = float(score.item())
            except Exception:
                value = float(score)
            self.last_score = value
            return value >= self.threshold
        except Exception:
            return False
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass

    @staticmethod
    def _write_wav(path: str, audio: np.ndarray, sample_rate: int):
        audio = np.clip(audio.flatten().astype(np.float32), -1.0, 1.0)
        if SOUNDFILE_AVAILABLE:
            sf.write(path, audio, sample_rate)
            return
        pcm = (audio * 32767.0).astype(np.int16)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())


def create_speaker_enrollment_wav(
    output_path: str,
    duration_sec: float = 6.0,
    sample_rate: int = 16000,
    device: Optional[int] = None,
    countdown_sec: float = 2.0,
) -> dict:
    """Record a speaker enrollment sample and save it as WAV."""
    if duration_sec <= 0:
        return {"ok": False, "reason": "duration_sec must be > 0"}
    if not SOUNDDEVICE_AVAILABLE:
        return {"ok": False, "reason": "sounddevice is not available"}

    if countdown_sec > 0:
        print(f"Recording starts in {countdown_sec:.1f}s...")
        time.sleep(countdown_sec)

    device_default_rate = None
    try:
        if device is not None:
            info = sd.query_devices(device)
        else:
            info = sd.query_devices(kind="input")
        if isinstance(info, dict):
            device_default_rate = int(info.get("default_samplerate", 0)) or None
    except Exception:
        device_default_rate = None

    candidates = _sample_rate_candidates(sample_rate, device_default_rate)
    audio = None
    used_sample_rate = None
    last_error = None
    for rate in candidates:
        try:
            sd.check_input_settings(
                device=device,
                channels=1,
                dtype="float32",
                samplerate=rate,
            )
            print(
                f"Recording enrollment voice for {duration_sec:.1f}s at {rate}Hz..."
            )
            frames = int(rate * duration_sec)
            recording = sd.rec(
                frames,
                samplerate=rate,
                channels=1,
                dtype=np.float32,
                device=device,
            )
            sd.wait()
            audio = recording.flatten().astype(np.float32)
            used_sample_rate = rate
            break
        except Exception as e:
            last_error = e
            continue

    if audio is None or used_sample_rate is None:
        return {
            "ok": False,
            "reason": (
                "recording failed: no compatible input sample rate "
                f"(tried {candidates}); last error: {last_error}"
            ),
        }

    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak < 1e-4:
        return {"ok": False, "reason": "captured audio is too quiet"}
    # Gentle peak normalization for more stable embeddings.
    if peak > 0:
        audio = np.clip(audio / max(peak, 1e-6) * 0.8, -1.0, 1.0)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    SpeechBrainSpeakerVerifier._write_wav(str(output), audio, used_sample_rate)
    return {
        "ok": True,
        "path": str(output),
        "duration_sec": duration_sec,
        "sample_rate": used_sample_rate,
        "peak": peak,
    }
