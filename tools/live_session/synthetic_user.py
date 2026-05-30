"""The synthetic user: a TTS voice, distinct from the assistant's, that speaks
scripted lines aloud through the real speakers (so the assistant hears them over
the air, exactly like a person in the room).

Distinct voice: a different speaker id (multi-speaker models) and/or a different
speed than the assistant uses, so a human listening to the recording can tell the
two apart. Attribution itself never relies on the acoustic difference -- the
harness *controls* the user audio, so it always knows which side is which.
"""
from __future__ import annotations

import logging
import wave
from pathlib import Path
from typing import Optional

log = logging.getLogger("speaker.live.user")


class SyntheticUser:
    def __init__(
        self,
        sherpa_config,
        *,
        speaker_id: Optional[int] = None,
        speed: Optional[float] = None,
        output_device=None,
    ) -> None:
        # Build a second TTS instance for the user voice from the same sherpa
        # config the assistant uses (so the model is already on disk). build_tts
        # returns None when no TTS model is configured -- a clear, early failure.
        from core.engines._sherpa_models import build_tts

        self._tts = build_tts(sherpa_config)
        if self._tts is None:
            raise RuntimeError(
                "the synthetic user needs a TTS voice but sherpa.tts_model is not "
                "configured -- set the sherpa TTS model paths in config.json"
            )
        assistant_sid = int(getattr(sherpa_config, "tts_speaker_id", 0) or 0)
        assistant_speed = float(getattr(sherpa_config, "tts_speed", 1.0) or 1.0)
        # Default the user voice to a *different* speaker id when the model is
        # multi-speaker; always nudge the speed so single-speaker models still
        # sound distinct. The user can override both.
        self._sid = speaker_id if speaker_id is not None else assistant_sid + 1
        self._speed = speed if speed is not None else round(assistant_speed * 1.12, 3)
        self._out = output_device

    @property
    def voice(self) -> dict:
        return {"speaker_id": self._sid, "speed": self._speed}

    def synthesize(self, text: str):
        """Synthesize ``text`` to (samples, sample_rate) without playing it."""
        import numpy as np

        # A bad speaker id on a single-speaker model raises; fall back to 0.
        try:
            audio = self._tts.generate(text, sid=self._sid, speed=self._speed)
        except Exception:  # noqa: BLE001
            log.warning("speaker id %d rejected; falling back to sid 0", self._sid)
            self._sid = 0
            audio = self._tts.generate(text, sid=self._sid, speed=self._speed)
        samples = np.asarray(audio.samples, dtype="float32").reshape(-1)
        sr = int(getattr(audio, "sample_rate", 0)) or 22050
        return samples, sr

    def say(self, text: str):
        """Synthesize + play ``text`` through the speakers, BLOCKING until it has
        finished playing. Returns (samples, sample_rate) so the caller can save
        the exact audio that was played."""
        samples, sr = self.synthesize(text)
        import sounddevice as sd

        sd.play(samples, sr, device=self._out)
        sd.wait()
        return samples, sr


def save_wav(samples, sample_rate: int, path: Path) -> Path:
    """Write float32 mono ``samples`` to a 16-bit PCM WAV. Best-effort."""
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(np.asarray(samples, dtype="float32"), -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm.tobytes())
    return path
