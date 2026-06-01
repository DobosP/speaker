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
import time
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
        volume: float = 1.0,
        noise_snr_db: Optional[float] = None,
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
        # Normalize the device the same way the engine does: '' -> system default,
        # a digit string -> the int index PortAudio expects (not a device name).
        from core.engines.sherpa import _norm_device

        self._out = _norm_device(output_device)
        # Playback amplitude scale in [0, 1]. 1.0 == native TTS amplitude
        # (today's behavior). Lowering it reduces the acoustic level hitting the
        # near-field mic -- one half of the over-the-air SNR knob. Only the PLAYED
        # buffer is scaled; the returned/saved clip stays full-scale (a clean
        # reference), so volume changes the loop, never the artifact.
        try:
            self._volume = float(volume)
        except (TypeError, ValueError):
            self._volume = 1.0
        self._volume = max(0.0, min(1.0, self._volume))
        # Optional broadband noise overlaid on the PLAYED buffer at this SNR (dB)
        # relative to the spoken line -- the controlled-noise half of the denoise
        # A/B (run denoise off vs on at the same SNR). None == clean (no noise).
        # Only the played buffer is corrupted; the saved/returned clip stays clean.
        try:
            self._noise_snr = float(noise_snr_db) if noise_snr_db is not None else None
        except (TypeError, ValueError):
            self._noise_snr = None

    @property
    def voice(self) -> dict:
        return {
            "speaker_id": self._sid, "speed": self._speed, "volume": self._volume,
            "noise_snr_db": self._noise_snr,
        }

    def _add_noise(self, play):
        """Overlay deterministic white noise on the play buffer at ``_noise_snr`` dB
        SNR relative to the line's own RMS. Deterministic (fixed seed) so an A/B's
        off/on runs see the SAME noise. Returns the buffer unchanged when no SNR set."""
        import numpy as np

        if self._noise_snr is None or play.size == 0:
            return play
        sig_rms = float(np.sqrt(np.mean(play.astype("float64") ** 2))) or 1e-6
        target_noise_rms = sig_rms / (10.0 ** (self._noise_snr / 20.0))
        rng = np.random.default_rng(20260601)
        noise = rng.standard_normal(play.shape[0]).astype("float32")
        noise *= target_noise_rms / (float(np.sqrt(np.mean(noise ** 2))) or 1e-6)
        return np.clip(play + noise, -1.0, 1.0).astype("float32")

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
        finished playing. Returns the ORIGINAL (samples, sample_rate) so the
        caller can save the clean audio, even if playback was resampled."""
        samples, sr = self.synthesize(text)
        import sounddevice as sd

        # The output device may not support the TTS native rate (e.g. 22050 Hz on
        # a device that only opens at 48000) -- the engine resamples its own TTS
        # for exactly this reason. Resample playback to the device's default rate
        # and fall back across a couple of common rates on a bad-rate error.
        play, play_sr = samples, sr
        try:
            info = sd.query_devices(self._out, "output")
            dev_sr = int(info.get("default_samplerate") or 0)
        except Exception:  # noqa: BLE001
            dev_sr = 0
        candidates = [r for r in (dev_sr, 48000, 44100, sr) if r]
        last_err: Exception | None = None
        for rate in dict.fromkeys(candidates):  # dedupe, keep order
            try:
                play = _resample(samples, sr, rate) if rate != sr else samples
                # Scale ONLY the buffer we play over the air; the returned
                # (samples, sr) stays full-scale so save_wav keeps a clean
                # reference clip independent of the acoustic level used.
                if self._volume != 1.0:
                    play = (play * self._volume).astype("float32")
                # Overlay controlled noise (denoise A/B) on the played buffer only.
                play = self._add_noise(play)
                # The assistant's engine may still be releasing the shared output
                # device (acoustic mode hands it back and forth). A transient
                # "Device unavailable" is not a bad rate -- retry the SAME rate a
                # few times before moving on, so we don't misread a race as a
                # rate-unsupported error and resample needlessly.
                for attempt in range(5):
                    try:
                        sd.play(play, rate, device=self._out)
                        sd.wait()
                        return samples, sr
                    except Exception as exc:  # noqa: BLE001
                        last_err = exc
                        if "unavailable" not in str(exc).lower() and "-9985" not in str(exc):
                            raise
                        time.sleep(0.15)  # device busy: let the engine finish closing
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                continue
        raise RuntimeError(f"could not play synthetic-user audio: {last_err}")


def _resample(samples, src_sr: int, dst_sr: int):
    import numpy as np

    x = np.asarray(samples, dtype="float32").reshape(-1)
    if src_sr == dst_sr or x.size == 0:
        return x
    n = int(round(x.shape[0] * float(dst_sr) / float(src_sr)))
    if n <= 0:
        return np.zeros(0, dtype="float32")
    idx = np.linspace(0.0, x.shape[0] - 1, num=n)
    return np.interp(idx, np.arange(x.shape[0]), x).astype("float32")


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
