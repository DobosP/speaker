"""Session recorder: write the exact 16 kHz mono audio the recognizer hears to a
WAV file, so a recorded run can be replayed bit-for-bit through the real
pipeline (``python -m core --engine replay``) and frozen into a regression test.

WAV (not MP3) on purpose: the replay engine needs lossless PCM, and 16 kHz mono
int16 is already compact (~32 KB/s). Uses only the stdlib ``wave`` module, so
recording adds no dependency.
"""
from __future__ import annotations

import wave
from typing import Optional


class WavRecorder:
    """Incremental 16-bit PCM WAV writer fed float32 blocks from the capture loop."""

    def __init__(self, path: str, sample_rate: int = 16000):
        self.path = path
        self.sample_rate = sample_rate
        self.frames = 0
        self._wf: Optional[wave.Wave_write] = wave.open(path, "wb")
        self._wf.setnchannels(1)
        self._wf.setsampwidth(2)  # int16
        self._wf.setframerate(sample_rate)

    def write(self, samples) -> None:
        """Append a float32 mono block (values in [-1, 1]) as int16 PCM."""
        if self._wf is None:
            return
        import numpy as np

        pcm = np.clip(np.asarray(samples, dtype="float32").reshape(-1), -1.0, 1.0)
        pcm = (pcm * 32767.0).astype("<i2")
        self._wf.writeframes(pcm.tobytes())
        self.frames += int(pcm.shape[0])

    @property
    def seconds(self) -> float:
        return self.frames / float(self.sample_rate or 1)

    def close(self) -> None:
        if self._wf is not None:
            self._wf.close()
            self._wf = None
