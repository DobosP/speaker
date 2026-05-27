"""Session recorder: write the exact 16 kHz mono audio the recognizer hears to a
WAV file, so a recorded run can be replayed bit-for-bit through the real
pipeline (``python -m core --engine replay``) and frozen into a regression test.

Concurrency: the real-time capture thread only enqueues a copy of each block
(cheap); a dedicated writer thread does the float32->int16 conversion and the
disk writes, so audio I/O never stalls the hot path. The two threads talk over
a bounded :class:`queue.Queue`.

WAV (not MP3) on purpose: the replay engine needs lossless PCM, and 16 kHz mono
int16 is already compact (~32 KB/s). Stdlib ``wave`` only -- no new dependency.
"""
from __future__ import annotations

import queue
import threading
import wave
from typing import Optional


class WavRecorder:
    """Background-threaded 16-bit PCM WAV writer fed float32 blocks."""

    def __init__(self, path: str, sample_rate: int = 16000, *, queue_max: int = 4096):
        self.path = path
        self.sample_rate = sample_rate
        self.frames = 0  # frames accepted for writing (updated on the hot path)
        self.dropped = 0  # frames dropped if the writer ever falls behind
        self._closed = False
        self._wf: Optional[wave.Wave_write] = wave.open(path, "wb")
        self._wf.setnchannels(1)
        self._wf.setsampwidth(2)  # int16
        self._wf.setframerate(sample_rate)
        self._q: "queue.Queue" = queue.Queue(maxsize=queue_max)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="wav-recorder", daemon=True)
        self._thread.start()

    def write(self, samples) -> None:
        """Enqueue a float32 mono block (in [-1, 1]). Non-blocking: the capture
        thread copies + hands off and returns immediately."""
        if self._closed:
            return
        import numpy as np

        block = np.array(samples, dtype="float32").reshape(-1)  # copy off the buffer
        try:
            self._q.put_nowait(block)
            self.frames += int(block.shape[0])
        except queue.Full:
            self.dropped += int(block.shape[0])

    def _run(self) -> None:
        import numpy as np

        while True:
            try:
                block = self._q.get(timeout=0.1)
            except queue.Empty:
                if self._stop.is_set():
                    break
                continue
            if self._wf is None:
                continue
            pcm = np.clip(block, -1.0, 1.0)
            self._wf.writeframes((pcm * 32767.0).astype("<i2").tobytes())

    @property
    def seconds(self) -> float:
        return self.frames / float(self.sample_rate or 1)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._thread.join(timeout=3.0)  # drains remaining queued blocks first
        if self._wf is not None:
            self._wf.close()
            self._wf = None
