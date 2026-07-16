"""Session recorder: write the exact 16 kHz mono audio the recognizer hears to a
WAV file, so a recorded run can be replayed bit-for-bit through the real
pipeline (``python -m core --engine replay``) and frozen into a regression test.

Concurrency: the real-time capture thread only enqueues a copy of each block
(cheap); a dedicated writer thread does the float32->int16 conversion and the
disk writes, so audio I/O never stalls the hot path. The two threads talk over
a bounded :class:`queue.Queue`.

Kill-safe on purpose (2026-07-06, run-20260706-231226): the writer patches the
RIFF/data sizes and flushes every ``flush_sec``, so the file on disk is a VALID
WAV at (almost) every instant. A run that dies to SIGTERM/SIGKILL -- which never
reaches ``close()`` -- still leaves playable audio up to the last flush. Stdlib
``wave`` could not do this (it only patches sizes in ``close()``), which is how
that run's diagnosis-critical audio evidence was lost; hence the manual header.

WAV (not MP3) on purpose: the replay engine needs lossless PCM, and 16 kHz mono
int16 is already compact (~32 KB/s). No new dependency.
"""
from __future__ import annotations

import os
import queue
import struct
import threading
import time
from typing import Optional


def _wav_header(sample_rate: int, data_bytes: int) -> bytes:
    """44-byte canonical PCM WAV header: mono, int16, ``sample_rate``."""
    return b"".join(
        (
            b"RIFF",
            struct.pack("<I", 36 + data_bytes),
            b"WAVE",
            b"fmt ",
            struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16),
            b"data",
            struct.pack("<I", data_bytes),
        )
    )


def sidecar_wav_path(record_path: str, tag: str) -> str:
    """Return a sibling WAV path without assuming ``record_path``'s suffix.

    ``tag`` is code-owned (for example ``"ref"`` or ``"pre-dsp"``), not a
    user-supplied filename.  Keeping this transform shared prevents the runtime
    and run-summary metadata from silently naming different artifacts.
    """
    suffix = f".{tag}.wav"
    if record_path.lower().endswith(".wav"):
        return record_path[:-4] + suffix
    return record_path + suffix


class WavRecorder:
    """Background-threaded 16-bit PCM WAV writer fed float32 blocks.

    The on-disk file stays a valid WAV throughout the recording (header patched
    + flushed every ``flush_sec``), so audio evidence survives a killed run."""

    def __init__(
        self,
        path: str,
        sample_rate: int = 16000,
        *,
        queue_max: int = 4096,
        flush_sec: float = 2.0,
    ):
        self.path = path
        self.sample_rate = sample_rate
        self.frames = 0  # frames accepted for writing (updated on the hot path)
        self.dropped = 0  # frames dropped if the writer ever falls behind
        self._closed = False
        self._flush_sec = max(0.1, float(flush_sec))
        # Raw voice is private even when the low-level core entry point is run
        # outside the launcher's restrictive umask.  O_NOFOLLOW also prevents a
        # stale/surprising symlink at the generated artifact path from redirecting
        # the recording.  Windows lacks O_NOFOLLOW/fchmod; its ACLs remain the
        # authority there, while chmod is still attempted for portable intent.
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        flags |= getattr(os, "O_BINARY", 0)
        fd = os.open(path, flags, 0o600)
        try:
            if hasattr(os, "fchmod"):
                os.fchmod(fd, 0o600)
            else:  # pragma: no cover - Windows permission semantics
                os.chmod(path, 0o600)
            self._fh: Optional[object] = os.fdopen(fd, "w+b")
        except BaseException:
            os.close(fd)
            raise
        self._fh.write(_wav_header(sample_rate, 0))
        self._data_bytes = 0
        self._last_flush = time.monotonic()
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
                self._maybe_flush()  # keep the header fresh through silence too
                continue
            if self._fh is None:
                continue
            pcm = np.clip(block, -1.0, 1.0)
            payload = (pcm * 32767.0).astype("<i2").tobytes()
            self._fh.write(payload)
            self._data_bytes += len(payload)
            self._maybe_flush()

    def _maybe_flush(self) -> None:
        """Patch the RIFF/data sizes and flush, at most once per ``flush_sec``.
        Writer-thread only. After this the file is a valid WAV up to every byte
        written so far -- the whole point: a SIGTERM/SIGKILL between flushes
        costs at most ``flush_sec`` of tail audio, never the file."""
        now = time.monotonic()
        if now - self._last_flush < self._flush_sec or self._fh is None:
            return
        self._patch_sizes()
        self._fh.flush()
        self._last_flush = now

    def _patch_sizes(self) -> None:
        fh = self._fh
        fh.seek(4)
        fh.write(struct.pack("<I", 36 + self._data_bytes))
        fh.seek(40)
        fh.write(struct.pack("<I", self._data_bytes))
        fh.seek(0, 2)  # back to the append position

    @property
    def seconds(self) -> float:
        return self.frames / float(self.sample_rate or 1)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._thread.join(timeout=3.0)  # drains remaining queued blocks first
        if self._fh is not None:
            self._patch_sizes()
            self._fh.flush()
            self._fh.close()
            self._fh = None
