"""
Session Recorder — capture live conversations for automated regression testing.

When attached to an ``AudioRecorder``, each TTS turn is saved together with
the mic audio received during playback and any barge-in events.  Recordings
can then be replayed as deterministic regression tests via:

    python scripts/generate_session_tests.py   # write pytest files from recordings
    python scripts/analyze_sessions.py          # print analytics report
    pytest tests/test_recorded_sessions.py      # replay-based regression suite

Directory layout::

    recordings/
    ├── session_20260208_143022/
    │   ├── metadata.json          # session-level stats (SR, noise floor, profile …)
    │   ├── turn_000/
    │   │   ├── tts_16k.npy        # TTS audio resampled to 16 kHz (for tests)
    │   │   ├── mic_16k.npy        # mic audio during TTS, resampled to 16 kHz
    │   │   └── turn.json          # per-turn events: barge_in_fired, RMS, echo_sim …
    │   ├── turn_001/
    │   │   └── …
    │   └── …
    └── aggregate.json             # cross-session summary for agents / CI dashboards

Usage in main.py::

    session_rec = SessionRecorder(recorder, profile="high")
    session_rec.start()
    try:
        run_assistant()
    finally:
        session_rec.stop()
        path = session_rec.save()
        print(f"Session recorded to {path}")
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


RECORDINGS_DIR = Path(__file__).parent.parent / "recordings"
TARGET_SR = 16_000  # tests always run at 16 kHz


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class BargeInEvent:
    timestamp_sec: float        # seconds since session start
    rms: float
    threshold: float
    voiced: bool
    echo_sim: float
    tts_turn_index: int         # which TTS turn was active


@dataclass
class TtsTurn:
    index: int
    start_sec: float            # seconds since session start
    end_sec: float = 0.0
    text: str = ""
    barge_in_fired: bool = False
    barge_in_events: List[BargeInEvent] = field(default_factory=list)
    mic_rms_mean: float = 0.0
    mic_rms_max: float = 0.0
    echo_sim_mean: float = 0.0


# ── Resampler helper ──────────────────────────────────────────────────────────

def _resample(audio: np.ndarray, from_sr: int, to_sr: int = TARGET_SR) -> np.ndarray:
    if from_sr == to_sr:
        return audio.astype(np.float32)
    if LIBROSA_AVAILABLE:
        return librosa.resample(
            audio.astype(np.float32), orig_sr=from_sr, target_sr=to_sr
        )
    # Naive nearest-neighbour fallback (low quality, only used when librosa missing)
    ratio = to_sr / from_sr
    out_len = int(len(audio) * ratio)
    indices = (np.arange(out_len) / ratio).astype(int)
    indices = np.clip(indices, 0, len(audio) - 1)
    return audio[indices].astype(np.float32)


# ── SessionRecorder ───────────────────────────────────────────────────────────

class SessionRecorder:
    """
    Records one live assistant session for later replay as regression tests.

    Attach *before* calling ``recorder.start()``; the recorder must still be
    running when ``save()`` is called.

    Args:
        recorder:  The ``AudioRecorder`` instance to monitor.
        profile:   Runtime profile string (for metadata).
    """

    def __init__(self, recorder=None, profile: str = "default"):
        self.recorder = recorder  # may be None until lazy attachment
        self.profile = profile

        ts = time.strftime("%Y%m%d_%H%M%S")
        self.session_id = f"session_{ts}"
        self.session_dir = RECORDINGS_DIR / self.session_id

        self._start_time: float = 0.0
        self._active: bool = False

        # Per-turn state
        self._turns: List[TtsTurn] = []
        self._current_turn: Optional[TtsTurn] = None

        # Accumulated mic chunks per turn (native sample rate)
        self._mic_chunks_native: List[np.ndarray] = []
        self._mic_echo_sims: List[float] = []

        # Pending TTS audio buffer (native SR, set by on_tts_start)
        self._pending_tts_native: Optional[np.ndarray] = None

        # Per-turn saved audio (native SR, indexed by turn index)
        self._turn_tts: Dict[int, np.ndarray] = {}
        self._turn_mic: Dict[int, np.ndarray] = {}

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self):
        """
        Begin recording.

        May be called either at construction time (if recorder is already
        available) or lazily on the first TTS turn once the recorder is live.
        """
        self._start_time = time.time()
        self._active = True

        if self.recorder is None:
            return  # hooks will be installed when recorder is attached later

        self._attach_recorder_hooks()

    def _attach_recorder_hooks(self):
        """Install on_interrupt and chunk hooks on self.recorder."""
        if self.recorder is None:
            return
        if getattr(self.recorder, "_session_recorder_hook_installed", False):
            return
        orig = getattr(self.recorder, "on_interrupt", None)

        def _interrupt_hook(info=None):
            if self._active:
                self._on_barge_in(info or {})
            if orig:
                return orig(info)

        self.recorder.on_interrupt = _interrupt_hook
        self.recorder._session_recorder = self
        self.recorder._session_recorder_hook_installed = True

    def stop(self):
        """Stop recording (does not save — call save() separately)."""
        self._active = False
        if self._current_turn is not None:
            self._flush_turn()

    # ── turn callbacks (called by main.py / VoiceAssistant) ──────────────────

    def on_tts_start(self, audio: np.ndarray, sample_rate: int, text: str = ""):
        """Call when TTS audio begins playing.  ``audio`` is the raw TTS array."""
        if not self._active:
            return
        # Chunked TTS (AudioPlayer._speak_chunked) invokes on_start once per sentence.
        # Without flushing, each new chunk overwrote _current_turn and dropped mic/TTS.
        if self._current_turn is not None:
            self._flush_turn()
        self._pending_tts_native = audio.flatten().astype(np.float32)
        turn = TtsTurn(
            index=len(self._turns),
            start_sec=time.time() - self._start_time,
            text=text,
        )
        self._current_turn = turn
        self._mic_chunks_native = []
        self._mic_echo_sims = []

    def on_tts_end(self):
        """Call when TTS playback finishes."""
        if not self._active:
            return
        self._flush_turn()

    def on_mic_chunk(
        self,
        chunk: np.ndarray,
        rms: float,
        echo_sim: float,
    ):
        """Called per mic chunk received while TTS is playing."""
        if not self._active or self._current_turn is None:
            return
        self._mic_chunks_native.append(chunk.copy())
        self._mic_echo_sims.append(echo_sim)

    # ── internal helpers ─────────────────────────────────────────────────────

    def _on_barge_in(self, info: Dict[str, Any]):
        if self._current_turn is None:
            return
        evt = BargeInEvent(
            timestamp_sec=time.time() - self._start_time,
            rms=float(info.get("rms", 0.0)),
            threshold=float(info.get("threshold", 0.0)),
            voiced=bool(info.get("voiced", False)),
            echo_sim=float(info.get("echo_sim", 0.0)),
            tts_turn_index=self._current_turn.index,
        )
        self._current_turn.barge_in_fired = True
        self._current_turn.barge_in_events.append(evt)

    def _flush_turn(self):
        """Finalise the current turn and store audio buffers."""
        turn = self._current_turn
        if turn is None:
            return
        turn.end_sec = time.time() - self._start_time

        if self._mic_chunks_native:
            mic = np.concatenate(self._mic_chunks_native).astype(np.float32)
            turn.mic_rms_mean = float(np.sqrt(np.mean(mic ** 2)))
            turn.mic_rms_max = float(np.max(np.abs(mic)))
            self._turn_mic[turn.index] = mic

        if self._mic_echo_sims:
            turn.echo_sim_mean = float(np.mean(self._mic_echo_sims))

        if self._pending_tts_native is not None:
            self._turn_tts[turn.index] = self._pending_tts_native
            self._pending_tts_native = None

        self._turns.append(turn)
        self._current_turn = None
        self._mic_chunks_native = []
        self._mic_echo_sims = []

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self) -> Path:
        """
        Write all recorded data to disk and update the aggregate log.

        Returns the path to the session directory.
        """
        os.makedirs(self.session_dir, exist_ok=True)

        sr_native = getattr(self.recorder, "device_sample_rate", TARGET_SR)

        for turn in self._turns:
            turn_dir = self.session_dir / f"turn_{turn.index:03d}"
            os.makedirs(turn_dir, exist_ok=True)

            # TTS audio
            if turn.index in self._turn_tts:
                tts_native = self._turn_tts[turn.index]
                tts_16k = _resample(tts_native, sr_native)
                np.save(str(turn_dir / "tts_16k.npy"), tts_16k)

            # Mic audio
            if turn.index in self._turn_mic:
                mic_native = self._turn_mic[turn.index]
                mic_16k = _resample(mic_native, sr_native)
                np.save(str(turn_dir / "mic_16k.npy"), mic_16k)

            # Turn metadata
            turn_meta = asdict(turn)
            # Convert BargeInEvent lists (they're already dicts via asdict)
            with open(turn_dir / "turn.json", "w") as f:
                json.dump(turn_meta, f, indent=2)

        # Session metadata
        noise_floor = getattr(self.recorder, "_noise_floor", None)
        meta = {
            "session_id": self.session_id,
            "start_time": self._start_time,
            "end_time": time.time(),
            "duration_sec": time.time() - self._start_time,
            "noise_floor": noise_floor,
            "device_sample_rate": sr_native,
            "target_test_sample_rate": TARGET_SR,
            "profile": self.profile,
            "num_tts_turns": len(self._turns),
            "num_barge_ins": sum(1 for t in self._turns if t.barge_in_fired),
            "false_positive_candidates": [
                asdict(evt)
                for t in self._turns
                for evt in t.barge_in_events
                # Heuristic: barge-in during TTS with no real user speech following
                # (voiced=False or very short) is likely a false positive candidate
                if not evt.voiced or evt.echo_sim > 0.20
            ],
            "turns": [asdict(t) for t in self._turns],
        }
        with open(self.session_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        self._update_aggregate(meta)
        return self.session_dir

    def _update_aggregate(self, meta: Dict[str, Any]):
        agg_path = RECORDINGS_DIR / "aggregate.json"
        agg: Dict[str, Any] = {"sessions": [], "summary": {}}
        if agg_path.exists():
            try:
                with open(agg_path) as f:
                    agg = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        entry = {
            "session_id": meta["session_id"],
            "start_time": meta["start_time"],
            "duration_sec": meta["duration_sec"],
            "noise_floor": meta["noise_floor"],
            "profile": meta["profile"],
            "num_tts_turns": meta["num_tts_turns"],
            "num_barge_ins": meta["num_barge_ins"],
            "false_positive_candidates": len(meta["false_positive_candidates"]),
        }
        # Replace existing entry if session_id matches (re-run)
        agg["sessions"] = [
            s for s in agg["sessions"] if s.get("session_id") != meta["session_id"]
        ]
        agg["sessions"].append(entry)

        sessions = agg["sessions"]
        total_bi = sum(s["num_barge_ins"] for s in sessions)
        total_fp = sum(s["false_positive_candidates"] for s in sessions)
        total_turns = sum(s["num_tts_turns"] for s in sessions)
        agg["summary"] = {
            "total_sessions": len(sessions),
            "total_tts_turns": total_turns,
            "total_barge_ins": total_bi,
            "total_false_positive_candidates": total_fp,
            "false_positive_rate": round(total_fp / max(total_turns, 1), 4),
            "avg_barge_ins_per_session": round(total_bi / max(len(sessions), 1), 2),
            "last_session": sessions[-1]["session_id"],
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        os.makedirs(RECORDINGS_DIR, exist_ok=True)
        with open(agg_path, "w") as f:
            json.dump(agg, f, indent=2)
