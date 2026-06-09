"""Shared, audio-free fixture + driver module for the barge-in test suite.

Built FIRST (the "foundation"); every barge-in test file imports from here and
contains NO test logic of its own. This module only loads the recorded live
failure and drives the REAL decision chain over it.

WHAT IT WRAPS
-------------
The single recorded live failure is ``run-20260609-203236``: the owner talked
over the assistant's TTS on the bare ALC285 laptop speaker. The fused-z-score
``AdaptiveDTD`` *fired* on the talk-over, yet the downstream capture-loop
integrator + one-per-run latch only converted a fire into an actual cut on the
turn-3 SHOUT (raw=0.0704, ~10x normal talk volume); the turn-2 NORMAL-volume
talk-over (raw 0.0024-0.0068) was missed (summary.json: turn-2
``barge_in_latency=null``, turn-3 ``0.0001``; two "barge-in REJECTED" log lines).
That is exactly the open P1 in CLAUDE.md: a normal talk-over MUST cut without a
shout, and the assistant must NOT self-interrupt on its own echo.

The high-fidelity audio-free replay is the per-frame DTD trace: every ``dtd:``
DEBUG line carries the EXACT (raw_rms, resid_rms, incoherent_fraction) the live
detector saw, so feeding those triples back through the REAL
``core.engines._dtd.AdaptiveDTD`` reproduces the live verdicts with no audio,
no models and no sound card.

VERIFIED DURING THE BUILD (load-bearing facts)
----------------------------------------------
* The trace has EXACTLY 204 ``dtd:`` lines, 9 of them ``fired=True``.
* The ``fired=True`` 0-based indices are ``(10, 11, 14, 34, 188, 189, 194,
  195, 196)`` -- confirmed by parsing the real file (NOT the planning estimate).
* Replaying the parsed (raw, resid, incoh) triples through a fresh
  ``build_live_dtd()`` reproduces ALL 204 ``fired`` verdicts EXACTLY (204/204).
* BUT the recomputed ``.last_D`` does NOT match the logged ``D`` (the live charts
  carried pre-trace EWMA state). Assert on the ``decide()`` bool / ``last_decided``
  -- NEVER on ``last_D``.
* Turn boundaries (inter-frame gaps > 1.0s) are at indices ``(88, 130, 185,
  197)``.
* Live params (sherpa.py:363-383 dataclass defaults, not overridden in
  config.local.json) -> ``AdaptiveDTD(k=5.0, weights=(0.2, 1.0, 0.0),
  confirm_frames=1, warmup_frames=5, chart_rel_floor=0.4)``; integrator
  ``barge_in_min_speech_sec=0.3`` (config.local.json override of the 0.2
  default), ``block_sec=0.1`` + ``decay=0.5`` hardcoded (sherpa.py:1404-1426),
  ``barge_in_refractory_sec=0.5`` + ``barge_in_suppress_sec=0.5``.

This file is named ``barge_fixtures.py`` (not ``test_*``) so pytest does not
collect it as a test module.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

# REAL detector under test -- imported, never reimplemented.
from core.engines._dtd import AdaptiveDTD

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
_REPO_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "barge_in"

#: The full DEBUG trace of the recorded live failure (committed copy).
TRACE_TXT = _FIXTURE_DIR / "run-20260609-203236.trace.txt"
#: Canonical pre-parsed frames (committed; regenerated from TRACE_TXT).
FRAMES_JSON = _FIXTURE_DIR / "run-20260609-203236.frames.json"
#: Short mono-16k clip spanning the turn-2 normal-volume talk-over (optional,
#: for a future real_model replay; none of the Tier0 tests need audio).
TALKOVER_WAV = _FIXTURE_DIR / "run-20260609-203236.talkover.wav"
#: Backwards-compatible alias used by some planning notes.
TRIMMED_WAV = TALKOVER_WAV

#: The full-length recorded WAV (lives in logs/, may be absent in CI checkouts).
FULL_WAV = _REPO_ROOT / "logs" / "runs" / "run-20260609-203236.wav"

# --------------------------------------------------------------------------- #
# Ground-truth constants (VERIFIED against the real trace, not estimated)
# --------------------------------------------------------------------------- #
#: 0-based indices of the 9 frames the live DTD scored fired=True.
LIVE_FIRE_INDICES: Tuple[int, ...] = (10, 11, 14, 34, 188, 189, 194, 195, 196)
#: Frame indices that begin a new turn (preceding inter-frame gap > 1.0s).
TURN_BOUNDARY_IDX: Tuple[int, ...] = (88, 130, 185, 197)

#: Live integrator / latch parameters (the capture-loop accumulator values).
LIVE_INTEGRATOR_PARAMS = dict(
    min_speech_sec=0.3,    # config.local.json barge_in_min_speech_sec (override of 0.2)
    block_sec=0.1,         # hardcoded block_sec (sherpa.py capture loop)
    decay=0.5,             # hardcoded leaky-integrator decay (sherpa.py:1426)
    refractory_sec=0.5,    # config.local.json barge_in_refractory_sec
    suppress_sec=0.5,      # config.json barge_in_suppress_sec
)

# Annotation labels written into the JSON fixture.
_TALKOVER_TURN2 = "talkover_turn2_normal"   # idx 10..14, raw 0.0024-0.0068
_TALKOVER_TURN3 = "talkover_turn3_shout"    # idx 185..196, escalates to 0.0704
_ECHO_ONLY = "echo_only"


# --------------------------------------------------------------------------- #
# Frame model + loader
# --------------------------------------------------------------------------- #
@dataclass
class Frame:
    """One capture block's DTD decision inputs + the recorded verdict.

    ``raw``/``resid``/``incoh`` are the REAL inputs to ``AdaptiveDTD.decide``;
    ``exp_*`` are what the live detector logged for that block (the ground truth
    a replay must reproduce).
    """

    idx: int
    ts: str                # 'HH:MM:SS.ffffff'
    t_sec: float           # seconds from frame 0
    raw: float             # raw_rms
    resid: float           # resid_rms (post-AEC)
    incoh: float           # incoherent_fraction
    exp_D: float           # logged fused decision statistic
    exp_fired: bool        # logged fired verdict (ground truth)
    exp_consec: int        # logged consecutive-fire counter
    exp_z_raw: float = 0.0
    exp_z_resid: float = 0.0
    exp_z_coh: float = 0.0
    annotation: str = _ECHO_ONLY


# Regex VERIFIED to match all 204 'dtd:' lines.
_DTD_RE = re.compile(
    r"(\d\d:\d\d:\d\d\.\d+).*dtd: D=([\d.]+) K=[\d.]+ fired=(True|False) "
    r"\(z_raw=(-?[\d.]+) z_resid=(-?[\d.]+) z_coh=(-?[\d.]+)\) "
    r"raw=([\d.]+) resid=([\d.]+) incoh=([\d.]+) consec=(\d+)"
)


def _annotate(idx: int) -> str:
    """Echo-only vs talk-over, per the VERIFIED evidence windows."""
    if 10 <= idx <= 14:
        return _TALKOVER_TURN2
    if 185 <= idx <= 196:
        return _TALKOVER_TURN3
    return _ECHO_ONLY


def _parse_trace(path: Path) -> List[Frame]:
    """Regex-parse every 'dtd:' line in the DEBUG trace into Frames."""
    text = path.read_text()
    raw_rows = []
    for line in text.splitlines():
        m = _DTD_RE.search(line)
        if m:
            raw_rows.append(m.groups())
    times = [datetime.strptime(r[0], "%H:%M:%S.%f") for r in raw_rows]
    t0 = times[0] if times else None
    frames: List[Frame] = []
    for i, r in enumerate(raw_rows):
        ts, D, fired, z_raw, z_resid, z_coh, raw, resid, incoh, consec = r
        frames.append(
            Frame(
                idx=i,
                ts=ts,
                t_sec=round((times[i] - t0).total_seconds(), 6),
                raw=float(raw),
                resid=float(resid),
                incoh=float(incoh),
                exp_D=float(D),
                exp_fired=fired == "True",
                exp_consec=int(consec),
                exp_z_raw=float(z_raw),
                exp_z_resid=float(z_resid),
                exp_z_coh=float(z_coh),
                annotation=_annotate(i),
            )
        )
    return frames


def _load_json_frames(path: Path) -> List[Frame]:
    """Load the canonical committed JSON parse."""
    doc = json.loads(path.read_text())
    frames: List[Frame] = []
    for f in doc["frames"]:
        frames.append(
            Frame(
                idx=f["idx"],
                ts=f["t"],
                t_sec=f["t_sec"],
                raw=f["raw"],
                resid=f["resid"],
                incoh=f["incoh"],
                exp_D=f["D"],
                exp_fired=f["fired"],
                exp_consec=f["consec"],
                exp_z_raw=f.get("z_raw", 0.0),
                exp_z_resid=f.get("z_resid", 0.0),
                exp_z_coh=f.get("z_coh", 0.0),
                annotation=f.get("annotation", _annotate(f["idx"])),
            )
        )
    return frames


def load_trace_frames() -> List[Frame]:
    """Return all 204 Frames of the recorded live failure.

    Prefers the canonical committed JSON parse (``FRAMES_JSON``) and falls back
    to re-parsing the DEBUG trace (``TRACE_TXT``). Asserts the count loudly so a
    truncated/corrupted fixture fails immediately, before any logic assertion.
    """
    if FRAMES_JSON.exists():
        frames = _load_json_frames(FRAMES_JSON)
    else:
        frames = _parse_trace(TRACE_TXT)
    assert len(frames) == 204, (
        f"expected 204 dtd frames, got {len(frames)} -- truncated trace fixture?"
    )
    parsed_fires = tuple(f.idx for f in frames if f.exp_fired)
    assert parsed_fires == LIVE_FIRE_INDICES, (
        f"fire indices drifted: {parsed_fires} != {LIVE_FIRE_INDICES}"
    )
    return frames


def turns(frames: Sequence[Frame]) -> List[List[Frame]]:
    """Split frames into turns at the verified > 1.0s inter-frame gaps."""
    out: List[List[Frame]] = []
    cur: List[Frame] = []
    for f in frames:
        if f.idx in TURN_BOUNDARY_IDX and cur:
            out.append(cur)
            cur = []
        cur.append(f)
    if cur:
        out.append(cur)
    return out


# --------------------------------------------------------------------------- #
# Selectors
# --------------------------------------------------------------------------- #
def talkover_frames(frames: Sequence[Frame]) -> dict:
    """The two human talk-over bursts, keyed so a test can target either.

    ``turn2_normal`` (idx 10..14, raw 0.0024-0.0068): a NORMAL-volume talk-over.
    ``turn3_shout`` (idx 185..196, escalating to raw=0.0704): the SHOUT that was
    the only thing that actually cut the assistant live.
    """
    return {
        "turn2_normal": [f for f in frames if f.annotation == _TALKOVER_TURN2],
        "turn3_shout": [f for f in frames if f.annotation == _TALKOVER_TURN3],
    }


def echo_only_frames(frames: Sequence[Frame]) -> List[Frame]:
    """Every frame that is NOT inside a talk-over burst (the echo floor).

    Includes the short pre-turn-2 echo floor, the long ~85s TTS playback
    stretch, and the post-shout echo tail. These must NEVER self-interrupt.
    """
    return [f for f in frames if f.annotation == _ECHO_ONLY]


# --------------------------------------------------------------------------- #
# REAL DTD construction
# --------------------------------------------------------------------------- #
def build_live_dtd() -> AdaptiveDTD:
    """Construct the EXACT live ``AdaptiveDTD`` (sherpa.py:752 + the dataclass
    defaults that config.local.json does not override).

    NOTE for callers: replaying the trace inputs through a fresh
    ``build_live_dtd()`` reproduces all 204 ``exp_fired`` verdicts EXACTLY, but
    ``.last_D`` drifts from the logged D (the live chart carried pre-trace EWMA
    state). Assert on ``decide()`` / ``.last_decided`` -- NOT on ``.last_D``.
    """
    return AdaptiveDTD(
        k=5.0,
        weights=(0.2, 1.0, 0.0),
        confirm_frames=1,
        warmup_frames=5,
        chart_rel_floor=0.4,
    )


# --------------------------------------------------------------------------- #
# Full-chain driver: REAL decide() + the mirrored capture-loop integrator
# --------------------------------------------------------------------------- #
@dataclass
class RunResult:
    """Outcome of walking the real chain frame-by-frame."""

    fires: List[Tuple[int, float]] = field(default_factory=list)  # (idx, t_sec)
    dtd_fire_count: int = 0
    _burst_start_t: Optional[float] = None  # t_sec of the first frame fed

    @property
    def first_fire_index(self) -> Optional[int]:
        return self.fires[0][0] if self.fires else None

    @property
    def first_fire_t_sec(self) -> Optional[float]:
        return self.fires[0][1] if self.fires else None

    @property
    def first_fire_latency_sec(self) -> Optional[float]:
        """Latency from the start of the fed burst to the first cut."""
        if not self.fires or self._burst_start_t is None:
            return None
        return self.fires[0][1] - self._burst_start_t

    def as_tuple(self) -> Tuple[Optional[int], Optional[float]]:
        """``(fired_index_or_None, latency_sec)`` for tuple-style callers."""
        return self.first_fire_index, self.first_fire_latency_sec


def run_frames(
    frames: Sequence[Frame],
    dtd: Optional[AdaptiveDTD] = None,
    *,
    vad_speech=True,
    reset_latch_per_turn: bool = False,
    params: Optional[dict] = None,
) -> RunResult:
    """Walk the REAL barge chain frame-by-frame and report the cuts.

    The seam under test -- ``AdaptiveDTD.decide`` -- is the REAL detector. The
    only mirrored part is the ~6-line capture-loop accumulator (latch + leaky
    integrator), which is NOT a separately importable function today; it is
    replicated here EXACTLY as core/engines/sherpa.py:1404-1426 so a refactor
    that extracts it can swap this for the import.

    For each frame:
      decided  = dtd.decide(raw, resid, incoh)            # REAL
      eligible = decided and vad(frame) and not latch     # sherpa.py:2134-2138
      if eligible: voiced_run += block_sec                # sherpa.py:1404
                   if voiced_run >= min_speech_sec:        # sherpa.py:1406
                       voiced_run = 0; latch = True        # sherpa.py:1407,1411
                       record a fire (-> on_barge_in)      # sherpa.py:1416
      else:        voiced_run *= decay                     # sherpa.py:1426

    ``vad_speech`` may be a bool or a ``Callable[[Frame], bool]``. The latch is
    only cleared by a turn boundary when ``reset_latch_per_turn`` (modelling the
    silent->speaking re-arm; live, ``stop_speaking`` clears it on a real cut).
    """
    if dtd is None:
        dtd = build_live_dtd()
    p = dict(LIVE_INTEGRATOR_PARAMS)
    if params:
        p.update(params)
    block_sec = float(p["block_sec"])
    min_speech_sec = float(p["min_speech_sec"])
    decay = float(p["decay"])

    def _vad_ok(fr: Frame) -> bool:
        return vad_speech(fr) if callable(vad_speech) else bool(vad_speech)

    result = RunResult()
    result._burst_start_t = frames[0].t_sec if frames else None
    voiced_run = 0.0
    latch = False

    for fr in frames:
        if reset_latch_per_turn and fr.idx in TURN_BOUNDARY_IDX:
            # silent->speaking re-arm: a new turn starts with a fresh latch
            latch = False
            voiced_run = 0.0

        decided = dtd.decide(fr.raw, fr.resid, fr.incoh)  # REAL AdaptiveDTD.decide
        if decided:
            result.dtd_fire_count += 1

        # sherpa.py:2134 -> latch checked FIRST; sherpa.py:2136 VAD gate.
        eligible = decided and _vad_ok(fr) and not latch
        if eligible:
            voiced_run += block_sec                         # sherpa.py:1404
            if voiced_run >= min_speech_sec:                # sherpa.py:1406
                voiced_run = 0.0                            # sherpa.py:1407
                latch = True                                # sherpa.py:1411
                result.fires.append((fr.idx, fr.t_sec))     # -> on_barge_in (1416)
        else:
            voiced_run *= decay                             # sherpa.py:1426

    return result


# --------------------------------------------------------------------------- #
# REAL gate seam: a SherpaOnnxEngine driving _looks_like_user / _barge_in_fire_eligible
# --------------------------------------------------------------------------- #
def make_block(rms_value: float, n: int = 1600):
    """A numpy block whose ``rms()`` equals ``rms_value`` (np.full -> constant).

    The REAL gate computes ``rms(samples)`` / ``rms(mic_raw)`` (core/engines/
    speaker_gate.rms), and the RMS of a constant array is that constant, so a
    frame's ``raw``/``resid`` become real numpy blocks the real gate measures
    back to the same level. ``n=1600`` ~ one 0.1s block at 16kHz.
    """
    import numpy as np

    return np.full(int(n), float(rms_value), dtype=np.float32)


class _FakeCoherence:
    """Stub coherence detector: ``_looks_like_user`` only reads
    ``last_incoherent_fraction`` from it (after calling ``decide``) to feed the
    DTD's incoherent-fraction feature. We expose a settable fraction so a caller
    can replay a frame's ``incoh`` through the REAL DTD path."""

    def __init__(self, incoherent_fraction: float = 0.9) -> None:
        self.last_incoherent_fraction = float(incoherent_fraction)

    def decide(self, mic_raw):  # noqa: D401 - matches EchoCoherenceDetector
        return None  # verdict unused on the DTD path; only the fraction matters


class _FakeVad:
    """Stub VAD: ``_barge_in_fire_eligible`` only calls ``is_speech_detected()``."""

    def __init__(self, speech: bool = True) -> None:
        self._speech = bool(speech)

    def is_speech_detected(self) -> bool:
        return self._speech

    def set_speech(self, speech: bool) -> None:
        self._speech = bool(speech)


def live_engine_with_dtd(*, incoherent_fraction: float = 0.9, vad_speech: bool = True):
    """A REAL ``SherpaOnnxEngine`` wired for the live DTD barge path, built
    WITHOUT loading any audio model.

    The engine is created via ``object.__new__`` (bypassing ``__init__`` ->
    ``_build`` which loads ONNX models) and only the attributes the gate seam
    touches are populated:

    * ``_dtd``           = ``build_live_dtd()``  (the REAL detector)
    * ``_echo_coherence``= a fake supplying the incoherent-fraction feature
    * ``_aec``           = a non-None sentinel (so PATH A -- DTD -- is taken)
    * ``_vad``           = a fake reporting speech
    * latch / floor / refractory state at their post-build defaults

    The REAL ``_looks_like_user`` (sherpa.py:1929) and
    ``_barge_in_fire_eligible`` (sherpa.py:2120) then run unmodified over
    ``make_block`` inputs whose rms equals a frame's raw/resid. Use
    ``engine._fake_coherence.last_incoherent_fraction = f`` and
    ``engine._fake_vad.set_speech(...)`` to vary the per-block inputs.
    """
    from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine

    eng = object.__new__(SherpaOnnxEngine)
    # Live-shaped config (the gate reads a handful of barge thresholds off it).
    eng.config = SherpaConfig(
        dtd_enabled=True,
        aec_enabled=True,
        aec_backend="dtln",
        coherence_barge_in_enabled=True,
        barge_in_enabled=True,
        barge_in_min_speech_sec=0.3,
        barge_in_refractory_sec=0.5,
        barge_in_suppress_sec=0.5,
        barge_in_residual_margin_db=10.0,
        input_loudness_margin_db=18.0,
    )
    # REAL detector under test.
    eng._dtd = build_live_dtd()
    # Stubs the gate reads (PATH A: DTD active because _aec + coherence non-None).
    eng._fake_coherence = _FakeCoherence(incoherent_fraction)
    eng._echo_coherence = eng._fake_coherence
    eng._aec = object()  # sentinel: only its non-None-ness gates PATH A
    eng._fake_vad = _FakeVad(vad_speech)
    eng._vad = eng._fake_vad
    # Barge state at post-build defaults.
    eng._barge_in_fired_this_run = False
    eng._barge_in_suppressed_until = 0.0
    eng._last_speaking_end = 0.0
    eng._raw_playback_floor_rms = 0.0
    eng._playback_floor_rms = 0.0
    eng._ambient_rms = 0.0
    eng._input_loudness_margin_db = 18.0
    return eng
