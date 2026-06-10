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
* Live DTD params (SherpaConfig dataclass defaults) -> ``AdaptiveDTD(k=5.0,
  weights=(0.2, 1.0, 0.0), confirm_frames=1, warmup_frames=5,
  chart_rel_floor=0.4)``. Temporal confirmation is the REAL
  ``BargeSustain(window_sec=0.5, block_sec=0.1, min_voiced_sec=0.2)`` -- the
  windowed sustain that replaced the old leaky ``voiced_run *= 0.5`` accumulator
  (the fix for the recorded miss; ``barge_in_min_speech_sec`` is now 0.2, the
  shipped default). ``barge_in_refractory_sec=0.5`` + ``barge_in_suppress_sec=0.5``.

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

# REAL detector + REAL temporal-confirmation integrator under test -- imported,
# never reimplemented.
from core.engines._dtd import AdaptiveDTD, BargeSustain

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
_REPO_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "barge_in"

#: The full DEBUG trace of the recorded live failure (committed copy).
TRACE_TXT = _FIXTURE_DIR / "run-20260609-203236.trace.txt"
#: Canonical pre-parsed frames (committed; regenerated from TRACE_TXT).
FRAMES_JSON = _FIXTURE_DIR / "run-20260609-203236.frames.json"

#: The SELF-INTERRUPT live failure (2026-06-10 fix target). On the SAME starved
#: laptop mic + open speaker, the assistant cut ITSELF off twice on reply-onset
#: echo while speaking (barge-in detected at 23:46:54.969 and 23:46:57.255).
#: Two consecutive replies each fired the DTD on 2 echo transients within a 0.5s
#: window, tripping BargeSustain(need=2). The raw/residual levels of those echo
#: fires (raw 0.0026-0.0058, resid 0.0008-0.0018) OVERLAP a real talk-over's raw
#: but sit at/just-above the residual echo floor -- the residual-floor gate is
#: what separates them. No committed JSON: parsed live from the trace.
TRACE_TXT_SELF = _FIXTURE_DIR / "run-20260609-234435.trace.txt"
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
#: ``echo_only``-annotated frames that are in fact short residual ACOUSTIC
#: EVENTS, not clean TTS echo: their post-AEC residual stands 16-32x above the
#: local echo floor -- the same band as a normal-volume talk-over on this
#: starved mic (turn-2 resid 0.0024-0.0041), so no level-based detector can
#: separate them from a quiet interjection. idx 34 (resid 0.0104) fired even
#: LIVE; idx 111-112 (resid 0.0032/0.0016 over a 0.0001-0.0003 floor) only
#: stayed quiet live because the pre-fix charts had ABSORBED the earlier
#: talk-over (the contamination bug) -- their "safety" was the same defect as
#: the 4-second scream miss. Echo-safety assertions exclude them; the CLEAN
#: floor (everything else) is the non-regressable property.
ACOUSTIC_EVENT_IDX: Tuple[int, ...] = (34, 111, 112)

#: Live integrator / latch parameters -- the REAL ``BargeSustain`` knobs (the
#: capture-loop now uses ``core.engines._dtd.BargeSustain``, not a leaky
#: accumulator). These are the shipped SherpaConfig defaults.
LIVE_INTEGRATOR_PARAMS = dict(
    window_sec=0.5,        # barge_in_sustain_window_sec (SherpaConfig default)
    block_sec=0.1,         # hardcoded block_sec (sherpa.py capture loop)
    min_voiced_sec=0.2,    # barge_in_min_speech_sec (config.json default; the bug
                           #   fix lowered the config.local.json override 0.3 -> 0.2)
    refractory_sec=0.5,    # barge_in_refractory_sec
    suppress_sec=0.5,      # barge_in_suppress_sec
)

# Annotation labels written into the JSON fixture.
_TALKOVER_TURN2 = "talkover_turn2_normal"   # idx 10..14, raw 0.0024-0.0068
_TALKOVER_TURN3 = "talkover_turn3_shout"    # idx 185..196, escalates to 0.0704
_ECHO_ONLY = "echo_only"

# --------------------------------------------------------------------------- #
# run-20260609-234435 SELF-INTERRUPT ground truth (VERIFIED against the trace)
# --------------------------------------------------------------------------- #
#: 28 dtd frames; 4 of them fired=True. 0-based indices of the fires.
SELF_FIRE_INDICES: Tuple[int, ...] = (15, 18, 26, 27)
#: 0-based indices of the FIRST dtd frame of each reply (silent->speaking
#: transition, detected from the trace's ``speaking:`` markers). Reply A (idx
#: 0..7, no fire) preceded reply B (idx 8..18, the first self-interrupt) and
#: reply C (idx 19..27, the second self-interrupt).
SELF_SPEAKING_START_IDX: Tuple[int, ...] = (0, 8, 19)
#: The two reply-onset ECHO bursts that WRONGLY cut the assistant live. These are
#: the frames that must NOT produce a barge-in after the fix.
SELF_INTERRUPT_WINDOW_B = range(8, 19)   # reply B (fires at 15, 18)
SELF_INTERRUPT_WINDOW_C = range(19, 28)  # reply C (fires at 26, 27)


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
    #: True iff this is the FIRST dtd frame of a new reply (a silent->speaking
    #: transition). The engine re-arms its per-reply barge state here -- the
    #: drivers mirror that by resetting the learned floors + the BargeSustain
    #: window. Set by the 234435 loader from the trace's ``speaking:`` markers;
    #: the 203236 loader leaves it False (that fixture's turn boundaries are
    #: modelled via ``TURN_BOUNDARY_IDX`` + ``reset_latch_per_turn`` instead).
    reply_start: bool = False


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


# A ``speaking:`` marker line precedes the first dtd frame of every reply.
_SPEAKING_RE = re.compile(r"speaking: .* \(queue depth=\d+\)")


def _parse_self_trace(path: Path) -> List[Frame]:
    """Parse the run-20260609-234435 SELF-INTERRUPT trace into Frames.

    Identical dtd-line parse to ``_parse_trace``, but ALSO walks the interleaved
    ``speaking:`` markers so the first dtd frame after each marker is tagged
    ``reply_start=True`` -- the silent->speaking transition the engine re-arms its
    per-reply barge state on. That tag is what lets the driver mirror the fix
    (reset the learned floors + BargeSustain at each reply onset) faithfully,
    rather than hard-coding the boundaries.
    """
    text = path.read_text()
    rows = []           # (groups, reply_start_bool)
    pending_start = True  # the very first dtd frame begins reply A
    for line in text.splitlines():
        if _SPEAKING_RE.search(line):
            pending_start = True
            continue
        m = _DTD_RE.search(line)
        if m:
            rows.append((m.groups(), pending_start))
            pending_start = False
    times = [datetime.strptime(r[0][0], "%H:%M:%S.%f") for r in rows]
    t0 = times[0] if times else None
    frames: List[Frame] = []
    for i, (r, is_start) in enumerate(rows):
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
                annotation=_ECHO_ONLY,  # the WHOLE trace is the assistant's own echo
                reply_start=is_start,
            )
        )
    return frames


def load_self_interrupt_frames() -> List[Frame]:
    """Return the 28 Frames of the SELF-INTERRUPT live failure (234435).

    Every frame is the assistant's own TTS echo (no human talk-over in this run);
    the live detector wrongly cut twice on reply-onset echo. Asserts the shape
    (frame count, fire indices, reply-start indices) loudly so a corrupted fixture
    fails before any logic assertion.
    """
    frames = _parse_self_trace(TRACE_TXT_SELF)
    assert len(frames) == 28, (
        f"expected 28 dtd frames in 234435, got {len(frames)} -- truncated fixture?"
    )
    parsed_fires = tuple(f.idx for f in frames if f.exp_fired)
    assert parsed_fires == SELF_FIRE_INDICES, (
        f"234435 fire indices drifted: {parsed_fires} != {SELF_FIRE_INDICES}"
    )
    parsed_starts = tuple(f.idx for f in frames if f.reply_start)
    assert parsed_starts == SELF_SPEAKING_START_IDX, (
        f"234435 reply-start indices drifted: {parsed_starts} != "
        f"{SELF_SPEAKING_START_IDX}"
    )
    return frames


def self_interrupt_windows(frames: Sequence[Frame]) -> dict:
    """The two reply-onset ECHO bursts that wrongly cut the assistant live.

    ``reply_B`` (idx 8..18, fires at 15+18) and ``reply_C`` (idx 19..27, fires at
    26+27). Each is a full reply STARTING at its first frame (``reply_start``), so
    a driver fed one of these lists re-arms exactly as the engine did. These must
    produce NO barge-in cut after the fix.
    """
    return {
        "reply_B": [f for f in frames if f.idx in SELF_INTERRUPT_WINDOW_B],
        "reply_C": [f for f in frames if f.idx in SELF_INTERRUPT_WINDOW_C],
    }


# --------------------------------------------------------------------------- #
# run-20260610-003800 MISSED-TALK-OVER ground truth (the contamination failure)
# --------------------------------------------------------------------------- #
#: The 2026-06-10 plan's step-3 target: the owner's talk-overs were ABSORBED into
#: the DTD chart baselines. Two recorded misses: a normal talk-over at 00:45:08
#: (resid 0.0134-0.0449 vs a true echo floor ~0.0002-0.0008 -- z_resid logged
#: 0.00 on every frame, never fired) and the SCREAM at 00:46:15-19 (resid
#: 0.0443-0.1553, z_resid 0.00 for ~3.7s; the cut only landed at 00:46:19.26).
#: This trace logs the NEW dtd format (with gated=/resid_floor= fields).
TRACE_TXT_MISS = _FIXTURE_DIR / "run-20260610-003800.trace.txt"

# Regex for the gated dtd format this run logs (fired= and gated= both present).
_DTD_GATED_RE = re.compile(
    r"(\d\d:\d\d:\d\d\.\d+).*dtd: D=([\d.]+) K=[\d.]+ fired=(True|False) "
    r"gated=(?:True|False) "
    r"\(z_raw=(-?[\d.]+) z_resid=(-?[\d.]+) z_coh=(-?[\d.]+)\) "
    r"raw=([\d.]+) resid=([\d.]+) incoh=([\d.]+) resid_floor=[\d.]+ consec=(\d+)"
)


def load_miss_frames() -> List[Frame]:
    """All dtd frames of the run-20260610-003800 contamination failure, in
    order, with ``reply_start`` tagged from the interleaved ``speaking:``
    markers (same convention as the 234435 loader). Asserts the shape loudly."""
    text = TRACE_TXT_MISS.read_text()
    rows = []
    pending_start = True
    for line in text.splitlines():
        if _SPEAKING_RE.search(line):
            pending_start = True
            continue
        m = _DTD_GATED_RE.search(line)
        if m:
            rows.append((m.groups(), pending_start))
            pending_start = False
    assert rows, "no gated dtd lines parsed from the 003800 trace -- format drift?"
    times = [datetime.strptime(r[0][0], "%H:%M:%S.%f") for r in rows]
    t0 = times[0]
    frames: List[Frame] = []
    for i, (r, is_start) in enumerate(rows):
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
                reply_start=is_start,
            )
        )
    return frames


def miss_windows(frames: Sequence[Frame]) -> dict:
    """The two recorded missed-talk-over bursts, selected by wall-clock window.

    ``talkover_0045``: the 00:45:08-09 normal talk-over the live detector NEVER
    fired on (z_resid logged 0.00 against a warmup-poisoned baseline).
    ``scream_0046``: the 00:46:15-19 SCREAM that took ~3.7s to cut.
    """
    def _in(f: Frame, *prefixes: str) -> bool:
        return any(f.ts.startswith(p) for p in prefixes)

    return {
        "talkover_0045": [
            f for f in frames if _in(f, "00:45:08", "00:45:09")
        ],
        "scream_0046": [
            f for f in frames
            if _in(f, "00:46:15", "00:46:16", "00:46:17", "00:46:18", "00:46:19")
        ],
    }


def clean_echo_levels(frames: Sequence[Frame], *, max_resid: float = 0.001) -> List[Frame]:
    """Frames at the run's true echo floor (resid at/below ``max_resid`` --
    the genuine quiet playback level this device recorded between bursts).
    Used to model the engine's VAD-quiet ``observe_echo`` learning tap, which
    feeds the charts exactly these quiet blocks live but is absent from the
    trace (only VAD-speech blocks produced ``dtd:`` lines)."""
    return [f for f in frames if f.resid <= max_resid]


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


def frames_with_turn_starts(frames: Sequence[Frame]) -> List[Frame]:
    """The 203236 frames with ``reply_start`` set at the verified turn
    boundaries (``TURN_BOUNDARY_IDX``), so the ENGINE driver
    (``run_frames_engine``) re-arms per turn exactly as the live capture loop's
    silent->speaking transition does. The 203236 fixture predates the
    ``reply_start`` tag; this derives it from the same ground truth the
    ``turns()`` splitter uses."""
    out: List[Frame] = []
    for f in frames:
        if f.idx in TURN_BOUNDARY_IDX and not f.reply_start:
            from dataclasses import replace

            f = replace(f, reply_start=True)
        out.append(f)
    return out


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
def build_live_dtd(*, legacy: bool = False) -> AdaptiveDTD:
    """Construct the live ``AdaptiveDTD`` (sherpa.py construction + the dataclass
    defaults that config.local.json does not override).

    ``legacy=False`` (default) mirrors the SHIPPED engine -- including the
    2026-06-10 anti-contamination defaults (persistent charts, z-freeze,
    robust warm-up seed). ``legacy=True`` reproduces the detector EXACTLY as it
    ran in the recorded sessions (pre-fix math) -- use it ONLY for the
    parse/replay fidelity pin (``test_live_dtd_reproduces_every_recorded_fired_
    verdict``), where replaying the 203236 trace must reproduce all 204 recorded
    verdicts.

    NOTE for callers: ``.last_D`` drifts from the logged D (the live chart
    carried pre-trace EWMA state). Assert on ``decide()`` / ``.last_decided``
    -- NOT on ``.last_D``.
    """
    if legacy:
        return AdaptiveDTD(
            k=5.0,
            weights=(0.2, 1.0, 0.0),
            confirm_frames=1,
            warmup_frames=5,
            chart_rel_floor=0.4,
            chart_z_freeze=0.0,
            chart_robust_seed=False,
            persistent_charts=False,
        )
    return AdaptiveDTD(
        k=5.0,
        weights=(0.2, 1.0, 0.0),
        confirm_frames=1,
        warmup_frames=5,
        chart_rel_floor=0.4,
    )


def build_live_sustain(params: Optional[dict] = None) -> BargeSustain:
    """Construct the REAL ``BargeSustain`` with the live capture-loop knobs.

    This is the SAME class the engine's capture loop runs (core/engines/sherpa.py
    builds it from ``barge_in_sustain_window_sec`` + ``barge_in_min_speech_sec``),
    imported -- not reimplemented -- so the driver faithfully mirrors the engine.
    """
    p = dict(LIVE_INTEGRATOR_PARAMS)
    if params:
        p.update(params)
    return BargeSustain(
        window_sec=float(p["window_sec"]),
        block_sec=float(p["block_sec"]),
        min_voiced_sec=float(p["min_voiced_sec"]),
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

    BOTH seams under test are REAL, imported code: the per-frame verdict is
    ``AdaptiveDTD.decide`` and the temporal confirmation is
    ``BargeSustain.update`` -- the SAME class the engine's capture loop runs
    (core/engines/sherpa.py). Nothing is reimplemented here; the driver only wires
    the eligibility (latch + VAD gate, mirroring sherpa.py:2134-2138) into the real
    integrator and records the cuts.

    For each frame:
      decided  = dtd.decide(raw, resid, incoh)            # REAL AdaptiveDTD
      eligible = decided and vad(frame) and not latch     # sherpa.py:2134-2138
      if sustain.update(eligible):                        # REAL BargeSustain
          sustain.reset(); latch = True                   # sherpa.py fire path
          record a fire (-> on_barge_in)

    ``vad_speech`` may be a bool or a ``Callable[[Frame], bool]``. The latch is
    only cleared by a turn boundary when ``reset_latch_per_turn`` (modelling the
    silent->speaking re-arm; live, ``stop_speaking`` clears it on a real cut).
    """
    if dtd is None:
        dtd = build_live_dtd()
    sustain = build_live_sustain(params)  # REAL core.engines._dtd.BargeSustain

    def _vad_ok(fr: Frame) -> bool:
        return vad_speech(fr) if callable(vad_speech) else bool(vad_speech)

    result = RunResult()
    result._burst_start_t = frames[0].t_sec if frames else None
    latch = False

    for fr in frames:
        if reset_latch_per_turn and fr.idx in TURN_BOUNDARY_IDX:
            # silent->speaking re-arm: a new turn starts with a fresh latch + window
            latch = False
            sustain.reset()

        decided = dtd.decide(fr.raw, fr.resid, fr.incoh)  # REAL AdaptiveDTD.decide
        if decided:
            result.dtd_fire_count += 1

        # sherpa.py:2134 -> latch checked FIRST; sherpa.py:2136 VAD gate.
        eligible = decided and _vad_ok(fr) and not latch
        if sustain.update(eligible):                        # REAL BargeSustain.update
            sustain.reset()                                 # fire -> clear the window
            latch = True
            result.fires.append((fr.idx, fr.t_sec))         # -> on_barge_in

    return result


# --------------------------------------------------------------------------- #
# Full-chain ENGINE driver: REAL SherpaOnnxEngine._barge_in_fire_eligible
# --------------------------------------------------------------------------- #
def run_frames_engine(
    frames: Sequence[Frame],
    *,
    vad_speech=True,
    residual_floor_margin_db: Optional[float] = None,
    params: Optional[dict] = None,
) -> RunResult:
    """Walk the FULL real barge chain through the ENGINE seam and report cuts.

    This is the highest-fidelity audio-free driver: it drives the REAL
    ``SherpaOnnxEngine._barge_in_fire_eligible`` -> ``_looks_like_user`` (which now
    applies the 2026-06-10 residual-floor gate), the REAL ``_update_playback_floor``
    / ``_update_raw_playback_floor`` (so the gate keys off the SAME learned echo
    floor the engine learns), and the REAL ``core.engines._dtd.BargeSustain`` --
    NOTHING is reimplemented. Per frame it mirrors the capture loop exactly:

        eng._update_playback_floor(rms(resid))       # sherpa.py:1402
        eng._update_raw_playback_floor(rms(raw))     # sherpa.py:1410
        eligible = eng._barge_in_fire_eligible(...)  # sherpa.py:1433 (-> gated DTD)
        if sustain.update(eligible): cut             # sherpa.py:1434

    On a frame tagged ``reply_start`` it re-arms exactly as ``_playback_loop`` does
    on the silent->speaking transition: clears the BargeSustain window, zeroes the
    learned floors (re-bootstrap per reply), and drops the latch. ``mic_raw`` /
    ``samples`` are real numpy blocks whose rms equals the frame's raw / resid
    (see ``make_block``), so the real gate measures them back to those levels.
    """
    eng = live_engine_with_dtd()
    if residual_floor_margin_db is not None:
        eng.config.dtd_residual_floor_margin_db = float(residual_floor_margin_db)
    sustain = build_live_sustain(params)  # REAL core.engines._dtd.BargeSustain

    def _vad_ok(fr: Frame) -> bool:
        return vad_speech(fr) if callable(vad_speech) else bool(vad_speech)

    result = RunResult()
    result._burst_start_t = frames[0].t_sec if frames else None
    latch = False
    # First frame begins a reply run regardless of its explicit flag (a fresh
    # engine starts with zeroed floors); subsequent reply_start frames re-arm.
    first = True

    for fr in frames:
        if fr.reply_start or first:
            first = False
            sustain.reset()
            eng._playback_floor_rms = 0.0       # re-bootstrap the per-reply floors
            eng._raw_playback_floor_rms = 0.0
            eng._barge_in_fired_this_run = False
            eng._dtd.new_run()  # run boundary, as the engine does (charts persist by default)
            latch = False

        samples = make_block(fr.resid)   # post-AEC residual block
        mic_raw = make_block(fr.raw)     # RAW pre-AEC block
        eng._fake_coherence.last_incoherent_fraction = fr.incoh
        eng._fake_vad.set_speech(_vad_ok(fr))

        # Mirror the capture loop: learn the floor BEFORE the barge check.
        eng._update_playback_floor(fr.resid)     # REAL floor update (rms of const = resid)
        eng._update_raw_playback_floor(fr.raw)

        # REAL gate: latch + VAD + _looks_like_user (now residual-floor gated).
        eligible = eng._barge_in_fire_eligible(samples, mic_raw) and not latch
        if eng._dtd.last_decided:
            result.dtd_fire_count += 1
        if sustain.update(eligible):             # REAL BargeSustain.update
            sustain.reset()
            latch = True
            eng._barge_in_fired_this_run = True
            result.fires.append((fr.idx, fr.t_sec))

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
