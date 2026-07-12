"""Replay driver for the recorded owner-voice harness -- a SHARED, importable
test seam, NOT a test module.

It is named ``*_driver`` (not ``*_test``/``test_*``) so pytest's default
collection skips it: it exposes helpers, not test functions. The companion test
file (``tests/replay_recorded_voice_test.py``) imports this module and stays
pure assertions; ALL pipeline-wiring glue lives here.

Two paths, both feeding the OWNER's real extracted speech through the REAL
pipeline (no reimplementation of ASR/endpoint/the brain):

* ``run_turns`` -- the DEFAULT, CI-runnable path. A :class:`FileReplayEngine`
  (real sherpa recognizer + TTS, no sound card) + :class:`EchoLLM`
  (deterministic) driven by the FULL headless runtime
  (:func:`core.app.build_runtime`). No ``sounddevice`` patching, hardware-free,
  turn-at-a-time. This is where the per-utterance / multi-turn tests run.

* ``run_barge`` -- the fake-stream barge-OVERLAP path. FileReplay is single-threaded
  and cannot model concurrent talk-over, so this is the ONLY place the
  ``InjectingInputStream`` machinery (from ``tools.live_session.driver``) is
  needed: it patches both sounddevice streams and queries BEFORE the engine
  starts, pushes a base clip, proves true first audio plus a floor-only no-cut
  control, then consumes the barge clip WHILE the assistant is speaking. Success
  requires an ordered same-token stop/FIFO receipt. It self-skips only when the
  optional sounddevice dependency is absent; no hardware is opened.

Privacy / CI-safety: every entry point self-skips when its prerequisites are
absent -- sherpa models unconfigured, the local clip dir/file missing, or (for
barge) no sounddevice. The owner's voice clips live only under the gitignored
``logs/fixture_audio/`` and never enter git.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
import time
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Optional

import pytest

# Repo root: this file is tests/replay_voice_driver.py -> parent.parent.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_MANIFEST_PATH = _REPO_ROOT / "tests" / "fixtures" / "recorded_voice_manifest.json"
# Fallback clip dir if the manifest can't be read; the manifest's ``clip_dir``
# wins when present.
_DEFAULT_CLIP_DIR = "logs/fixture_audio"
_REQUIRE_RECORDED_ENV = "SPEAKER_REQUIRE_RECORDED"
_REFERENCE_CLIP_IDS = {f"utterance-{idx:02d}" for idx in range(6)}
_REFERENCE_BARGE_IDS = {"barge-00", "barge-01"}
_REFERENCE_CASES = {
    ("owner-overlap-00", "utterance-01", "barge-00"),
    ("owner-overlap-01", "utterance-05", "barge-01"),
}


def require_recorded_prerequisite(condition: bool, reason: str) -> None:
    """Skip optional corpus prerequisites, or fail the explicit landing gate."""
    if condition:
        return
    if os.environ.get(_REQUIRE_RECORDED_ENV) == "1":
        pytest.fail(
            f"required recorded gate prerequisite unavailable: {reason}",
            pytrace=False,
        )
    pytest.skip(reason)


def _optional_module_or_skip(name: str):
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        require_recorded_prerequisite(False, f"cannot import {name}: {exc}")
        raise AssertionError("unreachable") from exc


# --------------------------------------------------------------------------- #
# Manifest                                                                      #
# --------------------------------------------------------------------------- #
def load_manifest() -> dict:
    """Read the committed coordinate manifest (text/JSON only -- no audio).

    The single source of truth for clip ids + run/start/end/expected_text. All
    consumers read coordinates from here; none hard-code timestamps."""
    with open(_MANIFEST_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def validate_manifest(manifest: dict) -> dict[str, dict]:
    """Validate private-waveform provenance and explicit same-run pairings."""
    clips = manifest.get("clips", [])
    barges = manifest.get("barge", [])
    cases = manifest.get("barge_cases", [])
    if not isinstance(clips, list) or not isinstance(barges, list):
        raise ValueError("recorded manifest clips/barge must be lists")
    if not isinstance(cases, list) or not cases:
        raise ValueError("recorded manifest requires explicit barge_cases")

    lookup: dict[str, dict] = {}
    hashes: set[str] = set()
    clip_ids: set[str] = set()
    barge_ids: set[str] = set()
    for kind, entries in (("clip", clips), ("barge", barges)):
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f"recorded manifest {kind} entry must be an object")
            clip_id = str(entry.get("id") or "")
            if not clip_id or clip_id in lookup:
                raise ValueError(f"recorded manifest duplicate/blank id: {clip_id!r}")
            digest = str(entry.get("sha256") or "")
            if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError(f"recorded manifest {clip_id!r} needs lowercase SHA-256")
            if digest in hashes:
                raise ValueError(f"recorded manifest duplicate SHA-256 for {clip_id!r}")
            if not isinstance(entry.get("run"), str) or not entry["run"].strip():
                raise ValueError(f"recorded manifest {clip_id!r} has no run")
            if kind == "clip" and (
                not isinstance(entry.get("expected_text"), str)
                or not entry["expected_text"].strip()
            ):
                raise ValueError(
                    f"recorded manifest {clip_id!r} needs expected_text"
                )
            start = entry.get("start_sec")
            end = entry.get("end_sec")
            if (
                isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, Real)
                or not isinstance(end, Real)
                or not math.isfinite(float(start))
                or not math.isfinite(float(end))
                or float(start) < 0.0
                or float(end) <= float(start)
            ):
                raise ValueError(
                    f"recorded manifest {clip_id!r} needs a finite positive window"
                )
            if "speech_sec" in entry:
                speech_sec = entry["speech_sec"]
                window_sec = float(end) - float(start)
                if (
                    isinstance(speech_sec, bool)
                    or not isinstance(speech_sec, Real)
                    or not math.isfinite(float(speech_sec))
                    or float(speech_sec) <= 0.0
                    or float(speech_sec) > window_sec + 1e-6
                ):
                    raise ValueError(
                        f"recorded manifest {clip_id!r} has invalid speech_sec"
                    )
            hashes.add(digest)
            lookup[clip_id] = entry
            (clip_ids if kind == "clip" else barge_ids).add(clip_id)

    if clip_ids != _REFERENCE_CLIP_IDS or barge_ids != _REFERENCE_BARGE_IDS:
        raise ValueError("recorded manifest must contain the exact reference corpus")

    case_ids: set[str] = set()
    promoted_cases: set[tuple[str, str, str]] = set()
    case_pairs: set[tuple[str, str]] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("recorded manifest barge case must be an object")
        case_id = str(case.get("id") or "")
        if not case_id or case_id in case_ids:
            raise ValueError(f"recorded manifest duplicate/blank case id: {case_id!r}")
        case_ids.add(case_id)
        base_id = str(case.get("base_id") or "")
        barge_id = str(case.get("barge_id") or "")
        if base_id not in clip_ids or barge_id not in barge_ids:
            raise ValueError(f"recorded manifest case {case_id!r} has invalid pairing")
        run = case.get("run")
        if lookup[base_id].get("run") != run or lookup[barge_id].get("run") != run:
            raise ValueError(f"recorded manifest case {case_id!r} is not same-run")
        pair = (base_id, barge_id)
        if pair in case_pairs:
            raise ValueError(f"recorded manifest case {case_id!r} duplicates a pairing")
        case_pairs.add(pair)
        promoted_cases.add((case_id, base_id, barge_id))
    if promoted_cases != _REFERENCE_CASES:
        raise ValueError("recorded manifest must contain the exact promoted cases")
    return lookup


def clip_dir() -> Path:
    """The local (gitignored) directory the extraction tool writes clips into.

    Reads ``clip_dir`` from the manifest when available; falls back to the
    documented default otherwise. Returned absolute so callers are cwd-agnostic."""
    rel = _DEFAULT_CLIP_DIR
    try:
        rel = str(load_manifest().get("clip_dir") or _DEFAULT_CLIP_DIR)
    except (OSError, ValueError):
        pass
    p = Path(rel)
    return p if p.is_absolute() else _REPO_ROOT / p


# --------------------------------------------------------------------------- #
# Skip idioms (privacy + CI-safe)                                              #
# --------------------------------------------------------------------------- #
def sherpa_config_or_skip():
    """Build the machine-local :class:`SherpaConfig` or ``pytest.skip``.

    The EXACT idiom from ``tests/test_replay_recorded.py``: importorskip
    ``sherpa_onnx``, then ``load_config`` -> ``apply_device_profile`` ->
    ``SherpaConfig.from_dict``, and skip if the ASR encoder path is unset or
    missing on disk. The caller MUST set ``SPEAKER_NO_LOCAL_CONFIG=0`` first
    (via ``monkeypatch.setenv``) so the machine-local model paths are read --
    the hermetic-test default (conftest.py) disables that overlay."""
    _optional_module_or_skip("sherpa_onnx")
    from core.config import apply_device_profile, load_config
    from core.engines.sherpa import SherpaConfig

    cfg = load_config()
    cfg = apply_device_profile(cfg, cfg.get("device", "desktop"))
    scfg = SherpaConfig.from_dict(cfg.get("sherpa", {}))
    if not getattr(scfg, "asr_encoder", "") or not os.path.exists(scfg.asr_encoder):
        require_recorded_prerequisite(
            False,
            "sherpa ASR models not configured (set paths in config.local.json)",
        )
    return scfg


def _runtime_config():
    """The full config (device-profile-applied) the runtime is built from.

    Same load path as ``sherpa_config_or_skip`` but returns the whole dict so
    ``build_runtime`` sees the real ``llm``/``commands``/``memory`` blocks. The
    sherpa block inside matches the ``SherpaConfig`` the engine is given."""
    from core.config import apply_device_profile, load_config

    cfg = load_config()
    return apply_device_profile(cfg, cfg.get("device", "desktop"))


def load_clip(clip_id: str):
    """Return ``(samples, sample_rate)`` for an extracted clip, or skip.

    Reads ``<clip_dir>/<id>.wav`` via the existing
    ``core.engines.file_replay.load_waveform`` (16 kHz mono float32). Skips
    cleanly when the local clip dir or the specific file is absent -- so a fresh
    clone with models but no extracted clips skips rather than errors, and the
    owner's voice audio never needs to be present in CI."""
    from core.engines.file_replay import load_waveform

    cdir = clip_dir()
    if not cdir.exists():
        require_recorded_prerequisite(
            False,
            f"clip dir {cdir} absent -- run `python -m tools.extract_voice_clips` "
            "locally to extract the owner-voice clips (gitignored).",
        )
    path = cdir / f"{clip_id}.wav"
    if not path.exists():
        require_recorded_prerequisite(
            False,
            f"clip {path} absent (not extracted on this machine)",
        )
    metadata = validate_manifest(load_manifest()).get(clip_id)
    if metadata is None:
        raise ValueError(f"recorded manifest has no entry for {clip_id!r}")
    expected_hash = metadata["sha256"]
    with open(path, "rb") as fh:
        actual_hash = hashlib.file_digest(fh, "sha256").hexdigest()
    if actual_hash != expected_hash:
        raise AssertionError(
            f"clip {clip_id!r} SHA-256 mismatch: re-extract from the "
            "manifest-pinned source window"
        )
    return load_waveform(str(path))


# --------------------------------------------------------------------------- #
# Result type                                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class TurnResult:
    """One turn's captured outcome through the real pipeline."""

    asr_final: str
    response: str
    first_audio_latency: Optional[float] = None
    barge_in_latency: Optional[float] = None
    asr_final_stamped: bool = False
    # Barge-only: True when the assistant's TTS was actually cut (stop_speaking
    # fired after injection AND BARGE_IN_STOP attested a live FIFO cut). The test
    # asserts on this rather than reaching into the engine.
    barged: bool = False


@dataclass(frozen=True)
class BargeResult:
    """Causally bound evidence for one recorded-owner playback cut."""

    asr_final: str
    response: str
    target_turn_token: Optional[int]
    metric_turn_token: Optional[int]
    base_fully_consumed_at: Optional[float]
    tts_first_audio_at: Optional[float]
    barge_first_consumed_at: Optional[float]
    barge_fully_consumed_at: Optional[float]
    barge_in_at: Optional[float]
    stop_call_at: Optional[float]
    barge_in_stop_at: Optional[float]
    floor_control_clean: bool
    speaking_before_barge: bool

    @property
    def barge_in_latency(self) -> Optional[float]:
        if self.barge_in_at is None or self.barge_in_stop_at is None:
            return None
        return self.barge_in_stop_at - self.barge_in_at

    @property
    def barged(self) -> bool:
        return causal_barge_cut(self)

    @property
    def owner_to_stop_latency(self) -> Optional[float]:
        if self.barge_first_consumed_at is None or self.barge_in_stop_at is None:
            return None
        return self.barge_in_stop_at - self.barge_first_consumed_at


def causal_barge_cut(result: BargeResult) -> bool:
    """Return whether one exact metrics turn proves the injected causal chain."""
    if (
        result.target_turn_token is None
        or result.metric_turn_token != result.target_turn_token
        or not result.floor_control_clean
        or not result.speaking_before_barge
    ):
        return False
    values = (
        result.base_fully_consumed_at,
        result.tts_first_audio_at,
        result.barge_first_consumed_at,
        result.barge_fully_consumed_at,
        result.barge_in_at,
        result.stop_call_at,
        result.barge_in_stop_at,
    )
    if any(
        value is None or not math.isfinite(float(value))
        for value in values
    ):
        return False
    (
        base_drained,
        first_audio,
        barge_started,
        barge_drained,
        barge_in,
        stop_call,
        fifo_stop,
    ) = (float(value) for value in values)
    return bool(
        base_drained <= first_audio <= barge_started
        and barge_started <= barge_drained
        and barge_started <= barge_in <= stop_call <= fifo_stop
        and fifo_stop <= barge_drained + RECORDED_BARGE_DRAIN_GRACE_SEC
        and fifo_stop - barge_started <= RECORDED_OWNER_TO_STOP_MAX_SEC
    )


# --------------------------------------------------------------------------- #
# Turn-taking path (DEFAULT, hardware-free, CI-runnable)                        #
# --------------------------------------------------------------------------- #
def _build_headless_runtime(scfg, engine, llm, *, config=None):
    """Assemble the FULL headless runtime around an already-built engine.

    Drives the REAL brain via ``core.app.build_runtime`` -- the same assembly
    the CLI uses -- with ``fast_llm=None`` and the configured router. Not a
    reimplementation: every continuation / capability / endpoint wire is the
    production one."""
    from always_on_agent.events import Mode
    from core.app import build_runtime
    from core.llm import EchoLLM
    from core.routing import build_router

    config = config or _runtime_config()
    return build_runtime(
        config,
        engine=engine,
        llm=llm or EchoLLM(),
        fast_llm=None,
        router=build_router(config),
        start_mode=Mode.ASSISTANT,
    )


def run_turns(scfg, clip_ids, *, llm=None) -> list[TurnResult]:
    """Replay an ORDERED list of clips through one headless runtime, turn by turn.

    Builds a :class:`FileReplayEngine` (real recognizer + TTS, no sound card) +
    an :class:`EchoLLM` (deterministic) and drives the full real brain. Per clip
    it closes the prior metrics turn, replays the samples (the engine fires
    ``on_final`` -> the brain answers -> the engine ``speak``s), waits for idle,
    and snapshots ``engine.last_final`` + ``engine.spoken[-1]`` + the latest
    metrics record. Returns one :class:`TurnResult` per clip id, in order.

    This is the CI/default path: no ``sounddevice`` patching, no hardware,
    deterministic. The clips are skipped via :func:`load_clip` if not extracted.
    """
    from core.engine import EngineCallbacks
    from core.engines.file_replay import FileReplayEngine
    from core.llm import EchoLLM
    from core.metrics import ASR_FINAL

    # Resolve all clips up front (so a missing clip skips the test before we
    # spin up models).
    manifest = load_manifest()
    metadata = validate_manifest(manifest)
    loaded = [(cid, load_clip(cid), metadata.get(cid, {})) for cid in clip_ids]

    engine = FileReplayEngine(scfg)
    runtime = _build_headless_runtime(scfg, engine, llm or EchoLLM())
    runtime.start(run_bus=True)
    results: list[TurnResult] = []
    try:
        for _cid, (samples, sample_rate), clip_meta in loaded:
            spoken_before = len(engine.spoken)
            finals_before = len(engine.finals)
            runtime.metrics.close_turn()
            engine.replay_samples(
                samples,
                sample_rate,
                speech_sec=clip_meta.get("speech_sec"),
            )
            runtime.wait_idle(timeout=30.0)

            final_count = len(engine.finals) - finals_before
            if final_count != 1:
                raise RuntimeError(
                    f"recorded clip {_cid!r} emitted {final_count} finals; expected one"
                )

            recs = runtime.metrics.records()
            record = recs[-1] if recs else None
            response = engine.spoken[-1] if len(engine.spoken) > spoken_before else ""
            results.append(
                TurnResult(
                    asr_final=engine.last_final,
                    response=response,
                    first_audio_latency=(
                        record.first_audio_latency if record is not None else None
                    ),
                    barge_in_latency=(
                        record.barge_in_latency if record is not None else None
                    ),
                    asr_final_stamped=bool(record is not None and ASR_FINAL in record.stamps),
                )
            )
    finally:
        runtime.stop()
    return results


# --------------------------------------------------------------------------- #
# Barge-overlap path (needs the inject machinery + sounddevice)                 #
# --------------------------------------------------------------------------- #
def _import_inject_machinery_or_skip():
    """Import the live-session inject seam + sounddevice, or skip.

    The barge-overlap path is the only one that needs a concurrent capture loop
    (FileReplay is turn-at-a-time). It rides on ``tools.live_session.driver``'s
    ``InjectingInputStream``/``_NullOutputStream`` + ``make_recording_engine``.
    Missing sounddevice is an optional-host skip. A repository import failure is
    a harness defect and must fail the gate rather than being hidden as a skip."""
    sd = _optional_module_or_skip("sounddevice")
    from tools.live_session.driver import (
        InjectingInputStream,
        _NullOutputStream,
        apply_inject_profile,
        make_recording_engine,
    )
    return (
        sd,
        InjectingInputStream,
        _NullOutputStream,
        apply_inject_profile,
        make_recording_engine,
    )


# Max wait for a barge to register a stop after it's injected (VAD accumulation
# lags the inject by ~1-1.5s). Mirrors live_session.driver._BARGE_STOP_TIMEOUT.
_BARGE_STOP_TIMEOUT = 4.0
# Max wait for the assistant to BEGIN speaking the base-clip answer before we
# inject the barge over it.
_SPEAKING_START_TIMEOUT = 30.0
# One capture block plus bounded scheduling/metrics handoff after the final
# injected sample. A cut later than this is not attributed to this waveform.
RECORDED_BARGE_DRAIN_GRACE_SEC = 0.25
# Owner first consumed sample -> actual FIFO cut. This includes the manifest's
# 100 ms extraction pre-pad, the recorded gate's production 200 ms sustain
# requirement, and capture/control scheduling while remaining a usable spoken-
# interrupt ceiling.
RECORDED_OWNER_TO_STOP_MAX_SEC = 1.0
RECORDED_BARGE_MIN_SPEECH_SEC = 0.2
_LONG_BARGE_REPLY = (
    "The deliberate validation answer stays long enough for a real overlap. "
    "It describes a quiet town, a patient clockmaker, several winding streets, "
    "and the small choices people make while an ordinary afternoon passes. "
    "Every sentence is intentionally concrete and unhurried so the playback "
    "FIFO remains active through the negative control window and the later "
    "recorded interruption can be attributed to one exact assistant turn."
)


def recorded_floor_control_deadline(
    *,
    first_audio_observed_at: float,
    playback_onset_at: float,
    onset_grace_sec: float,
    sustain_window_sec: float,
    block_sec: float,
) -> float:
    """End of a complete armed floor-only observation after true first audio."""
    values = (
        first_audio_observed_at,
        playback_onset_at,
        onset_grace_sec,
        sustain_window_sec,
        block_sec,
    )
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("recorded floor-control timing must be finite")
    if min(onset_grace_sec, sustain_window_sec, block_sec) < 0.0:
        raise ValueError("recorded floor-control durations must be nonnegative")
    armed_at = max(
        float(first_audio_observed_at),
        float(playback_onset_at) + float(onset_grace_sec),
    )
    return armed_at + float(sustain_window_sec) + float(block_sec)


def run_barge(scfg, base_clip_id: str, barge_clip_id: str) -> BargeResult:
    """Drive a real barge-OVERLAP: the owner talks over the assistant's TTS.

    Patches ``sd.InputStream``/``sd.OutputStream`` BEFORE the engine starts (so
    the real mic/speaker are never opened), pushes the base clip to elicit an
    fixed long answer, waits for that turn's true ``TTS_FIRST_AUDIO``, verifies a
    floor-only no-cut control, then consumes the barge clip WHILE it speaks. The
    real capture loop drives the runtime stop. A cut requires causally ordered
    sample receipt, BARGE_IN, stop call, and BARGE_IN_STOP timestamps on the same
    metrics token. ``response`` carries the interrupted assistant text.

    Self-skips if the inject machinery / sounddevice is unavailable, or if the
    clips aren't extracted locally.
    """
    (
        sd,
        InjectingInputStream,
        _NullOutputStream,
        apply_inject_profile,
        make_recording_engine,
    ) = _import_inject_machinery_or_skip()

    base_samples, base_sr = load_clip(base_clip_id)
    barge_samples, barge_sr = load_clip(barge_clip_id)

    from core.llm import EchoLLM
    from core.metrics import ASR_FINAL, BARGE_IN, BARGE_IN_STOP, TTS_FIRST_AUDIO

    config = _runtime_config()
    apply_inject_profile(config)
    config.setdefault("sherpa", {}).update(
        barge_in_enabled=True,
        # These owner clips are sustained. Keep the synthetic 418 ms command's
        # one-block echo-free profile, but make the promoted recorded gate run
        # the production two-block temporal policy through the full pipeline.
        barge_in_min_speech_sec=RECORDED_BARGE_MIN_SPEECH_SEC,
    )
    # A SherpaOnnxEngine subclass that records spoken()/stop_speaking() so we can
    # poll stopped_after() -- the FileReplayEngine can't model concurrent
    # talk-over. Reuses live_session.driver.make_recording_engine verbatim.
    engine, _cfg = make_recording_engine(config)
    runtime = _build_headless_runtime(
        scfg,
        engine,
        EchoLLM(reply=_LONG_BARGE_REPLY),
        config=config,
    )

    # Patch the device seams BEFORE runtime.start() (which calls engine.start()),
    # exactly as tools/live_session/driver.py:433-441 does.
    holder: dict = {"input_calls": 0, "output_calls": 0}

    def _input_factory(*args, samplerate=16000, **kwargs):
        holder["input_calls"] += 1
        stream = InjectingInputStream(int(samplerate) or 16000)
        holder["stream"] = stream  # last opened wins == the one that sticks
        return stream

    def _output_factory(*args, **kwargs):
        holder["output_calls"] += 1
        return _NullOutputStream(*args, **kwargs)

    def _query_devices(_device=None, kind=None):
        return {
            "name": f"inject-{kind or 'device'}",
            "default_samplerate": 16000 if kind == "input" else 48000,
        }

    def _check_settings(*args, **kwargs):
        del args, kwargs
        return None

    orig_input = sd.InputStream
    orig_output = sd.OutputStream
    orig_query = sd.query_devices
    orig_check_input = sd.check_input_settings
    orig_check_output = sd.check_output_settings
    sd.InputStream = _input_factory
    sd.OutputStream = _output_factory
    sd.query_devices = _query_devices
    sd.check_input_settings = _check_settings
    sd.check_output_settings = _check_settings
    try:
        runtime.start(run_bus=True)
        # Wait out any warm-up so the first turn runs on warm models.
        ready = getattr(runtime, "warm_ready", None)
        if ready is not None:
            ready.wait(timeout=120.0)

        inject_stream = holder.get("stream")
        if inject_stream is None:
            raise RuntimeError("inject mode: the engine never opened an input stream")
        if holder["input_calls"] < 1:
            raise RuntimeError("inject mode: fake input stream was not opened")
        effective = engine.config
        from core.engines._dtd import BargeSustain

        recorded_sustain = BargeSustain(
            window_sec=effective.barge_in_sustain_window_sec,
            block_sec=effective.block_sec,
            min_voiced_sec=effective.barge_in_min_speech_sec,
        )
        if not (
            effective.aec_enabled is False
            and effective.capture_voice_comm is False
            and effective.barge_word_cut_enabled is False
            and effective.coherence_warmup_frames == 0
            and effective.coherence_confirm_frames == 1
            and effective.coherence_max_delay_ms == 0.0
            and effective.barge_in_min_speech_sec
            == RECORDED_BARGE_MIN_SPEECH_SEC
            and effective.block_sec == 0.1
            and recorded_sustain.need_frames == 2
            and recorded_sustain.window_frames >= 2
            and effective.barge_in_enabled is True
            and engine._aec is None
            and engine._os_echo_route_verified is False
        ):
            raise RuntimeError("inject mode: effective no-device authority profile drifted")

        sr = inject_stream._sr

        def _to_inject_rate(samples, src_sr):
            from tools.live_session.synthetic_user import _resample

            if int(src_sr) == int(sr):
                import numpy as np

                return np.asarray(samples, dtype="float32").reshape(-1)
            return _resample(samples, int(src_sr), int(sr))

        # 1) Push the base clip; the real ASR transcribes it and the brain
        #    answers, so the assistant starts speaking.
        spoken_base = engine.spoken_count()
        stop_base = engine.stop_count()
        baseline_metric_tokens = {
            record.turn_token for record in runtime.metrics.records()
        }
        transcript_base = len(
            list(getattr(runtime.supervisor.state, "transcript_log", []) or [])
        )
        base_receipt = inject_stream.push(_to_inject_rate(base_samples, base_sr))
        if not base_receipt.wait_drained(timeout=15.0):
            raise RuntimeError("recorded barge setup failed: base clip never drained")

        # 2) Bind the answer to one metrics turn and wait for true first audio.
        deadline = time.time() + _SPEAKING_START_TIMEOUT
        target_token = None
        target_record = None
        while time.time() < deadline:
            candidate = runtime.metrics.current_turn_token()
            records = runtime.metrics.records()
            record = next(
                (r for r in records if r.turn_token == candidate),
                None,
            )
            if (
                candidate is not None
                and record is not None
                and ASR_FINAL in record.stamps
                and TTS_FIRST_AUDIO in record.stamps
                and engine.is_speaking
            ):
                target_token = candidate
                target_record = record
                break
            time.sleep(0.02)
        if target_record is None:
            raise RuntimeError(
                "recorded barge setup failed: base turn never reached true first audio"
            )
        if holder["output_calls"] < 1:
            raise RuntimeError("inject mode: fake output stream was not opened")
        first_audio_observed_at = time.monotonic()
        transcript = list(
            getattr(runtime.supervisor.state, "transcript_log", []) or []
        )
        base_asr_final = transcript[-1] if transcript else ""

        # 3) Negative control: wait past the actual synth-onset grace, then keep
        # the deterministic floor running for a complete sustain window plus one
        # capture block. A green control therefore observes the armed detector,
        # not its intentional startup suppression.
        control_until = recorded_floor_control_deadline(
            first_audio_observed_at=first_audio_observed_at,
            playback_onset_at=float(engine._playback_onset_at),
            onset_grace_sec=float(effective.barge_in_playback_onset_grace_sec),
            sustain_window_sec=float(effective.barge_in_sustain_window_sec),
            block_sec=float(effective.block_sec),
        )
        control_duration = (
            float(effective.barge_in_sustain_window_sec)
            + float(effective.block_sec)
        )
        control_started_at = control_until - control_duration
        time.sleep(max(0.0, control_started_at - time.monotonic()))
        floor_samples_before = inject_stream.floor_samples_delivered
        time.sleep(max(0.0, control_until - time.monotonic()))
        required_floor_samples = max(1, round(control_duration * sr))
        floor_wait_deadline = time.monotonic() + float(effective.block_sec) + 0.5
        while (
            inject_stream.floor_samples_delivered - floor_samples_before
            < required_floor_samples
            and time.monotonic() < floor_wait_deadline
        ):
            time.sleep(0.01)
        floor_samples_observed = (
            inject_stream.floor_samples_delivered - floor_samples_before
        )
        target_record = next(
            (
                r
                for r in runtime.metrics.records()
                if r.turn_token == target_token
            ),
            None,
        )
        base_final_records = [
            record
            for record in runtime.metrics.records()
            if record.turn_token not in baseline_metric_tokens
            and ASR_FINAL in record.stamps
        ]
        transcript_now = list(
            getattr(runtime.supervisor.state, "transcript_log", []) or []
        )
        floor_control_clean = bool(
            target_record is not None
            and len(base_final_records) == 1
            and base_final_records[0].turn_token == target_token
            and len(transcript_now) == transcript_base + 1
            and BARGE_IN not in target_record.stamps
            and BARGE_IN_STOP not in target_record.stamps
            and engine.stop_count() == stop_base
            and engine.is_speaking
            and floor_samples_observed >= required_floor_samples
        )
        if not floor_control_clean:
            raise RuntimeError(
                "recorded barge negative control failed: floor/self-audio cut target"
            )

        # 4) Push the barge clip into the SAME buffer WHILE it speaks -- real
        #    overlap. The capture loop reads it concurrently and the endpoint
        #    fires, driving stop_speaking().
        speaking_before_barge = engine.is_speaking
        barge_receipt = inject_stream.push(_to_inject_rate(barge_samples, barge_sr))
        if not barge_receipt.wait_started(timeout=5.0):
            raise RuntimeError("recorded barge clip was enqueued but never consumed")
        barge_started = barge_receipt.first_consumed_at

        # 5) Poll the exact target record and post-consumption stop calls.
        stop_deadline = time.time() + _BARGE_STOP_TIMEOUT
        while time.time() < stop_deadline:
            target_record = next(
                (
                    r
                    for r in runtime.metrics.records()
                    if r.turn_token == target_token
                ),
                None,
            )
            stops = [
                t
                for t in engine.stops_since(stop_base)
                if barge_started is not None and t >= barge_started
            ]
            if (
                target_record is not None
                and BARGE_IN_STOP in target_record.stamps
                and stops
            ):
                break
            time.sleep(0.05)
        if not barge_receipt.wait_drained(timeout=15.0):
            raise RuntimeError("recorded barge clip never fully drained")

        runtime.wait_idle(timeout=30.0)
        runtime.metrics.close_turn()

        target_record = next(
            (
                r
                for r in runtime.metrics.records()
                if r.turn_token == target_token
            ),
            None,
        )
        stamps = target_record.stamps if target_record is not None else {}
        stops = [
            t
            for t in engine.stops_since(stop_base)
            if barge_started is not None and t >= barge_started
        ]

        # SherpaOnnxEngine (the recording subclass) exposes spoken text via
        # spoken_since() (list[(text, ts)]), not a .spoken list, and has no
        # .last_final attr -- the recognized finals live in the supervisor's
        # transcript log. Pull both from the real surfaces.
        spoken = engine.spoken_since(spoken_base)
        response = " ".join(text for text, _at in spoken)
        return BargeResult(
            asr_final=base_asr_final,
            response=response,
            target_turn_token=target_token,
            metric_turn_token=(
                target_record.turn_token if target_record is not None else None
            ),
            base_fully_consumed_at=base_receipt.fully_consumed_at,
            tts_first_audio_at=stamps.get(TTS_FIRST_AUDIO),
            barge_first_consumed_at=barge_receipt.first_consumed_at,
            barge_fully_consumed_at=barge_receipt.fully_consumed_at,
            barge_in_at=stamps.get(BARGE_IN),
            stop_call_at=(stops[0] if stops else None),
            barge_in_stop_at=stamps.get(BARGE_IN_STOP),
            floor_control_clean=floor_control_clean,
            speaking_before_barge=speaking_before_barge,
        )
    finally:
        try:
            runtime.stop()
            alive = [
                name
                for name, worker in (
                    ("capture", engine._capture_thread),
                    ("playback", engine._play_thread),
                    ("final", engine._final_thread),
                    ("receipt", engine._receipt_thread),
                )
                if worker is not None and worker.is_alive()
            ]
            if alive:
                raise RuntimeError(
                    "recorded barge workers survived teardown: " + ", ".join(alive)
                )
        finally:
            sd.InputStream = orig_input
            sd.OutputStream = orig_output
            sd.query_devices = orig_query
            sd.check_input_settings = orig_check_input
            sd.check_output_settings = orig_check_output


__all__ = [
    "TurnResult",
    "BargeResult",
    "RECORDED_BARGE_DRAIN_GRACE_SEC",
    "RECORDED_BARGE_MIN_SPEECH_SEC",
    "RECORDED_OWNER_TO_STOP_MAX_SEC",
    "causal_barge_cut",
    "load_manifest",
    "validate_manifest",
    "clip_dir",
    "load_clip",
    "recorded_floor_control_deadline",
    "require_recorded_prerequisite",
    "sherpa_config_or_skip",
    "run_turns",
    "run_barge",
]
