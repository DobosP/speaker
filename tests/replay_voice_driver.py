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

* ``run_barge`` -- the barge-OVERLAP path. FileReplay is single-threaded and
  cannot model concurrent talk-over, so this is the ONLY place the
  ``InjectingInputStream`` machinery (from ``tools.live_session.driver``) is
  needed: it patches ``sd.InputStream``/``sd.OutputStream`` BEFORE the engine
  starts, pushes the base clip, then pushes the barge clip WHILE the assistant
  is speaking, and polls ``engine.stopped_after(t)`` to confirm the cut. It
  self-skips if ``tools.live_session`` or ``sounddevice`` can't be imported.

Privacy / CI-safety: every entry point self-skips when its prerequisites are
absent -- sherpa models unconfigured, the local clip dir/file missing, or (for
barge) no sounddevice. The owner's voice clips live only under the gitignored
``logs/fixture_audio/`` and never enter git.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest

# Repo root: this file is tests/replay_voice_driver.py -> parent.parent.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_MANIFEST_PATH = _REPO_ROOT / "tests" / "fixtures" / "recorded_voice_manifest.json"
# Fallback clip dir if the manifest can't be read; the manifest's ``clip_dir``
# wins when present.
_DEFAULT_CLIP_DIR = "logs/fixture_audio"


# --------------------------------------------------------------------------- #
# Manifest                                                                      #
# --------------------------------------------------------------------------- #
def load_manifest() -> dict:
    """Read the committed coordinate manifest (text/JSON only -- no audio).

    The single source of truth for clip ids + run/start/end/expected_text. All
    consumers read coordinates from here; none hard-code timestamps."""
    with open(_MANIFEST_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


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
    pytest.importorskip("sherpa_onnx")
    from core.config import apply_device_profile, load_config
    from core.engines.sherpa import SherpaConfig

    cfg = load_config()
    cfg = apply_device_profile(cfg, cfg.get("device", "desktop"))
    scfg = SherpaConfig.from_dict(cfg.get("sherpa", {}))
    if not getattr(scfg, "asr_encoder", "") or not os.path.exists(scfg.asr_encoder):
        pytest.skip("sherpa ASR models not configured (set paths in config.local.json)")
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
        pytest.skip(
            f"clip dir {cdir} absent -- run `python -m tools.extract_voice_clips` "
            "locally to extract the owner-voice clips (gitignored)."
        )
    path = cdir / f"{clip_id}.wav"
    if not path.exists():
        pytest.skip(f"clip {path} absent (not extracted on this machine)")
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
    # fired after the barge was injected). The test asserts on this rather than
    # reaching into the engine, so it stays pure assertions.
    barged: bool = False


# --------------------------------------------------------------------------- #
# Turn-taking path (DEFAULT, hardware-free, CI-runnable)                        #
# --------------------------------------------------------------------------- #
def _build_headless_runtime(scfg, engine, llm):
    """Assemble the FULL headless runtime around an already-built engine.

    Drives the REAL brain via ``core.app.build_runtime`` -- the same assembly
    the CLI uses -- with ``fast_llm=None`` and the configured router. Not a
    reimplementation: every continuation / capability / endpoint wire is the
    production one."""
    from always_on_agent.events import Mode
    from core.app import build_runtime
    from core.llm import EchoLLM
    from core.routing import build_router

    config = _runtime_config()
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
    loaded = [(cid, load_clip(cid)) for cid in clip_ids]

    engine = FileReplayEngine(scfg)
    runtime = _build_headless_runtime(scfg, engine, llm or EchoLLM())
    runtime.start(run_bus=True)
    results: list[TurnResult] = []
    try:
        for _cid, (samples, sample_rate) in loaded:
            spoken_before = len(engine.spoken)
            runtime.metrics.close_turn()
            engine.replay_samples(samples, sample_rate)
            runtime.wait_idle(timeout=30.0)

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
    Either import failing -> the test self-skips cleanly."""
    sd = pytest.importorskip("sounddevice")
    try:
        from tools.live_session.driver import (  # noqa: F401
            InjectingInputStream,
            _NullOutputStream,
            make_recording_engine,
        )
    except Exception as exc:  # noqa: BLE001 - any import failure -> skip
        pytest.skip(f"tools.live_session unavailable for the barge path: {exc!r}")
    return sd, InjectingInputStream, _NullOutputStream, make_recording_engine


# Max wait for a barge to register a stop after it's injected (VAD accumulation
# lags the inject by ~1-1.5s). Mirrors live_session.driver._BARGE_STOP_TIMEOUT.
_BARGE_STOP_TIMEOUT = 4.0
# Max wait for the assistant to BEGIN speaking the base-clip answer before we
# inject the barge over it.
_SPEAKING_START_TIMEOUT = 30.0


def run_barge(scfg, base_clip_id: str, barge_clip_id: str) -> TurnResult:
    """Drive a real barge-OVERLAP: the owner talks over the assistant's TTS.

    Patches ``sd.InputStream``/``sd.OutputStream`` BEFORE the engine starts (so
    the real mic/speaker are never opened), pushes the base clip to elicit an
    answer, waits until the assistant ``is_speaking``, then pushes the barge
    clip into the same buffer WHILE it speaks. The real capture loop reads the
    overlapped audio concurrently, the endpointer fires, and the runtime calls
    ``stop_speaking`` -- detected via ``engine.stopped_after(t)``. Returns a
    :class:`TurnResult` whose ``barge_in_latency`` is read from the metrics
    record; ``response`` carries the (possibly interrupted) assistant text.

    Self-skips if the inject machinery / sounddevice is unavailable, or if the
    clips aren't extracted locally.
    """
    sd, InjectingInputStream, _NullOutputStream, make_recording_engine = (
        _import_inject_machinery_or_skip()
    )

    base_samples, base_sr = load_clip(base_clip_id)
    barge_samples, barge_sr = load_clip(barge_clip_id)

    from core.llm import EchoLLM
    from core.metrics import BARGE_IN

    config = _runtime_config()
    # A SherpaOnnxEngine subclass that records spoken()/stop_speaking() so we can
    # poll stopped_after() -- the FileReplayEngine can't model concurrent
    # talk-over. Reuses live_session.driver.make_recording_engine verbatim.
    engine, _cfg = make_recording_engine(config)
    runtime = _build_headless_runtime(scfg, engine, EchoLLM())

    # Patch the device seams BEFORE runtime.start() (which calls engine.start()),
    # exactly as tools/live_session/driver.py:433-441 does.
    holder: dict = {}

    def _input_factory(*args, samplerate=16000, **kwargs):
        stream = InjectingInputStream(int(samplerate) or 16000)
        holder["stream"] = stream  # last opened wins == the one that sticks
        return stream

    orig_input = sd.InputStream
    orig_output = sd.OutputStream
    sd.InputStream = _input_factory
    sd.OutputStream = _NullOutputStream
    try:
        runtime.start(run_bus=True)
        # Wait out any warm-up so the first turn runs on warm models.
        ready = getattr(runtime, "warm_ready", None)
        if ready is not None:
            ready.wait(timeout=120.0)

        inject_stream = holder.get("stream")
        if inject_stream is None:
            raise RuntimeError("inject mode: the engine never opened an input stream")

        sr = inject_stream._sr

        def _to_inject_rate(samples, src_sr):
            from tools.live_session.synthetic_user import _resample

            if int(src_sr) == int(sr):
                import numpy as np

                return np.asarray(samples, dtype="float32").reshape(-1)
            return _resample(samples, int(src_sr), int(sr))

        # 1) Push the base clip; the real ASR transcribes it and the brain
        #    answers, so the assistant starts speaking.
        inject_stream.push(_to_inject_rate(base_samples, base_sr))

        # 2) Wait until the assistant actually begins speaking the answer.
        deadline = time.time() + _SPEAKING_START_TIMEOUT
        while time.time() < deadline and not engine.is_speaking:
            time.sleep(0.02)

        barge_t = time.perf_counter()
        # 3) Push the barge clip into the SAME buffer WHILE it speaks -- real
        #    overlap. The capture loop reads it concurrently and the endpoint
        #    fires, driving stop_speaking().
        inject_stream.push(_to_inject_rate(barge_samples, barge_sr))

        # 4) Poll stopped_after() to confirm the assistant was cut.
        stop_deadline = time.time() + _BARGE_STOP_TIMEOUT
        while time.time() < stop_deadline and not engine.stopped_after(barge_t):
            time.sleep(0.05)
        barged = bool(engine.stopped_after(barge_t))

        runtime.wait_idle(timeout=30.0)
        runtime.metrics.close_turn()

        # The barge metric is stamped on the turn that was speaking when the
        # barge fired; pick the most recent record that carries a BARGE_IN stamp.
        barge_latency = None
        for record in reversed(runtime.metrics.records()):
            if BARGE_IN in record.stamps:
                barge_latency = record.barge_in_latency
                break

        # SherpaOnnxEngine (the recording subclass) exposes spoken text via
        # spoken_since() (list[(text, ts)]), not a .spoken list, and has no
        # .last_final attr -- the recognized finals live in the supervisor's
        # transcript log. Pull both from the real surfaces.
        spoken = engine.spoken_since(0)
        response = spoken[-1][0] if spoken else ""
        transcript = list(getattr(runtime.supervisor.state, "transcript_log", []) or [])
        asr_final = transcript[-1] if transcript else ""
        return TurnResult(
            asr_final=asr_final,
            response=response,
            first_audio_latency=None,
            barge_in_latency=barge_latency,
            barged=barged,
        )
    finally:
        try:
            runtime.stop()
        finally:
            sd.InputStream = orig_input
            sd.OutputStream = orig_output


__all__ = [
    "TurnResult",
    "load_manifest",
    "clip_dir",
    "load_clip",
    "sherpa_config_or_skip",
    "run_turns",
    "run_barge",
]
