#!/usr/bin/env python3
"""Live audio self-interruption (TTS-echo) diagnostic -- PLAYS AUDIO OUT LOUD.

Drives the REAL ``sherpa`` engine (mic + speakers) to measure whether the
assistant's own TTS, played through the speakers, is re-captured by the mic
strongly enough to trip a barge-in (a *self-interruption*), and how the
unenrolled output-margin gate (``barge_in_output_margin_db``) responds.

This is a quiet echo-only diagnostic: it observes assistant playback while the
operator remains silent. It does not select or tune a configuration, demonstrate
a real human talk-over, or complete physical acceptance. The gate compares mic
RMS (post speaker-volume + room coupling) against ``_playback_level`` (the
pre-volume TTS *buffer* RMS), so the observed relationship includes the active
speaker route, output level, and room coupling.

Diagnostic-only examples (neither command supplies acceptance evidence):

    python -m tools.echo_probe --margin-db 0 --sentences 4
    python -m tools.echo_probe --margin-db 6 --gain 1.0

Prints a JSON summary so a run is interpretable even if nothing coupled:
``self_interruptions`` (barge-ins fired during the assistant's own speech),
the (mic_rms, playback_level, gate_passed) samples the gate evaluated, the
mic/playback ratio in dB, the peak playback level, and an additive
``aec_reference_delay`` block separating the configured seed from the final
accepted/current operating delay when shutdown proves the capture writer stopped.
Physical acceptance requires a bare-speaker session started with ``./live.sh``;
talk over playback on the bare speaker, then analyze that exact retained run with
``python -m tools.live_audio_ab logs/runs/run-<id>.txt``.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
import threading
import time
from collections.abc import Callable

SENTENCES = [
    "This is a live audio calibration test of the speaker assistant.",
    "I am checking whether my own voice is captured back by the microphone.",
    "The barge in gate should not interrupt me while I am still speaking.",
    "If you hear this whole message without it cutting off, suppression works.",
]

_STIMULUS_HASH_DOMAIN = b"speaker.echo-probe.spoken-text.v1\0"
_STIMULUS_ID_PREFIX = "echo-probe-spoken-text-v1:sha256:"


@dataclasses.dataclass(frozen=True, slots=True)
class _StimulusPlan:
    """Frozen sentence cycle and effective count for one probe invocation."""

    cycle: tuple[str, ...]
    count: int

    def __iter__(self):
        for index in range(self.count):
            yield self.cycle[index % len(self.cycle)]


def _build_stimulus_plan(requested_count: int) -> _StimulusPlan:
    if type(requested_count) is not int:
        raise TypeError("stimulus sentence count must be an exact int")
    cycle = tuple(SENTENCES)
    if not cycle:
        raise ValueError("echo probe requires at least one stimulus sentence")
    if any(type(text) is not str for text in cycle):
        raise TypeError("stimulus sentences must be exact strings")
    return _StimulusPlan(cycle=cycle, count=max(1, requested_count))


class _StimulusIdBuilder:
    """Incrementally bind the exact text of normally returned speak calls."""

    def __init__(self) -> None:
        self._digest = hashlib.sha256(_STIMULUS_HASH_DOMAIN)
        self._recorded_count = 0

    def record(self, text: str) -> None:
        if type(text) is not str:
            raise TypeError("stimulus text must be an exact str")
        encoded = text.encode("utf-8")
        self._digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        self._digest.update(encoded)
        self._recorded_count += 1

    def stimulus_id(self) -> str:
        if self._recorded_count == 0:
            raise ValueError("cannot identify an empty stimulus sequence")
        return _STIMULUS_ID_PREFIX + self._digest.hexdigest()


_PHYSICAL_ACCEPTANCE_GUIDANCE = (
    "Physical acceptance requires a bare-speaker session started with `./live.sh`; "
    "talk over playback on the bare speaker, then analyze that exact retained run with "
    "`python -m tools.live_audio_ab logs/runs/run-<id>.txt`."
)
_NO_COUPLING_GUIDANCE = (
    "Without measured echo coupling, this quiet echo-only diagnostic is "
    "inconclusive and does not select or tune a configuration."
)


def _rms(samples) -> float:
    import numpy as np

    a = np.asarray(samples, dtype="float32").reshape(-1)
    if a.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(a * a)))


def _samples_to_ms(samples: int, sample_rate_hz: int) -> float:
    return round(1000.0 * int(samples) / int(sample_rate_hz), 3)


def _capture_quiesced_after_stop(engine, *, stop_completed: bool) -> bool:
    """Whether bounded shutdown proved the capture-only calibrator is stable."""
    if not stop_completed:
        return False
    running = getattr(engine, "_running", None)
    retained = getattr(engine, "_capture_resource_hold", None)
    running_is_set = getattr(running, "is_set", None)
    retained_is_set = getattr(retained, "is_set", None)
    if not callable(running_is_set) or not callable(retained_is_set):
        return False
    try:
        return not bool(running_is_set()) and not bool(retained_is_set())
    except Exception:
        return False


@dataclasses.dataclass(frozen=True, slots=True)
class _ProbeLifecycleResult:
    """Closed outcome of one start/probe/stop ownership attempt."""

    start_error_type: str | None = None
    probe_error_type: str | None = None
    stop_error_type: str | None = None

    @property
    def stop_completed(self) -> bool:
        return self.stop_error_type is None

    @property
    def report_allowed(self) -> bool:
        return (
            self.start_error_type is None
            and self.probe_error_type is None
            and self.stop_error_type is None
        )


def _run_probe_lifecycle(
    start: Callable[[], None],
    probe: Callable[[], None],
    stop: Callable[[], None],
) -> _ProbeLifecycleResult:
    """Invoke one start and exactly one stop, preserving Ctrl-C precedence."""
    start_error_type = None
    probe_error_type = None
    stop_error_type = None
    primary_interrupt: KeyboardInterrupt | None = None
    stop_interrupt: KeyboardInterrupt | None = None

    try:
        try:
            start()
        except KeyboardInterrupt as exc:
            primary_interrupt = exc
        except BaseException as exc:  # SystemExit(0) must not become success.
            start_error_type = type(exc).__name__
        else:
            try:
                probe()
            except KeyboardInterrupt as exc:
                primary_interrupt = exc
            except BaseException as exc:  # SystemExit(0) must not become success.
                probe_error_type = type(exc).__name__
    finally:
        try:
            stop()
        except KeyboardInterrupt as exc:
            stop_interrupt = exc
        except BaseException as exc:  # SystemExit(0) must not become success.
            stop_error_type = type(exc).__name__

    if primary_interrupt is not None:
        raise primary_interrupt
    if stop_interrupt is not None:
        raise stop_interrupt
    return _ProbeLifecycleResult(
        start_error_type=start_error_type,
        probe_error_type=probe_error_type,
        stop_error_type=stop_error_type,
    )


def _emit_probe_result(payload: object, *, success: bool) -> int:
    """Write one strict terminal JSON object with exit/report coherence."""
    try:
        encoded = json.dumps(payload, indent=2, allow_nan=False)
    except KeyboardInterrupt:
        raise
    except BaseException:
        encoded = json.dumps({"error": "probe-result-serialization-failed"})
        success = False
    sys.stdout.write(encoded + "\n")
    return 0 if success else 1


def _aec_reference_delay_summary(engine, sherpa_cfg, *, stop_completed: bool) -> dict:
    """Build the additive seed-versus-runtime delay diagnostic.

    The capture thread is the calibrator's sole writer.  A bounded ``stop`` may
    retain that owner, so no calibrator or operating-delay field is read unless
    shutdown proved quiescence.  The legacy strategy field remains the configured
    seed; this block reports actual AEC construction and explicit acceptance.
    """
    sample_rate_hz = int(getattr(sherpa_cfg, "sample_rate"))
    configured_seed_ms = getattr(sherpa_cfg, "aec_ref_delay_ms")
    configured_seed_samples = int(sample_rate_hz * float(configured_seed_ms) / 1000.0)
    aec_enabled_configured = bool(getattr(sherpa_cfg, "aec_enabled", False))
    auto_delay_enabled = bool(getattr(sherpa_cfg, "aec_auto_delay", False))
    aec_active = getattr(engine, "_aec", None) is not None
    calibrator = getattr(engine, "_aec_delay_cal", None)
    auto_delay_active = bool(
        aec_active and auto_delay_enabled and calibrator is not None
    )

    out = {
        "measurement_state": None,
        "configured_seed_ms": configured_seed_ms,
        "configured_seed_samples": configured_seed_samples,
        "sample_rate_hz": sample_rate_hz,
        "aec_enabled_configured": aec_enabled_configured,
        "aec_active": aec_active,
        "auto_delay_enabled": auto_delay_enabled,
        "auto_delay_active": auto_delay_active,
        "runtime_snapshot_available": False,
        "runtime_estimate_accepted": False,
        "effective_seed_samples": None,
        "operating_delay_samples": None,
        "operating_delay_ms": None,
    }
    if not aec_active:
        out["measurement_state"] = (
            "aec_unavailable" if aec_enabled_configured else "aec_disabled"
        )
        return out

    if not _capture_quiesced_after_stop(engine, stop_completed=stop_completed):
        out["measurement_state"] = "snapshot_unavailable"
        out["runtime_estimate_accepted"] = None
        return out

    if not auto_delay_enabled:
        try:
            operating = int(getattr(engine, "_aec_ref_delay"))
            if operating < 0 or operating != configured_seed_samples:
                raise ValueError("inconsistent fixed delay")
        except Exception:
            out["measurement_state"] = "snapshot_unavailable"
            out["runtime_estimate_accepted"] = None
            return out
        out.update(
            {
                "measurement_state": "fixed_configured_delay",
                "runtime_snapshot_available": True,
                "effective_seed_samples": configured_seed_samples,
                "operating_delay_samples": operating,
                "operating_delay_ms": _samples_to_ms(operating, sample_rate_hz),
            }
        )
        return out

    if calibrator is None:
        out["measurement_state"] = "calibrator_unavailable"
        out["runtime_estimate_accepted"] = None
        return out

    try:
        snapshot = calibrator.snapshot()
        snapshot_rate = int(snapshot.sample_rate_hz)
        effective_seed = int(snapshot.seed_delay_samples)
        operating = int(snapshot.operating_delay_samples)
        engine_operating = int(getattr(engine, "_aec_ref_delay"))
        if type(snapshot.accepted_estimate) is not bool:
            raise ValueError("invalid delay snapshot acceptance state")
        accepted = snapshot.accepted_estimate
        if (
            snapshot_rate <= 0
            or snapshot_rate != sample_rate_hz
            or effective_seed < 0
            or operating < 0
            or (not accepted and operating != effective_seed)
            or engine_operating != operating
        ):
            raise ValueError("inconsistent delay snapshot")
    except Exception:
        out["measurement_state"] = "snapshot_unavailable"
        out["runtime_estimate_accepted"] = None
        return out

    out.update(
        {
            "measurement_state": (
                "accepted_measurement" if accepted else "awaiting_measurement"
            ),
            "sample_rate_hz": snapshot_rate,
            "runtime_snapshot_available": True,
            "runtime_estimate_accepted": accepted,
            "effective_seed_samples": effective_seed,
            "operating_delay_samples": operating,
            "operating_delay_ms": _samples_to_ms(operating, snapshot_rate),
        }
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Live TTS-echo / self-interruption probe.")
    ap.add_argument("--margin-db", type=float, default=0.0, help="barge_in_output_margin_db override")
    ap.add_argument("--gain", type=float, default=1.0, help="scale synthesized TTS output (in-app 'volume')")
    ap.add_argument("--sentences", type=int, default=4)
    ap.add_argument("--device", default=None, help="device_profile (default: config.device)")
    ap.add_argument("--config", default="config.json")
    # --- per-run strategy/device overrides (used by tools.interrupt_suite to sweep
    # the interrupt matrix without editing config.local.json) ---
    ap.add_argument("--input-device", default=None, help="mic device name/index override (e.g. 'AT2020USB-X', 'ALC285 Analog')")
    ap.add_argument("--output-device", default=None, help="speaker device name/index override")
    ap.add_argument("--coherence", choices=["on", "off"], default=None, help="force coherence_barge_in_enabled")
    ap.add_argument("--confirm-frames", type=int, default=None, help="coherence_confirm_frames override")
    ap.add_argument("--aec", choices=["on", "off"], default=None, help="force aec_enabled")
    ap.add_argument(
        "--ref-delay-ms",
        type=int,
        default=None,
        help="aec_ref_delay_ms configured-seed override; runtime auto-delay may replace it after accepting a measurement",
    )
    ap.add_argument("--relaxed-margin-db", type=float, default=None, help="aec_relaxed_margin_db override (level gate margin when AEC is on)")
    ap.add_argument("--speaker-gate", choices=["on", "off"], default=None, help="force speaker_gate_input (identity/user-detection gate in the barge path)")
    ap.add_argument("--min-speech-sec", type=float, default=None, help="barge_in_min_speech_sec override (sustained-speech needed to fire -- slower/higher confidence)")
    ap.add_argument("--loudness-margin-db", type=float, default=None, help="input_loudness_margin_db override (post-AEC barge: residual dB above the ambient floor)")
    ap.add_argument("--label", default=None, help="opaque label echoed back in the JSON (for the suite)")
    args = ap.parse_args()

    # utf-8 stdout so any non-ASCII never crashes the cp1252 console.
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    from core.config import apply_device_profile, load_config
    from core.engine import EngineCallbacks
    from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine

    cfg = load_config(args.config)
    device = args.device or cfg.get("device") or "desktop"
    cfg = apply_device_profile(cfg, device)
    sherpa_cfg = SherpaConfig.from_dict(cfg.get("sherpa", {}))

    # Collect all overrides and apply them in one pass (works whether SherpaConfig
    # is frozen or not). barge_in_output_margin_db is always set from --margin-db.
    overrides = {"barge_in_output_margin_db": args.margin_db}
    if args.input_device is not None:
        overrides["input_device"] = args.input_device
    if args.output_device is not None:
        overrides["output_device"] = args.output_device
    if args.coherence is not None:
        overrides["coherence_barge_in_enabled"] = (args.coherence == "on")
    if args.confirm_frames is not None:
        overrides["coherence_confirm_frames"] = int(args.confirm_frames)
    if args.aec is not None:
        overrides["aec_enabled"] = (args.aec == "on")
    if args.ref_delay_ms is not None:
        overrides["aec_ref_delay_ms"] = int(args.ref_delay_ms)
    if args.relaxed_margin_db is not None:
        overrides["aec_relaxed_margin_db"] = float(args.relaxed_margin_db)
    if args.speaker_gate is not None:
        overrides["speaker_gate_input"] = (args.speaker_gate == "on")
    if args.min_speech_sec is not None:
        overrides["barge_in_min_speech_sec"] = float(args.min_speech_sec)
    if args.loudness_margin_db is not None:
        overrides["input_loudness_margin_db"] = float(args.loudness_margin_db)
    for k, v in overrides.items():
        try:
            setattr(sherpa_cfg, k, v)
        except dataclasses.FrozenInstanceError:
            sherpa_cfg = dataclasses.replace(sherpa_cfg, **{k: v})

    if not getattr(sherpa_cfg, "tts_model", ""):
        return _emit_probe_result(
            {
                "error": (
                    "no sherpa.tts_model configured; run "
                    "`python -m tools.setup_models`"
                )
            },
            success=False,
        )

    engine = SherpaOnnxEngine(sherpa_cfg)

    barge_ins: list[float] = []
    gate_samples: list[tuple[float, float, bool]] = []
    # Coherence diagnostics per frame: (incoherent_fraction, baseline,
    # effective_margin, delay_ms, decided). The probe never talks back, so EVERY
    # frame is echo-only -- the run records the detector's effective margin and
    # whether it ever fired on the assistant.
    coh_samples: list[tuple[float, float, float, float, object]] = []
    # Adaptive-DTD diagnostics per frame: (D, z_raw, z_resid, z_coh, fired). The
    # probe never talks back, so EVERY frame is echo-only -- this records the fused
    # statistic D and whether the detector fired.
    dtd_samples: list[tuple[float, float, float, float, object]] = []
    peak_playback = 0.0
    stimulus_plan = _build_stimulus_plan(args.sentences)
    stimulus_id_builder = _StimulusIdBuilder()

    # Instrument the unenrolled echo gate to record exactly what it compares.
    _orig_llu = engine._looks_like_user

    def _wrapped_llu(samples, mic_raw=None):
        # The engine now calls _looks_like_user(samples, mic_raw): post-AEC block
        # for the level fallback + the RAW pre-AEC mic for the coherence detector.
        # Forward both so the probe instruments the exact same path the engine runs.
        passed = _orig_llu(samples, mic_raw)
        try:
            gate_samples.append((round(_rms(samples), 5), round(float(engine._playback_level), 5), bool(passed)))
            det = getattr(engine, "_echo_coherence", None)
            if det is not None and det.last_decided is not None:
                coh_samples.append((
                    round(float(det.last_incoherent_fraction), 4),
                    round(float(det.last_baseline), 4),
                    round(float(det.last_effective_margin), 4),
                    round(float(det.last_delay_ms), 1),
                    det.last_decided,
                ))
            dtd = getattr(engine, "_dtd", None)
            if dtd is not None and dtd.last_decided is not None:
                dtd_samples.append((
                    round(float(dtd.last_D), 3),
                    round(float(dtd.last_z_raw), 3),
                    round(float(dtd.last_z_resid), 3),
                    round(float(dtd.last_z_coh), 3),
                    dtd.last_decided,
                ))
        except Exception:
            pass
        return passed

    engine._looks_like_user = _wrapped_llu  # type: ignore[assignment]

    # In-app gain: scale synthesized output so we can sweep "volume" deterministically
    # (note: this scales BOTH the speaker output and the _playback_level reference,
    # so it does NOT expose the OS-volume scale mismatch -- vary OS volume for that).
    if args.gain != 1.0:
        _orig_synth = engine._synthesize

        def _scaled_synth(text, write):
            _orig_synth(text, lambda s: write(__import__("numpy").asarray(s, dtype="float32") * args.gain))

        engine._synthesize = _scaled_synth  # type: ignore[assignment]

    cb = EngineCallbacks(
        on_barge_in=lambda: barge_ins.append(time.monotonic()),
    )

    erle_acc = {"near2": 0.0, "post2": 0.0, "far_frames": 0}
    spoken = 0

    def _probe_body() -> None:
        nonlocal peak_playback, spoken

        # The engine builds ``_aec`` inside start(), so the ERLE wrapper and
        # every later warm/body/tail action belong inside the cleanup owner.
        if getattr(engine, "_aec", None) is not None:
            import numpy as _np

            _orig_proc = engine._aec.process_16k

            def _erle_proc(near, far):
                post = _orig_proc(near, far)
                try:
                    n = _np.asarray(near, dtype="float32").reshape(-1)
                    f = _np.asarray(far, dtype="float32").reshape(-1)
                    p = _np.asarray(post, dtype="float32").reshape(-1)
                    if f.size and float(_np.sqrt(_np.mean(f * f))) > 1e-4:
                        erle_acc["far_frames"] += 1
                        erle_acc["near2"] += float(_np.sum(n * n))
                        erle_acc["post2"] += float(_np.sum(p * p))
                except Exception:
                    pass
                return post

            engine._aec.process_16k = _erle_proc  # type: ignore[assignment]

        time.sleep(1.0)  # let the capture loop spin up
        for text in stimulus_plan:
            done = threading.Event()
            engine.speak(text, on_done=done.set)
            stimulus_id_builder.record(text)
            deadline = time.monotonic() + 20.0
            while not done.wait(0.05):
                lvl = float(engine._playback_level)
                if lvl > peak_playback:
                    peak_playback = lvl
                if time.monotonic() > deadline:
                    break
            spoken += 1
            time.sleep(0.4)
        time.sleep(0.5)

    lifecycle = _run_probe_lifecycle(
        lambda: engine.start(cb),
        _probe_body,
        engine.stop,
    )
    if not lifecycle.report_allowed:
        return _emit_probe_result(
            {
                "error": "probe-lifecycle-failed",
                "start_error_type": lifecycle.start_error_type,
                "probe_error_type": lifecycle.probe_error_type,
                "stop_error_type": lifecycle.stop_error_type,
            },
            success=False,
        )

    aec_reference_delay = _aec_reference_delay_summary(
        engine,
        sherpa_cfg,
        stop_completed=lifecycle.stop_completed,
    )

    ratios = [
        round(20.0 * math.log10(mic / pb), 1)
        for (mic, pb, _) in gate_samples
        if pb > 1e-6 and mic > 1e-6
    ]
    ratios_sorted = sorted(ratios)

    # Coherence diagnostic block. All frames are quiet echo-only because the probe
    # never talks over itself. Record fires and descriptive margin/headroom values;
    # they do not select a configuration or demonstrate a real talk-over.
    coh = None
    det = getattr(engine, "_echo_coherence", None)
    if coh_samples:
        fr = sorted(s[0] for s in coh_samples)
        echo_only = sorted(s[0] for s in coh_samples if s[4] is False)
        coh_fired = sum(1 for s in coh_samples if s[4] is True)
        delays = sorted(s[3] for s in coh_samples)
        p95 = echo_only[max(0, int(0.95 * (len(echo_only) - 1)))] if echo_only else None
        settled_baseline = coh_samples[-1][1]
        settled_margin = coh_samples[-1][2]  # detector's effective margin
        floor = float(getattr(sherpa_cfg, "coherence_margin_delta", 0.08))
        coh = {
            "active": det is not None,
            "frames": len(coh_samples),
            "coherence_fired_on_own_tts": coh_fired,  # >0 = self-interrupt risk
            "margin_floor": floor,
            "self_calibrated_margin": settled_margin,
            "margin_widened_by_room": (settled_margin > floor + 1e-6),
            "settled_baseline": settled_baseline,
            "echo_only_incoherent_p50": (echo_only[len(echo_only) // 2] if echo_only else None),
            "echo_only_incoherent_p95": p95,
            "headroom_p95": (round(settled_baseline + settled_margin - p95, 3) if p95 is not None else None),
            "median_delay_ms": (delays[len(delays) // 2] if delays else None),
            "incoherent_fraction_samples": fr[:40],
            "hint": (
                "coherence_fired_on_own_tts, headroom_p95, and "
                "self_calibrated_margin are descriptive quiet echo-only measurements. "
                "This diagnostic does not select or tune a configuration. "
                + _PHYSICAL_ACCEPTANCE_GUIDANCE
            ),
        }
    elif det is not None:
        coh = {
            "active": True,
            "frames": 0,
            "note": f"{_NO_COUPLING_GUIDANCE} {_PHYSICAL_ACCEPTANCE_GUIDANCE}",
        }

    # Adaptive-DTD diagnostic block. All frames are quiet echo-only because the
    # probe never talks over itself. Record D, K, headroom, and fires descriptively;
    # these measurements do not pick K or demonstrate a real talk-over.
    dtd_blk = None
    dtd = getattr(engine, "_dtd", None)
    if dtd_samples:
        Ds = sorted(s[0] for s in dtd_samples)
        fired = sum(1 for s in dtd_samples if s[4] is True)
        p95D = Ds[max(0, int(0.95 * (len(Ds) - 1)))]
        K = float(getattr(sherpa_cfg, "dtd_k", 5.0))
        dtd_blk = {
            "active": True,
            "frames": len(dtd_samples),
            "K": K,
            "dtd_fired_on_own_tts": fired,             # >0 = self-interrupt risk
            "echo_only_D_p50": Ds[len(Ds) // 2],
            "echo_only_D_p95": p95D,
            "echo_only_D_max": Ds[-1],
            "headroom_K_minus_p95D": round(K - p95D, 3),
            "D_samples": Ds[-40:],
            "hint": (
                "dtd_fired_on_own_tts and headroom_K_minus_p95D are descriptive "
                "quiet echo-only measurements. This diagnostic does not select or "
                "tune a configuration. " + _PHYSICAL_ACCEPTANCE_GUIDANCE
            ),
        }
    elif dtd is not None:
        dtd_blk = {
            "active": True,
            "frames": 0,
            "note": f"{_NO_COUPLING_GUIDANCE} {_PHYSICAL_ACCEPTANCE_GUIDANCE}",
        }

    erle_db = None
    if erle_acc["post2"] > 0 and erle_acc["near2"] > 0:
        erle_db = round(10.0 * math.log10(erle_acc["near2"] / erle_acc["post2"]), 1)

    out = {
        "label": args.label,
        "input_device": getattr(sherpa_cfg, "input_device", None),
        "strategy": {
            "coherence": getattr(sherpa_cfg, "coherence_barge_in_enabled", None),
            "confirm_frames": getattr(sherpa_cfg, "coherence_confirm_frames", None),
            "aec": getattr(sherpa_cfg, "aec_enabled", None),
            "aec_ref_delay_ms": getattr(sherpa_cfg, "aec_ref_delay_ms", None),
            "margin_db": args.margin_db,
        },
        "aec_reference_delay": aec_reference_delay,
        "erle_db": erle_db,
        "aec_far_active_frames": erle_acc["far_frames"],
        "gain": args.gain,
        "device": device,
        "sentences_spoken": spoken,
        "stimulus_id": stimulus_id_builder.stimulus_id(),
        "self_interruptions": len(barge_ins),
        "vad_flagged_during_play": len(gate_samples),
        "gate_passed_count": sum(1 for (_, _, p) in gate_samples if p),
        "peak_playback_level": round(peak_playback, 5),
        "median_mic_over_playback_dB": (ratios_sorted[len(ratios_sorted) // 2] if ratios_sorted else None),
        "max_mic_over_playback_dB": (ratios_sorted[-1] if ratios_sorted else None),
        "mic_over_playback_dB_samples": ratios[:40],
        "coherence": coh,
        "adaptive_dtd": dtd_blk,
        "note": (
            "self_interruptions>0 means the assistant's own TTS tripped barge-in. "
            "peak_playback_level~0 or vad_flagged_during_play==0 identifies a result "
            "without measured echo coupling. "
            f"{_NO_COUPLING_GUIDANCE} "
            "erle_db aggregates all AEC-active frames and may span the configured seed "
            "plus multiple accepted operating delays; it is not attributable only to "
            "the final delay snapshot. "
            f"{_PHYSICAL_ACCEPTANCE_GUIDANCE}"
        ),
    }
    return _emit_probe_result(out, success=True)


if __name__ == "__main__":
    raise SystemExit(main())
