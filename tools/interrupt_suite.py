#!/usr/bin/env python3
"""Interrupt (barge-in) test SUITE -- sweeps the interrupt matrix on real hardware.

PLAYS AUDIO OUT LOUD and captures the mic. For each (microphone x interrupt
strategy) cell it drives :mod:`tools.echo_probe` (the real sherpa engine) which
plays N TTS sentences and -- because the probe never talks over itself -- counts
every barge-in as a SELF-INTERRUPT. The suite tabulates self-interrupts +
coherence headroom across the matrix only to identify quiet-control candidates:
cells that receive the assistant's echo without self-firing. It does not show
that a real human talk-over interrupts the assistant.

    python -m tools.interrupt_suite                 # both mics, default matrix
    python -m tools.interrupt_suite --mics at2020   # one mic
    python -m tools.interrupt_suite --sentences 4

IMPORTANT: stay QUIET while it runs -- any voice during a cell looks like a real
barge and inflates that cell's self-interrupt count. The fires-on-a-REAL-barge
direction is covered by the offline unit tests (tests/test_echo_coherence.py).
The physical gate starts through `./live.sh`, talks over bare-speaker playback,
then analyzes that exact retained run with `python -m tools.live_audio_ab`; this
suite measures only the no-self-interrupt half.

Each cell is a fresh echo_probe subprocess (clean device open/close). A mic is
liveness-checked with a short arecord capture first; a dead/muted mic (the AT2020
touch-mute) is reported and skipped.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import wave
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

import numpy as np

# Microphone specs: label -> (engine device name, arecord device for liveness).
MICS = {
    "alc285": ("ALC285 Analog", "plughw:1,0"),  # laptop built-in mic
    "at2020": ("AT2020USB-X", "plughw:2,0"),  # external USB condenser
}
OUTPUT_DEVICE = "ALC285 Analog"  # the only real speaker on this box

# Interrupt strategies to sweep. Each is a set of echo_probe overrides.
STRATEGIES = [
    {
        "label": "coherence-confirm1",
        "coherence": "on",
        "confirm_frames": 1,
        "aec": "off",
        "margin_db": 0.0,
    },
    {
        "label": "coherence-confirm2",
        "coherence": "on",
        "confirm_frames": 2,
        "aec": "off",
        "margin_db": 0.0,
    },
    {
        "label": "coherence-confirm3",
        "coherence": "on",
        "confirm_frames": 3,
        "aec": "off",
        "margin_db": 0.0,
    },
    {
        "label": "level-margin6",
        "coherence": "off",
        "confirm_frames": 2,
        "aec": "off",
        "margin_db": 6.0,
    },
    {
        "label": "configured-aec",
        "coherence": "off",
        "confirm_frames": 2,
        "aec": "on",
        "margin_db": 0.0,
    },
]


class DiagnosticStatus(str, Enum):
    QUIET_CONTROL_CANDIDATE = "quiet-control-candidate"
    NO_SAFE_CANDIDATE = "no-safe-candidate"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class DiagnosticOutcome:
    status: DiagnosticStatus
    reason: str
    candidate_labels: tuple[str, ...]
    valid_coupled_cells: int
    uncoupled_zero_cells: int
    error_cells: int
    matrix_partial: bool
    live_validation_required: bool = True

    def __post_init__(self) -> None:
        DiagnosticOutcome._validate(self)

    def _validate(self) -> None:
        if type(self) is not DiagnosticOutcome:
            raise TypeError("outcome must be an exact DiagnosticOutcome")
        if type(self.status) is not DiagnosticStatus:
            raise TypeError("status must be a DiagnosticStatus")
        if type(self.reason) is not str or not self.reason:
            raise TypeError("reason must be a nonempty str")
        if type(self.candidate_labels) is not tuple or any(
            type(label) is not str or not label for label in self.candidate_labels
        ):
            raise TypeError("candidate_labels must contain exact nonempty strings")
        if self.candidate_labels != tuple(sorted(set(self.candidate_labels))):
            raise ValueError("candidate_labels must be unique and lexically sorted")
        counts = (
            self.valid_coupled_cells,
            self.uncoupled_zero_cells,
            self.error_cells,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise TypeError("diagnostic counts must be exact nonnegative integers")
        if type(self.matrix_partial) is not bool:
            raise TypeError("matrix_partial must be an exact bool")
        if self.matrix_partial != (self.error_cells > 0):
            raise ValueError("matrix_partial must describe error_cells")
        if self.live_validation_required is not True:
            raise ValueError("live_validation_required must remain true")

        reasons = {
            DiagnosticStatus.QUIET_CONTROL_CANDIDATE: {
                "coupled-zero-self-interruption-observed"
            },
            DiagnosticStatus.NO_SAFE_CANDIDATE: {
                "no-coupled-zero-self-interruption-observed"
            },
            DiagnosticStatus.INCONCLUSIVE: {
                "invalid-mic-selection",
                "no-valid-coupled-cell",
                "report-publication-failed",
            },
        }
        if self.reason not in reasons[self.status]:
            raise ValueError("reason does not match diagnostic status")
        if self.status is DiagnosticStatus.QUIET_CONTROL_CANDIDATE:
            if (
                not self.candidate_labels
                or len(self.candidate_labels) > self.valid_coupled_cells
            ):
                raise ValueError("candidate status requires coupled candidate labels")
        elif self.candidate_labels:
            raise ValueError("only candidate status may carry candidate labels")
        if (
            self.status is DiagnosticStatus.NO_SAFE_CANDIDATE
            and not self.valid_coupled_cells
        ):
            raise ValueError("no-safe status requires valid coupled evidence")
        if self.reason in {"invalid-mic-selection", "no-valid-coupled-cell"}:
            if self.valid_coupled_cells:
                raise ValueError("this inconclusive reason excludes coupled evidence")
        if self.reason == "invalid-mic-selection" and (
            self.uncoupled_zero_cells or not self.error_cells
        ):
            raise ValueError(
                "invalid mic selection requires an explicit selection error"
            )
        if self.reason == "report-publication-failed" and not self.error_cells:
            raise ValueError("publication failure requires an explicit error")

    @property
    def exit_code(self) -> int:
        DiagnosticOutcome._validate(self)
        return {
            DiagnosticStatus.QUIET_CONTROL_CANDIDATE: 0,
            DiagnosticStatus.NO_SAFE_CANDIDATE: 1,
            DiagnosticStatus.INCONCLUSIVE: 2,
        }[self.status]

    def as_dict(self) -> dict[str, object]:
        DiagnosticOutcome._validate(self)
        return {
            "status": self.status.value,
            "reason": self.reason,
            "candidate_labels": list(self.candidate_labels),
            "valid_coupled_cells": self.valid_coupled_cells,
            "uncoupled_zero_cells": self.uncoupled_zero_cells,
            "error_cells": self.error_cells,
            "matrix_partial": self.matrix_partial,
            "live_validation_required": self.live_validation_required,
        }


def classify_outcome(rows: Sequence[object]) -> DiagnosticOutcome:
    """Reduce matrix rows without performing I/O or selecting a winner."""
    candidate_labels: list[str] = []
    valid_coupled_cells = 0
    uncoupled_zero_cells = 0
    error_cells = 0

    for row in rows:
        if type(row) is not dict:
            error_cells += 1
            continue
        label = row.get("label")
        self_int = row.get("self_int")
        is_valid = (
            row.get("error") is None
            and type(label) is str
            and bool(label)
            and type(self_int) is int
            and self_int >= 0
            and type(row.get("coupled")) is bool
        )
        if not is_valid:
            error_cells += 1
            continue

        coupled = row["coupled"]
        if coupled:
            valid_coupled_cells += 1
            if self_int == 0:
                candidate_labels.append(label)
        elif self_int == 0:
            uncoupled_zero_cells += 1

    sorted_candidates = tuple(sorted(set(candidate_labels)))
    matrix_partial = error_cells > 0
    if sorted_candidates:
        status = DiagnosticStatus.QUIET_CONTROL_CANDIDATE
        reason = "coupled-zero-self-interruption-observed"
    elif valid_coupled_cells:
        status = DiagnosticStatus.NO_SAFE_CANDIDATE
        reason = "no-coupled-zero-self-interruption-observed"
    else:
        status = DiagnosticStatus.INCONCLUSIVE
        reason = "no-valid-coupled-cell"

    return DiagnosticOutcome(
        status=status,
        reason=reason,
        candidate_labels=sorted_candidates,
        valid_coupled_cells=valid_coupled_cells,
        uncoupled_zero_cells=uncoupled_zero_cells,
        error_cells=error_cells,
        matrix_partial=matrix_partial,
    )


def _wav_rms(
    path: str,
    *,
    opener: Callable[..., Any] | None = None,
) -> tuple[float, float, int]:
    open_wave = wave.open if opener is None else opener
    try:
        w = open_wave(path, "rb")
        n = w.getnframes()
        a = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
        w.close()
    except Exception:
        return 0.0, 0.0, 0
    if a.size == 0:
        return 0.0, 0.0, 0
    return (
        float(np.sqrt(np.mean(a * a))),
        float(np.max(np.abs(a))),
        int(np.count_nonzero(a)),
    )


def mic_is_live(
    arecord_dev: str,
    *,
    runner: Callable[..., Any] | None = None,
    rms_reader: Callable[[str], tuple[float, float, int]] | None = None,
    capture_path: str = "/tmp/interrupt_suite_liveness.wav",
) -> tuple[bool, float]:
    """Capture ~2 s and report whether the mic produces signal (vs hard-muted)."""
    run = subprocess.run if runner is None else runner
    read_rms = _wav_rms if rms_reader is None else rms_reader
    try:
        run(
            [
                "arecord",
                "-D",
                arecord_dev,
                "-f",
                "S16_LE",
                "-r",
                "44100",
                "-c",
                "1",
                "-d",
                "2",
                capture_path,
            ],
            check=True,
            capture_output=True,
            timeout=10,
        )
    except Exception:
        print(f"    liveness capture FAILED on {arecord_dev}")
        return False, 0.0
    rms, peak, nz = read_rms(capture_path)
    # A muted mic delivers (near-)exact zeros; a live one always has a noise floor.
    return (rms > 1e-5 and nz > 0), rms


def _reject_json_constant(_value: str) -> None:
    raise ValueError("non-finite JSON constant")


def run_cell(
    mic_name: str,
    strat: dict,
    sentences: int,
    timeout: float,
    *,
    runner: Callable[..., Any] | None = None,
    python_executable: str | None = None,
    output_device: str = OUTPUT_DEVICE,
) -> dict:
    run = subprocess.run if runner is None else runner
    label = f"{mic_name}/{strat['label']}"
    cmd = [
        sys.executable if python_executable is None else python_executable,
        "-m",
        "tools.echo_probe",
        "--label",
        label,
        "--input-device",
        mic_name,
        "--output-device",
        output_device,
        "--coherence",
        strat["coherence"],
        "--confirm-frames",
        str(strat["confirm_frames"]),
        "--aec",
        strat["aec"],
        "--margin-db",
        str(strat["margin_db"]),
        "--sentences",
        str(sentences),
    ]
    try:
        p = run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"label": label, "error": "timeout"}
    except Exception:
        return {"label": label, "error": "subprocess-failed"}
    returncode = getattr(p, "returncode", None)
    if type(returncode) is not int or returncode != 0:
        return {"label": label, "error": "subprocess-failed"}
    if type(getattr(p, "stdout", None)) is not str:
        return {"label": label, "error": "invalid-stdout"}
    out = p.stdout.strip()
    # echo_probe prints one JSON object to stdout; logs go to stderr. Be defensive
    # and parse from the first brace in case anything leaks onto stdout.
    i = out.find("{")
    if i < 0:
        return {"label": label, "error": "no-json"}
    try:
        result = json.loads(
            out[i:],
            parse_constant=_reject_json_constant,
        )
    except Exception:
        return {"label": label, "error": "invalid-json"}
    if type(result) is not dict:
        return {"label": label, "error": "json-not-object"}
    return result


def summarize(cell: dict) -> dict:
    coherence = cell.get("coherence")
    coh = coherence if type(coherence) is dict else {}
    error = cell.get("error")
    if coherence is not None and type(coherence) is not dict and error is None:
        error = "malformed-coherence"
    return {
        "label": cell.get("label"),
        "self_int": cell.get("self_interruptions"),
        "coh_fired": coh.get("coherence_fired_on_own_tts"),
        "headroom_p95": coh.get("headroom_p95"),
        "self_cal_margin": coh.get("self_calibrated_margin"),
        "vad_flagged": cell.get("vad_flagged_during_play"),
        "peak_play": cell.get("peak_playback_level"),
        "mic_over_play_dB": cell.get("median_mic_over_playback_dB"),
        "error": error,
    }


def _is_nonnegative_finite_number(value: object) -> bool:
    if type(value) is int:
        return value >= 0
    return type(value) is float and math.isfinite(value) and value >= 0


def coupled(s: dict) -> bool | None:
    """Did the assistant's echo actually reach the mic this cell? (Else the
    zero-self-interrupt result is vacuous -- volume too low / mic muted.)"""
    vad_flagged = s.get("vad_flagged")
    peak_play = s.get("peak_play")
    valid_vad = type(vad_flagged) is int and vad_flagged >= 0
    valid_peak = _is_nonnegative_finite_number(peak_play)
    if not valid_vad or not valid_peak:
        return None
    return vad_flagged > 0 and peak_play > 1e-4


def _invalid_mic_outcome() -> DiagnosticOutcome:
    return DiagnosticOutcome(
        status=DiagnosticStatus.INCONCLUSIVE,
        reason="invalid-mic-selection",
        candidate_labels=(),
        valid_coupled_cells=0,
        uncoupled_zero_cells=0,
        error_cells=1,
        matrix_partial=True,
    )


def _publish_report(path: str, report: dict[str, object]) -> None:
    parent = os.path.dirname(path) or "."
    if parent != ".":
        os.makedirs(parent, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=parent,
            prefix=".interrupt-suite-",
            suffix=".tmp",
            delete=False,
        ) as f:
            temporary_path = f.name
            json.dump(report, f, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass


def _render_outcome(outcome: DiagnosticOutcome) -> None:
    print("\n================ DIAGNOSTIC OUTCOME ================")
    print(f"  status: {outcome.status.value}")
    print(f"  reason: {outcome.reason}")
    if outcome.candidate_labels:
        print("  quiet-control candidates (all labels, lexical order):")
        for label in outcome.candidate_labels:
            print(f"    {label}")
        print("  These cells only show quiet playback without a self-interruption.")
    elif outcome.uncoupled_zero_cells:
        print(
            "  Zero-self-interruption cells lacked measured echo coupling, "
            "so they are not evidence."
        )
    if outcome.matrix_partial:
        print(f"  matrix: partial ({outcome.error_cells} invalid or failed cell(s))")
    print(
        "  Live gate still required: start the bare-speaker session with ./live.sh, "
        "talk over playback,"
    )
    print("  then inspect that exact retained run:")
    print("    python -m tools.live_audio_ab logs/runs/run-<id>.txt")


def main(
    argv: Sequence[str] | None = None,
    *,
    mic_probe: Callable[[str], tuple[bool, float]] | None = None,
    cell_probe: Callable[[str, dict, int, float], dict] | None = None,
    report_publisher: Callable[[str, dict[str, object]], None] | None = None,
) -> int:
    ap = argparse.ArgumentParser(
        description="Sweep the barge-in interrupt matrix on real hardware."
    )
    ap.add_argument("--mics", default="alc285,at2020", help="comma list: alc285,at2020")
    ap.add_argument("--sentences", type=int, default=3)
    ap.add_argument(
        "--timeout", type=float, default=90.0, help="per-cell subprocess timeout (s)"
    )
    ap.add_argument("--out", default="test-reports/interrupt_suite.json")
    args = ap.parse_args(argv)

    mics = [m.strip() for m in args.mics.split(",") if m.strip()]
    rows: list[dict] = []
    raw: list[dict] = []
    invalid_selection = not mics or any(mic not in MICS for mic in mics)
    if invalid_selection:
        print(f"invalid --mics selection; choose a comma list from: {', '.join(MICS)}")
        outcome = _invalid_mic_outcome()
    else:
        probe_mic = mic_is_live if mic_probe is None else mic_probe
        probe_cell = run_cell if cell_probe is None else cell_probe
        for mic in mics:
            engine_name, arecord_dev = MICS[mic]
            print(f"\n=== MIC: {mic} ({engine_name}) ===")
            try:
                liveness = probe_mic(arecord_dev)
            except Exception:
                liveness = None
            valid_liveness = (
                type(liveness) is tuple
                and len(liveness) == 2
                and type(liveness[0]) is bool
                and type(liveness[1]) is float
                and math.isfinite(liveness[1])
                and liveness[1] >= 0
            )
            if not valid_liveness:
                print("    liveness probe FAILED -- skipping")
                rows.append(
                    {
                        "label": f"{mic}/<all>",
                        "error": "mic-liveness-failed",
                        "self_int": None,
                    }
                )
                continue
            live, rms = liveness
            print(
                f"    liveness: rms={rms:.6f} -> {'LIVE' if live else 'DEAD/MUTED -- skipping'}"
            )
            if not live:
                rows.append(
                    {"label": f"{mic}/<all>", "error": "mic-dead", "self_int": None}
                )
                continue
            for strat in STRATEGIES:
                print(f"    cell: {strat['label']} ...", flush=True)
                expected_label = f"{engine_name}/{strat['label']}"
                try:
                    cell = probe_cell(engine_name, strat, args.sentences, args.timeout)
                except Exception:
                    cell = {"label": expected_label, "error": "cell-probe-failed"}
                if type(cell) is not dict:
                    cell = {"label": expected_label, "error": "cell-result-not-object"}
                cell["label"] = expected_label
                cell["_mic"] = mic
                raw.append(cell)
                try:
                    s = summarize(cell)
                except Exception:
                    s = {
                        "label": expected_label,
                        "self_int": None,
                        "coh_fired": None,
                        "headroom_p95": None,
                        "self_cal_margin": None,
                        "vad_flagged": None,
                        "peak_play": None,
                        "mic_over_play_dB": None,
                        "error": "cell-summary-failed",
                    }
                s["label"] = expected_label
                coupling = coupled(s)
                s["coupled"] = coupling
                if coupling is None and s.get("error") is None:
                    s["error"] = "malformed-coupling-input"
                rows.append(s)
                print(
                    f"        self_int={s['self_int']} coh_fired={s['coh_fired']} "
                    f"headroom_p95={s['headroom_p95']} coupled={s['coupled']} "
                    f"vad={s['vad_flagged']} peak={s['peak_play']} err={s['error']}"
                )
        outcome = classify_outcome(rows)

    # --- table ---
    print("\n================ INTERRUPT MATRIX ================")
    hdr = f"{'cell':28s} {'self_int':>8s} {'coh_fire':>8s} {'headroom':>9s} {'coupled':>7s} {'vad':>5s}"
    print(hdr)
    print("-" * len(hdr))
    for s in rows:
        if s.get("error") and s.get("self_int") is None:
            print(
                f"{str(s.get('label')):28s} {'--':>8s} {'--':>8s} {'--':>9s} {'--':>7s} {'--':>5s}  ({s['error']})"
            )
            continue
        print(
            f"{str(s.get('label')):28s} {str(s.get('self_int')):>8s} {str(s.get('coh_fired')):>8s} "
            f"{str(s.get('headroom_p95')):>9s} {str(s.get('coupled')):>7s} {str(s.get('vad_flagged')):>5s}"
        )

    report = {
        "rows": rows,
        "raw": raw,
        "diagnostic_outcome": outcome.as_dict(),
    }
    publish = _publish_report if report_publisher is None else report_publisher
    try:
        publish(args.out, report)
        print(f"\n[suite] raw results -> {args.out}")
    except Exception:
        print("[suite] could not publish report")
        outcome = replace(
            outcome,
            status=DiagnosticStatus.INCONCLUSIVE,
            reason="report-publication-failed",
            candidate_labels=(),
            error_cells=outcome.error_cells + 1,
            matrix_partial=True,
        )

    _render_outcome(outcome)
    return outcome.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
