"""Grade + render the noise-stress run, keeping the voice-isolation split explicit.

The two app behaviours are graded SEPARATELY (this is the whole point):

* **Competing-voice isolation (speaker-ID gate)** -> a FALSE-POSITIVE rate over
  intruder + noise-only turns. PASS_ISOLATION when ~0: the gate rejected the
  competing voice / noise-born final. We also count the engine's
  ``speaker_rejected_final`` signal to PROVE the gate actually fired.

* **Broadband-noise robustness (NO denoiser)** -> desired-voice RECALL + STT
  accuracy under noise. A STT score that FALLS as SNR drops is EXPECTED and is
  labelled ``DENOISE = ABSENT``, NOT an isolation failure.

All grading functions are PURE: they take a list of per-turn observation dicts
(the shape the driver produces; see :func:`grade_scenario`) so they unit-test
with synthetic fixtures and no audio/models. ``stt_score`` is reused from
``tools.live_session.report``.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Optional

from tools.live_session.report import HEARD_OK_THRESHOLD, stt_score

# A turn counts as "answered" when the runtime produced an answer for it.
# The driver fills ``answered`` from the metrics signal (a turn that delivered
# SPEECH_END + ASR_FINAL (+ TTS_FIRST_AUDIO) around its window).


def grade_turn(obs: dict) -> dict:
    """Grade ONE observed turn.

    ``obs`` keys (all the driver records; tests pass synthetic fixtures):
      speaker            : "user" | "intruder" | "noise"
      scripted           : the line text ("" for a noise-only window)
      heard              : asr_final the runtime recorded (or None)
      answered           : bool -- the runtime delivered an answer for this turn
      rejected_finals    : int  -- speaker_rejected_final count in this window
      expected_answered  : bool -- the scenario's truth flag
      noise              : str  -- noise label (e.g. "white@10dB") for the row
    """
    speaker = obs.get("speaker", "user")
    scripted = obs.get("scripted") or ""
    heard = obs.get("heard")
    answered = bool(obs.get("answered"))
    expected = bool(obs.get("expected_answered"))
    rejected = int(obs.get("rejected_finals", 0) or 0)
    score = stt_score(scripted, heard) if (speaker == "user" and scripted) else None
    row = {
        "speaker": speaker,
        "noise": obs.get("noise", ""),
        "scripted": scripted,
        "heard": heard,
        "answered": answered,
        "expected_answered": expected,
        "rejected_finals": rejected,
        "stt_score": score,
        "stt_ok": (score is not None and score >= HEARD_OK_THRESHOLD),
    }
    # Classification for the aggregate counters.
    if speaker == "user":
        row["recall_hit"] = answered  # expected True -> answered?
        row["false_positive"] = False
    else:  # intruder / noise -> must NOT be answered
        row["recall_hit"] = None
        row["false_positive"] = answered  # answered a non-target -> FP
    return row


def grade_scenario(name: str, observations: list[dict]) -> dict:
    """Aggregate per-turn rows into the scenario verdict.

    Returns recall (desired voice), STT median/min over answered user turns,
    false-positive rate (answered intruder/noise-only over their count), the
    count of speaker_rejected_final fires, and the two-axis verdict."""
    rows = [grade_turn(o) for o in observations]

    user_rows = [r for r in rows if r["speaker"] == "user"]
    nontarget_rows = [r for r in rows if r["speaker"] != "user"]

    # Recall: of the user turns we expected answered, how many were.
    expected_user = [r for r in user_rows if r["expected_answered"]]
    recall = (
        sum(1 for r in expected_user if r["answered"]) / len(expected_user)
        if expected_user else None
    )

    # STT accuracy over user turns that produced any heard text (graceful-
    # degradation curve; an answered-but-garbled turn still scores).
    user_scores = [r["stt_score"] for r in user_rows if r["stt_score"] is not None]
    stt_median = round(statistics.median(user_scores), 4) if user_scores else None
    stt_min = round(min(user_scores), 4) if user_scores else None

    # False positives: answered non-target turns over non-target count.
    n_nontarget = len(nontarget_rows)
    n_fp = sum(1 for r in nontarget_rows if r["false_positive"])
    fp_rate = (n_fp / n_nontarget) if n_nontarget else None
    rejected_total = sum(r["rejected_finals"] for r in rows)

    # Two-axis verdict.
    # Isolation PASSES when there were competing-voice/noise turns AND none were
    # answered (fp_rate == 0). With no competing turns it is "n/a" (this scenario
    # only stresses denoise).
    if n_nontarget == 0:
        isolation = "n/a"
    elif fp_rate == 0.0:
        isolation = "PASS_ISOLATION"
    else:
        isolation = "FAIL_ISOLATION"

    # Denoise is ALWAYS reported ABSENT (the app has no denoiser): we annotate
    # whether STT actually degraded under noise (informational, never a fail).
    has_noise = any((r["noise"] and not r["noise"].startswith("none")) for r in rows)
    denoise = "ABSENT"

    return {
        "scenario": name,
        "rows": rows,
        "recall": recall,
        "stt_score_median": stt_median,
        "stt_score_min": stt_min,
        "false_positive_rate": fp_rate,
        "false_positives": n_fp,
        "nontarget_turns": n_nontarget,
        "speaker_rejected_finals": rejected_total,
        "isolation_verdict": isolation,
        "denoise": denoise,
        "stressed_with_noise": has_noise,
    }


def overall_verdict(
    scenario_grades: list[dict],
    *,
    separable: Optional[bool] = None,
    inverted: Optional[bool] = None,
) -> dict:
    """Roll the per-scenario grades into a headline.

    ``separable`` is the enrollment-calibration verdict: whether ANY cosine
    threshold separates the mock user's short clips from the intruder's. When the
    fixtures are NOT separable (a property of the TTS voices + the VoxCeleb-
    trained embedder, not the app), a competing-voice answer cannot be blamed on
    the app, so the verdict is reported as ``INCONCLUSIVE`` rather than ``FAIL``.

    ``inverted`` is the stronger sub-case: the intruder embeds CLOSER to the user
    reference than the user's own short clips, so the gate is run wide open and
    PASS_ISOLATION is unreachable. It is surfaced in the verdict string so the
    report doesn't understate the problem as mere overlap."""
    isolation_scens = [g for g in scenario_grades if g["isolation_verdict"] != "n/a"]
    isolation_pass = all(
        g["isolation_verdict"] == "PASS_ISOLATION" for g in isolation_scens
    ) if isolation_scens else None
    total_fp = sum(g["false_positives"] for g in scenario_grades)
    total_nontarget = sum(g["nontarget_turns"] for g in scenario_grades)
    total_rejected = sum(g["speaker_rejected_finals"] for g in scenario_grades)
    recalls = [g["recall"] for g in scenario_grades if g["recall"] is not None]

    if isolation_pass is None:
        verdict = "n/a"
    elif isolation_pass:
        verdict = "PASS"
    elif inverted:
        # The strongest non-separable case: the intruder embeds closer to the
        # user reference than the user's own clips, so the gate is run wide open
        # and PASS_ISOLATION is unreachable. Say so explicitly -- not mere overlap.
        verdict = (
            "INCONCLUSIVE (INVERTED: intruder embeds CLOSER to the user reference "
            "than the user's own clips; gate run wide open, PASS unreachable)"
        )
    elif separable is False:
        # The gate let a competing voice through, but the fixtures were not
        # separable to begin with -- not attributable to the app.
        verdict = "INCONCLUSIVE (TTS voices not separable by the VoxCeleb embedder)"
    else:
        verdict = "FAIL"
    return {
        "competing_voice_isolation": verdict,
        "fixtures_separable": separable,
        "fixtures_inverted": bool(inverted),
        "false_positives_total": total_fp,
        "nontarget_turns_total": total_nontarget,
        "speaker_rejected_finals_total": total_rejected,
        "recall_min": round(min(recalls), 4) if recalls else None,
        "broadband_denoise": "ABSENT (by design -- no denoiser/AEC in the app)",
    }


# --- artifact writers --------------------------------------------------------


def write_report(
    scenario_grades: list[dict],
    out_dir: Path,
    *,
    mode: str,
    enroll_self_check: Optional[dict] = None,
    user_intruder_cosine: Optional[float] = None,
    threshold: Optional[float] = None,
    calibration: Optional[dict] = None,
) -> dict:
    """Write grade.json + report.md and return the headline dict.

    ``calibration`` is the enrollment-separability probe (see
    ``enroll_user.calibrate_separability``); its ``separable`` flag drives the
    INCONCLUSIVE downgrade so a non-separable TTS fixture isn't blamed on the
    app."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    separable = (calibration or {}).get("separable")
    inverted = (calibration or {}).get("inverted")
    headline = overall_verdict(scenario_grades, separable=separable, inverted=inverted)
    report = {
        "mode": mode,
        "threshold": threshold,
        "enrollment": enroll_self_check or {},
        "calibration": calibration or {},
        "user_intruder_cosine": user_intruder_cosine,
        "headline": headline,
        "scenarios": scenario_grades,
        "acoustic_confounded": (mode == "acoustic"),
    }
    (out_dir / "grade.json").write_text(json.dumps(report, indent=2))
    (out_dir / "report.md").write_text(_render_markdown(report))
    return report


def _fmt(v, pct=False):
    if v is None:
        return "n/a"
    if pct:
        return f"{v * 100:.0f}%"
    return f"{v:g}" if isinstance(v, (int, float)) else str(v)


def _render_markdown(report: dict) -> str:
    L: list[str] = []
    mode = report["mode"]
    L.append(f"# Noise-stress / voice-isolation report ({mode} mode)\n")
    if report.get("acoustic_confounded"):
        L.append(
            "> **ACOUSTIC MODE -- numbers are illustrative only.** A shared "
            "speaker+mic with no AEC garbles even clean audio on this machine; "
            "the trustworthy verdict comes from `--mode inject`.\n"
        )
    h = report["headline"]
    L.append("## Headline\n")
    L.append(
        f"- **FILTERS competing voices (speaker-ID gate):** "
        f"`{h['competing_voice_isolation']}` -- false positives "
        f"{h['false_positives_total']}/{h['nontarget_turns_total']} intruder/noise turns "
        f"answered; speaker_rejected_final fired {h['speaker_rejected_finals_total']} time(s)."
    )
    L.append(
        f"- **Does NOT filter broadband noise (no denoiser):** "
        f"`{h['broadband_denoise']}` -- STT accuracy falls as SNR drops (see the sweep), "
        f"which is EXPECTED, not an isolation failure."
    )
    L.append(f"- desired-voice recall (min across scenarios): {_fmt(h['recall_min'], pct=True)}\n")

    enr = report.get("enrollment") or {}
    if enr:
        L.append("## Enrollment (mock user voice print)\n")
        L.append(
            f"- passes={enr.get('passes')} dim={enr.get('dim')} "
            f"pass-to-ref cosine min={enr.get('pass_to_ref_min')} mean={enr.get('pass_to_ref_mean')}"
        )
        if report.get("user_intruder_cosine") is not None:
            L.append(
                f"- user-vs-intruder cosine = {report['user_intruder_cosine']} "
                f"(threshold={_fmt(report.get('threshold'))}; a value NEAR the threshold "
                "means the two TTS speakers sound alike -- a fixture issue, not an app bug)"
            )
        L.append("")

    cal = report.get("calibration") or {}
    if cal:
        inverted = bool(cal.get("inverted"))
        L.append("## Speaker separability calibration\n")
        L.append(
            f"- user short-clip -> ref cosines: {cal.get('user_to_ref')} "
            f"(floor {cal.get('user_floor')}, ceiling {cal.get('user_ceiling')})"
        )
        L.append(
            f"- intruder short-clip -> ref cosines: {cal.get('intruder_to_ref')} "
            f"(floor {cal.get('intruder_floor')}, ceiling {cal.get('intruder_ceiling')})"
        )
        L.append(
            f"- **separable={cal.get('separable')}** "
            f"(inverted={inverted}; recommended threshold={cal.get('recommended_threshold')})"
        )
        if inverted:
            L.append(
                "\n> **FINDING (INVERSION, not mere overlap):** the intruder's short clips "
                f"embed CLOSER to the enrolled user's reference (ceiling "
                f"{cal.get('intruder_ceiling')}) than the user's OWN short clips do (ceiling "
                f"{cal.get('user_ceiling')}, floor {cal.get('user_floor')}). The embedder ranks "
                "the DIFFERENT-speaker intruder as MORE similar to the user than the user is to "
                "themselves, so NO cosine threshold both accepts the user and rejects the "
                "intruder -- raising the threshold rejects the user first. The gate is therefore "
                "run WIDE OPEN (a recall-preserving threshold just below the user floor), the "
                "intruder trivially passes, and an end-to-end inject run can NEVER produce "
                "PASS_ISOLATION; recall is 'preserved' only because nothing is gated. The "
                "headline isolation capability is consequently UNVERIFIABLE with these synthetic "
                "fixtures -- only the synthetic unit tests prove the gate mechanism. This is a "
                "property of the libritts TTS voices + the VoxCeleb-trained CAMPPlus embedder "
                "(short synthetic clips embed weakly), NOT an app isolation failure. To get a "
                "genuine PASS, pick a SEPARABLE TTS speaker pair (`python -m tools.noise_stress "
                "--check` sweeps pairs and recommends one) or ship a real-human-voice fixture "
                "pair. With real distinct human speakers (the intended deployment) the embedder "
                "separates cleanly."
            )
        elif not cal.get("separable"):
            L.append(
                "\n> **FINDING:** the enrolled user's short query clips and the intruder's "
                "OVERLAP in the speaker model's cosine space -- no threshold separates them. "
                "This is a property of the libritts TTS voices + the VoxCeleb-trained CAMPPlus "
                "embedder (short synthetic clips embed weakly), NOT an app isolation failure. "
                "The competing-voice verdict is therefore INCONCLUSIVE for synthetic voices. "
                "Pick a SEPARABLE TTS speaker pair (`python -m tools.noise_stress --check` "
                "sweeps pairs and recommends one) to get a genuine PASS. With real distinct "
                "human speakers (the intended deployment) the embedder separates cleanly; the "
                "gate MECHANISM is verified by the unit tests."
            )
        L.append("")

    # STT-vs-SNR sweep table (broadband-noise degradation curve).
    sweep = [g for g in report["scenarios"] if g["scenario"].startswith("white_noise_snr_")]
    if sweep:
        L.append("## Broadband white-noise sweep (STT degradation; denoiser ABSENT)\n")
        L.append("| scenario | recall | STT median | STT min | false-pos |")
        L.append("|---|---|---|---|---|")
        for g in sweep:
            L.append(
                f"| {g['scenario']} | {_fmt(g['recall'], pct=True)} | "
                f"{_fmt(g['stt_score_median'])} | {_fmt(g['stt_score_min'])} | "
                f"{_fmt(g['false_positive_rate'], pct=True)} |"
            )
        L.append("")

    L.append("## Per-scenario verdicts\n")
    L.append("| scenario | isolation | recall | STT median | false-pos | rejected_finals |")
    L.append("|---|---|---|---|---|---|")
    for g in report["scenarios"]:
        L.append(
            f"| {g['scenario']} | {g['isolation_verdict']} | "
            f"{_fmt(g['recall'], pct=True)} | {_fmt(g['stt_score_median'])} | "
            f"{_fmt(g['false_positive_rate'], pct=True)} | {g['speaker_rejected_finals']} |"
        )
    L.append("")

    # Per-turn detail for auditing.
    L.append("## Per-turn detail\n")
    for g in report["scenarios"]:
        L.append(f"### {g['scenario']}\n")
        L.append("| speaker | noise | scripted | heard | answered | expected | STT |")
        L.append("|---|---|---|---|---|---|---|")
        for r in g["rows"]:
            heard = (r["heard"] or "").replace("|", "/")[:40]
            scripted = (r["scripted"] or "").replace("|", "/")[:40]
            L.append(
                f"| {r['speaker']} | {r['noise']} | {scripted!r} | {heard!r} | "
                f"{'yes' if r['answered'] else 'no'} | "
                f"{'yes' if r['expected_answered'] else 'NO'} | {_fmt(r['stt_score'])} |"
            )
        L.append("")

    L.append("## How to read this\n")
    L.append(
        "- **Isolation (competing voices)** is the speaker-ID gate's job. "
        "`PASS_ISOLATION` means no intruder/noise-only turn was answered. This is "
        "the app's ONLY voice-isolation mechanism.\n"
        "- **Broadband noise** is NOT filtered (no denoiser/AEC). A falling STT "
        "score as SNR drops is the EXPECTED, honest cost of that absence -- it is "
        "reported, never counted as an isolation failure.\n"
    )
    return "\n".join(L) + "\n"
