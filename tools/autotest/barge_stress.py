"""Barge-in stress validation on the REAL open speaker (the P1 requirement).

CLAUDE.md hard requirement: open-speaker barge-in must work on the bare laptop
speaker -- the assistant must NOT self-interrupt on its own TTS leaking into the
mic, yet MUST cut off promptly on a real talk-over. A single voice-tier run tests
this once (S2 + S3); this harness runs MANY trials of each in one engine session
and reports rates + latencies, so the requirement can be validated with
statistical confidence rather than a single anecdote.

Topology: **shared laptop speaker** (the canonical open-speaker case). The
assistant's TTS and the injected "user" talk-overs both play out the default sink
(the laptop speaker), so the coherence/DTD detector must separate the assistant's
own echo (coherent with the playback reference -> no barge) from a real talk-over
(incoherent -> barge). No JBL / no near-far separation needed.

Two trial types, attributed by the live stdout markers the engine already emits
(`barge-in detected` = a cut fired; `barge-in REJECTED ... did not trip the gate`
= voiced speech during playback that did NOT cut):

* **self-interrupt (false-positive)** -- inject a long prompt, let the assistant
  reply with NOTHING said over it; any `barge-in detected` during its own reply is
  a FALSE self-interrupt (the failure the hard requirement forbids).
* **talk-over (true-positive)** -- inject a long prompt, wait until the assistant
  is speaking, then inject a LOUD talk-over at a varied offset; a `barge-in
  detected` within the window is a success, with latency = inject -> cut.

Run (makes real sound; assistant on the laptop speaker):
    .venv/bin/python -m tools.autotest.barge_stress \
        --utterances recordings/owner --llm ollama --model gemma3:4b \
        --n-self 5 --n-barge 8
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

from . import acoustics as acoustics_mod
from . import audio, clips as clips_mod
from .voice_loop import _bundle_paths, _engine_args, _running_engine

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class Trial:
    kind: str                       # "self_interrupt" | "talk_over"
    prompt: str
    started: bool                   # assistant actually began speaking
    fired: int                      # barge-ins observed in this trial's window
    latency_s: Optional[float] = None   # talk-over: inject -> cut (wall clock)
    delay_s: Optional[float] = None     # talk-over: how far into the reply
    barge_phrase: str = ""
    passed: bool = False


@dataclass
class StressResult:
    ok: bool
    run_id: Optional[str]
    summary_path: Optional[str]
    wav_path: Optional[str]
    trials: list = field(default_factory=list)
    fp_rate: float = 0.0            # self-interrupts per reply (want 0.0)
    tp_rate: float = 0.0            # talk-overs cut (want 1.0)
    latencies: list = field(default_factory=list)
    rejected_total: int = 0
    error: str = ""


# offsets into the reply at which the talk-over is injected (cycled), so the cut
# is exercised early, mid and late in the assistant's speech.
_BARGE_DELAYS = [0.6, 1.0, 1.6, 2.3]


def run_barge_stress(
    *, repo_root: str, sherpa_cfg: dict, llm_kind: str, model: str, out_dir: str,
    utterances_dir: Optional[str], n_self: int = 5, n_barge: int = 8,
    barge_window_s: float = 7.0, inject_sink: Optional[str] = None,
) -> StressResult:
    os.makedirs(out_dir, exist_ok=True)
    clips_by_role, _src = clips_mod.get_clips(
        os.path.join(out_dir, "clips"), sherpa_cfg, utterances_dir
    )
    # long-reply prompts (so there is a window to talk over); fall back to round_trip
    prompts = clips_by_role.get("speak") or clips_by_role.get("round_trip") or []
    barges = clips_by_role.get("barge") or []
    if not prompts or not barges:
        return StressResult(ok=False, run_id=None, summary_path=None, wav_path=None,
                            error="need 'speak'/'round_trip' prompts and 'barge' clips")

    # default: the locked rig -- user voice on the JBL, assistant on the default
    # (laptop) sink. inject_sink=None falls back to the shared default sink.
    ac = acoustics_mod.make_acoustics("speaker", inject_sink=inject_sink)
    tgt = ac.inject_target                                            # None -> default sink
    lead = getattr(ac, "inject_lead_in_ms", 0)
    args = _engine_args(llm_kind, model, ac.uses_real_device)

    trials: list[Trial] = []
    run_id = None
    with ac.session():
        log_path = os.path.join(out_dir, "engine_stdout.log")
        with _running_engine(args, repo_root, log_path, ac) as proc:
            # ---- self-interrupt (false-positive) trials -------------------- #
            for i in range(n_self):
                sp = prompts[i % len(prompts)]
                proc.wait_idle(timeout=8.0)
                spk = proc.count("speaking")
                b0 = proc.count("barge")
                audio.inject(tgt, sp.path, volume_pct=ac.inject_gain, lead_in_ms=lead)
                started = proc.wait_speaking(spk, timeout=25.0)
                proc.wait_idle(timeout=35.0)
                false_barges = proc.count("barge") - b0
                trials.append(Trial(
                    kind="self_interrupt", prompt=sp.text, started=started,
                    fired=false_barges, passed=(false_barges == 0),
                ))

            # ---- talk-over (true-positive) trials -------------------------- #
            for i in range(n_barge):
                sp = prompts[i % len(prompts)]
                bg = barges[i % len(barges)]
                delay = _BARGE_DELAYS[i % len(_BARGE_DELAYS)]
                proc.wait_idle(timeout=8.0)
                spk = proc.count("speaking")
                audio.inject(tgt, sp.path, volume_pct=ac.inject_gain, lead_in_ms=lead)
                started = proc.wait_speaking(spk, timeout=25.0)
                time.sleep(delay)
                b0 = proc.count("barge")
                t_inject = time.monotonic()
                # talk-over must out-shout the reply to clear the echo floor
                audio.inject(tgt, bg.path, volume_pct=min(400, ac.inject_gain + 100),
                             lead_in_ms=lead)
                fired = 0
                t_fired = None
                end = time.monotonic() + barge_window_s
                while time.monotonic() < end:
                    if proc.count("barge") - b0 >= 1:
                        fired = 1
                        t_fired = time.monotonic()
                        break
                    time.sleep(0.05)
                proc.wait_idle(timeout=20.0)
                trials.append(Trial(
                    kind="talk_over", prompt=sp.text, started=started, fired=fired,
                    latency_s=(round(t_fired - t_inject, 3) if t_fired else None),
                    delay_s=delay, barge_phrase=bg.text, passed=(started and fired >= 1),
                ))

            run_id = proc.run_id
            rejected_total = proc.count("barge_rejected")

    summary, wav, _ref = _bundle_paths(repo_root, run_id)
    self_trials = [t for t in trials if t.kind == "self_interrupt"]
    over_trials = [t for t in trials if t.kind == "talk_over"]
    started_overs = [t for t in over_trials if t.started]
    fp = sum(t.fired for t in self_trials)
    fp_rate = fp / max(1, len(self_trials))
    tp = sum(1 for t in started_overs if t.fired >= 1)
    tp_rate = tp / max(1, len(started_overs))
    lats = [t.latency_s for t in over_trials if t.latency_s is not None]
    return StressResult(
        ok=True, run_id=run_id, summary_path=summary, wav_path=wav,
        trials=[asdict(t) for t in trials], fp_rate=fp_rate, tp_rate=tp_rate,
        latencies=lats, rejected_total=rejected_total,
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tools.autotest.barge_stress", description=__doc__)
    ap.add_argument("--llm", choices=["echo", "ollama"], default="ollama")
    ap.add_argument("--model", default="gemma3:4b")
    ap.add_argument("--utterances", default="recordings/owner")
    ap.add_argument("--n-self", type=int, default=5, dest="n_self")
    ap.add_argument("--n-barge", type=int, default=8, dest="n_barge")
    from . import ota_setup
    ap.add_argument("--inject-sink", default=ota_setup.USER_INJECT_SINK, dest="inject_sink",
                    help="sink to play the 'user' clips into (default = the locked JBL rig); "
                         "'' / 'default' = the shared default sink")
    ap.add_argument("--no-setup", action="store_true", dest="no_setup",
                    help="do NOT apply/hold the locked OTA rig (assume it's already set)")
    args = ap.parse_args(argv)

    from core.config import load_config
    sherpa_cfg = load_config().get("sherpa", {})
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(REPO, "logs", "autotest", f"barge-stress-{stamp}")
    utt = args.utterances if os.path.isabs(args.utterances) else os.path.join(REPO, args.utterances)
    inject_sink = None if args.inject_sink in ("", "default") else args.inject_sink

    if not args.no_setup:
        info = ota_setup.apply()
        print("[barge-stress] applied the LOCKED 'real conversation' rig: "
              f"assistant->{info['assistant_sink'].split('.')[-1]}@{info['assistant_volume_pct']}%, "
              f"user->JBL@{info['user_volume_pct']}%, mic ADC {info['mic_adc_capture_pct']}% "
              f"(remember sherpa.aec_ref_delay_ms={info['aec_ref_delay_ms']})")

    pin = ota_setup.gain_pinner() if not args.no_setup else contextlib.nullcontext()
    with pin:
        r = run_barge_stress(
            repo_root=REPO, sherpa_cfg=sherpa_cfg, llm_kind=args.llm, model=args.model,
            out_dir=out_dir, utterances_dir=utt, n_self=args.n_self, n_barge=args.n_barge,
            inject_sink=inject_sink,
        )
    if r.error:
        print(f"[barge-stress] FAIL: {r.error}")
        return 1

    print(f"\n[barge-stress] run_id={r.run_id}  (assistant on the laptop speaker)")
    print("  SELF-INTERRUPT (false-positive) trials -- want 0 barges during own reply:")
    for t in (x for x in r.trials if x["kind"] == "self_interrupt"):
        flag = "ok " if t["passed"] else "FAIL"
        print(f"    [{flag}] started={t['started']} false_barges={t['fired']}  prompt={t['prompt'][:46]!r}")
    print("  TALK-OVER (true-positive) trials -- want a cut every time:")
    for t in (x for x in r.trials if x["kind"] == "talk_over"):
        flag = "ok " if t["passed"] else "FAIL"
        lat = f"{t['latency_s']:.2f}s" if t["latency_s"] is not None else "  -  "
        print(f"    [{flag}] @{t['delay_s']}s start cut={t['fired']} lat={lat}  over={t['barge_phrase'][:34]!r}")
    lat_mean = sum(r.latencies) / len(r.latencies) if r.latencies else None
    print(f"\n  self-interrupt FP rate = {r.fp_rate:.2f} per reply   (want 0.00)")
    print(f"  talk-over cut rate     = {r.tp_rate:.2f}             (want 1.00)")
    if lat_mean is not None:
        print(f"  cut latency            = mean {lat_mean:.2f}s  min {min(r.latencies):.2f}  max {max(r.latencies):.2f}  (n={len(r.latencies)})")
    print(f"  voiced-during-playback REJECTED (engine) = {r.rejected_total}")

    out = os.path.join(out_dir, "scorecard.json")
    with open(out, "w") as f:
        json.dump(asdict(r), f, indent=2, default=str)
    # verdict: the P1 requirement = no self-interrupts AND every started talk-over cut
    started_overs = sum(1 for t in r.trials if t["kind"] == "talk_over" and t["started"])
    passed = (r.fp_rate == 0.0) and (r.tp_rate >= 1.0) and started_overs > 0
    print(f"\n  scorecard: {out}")
    print(f"  VERDICT: {'PASS' if passed else 'REVIEW'} "
          f"(fp_rate={r.fp_rate:.2f}, tp_rate={r.tp_rate:.2f}, started_overs={started_overs})")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
