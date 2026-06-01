"""Real-voice turn-completion check for the Smart Turn v3 prosody model.

Does the Smart Turn ONNX (pipecat-ai/smart-turn-v3, BSD-2) actually tell a
FINISHED turn from a MID-THOUGHT one -- on YOUR real voice over the AT2020? The
synthetic-TTS harness can't answer this (the model is trained on human audio; on
TTS / read-enrollment / silence / noise it returns a flat ~0.97). So you record a
handful of natural COMPLETE turns and a handful of trailing-off INCOMPLETE turns,
and this scores them and reports whether they SEPARATE (complete high, incomplete
low) with a usable threshold. That is the go/no-go for the endpoint phase-2 lever
(a prosody detector that lets endpoint_min_silence drop to ~0.3-0.4s).

It does NOT touch core/config -- it only reads the mic + the ONNX and writes a
report under logs/turn_detect/<session>/. The inference matches the upstream
pipecat smart-turn inference.py EXACTLY (WhisperFeatureExtractor(chunk_length=8),
do_normalize=True, last-8s window padded to max_length; the ONNX output is read
as the completion probability directly, threshold 0.5).

Phases (run in order; speak each line when it says "listening"):
    python -m tools.turn_detect_check record --session s1 --role complete   --count 6
    python -m tools.turn_detect_check record --session s1 --role incomplete --count 6
    python -m tools.turn_detect_check report  --session s1
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

GATE_SR = 16000
DEFAULT_MODEL = "pretrained_models/sherpa/turn/smart-turn-v3.2-cpu.onnx"

# Suggested lines -- say these (or anything in the same spirit). The point is the
# PROSODY: a COMPLETE line lands with falling, finished intonation; an INCOMPLETE
# line trails off mid-thought as if you were about to keep going, then you stop.
PROMPTS = {
    "complete": [
        "What time is the meeting tomorrow.",
        "I had eggs and toast for breakfast.",
        "The capital of France is Paris.",
        "Please turn off the kitchen lights.",
        "I think that is everything for now.",
        "Let me know when you are ready.",
        "The weather looks nice today.",
        "I finished the report this morning.",
    ],
    "incomplete": [
        "I was thinking that maybe we could",
        "So the reason I called is because",
        "What I really want to know is whether",
        "Can you tell me the name of the",
        "I need you to remind me to",
        "The thing about it is that the",
        "Could you please look up the",
        "And then after that we should probably",
    ],
}


# --- Smart Turn inference (matches pipecat smart-turn inference.py) -----------


def _feature_extractor():
    try:
        from transformers import WhisperFeatureExtractor
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "the Smart Turn check needs `transformers` for the Whisper feature "
            f"extractor (pip install transformers): {exc}"
        ) from exc
    return WhisperFeatureExtractor(chunk_length=8, feature_size=80)


def _session(model_path: str):
    import onnxruntime as ort

    if not Path(model_path).exists():
        raise RuntimeError(
            f"Smart Turn model not found: {model_path}\n"
            "  download it (BSD-2, ~8MB):\n"
            "  curl -L -o " + model_path + " \\\n"
            "    https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/"
            "smart-turn-v3.2-cpu.onnx"
        )
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def smart_turn_probability(session, fe, samples, sample_rate: int = GATE_SR) -> float:
    """P(turn complete) for a mono float clip -- the upstream-exact path: keep the
    last 8s, Whisper log-mel (do_normalize), run the ONNX, read output[0][0]."""
    a = np.asarray(samples, dtype="float32").reshape(-1)
    if sample_rate != GATE_SR and a.size:
        idx = np.linspace(0, a.size - 1, int(round(a.size * GATE_SR / sample_rate)))
        a = np.interp(idx, np.arange(a.size), a).astype("float32")
    a = a[-8 * GATE_SR:]  # truncate_audio_to_last_n_seconds(n=8)
    feats = fe(
        a, sampling_rate=GATE_SR, return_tensors="np",
        padding="max_length", max_length=8 * GATE_SR, truncation=True, do_normalize=True,
    )["input_features"].astype("float32")  # (1, 80, 800)
    out = session.run(None, {"input_features": feats})[0]
    return float(np.asarray(out).reshape(-1)[0])


# --- phases ------------------------------------------------------------------


def _session_dir(session: str) -> Path:
    d = Path("logs/turn_detect") / session
    d.mkdir(parents=True, exist_ok=True)
    return d


def cmd_record(args) -> int:
    from core.config import _load_config
    from core.engines.sherpa import SherpaConfig, _norm_device
    from tools.voice_id_check import _native_rate, capture_utterances

    cfg = SherpaConfig.from_dict(_load_config().get("sherpa", {}))
    device = _norm_device(cfg.input_device or args.input_device)
    capture_sr = int(cfg.capture_samplerate) or _native_rate(device)
    gain = float(cfg.input_gain or 1.0)
    prompts = PROMPTS.get(args.role, [])
    print(f"recording role={args.role} count={args.count} device={device!r} "
          f"@ {capture_sr} Hz (gain {gain})")
    print(f"\nSpeak {args.role.upper()} turns. ", end="")
    if args.role == "incomplete":
        print("TRAIL OFF mid-thought (as if you'd keep going), then stop.\n")
    else:
        print("Say a FINISHED sentence with falling, done intonation.\n")
    for i in range(args.count):
        if i < len(prompts):
            print(f"  suggested line [{i + 1}]: \"{prompts[i]}\"")
    print()
    clips = capture_utterances(args.count, device=device, capture_sr=capture_sr, input_gain=gain)
    d = _session_dir(args.session)
    saved = 0
    for i, c in enumerate(clips):
        if not c:
            print(f"  clip {i + 1}: EMPTY (no speech captured)")
            continue
        np.save(d / f"{args.role}_{i:02d}.npy", c["samples"])
        saved += 1
    print(f"saved {saved}/{args.count} {args.role} clip(s) -> {d}")
    return 0 if saved else 1


def _load_role(d: Path, role: str):
    return [np.load(p) for p in sorted(d.glob(f"{role}_*.npy"))]


def classify_separability(complete: list[float], incomplete: list[float]) -> dict:
    """Pure go/no-go on whether the model separates complete vs incomplete turns.

    ``separable`` -> every complete scores above every incomplete (a clean gate);
    ``partial`` -> the MEANS separate by > 0.05 but the groups overlap;
    ``flat`` -> the means are within 0.05 (the model does not discriminate)."""
    c_min, c_mean = min(complete), statistics.fmean(complete)
    i_max, i_mean = max(incomplete), statistics.fmean(incomplete)
    margin = c_min - i_max
    if margin > 0:
        verdict = "separable"
    elif c_mean - i_mean > 0.05:
        verdict = "partial"
    else:
        verdict = "flat"
    return {
        "verdict": verdict,
        "complete_min": round(c_min, 4), "complete_mean": round(c_mean, 4),
        "incomplete_max": round(i_max, 4), "incomplete_mean": round(i_mean, 4),
        "margin": round(margin, 4),
        "threshold": round((c_min + i_max) / 2, 3) if margin > 0 else None,
    }


def cmd_report(args) -> int:
    d = _session_dir(args.session)
    complete = _load_role(d, "complete")
    incomplete = _load_role(d, "incomplete")
    if not complete or not incomplete:
        print(f"need BOTH complete_*.npy and incomplete_*.npy in {d}.\n"
              "  record both roles first (see --help).")
        return 1
    fe = _feature_extractor()
    sess = _session(args.model)

    def score_all(clips):
        return [round(smart_turn_probability(sess, fe, c), 4) for c in clips]

    comp = score_all(complete)
    inc = score_all(incomplete)
    print(f"\n=== Smart Turn v3 discrimination on YOUR real voice ===")
    print(f"  model: {args.model}")
    print(f"  COMPLETE   (want HIGH): {comp}")
    print(f"  INCOMPLETE (want LOW):  {inc}")
    v = classify_separability(comp, inc)
    print(f"\n  complete:   n={len(comp)} min={v['complete_min']:.3f} "
          f"mean={v['complete_mean']:.3f} max={max(comp):.3f}")
    print(f"  incomplete: n={len(inc)} min={min(inc):.3f} "
          f"mean={v['incomplete_mean']:.3f} max={v['incomplete_max']:.3f}")
    print("\n  --- verdict ---")
    if v["verdict"] == "separable":
        print(f"  SEPARABLE: every complete ({v['complete_min']:.3f}+) scores above "
              f"every incomplete ({v['incomplete_max']:.3f}-), margin {v['margin']:.3f}.")
        print(f"  -> the prosody model WORKS on your voice; threshold ~{v['threshold']}. "
              f"Build the detector + drop endpoint_min_silence toward ~0.3-0.4s.")
    elif v["verdict"] == "partial":
        print(f"  PARTIAL: means separate (complete {v['complete_mean']:.3f} > "
              f"incomplete {v['incomplete_mean']:.3f}) but the groups overlap "
              f"(complete floor {v['complete_min']:.3f} <= incomplete ceiling "
              f"{v['incomplete_max']:.3f}). Usable with a soft threshold / more "
              f"samples; not a clean gate yet.")
    else:
        print(f"  NOT discriminating: complete mean {v['complete_mean']:.3f} ~= "
              f"incomplete mean {v['incomplete_mean']:.3f}. The model does not "
              f"separate your complete vs incomplete turns -> the integration is "
              f"blocked (model/preproc issue, not just data). Do NOT lower the floor.")

    report = {"model": args.model, "complete": comp, "incomplete": inc, **v}
    (d / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\n  saved: {d / 'report.json'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="record complete/incomplete turns")
    r.add_argument("--session", required=True)
    r.add_argument("--role", required=True, choices=["complete", "incomplete"])
    r.add_argument("--count", type=int, default=6)
    r.add_argument("--input-device", default=None)
    r.set_defaults(func=cmd_record)
    rep = sub.add_parser("report", help="score the clips + report separability")
    rep.add_argument("--session", required=True)
    rep.add_argument("--model", default=DEFAULT_MODEL)
    rep.set_defaults(func=cmd_report)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
