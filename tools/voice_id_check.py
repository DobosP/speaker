"""Real-voice speaker-ID check on the live mic.

Enrolls the user's REAL voice from the (clean) mic, then measures whether the
speaker-ID gate (a) ACCEPTS the user's own voice -- high self-cosine = recall --
and (b) REJECTS the assistant's TTS voice and other speakers (separability).

This is the test the synthetic ``calibrate_separability()`` cannot do. That one
embeds libritts TTS voices and came back INVERTED (the intruder scored closer to
the user's enrollment than the user's own clips) -- a CAMPPlus-vs-TTS model
artifact. The open question it leaves is whether CLEAN, REAL mic audio makes the
gate usable for the actual user. This tool answers that with the user's own voice.

Design:
  * Onset-gated capture -- waits for you to start speaking and records until you
    stop, so exact timing doesn't matter. You just read each line when ready.
  * Pins the mic to its NATIVE rate (no ALSA reconfigure -> no AT2020 self-mute).
  * Reuses core.enroll (enroll_from_recordings / average_embeddings) and the
    sherpa speaker gate; embeds at 16 kHz like the live path. Applies the same
    input_gain so the reference matches what the gate sees live.
  * Writes clips (.npy) + report under logs/voice_id/<session>/. Does NOT touch
    the live enrollment.json or config.

Phases (run in order):
    python -m tools.voice_id_check record --session S --role enroll --count 4
    python -m tools.voice_id_check record --session S --role probe  --count 4
    python -m tools.voice_id_check report  --session S
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

GATE_SR = 16000


# --- onset-gated capture -----------------------------------------------------


def capture_utterances(
    count: int,
    *,
    device,
    capture_sr: int,
    target_sr: int = GATE_SR,
    input_gain: float = 1.0,
    onset_rms: float = 0.012,
    silence_hang: float = 1.0,
    min_dur: float = 0.4,
    max_dur: float = 8.0,
    max_wait: float = 20.0,
):
    """Record ``count`` utterances, each gated by speech onset/offset.

    Returns a list of dicts: {samples (float32 @ target_sr), dur, peak, rms}.
    """
    import sounddevice as sd

    from core.audio_frontend import AudioResampler, apply_gain_soft_limit

    block = max(1, int(capture_sr * 0.05))  # 50 ms blocks
    q: list = []
    stream = sd.InputStream(
        device=device, channels=1, samplerate=capture_sr, dtype="float32",
        blocksize=block, callback=lambda d, f, t, s: q.append(d.copy().reshape(-1)),
    )
    results = []
    offset_rms = onset_rms * 0.45
    stream.start()
    try:
        for idx in range(count):
            print(f"  [{idx + 1}/{count}] listening... (speak now)", flush=True)
            pre = []                       # short pre-roll so the 1st phoneme survives
            collecting, voiced, silence = [], False, 0.0
            t_wait0 = time.monotonic()
            while True:
                if not q:
                    time.sleep(0.01)
                    if not voiced and time.monotonic() - t_wait0 > max_wait:
                        print("    (timed out waiting for speech)", flush=True)
                        break
                    continue
                blk = q.pop(0)
                r = float(np.sqrt(np.mean(blk.astype("float64") ** 2))) if blk.size else 0.0
                if not voiced:
                    pre.append(blk)
                    if len(pre) > int(0.2 / 0.05):
                        pre.pop(0)
                    if r > onset_rms:
                        voiced = True
                        collecting = list(pre)
                        collecting.append(blk)
                else:
                    collecting.append(blk)
                    dur = sum(b.size for b in collecting) / capture_sr
                    if r < offset_rms:
                        silence += 0.05
                        if silence >= silence_hang and dur >= min_dur:
                            break
                    else:
                        silence = 0.0
                    if dur >= max_dur:
                        break
            if not collecting:
                results.append(None)
                continue
            raw = np.concatenate(collecting).astype("float32")
            if input_gain != 1.0:
                raw = apply_gain_soft_limit(raw, input_gain)
            clip = (
                AudioResampler(capture_sr, target_sr).process(raw, last=True)
                if capture_sr != target_sr else raw
            )
            clip = np.asarray(clip, dtype="float32").reshape(-1)
            dur = clip.size / target_sr
            peak = float(np.abs(clip).max()) if clip.size else 0.0
            rms = float(np.sqrt(np.mean(clip.astype("float64") ** 2))) if clip.size else 0.0
            print(f"    captured {dur:.2f}s  peak={peak:.3f} rms={rms:.4f}", flush=True)
            results.append({"samples": clip, "dur": dur, "peak": peak, "rms": rms})
            q.clear()  # drop anything buffered during processing
    finally:
        stream.stop()
        stream.close()
    return results


# --- helpers -----------------------------------------------------------------


def _sherpa_cfg():
    from core.config import _load_config
    from core.engines.sherpa import SherpaConfig

    return SherpaConfig.from_dict(_load_config().get("sherpa", {}))


def _native_rate(device) -> int:
    import sounddevice as sd

    try:
        return int(sd.query_devices(device, kind="input")["default_samplerate"])
    except Exception:
        return 44100


def _session_dir(session: str) -> Path:
    d = Path("logs/voice_id") / session
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cos(a, b) -> float:
    from core.engines.speaker_gate import cosine_similarity

    return cosine_similarity(list(a), list(b))


# --- phases ------------------------------------------------------------------


def cmd_record(args) -> int:
    cfg = _sherpa_cfg()
    device = cfg.input_device or args.input_device
    from core.engines.sherpa import _norm_device

    device = _norm_device(device)
    capture_sr = int(cfg.capture_samplerate) or _native_rate(device)
    gain = float(cfg.input_gain or 1.0)
    print(f"recording role={args.role} count={args.count} device={device!r} "
          f"@ {capture_sr} Hz (gain {gain}) -> 16 kHz")
    clips = capture_utterances(
        args.count, device=device, capture_sr=capture_sr, input_gain=gain
    )
    d = _session_dir(args.session)
    saved = 0
    for i, c in enumerate(clips):
        if not c:
            print(f"  clip {i + 1}: EMPTY (no speech captured)")
            continue
        np.save(d / f"{args.role}_{i:02d}.npy", c["samples"])
        saved += 1
    print(f"saved {saved}/{args.count} {args.role} clip(s) -> {d}")
    if saved < args.count:
        print("  (some clips were empty -- re-run this phase to top up / redo)")
    return 0 if saved else 1


def _embed_clips(gate, clips, sr=GATE_SR):
    out = []
    for c in clips:
        e = gate.embed(np.asarray(c, dtype="float32"), sr)
        if e:
            out.append(list(e))
    return out


def _load_role(d: Path, role: str):
    return [np.load(p) for p in sorted(d.glob(f"{role}_*.npy"))]


def _tts_negatives(cfg, n=3):
    """Embed a few assistant-TTS lines (the voice that bleeds into the mic in
    real use). Returns 16 kHz clips or [] if TTS is unavailable."""
    try:
        from tools.live_session.synthetic_user import SyntheticUser, _resample
    except Exception:
        return []
    lines = [
        "There are seven days in a week and twelve months in a year.",
        "The weather today looks clear with a gentle breeze from the west.",
        "I can answer questions, take notes, and search the web for you.",
    ][:n]
    sid = int(getattr(cfg, "tts_speaker_id", 0) or 0)
    out = []
    try:
        u = SyntheticUser(cfg, speaker_id=sid)
        for ln in lines:
            s, sr = u.synthesize(ln)
            out.append(_resample(s, sr, GATE_SR))
    except Exception as exc:  # noqa: BLE001
        print(f"  (TTS negative skipped: {exc})")
    return out


def _human_negatives():
    """Embed other REAL human voices from tests/voice_samples (concatenated per
    speaker so each probe is a few seconds). Returns {speaker: clip16k}."""
    from collections import defaultdict

    from core.engines.file_replay import load_waveform
    from tools.live_session.synthetic_user import _resample

    by_spk = defaultdict(list)
    for p in sorted(Path("tests/voice_samples").glob("*.wav")):
        # filenames like "2_nicolas_0.wav" -> speaker "nicolas"
        parts = p.stem.split("_")
        spk = parts[1] if len(parts) >= 2 else p.stem
        by_spk[spk].append(p)
    clips = {}
    for spk, paths in by_spk.items():
        segs = []
        for p in paths[:8]:
            s, sr = load_waveform(str(p))
            segs.append(_resample(s, sr, GATE_SR))
        if segs:
            clips[spk] = np.concatenate(segs).astype("float32")
    return clips


def _stats(name, sims):
    if not sims:
        return f"  {name:22s}: (none)"
    return (f"  {name:22s}: n={len(sims)} min={min(sims):.3f} "
            f"mean={sum(sims) / len(sims):.3f} max={max(sims):.3f}")


def cmd_report(args) -> int:
    cfg = _sherpa_cfg()
    from core.enroll import enroll_from_recordings
    from core.engines.speaker_gate import sherpa_speaker_gate

    model = cfg.speaker_embedding_model
    if not model or not Path(model).exists():
        print("no speaker-embedding model on disk -- run python -m tools.setup_models")
        return 2
    gate = sherpa_speaker_gate(model, threshold=0.5, num_threads=2, provider="cpu")

    d = _session_dir(args.session)
    enroll_clips = _load_role(d, "enroll")
    probe_clips = _load_role(d, "probe")
    if not enroll_clips:
        print(f"no enroll_*.npy in {d} -- run the enroll phase first.")
        return 1
    print(f"enroll clips: {len(enroll_clips)}   probe clips: {len(probe_clips)}")

    enrollment = enroll_from_recordings(gate, enroll_clips, model_path=model, sample_rate=GATE_SR)
    if enrollment is None:
        print("enrollment failed: no usable embedding from the enroll clips.")
        return 3
    ref = enrollment.embedding

    # POSITIVE: the user's own held-out probe clips vs the reference (recall).
    user_sims = [_cos(e, ref) for e in _embed_clips(gate, probe_clips)]
    # Also enroll-pass self-consistency (how tight the reference is).
    self_sims = [_cos(e, ref) for e in _embed_clips(gate, enroll_clips)]

    # NEGATIVE: assistant TTS voice + other real humans.
    tts_sims = [_cos(e, ref) for e in _embed_clips(gate, _tts_negatives(cfg))]
    human = _human_negatives()
    human_sims = {spk: _cos(_embed_clips(gate, [clip])[0], ref)
                  for spk, clip in human.items() if _embed_clips(gate, [clip])}

    print("\n=== speaker-ID separability on REAL mic audio ===")
    print(f"  reference: {enrollment.passes} enroll pass(es), dim={enrollment.dim}")
    print(_stats("USER self (probes)", user_sims))
    print(_stats("USER enroll-consistency", self_sims))
    print(_stats("ASSISTANT TTS voice", tts_sims))
    if human_sims:
        hv = list(human_sims.values())
        print(_stats("OTHER humans", hv))
        for spk, v in sorted(human_sims.items(), key=lambda kv: kv[1]):
            print(f"      - {spk:10s}: {v:.3f}")

    # Verdict: a threshold between the user's floor and the negatives' ceiling.
    neg = list(tts_sims) + list(human_sims.values())
    user_floor = min(user_sims) if user_sims else None
    neg_ceiling = max(neg) if neg else None
    print("\n  --- verdict ---")
    if user_floor is None:
        print("  no probe clips -- record the probe phase to measure recall.")
    elif neg_ceiling is None:
        print(f"  user floor={user_floor:.3f}; no negatives embedded.")
    elif user_floor > neg_ceiling:
        thr = round((user_floor + neg_ceiling) / 2, 3)
        margin = user_floor - neg_ceiling
        print(f"  SEPARABLE on clean audio: user floor {user_floor:.3f} > "
              f"negatives ceiling {neg_ceiling:.3f} (margin {margin:.3f}).")
        print(f"  recommended threshold ~= {thr}  -> gate ACCEPTS the user, "
              f"REJECTS the assistant TTS + other voices.")
    else:
        print(f"  NOT separable: user floor {user_floor:.3f} <= negatives "
              f"ceiling {neg_ceiling:.3f}. Clean audio did NOT yield a clean "
              f"threshold (the embedder still overlaps user vs negatives).")

    report = {
        "user_self": user_sims, "enroll_consistency": self_sims,
        "tts": tts_sims, "humans": human_sims,
        "user_floor": user_floor, "neg_ceiling": neg_ceiling,
    }
    (d / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\n  saved: {d / 'report.json'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="record enroll/probe utterances")
    r.add_argument("--session", required=True)
    r.add_argument("--role", required=True, choices=["enroll", "probe"])
    r.add_argument("--count", type=int, default=4)
    r.add_argument("--input-device", default=None)
    r.set_defaults(func=cmd_record)
    rep = sub.add_parser("report", help="build the reference + measure separability")
    rep.add_argument("--session", required=True)
    rep.set_defaults(func=cmd_report)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
