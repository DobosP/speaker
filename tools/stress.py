"""Stress / soak harness for the voice runtime.

Hammers the real control plane (VoiceRuntime -> brain -> supervisor -> cancel ->
TTS queue) and the engine's playback queue under load, and reports latency
percentiles, throughput, resource drift, and PASS/FAIL on production invariants.
The logic scenarios use ScriptedEngine + EchoLLM, so they need NO models, no
audio device, and no GPU -- they exercise threading, cancellation, backpressure,
and leaks deterministically. ``real`` drives the actual sherpa ASR + Ollama over
recorded audio for latency-under-load on a fully provisioned box.

    python -m tools.stress all
    python -m tools.stress turns --n 1000
    python -m tools.stress bargein --n 500
    python -m tools.stress queue --n 5000
    python -m tools.stress soak --seconds 30
    python -m tools.stress real --replay-dir logs/runs --rounds 5
"""
from __future__ import annotations

import argparse
import gc
import statistics
import sys
import threading
import time
from typing import Callable, Iterator, Optional


# --- resource sampling (stdlib only) ---------------------------------------
def _rss_mb() -> float:
    """Resident set size in MB (Linux /proc; falls back to resource.maxrss)."""
    try:
        with open("/proc/self/status", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:  # noqa: BLE001
        return 0.0


def _pct(values, p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))
    return s[k]


def _fmt_ms(values, label: str) -> str:
    if not values:
        return f"{label}: (none)"
    return (
        f"{label}: n={len(values)} "
        f"p50={_pct(values,50)*1000:.1f}ms p95={_pct(values,95)*1000:.1f}ms "
        f"p99={_pct(values,99)*1000:.1f}ms max={max(values)*1000:.1f}ms"
    )


class _Result:
    def __init__(self, name: str):
        self.name = name
        self.ok = True
        self.notes: list[str] = []

    def check(self, cond: bool, msg: str) -> None:
        self.notes.append(("PASS " if cond else "FAIL ") + msg)
        if not cond:
            self.ok = False

    def info(self, msg: str) -> None:
        self.notes.append("      " + msg)

    def render(self) -> str:
        head = f"[{'PASS' if self.ok else 'FAIL'}] {self.name}"
        return "\n".join([head] + ["   " + n for n in self.notes])


# A canned-reply LLM with a controllable per-token delay, so we can stress the
# cancel path against a "slow" model without Ollama.
class _SlowEchoLLM:
    def __init__(self, *, reply: str = "ok.", token_delay: float = 0.0):
        self.reply = reply
        self.token_delay = token_delay

    def generate(self, prompt: str, *, system=None, images=None) -> str:
        return self.reply

    def stream(self, prompt: str, *, system=None, images=None) -> Iterator[str]:
        for tok in self.reply.split():
            if self.token_delay:
                time.sleep(self.token_delay)
            yield tok + " "


def _build_runtime(**kw):
    from core.engines.scripted import ScriptedEngine
    from core.llm import EchoLLM
    from core.runtime import VoiceRuntime

    engine = ScriptedEngine(hold_speech=kw.pop("hold_speech", False))
    llm = kw.pop("llm", None) or EchoLLM(reply="The answer is forty two.")
    rt = VoiceRuntime(engine, llm, **kw)
    rt.start(run_bus=True)
    return rt, engine


# --- scenario: sustained turn throughput -----------------------------------
def scn_turns(n: int) -> _Result:
    r = _Result(f"turns (n={n}): sustained turn throughput + leak check")
    rt, engine = _build_runtime()
    gc.collect()
    rss0, threads0 = _rss_mb(), threading.active_count()
    latencies: list[float] = []
    completed = 0
    t_start = time.perf_counter()
    try:
        for i in range(n):
            before = len(engine.spoken)
            t0 = time.perf_counter()
            engine.final(f"question number {i}")
            if rt.wait_idle(timeout=5.0):
                if len(engine.spoken) > before:
                    latencies.append(time.perf_counter() - t0)
                    completed += 1
        wall = time.perf_counter() - t_start
    finally:
        rt.stop()
    gc.collect()
    rss1, threads1 = _rss_mb(), threading.active_count()
    r.info(_fmt_ms(latencies, "round-trip final->spoken"))
    r.info(f"throughput: {completed/max(wall,1e-9):.0f} turns/s over {wall:.1f}s")
    r.info(f"RSS {rss0:.1f}->{rss1:.1f} MB (delta {rss1-rss0:+.1f})  "
           f"threads {threads0}->{threads1}")
    r.check(completed == n, f"all {n} turns produced a reply ({completed}/{n})")
    r.check(rss1 - rss0 < 50.0, f"RSS growth bounded (<50MB): {rss1-rss0:+.1f}MB")
    r.check(threads1 <= threads0, "no thread leak after stop")
    return r


# --- scenario: barge-in / cancellation storm -------------------------------
def scn_bargein(n: int) -> _Result:
    r = _Result(f"bargein (n={n}): rapid barge-in + cancel storm")
    # A slow streamer so there is always in-flight work to cancel.
    rt, engine = _build_runtime(hold_speech=True, llm=_SlowEchoLLM(
        reply="this is a fairly long winded answer that keeps going and going",
        token_delay=0.002,
    ))
    storms = 0
    hung = 0
    try:
        for i in range(n):
            engine.final(f"tell me a story {i}")
            # let the turn start producing
            deadline = time.time() + 1.0
            while not engine.is_speaking and time.time() < deadline:
                time.sleep(0.001)
            # storm: several barge-ins in quick succession
            for _ in range(4):
                engine.barge_in()
            engine.finish_speaking()
            if not rt.wait_idle(timeout=5.0):
                hung += 1
            storms += 1
        in_storm = rt._watchdog.in_storm
    finally:
        rt.stop()
    r.info(f"completed {storms}/{n} storms, {hung} hung")
    r.info(f"watchdog detected storm state: {in_storm}")
    r.check(storms == n, f"all {n} storms completed")
    r.check(hung == 0, "no turn hung after a barge-in storm")
    # After all the cancellation, the runtime must still answer a normal turn.
    rt2, eng2 = _build_runtime()
    try:
        before = len(eng2.spoken)
        eng2.final("are you still alive")
        ok = rt2.wait_idle(timeout=5.0) and len(eng2.spoken) > before
    finally:
        rt2.stop()
    r.check(ok, "runtime still responsive after the storm")
    return r


# --- scenario: playback queue backpressure ---------------------------------
def scn_queue(n: int) -> _Result:
    r = _Result(f"queue (n={n}): playback-queue backpressure (drop-oldest, maxsize=64)")
    from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine

    eng = SherpaOnnxEngine(SherpaConfig())  # not started -> no playback thread draining
    done_count = {"n": 0}

    def on_done():
        done_count["n"] += 1

    max_observed = 0
    for i in range(n):
        eng._enqueue_play(f"sentence {i}", on_done)
        max_observed = max(max_observed, eng._play_q.qsize())
    qsize = eng._play_q.qsize()
    cap = eng._play_q.maxsize
    expected_dropped = max(0, n - cap)  # first `cap` fill; the rest each drop one
    r.info(f"final qsize={qsize}, max observed={max_observed}, cap={cap}")
    r.info(f"on_done fired {done_count['n']}/{n} (expect ~{expected_dropped} dropped-"
           f"sentence callbacks; remaining {qsize} are still queued)")
    r.check(max_observed <= cap, f"queue never exceeded its bound ({max_observed}<= {cap})")
    r.check(qsize <= cap, "queue stayed bounded (no unbounded growth)")
    r.check(done_count["n"] == expected_dropped,
            f"dropped sentences' on_done fired (no orphaned callbacks): "
            f"{done_count['n']}=={expected_dropped}")
    r.check(True, "flood completed without blocking or raising")
    return r


# --- scenario: soak (time-boxed leak watch) --------------------------------
def scn_soak(seconds: float) -> _Result:
    r = _Result(f"soak ({seconds:.0f}s): continuous turns, resource drift watch")
    rt, engine = _build_runtime()
    gc.collect()
    samples: list[tuple[float, float, int]] = []
    turns = 0
    t_end = time.perf_counter() + seconds
    last_sample = 0.0
    try:
        while time.perf_counter() < t_end:
            engine.final(f"keep going {turns}")
            rt.wait_idle(timeout=5.0)
            turns += 1
            now = time.perf_counter()
            if now - last_sample >= max(1.0, seconds / 15):
                samples.append((now, _rss_mb(), threading.active_count()))
                last_sample = now
    finally:
        rt.stop()
    if samples:
        rss_start, rss_end = samples[0][1], samples[-1][1]
        thr_max = max(s[2] for s in samples)
        r.info(f"turns={turns}, RSS {rss_start:.1f}->{rss_end:.1f} MB "
               f"(drift {rss_end-rss_start:+.1f}), max threads={thr_max}")
        # Allow some growth for caches; flag runaway growth.
        r.check(rss_end - rss_start < 40.0, f"no runaway RSS growth: {rss_end-rss_start:+.1f}MB")
        r.check(thr_max < 40, f"thread count stayed sane: max {thr_max}")
    else:
        r.info("no samples captured")
    return r


# --- scenario: real models under load (optional, needs models + Ollama) ----
def scn_real(replay_dir: str, rounds: int, model: str) -> _Result:
    r = _Result(f"real (rounds={rounds}): real ASR+Ollama latency under repeated load")
    import glob
    import json
    import os

    from core.app import _apply_device_profile, _build_llms, _load_config
    from core.engines.file_replay import FileReplayEngine, load_waveform
    from core.engines.sherpa import SherpaConfig
    from core.runtime import VoiceRuntime

    paths = sorted(glob.glob(os.path.join(replay_dir, "*.npy")))
    paths += sorted(glob.glob(os.path.join(replay_dir, "*.wav")))
    if not paths:
        r.check(False, f"no .npy/.wav fixtures in {replay_dir}")
        return r
    paths = paths[:1]  # one clean fixture, replayed many times

    cfg = _apply_device_profile(_load_config(), "desktop")
    sherpa_cfg = SherpaConfig.from_dict(cfg.get("sherpa", {}))
    if not sherpa_cfg.asr_encoder:
        r.check(False, "no ASR model configured (set config.local.json)")
        return r

    class _Args:
        llm = "ollama"; model = None; fast_model = None; device = "desktop"
    args = _Args(); args.model = model
    llm, fast_llm = _build_llms(args, cfg)

    engine = FileReplayEngine(sherpa_cfg)
    rt = VoiceRuntime(engine, llm, fast_llm=fast_llm, warm_on_start=True)
    rt.start(run_bus=True)
    time.sleep(4.0)  # let warm-up finish so round 1 isn't cold
    first_audio: list[float] = []
    rss0 = _rss_mb()
    rss_curve: list[float] = []
    try:
        for rnd in range(rounds):
            for p in paths:
                samples, sr = load_waveform(p)
                rt.metrics.close_turn()
                engine.replay_samples(samples, sr)
                rt.wait_idle(timeout=30.0)
            gc.collect()
            rss_curve.append(_rss_mb())
        rt.metrics.close_turn()
        for rec in rt.metrics.records():
            fa = rec.first_audio_latency
            if fa is not None and fa >= 0:
                first_audio.append(fa)
    finally:
        rt.stop()
    rss1 = _rss_mb()
    r.info(_fmt_ms(first_audio, "first-audio latency (warm, repeated)"))
    r.info(f"RSS {rss0:.1f}->{rss1:.1f} MB (delta {rss1-rss0:+.1f}) over {rounds} rounds")
    if rss_curve:
        r.info("RSS/round: " + " ".join(f"{v:.0f}" for v in rss_curve))
    r.check(bool(first_audio), "produced measurable warm turns")
    if first_audio:
        r.check(_pct(first_audio, 95) < 3.0, f"p95 warm first-audio < 3s: {_pct(first_audio,95):.2f}s")
    # Leak vs. plateau: onnxruntime arenas + client buffers grow then flatten.
    # A true leak keeps growing in the TAIL; judge the last third of rounds.
    if len(rss_curve) >= 6:
        tail = rss_curve[len(rss_curve)*2//3:]
        tail_growth = tail[-1] - tail[0]
        r.info(f"tail growth (last {len(tail)} rounds): {tail_growth:+.1f}MB")
        r.check(tail_growth < 15.0,
                f"RSS plateaued (no leak): tail growth {tail_growth:+.1f}MB < 15MB")
    else:
        r.check(rss1 - rss0 < 200.0, f"RSS bounded across {rounds} rounds: {rss1-rss0:+.1f}MB")
    return r


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="Voice-runtime stress/soak harness")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("all")
    p = sub.add_parser("turns"); p.add_argument("--n", type=int, default=1000)
    p = sub.add_parser("bargein"); p.add_argument("--n", type=int, default=300)
    p = sub.add_parser("queue"); p.add_argument("--n", type=int, default=5000)
    p = sub.add_parser("soak"); p.add_argument("--seconds", type=float, default=30.0)
    p = sub.add_parser("real")
    p.add_argument("--replay-dir", default="tests/fixture_audio")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--model", default="minicpm5-1b:q8")
    args = ap.parse_args(argv)

    results: list[_Result] = []
    t0 = time.perf_counter()
    if args.cmd == "all":
        results.append(scn_turns(1000))
        results.append(scn_bargein(300))
        results.append(scn_queue(5000))
        results.append(scn_soak(20.0))
    elif args.cmd == "turns":
        results.append(scn_turns(args.n))
    elif args.cmd == "bargein":
        results.append(scn_bargein(args.n))
    elif args.cmd == "queue":
        results.append(scn_queue(args.n))
    elif args.cmd == "soak":
        results.append(scn_soak(args.seconds))
    elif args.cmd == "real":
        results.append(scn_real(args.replay_dir, args.rounds, args.model))

    print("\n" + "=" * 72)
    for res in results:
        print(res.render())
        print("-" * 72)
    ok = all(r.ok for r in results)
    print(f"{'ALL PASS' if ok else 'SOME FAILED'}  ({time.perf_counter()-t0:.1f}s total)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
