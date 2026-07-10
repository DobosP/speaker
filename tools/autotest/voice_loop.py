"""Autonomous voice tier: drive the REAL sherpa engine end-to-end, no human.

The engine runs for real; "user" utterances are injected on a real timeline
(synth or the owner's recordings) and the run is analyzed from its bundle. The
acoustic path is pluggable (see :mod:`.acoustics`):

* ``cable``   -- digital null-sink loopback (~tens-of-ms delay). Silent, fast.
                 The AEC reference is auto-aligned to the cable delay (~40 ms by
                 default) so the echo is cancelled, later clips aren't dropped as
                 echo, and the live self-interrupt count is meaningful. The
                 cleanest mode for STT accuracy + barge-in.
* ``delay``   -- two sinks bridged with a ~260 ms ``module-loopback`` so the AEC
                 reference aligns the way it does on a real speaker. Silent.
* ``speaker`` -- TRUE over-the-air: real default speaker + mic, clips play out
                 the speaker. Real ~260 ms acoustic delay + room/speaker
                 coloring -- the genuine open-speaker condition. Makes sound;
                 gated behind ``make_sound=True``.

Live signals from ``--debug`` stdout: ``[live] engine running`` (ready),
``speaking:`` (a reply started), ``barge-in detected``, ``dropping self-echo
final``. STT accuracy (WER) + self-interrupt analysis are folded in by the CLI
from the recorded bundle.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from . import acoustics as acoustics_mod
from . import audio, clips as clips_mod

_RUN_ID_RE = re.compile(r"run-(\d{8}-\d{6})")


@dataclass
class VoiceRun:
    ok: bool
    mode: str
    run_id: Optional[str]
    summary_path: Optional[str]
    log_path: Optional[str]
    wav_path: Optional[str]
    ref_wav_path: Optional[str]
    ready: bool
    monitor_rms: float
    clip_source: str
    injected_refs: list[str]               # ground-truth transcripts injected (for WER)
    aec_delay_ms: Optional[int]
    markers: dict = field(default_factory=dict)
    scenarios: dict = field(default_factory=dict)
    detail: list[str] = field(default_factory=list)
    error: str = ""


class _Proc:
    """Runtime subprocess + a reader thread that scans stdout for markers."""

    def __init__(self, args: list[str], cwd: str, log_path: str):
        self.log_path = log_path
        self._fh = open(log_path, "w")
        env = dict(os.environ, PYTHONUNBUFFERED="1", SPEAKER_DEBUG="1")
        self.proc = subprocess.Popen(
            args, cwd=cwd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        self.ready = threading.Event()
        self.run_id: Optional[str] = None
        self.counts = {"speaking": 0, "barge": 0, "self_echo_drop": 0, "barge_rejected": 0}
        self._lock = threading.Lock()
        self._t = threading.Thread(target=self._read, daemon=True)
        self._t.start()

    def _read(self) -> None:
        for line in self.proc.stdout:  # type: ignore[union-attr]
            self._fh.write(line)
            self._fh.flush()
            if self.run_id is None:
                m = _RUN_ID_RE.search(line)
                if m:
                    self.run_id = m.group(1)
            if "[live] engine running" in line:
                self.ready.set()
            with self._lock:
                if "speaking:" in line:
                    self.counts["speaking"] += 1
                if "barge-in detected" in line:
                    self.counts["barge"] += 1
                if "barge-in REJECTED" in line:
                    self.counts["barge_rejected"] += 1
                if "dropping self-echo final" in line:
                    self.counts["self_echo_drop"] += 1

    def count(self, key: str) -> int:
        with self._lock:
            return self.counts[key]

    def wait_speaking(self, baseline: int, timeout: float) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.count("speaking") > baseline:
                return True
            time.sleep(0.05)
        return False

    def wait_idle(self, timeout: float, quiet: float = 2.5) -> bool:
        end = time.monotonic() + timeout
        last = self.count("speaking")
        last_change = time.monotonic()
        while time.monotonic() < end:
            c = self.count("speaking")
            if c != last:
                last, last_change = c, time.monotonic()
            elif time.monotonic() - last_change >= quiet:
                return True
            time.sleep(0.1)
        return False

    def stop(self, grace: float = 20.0) -> None:
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        with contextlib.suppress(Exception):
            self._fh.close()


@contextlib.contextmanager
def _aec_delay_override(repo_root: str, delay_ms: Optional[int]):
    """Temporarily set ``sherpa.aec_ref_delay_ms`` in config.local.json,
    restoring the file verbatim afterwards. No-op when ``delay_ms`` is None."""
    path = os.path.join(repo_root, "config.local.json")
    if delay_ms is None or not os.path.exists(path):
        yield
        return
    with open(path) as f:
        orig = f.read()
    try:
        d = json.loads(orig)
        d.setdefault("sherpa", {})["aec_ref_delay_ms"] = int(delay_ms)
        with open(path, "w") as f:
            f.write(json.dumps(d, indent=2))
        yield
    finally:
        with open(path, "w") as f:
            f.write(orig)


def _engine_args(llm_kind: str, model: str, real_device: bool) -> list[str]:
    args = [sys.executable, "-m", "core", "--engine", "sherpa", "--llm", llm_kind,
            "--record", "--debug", "--stream-tts"]
    if not real_device:                       # cable/delay reach PipeWire via the bridge
        args += ["--input-device", "pipewire", "--output-device", "pipewire"]
    if llm_kind == "ollama":
        args += ["--model", model, "--fast-model", model]
    return args


@contextlib.contextmanager
def _running_engine(args, repo_root, log_path, ac):
    """Launch the engine, (optionally) pin its streams onto ``ac``, wait ready."""
    proc = _Proc(args, cwd=repo_root, log_path=log_path)
    stop_mover = threading.Event()

    def _mover() -> None:
        while not stop_mover.is_set():
            with contextlib.suppress(Exception):
                ac.route(proc.proc.pid)
            stop_mover.wait(0.1)

    mt: Optional[threading.Thread] = None
    if ac.needs_routing:
        mt = threading.Thread(target=_mover, daemon=True)
        mt.start()
    try:
        ready = proc.ready.wait(timeout=90.0)
        routed = False
        end = time.monotonic() + 30.0
        while time.monotonic() < end:
            if ac.capture_ready(proc.proc.pid):
                routed = True
                break
            time.sleep(0.2)
        if not (ready and routed):
            raise RuntimeError(
                f"engine not ready (ready={ready} routed={routed}); see {log_path}"
            )
        time.sleep(1.5)   # settle (let the capture path stabilize)
        yield proc
    finally:
        stop_mover.set()
        proc.stop()


def _bundle_paths(repo_root: str, run_id: Optional[str]):
    if not run_id:
        return None, None, None
    base = os.path.join(repo_root, "logs", "runs", f"run-{run_id}")
    s, w, r = base + ".summary.json", base + ".wav", base + ".ref.wav"
    return (s if os.path.exists(s) else None,
            w if os.path.exists(w) else None,
            r if os.path.exists(r) else None)


def run_voice_loop(
    *,
    repo_root: str,
    sherpa_cfg: dict,
    llm_kind: str = "ollama",
    model: str = "minicpm5-1b:q8",
    out_dir: str,
    acoustics_mode: str = "cable",
    latency_ms: int = 260,
    utterances_dir: Optional[str] = None,
    aec_delay_ms: Optional[int] = None,
    make_sound: bool = False,
    inject_sink: Optional[str] = None,    # speaker mode: where the 'user' clips play
) -> VoiceRun:
    detail: list[str] = []
    os.makedirs(out_dir, exist_ok=True)

    if acoustics_mode == "speaker" and not make_sound:
        return VoiceRun(
            ok=False, mode=acoustics_mode, run_id=None, summary_path=None,
            log_path=None, wav_path=None, ref_wav_path=None, ready=False,
            monitor_rms=0.0, clip_source="", injected_refs=[], aec_delay_ms=None,
            error="speaker (real over-the-air) mode needs make_sound=True (it plays "
                  "out the real speaker + records the real mic). Pass --make-sound.",
        )

    clips_by_role, clip_source = clips_mod.get_clips(
        os.path.join(out_dir, "clips"), sherpa_cfg, utterances_dir
    )
    detail.append(f"clips: {clip_source}; roles={ {k: len(v) for k, v in clips_by_role.items()} }")

    def first(role: str):
        cl = clips_by_role.get(role) or clips_by_role.get("round_trip")
        return cl[0] if cl else None

    ac = acoustics_mod.make_acoustics(acoustics_mode, latency_ms=latency_ms, inject_sink=inject_sink)
    args = _engine_args(llm_kind, model, ac.uses_real_device)
    detail.append(f"acoustics={acoustics_mode}; engine={' '.join(args)}")

    injected_refs: list[str] = []
    scenarios: dict = {}
    run_id = log_path = None
    markers: dict = {}

    with _aec_delay_override(repo_root, aec_delay_ms):
        with ac.session():
            scen_log = os.path.join(out_dir, "engine_stdout.log")
            with _running_engine(args, repo_root, scen_log, ac) as proc:
                tgt = ac.inject_target

                # S1: round-trip(s) -- drives WER. In the echo-free cable (STT)
                # mode, score EVERY non-barge clip; in the echo modes keep S1 to
                # the round_trip clips (speak/barge are used by S2/S3).
                if ac.has_echo:
                    rt_clips = clips_by_role.get("round_trip", [])
                else:
                    rt_clips = [c for role, cl in clips_by_role.items()
                                if role != "barge" for c in cl]
                lead_in = getattr(ac, "inject_lead_in_ms", 0)
                spoke_any = False
                for c in rt_clips:
                    spk = proc.count("speaking")
                    audio.inject(tgt, c.path, volume_pct=ac.inject_gain, lead_in_ms=lead_in)
                    injected_refs.append(c.text)
                    spoke = proc.wait_speaking(spk, timeout=25.0)
                    spoke_any = spoke_any or spoke
                    proc.wait_idle(timeout=30.0)
                scenarios["s1_round_trip"] = {
                    "clips": len(rt_clips), "assistant_spoke": spoke_any,
                }
                detail.append(f"S1: {len(rt_clips)} round-trip clip(s), spoke={spoke_any}")

                # the speak clips: distinct prompts for S2 and S3 so the second
                # injection isn't a same-clip repeat the engine garbles.
                speak_clips = clips_by_role.get("speak") or [first("speak")]

                # S2/S3 need the echo/talk-over relationship -- skip them in the
                # echo-free cable mode (it's the clean STT path only).
                if not ac.has_echo:
                    note = "skipped: cable has no echo (use delay/speaker)"
                    scenarios["s2_self_interrupt"] = {"note": note}
                    scenarios["s3_barge_in"] = {"note": note}
                else:
                    # S2: self-interrupt -- inject one, NOTHING during the reply
                    proc.wait_idle(timeout=10.0)
                    sp = speak_clips[0]
                    spk = proc.count("speaking")
                    audio.inject(tgt, sp.path, volume_pct=ac.inject_gain, lead_in_ms=lead_in)
                    injected_refs.append(sp.text)
                    proc.wait_speaking(spk, timeout=25.0)
                    barge_at_start = proc.count("barge")
                    proc.wait_idle(timeout=35.0)
                    self_barges = proc.count("barge") - barge_at_start
                    scenarios["s2_self_interrupt"] = {
                        "barge_ins_during_own_reply": self_barges,
                        "self_echo_drops": proc.count("self_echo_drop"),
                        "live_pass": self_barges == 0,
                    }
                    detail.append(f"S2: self_barges={self_barges} (want 0)")

                    # S3: barge-in cut -- talk over a long reply. Let a sentence
                    # get going, then a LOUD talk-over (must out-shout the reply),
                    # and poll for the cut.
                    proc.wait_idle(timeout=10.0)
                    sp = speak_clips[1] if len(speak_clips) > 1 else speak_clips[0]
                    bg = first("barge")
                    spk = proc.count("speaking")
                    audio.inject(tgt, sp.path, volume_pct=ac.inject_gain, lead_in_ms=lead_in)
                    injected_refs.append(sp.text)
                    started = proc.wait_speaking(spk, timeout=25.0)
                    time.sleep(0.8)
                    barge_before = proc.count("barge")
                    # the barge clip is NOT scored: it deliberately overlaps, so
                    # it won't transcribe cleanly. It still needs the lead-in so a
                    # Bluetooth inject sink doesn't drop the talk-over's onset.
                    audio.inject(tgt, bg.path, volume_pct=min(400, ac.inject_gain + 100),
                                 lead_in_ms=lead_in)
                    barge_fired = 0
                    end = time.monotonic() + 7.0
                    while time.monotonic() < end:
                        barge_fired = proc.count("barge") - barge_before
                        if barge_fired >= 1:
                            break
                        time.sleep(0.1)
                    scenarios["s3_barge_in"] = {
                        "assistant_started": started,
                        "barge_ins_after_talkover": barge_fired,
                        "pass": started and barge_fired >= 1,
                    }
                    detail.append(f"S3: started={started} barge_ins={barge_fired} (want >=1)")

                run_id = proc.run_id
                markers = dict(proc.counts)
                log_path = proc.log_path

    summary, wav, ref = _bundle_paths(repo_root, run_id)
    monitor_rms = audio.wav_rms(wav)[0] if wav else 0.0
    return VoiceRun(
        ok=False,  # CLI computes ok after folding in WER + the replay probe
        mode=acoustics_mode, run_id=run_id, summary_path=summary, log_path=log_path,
        wav_path=wav, ref_wav_path=ref, ready=True, monitor_rms=monitor_rms,
        clip_source=clip_source, injected_refs=[r for r in injected_refs if r],
        aec_delay_ms=aec_delay_ms, markers=markers, scenarios=scenarios, detail=detail,
    )
