"""Autonomous voice tier: drive the REAL sherpa engine over a virtual cable.

No human, no real mic. We stand up a PipeWire null sink, launch the runtime
routed onto it (its own streams only), and inject TTS-synthesized "user"
utterances. Because the engine's capture pulls from the sink's monitor -- which
also carries the engine's own TTS -- this exercises the real-time capture
thread, the ``_audio_cb`` playback FIFO, AEC, and barge-in, and reproduces the
open-speaker self-interrupt condition (the assistant hears itself) headlessly.

A crucial subtlety: a *digital* loopback has a tiny (~40 ms) reference delay,
whereas the open speaker's acoustic delay is ~260 ms (the configured
``aec_ref_delay_ms``). Run the loopback with the configured 260 ms and the
DTLN reference is ~220 ms misaligned, so the AEC barely cancels and the
residual-path barge self-fires -- an ARTIFACT of the loopback, not the
open-speaker bug. So the tier first **calibrates**: it measures the loopback's
real delay with ``tools.aec_probe`` and temporarily aligns ``aec_ref_delay_ms``
to it, making the self-interrupt check fair. The verdict for the P1 leans on
the **delay-independent** coherence probe (``tools.replay_barge``), which is the
detector the project actually uses for open-speaker barge.

Scenarios (run with AEC aligned):

* **S1 round-trip** -- inject one utterance; expect a spoken reply.
* **S2 self-interrupt (P1)** -- inject one utterance, then NOTHING while the
  assistant replies; expect ZERO barge-ins from reply-start to reply-end.
* **S3 barge-in cut** -- once the assistant is speaking a long reply, inject a
  talk-over; expect a barge-in to fire.
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

from . import audio

_RUN_ID_RE = re.compile(r"run-(\d{8}-\d{6})")
_BEST_RE = re.compile(r"BEST:\s*delay=(\d+)ms\s*ERLE=([+\-0-9.]+)dB")


@dataclass
class VoiceRun:
    ok: bool
    run_id: Optional[str]
    summary_path: Optional[str]
    log_path: Optional[str]
    wav_path: Optional[str]
    ref_wav_path: Optional[str]
    ready: bool
    monitor_rms: float
    calibrated_delay_ms: Optional[int]
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
        self.counts = {"speaking": 0, "barge": 0, "self_echo_drop": 0}
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
        """Return once no new ``speaking:`` marker has appeared for ``quiet`` s."""
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


def _engine_args(llm_kind: str, model: str) -> list[str]:
    args = [
        sys.executable, "-m", "core",
        "--engine", "sherpa", "--llm", llm_kind,
        "--input-device", "pipewire", "--output-device", "pipewire",
        "--record", "--debug", "--stream-tts",
    ]
    if llm_kind == "ollama":
        args += ["--model", model, "--fast-model", model]
    return args


@contextlib.contextmanager
def _running_engine(args, repo_root, log_path, sink):
    """Launch the engine, pin its streams to ``sink``, wait until ready+routed."""
    proc = _Proc(args, cwd=repo_root, log_path=log_path)
    stop_mover = threading.Event()

    def _mover() -> None:
        while not stop_mover.is_set():
            with contextlib.suppress(Exception):
                audio.route_process_streams(proc.proc.pid, sink)
            stop_mover.wait(0.1)

    mt = threading.Thread(target=_mover, daemon=True)
    mt.start()
    try:
        ready = proc.ready.wait(timeout=90.0)
        routed = False
        end = time.monotonic() + 30.0
        while time.monotonic() < end:
            if audio.capture_is_routed(proc.proc.pid, sink):
                routed = True
                break
            time.sleep(0.2)
        if not (ready and routed):
            raise RuntimeError(
                f"engine not ready (ready={ready} routed={routed}); see {log_path}"
            )
        time.sleep(1.0)
        yield proc
    finally:
        stop_mover.set()
        proc.stop()


def _bundle_paths(repo_root: str, run_id: Optional[str]):
    if not run_id:
        return None, None, None
    base = os.path.join(repo_root, "logs", "runs", f"run-{run_id}")
    s = base + ".summary.json"
    w = base + ".wav"
    r = base + ".ref.wav"
    return (s if os.path.exists(s) else None,
            w if os.path.exists(w) else None,
            r if os.path.exists(r) else None)


def _measure_delay(repo_root: str, wav_path: str) -> Optional[int]:
    try:
        p = subprocess.run(
            [sys.executable, "-m", "tools.aec_probe", wav_path,
             "--backend", "dtln", "--max-delay-ms", "400"],
            cwd=repo_root, capture_output=True, text=True, timeout=240,
        )
        m = _BEST_RE.search(p.stdout)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return None


def run_voice_loop(
    *,
    repo_root: str,
    sherpa_cfg: dict,
    llm_kind: str = "ollama",
    model: str = "gemma3:4b",
    out_dir: str,
    sink_name: str = "cc_autotest_sink",
    aec_delay_ms: Optional[int] = None,   # explicit override of the AEC ref delay
    calibrate: bool = False,              # opt-in: measure loopback delay first
) -> VoiceRun:
    detail: list[str] = []
    os.makedirs(out_dir, exist_ok=True)

    # --- synthesize the injected "user" utterances ------------------------- #
    clips = {
        "cal": "testing one two three four five",
        "s1": "what is the capital of france",
        "s2": "please tell me a short story about a sailor and the sea",
        "s3": "tell me everything about the planets in the solar system",
        "barge": "excuse me wait a moment please",
    }
    paths: dict[str, str] = {}
    for key, text in clips.items():
        p = os.path.join(out_dir, f"inject_{key}.wav")
        dur = audio.synth_to_wav(text, p, sherpa_cfg=sherpa_cfg)
        paths[key] = p
        detail.append(f"synth {key}: {text!r} ({dur:.1f}s)")

    args = _engine_args(llm_kind, model)

    with audio.null_sink(sink_name) as sink:
        # --- calibration: measure the loopback's AEC reference delay -------- #
        if aec_delay_ms is None and calibrate:
            cal_log = os.path.join(out_dir, "engine_calibrate.log")
            cal_run_id = None
            try:
                with _running_engine(args, repo_root, cal_log, sink) as cproc:
                    spk0 = cproc.count("speaking")
                    audio.inject(sink, paths["cal"])
                    cproc.wait_speaking(spk0, timeout=25.0)
                    time.sleep(5.0)  # let the reply play so a ref.wav is recorded
                    cal_run_id = cproc.run_id
                cs, cw, cr = _bundle_paths(repo_root, cal_run_id)
                if cw:
                    aec_delay_ms = _measure_delay(repo_root, cw)
                detail.append(f"calibration run {cal_run_id}: loopback delay={aec_delay_ms}ms")
            except Exception as e:  # noqa: BLE001
                detail.append(f"calibration failed ({e}); using config delay")
            if aec_delay_ms is None:
                aec_delay_ms = 40  # sane loopback fallback if the probe didn't parse
                detail.append(f"calibration unparsed; falling back to {aec_delay_ms}ms")

        # --- scenarios with the AEC delay aligned to the loopback ---------- #
        scenarios: dict = {}
        monitor_rms = 0.0
        run_id = None
        scen_log = os.path.join(out_dir, "engine_stdout.log")
        with _aec_delay_override(repo_root, aec_delay_ms):
            with _running_engine(args, repo_root, scen_log, sink) as proc:
                # S1: round-trip
                spk = proc.count("speaking")
                audio.inject(sink, paths["s1"])
                spoke = proc.wait_speaking(spk, timeout=25.0)
                proc.wait_idle(timeout=30.0)
                scenarios["s1_round_trip"] = {
                    "injected": clips["s1"], "assistant_spoke": spoke,
                }
                detail.append(f"S1: spoke={spoke}")

                # S2: self-interrupt -- no user audio during the reply
                proc.wait_idle(timeout=10.0)
                spk = proc.count("speaking")
                audio.inject(sink, paths["s2"])
                proc.wait_speaking(spk, timeout=25.0)
                barge_at_start = proc.count("barge")  # excludes injection-induced barge
                proc.wait_idle(timeout=30.0)           # reply plays out, NOTHING injected
                self_barges = proc.count("barge") - barge_at_start
                scenarios["s2_self_interrupt"] = {
                    "injected": clips["s2"],
                    "barge_ins_during_own_reply": self_barges,
                    "self_echo_drops": proc.count("self_echo_drop"),
                    "live_pass": self_barges == 0,
                }
                detail.append(f"S2: self_barges={self_barges} (want 0)")

                # S3: barge-in cut -- talk over a long reply
                proc.wait_idle(timeout=10.0)
                spk = proc.count("speaking")
                audio.inject(sink, paths["s3"])
                started = proc.wait_speaking(spk, timeout=25.0)
                barge_before = proc.count("barge")
                time.sleep(1.0)
                audio.inject(sink, paths["barge"])   # the talk-over
                time.sleep(3.5)
                barge_fired = proc.count("barge") - barge_before
                scenarios["s3_barge_in"] = {
                    "injected": "long prompt + talk-over",
                    "assistant_started": started,
                    "barge_ins_after_talkover": barge_fired,
                    "pass": barge_fired >= 1,
                }
                detail.append(f"S3: started={started} barge_ins={barge_fired} (want >=1)")

                # sanity: the cable carried audio
                cap = os.path.join(out_dir, "monitor_sanity.wav")
                rec = subprocess.Popen(["pw-record", "--target", sink.monitor, cap])
                audio.inject(sink, paths["barge"])
                time.sleep(0.4)
                rec.terminate()
                with contextlib.suppress(Exception):
                    rec.wait(timeout=3)
                monitor_rms = audio.wav_rms(cap)[0]
                run_id = proc.run_id
                markers = dict(proc.counts)
                log_path = proc.log_path

    summary, wav, ref = _bundle_paths(repo_root, run_id)
    return VoiceRun(
        ok=False,  # the CLI decides ok after folding in the replay probe
        run_id=run_id, summary_path=summary, log_path=log_path,
        wav_path=wav, ref_wav_path=ref,
        ready=True, monitor_rms=monitor_rms,
        calibrated_delay_ms=aec_delay_ms,
        markers=markers, scenarios=scenarios, detail=detail,
    )
