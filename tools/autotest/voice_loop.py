"""Autonomous voice tier: drive the REAL sherpa engine end-to-end, no human.

The engine runs for real; "user" utterances are injected on a real timeline
(synth or the owner's recordings) and the run is analyzed from its bundle. The
acoustic path is pluggable (see :mod:`.acoustics`):

* ``cable``   -- digital injection with playback routed to a dead sink. Silent,
                 fast, and the cleanest mode for STT accuracy. It has no echo
                 relationship, so self-interrupt and barge-in are not covered.
* ``delay``   -- two sinks bridged with a ~260 ms ``module-loopback`` so the AEC
                 reference aligns the way it does on a real speaker. Silent.
* ``speaker`` -- TRUE over-the-air: real default speaker + mic, clips play out
                 the speaker. Real ~260 ms acoustic delay + room/speaker
                 coloring -- the genuine open-speaker condition. Makes sound;
                 gated behind ``make_sound=True``.

Live signals from ``--debug`` stdout: ``[live] engine running`` (ready), exact
generation-bearing final/playback onset/playback terminal markers, ``barge-in
detected``, and ``dropping self-echo final``. STT accuracy (WER) and finalized
bundle evidence are folded in by the CLI.
"""
from __future__ import annotations

import ast
import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Optional

from . import acoustics as acoustics_mod
from . import audio, clips as clips_mod
from .score import SttScore, wer

_RUN_ID_RE = re.compile(r"run-(\d{8}-\d{6})")
_FINAL_MARKER_RE = re.compile(
    r"final -> brain:\s*(?P<text>.+)\s+"
    r"\(mode=(?P<mode>\S+)\s+input_generation=(?P<generation>\d+)\)"
)
_PLAYBACK_STARTED_RE = re.compile(
    r"playback receipt started:\s+fragment=(?P<fragment>\S+)\s+"
    r"task=(?P<task>\S*)\s+input_generation=(?P<generation>\d+)"
)
_PLAYBACK_QUIESCENT_RE = re.compile(
    r"playback quiescent:\s+tracked assistant reply terminal\s+"
    r"task=(?P<task>\S*)\s+input_generation=(?P<generation>\d+)\s+"
    r"outcome=(?P<outcome>\S+)"
)
_BARGE_IN_RE = re.compile(r"(?:^|\|\s*)barge-in detected(?:\b|$)")
_PROMPT_MAX_WER = 0.50


@dataclass(frozen=True)
class RuntimeMarker:
    """One ordered runtime marker in the autonomous harness clock domain."""

    kind: str
    input_generation: Optional[int]
    sequence: int = 0
    text: str = ""
    task_id: str = ""
    fragment_id: str = ""
    outcome: str = ""


def parse_runtime_marker(line: str) -> Optional[RuntimeMarker]:
    """Parse ordered final/playback/barge evidence used by the harness.

    The final text is logged with ``%r``.  ``literal_eval`` recovers quotes and
    escapes without executing log content; malformed or generation-zero lines
    are ignored rather than becoming evidence.
    """

    match = _FINAL_MARKER_RE.search(line or "")
    if match is not None:
        try:
            text = ast.literal_eval(match.group("text"))
            generation = int(match.group("generation"))
        except (SyntaxError, ValueError):
            return None
        if not isinstance(text, str) or generation <= 0:
            return None
        return RuntimeMarker(
            kind="final",
            input_generation=generation,
            text=text,
        )

    match = _PLAYBACK_STARTED_RE.search(line or "")
    if match is not None:
        generation = int(match.group("generation"))
        if generation <= 0 or not match.group("task"):
            return None
        return RuntimeMarker(
            kind="playback_started",
            input_generation=generation,
            task_id=match.group("task"),
            fragment_id=match.group("fragment"),
        )

    match = _PLAYBACK_QUIESCENT_RE.search(line or "")
    if match is not None:
        generation = int(match.group("generation"))
        if generation <= 0 or not match.group("task"):
            return None
        return RuntimeMarker(
            kind="playback_quiescent",
            input_generation=generation,
            task_id=match.group("task"),
            outcome=match.group("outcome").lower(),
        )

    if _BARGE_IN_RE.search(line or "") is not None:
        return RuntimeMarker(kind="barge_in", input_generation=None)
    return None


@dataclass(frozen=True)
class PromptPlaybackBinding:
    """Causal proof for one injected prompt.

    Ordinary prompts require ``matching final -> exact task/generation start ->
    exact task/generation terminal`` in that order, with the scenario's
    required terminal outcome.  Commands require the same bounded-WER final but
    are excluded from playback grading: some mapped commands legitimately
    acknowledge, while others stay silent.
    """

    role: str
    reference: str
    recognized_text: str = ""
    input_generation: Optional[int] = None
    word_error_rate: Optional[float] = None
    final_sequence: Optional[int] = None
    task_id: str = ""
    playback_started_sequence: Optional[int] = None
    playback_quiescent_sequence: Optional[int] = None
    required_terminal_outcome: str = "completed"
    terminal_outcome: str = ""

    @property
    def expects_audio(self) -> bool:
        return self.role != "command"

    @property
    def recognized(self) -> bool:
        if self.final_sequence is None or self.input_generation is None:
            return False
        if not self.reference.strip():
            return True
        return bool(
            self.word_error_rate is not None
            and self.word_error_rate <= _PROMPT_MAX_WER
        )

    @property
    def audio_started(self) -> bool:
        return bool(self.task_id and self.playback_started_sequence is not None)

    @property
    def audio_quiescent(self) -> bool:
        return bool(
            self.audio_started
            and self.playback_quiescent_sequence is not None
            and self.playback_started_sequence < self.playback_quiescent_sequence
            and self.terminal_outcome == self.required_terminal_outcome
        )

    @property
    def passed(self) -> bool:
        if not self.recognized:
            return False
        if self.expects_audio:
            return self.audio_quiescent
        return True

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "reference": self.reference,
            "recognized_text": self.recognized_text,
            "input_generation": self.input_generation,
            "wer": self.word_error_rate,
            "final_sequence": self.final_sequence,
            "task_id": self.task_id,
            "playback_started_sequence": self.playback_started_sequence,
            "playback_quiescent_sequence": self.playback_quiescent_sequence,
            "required_terminal_outcome": self.required_terminal_outcome,
            "terminal_outcome": self.terminal_outcome,
            "expects_audio": self.expects_audio,
            "recognized": self.recognized,
            "audio_started": self.audio_started,
            "audio_quiescent": self.audio_quiescent,
            "passed": self.passed,
        }


def summarize_prompt_bindings(bindings: list[PromptPlaybackBinding]) -> dict:
    """Return fail-closed counts shared by the live runner and unit tests."""

    labelled = [binding for binding in bindings if binding.reference.strip()]
    audio = [binding for binding in bindings if binding.expects_audio]
    commands = [binding for binding in bindings if not binding.expects_audio]
    return {
        "expected_prompts": len(bindings),
        "recognized_prompts": sum(binding.recognized for binding in bindings),
        "expected_labelled_prompts": len(labelled),
        "recognized_labelled_prompts": sum(
            binding.recognized for binding in labelled
        ),
        "expected_audio_prompts": len(audio),
        "causal_audio_prompts": sum(binding.passed for binding in audio),
        "expected_commands": len(commands),
        "recognized_commands": sum(binding.recognized for binding in commands),
        "passed": bool(bindings) and all(binding.passed for binding in bindings),
    }


def score_prompt_bindings(bindings: list[PromptPlaybackBinding]) -> SttScore:
    """Score the exact generation-bound final selected for every prompt."""

    pairs: list[tuple[str, str, float]] = []
    matched = 0
    for binding in bindings:
        if not binding.reference.strip():
            continue
        distance = (
            float(binding.word_error_rate)
            if binding.word_error_rate is not None
            else 1.0
        )
        pairs.append((binding.reference, binding.recognized_text, distance))
        matched += int(binding.recognized)
    mean = sum(pair[2] for pair in pairs) / len(pairs) if pairs else 0.0
    return SttScore(pairs=pairs, mean_wer=round(mean, 3), n=matched)


class RuntimeMarkerLedger:
    """Thread-safe ordered marker ledger used by the subprocess reader."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._markers: list[RuntimeMarker] = []
        self._next_sequence = 0

    def observe(self, line: str) -> Optional[RuntimeMarker]:
        marker = parse_runtime_marker(line)
        if marker is None:
            return None
        with self._condition:
            self._next_sequence += 1
            marker = replace(marker, sequence=self._next_sequence)
            self._markers.append(marker)
            self._condition.notify_all()
        return marker

    def cursor(self) -> int:
        with self._condition:
            return self._next_sequence

    def snapshot(self) -> tuple[RuntimeMarker, ...]:
        with self._condition:
            return tuple(self._markers)

    def _matching_final(
        self,
        reference: str,
        *,
        after_sequence: int,
    ) -> tuple[Optional[RuntimeMarker], Optional[float]]:
        for marker in self._markers:
            if marker.sequence <= after_sequence or marker.kind != "final":
                continue
            if not reference.strip():
                return marker, None
            distance = wer(reference, marker.text)
            if distance <= _PROMPT_MAX_WER:
                return marker, distance
        return None, None

    def wait_matching_final(
        self,
        reference: str,
        *,
        after_sequence: int,
        timeout: float,
    ) -> tuple[Optional[RuntimeMarker], Optional[float]]:
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._condition:
            while True:
                marker, distance = self._matching_final(
                    reference,
                    after_sequence=after_sequence,
                )
                if marker is not None:
                    return marker, distance
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None, None
                self._condition.wait(timeout=min(0.05, remaining))

    def _first_playback(
        self,
        kind: str,
        *,
        input_generation: int,
        after_sequence: int,
        task_id: Optional[str] = None,
    ) -> Optional[RuntimeMarker]:
        for marker in self._markers:
            if (
                marker.sequence > after_sequence
                and marker.kind == kind
                and marker.input_generation == input_generation
                and (task_id is None or marker.task_id == task_id)
            ):
                return marker
        return None

    def wait_playback(
        self,
        kind: str,
        *,
        input_generation: int,
        after_sequence: int,
        timeout: float,
        task_id: Optional[str] = None,
    ) -> Optional[RuntimeMarker]:
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._condition:
            while True:
                marker = self._first_playback(
                    kind,
                    input_generation=input_generation,
                    after_sequence=after_sequence,
                    task_id=task_id,
                )
                if marker is not None:
                    return marker
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._condition.wait(timeout=min(0.05, remaining))

    def wait_barge(
        self,
        *,
        after_sequence: int,
        timeout: float,
    ) -> Optional[RuntimeMarker]:
        """Return the first concrete barge marker after a caller's baseline."""

        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._condition:
            while True:
                marker = next(
                    (
                        candidate
                        for candidate in self._markers
                        if candidate.sequence > after_sequence
                        and candidate.kind == "barge_in"
                    ),
                    None,
                )
                if marker is not None:
                    return marker
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._condition.wait(timeout=min(0.05, remaining))

    def wait_prompt_binding(
        self,
        reference: str,
        *,
        role: str,
        after_sequence: int,
        final_timeout: float = 25.0,
        start_timeout: float = 25.0,
        terminal_timeout: Optional[float] = 35.0,
        required_terminal_outcome: str = "completed",
    ) -> PromptPlaybackBinding:
        final, distance = self.wait_matching_final(
            reference,
            after_sequence=after_sequence,
            timeout=final_timeout,
        )
        if final is None:
            return PromptPlaybackBinding(
                role=role,
                reference=reference,
                required_terminal_outcome=required_terminal_outcome,
            )
        binding = PromptPlaybackBinding(
            role=role,
            reference=reference,
            recognized_text=final.text,
            input_generation=final.input_generation,
            word_error_rate=distance,
            final_sequence=final.sequence,
            required_terminal_outcome=required_terminal_outcome,
        )
        if not binding.expects_audio:
            return binding
        started = self.wait_playback(
            "playback_started",
            input_generation=final.input_generation,
            after_sequence=final.sequence,
            timeout=start_timeout,
        )
        if started is None:
            return binding
        binding = replace(
            binding,
            task_id=started.task_id,
            playback_started_sequence=started.sequence,
        )
        if terminal_timeout is None:
            return binding
        return self.wait_binding_terminal(binding, timeout=terminal_timeout)

    def wait_binding_terminal(
        self,
        binding: PromptPlaybackBinding,
        *,
        timeout: float,
        after_sequence: Optional[int] = None,
    ) -> PromptPlaybackBinding:
        if (
            binding.input_generation is None
            or binding.playback_started_sequence is None
            or not binding.task_id
        ):
            return binding
        terminal_after = binding.playback_started_sequence
        if after_sequence is not None:
            terminal_after = max(terminal_after, after_sequence)
        terminal = self.wait_playback(
            "playback_quiescent",
            input_generation=binding.input_generation,
            task_id=binding.task_id,
            after_sequence=terminal_after,
            timeout=timeout,
        )
        if terminal is None:
            return binding
        return replace(
            binding,
            playback_quiescent_sequence=terminal.sequence,
            terminal_outcome=terminal.outcome,
        )


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
    prompt_bindings: list[dict] = field(default_factory=list)
    prompt_evidence: dict = field(default_factory=dict)
    prompt_score: dict = field(default_factory=dict)
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
        self.counts = {
            "speaking": 0,
            "playback_started": 0,
            "playback_quiescent": 0,
            "barge": 0,
            "self_echo_drop": 0,
            "barge_rejected": 0,
        }
        self._lock = threading.Lock()
        self.marker_ledger = RuntimeMarkerLedger()
        self._t = threading.Thread(target=self._read, daemon=True)
        self._t.start()

    def _read(self) -> None:
        for line in self.proc.stdout:  # type: ignore[union-attr]
            self._fh.write(line)
            self._fh.flush()
            self.marker_ledger.observe(line)
            if self.run_id is None:
                m = _RUN_ID_RE.search(line)
                if m:
                    self.run_id = m.group(1)
            if "[live] engine running" in line:
                self.ready.set()
            with self._lock:
                if "speaking:" in line:
                    self.counts["speaking"] += 1
                if "playback receipt started:" in line:
                    self.counts["playback_started"] += 1
                if "playback quiescent:" in line:
                    self.counts["playback_quiescent"] += 1
                if "barge-in detected" in line:
                    self.counts["barge"] += 1
                if "barge-in REJECTED" in line:
                    self.counts["barge_rejected"] += 1
                if "dropping self-echo final" in line:
                    self.counts["self_echo_drop"] += 1

    def count(self, key: str) -> int:
        with self._lock:
            return self.counts[key]

    def marker_cursor(self) -> int:
        return self.marker_ledger.cursor()

    def wait_prompt_binding(
        self,
        reference: str,
        *,
        role: str,
        after_sequence: int,
        final_timeout: float = 25.0,
        start_timeout: float = 25.0,
        terminal_timeout: Optional[float] = 35.0,
        required_terminal_outcome: str = "completed",
    ) -> PromptPlaybackBinding:
        return self.marker_ledger.wait_prompt_binding(
            reference,
            role=role,
            after_sequence=after_sequence,
            final_timeout=final_timeout,
            start_timeout=start_timeout,
            terminal_timeout=terminal_timeout,
            required_terminal_outcome=required_terminal_outcome,
        )

    def wait_prompt_terminal(
        self,
        binding: PromptPlaybackBinding,
        *,
        timeout: float,
        after_sequence: Optional[int] = None,
    ) -> PromptPlaybackBinding:
        return self.marker_ledger.wait_binding_terminal(
            binding,
            timeout=timeout,
            after_sequence=after_sequence,
        )

    def wait_barge_marker(
        self,
        *,
        after_sequence: int,
        timeout: float = 0.0,
    ) -> Optional[RuntimeMarker]:
        return self.marker_ledger.wait_barge(
            after_sequence=after_sequence,
            timeout=timeout,
        )

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


def _engine_args(
    llm_kind: str,
    main_model: str,
    fast_model: str,
    real_device: bool,
) -> list[str]:
    args = [sys.executable, "-m", "core", "--engine", "sherpa", "--llm", llm_kind,
            "--record", "--debug", "--stream-tts"]
    if not real_device:                       # cable/delay reach PipeWire via the bridge
        args += ["--input-device", "pipewire", "--output-device", "pipewire"]
    if llm_kind == "ollama":
        args += ["--model", main_model, "--fast-model", fast_model]
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
    main_model: str = "gemma3:12b",
    fast_model: str = "minicpm5-1b:q8",
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
    args = _engine_args(
        llm_kind,
        main_model,
        fast_model,
        ac.uses_real_device,
    )
    detail.append(f"acoustics={acoustics_mode}; engine={' '.join(args)}")

    injected_refs: list[str] = []
    scenarios: dict = {}
    run_id = log_path = None
    markers: dict = {}
    prompt_bindings: list[PromptPlaybackBinding] = []

    with _aec_delay_override(repo_root, aec_delay_ms):
        with ac.session():
            scen_log = os.path.join(out_dir, "engine_stdout.log")
            with _running_engine(args, repo_root, scen_log, ac) as proc:
                tgt = ac.inject_target

                # S1: round-trip(s) -- drives WER. In the echo-free cable (STT)
                # mode, score EVERY non-barge clip; in the echo modes keep S1 to
                # the round_trip clips (speak/barge are used by S2/S3).
                if ac.has_echo:
                    rt_entries = [
                        ("round_trip", clip)
                        for clip in clips_by_role.get("round_trip", [])
                    ]
                else:
                    rt_entries = [
                        (role, clip)
                        for role, clips in clips_by_role.items()
                        if role != "barge"
                        for clip in clips
                    ]
                rt_clips = [clip for _role, clip in rt_entries]
                expected_audio_clips = sum(
                    role != "command" for role, _clip in rt_entries
                )
                lead_in = getattr(ac, "inject_lead_in_ms", 0)
                s1_bindings: list[PromptPlaybackBinding] = []
                for role, c in rt_entries:
                    marker_cursor = proc.marker_cursor()
                    audio.inject(tgt, c.path, volume_pct=ac.inject_gain, lead_in_ms=lead_in)
                    injected_refs.append(c.text)
                    binding = proc.wait_prompt_binding(
                        c.text,
                        role=role,
                        after_sequence=marker_cursor,
                    )
                    s1_bindings.append(binding)
                    prompt_bindings.append(binding)
                s1_evidence = summarize_prompt_bindings(s1_bindings)
                scenarios["s1_round_trip"] = {
                    "clips": len(rt_clips),
                    "expected_audio_clips": expected_audio_clips,
                    "assistant_spoke": s1_evidence["passed"],
                    "audio_started": sum(
                        binding.audio_started
                        for binding in s1_bindings
                        if binding.expects_audio
                    ),
                    "audio_terminal": s1_evidence["causal_audio_prompts"],
                    "evidence": s1_evidence,
                    "bindings": [binding.to_dict() for binding in s1_bindings],
                }
                detail.append(
                    f"S1: clips={len(rt_clips)} expected_audio={expected_audio_clips} "
                    f"recognized={s1_evidence['recognized_prompts']} "
                    f"causal_audio={s1_evidence['causal_audio_prompts']} "
                    f"recognized_commands={s1_evidence['recognized_commands']}"
                )

                # the speak clips: distinct prompts for S2 and S3 so the second
                # injection isn't a same-clip repeat the engine garbles.
                speak_clips = clips_by_role.get("speak") or [first("speak")]

                # S2/S3 need the echo/talk-over relationship -- skip them in the
                # echo-free cable mode (it's the clean STT path only).
                if not ac.has_echo:
                    note = "skipped: cable has no echo (use delay/speaker)"
                    scenarios["s2_self_interrupt"] = {
                        "status": "not_covered", "note": note,
                    }
                    scenarios["s3_barge_in"] = {
                        "status": "not_covered", "note": note,
                    }
                else:
                    # S2: self-interrupt -- inject one, NOTHING during the reply
                    sp = speak_clips[0]
                    marker_cursor = proc.marker_cursor()
                    # Baseline before the prompt: a self-cut can follow reply
                    # admission immediately, so sampling only after exact sink
                    # onset could subtract the event this scenario must catch.
                    barge_at_start = proc.count("barge")
                    audio.inject(tgt, sp.path, volume_pct=ac.inject_gain, lead_in_ms=lead_in)
                    injected_refs.append(sp.text)
                    self_binding = proc.wait_prompt_binding(
                        sp.text,
                        role="speak",
                        after_sequence=marker_cursor,
                    )
                    prompt_bindings.append(self_binding)
                    self_reply_started = self_binding.audio_started
                    self_reply_terminal = self_binding.audio_quiescent
                    self_barges = proc.count("barge") - barge_at_start
                    scenarios["s2_self_interrupt"] = {
                        "prompt_binding": self_binding.to_dict(),
                        "assistant_started": self_reply_started,
                        "assistant_terminal": self_reply_terminal,
                        "barge_ins_during_own_reply": self_barges,
                        "self_echo_drops": proc.count("self_echo_drop"),
                        "live_pass": (
                            self_reply_started
                            and self_reply_terminal
                            and self_barges == 0
                        ),
                        "status": (
                            "pass"
                            if self_reply_started and self_reply_terminal and self_barges == 0
                            else "fail"
                        ),
                    }
                    detail.append(
                        f"S2: started={self_reply_started} terminal={self_reply_terminal} "
                        f"self_barges={self_barges} (want start + 0)"
                    )

                    # S3: barge-in cut -- talk over a long reply. Let a sentence
                    # get going, then a LOUD talk-over (must out-shout the reply),
                    # and poll for the cut.
                    sp = speak_clips[1] if len(speak_clips) > 1 else speak_clips[0]
                    bg = first("barge")
                    marker_cursor = proc.marker_cursor()
                    audio.inject(tgt, sp.path, volume_pct=ac.inject_gain, lead_in_ms=lead_in)
                    injected_refs.append(sp.text)
                    barge_binding = proc.wait_prompt_binding(
                        sp.text,
                        role="speak",
                        after_sequence=marker_cursor,
                        terminal_timeout=None,
                        required_terminal_outcome="interrupted",
                    )
                    started = barge_binding.audio_started
                    time.sleep(0.8)
                    # the barge clip is NOT scored: it deliberately overlaps, so
                    # it won't transcribe cleanly. It still needs the lead-in so a
                    # Bluetooth inject sink doesn't drop the talk-over's onset.
                    barge_fired = 0
                    barge_marker = None
                    cut_at = None
                    injection_ended_at = None
                    cut_causal = False
                    with audio.play_injection(
                        tgt,
                        bg.path,
                        volume_pct=min(400, ac.inject_gain + 100),
                        lead_in_ms=lead_in,
                    ) as playback:
                        # Baseline only after paplay exists.  A self-cut while
                        # the injector is still being launched is not causal
                        # evidence for this talk-over.
                        barge_cursor = proc.marker_cursor()
                        end = playback.speech_onset_monotonic + 7.0
                        while time.monotonic() < end:
                            now = time.monotonic()
                            returncode = playback.process.poll()
                            if returncode is not None and injection_ended_at is None:
                                injection_ended_at = now
                            barge_marker = proc.wait_barge_marker(
                                after_sequence=barge_cursor,
                            )
                            if barge_marker is not None:
                                barge_fired = 1
                                cut_at = now
                                cut_causal = returncode is None or (
                                    returncode == 0
                                    and injection_ended_at is not None
                                    and now - injection_ended_at <= 0.15
                                )
                                break
                            time.sleep(0.05)
                    injection_ok = playback.process.returncode == 0
                    if barge_marker is not None:
                        # A natural/early terminal before the cut cannot satisfy
                        # S3.  Require the exact selected task+generation group
                        # to become terminal strictly after this concrete barge.
                        barge_binding = proc.wait_prompt_terminal(
                            barge_binding,
                            timeout=20.0,
                            after_sequence=barge_marker.sequence,
                        )
                    prompt_bindings.append(barge_binding)
                    terminal = barge_binding.audio_quiescent
                    barge_latency = (
                        round(cut_at - playback.speech_onset_monotonic, 3)
                        if cut_at is not None
                        else None
                    )
                    scenarios["s3_barge_in"] = {
                        "prompt_binding": barge_binding.to_dict(),
                        "assistant_started": started,
                        "barge_ins_after_talkover": barge_fired,
                        "barge_latency_s": barge_latency,
                        "assistant_terminal": terminal,
                        "barge_marker_sequence": (
                            barge_marker.sequence
                            if barge_marker is not None
                            else None
                        ),
                        "injection_ok": injection_ok,
                        "causal_cut": cut_causal,
                        "pass": (
                            started
                            and terminal
                            and injection_ok
                            and cut_causal
                            and barge_fired >= 1
                        ),
                        "status": "pass" if (
                            started
                            and terminal
                            and injection_ok
                            and cut_causal
                            and barge_fired >= 1
                        ) else "fail",
                    }
                    detail.append(
                        f"S3: started={started} terminal={terminal} "
                        f"injection_ok={injection_ok} causal={cut_causal} "
                        f"barge_ins={barge_fired} latency_s={barge_latency}"
                    )

                # Preserve the bounded waits above as the verdict.  A reply
                # that starts or terminates only after its deadline must remain
                # red; later markers must never turn that late work into a pass.
                final_s1 = prompt_bindings[: len(s1_bindings)]
                s1_evidence = summarize_prompt_bindings(final_s1)
                scenarios["s1_round_trip"].update(
                    assistant_spoke=s1_evidence["passed"],
                    audio_started=sum(
                        binding.audio_started
                        for binding in final_s1
                        if binding.expects_audio
                    ),
                    audio_terminal=s1_evidence["causal_audio_prompts"],
                    evidence=s1_evidence,
                    bindings=[binding.to_dict() for binding in final_s1],
                )

                run_id = proc.run_id
                markers = dict(proc.counts)
                log_path = proc.log_path

    summary, wav, ref = _bundle_paths(repo_root, run_id)
    monitor_rms = audio.wav_rms(wav)[0] if wav else 0.0
    prompt_evidence = summarize_prompt_bindings(prompt_bindings)
    prompt_score = score_prompt_bindings(prompt_bindings)
    return VoiceRun(
        ok=False,  # the CLI applies the pure WER/round-trip/coverage verdict
        mode=acoustics_mode, run_id=run_id, summary_path=summary, log_path=log_path,
        wav_path=wav, ref_wav_path=ref, ready=True, monitor_rms=monitor_rms,
        clip_source=clip_source, injected_refs=[r for r in injected_refs if r],
        aec_delay_ms=aec_delay_ms,
        prompt_bindings=[binding.to_dict() for binding in prompt_bindings],
        prompt_evidence=prompt_evidence,
        prompt_score={
            "mean_wer": prompt_score.mean_wer,
            "n": prompt_score.n,
            "pairs": [
                {"ref": reference, "hyp": hypothesis, "wer": distance}
                for reference, hypothesis, distance in prompt_score.pairs
            ],
        },
        markers=markers, scenarios=scenarios, detail=detail,
    )
