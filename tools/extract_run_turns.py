"""Extract per-admitted-turn clips + a manifest from one live run bundle.

A run bundle (``run-*.txt`` log + sibling ``run-*.wav`` mic recording, e.g.
under ``logs/live/<session>/``) records exactly what the production recognizer
heard and what it admitted. When a live session misrecognizes speech, the
per-turn audio windows are the replay unit every offline harness needs
(``tools.recorded_stt_eval`` / ``tools.local_gpu_stt_eval`` compare engines on
individual clips, not on a 100 s session WAV). This tool turns the bundle into
those clips:

1. Parse the run log's admitted-turn timeline: every
   ``asr final: '<text>' (raw '<RAW>')`` line is one admitted turn; every
   ``dropping raw final ...`` / ``dropping self-echo final ...`` line is a
   rejected endpoint. Both kinds reset the streaming segment, so the audio a
   turn's second pass saw is bounded by the *previous* endpoint event.
2. Anchor the WAV timeline to the wall clock. The session recorder receives
   exactly the capture-loop blocks, so WAV sample 0 corresponds to the
   ``capture loop started`` log line (the earlier "recording session audio ->"
   line only opens the file; calibration blocks are not recorded). The
   ``recorded N.Ns of session audio`` close line provides an independent
   cross-check that is reported in the manifest and warned about if it
   disagrees by more than 0.5 s.
3. Cut one clip per admitted turn: from the first ASR partial activity after
   the previous endpoint (minus ``--lead-pad``, clamped to that endpoint) to
   the ``asr final`` line time. The end already includes the endpoint rule's
   trailing silence; a stuck pre-VAD partial only widens the window back to
   the segment start, never across an endpoint.
4. Write ``<run_id>-turn-NN.wav`` (16 kHz mono 16-bit PCM) plus one
   ``<run_id>-turns.json`` manifest per bundle: window timestamps on both the
   WAV and log timelines, SHA-256 of each clip, and the admitted/raw ASR text
   the run produced -- so an engine A/B can be scored against exactly what the
   live run decided.

Privacy: clips are the owner's voice and the manifest embeds the run's
transcript text, so nothing here may ever be committable. The default output
dir is ``logs/live/run_turns/<run_id>`` (``logs/live/`` is gitignored). Any
``--out`` that resolves inside this repository is REFUSED unless git already
ignores it -- fail-closed, before anything is written -- and the tool still
drops a self-ignoring ``.gitignore`` into the output dir as belt and braces.

CLI::

    python -m tools.extract_run_turns logs/live/<session>/run-<id>.txt \
        [--wav run-<id>.wav] [--out DIR] [--lead-pad 3.0] [--tail-pad 0.0] \
        [--dry-run]

Exit code 0 on success (clips written or --dry-run), 2 on a refused output
dir, unusable bundle, or no admitted turns.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from tools.extract_voice_clips import ensure_out_dir, peak_rms, write_wav_16k_pcm
from tools.recorded_stt_eval import _sha256_file

# --- log line patterns -----------------------------------------------------

_TIME_PAT = re.compile(r"^(\d+):(\d+):(\d+)\.(\d+)\s")
_RUN_START_PAT = re.compile(r"run (\S+) started")
_CAPTURE_LOOP_PAT = re.compile(r"capture loop started")
_RECORDED_CLOSE_PAT = re.compile(r"recorded ([\d.]+)s of session audio")
# Greedy inner groups: the transcript itself may contain quotes.
_ADMITTED_FINAL_PAT = re.compile(r"asr final: '(.*)' \(raw '(.*)'\)\s*$")
_DROPPED_FINAL_PAT = re.compile(
    r"dropping (?:raw final|self-echo final)"
)
# Any streaming-partial activity marks audible input inside the current
# segment, whether or not the runtime published it.
_PARTIAL_ACTIVITY_PAT = re.compile(
    r"(?:asr partial: '"
    r"|suppressing pre-VAD ASR partial: '"
    r"|suppressing ASR partial without calibrated speech evidence: ')"
)

DEFAULT_LEAD_PAD_SEC = 3.0
DEFAULT_TAIL_PAD_SEC = 0.0
TARGET_SR = 16000
# The two wall-clock anchors (capture-loop start vs close-line arithmetic) sit
# ~60 ms apart on a healthy bundle; a larger gap means the anchor is suspect.
ANCHOR_MISMATCH_WARN_SEC = 0.5


@dataclass(frozen=True)
class AdmittedTurn:
    """One admitted final with its production segment boundaries (log time)."""

    index: int
    final_t: float
    segment_start_t: float
    onset_t: Optional[float]
    admitted_text: str
    streaming_raw: str


@dataclass(frozen=True)
class ParsedRunLog:
    run_id: str
    capture_t0: Optional[float]      # "capture loop started" wall-clock
    close_t: Optional[float]         # "recorded N.Ns of session audio" line
    recorded_sec: Optional[float]    # duration from the close line
    turns: tuple[AdmittedTurn, ...]


class BundleError(RuntimeError):
    """The bundle cannot be extracted (missing anchor/turns/WAV)."""


# --- parsing ---------------------------------------------------------------

def _line_time(line: str, last_t: Optional[float]) -> Optional[float]:
    """HH:MM:SS.mmm -> monotonic seconds; unwraps one midnight rollover."""
    m = _TIME_PAT.match(line)
    if not m:
        return None
    h, mi, s, ms = (int(g) for g in m.groups())
    t = h * 3600 + mi * 60 + s + ms / 10 ** len(m.group(4))
    if last_t is not None:
        # A run log is chronological; a large backwards jump is midnight.
        while t < last_t - 3600:
            t += 86400.0
    return t


def parse_run_log(path: str) -> ParsedRunLog:
    """Extract the admitted-turn timeline from one run log."""
    run_id = os.path.splitext(os.path.basename(path))[0]
    capture_t0: Optional[float] = None
    close_t: Optional[float] = None
    recorded_sec: Optional[float] = None

    turns: list[AdmittedTurn] = []
    last_endpoint_t: Optional[float] = None  # previous endpoint (any kind)
    pending_onset: Optional[float] = None    # first partial after that endpoint
    last_t: Optional[float] = None

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            t = _line_time(line, last_t)
            if t is None:
                continue  # continuation lines (multi-line LLM prompts etc.)
            last_t = t

            m = _RUN_START_PAT.search(line)
            if m and "|" in line and " speaker | " in line:
                run_id = f"run-{m.group(1)}"
                continue
            if capture_t0 is None and _CAPTURE_LOOP_PAT.search(line):
                capture_t0 = t
                # Segments start with the capture loop, not the log header.
                if last_endpoint_t is None:
                    last_endpoint_t = t
                continue
            m = _RECORDED_CLOSE_PAT.search(line)
            if m:
                close_t = t
                recorded_sec = float(m.group(1))
                continue
            if _PARTIAL_ACTIVITY_PAT.search(line):
                if pending_onset is None:
                    pending_onset = t
                continue
            m = _ADMITTED_FINAL_PAT.search(line)
            if m:
                segment_start = (
                    last_endpoint_t if last_endpoint_t is not None else t
                )
                turns.append(
                    AdmittedTurn(
                        index=len(turns),
                        final_t=t,
                        segment_start_t=segment_start,
                        onset_t=pending_onset,
                        admitted_text=m.group(1),
                        streaming_raw=m.group(2),
                    )
                )
                last_endpoint_t = t
                pending_onset = None
                continue
            if _DROPPED_FINAL_PAT.search(line):
                last_endpoint_t = t
                pending_onset = None
                continue

    return ParsedRunLog(
        run_id=run_id,
        capture_t0=capture_t0,
        close_t=close_t,
        recorded_sec=recorded_sec,
        turns=tuple(turns),
    )


# --- output-dir privacy guard -----------------------------------------------

def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git_check_ignore(repo_root: str, probe_path: str) -> Optional[bool]:
    """True/False = git's verdict for ``probe_path``; None = git unusable."""
    try:
        proc = subprocess.run(
            ["git", "-C", repo_root, "check-ignore", "-q", "--", probe_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return None
    if proc.returncode == 0:
        return True
    if proc.returncode == 1:
        return False
    return None  # 128: not a repo / other failure -- treat as unknown


def out_dir_refusal(
    out_dir: str,
    manifest_name: str,
    *,
    repo_root: Optional[str] = None,
    check_ignore=None,
) -> Optional[str]:
    """Reason to refuse ``out_dir``, or None when it is safe.

    Fail-closed: inside the repo, the manifest path (the least-ignorable file
    this tool writes -- ``*.wav`` is globally ignored but ``*.json`` is not)
    must already be ignored by git *before* anything is created. An unusable
    git inside the repo is a refusal, not a pass.
    """
    if check_ignore is None:
        check_ignore = _git_check_ignore  # late-bound so tests can monkeypatch
    root = os.path.realpath(repo_root if repo_root is not None else _repo_root())
    resolved = os.path.realpath(out_dir)
    try:
        inside = os.path.commonpath([root, resolved]) == root
    except ValueError:  # different drives (Windows)
        inside = False
    if not inside:
        return None
    probe = os.path.join(resolved, manifest_name)
    verdict = check_ignore(root, probe)
    if verdict is True:
        return None
    if verdict is False:
        return (
            f"refusing --out {out_dir!r}: it is inside the repository and "
            f"{probe!r} is not gitignored. Owner-voice clips and run "
            "transcripts must never be committable; use a gitignored dir "
            "(e.g. logs/live/run_turns/...) or a dir outside the repo."
        )
    return (
        f"refusing --out {out_dir!r}: it is inside the repository and git "
        "could not confirm it is ignored (fail-closed)."
    )


# --- extraction --------------------------------------------------------------

def extract_bundle(
    log_path: str,
    wav_path: Optional[str] = None,
    out_dir: Optional[str] = None,
    *,
    lead_pad_sec: float = DEFAULT_LEAD_PAD_SEC,
    tail_pad_sec: float = DEFAULT_TAIL_PAD_SEC,
    dry_run: bool = False,
    repo_root: Optional[str] = None,
) -> dict:
    """Extract every admitted turn of one bundle. Returns the manifest dict.

    Raises :class:`BundleError` when the bundle is unusable and
    :class:`SystemExit` (code 2) when the output dir is refused.
    """
    from core.engines.file_replay import load_waveform

    parsed = parse_run_log(log_path)
    if not parsed.turns:
        raise BundleError(f"{log_path}: no admitted 'asr final' turns found")

    if wav_path is None:
        wav_path = os.path.splitext(log_path)[0] + ".wav"
    if not os.path.exists(wav_path):
        raise BundleError(f"mic recording not found: {wav_path}")

    samples, sr = load_waveform(wav_path)
    if sr != TARGET_SR:
        raise BundleError(f"{wav_path}: expected {TARGET_SR} Hz, got {sr} Hz")
    wav_dur = samples.size / float(sr)

    # WAV sample 0 <-> wall clock. Primary anchor: capture-loop start (the
    # recorder is fed capture-loop blocks). Fallback: close line arithmetic.
    close_anchor = (
        parsed.close_t - parsed.recorded_sec
        if parsed.close_t is not None and parsed.recorded_sec is not None
        else None
    )
    if parsed.capture_t0 is not None:
        wav_t0 = parsed.capture_t0
        anchor = "capture_loop_started"
    elif close_anchor is not None:
        wav_t0 = close_anchor
        anchor = "recorded_close_line"
    else:
        raise BundleError(
            f"{log_path}: no WAV timeline anchor ('capture loop started' or "
            "'recorded N.Ns of session audio' line)"
        )
    anchor_delta = (
        round(close_anchor - wav_t0, 3) if close_anchor is not None else None
    )
    if anchor_delta is not None and abs(anchor_delta) > ANCHOR_MISMATCH_WARN_SEC:
        print(
            f"warning: WAV anchors disagree by {anchor_delta:+.3f}s "
            "(capture-loop vs close-line); windows may be shifted",
            file=sys.stderr,
        )

    if out_dir is None:
        out_dir = os.path.join(
            repo_root if repo_root is not None else _repo_root(),
            "logs",
            "live",
            "run_turns",
            parsed.run_id,
        )
    manifest_name = f"{parsed.run_id}-turns.json"
    refusal = out_dir_refusal(out_dir, manifest_name, repo_root=repo_root)
    if refusal is not None:
        print(f"error: {refusal}", file=sys.stderr)
        raise SystemExit(2)

    clips = []
    for turn in parsed.turns:
        onset = turn.onset_t if turn.onset_t is not None else turn.segment_start_t
        start_t = max(turn.segment_start_t, onset - lead_pad_sec)
        end_t = turn.final_t + tail_pad_sec
        start_sec = max(0.0, start_t - wav_t0)
        end_sec = min(wav_dur, max(start_sec, end_t - wav_t0))
        i0 = int(round(start_sec * sr))
        i1 = int(round(end_sec * sr))
        clip = samples[i0:i1]
        clip_id = f"turn-{turn.index:02d}"
        file_name = f"{parsed.run_id}-{clip_id}.wav"
        entry = {
            "id": clip_id,
            "file": file_name,
            "start_sec": round(start_sec, 3),
            "end_sec": round(end_sec, 3),
            "duration_sec": round(clip.size / float(sr), 3),
            "rms_peak": round(peak_rms(clip, sr), 4),
            "admitted_text": turn.admitted_text,
            "streaming_raw": turn.streaming_raw,
            "segment_start_log_sec": round(turn.segment_start_t, 3),
            "onset_log_sec": (
                round(turn.onset_t, 3) if turn.onset_t is not None else None
            ),
            "final_log_sec": round(turn.final_t, 3),
        }
        print(
            f"  {clip_id}: wav [{entry['start_sec']:.2f}..{entry['end_sec']:.2f}s]"
            f" {entry['duration_sec']:.2f}s RMS_peak={entry['rms_peak']:.4f}"
            f" admitted={turn.admitted_text!r}"
        )
        if not dry_run:
            ensure_out_dir(out_dir)
            clip_path = os.path.join(out_dir, file_name)
            write_wav_16k_pcm(clip_path, clip, sr)
            entry["sha256"] = _sha256_file(Path(clip_path))
        clips.append(entry)

    manifest = {
        "run_id": parsed.run_id,
        "source_log": os.path.abspath(log_path),
        "source_wav": os.path.abspath(wav_path),
        "source_wav_sha256": _sha256_file(Path(wav_path)),
        "sample_rate": sr,
        "wav_t0_anchor": anchor,
        "wav_t0_log_sec": round(wav_t0, 3),
        "close_anchor_delta_sec": anchor_delta,
        "lead_pad_sec": lead_pad_sec,
        "tail_pad_sec": tail_pad_sec,
        "clip_dir": os.path.abspath(out_dir),
        "clips": clips,
    }
    if not dry_run:
        manifest_path = os.path.join(out_dir, manifest_name)
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.chmod(manifest_path, 0o600)
        print(f"wrote {len(clips)} clip(s) + {manifest_name} -> {out_dir}")
    else:
        print(f"dry-run: {len(clips)} clip(s) would be written -> {out_dir}")
    return manifest


# --- CLI ---------------------------------------------------------------------

def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.extract_run_turns",
        description=(
            "Slice one run bundle (run-*.txt log + mic run-*.wav) into "
            "per-admitted-turn clips + a manifest, in a git-ignored dir."
        ),
    )
    parser.add_argument("log", help="run log, e.g. logs/live/<session>/run-<id>.txt")
    parser.add_argument(
        "--wav",
        default=None,
        help="mic recording (default: sibling <log>.wav; NOT the .ref.wav)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "output dir (default: logs/live/run_turns/<run_id>; refused if "
            "inside the repo and not gitignored)"
        ),
    )
    parser.add_argument("--lead-pad", type=float, default=DEFAULT_LEAD_PAD_SEC)
    parser.add_argument("--tail-pad", type=float, default=DEFAULT_TAIL_PAD_SEC)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the windows without writing clips or the manifest",
    )
    args = parser.parse_args(argv)
    try:
        extract_bundle(
            args.log,
            args.wav,
            args.out,
            lead_pad_sec=args.lead_pad,
            tail_pad_sec=args.tail_pad,
            dry_run=args.dry_run,
        )
    except BundleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
