"""Extract the owner's utterance clips from committed run recordings.

Part 1 ("EXTRACT") of the recorded-voice replay harness. It reads the
**committed text manifest** (``tests/fixtures/recorded_voice_manifest.json`` --
run/start/end/expected_text coordinates only, no audio) and slices the
corresponding owner-voice windows out of the already-committed session
recordings under ``logs/runs/run-*.wav``, writing them to a **local,
git-ignored** clip directory so the replay tests have real audio to drive the
pipeline with -- *without* the voice audio ever entering git.

Why this is privacy-safe:

* It writes ONLY under ``logs/fixture_audio/`` (the manifest's ``clip_dir``).
  The ``.wav`` clips there match ``.gitignore``'s ``*.wav`` rule and are NOT
  re-included (the ``!logs/runs/*.wav`` exception only covers ``logs/runs/``),
  so ``git add`` cannot stage them. As an extra belt-and-braces guard the tool
  drops a ``.gitignore`` (``*``) into the output dir on first write, so the
  whole directory -- including the text ``.json`` sidecars -- is self-ignoring.
* The committed timestamps in the manifest are treated as **ground truth**.
  The offline curation step that produced them (Silero-VAD / whisper
  segmentation) is intentionally NOT bundled here -- this shipped tool is pure
  stdlib + numpy + the existing ``core.engines.file_replay.load_waveform``, so
  extraction works on any machine that has the run WAVs and needs no models.

CLI::

    python -m tools.extract_voice_clips \
        [--manifest tests/fixtures/recorded_voice_manifest.json] \
        [--out logs/fixture_audio] \
        [--dry-run]

For each manifest entry (both ``clips`` and ``barge``):

1. Resolve ``logs/runs/<run>.wav``. If the source WAV is absent, print a skip
   line and continue -- older runs lack WAVs, and the manifest may list more
   than is locally extractable. A missing source never fails the whole run.
2. Load it via :func:`core.engines.file_replay.load_waveform` (float32 mono +
   sample rate -- reused, not re-implemented).
3. Slice ``[start_sec, end_sec]`` with the corpus's 100 ms pre/post padding
   (``max(0, start - 0.1)`` .. ``end + 0.1``); assert the source is 16 kHz.
4. Write ``<out>/<id>.wav`` as 16 kHz mono 16-bit PCM via stdlib ``wave``.
5. Write a sibling ``<out>/<id>.json`` carrying
   ``{expected_text, run, start_sec, end_sec}`` so each clip is self-describing
   even when detached from the manifest.
6. Print an RMS-peak sanity line per clip and warn (not fail) if the peak RMS
   is below the corpus LOW_CONFIDENCE floor (0.02).

Exit code 0 on success (including all-skipped); 2 only if the manifest itself
cannot be read.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import wave
from typing import Iterable, Optional

# --- constants -----------------------------------------------------------

DEFAULT_MANIFEST = "tests/fixtures/recorded_voice_manifest.json"
DEFAULT_OUT = "logs/fixture_audio"
RUNS_DIR = "logs/runs"

PAD_SEC = 0.1          # corpus 100 ms pre/post padding
TARGET_SR = 16000      # 16 kHz mono is the only supported clip rate
LOW_CONFIDENCE_RMS = 0.02  # corpus LOW_CONFIDENCE floor


# --- helpers -------------------------------------------------------------

def load_manifest(path: str) -> dict:
    """Read the committed text manifest. Raises on a missing/invalid file."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: manifest must be a JSON object")
    return data


def iter_entries(manifest: dict) -> Iterable[dict]:
    """Yield every extractable entry from ``clips`` then ``barge``.

    ``barge`` entries carry no ``expected_text`` (they are overlap windows, not
    transcribed utterances), so a placeholder is used for the sidecar.
    """
    for entry in manifest.get("clips", []) or []:
        yield dict(entry)
    for entry in manifest.get("barge", []) or []:
        e = dict(entry)
        e.setdefault("expected_text", "")  # barge windows have no transcript
        yield e


def ensure_out_dir(out_dir: str) -> None:
    """Create the output dir and drop a self-ignoring ``.gitignore`` in it.

    Belt-and-braces: even the text ``.json`` sidecars (which are privacy-safe
    -- identical coordinates to the committed manifest) cannot be staged by
    accident, and the voice ``.wav`` clips stay strictly local.
    """
    os.makedirs(out_dir, exist_ok=True)
    gi = os.path.join(out_dir, ".gitignore")
    if not os.path.exists(gi):
        with open(gi, "w", encoding="utf-8") as fh:
            fh.write(
                "# Local-only extracted owner-voice clips + sidecars.\n"
                "# Never commit voice audio; this dir is reproducible from\n"
                "# tests/fixtures/recorded_voice_manifest.json + logs/runs/*.wav.\n"
                "*\n"
            )


def peak_rms(samples, sr: int) -> float:
    """Peak RMS across 100 ms chunks (matches the corpus RMS_peak metric)."""
    import numpy as np

    if samples.size == 0:
        return 0.0
    chunk = max(1, int(round(0.1 * sr)))
    n_chunks = int(math.ceil(samples.size / chunk))
    peak = 0.0
    for i in range(n_chunks):
        seg = samples[i * chunk : (i + 1) * chunk]
        if seg.size == 0:
            continue
        rms = float(np.sqrt(np.mean(np.square(seg.astype("float64")))))
        if rms > peak:
            peak = rms
    return peak


def write_wav_16k_pcm(path: str, samples, sr: int) -> None:
    """Write ``samples`` (float32 [-1, 1]) as 16 kHz mono 16-bit PCM."""
    import numpy as np

    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


# --- extraction ----------------------------------------------------------

def extract_entry(
    entry: dict,
    out_dir: str,
    runs_dir: str,
    *,
    dry_run: bool,
    wav_cache: dict,
) -> str:
    """Extract one manifest entry.

    Returns one of ``"written"``, ``"skipped"`` (missing source / bad entry),
    or ``"dry"`` (dry-run). Prints a human-readable line for each.
    ``wav_cache`` memoizes loaded run WAVs so a run referenced by several clips
    is only decoded once.
    """
    from core.engines.file_replay import load_waveform

    clip_id = entry.get("id")
    run = entry.get("run")
    if not clip_id or not run:
        print(f"  skip: entry missing id/run -> {entry!r}")
        return "skipped"

    start = float(entry.get("start_sec", 0.0))
    end = float(entry.get("end_sec", 0.0))
    expected = entry.get("expected_text", "")

    src = os.path.join(runs_dir, f"{run}.wav")
    if not os.path.exists(src):
        print(f"  skip: {clip_id}: source WAV absent ({src})")
        return "skipped"

    if src in wav_cache:
        samples, sr = wav_cache[src]
    else:
        samples, sr = load_waveform(src)
        wav_cache[src] = (samples, sr)

    if sr != TARGET_SR:
        # The corpus guarantees all run recordings are 16 kHz mono; a mismatch
        # means something is wrong with the source, so surface it loudly rather
        # than silently resampling.
        raise AssertionError(
            f"{src}: expected {TARGET_SR} Hz, got {sr} Hz (clip {clip_id})"
        )

    pad_start = max(0.0, start - PAD_SEC)
    pad_end = end + PAD_SEC
    i0 = int(round(pad_start * sr))
    i1 = min(samples.size, int(round(pad_end * sr)))
    clip = samples[i0:i1]

    rms = peak_rms(clip, sr)
    dur = clip.size / float(sr) if sr else 0.0
    warn = "  [WARN below LOW_CONFIDENCE]" if rms < LOW_CONFIDENCE_RMS else ""
    label = f' "{expected}"' if expected else " (barge window)"
    print(
        f"  {clip_id}: {run} [{pad_start:.2f}..{pad_end:.2f}s] "
        f"{dur:.2f}s RMS_peak={rms:.4f}{warn}{label}"
    )

    if dry_run:
        return "dry"

    wav_path = os.path.join(out_dir, f"{clip_id}.wav")
    json_path = os.path.join(out_dir, f"{clip_id}.json")
    write_wav_16k_pcm(wav_path, clip, sr)
    sidecar = {
        "expected_text": expected,
        "run": run,
        "start_sec": start,
        "end_sec": end,
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(sidecar, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return "written"


def run(
    manifest_path: str,
    out_dir: str,
    *,
    runs_dir: str = RUNS_DIR,
    dry_run: bool = False,
) -> int:
    """Extract every manifest entry. Returns a process exit code."""
    try:
        manifest = load_manifest(manifest_path)
    except FileNotFoundError:
        print(f"error: manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"error: cannot read manifest {manifest_path}: {exc}", file=sys.stderr)
        return 2

    # The manifest may name its own clip dir; the --out flag overrides it.
    if out_dir is None:
        out_dir = manifest.get("clip_dir", DEFAULT_OUT)

    print(f"manifest: {manifest_path}")
    print(f"out: {out_dir}{' (dry-run)' if dry_run else ''}")

    if not dry_run:
        ensure_out_dir(out_dir)

    written = 0
    skipped = 0
    wav_cache: dict = {}
    for entry in iter_entries(manifest):
        result = extract_entry(
            entry, out_dir, runs_dir, dry_run=dry_run, wav_cache=wav_cache
        )
        if result == "written":
            written += 1
        elif result == "skipped":
            skipped += 1

    print(f"summary: {written} written, {skipped} skipped (missing source)")
    return 0


# --- CLI -----------------------------------------------------------------

def _repo_root() -> str:
    # tools/extract_voice_clips.py -> repo root is one level up from tools/.
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.extract_voice_clips",
        description=(
            "Extract owner-voice clips from logs/runs/*.wav into a local, "
            "git-ignored dir, driven by the committed text manifest."
        ),
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help=f"path to the text manifest (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "output clip dir (default: the manifest's clip_dir, "
            f"else {DEFAULT_OUT})"
        ),
    )
    parser.add_argument(
        "--runs-dir",
        default=None,
        help=f"dir holding run-*.wav recordings (default: {RUNS_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be extracted without writing any files",
    )
    args = parser.parse_args(argv)

    # Resolve relative paths against the repo root so the tool works from any
    # cwd (the agent harness resets cwd between calls).
    root = _repo_root()

    def _abs(p: Optional[str]) -> Optional[str]:
        if p is None:
            return None
        return p if os.path.isabs(p) else os.path.join(root, p)

    return run(
        _abs(args.manifest),
        _abs(args.out),
        runs_dir=_abs(args.runs_dir) or os.path.join(root, RUNS_DIR),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
