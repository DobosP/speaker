"""Guided recording studio for autonomous-test clips.

Run ``python -m tools.autotest record --out <dir>`` and just read what's on the
screen: each line is shown big, a ``3..2..1..GO`` countdown starts, it records
while you speak and auto-stops when you finish (VAD), trims the silence, takes a
short breath, and moves to the next. At the end it writes ``manifest.json`` so
the clips drop straight into ``tools.autotest voice --utterances <dir>``.

The script below covers the app's capabilities -- plain questions, instant
commands, long prompts (so the assistant talks a while), barge-in talk-overs, a
memory fact/recall pair, and a **corrections** group: self-corrections ("five --
no, ten minutes"), repeats/stutters ("what what time is it"), and stretched
words ("Lonnndon") that test the final text fed to the LLM and whether the
system still makes sense of messy input. Filter with ``--group`` or shorten with
``--limit``; ``--review`` lets you keep/redo each take. No human/mic? ``--dry-run``
prints the script and ``--simulate`` synthesizes each line with the TTS voice
(used to self-test this tool).
"""
from __future__ import annotations

import os
import sys
import time
import wave
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Line:
    group: str
    role: str          # round_trip | speak | barge | command
    text: str          # the words you actually SAY (ground truth for WER)
    tip: str = ""      # how to say it
    tag: str = ""      # extra label (e.g. memory_fact) -- ignored by the harness
    intent: str = ""   # what it should MEAN, for judging the reply (disfluent cases)


# --- the capability-covering script ---------------------------------------- #
SCRIPT: list[Line] = [
    # plain questions: STT -> LLM -> TTS round-trip (scored for WER)
    Line("questions", "round_trip", "what time is it", "Normal, relaxed."),
    Line("questions", "round_trip", "what is the capital of france", "Normal pace."),
    Line("questions", "round_trip", "what's the weather like today"),
    Line("questions", "round_trip", "how many ounces are in a pound", "Numbers + units."),
    Line("questions", "round_trip", "what is twenty three plus nineteen", "Say the numbers clearly."),
    Line("questions", "round_trip", "who wrote romeo and juliet"),
    Line("questions", "round_trip", "could you turn the volume down a little", "A polite request."),

    # instant commands: the keyword fast-path (no LLM)
    Line("commands", "command", "stop", "Crisp, like an order."),
    Line("commands", "command", "pause", "Crisp."),
    Line("commands", "command", "cancel that", "Crisp."),
    Line("commands", "command", "louder", "Crisp."),
    Line("commands", "command", "next", "Crisp."),

    # long prompts: make the assistant talk a while (self-interrupt + barge window)
    Line("long", "speak", "please tell me a short story about a sailor and the sea", "Natural."),
    Line("long", "speak", "explain how a rainbow forms in simple words", "Natural."),
    Line("long", "speak", "give me a simple recipe for pancakes", "Natural."),

    # barge-in: how you'd cut in while it's talking
    Line("barge", "barge", "wait stop for a second", "Say it like you're INTERRUPTING."),
    Line("barge", "barge", "no that's not what i meant", "Interrupting, a bit insistent."),
    Line("barge", "barge", "hold on let me rephrase", "Interrupting."),
    Line("barge", "barge", "actually never mind", "Interrupting."),

    # memory: state a fact now, ask for it back later
    Line("memory", "round_trip", "my favorite color is teal", "State it plainly.", "memory_fact"),
    Line("memory", "round_trip", "remember that my dog's name is rex", "State it plainly.", "memory_fact"),
    Line("memory", "round_trip", "i have a meeting on friday at three", "State it plainly.", "memory_fact"),
    Line("memory", "round_trip", "what is my favorite color", "Ask it back.", "memory_recall"),
    Line("memory", "round_trip", "what is my dog's name", "Ask it back.", "memory_recall"),
    Line("memory", "round_trip", "when is my meeting", "Ask it back.", "memory_recall"),

    # harder speech: speed, homophones, longer
    Line("natural", "round_trip", "set an alarm for seven thirty in the morning", "A bit quicker."),
    Line("natural", "round_trip", "i'd like to hear the news headlines for today"),
    Line("natural", "round_trip", "how do you spell the word necessary", "Spelling request."),

    # corrections & repairs: messy, real-world speech. `text` is what you SAY
    # (scored for WER); `intent` is what it should MEAN (judge the reply). These
    # test the final text fed to the LLM and the system's ability to make sense
    # of self-corrections, repeats, and stretched words.
    Line("corrections", "round_trip", "set a timer for five no wait ten minutes",
         "Correct yourself mid-sentence (say 'five', then 'no wait ten').",
         intent="set a 10-minute timer"),
    Line("corrections", "round_trip", "what's the weather in paris i mean london",
         "Correct the city mid-sentence.", intent="weather in London"),
    Line("corrections", "round_trip", "remind me to call john no james",
         "Correct the name.", intent="remind me to call James"),
    Line("corrections", "round_trip", "what what time is it",
         "Repeat the first word, like a stumble.", intent="what time is it"),
    Line("corrections", "round_trip", "play play the jazz music",
         "Repeat the first word.", intent="play jazz music"),
    Line("corrections", "round_trip", "turn the volume down down please",
         "Repeat 'down'.", intent="lower the volume"),
    Line("corrections", "round_trip", "what is the weather in london",
         "STRETCH 'London' so it's clear: Lonnndon.", intent="weather in London"),
    Line("corrections", "round_trip", "navigate to the nearest restaurant",
         "STRETCH 'restaurant': res-tau-rant.", intent="navigate to nearest restaurant"),
    Line("corrections", "round_trip", "set an alarm for seven seven thirty a m",
         "Say 'seven', then correct to 'seven thirty'.", intent="alarm at 7:30 am"),
]


# --- audio helpers --------------------------------------------------------- #
def _save_wav(path: str, samples: np.ndarray, sr: int) -> None:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)


def _trim_silence(samples: np.ndarray, sr: int, floor: float = 0.015,
                  pad: float = 0.12) -> np.ndarray:
    """Trim leading/trailing silence by frame energy; keep ``pad`` s around."""
    if samples.size == 0:
        return samples
    win = max(1, int(sr * 0.02))
    n = len(samples) // win
    if n == 0:
        return samples
    frames = samples[: n * win].reshape(n, win)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    thresh = max(floor, 0.1 * float(rms.max()))
    voiced = np.where(rms > thresh)[0]
    if voiced.size == 0:
        return samples
    a = max(0, int(voiced[0] * win - pad * sr))
    b = min(len(samples), int((voiced[-1] + 1) * win + pad * sr))
    return samples[a:b]


def _record_vad(sr: int, device, max_s: float, silence_hang: float = 1.0,
                onset_grace: float = 6.0) -> np.ndarray:
    """Record until ~``silence_hang`` s of quiet follows speech, or ``max_s``.

    Adapts to your pace -- start talking after GO and stop when you're done."""
    import sounddevice as sd

    block = int(sr * 0.05)
    floor_blocks: list[float] = []
    chunks: list[np.ndarray] = []
    started = False
    last_voice = None
    t0 = time.monotonic()
    thresh = 0.02
    with sd.InputStream(samplerate=sr, channels=1, dtype="float32",
                        blocksize=block, device=device) as st:
        while True:
            data, _ = st.read(block)
            x = data[:, 0].copy()
            chunks.append(x)
            rms = float(np.sqrt(np.mean(x ** 2)))
            elapsed = time.monotonic() - t0
            # first ~0.3s: learn the noise floor -> adaptive threshold
            if elapsed < 0.3:
                floor_blocks.append(rms)
                if floor_blocks:
                    thresh = max(0.02, 4.0 * (sum(floor_blocks) / len(floor_blocks)))
                continue
            if rms > thresh:
                started = True
                last_voice = time.monotonic()
            if started and last_voice and (time.monotonic() - last_voice) > silence_hang:
                break
            if elapsed > max_s:
                break
            if not started and elapsed > onset_grace:
                break  # nothing said
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


# --- UX -------------------------------------------------------------------- #
def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.lower())[:32].strip("_")


def _countdown(secs: int = 3) -> None:
    for k in range(secs, 0, -1):
        sys.stdout.write(f"\r  recording in  {k} ... ")
        sys.stdout.flush()
        time.sleep(1.0)
    sys.stdout.write("\r  \033[1m\033[92m🔴 GO — read it now\033[0m            \n")
    sys.stdout.flush()
    sys.stdout.write("\a")  # terminal bell
    sys.stdout.flush()


def _banner(i: int, total: int, line: Line) -> None:
    print("\n" + "=" * 64)
    print(f"  [{i}/{total}]   group: {line.group}   ·   role: {line.role}")
    if line.tip:
        print(f"  ({line.tip})")
    print()
    print(f"      \033[1m\033[96m▶  \"{line.text}\"\033[0m")
    print()


# --- main flow ------------------------------------------------------------- #
def run_record(
    *,
    out_dir: str,
    device=None,
    samplerate: int = 16000,
    groups: Optional[list[str]] = None,
    limit: Optional[int] = None,
    review: bool = False,
    dry_run: bool = False,
    simulate: bool = False,
    sherpa_cfg: Optional[dict] = None,
    break_sec: float = 1.5,
) -> str:
    lines = [ln for ln in SCRIPT if not groups or ln.group in groups]
    if limit:
        lines = lines[:limit]
    os.makedirs(out_dir, exist_ok=True)
    total = len(lines)

    print(f"\n  Recording {total} clips into {out_dir}")
    print("  Read each line aloud after GO. It auto-stops when you finish.")
    if review:
        print("  Review on: after each take press ENTER to keep, r+ENTER to redo.")
    print()

    manifest: list[dict] = []
    for i, ln in enumerate(lines, 1):
        _banner(i, total, ln)
        fname = f"{ln.group}_{ln.role}_{i:02d}_{_slug(ln.text)}.wav"
        path = os.path.join(out_dir, fname)

        if dry_run:
            print("  (dry-run: not recording)")
            manifest.append({"file": fname, "text": ln.text, "role": ln.role,
                             "group": ln.group, "tag": ln.tag, "intent": ln.intent})
            continue

        while True:
            if simulate:
                from . import audio
                audio.synth_to_wav(ln.text, path, sherpa_cfg=sherpa_cfg or {})
                dur = audio.wav_rms(path)[2] / samplerate
                print(f"  (simulated synth clip, {dur:.1f}s)")
            else:
                _countdown(3)
                max_s = min(15.0, max(4.0, len(ln.text.split()) * 0.7 + 3.0))
                samples = _record_vad(samplerate, device, max_s)
                samples = _trim_silence(samples, samplerate)
                _save_wav(path, samples, samplerate)
                secs = len(samples) / samplerate
                print(f"  ✓ saved  {fname}   ({secs:.1f}s of speech)")
                if secs < 0.4:
                    print("  \033[93m⚠ that was very short/quiet — consider a redo\033[0m")

            if review and not simulate:
                ans = input("    keep [ENTER] / redo [r] / skip [s]: ").strip().lower()
                if ans == "r":
                    continue
                if ans == "s":
                    os.remove(path) if os.path.exists(path) else None
                    break
            manifest.append({"file": fname, "text": ln.text, "role": ln.role,
                             "group": ln.group, "tag": ln.tag, "intent": ln.intent})
            break

        if not simulate and not dry_run and i < total:
            print(f"  … next in {break_sec:.0f}s")
            time.sleep(break_sec)

    man_path = os.path.join(out_dir, "manifest.json")
    import json
    with open(man_path, "w") as f:
        json.dump({"clips": manifest}, f, indent=2)

    print("\n" + "=" * 64)
    print(f"  Done — {len(manifest)} clips + manifest.json in {out_dir}")
    print("  Try it:")
    print(f"    .venv/bin/python -m tools.autotest voice --acoustics cable   --utterances {out_dir}")
    print(f"    .venv/bin/python -m tools.autotest voice --acoustics speaker --make-sound --utterances {out_dir}")
    return man_path
