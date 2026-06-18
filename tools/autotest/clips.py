"""Where the injected 'user' utterances come from.

Two sources behind one shape (``clips_by_role: dict[str, list[Clip]]``):

* :func:`synth_clips` -- render a default script with the engine's own VITS
  voice (self-contained; good for plumbing + the silent modes).
* :func:`load_recorded` -- load the owner's real recordings from a directory +
  ``manifest.json`` (real-voice STT realism; each clip carries its ground-truth
  transcript for WER scoring).

Roles consumed by the voice tier: ``round_trip`` (ask -> expect a reply; one or
more, scored for WER), ``speak`` (a longer prompt so there's a self-interrupt
window), ``barge`` (a talk-over phrase), ``command`` (a KWS word). A manifest
may supply any subset; missing roles fall back to synth.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

from . import audio

# the default synthesized script (role -> list of (text,))
_DEFAULT_SCRIPT = {
    "round_trip": ["what is the capital of france", "name three primary colors"],
    # two distinct long-reply prompts: speak[0] for the self-interrupt window,
    # speak[1] for the barge-in window (so the 2nd isn't a garbled repeat).
    "speak": [
        "please tell me a short story about a sailor and the sea",
        "tell me everything about the planets in the solar system",
    ],
    "barge": ["excuse me wait a moment please"],
    "command": ["stop"],
}


@dataclass
class Clip:
    role: str
    path: str
    text: str        # ground-truth transcript ("" if unknown)
    source: str      # "synth" | "recorded"


def synth_clips(out_dir: str, sherpa_cfg: dict) -> dict[str, list[Clip]]:
    """Render the default script to WAVs with the engine's VITS voice."""
    os.makedirs(out_dir, exist_ok=True)
    by_role: dict[str, list[Clip]] = {}
    for role, texts in _DEFAULT_SCRIPT.items():
        for i, text in enumerate(texts):
            p = os.path.join(out_dir, f"synth_{role}_{i}.wav")
            audio.synth_to_wav(text, p, sherpa_cfg=sherpa_cfg)
            by_role.setdefault(role, []).append(Clip(role, p, text, "synth"))
    return by_role


def load_recorded(directory: str) -> dict[str, list[Clip]]:
    """Load real recordings from ``directory/manifest.json``.

    manifest.json: ``{"clips": [{"file","text","role"}, ...]}``. ``role`` defaults
    to ``round_trip``; ``text`` defaults to "" (no WER for that clip)."""
    man = os.path.join(directory, "manifest.json")
    if not os.path.exists(man):
        raise FileNotFoundError(f"no manifest.json in {directory}")
    with open(man) as f:
        data = json.load(f)
    by_role: dict[str, list[Clip]] = {}
    for entry in data.get("clips", []):
        path = os.path.join(directory, entry["file"])
        if not os.path.exists(path):
            raise FileNotFoundError(f"manifest references missing file: {path}")
        role = entry.get("role", "round_trip")
        clip = Clip(role, path, entry.get("text", ""), "recorded")
        by_role.setdefault(role, []).append(clip)
    return by_role


def get_clips(
    out_dir: str, sherpa_cfg: dict, utterances_dir: Optional[str]
) -> tuple[dict[str, list[Clip]], str]:
    """Recorded clips when ``utterances_dir`` is given (synth-filling any missing
    role), else the full synth script. Returns ``(by_role, source_label)``."""
    if not utterances_dir:
        return synth_clips(out_dir, sherpa_cfg), "synth"
    recorded = load_recorded(utterances_dir)
    # fill any role the manifest omitted with a synth clip so scenarios still run
    synth = synth_clips(out_dir, sherpa_cfg)
    for role, clips in synth.items():
        recorded.setdefault(role, clips)
    return recorded, f"recorded:{utterances_dir}"
