"""Speaker enrollment: capture the user's voice once and persist an embedding
the engine loads at startup to gate barge-in and input on speaker identity.

Why this exists (``docs/target_architecture.md`` §9.9): without speaker-ID the
assistant's own TTS leaks into the mic on a laptop-speaker setup and triggers
barge-in storms, and any nearby voice (a TV, another person, a read-aloud
quotation) is answered as if addressed to it -- see
``logs/runs/run-20260528-004726`` for both symptoms. The gate in
``core/engines/speaker_gate.py`` fixes this, but only once a reference voice is
enrolled. This module is that enrollment step:

    python -m core --enroll              # record + save the embedding
    python -m core --engine sherpa       # now gated on your voice

The embedding (not the raw audio) is persisted as small JSON next to the
models. Persisting the vector -- rather than re-extracting from a WAV every
boot -- keeps startup cheap and lets enrollment average several passes for a
more robust reference. The recorder and the gate are both injectable so the
averaging/persistence logic is unit-testable with no microphone and no model.
"""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional, Sequence

from .engines.speaker_gate import SpeakerGate, sherpa_speaker_gate

log = logging.getLogger("speaker.enroll")

# Where the enrollment vector lands when the config doesn't pin a path. Under
# pretrained_models/sherpa/ which is gitignored -- enrollment is personal and
# machine-specific, never committed.
DEFAULT_ENROLL_PATH = os.path.join("pretrained_models", "sherpa", "speaker", "enrollment.json")

# A float32 audio block recorder: takes seconds, returns mono samples in [-1, 1]
# at the enrollment sample rate. Injected so tests can avoid the microphone.
Recorder = Callable[[float], Sequence[float]]


@dataclass
class Enrollment:
    """A persisted speaker reference: the averaged embedding + provenance."""

    model: str            # absolute path of the embedding model that produced it
    embedding: list[float]
    sample_rate: int = 16000
    passes: int = 1
    created: str = ""

    @property
    def dim(self) -> int:
        return len(self.embedding)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "dim": self.dim,
            "passes": self.passes,
            "sample_rate": self.sample_rate,
            "created": self.created or datetime.now().isoformat(timespec="seconds"),
            "embedding": list(self.embedding),
        }


# --- pure embedding math -----------------------------------------------------


def l2_normalize(vec: Sequence[float]) -> list[float]:
    """Scale a vector to unit length (a zero vector is returned unchanged)."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return list(vec)
    return [x / norm for x in vec]


def average_embeddings(embeddings: Sequence[Sequence[float]]) -> list[float]:
    """Mean of L2-normalized embeddings, renormalized.

    Normalizing each pass first means a louder/longer recording doesn't
    dominate the average; renormalizing the mean keeps the reference on the
    unit sphere where cosine similarity is well-behaved. Raises ``ValueError``
    on an empty list or a dimension mismatch (a model/recording bug worth
    surfacing rather than silently averaging garbage)."""
    if not embeddings:
        raise ValueError("no embeddings to average")
    dim = len(embeddings[0])
    if dim == 0:
        raise ValueError("embeddings are empty vectors")
    acc = [0.0] * dim
    for emb in embeddings:
        if len(emb) != dim:
            raise ValueError(f"embedding dim mismatch: {len(emb)} != {dim}")
        for i, x in enumerate(l2_normalize(emb)):
            acc[i] += x
    return l2_normalize([x / len(embeddings) for x in acc])


# --- persistence -------------------------------------------------------------


def save_enrollment(path: str, enrollment: Enrollment) -> None:
    """Write the enrollment vector as JSON, creating parent dirs as needed."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(enrollment.to_dict(), fh, indent=2)


def load_enrollment(path: str) -> Enrollment:
    """Read an enrollment JSON. Raises ``ValueError`` if it has no embedding."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    embedding = data.get("embedding")
    if not isinstance(embedding, list) or not embedding:
        raise ValueError(f"{path}: no 'embedding' array")
    return Enrollment(
        model=str(data.get("model", "")),
        embedding=[float(x) for x in embedding],
        sample_rate=int(data.get("sample_rate", 16000)),
        passes=int(data.get("passes", 1)),
        created=str(data.get("created", "")),
    )


def enrollment_matches_model(enrollment: Enrollment, model_path: str) -> bool:
    """True if the enrollment was produced by ``model_path``.

    Embeddings from different models live in incomparable vector spaces, so a
    reference from model A is meaningless against model B. We compare the
    recorded model path; an empty recorded path (older/hand-written file) is
    treated as a match so we don't reject a deliberately-provided vector."""
    recorded = (enrollment.model or "").strip()
    if not recorded:
        return True
    return os.path.abspath(recorded) == os.path.abspath(model_path or "")


# --- embedding from recordings (gate injected) -------------------------------


def enroll_from_recordings(
    gate: SpeakerGate,
    recordings: Sequence[Sequence[float]],
    *,
    model_path: str,
    sample_rate: int = 16000,
) -> Optional[Enrollment]:
    """Embed each recording through ``gate`` and average into an Enrollment.

    Returns ``None`` if the model couldn't produce a usable embedding for any
    recording (so the caller can report a clear failure instead of saving an
    empty reference). Recordings that fail to embed are skipped."""
    vectors: list[list[float]] = []
    for rec in recordings:
        emb = gate.embed(rec, sample_rate)
        if emb:
            vectors.append([float(x) for x in emb])
    if not vectors:
        return None
    return Enrollment(
        model=os.path.abspath(model_path) if model_path else "",
        embedding=average_embeddings(vectors),
        sample_rate=sample_rate,
        passes=len(vectors),
        created=datetime.now().isoformat(timespec="seconds"),
    )


# --- microphone capture ------------------------------------------------------


def record_once(seconds: float, sample_rate: int = 16000, *, device=None, input_gain: float = 1.0):
    """Block for ``seconds`` and return mono float32 samples at ``sample_rate``.

    Opens the mic at ``sample_rate`` and, if the device rejects it, records at
    the device's native rate and resamples -- mirroring the engine's capture
    fallback so enrollment audio matches what the recognizer hears."""
    import numpy as np
    import sounddevice as sd

    from .audio_frontend import AudioResampler, apply_gain_soft_limit
    from .engines.sherpa import _norm_device

    dev = _norm_device(device)
    try:
        frames = int(seconds * sample_rate)
        audio = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="float32", device=dev)
        sd.wait()
        capture_sr = sample_rate
    except sd.PortAudioError:
        capture_sr = int(sd.query_devices(dev, kind="input")["default_samplerate"])
        frames = int(seconds * capture_sr)
        audio = sd.rec(frames, samplerate=capture_sr, channels=1, dtype="float32", device=dev)
        sd.wait()
    samples = np.asarray(audio, dtype="float32").reshape(-1)
    # Identical processing to the live capture path so the enrolled embedding
    # matches what the recognizer / speaker-gate hear live: gain (soft-knee
    # limiter, not a hard clip) BEFORE an anti-aliased downsample to the model
    # rate. The old path hard-clipped then linear-resampled -> a corrupted,
    # mismatched reference (the live speaker gate was rejecting the real user).
    if input_gain != 1.0:
        samples = apply_gain_soft_limit(samples, input_gain)
    if capture_sr != sample_rate:
        samples = AudioResampler(capture_sr, sample_rate).process(samples, last=True)
    return samples


# --- config persistence (machine-local overrides) ----------------------------


def _persist_local(config_path: str, updates: dict) -> None:
    """Merge ``sherpa`` path updates into the gitignored local config file.

    Same contract as ``tools.setup_models``: model/enrollment paths are
    machine-specific, so they live in ``config.local.json`` (merged over the
    committed ``config.json`` template at load), not in the template itself."""
    cfg: dict = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
    sherpa = cfg.setdefault("sherpa", {})
    for key, val in updates.items():
        if val:
            sherpa[key] = os.path.abspath(val)
    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)


# --- CLI orchestration -------------------------------------------------------


def run_enrollment(
    config: dict,
    *,
    passes: int = 3,
    seconds: float = 4.0,
    config_path: str = "config.local.json",
    recorder: Optional[Recorder] = None,
    gate: Optional[SpeakerGate] = None,
    out: Callable[[str], None] = print,
) -> int:
    """Record ``passes`` short clips, average them, and persist the embedding.

    ``recorder`` and ``gate`` are injected in tests (a synthetic recorder and a
    fake-embed gate) so this runs with no microphone and no model. In
    production both are built from the ``sherpa`` config block. Returns a shell
    exit code (0 = enrolled)."""
    sherpa = config.get("sherpa", {}) or {}
    model_path = sherpa.get("speaker_embedding_model", "")
    if not model_path:
        out(
            "No speaker-embedding model configured (sherpa.speaker_embedding_model).\n"
            "  Run once:   python -m tools.setup_models\n"
            "That downloads the speaker-ID model and writes its path into config."
        )
        return 2

    sample_rate = int(sherpa.get("sample_rate", 16000))
    if gate is None:
        gate = sherpa_speaker_gate(
            model_path,
            threshold=float(sherpa.get("speaker_threshold", 0.5)),
            num_threads=int(sherpa.get("asr_num_threads") or sherpa.get("num_threads") or 1) or 1,
            provider=str(sherpa.get("provider", "cpu")),
        )
    if recorder is None:
        def recorder(secs: float):  # type: ignore[misc]
            return record_once(
                secs,
                sample_rate,
                device=sherpa.get("input_device"),
                input_gain=float(sherpa.get("input_gain", 1.0) or 1.0),
            )

    out(f"Enrolling your voice: {passes} clip(s) of ~{seconds:.0f}s each.")
    recordings = []
    for i in range(max(1, passes)):
        out(f"  [{i + 1}/{passes}] Speak naturally now...")
        recordings.append(recorder(seconds))
        out("  ...captured.")

    enrollment = enroll_from_recordings(
        gate, recordings, model_path=model_path, sample_rate=sample_rate
    )
    if enrollment is None:
        out(
            "Enrollment failed: the model produced no usable embedding from the "
            "recordings. Check the mic level (try --input-gain) and re-run."
        )
        return 3

    enroll_path = sherpa.get("speaker_enroll_embedding") or DEFAULT_ENROLL_PATH
    save_enrollment(enroll_path, enrollment)
    _persist_local(
        config_path,
        {"speaker_embedding_model": model_path, "speaker_enroll_embedding": enroll_path},
    )

    # Self-check: how consistent were the passes? Low spread => a clean
    # reference; high spread warns the user before they rely on it.
    sims = [
        _cos(vec, enrollment.embedding)
        for vec in (gate.embed(r, sample_rate) for r in recordings)
        if vec
    ]
    out("")
    out(f"Enrolled from {enrollment.passes} clip(s) (dim={enrollment.dim}).")
    if sims:
        out(f"  pass-to-reference similarity: min={min(sims):.2f} mean={sum(sims) / len(sims):.2f}")
        if min(sims) < 0.5:
            out("  WARNING: passes disagree -- re-run --enroll in a quieter room for a cleaner voice print.")
    out(f"  saved: {os.path.abspath(enroll_path)}")
    out(f"  wired: {config_path} (sherpa.speaker_enroll_embedding)")
    out("\nNow run:  python -m core --engine sherpa")
    return 0


def _cos(a: Sequence[float], b: Sequence[float]) -> float:
    from .engines.speaker_gate import cosine_similarity

    return cosine_similarity(list(a), list(b))
