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

import hashlib
import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Mapping, Optional, Sequence

from .engines.speaker_gate import SpeakerGate, sherpa_speaker_gate

log = logging.getLogger("speaker.enroll")

# Where the enrollment vector lands when the config doesn't pin a path. Under
# pretrained_models/sherpa/ which is gitignored -- enrollment is personal and
# machine-specific, never committed.
DEFAULT_ENROLL_PATH = os.path.join("pretrained_models", "sherpa", "speaker", "enrollment.json")

# A float32 audio block recorder: takes seconds, returns mono samples in [-1, 1]
# at the enrollment sample rate. Injected so tests can avoid the microphone.
Recorder = Callable[[float], Sequence[float]]


# Bump only when the model-visible enrollment capture chain changes semantics.
# The fingerprint below covers the active stage configuration; this version covers
# the implementation/order contract itself.
ENROLLMENT_FRONTEND_VERSION = 1


@dataclass(frozen=True)
class EnrollmentFrontendProvenance:
    """Stable identity of the model-visible capture chain used for enrollment.

    ``fingerprint`` is SHA-256 over a canonical, versioned descriptor of the
    stages that ACTUALLY built (not merely requested config flags). ``summary`` is
    persisted only for an actionable mismatch warning; matching never trusts it.
    ``raw_baseline`` lets legacy enrollment files (which have no provenance) remain
    compatible only with the old no-AGC/no-APM/no-denoise path.
    """

    version: int
    fingerprint: str
    summary: str
    raw_baseline: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "version": int(self.version),
            "fingerprint": self.fingerprint,
            "summary": self.summary,
            "raw_baseline": bool(self.raw_baseline),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "EnrollmentFrontendProvenance":
        try:
            version = int(data["version"])
            fingerprint = str(data["fingerprint"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid enrollment front-end provenance") from exc
        if version <= 0 or not fingerprint:
            raise ValueError("invalid enrollment front-end provenance")
        return cls(
            version=version,
            fingerprint=fingerprint,
            summary=str(data.get("summary", "unknown front end")),
            raw_baseline=bool(data.get("raw_baseline", False)),
        )


def _cfg(config: object, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _stable_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _artifact_id(path: object) -> str:
    text = str(path or "").strip()
    return os.path.normcase(os.path.abspath(text)) if text else ""


def make_enrollment_frontend_provenance(
    config: object,
    *,
    input_agc: object | None,
    idle_apm: object | None,
    denoiser: object | None,
    apm_owns_ns: bool,
) -> EnrollmentFrontendProvenance:
    """Describe the active idle-speech front end in a stable, hashable form.

    The object arguments are the processors that ACTUALLY built. This is
    deliberate: a configured-but-missing GTCRN/APM must fingerprint as inactive,
    so an enrollment made while it failed open is rejected automatically if the
    processor becomes available later.
    """

    agc_active = input_agc is not None
    idle_apm_active = idle_apm is not None
    denoise_active = denoiser is not None and not bool(apm_owns_ns)
    input_gain = float(_cfg(config, "input_gain", 1.0) or 1.0)
    capture_voice_comm = bool(_cfg(config, "capture_voice_comm", False))

    gain: dict[str, object]
    if agc_active:
        gain = {
            "kind": "input_agc",
            "target_rms": float(_cfg(config, "input_agc_target_rms", 0.12)),
            "max_gain": float(_cfg(config, "input_agc_max_gain", 12.0)),
            "noise_floor_rms": float(_cfg(config, "input_agc_noise_floor_rms", 0.004)),
            "rise": float(_cfg(config, "input_agc_rise", 0.08)),
            "fall": float(_cfg(config, "input_agc_fall", 0.4)),
        }
    else:
        gain = {"kind": "static", "gain": input_gain}

    apm: dict[str, object] = {"active": idle_apm_active}
    if idle_apm_active:
        apm.update({
            "backend": str(_cfg(config, "aec_backend", "apm") or "apm").lower(),
            "noise_suppression": bool(getattr(idle_apm, "suppresses_noise", False)),
            "gain_control": bool(_cfg(config, "apm_gain_control", False)),
            "high_pass_filter": bool(_cfg(config, "apm_high_pass_filter", True)),
        })

    denoise: dict[str, object] = {"active": denoise_active}
    if denoise_active:
        denoise["model"] = _artifact_id(_cfg(config, "denoise_model", ""))

    descriptor = {
        "schema": ENROLLMENT_FRONTEND_VERSION,
        "capture": {
            # OS echo cancellation remains upstream in the selected/default input
            # device. These fields identify that logical capture domain; Linux's
            # externally-repointed default cannot be introspected from the app.
            "input_device": _stable_value(_cfg(config, "input_device", None)),
            "voice_comm": capture_voice_comm,
            "sample_rate": int(_cfg(config, "sample_rate", 16000) or 16000),
            "capture_samplerate": int(_cfg(config, "capture_samplerate", 0) or 0),
            "resampler_quality": str(_cfg(config, "resampler_quality", "HQ") or "HQ"),
            "block_sec": float(_cfg(config, "block_sec", 0.1) or 0.1),
        },
        "gain": gain,
        "idle_apm": apm,
        "denoise": denoise,
    }
    canonical = json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
    fingerprint = "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    stages: list[str] = []
    if capture_voice_comm:
        stages.append("os-voice-comm")
    if agc_active:
        stages.append("input-agc")
    elif input_gain != 1.0:
        stages.append(f"static-gain({input_gain:g})")
    if idle_apm_active:
        stages.append("always-on-apm+ns" if apm_owns_ns else "always-on-apm")
    if denoise_active:
        stages.append("gtcrn")

    raw_baseline = (
        not capture_voice_comm
        and not agc_active
        and input_gain == 1.0
        and not idle_apm_active
        and not denoise_active
    )
    return EnrollmentFrontendProvenance(
        version=ENROLLMENT_FRONTEND_VERSION,
        fingerprint=fingerprint,
        summary=" -> ".join(stages) if stages else "raw baseline",
        raw_baseline=raw_baseline,
    )


class EnrollmentFrontend:
    """Blockwise copy of the live idle-user capture chain for enrollment.

    Order is load-bearing and mirrors ``SherpaOnnxEngine._capture_loop``:
    InputAGC (which takes precedence over static gain), resample to the model
    rate, always-on APM against a zero far-end, then GTCRN unless APM owns NS.
    Processors are duck-typed/injectable so tests need no device or model.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        block_sec: float = 0.1,
        resampler_quality: str = "HQ",
        input_gain: float = 1.0,
        input_agc: object | None = None,
        idle_apm: object | None = None,
        denoiser: object | None = None,
        apm_owns_ns: bool = False,
        provenance: EnrollmentFrontendProvenance,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.block_sec = float(block_sec)
        self.resampler_quality = str(resampler_quality)
        self.input_gain = float(input_gain)
        self.input_agc = input_agc
        self.idle_apm = idle_apm
        self.denoiser = denoiser
        self.apm_owns_ns = bool(apm_owns_ns)
        self.provenance = provenance

    def process(self, samples: Sequence[float], capture_sample_rate: int):
        """Process one recorded clip in live-sized blocks; return model-rate audio."""
        import numpy as np

        from .audio_frontend import AudioResampler, apply_gain_soft_limit

        captured = np.asarray(samples, dtype="float32").reshape(-1)
        if captured.size == 0:
            return captured
        capture_sr = int(capture_sample_rate)
        block = max(1, int(capture_sr * self.block_sec))
        resampler = AudioResampler(
            capture_sr, self.sample_rate, quality=self.resampler_quality
        )
        output: list[object] = []
        for start in range(0, captured.size, block):
            stop = min(captured.size, start + block)
            chunk = captured[start:stop]
            if self.input_agc is not None:
                chunk = self.input_agc.process(chunk)
            elif self.input_gain != 1.0:
                chunk = apply_gain_soft_limit(chunk, self.input_gain)
            chunk = resampler.process(chunk, last=stop >= captured.size)
            if self.idle_apm is not None:
                far = np.zeros(np.asarray(chunk).size, dtype="float32")
                chunk = self.idle_apm.process_16k(chunk, far)
            if self.denoiser is not None and not self.apm_owns_ns:
                chunk = self.denoiser.process_16k(chunk)
            arr = np.asarray(chunk, dtype="float32").reshape(-1)
            if arr.size:
                output.append(arr)
        if not output:
            return np.zeros(0, dtype="float32")
        return np.concatenate(output).astype("float32", copy=False)


def build_enrollment_frontend(sherpa: Mapping[str, object]) -> EnrollmentFrontend:
    """Build the same active processors the live idle capture path would use."""
    from .audio_frontend import InputAGC
    from .engines._aec import build_aec
    from .engines._denoiser import build_denoiser
    from .engines.sherpa import SherpaConfig

    config = SherpaConfig.from_dict(dict(sherpa))
    input_agc = (
        InputAGC(
            target_rms=config.input_agc_target_rms,
            max_gain=config.input_agc_max_gain,
            noise_floor_rms=config.input_agc_noise_floor_rms,
            rise=config.input_agc_rise,
            fall=config.input_agc_fall,
        )
        if config.input_agc else None
    )
    aec = build_aec(config)
    idle_apm = aec if aec is not None and bool(getattr(aec, "always_on", False)) else None
    apm_owns_ns = bool(
        idle_apm is not None and getattr(idle_apm, "suppresses_noise", False)
    )
    denoiser = build_denoiser(config)
    provenance = make_enrollment_frontend_provenance(
        config,
        input_agc=input_agc,
        idle_apm=idle_apm,
        denoiser=denoiser,
        apm_owns_ns=apm_owns_ns,
    )
    return EnrollmentFrontend(
        sample_rate=config.sample_rate,
        block_sec=config.block_sec,
        resampler_quality=config.resampler_quality,
        input_gain=config.input_gain,
        input_agc=input_agc,
        idle_apm=idle_apm,
        denoiser=denoiser,
        apm_owns_ns=apm_owns_ns,
        provenance=provenance,
    )


@dataclass
class Enrollment:
    """A persisted speaker reference: the averaged embedding + provenance."""

    model: str            # absolute path of the embedding model that produced it
    embedding: list[float]
    sample_rate: int = 16000
    passes: int = 1
    created: str = ""
    frontend: Optional[EnrollmentFrontendProvenance] = None

    @property
    def dim(self) -> int:
        return len(self.embedding)

    def to_dict(self) -> dict:
        data: dict[str, object] = {
            "model": self.model,
            "dim": self.dim,
            "passes": self.passes,
            "sample_rate": self.sample_rate,
            "created": self.created or datetime.now().isoformat(timespec="seconds"),
            "embedding": list(self.embedding),
        }
        if self.frontend is not None:
            data["frontend"] = self.frontend.to_dict()
        return data


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
    frontend_data = data.get("frontend")
    frontend = None
    if frontend_data is not None:
        if not isinstance(frontend_data, Mapping):
            raise ValueError(f"{path}: invalid 'frontend' provenance")
        frontend = EnrollmentFrontendProvenance.from_dict(frontend_data)
    return Enrollment(
        model=str(data.get("model", "")),
        embedding=[float(x) for x in embedding],
        sample_rate=int(data.get("sample_rate", 16000)),
        passes=int(data.get("passes", 1)),
        created=str(data.get("created", "")),
        frontend=frontend,
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


def enrollment_matches_frontend(
    enrollment: Enrollment, active: EnrollmentFrontendProvenance
) -> bool:
    """Whether persisted and active model-visible capture domains are compatible.

    Legacy files have no front-end provenance. They remain usable only on the
    old raw baseline; accepting one after AGC/APM/denoise becomes active would
    let a stale embedding reject the owner, so that case fails open instead.
    """
    saved = enrollment.frontend
    if saved is None:
        return active.raw_baseline
    return (
        saved.version == active.version
        and saved.fingerprint == active.fingerprint
    )


# --- embedding from recordings (gate injected) -------------------------------


def enroll_from_recordings(
    gate: SpeakerGate,
    recordings: Sequence[Sequence[float]],
    *,
    model_path: str,
    sample_rate: int = 16000,
    frontend: Optional[EnrollmentFrontendProvenance] = None,
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
        frontend=(
            frontend
            if frontend is not None
            else make_enrollment_frontend_provenance(
                {"sample_rate": sample_rate},
                input_agc=None,
                idle_apm=None,
                denoiser=None,
                apm_owns_ns=False,
            )
        ),
    )


# --- microphone capture ------------------------------------------------------


def record_once(
    seconds: float,
    sample_rate: int = 16000,
    *,
    device=None,
    input_gain: float = 1.0,
    capture_samplerate: int = 0,
    capture_voice_comm: bool = False,
    frontend: Optional[EnrollmentFrontend] = None,
):
    """Block for ``seconds`` and return mono float32 samples at ``sample_rate``.

    ``capture_samplerate`` PINS the mic open rate exactly like the live engine
    (sherpa.capture_samplerate). This matters: some USB mics (AT2020) SELF-MUTE
    when ALSA reconfigures them to a non-native rate, so probing 16000 first (the
    old fallback below) made enrollment capture near-SILENCE -- a self-consistent
    garbage embedding that rejected the real user live (MEASURED 2026-06-01: disk
    enrollment 0.05-0.26 vs the user's own live voice, while a live-audio enrollment
    self-scored 0.76-0.94). When pinned we open ONLY at that rate (never probe), so
    the mic stays live and the enrolled embedding matches what the gate hears live.
    When 0, the legacy probe-then-fallback path is kept."""
    import numpy as np
    import sounddevice as sd

    from .engines.sherpa import _norm_device

    dev = _norm_device(device)
    extra_settings = None
    if capture_voice_comm:
        try:
            extra_settings = sd.WasapiSettings(communications=True)
        except Exception as exc:  # noqa: BLE001 - same fail-open contract as live capture
            log.info(
                "capture_voice_comm set for enrollment but WASAPI Communications "
                "is unavailable (%s); using the selected/default input device",
                exc,
            )

    def _record(frames: int, rate: int):
        kwargs = dict(
            samplerate=rate, channels=1, dtype="float32", device=dev
        )
        if extra_settings is not None:
            kwargs["extra_settings"] = extra_settings
        return sd.rec(frames, **kwargs)

    pinned = int(capture_samplerate or 0)
    if pinned > 0:
        capture_sr = pinned
        frames = int(seconds * capture_sr)
        audio = _record(frames, capture_sr)
        sd.wait()
    else:
        try:
            frames = int(seconds * sample_rate)
            audio = _record(frames, sample_rate)
            sd.wait()
            capture_sr = sample_rate
        except sd.PortAudioError:
            capture_sr = int(sd.query_devices(dev, kind="input")["default_samplerate"])
            frames = int(seconds * capture_sr)
            audio = _record(frames, capture_sr)
            sd.wait()
    samples = np.asarray(audio, dtype="float32").reshape(-1)
    # OS echo cancellation is already upstream in ``audio`` via the same selected
    # / default capture device (and WASAPI Communications request where supported).
    # The in-app stages below run blockwise in the same order as live idle speech.
    if frontend is None:
        provenance = make_enrollment_frontend_provenance(
            {"sample_rate": sample_rate, "input_gain": input_gain},
            input_agc=None,
            idle_apm=None,
            denoiser=None,
            apm_owns_ns=False,
        )
        frontend = EnrollmentFrontend(
            sample_rate=sample_rate,
            input_gain=input_gain,
            provenance=provenance,
        )
    samples = frontend.process(samples, capture_sr)
    return _vad_trim(samples, sample_rate)


def _vad_trim(samples, sample_rate: int, *, win: float = 0.02,
              thresh_ratio: float = 0.15, pad: float = 0.1):
    """Trim leading/trailing near-silence to the voiced region (+ a little pad), so
    the enrolled embedding is of SPEECH, not the quiet head/tail of the fixed
    record window -- silence content shifts the speaker embedding. Pure numpy;
    returns the clip unchanged when it can't find a voiced region."""
    import numpy as np

    a = np.asarray(samples, dtype="float32").reshape(-1)
    w = max(1, int(sample_rate * win))
    if a.size < 2 * w:
        return a
    n = (a.size // w) * w
    e = np.sqrt((a[:n].reshape(-1, w) ** 2).mean(axis=1))
    peak = float(e.max()) if e.size else 0.0
    if peak <= 0.0:
        return a
    voiced = np.where(e >= peak * thresh_ratio)[0]
    if voiced.size == 0:
        return a
    start = max(0, int(voiced[0]) * w - int(pad * sample_rate))
    end = min(a.size, (int(voiced[-1]) + 1) * w + int(pad * sample_rate))
    return a[start:end]


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
    frontend: Optional[EnrollmentFrontend] = None,
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
    if frontend is None:
        frontend = build_enrollment_frontend(sherpa)
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
                capture_samplerate=int(sherpa.get("capture_samplerate", 0) or 0),
                capture_voice_comm=bool(sherpa.get("capture_voice_comm", False)),
                frontend=frontend,
            )

    out(f"Enrolling your voice: {passes} clip(s) of ~{seconds:.0f}s each.")
    recordings = []
    for i in range(max(1, passes)):
        out(f"  [{i + 1}/{passes}] Speak naturally now...")
        recordings.append(recorder(seconds))
        out("  ...captured.")

    enrollment = enroll_from_recordings(
        gate,
        recordings,
        model_path=model_path,
        sample_rate=sample_rate,
        frontend=frontend.provenance,
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
    out(f"  capture front end: {enrollment.frontend.summary}")
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
