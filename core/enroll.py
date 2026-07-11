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
import shlex
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Callable, Mapping, Optional, Sequence

from .audio_frontend import InputAGC
from .engines.speaker_gate import (
    SpeakerGate,
    sherpa_speaker_gate,
    trim_to_voiced_region,
)

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
ENROLLMENT_FRONTEND_VERSION = 5


def _agc_floor_bucket_db(value: float) -> int:
    floor = max(1e-8, float(value))
    return 3 * round((20.0 * math.log10(floor)) / 3.0)


@dataclass(frozen=True)
class CaptureResolution:
    """The capture domain that actually produced model-visible samples.

    Config intent is not enough here: PortAudio may fall back to another device
    or rate, the resampler backend depends on installed libraries, and an OS
    voice-processing request may fail to apply.  Enrollment and live gating only
    compare domains after those facts are known.
    """

    route: str
    capture_sample_rate: int
    model_sample_rate: int
    resampler: str
    voice_comm: str = "none"
    input_agc_noise_floor_rms: Optional[float] = None

    def descriptor(self) -> dict[str, object]:
        out: dict[str, object] = {
            "route": self.route,
            "capture_sample_rate": int(self.capture_sample_rate),
            "model_sample_rate": int(self.model_sample_rate),
            "resampler": self.resampler,
            "voice_comm": self.voice_comm,
        }
        if self.input_agc_noise_floor_rms is not None:
            # Retained for diagnostics and migration of v2 fingerprints. The
            # stable v5 front-end identity deliberately excludes this per-run
            # ambient calibration value.
            out["input_agc_noise_floor_db_3"] = _agc_floor_bucket_db(
                self.input_agc_noise_floor_rms
            )
        return out


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
    # Runtime-only migration aliases. Active v5 provenance carries old hashes
    # only when the model-visible chain did not use InputAGC. Pre-v5 AGC audio
    # used a different applied-gain algorithm and must be re-enrolled. These
    # aliases are never persisted.
    compatible_fingerprints: frozenset[str] = field(
        default_factory=frozenset,
        compare=False,
        repr=False,
    )

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


def _configured_capture_resolution(config: object) -> CaptureResolution:
    """Conservative unresolved domain for pure helpers and legacy tests.

    Production enrollment/live startup replace this with a resolution derived
    from a successfully opened stream.  Keeping the marker in the fingerprint
    prevents an unresolved injected capture from masquerading as production.
    """
    model_sr = int(_cfg(config, "sample_rate", 16000) or 16000)
    configured_sr = int(_cfg(config, "capture_samplerate", 0) or 0)
    capture_sr = configured_sr or model_sr
    selector = _stable_value(_cfg(config, "input_device", None))
    return CaptureResolution(
        route=f"unresolved:{selector!r}",
        capture_sample_rate=capture_sr,
        model_sample_rate=model_sr,
        resampler="identity" if capture_sr == model_sr else "unresolved",
        voice_comm=(
            "requested-unverified"
            if bool(_cfg(config, "capture_voice_comm", False))
            else "none"
        ),
    )


def make_enrollment_frontend_provenance(
    config: object,
    *,
    input_agc: object | None,
    idle_apm: object | None,
    denoiser: object | None,
    apm_owns_ns: bool,
    capture: Optional[CaptureResolution] = None,
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
    capture = capture or _configured_capture_resolution(config)

    gain: dict[str, object]
    if agc_active:
        gain = {
            "kind": "input_agc",
            "algorithm": str(
                getattr(
                    input_agc,
                    "algorithm",
                    InputAGC.algorithm,
                )
            ),
            "target_rms": float(
                getattr(input_agc, "target_rms", _cfg(config, "input_agc_target_rms", 0.12))
            ),
            "max_gain": float(
                getattr(input_agc, "max_gain", _cfg(config, "input_agc_max_gain", 12.0))
            ),
            "rise": float(
                getattr(input_agc, "rise", _cfg(config, "input_agc_rise", 0.08))
            ),
            "fall": float(
                getattr(input_agc, "fall", _cfg(config, "input_agc_fall", 0.4))
            ),
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

    # Ambient calibration is runtime state, not capture-chain identity. It can
    # legitimately change on every start while the device, route, resampler,
    # AGC algorithm, and model-visible processors stay identical.
    stable_capture = {
        key: value
        for key, value in capture.descriptor().items()
        if key != "input_agc_noise_floor_db_3"
    }
    descriptor = {
        "schema": ENROLLMENT_FRONTEND_VERSION,
        "capture": {
            **stable_capture,
            "resampler_quality": str(_cfg(config, "resampler_quality", "HQ") or "HQ"),
            "block_sec": float(_cfg(config, "block_sec", 0.1) or 0.1),
        },
        "gain": gain,
        "idle_apm": apm,
        "denoise": denoise,
    }
    canonical = json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
    fingerprint = "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    # V5 changes model-visible InputAGC output, so no v2/v3/v4 AGC fingerprint is
    # compatible. Chains without InputAGC are byte-identical: retain bounded
    # aliases for their exact v2/v3/v4 descriptors so those owners do not re-enroll
    # for an unrelated schema bump. Different routes/processors still cannot
    # match because every other descriptor field remains fixed.
    compatible_fingerprints: set[str] = set()
    if not agc_active:
        for legacy_version in (2, 3, 4):
            legacy_descriptor = {
                **descriptor,
                "schema": legacy_version,
            }
            legacy_canonical = json.dumps(
                legacy_descriptor, sort_keys=True, separators=(",", ":")
            )
            compatible_fingerprints.add(
                "sha256:"
                + hashlib.sha256(legacy_canonical.encode("utf-8")).hexdigest()
            )

    stages: list[str] = []
    if capture.voice_comm != "none":
        stages.append(capture.voice_comm)
    if agc_active:
        stages.append("input-agc-current-signal")
    elif input_gain != 1.0:
        stages.append(f"static-gain({input_gain:g})")
    if idle_apm_active:
        stages.append("always-on-apm+ns" if apm_owns_ns else "always-on-apm")
    if denoise_active:
        stages.append("gtcrn")

    raw_baseline = (
        capture.voice_comm == "none"
        and capture.capture_sample_rate == capture.model_sample_rate
        and capture.resampler == "identity"
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
        compatible_fingerprints=frozenset(compatible_fingerprints),
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
        config: object | None = None,
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
        self.config = config if config is not None else {
            "sample_rate": self.sample_rate,
            "block_sec": self.block_sec,
            "resampler_quality": self.resampler_quality,
            "input_gain": self.input_gain,
        }
        self.capture_resolution: Optional[CaptureResolution] = None
        # Enrollment admission compares pre-gain speech with the *measured*
        # pre-gain ambient from this run. InputAGC.noise_floor_rms is a different
        # operating point: compute_input_calibration has already multiplied the
        # ambient by its headroom and clamped it for runtime gain control. Reusing
        # that expanded gate and adding another voice margin rejects valid quiet
        # mics (the live ROG route measured 0.00018 ambient / ~0.003 speech while
        # the AGC minimum was 0.004).
        self.measured_pre_gain_ambient_rms: Optional[float] = None

    def bind_capture(self, capture: CaptureResolution) -> None:
        """Bind once to the actual route/rate used for this enrollment run."""
        if self.capture_resolution is not None and self.capture_resolution != capture:
            raise RuntimeError(
                "capture domain changed during enrollment: "
                f"{self.capture_resolution.route}@{self.capture_resolution.capture_sample_rate} "
                f"-> {capture.route}@{capture.capture_sample_rate}"
            )
        self.capture_resolution = capture
        self.provenance = make_enrollment_frontend_provenance(
            self.config,
            input_agc=self.input_agc,
            idle_apm=self.idle_apm,
            denoiser=self.denoiser,
            apm_owns_ns=self.apm_owns_ns,
            capture=capture,
        )

    def _pre_gain_model_blocks(
        self, samples: Sequence[float], capture_sample_rate: int
    ) -> list:
        """Return blockwise model-band PCM before gain or other processors."""
        import numpy as np

        from .audio_frontend import AudioResampler

        captured = np.asarray(samples, dtype="float32").reshape(-1)
        block = max(1, int(int(capture_sample_rate) * self.block_sec))
        resampler = AudioResampler(
            int(capture_sample_rate), self.sample_rate, quality=self.resampler_quality
        )
        blocks = []
        for start in range(0, captured.size, block):
            stop = min(captured.size, start + block)
            out = resampler.process(captured[start:stop], last=stop >= captured.size)
            arr = np.asarray(out, dtype="float32").reshape(-1)
            if arr.size:
                blocks.append(arr)
        return blocks

    def pre_gain_model_audio(
        self, samples: Sequence[float], capture_sample_rate: int
    ):
        """Return the exact pre-gain frequency domain used by calibration."""
        import numpy as np

        blocks = self._pre_gain_model_blocks(samples, capture_sample_rate)
        return (
            np.concatenate(blocks)
            if blocks
            else np.zeros(0, dtype="float32")
        )

    def calibrate(self, samples: Sequence[float], capture_sample_rate: int) -> dict:
        """Apply the live pre-AGC ambient calibration to this front end."""
        from .audio_frontend import compute_input_calibration

        blocks = self._pre_gain_model_blocks(samples, capture_sample_rate)
        cal = compute_input_calibration(blocks)
        self.measured_pre_gain_ambient_rms = float(cal["ambient_rms"])
        if self.input_agc is not None:
            self.input_agc.noise_floor_rms = cal["noise_floor_rms"]
        return cal

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
        config=config,
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
    if saved.version == active.version:
        return saved.fingerprint == active.fingerprint
    return (
        saved.version in {2, 3, 4}
        and active.version == 5
        and saved.fingerprint in active.compatible_fingerprints
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


class EnrollmentCaptureError(RuntimeError):
    """A production capture domain could not be opened or verified safely."""


class EnrollmentVoiceError(RuntimeError):
    """A production clip lacked enough pre-gain voice evidence to enroll."""


def verify_required_os_echo_route(
    sherpa: Mapping[str, object],
    *,
    platform: Optional[str] = None,
    pipewire_probe: Optional[Callable[[], object]] = None,
) -> str:
    """Return the verified upstream voice-processing mode, or raise.

    Linux word-cut relies on PipeWire's echo-cancel source *and* sink.  The
    normal runtime readiness gate checks the same state, but ``--enroll`` is
    intentionally outside that gate, so enrollment must refuse to persist an
    unverifiable reference itself.
    """
    platform = platform or sys.platform
    aec_enabled = bool(sherpa.get("aec_enabled", False))
    wants_word_cut = (
        bool(sherpa.get("barge_in_enabled", True))
        and bool(sherpa.get("barge_word_cut_enabled", False))
        and not aec_enabled
    )
    wants_voice_comm = bool(sherpa.get("capture_voice_comm", False))
    if platform.startswith("linux"):
        from .readiness import _check_pipewire_echo_route, probe_pipewire_state

        probe = pipewire_probe or probe_pipewire_state
        state = probe()
        if wants_word_cut or wants_voice_comm:
            ok, detail = _check_pipewire_echo_route(sherpa, state)
            if not ok:
                raise EnrollmentCaptureError(
                    "required PipeWire echo-cancel route is not verifiable: " + detail
                )
            return "pipewire-echo-cancel:" + detail

        # Even when OS EC is not required, a Linux PipeWire/Pulse *default* can
        # be repointed between raw and echo-cancel nodes without changing the
        # PortAudio selector. Record those actual defaults whenever pactl can
        # inspect them so the two capture domains cannot share a fingerprint.
        selector = str(sherpa.get("input_device", "") or "").strip().lower()
        defaultish = not selector or selector in {"default", "pipewire", "pulse"}
        if state is not None and defaultish:
            return (
                "pipewire-default:source=" + state.default_source
                + ";sink=" + state.default_sink
            )
        if selector in {"pipewire", "pulse"} and state is None:
            raise EnrollmentCaptureError(
                "the selected PipeWire/Pulse default route could not be identified"
            )
        return "none"
    if platform.startswith("win") and wants_voice_comm:
        # The concrete WASAPI setting is verified when the stream is opened.
        return "wasapi-pending"
    if wants_voice_comm or wants_word_cut:
        raise EnrollmentCaptureError(
            "the configured OS echo-cancel capture route cannot be verified "
            f"on platform {platform!r}"
        )
    return "none"


def _capture_route_identity(sd, selector) -> str:
    """Stable identity for the selected route after a successful open."""
    selected = "default" if selector is None else repr(selector)
    try:
        info = sd.query_devices(selector, kind="input")
    except Exception:
        return f"selector={selected};device=unknown"
    if isinstance(info, Mapping):
        name = str(info.get("name", "?"))
        hostapi = info.get("hostapi", "?")
    else:
        name = str(getattr(info, "name", "?"))
        hostapi = getattr(info, "hostapi", "?")
    host_name = str(hostapi)
    try:
        host = sd.query_hostapis(hostapi)
        if isinstance(host, Mapping):
            host_name = str(host.get("name", hostapi))
        else:
            host_name = str(getattr(host, "name", hostapi))
    except Exception:
        pass
    return f"selector={selected};host={host_name};device={name}"


def make_capture_resolution(
    sd,
    selector,
    *,
    capture_sample_rate: int,
    model_sample_rate: int,
    resampler_quality: str,
    voice_comm: str,
    input_agc: object | None,
) -> CaptureResolution:
    """Describe a successfully opened stream using the actual resampler."""
    from .audio_frontend import AudioResampler

    resampler = AudioResampler(
        int(capture_sample_rate), int(model_sample_rate), quality=resampler_quality
    )
    floor = (
        float(getattr(input_agc, "noise_floor_rms"))
        if input_agc is not None
        else None
    )
    return CaptureResolution(
        route=_capture_route_identity(sd, selector),
        capture_sample_rate=int(capture_sample_rate),
        model_sample_rate=int(model_sample_rate),
        resampler=resampler.kind,
        voice_comm=voice_comm,
        input_agc_noise_floor_rms=floor,
    )


def _capture_raw_once(
    seconds: float,
    sample_rate: int,
    *,
    device=None,
    capture_samplerate: int = 0,
    capture_voice_comm: bool = False,
    resampler_quality: str = "HQ",
    input_agc: object | None = None,
    os_echo_mode: str = "none",
    platform: Optional[str] = None,
    route_config: Optional[Mapping[str, object]] = None,
):
    """Capture raw PCM through the same ordered open ladder as live runtime."""
    import numpy as np
    import sounddevice as sd

    from .audio_frontend import CLEAN_CAPTURE_RATES
    from .engines.sherpa import _capture_attempts, _norm_device

    platform = platform or sys.platform
    dev = _norm_device(device)
    try:
        dev_sr = int(sd.query_devices(dev, kind="input")["default_samplerate"])
    except Exception:
        dev_sr = int(sample_rate)

    def _supports(candidate, rate: int) -> bool:
        try:
            sd.check_input_settings(
                device=candidate, samplerate=rate, channels=1, dtype="float32"
            )
            return True
        except Exception:
            return False

    attempts = _capture_attempts(
        dev,
        preferred_sr=int(sample_rate),
        dev_sr_in=dev_sr,
        pinned_sr=int(capture_samplerate or 0),
        clean_rates=CLEAN_CAPTURE_RATES,
        supports=_supports,
    )
    extra_settings = None
    applied_voice_comm = os_echo_mode
    if capture_voice_comm and platform.startswith("win"):
        try:
            extra_settings = sd.WasapiSettings(communications=True)
        except Exception as exc:
            raise EnrollmentCaptureError(
                "capture_voice_comm is configured but WASAPI Communications "
                f"could not be applied: {exc}"
            ) from exc
        applied_voice_comm = "wasapi-communications"
    elif capture_voice_comm and os_echo_mode == "none":
        raise EnrollmentCaptureError(
            "capture_voice_comm is configured but no verified OS voice route was applied"
        )

    last_exc: Optional[Exception] = None
    for attempt in attempts:
        try:
            kwargs = {
                "samplerate": int(attempt.samplerate),
                "channels": 1,
                "dtype": "float32",
                "device": attempt.device,
            }
            if extra_settings is not None:
                kwargs["extra_settings"] = extra_settings
            frames = int(float(seconds) * int(attempt.samplerate))
            audio = sd.rec(frames, **kwargs)
            sd.wait()
        except Exception as exc:  # noqa: BLE001 - mirror live open fallback
            last_exc = exc
            continue
        actual_voice_mode = applied_voice_comm
        if route_config is not None and platform.startswith("linux"):
            actual_route = dict(route_config)
            actual_route["input_device"] = attempt.device
            # A configured EC selector may have failed and fallen back to a raw
            # default. Verify the successful attempt itself before labeling the
            # audio as OS-cancelled.
            actual_voice_mode = verify_required_os_echo_route(
                actual_route, platform=platform
            )
        resolution = make_capture_resolution(
            sd,
            attempt.device,
            capture_sample_rate=int(attempt.samplerate),
            model_sample_rate=int(sample_rate),
            resampler_quality=resampler_quality,
            voice_comm=actual_voice_mode,
            input_agc=input_agc,
        )
        return np.asarray(audio, dtype="float32").reshape(-1), resolution
    raise EnrollmentCaptureError(
        f"could not open the configured capture route: {last_exc or 'no attempts'}"
    )


def record_once(
    seconds: float,
    sample_rate: int = 16000,
    *,
    device=None,
    input_gain: float = 1.0,
    capture_samplerate: int = 0,
    capture_voice_comm: bool = False,
    frontend: Optional[EnrollmentFrontend] = None,
    os_echo_mode: Optional[str] = None,
    route_config: Optional[Mapping[str, object]] = None,
    require_voice_evidence: bool = False,
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
    if os_echo_mode is None:
        os_echo_mode = verify_required_os_echo_route(
            {
                "input_device": device,
                "capture_voice_comm": capture_voice_comm,
            }
        )
    input_agc = getattr(frontend, "input_agc", None)
    samples, resolution = _capture_raw_once(
        seconds,
        sample_rate,
        device=device,
        capture_samplerate=capture_samplerate,
        capture_voice_comm=capture_voice_comm,
        resampler_quality=getattr(frontend, "resampler_quality", "HQ"),
        input_agc=input_agc,
        os_echo_mode=os_echo_mode,
        route_config=route_config,
    )
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
    if hasattr(frontend, "bind_capture"):
        frontend.bind_capture(resolution)
    if require_voice_evidence:
        ambient_floor = getattr(frontend, "measured_pre_gain_ambient_rms", None)
        input_agc = getattr(frontend, "input_agc", None)
        if ambient_floor is None and input_agc is not None:
            # No live calibration ran: preserve the historical configured AGC
            # gate rather than inventing a lower admission threshold.
            ambient_floor = float(input_agc.noise_floor_rms)
        admitted, voiced_sec = _has_enrollment_voice_evidence(
            frontend.pre_gain_model_audio(
                samples, resolution.capture_sample_rate
            ),
            frontend.sample_rate,
            ambient_floor_rms=ambient_floor,
        )
        if not admitted:
            raise EnrollmentVoiceError(
                f"only {voiced_sec:.2f}s of pre-gain voice rose above the "
                "capture floor; check the mic/route and speak naturally"
            )
    samples = frontend.process(samples, resolution.capture_sample_rate)
    return _vad_trim(samples, sample_rate)


def _vad_trim(samples, sample_rate: int, *, win: float = 0.02,
              thresh_ratio: float = 0.15, pad: float = 0.1):
    """Backward-compatible alias for the shared speaker-embedding envelope."""
    return trim_to_voiced_region(
        samples,
        sample_rate,
        win=win,
        thresh_ratio=thresh_ratio,
        pad=pad,
    )


def _has_enrollment_voice_evidence(
    samples: Sequence[float],
    sample_rate: int,
    *,
    ambient_floor_rms: Optional[float],
    min_voiced_sec: float = 0.35,
    margin_db: float = 6.0,
) -> tuple[bool, float]:
    """Reject production clips that contain no sustained voice-like energy.

    The calibrated capture floor is the primary operating point. When a profile
    has no InputAGC/calibration floor, require within-clip dynamic contrast so a
    constant silence/noise bed cannot enroll merely because an embedder returns
    a vector for it. This is an enrollment admission check, not runtime VAD.
    """
    import numpy as np

    audio = np.asarray(samples, dtype="float32").reshape(-1)
    frame = max(1, int(round(max(1, sample_rate) * 0.02)))
    if audio.size < frame:
        return False, 0.0
    levels = []
    for start in range(0, audio.size - frame + 1, frame):
        block = audio[start:start + frame].astype("float64")
        levels.append(float(np.sqrt(np.mean(block * block))))
    if not levels:
        return False, 0.0

    floor = float(ambient_floor_rms or 0.0)
    if floor > 0.0:
        # Keep an absolute numerical floor even when a muted/near-zero device
        # calibrated an extremely small ambient. A brief click still cannot
        # satisfy min_voiced_sec.
        threshold = max(
            1e-4,
            floor * (10.0 ** (float(margin_db) / 20.0)),
        )
    else:
        # No calibrated absolute operating point: a natural spoken clip has
        # pauses/phoneme dynamics, while steady silence or a constant noise bed
        # does not. A small normalized-audio sanity floor prevents a padded
        # numerical near-zero burst from becoming an enrollment.
        quiet = float(np.percentile(levels, 10.0))
        threshold = max(1e-4, quiet * 2.0)
    voiced_frames = sum(level > threshold for level in levels)
    voiced_sec = voiced_frames * frame / max(1, sample_rate)
    return voiced_sec >= max(0.0, float(min_voiced_sec)), voiced_sec


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
    production_capture = recorder is None
    os_echo_mode = "none"
    enrollment_frontend: EnrollmentFrontendProvenance
    if production_capture:
        try:
            os_echo_mode = verify_required_os_echo_route(sherpa)
        except EnrollmentCaptureError as exc:
            out(f"Enrollment aborted: {exc}")
            out("  Fix the configured OS echo-cancel route, then re-run --enroll.")
            return 4
        if frontend is None:
            frontend = build_enrollment_frontend(sherpa)
        if bool(sherpa.get("input_calibrate", False)):
            cal_sec = float(sherpa.get("input_calibrate_sec", 1.5) or 0.0)
            if cal_sec > 0.0:
                out(f"Calibrating {cal_sec:.1f}s of room tone -- stay quiet...")
                try:
                    ambient, resolution = _capture_raw_once(
                        cal_sec,
                        sample_rate,
                        device=sherpa.get("input_device"),
                        capture_samplerate=int(
                            sherpa.get("capture_samplerate", 0) or 0
                        ),
                        capture_voice_comm=bool(
                            sherpa.get("capture_voice_comm", False)
                        ),
                        resampler_quality=str(
                            sherpa.get("resampler_quality", "HQ") or "HQ"
                        ),
                        input_agc=frontend.input_agc,
                        os_echo_mode=os_echo_mode,
                        route_config=sherpa,
                    )
                    calibration = frontend.calibrate(
                        ambient, resolution.capture_sample_rate
                    )
                    if frontend.input_agc is not None:
                        resolution = replace(
                            resolution,
                            input_agc_noise_floor_rms=float(
                                frontend.input_agc.noise_floor_rms
                            ),
                        )
                    frontend.bind_capture(resolution)
                except (EnrollmentCaptureError, RuntimeError) as exc:
                    out(f"Enrollment aborted: capture calibration failed: {exc}")
                    return 4
                out(
                    "  ambient calibrated: "
                    f"noise floor={calibration['noise_floor_rms']:.4f}"
                )
    else:
        # A plain injected recorder has no evidence that it applied the configured
        # processors or OS route. Persist a synthetic RAW domain, never the built
        # production front end, so test/replay PCM cannot create a reference that
        # later rejects the owner on a real mic.
        injected_capture = CaptureResolution(
            route="injected-raw",
            capture_sample_rate=sample_rate,
            model_sample_rate=sample_rate,
            resampler="identity",
        )
        enrollment_frontend = make_enrollment_frontend_provenance(
            {"sample_rate": sample_rate},
            input_agc=None,
            idle_apm=None,
            denoiser=None,
            apm_owns_ns=False,
            capture=injected_capture,
        )
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
                os_echo_mode=os_echo_mode,
                route_config=sherpa,
                require_voice_evidence=True,
            )

    out(f"Enrolling your voice: {passes} clip(s) of ~{seconds:.0f}s each.")
    recordings = []
    try:
        for i in range(max(1, passes)):
            out(f"  [{i + 1}/{passes}] Speak naturally now...")
            recordings.append(recorder(seconds))
            out("  ...captured.")
    except EnrollmentVoiceError as exc:
        out(f"Enrollment failed: {exc}; no reference was saved.")
        return 3
    except (EnrollmentCaptureError, RuntimeError) as exc:
        out(f"Enrollment aborted: capture failed safely: {exc}")
        return 4

    if production_capture:
        assert frontend is not None
        if frontend.capture_resolution is None:
            out(
                "Enrollment aborted: recorder did not report an actual capture "
                "route/rate; refusing to label raw audio as processed."
            )
            return 4
        enrollment_frontend = frontend.provenance

    enrollment = enroll_from_recordings(
        gate,
        recordings,
        model_path=model_path,
        sample_rate=sample_rate,
        frontend=enrollment_frontend,
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
    launch = ["python", "-m", "core", "--engine", "sherpa"]
    for flag, key in (
        ("--input-device", "input_device"),
        ("--output-device", "output_device"),
    ):
        selector = sherpa.get(key)
        if selector not in (None, ""):
            launch.extend((flag, str(selector)))
    out("\nNow run:  " + " ".join(shlex.quote(part) for part in launch))
    return 0


def _cos(a: Sequence[float], b: Sequence[float]) -> float:
    from .engines.speaker_gate import cosine_similarity

    return cosine_similarity(list(a), list(b))
