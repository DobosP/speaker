"""
Speech-to-Text module with multiple backends.

Supports:
- faster-whisper (recommended, 4x faster) – batch mode
- Standard Whisper via transformers (fallback) – batch mode
- pywhispercpp (streaming / low-resource) – streaming partial transcripts
"""
import numpy as np
import threading
import warnings
from typing import Callable, Optional, Generator

from utils import backend_trace

# Serialize faster-whisper final inference with whisper.cpp partial work (peak CPU).
stt_inference_lock = threading.RLock()

warnings.filterwarnings("ignore", message=".*chunk_length_s.*")

STT_RUNTIME_PROFILES = {
    "edge": {"model_type": "whispercpp", "model_id": "tiny", "n_threads": 2},
    "balanced": {"model_type": "whisper", "model_id": "base", "n_threads": 4},
    "max_quality": {"model_type": "whisper", "model_id": "small", "n_threads": 6},
}


def resolve_stt_runtime(
    runtime_profile: str = "balanced",
    model_id: Optional[str] = None,
) -> dict:
    """Resolve the concrete STT backend, model id, and thread budget."""
    selected = STT_RUNTIME_PROFILES.get(
        runtime_profile, STT_RUNTIME_PROFILES["balanced"]
    ).copy()
    if model_id:
        selected["model_id"] = model_id
    if str(selected["model_id"]).startswith("moonshine"):
        selected["model_type"] = "moonshine"
    return selected


def _is_heavy_stt_model(model_id: str) -> bool:
    """Heuristics: large/medium class models are unsuitable for live partial STT."""
    m = (model_id or "").lower()
    if "moonshine" in m:
        return False
    return any(
        x in m
        for x in (
            "large",
            "medium",
            "distil-large",
            "1.5g",
        )
    )


def resolve_partial_stt_config(
    *,
    partial_model: Optional[str],
    partial_backend: Optional[str],
    final_model_id: str,
    n_threads: int,
) -> dict[str, str | int | bool]:
    """
    Choose a small, fast model for live partial transcription in controller mode.

    Never default to the same huge model as final STT (avoids multi-GB downloads
    for whisper.cpp while the user is talking).
    """
    backend = (partial_backend or "whispercpp").strip().lower()
    if backend not in ("whispercpp", "whisper", "moonshine"):
        backend = "whispercpp"
    model_id = (partial_model or "tiny").strip() or "tiny"
    if model_id == final_model_id and _is_heavy_stt_model(final_model_id):
        model_id = "tiny"
    if _is_heavy_stt_model(model_id) and not (partial_model and partial_model.strip()):
        model_id = "tiny"
    if _is_heavy_stt_model(model_id) and partial_model and partial_model.strip():
        # User asked for a heavy partial model — unsafe for realtime; downgrade.
        model_id = "tiny"
    if backend == "moonshine" and not str(model_id).startswith("moonshine"):
        model_id = f"moonshine:{model_id}" if model_id in ("tiny", "base") else "moonshine:tiny"
    return {
        "model_id": model_id,
        "backend": backend,
        "n_threads": max(1, int(n_threads)),
    }

# ── faster-whisper (recommended) ─────────────────────────────────────────────
try:
    from faster_whisper import WhisperModel as FasterWhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

# ── transformers whisper (fallback) ──────────────────────────────────────────
try:
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# ── pywhispercpp (streaming / low-resource) ──────────────────────────────────
try:
    from pywhispercpp.model import Model as WhisperCppModel
    WHISPERCPP_AVAILABLE = True
except ImportError:
    WHISPERCPP_AVAILABLE = False

# ── Moonshine ONNX (edge-optimised, 26 MB, English, 0.05 RTF on CPU) ─────────
# PyPI package: ``pip install useful-moonshine-onnx`` (imports as ``moonshine_onnx``)
try:
    import moonshine_onnx as _moonshine_onnx
    MOONSHINE_AVAILABLE = True
except ImportError:
    MOONSHINE_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
#  MoonshineSTT – edge-optimised ONNX ASR (Tier 1 / low)
# ═══════════════════════════════════════════════════════════════════════════════

class MoonshineSTT:
    """
    Moonshine ONNX speech-to-text (English only, ~26 MB tiny / ~57 MB base).

    Real-time factor on CPU: ~0.05 (20× real-time) — fastest open-source
    English ASR available without GPU.  Streaming-capable for future use.

    Model IDs accepted: ``"moonshine:tiny"`` or ``"moonshine:base"``.
    Install: ``pip install moonshine-onnx``
    """

    _instance = None
    _model_key: Optional[str] = None

    def __new__(cls, model_size: str = "tiny"):
        key = f"moonshine:{model_size}"
        if cls._instance is None or cls._model_key != key:
            cls._instance = super().__new__(cls)
            cls._model_key = key
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_size: str = "tiny"):
        if self._initialized:
            return
        if not MOONSHINE_AVAILABLE:
            raise RuntimeError(
                "Moonshine ONNX not installed. "
                "Install with: pip install useful-moonshine-onnx"
            )
        self.sample_rate = 16000
        self.model_size = model_size
        self.backend = "moonshine"
        print(f"Loading Moonshine ONNX model: {model_size} ...")
        # The moonshine_onnx package exposes a simple transcribe() function;
        # we keep a reference to the module for lazy-model loading.
        self._mod = _moonshine_onnx
        self._initialized = True
        print(f"Moonshine ONNX loaded ({model_size}, ~26-57 MB, English)")

    def transcribe(self, audio: np.ndarray, language: str = "en") -> str:
        """Transcribe *audio* (16 kHz float32 numpy array) to text."""
        if len(audio) == 0:
            return ""
        audio = audio.flatten().astype(np.float32)
        try:
            # API: transcribe(audio, model="tiny"|"base"|"moonshine/tiny"|...)
            result = self._mod.transcribe(audio, model=self.model_size)
            if isinstance(result, list):
                return " ".join(s.strip() for s in result if s).strip()
            if isinstance(result, str):
                return result.strip()
            return ""
        except Exception:
            return ""


# ═══════════════════════════════════════════════════════════════════════════════
#  WhisperSTT – batch transcription (existing)
# ═══════════════════════════════════════════════════════════════════════════════

class WhisperSTT:
    """
    Whisper Speech-to-Text with automatic backend selection.
    Uses faster-whisper if available, falls back to transformers.
    """

    _instance = None
    _model_id = None

    def __new__(cls, model_id: str = "base"):
        if cls._instance is None or cls._model_id != model_id:
            cls._instance = super().__new__(cls)
            cls._model_id = model_id
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_id: str = "base"):
        if self._initialized:
            return

        self.sample_rate = 16000
        self.model_id = model_id

        if FASTER_WHISPER_AVAILABLE:
            self._init_faster_whisper(model_id)
        elif TRANSFORMERS_AVAILABLE:
            self._init_transformers_whisper(model_id)
        else:
            raise RuntimeError(
                "No STT backend available. "
                "Install faster-whisper or transformers."
            )

        self._initialized = True

    def _init_faster_whisper(self, model_id: str):
        print(f"Loading faster-whisper model: {model_id}...")
        model_map = {
            "openai/whisper-tiny": "tiny",
            "openai/whisper-base": "base",
            "openai/whisper-small": "small",
            "openai/whisper-medium": "medium",
            "openai/whisper-large": "large-v3",
            "openai/whisper-large-v3": "large-v3",
            "openai/whisper-large-v3-turbo": "large-v3-turbo",
        }
        model_size = model_map.get(model_id, model_id)

        try:
            import torch as _torch
            if _torch.cuda.is_available():
                device, compute_type = "cuda", "float16"
            else:
                device, compute_type = "cpu", "int8"
        except Exception:
            device, compute_type = "cpu", "int8"

        print(f"   Device: {device}, Compute: {compute_type}")
        try:
            self.model = FasterWhisperModel(
                model_size, device=device, compute_type=compute_type
            )
        except RuntimeError as exc:
            err = str(exc).lower()
            if device == "cuda" and (
                "out of memory" in err or "cuda" in err
            ):
                print(
                    f"   CUDA load failed ({exc!r}); "
                    "freeing cache and falling back to CPU int8"
                )
                try:
                    import torch as _tc
                    _tc.cuda.empty_cache()
                except Exception:
                    pass
                device, compute_type = "cpu", "int8"
                self.model = FasterWhisperModel(
                    model_size, device=device, compute_type=compute_type
                )
            else:
                raise
        self.backend = "faster-whisper"
        print("faster-whisper loaded (4x faster)!")

    def _init_transformers_whisper(self, model_id: str):
        print(f"Loading Whisper model: {model_id}...")
        if not model_id.startswith("openai/"):
            model_id = f"openai/whisper-{model_id}"

        import torch as _torch
        self.device = "cuda:0" if _torch.cuda.is_available() else "cpu"
        self.torch_dtype = (
            _torch.float16 if _torch.cuda.is_available() else _torch.float32
        )
        print(f"   Device: {self.device}")

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(self.device)

        processor = AutoProcessor.from_pretrained(model_id)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
        self.backend = "transformers"
        print("Whisper model loaded!")

    def transcribe(self, audio: np.ndarray, language: str = "en") -> str:
        if len(audio) == 0:
            return ""
        audio = audio.flatten().astype(np.float32)

        if self.backend == "faster-whisper":
            segments, _ = self.model.transcribe(
                audio, language=language, beam_size=5, vad_filter=True
            )
            text = " ".join(segment.text for segment in segments)
        else:
            result = self.pipe(
                {"raw": audio, "sampling_rate": self.sample_rate},
                generate_kwargs={"language": language},
            )
            text = result["text"]
        return text.strip()


# ═══════════════════════════════════════════════════════════════════════════════
#  StreamingSTT – real-time partial transcripts via whisper.cpp
# ═══════════════════════════════════════════════════════════════════════════════

class StreamingSTT:
    """
    Streaming speech-to-text using pywhispercpp (whisper.cpp bindings).

    Gives partial transcripts as the user speaks, enabling faster response
    times and better turn-taking.

    Usage::

        stt = StreamingSTT(model="base")
        partial = stt.transcribe_partial(audio_chunk_16khz)
        final  = stt.transcribe_final(full_audio_16khz)

    For low-resource devices, use model="tiny" (~40MB RAM).
    """

    _instance = None
    _model_id = None

    def __new__(cls, model: str = "base", n_threads: int = 4):
        cache_key = f"{model}:{n_threads}"
        if cls._instance is None or cls._model_id != cache_key:
            cls._instance = super().__new__(cls)
            cls._model_id = cache_key
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model: str = "base", n_threads: int = 4):
        if self._initialized:
            return

        if not WHISPERCPP_AVAILABLE:
            raise RuntimeError(
                "pywhispercpp not installed. "
                "Install with: pip install pywhispercpp"
            )

        self.model_id = model
        self.n_threads = max(1, int(n_threads))
        self.cache_key = f"{self.model_id}:{self.n_threads}"
        print(f"Loading whisper.cpp model: {model}...")
        self._model = WhisperCppModel(
            model=model,
            redirect_whispercpp_logs_to=False,
            n_threads=self.n_threads,
        )
        self.sample_rate = 16000
        self.backend = "whispercpp"
        self._initialized = True
        print(f"whisper.cpp loaded (model: {model}, threads: {self.n_threads})")

    # whisper.cpp rejects buffers shorter than ~100 ms (see whisper_full_with_state).
    _min_partial_samples: int = int(0.11 * 16000)

    def transcribe_partial(
        self,
        audio: np.ndarray,
        language: str = "en",
        *,
        _trace_as_partial: bool = True,
    ) -> str:
        """
        Transcribe a partial audio buffer.

        This is meant to be called repeatedly with growing audio buffers
        to get incremental results.  The underlying model processes the
        entire buffer each time (whisper.cpp is fast enough for this on CPU).
        """
        if len(audio) == 0:
            return ""
        audio = audio.flatten().astype(np.float32)
        if len(audio) < self._min_partial_samples:
            pad = self._min_partial_samples - len(audio)
            audio = np.concatenate([audio, np.zeros(pad, dtype=np.float32)])
        try:
            with stt_inference_lock:
                if _trace_as_partial:
                    backend_trace.record_stt_partial(self.model_id, len(audio))
                segments = self._model.transcribe(audio, language=language)
            return " ".join(seg.text for seg in segments).strip()
        except Exception:
            return ""

    def transcribe_final(self, audio: np.ndarray, language: str = "en") -> str:
        """Transcribe the final complete utterance."""
        return self.transcribe_partial(audio, language, _trace_as_partial=False)


# ═══════════════════════════════════════════════════════════════════════════════
#  Module-level helpers
# ═══════════════════════════════════════════════════════════════════════════════

_stt_model: Optional[WhisperSTT] = None
_moonshine_stt: Optional[MoonshineSTT] = None
_streaming_stt: Optional[StreamingSTT] = None


def get_stt_model(model_id: str = "base"):
    """Get the singleton batch STT model instance.

    Accepts Whisper model IDs (``"base"``, ``"distil-medium.en"``, …) or
    Moonshine model IDs (``"moonshine:tiny"``, ``"moonshine:base"``).
    """
    global _stt_model, _moonshine_stt
    if model_id.startswith("moonshine"):
        size = model_id.split(":")[-1] if ":" in model_id else "tiny"
        if _moonshine_stt is None or _moonshine_stt.model_size != size:
            _moonshine_stt = MoonshineSTT(size)
        return _moonshine_stt
    # Recreate when the requested Whisper model id changes (tests and CLI use
    # different sizes; the old code always returned the first-loaded model).
    need_new = (
        _stt_model is None
        or getattr(_stt_model, "model_id", None) != model_id
    )
    if need_new and _stt_model is not None:
        old = _stt_model
        _stt_model = None
        try:
            if hasattr(old, "model"):
                del old.model
        except Exception:
            pass
        import gc
        gc.collect()
        try:
            import torch as _tc
            _tc.cuda.empty_cache()
        except Exception:
            pass
    if _stt_model is None:
        _stt_model = WhisperSTT(model_id)
    return _stt_model


def get_streaming_stt(
    model_id: str = "base",
    n_threads: int = 4,
) -> Optional[StreamingSTT]:
    """Get the singleton streaming STT model (returns None if unavailable)."""
    global _streaming_stt
    if not WHISPERCPP_AVAILABLE:
        return None
    cache_key = f"{model_id}:{max(1, int(n_threads))}"
    if _streaming_stt is None or getattr(_streaming_stt, "cache_key", None) != cache_key:
        try:
            _streaming_stt = StreamingSTT(model_id, n_threads=n_threads)
        except Exception:
            return None
    return _streaming_stt


def transcribe_audio(
    audio: np.ndarray,
    model_id: str = "base",
    model_type: str = "whisper",
    **kwargs,
) -> str:
    """
    Transcribe audio to text.

    Args:
        audio: Audio data as numpy array (16 kHz float32)
        model_id: Model identifier — e.g. ``"base"``, ``"distil-medium.en"``,
            ``"large-v3-turbo"``, ``"moonshine:tiny"``, ``"moonshine:base"``
        model_type: ``"whisper"``, ``"whispercpp"``, or ``"moonshine"``

    Returns:
        Transcribed text (stripped)
    """
    flat = np.asarray(audio).flatten()
    n_samples = int(flat.size)

    with stt_inference_lock:
        backend_trace.record_stt_final(model_id, model_type, n_samples)
        # Moonshine can be requested via model_type or via model_id prefix
        if model_type == "moonshine" or model_id.startswith("moonshine"):
            model = get_stt_model(
                model_id if model_id.startswith("moonshine") else "moonshine:tiny"
            )
            return model.transcribe(audio)

        if model_type == "whispercpp":
            stt = get_streaming_stt(
                model_id,
                n_threads=kwargs.get("n_threads", 4),
            )
            if stt is None:
                # Fall back to batch whisper
                model = get_stt_model(model_id)
                return model.transcribe(audio)
            return stt.transcribe_final(audio)

        if model_type != "whisper":
            raise ValueError(f"Unsupported model type: {model_type}")

        model = get_stt_model(model_id)
        return model.transcribe(audio)


# ═══════════════════════════════════════════════════════════════════════════════
#  Test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing STT module...")
    print(f"faster-whisper available: {FASTER_WHISPER_AVAILABLE}")
    print(f"transformers available: {TRANSFORMERS_AVAILABLE}")
    print(f"pywhispercpp available: {WHISPERCPP_AVAILABLE}")

    model = get_stt_model("base")
    print(f"Backend: {model.backend}")

    silence = np.zeros(16000, dtype=np.float32)
    result = model.transcribe(silence)
    print(f"Silence transcription: '{result}'")

    if WHISPERCPP_AVAILABLE:
        streaming = get_streaming_stt("base")
        if streaming:
            result2 = streaming.transcribe_partial(silence)
            print(f"Streaming silence: '{result2}'")
