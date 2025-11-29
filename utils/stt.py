"""
Speech-to-Text module with multiple backends.

Supports:
- faster-whisper (recommended, 4x faster)
- Standard Whisper (fallback)
"""
import numpy as np
import warnings

warnings.filterwarnings("ignore", message=".*chunk_length_s.*")

# Try to import faster-whisper first (recommended)
try:
    from faster_whisper import WhisperModel as FasterWhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

# Fallback to standard whisper
try:
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class WhisperSTT:
    """
    Whisper Speech-to-Text with automatic backend selection.
    Uses faster-whisper if available, falls back to transformers.
    """
    
    _instance = None
    _model_id = None
    
    def __new__(cls, model_id: str = "base"):
        """Singleton pattern - only create one instance per model."""
        if cls._instance is None or cls._model_id != model_id:
            cls._instance = super().__new__(cls)
            cls._model_id = model_id
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, model_id: str = "base"):
        """Initialize the Whisper model."""
        if self._initialized:
            return
        
        self.sample_rate = 16000
        self.model_id = model_id
        
        # Try faster-whisper first (4x faster)
        if FASTER_WHISPER_AVAILABLE:
            self._init_faster_whisper(model_id)
        elif TRANSFORMERS_AVAILABLE:
            self._init_transformers_whisper(model_id)
        else:
            raise RuntimeError("No STT backend available. Install faster-whisper or transformers.")
        
        self._initialized = True
    
    def _init_faster_whisper(self, model_id: str):
        """Initialize faster-whisper backend."""
        print(f"🔄 Loading faster-whisper model: {model_id}...")
        
        # Map model names
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
        
        # Determine compute type and device
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                compute_type = "float16"
            else:
                device = "cpu"
                compute_type = "int8"
        except:
            device = "cpu"
            compute_type = "int8"
        
        print(f"   Device: {device}, Compute: {compute_type}")
        
        self.model = FasterWhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )
        self.backend = "faster-whisper"
        print("✅ faster-whisper loaded (4x faster)!")
    
    def _init_transformers_whisper(self, model_id: str):
        """Initialize transformers whisper backend (fallback)."""
        print(f"🔄 Loading Whisper model: {model_id}...")
        
        # Ensure full model path
        if not model_id.startswith("openai/"):
            model_id = f"openai/whisper-{model_id}"
        
        import torch
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
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
        print("✅ Whisper model loaded!")
    
    def transcribe(self, audio: np.ndarray, language: str = "en") -> str:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio data as numpy array (float32, mono, 16kHz)
            language: Language code (e.g., 'en', 'es', 'fr')
            
        Returns:
            Transcribed text
        """
        if len(audio) == 0:
            return ""
        
        audio = audio.flatten().astype(np.float32)
        
        if self.backend == "faster-whisper":
            segments, _ = self.model.transcribe(
                audio,
                language=language,
                beam_size=5,
                vad_filter=True,  # Filter out silence
            )
            text = " ".join(segment.text for segment in segments)
        else:
            result = self.pipe(
                {"raw": audio, "sampling_rate": self.sample_rate},
                generate_kwargs={"language": language},
            )
            text = result["text"]
        
        return text.strip()


# Global instance
_stt_model = None


def get_stt_model(model_id: str = "base") -> WhisperSTT:
    """Get the singleton STT model instance."""
    global _stt_model
    if _stt_model is None:
        _stt_model = WhisperSTT(model_id)
    return _stt_model


def transcribe_audio(
    audio: np.ndarray,
    model_id: str = "base",
    model_type: str = "whisper",
    **kwargs
) -> str:
    """
    Transcribe audio to text.
    
    Args:
        audio: Audio data as numpy array
        model_id: Model identifier (e.g., "base", "small", "large-v3-turbo")
        model_type: Type of model (currently only "whisper" supported)
        
    Returns:
        Transcribed text
    """
    if model_type != "whisper":
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model = get_stt_model(model_id)
    return model.transcribe(audio)


# Test
if __name__ == "__main__":
    print("Testing STT module...")
    print(f"faster-whisper available: {FASTER_WHISPER_AVAILABLE}")
    print(f"transformers available: {TRANSFORMERS_AVAILABLE}")
    
    model = get_stt_model("base")
    print(f"Backend: {model.backend}")
    
    # Test with silence
    silence = np.zeros(16000, dtype=np.float32)
    result = model.transcribe(silence)
    print(f"Silence transcription: '{result}'")
