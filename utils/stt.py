"""
Speech-to-Text Model module with singleton pattern.
The model is loaded ONCE and reused for all transcriptions.
"""
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np
import warnings

# Suppress the chunk_length_s warning
warnings.filterwarnings("ignore", message=".*chunk_length_s.*")


class WhisperSTT:
    """
    Whisper Speech-to-Text model with singleton pattern.
    Model is loaded once and reused for efficiency.
    """
    
    _instance = None
    _model_id = None
    
    def __new__(cls, model_id: str = "openai/whisper-base"):
        """Singleton pattern - only create one instance per model."""
        if cls._instance is None or cls._model_id != model_id:
            cls._instance = super().__new__(cls)
            cls._model_id = model_id
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, model_id: str = "openai/whisper-base"):
        """Initialize the Whisper model (only runs once due to singleton)."""
        if self._initialized:
            return
        
        print(f"🔄 Loading Whisper model: {model_id}...")
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.sample_rate = 16000
        
        print(f"   Device: {self.device}")
        
        # Load model
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        # Create pipeline
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
            # Don't use chunking for short audio (live speech)
        )
        
        self._initialized = True
        print("✅ Whisper model loaded!")
    
    def transcribe(self, audio: np.ndarray, language: str = "english") -> str:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio data as numpy array (float32, mono, 16kHz)
            language: Language for transcription
            
        Returns:
            Transcribed text
        """
        if len(audio) == 0:
            return ""
        
        # Ensure audio is the right format
        audio = audio.flatten().astype(np.float32)
        
        # Transcribe
        result = self.pipe(
            {"raw": audio, "sampling_rate": self.sample_rate},
            generate_kwargs={"language": language},
        )
        
        return result["text"].strip()


# Global instance for easy access
_stt_model = None


def get_stt_model(model_id: str = "openai/whisper-base") -> WhisperSTT:
    """Get the singleton STT model instance."""
    global _stt_model
    if _stt_model is None:
        _stt_model = WhisperSTT(model_id)
    return _stt_model


def transcribe_audio(
    audio: np.ndarray,
    model_id: str = "openai/whisper-base",
    model_type: str = "whisper",
    **kwargs
) -> str:
    """
    Transcribe audio to text.
    
    This function maintains backward compatibility but uses singleton pattern internally.
    
    Args:
        audio: Audio data as numpy array
        model_id: Model identifier (e.g., "openai/whisper-base", "openai/whisper-small")
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
    import numpy as np
    
    print("Testing STT Model...")
    
    # First call - loads model
    model1 = get_stt_model()
    
    # Second call - reuses model (singleton)
    model2 = get_stt_model()
    
    print(f"Same instance: {model1 is model2}")  # Should be True
    
    # Test with silence (should return empty or minimal)
    silence = np.zeros(16000, dtype=np.float32)  # 1 second of silence
    result = model1.transcribe(silence)
    print(f"Silence transcription: '{result}'")
