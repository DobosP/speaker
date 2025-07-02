import os
import time
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np

class STTModel:
    def transcribe(self, audio_data, **kwargs):
        raise NotImplementedError("Subclasses should implement this method")

class WhisperModel(STTModel):
    def __init__(self, model_id="openai/whisper-base", device=None, torch_dtype=None, sr=16000):
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype or (torch.float16 if torch.cuda.is_available() else torch.float32)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
        self.sr = sr

    def transcribe(self, audio_data, **kwargs):
        result = self.pipe(audio_data.flatten(), generate_kwargs={"language": "english"})
        return result["text"]

def transcribe_audio(audio_data, model_type="whisper", model_id="openai/whisper-base", **kwargs):
    if model_type == "whisper":
        model = WhisperModel(model_id=model_id, **kwargs)
        return model.transcribe(audio_data)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

# Example usage
if __name__ == "__main__":
    # This example will not work as it expects an audio file path
    # and the new implementation expects audio data.
    pass