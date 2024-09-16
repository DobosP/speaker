import os
import time
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from whisper_jax import FlaxWhisperPipline
from openai import OpenAI

class STTModel:
    def transcribe(self, audio_path, **kwargs):
        raise NotImplementedError("Subclasses should implement this method")

class WhisperModel(STTModel):
    def __init__(self, model_id="openai/whisper-large-v3", device=None, torch_dtype=None):
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

    def transcribe(self, audio_path, **kwargs):
        result = self.pipe(audio_path, generate_kwargs={"language": "english"})
        return result["text"]

class JAXWhisperModel(STTModel):
    def __init__(self, model_id="openai/whisper-large-v3"):
        self.pipeline = FlaxWhisperPipline(model_id)

    def transcribe(self, audio_path, **kwargs):
        return self.pipeline(audio_path)

class OpenAIWhisperModel(STTModel):
    def __init__(self, model_id="whisper-2"):
        self.model = OpenAI()
        self.model_id = model_id

    def transcribe(self, audio_path, **kwargs):
        result = self.model.audio.transcribe(audio_path, model=self.model_id)
        return result["text"]

class CustomSTTModel(STTModel):
    def __init__(self, custom_model):
        self.custom_model = custom_model

    def transcribe(self, audio_path, **kwargs):
        return self.custom_model.transcribe(audio_path, **kwargs)
def transcribe_audio(audio_path, model_type="whisper", **kwargs):
    model_classes = {
        "whisper": WhisperModel,
        "jax_whisper": JAXWhisperModel,
        "openai_whisper": OpenAIWhisperModel,
        "custom": CustomSTTModel
    }

    if model_type not in model_classes:
        raise ValueError(f"Unsupported model type: {model_type}")

    model = model_classes[model_type](**kwargs)
    return model.transcribe(audio_path)

# Example usage
if __name__ == "__main__":
    audio_path = '/home/dobo/projects/speaker/recordings/2_min_sample.mp3'
    model_type = "whisper"  # Change to "jax_whisper", "openai_whisper", or "custom" as needed
    transcription = transcribe_audio(audio_path, model_type=model_type)
    print(transcription)