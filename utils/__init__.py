"""
Voice Assistant Utilities

This package provides cross-platform audio recording, speech-to-text,
and LLM integration for building voice assistants.
"""

from .audio import AudioRecorder, AudioPlayer, list_audio_devices
from .stt import WhisperSTT, get_stt_model, transcribe_audio
from .llm import get_llm, LocalLLM

__all__ = [
    # Audio
    'AudioRecorder',
    'AudioPlayer', 
    'list_audio_devices',
    # Speech-to-Text
    'WhisperSTT',
    'get_stt_model',
    'transcribe_audio',
    # LLM
    'get_llm',
    'LocalLLM',
]

