"""
Voice Assistant Utilities

This package provides cross-platform audio recording, speech-to-text,
LLM integration, and persistent memory for building voice assistants.
"""

from .audio import AudioRecorder, AudioPlayer, list_audio_devices
from .stt import WhisperSTT, get_stt_model, transcribe_audio
from .llm import get_llm, LocalLLM

# Memory is optional (requires psycopg2)
try:
    from .memory import MemoryManager, Message, create_memory_manager
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False

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
    # Memory
    'MemoryManager',
    'Message',
    'create_memory_manager',
    'MEMORY_AVAILABLE',
]

