"""Voice-assistant utilities.

Post-refactor this package retains only the Postgres-backed smart memory
subsystem; the legacy audio/STT/TTS/LLM modules were removed in favour of the
``core/`` runtime (see ``docs/target_architecture.md``).
"""

try:
    from .memory import MemoryManager, Message, create_memory_manager

    MEMORY_AVAILABLE = True
except ImportError:  # psycopg2 / numpy optional at import time
    MEMORY_AVAILABLE = False

__all__ = [
    "MemoryManager",
    "Message",
    "create_memory_manager",
    "MEMORY_AVAILABLE",
]
