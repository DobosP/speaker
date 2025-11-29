#!/usr/bin/env python3
"""
Cross-Platform Voice Assistant with Memory
Works on Windows, Mac, and Linux without OS-specific dependencies.

Features:
- Speech-to-text with faster-whisper
- Text-to-speech with edge-tts
- Barge-in support (interrupt while assistant speaks)
- Multi-layer memory (recent, summaries, vector search)
- PostgreSQL + pgvector for persistent memory

Usage:
    python main.py                      # Use defaults
    python main.py --list-devices       # List audio devices
    python main.py --input-device 1     # Use specific input device
    python main.py --llm-model llama3   # Use specific LLM model
    python main.py --stt-model small    # Use specific STT model
    python main.py --no-memory          # Disable persistent memory
    python main.py --new-session        # Start fresh session
"""
import argparse
import signal
import threading
import json
import os

from utils.audio import AudioRecorder, AudioPlayer, list_audio_devices
from utils.stt import get_stt_model, transcribe_audio
from utils.llm import get_llm

# Memory is optional
try:
    from utils.memory import MemoryManager, POSTGRES_AVAILABLE, EMBEDDINGS_AVAILABLE
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    POSTGRES_AVAILABLE = False
    EMBEDDINGS_AVAILABLE = False


def load_config() -> dict:
    """Load configuration from config.json."""
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


class VoiceAssistant:
    """
    Cross-platform voice assistant that listens, transcribes, and responds.
    
    Features:
    - Barge-in: interrupt the assistant by speaking
    - Multi-layer memory with PostgreSQL + pgvector
    - Conversation history and semantic search
    """
    
    def __init__(
        self,
        llm_model: str = "llama2",
        stt_model: str = "base",
        input_device: int = None,
        output_device: int = None,
        vad_threshold: float = 0.01,
        tts_voice: str = "en-US",
        enable_memory: bool = True,
        session_id: str = None,
        db_url: str = None,
    ):
        self.llm_model = llm_model
        self.stt_model = stt_model
        self.input_device = input_device
        self.output_device = output_device
        self.vad_threshold = vad_threshold
        self.tts_voice = tts_voice
        self.enable_memory = enable_memory and MEMORY_AVAILABLE
        self.session_id = session_id
        self.db_url = db_url
        
        self._shutdown_event = threading.Event()
        self._llm = None
        self._recorder = None
        self._player = None
        self._memory = None
        self._speaking_lock = threading.Lock()
        self._response_thread = None
    
    def _on_barge_in(self):
        """Called when user speaks while assistant is talking (barge-in)."""
        if self._player:
            self._player.stop()
        if self._recorder:
            self._recorder.set_assistant_speaking(False)
    
    def _on_speech_detected(self, audio_data):
        """Called when the user finishes speaking. Runs processing in separate thread."""
        if self._shutdown_event.is_set():
            return
        
        # Process in a separate thread so audio processing can continue for barge-in
        self._response_thread = threading.Thread(
            target=self._process_and_respond,
            args=(audio_data,),
            daemon=True
        )
        self._response_thread.start()
    
    def _process_and_respond(self, audio_data):
        """Process audio and generate response (runs in separate thread)."""
        try:
            # Transcribe the audio
            transcription = transcribe_audio(audio_data, model_id=self.stt_model)
            
            if not transcription or transcription.strip() == "":
                return
            
            print(f"\n🎤 You: {transcription}")
            
            # Log to file
            with open("live_transcript.txt", "a") as f:
                f.write(f"You: {transcription}\n")
            
            # Check for stop command
            if "stop" in transcription.lower() or "quit" in transcription.lower():
                print("\n👋 Goodbye!")
                self._shutdown_event.set()
                return
            
            # Add user message to memory
            if self._memory:
                self._memory.add_message("user", transcription)
            
            # Get LLM response with memory context
            if self._llm:
                # Get context from memory
                context = None
                history = None
                
                if self._memory:
                    context = self._memory.get_context_for_llm(transcription)
                    history = self._memory.get_chat_history()
                
                # Get response with context
                response = self._llm.get_response(
                    transcription,
                    context=context,
                    history=history
                )
                
                print(f"🤖 Assistant: {response}")
                
                # Add assistant response to memory
                if self._memory:
                    self._memory.add_message("assistant", response)
                
                # Log to file
                with open("live_transcript.txt", "a") as f:
                    f.write(f"Assistant: {response}\n")
                
                # Speak the response (with barge-in support)
                self._speak(response)
                
        except Exception as e:
            print(f"❌ Error processing speech: {e}")
            import traceback
            traceback.print_exc()
    
    def _speak(self, text: str):
        """Speak text with barge-in support - user can interrupt by speaking."""
        if self._player and self._recorder:
            with self._speaking_lock:
                # Enable barge-in mode (keeps listening for interrupts)
                self._recorder.set_assistant_speaking(True)
                
                try:
                    self._player.speak(text)
                finally:
                    # Return to normal recording mode
                    self._recorder.set_assistant_speaking(False)
    
    def run(self):
        """Run the voice assistant."""
        print("\n" + "=" * 50)
        print("🎙️  Cross-Platform Voice Assistant with Memory")
        print("=" * 50)
        
        # Initialize Memory (optional)
        if self.enable_memory:
            print("\n📦 Initializing Memory...")
            try:
                self._memory = MemoryManager(
                    db_url=self.db_url,
                    session_id=self.session_id,
                )
                stats = self._memory.get_conversation_stats()
                print(f"✅ Memory initialized")
                if stats.get('total_messages', 0) > 0:
                    print(f"   📊 {stats['total_messages']} messages in history")
            except Exception as e:
                print(f"⚠️  Memory initialization failed: {e}")
                print("   Continuing without persistent memory.")
                self._memory = None
        else:
            print("\n📦 Memory: Disabled (use --enable-memory to activate)")
        
        # Initialize STT model (singleton - only loads once)
        print("\n📦 Initializing Speech-to-Text model...")
        get_stt_model(self.stt_model)
        
        # Initialize LLM
        print("\n📦 Initializing LLM...")
        try:
            self._llm = get_llm(llm_type="local", model=self.llm_model)
            print(f"✅ LLM initialized: {self.llm_model}")
        except Exception as e:
            print(f"❌ Could not initialize LLM: {e}")
            print("   Make sure Ollama is running: ollama serve")
            return
        
        # Initialize audio components
        print("\n📦 Initializing audio...")
        try:
            self._player = AudioPlayer(
                output_device=self.output_device,
                voice=self.tts_voice
            )
            self._recorder = AudioRecorder(
                callback=self._on_speech_detected,
                vad_threshold=self.vad_threshold,
                device=self.input_device,
                on_interrupt=self._on_barge_in,  # Enable barge-in
            )
            print("✅ Audio initialized (barge-in enabled)")
        except Exception as e:
            print(f"❌ Could not initialize audio: {e}")
            return
        
        # Start listening
        print("\n" + "-" * 50)
        print("🎤 Listening... (say 'stop' or 'quit' to exit)")
        print("💡 Tip: You can interrupt the assistant by speaking!")
        if self._memory:
            print("💾 Memory: Active (I remember our conversations)")
        print("-" * 50 + "\n")
        
        self._recorder.start()
        
        # Main loop
        try:
            while not self._shutdown_event.is_set():
                self._shutdown_event.wait(timeout=0.1)
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Clean up resources."""
        print("\n🧹 Cleaning up...")
        
        if self._recorder:
            self._recorder.stop()
        
        if self._player:
            self._player.cleanup()
        
        if self._memory:
            stats = self._memory.get_conversation_stats()
            print(f"   💾 Session saved ({stats.get('recent_messages', 0)} messages)")
            self._memory.close()
        
        print("✅ Cleanup complete")
    
    def shutdown(self):
        """Signal shutdown from outside."""
        self._shutdown_event.set()


def main():
    parser = argparse.ArgumentParser(
        description="Cross-Platform Voice Assistant with Memory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit",
    )
    parser.add_argument(
        "--input-device",
        type=int,
        default=None,
        help="Input device index (use --list-devices to see available devices)",
    )
    parser.add_argument(
        "--output-device",
        type=int,
        default=None,
        help="Output device index (use --list-devices to see available devices)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="LLM model to use (default: from config or llama2)",
    )
    parser.add_argument(
        "--stt-model",
        type=str,
        default=None,
        help="Speech-to-text model (default: from config or base)",
    )
    parser.add_argument(
        "--tts-voice",
        type=str,
        default=None,
        help="TTS voice (en-US, en-US-male, en-GB, en-AU)",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=None,
        help="Voice activity detection threshold (default: from config or 0.01)",
    )
    # Memory arguments
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable persistent memory (in-memory only)",
    )
    parser.add_argument(
        "--new-session",
        action="store_true",
        help="Start a new session (don't load previous history)",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Use a specific session ID (to continue a previous session)",
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=None,
        help="PostgreSQL database URL (default: from DATABASE_URL env or localhost)",
    )
    
    args = parser.parse_args()
    
    # List devices and exit
    if args.list_devices:
        list_audio_devices()
        return
    
    # Load config
    config = load_config()
    
    # Get settings from args or config
    llm_model = args.llm_model or config.get("llm_model", "llama2")
    stt_model = args.stt_model or config.get("stt_model", "base")
    tts_voice = args.tts_voice or config.get("tts_voice", "en-US")
    vad_threshold = args.vad_threshold or config.get("vad_threshold", 0.01)
    
    # Session ID (None = auto-generate)
    session_id = None if args.new_session else args.session_id
    
    # Create and run assistant
    assistant = VoiceAssistant(
        llm_model=llm_model,
        stt_model=stt_model,
        input_device=args.input_device or config.get("input_device"),
        output_device=args.output_device or config.get("output_device"),
        vad_threshold=vad_threshold,
        tts_voice=tts_voice,
        enable_memory=not args.no_memory,
        session_id=session_id,
        db_url=args.db_url,
    )
    
    # Handle signals for graceful shutdown
    def signal_handler(signum, frame):
        print("\n\n⚠️  Shutdown signal received")
        assistant.shutdown()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    assistant.run()


if __name__ == "__main__":
    main()
