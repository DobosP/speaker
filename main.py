#!/usr/bin/env python3
"""
Cross-Platform Voice Assistant
Works on Windows, Mac, and Linux without OS-specific dependencies.

Usage:
    python main.py                      # Use defaults
    python main.py --list-devices       # List audio devices
    python main.py --input-device 1     # Use specific input device
    python main.py --llm-model llama3   # Use specific LLM model
    python main.py --stt-model openai/whisper-small  # Use specific STT model
"""
import argparse
import signal
import threading
import json
import os

from utils.audio import AudioRecorder, AudioPlayer, list_audio_devices
from utils.stt import get_stt_model, transcribe_audio
from utils.llm import get_llm


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
    Supports barge-in: interrupt the assistant by speaking.
    """
    
    def __init__(
        self,
        llm_model: str = "llama2",
        stt_model: str = "openai/whisper-base",
        input_device: int = None,
        output_device: int = None,
        vad_threshold: float = 0.01,
    ):
        self.llm_model = llm_model
        self.stt_model = stt_model
        self.input_device = input_device
        self.output_device = output_device
        self.vad_threshold = vad_threshold
        
        self._shutdown_event = threading.Event()
        self._llm = None
        self._recorder = None
        self._player = None
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
            
            # Get LLM response
            if self._llm:
                prompt = f"Respond to the following in a single, short sentence: {transcription}"
                response = self._llm.get_response(prompt)
                
                print(f"🤖 Assistant: {response}")
                
                # Log to file
                with open("live_transcript.txt", "a") as f:
                    f.write(f"Assistant: {response}\n")
                
                # Speak the response (with barge-in support)
                self._speak(response)
                
        except Exception as e:
            print(f"❌ Error processing speech: {e}")
    
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
        print("🎙️  Cross-Platform Voice Assistant")
        print("=" * 50)
        
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
            self._player = AudioPlayer(output_device=self.output_device)
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
        
        print("✅ Cleanup complete")
    
    def shutdown(self):
        """Signal shutdown from outside."""
        self._shutdown_event.set()


def main():
    parser = argparse.ArgumentParser(
        description="Cross-Platform Voice Assistant",
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
        default="llama2",
        help="LLM model to use (default: llama2)",
    )
    parser.add_argument(
        "--stt-model",
        type=str,
        default="openai/whisper-base",
        help="Speech-to-text model (default: openai/whisper-base)",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=None,
        help="Voice activity detection threshold (default: from config or 0.01)",
    )
    
    args = parser.parse_args()
    
    # List devices and exit
    if args.list_devices:
        list_audio_devices()
        return
    
    # Load config
    config = load_config()
    
    # Get VAD threshold from args or config
    vad_threshold = args.vad_threshold
    if vad_threshold is None:
        vad_threshold = config.get("vad_threshold", 0.01)
    
    # Create and run assistant
    assistant = VoiceAssistant(
        llm_model=args.llm_model,
        stt_model=args.stt_model,
        input_device=args.input_device,
        output_device=args.output_device,
        vad_threshold=vad_threshold,
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
