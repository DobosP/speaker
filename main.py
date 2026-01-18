#!/usr/bin/env python3
"""
Cross-Platform Voice Assistant with Memory
Works on Windows, Mac, and Linux without OS-specific dependencies.

Features:
- Speech-to-text with faster-whisper
- Text-to-speech with Supertonic (fast, on-device)
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
# Disable CUDA to avoid cuDNN compatibility issues (must be before torch import)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse
import signal
import threading
import json
import queue

from utils.audio import AudioRecorder, AudioPlayer, list_audio_devices
from utils.stt import get_stt_model, transcribe_audio
from utils.llm import get_llm
from utils.dialogue_controller import DialogueController, BargeInInfo

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
        barge_in_pre_roll_sec: float = 0.3,
        barge_in_min_speech_sec: float = 0.2,
        barge_in_rms_ratio: float = 2.0,
        barge_in_cooldown_sec: float = 0.5,
        barge_in_use_webrtcvad: bool = True,
        mode: str = "asr",
        controller_config: dict | None = None,
        partial_interval_sec: float = 1.0,
        adaptive_vad: bool = False,
        vad_noise_multiplier: float = 2.5,
        vad_noise_floor_min: float = 0.003,
        calibrate_on_start: bool = True,
        calibrate_duration_sec: float = 2.5,
        aec_enabled: bool = True,
        aec_strength: float = 0.8,
        aec_max_ref_sec: float = 20.0,
        simple_voiced_fallback: bool = True,
        barge_in_debug: bool = False,
        barge_in_min_delay_sec: float = 0.4,
        echo_corr_threshold: float = 0.6,
        barge_in_min_delay_after_ref_sec: float = 0.6,
        barge_in_min_rms_ratio: float = 1.5,
        stop_mode: str = "exact",
        stop_phrases: tuple[str, ...] = ("stop", "quit", "exit"),
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
        self.barge_in_pre_roll_sec = barge_in_pre_roll_sec
        self.barge_in_min_speech_sec = barge_in_min_speech_sec
        self.barge_in_rms_ratio = barge_in_rms_ratio
        self.barge_in_cooldown_sec = barge_in_cooldown_sec
        self.barge_in_use_webrtcvad = barge_in_use_webrtcvad
        self.mode = mode
        self.controller_config = controller_config or {}
        self.partial_interval_sec = partial_interval_sec
        self.adaptive_vad = adaptive_vad
        self.vad_noise_multiplier = vad_noise_multiplier
        self.vad_noise_floor_min = vad_noise_floor_min
        self.calibrate_on_start = calibrate_on_start
        self.calibrate_duration_sec = calibrate_duration_sec
        self.aec_enabled = aec_enabled
        self.aec_strength = aec_strength
        self.aec_max_ref_sec = aec_max_ref_sec
        self.simple_voiced_fallback = simple_voiced_fallback
        self.barge_in_debug = barge_in_debug
        self.barge_in_min_delay_sec = barge_in_min_delay_sec
        self.echo_corr_threshold = echo_corr_threshold
        self.barge_in_min_delay_after_ref_sec = barge_in_min_delay_after_ref_sec
        self.barge_in_min_rms_ratio = barge_in_min_rms_ratio
        self._stop_mode = stop_mode
        self._stop_phrases = stop_phrases
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
        self._controller = None
        self._partial_audio_queue = None
        self._partial_worker = None
    
    def _on_barge_in(self, info=None):
        """Called when user speaks while assistant is talking (barge-in)."""
        should_stop = True
        if self._controller and info:
            barge_info = BargeInInfo(
                rms=info.get("rms", 0.0),
                threshold=info.get("threshold", 0.0),
                voiced=bool(info.get("voiced", False)),
                duration_sec=info.get("duration_sec", 0.0),
                timestamp=info.get("timestamp", 0.0),
                echo=bool(info.get("echo", False)),
            )
            should_stop = self._controller.should_stop_speaking(barge_info)

        if should_stop and self._player:
            self._player.stop()
        elif self.barge_in_debug and self._controller:
            print(f"ℹ️  Barge-in ignored: {self._controller.last_reason()}")
        return should_stop
    
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
            if self._controller and self._controller.should_ignore_transcript(transcription):
                return
            
            print(f"\n🎤 You: {transcription}")
            
            # Log to file
            with open("live_transcript.txt", "a") as f:
                f.write(f"You: {transcription}\n")
            
            # Check for stop command (exact or prefix only)
            normalized = transcription.strip().lower()
            stop_phrases = self._stop_phrases
            if self._stop_mode == "exact":
                matched = normalized in stop_phrases
            else:
                matched = any(normalized.startswith(p) for p in stop_phrases)
            if matched:
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
                if self._controller:
                    self._controller.on_assistant_text(text)
                    self._controller.on_tts_start()
                
                try:
                    self._player.speak(
                        text,
                        on_start=self._on_tts_start,
                    )
                finally:
                    # Return to normal recording mode
                    if not self._recorder.is_barge_in_active():
                        self._recorder.set_assistant_speaking(False)
                    if self._controller:
                        self._controller.on_tts_end()

    def _on_tts_start(self, audio_data=None, sample_rate=None):
        """Provide TTS audio reference to the recorder for echo cancellation."""
        if self._recorder and audio_data is not None and sample_rate:
            self._recorder.set_echo_reference(audio_data, sample_rate)

    def _on_partial_audio(self, audio_data):
        """Receive partial audio for streaming ASR in controller mode."""
        if not self._partial_audio_queue:
            return
        try:
            # Drop if queue is full to avoid lag
            self._partial_audio_queue.put_nowait(audio_data)
        except queue.Full:
            pass

    def _partial_transcribe_loop(self):
        """Background worker for streaming ASR partials."""
        while not self._shutdown_event.is_set():
            try:
                audio_data = self._partial_audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                text = transcribe_audio(audio_data, model_id=self.stt_model)
                if self._controller:
                    self._controller.on_partial_transcript(text)
            except Exception:
                continue
    
    def run(self):
        """Run the voice assistant."""
        print("\n" + "=" * 50)
        print("🎙️  Cross-Platform Voice Assistant with Memory")
        print("=" * 50)
        if self.mode == "controller":
            print("🧭 Dialogue Controller: Enabled (hybrid streaming ASR + LLM + TTS)")
        
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
            if self.mode == "controller":
                self._controller = DialogueController(**self.controller_config)
                self._partial_audio_queue = queue.Queue(maxsize=2)
                self._partial_worker = threading.Thread(
                    target=self._partial_transcribe_loop,
                    daemon=True
                )
                self._partial_worker.start()
            self._recorder = AudioRecorder(
                callback=self._on_speech_detected,
                vad_threshold=self.vad_threshold,
                device=self.input_device,
                on_interrupt=self._on_barge_in,  # Enable barge-in
                barge_in_pre_roll_sec=self.barge_in_pre_roll_sec,
                barge_in_min_speech_sec=self.barge_in_min_speech_sec,
                barge_in_rms_ratio=self.barge_in_rms_ratio,
                barge_in_cooldown_sec=self.barge_in_cooldown_sec,
                use_webrtcvad=self.barge_in_use_webrtcvad,
                adaptive_vad=self.adaptive_vad,
                vad_noise_multiplier=self.vad_noise_multiplier,
                vad_noise_floor_min=self.vad_noise_floor_min,
                aec_enabled=self.aec_enabled,
                aec_strength=self.aec_strength,
                aec_max_ref_sec=self.aec_max_ref_sec,
                simple_voiced_fallback=self.simple_voiced_fallback,
                barge_in_min_delay_sec=self.barge_in_min_delay_sec,
                barge_in_min_delay_after_ref_sec=self.barge_in_min_delay_after_ref_sec,
                barge_in_min_rms_ratio=self.barge_in_min_rms_ratio,
                echo_corr_threshold=self.echo_corr_threshold,
                partial_callback=self._on_partial_audio if self.mode == "controller" else None,
                partial_interval_sec=self.partial_interval_sec,
            )
            print("✅ Audio initialized (barge-in enabled)")
        except Exception as e:
            print(f"❌ Could not initialize audio: {e}")
            return

        # Optional startup calibration for adaptive VAD/barge-in
        if self.adaptive_vad and self.calibrate_on_start:
            print("\n🎛️  Calibrating ambient noise (please stay silent)...")
            result = self._recorder.calibrate(self.calibrate_duration_sec)
            if result.get("calibrated"):
                print(
                    f"✅ Calibration complete (noise_floor={result.get('noise_floor'):.4f}, "
                    f"samples={result.get('samples')})"
                )
            else:
                print(f"⚠️  Calibration skipped: {result.get('reason')}")
        
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
    parser.add_argument(
        "--mode",
        type=str,
        choices=["asr", "controller"],
        default=None,
        help="Run mode: 'asr' (default) or 'controller' (hybrid streaming + dialogue controller)",
    )
    parser.add_argument(
        "--partial-interval",
        type=float,
        default=None,
        help="Seconds between partial ASR callbacks in controller mode (default: config or 1.0)",
    )
    parser.add_argument(
        "--controller-min-interrupt-delay",
        type=float,
        default=None,
        help="Minimum seconds after TTS starts before allowing barge-in (default: config or 0.2)",
    )
    parser.add_argument(
        "--controller-min-barge-in",
        type=float,
        default=None,
        help="Minimum barge-in duration to stop TTS (default: config or 0.2)",
    )
    parser.add_argument(
        "--controller-min-partial-chars",
        type=int,
        default=None,
        help="Minimum partial transcript length to accept barge-in (default: config or 3)",
    )
    parser.add_argument(
        "--controller-max-partial-age",
        type=float,
        default=None,
        help="Max age of partial transcript in seconds (default: config or 1.5)",
    )
    parser.add_argument(
        "--controller-echo-similarity",
        type=float,
        default=None,
        help="Similarity threshold to treat partials as TTS echo (default: config or 0.7)",
    )
    parser.add_argument(
        "--controller-allow-rms-fallback",
        action="store_true",
        default=None,
        help="Allow RMS-only fallback barge-in (default: config or false)",
    )
    parser.add_argument(
        "--controller-require-partial",
        action="store_true",
        default=None,
        help="Require partial transcript for barge-in (default: config or true)",
    )
    parser.add_argument(
        "--controller-strong-voiced-multiplier",
        type=float,
        default=None,
        help="Require voiced RMS over threshold by multiplier (default: config or 2.5)",
    )
    parser.add_argument(
        "--controller-ignore-phrases",
        type=str,
        default=None,
        help="Comma-separated phrases to ignore (default: config or 'uh,um,')",
    )
    parser.add_argument(
        "--adaptive-vad",
        action="store_true",
        default=None,
        help="Enable adaptive VAD thresholding from ambient noise",
    )
    parser.add_argument(
        "--vad-noise-multiplier",
        type=float,
        default=None,
        help="Adaptive VAD noise multiplier (default: config or 2.5)",
    )
    parser.add_argument(
        "--vad-noise-floor-min",
        type=float,
        default=None,
        help="Adaptive VAD minimum noise floor (default: config or 0.003)",
    )
    parser.add_argument(
        "--calibrate-on-start",
        action="store_true",
        default=None,
        help="Calibrate ambient noise at startup (default: config or true)",
    )
    parser.add_argument(
        "--calibrate-duration",
        type=float,
        default=None,
        help="Seconds to sample ambient noise for calibration (default: config or 2.5)",
    )
    parser.add_argument(
        "--aec-enabled",
        action="store_true",
        default=None,
        help="Enable lightweight echo cancellation using TTS reference (default: config or true)",
    )
    parser.add_argument(
        "--aec-strength",
        type=float,
        default=None,
        help="AEC subtraction strength 0-1 (default: config or 0.8)",
    )
    parser.add_argument(
        "--aec-max-ref-sec",
        type=float,
        default=None,
        help="Max seconds of TTS reference to keep (default: config or 20.0)",
    )
    parser.add_argument(
        "--simple-voiced-fallback",
        action="store_true",
        default=None,
        help="Enable simple voiced heuristic when WebRTC VAD unavailable",
    )
    parser.add_argument(
        "--barge-in-debug",
        action="store_true",
        default=None,
        help="Log why barge-in was ignored (controller mode)",
    )
    parser.add_argument(
        "--barge-in-min-delay",
        type=float,
        default=None,
        help="Minimum delay after TTS starts before barge-in (default: config or 0.4)",
    )
    parser.add_argument(
        "--echo-corr-threshold",
        type=float,
        default=None,
        help="Echo correlation threshold to block barge-in (default: config or 0.6)",
    )
    parser.add_argument(
        "--barge-in-min-delay-after-ref",
        type=float,
        default=None,
        help="Min delay after TTS reference set (default: config or 0.6)",
    )
    parser.add_argument(
        "--barge-in-min-rms-ratio",
        type=float,
        default=None,
        help="Require RMS above noise floor by ratio (default: config or 1.5)",
    )
    parser.add_argument(
        "--stop-mode",
        type=str,
        choices=["exact", "prefix"],
        default=None,
        help="Stop command matching: exact or prefix (default: config or exact)",
    )
    parser.add_argument(
        "--stop-phrases",
        type=str,
        default=None,
        help="Comma-separated stop phrases (default: config or 'stop,quit,exit')",
    )
    parser.add_argument(
        "--barge-in-pre-roll",
        type=float,
        default=None,
        help="Seconds of audio to keep before barge-in trigger (default: config or 0.3)",
    )
    parser.add_argument(
        "--barge-in-min-speech",
        type=float,
        default=None,
        help="Minimum speech duration to trigger barge-in (default: config or 0.2)",
    )
    parser.add_argument(
        "--barge-in-rms-ratio",
        type=float,
        default=None,
        help="RMS ratio above noise floor to trigger barge-in (default: config or 2.0)",
    )
    parser.add_argument(
        "--barge-in-cooldown",
        type=float,
        default=None,
        help="Cooldown between barge-in triggers in seconds (default: config or 0.5)",
    )
    parser.add_argument(
        "--barge-in-use-webrtcvad",
        action="store_true",
        default=None,
        help="Use WebRTC VAD for barge-in if available",
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
    mode = args.mode or config.get("mode", "asr")
    partial_interval_sec = args.partial_interval or config.get("partial_interval_sec", 1.0)
    adaptive_vad = (
        args.adaptive_vad
        if args.adaptive_vad is not None
        else config.get("adaptive_vad", False)
    )
    vad_noise_multiplier = args.vad_noise_multiplier or config.get("vad_noise_multiplier", 2.5)
    vad_noise_floor_min = args.vad_noise_floor_min or config.get("vad_noise_floor_min", 0.003)
    calibrate_on_start = (
        args.calibrate_on_start
        if args.calibrate_on_start is not None
        else config.get("calibrate_on_start", True)
    )
    calibrate_duration_sec = args.calibrate_duration or config.get("calibrate_duration_sec", 2.5)
    aec_enabled = (
        args.aec_enabled
        if args.aec_enabled is not None
        else config.get("aec_enabled", True)
    )
    aec_strength = args.aec_strength or config.get("aec_strength", 0.8)
    aec_max_ref_sec = args.aec_max_ref_sec or config.get("aec_max_ref_sec", 20.0)
    simple_voiced_fallback = (
        args.simple_voiced_fallback
        if args.simple_voiced_fallback is not None
        else config.get("simple_voiced_fallback", True)
    )
    barge_in_debug = (
        args.barge_in_debug
        if args.barge_in_debug is not None
        else config.get("barge_in_debug", False)
    )
    barge_in_min_delay_sec = args.barge_in_min_delay or config.get("barge_in_min_delay_sec", 0.4)
    echo_corr_threshold = args.echo_corr_threshold or config.get("echo_corr_threshold", 0.6)
    barge_in_min_delay_after_ref_sec = (
        args.barge_in_min_delay_after_ref
        or config.get("barge_in_min_delay_after_ref_sec", 0.6)
    )
    barge_in_min_rms_ratio = args.barge_in_min_rms_ratio or config.get("barge_in_min_rms_ratio", 1.5)
    stop_mode = args.stop_mode or config.get("stop_mode", "exact")
    stop_phrases = config.get("stop_phrases", ["stop", "quit", "exit"])
    if args.stop_phrases is not None:
        stop_phrases = [p.strip().lower() for p in args.stop_phrases.split(",") if p.strip()]
    barge_in_pre_roll_sec = args.barge_in_pre_roll or config.get("barge_in_pre_roll_sec", 0.3)
    barge_in_min_speech_sec = args.barge_in_min_speech or config.get("barge_in_min_speech_sec", 0.2)
    barge_in_rms_ratio = args.barge_in_rms_ratio or config.get("barge_in_rms_ratio", 2.0)
    barge_in_cooldown_sec = args.barge_in_cooldown or config.get("barge_in_cooldown_sec", 0.5)
    barge_in_use_webrtcvad = (
        args.barge_in_use_webrtcvad
        if args.barge_in_use_webrtcvad is not None
        else config.get("barge_in_use_webrtcvad", True)
    )
    ignore_phrases = config.get("controller_ignore_phrases", ["", ".", "uh", "um"])
    if args.controller_ignore_phrases is not None:
        ignore_phrases = [p.strip().lower() for p in args.controller_ignore_phrases.split(",")]
    controller_config = {
        "min_interrupt_delay_sec": args.controller_min_interrupt_delay
        or config.get("controller_min_interrupt_delay_sec", 0.2),
        "min_barge_in_sec": args.controller_min_barge_in
        or config.get("controller_min_barge_in_sec", 0.2),
        "min_partial_chars": args.controller_min_partial_chars
        or config.get("controller_min_partial_chars", 3),
        "max_partial_age_sec": args.controller_max_partial_age
        or config.get("controller_max_partial_age_sec", 1.5),
        "echo_similarity_threshold": args.controller_echo_similarity
        or config.get("controller_echo_similarity_threshold", 0.7),
        "allow_rms_fallback": (
            args.controller_allow_rms_fallback
            if args.controller_allow_rms_fallback is not None
            else config.get("controller_allow_rms_fallback", False)
        ),
        "require_partial_for_barge_in": (
            args.controller_require_partial
            if args.controller_require_partial is not None
            else config.get("controller_require_partial_for_barge_in", True)
        ),
        "strong_voiced_multiplier": args.controller_strong_voiced_multiplier
        or config.get("controller_strong_voiced_multiplier", 2.5),
        "ignore_phrases": tuple(ignore_phrases),
    }
    
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
        barge_in_pre_roll_sec=barge_in_pre_roll_sec,
        barge_in_min_speech_sec=barge_in_min_speech_sec,
        barge_in_rms_ratio=barge_in_rms_ratio,
        barge_in_cooldown_sec=barge_in_cooldown_sec,
        barge_in_use_webrtcvad=barge_in_use_webrtcvad,
        mode=mode,
        controller_config=controller_config,
        partial_interval_sec=partial_interval_sec,
        adaptive_vad=adaptive_vad,
        vad_noise_multiplier=vad_noise_multiplier,
        vad_noise_floor_min=vad_noise_floor_min,
        calibrate_on_start=calibrate_on_start,
        calibrate_duration_sec=calibrate_duration_sec,
        aec_enabled=aec_enabled,
        aec_strength=aec_strength,
        aec_max_ref_sec=aec_max_ref_sec,
        simple_voiced_fallback=simple_voiced_fallback,
        barge_in_debug=barge_in_debug,
        barge_in_min_delay_sec=barge_in_min_delay_sec,
        echo_corr_threshold=echo_corr_threshold,
        barge_in_min_delay_after_ref_sec=barge_in_min_delay_after_ref_sec,
        barge_in_min_rms_ratio=barge_in_min_rms_ratio,
        stop_mode=stop_mode,
        stop_phrases=tuple(stop_phrases),
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
