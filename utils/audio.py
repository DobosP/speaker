"""
Cross-platform audio input/output module.
Works on Windows, Mac, and Linux without any OS-specific dependencies.
"""
import numpy as np
import sounddevice as sd
import threading
import queue
import time
import subprocess
from typing import Callable, Optional
import tempfile
import os

# For resampling when device doesn't support 16kHz
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

# For cross-platform audio playback
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

try:
    import edge_tts
    import asyncio
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

# Supertonic - Lightning fast on-device TTS
try:
    from supertonic import TTS as SupertonicTTS
    import soundfile as sf
    SUPERTONIC_AVAILABLE = True
except ImportError:
    SUPERTONIC_AVAILABLE = False


def list_audio_devices():
    """List all available audio devices with their indices."""
    devices = sd.query_devices()
    print("\n=== Available Audio Devices ===")
    for i, device in enumerate(devices):
        device_type = []
        if device['max_input_channels'] > 0:
            device_type.append("INPUT")
        if device['max_output_channels'] > 0:
            device_type.append("OUTPUT")
        type_str = "/".join(device_type) if device_type else "UNKNOWN"
        print(f"  [{i}] {device['name']} ({type_str})")
    
    default_input = sd.query_devices(kind='input')
    default_output = sd.query_devices(kind='output')
    print(f"\nDefault Input:  {default_input['name']}")
    print(f"Default Output: {default_output['name']}")
    print("=" * 35)
    return devices


def get_default_input_device() -> int:
    """Get the default input device index."""
    return sd.default.device[0]


def get_default_output_device() -> int:
    """Get the default output device index."""
    return sd.default.device[1]


class AudioRecorder:
    """
    Cross-platform audio recorder using sounddevice.
    Uses Voice Activity Detection (VAD) to detect when the user is speaking.
    Automatically handles sample rate conversion for Whisper compatibility.
    Supports barge-in: can detect voice even while assistant is speaking.
    """
    
    def __init__(
        self,
        callback: Callable[[np.ndarray], None],
        target_sample_rate: int = 16000,
        vad_threshold: float = 0.01,
        silence_duration: float = 1.5,
        device: Optional[int] = None,
        on_interrupt: Optional[Callable[[], None]] = None,
    ):
        """
        Initialize the audio recorder.
        
        Args:
            callback: Function to call when speech is detected (receives audio as numpy array at 16kHz)
            target_sample_rate: Target sample rate for output (16000 is optimal for Whisper)
            vad_threshold: RMS threshold for voice activity detection
            silence_duration: Seconds of silence to wait before considering speech complete
            device: Input device index (None = use default)
            on_interrupt: Callback when user speaks during assistant playback (barge-in)
        """
        self.callback = callback
        self.on_interrupt = on_interrupt
        self.target_sample_rate = target_sample_rate
        self.vad_threshold = vad_threshold
        self.silence_duration = silence_duration
        self.device = device
        
        # Get device's native sample rate
        self.device_sample_rate = self._get_device_sample_rate()
        self.needs_resampling = self.device_sample_rate != target_sample_rate
        
        if self.needs_resampling:
            if not LIBROSA_AVAILABLE:
                print(f"⚠️  Device uses {self.device_sample_rate}Hz, but librosa not available for resampling.")
                print("   Install with: pip install librosa")
                # Fall back to device rate
                self.target_sample_rate = self.device_sample_rate
                self.needs_resampling = False
            else:
                print(f"📊 Device sample rate: {self.device_sample_rate}Hz → resampling to {target_sample_rate}Hz")
        
        # State management
        self.is_recording = False
        self.assistant_is_speaking = False  # True when assistant is playing audio
        self._stop_event = threading.Event()
        
        # VAD state
        self._audio_buffer = np.array([], dtype=np.float32)
        self._is_speaking = False
        self._silence_start = None
        self._interrupt_triggered = False  # Prevent multiple interrupts
        
        # Audio queue for thread-safe communication
        self._audio_queue = queue.Queue()
        self._worker_thread = None
    
    # Backward compatibility
    @property
    def is_paused(self) -> bool:
        return self.assistant_is_speaking
    
    @is_paused.setter
    def is_paused(self, value: bool):
        self.assistant_is_speaking = value
    
    def _get_device_sample_rate(self) -> int:
        """Get the native sample rate for the recording device."""
        try:
            # Get the actual device index (resolve None to default)
            if self.device is None:
                # Get default input device index from sounddevice
                default_input_idx = sd.default.device[0]
                if default_input_idx is not None and default_input_idx >= 0:
                    self.device = default_input_idx
                else:
                    # Find first input device manually
                    devices = sd.query_devices()
                    for i, dev in enumerate(devices):
                        if dev['max_input_channels'] > 0:
                            self.device = i
                            break
            
            device_info = sd.query_devices(self.device)
            native_rate = int(device_info['default_samplerate'])
            return native_rate
        except Exception as e:
            print(f"⚠️  Could not query device sample rate: {e}")
            # Try to get any input device as fallback
            try:
                devices = sd.query_devices()
                for i, dev in enumerate(devices):
                    if dev['max_input_channels'] > 0:
                        self.device = i
                        return int(dev['default_samplerate'])
            except:
                pass
            return 44100  # Common fallback
    
    def _resample_audio(self, audio: np.ndarray) -> np.ndarray:
        """Resample audio from device rate to target rate."""
        if not self.needs_resampling:
            return audio
        
        return librosa.resample(
            audio,
            orig_sr=self.device_sample_rate,
            target_sr=self.target_sample_rate
        )
    
    @property
    def sample_rate(self) -> int:
        """Return the output sample rate (for backward compatibility)."""
        return self.target_sample_rate
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback function called by sounddevice for each audio block."""
        if status and "overflow" not in str(status):
            print(f"Audio status: {status}")
        
        # Always capture audio - we need it for barge-in detection
        audio_data = indata.flatten().astype(np.float32)
        self._audio_queue.put(audio_data)
    
    def _process_audio(self):
        """Worker thread that processes audio from the queue."""
        while not self._stop_event.is_set():
            try:
                audio_chunk = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            # Calculate RMS energy for VAD
            rms = np.sqrt(np.mean(audio_chunk ** 2))
            
            # Display audio level
            level_bar = "█" * int(min(rms * 500, 20))
            status = "🔊" if self.assistant_is_speaking else "🎤"
            print(f"\r{status} [{level_bar:<20}] {rms:.4f}", end="", flush=True)
            
            # === BARGE-IN MODE (assistant speaking) ===
            if self.assistant_is_speaking:
                # Only check for barge-in, don't record
                if rms > self.vad_threshold and not self._interrupt_triggered:
                    print("\r" + " " * 60 + "\r", end="", flush=True)
                    print(f"🛑 BARGE-IN! (RMS: {rms:.4f} > threshold: {self.vad_threshold:.4f})")
                    self._interrupt_triggered = True
                    if self.on_interrupt:
                        self.on_interrupt()
                # Skip normal processing during assistant speech
                continue
            
            # === NORMAL RECORDING MODE ===
            if self._is_speaking:
                # Only accumulate audio that's above a minimum energy threshold
                # This prevents recording pure silence at the end
                if rms > self.vad_threshold * 0.3:
                    self._audio_buffer = np.append(self._audio_buffer, audio_chunk)
                
                if rms < self.vad_threshold:
                    # Silence detected - start counting
                    if self._silence_start is None:
                        self._silence_start = time.time()
                    elif time.time() - self._silence_start > self.silence_duration:
                        self._finish_recording()
                else:
                    # Still speaking - reset silence timer
                    self._silence_start = None
            else:
                # Waiting for speech to start
                if rms > self.vad_threshold:
                    print("\r" + " " * 50 + "\r", end="", flush=True)
                    self._is_speaking = True
                    self._silence_start = None
                    self._audio_buffer = audio_chunk.copy()
    
    def _finish_recording(self):
        """Finish recording and send audio to callback."""
        print("\r" + " " * 50 + "\r", end="", flush=True)
        
        # Check if audio has enough energy to be real speech
        if len(self._audio_buffer) == 0:
            self._reset_recording_state()
            return
            
        avg_energy = np.sqrt(np.mean(self._audio_buffer ** 2))
        duration = len(self._audio_buffer) / self.device_sample_rate
        
        # Skip if average energy is too low (likely silence/noise, not speech)
        if avg_energy < self.vad_threshold * 0.5:
            print(f"⏭️  Skipped {duration:.1f}s (too quiet: {avg_energy:.4f})")
            self._reset_recording_state()
            return
        
        # Skip if too short
        min_duration = 0.5  # seconds
        if duration < min_duration:
            print(f"⏭️  Skipped {duration:.1f}s (too short)")
            self._reset_recording_state()
            return
        
        print(f"📝 Processing {duration:.1f}s of audio (energy: {avg_energy:.4f})...")
        
        # Resample to target rate before sending to callback
        resampled = self._resample_audio(self._audio_buffer.copy())
        self.callback(resampled)
        
        self._reset_recording_state()
    
    def _reset_recording_state(self):
        """Reset recording state."""
        self._audio_buffer = np.array([], dtype=np.float32)
        self._is_speaking = False
        self._silence_start = None
    
    def start(self):
        """Start recording audio."""
        if self.is_recording:
            return
        
        self.is_recording = True
        self._stop_event.clear()
        self._use_parec = False
        self._parec_process = None
        
        # Start the processing thread
        self._worker_thread = threading.Thread(target=self._process_audio, daemon=True)
        self._worker_thread.start()
        
        # Try sounddevice first, fall back to parec if it fails
        try:
            self._stream = sd.InputStream(
                device=self.device,
                channels=1,
                samplerate=self.device_sample_rate,
                dtype=np.float32,
                callback=self._audio_callback,
                blocksize=1024,
            )
            self._stream.start()
            print(f"🎤 Audio recording started at {self.device_sample_rate}Hz (sounddevice)")
        except Exception as e:
            # Fall back to parec (PulseAudio/PipeWire)
            print(f"⚠️  sounddevice failed, using parec (PipeWire)...")
            self._use_parec = True
            self._start_parec()
    
    def _start_parec(self):
        """Start recording using parec (PulseAudio/PipeWire fallback)."""
        # Use 16kHz directly since parec supports it
        self.device_sample_rate = 16000
        self.needs_resampling = False
        
        self._parec_process = subprocess.Popen(
            ['parec', '--rate=16000', '--channels=1', '--format=float32le'],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        
        # Start thread to read from parec
        self._parec_thread = threading.Thread(target=self._read_parec, daemon=True)
        self._parec_thread.start()
        print(f"🎤 Audio recording started at 16000Hz (parec/PipeWire)")
    
    def _read_parec(self):
        """Read audio data from parec process."""
        chunk_size = 1024 * 4  # 1024 samples * 4 bytes per float32
        while not self._stop_event.is_set() and self._parec_process.poll() is None:
            data = self._parec_process.stdout.read(chunk_size)
            if data:
                audio = np.frombuffer(data, dtype=np.float32)
                self._audio_queue.put(audio)
    
    def stop(self):
        """Stop recording audio."""
        if not self.is_recording:
            return
        
        self.is_recording = False
        self._stop_event.set()
        
        # Stop sounddevice stream
        if hasattr(self, '_stream') and self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except:
                pass
        
        # Stop parec process
        if hasattr(self, '_parec_process') and self._parec_process:
            try:
                self._parec_process.terminate()
                self._parec_process.wait(timeout=1)
            except:
                pass
        
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        
        print("\n🛑 Audio recording stopped")
    
    def set_assistant_speaking(self, speaking: bool):
        """
        Set whether the assistant is currently speaking.
        When True, enables barge-in detection.
        When False, resumes normal recording.
        """
        if speaking:
            # Assistant started speaking - enable barge-in mode
            self.assistant_is_speaking = True
            self._interrupt_triggered = False
            # Clear any in-progress recording
            self._is_speaking = False
            self._silence_start = None
            self._audio_buffer = np.array([], dtype=np.float32)
            print(f"\n💬 Speaking... (interrupt by talking, threshold > {self.vad_threshold:.3f})")
        else:
            # Assistant stopped speaking - fully reset to normal mode
            self.assistant_is_speaking = False
            self._interrupt_triggered = False
            self._is_speaking = False
            self._silence_start = None
            self._audio_buffer = np.array([], dtype=np.float32)
            print("\n🎤 Ready for input...")
    
    def pause(self):
        """Pause recording (backward compatibility - use set_assistant_speaking instead)."""
        self.set_assistant_speaking(True)
    
    def resume(self):
        """Resume recording (backward compatibility - use set_assistant_speaking instead)."""
        self.set_assistant_speaking(False)


class AudioPlayer:
    """
    Cross-platform audio player with multiple TTS backends.
    Priority: Supertonic (fast local) > edge-tts (online) > gTTS (fallback)
    Supports interruption for barge-in functionality.
    """
    
    # Available edge-tts voices (natural sounding)
    EDGE_VOICES = {
        "en-US": "en-US-AriaNeural",      # Female, natural
        "en-US-male": "en-US-GuyNeural",  # Male, natural
        "en-GB": "en-GB-SoniaNeural",     # British female
        "en-AU": "en-AU-NatashaNeural",   # Australian female
    }
    
    # Supertonic voice mapping
    SUPERTONIC_VOICES = {
        "en-US": "M1",           # Male voice
        "en-US-male": "M1",      # Male voice
        "en-US-female": "F1",    # Female voice (if available)
    }
    
    def __init__(self, output_device: Optional[int] = None, voice: str = "en-US", prefer_local: bool = True):
        """
        Initialize the audio player.
        
        Args:
            output_device: Output device index (pygame uses system default)
            voice: Voice to use (e.g., "en-US", "en-US-male", "en-GB")
            prefer_local: If True, prefer Supertonic (local) over edge-tts (online)
        """
        if not PYGAME_AVAILABLE:
            raise RuntimeError("pygame is required for audio playback. Install with: pip install pygame")
        
        self.output_device = output_device
        self._is_playing = False
        self._current_file = None
        self.voice = voice
        self._supertonic_tts = None
        self._supertonic_style = None
        
        # Determine TTS backend (priority: supertonic > edge-tts > gtts)
        if prefer_local and SUPERTONIC_AVAILABLE:
            self.tts_backend = "supertonic"
            self._init_supertonic()
        elif EDGE_TTS_AVAILABLE:
            self.tts_backend = "edge-tts"
            self.voice = self.EDGE_VOICES.get(voice, voice)
            print(f"🔊 TTS: edge-tts (voice: {self.voice})")
        elif GTTS_AVAILABLE:
            self.tts_backend = "gtts"
            print("🔊 TTS: gTTS (fallback)")
        else:
            self.tts_backend = None
            print("⚠️  No TTS backend available!")
        
        # Initialize pygame mixer
        pygame.mixer.init()
    
    def _init_supertonic(self):
        """Initialize Supertonic TTS engine."""
        try:
            print("🔊 Initializing Supertonic TTS (fast, on-device)...")
            self._supertonic_tts = SupertonicTTS(auto_download=True)
            voice_name = self.SUPERTONIC_VOICES.get(self.voice, "M1")
            self._supertonic_style = self._supertonic_tts.get_voice_style(voice_name=voice_name)
            print(f"✅ TTS: Supertonic (voice: {voice_name}, ~10x realtime)")
        except Exception as e:
            print(f"⚠️  Supertonic init failed: {e}, falling back...")
            self.tts_backend = "edge-tts" if EDGE_TTS_AVAILABLE else "gtts" if GTTS_AVAILABLE else None
            if self.tts_backend == "edge-tts":
                self.voice = self.EDGE_VOICES.get(self.voice, self.voice)
                print(f"🔊 TTS: edge-tts (voice: {self.voice})")
    
    def _supertonic_synthesize(self, text: str, output_file: str):
        """Synthesize speech using Supertonic (fast, on-device)."""
        audio, duration = self._supertonic_tts.synthesize(text, voice_style=self._supertonic_style)
        # Audio is shape (1, samples) - flatten it
        audio = audio.flatten()
        # Save as WAV (Supertonic uses 44100 Hz)
        sf.write(output_file, audio, 44100)
    
    async def _edge_tts_synthesize(self, text: str, output_file: str):
        """Synthesize speech using edge-tts."""
        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(output_file)
    
    def _synthesize_speech(self, text: str, output_file: str):
        """Synthesize speech to file using available backend."""
        if self.tts_backend == "supertonic":
            self._supertonic_synthesize(text, output_file)
        elif self.tts_backend == "edge-tts":
            # Run async edge-tts in sync context
            asyncio.run(self._edge_tts_synthesize(text, output_file))
        elif self.tts_backend == "gtts":
            tts = gTTS(text=text, lang='en')
            tts.save(output_file)
        else:
            raise RuntimeError("No TTS backend available")
    
    def speak(self, text: str, on_start: Callable = None, on_end: Callable = None) -> bool:
        """
        Convert text to speech and play it.
        
        Args:
            text: Text to speak
            on_start: Callback when speech starts
            on_end: Callback when speech ends
            
        Returns:
            True if speech completed, False if interrupted
        """
        if self.tts_backend is None:
            print(f"[TTS not available] {text}")
            return True
        
        try:
            # Create temporary audio file (wav for supertonic, mp3 for others)
            suffix = '.wav' if self.tts_backend == "supertonic" else '.mp3'
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as fp:
                self._current_file = fp.name
            
            # Synthesize speech
            self._synthesize_speech(text, self._current_file)
            
            # Signal start
            if on_start:
                on_start()
            
            self._is_playing = True
            
            # Play audio
            pygame.mixer.music.load(self._current_file)
            pygame.mixer.music.play()
            
            # Wait for playback to complete or be interrupted
            while pygame.mixer.music.get_busy() and self._is_playing:
                time.sleep(0.05)
            
            completed = not pygame.mixer.music.get_busy()
            
        except Exception as e:
            print(f"Error in speak: {e}")
            completed = False
        finally:
            self._is_playing = False
            
            # Signal end
            if on_end:
                on_end()
            
            # Clean up temp file
            if self._current_file and os.path.exists(self._current_file):
                try:
                    os.remove(self._current_file)
                except:
                    pass
                self._current_file = None
        
        return completed
    
    def stop(self):
        """Stop current playback (for barge-in)."""
        if self._is_playing:
            self._is_playing = False
            pygame.mixer.music.stop()
            print("--- Speech interrupted ---")
    
    def is_playing(self) -> bool:
        """Check if currently playing audio."""
        return self._is_playing and pygame.mixer.music.get_busy()
    
    def cleanup(self):
        """Clean up resources."""
        pygame.mixer.quit()


# Quick test
if __name__ == "__main__":
    print("Testing cross-platform audio module...")
    list_audio_devices()
    
    def on_speech(audio):
        print(f"\n[Got {len(audio) / 16000:.1f}s of audio]")
    
    recorder = AudioRecorder(callback=on_speech)
    recorder.start()
    
    try:
        input("Press Enter to stop...\n")
    except KeyboardInterrupt:
        pass
    
    recorder.stop()

