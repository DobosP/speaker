import subprocess
import numpy as np
import queue
import threading
import time
import signal

class LiveListener:
    def __init__(self, callback_func, interrupt_func, block_size=512, vad_threshold=0.02, silence_duration=1.5):
        self.sr = 16000
        self.block_size = block_size
        self.callback_func = callback_func
        self.interrupt_func = interrupt_func
        self.q = queue.Queue(maxsize=500)
        self.proc = None
        self.running = False

        # State management
        self.base_vad_threshold = vad_threshold
        self.vad_threshold = vad_threshold
        self.silence_duration = silence_duration
        self.audio_buffer = np.array([], dtype=np.float32)
        self.is_speaking = False
        self.silence_started_at = None
        self.assistant_is_speaking = False

        # Threads
        self.capture_thread = threading.Thread(target=self._capture_audio)
        self.capture_thread.daemon = True
        self.worker_thread = threading.Thread(target=self._process_queue)
        self.worker_thread.daemon = True

    def purge_queue(self):
        """Clears the audio queue."""
        with self.q.mutex:
            self.q.queue.clear()

    def _capture_audio(self):
        command = [
            'parec', '--device=alsa_input.pci-0000_00_1f.3.analog-stereo',
            '--format=s16le', '--rate=16000', '--channels=1'
        ]
        try:
            self.proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"!!! Critical Error: Failed to start audio capture. Error: {e} !!!")
            self.running = False
            return

        while self.running:
            try:
                data = self.proc.stdout.read(self.block_size)
                if not data: break
                indata = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                if not self.q.full(): self.q.put(indata)
            except Exception:
                break

    def _process_queue(self):
        while self.running:
            try:
                indata = self.q.get(timeout=0.1)
                rms = np.sqrt(np.mean(indata**2))

                # Dynamic VAD threshold
                if self.assistant_is_speaking:
                    self.vad_threshold = self.base_vad_threshold * 2.0 # Require louder speech to interrupt
                else:
                    self.vad_threshold = self.base_vad_threshold

                if self.is_speaking:
                    self.audio_buffer = np.append(self.audio_buffer, indata)
                    if rms < self.vad_threshold:
                        if self.silence_started_at is None:
                            self.silence_started_at = time.time()
                        elif time.time() - self.silence_started_at > self.silence_duration:
                            utterance = self.audio_buffer
                            self.audio_buffer = np.array([], dtype=np.float32)
                            self.is_speaking = False
                            if not self.assistant_is_speaking:
                                self.callback_func(utterance)
                    else:
                        self.silence_started_at = None

                elif rms > self.vad_threshold:
                    if self.assistant_is_speaking:
                        self.interrupt_func()

                    self.audio_buffer = np.array([], dtype=np.float32)
                    self.is_speaking = True
                    self.silence_started_at = None
                    self.audio_buffer = np.append(self.audio_buffer, indata)

                self.q.task_done()
            except queue.Empty:
                if self.is_speaking and len(self.audio_buffer) > 0 and not self.assistant_is_speaking:
                    utterance = self.audio_buffer
                    self.audio_buffer = np.array([], dtype=np.float32)
                    self.is_speaking = False
                    self.callback_func(utterance)
                continue
            except Exception as e:
                print(f"Error in VAD thread: {e}")

    def start(self):
        print("--- Voice Assistant Activated ---")
        self.running = True
        self.capture_thread.start()
        self.worker_thread.start()

    def stop(self):
        print("\n--- Shutting down... ---")
        self.running = False
        if self.proc:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.capture_thread.is_alive(): self.capture_thread.join(timeout=1)
        if self.worker_thread.is_alive(): self.worker_thread.join(timeout=1)
        
if __name__ == "__main__":
    # Simple test for the listener
    def print_chunk_info(chunk_data):
        print(f"--- VAD triggered: Processing audio chunk of size {len(chunk_data)} ---")

    def interrupt():
        print("--- INTERRUPT ---")

    listener = LiveListener(callback_func=print_chunk_info, interrupt_func=interrupt)
    listener.start()
    
    try:
        while listener.running:
            time.sleep(1)
    except KeyboardInterrupt:
        listener.stop()
        print("--- Test finished ---")



