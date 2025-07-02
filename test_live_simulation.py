import os
import time
import threading
import numpy as np
import librosa
from utils.speach_live import LiveListener
from main import process_audio_chunk # Import the real processing function

# --- Test Setup ---
TEST_DURATION_SECONDS = 10
TRANSCRIPT_FILE = "live_transcript.txt"
AUDIO_FILE = 'temp/2_min_sample.mp3_chunk_0.wav'

def test_live_audio_simulation():
    """
    This test provides the most realistic simulation of the live application.
    It replaces the arecord microphone input with a looped audio file,
    and uses the real processing logic from main.py.

    The test passes if the live_transcript.txt file is created and contains text.
    """
    print("--- Running Live Audio Simulation Test ---")

    # 1. Clean up previous test runs
    if os.path.exists(TRANSCRIPT_FILE):
        os.remove(TRANSCRIPT_FILE)

    # 2. Load the audio file to be used as a simulated microphone stream
    try:
        audio_data, sr = librosa.load(AUDIO_FILE, sr=16000)
        # Convert to the format arecord would produce (int16)
        audio_int16 = (audio_data * 32767).astype(np.int16)
        print(f"Successfully loaded test audio file: {AUDIO_FILE}")
    except Exception as e:
        print(f"!!! Failed to load audio file: {e} !!!")
        return

    # 3. Monkey-patch the _capture_audio method to simulate live input
    original_capture_method = LiveListener._capture_audio
    
    def simulated_capture_audio(self):
        print("--- Starting SIMULATED audio capture ---")
        start_time = time.time()
        audio_index = 0
        
        while self.running and (time.time() - start_time) < TEST_DURATION_SECONDS:
            # Read a block of data from the simulated audio file
            start = audio_index
            end = audio_index + self.block_size
            data_to_queue = audio_int16[start:end]
            
            # Loop the audio if it finishes
            audio_index = end
            if end >= len(audio_int16):
                audio_index = 0

            # Convert to the float32 format the processing queue expects
            indata = data_to_queue.astype(np.float32) / 32768.0
            if not self.q.full():
                self.q.put(indata)
            
            time.sleep(0.05) # Simulate real-time audio stream
        
        print("--- SIMULATED audio capture finished ---")

    LiveListener._capture_audio = simulated_capture_audio

    # 4. Run the application with the simulated audio
    listener = LiveListener(callback_func=process_audio_chunk)
    listener.start()

    # Let the test run for a defined duration
    print(f"--- Test will run for {TEST_DURATION_SECONDS} seconds ---")
    time.sleep(TEST_DURATION_SECONDS)

    # 5. Stop the application and clean up
    listener.stop()
    LiveListener._capture_audio = original_capture_method # Restore original method
    print("--- Test run finished. Now checking results... ---")

    # 6. Verify the results
    if os.path.exists(TRANSCRIPT_FILE):
        with open(TRANSCRIPT_FILE, "r") as f:
            content = f.read()
        if content.strip():
            print(f"\nSUCCESS: The file '{TRANSCRIPT_FILE}' was created and contains text.")
            print("--- Transcript ---")
            print(content)
        else:
            print(f"\nFAILURE: The file '{TRANSCRIPT_FILE}' was created but is EMPTY.")
    else:
        print(f"\nFAILURE: The transcript file '{TRANSCRIPT_FILE}' was NOT created.")

if __name__ == "__main__":
    test_live_audio_simulation()
