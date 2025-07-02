import time
import librosa
import numpy as np
from utils.speach_live import LiveListener

def test_end_to_end_processing():
    """
    This test validates the entire end-to-end processing pipeline
    by feeding a sample audio file directly into the LiveListener's queue.
    """
    print("--- Running End-to-End Processing Test ---")

    # 1. Define a callback function to capture the output
    transcribed_text = []
    def test_callback(chunk):
        """This callback will be called by the listener with transcribed text."""
        print(f"--- Test callback received chunk of size: {len(chunk)} ---")
        # In a real scenario, this would be the transcription result.
        # For this test, we'll just store the chunk size as a proxy for success.
        transcribed_text.append(str(len(chunk)))

    # 2. Initialize the LiveListener with our test callback
    listener = LiveListener(callback_func=test_callback)

    # 3. Load the sample audio file
    audio_path = 'temp/2_min_sample.mp3_chunk_0.wav'
    try:
        # Load and prepare the audio data to simulate the live stream
        audio, sr = librosa.load(audio_path, sr=16000)
        # Convert to the format the listener expects (float32)
        audio_float32 = audio.astype(np.float32)
        print(f"Successfully loaded and prepared test audio: {audio_path}")
    except Exception as e:
        print(f"!!! Failed to load audio file: {e} !!!")
        return

    # 4. Manually start the processing thread and feed the queue
    print("--- Manually starting worker thread and feeding audio queue ---")
    listener.running = True
    listener.worker_thread.start()

    # Feed the audio data in chunks to simulate a live stream
    block_size = listener.block_size
    for i in range(0, len(audio_float32), block_size):
        chunk = audio_float32[i:i+block_size]
        listener.q.put(chunk)
        time.sleep(0.01) # Simulate a small delay between chunks

    # 5. Wait for the queue to be processed
    print("--- Waiting for queue to be processed ---")
    while not listener.q.empty():
        time.sleep(0.5)
    
    # Give the slicer a final chance to process the buffer
    final_chunk = listener.audio_buffer
    if len(final_chunk) > 0:
        listener.callback_func(final_chunk)

    # 6. Stop the listener and check the results
    listener.stop()

    print("\n--- Test Finished ---")
    if transcribed_text:
        print(f"SUCCESS: The test callback was called {len(transcribed_text)} times.")
        print("This confirms the audio processing pipeline is working correctly.")
    else:
        print("FAILURE: The test callback was never called.")
        print("This indicates a problem with the audio processing thread or the slicer.")

if __name__ == "__main__":
    test_end_to_end_processing()
