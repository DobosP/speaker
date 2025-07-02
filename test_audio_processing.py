import os
import numpy as np
import librosa
from utils.slicer import Slicer
from utils.STTModel import transcribe_audio

def test_audio_processing_pipeline():
    """
    This test verifies the audio processing pipeline (slicing and transcription)
    using the sample audio files located in the temp/ directory.
    """
    print("--- Running Audio Processing Test ---")
    
    temp_dir = 'temp'
    if not os.path.exists(temp_dir):
        print(f"!!! Test directory '{temp_dir}' not found. Skipping test. !!!")
        return

    # Find the first .wav file in the temp directory
    audio_file = next((f for f in os.listdir(temp_dir) if f.endswith('.wav')), None)
    
    if not audio_file:
        print(f"!!! No .wav files found in '{temp_dir}'. Skipping test. !!!")
        return

    audio_path = os.path.join(temp_dir, audio_file)
    print(f"--- Testing with audio file: {audio_path} ---")

    # 1. Load the audio file
    try:
        # Load audio and resample to 16000 Hz, which is what the model expects
        audio, sr = librosa.load(audio_path, sr=16000)
        print(f"Successfully loaded audio. Duration: {len(audio)/sr:.2f}s")
    except Exception as e:
        print(f"!!! Failed to load audio file: {e} !!!")
        return

    # 2. Initialize the Slicer
    # These parameters can be tuned for better performance
    slicer = Slicer(
        sr=sr,
        threshold=-40,      # Audio level threshold for silence
        min_length=5000,    # Minimum length of an audio chunk in ms
        min_interval=300,   # Minimum silence interval between chunks in ms
        hop_size=10,        # Step size for analysis in ms
        max_sil_kept=500    # Keep max 500ms of silence at the start/end of chunks
    )

    # 3. Process the audio through the slicer
    chunks = slicer.slice(audio)
    print(f"--- Sliced audio into {len(chunks)} chunks ---")

    if not chunks:
        print("!!! Slicer did not produce any chunks. The audio might be too short or silent. !!!")
        return

    # 4. Transcribe each chunk
    full_transcription = []
    for i, chunk in enumerate(chunks):
        print(f"\n--- Transcribing Chunk {i+1}/{len(chunks)} ---")
        try:
            # Ensure the chunk is not empty
            if len(chunk) > 0:
                transcription = transcribe_audio(chunk, model_type="whisper", model_id="openai/whisper-base")
                if transcription:
                    print(f"Transcription: {transcription}")
                    full_transcription.append(transcription)
                else:
                    print("Transcription resulted in an empty string.")
            else:
                print("Skipping empty chunk.")
        except Exception as e:
            print(f"!!! Error during transcription of chunk {i+1}: {e} !!!")

    print("\n--- Test Finished ---")
    if full_transcription:
        print("\n--- Full Transcription ---")
        final_text = " ".join(full_transcription)
        print(final_text)
        # Save the final transcription to a file
        with open("test_transcription_output.txt", "w") as f:
            f.write(final_text)
        print("\nFull transcription saved to 'test_transcription_output.txt'")
    else:
        print("\n--- No transcription was produced. ---")


if __name__ == "__main__":
    test_audio_processing_pipeline()
