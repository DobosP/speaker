import os
import librosa
from utils.STTModel import transcribe_audio
from utils.llm import get_llm

def main():
    print("--- Starting Pipeline Test ---")
    
    # Initialize LLM
    print("Initializing LLM...")
    try:
        llm = get_llm(llm_type="local", model="llama2")
    except (ValueError, ConnectionError) as e:
        print(f"Error: {e}")
        exit()

    # Get list of audio files
    temp_dir = "temp"
    audio_files = [f for f in os.listdir(temp_dir) if f.endswith(".wav")]
    if not audio_files:
        print("No .wav files found in the temp directory. Exiting test.")
        return

    # Process each audio file
    for audio_file in audio_files:
        print(f"\n--- Processing {audio_file} ---")
        audio_path = os.path.join(temp_dir, audio_file)
        
        # Read audio file
        audio_data, sr = librosa.load(audio_path, sr=16000)
        
        # Transcribe audio
        print("Transcribing audio...")
        transcription = transcribe_audio(audio_data, model_type="whisper", model_id="openai/whisper-base")
        print(f"Transcription: {transcription}")
        
        # Get LLM response
        if transcription:
            print("Getting LLM response...")
            llm_response = llm.get_response(transcription)
            print(f"LLM Response: {llm_response}")

    print("\n--- Pipeline Test Finished ---")

if __name__ == "__main__":
    main()
