import os
import argparse
import time
import subprocess
import signal
from utils.speach_live import LiveListener
from utils.STTModel import transcribe_audio
from utils.llm import get_llm

# --- Global State ---
llm = None
listener = None

def speak(text):
    """Calls the speak.py script to speak the text."""
    print(f"Assistant: {text}\n")
    try:
        subprocess.run(["./speak.py", text], check=True)
    except Exception as e:
        print(f"Error in speak: {e}")

def is_video_outro(text):
    """Checks for common video outro phrases."""
    text = text.lower()
    return any(phrase in text for phrase in ["watching", "video", "subscribe", "channel", "next time"])

def process_audio_chunk(chunk_data, stt_model_id="openai/whisper-base"):
    """Processes a complete audio utterance."""
    try:
        transcription = transcribe_audio(chunk_data, model_type="whisper", model_id=stt_model_id)
        if transcription and not is_video_outro(transcription):
            print(f"You: {transcription}\n")
            with open("live_transcript.txt", "a") as f:
                f.write(f"You: {transcription}\n")

            if "stop" in transcription.lower():
                if listener: listener.stop()
                return
            
            if llm:
                prompt = f"Respond to the following in a single, short sentence: {transcription}"
                llm_response = llm.get_response(prompt)
                with open("live_transcript.txt", "a") as f:
                    f.write(f"Assistant: {llm_response}\n")
                speak(llm_response)
    except Exception as e:
        print(f"An error occurred during processing: {e}")

def shutdown_handler(signum, frame):
    """Gracefully shuts down the application."""
    print("\n--- Shutdown signal received. Cleaning up. ---")
    if listener:
        listener.running = False

def main():
    global llm, listener
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_model", type=str, default="llama2", help="LLM model to use")
    parser.add_argument("--stt_model", type=str, default="openai/whisper-base", help="STT model to use")
    args = parser.parse_args()

    # Setup virtual audio devices
    subprocess.run(["./setup_audio.sh"], check=True)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    print("--- Initializing LLM... ---")
    try:
        llm = get_llm(llm_type="local", model=args.llm_model)
    except Exception as e:
        print(f"!!! Critical Error: Could not initialize LLM. Error: {e} !!!")
        subprocess.run(["./cleanup_audio.sh"], check=True)
        exit()
    print("--- LLM Initialized. ---")

    listener = LiveListener(callback_func=process_audio_chunk)
    listener.start()

    start_time = time.time()
    try:
        while listener.running and (time.time() - start_time) < 30:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        if listener: listener.stop()
        # Cleanup virtual audio devices
        subprocess.run(["./cleanup_audio.sh"], check=True)


if __name__ == "__main__":
    main()





