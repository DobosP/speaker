import sys
from gtts import gTTS
import tempfile
import subprocess

def speak(text):
    """Speaks the text using gTTS and plays it through the dedicated assistant output."""
    try:
        with tempfile.NamedTemporaryFile(delete=True, suffix='.mp3') as fp:
            tts = gTTS(text=text, lang='en')
            tts.save(fp.name)
            # Play on the dedicated virtual sink
            subprocess.run(["paplay", f"--device=assistant_output", fp.name], check=True)
    except Exception as e:
        print(f"Error in speak: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        speak(sys.argv[1])
