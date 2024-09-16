import os
import librosa  # Optional. Use any library you like to read audio files.
# Optional. Use any library you like to write audio files.
import soundfile as sf
from utils.slicer import Slicer
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import time

# Load the audio file
audio_path = '/home/dobo/projects/speaker/recordings/2_min_sample.mp3'
audio, sr = librosa.load(audio_path, sr=None, mono=False)
output_dir = 'clips'
transcript_dir = 'transcripts'
file_name = audio_path.split('/')[-1]
temp_dir = 'temp'





# Initialize Slicer with the given parameters
slicer = Slicer(
    sr=sr,
    threshold=-40,
    min_length=20000,
    min_interval=300,
    hop_size=10,
    max_sil_kept=500
)

# Slice the audio into chunks
chunks = slicer.slice(audio)

# Setup paths for sliced audio files
os.makedirs(output_dir, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)

chunk_files = []
# Create a directory for each chunk and include the interval in its name
for i, chunk in enumerate(chunks):
    chunk_file = os.path.join(temp_dir, f"{file_name}_chunk_{i}.wav")
    sf.write(chunk_file, chunk, sr)
    chunk_files.append(chunk_file)


# Setup the model and pipeline
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

# Process each chunk with the speech recognition pipeline and aggregate results
all_texts = []
for i, chunk_file in enumerate(chunk_files):
    start_time = time.time()
    result = pipe(chunk_file, generate_kwargs={"language": "english"})
    elapsed_time = time.time() - start_time
    print(
        f"Processed segment {i+1}/{len(chunk_files)} in {elapsed_time:.2f} seconds.")
    all_texts.append(result["text"])

# Combine all texts into a single string
full_text = ' '.join(all_texts)

os.makedirs(transcript_dir, exist_ok=True)

# Save the transcribed text to a file
with open(os.path.join(transcript_dir, f"{file_name}_transcription.txt"), "w") as writer:
    writer.write(full_text)
