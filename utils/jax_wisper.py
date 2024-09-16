from whisper_jax import FlaxWhisperPipline

# instantiate pipeline
pipeline = FlaxWhisperPipline("openai/whisper-large-v3")

# JIT compile the forward call - slow, but we only do once
text = pipeline("recordings/2_min_sample.mp3")


print(text)

with open("test.txt", "w") as writer:
    writer.write(text)
