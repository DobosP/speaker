# Voice Assistant

A cross-platform voice assistant that lets you have natural conversations with your computer using open-source libraries.

## Features

- 🎤 **Cross-platform audio** - Works on Windows, Mac, and Linux
- 🗣️ **Speech-to-Text** - faster-whisper (4x faster than standard Whisper)
- 🤖 **Local LLM** - Powered by Ollama (llama2, llama3, etc.)
- 🔊 **Text-to-Speech** - Natural Microsoft Edge voices via edge-tts
- ⚡ **Barge-in** - Interrupt the assistant mid-speech by talking
- 🎯 **Voice Activity Detection** - Automatic speech start/stop detection

## Quick Start

### Prerequisites

1. **Python 3.10+**
2. **Ollama** - Install from [ollama.ai](https://ollama.ai) and run:
   ```bash
   ollama serve
   ollama pull llama2
   ```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/voice-assistant.git
cd voice-assistant

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# Run the assistant
python main.py

# List available audio devices
python main.py --list-devices

# Use specific input device
python main.py --input-device 2

# Use different LLM model
python main.py --llm-model llama3

# Use different STT model
python main.py --stt-model openai/whisper-small
```

### Commands

- **Say anything** - The assistant will respond
- **Say "stop" or "quit"** - Exit the application
- **Speak while assistant is talking** - Interrupt (barge-in)

## Configuration

Edit `config.json` to customize:

```json
{
  "vad_threshold": 0.005,    // Voice detection sensitivity
  "silence_duration": 1.5,   // Seconds of silence before processing
  "sample_rate": 16000,      // Audio sample rate
  "input_device": null,      // null = auto-detect
  "output_device": null,     // null = auto-detect
  "llm_model": "llama2",     // Ollama model name
  "stt_model": "base",       // Whisper model (tiny/base/small/medium/large-v3)
  "tts_voice": "en-US"       // TTS voice (en-US/en-US-male/en-GB/en-AU)
}
```

### STT Models (faster-whisper)

| Model | Speed | Quality | VRAM |
|-------|-------|---------|------|
| tiny | Fastest | Good | ~1GB |
| base | Fast | Better | ~1GB |
| small | Medium | Great | ~2GB |
| medium | Slow | Excellent | ~5GB |
| large-v3 | Slowest | Best | ~10GB |

### TTS Voices (edge-tts)

| Voice | Description |
|-------|-------------|
| en-US | US English (Aria, female) |
| en-US-male | US English (Guy, male) |
| en-GB | British English (Sonia, female) |
| en-AU | Australian English (Natasha, female) |

### VAD Threshold Tuning

| Value | Use Case |
|-------|----------|
| 0.001 | Very quiet environment, sensitive mic |
| 0.005 | Normal room, typical mic |
| 0.01  | Noisy environment |
| 0.02+ | Very noisy or if getting false triggers |

## Project Structure

```
voice-assistant/
├── main.py              # Main application entry point
├── config.json          # Configuration file
├── requirements.txt     # Python dependencies
├── run_tests.py         # Test runner
├── utils/
│   ├── audio.py         # Cross-platform audio recording/playback
│   ├── stt.py           # Speech-to-text (Whisper)
│   └── llm.py           # LLM integration (Ollama)
└── tests/
    ├── test_audio.py    # Audio module tests
    ├── test_stt.py      # STT module tests
    ├── test_llm.py      # LLM module tests
    └── test_integration.py  # Integration tests
```

## Running Tests

```bash
# Run all tests
python run_tests.py

# Run specific test suite
python run_tests.py --audio    # Audio tests only
python run_tests.py --stt      # STT tests only
python run_tests.py --quick    # Fast tests (skip model loading)
```

## Troubleshooting

### No audio input detected
- Run `python main.py --list-devices` to see available devices
- Try specifying a device: `python main.py --input-device <number>`

### Barge-in not working
- Lower the VAD threshold in `config.json`
- Speak louder or move closer to the microphone

### Whisper hallucinating (transcribing silence)
- Increase the VAD threshold in `config.json`
- Check for background noise

### LLM not responding
- Make sure Ollama is running: `ollama serve`
- Check the model is downloaded: `ollama list`

## License

MIT License - feel free to use and modify.

## Contributing

Contributions welcome! Please open an issue or PR.

