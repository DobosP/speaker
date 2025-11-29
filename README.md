# Voice Assistant

A cross-platform voice assistant with persistent memory that lets you have natural conversations with your computer using open-source libraries.

## Features

- 🎤 **Cross-platform audio** - Works on Windows, Mac, and Linux
- 🗣️ **Speech-to-Text** - faster-whisper (4x faster than standard Whisper)
- 🤖 **Local LLM** - Powered by Ollama (llama2, llama3, etc.)
- 🔊 **Text-to-Speech** - Natural Microsoft Edge voices via edge-tts
- ⚡ **Barge-in** - Interrupt the assistant mid-speech by talking
- 🎯 **Voice Activity Detection** - Automatic speech start/stop detection
- 💾 **Persistent Memory** - PostgreSQL + pgvector for long-term memory
- 🔍 **Semantic Search** - Find past conversations by meaning

## Memory Architecture

The assistant uses a multi-layer memory system:

```
┌─────────────────────────────────────────────────────────────┐
│                    MEMORY LAYERS                             │
├─────────────────────────────────────────────────────────────┤
│ Layer 1: Recent Messages (Short-term)                       │
│   → Last 20 messages in current session                     │
│   → Full context for immediate conversation                 │
├─────────────────────────────────────────────────────────────┤
│ Layer 2: Conversation Summaries (Medium-term)               │
│   → Auto-generated when context gets too long               │
│   → Condensed history with key topics                       │
├─────────────────────────────────────────────────────────────┤
│ Layer 3: Vector Embeddings (Long-term)                      │
│   → Semantic search over all past conversations             │
│   → Powered by pgvector + sentence-transformers             │
├─────────────────────────────────────────────────────────────┤
│ Layer 4: User Profile                                       │
│   → Learned preferences (name, interests, etc.)             │
│   → Persists across all sessions                            │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

1. **Python 3.10+**
2. **Ollama** - Install from [ollama.ai](https://ollama.ai) and run:
   ```bash
   ollama serve
   ollama pull llama2
   ```
3. **PostgreSQL with pgvector** (optional, for persistent memory):
   ```bash
   # Ubuntu/Debian
   sudo apt install postgresql postgresql-contrib
   
   # Install pgvector extension
   # See: https://github.com/pgvector/pgvector
   ```

### Installation

```bash
# Clone the repository
git clone https://github.com/DobosP/speaker.git
cd speaker

# Install dependencies
pip install -r requirements.txt

# (Optional) Setup database for persistent memory
python setup_database.py --create-db
```

### Usage

```bash
# Run the assistant (with memory)
python main.py

# Run without persistent memory
python main.py --no-memory

# Start a fresh session
python main.py --new-session

# Continue a specific session
python main.py --session-id abc123

# Use custom database
python main.py --db-url "postgresql://user:pass@host/db"

# List available audio devices
python main.py --list-devices

# Use specific input device
python main.py --input-device 2

# Use different LLM model
python main.py --llm-model llama3

# Use different STT model
python main.py --stt-model small

# Use different TTS voice
python main.py --tts-voice en-GB
```

### Commands

- **Say anything** - The assistant will respond
- **Say "stop" or "quit"** - Exit the application
- **Speak while assistant is talking** - Interrupt (barge-in)
- **Reference past conversations** - "What did I tell you last time about..."

## Configuration

Edit `config.json` to customize:

```json
{
  "vad_threshold": 0.005,
  "silence_duration": 1.5,
  "sample_rate": 16000,
  "input_device": null,
  "output_device": null,
  "llm_model": "llama2",
  "stt_model": "base",
  "tts_voice": "en-US"
}
```

### Environment Variables

```bash
# Database connection (optional)
export DATABASE_URL="postgresql://localhost/voice_assistant"
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
├── setup_database.py    # Database setup script
├── run_tests.py         # Test runner
├── utils/
│   ├── audio.py         # Cross-platform audio recording/playback
│   ├── stt.py           # Speech-to-text (faster-whisper)
│   ├── llm.py           # LLM integration (Ollama)
│   └── memory.py        # Multi-layer memory system
└── tests/
    ├── test_audio.py    # Audio module tests
    ├── test_stt.py      # STT module tests
    ├── test_llm.py      # LLM module tests
    └── test_integration.py  # Integration tests
```

## Database Setup

For persistent memory across sessions:

```bash
# Create database and tables
python setup_database.py --create-db --db-name voice_assistant

# Verify setup
python setup_database.py --verify-only

# Custom host/user
python setup_database.py --create-db --host localhost --user myuser --password mypass
```

### PostgreSQL + pgvector Installation

```bash
# Ubuntu/Debian
sudo apt install postgresql postgresql-contrib

# Install pgvector (required for semantic search)
cd /tmp
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install

# Enable extension in your database
psql -d voice_assistant -c "CREATE EXTENSION vector;"
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

### Database connection failed
- Check PostgreSQL is running: `sudo systemctl status postgresql`
- Verify connection URL: `psql "postgresql://localhost/voice_assistant"`
- Run without memory: `python main.py --no-memory`

## License

MIT License - feel free to use and modify.

## Contributing

Contributions welcome! Please open an issue or PR.
