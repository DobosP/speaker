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
3. **PostgreSQL with pgvector** (optional, for persistent memory)

### Installation

```bash
# Clone the repository
git clone https://github.com/DobosP/speaker.git
cd speaker

# Install Python dependencies
pip install -r requirements.txt

# (Optional) Setup database for persistent memory
# Option 1: Automated setup (recommended)
./setup.sh

# Option 2: Manual setup
python setup_database.py --create-db
```

### Database Setup (Optional but Recommended)

The voice assistant can work without a database (in-memory only), but for persistent memory across sessions, you need PostgreSQL.

**📖 For detailed setup instructions, see [SETUP.md](SETUP.md)**

#### Automated Setup (Easiest)

```bash
# Run the automated setup script
./setup.sh
```

This script will:
- ✅ Check PostgreSQL installation
- ✅ Install pgvector if needed
- ✅ Create database and user
- ✅ Set up all tables
- ✅ Create `.env` file from `env.example`

#### Manual Setup

**Step 1: Install PostgreSQL and pgvector**

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y postgresql postgresql-contrib

# Install pgvector extension
sudo apt-get install -y postgresql-16-pgvector
# Or for other PostgreSQL versions:
# sudo apt-get install -y postgresql-<version>-pgvector
```

**Step 2: Create Database**

```bash
# Create database (as postgres user)
sudo -u postgres createdb voice_assistant

# Create user (optional, for password auth)
sudo -u postgres createuser dobo
# Or with password:
# sudo -u postgres psql -c "CREATE USER dobo WITH PASSWORD 'yourpassword';"
```

**Step 3: Enable pgvector Extension**

```bash
# Enable vector extension (requires superuser)
sudo -u postgres psql -d voice_assistant -c "CREATE EXTENSION vector;"
```

**Step 4: Grant Permissions**

```bash
# Grant schema permissions (for peer authentication)
sudo -u postgres psql -d voice_assistant -c "
    GRANT ALL ON SCHEMA public TO dobo;
    GRANT ALL PRIVILEGES ON DATABASE voice_assistant TO dobo;
"
```

**Step 5: Run Setup Script**

```bash
# For peer authentication (no password)
python setup_database.py --db-url "postgresql:///voice_assistant"

# For password authentication
python setup_database.py --db-url "postgresql://dobo:password@localhost/voice_assistant"
```

**Step 6: Set Environment Variable (Optional)**

```bash
# Add to ~/.bashrc or ~/.zshrc
export DATABASE_URL="postgresql:///voice_assistant"
# Or with password:
# export DATABASE_URL="postgresql://dobo:password@localhost/voice_assistant"
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

For persistent memory across sessions, see **[SETUP.md](SETUP.md)** for detailed instructions.

**Quick setup:**
```bash
# Automated setup (recommended)
./setup.sh

# Or manual setup
python setup_database.py --create-db --db-name voice_assistant
```

**Without database (in-memory only):**
```bash
python main.py --no-memory
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
