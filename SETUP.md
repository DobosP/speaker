# Voice Assistant Setup Guide

This guide will help you set up the voice assistant with persistent memory.

> **Credentials & API tokens** (HuggingFace, the `GIT_HUB_TOKEN` admin key,
> LiveKit) are documented separately in [`CREDENTIALS.md`](CREDENTIALS.md). This
> guide covers the local PostgreSQL memory store only.

## Quick Setup (Automated)

The easiest way to set up everything:

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Run automated setup script
./setup.sh
```

The script will:
- ✅ Check PostgreSQL installation
- ✅ Install pgvector if needed
- ✅ Create database and configure permissions
- ✅ Set up all tables
- ✅ Create `.env` file

## Manual Setup

If you prefer to set up manually or the automated script doesn't work:

### Step 1: Install PostgreSQL

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y postgresql postgresql-contrib

# Verify installation
psql --version
```

### Step 2: Install pgvector Extension

```bash
# Ubuntu/Debian (PostgreSQL 16)
sudo apt-get install -y postgresql-16-pgvector

# For other PostgreSQL versions, replace 16 with your version
# Check your version: psql --version
```

### Step 3: Create Database

**Option A: Peer Authentication (No Password - Recommended for Local)**

```bash
# Create database
sudo -u postgres createdb voice_assistant

# Grant permissions to your user
sudo -u postgres psql -d voice_assistant -c "
    GRANT ALL ON SCHEMA public TO $USER;
    GRANT ALL PRIVILEGES ON DATABASE voice_assistant TO $USER;
"

# Enable pgvector extension
sudo -u postgres psql -d voice_assistant -c "CREATE EXTENSION vector;"
```

**Option B: Password Authentication**

```bash
# Create database
sudo -u postgres createdb voice_assistant

# Create user with password
sudo -u postgres psql -c "
    CREATE USER dobo WITH PASSWORD 'yourpassword';
    GRANT ALL PRIVILEGES ON DATABASE voice_assistant TO dobo;
"

# Enable pgvector extension
sudo -u postgres psql -d voice_assistant -c "CREATE EXTENSION vector;"

# Grant schema permissions
sudo -u postgres psql -d voice_assistant -c "GRANT ALL ON SCHEMA public TO dobo;"
```

### Step 4: Apply the Schema (Migrations)

`python -m tools.migrate apply` is **the single schema path** — it owns all
table/index DDL (correct unconstrained-vector + per-row dim + partial HNSW;
migrations are idempotent and additive, so they also reconcile legacy DBs).

```bash
# For peer authentication (no password)
python -m tools.migrate apply --database-url "postgresql:///voice_assistant"

# For password authentication
python -m tools.migrate apply --database-url "postgresql://dobo:yourpassword@localhost/voice_assistant"

# (If DATABASE_URL is exported, you can omit --database-url entirely)
python -m tools.migrate apply
```

> `setup_database.py` is now a thin role-create/verify wrapper around this
> migrations path — use `--create-db` to bootstrap the database and
> `--verify-only` to health-check it (see below); it no longer carries any
> schema SQL of its own.

### Step 5: Verify Setup

```bash
# Show which migrations are applied / pending
python -m tools.migrate status --database-url "postgresql:///voice_assistant"

# Or run the read-only health check via the wrapper
python setup_database.py --db-url "postgresql:///voice_assistant" --verify-only
```

The `--verify-only` check should show:
```
📊 Database Status:
   Tables found: messages, summaries, user_profile
   Total messages: 0
   Total summaries: 0
   pgvector: ✅ Enabled
```

## Configuration

### Using .env File (Recommended)

Create a `.env` file in the project root:

```bash
# For peer authentication (no password)
DATABASE_URL=postgresql:///voice_assistant

# For password authentication
# DATABASE_URL=postgresql://dobo:password@localhost/voice_assistant
```

The voice assistant will automatically read this file.

### Using Environment Variable

```bash
# Add to ~/.bashrc or ~/.zshrc
export DATABASE_URL="postgresql:///voice_assistant"
```

### Using Command Line

There is no `--db-url` runtime flag. The runtime is `python -m core`; it reads
`DATABASE_URL` from the environment / `.env` (see `MEMORY.md` and the `memory`
block in `config.json`):

```bash
DATABASE_URL="postgresql:///voice_assistant" python -m core
```

## Troubleshooting

### "connection to server failed"

**Problem:** PostgreSQL is not running.

**Solution:**
```bash
sudo systemctl start postgresql
sudo systemctl enable postgresql  # Start on boot
```

### "extension vector is not available"

**Problem:** pgvector is not installed.

**Solution:**
```bash
# Install pgvector
sudo apt-get install -y postgresql-16-pgvector

# Enable extension
sudo -u postgres psql -d voice_assistant -c "CREATE EXTENSION vector;"
```

### "permission denied for schema public"

**Problem:** Your user doesn't have permissions.

**Solution:**
```bash
sudo -u postgres psql -d voice_assistant -c "
    GRANT ALL ON SCHEMA public TO $USER;
    GRANT ALL PRIVILEGES ON DATABASE voice_assistant TO $USER;
"
```

### "fe_sendauth: no password supplied"

**Problem:** PostgreSQL requires password authentication.

**Solutions:**

1. **Use peer authentication (easiest for local):**
   ```bash
   # Edit pg_hba.conf
   sudo nano /etc/postgresql/*/main/pg_hba.conf
   
   # Change this line:
   # local   all             all                                     md5
   # To:
   # local   all             all                                     peer
   
   # Restart PostgreSQL
   sudo systemctl restart postgresql
   ```

2. **Or use password in connection string:**
   ```bash
   python -m tools.migrate apply --database-url "postgresql://user:password@localhost/voice_assistant"
   ```

### "Must be superuser to create this extension"

**Problem:** Only superuser can create extensions.

**Solution:**
```bash
sudo -u postgres psql -d voice_assistant -c "CREATE EXTENSION vector;"
```

## Running Without Database

If you don't want to set up PostgreSQL, the assistant works in **in-memory
mode** automatically. There is no `--no-memory` flag: when `DATABASE_URL` is
unset (and the `memory` config block doesn't force the Postgres backend), the
runtime falls back to the in-RAM `SessionMemory`.

```bash
python -m core
```

Note: History will be lost when you restart the application.

## Next Steps

Once setup is complete:

1. **Provision Ollama models** named in `config.json`'s `llm` block. If Ollama is
   not already running as a service, start it in a separate terminal for this
   one-time provisioning step:

   Terminal A (only when no Ollama service is running):

   ```bash
   ollama serve
   ```

   Terminal B:

   ```bash
   ollama pull gemma3:12b
   python -m tools.setup_minicpm
   ```

   Terminal A can be stopped with Ctrl-C after provisioning. The Linux live
   launcher later starts or reuses Ollama for each physical session.

2. **Run the assistant** (the legacy `main.py` was deleted 2026-05-26 —
   see `docs/adr/0002`):
   ```bash
   ./live.sh                              # Linux: setup + private recorded session
   python -m core --engine sherpa          # low-level; audio route already prepared
   python -m core --engine console --llm echo   # no audio/models smoke test
   ```

   `./live.sh` writes each capture to a new ignored directory under `logs/live/`
   and never prunes it. Direct core keeps the newest 20 untracked `logs/runs/`
   bundles; tracked historical fixtures are protected. On the Linux OS-EC path,
   standalone `python -m tools.doctor` also assumes the transient audio route is
   already prepared; `./live.sh` creates it and requires full `READY` before
   opening the microphone.

3. **Test memory:**
   - Say: "My name is John"
   - Later: "What's my name?"
   - The assistant should remember!

## Additional Resources

- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [pgvector GitHub](https://github.com/pgvector/pgvector)
- [Ollama Documentation](https://ollama.ai/docs)
