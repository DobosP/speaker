# Voice Assistant Setup Guide

This guide will help you set up the voice assistant with persistent memory.

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

### Step 4: Run Setup Script

```bash
# For peer authentication (no password)
python setup_database.py --db-url "postgresql:///voice_assistant"

# For password authentication
python setup_database.py --db-url "postgresql://dobo:yourpassword@localhost/voice_assistant"
```

### Step 5: Verify Setup

```bash
python setup_database.py --db-url "postgresql:///voice_assistant" --verify-only
```

You should see:
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

```bash
python main.py --db-url "postgresql:///voice_assistant"
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
   python setup_database.py --db-url "postgresql://user:password@localhost/voice_assistant"
   ```

### "Must be superuser to create this extension"

**Problem:** Only superuser can create extensions.

**Solution:**
```bash
sudo -u postgres psql -d voice_assistant -c "CREATE EXTENSION vector;"
```

## Running Without Database

If you don't want to set up PostgreSQL, the assistant will work in **in-memory mode**:

```bash
python main.py --no-memory
```

Note: History will be lost when you restart the application.

## Next Steps

Once setup is complete:

1. **Start Ollama:**
   ```bash
   ollama serve
   ollama pull llama2
   ```

2. **Run the assistant:**
   ```bash
   python main.py
   ```

3. **Test memory:**
   - Say: "My name is John"
   - Later: "What's my name?"
   - The assistant should remember!

## Additional Resources

- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [pgvector GitHub](https://github.com/pgvector/pgvector)
- [Ollama Documentation](https://ollama.ai/docs)

