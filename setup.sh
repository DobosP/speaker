#!/bin/bash
# Voice Assistant Automated Setup Script
# This script sets up the database and environment for the voice assistant

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_header() {
    echo ""
    echo "=================================================="
    echo "🎙️  Voice Assistant Setup"
    echo "=================================================="
    echo ""
}

# Check if .env exists
if [ ! -f .env ]; then
    print_warning ".env file not found. Creating from env.example..."
    if [ -f env.example ]; then
        cp env.example .env
        print_success "Created .env file. Please edit it with your settings."
        print_info "Opening .env in editor... (you can close it if you want to edit manually)"
        ${EDITOR:-nano} .env 2>/dev/null || true
    else
        print_error "env.example not found!"
        exit 1
    fi
fi

# Load .env file
export $(grep -v '^#' .env | xargs)

# Set defaults
DB_NAME=${DB_NAME:-voice_assistant}
DB_USER=${DB_USER:-$USER}
DB_HOST=${DB_HOST:-localhost}
DB_PORT=${DB_PORT:-5432}
PG_SUPERUSER=${PG_SUPERUSER:-postgres}

print_header

# Step 1: Check PostgreSQL
print_info "Step 1: Checking PostgreSQL..."
if ! command -v psql &> /dev/null; then
    print_error "PostgreSQL client (psql) not found!"
    print_info "Install with: sudo apt-get install postgresql-client"
    exit 1
fi

# Check if PostgreSQL is running
if ! pg_isready -h "$DB_HOST" -p "$DB_PORT" &> /dev/null; then
    print_error "PostgreSQL is not running on $DB_HOST:$DB_PORT"
    print_info "Start it with: sudo systemctl start postgresql"
    exit 1
fi
print_success "PostgreSQL is running"

# Step 2: Check pgvector
print_info "Step 2: Checking pgvector extension..."
if ! dpkg -l | grep -q postgresql.*pgvector; then
    print_warning "pgvector not installed. Installing..."
    sudo apt-get update
    sudo apt-get install -y postgresql-16-pgvector || {
        print_error "Failed to install pgvector"
        print_info "You may need to install it manually. See README.md"
        exit 1
    }
    print_success "pgvector installed"
else
    print_success "pgvector is installed"
fi

# Step 3: Create database and user
print_info "Step 3: Setting up database..."

# Build connection string for superuser
if [ -n "$DB_PASSWORD" ]; then
    SUPER_CONN="postgresql://$PG_SUPERUSER:$DB_PASSWORD@$DB_HOST:$DB_PORT/postgres"
else
    SUPER_CONN="postgresql://$PG_SUPERUSER@$DB_HOST:$DB_PORT/postgres"
fi

# Check if database exists
DB_EXISTS=$(sudo -u "$PG_SUPERUSER" psql -lqt -h "$DB_HOST" -p "$DB_PORT" | cut -d \| -f 1 | grep -w "$DB_NAME" | wc -l)

if [ "$DB_EXISTS" -eq 0 ]; then
    print_info "Creating database: $DB_NAME"
    sudo -u "$PG_SUPERUSER" psql -h "$DB_HOST" -p "$DB_PORT" -c "CREATE DATABASE $DB_NAME;" || {
        print_error "Failed to create database"
        exit 1
    }
    print_success "Database created"
else
    print_info "Database '$DB_NAME' already exists"
fi

# Create user if password is provided
if [ -n "$DB_PASSWORD" ]; then
    print_info "Creating/updating user: $DB_USER"
    sudo -u "$PG_SUPERUSER" psql -h "$DB_HOST" -p "$DB_PORT" -c "
        DO \$\$
        BEGIN
            IF NOT EXISTS (SELECT FROM pg_user WHERE usename = '$DB_USER') THEN
                CREATE USER $DB_USER WITH PASSWORD '$DB_PASSWORD';
            ELSE
                ALTER USER $DB_USER WITH PASSWORD '$DB_PASSWORD';
            END IF;
        END
        \$\$;
    " || print_warning "Could not create/update user (may already exist)"
    
    # Grant privileges
    sudo -u "$PG_SUPERUSER" psql -h "$DB_HOST" -p "$DB_PORT" -d "$DB_NAME" -c "
        GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;
        GRANT ALL ON SCHEMA public TO $DB_USER;
    " || print_warning "Could not grant privileges"
fi

# Step 4: Enable pgvector extension
print_info "Step 4: Enabling pgvector extension..."
sudo -u "$PG_SUPERUSER" psql -h "$DB_HOST" -p "$DB_PORT" -d "$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS vector;" || {
    print_error "Failed to enable pgvector extension"
    print_info "You may need to run manually: sudo -u postgres psql -d $DB_NAME -c 'CREATE EXTENSION vector;'"
    exit 1
}
print_success "pgvector extension enabled"

# Step 5: Grant schema permissions (for peer auth)
if [ -z "$DB_PASSWORD" ]; then
    print_info "Granting schema permissions..."
    sudo -u "$PG_SUPERUSER" psql -h "$DB_HOST" -p "$DB_PORT" -d "$DB_NAME" -c "
        GRANT ALL ON SCHEMA public TO $DB_USER;
        GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;
    " || print_warning "Could not grant permissions (may already be set)"
fi

# Step 6: Run Python setup script
print_info "Step 5: Creating tables..."

# Build database URL
if [ -n "$DB_PASSWORD" ]; then
    DB_URL="postgresql://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME"
else
    DB_URL="postgresql:///$DB_NAME"
fi

# Update .env with final DATABASE_URL
sed -i "s|^DATABASE_URL=.*|DATABASE_URL=$DB_URL|" .env

# Run Python setup
python3 setup_database.py --db-url "$DB_URL" || {
    print_error "Failed to create tables"
    exit 1
}

print_success "Tables created"

# Step 7: Verify setup
print_info "Step 6: Verifying setup..."
python3 setup_database.py --db-url "$DB_URL" --verify-only || {
    print_warning "Verification had issues, but setup may still work"
}

# Summary
echo ""
echo "=================================================="
print_success "Setup Complete!"
echo "=================================================="
echo ""
print_info "Database URL: $DB_URL"
print_info "Configuration saved to: .env"
echo ""
print_info "You can now run the voice assistant:"
echo "  python main.py"
echo ""
print_info "Or with explicit database URL:"
echo "  python main.py --db-url \"$DB_URL\""
echo ""

