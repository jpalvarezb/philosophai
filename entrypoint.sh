#!/bin/bash
set -e

# Database download URL (set via environment variable)
DB_URL="${DB_DOWNLOAD_URL:-}"

# Force re-download if requested (removes stale DB)
if [ "${FORCE_DB_DOWNLOAD:-}" = "1" ]; then
    echo "🔄 FORCE_DB_DOWNLOAD=1 — removing existing database..."
    rm -f /data/philosoph.duckdb /data/philosoph.duckdb.wal /data/philosoph.duckdb.gz
fi

# Check if database exists
if [ ! -f "/data/philosoph.duckdb" ]; then
    if [ -z "$DB_URL" ]; then
        echo "❌ ERROR: Database not found and DB_DOWNLOAD_URL not set"
        echo "Please set DB_DOWNLOAD_URL environment variable with a public URL to the database"
        exit 1
    fi
    
    echo "📥 Downloading database from $DB_URL..."
    cd /data
    
    # Download with retry logic
    for i in {1..3}; do
        if wget -q --show-progress "$DB_URL" -O philosoph.duckdb.gz; then
            echo "✓ Download complete"
            break
        else
            echo "⚠️  Download failed (attempt $i/3), retrying..."
            sleep 5
        fi
    done
    
    if [ ! -f "philosoph.duckdb.gz" ]; then
        echo "❌ Failed to download database after 3 attempts"
        exit 1
    fi
    
    echo "📦 Decompressing database..."
    gunzip philosoph.duckdb.gz
    echo "✓ Database ready"
fi

# Start the application
echo "🚀 Starting application..."
exec uvicorn src.api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
