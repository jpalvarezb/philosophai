FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files and source code
COPY pyproject.toml ./
COPY src/ ./src/

# Install Python dependencies + package
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Download spaCy model if needed
RUN python -m spacy download en_core_web_sm || true

# Copy UI
COPY ui/ ./ui/

# Copy entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run via entrypoint
ENTRYPOINT ["/entrypoint.sh"]
