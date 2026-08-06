FROM python:3.11-slim

# Install system dependencies for building C extensions (numpy, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Layer 1: Python dependencies (cached unless requirements.txt changes) ──
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Layer 2: Application source code ──
# Copy only what the server needs at runtime.
# server/data/ is excluded via .dockerignore (mounted as a Docker volume).
# frontend/ and client/ are excluded via .dockerignore.
COPY server/ ./server/
COPY src/    ./src/
COPY static/ ./static/

# FastAPI serves from server/ using relative paths to data/ and ../static/
WORKDIR /app/server

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
