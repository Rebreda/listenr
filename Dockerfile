# Functional Dockerfile for listenr CLI ASR project
FROM python:3.13-slim

# Install system dependencies for audio and build tools
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    pulseaudio-utils \
    build-essential \
    cmake \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy project files
COPY . /app

# Create and activate virtual environment
RUN python -m venv /app/venv
ENV VIRTUAL_ENV=/app/venv
ENV PATH="/app/venv/bin:$PATH"

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Expose config and output directories as volumes (optional)
VOLUME ["/root/.config/asr-indicator", "/app/output"]

# Default command: run main.py
CMD ["python", "main.py"]
