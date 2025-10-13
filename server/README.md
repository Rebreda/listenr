# Listenr Web Server

A lightweight, mobile-friendly web application for real-time voice transcription using Whisper ASR.

## Features

- **Real-time Recording**: Stream audio directly from your phone or computer's microphone
- **Automatic Transcription**: Powered by OpenAI's Whisper model via faster-whisper
- **Smart Processing**: Optional LLM post-processing for improved accuracy
- **Mobile-Friendly**: Beautiful responsive UI that works great on phones
- **Secure Storage**: UUID-based filenames, organized by date
- **JSON API**: Clean REST API with comprehensive metadata
- **Audio Playback**: Listen to recorded clips directly from the web interface

## Quick Start

### Prerequisites

- Python 3.8+
- ffmpeg (for audio conversion)
- CUDA (optional, for GPU acceleration)

### Installation

1. **Install system dependencies**:
   ```bash
   # Ubuntu/Debian
   sudo apt install ffmpeg

   # macOS
   brew install ffmpeg
   ```

2. **Install Python dependencies**:
   ```bash
   cd /home/g/Code/listenr
   pip install -r requirements.txt
   ```

3. **Run the server**:
   ```bash
   python server/app.py
   ```

4. **Open in browser**:
   - On your computer: http://localhost:5000
   - On your phone (same network): http://YOUR_IP:5000

## Configuration

Use environment variables to customize the server:

```bash
# Storage location (default: ~/listenr_web)
export LISTENR_STORAGE=~/my_recordings

# Server port (default: 5000)
export LISTENR_PORT=8080

# Server host (default: 0.0.0.0)
export LISTENR_HOST=0.0.0.0

# Enable LLM post-processing (default: false)
export LISTENR_USE_LLM=true

# Then run
python server/app.py
```

## Architecture

The system is organized into clean, separated layers:

```
listenr/
├── asr.py              # Core Whisper ASR + VAD (system-level)
├── web_asr.py          # Web-friendly JSON wrapper
├── server/
│   ├── app.py          # Flask web server
│   └── templates/
│       └── index.html  # Web UI
└── config_manager.py   # Configuration
```

### Layer Responsibilities

1. **asr.py**: Core ASR system
   - Whisper model loading and inference
   - Silero VAD for speech detection
   - Audio processing and recording
   - LLM integration (optional)

2. **web_asr.py**: Web integration layer
   - Wraps ASR system for web use
   - JSON request/response formatting
   - File storage and organization
   - Metadata management

3. **server/app.py**: Flask web server
   - HTTP endpoints and routing
   - File upload handling
   - Audio format conversion (ffmpeg)
   - Security and validation

4. **server/templates/index.html**: Client interface
   - Browser audio recording
   - Real-time streaming to server
   - Transcript display
   - Responsive mobile design

## API Endpoints

### `POST /upload`
Upload and transcribe audio.

**Request**: `multipart/form-data` with `audio_data` field

**Response**:
```json
{
  "success": true,
  "transcription": "Hello world",
  "corrected_text": "Hello, world!",
  "timestamp": "2025-10-12T10:30:00Z",
  "audio_url": "/audio/2025-10-12/clip_2025-10-12_abc123.wav",
  "duration": 2.5,
  "metadata": {
    "date": "2025-10-12",
    "uuid": "abc123",
    "llm_enabled": true,
    "llm_applied": true
  }
}
```

### `GET /audio/<date>/<filename>`
Retrieve stored audio file.

### `GET /transcript/<date>/<uuid>`
Retrieve stored transcript with full metadata.

### `GET /health`
Health check endpoint.

## LLM Post-Processing

Enable LLM post-processing for improved transcription accuracy:

1. **Install Ollama**:
   ```bash
   # Install from https://ollama.ai
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

2. **Pull a model**:
   ```bash
   ollama pull gemma2:2b
   ```

3. **Run with LLM enabled**:
   ```bash
   export LISTENR_USE_LLM=true
   python server/app.py
   ```

The LLM will:
- Fix punctuation and capitalization
- Correct common transcription errors
- Improve formatting

## Mobile Usage

Perfect for recording on-the-go:

1. Start the server on your computer
2. Find your local IP: `ip addr` (Linux) or `ipconfig` (Windows)
3. Open `http://YOUR_IP:5000` on your phone
4. Grant microphone permissions
5. Tap to record, tap to stop
6. View transcripts instantly

## Storage Structure

Recordings are organized by date:

```
~/listenr_web/
├── audio/
│   └── 2025-10-12/
│       ├── clip_2025-10-12_abc123.wav
│       └── clip_2025-10-12_def456.wav
└── transcripts/
    └── 2025-10-12/
        ├── transcript_2025-10-12_abc123.json
        └── transcript_2025-10-12_def456.json
```

## Tips

- **GPU Acceleration**: The server will automatically use CUDA if available
- **Model Selection**: Configure Whisper model in `config.ini` (base, small, medium, large)
- **Audio Quality**: Phone recordings work great! The system handles noise and compression well
- **Network**: Use HTTPS (nginx + certbot) for secure remote access
- **Performance**: First transcription takes longer (model loading), subsequent ones are fast

## Troubleshooting

**"ffmpeg not found"**:
```bash
sudo apt install ffmpeg  # Linux
brew install ffmpeg      # macOS
```

**"Could not access microphone"**:
- Grant browser microphone permissions
- Use HTTPS for remote access (required by browsers)

**Slow transcription**:
- Use a smaller Whisper model (base, small)
- Enable GPU acceleration
- Check CPU usage

## Production Deployment

For production use, run behind gunicorn + nginx:

```bash
# Install gunicorn
pip install gunicorn

# Run with gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 server.app:app

# Configure nginx for HTTPS and static file serving
```

## License

Part of the Listenr project.
