# Listenr - Live Voice Transcription

**Hands-free, continuous voice transcription powered by Lemonade Server (Whisper.cpp + LLMs)**

Listenr provides real-time speech-to-text transcription with automatic speech detection. No more clicking start/stop buttons - just speak naturally and watch your words appear instantly!

## Features

- ✨ **Continuous Listening**: Microphone stays open, transcripts appear automatically
- 🎯 **Smart Speech Detection**: Silero VAD automatically segments your speech
- 🚀 **Real-Time**: WebSocket streaming for minimal latency (~500ms-2s)
- 📱 **Mobile-First**: Beautiful touch-optimized interface for phones
- 🤖 **Optional LLM**: Post-process with local LLMs via Lemonade Server for improved accuracy
- 💾 **Auto-Save**: All audio clips and transcripts saved with metadata
- 🌐 **JSON API**: Consistent structured responses everywhere

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r server/requirements.txt

# Install ffmpeg (if not already installed)
# Linux:
sudo apt install ffmpeg
# macOS:
brew install ffmpeg
```

### Run the Server

```bash
# Basic usage
python server/app.py

# Open in browser:
# - Computer: http://localhost:5000
# - Phone: http://YOUR_IP:5000
```


### With LLM (Optional)

```bash
# Install Lemonade Server (https://github.com/lemonade-org/lemonade)
# Download and run models as needed (see Lemonade docs)

# Start Lemonade Server (default port 8000)
lemonade-server serve

# Run with LLM enabled
export LISTENR_USE_LLM=true
python server/app.py
```

## Usage

### Web Interface

1. Open http://localhost:5000 in your browser
2. Click the big microphone button once
3. Grant microphone permissions
4. Start talking naturally
5. Pause between thoughts
6. Watch transcripts appear automatically!

**That's it!** No more clicking buttons. The system automatically:
- Detects when you start speaking
- Records your speech
- Detects when you pause/finish
- Transcribes the audio
- Displays the transcript
- Saves everything to disk

### Command Line


Use the Lemonade-powered ASR system directly from terminal:

```bash
# Transcribe an audio file
python unified_asr.py --audio path/to/audio.wav

# With LLM
python unified_asr.py --llm --audio path/to/audio.wav
```

### Mobile Usage

Perfect for hands-free recording on your phone:

1. Start server on your computer
2. Find your IP: `ip addr show | grep inet`
3. Open `http://YOUR_IP:5000` on your phone
4. Tap mic button once
5. Put phone in pocket
6. Start talking!

All transcripts save automatically with audio clips and metadata.

## Architecture

### Clean Separation

```
listenr/
├── unified_asr.py          # Core ASR system (Lemonade API: Whisper.cpp + LLM)
├── config_manager.py       # Configuration management
├── llm_processor.py        # Lemonade API wrappers for LLM/ASR
├── server/
│   ├── app.py              # Flask WebSocket server
│   ├── requirements.txt    # Python dependencies
│   └── templates/
│       └── index.html      # Web interface
└── README.md               # This file
```

### Data Flow

```
Browser Microphone
    ↓ (continuous audio)
WebSocket Connection
    ↓ (base64 chunks)
UnifiedASR.process_vad_chunk()
    ↓ (VAD segmentation)
Lemonade Whisper Transcription
    ↓ (optional)
Lemonade LLM Post-Processing
    ↓
JSON Response → WebSocket → Browser
    ↓
Display + Save to Disk
```

## Unified ASR System

All functionality uses a single `unified_asr.py` implementation that works in three modes:

asr = UnifiedASR(mode='cli')
asr.start_cli()  # Continuous terminal transcription
asr = UnifiedASR(mode='stream')
asr.start_stream(callback=callback)

### CLI Mode
```bash
python unified_asr.py --audio path/to/audio.wav
```

All modes return consistent JSON (see Lemonade API docs for details).

## Configuration

Edit `config.ini` to customize:

```ini
[VAD]
speech_threshold = 0.5        # Higher = less sensitive
min_speech_duration_s = 0.3   # Minimum speech length
max_silence_duration_s = 0.8  # Pause before ending segment

[Audio]
sample_rate = 16000
leading_silence_s = 0.3       # Silence before speech
trailing_silence_s = 0.3      # Silence after speech

## Lemonade Server Model Selection
#
# Model names must match those available in your Lemonade Server instance.
# See Lemonade docs for model management (install, load, list, etc).

[LLM]
enabled = true
model = Qwen3-0.6B-GGUF
temperature = 0.1

[Whisper]
model = Whisper-Tiny
```

Environment variables override config:

```bash
export LISTENR_STORAGE=~/my_recordings  # Storage directory
export LISTENR_PORT=8080                # Server port
export LISTENR_HOST=0.0.0.0             # Server host
export LISTENR_USE_LLM=true             # Enable LLM
```

## Storage

All recordings are automatically organized by date:

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

Each transcript JSON includes:
- Raw transcription
- LLM-corrected text (if enabled)
- Audio file path and URL
- Duration, sample rate
- Timestamp, UUID
- Language detection
- All metadata

## Tips

### For Best Results

1. **Speak Clearly**: Normal conversational pace works best
2. **Pause Between Thoughts**: Helps VAD segment naturally (0.5-1s)
3. **Quiet Background**: Reduces false positives
4. **Good Microphone**: Phone/laptop mics work fine, headset is better
5. **Local Network**: Keep phone and server on same network for best latency

### Tuning VAD

If you get too many/few segments:

```ini
# More sensitive (segments shorter speech)
[VAD]
speech_threshold = 0.3
max_silence_duration_s = 0.5

# Less sensitive (waits longer)
[VAD]
speech_threshold = 0.7
max_silence_duration_s = 1.5
```

### GPU Acceleration

```ini
[Whisper]
device = cuda           # Use GPU
compute_type = float16  # GPU precision
```

Much faster transcription on NVIDIA GPUs!

## Troubleshooting

**WebSocket keeps disconnecting**:
- Check firewall settings
- Verify network stability
- Check server logs for errors

**Microphone not working**:
- Grant browser microphone permissions
- Check if another app is using the mic
- Try HTTPS for remote access (required by browsers)


**Transcripts are delayed**:
- Use smaller Whisper model (e.g., Whisper-Tiny)
- Ensure Lemonade Server is running and responsive
- Check CPU usage
- Reduce `max_silence_duration_s` for faster segmentation

**Too many/few segments**:
- Adjust `speech_threshold` in config.ini
- Adjust `max_silence_duration_s`
- Check background noise levels

## Advanced Usage


### Custom Integration

Use the Lemonade API directly for advanced use cases. See Lemonade Server documentation for full API details and endpoints.

### Custom Callback

```python
def my_callback(result):
    print(f"[{result['timestamp']}] {result['transcription']}")
    # Send to database, API, etc.

asr = UnifiedASR(mode='stream')
asr.start_stream(callback=my_callback)
```

## Documentation

- `LIVE_STREAMING.md` - Deep dive on live streaming mode
- `config.ini` - All configuration options
- `prd/` - Original design documents

## License

Mozilla Public License Version 2.0 - see `LICENSE`

## Acknowledgments

- [Lemonade Server](https://github.com/lemonade-org/lemonade) - Unified local inference API
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) - Fast local ASR
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Fast local LLMs

---

**Enjoy hands-free, continuous voice transcription!** 🎤
