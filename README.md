# Listenr - Live Voice Transcription

**Hands-free, continuous voice transcription powered by Whisper ASR**

Listenr provides real-time speech-to-text transcription with automatic speech detection. No more clicking start/stop buttons - just speak naturally and watch your words appear instantly!

## Features

- ✨ **Continuous Listening**: Microphone stays open, transcripts appear automatically
- 🎯 **Smart Speech Detection**: Silero VAD automatically segments your speech
- 🚀 **Real-Time**: WebSocket streaming for minimal latency (~500ms-2s)
- 📱 **Mobile-First**: Beautiful touch-optimized interface for phones
- 🤖 **Optional LLM**: Post-process with Ollama for improved accuracy
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
# Install Ollama from https://ollama.ai
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull gemma2:2b

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

Use the unified ASR system directly from terminal:

```bash
# CLI mode (continuous terminal transcription)
python unified_asr.py

# With LLM
python unified_asr.py --llm

# Custom storage
python unified_asr.py --storage ~/my_clips
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
├── unified_asr.py          # Core ASR system (Whisper + VAD + JSON)
├── config_manager.py       # Configuration management
├── llm_processor.py        # Optional LLM post-processing
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
Whisper Transcription
    ↓ (optional)
LLM Post-Processing
    ↓
JSON Response → WebSocket → Browser
    ↓
Display + Save to Disk
```

## Unified ASR System

All functionality uses a single `unified_asr.py` implementation that works in three modes:

### CLI Mode
```python
from unified_asr import UnifiedASR

asr = UnifiedASR(mode='cli')
asr.start_cli()  # Continuous terminal transcription
```

### Web Mode (Single File)
```python
asr = UnifiedASR(mode='web')
result = asr.process_audio(audio_data, sample_rate)
# Returns JSON with transcription + metadata
```

### Stream Mode (Continuous)
```python
def callback(result):
    print(result['transcription'])

asr = UnifiedASR(mode='stream')
asr.start_stream(callback=callback)
```

All modes return consistent JSON:
```json
{
  "success": true,
  "transcription": "raw whisper output",
  "corrected_text": "LLM-corrected version",
  "timestamp": "2025-10-12T10:30:00Z",
  "audio": {
    "path": "/path/to/clip.wav",
    "url": "/audio/2025-10-12/clip_abc123.wav",
    "duration": 3.5,
    "sample_rate": 16000
  },
  "metadata": {
    "date": "2025-10-12",
    "uuid": "abc123",
    "llm_applied": true,
    "language": "en",
    "mode": "stream"
  }
}
```

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

[Whisper]
model_size = base             # tiny, base, small, medium, large
device = cpu                  # cpu or cuda
compute_type = int8           # int8, float16, float32

[LLM]
enabled = true
model = gemma2:2b
temperature = 0.1
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
- Use smaller Whisper model (base or tiny)
- Enable GPU if available
- Check CPU usage
- Reduce `max_silence_duration_s` for faster segmentation

**Too many/few segments**:
- Adjust `speech_threshold` in config.ini
- Adjust `max_silence_duration_s`
- Check background noise levels

## Advanced Usage

### Custom Integration

```python
from unified_asr import UnifiedASR

# Create ASR instance
asr = UnifiedASR(
    mode='web',
    use_llm=True,
    storage_base='~/my_storage'
)

# Process audio file
import soundfile as sf
audio, sr = sf.read('recording.wav')

result = asr.process_audio(audio, sr, metadata={'user': 'john'})

print(result['transcription'])
print(result['audio']['path'])
```

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

- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - Fast inference
- [Silero VAD](https://github.com/snakers4/silero-vad) - Voice activity detection
- [Ollama](https://ollama.ai) - Local LLM inference

---

**Enjoy hands-free, continuous voice transcription!** 🎤
