# Listenr Live Streaming Mode

**Continuous, hands-free voice transcription with real-time updates!**

No more start/stop buttons - just open the page, enable your mic, and speak naturally. Transcripts appear live as you talk.

## Quick Start

```bash
# Install dependencies
pip install flask-sock

# Run the live streaming server
python server/app_stream.py

# Open in browser
# Computer: http://localhost:5000
# Phone: http://YOUR_IP:5000
```

## What's Different?

### Old Way (app.py)
- Click to start recording
- Speak
- Click to stop
- Wait for upload & processing
- See transcript

### New Way (app_stream.py)
- Click once to enable mic
- **Microphone stays open**
- Speak naturally (pause between thoughts)
- **Transcripts appear automatically**
- Keep talking, keep getting transcripts!

## How It Works

### Architecture

1. **Browser** → Continuous mic access, streams audio chunks via WebSocket
2. **WebSocket** → Real-time bidirectional connection (no HTTP overhead)
3. **UnifiedASR** → Single ASR implementation with VAD
4. **VAD (Voice Activity Detection)** → Automatically detects when you're speaking
5. **Whisper** → Transcribes each speech segment
6. **JSON Response** → Sent back instantly via WebSocket

### The Magic: VAD

The system uses Silero VAD to:
- Detect when you start speaking
- Collect audio while you talk
- Detect when you pause/stop
- Automatically transcribe the segment
- Reset and wait for next speech

This means:
- No manual start/stop needed
- Natural conversation flow
- Optimal segment boundaries
- No awkward button pressing

## Unified ASR System

### One File, Three Modes

**unified_asr.py** replaces both `asr.py` and `web_asr.py` with a single implementation:

```python
from unified_asr import UnifiedASR

# CLI mode (like original asr.py)
asr = UnifiedASR(mode='cli')
asr.start_cli()

# Web mode (like web_asr.py)
asr = UnifiedASR(mode='web')
result = asr.process_audio(audio_data, sample_rate)

# Stream mode (new!)
asr = UnifiedASR(mode='stream')
asr.start_stream(callback=my_callback)
```

### JSON Everywhere

All methods return consistent JSON structures:

```json
{
  "success": true,
  "transcription": "raw whisper output",
  "corrected_text": "LLM-corrected text (if enabled)",
  "timestamp": "2025-10-12T10:30:00Z",
  "audio": {
    "path": "/home/user/listenr_web/audio/2025-10-12/clip_abc123.wav",
    "url": "/audio/2025-10-12/clip_abc123.wav",
    "duration": 3.5,
    "sample_rate": 16000
  },
  "metadata": {
    "date": "2025-10-12",
    "uuid": "abc123",
    "transcript_path": "...",
    "llm_enabled": true,
    "llm_applied": true,
    "language": "en",
    "language_probability": 0.98,
    "mode": "stream"
  }
}
```

## Usage

### Basic Usage

```bash
# Start the server
python server/app_stream.py

# Open http://localhost:5000
# Click the big mic button
# Start talking!
```

### With LLM Post-Processing

```bash
# Enable LLM for better accuracy
export LISTENR_USE_LLM=true
python server/app_stream.py
```

### Custom Storage

```bash
export LISTENR_STORAGE=~/my_recordings
python server/app_stream.py
```

### On Your Phone

Perfect for hands-free mobile transcription:

1. Start server on your computer
2. Find your IP: `ip addr show | grep inet`
3. Open `http://YOUR_IP:5000` on phone
4. Tap mic button once
5. Put phone in pocket, start talking!

## Features

### Real-Time Streaming
- Continuous microphone access
- Audio chunks sent via WebSocket
- No file uploads needed
- Minimal latency

### Smart Speech Detection
- Automatic speech boundary detection
- Ignores background noise
- Natural pause handling
- Configurable sensitivity (config.ini)

### Live UI Feedback
- Mic button pulses when processing
- "LIVE" badge shows active listening
- Status updates in real-time
- Transcripts appear instantly

### Mobile-Optimized
- Touch-friendly interface
- Big tap targets
- Responsive design
- Works great on phones

### All Your Data
- Audio clips saved automatically
- JSON transcripts with metadata
- Date-organized storage
- Playback links in UI

## Configuration

All settings from config.ini apply:

```ini
[VAD]
speech_threshold = 0.5
min_speech_duration_s = 0.3
max_silence_duration_s = 0.8

[Audio]
leading_silence_s = 0.3
trailing_silence_s = 0.3
sample_rate = 16000

[Whisper]
model_size = base
device = cpu

[LLM]
enabled = true
model = gemma2:2b
```

Adjust these to fine-tune:
- `speech_threshold`: Higher = less sensitive (ignore quiet sounds)
- `min_speech_duration_s`: Minimum speech length to transcribe
- `max_silence_duration_s`: How long to wait before ending segment

## Technical Details

### WebSocket Protocol

**Client → Server (audio data)**:
```json
{
  "type": "audio",
  "audio": "base64_encoded_float32_array",
  "sample_rate": 16000,
  "format": "float32"
}
```

**Server → Client (transcription)**:
```json
{
  "type": "transcription",
  "data": { ... full JSON result ... }
}
```

**Server → Client (status)**:
```json
{
  "type": "status",
  "message": "connected",
  "llm_enabled": true
}
```

### Audio Pipeline

1. Browser MediaRecorder → ScriptProcessor
2. ScriptProcessor → Float32Array chunks (4096 samples)
3. Chunks → Base64 encoding
4. WebSocket → Server
5. Server → Accumulate to VAD chunk size (512 samples)
6. VAD → Detect speech boundaries
7. Speech segment → Whisper
8. Whisper result → Optional LLM
9. JSON → WebSocket → Browser
10. Browser → Display transcript

### Performance

- **Latency**: ~500ms-2s from speech end to transcript
- **Throughput**: Handles continuous speech
- **Memory**: Models loaded once, shared across connections
- **Concurrency**: Multiple clients supported (each gets own VAD state)

## Comparison

| Feature | app.py (Upload) | app_stream.py (Live) |
|---------|----------------|---------------------|
| Mode | One-shot | Continuous |
| Mic Control | Start/Stop buttons | Always on |
| Connection | HTTP POST | WebSocket |
| Upload | File upload | Streaming chunks |
| VAD | None | Built-in |
| Latency | 2-5s | 0.5-2s |
| Use Case | Single recordings | Hands-free conversation |

## Tips

### For Best Results

1. **Speak Clearly**: Normal conversational pace works best
2. **Pause Between Thoughts**: Helps VAD segment naturally
3. **Quiet Background**: Reduces false positives
4. **Good Mic**: Phone/laptop mics work fine, better mic = better results
5. **Network**: Local network for best latency

### Troubleshooting

**"Mic not working"**:
- Check browser permissions
- Try HTTPS for remote access (browsers require secure context)
- Make sure no other app is using the mic

**"Transcripts delayed"**:
- Check CPU usage (transcription is compute-heavy)
- Use smaller Whisper model (base or tiny)
- Enable GPU if available

**"Too many/few segments"**:
- Adjust `speech_threshold` in config.ini
- Adjust `max_silence_duration_s` for faster/slower segmentation

**"WebSocket disconnects"**:
- Check server logs
- Verify firewall settings
- Use stable network connection

## CLI Mode with Unified ASR

You can also use the unified system from command line:

```bash
# CLI mode (replaces original asr.py)
python unified_asr.py

# With LLM
python unified_asr.py --llm

# Custom storage
python unified_asr.py --storage ~/my_clips
```

This gives you the same live transcription experience in the terminal!

## Migration from Old System

### If you were using asr.py

```python
# Old
from asr import WhisperASR
asr = WhisperASR()
asr.start()

# New
from unified_asr import UnifiedASR
asr = UnifiedASR(mode='cli')
asr.start_cli()
```

### If you were using web_asr.py

```python
# Old
from web_asr import WebASRProcessor
processor = WebASRProcessor()
result = processor.process_audio(audio_data, sample_rate)

# New
from unified_asr import UnifiedASR
asr = UnifiedASR(mode='web')
result = asr.process_audio(audio_data, sample_rate)
```

All methods return the same JSON structure!

## Why This Is Better

1. **No Button Fatigue**: No more clicking start/stop repeatedly
2. **Natural Flow**: Talk naturally, system handles segmentation
3. **Instant Feedback**: See transcripts as you speak
4. **Mobile-Friendly**: Perfect for phone use (hands-free!)
5. **Unified Codebase**: One ASR implementation for all modes
6. **JSON Everywhere**: Consistent data format
7. **Real-Time**: WebSocket = lower latency than HTTP

## Next Steps

- **Try It**: `python server/app_stream.py` and open in browser
- **Customize**: Edit config.ini to tune VAD sensitivity
- **Mobile**: Access from your phone on local network
- **Integrate**: Use UnifiedASR in your own projects

Enjoy truly hands-free, continuous voice transcription!
