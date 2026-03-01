
# Listenr: Local Dataset Collection for ASR Training

Listenr is a privacy-first tool for collecting real-world audio and high-quality transcriptions, designed to help users and researchers build better automatic speech recognition (ASR) models. Instead of sending your voice to the cloud, Listenr runs entirely on your own hardware, using open-source models and local inference to capture, transcribe, and correct natural speech. The result is a rich, automatically-enhanced dataset that is ideal for training and fine-tuning ASR systems.

## Why Listenr?

- **Local-Only, Private by Design:** All processing happens on your machine. No audio or text is sent to external servers or cloud APIs.
- **Open Models, Open Hardware:** Uses Lemonade Server to run open-source models (Whisper.cpp, Llama.cpp, etc.) on your CPU, GPU, or NPU.
- **Automatic Correction Pipeline:** Transcriptions are improved using a local LLM, which leverages conversation context (previous and next utterances) to correct errors, punctuation, and grammar—creating a more accurate text corpus.
- **Real-World Data:** Collects natural, conversational speech in realistic environments, not just lab conditions.
- **Dataset-Ready Output:** Audio clips and their corrected transcriptions are saved together, ready for use in ASR model training or fine-tuning.

## How It Works

1. **Continuous Recording:** Listenr captures audio as you speak, segmenting speech automatically using VAD (voice activity detection).
2. **Local Transcription:** Each segment is transcribed using a local Whisper model via Lemonade Server.
3. **Contextual Correction:** A local LLM reviews the transcription, using surrounding conversation context to correct mistakes and improve fluency.
4. **Corpus Creation:** Both the original and corrected transcriptions, along with the audio, are saved for each segment—building a high-quality dataset as you use the tool.

## Why This Matters for ASR Training

Training and fine-tuning ASR models requires large, diverse, and accurate datasets. Listenr makes it easy to collect such data in real-world conditions, with minimal manual effort. The automatic correction pipeline means you get not just raw machine transcriptions, but also contextually-improved, human-like text—boosting the quality of your training corpus. This is especially valuable for:

- **Domain Adaptation:** Collect speech in your target environment (e.g., meetings, home, fieldwork) to adapt models to your needs.
- **Accent and Language Coverage:** Gather data from real users, capturing natural variation in speech.
- **Error Correction Research:** Study how LLMs can post-process ASR output for better downstream performance.

## Quick Start


## Features

- Continuous listening and segmentation (no start/stop buttons)
- Smart VAD-based speech detection
- Real-time transcription and correction pipeline
- Mobile-friendly web interface
- All data saved locally, organized for dataset use

## Quick Start

brew install ffmpeg

### Installation

```bash
# Install Python dependencies
pip install -r server/requirements.txt

# (Optional) Install ffmpeg if you want to support a wide range of audio file formats for upload or batch processing.
# For most live microphone/web use, ffmpeg is not required unless you plan to import/export non-wav files.
# Linux:
sudo apt install ffmpeg
# macOS:
brew install ffmpeg
```


### Start Lemonade Server

Lemonade Server is required for all local inference. See https://lemonade-server.ai for installation and model management.

```bash
# Install Lemonade Server (see https://lemonade-server.ai)
# Download and run models as needed (see Lemonade docs)

# Start Lemonade Server (default port 8000)
lemonade-server serve
```

### Run Listenr

```bash
# Start the Listenr web server
python server/app.py

# Open in browser:
# - Computer: http://localhost:5000
# - Phone: http://YOUR_IP:5000
```



### Enable LLM Correction (Optional)

To enable local LLM-based correction, make sure you have a compatible LLM model downloaded and available in Lemonade Server. Then set the environment variable before starting Listenr:

```bash
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


The system automatically:
- Detects when you start and stop speaking
- Records and segments your speech
- Transcribes and corrects each segment
- Saves audio and both raw/corrected text for every utterance

### Command Line


Use the Lemonade-powered ASR system directly from terminal:

```bash
# Transcribe an audio file
python unified_asr.py --audio path/to/audio.wav

# With LLM
python unified_asr.py --llm --audio path/to/audio.wav
```

### Mobile Usage


Great for hands-free, natural data collection on your phone:

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

asr = UnifiedASR(mode='cli')
asr.start_cli()  # Continuous terminal transcription
asr = UnifiedASR(mode='stream')
asr.start_stream(callback=callback)

## Unified ASR System

All functionality uses a single `unified_asr.py` implementation. You can use it for batch, streaming, or web-based collection. All outputs are saved in a consistent, dataset-friendly format (see Lemonade API docs for details).

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
model = gpt-oss-20b-mxfp4-GGUF
temperature = 0.1

[Whisper]
model = Whisper-Large-v3-Turbo
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


## Tips for High-Quality Data

- Speak naturally, as you would in real conversations
- Pause briefly between thoughts to help segmentation
- Use a quiet environment for best results, but real-world noise is valuable for robust models
- Use a good microphone if possible

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

If your hardware supports it, Lemonade Server can use your GPU for faster inference. See Lemonade documentation for details.

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


