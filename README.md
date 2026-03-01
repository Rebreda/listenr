
# Listenr: Local Dataset Collection for ASR Training

Listenr is a privacy-first tool for collecting real-world audio and high-quality transcriptions, designed to help build better automatic speech recognition (ASR) models. All processing runs locally on your hardware via [Lemonade Server](https://lemonade-server.ai) — no audio or text leaves your machine.

## Why Listenr?

- **Local-only, private by design.** No cloud APIs. All inference runs on your CPU, GPU, or NPU via Lemonade Server.
- **Open models.** Uses Whisper.cpp for transcription and any GGUF-compatible LLM for post-processing correction.
- **Automatic correction pipeline.** A local LLM cleans up punctuation, grammar, and homophones — producing a higher-quality training corpus than raw Whisper output alone.
- **Real-world data.** Collects natural, conversational speech in realistic environments.
- **Dataset-ready output.** Every utterance is saved with its audio clip and both raw/corrected transcriptions. A single command builds train/dev/test splits.

## How It Works

1. **Capture.** `listenr_cli.py` streams your microphone to Lemonade's `/realtime` WebSocket endpoint at ~85 ms chunks.
2. **VAD.** Lemonade's built-in server-side voice activity detection segments speech boundaries automatically.
3. **Transcribe.** Lemonade runs Whisper.cpp on each speech segment and returns a transcript.
4. **Correct (optional).** The transcript is sent to a local LLM via Lemonade's chat completions API for punctuation and grammar correction.
5. **Save.** Each utterance is saved as a `.wav` file and a `.json` metadata file, organized by date.
6. **Build dataset.** `build_dataset.py` aggregates all saved recordings into train/dev/test CSV (or HuggingFace `datasets`) splits.

## Project Layout

```
listenr/
├── listenr_cli.py       # CLI: mic → Lemonade /realtime → save recordings
├── unified_asr.py       # Core: LemonadeUnifiedASR class (streaming + batch)
├── llm_processor.py     # Lemonade HTTP wrappers: transcription + LLM correction
├── config_manager.py    # Config loader (~/.config/listenr/config.ini)
├── build_dataset.py     # Build train/dev/test splits from saved recordings
├── server/
│   ├── app.py           # Flask/WebSocket server for browser-based capture
│   └── templates/
│       ├── index_visual.html   # Full-featured web UI
│       └── index.html          # Simple web UI
├── requirements.txt     # Python dependencies
└── prd/                 # Design documents
```

## Requirements

- [Lemonade Server](https://lemonade-server.ai) running on `localhost:8000`
- A Whisper model loaded in Lemonade (`Whisper-Tiny`, `Whisper-Base`, or `Whisper-Small`)
- Python 3.11+ with `uv` (recommended) or `pip`

## Installation

```bash
# Clone the repo
git clone https://github.com/Rebreda/listenr
cd listenr

# Install dependencies
uv pip install -r requirements.txt
# or: pip install -r requirements.txt
```

## Start Lemonade Server

Lemonade Server must be running before using Listenr. See [lemonade-server.ai](https://lemonade-server.ai) for installation.

```bash
lemonade-server serve
```

Load the models you plan to use (Lemonade auto-downloads on first use):

```bash
# Load Whisper for transcription
lemonade-server load Whisper-Small

# Load an LLM for correction (optional)
lemonade-server load Qwen3-0.6B-GGUF
```

## Usage

### CLI — Real-Time Microphone Capture

The primary tool for dataset collection. Streams your mic to Lemonade and saves each utterance.

```bash
# Record and save everything (default)
uv run listenr_cli.py

# Don't save to disk — just print transcriptions
uv run listenr_cli.py --no-save

# Also print the raw Whisper output before LLM correction
uv run listenr_cli.py --show-raw
```

Output while running:

```
🎤 Listenr CLI — streaming to Lemonade
   Model  : Whisper-Small
   WS URL : ws://localhost:8001/realtime?model=Whisper-Small
   LLM    : enabled (Qwen3-0.6B-GGUF)
   Save   : yes → ~/.listenr/audio_clips
   Press Ctrl+C to stop.

  [ASR] Hello, this is a test.
  [SAVED] ~/.listenr/audio_clips/audio/2026-02-28/clip_2026-02-28_abc123.wav (2.4s, improved=True)
```

### Build a Dataset

After collecting recordings, generate train/dev/test splits:

```bash
# Default: 80/10/10 CSV splits in ~/listenr_dataset/
uv run build_dataset.py

# Custom output and split ratio
uv run build_dataset.py --output ~/my_dataset --split 90/5/5

# Exclude very short clips
uv run build_dataset.py --min-duration 1.0

# HuggingFace datasets format
uv run build_dataset.py --format hf

# Preview stats without writing files
uv run build_dataset.py --dry-run
```

Output CSV columns: `uuid`, `split`, `audio_path`, `raw_transcription`, `corrected_transcription`, `is_improved`, `duration_s`, `sample_rate`, `whisper_model`, `llm_model`, `timestamp`.

### Batch Transcription

Transcribe a single audio file using the Lemonade HTTP API:

```bash
uv run unified_asr.py --audio path/to/audio.wav --whisper-model Whisper-Small

# With LLM correction
uv run unified_asr.py --llm --audio path/to/audio.wav
```

### Web Interface

For browser-based capture (useful for phone recording or multi-user collection):

```bash
python server/app.py
# Open http://localhost:5000

# With LLM correction enabled
LISTENR_USE_LLM=true python server/app.py
```

Environment variables for the web server:

| Variable | Default | Description |
|---|---|---|
| `LISTENR_PORT` | `5000` | HTTP port |
| `LISTENR_HOST` | `0.0.0.0` | Bind address |
| `LISTENR_USE_LLM` | `false` | Enable LLM post-processing |
| `LISTENR_STORAGE` | from config | Storage base directory |

For mobile access, open `http://YOUR_LAN_IP:5000` on your phone. Note: browsers require HTTPS for microphone access on non-localhost origins — use a reverse proxy with TLS for remote use.

## Configuration

Config is stored at `~/.config/listenr/config.ini` and created with defaults on first run.

```ini
[Lemonade]
# HTTP API base — WebSocket port is discovered dynamically via GET /api/v1/health
api_base = http://localhost:8000/api/v1

[Whisper]
# Lemonade whisper.cpp backend: Whisper-Tiny, Whisper-Base, Whisper-Small
model = Whisper-Small

[Audio]
# Lemonade /realtime requires 16kHz mono PCM16 — do not change sample_rate or channels
sample_rate = 16000
channels = 1
# Chunk size in frames per mic read (~85ms at 16kHz = 1360 frames, as recommended by Lemonade spec)
blocksize = 1360
input_device = default  # 'default' or device index (see sounddevice docs)

[VAD]
# Server-side VAD — these are sent to Lemonade via session.update, not processed locally
threshold = 0.01          # RMS energy threshold for speech detection
silence_duration_ms = 800 # Silence (ms) to trigger end of utterance
prefix_padding_ms = 250   # Minimum speech (ms) before transcription triggers

[LLM]
enabled = true
model = Qwen3-0.6B-GGUF   # Must be loaded in Lemonade
api_base = http://localhost:8000/api/v1
temperature = 0.3
max_tokens = 150
timeout = 10

[Storage]
audio_clips_path = ~/.listenr/audio_clips
audio_clips_enabled = true
retention_days = 90
max_storage_gb = 10
clip_format = wav
```

### VAD Tuning

VAD runs server-side in Lemonade. Adjust in `[VAD]`:

| Goal | Setting |
|---|---|
| Segment shorter utterances | Lower `silence_duration_ms` (e.g. `500`) |
| Avoid cutting off speech | Raise `silence_duration_ms` (e.g. `1200`) |
| Ignore background noise | Raise `threshold` (e.g. `0.05`) |
| Capture quiet speech | Lower `threshold` (e.g. `0.005`) |

## Storage Layout

```
~/.listenr/audio_clips/
├── audio/
│   └── 2026-02-28/
│       ├── clip_2026-02-28_abc123.wav
│       └── clip_2026-02-28_def456.wav
└── transcripts/
    └── 2026-02-28/
        ├── transcript_2026-02-28_abc123.json
        └── transcript_2026-02-28_def456.json
```

Each transcript JSON:

```json
{
  "uuid": "abc123",
  "timestamp": "2026-02-28T14:30:00+00:00",
  "audio_path": "/home/user/.listenr/audio_clips/audio/2026-02-28/clip_2026-02-28_abc123.wav",
  "raw_transcription": "im going to the store two buy some milk",
  "corrected_transcription": "I'm going to the store to buy some milk.",
  "is_improved": true,
  "whisper_model": "Whisper-Small",
  "llm_model": "Qwen3-0.6B-GGUF",
  "duration_s": 2.4,
  "sample_rate": 16000
}
```

## Troubleshooting

**`Could not discover Lemonade websocket port`**
Lemonade Server is not running or not reachable on port 8000. Start it with `lemonade-server serve` and check `GET http://localhost:8000/api/v1/health`.

**No transcriptions appear**
- Confirm the Whisper model is loaded: `GET http://localhost:8000/api/v1/models`
- Lower `threshold` in `[VAD]` if your mic input is quiet
- Try `--show-raw` to see if raw Whisper output is coming through

**LLM correction not working**
- Check `LLM.enabled = true` in config
- Confirm the LLM model name matches a model loaded in Lemonade
- LLM errors are non-fatal — the raw transcript is saved regardless

**Browser mic not working**
- Grant microphone permissions when prompted
- On non-localhost origins, browsers require HTTPS for mic access

**Too many / too few segments**
Adjust `[VAD] silence_duration_ms` and `threshold` in your config file.

## License

Mozilla Public License Version 2.0 — see `LICENSE`.

## Acknowledgments

- [Lemonade Server](https://lemonade-server.ai) — unified local inference API
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) — fast local ASR
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — fast local LLMs


