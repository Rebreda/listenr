
# Listenr: Local Dataset Collection for ASR Training

Listenr is a privacy-first tool for collecting real-world audio and high-quality transcriptions, designed to help build better automatic speech recognition (ASR) models. All processing runs locally on your hardware via [Lemonade Server](https://lemonade-server.ai) — no audio or text leaves your machine.

![Listenr CLI streaming — example output](screenshot.png)

## Why Listenr?

- **Local-only, private by design.** No cloud APIs. All inference runs on your CPU, GPU, or NPU via Lemonade Server.
- **Open models.** Uses Whisper.cpp for transcription and any GGUF-compatible LLM for post-processing correction.
- **Automatic correction pipeline.** A local LLM cleans up punctuation, grammar, and homophones — producing a higher-quality training corpus than raw Whisper output alone.
- **Real-world data.** Collects natural, conversational speech in realistic environments.
- **Dataset-ready output.** Every utterance is saved with its audio clip, a per-clip JSON, and appended to a single `manifest.jsonl`. One command builds train/dev/test splits.

## How It Works

1. **Capture.** `listenr` streams your microphone to Lemonade's `/realtime` WebSocket in ~85 ms chunks. Audio is captured at the device's native rate and resampled to 16 kHz before sending.
2. **VAD.** Lemonade's built-in server-side voice activity detection segments speech boundaries automatically.
3. **Transcribe.** Lemonade runs Whisper.cpp on each speech segment and streams back interim and final transcripts.
4. **Correct (optional).** The final transcript is sent to a local LLM via Lemonade's chat completions API. The LLM returns a cleaned transcript, an `is_improved` flag, and content `categories`.
5. **Save.** Each utterance is saved as a `.wav` clip and appended to `manifest.jsonl`.
6. **Build dataset.** `build_dataset.py` reads the manifest and writes train/dev/test CSV splits.



## Requirements

- [Lemonade Server](https://lemonade-server.ai) running on `localhost:8000`
- Python 3.13+ with `uv` (recommended) or `pip`
- A microphone accessible via PipeWire or ALSA

## Installation

```bash
git clone https://github.com/Rebreda/listenr
cd listenr
uv pip install -e .
```

Then run commands via `uv run` (no activation needed):

```bash
uv run listenr
```

Or activate the venv once per session:

```bash
source .venv/bin/activate
listenr
```

## Start Lemonade Server

```bash
lemonade-server serve
```

Listenr will automatically call `POST /api/v1/load` on startup to load the configured models. On first use, Lemonade will download them.

## Usage

### CLI — Real-Time Microphone Capture

```bash
# Record and save everything (default)
uv run listenr

# Don't save to disk — just print transcriptions
uv run listenr --no-save

# Also print the raw Whisper output before LLM correction
uv run listenr --show-raw

# Verbose debug output (WebSocket messages, mic RMS, etc.)
uv run listenr --debug
```

Example output:

```
🎤 Listenr CLI — streaming to Lemonade
   Model  : Whisper-Large-v3-Turbo
   WS URL : ws://localhost:9000/realtime?model=Whisper-Large-v3-Turbo
   LLM    : enabled (gpt-oss-20b-mxfp4-GGUF)
   Save   : yes → ~/.listenr/audio_clips
   Press Ctrl+C to stop.

  [ASR] I'm going to the store to buy some milk.  [dictation]
  [SAVED] ~/.listenr/audio_clips/audio/2026-02-28/clip_2026-02-28_abc123.wav (2.4s)
```

Press **Ctrl+C** to stop. Listenr will unload all models from Lemonade before exiting.

### Build a Dataset

After collecting recordings, generate train/dev/test splits from `manifest.jsonl`:

```bash
# Default: 80/10/10 CSV splits in ~/listenr_dataset/
uv run listenr-build-dataset

# Custom output directory and split ratio
uv run listenr-build-dataset --output ~/my_dataset --split 90/5/5

# Exclude very short clips
uv run listenr-build-dataset --min-duration 1.0

# HuggingFace datasets format
uv run listenr-build-dataset --format hf

# Preview stats without writing files
uv run listenr-build-dataset --dry-run
```

Output CSV columns: `uuid`, `split`, `audio_path`, `raw_transcription`, `corrected_transcription`, `is_improved`, `categories`, `duration_s`, `sample_rate`, `whisper_model`, `llm_model`, `timestamp`.

### Batch Transcription

Transcribe a single audio file:

```bash
python -m listenr.unified_asr --audio path/to/audio.wav --whisper-model Whisper-Large-v3-Turbo

# With LLM correction
python -m listenr.unified_asr --llm --audio path/to/audio.wav
```

## Configuration

Config is created with defaults at `~/.config/listenr/config.ini` on first run.


### Finding your input device

```bash
python -c "import sounddevice as sd; [print(f'{i}: {d[\"name\"]}') for i, d in enumerate(sd.query_devices()) if d['max_input_channels'] > 0]"
```

Set `input_device` to the device name (partial match works) or its index number.

### VAD Tuning

| Goal | Setting |
|---|---|
| Shorter segments | Lower `silence_duration_ms` (e.g. `500`) |
| Avoid cutting off speech | Raise `silence_duration_ms` (e.g. `1200`) |
| Ignore background noise | Raise `threshold` (e.g. `0.05`) |
| Capture quiet speech | Lower `threshold` (e.g. `0.005`) |


### manifest.jsonl

One JSON object per line — append-only, easy to query:

```bash
# All improved clips
jq 'select(.is_improved == true)' ~/.listenr/audio_clips/manifest.jsonl

# Clips tagged as commands
jq 'select(.categories[] == "command")' ~/.listenr/audio_clips/manifest.jsonl

# Load into pandas
python -c "import pandas as pd; df = pd.read_json('~/.listenr/audio_clips/manifest.jsonl', lines=True); print(df.head())"
```

### manifest.jsonl

**No transcriptions appear / `[SAVE SKIPPED] pcm_buffer is empty`**
- Check that Lemonade is running: `curl http://localhost:8000/api/v1/health`
- Run with `--debug` to see mic RMS values and WebSocket messages
- If RMS stays near `0.000`, your `input_device` is wrong — list devices and update config (see above)
- Lower `threshold` in `[VAD]` if your mic is quiet

**LLM correction not working / model answers the transcription instead of fixing it**
- Confirm `LLM.enabled = true` and the model name matches one loaded in Lemonade
- Check `curl http://localhost:8000/api/v1/models` to see loaded models
- LLM errors are non-fatal — the raw transcript is saved regardless

**`Could not discover Lemonade websocket port`**
Lemonade is not running or not reachable on port 8000. Run `lemonade-server serve` first.

**Too many / too few segments**
Adjust `[VAD] silence_duration_ms` and `threshold` in your config.

## Available Models (via Lemonade)

| Model | Type | Notes |
|---|---|---|
| `Whisper-Base` | ASR | Fast, lower accuracy |
| `Whisper-Large-v3-Turbo` | ASR | Best accuracy |
| `gpt-oss-20b-mxfp4-GGUF` | LLM | Good correction quality |
| `Gemma-3-4b-it-GGUF` | LLM | Lighter alternative |
| `DeepSeek-Qwen3-8B-GGUF` | LLM | Lighter alternative |

List all models available on your Lemonade instance:
```bash
curl -s http://localhost:8000/api/v1/models | python3 -c "import sys,json; [print(m['id']) for m in json.load(sys.stdin)['data']]"
```

## License

Mozilla Public License Version 2.0 — see `LICENSE`.

## Acknowledgments

- [Lemonade Server](https://lemonade-server.ai) — unified local inference API
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) — fast local ASR
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — fast local LLMs

