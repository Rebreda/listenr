# Recording

`listenr` streams your microphone to Lemonade in real time, receives Whisper
transcriptions, optionally corrects them with a local LLM, and saves each
utterance as a `.wav` clip.

---

## Basic usage

```bash
# Record and save everything (default)
uv run listenr

# Print transcriptions only — nothing saved to disk
uv run listenr --no-save

# Also print the raw Whisper output before LLM correction
uv run listenr --show-raw

# Verbose debug output (WebSocket messages, mic RMS, etc.)
uv run listenr --debug
```

Press **Ctrl+C** to stop. Listenr will unload all models from Lemonade before exiting.

---

## Example output

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

---

## How it works

1. **Capture** — audio is streamed from your microphone at the device's native
   sample rate and resampled to 16 kHz PCM-16 before sending.
2. **VAD** — Lemonade's server-side voice activity detection segments speech
   boundaries automatically.
3. **Transcribe** — Lemonade runs Whisper.cpp on each speech segment and streams
   back interim and final transcripts.
4. **Correct (optional)** — the final transcript is sent to a local LLM. The
   LLM returns a cleaned transcript, an `is_improved` flag, and content
   `categories`. LLM errors are non-fatal — the raw transcript is saved regardless.
5. **Save** — each utterance is saved as:
   - `~/.listenr/audio_clips/audio/<date>/clip_<uuid>.wav`
   - a line appended to `~/.listenr/audio_clips/manifest.jsonl`

---

## Batch / file transcription

Transcribe an existing audio file without the microphone:

```bash
python -m listenr.unified_asr \
    --audio path/to/audio.wav \
    --whisper-model Whisper-Large-v3-Turbo

# With LLM correction
python -m listenr.unified_asr --llm --audio path/to/audio.wav
```

---

## manifest.jsonl

Every recorded utterance is appended to `~/.listenr/audio_clips/manifest.jsonl`
as a single JSON object per line. It is append-only and easy to inspect:

```bash
# All LLM-improved clips
jq 'select(.is_improved == true)' ~/.listenr/audio_clips/manifest.jsonl

# Clips tagged as commands
jq 'select(.categories[] == "command")' ~/.listenr/audio_clips/manifest.jsonl

# Load into pandas
python -c "
import pandas as pd
df = pd.read_json('~/.listenr/audio_clips/manifest.jsonl', lines=True)
print(df.head())
"
```
