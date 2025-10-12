# listnr — Local, privacy-first real-time ASR (Whisper + VAD)

Listenr is a lightweight, privacy-first command-line Automatic Speech Recognition (ASR) service for Linux. It records audio from your microphone, detects speech with Silero VAD, and transcribes speech locally using Whisper-compatible backends (now supporting AMD via whisper-live). The project focuses on local processing, configurable outputs (plain text and JSON), optional LLM post-processing, and optional audio clip storage for dataset building.

Key features
- Local transcription (offline) using Whisper-compatible backends (whisper-live for AMD, other backends supported).
- Real-time VAD-based streaming with Silero to auto-detect speech and pauses.
- CLI-first: transcripts print to terminal and optionally append to text/JSON files.
- Optional LLM post-processing (Ollama) for punctuation, capitalization and correction.
- Optional audio clip saving and a Dataset Manager pipeline (exports in CommonVoice / HuggingFace / CSV formats).

Quick links
- Config file: `~/.config/listnr/config.ini`
- PRD & design notes: `./prd/prd.md` and `./prd/dataset-manager-prd.md`

Requirements (summary)
- Python 3.9+ and a virtualenv
- ffmpeg, libsndfile (for reading/writing audio)
- sounddevice, soundfile, numpy, torch (see `requirements.txt`)
- whisper-live server (optional, for AMD GPU support — see notes below)

Setup (short)
1. Clone the repository and enter it:

```sh
git clone https://github.com/<owner>/listenr
cd listenr
```

2. Create & activate a virtual environment, then install Python dependencies:

```sh
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

3. (Optional) If you want AMD support via whisper-live, install and run a whisper-live server and ensure it is reachable at the host/port configured in `config.ini`. The code also supports other Whisper backends if you prefer to use them directly.

Running

Start the service from the repo root:

```sh
python asr.py
```

Options:
- `--no-llm` — disable LLM post-processing
- `-m, --model` — override model name from the config
- `-d, --device` — override device (cpu|cuda)
- `-o, --output` — override output file
- `--list-devices` — list audio input devices
- `--edit-config` — open config file in $EDITOR

Outputs
- Plain text log file (configurable in `[Output]`)
- JSON session file via `output_handler.py` (planned/implemented as per PRD)
- Optional audio clip storage (`~/.listnr/audio_clips`) for dataset creation

Dataset manager & export
The project includes design and plans for a Dataset Manager component (see `prd/dataset-manager-prd.md`). When enabled in the config, `asr.py` will append manifest entries (JSONL) describing each saved clip and transcription. The Dataset Manager can then process, score, and export curated datasets in CommonVoice, HuggingFace, or CSV formats.

Whisper-live (AMD) notes
- The repository now includes support for using a local whisper-live transcription server (TranscriptionClient). If you plan to use an AMD GPU or prefer running a model server process, run the whisper-live server and ensure your `config.ini` points to the correct host/port. The PRD and code include an example TranscriptionClient usage.

Configuration
- All runtime settings live in `~/.config/listnr/config.ini` (created automatically on first run). Important sections:
  - `[Audio]` — sample rate, channels, input device, leading/trailing silence
  - `[VAD]` — thresholds and chunk sizes
  - `[Whisper]` — model name, device
  - `[Storage]` — audio clip storage settings (path, retention, format)
  - `[Output]` — text/json file paths and output format
  - `[LLM]` — enable/disable LLM corrections and provider settings

Development & extension points
- `output_handler.py` — JSON session and audio↔text mapping (see PRD)
- `cleanup_service.py` — retention / cleanup logic for stored clips (see PRD)
- `dataset-manager` — separate component to process manifests and export datasets

Troubleshooting
- If audio doesn't record, check input device with `python -m sounddevice` or use `--list-devices`.
- If whisper-live is used, ensure the server is running and reachable.
- For GPU issues, ensure your drivers and PyTorch/CUDA versions match your hardware.

Contributing
- See `prd/` for the design and implementation roadmap. Contributions that implement the Dataset Manager, output handler, or cleanup service are welcome.

License
- Mozilla Public License Version 2.0 — see `LICENSE`.
