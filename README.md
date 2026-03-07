
# Listenr: Local Dataset Collection for ASR Training

Listenr is a privacy-first tool for collecting real-world audio and high-quality transcriptions, designed to help build better automatic speech recognition (ASR) models. All processing runs locally on your hardware via [Lemonade Server](https://lemonade-server.ai) — no audio or text leaves your machine.

![Listenr CLI streaming — example output](screenshot.png)

## Why Listenr?

- **Local-only, private by design.** No cloud APIs. All inference runs on your CPU, GPU, or NPU via Lemonade Server.
- **Open models.** Uses Whisper.cpp for transcription and any GGUF-compatible LLM for post-processing correction.
- **Automatic correction pipeline.** A local LLM cleans up punctuation, grammar, and homophones — producing a higher-quality training corpus than raw Whisper output alone.
- **Real-world data.** Collects natural, conversational speech in realistic environments.
- **Dataset-ready output.** Every utterance is saved with its audio clip and appended to a single `manifest.jsonl`. One command builds train/dev/test splits.

## How It Works

1. **Capture** — `listenr` streams your microphone to Lemonade's `/realtime` WebSocket in ~85 ms chunks, resampled to 16 kHz.
2. **VAD** — Lemonade's built-in voice activity detection segments speech boundaries automatically.
3. **Transcribe** — Lemonade runs Whisper.cpp on each segment and streams back transcripts.
4. **Correct (optional)** — a local LLM cleans the transcript and tags content categories.
5. **Save** — each utterance is saved as a `.wav` clip and a line in `manifest.jsonl`.
6. **Build dataset** — `listenr-build-dataset` writes train/dev/test splits from the manifest.
7. **Fine-tune** — `listenr-finetune` trains a LoRA adapter on top of a Whisper base model using your collected data.
8. **Merge** — `listenr-merge` folds the adapter into the base model, producing a self-contained model that needs only `transformers`.
9. **Test** — `scripts/test_merged.py` runs the merged model against your clips and compares output to the original Whisper transcriptions.

## Quick Start

```bash
git clone https://github.com/Rebreda/listenr
cd listenr
uv pip install -e .
lemonade-server serve   # in another terminal
uv run listenr
```

See [docs/setup.md](docs/setup.md) for full installation instructions.

## Documentation

| Guide | Description |
|---|---|
| [docs/setup.md](docs/setup.md) | Installation, Lemonade Server, microphone setup |
| [docs/configuration.md](docs/configuration.md) | Full `config.ini` reference, VAD tuning, available models |
| [docs/recording.md](docs/recording.md) | CLI usage, how recording works, batch transcription |
| [docs/dataset.md](docs/dataset.md) | Building train/dev/test splits, CSV and HF formats |
| [docs/finetune-amd.md](docs/finetune-amd.md) | Fine-tuning Whisper on AMD GPU via ROCm + Podman, merging, and inference testing |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Common errors and fixes |

## License

Mozilla Public License Version 2.0 — see `LICENSE`.

## Acknowledgments

- [Lemonade Server](https://lemonade-server.ai) — unified local inference API
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) — fast local ASR
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — fast local LLMs
