
<div align="center">

<img src="https://raw.githubusercontent.com/Rebreda/listenr/main/assets/images/logo.png" alt="Listenr Logo" width="64">

# Listenr

**Build better speech-to-text and ASR models entirely on your machine.**

Record your voice. Clean it up with local AI. Fine-tune Whisper or Moonshine. Deploy something that's actually yours.

[![PyPI](https://img.shields.io/pypi/v/listenr?logo=pypi&logoColor=white)](https://pypi.org/project/listenr/)
[![Python](https://img.shields.io/pypi/pyversions/listenr?logo=python&logoColor=white)](https://pypi.org/project/listenr/)
[![License](https://img.shields.io/pypi/l/listenr)](https://github.com/Rebreda/listenr/blob/main/LICENSE)
[![Tests](https://github.com/Rebreda/listenr/actions/workflows/tests.yml/badge.svg)](https://github.com/Rebreda/listenr/actions/workflows/tests.yml)

<a href="https://quickthoughts.ca/posts/listenr-asr-training-data-problem/">Walkthrough</a> &nbsp;|&nbsp;
<a href="https://github.com/Rebreda/listenr/blob/main/docs/setup.md">Setup</a> &nbsp;|&nbsp;
<a href="https://github.com/Rebreda/listenr/blob/main/docs/configuration.md">Configuration</a> &nbsp;|&nbsp;
<a href="https://github.com/Rebreda/listenr/blob/main/docs/recording.md">Recording</a> &nbsp;|&nbsp;
<a href="https://github.com/Rebreda/listenr/blob/main/docs/dataset.md">Dataset</a> &nbsp;|&nbsp;
<a href="https://github.com/Rebreda/listenr/blob/main/docs/troubleshooting.md">Troubleshooting</a>

<a href="https://lemonade-server.ai" target="_blank" rel="noopener">
  <img
    src="https://raw.githubusercontent.com/lemonade-sdk/assets/main/challenge/lemonade-developer-challenge-winner-badge@2x.png"
    alt="Lemonade Developer Challenge Winner"
    width="200"
    height="44"
  >
</a>

</div>

---

![Listenr CLI streaming - example output](https://raw.githubusercontent.com/Rebreda/listenr/main/assets/images/screenshot.png)

## How it works

1. **Create good data** - Use Listenr to record and collect natural speech with domain-specific vocabulary that generic models miss.
2. **Process & improve** - Pipe it through [Lemonade](https://lemonade-server.ai) or any OpenAI-compatible provider to transcribe with Whisper and automatically correct grammar, punctuation, and homophones using a local LLM.
3. **Fine-tune & deploy** - Use Listenr to build train/dev/test splits and fine-tune Whisper or Moonshine with LoRA. Merge the adapter into a self-contained model you can deploy.

Everything stays local - no audio, text, or weights ever leave on your machine.

## Get started

**Install Lemonade and pull models:**

Lemonade guide: [lemonade-server.ai/docs/guide/install](https://lemonade-server.ai/docs/guide/install/)

```bash
# after installing locally, download default models
lemonade pull Whisper-Base
lemonade pull gpt-oss-20b-mxfp4-GGUF
```


**Install Listenr and start recording:**
```bash
uv tool install listenr   # or: pipx install listenr
listenr record            # start recording
```

On [PyPI](https://pypi.org/project/listenr/). Python 3.11 or newer. The core
install covers recording, transcription and dataset building; fine-tuning and
the dataset importers live behind extras, listed in
[docs/setup.md](https://github.com/Rebreda/listenr/blob/main/docs/setup.md).

Working on Listenr itself? Clone the repo and `uv pip install -e ".[dev]"` instead.

**Once you have recordings, process & fine-tune:**
```bash
# Build train/dev/test splits from your manifest
listenr build-dataset --format hf

# Fine-tune Whisper or Moonshine (see docs/finetune-amd.md for AMD GPUs)
podman compose run --rm finetune

# Merge the LoRA adapter into a standalone model
podman compose run --rm merge

# Evaluate it on the held-out test split
listenr eval --compare-base --keyword YourDomainWord
```

See [docs/setup.md](https://github.com/Rebreda/listenr/blob/main/docs/setup.md) for full installation details.

If you want to mix in an external ASR dataset, use the optional importers to
write a separate Listenr-compatible manifest — `listenr import-mdc <dataset-id>`
(Mozilla Data Collective) or `listenr import-hf <dataset-id>` (Hugging Face) —
then pass that manifest to `listenr build-dataset` alongside your normal one.
See [docs/dataset.md](https://github.com/Rebreda/listenr/blob/main/docs/dataset.md) for details.

## Under the hood

**Recording & transcription** - Listenr streams your microphone to Lemonade's `/realtime` WebSocket in ~85 ms chunks (16 kHz). Lemonade's voice activity detection segments speech, runs Whisper.cpp, and streams back transcripts.

**Auto-correction** - A local LLM cleans up punctuation, grammar, and homophones, producing a higher-quality training corpus than raw Whisper output alone.

**Dataset & fine-tuning** - Listenr saves each utterance as a `.wav` clip and a line in `manifest.jsonl`. One command builds train/dev/test splits in HuggingFace format. Another command fine-tunes any `openai/whisper-*` or `UsefulSensors/moonshine-*` model using LoRA (works on AMD and NVIDIA GPUs via Podman). Moonshine is the smaller, English-only, edge-oriented option; Whisper is the multilingual all-rounder.

**Deployment** - `listenr merge` folds the LoRA adapter into a self-contained model that loads with plain `transformers`. No PEFT dependency. Run inference locally or deploy it anywhere.

## Documentation

| Guide | Description |
|---|---|
| [docs/setup.md](https://github.com/Rebreda/listenr/blob/main/docs/setup.md) | Installation, Lemonade Server, microphone setup |
| [docs/configuration.md](https://github.com/Rebreda/listenr/blob/main/docs/configuration.md) | Full `config.toml` reference, VAD tuning, available models |
| [docs/recording.md](https://github.com/Rebreda/listenr/blob/main/docs/recording.md) | CLI usage, how recording works, batch transcription |
| [docs/dataset.md](https://github.com/Rebreda/listenr/blob/main/docs/dataset.md) | Building train/dev/test splits, CSV and HF formats, and the optional Mozilla Data Collective import |
| [docs/finetune-amd.md](https://github.com/Rebreda/listenr/blob/main/docs/finetune-amd.md) | Fine-tuning Whisper on AMD GPU via ROCm + Podman, merging, and inference testing |
| [docs/troubleshooting.md](https://github.com/Rebreda/listenr/blob/main/docs/troubleshooting.md) | Common errors and fixes |

## Acknowledgments

- [Lemonade Server](https://lemonade-server.ai) - unified local inference API
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) - fast local ASR
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - fast local LLMs

## License

Mozilla Public License Version 2.0 - see `LICENSE`.
