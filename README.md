
<div align="center">

<img src="assets/images/logo.png" alt="Listenr Logo" width="64">

# Listenr

**Build better speech-to-text and ASR models entirely on your machine.**

Record your voice. Clean it up with local AI. Fine-tune a Whisper model. Deploy something that's actually yours.

<a href="https://quickthoughts.ca/posts/listenr-asr-training-data-problem/">Walkthrough</a> &nbsp;|&nbsp;
<a href="docs/setup.md">Setup</a> &nbsp;|&nbsp;
<a href="docs/configuration.md">Configuration</a> &nbsp;|&nbsp;
<a href="docs/recording.md">Recording</a> &nbsp;|&nbsp;
<a href="docs/troubleshooting.md">Troubleshooting</a>

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

![Listenr CLI streaming - example output](assets/images/screenshot.png)

Everything stays local  - no audio, text, or weights ever leave your machine.

## Documentation

| Guide | Description |
|---|---|
| [docs/setup.md](docs/setup.md) | Installation, Lemonade Server, microphone setup |
| [docs/recording.md](docs/recording.md) | CLI usage, how recording works, batch transcription |
| [docs/configuration.md](docs/configuration.md) | Full `config.ini` reference, VAD tuning, available models |
| [docs/dataset.md](docs/dataset.md) | Building train/dev/test splits, CSV and HF formats |
| [docs/finetune-amd.md](docs/finetune-amd.md) | Fine-tuning Whisper on AMD GPU via ROCm + Podman |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Common errors and fixes |

## Acknowledgments

- [Lemonade Server](https://lemonade-server.ai)  - unified local inference API
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp)  - fast local ASR
- [llama.cpp](https://github.com/ggerganov/llama.cpp)  - fast local LLMs

## License

Mozilla Public License Version 2.0  - see `LICENSE`.
