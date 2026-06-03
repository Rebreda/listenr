# Setup

## Requirements

| Requirement | Notes |
|---|---|
| Python 3.13+ | via `uv`, `pyenv`, or system package manager |
| [Lemonade Server](https://lemonade-server.ai) | runs locally on `localhost:13305` |
| Microphone | accessible via PipeWire or ALSA (Linux) |
| `uv` | recommended Python package manager |

---

## Install

```bash
git clone https://github.com/Rebreda/listenr
cd listenr
uv pip install -e .
```

### Optional: fine-tuning dependencies

Only needed if you plan to run `listenr-finetune`. Requires PyTorch.

```bash
uv pip install -e ".[finetune]"
```

> For AMD GPU fine-tuning, use the ROCm container instead  - see [finetune-amd.md](finetune-amd.md).

---

## Run without activating the venv

```bash
uv run listenr
uv run listenr-build-dataset
uv run listenr-finetune
```

## Or activate once per session

```bash
source .venv/bin/activate
listenr
```

---

## Install and start Lemonade Server

Listenr talks to Lemonade over HTTP/WebSocket on `localhost:13305`. It must be running before you start `listenr`.

**Ubuntu (recommended):**
```bash
sudo add-apt-repository ppa:lemonade-team/stable
sudo apt install lemonade-server
```

**Snap:**
```bash
sudo snap install lemonade-server
```

> For other platforms (Windows, macOS, Fedora, Arch, Docker) see the [Lemonade install guide](https://lemonade-server.ai/docs/guide/install/).

The package installs a system service that starts automatically. Pull the models Listenr needs before recording:

```bash
lemonade pull Whisper-Base
lemonade pull gpt-oss-20b-mxfp4-GGUF
```

> First-time pulls download weights from Hugging Face. Allow a few minutes depending on connection speed. You can swap models in `~/.config/listenr/config.ini`  - see [configuration.md](configuration.md) for options.

Verify the server is reachable:

```bash
curl http://localhost:13305/v1/health
```

Listenr calls `POST /v1/load` automatically on startup to ensure models are in memory before recording begins.

---

## Finding your microphone device

```bash
python -c "
import sounddevice as sd
for i, d in enumerate(sd.query_devices()):
    if d['max_input_channels'] > 0:
        print(f\"{i}: {d['name']}\")
"
```

Set `input_device` in `~/.config/listenr/config.ini` to the device name (partial
match works) or its index number. See [configuration.md](configuration.md).
