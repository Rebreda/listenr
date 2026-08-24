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
uv tool install listenr   # or: pipx install listenr
```

Python 3.11 or newer. To work on Listenr itself, clone the repo and install it
editable instead:

```bash
git clone https://github.com/Rebreda/listenr
cd listenr
uv pip install -e ".[dev]"
```

### Optional extras

The core install covers recording, transcription, and dataset building. Each
of the heavier commands lives behind an extra:

| Extra | Enables | Pulls in |
|---|---|---|
| `finetune` | `listenr finetune`, `merge`, `eval` | transformers, peft, accelerate |
| `mdc` | `listenr import-mdc` | datacollective, pandas |
| `hf` | `listenr import-hf` | datasets, pandas |
| `categorize` | `listenr categorize` | sentence-transformers |

```bash
uv pip install "listenr[finetune]"
```

Running a command without its extra prints the exact install line you need.

### AMD GPUs: install a ROCm torch first

`listenr` never pins torch, so the `finetune` extra resolves it transitively
from default PyPI, and default PyPI serves CUDA builds. On an AMD machine that
wheel installs cleanly, imports without complaint, and reports no devices, so
training silently falls back to CPU. Install a ROCm build **before** the extra:

```bash
uv pip install --torch-backend=rocm torch
uv pip install "listenr[finetune]"
```

or with pip:

```bash
pip install --index-url https://download.pytorch.org/whl/rocm6.4 torch
pip install "listenr[finetune]"
```

The ROCm wheels bundle their own runtime, so a host ROCm installation is not
required. Only the kernel driver is.

Check that it worked before spending a training run on it:

```bash
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())"
```

A ROCm build prints a HIP version and `True`. A CUDA build on an AMD box
prints something like `2.12.0+cu130 None False`. `listenr finetune` now refuses
to start in that state rather than quietly training on CPU.

For the container route instead, which avoids all of this, see
[finetune-amd.md](finetune-amd.md).

> For AMD GPU fine-tuning, use the ROCm container instead  - see [finetune-amd.md](finetune-amd.md).

---

## Run without activating the venv

```bash
listenr record
listenr build-dataset
listenr finetune
```

Run `listenr --help` to list all commands; each command has its own `--help`.

## Or activate once per session

```bash
source .venv/bin/activate
listenr record
```

---

## Install and start Lemonade Server

Listenr talks to Lemonade over HTTP/WebSocket on `localhost:13305`. It must be running before you start `listenr record`.

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

> First-time pulls download weights from Hugging Face. Allow a few minutes depending on connection speed. You can swap models in `~/.config/listenr/config.toml`  - see [configuration.md](configuration.md) for options.

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

Set `input_device` in `~/.config/listenr/config.toml` to the device name (partial
match works) or its index number. See [configuration.md](configuration.md).
