# Setup

## Requirements

| Requirement | Notes |
|---|---|
| Python 3.13+ | via `uv`, `pyenv`, or system package manager |
| [Lemonade Server](https://lemonade-server.ai) | runs locally on `localhost:8000` |
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

> For AMD GPU fine-tuning, use the ROCm container instead — see [finetune-amd.md](finetune-amd.md).

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

## Start Lemonade Server

Listenr talks to Lemonade over HTTP/WebSocket — it must be running before you start `listenr`.

```bash
lemonade-server serve
```

On first run, Lemonade will download the configured models. Listenr calls
`POST /api/v1/load` automatically on startup.

Check it is reachable:

```bash
curl http://localhost:8000/api/v1/health
```

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
