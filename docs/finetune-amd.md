# Fine-tuning on AMD GPU (ROCm)

This guide covers running `listenr-finetune` inside the official AMD ROCm
PyTorch container using Docker/Podman. The fine-tuning code works on the host too if
you already have ROCm PyTorch installed — the container is just the easiest way
to get a working GPU environment.

> **Note:** Real-time microphone capture (`listenr`) does not work inside the
> container. Record audio on the host first, build a dataset, then fine-tune
> here.

---

## Prerequisites

- AMD GPU with ROCm driver installed on the host
- Docker or Podman installed (`podman --version`)
- At least ~50 GB free disk space (model cache + audio data + checkpoints)
- Recordings collected on the host via `listenr` (see [recording.md](recording.md))

---

## 1. Pull the image

Use the specific AMD-tested stable tag rather than `latest`:

```bash
podman pull rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1
```

> The `latest` tag can point to an untested or preview build. The versioned
> tag is what AMD validates and documents.

---

## 2. Build the dataset (on host or in container)

**On the host** (simplest, if you have the finetune deps installed):

```bash
uv run listenr-build-dataset --format hf
# Output: ~/listenr_dataset/hf_dataset
```

**Inside the container** (if you only want the finetune stack there):

```bash
podman run -it \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --ipc=host \
    --device=/dev/kfd \
    --device=/dev/dri/card0 \
    --device=/dev/dri/card1 \
    --device=/dev/dri/renderD128 \
    --device=/dev/dri/renderD129 \
    --group-add keep-groups \
    -v ~/.listenr:/data/listenr \
    -v ~/listenr_dataset:/data/dataset \
    -w /app \
    listenr-rocm \
    listenr-build-dataset \
        --manifest /data/listenr/audio_clips/manifest.jsonl \
        --output /data/dataset \
        --format hf \
        --remap-audio-prefix /home/$(whoami)/.listenr/audio_clips:/data/listenr/audio_clips
```

---

## 3. Build the listenr image

From the repo root:

```bash
podman build -t listenr-rocm .
```

This installs `listenr` with all `[finetune]` extras on top of
`rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1`.
The ROCm-aware PyTorch wheel is pinned during the build so `pip` cannot
replace it with a CPU-only build from PyPI.

---

## 4. Run fine-tuning

```bash
podman run -it \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device=/dev/kfd \
    --device=/dev/dri/card0 \
    --device=/dev/dri/card1 \
    --device=/dev/dri/renderD128 \
    --device=/dev/dri/renderD129 \
    --group-add keep-groups \
    --ipc=host \
    -e HIP_VISIBLE_DEVICES=0 \
    -v ~/listenr_dataset:/data/dataset \
    -v ~/listenr_finetune:/data/adapter \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -w /app \
    listenr-rocm \
    listenr-finetune \
        --dataset /data/dataset/hf_dataset \
        --output /data/adapter \
        --bf16
```

The adapter checkpoint is written to `~/listenr_finetune` on the host via
the bind mount.

---

## 5. Using podman compose

Copy `.env.example` to `.env` and edit for your system, then:

```bash
# Build dataset
podman compose run --rm build-dataset

# Fine-tune
podman compose run --rm finetune

# Pass extra args
podman compose run --rm finetune --max-steps 500 --lora-r 16
```

---

## Common flags for listenr-finetune

| Flag | Default | Description |
|---|---|---|
| `--dataset DIR` | `~/listenr_dataset/hf_dataset` | HF dataset directory |
| `--output DIR` | `~/listenr_finetune` | Adapter checkpoint output |
| `--base-model ID` | `openai/whisper-small` | HuggingFace model to fine-tune |
| `--bf16` | off | bf16 mixed precision — **use this on AMD** |
| `--fp16` | off | fp16 mixed precision — CUDA only, not recommended for AMD |
| `--max-steps N` | `2000` | Total training steps |
| `--batch-size N` | `8` | Per-device batch size |
| `--lora-r N` | `8` | LoRA rank |
| `--language LANG` | `english` | Target language |
| `--dry-run` | off | Load data + model, print stats, then exit |

---

## GPU selection

If you have multiple GPUs, pin to one to avoid imbalance issues:

```bash
-e HIP_VISIBLE_DEVICES=0   # use GPU 0 only
```

Check GPU IDs, names, and memory:

```bash
rocm-smi
```

> **Note on `--group-add keep-groups`:** The `rocm/pytorch` image uses an
> Ubuntu base with different GIDs for the `render` and `video` groups than
> Fedora/RHEL hosts. Passing `--group-add render` resolves to the wrong GID
> inside the container, so the process can't access `/dev/kfd`. `keep-groups`
> passes the host user's actual numeric supplementary GIDs directly,
> bypassing the name→GID mismatch entirely.

---

## If your GPU's gfx version is unsupported by ROCm

Some GPUs (especially consumer RDNA2/RDNA3 cards) need the override:

```bash
-e HSA_OVERRIDE_GFX_VERSION=10.3.0   # RX 6000 series (RDNA2)
-e HSA_OVERRIDE_GFX_VERSION=11.0.0   # RX 7000 series (RDNA3)
```

Or set it in your `.env` file (it's read automatically by `podman compose`).

---

## 6. Merge the adapter into a standalone model

The fine-tuned LoRA adapter stores only weight *deltas*, not the full model.
To produce a standalone `WhisperForConditionalGeneration` that can be loaded
**without PEFT installed**, run `listenr-merge`.

> **Note:** Merging is pure matrix arithmetic — no GPU required. The merge
> service (and host command) runs entirely on CPU.

### In the container (recommended)

```bash
podman run --rm \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v ~/listenr_finetune:/data/adapter \
  -v ~/listenr_merged:/data/merged \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -w /app listenr-rocm \
  listenr-merge --adapter /data/adapter --output /data/merged
```

> No GPU passthrough flags needed — merging runs on CPU.

### With podman compose

```bash
podman compose run --rm merge
```

### On the host

```bash
listenr-merge --adapter ~/listenr_finetune --output ~/listenr_merged
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--adapter PATH` | `~/listenr_finetune` | LoRA adapter directory (output of `listenr-finetune`) |
| `--output PATH` | `~/listenr_finetune_merged` | Destination for the merged model |
| `--dry-run` | off | Validate inputs and print plan without writing files |

### What it writes

`~/listenr_merged/` will contain a fully self-contained Whisper model:

```
model.safetensors        ← merged weights (size depends on base model)
config.json
tokenizer.json
tokenizer_config.json
generation_config.json
processor_config.json
```

### Using the merged model

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline

model = WhisperForConditionalGeneration.from_pretrained("~/listenr_merged")
processor = WhisperProcessor.from_pretrained("~/listenr_merged")

# or, using the pipeline API
asr = pipeline("automatic-speech-recognition", model="~/listenr_merged")
result = asr("recording.wav")
print(result["text"])
```

The merged model requires only `transformers` — no `peft` needed at inference time.

---

## 7. Test inference

`scripts/test_merged.py` loads the merged model and runs it against clips from
your `manifest.jsonl`, showing a side-by-side comparison of the original
Whisper output and the fine-tuned model's output.

```bash
# Run against the 20 most recent clips (default)
python scripts/test_merged.py

# Test clips where a specific word appears in the corrected ground truth.
# Checks whether the fine-tuned model now produces that word.
python scripts/test_merged.py --keyword Claude --keyword Cursor --n 50

# Skip clips shorter than 1 second (reduces noise from clipped recordings)
python scripts/test_merged.py --min-duration 1.0

# Transcribe a single file
python scripts/test_merged.py --audio path/to/clip.wav
```

### Keyword recall summary

When `--keyword` is used, the script prints a recall summary after all clips:

```
  Keyword recall
    Claude                4/5  (80%)  ████░
    Cursor                3/3  (100%) ███
```

Each row shows how many clips where the word appeared in the ground truth
(`corrected_transcription`) were also produced correctly by the fine-tuned
model. A miss means the model still mangled that word for that clip.

### Options

| Flag | Default | Description |
|---|---|---|
| `--model PATH` | `~/listenr_merged` | Merged model directory |
| `--manifest PATH` | `~/.listenr/audio_clips/manifest.jsonl` | Manifest file |
| `--audio PATH` | — | Transcribe a single file instead of manifest clips |
| `--n N` | `20` | Number of clips to test |
| `--min-duration S` | `0.5` | Skip clips shorter than this (seconds) |
| `--keyword WORD` | — | Filter to clips with WORD in ground truth; repeatable |
