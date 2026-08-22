# Fine-tuning on AMD GPU (ROCm)

LoRA fine-tune Whisper or Moonshine on your own recordings using AMD ROCm +
Podman. Everything stays on your machine.

> The fine-tune code works directly on the host if you already have ROCm
> PyTorch installed. The container is just the easiest way to get a
> working GPU environment.

> Real-time microphone capture (`listenr record`) does **not** work inside the
> container. Record on the host first, then fine-tune here.

---

## Quickstart

Assuming you have an AMD GPU with ROCm drivers, Podman, and a manifest of
recordings at `~/.listenr/audio_clips/manifest.jsonl`:

```bash
podman build -t listenr-rocm .                    # 1. build the image (~5 min)
scripts/setup-env.sh                              # 2. write .env from $HOME
podman compose run --rm build-dataset             # 3. build train/dev/test splits
podman compose run --rm finetune                  # 4. fine-tune (bf16, 2000 steps)
podman compose run --rm merge                     # 5. merge adapter → standalone model
listenr eval --compare-base --model ~/listenr_merged   # 6. base vs fine-tuned
```

That's it. The rest of this doc explains what each step does and how to
customize it.

---

## Prerequisites

- AMD GPU with ROCm drivers installed on the host (`rocm-smi` works)
- Podman (`podman --version`)  - Docker works too with the same flags
- ~50 GB free disk space (image + model cache + audio data + checkpoints)
- Recordings collected on the host via `listenr record` ([recording.md](recording.md))

---

## 1. Build the listenr image

```bash
podman build -t listenr-rocm .
```

This layers `listenr[finetune]` on top of
`rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1`  - AMD's
tested-stable PyTorch container.

The ROCm-aware PyTorch wheel is pinned during the build so `pip` cannot
silently replace it with the CPU-only wheel from PyPI.

> Why pin a specific tag instead of `latest`? AMD validates this exact
> tag. `latest` can point to an untested preview build.

---

## 2. Generate `.env`

```bash
scripts/setup-env.sh
```

Writes `.env` with `$HOME`-relative paths and sensible GPU defaults:

```
LISTENR_HOST_DATA=$HOME/.listenr
LISTENR_HOST_DATASET=$HOME/listenr_dataset
LISTENR_HOST_FINETUNE=$HOME/listenr_finetune
LISTENR_HOST_MERGED=$HOME/listenr_merged
HF_CACHE=$HOME/.cache/huggingface
HIP_VISIBLE_DEVICES=0
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
```

Edit `.env` if your paths are different or you need to set
`HSA_OVERRIDE_GFX_VERSION` (see [GPU notes](#gpu-notes) below).

> Why a script instead of compose interpolation? podman-compose 1.5.0
> doesn't expand `$HOME` inside `${VAR:-default}` substitutions, so we
> materialize the paths once into `.env`.

---

## 3. Build the dataset

```bash
podman compose run --rm build-dataset
```

Reads `~/.listenr/audio_clips/manifest.jsonl`, splits into train/dev/test
(80/10/10 by default), and writes a HuggingFace Arrow dataset to
`~/listenr_dataset/hf_dataset/`.

The `--remap-audio-prefix` flag in the compose service rewrites the
absolute host paths stored in `manifest.jsonl` so the audio files resolve
inside the container.

---

## 4. Fine-tune

```bash
podman compose run --rm finetune                       # defaults: bf16, 2000 steps
podman compose run --rm finetune --max-steps 500       # appends to defaults
podman compose run --rm finetune --lora-r 16 --batch-size 4
```

Extra args **append** to the service entrypoint (which sets
`--dataset/--output/--bf16`), so you can override hyperparameters without
losing the path setup.

Adapter checkpoints land in `~/listenr_finetune/`, owned by your host UID
(thanks to `userns_mode: keep-id` in the compose file).

### Common flags

| Flag | Default | Description |
|---|---|---|
| `--base-model ID` | `openai/whisper-small` | HuggingFace model to fine-tune |
| `--max-steps N` | `2000` | Total training steps |
| `--batch-size N` | `8` | Per-device batch size |
| `--lora-r N` | `8` | LoRA rank (higher = more capacity, more VRAM) |
| `--language LANG` | `english` | Target language (Whisper only) |
| `--bf16` | on (in compose) | bf16 mixed precision  - **use this on AMD** |
| `--fp16` | off | fp16  - CUDA only, not recommended for AMD |
| `--dry-run` | off | Load data + model, print stats, exit |

Full list: `podman compose run --rm finetune --help`.

### Choosing a base model

Listenr fine-tunes two encoder-decoder families, and picks the right data
pipeline for whichever one `--base-model` names:

| Family | Example ids | Notes |
|---|---|---|
| Whisper | `openai/whisper-tiny` … `openai/whisper-large-v3-turbo` | Multilingual; `--language` and `--task` apply |
| Moonshine | `UsefulSensors/moonshine-tiny`, `UsefulSensors/moonshine-base` | English-only, much smaller, built for edge and low-latency use; `--language`/`--task` are ignored |

```bash
podman compose run --rm finetune --base-model UsefulSensors/moonshine-base
```

Moonshine is the cheaper experiment: `moonshine-tiny` is 27M parameters
against `whisper-small`'s 244M, so a run finishes far sooner and fits in much
less VRAM. Whisper remains the better choice for multilingual work or when you
need the accuracy of the large checkpoints.

The two differ in what the encoder consumes. Whisper takes a log-Mel
spectrogram padded to a fixed 30 s window; Moonshine takes the raw waveform at
its natural length. That is why the collator pads Moonshine batches and not
Whisper ones. `listenr merge` and `listenr eval` both detect the family from
the checkpoint, so the rest of the flow is unchanged.

CTC and transducer models (Parakeet, Wav2Vec2) are **not** supported: they use
a different loss and label layout, so they need more than a new table entry.

> **Compose gotcha:** `podman compose run SERVICE EXTRA_ARGS` *replaces*
> the `command:` list but leaves `entrypoint:` alone. This repo puts the
> required defaults (`--dataset/--output/--bf16`) in `entrypoint:` so
> extras don't wipe them out. If you fork the compose file, keep that
> split.

---

## 5. Merge the adapter

```bash
podman compose run --rm merge
```

The LoRA adapter stores only weight *deltas*. Merge folds them back into
the base model and writes a standalone `WhisperForConditionalGeneration`
to `~/listenr_merged/`  - loadable with plain `transformers`, no PEFT
required at inference time.

Output (~926 MB for whisper-small):

```
~/listenr_merged/
├── model.safetensors      ← merged weights
├── config.json
├── tokenizer.json / tokenizer_config.json
├── generation_config.json
└── processor_config.json
```

> Merge is pure matrix arithmetic  - no GPU needed, and loading ROCm
> would actually segfault during `PeftModel.merge_and_unload()`. The
> `merge` service forces CPU via `HIP_VISIBLE_DEVICES=-1`.

### Merge options

| Flag | Default | Description |
|---|---|---|
| `--adapter PATH` | `~/listenr_finetune` | LoRA adapter directory |
| `--output PATH` | `~/listenr_merged` | Destination for the merged model |
| `--dry-run` | off | Validate inputs and print plan without writing |

---

## 6. Evaluate

```bash
# Compare original vs fine-tuned on a single file
listenr eval --audio path/to/clip.wav

# Evaluate the held-out test split, base model side-by-side, with WER
# (--model: the container merge flow writes to ~/listenr_merged, see .env)
listenr eval --compare-base --model ~/listenr_merged

# Recall check: did the fine-tune learn your domain words?
listenr eval --compare-base --keyword Claude --keyword Cursor --n 50
```

`listenr eval` runs the merged model over the **test split** of the dataset
written by `listenr build-dataset --format hf` — clips the fine-tune never saw —
and reports corpus WER against the ground-truth transcriptions. With
`--compare-base`, the base model transcribes the same clips so you can see
exactly what fine-tuning changed:

```
  BASE                                      FINE-TUNED (merged)
  So what's good, my guy?                   So what's good, my guy?

  WER vs ground truth (Whisper English normalization)
    base          21.3%
    fine-tuned    14.5%
```

With `--keyword`, you also get a per-model recall summary across all matching clips:
```
  Keyword recall — fine-tuned
    Claude                4/5  (80%)  ████░
    Cursor                3/3  (100%) ███
```

### Eval options

| Flag | Default | Description |
|---|---|---|
| `--model DIR` | `~/listenr_finetune_merged` | Merged model directory |
| `--dataset DIR` | `~/listenr_dataset/hf_dataset` | Dataset from `listenr build-dataset --format hf` |
| `--split NAME` | `test` | Dataset split to evaluate |
| `--n N` | `50` | Maximum number of clips |
| `--compare-base` | off | Also run the base model on every clip |
| `--base-model ID` | (auto from merged config) | Base model for comparison |
| `--audio PATH` |  - | Compare base vs fine-tuned on a single file instead |
| `--keyword WORD` |  - | Filter to clips with WORD in ground truth; repeatable |

### Using the merged model directly

```python
from transformers import pipeline

asr = pipeline("automatic-speech-recognition", model="~/listenr_merged")
print(asr("recording.wav")["text"])
```

---

## GPU notes

### Selecting a GPU

```bash
rocm-smi                              # list GPUs
HIP_VISIBLE_DEVICES=1 podman compose run --rm finetune    # pin to GPU 1
```

On multi-GPU systems with mismatched cards, pin to one  - running on both
can cause imbalance segfaults during training.

### Unsupported gfx version

Some consumer cards need a gfx override. Uncomment the relevant line in
`.env`:

```
HSA_OVERRIDE_GFX_VERSION=10.3.0   # RX 6000 series (RDNA2)
HSA_OVERRIDE_GFX_VERSION=11.0.0   # RX 7000 series (RDNA3)
```

> **Never** set `HSA_OVERRIDE_GFX_VERSION=""`  - an empty string is not
> the same as unset and will crash ROCm at startup.

### Why `--group-add keep-groups` instead of `--group-add video`

The Ubuntu base image's `render`/`video` GIDs (991, 44) don't match
Fedora/RHEL hosts (105, 39). Passing `--group-add render` resolves to
the wrong GID inside the container and `/dev/kfd` access fails.
`keep-groups` passes the host's numeric GIDs through directly, bypassing
the name→GID mismatch.

This is the one place we deviate from AMD's stock guidance, which
assumes Ubuntu hosts.

### Why `--userns=keep-id`

Without it, files written by the container end up owned by a subuid
(`/etc/subuid`) and look root-ish on the host. With it, the container
process runs as your host UID and outputs are owned by you.

This requires MIOpen's cache to live outside `/home/ubuntu` (which the
host UID doesn't own). The Dockerfile bakes
`MIOPEN_USER_DB_PATH=MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen` for this
reason  - without those vars, the first `conv1d` call crashes with
`miopenStatusUnknownError`.

Since `/tmp` is ephemeral in `--rm` containers, JIT kernels recompile
(~30–60 s warm-up) on every run. To persist, add a named volume in
`docker-compose.yml`:

```yaml
volumes:
  - miopen-cache:/tmp/miopen
```

---

## Without compose (manual `podman run`)

The compose file is the recommended path, but here's the equivalent
manual command for reference:

```bash
podman run --rm -it \
    --userns=keep-id \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add keep-groups \
    --ipc=host \
    -e HIP_VISIBLE_DEVICES=0 \
    -v ~/listenr_dataset:/data/dataset \
    -v ~/listenr_finetune:/data/adapter \
    -v ~/.config/listenr:/home/ubuntu/.config/listenr \
    -v ~/.cache/huggingface:/home/ubuntu/.cache/huggingface \
    -w /app \
    listenr-rocm \
    listenr finetune \
        --dataset /data/dataset/hf_dataset \
        --output /data/adapter \
        --bf16
```

The `listenr finetune` line **must** include `--dataset/--output/--bf16`
 - these are not defaults in the CLI, only in the compose service.

---

## On the host (no container)

If you have ROCm PyTorch already installed:

```bash
uv pip install "listenr[finetune]"
listenr build-dataset --format hf
listenr finetune --bf16
listenr merge
listenr eval --compare-base
```
