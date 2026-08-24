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

- AMD GPU with the kernel driver loaded, so `/dev/kfd` and `/dev/dri/renderD*`
  exist. A host ROCm userspace is **not** required: the container ships the
  whole thing, and `rocm-smi` need not be installed or work on the host.
- Read and write access to those device nodes. Check with `ls -l /dev/kfd`.
  If they are `crw-rw----` rather than `crw-rw-rw-`, add yourself to the
  owning group (usually `render`) and log back in:
  `sudo usermod -aG render $USER`. The compose file passes your existing
  groups through with `keep-groups`, which cannot grant a group you are not
  already in.
- Podman (`podman --version`). Docker will **not** work with this compose
  file: `userns_mode: keep-id` and `group_add: keep-groups` are podman-only.
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
| `--output PATH` | `~/listenr_finetune_merged` | Destination for the merged model. The compose flow overrides this to `/data/merged`, which is `$LISTENR_HOST_MERGED` on the host. |
| `--dry-run` | off | Validate inputs and print plan without writing |

---

## 6. Evaluate

```bash
# In the container, which is where the ROCm torch lives
podman compose run --rm eval --compare-base
podman compose run --rm eval --compare-base --keyword Claude --n 50

# Or on the host, if you installed a ROCm torch there
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
# rocm-smi lives in the image, so run it there rather than on the host
podman compose run --rm --entrypoint rocm-smi finetune
HIP_VISIBLE_DEVICES=1 podman compose run --rm finetune    # pin to GPU 1
```

On multi-GPU systems with mismatched cards, pin to one  - running on both
can cause imbalance segfaults during training.

### Unsupported gfx version

Fewer cards need this than they used to. ROCm 7.2 added official support for
RDNA3 (RX 7900 XTX/XT/GRE, 7800 XT, 7700 XT) and RDNA4 (RX 9070 family, 9060
family), so on those cards leave the override unset. Try a run without it
first; only reach for the override if ROCm reports the device as unsupported.

Strix Halo (gfx1151) needs no override on ROCm 7.2. Measured on a Radeon
8060S with the 7.2 image: `torch.cuda.get_device_name(0)` returns
`AMD Radeon Graphics` and the capability is `(11, 5)`, which is gfx1151
reporting itself correctly. Only set an override if your ROCm is older than
the support your card needs.

Where it is still needed, uncomment the relevant line in `.env`:

```
HSA_OVERRIDE_GFX_VERSION=10.3.0   # RX 6000 series (RDNA2, gfx103x)
HSA_OVERRIDE_GFX_VERSION=11.0.0   # cards your ROCm does not list natively
```

> **Never** set `HSA_OVERRIDE_GFX_VERSION=""`  - an empty string is not
> the same as unset and will crash ROCm at startup.

> The value names the ISA you want to be treated as, not the one you have.
> Setting a gfx1151 part to `11.5.1` is a no-op, because 11.5.1 is gfx1151.
> `11.0.0` asks to be treated as gfx1100. Setting it on ROCm 7.2 does no harm
> but is not needed, and telling people to set it implies it is required.

### `--ipc=host` is required

Allocations fail without it, and the error names memory rather than IPC, which
sends you looking in the wrong place:

```
Memory critical error by agent node-0 ... Reason: Memory in use.
```

That appears for any allocation, including a 1024x1024 matmul on an otherwise
idle GPU. The compose services that touch the GPU already set `ipc: host`, so
`podman compose run` is unaffected. It matters when you run `podman run` by
hand. This invocation is verified working on a Radeon 8060S with no sudo, no
render group, no host ROCm and no gfx override:

```bash
podman run --rm --device=/dev/kfd --device=/dev/dri --ipc=host \
  --group-add keep-groups --security-opt seccomp=unconfined listenr-rocm \
  python3 -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Why `--bf16` on AMD

Measured on a Radeon 8060S, ROCm 7.2, torch 2.9.1+rocm7.2.0:

| Precision | Size | Throughput |
|---|---|---|
| fp32 | 1024² | 2.35 TFLOP/s |
| fp32 | 2048² | 2.38 TFLOP/s |
| fp32 | 4096² | 2.45 TFLOP/s |
| bf16 | 4096² | 23.69 TFLOP/s |
| fp16 | 4096² | 22.90 TFLOP/s |

bf16 is close to ten times fp32 on this part, which is why `--bf16` is the
default in the compose entrypoint rather than a suggestion.

### APUs with unified memory (Strix Halo / Ryzen AI MAX)

On an APU there is no separate VRAM pool. The GPU and CPU share one pool, and
the practical limit on model size is the GTT (shared memory) budget, not a
BIOS VRAM carve-out.

Check what you currently have:

```bash
awk '{printf "%.1f GB\n", $1*4096/1024/1024/1024}' /sys/module/ttm/parameters/pages_limit
```

The default is roughly half of system RAM. AMD's guidance is to keep the BIOS
VRAM reservation small (0.5 GB is enough) and raise the shared limit instead,
since a large carve-out permanently reserves memory that GTT would otherwise
hand out on demand:

```bash
pipx install amd-debug-tools
amd-ttm --set 48          # GB of GPU-accessible shared memory
```

Strix Halo needs Linux 6.18.4 or newer for the KFD driver fixes; on older
kernels GPU compute initialization can fail outright. Fedora 43+, Ubuntu
26.04 and Arch carry the fixes already.

Two consequences specific to a shared pool:

`torch.cuda.mem_get_info()` reports the unified aperture, not real VRAM. On a
31 GiB machine it reports roughly 96 GiB. Anything that sizes a batch from that
number will overcommit badly. Size from system RAM and the GTT limit instead.

A local inference server competes with the trainer for the same memory. During
testing, Lemonade was holding Whisper-Large-v3-Turbo on the iGPU at 4.6 GiB.
Unload it before a training run:

```bash
curl -X POST http://localhost:8080/api/v1/unload
```

Full detail: [AMD Strix Halo system optimization](https://rocm.docs.amd.com/en/docs-7.2.0/how-to/system-optimization/strixhalo.html).

> gfx1151 is not on AMD's official ROCm support matrix. The ROCm PyTorch
> wheels target gfx1100 and run on gfx1151 through ISA compatibility, so it
> works but is not a supported configuration, and there are open crash reports
> against it. Prefer smaller base models (`whisper-small`,
> `moonshine-base`) and confirm a `--dry-run` completes before a long run.

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
