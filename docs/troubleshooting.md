# Troubleshooting

---

## Recording / CLI

### No transcriptions appear / `[SAVE SKIPPED] pcm_buffer is empty`

- Check Lemonade is running: `curl http://localhost:8000/api/v1/health`
- Run with `--debug` to see mic RMS values and WebSocket messages
- If RMS stays near `0.000`, your `input_device` setting is wrong — list
  devices and update config (see [setup.md](setup.md))
- Lower `threshold` in `[VAD]` if your mic is quiet

### LLM correction not working / model answers the prompt instead of fixing it

- Confirm `LLM.enabled = true` and the model name matches one loaded in Lemonade
- Check `curl http://localhost:8000/api/v1/models` to see loaded models
- LLM errors are non-fatal — the raw transcript is saved regardless

### `Could not discover Lemonade websocket port`

Lemonade is not running or not reachable on `localhost:8000`.
Run `lemonade-server serve` and wait for it to finish starting.

### Too many / too few segments

Adjust `[VAD] silence_duration_ms` and `threshold` in your config.
See [configuration.md](configuration.md) for guidance.

---

## Dataset building

### `Valid entries: 0, skipped: N`

All entries failed validation. The most common causes:

1. **Audio files not found** — `manifest.jsonl` stores absolute host paths.
   Inside a container, use `--remap-audio-prefix OLD:NEW` to rewrite them:
   ```bash
   --remap-audio-prefix /home/you/.listenr/audio_clips:/data/listenr/audio_clips
   ```
2. **Clips too short** — lower `--min-duration` (default: `0.3s`)
3. **Transcripts too short** — lower `--min-chars` (default: `2`)

Run with `--dry-run` to see counts before committing any writes.

### Only train/test splits, no dev split

Not enough recordings. With fewer than ~10 clips, the 80/10/10 split rounds to
zero dev entries. Collect more data, or adjust `--split`, e.g. `--split 80/20/0`.

---

## Fine-tuning (AMD / ROCm)

### Container exits with code 139 (segfault)

Usually caused by GPU imbalance on a multi-GPU system. Restrict to one GPU:

```bash
-e HIP_VISIBLE_DEVICES=0
```

### `HSA_STATUS_ERROR_INVALID_ISA` or GPU not detected in container

Your GPU's gfx version may not be natively supported by the ROCm version in the
image. Override it:

```bash
-e HSA_OVERRIDE_GFX_VERSION=10.3.0   # RX 6000 series (RDNA2)
-e HSA_OVERRIDE_GFX_VERSION=11.0.0   # RX 7000 series (RDNA3)
```

Check your GPU model: `rocm-smi --showproductname`

### `cannot set shmsize when running in the host IPC Namespace`

Cannot combine `--shm-size` with `--ipc=host`. Drop `--shm-size` — when
`--ipc=host` is set, the container shares the host's `/dev/shm` directly.

### `command not found: docker` — you have Podman

Fedora and other distributions ship `podman` instead of `docker`. Replace
`docker` with `podman` in all commands. They are CLI-compatible for `run` and
`build`. For compose, use `podman compose` (requires `podman-compose` package).

### Out of GPU memory (OOM)

Reduce memory usage:

```bash
listenr-finetune --batch-size 2 --grad-accum 8 --bf16
```

The effective batch size is `batch_size × grad_accum`. Keeping that product the
same (e.g. `8×2 = 16` → `2×8 = 16`) preserves training dynamics.

### `pip install` ignores `requires-python` warning

The `rocm/pytorch` image ships Python 3.12; listenr requires `>=3.13`. The
codebase uses no 3.13-specific syntax — the `--ignore-requires-python` flag
is safe here. If a future change introduces 3.13-only syntax, update the
`FROM` line in `Dockerfile` to a newer base image.
