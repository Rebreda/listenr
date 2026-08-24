# Troubleshooting

---

## Recording / CLI

### No transcriptions appear / `[SAVE SKIPPED] pcm_buffer is empty`

- Check Lemonade is running: `curl http://localhost:13305/v1/health`
- Run with `--debug` to see mic RMS values and WebSocket messages
- If RMS stays near `0.000`, your `input_device` setting is wrong: list
  devices and update config (see [setup.md](setup.md))
- Lower `threshold` in `[VAD]` if your mic is quiet

### LLM correction not working / model answers the prompt instead of fixing it

- Confirm `LLM.enabled = true` and the model name matches one loaded in Lemonade
- Check `curl http://localhost:13305/api/v1/models` to see loaded models
- LLM errors are non-fatal: the raw transcript is saved regardless

### `Could not discover Lemonade websocket port`

Lemonade is not running or not reachable on `localhost:13305`.
Install it (`sudo apt install lemonade-server`) and verify with `curl http://localhost:13305/v1/health`.

### Too many / too few segments

Adjust `[VAD] silence_duration_ms` and `threshold` in your config.
See [configuration.md](configuration.md) for guidance.

---

## Dataset building

### `Valid entries: 0, skipped: N`

All entries failed validation. The most common causes:

1. **Audio files not found**: `manifest.jsonl` stores absolute host paths.
   Inside a container, use `--remap-audio-prefix OLD:NEW` to rewrite them:
   ```bash
   --remap-audio-prefix /home/you/.listenr/audio_clips:/data/listenr/audio_clips
   ```
2. **Clips too short**: lower `--min-duration` (default: `0.3s`)
3. **Transcripts too short**: lower `--min-chars` (default: `2`)

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

### `torch.cuda.is_available()` returns False in built image but True in base image

Two known causes:

1. **`HSA_OVERRIDE_GFX_VERSION` set to empty string**: an empty string is not
   the same as unset and causes ROCm to fail silently. Never set this variable
   unless you have a specific gfx version to override; omit it entirely otherwise.

2. **pip replaced the ROCm torch wheel with a CPU-only build**: installs of
   `transformers` or other packages can pull a vanilla `torch` from PyPI as a
   dependency, silently replacing the ROCm wheel. The `Dockerfile` guards
   against this with a `pip freeze` constraint file before installing extras.
   Verify with: `python3 -c "import torch; print(torch.version.hip)"`: if this prints `None`, the ROCm wheel was replaced.

3. **Using `rocm/pytorch:latest`**: the `latest` tag can point to a preview
   or less-tested build. Use the specific stable tag:
   `rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1`

Your GPU's gfx version may not be natively supported by the ROCm version in the
image. Override it:

```bash
-e HSA_OVERRIDE_GFX_VERSION=10.3.0   # RX 6000 series (RDNA2)
-e HSA_OVERRIDE_GFX_VERSION=11.0.0   # RX 7000 series (RDNA3)
```

Check your GPU model: `rocm-smi --showproductname`

### `cannot set shmsize when running in the host IPC Namespace`

Cannot combine `--shm-size` with `--ipc=host`. Drop `--shm-size`: when
`--ipc=host` is set, the container shares the host's `/dev/shm` directly.

### `command not found: docker`: you have Podman

Fedora and other distributions ship `podman` instead of `docker`. Replace
`docker` with `podman` in all commands. They are CLI-compatible for `run` and
`build`. For compose, use `podman compose` (requires `podman-compose` package).

### `RuntimeError: volume [...] not defined in top level` with podman-compose

podman-compose cannot parse nested variable defaults like
`${VAR:-${HOME}/path}` in `volumes:`. The compose file avoids this by using
plain `${VAR}` references, which **requires a `.env` file** to be present.

If you haven't created one yet:

```bash
cp .env.example .env
# then edit .env to replace /home/you/ with your actual home directory
sed -i "s|/home/you/|$HOME/|g" .env
```

Verify all `LISTENR_*` variables are set:

```bash
grep LISTENR .env
```

### Out of GPU memory (OOM)

Reduce memory usage:

```bash
listenr finetune --batch-size 2 --grad-accum 8 --bf16
```

The effective batch size is `batch_size × grad_accum`. Keeping that product the
same (e.g. `8×2 = 16` → `2×8 = 16`) preserves training dynamics.

### `podman build` hangs for a long time on `apt-get`

The build sits on the `apt-get install libsndfile1 ffmpeg` layer far longer
than the twenty seconds it should take, with no error. Two causes, and they
look identical from outside.

A host firewall blocking DNS from inside the build. OpenSnitch does this: apt
reports `Temporary failure resolving archive.ubuntu.com` and retries forever,
and because the prompt never surfaces it reads as a broken Dockerfile. Check
your firewall's event log and allow the build.

No usable IPv6 route while the mirror resolves to IPv6 only. Check from a
container:

```bash
podman run --rm <image> sh -c \
  'curl -s -o /dev/null -w "v4 %{http_code}\n" -4 http://archive.ubuntu.com/ubuntu/;
   curl -s -o /dev/null -w "v6 %{http_code}\n" -6 http://archive.ubuntu.com/ubuntu/'
```

If v4 answers and v6 does not, force apt onto IPv4 for the build:

```bash
podman build --network=host -t listenr-rocm .
```

To tell a hang from slow progress, check whether the build is writing anything:

```bash
pgrep -af "apt-get install"
find ~/.local/share/containers/storage/overlay -maxdepth 1 -newermt '-2 minutes' | wc -l
```

A count of zero over several minutes means it is stuck, not slow.

### Every clip is exactly `max_segment_s` long

Symptom, from `listenr record --debug`:

```
max_segment_s (20.0s) reached after 20.1s - forcing commit
```

repeating, with no `speech_stopped` between segments, and every saved clip the
same length. The VAD never ends a segment, so the recording is chopped into
arbitrary fixed-length pieces and sentences are cut mid-phrase.

The cause is `vad.threshold` sitting below your room's noise floor, so the gate
never closes. Compare the two. The debug output prints the RMS of every chunk:

```
[DEBUG] Mic: 9720 chunks sent, RMS=0.0317, ...
```

Watch it while you are silent. That is your noise floor. `vad.threshold` must
sit above it and below your speech. Set it in `config.toml`:

```toml
[vad]
threshold = 0.045
```

If the floor is close to your speech level, raising the threshold alone will
start clipping quiet word endings. Fix the input instead: check the gain on the
device `audio.input_device` selects, move the microphone closer, or stop the
noise source. Fan noise counts, and on a machine that is also training a model
it will not be constant.

`max_segment_s` is a backstop against Whisper's 30 second window, not a
segmenter. If it is firing every time, VAD is not working.

### A clip is dropped as "audio and transcript do not match"

`build-dataset` reports this in the skip breakdown, in two directions.

`transcript belongs to different audio` means far more words than the clip
could contain. `audio holds far more speech than the transcript` means the
reverse, most often a 20 second clip labelled with two words, which is what a
VAD that never fires produces.

Both are dropped rather than trained on. A row whose audio and label disagree
teaches the model either to invent words or to ignore its input.

