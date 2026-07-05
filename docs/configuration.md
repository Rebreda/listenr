# Configuration

All settings live in one typed settings object ([`listenr/settings.py`](../src/listenr/settings.py))
and can be set three ways. Precedence, highest first:

1. **CLI flags** — per-run overrides (see `--help` on each command)
2. **Environment variables** — `LISTENR_<SECTION>__<KEY>`, e.g.
   `LISTENR_FINETUNE__MAX_STEPS=500`. Also read from a `.env` file in the
   working directory. Ideal for containers.
3. **`~/.config/listenr/config.toml`** — persistent user defaults
   (override the file location with `LISTENR_CONFIG=/path/to/config.toml`)

No config file is required — every setting has a sensible default. Invalid
values (wrong type, unknown enum) fail loudly at startup with a clear error.

To start from a fully commented template:

```bash
mkdir -p ~/.config/listenr && cp examples/config.toml ~/.config/listenr/
```

---

## Full reference (`config.toml`)

Every key is optional; the values shown are the defaults.

```toml
[whisper]
# Whisper model served by Lemonade: Whisper-Tiny, Whisper-Base, Whisper-Large-v3-Turbo
model = "Whisper-Base"

[audio]
sample_rate = 48000     # mic capture rate; resampled to 16 kHz internally
channels = 1
blocksize = 4096        # frames per mic read (~85 ms)
# Device name (partial match), index number, or "default" for system default.
input_device = "pipewire"

[vad]
threshold = 0.05             # RMS energy; raise (0.08-0.15) to ignore noise
silence_duration_ms = 800    # silence needed to end a segment
prefix_padding_ms = 250
max_segment_s = 12.0         # hard cap; Whisper hallucinates above ~20 s

[llm]
enabled = true
model = "gpt-oss-20b-mxfp4-GGUF"
api_base = "http://localhost:13305/api/v1"   # Lemonade Server (OpenAI-compatible)
temperature = 0.3
max_tokens = 1500
timeout = 30
context_window = 10     # preceding segments passed as LLM context

[storage]
audio_clips_path = "~/.listenr/audio_clips"

[dataset]
output_path = "~/listenr_dataset"
split = "80/10/10"
min_duration = 0.3
min_chars = 2
seed = 42
format = "csv"          # csv | hf | both
strip_tags = true       # strip noise tags like (music)

[finetune]
base_model = "openai/whisper-small"
language = "english"
task = "transcribe"     # transcribe | translate
lora_r = 8
lora_alpha = 32
lora_dropout = 0.1
lora_target_modules = ["q_proj", "v_proj"]
freeze_encoder = true
learning_rate = 1e-4
warmup_steps = 100
max_steps = 2000
batch_size = 8
grad_accum_steps = 2
fp16 = false            # CUDA mixed precision
bf16 = false            # recommended on AMD ROCm (RDNA2+)
output_dir = "~/listenr_finetune"
eval_steps = 200
save_steps = 400
generation_max_length = 128

# Keyword corrections passed to the LLM: misheard -> correct (case-insensitive).
# Defining this table replaces the built-in examples.
[corrections]
"clod" = "Claude Code"
"open ai" = "OpenAI"
```

---

## Environment variables

Any key above can be set as `LISTENR_<SECTION>__<KEY>` (double underscore
between section and key). Examples:

```bash
# One-off training override, no file edits:
LISTENR_FINETUNE__BASE_MODEL=openai/whisper-tiny listenr-finetune

# Container-friendly (no config mount needed):
podman compose run --rm -e LISTENR_FINETUNE__MAX_STEPS=500 finetune

# List values are comma-separated:
LISTENR_FINETUNE__LORA_TARGET_MODULES=q_proj,k_proj,v_proj
```

---

## VAD tuning

Voice Activity Detection controls how speech segments are carved out of the
audio stream. Adjust these two settings in `[vad]`:

| Goal | Setting |
|---|---|
| Shorter segments / snappier cuts | Lower `silence_duration_ms` (e.g. `500`) |
| Avoid cutting off speech | Raise `silence_duration_ms` (e.g. `1200`) |
| Ignore background noise | Raise `threshold` (e.g. `0.05`) |
| Capture quiet speech | Lower `threshold` (e.g. `0.005`) |

---

## Available Lemonade models

List all models currently loaded on your Lemonade instance:

```bash
curl -s http://localhost:13305/api/v1/models | \
  python3 -c "import sys,json; [print(m['id']) for m in json.load(sys.stdin)['data']]"
```

Common options:

| Model | Type | Notes |
|---|---|---|
| `Whisper-Base` | ASR | Fast, lower accuracy |
| `Whisper-Large-v3-Turbo` | ASR | Best accuracy |
| `gpt-oss-20b-mxfp4-GGUF` | LLM | Good correction quality |
| `Gemma-3-4b-it-GGUF` | LLM | Lighter alternative |
| `DeepSeek-Qwen3-8B-GGUF` | LLM | Lighter alternative |
