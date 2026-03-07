# Configuration

Config is created with defaults at `~/.config/listenr/config.ini` on first run.
Edit it directly — changes take effect on the next `listenr` invocation.

---

## Full reference

```ini
[Lemonade]
api_base = http://localhost:8000/api/v1

[Whisper]
model = Whisper-Tiny

[Audio]
sample_rate = 48000
channels = 1
blocksize = 4096
# Device name (partial match) or index number. Leave blank for system default.
input_device = pipewire

[VAD]
threshold = 0.05
silence_duration_ms = 800
prefix_padding_ms = 250

[LLM]
enabled = true
model = gpt-oss-20b-mxfp4-GGUF
api_base = http://localhost:8000/api/v1
temperature = 0.3
max_tokens = 1500
timeout = 30
context_window = 10

[Storage]
audio_clips_path = ~/.listenr/audio_clips
audio_clips_enabled = true
retention_days = 90
max_storage_gb = 10.0

[Dataset]
output_path = ~/listenr_dataset
split = 80/10/10
min_duration = 0.3
min_chars = 2
seed = 42
format = csv
strip_tags = true

[Finetune]
base_model = openai/whisper-small
language = english
task = transcribe
lora_r = 8
lora_alpha = 32
lora_dropout = 0.1
lora_target_modules = q_proj,v_proj
freeze_encoder = true
learning_rate = 0.0001
warmup_steps = 100
max_steps = 2000
batch_size = 8
grad_accum_steps = 2
fp16 = false
bf16 = false
output_dir = ~/listenr_finetune
eval_steps = 200
save_steps = 400
generation_max_length = 128

[Output]
# Optional: write all transcriptions to a file as well as stdout
file =
llm_file =
line_format = [{timestamp}] {text}
timestamp_format = %Y-%m-%d %H:%M:%S
show_raw = false

[Logging]
level = INFO
file =
```

---

## VAD tuning

Voice Activity Detection controls how speech segments are carved out of the
audio stream. Adjust these two settings in `[VAD]`:

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
curl -s http://localhost:8000/api/v1/models | \
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
