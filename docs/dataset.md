# Building a Dataset

`listenr-build-dataset` reads `manifest.jsonl` and writes train/dev/test splits
in CSV and/or HuggingFace datasets format, ready to pass to `listenr-finetune`.

---

## Usage

```bash
# Default: 80/10/10 CSV splits → ~/listenr_dataset/
uv run listenr-build-dataset

# Custom output directory and split ratio
uv run listenr-build-dataset --output ~/my_dataset --split 90/5/5

# Exclude very short or sparse clips
uv run listenr-build-dataset --min-duration 1.0 --min-chars 10

# HuggingFace datasets format (required for listenr-finetune)
uv run listenr-build-dataset --format hf

# Both CSV and HF at once
uv run listenr-build-dataset --format both

# Preview stats without writing anything
uv run listenr-build-dataset --dry-run
```

---

## All options

| Flag | Default | Description |
|---|---|---|
| `--manifest PATH` | `~/.listenr/audio_clips/manifest.jsonl` | Input manifest file |
| `--output DIR` | `~/listenr_dataset` | Output directory |
| `--split TRAIN/DEV/TEST` | `80/10/10` | Split percentages |
| `--min-duration SECS` | `0.3` | Minimum clip duration |
| `--min-chars N` | `2` | Minimum transcript length (non-whitespace chars) |
| `--format csv\|hf\|both` | `csv` | Output format |
| `--seed N` | `42` | Random seed for reproducible splits |
| `--no-strip-tags` | off | Keep noise tags like `(music)` in transcriptions |
| `--remap-audio-prefix OLD:NEW` | — | Rewrite audio path prefix (useful in containers) |
| `--dry-run` | off | Print stats and exit without writing files |

---

## Output: CSV

Three files in `--output`:

```
train.csv
dev.csv
test.csv
```

Columns: `uuid`, `split`, `audio_path`, `raw_transcription`,
`corrected_transcription`, `is_improved`, `categories`, `duration_s`,
`sample_rate`, `whisper_model`, `llm_model`, `timestamp`.

---

## Output: HuggingFace datasets

Creates an `hf_dataset/` directory loadable with:

```python
from datasets import load_from_disk
ds = load_from_disk("~/listenr_dataset/hf_dataset")
print(ds)
```

The `Audio` feature is loaded lazily — audio files are read from disk only
when the batch is accessed. Pass this directory directly to `listenr-finetune`.

---

## Path remapping (container use)

`manifest.jsonl` stores absolute host paths. When running inside a container
where your data is mounted at a different location, use `--remap-audio-prefix`
to fix them at read time:

```bash
listenr-build-dataset \
    --manifest /data/listenr/audio_clips/manifest.jsonl \
    --output /data/dataset \
    --format hf \
    --remap-audio-prefix /home/you/.listenr/audio_clips:/data/listenr/audio_clips
```

The original `manifest.jsonl` is never modified.
