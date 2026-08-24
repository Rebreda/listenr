# Building a Dataset

`listenr build-dataset` reads `manifest.jsonl` and writes train/dev/test splits
in CSV and/or HuggingFace datasets format, ready to pass to `listenr finetune`.

---

## Usage

```bash
# Default: 80/10/10 CSV splits → ~/listenr_dataset/
listenr build-dataset

# Custom output directory and split ratio
listenr build-dataset --output ~/my_dataset --split 90/5/5

# Exclude very short or sparse clips
listenr build-dataset --min-duration 1.0 --min-chars 10

# HuggingFace datasets format (required for listenr finetune)
listenr build-dataset --format hf

# Both CSV and HF at once
listenr build-dataset --format both

# Preview stats without writing anything
listenr build-dataset --dry-run

# Combine your local manifest with an imported MDC manifest
listenr build-dataset \
    --manifest ~/.listenr/audio_clips/manifest.jsonl \
    --manifest ~/.listenr/audio_clips/imports/mdc/<dataset-id>/manifest.jsonl \
    --format hf
```

---

## All options

| Flag | Default | Description |
|---|---|---|
| `--manifest PATH` | `~/.listenr/audio_clips/manifest.jsonl` | Input manifest file. Pass multiple times to combine manifests. |
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
when the batch is accessed. Pass this directory directly to `listenr finetune`.

---

## Path remapping (container use)

`manifest.jsonl` stores absolute host paths. When running inside a container
where your data is mounted at a different location, use `--remap-audio-prefix`
to fix them at read time:

```bash
listenr build-dataset \
    --manifest /data/listenr/audio_clips/manifest.jsonl \
    --output /data/dataset \
    --format hf \
    --remap-audio-prefix /home/you/.listenr/audio_clips:/data/listenr/audio_clips
```

The original `manifest.jsonl` is never modified.

---

## Importing external datasets

You can mix third-party ASR datasets into fine-tuning without changing Listenr's
existing flow. Each importer is optional (behind its own extra and lazily
imported) and non-destructive: it writes a *separate* manifest under
`~/.listenr/audio_clips/imports/<source>/<dataset>/manifest.jsonl` and never
touches your primary `manifest.jsonl`. You then pass one or more manifests to
`listenr build-dataset`.

Every importer normalises its source onto the same manifest schema via a shared
mapping (first matching column wins). Both importers accept per-dataset column
overrides for unusual layouts:

| Flag | Description |
|---|---|
| `--audio-column NAME` | Source column holding the audio path/clip |
| `--text-column NAME` | Source column holding the transcription |
| `--split-column NAME` | Source column holding the split name |

### Mozilla Data Collective

The `datacollective` SDK extracts real audio files and returns a DataFrame whose
ASR columns are normally `audio_path` / `transcription`.

The key is read from the environment; a gitignored `.env` in the repo root is
loaded automatically if present.

```bash
uv pip install "listenr[mdc]"
export MDC_API_KEY=your-api-key-here   # or put MDC_API_KEY=... in .env

listenr import-mdc <dataset-id>
# -> ~/.listenr/audio_clips/imports/mdc/<dataset-id>/manifest.jsonl
```

### Hugging Face

HF stores audio as an in-memory feature, so the importer materialises each clip
to a WAV under the import's `audio/` directory. Defaults target Common
Voice-style datasets (`audio` + `sentence`).

```bash
uv pip install "listenr[hf]"

listenr import-hf mozilla-foundation/common_voice_17_0 --config en --split train
# -> ~/.listenr/audio_clips/imports/hf/mozilla-foundation__common_voice_17_0/manifest.jsonl
```

### Building from imported manifests

Use an imported manifest alone, or combine it with your own recordings:

```bash
listenr build-dataset \
    --manifest ~/.listenr/audio_clips/manifest.jsonl \
    --manifest ~/.listenr/audio_clips/imports/mdc/<dataset-id>/manifest.jsonl \
    --format hf
```

`listenr finetune` stays unchanged and still consumes the generated `hf_dataset/`.

---

## Filtering a manifest by topic

To fine-tune on a single domain (e.g. technology/AI clips out of a mixed
corpus), filter any manifest by topic before building. `listenr categorize`
embeds each transcription and keeps only clips whose best cosine similarity to
one of your topic phrases meets `--threshold`. The output is a normal manifest,
so it feeds straight into `listenr build-dataset`. The input manifest is never
modified.

```bash
uv pip install "listenr[categorize]"

# Tune the threshold first with --dry-run (prints match count + top hits):
listenr categorize <input>/manifest.jsonl \
    --topic "technology" --topic "artificial intelligence" --topic "software" \
    --threshold 0.35 --dry-run

# Then write the filtered manifest:
listenr categorize <input>/manifest.jsonl \
    --topics-file topics.txt \
    --threshold 0.35 \
    --output ~/.listenr/audio_clips/imports/tech_only.jsonl

listenr build-dataset --manifest ~/.listenr/audio_clips/imports/tech_only.jsonl --format hf
```

Topics can be given inline (repeat `--topic`) or one-per-line via `--topics-file`.
Each kept record is annotated with `topic_score` and the matched `category`.
Use `--keep-all` to annotate every clip instead of dropping non-matches.

> Similarity thresholds are corpus-dependent — start with `--dry-run` and adjust.
> Note that filtering only finds what's in the source: a lifestyle-speech
> dataset has few technology clips no matter the threshold, so pick a source
> that actually contains your target domain.

## Splits from an imported corpus

`build-dataset` keeps the train/dev/test split an imported corpus assigned,
rather than reshuffling it. This is the default whenever any record carries a
`source_split`, which the importers write.

It matters more than it looks. Public corpora keep speakers disjoint across
their splits on purpose. Reshuffling puts the same voice in train and test, so
the model is scored on speakers it trained on and any WER improvement is
partly an artefact. It also makes your number incomparable to every published
baseline for that corpus, because you are no longer evaluating on its test set.

Records with no `source_split`, including everything from `listenr record`, go
to train. They can never reach dev or test, so they cannot affect an
evaluation. The command reports how many were placed that way.

To reshuffle everything anyway:

```bash
listenr build-dataset --no-preserve-splits --split 80/10/10
```

`--split` only sets the ratios for a reshuffle. It has no effect when splits
are preserved.

## Noise tag stripping changes the labels

`strip_tags` is on by default and removes any bracketed or parenthesised span
of 1 to 60 characters, so `(music)`, `[Applause]` and inline `[disfluency]`
markers all go. That is usually what you want for training, but it does mean
the label text no longer matches the corpus exactly, and a clip that is mostly
markers can fall under `--min-chars` and be dropped entirely.

`build-dataset` now reports why records were skipped, broken down by reason,
so a large drop count is traceable:

```
Valid entries: 2195, skipped: 230
  skipped 168: transcript under --min-chars (10) after tag stripping
  skipped  62: shorter than --min-duration (1.0s)
```

Pass `--no-strip-tags` to keep the markers.
