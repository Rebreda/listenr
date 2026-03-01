#!/usr/bin/env python3
"""
build_dataset.py — Build train/dev/test splits from Listenr recordings.

Reads manifest.jsonl saved by the CLI, filters/validates entries, and writes
CSV (and optionally HuggingFace datasets) split files.

Usage:
    listenr-build-dataset [options]

Examples:
    # Default: 80/10/10 split, CSV output in ~/listenr_dataset/
    listenr-build-dataset

    # Custom output directory and split ratio
    listenr-build-dataset --output ~/my_dataset --split 90/5/5

    # Only include clips longer than 1 second, HuggingFace format
    listenr-build-dataset --min-duration 1.0 --format hf

    # Preview without writing files
    listenr-build-dataset --dry-run
"""

import argparse
import csv
import json
import logging
import random
import sys
from pathlib import Path

from listenr.constants import (
    DATASET_FORMAT,
    DATASET_MIN_CHARS,
    DATASET_MIN_DURATION,
    DATASET_OUTPUT,
    DATASET_SEED,
    DATASET_SPLIT,
    STORAGE_BASE,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("listenr.build_dataset")

# ---------------------------------------------------------------------------
# Defaults (sourced from constants, which read from config at import time)
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT       = DATASET_OUTPUT
DEFAULT_SPLIT        = DATASET_SPLIT
DEFAULT_MIN_DURATION = DATASET_MIN_DURATION
DEFAULT_MIN_CHARS    = DATASET_MIN_CHARS
DEFAULT_SEED         = DATASET_SEED
DEFAULT_FORMAT       = DATASET_FORMAT

CSV_COLUMNS = [
    "uuid",
    "split",
    "audio_path",
    "raw_transcription",
    "corrected_transcription",
    "is_improved",
    "duration_s",
    "sample_rate",
    "whisper_model",
    "llm_model",
    "timestamp",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _manifest_path() -> Path:
    """Return the manifest.jsonl path from config."""
    return STORAGE_BASE / "manifest.jsonl"


def load_manifest(manifest_path: Path) -> list[dict]:
    """Load all records from manifest.jsonl."""
    if not manifest_path.exists():
        logger.warning(f"Manifest not found: {manifest_path}")
        return []
    records = []
    with open(manifest_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.debug(f"Skipping malformed line: {e}")
    return records


def validate_entry(
    data: dict,
    min_duration: float,
    min_chars: int,
) -> dict | None:
    """Validate a manifest record; return None if it fails."""
    for field in ("uuid", "raw_transcription", "audio_path"):
        if not data.get(field):
            logger.debug(f"Skipping record {data.get('uuid', '?')}: missing field '{field}'")
            return None

    duration = float(data.get("duration_s") or 0.0)
    if duration < min_duration:
        logger.debug(f"Skipping {data['uuid']}: duration {duration:.2f}s < {min_duration}s")
        return None

    transcript = (data.get("corrected_transcription") or data.get("raw_transcription") or "").strip()
    if len(transcript.replace(" ", "")) < min_chars:
        logger.debug(f"Skipping {data['uuid']}: transcript too short")
        return None

    audio_path = Path(data["audio_path"]).expanduser()
    if not audio_path.exists():
        logger.debug(f"Skipping {data['uuid']}: audio file missing at {audio_path}")
        return None

    return {
        "uuid": data.get("uuid", ""),
        "audio_path": str(audio_path.resolve()),
        "raw_transcription": data.get("raw_transcription", ""),
        "corrected_transcription": data.get("corrected_transcription") or data.get("raw_transcription", ""),
        "is_improved": str(data.get("is_improved", False)).lower() == "true",
        "duration_s": duration,
        "sample_rate": int(data.get("sample_rate") or 16000),
        "whisper_model": data.get("whisper_model", ""),
        "llm_model": data.get("llm_model", ""),
        "timestamp": data.get("timestamp", ""),
    }


def parse_split(split_str: str) -> tuple[float, float, float]:
    """Parse 'train/dev/test' percentage string into floats that sum to 1.0."""
    parts = split_str.split("/")
    if len(parts) != 3:
        raise ValueError(f"Split must be in format TRAIN/DEV/TEST, got: {split_str!r}")
    values = [float(p) for p in parts]
    total = sum(values)
    if total <= 0:
        raise ValueError("Split values must sum to a positive number")
    return tuple(v / total for v in values)  # type: ignore[return-value]


def assign_splits(
    entries: list[dict],
    train_frac: float,
    dev_frac: float,
    seed: int = 42,
) -> list[dict]:
    """Shuffle entries and assign split labels in-place."""
    rng = random.Random(seed)
    shuffled = entries[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    train_end = int(n * train_frac)
    dev_end = train_end + int(n * dev_frac)
    for i, entry in enumerate(shuffled):
        if i < train_end:
            entry["split"] = "train"
        elif i < dev_end:
            entry["split"] = "dev"
        else:
            entry["split"] = "test"
    return shuffled


def write_csv(entries: list[dict], output_dir: Path, split: str) -> Path:
    """Write entries for a single split to CSV."""
    split_entries = [e for e in entries if e["split"] == split]
    out_path = output_dir / f"{split}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(split_entries)
    return out_path


def write_hf_dataset(entries: list[dict], output_dir: Path) -> None:
    """Write a HuggingFace DatasetDict to output_dir (requires 'datasets' package)."""
    try:
        from datasets import Dataset, DatasetDict, Audio as HFAudio  # type: ignore
    except ImportError:
        logger.error(
            "The 'datasets' package is required for HuggingFace format. "
            "Install it with: uv pip install datasets"
        )
        sys.exit(1)

    splits_dict: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    for e in entries:
        splits_dict[e["split"]].append(e)

    hf_splits = {}
    for split_name, split_entries in splits_dict.items():
        if not split_entries:
            continue
        ds = Dataset.from_list(split_entries)
        ds = ds.cast_column("audio_path", HFAudio(sampling_rate=16000))
        hf_splits[split_name] = ds

    dd = DatasetDict(hf_splits)
    dd.save_to_disk(str(output_dir / "hf_dataset"))
    logger.info(f"HuggingFace dataset saved to {output_dir / 'hf_dataset'}")


def print_stats(entries: list[dict]) -> None:
    """Print a summary of the dataset."""
    total = len(entries)
    if total == 0:
        logger.info("No valid entries.")
        return
    split_counts = {}
    for e in entries:
        split_counts[e["split"]] = split_counts.get(e["split"], 0) + 1
    total_dur = sum(e["duration_s"] for e in entries)
    improved = sum(1 for e in entries if e["is_improved"])
    models = {e["whisper_model"] for e in entries if e["whisper_model"]}

    print("\n----------- Dataset Summary -----------")
    print(f"  Total utterances : {total:,}")
    print(f"  Total duration   : {total_dur / 60:.1f} minutes ({total_dur:.0f}s)")
    print(f"  LLM improved     : {improved:,} ({100 * improved / total:.1f}%)")
    print(f"  Whisper models   : {', '.join(sorted(models)) or 'unknown'}")
    print(f"  Splits           :", end="")
    for s in ("train", "dev", "test"):
        n = split_counts.get(s, 0)
        print(f"  {s}={n}", end="")
    print()
    print("---------------------------------------\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build train/dev/test dataset splits from Listenr recordings."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to manifest.jsonl (default: from config)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory for dataset files (default: from config, currently {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help=f"Train/dev/test split percentages, e.g. 80/10/10 (default: from config, currently {DEFAULT_SPLIT})",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=DEFAULT_MIN_DURATION,
        help=f"Minimum clip duration in seconds (default: from config, currently {DEFAULT_MIN_DURATION})",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=DEFAULT_MIN_CHARS,
        help=f"Minimum non-whitespace chars in transcription (default: from config, currently {DEFAULT_MIN_CHARS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducible splits (default: from config, currently {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "hf", "both"],
        default=DEFAULT_FORMAT,
        help=f"Output format: csv, hf (HuggingFace datasets), or both (default: from config, currently {DEFAULT_FORMAT})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats and exit without writing files",
    )
    args = parser.parse_args()

    manifest_path = args.manifest or _manifest_path()
    output_dir = Path(args.output).expanduser()

    try:
        train_frac, dev_frac, _test_frac = parse_split(args.split)
    except ValueError as e:
        logger.error(f"Invalid --split value: {e}")
        sys.exit(1)

    records = load_manifest(manifest_path)
    logger.info(f"Loaded {len(records)} records from {manifest_path}")

    entries = []
    skipped = 0
    for rec in records:
        entry = validate_entry(rec, args.min_duration, args.min_chars)
        if entry:
            entries.append(entry)
        else:
            skipped += 1

    logger.info(f"Valid entries: {len(entries)}, skipped: {skipped}")

    if not entries:
        logger.error("No valid entries found. Check your recordings directory.")
        sys.exit(1)

    entries = assign_splits(entries, train_frac, dev_frac, seed=args.seed)
    print_stats(entries)

    if args.dry_run:
        logger.info("Dry run — no files written.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.format in ("csv", "both"):
        for split_name in ("train", "dev", "test"):
            out_path = write_csv(entries, output_dir, split_name)
            n = sum(1 for e in entries if e["split"] == split_name)
            logger.info(f"Wrote {n:,} entries -> {out_path}")

    if args.format in ("hf", "both"):
        write_hf_dataset(entries, output_dir)

    logger.info(f"Dataset written to {output_dir}")


if __name__ == "__main__":
    main()
