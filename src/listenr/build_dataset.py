#!/usr/bin/env python3
"""
build_dataset.py — Build train/dev/test splits from Listenr recordings.

Reads manifest.jsonl saved by the CLI, filters/validates entries, and writes
CSV (and optionally HuggingFace datasets) split files.

Usage:
    listenr build-dataset [options]

Examples:
    # Default: 80/10/10 split, CSV output in ~/listenr_dataset/
    listenr build-dataset

    # Custom output directory and split ratio
    listenr build-dataset --output ~/my_dataset --split 90/5/5

    # Only include clips longer than 1 second, HuggingFace format
    listenr build-dataset --min-duration 1.0 --format hf

    # Preview without writing files
    listenr build-dataset --dry-run
"""

import argparse
import csv
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path

from listenr.settings import settings
from listenr.transcript_utils import implausible_speech_rate, strip_noise_tags

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("listenr.build_dataset")

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
    return settings.storage.audio_clips_path / "manifest.jsonl"


def load_manifests(manifest_paths: list[Path]) -> list[dict]:
    """Load and concatenate records from one or more manifest files."""
    records: list[dict] = []
    for manifest_path in manifest_paths:
        records.extend(load_manifest(manifest_path))
    return records


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
    strip_tags: bool = True,
    reasons: "Counter[str] | None" = None,
) -> dict | None:
    """Validate a manifest record; return None if it fails.

    Parameters
    ----------
    strip_tags : if True, parenthesised/bracketed noise tags such as (music) or
                 [Applause] are stripped from both transcription fields before
                 validation and output.
    reasons    : optional Counter, incremented with why a record was dropped.
                 The per-record detail is logged at DEBUG, so without this a
                 caller can only report a bare skip count and cannot tell short
                 clips from missing audio from over-stripped transcripts.
    """
    def _drop(reason: str) -> None:
        if reasons is not None:
            reasons[reason] += 1
    for field in ("uuid", "raw_transcription", "audio_path"):
        if not data.get(field):
            logger.debug(f"Skipping record {data.get('uuid', '?')}: missing field '{field}'")
            _drop(f"missing field '{field}'")
            return None

    duration = float(data.get("duration_s") or 0.0)

    # A separate case from "too short", and a separate reason. Older sessions
    # wrote zero-frame WAVs with a transcript attached: the file exists, so any
    # check that tests for path existence passes, and only the frame count
    # gives it away. Reporting these as "shorter than --min-duration" is true
    # but points at the wrong problem, and --min-duration 0 would let them
    # through entirely.
    if duration == 0.0:
        logger.debug(f"Skipping {data['uuid']}: clip contains no audio")
        _drop("clip contains no audio (zero duration)")
        return None

    if duration < min_duration:
        logger.debug(f"Skipping {data['uuid']}: duration {duration:.2f}s < {min_duration}s")
        _drop(f"shorter than --min-duration ({min_duration}s)")
        return None

    raw = data.get("raw_transcription", "") or ""
    corrected = data.get("corrected_transcription") or raw

    if strip_tags:
        raw = strip_noise_tags(raw).strip()
        corrected = strip_noise_tags(corrected).strip()

    # Use the raw transcription for the min_chars check (authoritative source)
    if len(raw.replace(" ", "")) < min_chars:
        logger.debug(f"Skipping {data['uuid']}: transcript too short after tag stripping")
        _drop(f"transcript under --min-chars ({min_chars}) after tag stripping")
        return None

    audio_path = Path(data["audio_path"]).expanduser()
    if not audio_path.exists():
        logger.debug(f"Skipping {data['uuid']}: audio file missing at {audio_path}")
        _drop("audio file missing on disk")
        return None

    # Backstop against a clip whose audio and transcript came from different
    # segments. Such a row teaches the model to emit a full sentence from near
    # silence, which is how you train a hallucinator, and nothing else here
    # would catch it: both halves are individually valid.
    rate = implausible_speech_rate(raw, duration)
    if rate is not None:
        logger.debug(
            f"Skipping {data['uuid']}: {rate:.0f} words/s over {duration:.3f}s "
            "means the audio and transcript do not match"
        )
        _drop("audio and transcript do not match (impossible speech rate)")
        return None

    return {
        "uuid": data.get("uuid", ""),
        "audio_path": str(audio_path.resolve()),
        "raw_transcription": raw,
        "corrected_transcription": corrected,
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


#: Values a source corpus may use for its splits, normalised to ours.
_SOURCE_SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "dev": "dev",
    "validation": "dev",
    "valid": "dev",
    "eval": "dev",
    "test": "test",
    "testing": "test",
}


def normalise_source_split(value: object) -> str | None:
    """Map a corpus's own split name onto train/dev/test, or None if unknown."""
    if not isinstance(value, str):
        return None
    return _SOURCE_SPLIT_ALIASES.get(value.strip().lower())


def count_source_splits(entries: list[dict]) -> tuple[int, int]:
    """Return (labelled, unlabelled) counts for source-assigned splits."""
    labelled = sum(
        1 for e in entries if normalise_source_split(e.get("source_split")) is not None
    )
    return labelled, len(entries) - labelled


def preserve_source_splits(entries: list[dict]) -> list[dict]:
    """Adopt each entry's source split; put unlabelled entries in train.

    Real corpora are not uniformly labelled. MDC Spontaneous Speech 4.0 ships
    2,425 records of which 120 carry no split at all. Sending those to train is
    the conservative choice: test stays exactly the corpus's own test set, so
    an unlabelled clip can never contaminate the evaluation or inflate a WER
    improvement. The caller reports how many were placed this way.
    """
    for entry in entries:
        entry["split"] = normalise_source_split(entry.get("source_split")) or "train"
    return entries


def shuffle_splits(
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


def assign_splits(
    entries: list[dict],
    train_frac: float,
    dev_frac: float,
    seed: int = 42,
    preserve: bool | None = None,
) -> tuple[list[dict], str]:
    """Assign train/dev/test labels, and report which way it was decided.

    A random reshuffle is the right default for your own recordings, and the
    wrong thing for an imported corpus. Public corpora keep speakers disjoint
    across their splits deliberately; reshuffling puts the same speaker in
    train and test, so the model is evaluated on voices it trained on and any
    WER improvement is inflated. It also makes the result incomparable to
    every published baseline for that corpus, because it is no longer that
    corpus's test set.

    The importers already record the source's own split as ``source_split``,
    so the information is there to respect.

    preserve:
        True  keep the source splits, and fail if nothing carries one.
        False always reshuffle.
        None  keep them when anything carries one, otherwise reshuffle.

    Entries with no source split go to train, never to dev or test, so an
    unlabelled clip cannot contaminate the evaluation.

    Returns ``(entries, how)`` where *how* is "preserved" or "shuffled", so the
    caller can say which happened. Silently reshuffling is how this goes
    unnoticed.
    """
    labelled, _unlabelled = count_source_splits(entries)

    if preserve and labelled == 0:
        raise ValueError(
            "--preserve-splits was requested but no entry has a usable "
            "source_split. Records from `listenr record` never have one; only "
            "imported corpora do. Drop the flag to reshuffle."
        )

    if preserve is False or labelled == 0:
        return shuffle_splits(entries, train_frac, dev_frac, seed=seed), "shuffled"

    return preserve_source_splits(entries), "preserved"


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
        from datasets import Dataset, DatasetDict  # type: ignore
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
        # Keep audio_path as a plain string so datasets never tries to decode
        # it automatically (datasets 4+ requires torchcodec for Audio features).
        # prepare_example loads the WAV on-the-fly with soundfile instead.
        ds = Dataset.from_list(split_entries)
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
        action="append",
        default=None,
        help=(
            "Path to manifest.jsonl. Pass more than once to combine multiple "
            "manifests. Defaults to the configured primary manifest when omitted."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.dataset.output_path,
        help=f"Output directory for dataset files (default: from config, currently {settings.dataset.output_path})",
    )
    parser.add_argument(
        "--split",
        default=settings.dataset.split,
        help=f"Train/dev/test split percentages, e.g. 80/10/10 (default: from config, currently {settings.dataset.split})",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=settings.dataset.min_duration,
        help=f"Minimum clip duration in seconds (default: from config, currently {settings.dataset.min_duration})",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=settings.dataset.min_chars,
        help=f"Minimum non-whitespace chars in transcription (default: from config, currently {settings.dataset.min_chars})",
    )
    parser.add_argument(
        "--preserve-splits",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Keep the train/dev/test split the source corpus assigned, instead "
            "of reshuffling. Default: keep them when every record has one, "
            "reshuffle otherwise. Reshuffling an imported corpus breaks its "
            "speaker-disjoint splits and inflates any WER improvement."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=settings.dataset.seed,
        help=f"Random seed for reproducible splits (default: from config, currently {settings.dataset.seed})",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "hf", "both"],
        default=settings.dataset.format,
        help=f"Output format: csv, hf (HuggingFace datasets), or both (default: from config, currently {settings.dataset.format})",
    )
    parser.add_argument(
        "--no-strip-tags",
        action="store_true",
        default=not settings.dataset.strip_tags,
        help="Preserve parenthesised/bracketed noise tags (e.g. (music)) in transcriptions",
    )
    parser.add_argument(
        "--remap-audio-prefix",
        metavar="OLD:NEW",
        default=None,
        help=(
            "Rewrite the leading path component of every audio_path in the manifest. "
            "Useful when running inside a container where the host path is mounted at "
            "a different location. Example: "
            "--remap-audio-prefix /home/you/.listenr/audio_clips:/data/listenr/audio_clips"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats and exit without writing files",
    )
    args = parser.parse_args()

    # Parse the prefix remap if provided
    remap_old: str | None = None
    remap_new: str | None = None
    if args.remap_audio_prefix:
        parts = args.remap_audio_prefix.split(":", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            logger.error("--remap-audio-prefix must be in the form OLD:NEW (two non-empty paths separated by ':')") 
            sys.exit(1)
        remap_old, remap_new = parts[0], parts[1]

    manifest_paths = args.manifest or [_manifest_path()]
    output_dir = Path(args.output).expanduser()

    try:
        train_frac, dev_frac, _test_frac = parse_split(args.split)
    except ValueError as e:
        logger.error(f"Invalid --split value: {e}")
        sys.exit(1)

    records = load_manifests(manifest_paths)
    logger.info(
        "Loaded %d record(s) from %d manifest(s)",
        len(records),
        len(manifest_paths),
    )

    if remap_old is not None:
        remapped = 0
        for rec in records:
            p = rec.get("audio_path", "")
            if p.startswith(remap_old):
                rec["audio_path"] = remap_new + p[len(remap_old):]
                remapped += 1
        logger.info(f"Remapped {remapped}/{len(records)} audio paths: {remap_old!r} -> {remap_new!r}")

    entries = []
    skipped = 0
    skip_reasons: Counter[str] = Counter()
    for rec in records:
        entry = validate_entry(
            rec,
            args.min_duration,
            args.min_chars,
            strip_tags=not args.no_strip_tags,
            reasons=skip_reasons,
        )
        if entry:
            entries.append(entry)
        else:
            skipped += 1

    logger.info(f"Valid entries: {len(entries)}, skipped: {skipped}")
    for reason, count in skip_reasons.most_common():
        logger.info(f"  skipped {count}: {reason}")

    if not entries:
        logger.error("No valid entries found. Check your recordings directory.")
        sys.exit(1)

    try:
        entries, how = assign_splits(
            entries, train_frac, dev_frac, seed=args.seed, preserve=args.preserve_splits
        )
    except ValueError as exc:
        logger.error(str(exc))
        sys.exit(1)

    if how == "preserved":
        labelled, unlabelled = count_source_splits(entries)
        logger.info(
            "Splits: kept the source corpus's own train/dev/test for %d record(s). "
            "Speakers stay disjoint and results stay comparable to its published baselines.",
            labelled,
        )
        if unlabelled:
            logger.info(
                "  %d record(s) had no source split and went to train, so they "
                "cannot affect the evaluation. Pass --no-preserve-splits to "
                "reshuffle everything instead.",
                unlabelled,
            )
    else:
        logger.info(f"Splits: shuffled at {args.split} with seed {args.seed}.")
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
