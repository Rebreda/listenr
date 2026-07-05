#!/usr/bin/env python3
"""Import Hugging Face ASR datasets into a Listenr manifest.

Audio is loaded with ``decode=False`` (raw bytes, no decode backend needed)
and materialised to WAV by the shared manifest core. Defaults target Common
Voice-style datasets (``audio`` + ``sentence``); other layouts can be mapped
with ``--audio-column`` / ``--text-column``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from listenr.importers import manifest as m
from listenr.importers._cli import add_common_arguments, load_env, log_summary
from listenr.importers.mapping import FieldMapping

logger = logging.getLogger("listenr.importers.hf")

SOURCE = "hf"
HF_MAPPING = FieldMapping(audio=("audio", "audio_path"))


def _pick_audio_column(column_names: list[str], mapping: FieldMapping) -> str:
    for candidate in mapping.audio:
        if candidate in column_names:
            return candidate
    raise RuntimeError(
        f"None of the audio columns {mapping.audio} are present. "
        f"Available columns: {column_names}. Use --audio-column to choose one."
    )


def _iter_rows(dataset_id: str, config: str | None, split: str | None, mapping: FieldMapping):
    """Yield row dicts (with a ``split`` key) for a HF dataset, audio undecoded."""
    try:
        from datasets import Audio, load_dataset  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required for Hugging Face imports. Install it with: uv pip install -e .[hf]"
        ) from exc

    loaded = load_dataset(dataset_id, config, split=split)

    # Normalise to a mapping of split-name -> Dataset.
    if hasattr(loaded, "column_names") and isinstance(loaded.column_names, dict):
        splits = dict(loaded)  # DatasetDict
    elif isinstance(getattr(loaded, "column_names", None), list):
        splits = {split or "train": loaded}  # single Dataset
    else:
        raise RuntimeError("Unsupported object returned by datasets.load_dataset.")

    def generate() -> Iterator[Mapping[str, Any]]:
        for split_name, ds in splits.items():
            audio_col = _pick_audio_column(list(ds.column_names), mapping)
            ds = ds.cast_column(audio_col, Audio(decode=False))
            for row in ds:
                yield {**row, "split": row.get("split") or split_name}

    return generate()


def import_hf_dataset(
    dataset_id: str,
    manifest_path: Path,
    *,
    config: str | None = None,
    split: str | None = None,
    mapping: FieldMapping = HF_MAPPING,
    append: bool = False,
) -> dict[str, Any]:
    rows = _iter_rows(dataset_id, config, split, mapping)
    return m.import_rows(
        rows,
        source=SOURCE,
        dataset_id=dataset_id,
        mapping=mapping,
        manifest_path=manifest_path,
        audio_dir=m.default_audio_dir(SOURCE, dataset_id),
        dataset_name=dataset_id,
        append=append,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load a Hugging Face ASR dataset and write a Listenr-compatible manifest."
    )
    parser.add_argument("dataset_id", help="Hugging Face dataset id, e.g. mozilla-foundation/common_voice_17_0")
    parser.add_argument("--config", default=None, help="Dataset config/language name (e.g. 'en').")
    parser.add_argument(
        "--split",
        default=None,
        help="Single split to import (e.g. 'train'). Omit to import all splits.",
    )
    add_common_arguments(parser)
    args = parser.parse_args()
    load_env()

    manifest_path = args.manifest or m.default_manifest_path(SOURCE, args.dataset_id)
    mapping = HF_MAPPING.with_overrides(
        audio=args.audio_column, text=args.text_column, split=args.split_column
    )

    try:
        summary = import_hf_dataset(
            args.dataset_id,
            manifest_path=manifest_path,
            config=args.config,
            split=args.split,
            mapping=mapping,
            append=args.append,
        )
    except RuntimeError as exc:
        logger.error(str(exc))
        sys.exit(1)

    log_summary(logger, summary)


if __name__ == "__main__":
    main()
