#!/usr/bin/env python3
"""Import Mozilla Data Collective ASR datasets into a Listenr manifest.

The ``datacollective`` SDK applies each dataset's schema internally and hands
back a pandas DataFrame with normalised (logical) column names — typically
``audio_path`` and ``transcription`` for ASR. We convert those rows onto the
shared manifest core; unusual column names can be handled per dataset with
``--audio-column`` / ``--text-column`` without code changes.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from listenr.importers import manifest as m
from listenr.importers._cli import add_common_arguments, load_env, log_summary
from listenr.importers.mapping import FieldMapping

logger = logging.getLogger("listenr.importers.mdc")

SOURCE = "mdc"
MDC_MAPPING = FieldMapping()  # defaults (audio_path / transcription) fit MDC ASR


def load_rows(dataset_id: str, enable_logging: bool = False) -> tuple[list[dict], str]:
    """Load an MDC dataset into row dicts plus its display name."""
    try:
        from datacollective import get_dataset_details, load_dataset  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "datacollective is required for MDC imports. Install it with: uv pip install -e .[mdc]"
        ) from exc

    dataset_name = ""
    try:
        details = get_dataset_details(dataset_id)
        if isinstance(details, dict):
            dataset_name = str(details.get("name") or details.get("title") or "").strip()
    except Exception as exc:  # pragma: no cover - details are best-effort metadata
        logger.debug("Failed to fetch MDC dataset details: %s", exc)

    try:
        df = load_dataset(dataset_id, enable_logging=enable_logging)
    except TypeError:
        df = load_dataset(dataset_id)

    if not isinstance(df, pd.DataFrame):
        raise RuntimeError("MDC returned an unsupported dataset object; expected a pandas DataFrame.")

    return df.to_dict("records"), dataset_name


def import_mdc_dataset(
    dataset_id: str,
    manifest_path: Path,
    *,
    mapping: FieldMapping = MDC_MAPPING,
    append: bool = False,
    enable_logging: bool = False,
) -> dict[str, Any]:
    rows, dataset_name = load_rows(dataset_id, enable_logging=enable_logging)
    return m.import_rows(
        rows,
        source=SOURCE,
        dataset_id=dataset_id,
        mapping=mapping,
        manifest_path=manifest_path,
        audio_dir=m.default_audio_dir(SOURCE, dataset_id),
        dataset_name=dataset_name,
        append=append,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download an MDC ASR dataset and write a Listenr-compatible manifest."
    )
    parser.add_argument("dataset_id", help="Mozilla Data Collective dataset id or slug")
    add_common_arguments(parser)
    parser.add_argument(
        "--enable-mdc-logging",
        action="store_true",
        help="Enable verbose logging from the datacollective SDK.",
    )
    args = parser.parse_args()
    load_env()

    manifest_path = args.manifest or m.default_manifest_path(SOURCE, args.dataset_id)
    mapping = MDC_MAPPING.with_overrides(
        audio=args.audio_column, text=args.text_column, split=args.split_column
    )

    try:
        summary = import_mdc_dataset(
            args.dataset_id,
            manifest_path=manifest_path,
            mapping=mapping,
            append=args.append,
            enable_logging=args.enable_mdc_logging,
        )
    except RuntimeError as exc:
        logger.error(str(exc))
        sys.exit(1)

    log_summary(logger, summary)


if __name__ == "__main__":
    main()
