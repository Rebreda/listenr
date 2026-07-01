"""Shared CLI plumbing for the source importers."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_env() -> None:
    """Load a local ``.env`` into the environment if python-dotenv is available.

    Importers authenticate via environment variables (``MDC_API_KEY`` for MDC,
    ``HF_TOKEN`` for gated Hugging Face datasets). Loading ``.env`` here lets
    users keep those in a gitignored file instead of exporting them each run.
    No-ops when python-dotenv isn't installed.
    """
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        return
    load_dotenv(find_dotenv(usecwd=True))


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags every importer shares: manifest destination and column overrides."""
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Destination manifest.jsonl (defaults to a per-dataset path under ~/.listenr).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to an existing manifest instead of overwriting it.",
    )
    parser.add_argument("--audio-column", default=None, help="Override the audio source column.")
    parser.add_argument("--text-column", default=None, help="Override the transcription source column.")
    parser.add_argument("--split-column", default=None, help="Override the split source column.")


def log_summary(logger: logging.Logger, summary: dict[str, Any]) -> None:
    logger.info(
        "Imported %d row(s) from %s dataset %s into %s%s",
        summary["imported"],
        summary["source"],
        summary["dataset_id"],
        summary["manifest_path"],
        f" (skipped {summary['skipped']})" if summary["skipped"] else "",
    )
