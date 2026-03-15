#!/usr/bin/env python3
"""
retranscribe.py — Re-run Whisper (and optionally LLM) on saved audio clips.

Reads manifest.jsonl, re-sends each matching WAV file to Lemonade, and
rewrites the manifest with updated ``raw_transcription`` (and optionally
``corrected_transcription``) fields.

Usage examples::

    # Re-transcribe every clip (uses the configured default Whisper model)
    listenr-retranscribe

    # Use a different model
    listenr-retranscribe --model Whisper-Large-v3-Turbo

    # Only clips whose current raw transcript matches a regex
    listenr-retranscribe --match "hello world"

    # Only specific UUIDs
    listenr-retranscribe --uuid abc123 def456

    # Re-transcribe AND re-run LLM correction
    listenr-retranscribe --llm

    # Preview what would change without modifying the manifest
    listenr-retranscribe --dry-run

    # Custom manifest location
    listenr-retranscribe --manifest /data/recordings/manifest.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import tempfile
from pathlib import Path

from listenr.constants import STORAGE_BASE, WHISPER_MODEL
from listenr.llm_processor import lemonade_llm_correct, lemonade_transcribe_audio
from listenr.transcript_utils import clean_transcript

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("listenr.retranscribe")


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _default_manifest() -> Path:
    return STORAGE_BASE / "manifest.jsonl"


def _load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            logger.warning("Skipping malformed line %d in manifest: %s", lineno, exc)
    return records


def _write_manifest(path: Path, records: list[dict]) -> None:
    """Atomically rewrite the manifest."""
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _should_process(
    record: dict,
    uuids: set[str] | None,
    pattern: re.Pattern | None,
) -> bool:
    """Return True if this record passes the active filters."""
    if uuids is not None and record.get("uuid") not in uuids:
        return False
    if pattern is not None:
        text = record.get("raw_transcription", "")
        if not pattern.search(text):
            return False
    return True


def retranscribe(
    manifest_path: Path,
    *,
    model: str | None = None,
    use_llm: bool = False,
    uuids: set[str] | None = None,
    pattern: re.Pattern | None = None,
    dry_run: bool = False,
) -> dict:
    """Re-transcribe audio clips referenced in a manifest.

    Parameters
    ----------
    manifest_path:
        Path to ``manifest.jsonl``.
    model:
        Whisper model name passed to Lemonade.  Defaults to the configured
        ``WHISPER_MODEL`` constant.
    use_llm:
        If True, also re-run LLM correction on the new raw transcript.
    uuids:
        If provided, only process records whose ``uuid`` is in this set.
    pattern:
        If provided, only process records whose current ``raw_transcription``
        matches this compiled regex.
    dry_run:
        If True, perform all network calls and log what would change, but do
        not write back to the manifest.

    Returns
    -------
    Summary dict with keys ``total``, ``processed``, ``updated``, ``errors``,
    ``skipped``.
    """
    effective_model = model or WHISPER_MODEL
    records = _load_manifest(manifest_path)

    total = len(records)
    processed = updated = errors = skipped = 0

    for record in records:
        uuid = record.get("uuid", "?")

        if not _should_process(record, uuids, pattern):
            skipped += 1
            continue

        audio_path = record.get("audio_path", "")
        if not audio_path or not Path(audio_path).exists():
            logger.warning("[%s] audio file not found: %s — skipping", uuid, audio_path)
            errors += 1
            continue

        processed += 1
        logger.info("[%s] transcribing %s …", uuid, Path(audio_path).name)

        try:
            new_raw = lemonade_transcribe_audio(audio_path, model=effective_model)
        except Exception as exc:
            logger.error("[%s] transcription failed: %s", uuid, exc)
            errors += 1
            continue

        # Clean the new transcript the same way the live pipeline does.
        action, cleaned_raw = clean_transcript(new_raw)
        if action == "drop":
            logger.info("[%s] result is a hallucination — keeping original", uuid)
            skipped += 1
            processed -= 1
            continue

        old_raw = record.get("raw_transcription", "")
        changed = cleaned_raw != old_raw

        new_corrected = cleaned_raw
        llm_model_used = None
        is_improved = False

        if use_llm:
            try:
                llm_result = lemonade_llm_correct(cleaned_raw)
                if "error" not in llm_result:
                    new_corrected = llm_result.get("corrected_text", cleaned_raw)
                    is_improved = bool(llm_result.get("is_improved", False))
                    llm_model_used = llm_result.get("model") if is_improved else None
                    if new_corrected != record.get("corrected_transcription", ""):
                        changed = True
                else:
                    logger.warning("[%s] LLM error: %s", uuid, llm_result["error"])
            except Exception as exc:
                logger.warning("[%s] LLM call failed: %s", uuid, exc)

        if changed:
            updated += 1
            if dry_run:
                logger.info(
                    "[%s] (dry-run) raw: %r → %r",
                    uuid,
                    old_raw,
                    cleaned_raw,
                )
            else:
                record["raw_transcription"] = cleaned_raw
                record["corrected_transcription"] = (
                    new_corrected if new_corrected else cleaned_raw
                )
                record["whisper_model"] = effective_model
                if use_llm:
                    record["is_improved"] = is_improved
                    record["llm_model"] = llm_model_used
        else:
            logger.info("[%s] no change", uuid)

    if not dry_run and updated:
        _write_manifest(manifest_path, records)
        logger.info("Manifest updated: %s", manifest_path)
    elif dry_run and updated:
        logger.info("(dry-run) %d record(s) would be updated — manifest not written", updated)

    return {
        "total": total,
        "processed": processed,
        "updated": updated,
        "errors": errors,
        "skipped": skipped,
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="listenr-retranscribe",
        description="Re-run Whisper (and optionally LLM) on saved audio clips.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to manifest.jsonl  (default: <storage_base>/manifest.jsonl)",
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="MODEL",
        help="Whisper model to use (default: configured WHISPER_MODEL)",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Also re-run LLM correction on the new raw transcript",
    )
    parser.add_argument(
        "--uuid",
        nargs="+",
        metavar="UUID",
        help="Only process clips with these UUIDs",
    )
    parser.add_argument(
        "--match",
        metavar="REGEX",
        help="Only process clips whose raw_transcription matches this regex",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change without modifying the manifest",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    manifest = args.manifest or _default_manifest()
    if not manifest.exists():
        logger.error("Manifest not found: %s", manifest)
        sys.exit(1)

    uuids: set[str] | None = set(args.uuid) if args.uuid else None

    pattern: re.Pattern | None = None
    if args.match:
        try:
            pattern = re.compile(args.match, re.IGNORECASE)
        except re.error as exc:
            logger.error("Invalid --match regex: %s", exc)
            sys.exit(1)

    summary = retranscribe(
        manifest,
        model=args.model,
        use_llm=args.llm,
        uuids=uuids,
        pattern=pattern,
        dry_run=args.dry_run,
    )

    print(
        f"\nDone — {summary['processed']} processed, "
        f"{summary['updated']} updated, "
        f"{summary['skipped']} skipped, "
        f"{summary['errors']} errors "
        f"(of {summary['total']} total)."
    )
    if summary["errors"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
