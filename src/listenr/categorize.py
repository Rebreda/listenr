#!/usr/bin/env python3
"""Filter a Listenr manifest to clips whose transcription matches a topic.

Uses sentence-embedding similarity: every transcription is embedded once and
scored (cosine) against one or more topic phrases you supply — e.g.
``--topic "technology" --topic "artificial intelligence"``. Each record's score
is the best match across your topics. Records at or above ``--threshold`` are
written to a new manifest that plugs straight into ``listenr-build-dataset``.

This is source-agnostic: it works on any manifest (your own recordings, an MDC
import, an HF import, or a combination). It never modifies the input manifest.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from listenr.build_dataset import load_manifest
from listenr.importers.manifest import write_manifest

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("listenr.categorize")

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _text(record: dict, text_field: str | None = None) -> str:
    if text_field:
        return str(record.get(text_field, "") or "").strip()
    return str(record.get("corrected_transcription") or record.get("raw_transcription") or "").strip()


def build_encoder(model_name: str = DEFAULT_MODEL):
    """Return an ``encode(texts) -> np.ndarray`` of L2-normalised embeddings."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for categorization. "
            "Install it with: uv pip install -e .[categorize]"
        ) from exc

    model = SentenceTransformer(model_name)

    def encode(texts: list[str]) -> np.ndarray:
        return np.asarray(
            model.encode(
                list(texts),
                normalize_embeddings=True,
                batch_size=64,
                show_progress_bar=len(texts) > 256,
            )
        )

    return encode


def score_records(
    records: list[dict],
    topics: list[str],
    encode,
    text_field: str | None = None,
) -> list[tuple[float, str, dict]]:
    """Return ``(best_score, best_topic, record)`` for each record.

    Embeddings are L2-normalised, so a dot product is cosine similarity.
    """
    if not records:
        return []
    topic_emb = encode(topics)  # (T, d)
    text_emb = encode([_text(r, text_field) for r in records])  # (N, d)
    sims = text_emb @ topic_emb.T  # (N, T)

    results: list[tuple[float, str, dict]] = []
    for record, row in zip(records, sims):
        best = int(np.argmax(row))
        results.append((float(row[best]), topics[best], record))
    return results


def filter_records(
    scored: list[tuple[float, str, dict]],
    threshold: float,
    keep_all: bool = False,
    annotate: bool = True,
) -> list[dict]:
    """Keep matches (or all, if ``keep_all``), annotating score/category."""
    kept: list[dict] = []
    for score, topic, record in scored:
        matched = score >= threshold
        if not matched and not keep_all:
            continue
        out = dict(record)
        if annotate:
            out["topic_score"] = round(score, 4)
            out["category"] = topic if matched else ""
            out["topic_matched"] = matched
        kept.append(out)
    return kept


def _load_topics(topic_args: list[str] | None, topics_file: Path | None) -> list[str]:
    topics: list[str] = list(topic_args or [])
    if topics_file:
        for line in topics_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                topics.append(line)
    return topics


def _report(scored: list[tuple[float, str, dict]], threshold: float) -> None:
    scores = np.array([s for s, _, _ in scored]) if scored else np.array([0.0])
    matched = int((scores >= threshold).sum())
    logger.info(
        "Scored %d clip(s): %d match at threshold %.2f (score min/median/max = %.3f/%.3f/%.3f)",
        len(scored),
        matched,
        threshold,
        float(scores.min()),
        float(np.median(scores)),
        float(scores.max()),
    )
    ranked = sorted(scored, key=lambda t: t[0], reverse=True)
    logger.info("Top matches:")
    for score, topic, record in ranked[:5]:
        logger.info("  %.3f [%s] %s", score, topic, _text(record)[:80])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter a Listenr manifest to clips matching one or more topics via embeddings."
    )
    parser.add_argument("manifest", type=Path, help="Input manifest.jsonl (never modified).")
    parser.add_argument(
        "--topic",
        action="append",
        dest="topics",
        default=None,
        metavar="PHRASE",
        help="A topic phrase to match, e.g. --topic 'technology'. Repeatable.",
    )
    parser.add_argument(
        "--topics-file",
        type=Path,
        default=None,
        help="File with one topic phrase per line (# comments allowed).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Minimum cosine similarity to keep a clip (default: 0.35).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination manifest of matching clips. Required unless --dry-run.",
    )
    parser.add_argument(
        "--keep-all",
        action="store_true",
        help="Write every clip (annotated) instead of only matches.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Embedding model (default: {DEFAULT_MODEL}).")
    parser.add_argument("--text-field", default=None, help="Manifest field to classify (default: corrected/raw transcription).")
    parser.add_argument("--dry-run", action="store_true", help="Report match counts and top hits without writing.")
    args = parser.parse_args()

    topics = _load_topics(args.topics, args.topics_file)
    if not topics:
        logger.error("Provide at least one --topic or a --topics-file.")
        sys.exit(1)
    if not args.dry_run and args.output is None:
        logger.error("--output is required unless --dry-run is set.")
        sys.exit(1)

    records = load_manifest(args.manifest)
    if not records:
        logger.error("No records found in %s", args.manifest)
        sys.exit(1)
    logger.info("Loaded %d record(s); topics: %s", len(records), ", ".join(topics))

    try:
        encode = build_encoder(args.model)
    except RuntimeError as exc:
        logger.error(str(exc))
        sys.exit(1)

    scored = score_records(records, topics, encode, text_field=args.text_field)
    _report(scored, args.threshold)

    if args.dry_run:
        return

    kept = filter_records(scored, args.threshold, keep_all=args.keep_all)
    write_manifest(kept, args.output)
    logger.info("Wrote %d record(s) -> %s", len(kept), args.output)


if __name__ == "__main__":
    main()
