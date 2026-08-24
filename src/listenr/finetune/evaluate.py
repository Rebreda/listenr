#!/usr/bin/env python3
"""
evaluate.py — Evaluate the merged fine-tuned ASR model on held-out data.

Transcribes the test split of the dataset written by ``listenr build-dataset
--format hf`` with the merged model, compares against the ground-truth
transcriptions, and reports corpus WER (with Whisper's English text
normalization) plus optional per-keyword recall.  With ``--compare-base`` the
base model is run on the same clips so the report shows what fine-tuning
actually changed.

Usage examples::

    # Evaluate the merged model on the test split (all defaults from config)
    listenr eval

    # Also run the base model side-by-side
    listenr eval --compare-base

    # Track specific domain words the fine-tune was meant to fix
    listenr eval --compare-base --keyword Claude --keyword Cursor

    # A different split, fewer clips, custom paths
    listenr eval --split dev --n 10 --model ~/my_merged --dataset ~/my_dataset/hf_dataset

    # Single audio file — prints base vs fine-tuned side-by-side
    listenr eval --audio path/to/clip.wav

Requires the ``finetune`` optional dependencies::

    uv pip install -e ".[finetune]"
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import textwrap
import warnings
from pathlib import Path

from listenr.finetune.merge import DEFAULT_OUTPUT_DIR as DEFAULT_MERGED_DIR
from listenr.settings import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("listenr.finetune.evaluate")

DEFAULT_SPLIT = "test"
DEFAULT_N = 50
_COL_WIDTH = 40


# ---------------------------------------------------------------------------
# Pure helpers (importable without torch/transformers — see tests)
# ---------------------------------------------------------------------------

def keyword_hits(text: str, keywords: list[str]) -> list[str]:
    """Return the subset of *keywords* found (case-insensitive) in *text*.

    Matches at word boundaries so short keywords like "ai" don't hit inside
    unrelated words ("said"); suffixes still match ("robot" -> "robotics").
    """
    t = text.lower()
    return [kw for kw in keywords if re.search(r"\b" + re.escape(kw.lower()), t)]


def keyword_hit_map(reference: str, hypothesis: str, keywords: list[str]) -> dict[str, bool]:
    """For each keyword expected in *reference*, whether *hypothesis* produced it."""
    expected = keyword_hits(reference, keywords)
    found = [kw.lower() for kw in keyword_hits(hypothesis, keywords)]
    return {kw: kw.lower() in found for kw in expected}


def resolve_base_model(merged_path: Path, override: str | None = None) -> str:
    """Determine which base model to compare against.

    Reads the merged model's config.json (``_name_or_path`` is stamped by
    ``listenr merge``); falls back to the configured finetune base model.
    """
    if override:
        return override
    cfg = merged_path / "config.json"
    if cfg.exists():
        try:
            data = json.loads(cfg.read_text())
            name = data.get("_name_or_path") or data.get("base_model_name_or_path")
            # A path pointing back at a local merged dir is not a base model id.
            if name and not Path(str(name)).expanduser().is_dir():
                return name
        except (OSError, json.JSONDecodeError):
            pass
    return settings.finetune.base_model


def select_examples(
    rows: list[dict],
    n: int,
    keywords: list[str] | None = None,
) -> list[dict]:
    """Return up to *n* evaluable rows from a dataset split.

    A row is evaluable when its audio file exists and it has a non-empty
    ``corrected_transcription`` (the ground truth).  If *keywords* is given,
    only rows whose ground truth contains at least one keyword are kept —
    these are the clips where the fine-tune had something to learn.
    """
    kw = [k.lower() for k in keywords] if keywords else []
    selected = []
    for row in rows:
        if not row.get("corrected_transcription"):
            continue
        if not Path(row.get("audio_path", "")).exists():
            continue
        if kw and not any(k in row["corrected_transcription"].lower() for k in kw):
            continue
        selected.append(row)
        if len(selected) >= n:
            break
    return selected


def compute_wer(refs: list[str], hyps: list[str]) -> float | None:
    """Corpus WER (%) with Whisper's official English text normalization.

    Returns None when there are no usable reference/hypothesis pairs or when
    the optional scoring dependencies are missing.
    """
    try:
        import jiwer
        from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    except ImportError:
        return None

    normalize = EnglishTextNormalizer({})
    pairs = [(normalize(r), normalize(h)) for r, h in zip(refs, hyps)]
    pairs = [(r, h) for r, h in pairs if r]
    if not pairs:
        return None
    return 100 * jiwer.wer([r for r, _ in pairs], [h for _, h in pairs])


def tally_keywords(
    results: list[dict[str, dict[str, bool]]],
) -> dict[str, dict[str, list[int]]]:
    """Aggregate per-clip keyword hit maps into {model: {keyword: [hits, expected]}}."""
    tally: dict[str, dict[str, list[int]]] = {}
    for result in results:
        for model, hit_map in result.items():
            model_tally = tally.setdefault(model, {})
            for kw, hit in hit_map.items():
                counts = model_tally.setdefault(kw, [0, 0])
                counts[1] += 1
                if hit:
                    counts[0] += 1
    return tally


# ---------------------------------------------------------------------------
# Model + dataset loading (requires the finetune extras)
# ---------------------------------------------------------------------------

def make_asr(model_id_or_path: str | Path, device: int | str | None = None):
    """Load an ASR pipeline for *model_id_or_path*.

    *device* follows the transformers convention: an int index, "cpu", or None
    to pick automatically. Auto means GPU 0 when one is usable. ROCm torch
    aliases the cuda namespace, so the same check covers AMD and NVIDIA.
    """
    try:
        import torch
        from transformers import pipeline
        from transformers.utils import logging as hf_logging
    except ImportError:
        print(
            "ERROR: transformers and torch are required. Install with:\n"
            "  uv pip install \"listenr[finetune]\"",
            file=sys.stderr,
        )
        sys.exit(1)

    # Silence noisy well-known Whisper quirks (attention_mask, logits-processor
    # duplication) that don't affect transcription quality.
    hf_logging.set_verbosity_error()
    warnings.filterwarnings("ignore", message=".*attention_mask.*")
    warnings.filterwarnings("ignore", message=".*logits_process.*")

    if device is None:
        device = 0 if torch.cuda.is_available() else -1
    elif device == "cpu":
        device = -1
    from listenr.finetune.preflight import describe_accelerator, describe_accelerator_line

    logger.info(describe_accelerator_line(describe_accelerator()))
    logger.info(f"Loading {model_id_or_path} (device={device}) ...")
    return pipeline(
        "automatic-speech-recognition",
        model=str(model_id_or_path),
        device=device,
        # Required for clips longer than 30s (long-form generation); harmless
        # for short clips.
        return_timestamps=True,
    )


def _generate_kwargs(model_id_or_path: str | Path) -> dict:
    """Generation kwargs that match the fine-tune task.

    Base and merged models must be decoded the same way to be compared
    fairly. Language and task tokens only exist on multilingual families.
    Passing them to an English-only model (Moonshine) is an error, so they
    are omitted there.
    """
    from listenr.finetune.architectures import detect

    try:
        arch = detect(str(model_id_or_path))
    except Exception:
        return {}
    if not arch.supports_language_and_task:
        return {}
    return {"language": settings.finetune.language, "task": settings.finetune.task}


def _transcribe(asr, audio_path: str | Path) -> str:
    kwargs = _generate_kwargs(asr.model.name_or_path)
    return asr(str(audio_path), generate_kwargs=kwargs)["text"].strip()


def load_split(dataset_path: Path, split: str) -> list[dict]:
    """Load one split of the on-disk DatasetDict as a list of row dicts."""
    try:
        from datasets import load_from_disk
    except ImportError:
        print(
            "ERROR: datasets is required. Install with:\n"
            "  uv pip install -e \".[finetune]\"",
            file=sys.stderr,
        )
        sys.exit(1)

    dataset = load_from_disk(str(dataset_path))
    if split not in dataset:
        print(
            f"ERROR: split '{split}' not in dataset (has: {', '.join(dataset)})",
            file=sys.stderr,
        )
        sys.exit(1)
    return list(dataset[split])


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _col(text: str, width: int = _COL_WIDTH) -> str:
    return "\n".join(textwrap.wrap(text, width)) if text else "(empty)"


def _print_columns(left_label: str, left: str, right_label: str, right: str) -> None:
    w = _COL_WIDTH
    left_lines = _col(left).split("\n")
    right_lines = _col(right).split("\n")
    print(f"  {left_label:<{w}}  {right_label:<{w}}")
    for i in range(max(len(left_lines), len(right_lines))):
        l = left_lines[i] if i < len(left_lines) else ""
        r = right_lines[i] if i < len(right_lines) else ""
        print(f"  {l:<{w}}  {r}")


def _print_keyword_line(label: str, hit_map: dict[str, bool]) -> None:
    hits = [kw for kw, ok in hit_map.items() if ok]
    misses = [kw for kw, ok in hit_map.items() if not ok]
    parts = []
    if hits:
        parts.append("HIT:  " + ", ".join(hits))
    if misses:
        parts.append("MISS: " + ", ".join(misses))
    if parts:
        print(f"  Keywords ({label}) — " + "   ".join(parts))


def print_clip_result(
    idx: int,
    total: int,
    row: dict,
    merged_text: str,
    base_text: str | None,
    keywords: list[str],
) -> dict[str, dict[str, bool]]:
    """Print one clip's comparison; return keyword hit maps for tallying."""
    reference = row["corrected_transcription"]
    audio_name = Path(row["audio_path"]).name
    duration = float(row.get("duration_s") or 0)

    print(f"\n{'─' * 90}")
    print(f"  Clip {idx}/{total}  {audio_name}  ({duration:.1f}s)")
    print(f"{'─' * 90}")
    if base_text is not None:
        _print_columns("BASE", base_text, "FINE-TUNED (merged)", merged_text)
    else:
        _print_columns("GROUND TRUTH", reference, "FINE-TUNED (merged)", merged_text)
    if base_text is not None:
        print("\n  GROUND TRUTH")
        print(f"  {_col(reference, 2 * _COL_WIDTH)}")

    result: dict[str, dict[str, bool]] = {"merged": {}}
    if keywords:
        result["merged"] = keyword_hit_map(reference, merged_text, keywords)
        print()
        if base_text is not None:
            result["base"] = keyword_hit_map(reference, base_text, keywords)
            _print_keyword_line("base", result["base"])
        _print_keyword_line("fine-tuned", result["merged"])
    return result


def print_summary(
    refs: list[str],
    hyps: dict[str, list[str]],
    keyword_tally: dict[str, dict[str, list[int]]],
) -> None:
    print(f"\n{'─' * 90}")
    wer_rows = [
        (name, compute_wer(refs, texts))
        for name, texts in (("base", hyps.get("base", [])), ("fine-tuned", hyps["merged"]))
        if texts
    ]
    wer_rows = [(name, wer) for name, wer in wer_rows if wer is not None]
    if wer_rows:
        print("\n  WER vs ground truth (Whisper English normalization)")
        for name, wer in wer_rows:
            print(f"    {name:<12}  {wer:.1f}%")

    for model, kw_tally in keyword_tally.items():
        print(f"\n  Keyword recall — {model}")
        for kw, (hits, expected) in kw_tally.items():
            pct = 100 * hits / expected if expected else 0
            bar = ("█" * hits) + ("░" * (expected - hits))
            print(f"    {kw:<20}  {hits}/{expected}  ({pct:.0f}%)  {bar}")
    print(f"{'─' * 90}\n")


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def evaluate_split(args: argparse.Namespace) -> None:
    dataset_path = args.dataset or settings.dataset.output_path / "hf_dataset"
    if not Path(dataset_path).exists():
        print(
            f"ERROR: dataset not found at {dataset_path}\n"
            "Run:  listenr build-dataset --format hf",
            file=sys.stderr,
        )
        sys.exit(1)

    rows = load_split(Path(dataset_path), args.split)
    examples = select_examples(rows, args.n, args.keywords or None)
    if not examples:
        kw_note = (
            f" with '{', '.join(args.keywords)}' in the ground truth" if args.keywords else ""
        )
        print(f"No evaluable clips in split '{args.split}'{kw_note}.", file=sys.stderr)
        sys.exit(1)

    merged_asr = make_asr(args.model, args.device)
    base_asr = make_asr(resolve_base_model(args.model, args.base_model), args.device) if args.compare_base else None

    print(f"\nEvaluating {len(examples)} clips from split '{args.split}' of {dataset_path}")
    if args.keywords:
        print(f"  keywords : {', '.join(args.keywords)}")
        print("  question : does the fine-tuned model now produce the keyword?")

    refs: list[str] = []
    hyps: dict[str, list[str]] = {"merged": [], "base": []}
    results = []
    for i, row in enumerate(examples, 1):
        merged_text = _transcribe(merged_asr, row["audio_path"])
        base_text = _transcribe(base_asr, row["audio_path"]) if base_asr else None
        results.append(
            print_clip_result(i, len(examples), row, merged_text, base_text, args.keywords)
        )
        refs.append(row["corrected_transcription"])
        hyps["merged"].append(merged_text)
        if base_text is not None:
            hyps["base"].append(base_text)

    print_summary(refs, hyps, tally_keywords(results) if args.keywords else {})


def evaluate_single(args: argparse.Namespace) -> None:
    if not args.audio.exists():
        print(f"ERROR: audio file not found: {args.audio}", file=sys.stderr)
        sys.exit(1)
    base_asr = make_asr(resolve_base_model(args.model, args.base_model), args.device)
    base_text = _transcribe(base_asr, args.audio)
    merged_asr = make_asr(args.model, args.device)
    merged_text = _transcribe(merged_asr, args.audio)
    print()
    _print_columns("BASE", base_text, "FINE-TUNED (merged)", merged_text)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the merged fine-tuned ASR model on the held-out test split.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MERGED_DIR,
        metavar="DIR",
        help=f"Merged model directory produced by `listenr merge` (default: {DEFAULT_MERGED_DIR})",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        metavar="DIR",
        help="hf_dataset directory written by `listenr build-dataset --format hf` "
        f"(default: {settings.dataset.output_path / 'hf_dataset'})",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help=f"Dataset split to evaluate (default: {DEFAULT_SPLIT})",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=DEFAULT_N,
        help=f"Maximum number of clips to evaluate (default: {DEFAULT_N})",
    )
    parser.add_argument(
        "--keyword",
        dest="keywords",
        action="append",
        default=[],
        metavar="WORD",
        help="Only evaluate clips whose ground truth contains WORD and report "
        "whether each model produced it. Repeat: --keyword Claude --keyword Cursor",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        metavar="MODEL_ID",
        help="Base model for comparison (default: read from the merged model's "
        "config, falling back to the configured finetune base model)",
    )
    parser.add_argument(
        "--compare-base",
        action="store_true",
        help="Also transcribe every clip with the base model for a side-by-side "
        "comparison and per-model WER",
    )
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Device for inference: an index like 0, or 'cpu'. "
            "Default picks GPU 0 when one is usable, otherwise CPU."
        ),
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=None,
        help="Compare base vs fine-tuned on a single audio file instead of a dataset split",
    )
    args = parser.parse_args()

    if not args.model.exists():
        print(
            f"ERROR: merged model not found at {args.model}\n"
            "Run:  listenr merge",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.audio:
        evaluate_single(args)
    else:
        evaluate_split(args)


if __name__ == "__main__":
    main()
