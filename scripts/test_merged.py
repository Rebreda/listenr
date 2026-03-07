#!/usr/bin/env python3
"""
test_merged.py — Transcribe listenr audio clips with the merged fine-tuned model
and compare to the original Whisper / LLM-corrected transcriptions stored in
manifest.jsonl.

Usage:
    # Use default paths (~/listenr_merged, ~/.listenr/audio_clips/manifest.jsonl)
    uv run python scripts/test_merged.py

    # Custom model or manifest path
    uv run python scripts/test_merged.py --model ~/listenr_merged --manifest ~/.listenr/audio_clips/manifest.jsonl

    # Limit to 10 clips, skip very short ones
    uv run python scripts/test_merged.py --n 10 --min-duration 1.0

    # Point at a single audio file instead of the manifest
    uv run python scripts/test_merged.py --audio path/to/clip.wav

Requires: transformers, soundfile, torch
    uv pip install -e ".[finetune]"
"""

import argparse
import json
import logging
import sys
import textwrap
import warnings
from pathlib import Path

# Silence noisy well-known Whisper quirks (attention_mask, logits-processor
# duplication) that don't affect transcription quality.
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*attention_mask.*")
warnings.filterwarnings("ignore", message=".*logits_process.*")

try:
    # transformers has its own logging wrapper; suppress at that level too.
    from transformers.utils import logging as _hf_logging
    _hf_logging.set_verbosity_error()
except ImportError:
    pass  # transformers not yet imported; will be silenced at model-load time


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL    = Path("~/listenr_merged").expanduser()
DEFAULT_MANIFEST = Path("~/.listenr/audio_clips/manifest.jsonl").expanduser()
DEFAULT_N        = 20
DEFAULT_MIN_DUR  = 0.5   # seconds — skip sub-500ms clips (often noise)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(model_path: Path):
    """Load the merged WhisperForConditionalGeneration and processor."""
    try:
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
    except ImportError:
        print("ERROR: transformers and torch required.\n"
              "  uv pip install -e '.[finetune]'", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model from {model_path} ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperForConditionalGeneration.from_pretrained(str(model_path))
    model = model.to(device)
    model.eval()
    processor = WhisperProcessor.from_pretrained(str(model_path))
    print(f"  device : {device}")
    print(f"  params : {sum(p.numel() for p in model.parameters()):,}")
    return model, processor, device


def transcribe(audio_path: Path, model, processor, device: str) -> str:
    """Return the model's transcription of a single WAV file."""
    import torch
    import soundfile as sf

    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)   # stereo → mono

    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            language="english",
            task="transcribe",
        )

    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()


def load_manifest_entries(
    manifest_path: Path,
    n: int,
    min_duration: float,
    keywords: list[str] | None = None,
) -> list[dict]:
    """Return up to *n* valid manifest entries with existing audio files.

    If *keywords* is given, only entries whose corrected_transcription contains
    at least one keyword are returned.  These are the clips where the ground
    truth target has the word — the test is whether the fine-tuned model now
    produces it.
    """
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    kw_lower = [k.lower() for k in keywords] if keywords else []

    decoder = json.JSONDecoder()
    entries = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Use raw_decode so that multiple JSON objects concatenated on the
            # same line (a manifest write bug) are each parsed individually.
            pos = 0
            while pos < len(line):
                # Skip whitespace between objects
                while pos < len(line) and line[pos] in " \t\r\n":
                    pos += 1
                if pos >= len(line):
                    break
                try:
                    rec, pos = decoder.raw_decode(line, pos)
                except json.JSONDecodeError:
                    break
                audio = Path(rec.get("audio_path", ""))
                if not audio.exists():
                    continue
                if float(rec.get("duration_s") or 0) < min_duration:
                    continue
                if not rec.get("raw_transcription"):
                    continue
                if kw_lower:
                    # keyword must appear in corrected_transcription (the target)
                    corrected = (rec.get("corrected_transcription") or "").lower()
                    if not any(kw in corrected for kw in kw_lower):
                        continue
                entries.append(rec)

    # Most recent first
    entries.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return entries[:n]


def _col(text: str, width: int) -> str:
    """Wrap *text* to *width* columns."""
    return "\n".join(textwrap.wrap(text or "(empty)", width)) if text else "(empty)"


def keyword_hits(text: str, keywords: list[str]) -> list[str]:
    """Return the subset of *keywords* found (case-insensitive) in *text*."""
    t = text.lower()
    return [kw for kw in keywords if kw.lower() in t]


def print_result(
    idx: int,
    rec: dict | None,
    merged_text: str,
    audio_path: Path,
    keywords: list[str] | None = None,
) -> dict[str, bool]:
    """Print one result row.  Returns a {keyword: hit} map for tallying.

    When keywords are active the layout is:
      RAW (Whisper)          — what the base model produced (likely wrong)
      FINE-TUNED (merged)    — what the new model produces
      CORRECTED (ground truth) — the LLM-corrected target
    Hit = keyword found in merged output.  Expected = keyword in corrected.
    """
    raw       = rec.get("raw_transcription", "")       if rec else ""
    corrected = rec.get("corrected_transcription", "")  if rec else ""
    duration  = rec.get("duration_s", 0)               if rec else 0
    whisper   = rec.get("whisper_model", "")            if rec else ""

    w = 40
    print(f"\n{'─' * 90}")
    print(f"  Clip {idx:>2}  {audio_path.name}  ({duration:.1f}s  {whisper})")
    print(f"{'─' * 90}")

    if keywords:
        # Three-row layout: raw | merged | corrected (ground truth)
        print(f"  {'RAW (Whisper)':<{w}}  {'FINE-TUNED (merged)':<{w}}")
        print(f"  {_col(raw, w):<{w}}  {_col(merged_text, w):<{w}}")
        print(f"\n  CORRECTED (ground truth)")
        print(f"  {_col(corrected, w)}")
    else:
        print(f"  {'ORIGINAL (Whisper)':<{w}}  {'FINE-TUNED (merged)':<{w}}")
        print(f"  {_col(raw, w):<{w}}  {_col(merged_text, w):<{w}}")
        if corrected and corrected != raw:
            print(f"\n  LLM-corrected ground truth")
            print(f"  {_col(corrected, w)}")

    hit_map: dict[str, bool] = {}
    if keywords:
        # expected = keywords present in corrected_transcription (the target)
        expected = keyword_hits(corrected, keywords)
        found    = keyword_hits(merged_text, keywords)
        hit_map  = {kw: (kw.lower() in [f.lower() for f in found]) for kw in expected}
        hits   = [kw for kw, ok in hit_map.items() if ok]
        misses = [kw for kw, ok in hit_map.items() if not ok]
        parts  = []
        if hits:
            parts.append("HIT:  " + ", ".join(hits))
        if misses:
            parts.append("MISS: " + ", ".join(misses))
        if parts:
            print(f"\n  Keywords — " + "   ".join(parts))

    return hit_map


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test the merged fine-tuned Whisper model on listenr audio clips."
    )
    parser.add_argument("--model",    type=Path, default=DEFAULT_MODEL,
                        help=f"Path to merged model directory (default: {DEFAULT_MODEL})")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help=f"Path to manifest.jsonl (default: {DEFAULT_MANIFEST})")
    parser.add_argument("--audio",    type=Path, default=None,
                        help="Transcribe a single audio file instead of manifest clips")
    parser.add_argument("--n",        type=int, default=DEFAULT_N,
                        help=f"Number of clips to test (default: {DEFAULT_N})")
    parser.add_argument("--min-duration", type=float, default=DEFAULT_MIN_DUR,
                        help=f"Skip clips shorter than this (default: {DEFAULT_MIN_DUR}s)")
    parser.add_argument("--keyword", dest="keywords", action="append", default=[],
                        metavar="WORD",
                        help="Only test clips where WORD appears in corrected_transcription "
                             "(ground truth). Shows raw Whisper vs fine-tuned output and "
                             "whether the new model now produces the keyword. "
                             "Repeat: --keyword Claude --keyword Cursor")
    args = parser.parse_args()

    if not args.model.exists():
        print(f"ERROR: model directory not found: {args.model}\n"
              "Run 'podman compose run --rm merge' first.", file=sys.stderr)
        sys.exit(1)

    model, processor, device = load_model(args.model)

    # ── single file mode ──────────────────────────────────────────────────────
    if args.audio:
        if not args.audio.exists():
            print(f"ERROR: audio file not found: {args.audio}", file=sys.stderr)
            sys.exit(1)
        print(f"\nTranscribing {args.audio} ...")
        text = transcribe(args.audio, model, processor, device)
        print(f"\n  {text}")
        return

    # ── manifest mode ─────────────────────────────────────────────────────────
    keywords = args.keywords or []

    entries = load_manifest_entries(
        args.manifest, args.n, args.min_duration,
        keywords or None,
    )
    if not entries:
        kw_note = f" with '{', '.join(keywords)}' in corrected_transcription" if keywords else ""
        print(f"No valid entries found in manifest{kw_note}.", file=sys.stderr)
        sys.exit(1)

    print(f"\nTesting {len(entries)} clips from {args.manifest}")
    if keywords:
        print(f"  keywords : {', '.join(keywords)}")
        print(f"  filter   : keyword present in corrected_transcription (ground truth)")
        print(f"  question : does the fine-tuned model now produce the keyword?")
    print()

    total = len(entries)
    # tally[keyword] = [hits, total_expected]
    tally: dict[str, list[int]] = {kw: [0, 0] for kw in keywords}

    for i, rec in enumerate(entries, 1):
        audio_path = Path(rec["audio_path"])
        print(f"  [{i}/{total}] {audio_path.name} ...", end="", flush=True)
        merged_text = transcribe(audio_path, model, processor, device)
        print(f"\r", end="")
        hit_map = print_result(i, rec, merged_text, audio_path, keywords or None)
        for kw, hit in hit_map.items():
            tally[kw][1] += 1
            if hit:
                tally[kw][0] += 1

    print(f"\n{'─' * 90}")
    print(f"  Done — {total} clips transcribed with {args.model.name}")
    if tally:
        print(f"\n  Keyword recall")
        for kw, (hits, total_kw) in tally.items():
            pct = 100 * hits / total_kw if total_kw else 0
            bar = ("█" * hits) + ("░" * (total_kw - hits))
            print(f"    {kw:<20}  {hits}/{total_kw}  ({pct:.0f}%)  {bar}")
    print(f"{'─' * 90}\n")


if __name__ == "__main__":
    main()
