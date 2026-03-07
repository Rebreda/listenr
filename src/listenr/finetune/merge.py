#!/usr/bin/env python3
"""
merge.py — Merge a LoRA adapter into the base Whisper model.

After fine-tuning with ``listenr-finetune``, the output directory contains
only the compact LoRA adapter weights.  This script folds those weights
permanently into the base model and saves a self-contained
``WhisperForConditionalGeneration`` that can be loaded anywhere without PEFT
installed.

Usage:
    listenr-merge [options]

Examples:
    # Merge adapter from default location (~/listenr_finetune) into ~/listenr_merged
    listenr-merge

    # Custom paths
    listenr-merge --adapter ~/my_adapter --output ~/my_merged_model

    # Preview: show what would happen without writing files
    listenr-merge --dry-run
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from listenr.constants import FINETUNE_OUTPUT_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("listenr.finetune.merge")

# Default merged-model output sits beside the adapter directory.
DEFAULT_ADAPTER_DIR = FINETUNE_OUTPUT_DIR
DEFAULT_OUTPUT_DIR = FINETUNE_OUTPUT_DIR.parent / (FINETUNE_OUTPUT_DIR.name + "_merged")


# ---------------------------------------------------------------------------
# Core merge logic (importable for testing)
# ---------------------------------------------------------------------------

def read_base_model_id(adapter_dir: Path) -> str:
    """Return the base model ID stored in *adapter_dir*/adapter_config.json.

    Raises
    ------
    FileNotFoundError
        If adapter_config.json does not exist in *adapter_dir*.
    KeyError
        If the JSON does not contain a ``base_model_name_or_path`` key.
    """
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"adapter_config.json not found in {adapter_dir}. "
            "Make sure --adapter points to the directory produced by listenr-finetune."
        )
    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)
    model_id = data.get("base_model_name_or_path")
    if not model_id:
        raise KeyError(
            f"'base_model_name_or_path' key missing from {config_path}. "
            "The adapter_config.json may be malformed."
        )
    return model_id


def merge_adapter(adapter_dir: Path, output_dir: Path, dry_run: bool = False) -> None:
    """Load a PEFT LoRA adapter, merge it into the base model, and save.

    The output directory will contain a standalone ``WhisperForConditionalGeneration``
    (in safetensors format) plus the processor artifacts copied from ``adapter_dir``
    so the directory is fully self-contained.

    Parameters
    ----------
    adapter_dir:
        Path to the directory produced by ``listenr-finetune`` (contains
        ``adapter_model.safetensors`` and ``adapter_config.json``).
    output_dir:
        Destination for the merged model.  Created if absent.
    dry_run:
        If True, perform all validation steps but do not write any files.
    """
    try:
        from peft import PeftModel
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
    except ImportError:
        logger.error(
            "transformers and peft are required. Install with:\n"
            "  uv pip install -e '.[finetune]'"
        )
        sys.exit(1)

    adapter_dir = adapter_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

    # ── validate inputs ───────────────────────────────────────────────────────
    if not adapter_dir.exists():
        logger.error(f"Adapter directory not found: {adapter_dir}")
        sys.exit(1)

    base_model_id = read_base_model_id(adapter_dir)
    logger.info(f"Adapter: {adapter_dir}")
    logger.info(f"Base model: {base_model_id}")
    logger.info(f"Output: {output_dir}")

    if dry_run:
        logger.info("Dry run — no files written.")
        return

    # ── load ──────────────────────────────────────────────────────────────────
    logger.info(f"Loading base model '{base_model_id}' ...")
    base_model = WhisperForConditionalGeneration.from_pretrained(
        base_model_id,
        device_map="cpu",
    )

    logger.info("Loading LoRA adapter ...")
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))

    # ── merge & unload ────────────────────────────────────────────────────────
    # merge_and_unload() folds the LoRA delta (α/r · BA) into each adapted
    # weight matrix and returns a plain WhisperForConditionalGeneration with
    # no remaining PEFT structure.
    logger.info("Merging LoRA weights into base model ...")
    merged_model = peft_model.merge_and_unload()

    # ── save model ────────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving merged model to {output_dir} ...")
    merged_model.save_pretrained(str(output_dir), safe_serialization=True)

    # ── save processor / tokenizer ────────────────────────────────────────────
    # Re-save the processor so the output directory is fully self-contained
    # and can be loaded with from_pretrained() without the adapter directory.
    logger.info("Saving processor ...")
    try:
        processor = WhisperProcessor.from_pretrained(str(adapter_dir))
        processor.save_pretrained(str(output_dir))
    except Exception as exc:  # noqa: BLE001
        # Processor save is best-effort; a missing tokenizer file is not fatal.
        logger.warning(f"Could not save processor ({exc}); skipping.")

    _print_summary(output_dir, base_model_id)


def _print_summary(output_dir: Path, base_model_id: str) -> None:
    """Print a short summary of what was written."""
    files = sorted(output_dir.iterdir())
    total_bytes = sum(f.stat().st_size for f in files if f.is_file())
    print("\n----------- Merge Complete -----------")
    print(f"  Base model : {base_model_id}")
    print(f"  Output     : {output_dir}")
    print(f"  Files      : {len(files)}")
    print(f"  Total size : {total_bytes / 1_048_576:.1f} MB")
    print()
    print("  Load with:")
    print("    from transformers import WhisperForConditionalGeneration, WhisperProcessor")
    print(f"    model = WhisperForConditionalGeneration.from_pretrained('{output_dir}')")
    print(f"    processor = WhisperProcessor.from_pretrained('{output_dir}')")
    print("--------------------------------------\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge a LoRA adapter into its base Whisper model, producing a "
            "self-contained model directory usable without PEFT."
        )
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=DEFAULT_ADAPTER_DIR,
        help=(
            f"Path to the adapter directory produced by listenr-finetune "
            f"(default: {DEFAULT_ADAPTER_DIR})"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            f"Destination directory for the merged model "
            f"(default: {DEFAULT_OUTPUT_DIR})"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print what would happen without writing files.",
    )
    args = parser.parse_args()

    merge_adapter(
        adapter_dir=args.adapter,
        output_dir=args.output,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
