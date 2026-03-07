#!/usr/bin/env python3
"""
train.py — CLI entry point for Whisper LoRA fine-tuning.

Loads a HuggingFace DatasetDict produced by ``listenr-build-dataset --format hf``,
prepares features, wraps the base Whisper model with LoRA adapters, and runs
``Seq2SeqTrainer``.  Only the adapter weights are saved — the base model is not
copied.

Usage:
    listenr-finetune [options]

Examples:
    # Fine-tune on your listenr dataset with all defaults from config
    listenr-finetune

    # Custom dataset location and output dir
    listenr-finetune --dataset ~/listenr_dataset/hf_dataset --output ~/my_adapter

    # Quick smoke-test: load data + model, print stats, then exit
    listenr-finetune --dry-run

    # Override training budget
    listenr-finetune --max-steps 500 --eval-steps 100 --save-steps 200

    # AMD ROCm GPU: use bf16 instead of fp16
    listenr-finetune --bf16
"""

import argparse
import logging
import sys
from pathlib import Path

from listenr.constants import (
    DATASET_OUTPUT,
    FINETUNE_BASE_MODEL,
    FINETUNE_BATCH_SIZE,
    FINETUNE_BF16,
    FINETUNE_EVAL_STEPS,
    FINETUNE_FP16,
    FINETUNE_FREEZE_ENCODER,
    FINETUNE_GENERATION_MAX_LENGTH,
    FINETUNE_GRAD_ACCUM_STEPS,
    FINETUNE_LANGUAGE,
    FINETUNE_LEARNING_RATE,
    FINETUNE_LORA_ALPHA,
    FINETUNE_LORA_DROPOUT,
    FINETUNE_LORA_R,
    FINETUNE_LORA_TARGET_MODULES,
    FINETUNE_MAX_STEPS,
    FINETUNE_OUTPUT_DIR,
    FINETUNE_SAVE_STEPS,
    FINETUNE_TASK,
    FINETUNE_WARMUP_STEPS,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("listenr.finetune.train")


def _resolve_dataset_path(dataset_arg: Path | None, dataset_output: Path) -> Path:
    """Return the hf_dataset directory to load from.

    If *dataset_arg* is given explicitly, use it.  Otherwise fall back to
    ``<Dataset.output_path>/hf_dataset`` (the default location written by
    ``listenr-build-dataset --format hf``).
    """
    if dataset_arg is not None:
        return Path(dataset_arg).expanduser()
    return dataset_output / "hf_dataset"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune Whisper with LoRA adapters on your listenr dataset."
    )

    # --- data ---
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Path to the hf_dataset directory written by listenr-build-dataset "
            f"(default: {DATASET_OUTPUT / 'hf_dataset'})"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=FINETUNE_OUTPUT_DIR,
        metavar="DIR",
        help=f"Where to save the LoRA adapter checkpoint (default: {FINETUNE_OUTPUT_DIR})",
    )

    # --- model ---
    parser.add_argument(
        "--base-model",
        default=FINETUNE_BASE_MODEL,
        metavar="MODEL_ID",
        help=f"HuggingFace model id to fine-tune (default: {FINETUNE_BASE_MODEL})",
    )
    parser.add_argument(
        "--language",
        default=FINETUNE_LANGUAGE,
        help=f"Target language for the processor (default: {FINETUNE_LANGUAGE})",
    )
    parser.add_argument(
        "--task",
        default=FINETUNE_TASK,
        choices=["transcribe", "translate"],
        help=f"Task token prepended during tokenisation (default: {FINETUNE_TASK})",
    )
    parser.add_argument(
        "--no-freeze-encoder",
        dest="freeze_encoder",
        action="store_false",
        default=FINETUNE_FREEZE_ENCODER,
        help="Train the encoder too (default: freeze it)",
    )

    # --- LoRA ---
    parser.add_argument("--lora-r",       type=int,   default=FINETUNE_LORA_R,
                        help=f"LoRA rank (default: {FINETUNE_LORA_R})")
    parser.add_argument("--lora-alpha",   type=int,   default=FINETUNE_LORA_ALPHA,
                        help=f"LoRA scaling factor (default: {FINETUNE_LORA_ALPHA})")
    parser.add_argument("--lora-dropout", type=float, default=FINETUNE_LORA_DROPOUT,
                        help=f"LoRA dropout (default: {FINETUNE_LORA_DROPOUT})")

    # --- training ---
    parser.add_argument("--max-steps",       type=int,   default=FINETUNE_MAX_STEPS,
                        help=f"Total training steps (default: {FINETUNE_MAX_STEPS})")
    parser.add_argument("--batch-size",      type=int,   default=FINETUNE_BATCH_SIZE,
                        help=f"Per-device train batch size (default: {FINETUNE_BATCH_SIZE})")
    parser.add_argument("--grad-accum",      type=int,   default=FINETUNE_GRAD_ACCUM_STEPS,
                        help=f"Gradient accumulation steps (default: {FINETUNE_GRAD_ACCUM_STEPS})")
    parser.add_argument("--learning-rate",   type=float, default=FINETUNE_LEARNING_RATE,
                        help=f"Learning rate (default: {FINETUNE_LEARNING_RATE})")
    parser.add_argument("--warmup-steps",    type=int,   default=FINETUNE_WARMUP_STEPS,
                        help=f"LR warmup steps (default: {FINETUNE_WARMUP_STEPS})")
    parser.add_argument("--eval-steps",      type=int,   default=FINETUNE_EVAL_STEPS,
                        help=f"Evaluate every N steps (default: {FINETUNE_EVAL_STEPS})")
    parser.add_argument("--save-steps",      type=int,   default=FINETUNE_SAVE_STEPS,
                        help=f"Save checkpoint every N steps (default: {FINETUNE_SAVE_STEPS})")
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=FINETUNE_FP16,
        help="Enable fp16 mixed precision (CUDA GPUs; not recommended for AMD ROCm)",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=FINETUNE_BF16,
        help="Enable bf16 mixed precision (recommended for AMD ROCm RDNA2+)",
    )
    parser.add_argument(
        "--report-to",
        default="tensorboard",
        metavar="BACKEND",
        help="Reporting backend(s) for the Trainer, e.g. 'tensorboard', 'wandb', or 'none' (default: tensorboard)",
    )

    # --- misc ---
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load data + model, print stats, then exit without training",
    )

    args = parser.parse_args()

    # Deferred imports — only required when actually running, so the package is
    # importable without the finetune extras installed.
    try:
        from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
    except ImportError:
        print(
            "ERROR: transformers is required. Install with:\n"
            "  uv pip install -e \".[finetune]\"",
            file=sys.stderr,
        )
        sys.exit(1)

    from listenr.finetune.data import make_processor, make_dataset, WhisperDataCollator
    from listenr.finetune.model import (
        load_base_model,
        make_lora_config,
        apply_lora,
        freeze_encoder,
        count_trainable_params,
    )
    from listenr.finetune.metrics import make_compute_metrics

    # -----------------------------------------------------------------------
    # 1. Dataset
    # -----------------------------------------------------------------------
    dataset_path = _resolve_dataset_path(args.dataset, DATASET_OUTPUT)
    if not dataset_path.exists():
        logger.error(
            f"Dataset not found at {dataset_path}.\n"
            "Run:  listenr-build-dataset --format hf"
        )
        sys.exit(1)

    logger.info(f"Loading dataset from {dataset_path}")
    processor = make_processor(args.base_model, args.language, args.task)
    dataset = make_dataset(dataset_path, processor)
    logger.info(f"Dataset splits: { {k: len(v) for k, v in dataset.items()} }")

    # -----------------------------------------------------------------------
    # 2. Model + LoRA
    # -----------------------------------------------------------------------
    logger.info(f"Loading base model: {args.base_model}")
    model = load_base_model(args.base_model)

    # Set generation config to avoid deprecation warnings and force correct tokens.
    model.generation_config.language = args.language
    model.generation_config.task = args.task
    model.generation_config.forced_decoder_ids = None

    lora_cfg = make_lora_config(
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=FINETUNE_LORA_TARGET_MODULES,
    )
    model = apply_lora(model, lora_cfg)

    if args.freeze_encoder:
        freeze_encoder(model)
        logger.info("Encoder frozen.")

    trainable, total = count_trainable_params(model)
    logger.info(
        f"Trainable params: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)"
    )

    if args.dry_run:
        logger.info("Dry run — exiting without training.")
        return

    # -----------------------------------------------------------------------
    # 3. Training
    # -----------------------------------------------------------------------
    output_dir = Path(args.output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    data_collator = WhisperDataCollator(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    compute_metrics = make_compute_metrics(processor.tokenizer)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        gradient_checkpointing=True,
        fp16=args.fp16,
        bf16=args.bf16,
        eval_strategy="steps",
        per_device_eval_batch_size=max(1, args.batch_size // 2),
        predict_with_generate=True,
        generation_max_length=FINETUNE_GENERATION_MAX_LENGTH,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=max(1, args.eval_steps // 4),
        report_to=args.report_to,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        remove_unused_columns=False,  # keep audio columns until collator runs
    )

    train_split = dataset.get("train")
    eval_split = dataset.get("dev") or dataset.get("test")

    if train_split is None:
        logger.error("Dataset has no 'train' split.")
        sys.exit(1)
    if eval_split is None:
        logger.warning("No 'dev' or 'test' split found — evaluation will be skipped.")

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_split,
        eval_dataset=eval_split,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    logger.info("Starting training…")
    trainer.train()

    # Save only the adapter weights.
    logger.info(f"Saving LoRA adapter to {output_dir}")
    model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))
    logger.info("Done.")


if __name__ == "__main__":
    main()
