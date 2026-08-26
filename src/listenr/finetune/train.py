#!/usr/bin/env python3
"""
train.py — CLI entry point for Whisper LoRA fine-tuning.

Loads a HuggingFace DatasetDict produced by ``listenr build-dataset --format hf``,
prepares features, wraps the base Whisper model with LoRA adapters, and runs
``Seq2SeqTrainer``.  Only the adapter weights are saved, the base model is not
copied.

Usage:
    listenr finetune [options]

Examples:
    # Fine-tune on your listenr dataset with all defaults from config
    listenr finetune

    # Custom dataset location and output dir
    listenr finetune --dataset ~/listenr_dataset/hf_dataset --output ~/my_adapter

    # Quick smoke-test: load data + model, print stats, then exit
    listenr finetune --dry-run

    # Override training budget
    listenr finetune --max-steps 500 --eval-steps 100 --save-steps 200

    # AMD ROCm GPU: use bf16 instead of fp16
    listenr finetune --bf16
"""

import argparse
import logging
import sys
from pathlib import Path

from listenr.settings import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("listenr.finetune.train")


def _resolve_dataset_path(dataset_arg: Path | None, dataset_output: Path) -> Path:
    """Return the hf_dataset directory to load from.

    If *dataset_arg* is given explicitly, use it.  Otherwise fall back to
    ``<Dataset.output_path>/hf_dataset`` (the default location written by
    ``listenr build-dataset --format hf``).
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
            "Path to the hf_dataset directory written by listenr build-dataset "
            f"(default: {settings.dataset.output_path / 'hf_dataset'})"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.finetune.output_dir,
        metavar="DIR",
        help=f"Where to save the LoRA adapter checkpoint (default: {settings.finetune.output_dir})",
    )

    # --- model ---
    parser.add_argument(
        "--base-model",
        default=settings.finetune.base_model,
        metavar="MODEL_ID",
        help=(
            "HuggingFace model id to fine-tune - any openai/whisper-* or "
            "UsefulSensors/moonshine-* checkpoint "
            f"(default: {settings.finetune.base_model})"
        ),
    )
    parser.add_argument(
        "--language",
        default=settings.finetune.language,
        help=(
            "Target language for the processor; ignored by English-only models "
            f"such as Moonshine (default: {settings.finetune.language})"
        ),
    )
    parser.add_argument(
        "--task",
        default=settings.finetune.task,
        choices=["transcribe", "translate"],
        help=(
            "Task token prepended during tokenisation; ignored by English-only "
            f"models such as Moonshine (default: {settings.finetune.task})"
        ),
    )
    parser.add_argument(
        "--no-freeze-encoder",
        dest="freeze_encoder",
        action="store_false",
        default=settings.finetune.freeze_encoder,
        help="Train the encoder too (default: freeze it)",
    )

    # --- LoRA ---
    parser.add_argument("--lora-r",       type=int,   default=settings.finetune.lora_r,
                        help=f"LoRA rank (default: {settings.finetune.lora_r})")
    parser.add_argument("--lora-alpha",   type=int,   default=settings.finetune.lora_alpha,
                        help=f"LoRA scaling factor (default: {settings.finetune.lora_alpha})")
    parser.add_argument("--lora-dropout", type=float, default=settings.finetune.lora_dropout,
                        help=f"LoRA dropout (default: {settings.finetune.lora_dropout})")

    # --- training ---
    parser.add_argument("--max-steps",       type=int,   default=settings.finetune.max_steps,
                        help=f"Total training steps (default: {settings.finetune.max_steps})")
    parser.add_argument("--batch-size",      type=int,   default=settings.finetune.batch_size,
                        help=f"Per-device train batch size (default: {settings.finetune.batch_size})")
    parser.add_argument("--grad-accum",      type=int,   default=settings.finetune.grad_accum_steps,
                        help=f"Gradient accumulation steps (default: {settings.finetune.grad_accum_steps})")
    parser.add_argument("--learning-rate",   type=float, default=settings.finetune.learning_rate,
                        help=f"Learning rate (default: {settings.finetune.learning_rate})")
    parser.add_argument("--warmup-steps",    type=int,   default=settings.finetune.warmup_steps,
                        help=f"LR warmup steps (default: {settings.finetune.warmup_steps})")
    parser.add_argument("--eval-steps",      type=int,   default=settings.finetune.eval_steps,
                        help=f"Evaluate every N steps (default: {settings.finetune.eval_steps})")
    parser.add_argument("--save-steps",      type=int,   default=settings.finetune.save_steps,
                        help=f"Save checkpoint every N steps (default: {settings.finetune.save_steps})")
    # BooleanOptionalAction, not store_true: these default to the config file,
    # and store_true gives no way to turn off a value set there.
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=settings.finetune.fp16,
        help="fp16 mixed precision (CUDA GPUs; not recommended for AMD ROCm)",
    )
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=settings.finetune.bf16,
        help="bf16 mixed precision (recommended for AMD ROCm RDNA2+)",
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
            "  uv pip install \"listenr[finetune]\"",
            file=sys.stderr,
        )
        sys.exit(1)

    from listenr.finetune.architectures import UnsupportedArchitecture, detect
    from listenr.finetune.data import make_processor, make_dataset, SpeechDataCollator
    from listenr.finetune.preflight import (
        check_all,
        describe_accelerator,
        describe_accelerator_line,
        format_problems,
    )
    from listenr.finetune.model import (
        load_base_model,
        make_lora_config,
        apply_lora,
        freeze_encoder,
        count_trainable_params,
    )
    from listenr.finetune.metrics import make_compute_metrics

    # -----------------------------------------------------------------------
    # 0. Pre-flight
    # -----------------------------------------------------------------------
    # These are seconds of work that otherwise surface as a traceback minutes
    # in, after the dataset and the model have both loaded.
    accelerator = describe_accelerator()
    logger.info(describe_accelerator_line(accelerator))

    problems = check_all(
        fp16=args.fp16,
        bf16=args.bf16,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        accelerator=accelerator,
    )
    if problems:
        logger.warning("Pre-flight found %d problem(s):\n%s", len(problems), format_problems(problems))
    if any(p.severity == "error" for p in problems):
        logger.error("Refusing to start. Fix the errors above, or pass --dry-run to re-check.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # 1. Dataset
    # -----------------------------------------------------------------------
    dataset_path = _resolve_dataset_path(args.dataset, settings.dataset.output_path)
    if not dataset_path.exists():
        logger.error(
            f"Dataset not found at {dataset_path}.\n"
            "Run:  listenr build-dataset --format hf"
        )
        sys.exit(1)

    try:
        arch = detect(args.base_model)
    except UnsupportedArchitecture as exc:
        logger.error(str(exc))
        sys.exit(1)
    logger.info(f"Architecture: {arch.model_type}")

    logger.info(f"Loading dataset from {dataset_path}")
    processor = make_processor(args.base_model, args.language, args.task, arch)
    dataset = make_dataset(dataset_path, processor, arch.feature_key)
    logger.info(f"Dataset splits: { {k: len(v) for k, v in dataset.items()} }")

    # -----------------------------------------------------------------------
    # 2. Model + LoRA
    # -----------------------------------------------------------------------
    logger.info(f"Loading base model: {args.base_model}")
    model = load_base_model(args.base_model)

    # Set generation config to avoid deprecation warnings and force correct
    # tokens. English-only families (Moonshine) have no language/task tokens.
    if arch.supports_language_and_task:
        model.generation_config.language = args.language
        model.generation_config.task = args.task
        model.generation_config.forced_decoder_ids = None

    lora_cfg = make_lora_config(
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=settings.finetune.lora_target_modules,
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
        logger.info(
            "Dry run: dataset loaded, model built, pre-flight clean. "
            "Exiting without training."
        )
        return

    # -----------------------------------------------------------------------
    # 3. Training
    # -----------------------------------------------------------------------
    output_dir = Path(args.output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Record the invocation before training starts. trainer_state.json will
    # hold the metrics but says nothing about what was run; without this, an
    # adapter directory cannot tell you what produced it, and a crashed run
    # leaves no evidence of what was attempted.
    from listenr.finetune import report
    run_record = report.finetune_run_record(
        args=vars(args),
        base_model=args.base_model,
        architecture=arch.model_type,
        dataset=str(dataset_path),
        dataset_splits={k: len(v) for k, v in dataset.items()},
        trainable_params=trainable,
        total_params=total,
        accelerator=describe_accelerator_line(accelerator),
    )
    run_record_path = output_dir / "run.json"
    report.write_json(run_record_path, run_record)

    data_collator = SpeechDataCollator(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        feature_key=arch.feature_key,
        pad_features=arch.pad_features,
        pad_to_multiple=arch.pad_to_multiple,
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
        generation_max_length=settings.finetune.generation_max_length,
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

    run_record["ended_utc"] = report.utc_now()
    run_record["status"] = "completed"
    report.write_json(run_record_path, run_record)
    logger.info("Done.")


if __name__ == "__main__":
    main()
