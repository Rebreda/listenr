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
import os
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
        description="Fine-tune Whisper or Moonshine on your listenr dataset, with LoRA or in full."
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
        help=(
            "Where to save the LoRA adapter, or the whole model with --method full "
            f"(default: {settings.finetune.output_dir})"
        ),
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
            "Target language for the processor; ignored by Moonshine, which has "
            f"no language tokens (default: {settings.finetune.language})"
        ),
    )
    parser.add_argument(
        "--task",
        default=settings.finetune.task,
        choices=["transcribe", "translate"],
        help=(
            "Task token prepended during tokenisation; ignored by Moonshine, "
            f"which has no task tokens (default: {settings.finetune.task})"
        ),
    )
    parser.add_argument(
        "--no-freeze-encoder",
        dest="freeze_encoder",
        action="store_false",
        default=settings.finetune.freeze_encoder,
        help="Train the encoder too (default: freeze it)",
    )

    parser.add_argument(
        "--method",
        choices=["lora", "full"],
        default=settings.finetune.method,
        help=(
            "'lora' trains a small adapter to merge afterwards; 'full' updates "
            "every weight and saves a standalone model, with nothing to merge "
            f"(default: {settings.finetune.method})"
        ),
    )
    parser.add_argument(
        "--from-scratch",
        action=argparse.BooleanOptionalAction,
        default=settings.finetune.from_scratch,
        help=(
            "Use --base-model's architecture with randomly initialised weights. "
            "Requires --method full"
        ),
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
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=settings.finetune.gradient_checkpointing,
        help="Trade about a third more compute for less activation memory "
        f"(default: {settings.finetune.gradient_checkpointing})",
    )
    parser.add_argument(
        "--group-by-length",
        action=argparse.BooleanOptionalAction,
        default=settings.finetune.group_by_length,
        help="Batch clips of similar length to cut padding "
        f"(default: {settings.finetune.group_by_length})",
    )
    parser.add_argument("--eval-max-samples", type=int, default=settings.finetune.eval_max_samples,
                        help="Evaluate on at most N dev clips during training, a fixed random "
                        "subset; 0 = all (default: %(default)s)")
    parser.add_argument("--eval-batch-size", type=int, default=settings.finetune.eval_batch_size,
                        help="Batch size for evaluation; 0 = --batch-size (default: %(default)s)")
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=settings.finetune.pin_memory,
        help=f"Pin host memory for batches (default: {settings.finetune.pin_memory})",
    )
    parser.add_argument("--early-stopping-patience", type=int,
                        default=settings.finetune.early_stopping_patience,
                        help="Stop after this many evaluations without dev WER improving by more "
                        "than --early-stopping-threshold; 0 = never (default: %(default)s)")
    parser.add_argument("--early-stopping-threshold", type=float,
                        default=settings.finetune.early_stopping_threshold,
                        help="Smallest WER drop, in points, that counts as improving "
                        "(default: %(default)s)")
    parser.add_argument("--preprocess-workers", type=int, default=settings.finetune.preprocess_workers,
                        help="Processes for decoding audio before training; 0 = half the "
                        "CPU cores (default: %(default)s)")
    parser.add_argument("--dataloader-workers", type=int, default=settings.finetune.dataloader_workers,
                        help="Processes collating batches during training; 0 = none "
                        "(default: %(default)s)")
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

    if args.from_scratch and args.method != "full":
        parser.error("--from-scratch requires --method full: an adapter on random weights learns nothing useful")

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
    preprocess_workers = args.preprocess_workers or max(1, (os.cpu_count() or 2) // 2)
    dataset = make_dataset(dataset_path, processor, arch.feature_key, num_proc=preprocess_workers)
    logger.info(f"Dataset splits: { {k: len(v) for k, v in dataset.items()} }")

    # -----------------------------------------------------------------------
    # 2. Model (+ LoRA)
    # -----------------------------------------------------------------------
    if args.from_scratch:
        logger.info(f"Initialising {args.base_model}'s architecture with random weights")
    else:
        logger.info(f"Loading base model: {args.base_model}")
    model = load_base_model(args.base_model, from_scratch=args.from_scratch)

    # Set generation config to avoid deprecation warnings and force correct
    # tokens. Moonshine has no language/task tokens.
    if arch.supports_language_and_task:
        model.generation_config.language = args.language
        model.generation_config.task = args.task
        model.generation_config.forced_decoder_ids = None

    if args.method == "lora":
        lora_cfg = make_lora_config(
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=settings.finetune.lora_target_modules,
        )
        model = apply_lora(model, lora_cfg)

    if args.freeze_encoder and args.from_scratch:
        # A frozen random encoder would hand the decoder noise for the whole run.
        logger.info("Training from scratch: the encoder is trained too.")
    elif args.freeze_encoder:
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
        gradient_checkpointing=args.gradient_checkpointing,
        train_sampling_strategy="group_by_length" if args.group_by_length else "random",
        length_column_name="length",
        dataloader_pin_memory=args.pin_memory,
        dataloader_num_workers=args.dataloader_workers,
        dataloader_persistent_workers=args.dataloader_workers > 0,
        fp16=args.fp16,
        bf16=args.bf16,
        eval_strategy="steps",
        per_device_eval_batch_size=args.eval_batch_size or args.batch_size,
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
    if eval_split is not None and args.eval_max_samples and len(eval_split) > args.eval_max_samples:
        eval_split = eval_split.shuffle(seed=42).select(range(args.eval_max_samples))
        logger.info(f"Evaluating on a fixed {args.eval_max_samples}-clip subset of dev during training")

    if train_split is None:
        logger.error("Dataset has no 'train' split.")
        sys.exit(1)
    if eval_split is None:
        logger.warning("No 'dev' or 'test' split found — evaluation will be skipped.")

    callbacks = []
    if args.early_stopping_patience > 0:
        from transformers import EarlyStoppingCallback
        # The best checkpoint is still the one kept: load_best_model_at_end.
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold,
        ))
        logger.info(
            f"Early stopping after {args.early_stopping_patience} evaluations without "
            f"dev WER improving by more than {args.early_stopping_threshold} points"
        )

    trainer = Seq2SeqTrainer(
        args=training_args,
        callbacks=callbacks,
        model=model,
        train_dataset=train_split,
        eval_dataset=eval_split,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    logger.info("Starting training…")
    trainer.train()

    # LoRA saves only the adapter weights; full saves a standalone model.
    if args.method == "lora":
        logger.info(f"Saving LoRA adapter to {output_dir}")
    else:
        logger.info(f"Saving model to {output_dir} (standalone: no merge needed)")
    model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))

    run_record["ended_utc"] = report.utc_now()
    run_record["status"] = "completed"
    report.write_json(run_record_path, run_record)
    logger.info("Done.")


if __name__ == "__main__":
    main()
