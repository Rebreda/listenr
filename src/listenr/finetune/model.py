"""
model.py — Model loading and LoRA configuration for Whisper fine-tuning.

All functions are pure (no global state) to make them easy to test in
isolation.

Requires the ``finetune`` optional dependencies::

    uv pip install "listenr[finetune]"

Public API
----------
load_base_model(model_id)                      -> WhisperForConditionalGeneration
make_lora_config(r, alpha, dropout, modules)   -> LoraConfig
apply_lora(model, lora_config)                 -> PeftModel
freeze_encoder(model)                          -> None  (in-place)
count_trainable_params(model)                  -> (trainable, total)
"""

from __future__ import annotations

import sys
from typing import Any


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_base_model(model_id: str) -> Any:
    """Load the pre-trained Whisper model from *model_id*.

    The model is returned in full precision (fp32).  The caller is responsible
    for casting to fp16/bf16 via ``Seq2SeqTrainingArguments`` at training time.
    """
    try:
        from transformers import WhisperForConditionalGeneration
    except ImportError:
        print(
            "ERROR: transformers is required. Install with:\n"
            "  uv pip install \"listenr[finetune]\"",
            file=sys.stderr,
        )
        sys.exit(1)

    return WhisperForConditionalGeneration.from_pretrained(model_id)


# ---------------------------------------------------------------------------
# LoRA configuration
# ---------------------------------------------------------------------------

def make_lora_config(
    r: int,
    alpha: int,
    dropout: float,
    target_modules: list[str],
) -> Any:
    """Build a PEFT :class:`LoraConfig` for a Whisper Seq2Seq model.

    Parameters
    ----------
    r:
        LoRA rank.  Higher values → more expressiveness but more parameters.
        ``8`` is a good default for domain adaptation.
    alpha:
        LoRA scaling factor.  Effective scale = ``alpha / r``; keeping
        ``alpha = 4 * r`` is a common heuristic.
    dropout:
        Dropout applied to the LoRA layers during training.
    target_modules:
        Which attention projections to adapt.  Whisper decoder attention has
        ``q_proj``, ``k_proj``, ``v_proj``, ``out_proj``; adapting just
        ``q_proj`` and ``v_proj`` is the minimal effective set.
    """
    try:
        from peft import LoraConfig, TaskType
    except ImportError:
        print(
            "ERROR: peft is required. Install with:\n"
            "  uv pip install \"listenr[finetune]\"",
            file=sys.stderr,
        )
        sys.exit(1)

    return LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
    )


# ---------------------------------------------------------------------------
# Apply LoRA
# ---------------------------------------------------------------------------

def apply_lora(model: Any, lora_config: Any) -> Any:
    """Wrap *model* with PEFT LoRA adapters defined by *lora_config*.

    Returns the wrapped :class:`peft.PeftModel`.  Only the LoRA adapter
    parameters will have ``requires_grad=True``; all other weights are frozen
    by PEFT automatically.
    """
    try:
        from peft import get_peft_model
    except ImportError:
        print(
            "ERROR: peft is required. Install with:\n"
            "  uv pip install \"listenr[finetune]\"",
            file=sys.stderr,
        )
        sys.exit(1)

    return get_peft_model(model, lora_config)


# ---------------------------------------------------------------------------
# Encoder freezing
# ---------------------------------------------------------------------------

def freeze_encoder(model: Any) -> None:
    """Freeze all parameters in the Whisper encoder (in-place).

    The encoder already extracts high-quality audio features from pre-training;
    only the decoder needs domain adaptation for a new vocabulary/style.

    Works both on a raw ``WhisperForConditionalGeneration`` and on a
    ``PeftModel`` wrapping one.
    """
    # PeftModel wraps the base model under .base_model.model; raw Whisper has
    # .model.encoder directly.
    try:
        encoder = model.base_model.model.model.encoder
    except AttributeError:
        encoder = model.model.encoder

    for param in encoder.parameters():
        param.requires_grad = False


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

def count_trainable_params(model: Any) -> tuple[int, int]:
    """Return ``(trainable_params, total_params)`` for *model*.

    Useful for logging::

        trainable, total = count_trainable_params(model)
        print(f"Training {trainable:,} / {total:,} params ({100*trainable/total:.2f}%)")
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
