"""
metrics.py — Evaluation metrics for Whisper fine-tuning.

Provides a factory that returns a ``compute_metrics`` closure compatible with
:class:`transformers.Seq2SeqTrainer`.

Requires the ``finetune`` optional dependencies::

    uv pip install -e ".[finetune]"

Public API
----------
make_compute_metrics(tokenizer) -> Callable[[EvalPrediction], dict]
"""

from __future__ import annotations

import sys
from typing import Any, Callable


def make_compute_metrics(tokenizer: Any) -> Callable:
    """Return a ``compute_metrics`` function for use with ``Seq2SeqTrainer``.

    The returned function computes Word Error Rate (WER) between the model's
    predicted token ids and the ground-truth label ids.

    Parameters
    ----------
    tokenizer:
        A ``WhisperTokenizer`` (or ``WhisperProcessor``) used to decode ids
        back to text strings.

    Example
    -------
    ::

        compute_metrics = make_compute_metrics(processor.tokenizer)
        trainer = Seq2SeqTrainer(..., compute_metrics=compute_metrics)
    """
    try:
        import evaluate
    except ImportError:
        print(
            "ERROR: evaluate is required. Install with:\n"
            "  uv pip install -e \".[finetune]\"",
            file=sys.stderr,
        )
        sys.exit(1)

    wer_metric = evaluate.load("wer")

    def compute_metrics(pred: Any) -> dict[str, float]:
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 (padding / ignored tokens) with the pad token id so the
        # tokenizer can decode them without errors.
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100.0 * wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    return compute_metrics
