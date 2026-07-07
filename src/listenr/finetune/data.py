"""
data.py — Dataset preparation for Whisper LoRA fine-tuning.

All functions are pure (no global state, no side effects) to keep them
easy to test and reuse.

Requires the ``finetune`` optional dependencies::

    uv pip install -e ".[finetune]"

Public API
----------
make_processor(model_id, language, task)  -> WhisperProcessor
prepare_example(batch, processor)         -> dict with input_features + labels
make_dataset(hf_dataset_path, processor)  -> DatasetDict with train/dev/test
WhisperDataCollator                       -> dataclass, handles per-batch padding
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import WhisperProcessor  # noqa: F401


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

def make_processor(model_id: str, language: str, task: str) -> "WhisperProcessor":
    """Load the WhisperProcessor (feature extractor + tokenizer) from *model_id*.

    Parameters
    ----------
    model_id:
        HuggingFace Hub identifier, e.g. ``"openai/whisper-small"``.
    language:
        Target language, e.g. ``"english"``.  Passed to the tokenizer so the
        correct language token is prepended during encoding.
    task:
        ``"transcribe"`` (default) or ``"translate"``.
    """
    try:
        from transformers import WhisperProcessor as _WhisperProcessor
    except ImportError:
        print(
            "ERROR: transformers is required. Install with:\n"
            "  uv pip install -e \".[finetune]\"",
            file=sys.stderr,
        )
        sys.exit(1)

    return _WhisperProcessor.from_pretrained(model_id, language=language, task=task)


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

DEFAULT_TARGET_SAMPLE_RATE = 16_000


def _resample(array: Any, orig_sr: int, target_sr: int) -> Any:
    """Resample a mono waveform to *target_sr* (no-op when already matching).

    Whisper's feature extractor assumes a fixed sample rate (16 kHz) and does
    not resample. Imported datasets are frequently at other rates (e.g. MDC
    Common Voice clips are 32 kHz), so we resample here before feature
    extraction; otherwise the log-Mel features are computed against the wrong
    rate and training silently degrades.
    """
    if int(orig_sr) == int(target_sr):
        return array
    from math import gcd

    from scipy.signal import resample_poly

    divisor = gcd(int(orig_sr), int(target_sr))
    up = int(target_sr) // divisor
    down = int(orig_sr) // divisor
    return resample_poly(array, up, down).astype("float32")


def prepare_example(batch: dict, processor: Any) -> dict:
    """Convert a single dataset example into model-ready tensors.

    Expects the dataset to have been created by ``listenr build-dataset --format hf``
    (or ``both``).  ``audio_path`` may be either:

    * A plain file-path string — loaded on-the-fly with ``soundfile``.
    * A decoded HuggingFace :class:`datasets.Audio` dict with keys
      ``array`` and ``sampling_rate`` — used directly (legacy / test usage).

    Audio is resampled to the feature extractor's expected rate (16 kHz) so
    that clips imported at other sample rates train correctly.

    Returns a dict with:

    ``input_features``
        Log-Mel spectrogram, shape ``(80, 3000)``, as produced by the Whisper
        feature extractor.
    ``labels``
        Token ids for ``corrected_transcription``.
    """
    audio = batch["audio_path"]

    # Handle plain path string — load with soundfile to avoid the torchcodec
    # requirement introduced in datasets 4+.
    if isinstance(audio, str):
        import soundfile as sf  # already a listenr core dependency
        array, sample_rate = sf.read(audio, dtype="float32")
        if array.ndim > 1:          # stereo → mono
            array = array.mean(axis=1)
    else:
        array = audio["array"]
        sample_rate = audio["sampling_rate"]

    # Whisper is fixed at 16 kHz; fall back to that when the extractor doesn't
    # report a real rate (e.g. mocked processors in tests).
    target_sr = getattr(processor.feature_extractor, "sampling_rate", None)
    if not isinstance(target_sr, int):
        target_sr = DEFAULT_TARGET_SAMPLE_RATE

    array = _resample(array, sample_rate, target_sr)

    input_features = processor.feature_extractor(
        array,
        sampling_rate=target_sr,
    ).input_features[0]

    labels = processor.tokenizer(batch["corrected_transcription"]).input_ids

    return {"input_features": input_features, "labels": labels}


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def make_dataset(hf_dataset_path: Path, processor: Any) -> Any:
    """Load the on-disk HuggingFace DatasetDict and apply feature preparation.

    *hf_dataset_path* should be the ``hf_dataset/`` subdirectory written by
    ``listenr build-dataset --format hf``.

    The returned DatasetDict has columns ``input_features`` and ``labels``
    ready for the Trainer.
    """
    try:
        from datasets import load_from_disk
    except ImportError:
        print(
            "ERROR: datasets is required. Install with:\n"
            "  uv pip install -e \".[finetune]\"",
            file=sys.stderr,
        )
        sys.exit(1)

    dataset = load_from_disk(str(hf_dataset_path))

    # Map feature prep; remove raw columns afterwards to save memory.
    raw_columns = dataset.column_names
    # column_names is a dict for DatasetDict
    cols_to_remove = list(next(iter(raw_columns.values())) if isinstance(raw_columns, dict) else raw_columns)

    dataset = dataset.map(
        lambda batch: prepare_example(batch, processor),
        remove_columns=cols_to_remove,
    )
    return dataset


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------

@dataclass
class WhisperDataCollator:
    """Collate a list of ``prepare_example`` outputs into a padded batch.

    Handles the two-part padding requirement of Whisper:

    * ``input_features`` — already fixed-size (80 × 3000); just stack into
      a tensor.
    * ``labels`` — variable-length; pad to the longest sequence in the batch
      and replace padding positions with ``-100`` so they are ignored by the
      cross-entropy loss.  The BOS token is trimmed if present (Whisper's
      Trainer appends it automatically).
    """

    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: list[dict]) -> dict:
        try:
            import torch
        except ImportError:
            print(
                "ERROR: torch is required. Install with:\n"
                "  uv pip install -e \".[finetune]\"",
                file=sys.stderr,
            )
            sys.exit(1)

        # --- input features (fixed size → just convert to tensors) ---
        input_batch = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_batch, return_tensors="pt")

        # --- labels (variable length → pad + mask) ---
        label_batch = [{"input_ids": f["labels"]} for f in features]
        labels_padded = self.processor.tokenizer.pad(label_batch, return_tensors="pt")
        labels = labels_padded.input_ids.masked_fill(
            labels_padded.attention_mask.ne(1), -100
        )

        # Trim leading BOS token if the tokenizer prepended it; Trainer adds it back.
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
