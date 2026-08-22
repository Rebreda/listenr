"""
data.py — Dataset preparation for speech seq2seq LoRA fine-tuning.

Covers every family in :mod:`listenr.finetune.architectures` (Whisper,
Moonshine). The architecture only decides which tensor the encoder wants and
whether it needs padding; the rest of the path is shared.

All functions are pure (no global state, no side effects) to keep them
easy to test and reuse.

Requires the ``finetune`` optional dependencies::

    uv pip install "listenr[finetune]"

Public API
----------
make_processor(model_id, language, task)  -> processor for the detected family
prepare_example(batch, processor, ...)    -> dict with encoder input + labels
make_dataset(hf_dataset_path, processor)  -> DatasetDict with train/dev/test
SpeechDataCollator                        -> dataclass, handles per-batch padding
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from listenr.finetune.architectures import Architecture, detect

if TYPE_CHECKING:
    from transformers import ProcessorMixin  # noqa: F401


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

def make_processor(
    model_id: str,
    language: str,
    task: str,
    architecture: Architecture | None = None,
) -> "ProcessorMixin":
    """Load the processor (feature extractor + tokenizer) for *model_id*.

    Parameters
    ----------
    model_id:
        HuggingFace Hub identifier, e.g. ``"openai/whisper-small"`` or
        ``"UsefulSensors/moonshine-base"``.
    language:
        Target language, e.g. ``"english"``.  Passed to the tokenizer so the
        correct language token is prepended during encoding.  Ignored by
        English-only families such as Moonshine.
    task:
        ``"transcribe"`` (default) or ``"translate"``.  Ignored as above.
    architecture:
        Detected from *model_id* when omitted.
    """
    try:
        from transformers import AutoProcessor
    except ImportError:
        print(
            "ERROR: transformers is required. Install with:\n"
            "  uv pip install \"listenr[finetune]\"",
            file=sys.stderr,
        )
        sys.exit(1)

    arch = architecture or detect(model_id)
    if arch.supports_language_and_task:
        processor = AutoProcessor.from_pretrained(model_id, language=language, task=task)
    else:
        processor = AutoProcessor.from_pretrained(model_id)

    _ensure_pad_token(processor, model_id)
    return processor


def _ensure_pad_token(processor: Any, model_id: str) -> None:
    """Give the tokenizer a pad token when the checkpoint ships without one.

    The collator pads labels to the longest sequence in the batch, and
    ``tokenizer.pad`` refuses to run without a pad token. Whisper's tokenizer
    has one. Moonshine's exposes no special tokens at all, but its model
    config names a ``pad_token_id``, so read the token back from there.

    Reusing EOS as padding is safe here: the collator replaces padding
    positions with -100 before the loss sees them.
    """
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None or getattr(tokenizer, "pad_token", None) is not None:
        return

    pad_token = getattr(tokenizer, "eos_token", None) or _pad_token_from_config(
        model_id, tokenizer
    )
    if pad_token is not None:
        tokenizer.pad_token = pad_token


def _pad_token_from_config(model_id: str, tokenizer: Any) -> str | None:
    """Resolve a pad token string from the checkpoint config's token ids."""
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_id)
    except Exception:
        return None

    for attr in ("pad_token_id", "eos_token_id"):
        token_id = getattr(config, attr, None)
        if token_id is None:
            continue
        try:
            return tokenizer.convert_ids_to_tokens(token_id)
        except Exception:
            continue
    return None


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


def prepare_example(
    batch: dict,
    processor: Any,
    feature_key: str = "input_features",
) -> dict:
    """Convert a single dataset example into model-ready tensors.

    Expects the dataset to have been created by ``listenr build-dataset --format hf``
    (or ``both``).  ``audio_path`` may be either:

    * A plain file-path string — loaded on-the-fly with ``soundfile``.
    * A decoded HuggingFace :class:`datasets.Audio` dict with keys
      ``array`` and ``sampling_rate`` — used directly (legacy / test usage).

    Audio is resampled to the feature extractor's expected rate (16 kHz) so
    that clips imported at other sample rates train correctly.

    Returns a dict with:

    *feature_key*
        Whatever the family's encoder consumes: ``input_features`` (Whisper's
        log-Mel spectrogram, shape ``(80, 3000)``) or ``input_values``
        (Moonshine's raw waveform, variable length).
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

    # Every supported family is fixed at 16 kHz; fall back to that when the
    # extractor doesn't report a real rate (e.g. mocked processors in tests).
    target_sr = getattr(processor.feature_extractor, "sampling_rate", None)
    if not isinstance(target_sr, int):
        target_sr = DEFAULT_TARGET_SAMPLE_RATE

    array = _resample(array, sample_rate, target_sr)

    extracted = processor.feature_extractor(array, sampling_rate=target_sr)

    labels = processor.tokenizer(batch["corrected_transcription"]).input_ids

    return {feature_key: getattr(extracted, feature_key)[0], "labels": labels}


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def make_dataset(
    hf_dataset_path: Path,
    processor: Any,
    feature_key: str = "input_features",
) -> Any:
    """Load the on-disk HuggingFace DatasetDict and apply feature preparation.

    *hf_dataset_path* should be the ``hf_dataset/`` subdirectory written by
    ``listenr build-dataset --format hf``.

    The returned DatasetDict has columns *feature_key* and ``labels`` ready
    for the Trainer.
    """
    try:
        from datasets import load_from_disk
    except ImportError:
        print(
            "ERROR: datasets is required. Install with:\n"
            "  uv pip install \"listenr[finetune]\"",
            file=sys.stderr,
        )
        sys.exit(1)

    dataset = load_from_disk(str(hf_dataset_path))

    # Map feature prep; remove raw columns afterwards to save memory.
    raw_columns = dataset.column_names
    # column_names is a dict for DatasetDict
    cols_to_remove = list(next(iter(raw_columns.values())) if isinstance(raw_columns, dict) else raw_columns)

    dataset = dataset.map(
        lambda batch: prepare_example(batch, processor, feature_key),
        remove_columns=cols_to_remove,
    )
    return dataset


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------

@dataclass
class SpeechDataCollator:
    """Collate a list of ``prepare_example`` outputs into a padded batch.

    Handles the two-part padding requirement of an encoder-decoder ASR model:

    * The encoder input, named by *feature_key*.  Whisper's feature extractor
      has already padded every clip to a fixed 80 × 3000 window, so the batch
      only needs stacking.  Moonshine keeps clips at their natural length, so
      set *pad_features* to pad to the longest clip in the batch and emit the
      ``attention_mask`` its encoder expects.
    * ``labels`` — variable-length; pad to the longest sequence in the batch
      and replace padding positions with ``-100`` so they are ignored by the
      cross-entropy loss.  The BOS token is trimmed if present (the Trainer
      appends it automatically).
    """

    processor: Any
    decoder_start_token_id: int
    feature_key: str = "input_features"
    pad_features: bool = False

    def __call__(self, features: list[dict]) -> dict:
        try:
            import torch
        except ImportError:
            print(
                "ERROR: torch is required. Install with:\n"
                "  uv pip install \"listenr[finetune]\"",
                file=sys.stderr,
            )
            sys.exit(1)

        # --- encoder input ---
        input_batch = [{self.feature_key: f[self.feature_key]} for f in features]
        pad_kwargs: dict[str, Any] = {"return_tensors": "pt"}
        if self.pad_features:
            pad_kwargs.update(padding="longest", return_attention_mask=True)
        batch = self.processor.feature_extractor.pad(input_batch, **pad_kwargs)

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


#: Historical name kept so existing configs and imports keep working.
WhisperDataCollator = SpeechDataCollator
