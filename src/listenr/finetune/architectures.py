"""
architectures.py — Per-architecture differences for speech seq2seq fine-tuning.

Whisper and Moonshine are both encoder-decoder ASR models and share almost the
whole pipeline: LoRA targets, encoder freezing, the Trainer loop, merging. They
disagree on two things that reach into the data path:

* **What the encoder eats.** Whisper takes a log-Mel spectrogram padded to a
  fixed 30 s window (``input_features``); Moonshine takes the raw waveform at
  its natural length (``input_values``), so batches need real padding and an
  attention mask.
* **Language/task tokens.** Whisper's tokenizer prepends them and its
  generation config carries them; both Moonshine variants are English-only
  and have neither.

LoRA targets are *not* listed here: the minimal effective set (``q_proj``,
``v_proj``) is spelled the same in both, so ``settings.finetune`` stays the one
place that configures them. The families do differ on the output projection
(Whisper ``out_proj``, Moonshine ``o_proj``). PEFT raises if a configured
target matches nothing, so a mismatch fails loudly rather than training a
no-op adapter.

Adding another encoder-decoder model should be a matter of adding an entry
below. CTC and transducer models (Parakeet, Wav2Vec2) are a larger change:
they use a different loss and label layout, so they are not represented here.

Public API
----------
Architecture                  -> frozen dataclass describing one model family
SUPPORTED                     -> {model_type: Architecture}
model_type_of(model_id)       -> str | None
detect(model_id)              -> Architecture
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Architecture:
    """How one model family differs from the shared fine-tuning path."""

    model_type: str
    #: Name of the encoder input tensor, and the key ``prepare_example`` emits.
    feature_key: str
    #: True when examples arrive at variable length and the collator must pad
    #: them (and request an attention mask). Whisper's feature extractor
    #: already pads every clip to 30 s, so it does not.
    pad_features: bool
    #: Whether the tokenizer and generation config accept language/task tokens.
    supports_language_and_task: bool
    #: Pad the encoder input up to a multiple of this many samples. Streaming
    #: models consume fixed frames and reshape to them, so a batch padded to an
    #: arbitrary length fails inside the encoder rather than at the boundary.
    pad_to_multiple: int | None = None


WHISPER = Architecture(
    model_type="whisper",
    feature_key="input_features",
    pad_features=False,
    supports_language_and_task=True,
)

MOONSHINE = Architecture(
    model_type="moonshine",
    feature_key="input_values",
    pad_features=True,
    supports_language_and_task=False,
)

# Moonshine's streaming variant. A separate model_type and a separate class in
# transformers, not a flag on the offline one, but it eats the same raw
# waveform and is English-only, so it takes the same path here. Its tokenizer
# does carry a pad token, unlike the offline model's.
MOONSHINE_STREAMING = Architecture(
    model_type="moonshine_streaming",
    feature_key="input_values",
    pad_features=True,
    supports_language_and_task=False,
    # The encoder reshapes its input to [batch, -1, 80], one 5 ms frame at
    # 16 kHz, so a length that is not a multiple of 80 raises there rather
    # than being handled. Padding to the longest clip in a batch lands on a
    # multiple only by luck.
    pad_to_multiple=80,
)

SUPPORTED: dict[str, Architecture] = {
    arch.model_type: arch for arch in (WHISPER, MOONSHINE, MOONSHINE_STREAMING)
}


class UnsupportedArchitecture(RuntimeError):
    """Raised when a model id resolves to a family we cannot fine-tune."""


def _from_model_type(model_type: str) -> Architecture:
    try:
        return SUPPORTED[model_type]
    except KeyError:
        raise UnsupportedArchitecture(
            f"Model type '{model_type}' is not supported for fine-tuning. "
            f"Supported: {', '.join(sorted(SUPPORTED))}."
        ) from None


def model_type_of(model_id: str) -> str | None:
    """Read the checkpoint's ``model_type``, or None when it cannot be read.

    transformers is an optional dependency and the config may be unreachable
    offline, so both cases return None and leave the caller to fall back.
    """
    try:
        from transformers import AutoConfig

        return AutoConfig.from_pretrained(model_id).model_type
    except Exception:
        return None


def detect(model_id: str) -> Architecture:
    """Return the :class:`Architecture` for *model_id*.

    Resolves the real ``model_type`` from the checkpoint's config, so local
    paths and renamed forks work as well as canonical hub ids. Falls back to
    matching the id text when the config cannot be read, which keeps the
    function usable without the finetune extras installed.
    """
    model_type = model_type_of(model_id)
    if model_type is None:
        return _from_name(model_id)
    return _from_model_type(model_type)


def _from_name(model_id: str) -> Architecture:
    # Hub ids separate words with hyphens and model_type uses underscores, so
    # compare with both flattened. Longest first, or "moonshine-streaming-small"
    # matches the offline "moonshine" and silently picks the wrong family.
    lowered = model_id.lower().replace("-", "_")
    for arch in sorted(SUPPORTED.values(), key=lambda a: -len(a.model_type)):
        if arch.model_type in lowered:
            return arch
    raise UnsupportedArchitecture(
        f"Could not determine the architecture of '{model_id}' from its config "
        f"or its name. Supported: {', '.join(sorted(SUPPORTED))}."
    )
