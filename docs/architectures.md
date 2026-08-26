# Model architectures

Listenr fine-tunes three model families. They share almost everything: the same
LoRA setup, the same encoder freezing, the same training loop, the same merge
and evaluation. This page is about the handful of places where they genuinely
differ, and where that difference lives in the code.

## What is supported

| Family | Example checkpoints | Notes |
|---|---|---|
| Whisper | `openai/whisper-tiny` … `openai/whisper-large-v3-turbo` | Multilingual. `--language` and `--task` apply. |
| Moonshine | `UsefulSensors/moonshine-tiny`, `UsefulSensors/moonshine-base` | English-only, much smaller, built for edge use. |
| Moonshine streaming | `moonshine-ai/moonshine-streaming-small` | English-only. Ships its own neural VAD when served. |

Pick one with `--base-model`. Nothing else changes:

```bash
listenr finetune --base-model moonshine-ai/moonshine-streaming-small
```

`merge` and `eval` read the family from the checkpoint, so they need no flag.

CTC and transducer models (Parakeet, Wav2Vec2) are **not** supported. They use a
different loss and a different label layout, so they need more than a new entry
in the table below.

## Where the families diverge

Every difference is a field on `Architecture` in
[`src/listenr/finetune/architectures.py`](../src/listenr/finetune/architectures.py).
If you are adding a family, this table is the thing to fill in.

| | `whisper` | `moonshine` | `moonshine_streaming` |
|---|---|---|---|
| `feature_key` | `input_features` | `input_values` | `input_values` |
| `pad_features` | `False` | `True` | `True` |
| `pad_to_multiple` | – | – | `80` |
| `supports_language_and_task` | `True` | `False` | `False` |

### What the encoder eats

Whisper takes a log-Mel spectrogram. Its feature extractor already pads every
clip to a fixed 30 second window, so a batch only needs stacking and carries no
attention mask.

Both Moonshine variants take the raw waveform at its natural length, so batches
need real padding and an attention mask. That is `pad_features`.

### Frame alignment

The streaming encoder reshapes its input to `[batch, -1, 80]`, one 5 ms frame at
16 kHz, and raises on any other length:

```
RuntimeError: shape '[4, -1, 80]' is invalid for input of size 4246272
```

Padding a batch to its longest clip hits a multiple of 80 only by chance, so
`pad_to_multiple` rounds up. Nothing else needs it today.

### Language and task tokens

Whisper's tokenizer prepends them and its generation config carries them. Both
Moonshine variants are English-only and have neither, so `--language` and
`--task` are accepted and ignored there, and `eval` omits them from its
generation arguments rather than passing something the model will reject.

### Pad tokens

Not a table field, because it is handled by falling back rather than by
declaring it.

Whisper's tokenizer has a pad token. Streaming Moonshine's does too. Offline
Moonshine's exposes no special tokens at all, `pad_token`, `eos_token` and
`bos_token` are all `None`, so padding labels fails with *"Asking to pad but the
tokenizer does not have a padding token"*. Its model config does name a
`pad_token_id`, so `make_processor` reads the token back from there. Padding
positions become `-100` before the loss sees them, so reusing EOS is safe.

### LoRA target names

Also not a table field, because the default works everywhere. Whisper names its
output projection `out_proj` and both Moonshine variants name it `o_proj`, but
the minimal effective set is `q_proj` and `v_proj`, spelled the same in all
three. If you configure `lora_target_modules` yourself, check the names against
your model. PEFT raises when a target matches nothing, so a mismatch fails
loudly rather than training an adapter that does nothing.

## How the family is chosen

`detect()` reads the real `model_type` from the checkpoint config, so local
paths and renamed forks resolve correctly. If the config cannot be read, for
example without the finetune extras installed, it falls back to matching the
model id text.

That fallback compares with hyphens and underscores flattened, and takes the
longest match first. Without that, `moonshine-streaming-small` matches
`moonshine` and silently trains with the wrong processor.

An unsupported family raises `UnsupportedArchitecture` naming what is
supported, rather than failing later with something obscure.

## Adding a family

For another encoder-decoder model, add an entry to `SUPPORTED` and check
whether any existing field needs a new value. If it needs something none of the
fields express, add a field rather than a branch at the call site: the point of
the table is that every difference is visible in one place.

For CTC or transducer models, this table is not enough. The loss, the label
layout and the collator all differ, which is a larger change than a new row.

## Serving is not the same as fine-tuning

Lemonade serves `Moonshine-Medium-Streaming`, which is
[`moonshine-ai/moonshine-streaming`](https://huggingface.co/moonshine-ai/moonshine-streaming).
That repository ships ONNX graphs only, with no PyTorch weights, so it can be
served but not fine-tuned.
[`moonshine-streaming-small`](https://huggingface.co/moonshine-ai/moonshine-streaming-small)
is the transformers checkpoint and is what `listenr finetune` takes.
