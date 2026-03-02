"""
constants.py — Typed, config-backed constants for the listenr package.

All values are read **once** at import time from ``~/.config/listenr/config.ini``
(via :mod:`listenr.config_manager`).  

Downstream modules should import individual names::

    from listenr.constants import CAPTURE_RATE, LLM_MODEL, WHISPER_MODEL

If you need to refresh constants at runtime (e.g. tests that patch config),
call :func:`reload` to re-read all values from the current config state.
"""

from __future__ import annotations

from pathlib import Path

import listenr.config_manager as cfg

# ---------------------------------------------------------------------------
# Lemonade
# ---------------------------------------------------------------------------

LEMONADE_API_BASE: str = (
    cfg.get_setting("Lemonade", "api_base", "http://localhost:8000/api/v1")
    or "http://localhost:8000/api/v1"
)

# ---------------------------------------------------------------------------
# Whisper
# ---------------------------------------------------------------------------

WHISPER_MODEL: str = (
    cfg.get_setting("Whisper", "model", "Whisper-Tiny") or "Whisper-Tiny"
)

# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

CAPTURE_RATE: int = cfg.get_int_setting("Audio", "sample_rate", 48000)
CHANNELS: int = cfg.get_int_setting("Audio", "channels", 1)
CHUNK_SIZE: int = cfg.get_int_setting("Audio", "blocksize", 4096)
INPUT_DEVICE: str | None = (
    cfg.get_setting("Audio", "input_device", "pipewire") or None
)
if INPUT_DEVICE == "default":
    INPUT_DEVICE = None

# Lemonade /realtime always requires 16 kHz PCM-16 — this is not configurable.
ASR_RATE: int = 16000

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

STORAGE_BASE: Path = Path(
    cfg.get_setting("Storage", "audio_clips_path", "~/.listenr/audio_clips")
    or "~/.listenr/audio_clips"
).expanduser()

STORAGE_CLIPS_ENABLED: bool = cfg.get_bool_setting(
    "Storage", "audio_clips_enabled", True
)
STORAGE_RETENTION_DAYS: int = cfg.get_int_setting("Storage", "retention_days", 90)
STORAGE_MAX_GB: float = cfg.get_float_setting("Storage", "max_storage_gb", 10.0)

# ---------------------------------------------------------------------------
# VAD
# ---------------------------------------------------------------------------

VAD_THRESHOLD: float = cfg.get_float_setting("VAD", "threshold", 0.05)
VAD_SILENCE_MS: int = cfg.get_int_setting("VAD", "silence_duration_ms", 800)
VAD_PREFIX_PADDING_MS: int = cfg.get_int_setting("VAD", "prefix_padding_ms", 250)

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

LLM_ENABLED: bool = cfg.get_bool_setting("LLM", "enabled", True)
LLM_MODEL: str = (
    cfg.get_setting("LLM", "model", "gpt-oss-20b-mxfp4-GGUF")
    or "gpt-oss-20b-mxfp4-GGUF"
)
LLM_API_BASE: str = (
    cfg.get_setting("LLM", "api_base", "http://localhost:8000/api/v1")
    or "http://localhost:8000/api/v1"
)
LLM_TEMPERATURE: float = cfg.get_float_setting("LLM", "temperature", 0.3)
LLM_MAX_TOKENS: int = cfg.get_int_setting("LLM", "max_tokens", 1500)
LLM_TIMEOUT: int = cfg.get_int_setting("LLM", "timeout", 30)
LLM_CONTEXT_WINDOW: int = cfg.get_int_setting("LLM", "context_window", 10)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

DATASET_OUTPUT: Path = Path(
    cfg.get_setting("Dataset", "output_path", "~/listenr_dataset")
    or "~/listenr_dataset"
).expanduser()
DATASET_SPLIT: str = cfg.get_setting("Dataset", "split", "80/10/10") or "80/10/10"
DATASET_MIN_DURATION: float = cfg.get_float_setting("Dataset", "min_duration", 0.3)
DATASET_MIN_CHARS: int = cfg.get_int_setting("Dataset", "min_chars", 2)
DATASET_SEED: int = cfg.get_int_setting("Dataset", "seed", 42)

_VALID_DATASET_FORMATS: frozenset[str] = frozenset({"csv", "hf", "both"})
_raw_dataset_format: str = cfg.get_setting("Dataset", "format", "csv") or "csv"
if _raw_dataset_format not in _VALID_DATASET_FORMATS:
    import warnings
    warnings.warn(
        f"Config [Dataset] format={_raw_dataset_format!r} is not a recognised value "
        f"({', '.join(sorted(_VALID_DATASET_FORMATS))}); falling back to 'csv'.",
        UserWarning,
        stacklevel=2,
    )
    _raw_dataset_format = "csv"
DATASET_FORMAT: str = _raw_dataset_format

# ---------------------------------------------------------------------------
# Output / transcript files
# ---------------------------------------------------------------------------

OUTPUT_FILE: Path | None = (
    Path(v).expanduser()
    if (v := cfg.get_setting("Output", "file", ""))
    else None
)
OUTPUT_LLM_FILE: Path | None = (
    Path(v).expanduser()
    if (v := cfg.get_setting("Output", "llm_file", ""))
    else None
)
OUTPUT_LINE_FORMAT: str = (
    cfg.get_setting("Output", "line_format", "[{timestamp}] {text}")
    or "[{timestamp}] {text}"
)
OUTPUT_TIMESTAMP_FORMAT: str = (
    cfg.get_setting("Output", "timestamp_format", "%Y-%m-%d %H:%M:%S")
    or "%Y-%m-%d %H:%M:%S"
)
OUTPUT_SHOW_RAW: bool = cfg.get_bool_setting("Output", "show_raw", False)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL: str = cfg.get_setting("Logging", "level", "INFO") or "INFO"
LOG_FILE: Path | None = (
    Path(v).expanduser()
    if (v := cfg.get_setting("Logging", "file", ""))
    else None
)


# ---------------------------------------------------------------------------
# Reload helper (used by tests and advanced callers)
# ---------------------------------------------------------------------------

def reload() -> None:
    """Re-read all constants from the current config state (in-place update).

    Useful in tests that patch :mod:`listenr.config_manager` after import::

        cfg.update_setting('LLM', 'model', 'my-test-model')
        import listenr.constants as C
        C.reload()
        assert C.LLM_MODEL == 'my-test-model'
    """
    import sys
    import importlib

    # Re-execute this module in the same module object so all names are updated
    # in place — existing ``from listenr.constants import X`` bindings in already-
    # imported modules won't see the change, but direct attribute access on the
    # module object (``constants.X``) will.
    module = sys.modules[__name__]
    importlib.reload(module)
