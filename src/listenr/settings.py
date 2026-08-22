"""Typed application settings, loaded once at import.

Single source of truth for every tunable in listenr. Precedence, highest first:

1. Environment variables — ``LISTENR_<SECTION>__<KEY>``, e.g.
   ``LISTENR_FINETUNE__MAX_STEPS=500``
2. A ``.env`` file in the working directory (same variable names)
3. ``~/.config/listenr/config.toml`` (override the path with ``LISTENR_CONFIG``)
4. The defaults declared on the models below

Usage::

    from listenr.settings import settings
    settings.finetune.max_steps
"""

from __future__ import annotations

import json
import os
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from typing import Annotated

from pydantic import AfterValidator, BaseModel, field_validator
from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    EnvSettingsSource,
    NoDecode,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

CONFIG_FILE = Path.home() / ".config" / "listenr" / "config.toml"

# Lemonade /realtime always requires 16 kHz PCM-16 — not configurable.
ASR_RATE: int = 16_000

ExpandedPath = Annotated[Path, AfterValidator(lambda p: p.expanduser())]


class WhisperSettings(BaseModel):
    # Whisper model served by Lemonade (whisper.cpp backend), e.g.
    # Whisper-Tiny, Whisper-Base, Whisper-Large-v3-Turbo.
    model: str = "Whisper-Base"


class AudioSettings(BaseModel):
    # Mic capture rate — must match the device's native rate; resampled to
    # 16 kHz internally before sending to Lemonade /realtime.
    sample_rate: int = 48_000
    channels: int = 1
    # Frames per mic read (~85 ms of audio at 48 kHz).
    blocksize: int = 4_096
    # 'pipewire', a device name, an index — or None/'default' for the system default.
    input_device: str | None = "pipewire"

    @field_validator("input_device")
    @classmethod
    def _default_device_is_none(cls, v: str | None) -> str | None:
        return None if v in ("", "default") else v


class StorageSettings(BaseModel):
    audio_clips_path: ExpandedPath = Path("~/.listenr/audio_clips").expanduser()


class VADSettings(BaseModel):
    # Server-side VAD settings passed to Lemonade via session.update.
    # RMS energy threshold; raise (0.08–0.15) to ignore background noise.
    threshold: float = 0.05
    # ms of silence required to end a speech segment.
    silence_duration_ms: int = 800
    # Minimum speech duration (ms) before triggering transcription.
    prefix_padding_ms: int = 250
    # Client-side cap on segment length (s); Whisper hallucinates above ~20 s.
    max_segment_s: float = 12.0


class LLMSettings(BaseModel):
    enabled: bool = True
    model: str = "gpt-oss-20b-mxfp4-GGUF"
    # Lemonade Server API base (OpenAI-compatible).
    api_base: str = "http://localhost:13305/api/v1"
    temperature: float = 0.3
    max_tokens: int = 1_500
    timeout: int = 30
    # Preceding segments passed as context to the LLM.
    context_window: int = 10


class DatasetSettings(BaseModel):
    output_path: ExpandedPath = Path("~/listenr_dataset").expanduser()
    # Train/dev/test split percentages.
    split: str = "80/10/10"
    min_duration: float = 0.3
    min_chars: int = 2
    seed: int = 42
    format: Literal["csv", "hf", "both"] = "csv"
    # Strip parenthesised noise tags like (music) from transcriptions.
    strip_tags: bool = True


class FinetuneSettings(BaseModel):
    base_model: str = "openai/whisper-small"
    language: str = "english"
    task: Literal["transcribe", "translate"] = "transcribe"
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    # Accepts a list (TOML) or a comma-separated string (env var).
    lora_target_modules: Annotated[list[str], NoDecode] = ["q_proj", "v_proj"]
    freeze_encoder: bool = True
    learning_rate: float = 1e-4
    warmup_steps: int = 100
    max_steps: int = 2_000
    batch_size: int = 8
    grad_accum_steps: int = 2
    fp16: bool = False
    # bf16 is the recommended mixed precision on AMD ROCm (RDNA2+).
    bf16: bool = False
    output_dir: ExpandedPath = Path("~/listenr_finetune").expanduser()
    eval_steps: int = 200
    save_steps: int = 400
    generation_max_length: int = 128

    @field_validator("lora_target_modules", mode="before")
    @classmethod
    def _split_csv(cls, v: object) -> object:
        return v.split(",") if isinstance(v, str) else v


# Keyword corrections passed to the LLM to fix common STT misrecognitions,
# {misheard: correct}. Keys are matched case-insensitively.
DEFAULT_CORRECTIONS: dict[str, str] = {
    "clod": "Claude Code",
    "clode": "Claude Code",
    "cloud code": "Claude Code",
    "clock code": "Claude Code",
    "open ai": "OpenAI",
    "unsurropic": "Anthropic",
    "anthropic": "Anthropic",
}


def _is_json(value: str | None) -> bool:
    try:
        json.loads(value or "")
    except ValueError:
        return False
    return True


class _IgnoreSectionVars:
    """Drop bare ``LISTENR_<SECTION>`` variables before pydantic decodes them.

    Each section below is a nested model, so pydantic treats a bare
    ``LISTENR_FINETUNE`` as JSON for the whole section and raises at import
    when it holds anything else. docker-compose and older ``.env`` files used
    exactly those names for host paths (now ``LISTENR_HOST_*``), so ignore
    them with a warning rather than crashing. Per-key overrides such as
    ``LISTENR_FINETUNE__MAX_STEPS`` use the ``__`` delimiter and are untouched.
    """

    def _load_env_vars(self) -> Mapping[str, str | None]:
        env_vars = super()._load_env_vars()  # type: ignore[misc]
        unusable = {
            key
            for key in _SECTION_ENV_VARS & set(env_vars)
            if not _is_json(env_vars[key])
        }
        if not unusable:
            return env_vars
        warnings.warn(
            "Ignoring environment variable(s) "
            + ", ".join(sorted(v.upper() for v in unusable))
            + ": each names a whole settings section, so its value must be JSON. "
            "Set a single key with the __ delimiter instead, e.g. "
            "LISTENR_FINETUNE__MAX_STEPS=500. Host mount paths for "
            "docker-compose are now named LISTENR_HOST_*. Update your .env.",
            RuntimeWarning,
            stacklevel=2,
        )
        return {k: v for k, v in env_vars.items() if k not in unusable}


class _EnvSource(_IgnoreSectionVars, EnvSettingsSource):
    pass


class _DotEnvSource(_IgnoreSectionVars, DotEnvSettingsSource):
    pass


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LISTENR_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    whisper: WhisperSettings = WhisperSettings()
    audio: AudioSettings = AudioSettings()
    storage: StorageSettings = StorageSettings()
    vad: VADSettings = VADSettings()
    llm: LLMSettings = LLMSettings()
    dataset: DatasetSettings = DatasetSettings()
    finetune: FinetuneSettings = FinetuneSettings()
    corrections: dict[str, str] = DEFAULT_CORRECTIONS

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        toml_file = Path(os.environ.get("LISTENR_CONFIG", CONFIG_FILE))
        return (
            init_settings,
            _EnvSource(settings_cls),
            _DotEnvSource(settings_cls),
            TomlConfigSettingsSource(settings_cls, toml_file=toml_file),
        )


# Env names that address a whole section (e.g. LISTENR_FINETUNE) rather than a
# single key. Populated after the class so it tracks the fields automatically.
_SECTION_ENV_VARS = {f"listenr_{name}" for name in Settings.model_fields}

settings = Settings()
