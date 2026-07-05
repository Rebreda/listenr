"""Declarative mapping from a source dataset's columns to Listenr fields."""

from __future__ import annotations

from dataclasses import dataclass, replace

DEFAULT_TEXT_COLUMNS = (
    "transcription",
    "sentence",
    "text",
    "raw_transcription",
    "corrected_transcription",
)


@dataclass(frozen=True)
class FieldMapping:
    """Ordered candidate source columns per field; first non-empty match wins.

    ``audio`` and ``text`` are required for a row to be imported.
    """

    audio: tuple[str, ...] = ("audio_path", "audio")
    text: tuple[str, ...] = DEFAULT_TEXT_COLUMNS
    split: tuple[str, ...] = ("split",)

    def with_overrides(
        self,
        *,
        audio: str | None = None,
        text: str | None = None,
        split: str | None = None,
    ) -> "FieldMapping":
        """Return a copy where each given column becomes the sole candidate."""
        changes: dict[str, tuple[str, ...]] = {}
        if audio:
            changes["audio"] = (audio,)
        if text:
            changes["text"] = (text,)
        if split:
            changes["split"] = (split,)
        return replace(self, **changes)
