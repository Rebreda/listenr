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

    ``audio`` and ``text`` are required for a row to be imported. ``split``
    and ``speaker`` are optional; build-dataset uses the speaker to keep test
    voices out of training.
    """

    audio: tuple[str, ...] = ("audio_path", "audio")
    text: tuple[str, ...] = DEFAULT_TEXT_COLUMNS
    split: tuple[str, ...] = ("split",)
    speaker: tuple[str, ...] = ("speaker_id", "client_id")

    def with_overrides(
        self,
        *,
        audio: str | None = None,
        text: str | None = None,
        split: str | None = None,
        speaker: str | None = None,
    ) -> "FieldMapping":
        """Return a copy where each given column becomes the sole candidate."""
        changes: dict[str, tuple[str, ...]] = {}
        if audio:
            changes["audio"] = (audio,)
        if text:
            changes["text"] = (text,)
        if split:
            changes["split"] = (split,)
        if speaker:
            changes["speaker"] = (speaker,)
        return replace(self, **changes)
