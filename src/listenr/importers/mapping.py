"""Declarative mapping from a source dataset's columns to Listenr fields.

This is Listenr's own, small analogue of the per-dataset column mappings that
data platforms (e.g. Mozilla Data Collective) apply internally. Every source
provides a sensible default :class:`FieldMapping`; a caller can override any of
the important columns per dataset (via CLI flags) without touching code.

For each Listenr field we keep an ordered tuple of *candidate* source columns
and use the first one present in a given row. That is why, for example, both
MDC (``transcription``) and Hugging Face Common Voice (``sentence``) work out of
the box against the same default text mapping.
"""

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
    """Ordered candidate source columns for each Listenr manifest field.

    The first candidate found (and non-empty) in a row wins. ``audio`` and
    ``text`` are required for a row to be imported; the rest are optional and
    fall back to reading the audio file directly when absent.
    """

    audio: tuple[str, ...] = ("audio_path", "audio")
    text: tuple[str, ...] = DEFAULT_TEXT_COLUMNS
    split: tuple[str, ...] = ("split",)
    duration_s: tuple[str, ...] = ("duration_s",)
    duration_ms: tuple[str, ...] = ("duration_ms",)
    sample_rate: tuple[str, ...] = ("sample_rate", "sampling_rate")
    timestamp: tuple[str, ...] = ("timestamp",)

    def with_overrides(
        self,
        *,
        audio: str | None = None,
        text: str | None = None,
        split: str | None = None,
    ) -> "FieldMapping":
        """Return a copy with per-dataset column overrides applied.

        An override makes the given column the sole (highest priority) candidate
        so a dataset with unusual column names can be imported without code
        changes: ``--audio-column path --text-column caption``.
        """
        changes: dict[str, tuple[str, ...]] = {}
        if audio:
            changes["audio"] = (audio,)
        if text:
            changes["text"] = (text,)
        if split:
            changes["split"] = (split,)
        return replace(self, **changes)
