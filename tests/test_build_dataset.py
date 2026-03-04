"""
Unit tests for build_dataset helpers — validate_entry() and tag-stripping behaviour.
Run with:  python -m pytest tests/
"""

import pytest
from pathlib import Path

from listenr.build_dataset import validate_entry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(tmp_path: Path, **overrides) -> dict:
    """Return a minimal valid manifest record with a real audio file on disk."""
    audio = tmp_path / "clip.wav"
    audio.touch()
    base = {
        "uuid": "test-uuid-1",
        "audio_path": str(audio),
        "raw_transcription": "Hello world.",
        "corrected_transcription": "Hello world.",
        "duration_s": 1.0,
        "sample_rate": 16000,
        "whisper_model": "Whisper-Tiny",
        "llm_model": "",
        "timestamp": "2026-03-03 10:00:00",
        "is_improved": "false",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Required-field / filter checks
# ---------------------------------------------------------------------------

class TestValidateEntryFilters:
    def test_valid_record_passes(self, tmp_path):
        rec = _make_record(tmp_path)
        result = validate_entry(rec, min_duration=0.3, min_chars=2)
        assert result is not None
        assert result["uuid"] == "test-uuid-1"

    def test_missing_uuid_rejected(self, tmp_path):
        rec = _make_record(tmp_path, uuid="")
        assert validate_entry(rec, 0.3, 2) is None

    def test_missing_raw_transcription_rejected(self, tmp_path):
        rec = _make_record(tmp_path, raw_transcription="")
        assert validate_entry(rec, 0.3, 2) is None

    def test_missing_audio_path_rejected(self, tmp_path):
        rec = _make_record(tmp_path, audio_path="")
        assert validate_entry(rec, 0.3, 2) is None

    def test_nonexistent_audio_file_rejected(self, tmp_path):
        rec = _make_record(tmp_path, audio_path=str(tmp_path / "missing.wav"))
        assert validate_entry(rec, 0.3, 2) is None

    def test_duration_below_minimum_rejected(self, tmp_path):
        rec = _make_record(tmp_path, duration_s=0.1)
        assert validate_entry(rec, min_duration=0.5, min_chars=2) is None

    def test_duration_at_minimum_passes(self, tmp_path):
        rec = _make_record(tmp_path, duration_s=0.5)
        assert validate_entry(rec, min_duration=0.5, min_chars=2) is not None

    def test_transcript_too_short_rejected(self, tmp_path):
        rec = _make_record(tmp_path, raw_transcription="a", corrected_transcription="a")
        assert validate_entry(rec, 0.3, min_chars=5) is None

    def test_corrected_falls_back_to_raw(self, tmp_path):
        rec = _make_record(tmp_path, corrected_transcription=None)
        result = validate_entry(rec, 0.3, 2)
        assert result is not None
        assert result["corrected_transcription"] == result["raw_transcription"]


# ---------------------------------------------------------------------------
# strip_tags=True (default) — tags removed from output fields
# ---------------------------------------------------------------------------

class TestValidateEntryStripTags:
    def test_paren_tag_stripped_from_corrected(self, tmp_path):
        rec = _make_record(tmp_path, corrected_transcription="(music) Hello world.")
        result = validate_entry(rec, 0.3, 2, strip_tags=True)
        assert result is not None
        assert "(music)" not in result["corrected_transcription"]
        assert result["corrected_transcription"] == "Hello world."

    def test_paren_tag_stripped_from_raw(self, tmp_path):
        rec = _make_record(tmp_path, raw_transcription="(sort music) Hello world.")
        result = validate_entry(rec, 0.3, 2, strip_tags=True)
        assert result is not None
        assert "(sort music)" not in result["raw_transcription"]

    def test_bracket_tag_stripped(self, tmp_path):
        rec = _make_record(tmp_path, corrected_transcription="[Applause] Thank you.")
        result = validate_entry(rec, 0.3, 2, strip_tags=True)
        assert result is not None
        assert "[Applause]" not in result["corrected_transcription"]

    def test_only_tag_becomes_too_short_rejected(self, tmp_path):
        """A transcript that is *only* a tag should be rejected after stripping."""
        rec = _make_record(tmp_path,
                           raw_transcription="(music)",
                           corrected_transcription="(music)")
        assert validate_entry(rec, 0.3, min_chars=2, strip_tags=True) is None

    def test_multiple_tags_all_stripped(self, tmp_path):
        rec = _make_record(tmp_path,
                           corrected_transcription="(music) Hello (applause) world.")
        result = validate_entry(rec, 0.3, 2, strip_tags=True)
        assert result is not None
        assert result["corrected_transcription"] == "Hello world."

    def test_no_tags_unchanged(self, tmp_path):
        rec = _make_record(tmp_path, corrected_transcription="No tags here.")
        result = validate_entry(rec, 0.3, 2, strip_tags=True)
        assert result is not None
        assert result["corrected_transcription"] == "No tags here."


# ---------------------------------------------------------------------------
# strip_tags=False — tags preserved verbatim
# ---------------------------------------------------------------------------

class TestValidateEntryNoStrip:
    def test_paren_tag_preserved(self, tmp_path):
        rec = _make_record(tmp_path, corrected_transcription="(music) Hello world.")
        result = validate_entry(rec, 0.3, 2, strip_tags=False)
        assert result is not None
        assert "(music)" in result["corrected_transcription"]

    def test_bracket_tag_preserved(self, tmp_path):
        rec = _make_record(tmp_path, corrected_transcription="[Applause] Hi.")
        result = validate_entry(rec, 0.3, 2, strip_tags=False)
        assert result is not None
        assert "[Applause]" in result["corrected_transcription"]

    def test_min_chars_applied_to_raw_including_tags(self, tmp_path):
        """With strip_tags=False the tag text counts toward min_chars."""
        rec = _make_record(tmp_path,
                           raw_transcription="(music)",
                           corrected_transcription="(music)")
        # "(music)" is 7 non-space chars — should pass min_chars=2
        result = validate_entry(rec, 0.3, min_chars=2, strip_tags=False)
        assert result is not None
