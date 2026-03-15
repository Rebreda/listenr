"""
Unit tests for listenr.retranscribe.

All network calls (Lemonade, LLM) are mocked so no server is needed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from listenr.retranscribe import (
    _load_manifest,
    _should_process,
    _write_manifest,
    retranscribe,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_wav(path: Path, duration_s: float = 0.5, rate: int = 16000) -> None:
    samples = np.zeros(int(rate * duration_s), dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, rate, subtype="PCM_16")


def _make_record(
    tmp_path: Path,
    uid: str = "aabbcc112233",
    raw: str = "hello world",
    corrected: str = "Hello world.",
    model: str = "Whisper-Tiny",
) -> dict:
    audio = tmp_path / "audio" / f"clip_{uid}.wav"
    _write_wav(audio)
    return {
        "uuid": uid,
        "timestamp": "2026-03-08T10:00:00+00:00",
        "audio_path": str(audio),
        "raw_transcription": raw,
        "corrected_transcription": corrected,
        "is_improved": corrected != raw,
        "categories": [],
        "whisper_model": model,
        "llm_model": None,
        "duration_s": 0.5,
        "sample_rate": 16000,
    }


def _make_manifest(tmp_path: Path, records: list[dict]) -> Path:
    manifest = tmp_path / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return manifest


# ---------------------------------------------------------------------------
# _load_manifest
# ---------------------------------------------------------------------------

class TestLoadManifest:
    def test_returns_empty_for_missing_file(self, tmp_path):
        assert _load_manifest(tmp_path / "missing.jsonl") == []

    def test_loads_all_records(self, tmp_path):
        records = [{"uuid": "a", "raw_transcription": "one"}, {"uuid": "b", "raw_transcription": "two"}]
        manifest = _make_manifest(tmp_path, records)
        loaded = _load_manifest(manifest)
        assert len(loaded) == 2
        assert loaded[0]["uuid"] == "a"
        assert loaded[1]["uuid"] == "b"

    def test_skips_blank_lines(self, tmp_path):
        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text('{"uuid":"a"}\n\n{"uuid":"b"}\n')
        loaded = _load_manifest(manifest)
        assert len(loaded) == 2

    def test_skips_malformed_lines(self, tmp_path):
        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text('{"uuid":"a"}\nnot-json\n{"uuid":"b"}\n')
        loaded = _load_manifest(manifest)
        assert len(loaded) == 2


# ---------------------------------------------------------------------------
# _write_manifest
# ---------------------------------------------------------------------------

class TestWriteManifest:
    def test_creates_file(self, tmp_path):
        manifest = tmp_path / "manifest.jsonl"
        _write_manifest(manifest, [{"uuid": "x"}])
        assert manifest.exists()

    def test_roundtrip(self, tmp_path):
        records = [{"uuid": "a", "val": 1}, {"uuid": "b", "val": 2}]
        manifest = tmp_path / "manifest.jsonl"
        _write_manifest(manifest, records)
        loaded = _load_manifest(manifest)
        assert loaded == records

    def test_no_tmp_file_left_behind(self, tmp_path):
        manifest = tmp_path / "manifest.jsonl"
        _write_manifest(manifest, [{"uuid": "x"}])
        assert not (tmp_path / "manifest.jsonl.tmp").exists()


# ---------------------------------------------------------------------------
# _should_process
# ---------------------------------------------------------------------------

class TestShouldProcess:
    def test_no_filters_always_true(self):
        record = {"uuid": "abc", "raw_transcription": "hello"}
        assert _should_process(record, uuids=None, pattern=None) is True

    def test_uuid_filter_match(self):
        record = {"uuid": "abc", "raw_transcription": "hello"}
        assert _should_process(record, uuids={"abc"}, pattern=None) is True

    def test_uuid_filter_no_match(self):
        record = {"uuid": "abc", "raw_transcription": "hello"}
        assert _should_process(record, uuids={"xyz"}, pattern=None) is False

    def test_pattern_filter_match(self):
        record = {"uuid": "abc", "raw_transcription": "hello world"}
        pat = re.compile("hello", re.IGNORECASE)
        assert _should_process(record, uuids=None, pattern=pat) is True

    def test_pattern_filter_no_match(self):
        record = {"uuid": "abc", "raw_transcription": "goodbye"}
        pat = re.compile("hello", re.IGNORECASE)
        assert _should_process(record, uuids=None, pattern=pat) is False

    def test_both_filters_must_pass(self):
        record = {"uuid": "abc", "raw_transcription": "hello world"}
        pat = re.compile("hello", re.IGNORECASE)
        # uuid filter fails even though pattern passes
        assert _should_process(record, uuids={"xyz"}, pattern=pat) is False


# ---------------------------------------------------------------------------
# retranscribe — dry run
# ---------------------------------------------------------------------------

class TestRetranscribeDryRun:
    def test_manifest_not_modified_on_dry_run(self, tmp_path):
        record = _make_record(tmp_path)
        manifest = _make_manifest(tmp_path, [record])
        original_mtime = manifest.stat().st_mtime

        with patch("listenr.retranscribe.lemonade_transcribe_audio", return_value="new text"):
            retranscribe(manifest, dry_run=True)

        assert manifest.stat().st_mtime == original_mtime

    def test_dry_run_summary_shows_updated_count(self, tmp_path):
        record = _make_record(tmp_path, raw="old text")
        manifest = _make_manifest(tmp_path, [record])

        with patch("listenr.retranscribe.lemonade_transcribe_audio", return_value="new text"):
            summary = retranscribe(manifest, dry_run=True)

        assert summary["updated"] == 1
        assert summary["processed"] == 1

    def test_dry_run_no_change_when_transcript_identical(self, tmp_path):
        record = _make_record(tmp_path, raw="same text")
        manifest = _make_manifest(tmp_path, [record])

        with patch("listenr.retranscribe.lemonade_transcribe_audio", return_value="same text"):
            summary = retranscribe(manifest, dry_run=True)

        assert summary["updated"] == 0


# ---------------------------------------------------------------------------
# retranscribe — live (writes manifest)
# ---------------------------------------------------------------------------

class TestRetranscribeLive:
    def test_updates_raw_transcription(self, tmp_path):
        record = _make_record(tmp_path, raw="old text", corrected="Old text.")
        manifest = _make_manifest(tmp_path, [record])

        with patch("listenr.retranscribe.lemonade_transcribe_audio", return_value="brand new"):
            retranscribe(manifest)

        loaded = _load_manifest(manifest)
        assert loaded[0]["raw_transcription"] == "brand new"

    def test_corrected_falls_back_to_new_raw_when_no_llm(self, tmp_path):
        record = _make_record(tmp_path, raw="old text", corrected="Old text.")
        manifest = _make_manifest(tmp_path, [record])

        with patch("listenr.retranscribe.lemonade_transcribe_audio", return_value="brand new"):
            retranscribe(manifest, use_llm=False)

        loaded = _load_manifest(manifest)
        assert loaded[0]["corrected_transcription"] == "brand new"

    def test_whisper_model_updated_in_manifest(self, tmp_path):
        record = _make_record(tmp_path, model="Whisper-Tiny")
        manifest = _make_manifest(tmp_path, [record])

        with patch("listenr.retranscribe.lemonade_transcribe_audio", return_value="hello"):
            retranscribe(manifest, model="Whisper-Large-v3-Turbo")

        loaded = _load_manifest(manifest)
        assert loaded[0]["whisper_model"] == "Whisper-Large-v3-Turbo"

    def test_unchanged_record_not_counted_as_updated(self, tmp_path):
        record = _make_record(tmp_path, raw="same text")
        manifest = _make_manifest(tmp_path, [record])

        with patch("listenr.retranscribe.lemonade_transcribe_audio", return_value="same text"):
            summary = retranscribe(manifest)

        assert summary["updated"] == 0

    def test_missing_audio_file_counted_as_error(self, tmp_path):
        record = _make_record(tmp_path)
        record["audio_path"] = str(tmp_path / "nonexistent.wav")
        manifest = _make_manifest(tmp_path, [record])

        with patch("listenr.retranscribe.lemonade_transcribe_audio") as mock_asr:
            summary = retranscribe(manifest)

        mock_asr.assert_not_called()
        assert summary["errors"] == 1
        assert summary["processed"] == 0

    def test_transcription_error_counted_as_error(self, tmp_path):
        record = _make_record(tmp_path)
        manifest = _make_manifest(tmp_path, [record])

        with patch(
            "listenr.retranscribe.lemonade_transcribe_audio",
            side_effect=RuntimeError("timeout"),
        ):
            summary = retranscribe(manifest)

        assert summary["errors"] == 1

    def test_hallucination_result_not_written(self, tmp_path):
        record = _make_record(tmp_path, raw="original text")
        manifest = _make_manifest(tmp_path, [record])

        # A pure brackets-only string is detected as a hallucination
        with patch(
            "listenr.retranscribe.lemonade_transcribe_audio",
            return_value="[BLANK_AUDIO]",
        ):
            retranscribe(manifest)

        loaded = _load_manifest(manifest)
        assert loaded[0]["raw_transcription"] == "original text"

    def test_multiple_records_all_updated(self, tmp_path):
        records = [
            _make_record(tmp_path, uid="aaa111222333", raw="one"),
            _make_record(tmp_path, uid="bbb444555666", raw="two"),
        ]
        manifest = _make_manifest(tmp_path, records)

        with patch(
            "listenr.retranscribe.lemonade_transcribe_audio",
            side_effect=["updated one", "updated two"],
        ):
            summary = retranscribe(manifest)

        assert summary["updated"] == 2
        loaded = _load_manifest(manifest)
        texts = [r["raw_transcription"] for r in loaded]
        assert texts == ["updated one", "updated two"]


# ---------------------------------------------------------------------------
# retranscribe — with LLM
# ---------------------------------------------------------------------------

class TestRetranscribeWithLLM:
    def _llm_result(self, corrected: str, improved: bool = True) -> dict:
        return {
            "corrected": corrected,
            "is_improved": improved,
            "model": "mock-llm",
            "categories": [],
        }

    def test_corrected_transcription_updated_when_improved(self, tmp_path):
        record = _make_record(tmp_path, raw="old text")
        manifest = _make_manifest(tmp_path, [record])

        with (
            patch("listenr.retranscribe.lemonade_transcribe_audio", return_value="new text"),
            patch(
                "listenr.retranscribe.lemonade_llm_correct",
                return_value=self._llm_result("New text.", improved=True),
            ),
        ):
            retranscribe(manifest, use_llm=True)

        loaded = _load_manifest(manifest)
        assert loaded[0]["corrected_transcription"] == "New text."
        assert loaded[0]["is_improved"] is True
        assert loaded[0]["llm_model"] == "mock-llm"

    def test_llm_model_none_when_not_improved(self, tmp_path):
        record = _make_record(tmp_path, raw="old text")
        manifest = _make_manifest(tmp_path, [record])

        with (
            patch("listenr.retranscribe.lemonade_transcribe_audio", return_value="new text"),
            patch(
                "listenr.retranscribe.lemonade_llm_correct",
                return_value=self._llm_result("new text", improved=False),
            ),
        ):
            retranscribe(manifest, use_llm=True)

        loaded = _load_manifest(manifest)
        assert loaded[0]["llm_model"] is None

    def test_llm_error_does_not_prevent_raw_update(self, tmp_path):
        record = _make_record(tmp_path, raw="old text")
        manifest = _make_manifest(tmp_path, [record])

        with (
            patch("listenr.retranscribe.lemonade_transcribe_audio", return_value="new text"),
            patch(
                "listenr.retranscribe.lemonade_llm_correct",
                side_effect=RuntimeError("server down"),
            ),
        ):
            summary = retranscribe(manifest, use_llm=True)

        loaded = _load_manifest(manifest)
        assert loaded[0]["raw_transcription"] == "new text"
        assert summary["updated"] == 1

    def test_llm_error_response_does_not_prevent_raw_update(self, tmp_path):
        record = _make_record(tmp_path, raw="old text")
        manifest = _make_manifest(tmp_path, [record])

        with (
            patch("listenr.retranscribe.lemonade_transcribe_audio", return_value="new text"),
            patch(
                "listenr.retranscribe.lemonade_llm_correct",
                return_value={"error": "model not loaded"},
            ),
        ):
            summary = retranscribe(manifest, use_llm=True)

        loaded = _load_manifest(manifest)
        assert loaded[0]["raw_transcription"] == "new text"
        assert summary["updated"] == 1


# ---------------------------------------------------------------------------
# retranscribe — filters
# ---------------------------------------------------------------------------

class TestRetranscribeFilters:
    def test_uuid_filter_only_processes_matching(self, tmp_path):
        records = [
            _make_record(tmp_path, uid="aaa111222333", raw="one"),
            _make_record(tmp_path, uid="bbb444555666", raw="two"),
        ]
        manifest = _make_manifest(tmp_path, records)

        with patch(
            "listenr.retranscribe.lemonade_transcribe_audio", return_value="updated"
        ):
            summary = retranscribe(manifest, uuids={"aaa111222333"})

        assert summary["processed"] == 1
        assert summary["skipped"] == 1
        loaded = _load_manifest(manifest)
        assert loaded[0]["raw_transcription"] == "updated"
        assert loaded[1]["raw_transcription"] == "two"

    def test_regex_filter_only_processes_matching(self, tmp_path):
        records = [
            _make_record(tmp_path, uid="aaa111222333", raw="hello world"),
            _make_record(tmp_path, uid="bbb444555666", raw="goodbye world"),
        ]
        manifest = _make_manifest(tmp_path, records)

        with patch(
            "listenr.retranscribe.lemonade_transcribe_audio", return_value="updated"
        ):
            summary = retranscribe(
                manifest, pattern=re.compile("hello", re.IGNORECASE)
            )

        assert summary["processed"] == 1
        loaded = _load_manifest(manifest)
        assert loaded[0]["raw_transcription"] == "updated"
        assert loaded[1]["raw_transcription"] == "goodbye world"

    def test_summary_counts_are_consistent(self, tmp_path):
        records = [
            _make_record(tmp_path, uid="aaa111222333", raw="hello there"),
            _make_record(tmp_path, uid="bbb444555666", raw="goodbye world"),
            _make_record(tmp_path, uid="ccc777888999", raw="hello again"),
        ]
        manifest = _make_manifest(tmp_path, records)

        with patch(
            "listenr.retranscribe.lemonade_transcribe_audio", return_value="new"
        ):
            summary = retranscribe(
                manifest, pattern=re.compile("^hello", re.IGNORECASE)
            )

        assert summary["total"] == 3
        assert summary["processed"] == 2
        assert summary["skipped"] == 1
