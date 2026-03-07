"""
Unit tests for build_dataset — load_manifest, validate_entry, parse_split,
assign_splits, write_csv, and the --remap-audio-prefix feature.

Run with:  python -m pytest tests/test_build_dataset.py
"""

import csv
import json
import numpy as np
import soundfile as sf
import pytest
from pathlib import Path

from listenr.build_dataset import (
    load_manifest,
    validate_entry,
    parse_split,
    assign_splits,
    write_csv,
    CSV_COLUMNS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wav(path: Path, duration_s: float = 2.0, sample_rate: int = 16_000) -> Path:
    """Write a silent WAV file and return its path."""
    samples = np.zeros(int(duration_s * sample_rate), dtype="float32")
    sf.write(str(path), samples, sample_rate)
    return path


def _record(
    tmp_path: Path,
    uuid: str = "abc123",
    raw: str = "hello world",
    corrected: str = "hello world",
    duration_s: float = 2.0,
    audio_name: str = "clip.wav",
    create_file: bool = True,
) -> dict:
    """Return a manifest record with an optional real WAV file."""
    audio_path = tmp_path / audio_name
    if create_file:
        _wav(audio_path, duration_s)
    return {
        "uuid": uuid,
        "raw_transcription": raw,
        "corrected_transcription": corrected,
        "audio_path": str(audio_path),
        "duration_s": duration_s,
        "sample_rate": 16000,
        "whisper_model": "whisper-base",
        "llm_model": "gpt-oss",
        "is_improved": "True",
        "timestamp": "2025-01-01T00:00:00",
    }


# ---------------------------------------------------------------------------
# load_manifest
# ---------------------------------------------------------------------------

class TestLoadManifest:
    def test_returns_empty_list_when_file_missing(self, tmp_path):
        result = load_manifest(tmp_path / "no_such_file.jsonl")
        assert result == []

    def test_loads_valid_jsonl(self, tmp_path):
        mf = tmp_path / "manifest.jsonl"
        records = [{"uuid": "a"}, {"uuid": "b"}]
        mf.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        loaded = load_manifest(mf)
        assert len(loaded) == 2
        assert loaded[0]["uuid"] == "a"
        assert loaded[1]["uuid"] == "b"

    def test_skips_blank_lines(self, tmp_path):
        mf = tmp_path / "manifest.jsonl"
        mf.write_text('{"uuid":"a"}\n\n{"uuid":"b"}\n')
        assert len(load_manifest(mf)) == 2

    def test_skips_malformed_json_lines(self, tmp_path):
        mf = tmp_path / "manifest.jsonl"
        mf.write_text('{"uuid":"a"}\nNOT_JSON\n{"uuid":"b"}\n')
        loaded = load_manifest(mf)
        assert len(loaded) == 2

    def test_empty_file_returns_empty_list(self, tmp_path):
        mf = tmp_path / "manifest.jsonl"
        mf.write_text("")
        assert load_manifest(mf) == []


# ---------------------------------------------------------------------------
# validate_entry
# ---------------------------------------------------------------------------

class TestValidateEntryValid:
    def test_returns_dict_for_valid_record(self, tmp_path):
        rec = _record(tmp_path)
        entry = validate_entry(rec, min_duration=1.0, min_chars=3)
        assert entry is not None
        assert entry["uuid"] == "abc123"
        assert entry["raw_transcription"] == "hello world"

    def test_corrected_transcription_preserved(self, tmp_path):
        rec = _record(tmp_path, corrected="Hello World.")
        entry = validate_entry(rec, min_duration=1.0, min_chars=3)
        assert entry["corrected_transcription"] == "Hello World."

    def test_corrected_falls_back_to_raw_when_empty(self, tmp_path):
        rec = _record(tmp_path, corrected="")
        entry = validate_entry(rec, min_duration=1.0, min_chars=3)
        assert entry["corrected_transcription"] == "hello world"

    def test_is_improved_parsed_as_bool(self, tmp_path):
        rec = _record(tmp_path)
        rec["is_improved"] = "True"
        entry = validate_entry(rec, min_duration=1.0, min_chars=3)
        assert entry["is_improved"] is True

    def test_is_improved_false_string(self, tmp_path):
        rec = _record(tmp_path)
        rec["is_improved"] = "False"
        entry = validate_entry(rec, min_duration=1.0, min_chars=3)
        assert entry["is_improved"] is False

    def test_audio_path_resolved_to_absolute(self, tmp_path):
        rec = _record(tmp_path)
        entry = validate_entry(rec, min_duration=1.0, min_chars=3)
        assert Path(entry["audio_path"]).is_absolute()


class TestValidateEntryRejects:
    def test_missing_uuid_returns_none(self, tmp_path):
        rec = _record(tmp_path)
        del rec["uuid"]
        assert validate_entry(rec, min_duration=0.0, min_chars=0) is None

    def test_empty_uuid_returns_none(self, tmp_path):
        rec = _record(tmp_path)
        rec["uuid"] = ""
        assert validate_entry(rec, min_duration=0.0, min_chars=0) is None

    def test_missing_raw_transcription_returns_none(self, tmp_path):
        rec = _record(tmp_path)
        del rec["raw_transcription"]
        assert validate_entry(rec, min_duration=0.0, min_chars=0) is None

    def test_missing_audio_path_field_returns_none(self, tmp_path):
        rec = _record(tmp_path)
        del rec["audio_path"]
        assert validate_entry(rec, min_duration=0.0, min_chars=0) is None

    def test_duration_below_minimum_returns_none(self, tmp_path):
        rec = _record(tmp_path, duration_s=0.3)
        assert validate_entry(rec, min_duration=0.5, min_chars=0) is None

    def test_duration_exactly_at_minimum_passes(self, tmp_path):
        rec = _record(tmp_path, duration_s=0.5)
        assert validate_entry(rec, min_duration=0.5, min_chars=0) is not None

    def test_transcript_too_short_returns_none(self, tmp_path):
        rec = _record(tmp_path, raw="hi")  # 2 non-whitespace chars
        assert validate_entry(rec, min_duration=0.0, min_chars=5) is None

    def test_missing_audio_file_returns_none(self, tmp_path):
        rec = _record(tmp_path, create_file=False)
        assert validate_entry(rec, min_duration=0.0, min_chars=0) is None


class TestValidateEntryTagStripping:
    def test_noise_tags_stripped_by_default(self, tmp_path):
        rec = _record(tmp_path, raw="(music) hello world")
        entry = validate_entry(rec, min_duration=0.0, min_chars=3, strip_tags=True)
        assert entry is not None
        assert "(music)" not in entry["raw_transcription"]

    def test_noise_tags_preserved_when_strip_false(self, tmp_path):
        rec = _record(tmp_path, raw="(music) hello world")
        entry = validate_entry(rec, min_duration=0.0, min_chars=3, strip_tags=False)
        assert entry is not None
        assert "(music)" in entry["raw_transcription"]

    def test_pure_noise_tag_rejected_after_strip(self, tmp_path):
        # "(music)" alone → after stripping → empty → too short → None
        rec = _record(tmp_path, raw="(music)")
        assert validate_entry(rec, min_duration=0.0, min_chars=3, strip_tags=True) is None

    def test_pure_noise_tag_accepted_without_strip(self, tmp_path):
        # With strip_tags=False the 7 chars of "(music)" pass min_chars=3
        rec = _record(tmp_path, raw="(music)")
        assert validate_entry(rec, min_duration=0.0, min_chars=3, strip_tags=False) is not None


# ---------------------------------------------------------------------------
# parse_split
# ---------------------------------------------------------------------------

class TestParseSplit:
    def test_equal_thirds(self):
        train, dev, test = parse_split("80/10/10")
        assert abs(train - 0.8) < 1e-9
        assert abs(dev - 0.1) < 1e-9
        assert abs(test - 0.1) < 1e-9

    def test_fractions_sum_to_one(self):
        fracs = parse_split("70/15/15")
        assert abs(sum(fracs) - 1.0) < 1e-9

    def test_non_percentage_ratio_normalised(self):
        train, dev, test = parse_split("1/1/1")
        assert abs(train - 1 / 3) < 1e-9

    def test_wrong_number_of_parts_raises(self):
        with pytest.raises(ValueError):
            parse_split("80/20")

    def test_zero_total_raises(self):
        with pytest.raises(ValueError):
            parse_split("0/0/0")

    def test_non_numeric_raises(self):
        with pytest.raises(ValueError):
            parse_split("a/b/c")


# ---------------------------------------------------------------------------
# assign_splits
# ---------------------------------------------------------------------------

class TestAssignSplits:
    def _make_entries(self, n: int) -> list[dict]:
        return [{"uuid": str(i)} for i in range(n)]

    def test_all_entries_assigned_a_split(self):
        entries = assign_splits(self._make_entries(10), 0.8, 0.1)
        assert all("split" in e for e in entries)
        assert all(e["split"] in ("train", "dev", "test") for e in entries)

    def test_split_counts_approximate_fractions(self):
        entries = assign_splits(self._make_entries(100), 0.8, 0.1, seed=0)
        counts = {"train": 0, "dev": 0, "test": 0}
        for e in entries:
            counts[e["split"]] += 1
        assert counts["train"] == 80
        assert counts["dev"] == 10
        assert counts["test"] == 10

    def test_deterministic_with_same_seed(self):
        a = assign_splits(self._make_entries(20), 0.8, 0.1, seed=42)
        b = assign_splits(self._make_entries(20), 0.8, 0.1, seed=42)
        assert [e["split"] for e in a] == [e["split"] for e in b]

    def test_different_seeds_differ(self):
        a = assign_splits(self._make_entries(20), 0.8, 0.1, seed=1)
        b = assign_splits(self._make_entries(20), 0.8, 0.1, seed=2)
        # The label sequence is always 16×train, 2×dev, 2×test — but which
        # UUID ends up in which bucket differs by seed; compare those sets.
        def _train_uuids(result):
            return {e["uuid"] for e in result if e["split"] == "train"}
        assert _train_uuids(a) != _train_uuids(b)

    def test_does_not_mutate_original_list(self):
        orig = self._make_entries(10)
        ids_before = [e["uuid"] for e in orig]
        assign_splits(orig, 0.8, 0.1)
        assert [e["uuid"] for e in orig] == ids_before

    def test_single_entry_assigned_test(self):
        entries = assign_splits([{"uuid": "x"}], 0.8, 0.1)
        assert entries[0]["split"] == "test"


# ---------------------------------------------------------------------------
# write_csv
# ---------------------------------------------------------------------------

class TestWriteCsv:
    def _sample_entries(self, tmp_path: Path, n: int = 3) -> list[dict]:
        entries = []
        for i in range(n):
            wav = tmp_path / f"clip_{i}.wav"
            _wav(wav)
            entries.append({
                "uuid": f"u{i}",
                "split": "train" if i < 2 else "test",
                "audio_path": str(wav),
                "raw_transcription": f"text {i}",
                "corrected_transcription": f"text {i}",
                "is_improved": False,
                "duration_s": 2.0,
                "sample_rate": 16000,
                "whisper_model": "base",
                "llm_model": "",
                "timestamp": "2025-01-01T00:00:00",
            })
        return entries

    def test_creates_csv_file(self, tmp_path):
        entries = self._sample_entries(tmp_path)
        out = write_csv(entries, tmp_path, "train")
        assert out.exists()
        assert out.name == "train.csv"

    def test_csv_has_correct_columns(self, tmp_path):
        entries = self._sample_entries(tmp_path)
        out = write_csv(entries, tmp_path, "train")
        with open(out, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert list(reader.fieldnames) == CSV_COLUMNS

    def test_csv_contains_only_requested_split(self, tmp_path):
        entries = self._sample_entries(tmp_path)  # 2 train, 1 test
        out = write_csv(entries, tmp_path, "train")
        with open(out, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert all(r["split"] == "train" for r in rows)

    def test_empty_split_writes_header_only(self, tmp_path):
        entries = self._sample_entries(tmp_path)
        # No "dev" entries in the sample
        out = write_csv(entries, tmp_path, "dev")
        with open(out, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows == []

    def test_csv_row_values_match_entries(self, tmp_path):
        entries = self._sample_entries(tmp_path)
        out = write_csv(entries, tmp_path, "train")
        with open(out, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        uuids = {r["uuid"] for r in rows}
        assert uuids == {"u0", "u1"}


# ---------------------------------------------------------------------------
# remap_audio_prefix logic  (extracted from main())
# ---------------------------------------------------------------------------

class TestRemapAudioPrefix:
    """Test the path-rewrite logic used by --remap-audio-prefix."""

    def _remap(self, records: list[dict], old: str, new: str) -> list[dict]:
        """Apply the same remap loop as main()."""
        import copy
        records = copy.deepcopy(records)
        for rec in records:
            p = rec.get("audio_path", "")
            if p.startswith(old):
                rec["audio_path"] = new + p[len(old):]
        return records

    def test_prefix_replaced(self):
        records = [{"audio_path": "/host/audio/clip.wav"}]
        result = self._remap(records, "/host/audio", "/container/audio")
        assert result[0]["audio_path"] == "/container/audio/clip.wav"

    def test_only_matching_prefix_replaced(self):
        records = [
            {"audio_path": "/host/audio/a.wav"},
            {"audio_path": "/other/path/b.wav"},
        ]
        result = self._remap(records, "/host/audio", "/container/audio")
        assert result[0]["audio_path"] == "/container/audio/a.wav"
        assert result[1]["audio_path"] == "/other/path/b.wav"

    def test_nested_path_preserved_after_prefix(self):
        records = [{"audio_path": "/host/audio/2025/01/clip.wav"}]
        result = self._remap(records, "/host/audio", "/data")
        assert result[0]["audio_path"] == "/data/2025/01/clip.wav"

    def test_no_trailing_slash_double_slash(self):
        """Ensure no double-slash when old path ends with slash."""
        records = [{"audio_path": "/host/audio/clip.wav"}]
        result = self._remap(records, "/host/audio/", "/container/audio/")
        # "/host/audio/" prefix + "clip.wav" suffix → "/container/audio/clip.wav"
        assert result[0]["audio_path"] == "/container/audio/clip.wav"

    def test_empty_audio_path_unaffected(self):
        records = [{"audio_path": ""}]
        result = self._remap(records, "/host", "/container")
        assert result[0]["audio_path"] == ""

    def test_original_records_not_mutated(self):
        records = [{"audio_path": "/host/clip.wav"}]
        self._remap(records, "/host", "/container")
        # Original should be unchanged (we deepcopy in helper)
        assert records[0]["audio_path"] == "/host/clip.wav"
