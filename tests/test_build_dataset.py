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
    load_manifests,
    validate_entry,
    parse_split,
    assign_splits,
    count_source_splits,
    normalise_source_split,
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


class TestLoadManifests:
    def test_combines_records_from_multiple_files(self, tmp_path):
        left = tmp_path / "left.jsonl"
        right = tmp_path / "right.jsonl"
        left.write_text('{"uuid":"a"}\n', encoding="utf-8")
        right.write_text('{"uuid":"b"}\n{"uuid":"c"}\n', encoding="utf-8")

        loaded = load_manifests([left, right])

        assert [row["uuid"] for row in loaded] == ["a", "b", "c"]


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
        entries = assign_splits(self._make_entries(10), 0.8, 0.1)[0]
        assert all("split" in e for e in entries)
        assert all(e["split"] in ("train", "dev", "test") for e in entries)

    def test_split_counts_approximate_fractions(self):
        entries = assign_splits(self._make_entries(100), 0.8, 0.1, seed=0)[0]
        counts = {"train": 0, "dev": 0, "test": 0}
        for e in entries:
            counts[e["split"]] += 1
        assert counts["train"] == 80
        assert counts["dev"] == 10
        assert counts["test"] == 10

    def test_deterministic_with_same_seed(self):
        a = assign_splits(self._make_entries(20), 0.8, 0.1, seed=42)[0]
        b = assign_splits(self._make_entries(20), 0.8, 0.1, seed=42)[0]
        assert [e["split"] for e in a] == [e["split"] for e in b]

    def test_different_seeds_differ(self):
        a = assign_splits(self._make_entries(20), 0.8, 0.1, seed=1)[0]
        b = assign_splits(self._make_entries(20), 0.8, 0.1, seed=2)[0]
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
        entries = assign_splits([{"uuid": "x"}], 0.8, 0.1)[0]
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


# ---------------------------------------------------------------------------
# Source split preservation
# ---------------------------------------------------------------------------


class TestPreserveSourceSplits:
    """Reshuffling an imported corpus breaks speaker-disjoint splits.

    Public corpora keep speakers out of each other's splits on purpose. A
    random reshuffle puts the same voice in train and test, so the model is
    scored on speakers it trained on and the WER gain is partly fake.
    """

    @staticmethod
    def _imported(n, split_name="train"):
        return [{"uuid": f"u{i}", "source_split": split_name} for i in range(n)]

    def test_source_splits_are_kept_by_default(self):
        entries = [
            {"uuid": "a", "source_split": "train"},
            {"uuid": "b", "source_split": "test"},
            {"uuid": "c", "source_split": "validation"},
        ]
        result, how = assign_splits(entries, 0.8, 0.1)
        assert how == "preserved"
        assert [e["split"] for e in result] == ["train", "test", "dev"]

    def test_own_recordings_still_shuffle(self):
        """Records from `listenr record` have no source_split."""
        entries = [{"uuid": f"u{i}"} for i in range(10)]
        result, how = assign_splits(entries, 0.8, 0.1)
        assert how == "shuffled"
        assert all(e["split"] in ("train", "dev", "test") for e in result)

    def test_partial_source_splits_are_still_preserved(self):
        """Real corpora are not uniformly labelled: MDC has 120 of 2,425 blank."""
        entries = [{"uuid": "a", "source_split": "test"}, {"uuid": "b"}]
        result, how = assign_splits(entries, 0.8, 0.1)
        assert how == "preserved"
        assert result[0]["split"] == "test"

    def test_unlabelled_entries_go_to_train_never_test(self):
        """An unlabelled clip must not be able to contaminate the evaluation."""
        entries = [{"uuid": "a", "source_split": "test"}] + [
            {"uuid": f"u{i}"} for i in range(50)
        ]
        result, _ = assign_splits(entries, 0.8, 0.1)
        unlabelled = [e for e in result if e["uuid"] != "a"]
        assert {e["split"] for e in unlabelled} == {"train"}

    def test_no_preserve_splits_forces_a_reshuffle(self):
        entries = self._imported(10)
        _, how = assign_splits(entries, 0.8, 0.1, preserve=False)
        assert how == "shuffled"

    def test_preserve_splits_with_nothing_labelled_is_a_loud_error(self):
        entries = [{"uuid": "a"}, {"uuid": "b"}]
        with pytest.raises(ValueError, match="no entry has a usable"):
            assign_splits(entries, 0.8, 0.1, preserve=True)

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("train", "train"),
            ("Training", "train"),
            ("validation", "dev"),
            ("valid", "dev"),
            ("DEV", "dev"),
            ("test", "test"),
            ("  Test  ", "test"),
            ("holdout", None),
            (None, None),
            (3, None),
        ],
    )
    def test_source_split_names_are_normalised(self, raw, expected):
        assert normalise_source_split(raw) == expected

    def test_count_source_splits(self):
        entries = [{"source_split": "train"}, {"source_split": "bogus"}, {}]
        assert count_source_splits(entries) == (1, 2)

    def test_preserved_splits_do_not_reorder_entries(self):
        """Preserving must not shuffle, or speaker grouping is lost anyway."""
        entries = [{"uuid": f"u{i}", "source_split": "train"} for i in range(20)]
        result, _ = assign_splits(entries, 0.8, 0.1)
        assert [e["uuid"] for e in result] == [f"u{i}" for i in range(20)]


class TestSkipReasons:
    """A bare skip count cannot tell short clips from missing audio."""

    def test_reasons_counter_records_why(self, tmp_path):
        from collections import Counter

        from listenr.build_dataset import validate_entry

        reasons: Counter[str] = Counter()
        # Missing a required field.
        validate_entry({"uuid": "a", "audio_path": "x"}, 0.0, 0, reasons=reasons)
        # Too short.
        wav = tmp_path / "c.wav"
        wav.write_bytes(b"")
        validate_entry(
            {"uuid": "b", "raw_transcription": "hi", "audio_path": str(wav), "duration_s": 0.1},
            min_duration=1.0,
            min_chars=0,
            reasons=reasons,
        )
        # Audio not on disk.
        validate_entry(
            {"uuid": "c", "raw_transcription": "hello there", "audio_path": "/nope.wav", "duration_s": 5},
            min_duration=0.0,
            min_chars=0,
            reasons=reasons,
        )
        assert sum(reasons.values()) == 3
        joined = " ".join(reasons)
        assert "missing field" in joined
        assert "min-duration" in joined
        assert "audio file missing" in joined

    def test_counter_is_optional(self):
        from listenr.build_dataset import validate_entry

        assert validate_entry({"uuid": "a"}, 0.0, 0) is None


class TestSplitsEndToEnd:
    """Runs the real CLI from a manifest file on disk.

    The unit tests for assign_splits passed while split preservation was
    completely unreachable in practice, because they fed it dicts that still
    had source_split. validate_entry rebuilds each record from a whitelist,
    and source_split was not on it, so by the time the preserve logic ran the
    field was always gone. Only a test that starts where a user starts catches
    that.
    """

    @staticmethod
    def _manifest(tmp_path, rows):
        import soundfile as sf

        manifest = tmp_path / "manifest.jsonl"
        with open(manifest, "w") as f:
            for i, source_split in enumerate(rows):
                wav = tmp_path / f"clip{i}.wav"
                sf.write(str(wav), np.zeros(16000, dtype="float32"), 16000, subtype="PCM_16")
                record = {
                    "uuid": f"u{i}",
                    "audio_path": str(wav),
                    "raw_transcription": "a sentence long enough to survive validation",
                    "corrected_transcription": "a sentence long enough to survive validation",
                    "duration_s": 1.0,
                    "sample_rate": 16000,
                }
                if source_split is not None:
                    record["source_split"] = source_split
                f.write(json.dumps(record) + "\n")
        return manifest

    @staticmethod
    def _run(monkeypatch, manifest, output, *extra):
        from listenr.build_dataset import main

        monkeypatch.setattr(
            "sys.argv",
            ["listenr build-dataset", "--manifest", str(manifest),
             "--output", str(output), "--format", "csv", *extra],
        )
        main()

    @staticmethod
    def _counts(output):
        counts = {}
        for split in ("train", "dev", "test"):
            path = output / f"{split}.csv"
            if path.exists():
                rows = list(csv.DictReader(open(path)))
                if rows:
                    counts[split] = len(rows)
        return counts

    def test_corpus_splits_survive_the_whole_pipeline(self, tmp_path, monkeypatch):
        rows = ["train"] * 10 + ["validation"] * 3 + ["test"] * 4
        manifest = self._manifest(tmp_path, rows)
        output = tmp_path / "out"
        self._run(monkeypatch, manifest, output)
        assert self._counts(output) == {"train": 10, "dev": 3, "test": 4}

    def test_preserve_splits_flag_does_not_error_on_a_real_manifest(self, tmp_path, monkeypatch):
        """The 0.2.0 bug: this errored with 'no entry has a usable source_split'."""
        manifest = self._manifest(tmp_path, ["train"] * 8 + ["test"] * 2)
        output = tmp_path / "out"
        self._run(monkeypatch, manifest, output, "--preserve-splits")
        assert self._counts(output) == {"train": 8, "test": 2}

    def test_unlabelled_records_go_to_train(self, tmp_path, monkeypatch):
        manifest = self._manifest(tmp_path, ["train"] * 5 + ["test"] * 2 + [None] * 3)
        output = tmp_path / "out"
        self._run(monkeypatch, manifest, output)
        assert self._counts(output) == {"train": 8, "test": 2}

    def test_no_preserve_splits_reshuffles(self, tmp_path, monkeypatch):
        manifest = self._manifest(tmp_path, ["test"] * 20)
        output = tmp_path / "out"
        self._run(monkeypatch, manifest, output, "--no-preserve-splits", "--split", "80/10/10")
        counts = self._counts(output)
        assert counts.get("train", 0) == 16, counts

    def test_own_recordings_with_no_source_split_still_shuffle(self, tmp_path, monkeypatch):
        manifest = self._manifest(tmp_path, [None] * 20)
        output = tmp_path / "out"
        self._run(monkeypatch, manifest, output, "--split", "80/10/10")
        assert self._counts(output).get("train", 0) == 16

    def test_source_split_is_recorded_in_the_output(self, tmp_path, monkeypatch):
        """So a built dataset says whether its splits are the corpus's own."""
        manifest = self._manifest(tmp_path, ["test"] * 3)
        output = tmp_path / "out"
        self._run(monkeypatch, manifest, output)
        rows = list(csv.DictReader(open(output / "test.csv")))
        assert all(r["source_split"] == "test" for r in rows)
