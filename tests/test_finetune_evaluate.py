"""Tests for listenr.finetune.evaluate (the `listenr eval` command)."""

import json
from pathlib import Path

import pytest

from listenr.finetune.evaluate import (
    compute_wer,
    keyword_hit_map,
    keyword_hits,
    resolve_base_model,
    select_examples,
    tally_keywords,
)


# ---------------------------------------------------------------------------
# keyword matching
# ---------------------------------------------------------------------------

class TestKeywordHits:
    def test_case_insensitive(self):
        assert keyword_hits("I use Claude Code daily", ["claude"]) == ["claude"]

    def test_word_boundary_prefix(self):
        # "ai" must not match inside "said"
        assert keyword_hits("he said hello", ["ai"]) == []

    def test_suffix_still_matches(self):
        assert keyword_hits("the robotics lab", ["robot"]) == ["robot"]

    def test_multiple_keywords(self):
        assert keyword_hits("Claude and Cursor", ["Claude", "Cursor", "vim"]) == [
            "Claude",
            "Cursor",
        ]


class TestKeywordHitMap:
    def test_hit_and_miss(self):
        hit_map = keyword_hit_map(
            reference="tell Claude to open Cursor",
            hypothesis="tell cloud to open Cursor",
            keywords=["Claude", "Cursor"],
        )
        assert hit_map == {"Claude": False, "Cursor": True}

    def test_only_expected_keywords_tracked(self):
        # "vim" is not in the reference, so it is not scored at all
        hit_map = keyword_hit_map("open the file", "open the file in vim", ["vim"])
        assert hit_map == {}


class TestTallyKeywords:
    def test_aggregates_across_clips_and_models(self):
        results = [
            {"merged": {"Claude": True}, "base": {"Claude": False}},
            {"merged": {"Claude": False}, "base": {"Claude": False}},
            {"merged": {}, "base": {}},
        ]
        tally = tally_keywords(results)
        assert tally["merged"]["Claude"] == [1, 2]
        assert tally["base"]["Claude"] == [0, 2]


# ---------------------------------------------------------------------------
# base-model resolution
# ---------------------------------------------------------------------------

class TestResolveBaseModel:
    def test_override_wins(self, tmp_path):
        assert resolve_base_model(tmp_path, "openai/whisper-tiny") == "openai/whisper-tiny"

    def test_reads_merged_config(self, tmp_path):
        (tmp_path / "config.json").write_text(
            json.dumps({"_name_or_path": "openai/whisper-small"})
        )
        assert resolve_base_model(tmp_path) == "openai/whisper-small"

    def test_local_dir_name_is_rejected(self, tmp_path):
        # _name_or_path pointing back at a local directory is not a base model id
        local = tmp_path / "merged_copy"
        local.mkdir()
        (tmp_path / "config.json").write_text(json.dumps({"_name_or_path": str(local)}))
        from listenr.settings import settings

        assert resolve_base_model(tmp_path) == settings.finetune.base_model

    def test_missing_config_falls_back_to_settings(self, tmp_path):
        from listenr.settings import settings

        assert resolve_base_model(tmp_path) == settings.finetune.base_model

    def test_malformed_config_falls_back(self, tmp_path):
        (tmp_path / "config.json").write_text("{not json")
        from listenr.settings import settings

        assert resolve_base_model(tmp_path) == settings.finetune.base_model


# ---------------------------------------------------------------------------
# example selection
# ---------------------------------------------------------------------------

@pytest.fixture
def rows(tmp_path):
    """Four dataset rows: three with real audio files, one with a missing file."""
    def row(name, text):
        audio = tmp_path / name
        audio.write_bytes(b"fake")
        return {"audio_path": str(audio), "corrected_transcription": text}

    return [
        row("a.wav", "ask Claude a question"),
        row("b.wav", "plain sentence"),
        {"audio_path": str(tmp_path / "missing.wav"), "corrected_transcription": "gone"},
        row("c.wav", ""),
    ]


class TestSelectExamples:
    def test_skips_missing_audio_and_empty_text(self, rows):
        selected = select_examples(rows, n=10)
        assert [Path(r["audio_path"]).name for r in selected] == ["a.wav", "b.wav"]

    def test_keyword_filter(self, rows):
        selected = select_examples(rows, n=10, keywords=["claude"])
        assert len(selected) == 1
        assert "Claude" in selected[0]["corrected_transcription"]

    def test_limit(self, rows):
        assert len(select_examples(rows, n=1)) == 1


# ---------------------------------------------------------------------------
# WER
# ---------------------------------------------------------------------------

class TestComputeWer:
    def test_perfect_match_after_normalization(self):
        pytest.importorskip("jiwer")
        pytest.importorskip("transformers")
        # Case and punctuation differences are removed by Whisper normalization
        assert compute_wer(["Hello, world!"], ["hello world"]) == 0.0

    def test_all_wrong(self):
        pytest.importorskip("jiwer")
        pytest.importorskip("transformers")
        assert compute_wer(["good morning"], ["completely different"]) == 100.0

    def test_empty_refs_return_none(self):
        pytest.importorskip("jiwer")
        pytest.importorskip("transformers")
        assert compute_wer([], []) is None
        assert compute_wer([""], ["something"]) is None
