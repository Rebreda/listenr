"""Unit tests for listenr.categorize (embedding logic mocked — no model needed)."""

import numpy as np

from listenr.categorize import (
    _load_topics,
    _text,
    cached_encoder,
    filter_records,
    score_records,
)


def fake_encode(texts):
    """Deterministic 2-D unit vectors: tech-ish text/topics -> [1,0], else [0,1]."""
    tech_terms = ("tech", "software", "ai", "computer", "artificial intelligence")
    vecs = []
    for t in texts:
        low = t.lower()
        vecs.append([1.0, 0.0] if any(term in low for term in tech_terms) else [0.0, 1.0])
    return np.asarray(vecs)


class TestText:
    def test_prefers_corrected_then_raw(self):
        assert _text({"corrected_transcription": "c", "raw_transcription": "r"}) == "c"
        assert _text({"raw_transcription": "r"}) == "r"
        assert _text({"raw_transcription": "r"}, text_field="corrected_transcription") == ""


class TestLoadTopics:
    def test_combines_args_and_file(self, tmp_path):
        f = tmp_path / "topics.txt"
        f.write_text("# a comment\nrobotics\n\nmachine learning\n", encoding="utf-8")
        assert _load_topics(["technology"], f) == ["technology", "robotics", "machine learning"]


class TestScoreAndFilter:
    def _records(self):
        return [
            {"uuid": "1", "raw_transcription": "I love software and AI"},
            {"uuid": "2", "raw_transcription": "My favourite sport is hiking"},
            {"uuid": "3", "raw_transcription": "new computer chips are fast"},
        ]

    def test_scores_pick_best_topic(self):
        scored = score_records(self._records(), ["technology", "AI"], fake_encode)
        by_uuid = {r["uuid"]: r["topic_score"] for r in scored}
        assert by_uuid["1"] == 1.0  # tech text vs tech topic -> cosine 1
        assert by_uuid["2"] == 0.0  # non-tech
        assert by_uuid["3"] == 1.0

    def test_score_records_annotates_copies(self):
        records = self._records()
        scored = score_records(records, ["technology"], fake_encode)
        assert all("topic_score" in r and "category" in r for r in scored)
        assert "topic_score" not in records[0]  # originals untouched

    def test_filter_keeps_only_matches_and_flags(self):
        scored = score_records(self._records(), ["technology"], fake_encode)
        kept = filter_records(scored, threshold=0.5)
        assert [r["uuid"] for r in kept] == ["1", "3"]
        assert all(r["topic_matched"] and r["category"] == "technology" for r in kept)

    def test_keep_all_flags_non_matches(self):
        scored = score_records(self._records(), ["technology"], fake_encode)
        kept = filter_records(scored, threshold=0.5, keep_all=True)
        assert [r["uuid"] for r in kept] == ["1", "2", "3"]
        assert next(r for r in kept if r["uuid"] == "2")["topic_matched"] is False

    def test_empty_records(self):
        assert score_records([], ["technology"], fake_encode) == []


# ---------------------------------------------------------------------------
# cached_encoder
# ---------------------------------------------------------------------------


class TestCachedEncoder:
    def _counting(self):
        calls = []

        def encode(texts):
            calls.append(list(texts))
            return fake_encode(texts)

        return encode, calls

    def test_only_embeds_cache_misses(self, tmp_path):
        encode, calls = self._counting()
        enc = cached_encoder(encode, tmp_path / "emb.npz")
        enc(["software"])
        enc(["software", "hiking"])  # only "hiking" is new
        assert calls == [["software"], ["hiking"]]

    def test_persists_across_reload(self, tmp_path):
        cache_path = tmp_path / "emb.npz"
        encode, calls = self._counting()
        out1 = cached_encoder(encode, cache_path)(["software", "hiking"])
        assert cache_path.exists()

        encode2, calls2 = self._counting()
        out2 = cached_encoder(encode2, cache_path)(["software", "hiking"])
        np.testing.assert_array_equal(out1, out2)
        assert calls2 == []  # everything served from the reloaded cache
