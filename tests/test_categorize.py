"""Unit tests for listenr.categorize (embedding logic mocked — no model needed)."""

import numpy as np

from listenr.categorize import _load_topics, _text, filter_records, score_records


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
        by_uuid = {rec["uuid"]: (score, topic) for score, topic, rec in scored}
        assert by_uuid["1"][0] == 1.0  # tech text vs tech topic -> cosine 1
        assert by_uuid["2"][0] == 0.0  # non-tech
        assert by_uuid["3"][0] == 1.0

    def test_filter_keeps_only_matches_and_annotates(self):
        scored = score_records(self._records(), ["technology"], fake_encode)
        kept = filter_records(scored, threshold=0.5)
        assert [r["uuid"] for r in kept] == ["1", "3"]
        assert all(r["topic_matched"] and r["category"] == "technology" for r in kept)
        assert all("topic_score" in r for r in kept)

    def test_keep_all_annotates_without_dropping(self):
        scored = score_records(self._records(), ["technology"], fake_encode)
        kept = filter_records(scored, threshold=0.5, keep_all=True)
        assert [r["uuid"] for r in kept] == ["1", "2", "3"]
        non_match = next(r for r in kept if r["uuid"] == "2")
        assert non_match["topic_matched"] is False
        assert non_match["category"] == ""

    def test_empty_records(self):
        assert score_records([], ["technology"], fake_encode) == []
