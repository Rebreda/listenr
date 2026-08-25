"""Tests for listenr.finetune.report (run records — no torch needed)."""

import json

from listenr.finetune.report import (
    eval_clip_record,
    eval_report,
    finetune_run_record,
    write_json,
)


class TestWriteJson:
    def test_creates_parents_and_roundtrips(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "results.json"
        write_json(path, {"a": 1})
        assert json.loads(path.read_text()) == {"a": 1}

    def test_atomic_no_tmp_left_behind(self, tmp_path):
        path = tmp_path / "results.json"
        write_json(path, {"a": 1})
        assert list(tmp_path.iterdir()) == [path]

    def test_overwrites_previous_record(self, tmp_path):
        path = tmp_path / "results.json"
        write_json(path, {"run": 1})
        write_json(path, {"run": 2})
        assert json.loads(path.read_text()) == {"run": 2}

    def test_unicode_survives(self, tmp_path):
        # ensure_ascii=False: transcripts are prose, not escape sequences.
        path = tmp_path / "results.json"
        write_json(path, {"text": "naïve café"})
        assert "naïve café" in path.read_text()


class TestEvalClipRecord:
    ROW = {
        "uuid": "abc123",
        "audio_path": "/data/clip.wav",
        "duration_s": 4.2,
        "corrected_transcription": "I use Claude daily",
        "split": "test",
    }

    def test_carries_identity_and_reference(self):
        rec = eval_clip_record(self.ROW, {"fine_tuned": "I use Claude daily"})
        assert rec["uuid"] == "abc123"
        assert rec["audio_path"] == "/data/clip.wav"
        assert rec["reference"] == "I use Claude daily"
        assert rec["hypotheses"] == {"fine_tuned": "I use Claude daily"}

    def test_keyword_maps_optional(self):
        rec = eval_clip_record(self.ROW, {"fine_tuned": "x"})
        assert "keywords" not in rec
        rec = eval_clip_record(
            self.ROW, {"fine_tuned": "x"}, {"fine_tuned": {"Claude": False}}
        )
        assert rec["keywords"] == {"fine_tuned": {"Claude": False}}

    def test_missing_optional_row_fields(self):
        rec = eval_clip_record({"audio_path": "/a.wav"}, {"base": "hello"})
        assert rec["uuid"] is None
        assert rec["reference"] is None


class TestEvalReport:
    def test_shape_and_stamps(self):
        rep = eval_report(
            mode="split",
            model="/data/merged",
            base_model="openai/whisper-tiny",
            dataset="/data/hf_dataset",
            split="test",
            n_requested=50,
            keywords=["Claude"],
            wer_pct={"base": 37.5, "fine_tuned": 19.5},
            keyword_recall={"fine_tuned": {"Claude": [4, 5]}},
            clips=[{"uuid": "x"}],
        )
        assert rep["listenr_version"]
        assert rep["created_utc"]
        assert rep["wer_normalization"] == "whisper_english"
        assert rep["n_evaluated"] == 1
        # tally pairs become named fields: a results file should not require
        # knowing that index 0 means hits.
        assert rep["keyword_recall"]["fine_tuned"]["Claude"] == {
            "hits": 4,
            "expected": 5,
        }

    def test_none_base_model(self):
        rep = eval_report(
            mode="split", model="/m", base_model=None, dataset=None, split="test",
            n_requested=1, keywords=[], wer_pct={}, keyword_recall={}, clips=[],
        )
        assert rep["base_model"] is None
        assert rep["dataset"] is None

    def test_json_serializable(self, tmp_path):
        rep = eval_report(
            mode="single", model="/m", base_model="b", dataset=None, split=None,
            n_requested=None, keywords=[], wer_pct={"fine_tuned": 12.3},
            keyword_recall={}, clips=[],
        )
        write_json(tmp_path / "r.json", rep)  # must not raise


class TestFinetuneRunRecord:
    def kwargs(self, **over):
        base = dict(
            args={"lora_r": 32, "output": "/data/adapter"},
            base_model="openai/whisper-tiny",
            architecture="whisper",
            dataset="/data/hf_dataset",
            dataset_splits={"train": 1382, "dev": 385, "test": 337},
            trainable_params=1_277_952,
            total_params=39_923_328,
            accelerator="GPU 0: AMD Radeon Graphics",
        )
        base.update(over)
        return base

    def test_starts_incomplete(self):
        rec = finetune_run_record(**self.kwargs())
        assert rec["status"] == "started"
        assert rec["ended_utc"] is None
        assert rec["started_utc"]

    def test_trainable_pct(self):
        rec = finetune_run_record(**self.kwargs())
        assert rec["trainable_pct"] == 3.2

    def test_zero_total_params_no_division_error(self):
        rec = finetune_run_record(**self.kwargs(trainable_params=0, total_params=0))
        assert rec["trainable_pct"] is None

    def test_paths_in_args_serialize(self, tmp_path):
        from pathlib import Path

        rec = finetune_run_record(**self.kwargs(args={"output": Path("/data/adapter")}))
        write_json(tmp_path / "run.json", rec)
        saved = json.loads((tmp_path / "run.json").read_text())
        assert saved["args"]["output"] == "/data/adapter"
