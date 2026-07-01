"""Unit tests for the source importers and their shared manifest core."""

import io
import json
import sys
import types
from pathlib import Path

import numpy as np
import soundfile as sf

from listenr.importers import manifest as m
from listenr.importers.mapping import FieldMapping


def _wav(path: Path, duration_s: float = 1.5, sample_rate: int = 16_000) -> Path:
    samples = np.zeros(int(duration_s * sample_rate), dtype="float32")
    sf.write(str(path), samples, sample_rate)
    return path


# ---------------------------------------------------------------------------
# FieldMapping
# ---------------------------------------------------------------------------


class TestFieldMapping:
    def test_overrides_take_priority_as_sole_candidate(self):
        mapping = FieldMapping().with_overrides(audio="path", text="caption")
        assert mapping.audio == ("path",)
        assert mapping.text == ("caption",)
        # Unset overrides leave defaults untouched.
        assert mapping.split == ("split",)

    def test_none_overrides_are_ignored(self):
        base = FieldMapping()
        assert base.with_overrides(audio=None, text=None) == base


# ---------------------------------------------------------------------------
# records_from_rows — file-based audio (MDC shape)
# ---------------------------------------------------------------------------


class TestRecordsFromFileRows:
    def test_builds_records_from_paths(self, tmp_path):
        audio_path = _wav(tmp_path / "clip.wav")
        rows = [{"audio_path": str(audio_path), "transcription": "hello", "split": "train"}]

        records, skipped = m.records_from_rows(
            rows,
            source="mdc",
            dataset_id="ds-1",
            mapping=FieldMapping(),
            audio_dir=tmp_path / "audio",
            dataset_name="Sample",
        )

        assert skipped == 0
        assert records[0]["audio_path"] == str(audio_path.resolve())
        assert records[0]["raw_transcription"] == "hello"
        assert records[0]["corrected_transcription"] == "hello"
        assert records[0]["source"] == "mdc"
        assert records[0]["source_dataset_id"] == "ds-1"
        assert records[0]["source_dataset_name"] == "Sample"
        assert records[0]["source_split"] == "train"
        assert records[0]["sample_rate"] == 16_000
        assert records[0]["duration_s"] == 1.5

    def test_skips_missing_file_and_missing_text(self, tmp_path):
        audio_path = _wav(tmp_path / "clip.wav")
        rows = [
            {"audio_path": str(audio_path), "transcription": "ok"},
            {"audio_path": str(tmp_path / "missing.wav"), "transcription": "gone"},
            {"audio_path": str(audio_path), "transcription": ""},
        ]

        records, skipped = m.records_from_rows(
            rows, source="mdc", dataset_id="ds", mapping=FieldMapping(), audio_dir=tmp_path
        )

        assert len(records) == 1
        assert skipped == 2

    def test_prefers_column_duration_and_sample_rate(self, tmp_path):
        audio_path = _wav(tmp_path / "clip.wav", duration_s=2.0, sample_rate=8_000)
        rows = [{"audio_path": str(audio_path), "transcription": "hi", "duration_ms": 750, "sample_rate": 22_050}]

        records, _ = m.records_from_rows(
            rows, source="mdc", dataset_id="ds", mapping=FieldMapping(), audio_dir=tmp_path
        )

        assert records[0]["duration_s"] == 0.75
        assert records[0]["sample_rate"] == 22_050


# ---------------------------------------------------------------------------
# records_from_rows — in-memory audio (array / bytes) gets materialised to WAV
# ---------------------------------------------------------------------------


class TestRecordsFromInMemoryAudio:
    def test_materialises_audio_array_to_wav(self, tmp_path):
        array = np.zeros(8_000, dtype="float32")
        rows = [{"audio": {"array": array, "sampling_rate": 16_000, "path": "clip.mp3"}, "sentence": "hola"}]
        audio_dir = tmp_path / "audio"

        records, skipped = m.records_from_rows(
            rows,
            source="hf",
            dataset_id="cv/en",
            mapping=FieldMapping(audio=("audio",)),
            audio_dir=audio_dir,
        )

        assert skipped == 0
        written = Path(records[0]["audio_path"])
        assert written.exists() and written.parent == audio_dir.resolve()
        assert records[0]["raw_transcription"] == "hola"
        assert records[0]["sample_rate"] == 16_000
        assert records[0]["duration_s"] == 0.5

    def test_materialises_audio_bytes_to_wav(self, tmp_path):
        buf = io.BytesIO()
        sf.write(buf, np.zeros(4_000, dtype="float32"), 8_000, format="WAV")
        rows = [{"audio": {"bytes": buf.getvalue(), "path": None}, "sentence": "bonjour"}]

        records, skipped = m.records_from_rows(
            rows, source="hf", dataset_id="ds", mapping=FieldMapping(audio=("audio",)), audio_dir=tmp_path
        )

        assert skipped == 0
        assert Path(records[0]["audio_path"]).exists()
        assert records[0]["sample_rate"] == 8_000


# ---------------------------------------------------------------------------
# write_manifest
# ---------------------------------------------------------------------------


class TestWriteManifest:
    def test_writes_and_appends(self, tmp_path):
        manifest_path = tmp_path / "manifest.jsonl"
        m.write_manifest([{"uuid": "a"}], manifest_path)
        m.write_manifest([{"uuid": "b"}], manifest_path, append=True)

        lines = manifest_path.read_text(encoding="utf-8").splitlines()
        assert [json.loads(line)["uuid"] for line in lines] == ["a", "b"]


# ---------------------------------------------------------------------------
# Source loaders with mocked SDKs (no network)
# ---------------------------------------------------------------------------


class TestMDCLoader:
    def test_import_mdc_dataset_writes_manifest(self, tmp_path, monkeypatch):
        import pandas as pd

        audio_path = _wav(tmp_path / "clip.wav")
        fake = types.ModuleType("datacollective")
        fake.load_dataset = lambda dataset_id, **kw: pd.DataFrame(
            [{"audio_path": str(audio_path), "transcription": "hello mdc", "split": "train"}]
        )
        fake.get_dataset_details = lambda dataset_id: {"name": "Demo"}  # no enable_logging kwarg
        monkeypatch.setitem(sys.modules, "datacollective", fake)

        from listenr.importers import mdc

        out = tmp_path / "manifest.jsonl"
        summary = mdc.import_mdc_dataset("demo-1", manifest_path=out)

        assert summary["imported"] == 1
        assert summary["dataset_name"] == "Demo"
        record = json.loads(out.read_text().strip())
        assert record["source"] == "mdc"
        assert record["raw_transcription"] == "hello mdc"
