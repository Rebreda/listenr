"""
Unit tests for listenr.finetune.merge

Covers:
  - read_base_model_id()   — pure JSON parsing, no GPU or network needed
  - merge_adapter()        — tested via mocking; GPU/download not needed
  - dry_run path           — nothing written
  - error paths            — missing dir, bad config
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from listenr.finetune.merge import read_base_model_id, merge_adapter


# ---------------------------------------------------------------------------
# read_base_model_id
# ---------------------------------------------------------------------------

class TestReadBaseModelId:
    def test_reads_model_id_from_valid_config(self, tmp_path):
        (tmp_path / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "openai/whisper-small"})
        )
        assert read_base_model_id(tmp_path) == "openai/whisper-small"

    def test_raises_file_not_found_when_config_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="adapter_config.json not found"):
            read_base_model_id(tmp_path)

    def test_raises_key_error_when_field_missing(self, tmp_path):
        (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 8}))
        with pytest.raises(KeyError, match="base_model_name_or_path"):
            read_base_model_id(tmp_path)

    def test_raises_key_error_when_field_is_empty_string(self, tmp_path):
        (tmp_path / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": ""})
        )
        with pytest.raises(KeyError, match="base_model_name_or_path"):
            read_base_model_id(tmp_path)

    def test_different_model_id(self, tmp_path):
        (tmp_path / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "openai/whisper-large-v3"})
        )
        assert read_base_model_id(tmp_path) == "openai/whisper-large-v3"


# ---------------------------------------------------------------------------
# merge_adapter — dry_run (no imports needed)
# ---------------------------------------------------------------------------

class TestMergeAdapterDryRun:
    def _make_valid_adapter_dir(self, tmp_path: Path) -> Path:
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "openai/whisper-small"})
        )
        return tmp_path

    def test_dry_run_writes_no_files(self, tmp_path):
        adapter_dir = self._make_valid_adapter_dir(tmp_path / "adapter")
        output_dir = tmp_path / "merged"
        merge_adapter(adapter_dir, output_dir, dry_run=True)
        # dry_run must not create the output directory at all
        assert not output_dir.exists()

    def test_dry_run_returns_without_loading_model(self, tmp_path):
        adapter_dir = self._make_valid_adapter_dir(tmp_path / "adapter")
        with patch("listenr.finetune.merge.read_base_model_id",
                   return_value="openai/whisper-small") as mock_rid:
            merge_adapter(adapter_dir, tmp_path / "out", dry_run=True)
            # read_base_model_id is called to validate the config
            mock_rid.assert_called_once_with(adapter_dir.resolve())


# ---------------------------------------------------------------------------
# merge_adapter — adapter dir missing → sys.exit
# ---------------------------------------------------------------------------

class TestMergeAdapterMissingDir:
    def test_exits_when_adapter_dir_missing(self, tmp_path):
        with pytest.raises(SystemExit):
            with patch("listenr.finetune.merge.read_base_model_id",
                       side_effect=FileNotFoundError("not found")):
                merge_adapter(tmp_path / "no_such_dir", tmp_path / "out")


# ---------------------------------------------------------------------------
# merge_adapter — full merge path (mocked transformers + peft)
# ---------------------------------------------------------------------------

class TestMergeAdapterFullMock:
    """
    Test the merge pipeline without a GPU or network by mocking out the heavy
    transformers / peft imports and verifying the call sequence.
    """

    def _make_adapter_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "adapter"
        d.mkdir()
        (d / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "openai/whisper-small"})
        )
        (d / "adapter_model.safetensors").write_bytes(b"\x00" * 8)
        return d

    def _run_merge(self, adapter_dir: Path, output_dir: Path) -> dict:
        """Patch heavy deps and run merge_adapter; return recorded mock objects."""
        mock_base_model = MagicMock()
        mock_merged_model = MagicMock()
        mock_peft_model = MagicMock()
        mock_peft_model.merge_and_unload.return_value = mock_merged_model

        mock_processor = MagicMock()

        mock_peft_cls = MagicMock()
        mock_peft_cls.from_pretrained.return_value = mock_peft_model

        mock_whisper_cls = MagicMock()
        mock_whisper_cls.from_pretrained.return_value = mock_base_model

        mock_processor_cls = MagicMock()
        mock_processor_cls.from_pretrained.return_value = mock_processor

        import sys
        fake_peft = MagicMock()
        fake_peft.PeftModel = mock_peft_cls
        fake_transformers = MagicMock()
        fake_transformers.WhisperForConditionalGeneration = mock_whisper_cls
        fake_transformers.WhisperProcessor = mock_processor_cls

        with patch.dict(sys.modules, {
            "peft": fake_peft,
            "transformers": fake_transformers,
        }):
            merge_adapter(adapter_dir, output_dir, dry_run=False)

        return {
            "base_model": mock_base_model,
            "peft_model": mock_peft_model,
            "merged_model": mock_merged_model,
            "processor": mock_processor,
            "whisper_cls": mock_whisper_cls,
            "peft_cls": mock_peft_cls,
            "processor_cls": mock_processor_cls,
        }

    def test_loads_base_model_with_correct_id(self, tmp_path):
        adapter_dir = self._make_adapter_dir(tmp_path)
        output_dir = tmp_path / "merged"
        mocks = self._run_merge(adapter_dir, output_dir)
        mocks["whisper_cls"].from_pretrained.assert_called_once_with(
            "openai/whisper-small"
        )

    def test_loads_peft_model_from_adapter_dir(self, tmp_path):
        adapter_dir = self._make_adapter_dir(tmp_path)
        output_dir = tmp_path / "merged"
        mocks = self._run_merge(adapter_dir, output_dir)
        mocks["peft_cls"].from_pretrained.assert_called_once_with(
            mocks["base_model"], str(adapter_dir.resolve())
        )

    def test_calls_merge_and_unload(self, tmp_path):
        adapter_dir = self._make_adapter_dir(tmp_path)
        output_dir = tmp_path / "merged"
        mocks = self._run_merge(adapter_dir, output_dir)
        mocks["peft_model"].merge_and_unload.assert_called_once()

    def test_saves_merged_model_to_output_dir(self, tmp_path):
        adapter_dir = self._make_adapter_dir(tmp_path)
        output_dir = tmp_path / "merged"
        mocks = self._run_merge(adapter_dir, output_dir)
        mocks["merged_model"].save_pretrained.assert_called_once_with(
            str(output_dir.resolve()), safe_serialization=True
        )

    def test_creates_output_directory(self, tmp_path):
        adapter_dir = self._make_adapter_dir(tmp_path)
        output_dir = tmp_path / "nested" / "merged"
        self._run_merge(adapter_dir, output_dir)
        assert output_dir.exists()

    def test_saves_processor(self, tmp_path):
        adapter_dir = self._make_adapter_dir(tmp_path)
        output_dir = tmp_path / "merged"
        mocks = self._run_merge(adapter_dir, output_dir)
        mocks["processor_cls"].from_pretrained.assert_called_once_with(
            str(adapter_dir.resolve())
        )
        mocks["processor"].save_pretrained.assert_called_once_with(
            str(output_dir.resolve())
        )


# ---------------------------------------------------------------------------
# _print_summary — smoke test (no assertions, just ensure no crash)
# ---------------------------------------------------------------------------

class TestPrintSummary:
    def test_runs_without_error(self, tmp_path, capsys):
        from listenr.finetune.merge import _print_summary
        # Create a dummy file so total_bytes > 0
        (tmp_path / "model.safetensors").write_bytes(b"\x00" * 1024)
        _print_summary(tmp_path, "openai/whisper-small")
        captured = capsys.readouterr()
        assert "openai/whisper-small" in captured.out
        assert str(tmp_path) in captured.out
