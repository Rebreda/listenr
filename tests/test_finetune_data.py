"""
test_finetune_data.py — Unit tests for listenr.finetune.data

Tests cover the pure data-preparation logic (prepare_example, WhisperDataCollator)
using lightweight mocks so no GPU, network access, or optional dependencies are
required to run them.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from listenr.finetune.data import WhisperDataCollator, prepare_example


# ---------------------------------------------------------------------------
# Helpers — lightweight mock processor
# ---------------------------------------------------------------------------

def _make_processor(pad_token_id: int = 50256) -> MagicMock:
    """Return a minimal mock WhisperProcessor sufficient for these tests."""
    processor = MagicMock()

    # feature_extractor.pad returns a dict-like with "input_features" tensor
    def _fe_pad(features, return_tensors):
        import torch
        stacked = torch.stack([torch.tensor(f["input_features"]) for f in features])
        return {"input_features": stacked}

    processor.feature_extractor.pad.side_effect = _fe_pad

    # tokenizer.pad returns padded input_ids + attention_mask
    def _tok_pad(label_batch, return_tensors):
        import torch
        seqs = [f["input_ids"] for f in label_batch]
        max_len = max(len(s) for s in seqs)
        padded = [s + [pad_token_id] * (max_len - len(s)) for s in seqs]
        mask = [[1] * len(s) + [0] * (max_len - len(s)) for s in seqs]
        return MagicMock(
            input_ids=torch.tensor(padded),
            attention_mask=torch.tensor(mask),
        )

    processor.tokenizer.pad.side_effect = _tok_pad
    processor.tokenizer.pad_token_id = pad_token_id
    return processor


def _make_audio_dict(duration_s: float = 0.5, sample_rate: int = 16000) -> dict:
    """Return a decoded HF Audio dict with a sine-wave array."""
    n = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, n, endpoint=False)
    array = (np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return {"array": array, "sampling_rate": sample_rate}


# ---------------------------------------------------------------------------
# prepare_example
# ---------------------------------------------------------------------------

class TestPrepareExample:
    def test_returns_input_features_and_labels(self):
        audio = _make_audio_dict()
        processor = MagicMock()
        processor.feature_extractor.return_value.input_features = [np.zeros((80, 3000), dtype=np.float32)]
        processor.tokenizer.return_value.input_ids = [1, 2, 3, 4]

        result = prepare_example(
            {"audio_path": audio, "corrected_transcription": "hello world"},
            processor,
        )

        assert "input_features" in result
        assert "labels" in result

    def test_feature_extractor_called_with_correct_sample_rate(self):
        audio = _make_audio_dict(sample_rate=16000)
        processor = MagicMock()
        processor.feature_extractor.return_value.input_features = [np.zeros((80, 3000))]
        processor.tokenizer.return_value.input_ids = [1]

        prepare_example(
            {"audio_path": audio, "corrected_transcription": "hi"},
            processor,
        )

        call_kwargs = processor.feature_extractor.call_args
        assert call_kwargs.kwargs["sampling_rate"] == 16000

    def test_tokenizer_called_with_corrected_transcription(self):
        audio = _make_audio_dict()
        processor = MagicMock()
        processor.feature_extractor.return_value.input_features = [np.zeros((80, 3000))]
        processor.tokenizer.return_value.input_ids = [5, 6]

        prepare_example(
            {"audio_path": audio, "corrected_transcription": "test sentence"},
            processor,
        )

        processor.tokenizer.assert_called_once_with("test sentence")

    def test_input_features_is_first_element(self):
        """prepare_example takes .input_features[0] — the first example in the batch."""
        expected = np.ones((80, 3000), dtype=np.float32)
        audio = _make_audio_dict()
        processor = MagicMock()
        processor.feature_extractor.return_value.input_features = [expected, np.zeros((80, 3000))]
        processor.tokenizer.return_value.input_ids = [1]

        result = prepare_example({"audio_path": audio, "corrected_transcription": "x"}, processor)
        np.testing.assert_array_equal(result["input_features"], expected)


# ---------------------------------------------------------------------------
# WhisperDataCollator
# ---------------------------------------------------------------------------

class TestWhisperDataCollator:
    """Tests rely on actual torch tensors via the mock _fe_pad / _tok_pad helpers."""

    @pytest.fixture()
    def collator(self):
        processor = _make_processor(pad_token_id=50256)
        return WhisperDataCollator(processor=processor, decoder_start_token_id=50258)

    def _make_features(self, label_lengths: list[int]) -> list[dict]:
        """Return fake prepared examples with varying label lengths."""
        return [
            {
                "input_features": np.zeros((80, 3000), dtype=np.float32),
                "labels": list(range(1, length + 1)),
            }
            for length in label_lengths
        ]

    def test_batch_has_input_features_and_labels(self, collator):
        batch = collator(self._make_features([3, 5]))
        assert "input_features" in batch
        assert "labels" in batch

    def test_input_features_shape(self, collator):
        import torch
        features = self._make_features([3, 4])
        batch = collator(features)
        assert batch["input_features"].shape == torch.Size([2, 80, 3000])

    def test_labels_padded_to_max_length(self, collator):
        batch = collator(self._make_features([3, 6]))
        # All label rows should have the same length after padding
        assert batch["labels"].shape[1] == 6

    def test_padding_positions_replaced_with_minus100(self, collator):
        """Shorter labels should have -100 in the padded positions."""
        batch = collator(self._make_features([2, 5]))
        short_row = batch["labels"][0]  # length 2, padded to 5
        assert (short_row[2:] == -100).all(), \
            f"Expected padding -100 from index 2, got {short_row}"

    def test_bos_trimmed_when_all_rows_start_with_decoder_start_token(self, collator):
        """If every label sequence starts with the BOS token, it should be removed."""
        bos = 50258
        features = [
            {"input_features": np.zeros((80, 3000)), "labels": [bos, 1, 2]},
            {"input_features": np.zeros((80, 3000)), "labels": [bos, 3, 4, 5]},
        ]
        batch = collator(features)
        # After BOS removal both rows should NOT start with 50258
        assert batch["labels"][0, 0].item() != bos
        assert batch["labels"][1, 0].item() != bos

    def test_bos_not_trimmed_when_mixed_start_tokens(self, collator):
        """BOS trim only happens when ALL rows start with the BOS token."""
        bos = 50258
        features = [
            {"input_features": np.zeros((80, 3000)), "labels": [bos, 1, 2]},
            {"input_features": np.zeros((80, 3000)), "labels": [99, 3, 4]},  # no BOS
        ]
        batch = collator(features)
        assert batch["labels"][0, 0].item() == bos
