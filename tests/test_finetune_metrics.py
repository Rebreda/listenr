"""
test_finetune_metrics.py — Unit tests for listenr.finetune.metrics

Tests the make_compute_metrics factory using a mocked evaluate.load so the
`evaluate` optional dependency does not need to be installed.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tokenizer(decode_map: dict | None = None) -> MagicMock:
    """Return a mock tokenizer whose batch_decode uses *decode_map*.

    *decode_map* maps a tuple of the token ids to the decoded string.
    If not provided, the tokenizer returns ``" ".join(str(i) for i in ids)``.
    """
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 50256

    def _batch_decode(ids_list, skip_special_tokens=True):
        if decode_map:
            return [decode_map.get(tuple(row.tolist() if hasattr(row, "tolist") else row), "")
                    for row in ids_list]
        return [" ".join(str(i) for i in (row.tolist() if hasattr(row, "tolist") else row))
                for row in ids_list]

    tokenizer.batch_decode.side_effect = _batch_decode
    return tokenizer


def _make_pred(predictions, label_ids):
    """Return an EvalPrediction-like namespace."""
    return SimpleNamespace(
        predictions=np.array(predictions),
        label_ids=np.array(label_ids, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# make_compute_metrics
# ---------------------------------------------------------------------------

class TestMakeComputeMetrics:
    @pytest.fixture()
    def mock_evaluate(self):
        """Inject a deterministic WER metric via sys.modules so the lazy import works."""
        metric = MagicMock()

        def _compute(predictions, references):
            wrong = sum(p != r for p, r in zip(predictions, references))
            return wrong / len(references) if references else 0.0

        metric.compute.side_effect = _compute

        mock_module = MagicMock()
        mock_module.load.return_value = metric

        with patch.dict(sys.modules, {"evaluate": mock_module}):
            yield mock_module, metric

    def test_returns_callable(self, mock_evaluate):
        from listenr.finetune.metrics import make_compute_metrics
        fn = make_compute_metrics(_make_tokenizer())
        assert callable(fn)

    def test_evaluate_load_called_with_wer(self, mock_evaluate):
        mock_mod, _ = mock_evaluate
        from listenr.finetune.metrics import make_compute_metrics
        make_compute_metrics(_make_tokenizer())
        mock_mod.load.assert_called_once_with("wer")

    def test_perfect_predictions_give_zero_wer(self, mock_evaluate):
        from listenr.finetune.metrics import make_compute_metrics

        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        # Both pred and label decode to the same string
        tokenizer.batch_decode.return_value = ["hello world", "test"]

        fn = make_compute_metrics(tokenizer)
        pred = _make_pred([[1, 2], [3, 4]], [[1, 2], [3, 4]])
        result = fn(pred)

        assert "wer" in result
        assert result["wer"] == 0.0

    def test_all_wrong_predictions_give_100_wer(self, mock_evaluate):
        _, metric = mock_evaluate
        # Override metric to return 1.0 (100%) for all-wrong case
        metric.compute.return_value = 1.0

        from listenr.finetune.metrics import make_compute_metrics

        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.batch_decode.side_effect = [
            ["hello", "world"],   # predictions
            ["wrong", "answer"],  # labels
        ]

        fn = make_compute_metrics(tokenizer)
        pred = _make_pred([[1], [2]], [[3], [4]])
        result = fn(pred)

        assert result["wer"] == 100.0

    def test_padding_tokens_replaced_before_decode(self, mock_evaluate):
        """label_ids -100 positions must be replaced with pad_token_id before decode."""
        from listenr.finetune.metrics import make_compute_metrics

        pad_id = 50256
        tokenizer = MagicMock()
        tokenizer.pad_token_id = pad_id
        tokenizer.batch_decode.return_value = ["hello"]

        fn = make_compute_metrics(tokenizer)
        # -100 at position 1 should become pad_id
        label_ids = np.array([[1, -100, 3]])
        pred = _make_pred([[1, 2, 3]], label_ids)
        fn(pred)

        # After replacement, no -100 should remain in what was passed to decode
        decoded_label_arg = tokenizer.batch_decode.call_args_list[1][0][0]
        assert -100 not in decoded_label_arg

    def test_result_wer_is_scaled_to_percent(self, mock_evaluate):
        """WER should be stored as a percentage (metric returns 0–1, we multiply by 100)."""
        _, metric = mock_evaluate
        metric.compute.return_value = 0.35  # 35%

        from listenr.finetune.metrics import make_compute_metrics

        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.batch_decode.return_value = ["x"]

        fn = make_compute_metrics(tokenizer)
        result = fn(_make_pred([[1]], [[1]]))

        assert abs(result["wer"] - 35.0) < 1e-6
