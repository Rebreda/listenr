"""
test_finetune_model.py — Unit tests for listenr.finetune.model

Tests the LoRA config factory, encoder freezing, and parameter counting using
small synthetic torch modules — no GPU or model download required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn


# ---------------------------------------------------------------------------
# make_lora_config
# ---------------------------------------------------------------------------

class TestMakeLoraConfig:
    def test_returns_lora_config_with_correct_values(self):
        peft = pytest.importorskip("peft")
        from listenr.finetune.model import make_lora_config

        cfg = make_lora_config(r=4, alpha=16, dropout=0.05, target_modules=["q_proj"])

        assert cfg.r == 4
        assert cfg.lora_alpha == 16
        assert cfg.lora_dropout == 0.05
        # PEFT stores target_modules as a set in newer versions.
        assert set(cfg.target_modules) == {"q_proj"}

    def test_task_type_is_seq2seq(self):
        peft = pytest.importorskip("peft")
        from peft import TaskType
        from listenr.finetune.model import make_lora_config

        cfg = make_lora_config(r=8, alpha=32, dropout=0.1, target_modules=["v_proj"])
        assert cfg.task_type == TaskType.SEQ_2_SEQ_LM

    def test_inference_mode_is_false(self):
        pytest.importorskip("peft")
        from listenr.finetune.model import make_lora_config

        cfg = make_lora_config(r=8, alpha=32, dropout=0.1, target_modules=["q_proj"])
        assert cfg.inference_mode is False

    def test_multiple_target_modules(self):
        pytest.importorskip("peft")
        from listenr.finetune.model import make_lora_config

        cfg = make_lora_config(r=8, alpha=32, dropout=0.1,
                               target_modules=["q_proj", "v_proj"])
        assert set(cfg.target_modules) == {"q_proj", "v_proj"}


# ---------------------------------------------------------------------------
# freeze_encoder — tested with a small synthetic model
# ---------------------------------------------------------------------------

class _FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 4))


class _FakeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 4))


class _FakeWhisperInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _FakeEncoder()
        self.decoder = _FakeDecoder()  # unused but present for realism


class _FakeWhisper(nn.Module):
    """Mimics the attribute path used by freeze_encoder on a raw Whisper model."""
    def __init__(self):
        super().__init__()
        self.model = _FakeWhisperInner()


class _FakePeftWhisper(nn.Module):
    """Mimics the attribute path used by freeze_encoder on a PeftModel."""
    def __init__(self):
        super().__init__()
        self.base_model = MagicMock()
        inner = _FakeWhisperInner()
        self.base_model.model.model = inner
        # Give access to encoder params for assertion
        self._inner = inner


class TestFreezeEncoder:
    def test_encoder_params_frozen_on_raw_model(self):
        from listenr.finetune.model import freeze_encoder

        model = _FakeWhisper()
        # Confirm trainable before freezing
        assert model.model.encoder.weight.requires_grad

        freeze_encoder(model)

        assert not model.model.encoder.weight.requires_grad

    def test_decoder_params_still_trainable_after_freeze(self):
        from listenr.finetune.model import freeze_encoder

        model = _FakeWhisper()
        freeze_encoder(model)

        # Decoder should be unaffected
        assert model.model.decoder.weight.requires_grad

    def test_encoder_params_frozen_on_peft_model(self):
        from listenr.finetune.model import freeze_encoder

        model = _FakePeftWhisper()
        # Confirm trainable before
        assert model._inner.encoder.weight.requires_grad

        freeze_encoder(model)

        assert not model._inner.encoder.weight.requires_grad


# ---------------------------------------------------------------------------
# count_trainable_params
# ---------------------------------------------------------------------------

class TestCountTrainableParams:
    def test_all_params_trainable_by_default(self):
        from listenr.finetune.model import count_trainable_params

        model = nn.Linear(4, 4)  # weight (16) + bias (4) = 20 params
        trainable, total = count_trainable_params(model)
        assert trainable == total == 20

    def test_frozen_params_not_counted_as_trainable(self):
        from listenr.finetune.model import count_trainable_params

        model = nn.Linear(4, 4)
        for p in model.parameters():
            p.requires_grad = False

        trainable, total = count_trainable_params(model)
        assert trainable == 0
        assert total == 20

    def test_partial_freeze(self):
        from listenr.finetune.model import count_trainable_params

        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        # Freeze only the first sub-module
        for p in model[0].parameters():
            p.requires_grad = False

        trainable, total = count_trainable_params(model)
        assert total == 40   # two Linear(4,4)
        assert trainable == 20  # only second layer

    def test_returns_tuple_of_two_ints(self):
        from listenr.finetune.model import count_trainable_params

        model = nn.Linear(2, 2)
        result = count_trainable_params(model)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(v, int) for v in result)
