"""Tests for the finetune CLI's argument rules (no model or data needed)."""

from __future__ import annotations

import sys

import pytest

from listenr.finetune import train


def test_from_scratch_requires_full(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["finetune", "--from-scratch", "--method", "lora"])
    with pytest.raises(SystemExit) as exc:
        train.main()
    assert exc.value.code == 2
    assert "--from-scratch requires --method full" in capsys.readouterr().err


def test_seed_flag_defaults_to_settings(monkeypatch):
    import argparse
    from listenr.settings import settings
    assert settings.finetune.seed == 42
