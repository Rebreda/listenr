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


# ---------------------------------------------------------------------------
# Resuming an interrupted run

import json  # noqa: E402


def _record(**args):
    base = {"method": "lora", "learning_rate": 3e-4, "max_steps": 1000, "output": "/x", "resume": True,
            "dataloader_workers": 0}
    return {"status": "started", "started_utc": "2026-01-01T00:00:00+00:00",
            "args": base | args, "base_model": "m", "dataset": "/d", "dataset_splits": {"train": 10}}


def _leave_run(tmp_path, record, steps=(500, 1000)):
    (tmp_path / "run.json").write_text(json.dumps(record))
    for s in steps:
        (tmp_path / f"checkpoint-{s}").mkdir()
        (tmp_path / f"checkpoint-{s}" / "trainer_state.json").write_text("{}")


def test_resume_picks_last_checkpoint_of_same_run(tmp_path):
    _leave_run(tmp_path, _record())
    got = train.find_resume_checkpoint(tmp_path, _record(output="/y", dataloader_workers=4), resume=True)
    assert got == tmp_path / "checkpoint-1000"


def test_resume_nothing_to_resume(tmp_path):
    assert train.find_resume_checkpoint(tmp_path, _record(), resume=True) is None


def test_resume_refuses_a_different_run(tmp_path):
    _leave_run(tmp_path, _record())
    with pytest.raises(SystemExit) as exc:
        train.find_resume_checkpoint(tmp_path, _record(learning_rate=1e-4), resume=True)
    assert "learning_rate" in str(exc.value)


def test_resume_refuses_a_completed_run(tmp_path):
    _leave_run(tmp_path, _record() | {"status": "completed"})
    with pytest.raises(SystemExit) as exc:
        train.find_resume_checkpoint(tmp_path, _record(), resume=True)
    assert "completed" in str(exc.value)


def test_resume_refuses_checkpoints_without_run_record(tmp_path):
    (tmp_path / "checkpoint-500").mkdir()
    with pytest.raises(SystemExit):
        train.find_resume_checkpoint(tmp_path, _record(), resume=True)


def test_no_resume_moves_checkpoints_aside(tmp_path):
    _leave_run(tmp_path, _record() | {"status": "completed"})
    assert train.find_resume_checkpoint(tmp_path, _record(), resume=False) is None
    assert not list(tmp_path.glob("checkpoint-*"))
    (aside,) = tmp_path.glob("previous-*")
    assert sorted(p.name for p in aside.iterdir()) == ["checkpoint-1000", "checkpoint-500"]
