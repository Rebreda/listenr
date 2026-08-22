"""Tests for the unified `listenr` CLI dispatcher."""

import importlib
import importlib.util

import pytest

from listenr.main import COMMANDS, EXTRAS, main


def _run(monkeypatch, *argv):
    monkeypatch.setattr("sys.argv", ["listenr", *argv])
    return main()


def test_no_args_prints_help(monkeypatch, capsys):
    assert _run(monkeypatch) == 0
    out = capsys.readouterr().out
    assert "usage: listenr" in out
    for command in COMMANDS:
        assert command in out


def test_help_flag(monkeypatch, capsys):
    assert _run(monkeypatch, "--help") == 0
    assert "usage: listenr" in capsys.readouterr().out


def test_version_flag(monkeypatch, capsys):
    assert _run(monkeypatch, "--version") == 0
    assert capsys.readouterr().out.startswith("listenr ")


def test_unknown_command(monkeypatch, capsys):
    assert _run(monkeypatch, "frobnicate") == 2
    assert "unknown command" in capsys.readouterr().err


def test_all_command_modules_resolve():
    for module_name, _ in COMMANDS.values():
        assert importlib.util.find_spec(module_name) is not None, module_name


def test_dispatch_rewrites_argv(monkeypatch, capsys):
    # build-dataset only needs core deps; --help exits 0 via argparse
    with pytest.raises(SystemExit) as excinfo:
        _run(monkeypatch, "build-dataset", "--help")
    assert excinfo.value.code == 0
    assert "listenr build-dataset" in capsys.readouterr().out


def test_every_extra_names_a_real_command():
    assert set(EXTRAS) <= set(COMMANDS)


def _fail_importing(monkeypatch, target, message):
    """Make importing *target* fail, leaving every other import alone.

    pytest's own machinery imports modules while running the test, so a
    blanket failure would take those down with it.
    """
    real = importlib.import_module

    def fake(name, *args, **kwargs):
        if name == target:
            raise ImportError(message)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake)


def test_missing_extra_reports_the_install_command(monkeypatch, capsys):
    _fail_importing(monkeypatch, "listenr.finetune.train", "No module named 'peft'")
    assert _run(monkeypatch, "finetune") == 1
    err = capsys.readouterr().err
    assert "listenr[finetune]" in err


def test_import_error_in_a_core_command_is_not_swallowed(monkeypatch):
    _fail_importing(monkeypatch, "listenr.cli", "genuinely broken")
    with pytest.raises(ImportError):
        _run(monkeypatch, "record")
