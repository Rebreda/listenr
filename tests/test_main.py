"""Tests for the unified `listenr` CLI dispatcher."""

import importlib.util

import pytest

from listenr.main import COMMANDS, main


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
