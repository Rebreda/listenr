"""listenr.cli must import without a working audio stack."""

import importlib
import sys


def test_cli_imports_when_sounddevice_cannot(monkeypatch):
    # None in sys.modules makes `import sounddevice` raise, as it does on a
    # machine where PortAudio finds PulseAudio libraries but no server.
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    monkeypatch.delitem(sys.modules, "listenr.cli", raising=False)
    cli = importlib.import_module("listenr.cli")
    assert callable(cli.main)
