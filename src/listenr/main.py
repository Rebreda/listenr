#!/usr/bin/env python3
"""Unified ``listenr`` command-line entry point.

Dispatches ``listenr <command> ...`` to the matching module's ``main()``.
Modules are imported lazily so optional extras (finetune, mdc, hf,
categorize) are only required when their command is actually used.
"""

import importlib
import importlib.metadata
import sys

# command -> pip extra that supplies its heavy dependencies
EXTRAS: dict[str, str] = {
    "categorize": "categorize",
    "import-mdc": "mdc",
    "import-hf": "hf",
    "finetune": "finetune",
    "merge": "finetune",
    "eval": "finetune",
}

# command -> (module exposing main(), one-line help)
COMMANDS: dict[str, tuple[str, str]] = {
    "record": ("listenr.cli", "Record from the microphone with live transcription"),
    "asr": ("listenr.unified_asr", "Transcribe an audio file (batch or streaming)"),
    "retranscribe": ("listenr.retranscribe", "Re-run Whisper (and optionally LLM) on saved clips"),
    "build-dataset": ("listenr.build_dataset", "Build train/dev/test splits from recordings"),
    "categorize": ("listenr.categorize", "Filter a manifest to clips matching topics via embeddings"),
    "import-mdc": ("listenr.importers.mdc", "Import a Mozilla Data Collective ASR dataset"),
    "import-hf": ("listenr.importers.hf", "Import a Hugging Face ASR dataset"),
    "finetune": ("listenr.finetune.train", "Fine-tune Whisper or Moonshine with LoRA"),
    "merge": ("listenr.finetune.merge", "Merge a LoRA adapter into a standalone model"),
    "eval": ("listenr.finetune.evaluate", "Evaluate the merged model on the held-out test split"),
}


def _version() -> str:
    try:
        return importlib.metadata.version("listenr")
    except importlib.metadata.PackageNotFoundError:
        return "unknown (not installed)"


def _print_help() -> None:
    width = max(len(name) for name in COMMANDS)
    lines = "\n".join(f"  {name:<{width}}  {desc}" for name, (_, desc) in COMMANDS.items())
    print(
        f"""usage: listenr <command> [options]

Build better speech-to-text and ASR models entirely on your machine.

commands:
{lines}

Run `listenr <command> --help` for command-specific options.
"""
    )


def main() -> int | None:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help", "help"):
        _print_help()
        return 0
    if argv[0] in ("-V", "--version", "version"):
        print(f"listenr {_version()}")
        return 0

    command = argv[0]
    if command not in COMMANDS:
        print(f"listenr: unknown command '{command}'\n", file=sys.stderr)
        _print_help()
        return 2

    module_name, _ = COMMANDS[command]
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        extra = EXTRAS.get(command)
        if extra is None:
            raise
        print(
            f"listenr: '{command}' needs the '{extra}' extra, which is not installed "
            f"({exc}).\n  uv pip install 'listenr[{extra}]'",
            file=sys.stderr,
        )
        return 1
    # Subcommand modules parse sys.argv themselves; rewrite it so their
    # argparse help shows "listenr <command>" as the program name.
    sys.argv = [f"listenr {command}", *argv[1:]]
    return module.main()


if __name__ == "__main__":
    sys.exit(main())
