"""Persistent records for fine-tuning and evaluation runs.

Both ``listenr finetune`` and ``listenr eval`` used to report only to stdout.
Training kept metrics as a side effect of the HuggingFace trainer
(``trainer_state.json``, tensorboard under ``runs/``) but recorded nothing
about its own invocation, so an adapter directory could not tell you what
produced it. Evaluation persisted nothing at all: every WER figure and
keyword table lived and died in the terminal.

That makes runs impossible to compare after the fact, which defeats the point
of evaluating. ``listenr eval --output`` writes the full result, including
per-clip hypotheses so a report can be re-scored later under a different
normalization without re-running inference, and ``listenr finetune`` always
drops a ``run.json`` beside the adapter recording what was run, on what, and
with which settings.

Pure module: importable without torch/transformers, like the helpers in
``evaluate.py`` — see tests/test_finetune_report.py.
"""

from __future__ import annotations

import datetime
import importlib.metadata
import json
from pathlib import Path


def listenr_version() -> str:
    try:
        return importlib.metadata.version("listenr")
    except importlib.metadata.PackageNotFoundError:
        return "unknown (not installed)"


def utc_now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def write_json(path: Path, data: dict) -> Path:
    """Write *data* as JSON to *path*, atomically.

    Same tmp-then-replace pattern as the manifest writer in storage.py: a
    crash mid-write must not leave a half-written record where a previous
    good one stood.
    """
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    tmp.replace(path)
    return path


def _jsonable(value):
    """Paths become strings; everything argparse produces is otherwise JSON-safe."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


# ---------------------------------------------------------------------------
# Evaluation report
# ---------------------------------------------------------------------------

def eval_clip_record(
    row: dict,
    hypotheses: dict[str, str],
    keyword_maps: dict[str, dict[str, bool]] | None = None,
) -> dict:
    """One clip's row for the report: identity, ground truth, and what each
    model heard. ``hypotheses`` and ``keyword_maps`` are keyed by model label
    (``fine_tuned``, ``base``)."""
    record = {
        "uuid": row.get("uuid"),
        "audio_path": row.get("audio_path"),
        "duration_s": row.get("duration_s"),
        "reference": row.get("corrected_transcription"),
        "hypotheses": dict(hypotheses),
    }
    if keyword_maps:
        record["keywords"] = {model: dict(hits) for model, hits in keyword_maps.items()}
    return record


def eval_report(
    *,
    mode: str,
    model: str,
    base_model: str | None,
    dataset: str | None,
    split: str | None,
    n_requested: int | None,
    keywords: list[str],
    wer_pct: dict[str, float],
    keyword_recall: dict[str, dict[str, list[int]]],
    clips: list[dict],
) -> dict:
    """Assemble the full evaluation result.

    Everything the terminal report shows, plus the per-clip hypotheses that
    make the aggregate auditable. Aggregates say whether something moved;
    the clip rows say why.
    """
    return {
        "listenr_version": listenr_version(),
        "created_utc": utc_now(),
        "mode": mode,
        "model": str(model),
        "base_model": str(base_model) if base_model else None,
        "dataset": str(dataset) if dataset else None,
        "split": split,
        "n_requested": n_requested,
        "n_evaluated": len(clips),
        "keywords": list(keywords),
        "wer_normalization": "whisper_english",
        # Two decimals: a corpus WER's third decimal is noise, and a results
        # file should not imply precision the sample size cannot support.
        "wer_pct": {m: round(v, 2) for m, v in wer_pct.items()},
        "keyword_recall": {
            model: {kw: {"hits": hits, "expected": expected}
                    for kw, (hits, expected) in tally.items()}
            for model, tally in keyword_recall.items()
        },
        "clips": clips,
    }


# ---------------------------------------------------------------------------
# Fine-tune run record
# ---------------------------------------------------------------------------

def finetune_run_record(
    *,
    args: dict,
    base_model: str,
    architecture: str,
    dataset: str,
    dataset_splits: dict[str, int],
    trainable_params: int,
    total_params: int,
    accelerator: str,
) -> dict:
    """The invocation record written beside the adapter as ``run.json``.

    ``trainer_state.json`` holds the metrics but not the invocation; this
    holds the invocation. Written with ``status: "started"`` before training
    so a crashed run still leaves evidence of what was attempted, then
    rewritten with ``status: "completed"`` at the end.
    """
    return {
        "listenr_version": listenr_version(),
        "started_utc": utc_now(),
        "ended_utc": None,
        "status": "started",
        "args": _jsonable(dict(args)),
        "base_model": base_model,
        "architecture": architecture,
        "dataset": str(dataset),
        "dataset_splits": dict(dataset_splits),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_pct": round(100 * trainable_params / total_params, 2) if total_params else None,
        "accelerator": accelerator,
    }
