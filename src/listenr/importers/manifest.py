"""Shared core for turning source rows into a Listenr manifest.

Importers normalise their dataset into an iterable of plain ``dict`` rows and
hand them here: column picking via :class:`~listenr.importers.mapping.FieldMapping`,
materialising audio to WAV where needed, and writing ``manifest.jsonl``.
"""

from __future__ import annotations

import io
import json
import logging
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import soundfile as sf

from listenr.importers.mapping import FieldMapping
from listenr.settings import settings

logger = logging.getLogger("listenr.importers")


def default_manifest_path(source: str, dataset_id: str) -> Path:
    return settings.storage.audio_clips_path / "imports" / source / _slug(dataset_id) / "manifest.jsonl"


def default_audio_dir(source: str, dataset_id: str) -> Path:
    return settings.storage.audio_clips_path / "imports" / source / _slug(dataset_id) / "audio"


def _slug(value: str) -> str:
    return value.strip().replace("/", "__").replace(" ", "_") or "dataset"


try:  # pandas ships with the importer extras, not with the core install
    from pandas import isna as _isna
except ImportError:  # pragma: no cover - exercised by the core-only install
    from numpy import isnan as _isna


def _is_missing(value: Any) -> bool:
    """True for ``None`` and for null markers (NaN, NaT, ``pd.NA``).

    Importer rows come straight out of a DataFrame, so nulls arrive in
    whatever form the source used. Non-scalars (arrays) raise here and are
    treated as present.
    """
    if value is None:
        return True
    try:
        return bool(_isna(value))
    except (TypeError, ValueError):
        return False


def _as_text(value: Any) -> str:
    if _is_missing(value):
        return ""
    return str(value).strip()


def _pick(row: Mapping[str, Any], candidates: tuple[str, ...]) -> Any:
    for key in candidates:
        if key in row:
            value = row[key]
            if not _is_missing(value):
                return value
    return None


def _resolve_audio(
    value: Any,
    dest: Path,
) -> tuple[Path, float | None, int | None] | None:
    """Resolve a row's audio value — a filesystem path, or a Hugging Face
    ``Audio`` dict with ``bytes``/``path`` — to ``(wav_path, duration_s,
    sample_rate)``. Duration/sample rate are ``None`` unless already known
    from decoding; ``None`` overall if the audio can't be resolved.
    """
    if isinstance(value, (str, Path)):
        text = str(value).strip()
        if not text:
            return None
        path = Path(text).expanduser()
        if not path.exists():
            return None
        return path.resolve(), None, None

    if isinstance(value, Mapping):
        raw_bytes = value.get("bytes")
        if raw_bytes:
            data, sr = sf.read(io.BytesIO(raw_bytes))
            dest.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(dest), data, sr)
            return dest.resolve(), len(data) / float(sr), int(sr)

        path_value = value.get("path")
        if path_value:
            path = Path(str(path_value)).expanduser()
            if path.exists():
                return path.resolve(), None, None

    return None


def _identity(value: Any, index: int) -> str:
    """Stable per-row key for deterministic uuids across re-imports."""
    if isinstance(value, (str, Path)):
        return str(value)
    if isinstance(value, Mapping):
        path_value = value.get("path")
        if path_value:
            return str(path_value)
    return f"row:{index}"


def records_from_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    source: str,
    dataset_id: str,
    mapping: FieldMapping,
    audio_dir: Path,
    dataset_name: str = "",
) -> tuple[list[dict], int]:
    """Convert normalised source rows into Listenr manifest records.

    Rows missing a usable audio file or transcription are skipped and counted.
    """
    records: list[dict] = []
    skipped = 0

    for index, row in enumerate(rows):
        transcription = _as_text(_pick(row, mapping.text))
        audio_value = _pick(row, mapping.audio)
        if not transcription or audio_value is None:
            skipped += 1
            continue

        row_uuid = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{source}:{dataset_id}:{_identity(audio_value, index)}",
            )
        )

        resolved = _resolve_audio(audio_value, audio_dir / f"{row_uuid}.wav")
        if resolved is None:
            logger.warning("Skipping row %s: could not resolve audio", index)
            skipped += 1
            continue
        audio_path, duration_s, sample_rate = resolved

        if duration_s is None or sample_rate is None:
            info = sf.info(str(audio_path))
            duration_s, sample_rate = float(info.duration), int(info.samplerate)

        record = {
            "uuid": row_uuid,
            "audio_path": str(audio_path),
            "raw_transcription": transcription,
            "corrected_transcription": transcription,
            "is_improved": False,
            "duration_s": duration_s,
            "sample_rate": sample_rate,
            "whisper_model": "",
            "llm_model": "",
            "timestamp": _as_text(row.get("timestamp")),
            "source": source,
            "source_dataset_id": dataset_id,
        }
        if dataset_name:
            record["source_dataset_name"] = dataset_name
        split = _as_text(_pick(row, mapping.split))
        if split:
            record["source_split"] = split

        records.append(record)

    return records, skipped


def write_manifest(records: list[dict], manifest_path: Path, append: bool = False) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(manifest_path, mode, encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def import_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    source: str,
    dataset_id: str,
    mapping: FieldMapping,
    manifest_path: Path,
    audio_dir: Path,
    dataset_name: str = "",
    append: bool = False,
) -> dict[str, Any]:
    """Build records from ``rows`` and write them to ``manifest_path``."""
    records, skipped = records_from_rows(
        rows,
        source=source,
        dataset_id=dataset_id,
        mapping=mapping,
        audio_dir=audio_dir,
        dataset_name=dataset_name,
    )
    if not records:
        raise RuntimeError(
            "No usable rows were found after validating audio and transcriptions."
        )
    write_manifest(records, manifest_path, append=append)
    return {
        "source": source,
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "manifest_path": manifest_path,
        "imported": len(records),
        "skipped": skipped,
    }
