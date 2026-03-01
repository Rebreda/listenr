"""
storage.py — Persist audio clips and transcription records to disk.

Single public function:
    save_recording(pcm_frames, raw_text, corrected_text, *, ...) -> dict

Layout on disk:
    <base>/
        audio/
            YYYY-MM-DD/
                clip_YYYY-MM-DD_<uid>.wav
        manifest.jsonl          ← append-only, one JSON object per line

Individual per-clip transcript JSON files are intentionally NOT written.
The manifest.jsonl is the single source of truth and is trivially queryable
with jq, pandas, or any JSONL reader.
"""

import json
import uuid
import numpy as np
import soundfile as sf
from pathlib import Path
from datetime import datetime, timezone


def save_recording(
    pcm_frames: list[bytes],
    raw_text: str,
    corrected_text: str,
    *,
    storage_base: Path,
    asr_rate: int = 16000,
    whisper_model: str = '',
    llm_model: str | None = None,
    is_improved: bool = False,
    categories: list[str] | None = None,
) -> dict:
    """
    Save a single recorded speech clip and append a record to manifest.jsonl.

    Parameters
    ----------
    pcm_frames      : list of raw PCM-16 byte chunks (little-endian int16, mono)
    raw_text        : original Whisper transcription (already noise-stripped)
    corrected_text  : LLM-corrected text, or same as raw_text if LLM not used
    storage_base    : root directory, e.g. Path('~/.listenr/audio_clips').expanduser()
    asr_rate        : sample rate of the PCM data (always 16000 for Lemonade)
    whisper_model   : name of the Whisper model used
    llm_model       : name of the LLM used for correction (None if not used)
    is_improved     : whether the LLM made a meaningful correction
    categories      : LLM-assigned content category labels

    Returns
    -------
    The record dict that was appended to manifest.jsonl.
    """
    ts = datetime.now(timezone.utc)
    date_str = ts.strftime('%Y-%m-%d')
    uid = uuid.uuid4().hex[:12]

    audio_dir = storage_base / 'audio' / date_str
    audio_dir.mkdir(parents=True, exist_ok=True)

    audio_path = audio_dir / f"clip_{date_str}_{uid}.wav"

    # Reconstruct float32 from PCM-16 bytes and write WAV
    audio_np = np.frombuffer(b''.join(pcm_frames), dtype='<i2').astype(np.float32) / 32767.0
    sf.write(str(audio_path), audio_np, asr_rate, subtype='PCM_16')

    record = {
        'uuid': uid,
        'timestamp': ts.isoformat(),
        'audio_path': str(audio_path),
        'raw_transcription': raw_text,
        'corrected_transcription': corrected_text if corrected_text else raw_text,
        'is_improved': is_improved,
        'categories': categories or [],
        'whisper_model': whisper_model,
        'llm_model': llm_model if is_improved else None,
        'duration_s': round(len(audio_np) / asr_rate, 3),
        'sample_rate': asr_rate,
    }

    manifest_path = storage_base / 'manifest.jsonl'
    with open(manifest_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

    return record
