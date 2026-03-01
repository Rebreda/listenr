"""
Unit tests for storage.save_recording().
Run with:  python -m pytest tests/
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
import soundfile as sf
import pytest
from pathlib import Path
from storage import save_recording


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _silence_frames(duration_s: float = 0.5, rate: int = 16000) -> list[bytes]:
    """Return PCM-16 silence as a list of byte chunks."""
    samples = np.zeros(int(rate * duration_s), dtype='<i2')
    return [samples.tobytes()]


def _tone_frames(duration_s: float = 0.5, rate: int = 16000, freq: float = 440.0) -> list[bytes]:
    """Return PCM-16 sine wave as a list of byte chunks."""
    t = np.linspace(0, duration_s, int(rate * duration_s), endpoint=False)
    samples = (np.sin(2 * np.pi * freq * t) * 16000).astype('<i2')
    return [samples.tobytes()]


# ---------------------------------------------------------------------------
# Basic record structure
# ---------------------------------------------------------------------------

class TestSaveRecordingFields:
    def test_returns_dict_with_required_keys(self, tmp_path):
        record = save_recording(
            _silence_frames(),
            raw_text='Hello world.',
            corrected_text='Hello world.',
            storage_base=tmp_path,
        )
        for key in ('uuid', 'timestamp', 'audio_path', 'raw_transcription',
                    'corrected_transcription', 'is_improved', 'categories',
                    'whisper_model', 'llm_model', 'duration_s', 'sample_rate'):
            assert key in record, f"Missing key: {key}"

    def test_transcript_fields_correct(self, tmp_path):
        record = save_recording(
            _silence_frames(),
            raw_text='raw text',
            corrected_text='corrected text',
            storage_base=tmp_path,
            whisper_model='Whisper-Large-v3-Turbo',
            llm_model='gpt-oss',
            is_improved=True,
            categories=['note'],
        )
        assert record['raw_transcription'] == 'raw text'
        assert record['corrected_transcription'] == 'corrected text'
        assert record['is_improved'] is True
        assert record['categories'] == ['note']
        assert record['whisper_model'] == 'Whisper-Large-v3-Turbo'
        assert record['llm_model'] == 'gpt-oss'

    def test_llm_model_none_when_not_improved(self, tmp_path):
        record = save_recording(
            _silence_frames(),
            raw_text='Hello.',
            corrected_text='Hello.',
            storage_base=tmp_path,
            llm_model='gpt-oss',
            is_improved=False,
        )
        assert record['llm_model'] is None

    def test_corrected_falls_back_to_raw_when_empty(self, tmp_path):
        record = save_recording(
            _silence_frames(),
            raw_text='the raw text',
            corrected_text='',
            storage_base=tmp_path,
        )
        assert record['corrected_transcription'] == 'the raw text'

    def test_categories_defaults_to_empty_list(self, tmp_path):
        record = save_recording(
            _silence_frames(), 'hi', 'hi', storage_base=tmp_path,
        )
        assert record['categories'] == []

    def test_duration_is_reasonable(self, tmp_path):
        record = save_recording(
            _silence_frames(duration_s=1.0),
            raw_text='x', corrected_text='x',
            storage_base=tmp_path,
        )
        assert 0.9 < record['duration_s'] < 1.1

    def test_sample_rate_stored(self, tmp_path):
        record = save_recording(
            _silence_frames(), 'x', 'x', storage_base=tmp_path, asr_rate=16000,
        )
        assert record['sample_rate'] == 16000

    def test_uuid_is_12_hex_chars(self, tmp_path):
        record = save_recording(_silence_frames(), 'x', 'x', storage_base=tmp_path)
        assert len(record['uuid']) == 12
        assert all(c in '0123456789abcdef' for c in record['uuid'])


# ---------------------------------------------------------------------------
# File system — WAV file
# ---------------------------------------------------------------------------

class TestSaveRecordingAudio:
    def test_wav_file_created(self, tmp_path):
        record = save_recording(_silence_frames(), 'x', 'x', storage_base=tmp_path)
        assert Path(record['audio_path']).exists()

    def test_wav_file_is_readable(self, tmp_path):
        record = save_recording(_tone_frames(), 'x', 'x', storage_base=tmp_path)
        audio, sr = sf.read(record['audio_path'])
        assert sr == 16000
        assert len(audio) > 0

    def test_audio_path_inside_storage_base(self, tmp_path):
        record = save_recording(_silence_frames(), 'x', 'x', storage_base=tmp_path)
        assert Path(record['audio_path']).is_relative_to(tmp_path)

    def test_audio_stored_under_date_subdir(self, tmp_path):
        record = save_recording(_silence_frames(), 'x', 'x', storage_base=tmp_path)
        # path should be <base>/audio/YYYY-MM-DD/clip_...wav
        parts = Path(record['audio_path']).parts
        assert 'audio' in parts

    def test_multiple_clips_get_unique_paths(self, tmp_path):
        r1 = save_recording(_silence_frames(), 'a', 'a', storage_base=tmp_path)
        r2 = save_recording(_silence_frames(), 'b', 'b', storage_base=tmp_path)
        assert r1['audio_path'] != r2['audio_path']
        assert r1['uuid'] != r2['uuid']


# ---------------------------------------------------------------------------
# File system — no individual transcript JSON files
# ---------------------------------------------------------------------------

class TestNoTranscriptJson:
    def test_no_json_files_written(self, tmp_path):
        save_recording(_silence_frames(), 'hello', 'hello', storage_base=tmp_path)
        json_files = list(tmp_path.rglob('*.json'))
        assert json_files == [], f"Unexpected JSON files: {json_files}"

    def test_no_transcripts_directory(self, tmp_path):
        save_recording(_silence_frames(), 'hello', 'hello', storage_base=tmp_path)
        assert not (tmp_path / 'transcripts').exists()


# ---------------------------------------------------------------------------
# File system — manifest.jsonl
# ---------------------------------------------------------------------------

class TestManifest:
    def test_manifest_created(self, tmp_path):
        save_recording(_silence_frames(), 'x', 'x', storage_base=tmp_path)
        assert (tmp_path / 'manifest.jsonl').exists()

    def test_manifest_has_one_line_per_call(self, tmp_path):
        save_recording(_silence_frames(), 'first', 'first', storage_base=tmp_path)
        save_recording(_silence_frames(), 'second', 'second', storage_base=tmp_path)
        lines = (tmp_path / 'manifest.jsonl').read_text().strip().splitlines()
        assert len(lines) == 2

    def test_manifest_lines_are_valid_json(self, tmp_path):
        save_recording(_silence_frames(), 'hello world', 'Hello world.', storage_base=tmp_path)
        for line in (tmp_path / 'manifest.jsonl').read_text().strip().splitlines():
            parsed = json.loads(line)
            assert isinstance(parsed, dict)

    def test_manifest_record_matches_return_value(self, tmp_path):
        record = save_recording(_silence_frames(), 'test', 'Test.', storage_base=tmp_path)
        line = (tmp_path / 'manifest.jsonl').read_text().strip()
        manifest_record = json.loads(line)
        assert manifest_record['uuid'] == record['uuid']
        assert manifest_record['raw_transcription'] == 'test'
        assert manifest_record['corrected_transcription'] == 'Test.'

    def test_manifest_appends_not_overwrites(self, tmp_path):
        save_recording(_silence_frames(), 'one', 'one', storage_base=tmp_path)
        save_recording(_silence_frames(), 'two', 'two', storage_base=tmp_path)
        save_recording(_silence_frames(), 'three', 'three', storage_base=tmp_path)
        lines = (tmp_path / 'manifest.jsonl').read_text().strip().splitlines()
        texts = [json.loads(l)['raw_transcription'] for l in lines]
        assert texts == ['one', 'two', 'three']

    def test_manifest_handles_unicode(self, tmp_path):
        record = save_recording(
            _silence_frames(), 'café résumé 日本語', 'café résumé 日本語',
            storage_base=tmp_path,
        )
        line = (tmp_path / 'manifest.jsonl').read_text(encoding='utf-8').strip()
        manifest_record = json.loads(line)
        assert manifest_record['raw_transcription'] == 'café résumé 日本語'
