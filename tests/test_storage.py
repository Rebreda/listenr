"""
Unit tests for storage.save_recording().
Run with:  python -m pytest tests/
"""

import json
import numpy as np
import soundfile as sf
import pytest
from pathlib import Path
from listenr.storage import save_recording


# ---------------------------------------------------------------------------
# Basic record structure
# ---------------------------------------------------------------------------

class TestSaveRecordingFields:
    def test_returns_dict_with_required_keys(self, tmp_path, silence_frames):
        record = save_recording(
            silence_frames, 'Hello world.', 'Hello world.', storage_base=tmp_path,
        )
        for key in ('uuid', 'timestamp', 'audio_path', 'raw_transcription',
                    'corrected_transcription', 'is_improved', 'categories',
                    'whisper_model', 'llm_model', 'duration_s', 'sample_rate'):
            assert key in record, f"Missing key: {key}"

    def test_transcript_fields_correct(self, tmp_path, silence_frames):
        record = save_recording(
            silence_frames,
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

    def test_llm_model_none_when_not_improved(self, tmp_path, silence_frames):
        record = save_recording(
            silence_frames, 'Hello.', 'Hello.',
            storage_base=tmp_path, llm_model='gpt-oss', is_improved=False,
        )
        assert record['llm_model'] is None

    def test_corrected_falls_back_to_raw_when_empty(self, tmp_path, silence_frames):
        record = save_recording(
            silence_frames, 'the raw text', '', storage_base=tmp_path,
        )
        assert record['corrected_transcription'] == 'the raw text'

    def test_categories_defaults_to_empty_list(self, tmp_path, silence_frames):
        record = save_recording(silence_frames, 'hi', 'hi', storage_base=tmp_path)
        assert record['categories'] == []

    def test_duration_is_reasonable(self, tmp_path, silence_frames):
        # silence_frames fixture is 0.5s at 16kHz
        record = save_recording(silence_frames, 'x', 'x', storage_base=tmp_path)
        assert 0.4 < record['duration_s'] < 0.6

    def test_sample_rate_stored(self, tmp_path, silence_frames):
        record = save_recording(silence_frames, 'x', 'x', storage_base=tmp_path, asr_rate=16000)
        assert record['sample_rate'] == 16000

    def test_uuid_is_12_hex_chars(self, tmp_path, silence_frames):
        record = save_recording(silence_frames, 'x', 'x', storage_base=tmp_path)
        assert len(record['uuid']) == 12
        assert all(c in '0123456789abcdef' for c in record['uuid'])


# ---------------------------------------------------------------------------
# File system -- WAV file
# ---------------------------------------------------------------------------

class TestSaveRecordingAudio:
    def test_wav_file_created(self, tmp_path, silence_frames):
        record = save_recording(silence_frames, 'x', 'x', storage_base=tmp_path)
        assert Path(record['audio_path']).exists()

    def test_wav_file_is_readable(self, tmp_path, tone_frames):
        record = save_recording(tone_frames, 'x', 'x', storage_base=tmp_path)
        audio, sr = sf.read(record['audio_path'])
        assert sr == 16000
        assert len(audio) > 0

    def test_audio_path_inside_storage_base(self, tmp_path, silence_frames):
        record = save_recording(silence_frames, 'x', 'x', storage_base=tmp_path)
        assert Path(record['audio_path']).is_relative_to(tmp_path)

    def test_audio_stored_under_date_subdir(self, tmp_path, silence_frames):
        record = save_recording(silence_frames, 'x', 'x', storage_base=tmp_path)
        parts = Path(record['audio_path']).parts
        assert 'audio' in parts

    def test_multiple_clips_get_unique_paths(self, tmp_path, silence_frames):
        r1 = save_recording(silence_frames, 'a', 'a', storage_base=tmp_path)
        r2 = save_recording(silence_frames, 'b', 'b', storage_base=tmp_path)
        assert r1['audio_path'] != r2['audio_path']
        assert r1['uuid'] != r2['uuid']


# ---------------------------------------------------------------------------
# File system -- no individual transcript JSON files
# ---------------------------------------------------------------------------

class TestNoTranscriptJson:
    def test_no_json_files_written(self, tmp_path, silence_frames):
        save_recording(silence_frames, 'hello', 'hello', storage_base=tmp_path)
        json_files = list(tmp_path.rglob('*.json'))
        assert json_files == [], f"Unexpected JSON files: {json_files}"

    def test_no_transcripts_directory(self, tmp_path, silence_frames):
        save_recording(silence_frames, 'hello', 'hello', storage_base=tmp_path)
        assert not (tmp_path / 'transcripts').exists()


# ---------------------------------------------------------------------------
# File system -- manifest.jsonl
# ---------------------------------------------------------------------------

class TestManifest:
    def test_manifest_created(self, tmp_path, silence_frames):
        save_recording(silence_frames, 'x', 'x', storage_base=tmp_path)
        assert (tmp_path / 'manifest.jsonl').exists()

    def test_manifest_has_one_line_per_call(self, tmp_path, silence_frames):
        save_recording(silence_frames, 'first', 'first', storage_base=tmp_path)
        save_recording(silence_frames, 'second', 'second', storage_base=tmp_path)
        lines = (tmp_path / 'manifest.jsonl').read_text().strip().splitlines()
        assert len(lines) == 2

    def test_manifest_lines_are_valid_json(self, tmp_path, silence_frames):
        save_recording(silence_frames, 'hello world', 'Hello world.', storage_base=tmp_path)
        for line in (tmp_path / 'manifest.jsonl').read_text().strip().splitlines():
            assert isinstance(json.loads(line), dict)

    def test_manifest_record_matches_return_value(self, tmp_path, silence_frames):
        record = save_recording(silence_frames, 'test', 'Test.', storage_base=tmp_path)
        manifest_record = json.loads((tmp_path / 'manifest.jsonl').read_text().strip())
        assert manifest_record['uuid'] == record['uuid']
        assert manifest_record['raw_transcription'] == 'test'
        assert manifest_record['corrected_transcription'] == 'Test.'

    def test_manifest_appends_not_overwrites(self, tmp_path, silence_frames):
        for text in ('one', 'two', 'three'):
            save_recording(silence_frames, text, text, storage_base=tmp_path)
        lines = (tmp_path / 'manifest.jsonl').read_text().strip().splitlines()
        texts = [json.loads(l)['raw_transcription'] for l in lines]
        assert texts == ['one', 'two', 'three']

    def test_manifest_handles_unicode(self, tmp_path, silence_frames):
        save_recording(silence_frames, 'cafe resume', 'cafe resume', storage_base=tmp_path)
        manifest_record = json.loads(
            (tmp_path / 'manifest.jsonl').read_text(encoding='utf-8').strip()
        )
        assert manifest_record['raw_transcription'] == 'cafe resume'
