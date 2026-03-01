"""
Unit tests for llm_processor._parse_llm_correction().
Run with:  python -m pytest tests/
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from llm_processor import _parse_llm_correction


ORIGINAL = 'yeah tell me uh a joke'


class TestParseLlmCorrectionValidJson:
    def test_basic_correction(self):
        raw = '{"corrected": "Yeah, tell me a joke.", "is_improved": true, "categories": ["request"]}'
        result = _parse_llm_correction(raw, ORIGINAL)
        assert result['corrected'] == 'Yeah, tell me a joke.'
        assert result['is_improved'] is True
        assert result['categories'] == ['request']

    def test_not_improved(self):
        raw = '{"corrected": "yeah tell me uh a joke", "is_improved": false, "categories": ["unclear"]}'
        result = _parse_llm_correction(raw, ORIGINAL)
        assert result['is_improved'] is False
        assert result['corrected'] == 'yeah tell me uh a joke'

    def test_multiple_categories(self):
        raw = '{"corrected": "Set a timer.", "is_improved": true, "categories": ["command", "navigation"]}'
        result = _parse_llm_correction(raw, 'set a timer')
        assert result['categories'] == ['command', 'navigation']

    def test_categories_normalised_to_lowercase(self):
        raw = '{"corrected": "Hello.", "is_improved": true, "categories": ["Greeting", "CASUAL"]}'
        result = _parse_llm_correction(raw, 'hello')
        assert result['categories'] == ['greeting', 'casual']

    def test_missing_corrected_falls_back_to_original(self):
        raw = '{"is_improved": false, "categories": []}'
        result = _parse_llm_correction(raw, ORIGINAL)
        assert result['corrected'] == ORIGINAL

    def test_empty_corrected_falls_back_to_original(self):
        raw = '{"corrected": "", "is_improved": false, "categories": []}'
        result = _parse_llm_correction(raw, ORIGINAL)
        assert result['corrected'] == ORIGINAL

    def test_non_list_categories_wrapped(self):
        raw = '{"corrected": "Hi.", "is_improved": true, "categories": "greeting"}'
        result = _parse_llm_correction(raw, 'hi')
        assert isinstance(result['categories'], list)

    def test_is_improved_inferred_when_missing(self):
        # corrected differs from original → is_improved should be True
        raw = '{"corrected": "Yeah, tell me a joke.", "categories": []}'
        result = _parse_llm_correction(raw, ORIGINAL)
        assert result['is_improved'] is True

    def test_whitespace_around_json(self):
        raw = '  \n{"corrected": "Hello.", "is_improved": true, "categories": []}  \n'
        result = _parse_llm_correction(raw, 'hello')
        assert result['corrected'] == 'Hello.'


class TestParseLlmCorrectionMarkdownFences:
    def test_json_code_fence_stripped(self):
        raw = '```json\n{"corrected": "Hello.", "is_improved": true, "categories": []}\n```'
        result = _parse_llm_correction(raw, 'hello')
        assert result['corrected'] == 'Hello.'

    def test_plain_code_fence_stripped(self):
        raw = '```\n{"corrected": "Hello.", "is_improved": true, "categories": []}\n```'
        result = _parse_llm_correction(raw, 'hello')
        assert result['corrected'] == 'Hello.'


class TestParseLlmCorrectionFallback:
    def test_plain_text_response_used_as_corrected(self):
        # Model ignored JSON instruction and returned plain text
        raw = 'Yeah, tell me a joke.'
        result = _parse_llm_correction(raw, ORIGINAL)
        assert result['corrected'] == 'Yeah, tell me a joke.'
        assert result['categories'] == ['unclear']

    def test_long_response_discarded(self):
        # Model returned a paragraph — too long to be a corrected transcription
        raw = ('This is a very long response that the model generated instead of following '
               'the JSON instruction. It goes on and on for many many many many many many '
               'many many many many many many many many many words.') * 3
        result = _parse_llm_correction(raw, ORIGINAL)
        # Should fall back to original, not use the rambling output
        assert result['corrected'] == ORIGINAL

    def test_empty_response_falls_back_to_original(self):
        result = _parse_llm_correction('', ORIGINAL)
        assert result['corrected'] == ORIGINAL

    def test_broken_json_falls_back(self):
        raw = '{"corrected": "oops'  # truncated JSON — short + no newline, used as corrected
        result = _parse_llm_correction(raw, ORIGINAL)
        assert result['corrected'] == raw
        assert result['categories'] == ['unclear']

    def test_return_keys_always_present(self):
        for bad in ('', '{}', 'not json at all', '{"corrected": null}'):
            result = _parse_llm_correction(bad, ORIGINAL)
            assert 'corrected' in result
            assert 'is_improved' in result
            assert 'categories' in result
