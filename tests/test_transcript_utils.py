"""
Unit tests for transcript_utils — is_hallucination() and strip_noise_tags().
Run with:  python -m pytest tests/
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from transcript_utils import is_hallucination, strip_noise_tags, clean_transcript


# ---------------------------------------------------------------------------
# is_hallucination() — should return True
# ---------------------------------------------------------------------------

class TestIsHallucinationTrue:
    def test_empty_string(self):
        assert is_hallucination('') is True

    def test_whitespace_only(self):
        assert is_hallucination('   ') is True

    def test_punctuation_only(self):
        assert is_hallucination('...') is True
        assert is_hallucination(' . , ! ') is True

    def test_parenthesised_tag(self):
        assert is_hallucination('(soft music)') is True

    def test_parenthesised_tag_with_whitespace(self):
        assert is_hallucination('  (soft music)  ') is True

    def test_bracketed_tag(self):
        assert is_hallucination('[Applause]') is True

    def test_blank_audio_token(self):
        assert is_hallucination('[BLANK_AUDIO]') is True

    def test_inaudible_token(self):
        assert is_hallucination('[INAUDIBLE]') is True

    def test_inaudible_with_spaces(self):
        assert is_hallucination('[ Inaudible ]') is True

    def test_sigh_token(self):
        assert is_hallucination('[SIGH]') is True

    def test_uh_filler_single(self):
        assert is_hallucination('uh') is True

    def test_um_filler_single(self):
        assert is_hallucination('um') is True

    def test_uh_repeated(self):
        assert is_hallucination('uh uh') is True

    def test_uh_um_mixed(self):
        assert is_hallucination('uh, um') is True

    def test_youtube_thank_you(self):
        assert is_hallucination('Thank you for watching') is True
        assert is_hallucination('THANK YOU FOR WATCHING!') is True

    def test_youtube_subscribe(self):
        assert is_hallucination('Please subscribe') is True
        assert is_hallucination('like and subscribe') is True

    def test_subtitles_by(self):
        assert is_hallucination('Subtitles by SomeUser') is True

    def test_ellipsis_unicode(self):
        assert is_hallucination('…') is True


# ---------------------------------------------------------------------------
# is_hallucination() — should return False (real speech)
# ---------------------------------------------------------------------------

class TestIsHallucinationFalse:
    def test_plain_sentence(self):
        assert is_hallucination('Hello, how are you?') is False

    def test_short_sentence(self):
        assert is_hallucination('Yes.') is False

    def test_sentence_with_parenthetical(self):
        # Mixed: has real speech AND a noise tag — not a pure hallucination
        assert is_hallucination('(music) I think if we keep going.') is False

    def test_sentence_with_bracket(self):
        assert is_hallucination('He said [laughing] that it was fine.') is False

    def test_numbers(self):
        assert is_hallucination('The answer is 42.') is False


# ---------------------------------------------------------------------------
# strip_noise_tags() — tag removal
# ---------------------------------------------------------------------------

class TestStripNoiseTags:
    def test_no_tags_unchanged(self):
        assert strip_noise_tags('Hello world.') == 'Hello world.'

    def test_single_paren_tag_removed(self):
        assert strip_noise_tags('(soft music)') == ''

    def test_single_bracket_tag_removed(self):
        assert strip_noise_tags('[Applause]') == ''

    def test_inline_paren_tag(self):
        result = strip_noise_tags('Hello (background noise) world.')
        assert result == 'Hello world.'

    def test_inline_bracket_tag(self):
        result = strip_noise_tags('He said [laughing] it was fine.')
        assert result == 'He said it was fine.'

    def test_leading_tag(self):
        result = strip_noise_tags('(soft music)\n I think if we keep going.')
        assert result == 'I think if we keep going.'

    def test_trailing_tag(self):
        result = strip_noise_tags('I think if we keep going. (applause)')
        assert result == 'I think if we keep going.'

    def test_multiple_tags(self):
        result = strip_noise_tags('(music) Hello [crosstalk] world (applause).')
        assert result == 'Hello world .'

    def test_all_noise_tags_becomes_empty(self):
        assert strip_noise_tags('(music) (applause)') == ''

    def test_newlines_collapsed(self):
        result = strip_noise_tags('Line one\n\nLine two')
        assert result == 'Line one Line two'

    def test_extra_spaces_collapsed(self):
        result = strip_noise_tags('Word   (noise)   other')
        assert result == 'Word other'

    def test_tag_too_long_not_removed(self):
        # Tags longer than 60 chars should NOT be stripped — they're probably real text
        long_tag = '(' + 'x' * 65 + ')'
        result = strip_noise_tags(f'Hello {long_tag} world.')
        assert long_tag in result

    def test_empty_string(self):
        assert strip_noise_tags('') == ''


# ---------------------------------------------------------------------------
# clean_transcript() — combined pipeline
# ---------------------------------------------------------------------------

class TestCleanTranscript:
    def test_hallucination_returns_drop(self):
        action, _ = clean_transcript('(soft music)')
        assert action == 'drop'

    def test_empty_returns_drop(self):
        action, _ = clean_transcript('')
        assert action == 'drop'

    def test_all_tags_returns_drop(self):
        action, _ = clean_transcript('(music) (applause)')
        assert action == 'drop'

    def test_clean_text_returns_ok(self):
        action, result = clean_transcript('Hello world.')
        assert action == 'ok'
        assert result == 'Hello world.'

    def test_text_with_tags_returns_strip(self):
        action, result = clean_transcript('(music) I think if we keep going.')
        assert action == 'strip'
        assert result == 'I think if we keep going.'

    def test_strip_returns_cleaned_text(self):
        action, result = clean_transcript('Hello [laughing] world.')
        assert action == 'strip'
        assert result == 'Hello world.'

    def test_leading_whitespace_normalised(self):
        action, result = clean_transcript('  Hello world.  ')
        assert action == 'ok'
        assert result == 'Hello world.'
