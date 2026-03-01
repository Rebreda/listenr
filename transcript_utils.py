"""
transcript_utils.py — Whisper transcription post-processing helpers.

Provides two public functions:
    is_hallucination(text)  → bool   — True if the text is a noise-only hallucination
    strip_noise_tags(text)  → str    — Remove inline (sound effect) / [token] tags

These are intentionally pure functions with no dependencies on config or I/O so
they are easy to unit test and reuse.
"""

import re

# ---------------------------------------------------------------------------
# Hallucination detection
# ---------------------------------------------------------------------------

# Patterns that indicate the entire transcript is a Whisper noise hallucination.
# Matched against the full (stripped) text.
_HALLUCINATION_PATTERNS: list[re.Pattern] = [
    # Entire text is a single parenthesised or bracketed tag, e.g. "(soft music)", "[Applause]"
    re.compile(r'^\s*[\(\[][^\)\]]{1,60}[\)\]]\s*$', re.IGNORECASE),
    # Whisper special blank/noise tokens
    re.compile(r'^\s*\[BLANK_AUDIO\]\s*$', re.IGNORECASE),
    re.compile(r'^\s*\[INAUDIBLE\]\s*$', re.IGNORECASE),
    re.compile(r'^\s*\[SIGH\]\s*$', re.IGNORECASE),
    re.compile(r'^\s*\[ Inaudible \]\s*$', re.IGNORECASE),
    # Punctuation / whitespace only
    re.compile(r'^[\s\.\,\!\?\-\u2026]+$'),
    # Filler-only lines: "uh", "um", "uh uh", "..."
    re.compile(r'^(\buh\b[\s,]*|\bum\b[\s,]*|\.){1,5}\s*$', re.IGNORECASE),
    # YouTube clichés that fill the whole segment (nothing meaningful after them)
    re.compile(r'^\s*(thank you for watching|please subscribe|like and subscribe)[.!\s]*$', re.IGNORECASE),
    # Attribution lines — "Subtitles by X", "Transcribed by X", etc.
    re.compile(r'^\s*(subtitles by|transcribed by|captioned by)\b', re.IGNORECASE),
]


def is_hallucination(text: str) -> bool:
    """
    Return True if *text* appears to be a Whisper noise/silence hallucination
    rather than real speech.

    Matches are made against the whole stripped string, so a line that is
    *only* a sound-effect tag (e.g. "(soft music)") returns True, but a line
    that mixes speech and tags (e.g. "(music) Hello there") returns False —
    use strip_noise_tags() to clean the latter.
    """
    t = text.strip()
    if not t:
        return True
    return any(p.search(t) for p in _HALLUCINATION_PATTERNS)


# ---------------------------------------------------------------------------
# Inline noise-tag stripping
# ---------------------------------------------------------------------------

# Matches inline sound/noise tags anywhere within text:
#   (soft music)  [APPLAUSE]  (speaking in foreign language)  [crosstalk]
# Capped at 60 chars to avoid matching legit parenthetical phrases.
_INLINE_TAG_RE = re.compile(r'[\(\[][^\)\]]{1,60}[\)\]]', re.IGNORECASE)


def strip_noise_tags(text: str) -> str:
    """
    Remove Whisper inline sound-effect tags from *text* and normalise whitespace.

    Examples
    --------
    >>> strip_noise_tags("(soft music)\\n I think if we keep going.")
    "I think if we keep going."
    >>> strip_noise_tags("Hello [crosstalk] world.")
    "Hello world."
    >>> strip_noise_tags("(music)")   # all noise → empty string
    ""
    """
    cleaned = _INLINE_TAG_RE.sub('', text)
    # Collapse newlines and runs of spaces left behind by removed tags
    cleaned = re.sub(r'\n+', ' ', cleaned)
    cleaned = re.sub(r' {2,}', ' ', cleaned)
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Combined convenience helper used by the CLI
# ---------------------------------------------------------------------------

def clean_transcript(text: str) -> tuple[str | None, str]:
    """
    Full cleaning pipeline for a raw Whisper transcript.

    Returns
    -------
    (action, result)
        action  : 'drop'   — the text is pure hallucination, discard it
                  'strip'  — noise tags were removed, result is the cleaned text
                  'ok'     — text was already clean, result is unchanged
        result  : the cleaned text (or the original if action == 'drop')
    """
    if is_hallucination(text):
        return ('drop', text)

    stripped = strip_noise_tags(text)

    if not stripped:
        # Stripping removed everything — was all noise tags
        return ('drop', text)

    if stripped != text.strip():
        return ('strip', stripped)

    return ('ok', text.strip())
