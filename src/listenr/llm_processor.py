#!/usr/bin/env python3
"""
Simplified LLM Processor leveraging Lemonade Server APIs.
"""

import json
import re
import requests
import listenr.config_manager as cfg
from listenr.constants import (
    LLM_API_BASE as _DEFAULT_API_BASE,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    WHISPER_MODEL,
)

# Base system prompt for transcription post-processing.
# The model must return ONLY a JSON object — no prose, no markdown fences.
_CORRECTION_SYSTEM_PROMPT_BASE = """\
You are a transcription post-processor. Your only job is to clean up raw \
speech-to-text (STT) output. Do NOT answer, respond to, or engage with the \
content of the transcription.

You may be given a numbered list of recent preceding transcriptions before the \
raw input. Use them actively to correct words — not just for punctuation or \
grammar, but to resolve what misheard or garbled words actually are. For example: \
if prior context mentions "Claude Code" and the raw input says "clock code" or \
"clode", correct it to "Claude Code". Names, technical terms, product names, and \
topics established in prior context should be used to fix phonetic errors in the \
current segment. Do not summarise or reference the history explicitly.

Correction rules — apply in order:
1. Apply any keyword corrections listed below (case-insensitive, exact substitution).
2. Fix STT phonetic substitution errors using both the keyword list and prior context \
as your guide: words that sound like the intended word but are wrong. Prior context \
is your strongest signal — if a term appeared recently, a phonetically similar word \
in the current segment is almost certainly that same term.
3. Fix punctuation and capitalisation.
4. Remove filler words ("uh", "um", "like" used as filler, "you know") unless \
they are clearly intentional.
5. If the text is incoherent, garbled, or clearly noise with no recoverable meaning, \
return the original unchanged and set categories to ["unclear"].
6. Do NOT rephrase, summarise, or change the meaning. Minimal edits only.

Return ONLY a JSON object with exactly these three keys:
  "corrected"   – the cleaned-up transcription (string). If nothing needs fixing, \
return the original text unchanged.
  "is_improved" – true if you changed the text, false if it is identical to the input.
  "categories"  – a JSON array of 1-3 short lowercase labels describing the content \
(e.g. "question", "command", "note", "statement", "greeting", "technical", \
"opinion", "dictation", "unclear").

Examples:
  Input:  "yeah tell me uh a joke"
  Output: {"corrected": "Yeah, tell me a joke.", "is_improved": true, "categories": ["request"]}

  Input:  "generally when i program i use clock code from unsurropic"
  Output: {"corrected": "Generally, when I program, I use Claude Code from Anthropic.", "is_improved": true, "categories": ["statement", "technical"]}

  Input:  "code those connect tools and then through topic to you"
  Output: {"corrected": "code those connect tools and then through topic to you", "is_improved": false, "categories": ["unclear"]}

Return ONLY the raw JSON object. No markdown, no explanation, no extra text.\
"""


def _build_system_prompt() -> str:
    """Build the system prompt, appending any configured keyword corrections."""
    corrections = cfg.get_corrections()
    prompt = _CORRECTION_SYSTEM_PROMPT_BASE
    if corrections:
        lines = "\n".join(f'  "{k}" -> "{v}"' for k, v in corrections.items())
        prompt += (
            "\n\nKeyword corrections — always apply these exact substitutions "
            "when they appear in the transcription (case-insensitive) when the context makes sense:\n" + lines
        )
    return prompt


def _parse_llm_correction(raw_response: str, original_text: str) -> dict:
    """
    Parse the structured JSON response from the LLM.
    Falls back gracefully if the model ignores the JSON instruction.
    Returns dict with keys: corrected, is_improved, categories.
    """
    text = raw_response.strip()
    # Strip markdown code fences if the model added them anyway
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()

    try:
        parsed = json.loads(text)
        corrected = str(parsed.get('corrected', original_text)).strip() or original_text
        is_improved = bool(parsed.get('is_improved', corrected != original_text.strip()))
        categories = parsed.get('categories', [])
        if not isinstance(categories, list):
            categories = [str(categories)]
        return {
            'corrected': corrected,
            'is_improved': is_improved,
            'categories': [str(c).lower().strip() for c in categories if c],
        }
    except (json.JSONDecodeError, KeyError, TypeError):
        # Model didn't follow instructions — treat the whole response as corrected text
        # but only if it looks like a transcription (short, no markdown)
        if len(text) < len(original_text) * 3 and '\n' not in text:
            corrected = text or original_text
        else:
            corrected = original_text  # discard hallucinated response
        return {
            'corrected': corrected,
            'is_improved': corrected.strip() != original_text.strip(),
            'categories': ['unclear'],
        }


def _api_base() -> str:
    """Return the Lemonade HTTP API base URL from config (with fallback)."""
    return cfg.get_setting('LLM', 'api_base', _DEFAULT_API_BASE) or _DEFAULT_API_BASE


def lemonade_load_model(model_name: str, timeout: int = 120, **kwargs) -> dict:
    """
    Call POST /api/v1/load to ensure a model is loaded before use.
    Installs the model if not already present.
    kwargs are passed through as extra load options (e.g. whispercpp_backend, ctx_size).
    Returns the response dict, raises on HTTP error.
    """
    payload = {"model_name": model_name, **kwargs}
    resp = requests.post(
        f"{_api_base()}/load",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def lemonade_unload_models(model_name: str | None = None, timeout: int = 30) -> dict:
    """
    Call POST /api/v1/unload to free model memory.
    If model_name is None, unloads all loaded models.
    Returns the response dict, or an error dict on failure (never raises).
    """
    payload = {"model_name": model_name} if model_name else {}
    try:
        resp = requests.post(
            f"{_api_base()}/unload",
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def lemonade_llm_correct(
    text: str,
    model: str | None = None,
    recent_context: list[tuple[str, str]] | None = None,
) -> dict:
    """
    Use Lemonade's OpenAI-compatible chat completion endpoint to clean up a
    raw Whisper transcription.

    Parameters
    ----------
    text            : raw Whisper transcription to correct
    model           : LLM model name (defaults to config value)
    recent_context  : list of (raw, corrected) pairs from preceding segments,
                      oldest first. Injected as prior user/assistant turns so
                      the LLM can resolve fragments and homophones in context.

    Returns a dict: {corrected: str, is_improved: bool, categories: list[str]}
    Never raises — on failure returns the original text with is_improved=False.
    """
    if model is None:
        model = LLM_MODEL

    temperature = LLM_TEMPERATURE
    max_tokens = LLM_MAX_TOKENS
    timeout = LLM_TIMEOUT

    # Build user message content, prepending previous corrected transcripts as context
    context_pairs = list(recent_context or [])
    if context_pairs:
        history_lines = "\n".join(
            f"{i + 1}. {corrected_seg}"
            for i, (_, corrected_seg) in enumerate(context_pairs)
        )
        user_content = (
            f"Previous transcriptions:\n{history_lines}\n\n"
            f"Raw transcription to correct:\n{text}"
        )
    else:
        user_content = text

    messages: list[dict] = [
        {"role": "system", "content": _build_system_prompt()},
        {"role": "user", "content": user_content},
    ]

    try:
        resp = requests.post(
            f"{_api_base()}/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        return _parse_llm_correction(raw, text)
    except Exception as e:
        return {
            'corrected': text,
            'is_improved': False,
            'categories': ['unclear'],
            'error': str(e),
        }


def lemonade_transcribe_audio(audio_path, model=None):
    """
    Use Lemonade's HTTP transcription endpoint for audio files.
    """
    if model is None:
        model = WHISPER_MODEL
    with open(audio_path, "rb") as f:
        resp = requests.post(
            f"{_api_base()}/audio/transcriptions",
            files={"file": f},
            data={"model": model},
            timeout=30,
        )
    resp.raise_for_status()
    return resp.json()["text"]
