#!/usr/bin/env python3
"""
Simplified LLM Processor leveraging Lemonade Server APIs.
"""

import json
import re
import requests
import listenr.config_manager as cfg

_DEFAULT_API_BASE = "http://localhost:8000/api/v1"

# System prompt for transcription post-processing.
# The model must return ONLY a JSON object — no prose, no markdown fences.
_CORRECTION_SYSTEM_PROMPT = """\
You are a transcription post-processor. Your only job is to clean up the raw \
speech-to-text output provided by the user. Do NOT answer, respond to, or engage \
with the content of the transcription.

Return ONLY a JSON object with exactly these three keys:
  "corrected"   – the cleaned-up transcription text (string). Fix punctuation, \
capitalisation, filler words, and obvious STT errors. If nothing needs fixing, \
return the original text unchanged.
  "is_improved" – true if you changed the text, false if it is identical to the input.
  "categories"  – a JSON array of short lowercase category labels that describe \
the content (e.g. ["question", "command", "note", "greeting", "technical", \
"navigation", "dictation"]). Include 1-3 labels. Use "unclear" if the audio is \
too noisy or blank to categorise.

Example input:  "yeah tell me uh a joke"
Example output: {"corrected": "Yeah, tell me a joke.", "is_improved": true, "categories": ["request"]}

Return ONLY the raw JSON object. No markdown, no explanation, no extra text.\
"""


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


def lemonade_llm_correct(text: str, model: str | None = None) -> dict:
    """
    Use Lemonade's OpenAI-compatible chat completion endpoint to clean up a
    raw Whisper transcription.

    Returns a dict: {corrected: str, is_improved: bool, categories: list[str]}
    Never raises — on failure returns the original text with is_improved=False.
    """
    if model is None:
        model = cfg.get_setting('LLM', 'model', 'gpt-oss-20b-mxfp4-GGUF')

    temperature = cfg.get_float_setting('LLM', 'temperature', 0.1)
    max_tokens = cfg.get_int_setting('LLM', 'max_tokens', 150)
    timeout = cfg.get_int_setting('LLM', 'timeout', 30)

    try:
        resp = requests.post(
            f"{_api_base()}/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": _CORRECTION_SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
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
        model = cfg.get_setting('Whisper', 'model', 'Whisper-Tiny')
    with open(audio_path, "rb") as f:
        resp = requests.post(
            f"{_api_base()}/audio/transcriptions",
            files={"file": f},
            data={"model": model},
            timeout=30,
        )
    resp.raise_for_status()
    return resp.json()["text"]
