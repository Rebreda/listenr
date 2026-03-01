#!/usr/bin/env python3
"""
Simplified LLM Processor leveraging Lemonade Server APIs.
"""

import requests
import config_manager as cfg

_DEFAULT_API_BASE = "http://localhost:8000/api/v1"


def _api_base() -> str:
    """Return the Lemonade HTTP API base URL from config (with fallback)."""
    return cfg.get_setting('LLM', 'api_base', _DEFAULT_API_BASE) or _DEFAULT_API_BASE


def lemonade_llm_correct(text, model=None, system_prompt=None):
    """
    Use Lemonade's OpenAI-compatible chat completion endpoint for LLM correction.
    """
    if model is None:
        model = cfg.get_setting('LLM', 'model', 'Qwen3-0.6B-GGUF')
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": text})
    resp = requests.post(
        f"{_api_base()}/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()

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

# Example usage:
if __name__ == "__main__":
    # Example: LLM correction
    print(lemonade_llm_correct("im going to the store two by some milk"))

    # Example: Audio transcription
    # print(lemonade_transcribe_audio("test.wav"))
