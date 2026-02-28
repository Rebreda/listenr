#!/usr/bin/env python3
"""
Simplified LLM Processor leveraging Lemonade Server APIs.
"""

import requests

LEMONADE_API_BASE = "http://localhost:8000/api/v1"

def lemonade_llm_correct(text, model="Qwen3-0.6B-GGUF", system_prompt=None):
    """
    Use Lemonade's OpenAI-compatible chat completion endpoint for LLM correction.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": text})
    resp = requests.post(
        f"{LEMONADE_API_BASE}/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()

def lemonade_transcribe_audio(audio_path, model="Whisper-Tiny"):
    """
    Use Lemonade's HTTP transcription endpoint for audio files.
    """
    with open(audio_path, "rb") as f:
        resp = requests.post(
            f"{LEMONADE_API_BASE}/audio/transcriptions",
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
