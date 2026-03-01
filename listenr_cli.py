#!/usr/bin/env python3
"""
Listenr CLI: Real-time Microphone Streaming to Lemonade ASR

Streams microphone to Lemonade's /realtime WebSocket endpoint.
Prints transcriptions as they arrive. Saves audio + transcripts for dataset use.
All config loaded from config_manager.py.

Usage:
    uv run listenr_cli.py
    uv run listenr_cli.py --no-save       # Don't save recordings
    uv run listenr_cli.py --show-raw      # Show raw Whisper transcription too
"""

import asyncio
import json
import logging
import uuid
import argparse
import requests
import numpy as np
import sounddevice as sd
import soundfile as sf
from pathlib import Path
from datetime import datetime, timezone

from unified_asr import LemonadeUnifiedASR
from llm_processor import lemonade_llm_correct
import config_manager as cfg

logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

# Audio settings from config
CHUNK_SIZE = cfg.get_int_setting('Audio', 'blocksize', 1360)
SAMPLE_RATE = cfg.get_int_setting('Audio', 'sample_rate', 16000)
CHANNELS = cfg.get_int_setting('Audio', 'channels', 1)
INPUT_DEVICE = cfg.get_setting('Audio', 'input_device', 'default') or None
if INPUT_DEVICE == 'default':
    INPUT_DEVICE = None

# Storage
STORAGE_BASE = Path(
    cfg.get_setting('Storage', 'audio_clips_path', '~/listenr_recordings') or '~/listenr_recordings'
).expanduser()

# LLM settings
USE_LLM = cfg.get_bool_setting('LLM', 'enabled', False)
LLM_MODEL = cfg.get_setting('LLM', 'model', 'Qwen3-0.6B-GGUF') or 'Qwen3-0.6B-GGUF'
WHISPER_MODEL = cfg.get_setting('Whisper', 'model', 'Whisper-Tiny') or 'Whisper-Tiny'


def get_lemonade_ws_url():
    """Discover Lemonade WebSocket URL from /api/v1/health."""
    api_base = cfg.get_setting('LLM', 'api_base', 'http://localhost:8000/api/v1') or 'http://localhost:8000/api/v1'
    health_url = api_base.rstrip('/').replace('/api/v1', '') + '/api/v1/health'
    try:
        resp = requests.get(health_url, timeout=2)
        resp.raise_for_status()
        data = resp.json()
        ws_port = data.get('websocket_port', 8001)
        return f"ws://localhost:{ws_port}/realtime"
    except Exception as e:
        print(f"⚠️  Could not discover Lemonade websocket port ({e}), using default :8001")
        return "ws://localhost:8001/realtime"


def save_recording(pcm_frames: list, raw_text: str, corrected_text: str) -> dict:
    """Save audio and transcript JSON to the storage directory."""
    ts = datetime.now(timezone.utc)
    date_str = ts.strftime('%Y-%m-%d')
    uid = uuid.uuid4().hex[:12]

    audio_dir = STORAGE_BASE / 'audio' / date_str
    transcript_dir = STORAGE_BASE / 'transcripts' / date_str
    audio_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir.mkdir(parents=True, exist_ok=True)

    audio_path = audio_dir / f"clip_{date_str}_{uid}.wav"
    transcript_path = transcript_dir / f"transcript_{date_str}_{uid}.json"

    # Reconstruct float32 audio from pcm16 bytes and save
    audio_np = np.frombuffer(b''.join(pcm_frames), dtype=np.int16).astype(np.float32) / 32767.0
    sf.write(str(audio_path), audio_np, SAMPLE_RATE, subtype='PCM_16')

    is_improved = bool(corrected_text and corrected_text.strip() != raw_text.strip())
    record = {
        'uuid': uid,
        'timestamp': ts.isoformat(),
        'audio_path': str(audio_path),
        'raw_transcription': raw_text,
        'corrected_transcription': corrected_text if corrected_text else raw_text,
        'is_improved': is_improved,
        'whisper_model': WHISPER_MODEL,
        'llm_model': LLM_MODEL if is_improved else None,
        'duration_s': round(len(audio_np) / SAMPLE_RATE, 3),
        'sample_rate': SAMPLE_RATE,
    }
    with open(transcript_path, 'w') as f:
        json.dump(record, f, indent=2)

    return record


async def mic_stream(pcm_buffer: list):
    """Async generator yielding raw PCM16 bytes from mic, also buffering for saving."""
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype='float32',
        blocksize=CHUNK_SIZE,
        device=INPUT_DEVICE
    ) as stream:
        while True:
            audio_chunk, _ = stream.read(CHUNK_SIZE)
            pcm16 = (np.clip(audio_chunk, -1, 1) * 32767).astype(np.int16).tobytes()
            pcm_buffer.append(pcm16)
            yield pcm16


async def main(save: bool, show_raw: bool):
    ws_url = get_lemonade_ws_url()
    ws_url_with_model = f"{ws_url}?model={WHISPER_MODEL}"
    print(f"🎤 Listenr CLI — streaming to Lemonade")
    print(f"   Model  : {WHISPER_MODEL}")
    print(f"   WS URL : {ws_url_with_model}")
    print(f"   LLM    : {'enabled (' + LLM_MODEL + ')' if USE_LLM else 'disabled'}")
    print(f"   Save   : {'yes → ' + str(STORAGE_BASE) if save else 'no'}")
    print("   Press Ctrl+C to stop.\n")

    asr = LemonadeUnifiedASR(use_llm=False)  # We handle LLM correction ourselves for saving
    pcm_buffer: list = []
    pending_raw: list = []  # buffer of raw texts in current speech segment

    async for result in asr.stream_transcribe(mic_stream(pcm_buffer), lemonade_ws_url=ws_url_with_model):
        msg_type = result.get('type', '')

        if msg_type == 'conversation.item.input_audio_transcription.completed':
            raw_text = result.get('transcript', '').strip()
            if not raw_text:
                continue

            corrected_text = raw_text
            if USE_LLM:
                try:
                    corrected_text = lemonade_llm_correct(raw_text, model=LLM_MODEL)
                except Exception as e:
                    print(f"  ⚠️  LLM correction failed: {e}")

            if show_raw and corrected_text != raw_text:
                print(f"  [RAW] {raw_text}")
            print(f"  [ASR] {corrected_text}")

            if save and pcm_buffer:
                record = save_recording(list(pcm_buffer), raw_text, corrected_text)
                print(f"  [SAVED] {record['audio_path']} ({record['duration_s']}s, improved={record['is_improved']})")
                pcm_buffer.clear()

        elif msg_type == 'input_audio_buffer.speech_started':
            pcm_buffer.clear()  # start fresh buffer for this segment

        elif msg_type == 'error':
            print(f"  ❌ Error from Lemonade: {result.get('message', result)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Listenr CLI — real-time ASR via Lemonade')
    parser.add_argument('--no-save', action='store_true', help='Do not save recordings to disk')
    parser.add_argument('--show-raw', action='store_true', help='Show raw Whisper transcription')
    args = parser.parse_args()

    try:
        asyncio.run(main(save=not args.no_save, show_raw=args.show_raw))
    except KeyboardInterrupt:
        print("\n👋 Stopped.")
