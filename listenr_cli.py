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
from math import gcd
from scipy.signal import resample_poly
from pathlib import Path
from datetime import datetime, timezone

from unified_asr import LemonadeUnifiedASR
from llm_processor import lemonade_llm_correct, lemonade_load_model, lemonade_unload_models
import config_manager as cfg

logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
log = logging.getLogger('listenr_cli')

# Audio settings from config
CAPTURE_RATE = cfg.get_int_setting('Audio', 'sample_rate', 16000)  # mic capture rate
ASR_RATE = 16000  # Lemonade /realtime always requires 16kHz PCM16
CHUNK_SIZE = cfg.get_int_setting('Audio', 'blocksize', 1360)
SAMPLE_RATE = CAPTURE_RATE  # kept for backward compat (save_recording uses ASR_RATE directly)
CHANNELS = cfg.get_int_setting('Audio', 'channels', 1)
INPUT_DEVICE = cfg.get_setting('Audio', 'input_device', 'default') or None
if INPUT_DEVICE == 'default':
    INPUT_DEVICE = None

# Compute resample ratio once (e.g. 44100→16000 = up 160, down 441)
_gcd = gcd(CAPTURE_RATE, ASR_RATE)
_RESAMPLE_UP = ASR_RATE // _gcd
_RESAMPLE_DOWN = CAPTURE_RATE // _gcd
_NEED_RESAMPLE = (CAPTURE_RATE != ASR_RATE)

# Storage
STORAGE_BASE = Path(
    cfg.get_setting('Storage', 'audio_clips_path', '~/listenr_recordings') or '~/listenr_recordings'
).expanduser()

# LLM settings
USE_LLM = cfg.get_bool_setting('LLM', 'enabled', False)
LLM_MODEL = cfg.get_setting('LLM', 'model', 'gpt-oss-20b-mxfp4-GGUF') or 'gpt-oss-20b-mxfp4-GGUF'
WHISPER_MODEL = cfg.get_setting('Whisper', 'model', 'Whisper-Large-v3-Turbo') or 'Whisper-Large-v3-Turbo'


def get_lemonade_ws_url() -> str:
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


def ensure_models_loaded(debug: bool = False) -> None:
    """
    Load the Whisper model (and optionally the LLM) via POST /api/v1/load.
    This is idempotent — Lemonade is a no-op if the model is already loaded.
    """
    print(f"⏳ Loading Whisper model '{WHISPER_MODEL}' in Lemonade...", flush=True)
    try:
        result = lemonade_load_model(WHISPER_MODEL)
        print(f"✅ Whisper: {result.get('message', result)}")
    except requests.HTTPError as e:
        print(f"❌ Failed to load Whisper model '{WHISPER_MODEL}': {e.response.text}")
        raise
    except Exception as e:
        print(f"❌ Failed to load Whisper model '{WHISPER_MODEL}': {e}")
        raise

    if USE_LLM:
        print(f"⏳ Loading LLM '{LLM_MODEL}' in Lemonade...", flush=True)
        try:
            result = lemonade_load_model(LLM_MODEL)
            print(f"✅ LLM: {result.get('message', result)}")
        except requests.HTTPError as e:
            print(f"⚠️  Failed to load LLM '{LLM_MODEL}': {e.response.text} — LLM correction disabled")
        except Exception as e:
            print(f"⚠️  Failed to load LLM '{LLM_MODEL}': {e} — LLM correction disabled")


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

    # Reconstruct float32 audio from pcm16 bytes (mono, little-endian int16) and save
    audio_np = np.frombuffer(b''.join(pcm_frames), dtype='<i2').astype(np.float32) / 32767.0
    sf.write(str(audio_path), audio_np, ASR_RATE, subtype='PCM_16')

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
        'duration_s': round(len(audio_np) / ASR_RATE, 3),
        'sample_rate': ASR_RATE,
    }
    with open(transcript_path, 'w') as f:
        json.dump(record, f, indent=2)

    return record


async def mic_stream(pcm_buffer: list, debug: bool = False):
    """Async generator yielding 16kHz PCM16 bytes from mic, also buffering for saving.

    Captures at CAPTURE_RATE (from config) and resamples to ASR_RATE (16kHz) if needed.
    PipeWire and most USB mics require their native rate (e.g. 44100Hz); Lemonade always
    expects 16kHz PCM16, so we resample here before sending.

    stream.read() is a blocking call (~85ms per chunk). We offload it to a thread pool
    executor so it never blocks the asyncio event loop — otherwise _recv_messages in
    stream_transcribe would never get a turn to process incoming WebSocket messages.
    """
    loop = asyncio.get_event_loop()
    chunks_sent = 0
    with sd.InputStream(
        samplerate=CAPTURE_RATE,
        channels=CHANNELS,
        dtype='float32',
        blocksize=CHUNK_SIZE,
        device=INPUT_DEVICE
    ) as stream:
        if debug:
            resample_note = f" → resampled to {ASR_RATE}Hz" if _NEED_RESAMPLE else ""
            print(f"  [DEBUG] Mic open: {CAPTURE_RATE}Hz{resample_note}, {CHANNELS}ch, "
                  f"blocksize={CHUNK_SIZE} (~{CHUNK_SIZE/CAPTURE_RATE*1000:.0f}ms/chunk), "
                  f"device={stream.device}")
        while True:
            # Run the blocking read in a thread so the event loop stays free
            audio_chunk, overflowed = await loop.run_in_executor(
                None, stream.read, CHUNK_SIZE
            )
            if overflowed and debug:
                print("  [DEBUG] ⚠️  Mic buffer overflowed (CPU too slow?)")
            # audio_chunk shape: (blocksize, channels) — squeeze to 1D mono
            mono = audio_chunk[:, 0] if audio_chunk.ndim > 1 else audio_chunk
            rms = float(np.sqrt(np.mean(mono ** 2)))
            # Resample to 16kHz if capture rate differs
            if _NEED_RESAMPLE:
                mono = resample_poly(mono, _RESAMPLE_UP, _RESAMPLE_DOWN).astype(np.float32)
            # Convert float32 [-1, 1] → int16 PCM, little-endian bytes
            pcm16 = (np.clip(mono, -1.0, 1.0) * 32767.0).astype('<i2').tobytes()
            pcm_buffer.append(pcm16)
            chunks_sent += 1
            if debug and chunks_sent % 24 == 0:  # print every ~2 seconds
                print(f"  [DEBUG] Mic: {chunks_sent} chunks sent, RMS={rms:.4f}, "
                      f"pcm16_bytes={len(pcm16)}", flush=True)
            yield pcm16


async def main(save: bool, show_raw: bool, debug: bool):
    ensure_models_loaded(debug=debug)
    ws_url = get_lemonade_ws_url()
    ws_url_with_model = f"{ws_url}?model={WHISPER_MODEL}"
    print(f"🎤 Listenr CLI — streaming to Lemonade")
    print(f"   Model  : {WHISPER_MODEL}")
    print(f"   WS URL : {ws_url_with_model}")
    print(f"   LLM    : {'enabled (' + LLM_MODEL + ')' if USE_LLM else 'disabled'}")
    print(f"   Save   : {'yes → ' + str(STORAGE_BASE) if save else 'no'}")
    print(f"   Debug  : {'on' if debug else 'off  (use --debug to enable)'}")
    print("   Press Ctrl+C to stop.\n")

    if debug:
        # Raise log level for websockets library too
        logging.getLogger('listenr_cli').setLevel(logging.DEBUG)
        logging.getLogger('unified_asr').setLevel(logging.DEBUG)
        logging.getLogger('lemonade_unified_asr').setLevel(logging.DEBUG)
        logging.getLogger('websockets').setLevel(logging.INFO)
        logging.root.setLevel(logging.DEBUG)

    asr = LemonadeUnifiedASR(use_llm=False)  # We handle LLM correction ourselves for saving
    pcm_buffer: list = []
    pending_raw: list = []  # buffer of raw texts in current speech segment

    async for result in asr.stream_transcribe(
        mic_stream(pcm_buffer, debug=debug),
        lemonade_ws_url=ws_url_with_model,
        debug=debug,
    ):
        msg_type = result.get('type', '')

        if debug and msg_type not in (
            'conversation.item.input_audio_transcription.completed',
            'conversation.item.input_audio_transcription.delta',
            'input_audio_buffer.speech_started',
            'error',
        ):
            print(f"  [DEBUG] ← {msg_type}: {json.dumps(result)}", flush=True)

        if msg_type == 'conversation.item.input_audio_transcription.delta':
            # Interim result — print in-place so the line updates as Whisper refines it
            interim = result.get('transcript', '').strip()
            if interim:
                print(f"\r  [ASR] {interim} …", end='', flush=True)

        elif msg_type == 'conversation.item.input_audio_transcription.completed':
            raw_text = result.get('transcript', '').strip()
            if not raw_text:
                continue

            # Clear the interim line
            print('\r' + ' ' * 80 + '\r', end='', flush=True)

            corrected_text = raw_text
            if USE_LLM:
                try:
                    corrected_text = lemonade_llm_correct(raw_text, model=LLM_MODEL)
                except Exception as e:
                    print(f"  ⚠️  LLM correction failed: {e}")

            if show_raw and corrected_text != raw_text:
                print(f"  [RAW] {raw_text}")
            print(f"  [ASR] {corrected_text}")

            if save:
                if pcm_buffer:
                    record = save_recording(list(pcm_buffer), raw_text, corrected_text)
                    print(f"  [SAVED] {record['audio_path']} ({record['duration_s']}s, improved={record['is_improved']})")
                    pcm_buffer.clear()
                else:
                    print(f"  ⚠️  [SAVE SKIPPED] pcm_buffer is empty — no audio captured for this segment", flush=True)

        elif msg_type == 'input_audio_buffer.speech_started':
            if debug:
                print(f"  [DEBUG] 🗣  speech_started — clearing pcm_buffer (had {len(pcm_buffer)} chunks)", flush=True)
            pcm_buffer.clear()  # start fresh buffer for this segment

        elif msg_type == 'error':
            print(f"  ❌ Error from Lemonade: {result.get('message', result)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Listenr CLI — real-time ASR via Lemonade')
    parser.add_argument('--no-save', action='store_true', help='Do not save recordings to disk')
    parser.add_argument('--show-raw', action='store_true', help='Show raw Whisper transcription')
    parser.add_argument('--debug', action='store_true', help='Verbose debug output (WS messages, mic RMS, etc.)')
    args = parser.parse_args()

    try:
        asyncio.run(main(save=not args.no_save, show_raw=args.show_raw, debug=args.debug))
    except KeyboardInterrupt:
        print("\n⏳ Unloading models from Lemonade...", flush=True)
        result = lemonade_unload_models()
        if 'error' in result:
            print(f"  ⚠️  Unload failed: {result['error']}")
        else:
            print(f"  ✅ Models unloaded.")
        print("👋 Stopped.")
