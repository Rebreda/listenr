#!/usr/bin/env python3
"""
Listenr CLI: Real-time Microphone Streaming to Lemonade ASR

Streams microphone to Lemonade's /realtime WebSocket endpoint.
Prints transcriptions as they arrive. Saves audio + transcripts for dataset use.
All config loaded from config_manager.

Usage:
    listenr
    listenr --no-save       # Don't save recordings
    listenr --show-raw      # Show raw Whisper transcription too
    listenr --debug         # Verbose debug output
"""

import asyncio
import logging
import argparse
import json
import requests
import numpy as np
import sounddevice as sd
from collections import deque
from math import gcd
from scipy.signal import resample_poly

from listenr.unified_asr import LemonadeUnifiedASR
from listenr.llm_processor import lemonade_llm_correct, lemonade_load_model, lemonade_unload_models
from listenr.transcript_utils import is_hallucination, strip_noise_tags
from listenr.storage import save_recording, patch_manifest_record
from listenr.retranscribe import retranscribe_clip
from listenr.constants import (
    ASR_RATE,
    CAPTURE_RATE,
    CHANNELS,
    CHUNK_SIZE,
    INPUT_DEVICE,
    LLM_CONTEXT_WINDOW,
    LLM_ENABLED as USE_LLM,
    LLM_MODEL,
    STORAGE_BASE,
    VAD_MAX_SEGMENT_S,
    VAD_SILENCE_MS,
    VAD_THRESHOLD,
    WHISPER_MODEL,
)

logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
log = logging.getLogger('listenr.cli')

# Compute resample ratio once (e.g. 48000→16000 = up 1, down 3)
_gcd = gcd(CAPTURE_RATE, ASR_RATE)
_RESAMPLE_UP = ASR_RATE // _gcd
_RESAMPLE_DOWN = CAPTURE_RATE // _gcd
_NEED_RESAMPLE = (CAPTURE_RATE != ASR_RATE)


def get_lemonade_ws_url() -> str:
    """Discover Lemonade WebSocket URL from /api/v1/health."""
    from listenr.constants import LLM_API_BASE
    health_url = LLM_API_BASE.rstrip('/').replace('/api/v1', '') + '/api/v1/health'
    try:
        resp = requests.get(health_url, timeout=2)
        resp.raise_for_status()
        data = resp.json()
        ws_port = data.get('websocket_port', 8001)
        return f"ws://localhost:{ws_port}/realtime"
    except Exception as e:
        print(f"Could not discover Lemonade websocket port ({e}), using default :8001")
        return "ws://localhost:8001/realtime"


def ensure_models_loaded(debug: bool = False) -> None:
    """
    Load the Whisper model (and optionally the LLM) via POST /api/v1/load.
    This is idempotent — Lemonade is a no-op if the model is already loaded.
    """
    print(f"Loading Whisper model '{WHISPER_MODEL}' in Lemonade...", flush=True)
    try:
        result = lemonade_load_model(WHISPER_MODEL)
        print(f"Whisper ready: {result.get('message', result)}")
    except requests.HTTPError as e:
        print(f"ERROR: Failed to load Whisper model '{WHISPER_MODEL}': {e.response.text}")
        raise
    except Exception as e:
        print(f"ERROR: Failed to load Whisper model '{WHISPER_MODEL}': {e}")
        raise

    if USE_LLM:
        print(f"Loading LLM '{LLM_MODEL}' in Lemonade...", flush=True)
        try:
            result = lemonade_load_model(LLM_MODEL)
            print(f"LLM ready: {result.get('message', result)}")
        except requests.HTTPError as e:
            print(f"WARNING: Failed to load LLM '{LLM_MODEL}': {e.response.text} -- LLM correction disabled")
        except Exception as e:
            print(f"WARNING: Failed to load LLM '{LLM_MODEL}': {e} -- LLM correction disabled")


async def mic_stream(pcm_buffer: list, debug: bool = False):
    """Async generator yielding 16kHz PCM16 bytes from mic, also buffering for saving.

    Captures at CAPTURE_RATE (from config) and resamples to ASR_RATE (16kHz) if needed.
    stream.read() is offloaded to a thread pool executor so it never blocks the asyncio
    event loop — otherwise _recv_messages would never get a turn to process incoming
    WebSocket messages.
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
            audio_chunk, overflowed = await loop.run_in_executor(
                None, stream.read, CHUNK_SIZE
            )
            if overflowed and debug:
                print("  [DEBUG] WARNING: Mic buffer overflowed (CPU too slow?)")
            mono = audio_chunk[:, 0] if audio_chunk.ndim > 1 else audio_chunk
            rms = float(np.sqrt(np.mean(mono ** 2)))
            if _NEED_RESAMPLE:
                mono = resample_poly(mono, _RESAMPLE_UP, _RESAMPLE_DOWN).astype(np.float32)
            pcm16 = (np.clip(mono, -1.0, 1.0) * 32767.0).astype('<i2').tobytes()
            pcm_buffer.append(pcm16)
            chunks_sent += 1
            if debug and chunks_sent % 24 == 0:
                print(f"  [DEBUG] Mic: {chunks_sent} chunks sent, RMS={rms:.4f}, "
                      f"pcm16_bytes={len(pcm16)}", flush=True)
            yield pcm16


async def _run(save: bool, show_raw: bool, debug: bool, retranscribe: bool = False):
    ensure_models_loaded(debug=debug)
    ws_url = get_lemonade_ws_url()
    ws_url_with_model = f"{ws_url}?model={WHISPER_MODEL}"
    print(f"Listenr CLI -- streaming to Lemonade")
    print(f"   Model  : {WHISPER_MODEL}")
    print(f"   WS URL : {ws_url_with_model}")
    print(f"   LLM    : {'enabled (' + LLM_MODEL + ')' if USE_LLM else 'disabled'}")
    print(f"   Save   : {'yes -> ' + str(STORAGE_BASE) if save else 'no'}")
    print(f"   VAD    : silence={VAD_SILENCE_MS}ms  threshold={VAD_THRESHOLD}  max_segment={VAD_MAX_SEGMENT_S}s")
    print(f"   Debug  : {'on' if debug else 'off  (use --debug to enable)'}")
    print(f"   Batch  : {'on (batch retranscribe after save)' if retranscribe else 'off'}")
    print("   Press Ctrl+C to stop.\n")

    if debug:
        logging.getLogger('listenr.cli').setLevel(logging.DEBUG)
        logging.getLogger('unified_asr').setLevel(logging.DEBUG)
        logging.getLogger('lemonade_unified_asr').setLevel(logging.DEBUG)
        logging.getLogger('websockets').setLevel(logging.INFO)
        logging.root.setLevel(logging.DEBUG)

    asr = LemonadeUnifiedASR(use_llm=False)  # LLM correction handled here for saving
    pcm_buffer: list = []
    # Rolling window of (raw, corrected) pairs passed as context to the LLM
    llm_context: deque[tuple[str, str]] = deque(maxlen=LLM_CONTEXT_WINDOW)

    async for result in asr.stream_transcribe(
        mic_stream(pcm_buffer, debug=debug),
        lemonade_ws_url=ws_url_with_model,
        debug=debug,
        max_segment_s=VAD_MAX_SEGMENT_S,
    ):
        msg_type = result.get('type', '')

        if debug and msg_type not in (
            'conversation.item.input_audio_transcription.completed',
            'conversation.item.input_audio_transcription.delta',
            'input_audio_buffer.speech_started',
            'error',
        ):
            print(f"  [DEBUG] <- {msg_type}: {json.dumps(result)}", flush=True)

        if msg_type == 'conversation.item.input_audio_transcription.delta':
            interim = result.get('transcript', '').strip()
            if interim:
                print(f"\r  [ASR] {interim} ...", end='', flush=True)

        elif msg_type == 'conversation.item.input_audio_transcription.completed':
            raw_text = result.get('transcript', '').strip()
            if not raw_text:
                continue

            if is_hallucination(raw_text):
                print('\r' + ' ' * 80 + '\r', end='', flush=True)
                if debug:
                    print(f"  [DEBUG] hallucination dropped: {raw_text!r}", flush=True)
                pcm_buffer.clear()
                continue

            stripped_text = strip_noise_tags(raw_text)
            if not stripped_text:
                print('\r' + ' ' * 80 + '\r', end='', flush=True)
                if debug:
                    print(f"  [DEBUG] all-noise stripped: {raw_text!r}", flush=True)
                pcm_buffer.clear()
                continue
            if stripped_text != raw_text and debug:
                print(f"  [DEBUG] noise tags stripped: {raw_text!r} -> {stripped_text!r}", flush=True)
            raw_text = stripped_text

            print('\r' + ' ' * 80 + '\r', end='', flush=True)

            corrected_text = raw_text
            is_improved = False
            categories: list = []

            if USE_LLM:
                llm_result = lemonade_llm_correct(
                    raw_text,
                    model=LLM_MODEL,
                    recent_context=list(llm_context),
                )
                corrected_text = llm_result['corrected']
                is_improved = llm_result['is_improved']
                categories = llm_result.get('categories', [])
                if 'error' in llm_result and debug:
                    print(f"  WARNING: LLM error: {llm_result['error']}")

            # Add to context window regardless of whether LLM was used
            llm_context.append((raw_text, corrected_text))

            if show_raw and is_improved:
                print(f"  [RAW] {raw_text}")
            cats = f"  [{', '.join(categories)}]" if categories else ""
            print(f"  [ASR] {corrected_text}{cats}")

            if save:
                if pcm_buffer:
                    record = save_recording(
                        list(pcm_buffer), raw_text, corrected_text,
                        storage_base=STORAGE_BASE,
                        asr_rate=ASR_RATE,
                        whisper_model=WHISPER_MODEL,
                        llm_model=LLM_MODEL,
                        is_improved=is_improved,
                        categories=categories,
                    )
                    print(f"  [SAVED] {record['audio_path']} ({record['duration_s']}s)")
                    if retranscribe:
                        try:
                            patch_fields = retranscribe_clip(
                                record['audio_path'],
                                model=WHISPER_MODEL,
                                use_llm=USE_LLM,
                                llm_model=LLM_MODEL,
                                llm_context=list(llm_context),
                            )
                            if patch_fields is not None and patch_fields['raw_transcription'] != raw_text:
                                patch_manifest_record(
                                    STORAGE_BASE / 'manifest.jsonl',
                                    record['uuid'],
                                    patch_fields,
                                )
                                if show_raw:
                                    print(f"  [BATCH] {patch_fields['raw_transcription']}")
                        except Exception as exc:
                            if debug:
                                print(f"  [DEBUG] batch retranscribe failed: {exc}")
                    pcm_buffer.clear()
                else:
                    print(f"  WARNING: pcm_buffer is empty -- no audio captured for this segment", flush=True)

        elif msg_type == 'input_audio_buffer.speech_started':
            if debug:
                print(f"  [DEBUG] speech_started -- clearing pcm_buffer (had {len(pcm_buffer)} chunks)", flush=True)
            pcm_buffer.clear()

        elif msg_type == 'error':
            print(f"  ERROR: {result.get('message', result)}")


def main():
    parser = argparse.ArgumentParser(description='Listenr CLI — real-time ASR via Lemonade')
    parser.add_argument('--no-save', action='store_true', help='Do not save recordings to disk')
    parser.add_argument('--show-raw', action='store_true', help='Show raw Whisper transcription')
    parser.add_argument('--retranscribe', action='store_true',
                        help='After saving each clip, re-transcribe with batch endpoint for higher accuracy')
    parser.add_argument('--debug', action='store_true', help='Verbose debug output (WS messages, mic RMS, etc.)')
    args = parser.parse_args()

    try:
        asyncio.run(_run(save=not args.no_save, show_raw=args.show_raw, debug=args.debug, retranscribe=args.retranscribe))
    except KeyboardInterrupt:
        print("\nUnloading models from Lemonade...", flush=True)
        result = lemonade_unload_models()
        if 'error' in result:
            print(f"  WARNING: Unload failed: {result['error']}")
        else:
            print(f"  Models unloaded.")
        print("Stopped.")


if __name__ == '__main__':
    main()
