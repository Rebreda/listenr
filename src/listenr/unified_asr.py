#!/usr/bin/env python3
"""
Unified ASR System - Works for both CLI and Web

This module provides a single ASR implementation that:
- Returns JSON responses for all transcription results
- Works seamlessly for both command-line and web usage
- Supports both batch (file) and streaming (real-time) modes (via Lemonade HTTP and WebSocket endpoints)
- Includes optional LLM post-processing
- Handles storage and metadata consistently
"""

import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import requests
import websockets
import asyncio

import listenr.config_manager as cfg
from listenr.llm_processor import lemonade_llm_correct, lemonade_transcribe_audio

logger = logging.getLogger('unified_asr')


# --- LemonadeUnifiedASR: Use Lemonade Server for ASR and LLM ---

class LemonadeUnifiedASR:
    """Unified ASR using Lemonade Server for both ASR and LLM."""

    def __init__(self, use_llm=True):
        self.use_llm = use_llm
        self.logger = logging.getLogger('lemonade_unified_asr')

    def transcribe_and_correct(self, audio_path, whisper_model="Whisper-Tiny", llm_model="Qwen3-0.6B-GGUF", system_prompt=None):
        try:
            raw_text = lemonade_transcribe_audio(audio_path, model=whisper_model)
            corrected_text = None
            if self.use_llm:
                corrected_text = lemonade_llm_correct(raw_text, model=llm_model)
            else:
                corrected_text = raw_text
            return {"raw_text": raw_text, "corrected_text": corrected_text}
        except Exception as e:
            self.logger.error(f"Transcription or LLM correction failed: {e}")
            return {"error": str(e)}

    async def stream_transcribe(self, audio_stream, whisper_model=None, on_result=None, lemonade_ws_url=None, debug=False):
        """
        Stream audio to Lemonade's /realtime WebSocket endpoint and yield transcriptions.

        audio_stream: async generator yielding raw PCM16 bytes chunks (16kHz, mono)
        on_result: optional callback for each result dict
        lemonade_ws_url: full WebSocket URL including ?model= param (overrides config)
        debug: if True, all server messages are yielded (not just transcriptions/errors)

        Lemonade /realtime protocol:
          Client → Server:
            session.update           — configure VAD settings
            input_audio_buffer.append — base64-encoded PCM16 audio chunk
            input_audio_buffer.commit — force transcription of remaining buffer
          Server → Client:
            session.created / session.updated
            input_audio_buffer.speech_started / speech_stopped / committed / cleared
            conversation.item.input_audio_transcription.delta  — partial
            conversation.item.input_audio_transcription.completed — final
            error
        """
        import base64

        if whisper_model is None:
            whisper_model = cfg.get_setting('Whisper', 'model', 'Whisper-Large-v3-Turbo')
        if lemonade_ws_url is None:
            api_base = cfg.get_setting('LLM', 'api_base', 'http://localhost:8000/api/v1') or 'http://localhost:8000/api/v1'
            try:
                resp = requests.get(f"{api_base}/health", timeout=5)
                ws_port = resp.json().get('websocket_port', 8001)
            except Exception:
                ws_port = 8001
            lemonade_ws_url = f"ws://localhost:{ws_port}/realtime?model={whisper_model}"

        session_update = {
            "type": "session.update",
            "session": {
                "model": whisper_model,
                "turn_detection": {
                    "threshold": cfg.get_float_setting('VAD', 'threshold', 0.01),
                    "silence_duration_ms": cfg.get_int_setting('VAD', 'silence_duration_ms', 800),
                    "prefix_padding_ms": cfg.get_int_setting('VAD', 'prefix_padding_ms', 250),
                },
            },
        }
        if debug:
            print(f"  [DEBUG] Connecting to {lemonade_ws_url}")
            print(f"  [DEBUG] session.update → {json.dumps(session_update)}")

        async with websockets.connect(lemonade_ws_url, max_size=10 * 1024 * 1024) as ws:
            await ws.send(json.dumps(session_update))

            result_queue: asyncio.Queue = asyncio.Queue()

            async def _send_audio():
                chunks = 0
                async for chunk in audio_stream:
                    b64_audio = base64.b64encode(chunk).decode('utf-8')
                    await ws.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": b64_audio,
                    }))
                    chunks += 1
                if debug:
                    print(f"  [DEBUG] Audio stream ended after {chunks} chunks, sending commit")
                await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                await result_queue.put(None)  # sentinel: sender is done

            _ALWAYS_FORWARD = {
                'conversation.item.input_audio_transcription.completed',
                'conversation.item.input_audio_transcription.delta',
                'input_audio_buffer.speech_started',
                'error',
            }
            _DEBUG_FORWARD = {
                'session.created',
                'session.updated',
                'input_audio_buffer.speech_stopped',
                'input_audio_buffer.committed',
                'input_audio_buffer.cleared',
            }

            async def _recv_messages():
                async for raw in ws:
                    msg = json.loads(raw)
                    msg_type = msg.get('type', '')
                    if msg_type in _ALWAYS_FORWARD:
                        await result_queue.put(msg)
                    elif debug and msg_type in _DEBUG_FORWARD:
                        await result_queue.put(msg)
                    elif msg_type in ('response.done', 'session.closed'):
                        if debug:
                            print(f"  [DEBUG] WS closed by server: {msg_type}")
                        break
                    elif debug:
                        print(f"  [DEBUG] Ignored message type: {msg_type}")

            send_task = asyncio.ensure_future(_send_audio())
            recv_task = asyncio.ensure_future(_recv_messages())

            try:
                while True:
                    item = await result_queue.get()
                    if item is None:
                        # Drain any remaining messages briefly after sender finishes
                        try:
                            while True:
                                item = await asyncio.wait_for(result_queue.get(), timeout=3.0)
                                if item is None:
                                    break
                                if on_result:
                                    on_result(item)
                                yield item
                        except asyncio.TimeoutError:
                            pass
                        break
                    if on_result:
                        on_result(item)
                    yield item
            finally:
                send_task.cancel()
                recv_task.cancel()
                for t in (send_task, recv_task):
                    try:
                        await t
                    except (asyncio.CancelledError, Exception):
                        pass


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Lemonade Unified ASR — CLI mode')
    parser.add_argument('--llm', action='store_true', help='Enable LLM post-processing')
    parser.add_argument('--audio', type=str, help='Path to audio file to transcribe')
    parser.add_argument('--whisper-model', type=str, default='Whisper-Tiny', help='Whisper model name')
    parser.add_argument('--llm-model', type=str, default='Qwen3-0.6B-GGUF', help='LLM model name')
    args = parser.parse_args()

    asr = LemonadeUnifiedASR(use_llm=args.llm)
    if args.audio:
        result = asr.transcribe_and_correct(
            args.audio,
            whisper_model=args.whisper_model,
            llm_model=args.llm_model,
        )
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"\n[RESULT] Raw: {result['raw_text']}\n[RESULT] Corrected: {result['corrected_text']}")
    else:
        print("Please provide --audio path/to/audio.wav to transcribe.")
