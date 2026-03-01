#!/usr/bin/env python3
"""
Unified ASR System - Works for both CLI and Web


This module provides a single ASR implementation that:
- Returns JSON responses for all transcription results
- Works seamlessly for both command-line and web usage
- Supports both batch (file) and streaming (real-time) modes (via Lemonade HTTP and WebSocket endpoints)
- Includes optional LLM post-processing
- Handles storage and metadata consistently

Usage:
    # CLI mode
    from unified_asr import UnifiedASR
    asr = UnifiedASR()
    asr.start_cli()

    # Web mode (process single audio)
    asr = UnifiedASR(mode='web')
    result = asr.process_audio(audio_data, sample_rate)

    # Streaming mode (continuous)
    asr = UnifiedASR(mode='stream')
    asr.start_stream(callback=my_callback)
"""

import os
import sys
import json
import uuid
import time
import queue
import threading
import tempfile
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List


import numpy as np
import soundfile as sf
import requests
import websockets
import asyncio

# Import config manager
import config_manager as cfg


# Import Lemonade LLM and ASR functions
from llm_processor import lemonade_llm_correct, lemonade_transcribe_audio

logger = logging.getLogger('unified_asr')



# --- LemonadeUnifiedASR: Use Lemonade Server for ASR and LLM ---

class LemonadeUnifiedASR:
    """Unified ASR using Lemonade Server for both ASR and LLM"""
    def __init__(self, use_llm=True):
        self.use_llm = use_llm
        import logging
        self.logger = logging.getLogger('lemonade_unified_asr')

    def transcribe_and_correct(self, audio_path, whisper_model="Whisper-Tiny", llm_model="Qwen3-0.6B-GGUF", system_prompt=None):
        try:
            raw_text = lemonade_transcribe_audio(audio_path, model=whisper_model)
            corrected_text = None
            if self.use_llm:
                corrected_text = lemonade_llm_correct(raw_text, model=llm_model, system_prompt=system_prompt)
            else:
                corrected_text = raw_text
            return {"raw_text": raw_text, "corrected_text": corrected_text}
        except Exception as e:
            self.logger.error(f"Transcription or LLM correction failed: {e}")
            return {"error": str(e)}

    async def stream_transcribe(self, audio_stream, whisper_model=None, on_result=None, lemonade_ws_url=None):
        """
        Stream audio to Lemonade's /realtime WebSocket endpoint and yield transcriptions.
        audio_stream: async generator yielding bytes (PCM or WAV chunks)
        on_result: optional callback for each result
        lemonade_ws_url: override Lemonade WebSocket URL (default from config)
        """
        import websockets
        import json
        # Use config defaults if not provided
        if whisper_model is None:
            whisper_model = cfg.get_setting('Whisper', 'model', 'Whisper-Tiny')
        if lemonade_ws_url is None:
            # Try both Lemonade and LLM config sections for endpoint
            lemonade_ws_url = cfg.get_setting('Lemonade', 'realtime_ws_url', None)
            if not lemonade_ws_url:
                api_base = cfg.get_setting('LLM', 'api_base', None)
                if api_base:
                    lemonade_ws_url = api_base.replace('/api/v1', '/realtime')
                else:
                    lemonade_ws_url = 'ws://localhost:8000/realtime'
        params = {"model": whisper_model}
        url = lemonade_ws_url + "?" + "&".join(f"{k}={v}" for k,v in params.items())
        async with websockets.connect(url, max_size=10*1024*1024) as ws:
            async for chunk in audio_stream:
                await ws.send(chunk)
                response = await ws.recv()
                result = json.loads(response)
                if on_result:
                    on_result(result)
                yield result
            await ws.send("__END__")
            # Final result
            response = await ws.recv()
            result = json.loads(response)
            if on_result:
                on_result(result)
            yield result

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Lemonade Unified ASR - CLI mode')
    parser.add_argument('--llm', action='store_true', help='Enable LLM post-processing')
    parser.add_argument('--audio', type=str, help='Path to audio file to transcribe')
    parser.add_argument('--whisper-model', type=str, default='Whisper-Tiny', help='Whisper model name')
    parser.add_argument('--llm-model', type=str, default='Qwen3-0.6B-GGUF', help='LLM model name')
    parser.add_argument('--system-prompt', type=str, default=None, help='System prompt for LLM')
    args = parser.parse_args()

    asr = LemonadeUnifiedASR(use_llm=args.llm)
    if args.audio:
        result = asr.transcribe_and_correct(
            args.audio,
            whisper_model=args.whisper_model,
            llm_model=args.llm_model,
            system_prompt=args.system_prompt
        )
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"\n[RESULT] Raw: {result['raw_text']}\n[RESULT] Corrected: {result['corrected_text']}")
    else:
        print("Please provide --audio path/to/audio.wav to transcribe.")
