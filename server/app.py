#!/usr/bin/env python3
"""
Listenr Web Server - Live Streaming Transcription

Real-time audio transcription via Lemonade Server WebSocket API.
Browser captures microphone → Lemonade Whisper → optional LLM correction.

Environment variables:
    LISTENR_STORAGE: Base directory for storage (default from config)
    LISTENR_PORT: Port to run on (default: 5000)
    LISTENR_HOST: Host to bind to (default: 0.0.0.0)
    LISTENR_USE_LLM: Enable LLM post-processing (default: false)

Usage:
    python server/app.py
"""

import os
import sys
import asyncio
import logging
import json
import base64
from pathlib import Path
from datetime import datetime, timezone

from flask import Flask, render_template_string, jsonify, request
from flask_sock import Sock
import requests

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from unified_asr import LemonadeUnifiedASR
import config_manager as cfg

# ---------------------------
# Configuration
# ---------------------------
STORAGE_BASE = Path(os.environ.get(
    'LISTENR_STORAGE',
    cfg.get_setting('Storage', 'audio_clips_path', '~/listenr_recordings')
)).expanduser()
PORT = int(os.environ.get('LISTENR_PORT', '5000'))
HOST = os.environ.get('LISTENR_HOST', '0.0.0.0')
USE_LLM = os.environ.get('LISTENR_USE_LLM', 'false').lower() in ('true', '1', 'yes')

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger('listenr_server')

# ---------------------------
# Flask App
# ---------------------------
app = Flask(__name__, static_folder='static', template_folder='templates')
sock = Sock(app)


def get_lemonade_ws_url():
    """Discover Lemonade WebSocket URL dynamically via /api/v1/health."""
    api_base = cfg.get_setting('LLM', 'api_base', 'http://localhost:8000/api/v1') or 'http://localhost:8000/api/v1'
    health_url = api_base.rstrip('/').replace('/api/v1', '') + '/api/v1/health'
    model = cfg.get_setting('Whisper', 'model', 'Whisper-Tiny')
    try:
        resp = requests.get(health_url, timeout=2)
        resp.raise_for_status()
        data = resp.json()
        ws_port = data.get('websocket_port', 8001)
        return f"ws://localhost:{ws_port}/realtime?model={model}"
    except Exception as e:
        logger.warning(f"Could not discover Lemonade websocket port: {e}")
        return f"ws://localhost:8001/realtime?model={model}"


def get_asr():
    return LemonadeUnifiedASR(use_llm=USE_LLM)


# ---------------------------
# Config API for frontend
# ---------------------------
@app.route('/api/config')
def api_config():
    """Return Lemonade WebSocket URL and model for the frontend."""
    ws_url = get_lemonade_ws_url()
    model = cfg.get_setting('Whisper', 'model', 'Whisper-Tiny')
    return jsonify({
        'ws_url': ws_url,
        'model': model,
        'llm_enabled': USE_LLM,
    })


# ---------------------------
# WebSocket Handler
# ---------------------------
@sock.route('/transcribe')
def transcribe_websocket(ws):
    """
    WebSocket endpoint: receives audio from browser, streams to Lemonade,
    returns transcription results.

    Browser sends:
        {"type": "audio", "audio": "<base64 PCM16>"}
        {"type": "ping"}

    Server sends:
        {"type": "status", "message": "connected", ...}
        {"type": "transcription", "data": {...}}
        {"type": "error", "message": "..."}
    """
    logger.info(f"WebSocket connection from {request.remote_addr}")
    asr = get_asr()

    try:
        ws.send(json.dumps({
            'type': 'status',
            'message': 'connected',
            'llm_enabled': USE_LLM
        }))

        ws_url = get_lemonade_ws_url()
        logger.info(f"Forwarding to Lemonade: {ws_url}")

        async def audio_stream():
            while True:
                try:
                    message = ws.receive(timeout=30)
                except Exception:
                    break
                if message is None:
                    break
                try:
                    data = json.loads(message)
                    if data.get('type') == 'audio':
                        yield base64.b64decode(data['audio'])
                    elif data.get('type') == 'ping':
                        ws.send(json.dumps({'type': 'pong'}))
                    elif data.get('type') == 'stop':
                        break
                except Exception:
                    break

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            async def forward():
                async for result in asr.stream_transcribe(audio_stream(), lemonade_ws_url=ws_url):
                    try:
                        ws.send(json.dumps({'type': 'transcription', 'data': result}))
                    except Exception:
                        break
            loop.run_until_complete(forward())
        finally:
            loop.close()

    except Exception as e:
        logger.exception(f"WebSocket error: {e}")
        try:
            ws.send(json.dumps({'type': 'error', 'message': str(e)}))
        except Exception:
            pass
    finally:
        logger.info("WebSocket connection closed")


# ---------------------------
# HTTP Routes
# ---------------------------
@app.route('/')
def index():
    with open(Path(__file__).parent / 'templates' / 'index_visual.html', 'r') as f:
        return render_template_string(f.read())


@app.route('/simple')
def simple():
    with open(Path(__file__).parent / 'templates' / 'index.html', 'r') as f:
        return render_template_string(f.read())


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'time': datetime.now(timezone.utc).isoformat(),
        'storage': str(STORAGE_BASE),
        'llm_enabled': USE_LLM,
    })


# ---------------------------
# Security Headers
# ---------------------------
@app.after_request
def set_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['Referrer-Policy'] = 'no-referrer-when-downgrade'
    return response


# ---------------------------
# Main
# ---------------------------
if __name__ == '__main__':
    logger.info('=' * 60)
    logger.info('Listenr Live Streaming Server')
    logger.info(f'Storage: {STORAGE_BASE}')
    logger.info(f'LLM: {"enabled" if USE_LLM else "disabled"}')
    logger.info(f'Starting on {HOST}:{PORT}')
    logger.info('=' * 60)

    STORAGE_BASE.mkdir(parents=True, exist_ok=True)
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
