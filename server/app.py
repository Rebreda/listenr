#!/usr/bin/env python3
"""
Listenr Web Server - Live Streaming Transcription

Real-time continuous audio transcription with WebSocket support.
Microphone stays open, transcripts appear live as you speak.

Features:
- Continuous microphone streaming (no start/stop!)
- Real-time transcription via WebSockets
- Automatic speech segmentation with VAD
- JSON responses with full metadata
- Mobile-optimized interface
- Optional LLM post-processing

Environment variables:
    LISTENR_STORAGE: Base directory for storage (default: ~/listenr_web)
    LISTENR_PORT: Port to run on (default: 5000)
    LISTENR_HOST: Host to bind to (default: 0.0.0.0)
    LISTENR_USE_LLM: Enable LLM post-processing (default: false)

Usage:
    pip install -r server/requirements.txt
    python server/app.py
"""

import os
import sys
import logging
import json
import base64
from pathlib import Path
from datetime import datetime, timezone

from flask import Flask, render_template_string, jsonify, request
from flask_sock import Sock
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from unified_asr import UnifiedASR

# ---------------------------
# Configuration
# ---------------------------
STORAGE_BASE = Path(os.environ.get('LISTENR_STORAGE', Path.home() / 'listenr_web'))
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
logger = logging.getLogger('listenr_stream')

# ---------------------------
# Flask App with WebSocket
# ---------------------------
app = Flask(__name__, static_folder='static', template_folder='templates')
sock = Sock(app)

# Global ASR instance (shared across connections)
asr_instance = None


def get_asr():
    """Get or create ASR instance"""
    global asr_instance
    if asr_instance is None:
        logger.info("Initializing ASR...")
        asr_instance = UnifiedASR(mode='stream', use_llm=USE_LLM, storage_base=str(STORAGE_BASE))
    return asr_instance


# ---------------------------
# WebSocket Handler
# ---------------------------
@sock.route('/transcribe')
def transcribe_websocket(ws):
    """
    WebSocket endpoint for live audio streaming.

    Protocol:
        Client sends: JSON messages with base64-encoded audio
        {
            "audio": "base64_encoded_audio_data",
            "sample_rate": 16000,
            "format": "float32"
        }

        Server sends: JSON transcription results
        {
            "type": "transcription",
            "data": { ... full JSON result ... }
        }

        Server also sends status messages:
        {
            "type": "status",
            "message": "connected|processing|error"
        }
    """
    logger.info(f"WebSocket connection established from {request.remote_addr}")

    try:
        asr = get_asr()

        # Send connection confirmation
        try:
            ws.send(json.dumps({
                'type': 'status',
                'message': 'connected',
                'llm_enabled': USE_LLM
            }))
        except Exception as e:
            logger.error(f"Failed to send connection confirmation: {e}")
            return

        # Audio processing state
        accumulated_audio = np.array([], dtype=np.float32)

        while True:
            # Receive audio data from client
            try:
                message = ws.receive(timeout=30)  # 30 second timeout
                if message is None:
                    logger.info("WebSocket client disconnected (None message)")
                    break
            except Exception as e:
                # Connection closed or timeout
                if "Connection closed" in str(e) or "ConnectionClosed" in str(type(e).__name__):
                    logger.info("WebSocket connection closed by client")
                else:
                    logger.warning(f"WebSocket receive error: {e}")
                break

            try:
                data = json.loads(message)

                if data.get('type') == 'audio':
                    # Decode audio
                    audio_b64 = data['audio']
                    audio_bytes = base64.b64decode(audio_b64)
                    audio_array = np.frombuffer(audio_bytes, dtype=np.float32)

                    # Accumulate audio
                    accumulated_audio = np.concatenate([accumulated_audio, audio_array])

                    # Process in VAD-sized chunks
                    while len(accumulated_audio) >= asr.vad_chunk_size:
                        vad_chunk = accumulated_audio[:asr.vad_chunk_size]
                        accumulated_audio = accumulated_audio[asr.vad_chunk_size:]

                        # Process chunk (this handles VAD and transcription)
                        result = asr.process_vad_chunk(vad_chunk)

                        if result:
                            # Send transcription result to client
                            try:
                                ws.send(json.dumps({
                                    'type': 'transcription',
                                    'data': result
                                }))
                            except Exception as e:
                                logger.error(f"Failed to send transcription: {e}")
                                break

                elif data.get('type') == 'ping':
                    # Keepalive
                    try:
                        ws.send(json.dumps({'type': 'pong'}))
                    except:
                        break

            except json.JSONDecodeError:
                logger.error("Invalid JSON received")
                try:
                    ws.send(json.dumps({
                        'type': 'error',
                        'message': 'Invalid JSON'
                    }))
                except:
                    break

            except Exception as e:
                logger.exception(f"Error processing audio chunk: {e}")
                try:
                    ws.send(json.dumps({
                        'type': 'error',
                        'message': str(e)
                    }))
                except:
                    break

    except Exception as e:
        logger.exception(f"WebSocket handler error: {e}")
    finally:
        logger.info("WebSocket connection closed")
        # Reset VAD state for this client
        try:
            asr.vad_iterator.reset_states()
        except:
            pass


# ---------------------------
# HTTP Routes
# ---------------------------
@app.route('/')
def index():
    """Serve the visual live streaming interface (default)"""
    with open(Path(__file__).parent / 'templates' / 'index_visual.html', 'r') as f:
        return render_template_string(f.read())


@app.route('/simple')
def simple():
    """Serve the simple list-based interface"""
    with open(Path(__file__).parent / 'templates' / 'index.html', 'r') as f:
        return render_template_string(f.read())


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'time': datetime.now(timezone.utc).isoformat(),
        'storage': str(STORAGE_BASE),
        'llm_enabled': USE_LLM,
        'mode': 'streaming'
    })


# ---------------------------
# Security Headers
# ---------------------------
@app.after_request
def set_security_headers(response):
    """Add security headers"""
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
    logger.info('=' * 60)
    logger.info(f'Storage: {STORAGE_BASE}')
    logger.info(f'LLM post-processing: {"enabled" if USE_LLM else "disabled"}')
    logger.info(f'Mode: Continuous streaming with WebSockets')
    logger.info(f'Starting server on {HOST}:{PORT}')
    logger.info('=' * 60)

    # Ensure storage exists
    STORAGE_BASE.mkdir(parents=True, exist_ok=True)

    # Run the app
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
