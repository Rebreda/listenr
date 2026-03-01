import asyncio
import numpy as np
from flask import Flask, render_template_string, jsonify, request
from flask_sock import Sock

# ---------------------------
# Flask App with WebSocket
# ---------------------------


# ---------------------------
# Config API for frontend (must come after app is defined)
# ---------------------------
@app.route('/api/config')
def api_config():
    from config_manager import get_setting
    # Get WebSocket URL and model from config
    ws_url = get_setting('Lemonade', 'realtime_ws_url', None)
    if not ws_url:
        api_base = get_setting('LLM', 'api_base', None)
        if api_base:
            ws_url = api_base.replace('/api/v1', '/realtime')
        else:
            ws_url = 'ws://localhost:8000/realtime'
    model = get_setting('Whisper', 'model', 'Whisper-Tiny')
    return jsonify({
        'ws_url': ws_url,
        'model': model
    })


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



def get_asr():
    """Get or create LemonadeUnifiedASR instance"""
    # No stateful model needed, just return a new instance (stateless Lemonade client)
    return LemonadeUnifiedASR(use_llm=USE_LLM)


# ---------------------------
# WebSocket Handler
# ---------------------------

# --- Lemonade-native streaming: forward audio chunks to Lemonade /realtime WebSocket ---
@sock.route('/transcribe')
def transcribe_websocket(ws):
    """
    WebSocket endpoint for live audio streaming (Lemonade-native).
    Client sends: JSON with base64-encoded audio chunks.
    Server streams back Lemonade results as they arrive.
    """
    logger.info(f"WebSocket connection established from {request.remote_addr}")
    asr = get_asr()
    try:
        ws.send(json.dumps({
            'type': 'status',
            'message': 'connected',
            'llm_enabled': USE_LLM
        }))

        async def audio_stream():
            while True:
                message = ws.receive()
                if message is None:
                    break
                data = json.loads(message)
                if data.get('type') == 'audio':
                    audio_b64 = data['audio']
                    audio_bytes = base64.b64decode(audio_b64)
                    yield audio_bytes
                elif data.get('type') == 'ping':
                    ws.send(json.dumps({'type': 'pong'}))

        # Run the Lemonade WebSocket client in an event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            async def forward_results():
                async for result in asr.stream_transcribe(audio_stream()):
                    ws.send(json.dumps({
                        'type': 'transcription',
                        'data': result
                    }))
            loop.run_until_complete(forward_results())
        finally:
            loop.close()
    except Exception as e:
        logger.exception(f"WebSocket handler error: {e}")
    finally:
        logger.info("WebSocket connection closed")


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
