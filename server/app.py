#!/usr/bin/env python3
"""
Listenr Web Server - Real-time Audio Transcription

A lightweight Flask server for streaming microphone audio from browsers
to the Whisper ASR system for transcription.

Features:
- Real-time audio streaming from browser microphone
- WebM/Ogg audio format support with automatic conversion
- Secure file uploads with MIME type validation
- JSON API responses with comprehensive metadata
- Audio file serving and transcript retrieval
- Health check and status endpoints

Environment variables:
    LISTENR_STORAGE: Base directory for audio/transcript storage (default: ~/listenr_web)
    LISTENR_PORT: Port to run on (default: 5000)
    LISTENR_HOST: Host to bind to (default: 0.0.0.0)
    LISTENR_USE_LLM: Enable LLM post-processing (default: false)

Usage:
    python server/app.py
    # or
    export LISTENR_STORAGE=~/my_recordings
    export LISTENR_USE_LLM=true
    python server/app.py
"""

import os
import sys
import logging
import subprocess
import tempfile
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory, render_template_string, abort
from werkzeug.utils import secure_filename
import soundfile as sf

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from web_asr import WebASRProcessor

# ---------------------------
# Configuration
# ---------------------------
STORAGE_BASE = Path(os.environ.get('LISTENR_STORAGE', Path.home() / 'listenr_web'))
PORT = int(os.environ.get('LISTENR_PORT', '5000'))
HOST = os.environ.get('LISTENR_HOST', '0.0.0.0')
USE_LLM = os.environ.get('LISTENR_USE_LLM', 'false').lower() in ('true', '1', 'yes')

ALLOWED_MIME = {
    'audio/webm',
    'audio/ogg',
    'audio/wav',
    'audio/x-wav',
    'audio/mpeg',
    'audio/mp3',
    'audio/mp4',
    'audio/x-m4a'
}
MAX_CONTENT_LENGTH = 20 * 1024 * 1024  # 20 MB upload limit
FFMPEG_TIMEOUT = 30  # seconds

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger('listenr_web')

# ---------------------------
# Flask App
# ---------------------------
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Initialize ASR processor (lazy loading - models loaded on first request)
processor = None


def get_processor():
    """Lazy initialization of ASR processor"""
    global processor
    if processor is None:
        logger.info("Initializing ASR processor...")
        processor = WebASRProcessor(storage_base=str(STORAGE_BASE), use_llm=USE_LLM)
    return processor


# ---------------------------
# Security Headers
# ---------------------------
@app.after_request
def set_security_headers(response):
    """Add security headers to all responses"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['Referrer-Policy'] = 'no-referrer-when-downgrade'
    return response


# ---------------------------
# Helper Functions
# ---------------------------
def ffmpeg_convert_to_wav(input_path: str, output_path: str, target_rate: int = 16000) -> None:
    """
    Convert audio file to WAV format using ffmpeg.

    Args:
        input_path: Path to input audio file
        output_path: Path to output WAV file
        target_rate: Target sample rate (default: 16000)

    Raises:
        subprocess.CalledProcessError: If ffmpeg fails
        subprocess.TimeoutExpired: If ffmpeg times out
    """
    cmd = [
        'ffmpeg', '-y',
        '-hide_banner', '-loglevel', 'error',
        '-i', input_path,
        '-ac', '1',  # Mono
        '-ar', str(target_rate),  # Sample rate
        '-c:a', 'pcm_s16le',  # 16-bit PCM
        output_path
    ]
    subprocess.run(cmd, check=True, timeout=FFMPEG_TIMEOUT, capture_output=True)


# ---------------------------
# Routes
# ---------------------------
@app.route('/')
def index():
    """Serve the main web interface"""
    with open(Path(__file__).parent / 'templates' / 'index.html', 'r') as f:
        return render_template_string(f.read())


@app.route('/health')
def health():
    """Health check endpoint"""
    from datetime import datetime, timezone
    return jsonify({
        'status': 'ok',
        'time': datetime.now(timezone.utc).isoformat(),
        'storage': str(STORAGE_BASE),
        'llm_enabled': USE_LLM
    })


@app.route('/audio/<date>/<filename>')
def serve_audio(date: str, filename: str):
    """
    Serve audio files from storage.

    Args:
        date: Date string (YYYY-MM-DD)
        filename: Audio filename

    Returns:
        Audio file or 404 if not found
    """
    date_clean = secure_filename(date)
    filename_clean = secure_filename(filename)

    proc = get_processor()
    audio_path = proc.get_audio_path(date_clean, filename_clean)

    if audio_path is None:
        abort(404)

    return send_from_directory(str(audio_path.parent), audio_path.name)


@app.route('/transcript/<date>/<uuid_str>')
def get_transcript(date: str, uuid_str: str):
    """
    Retrieve a stored transcript by date and UUID.

    Args:
        date: Date string (YYYY-MM-DD)
        uuid_str: UUID string

    Returns:
        JSON transcript or 404 if not found
    """
    date_clean = secure_filename(date)
    uuid_clean = secure_filename(uuid_str)

    proc = get_processor()
    transcript = proc.get_transcript(date_clean, uuid_clean)

    if transcript is None:
        return jsonify({'error': 'Transcript not found'}), 404

    return jsonify(transcript)


@app.route('/upload', methods=['POST'])
def upload():
    """
    Upload and process audio data.

    Accepts multipart/form-data with an 'audio_data' file field.

    Returns:
        JSON response with transcription and metadata
    """
    # Check for audio data
    if 'audio_data' not in request.files:
        return jsonify({'error': 'No audio_data field in request'}), 400

    audio_file = request.files['audio_data']
    if audio_file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # Validate MIME type (warn but don't reject)
    content_type = audio_file.mimetype
    if content_type not in ALLOWED_MIME:
        logger.warning(f'Unusual MIME type uploaded: {content_type}')

    # Process the audio file
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Save uploaded file
        input_filename = secure_filename(audio_file.filename) or 'upload.webm'
        input_path = tmpdir_path / input_filename
        audio_file.save(str(input_path))

        # Convert to WAV using ffmpeg
        wav_path = tmpdir_path / 'converted.wav'
        try:
            ffmpeg_convert_to_wav(str(input_path), str(wav_path))
        except subprocess.CalledProcessError as e:
            logger.error(f'ffmpeg conversion failed: {e.stderr.decode() if e.stderr else str(e)}')
            return jsonify({
                'error': 'Audio conversion failed',
                'detail': 'Could not convert audio to WAV format'
            }), 500
        except subprocess.TimeoutExpired:
            logger.error('ffmpeg conversion timed out')
            return jsonify({'error': 'Audio conversion timed out'}), 500
        except FileNotFoundError:
            logger.error('ffmpeg not found')
            return jsonify({
                'error': 'Server configuration error',
                'detail': 'ffmpeg not installed'
            }), 500

        # Read WAV file
        try:
            audio_array, sample_rate = sf.read(str(wav_path), always_2d=False)

            # Ensure mono
            if audio_array.ndim == 2:
                audio_array = audio_array.mean(axis=1)

        except Exception as e:
            logger.exception('Failed to read converted WAV file')
            return jsonify({
                'error': 'Failed to read audio data',
                'detail': str(e)
            }), 500

        # Process with ASR
        try:
            proc = get_processor()
            result = proc.process_audio(audio_array, sample_rate)

            if not result.get('success'):
                return jsonify({
                    'error': 'Transcription failed',
                    'detail': result.get('error')
                }), 500

            # Build response
            response = {
                'success': True,
                'transcription': result.get('transcription', ''),
                'corrected_text': result.get('corrected_text'),
                'timestamp': result.get('timestamp'),
                'audio_url': result['audio']['url'],
                'duration': result['audio']['duration'],
                'metadata': result.get('metadata', {})
            }

            return jsonify(response)

        except Exception as e:
            logger.exception('ASR processing failed')
            return jsonify({
                'error': 'Failed to process audio',
                'detail': str(e)
            }), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large errors"""
    return jsonify({
        'error': 'File too large',
        'detail': f'Maximum upload size is {MAX_CONTENT_LENGTH / 1024 / 1024:.0f} MB'
    }), 413


# ---------------------------
# Main
# ---------------------------
if __name__ == '__main__':
    logger.info('=' * 60)
    logger.info('Listenr Web Server')
    logger.info('=' * 60)
    logger.info(f'Storage: {STORAGE_BASE}')
    logger.info(f'LLM post-processing: {"enabled" if USE_LLM else "disabled"}')
    logger.info(f'Max upload size: {MAX_CONTENT_LENGTH / 1024 / 1024:.0f} MB')
    logger.info(f'Starting server on {HOST}:{PORT}')
    logger.info('=' * 60)

    # Ensure storage directories exist
    STORAGE_BASE.mkdir(parents=True, exist_ok=True)

    # Run the app
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
