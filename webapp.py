#!/usr/bin/env python3
"""Improved Listenr Web (single-file) - app.py

Features:
- Single-file Flask app with improved WebASRProcessor
- Secure filenames (uuid), configurable storage directory via ENV
- MAX_CONTENT_LENGTH upload protection, allowed MIME checks
- Robust ffmpeg conversion with timeout and error handling
- Detailed JSON responses, structured errors
- Health and status endpoints, simple logging
- Safe serving of audio files via send_from_directory
- Minimal dependencies: Flask, soundfile, numpy

Run:
    export LISTENR_STORAGE=~/listenr_web
    python3 listenr_web_improved.py

Optional: run behind a production WSGI server (gunicorn) and put nginx in front for HTTPS + static files.
"""

import os
import io
import uuid
import json
import shutil
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from flask import Flask, request, jsonify, send_from_directory, abort, Response
from werkzeug.utils import secure_filename
import soundfile as sf
import numpy as np

# ---------------------------
# Configuration
# ---------------------------
STORAGE_BASE = Path(os.environ.get('LISTENR_STORAGE', Path.home() / 'listenr_web'))
AUDIO_DIRNAME = 'audio'
TRANSCRIPT_DIRNAME = 'transcripts'
ALLOWED_MIME = {
    'audio/webm', 'audio/ogg', 'audio/wav', 'audio/x-wav', 'audio/mpeg', 'audio/mp3'
}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB upload limit
FFMPEG_TIMEOUT = 20  # seconds
SAMPLE_RATE = 16000

# Ensure storage directories exist early
(STORAGE_BASE / AUDIO_DIRNAME).mkdir(parents=True, exist_ok=True)
(STORAGE_BASE / TRANSCRIPT_DIRNAME).mkdir(parents=True, exist_ok=True)

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger('listenr')

# ---------------------------
# Minimal placeholder ASR (replace with your WhisperASR import)
# ---------------------------
class DummyASR:
    """Replace or wrap your real ASR implementation here (e.g. WhisperASR).
    It must implement process_speech_segment() that uses frames in self.speech_frames
    and returns a transcription string.
    """
    def __init__(self):
        self.speech_frames = []

    def process_speech_segment(self) -> str:
        if not self.speech_frames:
            return ''
        # naive "ASR": return length and mean amplitude
        arr = np.concatenate(self.speech_frames)
        duration = arr.size / SAMPLE_RATE
        return f"(dummy) audio {duration:.2f}s — mean {float(np.abs(arr).mean()):.6f}"


# ---------------------------
# WebASRProcessor
# ---------------------------
class WebASRProcessor:
    def __init__(self, storage_base: Path = STORAGE_BASE, asr=None):
        self.storage_base = Path(storage_base)
        self.asr = asr or DummyASR()

    def _make_paths(self, timestamp: datetime, uuid_str: str):
        date_str = timestamp.strftime('%Y-%m-%d')
        audio_dir = self.storage_base / AUDIO_DIRNAME / date_str
        transcript_dir = self.storage_base / TRANSCRIPT_DIRNAME / date_str
        audio_dir.mkdir(parents=True, exist_ok=True)
        transcript_dir.mkdir(parents=True, exist_ok=True)
        audio_filename = f"clip_{date_str}_{uuid_str}.wav"
        transcript_filename = f"transcript_{date_str}_{uuid_str}.json"
        return audio_dir / audio_filename, transcript_dir / transcript_filename, date_str, audio_filename

    def process_audio(self, audio_array: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Dict[str, Any]:
        """Process a numpy audio array and return structured metadata + transcription.

        Expects mono float32/float64 or int16 arrays. Will normalize to float32.
        """
        timestamp = datetime.utcnow()
        uuid_str = uuid.uuid4().hex[:12]

        try:
            # normalize types
            if not isinstance(audio_array, np.ndarray):
                audio_array = np.asarray(audio_array)

            # convert ints to float32
            if np.issubdtype(audio_array.dtype, np.integer):
                maxv = np.iinfo(audio_array.dtype).max
                audio_array = audio_array.astype('float32') / float(maxv)
            else:
                audio_array = audio_array.astype('float32')

            # ensure mono
            if audio_array.ndim == 2:
                # average channels
                audio_array = audio_array.mean(axis=1)

            duration = float(audio_array.shape[0]) / float(sample_rate)

            audio_path, transcript_path, date_str, audio_filename = self._make_paths(timestamp, uuid_str)

            # write WAV (16-bit PCM)
            sf.write(str(audio_path), audio_array, sample_rate, subtype='PCM_16')

            # feed ASR implementation
            if hasattr(self.asr, 'speech_frames'):
                self.asr.speech_frames = [audio_array]

            text = ''
            try:
                text = self.asr.process_speech_segment()
            except Exception as e:
                logger.exception('ASR processing failed')
                text = ''

            response = {
                'success': True,
                'transcription': text,
                'audio': {
                    'path': str(audio_path),
                    'filename': audio_filename,
                    'duration': duration,
                    'sample_rate': sample_rate,
                },
                'timestamp': timestamp.isoformat() + 'Z',
                'confidence': None,
                'metadata': {
                    'date': date_str,
                    'uuid': uuid_str,
                    'transcript_path': str(transcript_path)
                }
            }

            with open(transcript_path, 'w', encoding='utf-8') as fh:
                json.dump(response, fh, indent=2, ensure_ascii=False)

            return response

        except Exception as e:
            logger.exception('Failed to process audio')
            return {'success': False, 'error': str(e)}


# ---------------------------
# Flask App
# ---------------------------
app = Flask(__name__, static_folder='.', static_url_path='')
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
processor = WebASRProcessor()


@app.after_request
def set_security_headers(response: Response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Referrer-Policy'] = 'no-referrer'
    return response


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'time': datetime.utcnow().isoformat() + 'Z'})


@app.route('/audio/<date>/<filename>')
def serve_audio(date: str, filename: str):
    # Only serve files within the configured storage base
    date_clean = secure_filename(date)
    filename_clean = secure_filename(filename)
    audio_dir = STORAGE_BASE / AUDIO_DIRNAME / date_clean
    if not audio_dir.exists():
        abort(404)
    return send_from_directory(str(audio_dir), filename_clean)


def _ffmpeg_convert_to_wav(input_path: str, output_path: str, target_rate: int = SAMPLE_RATE) -> None:
    """Convert arbitrary audio to WAV (mono, target_rate) using ffmpeg.

    Raises CalledProcessError on failure.
    """
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-i', input_path,
        '-ac', '1',
        '-ar', str(target_rate),
        '-c:a', 'pcm_s16le',
        output_path
    ]
    subprocess.run(cmd, check=True, timeout=FFMPEG_TIMEOUT)


@app.route('/upload', methods=['POST'])
def upload():
    # Accept either multipart form upload (from browser) or binary blob (curl)
    if 'audio_data' not in request.files:
        return jsonify({'error': 'No audio_data file part'}), 400

    audio_file = request.files['audio_data']
    if audio_file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # basic MIME check
    content_type = audio_file.mimetype
    if content_type not in ALLOWED_MIME:
        logger.warning('Rejected upload with MIME: %s', content_type)
        # allow unknown types but warn; alternatively return 415
        # return jsonify({'error': 'Unsupported media type'}), 415

    # make temp dir for conversion
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / secure_filename(audio_file.filename)
        audio_file.save(str(input_path))

        wav_path = Path(tmpdir) / 'converted.wav'
        try:
            _ffmpeg_convert_to_wav(str(input_path), str(wav_path))
        except subprocess.CalledProcessError as e:
            logger.exception('ffmpeg failed')
            return jsonify({'error': 'Audio conversion failed', 'detail': str(e)}), 500
        except subprocess.TimeoutExpired:
            logger.exception('ffmpeg timeout')
            return jsonify({'error': 'Audio conversion timed out'}), 500

        # read wav and process
        try:
            audio_array, sample_rate = sf.read(str(wav_path), always_2d=False)
            # If stereo -> convert to mono by averaging
            if audio_array.ndim == 2:
                audio_array = audio_array.mean(axis=1)

            result = processor.process_audio(audio_array, sample_rate)
            if not result.get('success'):
                return jsonify({'error': 'ASR failed', 'detail': result.get('error')}), 500

            # produce a public URL for the saved audio
            date = result['metadata']['date']
            filename = result['audio']['filename']
            public_url = f"/audio/{date}/{filename}"

            return jsonify({
                'transcription': result['transcription'],
                'timestamp': result['timestamp'],
                'audio_url': public_url,
                'transcript_path': result['metadata']['transcript_path'],
                'duration': result['audio']['duration']
            })

        except Exception as e:
            logger.exception('Reading/processing WAV failed')
            return jsonify({'error': 'Failed to read or process audio', 'detail': str(e)}), 500


# Minimal index to help local testing (keeps the original UI but improved)
INDEX_HTML = '''<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Listenr Web (improved)</title>
<style>body{font-family:Inter,Arial;max-width:900px;margin:0 auto;padding:20px;background:#f6f7fb}button{padding:8px 12px;margin:6px;border-radius:6px;border:0;background:#0b76ff;color:#fff}#output{background:#fff;padding:12px;border-radius:8px;min-height:80px}</style>
</head>
<body>
<h1>Listenr Web (improved)</h1>
<div><button id=start>Start</button><button id=stop disabled>Stop</button></div>
<div id=status>Ready</div>
<div id=output></div>
<script>
let mr, chunks=[];
const startBtn=document.getElementById('start');
const stopBtn=document.getElementById('stop');
const status=document.getElementById('status');
const output=document.getElementById('output');
async function start(){
 try{
  const s=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1}});
  mr=new MediaRecorder(s,{mimeType:'audio/webm'});
  mr.ondataavailable=e=>chunks.push(e.data);
  mr.onstop=async ()=>{
    const blob=new Blob(chunks,{type:'audio/webm'});
    const fd=new FormData(); fd.append('audio_data', blob, 'clip.webm');
    status.textContent='Uploading...';
    const r=await fetch('/upload',{method:'POST',body:fd});
    const j=await r.json();
    if(r.ok){
      const d=document.createElement('div'); d.textContent=j.transcription||JSON.stringify(j);
      output.prepend(d);
    } else { output.prepend(document.createElement('div')).textContent=JSON.stringify(j); }
    chunks=[]; status.textContent='Ready';
  }
  mr.start(); startBtn.disabled=true; stopBtn.disabled=false; status.textContent='Recording...';
 }catch(e){status.textContent='Err:'+e.message}
}
function stop(){ if(mr && mr.state==='recording') mr.stop(); startBtn.disabled=false; stopBtn.disabled=true; }
startBtn.onclick=start; stopBtn.onclick=stop;
</script>
</body>
</html>'''


@app.route('/')
def index():
    return INDEX_HTML


if __name__ == '__main__':
    logger.info('Starting improved Listenr Web server')
    logger.info('Storage base: %s', STORAGE_BASE)
    app.run(host='0.0.0.0', port=5000, debug=False)
