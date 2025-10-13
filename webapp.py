from flask import Flask, request, jsonify, send_from_directory
import os
import tempfile
import subprocess
from web_asr import WebASRProcessor
import numpy as np
import soundfile as sf
import io

# Initialize Flask with static files served from the same directory
app = Flask(__name__, static_folder='.')
asr = WebASRProcessor()  # Initialize the web ASR system

# Set up routes for serving audio files
@app.route('/audio/<date>/<filename>')
def serve_audio(date, filename):
    audio_dir = os.path.expanduser(f"~/listenr_web/audio/{date}")
    return send_from_directory(audio_dir, filename)

def convert_audio_to_wav(input_path, output_path):
    """Convert audio to WAV format using ffmpeg"""
    try:
        subprocess.run([
            'ffmpeg', '-i', input_path,
            '-ar', '16000',  # Set sample rate to 16kHz
            '-ac', '1',      # Convert to mono
            '-c:a', 'pcm_s16le',  # Use 16-bit PCM encoding
            output_path
        ], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        app.logger.error(f"FFmpeg conversion failed: {e.stderr.decode()}")
        return False

# HTML template for the web interface - kept in a string to avoid extra files
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Listenr Web</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            max-width: 800px; 
            margin: 0 auto; 
            padding: 20px;
            background: #f0f0f0;
        }
        .controls { 
            text-align: center; 
            margin: 20px 0;
        }
        button { 
            padding: 10px 20px;
            margin: 5px;
            font-size: 16px;
            cursor: pointer;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
        }
        button:disabled {
            background: #cccccc;
        }
        #output {
            background: white;
            padding: 20px;
            border-radius: 5px;
            min-height: 100px;
            white-space: pre-wrap;
        }
        .status {
            color: #666;
            font-style: italic;
            margin: 10px 0;
        }
        .transcript-entry {
            border-bottom: 1px solid #eee;
            padding: 10px 0;
            margin-bottom: 15px;
        }
        .timestamp {
            color: #888;
            font-size: 0.8em;
            margin-bottom: 5px;
        }
        .text {
            margin: 8px 0;
            font-size: 1.1em;
        }
        .audio-player {
            margin: 10px 0;
            background: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
        }
        .audio-player audio {
            width: 100%;
            margin-top: 5px;
        }
        #volumeMeter {
            width: 100%;
            height: 20px;
            background-color: #ddd;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        #volumeBar {
            width: 0%;
            height: 100%;
            background-color: #4CAF50;
            transition: width 0.1s;
        }
    </style>
</head>
<body>
    <h1>Listenr Web Interface</h1>
    <div class="controls">
        <button id="startBtn">Start Recording</button>
        <button id="stopBtn" disabled>Stop Recording</button>
    </div>
    <div class="status" id="status">Ready</div>
    <div id="output"></div>

    <script>
        let mediaRecorder;
        let audioContext;
        let analyser;
        let audioChunks = [];
        let isRecording = false;
        let silenceTimer = null;
        let lastAudioLevel = 0;
        
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const status = document.getElementById('status');
        const output = document.getElementById('output');
        
        // Create volume meter
        const volumeMeterDiv = document.createElement('div');
        volumeMeterDiv.id = 'volumeMeter';
        const volumeBar = document.createElement('div');
        volumeBar.id = 'volumeBar';
        volumeMeterDiv.appendChild(volumeBar);
        document.querySelector('.controls').appendChild(volumeMeterDiv);

        // Check if mediaDevices is supported
        const checkMediaSupport = async () => {
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                if (window.location.protocol !== 'https:' && window.location.hostname !== 'localhost') {
                    throw new Error('Media access requires HTTPS or localhost. Try accessing via localhost or enable HTTPS.');
                }
                throw new Error('Media devices not supported in this browser.');
            }
        };

        // Process audio levels
        const processAudioLevel = (analyser) => {
            const array = new Uint8Array(analyser.frequencyBinCount);
            analyser.getByteFrequencyData(array);
            const values = array.reduce((a, b) => a + b) / array.length;
            const level = Math.min(100, Math.round((values / 128) * 100));
            volumeBar.style.width = level + '%';
            
            // Auto-stop on silence (if level is very low for 2 seconds)
            if (isRecording) {
                if (level < 5) {
                    if (!silenceTimer) {
                        silenceTimer = setTimeout(() => {
                            if (lastAudioLevel < 5) {
                                stopRecording();
                            }
                            silenceTimer = null;
                        }, 2000);
                    }
                } else {
                    if (silenceTimer) {
                        clearTimeout(silenceTimer);
                        silenceTimer = null;
                    }
                }
                lastAudioLevel = level;
            }
            
            requestAnimationFrame(() => processAudioLevel(analyser));
        };

        const startRecording = async () => {
            try {
                await checkMediaSupport();
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        channelCount: 1,
                        sampleRate: 16000,
                        echoCancellation: true,
                        noiseSuppression: true
                    } 
                });

                // Set up audio context for volume monitoring
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                const source = audioContext.createMediaStreamSource(stream);
                analyser = audioContext.createAnalyser();
                analyser.smoothingTimeConstant = 0.3;
                analyser.fftSize = 1024;
                source.connect(analyser);
                processAudioLevel(analyser);

                mediaRecorder = new MediaRecorder(stream, {
                    mimeType: 'audio/webm;codecs=opus'
                });
                
                isRecording = true;
                
                mediaRecorder.ondataavailable = (event) => {
                    audioChunks.push(event.data);
                };

                mediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    const formData = new FormData();
                    formData.append('audio_data', audioBlob);

                    status.textContent = 'Processing audio...';
                    try {
                        console.log('Sending audio for processing...');
                        const response = await fetch('/upload', {
                            method: 'POST',
                            body: formData
                        });
                        
                        console.log('Received response:', response);
                        const data = await response.json();
                        console.log('Parsed data:', data);
                        
                        if (data.transcription) {
                            const entry = document.createElement('div');
                            entry.className = 'transcript-entry';
                            
                            // Format the duration nicely
                            const duration = data.duration ? 
                                `(${data.duration.toFixed(1)}s)` : '';
                            
                            // Create entry with audio player if available
                            entry.innerHTML = `
                                <div class="timestamp">${new Date().toLocaleTimeString()} ${duration}</div>
                                <div class="text">${data.transcription}</div>
                                ${data.audio_file ? `
                                    <div class="audio-player">
                                        <audio controls src="/audio/${data.audio_file}"></audio>
                                    </div>
                                ` : ''}
                            `;
                            
                            output.insertBefore(entry, output.firstChild);
                            console.log('Added entry to output');
                        } else {
                            console.warn('No transcription in response:', data);
                        }
                    } catch (err) {
                        console.error('Error processing audio:', err);
                        status.textContent = 'Error: ' + err.message;
                    }
                    
                    audioChunks = [];
                    
                    // Auto-restart recording if still active
                    if (isRecording) {
                        mediaRecorder.start();
                        status.textContent = 'Recording...';
                    } else {
                        status.textContent = 'Ready';
                    }
                };

                mediaRecorder.start();
                startBtn.disabled = true;
                stopBtn.disabled = false;
                status.textContent = 'Recording...';
            } catch (err) {
                status.textContent = 'Error: ' + err.message;
                isRecording = false;
            }
        };

        const stopRecording = () => {
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
            }
            if (audioContext) {
                audioContext.close();
            }
            startBtn.disabled = false;
            stopBtn.disabled = true;
            status.textContent = 'Stopped';
            isRecording = false;
            volumeBar.style.width = '0%';
        };

        startBtn.onclick = startRecording;
        stopBtn.onclick = stopRecording;
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return HTML_TEMPLATE

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'audio_data' not in request.files:
            return jsonify({'error': 'No audio data provided'}), 400
        
        audio_file = request.files['audio_data']
        
        # Create a temporary directory for audio processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the uploaded file
            input_path = os.path.join(temp_dir, 'input.webm')
            audio_file.save(input_path)
            
            try:
                # Convert webm to wav using ffmpeg
                wav_path = os.path.join(temp_dir, 'output.wav')
                if not convert_audio_to_wav(input_path, wav_path):
                    return jsonify({'error': 'Audio conversion failed'}), 500
                
                # Read the wav file using soundfile
                audio_data, sample_rate = sf.read(wav_path)
                
                # Process with WhisperASR and get timestamp
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Process with WebASRProcessor
                result = asr.process_audio(audio_data, sample_rate)
                
                if not result['success']:
                    raise Exception(result['error'])
                
                return jsonify({
                    'transcription': result['transcription'],
                    'timestamp': result['timestamp'],
                    'audio_file': result['audio']['path'],
                    'transcript_file': result['metadata']['transcript_path'],
                    'duration': result['audio']['duration']
                })
                
            except Exception as e:
                app.logger.error(f"Error processing audio: {str(e)}")
                return jsonify({'error': 'Failed to process audio'}), 500
                
    except Exception as e:
        app.logger.error(f"Server error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Starting Listenr Web Server...")
    print("To access from other devices, use one of these URLs:")
    print("1. Local testing (recommended): http://localhost:5000")
    print("2. Local network: http://<your-computer-ip>:5000")
    print("\nNote: For microphone access, you must either:")
    print("- Use localhost URL")
    print("- Set up HTTPS with SSL certificates")
    print("- Use a reverse proxy with HTTPS (like nginx)")
    
    # Run the app on all interfaces
    app.run(host='0.0.0.0', port=5000, debug=False)
