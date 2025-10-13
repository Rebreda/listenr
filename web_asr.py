#!/usr/bin/env python3
"""
Web API version of the ASR processor
Returns structured responses suitable for web applications
"""

import os
import json
from datetime import datetime
from typing import Dict, Any
from asr import WhisperASR
import soundfile as sf

class WebASRProcessor:
    def __init__(self):
        self.asr = WhisperASR()
        
    def process_audio(self, audio_data: Any, sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Process audio data and return a structured response
        Returns a dictionary containing:
        - transcription: The transcribed text
        - audio_path: Path to the stored audio file
        - duration: Audio duration in seconds
        - timestamp: Processing timestamp
        - confidence: Transcription confidence score
        """
        try:
            # Get current timestamp
            timestamp = datetime.now()
            date_str = timestamp.strftime("%Y-%m-%d")
            time_str = timestamp.strftime("%H%M%S")
            
            # Create storage directories if they don't exist
            storage_base = os.path.expanduser("~/listenr_web")
            audio_dir = os.path.join(storage_base, "audio", date_str)
            transcript_dir = os.path.join(storage_base, "transcripts", date_str)
            os.makedirs(audio_dir, exist_ok=True)
            os.makedirs(transcript_dir, exist_ok=True)
            
            # Generate filenames
            audio_filename = f"clip_{date_str}_{time_str}.wav"
            transcript_filename = f"transcript_{date_str}_{time_str}.json"
            audio_path = os.path.join(audio_dir, audio_filename)
            transcript_path = os.path.join(transcript_dir, transcript_filename)
            
            # Save audio file
            sf.write(audio_path, audio_data, sample_rate)
            
            # Get audio duration
            duration = len(audio_data) / sample_rate
            
            # Feed the audio data to ASR
            self.asr.speech_frames = [audio_data]
            
            # Process the audio
            text = self.asr.process_speech_segment()
            confidence = 1.0  # Default confidence
            
            # Create response structure
            response = {
                'success': True,
                'transcription': text,
                'audio': {
                    'path': audio_path,
                    'duration': duration,
                    'url': f'/audio/{date_str}/{audio_filename}',
                    'sample_rate': sample_rate
                },
                'timestamp': timestamp.isoformat(),
                'confidence': confidence,
                'metadata': {
                    'date': date_str,
                    'time': time_str,
                    'transcript_path': transcript_path
                }
            }
            
            # Save transcript with metadata
            with open(transcript_path, 'w') as f:
                json.dump(response, f, indent=2)
            
            return response
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }