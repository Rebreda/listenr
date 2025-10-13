#!/usr/bin/env python3
"""
Web API version of the ASR processor
Returns structured JSON responses suitable for web applications

This module provides a clean interface between the Flask webapp and the core ASR system.
It handles audio processing, file storage, and JSON response formatting.
"""

import os
import json
import uuid
import tempfile
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import soundfile as sf

logger = logging.getLogger('web_asr')


class WebASRProcessor:
    """
    Web-friendly wrapper around the WhisperASR system.

    Features:
    - Processes numpy audio arrays from web uploads
    - Returns structured JSON responses
    - Stores audio clips and transcripts with metadata
    - UUID-based filenames for security
    - Optional LLM post-processing support
    """

    def __init__(self,
                 storage_base: Optional[str] = None,
                 use_llm: bool = False):
        """
        Initialize the web ASR processor.

        Args:
            storage_base: Base directory for storing audio/transcripts (defaults to ~/listenr_web)
            use_llm: Enable LLM post-processing for improved accuracy
        """
        # Lazy import to avoid loading models until needed
        from asr import WhisperASR

        self.asr = WhisperASR(use_llm=use_llm)
        self.use_llm = use_llm

        # Setup storage
        if storage_base:
            self.storage_base = Path(storage_base).expanduser()
        else:
            self.storage_base = Path.home() / "listenr_web"

        self.audio_dir = self.storage_base / "audio"
        self.transcript_dir = self.storage_base / "transcripts"

        # Create directories
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.transcript_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"WebASRProcessor initialized (storage: {self.storage_base}, LLM: {use_llm})")

    def process_audio(self,
                     audio_data: np.ndarray,
                     sample_rate: int = 16000,
                     metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process audio data and return a structured JSON response.

        Args:
            audio_data: Numpy array containing audio samples (mono, float32)
            sample_rate: Sample rate of the audio
            metadata: Optional additional metadata to include in response

        Returns:
            Dictionary containing:
            - success: Boolean indicating success/failure
            - transcription: The transcribed text (raw)
            - corrected_text: LLM-corrected text (if LLM enabled)
            - audio: Audio file information
            - timestamp: ISO format timestamp
            - confidence: Transcription confidence (placeholder)
            - metadata: Additional metadata including paths and UUID
        """
        try:
            # Generate unique identifier and timestamp
            timestamp = datetime.utcnow()
            date_str = timestamp.strftime("%Y-%m-%d")
            uuid_str = uuid.uuid4().hex[:12]

            # Create date-organized directories
            audio_date_dir = self.audio_dir / date_str
            transcript_date_dir = self.transcript_dir / date_str
            audio_date_dir.mkdir(parents=True, exist_ok=True)
            transcript_date_dir.mkdir(parents=True, exist_ok=True)

            # Generate filenames
            audio_filename = f"clip_{date_str}_{uuid_str}.wav"
            transcript_filename = f"transcript_{date_str}_{uuid_str}.json"
            audio_path = audio_date_dir / audio_filename
            transcript_path = transcript_date_dir / transcript_filename

            # Ensure audio is float32 and mono
            if not isinstance(audio_data, np.ndarray):
                audio_data = np.asarray(audio_data, dtype=np.float32)

            if audio_data.dtype != np.float32:
                if np.issubdtype(audio_data.dtype, np.integer):
                    audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
                else:
                    audio_data = audio_data.astype(np.float32)

            if audio_data.ndim == 2:
                audio_data = audio_data.mean(axis=1)

            # Calculate duration
            duration = float(len(audio_data)) / float(sample_rate)

            # Save audio file
            sf.write(str(audio_path), audio_data, sample_rate, subtype='PCM_16')
            logger.info(f"Saved audio clip: {audio_path} ({duration:.2f}s)")

            # Process with ASR
            # Feed the audio data directly to the ASR system
            self.asr.speech_frames = [audio_data]

            # Get transcription using the core ASR method
            raw_text = ''
            corrected_text = None

            try:
                # Save to temp file for Whisper processing
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    sf.write(tmp_file.name, audio_data, sample_rate)
                    tmp_path = tmp_file.name

                try:
                    # Use Whisper to transcribe
                    segments, info = self.asr.whisper_model.transcribe(
                        tmp_path,
                        beam_size=5,
                        best_of=5,
                        temperature=0.0,
                        condition_on_previous_text=False,
                        vad_filter=False,
                        word_timestamps=False
                    )

                    raw_text = ' '.join(segment.text.strip() for segment in segments)

                    # Process with LLM if enabled and available
                    if self.use_llm and self.asr.llm_processor and self.asr.llm_processor.available:
                        corrected_text = self.asr.llm_processor.process(raw_text, use_context=False)
                        logger.info(f"LLM correction applied")

                finally:
                    os.unlink(tmp_path)

            except Exception as e:
                logger.error(f"ASR processing error: {e}")
                raw_text = ''
                corrected_text = None

            # Build response
            response = {
                'success': True,
                'transcription': raw_text,
                'corrected_text': corrected_text or raw_text,
                'audio': {
                    'path': str(audio_path),
                    'filename': audio_filename,
                    'url': f'/audio/{date_str}/{audio_filename}',
                    'duration': duration,
                    'sample_rate': sample_rate
                },
                'timestamp': timestamp.isoformat() + 'Z',
                'confidence': None,  # Placeholder for future confidence scoring
                'metadata': {
                    'date': date_str,
                    'uuid': uuid_str,
                    'transcript_path': str(transcript_path),
                    'llm_enabled': self.use_llm,
                    'llm_applied': corrected_text is not None and corrected_text != raw_text
                }
            }

            # Add custom metadata if provided
            if metadata:
                response['metadata'].update(metadata)

            # Save transcript JSON
            with open(transcript_path, 'w', encoding='utf-8') as f:
                json.dump(response, f, indent=2, ensure_ascii=False)

            logger.info(f"Transcription: {raw_text[:100]}")
            return response

        except Exception as e:
            logger.exception(f"Failed to process audio: {e}")
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }

    def get_audio_path(self, date: str, filename: str) -> Optional[Path]:
        """
        Get the full path to an audio file.

        Args:
            date: Date string (YYYY-MM-DD format)
            filename: Audio filename

        Returns:
            Path object if file exists, None otherwise
        """
        audio_path = self.audio_dir / date / filename
        return audio_path if audio_path.exists() else None

    def get_transcript(self, date: str, uuid_str: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a stored transcript by date and UUID.

        Args:
            date: Date string (YYYY-MM-DD format)
            uuid_str: UUID string

        Returns:
            Transcript dictionary if found, None otherwise
        """
        transcript_filename = f"transcript_{date}_{uuid_str}.json"
        transcript_path = self.transcript_dir / date / transcript_filename

        if transcript_path.exists():
            try:
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load transcript {transcript_path}: {e}")

        return None