#!/usr/bin/env python3
"""
Unified ASR System - Works for both CLI and Web

This module provides a single ASR implementation that:
- Returns JSON responses for all transcription results
- Works seamlessly for both command-line and web usage
- Supports both batch (file) and streaming (real-time) modes
- Includes optional LLM post-processing
- Maintains VAD for speech detection
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
import sounddevice as sd
import soundfile as sf
import torch

# Import config manager
import config_manager as cfg

# Import LLM processor if available
try:
    from llm_processor import LLMProcessor
    llm_available = True
except ImportError:
    llm_available = False

logger = logging.getLogger('unified_asr')


class UnifiedASR:
    """
    Unified ASR system that works for CLI, Web, and Streaming modes.

    All methods return JSON-serializable dictionaries with consistent structure.
    """

    def __init__(self,
                 mode: str = 'cli',
                 use_llm: bool = False,
                 storage_base: Optional[str] = None):
        """
        Initialize the unified ASR system.

        Args:
            mode: Operation mode ('cli', 'web', or 'stream')
            use_llm: Enable LLM post-processing
            storage_base: Base directory for storage (defaults based on mode)
        """
        self.mode = mode

        # Setup logging
        log_level = getattr(logging, cfg.get_setting('Logging', 'level', 'INFO'), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        )

        # Initialize LLM processor if requested
        self.use_llm = use_llm and llm_available and cfg.get_bool_setting('LLM', 'enabled', True)
        self.llm_processor = None

        if self.use_llm:
            try:
                llm_model = cfg.get_setting('LLM', 'model', 'gemma2:2b')
                llm_host = cfg.get_setting('LLM', 'ollama_host', 'http://localhost:11434')
                llm_temp = cfg.get_float_setting('LLM', 'temperature', 0.1)
                llm_context = cfg.get_int_setting('LLM', 'context_window', 3)
                llm_max_tokens = cfg.get_int_setting('LLM', 'max_tokens', 100)
                llm_timeout = cfg.get_int_setting('LLM', 'timeout', 10)

                self.llm_processor = LLMProcessor(
                    model=llm_model,
                    ollama_host=llm_host,
                    context_window=llm_context,
                    temperature=llm_temp,
                    max_tokens=llm_max_tokens,
                    timeout=llm_timeout
                )

                if self.llm_processor.available:
                    logger.info(f"LLM post-processing enabled with {llm_model}")
                else:
                    logger.warning("LLM processor not available")
                    self.use_llm = False
            except Exception as e:
                logger.error(f"Failed to initialize LLM: {e}")
                self.use_llm = False

        # Setup storage
        if storage_base:
            self.storage_base = Path(storage_base).expanduser()
        elif mode == 'web':
            self.storage_base = Path.home() / 'listenr_web'
        else:
            self.storage_base = Path.home() / '.listenr'

        self.audio_dir = self.storage_base / 'audio'
        self.transcript_dir = self.storage_base / 'transcripts'
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.transcript_dir.mkdir(parents=True, exist_ok=True)

        # Audio settings
        self.sample_rate = cfg.get_int_setting('Audio', 'sample_rate', 16000)
        self.channels = cfg.get_int_setting('Audio', 'channels', 1)
        self.blocksize = cfg.get_int_setting('Audio', 'blocksize', 1024)
        self.vad_chunk_size = cfg.get_int_setting('VAD', 'vad_chunk_size', 512)

        # VAD settings
        self.speech_threshold = cfg.get_float_setting('VAD', 'speech_threshold', 0.5)
        self.min_speech_duration_s = cfg.get_float_setting('VAD', 'min_speech_duration_s', 0.3)
        self.max_silence_duration_s = cfg.get_float_setting('VAD', 'max_silence_duration_s', 0.8)
        self.leading_silence_s = cfg.get_float_setting('Audio', 'leading_silence_s', 0.3)
        self.trailing_silence_s = cfg.get_float_setting('Audio', 'trailing_silence_s', 0.3)

        # Load models
        self.load_whisper_model()
        self.load_vad_model()

        # Streaming state
        self.running = False
        self.audio_queue = queue.Queue()
        self.is_speech = False
        self.speech_frames = []
        self.silence_chunks = 0
        self.audio_buffer = []
        self.max_buffer_chunks = int(self.leading_silence_s * self.sample_rate / self.vad_chunk_size)
        self.stream_callback = None

        logger.info(f"UnifiedASR initialized (mode={mode}, storage={self.storage_base}, LLM={self.use_llm})")

    def load_whisper_model(self):
        """Load Whisper model (faster-whisper)"""
        model_size = cfg.get_setting('Whisper', 'model_size', 'base')
        device = cfg.get_setting('Whisper', 'device', 'cpu')
        compute_type = cfg.get_setting('Whisper', 'compute_type', 'float16')

        if device in ('cuda', 'gpu') and compute_type == 'int8':
            compute_type = 'float16'
        elif device == 'cpu' and compute_type not in ('int8', 'float32'):
            compute_type = 'int8'

        logger.info(f"Loading Whisper model: {model_size} on {device} (compute_type={compute_type})")
        try:
            from faster_whisper import WhisperModel
            self.whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
            logger.info("Whisper model loaded successfully")
        except ImportError:
            logger.error("faster-whisper not installed. Install with: pip install faster-whisper")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            sys.exit(1)

    def load_vad_model(self):
        """Load Silero VAD model"""
        logger.info("Loading Silero VAD model")
        try:
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=True
            )
            self.vad_model = model
            (_, _, _, VADIterator, _) = utils
            self.vad_iterator = VADIterator(model, threshold=self.speech_threshold)
            logger.info("VAD model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load VAD model: {e}")
            sys.exit(1)

    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Transcribe audio data using Whisper.

        Args:
            audio_data: Numpy array (mono, float32)
            sample_rate: Sample rate

        Returns:
            Dictionary with transcription results
        """
        # Save to temp file for Whisper
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sf.write(tmp_file.name, audio_data, sample_rate)
            tmp_path = tmp_file.name

        try:
            # Transcribe
            beam_size = cfg.get_int_setting('Whisper', 'beam_size', 5)
            best_of = cfg.get_int_setting('Whisper', 'best_of', 5)
            temperature = cfg.get_float_setting('Whisper', 'temperature', 0.0)

            segments, info = self.whisper_model.transcribe(
                tmp_path,
                beam_size=beam_size,
                best_of=best_of,
                temperature=temperature,
                condition_on_previous_text=False,
                vad_filter=False,
                word_timestamps=False
            )

            raw_text = ' '.join(segment.text.strip() for segment in segments)

            # Apply LLM if enabled
            corrected_text = None
            if self.use_llm and self.llm_processor and self.llm_processor.available and raw_text:
                try:
                    corrected_text = self.llm_processor.process(raw_text, use_context=False)
                except Exception as e:
                    logger.error(f"LLM processing failed: {e}")
                    corrected_text = None

            return {
                'success': True,
                'raw_text': raw_text,
                'corrected_text': corrected_text or raw_text,
                'language': info.language if hasattr(info, 'language') else None,
                'language_probability': float(info.language_probability) if hasattr(info, 'language_probability') else None
            }

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'raw_text': '',
                'corrected_text': ''
            }
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass

    def process_audio(self,
                     audio_data: np.ndarray,
                     sample_rate: int = 16000,
                     metadata: Optional[Dict[str, Any]] = None,
                     save: bool = True) -> Dict[str, Any]:
        """
        Process audio data and return JSON result.

        Args:
            audio_data: Numpy array containing audio samples
            sample_rate: Sample rate of the audio
            metadata: Optional additional metadata
            save: Whether to save audio and transcript files

        Returns:
            JSON-serializable dictionary with all results and metadata
        """
        timestamp = datetime.now(timezone.utc)
        date_str = timestamp.strftime("%Y-%m-%d")
        uuid_str = uuid.uuid4().hex[:12]

        try:
            # Normalize audio
            if not isinstance(audio_data, np.ndarray):
                audio_data = np.asarray(audio_data, dtype=np.float32)

            if audio_data.dtype != np.float32:
                if np.issubdtype(audio_data.dtype, np.integer):
                    audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
                else:
                    audio_data = audio_data.astype(np.float32)

            if audio_data.ndim == 2:
                audio_data = audio_data.mean(axis=1)

            duration = float(len(audio_data)) / float(sample_rate)

            # Transcribe
            transcription_result = self.transcribe_audio(audio_data, sample_rate)

            # Prepare paths
            audio_path = None
            transcript_path = None
            audio_url = None

            if save:
                audio_date_dir = self.audio_dir / date_str
                transcript_date_dir = self.transcript_dir / date_str
                audio_date_dir.mkdir(parents=True, exist_ok=True)
                transcript_date_dir.mkdir(parents=True, exist_ok=True)

                audio_filename = f"clip_{date_str}_{uuid_str}.wav"
                transcript_filename = f"transcript_{date_str}_{uuid_str}.json"
                audio_path = audio_date_dir / audio_filename
                transcript_path = transcript_date_dir / transcript_filename
                audio_url = f"/audio/{date_str}/{audio_filename}"

                # Save audio file
                sf.write(str(audio_path), audio_data, sample_rate, subtype='PCM_16')
                logger.info(f"Saved audio: {audio_path} ({duration:.2f}s)")

            # Build response
            response = {
                'success': transcription_result['success'],
                'transcription': transcription_result['raw_text'],
                'corrected_text': transcription_result['corrected_text'],
                'timestamp': timestamp.isoformat(),
                'audio': {
                    'path': str(audio_path) if audio_path else None,
                    'url': audio_url,
                    'duration': duration,
                    'sample_rate': sample_rate
                },
                'metadata': {
                    'date': date_str,
                    'uuid': uuid_str,
                    'transcript_path': str(transcript_path) if transcript_path else None,
                    'llm_enabled': self.use_llm,
                    'llm_applied': self.use_llm and transcription_result['corrected_text'] != transcription_result['raw_text'],
                    'language': transcription_result.get('language'),
                    'language_probability': transcription_result.get('language_probability'),
                    'mode': self.mode
                }
            }

            # Add custom metadata
            if metadata:
                response['metadata'].update(metadata)

            # Save transcript
            if save and transcript_path:
                with open(transcript_path, 'w', encoding='utf-8') as f:
                    json.dump(response, f, indent=2, ensure_ascii=False)

            logger.info(f"Transcription: {response['transcription'][:100]}")
            return response

        except Exception as e:
            logger.exception(f"Failed to process audio: {e}")
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

    def process_vad_chunk(self, audio_chunk: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Process a single VAD-sized audio chunk for streaming mode.

        Returns:
            JSON result if speech segment completed, None otherwise
        """
        # Keep buffer for leading silence
        self.audio_buffer.append(audio_chunk)
        if len(self.audio_buffer) > self.max_buffer_chunks:
            self.audio_buffer.pop(0)

        try:
            audio_tensor = torch.from_numpy(audio_chunk).float()
            speech_dict = self.vad_iterator(audio_tensor, return_seconds=False)

            if speech_dict:
                if 'start' in speech_dict:
                    if not self.is_speech:
                        logger.debug("Speech started")
                        self.is_speech = True
                        self.speech_frames = []

                        # Add leading silence
                        if self.leading_silence_s > 0 and len(self.audio_buffer) > 1:
                            for buffered_chunk in self.audio_buffer[:-1]:
                                self.speech_frames.append(buffered_chunk)

                    self.speech_frames.append(audio_chunk)
                    self.silence_chunks = 0

                elif 'end' in speech_dict:
                    if self.is_speech and self.speech_frames:
                        self.speech_frames.append(audio_chunk)
            else:
                if self.is_speech:
                    self.speech_frames.append(audio_chunk)
                    self.silence_chunks += 1

                    max_silence_chunks = int(self.max_silence_duration_s * self.sample_rate / self.vad_chunk_size)

                    if self.silence_chunks > max_silence_chunks:
                        logger.debug("Max silence reached, ending speech")

                        # Add trailing silence
                        trailing_chunks = int(self.trailing_silence_s * self.sample_rate / self.vad_chunk_size)
                        for _ in range(trailing_chunks):
                            self.speech_frames.append(np.zeros(self.vad_chunk_size, dtype=np.float32))

                        self.is_speech = False

                        if self.speech_frames:
                            # Concatenate and check duration
                            audio_data = np.concatenate(self.speech_frames)
                            duration = len(audio_data) / self.sample_rate

                            if duration >= self.min_speech_duration_s:
                                # Process this segment
                                result = self.process_audio(audio_data, self.sample_rate, save=True)

                                # Reset state
                                self.speech_frames = []
                                self.silence_chunks = 0
                                self.vad_iterator.reset_states()

                                return result

                        self.speech_frames = []
                        self.silence_chunks = 0
                        self.vad_iterator.reset_states()

        except Exception as e:
            logger.error(f"VAD processing error: {e}")

        return None

    def audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream"""
        if status:
            logger.warning(f"Audio callback status: {status}")

        audio = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        self.audio_queue.put(audio.copy())

    def process_stream(self):
        """Process audio stream with VAD"""
        accumulated_audio = np.array([], dtype=np.float32)

        while self.running:
            try:
                audio_chunk = self.audio_queue.get(timeout=0.1)
                accumulated_audio = np.concatenate([accumulated_audio, audio_chunk])

                while len(accumulated_audio) >= self.vad_chunk_size:
                    vad_chunk = accumulated_audio[:self.vad_chunk_size]
                    accumulated_audio = accumulated_audio[self.vad_chunk_size:]

                    result = self.process_vad_chunk(vad_chunk)

                    if result and self.stream_callback:
                        self.stream_callback(result)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Stream processing error: {e}")

    def start_stream(self, callback: Callable[[Dict[str, Any]], None], device=None):
        """
        Start continuous audio streaming with VAD and transcription.

        Args:
            callback: Function called with JSON result for each transcription
            device: Audio input device (None for default)
        """
        self.stream_callback = callback
        self.running = True

        # Start processing thread
        process_thread = threading.Thread(target=self.process_stream, daemon=True)
        process_thread.start()

        # Start audio stream
        try:
            logger.info(f"Starting audio stream (device: {device or 'default'})")
            with sd.InputStream(
                device=device,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.blocksize,
                dtype='float32',
                callback=self.audio_callback
            ):
                while self.running:
                    time.sleep(0.1)

        except Exception as e:
            logger.error(f"Audio stream error: {e}")
            self.running = False

        process_thread.join(timeout=2)

    def stop_stream(self):
        """Stop streaming"""
        self.running = False

    def start_cli(self):
        """Start CLI mode (continuous transcription with console output)"""
        print("\n🎤 Listenr - Unified ASR")
        print(f"Storage: {self.storage_base}")
        print(f"LLM: {'enabled' if self.use_llm else 'disabled'}")
        print("Press Ctrl+C to stop.\n")

        def cli_callback(result: Dict[str, Any]):
            timestamp = datetime.fromisoformat(result['timestamp']).strftime('%H:%M:%S')
            text = result['corrected_text'] if self.use_llm else result['transcription']
            print(f"\n[{timestamp}] {text}")

        try:
            self.start_stream(callback=cli_callback)
        except KeyboardInterrupt:
            print("\n\nStopping...")
            self.stop_stream()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Unified ASR - CLI mode')
    parser.add_argument('--llm', action='store_true', help='Enable LLM post-processing')
    parser.add_argument('--storage', help='Storage directory')
    args = parser.parse_args()

    asr = UnifiedASR(mode='cli', use_llm=args.llm, storage_base=args.storage)
    asr.start_cli()
