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

import requests

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

    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int, streaming: bool = False) -> Dict[str, Any]:
        """
        Transcribe audio data using Whisper.

        Args:
            audio_data: Numpy array (mono, float32)
            sample_rate: Sample rate
            streaming: If True, enable streaming mode for faster partial results

        Returns:
            Dictionary with transcription results
        """
        # Save to temp file for Whisper
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sf.write(tmp_file.name, audio_data, sample_rate)
            tmp_path = tmp_file.name

        try:
            # Optimized settings for real-time processing
            if streaming:
                beam_size = 1  # Greedy search for speed
                best_of = 1
                temperature = 0.0
            else:
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
                word_timestamps=True if streaming else False,  # Get word timings for streaming
                language='en'  # Force language for speed (remove if multilingual)
            )

            # Collect segments with timing info
            segments_list = []
            raw_text_parts = []

            for segment in segments:
                text = segment.text.strip()
                if text:
                    raw_text_parts.append(text)
                    segments_list.append({
                        'text': text,
                        'start': segment.start,
                        'end': segment.end,
                        'words': [{'word': w.word, 'start': w.start, 'end': w.end}
                                 for w in segment.words] if hasattr(segment, 'words') and segment.words else []
                    })

            raw_text = ' '.join(raw_text_parts)

            # Apply LLM if enabled (skip in streaming for speed)
            corrected_text = None
            if not streaming and self.use_llm and self.llm_processor and self.llm_processor.available and raw_text:
                try:
                    corrected_text = self.llm_processor.process(raw_text, use_context=False)
                except Exception as e:
                    logger.error(f"LLM processing failed: {e}")
                    corrected_text = None

            return {
                'success': True,
                'raw_text': raw_text,
                'corrected_text': corrected_text or raw_text,
                'segments': segments_list,
                'language': info.language if hasattr(info, 'language') else None,
                'language_probability': float(info.language_probability) if hasattr(info, 'language_probability') else None
            }

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'raw_text': '',
                'corrected_text': '',
                'segments': []
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

            # Transcribe (use streaming mode for real-time)
            streaming = self.mode == 'stream'
            transcription_result = self.transcribe_audio(audio_data, sample_rate, streaming=streaming)

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
                'segments': transcription_result.get('segments', []),
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
                    'mode': self.mode,
                    'streaming': streaming
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
