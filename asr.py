#!/usr/bin/env python3
"""
Simple Local ASR with Whisper and VAD
Uses config_manager.py for all configuration
Optional LLM post-processing via llm_processor.py
"""

import os
import sys
import time
import queue
import threading
import tempfile
import argparse
from datetime import datetime

import numpy as np
import sounddevice as sd
import soundfile as sf

import requests

# Import config manager
import config_manager as cfg


# Import Lemonade LLM and ASR functions
from llm_processor import lemonade_llm_correct, lemonade_transcribe_audio



# --- LemonadeASR: Use Lemonade Server for ASR and LLM ---
class LemonadeASR:
    """ASR system using Lemonade Server for both ASR and LLM"""
    def __init__(self, use_llm=True):
        import logging
        log_level = getattr(logging, cfg.get_setting('Logging', 'level'), logging.INFO)
        log_file = cfg.get_setting('Logging', 'file')
        log_handlers = [logging.StreamHandler()]
        if log_file:
            log_handlers.append(logging.FileHandler(log_file))
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=log_handlers
        )
        self.logger = logging.getLogger(__name__)
        self.use_llm = use_llm
        self.output_file = cfg.get_setting('Output', 'file')
        self.llm_output_file = cfg.get_setting('Output', 'llm_file', '')
        self.output_format = cfg.get_setting('Output', 'format')
        self.timestamp_format = cfg.get_setting('Output', 'timestamp_format')
        self.show_raw = cfg.get_bool_setting('Output', 'show_raw', False)

    def transcribe_and_correct(self, audio_path, whisper_model="Whisper-Tiny", llm_model="Qwen3-0.6B-GGUF", system_prompt=None):
        """Transcribe audio and optionally correct with LLM via Lemonade Server"""
        try:
            raw_text = lemonade_transcribe_audio(audio_path, model=whisper_model)
            corrected_text = None
            if self.use_llm:
                corrected_text = lemonade_llm_correct(raw_text, model=llm_model, system_prompt=system_prompt)
            else:
                corrected_text = raw_text
            self.output_transcription(raw_text, prefix="[RAW]" if self.use_llm else "")
            if self.use_llm:
                self.output_transcription(corrected_text, prefix="[LLM]")
            if self.output_file:
                self.save_transcription(raw_text, self.output_file, prefix="[RAW]" if self.use_llm else "")
            if self.llm_output_file and self.use_llm:
                self.save_transcription(corrected_text, self.llm_output_file, prefix="[LLM]")
            return {"raw_text": raw_text, "corrected_text": corrected_text}
        except Exception as e:
            self.logger.error(f"Transcription or LLM correction failed: {e}")
            return {"error": str(e)}

    def output_transcription(self, text, prefix=""):
        timestamp = datetime.now().strftime(self.timestamp_format)
        full_text = f"{prefix} {text}".strip() if prefix else text
        output = self.output_format.format(timestamp=timestamp, text=full_text)
        print(f"\n{output}")

    def save_transcription(self, text, filepath, prefix=""):
        try:
            timestamp = datetime.now().strftime(self.timestamp_format)
            full_text = f"{prefix} {text}".strip() if prefix else text
            output = self.output_format.format(timestamp=timestamp, text=full_text)
            output_path = os.path.expanduser(filepath)
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            with open(output_path, 'a', encoding='utf-8') as f:
                f.write(f"{output}\n")
            self.logger.debug(f"Saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save transcription: {e}")
    
    def load_whisper_model(self):
        """Load Whisper model based on config (faster-whisper)"""
        model_size = cfg.get_setting('Whisper', 'model_size', 'base')
        device = cfg.get_setting('Whisper', 'device', 'cpu')
        compute_type = cfg.get_setting('Whisper', 'compute_type', 'float16')

        # Auto-adjust compute_type based on device
        if device in ('cuda', 'gpu') and compute_type == 'int8':
            compute_type = 'float16'
            self.logger.info("Using float16 compute type for CUDA/GPU")
        elif device == 'cpu' and compute_type not in ('int8', 'float32'):
            compute_type = 'int8'
            self.logger.info("Using int8 compute type for CPU")

        self.logger.info(f"Loading Whisper model: {model_size} on {device} (faster-whisper, compute_type={compute_type})")
        try:
            from faster_whisper import WhisperModel
            self.whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
            self.logger.info("Whisper model loaded successfully (faster-whisper)")
        except ImportError:
            self.logger.error("faster-whisper not installed. Install with: pip install faster-whisper")
            sys.exit(1)
        except Exception as e:
            self.logger.error(f"Failed to load Whisper model: {e}")
            sys.exit(1)
    
    def load_vad_model(self):
        """Load Silero VAD model"""
        self.logger.info("Loading Silero VAD model")
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
            self.logger.info("VAD model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load VAD model: {e}")
            sys.exit(1)
    
    def audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream"""
        if status:
            self.logger.warning(f"Audio callback status: {status}")
        
        # Convert to mono if needed
        audio = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        
        # Add to processing queue
        self.audio_queue.put(audio.copy())
    
    def process_audio(self):
        """Process audio chunks with VAD"""
        self.logger.info("Starting audio processing thread")
        
        # Buffer to accumulate audio until we have enough for VAD
        accumulated_audio = np.array([], dtype=np.float32)
        
        while self.running:
            try:
                # Get audio chunk with timeout
                audio_chunk = self.audio_queue.get(timeout=0.1)
                
                # Accumulate audio
                accumulated_audio = np.concatenate([accumulated_audio, audio_chunk])
                
                # Process in VAD-sized chunks
                while len(accumulated_audio) >= self.vad_chunk_size:
                    # Extract VAD-sized chunk
                    vad_chunk = accumulated_audio[:self.vad_chunk_size]
                    accumulated_audio = accumulated_audio[self.vad_chunk_size:]
                    
                    # Process this chunk
                    self.process_vad_chunk(vad_chunk)
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in audio processing: {e}")
                
        self.logger.info("Audio processing thread stopped")
    
    def process_vad_chunk(self, audio_chunk):
        """Process a single VAD-sized audio chunk"""

        # Concatenate all frames
            parser = argparse.ArgumentParser(
                description='ASR with Lemonade Server (Whisper + LLM)',
                formatter_class=argparse.RawDescriptionHelpFormatter,
                epilog=f"""
        Config file: {cfg.CONFIG_FILE}
        Edit this file to change settings permanently.

        Examples:
          %(prog)s --audio path/to/audio.wav         # Transcribe audio file
          %(prog)s --no-llm                         # Disable LLM post-processing
                """
            )
            parser.add_argument('--no-llm', action='store_true', help='Disable LLM post-processing')
            parser.add_argument('--audio', type=str, help='Path to audio file to transcribe')
            parser.add_argument('--whisper-model', type=str, default='Whisper-Tiny', help='Whisper model name')
            parser.add_argument('--llm-model', type=str, default='Qwen3-0.6B-GGUF', help='LLM model name')
            parser.add_argument('--system-prompt', type=str, default=None, help='System prompt for LLM')
            args = parser.parse_args()

            asr = LemonadeASR(use_llm=not args.no_llm)
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
        audio_data = np.concatenate(self.speech_frames)
        
        # Check minimum duration
        duration = len(audio_data) / self.sample_rate
        if duration < self.min_speech_duration_s:
            self.logger.debug(f"Speech segment too short ({duration:.2f}s), skipping")
            return
        
        clip_metadata = None
        clip_timestamp = datetime.now()

        if self.audio_clips_enabled:
            clip_extension = 'wav'
            if self.clip_format in {'wav', 'flac'}:
                clip_extension = self.clip_format
            elif self.clip_format:
                self.logger.warning(
                    f"Unsupported clip format '{self.clip_format}' requested; defaulting to WAV."
                )

            clip_dir = os.path.join(self.audio_clips_dir, clip_timestamp.strftime('%Y-%m-%d'))
            try:
                os.makedirs(clip_dir, exist_ok=True)
                clip_filename = (
                    f"clip_{clip_timestamp.strftime('%Y%m%d_%H%M%S')}_{self.clip_counter:03d}.{clip_extension}"
                )
                clip_path = os.path.join(clip_dir, clip_filename)
                sf.write(clip_path, audio_data, self.sample_rate)
                clip_metadata = {
                    'audio_file': clip_path,
                    'timestamp': clip_timestamp.isoformat(),
                    'duration_ms': int(len(audio_data) * 1000 / self.sample_rate),
                }
                self.logger.info(f"Saved audio clip to {clip_path}")
                self.clip_counter += 1
            except Exception as exc:
                self.logger.error(f"Failed to save audio clip: {exc}")
                clip_metadata = None
        
        self.logger.info(f"Processing speech segment ({duration:.2f}s)")
        
        # Save to temporary file for Whisper
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sf.write(tmp_file.name, audio_data, self.sample_rate)
            tmp_path = tmp_file.name
        
        try:
            # Transcribe with Whisper
            beam_size = cfg.get_int_setting('Whisper', 'beam_size')
            best_of = cfg.get_int_setting('Whisper', 'best_of')
            temperature = cfg.get_float_setting('Whisper', 'temperature')
            condition_on_previous = cfg.get_bool_setting('Whisper', 'condition_on_previous_text', False)
            vad_filter = cfg.get_bool_setting('Whisper', 'vad_filter', False)
            corrected_text = None
            
            # Build transcribe arguments
            transcribe_kwargs = {
                'beam_size': beam_size,
                'best_of': best_of,
                'temperature': temperature,
                'condition_on_previous_text': condition_on_previous,
                'vad_filter': vad_filter,
                'word_timestamps': False,  # Faster without word timestamps
            }
            
            # Add initial prompt for better context if needed
            if self.last_transcription and condition_on_previous:
                # Use last transcription as context (helps with continuing thoughts)
                transcribe_kwargs['initial_prompt'] = self.last_transcription[-200:]  # Last 200 chars
            
            segments, info = self.whisper_model.transcribe(tmp_path, **transcribe_kwargs)
            
            # Get full text
            raw_text = ' '.join(segment.text.strip() for segment in segments)
            
            if raw_text:
                # Store for context
                self.last_transcription = raw_text
                
                # Always save raw transcription if file configured
                if self.output_file:
                    self.save_transcription(raw_text, self.output_file, prefix="[RAW]" if self.use_llm else "")
                
                # Process with LLM if enabled
                if self.use_llm and self.llm_processor and self.llm_processor.available:
                    corrected_text = self.llm_processor.process(raw_text, use_context=True)
                    
                    # Output based on settings
                    if self.show_raw:
                        self.output_transcription(raw_text, prefix="[RAW]")
                    
                    self.output_transcription(corrected_text, prefix="")
                    
                    # Save LLM output if configured
                    if self.llm_output_file:
                        self.save_transcription(corrected_text, self.llm_output_file, prefix="")
                else:
                    # No LLM processing - just output raw
                    self.output_transcription(raw_text, prefix="")
                    corrected_text = raw_text if raw_text else None

                if clip_metadata is not None and raw_text:
                    clip_metadata['raw_text'] = raw_text
                    clip_metadata['corrected_text'] = corrected_text or raw_text
                    return clip_metadata
            
        except Exception as e:
            self.logger.error(f"Transcription error: {e}")
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass
    
    def output_transcription(self, text, prefix=""):
        """Output transcribed text to console"""
        timestamp = datetime.now().strftime(self.timestamp_format)
        
        # Format output
        full_text = f"{prefix} {text}".strip() if prefix else text
        output = self.output_format.format(timestamp=timestamp, text=full_text)
        
        # Console output
        print(f"\n{output}")
    
    def save_transcription(self, text, filepath, prefix=""):
        """Save transcription to file"""
        try:
            timestamp = datetime.now().strftime(self.timestamp_format)
            full_text = f"{prefix} {text}".strip() if prefix else text
            output = self.output_format.format(timestamp=timestamp, text=full_text)
            
            output_path = os.path.expanduser(filepath)
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            
            with open(output_path, 'a', encoding='utf-8') as f:
                f.write(f"{output}\n")
                
            self.logger.debug(f"Saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save transcription: {e}")
    
    def list_audio_devices(self):
        """List available audio input devices"""
        print("\nAvailable audio input devices:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                default = " (DEFAULT)" if i == sd.default.device[0] else ""
                print(f"  [{i}] {device['name']} - {device['max_input_channels']} channels{default}")
    
    def get_audio_device(self):
        """Get audio device from config"""
        device_config = cfg.get_setting('Audio', 'input_device')
        if device_config.lower() == 'default':
            return None
        try:
            return int(device_config)
        except ValueError:
            self.logger.warning(f"Invalid device '{device_config}', using default")
            return None
    
    def start(self):
        """Start ASR system"""
        self.logger.info("Starting ASR system")
        if self.use_llm:
            self.logger.info("LLM post-processing enabled")
        
        device = self.get_audio_device()
        self.running = True
        
        # Start processing thread
        process_thread = threading.Thread(target=self.process_audio, daemon=True)
        process_thread.start()
        
        # Start audio stream
        try:
            self.logger.info(f"Starting audio stream (device: {device or 'default'})")
            with sd.InputStream(
                device=device,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.blocksize,
                dtype='float32',
                callback=self.audio_callback
            ):
                print("\n🎤 ASR is running. Speak into your microphone.")
                if self.use_llm:
                    print("✨ LLM post-processing enabled for improved accuracy")
                print(f"Config: {cfg.CONFIG_FILE}")
                print("Press Ctrl+C to stop.\n")
                
                # Keep running until interrupted
                while self.running:
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\n\nStopping ASR...")
            self.running = False
        except Exception as e:
            self.logger.error(f"Audio stream error: {e}")
            self.running = False
        
        # Wait for processing thread
        process_thread.join(timeout=2)
        self.logger.info("ASR system stopped")


def main():
    parser = argparse.ArgumentParser(
        description='Local ASR with Whisper, VAD, and optional LLM post-processing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Config file: {cfg.CONFIG_FILE}
Edit this file to change settings permanently.

LLM Integration (if llm_processor.py is available):
  Requires Ollama running locally:
    1. Install: https://ollama.ai
    2. Pull model: ollama pull gemma2:2b
    3. Start: ollama serve

Examples:
  %(prog)s                     # Use config file settings
  %(prog)s --no-llm            # Disable LLM post-processing
  %(prog)s -m large-v2         # Override model size
  %(prog)s -d cuda             # Override device
  %(prog)s -o transcripts.txt  # Override output file
  %(prog)s --list-devices      # List available audio devices
  %(prog)s --edit-config       # Open config in editor
        """
    )
    
    parser.add_argument('--no-llm',
                       action='store_true',
                       help='Disable LLM post-processing')
    parser.add_argument('-m', '--model',
                       help='Override Whisper model size')
    parser.add_argument('-d', '--device',
                       choices=['cpu', 'cuda'],
                       help='Override compute device')
    parser.add_argument('-o', '--output',
                       help='Override output file')
    parser.add_argument('--list-devices',
                       action='store_true',
                       help='List available audio devices and exit')
    parser.add_argument('--edit-config',
                       action='store_true',
                       help='Open config file in default editor')
    parser.add_argument('--show-config',
                       action='store_true',
                       help='Show current configuration and exit')
    
    args = parser.parse_args()
    
    # Handle special actions
    if args.edit_config:
        import subprocess
        editor = os.environ.get('EDITOR', 'nano')
        subprocess.call([editor, cfg.CONFIG_FILE])
        return
    
    if args.show_config:
        print(f"Configuration from {cfg.CONFIG_FILE}:\n")
        for section in cfg.config.sections():
            print(f"[{section}]")
            for key, value in cfg.config.items(section):
                print(f"  {key} = {value}")
            print()
        return
    
    # Apply command-line overrides
    if args.model:
        cfg.update_setting('Whisper', 'model_size', args.model)
    if args.device:
        cfg.update_setting('Whisper', 'device', args.device)
    if args.output:
        cfg.update_setting('Output', 'file', args.output)
    
    # Create and start ASR
    asr = WhisperASR(use_llm=not args.no_llm)
    
    if args.list_devices:
        asr.list_audio_devices()
        return
    
    asr.start()



if __name__ == '__main__':
    main()
