# Technical Product Requirements Document
## Real-Time ASR Transcription System with LLM Correction
### Version 2.0 - Updated with Current Implementation Status

### 1. Executive Summary

**Product Name:** listnr - Real-Time ASR Transcription System  
**Version:** 1.0.0  
**Date:** October 2025  
**Status:** Partial Implementation - Enhancement Required

#### 1.1 Current State
The system currently implements:
- ✅ Real-time microphone capture with Silero VAD
- ✅ Fast Whisper transcription
- ✅ Ollama-based LLM correction (basic)
- ✅ INI configuration management
- ✅ Text file output
- ❌ Audio clip storage with naming convention
- ❌ JSON output format
- ❌ Audio-to-transcription mapping
- ❌ Comprehensive LLM correction patterns

---

### 2. Implementation Status & Required Changes

#### 2.1 Existing Components

| Component | Current Status | Required Changes |
|-----------|---------------|------------------|
| **asr.py** | Main script with VAD and transcription | Add audio clip saving, JSON output |
| **audio_recorder.py** | Silero VAD implementation | Already good, minor updates for clip saving |
| **config_manager.py** | Basic INI config | Add audio storage settings |
| **llm_processor.py** | Basic Ollama integration | Enhance correction types, add mapping support |
| **asr_processor.py** | Not implemented | Create for modular transcription handling |
| **output_handler.py** | Not implemented | Create for JSON formatting and mapping |

#### 2.2 Architecture Changes Needed

```diff
Current Architecture:
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Audio Input │ --> │ VAD Process │ --> │ Transcribe  │
└─────────────┘     └─────────────┘     └─────────────┘
                                              │
                                              v
                                        ┌─────────────┐
                                        │ LLM Correct │
                                        └─────────────┘
                                              │
                                              v
                                        ┌─────────────┐
                                        │  Text File  │
                                        └─────────────┘

Target Architecture:
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Audio Input │ --> │ VAD Process │ --> │ Save Clip   │
└─────────────┘     └─────────────┘     └─────────────┘
                                              │
                                              v
                                        ┌─────────────┐
                                        │ Transcribe  │
                                        └─────────────┘
                                              │
                                              v
                                        ┌─────────────┐
                                        │ LLM Correct │
                                        └─────────────┘
                                              │
                                              v
                                        ┌─────────────┐
                                        │ JSON Output │
                                        │  + Mapping  │
                                        └─────────────┘
```

---

### 3. Required Modifications

#### 3.1 Audio Clip Storage (NEW FEATURE)

**Location:** Modify `asr.py` in `process_speech_segment()` method

```python
def process_speech_segment(self):
    """Process a complete speech segment"""
    if not self.speech_frames:
        return
    
    # Concatenate all frames
    audio_data = np.concatenate(self.speech_frames)
    
    # NEW: Generate clip filename
    timestamp = datetime.now()
    clip_filename = f"clip_{timestamp.strftime('%Y%m%d_%H%M%S')}_{self.clip_counter:03d}.wav"
    clip_path = os.path.join(self.audio_clips_dir, timestamp.strftime('%Y-%m-%d'), clip_filename)
    
    # NEW: Save audio clip
    os.makedirs(os.path.dirname(clip_path), exist_ok=True)
    sf.write(clip_path, audio_data, self.sample_rate)
    
    # Existing transcription code...
    # ...
    
    # NEW: Return clip metadata with transcription
    return {
        'audio_file': clip_path,
        'timestamp': timestamp.isoformat(),
        'duration_ms': len(audio_data) * 1000 / self.sample_rate,
        'raw_text': raw_text,
        'corrected_text': corrected_text
    }
```

#### 3.2 Configuration Updates

**Location:** Add to `config_manager.py` DEFAULT_CONFIG

```python
DEFAULT_CONFIG = {
    # ... existing config ...
    
    'Storage': {
        'audio_clips_enabled': 'true',
        'audio_clips_path': '~/.listnr/audio_clips',
        'retention_days': '7',
        'max_storage_gb': '10',
        'clip_format': 'wav',  # wav, flac, mp3
        'clip_quality': '16000',  # sample rate for saved clips
    },
    
    'Output': {
        'file': '~/transcripts_raw.txt',
        'llm_file': '~/transcripts_clean.txt',
        'json_file': '~/transcripts.json',  # NEW
        'format': '[{timestamp}] {text}',
        'timestamp_format': '%Y-%m-%d %H:%M:%S',
        'show_raw': 'false',
        'output_mode': 'both',  # text, json, both
        'include_audio_mapping': 'true',  # NEW
        'include_word_timestamps': 'false',  # NEW
        'include_confidence_scores': 'true',  # NEW
    },
    
    'LLM': {
        'enabled': 'true',
        'provider': 'ollama',  # ollama, openai
        'model': 'gemma2:2b',
        'ollama_host': 'http://localhost:11434',
        'openai_api_key_env': 'OPENAI_API_KEY',  # NEW - for future OpenAI support
        'temperature': '0.3',
        'context_window': '5',
        'max_tokens': '150',
        'timeout': '10',
        'correction_types': 'homophone,number,contextual,grammar,punctuation',  # NEW
        'correction_threshold': '0.7',  # NEW - confidence threshold
    }
}
```

#### 3.3 Output Handler Implementation (NEW FILE)

**Create:** `output_handler.py`

```python
"""
Output Handler for listnr
Manages JSON output format and audio-to-transcription mapping
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import config_manager as cfg
import logging

logger = logging.getLogger(__name__)

class OutputHandler:
    def __init__(self):
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.segments = []
        self.json_file = os.path.expanduser(cfg.get_setting('Output', 'json_file'))
        self.include_mapping = cfg.get_bool_setting('Output', 'include_audio_mapping')
        self.include_confidence = cfg.get_bool_setting('Output', 'include_confidence_scores')
        
    def add_segment(self, 
                   audio_file: Optional[str],
                   raw_text: str,
                   corrected_text: Optional[str],
                   start_time: datetime,
                   end_time: datetime,
                   corrections: Optional[List[Dict]] = None,
                   confidence: float = 0.0):
        """Add a transcription segment"""
        
        segment = {
            "id": f"seg_{len(self.segments) + 1:03d}",
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_ms": int((end_time - start_time).total_seconds() * 1000),
            "raw_text": raw_text
        }
        
        if self.include_mapping and audio_file:
            segment["audio_file"] = os.path.basename(audio_file)
            segment["audio_path"] = audio_file
        
        if corrected_text:
            segment["corrected_text"] = corrected_text
            
        if corrections:
            segment["corrections"] = corrections
            
        if self.include_confidence:
            segment["confidence"] = confidence
        
        self.segments.append(segment)
        self.save_json()
        
    def save_json(self):
        """Save current session to JSON file"""
        output = {
            "version": "1.0",
            "session_id": self.session_id,
            "segments": self.segments,
            "metadata": {
                "whisper_model": cfg.get_setting('Whisper', 'model_size'),
                "llm_provider": cfg.get_setting('LLM', 'provider'),
                "llm_model": cfg.get_setting('LLM', 'model'),
                "vad_model": "silero",
                "total_segments": len(self.segments),
                "session_start": self.session_id
            }
        }
        
        try:
            os.makedirs(os.path.dirname(self.json_file) or '.', exist_ok=True)
            with open(self.json_file, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved JSON to {self.json_file}")
        except Exception as e:
            logger.error(f"Failed to save JSON: {e}")
            
    def get_audio_mapping(self) -> Dict[str, str]:
        """Get mapping of audio files to transcriptions"""
        mapping = {}
        for segment in self.segments:
            if 'audio_file' in segment:
                text = segment.get('corrected_text', segment.get('raw_text', ''))
                mapping[segment['audio_file']] = text
        return mapping
```

#### 3.4 Enhanced LLM Processor

**Modify:** `llm_processor.py` - Add comprehensive correction patterns

```python
class LLMProcessor:
    def __init__(self, ...existing params...):
        # ... existing init code ...
        
        # Enhanced system prompt with specific correction types
        self.system_prompt = """You are a transcription editor specializing in speech-to-text error correction.

CORRECTION TYPES TO APPLY:
1. HOMOPHONES: Fix words that sound similar but are spelled differently (by→buy, there→their, two→to/too)
2. NUMBERS: Convert spoken numbers to appropriate format based on context (2→two for general text, keep digits for addresses/codes)
3. CONTEXTUAL: Fix words that don't make sense in context (abuse→use, weather→whether)
4. GRAMMAR: Fix subject-verb agreement, tense consistency
5. PUNCTUATION: Add appropriate punctuation and capitalization
6. CONTRACTIONS: Maintain as spoken (don't change don't to do not)

RULES:
- Output ONLY the corrected text
- Never add words or ideas not in the original
- Preserve speaker's tone and meaning
- If uncertain, prefer minimal changes"""
        
        # Track correction patterns for reporting
        self.correction_patterns = {
            'homophones': [
                (r'\bby\b(?=\s+(some|any|the|a|an|my|your|his|her))', 'buy'),
                (r'\btwo\b(?=\s+(by|for|of))', 'to'),
                (r'\bthere\b(?=\s+(is|are|was|were))', "there"),
                (r'\btheir\b(?=\s+(going|coming|walking))', "they're"),
                (r'\byour\b(?=\s+(going|coming|walking))', "you're"),
                (r'\bits\b(?=\s+(going|coming|been))', "it's"),
                (r'\bhear\b(?=\s+(me|you|them|us))', 'here'),
                (r'\bweather\b(?=\s+(or|to|you|I))', 'whether'),
            ]
        }
    
    def detect_corrections(self, original: str, corrected: str) -> List[Dict]:
        """Detect what corrections were made"""
        corrections = []
        
        # Simple word-level diff
        original_words = original.split()
        corrected_words = corrected.split()
        
        # Use difflib to find changes
        import difflib
        matcher = difflib.SequenceMatcher(None, original_words, corrected_words)
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                corrections.append({
                    'type': 'word_replacement',
                    'original': ' '.join(original_words[i1:i2]),
                    'corrected': ' '.join(corrected_words[j1:j2]),
                    'position': i1
                })
            elif tag == 'insert':
                corrections.append({
                    'type': 'insertion',
                    'inserted': ' '.join(corrected_words[j1:j2]),
                    'position': i1
                })
        
        return corrections
```

#### 3.5 Audio Cleanup Service (NEW FILE)

**Create:** `cleanup_service.py`

```python
"""
Cleanup Service for audio clips
Removes old clips based on retention policy
"""

import os
import time
from datetime import datetime, timedelta
import config_manager as cfg
import logging

logger = logging.getLogger(__name__)

class CleanupService:
    def __init__(self):
        self.audio_path = os.path.expanduser(cfg.get_setting('Storage', 'audio_clips_path'))
        self.retention_days = cfg.get_int_setting('Storage', 'retention_days')
        self.max_storage_gb = cfg.get_float_setting('Storage', 'max_storage_gb')
        
    def cleanup_old_files(self):
        """Remove audio files older than retention period"""
        if not os.path.exists(self.audio_path):
            return
        
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        removed_count = 0
        removed_size = 0
        
        for root, dirs, files in os.walk(self.audio_path):
            for file in files:
                if file.endswith(('.wav', '.flac', '.mp3')):
                    file_path = os.path.join(root, file)
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    
                    if file_time < cutoff_date:
                        size = os.path.getsize(file_path)
                        os.remove(file_path)
                        removed_count += 1
                        removed_size += size
                        logger.debug(f"Removed old clip: {file}")
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} files ({removed_size / 1024 / 1024:.2f} MB)")
    
    def check_storage_limit(self):
        """Check if storage exceeds limit"""
        total_size = 0
        for root, dirs, files in os.walk(self.audio_path):
            for file in files:
                if file.endswith(('.wav', '.flac', '.mp3')):
                    total_size += os.path.getsize(os.path.join(root, file))
        
        size_gb = total_size / (1024 ** 3)
        if size_gb > self.max_storage_gb:
            logger.warning(f"Storage limit exceeded: {size_gb:.2f} GB / {self.max_storage_gb} GB")
            # Could implement automatic cleanup of oldest files here
        
        return size_gb
```

---

### 4. Implementation Roadmap

#### Phase 1: Core Audio Storage (Week 1)
- [x] ~~Basic audio capture and VAD~~ ✅ Already implemented
- [x] ~~Whisper transcription~~ ✅ Already implemented
- [x] Add audio clip saving to `asr.py`
- [x] Update config with storage settings
- [ ] Create cleanup service

#### Phase 2: JSON Output & Mapping (Week 1-2)
- [ ] Implement `output_handler.py`
- [ ] Integrate JSON output into `asr.py`
- [ ] Add audio-to-transcription mapping
- [ ] Test end-to-end JSON generation

#### Phase 3: Enhanced LLM Processing (Week 2)
- [x] ~~Basic LLM correction~~ ✅ Already implemented
- [x] Add correction pattern detection
- [ ] Implement confidence scoring
- [ ] Add OpenAI API support (optional)
- [ ] Track correction statistics

#### Phase 4: Testing & Optimization (Week 3)
- [ ] Performance testing with long recordings
- [ ] Storage limit testing
- [ ] JSON output validation
- [ ] Docker container optimization

---

### 5. Migration Guide

#### 5.1 Config File Migration
Users will need to update their existing config files with new sections:

```bash
# Backup existing config
cp ~/.config/listnr/config.ini ~/.config/listnr/config.ini.backup

# The application will add new sections automatically on first run
```

#### 5.2 Breaking Changes
- JSON output format is new - no breaking changes for text output
- Audio clips storage is optional (can be disabled in config)
- Existing LLM functionality remains unchanged

#### 5.3 New Dependencies
```txt
# No new dependencies required - using existing libraries
# Optional: openai>=1.0.0 for OpenAI support
```

---

### 6. Testing Requirements

#### 6.1 Unit Tests Needed
```python
# test_output_handler.py
- test_json_generation()
- test_audio_mapping()
- test_segment_addition()

# test_cleanup_service.py  
- test_retention_cleanup()
- test_storage_limit_check()

# test_audio_storage.py
- test_clip_saving()
- test_directory_structure()
- test_filename_generation()
```

#### 6.2 Integration Tests
- End-to-end recording → clip storage → transcription → JSON output
- Multi-session handling
- Storage limit enforcement
- Cleanup service scheduling

---

### 7. Example Usage

#### 7.1 Basic Usage (Current)
```bash
python asr.py  # Text output only
```

#### 7.2 Enhanced Usage (After Implementation)
```bash
python asr.py --json  # Enable JSON output with audio mapping
python asr.py --no-clips  # Disable audio clip storage
python asr.py --cleanup  # Run cleanup service
```

#### 7.3 Sample JSON Output
```json
{
  "version": "1.0",
  "session_id": "20251010_143022",
  "segments": [
    {
      "id": "seg_001",
      "audio_file": "clip_20251010_143022_001.wav",
      "audio_path": "~/.listnr/audio_clips/2025-10-10/clip_20251010_143022_001.wav",
      "start_time": "2025-10-10T14:30:22.000Z",
      "end_time": "2025-10-10T14:30:27.500Z",
      "duration_ms": 5500,
      "raw_text": "I went to the store to by 2 bags of milk",
      "corrected_text": "I went to the store to buy two bags of milk",
      "confidence": 0.92,
      "corrections": [
        {
          "type": "word_replacement",
          "original": "by",
          "corrected": "buy",
          "position": 5
        },
        {
          "type": "word_replacement",
          "original": "2",
          "corrected": "two",
          "position": 6
        }
      ]
    }
  ],
  "metadata": {
    "whisper_model": "base.en",
    "llm_provider": "ollama",
    "llm_model": "gemma2:2b",
    "vad_model": "silero",
    "total_segments": 1,
    "session_start": "20251010_143022"
  }
}
```

---

### 8. Notes on Current Implementation

#### Strengths
- ✅ Solid VAD implementation with Silero
- ✅ Good config management system
- ✅ Working LLM integration with Ollama
- ✅ Docker support

#### Areas for Enhancement
- ❌ No JSON output format
- ❌ No audio-to-text mapping
- ❌ No storage management

#### Recommendations
2. Implement JSON output handler - provides structured data
4. Add cleanup service - prevents unbounded storage growth