# Product Requirements Document: Dataset Manager
## Fine-Tuning Data Pipeline for listnr

### Document Information
- **Component:** Dataset Manager for listnr
- **Version:** 1.0.0
- **Date:** October 2025
- **Status:** Specification
- **Parent PRD:** listnr-prd-v2.md

---

## 1. Executive Summary

### 1.1 Purpose
The Dataset Manager is a standalone component that processes raw transcription data from listnr into curated, fine-tuning-ready datasets compatible with Whisper fine-tuning workflows, including Mozilla's speech-to-text-finetune and Hugging Face transformers.

### 1.2 Core Principles
- **Non-invasive:** Never modifies original audio clips or transcriptions
- **Incremental:** Processes data in batches without blocking real-time operations
- **Quality-focused:** Automatically filters and scores segments for training suitability
- **Format-agnostic:** Exports to multiple fine-tuning framework formats

---

## 2. Required Metadata from listnr

### 2.1 Mandatory Fields (Must Collect in Real-Time)

These fields MUST be collected by listnr during transcription for dataset manager to function:

```python
class RequiredMetadata:
    # Audio Properties
    audio_filepath: str          # Full path to saved audio clip
    audio_duration_seconds: float  # Exact duration in seconds
    audio_sample_rate: int       # Sample rate (must be 16000 for Whisper)
    audio_channels: int          # Number of channels (1=mono, 2=stereo)
    audio_format: str            # File format (wav, flac, mp3)
    
    # Transcription Data
    transcription_raw: str       # Original Whisper output (uncorrected)
    transcription_corrected: str # LLM-corrected version (if available)
    transcription_timestamp: str # ISO 8601 timestamp of transcription
    
    # Quality Indicators
    vad_triggered: bool          # Whether VAD detected speech
    vad_confidence: float        # VAD confidence score (0.0-1.0)
    whisper_model: str           # Model used (tiny, base, small, etc.)
    
    # Session Context
    session_id: str              # Unique session identifier
    segment_index: int           # Sequential number within session
    user_id: str                 # Anonymous user/device identifier
```

### 2.2 Optional But Valuable Fields

Collect if possible for better quality assessment:

```python
class OptionalMetadata:
    # Enhanced Quality Metrics
    whisper_confidence: float    # Whisper's confidence score if available
    whisper_language: str        # Detected language code
    whisper_language_probability: float  # Language detection confidence
    
    # LLM Correction Details
    llm_model: str              # Model used for correction
    llm_corrections_made: int   # Number of changes made
    llm_correction_types: List[str]  # Types of corrections applied
    
    # Audio Quality
    audio_max_amplitude: float  # Peak amplitude (for clipping detection)
    audio_mean_amplitude: float # Average amplitude
    silence_ratio: float        # Percentage of silence in clip
    
    # Processing Performance
    transcription_time_ms: float  # Time taken to transcribe
    correction_time_ms: float     # Time taken for LLM correction
    
    # Environmental Context
    noise_detected: bool        # If background noise was detected
    multiple_speakers: bool     # If multiple voices detected
```

### 2.3 Metadata Storage Format

Listnr should append this metadata to a JSONL file in real-time:

**Location:** `~/.listnr/datasets/raw/manifest.jsonl`

**Format (one JSON object per line):**
```json
{
  "timestamp": "2025-10-10T14:30:22.123Z",
  "segment_id": "seg_20251010_143022_001",
  "audio": {
    "filepath": "/home/user/.listnr/audio_clips/2025-10-10/clip_20251010_143022_001.wav",
    "duration_seconds": 5.47,
    "sample_rate": 16000,
    "channels": 1,
    "format": "wav"
  },
  "transcription": {
    "raw": "i went to the store to by some milk",
    "corrected": "I went to the store to buy some milk.",
    "model": "whisper-base.en",
    "language": "en",
    "language_probability": 0.99
  },
  "quality": {
    "vad_confidence": 0.95,
    "whisper_confidence": 0.87,
    "corrections_made": 2
  },
  "session": {
    "session_id": "session_20251010_143000",
    "segment_index": 1,
    "user_id": "device_001"
  }
}
```

---

## 3. Dataset Manager Processing Pipeline

### 3.1 Quality Scoring Algorithm

The dataset manager calculates a composite quality score:

```python
def calculate_quality_score(metadata: dict) -> float:
    score = 0.0
    weights = {
        'duration': 0.25,      # Optimal duration (2-15 seconds)
        'confidence': 0.35,    # VAD and Whisper confidence
        'corrections': 0.20,   # Fewer corrections = better
        'audio_quality': 0.20  # SNR, no clipping
    }
    
    # Duration scoring (optimal: 2-15 seconds)
    duration = metadata['audio']['duration_seconds']
    if 2 <= duration <= 15:
        score += weights['duration'] * 1.0
    elif 1 <= duration < 2 or 15 < duration <= 30:
        score += weights['duration'] * 0.7
    else:
        score += weights['duration'] * 0.3
    
    # Confidence scoring
    vad_conf = metadata['quality'].get('vad_confidence', 0.5)
    whisper_conf = metadata['quality'].get('whisper_confidence', 0.5)
    avg_confidence = (vad_conf + whisper_conf) / 2
    score += weights['confidence'] * avg_confidence
    
    # Correction scoring (fewer is better)
    corrections = metadata['quality'].get('corrections_made', 0)
    if corrections == 0:
        score += weights['corrections'] * 1.0
    elif corrections <= 2:
        score += weights['corrections'] * 0.8
    elif corrections <= 5:
        score += weights['corrections'] * 0.5
    else:
        score += weights['corrections'] * 0.2
    
    # Audio quality (placeholder - implement SNR calculation)
    score += weights['audio_quality'] * 0.8
    
    return min(score, 1.0)
```

### 3.2 Categorization Thresholds

```python
QUALITY_THRESHOLDS = {
    'excellent': 0.9,   # Auto-accept for training
    'good': 0.75,       # Accept with minor review
    'fair': 0.6,        # Requires manual review
    'poor': 0.0         # Auto-reject
}
```

### 3.3 Dataset Split Assignment

```python
def assign_split(segment_index: int, total_segments: int) -> str:
    """Deterministic split assignment based on position"""
    train_ratio = 0.8
    val_ratio = 0.15
    test_ratio = 0.05
    
    position = segment_index / total_segments
    
    if position < train_ratio:
        return 'train'
    elif position < train_ratio + val_ratio:
        return 'validation'
    else:
        return 'test'
```

---

## 4. Export Formats

### 4.1 Common Voice TSV Format

**File:** `exports/common_voice/train.tsv`
```tsv
client_id	path	sentence	up_votes	down_votes	age	gender	accent	locale	segment_duration
device_001	clips/common_voice_en_001.mp3	I went to the store to buy some milk.	1	0			en	5470
```

### 4.2 Hugging Face Dataset Format

**File:** `exports/huggingface/dataset.json`
```json
{
  "train": [
    {
      "audio": {
        "path": "audio/train/clip_001.wav",
        "array": null,
        "sampling_rate": 16000
      },
      "sentence": "I went to the store to buy some milk.",
      "duration": 5.47,
      "quality_score": 0.92
    }
  ]
}
```

### 4.3 Simple CSV Format

**File:** `exports/csv/train.csv`
```csv
audio_path,text,duration,quality,split
audio/clip_001.wav,"I went to the store to buy some milk.",5.47,0.92,train
audio/clip_002.wav,"The weather is nice today.",3.21,0.88,train
```

---

## 5. CLI Interface

### 5.1 Commands

```bash
# Initialize dataset manager
dataset-manager init [--config config.ini]

# Process pending segments from manifest
dataset-manager process [--batch-size 100] [--since "1 hour ago"]

# View statistics
dataset-manager stats [--detailed]

# Review segments needing attention
dataset-manager review [--quality-range 0.6-0.75]

# Export dataset
dataset-manager export --format [csv|huggingface|common_voice] \
                      --min-quality 0.8 \
                      --min-samples 100 \
                      --output ./my_dataset

# Clean up old/rejected segments
dataset-manager cleanup [--older-than 30d] [--rejected]

# Validate dataset quality
dataset-manager validate [--fix-issues]
```

### 5.2 Configuration File

**Location:** `~/.listnr/dataset_config.ini`

```ini
[General]
manifest_path = ~/.listnr/datasets/raw/manifest.jsonl
dataset_base_path = ~/.listnr/datasets
export_path = ~/.listnr/datasets/exports

[Processing]
batch_size = 100
auto_process = true
process_interval_seconds = 3600
min_segments_to_process = 10

[Quality]
min_duration_seconds = 1.0
max_duration_seconds = 30.0
min_quality_score = 0.6
auto_accept_threshold = 0.9
auto_reject_threshold = 0.5

[Splits]
train_ratio = 0.8
validation_ratio = 0.15
test_ratio = 0.05
stratify_by_quality = true
seed = 42

[Export]
formats = csv,huggingface
include_metadata = true
normalize_audio = true
target_sample_rate = 16000
convert_to_mono = true

[Cleanup]
retention_days = 90
max_storage_gb = 50
keep_rejected = false
archive_before_delete = true
```

---

## 6. Integration Points

### 6.1 Listnr Integration

Minimal changes required in listnr's `asr.py`:

```python
def process_speech_segment(self):
    # ... existing transcription code ...
    
    # NEW: Append metadata for dataset manager
    if cfg.get_bool_setting('Dataset', 'enable_collection'):
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'segment_id': f"seg_{self.session_id}_{self.segment_counter:03d}",
            'audio': {
                'filepath': clip_path,
                'duration_seconds': duration,
                'sample_rate': self.sample_rate,
                'channels': 1,
                'format': 'wav'
            },
            'transcription': {
                'raw': raw_text,
                'corrected': corrected_text,
                'model': self.whisper_model_name,
                'language': detected_language,
                'language_probability': language_prob
            },
            'quality': {
                'vad_confidence': vad_confidence,
                'whisper_confidence': whisper_confidence,
                'corrections_made': len(corrections) if corrections else 0
            },
            'session': {
                'session_id': self.session_id,
                'segment_index': self.segment_counter,
                'user_id': self.user_id
            }
        }
        
        # Append to manifest (fast, non-blocking)
        manifest_path = Path(cfg.get_setting('Dataset', 'manifest_path'))
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, 'a') as f:
            f.write(json.dumps(metadata) + '\n')
```

### 6.2 Systemd Service (Optional)

**File:** `/etc/systemd/user/listnr-dataset.service`

```ini
[Unit]
Description=Listnr Dataset Manager
After=network.target

[Service]
Type=oneshot
ExecStart=/usr/bin/python3 /usr/local/bin/dataset-manager process
StandardOutput=journal

[Install]
WantedBy=default.target
```

**Timer:** `/etc/systemd/user/listnr-dataset.timer`

```ini
[Unit]
Description=Process listnr datasets hourly

[Timer]
OnCalendar=hourly
Persistent=true

[Install]
WantedBy=timers.target
```

---

## 7. Success Metrics

### 7.1 Quality Metrics
- **Target:** 80% of segments score > 0.75 quality
- **Rejection Rate:** < 15% of segments
- **Review Required:** < 10% of segments

### 7.2 Performance Metrics
- **Processing Speed:** > 100 segments/second
- **Export Time:** < 1 minute for 1000 segments
- **Storage Efficiency:** < 2x original audio size

### 7.3 Fine-Tuning Readiness
- **Minimum Dataset:** 500 high-quality segments
- **Duration Target:** 1+ hours of training audio
- **Format Compatibility:** 100% compatible with target frameworks

---

## 8. Future Enhancements

### Phase 2 Features
- Web UI for dataset review and curation
- Active learning (prioritize uncertain segments for review)
- Speaker diarization and multi-speaker support
- Automatic noise reduction and audio enhancement
- Cross-validation of transcriptions

### Phase 3 Features
- Direct integration with fine-tuning pipelines
- Automatic quality improvement suggestions
- Dataset versioning and changelog
- Collaborative review system
- Export to cloud storage (S3, GCS)

---

## 9. References

- Parent PRD: `listnr-prd-v2.md`
- Mozilla Common Voice Format: https://commonvoice.mozilla.org/en/datasets
- Hugging Face Datasets: https://huggingface.co/docs/datasets
- Whisper Fine-tuning: https://github.com/openai/whisper/discussions/988