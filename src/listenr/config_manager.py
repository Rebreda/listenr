import configparser
import os
import sys
import logging

APP_NAME = 'listenr'
CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".config", APP_NAME)
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.ini")

# Define defaults
DEFAULT_CONFIG = {
    'Lemonade': {
        # HTTP API base for Lemonade Server (LLM, ASR batch, health)
        # The /realtime WebSocket port is discovered dynamically via GET /api/v1/health -> websocket_port
        'api_base': 'http://localhost:8000/api/v1',
    },
    'Whisper': {
        # Whisper model name served by Lemonade (whisper.cpp backend).
        # Available: Whisper-Tiny, Whisper-Base, Whisper-Large-v3-Turbo
        'model': 'Whisper-Base',
    },
    'Audio': {
        # Mic capture rate — must match the device's native rate.
        # cli.py resamples to 16kHz internally before sending to Lemonade /realtime.
        'sample_rate': '48000',
        'channels': '1',
        # Chunk size in frames per mic read (~85ms worth of audio).
        'blocksize': '4096',
        'input_device': 'pipewire',  # 'pipewire', device name, index, or 'default'
    },
    'Storage': {
        'audio_clips_enabled': 'true',
        'audio_clips_path': '~/.listenr/audio_clips',
        'retention_days': '90',
        'max_storage_gb': '10',
        'clip_format': 'wav',
    },
    'VAD': {
        # Server-side VAD settings sent via session.update on the /realtime WebSocket.
        # All VAD processing happens in Lemonade; these are passed through as-is.
        # threshold: RMS energy threshold for speech detection (raise to ignore background noise)
        'threshold': '0.05',
        # silence_duration_ms: ms of silence to trigger speech end and transcription
        'silence_duration_ms': '800',
        # prefix_padding_ms: minimum speech duration (ms) before triggering transcription
        'prefix_padding_ms': '250',
    },
    'LLM': {
        'enabled': 'true',  # Enable LLM post-processing of transcriptions
        'model': 'gpt-oss-20b-mxfp4-GGUF',  # LLM model name (must be loaded in Lemonade)
        'api_base': 'http://localhost:8000/api/v1',  # Lemonade Server API base
        'temperature': '0.3',
        'max_tokens': '1500',
        'timeout': '30',
        'context_window': '10',  # Number of preceding segments passed as context to the LLM
    },
    'Dataset': {
        'output_path': '~/listenr_dataset',   # Where build_dataset writes CSV/HF output
        'split': '80/10/10',                  # Train/dev/test split percentages
        'min_duration': '0.3',                # Minimum clip duration in seconds
        'min_chars': '2',                     # Minimum non-whitespace chars in transcription
        'seed': '42',                         # Random seed for reproducible splits
        'format': 'csv',                      # Output format: csv, hf, or both
    },
    'Finetune': {
        'base_model': 'openai/whisper-small',  # HuggingFace model id to fine-tune
        'language': 'english',                 # Target language for the processor/tokenizer
        'task': 'transcribe',                  # 'transcribe' or 'translate'
        'lora_r': '8',                         # LoRA rank
        'lora_alpha': '32',                    # LoRA scaling factor
        'lora_dropout': '0.1',                 # LoRA dropout
        'lora_target_modules': 'q_proj,v_proj', # Comma-separated decoder attention projections
        'freeze_encoder': 'true',              # Freeze Whisper encoder weights during training
        'learning_rate': '1e-4',               # AdamW learning rate
        'warmup_steps': '100',
        'max_steps': '2000',
        'batch_size': '8',                     # Per-device train batch size
        'grad_accum_steps': '2',               # Gradient accumulation steps
        'fp16': 'false',                       # Mixed precision fp16 (CUDA; use bf16 for AMD ROCm)
        'bf16': 'false',                       # Mixed precision bf16 (recommended for AMD ROCm RDNA2+)
        'output_dir': '~/listenr_finetune',    # Where adapter checkpoints are saved
        'eval_steps': '200',                   # Evaluate every N training steps
        'save_steps': '400',                   # Save checkpoint every N training steps
        'generation_max_length': '128',        # Max decode length during evaluation
    },
    'Output': {
        'file': '~/transcripts_raw.txt',
        'llm_file': '~/transcripts_clean.txt',
        'line_format': '[{timestamp}] {text}',
        'timestamp_format': '%%Y-%%m-%%d %%H:%%M:%%S',  # Double %% for configparser escaping
        'show_raw': 'false',
    },
    'Logging': {
        'level': 'INFO',  # DEBUG, INFO, WARNING, ERROR
        'file': '',  # Empty means console only
    },
    'Corrections': {
        # Keyword corrections passed to the LLM to fix common STT misrecognitions.
        # Format: incorrect_word = Correct Word  (keys are case-insensitive)
        'clod': 'Claude Code',
        'clode': 'Claude Code',
        'cloud code': 'Claude Code',
        'clock code': 'Claude Code',
        'open ai': 'OpenAI',
        'unsurropic': 'Anthropic',
        'anthropic': 'Anthropic',
    }
}


# Initialize ConfigParser instance with interpolation disabled
config = configparser.ConfigParser(
    inline_comment_prefixes=('#', ';'),
    interpolation=None  # Disable interpolation to treat % signs literally
)

def load_config():
    """Loads config, creates default if needed, returns config object."""
    global config

    # Reset parser and read defaults first
    config = configparser.ConfigParser(
        inline_comment_prefixes=('#', ';'),
        interpolation=None
    )

    # Load defaults using read_dict
    config.read_dict(DEFAULT_CONFIG)

    if not os.path.exists(CONFIG_FILE):
        print(f"Config file not found. Creating default at: {CONFIG_FILE}")
        try:
            os.makedirs(CONFIG_DIR, exist_ok=True)
            with open(CONFIG_FILE, 'w') as configfile:
                configfile.write(f"# {APP_NAME} Configuration File\n")
                configfile.write("# Edit this file to customize ASR settings\n")
                configfile.write("# Lemonade Server: https://lemonade-server.ai\n\n")

                for section in config.sections():
                    configfile.write(f"[{section}]\n")
                    for key, value in config.items(section):
                        if key == 'timestamp_format' and '%%' in value:
                            value = value.replace('%%', '%')
                        configfile.write(f"{key} = {value}\n")
                    configfile.write("\n")

            print("Default config file created.")
        except OSError as e:
            print(f"ERROR: Could not create default config: {e}", file=sys.stderr)
            return config

    try:
        loaded_files = config.read(CONFIG_FILE)
        if loaded_files:
            print(f"Loaded config from {CONFIG_FILE}")
        else:
            print(f"Warning: Could not read config file {CONFIG_FILE}. Using defaults.")
            config = configparser.ConfigParser(inline_comment_prefixes=('#', ';'), interpolation=None)
            config.read_dict(DEFAULT_CONFIG)

    except configparser.Error as e:
        print(f"ERROR reading config file {CONFIG_FILE}: {e}", file=sys.stderr)
        print("Using internal defaults.", file=sys.stderr)
        config = configparser.ConfigParser(inline_comment_prefixes=('#', ';'), interpolation=None)
        config.read_dict(DEFAULT_CONFIG)

    # Perform basic validation
    try:
        get_int_setting('Audio', 'sample_rate')
        get_int_setting('Audio', 'channels')
        get_int_setting('Audio', 'blocksize')
        get_float_setting('VAD', 'threshold')
        get_int_setting('VAD', 'silence_duration_ms')
        get_int_setting('VAD', 'prefix_padding_ms')

    except ValueError as e:
        print(f"ERROR: Config file has invalid number format: {e}", file=sys.stderr)
        print("Please check numeric settings.", file=sys.stderr)
        sys.exit(1)

    return config

# Helper functions for getting settings
def get_setting(section, key, fallback_value=None):
    """Gets a string setting, falling back to default dict then provided fallback."""
    default_from_dict = DEFAULT_CONFIG.get(section, {}).get(key)
    final_fallback = fallback_value if default_from_dict is None else default_from_dict
    value = config.get(section, key, fallback=final_fallback)

    if key == 'timestamp_format' and value and '%%' in value:
        value = value.replace('%%', '%')

    return value

def get_int_setting(section, key, fallback_value=0):
    """Gets an int setting, falling back safely."""
    default_from_dict_str = DEFAULT_CONFIG.get(section, {}).get(key)
    try:
        final_fallback = int(default_from_dict_str) if default_from_dict_str is not None else fallback_value
    except (ValueError, TypeError):
        final_fallback = fallback_value

    try:
        return config.getint(section, key, fallback=final_fallback)
    except (ValueError, TypeError):
        print(f"Warning: Invalid integer for [{section}]{key}. Using fallback {final_fallback}.", file=sys.stderr)
        return final_fallback

def get_float_setting(section, key, fallback_value=0.0):
    """Gets a float setting, falling back safely."""
    default_from_dict_str = DEFAULT_CONFIG.get(section, {}).get(key)
    try:
        final_fallback = float(default_from_dict_str) if default_from_dict_str is not None else fallback_value
    except (ValueError, TypeError):
        final_fallback = fallback_value

    try:
        return config.getfloat(section, key, fallback=final_fallback)
    except (ValueError, TypeError):
        print(f"Warning: Invalid float for [{section}]{key}. Using fallback {final_fallback}.", file=sys.stderr)
        return final_fallback

def get_bool_setting(section, key, fallback_value=False):
    """Gets a boolean setting, falling back safely."""
    default_from_dict_str = DEFAULT_CONFIG.get(section, {}).get(key)
    try:
        final_fallback = default_from_dict_str.lower() == 'true' if default_from_dict_str is not None else fallback_value
    except AttributeError:
        final_fallback = bool(default_from_dict_str) if default_from_dict_str is not None else fallback_value

    try:
        return config.getboolean(section, key, fallback=final_fallback)
    except (ValueError, TypeError):
        print(f"Warning: Invalid boolean for [{section}]{key}. Using fallback {final_fallback}.", file=sys.stderr)
        return final_fallback

def get_corrections() -> dict[str, str]:
    """Return keyword corrections dict {incorrect: correct} from [Corrections] section."""
    if not config.has_section('Corrections'):
        return {}
    return dict(config.items('Corrections'))

def save_config():
    """Save current config to file."""
    try:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        with open(CONFIG_FILE, 'w') as configfile:
            configfile.write(f"# {APP_NAME} Configuration File\n")
            configfile.write("# Edit this file to customize ASR settings\n\n")

            for section in config.sections():
                configfile.write(f"[{section}]\n")
                for key, value in config.items(section):
                    if key == 'timestamp_format' and '%' in value and '%%' not in value:
                        value = value.replace('%', '%%')
                    configfile.write(f"{key} = {value}\n")
                configfile.write("\n")
    except Exception as e:
        print(f"Error saving config: {e}")

def update_setting(section, key, value):
    """Update a setting in the config."""
    if not config.has_section(section):
        config.add_section(section)
    config.set(section, key, str(value))

# Load config when the module is imported
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
config = load_config()
