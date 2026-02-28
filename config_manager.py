import configparser
import os
import sys
import logging

APP_NAME = 'listenr'  # Updated app name
CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".config", APP_NAME)
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.ini")

# Define defaults
DEFAULT_CONFIG = {
    'Whisper': {
        'model_size': 'small',  # base or small for speed, medium/large for accuracy
        'device': 'cpu',
        'compute_type': 'int8',  # int8 for CPU, float16 for GPU
        'beam_size': '1',  # Greedy search for streaming (1 = fastest)
        'best_of': '1',  # Single pass for speed
        'temperature': '0.0',
        'condition_on_previous_text': 'false',  # Disable for streaming independence
        'vad_filter': 'false',  # We handle VAD ourselves
    },
    'Audio': {
        'sample_rate': '16000',
        'channels': '1',
        'blocksize': '2048',  # Smaller chunks for lower latency
        'input_device': 'default',  # 'default' or device number
        'leading_silence_s': '0.2',  # Less pre-context for faster response
        'trailing_silence_s': '0.3',  # Less post-context for faster response
        'max_recording_duration_s': '10',  # Shorter max for streaming
    },
    'Storage': {
        'audio_clips_enabled': 'true',
        'audio_clips_path': '~/.listenr/audio_clips',
        'retention_days': '90',
        'max_storage_gb': '10',
        'clip_format': 'wav',
        'clip_quality': '16000',
    },
    'VAD': {
        'speech_threshold': '0.5',  # Balanced sensitivity
        'min_speech_duration_s': '0.5',  # Catch short phrases
        'max_silence_duration_s': '0.8',  # Quick cutoff for streaming feel
        'vad_chunk_size': '512',  # For 16kHz, Silero expects 512
        'patience_chunks': '5',  # Fewer chunks for faster response
    },
    'LLM': {
        'enabled': 'true',  # Enable LLM post-processing
        'model': 'Qwen3-0.6B-GGUF',  # Model name (must match Lemonade Server model)
        'api_base': 'http://localhost:8000/api/v1',  # Lemonade Server API endpoint (see https://lemonade-server.ai)
        # Lemonade: http://localhost:8000/api/v1
        # See https://lemonade-server.ai for docs and model management
        'temperature': '0.3',  # Low temperature for consistency
        'context_window': '5',  # Number of previous transcriptions to use as context
        'max_tokens': '150',  # Maximum tokens to generate
        'timeout': '10',  # API timeout in seconds
        'correction_types': 'punctuation,capitalization,grammar,homophone,numeric,spacing',
        'correction_threshold': '0.7',
        'fallback_processing': 'true',
    },
    'Output': {
        'file': '~/transcripts_raw.txt',  # Empty means console only
        'llm_file': '~/transcripts_clean.txt',  # Separate file for LLM-processed output
        'format': '[{timestamp}] {text}',  # Output format
        'timestamp_format': '%%Y-%%m-%%d %%H:%%M:%%S',  # Double %% for escaping
        'show_raw': 'false',  # Show raw transcriptions when LLM is enabled
    },
    'Logging': {
        'level': 'INFO',  # DEBUG, INFO, WARNING, ERROR
        'file': '',  # Empty means console only
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
            # Write the fully populated config object
            with open(CONFIG_FILE, 'w') as configfile:
                # Write header comment
                configfile.write(f"# {APP_NAME} Configuration File\n")
                configfile.write("# Edit this file to customize ASR settings\n")
                configfile.write("# Lemonade Server: https://lemonade-server.ai\n\n")
                
                for section in config.sections():
                    configfile.write(f"[{section}]\n")
                    for key, value in config.items(section):
                        # Special handling for timestamp_format - write with single %
                        if key == 'timestamp_format' and '%%' in value:
                            value = value.replace('%%', '%')
                        configfile.write(f"{key} = {value}\n")
                    configfile.write("\n")
                    
            print("Default config file created.")
        except OSError as e:
            print(f"ERROR: Could not create default config: {e}", file=sys.stderr)
            return config

    try:
        # Read user's file, overriding defaults where specified
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
        get_int_setting('Whisper', 'beam_size')
        get_float_setting('VAD', 'speech_threshold')
        get_float_setting('VAD', 'min_speech_duration_s')
        get_float_setting('VAD', 'max_silence_duration_s')
        
        # Check VAD requirements
        sr = get_int_setting('Audio', 'sample_rate')
        if sr not in [8000, 16000]:
            logging.warning(f"VAD typically requires sample rate 8000 or 16000, configured: {sr}")
        
        vad_chunk = get_int_setting('VAD', 'vad_chunk_size')
        expected_for_16k = 512
        if sr == 16000 and vad_chunk != expected_for_16k:
            logging.warning(f"For 16kHz, VAD chunk size should be {expected_for_16k}, configured: {vad_chunk}")

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
    
    # Handle timestamp format specifically - convert %% to % when reading
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

def save_config():
    """Save current config to file"""
    try:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        with open(CONFIG_FILE, 'w') as configfile:
            configfile.write(f"# {APP_NAME} Configuration File\n")
            configfile.write("# Edit this file to customize ASR settings\n\n")
            
            for section in config.sections():
                configfile.write(f"[{section}]\n")
                for key, value in config.items(section):
                    # Special handling for timestamp_format
                    if key == 'timestamp_format' and '%' in value and not '%%' in value:
                        value = value.replace('%', '%%')
                    configfile.write(f"{key} = {value}\n")
                configfile.write("\n")
    except Exception as e:
        print(f"Error saving config: {e}")

def update_setting(section, key, value):
    """Update a setting in the config"""
    if not config.has_section(section):
        config.add_section(section)
    config.set(section, key, str(value))

# Load config when the module is imported
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
config = load_config()
