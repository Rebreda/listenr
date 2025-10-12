import logging
import config_manager as cfg
logger = logging.getLogger(getattr(cfg, 'APP_NAME', 'listenr'))
import subprocess

def run_command(cmd_list):
    """Runs a command, returns stdout, handles errors."""
    try:
        # Use capture_output=True to prevent command output unless error
        result = subprocess.run(cmd_list, capture_output=True, text=True, check=True, timeout=5)
        return result.stdout.strip()
    except FileNotFoundError:
        # Log error only if it's not the initial notify-send check failing
        print(f"Command not found: {cmd_list[0]}")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {' '.join(cmd_list)} -> {e.stderr}")
        return None
    except subprocess.TimeoutExpired:
        print(f"Command timed out: {' '.join(cmd_list)}")
        return None
    except Exception as e:
        print(f"Unexpected error running {' '.join(cmd_list)}: {e}")
        return None