# Local ASR Listener (using Whisper)

This project provides a command-line ASR (Automatic Speech Recognition) service for Linux. It records audio from your microphone, detects speech segments using VAD, and transcribes them locally using the Whisper model (`faster-whisper`).

**Features:**

- **Local Transcription:** Uses `faster-whisper` for efficient local ASR (CPU or NVIDIA GPU). No cloud or OpenAI dependencies.
- **VAD Streaming:** Uses Silero VAD to automatically detect speech and pauses, so you don't need to manually start/stop recording for each phrase.
- **CLI Only:** No desktop UI, notifications, or hotkeys. All output is printed to the terminal or saved to a file.
- **Minimal Dependencies:** Only core ASR, VAD, and audio libraries required.
- **Configurable:** Settings managed via `~/.config/asr-indicator/config.ini`.
- **Robust Post-Processing:** See `processing_examples.md` for the correction patterns applied to Whisper output (punctuation, capitalization, homophones, and more).

## Requirements

**1. System Dependencies:**

- **Python:** Python 3.9 or higher (with `venv`).
- **Core Audio:** `ffmpeg`, `libsndfile1`, `pulseaudio-utils` (for audio recording and playback).
  ```bash
  sudo apt update
  sudo apt install ffmpeg libsndfile1 pulseaudio-utils python3-venv
  ```
- **Build Tools:** Needed for some Python packages (if wheels aren't available).
  ```bash
  sudo apt install build-essential cmake python3-dev
  ```

**2. Python Dependencies:**

- All required Python packages are listed in `requirements.txt`.
- PyTorch is required for Whisper and VAD. Install the correct version for your hardware (CPU or CUDA) before installing other packages.

**3. NVIDIA GPU (Optional but Recommended):**

- For significantly faster transcription (`device = cuda` in config).
- Requires:
  - NVIDIA Drivers installed.
  - Matching CUDA Toolkit installed system-wide.
  - cuDNN library installed system-wide.
  - PyTorch installed with the correct CUDA version (see step 2).
- Setup can be complex; follow official NVIDIA documentation for your Linux distribution.

**4. Python Dependencies:**

- Listed in `requirements.txt`.

## Setup Instructions

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/<owner>/listenr
   cd listenr
   ```

2. **Install System Dependencies:**
   ```bash
   sudo apt update
   sudo apt install ffmpeg libsndfile1 pulseaudio-utils python3-venv build-essential cmake python3-dev
   ```

3. **Create & Activate Python Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate.fish   # For fish shell
   # or
   source venv/bin/activate        # For bash/zsh
   ```

4. **Install Python Dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **Run the Service (CLI):**
   ```bash
   python main.py
   ```

   The service will listen for audio and print transcriptions to the terminal or append them to the configured output file (see config).
   Stop the service with Ctrl+C.

## Usage

1. Start the service: `python main.py`
2. Speak into your microphone. The service will automatically detect speech and transcribe segments.
3. Transcriptions are printed to the terminal or saved to a file (see `[Output]` section in config).
4. Stop the service with Ctrl+C.

## Troubleshooting

- **Dependencies:** Make sure all system and Python dependencies are installed in your venv. Use `pip list` to check.
- **Audio Device:** If no audio is recorded, check the `input_device` setting in `config.ini`. Use `python -m sounddevice` to list available devices.
- **VAD Tuning:** If transcription cuts off too early or waits too long, adjust `silence_duration_ms` and `speech_threshold` in `config.ini`.
- **Whisper Model:** Ensure the `model_size` exists and you have enough RAM/VRAM. Check `device` and `compute_type` settings for your hardware.


## License

This project is licensed under the **Mozilla Public License Version 2.0**. See the `LICENSE` file for details.
