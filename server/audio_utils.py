import subprocess
import os
import soundfile as sf
import numpy as np

def convert_to_wav(src_path, dst_path):
    """Convert arbitrary audio file to 16kHz mono WAV using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", src_path,
        "-ac", "1", "-ar", "16000",
        "-f", "wav", dst_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def normalize_audio(filepath):
    """Ensure audio data is valid float32 mono."""
    data, samplerate = sf.read(filepath)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    sf.write(filepath, data, samplerate)
