# Listenr — AMD ROCm fine-tuning image
#
# Base: official AMD ROCm PyTorch image (Python 3.12, ROCm-aware PyTorch pre-built).
# Pull:  podman pull rocm/pytorch:latest
# Build: podman build -t listenr-rocm .
# Run:   podman compose run --rm finetune
#
# NOTE: The rocm/pytorch base ships Python 3.12. listenr's pyproject.toml
#       requires Python >=3.13. We install with --ignore-requires-python
#       because the codebase uses no 3.13-only syntax — the constraint exists
#       to encourage modern local installs, not to block Docker usage.
#
# NOTE: sounddevice (microphone capture) will not work inside this container.
#       This image is intended for fine-tuning only (listenr-finetune /
#       listenr-build-dataset), not real-time audio capture.

FROM rocm/pytorch:latest

# ── system packages ──────────────────────────────────────────────────────────
# libsndfile1   : required by soundfile (audio I/O in finetune data pipeline)
# ffmpeg        : optional but useful for converting audio files
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ── project install ──────────────────────────────────────────────────────────
WORKDIR /app
COPY . /app

# Install core + finetune extras.
# --ignore-requires-python: base image has Python 3.10; constraint is 3.13+
#   but no 3.13-specific syntax is used in this codebase.
RUN pip install --no-cache-dir \
        --ignore-requires-python \
        -e ".[finetune]"

# ── runtime defaults ─────────────────────────────────────────────────────────
# Override HSA_OVERRIDE_GFX_VERSION if your GPU reports an unsupported gfx.
# Set HIP_VISIBLE_DEVICES=0 to restrict to a single GPU (avoids imbalance
# warnings and segfaults on multi-GPU systems).
ENV HSA_OVERRIDE_GFX_VERSION="" \
    HIP_VISIBLE_DEVICES="0"

CMD ["listenr-finetune", "--help"]
